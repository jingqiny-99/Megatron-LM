# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-call pipeline roles (``HybridStackCallRoles``) and the ring's call context.

* Reader census (AST): the forward path of ``HybridModel`` / ``HybridStack`` (and the loop
  stacks) reads the per-call accessors (``call_pre_process`` / ``call_post_process`` /
  ``call_mtp_process``), never the built flags; the built flags are read only by construction,
  the accessors themselves and the bookkeeping of which modules exist (checkpoint keys, the
  embedding / output-weight ties, offload marking). A new raw read anywhere else fails.
* A PP = 1 ``HybridModel`` (plain and loop decoder) with roles set skips the embedding, the
  head and MTP as told, and runs exactly the built forward with all roles on; built flags are
  unchanged after calls and after an exception inside a call.
* The ring chunk's tripwires: a call without the ring's ``LoopRingCall`` (``RING_EAGER_CALL``)
  and roles that do not fit the chunk and round (``RING_CALL_ROLES``).
* ``LoopRingRoundView``: no parameters and no ``vp_stage`` of its own, everything else through the
  ``.module`` walk, its roles per round, ``fp32_output`` on the final view only, the microbatch
  check, the call context cleared on errors. ``ring_sample_counts``.
"""

import ast
import contextlib
import dataclasses
import inspect

import pytest
import torch
from torch import nn

from megatron.core.models.common.language_module import language_module
from megatron.core.models.hybrid import hybrid_block, hybrid_model
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackCallRoles
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.loop import attn_res_stack, guards, hybrid_stack, ring_pp
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.loop.ring_pp import (
    LoopRingCall,
    LoopRingRoundView,
    loop_ring_call,
    ring_sample_counts,
)
from megatron.core.utils import get_attr_wrapped_model, get_model_config, get_model_type
from tests.unit_tests.transformer.loop import hybrid_loop_utils as hlu
from tests.unit_tests.transformer.loop import ring_emulation as emu

mhc_on_this_device = hlu.mhc_on_this_device
cpu_mtp = hlu.cpu_mtp

_RING_PREFIX = "Loop ring does not support: "

# ---------------------------------------------------------------------------------------------
# Reader census
# ---------------------------------------------------------------------------------------------

FLAGS = frozenset(("pre_process", "post_process", "mtp_process"))
_CENSUS_MODULES = (hybrid_model, hybrid_block, hybrid_stack, attn_res_stack, language_module)
# The methods that read the built flags: construction, the accessors, and the bookkeeping of
# which modules exist on the stage (offload marking, the embedding / output-weight ties and
# their group, checkpoint keys). Never per-call execution.
ALLOWED_RAW_READERS = {
    ("HybridModel", "__init__"),
    ("HybridModel", "call_pre_process"),
    ("HybridModel", "call_post_process"),
    ("HybridModel", "call_mtp_process"),
    ("HybridModel", "preprocess_for_fine_grained_offloading"),
    ("HybridStack", "__init__"),
    ("HybridStack", "call_pre_process"),
    ("HybridStack", "call_post_process"),
    ("LanguageModule", "_is_in_embd_group"),
    ("LanguageModule", "setup_embeddings_and_output_layer"),
    ("LanguageModule", "shared_embedding_or_output_weight"),
    ("LanguageModule", "sharded_state_dict"),
    ("LanguageModule", "tie_embeddings_and_output_weights_state_dict"),
}
# Forward-path methods and the accessors each must read.
FORWARD_PATH = {
    ("HybridModel", "forward"): set(),
    ("HybridModel", "embed"): {"call_pre_process"},
    ("HybridModel", "rope_ctx"): set(),
    ("HybridModel", "head_loss"): {"call_post_process", "call_mtp_process"},
    ("HybridStack", "forward"): set(),
    ("HybridStack", "_prepare_forward"): {"call_pre_process"},
    ("HybridStack", "_layer_loop"): set(),
    ("HybridStack", "_run_layer"): set(),
    ("HybridStack", "_finalize_forward"): {"call_pre_process", "call_post_process"},
}


def _flag_reads(function):
    """The built flags ``function`` reads from ``self`` (``self.<flag>`` or ``getattr(self,
    '<flag>', ...)``) and the accessors it reads (``self.call_<flag>``)."""
    raw, accessors = set(), set()
    for node in ast.walk(function):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                if node.attr in FLAGS:
                    raw.add(node.attr)
                elif node.attr.startswith("call_") and node.attr[5:] in FLAGS:
                    accessors.add(node.attr)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "self"
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in FLAGS
        ):
            raw.add(node.args[1].value)
    return raw, accessors


def _census():
    """``{(class, method): (raw reads, accessor reads)}`` over the census modules."""
    census = {}
    for module in _CENSUS_MODULES:
        tree = ast.parse(inspect.getsource(module))
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            for function in cls.body:
                if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    census[(cls.name, function.name)] = _flag_reads(function)
    return census


def test_reader_census():
    census = _census()
    raw_readers = {key for key, (raw, _) in census.items() if raw}
    assert raw_readers == ALLOWED_RAW_READERS, (
        f"new raw readers {sorted(raw_readers - ALLOWED_RAW_READERS)}, "
        f"gone {sorted(ALLOWED_RAW_READERS - raw_readers)}"
    )
    for key, accessors in FORWARD_PATH.items():
        raw, read = census[key]
        assert not raw, key
        assert accessors <= read, (key, read)
    # The loop stacks' forward paths read no flag at all (the ring reads its LoopRingCall).
    for (cls, method), (raw, _) in census.items():
        if cls in ("LoopHybridStack", "LoopAttnResHybridStack"):
            assert not raw, (cls, method)


def test_roles_only_switch_built_processing_off():
    stack = HybridStack.__new__(HybridStack)
    nn.Module.__init__(stack)
    for pre in (False, True):
        for post in (False, True):
            stack.pre_process, stack.post_process = pre, post
            stack.call_roles = None
            assert (stack.call_pre_process, stack.call_post_process) == (pre, post)
            for role in (False, True):
                stack.call_roles = HybridStackCallRoles(role, role, role)
                assert stack.call_pre_process == (pre and role)
                assert stack.call_post_process == (post and role)


# ---------------------------------------------------------------------------------------------
# A PP = 1 model with roles set
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _model_parallel():
    with hlu.model_parallel_once():
        yield


def _pp1_model(decoder):
    """A PP = 1 local-layer HybridModel with mHC 4 and one MTP depth: a plain HybridStack or a
    loop stack."""
    if decoder == "loop":
        return hlu.build_local_loop_model(
            "*-*-*-/*-", prelude=2, coda=2, loop_iterations=2, **emu.MHC
        )
    config = hlu.dense_config(
        4, is_hybrid_model=True, mtp_num_layers=1, mtp_loss_scaling_factor=0.1, **emu.MHC
    )
    torch.manual_seed(1234)
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hlu.local_model_spec(HybridStack, hyper_connections=True),
        vocab_size=hlu.VOCAB,
        max_sequence_length=hlu.SEQ,
        hybrid_layer_pattern="*-*-/*-",
        position_embedding_type="none",
    )
    return model.to(hlu.device())


class _Calls:
    """Counts the forward calls of named modules."""

    def __init__(self, **modules):
        self.counts = dict.fromkeys(modules, 0)
        self.outputs = {}
        for name, module in modules.items():
            module.register_forward_hook(self._hook(name))

    def _hook(self, name):
        def hook(module, args, output):
            del module, args
            self.counts[name] += 1
            self.outputs[name] = output

        return hook

    def reset(self):
        """Zero the counts."""
        self.counts = dict.fromkeys(self.counts, 0)


def _flags(model):
    return (
        model.pre_process,
        model.post_process,
        model.mtp_process,
        model.decoder.pre_process,
        model.decoder.post_process,
    )


@pytest.mark.parametrize("decoder", ["plain", "loop"])
def test_pp1_model_with_roles_skips_embedding_head_and_mtp(
    _model_parallel, mhc_on_this_device, cpu_mtp, decoder
):
    del mhc_on_this_device, cpu_mtp
    model = _pp1_model(decoder)
    built = _flags(model)
    calls = _Calls(
        embedding=model.embedding,
        decoder=model.decoder,
        mtp=model.mtp,
        output_layer=model.output_layer,
    )
    batch = emu.make_batches(1)[0]
    tokens, positions, labels = batch["tokens"], batch["position_ids"], batch["labels"]

    def run(roles, input_tensor=None):
        calls.reset()
        model.set_input_tensor(input_tensor)
        with loop_ring_call(model.decoder, roles) if roles else contextlib.nullcontext():
            output = model(tokens, positions, None, labels=labels)
        assert model.decoder.call_roles is None and _flags(model) == built
        return output, dict(calls.counts)

    full, counts = run(None)
    assert counts == dict(embedding=2, decoder=1, mtp=1, output_layer=2)  # main + MTP
    streams = calls.outputs["decoder"][1]  # the pre-contraction mHC streams
    on = HybridStackCallRoles(True, True, True)
    same, counts = run(on)
    assert torch.equal(same, full) and counts == dict(embedding=2, decoder=1, mtp=1, output_layer=2)
    # No embedding: the decoder reads the (stream-expanded) input tensor.
    with torch.no_grad():
        embedded = HyperConnectionModule.input_expand(
            model.embedding(input_ids=tokens, position_ids=positions), 4
        )
    output, counts = run(HybridStackCallRoles(False, True, True), embedded)
    assert torch.equal(output, full)
    assert counts == dict(embedding=1, decoder=1, mtp=1, output_layer=2)  # MTP's lookup only
    # No head and no MTP: the decoder's streams come back, no contraction.
    output, counts = run(HybridStackCallRoles(True, False, False))
    assert torch.equal(output, streams) and output.shape[-1] == 4 * hlu.HIDDEN
    assert counts == dict(embedding=1, decoder=1, mtp=0, output_layer=0)
    # The head without MTP: the same LM loss, MTP skipped.
    output, counts = run(HybridStackCallRoles(True, True, False))
    assert torch.equal(output, full)
    assert counts == dict(embedding=1, decoder=1, mtp=0, output_layer=1)


def test_call_roles_cleared_after_an_exception(_model_parallel, mhc_on_this_device, cpu_mtp):
    del mhc_on_this_device, cpu_mtp
    model = _pp1_model("plain")
    built = _flags(model)
    with pytest.raises(RuntimeError, match="inside the call"):
        with loop_ring_call(model.decoder, HybridStackCallRoles(False, False, False)):
            assert not model.call_pre_process and not model.call_post_process
            assert not model.call_mtp_process
            raise RuntimeError("inside the call")
    assert model.decoder.call_roles is None and _flags(model) == built
    assert model.call_pre_process and model.call_post_process and model.call_mtp_process
    # A call inside another call of the same stack is refused; the outer roles stay.
    outer = HybridStackCallRoles(True, False, False)
    with loop_ring_call(model.decoder, outer):
        with pytest.raises(RuntimeError, match="inside another call"):
            with loop_ring_call(model.decoder, HybridStackCallRoles(True, True, True)):
                pass
        assert model.decoder.call_roles is outer
    assert model.decoder.call_roles is None


# ---------------------------------------------------------------------------------------------
# Ring chunks: tripwires and round views
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def ring(_model_parallel, mhc_on_this_device, cpu_mtp, monkeypatch):
    """A dense ring at P = 2, k = 2 (all views 'C') built from a PP = 1 model."""
    del mhc_on_this_device, cpu_mtp
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    _, state, _ = emu.run_reference("dense", emu.make_batches(1))
    return emu.build_ring("dense", 2, 2, "all-C", state)


def _hidden(width):
    return torch.randn(hlu.SEQ, hlu.BATCH, width, device=hlu.device())


def _reported(excinfo):
    message = str(excinfo.value)
    assert message.startswith(_RING_PREFIX), message
    return message[len(_RING_PREFIX) :].split(", ")


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU emulation of the ring")
def test_ring_chunk_tripwires(ring):
    first = ring.chunk_model(0, 0).decoder  # round start
    last = ring.chunk_model(1, 1).decoder  # round end
    middle = ring.chunk_model(1, 0).decoder
    streams = 4 * hlu.HIDDEN
    # Without the ring's roles (none, or generic ones) a ring chunk refuses to run.
    for roles in (None, HybridStackCallRoles(False, False, False)):
        middle.set_input_tensor(_hidden(streams))
        with pytest.raises(ValueError) as excinfo:
            if roles is None:
                middle(None, attention_mask=None)
            else:
                with loop_ring_call(middle, roles):
                    middle(None, attention_mask=None)
        assert _reported(excinfo) == [guards.RING_EAGER_CALL]
        assert middle.call_roles is None

    def call(stack, hidden, **fields):
        roles = dict(pre_process=False, post_process=False, mtp_process=False)
        roles.update(fields)
        stack.set_input_tensor(hidden)
        with loop_ring_call(stack, LoopRingCall(**roles)):
            return stack(hidden, attention_mask=None)

    bad = [
        (first, _hidden(hlu.HIDDEN), dict(pre_process=True, round=1)),  # embedding in round 2
        (first, _hidden(streams), dict(round=0)),  # round 1 without the embedding
        (last, _hidden(streams), dict(post_process=True, round=0)),  # head before round R
        (last, _hidden(streams), dict(round=3)),  # round R without the head
        (middle, _hidden(streams), dict(mtp_process=True, round=3)),  # MTP off the round end
        (middle, _hidden(streams), dict(round=4)),  # no round 5
    ]
    for stack, hidden, fields in bad:
        with pytest.raises(ValueError) as excinfo:
            call(stack, hidden, **fields)
        assert _reported(excinfo) == [guards.RING_CALL_ROLES], fields
        assert stack.call_roles is None
    # A later round without its round-1 call has no injection.
    with pytest.raises(RuntimeError, match="has no injection"):
        call(first, _hidden(streams), round=2, microbatch=5)
    assert first.ring_injection_keys() == ()


class _Recorder(nn.Module):
    """A wrapper that records the ``fp32_output`` it gets and does not run the model."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.seen = []

    def forward(self, *args, fp32_output=None, **kwargs):
        del args, kwargs
        self.seen.append(fp32_output)
        return None


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU emulation of the ring")
def test_round_views(ring, monkeypatch):
    stages, chunks, views = ring.stages, ring.chunks, emu.ROUNDS * ring.chunks
    for stage in range(stages):
        for view_index, view in enumerate(ring.views[stage]):
            chunk = view_index % chunks
            model = ring.chunk_model(stage, chunk)
            assert list(view.parameters()) == [] and list(view.children()) == []
            assert getattr(view, "vp_stage", None) is None
            assert get_attr_wrapped_model(view, "vp_stage") == chunk == model.vp_stage
            assert get_attr_wrapped_model(view, "decoder") is model.decoder
            assert get_attr_wrapped_model(view, "set_input_tensor") == model.set_input_tensor
            assert get_attr_wrapped_model(view, "pg_collection") is model.pg_collection
            assert get_model_type(view) == model.model_type
            assert get_model_config(view) is ring.step_config
            assert get_model_config(view).virtual_pipeline_model_parallel_size == views
            assert not hasattr(view, "set_is_first_microbatch")
            assert view.view_index == view_index and view.views == views
            round_index = view_index // chunks
            roles = view.roles_for(3)
            first = stage == 0 and chunk == 0
            last = stage == stages - 1 and chunk == chunks - 1
            assert roles == LoopRingCall(
                pre_process=first and round_index == 0,
                post_process=last and round_index == emu.ROUNDS - 1,
                mtp_process=last and round_index == emu.ROUNDS - 1,
                round=round_index,
                microbatch=3,
                checkpoint=True,
            )
    # Through a Float16Module only the final view asks for fp32 outputs.
    monkeypatch.setattr(ring_pp, "_wraps_float16_module", lambda module: True)
    model = ring.chunk_model(stages - 1, chunks - 1)
    recorder = _Recorder(model)
    wrapped = [
        LoopRingRoundView(
            recorder,
            ring.step_config,
            round_index=view_index // chunks,
            chunk_index=chunks - 1,
            chunks=chunks,
            checkpoint=False,
        )
        for view_index in range(chunks - 1, views, chunks)
    ]
    model.decoder.current_microbatch = 0
    for view in wrapped:
        assert view(None) is None and view.calls == 1
        assert model.decoder.call_roles is None
    assert recorder.seen == [False] * (emu.ROUNDS - 1) + [True]
    # The i-th call of a step must run microbatch i.
    with pytest.raises(RuntimeError, match="got microbatch 0"):
        wrapped[0](None)
    assert wrapped[0].calls == 1 and model.decoder.call_roles is None
    wrapped[0].reset_calls()
    assert wrapped[0](None) is None
    del model.decoder.current_microbatch


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU emulation of the ring")
def test_round_view_call_cleared_after_an_error(ring):
    """A view call that raises inside the model (round 3 of a microbatch whose round 1 never ran:
    no injection) leaves no roles set, the built flags unchanged and its call count unchanged."""
    view = ring.views[0][2 * ring.chunks]  # round 3 on the round-start chunk
    model = ring.chunk_model(0, 0)
    built = emu.built_flags(model)
    model.set_input_tensor(_hidden(4 * hlu.HIDDEN))
    batch = emu.make_batches(1)[0]
    with pytest.raises(RuntimeError, match="has no injection"):
        view(batch["tokens"], batch["position_ids"], None)
    assert model.decoder.call_roles is None and emu.built_flags(model) == built
    assert view.calls == 0 and model.decoder.ring_injection_keys() == ()


def test_round_view_needs_a_ring_chunk(_model_parallel, mhc_on_this_device, cpu_mtp):
    del mhc_on_this_device, cpu_mtp
    model = _pp1_model("loop")
    assert isinstance(model.decoder, LoopHybridStack) and model.decoder.loop_ring is None
    with pytest.raises(ValueError, match="needs a ring chunk"):
        LoopRingRoundView(
            model, model.config, round_index=0, chunk_index=0, chunks=1, checkpoint=False
        )


def test_ring_sample_counts():
    counts = [
        {0: [4, 1, 0, 0], 1: [4, 1, 0, 0]},
        {0: [4, 0, 0, 0], 1: [3, 0, 0, 0]},
        {0: [4, 0, 1, 1]},
    ]
    assert ring_sample_counts(counts) == {0: (4, 4, 1, 1, 1), 1: (0, 4, 1, 0, 0)}
    assert dataclasses.is_dataclass(LoopRingCall)
    assert issubclass(LoopRingCall, HybridStackCallRoles)
