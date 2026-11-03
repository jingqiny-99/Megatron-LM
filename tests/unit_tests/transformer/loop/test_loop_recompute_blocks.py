# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Block and edge ``loop_recompute`` of the hybrid loop (``loop_recompute_block_stride`` /
``loop_recompute_edges``; ``LoopHybridStack._checkpointed_blocks``).

With a stride ``s`` each Huginn round is cut into ``ceil(C / s)`` non-reentrant checkpoints of
``s`` core sublayers (the adapter in the first block, the carrier diagnostics and the loop norm
in the last); with edges the prelude and the coda are cut the same way. Each checkpoint has its
own routing tape and keeps only its inputs. The math does not change:

* Bitwise: the output (loss), the input gradient and every parameter gradient of strides 1, 2,
  3 (odd, not dividing the core) and C, edges off and on, equal the round checkpoint and no
  recompute bitwise. Dense and 4-stream mHC stacks, truncated BPTT, an untied core, the
  residual carrier, per-row depths, the carrier diagnostics, a routed MoE core, and HybridModels
  with MTP (dense and mHC).
* Routing tape: a routed MoE stub whose replays see perturbed router inputs (a stand-in for a
  recompute that is not bitwise its forward). One tape per checkpoint, labels unique per
  (microbatch, round, block), each replay routes as its forward, every tape is consumed, and
  the backward raises no ``CheckpointError``.
* FLOP rows: the forward and backward rows do not change; the recompute row is rounds x core,
  plus the prelude and the coda with edges (never the output layer or MTP).
* Saved activations: with edges nothing inside the layer loop is saved outside a checkpoint;
  the checkpoints hold ``ceil(P/s) + R ceil(C/s) + ceil(D/s)`` block inputs plus the injection;
  what the embedding, the head and MTP save does not change.
* Guards: every new rule, by name, where it runs.

Local layers in fp32 run on CPU (gloo, one rank; mHC needs TORCHDYNAMO_DISABLE=1) and CUDA.
The DSv4 model case (W / C / H attention, MoE with a hash-routed prelude layer, mHC 4, MTP) is
CUDA-only. Model parallel is initialized once per module.
"""

import contextlib
import copy
import math
import re

import pytest
import torch
from torch.utils.checkpoint import set_checkpoint_early_stop

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.transformer import forward_replay, loop_transformer_block
from megatron.core.transformer.forward_replay import in_forward_replay
from megatron.core.transformer.loop import guards, hybrid_stack
from megatron.core.transformer.loop.attn_res_stack import LoopAttnResHybridStack
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.unit_tests.transformer.loop import hybrid_loop_utils
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    BATCH,
    CUDA,
    HIDDEN,
    SEQ,
    VOCAB,
    build_local_loop_model,
    build_local_loop_stack,
    device,
    local_hybrid_submodules,
    model_parallel_once,
    rope,
)

# The CPU mHC and MTP fixtures, made visible to pytest in this module.
mhc_on_this_device = hybrid_loop_utils.mhc_on_this_device
cpu_mtp = hybrid_loop_utils.cpu_mtp

gpu = pytest.mark.skipif(not CUDA, reason="TE / MoE / DSv4 layers need CUDA")

PATTERN = "*-" * 6  # prelude '*-*-', core '*-*-', coda '*-*-'
P = C = D = 4
MHC = dict(enable_mhc_connections=True, mhc_num_residual_streams=4, mhc_sinkhorn_iterations=5)
# 3 is odd and does not divide C (blocks of 3 and 1); C is one block per round.
STRIDES = (1, 2, 3, C)
BLOCK_MODES = [(stride, edges) for stride in STRIDES for edges in (False, True)]
MODES = ["round"] + BLOCK_MODES
RECOMPUTE_ROW = "loop recompute flops per token"


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


def recompute_fields(mode):
    """The ``loop_recompute*`` fields of ``mode``: 'none', 'round' or ``(stride, edges)``."""
    if mode == "none":
        return {}
    if mode == "round":
        return dict(loop_recompute=True)
    stride, edges = mode
    return dict(loop_recompute=True, loop_recompute_block_stride=stride, loop_recompute_edges=edges)


def _mode_id(mode):
    return mode if isinstance(mode, str) else f"s{mode[0]}{'-edges' if mode[1] else ''}"


def _blocks(length, stride):
    return math.ceil(length / stride)


def _stack(mode, *, pattern=PATTERN, prelude=P, coda=D, **overrides):
    fields = dict(loop_iterations=2)
    fields.update(overrides)
    return build_local_loop_stack(pattern, prelude, coda, **recompute_fields(mode), **fields)


def _inputs(seed=7):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(SEQ, BATCH, HIDDEN, generator=generator).to(device())
    upstream = torch.randn(SEQ, BATCH, HIDDEN, generator=generator).to(device())
    return x, upstream


def _run(stack, x, upstream, depths=None):
    """Forward + backward of ``sum(y * upstream) + loop losses``; detached
    ``(y, dx, grads, metrics)``."""
    x = x.detach().clone().requires_grad_(True)
    stack.zero_grad(set_to_none=True)
    if depths is not None:
        stack.set_loop_depths(torch.tensor(depths, device=device()))
    y = stack(x, attention_mask=None, rotary_pos_emb=rope())
    ((y * upstream).sum() + sum(stack.loop_losses.values())).backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in stack.named_parameters()
        if param.grad is not None
    }
    metrics = {name: value.detach().clone() for name, value in stack.loop_metrics.items()}
    return y.detach().clone(), x.grad.detach().clone(), grads, metrics


def _assert_bitwise(actual, expected, label):
    """Outputs, input gradients, every parameter gradient and every loop metric except the
    recompute FLOP row (which depends on the mode) are bitwise equal."""
    *tensors, grads, metrics = actual
    *tensors_ref, grads_ref, metrics_ref = expected
    for name, value, reference in zip(("output", "input grad"), tensors, tensors_ref):
        assert torch.equal(value, reference), f"{label}: {name}"
    assert grads.keys() == grads_ref.keys(), label
    for name, grad in grads_ref.items():
        assert torch.equal(grads[name], grad), f"{label}: {name}"
    assert metrics.keys() == metrics_ref.keys(), label
    for name, value in metrics_ref.items():
        if name != RECOMPUTE_ROW:
            assert torch.equal(metrics[name], value), f"{label}: {name}"


# ---------------------------------------------------------------------------------------------
# Bitwise: blocks = the round checkpoint = no recompute
# ---------------------------------------------------------------------------------------------

# variant: (stack overrides, per-row depths, carrier diagnostics)
_VARIANTS = {
    "dense": ({}, None, False),
    "truncated_bptt": (dict(loop_iterations=3, loop_bptt_iterations=2), None, False),
    # Core 8 = two untied segments of 4: each round runs its own segment, adapter and norm.
    "untied": (dict(prelude=2, coda=2, loop_untie_core=True), None, False),
    "residual_carrier": (dict(loop_carrier="residual"), None, False),
    "row_depths": ({}, (1, 2), False),
    "diagnostics": ({}, None, True),
}


def _check_modes(build, run, modes=MODES):
    """``run(build(mode))`` for every mode equals ``run(build('none'))`` bitwise."""
    reference_stack = build("none")
    reference = run(reference_stack)
    for mode in modes:
        stack = build(mode)
        stack.load_state_dict(reference_stack.state_dict())
        _assert_bitwise(run(stack), reference, _mode_id(mode))


@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_blocks_are_bitwise_the_round_checkpoint_and_no_recompute(variant):
    overrides, depths, diagnostics = _VARIANTS[variant]
    x, upstream = _inputs()

    def build(mode):
        stack = _stack(mode, **overrides)
        stack.enable_carrier_diagnostics(diagnostics)
        return stack

    def run(stack):
        result = _run(stack, x, upstream, depths)
        if diagnostics:
            # Recorded once per training forward, never by a replay: prelude and both rounds.
            assert {name for name in result[3] if name.startswith("loop state rms")} == {
                "loop state rms prelude",
                "loop state rms r1",
                "loop state rms r2",
            }
        return result

    _check_modes(build, run)


def test_blocks_are_bitwise_on_mhc_streams(mhc_on_this_device):
    x, upstream = _inputs()
    _check_modes(lambda mode: _stack(mode, **MHC), lambda stack: _run(stack, x, upstream))


def _model_run(model, tokens, labels):
    """Forward + backward of the per-token loss sum of a HybridModel (with MTP)."""
    model.zero_grad(set_to_none=True)
    position_ids = torch.arange(SEQ, device=device()).repeat(BATCH, 1)
    loss = model(tokens, position_ids, None, labels=labels)
    loss.sum().backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    metrics = {name: value.detach().clone() for name, value in model.decoder.loop_metrics.items()}
    return loss.detach().clone(), loss.detach().clone(), grads, metrics


def _tokens():
    generator = torch.Generator().manual_seed(5)
    tokens = torch.randint(0, VOCAB, (BATCH, SEQ), generator=generator).to(device())
    labels = torch.randint(0, VOCAB, (BATCH, SEQ), generator=generator).to(device())
    return tokens, labels


def _model(mode, **overrides):
    return build_local_loop_model(
        PATTERN + "/*-", prelude=P, coda=D, loop_iterations=2, **recompute_fields(mode), **overrides
    )


@pytest.mark.parametrize("mhc", [False, True], ids=["dense", "mhc"])
def test_blocks_are_bitwise_in_a_model_with_mtp(request, cpu_mtp, mhc):
    overrides = {}
    if mhc:
        request.getfixturevalue("mhc_on_this_device")
        overrides = MHC
    tokens, labels = _tokens()
    _check_modes(
        lambda mode: _model(mode, **overrides), lambda model: _model_run(model, tokens, labels)
    )


# ---------------------------------------------------------------------------------------------
# Routing tape per block (a routed MoE stub, perturbed replays)
# ---------------------------------------------------------------------------------------------

EXPERTS, TOPK = 4, 2
MOE_PATTERN = "*E" * 6  # an 'E' in every attention + MoE block
STUB_MOE = dict(num_moe_experts=EXPERTS, moe_ffn_hidden_size=128, moe_router_topk=TOPK)


class TapedStubMoE(MegatronModule):
    """A test-only routed MoE MLP (``test_loop_routing_tape_sites.RoutedStubMoE``'s routing:
    a softmax top-``TOPK`` of ``EXPERTS`` through ``topk_routing_with_score_function``, one
    linear expert per expert on its own rows, so the saved tensors' shapes follow the routing).

    With ``perturb_replay`` the router input of an activation replay gets seeded noise. ``log``
    gets one ``(replay, tape, indices used, indices its own scores choose)`` per call, ``tape``
    being the active routing tape (None outside one)."""

    perturb_replay = False
    log: list = []

    def __init__(self, config, pg_collection=None, is_mtp_layer=False, name=None, **kwargs):
        del pg_collection, is_mtp_layer, name, kwargs
        super().__init__(config=config)
        hidden, dtype = config.hidden_size, config.params_dtype
        self.router = torch.nn.Linear(hidden, EXPERTS, bias=False, dtype=dtype)
        self.experts = torch.nn.ModuleList(
            torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype) for _ in range(EXPERTS)
        )

    def forward(self, hidden_states, padding_mask=None, **kwargs):
        """Route, run each expert on its rows, combine with the top-k probabilities."""
        del padding_mask, kwargs
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        replay = in_forward_replay()
        router_input = x
        if TapedStubMoE.perturb_replay and replay:
            noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(9))
            router_input = x + noise.to(x) * x.detach().std()
        scores = self.router(router_input)
        probs, indices = topk_routing_with_score_function(
            scores, TOPK, score_function="softmax", dense_output=True
        )
        own = torch.topk(scores.detach().float().softmax(-1), TOPK, dim=-1).indices
        tape = forward_replay._ROUTING_TAPE.get()  # pylint: disable=protected-access
        TapedStubMoE.log.append(
            (replay, None if tape is None else tape[0], indices.detach().clone(), own)
        )
        out = torch.zeros_like(x)
        for expert, linear in enumerate(self.experts):
            rows, slot = (indices == expert).nonzero(as_tuple=True)
            out = out.index_add(0, rows, linear(x[rows]) * probs[rows, slot, None])
        return out.view_as(hidden_states), None


def _stub_moe_layer_spec():
    local = get_gpt_layer_local_submodules(normalization="RMSNorm")
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            pre_mlp_layernorm=local.pre_mlp_layernorm, mlp=TapedStubMoE, mlp_bda=local.mlp_bda
        ),
    )


@pytest.fixture(autouse=True)
def _stub_state(monkeypatch):
    monkeypatch.setattr(TapedStubMoE, "perturb_replay", False)
    monkeypatch.setattr(TapedStubMoE, "log", [])


def _moe_stack(mode, **overrides):
    submodules = local_hybrid_submodules()
    submodules.moe_layer = _stub_moe_layer_spec()
    return _stack(mode, pattern=MOE_PATTERN, submodules=submodules, **STUB_MOE, **overrides)


def test_moe_blocks_are_bitwise_and_replay_their_tapes():
    x, upstream = _inputs()
    _check_modes(
        lambda mode: _moe_stack(mode, loop_iterations=3), lambda stack: _run(stack, x, upstream)
    )


@pytest.fixture
def recorded_checkpoints(monkeypatch):
    """Record every loop checkpoint's label (in forward order) and every routing tape made."""
    labels, tapes = [], []
    context_fn = hybrid_stack.loop_checkpoint_context_fn

    def recording_context_fn(prefix):
        inner = context_fn(prefix)

        def wrapped():
            labels.append(prefix)
            return inner()

        return wrapped

    class RecordingTape(forward_replay.RoutingTape):
        """A routing tape that registers itself."""

        def __init__(self):
            super().__init__()
            tapes.append(self)

    monkeypatch.setattr(hybrid_stack, "loop_checkpoint_context_fn", recording_context_fn)
    monkeypatch.setattr(loop_transformer_block, "RoutingTape", RecordingTape)
    return labels, tapes


def _expected_labels(microbatch, stride, rounds, edges=True):
    """The checkpoint labels of one forward, in order."""
    prefix = f"loop_hybrid/mb={microbatch}"
    labels = []
    if edges:
        labels += [f"{prefix}/prelude/block={k}" for k in range(1, _blocks(P, stride) + 1)]
    for r in range(1, rounds + 1):
        labels += [f"{prefix}/loop={r}/block={k}" for k in range(1, _blocks(C, stride) + 1)]
    if edges:
        labels += [f"{prefix}/coda/block={k}" for k in range(1, _blocks(D, stride) + 1)]
    return labels


def _moe_calls_per_block(stride, rounds):
    """The MoE sublayers of each block of one forward, in checkpoint order."""
    counts = []
    for start, length, repeats in ((0, P, 1), (P, C, rounds), (P + C, D, 1)):
        segment = MOE_PATTERN[start : start + length]
        cuts = range(0, length, stride)
        counts += [segment[cut : cut + stride].count("E") for cut in cuts] * repeats
    return counts


@pytest.mark.parametrize("stride", [1, 3])
def test_every_block_has_its_own_tape_and_replays_its_routing(recorded_checkpoints, stride):
    """Perturbed replays: one tape per block, unique labels per (microbatch, round, block),
    each replay routes exactly as its forward, every tape is fully replayed, no
    ``CheckpointError`` (two microbatches)."""
    labels, tapes = recorded_checkpoints
    rounds = 3
    stack = _moe_stack((stride, True), loop_iterations=rounds)
    TapedStubMoE.perturb_replay = True
    x, upstream = _inputs()
    for _ in range(2):
        # Without early stop every replay runs to its block's end, so replay_routing also
        # checks that each tape was consumed.
        with set_checkpoint_early_stop(False):
            _, dx, grads, _ = _run(stack, x, upstream)
        assert torch.isfinite(dx).all()
        assert any(name.endswith("router.weight") for name in grads)
        assert all(torch.isfinite(grad).all() for grad in grads.values())

    expected = _expected_labels(0, stride, rounds) + _expected_labels(1, stride, rounds)
    assert labels == expected
    assert len(set(labels)) == len(labels)
    assert len(tapes) == len(labels)  # one tape per checkpoint call
    assert [len(tape.indices) for tape in tapes] == 2 * _moe_calls_per_block(stride, rounds)
    assert all(tape.cursor == len(tape.indices) for tape in tapes)

    # Every call routed under a block's tape; each replay used its forward's choices in order,
    # and the perturbation would have changed some of them.
    forwards = {id(tape): [] for tape in tapes}
    replays = {id(tape): [] for tape in tapes}
    flipped = 0
    for replay, tape, indices, own in TapedStubMoE.log:
        assert tape is not None
        (replays if replay else forwards)[id(tape)].append(indices)
        if replay:
            flipped += int(not torch.equal(own.sort(-1).values, indices.sort(-1).values))
    for tape in tapes:
        recorded = forwards[id(tape)]
        assert len(replays[id(tape)]) == len(recorded) == len(tape.indices)
        for used, chosen in zip(replays[id(tape)], recorded):
            assert torch.equal(used, chosen)
    assert flipped > 0


# ---------------------------------------------------------------------------------------------
# FLOP rows
# ---------------------------------------------------------------------------------------------


def _pass_flops(stack, start, end, extra=0):
    tokens = SEQ * BATCH
    return stack.range_forward_flops(start, end, tokens, tokens * SEQ, SEQ, extra=extra)


@pytest.mark.parametrize("mode", ["none"] + MODES, ids=_mode_id)
def test_flop_rows(mode):
    rounds = 2
    stack = _stack(mode, loop_iterations=rounds)
    x, upstream = _inputs()
    _run(stack, x, upstream)
    tokens = SEQ * BATCH
    prelude = _pass_flops(stack, 0, P)
    core = _pass_flops(stack, P, P + C, extra=stack.round_extra_forward_flops(tokens))
    coda = _pass_flops(stack, P + C, P + C + D)
    forward = prelude + rounds * core + coda
    if mode == "none":
        recompute = 0
    else:
        edges = mode != "round" and mode[1]
        recompute = rounds * core + (prelude + coda if edges else 0)
    metrics = stack.loop_metrics
    assert metrics["loop forward flops per token"].item() == pytest.approx(forward / tokens)
    assert metrics["loop backward flops per token"].item() == pytest.approx(2 * forward / tokens)
    assert metrics[RECOMPUTE_ROW].item() == pytest.approx(recompute / tokens)
    if mode != "none" and mode != "round" and mode[1]:
        # Without grad nothing is checkpointed and nothing is recomputed.
        with torch.no_grad():
            stack(x, attention_mask=None, rotary_pos_emb=rope())
        assert stack.loop_metrics["loop forward flops per token"].item() == pytest.approx(
            forward / tokens
        )
        assert stack.loop_metrics[RECOMPUTE_ROW] == 0


def test_edges_never_recompute_the_head_or_mtp(cpu_mtp):
    tokens, labels = _tokens()
    rows = {}
    for mode in ("none", "round", (2, True)):
        model = _model(mode)
        _model_run(model, tokens, labels)
        rows[_mode_id(mode)] = {k: v.item() for k, v in model.decoder.loop_metrics.items()}
    decoder = model.decoder
    count = SEQ * BATCH
    prelude = _pass_flops(decoder, 0, P)
    core = _pass_flops(decoder, P, P + C, extra=decoder.round_extra_forward_flops(count))
    coda = _pass_flops(decoder, P + C, P + C + D)
    head = sum(work for work, _ in decoder.head_work_terms(count, count * SEQ, SEQ))
    assert head > 0
    for name in ("loop forward flops per token", "loop backward flops per token"):
        assert rows["s2-edges"][name] == rows["round"][name] == rows["none"][name]
    assert rows["none"][RECOMPUTE_ROW] == 0
    assert rows["round"][RECOMPUTE_ROW] == pytest.approx(2 * core / count)
    assert rows["s2-edges"][RECOMPUTE_ROW] == pytest.approx((prelude + 2 * core + coda) / count)


# ---------------------------------------------------------------------------------------------
# Saved activations
# ---------------------------------------------------------------------------------------------


def _storage(tensor):
    storage = tensor.untyped_storage()
    return storage.data_ptr(), storage.nbytes()


@contextlib.contextmanager
def saved_activations(module):
    """Account the activations a forward of ``module`` keeps for its backward.

    Yields a dict that, after the scope, holds: ``in_loop`` / ``outside_loop`` (the storages
    autograd saves outside every activation checkpoint's internals, from inside / outside a
    loop stack's ``_layer_loop``) and ``checkpoint_inputs`` (the storages passed to a loop
    checkpoint), each ``{data_ptr: nbytes}`` without parameters. Inside a checkpoint its own
    saved-tensor hooks take precedence, so its internals are not seen; a non-reentrant
    checkpoint keeps its inputs by saving them through a no-op autograd function, outside its
    hooks, so they appear in ``in_loop`` too."""
    params = {_storage(p)[0] for p in module.parameters()}
    found = {"in_loop": {}, "outside_loop": {}, "checkpoint_inputs": {}}
    inside = [0]
    layer_loop = LoopHybridStack._layer_loop
    real_checkpoint = hybrid_stack.checkpoint

    def tracked_layer_loop(self, *args, **kwargs):
        inside[0] += 1
        try:
            return layer_loop(self, *args, **kwargs)
        finally:
            inside[0] -= 1

    def tracked_checkpoint(function, *args, **kwargs):
        for arg in args:
            if isinstance(arg, torch.Tensor) and _storage(arg)[0] not in params:
                ptr, nbytes = _storage(arg)
                found["checkpoint_inputs"][ptr] = nbytes
        return real_checkpoint(function, *args, **kwargs)

    def pack(tensor):
        ptr, nbytes = _storage(tensor)
        if ptr not in params and nbytes:
            found["in_loop" if inside[0] else "outside_loop"][ptr] = nbytes
        return tensor

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LoopHybridStack, "_layer_loop", tracked_layer_loop)
        patch.setattr(hybrid_stack, "checkpoint", tracked_checkpoint)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            yield found


def _stack_activations(stack, x):
    with saved_activations(stack) as found:
        y = stack(
            x.detach().clone().requires_grad_(True), attention_mask=None, rotary_pos_emb=rope()
        )
    y.sum().backward()
    return found


@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("mhc", [False, True], ids=["dense", "mhc"])
def test_blocks_keep_only_their_inputs(request, mhc, stride):
    overrides = {}
    if mhc:
        request.getfixturevalue("mhc_on_this_device")
        overrides = MHC
    rounds = 2
    x, _ = _inputs()
    streams = 4 if mhc else 1
    activation = SEQ * BATCH * streams * HIDDEN * x.element_size()

    blocks = _stack_activations(_stack((stride, True), **overrides), x)
    # Everything the layer loop keeps is a checkpoint input: the block inputs (the round state
    # of each round's first block included) plus the injection, once.
    assert blocks["in_loop"] == blocks["checkpoint_inputs"]
    held = _blocks(P, stride) + rounds * _blocks(C, stride) + _blocks(D, stride) + 1
    assert sorted(blocks["checkpoint_inputs"].values()) == [activation] * held

    def beyond_checkpoints(found):
        """Bytes the layer loop keeps besides the checkpoint inputs."""
        inputs = found["checkpoint_inputs"]
        return sum(nbytes for ptr, nbytes in found["in_loop"].items() if ptr not in inputs)

    core_only = _stack_activations(_stack((stride, False), **overrides), x)
    round_mode = _stack_activations(_stack("round", **overrides), x)
    none = _stack_activations(_stack("none", **overrides), x)
    assert core_only["checkpoint_inputs"].keys() <= core_only["in_loop"].keys()
    assert len(core_only["checkpoint_inputs"]) == rounds * _blocks(C, stride) + 1
    assert len(round_mode["checkpoint_inputs"]) == rounds + 1
    # Without edges the prelude and the coda keep their activations, as in round mode.
    assert beyond_checkpoints(core_only) == beyond_checkpoints(round_mode) > 0
    assert none["checkpoint_inputs"] == {}
    assert sum(none["in_loop"].values()) > sum(round_mode["in_loop"].values())


def test_edges_leave_the_embedding_head_and_mtp_activations(cpu_mtp):
    tokens, labels = _tokens()
    position_ids = torch.arange(SEQ, device=device()).repeat(BATCH, 1)
    found = {}
    for mode in ("none", (2, True)):
        model = _model(mode)
        with saved_activations(model) as found[_mode_id(mode)]:
            loss = model(tokens, position_ids, None, labels=labels)
        loss.sum().backward()
    blocks, none = found["s2-edges"], found["none"]
    assert blocks["in_loop"] == blocks["checkpoint_inputs"]
    held = _blocks(P, 2) + 2 * _blocks(C, 2) + _blocks(D, 2) + 1
    assert len(blocks["checkpoint_inputs"]) == held
    assert sum(none["in_loop"].values()) > sum(blocks["in_loop"].values())
    # The embedding, the output layer, the loss and MTP save the same bytes in both modes.
    assert sorted(blocks["outside_loop"].values()) == sorted(none["outside_loop"].values())


# ---------------------------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------------------------

_PREFIX = "Hybrid loop does not support: "


def _hybrid_config(**overrides):
    """A dense hybrid loop config of PATTERN: prelude 4, core 4, coda 4."""
    fields = dict(
        num_layers=len(PATTERN),
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        is_hybrid_model=True,
        loop_architecture="huginn",
        loop_prelude_sublayers=P,
        loop_coda_sublayers=D,
    )
    fields.update(overrides)
    return TransformerConfig(**fields)


def _violating(config, **fields):
    """A copy of a valid config with ``fields`` set directly (no ``__post_init__``)."""
    config = copy.copy(config)
    for name, value in fields.items():
        assert hasattr(config, name), name
        setattr(config, name, value)
    return config


def _reported(excinfo):
    message = str(excinfo.value)
    assert message.startswith(_PREFIX), message
    return message[len(_PREFIX) :].split(", ")


_CONFIG_RULES = [
    (guards.RECOMPUTE_BLOCKS_WITHOUT_RECOMPUTE, dict(loop_recompute_block_stride=2)),
    (
        guards.RECOMPUTE_BLOCKS_WITHOUT_RECOMPUTE,
        dict(loop_recompute_block_stride=2, loop_recompute_edges=True),
    ),
    (guards.RECOMPUTE_EDGES_WITHOUT_STRIDE, dict(loop_recompute=True, loop_recompute_edges=True)),
    (
        guards.RECOMPUTE_BLOCKS_MOR,
        dict(loop_recompute=True, loop_recompute_block_stride=2, loop_architecture="mor"),
    ),
    (
        guards.RECOMPUTE_BLOCKS_ATTN_RES,
        dict(loop_recompute=True, loop_recompute_block_stride=2, loop_carrier="attn_res"),
    ),
]


@pytest.mark.parametrize(
    "rule,fields", _CONFIG_RULES, ids=[f"{rule[:40]}-{'-'.join(f)}" for rule, f in _CONFIG_RULES]
)
def test_config_rule_is_reported_alone(rule, fields):
    with pytest.raises(ValueError) as excinfo:
        guards.validate_hybrid_loop_config(_violating(_hybrid_config(), **fields))
    assert _reported(excinfo) == [rule]


def test_rule_names_survive_the_message_list():
    names = [
        guards.RECOMPUTE_BLOCKS_WITHOUT_RECOMPUTE,
        guards.RECOMPUTE_EDGES_WITHOUT_STRIDE,
        guards.RECOMPUTE_STRIDE_RANGE,
        guards.RECOMPUTE_BLOCKS_MOR,
        guards.RECOMPUTE_BLOCKS_ATTN_RES,
        guards.RECOMPUTE_BLOCKS_WITHOUT_HYBRID_LOOP,
    ]
    assert len(set(names)) == len(names)
    assert not any(", " in name for name in names)  # the guards join rule names with ", "


def test_edges_alone_break_two_rules():
    config = _violating(_hybrid_config(), loop_recompute_edges=True)
    with pytest.raises(ValueError) as excinfo:
        guards.validate_hybrid_loop_config(config)
    assert _reported(excinfo) == [
        guards.RECOMPUTE_BLOCKS_WITHOUT_RECOMPUTE,
        guards.RECOMPUTE_EDGES_WITHOUT_STRIDE,
    ]


@pytest.mark.parametrize(
    "fields",
    [
        dict(),
        dict(loop_recompute=True),
        dict(loop_recompute=True, loop_recompute_block_stride=1),
        dict(loop_recompute=True, loop_recompute_block_stride=C, loop_recompute_edges=True),
        dict(loop_recompute=True, loop_recompute_block_stride=3, loop_carrier="residual"),
    ],
)
def test_block_configs_pass(fields):
    config = _hybrid_config(**fields)
    guards.validate_hybrid_loop_config(config)
    assert guards.resolve_hybrid_loop_roles(config, PATTERN, None).core == C


@pytest.mark.parametrize(
    "stride,valid", [(0, False), (1, True), (3, True), (C, True), (C + 1, False)]
)
def test_stride_range_is_a_role_rule(stride, valid):
    config = _violating(_hybrid_config(loop_recompute=True), loop_recompute_block_stride=stride)
    guards.validate_hybrid_loop_config(config)  # the range needs the pattern
    if valid:
        guards.resolve_hybrid_loop_roles(config, PATTERN, None)
        return
    with pytest.raises(ValueError) as excinfo:
        guards.resolve_hybrid_loop_roles(config, PATTERN, None)
    assert _reported(excinfo) == [guards.RECOMPUTE_STRIDE_RANGE]


def test_stride_range_counts_one_untied_segment():
    # Core 8 = two untied segments of 4: one round runs 4 sublayers.
    config = _hybrid_config(
        loop_prelude_sublayers=2,
        loop_coda_sublayers=2,
        loop_iterations=2,
        loop_untie_core=True,
        loop_recompute=True,
        loop_recompute_block_stride=4,
    )
    assert guards.resolve_hybrid_loop_roles(config, PATTERN, None).core == 8
    with pytest.raises(ValueError) as excinfo:
        guards.resolve_hybrid_loop_roles(
            _violating(config, loop_recompute_block_stride=5), PATTERN, None
        )
    assert _reported(excinfo) == [guards.RECOMPUTE_STRIDE_RANGE]


_NON_HYBRID = {
    "no loop": dict(num_layers=4),
    "gpt loop": dict(
        num_layers=4,
        loop_architecture="huginn",
        loop_prelude_layers=1,
        loop_core_layers=2,
        loop_recompute=True,
    ),
}


@pytest.mark.parametrize(
    "fields", [dict(loop_recompute_block_stride=2), dict(loop_recompute_edges=True)]
)
@pytest.mark.parametrize("loop", list(_NON_HYBRID))
def test_block_fields_need_a_hybrid_loop(loop, fields):
    base = dict(
        hidden_size=64,
        num_attention_heads=4,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
    )
    message = "Loop does not support: " + guards.RECOMPUTE_BLOCKS_WITHOUT_HYBRID_LOOP
    with pytest.raises(ValueError, match=re.escape(message)):
        TransformerConfig(**base, **_NON_HYBRID[loop], **fields)
    TransformerConfig(**base, **_NON_HYBRID[loop])  # the same config without them is valid


def test_stacks_run_the_block_rules():
    with pytest.raises(ValueError, match=re.escape(guards.RECOMPUTE_STRIDE_RANGE)):
        _stack((C + 1, False))
    with pytest.raises(ValueError, match=re.escape(guards.RECOMPUTE_BLOCKS_MOR)):
        _stack((2, False), loop_architecture="mor", loop_token_execution="masked")
    with pytest.raises(ValueError, match=re.escape(guards.RECOMPUTE_BLOCKS_ATTN_RES)):
        _stack((2, False), module=LoopAttnResHybridStack, loop_carrier="attn_res")
    with pytest.raises(ValueError, match=re.escape(guards.RECOMPUTE_EDGES_WITHOUT_STRIDE)):
        _stack("round", loop_recompute_edges=True)
    # Round mode on the same stacks is unchanged.
    _stack("round", loop_architecture="mor", loop_token_execution="masked")
    _stack("round", module=LoopAttnResHybridStack, loop_carrier="attn_res")


def test_rsbf_still_rejects_loop_recompute():
    config = _violating(
        _hybrid_config(loop_schedule="rsbf", calculate_per_token_loss=True),
        loop_recompute=True,
        loop_recompute_block_stride=2,
        loop_recompute_edges=True,
    )
    with pytest.raises(ValueError, match="Loop RSBF does not support") as excinfo:
        guards.validate_rsbf_config(config)
    assert guards.RSBF_LOOP_RECOMPUTE in str(excinfo.value)


# ---------------------------------------------------------------------------------------------
# DSv4 model (CUDA): mHC 4, a hash-routed prelude layer, MTP
# ---------------------------------------------------------------------------------------------

DSV4_PATTERN = "WECEHECEHE/WE"  # prelude 'WE' (hash E = layer 2), core 'CEHECE', coda 'HE'
DSV4_RATIOS = {"W": 0, "C": 4, "H": 128, "E": 0}
DSV4_MODES = ["round"] + [(stride, edges) for stride in (1, 2, 4, 6) for edges in (False, True)]


@pytest.fixture
def deterministic_algorithms(monkeypatch):
    """Deterministic kernels for the bitwise GPU comparisons (set after the module's model
    parallel initialization, which clears the NVTE variables)."""
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _dsv4_config(mode):
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    main, mtp = DSV4_PATTERN.split("/")
    return MLATransformerConfig(
        num_layers=len(main),
        hidden_size=512,
        num_attention_heads=8,
        multi_latent_attention=True,
        q_lora_rank=192,
        kv_lora_rank=64,
        qk_head_dim=16,
        qk_pos_emb_head_dim=8,
        v_head_dim=16,
        rope_type="rope",
        rotary_base=10000,
        rotary_percent=1.0,
        experimental_attention_variant="dsv4_hybrid",
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=32,
        # The sparse indexer loss is on, as in the Flash configs: every replay must enter the
        # indexer-loss padding scope.
        dsa_indexer_loss_coeff=1e-2,
        dsa_indexer_use_sparse_loss=True,
        csa_window_size=32,
        csa_compress_ratios=[DSV4_RATIOS[symbol] for symbol in main + mtp],
        # The Flash router: sqrtsoftplus, top-k scaling, expert bias, sequence aux loss, and
        # a shared expert.
        num_moe_experts=8,
        moe_ffn_hidden_size=256,
        moe_shared_expert_intermediate_size=256,
        moe_router_topk=2,
        moe_router_score_function="sqrtsoftplus",
        moe_router_topk_scaling_factor=1.5,
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=1e-4,
        moe_router_enable_expert_bias=True,
        moe_token_dispatcher_type="alltoall",
        moe_n_hash_layers=1,
        actual_vocab_size=VOCAB,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=5,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        loop_architecture="huginn",
        loop_iterations=2,
        loop_prelude_sublayers=2,
        loop_coda_sublayers=2,
        deterministic_mode=True,
        attention_backend=AttnBackend.unfused,
        **recompute_fields(mode),
    )


def _dsv4_model(mode):
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_dsv4_stack_spec

    config = _dsv4_config(mode)
    torch.manual_seed(1234)
    model = HybridModel(
        config=config,
        hybrid_stack_spec=loop_hybrid_dsv4_stack_spec(config),
        vocab_size=VOCAB,
        max_sequence_length=1024,
        hybrid_layer_pattern=DSV4_PATTERN,
        position_embedding_type="rope",
    ).cuda()
    assert type(model.decoder) is LoopHybridStack
    assert type(model.mtp.layers[0].mtp_model_layer) is HybridStack
    assert model.decoder.layers[1].inner_layer.mlp.router.is_hash_layer
    return model


def _dsv4_run(model, tokens, labels):
    """One forward + backward; the loss, every gradient, and the decoder layer calls as
    ``(layer number, has input ids, in a replay)``."""
    from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker

    set_moe_metrics_tracker(MoEMetricsTracker())
    calls = []
    hooks = [
        layer.register_forward_pre_hook(
            lambda module, args, kwargs: calls.append(
                (module.layer_number, kwargs.get("input_ids") is not None, in_forward_replay())
            ),
            with_kwargs=True,
        )
        for layer in model.decoder.layers
    ]
    model.zero_grad(set_to_none=True)
    position_ids = torch.arange(tokens.shape[1], device="cuda").repeat(tokens.shape[0], 1)
    try:
        loss = model(tokens, position_ids, None, labels=labels)
        loss.float().mean().backward()
    finally:
        for hook in hooks:
            hook.remove()
    grads = {
        name: param.grad.detach().float().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return loss.detach().float().clone(), grads, calls


def _rel_cos(actual, expected):
    """``(relative norm of the difference, cosine)`` of two tensors."""
    a, b = actual.flatten().double(), expected.flatten().double()
    if not b.norm() and not a.norm():
        return 0.0, 1.0
    rel = ((a - b).norm() / b.norm()).item() if b.norm() else float("inf")
    return rel, torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _close(actual, expected):
    """The non-bitwise fallback gate: cosine >= 0.99999 and relative norm <= 1e-5."""
    rel, cosine = _rel_cos(actual, expected)
    return cosine >= 0.99999 and rel <= 1e-5


def _fallback_report(grads, grads_ref, rerun_grads, failing, limit=10):
    """The failing parameters with their difference and the reference's own rerun floor."""
    rows = []
    for name in failing[:limit]:
        rel, cosine = _rel_cos(grads[name], grads_ref[name])
        floor, _ = _rel_cos(rerun_grads[name], grads_ref[name])
        rows.append(f"{name}: rel {rel:.2e} cos {cosine:.7f} (reference rerun rel {floor:.2e})")
    return "; ".join(rows)


@gpu
def test_dsv4_model_blocks_and_edges_are_bitwise(deterministic_algorithms):
    """DSv4 W / C / H attention (unfused), MoE with the hash-routed prelude layer, mHC 4, MTP:
    loss and every gradient of every mode equal no recompute bitwise. A mismatch is diagnosed by
    a self-rerun of the reference; only if that rerun is not bitwise (a non-deterministic
    kernel) does the cosine / relative-norm gate apply. The hash layer gets its ids in its
    forward call and in every prelude replay."""
    import unittest.mock

    try:
        from fast_hadamard_transform import hadamard_transform  # noqa: F401

        hadamard_patch = contextlib.nullcontext()
    except ImportError:
        hadamard_patch = unittest.mock.patch(
            "megatron.core.transformer.experimental_attention_variant.csa.rotate_activation",
            lambda x: x * (x.size(-1) ** -0.5),
        )
    seq = 256
    generator = torch.Generator().manual_seed(5)
    tokens = torch.randint(0, VOCAB, (BATCH, seq), generator=generator).cuda()
    labels = torch.randint(0, VOCAB, (BATCH, seq), generator=generator).cuda()
    hash_layer = 2
    with hadamard_patch:
        reference_model = _dsv4_model("none")
        loss_ref, grads_ref, calls_ref = _dsv4_run(reference_model, tokens, labels)
        assert [number for number, has_ids, _ in calls_ref if has_ids] == [hash_layer]
        deterministic, rerun_grads = None, None
        for mode in DSV4_MODES:
            model = _dsv4_model(mode)
            model.load_state_dict(reference_model.state_dict())
            loss, grads, calls = _dsv4_run(model, tokens, labels)
            label = _mode_id(mode)
            hash_calls = [(ids, replay) for number, ids, replay in calls if number == hash_layer]
            edges = mode != "round" and mode[1]
            assert hash_calls == ([(True, False), (True, True)] if edges else [(True, False)])
            assert grads.keys() == grads_ref.keys()
            bitwise = torch.equal(loss, loss_ref) and all(
                torch.equal(grads[name], grads_ref[name]) for name in grads_ref
            )
            print(f"[T1] {label}: {'bitwise' if bitwise else 'not bitwise'} ({len(grads)} grads)")
            if bitwise:
                continue
            if deterministic is None:
                rerun_loss, rerun_grads, _ = _dsv4_run(reference_model, tokens, labels)
                deterministic = torch.equal(rerun_loss, loss_ref) and all(
                    torch.equal(rerun_grads[name], grads_ref[name]) for name in grads_ref
                )
                print(f"[T1] the reference repeats itself bitwise: {deterministic}")
            if deterministic:
                worst = max(
                    ((grads[name] - grads_ref[name]).abs().max().item(), name) for name in grads_ref
                )
                pytest.fail(f"{label}: not bitwise on deterministic kernels (worst {worst})")
            failing = [name for name in grads_ref if not _close(grads[name], grads_ref[name])]
            report = _fallback_report(grads, grads_ref, rerun_grads, failing)
            assert _close(loss, loss_ref) and not failing, f"{label}: {len(failing)}: {report}"
