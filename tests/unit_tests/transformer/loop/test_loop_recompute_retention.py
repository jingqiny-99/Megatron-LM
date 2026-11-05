# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Loop recompute (``loop_recompute``, one non-reentrant ``torch.utils.checkpoint`` per round)
must not keep a round's activation replay alive once that round has back-propagated.

Cases: the injection, residual and attention-residual (k = 1, 2) carriers of the Huginn round and
the masked MoR recurrence, on local stacks with prelude 2, core 4 and coda 2 sublayers (3 rounds):
a dense stack (``'*-*-*-*-'``) and a MoE stack (``'*E*E*E*E'``, a real ``MoELayer`` with the
alltoall dispatcher and local experts in every 'E').

* ``test_replays_are_freed`` (dense) / ``test_moe_replays_are_freed`` (MoE), CPU or CUDA: a
  ``TorchDispatchMode`` active during the backward keeps a storage weak reference of every tensor
  an op allocates inside a replay (``forward_replay.in_forward_replay()``; outputs aliasing an op
  input are not new storage). When a replay starts, no storage of an earlier replay may still be
  alive (rounds back-propagate in sequence, so at most one replay is needed at a time), and after
  the backward, with the output dropped and ``gc.collect()``, none may be alive at all. On the MoE
  stack the only exception is the alltoall dispatcher's integer token counts
  (``num_global_tokens_per_local_expert``, ``EXPERTS`` int64 per MoE sublayer, no autograd
  graph), which every carrier, with or without recompute, keeps until the layer's next forward;
  no dispatcher may still hold its routing tensors (``probs`` carries the router's graph);
* ``test_recompute_is_bitwise``: ``loop_recompute`` changes no output or gradient, per parameter,
  and neither does the checkpoint early stop that ``loop_round_checkpoint`` turns off (a run with
  it on, as before the fix, is bitwise the same);
* ``test_cuda_memory_after_backward`` (CUDA): ``torch.cuda.memory_allocated()`` after the
  backward equals the value before the forward plus the parameter gradients, within one
  ``[S, B, H]`` activation.

The defect these guard (stage-9 deep probe, EOS H100, vendor a696b63a8): torch's default early
stop ends a replay by raising inside the op that packs the round's last saved tensor. For a round
that returns a layer's output (residual, attention residual, MoR) that op is inside the round's
last sublayer, so the layer's success-path cleanup is skipped and its per-call state
(``TransformerLayer.mlp_norm_manager``, the alltoall dispatcher's ``probs`` / ``routing_map`` /
permutation map) keeps the cut-off replay's whole graph alive until the layer runs again
(``loop_carrier='attn_res'``: 35.6 GiB allocated between steps at h1024_c54 vs 7.0 GiB for
loop-mhc, and an OOM in the first backward at h1536_c54). ``loop_round_checkpoint`` turns early
stop off for those rounds; the injection round ends in the loop norm and keeps it.
"""

import contextlib
import gc
from typing import NamedTuple

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.transformer import loop_transformer_block
from megatron.core.transformer.forward_replay import in_forward_replay
from megatron.core.transformer.loop.attn_res_stack import LoopAttnResHybridStack
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    MoETransformerLayer,
    TransformerLayerSubmodules,
)
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    CUDA,
    HIDDEN,
    SEQ,
    build_local_loop_stack,
    device,
    local_hybrid_submodules,
    model_parallel_once,
    rope,
)

PATTERN = "*-*-*-*-"
MOE_PATTERN = "*E*E*E*E"
ROUNDS = 3
EXPERTS = 4
# (carrier, attention-residual blocks per round)
CARRIERS = [("injection", 1), ("residual", 1), ("attn_res", 1), ("attn_res", 2)]
CARRIER_IDS = ["injection", "residual", "attn_res-k1", "attn_res-k2"]
# The carriers and the masked MoR recurrence ("mor": loop_architecture='mor').
CASES = CARRIERS + [("mor", 1)]
CASE_IDS = CARRIER_IDS + ["mor"]
MOE = dict(
    num_moe_experts=EXPERTS,
    moe_ffn_hidden_size=HIDDEN,
    moe_router_topk=2,
    moe_token_dispatcher_type="alltoall",
    moe_router_dtype="fp32",
    moe_aux_loss_coeff=0.0,
)
# The integer token counts an alltoall dispatcher keeps until its next forward (every carrier).
MOE_METADATA_BYTES = MOE_PATTERN.count("E") * EXPERTS * torch.int64.itemsize
_DISPATCHER_ROUTING = ("probs", "routing_map", "reversed_local_input_permutation_mapping")


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


class _FakeStream:
    """A no-op CUDA stream / event (CPU only)."""

    def wait_stream(self, *args, **kwargs):
        """No-op."""

    def wait(self, *args, **kwargs):
        """No-op."""

    def synchronize(self):
        """No-op."""

    def record_event(self, *args, **kwargs):
        """A no-op event."""
        return _FakeStream()

    def __eq__(self, other):
        return isinstance(other, _FakeStream)

    def __hash__(self):
        return 0


@pytest.fixture
def moe_on_this_device(monkeypatch):
    """A fresh MoE metrics tracker (this module's stacks differ from other modules' in depth);
    on CPU, no-op CUDA streams, which the alltoall dispatcher and MoE layer create and query
    (environment limitation, not the code under test). CUDA runs unchanged."""
    set_moe_metrics_tracker(MoEMetricsTracker())
    if not CUDA:
        stream = _FakeStream()
        monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: stream)
        monkeypatch.setattr(torch.cuda, "stream", lambda *args, **kwargs: contextlib.nullcontext())
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    yield
    set_moe_metrics_tracker(MoEMetricsTracker())


def _moe_submodules():
    """``local_hybrid_submodules`` with a local ``MoETransformerLayer`` (alltoall, local
    experts) as the 'E' sublayer."""
    local = get_gpt_layer_local_submodules(normalization="RMSNorm")
    submodules = local_hybrid_submodules()
    submodules.moe_layer = ModuleSpec(
        module=MoETransformerLayer,
        submodules=TransformerLayerSubmodules(
            pre_mlp_layernorm=local.pre_mlp_layernorm,
            mlp=get_moe_module_spec(use_te=False, num_experts=EXPERTS, moe_grouped_gemm=False),
            mlp_bda=local.mlp_bda,
        ),
    )
    return submodules


def _stack(carrier, k, *, moe=False, recompute=True):
    fields = dict(loop_iterations=ROUNDS, loop_recompute=recompute)
    if carrier == "mor":
        fields.update(loop_architecture="mor", loop_token_execution="masked")
    else:
        fields.update(loop_carrier=carrier)
    if carrier == "attn_res":
        fields.update(module=LoopAttnResHybridStack, loop_attn_res_blocks_per_round=k)
    if moe:
        fields.update(submodules=_moe_submodules(), **MOE)
    return build_local_loop_stack(MOE_PATTERN if moe else PATTERN, 2, 2, **fields)


def _forward(stack, x):
    return stack(x, attention_mask=None, rotary_pos_emb=rope())


def _held_layer_inputs(stack):
    """``(sublayer, attribute)`` of the layers still holding a norm manager (and its input)."""
    return [
        (index, name)
        for index, layer in enumerate(stack.layers)
        for name in ("attn_norm_manager", "mlp_norm_manager")
        if getattr(layer, name, None) is not None
    ]


def _held_dispatcher_routing(stack):
    """``(sublayer, attribute)`` of the MoE dispatchers still holding a routing tensor."""
    return [
        (index, name)
        for index, layer in enumerate(stack.layers)
        for name in _DISPATCHER_ROUTING
        if getattr(getattr(getattr(layer, "mlp", None), "token_dispatcher", None), name, None)
        is not None
    ]


def _is_metadata(dtype):
    """Integer counts / indices: no autograd graph, not an activation."""
    return dtype in (torch.int32, torch.int64)


class _Storage(NamedTuple):
    op: str
    nbytes: int
    dtype: torch.dtype
    ref: StorageWeakRef


class _ReplayStorages(TorchDispatchMode):
    """Weak references to the storages that ops allocate inside activation replays.

    A replay is a run of ops under ``in_forward_replay()``; a new replay starts at the first
    replay op after a non-replay op (a backward op). At the start of each replay the bytes of
    earlier replays still alive are recorded in ``alive_at_replay_start``, and those of them
    that are not integer metadata in ``activations_at_replay_start``.
    """

    def __init__(self):
        super().__init__()
        self.replays = []  # per replay: [_Storage]
        self.alive_at_replay_start = []
        self.activations_at_replay_start = []
        self._in_replay = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        replay = in_forward_replay()
        if replay and not self._in_replay:
            alive = self.alive()
            self.alive_at_replay_start.append(sum(s.nbytes for _, s in alive))
            self.activations_at_replay_start.append(
                sum(s.nbytes for _, s in alive if not _is_metadata(s.dtype))
            )
            self.replays.append([])
        self._in_replay = replay
        if replay:
            inputs = {
                t.untyped_storage().data_ptr()
                for t in tree_flatten((args, kwargs))[0]
                if isinstance(t, torch.Tensor)
            }
            for t in tree_flatten(out)[0]:
                if not isinstance(t, torch.Tensor):
                    continue
                storage = t.untyped_storage()
                if storage.nbytes() > 0 and storage.data_ptr() not in inputs:
                    self.replays[-1].append(
                        _Storage(str(func), storage.nbytes(), t.dtype, StorageWeakRef(storage))
                    )
        return out

    def alive(self):
        """``(replay index, _Storage)`` of every replay storage still alive."""
        return [
            (index, storage)
            for index, replay in enumerate(self.replays)
            for storage in replay
            if not storage.ref.expired()
        ]


def _backward_tracked(stack):
    """Forward, then the backward of ``sum(y * upstream)`` under ``_ReplayStorages``; the output
    is dropped and garbage collected before returning the tracker."""
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(SEQ, 2, HIDDEN, generator=generator).to(device()).requires_grad_(True)
    upstream = torch.randn(SEQ, 2, HIDDEN, generator=generator).to(device())
    tracker = _ReplayStorages()
    y = _forward(stack, x)
    with tracker:
        (y * upstream).sum().backward()
    del y
    gc.collect()
    # The test has teeth: one replay per round, each allocating storage.
    assert len(tracker.replays) == ROUNDS, len(tracker.replays)
    assert all(tracker.replays), [len(replay) for replay in tracker.replays]
    return tracker


def _report(name, stack, tracker):
    alive = tracker.alive()
    replay_bytes = [sum(s.nbytes for s in replay) for replay in tracker.replays]
    return (
        f"{name}: replay bytes {replay_bytes}; alive at each replay start "
        f"{tracker.alive_at_replay_start} (activations {tracker.activations_at_replay_start}); "
        f"alive after backward {len(alive)} storages, {sum(s.nbytes for _, s in alive)} B "
        f"(per replay {[sum(s.nbytes for i, s in alive if i == r) for r in range(ROUNDS)]}; "
        f"activations {[(s.op, s.dtype, s.nbytes) for _, s in alive if not _is_metadata(s.dtype)][:6]}"
        f"); layers holding a norm manager {_held_layer_inputs(stack)}; dispatchers holding "
        f"routing tensors {_held_dispatcher_routing(stack)}"
    )


@pytest.mark.parametrize("carrier,k", CASES, ids=CASE_IDS)
def test_replays_are_freed(carrier, k):
    stack = _stack(carrier, k)
    tracker = _backward_tracked(stack)
    report = _report(f"{carrier} k={k}", stack, tracker)
    assert tracker.alive_at_replay_start == [0] * ROUNDS, report
    assert not tracker.alive(), report


@pytest.mark.usefixtures("moe_on_this_device")
@pytest.mark.parametrize("carrier,k", CASES, ids=CASE_IDS)
def test_moe_replays_are_freed(carrier, k):
    stack = _stack(carrier, k, moe=True)
    tracker = _backward_tracked(stack)
    report = _report(f"MoE {carrier} k={k}", stack, tracker)
    alive = tracker.alive()
    assert tracker.activations_at_replay_start == [0] * ROUNDS, report
    assert not [s for _, s in alive if not _is_metadata(s.dtype)], report
    assert sum(s.nbytes for _, s in alive) <= MOE_METADATA_BYTES, report
    assert max(tracker.alive_at_replay_start) <= MOE_METADATA_BYTES, report
    assert not _held_layer_inputs(stack), report
    assert not _held_dispatcher_routing(stack), report


def _run(stack):
    """Forward + backward of ``sum(y * upstream)`` plus the loop losses (MoR's router losses):
    ``(y, dx, {name: grad})``, detached."""
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(SEQ, 2, HIDDEN, generator=generator).to(device()).requires_grad_(True)
    upstream = torch.randn(SEQ, 2, HIDDEN, generator=generator).to(device())
    y = _forward(stack, x)
    loss = (y * upstream).sum() + sum(getattr(stack, "loop_losses", {}).values())
    loss.backward()
    grads = {name: p.grad for name, p in stack.named_parameters() if p.grad is not None}
    return y.detach(), x.grad, grads


@pytest.mark.usefixtures("moe_on_this_device")
@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
@pytest.mark.parametrize("carrier,k", CASES, ids=CASE_IDS)
def test_recompute_is_bitwise(monkeypatch, carrier, k, moe):
    reference = _stack(carrier, k, moe=moe, recompute=False)
    state = reference.state_dict()
    expected = _run(reference)
    recomputed = _stack(carrier, k, moe=moe)
    recomputed.load_state_dict(state)
    runs = {"loop_recompute": _run(recomputed)}
    # The same with the checkpoint early stop on at every round (the behavior before the fix).
    monkeypatch.setattr(
        loop_transformer_block, "set_checkpoint_early_stop", lambda enable: contextlib.nullcontext()
    )
    early_stop = _stack(carrier, k, moe=moe)
    early_stop.load_state_dict(state)
    runs["loop_recompute, early stop"] = _run(early_stop)

    y_ref, dx_ref, grads_ref = expected
    assert grads_ref
    for name, (y, dx, grads) in runs.items():
        assert torch.equal(y, y_ref), name
        assert torch.equal(dx, dx_ref), name
        assert grads.keys() == grads_ref.keys(), name
        for parameter, grad in grads_ref.items():
            assert torch.equal(grads[parameter], grad), (name, parameter)


@pytest.mark.skipif(not CUDA, reason="torch.cuda.memory_allocated needs CUDA")
@pytest.mark.parametrize("carrier,k", CARRIERS, ids=CARRIER_IDS)
def test_cuda_memory_after_backward(carrier, k):
    batch = 32
    gc.collect()
    torch.cuda.synchronize()
    stack = _stack(carrier, k)
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(SEQ, batch, HIDDEN, generator=generator).to(device())
    upstream = torch.randn(SEQ, batch, HIDDEN, generator=generator).to(device())
    rotary = rope()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()

    y = stack(x, attention_mask=None, rotary_pos_emb=rotary)
    (y * upstream).sum().backward()
    del y
    gc.collect()
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()

    # The caching allocator rounds every block up to 512 bytes.
    grad_bytes = sum(
        -(-param.grad.untyped_storage().nbytes() // 512) * 512
        for param in stack.parameters()
        if param.grad is not None
    )
    activation = SEQ * batch * HIDDEN * x.element_size()
    retained = after - before - grad_bytes
    assert retained < activation, (
        f"{carrier} k={k}: {retained} B allocated after the backward beyond the parameter "
        f"gradients ({grad_bytes} B); one [S, B, H] activation is {activation} B; layers "
        f"holding a norm manager {_held_layer_inputs(stack)}"
    )
