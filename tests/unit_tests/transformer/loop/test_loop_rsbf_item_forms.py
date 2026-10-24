# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Block-checkpoint and nested CORE items of the RSBF executor (M8b / M8;
``loop_rsbf_item_form='block'`` / ``'nested'``).

The block item of round ``r`` recomputes the round in blocks of ``s`` core sublayers, last
block first: each block from a fresh leaf on its input (``s_{r-1}`` for block 0, the input the
sweep stashed for later blocks), back-propagated at once, its leaf's gradient seeding the
previous block. The nested item stashes nothing in the sweep: it first rebuilds the round's
prefix ``[0, rebuild_end)`` in sweep mode (grad enabled, graph dropped per sublayer, statistics
gated) into an item-local layer stash, then runs its blocks (stride 1 by default) as the block
item does. Since a block input feeds only its own block and every core parameter belongs to
one sublayer, both items equal the whole-round item bitwise (losses, parameter gradients,
``dL/de``) at one EU in any dtype.

CPU cases (one EU, gloo; the dense core ``'*-*-'`` has L = 4 sublayers, strides 1 and 3 split
an attention from its MLP, 3 does not divide L):

* I1 the sweep's per-sublayer detach against the graph-keeping sweep (``sweep_detach``);
* I2 every block stride and every nested stride against the whole round on the dense, padded,
  MTP, RoPE and mHC models with ``check_recompute`` (every recomputed sublayer output and
  every rebuilt input equals the sweep's), ``verify_stash`` and ``poison_on_free``; block
  strides 1, 2 and L against the nested item; I3 against the eager reference (the stage-2 fp32
  gate);
* I5 statistics counted once (sweep) and replayed in the block recomputes and the rebuild;
* I6 the stash lifetimes: put by the sweep, freed right after the backward of its block (last
  block first), its bytes per kind; the nested item's layer stash (item-local, nothing in the
  store, dropped last block first); I7 FLOP records (4 forward units, plus the rebuild's
  ``rebuild_end / L`` for the nested item);
* I8 an online differentiable criterion; I9 waves and the sample-order fill;
* ``verify_stash`` catching a write the version counter cannot see; ``_stash_tensor``; the
  rebuild check naming a differing rebuilt input.

The multi-EU cases (W = 2 / 4, item moves carrying the stash, lockstep nulls per block and per
rebuild) are in ``test_loop_rsbf_multi_eu.py``; the MoE case (local MoE does not run on the CPU
venv) and the DSv4-like pattern are the GPU cases here: DSV4-TINY-like
``'HEHECEHECEHECE/WE'`` (mHC 4, MTP 1, a hash layer, alltoall at EP 1, unfused attention,
indexer loss, expert bias) in bf16, block strides 1, 2 and 4 and nested strides 1 and 2
against the whole round bitwise with ``check_recompute``, ``verify_stash`` and equal
expert-bias counts.
"""

import contextlib
import dataclasses
import re
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.forward_replay import in_forward_replay, statistics_weight
from megatron.core.transformer.loop.cells import CoreRound
from megatron.core.transformer.loop.program import (
    CoreBackwardPlan,
    LoopPhase,
    LoopProgram,
    SampleKey,
)
from megatron.core.transformer.loop.rsbf_executor import (
    RECOMPUTE_DIFFERS,
    STASH_MODIFIED,
    LoopRSBFExecutor,
)
from megatron.core.transformer.loop.state_store import LoopStateStore
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
from tests.unit_tests.transformer.loop import hybrid_loop_utils
from tests.unit_tests.transformer.loop import test_loop_rsbf_executor as base
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    CUDA,
    HIDDEN,
    SEQ,
    VOCAB,
    model_parallel_once,
)
from tests.unit_tests.transformer.loop.rsbf_test_criteria import HaltingHead, OnlineHaltingCriterion

# Fixtures used by the shared helpers, made visible to pytest in this module.
mhc_on_this_device = hybrid_loop_utils.mhc_on_this_device
cpu_mtp = hybrid_loop_utils.cpu_mtp
cpu_rope = hybrid_loop_utils.cpu_rope
hybrid = base.hybrid
deterministic_algorithms = base.deterministic_algorithms

DEPTHS = base.DEPTHS  # (1, 3, 2, 4) at R_max 4
CORE = 4  # the dense core '*-*-'
# Block strides of the dense core: 1 and 3 split an attention from its MLP; 3 does not divide.
STRIDES = (1, 2, 3)
SPLIT_PAIRS = {1: (1, 3), 2: (), 3: (3,)}
# Nested strides of the dense core and the end of their rebuild (the last block's start).
NESTED_STRIDES = (1, 2, 3)
REBUILD_END = {1: 3, 2: 2, 3: 3}

gpu = pytest.mark.skipif(not CUDA, reason="MoE / DSv4 layers need CUDA")


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        try:
            yield
        finally:
            base.get_loop_rsbf_schedule().reset()


def _block(stride):
    return dict(loop_rsbf_item_form="block", loop_rsbf_block_stride=stride)


def _nested(stride=None):
    return dict(loop_rsbf_item_form="nested", loop_rsbf_block_stride=stride)


def _assert_same_step(actual, expected):
    """Per-sample losses and every parameter gradient bitwise."""
    (reports, _, grads), (expected_reports, _, expected_grads) = actual, expected
    base._assert_losses_equal(base._lm_losses(reports), base._lm_losses(expected_reports))
    base._assert_grads_equal(grads, expected_grads)


# ---------------------------------------------------------------------------
# I1: the per-sublayer detach of the sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["whole_round", "block2", "nested1"])
@pytest.mark.parametrize("name", ["dense", "mhc_mtp"])
def test_sweep_detach_per_sublayer_changes_nothing(request, hybrid, name, form):
    """The sweep-mode forwards dropping their graph at every core sublayer (the default) and
    only at the inputs they store (``sweep_detach='block'``; for the whole round: the stage-3
    sweep that keeps the round's graph; for the nested item also its rebuild) give bitwise the
    same step."""
    overrides = {"whole_round": {}, "block2": _block(2), "nested1": _nested()}[form]
    model = base._model(name, request, **overrides)
    batches = base._batches(len(DEPTHS))
    detached = base._rsbf(model, hybrid, batches, DEPTHS)
    attached = base._rsbf(model, hybrid, batches, DEPTHS, sweep_detach="block")
    _assert_same_step(detached, attached)
    with pytest.raises(ValueError, match="unknown sweep_detach"):
        base._executor(model, DEPTHS, sweep_detach="round")


# ---------------------------------------------------------------------------
# I2 / I3 / I4: block items against the whole round and the eager reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("name", base.CASES)
def test_block_items_equal_the_whole_round(request, hybrid, name, stride, deterministic_algorithms):
    """Every stride (pair-splitting and non-dividing included) gives the whole-round item's
    per-sample losses and gradients bitwise; with check_recompute every recomputed sublayer
    output equals the sweep's, verify_stash finds every stash entry unchanged, poisoned frees
    change nothing; no stash entry needed a copy; the store ends every wave empty."""
    batches = base._batches(len(DEPTHS), padded=name.endswith("padded"))
    whole = base._rsbf(base._model(name, request), hybrid, batches, DEPTHS)
    model = base._model(name, request, **_block(stride))
    blocked = base._rsbf(
        model, hybrid, batches, DEPTHS, check_recompute=True, verify_stash=True, poison_on_free=True
    )
    _assert_same_step(blocked, whole)
    executor = blocked[1]
    program, stats = executor.program, executor.stats
    plan = program.uniform_core_plan
    assert plan == CoreBackwardPlan.strided(CORE, stride) and program.item_form == "block"
    assert program.core_parameter_overlap() == []
    assert stats.stash_copies == 0 and stats.split_pairs == SPLIT_PAIRS[stride]
    stashed = len(plan.stashed_block_inputs())
    # Every stash entry (block inputs and traces) is verified once.
    assert stats.stash_verify_syncs == (stashed + CORE) * sum(DEPTHS)
    for peak in stats.store_peaks:
        assert peak.live_entries == 0 and peak.peak_bytes_by_kind.get("block", 0) > 0
    assert executor.store.wave is None


@pytest.mark.parametrize("stride", STRIDES)
def test_block_items_equal_the_eager_reference(request, hybrid, stride, deterministic_algorithms):
    """I3: the block item against the eager model under the standard schedule (the stage-2
    gate: losses bitwise, gradients within the fp32 re-association bound)."""
    model = base._model("mhc_mtp", request, **_block(stride))
    batches = base._batches(len(DEPTHS))
    expected_losses, expected = base._eager(model, hybrid, batches, DEPTHS)
    reports, _, actual = base._rsbf(model, hybrid, batches, DEPTHS)
    base._assert_losses_equal(base._lm_losses(reports), expected_losses)
    base._assert_grads_close(
        actual, expected, base._abs_sample_grads(model, hybrid, batches, DEPTHS)
    )


# ---------------------------------------------------------------------------
# I5: statistics counted once
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stride", STRIDES)
def test_block_recomputes_replay_and_sweeps_count(request, hybrid, stride):
    """Each core sublayer runs once per sample-round in the sweep (counted, outside the replay
    gate) and once in the block recomputes (inside it); every call has grad enabled (the
    training path) and the step's statistics weight."""
    model = base._model("dense", request, **_block(stride))
    roles = model.decoder.loop_roles
    with base._layer_spy(model) as calls:
        base._rsbf(model, hybrid, base._batches(len(DEPTHS), 2), DEPTHS)
    assert statistics_weight() == 1.0 and not in_forward_replay()
    assert all(grad and weight == 0.5 for _, _, grad, weight in calls)
    total = sum(DEPTHS)
    for index in range(roles.core_start, roles.core_end):
        replays = [replay for layer, replay, _, _ in calls if layer == index]
        assert (replays.count(False), replays.count(True)) == (total, total), index


# ---------------------------------------------------------------------------
# I6 / I7: stash lifetimes, bytes and FLOPs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stride", STRIDES)
def test_stash_entries_live_from_the_sweep_to_their_block(request, hybrid, monkeypatch, stride):
    """Every block input is put by its round's sweep and freed by its CORE item right after
    the backward of its block, last block first; with the sample-order fill every round is
    swept before the first item, so the stash peaks at (n_b - 1) states per sample-round."""
    model = base._model("dense", request, **_block(stride))
    plan = CoreBackwardPlan.strided(CORE, stride)
    events = []
    real_put, real_free = LoopStateStore.put_block_input, LoopStateStore.free_block_input

    def put(store, sample, r, sublayer, x, *, kind="block"):
        events.append(("put", kind, sample, r, sublayer))
        return real_put(store, sample, r, sublayer, x, kind=kind)

    def free(store, sample, r, sublayer, *, kind="block"):
        events.append(("free", kind, sample, r, sublayer))
        return real_free(store, sample, r, sublayer, kind=kind)

    monkeypatch.setattr(LoopStateStore, "put_block_input", put)
    monkeypatch.setattr(LoopStateStore, "free_block_input", free)
    _, executor, _ = base._rsbf(
        model, hybrid, base._batches(len(DEPTHS)), DEPTHS, fill="sample_order"
    )
    starts = plan.stashed_block_inputs()
    rounds = [(g, r) for g, depth in enumerate(DEPTHS) for r in range(1, depth + 1)]
    puts = [event[2:] for event in events if event[0] == "put"]
    frees = [event[2:] for event in events if event[0] == "free"]
    assert sorted(puts) == sorted((g, r, l) for g, r in rounds for l in starts)
    assert sorted(frees) == sorted(puts)
    # Within each item: the later blocks' inputs are freed first.
    for g, r in rounds:
        assert [l for gg, rr, l in frees if (gg, rr) == (g, r)] == sorted(starts, reverse=True)
    # Sample order: every put precedes every free.
    assert max(i for i, e in enumerate(events) if e[0] == "put") < min(
        i for i, e in enumerate(events) if e[0] == "free"
    )
    state = SEQ * HIDDEN * 4  # [S, 1, H] fp32
    (peak,) = executor.stats.store_peaks
    assert peak.peak_bytes_by_kind["block"] == len(starts) * sum(DEPTHS) * state
    assert peak.live_bytes_by_kind == {}


@pytest.mark.parametrize("stride", STRIDES)
def test_block_item_flops_equal_the_whole_round(request, hybrid, stride):
    """I7: a block item recomputes the round once, like the whole round (4 forward units per
    sample-round): the same work records and loop FLOP reports."""
    batches = base._batches(len(DEPTHS))
    whole_reports, whole, _ = base._rsbf(base._model("mtp", request), hybrid, batches, DEPTHS)
    model = base._model("mtp", request, **_block(stride))
    reports, executor, _ = base._rsbf(model, hybrid, batches, DEPTHS)
    assert executor.program.uniform_core_plan.forward_units() == 4.0
    assert executor.stats.work == whole.stats.work
    for report, expected in zip(reports, whole_reports):
        for key in base.rsbf_executor._LOOP_REPORT_KEYS:  # pylint: disable=protected-access
            assert torch.equal(report[key], expected[key]), key


# ---------------------------------------------------------------------------
# I8 / I9: an online criterion, waves and fills
# ---------------------------------------------------------------------------


def _online(model):
    head = HaltingHead(
        model.decoder.loop_num_streams, model.config.hidden_size, device=base.device(), seed=3
    )
    return OnlineHaltingCriterion(head, tau=1.4, carry=0.05)


@pytest.mark.parametrize("stride", (1, 3))
def test_block_items_with_an_online_criterion(request, hybrid, stride):
    """I8: an online differentiable criterion (aux loss, carried criterion state, gradients
    into ``init_state``): the block item gives the whole round's depths, losses and gradients
    (model and halting head) bitwise; the criterion seeds enter the last block only."""
    batches = base._batches(len(DEPTHS))
    results = []
    for overrides in ({}, _block(stride)):
        model = base._model("dense", request, **overrides)
        criterion = _online(model)
        reports, executor, grads = base._rsbf(
            model, hybrid, batches, criterion=criterion, extra_modules=(criterion.head,)
        )
        depths = [executor.stats.depths[SampleKey(i, 0)] for i in range(len(DEPTHS))]
        results.append((reports, executor, grads, depths))
    assert results[0][3] == results[1][3] and len(set(results[0][3])) > 1
    _assert_same_step(results[1][:3], results[0][:3])
    assert "1.w" in results[0][2] and "1.v" in results[0][2]


@pytest.mark.parametrize("stride", (1, 3))
def test_block_items_with_waves_and_the_sample_order_fill(request, hybrid, stride):
    """I9: two waves and the sample-order fill: the block item equals the whole round under
    the same schedule bitwise."""
    batches = base._batches(8)
    depths = DEPTHS + (2, 1, 3, 4)
    options = dict(waves=2, fill="sample_order")
    whole = base._rsbf(base._model("mhc", request), hybrid, batches, depths, **options)
    model = base._model("mhc", request, **_block(stride))
    blocked = base._rsbf(model, hybrid, batches, depths, check_recompute=True, **options)
    _assert_same_step(blocked, whole)
    assert len(blocked[1].stats.store_peaks) == 2


# ---------------------------------------------------------------------------
# The stash's own checks
# ---------------------------------------------------------------------------


def test_verify_stash_catches_a_write_the_version_counter_misses(request, hybrid, monkeypatch):
    """A write into a stashed activation that bypasses autograd (``.data``, as a kernel writing
    through a raw pointer would) passes the store's version check; ``verify_stash`` compares
    the entry with its capture-time clone and raises naming the entry."""
    model = base._model("dense", request, **_block(2))
    real = LoopRSBFExecutor._stash_put

    def corrupting_put(executor, kind, g, r, sublayer, tensor):
        real(executor, kind, g, r, sublayer, tensor)
        if kind == "block":
            tensor.data.add_(1.0)

    monkeypatch.setattr(LoopRSBFExecutor, "_stash_put", corrupting_put)
    with pytest.raises(RuntimeError, match=rf"{STASH_MODIFIED}: x\[sample \d+, round \d+"):
        base._rsbf(model, hybrid, base._batches(len(DEPTHS)), DEPTHS, verify_stash=True)


def test_an_in_place_write_during_the_sweep_is_caught(request, hybrid):
    """A core sublayer writing into its (detached) input in place during the sweep would
    corrupt a captured block input after its version was read: the sweep checks every
    captured tensor's version counter before stashing it."""
    model = base._model("dense", request, **_block(2))
    layer = model.decoder.layers[model.decoder.loop_roles.core_start + 2]

    def write_input(module, args, kwargs, output):
        del module, output
        hidden = args[0] if args else kwargs["hidden_states"]
        if not in_forward_replay():
            hidden.mul_(1.0)

    handle = layer.register_forward_hook(write_input, with_kwargs=True)
    try:
        with pytest.raises(RuntimeError, match=f"{STASH_MODIFIED}.*during the sweep"):
            base._rsbf(model, hybrid, base._batches(len(DEPTHS)), DEPTHS)
    finally:
        handle.remove()


def test_stash_tensor_aliases_owning_tensors_and_copies_views(request):
    """``_stash_tensor`` keeps a contiguous tensor that owns its whole storage (no copy) and
    copies a view, a non-contiguous tensor or an offset slice (``stash_copies``)."""
    model = base._model("dense", request, **_block(2))
    program = LoopProgram(model)
    executor = LoopRSBFExecutor(program, pg_collection=SimpleNamespace(dp=None))
    owning = torch.randn(SEQ, 1, HIDDEN)
    assert executor._stash_tensor(owning) is owning  # pylint: disable=protected-access
    wide = torch.randn(SEQ, 2, HIDDEN)
    for view in (wide[:, :1], owning.transpose(0, 2).transpose(0, 2)[1:], wide[:, 1:]):
        copy = executor._stash_tensor(view)  # pylint: disable=protected-access
        assert copy is not view and torch.equal(copy, view) and copy.is_contiguous()
        assert copy.untyped_storage().nbytes() == copy.numel() * copy.element_size()
    assert executor.stats.stash_copies == 3


# ---------------------------------------------------------------------------
# The nested item (M8): rebuild, then per-block recompute + backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stride", NESTED_STRIDES)
@pytest.mark.parametrize("name", base.CASES)
def test_nested_items_equal_the_whole_round(
    request, hybrid, name, stride, deterministic_algorithms
):
    """I2: every nested stride (per sublayer, per attention + MLP pair, non-dividing) gives the
    whole-round item's per-sample losses and gradients bitwise; with check_recompute every
    rebuilt input and every recomputed sublayer output equals the sweep's; poisoned frees
    (store and layer stash) change nothing; the rebuild runs once per CORE item and holds
    the inputs of every later block; nothing but check_recompute's traces enters the store."""
    batches = base._batches(len(DEPTHS), padded=name.endswith("padded"))
    whole = base._rsbf(base._model(name, request), hybrid, batches, DEPTHS)
    model = base._model(name, request, **_nested(stride))
    nested = base._rsbf(
        model, hybrid, batches, DEPTHS, check_recompute=True, verify_stash=True, poison_on_free=True
    )
    _assert_same_step(nested, whole)
    executor = nested[1]
    program, stats = executor.program, executor.stats
    plan = program.uniform_core_plan
    assert plan == CoreBackwardPlan.nested(CORE, stride) and program.item_form == "nested"
    assert plan.rebuild_end == REBUILD_END[stride]
    assert program.core_parameter_overlap() == []
    assert stats.rebuilds == sum(DEPTHS)
    assert stats.layer_stash_peak == len(plan.blocks) - 1
    assert stats.split_pairs == SPLIT_PAIRS[stride] and stats.stash_copies == 0
    # The store holds no block inputs; each trace entry is verified once (when taken).
    assert stats.stash_verify_syncs == CORE * sum(DEPTHS)
    for peak in stats.store_peaks:
        assert peak.live_entries == 0 and peak.peak_bytes_by_kind.get("block", 0) == 0
    assert executor.store.wave is None


@pytest.mark.parametrize("stride", (1, 2, CORE))
@pytest.mark.parametrize("name", ["dense_padded", "mhc_mtp"])
def test_block_items_equal_the_nested_item(request, hybrid, name, stride, deterministic_algorithms):
    """The M8b gate: block items of stride 1, 2 and L (the whole round) against the nested
    item (per sublayer): per-sample losses and gradients bitwise."""
    batches = base._batches(len(DEPTHS), padded=name.endswith("padded"))
    nested = base._rsbf(base._model(name, request, **_nested()), hybrid, batches, DEPTHS)
    model = base._model(name, request, **_block(stride))
    blocked = base._rsbf(model, hybrid, batches, DEPTHS, check_recompute=True)
    _assert_same_step(blocked, nested)


@pytest.mark.parametrize("stride", (1, 2))
def test_nested_items_equal_the_eager_reference(request, hybrid, stride, deterministic_algorithms):
    """I3: the nested item against the eager model under the standard schedule (the stage-2
    gate: losses bitwise, gradients within the fp32 re-association bound)."""
    model = base._model("mhc_mtp", request, **_nested(stride))
    batches = base._batches(len(DEPTHS))
    expected_losses, expected = base._eager(model, hybrid, batches, DEPTHS)
    reports, _, actual = base._rsbf(model, hybrid, batches, DEPTHS)
    base._assert_losses_equal(base._lm_losses(reports), expected_losses)
    base._assert_grads_close(
        actual, expected, base._abs_sample_grads(model, hybrid, batches, DEPTHS)
    )


@pytest.mark.parametrize("stride", NESTED_STRIDES)
def test_nested_rebuilds_and_recomputes_replay_and_sweeps_count(request, hybrid, stride):
    """I5: each core sublayer runs once per sample-round in the sweep (counted, outside the
    replay gate); in the CORE item the rebuild runs sublayers ``[0, rebuild_end)`` and the
    block recomputes every sublayer, all inside the gate; every call has grad enabled (the
    training path, D35) and the step's statistics weight."""
    model = base._model("dense", request, **_nested(stride))
    roles = model.decoder.loop_roles
    with base._layer_spy(model) as calls:
        base._rsbf(model, hybrid, base._batches(len(DEPTHS), 2), DEPTHS)
    assert statistics_weight() == 1.0 and not in_forward_replay()
    assert all(grad and weight == 0.5 for _, _, grad, weight in calls)
    total = sum(DEPTHS)
    for index in range(roles.core_start, roles.core_end):
        rebuilt = index - roles.core_start < REBUILD_END[stride]
        replays = [replay for layer, replay, _, _ in calls if layer == index]
        expected = (total, 2 * total if rebuilt else total)
        assert (replays.count(False), replays.count(True)) == expected, index


@pytest.mark.parametrize("stride", NESTED_STRIDES)
def test_nested_layer_stash_lives_inside_its_item(request, hybrid, monkeypatch, stride):
    """I6: the nested item puts nothing into the state store; its rebuild (one per CORE item,
    inside the replay gate, up to the last block's start, without the epilogue) captures the
    inputs of the later blocks into the item-local layer stash, and the item drops each entry
    right after its block's backward, last block first."""
    model = base._model("dense", request, **_nested(stride))
    later = CoreBackwardPlan.nested(CORE, stride).rebuilt_layer_inputs()
    rebuilds, drops, puts = [], [], []
    real_forward = CoreRound.forward_detached
    real_drop = LoopRSBFExecutor._drop_layer_input
    real_put = LoopStateStore.put_block_input

    def forward_detached(core, *args, **kwargs):
        out = real_forward(core, *args, **kwargs)
        if kwargs["phase"] == LoopPhase.REBUILD:
            rebuilds.append(
                (
                    kwargs["r"],
                    kwargs["end"],
                    tuple(kwargs["capture"]),
                    tuple(sorted(out.captured)),
                    out.outputs,
                    in_forward_replay(),
                    torch.is_grad_enabled(),
                )
            )
        return out

    def drop(executor, layer_stash, sublayer):
        drops.append((sublayer, len(layer_stash)))
        return real_drop(executor, layer_stash, sublayer)

    def put(store, *args, **kwargs):
        puts.append(args[:3])
        return real_put(store, *args, **kwargs)

    monkeypatch.setattr(CoreRound, "forward_detached", forward_detached)
    monkeypatch.setattr(LoopRSBFExecutor, "_drop_layer_input", drop)
    monkeypatch.setattr(LoopStateStore, "put_block_input", put)
    _, executor, _ = base._rsbf(
        model, hybrid, base._batches(len(DEPTHS)), DEPTHS, fill="sample_order"
    )
    rounds = sorted(r for depth in DEPTHS for r in range(1, depth + 1))
    assert sorted(entry[0] for entry in rebuilds) == rounds
    for _, end, capture, captured, outputs, replay, grad in rebuilds:
        assert (end, capture, captured) == (REBUILD_END[stride], later, later)
        assert outputs is None and replay and grad
    one_item = [(sublayer, len(later) - i) for i, sublayer in enumerate(reversed(later))]
    assert drops == one_item * sum(DEPTHS)
    assert puts == []
    stats = executor.stats
    assert stats.rebuilds == sum(DEPTHS) and stats.layer_stash_peak == len(later)
    (peak,) = stats.store_peaks
    assert peak.live_entries == 0 and peak.peak_bytes_by_kind.get("block", 0) == 0


@pytest.mark.parametrize("stride", NESTED_STRIDES)
def test_nested_item_flops_add_the_rebuild(request, hybrid, stride):
    """I7: the nested item's work records are the whole round's except that each CORE item's
    recompute adds its rebuild (the adapter plus core sublayers ``[0, rebuild_end)``): ``4 +
    rebuild_end / L`` forward units per sample-round. The loop FLOP reports follow."""
    batches = base._batches(len(DEPTHS))
    whole_reports, whole, _ = base._rsbf(base._model("mtp", request), hybrid, batches, DEPTHS)
    model = base._model("mtp", request, **_nested(stride))
    reports, executor, _ = base._rsbf(model, hybrid, batches, DEPTHS)
    end = REBUILD_END[stride]
    assert executor.program.uniform_core_plan.forward_units() == 4.0 + end / CORE
    stack, roles = model.decoder, model.decoder.loop_roles
    rebuild = float(
        stack.range_forward_flops(
            roles.core_start,
            roles.core_start + end,
            SEQ,
            SEQ * SEQ,
            SEQ,
            extra=stack.round_extra_forward_flops(SEQ),
        )
    )
    assert 0.0 < rebuild
    work, whole_work = executor.stats.work, whole.stats.work
    assert len(work) == len(whole_work)
    for record, expected in zip(work, whole_work):
        if record.kind == "core_item":
            assert record.recompute == expected.recompute + rebuild
            assert dataclasses.replace(record, recompute=expected.recompute) == expected
        else:
            assert record == expected
    recompute_key = "loop recompute flops per token"
    for report, expected, depth in zip(reports, whole_reports, DEPTHS):
        for key in base.rsbf_executor._LOOP_REPORT_KEYS:  # pylint: disable=protected-access
            if key != recompute_key:
                assert torch.equal(report[key], expected[key]), key
        assert report[recompute_key][1] == expected[recompute_key][1]
        assert report[recompute_key][0].item() == pytest.approx(
            expected[recompute_key][0].item() + depth * rebuild, rel=1e-6
        )


@pytest.mark.parametrize("stride", (1, 3))
def test_nested_items_with_an_online_criterion(request, hybrid, stride):
    """I8: an online differentiable criterion: the nested item gives the whole round's depths,
    losses and gradients (model and halting head) bitwise; the rebuild runs no epilogue, so
    the criterion's seeds enter the last block only."""
    batches = base._batches(len(DEPTHS))
    results = []
    for overrides in ({}, _nested(stride)):
        model = base._model("dense", request, **overrides)
        criterion = _online(model)
        reports, executor, grads = base._rsbf(
            model, hybrid, batches, criterion=criterion, extra_modules=(criterion.head,)
        )
        depths = [executor.stats.depths[SampleKey(i, 0)] for i in range(len(DEPTHS))]
        results.append((reports, executor, grads, depths))
    assert results[0][3] == results[1][3] and len(set(results[0][3])) > 1
    _assert_same_step(results[1][:3], results[0][:3])
    assert results[1][1].stats.rebuilds == sum(results[1][3])


@pytest.mark.parametrize("stride", (1, 3))
def test_nested_items_with_waves_and_the_sample_order_fill(request, hybrid, stride):
    """I9: two waves and the sample-order fill: the nested item equals the whole round under
    the same schedule bitwise."""
    batches = base._batches(8)
    depths = DEPTHS + (2, 1, 3, 4)
    options = dict(waves=2, fill="sample_order")
    whole = base._rsbf(base._model("mhc", request), hybrid, batches, depths, **options)
    model = base._model("mhc", request, **_nested(stride))
    nested = base._rsbf(model, hybrid, batches, depths, check_recompute=True, **options)
    _assert_same_step(nested, whole)
    assert len(nested[1].stats.store_peaks) == 2


def test_the_rebuild_check_names_a_differing_rebuilt_input(request, hybrid, monkeypatch):
    """check_recompute compares every input the nested rebuild stores with the sweep's
    output of the sublayer before it: a rebuilt input that differs (written through
    ``.data``: invisible to autograd and to the version counter) is reported under the
    rebuild's name (``"report"``, the other rebuilt inputs at zero) and raised under that name
    first (``"raise"``)."""
    model = base._model("dense", request, **_nested())
    real = CoreRound.forward_detached

    def corrupting(core, *args, **kwargs):
        out = real(core, *args, **kwargs)
        if kwargs["phase"] == LoopPhase.REBUILD:
            out.captured[2].data.add_(1.0)
        return out

    monkeypatch.setattr(CoreRound, "forward_detached", corrupting)
    batches = base._batches(len(DEPTHS))
    _, executor, _ = base._rsbf(model, hybrid, batches, DEPTHS, check_recompute="report")
    rebuilt = {
        name: value
        for name, value in executor.stats.recompute_diffs.items()
        if name.startswith("rebuild[")
    }
    assert len(rebuilt) == (CORE - 1) * sum(DEPTHS)  # inputs of sublayers 1, 2 and 3
    for name, value in rebuilt.items():
        assert (value > 0.5) == name.endswith("input of sublayer 2]"), (name, value)
    prefix = re.escape(RECOMPUTE_DIFFERS.split("{what}", maxsplit=1)[0])
    first = prefix + r"rebuild\[sample \d+, round \d+, input of sublayer 2\]"
    with pytest.raises(RuntimeError, match=first):
        base._rsbf(model, hybrid, batches, DEPTHS, check_recompute=True)


# ---------------------------------------------------------------------------
# GPU: the DSv4-like pattern (MoE, mHC, MTP, hash layer, indexer loss)
# ---------------------------------------------------------------------------

# DSV4-TINY-like: prelude 'HE', core 'HECEHECE' (L = 8), coda 'HECE', MTP 'WE'.
DSV4_PATTERN = "HEHECEHECEHECE/WE"
_RATIOS = {"W": 0, "C": 4, "H": 128, "E": 0}


def _dsv4_tiny(dtype, **overrides):
    """The DSV4-TINY-like loop model (the fields of ``test_loop_rsbf_executor._dsv4_model``
    on the 14-sublayer pattern), deterministic unfused attention, alltoall MoE at EP 1."""
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_dsv4_stack_spec
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    main, mtp = DSV4_PATTERN.split("/")
    fields = dict(
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
        dsa_indexer_loss_coeff=0.01,
        csa_window_size=32,
        csa_compress_ratios=[_RATIOS[symbol] for symbol in main + mtp],
        num_moe_experts=8,
        moe_ffn_hidden_size=256,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=1e-4,
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
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
        use_cpu_initialization=True,
        loop_architecture="huginn",
        loop_iterations=2,
        loop_prelude_sublayers=2,
        loop_coda_sublayers=4,
        deterministic_mode=True,
        attention_backend=AttnBackend.unfused,
        **base.RSBF,
    )
    fields.update(overrides)
    config = MLATransformerConfig(**fields)
    torch.manual_seed(1234)
    return HybridModel(
        config=config,
        hybrid_stack_spec=loop_hybrid_dsv4_stack_spec(config),
        vocab_size=VOCAB,
        max_sequence_length=1024,
        hybrid_layer_pattern=DSV4_PATTERN,
        position_embedding_type="rope",
    ).cuda()


def _reset_expert_bias_counts(model):
    for module in model.modules():
        if getattr(module, "local_tokens_per_expert", None) is not None:
            module.local_tokens_per_expert.zero_()


# Item forms of the DSV4-TINY-like core (L = 8): the block strides, the nested strides.
_DSV4_FORMS = {
    "block1": _block(1),
    "block2": _block(2),
    "block4": _block(4),
    "nested1": _nested(1),
    "nested2": _nested(2),
}


@gpu
def test_dsv4_item_forms_equal_the_whole_round(hybrid, deterministic_algorithms):
    """DSV4-TINY-like at one EU, prescribed depths (2, 1, 2, 2) at R_max 2: block strides 1, 2
    and 4 and nested strides 1 and 2 (2 = one attention + MoE pair per block; 1 splits every
    pair) give the whole round's per-sample losses and gradients bitwise, every rebuilt input
    and every recomputed sublayer output equals the sweep's (check_recompute), every stash
    entry is unchanged when read (verify_stash), and the expert-bias token counts are equal
    (one count per sample-round: the rebuilds and the recomputes replay)."""
    depths = (2, 1, 2, 2)
    batches = base._batches(len(depths), seq=256, dev=torch.device("cuda"))
    results = {}
    with base._hadamard_fallback():
        for form in (None,) + tuple(_DSV4_FORMS):
            overrides = {} if form is None else _DSV4_FORMS[form]
            model = _dsv4_tiny(torch.bfloat16, **overrides)
            assert model.decoder.loop_roles.core == 8
            set_moe_metrics_tracker(MoEMetricsTracker())
            _reset_expert_bias_counts(model)
            check = {} if form is None else dict(check_recompute=True, verify_stash=True)
            reports, executor, grads = base._rsbf(model, hybrid, batches, depths, **check)
            results[form] = (
                base._lm_losses(reports),
                grads,
                base._expert_bias_counts(model),
                executor.stats,
            )
            del model
    losses, grads, counts, _ = results[None]
    assert counts
    for form in _DSV4_FORMS:
        actual_losses, actual_grads, actual_counts, stats = results[form]
        base._assert_losses_equal(actual_losses, losses)
        base._assert_grads_equal(actual_grads, grads)
        assert actual_counts.keys() == counts.keys()
        for name, value in counts.items():
            assert torch.equal(actual_counts[name], value), (form, name)
        assert stats.stash_copies == 0 and stats.debug_syncs > 0
        assert stats.split_pairs == ((1, 3, 5, 7) if form.endswith("1") else ()), form
        if form.startswith("nested"):
            # nested(8, 1): 8 blocks, 7 rebuilt inputs; nested(8, 2): 4 blocks, 3.
            assert stats.rebuilds == sum(depths), form
            assert stats.layer_stash_peak == {"nested1": 7, "nested2": 3}[form]


@gpu
@pytest.mark.parametrize("form", [None, "block2"])
def test_dsv4_coda_up_proj_recompute_changes_no_value(hybrid, deterministic_algorithms, form):
    """D42 (``loop_rsbf_coda_recompute_mla_up_proj``): the CODA item checkpoints the up
    projections of the coda's 'H' / 'C' attentions and the MTP's 'W' attention (one
    checkpoint each per CODA item, none elsewhere); per-sample losses, every parameter
    gradient and the expert-bias counts equal the step without the request bitwise; the
    CODA work records add the replayed up projections as recompute FLOPs."""
    from megatron.core import tensor_parallel
    from megatron.core.transformer.loop import flops

    depths = (2, 1, 2, 2)
    batches = base._batches(len(depths), seq=256, dev=torch.device("cuda"))
    overrides = {} if form is None else _DSV4_FORMS[form]
    real = tensor_parallel.CheckpointWithoutOutput.checkpoint
    results = {}
    with base._hadamard_fallback():
        for request_recompute in (False, True):
            model = _dsv4_tiny(
                torch.bfloat16, loop_rsbf_coda_recompute_mla_up_proj=request_recompute, **overrides
            )
            set_moe_metrics_tracker(MoEMetricsTracker())
            _reset_expert_bias_counts(model)
            calls = []

            def counting(checkpoint, run_function, *args, calls=calls):
                calls.append(run_function.__name__)
                return real(checkpoint, run_function, *args)

            tensor_parallel.CheckpointWithoutOutput.checkpoint = counting
            try:
                reports, executor, grads = base._rsbf(model, hybrid, batches, depths)
            finally:
                tensor_parallel.CheckpointWithoutOutput.checkpoint = real
            results[request_recompute] = (
                base._lm_losses(reports),
                grads,
                base._expert_bias_counts(model),
                executor.stats.work,
                calls.count("qkv_up_proj_and_rope_apply"),
                [m._dsv4_compress_ratio for m in executor.program.coda_recompute_attentions()],
                sum(
                    flops.dsv4_up_proj_forward_flops(m.config, 256)
                    for m in executor.program.coda_recompute_attentions()
                ),
            )
            del model, executor
    losses, grads, counts, work, checkpoints, ratios, _ = results[False]
    actual_losses, actual_grads, actual_counts, actual_work, actual_checkpoints, recomputed, up = (
        results[True]
    )
    assert checkpoints == 0 and ratios == []
    # Coda 'HECE': the 'H' and 'C' attentions; MTP 'WE': the 'W' attention.
    assert recomputed == [128, 4, 0]
    assert actual_checkpoints == len(recomputed) * len(depths)
    base._assert_losses_equal(actual_losses, losses)
    base._assert_grads_equal(actual_grads, grads)
    for name, value in counts.items():
        assert torch.equal(actual_counts[name], value), name
    assert up > 0
    for record, expected in zip(actual_work, work, strict=True):
        if record.kind == "coda":
            assert record.recompute == pytest.approx(up, rel=1e-12)
            assert dataclasses.replace(record, recompute=0.0) == expected
        else:
            assert record == expected
