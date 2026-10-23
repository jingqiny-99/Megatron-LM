# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Multi-EU RSBF executor tests on CPU: W = 2, 4 and 8 EUs against one EU (dense models).

Each case runs in a spawned Gloo world of W CPU processes (``multi_eu_utils``; local gates,
skipped under torchrun and on CUDA hosts, where the GPU multi-EU cases of
``test_loop_rsbf_ddp.py`` run instead). Model parallelism is initialized once per worker
(TP = PP = 1), so the data-parallel group is the world: the RSBF domain. Every worker builds
the same model (seeded) and computes the reference itself: the one-EU executor (stage 2,
gated against the eager model) on the whole global batch, with the same per-sample depths.
Every arm runs each rank's slice (the pretraining sampler's: microbatch ``j`` of rank ``q`` is
global microbatch ``j W + q``) on W EUs and must give

* per-sample losses bitwise equal to the reference (``c = 1``: every cell call is batch-1 local
  compute, and moves copy bits),
* the reference's parameter gradients after summing ``.grad`` over the ranks, at rtol 1e-5
  with an absolute floor of 1e-6 x max sum_i |g_i| (the per-sample gradients' absolute sum:
  the fp32 bound of re-associating the per-item addends),
* every rank's store empty at each wave end, zero core work after exit (core sublayer calls
  summed over the ranks = 2 sum_g d_g), one host sync and no status exchange per step for
  known depths, and the reports / normalizer contract.

Cases: M1 / M2 (models x arms: rebalance on / off, p2p / all_to_all, tail item moves on / off,
drain / sample_order, MBS 2, shared / per-row attention masks, a sparse depth set with idle
EUs, a second step on the same executor, a checkpoint received back by an EU it left by a
move), M3 (waves), M4 (an online criterion with the status exchange; drain / p2p and
sample_order / all_to_all, one pass and waves), M5 (buggy plans raise on every rank, also
for an exited sample moved before its extra sweep), M7 (train_step's reduction and
finalize's token count), M9 (check_recompute, poison_on_free, check_plan), M8 (lockstep MoE
with a stub 'E' module whose every call is a collective over its EP group: null items replay
each sub-round type's calls, at EP = W and EP = 2 inside the domain), and the static-budget
over-budget flags (reduced over the domain: every rank raises, or the first rank warns).
"""

import os
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.transformer.loop.dispatcher import RSBFGroups
from megatron.core.transformer.loop.null_items import register_lockstep_module
from megatron.core.transformer.loop.program import LoopProgram, PrescribedDepthCriterion
from megatron.core.transformer.loop.rsbf_executor import (
    OVER_BUDGET,
    PLAN_MISMATCH,
    SWEEP_AFTER_EXIT,
    LoopMicrobatch,
    LoopRSBFExecutor,
)
from megatron.core.transformer.loop.rsbf_plan import SubRound, SubRoundType
from megatron.core.transformer.loop.state_store import LoopStateStore
from tests.unit_tests.transformer.loop.hybrid_loop_utils import SEQ, VOCAB, build_local_loop_model
from tests.unit_tests.transformer.loop.multi_eu_utils import (
    SPAWN_SKIP_REASON,
    RecordingLockstepMLP,
    StubLockstepAdapter,
    StubLog,
    cpu_worker_setup,
    cpu_worker_teardown,
    run_spawned,
    spawn_tests_enabled,
    stub_moe_layer_spec,
)
from tests.unit_tests.transformer.loop.rsbf_test_criteria import (
    ConstantExitCriterion,
    HaltingHead,
    OnlineHaltingCriterion,
    PaddingDepthCriterion,
    retire_early,
    retire_late,
)

pytestmark = pytest.mark.skipif(
    not spawn_tests_enabled() or torch.cuda.is_available(),
    reason=SPAWN_SKIP_REASON + "; CPU hosts only",
)

DENSE = "*-*-*-*-"  # prelude '*-', core '*-*-', coda '*-'
MHC = dict(enable_mhc_connections=True, mhc_num_residual_streams=4, mhc_sinkhorn_iterations=5)
R_MAX = 4
RSBF = dict(loop_schedule="rsbf", calculate_per_token_loss=True, loop_iterations=R_MAX)
TIMEOUT = 900.0


def _needs_dynamo_off(model_name):
    if "mhc" in model_name and os.environ.get("TORCHDYNAMO_DISABLE") != "1":
        pytest.skip("mHC on CPU needs TORCHDYNAMO_DISABLE=1 (its fused ops use torch.compile)")


# ---------------------------------------------------------------------------
# Worker-side helpers (run inside the spawned processes)
# ---------------------------------------------------------------------------


def _model(name):
    pattern = DENSE + ("/*-" if "mtp" in name else "")
    overrides = dict(MHC) if "mhc" in name else {}
    return build_local_loop_model(pattern, **RSBF, **overrides)


def _loss_func(loss_mask, output):
    """``pretrain_hybrid.loss_func``'s default branch (per-token sum, token count, report)."""
    losses = output.view(-1).float()
    mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * mask)
    num_tokens = mask.sum().clone().detach().to(torch.int)
    return (
        loss,
        num_tokens,
        {"lm loss": torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])},
    )


def _depths(kind, total, world, mbs):
    """Per-sample depths by global index. ``skew``: EU 0's samples run R_max rounds, the
    others 1 or 2 (moves at rounds 2 and 3, idle EUs in the tail); ``sparse``: one sample runs
    R_max rounds, every other exits at round 1 (rounds with fewer active samples than EUs);
    ``mixed``: a fixed pseudo-random spread; ``rearrive`` (W = 4, 2 per EU, ``sample_order``):
    sample 1 moves EU 1 -> EU 0 before round 2 and its CORE(1, 2) item moves back to EU 1
    with ``s_1``, a checkpoint that left EU 1 by a move and is received there again."""
    if kind == "rearrive":
        assert (total, world, mbs) == (8, 4, 1), (total, world, mbs)
        return dict(enumerate([1, 3, 3, 1, 1, 2, 2, 1]))
    depths = {}
    for g in range(total):
        rank, j = (g // mbs) % world, g // (mbs * world)
        if kind == "skew":
            depths[g] = R_MAX if rank == 0 else 1 + j % 2
        elif kind == "sparse":
            depths[g] = R_MAX if g == 1 else 1
        else:
            depths[g] = 1 + (7 * g + 3) % R_MAX
    return depths


def _rows(total, *, padded=False, mask=None, seed=5):
    """``mask``: None, ``"shared"`` (one ``[1, 1, S, S]`` mask for every row) or ``"rows"``
    (a ``[total, 1, S, S]`` mask per row); both are causal plus random masked keys (never the
    diagonal), so a moved sample computes the reference's loss only with its own mask."""
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, VOCAB, (total, SEQ), generator=generator)
    labels = torch.randint(0, VOCAB, (total, SEQ), generator=generator)
    loss_mask = (torch.rand(total, SEQ, generator=generator) > 0.2).float()
    padding = None
    if padded:
        padding = torch.zeros(total, SEQ, dtype=torch.bool)
        padding[1::2, -(SEQ // 4) :] = True
        loss_mask = loss_mask * (~padding).float()
    attention_mask = None
    if mask is not None:
        count = 1 if mask == "shared" else total
        causal = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
        extra = torch.rand(count, 1, SEQ, SEQ, generator=generator) > 0.7
        attention_mask = (causal | extra) & ~torch.eye(SEQ, dtype=torch.bool)
    return dict(
        tokens=tokens,
        labels=labels,
        position_ids=torch.arange(SEQ).repeat(total, 1),
        loss_mask=loss_mask,
        padding_mask=padding,
        attention_mask=attention_mask,
    )


def _microbatches(rows, *, mbs, world, rank, loss_funcs=None):
    """Rank ``rank``'s microbatches of the global batch (global microbatch ``j W + q``)."""
    total = rows["tokens"].shape[0]
    count = total // mbs
    mine = [k for k in range(count) if k % world == rank]
    batches = []
    for index, k in enumerate(mine):
        sl = slice(k * mbs, (k + 1) * mbs)
        loss_func = _loss_func if loss_funcs is None else loss_funcs[index]
        mask = rows.get("attention_mask")
        if mask is not None and mask.shape[0] == total:
            mask = mask[sl]
        batches.append(
            LoopMicrobatch(
                tokens=rows["tokens"][sl],
                position_ids=rows["position_ids"][sl],
                labels=rows["labels"][sl],
                loss_mask=rows["loss_mask"][sl],
                attention_mask=mask,
                padding_mask=None if rows["padding_mask"] is None else rows["padding_mask"][sl],
                loss_func=loss_func,
            )
        )
    return batches


def _next(iterator, model):
    del model
    return next(iterator)


def _grads(*modules):
    grads = {}
    for index, module in enumerate(modules):
        for name, param in module.named_parameters():
            if param.grad is not None:
                grads[f"{index}.{name}"] = param.grad.detach().clone()
        module.zero_grad(set_to_none=True)
    return grads


def _step(model, microbatches, criterion, *, mbs, domain=None, status=None, extra=(), **kwargs):
    """One RSBF step; returns (reports, executor, grads). ``domain`` None: one EU."""
    program = LoopProgram(model, exit_criterion=criterion, check_ports=True)
    groups = None if domain is None else RSBFGroups(domain=domain, status=status)
    executor = LoopRSBFExecutor(
        program, pg_collection=SimpleNamespace(dp=domain), groups=groups, **kwargs
    )
    model.zero_grad(set_to_none=True)
    for module in extra:
        module.zero_grad(set_to_none=True)
    reports = executor.run_step(
        _next, iter(microbatches), len(microbatches), model=model, micro_batch_size=mbs
    )
    return reports, executor, _grads(model, *extra)


def _losses(executor, reports):
    """{global index: 'lm loss' report} of this rank's samples."""
    keys = [record.key.global_index for record in executor.stats.loss_records]
    assert keys == sorted(keys) and len(keys) == len(reports)
    return {key: report["lm loss"] for key, report in zip(keys, reports)}


def _abs_sample_grads(model, rows, depths, *, extra=(), criterion_factory=None):
    """sum_i |g_i| over the samples' separate one-EU gradients (the floor's scale)."""
    total = {}
    count = rows["tokens"].shape[0]
    for g in range(count):
        one = {
            name: value[g : g + 1] if value is not None and value.shape[0] == count else value
            for name, value in rows.items()
        }
        criterion = (
            criterion_factory() if criterion_factory else PrescribedDepthCriterion({0: depths[g]})
        )
        _, _, grads = _step(
            model, _microbatches(one, mbs=1, world=1, rank=0), criterion, mbs=1, extra=extra
        )
        for name, grad in grads.items():
            total[name] = total[name] + grad.abs() if name in total else grad.abs()
    return total


def _assert_grads_close(actual, expected, abs_sum):
    assert actual.keys() == expected.keys(), actual.keys() ^ expected.keys()
    for name, grad in expected.items():
        floor = 1e-6 * abs_sum[name].max().item()
        torch.testing.assert_close(
            actual[name], grad, rtol=1e-5, atol=floor, msg=lambda text, n=name: f"{n}: {text}"
        )


def _sum_over_ranks(grads):
    for grad in grads.values():
        dist.all_reduce(grad)
    return grads


def _train_step_reduction(reports, group):
    """Copy of ``train_step``'s loss reduction: per key, the local ``[value, weight]`` rows
    summed, then one all-reduce over the DP group; the logged value is their ratio."""
    reduced = {}
    for key in reports[0]:
        total = torch.vstack([report[key] for report in reports]).sum(0)
        if group is not None:
            dist.all_reduce(total, group=group)
        reduced[key] = total
    return reduced


class _Rearrivals:
    """Counts checkpoints received on this EU after they left it by a move (``"sent"`` keys
    installed again with ``put_received_*``)."""

    def __init__(self):
        self.count = 0
        self._original = LoopStateStore._put
        counter, original = self, self._original

        def put(store, key, tensor, *, received=False):
            # pylint: disable=protected-access
            if received and store._released.get(key) == "sent":
                counter.count += 1
            return original(store, key, tensor, received=received)

        LoopStateStore._put = put

    def close(self):
        LoopStateStore._put = self._original


class _CoreCalls:
    """Counts forward calls of the core sublayers (sweeps and CORE items)."""

    def __init__(self, model):
        roles = model.decoder.loop_roles
        self.count = 0
        self.handles = [
            layer.register_forward_pre_hook(self._hook)
            for layer in model.decoder.layers[roles.core_start : roles.core_end]
        ]
        self.layers = roles.core

    def _hook(self, module, args):
        del module, args
        self.count += 1

    def close(self):
        for handle in self.handles:
            handle.remove()


# Arms: name -> executor keywords (plus "mbs").
ARMS = {
    "default": dict(),
    "all_to_all": dict(move_transport="all_to_all"),
    "no_rebalance": dict(rebalance=False),
    "no_item_moves": dict(tail_item_moves=False),
    "sample_order": dict(fill="sample_order"),
    "mbs2": dict(mbs=2),
    # Explicit attention masks travel with moved samples (batch slot "attention_mask").
    "mask_shared": dict(mbs=2, mask="shared"),
    "mask_rows": dict(mbs=2, mask="rows"),
}
# Arms run only by the cases that name them.
EXTRA_ARMS = {
    "sample_order_checked": dict(fill="sample_order", check_recompute=True, poison_on_free=True)
}


def _arms_worker(rank, world, spec):
    cpu_worker_setup()
    try:
        return _arms(rank, world, spec)
    finally:
        cpu_worker_teardown()


def _arms(rank, world, spec):
    """Run every arm of ``spec`` on this rank against the one-EU reference (asserting
    inside the worker); returns per-arm summaries."""
    domain = parallel_state.get_data_parallel_group()
    status = parallel_state.get_data_parallel_group_gloo()
    assert dist.get_world_size(domain) == world
    model = _model(spec["model"])
    per_rank, padded = spec["per_rank"], spec.get("padded", False)
    summaries = {}
    abs_sums = {}
    for arm in spec["arms"]:
        options = dict(ARMS[arm] if arm in ARMS else EXTRA_ARMS[arm])
        mbs = options.pop("mbs", 1)
        mask = options.pop("mask", None)
        options.update(spec.get("options", {}))
        total = per_rank * mbs * world
        rows = _rows(total, padded=padded, mask=mask)
        depths = _depths(spec["depths"], total, world, mbs)
        criterion = PrescribedDepthCriterion(depths)
        # The one-EU reference over the whole global batch (global microbatch order).
        ref_reports, ref_executor, ref_grads = _step(
            model, _microbatches(rows, mbs=mbs, world=1, rank=0), criterion, mbs=mbs
        )
        reference = _losses(ref_executor, ref_reports)
        if (total, mask) not in abs_sums:
            abs_sums[total, mask] = _abs_sample_grads(model, rows, depths)
        # W EUs.
        tokens_seen = []
        config = model.config
        config.finalize_model_grads_func = lambda models, num_tokens, **kwargs: tokens_seen.append(
            num_tokens.clone()
        )
        calls = _CoreCalls(model)
        rearrivals = _Rearrivals()
        try:
            microbatches = _microbatches(rows, mbs=mbs, world=world, rank=rank)
            reports, executor, grads = _step(
                model,
                microbatches,
                criterion,
                mbs=mbs,
                domain=domain,
                status=status if options.get("check_plan") else None,
                **options,
            )
        finally:
            rearrivals.close()
            calls.close()
            config.finalize_model_grads_func = None
        stats = executor.stats
        # Per-sample losses bitwise.
        losses = _losses(executor, reports)
        for key, loss in losses.items():
            assert torch.equal(loss, reference[key]), (arm, key, loss, reference[key])
        expected_home = sorted(
            (k * world + rank) * mbs + b for k in range(per_rank) for b in range(mbs)
        )
        assert sorted(losses) == expected_home, (arm, sorted(losses), expected_home)
        # Gradients summed over the EUs.
        _assert_grads_close(_sum_over_ranks(grads), ref_grads, abs_sums[total, mask])
        # Sync budget (known depths), store hygiene, zero core work after exit.
        assert stats.host_syncs == 1 and stats.status_exchanges == 0, (arm, stats.host_syncs)
        assert all(peak.live_entries == 0 for peak in stats.store_peaks)
        core_calls = torch.tensor([calls.count])
        dist.all_reduce(core_calls)
        assert core_calls.item() == 2 * calls.layers * sum(depths.values()), arm
        assert [
            stats.depths[key] for key in sorted(stats.depths, key=lambda k: k.global_index)
        ] == [depths[g] for g in range(total)]
        # Reports and normalizer (M7): finalize gets the home token count, unreduced.
        home_tokens = sum(int(report["lm loss"][1]) for report in reports)
        assert len(tokens_seen) == 1 and tokens_seen[0].item() == home_tokens
        keys = list(reports[0])
        assert all(list(report) == keys for report in reports)
        reduced = _train_step_reduction(reports, domain)
        expected = _train_step_reduction(ref_reports, None)
        assert reduced.keys() == expected.keys()
        for key in reduced:
            if key == "loop sample moves":
                assert reduced[key].tolist() == [
                    float(stats.moves["sample"]),
                    float(sum(depths.values())),
                ]
                assert expected[key][0].item() == 0.0
            elif key == "lm loss":
                assert reduced[key][1].item() == expected[key][1].item()
                torch.testing.assert_close(reduced[key], expected[key], rtol=1e-6, atol=0.0)
            else:
                torch.testing.assert_close(reduced[key], expected[key], rtol=1e-12, atol=0.0)
        if arm in spec.get("repeat", ()):
            # M6 over 2 steps: the same executor (cached dispatcher, reused upload buffers)
            # runs the step again: the same losses and gradients, one host sync, no exchange.
            dispatcher = executor.dispatcher
            model.zero_grad(set_to_none=True)
            again = executor.run_step(
                _next, iter(microbatches), len(microbatches), model=model, micro_batch_size=mbs
            )
            again_grads = _sum_over_ranks(_grads(model))
            assert executor.dispatcher is dispatcher, arm
            again_stats = executor.stats
            assert again_stats.host_syncs == 1 and again_stats.status_exchanges == 0, arm
            assert all(peak.live_entries == 0 for peak in again_stats.store_peaks)
            assert dict(again_stats.moves) == dict(stats.moves), arm
            for key, loss in _losses(executor, again).items():
                assert torch.equal(loss, reference[key]), (arm, "step 2", key)
            _assert_grads_close(again_grads, ref_grads, abs_sums[total, mask])
        summaries[arm] = dict(
            rearrivals=rearrivals.count,
            moves=dict(stats.moves),
            moves_by_round=stats.moves_by_round,
            active_by_round=stats.active_by_round,
            nulls=dict(stats.nulls),
            items=len(stats.item_order),
            digests=[digest.hex() for digest in stats.plan_digests],
        )
    return summaries


def _check_plan_identity(results):
    """Every rank holds the same plan (digests) and the same plan-derived stats."""
    first = results[0]
    for result in results[1:]:
        for arm, summary in result.items():
            for key in ("moves", "moves_by_round", "active_by_round", "nulls", "digests"):
                assert summary[key] == first[arm][key], (arm, key)


# ---------------------------------------------------------------------------
# M1 / M2 / M7: W EUs against one EU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world", [2, 4, 8])
def test_dense_arms_match_one_eu(world):
    """M1 / M2 / M6 / M7 on the dense model: every arm (rebalance on / off, both transports,
    tail item moves on / off, drain / sample_order, MBS 2, shared / per-row attention masks)
    gives the one-EU per-sample losses
    bitwise and its gradients within the gate; the default arm moves samples (skewed
    depths), the no-rebalance arm moves none; one host sync and no status exchange per
    step, also for a second step on the same executor (default and all_to_all arms)."""
    results = run_spawned(
        _arms_worker,
        world,
        dict(
            model="dense",
            depths="skew",
            per_rank=4,
            arms=list(ARMS),
            repeat=("default", "all_to_all"),
        ),
        timeout=TIMEOUT,
    )
    _check_plan_identity(results)
    summary = results[0]
    assert summary["default"]["moves"]["sample"] > 0
    assert summary["all_to_all"]["moves"] == summary["default"]["moves"]
    assert summary["no_rebalance"]["moves"]["sample"] == 0
    assert summary["no_item_moves"]["moves"]["item"] == 0
    if world >= 4:
        # The skewed tail moves backward items with their checkpoints and chain records.
        assert summary["default"]["moves"]["item"] > 0, summary["default"]["moves"]
        assert summary["default"]["moves"]["chain"] > 0, summary["default"]["moves"]
    # The masks are a batch slot of the moved samples.
    assert summary["mask_shared"]["moves"]["sample"] > 0
    assert summary["mask_rows"]["moves"] == summary["mask_shared"]["moves"]
    # Lockstep sub-rounds: EUs without an item of a sub-round's type run nulls.
    assert sum(summary["default"]["nulls"].values()) > 0
    # Items run where their checkpoint lives: the item counts per rank differ without
    # rebalancing (EU 0 holds every deep sample).
    items = [result["no_rebalance"]["items"] for result in results]
    assert len(set(items)) > 1, items


@pytest.mark.parametrize("world", [2, 8])
def test_mtp_mhc_padded_and_sparse_depths(world):
    """M1 / M2 on the 4-stream mHC + MTP model with padded rows (mixed depths), and the
    sparse depth set (one deep sample: rounds with fewer active samples than EUs, EUs
    without a sample running nulls)."""
    _needs_dynamo_off("mhc")
    results = run_spawned(
        _arms_worker,
        world,
        dict(model="mhc_mtp", depths="mixed", per_rank=2, padded=True, arms=["default"]),
        timeout=TIMEOUT,
    )
    _check_plan_identity(results)
    results = run_spawned(
        _arms_worker,
        world,
        dict(model="dense", depths="sparse", per_rank=2, arms=["default", "all_to_all"]),
        timeout=TIMEOUT,
    )
    _check_plan_identity(results)
    summary = results[0]["default"]
    assert min(summary["active_by_round"][0][1:]) < world  # a round with A < W
    assert summary["nulls"]["core_f"] > 0


@pytest.mark.parametrize("world", [2, 4])
def test_hygiene_checks_change_nothing(world):
    """M9: check_recompute (the expected state travels in chain records), poison_on_free and
    check_plan (digest gathers over the status group) give the plain arms' results."""
    _needs_dynamo_off("mhc")
    results = run_spawned(
        _arms_worker,
        world,
        dict(
            model="mhc",
            depths="skew",
            per_rank=2,
            arms=["default"],
            options=dict(check_recompute=True, poison_on_free=True, check_plan=True),
        ),
        timeout=TIMEOUT,
    )
    _check_plan_identity(results)
    assert results[0]["default"]["moves"]["sample"] > 0


def test_item_moves_back_to_a_former_sample_move_source():
    """M2 at W = 4 (``sample_order``): a checkpoint that left an EU by a sample move comes
    back to it with a later item move (``"sent"`` -> received again), plain and with
    check_recompute + poison_on_free; the one-EU losses bitwise and gradients within the
    gate."""
    results = run_spawned(
        _arms_worker,
        4,
        dict(
            model="dense",
            depths="rearrive",
            per_rank=2,
            arms=["sample_order", "sample_order_checked"],
        ),
        timeout=TIMEOUT,
    )
    _check_plan_identity(results)
    for arm in ("sample_order", "sample_order_checked"):
        assert results[0][arm]["moves"]["item"] > 0, results[0][arm]["moves"]
        assert sum(result[arm]["rearrivals"] for result in results) > 0, arm


# ---------------------------------------------------------------------------
# M3: waves across EUs
# ---------------------------------------------------------------------------


def _waves_worker(rank, world):
    cpu_worker_setup()
    try:
        domain = parallel_state.get_data_parallel_group()
        model = _model("dense")
        per_rank = 4
        total = per_rank * world
        rows = _rows(total)
        depths = _depths("mixed", total, world, 1)
        criterion = PrescribedDepthCriterion(depths)
        abs_sum = _abs_sample_grads(model, rows, depths)
        mine = _microbatches(rows, mbs=1, world=world, rank=rank)
        results = {}
        for waves in (1, 2, 4):
            reports, executor, grads = _step(
                model, mine, criterion, mbs=1, domain=domain, waves=waves
            )
            results[waves] = (_losses(executor, reports), _sum_over_ranks(grads), executor)
        base_losses, base_grads, _ = results[1]
        for waves, (losses, grads, executor) in results.items():
            assert losses.keys() == base_losses.keys()
            for key in losses:
                assert torch.equal(losses[key], base_losses[key]), (waves, key)
            _assert_grads_close(grads, base_grads, abs_sum)
            stats = executor.stats
            assert len(stats.store_peaks) == waves and stats.host_syncs == 1
            assert len(stats.moves_by_round) == waves
        return [len(results[w][2].stats.rounds_by_wave) for w in (1, 2, 4)]
    finally:
        cpu_worker_teardown()


def test_waves_match_one_pass():
    """M3 at W = 4: waves 2 and 4 against one pass: per-sample losses bitwise, gradients
    within the gate, one host sync per step, every wave planned over the same EUs."""
    results = run_spawned(_waves_worker, 4, timeout=TIMEOUT)
    assert results[0] == [1, 2, 4]


# ---------------------------------------------------------------------------
# M4: an online criterion (status exchange once per round)
# ---------------------------------------------------------------------------


def _online_worker(rank, world):
    cpu_worker_setup()
    try:
        domain = parallel_state.get_data_parallel_group()
        status = parallel_state.get_data_parallel_group_gloo()
        model = _model("mhc")
        head = HaltingHead(4, model.config.hidden_size, device=torch.device("cpu"), seed=3)

        def criterion():
            return OnlineHaltingCriterion(head, tau=1.4, carry=0.05)

        per_rank = 3
        total = per_rank * world
        rows = _rows(total, seed=11)
        ref_reports, ref_executor, ref_grads = _step(
            model, _microbatches(rows, mbs=1, world=1, rank=0), criterion(), mbs=1, extra=(head,)
        )
        reference = _losses(ref_executor, ref_reports)
        realized = {key.global_index: d for key, d in ref_executor.stats.depths.items()}
        assert len(set(realized.values())) > 1, realized
        abs_sum = _abs_sample_grads(
            model, rows, realized, extra=(head,), criterion_factory=criterion
        )
        mine = _microbatches(rows, mbs=1, world=world, rank=rank)
        out = {}
        arms = {
            "one_pass": dict(waves=1),
            "waves3": dict(waves=3),
            # sample_order fill over the all_to_all transport, one pass and in waves.
            "sample_order_a2a": dict(fill="sample_order", move_transport="all_to_all"),
            "sample_order_a2a_waves3": dict(
                waves=3, fill="sample_order", move_transport="all_to_all"
            ),
        }
        for name, options in arms.items():
            reports, executor, grads = _step(
                model,
                mine,
                criterion(),
                mbs=1,
                domain=domain,
                status=status,
                extra=(head,),
                check_plan=True,
                **options,
            )
            stats = executor.stats
            for key, loss in _losses(executor, reports).items():
                assert torch.equal(loss, reference[key]), (name, key)
            assert {key.global_index: d for key, d in stats.depths.items()} == realized
            _assert_grads_close(_sum_over_ranks(grads), ref_grads, abs_sum)
            rounds = sum(stats.rounds_by_wave)
            assert stats.status_exchanges == rounds == sum(stats.status_exchanges_by_wave)
            assert stats.host_syncs == rounds + 1
            assert stats.host_syncs_by_wave == stats.rounds_by_wave
            out[name] = (dict(stats.moves), stats.rounds_by_wave)
        return out
    finally:
        cpu_worker_teardown()


@pytest.mark.parametrize("world", [4, 8])
def test_online_criterion_exchanges_status_once_per_round(world):
    """M4 at W = 4 and 8: a learned halting criterion (mHC, 4 streams) whose exits are known
    only after each round: the realized depths, per-sample losses (bitwise) and the gradients
    of the model and the criterion (rounds and init_state) equal the one-EU run; one status
    exchange and one host sync per forward round of each wave, plus one per step; with drain
    and p2p, and with sample_order over all_to_all, each in one pass and in 3 waves. At W = 8
    the one-pass steps move samples (and their chain records back), so online exits of
    samples away from home and exit flags travelling with a move are covered."""
    _needs_dynamo_off("mhc")
    results = run_spawned(_online_worker, world, timeout=TIMEOUT)
    assert all(result == results[0] for result in results)
    if world == 8:
        for arm in ("one_pass", "sample_order_a2a"):
            moves, _ = results[0][arm]
            assert moves["sample"] > 0 and moves["chain"] > 0, (arm, moves)


# ---------------------------------------------------------------------------
# M5: buggy plans raise on every rank
# ---------------------------------------------------------------------------


def _violation_worker(rank, world, bug, online, case):
    """``case`` ``"skew"``: skewed depths, the buggy sample stays home. ``"moved"``: every
    sample exits at round 1 except sample ``world`` (3 rounds), so the late-retired sample 0
    (home EU 0) is moved to EU 1 by the rebalancer before its extra sweep; only the exit
    flags that travel with the move tell EU 1 that it already exited."""
    cpu_worker_setup()
    try:
        domain = parallel_state.get_data_parallel_group()
        status = parallel_state.get_data_parallel_group_gloo()
        model = _model("dense")
        total = 2 * world
        if case == "skew":
            depths = _depths("skew", total, world, 1)
            sample = 1  # home EU 1, exits at round 1; swept once more at round 2
        else:
            depths = {g: 3 if g == world else 1 for g in range(total)}
            sample = 0  # home EU 0, exits at round 1; moved to EU 1 and swept there
        rows = _rows(total)
        if online and case == "moved":
            # PaddingDepthCriterion: depth = padding tokens + 1.
            padding = torch.zeros(total, SEQ, dtype=torch.bool)
            for g, depth in depths.items():
                padding[g, SEQ - (depth - 1) :] = True
            rows["padding_mask"] = padding
            rows["loss_mask"] = rows["loss_mask"] * (~padding).float()
        original = LoopRSBFExecutor._newly_exited
        # The same bug on every rank (a planner bug is a pure function of the shared inputs,
        # so every rank has it); the device checks find it on the EU that runs the bad call.
        if bug == "late":
            LoopRSBFExecutor._newly_exited = retire_late(original, sample=sample)
        else:
            LoopRSBFExecutor._newly_exited = retire_early(original, sample=0, at_round=1)
        if online:
            criterion = PaddingDepthCriterion() if case == "moved" else ConstantExitCriterion(True)
        else:
            criterion = PrescribedDepthCriterion(depths)
        program = LoopProgram(model, exit_criterion=criterion, check_ports=True)
        executor = LoopRSBFExecutor(
            program,
            pg_collection=SimpleNamespace(dp=domain),
            groups=RSBFGroups(domain=domain, status=status),
        )
        mine = _microbatches(rows, mbs=1, world=world, rank=rank)
        message = None
        try:
            executor.run_step(_next, iter(mine), len(mine), model=model, micro_batch_size=1)
        except RuntimeError as error:
            message = str(error)
        finally:
            LoopRSBFExecutor._newly_exited = original
        return message, executor.stats.moves["sample"], executor._moves_of[sample]
    finally:
        cpu_worker_teardown()


@pytest.mark.parametrize(
    "bug,online,case,match",
    [
        ("late", False, "skew", SWEEP_AFTER_EXIT),
        ("early", False, "skew", PLAN_MISMATCH),
        ("late", True, "skew", SWEEP_AFTER_EXIT),
        ("late", False, "moved", SWEEP_AFTER_EXIT),
        ("late", True, "moved", SWEEP_AFTER_EXIT),
    ],
    ids=[
        "retire_late-known",
        "retire_early-known",
        "retire_late-online",
        "retire_late-moved-known",
        "retire_late-moved-online",
    ],
)
def test_buggy_plans_raise_on_every_rank(bug, online, case, match):
    """M5 at W = 2: a host plan that sweeps a sample after its exit or retires one early is
    caught by the device checks of the EU that ran the bad sweep, made global (known depths:
    the end-of-step flags all-reduce; online: byte G of the round's status exchange) and
    raised by every rank, naming the offending EU; no rank hangs. The ``moved`` cases move
    the exited sample to EU 1 before its extra sweep: EU 1 knows of the exit only through
    the exit flags of the move (without them the step would complete silently: EU 1 would
    re-detect the exit one round late, matching the late host plan)."""
    results = run_spawned(_violation_worker, 2, bug, online, case, timeout=TIMEOUT)
    messages = [message for message, _, _ in results]
    assert all(message is not None and match in message for message in messages), messages
    if case == "moved":
        # The exited sample left its home before the bad sweep (the same plan on every rank).
        assert all(moves > 0 and moved_g > 0 for _, moves, moved_g in results), results
    if online:
        # The EU that swept the retired sample names itself, the other names "another EU".
        offender = "on EU 1" if case == "moved" else "on EU"
        assert sum(offender in message for message in messages) == 1, messages
        assert sum("another EU" in message for message in messages) == 1, messages
        if case == "moved":
            assert SWEEP_AFTER_EXIT in messages[1] and "on EU 1" in messages[1], messages
    else:
        assert all("on domain EUs" in message for message in messages), messages
        assert len(set(messages)) == 1  # the same decision on every rank
        if case == "moved":
            assert "on domain EUs [1]" in messages[0], messages


# ---------------------------------------------------------------------------
# The loss-function contract
# ---------------------------------------------------------------------------


def _contract_worker(rank, world):
    cpu_worker_setup()
    try:
        domain = parallel_state.get_data_parallel_group()
        model = _model("dense")
        rows = _rows(2 * world)
        depths = _depths("mixed", 2 * world, world, 1)
        other = partial(_loss_func)
        mine = _microbatches(rows, mbs=1, world=world, rank=rank, loss_funcs=[_loss_func, other])
        with pytest.raises(ValueError, match="not equivalent"):
            _step(model, mine, PrescribedDepthCriterion(depths), mbs=1, domain=domain)
        # Partials of one function with identical arguments are equivalent.
        same = [partial(_loss_func), partial(_loss_func)]
        mine = _microbatches(rows, mbs=1, world=world, rank=rank, loss_funcs=same)
        reports, _, _ = _step(model, mine, PrescribedDepthCriterion(depths), mbs=1, domain=domain)
        # Known-depth criteria must cover the whole wave of the domain.
        local = {g: d for g, d in depths.items() if g % world == rank}
        mine = _microbatches(rows, mbs=1, world=world, rank=rank)
        with pytest.raises(ValueError, match="cover every sample of the domain's wave"):
            _step(model, mine, PrescribedDepthCriterion(local), mbs=1, domain=domain)
        return len(reports)
    finally:
        cpu_worker_teardown()


def test_loss_function_and_known_depth_contracts():
    """With data parallelism every microbatch carries an equivalent loss function, and
    known-depth criteria answer for every sample of the domain's wave (both checked on the
    host before any collective, so every rank raises)."""
    assert run_spawned(_contract_worker, 2, timeout=TIMEOUT) == [2, 2]


# ---------------------------------------------------------------------------
# M8: lockstep MoE (null items) with the stub 'E' module
# ---------------------------------------------------------------------------

# Prelude '*E' (2), core '*-*E' (4), coda '*E' (2), MTP '*E': one lockstep module per role.
STUB = "*E*-*E*E/*E"
STUB_MOE = dict(num_moe_experts=4, moe_ffn_hidden_size=128, moe_router_topk=2)
_TYPES = ("prelude_f", "core_f", "core_b", "coda", "prelude_b")
_BACKWARD = ("core_b", "coda", "prelude_b")


def _stub_model():
    """The stub model and its ``RecordingLockstepMLP`` modules (construction order: prelude E,
    core E, coda E, MTP E)."""
    RecordingLockstepMLP.instances = []
    model = build_local_loop_model(STUB, moe_layer=stub_moe_layer_spec(), **RSBF, **STUB_MOE)
    stubs = list(RecordingLockstepMLP.instances)
    assert [stub.index for stub in stubs] == [0, 1, 2, 3]
    return model, stubs


def _stub_groups(world, rank, layout):
    """This rank's (one-rank group, the stubs' EP group): ``"world"`` (EP = W) or ``"pairs"``
    (EP = 2 groups {0, 1}, {2, 3}, ...; every group created on every rank)."""
    own = [dist.new_group([r], backend="gloo") for r in range(world)][rank]
    if layout == "world":
        return own, dist.new_group(list(range(world)), backend="gloo")
    pairs = [dist.new_group([2 * k, 2 * k + 1], backend="gloo") for k in range(world // 2)]
    return own, pairs[rank // 2]


def _set_group(stubs, group):
    for stub in stubs:
        stub.group = group


class _SubRoundMarker:
    """While installed, ``StubLog.marker`` is ``(serial, sub-round type, null?)`` for the stub
    calls of each executed sub-round (None outside sub-rounds)."""

    def __init__(self):
        self.serial = 0
        self._original = LoopRSBFExecutor._execute
        marker, original = self, self._original

        def execute(executor, steps):
            for step in steps:
                if isinstance(step, SubRound):
                    StubLog.marker = (
                        marker.serial,
                        step.type.value,
                        step.items[executor.dispatcher.eu] is None,
                    )
                    marker.serial += 1
                else:
                    StubLog.marker = None
                original(executor, [step])
            StubLog.marker = None

        LoopRSBFExecutor._execute = execute

    def close(self):
        LoopRSBFExecutor._execute = self._original


def _expected_calls(plan):
    """Per sub-round type, the stub calls of a real item (= of a null): the type's modules
    forward, then for the backward types in reverse."""
    expected = {}
    for kind_type, entries in plan.by_type.items():
        calls = [(entry.module.index, "fwd") for entry in entries]
        if kind_type.value in _BACKWARD:
            calls += [(entry.module.index, "bwd") for entry in reversed(entries)]
        expected[kind_type.value] = calls
    return expected


def _lockstep_worker(rank, world, spec):
    cpu_worker_setup()
    unregister = register_lockstep_module(RecordingLockstepMLP, StubLockstepAdapter())
    try:
        domain = parallel_state.get_data_parallel_group()
        status = parallel_state.get_data_parallel_group_gloo()
        own, ep = _stub_groups(world, rank, spec["layout"])
        model, stubs = _stub_model()
        out = {}
        for depth_kind in spec["depths"]:
            per_rank = spec["per_rank"]
            total = per_rank * world
            rows = _rows(total)
            depths = _depths(depth_kind, total, world, 1)
            criterion = PrescribedDepthCriterion(depths)
            # The one-EU reference: the stubs exchange nothing (one-rank group, no lockstep).
            _set_group(stubs, own)
            ref_reports, ref_executor, ref_grads = _step(
                model, _microbatches(rows, mbs=1, world=1, rank=0), criterion, mbs=1
            )
            assert not ref_executor.lockstep.active
            reference = _losses(ref_executor, ref_reports)
            abs_sum = _abs_sample_grads(model, rows, depths)
            # W EUs, the stubs lockstep over their EP group.
            _set_group(stubs, ep)
            StubLog.reset()
            marker = _SubRoundMarker()
            try:
                reports, executor, grads = _step(
                    model,
                    _microbatches(rows, mbs=1, world=world, rank=rank),
                    criterion,
                    mbs=1,
                    domain=domain,
                    status=status,
                    check_lockstep=True,
                    check_plan=True,
                    **spec.get("options", {}),
                )
            finally:
                marker.close()
            plan = executor.lockstep
            # Discovery: prelude E, core E, coda E + the MTP depth's E.
            assert {t.value: [e.module.index for e in plan.by_type[t]] for t in plan.by_type} == {
                "prelude_f": [0],
                "prelude_b": [0],
                "core_f": [1],
                "core_b": [1],
                "coda": [2, 3],
            }
            assert [entry.name for entry in plan.by_type[SubRoundType.CODA]] == [
                "decoder.layers.7.mlp",
                "mtp.layers.0.mtp_model_layer.layers.1.mlp",
            ]
            for key, loss in _losses(executor, reports).items():
                assert torch.equal(loss, reference[key]), (depth_kind, key)
            _assert_grads_close(_sum_over_ranks(grads), ref_grads, abs_sum)
            assert all(stub.mismatches == 0 for stub in stubs), [s.mismatches for s in stubs]
            stats = executor.stats
            assert stats.host_syncs == 1 and all(p.live_entries == 0 for p in stats.store_peaks)
            out[depth_kind] = dict(
                logs=[entry for entry in StubLog.entries if entry[0] is not None],
                expected=_expected_calls(plan),
                null_moe_calls=stats.null_moe_calls,
                nulls=dict(stats.nulls),
                moves=dict(stats.moves),
            )
        return out
    finally:
        unregister()
        cpu_worker_teardown()


def _check_lockstep_logs(results, world, layout):
    """Every sub-round's stub calls equal the real item's calls of its type on every rank
    (real or null), so every EP group issued the same exchanges; returns the types that had
    a null next to a real item."""
    mixed = set()
    for depth_kind in results[0]:
        expected = results[0][depth_kind]["expected"]
        per_rank = []
        for result in results:
            by_serial = {}
            for (serial, kind, null), index, phase in result[depth_kind]["logs"]:
                entry = by_serial.setdefault(serial, [kind, null, []])
                entry[2].append((index, phase))
            per_rank.append(by_serial)
        serials = set(per_rank[0])
        assert all(set(by_serial) == serials for by_serial in per_rank)
        for serial in sorted(serials):
            kinds = {by_serial[serial][0] for by_serial in per_rank}
            assert len(kinds) == 1, (depth_kind, serial, kinds)
            (kind,) = kinds
            for rank, by_serial in enumerate(per_rank):
                assert by_serial[serial][2] == expected[kind], (
                    depth_kind,
                    serial,
                    rank,
                    by_serial[serial],
                    expected[kind],
                )
            nulls = {by_serial[serial][1] for by_serial in per_rank}
            if nulls == {True, False}:
                mixed.add(kind)
        # Sub-rounds without any lockstep call on a rank never happen: every type has one.
        assert all(expected[kind] for kind in _TYPES)
    del world, layout
    return mixed


@pytest.mark.parametrize("world,layout", [(2, "world"), (4, "world"), (4, "pairs"), (8, "world")])
def test_lockstep_nulls_replay_the_moe_calls_of_their_type(world, layout):
    """M8: the stub 'E' model (prelude E, core E, coda E, MTP E; every stub call all-gathers
    its tag over the stubs' EP group: the whole domain, or pairs of EUs) at W EUs against one
    EU: per-sample losses bitwise and gradients within the gate; discovery lists each type's
    MoE calls (CODA: the coda E, then the MTP depth's E); in every sub-round every rank, with a
    real item or a null, makes exactly the real item's calls of the type (forward, then for
    CORE-B / CODA / PRELUDE-B in reverse) and the tags agree across each EP group; nulls ran
    next to real items in every type that can have them (PRELUDE-F cannot: equal home
    counts); check_lockstep and check_plan pass."""
    results = run_spawned(
        _lockstep_worker,
        world,
        dict(layout=layout, depths=["skew", "sparse"], per_rank=2),
        timeout=TIMEOUT,
    )
    mixed = _check_lockstep_logs(results, world, layout)
    assert {"core_f", "core_b", "coda", "prelude_b"} <= mixed, mixed
    assert sum(result["sparse"]["null_moe_calls"] for result in results) > 0
    for depth_kind in ("skew", "sparse"):
        assert all(r[depth_kind]["nulls"] == results[0][depth_kind]["nulls"] for r in results)


def _over_budget_worker(rank, world, mode):
    """Rank 1's core E reports a static-budget overflow on its first forward call."""
    import logging

    cpu_worker_setup()
    unregister = register_lockstep_module(RecordingLockstepMLP, StubLockstepAdapter())
    warnings = []

    class _Capture(logging.Handler):
        def emit(self, record):
            warnings.append(record.getMessage())

    log = logging.getLogger("megatron.core.transformer.loop.rsbf_executor")
    handler = _Capture(level=logging.WARNING)
    log.addHandler(handler)
    try:
        domain = parallel_state.get_data_parallel_group()
        _, ep = _stub_groups(world, rank, "world")
        model, stubs = _stub_model()
        _set_group(stubs, ep)
        total = 2 * world
        rows = _rows(total)
        criterion = PrescribedDepthCriterion(_depths("mixed", total, world, 1))
        mine = _microbatches(rows, mbs=1, world=world, rank=rank)
        out = []
        for step in range(2):
            stubs[1].forward_calls = 0
            stubs[1].trip_over_budget = {1} if rank == 1 and step == 0 else set()
            program = LoopProgram(model, exit_criterion=criterion)
            executor = LoopRSBFExecutor(
                program,
                pg_collection=SimpleNamespace(dp=domain),
                groups=RSBFGroups(domain=domain),
                over_budget=mode,
            )
            message = None
            try:
                executor.run_step(_next, iter(mine), len(mine), model=model, micro_batch_size=1)
            except RuntimeError as error:
                message = str(error)
            stats = executor.stats
            out.append(
                (
                    message,
                    stats.flagged_ranks.get("over_budget"),
                    list(stats.over_budget_layers),
                    list(warnings),
                )
            )
            warnings.clear()
        return out
    finally:
        log.removeHandler(handler)
        unregister()
        cpu_worker_teardown()


@pytest.mark.parametrize("mode", ["raise", "warn"])
def test_static_budget_overflow_is_decided_on_every_rank(mode):
    """The over-budget flags of the lockstep modules are reset at every step start and reduced
    over the domain with the end-of-step flags: an overflow on EU 1 only makes every rank raise
    (``raise``) naming EU 1, with the overflowed layer named on EU 1; ``warn`` logs it once on
    the domain's first rank and every rank continues. The next step (no overflow) is clean."""
    results = run_spawned(_over_budget_worker, 2, mode, timeout=TIMEOUT)
    first = [result[0] for result in results]
    second = [result[1] for result in results]
    for message, flagged, layers, warnings in first:
        assert flagged == [1]
    assert first[0][2] == [] and first[1][2] == ["decoder.layers.5.mlp"]
    if mode == "raise":
        messages = [message for message, _, _, _ in first]
        assert all(
            m is not None and OVER_BUDGET in m and "on domain EUs [1]" in m for m in messages
        )
        assert "decoder.layers.5.mlp" in messages[1] and "decoder.layers" not in messages[0]
        assert all(not warnings for _, _, _, warnings in first)
    else:
        assert all(message is None for message, _, _, _ in first)
        assert len(first[0][3]) == 1 and OVER_BUDGET in first[0][3][0]
        assert first[1][3] == []
    for message, flagged, layers, warnings in second:
        assert message is None and flagged == [] and layers == [] and warnings == []
