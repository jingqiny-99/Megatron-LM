# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The realized exit-depth histogram of an RSBF step and the measured depth distribution.

* ``exit_depth_counts``: the histogram of host depths, range-checked.
* ``ExecutorStats.exit_counts`` after a step: the histogram of ``stats.depths`` (every wave),
  summing to the step's samples, for prescribed, sampled and online criteria; with one EU
  in-process and over spawned Gloo worlds of 2 and 4 EUs (identical on every rank, waves 1
  and 2); ``padded_tokens`` / ``issued_tokens``: this EU's home rows.
* ``LoopRSBFSchedule.exit_counts`` / ``.padding`` / ``.invalidate``.
* ``DepthDistribution``: the file that training writes (``--loop-exit-histogram-path``) read
  back as a pmf over an iteration window; draws bitwise the cost model's ``depths_for``.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.loop.planner import (
    EXIT_HISTOGRAM_FORMAT,
    R4_GEOM,
    R4_SKEW,
    R4_UNIF,
    DepthDistribution,
)
from megatron.core.transformer.loop.program import SampledDepthCriterion, SampleKey
from megatron.core.transformer.loop.rsbf_executor import LoopRSBFSchedule, exit_depth_counts
from tests.unit_tests.transformer.loop import test_loop_rsbf_executor as base
from tests.unit_tests.transformer.loop.hybrid_loop_utils import SEQ, model_parallel_once
from tests.unit_tests.transformer.loop.multi_eu_utils import (
    SPAWN_SKIP_REASON,
    cpu_worker_setup,
    cpu_worker_teardown,
    run_spawned,
    spawn_tests_enabled,
)

hybrid = base.hybrid
spawned = pytest.mark.skipif(
    not spawn_tests_enabled() or torch.cuda.is_available(),
    reason=SPAWN_SKIP_REASON + "; CPU hosts only",
)


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        try:
            yield
        finally:
            base.get_loop_rsbf_schedule().reset()


def _histogram(depths, max_rounds):
    return [sum(1 for d in depths if d == k) for k in range(1, max_rounds + 1)]


def _padded(batches):
    return sum(
        0 if batch["padding_mask"] is None else int(batch["padding_mask"].sum())
        for batch in batches
    )


# ---------------------------------------------------------------------------
# exit_depth_counts
# ---------------------------------------------------------------------------


def test_exit_depth_counts_bins_depths_one_to_r_max():
    assert exit_depth_counts([1, 3, 3, 4, 1, 1], 4) == [3, 0, 2, 1]
    assert exit_depth_counts([], 3) == [0, 0, 0]
    assert exit_depth_counts(iter([1]), 1) == [1]
    for bad in (0, 5, -1):
        with pytest.raises(ValueError, match="outside 1..R_max = 4"):
            exit_depth_counts([1, bad], 4)


# ---------------------------------------------------------------------------
# One EU (in-process)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("waves,mbs", [(1, 1), (2, 1), (2, 2)])
def test_step_histogram_counts_every_wave(request, hybrid, waves, mbs, padded):
    """Prescribed depths over 8 samples: the step's histogram is the histogram of every wave's
    realized depths and sums to the samples; the EU's padded and issued tokens are its rows'."""
    depths = (1, 3, 2, 4, 4, 1, 2, 4)
    model = base._model("dense", request)
    batches = base._batches(8, mbs, padded=padded, seed=7)
    _, executor, _ = base._rsbf(model, hybrid, batches, depths, waves=waves)
    stats = executor.stats
    assert len(stats.rounds_by_wave) == waves
    assert stats.exit_counts == _histogram(depths, base.R_MAX) == [2, 2, 1, 3]
    assert sum(stats.exit_counts) == len(stats.depths) == 8
    assert stats.exit_counts == exit_depth_counts(stats.depths.values(), base.R_MAX)
    assert stats.issued_tokens == 8 * SEQ
    assert stats.padded_tokens == _padded(batches) == (4 * (SEQ // 4) if padded else 0)


def test_online_step_histogram_is_the_realized_depths(request, hybrid):
    """An online criterion (exits known only after each round): the histogram bins the
    realized depths, which differ between samples."""
    model = base._model("dense", request)
    criterion = base._halting(model)
    batches = base._batches(6, seed=11)
    _, executor, _ = base._rsbf(
        model, hybrid, batches, criterion=criterion, extra_modules=(criterion.head,)
    )
    depths = [executor.stats.depths[SampleKey(i, 0)] for i in range(6)]
    assert len(set(depths)) > 1, depths
    assert executor.stats.exit_counts == _histogram(depths, base.R_MAX)
    assert sum(executor.stats.exit_counts) == 6


def test_exit_counts_before_the_end_of_step_read(request, hybrid, monkeypatch):
    """The histogram is host data, final before the end-of-step read (it adds no sync)."""
    model = base._model("dense", request)
    executor = base._executor(model, (2, 2, 3, 1))
    seen = []
    original = executor._end_of_step

    def end_of_step(dispatcher):
        seen.append(list(executor.stats.exit_counts))
        return original(dispatcher)

    monkeypatch.setattr(executor, "_end_of_step", end_of_step)
    batches = base._batches(4)
    executor.run_step(
        base._next_batch,
        iter(base._microbatches(model, hybrid, batches)),
        len(batches),
        model=model,
    )
    assert seen == [[1, 2, 1, 0]]


# ---------------------------------------------------------------------------
# The schedule's accessors
# ---------------------------------------------------------------------------


def _schedule_step(schedule, model, hybrid, batches, iteration):
    forward_step, iterator = base._forward_step_func(model, hybrid, batches)
    schedule.set_iteration(iteration)
    schedule(**base._schedule_kwargs(model, forward_step, iterator, len(batches)))


def test_schedule_exit_counts_padding_and_invalidate(request, hybrid):
    """``exit_counts`` / ``padding``: None before a model's first step, then its last step's
    histogram (sampled depths through the schedule entry; every EU of the module's DP group
    counts every domain sample) and this EU's (padded, issued) tokens. ``invalidate`` drops
    only that model's executor and keeps the iteration and the status group."""
    sampled = dict(
        loop_exit_criterion="sampled", loop_depth_probs=[0.25, 0.25, 0.25, 0.25], loop_depth_seed=3
    )
    first = base._model("dense", request, **sampled)
    second = base._model("dense", request, **sampled)
    schedule = LoopRSBFSchedule()
    assert schedule.exit_counts(first) is None and schedule.padding(first) is None
    batches = base._batches(4, padded=True)
    dp_size = torch.distributed.get_world_size(ProcessGroupCollection.use_mpu_process_groups().dp)
    _schedule_step(schedule, first, hybrid, batches, iteration=5)
    executor = schedule.cached_executor(first)
    counts = schedule.exit_counts(first)
    assert counts == executor.stats.exit_counts and counts is not executor.stats.exit_counts
    assert sum(counts) == len(executor.stats.depths) == 4 * dp_size
    assert counts == _histogram(executor.stats.depths.values(), base.R_MAX)
    assert len([c for c in counts if c]) > 1, counts
    assert schedule.padding(first) == (_padded(batches), 4 * SEQ)
    assert schedule.exit_counts(second) is None
    _schedule_step(schedule, second, hybrid, base._batches(4), iteration=6)
    assert schedule.padding(second) == (0, 4 * SEQ)

    marker = object()
    schedule._status_group = marker
    schedule.invalidate(first)
    assert schedule.cached_executor(first) is None and schedule.exit_counts(first) is None
    assert schedule.cached_executor(second) is not None and schedule.exit_counts(second)
    assert schedule.iteration == 6 and schedule._status_group is marker
    schedule.invalidate(first)  # nothing cached: a no-op
    schedule._status_group = None
    _schedule_step(schedule, first, hybrid, batches, iteration=7)
    assert schedule.cached_executor(first) is not executor
    assert sum(schedule.exit_counts(first)) == 4 * dp_size


# ---------------------------------------------------------------------------
# W EUs (spawned Gloo worlds)
# ---------------------------------------------------------------------------


def _multi_eu_worker(rank, world):
    """Every rank's histogram and padding per arm: sampled (known) depths in one pass and in
    two waves, and an online criterion over the status exchange; padded rows."""
    from megatron.core import parallel_state
    from megatron.core.transformer.loop.rsbf_executor import exit_depth_counts
    from tests.unit_tests.transformer.loop import test_loop_rsbf_multi_eu as multi
    from tests.unit_tests.transformer.loop.rsbf_test_criteria import (
        HaltingHead,
        OnlineHaltingCriterion,
    )

    cpu_worker_setup()
    try:
        domain = parallel_state.get_data_parallel_group()
        status = parallel_state.get_data_parallel_group_gloo()
        model = multi._model("dense")
        per_rank = 4
        rows = multi._rows(per_rank * world, padded=True, seed=9)
        mine = multi._microbatches(rows, mbs=1, world=world, rank=rank)
        head = HaltingHead(1, model.config.hidden_size, device=torch.device("cpu"), seed=3)
        arms = {
            "sampled": (lambda: SampledDepthCriterion([0.25] * 4, seed=3), dict(waves=1), ()),
            "sampled_waves2": (
                lambda: SampledDepthCriterion([0.25] * 4, seed=3),
                dict(waves=2),
                (),
            ),
            "online": (
                lambda: OnlineHaltingCriterion(head, tau=1.4, carry=0.05),
                dict(waves=1),
                (head,),
            ),
        }
        out = {}
        for name, (criterion, options, extra) in arms.items():
            _, executor, _ = multi._step(
                model,
                mine,
                criterion(),
                mbs=1,
                domain=domain,
                status=status,
                extra=extra,
                **options,
            )
            stats = executor.stats
            depths = {key.global_index: d for key, d in stats.depths.items()}
            assert stats.exit_counts == exit_depth_counts(depths.values(), multi.R_MAX)
            padded = sum(int(batch.padding_mask.sum()) for batch in mine)
            assert (stats.padded_tokens, stats.issued_tokens) == (padded, per_rank * SEQ)
            out[name] = (stats.exit_counts, sorted(depths.items()), padded)
        return out
    finally:
        cpu_worker_teardown()


@spawned
@pytest.mark.parametrize("world", [2, 4])
def test_exit_histogram_is_identical_on_every_eu(world):
    """W = 2 and 4: the histogram is plan-derived, equal on every rank, sums to the step's
    samples (4 per EU) and bins the realized depths; waves do not change it for known depths;
    each EU counts the padding of its own home rows."""
    results = run_spawned(_multi_eu_worker, world, timeout=900.0)
    samples = 4 * world
    for arm in ("sampled", "sampled_waves2", "online"):
        counts, depths, _ = results[0][arm]
        assert all(result[arm][:2] == (counts, depths) for result in results), arm
        assert sum(counts) == len(depths) == samples
        assert counts == _histogram([d for _, d in depths], base.R_MAX)
    assert results[0]["sampled"][:2] == results[0]["sampled_waves2"][:2]
    assert len(set(d for _, d in results[0]["sampled"][1])) > 1
    assert len(set(d for _, d in results[0]["online"][1])) > 1
    padded = [result["sampled"][2] for result in results]
    assert sum(padded) == (samples // 2) * (SEQ // 4)


# ---------------------------------------------------------------------------
# DepthDistribution: the measured histogram file
# ---------------------------------------------------------------------------


def _line(iteration, counts, **overrides):
    record = dict(
        format=EXIT_HISTOGRAM_FORMAT,
        iteration=iteration,
        R_max=len(counts),
        counts=counts,
        samples=sum(counts),
        padding_fraction=0.0,
        phase=dict(seq_length=SEQ, cp=1, gpus=1, ep=1, samples_per_eu=sum(counts), waves=1),
    )
    record.update(overrides)
    return json.dumps(record)


def _file(tmp_path, *lines):
    path = tmp_path / "exit_hist.jsonl"
    path.write_text("".join(line + "\n" for line in lines))
    return str(path)


def test_histogram_file_sums_a_window_of_iterations(tmp_path):
    """The pmf of the selected iterations' summed counts; a re-recorded iteration (a resumed
    run) keeps its last line; blank lines are skipped."""
    path = _file(
        tmp_path,
        _line(1, [8, 0, 0, 0]),
        _line(2, [5, 5, 5, 5]),
        _line(2, [1, 2, 3, 2]),
        "",
        _line(3, [3, 2, 1, 2]),
        _line(4, [0, 0, 0, 8]),
    )
    window = DepthDistribution.from_histogram_file(path, first_iteration=2, last_iteration=3)
    assert (window.kind, window.max_rounds) == ("pmf", 4)
    assert window.params == (4 / 16, 4 / 16, 4 / 16, 4 / 16)
    assert window.name == "histogram:exit_hist.jsonl[2..3]"
    whole = DepthDistribution.from_histogram_file(path, max_rounds=4)
    assert whole.params == (12 / 32, 4 / 32, 4 / 32, 12 / 32)
    assert whole.mean() == pytest.approx((12 + 8 + 12 + 48) / 32)
    assert DepthDistribution.from_histogram_file(path, last_iteration=1).params == (1, 0, 0, 0)


@pytest.mark.parametrize(
    "lines,kwargs,match",
    [
        ([_line(1, [1, 1, 1, 1])], dict(max_rounds=3), "R_max 4 in the file, the run has R_max 3"),
        ([_line(1, [1, 1, 1, 1]), _line(2, [1, 1])], {}, "mixed R_max"),
        ([_line(1, [1, 1, 1, 1])], dict(first_iteration=2), "no exit-histogram line"),
        ([_line(1, [0, 0])], {}, "no sample"),
        ([_line(1, [1, 1], R_max=3)], {}, "do not fit R_max"),
        ([_line(1, [1, -1])], {}, "do not fit R_max"),
        ([_line(1, [1, 1], format="other/1")], {}, "not an exit-histogram line"),
    ],
)
def test_histogram_file_refusals(tmp_path, lines, kwargs, match):
    with pytest.raises(ValueError, match=match):
        DepthDistribution.from_histogram_file(_file(tmp_path, *lines), **kwargs)


def test_depth_distributions_draw_like_the_cost_model():
    """``draw`` equals the cost model's ``dsv4.depths_for`` (toolkit 1afb272, docs/benchmarks/
    loop_transformer/rsbf-design/cost-model/dsv4/dsv4.py) on its first 16 draws of seeds 1000
    and 1001; a measured pmf equal to R4-unif draws as R4-unif; means and pmf validation."""
    expected = {
        (R4_GEOM, 1000): [3, 1, 4, 1, 2, 2, 2, 2, 4, 2, 2, 4, 3, 2, 1, 1],
        (R4_GEOM, 1001): [2, 3, 3, 1, 2, 3, 4, 4, 1, 4, 1, 3, 1, 2, 2, 2],
        (R4_UNIF, 1000): [4, 3, 1, 2, 2, 3, 4, 1, 3, 2, 2, 1, 3, 1, 2, 1],
        (R4_UNIF, 1001): [4, 1, 4, 4, 1, 4, 4, 2, 2, 3, 2, 3, 3, 1, 3, 3],
        (R4_SKEW, 1000): [4, 4, 1, 3, 3, 4, 4, 2, 4, 3, 4, 2, 4, 3, 3, 1],
        (R4_SKEW, 1001): [4, 1, 4, 4, 1, 4, 4, 3, 3, 4, 3, 4, 4, 1, 4, 4],
    }
    for (dist, seed), draws in expected.items():
        assert dist.draw(16, seed) == draws, (dist.name, seed)
    measured = DepthDistribution.pmf([16 / 64] * 4, "measured")
    assert measured.draw(64, 1000) == R4_UNIF.draw(64, 1000)
    p = 0.456
    assert R4_GEOM.mean() == pytest.approx((1 - (1 - p) ** 4) / p)
    assert sum(R4_GEOM.probabilities()) == pytest.approx(1.0)
    assert R4_UNIF.mean() == pytest.approx(2.5) and R4_SKEW.mean() == pytest.approx(3.2)
    with pytest.raises(ValueError, match="sum to 1"):
        DepthDistribution.pmf([0.5, 0.4])
    with pytest.raises(ValueError, match="0 < p <= 1"):
        DepthDistribution.geom(0.0, 4)
    assert DepthDistribution.geom(1.0, 3).draw(5, 0) == [1] * 5
