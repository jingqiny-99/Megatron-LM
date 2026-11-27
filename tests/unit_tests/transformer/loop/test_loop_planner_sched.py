# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the RSBF planner's schedule simulator (megatron/core/transformer/loop/planner_sched.py),
the port of the toolkit cost model's ``sched.simulate`` and ``pp_core_baseline.rsbf_waves``.

Pure host code: no process group, no device. The reference constants are the cost model's own
outputs (toolkit ``docs/benchmarks/loop_transformer/rsbf-design/cost-model/dsv4`` at toolkit
commit 1afb272, probe ``$IMPL/s53/ref_sched.py``); the port must reproduce them exactly.
"""

import pytest

from megatron.core.transformer.loop.planner import R4_GEOM, R4_UNIF, DepthDistribution
from megatron.core.transformer.loop.planner_sched import (
    choose_policy,
    draw_depths,
    run_step_schedule,
    simulate,
)

UNIT = dict(f=1.0, b=3.0, coda=1.0, pre=0.75, sweep=0.25)

# The X2 pick of the cost-model cross-check (DSV4-PRO-192E, 4K, r4_unif, N = EP = 64, n = 8,
# block 'w 6', one wave, moves AEM): the final charged costs and active scale that
# dsv4.evaluate_moves passes to run_sched under pp_core_baseline.rsbf_waves(1), and the results.
X2_D = {
    "f": 318.5241059576524,
    "b": 928.4310143867531,
    "coda": 126.92866770686365,
    "pre": 51.36038931533865,
    "sweep": 17.69752204255441,
}
X2_ACTIVE_SCALE = {
    "f": 0.3784476507696734,
    "b": 0.3895110064568649,
    "coda": 0.26380674583909874,
    "pre": 0.39117272156003574,
}
# (z, policy, waves, domain_eus, peak rule) -> (makespan ms, peak units / EU, peak sample-rounds / EU)
X2_RUNS = {
    (64, "bulk_only", 1, None, "max"): (27722.92331391447, 40.328125, 20.328125),
    (64, "bulk_only", 2, None, "max"): (28888.1340024959, 22.265625, 10.3125),
    (1, "fill", 1, None, "max"): (27036.243177737942, 43.21875, 20.328125),
    (8, "bulk_only", 1, 32, "max"): (27560.400602656875, 41.03125, 20.625),
    (64, "bulk_only", 1, None, "avg"): (27722.92331391447, 40.3125, 20.3125),
}


def test_one_sample_by_hand():
    """Sweep 0.25, one round 1, coda 1, core item 3, prelude item 0.75: no idle time. Memory:
    the sample's state (1) + its exit state (1), then the chain record (3) replaces the exit
    state: peak 4 units; one stashed sample-round from its sweep to its item."""
    out = simulate([1], 1, z=1, policy="fill", d=UNIT, sync=0.0, sr_track=0)
    assert out["makespan"] == 6.0 == out["work"]
    assert out["eff"] == 1.0
    assert out["peak_units_per_eu"] == 4
    assert out["peak_sr_per_eu"] == 1


def test_two_samples_one_eu_never_idle():
    """One EU, depths (2, 1), no sync: every task runs back to back (the longest remaining
    chain first), so the makespan is the work: 9.75 + 5.75."""
    d = dict(f=1.0, b=3.0, coda=1.0, pre=0.5, sweep=0.25)
    out = simulate([2, 1], 1, z=1, policy="fill", d=d, sync=0.0)
    assert out["makespan"] == 15.5 == out["work"]
    assert "peak_sr_per_eu" not in out


@pytest.mark.parametrize("key", sorted(X2_RUNS, key=str))
def test_reference_schedules(key):
    """The X2 pick's schedule (and its two-wave, z = 1 fill, two-domain and average-peak
    variants) equals the cost model's to the last bit."""
    z, policy, waves, domain_eus, rule = key
    out = run_step_schedule(
        R4_UNIF,
        8.0,
        64,
        z,
        X2_D,
        waves=waves,
        trials=2,
        policy=policy,
        active_scale=X2_ACTIVE_SCALE,
        sr_track=0,
        domain_eus=domain_eus,
        peak_rule=rule,
    )
    assert (out["makespan"], out["peak"], out["peak_sr"]) == X2_RUNS[key]
    assert out["nsamp"] == 512 and out["Wsim"] == 64
    assert out["depths"] == [R4_UNIF.draw(512, 1000), R4_UNIF.draw(512, 1001)]


def test_waves_add_makespans_and_take_the_max_peak():
    """Each wave is a full pass over a consecutive slice of the draw."""
    ds = R4_GEOM.draw(32, 1000)
    kw = dict(z=4, policy="bulk_only", d=X2_D, sync=0.3, active_scale=X2_ACTIVE_SCALE, sr_track=0)
    halves = [simulate(ds[:16], 8, **kw), simulate(ds[16:], 8, **kw)]
    out = run_step_schedule(
        R4_GEOM,
        4.0,
        8,
        4,
        X2_D,
        waves=2,
        trials=1,
        policy="bulk_only",
        active_scale=X2_ACTIVE_SCALE,
        sr_track=0,
        domain_eus=None,
    )
    assert out["makespan"] == halves[0]["makespan"] + halves[1]["makespan"]
    assert out["peak"] == max(h["peak_units_per_eu"] for h in halves)
    assert out["peak_sr"] == max(h["peak_sr_per_eu"] for h in halves)


def test_trials_average_the_makespan_and_max_or_average_the_peaks():
    runs = [
        simulate(R4_UNIF.draw(16, 1000 + s), 4, z=2, policy="fill", d=UNIT, sync=0.3)
        for s in range(3)
    ]
    kw = dict(waves=1, trials=3, policy="fill", active_scale=None, sr_track=None, domain_eus=None)
    peak_max = run_step_schedule(R4_UNIF, 4.0, 4, 2, UNIT, peak_rule="max", **kw)
    peak_avg = run_step_schedule(R4_UNIF, 4.0, 4, 2, UNIT, peak_rule="avg", **kw)
    assert peak_max["makespan"] == sum(r["makespan"] for r in runs) / 3 == peak_avg["makespan"]
    assert peak_max["peak"] == max(r["peak_units_per_eu"] for r in runs)
    assert peak_avg["peak"] == sum(r["peak_units_per_eu"] for r in runs) / 3
    assert peak_max["peak_sr"] is None


def test_domains_are_independent_slices():
    ds = R4_UNIF.draw(32, 7)
    kw = dict(z=2, policy="bulk_only", d=X2_D, sync=0.3, active_scale=X2_ACTIVE_SCALE, sr_track=0)
    whole = simulate(ds, 8, domain_W=4, **kw)
    parts = [simulate(ds[:16], 4, **kw), simulate(ds[16:], 4, **kw)]
    assert whole["makespan"] == max(p["makespan"] for p in parts)
    assert whole["work"] == parts[0]["work"] + parts[1]["work"]
    assert whole["peak_units_per_eu"] == max(p["peak_units_per_eu"] for p in parts)
    assert whole["peak_sr_per_eu"] == max(p["peak_sr_per_eu"] for p in parts)
    # domain_W >= W is one shared simulation.
    assert simulate(ds, 8, domain_W=8, **kw) == simulate(ds, 8, **kw)


def test_choose_policy_takes_the_shorter_nominal_schedule():
    kw = dict(waves=1, trials=2, active_scale=X2_ACTIVE_SCALE, domain_eus=None)
    policy, res = choose_policy(R4_UNIF, 8.0, 64, 64, X2_D, **kw)
    bulk = run_step_schedule(R4_UNIF, 8.0, 64, 64, X2_D, policy="bulk_only", sr_track=None, **kw)
    fill = run_step_schedule(R4_UNIF, 8.0, 64, 64, X2_D, policy="fill", sr_track=None, **kw)
    expected = "bulk_only" if bulk["makespan"] <= fill["makespan"] else "fill"
    assert policy == expected
    assert res["makespan"] == min(bulk["makespan"], fill["makespan"])
    # z = 1 has one policy.
    assert choose_policy(R4_UNIF, 8.0, 4, 1, X2_D, **kw)[0] == "fill"


def test_cache_returns_the_memoized_result():
    cache = {}
    kw = dict(waves=1, trials=1, policy="fill", active_scale=None, sr_track=None, domain_eus=None)
    first = run_step_schedule(R4_UNIF, 2.0, 2, 1, UNIT, cache=cache, **kw)
    assert run_step_schedule(R4_UNIF, 2.0, 2, 1, UNIT, cache=cache, **kw) is first
    assert len(cache) == 1


def test_bulk_only_runs_a_backward_sub_round_in_bulk_rounds():
    """With more active samples than EUs, bulk_only interleaves backward sub-rounds into the
    forward rounds; fill does not, so the two schedules differ."""
    ds = [1] * 16 + [4] * 16
    kw = dict(z=2, d=UNIT, sync=0.3, active_scale=None)
    assert (
        simulate(ds, 4, policy="bulk_only", **kw)["makespan"]
        != simulate(ds, 4, policy="fill", **kw)["makespan"]
    )


def test_rejections():
    with pytest.raises(ValueError, match="unknown schedule policy"):
        simulate([1], 1, z=1, policy="strict", d=UNIT, sync=0.0)
    with pytest.raises(ValueError, match="lockstep units"):
        simulate([1, 1, 1], 3, z=2, policy="fill", d=UNIT, sync=0.0)
    with pytest.raises(ValueError, match="waves"):
        run_step_schedule(
            R4_UNIF,
            3.0,
            2,
            1,
            UNIT,
            waves=4,
            trials=1,
            policy="fill",
            active_scale=None,
            sr_track=None,
            domain_eus=None,
        )
    with pytest.raises(ValueError, match="peak rule"):
        run_step_schedule(
            R4_UNIF,
            1.0,
            1,
            1,
            UNIT,
            waves=1,
            trials=1,
            policy="fill",
            active_scale=None,
            sr_track=None,
            domain_eus=None,
            peak_rule="median",
        )


def test_draw_depths_is_the_distribution_draw():
    dist = DepthDistribution.pmf((0.5, 0.0, 0.5), "x")
    assert draw_depths(dist, 10, 3) == dist.draw(10, 3)
    assert set(draw_depths(dist, 200, 4)) == {1, 3}
