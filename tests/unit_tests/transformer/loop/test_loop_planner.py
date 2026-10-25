# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the RSBF planner v1 (megatron/core/transformer/loop/planner.py): the option menu,
the selection rule, refusals, form policies, the plan and its application to a config, the
decision table and the JSON / digest outputs. The cost-model numbers themselves are checked in
test_loop_planner_costs.py and test_loop_planner_cost_model.py.

Pure host code: no process group; the guard-acceptance tests build CPU ``TransformerConfig``s.
"""

import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch.nn.functional as F

from megatron.core.transformer.loop import planner as P
from megatron.core.transformer.loop import planner_costs as C
from megatron.core.transformer.loop.guards import validate_rsbf_config
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig

TINY_DIMS = C.DSv4Dims(
    H=512,
    V=100352,
    NEXP=8,
    TOPK=2,
    FFN_E=2048,
    FFN_S=0,
    HEADS=8,
    DH=16,
    QL=192,
    OG=8,
    OL=1024,
    IDX_H=64,
    IDX_D=128,
    IDX_TOPK=512,
    WIN=128,
    NS=4,
)
# DSV4-TINY: HEHECEHECEHECE/WE = the DSV4-PRO-PROXY layer pattern (1 / 4 / 2 + MTP) at width 512.
TINY = C.ModelShape.from_dims(TINY_DIMS, (1, 4, 2), C.proxy7_ratios())
# An 8-layer core of the same family (above the D28 small-rung bound).
CORE8 = C.ModelShape.from_dims(TINY_DIMS, (1, 8, 2), [128] + [128, 4] * 5 + [0])
CORE1 = C.ModelShape.from_dims(TINY_DIMS, (1, 1, 2), [128, 4, 128, 4, 0])
CORE5 = C.ModelShape.from_dims(TINY_DIMS, (1, 5, 2), [128] + [128, 4, 128, 4, 128] + [4, 128] + [0])
H100 = P.PlannerHardware.preset("h100")
GB200 = P.PlannerHardware.preset("gb200")


def _dense_shape(core=2):
    """The dense '*-' loop of test_loop_guards.py (hidden 64, 2 heads x 32, MLP 128): plain
    attention + dense MLP, no MoE, no mHC, prelude / coda of one layer each."""
    dims = C.DSv4Dims(
        H=64, V=128, NEXP=0, TOPK=0, FFN_E=128, FFN_S=0, HEADS=4, DH=16, QL=0, OG=0, OL=0,
        IDX_H=0, IDX_D=0, IDX_TOPK=0, WIN=128, NS=1, KVH=4, FFN_D=128, MLP_MATS=2,
    )  # fmt: skip
    layer = 4 * 64 * 64 + 2 * 64 * 128 + 2 * 64
    params = C.ParamGroups(
        emb_head=2 * 128 * 64,
        prelude_dense=layer,
        core_dense=core * layer + 2 * 64 * 64 + 64,
        coda_dense=layer,
        mtp_dense=0,
        routed_experts=0,
        largest_dense_layer=layer,
        dense_by_kind={"ATT": layer},
    )
    return C.ModelShape(
        dims=dims,
        prelude=(("ATT", "mlp"),),
        core=(("ATT", "mlp"),) * core,
        coda=(("ATT", "mlp"),),
        mtp=(),
        params=params,
        act_bytes_per_token_layer=2e4,
        act_bytes_per_token_mtp=0.0,
        mla_up_bytes_per_token=0.0,
        elem_bytes_per_token=1e3,
        mhc_bytes_per_token=0.0,
    )


def _tiny_phase(**overrides):
    fields = dict(
        seq_length=4096,
        gpus=8,
        ep=8,
        samples_per_eu=8,
        max_rounds=4,
        lockstep_eus=8,
        microbatches_per_step=8,
        precision="bf16",
        graphs="eager",
        ep_overlap="raw",
    )
    fields.update(overrides)
    return P.PhaseGeometry(**fields)


_CACHE = {}


def _tiny_plan(**planner_kw):
    key = tuple(sorted(planner_kw.items(), key=lambda kv: kv[0]))
    if key not in _CACHE:
        kw = dict(planner_kw)
        hw = kw.pop("hardware", H100)
        phase = kw.pop("phase", _tiny_phase())
        depth = kw.pop("depth", P.R4_UNIF)
        _CACHE[key] = P.LoopRSBFPlanner(TINY, hw, **kw).plan(phase, depth)
    return _CACHE[key]


def _labels(results):
    return [(r.option.label, r.option.waves, r.option.moves, r.option.layout) for r in results]


# ----------------------------------------------------------------------------- inputs
def test_hardware_presets():
    assert GB200.hbm_cap_bytes == 175e9 and GB200.grace_bytes == 200e9
    assert GB200.grace_margin_bytes == 10e9 and GB200.ep_reserve_bytes == 20e9
    assert H100.hbm_cap_bytes == 75e9 and H100.grace_bytes == 0.0 and H100.ep_reserve_bytes == 8e9
    assert H100.peak_flops == 989e12 and H100.mfu_gemm == GB200.mfu_gemm
    custom = P.PlannerHardware.preset("gb200", planner_reserve_bytes=15e9, grace_bytes=100e9)
    assert custom.hbm_cap_bytes == 165e9 and custom.grace_bytes == 100e9
    with pytest.raises(ValueError, match="unknown planner hardware"):
        P.PlannerHardware.preset("tpu")


def test_phase_geometry():
    phase = P.PhaseGeometry(seq_length=65536, gpus=64, ep=64, samples_per_eu=8, max_rounds=4, cp=16)
    assert (phase.eus, phase.tokens_per_gpu, phase.lockstep, phase.domain_eus) == (4, 4096, 4, None)
    assert phase.dense_axis == 64 and phase.microbatches == 8
    assert dataclasses.replace(phase, lockstep_eus=2).lockstep == 2
    assert dataclasses.replace(phase, domain_gpus=32).domain_eus == 2
    for bad in (
        dict(gpus=63),
        dict(lockstep_eus=3),
        dict(domain_gpus=32, ep=64, cp=1),
        dict(samples_per_eu=0),
        dict(ep=128),  # EP larger than the GPUs
        dict(ep=48),  # EP not dividing the GPUs
        dict(domain_gpus=48, ep=32),  # EP not dividing a domain
        dict(microbatches_per_step=0),
        dict(microbatches_per_step=3),  # n = 8 is not MBS x 3
    ):
        with pytest.raises(P.PlannerUnsupported):
            dataclasses.replace(phase, **bad).validate()


def test_depth_distribution_from_config():
    sampled = SimpleNamespace(
        loop_iterations=4,
        loop_exit_criterion="sampled",
        loop_depth_probs=[0.1, 0.2, 0.3, 0.4],
        loop_min_rounds=3,
    )
    dist = P.DepthDistribution.from_config(sampled)
    assert dist.probabilities() == pytest.approx((0.0, 0.0, 0.6, 0.4))
    assert dist.name.startswith("config:sampled(")
    plain = P.DepthDistribution.from_config(
        SimpleNamespace(
            loop_iterations=4, loop_exit_criterion="sampled", loop_depth_probs=[0.25] * 4
        )
    )
    assert plain.params == (0.25, 0.25, 0.25, 0.25)
    fixed = P.DepthDistribution.from_config(SimpleNamespace(loop_iterations=3))
    assert fixed.params == (0.0, 0.0, 1.0) and fixed.mean() == 3.0
    assert set(fixed.draw(50, 1)) == {3}
    with pytest.raises(ValueError, match="measured exit histogram"):
        P.DepthDistribution.from_config(
            SimpleNamespace(loop_iterations=4, loop_exit_criterion="online")
        )


def test_capabilities_and_option_requirements():
    target = P.PlannerCapabilities.target()
    assert target.layer_window and target.grace_state_tier and target.optimizer_offload
    today = P.PlannerCapabilities()
    assert not today.layer_window and not today.optimizer_offload and today.coda_recompute
    opt = P.PlannerOption("block", 4, False, True, 1, "A", "resident")
    assert opt.label == "w+rc 4"
    assert opt.requires() == {"optimizer_offload", "coda_recompute"}
    regather = P.PlannerOption("block", 2, True, False, 2, "-", "layer_window")
    assert regather.label == "r 2" and regather.requires() == {"layer_window"}
    assert P.PlannerOption("nested", None, False, False, 1, "AEM", "resident").requires() == {
        "layer_window",
        "optimizer_offload",
    }


# ----------------------------------------------------------------------------- menu
def test_menu_of_tiny_today():
    """Resident: nested + strides 1..4 x (w, w+rc), moves '-' / A; Layer window (target only):
    nested + 4 x (w, w+rc, r, r+rc), all four move steps; waves 1, 2, 4, 8 (n = 8)."""
    planner = P.LoopRSBFPlanner(TINY, H100)
    options = planner.options(_tiny_phase())
    resident = [o for o in options if o.layout == "resident"]
    window = [o for o in options if o.layout == "layer_window"]
    assert len(resident) == (1 + 4 * 2) * 2 * 4
    assert len(window) == (1 + 4 * 4) * 4 * 4
    assert not any(o.regather for o in resident)
    assert {o.moves for o in resident} == {"-", "A"}
    assert {o.waves for o in options} == {1, 2, 4, 8}
    # Evaluation order: waves, layout, form (the cost model's spec order), moves.
    assert [o.label for o in resident[:6]] == ["nested", "nested", "w 1", "w 1", "w 2", "w 2"]


def test_menu_rules():
    dense = _dense_shape(core=2)
    options = P.LoopRSBFPlanner(dense, H100, layouts=("resident",)).options(_tiny_phase())
    assert not any(o.coda_recompute for o in options)  # no DSv4 attention in the coda / MTP
    assert {o.label for o in options} == {"nested", "w 1", "w 2"}
    # s = L_core is added when the strides miss it and L_core <= max(strides) ...
    strides = P.LoopRSBFPlanner(TINY, H100, strides_layers=(1, 2, 3, 6))._strides()
    assert strides == [1, 2, 3, 4]
    assert P.LoopRSBFPlanner(CORE5, H100, strides_layers=(1, 2, 4, 6))._strides() == [1, 2, 4, 5]
    # ... but not above the largest stride (DSV4-PRO-192E keeps the reference's grid), except
    # under d28 on a small core (its applied menu is the whole round).
    unconstrained = dict(form_policy="planner")
    assert P.LoopRSBFPlanner(TINY, H100, strides_layers=(1, 2, 3), **unconstrained)._strides() == [
        1,
        2,
        3,
    ]
    assert P.LoopRSBFPlanner(TINY, H100, strides_layers=(1, 2, 3))._strides() == [1, 2, 3, 4]
    assert P.LoopRSBFPlanner(CORE8, H100, strides_layers=(1, 2, 4))._strides() == [1, 2, 4]
    # No nested item on a one-layer core (the executor needs two blocks).
    assert "nested" not in {o.label for o in P.LoopRSBFPlanner(CORE1, H100).options(_tiny_phase())}


def test_waves_must_divide_n_and_the_microbatches():
    planner = P.LoopRSBFPlanner(TINY, H100)
    phase = _tiny_phase(samples_per_eu=8, microbatches_per_step=4)  # MBS 2
    assert {o.waves for o in planner.options(phase)} == {1, 2, 4}
    assert {o.waves for o in planner.options(_tiny_phase(samples_per_eu=6))} == {1, 2}
    assert {o.waves for o in planner.options(_tiny_phase(waves_allowed=False))} == {1}
    plan = planner.plan(phase, P.R4_UNIF)
    assert any("w=8 (does not divide n = 8 / the 4 microbatches)" in n for n in plan.notes)


# ----------------------------------------------------------------------------- selection
def _synthetic(base, mfu, hbm_gb, waves=1, fits=True, applicable=True):
    option = dataclasses.replace(base.option, waves=waves)
    return dataclasses.replace(
        base,
        option=option,
        mfu=mfu,
        hbm_bytes=hbm_gb * 1e9,
        grace_bytes=0.0,
        fits=fits,
        applicable=applicable,
    )


def test_selection_rule():
    """Highest round(mfu, 5), then the smallest HBM + Grace, then fewer waves; the first in
    order on a full tie; unfit options never; inapplicable ones only for the target pick."""
    base = _tiny_plan().applied
    select = P.LoopRSBFPlanner.select
    a = _synthetic(base, 0.123454, 10.0)
    b = _synthetic(base, 0.1234536, 9.0)  # same rounded MFU, smaller footprint
    assert select([a, b], applicable_only=True) is b
    c = _synthetic(base, 0.123456, 30.0)  # rounds up: higher MFU wins over footprint
    assert select([a, b, c], applicable_only=True) is c
    w2 = _synthetic(base, 0.2, 10.0, waves=2)
    w1 = _synthetic(base, 0.2, 10.0, waves=1)
    assert select([w2, w1], applicable_only=True) is w1
    first, second = _synthetic(base, 0.2, 10.0), _synthetic(base, 0.2, 10.0)
    assert select([first, second], applicable_only=True) is first
    unfit = _synthetic(base, 0.9, 1.0, fits=False)
    target_only = _synthetic(base, 0.5, 1.0, applicable=False)
    assert select([unfit, a], applicable_only=True) is a
    assert select([target_only, a], applicable_only=True) is a
    assert select([target_only, a], applicable_only=False) is target_only
    assert select([unfit], applicable_only=False) is None
    # Small models: the rounding follows the menu's best MFU, so throughput still decides.
    fast = _synthetic(base, 4.4e-6, 20.0)
    slow = _synthetic(base, 2.4e-6, 10.0)
    assert select([slow, fast], applicable_only=True) is fast
    near = _synthetic(base, 4.4e-6 * (1 + 1e-7), 10.0)  # equal at the menu's precision
    assert select([fast, near], applicable_only=True) is near
    digits = [P.mfu_digits([_synthetic(base, m, 1.0)]) for m in (0.3, 0.1, 0.05, 4.4e-6)]
    assert digits == [5, 5, 6, 10]
    assert P.mfu_digits([]) == 5


def test_selection_ranks_by_throughput_on_a_small_model():
    """The dense '*-' shape with MTP at S = 512 models MFUs around 1e-6: the plan still keeps
    the fastest fitting option (an absolute 1e-5 rounding tied them all to the footprint)."""
    shape = dataclasses.replace(_dense_shape(core=4), mtp=(("ATT", "mlp"),))
    planner = P.LoopRSBFPlanner(shape, H100, layouts=("resident",), form_policy="planner")
    plan = planner.plan(_tiny_phase(seq_length=512), P.R4_UNIF)
    menu = [r for r in plan.table if r.fits and r.applicable]
    assert max(r.mfu for r in menu) < 1e-4
    assert plan.applied.mfu == max(r.mfu for r in menu)
    assert plan.applied.step_ms == min(r.step_ms for r in menu)


# ----------------------------------------------------------------------------- refusals
def test_refusal_reasons_today():
    """h100, today's capabilities: Adam to Grace is refused for the Grace tier and for the
    trainer's reason, Layer-window options need M3, and the D28 policy excludes the strides."""
    caps = P.PlannerCapabilities(optimizer_offload_reason="--use-precision-aware-optimizer not set")
    plan = _tiny_plan(capabilities=caps)
    adam = next(r for r in plan.table if r.option.moves == "A" and r.option.layout == "resident")
    assert "no Grace tier" in adam.refusals
    assert (
        "needs Adam to Grace (excluded: --use-precision-aware-optimizer not set)" in adam.refusals
    )
    assert adam.missing == ("optimizer_offload",) and not adam.applicable
    window = next(r for r in plan.table if r.option.layout == "layer_window")
    assert "needs Layer window (M3)" in window.refusals and "layer_window" in window.missing
    stride2 = next(r for r in plan.table if r.option.label == "w 2" and r.option.moves == "-")
    assert stride2.fits and stride2.applicable
    assert stride2.refusals == ("excluded by form policy d28",)
    assert "Adam to Grace excluded: --use-precision-aware-optimizer not set" in plan.notes


def test_hbm_over_cap_and_round_state_overflow():
    """Over the cap; and round-state data that fits only by spilling to Grace, which no run has
    today: refused as HBM over cap (round-state data), fitting for the target."""
    small = P.PlannerHardware.preset("h100", hbm_bytes=10e9)
    planner = P.LoopRSBFPlanner(TINY, small, layouts=("resident",))
    results = planner.evaluate(_tiny_phase(), P.R4_UNIF)
    assert all(not r.fits for r in results)
    assert all(r.refusals[0].startswith("HBM over cap by ") for r in results)
    with pytest.raises(P.PlannerNoFit) as excinfo:
        planner.plan(_tiny_phase(), P.R4_UNIF)
    assert len(excinfo.value.table) == len(results)
    # A cap between the rest and the rest + the round-state data, Grace available.
    probe = P.LoopRSBFPlanner(TINY, GB200, layouts=("resident",), waves=(1,))
    whole = next(
        r
        for r in probe.evaluate(_tiny_phase(), P.R4_UNIF)
        if r.option.label == "w 4" and r.option.moves == "-"
    )
    rest = whole.hbm_bytes - whole.spill_hbm_bytes
    hw = P.PlannerHardware.preset(
        "gb200", hbm_bytes=rest + GB200.planner_reserve_bytes + whole.spill_hbm_bytes / 2
    )
    planner = P.LoopRSBFPlanner(TINY, hw, layouts=("resident",), waves=(1,))
    spilled = next(
        r
        for r in planner.evaluate(_tiny_phase(), P.R4_UNIF)
        if r.option.label == "w 4" and r.option.moves == "-"
    )
    assert spilled.spill_grace_bytes == pytest.approx(whole.spill_hbm_bytes / 2)
    assert spilled.fits and not spilled.applicable and spilled.missing == ("grace_state_tier",)
    assert spilled.refusals[0].startswith("HBM over cap by ")
    assert spilled.refusals[0].endswith("(round-state data; no Grace tier for it)")
    assert spilled.c2c_write > 0 and spilled.charged_share > 0
    # The same spill over a slow C2C link is refused by the C2C rule.
    slow = dataclasses.replace(hw, c2c_bytes_per_s=1e6)
    planner = P.LoopRSBFPlanner(
        TINY, slow, layouts=("resident",), waves=(1,), capabilities=P.PlannerCapabilities.target()
    )
    choked = next(
        r
        for r in planner.evaluate(_tiny_phase(), P.R4_UNIF)
        if r.option.label == "w 4" and r.option.moves == "-"
    )
    assert not choked.fits and any(x.startswith("C2C need ") for x in choked.refusals)


def test_round_state_overflow_without_grace_hardware():
    """h100 (no Grace tier): round-state data over the cap is refused as HBM over cap by the
    spilled bytes (named first), not as "no Grace tier", which stays for Adam to Grace."""
    kw = dict(layouts=("resident",), waves=(1,))
    whole = next(
        r
        for r in P.LoopRSBFPlanner(TINY, H100, **kw).evaluate(_tiny_phase(), P.R4_UNIF)
        if r.option.label == "w 4" and r.option.moves == "-"
    )
    rest = whole.hbm_bytes - whole.spill_hbm_bytes
    hw = P.PlannerHardware.preset(
        "h100", hbm_bytes=rest + H100.planner_reserve_bytes + whole.spill_hbm_bytes / 2
    )
    caps = P.PlannerCapabilities(optimizer_offload=True)
    results = P.LoopRSBFPlanner(TINY, hw, capabilities=caps, **kw).evaluate(
        _tiny_phase(), P.R4_UNIF
    )
    spilled = next(r for r in results if r.option.label == "w 4" and r.option.moves == "-")
    assert spilled.spill_grace_bytes == pytest.approx(whole.spill_hbm_bytes / 2)
    assert spilled.adam_grace_bytes == 0 and not spilled.fits
    assert spilled.refusals[0] == (
        f"HBM over cap by {spilled.spill_grace_bytes / 1e9:.2f} GB (round-state data; no Grace "
        "tier for it)"
    )
    assert "no Grace tier" not in spilled.refusals
    adam = next(r for r in results if r.option.label == "w 4" and r.option.moves == "A")
    assert adam.adam_grace_bytes > 0 and "no Grace tier" in adam.refusals


def test_grace_refusals_exclude_a_stuck_spill():
    """Round-state data that would spill to Grace without a Grace tier stays in HBM (refused
    as HBM over cap): the Grace capacity / margin refusals do not count it."""
    probe = P.LoopRSBFPlanner(TINY, GB200, layouts=("resident",), waves=(1,))
    whole = next(
        r
        for r in probe.evaluate(_tiny_phase(), P.R4_UNIF)
        if r.option.label == "w 4" and r.option.moves == "-"
    )
    rest = whole.hbm_bytes - whole.spill_hbm_bytes
    spill = whole.spill_hbm_bytes / 2
    hw = P.PlannerHardware.preset(
        "gb200",
        hbm_bytes=rest + GB200.planner_reserve_bytes + spill,
        grace_bytes=GB200.grace_margin_bytes + spill / 2,  # the spill would break the margin
    )
    planner = P.LoopRSBFPlanner(TINY, hw, layouts=("resident",), waves=(1,))
    stuck = next(
        r
        for r in planner.evaluate(_tiny_phase(), P.R4_UNIF)
        if r.option.label == "w 4" and r.option.moves == "-"
    )
    assert stuck.spill_grace_bytes == pytest.approx(spill) and not stuck.fits
    assert stuck.refusals == (
        f"HBM over cap by {spill / P.GB:.2f} GB (round-state data; no Grace tier for it)",
    )


# ----------------------------------------------------------------------------- shapes
@pytest.mark.parametrize("which", ["no_mtp", "no_prelude", "dense"])
def test_plan_without_mtp_or_prelude(which):
    """Empty roles (no MTP layer, no prelude) have no cost components: the planner evaluates
    and plans them, with both layouts."""
    shape = {
        "no_mtp": dataclasses.replace(TINY, mtp=()),
        "no_prelude": dataclasses.replace(TINY, prelude=()),
        "dense": _dense_shape(core=4),
    }[which]
    plan = P.LoopRSBFPlanner(shape, H100).plan(_tiny_phase(), P.R4_UNIF)
    assert plan.applied.fits and plan.applied.applicable
    assert plan.target is not None
    assert all(r.step_ms > 0 and r.mfu > 0 for r in plan.table)
    assert P.format_decision_table(plan)


def test_shape_needs_every_kind_in_dense_by_kind():
    params = dataclasses.replace(TINY.params, dense_by_kind={"CSA": 1.0, "HCA": 1.0})
    with pytest.raises(P.PlannerUnsupported, match=r"no entry for layer kind\(s\) \['W'\]"):
        dataclasses.replace(TINY, params=params)


def test_empty_menu_names_the_waves():
    with pytest.raises(P.PlannerNoFit, match=r"menu is empty .*w=2 \(waves need a fixed"):
        P.LoopRSBFPlanner(TINY, H100, waves=(2, 4)).plan(
            _tiny_phase(waves_allowed=False), P.R4_UNIF
        )
    with pytest.raises(P.PlannerNoFit, match=r"w=3 \(does not divide n = 8"):
        P.LoopRSBFPlanner(TINY, H100, waves=(3,)).plan(_tiny_phase(), P.R4_UNIF)


# ----------------------------------------------------------------------------- policies
def test_form_policies():
    d28 = _tiny_plan()
    assert d28.form_policy == "d28" and d28.core_layers == 4
    assert (d28.item_form, d28.block_stride, d28.stride_layers) == ("whole_round", None, 4)
    assert d28.applied.option.layout == "resident" and d28.applied.applicable
    assert d28.raw is not None and d28.raw.applicable and d28.raw.option.layout == "resident"
    assert round(d28.raw.mfu, 5) >= round(d28.applied.mfu, 5)
    unconstrained = _tiny_plan(form_policy="planner")
    assert unconstrained.applied is unconstrained.raw
    nested = _tiny_plan(form_policy="nested")
    assert (nested.item_form, nested.block_stride) == ("nested", 2)
    # d28 offers the whole round on a small core whatever the configured strides.
    few = P.LoopRSBFPlanner(TINY, H100, layouts=("resident",), strides_layers=(1, 2))
    assert few._strides() == [1, 2, 4]
    assert few.plan(_tiny_phase(), P.R4_UNIF).item_form == "whole_round"
    # A policy that excludes every fitting option says so and names the raw pick.
    with pytest.raises(P.PlannerNoFit, match="form policy nested excluded every fitting option"):
        P.LoopRSBFPlanner(CORE1, H100, layouts=("resident",), form_policy="nested").plan(
            _tiny_phase(), P.R4_UNIF
        )
    # Above the small-rung bound d28 leaves the menu alone.
    phase = _tiny_phase()
    big = P.LoopRSBFPlanner(CORE8, H100, layouts=("resident",)).plan(phase, P.R4_UNIF)
    assert big.applied is big.raw
    assert not any("excluded by form policy" in r.refusals for r in big.table)


def test_schedule_policy():
    """executor: bulk_only at z > 1, fill at z = 1 (what the executor runs); min_makespan: the
    cost model's per-form choice."""
    plan = _tiny_plan()
    assert {r.policy for r in plan.table} == {"bulk_only"}
    one = P.LoopRSBFPlanner(TINY, H100, layouts=("resident",), waves=(1,)).evaluate(
        _tiny_phase(gpus=1, ep=1, lockstep_eus=None), P.R4_UNIF
    )
    assert {r.policy for r in one} == {"fill"}
    chosen = P.LoopRSBFPlanner(
        TINY, H100, layouts=("resident",), waves=(1,), sched_policy="min_makespan"
    ).evaluate(_tiny_phase(), P.R4_UNIF)
    assert {r.policy for r in chosen} <= {"bulk_only", "fill"}


def test_nested_rebuild_fraction():
    assert P.executor_nested_rebuild_fraction(4) == 0.75
    assert P.executor_nested_rebuild_fraction(1) == 0.0
    kw = dict(layouts=("resident",), waves=(1,), move_steps=("-",))
    full = P.LoopRSBFPlanner(TINY, H100, **kw).evaluate(_tiny_phase(), P.R4_UNIF)
    part = P.LoopRSBFPlanner(TINY, H100, rebuild_fraction=0.75, **kw).evaluate(
        _tiny_phase(), P.R4_UNIF
    )
    nested_full = next(r for r in full if r.option.form == "nested")
    nested_part = next(r for r in part if r.option.form == "nested")
    assert nested_part.unit_ms["b"] < nested_full.unit_ms["b"]
    block = next(r for r in full if r.option.label == "w 2")
    assert next(r for r in part if r.option.label == "w 2").unit_ms == block.unit_ms


def test_samples_per_eu_warning_and_one_round():
    plan = P.LoopRSBFPlanner(TINY, H100, layouts=("resident",)).plan(
        _tiny_phase(samples_per_eu=2, microbatches_per_step=2), P.R4_UNIF
    )
    assert plan.warnings == ("n = 2 < 4 samples per EU (D21: n >= 4, n = 8 preferred at N = 64)",)
    one = P.LoopRSBFPlanner(TINY, H100, layouts=("resident",)).plan(
        _tiny_phase(max_rounds=1), P.DepthDistribution.pmf((1.0,), "one")
    )
    assert one.applied.fits and one.item_form == "whole_round"
    with pytest.raises(ValueError, match="R_max"):
        P.LoopRSBFPlanner(TINY, H100).evaluate(_tiny_phase(max_rounds=2), P.R4_UNIF)


def test_planner_argument_checks():
    for bad in (
        dict(form_policy="d42"),
        dict(sched_policy="greedy"),
        dict(peak_trials="median"),
        dict(move_steps=("-", "S")),
        dict(layouts=("sharded",)),
        dict(waves=(0,)),
        dict(waves=(1, -2)),
        dict(strides_layers=(0, 2)),
    ):
        with pytest.raises(ValueError):
            P.LoopRSBFPlanner(TINY, H100, **bad)


# ----------------------------------------------------------------------------- outputs
def test_decision_table_json_and_digest():
    plan = _tiny_plan()
    text = P.format_decision_table(plan)
    lines = text.splitlines()
    assert lines[0].startswith("[loop planner] phase S=4096 cp=1 N=8 EP=8 W=8 z=8 n=8")
    assert "hardware=h100 HBM cap 75.0 GB, no Grace tier" in lines[1]
    assert any("APPLIED (whole round)" in line for line in lines)
    assert any(line.startswith("  raw vs applied: ") for line in lines)
    # The resident whole round is also the fastest option overall (the Layer window adds its
    # gathers); a 20 GB cap that only the Layer window meets at w = 1 makes the target differ.
    assert "  target (M3 capabilities): same as applied" in lines
    capped = _tiny_plan(hardware=P.PlannerHardware.preset("h100", hbm_bytes=25e9))
    assert capped.target.option.layout == "layer_window" and capped.applied.option.waves == 2
    capped_lines = P.format_decision_table(capped).splitlines()
    assert "  requested but unavailable: core dense in the Layer window (M3)" in capped_lines
    assert lines[-1].startswith(
        "  plan: loop_rsbf_item_form=whole_round loop_rsbf_block_stride=None loop_rsbf_waves=1"
    )
    records = []
    log = SimpleNamespace(
        info=lambda fmt, *a: records.append(("info", fmt % a)),
        warning=lambda fmt, *a: records.append(("warning", fmt % a)),
    )
    warned = dataclasses.replace(plan, warnings=("n = 2 < 4 samples per EU",))
    warned_text = P.format_decision_table(warned)
    assert "  WARNING: n = 2 < 4 samples per EU" in warned_text.splitlines()
    assert P.log_decision_table(warned, log=log) == warned_text
    assert records == [("info", warned_text), ("warning", "loop planner: n = 2 < 4 samples per EU")]
    data = json.loads(json.dumps(plan.to_json()))
    assert data["config_overrides"] == plan.config_overrides()
    assert data["applied"]["label"] == plan.applied.option.label
    assert len(data["table"]) == len(plan.table) and data["digest"] == plan.digest
    assert data["hardware"]["name"] == "h100" and data["depth"]["name"] == "r4_unif"
    again = P.LoopRSBFPlanner(TINY, H100).plan(_tiny_phase(), P.R4_UNIF)
    assert again.digest == plan.digest
    other = P.LoopRSBFPlanner(TINY, H100).plan(_tiny_phase(), P.R4_SKEW)
    assert other.digest != plan.digest
    assert plan.config_overrides() == {
        "loop_rsbf_item_form": "whole_round",
        "loop_rsbf_block_stride": None,
        "loop_rsbf_waves": 1,
        "loop_rsbf_coda_recompute_mla_up_proj": False,
    }


# ----------------------------------------------------------------------------- application
_DSV4 = "HEHECEHECEHECE"
_RSBF = dict(loop_schedule="rsbf", calculate_per_token_loss=True)


def _dsv4_config(**overrides):
    """DSV4-TINY (as test_loop_guards.py): prelude 'HE', core 'HECEHECE', coda 'HECE'."""
    ratios = {"H": 128, "C": 4, "W": 0, "E": 0}
    fields = dict(
        num_layers=len(_DSV4),
        hidden_size=512,
        num_attention_heads=8,
        multi_latent_attention=True,
        q_lora_rank=192,
        kv_lora_rank=64,
        qk_head_dim=16,
        qk_pos_emb_head_dim=8,
        v_head_dim=16,
        rope_type="rope",
        experimental_attention_variant="dsv4_hybrid",
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=32,
        csa_window_size=32,
        csa_compress_ratios=[ratios[symbol] for symbol in _DSV4],
        num_moe_experts=8,
        moe_ffn_hidden_size=256,
        moe_router_topk=2,
        moe_token_dispatcher_type="alltoall",
        moe_num_hash_layers=1,
        hash_moe_vocab_size=128,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        is_hybrid_model=True,
        loop_architecture="huginn",
        loop_prelude_sublayers=2,
        loop_coda_sublayers=4,
        **_RSBF,
    )
    fields.update(overrides)
    return MLATransformerConfig(**fields)


def _dense_config(**overrides):
    """The dense '*-*-*-*-' loop (prelude 2, core 4, coda 2 sublayers)."""
    fields = dict(
        num_layers=8,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        gated_linear_unit=False,
        activation_func=F.gelu,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        loop_architecture="huginn",
        loop_prelude_sublayers=2,
        loop_coda_sublayers=2,
        **_RSBF,
    )
    fields.update(overrides)
    return TransformerConfig(**fields)


@pytest.mark.parametrize("which", ["tiny", "dense"])
def test_every_offered_option_passes_the_rsbf_guard(which):
    """Every applicable option's (form, stride, waves, D42) is accepted by the config guard on
    the TINY and dense '*-' configs (pair-aligned strides, whole round without a stride, nested
    stride 2)."""
    shape, config = (TINY, _dsv4_config()) if which == "tiny" else (_dense_shape(), _dense_config())
    planner = P.LoopRSBFPlanner(shape, H100, layouts=("resident",), form_policy="planner")
    forms = set()
    for option in planner.options(_tiny_phase()):
        form, stride = P.option_item_form(option, shape.core_layers)
        forms.add((form, stride, option.coda_recompute))
        for name, value in (
            ("loop_rsbf_item_form", form),
            ("loop_rsbf_block_stride", stride),
            ("loop_rsbf_waves", option.waves),
            ("loop_rsbf_coda_recompute_mla_up_proj", option.coda_recompute),
        ):
            setattr(config, name, value)
        validate_rsbf_config(config)
    expected = {("nested", 2, False), ("whole_round", None, False)}
    expected |= {("block", 2 * s, False) for s in range(1, shape.core_layers)}
    if which == "tiny":
        expected |= {(f, s, True) for f, s, _ in expected if f != "nested"}
    assert forms == expected


def test_apply_plan_to_config():
    plan = _tiny_plan(form_policy="planner")
    config = _dsv4_config()
    overrides = P.apply_plan_to_config(config, plan)
    assert overrides == plan.config_overrides()
    for name, value in overrides.items():
        assert getattr(config, name) == value
    # A rejected plan leaves the config as it was.
    before = {name: getattr(config, name) for name in overrides}
    bad = dataclasses.replace(plan, item_form="block", block_stride=99)
    with pytest.raises(ValueError, match="Loop RSBF does not support"):
        P.apply_plan_to_config(config, bad)
    assert {name: getattr(config, name) for name in overrides} == before
    with pytest.raises(P.PlannerUnsupported, match="cp = 2"):
        P.apply_plan_to_config(config, dataclasses.replace(plan, phase=_tiny_phase(cp=2)))
    # A plan made for another core size is stale: refused before touching the config.
    stale = dataclasses.replace(plan, core_layers=2)
    with pytest.raises(P.PlannerUnsupported, match="stale plan"):
        P.apply_plan_to_config(config, stale)
    assert {name: getattr(config, name) for name in overrides} == before
    with pytest.raises(P.PlannerUnsupported, match="rsbf"):
        P.apply_plan_to_config(SimpleNamespace(loop_schedule="standard"), plan)
