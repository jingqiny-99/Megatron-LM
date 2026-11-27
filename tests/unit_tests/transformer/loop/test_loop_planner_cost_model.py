# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Cost-model cross-check of the RSBF planner v1 (DSv4 plan §9.14, stage-5 design Section 5).

The planner evaluates the cost model's whole option menu on DSV4-PRO-192E (B) and
DSV4-PRO-PROXY (D) rows and must pick what the cost model picks: the same item form, stride,
variant, D42, waves and moves, with HBM / Grace within 0.002 GB, MFU within 1e-9 relative (the
unrounded value), step within 0.002 s and the simulator peaks within 1e-4.

Source of every constant: the committed toolkit cost model (``docs/benchmarks/loop_transformer/
rsbf-design/cost-model/dsv4``, toolkit commit 1afb272): ``pp_core_baseline.rsbf_job`` per wave
count (in-domain, ``PEAK_TRIALS = "max"``, two trials, margin 1, C2C 100 GB/s x 1.25), picked by
``plan_numbers.pick`` (fair moves, Grace <= 190 GB, key ``(round(mfu, 5), -(HBM + Grace),
-waves)``); probes ``$IMPL/s53/ref_rows.py`` / ``ref_picks.py`` (the full menus, unrounded) and
``ref_resident.py`` (the all-Resident rows, ``proxy_v2.run_one(core_res=True)``). The values agree
with ``results_pp_core.json`` and the plan's tables (DP §3.3, DP:206).

Settings of every X row: EP 64, N 64, one 64-GPU domain, two-chunk EP overlap, MXFP8,
whole-layer graphs, HybridEP reserve 20 GB (X4: 32 GB), Grace 200 GB, Layer-window layout, M3
capabilities, ``form_policy = "planner"``, ``sched_policy = "min_makespan"``. Pure host code.
"""

import dataclasses

import pytest

from megatron.core.transformer.loop import planner as P
from megatron.core.transformer.loop import planner_costs as C

B = C.ModelShape.from_dims(
    dataclasses.replace(C.PRO_DIMS, NEXP=192), (3, 54, 4), C.pro_ratios(3, 54, 4)
)
D = C.ModelShape.from_dims(C.PRO_DIMS, (1, 4, 2), C.proxy7_ratios())
SHAPES = {"B": B, "D": D}
STRIDES = {"B": (1, 2, 3, 4, 6, 9, 18), "D": (1, 2, 3, 4)}
DEPTHS = {"r4_geom": P.R4_GEOM, "r4_unif": P.R4_UNIF, "r4_skew": P.R4_SKEW}

# name: (model, S, cp, depth, n, reserve GB)
ROWS = {
    "X1": ("B", 4096, 1, "r4_geom", 8, 20),
    "X2": ("B", 4096, 1, "r4_unif", 8, 20),
    "X3": ("B", 4096, 1, "r4_skew", 8, 20),
    "X4": ("B", 4096, 1, "r4_skew", 8, 32),
    "X5": ("B", 8192, 1, "r4_geom", 8, 20),
    "X6": ("B", 8192, 1, "r4_skew", 8, 20),
    "X7": ("B", 4096, 1, "r4_geom", 4, 20),
    "X8": ("D", 4096, 1, "r4_unif", 8, 20),
    "X9": ("D", 8192, 1, "r4_skew", 8, 20),
    "X10": ("B", 65536, 16, "r4_unif", 8, 20),
}
# name: (spec, waves, moves, HBM GB, Grace GB, MFU, step s, peak units, peak sample-rounds),
# then the runner-up (spec, waves, moves, MFU).
PICKS = {
    "X1": (("w 6", 1, "AEM", 175.0, 163.4882014728775, 0.30729241754528014, 25.687008011313583, 34.609375, 16.609375), ("w 9", 1, "AEM", 0.30717035441911505)),  # noqa: E501
    "X2": (("w 6", 1, "AEM", 175.0, 171.8191377928775, 0.31064298580363886, 30.67215712038781, 40.328125, 20.328125), ("w 9", 1, "AEM", 0.3106162223042018)),  # noqa: E501
    "X3": (("w 9", 1, "AEM", 175.0, 179.8773489328775, 0.31966392296342006, 37.232115681942844, 45.796875, 25.796875), ("w 6", 1, "AEM", 0.3195905925967119)),  # noqa: E501
    "X4": (("r 9", 1, "AEM", 175.0, 185.8277567728775, 0.31769469956837476, 37.46289810716909, 45.796875, 25.796875), ("r+rc 9", 1, "AEM", 0.317454195843774)),  # noqa: E501
    "X5": (("w+rc 4", 4, "AEM", 175.0, 169.0426158488775, 0.2979662676503886, 54.361167591647046, 10.375, 4.375), ("w 6", 4, "AEM", 0.2978269790508267)),  # noqa: E501
    "X6": (("w 6", 4, "AEM", 175.0, 185.1227491848775, 0.314750652488976, 77.6195835223252, 12.671875, 6.671875), ("w+rc 4", 4, "AEM", 0.3145101850695743)),  # noqa: E501
    "X7": (("w 6", 1, "AEM", 167.6223282032525, 152.082731381625, 0.268008581121387, 14.802304930020632, 20.390625, 8.390625), ("w 4", 1, "AEM", 0.268008581121387)),  # noqa: E501
    "X8": (("w 2", 1, "-", 143.599796396765, 0.0, 0.3689312268599881, 2.9076965308511946, 42.328125, 20.328125), ("w 4", 1, "-", 0.3689312268599881)),  # noqa: E501
    "X9": (("w 4", 1, "-", 175.0, 0.9265335087650146, 0.37296713517302416, 6.8369445333549015, 47.796875, 25.796875), ("w 2", 1, "-", 0.37235390967675025)),  # noqa: E501
    "X10": (("w 9", 1, "AEM", 175.0, 176.1203011248775, 0.2915713625196236, 35.88168738036734, 43.25, 22.25), ("w 6", 1, "AEM", 0.2915621790103418)),  # noqa: E501
}  # fmt: skip
# The D28 pick (plan_numbers.pick with the spec filter "s = 4", the whole round on D).
D28_PICKS = {
    "X8": ("w 4", 1, "-", 144.08822720476502, 0.0, 0.3689312268599881, 2.9076965308511946),
    "X9": ("w 4", 1, "-", 175.0, 0.9265335087650146, 0.37296713517302416, 6.8369445333549015),
}
# DP:206 / plan_numbers C6 'r2res': D 4K, BF16, eager, raw EP, every dense group Resident, Adam and
# round states in HBM, no move, n 8; HBM GB of the nested item and of evaluate(whole_round=True).
RESIDENT_HBM = {
    "r4_geom": (139.155334550875, 142.900553110875, 34.609375),
    "r4_unif": (140.498560406875, 144.243778966875, 40.328125),
    "r4_skew": (141.783066006875, 145.528284566875, 45.796875),
}


def _planner(model, reserve_gb=20, **overrides):
    kw = dict(
        capabilities=P.PlannerCapabilities.target(),
        strides_layers=STRIDES[model],
        layouts=("layer_window",),
        form_policy="planner",
        sched_policy="min_makespan",
    )
    kw.update(overrides)
    hw = P.PlannerHardware.preset("gb200", ep_reserve_bytes=reserve_gb * 1e9)
    return P.LoopRSBFPlanner(SHAPES[model], hw, **kw)


def _phase(S, cp, n):
    return P.PhaseGeometry(
        seq_length=S, gpus=64, ep=64, samples_per_eu=n, max_rounds=4, cp=cp, domain_gpus=64
    )


_PLANS = {}


def _plan(name, **overrides):
    """The plan of row ``name`` (cached per module: each B row evaluates 464 options)."""
    key = (name, tuple(sorted(overrides.items())))
    if key not in _PLANS:
        model, S, cp, depth, n, reserve = ROWS[name]
        _PLANS[key] = _planner(model, reserve, **overrides).plan(_phase(S, cp, n), DEPTHS[depth])
    return _PLANS[key]


def _spec(result):
    o = result.option
    return (o.label, o.waves, o.moves)


@pytest.mark.parametrize("name", list(ROWS))
def test_planner_picks_what_the_cost_model_picks(name):
    plan = _plan(name)
    (spec, w, moves, hbm, grace, mfu, step, peak, peak_sr), runner = PICKS[name]
    pick = plan.applied
    assert plan.raw is pick and plan.target is pick  # unconstrained, every capability
    assert _spec(pick) == (spec, w, moves)
    assert pick.option.layout == "layer_window" and pick.fits and pick.applicable
    assert abs(pick.hbm_bytes / 1e9 - hbm) <= 0.002
    assert abs(pick.grace_bytes / 1e9 - grace) <= 0.002
    assert pick.mfu == pytest.approx(mfu, rel=1e-9, abs=0)
    assert abs(100 * pick.mfu - 100 * mfu) <= 0.0002
    assert abs(pick.step_ms / 1e3 - step) <= 0.002
    assert abs(pick.peak_units - peak) <= 1e-4
    assert abs(pick.peak_sample_rounds - peak_sr) <= 1e-4
    # The runner-up (diagnostic: what a sloppy port would have picked).
    others = [r for r in plan.table if r.fits and _spec(r) != (spec, w, moves)]
    second = P.LoopRSBFPlanner.select(others, applicable_only=False)
    assert _spec(second) == runner[:3]
    assert second.mfu == pytest.approx(runner[3], rel=1e-9, abs=0)
    # The plan maps the pick onto the executor's fields.
    if pick.option.form == "block" and pick.option.stride_layers < plan.core_layers:
        assert (plan.item_form, plan.block_stride) == ("block", 2 * pick.option.stride_layers)
    assert plan.waves == w and plan.optimizer_offload == ("A" in moves)
    assert plan.coda_recompute == pick.option.coda_recompute


def test_menu_size_is_the_reference_grid():
    """B: (1 nested + 7 strides x 4 variants) x 4 moves x 4 waves = 464 evaluations; D at
    n = 4 offers waves 1, 2, 4 only."""
    assert len(_plan("X1").table) == 464
    assert len(_plan("X8").table) == (1 + 4 * 4) * 4 * 4
    assert len(_plan("X7").table) == (1 + 7 * 4) * 4 * 3


@pytest.mark.parametrize("name", ["X8", "X9"])
def test_d28_whole_round_pick(name):
    """Under form policy d28 (L_core = 4 <= 4) the applied pick is the whole round (block s =
    L_core); X8's raw pick 'w 2' has the same MFU (D-S5-10)."""
    plan = _plan(name, form_policy="d28")
    spec, w, moves, hbm, grace, mfu, step = D28_PICKS[name]
    assert _spec(plan.applied) == (spec, w, moves)
    assert (plan.item_form, plan.block_stride, plan.stride_layers) == ("whole_round", None, 4)
    assert abs(plan.applied.hbm_bytes / 1e9 - hbm) <= 0.002
    assert abs(plan.applied.grace_bytes / 1e9 - grace) <= 0.002
    assert plan.applied.mfu == pytest.approx(mfu, rel=1e-9, abs=0)
    assert abs(plan.applied.step_ms / 1e3 - step) <= 0.002
    assert _spec(plan.raw) == PICKS[name][0][:3]
    assert round(plan.raw.mfu, 5) == round(plan.applied.mfu, 5)
    refused = [r for r in plan.table if "excluded by form policy d28" in r.refusals]
    assert refused and all(r.option.stride_layers != 4 for r in refused)


def test_grace_margin_refusal():
    """R1: X5's best two-wave option ('w 6', w 2, AEM, 31.419 %) uses 195.21 GB of Grace
    (margin 4.79 GB) and is refused only by the 10 GB margin; with no margin it is the pick."""
    plan = _plan("X5")
    (best2,) = [r for r in plan.table if _spec(r) == ("w 6", 2, "AEM")]
    assert abs(best2.grace_bytes / 1e9 - 195.2079531528775) <= 0.002
    assert best2.mfu == pytest.approx(0.3141928017972177, rel=1e-9, abs=0)
    assert not best2.fits
    assert best2.refusals == ("Grace margin 4.79 GB < 10.0 GB",)
    assert best2.mfu > plan.applied.mfu
    _, S, cp, depth, n, _ = ROWS["X5"]
    hw = P.PlannerHardware.preset("gb200", grace_margin_bytes=0.0)
    planner = P.LoopRSBFPlanner(
        B,
        hw,
        capabilities=P.PlannerCapabilities.target(),
        strides_layers=STRIDES["B"],
        waves=(2,),
        layouts=("layer_window",),
        form_policy="planner",
        sched_policy="min_makespan",
    )
    assert _spec(planner.plan(_phase(S, cp, n), DEPTHS[depth]).applied) == ("w 6", 2, "AEM")


def test_hbm_cap_and_margin_refusals_at_x2():
    """R2: at X2 every '-' option exceeds the HBM cap ('w 6', '-': 322.23 GB) and 'w 6', A
    (Grace 196.78 GB) is under the Grace margin."""
    plan = _plan("X2")
    minus = [r for r in plan.table if r.option.moves == "-"]
    assert minus and all(not r.fits for r in minus)
    assert all(any(x.startswith("HBM over cap by") for x in r.refusals) for r in minus)
    (w6,) = [r for r in minus if _spec(r) == ("w 6", 1, "-")]
    assert abs(w6.hbm_bytes / 1e9 - 322.23110405344) <= 0.002
    (w6a,) = [r for r in plan.table if _spec(r) == ("w 6", 1, "A")]
    assert abs(w6a.grace_bytes / 1e9 - 196.77999006944) <= 0.002 and not w6a.fits
    assert w6a.refusals == ("Grace margin 3.22 GB < 10.0 GB",)


def test_nested_reference_at_x2():
    """R3: forms restricted to nested at X2: Adam to Grace, HBM 171.86 / Grace 152.08 GB, MFU
    25.628 % (DP §3.3)."""
    plan = _plan("X2", form_policy="nested")
    assert _spec(plan.applied) == ("nested", 1, "A")
    assert (plan.item_form, plan.block_stride) == ("nested", 2)
    assert abs(plan.applied.hbm_bytes / 1e9 - 171.859589283815) <= 0.002
    assert abs(plan.applied.grace_bytes / 1e9 - 152.082731381625) <= 0.002
    assert plan.applied.mfu == pytest.approx(0.2562797116002562, rel=1e-9, abs=0)
    assert plan.applied.peak_sample_rounds is None


@pytest.mark.parametrize("depth", list(RESIDENT_HBM))
def test_resident_layout_rows(depth):
    """X11 / X12: the layout that runs today (every dense group Resident, BF16, eager, raw EP,
    no move). The nested item reproduces proxy_v2's HBM; the whole round (block s = L_core) is
    one round state above evaluate(whole_round=True), whose arena omits the block input the
    block form keeps (stage-5 deviation)."""
    nested_hbm, whole_hbm, peak = RESIDENT_HBM[depth]
    planner = _planner(
        "D", strides_layers=(4,), waves=(1,), move_steps=("-",), layouts=("resident",)
    )
    phase = dataclasses.replace(
        _phase(4096, 1, 8), precision="bf16", graphs="eager", ep_overlap="raw"
    )
    results = {r.option.label: r for r in planner.evaluate(phase, DEPTHS[depth])}
    assert abs(results["nested"].hbm_bytes / 1e9 - nested_hbm) <= 0.002
    state = 4096 * D.state_bytes_per_token
    assert abs((results["w 4"].hbm_bytes - state) / 1e9 - whole_hbm) <= 0.002
    for r in results.values():
        assert r.peak_units == peak and r.fits and r.grace_bytes == 0.0
        assert r.components[C.MEM_DENSE_SH] == 0.0 and r.components[C.MEM_DENSE_BUF] == 0.0


def test_no_grace_tier_picks_an_hbm_only_option():
    """The grace-0 rule on the h100 preset (DSV4-TINY at 8 GPUs, resident layout): a fitting
    HBM-only option is picked and every Adam-to-Grace option is refused with 'no Grace tier'
    (the margin never refuses an HBM-only option)."""
    tiny = C.ModelShape.from_dims(
        C.DSv4Dims(
            H=512, V=100352, NEXP=8, TOPK=2, FFN_E=2048, FFN_S=0, HEADS=8, DH=16, QL=192, OG=8,
            OL=1024, IDX_H=64, IDX_D=128, IDX_TOPK=512, WIN=128, NS=4,
        ),
        (1, 4, 2),
        C.proxy7_ratios(),
    )  # fmt: skip
    planner = P.LoopRSBFPlanner(
        tiny,
        P.PlannerHardware.preset("h100"),
        capabilities=P.PlannerCapabilities(optimizer_offload=True),
        layouts=("resident",),
    )
    phase = P.PhaseGeometry(
        seq_length=4096,
        gpus=8,
        ep=8,
        samples_per_eu=8,
        max_rounds=4,
        lockstep_eus=8,
        precision="bf16",
        graphs="eager",
        ep_overlap="raw",
    )
    plan = planner.plan(phase, P.R4_UNIF)
    assert plan.applied.grace_bytes == 0.0 and plan.applied.option.moves == "-"
    assert plan.applied.hbm_bytes <= planner.hardware.hbm_cap_bytes
    adam = [r for r in plan.table if "A" in r.option.moves]
    assert adam and all(not r.fits and "no Grace tier" in r.refusals for r in adam)
    assert all(r.fits for r in plan.table if r.option.moves == "-")
