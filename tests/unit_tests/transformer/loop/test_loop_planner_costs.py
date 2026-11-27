# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the RSBF planner's model, FLOP, time and memory terms
(megatron/core/transformer/loop/planner_costs.py), the port of the toolkit cost model.

Constants: the cost model's own outputs on its shapes (toolkit
``docs/benchmarks/loop_transformer/rsbf-design/cost-model/dsv4`` at toolkit commit 1afb272:
``dsv4.params`` / ``costs`` / ``memory`` / ``step_level_ms`` / ``model_flops_per_sample`` under
``proxy_dsv4.shape``, and ``pp_core_baseline.rsbf_resident_sync_ms``; probe
``$IMPL/s53/ref_resident.py``). B = DSV4-PRO-192E (Pro width, 192 experts, 3 / 54 / 4 + MTP),
D = DSV4-PRO-PROXY (1 / 4 / 2 + MTP); EP 64, N 64, two-chunk EP, MXFP8, whole-layer graphs.
The port must reproduce them exactly (same formulas, same float operation order).

The last tests build a small CPU loop model (gloo, model parallel initialized once for the
module) for ``ModelShape.from_model``.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.loop import flops as loop_flops
from megatron.core.transformer.loop import planner as P
from megatron.core.transformer.loop import planner_costs as C
from megatron.core.transformer.loop.planner import PlannerHardware
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    build_local_loop_model,
    cpu_mtp,
    model_parallel_once,
)

B = C.ModelShape.from_dims(
    dataclasses.replace(C.PRO_DIMS, NEXP=192), (3, 54, 4), C.pro_ratios(3, 54, 4)
)
D = C.ModelShape.from_dims(C.PRO_DIMS, (1, 4, 2), C.proxy7_ratios())
SHAPES = {"B": B, "D": D}
GB200 = PlannerHardware.preset("gb200")

# dsv4.params under proxy_dsv4.shape: dense groups, P_e, p_dense_layer(CSA / HCA / W).
PARAMS = {
    "B": (
        {
            "emb+head": 1853479941,
            "prelude dense": 1086315682,
            "core dense": 19872422884,
            "coda dense": 1464415192,
            "MTP dense": 449598651,
        },
        786381668352,
        (378099510, 354108086, 346702006),
    ),
    "D": (
        {
            "emb+head": 1853479941,
            "prelude dense": 355484342,
            "core dense": 1572738008,
            "coda dense": 734960108,
            "MTP dense": 450974907,
        },
        202937204736,
        (379475766, 355484342, 348078262),
    ),
}

# dsv4.costs(ph, prec="mxfp8", EP=64, domains=1, ep_overlap="two_chunk", graphs=True, **variant).d
VARIANTS = {
    "nested": dict(nested=True),
    "block_w": dict(nested=False),
    "block_r": dict(nested=False, whole_round=True),
    "block_w_rc": dict(nested=False, coda_rc=True),
    "nested_E": dict(nested=True, lw_edges=True),
    "nested_EM": dict(nested=True, lw_edges=True, emb_rs=True),
    "nested_nominal": dict(nested=True, nominal=True),
    "resident_bf16_eager_raw": dict(
        nested=True, lw_core=False, prec="bf16", graphs="eager", ep_overlap="raw"
    ),
}
# fmt: off
UNIT_COSTS = {
    ("B", 4096, "block_r"): (317.26106950558795, 934.8555928725161, 123.93997819213709, 49.263545416002714, 17.251962876109097),  # noqa: E501
    ("B", 4096, "block_w"): (317.26106950558795, 926.7469657840005, 123.93997819213709, 49.263545416002714, 17.251962876109097),  # noqa: E501
    ("B", 4096, "block_w_rc"): (317.26106950558795, 926.7469657840005, 127.48772092442508, 49.263545416002714, 17.251962876109097),  # noqa: E501
    ("B", 4096, "nested"): (317.26106950558795, 1244.0080352895884, 123.93997819213709, 49.263545416002714, 17.251962876109097),  # noqa: E501
    ("B", 4096, "nested_E"): (317.26106950558795, 1244.0080352895884, 126.16850130686365, 50.60022291533865, 17.69752204255441),  # noqa: E501
    ("B", 4096, "nested_EM"): (317.26106950558795, 1244.0080352895884, 126.92866770686365, 51.36038931533865, 17.69752204255441),  # noqa: E501
    ("B", 4096, "nested_nominal"): (276.13839441707233, 1113.031382935526, 117.4188581921371, 45.15087341600272, 14.945626876109095),  # noqa: E501
    ("B", 4096, "resident_bf16_eager_raw"): (729.5, 2555.2, 169.5, 101.89999999999999, 41.0),
    ("B", 8192, "block_r"): (646.3264281972486, 1904.2536117214174, 252.38555309535462, 100.6513466444051, 34.80166305379977),  # noqa: E501
    ("B", 8192, "block_w"): (646.3264281972486, 1896.144984632902, 252.38555309535462, 100.6513466444051, 34.80166305379977),  # noqa: E501
    ("B", 8192, "block_w_rc"): (646.3264281972486, 1896.144984632902, 259.4810385599306, 100.6513466444051, 34.80166305379977),  # noqa: E501
    ("B", 8192, "nested"): (646.3264281972486, 2542.4714128301503, 252.38555309535462, 100.6513466444051, 34.80166305379977),  # noqa: E501
    ("B", 8192, "nested_E"): (646.3264281972486, 2542.4714128301503, 254.61407621008118, 101.98802414374103, 35.247222220245085),  # noqa: E501
    ("B", 8192, "nested_EM"): (646.3264281972486, 2542.4714128301503, 255.37424261008118, 102.74819054374103, 35.247222220245085),  # noqa: E501
    ("B", 8192, "nested_nominal"): (572.6897051087329, 2313.9526164760878, 239.84331309535463, 92.92600264440509, 30.688991053799768),  # noqa: E501
    ("B", 8192, "resident_bf16_eager_raw"): (849.4413724418521, 3315.9143322085647, 307.3687859432266, 133.6103625885971, 46.510536101863764),  # noqa: E501
    ("D", 4096, "block_r"): (24.006127207769172, 69.84081616225212, 90.62568789355853, 15.88912557528015, 5.830412525093384),  # noqa: E501
    ("D", 4096, "block_w"): (24.006127207769172, 69.23791919865837, 90.62568789355853, 15.88912557528015, 5.830412525093384),  # noqa: E501
    ("D", 4096, "block_w_rc"): (24.006127207769172, 69.23791919865837, 92.75433353293133, 15.88912557528015, 5.830412525093384),  # noqa: E501
    ("D", 4096, "nested"): (24.006127207769172, 93.24404640642754, 90.62568789355853, 15.88912557528015, 5.830412525093384),  # noqa: E501
    ("D", 4096, "nested_E"): (24.006127207769172, 93.24404640642754, 91.95833276289447, 16.32653794922546, 5.976216649741821),  # noqa: E501
    ("D", 4096, "nested_EM"): (24.006127207769172, 93.24404640642754, 92.71849916289447, 17.086704349225464, 5.976216649741821),  # noqa: E501
    ("D", 4096, "nested_nominal"): (20.494782244175422, 82.60711455205255, 86.51301589355853, 14.18490157528015, 4.728300525093384),  # noqa: E501
    ("D", 4096, "resident_bf16_eager_raw"): (54.5, 190.2, 107.10519586565454, 34.3, 14.0),
    ("D", 8192, "block_r"): (48.421425324136315, 141.76833590201426, 183.257826782255, 31.403361247017447, 11.202528415672482),  # noqa: E501
    ("D", 8192, "block_w"): (48.421425324136315, 141.1654389384205, 183.257826782255, 31.403361247017447, 11.202528415672482),  # noqa: E501
    ("D", 8192, "block_w_rc"): (48.421425324136315, 141.1654389384205, 187.51511806100058, 31.403361247017447, 11.202528415672482),  # noqa: E501
    ("D", 8192, "nested"): (48.421425324136315, 189.58686426255682, 183.257826782255, 31.403361247017447, 11.202528415672482),  # noqa: E501
    ("D", 8192, "nested_E"): (48.421425324136315, 189.58686426255682, 184.59047165159095, 31.84077362096276, 11.34833254032092),  # noqa: E501
    ("D", 8192, "nested_EM"): (48.421425324136315, 189.58686426255682, 185.35063805159092, 32.60094002096276, 11.34833254032092),  # noqa: E501
    ("D", 8192, "nested_nominal"): (42.501632360542565, 171.72458840818183, 175.532482782255, 28.494913247017447, 9.498304415672482),  # noqa: E501
    ("D", 8192, "resident_bf16_eager_raw"): (63.46471882966257, 246.86990068466184, 216.216842726447, 42.33816028752945, 15.088306229176483),  # noqa: E501
}
# fmt: on
MOE_SHARE_B_4096_NESTED = {
    "f": 0.3799542748848387,
    "b": 0.3876010321271691,
    "coda": 0.27016818358262634,
    "pre": 0.40782252067342545,
}

# dsv4.memory(ph, EP=64, N=64, peak_units=40.0, **kw) (ep_ib: the 26 GB reserve), bytes; the
# reference's always-zero item-pairing arena is left out.
_EXPERTS_B = (24574427136.0, 49148854272.0)
_DENSE_B = (29122856796.0, 1863039645.375, 4537194120, 152082731381.625)
_EXPERTS_D = (6341787648.0, 12683575296.0)
_DENSE_D = (20369395788.0, 147444188.25, 4553709192, 38982157882.875)
MEMORY = {
    ("B", 4096, "nested"): (_EXPERTS_B, _DENSE_B, 22477373439.999996, None, 67108864, 1409286144),
    ("B", 4096, "block4"): (_EXPERTS_B, _DENSE_B, 22477373439.999996, None, 67108864, 1409286144),
    ("B", 4096, "block6w"): (_EXPERTS_B, _DENSE_B, 26457473024.0, 3780995100, 67108864, 1409286144),
    ("B", 8192, "nested"): (_EXPERTS_B, _DENSE_B, 44954746879.99999, None, 268435456, 2818572288),
    ("B", 8192, "block6w"): (
        _EXPERTS_B,
        _DENSE_B,
        52914946048.0,
        3780995100,
        268435456,
        2818572288,
    ),
    ("D", 4096, "nested"): (_EXPERTS_D, _DENSE_D, 13736509439.999998, None, 67108864, 1409286144),
    ("D", 4096, "block4"): (_EXPERTS_D, _DENSE_D, 17716609024.0, None, 67108864, 1409286144),
    ("D", 8192, "block4"): (_EXPERTS_D, _DENSE_D, 35433218048.0, None, 268435456, 2818572288),
}
MXFP8_CASTS = {"B": 1187137570.44, "D": 1598225237.64}
MEMORY_KW = {
    "nested": dict(nested=True),
    "block4": dict(nested=False, block_stride=4),
    "block6w": dict(nested=False, block_stride=6, block_window=True, coda_rc=True),
}
# step_level_ms(EP=64, N=64, domains=1, adam_on_grace), model_flops_per_sample(ph, d),
# rsbf_resident_sync_ms(moves).
STEP_LEVEL = {
    ("B", True): {"optimizer step on Grace": 2944.1656972596634},
    ("B", False): {"optimizer step in HBM": 95.05170711351563},
    ("D", True): {"optimizer step on Grace": 754.6545949120673},
    ("D", False): {"optimizer step in HBM": 24.363848676796877},
}
MODEL_FLOPS = {
    ("B", 4096, 1): 1300291123150848.0,
    ("B", 4096, 4): 4569626478379008.0,
    ("B", 8192, 1): 2666050734784512.0,
    ("B", 8192, 4): 9381173532819456.0,
    ("D", 4096, 1): 210783254347776.0,
    ("D", 4096, 4): 453362118033408.0,
    ("D", 8192, 1): 428217634652160.0,
    ("D", 8192, 4): 926445887029248.0,
}
RESIDENT_SYNC = {
    ("B", "-"): 39.81640577578125,
    ("B", "A"): 39.81640577578125,
    ("B", "AE"): 15.204327641015624,
    ("B", "AEM"): 5.068109213671875,
    ("D", "-"): 27.84878330390625,
    ("D", "AE"): 15.204327641015624,
    ("D", "AEM"): 5.068109213671875,
}


@pytest.mark.parametrize("model", ["B", "D"])
def test_parameter_groups_are_the_cost_model_params(model):
    shape = SHAPES[model]
    groups, routed, (csa, hca, window) = PARAMS[model]
    assert shape.params.groups() == groups
    assert shape.params.routed_experts == routed
    assert shape.params.largest_dense_layer == csa
    assert dict(shape.params.dense_by_kind) == {"CSA": csa, "HCA": hca, "W": window}
    # Byte constants at Pro width (proxy_dsv4.shape): act 0.97 / 1.05 MB, q-path 0.4 MB, mHC.
    assert shape.act_bytes_per_token_layer == 969999.9999999999
    assert shape.act_bytes_per_token_mtp == 1049999.9999999998
    assert shape.elem_bytes_per_token == 400000.0
    assert shape.mhc_bytes_per_token == 401408
    assert shape.state_bytes_per_token == 57344


def test_layer_kinds_of_the_cost_model_shapes():
    assert [k for k, _ in D.prelude] == ["HCA"]
    assert [k for k, _ in D.core] == ["HCA", "CSA", "HCA", "CSA"]
    assert [k for k, _ in D.coda] == ["HCA", "CSA"]
    assert D.mtp == (("W", "moe"),)
    assert B.core_layers == 54 and len(B.prelude) == 3 and len(B.coda) == 4
    assert B.core[0][0] == "HCA" and B.core[-1][0] == "CSA"  # D15: the alternation holds
    assert all(ffn == "moe" for _, ffn in B.prelude + B.core + B.coda + B.mtp)
    assert D.has_csa and D.has_coda_dsv4_attention and D.coda_dsv4_attentions == 3


@pytest.mark.parametrize("key", sorted(UNIT_COSTS))
def test_unit_costs_are_the_cost_model_costs(key):
    model, S, variant = key
    kw = dict(prec="mxfp8", EP=64, ep_overlap="two_chunk", graphs="full", lw_core=True)
    kw.update(VARIANTS[variant])
    shape = SHAPES[model]
    uc = C.unit_costs(shape, C.SeqPhase.of(shape, S), GB200, **kw)
    assert tuple(uc.d[k] for k in ("f", "b", "coda", "pre", "sweep")) == UNIT_COSTS[key]
    if key == ("B", 4096, "nested"):
        assert dict(uc.moe_share) == MOE_SHARE_B_4096_NESTED


def test_unit_cost_charges():
    """C2C bytes are charged on f / b as a hidden transfer (exposed part + kappa x the rest);
    the nested rebuild scales with rebuild_fraction; nominal costs are GPU time only."""
    ph = C.SeqPhase.of(D, 4096)
    kw = dict(prec="mxfp8", EP=64, ep_overlap="two_chunk", graphs="full", lw_core=True, nested=True)
    base = C.unit_costs(D, ph, GB200, **kw)
    cb = dict(f=1e9, b=3e9)
    charged = C.unit_costs(D, ph, GB200, c2c_bytes=cb, **kw)
    for task in ("f", "b"):
        t = cb[task] / GB200.c2c_bytes_per_s * 1e3
        exposed = max(0.0, t - base.split[task]["gpu"])
        assert charged.split[task]["c2c"] == exposed + GB200.kappa * (t - exposed)
        assert charged.d[task] == pytest.approx(base.d[task] + charged.split[task]["c2c"])
    for task in ("coda", "pre", "sweep"):
        assert charged.d[task] == base.d[task]
    block = C.unit_costs(D, ph, GB200, **dict(kw, nested=False))
    half = C.unit_costs(D, ph, GB200, rebuild_fraction=0.5, **kw)
    assert half.d["b"] == pytest.approx((base.d["b"] + block.d["b"]) / 2)
    nominal = C.unit_costs(D, ph, GB200, nominal=True, **kw)
    assert all(nominal.split[t]["ep"] == 0 for t in ("f", "b", "coda", "pre", "sweep"))


@pytest.mark.parametrize("key", sorted(MEMORY))
def test_memory_components_are_the_cost_model_memory(key):
    model, S, variant = key
    shape = SHAPES[model]
    hw = PlannerHardware.preset("gb200", ep_reserve_bytes=26e9)
    comps, adam_grace = C.memory_components(
        shape,
        C.SeqPhase.of(shape, S),
        hw,
        40.0,
        EP=64,
        N=64,
        prec="mxfp8",
        layout="layer_window",
        **MEMORY_KW[variant],
    )
    (expert_w, expert_g), (res, shards, buf, opt), act, window, indexer, graph_io = MEMORY[key]
    act_key = C.MEM_ACT_NESTED if variant == "nested" else C.MEM_ACT_BLOCK
    expected = {
        C.MEM_EXPERT_W: expert_w,
        C.MEM_EXPERT_G: expert_g,
        C.MEM_DENSE_RES: res,
        C.MEM_DENSE_SH: shards,
        C.MEM_DENSE_BUF: buf,
        C.MEM_OPT: opt,
        act_key: act,
        **({} if window is None else {C.MEM_WINDOW: window}),
        C.MEM_INDEXER: indexer,
        C.MEM_MXFP8: MXFP8_CASTS[model],
        C.MEM_GRAPH_IO: graph_io,
        C.MEM_STORE: 40.0 * graph_io / 6,
        C.MEM_RESERVE: 26e9,
        C.MEM_MISC: 8e9,
    }
    assert comps == expected
    assert list(comps) == list(expected)  # the reference's order (its sums)
    assert adam_grace == 0.0


def test_memory_moves_and_layouts():
    """A: the Adam state leaves HBM for Grace (12 bytes per parameter / N in the cost model,
    the shape's Grace bytes per parameter otherwise); E / M: Layer window / sharded grads;
    resident: every dense group Resident, no shard buffers, no block window."""
    ph = C.SeqPhase.of(D, 4096)
    kw = dict(EP=64, N=64, prec="mxfp8", nested=True)
    lw, _ = C.memory_components(D, ph, GB200, 40.0, layout="layer_window", **kw)
    a, adam = C.memory_components(
        D, ph, GB200, 40.0, layout="layer_window", adam_on_grace=True, **kw
    )
    assert a[C.MEM_OPT] == 0.0 and adam == lw[C.MEM_OPT]
    runtime = dataclasses.replace(D, grace_optimizer_bytes_per_param=16)
    _, adam16 = C.memory_components(
        runtime, ph, GB200, 40.0, layout="layer_window", adam_on_grace=True, **kw
    )
    assert adam16 == pytest.approx(adam * 16 / 12)
    groups = D.params.groups()
    em, _ = C.memory_components(
        D,
        ph,
        GB200,
        40.0,
        layout="layer_window",
        edges_mode="Layer window",
        emb_mode="Resident + sharded grads",
        **kw,
    )
    assert em[C.MEM_DENSE_RES] == 2 * groups["emb+head"]
    assert em[C.MEM_DENSE_SH] == pytest.approx(6 * sum(groups.values()) / 64, rel=1e-12)
    res, _ = C.memory_components(
        D, ph, GB200, 40.0, layout="resident", edges_mode="Layer window", block_stride=None, **kw
    )
    assert res[C.MEM_DENSE_RES] == pytest.approx(6 * sum(groups.values()), rel=1e-12)
    assert res[C.MEM_DENSE_SH] == 0.0 and res[C.MEM_DENSE_BUF] == 0.0
    blk, _ = C.memory_components(
        D,
        ph,
        GB200,
        40.0,
        EP=64,
        N=64,
        prec="bf16",
        layout="resident",
        nested=False,
        block_stride=2,
        block_window=True,
    )
    assert C.MEM_WINDOW not in blk and C.MEM_MXFP8 not in blk
    nodist = dataclasses.replace(D, distributed_optimizer=False)
    nd, _ = C.memory_components(nodist, ph, GB200, 40.0, layout="layer_window", **kw)
    assert nd[C.MEM_OPT] == 12 * (sum(groups.values()) + D.params.routed_experts / 64)
    with pytest.raises(ValueError, match="nested=False"):
        C.memory_components(D, ph, GB200, 1.0, layout="resident", block_stride=2, **kw)


def test_round_state_placement():
    """HBM first up to the cap, the rest on Grace after the Adam state."""
    hw = PlannerHardware.preset("gb200")  # cap 175 GB, Grace 200 GB
    comps = {"a": 150e9, C.MEM_STORE: 10e9}
    fits = C.place_round_state(comps, 0.0, stash_need=5e9, hw=hw)
    assert (fits.hbm, fits.on_grace, fits.share, fits.mem_ok) == (165e9, 0.0, 0.0, True)
    spill = C.place_round_state(comps, 100e9, stash_need=30e9, hw=hw)
    assert spill.in_hbm == 25e9 and spill.on_grace == 15e9 and spill.hbm == 175e9
    assert spill.grace == 115e9 and spill.share == 15e9 / 40e9 and spill.mem_ok
    over = C.place_round_state({"a": 180e9, C.MEM_STORE: 1e9}, 0.0, stash_need=0.0, hw=hw)
    assert over.room == -5e9 and over.in_hbm == 0.0 and not over.mem_ok and over.deficit == 5e9
    full = C.place_round_state(comps, 195e9, stash_need=30e9, hw=hw)
    assert full.grace == 210e9 and not full.mem_ok and full.deficit == 10e9
    none = C.place_round_state(comps, 0.0, stash_need=40e9, hw=PlannerHardware.preset("h100"))
    assert none.on_grace > 0 and not none.mem_ok  # no Grace tier


def test_c2c_traffic():
    fw, br, bw = C.c2c_traffic(0.5, 0.25, 3, 100.0)
    assert (fw, br, bw) == ((0.5 * 3 + 0.25) * 100.0, (0.5 * 3 + 0.25 * 4) * 100.0, 0.75 * 100.0)


@pytest.mark.parametrize("model", ["B", "D"])
def test_step_level_flops_and_resident_sync(model):
    shape = SHAPES[model]
    for adam in (True, False):
        assert (
            C.step_level_ms(shape, GB200, EP=64, N=64, adam_on_grace=adam)
            == STEP_LEVEL[(model, adam)]
        )
    for S in (4096, 8192):
        for depth in (1, 4):
            ph = C.SeqPhase.of(shape, S)
            assert (
                C.model_flops_per_sample(shape, ph, depth, GB200) == MODEL_FLOPS[(model, S, depth)]
            )
    for (m, moves), value in RESIDENT_SYNC.items():
        if m == model:
            assert C.resident_sync_ms(shape, GB200, moves, 64, "layer_window") == value
    # Resident layout: the core dense group is synced too (planner v1, est.).
    extra = 6 * shape.params.core_dense * 63 / 64 / GB200.nvl_bytes_per_s * 1e3
    assert C.resident_sync_ms(shape, GB200, "-", 64, "resident") == pytest.approx(
        RESIDENT_SYNC[(model, "-")] + extra
    )
    # More GPUs than EP: the EDP expert sync (IB); more than the dense axis: the replicate sync.
    wide = C.step_level_ms(shape, GB200, EP=64, N=128, adam_on_grace=False)
    assert set(wide) == {
        "expert grad RS (fp32) + param AG (bf16) over EDP, IB",
        "dense shard RS + AG over replicate group, IB",
        "optimizer step in HBM",
    }


def test_key_means():
    km = C.key_means(1024, 0, 1024, 128, 64)
    assert km["full"] == pytest.approx(1025 / 2)
    assert km["win"] == pytest.approx(sum(min(i + 1, 128) for i in range(1024)) / 1024)
    assert km["topk"] == pytest.approx(sum(min(64, (i + 1) // 4) for i in range(1024)) / 1024)
    tail = C.key_means(1024, 512, 1024, 128, 64)
    assert tail["full"] == pytest.approx(sum(range(513, 1025)) / 512)


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


def _tiny_config(**overrides):
    fields = dict(
        experimental_attention_variant="dsv4_hybrid",
        hidden_size=512,
        num_attention_heads=8,
        kv_channels=None,
        v_head_dim=16,
        q_lora_rank=192,
        o_groups=8,
        o_lora_rank=1024,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_ffn_hidden_size=None,
        ffn_hidden_size=2048,
        moe_shared_expert_intermediate_size=None,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=512,
        csa_window_size=128,
        csa_compress_ratios=None,
        enable_hyper_connections=True,
        num_residual_streams=4,
        num_query_groups=None,
        gated_linear_unit=True,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_dims_from_config():
    # KVH / FFN_D only matter for plain attention / dense MLP layers.
    assert C.DSv4Dims.from_config(_tiny_config(), 100352) == dataclasses.replace(
        TINY_DIMS, KVH=8, FFN_D=2048
    )
    dense = SimpleNamespace(
        experimental_attention_variant=None,
        hidden_size=64,
        num_attention_heads=2,
        kv_channels=32,
        v_head_dim=None,
        q_lora_rank=None,
        num_moe_experts=None,
        moe_router_topk=2,
        moe_ffn_hidden_size=None,
        ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=None,
        dsa_indexer_n_heads=None,
        dsa_indexer_head_dim=None,
        dsa_indexer_topk=None,
        csa_window_size=128,
        enable_hyper_connections=False,
        num_residual_streams=4,
        num_query_groups=1,
        gated_linear_unit=False,
    )
    dims = C.DSv4Dims.from_config(dense, 128)
    assert (dims.H, dims.HEADS, dims.DH, dims.KVH, dims.NEXP, dims.TOPK, dims.NS) == (
        64,
        2,
        32,
        1,
        0,
        0,
        1,
    )
    assert (dims.FFN_D, dims.MLP_MATS, dims.QL, dims.OG, dims.IDX_H) == (128, 2, 0, 0, 0)


def test_flops_agree_with_the_loop_flop_table():
    """Sanity against loop/flops.py (GEMM-only, router excluded): the dense rows and the
    DSv4 window / HCA attention within 1%; CSA without the indexer scoring within 1% (the
    table counts the indexer over the full square, the planner over causal candidates, so that
    term differs by convention, about 2x)."""
    S = 4096
    cfg = _tiny_config()
    km = C.key_means(S, 0, S, TINY_DIMS.WIN, TINY_DIMS.IDX_TOPK)
    router = 2 * TINY_DIMS.NEXP * TINY_DIMS.H
    for kind, ratio in (("W", 0), ("HCA", 128), ("CSA", 4)):
        f = C.layer_flops(kind, km, TINY_DIMS, "moe")
        mine = f["dense_mx"] + f["small_bf16"] - router + f["attn"]
        table = loop_flops.dsv4_attention_forward_flops(cfg, ratio, S, S * S, S) / S
        if kind == "CSA":
            table -= 2 * cfg.dsa_indexer_n_heads * cfg.dsa_indexer_head_dim / 4 * S
            assert f["idx"] == pytest.approx(
                km["cand"] * 2 * cfg.dsa_indexer_n_heads * cfg.dsa_indexer_head_dim
            )
        assert mine == pytest.approx(table, rel=0.01), kind
    f = C.layer_flops("CSA", km, TINY_DIMS, "moe")
    moe_table = loop_flops.moe_forward_flops(
        SimpleNamespace(
            gated_linear_unit=True,
            hidden_size=512,
            moe_ffn_hidden_size=2048,
            ffn_hidden_size=2048,
            moe_shared_expert_intermediate_size=0,
            moe_router_topk=2,
            moe_latent_size=None,
        ),
        S,
    )
    assert f["shared_mx"] + f["routed_mx"] == moe_table / S
    dense_cfg = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=2,
        kv_channels=32,
        num_query_groups=1,
        attention_output_gate=False,
        ffn_hidden_size=128,
        gated_linear_unit=True,
    )
    dims = C.DSv4Dims(
        H=64, V=128, NEXP=0, TOPK=0, FFN_E=128, FFN_S=0, HEADS=2, DH=32, QL=0, OG=0, OL=0,
        IDX_H=0, IDX_D=0, IDX_TOPK=0, WIN=128, NS=1, KVH=1, FFN_D=128, MLP_MATS=3,
    )  # fmt: skip
    km = C.key_means(S, 0, S, 128, 0)
    att = C.layer_flops("ATT", km, dims, "mlp")
    # QKV + output projections exactly; the causal core counts (S + 1) / 2 keys per query.
    assert att["dense_mx"] == 4 * 64 * (2 * 32 + 32)
    assert att["attn"] == (S + 1) / 2 * 4 * 2 * 32
    assert att["dense_mx"] + att["attn"] == pytest.approx(
        loop_flops.attention_forward_flops(dense_cfg, S, S * S) / S, rel=0.01
    )
    assert att["shared_mx"] == loop_flops.mlp_forward_flops(dense_cfg, S) / S
    assert att["small_bf16"] == 0 and att["routed_mx"] == 0 and att["mhc"] == 0
    assert att["idx"] == att["teacher"] == att["idx_bwd"] == 0.0


# ----------------------------------------------------------------------------- from_model (CPU model)
_DENSE_PATTERN = "*-*-*-*-*-*-"


@pytest.fixture(scope="module")
def _model_parallel():
    with model_parallel_once():
        yield


def test_shape_from_a_built_dense_loop_model(_model_parallel, cpu_mtp):
    """Every parameter lands in exactly one group; kinds / FFNs follow the pattern; the
    adapter and loop norm are core; MTP is its own group."""
    model = build_local_loop_model(_DENSE_PATTERN + "/*-", prelude=2, coda=2, loop_iterations=3)
    shape = C.ModelShape.from_model(model, vocab_size=128, expert_parallel_size=1)
    assert shape.prelude == (("ATT", "mlp"),)
    assert shape.core == (("ATT", "mlp"),) * 4
    assert shape.coda == (("ATT", "mlp"),)
    assert shape.mtp == (("ATT", "mlp"),)
    total = sum(p.numel() for p in model.parameters())
    p = shape.params
    assert p.dense + p.routed_experts == total
    assert p.routed_experts == 0 and p.expert_params == 0.0
    stack = model.decoder
    adapter = sum(q.numel() for q in stack.loop_adapter.parameters()) + sum(
        q.numel() for q in stack.loop_norm.parameters()
    )
    layer_pair = sum(q.numel() for q in stack.layers[2].parameters()) + sum(
        q.numel() for q in stack.layers[3].parameters()
    )
    assert p.core_dense == 4 * layer_pair + adapter
    assert p.prelude_dense == layer_pair and p.coda_dense == layer_pair
    assert p.mtp_dense == sum(q.numel() for q in model.mtp.parameters())
    assert p.largest_dense_layer == layer_pair and dict(p.dense_by_kind) == {"ATT": layer_pair}
    assert shape.dims.NS == 1 and shape.mhc_bytes_per_token == 0.0
    assert not shape.has_csa and not shape.has_coda_dsv4_attention
    override = C.ModelShape.from_model(
        model, vocab_size=128, expert_parallel_size=1, act_bytes_per_token=1000.0
    )
    assert override.act_bytes_per_token_layer == 1000.0
    assert override.act_bytes_per_token_mtp == pytest.approx(1000.0 * 1.05 / 0.97)
    # The built model plans end to end (both layouts, the MTP layer included).
    phase = P.PhaseGeometry(
        seq_length=512, gpus=1, ep=1, samples_per_eu=4, max_rounds=4, graphs="eager"
    )
    plan = P.LoopRSBFPlanner(shape, PlannerHardware.preset("h100")).plan(phase, P.R4_UNIF)
    assert plan.applied.fits and plan.applied.applicable


def test_shape_from_model_rejects_unpaired_roles(_model_parallel):
    model = build_local_loop_model("*-**--*-", prelude=2, coda=2)
    with pytest.raises(C.PlannerUnsupported, match="pairs"):
        C.ModelShape.from_model(model, vocab_size=128, expert_parallel_size=1)
    with pytest.raises(C.PlannerUnsupported, match="LoopHybridStack"):
        C.ModelShape.from_model(torch.nn.Linear(2, 2), vocab_size=128, expert_parallel_size=1)


def _dsv4_symbol_model(pattern, core, ratios=None):
    """A stub of a built DSv4 hybrid loop model: ``layer_type_list`` holds the C / H / W / D
    symbols as HybridStack builds them, and ``_loop_flop_rows`` carries a ratio only for 'D'
    (``LoopHybridStack._compress_ratio``); MTP 'WE'. One Linear per sublayer."""
    symbols = list(pattern)
    ratios = ratios or {}
    stack = torch.nn.Module()
    stack.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in symbols])
    stack.config = _tiny_config(mtp_num_layers=1)
    stack.layer_type_list = symbols
    stack._loop_flop_rows = [(s, None, ratios.get(i)) for i, s in enumerate(symbols)]
    stack.loop_roles = SimpleNamespace(core_start=core[0], core_end=core[1])
    model = torch.nn.Module()
    model.decoder = stack
    model.mtp = torch.nn.Linear(2, 2)
    model.mtp_pattern = "WE"
    return model


def test_shape_from_a_dsv4_symbol_model():
    """DSV4-TINY's HEHECEHECEHECE/WE: the kinds come from the C / H / W symbols (their FLOP
    rows carry no ratio); a 'D' takes its ratio from the stack's FLOP rows."""
    shape = C.ModelShape.from_model(
        _dsv4_symbol_model("HEHECEHECEHECE", (2, 10)), vocab_size=16, expert_parallel_size=1
    )
    tiny = C.ModelShape.from_dims(TINY_DIMS, (1, 4, 2), C.proxy7_ratios())
    assert shape.prelude == (("HCA", "moe"),) == tiny.prelude
    assert shape.core == (("HCA", "moe"), ("CSA", "moe")) * 2 == tiny.core
    assert shape.coda == (("HCA", "moe"), ("CSA", "moe")) == tiny.coda
    assert shape.mtp == (("W", "moe"),) == tiny.mtp
    d_model = _dsv4_symbol_model("HEDEWE", (2, 4), ratios={2: 4})
    assert C.ModelShape.from_model(d_model, vocab_size=16, expert_parallel_size=1).core == (
        ("CSA", "moe"),
    )
    with pytest.raises(C.PlannerUnsupported, match="compress ratio None has no planner cost row"):
        C.ModelShape.from_model(
            _dsv4_symbol_model("HEDEWE", (2, 4)), vocab_size=16, expert_parallel_size=1
        )
    with pytest.raises(C.PlannerUnsupported, match="sublayer 'M' has no planner cost row"):
        C.ModelShape.from_model(
            _dsv4_symbol_model("HEMEWE", (2, 4)), vocab_size=16, expert_parallel_size=1
        )


def test_shape_from_model_registers_the_mtp_kind_and_runtime_bytes():
    """The MTP layer's kind gets its dense count (DSV4-TINY's MTP 'W' appears in no main
    layer; the Layer window charges gathers per kind), and the runtime byte defaults apply."""
    model = _dsv4_symbol_model("HEHECEHECEHECE", (2, 10))
    shape = C.ModelShape.from_model(model, vocab_size=16, expert_parallel_size=1)
    # The stub's MTP block is one Linear(2, 2): no per-depth inner layer, the whole block.
    assert dict(shape.params.dense_by_kind) == {"HCA": 12, "CSA": 12, "W": 6}
    assert shape.grace_optimizer_bytes_per_param == 16
    assert shape.ce_bytes_per_token == 10.0 * 16 * 2  # main head + one MTP head
    override = C.ModelShape.from_model(
        model, vocab_size=16, expert_parallel_size=1, ce_bytes_per_token=3.0
    )
    assert override.ce_bytes_per_token == 3.0
    with pytest.raises(TypeError, match="expert_parallel_size"):
        C.ModelShape.from_model(model, vocab_size=16)  # pylint: disable=missing-kwoa
    with pytest.raises(ValueError, match="expert_parallel_size must be >= 1"):
        C.ModelShape.from_model(model, vocab_size=16, expert_parallel_size=0)


def test_plan_on_a_from_model_shape_with_both_layouts():
    """``LoopRSBFPlanner`` on a ``from_model`` shape with the default layouts (resident +
    Layer window): the Layer-window evaluation reaches the MTP kind."""
    model = _dsv4_symbol_model("HEHECEHECEHECE", (2, 10))
    shape = C.ModelShape.from_model(model, vocab_size=100352, expert_parallel_size=1)
    phase = P.PhaseGeometry(
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
    plan = P.LoopRSBFPlanner(shape, PlannerHardware.preset("h100")).plan(phase, P.R4_UNIF)
    assert {r.option.layout for r in plan.table} == {"resident", "layer_window"}
    assert plan.applied.fits and plan.applied.applicable
