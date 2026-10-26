# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The RSBF planner with the loop parameter store (``loop/planner.py``, ``loop/planner_costs.py``
and ``LoopParamStore.planner_terms`` / ``memory_terms``), on CPU.

* option -> store modes (``option_store_modes``) for every layout / move step / re-gather, the
  variant keys and their de-duplication;
* the shard axis (``PhaseGeometry.shard_axis``) as the dense axis, the move steps ``E`` / ``EM``
  without Adam to Grace (offered with the Layer-window layout only);
* the runtime memory model: ``memory_components(..., store_terms=...)`` equals the store's
  ``memory_report`` byte for byte (dense groups, shards, slots and scratch) for every mode set
  and item form, on a CPU-built store; terms of hypothetical modes leave the store unchanged;
  ``ModelShape.from_model`` counts the same groups before and after the install and after a
  step (compute parameters unbound);
* a variant the store cannot run (a block plan splitting its layer units) refuses its options;
* on a no-Grace preset (h100) with an HBM cap only the E + M option fits, the plan picks it;
  the plan's store modes in the JSON, the digest and the decision table; ``apply_plan_to_store``.

The cost-model cross-check rows (``store_terms=None``) are ``test_loop_planner_costs.py`` /
``test_loop_planner_cost_model.py`` (unchanged). Model parallel is initialized once for the
module (gloo world of one rank on CPU); the store is the trivial ``z = 1`` store
(``allow_trivial_shard``), whose memory accounting is the same as at ``z > 1`` per rank.
"""

import dataclasses

import pytest

from megatron.core.transformer.loop import planner as P
from megatron.core.transformer.loop import planner_costs as C
from megatron.core.transformer.loop.param_store import ResidencyMode, StoreModes, store_variant_key
from megatron.core.transformer.loop.param_store_binding import named_logical_parameters
from megatron.core.transformer.loop.program import CoreBackwardPlan
from tests.unit_tests.transformer.loop import param_store_utils as psu
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    SEQ,
    VOCAB,
    build_local_loop_model,
    model_parallel_once,
)

DENSE = "*-*-*-*-*-*-"  # prelude '*-', core '*-*-*-*-' (L_core = 4 layers), coda '*-'
CORE_SUBLAYERS = 8
R, RSG, LW = (
    ResidencyMode.RESIDENT,
    ResidencyMode.RESIDENT_SHARDED_GRADS,
    ResidencyMode.LAYER_WINDOW,
)
DEPTH = P.DepthDistribution.pmf((0.25, 0.25, 0.25, 0.25))


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


def _model():
    return build_local_loop_model(DENSE, prelude=2, coda=2)


def _store(model):
    return psu.install_store(model, psu.trivial_store_groups(), 1, allow_trivial_shard=True)


def _phase(**overrides):
    fields = dict(
        seq_length=4096,
        gpus=8,
        ep=1,
        samples_per_eu=8,
        max_rounds=4,
        lockstep_eus=8,
        microbatches_per_step=8,
        precision="bf16",
        graphs="eager",
        ep_overlap="raw",
        shard_axis=8,
    )
    fields.update(overrides)
    return P.PhaseGeometry(**fields)


def _caps(**overrides):
    fields = dict(layer_window=True, coda_recompute=False, optimizer_offload_reason="no Grace")
    fields.update(overrides)
    return P.PlannerCapabilities(**fields)


def _store_shape(model, store, planner_kwargs, phase):
    shape = C.ModelShape.from_model(model, vocab_size=VOCAB, expert_parallel_size=1)
    planner = P.LoopRSBFPlanner(shape, P.PlannerHardware.preset("h100"), **planner_kwargs)
    variants = P.store_planner_variants(planner.options(phase), shape.core_layers)
    return dataclasses.replace(shape, store_terms=store.planner_terms(variants))


# ----------------------------------------------------------------------------- option -> modes
@pytest.mark.parametrize(
    "layout,moves,regather,expected",
    [
        ("resident", "-", False, StoreModes()),
        ("resident", "A", False, StoreModes()),
        ("layer_window", "-", False, StoreModes(core=LW)),
        ("layer_window", "A", True, StoreModes(core=LW, core_regather=True)),
        ("layer_window", "E", False, StoreModes(core=LW, prelude=LW, coda=LW, mtp=LW)),
        ("layer_window", "AE", False, StoreModes(core=LW, prelude=LW, coda=LW, mtp=LW)),
        (
            "layer_window",
            "EM",
            False,
            StoreModes(core=LW, prelude=LW, coda=LW, mtp=LW, emb_head=RSG),
        ),
        (
            "layer_window",
            "AEM",
            True,
            StoreModes(core=LW, prelude=LW, coda=LW, mtp=LW, emb_head=RSG, core_regather=True),
        ),
    ],
)
def test_option_store_modes(layout, moves, regather, expected):
    form = "block"
    option = P.PlannerOption(form, 2, regather, False, 1, moves, layout)
    modes = P.option_store_modes(option)
    assert modes == expected
    assert not modes.edge_regather  # kept edges: one gather per edge unit per item
    # The key carries the modes and the item plan (block 2 layers = 4 core sublayers).
    assert P.option_store_key(option, 4) == store_variant_key(modes, "block", 4)
    assert P.option_store_key(dataclasses.replace(option, stride_layers=4), 4) == (
        store_variant_key(modes, "whole_round", None)
    )


def test_store_variants_are_distinct_modes_and_item_plans():
    planner = P.LoopRSBFPlanner(
        C.ModelShape.from_dims(C.PRO_DIMS, (1, 4, 2), C.proxy7_ratios()),
        P.PlannerHardware.preset("gb200"),
        capabilities=_caps(),
        strides_layers=(1, 2, 4),
        waves=(1, 2),
        move_steps=P.STORE_MOVE_STEPS,
    )
    options = planner.options(_phase(samples_per_eu=8, microbatches_per_step=8, gpus=64))
    variants = P.store_planner_variants(options, 4)
    keys = [store_variant_key(*variant) for variant in variants]
    assert len(keys) == len(set(keys))
    # Waves, D42 and Adam to Grace are not store variants: every option maps onto one.
    assert {P.option_store_key(o, 4) for o in options} == set(keys)
    # nested + 3 block strides (whole round at 4), resident (1 mode set) and layer window
    # (- / E / EM, w and r: 6 mode sets; nested has no r).
    assert len(keys) == 4 * 1 + 4 * 3 + 3 * 3


# ----------------------------------------------------------------------------- phase and menu
def test_shard_axis_is_the_dense_axis():
    assert _phase(shard_axis=None).dense_axis == 8
    assert _phase(shard_axis=4).dense_axis == 4
    assert _phase(shard_axis=4, domain_gpus=8).dense_axis == 4
    _phase(shard_axis=2).validate()
    with pytest.raises(P.PlannerUnsupported, match="shard axis z = 3"):
        _phase(shard_axis=3).validate()


def test_move_steps_e_and_em_without_adam_to_grace():
    shape = C.ModelShape.from_dims(C.PRO_DIMS, (1, 4, 2), C.proxy7_ratios())
    planner = P.LoopRSBFPlanner(
        shape,
        P.PlannerHardware.preset("h100"),
        capabilities=_caps(),
        strides_layers=(4,),
        waves=(1,),
        move_steps=P.STORE_MOVE_STEPS,
    )
    options = planner.options(_phase(gpus=64, samples_per_eu=8, microbatches_per_step=8))
    by_layout = {}
    for option in options:
        by_layout.setdefault(option.layout, set()).add(option.moves)
    assert by_layout["resident"] == {"-", "A"}
    assert by_layout["layer_window"] == set(P.STORE_MOVE_STEPS)
    assert P.PlannerOption("nested", None, False, False, 1, "EM", "layer_window").requires() == (
        frozenset({"layer_window"})
    )
    with pytest.raises(ValueError, match="unknown move step 'M'"):
        P.LoopRSBFPlanner(shape, P.PlannerHardware.preset("h100"), move_steps=("-", "M"))


# ----------------------------------------------------------------------------- runtime terms
MODE_SETS = (
    "all=r",
    "core=lw",
    "core=lw,edges=lw",
    "core=lw,edges=lw,emb_head=rsg",
    "all=rsg,emb_head=rsg",
    "core=lw,edges=rsg,emb_head=rsg,core_regather=on",
)
FORMS = (("whole_round", None), ("block", 4), ("nested", 2))


@pytest.mark.parametrize("text", MODE_SETS)
@pytest.mark.parametrize("form,stride", FORMS)
def test_runtime_memory_components_equal_the_store(text, form, stride):
    model = _model()
    store = _store(model)
    store.configure_modes(StoreModes(), core_plan=CoreBackwardPlan.whole_round(CORE_SUBLAYERS))
    configured = (store.modes, store.slot_count, store.scratch_count, store.core_plan)
    modes = StoreModes.parse(text)
    core_plan = psu.core_plan_of(form, CORE_SUBLAYERS, stride or 2)
    terms = store.memory_terms(modes, core_plan=core_plan)
    report = store.memory_report(modes, core_plan=core_plan)
    # Hypothetical modes change nothing in the store.
    assert (store.modes, store.slot_count, store.scratch_count, store.core_plan) == configured
    assert terms.modes == modes.describe() and terms.item_form == form
    assert terms.block_stride == (None if form == "whole_round" else stride)
    assert dict(terms.group_modes)["core"] == modes.core.value

    shape = C.ModelShape.from_model(model, vocab_size=VOCAB, expert_parallel_size=1)
    ph = C.SeqPhase.of(shape, 4096, 1)
    hw = P.PlannerHardware.preset("h100")
    block = form == "block"
    comps, _ = C.memory_components(
        shape,
        ph,
        hw,
        0.0,
        EP=1,
        N=8,
        prec="bf16",
        layout="layer_window" if modes.core is LW else "resident",
        nested=form == "nested",
        block_stride=(stride // 2) if block else None,
        block_window=block and not modes.core_regather,
        store_terms=terms,
    )
    # Window slots + fp32 scratch pool: exactly the store's buffers.
    assert comps[C.MEM_DENSE_BUF] == report.slot_bytes + report.scratch_bytes
    assert C.MEM_WINDOW not in comps  # the slots hold a kept block's units
    # Shards + fp32 shard gradients of every stored group, every mode.
    assert comps[C.MEM_DENSE_SH] == sum(report.shard_bytes.values())
    # Compute copies + accumulators by mode, plus the replicated rest (weight + fp32 grad).
    stored = {id(t.compute) for t in store.tensors()}
    replicated = sum(
        p.numel() * (p.element_size() + 4)
        for _, p in named_logical_parameters(model)
        if id(p) not in stored and getattr(p, "allreduce", True)
    )
    assert comps[C.MEM_DENSE_RES] == (
        replicated
        + sum(report.resident_copy_bytes.values())
        + sum(report.accumulator_bytes.values())
    )
    # The distributed optimizer over the replicate group (N / z ranks; z = 1 here).
    stored_numel = sum(t.numel for t in store.tensors())
    shards = sum(math_prod(t.shard_shape) for t in store.tensors())
    dense = sum(shape.params.groups().values())
    assert comps[C.MEM_OPT] == pytest.approx(12 * (shards + dense - stored_numel) / 8, rel=1e-12)
    # The cost-model terms (no store) differ: no shards in the resident layout, 12 pl_max.
    cost, _ = C.memory_components(
        shape, ph, hw, 0.0, EP=1, N=8, prec="bf16", layout="resident", nested=True
    )
    assert cost[C.MEM_DENSE_SH] == 0.0 and cost[C.MEM_DENSE_BUF] == 0.0


def math_prod(shape):
    out = 1
    for size in shape:
        out *= size
    return out


def test_coda_and_prelude_arenas_carry_the_store_transients():
    model = build_local_loop_model(DENSE, prelude=2, coda=2, vocab_size=8 * VOCAB)
    store = _store(model)
    modes = StoreModes.parse("core=lw,edges=lw,emb_head=rsg")
    terms = store.memory_terms(modes, core_plan=CoreBackwardPlan.whole_round(CORE_SUBLAYERS))
    # The output layer (8 x vocab) is larger than a pool entry: a transient in the coda.
    assert terms.head_scratch_coda > 0 and terms.rsg_transient_prelude > 0
    shape = C.ModelShape.from_model(model, vocab_size=8 * VOCAB, expert_parallel_size=1)
    ph = C.SeqPhase.of(shape, 16, 1)  # a tiny T: the store's terms dominate the arena
    hw = P.PlannerHardware.preset("h100")
    kwargs = dict(EP=1, N=8, prec="bf16", layout="layer_window", nested=False, block_stride=4)
    with_terms, _ = C.memory_components(shape, ph, hw, 0.0, store_terms=terms, **kwargs)
    zeroed = dataclasses.replace(
        terms,
        head_scratch_coda=0.0,
        rsg_transient_coda=0.0,
        hook_grad_coda=0.0,
        rsg_transient_prelude=0.0,
        hook_grad_prelude=0.0,
    )
    without, _ = C.memory_components(shape, ph, hw, 0.0, store_terms=zeroed, **kwargs)
    act = with_terms[C.MEM_ACT_BLOCK] - without[C.MEM_ACT_BLOCK]
    assert act > 0
    coda_extra = terms.head_scratch_coda + terms.rsg_transient_coda + terms.hook_grad_coda
    prelude = terms.rsg_transient_prelude + terms.hook_grad_prelude
    assert with_terms[C.MEM_ACT_BLOCK] >= max(prelude, coda_extra)


def test_from_model_counts_logical_parameters_through_a_step():
    model = _model()
    before = C.ModelShape.from_model(model, vocab_size=VOCAB, expert_parallel_size=1)
    store = _store(model)
    after = C.ModelShape.from_model(model, vocab_size=VOCAB, expert_parallel_size=1)
    assert after.params == before.params
    # A step leaves the compute parameters unbound (logical-shape placeholders).
    plan = CoreBackwardPlan.whole_round(CORE_SUBLAYERS)
    store.configure_modes(StoreModes.parse("core=lw,edges=lw,emb_head=rsg"), core_plan=plan)
    driver = psu.StoreDriver(model, store, depth=2, core_plan=plan)
    driver.run_step([psu.make_samples(0, 1, vocab=VOCAB, seq=SEQ)], 1)
    assert any(t.bound is None for t in store.tensors())
    stepped = C.ModelShape.from_model(model, vocab_size=VOCAB, expert_parallel_size=1)
    assert stepped.params == before.params
    # The mirror's canonical shards are never counted (they are registered parameters).
    assert any("loop_param_store.mirror" in name for name, _ in model.named_parameters())


# ----------------------------------------------------------------------------- planner
def test_store_terms_refuse_a_variant_the_store_cannot_run():
    model = _model()
    store = _store(model)
    lw = StoreModes(core=LW)
    terms = store.planner_terms([(lw, "block", 1), (lw, "block", 2), (lw, "block", 2)])
    split = terms[store_variant_key(lw, "block", 1)]
    assert isinstance(split, str) and "the parameter store cannot run it" in split
    assert not isinstance(terms[store_variant_key(lw, "block", 2)], str)
    assert len(terms) == 2
    # An option whose variant the store refused is not applicable (the refusal is listed).
    shape = dataclasses.replace(
        C.ModelShape.from_model(model, vocab_size=VOCAB, expert_parallel_size=1),
        store_terms={store_variant_key(lw, "block", 2): split},
    )
    planner = P.LoopRSBFPlanner(
        shape,
        P.PlannerHardware.preset("h100"),
        capabilities=_caps(),
        strides_layers=(1,),
        waves=(1,),
        move_steps=("-",),
        layouts=("layer_window",),
        form_policy="planner",
    )
    results = [r for r in planner.evaluate(_phase(), DEPTH) if r.option.form == "block"]
    assert results and all(not r.applicable for r in results)
    assert split in results[0].refusals
    missing = [r for r in planner.evaluate(_phase(), DEPTH) if r.option.form == "nested"]
    assert missing and "no parameter store terms" in missing[0].refusals[-1]


def _planner(shape, hardware, **kwargs):
    settings = dict(
        capabilities=_caps(),
        strides_layers=(4,),
        waves=(1,),
        move_steps=("-", "E", "EM"),
        form_policy="planner",
    )
    settings.update(kwargs)
    return P.LoopRSBFPlanner(shape, hardware, **settings)


def test_no_grace_preset_picks_e_and_m_when_they_are_needed_to_fit():
    model = _model()
    store = _store(model)
    phase = _phase()
    shape = _store_shape(
        model,
        store,
        dict(capabilities=_caps(), strides_layers=(4,), waves=(1,), move_steps=("-", "E", "EM")),
        phase,
    )
    hw = P.PlannerHardware.preset("h100")
    results = _planner(shape, hw).evaluate(phase, DEPTH)
    hbm = {(r.option.label, r.option.moves, r.option.layout): r.hbm_bytes for r in results}
    em = min(v for (_, moves, _), v in hbm.items() if moves == "EM")
    others = min(v for (_, moves, _), v in hbm.items() if moves != "EM")
    assert em < others  # E + M store the edges as shards and the head gradient sharded
    cap = (em + others) / 2
    tight = P.PlannerHardware.preset("h100", planner_reserve_bytes=hw.hbm_bytes - cap)
    plan = _planner(shape, tight).plan(phase, DEPTH)
    assert plan.applied.option.moves == "EM" and plan.applied.option.layout == "layer_window"
    assert not plan.optimizer_offload  # h100: no Grace tier
    assert plan.store_modes() == StoreModes(core=LW, prelude=LW, coda=LW, mtp=LW, emb_head=RSG)
    assert plan.unavailable == ()  # the store makes every move applicable
    report = plan.to_json()
    assert report["store_modes"] == plan.store_modes().describe()
    table = P.format_decision_table(plan)
    assert f"store modes: {plan.store_modes().describe()}" in table
    assert "requested but unavailable" not in table
    assert "parameter store: moves E / EM" in table
    assert P.option_store_modes(plan.applied.option) == plan.store_modes()
    # Without the store's capability the same moves are target-only.
    no_store = _planner(shape, tight, capabilities=_caps(layer_window=False))
    with pytest.raises(P.PlannerNoFit):
        no_store.plan(phase, DEPTH)


def test_apply_plan_to_store():
    model = _model()
    store = _store(model)
    phase = _phase()
    kwargs = dict(capabilities=_caps(), strides_layers=(4,), waves=(1,), move_steps=("E",))
    shape = _store_shape(model, store, kwargs, phase)
    plan = _planner(shape, P.PlannerHardware.preset("h100"), **kwargs).plan(phase, DEPTH)
    core_plan = psu.core_plan_of(plan.item_form, CORE_SUBLAYERS, plan.block_stride or 2)
    modes = P.apply_plan_to_store(store, plan, core_plan=core_plan)
    assert store.modes == modes == plan.store_modes()
    assert store.core_plan == core_plan
    # The store's runtime terms of the applied variant are what the planner charged.
    applied = shape.store_terms[P.option_store_key(plan.applied.option, shape.core_layers)]
    assert store.memory_terms() == applied
    without = dataclasses.replace(plan, capabilities=_caps(layer_window=False))
    with pytest.raises(P.PlannerUnsupported, match="without the parameter store"):
        P.apply_plan_to_store(store, without, core_plan=core_plan)
