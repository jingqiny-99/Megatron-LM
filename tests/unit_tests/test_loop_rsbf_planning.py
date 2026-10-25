# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The training wiring of the RSBF loop planner (megatron/training/loop_rsbf_planning.py,
``--loop-planner``), on CPU:

* the planner inputs from the training arguments: phase, depth distribution, hardware and
  overrides, capabilities (Adam to Grace per ``--loop-planner-optimizer-offload`` and its
  prerequisites, D42 from the model), the runtime planner settings;
* the CLI flags and ``_validate_loop_planner``;
* ``plan_loop_rsbf`` on a CPU-built dense ``'*-'`` loop model (local specs, gloo, model parallel
  initialized once for the module) in report and apply modes: the config fields and their
  ``args`` mirror, the waves re-validation, Adam to Grace applied only when the applied option
  has it, refusals (user-set RSBF flags, a cached executor, nothing fits), the decision table and
  the JSON report, a phase-start rerun that drops the model's cached executor;
* ``check_resume_offload``; the digest agreement over a spawned two-rank Gloo world;
* the hooks in ``setup_model_and_optimizer`` and ``load_checkpoint`` (source order).

The GPU counterpart (DSV4-TINY at EP = W through the executor, the forced Adam-to-Grace arm) is
``tests/unit_tests/transformer/loop/test_loop_planner_apply.py``.
"""

import ast
import contextlib
import inspect
import json
import sys
import textwrap
from argparse import Namespace

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.loop import planner as P
from megatron.core.transformer.loop.planner_costs import GB
from megatron.core.transformer.loop.rsbf_executor import get_loop_rsbf_schedule
from megatron.training import loop_rsbf_planning as planning
from megatron.training.arguments import _validate_loop_planner, parse_args
from megatron.training.utils import is_rank0
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    build_local_loop_model,
    model_parallel_once,
)
from tests.unit_tests.transformer.loop.multi_eu_utils import (
    SPAWN_SKIP_REASON,
    run_spawned,
    spawn_tests_enabled,
)

DENSE = "*-*-*-*-*-*-"  # prelude '*-', core '*-*-*-*-' (L_core = 4 layers), coda '*-'
RSBF = dict(
    loop_schedule="rsbf",
    calculate_per_token_loss=True,
    loop_iterations=4,
    loop_exit_criterion="sampled",
    loop_depth_probs=[0.25, 0.25, 0.25, 0.25],
)

# The parser defaults of the planner group (test_cli_defaults checks them against parse_args).
PLANNER_DEFAULTS = dict(
    loop_planner="off",
    loop_planner_hardware="gb200",
    loop_planner_hbm_cap_gb=None,
    loop_planner_grace_gb=None,
    loop_planner_grace_margin_gb=None,
    loop_planner_ep_reserve_gb=None,
    loop_planner_c2c_gbps=None,
    loop_planner_c2c_margin=None,
    loop_planner_strides=[1, 2, 3, 4, 6, 9, 18],
    loop_planner_waves=[1, 2, 4, 8],
    loop_planner_depths="config",
    loop_planner_histogram_path=None,
    loop_planner_act_bytes_per_token=None,
    loop_planner_optimizer_offload="auto",
    loop_planner_forms="d28",
    loop_planner_small_core_layers=4,
    loop_planner_report_path=None,
)


def _args(**overrides):
    """Training arguments of an 8-GPU RSBF run of the dense loop (TP = PP = CP = 1, DP = 8,
    MBS 1, GBS 64: 8 samples and microbatches per EU), the planner flags at their defaults."""
    values = dict(PLANNER_DEFAULTS)
    values.update(
        seq_length=4096,
        world_size=8,
        data_parallel_size=8,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        micro_batch_size=1,
        global_batch_size=64,
        loop_architecture="huginn",
        loop_prelude_sublayers=2,
        loop_coda_sublayers=2,
        loop_rsbf_item_form="whole_round",
        loop_rsbf_block_stride=None,
        loop_rsbf_waves=1,
        loop_rsbf_coda_recompute_mla_up_proj=False,
        optimizer_cpu_offload=False,
        optimizer_offload_fraction=1.0,
        use_precision_aware_optimizer=False,
        optimizer="adam",
        use_distributed_optimizer=True,
        accumulate_allreduce_grads_in_fp32=True,
        **RSBF,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.fixture(scope="module")
def _model_parallel():
    with model_parallel_once():
        try:
            yield
        finally:
            get_loop_rsbf_schedule().reset()


@pytest.fixture
def session():
    planning.reset_loop_planner_session()
    try:
        yield
    finally:
        planning.reset_loop_planner_session()


def _rank0_warns(*args, **kwargs):
    """``pytest.warns`` on rank 0, where ``warn_rank_0`` warns; a no-op context elsewhere."""
    return pytest.warns(*args, **kwargs) if is_rank0() else contextlib.nullcontext()


def _model(**overrides):
    fields = dict(RSBF)
    fields.update(overrides)
    return build_local_loop_model(DENSE, prelude=2, coda=2, **fields)


# ----------------------------------------------------------------------------- inputs
def test_phase_from_the_training_arguments():
    phase = planning.loop_planner_phase(_args())
    assert phase == P.PhaseGeometry(
        seq_length=4096,
        gpus=8,
        ep=1,
        samples_per_eu=8,
        max_rounds=4,
        cp=1,
        lockstep_eus=8,
        microbatches_per_step=8,
        precision="bf16",
        graphs="eager",
        ep_overlap="raw",
        waves_allowed=True,
    )
    # n = MBS x microbatches; the lockstep unit is every EU of the domain.
    phase = planning.loop_planner_phase(_args(micro_batch_size=2, expert_model_parallel_size=8))
    assert (phase.samples_per_eu, phase.microbatches, phase.ep, phase.lockstep) == (8, 4, 8, 8)
    phase = planning.loop_planner_phase(
        _args(world_size=8, data_parallel_size=4, context_parallel_size=2)
    )
    assert (phase.cp, phase.eus, phase.lockstep, phase.samples_per_eu) == (2, 4, 4, 16)
    # MXFP8 needs FP8 on (the recipe defaults to 'delayed' without --fp8-format).
    assert planning.loop_planner_phase(_args(fp8=None, fp8_recipe="mxfp8")).precision == "bf16"
    assert planning.loop_planner_phase(_args(fp8="e4m3", fp8_recipe="mxfp8")).precision == "mxfp8"
    assert planning.loop_planner_phase(_args(fp8="e4m3", fp8_recipe="delayed")).precision == "bf16"
    for changing in (
        dict(step_batch_size_schedule="0:32 10:64"),
        dict(decrease_batch_size_if_needed=True),
        dict(rampup_batch_size=[16, 16, 1000]),
    ):
        assert not planning.loop_planner_phase(_args(**changing)).waves_allowed, changing


def test_depths_from_the_config_histogram_and_r4(tmp_path):
    args = _args()
    assert planning.loop_planner_depths(args).params == (0.25, 0.25, 0.25, 0.25)
    # The config (the model's) wins over args; loop_min_rounds folds the low depths.
    config = Namespace(**dict(RSBF, loop_min_rounds=2))
    assert planning.loop_planner_depths(args, config).params == (0.0, 0.5, 0.25, 0.25)
    fixed = Namespace(loop_iterations=3, loop_exit_criterion="fixed")
    assert planning.loop_planner_depths(args, fixed).params == (0.0, 0.0, 1.0)
    for name, setting in (("r4_geom", P.R4_GEOM), ("r4_unif", P.R4_UNIF), ("r4_skew", P.R4_SKEW)):
        assert planning.loop_planner_depths(_args(loop_planner_depths=name)) is setting
    with pytest.raises(ValueError, match="R_max 4, the run has loop_iterations 3"):
        planning.loop_planner_depths(_args(loop_planner_depths="r4_skew"), fixed)
    path = tmp_path / "exit_hist.jsonl"
    line = dict(format=P.EXIT_HISTOGRAM_FORMAT, iteration=1, R_max=4, counts=[8, 4, 2, 2])
    path.write_text(json.dumps(dict(line, samples=16)) + "\n")
    measured = planning.loop_planner_depths(
        _args(loop_planner_depths="histogram", loop_planner_histogram_path=str(path))
    )
    assert measured.params == (0.5, 0.25, 0.125, 0.125)
    with pytest.raises(ValueError, match="R_max 4 in the file, the run has R_max 3"):
        planning.loop_planner_depths(
            _args(loop_planner_depths="histogram", loop_planner_histogram_path=str(path)), fixed
        )
    with pytest.raises(ValueError, match="needs --loop-planner-histogram-path"):
        planning.loop_planner_depths(_args(loop_planner_depths="histogram"))


def test_hardware_presets_and_overrides():
    assert planning.loop_planner_hardware(_args()) == P.PlannerHardware.preset("gb200")
    h100 = planning.loop_planner_hardware(_args(loop_planner_hardware="h100"))
    assert h100 == P.PlannerHardware.preset("h100")
    assert h100.hbm_cap_bytes == 75 * GB and h100.grace_bytes == 0
    hw = planning.loop_planner_hardware(
        _args(
            loop_planner_hbm_cap_gb=150.0,
            loop_planner_grace_gb=180.0,
            loop_planner_grace_margin_gb=4.0,
            loop_planner_ep_reserve_gb=32.0,
            loop_planner_c2c_gbps=80.0,
            loop_planner_c2c_margin=1.5,
        )
    )
    assert hw.hbm_cap_bytes == 150 * GB and hw.planner_reserve_bytes == 30 * GB
    assert (hw.grace_bytes, hw.grace_margin_bytes, hw.ep_reserve_bytes) == (
        180 * GB,
        4 * GB,
        32 * GB,
    )
    assert (hw.c2c_bytes_per_s, hw.c2c_margin) == (80 * GB, 1.5)


def test_optimizer_offload_capability_and_its_reasons():
    gb200 = P.PlannerHardware.preset("gb200")
    h100 = P.PlannerHardware.preset("h100")
    ready = _args(use_precision_aware_optimizer=True)
    assert planning.loop_optimizer_offload_blockers(ready, gb200) == []
    caps = planning.loop_planner_capabilities(ready, hardware=gb200)
    assert caps.optimizer_offload and caps.optimizer_offload_reason == ""
    assert not caps.layer_window and not caps.grace_state_tier
    reasons = planning.loop_optimizer_offload_blockers(
        _args(
            optimizer="sgd",
            decoupled_weight_decay=False,
            chunked_optimizer_state_offload=True,
            fp8_param_gather=True,
            fp8_recipe="mxfp8",
        ),
        h100,
    )
    assert [r.split(" ")[0] for r in reasons] == [
        "no",
        "--use-precision-aware-optimizer",
        "--optimizer",
        "CPU",
        "--chunked-optimizer-state-offload",
        "--fp8-param-gather",
    ]
    caps = planning.loop_planner_capabilities(_args(), hardware=h100)
    assert not caps.optimizer_offload
    assert caps.optimizer_offload_reason.startswith("no Grace tier (h100;")
    assert "--use-precision-aware-optimizer not set" in caps.optimizer_offload_reason
    never = planning.loop_planner_capabilities(
        _args(use_precision_aware_optimizer=True, loop_planner_optimizer_offload="never"),
        hardware=gb200,
    )
    assert not never.optimizer_offload and "never" in never.optimizer_offload_reason
    always = planning.loop_planner_capabilities(
        _args(loop_planner_optimizer_offload="always"), hardware=gb200
    )
    assert always.optimizer_offload
    # D42 without a model: from the shape (the dense loop has no DSv4 attention).
    tiny = P.costs.ModelShape.from_dims(P.costs.PRO_DIMS, (1, 4, 2), P.costs.proxy7_ratios())
    assert planning.loop_planner_capabilities(_args(), hardware=gb200, shape=tiny).coda_recompute
    assert not planning.loop_planner_capabilities(_args(), hardware=gb200).coda_recompute


def test_runtime_planner_settings():
    tiny = P.costs.ModelShape.from_dims(P.costs.PRO_DIMS, (1, 4, 2), P.costs.proxy7_ratios())
    hw = P.PlannerHardware.preset("gb200")
    caps = P.PlannerCapabilities()
    planner = planning.make_loop_planner(
        _args(
            loop_planner_strides=[1, 2],
            loop_planner_waves=[1, 2],
            loop_planner_forms="nested",
            loop_planner_small_core_layers=2,
        ),
        tiny,
        hw,
        caps,
    )
    assert planner.strides_layers == (1, 2) and planner.waves == (1, 2)
    assert planner.form_policy == "nested" and planner.small_core_layers == 2
    assert planner.sched_policy == "executor"
    assert planner.rebuild_fraction == P.executor_nested_rebuild_fraction(4) == 0.75
    assert planner.move_steps == ("-", "A", "AE", "AEM")
    always = planning.make_loop_planner(
        _args(loop_planner_optimizer_offload="always"), tiny, hw, caps
    )
    assert always.move_steps == ("A", "AE", "AEM")


def test_user_flags_are_the_non_default_owned_flags():
    assert planning.loop_planner_user_flags(_args()) == []
    assert planning.loop_planner_user_flags(Namespace()) == []
    user = planning.loop_planner_user_flags(
        _args(
            loop_rsbf_item_form="block",
            loop_rsbf_block_stride=2,
            loop_rsbf_waves=2,
            loop_rsbf_coda_recompute_mla_up_proj=True,
        )
    )
    assert user == [
        "--loop-rsbf-item-form='block'",
        "--loop-rsbf-block-stride=2",
        "--loop-rsbf-waves=2",
        "--loop-rsbf-coda-recompute-mla-up-proj=True",
    ]


# ----------------------------------------------------------------------------- CLI and validation
def test_cli_defaults_and_values(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test"])
    args = parse_args()
    assert {name: getattr(args, name) for name in PLANNER_DEFAULTS} == PLANNER_DEFAULTS
    monkeypatch.setattr(
        sys,
        "argv",
        "test --loop-planner apply --loop-planner-hardware h100 --loop-planner-hbm-cap-gb 60 "
        "--loop-planner-grace-gb 400 --loop-planner-grace-margin-gb 5 "
        "--loop-planner-ep-reserve-gb 4 --loop-planner-c2c-gbps 25 --loop-planner-c2c-margin 2 "
        "--loop-planner-strides 1 2 --loop-planner-waves 1 4 --loop-planner-depths histogram "
        "--loop-planner-histogram-path h.jsonl --loop-planner-act-bytes-per-token 1e5 "
        "--loop-planner-optimizer-offload always --loop-planner-forms planner "
        "--loop-planner-small-core-layers 2 --loop-planner-report-path plan.json".split(),
    )
    args = parse_args()
    assert {name: getattr(args, name) for name in PLANNER_DEFAULTS} == dict(
        loop_planner="apply",
        loop_planner_hardware="h100",
        loop_planner_hbm_cap_gb=60.0,
        loop_planner_grace_gb=400.0,
        loop_planner_grace_margin_gb=5.0,
        loop_planner_ep_reserve_gb=4.0,
        loop_planner_c2c_gbps=25.0,
        loop_planner_c2c_margin=2.0,
        loop_planner_strides=[1, 2],
        loop_planner_waves=[1, 4],
        loop_planner_depths="histogram",
        loop_planner_histogram_path="h.jsonl",
        loop_planner_act_bytes_per_token=1e5,
        loop_planner_optimizer_offload="always",
        loop_planner_forms="planner",
        loop_planner_small_core_layers=2,
        loop_planner_report_path="plan.json",
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        (dict(loop_planner="off", loop_planner_report_path="p"), "needs --loop-planner report"),
        (dict(loop_planner="off", loop_planner_histogram_path="h"), "needs --loop-planner report"),
        (dict(loop_schedule="standard"), "hybrid Huginn loop"),
        (dict(loop_architecture="mor"), "hybrid Huginn loop"),
        (dict(loop_prelude_sublayers=None), "hybrid Huginn loop"),
        (dict(loop_planner_hbm_cap_gb=0.0), "hbm-cap-gb must be > 0"),
        (dict(loop_planner_hbm_cap_gb=200.0), r"hbm-cap-gb must be in \(0, 180\]"),
        (dict(loop_planner_c2c_gbps=-1.0), "c2c-gbps must be > 0"),
        (dict(loop_planner_grace_gb=-1.0), "grace-gb must be >= 0"),
        (dict(loop_planner_waves=[0, 1]), "waves must be >= 1"),
        (dict(loop_planner_strides=[0]), "strides must be >= 1"),
        (dict(loop_planner_small_core_layers=-1), "small-core-layers"),
        (dict(loop_planner_depths="histogram"), "needs --loop-planner-histogram-path"),
        (dict(loop_planner_histogram_path="h"), "read with --loop-planner-depths histogram"),
        (dict(loop_planner_depths="r4_unif", loop_iterations=3), "R_max = 4 setting"),
        (dict(context_parallel_size=2), "needs --context-parallel-size 1"),
        (dict(loop_rsbf_waves=2), "set by the user: --loop-rsbf-waves=2"),
        (dict(loop_rsbf_item_form="nested"), "--loop-rsbf-item-form='nested'"),
        (dict(optimizer_cpu_offload=True), "pin it with --loop-planner-optimizer-offload always"),
        (
            dict(optimizer_cpu_offload=True, loop_planner_optimizer_offload="never"),
            "pin it with --loop-planner-optimizer-offload always",
        ),
        (
            dict(loop_planner_optimizer_offload="always", loop_planner_hardware="h100"),
            "always needs Adam to Grace to be possible: no Grace tier",
        ),
        (dict(loop_planner_optimizer_offload="always"), "--use-precision-aware-optimizer not set"),
    ],
)
def test_validator_rejections(overrides, match):
    args = _args(loop_planner="apply")
    for name, value in overrides.items():
        setattr(args, name, value)
    with pytest.raises(ValueError, match=match):
        _validate_loop_planner(args)


def test_validator_accepts():
    _validate_loop_planner(Namespace())  # off by default
    _validate_loop_planner(_args())
    for overrides in (
        dict(loop_planner="apply"),
        dict(loop_planner="apply", loop_rsbf_waves=1),  # the default value is not "set"
        dict(loop_planner="report", loop_rsbf_waves=2, loop_rsbf_item_form="nested"),
        dict(loop_planner="report", context_parallel_size=2, optimizer_cpu_offload=True),
        dict(loop_planner="apply", loop_planner_depths="r4_geom"),
        dict(
            loop_planner="apply",
            loop_planner_optimizer_offload="always",
            optimizer_cpu_offload=True,
            use_precision_aware_optimizer=True,
            loop_planner_hardware="h100",
            loop_planner_grace_gb=400.0,
        ),
    ):
        _validate_loop_planner(_args(**overrides))


# ----------------------------------------------------------------------------- plan_loop_rsbf
def test_apply_sets_the_config_and_args(_model_parallel, session, tmp_path, capsys, caplog):
    model = _model()
    path = tmp_path / "plans" / "plan.json"
    args = _args(
        loop_planner="apply", loop_planner_hardware="h100", loop_planner_report_path=str(path)
    )
    caplog.set_level("INFO", logger=planning.__name__)
    plan = planning.plan_loop_rsbf(args, [model], apply=True)
    # D28 on the 4-layer core: the whole round; no Grace tier, so no Adam to Grace.
    assert (plan.item_form, plan.block_stride, plan.waves) == ("whole_round", None, 1)
    assert not plan.optimizer_offload and not plan.coda_recompute
    for name, value in plan.config_overrides().items():
        assert getattr(model.config, name) == value
        assert getattr(model.decoder.config, name) == value
        assert getattr(args, name) == value
    assert args.optimizer_cpu_offload is False
    assert plan.capabilities.optimizer_offload_reason.startswith("no Grace tier")
    session_ = planning.get_loop_planner_session()
    assert session_.plan is plan and session_.phase == planning.loop_planner_phase(args)
    # The decision table, the "applied" line and the JSON report are rank 0's (``_log_plan``,
    # ``print_rank_0``); under a multi-rank torchrun the other ranks have none of them.
    if not is_rank0():
        return
    assert "[loop planner] phase S=4096 cp=1 N=8 EP=1 W=8" in caplog.text
    assert f"digest={plan.digest[:12]}" in caplog.text
    assert "loop planner: applied loop_rsbf_item_form=whole_round" in capsys.readouterr().out
    report = json.loads(path.read_text())
    assert report["digest"] == plan.digest and len(report["table"]) == len(plan.table)
    assert report["config_overrides"]["loop_rsbf_item_form"] == "whole_round"


def test_apply_a_planner_pick_that_is_not_the_default(_model_parallel, session):
    """Form policy 'nested' and waves 2 / 4 only: the config and args take the plan's nested
    item (stride 2) and its waves, re-validated against the microbatches."""
    model = _model()
    args = _args(
        loop_planner="apply",
        loop_planner_hardware="h100",
        loop_planner_forms="nested",
        loop_planner_waves=[2, 4],
    )
    plan = planning.plan_loop_rsbf(args, [model], apply=True)
    assert (plan.item_form, plan.block_stride) == ("nested", 2) and plan.waves in (2, 4)
    assert (model.config.loop_rsbf_item_form, model.config.loop_rsbf_block_stride) == ("nested", 2)
    assert args.loop_rsbf_waves == model.config.loop_rsbf_waves == plan.waves
    from megatron.core.transformer.loop.program import LoopProgram

    assert LoopProgram(model).item_form == "nested"


def _hbm_of(plan, moves):
    (row,) = [
        r
        for r in plan.table
        if r.option.layout == "resident"
        and r.option.moves == moves
        and r.option.waves == 1
        and P.option_item_form(r.option, plan.core_layers) == ("whole_round", None)
    ]
    return row.hbm_bytes


def test_apply_adam_to_grace_only_when_the_applied_option_has_it(_model_parallel, session):
    """A host tier (``--loop-planner-grace-gb``) and an HBM cap between the two whole-round
    options: only Adam to Grace fits, so it is applied (``optimizer_cpu_offload``, fraction 1);
    the same cap without the precision-aware optimizer leaves no whole round that fits (auto
    excludes Adam to Grace; only smaller block strides fit, which D28 excludes on this core)."""
    model = _model()
    base = dict(
        loop_planner="apply",
        loop_planner_hardware="h100",
        loop_planner_grace_gb=400.0,
        loop_planner_ep_reserve_gb=0.0,
        loop_planner_waves=[1],
    )
    probe = planning.LoopPlannerSession().on_phase_start(
        _args(**dict(base, loop_planner="report"), use_precision_aware_optimizer=True),
        [model],
        apply=False,
    )
    with_adam, without = _hbm_of(probe, "A"), _hbm_of(probe, "-")
    assert with_adam < without
    cap = (with_adam + without) / 2 / GB
    args = _args(**base, loop_planner_hbm_cap_gb=cap, use_precision_aware_optimizer=True)
    plan = planning.plan_loop_rsbf(args, [model], apply=True)
    assert plan.optimizer_offload and plan.applied.option.moves == "A"
    assert args.optimizer_cpu_offload is True and args.optimizer_offload_fraction == 1.0
    planning.reset_loop_planner_session()
    blocked = _args(**base, loop_planner_hbm_cap_gb=cap)
    with pytest.raises(P.PlannerNoFit, match="form policy d28 excluded every fitting option"):
        planning.plan_loop_rsbf(blocked, [model], apply=True)
    assert blocked.optimizer_cpu_offload is False
    # Report mode logs the misfit instead of raising.
    planning.reset_loop_planner_session()
    with _rank0_warns(UserWarning, match="excluded every fitting option"):
        assert (
            planning.plan_loop_rsbf(
                _args(**dict(base, loop_planner="report"), loop_planner_hbm_cap_gb=cap),
                [model],
                apply=False,
            )
            is None
        )


def test_apply_refuses_user_flags_and_a_built_executor(_model_parallel, session, monkeypatch):
    model = _model()
    args = _args(loop_planner="apply", loop_planner_hardware="h100", loop_rsbf_waves=2)
    with pytest.raises(ValueError, match="set by the user: --loop-rsbf-waves=2"):
        planning.plan_loop_rsbf(args, [model], apply=True)
    schedule = get_loop_rsbf_schedule()
    monkeypatch.setattr(schedule, "cached_executor", lambda module: object())
    with pytest.raises(RuntimeError, match="before the first training step"):
        planning.plan_loop_rsbf(_args(loop_planner_hardware="h100"), [model], apply=True)
    assert model.config.loop_rsbf_item_form == "whole_round"
    with pytest.raises(ValueError, match="one model chunk"):
        planning.plan_loop_rsbf(_args(), [model, model], apply=True)


def test_report_changes_nothing_and_checks_the_current_option(_model_parallel, session, capsys):
    model = _model(loop_rsbf_item_form="block", loop_rsbf_block_stride=2)
    args = _args(
        loop_planner="report",
        loop_planner_hardware="h100",
        loop_rsbf_item_form="block",
        loop_rsbf_block_stride=2,
    )
    plan = planning.plan_loop_rsbf(args, [model], apply=False)
    assert plan.item_form == "whole_round"
    assert (model.config.loop_rsbf_item_form, model.config.loop_rsbf_block_stride) == ("block", 2)
    assert (args.loop_rsbf_item_form, args.loop_rsbf_block_stride) == ("block", 2)
    out = capsys.readouterr().out
    if is_rank0():  # the report-mode verdict is print_rank_0'd
        assert "the current option (item form block stride 2 waves 1 D42 off moves -) fits" in out
    result, message = planning.loop_planner_current_option(
        plan, model.config, optimizer_offload=False
    )
    assert result.option.stride_layers == 1 and result.option.form == "block"
    # Not in the menu: an odd (pair-splitting) block stride.
    odd = Namespace(loop_rsbf_item_form="block", loop_rsbf_block_stride=3, loop_rsbf_waves=1)
    result, message = planning.loop_planner_current_option(plan, odd, optimizer_offload=False)
    assert result is None and "is not in the planner's menu" in message
    # Refused: Adam to Grace on an h100 without a host tier.
    result, message = planning.loop_planner_current_option(
        plan, model.config, optimizer_offload=True
    )
    assert not result.fits or not result.applicable
    assert "does not fit: " in message and "no Grace tier" in message


def test_rerun_at_a_phase_start_drops_the_cached_executor(_model_parallel, session):
    """A second ``on_phase_start`` (a phase switch) re-applies the plan and drops only this
    model's cached executor; the schedule's status group and iteration stay."""
    model, other = _model(), _model()
    args = _args(loop_planner="apply", loop_planner_hardware="h100")
    planning.plan_loop_rsbf(args, [model], apply=True)
    schedule = get_loop_rsbf_schedule()
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    schedule.configure(status_group="status")
    schedule.set_iteration(7)
    try:
        schedule.executor_for(model, pg_collection)
        schedule.executor_for(other, pg_collection)
        # The new phase: the planner now owns the flags it set (no user-flag refusal).
        args.loop_planner_waves = [2]
        plan = planning.plan_loop_rsbf(args, [model], apply=True)
        assert plan.waves == 2 == model.config.loop_rsbf_waves == args.loop_rsbf_waves
        assert schedule.cached_executor(model) is None
        assert schedule.cached_executor(other) is not None
        assert schedule.iteration == 7 and schedule._status_group == "status"
        # Adam to Grace cannot change once the optimizer is built.
        args.optimizer_cpu_offload = True
        with pytest.raises(P.PlannerUnsupported, match="pin it with"):
            planning.plan_loop_rsbf(args, [model], apply=True)
    finally:
        schedule.reset()


def test_shape_settings_from_the_arguments(_model_parallel):
    model = _model()
    model.config.cross_entropy_loss_fusion = True
    model.config.cross_entropy_fusion_impl = "linear"
    shape = planning.loop_planner_shape(
        _args(use_distributed_optimizer=False, accumulate_allreduce_grads_in_fp32=False), model
    )
    assert not shape.distributed_optimizer and shape.expert_grad_bytes == 2
    assert shape.ce_bytes_per_token is None  # the cost model's fused CE term
    assert shape.core_layers == 4 and shape.grace_optimizer_bytes_per_param == 16
    shape = planning.loop_planner_shape(_args(loop_planner_act_bytes_per_token=1e4), _model())
    assert shape.distributed_optimizer and shape.expert_grad_bytes == 4
    assert shape.act_bytes_per_token_layer == 1e4
    assert shape.ce_bytes_per_token == 10.0 * model.vocab_size


# ----------------------------------------------------------------------------- resume, digest, hooks
def test_check_resume_offload():
    planned = _args(loop_planner="apply", optimizer_cpu_offload=True)
    with pytest.raises(RuntimeError, match="--loop-planner-optimizer-offload never"):
        planning.check_resume_offload(planned, Namespace(optimizer_cpu_offload=False))
    with pytest.raises(RuntimeError, match="--loop-planner-optimizer-offload always"):
        planning.check_resume_offload(
            _args(loop_planner="apply"), Namespace(optimizer_cpu_offload=True)
        )
    planning.check_resume_offload(planned, Namespace(optimizer_cpu_offload=True))
    # Pinned by the user, not loading the optimizer state, no record, planner not applying.
    for args, saved in (
        (
            _args(
                loop_planner="apply",
                loop_planner_optimizer_offload="always",
                optimizer_cpu_offload=True,
            ),
            False,
        ),
        (_args(loop_planner="apply", loop_planner_optimizer_offload="never"), True),
        (_args(loop_planner="apply", no_load_optim=True), True),
        (_args(loop_planner="apply", finetune=True), True),
        (_args(loop_planner="report"), True),
        (_args(), True),
    ):
        planning.check_resume_offload(args, Namespace(optimizer_cpu_offload=saved))
    planning.check_resume_offload(planned, Namespace())
    planning.check_resume_offload(planned, None)


def _digest_worker(rank, world, same):
    from megatron.training.loop_rsbf_planning import _check_plan_digest

    digest = "ab" * 32 if same or rank == 0 else "ab" * 31 + "ac"
    try:
        _check_plan_digest(digest)
    except RuntimeError as error:
        return str(error)
    return None


@pytest.mark.skipif(not spawn_tests_enabled(), reason=SPAWN_SKIP_REASON)
def test_digest_agreement_over_the_world():
    assert run_spawned(_digest_worker, 2, True, timeout=120.0) == [None, None]
    results = run_spawned(_digest_worker, 2, False, timeout=120.0)
    assert all(r is not None and "plan digests differ" in r for r in results)


def test_digest_check_without_a_world_is_a_no_op():
    planning._check_plan_digest("00" * 32)


def _calls_in(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return [
        (node.lineno, node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
    ]


def test_setup_runs_the_planner_after_the_model_and_before_the_optimizer_config():
    from megatron.training import training

    calls = _calls_in(training.setup_model_and_optimizer)
    line = {name: lineno for lineno, name in sorted(calls, reverse=True)}
    assert line["_build_model_wrapper"] < line["plan_loop_rsbf"]
    assert line["plan_loop_rsbf"] < line["get_megatron_optimizer_config"]
    assert line["plan_loop_rsbf"] < line["load_checkpoint"]


def test_load_checkpoint_checks_the_resume_offload():
    from megatron.training import checkpointing

    calls = [name for _, name in sorted(_calls_in(checkpointing.load_checkpoint))]
    assert calls.count("check_resume_offload") == 2
