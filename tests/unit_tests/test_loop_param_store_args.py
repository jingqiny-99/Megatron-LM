# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The training wiring of the loop parameter store (``megatron/training/loop_param_store_setup.py``,
``--loop-param-store-*``), on CPU:

* the CLI flags and every argument rule (``loop_param_store_problems`` /
  ``_validate_loop_param_store``): the RSBF schedule on a hybrid Huginn loop, z dividing the
  domain, no module GTP / TP / PP / CP, the per-token loss, fp32 gradient reduction, no FSDP /
  delayed wgrad / FP8 primary weights / shared-expert overlap / average-in-collective, Adam or
  SGD, ``--data-parallel-random-init`` / ``--init-model-with-meta-device``, ``--save`` /
  ``--load`` / ``--pretrained-checkpoint`` refused (M3c), the store modes against
  ``--loop-planner apply``; the store flags need z > 1;
* the grid size, the store config from the debug switches, the model's collection with the
  gtp axes nulled, the install guard against a DDP-wrapped model, the checkpoint backstop, the
  evaluation refresh;
* the planner with a store on a CPU-built dense loop model (trivial ``z = 1`` store): the
  Layer-window capability, the move steps (``E`` / ``EM`` without Adam to Grace), the shard
  axis, the store modes <-> planner option map, ``auto`` modes (core Layer window exactly when
  the Resident option does not fit), apply mode setting the store's modes, report mode;
* the hooks in ``get_model`` / ``setup_model_and_optimizer`` / ``evaluate`` /
  ``save_checkpoint`` / ``load_checkpoint`` / ``initialize`` / the status group (source order).

The z > 1 training path (grid, DDP + DistOpt, evaluation, status group) is the GPU test
``tests/unit_tests/transformer/loop/test_loop_param_store_training.py``.
"""

import ast
import inspect
import re
import sys
import textwrap
from argparse import Namespace

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection, resolve_gtp_remat_group
from megatron.core.transformer.loop import planner as P
from megatron.core.transformer.loop.param_store import (
    CHECKPOINT_M3C,
    ResidencyMode,
    StoreModes,
    find_param_store,
)
from megatron.core.transformer.loop.rsbf_executor import get_loop_rsbf_schedule
from megatron.training import loop_param_store_setup as setup
from megatron.training import loop_rsbf_planning as planning
from megatron.training.arguments import _validate_loop_param_store, parse_args
from tests.unit_tests.transformer.loop import param_store_utils as psu
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    build_local_loop_model,
    model_parallel_once,
)

DENSE = "*-*-*-*-*-*-"  # prelude '*-', core '*-*-*-*-' (L_core = 4 layers), coda '*-'
LW, RSG, R = (
    ResidencyMode.LAYER_WINDOW,
    ResidencyMode.RESIDENT_SHARDED_GRADS,
    ResidencyMode.RESIDENT,
)
STORE_DEFAULTS = dict(
    loop_param_store_shard_size=1,
    loop_param_store_modes="auto",
    loop_param_store_min_numel=1 << 20,
    loop_param_store_binding="data",
    loop_param_store_separate_rs_group=False,
    loop_param_store_debug="off",
    loop_param_store_gather_engine="nccl",
    loop_param_store_reduce_engine="nccl",
)
RSBF = dict(
    loop_schedule="rsbf",
    calculate_per_token_loss=True,
    loop_iterations=4,
    loop_exit_criterion="sampled",
    loop_depth_probs=[0.25, 0.25, 0.25, 0.25],
)


def _args(**overrides):
    """An 8-GPU RSBF run of the dense loop with a store of z = 8 (DP = 8 EUs), the planner off."""
    values = dict(STORE_DEFAULTS)
    values.update(
        loop_param_store_shard_size=8,
        seq_length=4096,
        world_size=8,
        data_parallel_size=8,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        gtp_weight_remat_size=1,
        expert_gtp_weight_remat_size=1,
        micro_batch_size=1,
        global_batch_size=64,
        loop_architecture="huginn",
        loop_prelude_sublayers=2,
        loop_coda_sublayers=2,
        loop_rsbf_item_form="whole_round",
        loop_rsbf_block_stride=None,
        loop_rsbf_waves=1,
        loop_rsbf_coda_recompute_mla_up_proj=False,
        accumulate_allreduce_grads_in_fp32=True,
        use_distributed_optimizer=True,
        optimizer="adam",
        optimizer_cpu_offload=False,
        optimizer_offload_fraction=1.0,
        use_precision_aware_optimizer=False,
        loop_planner="off",
        loop_planner_hardware="h100",
        loop_planner_optimizer_offload="auto",
        loop_planner_depths="config",
        loop_planner_forms="d28",
        loop_planner_strides=[1, 2, 3, 4, 6, 9, 18],
        loop_planner_waves=[1, 2, 4, 8],
        loop_planner_small_core_layers=4,
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
            setup.reset_loop_param_store_groups()


@pytest.fixture
def session():
    planning.reset_loop_planner_session()
    try:
        yield
    finally:
        planning.reset_loop_planner_session()


def _model(**overrides):
    fields = dict(RSBF)
    fields.update(overrides)
    return build_local_loop_model(DENSE, prelude=2, coda=2, **fields)


def _store(model):
    return psu.install_store(model, psu.trivial_store_groups(), 1, allow_trivial_shard=True)


# ----------------------------------------------------------------------------- CLI and rules
def test_cli_defaults_and_values(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test"])
    args = parse_args()
    assert {name: getattr(args, name) for name in STORE_DEFAULTS} == STORE_DEFAULTS
    monkeypatch.setattr(
        sys,
        "argv",
        "test --loop-param-store-shard-size 8 --loop-param-store-modes core=lw,edges=lw "
        "--loop-param-store-min-numel 65536 --loop-param-store-binding distributed_weight "
        "--loop-param-store-separate-rs-group --loop-param-store-debug check,timing "
        "--loop-param-store-gather-engine copy --loop-param-store-reduce-engine copy".split(),
    )
    args = parse_args()
    assert {name: getattr(args, name) for name in STORE_DEFAULTS} == dict(
        loop_param_store_shard_size=8,
        loop_param_store_modes="core=lw,edges=lw",
        loop_param_store_min_numel=65536,
        loop_param_store_binding="distributed_weight",
        loop_param_store_separate_rs_group=True,
        loop_param_store_debug="check,timing",
        loop_param_store_gather_engine="copy",
        loop_param_store_reduce_engine="copy",
    )


def test_the_valid_run_and_the_default_have_no_problems():
    assert setup.loop_param_store_problems(Namespace()) == []
    assert setup.loop_param_store_problems(_args(loop_param_store_shard_size=1)) == []
    assert setup.loop_param_store_problems(_args()) == []
    for overrides in (
        dict(loop_param_store_modes="core=lw,edges=lw,emb_head=rsg,core_regather=on"),
        dict(loop_param_store_shard_size=2, loop_param_store_debug="check,poison,verify,timing"),
        dict(optimizer="sgd", loop_planner="apply"),
        dict(loop_param_store_gather_engine="copy", loop_param_store_reduce_engine="copy"),
        dict(fp8="hybrid", disable_parameter_transpose_cache=True),
    ):
        assert setup.loop_param_store_problems(_args(**overrides)) == [], overrides
    _validate_loop_param_store(_args())
    _validate_loop_param_store(Namespace())


@pytest.mark.parametrize(
    "overrides,match",
    [
        (dict(loop_param_store_shard_size=0), "must be >= 1"),
        (dict(loop_param_store_shard_size=1, loop_param_store_modes="core=lw"), "need --loop"),
        (dict(loop_param_store_shard_size=1, loop_param_store_debug="check"), "need --loop"),
        (dict(loop_param_store_shard_size=1, loop_param_store_separate_rs_group=True), "need"),
        (dict(loop_param_store_debug="check,trace"), r"unknown \['trace'\]"),
        (dict(loop_param_store_min_numel=-1), "min-numel must be >= 0"),
        (dict(loop_param_store_binding="te"), "--loop-param-store-binding 'te'"),
        (dict(loop_param_store_gather_engine="nvls"), "--loop-param-store-gather-engine 'nvls'"),
        (dict(loop_param_store_shard_size=1, loop_param_store_gather_engine="copy"), "need"),
        (dict(loop_param_store_reduce_engine="nvls"), "--loop-param-store-reduce-engine 'nvls'"),
        (dict(loop_param_store_shard_size=1, loop_param_store_reduce_engine="copy"), "need"),
        (dict(loop_schedule="standard"), "trains only through --loop-schedule rsbf"),
        (dict(loop_architecture="mor"), "needs a hybrid Huginn loop"),
        (dict(loop_prelude_sublayers=None), "needs a hybrid Huginn loop"),
        (dict(world_size=12, data_parallel_size=12), "does not divide the world size 12"),
        (dict(loop_param_store_shard_size=16), "does not divide the 8 data-parallel ranks"),
        (dict(gtp_weight_remat_size=2), "module GTP"),
        (dict(tensor_model_parallel_size=2), "--tensor-model-parallel-size > 1"),
        (dict(pipeline_model_parallel_size=2), "--pipeline-model-parallel-size > 1"),
        (dict(context_parallel_size=2), "--context-parallel-size > 1"),
        (dict(calculate_per_token_loss=False), "needs --calculate-per-token-loss"),
        (dict(accumulate_allreduce_grads_in_fp32=False), "accumulate-allreduce-grads-in-fp32"),
        (dict(ddp_average_in_collective=True), "--ddp-average-in-collective"),
        (dict(use_megatron_fsdp=True), "FSDP"),
        (dict(use_torch_fsdp2=True), "FSDP"),
        (dict(num_distributed_optimizer_instances=2), "optimizer-instances"),
        (dict(moe_shared_expert_overlap=True), "--moe-shared-expert-overlap"),
        (dict(delay_wgrad_compute=True), "--delay-wgrad-compute"),
        (dict(fp8_param_gather=True), "FP8 primary weights"),
        (dict(fp8="hybrid"), "--disable-parameter-transpose-cache"),
        (dict(fp4="nvfp4"), "--disable-parameter-transpose-cache"),
        (dict(optimizer="muon"), "--optimizer muon"),
        (dict(use_layer_wise_distributed_optimizer=True), "layer-wise"),
        (dict(data_parallel_random_init=True), "--data-parallel-random-init"),
        (dict(init_model_with_meta_device=True), "--init-model-with-meta-device"),
        (dict(save="/ckpt"), r"--save: checkpointing with the loop parameter store \(z = 8\)"),
        (dict(load="/ckpt"), "--load: checkpointing .* is M3c"),
        (dict(pretrained_checkpoint="/ckpt"), "--pretrained-checkpoint: .*shard-size 1"),
        (dict(loop_param_store_modes="core=xx"), "--loop-param-store-modes: store modes"),
        (dict(loop_param_store_modes="emb_head=lw"), "emb_head=layer_window"),
        (dict(loop_planner="apply", loop_param_store_modes="core=lw"), "planner apply sets"),
    ],
)
def test_argument_rules(overrides, match):
    problems = setup.loop_param_store_problems(_args(**overrides))
    assert problems and any(re.search(match, p) for p in problems), problems
    with pytest.raises(ValueError, match="loop parameter store: "):
        _validate_loop_param_store(_args(**overrides))


def test_distributed_weight_binding_needs_the_te_protocol():
    from megatron.core.transformer.loop.param_store_binding import (
        distributed_weight_unavailable_reason,
    )

    problems = setup.loop_param_store_problems(_args(loop_param_store_binding="distributed_weight"))
    reason = distributed_weight_unavailable_reason()
    if reason is None:
        assert problems == []
    else:
        assert problems == [f"--loop-param-store-binding distributed_weight: {reason}"]


def test_grid_size_and_store_config():
    assert setup.loop_param_store_grid_size(Namespace()) == 1
    assert setup.loop_param_store_grid_size(_args()) == 8
    assert setup.loop_param_store_grid_size(_args(loop_param_store_shard_size=1)) == 1
    assert setup.loop_param_store_grid_size(Namespace(gtp_weight_remat_size=4)) == 4
    config = setup.loop_param_store_config(
        _args(loop_param_store_debug="check, timing", loop_param_store_min_numel=7)
    )
    assert (config.shard_size, config.min_numel, config.binding) == (8, 7, "data")
    assert config.check_visits and config.timing
    assert not (config.poison_released or config.verify_gathers or config.separate_rs_group)
    assert config.gather_engine == "nccl"
    assert (
        setup.loop_param_store_config(_args(loop_param_store_gather_engine="copy")).gather_engine
        == "copy"
    )
    assert config.reduce_engine == "nccl"
    assert (
        setup.loop_param_store_config(_args(loop_param_store_reduce_engine="copy")).reduce_engine
        == "copy"
    )
    assert setup.loop_param_store_debug(Namespace(loop_param_store_debug="off")) == ()


def test_args_data_parallel_size_stays_the_full_axis(monkeypatch):
    """The store's grid is not folded into ``total_model_size``: ``gtp_weight_remat_size``
    stays 1 and ``data_parallel_size`` is W = 8, so every ``data_parallel_size x
    gtp_weight_remat_size`` product of the trainer is W; the grid size goes to
    ``initialize_model_parallel`` from ``loop_param_store_grid_size``. (The real
    ``validate_args`` on a non-loop GPT config: it computes the sizes, then the store's rule
    refuses the run, which is all this checks.)"""
    from megatron.training import initialize
    from megatron.training.arguments import validate_args

    monkeypatch.setattr(sys, "argv", ["test", "--loop-param-store-shard-size", "8"])
    args = parse_args()
    args.world_size = 8
    for name, value in dict(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        max_position_embeddings=1024,
        seq_length=1024,
        micro_batch_size=1,
        global_batch_size=8,
        train_iters=1,
        lr=1e-4,
        tokenizer_type="NullTokenizer",
        vocab_size=1024,
    ).items():
        setattr(args, name, value)
    with pytest.raises(ValueError, match="trains only through --loop-schedule rsbf"):
        validate_args(args)
    assert (args.data_parallel_size, args.gtp_weight_remat_size) == (8, 1)
    calls = [name for _, name in _calls_in(initialize._initialize_distributed)]
    assert "loop_param_store_grid_size" in calls


# ----------------------------------------------------------------------------- model wiring
def test_model_collection_nulls_only_the_gtp_axes(_model_parallel):
    pgc = ProcessGroupCollection.use_mpu_process_groups()
    model_pgc = setup.loop_model_pg_collection(pgc)
    assert model_pgc is not pgc
    assert model_pgc.gtp_remat is None and model_pgc.expt_gtp_remat is None
    assert resolve_gtp_remat_group(model_pgc, is_expert=False) is None
    assert resolve_gtp_remat_group(model_pgc, is_expert=True) is None
    assert resolve_gtp_remat_group(pgc, is_expert=False) is pgc.gtp_remat
    for name in ("tp", "pp", "dp", "dp_cp", "ep", "dp_gtp_remat", "dp_cp_gtp_remat"):
        assert getattr(model_pgc, name) is getattr(pgc, name)


def test_install_guards(_model_parallel):
    pgc = ProcessGroupCollection.use_mpu_process_groups()
    model = _model()
    assert (
        setup.install_loop_param_store([model], _args(loop_param_store_shard_size=1), pgc) is None
    )
    assert find_param_store(model) is None

    class FakeDDP(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.ddp_config = object()

    with pytest.raises(ValueError, match="before the DDP wrap"):
        setup.install_loop_param_store([FakeDDP(model)], _args(), pgc)
    groups = setup.loop_param_store_groups(_args(), pgc)
    assert groups.shard is pgc.gtp_remat and groups.replicate is pgc.dp
    assert groups.domain is pgc.dp_gtp_remat and groups.shard_rs is None


def test_checkpoint_backstop_and_refresh(_model_parallel):
    plain = _model()
    setup.check_loop_param_store_checkpoint([plain], "save_checkpoint")  # no store: silent
    assert not setup.refresh_loop_param_store([plain])
    model = _model()
    store = _store(model)
    with pytest.raises(NotImplementedError, match="save_checkpoint: " + CHECKPOINT_M3C[:40]):
        setup.check_loop_param_store_checkpoint([model], "save_checkpoint")
    with pytest.raises(NotImplementedError, match="M3c"):
        store.sharded_state_dict()
    assert not setup.refresh_loop_param_store([model])  # modes not configured: nothing to do
    store.configure_modes(StoreModes.parse("core=lw,edges=lw,emb_head=rsg"))
    store._fresh = False  # pylint: disable=protected-access
    assert setup.refresh_loop_param_store([model])
    assert store._fresh  # pylint: disable=protected-access
    assert all(t.bound is not None for u in store.units if u.group == "emb_head" for t in u.tensors)


def test_status_group_follows_data_parallelism(_model_parallel):
    import torch.distributed as dist  # pylint: disable=import-outside-toplevel

    assert setup.loop_rsbf_status_gloo_group(_args(use_gloo_process_groups=False)) is None
    group = setup.loop_rsbf_status_gloo_group(_args())
    world = dist.get_world_size()
    if world == 1:
        assert group is None  # no data parallelism
        return
    # Under torchrun (the CI bucket): one Gloo group over the full data axis, created once.
    assert dist.get_process_group_ranks(group) == list(range(world))
    assert dist.get_backend(group) == "gloo"
    assert setup.loop_rsbf_status_gloo_group(_args()) is group


# ----------------------------------------------------------------------------- the planner
def test_capabilities_move_steps_and_phase(_model_parallel):
    plain = _model()
    model = _model()
    _store(model)
    hw = planning.loop_planner_hardware(_args())
    assert planning.loop_planner_capabilities(_args(), model, hardware=hw).layer_window
    assert not planning.loop_planner_capabilities(_args(), plain, hardware=hw).layer_window
    assert planning.loop_planner_capabilities(_args(), None, hardware=hw).layer_window
    caps = planning.loop_planner_capabilities(_args(loop_param_store_shard_size=1), None)
    assert not caps.layer_window
    with_store = P.PlannerCapabilities(layer_window=True)
    without = P.PlannerCapabilities(layer_window=False)
    # No store: today's cumulative list (always: the A steps).
    assert planning.loop_planner_move_steps(_args(), hw, without) == P.MOVE_STEPS
    always = _args(loop_planner_optimizer_offload="always")
    assert planning.loop_planner_move_steps(always, hw, without) == ("A", "AE", "AEM")
    # With a store on H100 (no Grace tier): E and EM without Adam to Grace.
    assert planning.loop_planner_move_steps(_args(), hw, with_store) == ("-", "E", "EM")
    assert planning.loop_planner_move_steps(always, hw, with_store) == ("A", "AE", "AEM")
    grace = _args(
        loop_planner_hardware="gb200", use_precision_aware_optimizer=True, optimizer="adam"
    )
    gb200 = planning.loop_planner_hardware(grace)
    assert planning.loop_planner_move_steps(grace, gb200, with_store) == P.STORE_MOVE_STEPS
    assert planning.loop_planner_phase(_args()).shard_axis == 8
    assert planning.loop_planner_phase(_args(loop_param_store_shard_size=1)).shard_axis is None


def test_store_modes_map_back_onto_planner_options():
    for layout in ("resident", "layer_window"):
        for moves in P.STORE_MOVE_STEPS:
            for regather in (False, True):
                if layout == "resident" and (regather or moves not in ("-", "A")):
                    continue
                option = P.PlannerOption("block", 2, regather, False, 1, moves, layout)
                mapped = planning.loop_planner_store_option(P.option_store_modes(option))
                assert mapped == (layout, moves.replace("A", "").replace("-", ""), regather)
    for text in ("edges=rsg", "core=rsg", "prelude=lw", "emb_head=rsg", "edge_regather=on"):
        assert planning.loop_planner_store_option(StoreModes.parse(text)) is None


def test_auto_modes_pick_the_core_layer_window_exactly_when_resident_does_not_fit(_model_parallel):
    model = _model()
    store = _store(model)
    args = _args(loop_planner_hbm_cap_gb=None)
    res, lw, hardware = planning.loop_planner_auto_core_options(args, [model])
    assert res is not None and lw is not None and hardware.name == "h100"
    assert lw.hbm_bytes < res.hbm_bytes  # the core's compute copy and accumulator
    mode, reason = planning.loop_planner_auto_core_mode(args, [model])
    assert mode is R and "core Resident fits" in reason
    tight = _args(loop_planner_hbm_cap_gb=(res.hbm_bytes + lw.hbm_bytes) / 2 / 1e9)
    mode, reason = planning.loop_planner_auto_core_mode(tight, [model])
    assert mode is LW and "core Resident does not fit" in reason and "Layer window fits" in reason
    modes, how = setup.resolve_loop_param_store_modes(tight, [model])
    assert modes == StoreModes(core=LW) and how.startswith("auto: ")
    modes, how = setup.resolve_loop_param_store_modes(
        _args(loop_param_store_modes="core=lw,edges=lw"), [model]
    )
    assert modes == StoreModes(core=LW, prelude=LW, coda=LW, mtp=LW) and how == "explicit"
    # A run the planner cannot model: Resident, with the reason.
    broken = _args(loop_planner_depths="histogram")
    mode, reason = planning.loop_planner_auto_core_mode(broken, [model])
    assert mode is R and "cannot model this run" in reason
    # configure_loop_param_store sets them on the store (the config's core plan).
    assert setup.configure_loop_param_store(tight, [model]) == StoreModes(core=LW)
    assert store.modes == StoreModes(core=LW) and store.core_plan.form == "whole_round"
    assert setup.configure_loop_param_store(_args(loop_planner="apply"), [model]) is None


def test_apply_sets_the_store_modes(_model_parallel, session):
    model = _model()
    store = _store(model)
    args = _args(loop_planner="apply", loop_planner_hbm_cap_gb=None)
    plan = planning.plan_loop_rsbf(args, [model], apply=True)
    assert plan.capabilities.layer_window and plan.unavailable == ()
    assert plan.applied.option.moves in ("-", "E", "EM")  # no Grace tier on h100
    assert store.modes == plan.store_modes()
    assert store.core_plan.form == plan.item_form
    for name, value in plan.config_overrides().items():
        assert getattr(model.config, name) == value


def test_report_names_the_store_modes(_model_parallel, session, capsys):
    from megatron.training.utils import is_rank0

    model = _model()
    store = _store(model)
    store.configure_modes(StoreModes.parse("core=lw,edges=lw"))
    args = _args(loop_planner="report")
    plan = planning.plan_loop_rsbf(args, [model], apply=False)
    assert plan is not None and store.modes == StoreModes.parse("core=lw,edges=lw")
    current, message = planning.loop_planner_current_option(
        plan, model.config, optimizer_offload=False, store_modes=store.modes
    )
    assert current is not None and current.option.moves == "E"
    assert current.option.layout == "layer_window" and "layout layer_window" in message
    out = capsys.readouterr().out
    if is_rank0():
        assert "loop planner (report): store modes core=lw,prelude=lw" in out
    store.configure_modes(StoreModes.parse("edges=rsg"))
    current, message = planning.loop_planner_current_option(
        plan, model.config, optimizer_offload=False, store_modes=store.modes
    )
    assert current is None and "are not a planner option" in message


# ----------------------------------------------------------------------------- hooks (source)
def _calls_in(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return [
        (node.lineno, node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
    ]


def _first_lines(function):
    out = {}
    for lineno, name in sorted(_calls_in(function)):
        out.setdefault(name, lineno)
    return out


def test_training_hooks_in_order():
    from megatron.training import checkpointing, training

    line = _first_lines(training.get_model)
    assert line["loop_model_pg_collection"] < line["model_provider_func"]
    assert line["install_loop_param_store"] < line["wrap_model_chunks_with_ddp"]
    assert line["Float16Module"] < line["install_loop_param_store"]
    line = _first_lines(training._build_loop_param_store_models)
    assert line["build_distributed_models"] < line["install_loop_param_store"] < line["_ddp_wrap"]
    line = _first_lines(training.setup_model_and_optimizer)
    assert line["_build_model_wrapper"] < line["configure_loop_param_store"]
    assert line["configure_loop_param_store"] < line["plan_loop_rsbf"]
    assert line["plan_loop_rsbf"] < line["get_megatron_optimizer"]
    assert "_build_loop_param_store_models" in line
    line = _first_lines(training.evaluate)
    assert line["refresh_loop_param_store"] < line["forward_backward_func"]
    assert "loop_rsbf_status_gloo_group" in _first_lines(training.loop_rsbf_status_group)
    for function, what in (
        (checkpointing.save_checkpoint, "save_checkpoint"),
        (checkpointing.load_checkpoint, "load_checkpoint"),
    ):
        calls = [name for _, name in sorted(_calls_in(function))]
        assert "check_loop_param_store_checkpoint" in calls, what


def test_metrics_average_over_every_distinct_data_rank(_model_parallel):
    """Under the store's grid ``dp`` is the replicate group; the indexer loss is averaged over
    ``dp_gtp_remat`` (every rank with distinct microbatches), and over ``dp`` without the grid."""
    from types import SimpleNamespace

    from megatron.training import training

    class Group:
        def __init__(self, size):
            self._size = size

        def size(self):
            return self._size

    replicate, full, same = Group(1), Group(8), Group(8)
    pick = training.metric_data_parallel_group
    assert pick(SimpleNamespace(dp=replicate, dp_gtp_remat=full)) is full
    assert pick(SimpleNamespace(dp=full, dp_gtp_remat=same)) is full
    assert pick(SimpleNamespace(dp=full)) is full
    calls = _first_lines(training.training_log)
    assert calls["metric_data_parallel_group"] < calls["track_indexer_metrics"]
