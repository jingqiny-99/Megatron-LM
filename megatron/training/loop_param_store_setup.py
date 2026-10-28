# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Training side of the loop parameter store (``megatron/core/transformer/loop/param_store.py``).

``--loop-param-store-shard-size z`` (z > 1) runs a hybrid Huginn loop on the RSBF schedule with
the dense parameters held as canonical shards on a shard axis of ``z`` ranks: dev's GTP-remat
grid (``initialize_model_parallel(gtp_remat_size=z)``, ``initialize.py``), used **as a grid
only**. The model is built with the gtp axes of its process-group collection nulled
(``loop_model_pg_collection``: no TE wrapper attaches ``GTPShardedParam``, which needs
TE >= 2.19), the store is installed after the mixed-precision wrap and before the DDP wrap
(``install_loop_param_store``; DDP then buckets the shards over the replicate group), and its
residency modes are set before the first step (``configure_loop_param_store``: explicit
``--loop-param-store-modes``, ``auto`` or the planner's under ``--loop-planner apply``).
``args.data_parallel_size`` stays the full data-distribution degree W (z is not folded into
``total_model_size``: ``args.gtp_weight_remat_size`` stays 1), so every batch-size product of
the trainer keeps computing W.

Also here: the Gloo status group of the RSBF executor (the grid disables dev's Gloo groups),
the evaluation refresh of the Resident compute copies, the M3c checkpoint backstop and the
argument rules (``loop_param_store_problems``, called by ``arguments._validate_loop_param_store``).
"""

import copy
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple

import torch

from megatron.core.transformer.loop.param_store import (
    CHECKPOINT_M3C,
    LoopParamStore,
    LoopParamStoreConfig,
    ResidencyMode,
    StoreGroups,
    StoreModes,
    find_param_store,
    unwrap_hybrid_model,
)
from megatron.core.transformer.loop.param_store_comm import COPY_ENGINE_EXPERIMENTAL_NOTE

__all__ = [
    "LOOP_PARAM_STORE_DEBUG",
    "check_loop_param_store_checkpoint",
    "configure_loop_param_store",
    "install_loop_param_store",
    "loop_model_pg_collection",
    "loop_param_store_config",
    "loop_param_store_debug",
    "loop_param_store_grid_size",
    "loop_param_store_groups",
    "loop_param_store_problems",
    "loop_param_store_shard_size",
    "loop_rsbf_status_gloo_group",
    "refresh_loop_param_store",
    "reset_loop_param_store_groups",
    "resolve_loop_param_store_modes",
]

# --loop-param-store-debug switches -> LoopParamStoreConfig fields.
LOOP_PARAM_STORE_DEBUG = {
    "check": "check_visits",
    "poison": "poison_released",
    "verify": "verify_gathers",
    "timing": "timing",
}

# Process-wide groups this module creates (every rank creates every group once, in one order).
_GROUPS = SimpleNamespace(status_gloo={}, shard_rs=None)


def reset_loop_param_store_groups() -> None:
    """Forget the cached groups (tests; a new world in the same process)."""
    _GROUPS.status_gloo = {}
    _GROUPS.shard_rs = None


def loop_param_store_shard_size(args) -> int:
    """``--loop-param-store-shard-size`` (1: no store)."""
    value = getattr(args, "loop_param_store_shard_size", 1)
    return 1 if value is None else int(value)


def loop_param_store_grid_size(args) -> int:
    """The GTP-remat grid size ``initialize_model_parallel`` builds: the store's ``z``, or
    module GTP's ``gtp_weight_remat_size`` (the two are exclusive, ``loop_param_store_problems``).
    """
    return max(
        int(getattr(args, "gtp_weight_remat_size", 1) or 1), loop_param_store_shard_size(args)
    )


def loop_param_store_debug(args) -> Tuple[str, ...]:
    """``--loop-param-store-debug``: a comma list of ``check`` / ``poison`` / ``verify`` /
    ``timing`` (``off`` or empty: none)."""
    text = getattr(args, "loop_param_store_debug", None) or "off"
    if isinstance(text, (list, tuple)):
        text = ",".join(text)
    items = [item.strip().lower() for item in str(text).split(",") if item.strip()]
    return tuple(item for item in items if item != "off")


def loop_param_store_problems(args) -> List[str]:
    """Every reason the run's arguments cannot use the loop parameter store (empty when they
    can, or when no store is requested and no store flag is set). Reads every argument with
    ``getattr`` (stub namespaces in tests)."""
    z = loop_param_store_shard_size(args)
    modes = getattr(args, "loop_param_store_modes", "auto") or "auto"
    debug_text = getattr(args, "loop_param_store_debug", None) or "off"
    problems = []
    unknown = [d for d in loop_param_store_debug(args) if d not in LOOP_PARAM_STORE_DEBUG]
    if unknown:
        problems.append(
            f"--loop-param-store-debug {debug_text!r}: unknown {unknown} (a comma list of "
            f"{', '.join(LOOP_PARAM_STORE_DEBUG)}, or off)"
        )
    if getattr(args, "loop_param_store_min_numel", 1 << 20) < 0:
        problems.append("--loop-param-store-min-numel must be >= 0")
    binding = getattr(args, "loop_param_store_binding", "data")
    if binding not in ("data", "distributed_weight"):
        problems.append(f"--loop-param-store-binding {binding!r} (data or distributed_weight)")
    engine = getattr(args, "loop_param_store_gather_engine", "nccl")
    if engine not in ("nccl", "copy"):
        problems.append(f"--loop-param-store-gather-engine {engine!r} (nccl or copy)")
    reduce_engine = getattr(args, "loop_param_store_reduce_engine", "nccl")
    if reduce_engine not in ("nccl", "copy"):
        problems.append(f"--loop-param-store-reduce-engine {reduce_engine!r} (nccl or copy)")
    timeout_ms = getattr(args, "loop_param_store_copy_engine_timeout_ms", 0) or 0
    if timeout_ms < 0:
        problems.append("--loop-param-store-copy-engine-timeout-ms must be >= 0 (0 = wait)")
    elif timeout_ms > 0 and "copy" not in (engine, reduce_engine):
        problems.append(
            "--loop-param-store-copy-engine-timeout-ms needs --loop-param-store-gather-engine "
            "or --loop-param-store-reduce-engine copy (the NCCL engines have no barrier to time "
            "out)"
        )
    if z < 1:
        problems.append(f"--loop-param-store-shard-size must be >= 1, got {z}")
        return problems
    if z == 1:
        set_flags = [
            flag
            for flag, value in (
                ("--loop-param-store-modes", modes != "auto"),
                ("--loop-param-store-binding", binding != "data"),
                ("--loop-param-store-gather-engine", engine != "nccl"),
                ("--loop-param-store-reduce-engine", reduce_engine != "nccl"),
                ("--loop-param-store-copy-engine-timeout-ms", timeout_ms != 0),
                (
                    "--loop-param-store-separate-rs-group",
                    getattr(args, "loop_param_store_separate_rs_group", False),
                ),
                ("--loop-param-store-debug", bool(loop_param_store_debug(args))),
            )
            if value
        ]
        if set_flags:
            problems.append(f"{', '.join(set_flags)} need --loop-param-store-shard-size > 1")
        return problems

    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer.loop.guards import is_hybrid_loop_config

    if getattr(args, "loop_schedule", "standard") != "rsbf":
        problems.append(
            "the loop parameter store trains only through --loop-schedule rsbf (an eager step "
            "would leave the shard gradients unreduced)"
        )
    if not is_hybrid_loop_config(args) or getattr(args, "loop_architecture", "none") != "huginn":
        problems.append(
            "the loop parameter store needs a hybrid Huginn loop (--loop-architecture huginn "
            "with --loop-prelude-sublayers / --loop-coda-sublayers)"
        )
    world = getattr(args, "world_size", None)
    if world is not None and world % z:
        problems.append(f"--loop-param-store-shard-size {z} does not divide the world size {world}")
    dp = getattr(args, "data_parallel_size", None)
    if dp is not None and dp % z:
        problems.append(
            f"--loop-param-store-shard-size {z} does not divide the {dp} data-parallel ranks "
            "(the RSBF domain)"
        )
    if (
        getattr(args, "gtp_weight_remat_size", 1) > 1
        or getattr(args, "expert_gtp_weight_remat_size", 1) > 1
    ):
        problems.append(
            "module GTP (--tensor-parallel-num-weight-shards > --tensor-model-parallel-size) "
            "with the loop parameter store (the store reuses the GTP grid only)"
        )
    for name, flag in (
        ("tensor_model_parallel_size", "--tensor-model-parallel-size"),
        ("pipeline_model_parallel_size", "--pipeline-model-parallel-size"),
        ("context_parallel_size", "--context-parallel-size"),
    ):
        if getattr(args, name, 1) > 1:
            problems.append(f"{flag} > 1 with the loop parameter store")
    if not getattr(args, "calculate_per_token_loss", False):
        problems.append(
            "the loop parameter store needs --calculate-per-token-loss (finalize's shard-axis "
            "SUM of the replicated gradients is exact only under the per-token loss)"
        )
    if not getattr(args, "accumulate_allreduce_grads_in_fp32", False):
        problems.append(
            "the loop parameter store needs --accumulate-allreduce-grads-in-fp32 (the shard "
            "gradients are DDP's fp32 main_grad)"
        )
    if getattr(args, "ddp_average_in_collective", False):
        problems.append("--ddp-average-in-collective with the loop parameter store")
    if getattr(args, "use_megatron_fsdp", False) or getattr(args, "use_torch_fsdp2", False):
        problems.append("FSDP (--use-megatron-fsdp / --use-torch-fsdp2) with the store")
    if getattr(args, "num_distributed_optimizer_instances", 1) > 1:
        problems.append("--num-distributed-optimizer-instances > 1 with the store")
    if getattr(args, "moe_shared_expert_overlap", False):
        problems.append(
            "--moe-shared-expert-overlap with the store (its side stream would read windowed "
            "shared-expert weights outside the store's events)"
        )
    if getattr(args, "delay_wgrad_compute", False):
        problems.append(
            "--delay-wgrad-compute with the store (the wgrad would run after the "
            "segment's reduce-scatter)"
        )
    if getattr(args, "fp8_param_gather", False) or getattr(args, "fp8_param", False):
        problems.append("FP8 primary weights (--fp8-param-gather) with the store (it gathers bf16)")
    quantized = getattr(args, "fp8", None) is not None or getattr(args, "fp4", None) is not None
    if quantized and not getattr(args, "disable_parameter_transpose_cache", False):
        problems.append(
            "FP8 / FP4 with the store needs --disable-parameter-transpose-cache (a cached "
            "quantized weight per layer would defeat the window)"
        )
    optimizer = getattr(args, "optimizer", "adam")
    if optimizer not in ("adam", "sgd"):
        problems.append(
            f"--optimizer {optimizer} with the store (only adam / sgd: other optimizers read the "
            "unsharded shapes of GTP shards)"
        )
    if getattr(args, "use_layer_wise_distributed_optimizer", False):
        problems.append("the layer-wise distributed optimizer with the store")
    if getattr(args, "data_parallel_random_init", False):
        problems.append(
            "--data-parallel-random-init with the store (shard peers must initialize the same "
            "dense weights)"
        )
    if getattr(args, "init_model_with_meta_device", False):
        problems.append(
            "--init-model-with-meta-device with the store (install needs materialized weights "
            "before the DDP wrap)"
        )
    for name, flag in (
        ("save", "--save"),
        ("load", "--load"),
        ("pretrained_checkpoint", "--pretrained-checkpoint"),
    ):
        if getattr(args, name, None):
            problems.append(f"{flag}: {checkpoint_message(z)}")
    if modes != "auto":
        try:
            StoreModes.parse(modes)
        except ValueError as error:
            problems.append(f"--loop-param-store-modes: {error}")
        if getattr(args, "loop_planner", "off") == "apply":
            problems.append(
                "--loop-planner apply sets the store modes: leave --loop-param-store-modes at "
                "auto (or use --loop-planner report)"
            )
    if binding == "distributed_weight":
        # pylint: disable=import-outside-toplevel
        from megatron.core.transformer.loop.param_store_binding import (
            distributed_weight_unavailable_reason,
        )

        reason = distributed_weight_unavailable_reason()
        if reason is not None:
            problems.append(f"--loop-param-store-binding distributed_weight: {reason}")
    return problems


def checkpoint_message(z: int) -> str:
    """The M3c refusal text."""
    return (
        f"checkpointing with the loop parameter store (z = {z}) is M3c (not implemented); run "
        "with --loop-param-store-shard-size 1 to save or load"
    )


def loop_param_store_config(args) -> LoopParamStoreConfig:
    """The store settings of the run (``--loop-param-store-*``)."""
    debug = loop_param_store_debug(args)
    return LoopParamStoreConfig(
        shard_size=loop_param_store_shard_size(args),
        min_numel=int(getattr(args, "loop_param_store_min_numel", 1 << 20)),
        binding=getattr(args, "loop_param_store_binding", "data"),
        separate_rs_group=bool(getattr(args, "loop_param_store_separate_rs_group", False)),
        gather_engine=getattr(args, "loop_param_store_gather_engine", "nccl"),
        reduce_engine=getattr(args, "loop_param_store_reduce_engine", "nccl"),
        copy_engine_timeout_ms=int(
            getattr(args, "loop_param_store_copy_engine_timeout_ms", 0) or 0
        ),
        **{field: name in debug for name, field in LOOP_PARAM_STORE_DEBUG.items()},
    )


def loop_model_pg_collection(pg_collection: Any) -> Any:
    """A shallow copy of ``pg_collection`` with ``gtp_remat = expt_gtp_remat = None`` (set in
    ``vars()``, so ``resolve_gtp_remat_group`` returns None instead of falling back to the MPU
    globals): the model is built with the grid's groups but no module attaches
    ``GTPShardedParam``. Everything else (the replicate ``dp``, ``dp_gtp_remat``, ...) is
    shared with the original."""
    model_pgc = copy.copy(pg_collection)
    model_pgc.gtp_remat = None
    model_pgc.expt_gtp_remat = None
    return model_pgc


def _rank_lists(group: Any) -> List[Tuple[int, ...]]:
    """Every distinct rank list of ``group`` over the world (one ``all_gather_object``),
    sorted, so every rank can create every group in the same order."""
    mine = tuple(torch.distributed.get_process_group_ranks(group))
    gathered: List[Any] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, mine)
    return sorted(set(tuple(ranks) for ranks in gathered))


def _new_groups(rank_lists: Sequence[Tuple[int, ...]], **kwargs) -> Any:
    """``new_group`` over every list (every rank, the same order); returns this rank's."""
    rank = torch.distributed.get_rank()
    mine = None
    for ranks in rank_lists:
        group = torch.distributed.new_group(list(ranks), **kwargs)
        if rank in ranks:
            mine = group
    return mine


def loop_param_store_groups(args, pg_collection: Any) -> StoreGroups:
    """The store's groups from the grid's collection: the shard axis ``gtp_remat``, the RSBF
    domain (the full data-distribution group ``dp_gtp_remat``), the replicate ``dp`` (DDP /
    DistOpt), and with ``--loop-param-store-separate-rs-group`` a second communicator over each
    shard group (created once, every rank in the same order)."""
    shard = getattr(pg_collection, "gtp_remat", None)
    domain = getattr(pg_collection, "dp_gtp_remat", None)
    if domain is None:
        # pylint: disable=import-outside-toplevel
        from megatron.core import parallel_state

        domain = parallel_state.get_data_parallel_group(with_gtp_remat=True)
    shard_rs = None
    if getattr(args, "loop_param_store_separate_rs_group", False):
        if _GROUPS.shard_rs is None:
            _GROUPS.shard_rs = _new_groups(_rank_lists(shard))
        shard_rs = _GROUPS.shard_rs
    return StoreGroups(shard=shard, domain=domain, replicate=pg_collection.dp, shard_rs=shard_rs)


def install_loop_param_store(model: Any, args, pg_collection: Any) -> Optional[LoopParamStore]:
    """Install the store on ``model`` (the built, mixed-precision-wrapped chunk list, before the
    DDP wrap) when ``--loop-param-store-shard-size`` > 1; None otherwise. ``pg_collection``: the
    grid's full collection (the one DDP gets)."""
    if loop_param_store_shard_size(args) <= 1:
        return None
    if pg_collection is None:
        # pylint: disable=import-outside-toplevel
        from megatron.core.process_groups_config import ProcessGroupCollection

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    store = LoopParamStore.install(
        model, loop_param_store_config(args), loop_param_store_groups(args, pg_collection)
    )
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        stored = sum(t.numel for t in store.tensors())
        print(
            f"> loop parameter store: z = {store.shard_size}, {len(store.units)} units, "
            f"{stored} dense parameters stored as shards (binding {store.config.binding}"
            + (
                f"; local wgrad fusion off for {len(store.fusion_disabled)} modules"
                if store.fusion_disabled
                else ""
            )
            + ")",
            flush=True,
        )
    return store


def loop_rsbf_status_gloo_group(args) -> Optional[Any]:
    """The Gloo group of the RSBF status exchange under the store's grid (dev disables its Gloo
    groups under GTP-remat): one ``new_group(backend="gloo")`` per distinct full data-parallel
    rank list (every rank creates every one, in sorted order); None without data parallelism or
    with ``--no-use-gloo-process-groups``."""
    # pylint: disable=import-outside-toplevel
    from megatron.core import parallel_state

    if not getattr(args, "use_gloo_process_groups", True):
        return None
    group = parallel_state.get_data_parallel_group(with_gtp_remat=True)
    if torch.distributed.get_world_size(group) <= 1:
        return None
    key = tuple(torch.distributed.get_process_group_ranks(group))
    if key not in _GROUPS.status_gloo:
        lists = _rank_lists(group)
        rank = torch.distributed.get_rank()
        made = {}
        for ranks in lists:
            made[ranks] = torch.distributed.new_group(list(ranks), backend="gloo")
        _GROUPS.status_gloo.update({ranks: g for ranks, g in made.items() if rank in ranks})
    return _GROUPS.status_gloo[key]


def resolve_loop_param_store_modes(args, model: Any) -> Tuple[StoreModes, str]:
    """``(modes, how)``: the parsed ``--loop-param-store-modes``, or for ``auto`` the planner's
    memory-model choice of the core mode for the run's item form (core Layer window when the
    Resident option does not fit the HBM budget; edges and the embedding / head Resident), or
    core Resident with the reason when the planner cannot model the run."""
    text = getattr(args, "loop_param_store_modes", "auto") or "auto"
    if text != "auto":
        return StoreModes.parse(text), "explicit"
    # pylint: disable=import-outside-toplevel
    from megatron.training.loop_rsbf_planning import loop_planner_auto_core_mode

    core, reason = loop_planner_auto_core_mode(args, model)
    return StoreModes(core=core), f"auto: {reason}"


def configure_loop_param_store(args, model: Any) -> Optional[StoreModes]:
    """Set the store's modes before the first step (``--loop-param-store-modes``, or ``auto``),
    on the config's core item plan. A no-op without a store; under ``--loop-planner apply``
    the planner sets them (``apply_plan_to_store``)."""
    store = find_param_store(model)
    if store is None:
        return None
    rank0 = torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
    # Before the planner's early return: the warning holds whoever sets the modes.
    if rank0 and "copy" in (store.config.gather_engine, store.config.reduce_engine):
        print(f"> WARNING: {COPY_ENGINE_EXPERIMENTAL_NOTE}", flush=True)
    if getattr(args, "loop_planner", "off") == "apply":
        return None
    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer.loop.program import build_core_plan

    hybrid = unwrap_hybrid_model(model)
    modes, how = resolve_loop_param_store_modes(args, model)
    store.configure_modes(
        modes, core_plan=build_core_plan(hybrid.config, hybrid.decoder.loop_roles)
    )
    if rank0:
        print(f"> loop parameter store modes: {modes.describe()} ({how})", flush=True)
    return modes


def refresh_loop_param_store(model: Any) -> bool:
    """Before evaluation: re-gather the Resident / sharded-grads compute copies (stale after an
    optimizer step); Layer-window units are gathered by the eager visits. Returns whether a
    store was refreshed."""
    models = model if isinstance(model, (list, tuple)) else [model]
    refreshed = False
    for chunk in models:
        store = find_param_store(chunk)
        if store is None or store.modes is None:
            continue
        groups = {unit.group for unit in store.units}
        if any(store.modes.of(group) is not ResidencyMode.LAYER_WINDOW for group in groups):
            store.refresh()  # the same modes on every rank: every shard peer gathers
            refreshed = True
    return refreshed


def check_loop_param_store_checkpoint(model: Any, what: str) -> None:
    """The M3c backstop at the entry of ``save_checkpoint`` / ``load_checkpoint``: raises
    ``NotImplementedError`` when ``model`` holds a store (argument validation refuses
    ``--save`` / ``--load`` with z > 1 first)."""
    models = model if isinstance(model, (list, tuple)) else [model]
    for chunk in models:
        store = find_param_store(chunk)
        if store is not None:
            raise NotImplementedError(f"{what}: {CHECKPOINT_M3C} (z = {store.shard_size})")
