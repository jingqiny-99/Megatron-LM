# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Helpers of the loop parameter store's GPU checkpoint tests (not collected).

``save`` / ``load`` drive ``megatron.training.checkpointing.save_checkpoint`` /
``load_checkpoint`` the way training does on the store's GTP-remat grid (the replica ids over
the gtp-inclusive data-parallel group ``dp_cp_gtp_remat``, the RNG / optimizer replicas over the
replicate ``dp``), with mock arguments (``mock_checkpoint_args``, after
``tests/unit_tests/dist_checkpointing``'s ``init_checkpointing_mock_args``). The comparison
helpers read the logical weights (canonical shards all-gathered over the shard axis), the
distributed optimizer's fp32 mains and Adam moments per parameter name (this rank's ranges), the
whole run's optimizer state by logical name (``logical_optimizer_states``: comparable across z,
stored sets and store-free runs), and the shard pad rows (``poison_pad_rows`` before a load).
"""

import contextlib
from typing import Any, Dict, List
from unittest import mock

import torch
import torch.distributed as dist

from megatron.core.transformer.loop.param_store_checkpoint import RowShard
from megatron.core.utils import unwrap_model


@contextlib.contextmanager
def checkpoint_dir(path):
    """A checkpoint directory shared by every rank (removed afterwards)."""
    from tests.unit_tests.dist_checkpointing import TempNamedDir

    with TempNamedDir(path, sync=True) as ckpt_dir:
        yield ckpt_dir


@contextlib.contextmanager
def mock_checkpoint_args(
    ckpt_dir,
    *,
    fully_reshardable: bool = False,
    async_save: bool = False,
    fully_parallel: bool = False,
    dist_opt: bool = True,
    loop_config: Any = None,
    **overrides,
):
    """Mock training arguments for ``save_checkpoint`` / ``load_checkpoint`` (patched into
    ``megatron.training.checkpointing.get_args``). ``loop_config``: a model config whose
    ``loop_*`` fields are copied onto the arguments of the same name (the parser defaults
    otherwise), so the resume check (``check_loop_resume_state``, message 5) compares a loop
    run's real values; ``overrides`` (e.g. ``hybrid_layer_pattern``, ``loop_param_store_*``)
    are applied after. Also patched into ``megatron.training.async_utils.get_args``."""
    from megatron.training.arguments import parse_args
    from tests.unit_tests.dist_checkpointing.utils import init_checkpointing_mock_args

    args = parse_args(ignore_unknown_args=True)
    init_checkpointing_mock_args(args, ckpt_dir, fully_parallel=fully_parallel)
    args.use_distributed_optimizer = dist_opt
    args.save_tokenizer_assets = False
    args.dist_ckpt_optim_fully_reshardable = fully_reshardable
    args.async_save = async_save
    args.world_size = dist.get_world_size()
    if loop_config is not None:
        for name in [name for name in vars(args) if name.startswith("loop_")]:
            if hasattr(loop_config, name):
                setattr(args, name, getattr(loop_config, name))
    for name, value in overrides.items():
        setattr(args, name, value)
    # ``save`` finalizes async saves through ``async_utils.get_args`` (a separate binding).
    with (
        mock.patch("megatron.training.checkpointing.get_args", new=lambda: args),
        mock.patch("megatron.training.async_utils.get_args", new=lambda: args),
    ):
        with run_config():
            yield args


@contextlib.contextmanager
def run_config():
    """The training run config ``save_checkpoint`` / ``load_checkpoint`` read (``get_rng_state``:
    ``cfg.rng``; the logger settings): the default container of ``tests/unit_tests/conftest.py``'s
    ``run_config`` fixture (autouse only under ``tests/unit_tests/dist_checkpointing``), set for
    the context when none is set, the previous one restored afterwards."""
    from megatron.core.optimizer import OptimizerConfig
    from megatron.training import global_vars
    from megatron.training.config import (
        CheckpointConfig,
        LoggerConfig,
        PretrainConfigContainer,
        SchedulerConfig,
        TrainingConfig,
    )

    # pylint: disable=protected-access
    previous = global_vars._GLOBAL_RUN_CONFIG
    if previous is None:
        global_vars.set_run_config(
            PretrainConfigContainer(
                train=TrainingConfig(),
                model=None,
                optimizer=OptimizerConfig(),
                scheduler=SchedulerConfig(),
                logger=LoggerConfig(),
                checkpoint=CheckpointConfig(),
            )
        )
    try:
        yield global_vars.get_run_config()
    finally:
        global_vars._GLOBAL_RUN_CONFIG = previous


def save(ddp, optimizer, full, *, iteration: int) -> None:
    """``save_checkpoint`` of one model chunk with the grid's groups (async saves finalized)."""
    from megatron.training.async_utils import maybe_finalize_async_save
    from megatron.training.checkpointing import save_checkpoint
    from megatron.training.training import preprocess_common_state_dict

    save_checkpoint(
        iteration,
        [ddp],
        optimizer,
        None,
        0,
        preprocess_common_state_dict_fn=preprocess_common_state_dict,
        dp_cp_group=full.dp_cp_gtp_remat,
        dp_group=full.dp,
        dp_gtp_remat_group=full.dp_gtp_remat,
        expt_dp_group=full.expt_dp,
    )
    maybe_finalize_async_save(blocking=True)
    dist.barrier()


def load(ddp, optimizer, full) -> int:
    """``load_checkpoint`` (training resume) with the grid's groups; returns the iteration."""
    from megatron.training.checkpointing import load_checkpoint

    with mock.patch("megatron.training.checkpointing.check_checkpoint_args"):
        iteration, _ = load_checkpoint(
            [ddp],
            optimizer,
            None,
            restore_training_state=True,
            dp_cp_group=full.dp_cp_gtp_remat,
            dp_group=full.dp,
            expt_dp_group=full.expt_dp,
        )
    return iteration


def logical_weights(store, ddp) -> Dict[str, torch.Tensor]:
    """Every logical weight: stored ones all-gathered over the shard axis (pad rows cut), the
    others as registered."""
    out = {}
    stored = set()
    if store is not None:
        for tensor in store.tensors():
            stored.add(tensor.name)
            shard = tensor.shard.detach().clone()
            parts = [torch.empty_like(shard) for _ in range(store.shard_size)]
            dist.all_gather(parts, shard, group=store.groups.shard)
            out[tensor.name] = torch.cat(parts)[: tensor.rows]
    for name, param in unwrap_model(ddp).named_parameters():
        if not name.startswith("loop_param_store.") and name not in stored:
            out[name] = param.detach().clone()
    return out


def _param_names(ddp) -> Dict[int, str]:
    names = {}
    for name, param in unwrap_model(ddp).named_parameters():
        names[id(param)] = getattr(param, "loop_store_name", name)
    return names


def _instances(optimizer) -> List[Any]:
    chained = getattr(optimizer, "chained_optimizers", None)
    return list(chained) if chained is not None else [optimizer]


def optimizer_states(optimizer, ddp) -> Dict[str, torch.Tensor]:
    """``{"<param name>.<state>": tensor}``: the fp32 main param and every optimizer state tensor
    of each parameter this rank's optimizer holds (distributed: its local range; the generic
    optimizer: the whole parameter), ``step`` excluded."""
    names = _param_names(ddp)
    out = {}
    for index, instance in enumerate(_instances(optimizer)):
        index_map = getattr(instance, "model_param_group_index_map", None)
        if index_map is not None:
            for model_param in index_map:
                # pylint: disable=protected-access
                tensors = instance._get_main_param_and_optimizer_states(model_param)
                for key, tensor in tensors.items():
                    if key != "step":
                        out[f"{index}.{names[id(model_param)]}.{key}"] = tensor.detach().clone()
            continue
        inner = getattr(instance, "optimizer", instance)
        for group in inner.param_groups:
            for param in group["params"]:
                for key, tensor in inner.state.get(param, {}).items():
                    if key != "step" and isinstance(tensor, torch.Tensor):
                        name = names.get(id(param), getattr(param, "loop_store_name", "?"))
                        out[f"{index}.{name}.{key}"] = tensor.detach().clone()
    return out


def _local_state_pieces(optimizer, ddp):
    """This rank's ``(param name, state key, start, flat values)`` of every parameter range its
    optimizer holds: the distributed optimizer's local range (``_get_model_param_range_map``),
    the generic optimizer's whole parameter (start 0); ``step`` excluded; values on the CPU."""
    names = _param_names(ddp)
    pieces = []
    for instance in _instances(optimizer):
        index_map = getattr(instance, "model_param_group_index_map", None)
        if index_map is not None:
            for model_param in index_map:
                # pylint: disable=protected-access
                start = instance._get_model_param_range_map(model_param)["param"].start
                tensors = instance._get_main_param_and_optimizer_states(model_param)
                for key, tensor in tensors.items():
                    if key != "step":
                        flat = tensor.detach().reshape(-1).cpu().clone()
                        pieces.append((names[id(model_param)], key, start, flat))
            continue
        inner = getattr(instance, "optimizer", instance)
        for group in inner.param_groups:
            for param in group["params"]:
                name = names.get(id(param), getattr(param, "loop_store_name", None))
                assert name is not None, "a main parameter that is not a model parameter"
                tensors = {"param": param, **inner.state.get(param, {})}
                for key, tensor in tensors.items():
                    if key != "step" and isinstance(tensor, torch.Tensor):
                        pieces.append((name, key, 0, tensor.detach().reshape(-1).cpu().clone()))
    return pieces


def _local_layouts(store, ddp) -> Dict[str, Any]:
    """``{param name: ("stored", rows, rest) | ("plain", shape)}`` of this rank's parameters."""
    stored = {t.name: t for t in store.tensors()} if store is not None else {}
    out = {}
    for name, param in unwrap_model(ddp).named_parameters():
        name = getattr(param, "loop_store_name", name)
        tensor = stored.get(name)
        if tensor is not None:
            out[name] = ("stored", tensor.logical_shape[0], tuple(tensor.logical_shape[1:]))
        else:
            out[name] = ("plain", tuple(param.shape))
    return out


def _numel(shape) -> int:
    return int(torch.Size(shape).numel())


def logical_optimizer_states(optimizer, ddp, store):
    """The optimizer state of the whole run by logical name (a collective over the world).

    Returns ``(states, problems)``: ``states`` maps ``"<logical name>.<key>"`` (``param``: the
    fp32 main / master parameter; the optimizer's state tensors; ``step`` excluded) to the
    logical tensor, assembled from every rank's ranges (``all_gather_object``): a stored tensor
    from each shard rank's canonical-shard rows (pad rows cut), others as registered. The layout
    is the parameter's, so a store run and a store-free run (or two stored sets, two z) compare
    directly. ``problems`` lists an element held by no rank (uncovered), two ranks holding
    different values for one element (conflict), and a stored tensor's pad rows not exactly 0
    (pad)."""
    shard_rank = store.shard_rank if store is not None else 0
    z = store.shard_size if store is not None else 1
    local = (shard_rank, _local_layouts(store, ddp), _local_state_pieces(optimizer, ddp))
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    layouts: Dict[str, Any] = {}
    for _, rank_layouts, _ in gathered:
        layouts.update(rank_layouts)

    def per_of(rows):
        return -(-rows // z)

    # (name, key, shard rank) -> (values, covered)
    buffers: Dict[Any, Any] = {}
    problems: List[str] = []
    for rank_shard, _, pieces in gathered:
        for name, key, start, flat in pieces:
            layout = layouts[name]
            if layout[0] == "stored":
                slot, size = (name, key, rank_shard), per_of(layout[1]) * _numel(layout[2])
            else:
                slot, size = (name, key, 0), _numel(layout[1])
            if slot not in buffers:
                buffers[slot] = (
                    torch.zeros(size, dtype=flat.dtype),
                    torch.zeros(size, dtype=torch.bool),
                )
            values, covered = buffers[slot]
            end = start + flat.numel()
            seen = covered[start:end]
            if bool(seen.any()) and not torch.equal(values[start:end][seen], flat[seen]):
                problems.append(f"conflict {name}.{key}")
            values[start:end] = flat
            covered[start:end] = True
    states: Dict[str, torch.Tensor] = {}
    for name, key in sorted({(name, key) for name, key, _ in buffers}):
        layout = layouts[name]
        if layout[0] == "plain":
            values, covered = buffers[(name, key, 0)]
            if not bool(covered.all()):
                problems.append(f"uncovered {name}.{key}")
            states[f"{name}.{key}"] = values.view(layout[1])
            continue
        rows, rest = layout[1], layout[2]
        per = per_of(rows)
        parts = []
        for rank in range(z):
            entry = buffers.get((name, key, rank))
            if entry is None or not bool(entry[1].all()):
                problems.append(f"uncovered {name}.{key} shard {rank}")
                continue
            shard = entry[0].view(per, *rest)
            real = max(0, min(per, rows - rank * per))
            if shard[real:].numel() and bool((shard[real:] != 0).any()):
                problems.append(f"pad {name}.{key} shard {rank}")
            parts.append(shard[:real])
        states[f"{name}.{key}"] = torch.cat(parts) if parts else torch.empty(0)
    return states, sorted(problems)


def poison_pad_rows(store, value: float = 7.0) -> None:
    """Fill every canonical shard's pad rows with ``value`` (before a load: the load must zero
    them itself). A no-op without a store."""
    if store is None:
        return
    with torch.no_grad():
        for tensor in store.tensors():
            local = RowShard.of(tensor, store.shard_rank).local_rows
            if tensor.shard.shape[0] > local:
                tensor.shard[local:].fill_(value)


def pad_rows_zero(store) -> bool:
    """Every canonical shard's pad rows are exactly zero."""
    for tensor in store.tensors():
        local = RowShard.of(tensor, store.shard_rank).local_rows
        rest = tensor.shard.detach()[local:]
        if rest.numel() and bool((rest != 0).any()):
            return False
    return True


def mismatches(actual: Dict[str, torch.Tensor], expected: Dict[str, torch.Tensor]) -> List[str]:
    """Names whose tensors differ bitwise (or exist on one side only)."""
    bad = [
        name
        for name, tensor in expected.items()
        if name not in actual
        or actual[name].shape != tensor.shape
        or not torch.equal(actual[name], tensor)
    ]
    return sorted(bad + [name for name in actual if name not in expected])
