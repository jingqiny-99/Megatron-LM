# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Training side of hybrid loop checkpoints (M3c).

**Coverage backstop.** ``check_loop_sharded_state_dict`` raises when a state tensor of a hybrid
loop model (a ``HybridModel`` with a ``LoopHybridStack`` decoder) has no entry in its sharded
state dict, by the rule of ``megatron/core/transformer/loop/param_store_checkpoint.py``. A
missing entry would otherwise be silently re-initialized on resume (and, on the load template,
hidden by ``load_checkpoint``'s ``strict=False`` retry). It runs

* at setup (``setup_model_and_optimizer``, after the store and the planner, when ``--save`` or
  ``--load`` is set with ``--ckpt-format torch_dist``), so a chained job fails in its first
  minute rather than at its first ``--save-interval``;
* at the first save of the process (cached in the ``checkpointing_context``), on the state dict
  being saved;
* on every load template (torch_dist).

**The loop parameter store** (``--loop-param-store-shard-size`` z > 1) saves its canonical shards
as row shards of the logical tensors under their z = 1 keys (``param_store_checkpoint``), so a
torch_dist checkpoint loads at any z, in any residency mode, with or without a store; the
distributed optimizer's ``fully_reshardable`` format follows the same row shards. What stays
refused, with a precise message (numbers of the stage-8 design):

1. saving in a format other than torch_dist (``torch`` / ``torch_dcp`` / ``fsdp_dtensor`` write
   per (TP, PP, EP) rank from data-parallel rank 0, dropping z - 1 shards), or a local
   non-persistent checkpoint (``loop_checkpoint_problems`` and ``check_loop_save_format``);
2. loading the optimizer state of a legacy (``torch``) checkpoint (its weights load: the store
   slices the full tensors), and loading a ``torch_dcp`` / ``fsdp_dtensor`` checkpoint
   (``check_loop_load_format``);
3. a ``dp_reshardable`` (bucket-space) distributed-optimizer state whose bucket layout differs
   from this run's (a z, stored-set or EP change; dev would assert later, bare), decided per
   distributed-optimizer instance and data-parallel group index from the
   ``per_bucket_numel_unpadded`` lists recorded in the checkpoint's ``loop_resume`` entry, and
   from the saved store layout's z and stored-set digest, every rank raising together
   (``check_loop_optimizer_resharding``);
4. ``--distrib-optim-fully-reshardable-mem-efficient`` (it gathers over the distributed
   optimizer's Gloo group, which the GTP-remat grid does not create);
10. the chunked optimizer-state offload (untested with the store);
11. a GPT-to-hybrid compat load;
12. ``--loop-param-store-binding distributed_weight`` with ``--save`` / ``--load`` (untested).

**The ``loop_resume`` common-state entry** (``loop_resume_state``, saved next to the iteration
in ``common.pt``): derived loop state in plain types (the roles, the pattern, the store layout
and modes, the distributed-optimizer bucket layout of store runs, the applied RSBF plan, the
expert-parallel and world sizes). The loop argument values are not copied there: the resume
check reads the checkpoint's own saved ``args``.

**Resume state (message 5).** ``check_loop_resume_state`` (``load_checkpoint``, when the training
state is restored and not finetuning) compares every ``loop_*`` argument of the checkpoint's
saved ``args`` with this run's, the roles, the layer pattern and the applied RSBF plan
(``loop_rsbf_planning.check_resume_plan``); a numerics-relevant change stops the resume unless
``--loop-resume-changes allow``. Layout-neutral arguments (``is_layout_neutral``: the store
layout, every planner input, the listed logging settings) and the bitwise-interchangeable
item forms (``SWAPPABLE_LOOP_ARGS``) are notes. Everything else a resumed loop run needs is
restored by Megatron's own state or recomputed (the inventory of the stage-8 design, 7.3): the
weights and persistent buffers (router ``expert_bias``, hash tables) through the model entries,
the optimizer and scheduler, the iteration and consumed samples (data order), the per-rank RNG
state (the eager stack's ``--loop-sample-depth`` draws use the CPU default generator; the
data loader owns its generator, so building the iterators after the load does not shift it),
while the RSBF depth draws are counter-based on (seed, iteration, sample) and the executor,
store buffers, aux-loss scales and SBHD packing are rebuilt each step or at startup.

For any non-loop model every call here is a silent no-op.
"""

import argparse
import functools
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Set

import torch

from megatron.core.utils import unwrap_model

__all__ = [
    "COVERAGE_CHECKED_KEY",
    "LAYOUT_NEUTRAL_LOOP_ARGS",
    "LOOP_RESUME_KEY",
    "LOOP_RESUME_VERSION",
    "SWAPPABLE_LOOP_ARGS",
    "check_loop_load_format",
    "check_loop_optimizer_resharding",
    "check_loop_resume_state",
    "check_loop_save_format",
    "check_loop_sharded_state_dict",
    "is_layout_neutral",
    "is_loop_model",
    "loop_checkpoint_problems",
    "loop_hybrid_model",
    "loop_resume_state",
]

# The common-state entry of a hybrid loop checkpoint (``loop_resume_state``).
LOOP_RESUME_KEY = "loop_resume"
LOOP_RESUME_VERSION = 1
# The evidence line of message 10 (the stage-8 S8.0 chunked-offload arm).
CHUNKED_OFFLOAD_EVIDENCE = "S8.0 chunked-offload arm: not run (no GPU phase of stage 8 yet)"

# checkpointing_context key: the save-time coverage check passed in this process.
COVERAGE_CHECKED_KEY = "loop_sharded_state_dict_checked"


def loop_hybrid_model(model: Any) -> Optional[torch.nn.Module]:
    """The ``HybridModel`` with a ``LoopHybridStack`` decoder inside ``model`` (DDP /
    ``Float16Module`` wrappers, a one-chunk list), or None for any other model."""
    chunks = model if isinstance(model, (list, tuple)) else [model]
    if len(chunks) != 1:
        return None
    unwrapped = unwrap_model(chunks[0])
    if not isinstance(unwrapped, torch.nn.Module):
        return None
    decoder = unwrapped._modules.get("decoder")  # pylint: disable=protected-access
    if decoder is None:
        return None
    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack

    return unwrapped if isinstance(decoder, LoopHybridStack) else None


def is_loop_model(model: Any) -> bool:
    """True for a hybrid loop model (``loop_hybrid_model``)."""
    return loop_hybrid_model(model) is not None


def check_loop_sharded_state_dict(
    model: Any,
    *,
    when: str,
    sharded_state_dict: Optional[Any] = None,
    metadata: Optional[dict] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Raise when a state tensor of a hybrid loop model has no checkpoint entry.

    Args:
        model: the model (wrapped or not, a one-chunk list accepted); no-op for non-loop models.
        when: ``"setup"``, ``"save"`` or ``"load"`` (named in the message).
        sharded_state_dict: the model's (sharded) state dict as built for the save or the load
            template; built with ``metadata`` when None.
        metadata: ``sharded_state_dict`` metadata when the dict is built here.
        context: the ``checkpointing_context``: the check runs once per context (first save).
    """
    hybrid = loop_hybrid_model(model)
    if hybrid is None:
        return
    if context is not None and context.get(COVERAGE_CHECKED_KEY):
        return
    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer.loop.param_store_checkpoint import uncovered_state_names

    if sharded_state_dict is None:
        sharded_state_dict = hybrid.sharded_state_dict(metadata=metadata)
    names = uncovered_state_names(hybrid, sharded_state_dict)
    if names:
        raise RuntimeError(
            f"loop checkpoint ({when}): the sharded state dict has no entry for "
            f"{', '.join(names)} (a parameter or persistent buffer without an entry would be "
            "silently re-initialized on resume); give its module a sharded_state_dict or "
            "register it on a module the default walk reaches"
        )
    if context is not None:
        context[COVERAGE_CHECKED_KEY] = True


# ---------------------------------------------------------------------------------------------
# The loop parameter store: formats, optimizer resharding, the loop_resume entry
# ---------------------------------------------------------------------------------------------


def _store_of(model: Any):
    """The loop parameter store installed on ``model`` (a hybrid loop model), or None."""
    hybrid = loop_hybrid_model(model)
    if hybrid is None:
        return None
    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer.loop.param_store import LoopParamStore

    store = hybrid._modules.get("loop_param_store")  # pylint: disable=protected-access
    return store if isinstance(store, LoopParamStore) else None


def _shard_size(args) -> int:
    value = getattr(args, "loop_param_store_shard_size", 1)
    return 1 if value is None else int(value)


def _save_format_message(fmt: str, z: int) -> str:
    return (
        f"--ckpt-format {fmt} with the loop parameter store (z = {z}): {fmt} writes one file per "
        "(TP, PP, EP) rank from data-parallel rank 0, so z - 1 of the z canonical shards and "
        "their optimizer state would be dropped; save with --ckpt-format torch_dist (a global "
        "checkpoint)"
    )


LOCAL_NON_PERSISTENT_MESSAGE = (
    "local non-persistent checkpoints (--non-persistent-ckpt-type local) with the loop parameter "
    "store are not supported; use global non-persistent checkpoints or --save"
)


def loop_checkpoint_problems(args) -> List[str]:
    """The checkpoint argument rules of the loop parameter store (z > 1; empty otherwise),
    messages 1, 4, 10 and 12. Reads every argument with ``getattr``."""
    z = _shard_size(args)
    if z <= 1:
        return []
    problems = []
    saving = bool(getattr(args, "save", None))
    checkpointing = saving or any(
        getattr(args, name, None) for name in ("load", "pretrained_checkpoint")
    )
    fmt = getattr(args, "ckpt_format", "torch_dist")
    if saving and fmt != "torch_dist":
        problems.append(_save_format_message(fmt, z))
    if getattr(args, "non_persistent_ckpt_type", None) == "local" and getattr(
        args, "non_persistent_save_interval", None
    ):
        problems.append(LOCAL_NON_PERSISTENT_MESSAGE)
    if (
        checkpointing
        and getattr(args, "dist_ckpt_optim_fully_reshardable", False)
        and getattr(args, "distrib_optim_fully_reshardable_mem_efficient", False)
    ):
        problems.append(
            "--distrib-optim-fully-reshardable-mem-efficient with the loop parameter store: it "
            "gathers over the distributed optimizer's Gloo group, which the GTP-remat grid does "
            "not create; drop the flag (the NCCL gather is used)"
        )
    chunked = getattr(args, "chunked_optimizer_state_offload", False) or getattr(
        args, "offload_optimizer_states", False
    )
    if chunked and (getattr(args, "optimizer_state_offload_fraction", 1.0) or 0.0) > 0.0:
        problems.append(
            "--chunked-optimizer-state-offload with the loop parameter store is untested "
            f"({CHUNKED_OFFLOAD_EVIDENCE}); the plan's Grace optimizer state is "
            "--optimizer-cpu-offload"
        )
    if checkpointing and getattr(args, "loop_param_store_binding", "data") == "distributed_weight":
        problems.append(
            "checkpointing with --loop-param-store-binding distributed_weight is untested (the "
            "EOS / lyris containers lack TE >= 2.19); use the default data binding"
        )
    return problems


def check_loop_save_format(model: Any, *, ckpt_format: str, local: bool = False) -> None:
    """``save_checkpoint``'s backstop of message 1: with a store installed, only a global
    torch_dist checkpoint keeps every canonical shard (``local``: a local non-persistent
    checkpoint). A no-op without a store."""
    store = _store_of(model)
    if store is None:
        return
    if local:
        raise RuntimeError(LOCAL_NON_PERSISTENT_MESSAGE)
    if ckpt_format != "torch_dist":
        raise RuntimeError(_save_format_message(ckpt_format, store.shard_size))


def check_loop_load_format(
    model: Any, *, ckpt_format: Optional[str], load_optim: bool, gpt_compat: bool
) -> None:
    """``load_checkpoint``'s format rules with a store installed (messages 2 and 11): a legacy
    (``torch``) checkpoint loads its weights only, ``torch_dcp`` / ``fsdp_dtensor`` checkpoints
    and the GPT-to-hybrid maps not at all. A no-op without a store."""
    store = _store_of(model)
    if store is None or ckpt_format is None:
        return
    z = store.shard_size
    if gpt_compat:
        raise RuntimeError(
            "GPT-to-hybrid checkpoint maps with the loop parameter store are not supported; load "
            "the GPT checkpoint at z = 1 and re-save with --ckpt-convert-format torch_dist"
        )
    if ckpt_format == "torch" and load_optim:
        raise RuntimeError(
            "loading the optimizer state of a legacy (torch) checkpoint into a loop parameter "
            f"store run (z = {z}) is not supported: its distributed-optimizer files are per "
            "(TP, PP) rank, not per shard; load the weights only with --no-load-optim (the loop "
            "parameter store slices the full tensors), or re-save the checkpoint at z = 1 with "
            "--ckpt-convert-format torch_dist"
        )
    if ckpt_format in ("torch_dcp", "fsdp_dtensor"):
        raise RuntimeError(
            f"loading a {ckpt_format} checkpoint into a loop parameter store run (z = {z}) is not "
            "supported (its model state is keyed by the store's mirror shards); re-save it at "
            "z = 1 with --ckpt-convert-format torch_dist"
        )


def _distopt_instances(optimizer: Any) -> List[Any]:
    """The distributed-optimizer instances of a (chained) optimizer, in order."""
    if optimizer is None:
        return []
    chained = getattr(optimizer, "chained_optimizers", None)
    members = list(chained) if chained is not None else [optimizer]
    return [m for m in members if hasattr(m, "per_bucket_numel_unpadded")]


def _plain(value: Any) -> Any:
    """``value`` (nested lists / tuples / dicts of ints) as plain lists, dicts and ints."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return int(value)


def _local_buckets(optimizer: Any) -> Dict[str, List[Any]]:
    """``{instance: [data-parallel group index, per_bucket_numel_unpadded]}`` of this rank."""
    return {
        str(index): [
            int(instance.data_parallel_group_idx),
            _plain(instance.per_bucket_numel_unpadded),
        ]
        for index, instance in enumerate(_distopt_instances(optimizer))
    }


def _bucket_record(optimizer: Any) -> Dict[str, Dict[str, Any]]:
    """``{instance: {data-parallel group index: per_bucket_numel_unpadded}}`` over the world (a
    collective: every rank calls it)."""
    local = _local_buckets(optimizer)
    gathered: List[Any] = [local]
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local)
    record: Dict[str, Dict[str, Any]] = {}
    for rank_buckets in gathered:
        for instance, (group_idx, numels) in (rank_buckets or {}).items():
            record.setdefault(instance, {})[str(group_idx)] = numels
    return record


def loop_resume_state(args, model: Any, optimizer: Any) -> Optional[Dict[str, Any]]:
    """The ``loop_resume`` common-state entry of a hybrid loop model (None otherwise), plain
    types only (``torch.load(weights_only=True)``-safe): the roles, the pattern, the store's
    checkpoint layout and modes, the distributed-optimizer bucket layout (with a store: gathered
    over the world, a collective; ``save_checkpoint`` calls it on every rank of a torch_dist
    save, the only format a store saves), the applied RSBF plan
    (``loop_rsbf_planning.loop_plan_record``; None without ``--loop-planner apply``), the
    expert-parallel and world sizes."""
    hybrid = loop_hybrid_model(model)
    if hybrid is None:
        return None
    store = _store_of(model)
    roles = getattr(hybrid.decoder, "loop_roles", None)
    return {
        "version": LOOP_RESUME_VERSION,
        "roles": (
            None
            if roles is None
            else {"prelude": int(roles.prelude), "core": int(roles.core), "coda": int(roles.coda)}
        ),
        "pattern": getattr(args, "hybrid_layer_pattern", None),
        "store": store.checkpoint_layout() if store is not None else None,
        "store_modes": (
            store.modes.describe() if store is not None and store.modes is not None else None
        ),
        "buckets": _bucket_record(optimizer) if store is not None else None,
        "plan": _applied_plan_record(args),
        "ep": int(getattr(args, "expert_model_parallel_size", 1) or 1),
        "world": int(getattr(args, "world_size", 1) or 1),
    }


def _applied_plan_record(args) -> Optional[Dict[str, Any]]:
    """``loop_rsbf_planning.loop_plan_record`` of the process's applied plan (None without one:
    planner off, report mode)."""
    # pylint: disable=import-outside-toplevel
    from megatron.training.loop_rsbf_planning import get_loop_planner_session, loop_plan_record

    session = get_loop_planner_session()
    if not session.applied or session.plan is None:
        return None
    return loop_plan_record(
        session.plan, depth_source=getattr(args, "loop_planner_depths", "config")
    )


def _layout_text(layout: Optional[Dict[str, Any]], ep: Any) -> str:
    if not layout:
        return f"no store, EP {ep}"
    return (
        f"z = {layout.get('z')}, {layout.get('stored')} stored tensors (set "
        f"{str(layout.get('stored_sha'))[:12]}), min_numel {layout.get('min_numel')}, EP {ep}"
    )


def _max_over_world(flag: int) -> int:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return flag
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.distributed.get_backend() == "nccl"
        else torch.device("cpu")
    )
    value = torch.tensor([flag], dtype=torch.int32, device=device)
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
    return int(value.item())


def check_loop_optimizer_resharding(
    args,
    model: Any,
    optimizer: Any,
    saved: Optional[Dict[str, Any]],
    sharding_type: Optional[str],
    *,
    reduce_max: Optional[Callable[[int], int]] = None,
) -> None:
    """Message 3: with a store installed and a ``dp_reshardable`` checkpoint, every
    distributed-optimizer instance's ``per_bucket_numel_unpadded`` (for this rank's data-parallel
    group index) must equal the checkpoint's (dev asserts it later, bare), and the store's z and
    stored set (``stored_sha``: names and logical shapes in unit order) must equal the saved
    ones: equal per-bucket totals alone would let a reordered or exchanged stored set load
    misordered state. A checkpoint without a bucket record was saved without a store, whose
    layout differs. The mismatch flag is MAX-reduced over the world so every rank raises
    together (``reduce_max`` replaces the reduction in tests). A no-op without a store or for the
    model-space formats."""
    store = _store_of(model)
    if store is None or sharding_type != "dp_reshardable":
        return
    record = (saved or {}).get("buckets")
    saved_layout = (saved or {}).get("store") or {}
    layout = store.checkpoint_layout()
    mismatch = 0
    if record is None or any(saved_layout.get(key) != layout[key] for key in ("z", "stored_sha")):
        mismatch = 1
    else:
        for instance, (group_idx, numels) in _local_buckets(optimizer).items():
            if record.get(instance, {}).get(str(group_idx)) != numels:
                mismatch = 1
    mismatch = (reduce_max or _max_over_world)(mismatch)
    if not mismatch:
        return
    saved_text = (
        "saved without a loop parameter store, no bucket record"
        if record is None
        else _layout_text((saved or {}).get("store"), (saved or {}).get("ep"))
    )
    now = _layout_text(layout, getattr(args, "expert_model_parallel_size", 1))
    raise RuntimeError(
        "the checkpoint's distributed-optimizer state is in the dp_reshardable (bucket) format, "
        "whose layout depends on the loop parameter store (saved: "
        f"{saved_text}; this run: {now}); it reshards over the data-parallel (replicate) "
        "degree only. Save with --dist-ckpt-optim-fully-reshardable to change z, the stored set "
        "or EP, or load with --no-load-optim"
    )


# ---------------------------------------------------------------------------------------------
# Resume state: loop argument, role, pattern and plan changes (message 5)
# ---------------------------------------------------------------------------------------------

# Loop arguments whose change does not change what the resumed run computes beyond the fp32
# association of a different layout (the store), or not at all (the planner mode, this check's
# own switch, logging). Every ``loop_planner_*`` / ``loop_param_store_*`` argument is neutral
# too (``is_layout_neutral``). Logging and reporting arguments are listed by name, not matched by
# a suffix, so a new numerics-relevant ``loop_*_interval`` / ``_path`` argument is refused until
# it is classified here (test_loop_checkpoint_args pins the classification of every argument).
LAYOUT_NEUTRAL_LOOP_ARGS: FrozenSet[str] = frozenset(
    {
        "loop_param_store_shard_size",
        "loop_param_store_modes",
        "loop_param_store_min_numel",
        "loop_param_store_binding",
        "loop_param_store_separate_rs_group",
        "loop_param_store_gather_engine",
        "loop_param_store_reduce_engine",
        "loop_param_store_copy_engine_timeout_ms",
        "loop_param_store_debug",
        "loop_planner",
        "loop_resume_changes",
        "loop_exit_histogram_path",
    }
)
LAYOUT_NEUTRAL_PREFIXES = ("loop_planner_", "loop_param_store_")
# Store arguments whose change is reported (the layout the resumed run continues in).
_STORE_LAYOUT_PREFIX = "loop_param_store_"
_SILENT_STORE_ARGS = frozenset(
    {"loop_param_store_debug", "loop_param_store_copy_engine_timeout_ms"}
)
# Value changes proven bitwise-equivalent on resume (stage 4: the nested item equals the whole
# round bitwise); the block stride does not apply to either.
SWAPPABLE_LOOP_ARGS: Dict[str, Set[FrozenSet[Any]]] = {
    "loop_rsbf_item_form": {frozenset({"whole_round", "nested"})}
}

_ABSENT = object()


def is_layout_neutral(name: str) -> bool:
    """A loop argument whose change does not stop a resume (``LAYOUT_NEUTRAL_LOOP_ARGS`` and the
    planner and store prefixes)."""
    return name in LAYOUT_NEUTRAL_LOOP_ARGS or name.startswith(LAYOUT_NEUTRAL_PREFIXES)


@functools.lru_cache(maxsize=1)
def _parser_defaults() -> Dict[str, Any]:
    """The parser defaults of every ``loop_*`` argument (for an argument a checkpoint predates)."""
    # pylint: disable=import-outside-toplevel
    from megatron.training.arguments import add_megatron_arguments

    parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
    return {
        action.dest: action.default
        for action in parser._actions  # pylint: disable=protected-access
        if action.dest.startswith("loop_")
    }


def _plain_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    return value


def _loop_names(namespace: Any) -> Set[str]:
    try:
        names = vars(namespace)
    except TypeError:
        return set()
    return {name for name in names if name.startswith("loop_")}


def _swappable(name: str, old: Any, new: Any) -> bool:
    return any({old, new} <= group for group in SWAPPABLE_LOOP_ARGS.get(name, ()))


def _loop_argument_changes(saved_args: Any, args: Any):
    """``(refused, notes)``: the changed numerics-relevant ``loop_*`` arguments ``{name: (saved,
    now)}`` and the notes (store layout, swappable item forms, arguments this run no longer
    has). An argument the checkpoint predates compares against its parser default."""
    refused: Dict[str, Any] = {}
    notes: List[str] = []
    names = sorted(_loop_names(saved_args) | _loop_names(args))
    forms = (
        getattr(saved_args, "loop_rsbf_item_form", _ABSENT),
        getattr(args, "loop_rsbf_item_form", _ABSENT),
    )
    forms_swappable = _swappable("loop_rsbf_item_form", *forms)
    for name in names:
        old = getattr(saved_args, name, _ABSENT)
        new = getattr(args, name, _ABSENT)
        if new is _ABSENT:
            notes.append(
                f"loop resume: the checkpoint's argument {name}={old!r} is not one of this "
                "run's (ignored)"
            )
            continue
        if old is _ABSENT:
            old = _parser_defaults().get(name)
        if _plain_value(old) == _plain_value(new):
            continue
        if name.startswith(_STORE_LAYOUT_PREFIX):
            if name not in _SILENT_STORE_ARGS:
                notes.append(
                    f"loop resume: {name} {old!r} -> {new!r} (layout-neutral: the resumed run "
                    "continues within the fp32 association of the new layout, not bitwise)"
                )
            continue
        if is_layout_neutral(name):
            continue
        if name == "loop_rsbf_item_form" and forms_swappable:
            notes.append(f"loop resume: {name} {old!r} -> {new!r} (bitwise interchangeable)")
            continue
        if name == "loop_rsbf_block_stride" and forms_swappable:
            continue  # the whole round and the nested item take no block stride
        refused[name] = (old, new)
    return refused, notes


_LAYOUT_KEYS = ("z", "stored", "min_numel")
# The argument notes a store layout / mode note (``_store_layout_notes``) supersedes.
_STORE_LAYOUT_ARG_NOTES = tuple(
    f"loop resume: loop_param_store_{name} " for name in ("shard_size", "min_numel", "modes")
)


def _layout_brief(layout: Optional[Mapping[str, Any]]) -> str:
    if not layout:
        return "no store"
    return ", ".join(f"{key} {layout.get(key)}" for key in _LAYOUT_KEYS)


def _store_layout_notes(saved: Mapping[str, Any], store) -> List[str]:
    """Notes for a store layout (z, stored set, min_numel) or mode change against the
    checkpoint's ``loop_resume`` record."""
    then = saved.get("store")
    now = store.checkpoint_layout() if store is not None else None
    notes = []
    if (
        (then is None) != (now is None)
        or then is not None
        and any(then.get(key) != now.get(key) for key in _LAYOUT_KEYS)
    ):
        notes.append(
            f"loop resume: parameter store {_layout_brief(then)} -> {_layout_brief(now)} "
            "(layout-neutral: the resumed run continues within the fp32 association of the new "
            "layout)"
        )
    modes_now = store.modes.describe() if store is not None and store.modes is not None else None
    if saved.get("store_modes") != modes_now and then is not None and now is not None:
        notes.append(
            f"loop resume: store modes {saved.get('store_modes')} -> {modes_now} (layout-neutral)"
        )
    return notes


def check_loop_resume_state(
    args, model: Any, saved_state: Optional[Mapping[str, Any]], *, restore_training_state: bool
) -> List[str]:
    """Message 5: a resume of a hybrid loop model must keep the checkpoint's numerics.

    Called by ``load_checkpoint`` once the checkpoint's common state is read (torch_dist: before
    the model and optimizer state are loaded; legacy: with the full state), when the training
    state is restored and the run is not finetuning. Compares every ``loop_*`` argument of the
    checkpoint's saved ``args`` with this run's (``_loop_argument_changes``: layout-neutral and
    swappable changes are notes), the loop roles and the layer pattern recorded in the
    checkpoint's ``loop_resume`` entry, and the applied RSBF plan
    (``loop_rsbf_planning.check_resume_plan``). Raises ``RuntimeError`` (message 5) for any
    numerics-relevant change unless ``--loop-resume-changes allow`` (then every change is a
    note). Returns the notes, printed on rank 0. A no-op returning [] (nothing logged) for a
    non-loop model, without ``restore_training_state``, or when finetuning; every argument is
    read with ``getattr``."""
    hybrid = loop_hybrid_model(model)
    if hybrid is None or not restore_training_state or getattr(args, "finetune", False):
        return []
    # pylint: disable=import-outside-toplevel
    from megatron.training.loop_rsbf_planning import check_resume_plan, get_loop_planner_session
    from megatron.training.utils import print_rank_0

    state = saved_state or {}
    saved_args = state.get("args")
    saved = state.get(LOOP_RESUME_KEY)
    notes: List[str] = []
    refused: Dict[str, Any] = {}
    if saved_args is not None:
        refused, notes = _loop_argument_changes(saved_args, args)
    if saved is None:
        notes.append(
            "loop resume: the checkpoint has no loop_resume entry (saved before M3c or by a "
            "non-loop model); only its saved loop arguments are compared"
        )
        saved = {}
    roles = getattr(hybrid.decoder, "loop_roles", None)
    roles_now = (
        None
        if roles is None
        else {"prelude": int(roles.prelude), "core": int(roles.core), "coda": int(roles.coda)}
    )
    if saved.get("roles") is not None and saved.get("roles") != roles_now:
        refused["loop roles"] = (saved.get("roles"), roles_now)
    pattern_now = getattr(args, "hybrid_layer_pattern", None)
    if "pattern" in saved and saved.get("pattern") != pattern_now:
        refused["hybrid_layer_pattern"] = (saved.get("pattern"), pattern_now)
    if saved:
        layout_notes = _store_layout_notes(saved, _store_of(model))
        if layout_notes:
            # The effective layout note covers the arguments that set it (one note, not two).
            notes = [n for n in notes if not n.startswith(_STORE_LAYOUT_ARG_NOTES)]
        notes += layout_notes
        world_now = int(getattr(args, "world_size", 1) or 1)
        ep_now = int(getattr(args, "expert_model_parallel_size", 1) or 1)
        if saved.get("world") not in (None, world_now) or saved.get("ep") not in (None, ep_now):
            notes.append(
                f"loop resume: world {saved.get('world')} -> {world_now}, EP {saved.get('ep')} "
                f"-> {ep_now} (the data-parallel reduction changes: not bitwise)"
            )
    session = get_loop_planner_session()
    plan_problem, plan_notes = check_resume_plan(
        args,
        saved.get("plan"),
        session.plan if session.applied else None,
        pin_reason=session.pin_reason,
    )
    notes += plan_notes
    if refused or plan_problem:
        changes = ", ".join(f"{name}: {old!r} -> {new!r}" for name, (old, new) in refused.items())
        parts = [part for part in (changes, plan_problem) if part]
        if getattr(args, "loop_resume_changes", "refuse") != "allow":
            raise RuntimeError(
                "loop resume: this run differs from the checkpoint in "
                + "; ".join(parts)
                + "; the resumed loss curve would not continue the checkpoint's. Resume with the "
                "checkpoint's values (for the plan: --loop-planner report with the saved choice "
                "set by hand), or pass --loop-resume-changes allow"
            )
        notes += [f"loop resume (allowed by --loop-resume-changes allow): {part}" for part in parts]
    for note in notes:
        print_rank_0(note)
    return notes
