# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Training-side lines of the hybrid loop: the ring's per-step G1 line and memory line, the
unique-parameter census and the per-tensor parameter checksums.

* ``log_loop_ring_step`` (``train()``, after every training step of ``loop_schedule='ring'``):
  the step's G1 line from the schedule's own counters (``ring_pp.LoopRingStepRecord``),
  ``loop ring: rounds per sample min 4 max 4; prelude/coda/mtp per sample 1/1/1 (iteration N, M
  microbatches)``, and at log steps the memory line, the MAX over every rank of
  ``max_memory_allocated`` / ``max_memory_reserved`` with the rank that holds it (a collective on
  every rank).
* ``log_loop_param_census`` (``setup_model_and_optimizer``, after the model build): the number of
  **unique** parameters of a hybrid loop model, ``loop param census: unique parameters U = dense
  D + routed experts E``, with every pipeline rank's chunks counted once: the pipeline sum of the
  local parameters outside the routed experts, without the MTP stage's copy of the word
  embedding (a parameter marked ``shared``), plus EP x ETP x the pipeline sum of the local
  routed-expert parameters (each MoE layer's ``experts`` module). The same model printed by a
  PP = 1 run at expert-parallel size e reads ``D + E / e`` (Megatron's per-rank parameter line);
  the census lists that form for every e that divides the routed experts.
* ``log_loop_param_checksums`` (``--loop-param-checksums``, after the checkpoint load): one line
  per parameter or persistent buffer of a hybrid loop model, keyed by its **global** name (the
  PP = 1 name: a ring chunk's ``decoder.layers.<i>`` shifted by the chunk's offset, a routed
  expert's local index shifted to its global index), with the element count, the dtype, the
  exact integer sum of the raw element bits and the float64 sum and sum of squares. Replicas
  (data-parallel ranks, the MTP stage's embedding copy) must agree; a key whose holders disagree
  prints once per distinct value with ``MISMATCH``. Two runs of different layouts loaded from one
  checkpoint print the same lines.

For any model that is not a hybrid loop model the census and the checksums print nothing.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from megatron.training.loop_checkpoint import loop_hybrid_models
from megatron.training.utils import print_rank_0, print_rank_last

__all__ = [
    "log_loop_param_census",
    "log_loop_param_checksums",
    "log_loop_ring_step",
    "loop_param_census",
    "loop_param_checksum_lines",
    "loop_param_checksums",
    "loop_ring_memory_line",
]


# ---------------------------------------------------------------------------------------------
# The ring's step lines
# ---------------------------------------------------------------------------------------------


def loop_ring_memory_line() -> str:
    """``loop ring memory: max allocated X MB (rank r), max reserved Y MB (rank s)``: the MAX over
    every rank of this process's ``torch.cuda.max_memory_allocated`` / ``max_memory_reserved``
    (MB = 2**20 bytes, as Megatron's memory report), with the highest rank holding each maximum. A
    collective over the default group: every rank calls it."""
    device = torch.device("cuda", torch.cuda.current_device())
    local = torch.tensor(
        [torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
        dtype=torch.float64,
        device=device,
    )
    maxima = local.clone()
    torch.distributed.all_reduce(maxima, op=torch.distributed.ReduceOp.MAX)
    rank = torch.distributed.get_rank()
    holders = torch.where(
        local == maxima, torch.full_like(local, float(rank)), torch.full_like(local, -1.0)
    )
    torch.distributed.all_reduce(holders, op=torch.distributed.ReduceOp.MAX)
    (allocated, reserved), (allocated_rank, reserved_rank) = maxima.tolist(), holders.tolist()
    return (
        f"loop ring memory: max allocated {allocated / 2**20:.2f} MB (rank {int(allocated_rank)}), "
        f"max reserved {reserved / 2**20:.2f} MB (rank {int(reserved_rank)})"
    )


def log_loop_ring_step(args, schedule, iteration: int) -> None:
    """Print the last training step's G1 line (``schedule.last_step``) and, when ``iteration``
    is a log step, the memory line (``loop_ring_memory_line``: every rank must call this)."""
    record = schedule.last_step
    if record is not None and not record.forward_only:
        print_rank_last(
            f"{record.summary()} (iteration {iteration}, {record.microbatches} microbatches, "
            f"{record.views} round views per stage)"
        )
    interval = getattr(args, "log_interval", None) or 0
    if interval > 0 and iteration % interval == 0:
        print_rank_last(loop_ring_memory_line())


# ---------------------------------------------------------------------------------------------
# Unique-parameter census
# ---------------------------------------------------------------------------------------------


def _pipeline_group(model):
    """The pipeline group of a loop model's chunks (their ``pg_collection.pp``), or None."""
    hybrids = loop_hybrid_models(model)
    collection = getattr(hybrids[0], "pg_collection", None) if hybrids else None
    return getattr(collection, "pp", None)


def _routed_experts(hybrid) -> List[Tuple[str, int, int]]:
    """``(path, global index of its first local expert, expert-parallel rank)`` of every MoE
    layer's routed-experts module (the ``experts`` child of a module that holds
    ``num_local_experts`` and an ``ep_group``), the deepest path first. Structural, not by the
    ``allreduce`` attribute: at EP 1 Megatron reduces the experts with the dense parameters."""
    routed = []
    for name, module in hybrid.named_modules():
        local = getattr(module, "num_local_experts", None)
        experts = getattr(module, "experts", None)
        if local is None or not isinstance(experts, nn.Module) or not hasattr(module, "ep_group"):
            continue
        group = module.ep_group
        rank = group.rank() if group is not None else 0
        routed.append((f"{name}.experts" if name else "experts", rank * int(local), rank))
    return sorted(routed, key=lambda entry: -len(entry[0]))


def _routed_entry(name: str, routed: List[Tuple[str, int, int]]) -> Optional[Tuple[str, int, int]]:
    """The routed-experts module holding state entry ``name`` (``_routed_experts``), or None."""
    for entry in routed:
        if name.startswith(entry[0] + "."):
            return entry
    return None


def loop_param_census(model, args) -> Optional[Dict[str, Any]]:
    """The unique-parameter census of a hybrid loop model (module docstring), or None for any
    other model and with the loop parameter store (its prints are the unsharded counts already).
    A collective over the pipeline group: every rank calls it."""
    hybrids = loop_hybrid_models(model)
    if not hybrids or (getattr(args, "loop_param_store_shard_size", 1) or 1) > 1:
        return None
    dense = expert = 0
    for hybrid in hybrids:
        routed = _routed_experts(hybrid)
        for name, param in hybrid.named_parameters():
            if _routed_entry(name, routed) is not None:
                expert += param.numel()
            elif not getattr(param, "shared", False):
                dense += param.numel()
    counts = [dense, expert]
    group = _pipeline_group(model)
    if (
        group is not None
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size(group) > 1
    ):
        backend = torch.distributed.get_backend(group)
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if backend == "nccl"
            else torch.device("cpu")
        )
        totals = torch.tensor(counts, dtype=torch.int64, device=device)
        torch.distributed.all_reduce(totals, group=group)
        counts = [int(value) for value in totals.tolist()]
    expert_shards = (getattr(args, "expert_model_parallel_size", 1) or 1) * (
        getattr(args, "expert_tensor_parallel_size", 1) or 1
    )
    dense, routed = counts[0], counts[1] * expert_shards
    experts = getattr(args, "num_experts", None) or 0
    per_rank = {
        ep: dense + routed // ep
        for ep in range(1, experts + 1)
        if experts % ep == 0 and routed % ep == 0
    }
    return {"unique": dense + routed, "dense": dense, "routed": routed, "per_rank": per_rank}


def log_loop_param_census(model, args) -> Optional[Dict[str, Any]]:
    """Print the census (``loop_param_census``) on rank 0; returns it (every rank)."""
    census = loop_param_census(model, args)
    if census is None:
        return None
    forms = ", ".join(f"EP {ep}: {count}" for ep, count in census["per_rank"].items())
    print_rank_0(
        f"loop param census: unique parameters {census['unique']} = dense {census['dense']} "
        f"+ routed experts {census['routed']}"
        + (f"; per-rank form dense + routed / EP: {forms}" if forms else "")
    )
    return census


# ---------------------------------------------------------------------------------------------
# Per-tensor checksums keyed by global name
# ---------------------------------------------------------------------------------------------

_LAYER = re.compile(r"^decoder\.layers\.(\d+)\.")
_SEQUENTIAL_EXPERT = re.compile(r"^local_experts\.(\d+)\.")
_GROUPED_EXPERT = re.compile(r"^(linear_fc[12]\.(?:weight|bias))(\d+)$")
_BIT_VIEWS = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _global_name(name: str, offset: int, routed: List[Tuple[str, int, int]]) -> str:
    """The PP = 1 name of a chunk's state entry. An entry of a routed-experts module
    (``_routed_experts``) gets its global expert index (a TE grouped linear's ``weight<i>`` /
    ``bias<i>``, a sequential MLP's ``local_experts.<i>``; any other form an ``@ep<rank>``
    suffix); then ``decoder.layers.<i>`` is shifted by the chunk's first global sublayer."""
    entry = _routed_entry(name, routed)
    if entry is not None:
        prefix, start, rank = entry
        rest = name[len(prefix) + 1 :]
        mapped = _SEQUENTIAL_EXPERT.sub(lambda m: f"local_experts.{start + int(m.group(1))}.", rest)
        if mapped == rest:
            mapped = _GROUPED_EXPERT.sub(lambda m: f"{m.group(1)}{start + int(m.group(2))}", rest)
        if mapped == rest:
            mapped = f"{rest}@ep{rank}"
        name = f"{prefix}.{mapped}"
    if offset:
        name = _LAYER.sub(lambda m: f"decoder.layers.{int(m.group(1)) + offset}.", name)
    return name


def _checksum(tensor: torch.Tensor) -> Tuple[int, str, int, float, float]:
    """``(numel, dtype, exact sum of the raw element bits, float64 sum, float64 sum of squares)``;
    the bit sum is an integer sum (two's complement, order-independent), so it is equal exactly
    when the bits are."""
    data = tensor.detach().reshape(-1)
    bits_dtype = _BIT_VIEWS[data.element_size()]
    bits = int(data.view(bits_dtype).to(torch.int64).sum().item()) if data.numel() else 0
    values = data.to(torch.float64)
    return (
        data.numel(),
        str(tensor.dtype).replace("torch.", ""),
        bits,
        float(values.sum().item()),
        float((values * values).sum().item()),
    )


def loop_param_checksums(model) -> Dict[str, Tuple[int, str, int, float, float]]:
    """This rank's checksums of every parameter and persistent buffer of a hybrid loop model's
    chunks, keyed by global name (module docstring); empty for any other model. A name held by
    two chunks of this rank (the MTP stage's embedding copy at k > 1 is on another chunk than the
    embedding) keeps the first chunk's value, and the copy's under ``<name>#copy``."""
    checksums: Dict[str, Tuple[int, str, int, float, float]] = {}
    for hybrid in loop_hybrid_models(model):
        ring = getattr(hybrid.decoder, "loop_ring", None)
        offset = ring.offset if ring is not None else 0
        routed = _routed_experts(hybrid)
        for name, value in hybrid.state_dict(keep_vars=True).items():
            if not torch.is_tensor(value) or name.endswith("_extra_state"):
                continue
            key = _global_name(name, offset, routed)
            if key in checksums:
                key = f"{key}#copy"
            checksums[key] = _checksum(value)
    return checksums


def loop_param_checksum_lines(gathered: List[Dict[str, Tuple]]) -> List[str]:
    """The checksum lines of every rank's ``loop_param_checksums`` (sorted by name): one line
    per name whose holders agree, else one ``MISMATCH`` line per distinct value with its ranks.
    A ``#copy`` entry (a second holder on one rank) is merged into its name."""
    holders: Dict[str, Dict[Tuple, List[int]]] = {}
    for rank, checksums in enumerate(gathered):
        for key, value in (checksums or {}).items():
            name = key[: -len("#copy")] if key.endswith("#copy") else key
            holders.setdefault(name, {}).setdefault(tuple(value), []).append(rank)
    lines = []
    for name in sorted(holders):
        values = holders[name]
        for value, ranks in values.items():
            numel, dtype, bits, total, squares = value
            line = (
                f"loop param checksum {name}: numel {numel} dtype {dtype} bits {bits} "
                f"sum {total!r} sumsq {squares!r}"
            )
            if len(values) > 1:
                line += f" MISMATCH ranks {sorted(set(ranks))}"
            lines.append(line)
    return lines


def log_loop_param_checksums(model) -> List[str]:
    """Gather every rank's ``loop_param_checksums`` (a collective over the default group: every
    rank calls it) and print the lines (``loop_param_checksum_lines``) on rank 0; returns them on
    rank 0 (empty elsewhere and for a non-loop model)."""
    local = loop_param_checksums(model)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        gathered: List[Any] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if not any(gathered):
        return []
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return []
    lines = loop_param_checksum_lines(gathered)
    print_rank_0(f"loop param checksums: {len(lines)} lines follow")
    for line in lines:
        print_rank_0(line)
    return lines
