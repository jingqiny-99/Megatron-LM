# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The loop parameter store: canonical dense shards on the shard axis and their residency modes.

``LoopParamStore`` owns the *canonical shards* of the dense parameters of a hybrid loop model
(``HybridModel`` with a ``LoopHybridStack`` decoder) on the *shard axis*: ``z`` ranks of the RSBF
domain (the GTP-remat grid's ``gtp_remat`` group in training, the data-parallel group or a
subgroup in tests). Each eligible parameter is split along dim 0 (rows padded to a multiple of
``z``, GTP's layout) into one shard per rank; the shards are ``Parameter``s registered under the
store's *mirror* modules (``loop_param_store.mirror.<original name>``), tagged as GTP shards
(``is_gtp_weight_remat``, ``pad_length``, ``group``), so DDP over the replicate group, the
distributed optimizer, finalize's shard-axis SUM (which skips them) and the grad-norm dedup treat
them as dev's GTP shards. The original *compute* parameters are hidden from registration
(``param_store_binding``) and bound to full weights only around their use.

Parameters are grouped into *units* (the window unit: one attention + MLP / MoE layer pair, or one
sublayer; the loop adapter; each MTP depth's projection and its layer; the embedding; the output
layer) and units into *store groups* (``core``, ``prelude``, ``coda``, ``mtp``, ``emb_head``).
Each group has a residency mode (``StoreModes``):

``Resident``
    one full compute copy per unit, all-gathered at ``begin_step`` and bound for the whole step;
    a full fp32 accumulator per unit (the ``main_grad`` every item adds into), reduce-scattered
    into the shards at ``end_step``.
``Resident + sharded grads``
    the full compute copy as above; the gradient of each item goes into an fp32 scratch entry
    bound before the unit's backward and is reduce-scattered into the shards right after it.
``Layer window``
    only the shards are resident; the executor-ordered schedule all-gathers a unit into a window
    slot before its use (one gather ahead, in the order the item plan visits units) and releases
    it after (sweep) or after its backward (kept item units); gradients as for sharded grads.

Everything else (norms, mHC mappings, sinks, routers, experts, biases, small tensors) stays
replicated: experts are EP-resident and reduced by DDP as today, and finalize completes the
replicated dense gradients with the shard-axis SUM (``calculate_per_token_loss=True``).

The schedule of every RSBF sub-round type is static and identical on every rank
(``param_store_schedule``): the executor opens ``subround(kind, real=..., lookahead=...)``, wraps
each segment's forward in ``segment(seg)`` (the model's visit hooks do it) and each segment's
backward in ``backward_segment(seg)``; a null EU walks the same schedule through
``null_visitor(kind)``, so gathers are issued regardless of item validity and every rank issues
the same collectives in the same order. The collectives run on side streams (``param_store_comm``).
"""

import collections
import contextlib
import enum
import math
import re
import types
import warnings
import weakref
from dataclasses import dataclass, field
from typing import (
    Any,
    ContextManager,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
import torch.distributed as dist
from torch import nn

from megatron.core.transformer.loop.guards import is_block_boundary
from megatron.core.transformer.loop.param_store_binding import (
    MIRROR_FLAG_ATTR,
    RSBFDistributedWeight,
    StoreTensor,
    align_elements,
    bind_grad,
    bind_weight,
    check_segment_grads,
    classify_grad_capture,
    disable_local_wgrad_fusion,
    distributed_weight_unavailable_reason,
    is_local_fused_linear,
    is_te_linear,
    layout_tensors,
    make_store_tensor,
    named_logical_parameters,
    reset_uses,
    set_overwrite,
    shard_slice,
    te_distributed_weight_available,
    tensor_region,
    tensor_view,
    track_uses,
    unbind_grad,
    unbind_weight,
)
from megatron.core.transformer.loop.param_store_checkpoint import (
    install_load_hooks,
    logical_sharded_state_dict,
    store_checkpoint_layout,
)
from megatron.core.transformer.loop.param_store_comm import (
    COPY_ENGINE_EXPERIMENTAL_NOTE,
    GATHER_ENGINES,
    REDUCE_ENGINES,
    StoreComm,
)
from megatron.core.transformer.loop.param_store_schedule import (
    SegmentKey,
    SegmentOp,
    SubRoundSchedule,
    UnitEvent,
    build_subround_schedule,
)
from megatron.core.transformer.loop.program import block_boundaries_split_pairs, build_core_plan
from megatron.core.transformer.loop.rsbf_plan import SubRoundType

__all__ = [
    "STORE_GROUPS",
    "LoopParamStore",
    "LoopParamStoreConfig",
    "NullVisitor",
    "RSBFDistributedWeight",
    "ResidencyMode",
    "SegmentKey",
    "StoreGroups",
    "StoreMemoryReport",
    "StoreMemoryTerms",
    "StoreModes",
    "StoreStats",
    "StoreUnit",
    "copy_engine_bytes_estimate",
    "find_param_store",
    "named_logical_parameters",
    "store_variant_key",
    "te_distributed_weight_available",
    "unwrap_hybrid_model",
    "warn_copy_engines_experimental",
]

TRAIN_OUTSIDE_RSBF = (
    "the loop parameter store trains only through loop_schedule='rsbf': a grad-enabled visit "
    "outside an RSBF sub-round would leave the shard gradients unreduced"
)


class ResidencyMode(str, enum.Enum):
    """Where a store group's full weights and fp32 gradients live during a step."""

    RESIDENT = "resident"
    RESIDENT_SHARDED_GRADS = "resident_sharded_grads"
    LAYER_WINDOW = "layer_window"


_MODE_ALIASES = {
    "r": ResidencyMode.RESIDENT,
    "resident": ResidencyMode.RESIDENT,
    "rsg": ResidencyMode.RESIDENT_SHARDED_GRADS,
    "resident_sharded_grads": ResidencyMode.RESIDENT_SHARDED_GRADS,
    "lw": ResidencyMode.LAYER_WINDOW,
    "layer_window": ResidencyMode.LAYER_WINDOW,
}
_SWITCHES = {"on": True, "true": True, "1": True, "off": False, "false": False, "0": False}

STORE_GROUPS: Tuple[str, ...] = ("core", "prelude", "coda", "mtp", "emb_head")
EDGE_GROUPS: Tuple[str, ...] = ("prelude", "coda", "mtp")

R = ResidencyMode.RESIDENT
RSG = ResidencyMode.RESIDENT_SHARDED_GRADS
LW = ResidencyMode.LAYER_WINDOW


@dataclass(frozen=True)
class StoreModes:
    """The residency mode of each store group, and the re-gather switches.

    ``core_regather``: block / whole-round items re-gather each block unit for its backward (into
    its forward slot) instead of keeping it bound; ``edge_regather``: the same for the coda and
    prelude items (default: kept, one gather per edge unit per item).
    """

    core: ResidencyMode = R
    prelude: ResidencyMode = R
    coda: ResidencyMode = R
    mtp: ResidencyMode = R
    emb_head: ResidencyMode = R
    core_regather: bool = False
    edge_regather: bool = False

    def __post_init__(self):
        for group in STORE_GROUPS:
            object.__setattr__(self, group, ResidencyMode(getattr(self, group)))

    def validate(self) -> None:
        """Reject the modes the store does not support."""
        if self.emb_head is LW:
            raise ValueError(
                "store modes: emb_head=layer_window is not supported (the embedding and the "
                "output layer are resident; their gradients may be sharded: emb_head=rsg)"
            )

    def of(self, group: str) -> ResidencyMode:
        """The mode of store group ``group``."""
        if group not in STORE_GROUPS:
            raise ValueError(f"unknown store group {group!r} (one of {STORE_GROUPS})")
        return getattr(self, group)

    def describe(self) -> str:
        """``parse``-able text of these modes."""
        names = {R: "r", RSG: "rsg", LW: "lw"}
        parts = [f"{group}={names[self.of(group)]}" for group in STORE_GROUPS]
        parts.append(f"core_regather={'on' if self.core_regather else 'off'}")
        parts.append(f"edge_regather={'on' if self.edge_regather else 'off'}")
        return ",".join(parts)

    @staticmethod
    def parse(text: str) -> "StoreModes":
        """Parse ``"core=lw,edges=lw,emb_head=rsg,core_regather=on"``.

        Mode aliases ``r`` / ``rsg`` / ``lw`` (or the full names); ``edges`` sets prelude, coda
        and mtp; ``all`` sets every group; the switches take ``on`` / ``off``. Unnamed groups are
        Resident. ``"auto"`` is resolved by the trainer before this call.
        """
        values: Dict[str, Any] = {}
        text = (text or "").strip()
        if text.lower() == "auto":
            raise ValueError("store modes 'auto' are resolved by the trainer, not parsed")
        for item in filter(None, (part.strip() for part in text.split(","))):
            key, sep, value = item.partition("=")
            key, value = key.strip().lower(), value.strip().lower()
            if not sep or not value:
                raise ValueError(f"store modes: {item!r} is not key=value")
            if key in ("core_regather", "edge_regather"):
                if value not in _SWITCHES:
                    raise ValueError(f"store modes: {key} takes on / off, got {value!r}")
                values[key] = _SWITCHES[value]
                continue
            if value not in _MODE_ALIASES:
                raise ValueError(
                    f"store modes: unknown mode {value!r} for {key} (one of r, rsg, lw)"
                )
            groups = {"edges": EDGE_GROUPS, "all": STORE_GROUPS}.get(key, (key,))
            for group in groups:
                if group not in STORE_GROUPS:
                    raise ValueError(
                        f"store modes: unknown group {group!r} (one of "
                        f"{STORE_GROUPS + ('edges', 'all')})"
                    )
                values[group] = _MODE_ALIASES[value]
        modes = StoreModes(**values)
        modes.validate()
        return modes


@dataclass(frozen=True)
class LoopParamStoreConfig:
    """Settings of one store (training arguments; no ``TransformerConfig`` field).

    ``shard_size``: z, the shard axis size (1 = no store, except under ``allow_trivial_shard``).
    ``min_numel``: parameters with fewer elements stay replicated. ``binding``: ``"data"`` (the
    ``.data`` rebinding fallback, every TE) or ``"distributed_weight"`` (TE >= 2.19, TE linear
    weights only). ``separate_rs_group``: the reduce-scatters run on ``groups.shard_rs``.
    ``lookahead``: prefetch the next sub-round's first gather (and carry a unit across
    sub-rounds). Debug: ``timing`` (gather exposure events), ``check_visits`` (module pre-hooks
    assert every windowed module runs bound), ``poison_released`` (NaN-fill released slots and
    reduced scratch entries), ``verify_gathers`` (compare every window gather with a reference
    gathered at step begin), ``no_overwrite`` (never set ``overwrite_main_grad``: every target is
    zeroed and accumulated, the literal LW-vs-Resident gradient-bitwise case),
    ``record_calls`` (keep the host log of every collective), ``side_stream_delay_cycles``
    (CUDA: spin this many cycles on the all-gather / reduce-scatter stream before each
    collective, so a missing event wait reads poison or a stale buffer deterministically).
    ``allow_trivial_shard``: tests only, a ``z = 1`` store without collectives.
    ``align_bytes``: region alignment.
    ``gather_engine``: ``"nccl"`` (default) or ``"copy"``: the in-step all-gathers pull the
    peers' shards from a symmetric-memory mirror with copy-engine copies, so they take no SMs
    from the overlapped compute (CUDA, the shard group inside one NVLink domain; the mirror costs
    this rank's shard bytes once more, ``StoreMemoryReport.copy_engine_bytes``).
    ``copy_streams``: the copy engines' sub-streams per gather / reduce.
    ``reduce_engine``: ``"nccl"`` (default) or ``"copy"``: the reduce-scatters push each rank's
    rows into the owners' symmetric receive slots with copy-engine copies and sum the slots with
    one short kernel, so no long-lived NCCL kernel holds SMs during the backward (CUDA, one
    NVLink domain; ``z`` slots of the largest unit's fp32 shard, in ``copy_engine_bytes``).
    Deterministic and equal across store modes, but associated differently from NCCL's ring:
    equal to the NCCL engine within the fp32 association bound, not bitwise.
    The copy engines are **experimental** (``COPY_ENGINE_EXPERIMENTAL_NOTE``; a warning at store
    construction). ``copy_engine_timeout_ms``: their in-step symmetric-memory barrier timeout
    (0, the default: wait forever, like NCCL; a finite value turns a hang into a device-side
    trap that loses the CUDA context, for tests and benchmarks).
    """

    shard_size: int
    min_numel: int = 1 << 20
    binding: Literal["data", "distributed_weight"] = "data"
    separate_rs_group: bool = False
    lookahead: bool = True
    timing: bool = False
    check_visits: bool = False
    poison_released: bool = False
    verify_gathers: bool = False
    no_overwrite: bool = False
    record_calls: bool = False
    allow_trivial_shard: bool = False
    align_bytes: int = 256
    side_stream_delay_cycles: int = 0
    gather_engine: Literal["nccl", "copy"] = "nccl"
    copy_streams: int = 1
    reduce_engine: Literal["nccl", "copy"] = "nccl"
    copy_engine_timeout_ms: int = 0


def warn_copy_engines_experimental(config: LoopParamStoreConfig) -> bool:
    """Warn (``UserWarning``, ``COPY_ENGINE_EXPERIMENTAL_NOTE``) when ``config`` selects a copy
    engine; returns whether it did."""
    if config.gather_engine == "copy" or config.reduce_engine == "copy":
        warnings.warn(COPY_ENGINE_EXPERIMENTAL_NOTE, UserWarning, stacklevel=2)
        return True
    return False


def copy_engine_bytes_estimate(
    unit_tensor_numels: Sequence[Sequence[int]],
    dtype: torch.dtype,
    z: int,
    gather_engine: str,
    reduce_engine: str,
) -> int:
    """The copy engines' symmetric payload bytes on one rank before setup (pure): the gather
    mirror (every unit's tensors' shards, each at a 256-B aligned offset, ``dtype``) for
    ``gather_engine="copy"`` and the reduce-scatter receive slots (``z`` slots of the largest
    unit's aligned fp32 shard) for ``reduce_engine="copy"``; 0 for the NCCL engines.
    ``unit_tensor_numels``: per unit, its tensors' padded element counts (``numel_padded``)."""
    total = 0
    if gather_engine == "copy":
        mirror = 0
        for numels in unit_tensor_numels:
            for numel in numels:
                mirror = align_elements(mirror, dtype, 256)
                mirror += int(numel) // z
        total += max(mirror, 1) * _elem(dtype)
    if reduce_engine == "copy":
        slot = 0
        for numels in unit_tensor_numels:
            offset = 0
            for numel in numels:
                offset = align_elements(offset, torch.float32, 256)
                offset += int(numel) // z
            slot = max(slot, align_elements(offset, torch.float32, 256))
        total += max(z * max(slot, 1), 1) * 4
    return total


@dataclass(frozen=True)
class StoreGroups:
    """The process groups of a store: the shard axis inside the RSBF domain."""

    shard: Any  # z ranks: pg_collection.gtp_remat
    domain: Any  # the RSBF domain, W ranks: the full data-distribution group
    replicate: Any  # W / z ranks: pg_collection.dp (DDP / DistOpt)
    shard_rs: Any = None  # the shard's ranks on a second communicator (reduce-scatters)


@dataclass(eq=False)
class StoreUnit:
    """One window unit: the tensors bound and reduced together.

    Static: ``uid`` (discovery order, identical on every rank), ``key``, ``group``, the stack
    ``sublayers`` it covers, its owner modules (``roots``), its ``tensors`` laid out in one flat
    region (``numel_padded`` elements, aligned) and the shard elements. Runtime: the physical
    window slot, the gather's ready event, the Resident copy / accumulator, the bound gradient
    buffer.
    """

    uid: int
    key: str
    group: str
    sublayers: Tuple[int, ...]
    roots: Tuple[nn.Module, ...]
    tensors: Tuple[StoreTensor, ...]
    dtype: torch.dtype
    numel_padded: int
    shard_numel: int
    segments: Tuple[SegmentKey, ...] = ()
    # Runtime state.
    slot: Optional[int] = None
    ready: Any = None
    bound: bool = False
    copy: Optional[torch.Tensor] = None
    acc: Optional[torch.Tensor] = None
    grad_flat: Optional[torch.Tensor] = None
    grad_entry: int = -2  # -2: none, -1: transient, >= 0: pool entry

    @property
    def is_head_unit(self) -> bool:
        """Units visited only by the head segment (and the embedding): Megatron's fused linear
        may keep its forward-captured target there."""
        return self.group == "emb_head" or self.key.endswith(":eh_proj")


@dataclass
class StoreStats:
    """Counters of one step (reset at ``begin_step``)."""

    gathers: collections.Counter = field(default_factory=collections.Counter)
    reduces: collections.Counter = field(default_factory=collections.Counter)
    resident_reduces: collections.Counter = field(default_factory=collections.Counter)
    subrounds: collections.Counter = field(default_factory=collections.Counter)
    prefetches: int = 0
    carries: int = 0
    eager_gathers: int = 0
    gather_bytes: int = 0
    reduce_bytes: int = 0
    zeroed_targets: List[str] = field(default_factory=list)
    exposure_ms: List[float] = field(default_factory=list)
    # ``timing``: per pool-entry rebind, the compute stream's wait for the entry's previous
    # reduce-scatter (request to free event, ms; 0 when it had finished).
    scratch_wait_ms: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class StoreMemoryReport:
    """The store's memory on this rank, exact bytes."""

    shard_bytes: Mapping[str, int]  # per group: shard + fp32 shard grad (padding included)
    resident_copy_bytes: Mapping[str, int]
    accumulator_bytes: Mapping[str, int]
    slot_bytes: int
    slots: int
    scratch_bytes: int
    scratch_entries: int
    transient_scratch_peak: Mapping[str, int]  # per sub-round type
    hook_grad_peak: Mapping[str, int]  # per sub-round type: .grad of hook-class head tensors
    te_dummy_wgrad_bytes: int
    fusion_disabled: Tuple[str, ...]
    padding_bytes: int
    alignment_bytes: int
    # The copy engines' symmetric buffers (gather_engine="copy": the mirror; reduce_engine=
    # "copy": the receive slots; allocated at the first step, outside torch's caching allocator,
    # shared with later stores of the process), 0 for NCCL.
    copy_engine_bytes: int = 0

    def total_persistent(self) -> int:
        """Shards, copies, accumulators, slots, scratch, TE's dummies and the copy engine's
        mirror."""
        return (
            sum(self.shard_bytes.values())
            + sum(self.resident_copy_bytes.values())
            + sum(self.accumulator_bytes.values())
            + self.slot_bytes
            + self.scratch_bytes
            + self.te_dummy_wgrad_bytes
            + self.copy_engine_bytes
        )


@dataclass(frozen=True)
class StoreMemoryTerms:
    """What the planner's runtime memory model needs from the store (bytes / counts), for one
    set of modes and one core item plan (``modes`` / ``item_form`` / ``block_stride``).

    Per store group (``STORE_GROUPS``): ``group_modes`` (the ``ResidencyMode`` value),
    ``group_numel`` (the logical elements the store owns: the rest of the planner's group is
    replicated), ``group_copy_numel`` (the padded, aligned elements of the group's compute copy
    and accumulator) and ``group_shard_numel`` (this rank's shard elements). Buffers: ``slots``
    (K per sub-round type), ``slot_count`` x ``slot_numel`` window slots and
    ``scratch_entries`` x ``scratch_numel`` fp32 pool entries; the transient targets and the
    hook-class ``.grad`` of the coda / prelude items; ``elem_bytes`` of the stored dtype."""

    shard_axis: int
    unit_bytes_max: float
    adapter_unit: bool
    head_units: int
    slots: Mapping[str, int]
    scratch_entries: int
    head_scratch_coda: float  # transient targets of head-segment units (disjoint from the next)
    rsg_transient_coda: float  # transient targets of the other coda units
    rsg_transient_prelude: float
    hook_grad_coda: float
    hook_grad_prelude: float
    cast_units_kept: int
    te_dummy_wgrad: float
    pad_overhead: float
    modes: str = ""
    item_form: str = "whole_round"
    block_stride: Optional[int] = None
    elem_bytes: int = 2
    slot_count: int = 0
    slot_numel: int = 0
    scratch_numel: int = 0
    group_modes: Mapping[str, str] = field(default_factory=dict)
    group_numel: Mapping[str, int] = field(default_factory=dict)
    group_copy_numel: Mapping[str, int] = field(default_factory=dict)
    group_shard_numel: Mapping[str, int] = field(default_factory=dict)
    # The copy engines' symmetric buffers (``copy_engine_bytes_estimate`` for the store's
    # engines; 0 for NCCL): the planner charges them only when above 0.
    copy_engine_bytes: int = 0

    @property
    def buffer_bytes(self) -> int:
        """The persistent window slots and fp32 scratch pool."""
        return self.slot_count * self.slot_numel * self.elem_bytes + (
            self.scratch_entries * self.scratch_numel * 4
        )


def store_variant_key(modes: "StoreModes", item_form: str, block_stride: Optional[int]) -> str:
    """The key of one (modes, core item plan) variant in a table of ``StoreMemoryTerms``
    (``LoopParamStore.planner_terms``; the planner looks its options up by it)."""
    stride = "-" if block_stride is None else str(int(block_stride))
    return f"{modes.describe()}|{item_form}|{stride}"


@dataclass(frozen=True)
class _ModePlan:
    """Everything ``configure_modes`` derives from (modes, core plan): the per-unit modes, the
    schedules and the buffer sizes. Pure: computing one changes nothing in the store."""

    modes: "StoreModes"
    core_plan: Any
    unit_modes: Mapping[int, ResidencyMode]
    schedules: Mapping[SubRoundType, SubRoundSchedule]
    slot_numel: int
    scratch_numel: int
    slot_count: int
    scratch_count: int


class _Mirror(nn.Module):
    """An empty module of the mirror tree (holds canonical shards under the original names)."""

    def __init__(self):
        super().__init__()
        object.__setattr__(self, MIRROR_FLAG_ATTR, True)


@dataclass(eq=False)
class _Cursor:
    """The open sub-round: its schedule, the virtual -> physical slot map, the next op."""

    kind: SubRoundType
    schedule: SubRoundSchedule
    real: bool
    lookahead: Optional[SubRoundType]
    vmap: Dict[int, int]
    skip_first_gather: bool = False
    op: int = 0
    phase: str = "before"  # the null visitor's position within op ``op``
    open: Optional[SegmentOp] = None
    binds_done: int = 0
    binds_total: int = 0
    failed: bool = False


class NullVisitor:
    """A null EU's walk over a sub-round's segment ops: the same gathers, binds (waits only),
    releases and reduces (of zeros) as a real EU, without compute, so every communicator sees the
    same calls in the same order on every rank. ``before(k)`` / ``after(k)`` bracket op ``k``
    (the lockstep MoE null calls of that segment run in between); ``visit_all()`` walks the rest.
    """

    def __init__(self, store: "LoopParamStore", cursor: _Cursor):
        self._store = store
        self._cursor = cursor

    @property
    def ops(self) -> Tuple[SegmentOp, ...]:
        """The sub-round's segment ops."""
        return self._cursor.schedule.ops

    def before(self, k: int) -> None:
        """The events before op ``k``."""
        self._store._null_events(self._cursor, k, "before")  # pylint: disable=protected-access

    def after(self, k: int) -> None:
        """The events after op ``k``."""
        self._store._null_events(self._cursor, k, "after")  # pylint: disable=protected-access

    def visit(self, k: int) -> None:
        """Op ``k`` with nothing in between."""
        self.before(k)
        self.after(k)

    def visit_all(self) -> None:
        """Every remaining op."""
        for k in range(self._cursor.op, len(self.ops)):
            self.visit(k)


# ---------------------------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------------------------


def _is_ddp(module: nn.Module) -> bool:
    # pylint: disable=import-outside-toplevel
    from megatron.core.distributed import DistributedDataParallel

    return isinstance(
        module, (DistributedDataParallel, torch.nn.parallel.DistributedDataParallel)
    ) or hasattr(module, "ddp_config")


def _is_loop_hybrid(module: Any) -> bool:
    if not isinstance(module, nn.Module):
        return False
    decoder = module._modules.get("decoder")  # pylint: disable=protected-access
    return decoder is not None and hasattr(decoder, "loop_roles")


def unwrap_hybrid_model(model: Any, *, allow_ddp: bool = True) -> nn.Module:
    """The ``HybridModel`` (with a ``LoopHybridStack`` decoder) inside ``model``'s wrappers
    (``.module`` chain; a one-chunk list is accepted). ``allow_ddp=False`` raises on a DDP
    wrapper (the store is installed before the DDP wrap)."""
    if isinstance(model, (list, tuple)):
        if len(model) != 1:
            raise ValueError(f"the loop parameter store needs one model chunk, got {len(model)}")
        model = model[0]
    current = model
    while True:
        if not allow_ddp and isinstance(current, nn.Module) and _is_ddp(current):
            raise ValueError(
                "install the loop parameter store before the DDP wrap (DDP must bucket the "
                "canonical shards, not the full compute parameters)"
            )
        if _is_loop_hybrid(current):
            return current
        inner = getattr(current, "module", None)
        if inner is None or inner is current:
            raise ValueError(
                "no hybrid loop model (a HybridModel with a LoopHybridStack decoder) in "
                f"{type(model).__name__}"
            )
        current = inner


def find_param_store(model: Any) -> Optional["LoopParamStore"]:
    """The store installed on ``model`` (through DDP / Float16Module wrappers), or None."""
    try:
        hybrid = unwrap_hybrid_model(model)
    except ValueError:
        return None
    store = hybrid._modules.get("loop_param_store")  # pylint: disable=protected-access
    return store if isinstance(store, LoopParamStore) else None


_LAYER_NAME = re.compile(r"^decoder\.layers\.(\d+)\.")
_MTP_NAME = re.compile(r"^mtp\.layers\.(\d+)\.(.*)$")
# TE linear weights another consumer reads raw (absorbed MLA's up-projections): hook class.
_RAW_READ_OWNERS = ("linear_q_up_proj", "linear_kv_up_proj")


def _excluded_owner(modules: Sequence[nn.Module]) -> Optional[str]:
    """Why a parameter owned under ``modules`` (ancestors, owner last) stays replicated."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer.hyper_connection import HyperConnectionModule
    from megatron.core.transformer.moe.router import Router

    for module in modules:
        if isinstance(module, Router):
            return "router"
        if isinstance(module, HyperConnectionModule):
            return "mHC"
        if _is_norm_module(module):
            return "norm"
    return None


def _is_norm_module(module: nn.Module) -> bool:
    """A normalization module (``LayerNorm``, ``RMSNorm``, ``TENorm``'s TE norms, ``L2Norm``,
    ``LoopStreamRMSNorm``, ...): the class name *ends* in "norm". A substring test would also
    match fused norm + linear modules (``TELayerNormColumnParallelLinear``,
    ``TERMSNormDuplicatedLinear``), whose 2-D weights are the largest dense weights; their 1-D
    ``layer_norm_weight`` / ``layer_norm_bias`` are excluded by ``dim() < 2``."""
    return type(module).__name__.lower().endswith("norm")


# ---------------------------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------------------------


class LoopParamStore(nn.Module):
    """Canonical dense shards on the shard axis, bound to full weights per residency mode.

    Built by ``LoopParamStore.install(model, config, groups)`` (after the model is built and
    wrapped in ``Float16Module``, before the DDP wrap). Then ``configure_modes`` (after the DDP
    wrap is fine: the DDP layout does not depend on the modes), and per training step
    ``begin_step`` / ``subround`` + ``segment`` / ``backward_segment`` / ``null_visitor`` /
    ``end_step`` from the RSBF executor.
    """

    def __init__(self, hybrid: nn.Module, config: LoopParamStoreConfig, groups: StoreGroups):
        super().__init__()
        problems = self._validate(config, groups)
        if problems:
            raise ValueError("loop parameter store: " + "; ".join(problems))
        warn_copy_engines_experimental(config)
        object.__setattr__(self, "_hybrid_ref", weakref.ref(hybrid))
        self.config = config
        self.groups = groups
        self.shard_size = config.shard_size
        self.trivial = config.shard_size == 1
        self.shard_rank = 0 if self.trivial else dist.get_rank(group=groups.shard)
        self.mirror = _Mirror()
        self.units: Tuple[StoreUnit, ...] = ()
        self.modes: Optional[StoreModes] = None
        self.core_plan = None
        self.stats = StoreStats()
        self.fusion_disabled: Tuple[str, ...] = ()
        self.pair_core = True
        self.roles = None
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.comm: Optional[StoreComm] = None
        self._segment_units: Dict[SegmentKey, Tuple[int, ...]] = {}
        self._segment_tensors: Dict[SegmentKey, Tuple[StoreTensor, ...]] = {}
        self._unit_modes: Dict[int, ResidencyMode] = {}
        self._schedules: Dict[SubRoundType, SubRoundSchedule] = {}
        self.slot_count = 0
        self.slot_numel = 0
        self.scratch_count = 0
        self.scratch_numel = 0
        self._slots: List[torch.Tensor] = []
        self._slot_free: List[Any] = []
        self._slot_occupant: List[Optional[int]] = []
        self._slot_freed: List[int] = []
        self._scratch: List[torch.Tensor] = []
        self._scratch_free: List[Any] = []
        self._release_clock = 0
        self._cursor: Optional[_Cursor] = None
        self._prefetch: Optional[Tuple[SubRoundType, int, int]] = None
        self._carry: Optional[Tuple[SubRoundType, int, int]] = None
        self._in_step = False
        self._fresh = False
        self._exposure: List[Tuple[Any, Any]] = []
        self._scratch_exposure: List[Tuple[Any, Any]] = []
        self._verify_refs: Dict[int, torch.Tensor] = {}
        # The checkpoint writer election of the canonical shards: the rank in the replicate
        # group (None for the trivial test store: its entries keep the owners' replica ids).
        self.replicate_rank = None if self.trivial else dist.get_rank(group=groups.replicate)
        # Checkpoint state (``param_store_checkpoint``): the logical view's depth and root, the
        # dict keys of each stored tensor's entry, the row-shard merge outputs (factory sources
        # of the load pre-hook), what the last load consumed / missed, the load hooks.
        self._ckpt_view_depth = 0
        self._ckpt_root = None
        self._ckpt_entry_keys: Dict[str, Tuple[str, ...]] = {}
        # id -> merge output; entries vanish with their tensor (O(1) per merge, no id reuse).
        self._ckpt_merged: "weakref.WeakValueDictionary[int, torch.Tensor]" = (
            weakref.WeakValueDictionary()
        )
        self._ckpt_consumed: Dict[str, str] = {}
        self._ckpt_missing: Dict[str, List[str]] = {}
        self._ckpt_handles: List[Any] = []

    # -----------------------------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------------------------

    @staticmethod
    def _validate(config: LoopParamStoreConfig, groups: StoreGroups) -> List[str]:
        problems = []
        z = config.shard_size
        if z < 1:
            problems.append(f"shard_size must be >= 1, got {z}")
            return problems
        if z == 1 and not config.allow_trivial_shard:
            problems.append("shard_size 1 means no store (allow_trivial_shard is for tests)")
        if config.binding not in ("data", "distributed_weight"):
            problems.append(f"unknown binding {config.binding!r}")
        elif config.binding == "distributed_weight":
            reason = distributed_weight_unavailable_reason()
            if reason is not None:
                problems.append(reason)
        if config.min_numel < 0:
            problems.append(f"min_numel must be >= 0, got {config.min_numel}")
        if config.side_stream_delay_cycles < 0:
            problems.append("side_stream_delay_cycles must be >= 0")
        if config.gather_engine not in GATHER_ENGINES:
            problems.append(
                f"unknown gather_engine {config.gather_engine!r} (one of {GATHER_ENGINES})"
            )
        elif config.gather_engine == "copy" and z == 1:
            problems.append("gather_engine='copy' needs a shard axis z > 1")
        if config.reduce_engine not in REDUCE_ENGINES:
            problems.append(
                f"unknown reduce_engine {config.reduce_engine!r} (one of {REDUCE_ENGINES})"
            )
        elif config.reduce_engine == "copy" and z == 1:
            problems.append("reduce_engine='copy' needs a shard axis z > 1")
        if config.copy_streams < 1:
            problems.append("copy_streams must be >= 1")
        if config.copy_engine_timeout_ms < 0:
            problems.append(
                f"copy_engine_timeout_ms must be >= 0, got {config.copy_engine_timeout_ms}"
            )
        if groups.shard is None or groups.domain is None or groups.replicate is None:
            problems.append("the shard, domain and replicate groups are required")
            return problems
        shard = dist.get_process_group_ranks(groups.shard)
        domain = dist.get_process_group_ranks(groups.domain)
        replicate = dist.get_process_group_ranks(groups.replicate)
        if len(shard) != z:
            problems.append(f"shard_size {z} != the shard group's {len(shard)} ranks")
        if not set(shard) <= set(domain):
            problems.append(f"shard ranks {shard} are not inside the domain {domain}")
        if len(domain) != len(replicate) * len(shard):
            problems.append(
                f"domain size {len(domain)} != replicate size {len(replicate)} x shard size "
                f"{len(shard)}"
            )
        if groups.shard_rs is not None:
            if sorted(dist.get_process_group_ranks(groups.shard_rs)) != sorted(shard):
                problems.append("shard_rs must have the shard group's ranks")
        if config.separate_rs_group and groups.shard_rs is None:
            problems.append("separate_rs_group needs groups.shard_rs")
        return problems

    @staticmethod
    def install(model: Any, config: LoopParamStoreConfig, groups: StoreGroups) -> "LoopParamStore":
        """Install a store on ``model``'s ``HybridModel``: discover the units, slice the shards,
        hide the compute parameters, register ``hybrid.loop_param_store``. Raises when a store
        is already installed or ``model`` is DDP-wrapped."""
        hybrid = unwrap_hybrid_model(model, allow_ddp=False)
        if find_param_store(hybrid) is not None:
            raise ValueError("a loop parameter store is already installed on this model")
        for name, param in hybrid.named_parameters():
            if getattr(param, "is_gtp_weight_remat", False):
                raise ValueError(
                    f"{name} is a module-GTP shard: the store reuses the GTP grid only (build "
                    "the model with the gtp axes of its process group collection nulled)"
                )
        store = LoopParamStore(hybrid, config, groups)
        store._install(hybrid)  # pylint: disable=protected-access
        hybrid.loop_param_store = store
        object.__setattr__(hybrid, "_loop_store_hook", store)
        object.__setattr__(hybrid.decoder, "_loop_store_hook", store)
        install_load_hooks(store, hybrid)
        return store

    @property
    def hybrid(self) -> nn.Module:
        """The ``HybridModel`` this store is installed on."""
        hybrid = self._hybrid_ref()
        if hybrid is None:  # pragma: no cover - the model owns the store
            raise RuntimeError("the store's model is gone")
        return hybrid

    def _sublayer_units(self, roles, symbols: Sequence[str]) -> Dict[int, Tuple[str, Tuple]]:
        """Stack sublayer -> (store group, the sublayers of its unit)."""
        out = {}
        spans = (
            ("prelude", 0, roles.core_start, True),
            ("core", roles.core_start, roles.core_end, self.pair_core),
            ("coda", roles.core_end, len(symbols), True),
        )
        for group, lo, hi, pairs in spans:
            i = lo
            while i < hi:
                pair = (
                    pairs
                    and i + 1 < hi
                    and is_block_boundary(symbols, i)
                    and not is_block_boundary(symbols, i + 1)
                    and is_block_boundary(symbols, i + 2)
                )
                members = (i, i + 1) if pair else (i,)
                for member in members:
                    out[member] = (group, members)
                i += len(members)
        return out

    def _unit_of(self, name: str, sublayer_units: Mapping[int, Tuple[str, Tuple]]):
        """``(group, key, sublayers)`` of the unit a parameter name belongs to, or None."""
        match = _LAYER_NAME.match(name)
        if match:
            index = int(match.group(1))
            group, members = sublayer_units[index]
            return group, f"{group}:L{members[0]}", members
        if name.startswith("decoder.loop_adapter."):
            return "core", "core:adapter", ()
        match = _MTP_NAME.match(name)
        if match:
            depth, rest = match.group(1), match.group(2)
            part = "L" if rest.startswith("mtp_model_layer.") else "eh_proj"
            return "mtp", f"mtp:{depth}:{part}", ()
        if name.startswith("embedding."):
            return "emb_head", "emb_head:embedding", ()
        if name.startswith("output_layer."):
            return "emb_head", "emb_head:output", ()
        return None

    def _install(self, hybrid: nn.Module) -> None:
        config = self.config
        stack = hybrid.decoder
        self.roles = roles = stack.loop_roles
        symbols = tuple(stack.layer_type_list)
        model_config = hybrid.config
        try:
            core_plan = build_core_plan(model_config, roles)
            self.pair_core = not block_boundaries_split_pairs(core_plan, symbols, roles)
        except ValueError:
            self.pair_core = True
        sublayer_units = self._sublayer_units(roles, symbols)
        modules = dict(hybrid.named_modules())
        mtp_repeated = bool(getattr(model_config, "mtp_use_repeated_layer", False))
        params_dtype = getattr(model_config, "params_dtype", None)
        first = next(iter(hybrid.parameters()), None)
        self.device = first.device if first is not None else torch.device("cpu")

        # Discovery: registered parameters in module order, each once.
        found: Dict[str, List[Tuple[str, nn.Module, str, nn.Parameter, Tuple]]] = {}
        unit_info: Dict[str, Tuple[str, Tuple]] = {}
        seen = set()
        for module_name, module in hybrid.named_modules():
            for attr, param in list(module._parameters.items()):  # pylint: disable=W0212
                if param is None or id(param) in seen:
                    continue
                seen.add(id(param))
                name = f"{module_name}.{attr}" if module_name else attr
                unit = self._unit_of(name, sublayer_units)
                if unit is None or not self._eligible(param, module_name, modules):
                    continue
                if params_dtype is not None and param.dtype != params_dtype:
                    continue
                group, key, members = unit
                found.setdefault(key, []).append((name, module, attr, param, members))
                unit_info[key] = (group, members)
        dtypes = {entry[3].dtype for entries in found.values() for entry in entries}
        if len(dtypes) > 1:
            raise ValueError(
                f"loop parameter store: mixed parameter dtypes {sorted(map(str, dtypes))}"
            )
        if dtypes:
            self.dtype = dtypes.pop()

        has_mtp = getattr(hybrid, "mtp", None) is not None
        tied = bool(getattr(hybrid, "share_embeddings_and_output_weights", False))
        # A tied embedding read by a fused output layer: Megatron's fused linear captures
        # ``weight.main_grad`` at the head's forward, so the target is bound there (``forward``
        # class in the head segment); the embedding lookup's gradient still takes the hook.
        output_layer = getattr(hybrid, "output_layer", None)
        tied_fused_head = (
            tied and isinstance(output_layer, nn.Module) and is_local_fused_linear(output_layer)
        )
        units: List[StoreUnit] = []
        full: Dict[int, torch.Tensor] = {}
        disabled = []
        for uid, (key, entries) in enumerate(found.items()):
            group, members = unit_info[key]
            tensors = []
            head_unit = group == "emb_head" or key.endswith(":eh_proj")
            for name, module, attr, param, _ in entries:
                owner_attr = name.rsplit(".", 2)[-2] if name.count(".") >= 1 else ""
                parent = modules.get(name.rsplit(".", 2)[0]) if name.count(".") >= 2 else None
                raw_read = (
                    owner_attr in _RAW_READ_OWNERS
                    and parent is not None
                    and (
                        "mla" in type(parent).__name__.lower()
                        or "latent" in type(parent).__name__.lower()
                    )
                )
                if is_local_fused_linear(module, attr) and not head_unit:
                    disable_local_wgrad_fusion(module)
                    disabled.append(name.rsplit(".", 1)[0])
                capture = "hook" if raw_read else classify_grad_capture(module, attr)
                if tied_fused_head and key == "emb_head:embedding" and attr == "weight":
                    capture = "forward"
                if group in ("core", "prelude", "coda"):
                    single_use = key != "core:adapter" and not raw_read
                elif group == "mtp":
                    single_use = not mtp_repeated and not raw_read
                else:
                    single_use = False
                te_owned = is_te_linear(module)
                binding = (
                    config.binding if te_owned and capture == "lazy" and not raw_read else "data"
                )
                data = param.detach()
                tensor = make_store_tensor(
                    name,
                    module,
                    attr,
                    shard_size=self.shard_size,
                    binding=binding,
                    grad_capture=capture,
                    single_use=single_use,
                )
                full[id(tensor)] = data
                tensors.append(tensor)
            numel_padded = layout_tensors(tensors, config.align_bytes)
            segments: List[SegmentKey] = []
            if members:
                segments = [SegmentKey("sub", i) for i in members]
            elif key == "core:adapter":
                segments = [SegmentKey("adapter")]
            elif key == "emb_head:embedding":
                segments = [SegmentKey("embed")]
                if has_mtp or tied:
                    segments.append(SegmentKey("head"))
            else:
                segments = [SegmentKey("head")]
            units.append(
                StoreUnit(
                    uid=uid,
                    key=key,
                    group=group,
                    sublayers=tuple(members),
                    roots=tuple(dict.fromkeys(entry[1] for entry in entries)),
                    tensors=tuple(tensors),
                    dtype=self.dtype,
                    numel_padded=numel_padded,
                    shard_numel=sum(math.prod(t.shard_shape) for t in tensors),
                    segments=tuple(segments),
                )
            )
        self.units = tuple(units)
        self.fusion_disabled = tuple(dict.fromkeys(disabled))
        self._check_peer_weights(full)
        self._make_shards(full)
        del full
        for unit in self.units:
            for tensor in unit.tensors:
                if tensor.binding == "data":
                    unbind_weight(tensor)  # the full weights are dropped: only shards remain
                if tensor.grad_capture == "lazy" and tensor.single_use:
                    track_uses(tensor)
        segment_units: Dict[SegmentKey, List[int]] = {}
        segment_tensors: Dict[SegmentKey, List[StoreTensor]] = {}
        for unit in self.units:
            for seg in unit.segments:
                segment_units.setdefault(seg, []).append(unit.uid)
                for tensor in unit.tensors:
                    match = _LAYER_NAME.match(tensor.name)
                    if seg.kind != "sub" or (match and int(match.group(1)) == seg.index):
                        segment_tensors.setdefault(seg, []).append(tensor)
        self._segment_units = {seg: tuple(uids) for seg, uids in segment_units.items()}
        self._segment_tensors = {seg: tuple(ts) for seg, ts in segment_tensors.items()}
        if config.check_visits:
            self._install_visit_checks()
        self.comm = StoreComm(
            self.groups.shard,
            self.device,
            rs_group=self.groups.shard_rs if config.separate_rs_group else None,
            timing=config.timing,
            trivial=self.trivial,
            record_calls=config.record_calls,
            delay_cycles=config.side_stream_delay_cycles,
            gather_engine=config.gather_engine,
            copy_streams=config.copy_streams,
            reduce_engine=config.reduce_engine,
            copy_engine_timeout_ms=config.copy_engine_timeout_ms,
        )

    def _eligible(
        self, param: nn.Parameter, module_name: str, modules: Mapping[str, nn.Module]
    ) -> bool:
        """A dense parameter the store owns (else it stays replicated)."""
        if not param.requires_grad or not getattr(param, "allreduce", True):
            return False  # frozen, or an expert parameter (EP-resident, never gathered)
        if param.dim() < 2 or param.numel() < self.config.min_numel:
            return False
        if getattr(param, "keep_in_fp32", False) or getattr(param, "sequence_parallel", False):
            return False
        parts = module_name.split(".") if module_name else []
        ancestors = [modules[".".join(parts[:i])] for i in range(1, len(parts) + 1)]
        return _excluded_owner(ancestors) is None

    def _check_peer_weights(self, full: Mapping[int, torch.Tensor]) -> None:
        """Every shard peer must hold bitwise the same full weights before they are sliced: a
        position-weighted hash of each tensor's bit pattern, all-gathered and compared exactly."""
        if self.trivial or not full:
            return
        tensors = [t for unit in self.units for t in unit.tensors]
        hashes = [_weight_hash(full[id(tensor)]) for tensor in tensors]
        local = torch.stack(hashes).to(self.device)
        gathered = [torch.empty_like(local) for _ in range(self.shard_size)]
        dist.all_gather(gathered, local, group=self.groups.shard)
        for peer, other in enumerate(gathered):
            mismatch = (other != local).any(dim=1).nonzero().flatten().tolist()
            if mismatch:
                names = ", ".join(tensors[i].name for i in mismatch[:4])
                raise ValueError(
                    f"loop parameter store: shard peer {peer} holds different full weights for "
                    f"{names} (dense initialization must not depend on the data-parallel rank)"
                )

    def _make_shards(self, full: Mapping[int, torch.Tensor]) -> None:
        """Slice the canonical shards, tag them as GTP shards and register them on the mirror."""
        skip = {"main_grad", "overwrite_main_grad", "grad_added_to_main_grad", "loop_store_tensor"}
        for unit in self.units:
            for tensor in unit.tensors:
                data = shard_slice(full[id(tensor)], self.shard_size, self.shard_rank)
                shard = nn.Parameter(data.contiguous(), requires_grad=True)
                for key, value in tensor.compute.__dict__.items():
                    if key not in skip:
                        setattr(shard, key, value)
                shard.is_gtp_weight_remat = True
                shard.allreduce = getattr(tensor.compute, "allreduce", True)
                shard.pad_length = tensor.pad_length
                # GTP's logical / padded full shapes (read by the emerging optimizers for GTP
                # params); a plain shard has no ``group.size()``-derived properties.
                shard._unsharded_shape = tensor.logical_shape  # pylint: disable=W0212
                shard._unsharded_shape_padded = (  # pylint: disable=W0212
                    tensor.rows_padded,
                    *tensor.logical_shape[1:],
                )
                shard.group = self.groups.shard
                shard.gtp_replica_group = self.groups.replicate
                shard.loop_store_unit = unit.uid
                shard.loop_store_name = tensor.name
                tensor.shard = shard
                self._mirror_register(tensor.name, shard)

    def _mirror_register(self, name: str, shard: nn.Parameter) -> None:
        *path, leaf = name.split(".")
        module: nn.Module = self.mirror
        for part in path:
            child = module._modules.get(part)  # pylint: disable=protected-access
            if child is None:
                child = _Mirror()
                module.add_module(part, child)
            module = child
        module.register_parameter(leaf, shard)

    def _install_visit_checks(self) -> None:
        owners: Dict[int, Tuple[nn.Module, List[StoreTensor]]] = {}
        for unit in self.units:
            for tensor in unit.tensors:
                owners.setdefault(id(tensor.module), (tensor.module, []))[1].append(tensor)
        for module, tensors in owners.values():

            def check(_module, _args, tensors=tuple(tensors)):
                for tensor in tensors:
                    if tensor.bound is None:
                        raise RuntimeError(
                            f"loop param store: {tensor.name} is used while unbound (a module "
                            "ran outside its unit's window)"
                        )

            handle = module.register_forward_pre_hook(check)
            tensors[0].handles.append(handle)

    # -----------------------------------------------------------------------------------------
    # Modes and schedules
    # -----------------------------------------------------------------------------------------

    def mode_of(self, unit: StoreUnit) -> ResidencyMode:
        """The residency mode of ``unit`` (configured modes required)."""
        return self._unit_modes[unit.uid]

    def schedule(self, kind: SubRoundType) -> SubRoundSchedule:
        """The configured schedule of sub-round type ``kind``."""
        if not self._schedules:
            raise ValueError("configure the parameter store's modes first")
        return self._schedules[kind]

    def segment_units(self, seg: SegmentKey) -> Tuple[StoreUnit, ...]:
        """The units segment ``seg`` visits."""
        return tuple(self.units[uid] for uid in self._segment_units.get(seg, ()))

    @property
    def windows(self) -> bool:
        """True when the configured modes keep some unit in a Layer window (an eager visit of
        such a unit all-gathers it over the shard axis)."""
        return any(mode is LW for mode in self._unit_modes.values())

    @property
    def in_step(self) -> bool:
        """True between ``begin_step`` and ``end_step`` / ``abort_step``."""
        return self._in_step

    def visit(self, kind: str, index: int = -1) -> ContextManager[None]:
        """``segment(SegmentKey(kind, index))``: the forward visit the model's hooks open
        (``LoopHybridStack._run_range`` / ``_adapt``, ``HybridModel.embed`` / ``head_loss``),
        so the model code needs none of the store's types."""
        return self.segment(SegmentKey(kind, index))

    def configure_modes(self, modes: StoreModes, *, core_plan=None) -> None:
        """Set the residency modes and build every sub-round schedule (between steps).

        ``core_plan``: the core backward plan of the items (default: the config's). Pair units
        need a plan whose blocks do not split an attention sublayer from its MLP / MoE.
        """
        if self._in_step or self._cursor is not None:
            raise RuntimeError("store modes change only between steps")
        plan = self._plan_modes(modes, core_plan)
        self._unit_modes = dict(plan.unit_modes)
        self._schedules = dict(plan.schedules)
        self.slot_numel = plan.slot_numel
        self.scratch_numel = plan.scratch_numel
        self.slot_count = plan.slot_count
        self.scratch_count = plan.scratch_count
        self.modes = plan.modes
        self.core_plan = plan.core_plan
        self._set_overwrite_flags()

    def _current_plan(self) -> _ModePlan:
        if self.modes is None:
            raise ValueError("configure the parameter store's modes first")
        return _ModePlan(
            modes=self.modes,
            core_plan=self.core_plan,
            unit_modes=self._unit_modes,
            schedules=self._schedules,
            slot_numel=self.slot_numel,
            scratch_numel=self.scratch_numel,
            slot_count=self.slot_count,
            scratch_count=self.scratch_count,
        )

    def _plan_modes(self, modes: Any, core_plan=None) -> _ModePlan:
        """The per-unit modes, schedules and buffer sizes of ``modes`` under ``core_plan``
        (default: the config's), without changing the store. Raises ``ValueError`` when the
        plan splits the store's layer units or a mode is unsupported."""
        modes = StoreModes.parse(modes) if isinstance(modes, str) else modes
        modes.validate()
        hybrid = self.hybrid
        roles = self.roles
        symbols = tuple(hybrid.decoder.layer_type_list)
        if core_plan is None:
            core_plan = build_core_plan(hybrid.config, roles)
        split = block_boundaries_split_pairs(core_plan, symbols, roles)
        if (
            split
            and self.pair_core
            and any(u.group == "core" and len(u.sublayers) == 2 for u in self.units)
        ):
            raise ValueError(
                f"the core backward plan starts blocks at core sublayers {list(split)}, inside "
                "the store's layer units (use a pair-aligned item plan)"
            )
        unit_modes = {unit.uid: modes.of(unit.group) for unit in self.units}
        lw_units = [u for u in self.units if unit_modes[u.uid] is LW]
        grad_units = [u for u in self.units if unit_modes[u.uid] is not R]
        forward_units = {
            u.uid for u in grad_units if any(t.grad_capture == "forward" for t in u.tensors)
        }
        for uid in forward_units:
            if not self.units[uid].is_head_unit:  # pragma: no cover - install switched them
                raise ValueError(
                    f"{self.units[uid].key}: a forward-captured target outside the head"
                )
        slot_numel = max((u.numel_padded for u in lw_units), default=0)
        pooled = lw_units or [u for u in grad_units if u.group != "emb_head"]
        scratch_numel = max((u.numel_padded for u in pooled), default=0)
        oversize = {u.uid for u in grad_units if u.numel_padded > scratch_numel}
        schedules = {}
        for kind in SubRoundType:
            schedules[kind] = build_subround_schedule(
                kind,
                prelude=roles.prelude,
                core=roles.core,
                coda=roles.coda,
                segment_units=self._segment_units,
                weight_units=[u.uid for u in lw_units],
                grad_units=[u.uid for u in grad_units],
                forward_grad_units=forward_units,
                oversize_units=oversize,
                core_plan=core_plan,
                core_regather=modes.core_regather,
                edge_regather=modes.edge_regather,
            )
        return _ModePlan(
            modes=modes,
            core_plan=core_plan,
            unit_modes=unit_modes,
            schedules=schedules,
            slot_numel=slot_numel,
            scratch_numel=scratch_numel,
            slot_count=max(s.slots for s in schedules.values()),
            scratch_count=max(s.scratch_entries for s in schedules.values()),
        )

    def _set_overwrite_flags(self) -> None:
        for unit in self.units:
            sharded = self._unit_modes[unit.uid] is not R
            for tensor in unit.tensors:
                set_overwrite(
                    tensor,
                    sharded
                    and not self.config.no_overwrite
                    and tensor.grad_capture == "lazy"
                    and tensor.single_use,
                )

    # -----------------------------------------------------------------------------------------
    # Buffers
    # -----------------------------------------------------------------------------------------

    def _allocate(self) -> None:
        device, dtype = self.device, self.dtype
        self.comm.setup_copy_engine(self.units, dtype)  # once (copy engines only)
        poison = self.config.poison_released
        if len(self._slots) != self.slot_count or (
            self._slots and self._slots[0].numel() != self.slot_numel
        ):
            fill = float("nan") if poison else 0.0
            self._slots = [
                torch.full((self.slot_numel,), fill, dtype=dtype, device=device)
                for _ in range(self.slot_count)
            ]
            self._slot_free = [None] * self.slot_count
            self._slot_occupant = [None] * self.slot_count
            self._slot_freed = [p - self.slot_count for p in range(self.slot_count)]
        if len(self._scratch) != self.scratch_count or (
            self._scratch and self._scratch[0].numel() != self.scratch_numel
        ):
            fill = float("nan") if poison else 0.0
            self._scratch = [
                torch.full((self.scratch_numel,), fill, dtype=torch.float32, device=device)
                for _ in range(self.scratch_count)
            ]
            self._scratch_free = [None] * self.scratch_count
        for unit in self.units:
            mode = self._unit_modes[unit.uid]
            if mode is LW:
                for tensor in unit.tensors:
                    if tensor.bound is not None:
                        unbind_weight(tensor)
                unit.copy = None
                unit.bound = False
            elif unit.copy is None:
                unit.copy = torch.zeros(unit.numel_padded, dtype=dtype, device=device)
            if mode is R:
                if unit.acc is None:
                    unit.acc = torch.zeros(unit.numel_padded, dtype=torch.float32, device=device)
            else:
                unit.acc = None
                for tensor in unit.tensors:
                    unbind_grad(tensor)

    def _gather_copies(self, *, now: bool) -> None:
        """All-gather every Resident / RSG unit into its compute copy and bind it."""
        ready = None
        resident = [u for u in self.units if self._unit_modes[u.uid] is not LW]
        for unit in resident:
            if now:
                self.comm.gather_now(unit, unit.copy)
            else:
                ready = self.comm.gather(unit, unit.copy)
            self.stats.gathers[unit.key] += 1
            self.stats.gather_bytes += unit.shard_numel * self.shard_size * _elem(self.dtype)
        self.comm.wait(ready)  # the all-gather stream is in order: the last event covers all
        for unit in resident:
            for tensor in unit.tensors:
                bind_weight(tensor, tensor_view(unit.copy, tensor))
            unit.bound = True
        self._fresh = True

    # -----------------------------------------------------------------------------------------
    # The step
    # -----------------------------------------------------------------------------------------

    def begin_step(self) -> None:
        """Start a training step: buffers, the side streams' fence, the Resident / RSG gathers
        and binds, the Resident accumulators (zeroed, bound as ``main_grad``), the per-step
        ``overwrite_main_grad`` flags, fresh statistics."""
        if self.modes is None:
            raise ValueError(
                "configure the parameter store's modes (configure_modes) before the first step"
            )
        if self._in_step or self._cursor is not None:
            raise RuntimeError("store: begin_step inside a step (end_step / abort_step first)")
        self.stats = StoreStats()
        self._exposure = []
        self._scratch_exposure = []
        self._prefetch = self._carry = None
        self._allocate()
        self.comm.step_begin()
        self._gather_copies(now=False)
        for unit in self.units:
            if self._unit_modes[unit.uid] is R:
                for tensor in unit.tensors:
                    bind_grad(tensor, tensor_region(unit.acc, tensor, padded=True), zero=True)
        self._set_overwrite_flags()
        if self.config.verify_gathers:
            self._verify_refs = {}
            for unit in self.units:
                if self._unit_modes[unit.uid] is LW:
                    ref = torch.empty(unit.numel_padded, dtype=self.dtype, device=self.device)
                    self.comm.gather_now(unit, ref)
                    self._verify_refs[unit.uid] = ref
        self._in_step = True

    def end_step(self) -> None:
        """End a training step (after the last wave, before finalize): reduce-scatter every
        Resident accumulator into its shards, join the side streams, read the timing records,
        mark the compute copies stale (the optimizer updates the shards next)."""
        if self._cursor is not None:
            raise RuntimeError("store: end_step with a sub-round open")
        if not self._in_step:
            raise RuntimeError("store: end_step without begin_step")
        if self._carry is not None or self._prefetch is not None:
            self._drop_lookahead()  # before the check: a dropped lookahead is not a leak
        for unit in self.units:
            mode = self._unit_modes[unit.uid]
            if mode is LW and (
                unit.bound
                or unit.slot is not None
                or any(tensor.bound is not None for tensor in unit.tensors)
            ):
                raise RuntimeError(f"store: {unit.key} is still bound at end_step")
            if mode is not R and unit.grad_flat is not None:
                raise RuntimeError(f"store: {unit.key}'s gradient was not reduced")
        grad_ready = self.comm.event()
        for unit in self.units:
            if self._unit_modes[unit.uid] is R:
                self.comm.reduce(unit, unit.acc, wait_ready=grad_ready)
                self.stats.resident_reduces[unit.key] += 1
                self.stats.reduce_bytes += unit.numel_padded * 4
        self.comm.join()
        if self._exposure or self._scratch_exposure:
            torch.cuda.synchronize(self.device)
            for requested, ready in self._exposure:
                self.stats.exposure_ms.append(max(0.0, requested.elapsed_time(ready)))
            for requested, ready in self._scratch_exposure:
                self.stats.scratch_wait_ms.append(max(0.0, requested.elapsed_time(ready)))
            self._exposure = []
            self._scratch_exposure = []
        self._in_step = False
        self._fresh = False

    def abort_step(self) -> None:
        """Leave a step that raised: unbind every window unit and gradient target, drop the
        lookahead and transients, join the side streams. The next step starts clean."""
        self._cursor = None
        self._prefetch = self._carry = None
        for unit in self.units:
            mode = self._unit_modes.get(unit.uid, R)
            if mode is LW:
                for tensor in unit.tensors:
                    if tensor.bound is not None:
                        unbind_weight(tensor)
                unit.bound = False
            if mode is not R:
                for tensor in unit.tensors:
                    unbind_grad(tensor)
            unit.slot = None
            unit.ready = None
            unit.grad_flat = None
            unit.grad_entry = -2
        self._slot_occupant = [None] * len(self._slots)
        if self.comm is not None:
            self.comm.join()
        self._in_step = False

    def refresh(self) -> None:
        """Re-gather every Resident / RSG compute copy now (evaluation after an optimizer step)."""
        if self.modes is None:
            raise ValueError("configure the parameter store's modes first")
        if self._in_step:
            return  # the copies are fresh inside a step
        self._allocate()
        self._gather_copies(now=True)

    def _measure_unit_collectives(
        self, *, repeat: int = 10, unit_key: Optional[str] = None
    ) -> Dict[str, float]:
        """Diagnostic (benchmarks; not part of the training API): the standalone all-gather and
        reduce-scatter time of one unit with the store's engines (``StoreComm.
        _measure_unit_collectives``), between steps, after a step has set the buffers up.
        ``unit_key``: the unit (default: the largest core unit, else the largest unit). Gathers
        into a free window slot when one is large enough (else a temporary buffer) and
        reduce-scatters a scratch entry (else a zero buffer) into dropped outputs: no shard,
        ``main_grad``, statistic or call log changes. Returns ``{"gather_ms", "reduce_ms",
        "unit_bytes", "unit"}``."""
        if self._in_step or self._cursor is not None:
            raise RuntimeError("store: _measure_unit_collectives inside a step")
        if self.comm is None or not self.comm.cuda or self.trivial:
            raise RuntimeError("store: collectives are measured on CUDA with a shard axis z > 1")
        if not self.comm.copy_engines_ready:
            raise RuntimeError("store: run a step before measuring the copy engines")
        if unit_key is not None:
            matches = [unit for unit in self.units if unit.key == unit_key]
            if not matches:
                raise ValueError(f"store: no unit {unit_key!r}")
            unit = matches[0]
        else:
            core = [unit for unit in self.units if unit.group == "core"] or list(self.units)
            unit = max(core, key=lambda u: u.numel_padded)
        span = max(t.offset + t.numel_padded for t in unit.tensors)
        in_slot = bool(self._slots) and self.slot_numel >= span
        if in_slot:
            dest = self._slots[0]
        else:
            dest = torch.empty(max(span, unit.numel_padded), dtype=self.dtype, device=self.device)
        src = self._scratch[0] if self._scratch and self.scratch_numel >= span else None
        comm = self.comm
        result = dict(
            comm._measure_unit_collectives(  # pylint: disable=protected-access
                unit, dest, repeat=repeat, reduce_src=src
            )
        )
        if in_slot and self.config.poison_released:
            dest.fill_(float("nan"))  # a released slot stays poisoned
        result["unit"] = unit.key
        return result

    def _drop_lookahead(self) -> None:
        """Release a prefetched or carried unit that no sub-round used."""
        for entry in (self._prefetch, self._carry):
            if entry is None:
                continue
            _, uid, phys = entry
            unit = self.units[uid]
            self.comm.wait(unit.ready)
            self._release_unit(unit, phys)
        self._prefetch = self._carry = None

    # -----------------------------------------------------------------------------------------
    # Sub-rounds
    # -----------------------------------------------------------------------------------------

    @contextlib.contextmanager
    def subround(
        self, kind: SubRoundType, *, real: bool = True, lookahead: Optional[SubRoundType] = None
    ) -> Iterator[_Cursor]:
        """One lockstep sub-round of ``kind`` on this EU (``real``: an item runs; else a null EU
        walks ``null_visitor(kind)``). ``lookahead``: the type of the next sub-round of the same
        step list (its first gather is prefetched at this one's last bind, or a unit this one
        releases last is carried into it)."""
        if not self._in_step:
            raise RuntimeError("store: a sub-round outside begin_step / end_step")
        if self._cursor is not None:
            raise RuntimeError("store: a sub-round inside another sub-round")
        schedule = self.schedule(kind)
        if not self.config.lookahead:
            lookahead = None
        cursor = self._open_cursor(kind, schedule, real, lookahead)
        self._cursor = cursor
        self.stats.subrounds[(kind.value, "real" if real else "null")] += 1
        try:
            yield cursor
        except BaseException:
            cursor.failed = True
            self._cursor = None
            raise
        self._cursor = None
        if cursor.op != len(schedule.ops):
            nxt = schedule.ops[cursor.op] if cursor.op < len(schedule.ops) else None
            raise RuntimeError(
                f"store visit order: the {kind.value} sub-round ended after {cursor.op} of "
                f"{len(schedule.ops)} segment ops (next expected: {nxt})"
            )

    def _open_cursor(self, kind, schedule, real, lookahead) -> _Cursor:
        vmap: Dict[int, int] = {}
        skip = False
        entry = schedule.entry
        for name in ("_carry", "_prefetch"):
            held = getattr(self, name)
            if held is None:
                continue
            held_kind, uid, phys = held
            if held_kind is not kind or entry is None or entry[0] != uid:
                raise RuntimeError(
                    f"store: the previous sub-round looked ahead to {held_kind.value}, but a "
                    f"{kind.value} sub-round followed"
                )
            vmap[entry[1]] = phys
            skip = True
            setattr(self, name, None)
            if name == "_carry":
                self.stats.carries += 1
        free = sorted(
            (p for p in range(self.slot_count) if self._slot_occupant[p] is None),
            key=lambda p: (self._slot_freed[p], p),
        )
        for virtual in range(schedule.slots):
            if virtual not in vmap:
                if not free:  # pragma: no cover - slot_count is the max over the schedules
                    raise RuntimeError("store: no free window slot at a sub-round's start")
                vmap[virtual] = free.pop(0)
        return _Cursor(
            kind=kind,
            schedule=schedule,
            real=real,
            lookahead=lookahead,
            vmap=vmap,
            skip_first_gather=skip,
            binds_total=sum(1 for e in schedule.events if e.action == "bind"),
        )

    def null_visitor(self, kind: SubRoundType) -> NullVisitor:
        """The open null sub-round's visitor (``subround(kind, real=False)`` first)."""
        cursor = self._cursor
        if cursor is None or cursor.real or cursor.kind is not kind:
            raise RuntimeError(f"store: null_visitor({kind.value}) needs an open null sub-round")
        return NullVisitor(self, cursor)

    def _expect(self, cursor: _Cursor, direction: str, seg: SegmentKey) -> SegmentOp:
        ops = cursor.schedule.ops
        if cursor.op >= len(ops):
            raise RuntimeError(
                f"store visit order: the {cursor.kind.value} sub-round has no op left, got "
                f"{direction} {seg}"
            )
        op = ops[cursor.op]
        if op.direction != direction or op.segment != seg:
            raise RuntimeError(
                f"store visit order: expected {op}, got {direction} {seg} ({cursor.kind.value} "
                f"sub-round, op {cursor.op})"
            )
        return op

    @contextlib.contextmanager
    def segment(self, seg: SegmentKey) -> Iterator[None]:
        """A forward visit of segment ``seg`` (the model's visit hooks). Inside a sub-round it
        must be the schedule's next op; outside, an eager visit (no grad only)."""
        cursor = self._cursor
        if cursor is None:
            with self._eager_segment(seg):
                yield
            return
        if not cursor.real:
            raise RuntimeError(f"store: forward visit of {seg} inside a null sub-round")
        if cursor.open is not None:
            raise RuntimeError(f"store: segment {seg} visited inside {cursor.open}")
        op = self._expect(cursor, "fwd", seg)
        k = cursor.op
        self._run_events(cursor, k, "before")
        reset_uses(self._segment_tensors.get(seg, ()))
        cursor.open = op
        try:
            yield
        except BaseException:
            cursor.open = None
            cursor.failed = True
            raise
        cursor.open = None
        self._run_events(cursor, k, "after")
        cursor.op += 1

    @contextlib.contextmanager
    def backward_segment(self, seg: SegmentKey) -> Iterator[None]:
        """Around one segment's backward (the executor): binds the targets before it, checks
        the overwrite targets and reduce-scatters / releases after it."""
        cursor = self._cursor
        if cursor is None or not cursor.real:
            raise RuntimeError(f"store: backward of {seg} outside a real sub-round")
        if cursor.open is not None:
            raise RuntimeError(f"store: backward of {seg} inside {cursor.open}")
        op = self._expect(cursor, "bwd", seg)
        k = cursor.op
        self._run_events(cursor, k, "before")
        cursor.open = op
        try:
            yield
        except BaseException:
            cursor.open = None
            cursor.failed = True
            raise
        cursor.open = None
        zeroed = check_segment_grads(self._segment_tensors.get(seg, ()), seg)
        self.stats.zeroed_targets.extend(zeroed)
        self._run_events(cursor, k, "after")
        cursor.op += 1

    def _null_events(self, cursor: _Cursor, k: int, when: str) -> None:
        if self._cursor is not cursor:
            raise RuntimeError("store: a null visitor of a closed sub-round")
        expected = (cursor.op, cursor.phase)
        if (k, when) != expected:
            raise RuntimeError(
                f"store null visit order: expected {expected[1]} op {expected[0]}, got {when} "
                f"op {k}"
            )
        self._run_events(cursor, k, when)
        if when == "before":
            cursor.phase = "after"
        else:
            cursor.phase = "before"
            cursor.op += 1

    # -----------------------------------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------------------------------

    def _run_events(self, cursor: _Cursor, k: int, when: str) -> None:
        for event in cursor.schedule.at(k, when):
            handler = getattr(self, f"_on_{event.action}")
            handler(cursor, event)

    def _on_gather(self, cursor: _Cursor, event: UnitEvent) -> None:
        if cursor.skip_first_gather:
            cursor.skip_first_gather = False  # the first gather: prefetched or carried in
            return
        unit = self.units[event.unit]
        phys = cursor.vmap[event.slot]
        self._gather_into(unit, phys)

    def _gather_into(self, unit: StoreUnit, phys: int) -> None:
        if self._slot_occupant[phys] is not None:
            raise RuntimeError(
                f"store: window slot {phys} is held by {self.units[self._slot_occupant[phys]].key}"
                f" (gathering {unit.key})"
            )
        if unit.slot is not None:
            raise RuntimeError(f"store: {unit.key} is gathered twice")
        unit.ready = self.comm.gather(unit, self._slots[phys], wait_free=self._slot_free[phys])
        unit.slot = phys
        self._slot_occupant[phys] = unit.uid
        self.stats.gathers[unit.key] += 1
        self.stats.gather_bytes += unit.shard_numel * self.shard_size * _elem(self.dtype)

    def _on_bind(self, cursor: _Cursor, event: UnitEvent) -> None:
        unit = self.units[event.unit]
        phys = cursor.vmap[event.slot]
        if unit.slot != phys:
            raise RuntimeError(
                f"store: {unit.key} binds slot {phys} but was gathered into {unit.slot}"
            )
        if self.config.timing and self.comm.cuda and unit.ready is not None:
            self._exposure.append((self.comm.event(), unit.ready))
        self.comm.wait(unit.ready)
        slot = self._slots[phys]
        if self.config.verify_gathers:
            self._verify(unit, slot)
        if cursor.real:
            for tensor in unit.tensors:
                bind_weight(tensor, tensor_view(slot, tensor))
            unit.bound = True
        cursor.binds_done += 1
        if cursor.binds_done == cursor.binds_total and cursor.lookahead is not None:
            self._maybe_prefetch(cursor)

    def _verify(self, unit: StoreUnit, slot: torch.Tensor) -> None:
        ref = self._verify_refs.get(unit.uid)
        if ref is None:
            return
        for tensor in unit.tensors:
            if not torch.equal(tensor_view(slot, tensor), tensor_view(ref, tensor)):
                raise RuntimeError(f"store: the window gather of {tensor.name} is not the weight")

    def _maybe_prefetch(self, cursor: _Cursor) -> None:
        nxt = self._schedules[cursor.lookahead]
        if nxt.entry is None:
            return
        uid = nxt.entry[0]
        if uid in cursor.schedule.final_release:
            return  # carried instead (no second copy)
        unit = self.units[uid]
        if unit.slot is not None:
            return
        free = [p for p in range(self.slot_count) if self._slot_occupant[p] is None]
        if not free:
            return
        phys = min(free, key=lambda p: (self._slot_freed[p], p))
        self._gather_into(unit, phys)
        self._prefetch = (cursor.lookahead, uid, phys)
        self.stats.prefetches += 1

    def _on_release(self, cursor: _Cursor, event: UnitEvent) -> None:
        unit = self.units[event.unit]
        phys = unit.slot
        last = event.op_index == len(cursor.schedule.ops) - 1
        if last and cursor.lookahead is not None and self._prefetch is None:
            nxt = self._schedules[cursor.lookahead]
            if nxt.entry is not None and nxt.entry[0] == unit.uid:
                self._carry = (cursor.lookahead, unit.uid, phys)
                return
        self._release_unit(unit, phys)

    def _release_unit(self, unit: StoreUnit, phys: int) -> None:
        # Unbind whatever is bound, whether this EU's sub-round is real or null: a unit a real
        # sub-round carried into a null one is still bound from the real bind.
        for tensor in unit.tensors:
            if tensor.bound is not None:
                unbind_weight(tensor)
        if self.config.poison_released:
            self._slots[phys].narrow(0, 0, unit.numel_padded).fill_(float("nan"))
        self._slot_free[phys] = self.comm.event()
        self._slot_occupant[phys] = None
        self._slot_freed[phys] = self._release_clock
        self._release_clock += 1
        unit.slot = None
        unit.ready = None
        unit.bound = False

    def _on_bind_grad(self, cursor: _Cursor, event: UnitEvent) -> None:
        unit = self.units[event.unit]
        if unit.grad_flat is not None:
            raise RuntimeError(f"store: {unit.key}'s gradient target is bound twice")
        if event.scratch >= 0:
            free = self._scratch_free[event.scratch]
            if self.config.timing and self.comm.cuda and free is not None:
                self._scratch_exposure.append((self.comm.event(), free))
            self.comm.wait(free)
            flat = self._scratch[event.scratch]
        elif cursor.real:
            flat = torch.zeros(unit.numel_padded, dtype=torch.float32, device=self.device)
        else:
            flat = None  # the null reduce allocates its zeros on the reduce-scatter stream
        unit.grad_entry = event.scratch
        unit.grad_flat = flat
        if cursor.real:
            for tensor in unit.tensors:
                bind_grad(tensor, tensor_region(flat, tensor, padded=True))

    def _on_reduce(self, cursor: _Cursor, event: UnitEvent) -> None:
        unit = self.units[event.unit]
        entry = unit.grad_entry
        if cursor.real:
            free = self.comm.reduce(
                unit,
                unit.grad_flat,
                wait_ready=self.comm.event(),
                transient=entry < 0,
                poison=self.config.poison_released and entry >= 0,
            )
            for tensor in unit.tensors:
                unbind_grad(tensor)
        else:
            free = self.comm.zero_and_reduce(
                unit, unit.grad_flat, numel=unit.numel_padded, wait_ready=self.comm.event()
            )
        if entry >= 0:
            self._scratch_free[entry] = free
        unit.grad_flat = None
        unit.grad_entry = -2
        self.stats.reduces[(cursor.kind.value, unit.key)] += 1
        self.stats.reduce_bytes += unit.numel_padded * 4

    # -----------------------------------------------------------------------------------------
    # Eager visits (evaluation)
    # -----------------------------------------------------------------------------------------

    @contextlib.contextmanager
    def _eager_segment(self, seg: SegmentKey) -> Iterator[None]:
        if torch.is_grad_enabled():
            raise RuntimeError(TRAIN_OUTSIDE_RSBF)
        if self.modes is None:
            raise ValueError("configure the parameter store's modes before evaluation")
        units = self.segment_units(seg)
        windowed = [u for u in units if self._unit_modes[u.uid] is LW]
        if len(windowed) != len(units) and not self._fresh:
            raise RuntimeError(
                "store: the Resident compute copies are stale (an optimizer step updated the "
                "shards); call refresh() before evaluation"
            )
        buffers = []
        for unit in windowed:
            if unit.slot is not None:
                raise RuntimeError(f"store: eager visit of {unit.key} while it is in a window")
            flat = torch.empty(unit.numel_padded, dtype=self.dtype, device=self.device)
            self.comm.gather_now(unit, flat)
            self.stats.eager_gathers += 1
            buffers.append(flat)
            for tensor in unit.tensors:
                bind_weight(tensor, tensor_view(flat, tensor))
        try:
            yield
        finally:
            for unit in windowed:
                for tensor in unit.tensors:
                    unbind_weight(tensor)

    # -----------------------------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------------------------

    def bound_view(self, tensor: StoreTensor) -> torch.Tensor:
        """The full weight ``tensor`` is bound to (the DistributedWeight binding reads it)."""
        if tensor.bound is None:
            raise RuntimeError(f"loop param store: {tensor.name} is not bound")
        return tensor.bound

    def tensors(self) -> Iterator[StoreTensor]:
        """Every store tensor, in unit order."""
        for unit in self.units:
            yield from unit.tensors

    def memory_report(self, modes: Any = None, *, core_plan=None) -> StoreMemoryReport:
        """The store's exact memory on this rank under the configured modes, or under
        ``modes`` / ``core_plan`` (nothing in the store changes)."""
        plan = self._current_plan() if modes is None else self._plan_modes(modes, core_plan)
        return self._memory_report(plan)

    def _memory_report(self, plan: _ModePlan) -> StoreMemoryReport:
        elem = _elem(self.dtype)
        shard_bytes = collections.Counter()
        copies = collections.Counter()
        accs = collections.Counter()
        padding = alignment = 0
        for unit in self.units:
            mode = plan.unit_modes[unit.uid]
            shard_bytes[unit.group] += unit.shard_numel * (elem + 4)
            if mode is not LW:
                copies[unit.group] += unit.numel_padded * elem
            if mode is R:
                accs[unit.group] += unit.numel_padded * 4
            padding += sum(t.pad_length * t.row_elems for t in unit.tensors) * (elem + 4)
            alignment += (unit.numel_padded - sum(t.numel_padded for t in unit.tensors)) * elem
        transient = {}
        hook_grad = {}
        for kind, schedule in plan.schedules.items():
            transient[kind.value] = self._transient_peak(schedule)
            hook_grad[kind.value] = self._hook_grad_peak(schedule)
        return StoreMemoryReport(
            shard_bytes=dict(shard_bytes),
            resident_copy_bytes=dict(copies),
            accumulator_bytes=dict(accs),
            slot_bytes=plan.slot_count * plan.slot_numel * elem,
            slots=plan.slot_count,
            scratch_bytes=plan.scratch_count * plan.scratch_numel * 4,
            scratch_entries=plan.scratch_count,
            transient_scratch_peak=transient,
            hook_grad_peak=hook_grad,
            te_dummy_wgrad_bytes=self._te_dummy_bytes(),
            fusion_disabled=self.fusion_disabled,
            padding_bytes=padding // self.shard_size,  # the pad rows' shard bytes, per rank
            alignment_bytes=alignment,
            copy_engine_bytes=self._copy_engine_bytes(),
        )

    def _copy_engine_bytes(self) -> int:
        """The copy engines' symmetric buffers on this rank: the gather mirror (every unit's
        shards, 256-B aligned) and the reduce-scatter receive slots (``z`` slots of the largest
        unit's aligned fp32 shard). Once set up, the actual buffers (a buffer shared with an
        earlier, larger store of the process can exceed this store's layout); before,
        ``copy_engine_bytes_estimate``."""
        if self.config.gather_engine != "copy" and self.config.reduce_engine != "copy":
            return 0
        if self.comm is not None and self.comm.copy_engines_ready:
            return self.comm.copy_engine_bytes
        return self._copy_engine_estimate()

    def _copy_engine_estimate(self) -> int:
        """``copy_engine_bytes_estimate`` of this store's units and engines."""
        return copy_engine_bytes_estimate(
            [[tensor.numel_padded for tensor in unit.tensors] for unit in self.units],
            self.dtype,
            self.shard_size,
            self.config.gather_engine,
            self.config.reduce_engine,
        )

    def _transient_peak(self, schedule: SubRoundSchedule, head: Optional[bool] = None) -> int:
        """The peak of the transient (beyond the pool) fp32 gradient targets of ``schedule``;
        ``head``: only the units the head segment visits (True) or only the others (False), so
        the planner's ``head_scratch_coda`` and ``rsg_transient_coda`` are disjoint terms."""
        head_seg = SegmentKey("head")
        live = peak = 0
        for event in schedule.events:
            if event.scratch >= 0 or event.action not in ("bind_grad", "reduce"):
                continue
            unit = self.units[event.unit]
            if head is not None and (head_seg in unit.segments) != head:
                continue
            if event.action == "bind_grad":
                live += unit.numel_padded * 4
                peak = max(peak, live)
            else:
                live -= unit.numel_padded * 4
        return peak

    def _hook_grad_peak(self, schedule: SubRoundSchedule) -> int:
        """The bf16 ``.grad`` of hook-class embedding / head tensors live in a backward op."""
        peak = 0
        for op in schedule.ops:
            if op.direction != "bwd" or op.segment.kind not in ("embed", "head"):
                continue
            total = sum(
                t.numel * _elem(t.dtype)
                for unit in self.segment_units(op.segment)
                for t in unit.tensors
                if t.grad_capture == "hook"
            )
            peak = max(peak, total)
        return peak

    def _te_dummy_bytes(self) -> int:
        shapes = set()
        for tensor in self.tensors():
            if tensor.grad_capture == "lazy":
                shape = (
                    tensor.shard_shape
                    if tensor.binding == "distributed_weight"
                    else (tensor.logical_shape)
                )
                shapes.add((tuple(shape), tensor.dtype))
        return sum(math.prod(shape) * _elem(dtype) for shape, dtype in shapes)

    def memory_terms(
        self,
        modes: Any = None,
        *,
        core_plan=None,
        item_form: Optional[str] = None,
        block_stride: Optional[int] = None,
    ) -> StoreMemoryTerms:
        """The planner's runtime memory terms (logical shapes only, never ``p.shape``) under
        the configured modes and core plan, or under ``modes`` / ``core_plan`` (nothing in the
        store changes). ``item_form`` / ``block_stride``: the item-form label of the terms
        (default: read from the core plan)."""
        plan = self._current_plan() if modes is None else self._plan_modes(modes, core_plan)
        return self._memory_terms(plan, item_form=item_form, block_stride=block_stride)

    def _memory_terms(
        self, plan: _ModePlan, *, item_form: Optional[str], block_stride: Optional[int]
    ) -> StoreMemoryTerms:
        elem = _elem(self.dtype)
        eligible = [u for u in self.units if u.group != "emb_head"]
        report = self._memory_report(plan)
        coda = plan.schedules[SubRoundType.CODA]
        prelude = plan.schedules[SubRoundType.PRELUDE_B]
        head_lw = [u for u in self.units if u.group == "mtp" and plan.unit_modes[u.uid] is LW]
        total = sum(u.shard_numel * self.shard_size for u in self.units) or 1
        logical = sum(t.numel for t in self.tensors()) or 1
        kept = max((s.kept_units_max() for s in plan.schedules.values()), default=0)
        if item_form is None:
            item_form = plan.core_plan.form
            block_stride = None if item_form == "whole_round" else plan.core_plan.stride
        group_numel = collections.Counter()
        group_copy = collections.Counter()
        group_shard = collections.Counter()
        for unit in self.units:
            group_numel[unit.group] += sum(t.numel for t in unit.tensors)
            group_copy[unit.group] += unit.numel_padded
            group_shard[unit.group] += unit.shard_numel
        return StoreMemoryTerms(
            shard_axis=self.shard_size,
            unit_bytes_max=float(max((u.numel_padded for u in eligible), default=0) * elem),
            adapter_unit=any(u.key == "core:adapter" for u in self.units),
            head_units=len(head_lw),
            slots={kind.value: s.slots for kind, s in plan.schedules.items()},
            scratch_entries=plan.scratch_count,
            head_scratch_coda=float(self._transient_peak(coda, head=True)),
            rsg_transient_coda=float(self._transient_peak(coda, head=False)),
            rsg_transient_prelude=float(self._transient_peak(prelude)),
            hook_grad_coda=float(report.hook_grad_peak[SubRoundType.CODA.value]),
            hook_grad_prelude=float(report.hook_grad_peak[SubRoundType.PRELUDE_B.value]),
            cast_units_kept=kept,
            te_dummy_wgrad=float(report.te_dummy_wgrad_bytes),
            pad_overhead=total / logical - 1.0,
            modes=plan.modes.describe(),
            item_form=item_form,
            block_stride=block_stride,
            elem_bytes=elem,
            slot_count=plan.slot_count,
            slot_numel=plan.slot_numel,
            scratch_numel=plan.scratch_numel,
            group_modes={group: plan.modes.of(group).value for group in STORE_GROUPS},
            group_numel={group: group_numel.get(group, 0) for group in STORE_GROUPS},
            group_copy_numel={group: group_copy.get(group, 0) for group in STORE_GROUPS},
            group_shard_numel={group: group_shard.get(group, 0) for group in STORE_GROUPS},
            copy_engine_bytes=self._copy_engine_estimate(),
        )

    def planner_terms(
        self, variants: Sequence[Tuple["StoreModes", str, Optional[int]]]
    ) -> Dict[str, Any]:
        """``store_variant_key(modes, item_form, block_stride) -> StoreMemoryTerms`` for every
        ``(modes, item_form, block_stride)`` of ``variants`` (the planner's menu), or the
        reason string when the store cannot run that variant (a core plan that splits its layer
        units, an item form the core rejects). Nothing in the store changes."""
        out: Dict[str, Any] = {}
        for modes, item_form, block_stride in variants:
            key = store_variant_key(modes, item_form, block_stride)
            if key in out:
                continue
            try:
                core_plan = build_core_plan(
                    types.SimpleNamespace(
                        loop_rsbf_item_form=item_form, loop_rsbf_block_stride=block_stride
                    ),
                    self.roles,
                )
                out[key] = self._memory_terms(
                    self._plan_modes(modes, core_plan),
                    item_form=item_form,
                    block_stride=block_stride,
                )
            except ValueError as error:
                out[key] = f"the parameter store cannot run it ({error})"
        return out

    # -----------------------------------------------------------------------------------------
    # Checkpoints (M3c, ``param_store_checkpoint``)
    # -----------------------------------------------------------------------------------------

    @property
    def in_checkpoint_view(self) -> bool:
        """True while the owners build checkpoint entries in the logical view."""
        return self._ckpt_view_depth > 0

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets=(), metadata=None
    ):  # pylint: disable=unused-argument
        """``{}`` inside the logical view: the canonical shards are saved under their owners'
        logical keys (``HybridModel.sharded_state_dict``), never as mirror entries. Raises
        outside it."""
        if self.in_checkpoint_view:
            return {}
        raise RuntimeError(
            "the loop parameter store's shards are saved through HybridModel.sharded_state_dict "
            "(logical keys); do not call it on the store"
        )

    def logical_sharded_state_dict(self, root: nn.Module, produce, *, prefix: str = ""):
        """``produce()`` (``root``'s owners' sharded state dict) in the logical view, the stored
        tensors' entries converted to row-shard factories (``param_store_checkpoint``)."""
        return logical_sharded_state_dict(self, root, produce, prefix=prefix)

    def checkpoint_layout(self) -> Dict[str, Any]:
        """``store_checkpoint_layout``: z, ``min_numel``, binding, dtype, the stored set."""
        return store_checkpoint_layout(self)

    def on_checkpoint_load(self) -> None:
        """After the canonical shards were loaded (the load post-hook): every compute copy is
        unbound (the data binding's ``.data`` back on the NaN placeholder) and marked stale, so
        nothing reads a pre-load weight; the next ``begin_step`` (or ``refresh()`` before
        evaluation) gathers and binds the loaded shards. Copies, slots and scratch stay
        allocated; the copy engines' mirror is refreshed at every step begin. Raises inside a
        step."""
        if self._in_step or self._cursor is not None:
            raise RuntimeError("store: checkpoint load inside a training step")
        for unit in self.units:
            for tensor in unit.tensors:
                if tensor.bound is not None:
                    unbind_weight(tensor)
            unit.bound = False
        self._fresh = False
        self._verify_refs = {}

    def _mark_merged(self, tensor: torch.Tensor) -> None:
        """Record a row-shard merge output (a factory source for the load pre-hook)."""
        self._ckpt_merged[id(tensor)] = tensor

    def _is_merged(self, tensor: Any) -> bool:
        return self._ckpt_merged.get(id(tensor)) is tensor


_INT_OF = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
_HASH_CHUNK = 1 << 22  # elements per chunk: the int64 temporaries stay ~100 MB per tensor


def _weight_hash(full: torch.Tensor, chunk: int = _HASH_CHUNK) -> torch.Tensor:
    """Two position-weighted int64 sums of ``full``'s bit pattern, ``[2]``, accumulated chunk by
    chunk (the transient int64 widening is bounded by ``chunk``, not by the tensor)."""
    data = full.detach().contiguous().reshape(-1)
    bits_all = data.view(_INT_OF.get(data.element_size(), torch.int16))
    total = torch.zeros(2, dtype=torch.int64, device=data.device)
    for start in range(0, bits_all.numel(), chunk):
        bits = bits_all[start : start + chunk].to(torch.int64)
        position = torch.arange(start, start + bits.numel(), dtype=torch.int64, device=bits.device)
        total[0] += (bits * (position % 1000003 + 1)).sum()
        total[1] += (bits * (position % 999983 + 7) * (position % 13 + 1)).sum()
    return total


def _elem(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()
