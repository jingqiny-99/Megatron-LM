# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Round-input checkpoints of the RSBF executor.

The executor's forward sweep of round ``r`` of sample ``i`` produces the round state ``s_{i,r}``
and the exit-criterion state ``crit_{i,r}``; the prelude sweep produces the injection ``e_i`` and
``crit_{i,0}``. The backward items replay each round from these checkpoints (on fresh leaves),
so every entry lives from its sweep to the item that consumes it last:

=====================================  ==========================  ===========================
Entry                                  Written by                  Freed by
=====================================  ==========================  ===========================
``e_i``                                PRELUDE sweep               PRELUDE item
``crit_{i,0}``                         PRELUDE sweep               CORE(i, 1) item
``s_{i,r}``, ``crit_{i,r}`` (r < d_i)  CORE sweep of round r       CORE(i, r + 1) item
``s_{i,d}``, ``crit_{i,d}``            CORE sweep of round d_i     CODA item
``x_{i,r,l}`` (block input)            CORE sweep of round r       CORE(i, r), after the
                                                                   backward of block ``l``
``trace_{i,r,l}`` (check only)         CORE sweep of round r       CORE(i, r), after its check
=====================================  ==========================  ===========================

Block inputs (kind ``"block"``) are the block-checkpoint item's stash: the input of core
sublayer ``l`` (core-relative, ``1 <= l < L``) of round ``r``, captured by the sweep. Trace
entries (kind ``"trace"``, ``check_recompute`` only) are the sweep's output of every core
sublayer ``0 <= l < L``, compared with the item's recompute. Both have the round state's shape
and dtype; a trace entry and a block entry may share one storage (the output of sublayer
``l - 1`` is the input of sublayer ``l``), which is counted once.

``LoopStateStore`` holds them for one wave (one RSBF pass) with explicit frees: every entry is
written once, read any number of times and released exactly once, either freed or taken
(``take_state`` moves ownership to the caller, e.g. a chain record that compares the sweep's
state with the recompute). Misuse (a second write, a read or release of a missing, freed or
taken entry, an in-place change of a stored tensor, a wave that ends with live entries) raises
``StateStoreError`` with the key.

Multi-EU (one store per EU; keys are wave-global sample ids): an entry that leaves this EU by
a move is released as ``"sent"`` (``send_state`` / ``send_crit``, after the transfer
completed), and a copy that arrives by a move is installed with ``put_received_*``. A received
install is accepted only when the key is absent here and was never released this wave, or was
released as ``"sent"`` (a checkpoint that went away and comes back by a later move); an entry
that was consumed (``"taken"``) or evicted (``"freed"``) can never come back. Origin writes
(``put_*``) stay strictly write-once.

Every entry is in HBM. The placement policy (``StatePlacement``), the Grace tier and
``prefetch`` are interfaces that later milestones fill (M3 / M6a); the store already reports
the per-wave live-byte peak per tier and per kind.
"""

import enum
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, Optional, Protocol

import torch
from torch import Tensor

EntryKind = Literal["state", "crit", "injection", "block", "trace"]
# Kinds keyed by a core sublayer as well as a (sample, round), shaped like the round state.
SUBLAYER_KINDS = ("block", "trace")


class StateTier(enum.Enum):
    """Where an entry lives."""

    HBM = "hbm"
    # Interface only in M1: host memory over C2C (Grace), with prefetch before the consumer.
    GRACE = "grace"


class StatePlacement(Protocol):
    """Chooses the tier of each entry when it is written."""

    def tier(self, sample: int, r: int, kind: EntryKind, sublayer: int = -1) -> StateTier:
        """The tier of entry ``kind`` of ``sample`` at round ``r`` (0 for the injection);
        ``sublayer`` is passed (as a keyword) for the per-sublayer kinds ``"block"`` and
        ``"trace"`` only, so a policy without it serves every other kind."""


class HBMOnlyPlacement:
    """Every entry in HBM (M1)."""

    def tier(self, sample: int, r: int, kind: EntryKind, sublayer: int = -1) -> StateTier:
        """Always ``StateTier.HBM``."""
        del sample, r, kind, sublayer
        return StateTier.HBM


class StoreKey(NamedTuple):
    """One entry: ``kind`` of wave-local ``sample`` at round ``r`` (the injection uses 0);
    ``sublayer`` is the core-relative sublayer of a ``"block"`` / ``"trace"`` entry (-1
    otherwise)."""

    kind: EntryKind
    sample: int
    r: int
    sublayer: int = -1

    def __str__(self) -> str:
        if self.kind == "injection":
            return f"e[sample {self.sample}]"
        if self.kind in SUBLAYER_KINDS:
            symbol = "x" if self.kind == "block" else "trace"
            return f"{symbol}[sample {self.sample}, round {self.r}, sublayer {self.sublayer}]"
        symbol = "s" if self.kind == "state" else "crit"
        return f"{symbol}[sample {self.sample}, round {self.r}]"


@dataclass(frozen=True)
class StoreStats:
    """Live entries and bytes now, and the live-byte high-water mark of the current (or, from
    ``end_wave``, the finished) wave. Bytes count each underlying storage once; per kind, a
    storage's bytes count for the kind of its earliest-put live entry (re-attributed when that
    entry is released while another still holds the storage)."""

    live_entries: int
    live_bytes: int
    peak_bytes: int
    peak_bytes_by_tier: dict
    wave: Optional[int] = None
    live_bytes_by_kind: dict = field(default_factory=dict)
    peak_bytes_by_kind: dict = field(default_factory=dict)


class StateStoreError(RuntimeError):
    """A violation of the store's write-once / release-once discipline."""


@dataclass
class _Entry:
    tensor: Tensor
    tier: StateTier
    # The storage identity recorded at put (None: no bytes), so the release finds the same
    # bookkeeping even if the storage is resized or reallocated in between (offload paths).
    storage: Optional[tuple]
    # The tensor's version counter at put (None for an inference tensor, which has none).
    version: Optional[int]


@dataclass
class _Storage:
    """One live storage: the live entries on it (put order), its bytes, its tier and the kind
    its bytes are attributed to (that of ``keys[0]``)."""

    keys: list
    nbytes: int
    tier: StateTier


def _normalize_device(device) -> torch.device:
    """``device`` with an explicit index where tensors report one: an unindexed ``cuda`` becomes
    the current CUDA device (tensors allocated on "cuda" report ``cuda:<current>``), and ``cpu``
    drops any index (CPU tensors report plain ``cpu``)."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    if device.type == "cpu":
        return torch.device("cpu")
    return device


def _version(tensor: Tensor) -> Optional[int]:
    """The tensor's in-place version counter; None for an inference tensor (no counter)."""
    if tensor.is_inference():
        return None
    return tensor._version  # pylint: disable=protected-access


class LoopStateStore:
    """Round-input checkpoints ``s_r`` and ``crit_r`` per (sample, round), plus each sample's
    injection ``e``, for one wave at a time.

    Every entry is written once (by a sweep) and released explicitly by the item that consumes
    it last. Entries are stored detached (``put_*`` stores ``tensor.detach()``), so no autograd
    graph is kept alive by the store. The detach shares storage (and the version counter) with
    the caller's tensor: an in-place change of that tensor after the put would silently change
    the checkpoint, so every read and release checks the version counter and raises
    ``StateStoreError`` naming the key.

    Stored tensors must not alias long-lived buffers (module parameters or buffers, cached
    constants): with ``poison_on_free`` the release of the last entry on a storage fills it
    with NaN, and the NaN would then surface far from the store.

    Args:
        placement: the tier policy; M1 supports only HBM (any other tier raises
            ``NotImplementedError``).
        poison_on_free: fill a freed floating tensor with NaN before dropping it (tests), so a
            stale alias read after the free shows up. A storage still referenced by another
            live entry, or taken by a caller this wave, is never poisoned.
        act_dtype: the residual-stream dtype of states and injections (the ``"act"`` port);
            None accepts any floating dtype. Criterion states are always fp32.
        device: when given, every entry must be on it. An unindexed ``"cuda"`` means the
            current CUDA device at construction.
        core_sublayers: ``L``, the core sublayers of the loop; when given, block-input
            sublayers must be in ``1 .. L-1`` and trace sublayers in ``0 .. L-1`` (otherwise
            only the lower bounds are checked).
    """

    def __init__(
        self,
        *,
        placement: Optional[StatePlacement] = None,
        poison_on_free: bool = False,
        act_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        core_sublayers: Optional[int] = None,
    ):
        self.placement = placement if placement is not None else HBMOnlyPlacement()
        self.poison_on_free = poison_on_free
        self.act_dtype = act_dtype
        self.device = None if device is None else _normalize_device(device)
        self.core_sublayers = core_sublayers
        self._entries: dict[StoreKey, _Entry] = {}
        # Keys released this wave, "freed", "taken" or "sent" (error messages; write-once per
        # wave, except a "sent" key may be received again).
        self._released: dict[StoreKey, str] = {}
        # Per sample: the first [S, c, nH] put this wave (by a state or the injection) and the
        # key that set it; every later state, injection and crit row count is checked against it.
        self._shapes: dict[int, tuple[torch.Size, StoreKey]] = {}
        # Per sample: the first crit row count c put this wave, and its key.
        self._crit_rows: dict[int, tuple[int, StoreKey]] = {}
        # Live storages: id -> the live entries on it, its bytes, tier and attributed kind.
        self._storages: dict[tuple, _Storage] = {}
        # Storages taken by a caller this wave: never poisoned through another entry.
        self._taken_storages: set[tuple] = set()
        self._wave: Optional[int] = None
        self._live_bytes = {tier: 0 for tier in StateTier}
        self._peak_bytes = {tier: 0 for tier in StateTier}
        self._peak_total = 0
        # Live bytes per entry kind (each storage counted for one kind) and their peaks.
        self._live_kind_bytes: dict[str, int] = {}
        self._peak_kind_bytes: dict[str, int] = {}

    # -----------------------------------------------------------------------------------
    # Waves
    # -----------------------------------------------------------------------------------

    @property
    def wave(self) -> Optional[int]:
        """The open wave, None between waves."""
        return self._wave

    def begin_wave(self, wave: int) -> None:
        """Open wave ``wave``; the store must be empty and no wave open."""
        if self._wave is not None:
            raise StateStoreError(f"wave {wave} begins while wave {self._wave} is open")
        if self._entries:
            raise StateStoreError(f"wave {wave} begins with live entries: {self._describe_live()}")
        self._wave = wave
        self._released.clear()
        self._shapes.clear()
        self._crit_rows.clear()
        self._taken_storages.clear()
        self._peak_bytes = {tier: 0 for tier in StateTier}
        self._peak_total = 0
        self._peak_kind_bytes = dict(self._live_kind_bytes)

    def end_wave(self) -> StoreStats:
        """Close the open wave; every entry must have been released. Returns the wave's stats
        (its live-byte peaks)."""
        if self._wave is None:
            raise StateStoreError("end_wave without an open wave")
        if self._entries:
            raise StateStoreError(
                f"wave {self._wave} ends with {len(self._entries)} live entries: "
                f"{self._describe_live()}"
            )
        stats = self.stats()
        self._wave = None
        self._released.clear()
        self._shapes.clear()
        self._crit_rows.clear()
        self._taken_storages.clear()
        return stats

    def reset(self) -> None:
        """Drop every entry without poisoning and close the wave (error recovery)."""
        self._entries.clear()
        self._released.clear()
        self._shapes.clear()
        self._crit_rows.clear()
        self._storages.clear()
        self._taken_storages.clear()
        self._wave = None
        self._live_bytes = {tier: 0 for tier in StateTier}
        self._live_kind_bytes = {}

    def stats(self) -> StoreStats:
        """The current live entries and bytes and the open wave's peaks."""
        return StoreStats(
            live_entries=len(self._entries),
            live_bytes=sum(self._live_bytes.values()),
            peak_bytes=self._peak_total,
            peak_bytes_by_tier=dict(self._peak_bytes),
            wave=self._wave,
            live_bytes_by_kind={k: v for k, v in self._live_kind_bytes.items() if v},
            peak_bytes_by_kind=dict(self._peak_kind_bytes),
        )

    # -----------------------------------------------------------------------------------
    # Round states s_r (r >= 1; s_0 is the zero state built by the core cell)
    # -----------------------------------------------------------------------------------

    def put_state(self, sample: int, r: int, state: Tensor) -> None:
        """Checkpoint ``s_r`` ``[S, c, nH]`` of ``sample`` (``r >= 1``)."""
        self._check_round(r, minimum=1, what="state")
        self._put(StoreKey("state", sample, r), state)

    def get_state(self, sample: int, r: int) -> Optional[Tensor]:
        """``s_r`` of ``sample``; None for ``r = 0`` (the core cell builds the zero state)."""
        self._check_round(r, minimum=0, what="state")
        if r == 0:
            return None
        return self._get(StoreKey("state", sample, r))

    def free_state(self, sample: int, r: int) -> None:
        """Release ``s_r`` of ``sample`` (``r >= 1``)."""
        self._check_round(r, minimum=1, what="state")
        self._release(StoreKey("state", sample, r), take=False)

    def take_state(self, sample: int, r: int) -> Tensor:
        """Remove ``s_r`` of ``sample`` and hand it to the caller (``r >= 1``): released like a
        free, but never poisoned (the caller owns it now)."""
        self._check_round(r, minimum=1, what="state")
        return self._release(StoreKey("state", sample, r), take=True)

    # -----------------------------------------------------------------------------------
    # Exit-criterion states crit_r (r >= 0)
    # -----------------------------------------------------------------------------------

    def put_crit(self, sample: int, r: int, crit: Tensor) -> None:
        """Checkpoint ``crit_r`` ``[c, k]`` fp32 of ``sample`` (``r >= 0``). With ``k = 0`` the
        entry holds no bytes but keeps its lifetime checks."""
        self._check_round(r, minimum=0, what="crit")
        self._put(StoreKey("crit", sample, r), crit)

    def get_crit(self, sample: int, r: int) -> Tensor:
        """``crit_r`` of ``sample``."""
        self._check_round(r, minimum=0, what="crit")
        return self._get(StoreKey("crit", sample, r))

    def free_crit(self, sample: int, r: int) -> None:
        """Release ``crit_r`` of ``sample``."""
        self._check_round(r, minimum=0, what="crit")
        self._release(StoreKey("crit", sample, r), take=False)

    # -----------------------------------------------------------------------------------
    # Injections e
    # -----------------------------------------------------------------------------------

    def put_injection(self, sample: int, injection: Tensor) -> None:
        """Store the injection ``e`` ``[S, c, nH]`` of ``sample``."""
        self._put(StoreKey("injection", sample, 0), injection)

    def get_injection(self, sample: int) -> Tensor:
        """The injection of ``sample``."""
        return self._get(StoreKey("injection", sample, 0))

    def free_injection(self, sample: int) -> None:
        """Release the injection of ``sample``."""
        self._release(StoreKey("injection", sample, 0), take=False)

    def holds_injection(self, sample: int) -> bool:
        """Whether this store holds a live injection copy of ``sample``."""
        return StoreKey("injection", sample, 0) in self._entries

    # -----------------------------------------------------------------------------------
    # Block inputs x_{r,l} and sweep traces (per core sublayer)
    # -----------------------------------------------------------------------------------

    def put_block_input(
        self, sample: int, r: int, sublayer: int, x: Tensor, *, kind: EntryKind = "block"
    ) -> None:
        """Store the sweep's input of core sublayer ``sublayer`` of round ``r`` (``kind``
        ``"block"``, the block-checkpoint stash) or its output (``kind`` ``"trace"``,
        ``check_recompute``), ``[S, c, nH]``."""
        self._put(self._sublayer_key(kind, sample, r, sublayer), x)

    def get_block_input(
        self, sample: int, r: int, sublayer: int, *, kind: EntryKind = "block"
    ) -> Tensor:
        """The stored entry of ``sublayer`` of round ``r``."""
        return self._get(self._sublayer_key(kind, sample, r, sublayer))

    def free_block_input(
        self, sample: int, r: int, sublayer: int, *, kind: EntryKind = "block"
    ) -> None:
        """Release the entry of ``sublayer`` of round ``r`` (after its block's backward)."""
        self._release(self._sublayer_key(kind, sample, r, sublayer), take=False)

    def take_block_input(
        self, sample: int, r: int, sublayer: int, *, kind: EntryKind = "block"
    ) -> Tensor:
        """Remove the entry of ``sublayer`` of round ``r`` and hand it to the caller (never
        poisoned)."""
        return self._release(self._sublayer_key(kind, sample, r, sublayer), take=True)

    # -----------------------------------------------------------------------------------
    # Moves between EUs (multi-EU RSBF)
    # -----------------------------------------------------------------------------------

    def send_state(self, sample: int, r: int) -> None:
        """Release ``s_r`` of ``sample`` after it left this EU by a move (kind ``"sent"``).
        Call after the transfer completed; the bytes are poisoned like a free."""
        self._check_round(r, minimum=1, what="state")
        self._release(StoreKey("state", sample, r), take=False, kind="sent")

    def send_crit(self, sample: int, r: int) -> None:
        """Release ``crit_r`` of ``sample`` after it left this EU by a move."""
        self._check_round(r, minimum=0, what="crit")
        self._release(StoreKey("crit", sample, r), take=False, kind="sent")

    def put_received_state(self, sample: int, r: int, state: Tensor) -> None:
        """Install ``s_r`` of ``sample`` that arrived by a move."""
        self._check_round(r, minimum=1, what="state")
        self._put(StoreKey("state", sample, r), state, received=True)

    def put_received_crit(self, sample: int, r: int, crit: Tensor) -> None:
        """Install ``crit_r`` of ``sample`` that arrived by a move."""
        self._check_round(r, minimum=0, what="crit")
        self._put(StoreKey("crit", sample, r), crit, received=True)

    def put_received_injection(self, sample: int, injection: Tensor) -> None:
        """Install a copy of the injection of ``sample`` that arrived by a move."""
        self._put(StoreKey("injection", sample, 0), injection, received=True)

    def send_block_input(
        self, sample: int, r: int, sublayer: int, *, kind: EntryKind = "block"
    ) -> None:
        """Release a block-input (or trace) entry after it left this EU with a moved CORE
        item."""
        self._release(self._sublayer_key(kind, sample, r, sublayer), take=False, kind="sent")

    def put_received_block_input(
        self, sample: int, r: int, sublayer: int, x: Tensor, *, kind: EntryKind = "block"
    ) -> None:
        """Install a block-input (or trace) entry that arrived with a moved CORE item."""
        self._put(self._sublayer_key(kind, sample, r, sublayer), x, received=True)

    # -----------------------------------------------------------------------------------
    # Placement interface
    # -----------------------------------------------------------------------------------

    def prefetch(self, sample: int, r: int, kind: EntryKind = "state", sublayer: int = -1) -> None:
        """Bring entry ``kind`` of ``sample`` at round ``r`` (and ``sublayer`` for a block /
        trace entry) to HBM before its consumer runs (``r`` is ignored for the injection). A
        no-op while every entry is in HBM; checks that the entry is live (``s_0`` needs
        nothing)."""
        if kind == "injection":
            self._get(StoreKey("injection", sample, 0))
            return
        if kind in SUBLAYER_KINDS:
            self._get(self._sublayer_key(kind, sample, r, sublayer))
            return
        if kind not in ("state", "crit"):
            raise StateStoreError(f"prefetch: unknown entry kind {kind!r}")
        self._check_round(r, minimum=0, what=kind)
        if kind == "state" and r == 0:
            return
        self._get(StoreKey(kind, sample, r))

    # -----------------------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------------------

    @staticmethod
    def _check_round(r: int, *, minimum: int, what: str) -> None:
        if r < minimum:
            raise StateStoreError(f"{what} round must be >= {minimum}, got {r}")

    def _sublayer_key(self, kind: str, sample: int, r: int, sublayer: int) -> StoreKey:
        """The key of a block-input (``1 <= sublayer < L``) or trace (``0 <= sublayer < L``)
        entry of round ``r >= 1``."""
        if kind not in SUBLAYER_KINDS:
            raise StateStoreError(f"not a per-sublayer entry kind: {kind!r}")
        self._check_round(r, minimum=1, what=kind)
        low = 1 if kind == "block" else 0
        high = self.core_sublayers
        if sublayer < low or (high is not None and sublayer >= high):
            bound = "L" if high is None else str(high)
            raise StateStoreError(
                f"{kind} sublayer must be in {low} .. {bound} - 1 (core-relative), got {sublayer}"
            )
        return StoreKey(kind, sample, r, sublayer)

    def _validate(self, key: StoreKey, tensor: Tensor) -> None:
        if not isinstance(tensor, Tensor):
            raise StateStoreError(f"{key}: expected a tensor, got {type(tensor).__name__}")
        if self.device is not None and _normalize_device(tensor.device) != self.device:
            raise StateStoreError(f"{key}: device {tensor.device} is not {self.device}")
        first = self._shapes.get(key.sample)
        rows = self._crit_rows.get(key.sample)
        if key.kind == "crit":
            if tensor.dtype != torch.float32:
                raise StateStoreError(f"{key}: dtype {tensor.dtype} is not fp32")
            if tensor.dim() != 2:
                raise StateStoreError(f"{key}: shape {tuple(tensor.shape)} is not [c, k]")
            if first is not None and tensor.shape[0] != first[0][1]:
                raise StateStoreError(
                    f"{key}: {tensor.shape[0]} rows, {first[1]} has {first[0][1]}"
                )
            if rows is not None and tensor.shape[0] != rows[0]:
                raise StateStoreError(f"{key}: {tensor.shape[0]} rows, {rows[1]} has {rows[0]}")
            return
        if self.act_dtype is not None and tensor.dtype != self.act_dtype:
            raise StateStoreError(f"{key}: dtype {tensor.dtype} is not {self.act_dtype}")
        if self.act_dtype is None and not tensor.dtype.is_floating_point:
            raise StateStoreError(f"{key}: dtype {tensor.dtype} is not floating")
        if tensor.dim() != 3:
            raise StateStoreError(f"{key}: shape {tuple(tensor.shape)} is not [S, c, nH]")
        if first is not None and tensor.shape != first[0]:
            raise StateStoreError(
                f"{key}: shape {tuple(tensor.shape)} is not {tuple(first[0])} of {first[1]}"
            )
        if rows is not None and tensor.shape[1] != rows[0]:
            raise StateStoreError(f"{key}: {tensor.shape[1]} rows, {rows[1]} has {rows[0]}")

    def _put(self, key: StoreKey, tensor: Tensor, *, received: bool = False) -> None:
        if self._wave is None:
            raise StateStoreError(f"{key}: put outside a wave (call begin_wave first)")
        if key in self._entries:
            raise StateStoreError(f"{key}: already stored (an entry is written once)")
        released = self._released.get(key)
        if released is not None and not (received and released == "sent"):
            raise StateStoreError(
                f"{key}: already {released} this wave (an entry is written once"
                + ("; only an entry sent away by a move may come back)" if received else ")")
            )
        self._validate(key, tensor)
        stored = tensor.detach()
        if stored.requires_grad:  # pragma: no cover - detach() never requires grad
            raise StateStoreError(f"{key}: the stored tensor requires grad")
        if key.kind in SUBLAYER_KINDS:
            tier = self.placement.tier(key.sample, key.r, key.kind, sublayer=key.sublayer)
        else:
            tier = self.placement.tier(key.sample, key.r, key.kind)
        if tier is not StateTier.HBM:
            raise NotImplementedError(f"{key}: tier {tier} is an interface only (M1 is HBM only)")
        storage = self._storage_id(stored)
        self._entries[key] = _Entry(
            tensor=stored, tier=tier, storage=storage, version=_version(stored)
        )
        if key.kind == "crit":
            self._crit_rows.setdefault(key.sample, (stored.shape[0], key))
        else:
            self._shapes.setdefault(key.sample, (stored.shape, key))
        if storage is not None:
            ref = self._storages.get(storage)
            if ref is None:
                nbytes = stored.untyped_storage().nbytes()
                self._storages[storage] = _Storage(keys=[key], nbytes=nbytes, tier=tier)
                self._live_bytes[tier] += nbytes
                self._peak_bytes[tier] = max(self._peak_bytes[tier], self._live_bytes[tier])
                self._peak_total = max(self._peak_total, sum(self._live_bytes.values()))
                self._attribute(key.kind, nbytes)
            else:
                ref.keys.append(key)

    def _get(self, key: StoreKey) -> Tensor:
        entry = self._entries.get(key)
        if entry is None:
            raise StateStoreError(f"{key}: {self._missing(key)}")
        self._check_version(key, entry)
        return entry.tensor

    @staticmethod
    def _check_version(key: StoreKey, entry: _Entry) -> None:
        if entry.version is None:
            return
        version = _version(entry.tensor)
        if version != entry.version:
            raise StateStoreError(
                f"{key}: modified in place after the put (version {entry.version} -> {version}); "
                "the checkpoint no longer holds the sweep's value"
            )

    def _release(
        self, key: StoreKey, *, take: bool, kind: Optional[str] = None
    ) -> Optional[Tensor]:
        entry = self._entries.get(key)
        if entry is None:
            action = "take" if take else ("send" if kind == "sent" else "free")
            raise StateStoreError(f"{key}: cannot {action}, {self._missing(key)}")
        self._check_version(key, entry)
        del self._entries[key]
        self._released[key] = kind or ("taken" if take else "freed")
        tensor = entry.tensor
        storage = entry.storage
        last = True
        if storage is not None:
            ref = self._storages[storage]
            owner = ref.keys[0].kind
            ref.keys.remove(key)
            last = not ref.keys
            if last:
                del self._storages[storage]
                self._live_bytes[ref.tier] -= ref.nbytes
                self._attribute(owner, -ref.nbytes)
            elif ref.keys[0].kind != owner:
                # The earliest-put live entry changed kind: its kind owns the bytes now.
                self._attribute(owner, -ref.nbytes)
                self._attribute(ref.keys[0].kind, ref.nbytes)
            if take:
                self._taken_storages.add(storage)
        if take:
            return tensor
        if (
            self.poison_on_free
            and last
            and storage is not None
            and storage not in self._taken_storages
            and tensor.dtype.is_floating_point
            # Only the bytes recorded at put (not a storage resized or reallocated since).
            and self._storage_id(tensor) == storage
        ):
            with torch.no_grad():
                tensor.fill_(float("nan"))
        return None

    def _attribute(self, kind: str, nbytes: int) -> None:
        """Add ``nbytes`` (negative: remove) to ``kind``'s live bytes and its peak."""
        live = self._live_kind_bytes.get(kind, 0) + nbytes
        self._live_kind_bytes[kind] = live
        self._peak_kind_bytes[kind] = max(self._peak_kind_bytes.get(kind, 0), live)

    @staticmethod
    def _storage_id(tensor: Tensor) -> Optional[tuple]:
        """Identity of the tensor's storage; None for an empty storage (holds no bytes)."""
        storage = tensor.untyped_storage()
        if storage.nbytes() == 0:
            return None
        return (tensor.device, storage.data_ptr())

    def _missing(self, key: StoreKey) -> str:
        released = self._released.get(key)
        if released == "freed":
            return "already freed (use after free or double free)"
        if released == "taken":
            return "already taken (ownership moved to the caller)"
        if released == "sent":
            return "sent to another EU (it left this EU by a move)"
        return "never stored"

    def _describe_live(self, limit: int = 8) -> str:
        keys = sorted(self._entries, key=lambda k: (k.sample, k.r, k.kind, k.sublayer))
        text = ", ".join(str(key) for key in keys[:limit])
        if len(keys) > limit:
            text += f", ... ({len(keys) - limit} more)"
        return text
