# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The collective engine of the loop parameter store: dim-0 all-gathers and reduce-scatters.

``StoreComm`` gathers a unit's canonical shards into a flat unit buffer (a window slot or a
Resident compute copy) and reduce-scatters a flat fp32 gradient buffer (a scratch entry, a
transient or a Resident accumulator) into the shards' ``main_grad``. Every tensor of a unit is one
``all_gather_into_tensor`` / ``reduce_scatter_tensor`` over its padded region (rows padded to a
multiple of the shard size, GTP's dim-0 layout); on CUDA the calls of one unit are coalesced.

Streams and events (CUDA). Gathers run on a dedicated all-gather stream, reduce-scatters on a
dedicated reduce-scatter stream. ``gather`` waits the destination's ``free`` event (recorded on the
compute stream after the destination's last use) and returns a ``ready`` event the compute stream
waits before it binds the unit. ``reduce`` waits a ``grad_ready`` event (recorded on the compute
stream after the unit's backward), allocates its outputs on the reduce-scatter stream, adds them to
``shard.main_grad`` there, and returns a ``free`` event after which the source may be rewritten.
A transient source is handed to the reduce-scatter stream with ``record_stream``. ``step_begin``
makes both side streams wait the compute stream once per step (``zero_grad_buffer`` precedes
every add); ``join`` makes the compute stream wait both (before finalize reads ``main_grad``).

The AG and RS use one communicator (``groups.shard``) unless a second one with the same ranks is
given (``groups.shard_rs``): torch's per-group NCCL stream then serializes them, which keeps the
issue order safe. On CPU (Gloo) every call is synchronous and the events are None. A trivial
shard axis (``z = 1``, tests only) runs no collective: the gather copies, the reduce adds.

Copy engines (CUDA, one NVLink domain per shard group; **experimental** and opt-in,
``COPY_ENGINE_EXPERIMENTAL_NOTE``). ``gather_engine="copy"`` pulls the in-step gathers from a
symmetric-memory mirror of the shards (``_CopyEngineGather``); ``reduce_engine="copy"`` pushes
each rank's rows of a gradient into the owners' symmetric receive slots and sums the slots with
one short kernel (``_CopyEngineReduce``). Both move the bytes with ``cudaMemcpyAsync`` (no SMs
taken from the overlapped compute). Their symmetric buffers are allocated once per process and
shard group (``_shared_symmetric_buffer``) and never freed: a store built later in the same
process (a re-initialization, a test, a benchmark arm) reuses them. Guard rails: torch's
symmetric-memory API is checked before the first allocation (``copy_engine_api_check``); a
buffer never comes from torch's implicit symmetric pool (``_symmetric_empty``); every reuse /
allocate decision is checked to agree across the group (``_check_group_agreement``) and every
buffer is handshaken through its peers' header words under a finite barrier at setup
(``_SymmetricBuffer._handshake``), so a stale or mismatched symmetric handle fails loudly at
setup instead of deadlocking a step; the in-step barriers take an optional finite timeout
(``copy_engine_timeout_ms``; 0 = wait forever, like NCCL; a finite timeout traps the device).
"""

import importlib
import inspect
import itertools
import logging
import secrets
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from megatron.core.transformer.loop.param_store_binding import align_elements, tensor_region

__all__ = [
    "COPY_ENGINE_EXPERIMENTAL_NOTE",
    "GATHER_ENGINES",
    "REDUCE_ENGINES",
    "StoreComm",
    "copy_engine_allocation_paths",
    "copy_engine_api_check",
    "copy_engine_buffer_bytes",
    "copy_engine_unavailable_reason",
]

logger = logging.getLogger(__name__)

COPY_ENGINE_EXPERIMENTAL_NOTE = (
    "loop parameter store copy engines (gather_engine / reduce_engine = 'copy') are "
    "experimental: the shard group must lie in one NVLink domain reachable by torch symmetric "
    "memory; their symmetric buffers live for the process; the copy reduce-scatter is "
    "deterministic but equals NCCL only within the fp32 association bound"
)

# "nccl": the all-gathers are NCCL collectives (SM kernels). "copy": the in-step all-gathers pull
# the peers' shards from a symmetric-memory mirror with copy-engine copies (no SM; one NVLink
# domain per shard group), see ``_CopyEngineGather``.
GATHER_ENGINES: Tuple[str, ...] = ("nccl", "copy")
# "nccl": the reduce-scatters are NCCL collectives. "copy": copy-engine pushes into symmetric
# receive slots plus one local sum, see ``_CopyEngineReduce`` (deterministic, but associated
# differently from NCCL's ring: equal to the NCCL engine within the fp32 association bound).
REDUCE_ENGINES: Tuple[str, ...] = ("nccl", "copy")


def _symmetric_memory_class():
    """torch's ``_SymmetricMemory`` (one place, so tests can substitute it)."""
    # pylint: disable-next=import-outside-toplevel,no-name-in-module
    from torch._C._distributed_c10d import _SymmetricMemory

    return _SymmetricMemory


def _accepts_keyword(function: Any, name: str) -> bool:
    """Whether ``function`` takes the keyword ``name``: its signature, or for a pybind method
    (no introspectable signature, or only ``**kwargs``) its docstring's signature lines."""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        parameters = None
    if parameters is not None and not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        return name in parameters
    doc = getattr(function, "__doc__", None) or ""
    return f"{name}:" in doc or f"{name}=" in doc


def copy_engine_api_check() -> Dict[str, object]:
    """Pure introspection of the torch symmetric-memory API the copy engines need:
    ``{"symmetric_memory": importable, "empty_strided_p2p": the dedicated allocator,
    "barrier_timeout_ms": barrier(timeout_ms=) accepted, "torch": version}``. No allocation, no
    collective (the GPU job envprobes print it)."""
    result: Dict[str, object] = {
        "symmetric_memory": False,
        "empty_strided_p2p": False,
        "barrier_timeout_ms": False,
        "torch": torch.__version__,
    }
    try:
        importlib.import_module("torch.distributed._symmetric_memory")
        cls = _symmetric_memory_class()
    except (ImportError, AttributeError):
        return result
    result["symmetric_memory"] = True
    result["empty_strided_p2p"] = callable(getattr(cls, "empty_strided_p2p", None))
    barrier = getattr(cls, "barrier", None)
    result["barrier_timeout_ms"] = barrier is not None and _accepts_keyword(barrier, "timeout_ms")
    return result


def copy_engine_unavailable_reason() -> Optional[str]:
    """None when the copy engines' torch API is present, else the reason naming every missing
    piece (reported as "unavailable", never mistaken for an R3-A regression)."""
    check = copy_engine_api_check()
    missing = [
        what
        for key, what in (
            ("symmetric_memory", "torch.distributed._symmetric_memory"),
            ("empty_strided_p2p", "_SymmetricMemory.empty_strided_p2p"),
            ("barrier_timeout_ms", "_SymmetricMemory.barrier(timeout_ms=)"),
        )
        if not check[key]
    ]
    if not missing:
        return None
    return (
        f"copy engines unavailable in this torch ({check['torch']}): missing "
        f"{', '.join(missing)}; use the NCCL engines"
    )


# (group name, purpose) -> how the latest symmetric buffer of that purpose was allocated.
_ALLOCATION_PATHS: Dict[Tuple[str, str], str] = {}


def copy_engine_allocation_paths() -> Dict[Tuple[str, str], str]:
    """``(group name, purpose) -> allocator`` of every symmetric buffer this process allocated
    (always ``"empty_strided_p2p"``: the implicit-pool fallback is refused)."""
    return dict(_ALLOCATION_PATHS)


def _symmetric_empty(
    numel: int, dtype: torch.dtype, device: torch.device, *, group_name: str, purpose: str
) -> torch.Tensor:
    """A dedicated symmetric-memory allocation of ``numel`` elements.

    Allocated directly by the symmetric-memory allocator, never from torch's implicit symmetric
    MemPool (``symm_mem.empty``'s default): a pool hands a freed block back to the next request
    together with the block's cached rendezvous, so a store built after another one could get a
    reused allocation on some ranks and a fresh one on others (R3-A: the second copy-engine store
    on a sub-world shard group deadlocked in a barrier). A dedicated allocation starts at offset
    0 of its own rendezvous, and ``_shared_symmetric_buffer`` never frees it. A torch whose
    allocator takes another signature is refused (no silent pool fallback)."""
    try:
        tensor = _symmetric_memory_class().empty_strided_p2p((int(numel),), (1,), dtype, device)
    except TypeError as error:
        raise RuntimeError(
            f"copy engine: torch's symmetric allocator signature is unsupported ({error}); the "
            "implicit-pool fallback is refused (R3-A); use the NCCL engines"
        ) from error
    _ALLOCATION_PATHS[(str(group_name), str(purpose))] = "empty_strided_p2p"
    return tensor


def _all_gather_ints(group, device: torch.device, values: Sequence[int]) -> List[List[int]]:
    """Every group rank's ``values`` (int64, one all-gather on ``device``), in group-rank
    order. Setup-time only (a host synchronization)."""
    local = torch.tensor([int(v) for v in values], dtype=torch.int64, device=device)
    out = torch.empty(group.size() * local.numel(), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(out, local, group=group)
    return out.view(group.size(), local.numel()).tolist()


def _agreement_mismatch(records: Sequence[Sequence[int]]) -> Optional[str]:
    """Pure: None when every rank's record is equal, else a message listing every rank's."""
    first = tuple(records[0]) if records else ()
    if all(tuple(record) == first for record in records):
        return None
    return "; ".join(f"rank {r}: {tuple(record)}" for r, record in enumerate(records))


def _check_group_agreement(
    group, device: torch.device, record: Sequence[int], what: str, *, private: Sequence[int] = ()
) -> List[Tuple[int, ...]]:
    """All-gather ``record`` (and ``private``, not compared) over ``group``; raise RuntimeError
    on every rank with ``_agreement_mismatch``'s message when the records differ. Returns every
    group rank's ``private`` words. Setup-time only (never inside a step).

    It guards the Python-level reuse / allocate split (deterministic today, so a regression
    guard: a rank allocating while its peers reuse would hang in the rendezvous); the state
    below Python is covered by ``_SymmetricBuffer._handshake``."""
    rows = _all_gather_ints(group, device, list(record) + list(private))
    count = len(record)
    message = _agreement_mismatch([row[:count] for row in rows])
    if message is not None:
        ranks = dist.get_process_group_ranks(group)
        raise RuntimeError(
            f"copy engine: the ranks of shard group {ranks} disagree on the {what} "
            f"((reuse, requested elements, generation, handshake sequence) per group rank: "
            f"{message}); every rank "
            "must build the same copy-engine stores in the same order"
        )
    return [tuple(row[count:]) for row in rows]


_NONCE: List[int] = []


def _process_nonce() -> int:
    """A random int64 drawn once per process (the handshake's identity of this rank)."""
    if not _NONCE:
        _NONCE.append(secrets.randbits(62))
    return _NONCE[0]


class _SymmetricBuffer:
    """One symmetric allocation rendezvoused on a group, kept for the process.

    ``buffer`` is this rank's payload (``numel`` elements of ``dtype`` at offset 0 of the
    allocation) and ``peers[r]`` rank ``r``'s (this rank's own at ``rank``). After the payload,
    at a 256-B aligned offset, ``HEADER_INT64`` header words per rank carry the setup handshake
    (``_handshake``; not part of the payload or of ``nbytes``): magic, generation, the
    handshake's sequence number (so a reuse handshake cannot pass on the previous one's words),
    group rank and process nonce. ``last_use`` is an event
    recorded after the latest operation on the buffer (by any store, on its side stream); the
    next operation's stream waits it first (``acquire``), so stores built one after another
    never overlap on the device. ``owner`` is the token of the store whose data the buffer holds
    (the gather mirror, claimed at each refresh). ``timeout_ms``: the default barrier timeout
    (0 = wait forever); ``handshake_timeout_ms``: the setup handshake's, always finite."""

    HEADER_INT64 = 5  # (magic, generation, handshake sequence, group rank, process nonce)
    MAGIC = 0x4C50535354524331  # "LPSSTRC1"

    def __init__(
        self,
        group,
        device: torch.device,
        dtype: torch.dtype,
        numel: int,
        *,
        purpose: str = "",
        timeout_ms: int = 0,
        generation: int = 1,
        handshake_timeout_ms: int = 60_000,
    ):
        # pylint: disable-next=import-outside-toplevel
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group  # kept alive: the cache key's group name must name this group
        self.numel = int(numel)
        self.dtype = dtype
        self.purpose = purpose
        self.timeout_ms = int(timeout_ms)
        self.handshake_timeout_ms = int(handshake_timeout_ms)
        self.generation = int(generation)
        self.group_ranks = tuple(dist.get_process_group_ranks(group))
        self.payload_bytes = self.numel * torch.empty((), dtype=dtype).element_size()
        self.header_offset = align_elements(self.payload_bytes, torch.uint8, 256)  # bytes
        self.raw = _symmetric_empty(
            self.header_offset + self.HEADER_INT64 * 8,
            torch.uint8,
            device,
            group_name=group.group_name,
            purpose=purpose,
        )
        try:
            self._attach(symm_mem, group, dtype)
        except BaseException:
            # The allocation may already be rendezvoused: like an outgrown buffer, it is kept
            # (retired, counted in ``copy_engine_buffer_bytes``), never freed.
            _RETIRED_SYMMETRIC_BUFFERS.append(self)
            raise
        self.last_use: Optional[Any] = None
        self.owner: Optional[int] = None

    def _attach(self, symm_mem, group, dtype: torch.dtype) -> None:
        """Rendezvous ``raw`` over ``group`` and build the payload / header views (own and
        peers'); raises ``RuntimeError`` when the group does not map onto the handle."""
        where = (
            f"the {self.purpose} buffer on shard group ranks {list(self.group_ranks)} (the shard "
            "group must lie in one NVLink domain reachable by torch symmetric memory)"
        )
        try:
            self.handle = symm_mem.rendezvous(self.raw, group.group_name)
        except Exception as error:  # pylint: disable=broad-except
            raise RuntimeError(
                f"copy engine: symmetric-memory rendezvous of {where} failed: {error}"
            ) from error
        if self.handle is None or self.handle.world_size != group.size():
            raise RuntimeError(f"copy engine: symmetric-memory rendezvous of {where} failed")
        self.rank = self.handle.rank
        self.size = self.handle.world_size
        if self.rank != dist.get_rank(group):
            raise RuntimeError(
                f"copy engine: symmetric-memory rank {self.rank} != group rank "
                f"{dist.get_rank(group)} for {where}"
            )
        self.buffer = self.raw.narrow(0, 0, self.payload_bytes).view(dtype)
        self.header = self.raw.narrow(0, self.header_offset, self.HEADER_INT64 * 8).view(
            torch.int64
        )
        self.peers = [
            self.buffer if r == self.rank else self.handle.get_buffer(r, (self.numel,), dtype)
            for r in range(self.size)
        ]
        self.peer_headers = [
            (
                self.header
                if r == self.rank
                else self.handle.get_buffer(
                    r, (self.HEADER_INT64,), torch.int64, self.header_offset // 8
                )
            )
            for r in range(self.size)
        ]

    @property
    def nbytes(self) -> int:
        """This rank's payload bytes of the allocation (the header words excluded)."""
        return self.payload_bytes

    def acquire(self, stream) -> None:
        """``stream`` waits the latest operation on the buffer."""
        if self.last_use is not None:
            stream.wait_event(self.last_use)

    def release(self, stream) -> None:
        """Record the end of an operation on ``stream``."""
        event = torch.cuda.Event()
        event.record(stream)
        self.last_use = event

    def barrier(self, what: str = "", timeout_ms: Optional[int] = None) -> None:
        """A symmetric-memory barrier of the group on the current stream (channel 0).
        ``timeout_ms`` (default ``self.timeout_ms``; 0 = wait forever): a finite timeout turns
        a hang into a device-side trap with torch's message (the CUDA context is lost). With
        DEBUG logging, a host line names the engine, the group ranks and the generation first
        (the trap text itself does not). The store has no debug switch of its own: this
        module's logger level is the switch."""
        timeout = self.timeout_ms if timeout_ms is None else int(timeout_ms)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "copy engine %s barrier (%s buffer): shard group ranks %s, generation %d, "
                "timeout %d ms",
                what,
                self.purpose,
                list(self.group_ranks),
                self.generation,
                timeout,
            )
        self.handle.barrier(channel=0, timeout_ms=timeout)

    def _expected_header(self, generation: int, sequence: int, rank: int, nonce: int):
        return [self.MAGIC, int(generation), int(sequence), int(rank), int(nonce)]

    def _handshake(self, generation: int, nonces: Sequence[int], sequence: int = 1) -> None:
        """Setup-time check that every peer mapping reaches the peer's current buffer.

        On the current stream, after the buffer's last operation: write (MAGIC, generation,
        ``sequence``, group rank, this process's nonce) into this rank's header words; a barrier
        with the finite ``handshake_timeout_ms``; read every peer's header through its mapping
        and compare with (MAGIC, ``generation``, ``sequence``, peer rank, ``nonces[peer]``);
        all-gather the verdicts over the group (NCCL) and raise RuntimeError on every rank
        naming each disagreeing (rank, peer) pair. ``sequence`` (agreed across the group, one
        more per handshake of the buffer's key) makes every handshake's words new, so a peer
        whose write did not land fails on a reuse too. A stale handle or signal pad (the R3-A
        hypothesis) or a peer at another generation fails here, loudly, instead of deadlocking
        a step."""
        device = self.buffer.device
        current = torch.cuda.current_stream(device)
        self.acquire(current)
        mine = self._expected_header(generation, sequence, self.rank, nonces[self.rank])
        self.header.copy_(torch.tensor(mine, dtype=torch.int64, device=device))
        self.barrier("handshake", timeout_ms=self.handshake_timeout_ms)
        seen = torch.stack([header.clone() for header in self.peer_headers])
        expected = torch.tensor(
            [self._expected_header(generation, sequence, r, nonces[r]) for r in range(self.size)],
            dtype=torch.int64,
            device=device,
        )
        bad = (seen != expected).any(dim=1).to(torch.int64)
        verdict = torch.cat([bad, seen.reshape(-1)])
        gathered = torch.empty(self.size * verdict.numel(), dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(gathered, verdict, group=self.group)
        self.release(current)
        problems = []
        words = self.HEADER_INT64
        for rank, row in enumerate(gathered.view(self.size, -1).tolist()):
            for peer in range(self.size):
                if row[peer]:
                    magic, gen, seq, peer_rank, nonce = row[
                        self.size + words * peer : self.size + words * (peer + 1)
                    ]
                    problems.append(
                        f"rank {rank} sees peer {peer} at (magic {magic:#x}, generation {gen}, "
                        f"sequence {seq}, rank {peer_rank}, nonce {nonce}) != (magic "
                        f"{self.MAGIC:#x}, generation {generation}, sequence {sequence}, rank "
                        f"{peer}, nonce {nonces[peer]})"
                    )
        if problems:
            raise RuntimeError(
                f"copy engine: symmetric buffer handshake failed for the {self.purpose} buffer "
                f"on shard group ranks {list(self.group_ranks)} (a stale or mismatched "
                "symmetric handle, the R3-A class): " + "; ".join(problems[:8])
            )


# (group name, purpose, dtype, device index) -> the process's buffer. Never freed (see
# ``_symmetric_empty``): a larger request allocates a new buffer and retires the old one, which
# stays allocated (``_RETIRED_SYMMETRIC_BUFFERS``) so its memory is never handed out again.
_SYMMETRIC_BUFFERS: Dict[Tuple[str, str, torch.dtype, int], _SymmetricBuffer] = {}
_RETIRED_SYMMETRIC_BUFFERS: List[_SymmetricBuffer] = []
# The same keys -> buffers allocated so far (a buffer's generation is its allocation count).
_SYMMETRIC_GENERATIONS: Dict[Tuple[str, str, torch.dtype, int], int] = {}
# The same keys -> handshakes requested so far (each handshake writes the next number).
_HANDSHAKE_SEQUENCES: Dict[Tuple[str, str, torch.dtype, int], int] = {}
_OWNER_TOKENS = itertools.count(1)


def copy_engine_buffer_bytes() -> Dict[str, int]:
    """This process's symmetric payload bytes: ``live`` (the cached buffers) and ``retired``
    (outgrown buffers, kept allocated)."""
    return {
        "live": sum(entry.nbytes for entry in _SYMMETRIC_BUFFERS.values()),
        "retired": sum(entry.nbytes for entry in _RETIRED_SYMMETRIC_BUFFERS),
    }


def _shared_symmetric_buffer(
    group, device: torch.device, dtype: torch.dtype, numel: int, purpose: str, *, timeout_ms: int
) -> _SymmetricBuffer:
    """The process's symmetric buffer of ``purpose`` on ``group`` with at least ``numel``
    elements: the cached one when it fits, else a new one (a collective over ``group``: every
    rank of the group makes the same requests in the same order, so the ranks agree; checked by
    ``_check_group_agreement`` before any allocation, and the buffer is handshaken after).

    The cache changes only after success: a new buffer replaces (and retires) the cached one
    once it is built, and a buffer whose handshake fails is evicted to the retired list (kept
    allocated, never handed out again), so a failed request leaves no half-updated entry and
    the next request allocates afresh. The handshake sequence number advances on every request
    of every rank (also a failed one), so the ranks stay in step."""
    device = torch.device(device)
    numel = max(1, int(numel))
    key = (group.group_name, purpose, dtype, device.index or 0)
    entry = _SYMMETRIC_BUFFERS.get(key)
    reuse = entry is not None and entry.group is group and entry.numel >= numel
    generation = _SYMMETRIC_GENERATIONS.get(key, 0)
    sequence = _HANDSHAKE_SEQUENCES.get(key, 0) + 1
    _HANDSHAKE_SEQUENCES[key] = sequence
    nonces = _check_group_agreement(
        group,
        device,
        (int(reuse), numel, generation, sequence),
        f"{purpose} buffer",
        private=(_process_nonce(),),
    )
    if not reuse:
        new_entry = _SymmetricBuffer(
            group,
            device,
            dtype,
            numel,
            purpose=purpose,
            timeout_ms=timeout_ms,
            generation=generation + 1,
        )
        if entry is not None:
            _RETIRED_SYMMETRIC_BUFFERS.append(entry)
        entry = new_entry
        _SYMMETRIC_GENERATIONS[key] = entry.generation
        _SYMMETRIC_BUFFERS[key] = entry
    try:
        entry._handshake(  # pylint: disable=protected-access
            entry.generation, [row[0] for row in nonces], sequence
        )
    except BaseException:
        if _SYMMETRIC_BUFFERS.get(key) is entry:
            del _SYMMETRIC_BUFFERS[key]
            _RETIRED_SYMMETRIC_BUFFERS.append(entry)
        raise
    return entry


class _CopyEngineGather:
    """Copy-engine all-gathers of store units (torch symmetric memory, CUDA, one NVLink domain).

    Every rank keeps a mirror of its canonical shards (all units, every tensor at a 256-B aligned
    offset) in a symmetric buffer, refreshed once per step (``refresh``): a barrier (every peer
    has finished the previous step's pulls), the local shard copies, a barrier (every mirror is
    fresh). The shards do not change inside a step (the optimizer runs after ``end_step``), so a
    gather is only pulls: for every peer, ``copy_`` of its mirror region into the peer's rows of
    the destination, a ``cudaMemcpyAsync`` the copy engines run without SMs. The pulls of one
    gather are spread over ``streams`` sub-streams of the all-gather stream (one per peer, round
    robin) and joined back, so stream order on the all-gather stream still covers a gather.
    This is torch's low-contention all-gather (``_symmetric_memory._low_contention_all_gather``)
    with the input already in symmetric memory and the per-call barriers hoisted to the step.

    The mirror is the process's shared buffer of the shard group (``_shared_symmetric_buffer``),
    reused by a store built later. ``refresh`` claims it for this store (``owner``) after the
    previous user's last operation (``acquire``); a gather of a store that does not hold it
    raises (two stores stepping interleaved on one shard group would read each other's shards).

    Costs: the mirror (this rank's shard bytes of the store, once more) and two barrier kernels
    per step. The eager gathers (``gather_now``) stay NCCL: they run outside a step, where the
    mirror may be older than the shards.
    """

    def __init__(
        self,
        group,
        device: torch.device,
        units: Sequence[Any],
        dtype: torch.dtype,
        streams: int,
        *,
        timeout_ms: int = 0,
    ):
        self.size = group.size()
        self.timeout_ms = int(timeout_ms)
        self.offsets: Dict[Tuple[int, int], int] = {}
        total = 0
        for unit in units:
            for index, tensor in enumerate(unit.tensors):
                if tensor.dtype != dtype:
                    raise ValueError(
                        f"copy-engine gather: {tensor.name} is {tensor.dtype}, the store {dtype}"
                    )
                total = align_elements(total, dtype, 256)
                self.offsets[(unit.uid, index)] = total
                total += tensor.numel_padded // self.size
        self.numel = max(total, 1)
        self.dtype = dtype
        self.shared = _shared_symmetric_buffer(
            group, device, dtype, self.numel, "gather", timeout_ms=self.timeout_ms
        )
        self.token = next(_OWNER_TOKENS)
        self.streams = [torch.cuda.Stream(device=device) for _ in range(max(1, int(streams)))]

    @property
    def nbytes(self) -> int:
        """The mirror's bytes on this rank (the shared buffer: at least this store's layout)."""
        return self.shared.nbytes

    def refresh(self, units: Sequence[Any]) -> None:
        """On the current (all-gather) stream: barrier, copy the shards in, barrier."""
        shared = self.shared
        current = torch.cuda.current_stream(shared.buffer.device)
        shared.acquire(current)
        shared.owner = self.token
        shared.barrier("gather refresh", timeout_ms=self.timeout_ms)
        for unit in units:
            for index, tensor in enumerate(unit.tensors):
                shard = tensor.shard.detach().reshape(-1)
                shared.buffer.narrow(0, self.offsets[(unit.uid, index)], shard.numel()).copy_(shard)
        shared.barrier("gather refresh", timeout_ms=self.timeout_ms)
        shared.release(current)

    def gather(self, unit, dest_flat: torch.Tensor) -> None:
        """On the current (all-gather) stream: pull every peer's shards of ``unit``."""
        shared = self.shared
        if shared.owner != self.token:
            raise RuntimeError(
                "copy-engine gather: the shard group's mirror holds another store's shards "
                "(two copy-engine stores stepping interleaved on one shard group)"
            )
        current = torch.cuda.current_stream(dest_flat.device)
        used = min(len(self.streams), self.size)
        for stream in self.streams[:used]:
            stream.wait_stream(current)
        for step in range(self.size):
            peer = (shared.rank + step) % self.size
            source = shared.peers[peer]
            with torch.cuda.stream(self.streams[step % used]):
                for index, tensor in enumerate(unit.tensors):
                    rows = tensor.numel_padded // self.size
                    region = tensor_region(dest_flat, tensor, padded=True)
                    region.narrow(0, peer * rows, rows).copy_(
                        source.narrow(0, self.offsets[(unit.uid, index)], rows)
                    )
        for stream in self.streams[:used]:
            current.wait_stream(stream)
        shared.release(current)


class _CopyEngineReduce:
    """Copy-engine reduce-scatters of fp32 unit gradients (torch symmetric memory, CUDA).

    Every rank owns a symmetric receive buffer of ``z`` slots, one per source rank; a slot holds
    one unit's shard regions (fp32, every tensor at a 256-B aligned offset; slots are sized for
    the largest unit). A reduce, on the current (reduce-scatter) stream:

    1. barrier: every peer has summed its previous reduce out of its slots;
    2. pushes: for every peer ``p``, ``copy_`` of rows ``p`` of each tensor's padded region of the
       source into slot ``rank`` of ``p``'s buffer (``cudaMemcpyAsync`` over NVLink, no SMs),
       spread over ``streams`` sub-streams and joined back;
    3. barrier: every peer's rows for this rank have landed;
    4. per tensor, one sum over the ``z`` slots (a short local kernel) into a fresh shard.

    The sum has the same shapes on every rank, in every store mode and run, so the result is
    deterministic and equal across modes; it is associated differently from NCCL's ring
    reduce-scatter (equal within the fp32 association bound). This is torch's low-contention
    reduce-scatter (``_low_contention_reduce_scatter_with_workspace``) with a persistent
    workspace (``_shared_symmetric_buffer``). Costs: ``z`` slots of the largest unit's fp32 shard
    (one unit's fp32 gradient), two barrier kernels per reduce.
    """

    def __init__(
        self,
        group,
        device: torch.device,
        units: Sequence[Any],
        streams: int,
        *,
        timeout_ms: int = 0,
    ):
        self.size = group.size()
        self.timeout_ms = int(timeout_ms)
        self.offsets: Dict[Tuple[int, int], int] = {}
        slot = 0
        for unit in units:
            offset = 0
            for index, tensor in enumerate(unit.tensors):
                offset = align_elements(offset, torch.float32, 256)
                self.offsets[(unit.uid, index)] = offset
                offset += tensor.numel_padded // self.size
            slot = max(slot, align_elements(offset, torch.float32, 256))
        self.slot = max(slot, 1)
        self.shared = _shared_symmetric_buffer(
            group,
            device,
            torch.float32,
            self.size * self.slot,
            "reduce",
            timeout_ms=self.timeout_ms,
        )
        self.streams = [torch.cuda.Stream(device=device) for _ in range(max(1, int(streams)))]

    @property
    def nbytes(self) -> int:
        """The receive buffer's bytes on this rank."""
        return self.shared.nbytes

    def reduce(self, unit, src_flat: torch.Tensor) -> List[torch.Tensor]:
        """On the current (reduce-scatter) stream: reduce-scatter ``src_flat`` (fp32, ``unit``'s
        padded layout); returns this rank's reduced shard of every tensor (fresh tensors)."""
        shared = self.shared
        current = torch.cuda.current_stream(src_flat.device)
        shared.acquire(current)
        shared.barrier("reduce", timeout_ms=self.timeout_ms)
        used = min(len(self.streams), self.size)
        for stream in self.streams[:used]:
            stream.wait_stream(current)
        base = shared.rank * self.slot
        for step in range(self.size):
            peer = (shared.rank + step) % self.size
            target = shared.peers[peer]
            with torch.cuda.stream(self.streams[step % used]):
                for index, tensor in enumerate(unit.tensors):
                    rows = tensor.numel_padded // self.size
                    region = tensor_region(src_flat, tensor, padded=True)
                    target.narrow(0, base + self.offsets[(unit.uid, index)], rows).copy_(
                        region.narrow(0, peer * rows, rows)
                    )
        for stream in self.streams[:used]:
            current.wait_stream(stream)
        shared.barrier("reduce", timeout_ms=self.timeout_ms)
        slots = shared.buffer.narrow(0, 0, self.size * self.slot).view(self.size, self.slot)
        outs = []
        for index, tensor in enumerate(unit.tensors):
            rows = tensor.numel_padded // self.size
            outs.append(slots.narrow(1, self.offsets[(unit.uid, index)], rows).sum(dim=0))
        shared.release(current)
        return outs


class StoreComm:
    """All-gather / reduce-scatter of store units on side streams (synchronous on CPU).

    Args:
        shard_group: the shard axis (``z`` ranks) of the all-gathers.
        device: the store's device.
        rs_group: the reduce-scatter communicator (default ``shard_group``).
        timing: create timing-enabled events (exposure measurements).
        trivial: ``z = 1`` without collectives (tests).
        record_calls: keep a host log ``(op, unit key, tensors)`` of every call.
        delay_cycles: (debug, CUDA) spin each side stream this long before every collective,
            after its waits: a consumer that skipped an event wait then races deterministically.
        gather_engine: ``"nccl"`` or ``"copy"`` (the in-step gathers by copy engines from a
            symmetric-memory mirror, ``_CopyEngineGather``; CUDA, non-trivial axis; set up by
            ``setup_copy_engine`` before the first step).
        copy_streams: the copy engines' sub-streams per gather / reduce.
        reduce_engine: ``"nccl"`` or ``"copy"`` (the reduce-scatters by copy-engine pushes into
            symmetric receive slots and one local sum, ``_CopyEngineReduce``; CUDA, non-trivial
            axis; set up by ``setup_copy_engine``).
        copy_engine_timeout_ms: the copy engines' in-step symmetric-memory barrier timeout
            (0 = wait forever, like NCCL; a finite value traps the device on a hang, which
            loses the CUDA context: tests and benchmarks only). The setup handshake always
            uses a finite timeout.

    The copy engines are experimental (``COPY_ENGINE_EXPERIMENTAL_NOTE``).
    """

    def __init__(
        self,
        shard_group,
        device: torch.device,
        *,
        rs_group=None,
        timing: bool = False,
        trivial: bool = False,
        record_calls: bool = False,
        delay_cycles: int = 0,
        gather_engine: str = "nccl",
        copy_streams: int = 1,
        reduce_engine: str = "nccl",
        copy_engine_timeout_ms: int = 0,
    ):
        if gather_engine not in GATHER_ENGINES:
            raise ValueError(f"unknown gather engine {gather_engine!r} (one of {GATHER_ENGINES})")
        if reduce_engine not in REDUCE_ENGINES:
            raise ValueError(f"unknown reduce engine {reduce_engine!r} (one of {REDUCE_ENGINES})")
        self.device = torch.device(device)
        for what, engine in (("gather", gather_engine), ("reduce-scatter", reduce_engine)):
            if engine == "copy" and (self.device.type != "cuda" or trivial):
                raise ValueError(
                    f"the copy-engine {what} needs CUDA and a non-trivial shard axis (z > 1)"
                )
        if int(copy_engine_timeout_ms) < 0:
            raise ValueError(f"copy_engine_timeout_ms must be >= 0, got {copy_engine_timeout_ms}")
        self.copy_engine_timeout_ms = int(copy_engine_timeout_ms)
        self.gather_engine = gather_engine
        self.reduce_engine = reduce_engine
        self.copy_streams = int(copy_streams)
        self.copy_engine: Optional[_CopyEngineGather] = None
        self.copy_reduce: Optional[_CopyEngineReduce] = None
        self._copy_units: Tuple[Any, ...] = ()
        self.delay_cycles = int(delay_cycles)
        self.cuda = self.device.type == "cuda"
        self.ag_group = shard_group
        self.rs_group = shard_group if rs_group is None else rs_group
        self.timing = bool(timing)
        self.trivial = bool(trivial)
        self.log: Optional[List[Tuple[str, str, int]]] = [] if record_calls else None
        if self.cuda:
            self.ag_stream = torch.cuda.Stream(device=self.device)
            self.rs_stream = torch.cuda.Stream(device=self.device)
        else:
            self.ag_stream = None
            self.rs_stream = None

    # -----------------------------------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------------------------------

    def event(self, stream=None) -> Optional[Any]:
        """An event recorded on ``stream`` (default: the current stream); None on CPU."""
        if not self.cuda:
            return None
        event = torch.cuda.Event(enable_timing=self.timing)
        event.record(torch.cuda.current_stream(self.device) if stream is None else stream)
        return event

    def wait(self, event) -> None:
        """The current stream waits ``event`` (no-op for None / on CPU)."""
        if event is not None and self.cuda:
            torch.cuda.current_stream(self.device).wait_event(event)

    def _delay(self) -> None:
        """Spin the current (side) stream ``delay_cycles`` cycles (debug)."""
        if self.delay_cycles > 0 and self.cuda:
            torch.cuda._sleep(self.delay_cycles)  # pylint: disable=protected-access

    def setup_copy_engine(self, units: Sequence[Any], dtype: torch.dtype) -> None:
        """Set up the copy engines: the gather's symmetric mirror of ``units``' shards and the
        reduce-scatter's symmetric receive slots (collectives over the shard / reduce-scatter
        group, every rank with the same units, the first time a group needs them); a no-op for
        the NCCL engines or when set up already. Raises RuntimeError before any allocation when
        this torch lacks the symmetric-memory API (``copy_engine_unavailable_reason``)."""
        gather = self.gather_engine == "copy" and self.copy_engine is None
        reduce = self.reduce_engine == "copy" and self.copy_reduce is None
        if gather or reduce:
            reason = copy_engine_unavailable_reason()
            if reason is not None:
                raise RuntimeError(reason)
        if gather:
            self._copy_units = tuple(units)
            self.copy_engine = _CopyEngineGather(
                self.ag_group,
                self.device,
                self._copy_units,
                dtype,
                self.copy_streams,
                timeout_ms=self.copy_engine_timeout_ms,
            )
        if reduce:
            self.copy_reduce = _CopyEngineReduce(
                self.rs_group,
                self.device,
                tuple(units),
                self.copy_streams,
                timeout_ms=self.copy_engine_timeout_ms,
            )

    @property
    def copy_engines_ready(self) -> bool:
        """Every requested copy engine is set up (True for the NCCL engines)."""
        return (self.gather_engine != "copy" or self.copy_engine is not None) and (
            self.reduce_engine != "copy" or self.copy_reduce is not None
        )

    @property
    def copy_engine_bytes(self) -> int:
        """The copy engines' symmetric bytes on this rank: the gather mirror plus the
        reduce-scatter receive slots (0 for the NCCL engines)."""
        total = 0 if self.copy_engine is None else self.copy_engine.nbytes
        return total + (0 if self.copy_reduce is None else self.copy_reduce.nbytes)

    def step_begin(self) -> None:
        """Both side streams wait the compute stream (this step's ``zero_grad_buffer`` and the
        optimizer's shard updates precede every gather and every add). The copy engine
        refreshes its mirror on the all-gather stream."""
        if self.cuda:
            current = torch.cuda.current_stream(self.device)
            self.ag_stream.wait_stream(current)
            self.rs_stream.wait_stream(current)
            if not self.copy_engines_ready:
                raise RuntimeError("copy engine: setup_copy_engine was not called")
            if self.copy_engine is not None:
                with torch.cuda.stream(self.ag_stream):
                    self.copy_engine.refresh(self._copy_units)

    def join(self) -> None:
        """The compute stream waits every issued gather and reduce."""
        if self.cuda:
            current = torch.cuda.current_stream(self.device)
            current.wait_stream(self.ag_stream)
            current.wait_stream(self.rs_stream)

    def _measure_unit_collectives(
        self,
        unit,
        gather_dest: torch.Tensor,
        *,
        repeat: int = 10,
        reduce_src: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Diagnostic (benchmarks, between steps, CUDA, non-trivial axis): the standalone time
        of ``unit``'s all-gather and reduce-scatter with the configured engines, on the side
        streams. ``repeat`` timed gathers into ``gather_dest`` (copy engine: the mirror is
        refreshed first, as ``step_begin`` does), then ``repeat`` timed reduce-scatters of
        ``reduce_src`` (fp32, the unit's padded layout; a zero buffer when None) into scratch
        outputs that are dropped (never ``main_grad``: adding +0.0 would turn a -0.0 into
        +0.0); each timed series follows one untimed call (it absorbs the ranks' arrival skew).
        Nothing is logged. Returns ``{"gather_ms", "reduce_ms", "unit_bytes"}`` (ms per call,
        the unit's gathered bytes)."""
        if not self.cuda or self.trivial:
            raise RuntimeError("store collectives are measured on CUDA with a shard axis z > 1")
        if not self.copy_engines_ready:
            raise RuntimeError("copy engine: setup_copy_engine was not called")
        repeat = max(1, int(repeat))
        span = max(t.offset + t.numel_padded for t in unit.tensors)
        if reduce_src is None:
            reduce_src = torch.zeros(span, dtype=torch.float32, device=self.device)
        current = torch.cuda.current_stream(self.device)
        self.ag_stream.wait_stream(current)
        self.rs_stream.wait_stream(current)
        timing = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

        def gather() -> None:
            if self.copy_engine is not None:
                self.copy_engine.gather(unit, gather_dest)
            else:
                self._all_gather(unit, gather_dest)

        def reduce() -> None:
            if self.copy_reduce is not None:
                self.copy_reduce.reduce(unit, reduce_src)
                return
            regions = [tensor_region(reduce_src, t, padded=True) for t in unit.tensors]
            outs = [
                torch.empty(r.numel() // self.rs_group.size(), dtype=r.dtype, device=r.device)
                for r in regions
            ]
            _coalesced(
                dist.reduce_scatter_tensor, outs, regions, group=self.rs_group, device=self.device
            )

        # One untimed call of each before its timed loop: the first collective of a series
        # absorbs the ranks' arrival skew (and lazy communicator setup), not transfer time.
        with torch.cuda.stream(self.ag_stream):
            if self.copy_engine is not None:
                self.copy_engine.refresh(self._copy_units)
            gather()
            timing[0].record(self.ag_stream)
            for _ in range(repeat):
                gather()
            timing[1].record(self.ag_stream)
        with torch.cuda.stream(self.rs_stream):
            self.rs_stream.wait_event(timing[1])  # measured alone, not against the gathers
            reduce()
            timing[2].record(self.rs_stream)
            for _ in range(repeat):
                reduce()
            timing[3].record(self.rs_stream)
        self.join()
        torch.cuda.synchronize(self.device)
        elem = gather_dest.element_size()
        return {
            "gather_ms": timing[0].elapsed_time(timing[1]) / repeat,
            "reduce_ms": timing[2].elapsed_time(timing[3]) / repeat,
            "unit_bytes": sum(t.numel_padded for t in unit.tensors) * elem,
        }

    # -----------------------------------------------------------------------------------------
    # Collectives
    # -----------------------------------------------------------------------------------------

    def _record(self, op: str, unit) -> None:
        if self.log is not None:
            self.log.append((op, unit.key, len(unit.tensors)))

    def _all_gather(self, unit, dest_flat: torch.Tensor) -> None:
        outs = [tensor_region(dest_flat, t, padded=True) for t in unit.tensors]
        ins = [t.shard.detach().reshape(-1) for t in unit.tensors]
        for out, inp, tensor in zip(outs, ins, unit.tensors):
            if out.numel() != inp.numel() * (1 if self.trivial else self.ag_group.size()):
                raise RuntimeError(
                    f"store gather of {tensor.name}: region of {out.numel()} elements for a "
                    f"shard of {inp.numel()}"
                )
            if inp.dtype != out.dtype:
                raise RuntimeError(f"store gather of {tensor.name}: {inp.dtype} into {out.dtype}")
        if self.trivial:
            for out, inp in zip(outs, ins):
                out.copy_(inp)
            return
        _coalesced(dist.all_gather_into_tensor, outs, ins, group=self.ag_group, device=self.device)

    def _reduce_scatter_add(self, unit, src_flat: torch.Tensor) -> None:
        outs = []
        ins = []
        for tensor in unit.tensors:
            main_grad = getattr(tensor.shard, "main_grad", None)
            if main_grad is None:
                raise RuntimeError(
                    f"store reduce of {tensor.name}: its canonical shard has no main_grad (the "
                    "DDP grad buffer, or a test's fp32 buffer)"
                )
            region = tensor_region(src_flat, tensor, padded=True)
            if self.trivial:
                outs.append(region)
            elif self.copy_reduce is None:
                outs.append(
                    torch.empty(
                        region.numel() // self.rs_group.size(),
                        dtype=region.dtype,
                        device=region.device,
                    )
                )
                ins.append(region)
        if self.copy_reduce is not None:
            outs = self.copy_reduce.reduce(unit, src_flat)
        elif not self.trivial:
            _coalesced(
                dist.reduce_scatter_tensor, outs, ins, group=self.rs_group, device=self.device
            )
        for tensor, out in zip(unit.tensors, outs):
            main_grad = tensor.shard.main_grad
            if main_grad.numel() != out.numel():
                raise RuntimeError(
                    f"store reduce of {tensor.name}: {out.numel()} elements into a main_grad of "
                    f"{main_grad.numel()}"
                )
            main_grad.view(-1).add_(out)

    def gather(self, unit, dest_flat: torch.Tensor, *, wait_free=None) -> Optional[Any]:
        """All-gather ``unit``'s shards into ``dest_flat``; returns the ``ready`` event."""
        self._record("gather", unit)
        if not self.cuda:
            self._all_gather(unit, dest_flat)
            return None
        with torch.cuda.stream(self.ag_stream):
            if wait_free is not None:
                self.ag_stream.wait_event(wait_free)
            self._delay()
            if self.copy_engine is not None:
                self.copy_engine.gather(unit, dest_flat)
            else:
                self._all_gather(unit, dest_flat)
            return self.event(self.ag_stream)

    def gather_now(self, unit, dest_flat: torch.Tensor) -> None:
        """All-gather on the current stream (eager visits, refresh): ordered with compute."""
        self._record("gather", unit)
        self._all_gather(unit, dest_flat)

    def reduce(
        self,
        unit,
        src_flat: torch.Tensor,
        *,
        wait_ready=None,
        transient: bool = False,
        poison: bool = False,
    ) -> Optional[Any]:
        """Reduce-scatter ``src_flat`` (fp32, ``unit``'s padded layout) and add each tensor's
        shard into ``shard.main_grad``; returns the ``free`` event of ``src_flat``.
        ``poison`` NaN-fills the source after the reduce (pool entries, tests)."""
        self._record("reduce", unit)
        if not self.cuda:
            self._reduce_scatter_add(unit, src_flat)
            if poison:
                src_flat.fill_(float("nan"))
            return None
        if transient:
            src_flat.record_stream(self.rs_stream)
        with torch.cuda.stream(self.rs_stream):
            if wait_ready is not None:
                self.rs_stream.wait_event(wait_ready)
            self._delay()
            self._reduce_scatter_add(unit, src_flat)
            if poison:
                src_flat.fill_(float("nan"))
            return self.event(self.rs_stream)

    def zero_and_reduce(
        self, unit, src_flat: Optional[torch.Tensor], *, numel: int = 0, wait_ready=None
    ) -> Optional[Any]:
        """A null EU's reduce: zero the source on the reduce-scatter stream (a fresh one when
        ``src_flat`` is None, ``numel`` fp32 elements) and reduce it, so this EU's shards
        receive the other EUs' gradients. Returns the ``free`` event.

        ``wait_ready``: an event recorded on the compute stream at the point a real EU's reduce
        would wait (after the null item's segment, its MoE null collectives included), so the
        device order of this reduce-scatter against the EP collectives matches a real EU's."""
        self._record("reduce", unit)
        if not self.cuda:
            src = torch.zeros(numel, dtype=torch.float32) if src_flat is None else src_flat.zero_()
            self._reduce_scatter_add(unit, src)
            return None
        with torch.cuda.stream(self.rs_stream):
            if wait_ready is not None:
                self.rs_stream.wait_event(wait_ready)
            self._delay()
            if src_flat is None:
                src = torch.zeros(numel, dtype=torch.float32, device=self.device)
            else:
                src = src_flat.zero_()
            self._reduce_scatter_add(unit, src)
            return self.event(self.rs_stream)


def _coalesced(
    collective, outs: Sequence[torch.Tensor], ins: Sequence[torch.Tensor], *, group, device
) -> None:
    """``collective(out, inp, group=group)`` for every pair; one coalesced NCCL group on CUDA
    (waited by the current stream), plain synchronous calls on Gloo.

    The coalescing manager runs WITHOUT ``device``: the collectives take its fast path (one
    ``*_coalesced`` work, which ``wait`` releases). With ``device``, torch (2.12) also opens a
    ProcessGroupNCCL coalescing block and keeps only ``_end_coalescing``'s work; the fast-path
    work, which stashes every input and output for allocator safety, is never waited, and the
    process group keeps those tensors alive past the call (measured on H100: a released unit's
    fp32 gradient buffer stayed allocated, and the stashes accumulated across calls).
    """
    if not outs:
        return
    if device.type != "cuda" or len(outs) == 1:
        for out, inp in zip(outs, ins):
            collective(out, inp, group=group)
        return
    with dist._coalescing_manager(  # pylint: disable=protected-access
        group=group, async_ops=True
    ) as manager:
        for out, inp in zip(outs, ins):
            collective(out, inp, group=group, async_op=True)
    manager.wait()
