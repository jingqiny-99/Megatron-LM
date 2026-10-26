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

Copy engines (CUDA, one NVLink domain per shard group). ``gather_engine="copy"`` pulls the in-step
gathers from a symmetric-memory mirror of the shards (``_CopyEngineGather``);
``reduce_engine="copy"`` pushes each rank's rows of a gradient into the owners' symmetric receive
slots and sums the slots with one short kernel (``_CopyEngineReduce``). Both move the bytes with
``cudaMemcpyAsync`` (no SMs taken from the overlapped compute). Their symmetric buffers are
allocated once per process and shard group (``_shared_symmetric_buffer``) and never freed: a
store built later in the same process (a re-initialization, a test, a benchmark arm) reuses them.
"""

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from megatron.core.transformer.loop.param_store_binding import align_elements, tensor_region

__all__ = ["GATHER_ENGINES", "REDUCE_ENGINES", "StoreComm"]

# "nccl": the all-gathers are NCCL collectives (SM kernels). "copy": the in-step all-gathers pull
# the peers' shards from a symmetric-memory mirror with copy-engine copies (no SM; one NVLink
# domain per shard group), see ``_CopyEngineGather``.
GATHER_ENGINES: Tuple[str, ...] = ("nccl", "copy")
# "nccl": the reduce-scatters are NCCL collectives. "copy": copy-engine pushes into symmetric
# receive slots plus one local sum, see ``_CopyEngineReduce`` (deterministic, but associated
# differently from NCCL's ring: equal to the NCCL engine within the fp32 association bound).
REDUCE_ENGINES: Tuple[str, ...] = ("nccl", "copy")


def _symmetric_empty(numel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """A dedicated symmetric-memory allocation of ``numel`` elements.

    Allocated directly by the symmetric-memory allocator, not from torch's implicit symmetric
    MemPool (``symm_mem.empty``'s default): a pool hands a freed block back to the next request
    together with the block's cached rendezvous, so a store built after another one could get a
    reused allocation on some ranks and a fresh one on others (R3-A: the second copy-engine store
    on a sub-world shard group deadlocked in a barrier). A dedicated allocation starts at offset
    0 of its own rendezvous, and ``_shared_symmetric_buffer`` never frees it."""
    # pylint: disable-next=import-outside-toplevel
    import torch.distributed._symmetric_memory as symm_mem

    # pylint: disable-next=import-outside-toplevel,no-name-in-module
    from torch._C._distributed_c10d import _SymmetricMemory

    try:
        return _SymmetricMemory.empty_strided_p2p((numel,), (1,), dtype, device)
    except TypeError:  # pragma: no cover - torch builds with another signature
        return symm_mem.empty(numel, dtype=dtype, device=device)


class _SymmetricBuffer:
    """One symmetric allocation rendezvoused on a group, kept for the process.

    ``peers[r]`` is rank ``r``'s buffer (this rank's own at ``rank``). ``last_use`` is an event
    recorded after the latest operation on the buffer (by any store, on its side stream); the
    next operation's stream waits it first (``acquire``), so stores built one after another
    never overlap on the device. ``owner`` is the token of the store whose data the buffer holds
    (the gather mirror, claimed at each refresh)."""

    def __init__(self, group, device: torch.device, dtype: torch.dtype, numel: int):
        # pylint: disable-next=import-outside-toplevel
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group  # kept alive: the cache key's group name must name this group
        self.numel = int(numel)
        self.dtype = dtype
        self.buffer = _symmetric_empty(self.numel, dtype, device)
        self.handle = symm_mem.rendezvous(self.buffer, group.group_name)
        if self.handle is None or self.handle.world_size != group.size():
            raise RuntimeError("copy engine: symmetric-memory rendezvous failed")
        self.rank = self.handle.rank
        self.size = self.handle.world_size
        self.peers = [
            self.buffer if r == self.rank else self.handle.get_buffer(r, (self.numel,), dtype)
            for r in range(self.size)
        ]
        self.last_use: Optional[Any] = None
        self.owner: Optional[int] = None

    @property
    def nbytes(self) -> int:
        """This rank's bytes of the allocation."""
        return self.numel * self.buffer.element_size()

    def acquire(self, stream) -> None:
        """``stream`` waits the latest operation on the buffer."""
        if self.last_use is not None:
            stream.wait_event(self.last_use)

    def release(self, stream) -> None:
        """Record the end of an operation on ``stream``."""
        event = torch.cuda.Event()
        event.record(stream)
        self.last_use = event


# (group name, purpose, dtype, device index) -> the process's buffer. Never freed (see
# ``_symmetric_empty``): a larger request allocates a new buffer and retires the old one, which
# stays allocated (``_RETIRED_SYMMETRIC_BUFFERS``) so its memory is never handed out again.
_SYMMETRIC_BUFFERS: Dict[Tuple[str, str, torch.dtype, int], _SymmetricBuffer] = {}
_RETIRED_SYMMETRIC_BUFFERS: List[_SymmetricBuffer] = []
_OWNER_TOKENS = itertools.count(1)


def _shared_symmetric_buffer(
    group, device: torch.device, dtype: torch.dtype, numel: int, purpose: str
) -> _SymmetricBuffer:
    """The process's symmetric buffer of ``purpose`` on ``group`` with at least ``numel``
    elements: the cached one when it fits, else a new one (a collective over ``group``: every
    rank of the group makes the same requests in the same order, so the ranks agree)."""
    device = torch.device(device)
    key = (group.group_name, purpose, dtype, device.index or 0)
    entry = _SYMMETRIC_BUFFERS.get(key)
    if entry is not None and entry.group is group and entry.numel >= numel:
        return entry
    if entry is not None:
        _RETIRED_SYMMETRIC_BUFFERS.append(entry)
    entry = _SymmetricBuffer(group, device, dtype, max(1, int(numel)))
    _SYMMETRIC_BUFFERS[key] = entry
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
        self, group, device: torch.device, units: Sequence[Any], dtype: torch.dtype, streams: int
    ):
        self.size = group.size()
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
        self.shared = _shared_symmetric_buffer(group, device, dtype, self.numel, "gather")
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
        shared.handle.barrier(channel=0)
        for unit in units:
            for index, tensor in enumerate(unit.tensors):
                shard = tensor.shard.detach().reshape(-1)
                shared.buffer.narrow(0, self.offsets[(unit.uid, index)], shard.numel()).copy_(shard)
        shared.handle.barrier(channel=0)
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

    def __init__(self, group, device: torch.device, units: Sequence[Any], streams: int):
        self.size = group.size()
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
            group, device, torch.float32, self.size * self.slot, "reduce"
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
        shared.handle.barrier(channel=0)
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
        shared.handle.barrier(channel=0)
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
        the NCCL engines or when set up already."""
        if self.gather_engine == "copy" and self.copy_engine is None:
            self._copy_units = tuple(units)
            self.copy_engine = _CopyEngineGather(
                self.ag_group, self.device, self._copy_units, dtype, self.copy_streams
            )
        if self.reduce_engine == "copy" and self.copy_reduce is None:
            self.copy_reduce = _CopyEngineReduce(
                self.rs_group, self.device, tuple(units), self.copy_streams
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
