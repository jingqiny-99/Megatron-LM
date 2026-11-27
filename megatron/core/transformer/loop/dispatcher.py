# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The in-domain sample dispatcher of the multi-EU RSBF executor.

Moves the payloads of a plan's ``MovePhase`` (sample moves, chain records, item moves; see
``rsbf_plan.py``) between the EUs of one domain, and runs the per-round status exchange (the
exit bitmap).

Collective-order contract (the deadlock argument of multi-EU RSBF). Every rank of the domain
executes the same plan, hence the same sequence of move phases, MoE sub-rounds and status
exchanges:

* **Moves** use the domain group (``RSBFGroups.domain``, the data-parallel group). With the
  ``p2p`` transport a rank posts, for the phase's transfers in plan order, one ``isend`` per
  slot of every transfer it sends and one ``irecv`` per slot of every transfer it receives
  (slots in their declared order), as one ``batch_isend_irecv``; a rank without transfers in
  the phase posts nothing. Both ends of every pair post their ops in the same relative order,
  so matching is exact. With the ``all_to_all`` transport every rank of the domain calls one
  ``all_to_all_single`` per phase (zero splits allowed), even when it holds zero samples.
* **Status** uses a Gloo group over the same ranks (``RSBFGroups.status``): one
  ``all_reduce(MAX)`` of a CPU ``uint8[G + 1]`` bitmap per online round.
* Nothing else is issued here; a rank-local collective added inside a step would break the
  order.

Payload slots travel as their raw bytes (``uint8`` views of contiguous tensors), so every
dtype (bf16, fp32, int32, int64, bool) is copied bit-exactly over NCCL and Gloo alike; the
receiver allocates the slot's shape and dtype from the plan, so no size exchange is needed. A
zero-size slot (e.g. the criterion state at ``k = 0``) posts no op on either side.

NCCL requires the first ``batch_isend_irecv`` on a group to involve every rank, which
``warm_up`` does once (a ring of one byte).
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import torch
import torch.distributed as dist
from torch import Tensor

from megatron.core.transformer.loop.rsbf_plan import MovePhase, Transfer

__all__ = ["DispatcherStats", "LoopSampleDispatcher", "RSBFGroups"]

Transport = Literal["p2p", "all_to_all"]


@dataclass(frozen=True)
class RSBFGroups:
    """The process groups of one RSBF domain.

    ``domain``: the device group the moves run on (``pg_collection.dp``; NCCL on GPU, Gloo in
    CPU tests). ``status``: a Gloo group over the same ranks for the exit-bitmap exchange and
    the debug plan digests; required when the domain has more than one rank and the executor
    exchanges status (online criteria, ``check_plan``).
    """

    domain: Optional[dist.ProcessGroup]
    status: Optional[dist.ProcessGroup] = None


@dataclass
class DispatcherStats:
    """Counters of the dispatcher (per executor; the executor snapshots them per step)."""

    phases: int = 0  # exchange() calls
    p2p_ops: int = 0  # isend + irecv posted
    all_to_alls: int = 0  # all_to_all_single calls
    bytes_sent: int = 0
    bytes_received: int = 0
    status_exchanges: int = 0  # Gloo all-reduces of the exit bitmap
    digest_checks: int = 0  # Gloo all-gathers of plan digests (debug)


def _as_bytes(tensor: Tensor) -> Tensor:
    """The tensor's bytes as a 1-D ``uint8`` view (it must be contiguous)."""
    if not tensor.is_contiguous():
        raise ValueError("payload tensors must be contiguous")
    flat = tensor.reshape(-1)
    return flat if flat.dtype == torch.uint8 else flat.view(torch.uint8)


class LoopSampleDispatcher:
    """Transfers of one RSBF domain (see the module docstring).

    Args:
        groups: the domain and status groups.
        transport: ``"p2p"`` (``batch_isend_irecv`` on the non-empty pairs, default) or
            ``"all_to_all"`` (one ``all_to_all_single`` per phase over the whole domain).
        device: where the all-to-all staging buffers and the warm-up live (the domain group's
            device: CUDA for NCCL, CPU for Gloo).
        record_ops: keep a per-op log ``(phase, transfer order key, slot, peer, "send" |
            "recv")`` in ``op_log`` (tests).
    """

    def __init__(
        self,
        groups: RSBFGroups,
        *,
        transport: Transport = "p2p",
        device: torch.device,
        record_ops: bool = False,
    ):
        if transport not in ("p2p", "all_to_all"):
            raise ValueError(f"unknown move transport {transport!r}")
        self.groups = groups
        self.transport = transport
        self.device = torch.device(device)
        domain = groups.domain
        if domain is None or not dist.is_initialized():
            self.eu, self.num_eus = 0, 1
            self._ranks = [dist.get_rank() if dist.is_initialized() else 0]
        else:
            self.eu = dist.get_rank(domain)
            self.num_eus = dist.get_world_size(domain)
            self._ranks = [dist.get_global_rank(domain, e) for e in range(self.num_eus)]
        if groups.status is not None:
            status_ranks = sorted(dist.get_process_group_ranks(groups.status))
            if status_ranks != sorted(self._ranks):
                raise ValueError(
                    f"the RSBF status group's ranks {status_ranks} are not the domain's "
                    f"{sorted(self._ranks)} (a stale group?)"
                )
        self.stats = DispatcherStats()
        self.record_ops = record_ops
        self.op_log: list = []

    @property
    def ranks(self) -> list:
        """The global ranks of the domain's EUs, in EU order."""
        return list(self._ranks)

    # -----------------------------------------------------------------------------------
    # Moves
    # -----------------------------------------------------------------------------------

    def warm_up(self) -> None:
        """One all-rank ring exchange of one byte (NCCL's first ``batch_isend_irecv`` on a
        group must involve every rank). A no-op on one EU."""
        if self.num_eus == 1:
            return
        send = torch.full((1,), self.eu, dtype=torch.uint8, device=self.device)
        recv = torch.empty(1, dtype=torch.uint8, device=self.device)
        ops = [
            dist.P2POp(
                dist.isend, send, self._ranks[(self.eu + 1) % self.num_eus], self.groups.domain
            ),
            dist.P2POp(
                dist.irecv, recv, self._ranks[(self.eu - 1) % self.num_eus], self.groups.domain
            ),
        ]
        for work in dist.batch_isend_irecv(ops):
            work.wait()

    def exchange(
        self,
        phase: MovePhase,
        send: Callable[[Transfer], Sequence[Tensor]],
        recv: Callable[[Transfer], Sequence[Tensor]],
    ) -> dict:
        """Run one move phase.

        ``send(t)``: this EU's payload tensors of ``t`` (``t.src == eu``), one per slot, in
        slot order. ``recv(t)``: freshly allocated contiguous receive tensors of ``t``
        (``t.dst == eu``), one per slot, in slot order. Returns ``{t: received tensors}`` for
        this EU's incoming transfers once every op of the phase has completed (the sent
        tensors may be released then).
        """
        self.stats.phases += 1
        phase_id = self.stats.phases
        outgoing, incoming = [], []
        for t in phase.transfers:
            if not (0 <= t.src < self.num_eus and 0 <= t.dst < self.num_eus) or t.src == t.dst:
                raise ValueError(f"transfer {t} leaves the domain of {self.num_eus} EUs")
            if t.src == self.eu:
                tensors = list(send(t))
                self._check_count(t, tensors, "send")
                outgoing.append((t, tensors))
            if t.dst == self.eu:
                tensors = list(recv(t))
                self._check_count(t, tensors, "receive")
                incoming.append((t, tensors))
        if self.num_eus == 1:
            if outgoing or incoming:  # pragma: no cover - rejected above (src == dst)
                raise AssertionError("a one-EU domain has no transfers")
            return {}
        if self.transport == "p2p":
            self._p2p(phase_id, phase, outgoing, incoming)
        else:
            self._all_to_all(phase_id, outgoing, incoming)
        return {t: tensors for t, tensors in incoming}

    @staticmethod
    def _check_count(t: Transfer, tensors: list, what: str) -> None:
        if len(tensors) != len(t.slots):
            raise ValueError(
                f"{what} tensors of {t.kind} transfer of sample {t.sample}: {len(tensors)} for "
                f"slots {t.slots}"
            )

    def _log(self, phase_id: int, t: Transfer, slot: str, peer: int, direction: str) -> None:
        if self.record_ops:
            self.op_log.append((phase_id, t.order_key(), slot, peer, direction))

    def _p2p(self, phase_id: int, phase: MovePhase, outgoing: list, incoming: list) -> None:
        sends = dict(outgoing)
        recvs = dict(incoming)
        ops, keep = [], []
        group = self.groups.domain
        # Plan order: both ends of a pair post their ops in the same relative order.
        for t in phase.transfers:
            if t.src == self.eu:
                for slot, tensor in zip(t.slots, sends[t]):
                    data = _as_bytes(tensor.contiguous())
                    if data.numel() == 0:
                        continue
                    keep.append(data)
                    ops.append(dist.P2POp(dist.isend, data, self._ranks[t.dst], group))
                    self.stats.bytes_sent += data.numel()
                    self._log(phase_id, t, slot, t.dst, "send")
            if t.dst == self.eu:
                for slot, tensor in zip(t.slots, recvs[t]):
                    data = _as_bytes(tensor)
                    if data.numel() == 0:
                        continue
                    ops.append(dist.P2POp(dist.irecv, data, self._ranks[t.src], group))
                    self.stats.bytes_received += data.numel()
                    self._log(phase_id, t, slot, t.src, "recv")
        if not ops:
            return
        self.stats.p2p_ops += len(ops)
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        del keep

    def _all_to_all(self, phase_id: int, outgoing: list, incoming: list) -> None:
        W = self.num_eus
        send_splits, recv_splits = [0] * W, [0] * W
        chunks = []
        # Sends grouped by destination (ascending), transfers in plan order, slots in order.
        for dst in range(W):
            for t, tensors in outgoing:
                if t.dst != dst:
                    continue
                for slot, tensor in zip(t.slots, tensors):
                    data = _as_bytes(tensor.contiguous())
                    if data.numel():
                        chunks.append(data)
                        send_splits[dst] += data.numel()
                        self._log(phase_id, t, slot, dst, "send")
        targets = []
        for src in range(W):
            for t, tensors in incoming:
                if t.src != src:
                    continue
                for slot, tensor in zip(t.slots, tensors):
                    data = _as_bytes(tensor)
                    if data.numel():
                        targets.append(data)
                        recv_splits[src] += data.numel()
                        self._log(phase_id, t, slot, src, "recv")
        send_buf = (
            torch.cat([c.to(self.device) for c in chunks])
            if chunks
            else torch.empty(0, dtype=torch.uint8, device=self.device)
        )
        recv_buf = torch.empty(sum(recv_splits), dtype=torch.uint8, device=self.device)
        dist.all_to_all_single(
            recv_buf, send_buf, recv_splits, send_splits, group=self.groups.domain
        )
        self.stats.all_to_alls += 1
        self.stats.bytes_sent += send_buf.numel()
        self.stats.bytes_received += recv_buf.numel()
        offset = 0
        for data in targets:
            size = data.numel()
            data.copy_(recv_buf[offset : offset + size])
            offset += size

    # -----------------------------------------------------------------------------------
    # Status and debug
    # -----------------------------------------------------------------------------------

    def _status_group(self, what: str):
        if self.groups.status is None:
            raise ValueError(f"{what} over {self.num_eus} EUs needs an RSBF status group")
        return self.groups.status

    def exchange_status(self, bitmap: Tensor) -> Tensor:
        """The round's exit bitmap, reduced (MAX) over the domain: one Gloo all-reduce of the
        CPU ``uint8[G + 1]`` buffer (in place; returned). Byte ``G`` carries the EUs'
        after-exit violation flags. On one EU without a status group, the bitmap itself."""
        if bitmap.dtype != torch.uint8 or bitmap.device.type != "cpu" or bitmap.dim() != 1:
            raise ValueError("the status bitmap is a 1-D CPU uint8 tensor")
        if self.num_eus == 1 and self.groups.status is None:
            return bitmap
        group = self._status_group("the status exchange")
        dist.all_reduce(bitmap, op=dist.ReduceOp.MAX, group=group)
        self.stats.status_exchanges += 1
        return bitmap

    def gather_digests(self, digest: bytes) -> list:
        """Every EU's ``digest`` (equal length on every rank), in domain order, over the
        status group (debug: the plan-identity check). On one EU, ``[digest]``."""
        if self.num_eus == 1 and self.groups.status is None:
            return [digest]
        group = self._status_group("the plan digest check")
        local = torch.tensor(list(digest), dtype=torch.uint8)
        gathered = [torch.empty_like(local) for _ in range(self.num_eus)]
        dist.all_gather(gathered, local, group=group)
        self.stats.digest_checks += 1
        status_ranks = dist.get_process_group_ranks(group)
        return [bytes(gathered[status_ranks.index(rank)].tolist()) for rank in self._ranks]

    def check_identical(self, digest: bytes, what: str = "plan") -> None:
        """Raise on every rank when the EUs' digests differ (names the differing EUs)."""
        digests = self.gather_digests(digest)
        differing = [e for e, d in enumerate(digests) if d != digests[0]]
        if differing:
            raise RuntimeError(
                f"RSBF {what} differs across the domain: EUs {differing} disagree with EU 0"
            )
