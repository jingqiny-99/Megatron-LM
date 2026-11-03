# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The loop ring's schedule entry point end to end on CPU: one thread per pipeline rank.

Every thread runs the real ``ring_pp.LoopRingSchedule`` step, hence Megatron's unmodified
``forward_backward_pipelining_with_interleaving`` over the rank's round views and the real
``P2PCommunicator`` (``_communicate`` / ``_p2p_ops``, with their even / odd ordering and the
WORLD group at PP 2), on fake ring chunks. Only the transport and the collectives are replaced:

* ``Transport``: point-to-point with NCCL's ordering semantics. A ``(rank, group)`` pair is one
  in-order stream: its posted sends and receives complete in posting order. A send and a receive
  match by their order on their channel ``(sender, receiver, group)`` (no tags), and complete
  together once each is the oldest pending op of its own stream (a rendezvous). The data is
  copied when the pair completes, so a send buffer freed (pseudo-deallocated) before its send
  completed is reported. ``wait()`` blocks with a timeout: a deadlock fails the run.
* ``Collectives``: the ring schedule's small all-reduce / all-gather over the threads.

The fake chunks (``FakeChunk``) tag every activation and gradient with ``(stage, view,
microbatch, kind)`` and check what they receive: a forward input must come from the predecessor
view of the same microbatch (rank p - 1, or the last rank's previous view for the round wrap), a
gradient from the successor view. They record the calls, the recompute codes and the batches
they were served, and implement the ring chunk API the schedule uses (FLOPs, G1 counters,
injection keys).
"""

import collections
import contextlib
import functools
import threading
import traceback
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
from torch import nn

from megatron.core.pipeline_parallel import p2p_communication, schedules
from megatron.core.transformer.loop import ring_pp
from megatron.core.utils import get_attr_wrapped_model

FORWARD, BACKWARD = 1.0, 2.0
_local = threading.local()


def current_rank() -> int:
    """The global rank of the calling thread."""
    return _local.rank


class FakeGroup:
    """A process group seen from one rank: ``name`` (shared by its ranks), the global ``ranks``,
    ``rank()`` / ``size()``."""

    def __init__(self, name: str, ranks, global_rank: int):
        self.name = name
        self.ranks = tuple(ranks)
        self._rank = self.ranks.index(global_rank)

    def rank(self) -> int:
        """This rank's index in the group."""
        return self._rank

    def size(self) -> int:
        """The group's size."""
        return len(self.ranks)


WORLD = SimpleNamespace(name="world")


@dataclass(eq=False)
class _Op:
    kind: str  # "send" / "recv"
    rank: int
    group: str
    channel: tuple  # (sender, receiver, group)
    index: int
    tensor: torch.Tensor
    done: bool = False


class Transport:
    """Point-to-point with NCCL's stream ordering (module docstring)."""

    def __init__(self, timeout: float):
        self.cond = threading.Condition()
        self.timeout = timeout
        self.streams = collections.defaultdict(collections.deque)  # (rank, group) -> ops
        self.counts = collections.Counter()  # (channel, kind) -> ops posted
        self.ops: Dict[tuple, _Op] = {}
        self.error: Optional[str] = None
        self.pairs = 0

    def post(self, kind: str, peer: int, group, tensor: torch.Tensor) -> "_Request":
        rank = current_rank()
        channel = (rank, peer, group.name) if kind == "send" else (peer, rank, group.name)
        with self.cond:
            index = self.counts[(channel, kind)]
            self.counts[(channel, kind)] += 1
            op = _Op(kind, rank, group.name, channel, index, tensor)
            self.streams[(rank, group.name)].append(op)
            self.ops[(channel, kind, index)] = op
            self._progress()
        return _Request(self, op)

    def _progress(self) -> None:
        changed = True
        while changed and self.error is None:
            changed = False
            for stream in list(self.streams.values()):
                if not stream:
                    continue
                head = stream[0]
                other = "recv" if head.kind == "send" else "send"
                peer = self.ops.get((head.channel, other, head.index))
                if peer is None or peer.done:
                    continue
                peer_stream = self.streams[(peer.rank, peer.group)]
                if not peer_stream or peer_stream[0] is not peer:
                    continue
                send, recv = (head, peer) if head.kind == "send" else (peer, head)
                if send.tensor.shape != recv.tensor.shape:
                    self.error = (
                        f"P2P on {send.channel} #{send.index}: send buffer of shape "
                        f"{tuple(send.tensor.shape)} for a receive of {tuple(recv.tensor.shape)} "
                        "(freed before the send completed?)"
                    )
                    break
                with torch.no_grad():
                    recv.tensor.copy_(send.tensor)
                send.done = recv.done = True
                stream.popleft()
                peer_stream.popleft()
                self.pairs += 1
                changed = True
        self.cond.notify_all()

    def wait(self, op: _Op) -> None:
        with self.cond:
            finished = self.cond.wait_for(lambda: op.done or self.error, timeout=self.timeout)
            if self.error:
                raise RuntimeError(self.error)
            if not finished:
                heads = {
                    key: (stream[0].kind, stream[0].channel, stream[0].index)
                    for key, stream in self.streams.items()
                    if stream
                }
                self.error = f"P2P deadlock: rank {op.rank} waits on {op}; stream heads {heads}"
                self.cond.notify_all()
                raise RuntimeError(self.error)

    def abort(self, message: str) -> None:
        with self.cond:
            self.error = self.error or message
            self.cond.notify_all()


@dataclass
class _Request:
    transport: Transport
    op: _Op

    def wait(self) -> None:
        self.transport.wait(self.op)


class FakeDist:
    """``torch.distributed`` as ``p2p_communication`` uses it, on a ``Transport``."""

    def __init__(self, transport: Transport):
        self.transport = transport
        self.group = SimpleNamespace(WORLD=WORLD)

    def isend(self, tensor, dst, group):
        return self.transport.post("send", dst, group, tensor)

    def irecv(self, tensor, src, group):
        return self.transport.post("recv", src, group, tensor)

    @staticmethod
    def get_backend(group):
        del group
        return "nccl"  # the NCCL path of _p2p_ops (WORLD for one direction at PP 2)

    @staticmethod
    def get_global_rank(group, rank):
        return group.ranks[rank]


class Collectives:
    """The ring schedule's collectives over the threads (``ring_pp.dist``)."""

    ReduceOp = SimpleNamespace(SUM="sum", MIN="min", MAX="max")

    def __init__(self, transport: Transport, timeout: float):
        self.transport = transport
        self.timeout = timeout
        self.cond = threading.Condition()
        self.slots: Dict[tuple, dict] = {}
        self.calls = collections.Counter()

    @staticmethod
    def get_backend(group):
        del group
        return "gloo"

    def _exchange(self, group, tensor) -> List[torch.Tensor]:
        rank = current_rank()
        key = (group.name, group.ranks)
        with self.cond:
            index = self.calls[(key, rank)]
            self.calls[(key, rank)] += 1
            slot = self.slots.setdefault((key, index), {})
            slot[group.rank()] = tensor.detach().clone()
            self.cond.notify_all()
            if not self.cond.wait_for(lambda: len(slot) == group.size(), timeout=self.timeout):
                self.transport.abort(f"collective {key} #{index} timed out")
                raise RuntimeError(f"collective {key} #{index} timed out")
            return [slot[member] for member in range(group.size())]

    def all_reduce(self, tensor, op="sum", group=None):
        parts = torch.stack(self._exchange(group, tensor))
        reduced = {"sum": parts.sum(0), "min": parts.min(0).values, "max": parts.max(0).values}[op]
        tensor.copy_(reduced)

    def all_gather(self, tensor_list, tensor, group=None):
        for out, part in zip(tensor_list, self._exchange(group, tensor)):
            out.copy_(part)


class _TorchProxy:
    """A module stand-in: ``overrides`` first, then ``torch``."""

    def __init__(self, **overrides):
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(torch, name)


def _zeros_on_cpu(*args, device=None, **kwargs):
    return torch.zeros(*args, device="cpu" if device == "cuda" else device, **kwargs)


@contextlib.contextmanager
def fake_runtime(monkeypatch, timeout: float = 60.0):
    """Install the fake transport and collectives (module docstring); yields ``(transport,
    collectives)``."""
    transport = Transport(timeout)
    collectives = Collectives(transport, timeout)
    fake_dist = FakeDist(transport)
    cuda = SimpleNamespace(
        current_device=lambda: "cpu",
        is_current_stream_capturing=lambda: False,
        synchronize=lambda: None,
    )
    monkeypatch.setattr(p2p_communication, "dist", fake_dist)
    monkeypatch.setattr(p2p_communication, "torch", _TorchProxy(distributed=fake_dist, cuda=cuda))
    monkeypatch.setattr(schedules, "torch", _TorchProxy(zeros=_zeros_on_cpu))
    monkeypatch.setattr(ring_pp, "dist", collectives)
    yield transport, collectives


# ---------------------------------------------------------------------------------------------
# Fake ring chunks
# ---------------------------------------------------------------------------------------------


class _Tag(torch.autograd.Function):
    """Identity that tags its output ``(stage, view, microbatch, FORWARD)`` and its input gradient
    ``(..., BACKWARD)``, checking the gradient it receives against ``expect`` (the successor's
    tag; None for the final view, whose gradient comes from the loss)."""

    @staticmethod
    def forward(ctx, x, tag, expect, log):
        ctx.tag, ctx.expect, ctx.log = tag, expect, log
        out = x.clone()
        out[0, 0, :4] = torch.tensor(tag + (FORWARD,), dtype=out.dtype)
        return out

    @staticmethod
    def backward(ctx, grad):
        got = tuple(float(value) for value in grad.detach()[0, 0, :4])
        if ctx.expect is not None and got != ctx.expect + (BACKWARD,):
            raise RuntimeError(f"view {ctx.tag}: gradient tagged {got}, expected {ctx.expect}")
        ctx.log.append(("B",) + ctx.tag)
        out = grad.clone()
        out[0, 0, :4] = torch.tensor(ctx.tag + (BACKWARD,), dtype=out.dtype)
        return out, None, None, None


class FakeDecoder(nn.Module):
    """The ring chunk API ``LoopRingSchedule`` and ``LoopRingRoundView`` use: ``loop_ring``,
    ``config``, the ``call_roles`` slot, FLOPs, G1 counters, an always-empty injection store."""

    def __init__(self, config, *, offset, round_start, round_end, unit):
        super().__init__()
        self.config = config
        self.loop_ring = SimpleNamespace(
            offset=offset, round_start=round_start, round_end=round_end
        )
        self.pre_process, self.post_process = round_start, round_end
        self.call_roles = None
        self.layers = []
        self.unit = unit
        self.model_work = {}
        self._work = [0.0, 0.0, 0.0]
        self._counts = {}

    def call_work(self, round_index, code):
        """The forward FLOPs of one call: the chunk's unit, the prelude in round 1 of the
        round-start chunk, the coda in round R of the round-end chunk."""
        work = self.unit
        if self.loop_ring.round_start and round_index == 0:
            work += 0.5
        if self.loop_ring.round_end and round_index == self.config.loop_iterations - 1:
            work += 0.75
        return work, code == "C"

    def ring_microbatch_work(self, sequence, batch, codes):
        del sequence, batch
        total = [0.0, 0.0, 0.0]
        for round_index, code in enumerate(codes):
            work, recompute = self.call_work(round_index, code)
            total[0] += work
            total[1] += 2 * work
            total[2] += work if recompute else 0.0
        return tuple(total)

    def record(self, round_index, microbatch, checkpoint, *, pre, post):
        work, _ = self.call_work(round_index, "C" if checkpoint else "n")
        self._work[0] += work
        if torch.is_grad_enabled():
            self._work[1] += 2 * work
            self._work[2] += work if checkpoint else 0.0
        counts = self._counts.setdefault(microbatch, [0, 0, 0, 0])
        counts[0] += 1
        counts[1] += int(pre)
        counts[2] += int(post)

    def set_ring_model_work(self, sequence, batch, work):
        self.model_work[(sequence, batch)] = tuple(work)

    def take_ring_work(self):
        work, self._work = tuple(self._work), [0.0, 0.0, 0.0]
        return work

    def take_ring_counts(self):
        counts, self._counts = self._counts, {}
        return counts

    def ring_injection_keys(self):
        return ()


class FakeChunk(nn.Module):
    """One physical ring chunk (``HybridModel``'s place): tags and checks the ring's tensors
    (module docstring)."""

    def __init__(self, config, *, stage, stages, chunk, chunks, hidden, log):
        super().__init__()
        segments = stages * chunks
        segment = chunk * stages + stage
        self.decoder = FakeDecoder(
            config,
            offset=segment,
            round_start=segment == 0,
            round_end=segment == segments - 1,
            unit=1.0 + 0.25 * segment,
        )
        self.config = config
        self.pre_process = self.decoder.pre_process
        self.post_process = self.decoder.post_process
        self.mtp_process = False
        self.vp_stage = chunk if chunks > 1 else None
        self.model_type = None
        self.weight = nn.Parameter(torch.ones(()))
        self.stage, self.stages, self.chunk, self.chunks = stage, stages, chunk, chunks
        self.hidden = hidden
        self.log = log
        self.input_tensor = None
        self.served: List[tuple] = []  # (round, microbatch, batch["mb"])
        self.codes: List[tuple] = []  # (view, checkpoint)

    @property
    def loads_data(self) -> bool:
        """``get_batch``'s rule: tokens on the first chunk, labels on the last one."""
        return self.pre_process or self.post_process

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor[0] if isinstance(input_tensor, list) else input_tensor

    def _expect(self, view, microbatch, forward):
        """The tag the ring delivers to view ``view`` of this rank (None: none)."""
        rounds = self.config.loop_iterations
        views = rounds * self.chunks
        if forward:
            if self.stage > 0:
                return float(self.stage - 1), float(view), float(microbatch)
            return (float(self.stages - 1), float(view - 1), float(microbatch)) if view else None
        if self.stage < self.stages - 1:
            return float(self.stage + 1), float(view), float(microbatch)
        return (0.0, float(view + 1), float(microbatch)) if view < views - 1 else None

    def forward(self, batch, sequence, micro_batch):
        roles = self.decoder.call_roles
        view = roles.round * self.chunks + self.chunk
        microbatch = roles.microbatch
        self.codes.append((view, roles.checkpoint))
        if self.loads_data:
            self.served.append((roles.round, microbatch, batch["mb"]))
        if roles.pre_process:
            base = torch.full((sequence, micro_batch, self.hidden), float(batch["mb"]) + 1.0)
            x = base * self.weight
        else:
            x = self.input_tensor
            expect = self._expect(view, microbatch, forward=True)
            got = tuple(float(value) for value in x.detach()[0, 0, :4])
            if got != expect + (FORWARD,):
                raise RuntimeError(
                    f"rank {self.stage} view {view} mb {microbatch}: input tagged {got}, "
                    f"expected {expect}"
                )
            x = x * self.weight
        self.decoder.record(
            roles.round,
            microbatch,
            roles.checkpoint,
            pre=roles.pre_process,
            post=roles.post_process,
        )
        self.log.append(("F", float(self.stage), float(view), float(microbatch)))
        expect = None if roles.post_process else self._expect(view, microbatch, forward=False)
        out = _Tag.apply(x, (float(self.stage), float(view), float(microbatch)), expect, self.log)
        if roles.post_process:
            return out.float().mean(dim=-1).transpose(0, 1)  # [b, s] per-token "loss"
        return out


class CountingIterator:
    """A data source: dicts ``{"mb": i}``, counting ``next()`` calls."""

    def __init__(self):
        self.pulled = 0

    def __iter__(self):
        return self

    def __next__(self):
        item = {"mb": self.pulled}
        self.pulled += 1
        return item


def _loss_func(output_tensor):
    loss = output_tensor.float().sum()
    tokens = torch.tensor(output_tensor.numel(), dtype=torch.int)
    return loss, tokens, {"lm loss": torch.stack((loss.detach(), tokens.float()))}


def forward_step(data_iterator, model, *, sequence, micro_batch):
    """``pretrain_hybrid.forward_step``'s shape: a batch only on data-loading chunks."""
    chunk = get_attr_wrapped_model(model, "decoder", return_model_obj=True)
    batch = next(data_iterator) if chunk.loads_data else None
    return model(batch, sequence, micro_batch), _loss_func


@dataclass
class RankResult:
    """What one rank's thread produced."""

    record: object = None
    chunks: List[FakeChunk] = field(default_factory=list)
    sources: List[Optional[CountingIterator]] = field(default_factory=list)
    finalized: List[tuple] = field(default_factory=list)
    output: list = field(default_factory=list)
    log: list = field(default_factory=list)


def run_ring_threads(
    monkeypatch,
    config_factory,
    *,
    stages,
    chunks,
    microbatches,
    sequence=4,
    micro_batch=2,
    forward_only=False,
    finalize_wrapper=None,
    force_all_reduce=False,
    timeout=60.0,
) -> List[RankResult]:
    """One ring step on ``stages`` threads (module docstring). ``config_factory(stage)`` builds a
    rank's model config; ``finalize_wrapper(config, record)`` may wrap the config's recording
    finalize function as ``train_step`` does; ``force_all_reduce`` goes to the schedule. Returns
    every rank's ``RankResult``."""
    results = [RankResult() for _ in range(stages)]
    errors: List[Optional[str]] = [None] * stages
    with fake_runtime(monkeypatch, timeout=timeout) as (transport, _):

        def worker(stage):
            _local.rank = stage
            result = results[stage]
            try:
                config = config_factory(stage)
                hidden = schedules._get_pipeline_hidden_size(config)

                def finalize(model, num_tokens, pg_collection=None, force_all_reduce=False):
                    del pg_collection
                    result.finalized.append(
                        ("finalize", tuple(id(chunk) for chunk in model), force_all_reduce)
                    )

                config.finalize_model_grads_func = finalize
                if finalize_wrapper is not None:
                    finalize_wrapper(config, result.finalized)
                result.chunks = [
                    FakeChunk(
                        config,
                        stage=stage,
                        stages=stages,
                        chunk=chunk,
                        chunks=chunks,
                        hidden=hidden,
                        log=result.log,
                    )
                    for chunk in range(chunks)
                ]
                result.sources = [
                    CountingIterator() if chunk.loads_data else None for chunk in result.chunks
                ]
                pg_collection = SimpleNamespace(
                    pp=FakeGroup("pp", range(stages), stage),
                    tp=FakeGroup(f"tp{stage}", [stage], stage),
                    cp=FakeGroup(f"cp{stage}", [stage], stage),
                    tp_dp_cp=FakeGroup(f"tdc{stage}", [stage], stage),
                )
                schedule = ring_pp.LoopRingSchedule()
                with torch.no_grad() if forward_only else contextlib.nullcontext():
                    result.output = schedule(
                        forward_step_func=functools.partial(
                            forward_step, sequence=sequence, micro_batch=micro_batch
                        ),
                        data_iterator=(result.sources if chunks > 1 else result.sources[0]),
                        model=result.chunks,
                        num_microbatches=microbatches,
                        seq_length=sequence,
                        micro_batch_size=micro_batch,
                        forward_only=forward_only,
                        pg_collection=pg_collection,
                        force_all_reduce=force_all_reduce,
                    )
                result.record = schedule.last_step
            except BaseException:  # pylint: disable=broad-except
                errors[stage] = traceback.format_exc()
                transport.abort(f"rank {stage} failed")

        threads = [threading.Thread(target=worker, args=(stage,)) for stage in range(stages)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=4 * timeout)
        if any(thread.is_alive() for thread in threads):
            transport.abort("threads still alive")
            raise RuntimeError("ring threads did not finish")
    _raise_first(errors)
    return results


def _raise_first(errors: List[Optional[str]]) -> None:
    failed = [(stage, error) for stage, error in enumerate(errors) if error]
    if failed:
        # The first failure is the cause; the others usually abort on it.
        primary = next(
            ((stage, error) for stage, error in failed if "failed" not in error.splitlines()[-1]),
            failed[0],
        )
        raise RuntimeError(f"rank {primary[0]} failed:\n{primary[1]}")


# ---------------------------------------------------------------------------------------------
# Real ring chunks (the CPU emulation's local-layer HybridModels)
# ---------------------------------------------------------------------------------------------


class RealChunk(nn.Module):
    """A DDP stand-in around one real ring chunk (the ``.module`` chain and the model config)."""

    def __init__(self, module, config):
        super().__init__()
        self.module = module
        self.config = config

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def real_forward_step(data_iterator, model, loss_func):
    """``pretrain_hybrid.forward_step`` for the emulation's batches: a batch only on the
    data-loading chunks (tokens on the first, everything on the last), the loop metrics
    captured with the loss closure (``loss_func(loss_mask, metrics, output)``)."""
    hybrid = get_attr_wrapped_model(model, "decoder", return_model_obj=True)
    batch = next(data_iterator) if (hybrid.pre_process or hybrid.post_process) else {}
    if hybrid.pre_process and not hybrid.post_process:
        batch = {"tokens": batch["tokens"], "position_ids": batch["position_ids"]}
    output = model(
        batch.get("tokens"),
        batch.get("position_ids"),
        None,
        labels=batch.get("labels"),
        loss_mask=batch.get("loss_mask"),
    )
    return output, functools.partial(loss_func, batch.get("loss_mask"), hybrid.decoder.loop_metrics)


def run_real_ring_threads(
    monkeypatch, ring, batches, loss_func, *, sequence, micro_batch, timeout=60.0
) -> Dict[int, tuple]:
    """One step of the real ``LoopRingSchedule`` on the emulation's ring chunks (``ring``:
    ``ring_emulation.build_ring``), one thread per pipeline rank (module docstring). Every rank's
    data sources replay ``batches``. Returns ``{stage: (last_step, schedule output, finalize
    calls)}``."""
    stages, chunks = ring.stages, ring.chunks
    finalized: Dict[int, list] = collections.defaultdict(list)
    lock = threading.Lock()

    def finalize(model, num_tokens, pg_collection=None, force_all_reduce=False):
        del num_tokens, pg_collection, force_all_reduce
        with lock:
            finalized[current_rank()].append(len(model))

    ring.config.finalize_model_grads_func = finalize
    ring.config.deallocate_pipeline_outputs = True
    results: Dict[int, tuple] = {}
    errors: List[Optional[str]] = [None] * stages
    with fake_runtime(monkeypatch, timeout=timeout) as (transport, _):

        def worker(stage):
            _local.rank = stage
            try:
                wrapped = [
                    RealChunk(ring.chunk_model(stage, chunk), ring.config)
                    for chunk in range(chunks)
                ]
                sources = [
                    (
                        iter([dict(batch) for batch in batches])
                        if model.pre_process or model.post_process
                        else None
                    )
                    for model in (ring.chunk_model(stage, chunk) for chunk in range(chunks))
                ]
                collection = SimpleNamespace(
                    pp=FakeGroup("pp", range(stages), stage),
                    tp=FakeGroup(f"tp{stage}", [stage], stage),
                    cp=FakeGroup(f"cp{stage}", [stage], stage),
                    tp_dp_cp=FakeGroup(f"tdc{stage}", [stage], stage),
                )
                schedule = ring_pp.LoopRingSchedule()
                output = schedule(
                    forward_step_func=functools.partial(real_forward_step, loss_func=loss_func),
                    data_iterator=sources if chunks > 1 else sources[0],
                    model=wrapped,
                    num_microbatches=len(batches),
                    seq_length=sequence,
                    micro_batch_size=micro_batch,
                    pg_collection=collection,
                )
                results[stage] = (schedule.last_step, output, list(finalized[stage]))
            except BaseException:  # pylint: disable=broad-except
                errors[stage] = traceback.format_exc()
                transport.abort(f"rank {stage} failed")

        threads = [threading.Thread(target=worker, args=(stage,)) for stage in range(stages)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=4 * timeout)
        if any(thread.is_alive() for thread in threads):
            transport.abort("threads still alive")
            raise RuntimeError("ring threads did not finish")
    _raise_first(errors)
    return results
