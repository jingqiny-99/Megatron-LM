# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The loop ring: the hybrid Huginn loop over pipeline stages (``loop_schedule='ring'``).

Layout (``guards.validate_ring_layout``). The main pattern carries P x k ``'|'`` segments of the
unchanged layer sequence. Pipeline rank p builds its k chunks as ordinary stage modules
(segment ``j * P + p`` is its chunk j): the round-start chunk (segment 0) holds the embedding,
the prelude, the first core slice and the loop adapter; the round-end chunk (the last segment)
the last core slice, the loop norm, the coda, MTP and the output layer. Every parameter lives
once, on one chunk.

Execution. Megatron's interleaved schedule runs ``V = R x k`` round views per rank: the
schedule's model chunk ``v = r * k + j`` is ``LoopRingRoundView`` of physical chunk j in round r.
The schedule sends the last stage's chunk v output to the first stage's chunk v + 1, so a
microbatch passes the stage ring once per round. One call of chunk j in round r runs exactly its
slice of the PP = 1 Huginn loop (``LoopHybridStack``'s ring branch): on the round-start chunk in
round 1 the embedding (with the mHC stream expansion), the prelude and the injection store; on
the round-start chunk the adapter ``[state, e]``; the chunk's core slice; on the round-end chunk
the loop norm, and in round R the coda, the mHC contraction, MTP, the output layer and the loss.

This module holds what such a call needs besides the stack:

* ``LoopRingCall``: one call's roles. ``HybridStackCallRoles`` (``pre_process`` = the embedding
  on the round-start chunk in round 1, ``post_process`` / ``mtp_process`` = the head on the
  round-end chunk in round R; the built flags are never changed) plus the round, the microbatch
  and whether this view's recompute code is 'C'. ``loop_ring_call`` sets it on the chunk's
  decoder (``HybridStack.call_roles``) around one call.
* The injection (Huginn's ``e``, the prelude output, an input of every round's adapter) stays on
  the round-start chunk, in a store keyed by microbatch (``RingInjection``). Round 1 uses ``e``
  through ``_RingInjectionSource``, rounds 2..R a detached leaf of it through
  ``_RingInjectionUse``, whose backward adds the incoming gradient into the entry's sink. The
  schedule runs a microbatch's round-start backwards as round R, R - 1, ..., 1 (data
  dependencies), so the sink holds ``(g_R + ...) + g_2`` when round 1's backward returns
  ``g_1 + sink``: ``e``'s producer gets the sum of the four rounds once, equal bit for bit to
  the PP = 1 sum ``((g_4 + g_3) + g_2) + g_1`` (IEEE addition commutes). The round-1 backward
  releases the entry; without autograd the round-R call does.
* ``LoopRingRoundView``: the schedule's model chunk.
* ``ring_sample_counts``: the G1 counters of several chunks per microbatch.

The schedule entry point (``forward_backward_loop_ring``, ``LoopRingSchedule``; selected by
``megatron/training/training.py``'s ``get_loop_ring_forward_backward_func``) runs Megatron's
unmodified ``forward_backward_pipelining_with_interleaving`` once per step over the V = R x k
round views of the rank's k physical chunks:

* a per-step shallow copy of the model config (the *ring config*) with the virtual pipeline size
  V and the finalize wrapper, and a ``P2PCommunicator`` built from it (its virtual pipeline size
  sets the interleaved warmup count); the physical chunks keep the model config (``vp_stage``
  None at k = 1, j at k > 1), so ``Float16Module``, MTP placement and data loading behave as in
  ordinary PP / VPP;
* round views without ``vp_stage`` (the wrapped one is the physical index) and replay data
  iterators, one cache per physical chunk: the chunk's R views read microbatch m of the chunk's
  iterator, which is consumed exactly M times per step (``RingReplaySource``);
* the finalize wrapper (``RingFinalize``): the schedule's finalize call on the views runs the
  model config's finalize function, looked up when it is called (``train_step`` may have wrapped
  it, e.g. the chunked optimizer-state offload prefetch), once, on the k physical chunks;
* the model's FLOPs per microbatch (the pipeline sum of the chunks' analytic work) set on the
  round-end chunk before the schedule, and the step-end invariants after it: every view called M
  times, every replay consumed M times by every view, the injection stores empty, the call roles
  cleared, the built flags intact, the finalize run once, each chunk's recorded FLOPs M x its
  analytic work, and the G1 counters R / 1 / 1 / 1 per sample (core runs per chunk; prelude,
  coda and MTP once), reduced MIN / MAX over the pipeline and tensor x data ranks into the step's
  ``LoopRingStepRecord``.
"""

import contextlib
import copy
import math
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn

from megatron.core.models.hybrid.hybrid_block import HybridStackCallRoles
from megatron.core.utils import get_attr_wrapped_model, get_model_config


@dataclass(frozen=True)
class LoopRingCall(HybridStackCallRoles):
    """The roles of one ring chunk call (one round view's call of its physical chunk).

    Besides the pipeline roles (``HybridStackCallRoles``): ``round``, the 0-based round r;
    ``microbatch``, the microbatch the call runs (the injection store's key); ``checkpoint``,
    whether the view's recompute code is 'C' (``loop_ring_recompute_plan``): the round body (and,
    in round 1 on the round-start chunk, the prelude) is then an activation checkpoint.
    """

    round: int = 0
    microbatch: int = 0
    checkpoint: bool = False


@contextlib.contextmanager
def loop_ring_call(stack, call: LoopRingCall) -> Iterator[None]:
    """Set ``stack.call_roles = call`` for one call of a ring chunk and clear it on exit, also
    when the call raises (the built flags are never touched)."""
    if stack.call_roles is not None:
        raise RuntimeError(
            "A ring chunk call started inside another call of the same chunk "
            f"(roles {stack.call_roles})"
        )
    stack.call_roles = call
    try:
        yield
    finally:
        stack.call_roles = None


class RingInjection:
    """The injection (prelude output) of one microbatch, stored on the round-start chunk.

    The microbatch's round-1 call creates it under ``store[key]``; its rounds 2..R read
    ``tensor``. With ``grad_mode`` (autograd on and the prelude output requiring grad),
    ``tensor`` is a detached leaf of the prelude output (shared storage, no copy) and ``total``
    the sink: the sum, in arrival order, of the gradients rounds 2..R send it
    (``_RingInjectionUse``); round 1's backward (``_RingInjectionSource``) consumes it and
    releases the entry. Without ``grad_mode`` the round-R call releases it.
    """

    __slots__ = ("store", "key", "grad_mode", "tensor", "total")

    def __init__(self, store: Dict[int, "RingInjection"], key: int, embedding: Tensor):
        if key in store:
            raise RuntimeError(f"Ring injection store already holds microbatch {key}")
        self.store = store
        self.key = key
        self.grad_mode = torch.is_grad_enabled() and embedding.requires_grad
        self.tensor = embedding.detach().requires_grad_() if self.grad_mode else embedding
        self.total: Optional[Tensor] = None
        store[key] = self

    def add(self, gradient: Tensor) -> None:
        """Add one round's gradient into the sink (in ``tensor``'s dtype; the first one is
        copied, so the sink owns its buffer)."""
        self.total = gradient.clone() if self.total is None else self.total + gradient

    def release(self) -> None:
        """Remove the entry from its store and drop its tensors."""
        if self.store.get(self.key) is self:
            del self.store[self.key]
        self.tensor = None
        self.total = None


class _RingInjectionSource(torch.autograd.Function):
    """Round 1's injection on the round-start chunk: the identity (a view) on the prelude output.

    Its backward returns round 1's gradient plus the sink (the gradients rounds 2..R sent while
    their backwards ran, which the schedule's data dependencies order before round 1's) and
    releases the store entry, so the prelude output's producer gets one gradient, the sum over
    all rounds.
    """

    @staticmethod
    def forward(ctx, embedding: Tensor, entry: RingInjection) -> Tensor:
        """The identity (a view of the prelude output); keeps the store entry."""
        ctx.entry = entry
        return embedding.view_as(embedding)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        """Round 1's gradient plus the sink; releases the store entry."""
        entry, ctx.entry = ctx.entry, None
        total = entry.total
        entry.release()
        return (gradient if total is None else gradient + total), None


class _RingInjectionUse(torch.autograd.Function):
    """The injection of rounds 2..R on the round-start chunk: the identity (a view) on the stored
    detached leaf. Its backward adds the incoming gradient into the entry's sink and returns
    none, so the leaf never accumulates a gradient and the backward of a later round stops here
    (it never reaches the prelude, which runs once)."""

    @staticmethod
    def forward(ctx, leaf: Tensor, entry: RingInjection) -> Tensor:
        """The identity (a view of the stored leaf); keeps the store entry."""
        ctx.entry = entry
        return leaf.view_as(leaf)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        """Add this round's gradient into the entry's sink; no gradient for the leaf."""
        ctx.entry.add(gradient)
        return None, None


def ring_injection_source(embedding: Tensor, entry: RingInjection) -> Tensor:
    """Round 1's injection: ``_RingInjectionSource`` on the prelude output when the entry tracks
    gradients, else the prelude output itself."""
    if entry.grad_mode:
        return _RingInjectionSource.apply(embedding, entry)
    return embedding


def ring_injection_use(entry: RingInjection) -> Tensor:
    """The injection of a later round: ``_RingInjectionUse`` on the stored leaf when the entry
    tracks gradients, else the stored prelude output."""
    if entry.grad_mode != torch.is_grad_enabled():
        raise RuntimeError(
            f"Ring injection of microbatch {entry.key}: autograd was "
            f"{'on' if entry.grad_mode else 'off'} in round 1 but not in a later round"
        )
    if entry.grad_mode:
        return _RingInjectionUse.apply(entry.tensor, entry)
    return entry.tensor


def _wraps_float16_module(module: nn.Module) -> bool:
    """True if the ``.module`` chain from ``module`` passes a ``Float16Module``."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer.module import Float16Module

    while True:
        if isinstance(module, Float16Module):
            return True
        if not hasattr(module, "module"):
            return False
        module = module.module


class LoopRingRoundView(nn.Module):
    """Round view (r, j) of a ring stage: the interleaved schedule's model chunk ``v = r * k + j``.

    It runs physical chunk j (``module``, the stage's DDP-wrapped ``HybridModel``) for round r.
    ``module`` is an unregistered attribute, so ``parameters()`` is empty: optimizers,
    checkpoints and parameter counts see the physical chunks only. Everything a schedule looks up
    through ``get_attr_wrapped_model`` (``set_input_tensor``, ``decoder``, ``pg_collection``,
    ``model_type``, the physical ``vp_stage``) resolves through the ``.module`` walk; the view
    itself has ``config`` (the caller's, e.g. the per-step ring config whose virtual pipeline
    size is V) and no ``vp_stage``, ``set_is_first_microbatch`` or parameter.

    Each call sets the decoder's ``LoopRingCall`` (``loop_ring_call``): the embedding only in
    round 1 of a built ``pre_process`` chunk, the head (output layer, loss, MTP) only in round R
    of a built ``post_process`` chunk. Through a ``Float16Module`` the view passes
    ``fp32_output`` only on the final view (v = V - 1): the round-end chunk's earlier rounds
    stay in the model dtype on the ring. The i-th call since ``reset_calls`` must run microbatch
    i (the schedule table lists every chunk's microbatches in order); the decoder's
    ``current_microbatch`` (set by the schedule) is checked against it.
    """

    def __init__(
        self,
        module: nn.Module,
        config,
        *,
        round_index: int,
        chunk_index: int,
        chunks: int,
        checkpoint: bool,
    ):
        super().__init__()
        object.__setattr__(self, "module", module)
        self.config = config
        model = get_attr_wrapped_model(module, "decoder", return_model_obj=True)
        decoder = model.decoder
        if getattr(decoder, "loop_ring", None) is None:
            raise ValueError("LoopRingRoundView needs a ring chunk (loop_schedule='ring')")
        rounds = decoder.config.loop_iterations
        if not (0 <= round_index < rounds and 0 <= chunk_index < chunks):
            raise ValueError(
                f"Round view ({round_index}, {chunk_index}) outside {rounds} rounds x "
                f"{chunks} chunks"
            )
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_decoder", decoder)
        self.round_index = round_index
        self.chunk_index = chunk_index
        self.rounds = rounds
        self.view_index = round_index * chunks + chunk_index
        self.views = rounds * chunks
        self.checkpoint = bool(checkpoint)
        self._float16 = _wraps_float16_module(module)
        self.calls = 0
        if getattr(model, "mtp_process", False):
            # G1: the MTP block's own runs, counted on the decoder per microbatch.
            decoder.watch_ring_mtp(model.mtp)

    def reset_calls(self) -> None:
        """Start a new step: the next call runs microbatch 0."""
        self.calls = 0

    def roles_for(self, microbatch: int) -> LoopRingCall:
        """The ``LoopRingCall`` of this view's call on ``microbatch``."""
        last = self.round_index == self.rounds - 1
        model = self._model
        return LoopRingCall(
            pre_process=bool(model.pre_process) and self.round_index == 0,
            post_process=bool(model.post_process) and last,
            mtp_process=bool(getattr(model, "mtp_process", False)) and last,
            round=self.round_index,
            microbatch=microbatch,
            checkpoint=self.checkpoint,
        )

    def forward(self, *args, **kwargs):
        """Run the physical chunk for this round on the next microbatch (see the class)."""
        decoder = self._decoder
        microbatch = self.calls
        current = getattr(decoder, "current_microbatch", None)
        if current is not None and current != microbatch:
            raise RuntimeError(
                f"Ring round view {self.view_index}: call {self.calls} of the step got microbatch "
                f"{current} (the schedule runs every chunk's microbatches in order)"
            )
        if self._float16:
            kwargs["fp32_output"] = self.view_index == self.views - 1
        with loop_ring_call(decoder, self.roles_for(microbatch)):
            output = self.module(*args, **kwargs)
        self.calls += 1
        return output


def ring_sample_counts(
    chunk_counts: Sequence[Mapping[int, Sequence[int]]],
) -> Dict[int, Tuple[int, int, int, int, int]]:
    """The G1 counters of several ring chunks per microbatch.

    ``chunk_counts``: each chunk's ``LoopHybridStack.take_ring_counts()`` (per microbatch: core
    slice runs, prelude runs, coda runs, MTP runs). Returns, per microbatch, ``(fewest core runs
    of a chunk, most core runs of a chunk, prelude runs, coda runs, MTP runs)``, the last three
    summed over the chunks; a chunk that did not run a microbatch counts 0 core runs for it.
    Over every chunk of the model, a fixed-R ring gives ``(R, R, 1, 1, 1)`` (MTP 0 without MTP).
    """
    microbatches = sorted({key for counts in chunk_counts for key in counts})
    summary = {}
    for microbatch in microbatches:
        rows = [counts.get(microbatch, (0, 0, 0, 0)) for counts in chunk_counts]
        cores = [row[0] for row in rows]
        summary[microbatch] = (
            min(cores),
            max(cores),
            sum(row[1] for row in rows),
            sum(row[2] for row in rows),
            sum(row[3] for row in rows),
        )
    return summary


# ---------------------------------------------------------------------------------------------
# The schedule entry point: Megatron's interleaved schedule over the round views
# ---------------------------------------------------------------------------------------------


def _replay_copy(item: Any) -> Any:
    """What one round view receives of a replayed batch: a dict shallowly copied (the data path
    rebinds its entries, e.g. ``get_batch`` moves them to the GPU in place), anything else as
    is."""
    return dict(item) if isinstance(item, dict) else item


class RingReplaySource:
    """One physical chunk's data iterator, replayed to the chunk's R round views in one step.

    The first view to ask for microbatch m (round 1's: the views run a microbatch's rounds in
    order) pulls it from ``source``; the other views get the cached batch, which is released
    after its R-th read. The source is therefore consumed once per microbatch, in order, exactly
    as the standard schedule consumes it (a ``RerunDataIterator`` keeps its semantics).
    """

    def __init__(self, source: Iterator, rounds: int):
        self.source = source
        self.rounds = rounds
        self.pulled = 0
        self._cache: Dict[int, list] = {}  # microbatch -> [batch, reads]

    def get(self, microbatch: int) -> Any:
        """Microbatch ``microbatch`` of the source (pulled on its first read)."""
        entry = self._cache.get(microbatch)
        if entry is None:
            if microbatch != self.pulled:
                raise RuntimeError(
                    f"Ring replay: microbatch {microbatch} asked for after {self.pulled} pulled "
                    "(every round view reads the chunk's microbatches in order)"
                )
            entry = [next(self.source), 0]
            self.pulled += 1
            self._cache[microbatch] = entry
        entry[1] += 1
        if entry[1] == self.rounds:
            del self._cache[microbatch]
        return _replay_copy(entry[0])

    def pending(self) -> Tuple[int, ...]:
        """The microbatches pulled but not yet read by all R views."""
        return tuple(sorted(self._cache))


class RingReplayIterator:
    """A round view's data iterator: its i-th ``next()`` is microbatch i of its physical chunk's
    source (``RingReplaySource``), read while the view's call i runs (its data step comes
    before the model call), so the batch and the view's microbatch always agree."""

    def __init__(self, source: RingReplaySource, view: "LoopRingRoundView"):
        self.source = source
        self.view = view
        self.served = 0

    def __iter__(self):
        return self

    def __next__(self) -> Any:
        microbatch = self.served
        if self.view.calls != microbatch:
            raise RuntimeError(
                f"Ring replay of round view {self.view.view_index}: batch {microbatch} asked for "
                f"by call {self.view.calls} (one batch per call of a data-loading chunk)"
            )
        item = self.source.get(microbatch)
        self.served += 1
        return item


def ring_physical_chunks(model: Sequence[nn.Module]) -> List[nn.Module]:
    """The distinct physical chunks behind a list of round views (any other module stands for
    itself), by identity, in order of first appearance: round 1's views come first, so this is
    the chunk order j = 0..k-1."""
    chunks: List[nn.Module] = []
    seen = set()
    for entry in model:
        chunk = entry.module if isinstance(entry, LoopRingRoundView) else entry
        if id(chunk) not in seen:
            seen.add(id(chunk))
            chunks.append(chunk)
    return chunks


@dataclass(frozen=True)
class _RingChunkState:
    """One physical chunk of the step: the wrapped chunk (DDP), its ``HybridModel``, the decoder
    and the built pipeline flags (``pre_process``, ``post_process``, ``mtp_process`` of the
    model, ``pre_process``, ``post_process`` of the decoder), which must never change."""

    chunk: nn.Module
    model: nn.Module
    decoder: nn.Module
    built: Tuple[bool, bool, bool, bool, bool]


def _built_flags(model: nn.Module, decoder: nn.Module) -> Tuple[bool, bool, bool, bool, bool]:
    return (
        bool(model.pre_process),
        bool(model.post_process),
        bool(getattr(model, "mtp_process", False)),
        bool(decoder.pre_process),
        bool(decoder.post_process),
    )


def _check_chunks_idle(states: Sequence[_RingChunkState], when: str) -> None:
    """Raise unless every chunk's call roles are cleared and its built flags unchanged."""
    for index, state in enumerate(states):
        if state.decoder.call_roles is not None:
            raise RuntimeError(
                f"Loop ring ({when}): chunk {index} still has call roles {state.decoder.call_roles}"
            )
        if _built_flags(state.model, state.decoder) != state.built:
            raise RuntimeError(
                f"Loop ring ({when}): chunk {index}'s built pipeline flags changed from "
                f"{state.built} to {_built_flags(state.model, state.decoder)}"
            )


class RingFinalize:
    """The ring config's ``finalize_model_grads_func``.

    The interleaved schedule finalizes its model list, the V round views. This wrapper reduces
    them to the k physical chunks (``ring_physical_chunks``: each parameter belongs to one chunk,
    one DDP, one grad buffer), checks that the step left no call roles and no changed built flag,
    and calls the model config's ``finalize_model_grads_func`` **as it is when called**:
    ``train_step`` re-wraps it every step for the chunked optimizer-state offload (prefetch, then
    finalize), so the order is reduce to chunks, prefetch, finalize, once per step. On the views
    finalize would run the grad sync, the per-token scaling and the temporary-tensor reset once
    per view, and the expert-bias update would find no router (a view has no submodules).
    """

    def __init__(self, model_config: Any, states: Sequence[_RingChunkState]):
        self.model_config = model_config
        self.states = tuple(states)
        self.calls = 0

    def __call__(self, model, num_tokens=None, pg_collection=None, force_all_reduce=False):
        chunks = ring_physical_chunks(model)
        if len(chunks) != len(self.states) or any(
            chunk is not state.chunk for chunk, state in zip(chunks, self.states)
        ):
            raise RuntimeError(
                "Loop ring finalize: the schedule's model list does not reduce to this step's "
                f"{len(self.states)} physical chunks"
            )
        _check_chunks_idle(self.states, "finalize")
        finalize = self.model_config.finalize_model_grads_func
        if finalize is None:
            raise RuntimeError(
                "Loop ring finalize: the model config's finalize_model_grads_func was cleared "
                "during the step"
            )
        self.calls += 1
        return finalize(
            chunks, num_tokens, pg_collection=pg_collection, force_all_reduce=force_all_reduce
        )


@dataclass(frozen=True)
class LoopRingStepRecord:
    """What one ring step ran, over the whole pipeline and every tensor x data rank (G1).

    ``rounds``: the fewest and the most core runs of a chunk for one sample (R and R); ``prelude``
    / ``coda`` / ``mtp``: the fewest and the most runs per sample over the pipeline (1 and 1; MTP
    0 without MTP); ``model_work``: the model's ``(forward, backward, recompute)`` FLOPs per
    microbatch (``LoopHybridStack.ring_microbatch_work`` summed over the pipeline).
    """

    microbatches: int
    views: int
    forward_only: bool
    rounds: Tuple[int, int]
    prelude: Tuple[int, int]
    coda: Tuple[int, int]
    mtp: Tuple[int, int]
    model_work: Tuple[float, float, float]

    def summary(self) -> str:
        """The G1 line: ``loop ring: rounds per sample min R max R; prelude/coda/mtp per sample
        1/1/1`` (a count that differs between samples prints as ``lo-hi``)."""

        def span(bounds):
            low, high = bounds
            return f"{low}" if low == high else f"{low}-{high}"

        return (
            f"loop ring: rounds per sample min {self.rounds[0]} max {self.rounds[1]}; "
            f"prelude/coda/mtp per sample {span(self.prelude)}/{span(self.coda)}/"
            f"{span(self.mtp)}"
        )


def _collective_device(group) -> torch.device:
    """The device of a small collective on ``group``: the current GPU for NCCL, else the CPU."""
    if dist.get_backend(group) == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _check_chunk_counts(
    states: Sequence[_RingChunkState], counts: Sequence[Mapping[int, Sequence[int]]], microbatches
) -> None:
    """Raise unless every chunk ran every microbatch's core R times, and the prelude (round-start
    chunk), the coda and MTP (round-end chunk) once (this rank's chunks; G1's local half)."""
    for index, (state, chunk_counts) in enumerate(zip(states, counts)):
        ring = state.decoder.loop_ring
        rounds = state.decoder.config.loop_iterations
        expected = (
            rounds,
            int(ring.round_start),
            int(ring.round_end),
            int(ring.round_end and state.built[2]),
        )
        got = {key: tuple(value) for key, value in chunk_counts.items()}
        want = {microbatch: expected for microbatch in range(microbatches)}
        if got != want:
            wrong = sorted(set(got) ^ set(want)) or [m for m in want if got[m] != want[m]]
            raise RuntimeError(
                f"Loop ring: chunk {index} ran (core, prelude, coda, MTP) {got} where every "
                f"microbatch 0..{microbatches - 1} must run {expected} (first wrong: {wrong[:4]})"
            )


class LoopRingSchedule:
    """A ``forward_backward_func``-compatible schedule for the loop ring (module docstring).

    ``model`` is the rank's list of k physical ring chunks (one at k = 1), ``data_iterator`` one
    iterator per chunk (or one at k = 1). The model's per-microbatch FLOPs are all-reduced over the
    pipeline once per (model, sequence length, micro batch size) and cached here, never on the
    model. ``last_step`` holds the last step's ``LoopRingStepRecord``.
    """

    # Relative tolerance of the recorded-vs-analytic FLOP check (the DSv4 FLOP rows are not
    # integers; the two sums add the same terms in different orders).
    WORK_RTOL = 1e-9

    def __init__(self):
        self._model_work: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
        self.last_step: Optional[LoopRingStepRecord] = None

    def reset(self) -> None:
        """Forget the cached FLOPs and the last step (tests)."""
        self._model_work = weakref.WeakKeyDictionary()
        self.last_step = None

    # ------------------------------------------------------------------------------- setup

    @staticmethod
    def _chunk_states(chunks: Sequence[nn.Module], config: Any) -> List[_RingChunkState]:
        """The step's physical chunks, checked: ring chunks, in virtual-pipeline order, k of
        them for the model config's virtual pipeline size."""
        chunk_count = getattr(config, "virtual_pipeline_model_parallel_size", None) or 1
        if len(chunks) != chunk_count:
            raise ValueError(
                f"Loop ring: {len(chunks)} model chunks for the virtual pipeline size "
                f"{chunk_count} (one chunk per core slice of the stage)"
            )
        states = []
        for index, chunk in enumerate(chunks):
            model = get_attr_wrapped_model(chunk, "decoder", return_model_obj=True)
            decoder = model.decoder
            if getattr(decoder, "loop_ring", None) is None:
                raise ValueError(
                    "the loop ring schedule runs ring chunks only (loop_schedule='ring': "
                    "LoopHybridStack chunks of the '|' layout)"
                )
            vp_stage = getattr(model, "vp_stage", None)
            if chunk_count > 1 and vp_stage != index:
                raise ValueError(
                    f"Loop ring: model chunk {index} is virtual-pipeline chunk {vp_stage}"
                )
            states.append(_RingChunkState(chunk, model, decoder, _built_flags(model, decoder)))
        return states

    def _model_work_for(
        self, states, codes: str, rounds: int, sequence: int, batch: int, pp_group
    ) -> Tuple[float, float, float]:
        """The model's ``(forward, backward, recompute)`` FLOPs per microbatch of
        ``[sequence, batch]``: the sum of every chunk's ``ring_microbatch_work`` over the
        pipeline (one all-reduce per model and shape; every pipeline rank runs the same steps, so
        every rank asks at the same step)."""
        chunk_count = len(states)
        cache = self._model_work.setdefault(states[0].model, {})
        key = (int(sequence), int(batch))
        if key not in cache:
            local = [0.0, 0.0, 0.0]
            for index, state in enumerate(states):
                chunk_codes = tuple(codes[r * chunk_count + index] for r in range(rounds))
                work = state.decoder.ring_microbatch_work(sequence, batch, chunk_codes)
                local = [a + b for a, b in zip(local, work)]
            total = torch.tensor(local, dtype=torch.float64, device=_collective_device(pp_group))
            dist.all_reduce(total, group=pp_group)
            cache[key] = tuple(float(value) for value in total.tolist())
        return cache[key]

    # -------------------------------------------------------------------------------- step

    def __call__(
        self,
        *,
        forward_step_func,
        data_iterator,
        model,
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: Optional[int] = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
        first_val_step: Optional[bool] = None,
        adjust_tensor_shapes_fn: Optional[Callable] = None,
        p2p_communicator=None,
        pg_collection=None,
        force_all_reduce: Optional[bool] = False,
    ) -> list:
        # pylint: disable=import-outside-toplevel
        from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
        from megatron.core.pipeline_parallel.schedules import (
            _build_default_pg_collection,
            forward_backward_pipelining_with_interleaving,
        )
        from megatron.core.transformer.loop.guards import resolve_loop_ring_recompute_plan

        if adjust_tensor_shapes_fn is not None:
            raise ValueError("adjust_tensor_shapes_fn is not supported by the loop ring schedule")
        if p2p_communicator is not None:
            raise ValueError(
                "the loop ring schedule builds its own P2P communicator (virtual pipeline size "
                "R x k); a caller's communicator is not supported"
            )
        chunks = list(model) if isinstance(model, (list, tuple)) else [model]
        if isinstance(data_iterator, (list, tuple)):
            sources = list(data_iterator)
        else:
            sources = [data_iterator] if len(chunks) == 1 else []
        if len(sources) != len(chunks):
            raise ValueError("the loop ring schedule needs one data iterator per model chunk")
        if pg_collection is None:
            # Migration fallback, as in the standard schedules: train() passes None today.
            pg_collection = _build_default_pg_collection()
        model_config = get_model_config(chunks[0])
        states = self._chunk_states(chunks, model_config)
        chunk_count = len(states)
        rounds = model_config.loop_iterations
        views = rounds * chunk_count
        pp_group = pg_collection.pp
        stages, stage = pp_group.size(), pp_group.rank()
        if stages != model_config.pipeline_model_parallel_size:
            raise ValueError(
                f"Loop ring: the pipeline group has {stages} ranks, the config "
                f"{model_config.pipeline_model_parallel_size}"
            )
        for name in ("grad_sync_func", "param_sync_func"):
            if getattr(model_config, name, None) is not None:
                raise ValueError(
                    f"the loop ring schedule runs without overlapped grad reduce / param gather "
                    f"(config.{name} is set; the per-chunk sync functions would be indexed by "
                    "round view)"
                )
        if getattr(model_config, "num_microbatches_with_partial_activation_checkpoints", None):
            raise ValueError(
                "num_microbatches_with_partial_activation_checkpoints is not supported by the "
                "loop ring schedule (its recompute is the loop_ring_recompute_plan)"
            )
        codes = resolve_loop_ring_recompute_plan(model_config)[stage]

        # The model's FLOPs per microbatch, reported by the round-end chunk's head call.
        model_work = self._model_work_for(
            states, codes, rounds, seq_length, micro_batch_size, pp_group
        )
        for state in states:
            if state.decoder.loop_ring.round_end:
                state.decoder.set_ring_model_work(seq_length, micro_batch_size, model_work)

        # A clean start: no call in flight, nothing recorded.
        _check_chunks_idle(states, "step start")
        for state in states:
            if state.decoder.ring_injection_keys():
                raise RuntimeError(
                    "Loop ring: a previous step left injection store entries "
                    f"{state.decoder.ring_injection_keys()}"
                )
            state.decoder.take_ring_work()
            state.decoder.take_ring_counts()
        grad_enabled = torch.is_grad_enabled()

        # The ring config: the schedule's view of the model config (V chunks, the finalize
        # wrapper); the physical chunks keep the model config.
        finalize = RingFinalize(model_config, states)
        ring_config = copy.copy(model_config)
        ring_config.virtual_pipeline_model_parallel_size = views
        ring_config.finalize_model_grads_func = (
            finalize if model_config.finalize_model_grads_func is not None else None
        )
        communicator = P2PCommunicator(pp_group=pp_group, config=ring_config)

        round_views = [
            LoopRingRoundView(
                states[view % chunk_count].chunk,
                ring_config,
                round_index=view // chunk_count,
                chunk_index=view % chunk_count,
                chunks=chunk_count,
                checkpoint=codes[view] == "C",
            )
            for view in range(views)
        ]
        replays = [
            None if source is None else RingReplaySource(source, rounds) for source in sources
        ]
        view_iterators = [
            (
                None
                if replays[view.chunk_index] is None
                else RingReplayIterator(replays[view.chunk_index], view)
            )
            for view in round_views
        ]

        output = forward_backward_pipelining_with_interleaving(
            forward_step_func=forward_step_func,
            data_iterator=view_iterators,
            model=round_views,
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            forward_only=forward_only,
            collect_non_loss_data=collect_non_loss_data,
            first_val_step=first_val_step,
            p2p_communicator=communicator,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

        self._check_step_end(
            states,
            round_views,
            replays,
            finalize,
            codes,
            num_microbatches=num_microbatches,
            sequence=seq_length,
            batch=micro_batch_size,
            grad_enabled=grad_enabled,
            finalized=ring_config.finalize_model_grads_func is not None and not forward_only,
        )
        counts = [state.decoder.take_ring_counts() for state in states]
        _check_chunk_counts(states, counts, num_microbatches)
        self.last_step = self._reduce_counts(
            counts,
            num_microbatches,
            views,
            forward_only,
            model_work,
            pp_group,
            getattr(pg_collection, "tp_dp_cp", None),
        )
        return output

    # -------------------------------------------------------------------------- step end

    def _check_step_end(
        self,
        states,
        round_views,
        replays,
        finalize,
        codes,
        *,
        num_microbatches,
        sequence,
        batch,
        grad_enabled,
        finalized,
    ) -> None:
        """The step-end invariants of the module docstring (raise on the first violation)."""
        calls = [view.calls for view in round_views]
        if any(count != num_microbatches for count in calls):
            raise RuntimeError(
                f"Loop ring: round view calls {calls}, every view must run {num_microbatches}"
            )
        for index, replay in enumerate(replays):
            if replay is None:
                continue
            if replay.pulled not in (0, num_microbatches) or replay.pending():
                raise RuntimeError(
                    f"Loop ring replay of chunk {index}: {replay.pulled} batches pulled, "
                    f"microbatches {replay.pending()} not read by all views "
                    f"({num_microbatches} microbatches)"
                )
        _check_chunks_idle(states, "step end")
        for index, state in enumerate(states):
            keys = state.decoder.ring_injection_keys()
            if keys:
                raise RuntimeError(f"Loop ring: chunk {index}'s injection store holds {keys}")
        if finalize.calls != (1 if finalized else 0):
            raise RuntimeError(
                f"Loop ring: the finalize wrapper ran {finalize.calls} times in this step"
            )
        rounds = states[0].decoder.config.loop_iterations
        chunk_count = len(states)
        for index, state in enumerate(states):
            chunk_codes = tuple(codes[r * chunk_count + index] for r in range(rounds))
            analytic = state.decoder.ring_microbatch_work(sequence, batch, chunk_codes)
            if not grad_enabled:
                analytic = (analytic[0], 0.0, 0.0)
            recorded = state.decoder.take_ring_work()
            for phase, got, want in zip(("forward", "backward", "recompute"), recorded, analytic):
                if not math.isclose(got, num_microbatches * want, rel_tol=self.WORK_RTOL):
                    raise RuntimeError(
                        f"Loop ring: chunk {index} recorded {got} {phase} FLOPs in a step of "
                        f"{num_microbatches} microbatches, {want} per microbatch analytically"
                    )

    @staticmethod
    def _reduce_counts(
        counts, microbatches, views, forward_only, model_work, pp_group, data_group
    ) -> LoopRingStepRecord:
        """G1 over the whole run: per microbatch the fewest / most core runs of any chunk and
        the prelude / coda / MTP runs summed over the pipeline (an all-gather over the pipeline),
        then the fewest / most over the microbatches and every tensor x data rank (a MIN
        all-reduce over ``data_group``, the tensor x data x context ranks of this stage)."""
        device = _collective_device(pp_group)
        summary = ring_sample_counts(counts)
        rows = torch.tensor(
            [summary.get(microbatch, (0, 0, 0, 0, 0)) for microbatch in range(microbatches)],
            dtype=torch.int64,
            device=device,
        )
        gathered = [torch.empty_like(rows) for _ in range(pp_group.size())]
        dist.all_gather(gathered, rows, group=pp_group)
        stacked = torch.stack(gathered)  # [P, M, 5]
        per_sample = torch.stack(
            (
                stacked[:, :, 0].min(dim=0).values,
                stacked[:, :, 1].max(dim=0).values,
                stacked[:, :, 2].sum(dim=0),
                stacked[:, :, 3].sum(dim=0),
                stacked[:, :, 4].sum(dim=0),
            ),
            dim=1,
        )  # [M, 5]
        bounds = torch.stack(
            (
                per_sample[:, 0].min(),
                -per_sample[:, 1].max(),
                per_sample[:, 2].min(),
                -per_sample[:, 2].max(),
                per_sample[:, 3].min(),
                -per_sample[:, 3].max(),
                per_sample[:, 4].min(),
                -per_sample[:, 4].max(),
            )
        )
        if data_group is not None and data_group.size() > 1:
            dist.all_reduce(bounds, op=dist.ReduceOp.MIN, group=data_group)
        values = [int(value) for value in bounds.tolist()]
        return LoopRingStepRecord(
            microbatches=microbatches,
            views=views,
            forward_only=forward_only,
            rounds=(values[0], -values[1]),
            prelude=(values[2], -values[3]),
            coda=(values[4], -values[5]),
            mtp=(values[6], -values[7]),
            model_work=tuple(model_work),
        )


_DEFAULT_RING_SCHEDULE = LoopRingSchedule()


def get_loop_ring_schedule() -> LoopRingSchedule:
    """The module-level ring schedule (one FLOP cache and last-step record per process)."""
    return _DEFAULT_RING_SCHEDULE


def forward_backward_loop_ring(**kwargs) -> list:
    """Schedule entry: one step of the module-level ``LoopRingSchedule`` with the standard
    schedules' keywords (``forward_step_func``, ``data_iterator``, ``model``,
    ``num_microbatches``, ``seq_length``, ``micro_batch_size``, ``forward_only``,
    ``pg_collection``, ``force_all_reduce``, ...)."""
    return _DEFAULT_RING_SCHEDULE(**kwargs)
