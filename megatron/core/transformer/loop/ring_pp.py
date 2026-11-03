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
"""

import contextlib
from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from megatron.core.models.hybrid.hybrid_block import HybridStackCallRoles
from megatron.core.utils import get_attr_wrapped_model


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
