# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Scoped markers for forwards that repeat an already-counted forward.

A non-reentrant activation checkpoint (``torch.utils.checkpoint(..., use_reentrant=False)``)
replays its forward with grad enabled, so code that gates a forward statistic on
``torch.is_grad_enabled()`` would count it twice: once in the original forward and once in the
replay. ``forward_replay()`` marks the replay, and statistics that must count each forward once
skip themselves inside it:

* the router's expert-bias token counts (``TopKRouter._apply_expert_bias``'s call site),
* the MoE aux / z-loss metrics tracker (``MoEMetricsTracker.record``),
* the DSA indexer-loss tracker (``DSAIndexerLossLoggingHelper.save_loss_to_tracker``),
* the overload-factor forward record (``RecordDispatchTokenCountsFunction.forward``; its
  backward record is unchanged, so forward and backward records still pair).

Losses and their autograd attachments are not affected: a replay must rebuild them.

Not gated (callers that replay forwards must reject them or accept double counts): the
quantile-balancing router histograms, the ``global_aux_loss`` running average
(``global_tokens_per_expert`` / ``ga_steps``) and router replay / trace recording.

``forward_statistics_round(r, rounds)`` marks the forwards of loop round ``r`` (of ``rounds``):
the MoE aux / z-loss metrics tracker and the DSA indexer-loss tracker also add each value they
record into a per-(round, layer) bin, logged next to the per-layer sums (a loop's core layers
run once per round; prelude, coda and MTP records have no round).

``forward_statistics_weight(weight)`` scales the values the per-microbatch-mean loss trackers
(MoE aux / z-loss, DSA indexer loss) record. A schedule that runs the model once per sample
instead of once per microbatch (the loop RSBF executor) enters it with ``1 / micro_batch_size``,
so the log-time scale ``1 / num_microbatches`` still yields the per-microbatch mean. Counts
(expert-bias tokens) are not weighted.

``RoutingTape`` with ``record_routing`` / ``replay_routing``: the MoE top-k expert choices of
one non-reentrant activation checkpoint. A non-reentrant checkpoint backpropagates through the
ORIGINAL graph with the recomputed saved tensors, so every recomputed tensor must have the
original shape. A checkpoint that spans several MoE sublayers (a loop round under
``loop_recompute``) breaks that whenever the recompute is not bitwise equal to the forward
(non-deterministic kernels such as the MoE unpermute's ``scatter_add_``): a later router flips a
near-tied choice, the per-expert token counts change, and the backward fails with
``CheckpointError: Recomputed values ... have different metadata``. The checkpoint's forward
records every router's top-k indices (``moe_utils.topk_routing_with_score_function``) and the
recompute replays them in the same order, taking the scores at the recorded indices, so the
recompute routes exactly as the forward did (bitwise the same result when the recompute is
deterministic). The fused TE router has no hook, so taped forwards use the unfused path
(``TopKRouter.routing`` decides so before it picks its output format). With
``moe_router_fusion``, ``loop_recompute`` therefore changes the rounds' router numerics: the
taped rounds route unfused, the same model without ``loop_recompute`` routes them fused, and the
two paths agree only to rounding.

Not taped (the router paths that bypass ``compute_topk``'s tape or draw fresh random logits in
the recompute, which the loop validators refuse together with ``loop_recompute``,
``loop.guards.untaped_recompute_routing``): router replay (``moe_enable_routing_replay``, even
without a replay action), Sinkhorn routing, quantile balancing, forced load balancing and
force-biased logits (their random logits come from Megatron's CUDA RNG tracker, which
``torch.utils.checkpoint`` does not rewind). The hash routing table is input-id driven and
needs no tape.

Scope: the tape fixes the MoE expert choice, which sets the recomputed tensors' shapes. The
other discrete choices of a recompute that is not bitwise its forward are recomputed, not
replayed: the DSA / CSA indexer top-k, the aux-loss routing map and capacity token dropping
keep their shapes and can flip at near-ties, so the saved tensors follow the recompute's choice
there (a small, silent difference, as for any non-bitwise recompute).
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import List, Optional, Tuple

import torch

_FORWARD_REPLAY: ContextVar[int] = ContextVar("forward_replay", default=0)
_RECORD_WEIGHT: ContextVar[float] = ContextVar("forward_statistics_weight", default=1.0)
_RECORD_ROUND: ContextVar[Optional[tuple]] = ContextVar("forward_statistics_round", default=None)
_ROUTING_TAPE: ContextVar[Optional[tuple]] = ContextVar("routing_tape", default=None)


@contextmanager
def forward_replay() -> Iterator[None]:
    """Mark a forward that repeats an already-counted forward (an activation recompute).

    Nests; the marker is restored on exit, also when the body raises.
    """
    token = _FORWARD_REPLAY.set(_FORWARD_REPLAY.get() + 1)
    try:
        yield
    finally:
        _FORWARD_REPLAY.reset(token)


def in_forward_replay() -> bool:
    """True inside ``forward_replay()``."""
    return _FORWARD_REPLAY.get() > 0


@contextmanager
def forward_statistics_weight(weight: float) -> Iterator[None]:
    """Scale the values the per-microbatch-mean loss trackers record by ``weight`` (a Python
    float, > 0). Nested weights multiply; the weight is restored on exit."""
    weight = float(weight)
    if not weight > 0.0:
        raise ValueError(f"forward statistics weight must be positive, got {weight}")
    token = _RECORD_WEIGHT.set(_RECORD_WEIGHT.get() * weight)
    try:
        yield
    finally:
        _RECORD_WEIGHT.reset(token)


def statistics_weight() -> float:
    """The current record weight (1.0 outside ``forward_statistics_weight``)."""
    return _RECORD_WEIGHT.get()


@contextmanager
def forward_statistics_round(r: int, rounds: int) -> Iterator[None]:
    """Mark the forwards inside the scope as loop round ``r`` of ``rounds`` (``1 <= r <=
    rounds``): the loss trackers also bin what they record by round. The innermost round wins;
    the previous one is restored on exit."""
    r, rounds = int(r), int(rounds)
    if not 1 <= r <= rounds:
        raise ValueError(f"loop round {r} outside 1..{rounds}")
    token = _RECORD_ROUND.set((r, rounds))
    try:
        yield
    finally:
        _RECORD_ROUND.reset(token)


def statistics_round() -> Optional[tuple]:
    """``(r, rounds)`` inside ``forward_statistics_round``, else None."""
    return _RECORD_ROUND.get()


class RoutingTape:
    """The top-k expert indices the routers chose inside one checkpointed forward, in call
    order (see the module docstring). One tape per checkpoint call: its forward records and its
    recompute replays (``replay_routing`` restarts the tape on every entry; through
    ``loop_transformer_block.loop_checkpoint_context_fn`` a checkpoint frame is recomputed once,
    as before the tape: its recompute context manager is single-use)."""

    def __init__(self) -> None:
        self.indices: List[torch.Tensor] = []
        self.cursor = 0


@contextmanager
def record_routing(tape: RoutingTape) -> Iterator[None]:
    """Record every router top-k inside the scope onto ``tape`` (a checkpoint's forward)."""
    tape.indices.clear()
    tape.cursor = 0
    token = _ROUTING_TAPE.set((tape, False))
    try:
        yield
    finally:
        _ROUTING_TAPE.reset(token)


@contextmanager
def replay_routing(tape: RoutingTape) -> Iterator[None]:
    """Replay ``tape`` to the routers inside the scope (the checkpoint's recompute).

    A recompute that runs to the end of the checkpointed function must have used every recorded
    choice. Under torch's default early stop the recompute stops once its last saved tensor is
    rebuilt (``_StopRecomputationError`` raises through the scope), so this count check runs
    only with ``torch.utils.checkpoint.set_checkpoint_early_stop(False)`` (the loop round
    checkpoints of rounds that end inside a layer, ``loop_round_checkpoint``). It is a safety net: a
    recompute that routes fewer times also saves fewer tensors (CheckpointError), and one that
    routes more raises in ``taped_topk``."""
    tape.cursor = 0
    token = _ROUTING_TAPE.set((tape, True))
    try:
        yield
    finally:
        _ROUTING_TAPE.reset(token)
    if tape.cursor != len(tape.indices):
        raise RuntimeError(
            f"activation recompute routed {tape.cursor} times, its forward "
            f"{len(tape.indices)} times: the recompute does not repeat the forward's MoE layers"
        )


def routing_tape_active() -> bool:
    """True inside ``record_routing`` or ``replay_routing``."""
    return _ROUTING_TAPE.get() is not None


def taped_topk(
    scores: torch.Tensor, compute: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``compute(scores)`` -> ``(values, indices)`` of a router top-k, recorded or replayed by
    the active ``RoutingTape`` (unchanged outside one).

    Under a tape the forward and its recompute build the same autograd graph (a non-reentrant
    checkpoint matches their saved tensors one by one): the indices come from ``compute`` on
    the detached scores (forward) or from the tape (recompute), and the values are
    ``scores.gather(1, indices)`` in both, equal to ``torch.topk``'s values, with the same
    gradient (scattered to the distinct chosen indices)."""
    current = _ROUTING_TAPE.get()
    if current is None:
        return compute(scores)
    tape, replay = current
    if not replay:
        # Detached, not no_grad: the top-k's ``sorted`` follows the grad mode.
        _, indices = compute(scores.detach())
        tape.indices.append(indices)
        return scores.gather(1, indices), indices
    if tape.cursor >= len(tape.indices):
        raise RuntimeError(
            f"activation recompute routed more often than its forward ({len(tape.indices)} "
            "router calls recorded)"
        )
    indices = tape.indices[tape.cursor]
    tape.cursor += 1
    if indices.shape[0] != scores.shape[0] or indices.device != scores.device:
        raise RuntimeError(
            f"activation recompute router call {tape.cursor} has scores {tuple(scores.shape)} "
            f"on {scores.device}; its forward chose {tuple(indices.shape)} on {indices.device}"
        )
    return scores.gather(1, indices), indices
