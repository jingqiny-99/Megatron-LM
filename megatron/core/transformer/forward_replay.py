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
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

_FORWARD_REPLAY: ContextVar[int] = ContextVar("forward_replay", default=0)
_RECORD_WEIGHT: ContextVar[float] = ContextVar("forward_statistics_weight", default=1.0)
_RECORD_ROUND: ContextVar[Optional[tuple]] = ContextVar("forward_statistics_round", default=None)


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
