# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The loop ring's op order (``loop_schedule='ring'``): Megatron's interleaved schedule over
V = R x k round views, at the op level (CPU, no model).

Each pipeline rank's op list is Megatron's own: ``get_schedule_table`` and
``get_pp_rank_microbatches`` for V model chunks (the warmup count from a communicator whose
virtual pipeline size is V, as ``LoopRingSchedule`` builds it), cross-checked against
``convert_schedule_table_to_order``. The ring's data flow: view v on rank p sends its output to
(p + 1, v), and the last rank's view v to rank 0's view v + 1 (the round wrap); gradients flow
back. ``simulate`` runs every rank's list with **blocking receives** (an op waits until the
message it consumes has arrived) and buffered sends, one FIFO channel per (sender, receiver,
direction), and checks:

* FIFO pairing: every op consumes the message its predecessor (forward) or successor (backward)
  sent for the same microbatch, in channel order (real P2P pairs messages by order, not by tag);
* no deadlock: every rank finishes its list;
* every instance (view, microbatch) runs its forward before its backward, and a view's instances
  are released first-in first-out (the schedule pops its input / output lists from the front);
* the live-instance profile: rank p's peak of live instances equals the independent count
  ``min(2 (P - p - 1) + (V - 1) N + 1, V M)`` (warmup forwards, then one forward ahead).

Grid: P in {2, 3, 4, 8}, k in {1, 2, 4} (R = 4: V in {4, 8, 16}), microbatch group size
N in {P, 2P}, M in {N, N + P, 2N} (every M the real schedule accepts for that N). Mutations: any
two adjacent ops of the same kind on one rank swapped is detected; a communicator built from the
model config (virtual pipeline size None or k, not V) gives a different warmup count.
"""

import collections
import itertools
from types import SimpleNamespace

import pytest

from megatron.core.pipeline_parallel import schedules
from tests.unit_tests.transformer.loop.ring_utils import StageGroup

ROUNDS = 4
STAGES = (2, 3, 4, 8)
CHUNKS = (1, 2, 4)


def _communicator(stage, stages, virtual):
    """The fields of a ``P2PCommunicator`` that ``get_pp_rank_microbatches`` reads."""
    return SimpleNamespace(
        pp_group=StageGroup(stage, stages), virtual_pipeline_model_parallel_size=virtual
    )


def rank_ops(stages, stage, views, microbatches, group_size, virtual="ring"):
    """Rank ``stage``'s op list ``[(kind, view, microbatch)]`` and warmup count in Megatron's
    interleaved order, with the warmup count of a communicator of virtual pipeline size
    ``virtual`` (``"ring"``: V, as the ring schedule builds it; None: no virtual pipeline)."""
    virtual = views if virtual == "ring" else virtual
    table = schedules.get_schedule_table(microbatches, views, group_size)
    _, _, warmup, remaining = schedules.get_pp_rank_microbatches(
        microbatches, views, group_size, p2p_communicator=_communicator(stage, stages, virtual)
    )
    forward = [("F", view, microbatch) for microbatch, view in table]
    backward = [("B", views - 1 - view, microbatch) for microbatch, view in table]
    ops = forward[:warmup]
    for index in range(remaining):
        ops += [forward[warmup + index], backward[index]]
    ops += backward[remaining:]
    order = schedules.convert_schedule_table_to_order(warmup, views, table)
    assert [view + 1 if kind == "F" else -(view + 1) for kind, view, _ in ops] == order
    return ops, warmup


class ScheduleError(AssertionError):
    """A simulated op order violates a ring rule (FIFO pairing, deadlock, instance order)."""


def simulate(stages, views, ops_by_rank):
    """Run every rank's op list with blocking receives (module docstring). Returns the per-rank
    live-instance counts after every op; raises ``ScheduleError`` on a violation."""

    def successor(stage, view):
        if stage < stages - 1:
            return stage + 1, view
        return (0, view + 1) if view < views - 1 else None

    def predecessor(stage, view):
        if stage > 0:
            return stage - 1, view
        return (stages - 1, view - 1) if view > 0 else None

    channels = collections.defaultdict(collections.deque)  # (src, dst, kind) -> (view, mb)
    live = [collections.defaultdict(collections.deque) for _ in range(stages)]  # view -> mbs
    counts = [[] for _ in range(stages)]
    pointers = [0] * stages

    def needs(stage, op):
        """The (channel, message) an op consumes, or None."""
        kind, view, microbatch = op
        peer = predecessor(stage, view) if kind == "F" else successor(stage, view)
        if peer is None:
            return None
        return (peer[0], stage, kind), (peer[1], microbatch)

    def run(stage, op):
        kind, view, microbatch = op
        wanted = needs(stage, op)
        if wanted is not None:
            channel, message = wanted
            got = channels[channel].popleft()
            if got != message:
                raise ScheduleError(
                    f"rank {stage} {kind}(view {view}, mb {microbatch}) received {got} on "
                    f"channel {channel}, expected {message} (FIFO pairing)"
                )
        if kind == "F":
            live[stage][view].append(microbatch)
            target = successor(stage, view)
        else:
            queue = live[stage][view]
            if not queue:
                raise ScheduleError(f"rank {stage} backward of view {view} before its forward")
            if queue[0] != microbatch:
                raise ScheduleError(
                    f"rank {stage} view {view}: backward of mb {microbatch} while mb {queue[0]} "
                    "is the oldest live instance (instances are released first-in first-out)"
                )
            queue.popleft()
            target = predecessor(stage, view)
        if target is not None:
            channels[(stage, target[0], kind)].append((view, microbatch))
        counts[stage].append(sum(len(queue) for queue in live[stage].values()))

    while any(pointer < len(ops) for pointer, ops in zip(pointers, ops_by_rank)):
        progressed = False
        for stage, ops in enumerate(ops_by_rank):
            while pointers[stage] < len(ops):
                op = ops[pointers[stage]]
                wanted = needs(stage, op)
                if wanted is not None and not channels[wanted[0]]:
                    break  # blocked on a receive
                run(stage, op)
                pointers[stage] += 1
                progressed = True
        if not progressed:
            blocked = {
                stage: ops[pointers[stage]]
                for stage, ops in enumerate(ops_by_rank)
                if pointers[stage] < len(ops)
            }
            raise ScheduleError(f"deadlock under blocking receives: ranks blocked at {blocked}")
    leftover = {key: list(value) for key, value in channels.items() if value}
    if leftover:
        raise ScheduleError(f"messages never received: {leftover}")
    return counts


def _grid():
    for stages, chunks in itertools.product(STAGES, CHUNKS):
        for group_size in (stages, 2 * stages):
            for microbatches in sorted({group_size, group_size + stages, 2 * group_size}):
                yield stages, chunks, group_size, microbatches


def _ops(stages, chunks, group_size, microbatches):
    views = ROUNDS * chunks
    return [rank_ops(stages, stage, views, microbatches, group_size)[0] for stage in range(stages)]


@pytest.mark.parametrize("stages,chunks,group_size,microbatches", list(_grid()))
def test_ring_order_pairs_fifo_without_deadlock(stages, chunks, group_size, microbatches):
    views = ROUNDS * chunks
    ops = _ops(stages, chunks, group_size, microbatches)
    counts = simulate(stages, views, ops)
    for stage in range(stages):
        assert len(ops[stage]) == 2 * views * microbatches
        assert counts[stage][-1] == 0, "every forward had its backward"
        expected_peak = min(
            2 * (stages - stage - 1) + (views - 1) * group_size + 1, views * microbatches
        )
        assert max(counts[stage]) == expected_peak, (stage, max(counts[stage]), expected_peak)
        # Each (view, microbatch) runs once forward and once backward on every rank.
        assert collections.Counter((view, mb) for _, view, mb in ops[stage]) == {
            (view, mb): 2 for view in range(views) for mb in range(microbatches)
        }


@pytest.mark.parametrize(
    "stages,chunks,group_size,microbatches",
    [(2, 1, 2, 4), (2, 4, 4, 8), (3, 2, 3, 6), (4, 1, 4, 8), (4, 2, 8, 12), (8, 4, 8, 16)],
)
def test_adjacent_same_kind_swaps_are_detected(stages, chunks, group_size, microbatches):
    """Any two adjacent forwards (or backwards) of different instances swapped on one rank break
    FIFO pairing (or the instance order): the receiver consumes its channel in the sender's
    order, and every instance but the ring's ends both receives and sends."""
    views = ROUNDS * chunks
    base = _ops(stages, chunks, group_size, microbatches)
    simulate(stages, views, base)
    tried = 0
    for stage in range(stages):
        ops = base[stage]
        pairs = [i for i in range(len(ops) - 1) if ops[i][0] == ops[i + 1][0]]
        for index in (pairs[0], pairs[len(pairs) // 2], pairs[-1]):
            mutated = list(base)
            swapped = list(ops)
            swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
            mutated[stage] = swapped
            with pytest.raises(ScheduleError):
                simulate(stages, views, mutated)
            tried += 1
    assert tried == 3 * stages


@pytest.mark.parametrize("stages", STAGES)
def test_a_communicator_without_a_virtual_pipeline_gets_another_warmup(stages):
    """The warmup count's branch comes from the communicator's virtual pipeline size. At k = 1
    the model config has none: a communicator built from it gives the non-interleaved warmup
    (P - p - 1) on every rank instead of the ring's 2 (P - p - 1) + (V - 1) N, and that op list
    does not run as a ring. (At k > 1 the model config's k selects the same interleaved branch:
    the count itself is computed from the V model chunks the schedule is given.)"""
    views = ROUNDS
    microbatches = group_size = 2 * stages
    ops = []
    for stage in range(stages):
        _, ring_warmup = rank_ops(stages, stage, views, microbatches, group_size)
        stage_ops, model_warmup = rank_ops(
            stages, stage, views, microbatches, group_size, virtual=None
        )
        assert ring_warmup == 2 * (stages - stage - 1) + (views - 1) * group_size
        assert model_warmup == stages - stage - 1 != ring_warmup
        ops.append(stage_ops)
    with pytest.raises(ScheduleError):
        simulate(stages, views, ops)
    for chunks in (2, 4):
        _, with_k = rank_ops(stages, 0, ROUNDS * chunks, microbatches, group_size, virtual=chunks)
        _, with_v = rank_ops(stages, 0, ROUNDS * chunks, microbatches, group_size)
        assert with_k == with_v
