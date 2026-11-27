# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the loop parameter store's host schedules (``loop/param_store_schedule.py``).

Pure host logic, no model and no process group: the segment ops of every sub-round type and item
form, the unit events derived from them per residency mode, the static slot / scratch assignment
(K and G, one-ahead gathers, re-gathers into the forward slot, scratch entries rebound only after
their reduce was issued) and the replay check that rejects a broken schedule.

Unit numbering of the synthetic store below (pattern ``*-*-*-*-``, P = 2, C = 4, D = 2): 0 the
embedding, 1 the prelude pair (sublayers 0, 1), 2 / 3 the core pairs (2, 3) / (4, 5), 4 the coda
pair (6, 7), 5 the loop adapter, 6 the output layer, 7 / 8 an MTP depth's projection / layer.
"""

import pytest

from megatron.core.transformer.loop.param_store_schedule import (
    ScheduleError,
    SegmentKey,
    SegmentOp,
    SubRoundSchedule,
    UnitEvent,
    build_subround_schedule,
    check_schedule,
    derive_unit_events,
    segment_ops,
    simulate_slots,
)
from megatron.core.transformer.loop.program import CoreBackwardPlan
from megatron.core.transformer.loop.rsbf_plan import SubRoundType

PF, CF, CB, CODA, PB = (
    SubRoundType.PRELUDE_F,
    SubRoundType.CORE_F,
    SubRoundType.CORE_B,
    SubRoundType.CODA,
    SubRoundType.PRELUDE_B,
)
P, C, D = 2, 4, 2
EMB, PRE, CORE_A, CORE_B_, CODA_U, ADAPTER, OUT, MTP_P, MTP_L = range(9)


def sub(i):
    return SegmentKey("sub", i)


EMBED, HEAD, ADAPT = SegmentKey("embed"), SegmentKey("head"), SegmentKey("adapter")

SEGMENT_UNITS = {
    EMBED: (EMB,),
    sub(0): (PRE,),
    sub(1): (PRE,),
    sub(2): (CORE_A,),
    sub(3): (CORE_A,),
    sub(4): (CORE_B_,),
    sub(5): (CORE_B_,),
    sub(6): (CODA_U,),
    sub(7): (CODA_U,),
    ADAPT: (ADAPTER,),
    HEAD: (OUT, MTP_P, MTP_L, EMB),
}
CORE_UNITS = (CORE_A, CORE_B_, ADAPTER)
EDGE_UNITS = (PRE, CODA_U, MTP_P, MTP_L)

PLANS = {
    "whole_round": CoreBackwardPlan.whole_round(C),
    "block": CoreBackwardPlan.strided(C, 2),
    "nested": CoreBackwardPlan.nested(C, 2),
}


def ops_text(ops):
    return [f"{op.direction} {op.segment}:{op.policy}" for op in ops]


def schedule(kind, *, form="whole_round", weight=(), grads=(), forward=(), oversize=(), **kw):
    return build_subround_schedule(
        kind,
        prelude=P,
        core=C,
        coda=D,
        segment_units=SEGMENT_UNITS,
        weight_units=weight,
        grad_units=grads,
        forward_grad_units=forward,
        oversize_units=oversize,
        core_plan=PLANS[form],
        **kw,
    )


def actions(sched, unit=None):
    return [
        (e.op_index, e.when, e.action, e.unit, e.slot, e.scratch)
        for e in sched.events
        if unit is None or e.unit == unit
    ]


# ---------------------------------------------------------------------------------------------
# Segment ops
# ---------------------------------------------------------------------------------------------


def test_segment_ops_of_every_subround_type():
    assert ops_text(segment_ops(PF, prelude=P, core=C, coda=D)) == [
        "fwd embed:sweep",
        "fwd sub 0:sweep",
        "fwd sub 1:sweep",
    ]
    assert ops_text(segment_ops(CF, prelude=P, core=C, coda=D)) == [
        "fwd adapter:sweep",
        "fwd sub 2:sweep",
        "fwd sub 3:sweep",
        "fwd sub 4:sweep",
        "fwd sub 5:sweep",
    ]
    assert ops_text(segment_ops(CODA, prelude=P, core=C, coda=D)) == [
        "fwd sub 6:keep",
        "fwd sub 7:keep",
        "fwd head:keep",
        "bwd head:keep",
        "bwd sub 7:keep",
        "bwd sub 6:keep",
    ]
    assert ops_text(segment_ops(PB, prelude=P, core=C, coda=D, edge_regather=True)) == [
        "fwd embed:regather",
        "fwd sub 0:regather",
        "fwd sub 1:regather",
        "bwd sub 1:regather",
        "bwd sub 0:regather",
        "bwd embed:regather",
    ]


def test_core_b_segment_ops_per_item_form():
    def core_b(form, regather=False):
        return ops_text(
            segment_ops(
                CB, prelude=P, core=C, coda=D, core_plan=PLANS[form], core_regather=regather
            )
        )

    assert core_b("whole_round") == [
        "fwd adapter:keep",
        "fwd sub 2:keep",
        "fwd sub 3:keep",
        "fwd sub 4:keep",
        "fwd sub 5:keep",
        "bwd sub 5:keep",
        "bwd sub 4:keep",
        "bwd sub 3:keep",
        "bwd sub 2:keep",
        "bwd adapter:keep",
    ]
    # Blocks in reverse; the prologue (adapter) with block 0.
    assert core_b("block", regather=True) == [
        "fwd sub 4:regather",
        "fwd sub 5:regather",
        "bwd sub 5:regather",
        "bwd sub 4:regather",
        "fwd adapter:regather",
        "fwd sub 2:regather",
        "fwd sub 3:regather",
        "bwd sub 3:regather",
        "bwd sub 2:regather",
        "bwd adapter:regather",
    ]
    # Nested: the rebuild of [0, rebuild_end) sweeps first.
    assert core_b("nested")[:3] == ["fwd adapter:sweep", "fwd sub 2:sweep", "fwd sub 3:sweep"]
    assert core_b("nested")[3:] == core_b("block")


def test_core_b_needs_a_matching_plan():
    with pytest.raises(ValueError, match="core backward plan"):
        segment_ops(CB, prelude=P, core=C, coda=D)
    with pytest.raises(ValueError, match="covers 2 core sublayers"):
        segment_ops(CB, prelude=P, core=C, coda=D, core_plan=CoreBackwardPlan.whole_round(2))


def test_segment_key_validation():
    with pytest.raises(ValueError):
        SegmentKey("sub")
    with pytest.raises(ValueError):
        SegmentKey("head", 3)
    with pytest.raises(ValueError):
        SegmentKey("layer", 0)
    with pytest.raises(ValueError):
        SegmentOp("up", HEAD)
    assert str(sub(3)) == "sub 3" and str(SegmentOp("bwd", HEAD)) == "bwd head"


# ---------------------------------------------------------------------------------------------
# Unit events, K, G
# ---------------------------------------------------------------------------------------------


def test_resident_units_have_no_events():
    for kind in SubRoundType:
        sched = schedule(kind)
        assert sched.events == () and sched.slots == 0 and sched.scratch_entries == 0


def test_sweep_gathers_one_ahead_into_two_alternating_slots():
    sched = schedule(CF, weight=CORE_UNITS, grads=CORE_UNITS)
    assert sched.slots == 2 and sched.scratch_entries == 0
    assert actions(sched) == [
        (0, "before", "gather", ADAPTER, 0, -1),
        (0, "before", "bind", ADAPTER, 0, -1),
        (0, "before", "gather", CORE_A, 1, -1),  # one ahead, issued at the previous bind
        (0, "after", "release", ADAPTER, 0, -1),
        (1, "before", "bind", CORE_A, 1, -1),
        (1, "before", "gather", CORE_B_, 0, -1),
        (2, "after", "release", CORE_A, 1, -1),  # a pair unit: released after its MLP sublayer
        (3, "before", "bind", CORE_B_, 0, -1),
        (4, "after", "release", CORE_B_, 0, -1),
    ]
    assert sched.entry == (ADAPTER, 0) and sched.final_release == (CORE_B_,)


def test_kept_whole_round_holds_every_unit():
    sched = schedule(CB, weight=CORE_UNITS, grads=CORE_UNITS)
    # The adapter and both pairs stay bound to their backward: three slots.
    assert sched.slots == 3 and sched.kept_units_max() == 3
    reduce_ops = [(e.op_index, e.unit) for e in sched.events if e.action == "reduce"]
    # A pair is reduced after its attention sublayer's backward (its last backward op).
    assert reduce_ops == [(6, CORE_B_), (8, CORE_A), (9, ADAPTER)]
    binds = [(e.op_index, e.unit) for e in sched.events if e.action == "bind_grad"]
    assert binds == [(5, CORE_B_), (7, CORE_A), (9, ADAPTER)]
    assert sched.scratch_entries == 2


@pytest.mark.parametrize("form", ["whole_round", "block", "nested"])
def test_regathers_land_in_the_forward_slot(form):
    sched = schedule(CB, form=form, weight=CORE_UNITS, grads=CORE_UNITS, core_regather=True)
    assert sched.slots == 2, "re-gathered items need two slots"
    forward_slot = {}
    for event in sched.events:
        if event.action == "bind" and sched.ops[event.op_index].direction == "fwd":
            forward_slot[event.unit] = event.slot
        if event.action == "gather" and event.regather:
            assert event.slot == forward_slot[event.unit]
    regathered = {e.unit for e in sched.events if e.action == "gather" and e.regather}
    # The last unit of each block is kept (its backward follows its forward at once).
    assert CORE_B_ not in regathered and ADAPTER in regathered
    if form == "whole_round":
        assert CORE_A in regathered


def test_a_unit_regathered_right_after_its_own_release():
    # A coda pair under edge_regather with nothing gathered in between: the one-ahead gather of
    # its re-gather would be issued while the unit itself is still bound, so it waits for (and
    # follows) the unit's own release, into the same slot.
    sched = schedule(CODA, weight=(CODA_U,), grads=(CODA_U,), edge_regather=True)
    assert actions(sched, CODA_U)[:4] == [
        (0, "before", "gather", CODA_U, 0, -1),
        (0, "before", "bind", CODA_U, 0, -1),
        (1, "after", "release", CODA_U, 0, -1),
        (1, "after", "gather", CODA_U, 0, -1),
    ]
    assert sched.slots == 1
    # With the MTP units gathered in between, the re-gather is one ahead of the last of them.
    sched = schedule(CODA, weight=EDGE_UNITS, grads=EDGE_UNITS, edge_regather=True)
    gather = next(e for e in sched.events if e.unit == CODA_U and e.regather)
    assert (gather.op_index, gather.when, gather.slot) == (2, "before", 0)


def test_block_items_reuse_slots_across_blocks():
    sched = schedule(CB, form="block", weight=CORE_UNITS, grads=CORE_UNITS)
    assert sched.slots == 2  # block (4, 5) then block 0 (adapter + (2, 3))
    check_schedule(sched)


def test_coda_head_scratch_entries_and_forward_targets():
    # Move M (output / embedding RSG) and move E (MTP units LW): every head unit is grad-bound
    # in the head segment at once; the output layer's target is bound at its forward (E15).
    weight = (CODA_U, MTP_P, MTP_L)
    grads = weight + (OUT, EMB)
    sched = schedule(CODA, weight=weight, grads=grads, forward=(OUT,))
    head_fwd = 2
    bound_at_fwd = [e.unit for e in sched.events if e.action == "bind_grad" and e.op_index == 2]
    assert bound_at_fwd == [OUT]
    assert sched.scratch_entries == 5  # four head units grad-bound at once, plus one
    # The coda unit and the two MTP units are bound at once in the head segment; the last bind
    # has no gather after it in this sub-round (a cross-sub-round prefetch takes a free physical
    # slot when one exists).
    assert sched.slots == 3
    check_schedule(sched)
    assert sched.ops[head_fwd] == SegmentOp("fwd", HEAD, "keep")


def test_oversize_units_take_transients():
    sched = schedule(PB, weight=(PRE,), grads=(PRE, EMB), oversize=(EMB,))
    emb = [e for e in sched.events if e.unit == EMB]
    assert [e.action for e in emb] == ["bind_grad", "reduce"]
    assert all(e.scratch == -1 for e in emb)
    assert sched.transient_peak == {EMB: 1}


def test_entries_rotate_least_recently_reduced():
    sched = schedule(CB, weight=(), grads=CORE_UNITS)
    entries = [(e.unit, e.scratch) for e in sched.events if e.action == "bind_grad"]
    assert entries == [(CORE_B_, 0), (CORE_A, 1), (ADAPTER, 0)]


def test_derive_rejects_a_second_target_in_one_subround():
    # A forward-captured head target bound again after its reduce (one reduce per item and unit).
    ops = (SegmentOp("fwd", HEAD), SegmentOp("bwd", HEAD), SegmentOp("fwd", HEAD))
    with pytest.raises(ScheduleError, match="second gradient target"):
        derive_unit_events(
            ops, {HEAD: (OUT,)}, weight_units=(), grad_units=(OUT,), forward_grad_units=(OUT,)
        )


def test_check_schedule_rejects_broken_schedules():
    good = schedule(CB, weight=CORE_UNITS, grads=CORE_UNITS, core_regather=True)
    check_schedule(good)

    def broken(mutate):
        events = list(good.events)
        mutate(events)
        return SubRoundSchedule(
            kind=good.kind,
            ops=good.ops,
            events=tuple(events),
            slots=good.slots,
            scratch_entries=good.scratch_entries,
            entry=good.entry,
            final_release=good.final_release,
        )

    # A re-gather into another slot (the E14 hazard).
    def wrong_slot(events):
        i = next(i for i, e in enumerate(events) if e.action == "gather" and e.regather)
        e = events[i]
        events[i] = UnitEvent(e.op_index, e.when, "gather", e.unit, slot=1 - e.slot, regather=True)

    with pytest.raises(ScheduleError):
        check_schedule(broken(wrong_slot))

    # A unit left bound.
    with pytest.raises(ScheduleError, match="left bound"):
        check_schedule(broken(lambda events: events.pop()))

    # A scratch entry rebound before its holder's reduce was issued (the head segment binds
    # four targets at once).
    head = schedule(CODA, grads=(OUT, EMB, MTP_P, MTP_L))
    check_schedule(head)
    events = list(head.events)
    first = next(i for i, e in enumerate(events) if e.action == "bind_grad")
    second = next(i for i, e in enumerate(events) if e.action == "bind_grad" and i > first)
    e = events[second]
    events[second] = UnitEvent(
        e.op_index, e.when, "bind_grad", e.unit, scratch=events[first].scratch
    )
    with pytest.raises(ScheduleError, match="before unit"):
        check_schedule(
            SubRoundSchedule(
                kind=head.kind,
                ops=head.ops,
                events=tuple(events),
                slots=head.slots,
                scratch_entries=head.scratch_entries,
                entry=head.entry,
                final_release=head.final_release,
            )
        )


def test_simulation_grows_k_until_the_rules_hold():
    ops = segment_ops(CB, prelude=P, core=C, coda=D, core_plan=PLANS["whole_round"])
    raw = derive_unit_events(ops, SEGMENT_UNITS, weight_units=CORE_UNITS, grad_units=())
    _, slots, entries, _ = simulate_slots(ops, raw)
    assert (slots, entries) == (3, 0)
    _, slots, _, _ = simulate_slots(ops, raw, min_slots=5)
    assert slots == 5


def test_every_schedule_of_every_mode_mix_passes_the_replay_check():
    for form in PLANS:
        for core_regather in (False, True):
            for edge_regather in (False, True):
                for weight in ((), CORE_UNITS, CORE_UNITS + EDGE_UNITS, EDGE_UNITS):
                    grads = tuple(set(weight) | {OUT, EMB})
                    for kind in SubRoundType:
                        sched = schedule(
                            kind,
                            form=form,
                            weight=weight,
                            grads=grads,
                            forward=(OUT,),
                            core_regather=core_regather,
                            edge_regather=edge_regather,
                        )
                        check_schedule(sched)
                        # Every grad unit visited by a backward op is reduced exactly once.
                        bwd_units = {
                            u
                            for op in sched.ops
                            if op.direction == "bwd"
                            for u in SEGMENT_UNITS.get(op.segment, ())
                        }
                        reduced = [e.unit for e in sched.events if e.action == "reduce"]
                        assert sorted(reduced) == sorted(bwd_units & set(grads))
