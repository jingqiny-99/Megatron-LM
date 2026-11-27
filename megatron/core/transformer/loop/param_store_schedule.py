# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Host-side schedules of the loop parameter store (pure Python, identical on every rank).

A *segment* is the unit of forward visiting and of backward splitting: one sublayer of the
stack (``SegmentKey("sub", i)``), the loop adapter, the embedding call, or the model head (output
layer, MTP depths). Each RSBF sub-round type runs a fixed list of *segment ops*
(``SegmentOp``: a direction, a segment and the policy for the segment's units after the op):

==============  ======================================================================  ========
sub-round       segment ops (in order)                                                  policy
==============  ======================================================================  ========
PRELUDE-F       fwd embed, fwd sub 0 .. P-1                                             sweep
CORE-F          fwd adapter, fwd sub P .. P+C-1                                         sweep
CORE-B          [nested: fwd adapter, fwd sub P .. P+rebuild_end-1 (sweep)]; for each   keep /
                block in reverse: [fwd adapter], fwd its sublayers ascending, bwd its   regather
                sublayers descending, [bwd adapter]
CODA            fwd sub P+C .. L-1, fwd head, bwd head, bwd sub L-1 .. P+C              keep /
                                                                                        regather
PRELUDE-B       fwd embed, fwd sub 0 .. P-1, bwd sub P-1 .. 0, bwd embed                keep /
                                                                                        regather
==============  ======================================================================  ========

From the ops and the units each segment touches, ``build_subround_schedule`` derives the *unit
events* of the store (``UnitEvent``): ``gather`` (a unit's weights into a window slot), ``bind``
(the unit's parameters onto the slot), ``release``, ``bind_grad`` (its fp32 gradient target onto
a scratch entry) and ``reduce`` (the reduce-scatter of that target into the canonical shards).
Layer-window units get weight and gradient events, Resident + sharded grads units gradient
events only, Resident units none (the store handles them at step begin / end).

Slots and scratch entries are assigned statically (``simulate_slots``):

* gathers are issued one ahead in bind order (gather ``n`` right after bind ``n - 1``; gather 0
  at the sub-round's start, or earlier by the store's cross-sub-round lookahead);
* a first gather takes the free slot released earliest; a re-gather for the backward takes the
  unit's forward slot (address-stable windows: every tensor autograd saved in the forward,
  ``weight.t()`` or a ``.view``, reads the unit's own weights again);
* a scratch entry is rebound only after the reduce of its previous holder was issued (checked
  on the host: an event wait on an unrecorded event would be a silent no-op on the device);
* when a constraint fails, the slot (or entry) count grows by one and the simulation reruns, so
  ``K`` and ``G`` are the minimal counts under these rules. A unit is never relocated.

Slots here are *virtual*; the store maps them onto its physical slots per sub-round, which lets
it prefetch the next sub-round's first gather and carry a unit across sub-rounds.
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

from megatron.core.transformer.loop.rsbf_plan import SubRoundType

SegmentKind = Literal["sub", "adapter", "embed", "head"]
Direction = Literal["fwd", "bwd"]
Policy = Literal["sweep", "keep", "regather"]
EventAction = Literal["gather", "bind", "release", "bind_grad", "reduce"]
When = Literal["before", "after"]

SEGMENT_KINDS: Tuple[str, ...] = ("sub", "adapter", "embed", "head")
POLICIES: Tuple[str, ...] = ("sweep", "keep", "regather")

__all__ = [
    "SegmentKey",
    "SegmentOp",
    "ScheduleError",
    "SubRoundSchedule",
    "UnitEvent",
    "build_subround_schedule",
    "check_schedule",
    "derive_unit_events",
    "segment_ops",
    "simulate_slots",
    "store_segments_of_core_plan",
]


class ScheduleError(RuntimeError):
    """A store schedule that breaks a slot / scratch rule (host-side check)."""


@dataclass(frozen=True, order=True)
class SegmentKey:
    """One segment: a stack sublayer (``"sub"``, ``index`` = stack index), the loop adapter,
    the embedding call or the model head."""

    kind: str
    index: int = -1

    def __post_init__(self):
        if self.kind not in SEGMENT_KINDS:
            raise ValueError(f"unknown segment kind {self.kind!r} (one of {SEGMENT_KINDS})")
        if (self.kind == "sub") != (self.index >= 0):
            raise ValueError(f"segment {self.kind!r} with index {self.index}")

    def __str__(self) -> str:
        return f"sub {self.index}" if self.kind == "sub" else self.kind


@dataclass(frozen=True)
class SegmentOp:
    """One forward or backward call of a segment; ``policy`` says what happens to the segment's
    window units after a forward op (``sweep``: released; ``keep``: kept to their backward;
    ``regather``: released and re-gathered into the same slot for the backward)."""

    direction: str
    segment: SegmentKey
    policy: str = "sweep"

    def __post_init__(self):
        if self.direction not in ("fwd", "bwd"):
            raise ValueError(f"unknown direction {self.direction!r}")
        if self.policy not in POLICIES:
            raise ValueError(f"unknown policy {self.policy!r} (one of {POLICIES})")

    def __str__(self) -> str:
        return f"{self.direction} {self.segment}"


@dataclass(frozen=True)
class UnitEvent:
    """One store action on one unit, attached ``before`` or ``after`` op ``op_index``.

    ``slot`` is the virtual window slot (gather / bind / release), ``scratch`` the scratch pool
    entry (bind_grad / reduce; -1 = a zero-filled transient), ``regather`` marks a re-gather into
    the unit's forward slot for its backward.
    """

    op_index: int
    when: str
    action: str
    unit: int
    slot: int = -1
    scratch: int = -1
    regather: bool = False

    def __str__(self) -> str:
        where = f"slot {self.slot}" if self.slot >= 0 else ""
        if self.action in ("bind_grad", "reduce"):
            where = "transient" if self.scratch < 0 else f"entry {self.scratch}"
        extra = " (re-gather)" if self.regather else ""
        return f"{self.when} op {self.op_index}: {self.action} unit {self.unit} {where}{extra}"


@dataclass(frozen=True)
class SubRoundSchedule:
    """The store's schedule of one sub-round type (identical on every rank).

    ``slots`` / ``scratch_entries``: the virtual slots and pool entries it needs (K and G of
    this type). ``entry``: ``(unit, virtual slot)`` of its first gather (the lookahead target);
    ``final_release``: units released after the last op (the carry candidates).
    """

    kind: SubRoundType
    ops: Tuple[SegmentOp, ...]
    events: Tuple[UnitEvent, ...]
    slots: int
    scratch_entries: int
    entry: Optional[Tuple[int, int]]
    final_release: Tuple[int, ...]
    transient_peak: Mapping[int, int] = field(default_factory=dict)  # unit -> times transient
    _index: Dict[Tuple[int, str], Tuple[UnitEvent, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self):
        index: Dict[Tuple[int, str], List[UnitEvent]] = {}
        for event in self.events:
            index.setdefault((event.op_index, event.when), []).append(event)
        object.__setattr__(self, "_index", {key: tuple(value) for key, value in index.items()})

    def at(self, op_index: int, when: str) -> Tuple[UnitEvent, ...]:
        """The events attached ``when`` (``"before"`` / ``"after"``) op ``op_index``, in order."""
        return self._index.get((op_index, when), ())

    def gathers(self) -> Tuple[UnitEvent, ...]:
        """The gather events in issue order."""
        return tuple(event for event in self.events if event.action == "gather")

    def kept_units_max(self) -> int:
        """The most window units bound at once (the cast copies a kept item holds, D-S6-14)."""
        bound, peak = set(), 0
        for event in self.events:
            if event.action == "bind":
                bound.add(event.unit)
                peak = max(peak, len(bound))
            elif event.action == "release":
                bound.discard(event.unit)
        return peak


# ---------------------------------------------------------------------------------------------
# Segment ops
# ---------------------------------------------------------------------------------------------


def _sub(direction: str, index: int, policy: str) -> SegmentOp:
    return SegmentOp(direction, SegmentKey("sub", index), policy)


def store_segments_of_core_plan(core_plan, core_start: int, *, regather: bool) -> List[SegmentOp]:
    """The CORE-B segment ops of an item following ``core_plan`` (a ``CoreBackwardPlan``)."""
    policy = "regather" if regather else "keep"
    ops: List[SegmentOp] = []
    if core_plan.rebuild:
        ops.append(SegmentOp("fwd", SegmentKey("adapter"), "sweep"))
        ops += [_sub("fwd", core_start + i, "sweep") for i in range(core_plan.rebuild_end)]
    for block in reversed(core_plan.blocks):
        if block.with_prologue:
            ops.append(SegmentOp("fwd", SegmentKey("adapter"), policy))
        ops += [_sub("fwd", core_start + i, policy) for i in range(block.start, block.end)]
        ops += [
            _sub("bwd", core_start + i, policy) for i in reversed(range(block.start, block.end))
        ]
        if block.with_prologue:
            ops.append(SegmentOp("bwd", SegmentKey("adapter"), policy))
    return ops


def segment_ops(
    kind: SubRoundType,
    *,
    prelude: int,
    core: int,
    coda: int,
    core_plan=None,
    core_regather: bool = False,
    edge_regather: bool = False,
) -> Tuple[SegmentOp, ...]:
    """The segment ops of one sub-round of ``kind`` (P = ``prelude``, C = ``core``, D = ``coda``
    sublayers; ``core_plan`` is needed for CORE-B)."""
    edge = "regather" if edge_regather else "keep"
    core_end = prelude + core
    if kind is SubRoundType.PRELUDE_F:
        ops = [SegmentOp("fwd", SegmentKey("embed"), "sweep")]
        ops += [_sub("fwd", i, "sweep") for i in range(prelude)]
    elif kind is SubRoundType.CORE_F:
        ops = [SegmentOp("fwd", SegmentKey("adapter"), "sweep")]
        ops += [_sub("fwd", i, "sweep") for i in range(prelude, core_end)]
    elif kind is SubRoundType.CORE_B:
        if core_plan is None:
            raise ValueError("the CORE-B schedule needs the core backward plan")
        if core_plan.core_sublayers != core:
            raise ValueError(
                f"the core plan covers {core_plan.core_sublayers} core sublayers, the stack {core}"
            )
        ops = store_segments_of_core_plan(core_plan, prelude, regather=core_regather)
    elif kind is SubRoundType.CODA:
        coda_range = range(core_end, core_end + coda)
        ops = [_sub("fwd", i, edge) for i in coda_range]
        ops += [SegmentOp("fwd", SegmentKey("head"), "keep")]
        ops += [SegmentOp("bwd", SegmentKey("head"), "keep")]
        ops += [_sub("bwd", i, edge) for i in reversed(coda_range)]
    elif kind is SubRoundType.PRELUDE_B:
        ops = [SegmentOp("fwd", SegmentKey("embed"), edge)]
        ops += [_sub("fwd", i, edge) for i in range(prelude)]
        ops += [_sub("bwd", i, edge) for i in reversed(range(prelude))]
        ops += [SegmentOp("bwd", SegmentKey("embed"), edge)]
    else:  # pragma: no cover - SubRoundType is closed
        raise ValueError(f"unknown sub-round type {kind}")
    return tuple(ops)


# ---------------------------------------------------------------------------------------------
# Unit events
# ---------------------------------------------------------------------------------------------


def derive_unit_events(
    ops: Sequence[SegmentOp],
    segment_units: Mapping[SegmentKey, Sequence[int]],
    *,
    weight_units: Iterable[int],
    grad_units: Iterable[int],
    forward_grad_units: Iterable[int] = (),
) -> List[UnitEvent]:
    """The bind / release / bind_grad / reduce events of ``ops`` (no slots yet).

    ``weight_units`` (Layer window) are bound before their first op of a visit and released after
    its last one (a forward op under ``keep`` keeps the unit to its next use); ``grad_units``
    (Layer window and Resident + sharded grads) get their target bound before their first
    backward op, or before their forward op in the head segment for ``forward_grad_units``
    (Megatron's fused linear captures ``main_grad`` at forward), and reduced after their last
    backward op of the sub-round.
    """
    weight = frozenset(weight_units)
    grads = frozenset(grad_units)
    forward = frozenset(forward_grad_units)
    touches: Dict[int, List[int]] = {}
    for k, op in enumerate(ops):
        for unit in sorted(set(segment_units.get(op.segment, ()))):
            touches.setdefault(unit, []).append(k)
    last_bwd: Dict[int, int] = {}
    for unit, ks in touches.items():
        bwd = [k for k in ks if ops[k].direction == "bwd"]
        if bwd:
            last_bwd[unit] = bwd[-1]

    def next_touch(unit: int, k: int) -> Optional[int]:
        for other in touches[unit]:
            if other > k:
                return other
        return None

    events: List[UnitEvent] = []
    bound: set = set()
    grad_bound: set = set()
    reduced: set = set()
    for k, op in enumerate(ops):
        units = sorted(set(segment_units.get(op.segment, ())))
        for unit in units:
            if unit in weight and unit not in bound:
                events.append(UnitEvent(k, "before", "bind", unit))
                bound.add(unit)
        for unit in units:
            wants_grad = op.direction == "bwd" or (
                op.segment.kind == "head" and unit in forward and unit in last_bwd
            )
            if unit in grads and unit not in grad_bound and wants_grad:
                if unit in reduced:
                    raise ScheduleError(
                        f"unit {unit} needs a second gradient target in one sub-round (op {k}: "
                        f"{op}); one reduce per (item, unit) is the store's rule"
                    )
                events.append(UnitEvent(k, "before", "bind_grad", unit))
                grad_bound.add(unit)
        for unit in units:
            if unit in grad_bound and last_bwd.get(unit) == k:
                events.append(UnitEvent(k, "after", "reduce", unit))
                grad_bound.discard(unit)
                reduced.add(unit)
        for unit in units:
            if unit not in bound:
                continue
            nxt = next_touch(unit, k)
            if nxt is None:
                release = True
            elif nxt == k + 1:
                release = False
            elif op.direction == "fwd" and op.policy == "keep":
                release = False
            else:
                release = True
            if release:
                events.append(UnitEvent(k, "after", "release", unit))
                bound.discard(unit)
    if bound or grad_bound:  # pragma: no cover - the rules above close every visit
        raise ScheduleError(f"units left bound {sorted(bound)} / grad-bound {sorted(grad_bound)}")
    return events


# ---------------------------------------------------------------------------------------------
# Slot and scratch assignment
# ---------------------------------------------------------------------------------------------


class _NoFit(Exception):
    def __init__(self, what: str):
        super().__init__(what)
        self.what = what


def _place(
    ops: Sequence[SegmentOp],
    raw: Sequence[UnitEvent],
    slots: int,
    entries: int,
    oversize: FrozenSet[int],
) -> Tuple[List[UnitEvent], Dict[int, int]]:
    """One simulation at ``slots`` / ``entries``; raises ``_NoFit`` naming the short resource."""
    binds = [event for event in raw if event.action == "bind"]
    occupant: List[Optional[int]] = [None] * slots
    freed = [s - slots for s in range(slots)]  # never-used slots count as released earliest
    entry_holder: List[Optional[int]] = [None] * entries
    entry_freed = [e - entries for e in range(entries)]
    reserved: Dict[int, int] = {}  # unit -> slot reserved by its issued gather
    unit_slot: Dict[int, int] = {}  # unit -> slot while bound
    forward_slot: Dict[int, int] = {}
    unit_entry: Dict[int, int] = {}
    transients: Dict[int, int] = {}
    clock = 0
    out: List[UnitEvent] = []
    deferred: List[int] = []  # a re-gather of a unit still bound: issued at its release

    def issue(n: int, op_index: int, when: str) -> None:
        bind = binds[n]
        unit = bind.unit
        regather = ops[bind.op_index].direction == "bwd"
        if regather:
            if unit not in forward_slot:
                raise ScheduleError(f"unit {unit} is bound for a backward it was not forwarded in")
            slot = forward_slot[unit]
            if occupant[slot] == unit and unit in unit_slot:
                deferred.append(n)  # its own forward binding: re-gather right after release
                return
            if occupant[slot] is not None:
                raise _NoFit("slot")
        else:
            free = [s for s in range(slots) if occupant[s] is None]
            if not free:
                raise _NoFit("slot")
            slot = min(free, key=lambda s: (freed[s], s))
        occupant[slot] = unit
        reserved[unit] = slot
        out.append(UnitEvent(op_index, when, "gather", unit, slot=slot, regather=regather))

    if binds:
        issue(0, 0, "before")
    next_bind = 0
    for event in raw:
        if event.action == "bind":
            unit = event.unit
            slot = reserved.pop(unit)
            unit_slot[unit] = slot
            regather = ops[event.op_index].direction == "bwd"
            if not regather:
                forward_slot[unit] = slot
            out.append(UnitEvent(event.op_index, event.when, "bind", unit, slot=slot))
            next_bind += 1
            if next_bind < len(binds):
                issue(next_bind, event.op_index, event.when)
        elif event.action == "release":
            slot = unit_slot.pop(event.unit)
            occupant[slot] = None
            freed[slot] = clock
            clock += 1
            out.append(UnitEvent(event.op_index, event.when, "release", event.unit, slot=slot))
            if deferred and binds[deferred[0]].unit == event.unit:
                issue(deferred.pop(0), event.op_index, event.when)
        elif event.action == "bind_grad":
            if event.unit in oversize:
                entry = -1
                transients[event.unit] = transients.get(event.unit, 0) + 1
            else:
                free = [e for e in range(entries) if entry_holder[e] is None]
                if not free:
                    raise _NoFit("scratch")
                entry = min(free, key=lambda e: (entry_freed[e], e))
                entry_holder[entry] = event.unit
            unit_entry[event.unit] = entry
            out.append(
                UnitEvent(event.op_index, event.when, "bind_grad", event.unit, scratch=entry)
            )
        elif event.action == "reduce":
            entry = unit_entry.pop(event.unit)
            if entry >= 0:
                entry_holder[entry] = None
                entry_freed[entry] = clock
                clock += 1
            out.append(UnitEvent(event.op_index, event.when, "reduce", event.unit, scratch=entry))
        else:  # pragma: no cover - derive_unit_events emits nothing else
            raise ScheduleError(f"unexpected raw event {event}")
    return out, transients


def simulate_slots(
    ops: Sequence[SegmentOp],
    raw: Sequence[UnitEvent],
    *,
    oversize: Iterable[int] = (),
    min_slots: int = 1,
    min_entries: int = 2,
) -> Tuple[Tuple[UnitEvent, ...], int, int, Dict[int, int]]:
    """Assign virtual slots and scratch entries to ``raw`` (``derive_unit_events``' output).

    Returns ``(events with the gathers inserted, K, G, transients per unit)``: the smallest slot
    count (from ``min_slots`` up) under the one-ahead gather and the address-stable re-gather
    rules, and ``G = max(min_entries, 1 + the most pooled targets bound at once)`` entries (or
    more if the rotation needs them). ``oversize`` units take transients instead of entries.
    """
    oversized = frozenset(oversize)
    units = {event.unit for event in raw}
    # One entry beyond the most pooled targets bound at once, so the next unit's backward never
    # waits on the reduce-scatter still reading the previous entry.
    live = concurrent = 0
    for event in raw:
        if event.unit in oversized:
            continue
        if event.action == "bind_grad":
            live += 1
            concurrent = max(concurrent, live)
        elif event.action == "reduce":
            live -= 1
    min_entries = max(min_entries, concurrent + 1)
    has_binds = any(event.action == "bind" for event in raw)
    has_grads = any(event.action == "bind_grad" and event.unit not in oversized for event in raw)
    slots = max(min_slots, 1) if has_binds else 0
    entries = max(min_entries, 1) if has_grads else 0
    limit = 2 * len(units) + 4
    while True:
        try:
            events, transients = _place(ops, raw, slots, entries, oversized)
        except _NoFit as exc:
            if exc.what == "slot":
                slots += 1
            else:
                entries += 1
            if slots > limit or entries > limit:
                raise ScheduleError(
                    f"no slot / scratch assignment within {limit} slots and entries"
                ) from exc
            continue
        return tuple(events), slots, entries, transients


def check_schedule(schedule: SubRoundSchedule) -> None:
    """Replay ``schedule`` and raise ``ScheduleError`` on any broken rule.

    A gather into an occupied slot; a re-gather into a slot other than the unit's forward slot; a
    bind without its gather; a scratch entry rebound before the reduce of its previous holder was
    issued; a unit left bound or grad-bound at the end.
    """
    occupant: Dict[int, int] = {}
    gathered: Dict[int, int] = {}
    bound: Dict[int, int] = {}
    forward_slot: Dict[int, int] = {}
    holder: Dict[int, int] = {}
    grad_bound: set = set()
    for event in schedule.events:
        if event.slot >= schedule.slots or event.scratch >= schedule.scratch_entries:
            raise ScheduleError(f"{event}: beyond the schedule's slots / entries")
        if event.action == "gather":
            if event.slot in occupant:
                raise ScheduleError(f"{event}: slot held by unit {occupant[event.slot]}")
            if event.regather and forward_slot.get(event.unit) != event.slot:
                raise ScheduleError(
                    f"{event}: re-gather outside the unit's forward slot "
                    f"{forward_slot.get(event.unit)}"
                )
            occupant[event.slot] = event.unit
            gathered[event.unit] = event.slot
        elif event.action == "bind":
            if gathered.pop(event.unit, None) != event.slot:
                raise ScheduleError(f"{event}: bound without its gather into that slot")
            bound[event.unit] = event.slot
            if schedule.ops[event.op_index].direction == "fwd":
                forward_slot[event.unit] = event.slot
        elif event.action == "release":
            if bound.pop(event.unit, None) != event.slot:
                raise ScheduleError(f"{event}: released while not bound there")
            occupant.pop(event.slot, None)
        elif event.action == "bind_grad":
            if event.unit in grad_bound:
                raise ScheduleError(f"{event}: target bound twice")
            if event.scratch >= 0:
                if event.scratch in holder:
                    raise ScheduleError(
                        f"{event}: entry rebound before unit {holder[event.scratch]}'s reduce "
                        "was issued"
                    )
                holder[event.scratch] = event.unit
            grad_bound.add(event.unit)
        elif event.action == "reduce":
            if event.unit not in grad_bound:
                raise ScheduleError(f"{event}: reduce without a bound target")
            grad_bound.discard(event.unit)
            if event.scratch >= 0:
                if holder.get(event.scratch) != event.unit:
                    raise ScheduleError(f"{event}: reduce of an entry the unit does not hold")
                del holder[event.scratch]
        else:
            raise ScheduleError(f"unknown action in {event}")
    if bound or gathered or grad_bound:
        raise ScheduleError(
            f"left bound {sorted(bound)}, gathered {sorted(gathered)}, grad-bound "
            f"{sorted(grad_bound)}"
        )


def build_subround_schedule(
    kind: SubRoundType,
    *,
    prelude: int,
    core: int,
    coda: int,
    segment_units: Mapping[SegmentKey, Sequence[int]],
    weight_units: Iterable[int],
    grad_units: Iterable[int],
    forward_grad_units: Iterable[int] = (),
    oversize_units: Iterable[int] = (),
    core_plan=None,
    core_regather: bool = False,
    edge_regather: bool = False,
) -> SubRoundSchedule:
    """The checked schedule of one sub-round of ``kind`` (see the module docstring)."""
    ops = segment_ops(
        kind,
        prelude=prelude,
        core=core,
        coda=coda,
        core_plan=core_plan,
        core_regather=core_regather,
        edge_regather=edge_regather,
    )
    raw = derive_unit_events(
        ops,
        segment_units,
        weight_units=weight_units,
        grad_units=grad_units,
        forward_grad_units=forward_grad_units,
    )
    events, slots, entries, transients = simulate_slots(ops, raw, oversize=oversize_units)
    gathers = [event for event in events if event.action == "gather"]
    entry = (gathers[0].unit, gathers[0].slot) if gathers else None
    last = len(ops) - 1
    final_release = tuple(
        event.unit
        for event in events
        if event.action == "release" and event.op_index == last and event.when == "after"
    )
    schedule = SubRoundSchedule(
        kind=kind,
        ops=ops,
        events=events,
        slots=slots,
        scratch_entries=entries,
        entry=entry,
        final_release=final_release,
        transient_peak=transients,
    )
    check_schedule(schedule)
    return schedule
