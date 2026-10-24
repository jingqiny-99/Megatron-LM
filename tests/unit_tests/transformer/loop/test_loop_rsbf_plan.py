# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests of the multi-EU RSBF host planner (``loop/rsbf_plan.py``): pure host logic, no torch
and no process group, so every case runs anywhere (CPU, CI under torchrun).

P1 determinism and digests, P2 min-move rebalancing, P3 no moves at round 1 / without
rebalancing, P4 mirrored chain records and one-hop chain transfers, P5 placement, P6 typed
sub-rounds and the W = 1 stage-2 item order, P7 the move gate on synthetic workloads, P8
checkpoint lifetimes against a fake store, P9 online exits == the whole-step known-depth plan,
P10 the block-input stash (and check_recompute trace) slots of moved CORE items.
"""

import hashlib
import random

import pytest

from megatron.core.transformer.loop.rsbf_plan import (
    BATCH_SLOTS,
    CRIT_STATE,
    EXIT_FLAGS,
    GRAD_STATE,
    INJECTION,
    LOSS_RECORD,
    ROUND_INPUT,
    FreeInjection,
    MovePhase,
    SubRound,
    SubRoundType,
    Transfer,
    WavePlanner,
    _encode,
    block_slot,
    chain_slots,
    known_depth_plan,
    parse_sublayer_slot,
    trace_slot,
)

BATCH = ("tokens", "position_ids", "labels", "loss_mask")
FORWARD_TYPES = (SubRoundType.PRELUDE_F, SubRoundType.CORE_F)
WORKLOADS = {
    "R4-unif": [0.25, 0.25, 0.25, 0.25],
    "R4-geom": [0.4, 0.3, 0.2, 0.1],
    "R4-skew": [0.1, 0.2, 0.3, 0.4],
}
# The stage-2 executor test depth sets (test_loop_rsbf_executor.py).
STAGE2_DEPTHS = [(1, 3, 2, 4), (1, 3, 2, 4, 2, 1, 3, 4), (2, 1, 2, 2), (1, 3, 2, 4, 4, 2, 1, 1)]


def _home(num_eus, per_eu, mbs=1):
    """The executor's home map: microbatch j, row b of DP rank q is sample j*MBS*W + q*MBS + b."""
    return [(g // mbs) % num_eus for g in range(num_eus * per_eu)]


def _planner(num_eus, home, **kwargs):
    options = dict(
        max_rounds=4,
        fill="drain",
        rebalance=True,
        tail_item_moves=True,
        criterion_differentiable=False,
        check_recompute=False,
        batch_slots=BATCH,
        report_keys=1,
    )
    options.update(kwargs)
    return WavePlanner(num_eus=num_eus, home=home, **options)


def _online_plan(planner, depths, shuffle=None):
    """The online call order: exits of round r are recorded after slot r's backward phase."""
    steps = planner.prelude_steps()
    r = 1
    while planner.active() or planner.pending():
        swept = bool(planner.active())
        if swept:
            steps += planner.forward_steps(r)
        steps += planner.backward_steps(final=False)
        if swept:
            exited = [g for g in planner.active() if depths[g] == r]
            if shuffle is not None:
                shuffle.shuffle(exited)
            planner.record_exits(r, exited)
        r += 1
    steps += planner.backward_steps(final=True)
    return steps


def _random_depths(rng, count, max_rounds=4, probs=None):
    if probs is None:
        return [rng.randint(1, max_rounds) for _ in range(count)]
    return [rng.choices(range(1, len(probs) + 1), probs)[0] for _ in range(count)]


def _subrounds(steps, types=None):
    return [s for s in steps if isinstance(s, SubRound) and (types is None or s.type in types)]


def _transfers(steps, kind=None):
    return [
        t
        for s in steps
        if isinstance(s, MovePhase)
        for t in s.transfers
        if kind is None or t.kind == kind
    ]


def _placements(steps):
    """(kind, g, r) -> EU for every real item."""
    where = {}
    for step in _subrounds(steps):
        for e, item in enumerate(step.items):
            if item is not None:
                key = (item.kind, item.sample, item.round)
                assert key not in where, f"{key} runs twice"
                where[key] = e
    return where


# ---------------------------------------------------------------------------------------
# P8's fake store: the state store's write-once / release-once rules per EU plus the "sent"
# rule of moves, chain-record locations and batch presence.
# ---------------------------------------------------------------------------------------


class _FakeEU:
    def __init__(self):
        self.entries = set()
        self.released = {}
        self.batch = set()

    def put(self, key):
        assert key not in self.entries, f"{key} written twice"
        assert key not in self.released, f"{key} written after it was {self.released[key]}"
        self.entries.add(key)

    def put_received(self, key):
        assert key not in self.entries, f"{key} received while live"
        assert self.released.get(key) in (
            None,
            "sent",
        ), f"{key} received after it was {self.released[key]}"
        self.entries.add(key)

    def require(self, key):
        assert key in self.entries, f"{key} read but not live ({self.released.get(key)})"

    def release(self, key, how):
        self.require(key)
        self.entries.remove(key)
        self.released[key] = how


def _replay(steps, num_eus, home, *, check_recompute=False, core_stash=(), core_trace=0):
    """Execute a plan against fake per-EU stores; returns the EUs (asserts on any misuse).
    ``core_stash`` / ``core_trace``: the per-sublayer entries each CORE sweep puts and its CORE
    item consumes (``("block:<l>", g, r)`` / ``("trace:<l>", g, r)``)."""
    eus = [_FakeEU() for _ in range(num_eus)]
    sublayer_entries = [f"block:{l}" for l in core_stash] + [
        f"trace:{l}" for l in range(core_trace)
    ]
    for g, h in enumerate(home):
        eus[h].batch.add(g)
    chain_at = {}
    consume = "taken" if check_recompute else "freed"
    for step in steps:
        if isinstance(step, SubRound):
            for e, item in enumerate(step.items):
                if item is None:
                    continue
                eu, g, r = eus[e], item.sample, item.round
                assert g in eu.batch, f"{item} runs on EU {e} without the sample's batch"
                if item.kind == "prelude_sweep":
                    assert e == home[g]
                    eu.put(("injection", g, 0))
                    eu.put(("crit", g, 0))
                elif item.kind == "core_sweep":
                    eu.require(("injection", g, 0))
                    if r > 1:
                        eu.require(("state", g, r - 1))
                    eu.require(("crit", g, r - 1))
                    eu.put(("state", g, r))
                    eu.put(("crit", g, r))
                    for kind in sublayer_entries:
                        eu.put((kind, g, r))
                elif item.kind == "coda":
                    assert g not in chain_at
                    eu.release(("state", g, r), consume)
                    eu.release(("crit", g, r), "freed")
                    chain_at[g] = e
                elif item.kind == "core":
                    assert chain_at[g] == e, f"{item} on EU {e}, its record is on {chain_at[g]}"
                    eu.require(("injection", g, 0))
                    if r > 1:
                        eu.release(("state", g, r - 1), consume)
                    eu.release(("crit", g, r - 1), "freed")
                    # The round's own stash: on this EU (swept here, or carried by the item).
                    for kind in sublayer_entries:
                        how = "taken" if kind.startswith("trace") else consume
                        eu.release((kind, g, r), how)
                else:
                    assert item.kind == "prelude" and e == home[g]
                    assert chain_at.pop(g) == e
                    eu.release(("injection", g, 0), "freed")
        elif isinstance(step, MovePhase):
            # Every send of the phase reads the pre-phase state; installs follow.
            for t in step.transfers:
                src, dst = eus[t.src], eus[t.dst]
                assert 0 <= t.src < num_eus and 0 <= t.dst < num_eus and t.src != t.dst
                if not (t.kind == "item" and t.item == "core"):
                    # Only a moved CORE item carries a round's stash (a sample move carries
                    # the next round's input, never a stash).
                    assert not any(":" in kind for kind, _ in t.frees), t
                    assert not any(":" in slot for slot in t.slots), t
                if t.kind == "chain":
                    assert chain_at[t.sample] == t.src
                for kind, r in t.frees:
                    src.require((kind, t.sample, r))
                if INJECTION in t.slots:
                    src.require(("injection", t.sample, 0))
                if t.kind != "chain":
                    assert t.sample in src.batch
            for t in step.transfers:
                src, dst = eus[t.src], eus[t.dst]
                if t.kind == "chain":
                    chain_at[t.sample] = t.dst
                    continue
                for kind, r in t.frees:
                    src.release((kind, t.sample, r), "sent")
                    dst.put_received((kind, t.sample, r))
                if INJECTION in t.slots:
                    dst.put_received(("injection", t.sample, 0))
                elif t.item != "coda":  # a CODA item reads no injection
                    dst.require(("injection", t.sample, 0))
                if "tokens" in t.slots:
                    assert t.sample not in dst.batch
                    dst.batch.add(t.sample)
                else:
                    assert t.sample in dst.batch
        else:
            assert isinstance(step, FreeInjection)
            eus[step.eu].release(("injection", step.sample, 0), "freed")
    for e, eu in enumerate(eus):
        assert not eu.entries, f"EU {e} ends the wave with {sorted(eu.entries)}"
    assert not chain_at
    return eus


# ---------------------------------------------------------------------------------------
# P1: determinism
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_eus", [1, 2, 4, 8])
def test_plans_are_deterministic(num_eus):
    rng = random.Random(num_eus)
    home = _home(num_eus, 4, mbs=2)
    depths = _random_depths(rng, len(home))
    plans = []
    for repeat in range(3):
        planner = _planner(num_eus, home)
        steps = _online_plan(planner, depths, shuffle=random.Random(repeat) if repeat else None)
        plans.append((steps, planner.digest()))
    assert plans[0] == plans[1] == plans[2]
    other = list(depths)
    other[0] = other[0] % 4 + 1
    planner = _planner(num_eus, home)
    _online_plan(planner, other)
    assert planner.digest() != plans[0][1]
    # The digest covers the inputs too.
    assert _planner(num_eus, home).digest() != _planner(num_eus, home, fill="sample_order").digest()


# ---------------------------------------------------------------------------------------
# P2 / P3: min-move rebalancing
# ---------------------------------------------------------------------------------------


def _check_rebalancing(steps, actives, num_eus, home):
    """Per forward round: moves = sum of surplus above cap, after them max load = cap,
    endpoints in the domain, and the injection-copy preference (``actives[r]``: the samples
    swept in round ``r``)."""
    copies = [{h} for h in home]
    loc = list(home)
    for step in steps:
        if isinstance(step, MovePhase) and step.transfers[0].kind == "sample":
            r = step.transfers[0].round
            active = actives[r]
            counts = [0] * num_eus
            for g in active:
                counts[loc[g]] += 1
            cap = -(-len(active) // num_eus)
            assert len(step.transfers) == sum(max(0, c - cap) for c in counts)
            for t in step.transfers:
                assert t.kind == "sample" and t.round == r
                assert t.src == loc[t.sample] and 0 <= t.dst < num_eus
                counts[t.src] -= 1
                counts[t.dst] += 1
            assert max(counts) == cap
            for t in step.transfers:
                if INJECTION in t.slots:
                    # The planner would have chosen a copy holder with room left.
                    assert all(counts[x] == cap for x in copies[t.sample]), t
                    copies[t.sample].add(t.dst)
                else:
                    assert t.dst in copies[t.sample]
                loc[t.sample] = t.dst
        elif isinstance(step, MovePhase):
            for t in step.transfers:
                if INJECTION in t.slots:
                    copies[t.sample].add(t.dst)
        elif isinstance(step, FreeInjection):
            copies[step.sample].discard(step.eu)


def _forward_actives(depths):
    return {r: [g for g, d in enumerate(depths) if d >= r] for r in range(1, max(depths) + 1)}


@pytest.mark.parametrize("num_eus,per_eu,seed", [(2, 4, 0), (3, 3, 1), (4, 8, 2), (8, 4, 3)])
def test_min_move_rebalancing(num_eus, per_eu, seed):
    rng = random.Random(seed)
    home = _home(num_eus, per_eu)
    for _ in range(20):
        depths = _random_depths(rng, len(home))
        planner = _planner(num_eus, home)
        steps = known_depth_plan(planner, depths)
        actives = _forward_actives(depths)
        _check_rebalancing(steps, actives, num_eus, home)
        # CORE-F sub-rounds per round: cap; nulls = W * cap - A.
        for r, active in actives.items():
            subs = [s for s in _subrounds(steps, (SubRoundType.CORE_F,)) if _round_of(s) == r]
            cap = -(-len(active) // num_eus)
            assert len(subs) == cap
            assert sum(i is None for s in subs for i in s.items) == num_eus * cap - len(active)


def _round_of(subround):
    return next(i.round for i in subround.items if i is not None)


def test_rebalancing_prefers_an_injection_copy():
    """W = 2, home EUs alternate. Round 2: active {1, 3, 5} on EU 1 and {4} on EU 0, cap 2:
    sample 1 (lowest g, no copy elsewhere) moves to EU 0 with its injection and batch. Round 3:
    active {1, 4} both on EU 0, cap 1: sample 1, whose copy sits on EU 1 (room), is preferred
    over sample 4 and goes back without injection or batch."""
    home = [0, 1, 0, 1, 0, 1]
    depths = [1, 3, 1, 2, 3, 2]
    planner = _planner(2, home)
    steps = known_depth_plan(planner, depths)
    moves = _transfers(steps, "sample")
    assert [(t.sample, t.src, t.dst, t.round) for t in moves] == [(1, 1, 0, 2), (1, 0, 1, 3)]
    assert moves[0].slots == (ROUND_INPUT, CRIT_STATE, INJECTION, EXIT_FLAGS) + BATCH
    assert moves[1].slots == (ROUND_INPUT, CRIT_STATE, EXIT_FLAGS)
    assert moves[1].frees == (("state", 2), ("crit", 2))
    assert [planner.eu_of_round(1, r) for r in (1, 2, 3)] == [1, 0, 1]
    # The copy on EU 0 lives until CORE(1, 1) ran, then FreeInjection(1, 0).
    frees = [s for s in steps if isinstance(s, FreeInjection)]
    assert frees == [FreeInjection(1, 0)]
    _replay(steps, 2, home)


def test_injection_preference_sends_a_sample_back_without_e():
    """Search small plans for a move back to a copy holder: it never carries the injection."""
    rng = random.Random(7)
    found = 0
    for _ in range(400):
        num_eus = rng.randint(2, 4)
        home = _home(num_eus, rng.randint(2, 4))
        depths = _random_depths(rng, len(home))
        planner = _planner(num_eus, home)
        steps = known_depth_plan(planner, depths)
        held = [{h} for h in home]
        for t in _transfers(steps, "sample"):
            if t.dst in held[t.sample]:
                assert INJECTION not in t.slots and "tokens" not in t.slots
                found += 1
            else:
                assert INJECTION in t.slots and "tokens" in t.slots
            held[t.sample].add(t.dst)
    assert found > 0


@pytest.mark.parametrize("fill", ["drain", "sample_order"])
def test_no_moves_at_round_one_or_without_rebalancing(fill):
    rng = random.Random(3)
    for num_eus in (2, 4, 8):
        home = _home(num_eus, 4)
        for _ in range(10):
            depths = _random_depths(rng, len(home))
            planner = _planner(num_eus, home, fill=fill)
            steps = known_depth_plan(planner, depths)
            assert planner.stats.moves_by_round[0] == 0
            assert all(t.round >= 2 for t in _transfers(steps, "sample"))
            planner = _planner(num_eus, home, fill=fill, rebalance=False)
            steps = known_depth_plan(planner, depths)
            assert _transfers(steps, "sample") == []
            where = _placements(steps)
            for g, d in enumerate(depths):
                for r in range(1, d + 1):
                    assert where[("core_sweep", g, r)] == home[g]
            # Lockstep length = the largest per-EU active count.
            for r, active in _forward_actives(depths).items():
                counts = [sum(1 for g in active if home[g] == e) for e in range(num_eus)]
                subs = [s for s in _subrounds(steps, (SubRoundType.CORE_F,)) if _round_of(s) == r]
                assert len(subs) == max(counts)


# ---------------------------------------------------------------------------------------
# P4 / P5: chain records and placement
# ---------------------------------------------------------------------------------------


def _check_placement_and_chains(steps, planner, home, depths, num_eus, item_moves):
    where = _placements(steps)
    moved_items = {(t.item, t.sample, t.round): t for t in _transfers(steps, "item")}
    for g, d in enumerate(depths):
        assert planner.exit_round(g) == d
        assert where[("prelude_sweep", g, 0)] == home[g]
        assert where[("core_sweep", g, 1)] == home[g]
        for r in range(1, d + 1):
            assert where[("core_sweep", g, r)] == planner.eu_of_round(g, r)
        expected = {("coda", g, d): planner.eu_of_round(g, d)}
        expected.update({("core", g, r): planner.eu_of_round(g, r) for r in range(1, d + 1)})
        for key, eu in expected.items():
            moved = moved_items.get(key)
            if moved is None:
                assert where[key] == eu, key
            else:
                assert moved.src == eu and moved.dst == where[key], key
        # The loss record goes home: PRELUDE always at home.
        assert where[("prelude", g, 0)] == home[g]
    if not item_moves:
        assert not moved_items
    # One hop: a chain transfer goes from the EU that ran the producer to the EU that runs
    # the next item of the chain.
    for t in _transfers(steps, "chain"):
        assert where[(t.item, t.sample, t.round)] == t.src
        successor = ("core", t.sample, t.round) if t.item == "coda" else None
        if t.item == "core":
            successor = ("core", t.sample, t.round - 1) if t.round > 1 else None
            successor = successor or ("prelude", t.sample, 0)
        assert where[successor] == t.dst
        assert t.slots == chain_slots(
            t.item, t.round, criterion_differentiable=False, check_recompute=False
        )
    # Item moves only in chain-parallel sub-rounds, for CORE / CODA.
    for t in moved_items.values():
        assert t.item in ("core", "coda")


@pytest.mark.parametrize("item_moves", [False, True])
@pytest.mark.parametrize("fill", ["drain", "sample_order"])
def test_chain_records_mirror_the_sample_moves(item_moves, fill):
    rng = random.Random(11)
    for num_eus in (2, 3, 4, 8):
        home = _home(num_eus, 3)
        for _ in range(15):
            depths = _random_depths(rng, len(home))
            planner = _planner(num_eus, home, fill=fill, tail_item_moves=item_moves)
            steps = known_depth_plan(planner, depths)
            _check_placement_and_chains(steps, planner, home, depths, num_eus, item_moves)
            if item_moves:
                continue
            # I4: chain transfers are exactly the reverses of the sample moves.
            reverses = sorted(
                (t.sample, t.dst, t.src, t.round) for t in _transfers(steps, "sample")
            )
            chains = sorted((t.sample, t.src, t.dst, t.round) for t in _transfers(steps, "chain"))
            assert all(t.item == "core" for t in _transfers(steps, "chain"))
            assert chains == reverses


def test_item_moves_happen_only_in_the_tail():
    rng = random.Random(5)
    seen = 0
    for _ in range(200):
        num_eus = rng.randint(2, 6)
        home = _home(num_eus, rng.randint(1, 4))
        depths = _random_depths(rng, len(home))
        planner = _planner(num_eus, home)
        steps = known_depth_plan(planner, depths)
        for index, step in enumerate(steps):
            if isinstance(step, MovePhase) and any(t.kind == "item" for t in step.transfers):
                seen += 1
                nxt = steps[index + 1]
                assert isinstance(nxt, SubRound) and nxt.type in (
                    SubRoundType.CORE_B,
                    SubRoundType.CODA,
                )
                # Items still to run in this backward phase (this sub-round included).
                remaining = 0
                for later in steps[index + 1 :]:
                    if isinstance(later, SubRound) and later.type in FORWARD_TYPES:
                        break
                    if isinstance(later, SubRound):
                        remaining += sum(i is not None for i in later.items)
                assert remaining < 2 * num_eus
                for t in step.transfers:
                    if t.kind == "item":
                        assert nxt.items[t.dst] is not None
                        assert (nxt.items[t.dst].kind, nxt.items[t.dst].sample) == (
                            t.item,
                            t.sample,
                        )
    assert seen > 0


# ---------------------------------------------------------------------------------------
# P6: typed sub-rounds and W = 1
# ---------------------------------------------------------------------------------------


def _stage2_item_order(depths, fill):
    order = range(len(depths))
    if fill == "drain":
        order = sorted(order, key=lambda i: (depths[i], i))
    items = []
    for i in order:
        d = depths[i]
        items.append(("coda", i, d))
        items += [("core", i, r) for r in range(d, 0, -1)]
        items.append(("prelude", i, 0))
    return items


@pytest.mark.parametrize("fill", ["drain", "sample_order"])
@pytest.mark.parametrize("depths", STAGE2_DEPTHS)
@pytest.mark.parametrize("online", [False, True])
def test_one_eu_plan_is_the_stage2_item_order(fill, depths, online):
    home = [0] * len(depths)
    planner = _planner(1, home, fill=fill)
    steps = _online_plan(planner, depths) if online else known_depth_plan(planner, depths)
    items = [
        (s.items[0].kind, s.items[0].sample, s.items[0].round)
        for s in _subrounds(steps)
        if s.type not in FORWARD_TYPES
    ]
    assert items == _stage2_item_order(depths, fill)
    assert all(isinstance(s, SubRound) for s in steps)  # no moves, no FreeInjection
    assert all(i is not None for s in steps for i in s.items)  # no nulls
    sweeps = [(s.items[0].kind, s.items[0].sample) for s in _subrounds(steps, FORWARD_TYPES)]
    assert sweeps[: len(depths)] == [("prelude_sweep", g) for g in range(len(depths))]


def test_subrounds_are_typed_and_respect_chain_order():
    rng = random.Random(9)
    for num_eus in (2, 4, 8):
        home = _home(num_eus, 4)
        for fill in ("drain", "sample_order"):
            depths = _random_depths(rng, len(home))
            planner = _planner(num_eus, home, fill=fill)
            steps = known_depth_plan(planner, depths)
            done = set()
            for step in _subrounds(steps):
                assert len(step.items) == num_eus
                for item in step.items:
                    if item is None:
                        continue
                    assert item.type is step.type
                    g = item.sample
                    if item.kind == "core":
                        before = ("coda", g, depths[g]) if item.round == depths[g] else None
                        before = before or ("core", g, item.round + 1)
                        assert before in done, (item, before)
                    elif item.kind == "prelude":
                        assert ("core", g, 1) in done
                    done.add((item.kind, item.sample, item.round))
            stats = planner.stats
            assert stats.subrounds[SubRoundType.PRELUDE_F] == 4
            assert stats.nulls[SubRoundType.PRELUDE_F] == 0
            total = sum(stats.subrounds.values())
            assert total == len(_subrounds(steps))


def test_backward_type_follows_the_majority_then_priority():
    """The slot example of the design (8.7): W = 2, EU 0 loads samples 0-4 and EU 1 samples
    5-7; samples 3, 4 (EU 0) and 6, 7 (EU 1) exit at round 1. Their backward phase runs bulk
    CODA, CORE-B, PRELUDE-B sub-rounds per chain pair (a CORE / CODA tie goes to the
    highest-priority chain, 3), with no moves. Round 2 then moves sample 0 to EU 1."""
    home = [0, 0, 0, 0, 0, 1, 1, 1]
    depths = [2, 2, 2, 1, 1, 4, 1, 1]
    planner = _planner(2, home)
    prelude = planner.prelude_steps()
    assert len(prelude) == 5 and planner.stats.nulls[SubRoundType.PRELUDE_F] == 2
    planner.forward_steps(1)
    planner.record_exits(1, [g for g in range(8) if depths[g] == 1])
    backward = planner.backward_steps(final=False)
    got = [
        (s.type, tuple(None if i is None else (i.kind, i.sample) for i in s.items))
        for s in backward
    ]
    assert got == [
        (SubRoundType.CODA, (("coda", 3), ("coda", 6))),
        (SubRoundType.CORE_B, (("core", 3), ("core", 6))),
        (SubRoundType.PRELUDE_B, (("prelude", 3), ("prelude", 6))),
        (SubRoundType.CODA, (("coda", 4), ("coda", 7))),
        (SubRoundType.CORE_B, (("core", 4), ("core", 7))),
        (SubRoundType.PRELUDE_B, (("prelude", 4), ("prelude", 7))),
    ]
    # Round 2: EU 0 holds {0, 1, 2}, EU 1 {5}: cap 2, sample 0 moves to EU 1.
    forward = planner.forward_steps(2)
    assert isinstance(forward[0], MovePhase)
    (move,) = forward[0].transfers
    assert (move.sample, move.src, move.dst, move.round) == (0, 0, 1, 2)
    assert move.slots == (ROUND_INPUT, CRIT_STATE, INJECTION, EXIT_FLAGS) + BATCH
    assert move.frees == (("state", 1), ("crit", 1))
    cores = [tuple(None if i is None else i.sample for i in s.items) for s in forward[1:]]
    assert cores == [(1, 0), (2, 5)]


# ---------------------------------------------------------------------------------------
# P7: the move gate
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("workload", sorted(WORKLOADS))
@pytest.mark.parametrize("num_eus", [2, 4, 8])
@pytest.mark.parametrize("per_eu", [8, 16])
def test_move_gate_on_synthetic_workloads(workload, num_eus, per_eu):
    rng = random.Random(1234)
    moves = actives = 0
    bulk = tail = 0.0
    home = _home(num_eus, per_eu)
    for _ in range(50):
        depths = _random_depths(rng, len(home), probs=WORKLOADS[workload])
        planner = _planner(num_eus, home)
        steps = known_depth_plan(planner, depths)
        stats = planner.stats
        assert stats.sample_moves == len(_transfers(steps, "sample"))
        for m, a in zip(stats.moves_by_round, stats.active_by_round):
            moves += m
            actives += a
            if a >= 2 * num_eus:
                bulk = max(bulk, m / a)
            else:
                tail = max(tail, m / a)
        for r, active in _forward_actives(depths).items():
            cap = -(-len(active) // num_eus)
            subs = [s for s in _subrounds(steps, (SubRoundType.CORE_F,)) if _round_of(s) == r]
            assert sum(i is None for s in subs for i in s.items) == num_eus * cap - len(active)
    aggregate = moves / actives
    print(
        f"{workload} W={num_eus} n={per_eu}: aggregate {aggregate:.2%}, bulk-round max "
        f"{bulk:.1%}, tail-round max {tail:.1%}"
    )
    assert aggregate <= 0.05


# ---------------------------------------------------------------------------------------
# P8: lifetimes
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("check_recompute", [False, True])
def test_lifetimes_against_a_fake_store(check_recompute):
    rng = random.Random(21)
    back_to_source = to_copy_holder = 0
    for _ in range(300):
        num_eus = rng.randint(2, 8)
        per_eu = rng.randint(2, 8)
        home = _home(num_eus, per_eu, mbs=rng.choice([1, 2]) if per_eu % 2 == 0 else 1)
        depths = _random_depths(rng, len(home))
        fill = rng.choice(["drain", "sample_order"])
        planner = _planner(num_eus, home, fill=fill, check_recompute=check_recompute)
        steps = known_depth_plan(planner, depths)
        _replay(steps, num_eus, home, check_recompute=check_recompute)
        sources = {(t.sample, t.src) for t in _transfers(steps, "sample")}
        copies = [{h} for h in home]
        for step in steps:
            if not isinstance(step, MovePhase):
                if isinstance(step, FreeInjection):
                    copies[step.sample].discard(step.eu)
                continue
            for t in step.transfers:
                if t.kind == "item" and (t.sample, t.dst) in sources:
                    back_to_source += 1
                if t.kind == "item" and t.item == "core" and INJECTION not in t.slots:
                    assert t.dst in copies[t.sample]
                    if t.dst != home[t.sample]:
                        to_copy_holder += 1
                if INJECTION in t.slots:
                    copies[t.sample].add(t.dst)
    # Item moves to an EU holding an injection copy occur in the search (and replay cleanly).
    assert to_copy_holder > 0
    del back_to_source  # rare in random plans: constructed below


def test_item_move_back_to_a_former_sample_move_source():
    """Sample 1 moves EU 1 -> EU 0 before round 2 (s_1 is "sent" from EU 1); in the final
    phase's tail its CORE(1, 2) item moves back to the idle EU 1 with s_1, which EU 1 accepts
    (an entry that left by a move may come back; one that was consumed may not)."""
    home = [0, 1, 2, 3, 0, 1, 2, 3]
    depths = [1, 3, 3, 1, 1, 2, 2, 1]
    for check_recompute in (False, True):
        planner = _planner(4, home, fill="sample_order", check_recompute=check_recompute)
        steps = known_depth_plan(planner, depths)
        sample_moves = [(t.sample, t.src, t.dst, t.round) for t in _transfers(steps, "sample")]
        assert (1, 1, 0, 2) in sample_moves
        items = [(t.item, t.sample, t.src, t.dst, t.frees) for t in _transfers(steps, "item")]
        assert ("core", 1, 0, 1, (("state", 1), ("crit", 1))) in items
        _replay(steps, 4, home, check_recompute=check_recompute)


def test_fake_store_catches_a_bad_plan():
    """The replay is a real check: dropping a FreeInjection or a move breaks it."""
    home = [0, 0, 0, 1, 1, 1]
    depths = [4, 3, 4, 1, 1, 4]
    planner = _planner(2, home)
    steps = known_depth_plan(planner, depths)
    _replay(steps, 2, home)
    first_move = next(i for i, s in enumerate(steps) if isinstance(s, MovePhase))
    with pytest.raises(AssertionError):
        _replay(steps[:first_move] + steps[first_move + 1 :], 2, home)
    frees = [i for i, s in enumerate(steps) if isinstance(s, FreeInjection)]
    assert frees
    with pytest.raises(AssertionError):
        _replay(steps[: frees[0]] + steps[frees[0] + 1 :], 2, home)


# ---------------------------------------------------------------------------------------
# P10: the block-input stash moves with CORE items
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("check_recompute", [False, True])
def test_stash_slots_travel_with_core_item_moves(check_recompute):
    """Random depth sets at W = 2 / 4 / 8 with ``core_stash = (2, 4, 6)`` (and ``core_trace =
    8`` under check_recompute): every moved CORE item carries exactly its round's stash slots
    (after the stage-3 slots) and frees them; nothing else carries a per-sublayer slot; the
    fake stores (stash put by the sweep on EU_r(g), consumed by CORE(g, r)) end empty per EU;
    the steps are those of the plan without a stash except for the extra slots."""
    rng = random.Random(10)
    stash, trace = (2, 4, 6), (8 if check_recompute else 0)
    extra_slots = tuple(block_slot(l) for l in stash) + tuple(trace_slot(l) for l in range(trace))
    extra_frees = [f"block:{l}" for l in stash] + [f"trace:{l}" for l in range(trace)]
    moved = 0
    for _ in range(150):
        num_eus = rng.choice([2, 4, 8])
        home = _home(num_eus, rng.randint(2, 6))
        depths = _random_depths(rng, len(home))
        fill = rng.choice(["drain", "sample_order"])
        options = dict(fill=fill, check_recompute=check_recompute)
        planner = _planner(num_eus, home, core_stash=stash, core_trace=trace, **options)
        steps = known_depth_plan(planner, depths)
        _replay(
            steps,
            num_eus,
            home,
            check_recompute=check_recompute,
            core_stash=stash,
            core_trace=trace,
        )
        plain_steps = known_depth_plan(_planner(num_eus, home, **options), depths)
        assert len(plain_steps) == len(steps)
        for step, plain in zip(steps, plain_steps):
            if not isinstance(step, MovePhase):
                assert step == plain
                continue
            assert len(step.transfers) == len(plain.transfers)
            for t, u in zip(step.transfers, plain.transfers):
                if t.kind == "item" and t.item == "core":
                    moved += 1
                    assert t.slots == u.slots + extra_slots, t
                    assert t.frees == u.frees + tuple((kind, t.round) for kind in extra_frees)
                else:
                    assert t == u
    assert moved > 0


def test_stash_layout_enters_the_digest_and_defaults_keep_stage3():
    """The stash layout is part of the digest; the defaults keep the stage-3 header and step
    encodings (``kind{r}``); per-sublayer frees encode as ``kind@r`` so that ``("block:2",
    13)`` and ``("block:21", 3)`` differ."""
    home = _home(2, 2)
    header = (
        "W=2|home=0,1,0,1|R=4|fill=drain|rebalance=1|items=1|crit=0|check=0|"
        "batch=tokens/position_ids/labels/loss_mask|reports=1"
    )
    expected = hashlib.sha256(header.encode() + b"\n").digest()
    assert _planner(2, home).digest() == expected
    assert _planner(2, home, core_stash=(), core_trace=0).digest() == expected
    assert _planner(2, home, core_stash=(2,)).digest() != expected
    assert _planner(2, home, core_trace=4).digest() != expected
    assert (
        _planner(2, home, core_stash=(2,)).digest() != _planner(2, home, core_stash=(3,)).digest()
    )

    def move(frees):
        return MovePhase(
            (Transfer(kind="item", sample=0, src=0, dst=1, round=1, slots=(), frees=frees),)
        )

    assert _encode(move((("state", 13), ("crit", 3)))) == "M|item:0:0:1:1:::state13/crit3"
    assert _encode(move((("block:2", 13),))) != _encode(move((("block:21", 3),)))
    assert _encode(move((("block:2", 13),))).endswith(":block:2@13")
    assert parse_sublayer_slot(block_slot(3)) == ("block_input", 3)
    assert parse_sublayer_slot(trace_slot(0)) == ("trace", 0)
    assert parse_sublayer_slot("block:21") == ("block", 21)
    assert parse_sublayer_slot(ROUND_INPUT) == (ROUND_INPUT, -1)
    for bad in ((0,), (2, 1), (2, 2)):
        with pytest.raises(ValueError, match="core_stash"):
            _planner(2, home, core_stash=bad)
    with pytest.raises(ValueError, match="core_trace"):
        _planner(2, home, core_trace=-1)


# ---------------------------------------------------------------------------------------
# P9: online == known
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("fill", ["drain", "sample_order"])
def test_online_exits_reproduce_the_known_depth_plan(fill):
    """Feeding known depths round by round in the online call order gives the same forward
    steps and the same backward phases as the whole-step plan; only their interleaving
    differs (an online exit's chain runs one slot later). The whole-step plan equals the
    known call order driven incrementally."""
    rng = random.Random(31)
    for num_eus in (1, 2, 4, 8):
        home = _home(num_eus, 4)
        for _ in range(10):
            depths = _random_depths(rng, len(home))
            known = _planner(num_eus, home, fill=fill)
            known_steps = known_depth_plan(known, depths)
            online = _planner(num_eus, home, fill=fill)
            online_steps = _online_plan(online, depths)

            def split(steps):
                forward, backward = [], []
                for step in steps:
                    is_forward = (isinstance(step, SubRound) and step.type in FORWARD_TYPES) or (
                        isinstance(step, MovePhase) and step.transfers[0].kind == "sample"
                    )
                    (forward if is_forward else backward).append(step)
                return forward, backward

            assert split(known_steps) == split(online_steps)
            assert known.stats == online.stats
            again = _planner(num_eus, home, fill=fill)
            incremental = again.prelude_steps()
            r = 1
            while again.active() or again.pending():
                if again.active():
                    incremental += again.forward_steps(r)
                    again.record_exits(r, [g for g in again.active() if depths[g] == r])
                incremental += again.backward_steps(final=False)
                r += 1
            incremental += again.backward_steps(final=True)
            assert incremental == known_steps
            assert again.digest() == known.digest()
            if fill == "sample_order":
                assert online.digest() == known.digest()


# ---------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------


def test_planner_rejects_bad_calls():
    with pytest.raises(ValueError, match="home EUs"):
        _planner(2, [0, 2])
    with pytest.raises(ValueError, match="unknown fill"):
        _planner(2, [0, 1], fill="greedy")
    with pytest.raises(ValueError, match="unknown batch slots"):
        _planner(2, [0, 1], batch_slots=("ctx",))
    planner = _planner(2, [0, 1], max_rounds=2)
    with pytest.raises(ValueError, match="prelude"):
        planner.forward_steps(1)
    planner.prelude_steps()
    with pytest.raises(ValueError, match="out of order"):
        planner.forward_steps(2)
    planner.forward_steps(1)
    with pytest.raises(ValueError, match="not swept"):
        planner.record_exits(1, [5])
    with pytest.raises(ValueError, match="duplicate"):
        planner.record_exits(1, [0, 0])
    planner.record_exits(1, [])
    planner.forward_steps(2)
    planner.record_exits(2, [])
    with pytest.raises(RuntimeError, match="still active after R_max"):
        planner.forward_steps(3)
    with pytest.raises(ValueError, match="outside"):
        known_depth_plan(_planner(2, [0, 1], max_rounds=2), [1, 3])


def test_chain_slots():
    assert chain_slots("coda", 3, criterion_differentiable=True, check_recompute=False) == (
        GRAD_STATE,
        LOSS_RECORD,
    )
    assert chain_slots("core", 1, criterion_differentiable=False, check_recompute=True) == (
        "grad_injection",
        LOSS_RECORD,
    )
    assert chain_slots("core", 2, criterion_differentiable=True, check_recompute=True) == (
        GRAD_STATE,
        "grad_injection",
        "grad_crit",
        LOSS_RECORD,
        "expected_state",
    )
    with pytest.raises(ValueError):
        chain_slots("prelude", 0, criterion_differentiable=False, check_recompute=False)
    assert set(BATCH) <= set(BATCH_SLOTS)
    move = Transfer(kind="chain", sample=0, src=0, dst=1, round=1, slots=())
    assert (
        move.order_key()
        < Transfer(kind="item", sample=0, src=0, dst=1, round=1, slots=()).order_key()
    )
