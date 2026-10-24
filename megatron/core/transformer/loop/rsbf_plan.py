# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The host plan of one multi-EU RSBF wave: placement, moves, typed sub-rounds and lifetimes.

Every rank of a domain (``W`` EUs, one per rank of the domain group) runs ``WavePlanner`` on
identical inputs (the home map of the wave's samples, the exits, the config), so every rank
holds the same plan without communicating it; the executor only executes the plan's steps.
The planner is pure Python over integers (no torch, no process group): orders are sorted
lists, ties break by EU id then by sample id, and nothing depends on dict or set iteration
order or on any RNG.

Samples are wave-global ids ``g = 0 .. G-1`` (the global batch position minus the wave base);
``home[g]`` is the EU that loaded ``g``. A wave is planned as

1. ``prelude_steps()``: PRELUDE-F sub-rounds, each EU sweeping its home samples in ``g`` order.
2. Per forward round ``r``: ``forward_steps(r)``. From round 2 on, the min-move rebalancing moves
   exactly the surplus above ``cap = ceil(A_r / W)`` active samples per EU to EUs below it
   (preferring a destination that already holds the sample's injection copy), as one
   ``MovePhase``; then ``max_e |F_e(r)|`` CORE-F sub-rounds (``cap`` when rebalancing), an EU
   without a sample in a sub-round running a null of the sub-round's type.
3. ``record_exits(r, exited)``: the samples that exited at round ``r`` (known depths: host
   data right after ``forward_steps(r)``; online: the exchanged exit bitmap of round ``r``).
4. ``backward_steps(final=...)``: the backward phase of a slot. Every pending chain
   ``CODA(g), CORE(g, d) .. CORE(g, 1), PRELUDE(g)`` runs where its checkpoint lives (CODA and
   CORE(g, r) on the EU that swept the round, PRELUDE at home) in typed sub-rounds of one item
   (or a null) per EU (bulk-only fill). Below ``2 W`` remaining items the phase is
   chain-parallel: idle EUs receive CORE / CODA items together with their checkpoint (item
   moves). Chain records travel lazily, one hop, from the predecessor's EU to the EU that runs
   the successor.

Block-checkpoint items (``core_stash``): the sweep of round ``r`` stores the round's block
inputs on ``EU_r(g)``, the EU that runs CORE(g, r), so a sample move (which carries the round
input of the *next* round) never carries them; a moved CORE item carries its round's block
inputs (slots ``block_input:<l>``) and, under ``check_recompute``, the sweep's per-sublayer
outputs (``trace:<l>``, ``core_trace``) with it.

``known_depth_plan`` plans a whole wave from known depths (no per-round exchange). ``digest()``
covers the inputs, the exits and every emitted step, so ranks can compare plans in tests.

Moves never leave the domain: every ``src`` / ``dst`` is a domain index in ``0 .. W-1``.
"""

import enum
import hashlib
from dataclasses import dataclass, field
from typing import Literal, Sequence, Union

__all__ = [
    "BATCH_SLOTS",
    "FreeInjection",
    "MovePhase",
    "PlanItem",
    "PlanStats",
    "Step",
    "SubRound",
    "SubRoundType",
    "SUBROUND_TYPE_OF",
    "Transfer",
    "WavePlanner",
    "block_slot",
    "chain_slots",
    "known_depth_plan",
    "parse_sublayer_slot",
    "trace_slot",
]


class SubRoundType(enum.Enum):
    """The type of a lockstep sub-round (every EU runs one item of it, or its null)."""

    PRELUDE_F = "prelude_f"
    CORE_F = "core_f"
    CORE_B = "core_b"
    CODA = "coda"
    PRELUDE_B = "prelude_b"


ItemName = Literal["prelude_sweep", "core_sweep", "coda", "core", "prelude"]

# The sub-round type of each plan item kind (sweeps are forward types, chain items backward).
SUBROUND_TYPE_OF = {
    "prelude_sweep": SubRoundType.PRELUDE_F,
    "core_sweep": SubRoundType.CORE_F,
    "coda": SubRoundType.CODA,
    "core": SubRoundType.CORE_B,
    "prelude": SubRoundType.PRELUDE_B,
}

# Payload slot names (shapes and dtypes: the stage-3 port table). Sample and item moves:
ROUND_INPUT = "round_input"  # the state checkpoint the receiver's next cell call reads
CRIT_STATE = "crit_state"  # the criterion checkpoint of the same round ([c, 0] when k = 0)
INJECTION = "injection"  # e, only when the destination holds no copy
EXIT_FLAGS = "exit_flags"  # (exited, exit_round) int32
# The batch fields, sent on a sample's first arrival on an EU (the subset the batch has).
BATCH_SLOTS = ("tokens", "position_ids", "labels", "loss_mask", "padding_mask", "attention_mask")
# Chain moves (a ChainRecord's tensors):
GRAD_STATE = "grad_state"
GRAD_INJECTION = "grad_injection"
GRAD_CRIT = "grad_crit"
LOSS_RECORD = "loss_record"
EXPECTED_STATE = "expected_state"
# CORE item moves of a block-checkpoint plan: one slot per stashed block input
# (``block_input:<l>``) and, under check_recompute, per sweep trace entry (``trace:<l>``); their
# ``frees`` entries are ``("block:<l>", r)`` / ``("trace:<l>", r)``.
BLOCK_INPUT = "block_input"
TRACE = "trace"
# The ``frees`` kinds of those entries (the state store's entry kinds).
BLOCK_ENTRY = "block"
TRACE_ENTRY = "trace"


def block_slot(sublayer: int) -> str:
    """The payload slot of the block input of core sublayer ``sublayer``."""
    return f"{BLOCK_INPUT}:{sublayer}"


def trace_slot(sublayer: int) -> str:
    """The payload slot of the sweep's output of core sublayer ``sublayer``."""
    return f"{TRACE}:{sublayer}"


def parse_sublayer_slot(name: str) -> tuple:
    """``(base, sublayer)`` of a per-sublayer slot (``"block_input:3"`` -> ``("block_input",
    3)``) or of a ``frees`` kind (``"block:3"`` -> ``("block", 3)``); ``(name, -1)`` for any
    other name."""
    base, sep, index = name.partition(":")
    if not sep:
        return name, -1
    return base, int(index)


@dataclass(frozen=True)
class PlanItem:
    """One real cell call: ``round`` is 0 (prelude sweep / item), ``r`` (sweep / CORE) or the
    depth ``d`` (CODA)."""

    kind: ItemName
    sample: int
    round: int

    @property
    def type(self) -> SubRoundType:
        """The sub-round type this item runs in."""
        return SUBROUND_TYPE_OF[self.kind]


@dataclass(frozen=True)
class SubRound:
    """One lockstep sub-round: ``items[e]`` is EU ``e``'s item, None for a null of ``type``."""

    type: SubRoundType
    items: tuple


@dataclass(frozen=True)
class Transfer:
    """One payload from ``src`` to ``dst`` (domain indices).

    ``kind``: ``"sample"`` (a sample move before forward round ``round``), ``"chain"`` (the
    chain record produced by item ``item`` of round ``round``) or ``"item"`` (backward item
    ``item`` of round ``round`` moved with its checkpoint). ``slots`` are the payload's slot
    names in send order. ``frees``: the store entries ``(kind, round)`` the payload carries;
    the source releases them (``send_*``) after the send completes and the destination
    installs them (``put_received_*``). The injection travels as the ``injection`` slot; the
    source keeps its copy.
    """

    kind: Literal["sample", "chain", "item"]
    sample: int
    src: int
    dst: int
    round: int
    slots: tuple
    frees: tuple = ()
    item: str = ""

    def order_key(self) -> tuple:
        """The per-phase order of transfers (the per-pair op order of the dispatcher)."""
        return (self.sample, self.kind, self.round, self.item, self.src, self.dst)


@dataclass(frozen=True)
class MovePhase:
    """The transfers of one exchange, in their fixed order (``Transfer.order_key``)."""

    transfers: tuple


@dataclass(frozen=True)
class FreeInjection:
    """Free the injection copy of ``sample`` on EU ``eu`` (its last consumer has run)."""

    sample: int
    eu: int


Step = Union[SubRound, MovePhase, FreeInjection]


@dataclass
class PlanStats:
    """What the plan did (per wave): moves and actives per forward round, chain and item
    transfers, sub-rounds and nulls per type."""

    moves_by_round: list = field(default_factory=list)
    active_by_round: list = field(default_factory=list)
    chain_moves: int = 0
    item_moves: int = 0
    subrounds: dict = field(default_factory=lambda: {t: 0 for t in SubRoundType})
    nulls: dict = field(default_factory=lambda: {t: 0 for t in SubRoundType})

    @property
    def sample_moves(self) -> int:
        """Forward sample moves of the wave."""
        return sum(self.moves_by_round)


def chain_slots(
    item: str, r: int, *, criterion_differentiable: bool, check_recompute: bool
) -> tuple:
    """The slots of the chain record produced by ``item`` (``"coda"`` or ``"core"``) of round
    ``r`` (the depth for CODA): a pure function of the item, the criterion and
    ``check_recompute``. The loss record rides every chain transfer until the chain reaches
    home (PRELUDE never moves)."""
    if item == "coda":
        slots = [GRAD_STATE, LOSS_RECORD]
        if check_recompute:
            slots.append(EXPECTED_STATE)
        return tuple(slots)
    if item != "core":
        raise ValueError(f"no chain record leaves a {item!r} item")
    slots = []
    if r > 1:
        slots.append(GRAD_STATE)
    slots.append(GRAD_INJECTION)
    if criterion_differentiable:
        slots.append(GRAD_CRIT)
    slots.append(LOSS_RECORD)
    if check_recompute and r > 1:
        slots.append(EXPECTED_STATE)
    return tuple(slots)


def _encode(step: Step) -> str:
    """Canonical text of one step (the digest's input)."""
    if isinstance(step, SubRound):
        items = ",".join(
            "-" if it is None else f"{it.kind}:{it.sample}:{it.round}" for it in step.items
        )
        return f"S|{step.type.value}|{items}"
    if isinstance(step, MovePhase):
        parts = []
        for t in step.transfers:
            # "@" separates a per-sublayer kind ("block:2") from its round, so that
            # ("block:2", 13) and ("block:21", 3) differ; the stage-3 kinds keep "kind{r}".
            frees = "/".join(f"{kind}@{r}" if ":" in kind else f"{kind}{r}" for kind, r in t.frees)
            parts.append(
                f"{t.kind}:{t.sample}:{t.src}:{t.dst}:{t.round}:{t.item}:"
                f"{'/'.join(t.slots)}:{frees}"
            )
        return "M|" + ";".join(parts)
    if isinstance(step, FreeInjection):
        return f"F|{step.sample}|{step.eu}"
    raise TypeError(f"not a plan step: {step!r}")


class WavePlanner:
    """The host plan of one wave over ``num_eus`` EUs (see the module docstring).

    Args:
        num_eus: W, the domain size.
        home: ``home[g]`` = the EU that loaded wave sample ``g``.
        max_rounds: R_max; a forward round beyond it with active samples raises.
        fill: ``"drain"`` (a chain is planned in the backward phase of the slot after its exit
            is known) or ``"sample_order"`` (every chain after the last forward round, in
            ``g`` order).
        rebalance: min-move rebalancing before rounds ``r >= 2``; False keeps every sample on
            its home EU.
        tail_item_moves: move CORE / CODA items with their checkpoints to idle EUs in
            chain-parallel sub-rounds.
        criterion_differentiable: chain records carry ``grad_crit``.
        check_recompute: chain records carry ``expected_state``.
        batch_slots: the batch fields the wave's microbatches have (a subset of
            ``BATCH_SLOTS``, in that order), sent on a sample's first arrival on an EU.
        report_keys: the number of report keys of the loss record (digest only).
        core_stash: the core-relative sublayers whose input the sweep stashes for the CORE
            item (``CoreBackwardPlan.stashed_block_inputs``; the same for every round); a
            moved CORE item carries them.
        core_trace: ``L`` when the sweep stores its per-sublayer outputs for
            ``check_recompute`` (a moved CORE item carries them), else 0.
    """

    def __init__(
        self,
        *,
        num_eus: int,
        home: Sequence[int],
        max_rounds: int,
        fill: Literal["drain", "sample_order"],
        rebalance: bool,
        tail_item_moves: bool,
        criterion_differentiable: bool,
        check_recompute: bool,
        batch_slots: tuple,
        report_keys: int,
        core_stash: tuple = (),
        core_trace: int = 0,
    ):
        if num_eus < 1:
            raise ValueError(f"num_eus must be >= 1, got {num_eus}")
        if not home:
            raise ValueError("a wave needs at least one sample")
        if any(not 0 <= int(h) < num_eus for h in home):
            raise ValueError(f"home EUs must be in 0 .. {num_eus - 1}, got {list(home)}")
        if max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")
        if fill not in ("drain", "sample_order"):
            raise ValueError(f"unknown fill {fill!r}")
        unknown = [slot for slot in batch_slots if slot not in BATCH_SLOTS]
        if unknown:
            raise ValueError(f"unknown batch slots {unknown}")
        core_stash = tuple(int(l) for l in core_stash)
        if any(l < 1 for l in core_stash) or list(core_stash) != sorted(set(core_stash)):
            raise ValueError(f"core_stash must be ascending sublayers >= 1, got {core_stash}")
        if core_trace < 0:
            raise ValueError(f"core_trace must be >= 0, got {core_trace}")
        self.num_eus = num_eus
        self.home = tuple(int(h) for h in home)
        self.max_rounds = max_rounds
        self.fill = fill
        self.rebalance = rebalance
        self.tail_item_moves = tail_item_moves
        self.criterion_differentiable = criterion_differentiable
        self.check_recompute = check_recompute
        # Declared order, whatever order the caller passed them in.
        self.batch_slots = tuple(slot for slot in BATCH_SLOTS if slot in batch_slots)
        self.report_keys = report_keys
        self.core_stash = core_stash
        self.core_trace = int(core_trace)
        num = len(self.home)
        self._loc = list(self.home)  # EU holding the sample's latest round state
        self._eu_round: list = [[None] for _ in range(num)]  # [g][r] = EU_r(g), r >= 1
        self._exit = [0] * num
        self._active = list(range(num))
        self._pending: list = []  # exited, chain not yet planned, (exit round, g) order
        self._planned = [False] * num
        # EUs holding a live injection copy / ever holding the sample's batch.
        self._e_copies = [{h} for h in self.home]
        self._held = [{h} for h in self.home]
        self._prelude_done = False
        self._round = 0  # last forward round planned
        self._exits_round = 0  # last round whose exits were recorded
        self.steps: list = []
        self.stats = PlanStats()
        self._hash = hashlib.sha256()
        self._update(
            f"W={num_eus}|home={','.join(map(str, self.home))}|R={max_rounds}|fill={fill}|"
            f"rebalance={int(rebalance)}|items={int(tail_item_moves)}|"
            f"crit={int(criterion_differentiable)}|check={int(check_recompute)}|"
            f"batch={'/'.join(self.batch_slots)}|reports={report_keys}"
            # Only non-default stash layouts enter the digest (stage-3 digests unchanged).
            + (
                f"|stash={','.join(map(str, core_stash))}|trace={self.core_trace}"
                if core_stash or self.core_trace
                else ""
            )
        )

    # -----------------------------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------------------------

    @property
    def num_samples(self) -> int:
        """G, the wave's sample count."""
        return len(self.home)

    def active(self) -> list:
        """The active samples (not exited), ``g`` ascending."""
        return list(self._active)

    def pending(self) -> bool:
        """Whether chains wait for a drain backward phase (always False for sample_order)."""
        return self.fill == "drain" and bool(self._pending)

    def eu_of_round(self, g: int, r: int) -> int:
        """``EU_r(g)``: the EU that swept round ``r`` of ``g`` (``r >= 1``)."""
        rounds = self._eu_round[g]
        if not 1 <= r < len(rounds):
            raise ValueError(f"round {r} of sample {g} is not planned")
        return rounds[r]

    def exit_round(self, g: int) -> int:
        """The exit round (depth) of ``g``; 0 while it is active."""
        return self._exit[g]

    def location(self, g: int) -> int:
        """The EU holding ``g``'s latest round state (its next sweep or its CODA runs there)."""
        return self._loc[g]

    def digest(self) -> bytes:
        """sha256 of the inputs, the recorded exits and every step emitted so far."""
        return self._hash.copy().digest()

    # -----------------------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------------------

    def prelude_steps(self) -> list:
        """The PRELUDE-F sub-rounds: sub-round ``j`` runs each EU's ``j``-th home sample."""
        if self._prelude_done:
            raise ValueError("the prelude of this wave is already planned")
        self._prelude_done = True
        per_eu = [[] for _ in range(self.num_eus)]
        for g, h in enumerate(self.home):
            per_eu[h].append(g)
        steps = self._subrounds(SubRoundType.PRELUDE_F, "prelude_sweep", per_eu, 0)
        return self._emit(steps)

    def forward_steps(self, r: int) -> list:
        """Round ``r``: the sample moves (``r >= 2``, one MovePhase) and the CORE-F
        sub-rounds over the active samples."""
        if not self._prelude_done:
            raise ValueError("plan the prelude before the forward rounds")
        if r != self._round + 1 or self._exits_round != r - 1:
            raise ValueError(
                f"forward round {r} out of order (last planned {self._round}, exits recorded "
                f"through round {self._exits_round})"
            )
        if not self._active:
            raise ValueError(f"forward round {r} has no active sample")
        if r > self.max_rounds:
            raise RuntimeError(
                f"samples {self._active} still active after R_max = {self.max_rounds}"
            )
        self._round = r
        steps: list = []
        moves = self._rebalance(r) if (r >= 2 and self.rebalance) else []
        if moves:
            steps.append(MovePhase(tuple(sorted(moves, key=Transfer.order_key))))
        per_eu = [[] for _ in range(self.num_eus)]
        for g in self._active:
            per_eu[self._loc[g]].append(g)
            self._eu_round[g].append(self._loc[g])
        if self.rebalance and r >= 2:  # round 1 runs at home (equal home counts: also cap)
            cap = -(-len(self._active) // self.num_eus)
            longest = max(len(samples) for samples in per_eu)
            if longest != cap:  # pragma: no cover - the rebalancing guarantees it
                raise AssertionError(f"round {r}: lockstep length {longest} != {cap}")
        steps += self._subrounds(SubRoundType.CORE_F, "core_sweep", per_eu, r)
        self.stats.moves_by_round.append(len(moves))
        self.stats.active_by_round.append(len(self._active))
        return self._emit(steps)

    def record_exits(self, r: int, exited: Sequence[int]) -> None:
        """The samples that exited at round ``r`` (each swept in round ``r``)."""
        if r != self._round or self._exits_round != r - 1:
            raise ValueError(
                f"exits of round {r} out of order (last forward round {self._round}, exits "
                f"recorded through round {self._exits_round})"
            )
        newly = sorted(int(g) for g in exited)
        if len(set(newly)) != len(newly):
            raise ValueError(f"round {r}: duplicate exits {newly}")
        active = set(self._active)
        stray = [g for g in newly if g not in active]
        if stray:
            raise ValueError(f"round {r}: samples {stray} exit but were not swept in round {r}")
        for g in newly:
            self._exit[g] = r
        retired = set(newly)
        self._active = [g for g in self._active if g not in retired]
        self._pending.extend(newly)
        self._exits_round = r
        self._update(f"X|{r}|{','.join(map(str, newly))}")

    def _rebalance(self, r: int) -> list:
        """Move exactly the surplus above ``cap = ceil(A / W)`` (the lockstep bound) from the
        EUs above it to EUs below it; candidate samples per surplus EU (EU id order): those
        whose injection copy sits on an EU with room first, then the rest, each by ``g``;
        destination: the lowest-id EU with room holding an injection copy, else the EU with the
        most room (ties lowest id)."""
        W = self.num_eus
        counts = [0] * W
        for g in self._active:
            counts[self._loc[g]] += 1
        cap = -(-len(self._active) // W)
        room = [max(0, cap - c) for c in counts]
        moves = []
        for s in range(W):
            surplus = counts[s] - cap
            if surplus <= 0:
                continue
            mine = [g for g in self._active if self._loc[g] == s]
            preferred = [g for g in mine if any(room[x] > 0 for x in self._e_copies[g])]
            chosen = set(preferred)
            candidates = preferred + [g for g in mine if g not in chosen]
            for g in candidates[:surplus]:
                with_copy = sorted(x for x in self._e_copies[g] if room[x] > 0)
                if with_copy:
                    dst = with_copy[0]
                else:
                    dst = min((x for x in range(W) if room[x] > 0), key=lambda x: (-room[x], x))
                room[dst] -= 1
                moves.append(self._move_sample(g, s, dst, r))
        return moves

    def _move_sample(self, g: int, src: int, dst: int, r: int) -> Transfer:
        slots = [ROUND_INPUT, CRIT_STATE]
        if dst not in self._e_copies[g]:
            slots.append(INJECTION)
            self._e_copies[g].add(dst)
        slots.append(EXIT_FLAGS)
        slots += self._arrive(g, dst)
        self._loc[g] = dst
        return Transfer(
            kind="sample",
            sample=g,
            src=src,
            dst=dst,
            round=r,
            slots=tuple(slots),
            frees=(("state", r - 1), ("crit", r - 1)),
        )

    def _arrive(self, g: int, dst: int) -> list:
        """The batch slots of ``g`` if ``dst`` never held it (then it holds it)."""
        if dst in self._held[g]:
            return []
        self._held[g].add(dst)
        return list(self.batch_slots)

    def _subrounds(self, kind_type: SubRoundType, kind: str, per_eu: list, r: int) -> list:
        steps = []
        length = max(len(samples) for samples in per_eu)
        for j in range(length):
            items = tuple(
                PlanItem(kind, samples[j], r) if j < len(samples) else None for samples in per_eu
            )
            steps.append(SubRound(kind_type, items))
        return steps

    # -----------------------------------------------------------------------------------
    # Backward
    # -----------------------------------------------------------------------------------

    def backward_steps(self, *, final: bool) -> list:
        """The backward phase of a slot: ``drain`` plans the pending chains (``final=False``
        and, for any left, ``final=True``); ``sample_order`` plans every chain at
        ``final=True`` (after the last forward round) and nothing before."""
        if self.fill == "sample_order":
            if not final:
                return []
            if self._active:
                raise ValueError(f"samples {self._active} are still active at the final phase")
            chains = [g for g in range(self.num_samples) if not self._planned[g]]
        else:
            chains, self._pending = self._pending, []
        for g in chains:
            if self._planned[g]:  # pragma: no cover - pending lists each chain once
                raise AssertionError(f"chain of sample {g} planned twice")
            self._planned[g] = True
        if not chains:
            return []
        return self._emit(self._drain(chains))

    def _drain(self, chains: list) -> list:
        W = self.num_eus
        items = []
        place = []
        for g in chains:
            d = self._exit[g]
            chain = [PlanItem("coda", g, d)]
            chain += [PlanItem("core", g, r) for r in range(d, 0, -1)]
            chain.append(PlanItem("prelude", g, 0))
            items.append(chain)
            place.append(
                [self._eu_round[g][d]]
                + [self._eu_round[g][r] for r in range(d, 0, -1)]
                + [self.home[g]]
            )
        nxt = [0] * len(chains)
        record_at: list = [None] * len(chains)  # EU holding the chain's record
        remaining = sum(len(chain) for chain in items)
        steps: list = []
        while remaining:
            ready = [[] for _ in range(W)]  # chain indices, priority order
            for c, chain in enumerate(items):
                if nxt[c] < len(chain):
                    ready[place[c][nxt[c]]].append(c)
            kind_type = self._choose_type(ready, items, nxt)
            transfers = []
            if (
                self.tail_item_moves
                and remaining < 2 * W
                and kind_type in (SubRoundType.CORE_B, SubRoundType.CODA)
            ):
                transfers += self._item_moves(kind_type, ready, items, place, nxt)
            run: list = [None] * W
            for e in range(W):
                for c in ready[e]:
                    if items[c][nxt[c]].type is kind_type:
                        run[e] = c
                        break
            for e, c in enumerate(run):
                if c is None or record_at[c] is None or record_at[c] == e:
                    continue
                producer = items[c][nxt[c] - 1]
                transfers.append(
                    Transfer(
                        kind="chain",
                        sample=producer.sample,
                        src=record_at[c],
                        dst=e,
                        round=producer.round,
                        slots=chain_slots(
                            producer.kind,
                            producer.round,
                            criterion_differentiable=self.criterion_differentiable,
                            check_recompute=self.check_recompute,
                        ),
                        item=producer.kind,
                    )
                )
                self.stats.chain_moves += 1
            if transfers:
                steps.append(MovePhase(tuple(sorted(transfers, key=Transfer.order_key))))
            steps.append(
                SubRound(kind_type, tuple(None if c is None else items[c][nxt[c]] for c in run))
            )
            for e, c in enumerate(run):
                if c is None:
                    continue
                item = items[c][nxt[c]]
                record_at[c] = e
                nxt[c] += 1
                remaining -= 1
                g = item.sample
                if item.kind == "core" and item.round == 1:
                    # No CORE item of g, hence no reader and no further transfer of e_g,
                    # can follow: free every non-home copy.
                    for x in sorted(self._e_copies[g] - {self.home[g]}):
                        steps.append(FreeInjection(g, x))
                    self._e_copies[g] = {self.home[g]}
                elif item.kind == "prelude":
                    self._e_copies[g] = set()  # the PRELUDE item frees the home copy
        return steps

    @staticmethod
    def _choose_type(ready: list, items: list, nxt: list) -> SubRoundType:
        """The type most EUs have a ready item of; among ties, the type of the
        highest-priority ready item."""
        count = {}
        for chains in ready:
            for kind_type in {items[c][nxt[c]].type for c in chains}:
                count[kind_type] = count.get(kind_type, 0) + 1
        best = max(count.values())
        for c, chain in enumerate(items):
            if nxt[c] < len(chain) and count.get(chain[nxt[c]].type) == best:
                return chain[nxt[c]].type
        raise AssertionError("no ready item")  # pragma: no cover

    def _item_moves(
        self, kind_type: SubRoundType, ready: list, items: list, place: list, nxt: list
    ) -> list:
        """Chain-parallel balancing: while an EU has no ready item of ``kind_type`` and another
        has two or more, move the busiest EU's item with the longest remaining chain to the
        lowest idle EU, with its checkpoint. Updates ``ready`` and ``place``."""
        W = self.num_eus

        def of_type(e):
            return [c for c in ready[e] if items[c][nxt[c]].type is kind_type]

        idle = [e for e in range(W) if not of_type(e)]
        moves = []
        while idle:
            loads = [(len(of_type(e)), e) for e in range(W)]
            busiest = max(loads, key=lambda load: (load[0], -load[1]))
            if busiest[0] < 2:
                break
            b = busiest[1]
            c = max(of_type(b), key=lambda c: (items[c][nxt[c]].round, items[c][nxt[c]].sample))
            dst = idle.pop(0)
            moves.append(self._move_item(items[c][nxt[c]], b, dst))
            place[c][nxt[c]] = dst
            ready[b].remove(c)
            ready[dst] = sorted(ready[dst] + [c])
        return moves

    def _move_item(self, item: PlanItem, src: int, dst: int) -> Transfer:
        g, r = item.sample, item.round
        if item.kind == "coda":
            slots = [ROUND_INPUT, CRIT_STATE, EXIT_FLAGS]
            frees = (("state", r), ("crit", r))
        else:
            slots = [ROUND_INPUT] if r > 1 else []
            slots.append(CRIT_STATE)
            if dst not in self._e_copies[g]:
                slots.append(INJECTION)
                self._e_copies[g].add(dst)
            slots.append(EXIT_FLAGS)
            frees = ((("state", r - 1),) if r > 1 else ()) + (("crit", r - 1),)
        slots += self._arrive(g, dst)
        if item.kind == "core":
            # The round's block-input stash (and check_recompute trace) lives with its CORE
            # item: it moves with it and is released on the source.
            slots += [block_slot(l) for l in self.core_stash]
            slots += [trace_slot(l) for l in range(self.core_trace)]
            frees += tuple((f"{BLOCK_ENTRY}:{l}", r) for l in self.core_stash)
            frees += tuple((f"{TRACE_ENTRY}:{l}", r) for l in range(self.core_trace))
        self.stats.item_moves += 1
        return Transfer(
            kind="item",
            sample=g,
            src=src,
            dst=dst,
            round=r,
            slots=tuple(slots),
            frees=frees,
            item=item.kind,
        )

    # -----------------------------------------------------------------------------------
    # Bookkeeping
    # -----------------------------------------------------------------------------------

    def _emit(self, steps: list) -> list:
        for step in steps:
            if isinstance(step, SubRound):
                if len(step.items) != self.num_eus:  # pragma: no cover - by construction
                    raise AssertionError(f"sub-round with {len(step.items)} entries")
                self.stats.subrounds[step.type] += 1
                self.stats.nulls[step.type] += sum(item is None for item in step.items)
            self._update(_encode(step))
        self.steps += steps
        return steps

    def _update(self, text: str) -> None:
        self._hash.update(text.encode())
        self._hash.update(b"\n")


def known_depth_plan(planner: WavePlanner, depths: Sequence[int]) -> list:
    """The whole wave's steps from known depths (``depths[g]`` in ``1 .. R_max``), in the
    order the executor runs them: the prelude, then per slot ``r`` the forward steps of round
    ``r``, the exits of round ``r`` and the slot's backward phase, then the final phase."""
    if len(depths) != planner.num_samples:
        raise ValueError(f"{len(depths)} depths for {planner.num_samples} samples")
    bad = [g for g, d in enumerate(depths) if not 1 <= d <= planner.max_rounds]
    if bad:
        raise ValueError(f"depths of samples {bad} are outside 1 .. {planner.max_rounds}")
    steps = planner.prelude_steps()
    r = 1
    while planner.active() or planner.pending():
        if planner.active():
            steps += planner.forward_steps(r)
            planner.record_exits(r, [g for g in planner.active() if depths[g] == r])
        steps += planner.backward_steps(final=False)
        r += 1
    steps += planner.backward_steps(final=True)
    return steps
