# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""A bounded admission queue spanning one optimizer update.

A home starts its next core when its previous sample exits. All homes share the
physical cell type of a tick, but can execute different samples and rounds.
Only core ticks may need an empty EP source. Edges, MTP and prelude backward
always process real samples. No activation or KV bank leaves its home.
"""

import hashlib
from collections import deque

from .rsbf_plan import PlanItem, PlanStats, SUBROUND_TYPE_OF, SubRound, _encode


class RefillPlanner:
    def __init__(self, home, num_eus, max_rounds, window):
        self.home = tuple(home)
        self.max_rounds = max_rounds
        self.window = window
        self.waiting = [deque() for _ in range(num_eus)]
        for g, eu in enumerate(home):
            self.waiting[eu].append(g)
        if window < 1 or max_rounds < 1 or len({len(q) for q in self.waiting}) != 1:
            raise ValueError("RSBF refill needs equal home sample counts and a positive window and depth")
        self.forward = [deque() for _ in range(num_eus)]
        self.coda = [deque() for _ in range(num_eus)]
        self.backward = [deque() for _ in range(num_eus)]
        self.finish = [deque() for _ in range(num_eus)]
        self.live = 0  # PRELUDE-F and PRELUDE-B are issued for all homes together
        self.peak_live = 0
        self._pending_sweep = None
        self._depths = [0] * len(home)
        self.stats = PlanStats(moves_by_round=[0] * max_rounds, active_by_round=[0] * max_rounds)
        self._hash = hashlib.sha256()
        self.ticks = []

    def digest(self):
        return self._hash.digest()

    def exit_round(self, g):
        return self._depths[g]

    def eu_of_round(self, g, r):
        return self.home[g]

    def _issue(self, kind, visits):
        items = tuple(PlanItem(kind, *visit) if visit is not None else None for visit in visits)
        step = SubRound(SUBROUND_TYPE_OF[kind], items)
        self.stats.subrounds[step.type] += 1
        self.stats.nulls[step.type] += items.count(None)
        self._hash.update((_encode(step) + "\n").encode())
        self.ticks.append(step)
        return step

    def next_tick(self):
        """Issue one typed sub-round. A CORE-F must be followed by observe_exits."""
        if self._pending_sweep is not None:
            raise RuntimeError("observe the previous core sweep before issuing another tick")
        if all(self.finish):
            self.live -= 1
            return self._issue("prelude", [(q.popleft(), 0) for q in self.finish])
        if all(self.backward):
            return self._backward_tick()
        if all(self.coda):
            visits = [q.popleft() for q in self.coda]
            for q, visit in zip(self.backward, visits):
                q.append(visit)
            return self._issue("coda", visits)
        if self.waiting[0] and self.live < self.window and not all(self.forward):
            visits = []
            for pending, active in zip(self.waiting, self.forward):
                g = pending.popleft()
                active.append((g, 1))
                visits.append((g, 0))
            self.live += 1
            self.peak_live = max(self.peak_live, self.live)
            return self._issue("prelude_sweep", visits)
        if any(self.forward):
            visits = [q[0] if q else None for q in self.forward]
            for visit in visits:
                if visit is not None:
                    self.stats.active_by_round[visit[1] - 1] += 1
            self._pending_sweep = visits
            return self._issue("core_sweep", visits)
        if any(self.backward):
            return self._backward_tick()
        if self.live or self.waiting[0]:
            raise RuntimeError("RSBF refill has unfinished work and no executable cell")
        return None

    def _backward_tick(self):
        visits = []
        for q, done in zip(self.backward, self.finish):
            if not q:
                visits.append(None)
                continue
            g, r = q.popleft()
            visits.append((g, r))
            if r == 1:
                done.append(g)
            else:
                q.appendleft((g, r - 1))
        return self._issue("core", visits)

    def observe_exits(self, exited):
        """Consume the results of this tick's real device-side criterion evaluations."""
        visits = self._pending_sweep
        if visits is None:
            raise RuntimeError("no pending RSBF refill sweep")
        exited = set(exited)
        if not exited.issubset({v[0] for v in visits if v is not None}):
            raise ValueError("an exit bit named a sample outside the last sweep")
        for active, retired, visit in zip(self.forward, self.coda, visits):
            if visit is None:
                continue
            g, r = active.popleft()
            if g in exited:
                self._depths[g] = r
                retired.append((g, r))
            else:
                if r >= self.max_rounds:
                    raise RuntimeError(f"sample {g} continued beyond depth {self.max_rounds}")
                active.appendleft((g, r + 1))
        self._pending_sweep = None
