# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Schedule simulator of the RSBF loop planner (planner v1).

A port of the toolkit cost model's event-driven RSBF simulator (``sched.simulate``, restricted to
what the planner uses: no tail context parallelism (``cp_kmax = 1``), one round-input checkpoint
per round (``G_ckpt = 1``), no communication floors) and of its wave / domain / trial driver
(``pp_core_baseline.rsbf_waves``). The cost model is the reference: same formulas, same Python
float operation order, ``random.Random(1000 + trial)`` draws and ``heapq`` ordering, so the
planner reproduces its schedules exactly. Times are in ms per execution unit (EU).

One step of ``n`` samples per EU on ``W`` EUs grouped into lockstep units of ``z`` EUs:

* a prelude sweep (no grad) of every sample (``sweep``), balanced over the EUs;
* forward core rounds ``r = 1 .. R_max``: every sample still alive runs one sweep round (``f``);
  the round barrier (exit status) costs ``sync``;
* after its exit a sample's backward chain becomes ready: the coda item (``coda``), one core item
  per round (``b``, last round first), the prelude item (``pre``), one task per sub-round.

With ``z > 1`` every backward sub-round of a lockstep unit is typed: it runs items of one type
(core, coda or prelude) on every EU of the unit, nulls elsewhere (the EP dispatch / combine
sequence must match on every rank); ``active_scale`` shortens a sub-round whose EUs are only
partly real (the routed-expert GEMMs scale with the active senders). Policies: ``fill`` (idle
units take backward sub-rounds until the round ends, then everything drains) and
``bulk_only`` (also one mandatory backward sub-round after the forward sub-rounds while
``ceil(A_r / W) >= 2``; the executor's policy at ``z > 1``).

Memory is tracked in units of one round state per sample (``peak_units_per_eu``: input / output
per sample, the round-input checkpoints, the exit state until the coda, the chain record from the
coda to the prelude backward) and, with ``sr_track = 0``, in stashed sample-rounds
(``peak_sr_per_eu``: every round's block inputs from its sweep until its core item ends).
"""

import heapq
import math
from typing import Callable, Mapping, Optional, Sequence

__all__ = ["choose_policy", "draw_depths", "run_step_schedule", "simulate", "split_domains"]

POLICIES = ("fill", "bulk_only")


def draw_depths(dist, count: int, seed: int) -> list:
    """``count`` depths of ``dist`` (a ``planner.DepthDistribution``) from ``Random(seed)``,
    bitwise the cost model's ``depths_for``."""
    return dist.draw(count, seed)


def split_domains(
    fn: Callable[[Sequence[int], int], dict],
    depths: Sequence[int],
    W: int,
    domain_W: int,
    track_sr: bool,
) -> dict:
    """The ``W`` EUs as ``W // domain_W`` independent domains (``sched.split_domains``): domain
    ``k`` takes the contiguous slice ``k`` of the draw and runs on its own EUs. Makespan and
    peaks per EU are the max over domains, work is summed."""
    if W % domain_W:
        raise ValueError(f"{W} EUs do not split into domains of {domain_W}")
    domains = W // domain_W
    n = len(depths)
    if n % domains:
        raise ValueError(f"{n} samples do not split over {domains} domains")
    per = n // domains
    parts = [fn(depths[k * per : (k + 1) * per], domain_W) for k in range(domains)]
    makespan = max(p["makespan"] for p in parts)
    work = sum(p["work"] for p in parts)
    out = dict(
        makespan=makespan,
        work=work,
        eff=work / (W * makespan),
        peak_units_per_eu=max(p["peak_units_per_eu"] for p in parts),
    )
    if track_sr:
        out["peak_sr_per_eu"] = max(p["peak_sr_per_eu"] for p in parts)
    return out


def _peak(events: list) -> float:
    events.sort(key=lambda x: (x[0], x[1]))  # frees before allocations at equal times
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def simulate(
    depths: Sequence[int],
    W: int,
    *,
    z: int,
    policy: str,
    d: Mapping[str, float],
    sync: float,
    active_scale: Optional[Mapping[str, float]] = None,
    sr_track: Optional[int] = None,
    domain_W: Optional[int] = None,
) -> dict:
    """Simulate one RSBF pass of ``depths`` (one per sample) on ``W`` EUs in lockstep units of
    ``z`` EUs, with task durations ``d`` (``f``, ``b``, ``coda``, ``pre``, ``sweep``, ms).

    Returns ``makespan``, ``work``, ``eff`` and ``peak_units_per_eu`` (plus ``peak_sr_per_eu``
    when ``sr_track`` is not None: 0 tracks every round's stash until its core item ends,
    ``k >= 1`` a ring of the last ``k`` rounds per sample). ``domain_W`` < ``W``: every domain
    of ``domain_W`` EUs is simulated on its own (``split_domains``)."""
    if policy not in POLICIES:
        raise ValueError(f"unknown schedule policy {policy!r}, expected one of {POLICIES}")
    if domain_W is not None and W > domain_W:
        return split_domains(
            lambda ds, Wd: simulate(
                ds,
                Wd,
                z=z,
                policy=policy,
                d=d,
                sync=sync,
                active_scale=active_scale,
                sr_track=sr_track,
            ),
            depths,
            W,
            domain_W,
            sr_track is not None,
        )
    typed = z > 1
    asc = active_scale or {}
    srk = sr_track
    sr = []  # (time, +-1) stash events (sr_track only)

    def sr_fwd(r, t):
        if srk is None:
            return
        sr.append((t, +1))
        if srk and r > srk:
            sr.append((t, -1))

    def sr_item(i, r, t):
        if srk is None:
            return
        if not srk or r > depths[i] - srk:
            sr.append((t, -1))

    def scaled(D, kind, a):
        s_ = asc.get(kind, 0.0)
        return D * (1 - s_ * (1 - max(a, 0.05))) if s_ else D

    n = len(depths)
    if W % z:
        raise ValueError(f"{W} EUs do not form lockstep units of {z}")
    U = W // z
    R = max(depths)
    t0 = math.ceil(n / W) * d["sweep"]
    free = [t0] * U
    mem = [(0.0, n)]  # input / output state of every sample
    pos = [0] * n  # 0 coda, 1..d core (round d - pos + 1), d + 1 prelude, d + 2 done
    rem = [d["coda"] + depths[i] * d["b"] + d["pre"] for i in range(n)]
    future = []  # (ready_time, -rem, i)
    ready = []  # (-rem, i)
    done_chains = [0]

    def ckpt_units(r):
        return 1 if r >= 2 else 0

    def promote(t):
        while future and future[0][0] <= t + 1e-9:
            _, pr, i = heapq.heappop(future)
            heapq.heappush(ready, (pr, i))

    def task_of(i):
        p = pos[i]
        if p == 0:
            return "coda", d["coda"]
        if p <= depths[i]:
            return "core", d["b"]
        return "pre", d["pre"]

    def finish_task(i, kind, end):
        if kind == "coda":
            mem.append((end, -1))  # exit state consumed
            mem.append((end, +3))  # chain record
        elif kind == "core":
            r = depths[i] - pos[i] + 1
            c = ckpt_units(r)
            if c:
                mem.append((end, -c))
            sr_item(i, r, end)
        else:
            mem.append((end, -3))
            mem.append((end, -1))
        pos[i] += 1
        rem[i] -= {"coda": d["coda"], "core": d["b"], "pre": d["pre"]}[kind]
        if pos[i] <= depths[i] + 1:
            heapq.heappush(future, (end, -rem[i], i))
        else:
            done_chains[0] += 1

    def typed_subround(t):
        """One typed sub-round: items of the type of the highest-priority ready item (the
        longest remaining chain), at most one per EU of the unit."""
        by_kind = {"core": [], "coda": [], "pre": []}
        for pr, i in ready:
            by_kind[task_of(i)[0]].append((pr, i))
        kind = task_of(min(ready)[1])[0]
        cand = sorted(by_kind[kind])[:z]
        chosen = set(i for _, i in cand)
        rest = [(pr, i) for pr, i in ready if i not in chosen]
        ready[:] = rest
        heapq.heapify(ready)
        k = 1
        if kind == "core":
            D = scaled(d["b"] / (k * 1.0), "b", len(cand) * k / z)
        else:
            D = scaled(d[kind], kind, len(cand) / z)
        end = t + D
        for _, i in cand:
            finish_task(i, kind, end)
        return end

    def b_subround(t):
        """One backward sub-round of a unit starting at ``t``: the end time, None if nothing is
        ready."""
        promote(t)
        if not ready:
            return None
        if z == 1:
            _, i = heapq.heappop(ready)
            kind, dur = task_of(i)
            end = t + dur
            finish_task(i, kind, end)
            return end
        return typed_subround(t)

    End = t0
    for r in range(1, R + 1):
        act = [i for i in range(n) if depths[i] >= r]
        A = len(act)
        P = End + (sync if r > 1 else 0.0)
        k = 1
        base = d["f"] / (k * 1.0)
        dur = base
        unit_had_f = set()
        idx = 0
        ends = []
        per_sub = z // k
        heap = [(max(P, free[u]), u) for u in range(U)]
        heapq.heapify(heap)
        while idx < A:
            avail, u = heapq.heappop(heap)
            take = min(per_sub, A - idx)
            fd = dur if u not in unit_had_f else base
            if typed:
                fd = scaled(fd, "f", take * k / z)
            end = avail + fd
            for _ in range(take):
                i = act[idx]
                idx += 1
                c = ckpt_units(r)
                if c:
                    mem.append((avail, +c))
                sr_fwd(r, avail)
                if depths[i] == r:
                    mem.append((end, +1))
            free[u] = end
            unit_had_f.add(u)
            ends.append(end)
            heapq.heappush(heap, (end, u))
        End = max(ends)
        # This round's exits: their chains become ready after the status sync.
        for i in act:
            if depths[i] == r:
                heapq.heappush(future, (End + sync, -rem[i], i))
        bulk = A >= W and k == 1
        if policy == "bulk_only" and math.ceil(A / W) >= 2:
            for u in sorted(unit_had_f, key=lambda u: free[u]):
                e = b_subround(free[u])
                if e is not None:
                    free[u] = e
        # Idle units take backward sub-rounds until the round ends.
        heap = [(free[u], u) for u in range(U) if free[u] < End - 1e-9]
        heapq.heapify(heap)
        est = d["b"]
        while heap:
            t, u = heapq.heappop(heap)
            if t >= End - 1e-9:
                continue
            promote(t)
            if not ready:
                if future and future[0][0] < End - 1e-9:
                    nt = future[0][0]
                    free[u] = max(t, nt)
                    heapq.heappush(heap, (free[u], u))
                continue
            if bulk:
                nxt = task_of(ready[0][1])[1]
                if t + (nxt if z == 1 else est) > End + 1e-9:
                    continue
            e = b_subround(t)
            if e is None:
                continue
            free[u] = e
            if e < End - 1e-9:
                heapq.heappush(heap, (e, u))
    # Drain.
    heap = [(free[u], u) for u in range(U)]
    heapq.heapify(heap)
    while done_chains[0] < n:
        t, u = heapq.heappop(heap)
        if free[u] > t + 1e-9:
            heapq.heappush(heap, (free[u], u))
            continue
        promote(t)
        if not ready:
            if not future:
                break
            nt = future[0][0]
            free[u] = max(t, nt)
            heapq.heappush(heap, (free[u], u))
            continue
        e = b_subround(t)
        free[u] = e
        heapq.heappush(heap, (e, u))
    makespan = max(free)
    work = sum(d["sweep"] + depths[i] * (d["f"] + d["b"]) + d["coda"] + d["pre"] for i in range(n))
    out = dict(
        makespan=makespan, work=work, eff=work / (W * makespan), peak_units_per_eu=_peak(mem) / W
    )
    if srk is not None:
        out["peak_sr_per_eu"] = _peak(sr) / W
    return out


def run_step_schedule(
    depth,
    n: float,
    W: int,
    z: int,
    d: Mapping[str, float],
    *,
    waves: int,
    trials: int,
    policy: str,
    active_scale: Optional[Mapping[str, float]],
    sr_track: Optional[int],
    domain_eus: Optional[int],
    peak_rule: str = "max",
    sync: float = 0.3,
    wsim_max: Optional[int] = 1024,
    cache: Optional[dict] = None,
) -> dict:
    """One training step of ``n`` samples per EU on ``W`` EUs (``pp_core_baseline.rsbf_waves``).

    Per trial ``s`` the step's draw ``depth.draw(nsamp, 1000 + s)`` is cut into ``waves`` equal
    consecutive groups per domain (``domain_eus`` EUs, None = one domain), each a full RSBF pass
    with its own prelude sweep, rounds and drain: a domain's makespan is the sum over its waves,
    the step's the max over domains. Over trials the makespan is the mean, the memory peaks
    (``peak``, ``peak_sr``) the max (``peak_rule = "max"``) or the mean (``"avg"``). With
    ``W > wsim_max`` only ``Wsim`` EUs are simulated (the reference's ``Wsim_max``).

    Returns ``makespan``, ``peak``, ``peak_sr`` (None without ``sr_track``), ``depths`` (the
    draw of every trial), ``Wsim`` and ``nsamp``. ``cache`` (a dict) memoizes by inputs."""
    if peak_rule not in ("max", "avg"):
        raise ValueError(f"unknown peak rule {peak_rule!r}")
    key = None
    if cache is not None:
        key = (
            depth,
            n,
            W,
            z,
            tuple(sorted(d.items())),
            waves,
            trials,
            policy,
            None if active_scale is None else tuple(sorted(active_scale.items())),
            sr_track,
            domain_eus,
            peak_rule,
            sync,
            wsim_max,
        )
        if key in cache:
            return cache[key]
    Wsim = W if wsim_max is None else min(W, max(z, (wsim_max // z) * z))
    if domain_eus is not None and domain_eus < W and Wsim != W:
        raise ValueError(f"per-domain simulation needs every EU simulated ({W} EUs, {Wsim})")
    if domain_eus is not None and domain_eus >= Wsim:
        domain_eus = None
    nsamp = int(round(n * Wsim))
    nd = 1 if domain_eus is None else Wsim // domain_eus
    Wd = Wsim if domain_eus is None else domain_eus
    if nsamp % (waves * nd):
        raise ValueError(f"{nsamp} samples do not split into {waves} waves x {nd} domains")
    outs = []
    draws = []
    for s in range(trials):
        ds = depth.draw(nsamp, 1000 + s)
        draws.append(ds)
        per = nsamp // nd
        k = per // waves
        doms = []
        for q in range(nd):
            dom = ds[q * per : (q + 1) * per]
            doms.append(
                [
                    simulate(
                        dom[i * k : (i + 1) * k],
                        Wd,
                        z=z,
                        policy=policy if z > 1 else "fill",
                        d=d,
                        sync=sync,
                        active_scale=active_scale,
                        sr_track=sr_track,
                    )
                    for i in range(waves)
                ]
            )
        makespans = [sum(p_["makespan"] for p_ in parts) for parts in doms]
        parts = [p_ for dd in doms for p_ in dd]
        o = dict(
            makespan=max(makespans), peak_units_per_eu=max(p_["peak_units_per_eu"] for p_ in parts)
        )
        if sr_track is not None:
            o["peak_sr_per_eu"] = max(p_["peak_sr_per_eu"] for p_ in parts)
        outs.append(o)

    def agg(name):
        if peak_rule == "max":
            return max(o[name] for o in outs)
        return sum(o[name] for o in outs) / len(outs)

    res = dict(
        makespan=sum(o["makespan"] for o in outs) / len(outs),
        peak=agg("peak_units_per_eu"),
        peak_sr=agg("peak_sr_per_eu") if sr_track is not None else None,
        depths=draws,
        Wsim=Wsim,
        nsamp=nsamp,
    )
    if cache is not None:
        cache[key] = res
    return res


def choose_policy(
    depth,
    n: float,
    W: int,
    z: int,
    d0: Mapping[str, float],
    *,
    waves: int,
    trials: int,
    active_scale: Optional[Mapping[str, float]],
    domain_eus: Optional[int],
    peak_rule: str = "max",
    sync: float = 0.3,
    wsim_max: Optional[int] = 1024,
    cache: Optional[dict] = None,
) -> tuple:
    """The cost model's per-form schedule policy (``dsv4.evaluate_moves``): on the nominal
    (uncharged) costs ``d0``, ``bulk_only`` or ``fill`` when ``z > 1``, whichever simulates the
    shorter makespan (``bulk_only`` on a tie); ``fill`` when ``z = 1``. Returns ``(policy,
    result)``."""
    best = None
    for pol in ("bulk_only", "fill") if z > 1 else ("fill",):
        r0 = run_step_schedule(
            depth,
            n,
            W,
            z,
            d0,
            waves=waves,
            trials=trials,
            policy=pol,
            active_scale=active_scale,
            sr_track=None,
            domain_eus=domain_eus,
            peak_rule=peak_rule,
            sync=sync,
            wsim_max=wsim_max,
            cache=cache,
        )
        if best is None or r0["makespan"] < best[1]["makespan"]:
            best = (pol, r0)
    return best
