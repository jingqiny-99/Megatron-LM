# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Planner v1 of the RSBF loop (``LoopRSBFPlanner``).

From the model shape, the phase geometry (``S``, ``cp``, GPUs, EP, samples per EU ``n``,
``R_max``), a distribution of exit depths and the hardware (HBM cap, HybridEP reserve, Grace
capacity and its planning margin, C2C), the planner enumerates the executor's options

* item form: ``nested``, or the block checkpoint with stride ``s`` layers (``s = L_core`` is the
  whole round), weights resident (``w``) or re-gathered (``r``, Layer window only), with or
  without the D42 coda up-projection recompute (``+rc``);
* waves ``w`` dividing ``n`` and the microbatches per step;
* the planner moves in order (``-``, ``A`` Adam to Grace, ``AE`` + prelude / coda / MTP dense
  groups to the Layer window, ``AEM`` + embedding and head Resident with sharded grads); the
  round-state data (state store and block-input stash) always goes to the free HBM first, then
  Grace (the partial "states to Grace" spill);
* the dense layout: ``resident`` (every dense group Resident, what runs today) or
  ``layer_window`` (the core dense group in the Layer window, M3: the cost model's world);

evaluates each with a memory model and a time model ported from the toolkit cost model (the
schedule simulator of ``planner_sched``, the terms of ``planner_costs``), and keeps the fitting
option with the highest modelled MFU (``round(mfu, 5)``, one more digit per decade the menu's
best MFU lies below 0.1; ties to the smaller HBM + Grace, then fewer waves). It makes three
picks: the **target** (every option, M3 capabilities assumed), the **raw** pick over the
options this run can apply, and the **applied** pick, the raw menu filtered by the form policy
(D28: the whole round on small cores). ``apply_plan_to_config`` sets the item form, stride,
waves and D42 on the model config; Adam to Grace is the trainer's
(``LoopRSBFPlan.optimizer_offload``); the other moves are reported as requested but unavailable
until M3.

Exit depths: the config's distribution (``sampled`` / ``fixed``), the plan's R_max = 4 settings
(``R4_GEOM``, ``R4_UNIF``, ``R4_SKEW``) or the **measured** histogram that training writes per
step (``--loop-exit-histogram-path``, one JSON object per line, format ``EXIT_HISTOGRAM_FORMAT``;
``DepthDistribution.from_histogram_file``). Draws use the cost model's samplers op for op
(``random.Random(seed)``, one ``random()`` per sample through the CDF for a pmf, the
truncated-geometric loop for ``geom``), so a measured pmf runs through the same code path as the
R4 settings.
"""

import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from megatron.core.transformer.loop import planner_costs as costs
from megatron.core.transformer.loop.planner_costs import (
    GB,
    ModelShape,
    PlannerUnsupported,
    SeqPhase,
)
from megatron.core.transformer.loop.planner_sched import choose_policy, run_step_schedule
from megatron.core.transformer.loop.program import validate_depth_probs

logger = logging.getLogger(__name__)

__all__ = [
    "EXIT_HISTOGRAM_FORMAT",
    "DepthDistribution",
    "LoopRSBFPlan",
    "LoopRSBFPlanner",
    "OptionResult",
    "PhaseGeometry",
    "PlannerCapabilities",
    "PlannerHardware",
    "PlannerNoFit",
    "PlannerOption",
    "PlannerUnsupported",
    "R4_GEOM",
    "R4_SKEW",
    "R4_UNIF",
    "apply_plan_to_config",
    "executor_nested_rebuild_fraction",
    "format_decision_table",
    "log_decision_table",
    "option_item_form",
]

# The "format" field of every line of an exit-histogram file (megatron/training/
# loop_rsbf_planning.py writes it; the toolkit's cost-model reader checks it too).
EXIT_HISTOGRAM_FORMAT = "megatron-loop-exit-histogram/1"


def _draw_geom(p: float, max_rounds: int, rng: random.Random) -> int:
    depth = 1
    while depth < max_rounds and rng.random() > p:
        depth += 1
    return depth


def _draw_pmf(pmf: Sequence[float], rng: random.Random) -> int:
    u = rng.random()
    cumulative = 0.0
    for depth, probability in enumerate(pmf, 1):
        cumulative += probability
        if u < cumulative:
            return depth
    return len(pmf)


@dataclass(frozen=True)
class DepthDistribution:
    """A distribution of realized exit depths ``1..max_rounds``.

    ``kind == "pmf"``: ``params`` are ``P(d = 1..max_rounds)``; ``kind == "geom"``: ``params``
    is ``(p,)``, the geometric distribution truncated at ``max_rounds`` (``P(d = k) = p (1 -
    p)^(k - 1)`` below ``max_rounds``, the rest of the mass at ``max_rounds``). ``name`` labels
    the planner's decision table.
    """

    kind: Literal["pmf", "geom"]
    params: tuple
    max_rounds: int
    name: str

    @staticmethod
    def pmf(probs: Sequence[float], name: str = "pmf") -> "DepthDistribution":
        """``P(d = k)`` for ``k = 1..len(probs)`` (nonnegative, summing to 1 within 1e-6)."""
        probs = tuple(float(p) for p in probs)
        validate_depth_probs(probs, len(probs))
        return DepthDistribution("pmf", probs, len(probs), name)

    @staticmethod
    def geom(p: float, max_rounds: int, name: str = "geom") -> "DepthDistribution":
        """The geometric distribution of stop probability ``p`` truncated at ``max_rounds``."""
        if not 0.0 < p <= 1.0 or max_rounds < 1:
            raise ValueError(
                f"a truncated geometric depth needs 0 < p <= 1 and R_max >= 1, got p = {p}, "
                f"R_max = {max_rounds}"
            )
        return DepthDistribution("geom", (float(p),), int(max_rounds), name)

    @staticmethod
    def from_config(config: Any) -> "DepthDistribution":
        """The depth distribution a known-depth exit criterion realizes: ``sampled``:
        ``loop_depth_probs`` with the mass below ``loop_min_rounds`` moved to
        ``loop_min_rounds`` (the criterion clamps its draws); ``fixed``: every sample at
        ``loop_iterations``. An online criterion has no config distribution (``ValueError``:
        use a measured histogram)."""
        max_rounds = config.loop_iterations
        kind = getattr(config, "loop_exit_criterion", "fixed")
        if kind == "fixed":
            probs = [0.0] * (max_rounds - 1) + [1.0]
            return DepthDistribution.pmf(probs, "config:fixed")
        if kind != "sampled":
            raise ValueError(
                f"loop_exit_criterion={kind!r} has no depth distribution in the config; plan on "
                "a measured exit histogram (DepthDistribution.from_histogram_file)"
            )
        probs = [float(p) for p in config.loop_depth_probs]
        validate_depth_probs(probs, max_rounds)
        floor = getattr(config, "loop_min_rounds", 1)
        if floor > 1:
            folded = [0.0] * max_rounds
            for depth, p in enumerate(probs, 1):
                folded[max(depth, floor) - 1] += p
            probs = folded
        name = "config:sampled(" + ",".join(f"{p:g}" for p in probs) + ")"
        return DepthDistribution.pmf(probs, name)

    @staticmethod
    def from_histogram_file(
        path: str,
        *,
        first_iteration: Optional[int] = None,
        last_iteration: Optional[int] = None,
        max_rounds: Optional[int] = None,
    ) -> "DepthDistribution":
        """The measured pmf of an exit-histogram file: the per-step ``counts`` summed over the
        lines whose ``iteration`` lies in ``[first_iteration, last_iteration]`` (None: open),
        divided by their total. A resumed run records the iterations after its checkpoint
        again; the last line of an iteration wins. Refuses another format, mixed ``R_max``,
        an ``R_max`` other than ``max_rounds`` (when given) and an empty selection."""
        steps = {}
        rounds = set()
        with open(path, encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("format") != EXIT_HISTOGRAM_FORMAT:
                    raise ValueError(
                        f"{path}:{number}: not an exit-histogram line (format "
                        f"{record.get('format')!r}, expected {EXIT_HISTOGRAM_FORMAT!r})"
                    )
                iteration = int(record["iteration"])
                if first_iteration is not None and iteration < first_iteration:
                    continue
                if last_iteration is not None and iteration > last_iteration:
                    continue
                counts = [int(c) for c in record["counts"]]
                if len(counts) != int(record["R_max"]) or any(c < 0 for c in counts):
                    raise ValueError(f"{path}:{number}: counts {counts} do not fit R_max")
                rounds.add(len(counts))
                steps[iteration] = counts
        if len(rounds) > 1:
            raise ValueError(f"{path}: mixed R_max {sorted(rounds)} in the selected iterations")
        if not steps:
            raise ValueError(
                f"{path}: no exit-histogram line in iterations "
                f"[{first_iteration}, {last_iteration}]"
            )
        (found,) = rounds
        if max_rounds is not None and found != max_rounds:
            raise ValueError(f"{path}: R_max {found} in the file, the run has R_max {max_rounds}")
        totals = [sum(column) for column in zip(*steps.values())]
        samples = sum(totals)
        if samples == 0:
            raise ValueError(f"{path}: the selected iterations hold no sample")
        first, last = min(steps), max(steps)
        name = f"histogram:{os.path.basename(path)}[{first}..{last}]"
        return DepthDistribution("pmf", tuple(t / samples for t in totals), found, name)

    def probabilities(self) -> tuple:
        """``P(d = k)`` for ``k = 1..max_rounds``."""
        if self.kind == "pmf":
            return self.params
        (p,) = self.params
        head = tuple(p * (1.0 - p) ** (k - 1) for k in range(1, self.max_rounds))
        return head + ((1.0 - p) ** (self.max_rounds - 1),)

    def mean(self) -> float:
        """``E[d]``."""
        return math.fsum(k * p for k, p in enumerate(self.probabilities(), 1))

    def draw(self, count: int, seed: int) -> list:
        """``count`` depths from ``random.Random(seed)``, bitwise the cost model's
        ``depths_for`` (one ``random()`` per sample through the CDF for a pmf, the truncated
        geometric loop for ``geom``)."""
        rng = random.Random(seed)
        if self.kind == "pmf":
            return [_draw_pmf(self.params, rng) for _ in range(count)]
        (p,) = self.params
        return [_draw_geom(p, self.max_rounds, rng) for _ in range(count)]


# The R_max = 4 settings of the DSv4 plan (DP §1.2; the cost model's DEPTHS entries).
R4_GEOM = DepthDistribution.geom(0.456, 4, "r4_geom")
R4_UNIF = DepthDistribution.pmf((0.25, 0.25, 0.25, 0.25), "r4_unif")
R4_SKEW = DepthDistribution.pmf((0.105, 0.118, 0.249, 0.528), "r4_skew")


# ----------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class PlannerHardware:
    """Per-GPU hardware and model-calibration constants (bytes, B/s, FLOP/s, ms).

    ``gb200``: the cost model's ``dsv4.HW`` with the ``doc_s3`` profile (GB200 NVL72, MXFP8
    calibrations, Grace 200 GB unverified). ``h100`` (est., ranking only): H100 80 GB, no Grace
    tier (``grace_bytes = 0``: no option may use Grace), every fraction and host time as gb200.
    The HBM cap is ``hbm_bytes - planner_reserve_bytes``; the Grace margin applies only with a
    Grace tier (``grace_bytes > 0``)."""

    name: str
    peak_flops: float
    hbm_bytes: float
    planner_reserve_bytes: float
    grace_bytes: float
    grace_margin_bytes: float
    ep_reserve_bytes: float
    misc_bytes: float
    nvl_bytes_per_s: float
    ib_bytes_per_s: float
    c2c_bytes_per_s: float
    c2c_margin: float
    hbm_kernel_bw: float
    hbm_opt_bw: float
    lpddr_peak_per_gpu: float
    grace_opt_eff: float
    kappa: float
    kappa_ep: float
    two_chunk_exposed: float
    sync_ms: float
    mfu_gemm: float
    mfu_moe: float
    mfu_small: float
    mfu_sattn: float
    mfu_idx: float
    mxfp8_speedup: float
    mxfp8_quant: float
    attn_bwd_factor: float
    host_full_ms: float
    host_partial_fwd_ms: float
    host_partial_bwd_ms: float
    host_eager_fwd_ms: float
    host_eager_bwd_ms: float
    host_item_ms: float
    act_frag: float
    edp_ag_factor: float = 1.0
    dense_rep_factor: float = 1.0
    mfu_fwd_only_x1: bool = True

    @property
    def hbm_cap_bytes(self) -> float:
        """The planned HBM cap: usable HBM minus the planner reserve."""
        return self.hbm_bytes - self.planner_reserve_bytes

    @staticmethod
    def preset(name: Literal["gb200", "h100"], **overrides) -> "PlannerHardware":
        """A named preset with ``overrides`` (field names)."""
        gb200 = dict(
            name="gb200",
            peak_flops=2.5e15,
            hbm_bytes=180 * GB,
            planner_reserve_bytes=5 * GB,
            grace_bytes=200 * GB,
            grace_margin_bytes=10 * GB,
            ep_reserve_bytes=20 * GB,
            misc_bytes=8 * GB,
            nvl_bytes_per_s=720e9,
            ib_bytes_per_s=45e9,
            c2c_bytes_per_s=100e9,
            c2c_margin=1.25,
            hbm_kernel_bw=6.4e12,
            hbm_opt_bw=4.0e12,
            lpddr_peak_per_gpu=250e9,
            grace_opt_eff=0.65,
            kappa=0.15,
            kappa_ep=0.25,
            two_chunk_exposed=0.5,
            sync_ms=0.3,
            mfu_gemm=0.50,
            mfu_moe=0.40,
            mfu_small=0.45,
            mfu_sattn=0.35,
            mfu_idx=0.33,
            mxfp8_speedup=1.6,
            mxfp8_quant=0.10,
            attn_bwd_factor=2.0,
            host_full_ms=0.35,
            host_partial_fwd_ms=9.0,
            host_partial_bwd_ms=13.6,
            host_eager_fwd_ms=13.5,
            host_eager_bwd_ms=20.3,
            host_item_ms=0.5,
            act_frag=1.10,
        )
        if name == "gb200":
            fields_ = gb200
        elif name == "h100":
            # est.: H100 SXM 80 GB; host memory as the optimizer tier only when the caller names
            # it (grace_bytes > 0 with the PCIe host rate as c2c).
            fields_ = dict(
                gb200,
                name="h100",
                peak_flops=989e12,
                hbm_bytes=80 * GB,
                grace_bytes=0.0,
                ep_reserve_bytes=8 * GB,
                nvl_bytes_per_s=450e9,
                ib_bytes_per_s=50e9,
                c2c_bytes_per_s=50e9,
                hbm_kernel_bw=2.68e12,
                hbm_opt_bw=2.0e12,
                lpddr_peak_per_gpu=50e9,
            )
        else:
            raise ValueError(f"unknown planner hardware preset {name!r} (gb200, h100)")
        fields_ = dict(fields_)
        fields_.update(overrides)
        return PlannerHardware(**fields_)


@dataclass(frozen=True)
class PhaseGeometry:
    """One training phase: ``S`` tokens per sample over ``cp`` ranks, ``gpus`` GPUs, EP,
    ``samples_per_eu`` (``n = GBS cp / N``), ``R_max``.

    ``domain_gpus``: GPUs per independent domain (None: one domain of every GPU, Megatron; 64 in
    the cost model); ``lockstep_eus``: EUs in lockstep (``z``; None: ``EP / cp``, the cost model;
    the trainer passes ``W``: the executor's lockstep unit is the whole domain);
    ``microbatches_per_step``: per EU, waves must divide it (None: ``n``);
    ``waves_allowed``: False under a batch-size schedule (waves stay 1)."""

    seq_length: int
    gpus: int
    ep: int
    samples_per_eu: int
    max_rounds: int
    cp: int = 1
    domain_gpus: Optional[int] = None
    lockstep_eus: Optional[int] = None
    microbatches_per_step: Optional[int] = None
    precision: Literal["bf16", "mxfp8"] = "mxfp8"
    graphs: Literal["full", "partial", "eager"] = "full"
    ep_overlap: Literal["raw", "two_chunk"] = "two_chunk"
    waves_allowed: bool = True

    @property
    def eus(self) -> int:
        """``W = N / cp``."""
        return self.gpus // self.cp

    @property
    def tokens_per_gpu(self) -> int:
        """``T = S / cp``."""
        return self.seq_length // self.cp

    @property
    def lockstep(self) -> int:
        """``z``."""
        return self.lockstep_eus if self.lockstep_eus is not None else max(1, self.ep // self.cp)

    @property
    def domain_eus(self) -> Optional[int]:
        """EUs per domain (None: one domain)."""
        return None if self.domain_gpus is None else self.domain_gpus // self.cp

    @property
    def dense_axis(self) -> int:
        """The dense shard axis / Resident sync group: the domain's GPUs (all GPUs: one)."""
        return self.domain_gpus if self.domain_gpus is not None else self.gpus

    @property
    def microbatches(self) -> int:
        """Microbatches per step per EU."""
        return (
            self.samples_per_eu
            if self.microbatches_per_step is None
            else self.microbatches_per_step
        )

    def validate(self) -> None:
        """Raise ``PlannerUnsupported`` on a geometry planner v1 cannot model."""
        problems = []
        if self.gpus < 1 or self.cp < 1 or self.gpus % self.cp:
            problems.append(f"{self.gpus} GPUs do not split into CP groups of {self.cp}")
        if self.seq_length % self.cp:
            problems.append(f"S = {self.seq_length} does not split over cp = {self.cp}")
        if self.samples_per_eu < 1 or self.max_rounds < 1:
            problems.append("needs samples_per_eu >= 1 and R_max >= 1")
        elif self.microbatches < 1:
            problems.append(f"needs microbatches_per_step >= 1, got {self.microbatches}")
        elif self.samples_per_eu % self.microbatches:
            problems.append(
                f"n = {self.samples_per_eu} is not a multiple of the {self.microbatches} "
                "microbatches (n = micro batch size x microbatches)"
            )
        if self.ep < 1:
            problems.append(f"EP = {self.ep}")
        elif self.gpus >= 1 and (self.ep > self.gpus or self.gpus % self.ep):
            problems.append(f"EP = {self.ep} does not divide the {self.gpus} GPUs")
        elif self.domain_gpus is not None and self.domain_gpus % self.ep:
            problems.append(f"EP = {self.ep} does not divide a domain of {self.domain_gpus} GPUs")
        W = self.eus if self.gpus % max(self.cp, 1) == 0 else 0
        if W and W % self.lockstep:
            problems.append(f"{W} EUs do not form lockstep units of {self.lockstep}")
        if self.domain_gpus is not None and W:
            if self.domain_gpus % self.cp or self.ep > self.domain_gpus:
                problems.append(f"the EP group must fit a domain of {self.domain_gpus} GPUs")
            elif self.domain_eus < W and (W % self.domain_eus or self.domain_eus % self.lockstep):
                problems.append(f"{W} EUs do not split into domains of {self.domain_eus}")
        if problems:
            raise PlannerUnsupported("planner v1 cannot model this phase: " + "; ".join(problems))


@dataclass(frozen=True)
class PlannerCapabilities:
    """What this run can apply. ``layer_window``: the M3 parameter store (core and edge dense
    groups windowed, moves E / M, re-gathering); ``grace_state_tier``: round-state data on Grace
    (M6a); ``optimizer_offload``: Adam to Grace (optimizer CPU offload; the trainer decides and
    says why not in ``optimizer_offload_reason``); ``coda_recompute``: D42 (stage 4)."""

    layer_window: bool = False
    grace_state_tier: bool = False
    optimizer_offload: bool = False
    coda_recompute: bool = True
    optimizer_offload_reason: str = "not allowed by the trainer"

    @staticmethod
    def target() -> "PlannerCapabilities":
        """Every capability (the cost model's world: the cross-check and the target pick)."""
        return PlannerCapabilities(True, True, True, True, "")


@dataclass(frozen=True)
class PlannerOption:
    """One point of the option space."""

    form: Literal["block", "nested"]
    stride_layers: Optional[int]
    regather: bool
    coda_recompute: bool
    waves: int
    moves: Literal["-", "A", "AE", "AEM"]
    layout: Literal["resident", "layer_window"]

    @property
    def label(self) -> str:
        """The cost model's spec name: ``nested``, ``w 6``, ``w+rc 4``, ``r 9``, ``r+rc 9``."""
        if self.form == "nested":
            return "nested"
        return (
            ("r" if self.regather else "w")
            + ("+rc" if self.coda_recompute else "")
            + (f" {self.stride_layers}")
        )

    def requires(self) -> frozenset:
        """The capabilities the option needs (``grace_state_tier`` is added by the evaluation
        when round-state data spills to Grace)."""
        need = set()
        if self.layout == "layer_window" or self.regather or "E" in self.moves or "M" in self.moves:
            need.add("layer_window")
        if "A" in self.moves:
            need.add("optimizer_offload")
        if self.coda_recompute:
            need.add("coda_recompute")
        return frozenset(need)


@dataclass(frozen=True)
class OptionResult:
    """One evaluated option (bytes, ms; ``mfu`` a fraction).

    The placement follows the cost model (round-state data HBM first, then Grace); ``fits`` is
    the cost model's fit (HBM within the cap, Grace within capacity and above its margin, C2C
    within its cap) and holds under ``PlannerCapabilities.target()``. ``missing``: the
    capabilities the option needs and this run lacks (``applicable`` when none); ``refusals``:
    every reason the option cannot be applied in this run, in words."""

    option: PlannerOption
    hbm_bytes: float
    grace_bytes: float
    adam_grace_bytes: float
    store_bytes: float
    stash_need_bytes: float
    spill_hbm_bytes: float
    spill_grace_bytes: float
    components: Mapping[str, float]
    unit_ms: Mapping[str, float]
    makespan_ms: float
    step_level_ms: float
    resident_sync_ms: float
    step_ms: float
    ideal_ms: float
    sched_eff: float
    mfu: float
    peak_units: float
    peak_sample_rounds: Optional[float]
    c2c_write: float
    c2c_read: float
    charged_share: float
    iterations: int
    policy: str
    fits: bool
    refusals: Tuple[str, ...]
    applicable: bool
    missing: Tuple[str, ...]

    @property
    def footprint_gb(self) -> float:
        """HBM + Grace in GB (the tie-break, summed in GB as the cost model's pick)."""
        return self.hbm_bytes / GB + self.grace_bytes / GB

    def to_json(self) -> dict:
        """A JSON-ready dict (GB, ms, MFU %)."""
        o = self.option
        return dict(
            label=o.label,
            form=o.form,
            stride_layers=o.stride_layers,
            regather=o.regather,
            coda_recompute=o.coda_recompute,
            waves=o.waves,
            moves=o.moves,
            layout=o.layout,
            hbm_gb=self.hbm_bytes / GB,
            grace_gb=self.grace_bytes / GB,
            adam_grace_gb=self.adam_grace_bytes / GB,
            store_gb=self.store_bytes / GB,
            stash_need_gb=self.stash_need_bytes / GB,
            spill_hbm_gb=self.spill_hbm_bytes / GB,
            spill_grace_gb=self.spill_grace_bytes / GB,
            components_gb={k: v / GB for k, v in self.components.items()},
            unit_ms=dict(self.unit_ms),
            makespan_ms=self.makespan_ms,
            step_level_ms=self.step_level_ms,
            resident_sync_ms=self.resident_sync_ms,
            step_ms=self.step_ms,
            ideal_ms=self.ideal_ms,
            sched_eff=self.sched_eff,
            mfu_pct=100 * self.mfu,
            peak_units=self.peak_units,
            peak_sample_rounds=self.peak_sample_rounds,
            c2c_write_gbps=self.c2c_write / GB,
            c2c_read_gbps=self.c2c_read / GB,
            charged_share=self.charged_share,
            iterations=self.iterations,
            policy=self.policy,
            fits=self.fits,
            applicable=self.applicable,
            missing=list(self.missing),
            refusals=list(self.refusals),
        )


class PlannerNoFit(RuntimeError):
    """No applicable option fits; ``table`` holds every evaluated option."""

    def __init__(self, message: str, table: Sequence[OptionResult] = ()):
        super().__init__(message)
        self.table = tuple(table)


_UNAVAILABLE = {
    "E": "prelude/coda/MTP dense groups to the Layer window (M3)",
    "M": "embedding and head to Resident + sharded grads (M3)",
    "layout": "core dense in the Layer window (M3)",
    "regather": "re-gathered block weights (Layer window, M3)",
    "grace_state_tier": "round-state data on Grace (M6a)",
    "optimizer_offload": "Adam to Grace (optimizer CPU offload)",
    "coda_recompute": "D42 coda up-projection recompute",
}


@dataclass(frozen=True)
class LoopRSBFPlan:
    """The planner's decision for one phase.

    ``item_form`` / ``block_stride`` (core **sublayers**, the config field: ``2 s``) /
    ``waves`` / ``coda_recompute`` are what ``apply_plan_to_config`` sets; ``optimizer_offload``
    is Adam to Grace in the applied option (the trainer's ``optimizer_cpu_offload``).
    ``applied``: the best fitting applicable option the form policy allows; ``raw``: the best
    fitting applicable option of the whole menu; ``target``: the best fitting option under
    M3 capabilities; ``unavailable``: what the target needs that this run lacks; ``table``:
    every evaluated option; ``digest``: sha256 of the inputs and the choice (rank agreement)."""

    phase: PhaseGeometry
    depth: DepthDistribution
    hardware: PlannerHardware
    capabilities: PlannerCapabilities
    item_form: Literal["whole_round", "block", "nested"]
    block_stride: Optional[int]
    stride_layers: Optional[int]
    waves: int
    coda_recompute: bool
    optimizer_offload: bool
    applied: OptionResult
    raw: Optional[OptionResult]
    target: Optional[OptionResult]
    form_policy: str
    core_layers: int
    warnings: Tuple[str, ...]
    unavailable: Tuple[str, ...]
    notes: Tuple[str, ...]
    table: Tuple[OptionResult, ...]
    digest: str

    def config_overrides(self) -> Dict[str, Any]:
        """The ``TransformerConfig`` fields the plan sets."""
        return {
            "loop_rsbf_item_form": self.item_form,
            "loop_rsbf_block_stride": self.block_stride,
            "loop_rsbf_waves": self.waves,
            "loop_rsbf_coda_recompute_mla_up_proj": self.coda_recompute,
        }

    def to_json(self) -> dict:
        """The plan, its inputs and every evaluated option, JSON-ready."""
        return dict(
            phase=dataclasses.asdict(self.phase),
            depth=dict(
                kind=self.depth.kind,
                params=list(self.depth.params),
                max_rounds=self.depth.max_rounds,
                name=self.depth.name,
            ),
            hardware=dataclasses.asdict(self.hardware),
            capabilities=dataclasses.asdict(self.capabilities),
            form_policy=self.form_policy,
            core_layers=self.core_layers,
            config_overrides=self.config_overrides(),
            optimizer_offload=self.optimizer_offload,
            applied=self.applied.to_json(),
            raw=None if self.raw is None else self.raw.to_json(),
            target=None if self.target is None else self.target.to_json(),
            warnings=list(self.warnings),
            unavailable=list(self.unavailable),
            notes=list(self.notes),
            table=[r.to_json() for r in self.table],
            digest=self.digest,
        )


def executor_nested_rebuild_fraction(core_layers: int) -> float:
    """The rebuild a nested item of the executor runs, as a fraction of one sweep round: it
    rebuilds ``[0, rebuild_end)`` with ``rebuild_end = (n_b - 1) s`` (D-S4-3), i.e. ``(L - 1) /
    L`` at the planner's per-layer nested item. The cost model charges 1.0 (5 units)."""
    return (core_layers - 1) / core_layers if core_layers > 0 else 0.0


def option_item_form(option: PlannerOption, core_layers: int) -> Tuple[str, Optional[int]]:
    """``(loop_rsbf_item_form, loop_rsbf_block_stride)`` of an option on a core of
    ``core_layers`` layers: block ``s = L_core`` is the whole round (no stride), block ``s <
    L_core`` the block item with ``2 s`` core sublayers (pair-aligned), nested the per-layer
    nested item (stride 2 sublayers)."""
    if option.form == "nested":
        return "nested", 2
    if option.stride_layers == core_layers:
        return "whole_round", None
    return "block", 2 * option.stride_layers


def mfu_digits(results: Sequence[OptionResult]) -> int:
    """The rounding of the MFU tie rule over a menu: the cost model's ``round(mfu, 5)`` when its
    best MFU is in ``[0.1, 1)``, one more digit per decade below (small dev models model MFUs of
    1e-2 .. 1e-6, which an absolute 1e-5 would all tie). Every cross-check row keeps 5."""
    best = max((r.mfu for r in results), default=0.0)
    if not best > 0 or not math.isfinite(best):
        return 5
    return 5 + max(0, -math.floor(math.log10(best)) - 1)


def _select_key(result: OptionResult, digits: int = 5) -> tuple:
    return (round(result.mfu, digits), -result.footprint_gb, -result.option.waves)


# ----------------------------------------------------------------------------- the planner
class LoopRSBFPlanner:
    """Planner v1: enumerate, evaluate and select the RSBF executor's options.

    Args:
        shape: the model (``ModelShape.from_model`` at runtime, ``from_dims`` for the cost
            model's shapes).
        hardware: ``PlannerHardware`` (``preset("gb200" | "h100", ...)``).
        capabilities: what this run can apply (default: today's: resident layout, D42, no Adam
            to Grace unless the trainer allows it).
        strides_layers: block strides in layers (``s > L_core`` dropped; ``s = L_core``, the
            whole round, added when ``L_core <= max(strides)`` and missing, and always under
            ``form_policy = "d28"`` on a core of at most ``small_core_layers``).
        waves: wave counts (kept when they divide ``n`` and the microbatches).
        move_steps: the cumulative move steps (the resident layout offers ``-`` / ``A`` only).
        layouts: dense layouts to evaluate (None: ``resident`` and ``layer_window``).
        peak_trials: memory peaks over trials, ``max`` (the plan's physical rule) or ``avg``.
        trials: simulation trials (None: 2 if ``W <= 256`` else 1).
        form_policy: ``d28`` (the whole round on ``L_core <= small_core_layers``, the full menu
            above), ``nested`` (nested only) or ``planner`` (unconstrained; the cross-check).
        sched_policy: ``executor`` (``bulk_only`` at ``z > 1``, ``fill`` at ``z = 1``: what the
            executor runs) or ``min_makespan`` (the cost model's per-form choice).
        rebuild_fraction: the nested item's rebuild in sweep rounds (1.0: the cost model;
            ``executor_nested_rebuild_fraction(L_core)``: what the executor runs).
        wsim_max: simulate at most this many EUs (the cost model's ``Wsim_max``).
        prune_gb: a move step whose uncharged placement misses memory by more is not
            re-simulated (``evaluate_moves.prune_gb``).
    """

    def __init__(
        self,
        shape: ModelShape,
        hardware: PlannerHardware,
        *,
        capabilities: PlannerCapabilities = PlannerCapabilities(),
        strides_layers: Sequence[int] = (1, 2, 3, 4, 6, 9, 18),
        waves: Sequence[int] = (1, 2, 4, 8),
        move_steps: Sequence[str] = ("-", "A", "AE", "AEM"),
        layouts: Optional[Sequence[str]] = None,
        peak_trials: Literal["max", "avg"] = "max",
        trials: Optional[int] = None,
        form_policy: Literal["d28", "nested", "planner"] = "d28",
        small_core_layers: int = 4,
        sched_policy: Literal["executor", "min_makespan"] = "executor",
        rebuild_fraction: float = 1.0,
        wsim_max: Optional[int] = 1024,
        prune_gb: float = 2.0,
    ):
        if form_policy not in ("d28", "nested", "planner"):
            raise ValueError(f"unknown form policy {form_policy!r}")
        if sched_policy not in ("executor", "min_makespan"):
            raise ValueError(f"unknown schedule policy rule {sched_policy!r}")
        if peak_trials not in ("max", "avg"):
            raise ValueError(f"unknown peak rule {peak_trials!r}")
        for step in move_steps:
            if step not in ("-", "A", "AE", "AEM"):
                raise ValueError(f"unknown move step {step!r}")
        layouts = ("resident", "layer_window") if layouts is None else tuple(layouts)
        for layout in layouts:
            if layout not in ("resident", "layer_window"):
                raise ValueError(f"unknown layout {layout!r}")
        if any(int(w) < 1 for w in waves):
            raise ValueError(f"wave counts must be >= 1, got {tuple(waves)}")
        if any(int(s) < 1 for s in strides_layers):
            raise ValueError(f"block strides must be >= 1 layer, got {tuple(strides_layers)}")
        if shape.core_layers < 1:
            raise PlannerUnsupported("the model has no core layer")
        self.shape = shape
        self.hardware = hardware
        self.capabilities = capabilities
        self.strides_layers = tuple(sorted(set(int(s) for s in strides_layers)))
        self.waves = tuple(int(w) for w in waves)
        self.move_steps = tuple(move_steps)
        self.layouts = layouts
        self.peak_trials = peak_trials
        self.trials = trials
        self.form_policy = form_policy
        self.small_core_layers = small_core_layers
        self.sched_policy = sched_policy
        self.rebuild_fraction = rebuild_fraction
        self.wsim_max = wsim_max
        self.prune_gb = prune_gb

    # ------------------------------------------------------------------ menu
    def _strides(self) -> List[int]:
        L = self.shape.core_layers
        strides = [s for s in self.strides_layers if 1 <= s <= L]
        add_whole_round = bool(self.strides_layers) and L <= max(self.strides_layers)
        # D28's applied menu on a small core is the whole round: offer it whatever the strides.
        add_whole_round |= self.form_policy == "d28" and L <= self.small_core_layers
        if add_whole_round and L not in strides:
            strides.append(L)
        return strides

    def _specs(self, layout: str) -> List[Tuple[str, Optional[int], bool, bool]]:
        """(form, stride, regather, coda_rc) in the cost model's menu order."""
        specs = []
        if self.shape.core_layers >= 2:
            specs.append(("nested", None, False, False))
        rc_ok = self.shape.has_coda_dsv4_attention
        for regather, rc in ((False, False), (False, True), (True, False), (True, True)):
            if rc and not rc_ok:
                continue
            if regather and layout != "layer_window":
                continue
            for s in self._strides():
                specs.append(("block", s, regather, rc))
        return specs

    def _valid_waves(self, phase: PhaseGeometry) -> Tuple[List[int], List[str]]:
        valid, notes = [], []
        n, k = phase.samples_per_eu, phase.microbatches
        for w in self.waves:
            if w != 1 and not phase.waves_allowed:
                notes.append(f"w={w} (waves need a fixed batch size)")
            elif n % w or k % w:
                notes.append(f"w={w} (does not divide n = {n} / the {k} microbatches)")
            else:
                valid.append(w)
        return valid, notes

    def options(self, phase: PhaseGeometry) -> List[PlannerOption]:
        """Every option of the menu, in evaluation order (waves, layout, form, moves)."""
        out = []
        valid, _ = self._valid_waves(phase)
        for w in valid:
            for layout in self.layouts:
                steps = [m for m in self.move_steps if layout == "layer_window" or m in ("-", "A")]
                for form, s, regather, rc in self._specs(layout):
                    for moves in steps:
                        out.append(PlannerOption(form, s, regather, rc, w, moves, layout))
        return out

    # ------------------------------------------------------------------ evaluation
    def _trials(self, W: int) -> int:
        if self.trials is not None:
            return self.trials
        return 2 if W <= 256 else 1

    def evaluate(self, phase: PhaseGeometry, depth: DepthDistribution) -> List[OptionResult]:
        """Evaluate every option of ``options(phase)`` under ``depth`` (the cost model's
        ``evaluate_moves`` per item form and wave count, ``rsbf_job``'s resident sync)."""
        phase.validate()
        if depth.max_rounds != phase.max_rounds:
            raise ValueError(
                f"the depth distribution has R_max {depth.max_rounds}, the phase "
                f"{phase.max_rounds}"
            )
        ph = SeqPhase.of(self.shape, phase.seq_length, phase.cp)
        cache: dict = {}
        flops_cache: dict = {}
        out = []
        valid, _ = self._valid_waves(phase)
        for w in valid:
            for layout in self.layouts:
                steps = [m for m in self.move_steps if layout == "layer_window" or m in ("-", "A")]
                for form, s, regather, rc in self._specs(layout):
                    out += self._evaluate_form(
                        phase,
                        depth,
                        ph,
                        w,
                        layout,
                        form,
                        s,
                        regather,
                        rc,
                        steps,
                        cache,
                        flops_cache,
                    )
        return out

    def _evaluate_form(
        self, phase, depth, ph, w, layout, form, s, regather, rc, steps, cache, flops_cache
    ) -> List[OptionResult]:
        hw = self.hardware
        shape = self.shape
        W = phase.eus
        z = phase.lockstep
        N = phase.gpus
        n = float(phase.samples_per_eu * W) / W
        trials = self._trials(W)
        block = form == "block"
        L = shape.core_layers
        nblk = math.ceil(L / s) if block else 0
        stash_units = nblk - 1 if block else 0
        state = ph.state_gpu
        per_sr = stash_units * state
        z_lw = phase.dense_axis
        ckw = dict(
            prec=phase.precision,
            EP=phase.ep,
            ep_overlap=phase.ep_overlap,
            graphs=phase.graphs,
            nested=not block,
            lw_core=layout == "layer_window",
            z_lw=z_lw,
            rebuild_fraction=self.rebuild_fraction,
        )
        if block and regather:
            ckw["whole_round"] = True
        if rc:
            ckw["coda_rc"] = True
        skw = dict(
            trials=trials,
            domain_eus=phase.domain_eus,
            peak_rule=self.peak_trials,
            sync=hw.sync_ms,
            wsim_max=self.wsim_max,
            cache=cache,
        )

        def cost(code, cb):
            k = dict(ckw)
            if "E" in code:
                k["lw_edges"] = True
            if "M" in code:
                k["emb_rs"] = True
            if cb:
                k["c2c_bytes"] = cb
            return costs.unit_costs(shape, ph, hw, **k)

        base = cost("-", None)
        d0 = costs.unit_costs(shape, ph, hw, nominal=True, **ckw).d
        asc = base.moe_share
        if self.sched_policy == "min_makespan":
            pol, res0 = choose_policy(depth, n, W, z, d0, waves=w, active_scale=asc, **skw)
        else:
            pol = "bulk_only" if z > 1 else "fill"
            res0 = run_step_schedule(
                depth, n, W, z, d0, waves=w, policy=pol, active_scale=asc, sr_track=None, **skw
            )
        srk = 0 if block else None
        mkw = dict(
            EP=phase.ep, N=N, prec=phase.precision, layout=layout, nested=not block, z_lw=z_lw
        )
        if block:
            mkw.update(block_stride=s, block_window=not regather, coda_rc=rc)

        def place(code, res):
            comps, adam_grace = costs.memory_components(
                shape,
                ph,
                hw,
                res["peak"],
                adam_on_grace="A" in code,
                edges_mode="Layer window" if "E" in code else "Resident",
                emb_mode="Resident + sharded grads" if "M" in code else "Resident",
                **mkw,
            )
            need = (res["peak_sr"] + 1) * per_sr if block else 0.0
            return comps, costs.place_round_state(comps, adam_grace, stash_need=need, hw=hw)

        ideal = flops = None
        results = []
        for code in steps:
            cs = 0.0
            it = 0
            while True:
                if cs > 0:
                    fw, br, bw = costs.c2c_traffic(cs, cs, stash_units, state)
                    cb = dict(f=fw, b=max(br, bw))
                else:
                    cb = None
                uc = cost(code, cb)
                res = run_step_schedule(
                    depth,
                    n,
                    W,
                    z,
                    uc.d,
                    waves=w,
                    policy=pol,
                    active_scale=uc.moe_share,
                    sr_track=srk,
                    **skw,
                )
                comps, pl = place(code, res)
                if pl.share <= cs + 1e-4 or it >= 6:
                    break
                if it == 0 and not pl.mem_ok and pl.deficit > self.prune_gb * GB:
                    break
                cs = pl.share
                it += 1
            if ideal is None:
                Wsim = res["Wsim"]
                d0_ = d0
                ideal = sum(
                    sum(
                        d0_["sweep"] + x * (d0_["f"] + d0_["b"]) + d0_["coda"] + d0_["pre"]
                        for x in ds
                    )
                    / Wsim
                    for ds in res["depths"]
                ) / len(res["depths"])

                def mfps(x):
                    if x not in flops_cache:
                        flops_cache[x] = costs.model_flops_per_sample(shape, ph, x, hw)
                    return flops_cache[x]

                flops = sum(sum(mfps(x) for x in ds) * (W / Wsim) for ds in res["depths"]) / len(
                    res["depths"]
                )
            sh = pl.share
            fw, br, bw = costs.c2c_traffic(sh, sh, stash_units, state)
            d = uc.d
            wr = fw / (d["f"] / 1e3)
            rd = br / (d["b"] / 1e3)
            wb = bw / (d["b"] / 1e3)
            rate = max(wr, rd, wb)
            c2c_ok = hw.c2c_margin * rate <= hw.c2c_bytes_per_s
            stepl = costs.step_level_ms(
                shape, hw, EP=phase.ep, N=N, adam_on_grace="A" in code, dense_axis=z_lw
            )
            stepl_ms = sum(stepl.values())
            st_ms = res["makespan"] + stepl_ms
            mfu_snapshot = flops / (N * st_ms / 1e3 * hw.peak_flops)
            rs = costs.resident_sync_ms(shape, hw, code, z_lw, layout)
            step_ms = (res["makespan"] / 1e3 + stepl_ms / 1e3) * 1e3 + rs
            mfu = mfu_snapshot * (st_ms / 1e3) / (step_ms / 1e3)
            option = PlannerOption(form, s, regather, rc, w, code, layout)
            results.append(
                self._result(
                    option,
                    comps,
                    pl,
                    uc,
                    res,
                    pol,
                    ideal,
                    res0,
                    stepl_ms,
                    rs,
                    step_ms,
                    mfu,
                    max(wr, wb),
                    rd,
                    rate,
                    c2c_ok,
                    cs,
                    it,
                )
            )
        return results

    def _result(
        self,
        option,
        comps,
        pl,
        uc,
        res,
        pol,
        ideal,
        res0,
        stepl_ms,
        rs,
        step_ms,
        mfu,
        c2c_write,
        c2c_read,
        rate,
        c2c_ok,
        charged,
        iterations,
    ) -> OptionResult:
        hw = self.hardware
        caps = self.capabilities
        grace_margin_ok = pl.grace == 0 or (
            hw.grace_bytes > 0
            and pl.grace / GB <= (hw.grace_bytes - hw.grace_margin_bytes) / GB + 1e-9
        )
        fits = pl.mem_ok and c2c_ok and grace_margin_ok
        need = set(option.requires())
        if pl.on_grace > 0:
            need.add("grace_state_tier")
        missing = sorted(name for name in need if not getattr(caps, name))
        # Round-state data with nowhere to go but HBM (no Grace hardware, or no Grace tier in
        # the executor) is an HBM overflow (design 4.4), named first; "no Grace tier" is kept
        # for Adam to Grace.
        state_stuck = pl.on_grace > 0 and (hw.grace_bytes == 0 or "grace_state_tier" in missing)
        refusals = []
        if pl.room < 0:
            refusals.append(f"HBM over cap by {-pl.room / GB:.2f} GB")
        elif state_stuck:
            refusals.append(
                f"HBM over cap by {pl.on_grace / GB:.2f} GB (round-state data; no Grace tier "
                "for it)"
            )
        # This run's Grace: a stuck spill stays in HBM (refused above), so the Grace capacity
        # and margin are checked without it.
        grace_run = pl.grace - pl.on_grace if state_stuck else pl.grace
        if grace_run > 0 and hw.grace_bytes == 0:
            if pl.adam_grace > 0:
                refusals.append("no Grace tier")
        elif grace_run > hw.grace_bytes:
            refusals.append(f"Grace {grace_run / GB:.2f} GB > {hw.grace_bytes / GB:.1f} GB")
        elif grace_run > 0 and (
            grace_run / GB > (hw.grace_bytes - hw.grace_margin_bytes) / GB + 1e-9
        ):
            refusals.append(
                f"Grace margin {(hw.grace_bytes - grace_run) / GB:.2f} GB < "
                f"{hw.grace_margin_bytes / GB:.1f} GB"
            )
        if not c2c_ok:
            refusals.append(
                f"C2C need {rate / GB:.1f} GB/s > {hw.c2c_bytes_per_s / GB:.0f} GB/s / "
                f"{hw.c2c_margin:g}"
            )
        for name in missing:
            if name == "layer_window":
                refusals.append("needs Layer window (M3)")
            elif name == "optimizer_offload":
                refusals.append(f"needs Adam to Grace (excluded: {caps.optimizer_offload_reason})")
            elif name == "coda_recompute":
                refusals.append("needs D42 (not available)")
        return OptionResult(
            option=option,
            hbm_bytes=pl.hbm,
            grace_bytes=pl.grace,
            adam_grace_bytes=pl.adam_grace,
            store_bytes=pl.store,
            stash_need_bytes=pl.need,
            spill_hbm_bytes=pl.in_hbm,
            spill_grace_bytes=pl.on_grace,
            components=dict(comps),
            unit_ms=dict(uc.d),
            makespan_ms=res["makespan"],
            step_level_ms=stepl_ms,
            resident_sync_ms=rs,
            step_ms=step_ms,
            ideal_ms=ideal,
            sched_eff=ideal / res0["makespan"],
            mfu=mfu,
            peak_units=res["peak"],
            peak_sample_rounds=res["peak_sr"],
            c2c_write=c2c_write,
            c2c_read=c2c_read,
            charged_share=charged,
            iterations=iterations,
            policy=pol,
            fits=fits,
            refusals=tuple(refusals),
            applicable=not missing,
            missing=tuple(missing),
        )

    # ------------------------------------------------------------------ selection
    @staticmethod
    def select(results: Sequence[OptionResult], *, applicable_only: bool) -> Optional[OptionResult]:
        """The best fitting option: the highest ``round(mfu, mfu_digits(fitting))`` (5 at MFU
        >= 0.1), then the smallest HBM +
        Grace, then fewer waves; the first in evaluation order on a full tie
        (``plan_numbers.pick``)."""
        candidates = [r for r in results if r.fits and (r.applicable or not applicable_only)]
        digits = mfu_digits(candidates)
        best = None
        for r in candidates:
            if best is None or _select_key(r, digits) > _select_key(best, digits):
                best = r
        return best

    def _form_allowed(self, option: PlannerOption) -> bool:
        L = self.shape.core_layers
        if self.form_policy == "planner":
            return True
        if self.form_policy == "nested":
            return option.form == "nested"
        if L <= self.small_core_layers:
            return option.form == "block" and option.stride_layers == L
        return True

    def plan(self, phase: PhaseGeometry, depth: DepthDistribution) -> LoopRSBFPlan:
        """Evaluate and select; raises ``PlannerNoFit`` when no applicable option fits."""
        results = self.evaluate(phase, depth)
        if not results:
            _, wave_notes = self._valid_waves(phase)
            raise PlannerNoFit(
                "loop planner: the option menu is empty (no wave count offered: "
                + ("; ".join(wave_notes) or f"waves {list(self.waves)}")
                + ")",
                [],
            )
        policy_note = f"excluded by form policy {self.form_policy}"
        table = []
        for r in results:
            if not self._form_allowed(r.option):
                r = dataclasses.replace(r, refusals=r.refusals + (policy_note,))
            table.append(r)
        target = self.select(table, applicable_only=False)
        raw = self.select(table, applicable_only=True)
        applied = self.select(
            [r for r in table if self._form_allowed(r.option)], applicable_only=True
        )
        notes = self._notes(phase)
        if applied is None and raw is not None:
            raise PlannerNoFit(
                f"loop planner: form policy {self.form_policy} excluded every fitting option; "
                f"raw pick {raw.option.label} w={raw.option.waves} {raw.option.moves} "
                f"{raw.option.layout} (use --loop-planner-forms planner)",
                table,
            )
        if applied is None:
            raise PlannerNoFit(
                "loop planner: no applicable option fits "
                f"(S={phase.seq_length} cp={phase.cp} N={phase.gpus} EP={phase.ep} "
                f"n={phase.samples_per_eu} R_max={phase.max_rounds}, depths {depth.name}, "
                f"HBM cap {self.hardware.hbm_cap_bytes / GB:.1f} GB, form policy "
                f"{self.form_policy}); best refusals: "
                + "; ".join(
                    f"{r.option.label} w={r.option.waves} {r.option.moves} {r.option.layout}: "
                    + ", ".join(r.refusals)
                    for r in sorted(table, key=lambda r: (not r.applicable, r.hbm_bytes))[:3]
                ),
                table,
            )
        warnings = []
        if phase.samples_per_eu < 4:
            warnings.append(
                f"n = {phase.samples_per_eu} < 4 samples per EU (D21: n >= 4, n = 8 preferred at "
                "N = 64)"
            )
        unavailable = []
        if target is not None:
            o = target.option
            if o.layout == "layer_window" and not self.capabilities.layer_window:
                unavailable.append(_UNAVAILABLE["layout"])
            if o.regather and not self.capabilities.layer_window:
                unavailable.append(_UNAVAILABLE["regather"])
            for move in ("E", "M"):
                if move in o.moves and not self.capabilities.layer_window:
                    unavailable.append(_UNAVAILABLE[move])
            for name in ("grace_state_tier", "optimizer_offload", "coda_recompute"):
                if name in target.missing:
                    unavailable.append(_UNAVAILABLE[name])
        o = applied.option
        L = self.shape.core_layers
        item_form, stride = option_item_form(o, L)
        digest_input = dict(
            phase=dataclasses.asdict(phase),
            depth=[depth.kind, list(depth.params), depth.max_rounds],
            hardware=dataclasses.asdict(self.hardware),
            capabilities=dataclasses.asdict(self.capabilities),
            shape=_shape_key(self.shape),
            settings=[
                list(self.strides_layers),
                list(self.waves),
                list(self.move_steps),
                list(self.layouts),
                self.peak_trials,
                self.trials,
                self.form_policy,
                self.small_core_layers,
                self.sched_policy,
                self.rebuild_fraction,
            ],
            choice=[item_form, stride, o.waves, o.coda_recompute, o.moves, o.layout],
            mfu=repr(applied.mfu),
        )
        digest = hashlib.sha256(
            json.dumps(digest_input, sort_keys=True, default=str).encode()
        ).hexdigest()
        return LoopRSBFPlan(
            phase=phase,
            depth=depth,
            hardware=self.hardware,
            capabilities=self.capabilities,
            item_form=item_form,
            block_stride=stride,
            stride_layers=o.stride_layers,
            waves=o.waves,
            coda_recompute=o.coda_recompute,
            optimizer_offload="A" in o.moves,
            applied=applied,
            raw=raw,
            target=target,
            form_policy=self.form_policy,
            core_layers=L,
            warnings=tuple(warnings),
            unavailable=tuple(unavailable),
            notes=tuple(notes),
            table=tuple(table),
            digest=digest,
        )

    def _notes(self, phase: PhaseGeometry) -> List[str]:
        notes = ["not offered: cp axis (M11)"]
        _, wave_notes = self._valid_waves(phase)
        if wave_notes:
            notes.append("not offered: " + "; ".join(wave_notes))
        if "layer_window" not in self.layouts:
            notes.append("not offered: r variants and moves E / M (layout resident only)")
        elif not self.capabilities.layer_window:
            notes.append("layer_window options: target only (Layer window, M3)")
        if self.shape.core_layers < 2:
            notes.append("not offered: nested at L_core < 2")
        if not self.shape.has_coda_dsv4_attention:
            notes.append("not offered: +rc (no DSv4 attention in the coda / MTP)")
        if not self.capabilities.optimizer_offload:
            notes.append(f"Adam to Grace excluded: {self.capabilities.optimizer_offload_reason}")
        if len([s for s in self._strides() if s < self.shape.core_layers]) > 1:
            notes.append(
                "block strides tie on modelled time (no stride-dependent time term, as the cost "
                "model): the footprint picks the stride (calibrate at R2)"
            )
        return notes


def _shape_key(shape: ModelShape) -> dict:
    out = dataclasses.asdict(shape)
    out["params"]["dense_by_kind"] = dict(sorted(shape.params.dense_by_kind.items()))
    return out


# ----------------------------------------------------------------------------- application
def apply_plan_to_config(config: Any, plan: LoopRSBFPlan) -> Dict[str, Any]:
    """Set the plan's item form, block stride, waves and D42 on ``config`` (the model's
    ``TransformerConfig``, the object ``LoopProgram`` and the executor read when they are built
    at the first step) and re-run the RSBF config guard; restores the old values and re-raises
    when the guard rejects the plan (a planner bug). A plan for ``cp > 1``, or for a core of a
    different size than the config's (a stale plan: its whole round / strides would map onto
    another core), is not applicable (``PlannerUnsupported``). Returns the overrides."""
    if plan.phase.cp > 1:
        raise PlannerUnsupported(
            f"a plan for cp = {plan.phase.cp} is computed but not applicable (CP / THD is M11)"
        )
    if getattr(config, "loop_schedule", None) != "rsbf":
        raise PlannerUnsupported("the loop planner applies to loop_schedule='rsbf' only")
    from megatron.core.transformer.loop.guards import _config_core_sublayers, validate_rsbf_config

    core_sublayers = _config_core_sublayers(config)
    if core_sublayers is not None and core_sublayers != 2 * plan.core_layers:
        raise PlannerUnsupported(
            f"the plan is for a core of {plan.core_layers} layers ({2 * plan.core_layers} "
            f"sublayers); this config's core has {core_sublayers} sublayers (stale plan?)"
        )
    overrides = plan.config_overrides()
    old = {name: getattr(config, name) for name in overrides}
    for name, value in overrides.items():
        setattr(config, name, value)
    try:
        validate_rsbf_config(config)
    except Exception:
        for name, value in old.items():
            setattr(config, name, value)
        raise
    return overrides


# ----------------------------------------------------------------------------- reporting
def _row(index: int, r: OptionResult, hw: PlannerHardware, status: str) -> str:
    o = r.option
    if o.form == "nested":
        form, s = "nested", "-/2"
    else:
        form, s = "block", f"{o.stride_layers}/{2 * o.stride_layers}"
    var = "-" if o.form == "nested" else o.label.split()[0]
    margin = f"({(hw.grace_bytes - r.grace_bytes) / GB:.1f})" if hw.grace_bytes > 0 else "(-)"
    grace = f"{r.grace_bytes / GB:.2f} {margin}"
    spill = f"{r.spill_hbm_bytes / GB:.2f}+{r.spill_grace_bytes / GB:.2f}"
    c2c = "-/-" if r.spill_grace_bytes <= 0 else f"{r.c2c_write / GB:.0f}/{r.c2c_read / GB:.0f}"
    return (
        f"  {index:<3d}{form:<7s}{s:<9s}{var:<6s}{o.waves:<3d}{o.moves:<5s}{o.layout:<13s}"
        f"{r.hbm_bytes / GB:>8.2f}  {grace:<16s}{spill:>13s}  {c2c:>9s}  "
        f"{r.step_ms / 1e3:>8.3f}  {100 * r.mfu:>6.2f}   {status}"
    )


def format_decision_table(plan: LoopRSBFPlan, *, top: int = 12) -> str:
    """The decision table (a log text): the header, the ``top`` best fitting options (and the
    applied / raw / target picks), the refused options grouped by reason, the picks."""
    p, hw, caps = plan.phase, plan.hardware, plan.capabilities
    grace = (
        f"Grace {hw.grace_bytes / GB:.0f} GB (margin {hw.grace_margin_bytes / GB:.0f} GB), "
        f"C2C {hw.c2c_bytes_per_s / GB:.0f} GB/s x {hw.c2c_margin:g}"
        if hw.grace_bytes > 0
        else "no Grace tier"
    )
    layouts = sorted({r.option.layout for r in plan.table})
    lines = [
        f"[loop planner] phase S={p.seq_length} cp={p.cp} N={p.gpus} EP={p.ep} W={p.eus} "
        f"z={p.lockstep} n={p.samples_per_eu} microbatches={p.microbatches} R_max={p.max_rounds} "
        f"depths={plan.depth.name} (E[d] {plan.depth.mean():.3f})",
        f"  hardware={hw.name} HBM cap {hw.hbm_cap_bytes / GB:.1f} GB, {grace}, EP reserve "
        f"{hw.ep_reserve_bytes / GB:.0f} GB; precision {p.precision}, graphs {p.graphs}, EP "
        f"overlap {p.ep_overlap}; layouts {', '.join(layouts)}; capabilities: "
        + ", ".join(
            name
            for name in ("layer_window", "grace_state_tier", "optimizer_offload", "coda_recompute")
            if getattr(caps, name)
        ),
        f"  forms={plan.form_policy} (L_core {plan.core_layers})",
    ]
    lines += [f"  {note}" for note in plan.notes]
    lines += [f"  WARNING: {w}" for w in plan.warnings]
    lines.append(
        "  #  form   s(L/sub) var   w  moves layout        HBM GB  Grace GB (margin)  "
        "spill H+G GB  C2C w/r GB/s  step s   MFU %   status"
    )
    fitting = [r for r in plan.table if r.fits]
    digits = mfu_digits(fitting)
    ranked = sorted(fitting, key=lambda r: (_select_key(r, digits), r.applicable), reverse=True)
    shown = ranked[:top]
    for pick in (plan.applied, plan.raw, plan.target):
        if pick is not None and pick not in shown:
            shown.append(pick)
    for index, r in enumerate(shown, 1):
        tags = []
        if r is plan.applied:
            tags.append("APPLIED" + (" (whole round)" if plan.item_form == "whole_round" else ""))
        if r is plan.raw and r is not plan.applied:
            tags.append("RAW pick")
        if r is plan.target and r is not plan.applied:
            tags.append("TARGET pick")
        tags += list(r.refusals)
        lines.append(_row(index, r, hw, "; ".join(tags) or "fits"))
    refused = {}
    for r in plan.table:
        if r.fits:
            continue
        reason = r.refusals[0] if r.refusals else "does not fit"
        refused.setdefault(re.sub(r"\d+(\.\d+)?", "#", reason), []).append(r)
    for reason, rs in refused.items():
        lines.append(
            f"  refused ({len(rs)}): {reason}: e.g. {rs[0].option.label} w={rs[0].option.waves} "
            f"{rs[0].option.moves} {rs[0].option.layout} ({', '.join(rs[0].refusals)})"
        )

    def short(r):
        return f"{r.option.label} w={r.option.waves} {r.option.moves} {r.option.layout}"

    if plan.raw is not None:
        delta = 100 * (plan.applied.mfu - plan.raw.mfu)
        digits = mfu_digits([plan.raw, plan.applied])
        rounded = round(plan.applied.mfu, digits) == round(plan.raw.mfu, digits)
        same = "equal" if rounded else f"{delta:+.2f} pts"
        lines.append(f"  raw vs applied: {short(plan.raw)} vs {short(plan.applied)}, MFU {same}")
    if plan.target is None:
        lines.append("  target (M3 capabilities): nothing fits")
    elif plan.target is plan.applied:
        lines.append("  target (M3 capabilities): same as applied")
    else:
        lines.append(
            f"  target (M3 capabilities): {short(plan.target)} MFU {100 * plan.target.mfu:.2f} %"
        )
    for item in plan.unavailable:
        lines.append(f"  requested but unavailable: {item}")
    lines.append(
        f"  plan: loop_rsbf_item_form={plan.item_form} loop_rsbf_block_stride={plan.block_stride} "
        f"loop_rsbf_waves={plan.waves} coda_recompute={'on' if plan.coda_recompute else 'off'} "
        f"optimizer_offload={'on' if plan.optimizer_offload else 'off'} "
        f"HBM {plan.applied.hbm_bytes / GB:.2f} GB step {plan.applied.step_ms / 1e3:.3f} s "
        f"MFU {100 * plan.applied.mfu:.2f} % digest={plan.digest[:12]}"
    )
    return "\n".join(lines)


def log_decision_table(plan: LoopRSBFPlan, *, top: int = 12, log: Any = None) -> str:
    """Log the decision table at INFO and every plan warning at WARNING (``log``: a logger,
    default this module's); returns the table text."""
    log = logger if log is None else log
    text = format_decision_table(plan, top=top)
    log.info("%s", text)
    for warning in plan.warnings:
        log.warning("loop planner: %s", warning)
    return text
