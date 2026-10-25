# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Training side of the loop RSBF executor: the realized exit-depth histogram and the planner.

**Exit histogram.** After every RSBF training step ``train()`` calls
``record_loop_exit_histogram``: it reads the step's histogram
(``LoopRSBFSchedule.exit_counts``: ``counts[d - 1]`` samples of the step exited after round
``d``, plan-derived and identical on every rank, so no communication) and this
rank's padded / issued tokens, checks that the counts sum to the step's samples, keeps them on
``args`` as plain lists (``_loop_exit_counts``, ``_loop_padding``, the interval sums
``_loop_exit_counts_interval``; checkpoints pickle ``args``, so the process-wide
``LoopExitHistogramLog`` that holds the interval and the file is not kept there) and, on the
last rank with ``--loop-exit-histogram-path``, appends one JSON line per step::

    {"format": "megatron-loop-exit-histogram/1", "iteration": 17, "R_max": 4,
     "counts": [130, 128, 120, 134], "samples": 512, "padding_fraction": 0.031,
     "phase": {"seq_length": 4096, "cp": 1, "gpus": 8, "ep": 8, "samples_per_eu": 64,
               "waves": 1}}

That file is what the planner (``DepthDistribution.from_histogram_file``) and the toolkit's
cost model (``measured_depth.py``) re-run on. ``training_log`` writes this step's values to
TensorBoard / W&B at the TensorBoard interval (``log_loop_exit_histogram``) and appends the
interval sums to the console line (``loop_exit_histogram_log_string``). The padding fraction
is the logging rank's EU's (its home samples), not the step's.

**Planner** (``--loop-planner report|apply``). ``setup_model_and_optimizer`` calls
``plan_loop_rsbf`` after the model is built and before the optimizer config is read. Every rank
runs ``LoopRSBFPlanner`` (deterministic host code) on the built model's shape
(``ModelShape.from_model``), the phase (``loop_planner_phase``: S, cp, GPUs, EP, samples and
microbatches per EU, R_max; the executor's lockstep unit is the whole domain), a depth
distribution (``loop_planner_depths``: the config's, a measured exit histogram or an R4
setting), the hardware (``loop_planner_hardware``: a preset plus the HBM cap / Grace / reserve
/ C2C flags) and what this run can apply (``loop_planner_capabilities``: the resident layout,
D42 when the coda or MTP has a DSv4 attention, Adam to Grace per
``--loop-planner-optimizer-offload``). Rank 0 logs the decision table (and writes
``plan.to_json()`` to ``--loop-planner-report-path``); every rank checks that all ranks hold
the same plan digest.

* ``apply``: the planner owns ``--loop-rsbf-item-form``, ``--loop-rsbf-block-stride``,
  ``--loop-rsbf-waves`` and ``--loop-rsbf-coda-recompute-mla-up-proj`` (a value other than the
  parser default raises; ``report`` checks a user's choice instead). The plan's four values are
  set on the model's config (``apply_plan_to_config``: the object the RSBF program and executor
  read when the first step builds them) and mirrored into ``args``; Adam to Grace in the applied
  option sets ``optimizer_cpu_offload`` (``optimizer_offload_fraction`` 1) before the optimizer
  is built. The other planner moves are reported as requested but unavailable (M3).
  ``check_resume_offload`` refuses a resume whose planned optimizer offload differs from the
  checkpoint's (the optimizer state layout differs).
* ``report``: logs the plan and whether the current config's option fits; changes nothing.

``LoopPlannerSession.on_phase_start`` is the entry of every phase start (today: startup and
every resume). A later phase switch (M11) calls it again: the plan is re-applied and the
model's cached executor dropped (``LoopRSBFSchedule.invalidate``) so the next step rebuilds it.
The session lives in this module, not on ``args`` (checkpoints pickle ``args``).
"""

import json
import logging
import math
import os
from types import SimpleNamespace
from typing import Any, List, Mapping, Optional, Sequence

import torch

from megatron.core.num_microbatches_calculator import get_current_running_global_batch_size
from megatron.core.transformer.loop.planner import (
    EXIT_HISTOGRAM_FORMAT,
    R4_GEOM,
    R4_SKEW,
    R4_UNIF,
    DepthDistribution,
    LoopRSBFPlan,
    LoopRSBFPlanner,
    PhaseGeometry,
    PlannerCapabilities,
    PlannerHardware,
    PlannerNoFit,
    PlannerUnsupported,
    apply_plan_to_config,
    executor_nested_rebuild_fraction,
    format_decision_table,
    option_item_form,
)
from megatron.core.transformer.loop.planner_costs import GB, ModelShape
from megatron.core.utils import get_attr_wrapped_model, get_model_config
from megatron.training.utils import is_last_rank, is_rank0, print_rank_0, warn_rank_0

logger = logging.getLogger(__name__)

__all__ = [
    "LOOP_PLANNER_OWNED_FLAGS",
    "LoopExitHistogramLog",
    "LoopPlannerSession",
    "check_resume_offload",
    "get_loop_exit_histogram_log",
    "get_loop_planner_session",
    "log_loop_exit_histogram",
    "loop_exit_histogram_log_string",
    "loop_exit_histogram_phase",
    "loop_exit_histogram_scalars",
    "loop_optimizer_offload_blockers",
    "loop_planner_capabilities",
    "loop_planner_current_option",
    "loop_planner_depths",
    "loop_planner_hardware",
    "loop_planner_phase",
    "loop_planner_shape",
    "loop_planner_user_flags",
    "make_loop_planner",
    "plan_loop_rsbf",
    "record_loop_exit_histogram",
    "reset_loop_exit_histogram_log",
    "reset_loop_planner_session",
]

# The process's exit-histogram log (one training run per process; tests reset it).
_EXIT_HISTOGRAM = SimpleNamespace(log=None)


def _padding_fraction(padding: Optional[Sequence[int]]) -> float:
    if padding is None or not padding[1]:
        return 0.0
    return padding[0] / padding[1]


def _mean_depth(counts: Sequence[int]) -> float:
    return math.fsum(d * c for d, c in enumerate(counts, 1)) / sum(counts)


class LoopExitHistogramLog:
    """The exit histograms of the steps of a log interval, and the per-step JSONL file.

    Args:
        path: the JSONL file (appended, one line per step, flushed); None writes nothing (every
            rank but the last, or no ``--loop-exit-histogram-path``).
    """

    def __init__(self, path: Optional[str]):
        self.path = path
        # The --loop-exit-histogram-path the log was made for (None on the ranks that do not
        # write it too); get_loop_exit_histogram_log compares it.
        self.requested_path = path
        self._counts: list = []
        self._padded = 0
        self._issued = 0
        if path:
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)

    def record(
        self,
        iteration: int,
        counts: Sequence[int],
        samples: int,
        padding: Optional[Sequence[int]],
        phase: Mapping[str, int],
    ) -> None:
        """Add one step: its histogram ``counts`` (``counts[d - 1]`` samples of depth ``d``),
        its sample count, this EU's ``(padded, issued)`` tokens (None: unknown, logged as 0) and
        the phase geometry; appends the JSON line when the log has a path."""
        counts = [int(c) for c in counts]
        if self._counts and len(self._counts) != len(counts):
            raise ValueError(
                f"an exit histogram of R_max {len(counts)} in an interval of R_max "
                f"{len(self._counts)}"
            )
        self._counts = [a + b for a, b in zip(self._counts, counts)] if self._counts else counts
        if padding is not None:
            self._padded += int(padding[0])
            self._issued += int(padding[1])
        if self.path:
            line = {
                "format": EXIT_HISTOGRAM_FORMAT,
                "iteration": int(iteration),
                "R_max": len(counts),
                "counts": counts,
                "samples": int(samples),
                "padding_fraction": _padding_fraction(padding),
                "phase": dict(phase),
            }
            with open(self.path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(line) + "\n")
                stream.flush()

    def interval(self) -> list:
        """The element-wise sum of the interval's histograms ([] before its first step)."""
        return list(self._counts)

    def interval_padding(self) -> tuple:
        """``(padded, issued)`` tokens of this EU over the interval."""
        return self._padded, self._issued

    def reset_interval(self) -> None:
        """Start a new log interval."""
        self._counts = []
        self._padded = 0
        self._issued = 0


def get_loop_exit_histogram_log(args) -> LoopExitHistogramLog:
    """The process's exit-histogram log, created on first use (and again for another
    ``--loop-exit-histogram-path``); only the last rank writes the file."""
    path = getattr(args, "loop_exit_histogram_path", None)
    log = _EXIT_HISTOGRAM.log
    if log is None or log.requested_path != path:
        log = LoopExitHistogramLog(path if is_last_rank() else None)
        log.requested_path = path
        _EXIT_HISTOGRAM.log = log
    return log


def reset_loop_exit_histogram_log() -> None:
    """Forget the process's exit-histogram log (tests; a new run in the same process)."""
    _EXIT_HISTOGRAM.log = None


def loop_exit_histogram_phase(args, samples: int, *, waves: int) -> dict:
    """The phase geometry of a JSON line (the planner's ``PhaseGeometry`` inputs):
    ``samples_per_eu`` = the step's samples over the data-parallel ranks (the EUs)."""
    cp = getattr(args, "context_parallel_size", 1)
    gpus = getattr(args, "world_size", 1)
    eus = getattr(args, "data_parallel_size", None) or max(gpus // cp, 1)
    return {
        "seq_length": int(args.seq_length),
        "cp": int(cp),
        "gpus": int(gpus),
        "ep": int(getattr(args, "expert_model_parallel_size", 1)),
        "samples_per_eu": int(samples // eus),
        "waves": int(waves),
    }


def record_loop_exit_histogram(args, schedule, model, iteration: int) -> Optional[list]:
    """The post-``train_step`` hook of an RSBF run: records the step's exit histogram (see the
    module docstring) under ``iteration`` (the number ``training_log`` prints for the step).
    Returns the counts, or None when the schedule ran no step for ``model``. Raises when the
    counts do not sum to the step's samples (``get_current_running_global_batch_size()``: the
    batch the step ran, under a batch-size schedule or ``--decrease-batch-size-if-needed``
    too), an executor bug."""
    model = model[0] if isinstance(model, (list, tuple)) else model
    counts = schedule.exit_counts(model)
    if counts is None:
        return None
    samples = get_current_running_global_batch_size()
    if sum(counts) != samples:
        raise RuntimeError(
            f"loop exit histogram {counts} sums to {sum(counts)} samples, the step ran "
            f"{samples}"
        )
    padding = schedule.padding(model)
    log = get_loop_exit_histogram_log(args)
    executor = schedule.cached_executor(model)
    phase = loop_exit_histogram_phase(args, samples, waves=executor.waves)
    log.record(iteration, counts, samples, padding, phase)
    args._loop_exit_counts = counts
    args._loop_padding = None if padding is None else list(padding)
    args._loop_exit_counts_interval = log.interval()
    return counts


def loop_exit_histogram_scalars(counts: Sequence[int], padding: Optional[Sequence[int]]) -> dict:
    """The TensorBoard / W&B values of one step: ``loop exit depth count/d{d}``,
    ``loop exit depth fraction/d{d}``, ``loop exit depth mean`` and ``loop padding fraction``
    (the logging rank's EU)."""
    samples = sum(counts)
    scalars = {f"loop exit depth count/d{d}": int(c) for d, c in enumerate(counts, 1)}
    scalars.update({f"loop exit depth fraction/d{d}": c / samples for d, c in enumerate(counts, 1)})
    scalars["loop exit depth mean"] = _mean_depth(counts)
    scalars["loop padding fraction"] = _padding_fraction(padding)
    return scalars


def log_loop_exit_histogram(args, iteration: int, writer, wandb_writer) -> None:
    """Write this step's exit histogram (``args._loop_exit_counts``) to the TensorBoard and W&B
    writers that exist (``training_log`` calls it at the TensorBoard interval)."""
    counts = getattr(args, "_loop_exit_counts", None)
    if not counts:
        return
    scalars = loop_exit_histogram_scalars(counts, getattr(args, "_loop_padding", None))
    if writer:
        for name, value in scalars.items():
            writer.add_scalar(name, value, iteration)
    if wandb_writer:
        wandb_writer.log(scalars, iteration)


def loop_exit_histogram_log_string(args, *, reset: bool) -> str:
    """The console fragment of the log interval's exit histograms, e.g.
    ``' loop exit depths 1..4: 130 / 128 / 120 / 134 (mean 2.51, 512 samples) | loop padding
    fraction (logging rank's EU): 0.0310 |'``; '' without a recorded step. ``reset`` starts a
    new interval (``training_log``'s ``should_reset``)."""
    log = _EXIT_HISTOGRAM.log
    counts = [] if log is None else log.interval()
    if not counts:
        return ""
    text = (
        f" loop exit depths 1..{len(counts)}: {' / '.join(str(c) for c in counts)} "
        f"(mean {_mean_depth(counts):.2f}, {sum(counts)} samples) |"
        f" loop padding fraction (logging rank's EU): "
        f"{_padding_fraction(log.interval_padding()):.4f} |"
    )
    if reset:
        log.reset_interval()
        args._loop_exit_counts_interval = log.interval()
    return text


# ----------------------------------------------------------------------------- the planner
# (config field, CLI flag) the planner sets in apply mode.
LOOP_PLANNER_OWNED_FLAGS = (
    ("loop_rsbf_item_form", "--loop-rsbf-item-form"),
    ("loop_rsbf_block_stride", "--loop-rsbf-block-stride"),
    ("loop_rsbf_waves", "--loop-rsbf-waves"),
    ("loop_rsbf_coda_recompute_mla_up_proj", "--loop-rsbf-coda-recompute-mla-up-proj"),
)

# --loop-planner-depths settings other than 'config' / 'histogram'.
LOOP_PLANNER_FIXED_DEPTHS = {"r4_geom": R4_GEOM, "r4_unif": R4_UNIF, "r4_skew": R4_SKEW}


def _owned_flag_default(name: str) -> Any:
    # The CLI flags are generated from the TransformerConfig fields: their parser defaults are
    # the dataclass defaults.
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig.__dataclass_fields__[name].default


def loop_planner_user_flags(args) -> List[str]:
    """The planner-owned flags whose value differs from the parser default (``--flag=value``):
    set by the user, which ``--loop-planner apply`` refuses."""
    return [
        f"{flag}={getattr(args, name)!r}"
        for name, flag in LOOP_PLANNER_OWNED_FLAGS
        if getattr(args, name, _owned_flag_default(name)) != _owned_flag_default(name)
    ]


def loop_planner_hardware(args) -> PlannerHardware:
    """The ``--loop-planner-hardware`` preset with the HBM cap (``planner_reserve = HBM -
    cap``), Grace capacity and margin, HybridEP reserve and C2C overrides (GB, GB/s; None keeps
    the preset's)."""
    name = getattr(args, "loop_planner_hardware", "gb200")
    base = PlannerHardware.preset(name)
    overrides = {}
    cap = getattr(args, "loop_planner_hbm_cap_gb", None)
    if cap is not None:
        overrides["planner_reserve_bytes"] = base.hbm_bytes - cap * GB
    for flag, field_name, scale in (
        ("loop_planner_grace_gb", "grace_bytes", GB),
        ("loop_planner_grace_margin_gb", "grace_margin_bytes", GB),
        ("loop_planner_ep_reserve_gb", "ep_reserve_bytes", GB),
        ("loop_planner_c2c_gbps", "c2c_bytes_per_s", GB),
        ("loop_planner_c2c_margin", "c2c_margin", 1.0),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            overrides[field_name] = value * scale
    return PlannerHardware.preset(name, **overrides)


def loop_planner_phase(args) -> PhaseGeometry:
    """The phase of this run: ``S``, cp, GPUs (``world_size``: RSBF runs TP = PP = 1), EP,
    ``n`` = micro batch size x microbatches per EU, R_max; the lockstep unit is every EU of the
    domain (the executor's); eager cells, raw EP exposure (two-chunk overlap is M10); MXFP8
    only with ``--fp8-recipe mxfp8``. Waves stay 1 under a changing batch size."""
    cp = getattr(args, "context_parallel_size", 1)
    gpus = args.world_size
    eus = getattr(args, "data_parallel_size", None) or max(gpus // cp, 1)
    mbs = args.micro_batch_size
    # The microbatch calculator's divisor (as _validate_loop_rsbf_waves).
    per_step = mbs * eus * getattr(args, "gtp_weight_remat_size", 1)
    microbatches = args.global_batch_size // per_step
    mxfp8 = getattr(args, "fp8", None) is not None and getattr(args, "fp8_recipe", None) == "mxfp8"
    changing_batch = (
        getattr(args, "step_batch_size_schedule", None) is not None
        or getattr(args, "decrease_batch_size_if_needed", False)
        or getattr(args, "rampup_batch_size", None) is not None
    )
    return PhaseGeometry(
        seq_length=args.seq_length,
        gpus=gpus,
        ep=getattr(args, "expert_model_parallel_size", 1),
        samples_per_eu=mbs * microbatches,
        max_rounds=args.loop_iterations,
        cp=cp,
        lockstep_eus=max(gpus // cp, 1),
        microbatches_per_step=microbatches,
        precision="mxfp8" if mxfp8 else "bf16",
        graphs="eager",
        ep_overlap="raw",
        waves_allowed=not changing_batch,
    )


def loop_planner_depths(args, config: Any = None) -> DepthDistribution:
    """``--loop-planner-depths``: ``config`` (the exit criterion's distribution, from
    ``config`` or ``args``), ``histogram`` (``--loop-planner-histogram-path``, all its
    iterations) or an R4 setting (R_max must be 4)."""
    source = args if config is None else config
    mode = getattr(args, "loop_planner_depths", "config")
    max_rounds = source.loop_iterations
    if mode == "config":
        return DepthDistribution.from_config(source)
    if mode == "histogram":
        path = getattr(args, "loop_planner_histogram_path", None)
        if path is None:
            raise ValueError("--loop-planner-depths histogram needs --loop-planner-histogram-path")
        return DepthDistribution.from_histogram_file(path, max_rounds=max_rounds)
    if mode not in LOOP_PLANNER_FIXED_DEPTHS:
        raise ValueError(f"unknown --loop-planner-depths {mode!r}")
    depth = LOOP_PLANNER_FIXED_DEPTHS[mode]
    if depth.max_rounds != max_rounds:
        raise ValueError(
            f"--loop-planner-depths {mode} has R_max {depth.max_rounds}, the run has "
            f"loop_iterations {max_rounds}"
        )
    return depth


def loop_optimizer_offload_blockers(args, hardware: PlannerHardware) -> List[str]:
    """Why Adam to Grace (``--optimizer-cpu-offload``) cannot run here ([] when it can)."""
    blockers = []
    if hardware.grace_bytes <= 0:
        blockers.append(f"no Grace tier ({hardware.name}; --loop-planner-grace-gb names one)")
    if not getattr(args, "use_precision_aware_optimizer", False):
        blockers.append("--use-precision-aware-optimizer not set")
    if getattr(args, "optimizer", "adam") != "adam":
        blockers.append(f"--optimizer {getattr(args, 'optimizer', None)} (the planner models Adam)")
    if not getattr(args, "decoupled_weight_decay", True):
        blockers.append("CPU offload needs AdamW (decoupled weight decay)")
    if getattr(args, "chunked_optimizer_state_offload", False) or getattr(
        args, "offload_optimizer_states", False
    ):
        blockers.append("--chunked-optimizer-state-offload excludes optimizer CPU offload")
    if getattr(args, "fp8_param_gather", False) and getattr(args, "fp8_recipe", None) != "delayed":
        blockers.append("--fp8-param-gather with CPU offload needs --fp8-recipe delayed")
    return blockers


def _hybrid_of(module: Any) -> Any:
    return get_attr_wrapped_model(module, "head_loss", return_model_obj=True)


def loop_planner_shape(args, module: Any) -> ModelShape:
    """``ModelShape.from_model`` of the built model (one chunk, possibly wrapped): the padded
    vocabulary of the model, this run's EP, ``--loop-planner-act-bytes-per-token``, the
    distributed optimizer, the main-grad bytes, and the cost model's fused cross-entropy term
    with ``--cross-entropy-fusion-impl linear`` (no logits materialized)."""
    hybrid = _hybrid_of(module)
    config = hybrid.config
    vocab_size = getattr(hybrid, "vocab_size", None) or args.padded_vocab_size
    overrides = dict(
        distributed_optimizer=bool(getattr(args, "use_distributed_optimizer", False)),
        expert_grad_bytes=4 if getattr(args, "accumulate_allreduce_grads_in_fp32", True) else 2,
    )
    if (
        getattr(config, "cross_entropy_loss_fusion", False)
        and getattr(config, "cross_entropy_fusion_impl", "native") == "linear"
    ):
        overrides["ce_bytes_per_token"] = None
    return ModelShape.from_model(
        module,
        vocab_size=vocab_size,
        expert_parallel_size=getattr(args, "expert_model_parallel_size", 1),
        act_bytes_per_token=getattr(args, "loop_planner_act_bytes_per_token", None),
        **overrides,
    )


def _coda_recompute_available(module: Any) -> bool:
    """D42 has something to recompute: a ``DSv4HybridAttention`` in the coda sublayers or the
    MTP layers (``LoopProgram``'s rule)."""
    hybrid = _hybrid_of(module)
    if getattr(hybrid.config, "experimental_attention_variant", None) != "dsv4_hybrid":
        return False
    from megatron.core.transformer.loop.program import _coda_dsv4_attentions

    return bool(_coda_dsv4_attentions(hybrid, hybrid.decoder))


def loop_planner_capabilities(
    args,
    module: Any = None,
    *,
    hardware: Optional[PlannerHardware] = None,
    shape: Optional[ModelShape] = None,
) -> PlannerCapabilities:
    """What this run can apply: the resident layout only (no Layer window, no Grace tier for
    round-state data: M3 / M6a); D42 when the model (else ``shape``) has a DSv4 attention in
    the coda or the MTP layers; Adam to Grace per ``--loop-planner-optimizer-offload``
    (``auto``: when ``loop_optimizer_offload_blockers`` finds nothing, else excluded with those
    reasons; ``never``; ``always``: the menu is restricted to options with it)."""
    hardware = loop_planner_hardware(args) if hardware is None else hardware
    mode = getattr(args, "loop_planner_optimizer_offload", "auto")
    if mode == "never":
        offload, reason = False, "--loop-planner-optimizer-offload never"
    elif mode == "always":
        offload, reason = True, ""
    else:
        blockers = loop_optimizer_offload_blockers(args, hardware)
        offload, reason = not blockers, "; ".join(blockers)
    if module is not None:
        coda = _coda_recompute_available(module)
    else:
        coda = shape is not None and shape.has_coda_dsv4_attention
    return PlannerCapabilities(
        layer_window=False,
        grace_state_tier=False,
        optimizer_offload=offload,
        coda_recompute=coda,
        optimizer_offload_reason=reason,
    )


def make_loop_planner(
    args, shape: ModelShape, hardware: PlannerHardware, capabilities: PlannerCapabilities
) -> LoopRSBFPlanner:
    """The runtime planner: the CLI's strides / waves / form policy, the executor's schedule
    policy and nested rebuild; ``--loop-planner-optimizer-offload always`` keeps only the move
    steps with Adam to Grace."""
    always = getattr(args, "loop_planner_optimizer_offload", "auto") == "always"
    return LoopRSBFPlanner(
        shape,
        hardware,
        capabilities=capabilities,
        strides_layers=tuple(getattr(args, "loop_planner_strides", None) or (1, 2, 3, 4, 6, 9, 18)),
        waves=tuple(getattr(args, "loop_planner_waves", None) or (1, 2, 4, 8)),
        move_steps=("A", "AE", "AEM") if always else ("-", "A", "AE", "AEM"),
        form_policy=getattr(args, "loop_planner_forms", "d28"),
        small_core_layers=getattr(args, "loop_planner_small_core_layers", 4),
        sched_policy="executor",
        rebuild_fraction=executor_nested_rebuild_fraction(shape.core_layers),
    )


def loop_planner_current_option(
    plan: LoopRSBFPlan, config: Any, *, optimizer_offload: bool
) -> tuple:
    """``(result, message)``: the decision-table row of the config's current (item form,
    stride, waves, D42) with this run's optimizer offload in the resident layout (None when the
    planner's menu has no such option), and a one-line verdict for the log."""
    form = getattr(config, "loop_rsbf_item_form", "whole_round")
    stride = getattr(config, "loop_rsbf_block_stride", None)
    waves = getattr(config, "loop_rsbf_waves", 1)
    coda = bool(getattr(config, "loop_rsbf_coda_recompute_mla_up_proj", False))
    moves = "A" if optimizer_offload else "-"
    label = (
        f"item form {form} stride {stride} waves {waves} D42 {'on' if coda else 'off'} "
        f"moves {moves}"
    )
    match = None
    for result in plan.table:
        o = result.option
        if (
            option_item_form(o, plan.core_layers) == (form, stride)
            and o.waves == waves
            and o.coda_recompute == coda
            and o.moves == moves
            and o.layout == "resident"
        ):
            match = result
            break
    if match is None:
        return None, (
            f"loop planner (report): the current option ({label}) is not in the planner's menu "
            "(block strides are whole attention + FFN pairs, nested stride 2, waves from "
            "--loop-planner-waves dividing the microbatches)"
        )
    if match.fits and match.applicable:
        verdict = (
            f"fits (HBM {match.hbm_bytes / GB:.2f} GB, modelled step {match.step_ms / 1e3:.3f} s,"
            f" MFU {100 * match.mfu:.2f} %); the planner would apply {plan.item_form} stride "
            f"{plan.block_stride} waves {plan.waves}"
        )
    else:
        verdict = "does not fit: " + "; ".join(match.refusals or ("refused",))
    return match, f"loop planner (report): the current option ({label}) {verdict}"


def _check_plan_digest(digest: str) -> None:
    """Every rank of the world holds the same plan (one MAX all-reduce of the 32-byte digest
    and its negation); raises on every rank otherwise."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    if torch.distributed.get_world_size() == 1:
        return
    if "nccl" in str(torch.distributed.get_backend()):
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    local = torch.tensor(list(bytes.fromhex(digest)), dtype=torch.int32, device=device)
    both = torch.cat([local, -local])
    torch.distributed.all_reduce(both, op=torch.distributed.ReduceOp.MAX)
    if not torch.equal(both[: local.numel()], -both[local.numel() :]):
        raise RuntimeError(
            "loop planner: the ranks computed different plans (plan digests differ); every "
            "rank must see the same arguments, model shape and depth distribution"
        )


def _log_plan(args, plan: LoopRSBFPlan) -> None:
    """Rank 0: the decision table (INFO, printed when INFO is off), the plan warnings, the
    ``--loop-planner-report-path`` JSON."""
    if not is_rank0():
        return
    text = format_decision_table(plan)
    if logger.isEnabledFor(logging.INFO):
        logger.info("%s", text)
    else:
        print_rank_0(text)
    for warning in plan.warnings:
        warn_rank_0(f"loop planner: {warning}")
    path = getattr(args, "loop_planner_report_path", None)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(plan.to_json(), stream, indent=1)


class LoopPlannerSession:
    """The planner of one training process: the last phase and plan. ``on_phase_start`` runs at
    every phase start (startup and every resume today; a phase switch at M11)."""

    def __init__(self):
        self.phase: Optional[PhaseGeometry] = None
        self.plan: Optional[LoopRSBFPlan] = None
        self.applied = False

    def on_phase_start(self, args, model, *, apply: bool) -> Optional[LoopRSBFPlan]:
        """Plan the phase of ``args`` for ``model`` (a list of one model chunk, or the chunk),
        log the decision table and check the digest on every rank; ``apply`` sets the plan on
        the model config and ``args`` (module docstring), else the current option is checked.
        Returns the plan (None when nothing fits in report mode; apply raises
        ``PlannerNoFit``)."""
        models = list(model) if isinstance(model, (list, tuple)) else [model]
        if len(models) != 1:
            raise ValueError(
                "the loop planner needs one model chunk (RSBF runs PP = 1 without virtual "
                f"pipeline stages), got {len(models)}"
            )
        module = models[0]
        rerun = self.applied
        if apply and not rerun:
            user = loop_planner_user_flags(args)
            if user:
                raise ValueError(
                    "--loop-planner apply sets the RSBF item form, block stride, waves and D42; "
                    f"set by the user: {', '.join(user)}. Drop them, or use --loop-planner "
                    "report to check them"
                )
        config = get_model_config(module)
        hardware = loop_planner_hardware(args)
        shape = loop_planner_shape(args, module)
        capabilities = loop_planner_capabilities(args, module, hardware=hardware, shape=shape)
        phase = loop_planner_phase(args)
        depth = loop_planner_depths(args, config)
        planner = make_loop_planner(args, shape, hardware, capabilities)
        try:
            plan = planner.plan(phase, depth)
        except PlannerNoFit as error:
            if apply:
                raise
            warn_rank_0(f"loop planner (report): {error}")
            return None
        _log_plan(args, plan)
        _check_plan_digest(plan.digest)
        if apply:
            self._apply(args, module, config, plan, rerun=rerun)
            self.applied = True
        else:
            current, message = loop_planner_current_option(
                plan, config, optimizer_offload=bool(getattr(args, "optimizer_cpu_offload", False))
            )
            if current is not None and current.fits and current.applicable:
                print_rank_0(message)
            else:
                warn_rank_0(message)
        self.phase, self.plan = phase, plan
        return plan

    @staticmethod
    def _apply(args, module: Any, config: Any, plan: LoopRSBFPlan, *, rerun: bool) -> None:
        # Lazy: the executor and the argument module import training-side code.
        from megatron.core.transformer.loop.guards import rsbf_item_form_violations
        from megatron.core.transformer.loop.rsbf_executor import get_loop_rsbf_schedule
        from megatron.training.arguments import _validate_loop_rsbf_waves

        schedule = get_loop_rsbf_schedule()
        if schedule.cached_executor(module) is not None and not rerun:
            raise RuntimeError(
                "loop planner: an RSBF executor is already built for this model; the plan must "
                "be applied before the first training step"
            )
        offload = bool(getattr(args, "optimizer_cpu_offload", False))
        if rerun and plan.optimizer_offload != offload:
            raise PlannerUnsupported(
                "loop planner: the new phase plans Adam to Grace "
                f"{'on' if plan.optimizer_offload else 'off'}, the optimizer was built with it "
                f"{'on' if offload else 'off'}; pin it with --loop-planner-optimizer-offload "
                f"{'always' if offload else 'never'}"
            )
        stack = _hybrid_of(module).decoder
        violations = rsbf_item_form_violations(
            plan.item_form, plan.block_stride, stack.loop_roles.core
        )
        if violations:
            raise RuntimeError(
                f"loop planner: the plan's item form {plan.item_form} stride "
                f"{plan.block_stride} breaks {violations} on the built core of "
                f"{stack.loop_roles.core} sublayers (a planner bug)"
            )
        overrides = apply_plan_to_config(config, plan)
        if stack.config is not config:
            apply_plan_to_config(stack.config, plan)
        for name, value in overrides.items():
            setattr(args, name, value)
        _validate_loop_rsbf_waves(args)
        if plan.optimizer_offload and not offload:
            # Adam to Grace: the hybrid device optimizer with every optimizer state on the host.
            args.optimizer_cpu_offload = True
            args.optimizer_offload_fraction = 1.0
        if rerun:
            # The program and executor were built from the previous phase's config.
            schedule.invalidate(module)
        print_rank_0(
            "loop planner: applied "
            + " ".join(f"{name}={value}" for name, value in overrides.items())
            + f" optimizer_cpu_offload={bool(getattr(args, 'optimizer_cpu_offload', False))}"
        )


# The process's planner session (one training run per process; tests reset it).
_PLANNER = SimpleNamespace(session=None)


def get_loop_planner_session() -> LoopPlannerSession:
    """The process's planner session, created on first use."""
    if _PLANNER.session is None:
        _PLANNER.session = LoopPlannerSession()
    return _PLANNER.session


def reset_loop_planner_session() -> None:
    """Forget the process's planner session (tests; a new run in the same process)."""
    _PLANNER.session = None


def plan_loop_rsbf(args, model, *, apply: bool) -> Optional[LoopRSBFPlan]:
    """The ``setup_model_and_optimizer`` hook: ``get_loop_planner_session().on_phase_start``."""
    return get_loop_planner_session().on_phase_start(args, model, apply=apply)


def check_resume_offload(args, checkpoint_args) -> None:
    """A resume under ``--loop-planner apply`` with ``--loop-planner-optimizer-offload auto``
    (the planner decides Adam to Grace) must keep the checkpoint's ``optimizer_cpu_offload``:
    the optimizer state layouts differ. Skipped when the optimizer state is not loaded
    (``--no-load-optim``, finetuning) or the checkpoint does not record it."""
    if getattr(args, "loop_planner", "off") != "apply":
        return
    if getattr(args, "loop_planner_optimizer_offload", "auto") != "auto":
        return
    if getattr(args, "no_load_optim", False) or getattr(args, "finetune", False):
        return
    if checkpoint_args is None or not hasattr(checkpoint_args, "optimizer_cpu_offload"):
        return
    planned = bool(getattr(args, "optimizer_cpu_offload", False))
    saved = bool(checkpoint_args.optimizer_cpu_offload)
    if planned != saved:
        raise RuntimeError(
            f"loop planner: this run plans Adam to Grace {'on' if planned else 'off'} "
            f"(optimizer_cpu_offload={planned}), the checkpoint was saved with "
            f"optimizer_cpu_offload={saved}; the optimizer state layouts differ. Pin it to the "
            f"checkpoint's with --loop-planner-optimizer-offload {'always' if saved else 'never'}"
        )
