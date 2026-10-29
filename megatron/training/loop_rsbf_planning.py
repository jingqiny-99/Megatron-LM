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
  is built. With the loop parameter store (``--loop-param-store-shard-size`` > 1) the dense
  layout and the moves E (prelude / coda / MTP dense groups to the Layer window) and M
  (embedding and head to Resident + sharded grads) are applicable: the plan's store modes are
  set on the store (``apply_plan_to_store``), the move steps gain ``E`` / ``EM`` without Adam to
  Grace (H100 has no Grace tier), the dense shard axis is the store's ``z`` and the memory
  model takes the store's runtime terms per option (``loop_planner_with_store_terms``).
  Without a store those moves are reported as requested but unavailable.
  ``check_resume_offload`` refuses a resume whose planned optimizer offload differs from the
  checkpoint's (the optimizer state layout differs).
* ``report``: logs the plan and whether the current config's option fits; changes nothing.

``LoopPlannerSession.on_phase_start`` is the entry of every phase start (today: startup and
every resume). A later phase switch (M11) calls it again: the plan is re-applied and the
model's cached executor dropped (``LoopRSBFSchedule.invalidate``) so the next step rebuilds it.
The session lives in this module, not on ``args`` (checkpoints pickle ``args``).

**Resume (M3c).** An applied plan is recorded in the checkpoint's ``loop_resume`` common-state
entry (``loop_plan_record``: the choice, the store modes, the depth distribution, the phase, the
hardware preset, the digest). The planner runs before the checkpoint is loaded, so under
``apply`` with ``--load`` (not finetuning) the session first reads that record
(``read_saved_plan``) and **pins** the saved choice (``pin_saved_plan``) when this run's menu
holds it and the planner's own filter accepts it at this run's sizes (it fits, it is applicable,
the form policy allows it); otherwise it plans afresh and ``check_resume_plan`` (called by the
resume check in ``megatron/training/loop_checkpoint.py``) reports the difference, which
``--loop-resume-changes refuse`` (the default) turns into an error.
"""

import dataclasses
import hashlib
import json
import logging
import math
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from megatron.core.num_microbatches_calculator import get_current_running_global_batch_size
from megatron.core.transformer.loop.param_store import ResidencyMode, find_param_store
from megatron.core.transformer.loop.planner import (
    EXIT_HISTOGRAM_FORMAT,
    MOVE_STEPS,
    R4_GEOM,
    R4_SKEW,
    R4_UNIF,
    STORE_MOVE_STEPS,
    DepthDistribution,
    LoopRSBFPlan,
    LoopRSBFPlanner,
    PhaseGeometry,
    PlannerCapabilities,
    PlannerHardware,
    PlannerNoFit,
    PlannerUnsupported,
    apply_plan_to_config,
    apply_plan_to_store,
    executor_nested_rebuild_fraction,
    format_decision_table,
    option_item_form,
    store_planner_variants,
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
    "check_resume_plan",
    "get_loop_exit_histogram_log",
    "get_loop_planner_session",
    "log_loop_exit_histogram",
    "loop_exit_histogram_log_string",
    "loop_exit_histogram_phase",
    "loop_exit_histogram_scalars",
    "loop_optimizer_offload_blockers",
    "loop_planner_auto_core_mode",
    "loop_planner_auto_core_options",
    "loop_planner_capabilities",
    "loop_planner_current_option",
    "loop_planner_depths",
    "loop_planner_hardware",
    "loop_planner_move_steps",
    "loop_planner_phase",
    "loop_planner_shape",
    "loop_planner_store_option",
    "loop_planner_user_flags",
    "loop_planner_with_store_terms",
    "loop_plan_choice",
    "loop_plan_record",
    "make_loop_planner",
    "pin_saved_plan",
    "plan_loop_rsbf",
    "read_saved_plan",
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
    store_z = int(getattr(args, "loop_param_store_shard_size", 1) or 1)
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
        shard_axis=store_z if store_z > 1 else None,
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
    """What this run can apply: the Layer window (core and edge dense groups, moves E / M,
    re-gathered blocks) when the model holds a loop parameter store (else, without a model,
    when ``--loop-param-store-shard-size`` > 1), the resident layout only otherwise; no Grace
    tier for round-state data (M6a); D42 when the model (else ``shape``) has a DSv4 attention
    in the coda or the MTP layers; Adam to Grace per ``--loop-planner-optimizer-offload``
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
        layer_window = find_param_store(module) is not None
    else:
        coda = shape is not None and shape.has_coda_dsv4_attention
        layer_window = int(getattr(args, "loop_param_store_shard_size", 1) or 1) > 1
    return PlannerCapabilities(
        layer_window=layer_window,
        grace_state_tier=False,
        optimizer_offload=offload,
        coda_recompute=coda,
        optimizer_offload_reason=reason,
    )


def loop_planner_move_steps(
    args, hardware: PlannerHardware, capabilities: PlannerCapabilities
) -> tuple:
    """The move steps of the runtime menu. Without a parameter store: the cost model's
    cumulative ``-, A, AE, AEM`` (``--loop-planner-optimizer-offload always``: the A steps).
    With a store (``capabilities.layer_window``): ``-, A, E, EM, AE, AEM`` (E and M reachable
    without Adam to Grace, D-S6-17), minus the A steps when Adam to Grace cannot run here
    (``loop_optimizer_offload_blockers``), minus ``-`` / ``E`` / ``EM`` under ``always``."""
    always = getattr(args, "loop_planner_optimizer_offload", "auto") == "always"
    if not capabilities.layer_window:
        return tuple(s for s in MOVE_STEPS if "A" in s) if always else MOVE_STEPS
    steps = list(STORE_MOVE_STEPS)
    if always:
        steps = [s for s in steps if "A" in s]
    elif loop_optimizer_offload_blockers(args, hardware):
        steps = [s for s in steps if "A" not in s]
    return tuple(steps)


def loop_planner_with_store_terms(
    shape: ModelShape, planner: LoopRSBFPlanner, phase: PhaseGeometry, module: Any
) -> ModelShape:
    """``shape`` with the parameter store's runtime memory terms of every option of
    ``planner``'s menu (``LoopParamStore.planner_terms``; one host simulation per distinct
    modes / item form), or ``shape`` unchanged without a store."""
    store = find_param_store(module)
    if store is None:
        return shape
    variants = store_planner_variants(planner.options(phase), shape.core_layers)
    return dataclasses.replace(shape, store_terms=store.planner_terms(variants))


def loop_planner_store_option(modes: Any) -> Optional[tuple]:
    """``(layout, moves without A, regather)`` of the planner option whose store modes are
    ``modes`` (``option_store_modes``' inverse), or None when no option maps to them (a group
    in a mode the planner does not offer, e.g. edges in Resident + sharded grads)."""
    lw, rsg, res = (
        ResidencyMode.LAYER_WINDOW,
        ResidencyMode.RESIDENT_SHARDED_GRADS,
        ResidencyMode.RESIDENT,
    )
    edges = {modes.prelude, modes.coda, modes.mtp}
    if len(edges) != 1 or modes.core not in (lw, res) or modes.edge_regather:
        return None
    edge = edges.pop()
    if edge not in (lw, res) or modes.emb_head not in (rsg, res):
        return None
    moves = ("E" if edge is lw else "") + ("M" if modes.emb_head is rsg else "")
    if moves == "M":
        return None  # M is offered only with E
    layout = "layer_window" if modes.core is lw else "resident"
    if layout == "resident" and (moves or modes.core_regather):
        return None
    return layout, moves, bool(modes.core_regather)


def make_loop_planner(
    args, shape: ModelShape, hardware: PlannerHardware, capabilities: PlannerCapabilities
) -> LoopRSBFPlanner:
    """The runtime planner: the CLI's strides / waves / form policy, the executor's schedule
    policy and nested rebuild, and the move steps (``loop_planner_move_steps``)."""
    return LoopRSBFPlanner(
        shape,
        hardware,
        capabilities=capabilities,
        strides_layers=tuple(getattr(args, "loop_planner_strides", None) or (1, 2, 3, 4, 6, 9, 18)),
        waves=tuple(getattr(args, "loop_planner_waves", None) or (1, 2, 4, 8)),
        move_steps=loop_planner_move_steps(args, hardware, capabilities),
        form_policy=getattr(args, "loop_planner_forms", "d28"),
        small_core_layers=getattr(args, "loop_planner_small_core_layers", 4),
        sched_policy="executor",
        rebuild_fraction=executor_nested_rebuild_fraction(shape.core_layers),
    )


def loop_planner_current_option(
    plan: LoopRSBFPlan, config: Any, *, optimizer_offload: bool, store_modes: Any = None
) -> tuple:
    """``(result, message)``: the decision-table row of the config's current (item form,
    stride, waves, D42) with this run's optimizer offload, in the resident layout or, with a
    parameter store, in the layout / moves / re-gather of its configured ``store_modes``
    (None when the planner's menu has no such option), and a one-line verdict for the log."""
    form = getattr(config, "loop_rsbf_item_form", "whole_round")
    stride = getattr(config, "loop_rsbf_block_stride", None)
    waves = getattr(config, "loop_rsbf_waves", 1)
    coda = bool(getattr(config, "loop_rsbf_coda_recompute_mla_up_proj", False))
    layout, dense_moves, regather = "resident", "", False
    if store_modes is not None:
        mapped = loop_planner_store_option(store_modes)
        if mapped is None:
            return None, (
                f"loop planner (report): the store modes {store_modes.describe()} are not a "
                "planner option (edges and the core in Resident or the Layer window, the "
                "embedding / head Resident or with sharded grads only with the edges windowed)"
            )
        layout, dense_moves, regather = mapped
    moves = ("A" if optimizer_offload else "") + dense_moves or "-"
    label = (
        f"item form {form} stride {stride} waves {waves} D42 {'on' if coda else 'off'} "
        f"moves {moves}"
    )
    if store_modes is not None:
        label += f" layout {layout}" + (" re-gathered" if regather else "")
    match = None
    for result in plan.table:
        o = result.option
        if (
            option_item_form(o, plan.core_layers) == (form, stride)
            and o.waves == waves
            and o.coda_recompute == coda
            and o.moves == moves
            and o.layout == layout
            and bool(o.regather) == regather
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


def loop_planner_auto_core_options(args, model: Any) -> tuple:
    """``(resident, layer_window, hardware)``: the planner's evaluation (``OptionResult``, None
    when the menu has no such option) of the run's own item form, stride, waves, D42 and Adam
    to Grace with the core dense group Resident and in the Layer window (edges and the
    embedding / head Resident, no re-gather), under ``--loop-planner-hardware`` and its caps,
    with the parameter store's runtime terms. Raises ``PlannerUnsupported`` / ``ValueError``
    when the planner cannot model the run."""
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    module = models[0]
    config = get_model_config(module)
    form = getattr(config, "loop_rsbf_item_form", "whole_round")
    stride = getattr(config, "loop_rsbf_block_stride", None)
    waves = getattr(config, "loop_rsbf_waves", 1)
    coda = bool(getattr(config, "loop_rsbf_coda_recompute_mla_up_proj", False))
    offload = bool(getattr(args, "optimizer_cpu_offload", False))
    hardware = loop_planner_hardware(args)
    shape = loop_planner_shape(args, module)
    core_layers = shape.core_layers
    if form == "block":
        if stride is None or stride % 2:
            raise PlannerUnsupported(f"block stride {stride} is not whole layers")
        strides = (stride // 2,)
    else:
        strides = (core_layers,)
    capabilities = loop_planner_capabilities(args, module, hardware=hardware, shape=shape)
    phase = loop_planner_phase(args)
    depth = loop_planner_depths(args, config)

    def planner_of(shape_):
        return LoopRSBFPlanner(
            shape_,
            hardware,
            capabilities=capabilities,
            strides_layers=strides,
            waves=(waves,),
            move_steps=("A",) if offload else ("-",),
            form_policy="planner",
            sched_policy="executor",
            rebuild_fraction=executor_nested_rebuild_fraction(core_layers),
        )

    planner = planner_of(shape)
    planner = planner_of(loop_planner_with_store_terms(shape, planner, phase, module))
    results = planner.evaluate(phase, depth)

    def pick(layout):
        for result in results:
            o = result.option
            if (
                option_item_form(o, core_layers) == (form, stride)
                and o.waves == waves
                and o.coda_recompute == coda
                and o.layout == layout
                and not o.regather
            ):
                return result
        return None

    return pick("resident"), pick("layer_window"), hardware


def loop_planner_auto_core_mode(args, model: Any) -> tuple:
    """``(core mode, reason)`` of ``--loop-param-store-modes auto``
    (``loop_planner_auto_core_options``): core Layer window when the Resident option does not
    fit, else Resident; Resident with the reason when the planner cannot model the run."""
    resident, window = ResidencyMode.RESIDENT, ResidencyMode.LAYER_WINDOW
    try:
        res, lw, hardware = loop_planner_auto_core_options(args, model)
    except (PlannerUnsupported, PlannerNoFit, ValueError) as error:
        return resident, f"core Resident (the planner cannot model this run: {error})"
    cap = f"HBM cap {hardware.hbm_cap_bytes / GB:.1f} GB on {hardware.name}"
    if res is None:
        return resident, f"core Resident (the run's option is not in the planner's menu, {cap})"
    if res.fits:
        return resident, f"core Resident fits ({res.hbm_bytes / GB:.2f} GB, {cap})"
    why = "; ".join(res.refusals) or "does not fit"
    if lw is not None and lw.fits:
        return window, (
            f"core Resident does not fit ({why}); core Layer window fits "
            f"({lw.hbm_bytes / GB:.2f} GB, {cap})"
        )
    return window, f"core Resident does not fit ({why}); core Layer window, smaller ({cap})"


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
        # Resume: the checkpoint's plan record (``read_saved_plan``) and why its choice could
        # not be pinned (None: pinned, or nothing to pin).
        self.saved_plan: Optional[Dict[str, Any]] = None
        self.pin_reason: Optional[str] = None

    def on_phase_start(self, args, model, *, apply: bool) -> Optional[LoopRSBFPlan]:
        """Plan the phase of ``args`` for ``model`` (a list of one model chunk, or the chunk),
        log the decision table and check the digest on every rank; ``apply`` sets the plan on
        the model config and ``args`` (module docstring), else the current option is checked.
        On the first ``apply`` of a resume (``--load``, not finetuning) the checkpoint's applied
        choice is pinned when this run's menu holds it and it is feasible (``pin_saved_plan``).
        A feasible pin always wins: ``--loop-resume-changes allow`` only accepts the fresh plan
        when the pin fails (the resume check's message 5 then names the change); there is no
        "re-plan on resume" switch under ``apply``. To resume with another plan, use
        ``--loop-planner report`` with the new choice set by hand (``--loop-rsbf-item-form`` /
        ``--loop-rsbf-block-stride`` / ``--loop-rsbf-waves``) and ``--loop-resume-changes
        allow``. Returns the plan (None when nothing fits in report mode; apply raises
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
        if capabilities.layer_window:
            # The store's runtime memory terms of every option of the menu.
            shape = loop_planner_with_store_terms(shape, planner, phase, module)
            planner = make_loop_planner(args, shape, hardware, capabilities)
        saved = read_saved_plan(args) if apply and not rerun else None
        try:
            plan = planner.plan(phase, depth)
        except PlannerNoFit as error:
            if apply:
                raise
            warn_rank_0(f"loop planner (report): {error}")
            return None
        if saved is not None:
            fresh = plan
            plan, self.pin_reason = pin_saved_plan(plan, saved)
            self.saved_plan = saved
            if self.pin_reason is None and plan is not fresh:
                print_rank_0(
                    "loop planner: resumed with the checkpoint's plan "
                    f"{_choice_text(loop_plan_choice(plan))} (planned afresh: "
                    f"{_choice_text(loop_plan_choice(fresh))})"
                )
            elif self.pin_reason is not None:
                warn_rank_0(
                    "loop planner: the checkpoint's plan "
                    f"{_choice_text(saved.get('choice') or {})} could not be pinned "
                    f"({self.pin_reason}); planned afresh: "
                    f"{_choice_text(loop_plan_choice(plan))}"
                )
        _log_plan(args, plan)
        _check_plan_digest(plan.digest)
        if apply:
            self._apply(args, module, config, plan, rerun=rerun)
            self.applied = True
        else:
            store = find_param_store(module)
            current, message = loop_planner_current_option(
                plan,
                config,
                optimizer_offload=bool(getattr(args, "optimizer_cpu_offload", False)),
                store_modes=None if store is None else store.modes,
            )
            if current is not None and current.fits and current.applicable:
                print_rank_0(message)
            else:
                warn_rank_0(message)
            if store is not None:
                print_rank_0(
                    "loop planner (report): store modes "
                    f"{'unset' if store.modes is None else store.modes.describe()}; the plan's "
                    f"{plan.store_modes().describe()}"
                )
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
        store = find_param_store(module)
        store_text = ""
        if store is not None:
            from megatron.core.transformer.loop.program import build_core_plan

            modes = apply_plan_to_store(
                store, plan, core_plan=build_core_plan(stack.config, stack.loop_roles)
            )
            store_text = f" store_modes={modes.describe()}"
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
            + store_text
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


# ----------------------------------------------------------------------------- resume (M3c)
# Item forms whose resumes are bitwise interchangeable (the nested item equals the whole round
# bitwise, stage 4); the block stride does not apply to them.
SWAPPABLE_ITEM_FORMS = frozenset({"whole_round", "nested"})


def _option_choice(option: Any, core_layers: int) -> Dict[str, Any]:
    form, stride = option_item_form(option, core_layers)
    return {
        "item_form": form,
        "block_stride": None if stride is None else int(stride),
        "waves": int(option.waves),
        "coda_recompute": bool(option.coda_recompute),
        "optimizer_offload": "A" in option.moves,
        "moves": str(option.moves),
        "layout": str(option.layout),
        "regather": bool(option.regather),
    }


def loop_plan_choice(plan: LoopRSBFPlan) -> Dict[str, Any]:
    """The applied choice of ``plan`` in plain types: the four RSBF settings, Adam to Grace and
    the store layout of its option (moves, dense layout, re-gathered blocks)."""
    return _option_choice(plan.applied.option, plan.core_layers)


def _choice_text(choice: Mapping[str, Any]) -> str:
    return " ".join(f"{key}={choice.get(key)!r}" for key in sorted(choice))


def loop_plan_record(plan: LoopRSBFPlan, *, depth_source: str = "config") -> Dict[str, Any]:
    """The checkpoint record of an applied plan (``loop_resume["plan"]``), plain types only
    (``torch.load(weights_only=True)``-safe): the choice (``loop_plan_choice``), the store modes
    of the applied option, the depth distribution the planner used and its source
    (``--loop-planner-depths``), the phase, the hardware preset and the digest."""
    depth = plan.depth
    return {
        "choice": loop_plan_choice(plan),
        "store_modes": plan.store_modes().describe(),
        "depth": {
            "kind": str(depth.kind),
            "params": [float(value) for value in depth.params],
            "max_rounds": int(depth.max_rounds),
            "name": str(depth.name),
            "source": str(depth_source),
        },
        "phase": json.loads(json.dumps(dataclasses.asdict(plan.phase))),
        "hardware": str(plan.hardware.name),
        "digest": str(plan.digest),
    }


def read_saved_plan(args) -> Optional[Dict[str, Any]]:
    """The checkpoint's plan record under ``--loop-planner apply`` with ``--load`` set and not
    finetuning: ``loop_resume["plan"]`` of the checkpoint's common state, read the way
    ``load_args_from_checkpoint`` reads it (``_load_base_checkpoint(..., rank0=True)``, every
    rank, ``--ckpt-step`` honoured); None otherwise or when the checkpoint has no record (no
    checkpoint yet, a pre-M3c checkpoint, one saved without an applied plan, or
    ``--non-persistent-ckpt-type local``, which has no rank-0 read).

    Cost: a torch_dist checkpoint reads only ``common.pt``; a legacy ``torch`` checkpoint (no
    store: the store saves torch_dist only) loads rank 0's whole ``model_optim_rng.pt`` on every
    rank, as ``load_args_from_checkpoint`` (``--use-checkpoint-args``) already does. Accepted
    for that rare combination (legacy format under ``apply``) to keep the planner free of a new
    collective."""
    if getattr(args, "loop_planner", "off") != "apply" or getattr(args, "finetune", False):
        return None
    load_dir = getattr(args, "load", None)
    if not load_dir:
        return None
    if getattr(args, "non_persistent_ckpt_type", None) == "local":
        # A local non-persistent checkpoint is read through the training's
        # LocalCheckpointManager, which needs the model's sharded state dict: dev has no
        # rank-0 read of it (--use-checkpoint-args refuses 'local' for the same reason). The
        # saved choice is then not pinned; check_resume_plan still refuses a resume whose
        # fresh plan differs from the loaded checkpoint's (message 5).
        print_rank_0(
            "loop planner: --non-persistent-ckpt-type local has no rank-0 checkpoint read; the "
            "checkpoint's applied plan is not pinned (a changed plan is refused after the load)"
        )
        return None
    # pylint: disable=import-outside-toplevel
    from megatron.training.checkpointing import _load_base_checkpoint
    from megatron.training.loop_checkpoint import LOOP_RESUME_KEY

    state_dict, *_ = _load_base_checkpoint(load_dir, args, rank0=True)
    if not state_dict:
        return None
    record = (state_dict.get(LOOP_RESUME_KEY) or {}).get("plan")
    return dict(record) if record else None


def pin_saved_plan(
    plan: LoopRSBFPlan, saved: Mapping[str, Any]
) -> Tuple[LoopRSBFPlan, Optional[str]]:
    """``(plan, None)`` with the saved record's choice applied: ``plan`` itself when it already
    chose it, else ``plan`` with the menu row of that choice as its applied option (a pinned
    plan; its digest is derived from the fresh digest and the choice, the same on every rank).
    ``(plan, reason)`` unchanged when the choice cannot be pinned: ``"not in this run's menu"``
    or ``"infeasible: <the planner's refusals>"`` (it does not fit, is not applicable, or the
    form policy excludes it)."""
    choice = dict(saved.get("choice") or {})
    if not choice:
        return plan, "the checkpoint records no choice"
    if loop_plan_choice(plan) == choice:
        return plan, None
    row = next(
        (r for r in plan.table if _option_choice(r.option, plan.core_layers) == choice), None
    )
    if row is None:
        return plan, "not in this run's menu"
    if not (row.fits and row.applicable) or row.refusals:
        return plan, "infeasible: " + "; ".join(row.refusals or ("does not fit",))
    option = row.option
    form, stride = option_item_form(option, plan.core_layers)
    digest = hashlib.sha256(
        json.dumps([plan.digest, "pinned", choice], sort_keys=True).encode()
    ).hexdigest()
    pinned = dataclasses.replace(
        plan,
        applied=row,
        item_form=form,
        block_stride=stride,
        stride_layers=option.stride_layers,
        waves=option.waves,
        coda_recompute=option.coda_recompute,
        optimizer_offload="A" in option.moves,
        notes=plan.notes + (f"pinned to the checkpoint's choice ({_choice_text(choice)})",),
        digest=digest,
    )
    return pinned, None


def check_resume_plan(
    args,
    saved: Optional[Mapping[str, Any]],
    current: Optional[LoopRSBFPlan],
    *,
    pin_reason: Optional[str] = None,
) -> Tuple[Optional[str], List[str]]:
    """``(problem, notes)`` of a resume whose applied plan may differ from the checkpoint's.

    ``problem`` (None when equal, or when either side has no applied plan) names every changed
    field of the choice (``loop_plan_choice``) and, when both runs plan from the exit criterion
    (``--loop-planner-depths config``), a changed depth distribution; a measured histogram or
    an R4 setting is a planner input, never a problem. The digest is not compared (memory-model
    inputs that change no choice must not block a resume). A change between the whole round
    and the nested item (bitwise interchangeable, the block stride ignored) is a note.
    ``pin_reason`` (``LoopPlannerSession.pin_reason``) says why the saved choice was not
    pinned."""
    if saved is None or current is None:
        return None, []
    saved_choice = dict(saved.get("choice") or {})
    now = loop_plan_choice(current)
    changed = {
        key: (saved_choice.get(key), now.get(key))
        for key in sorted(set(saved_choice) | set(now))
        if saved_choice.get(key) != now.get(key)
    }
    notes = []
    forms = {saved_choice.get("item_form"), now.get("item_form")}
    if changed and set(changed) <= {"item_form", "block_stride"} and forms <= SWAPPABLE_ITEM_FORMS:
        notes.append(
            f"loop resume: RSBF item form {saved_choice.get('item_form')} -> "
            f"{now.get('item_form')} (bitwise interchangeable)"
        )
        changed = {}
    saved_depth = dict(saved.get("depth") or {})
    source = getattr(args, "loop_planner_depths", "config")
    if saved_depth and saved_depth.get("source", "config") == "config" and source == "config":
        depth_now = [
            current.depth.kind,
            [float(value) for value in current.depth.params],
            current.depth.max_rounds,
        ]
        depth_then = [saved_depth.get(key) for key in ("kind", "params", "max_rounds")]
        if depth_now != depth_then:
            changed["depth distribution"] = (depth_then, depth_now)
    if not changed:
        return None, notes
    text = ", ".join(f"{key}: {old!r} -> {new!r}" for key, (old, new) in changed.items())
    if pin_reason is not None:
        text += f"; the saved plan could not be pinned: {pin_reason}"
    return f"the RSBF plan: {text}", notes
