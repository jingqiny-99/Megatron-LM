# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Synthetic-task accuracy against the number of reasoning steps, per checkpoint.

The evaluation of the loop residual-carrier ablation's synthetic stage (p-hop induction,
variable tracking, multi-step arithmetic). The data sets come from the agentic-mcore-dev
toolkit (``synth-tasks generate``): one directory per task holding ``manifest.json`` and
``eval_s{s}.npz`` files, each with ``tokens [rows, seq_length]`` (whole problems packed per row,
row-tail padding 0), ``row_length [rows]`` and, per problem, ``answer_row``, ``answer_pos`` (the
answer token's position in its row), ``answer`` and ``steps``.

For every requested checkpoint iteration of ``--load`` (``--synth-eval-iterations``: ``all``,
``latest`` or a comma-separated list), the model of ``pretrain_hybrid.py`` is loaded without
optimizer or RNG state, every evaluation row runs one forward in eval mode through
``pretrain_hybrid.get_batch`` (the training batch path: the same position ids, masks, row-tail
padding mask, TP broadcast and CP split as training rows), and a problem is correct when the
argmax of the logits at its ANS position (``answer_pos - 1``) is the answer token. With
``--synth-eval-depths d ...`` every forward also runs at each uniform loop depth ``d`` through
``set_loop_depths``. Rank 0 appends one JSON record per (iteration, depth) to
``--synth-eval-output`` (default ``<load>/synth_eval.jsonl``):

    {"format": "synth-eval-v1", "iteration": 250, "depth": null, "load": ..., "data": ...,
     "tasks": {"phop": {"train_steps": [1, 16], "chance": 0.25,
                         "steps": {"1": [correct, total], ...},
                         "baseline": {"1": 0.38, ...}, "train_overlap": {"1": null, ...}},
               ...}}

which ``synth-tasks report`` turns into accuracy-vs-steps tables. ``baseline`` (the best
trivial heuristic's accuracy) and ``train_overlap`` (evaluation problems also in training) are
copied from the task manifest and describe the whole evaluation set, also when
``--synth-eval-max-rows`` scores only a subset.

Rows are split over the data-parallel ranks in equal counts; the last rank's shortfall is filled
with padding rows (a copy of row 0 with a zero loss mask, the convention of ``GPTDataset``'s
batch-padding sample) that are excluded from the counts, so every rank runs the same number of
forwards and an expert-parallel all-to-all cannot hang. Pipeline and context parallelism are
not supported (the synthetic stage runs neither).

Launch through job-launch with the toolkit training config
``loop_ablation_synth_eval_1gpu_h100`` (or the 8-GPU form) and the synthetic data config, e.g.

    uv run job-launch launch remote slurm --dev -m loop_ablation/h512_c24/loop_attnres_1 \\
        -d synth_mix -t loop_ablation_synth_eval_8gpu_h100 --cluster eos -- --load <run>/checkpoints
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULT_FORMAT = "synth-eval-v1"
EVAL_FIELDS = ("tokens", "row_length", "answer_row", "answer_pos", "answer", "steps")
_ITERATION_DIR = re.compile(r"^iter_(\d{7})$")


def add_synthetic_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The ``--synth-eval-*`` arguments."""
    group = parser.add_argument_group(title="synthetic-task evaluation")
    group.add_argument(
        "--synth-eval-data",
        type=str,
        required=True,
        help="a synth-tasks output root (one directory per task) or one task directory",
    )
    group.add_argument(
        "--synth-eval-output",
        type=str,
        default=None,
        help="JSONL file the records are appended to (default <load>/synth_eval.jsonl)",
    )
    group.add_argument(
        "--synth-eval-iterations",
        type=str,
        default="all",
        help="'all' (every iter_* directory of --load), 'latest', or a comma-separated list",
    )
    group.add_argument(
        "--synth-eval-tasks",
        type=str,
        default=None,
        help="comma-separated task names (default: every task directory)",
    )
    group.add_argument(
        "--synth-eval-depths",
        type=int,
        nargs="+",
        default=None,
        help="also evaluate at these uniform loop depths (set_loop_depths); default: only the "
        "model's own depth",
    )
    group.add_argument(
        "--synth-eval-max-rows",
        type=int,
        default=None,
        help="evaluate at most this many rows per evaluation set (smoke runs; the recorded "
        "baseline and train_overlap still describe the whole set)",
    )
    return parser


# ---------------------------------------------------------------------------------------------
# Pure helpers (no process group, no CUDA)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSet:
    """One ``eval_s{steps}.npz`` of one task."""

    task: str
    steps: int
    path: str
    train_steps: Sequence[int]
    chance: Optional[float]
    baseline: Optional[float] = None  # the best trivial heuristic's accuracy (manifest)
    train_overlap: Optional[int] = None  # problems that also occur in training (manifest)


def checkpoint_iterations(load_dir: str) -> List[int]:
    """The iterations of the ``iter_XXXXXXX`` checkpoint directories under ``load_dir``."""
    if not os.path.isdir(load_dir):
        raise ValueError(f"--load {load_dir} is not a directory")
    found = []
    for name in os.listdir(load_dir):
        match = _ITERATION_DIR.match(name)
        if match and os.path.isdir(os.path.join(load_dir, name)):
            found.append(int(match.group(1)))
    return sorted(found)


def select_iterations(spec: str, load_dir: str) -> List[int]:
    """The iterations ``--synth-eval-iterations`` names; every one must exist."""
    available = checkpoint_iterations(load_dir)
    if not available:
        raise ValueError(f"no iter_* checkpoint directories under {load_dir}")
    spec = spec.strip()
    if spec == "all":
        return available
    if spec == "latest":
        return available[-1:]
    try:
        wanted = [int(item) for item in spec.split(",") if item.strip()]
    except ValueError:
        raise ValueError(
            f"--synth-eval-iterations {spec!r}: expected 'all', 'latest' or integers"
        ) from None
    missing = [iteration for iteration in wanted if iteration not in available]
    if missing or not wanted:
        raise ValueError(
            f"--synth-eval-iterations: no checkpoint for {missing or spec!r} under {load_dir} "
            f"(available: {available})"
        )
    return wanted


def discover_eval_sets(data_dir: str, tasks: Optional[Iterable[str]] = None) -> List[EvalSet]:
    """The evaluation sets of every task directory under ``data_dir`` (or of ``data_dir``
    itself), in task-name and step order, as each task's ``manifest.json`` lists them."""
    if os.path.isfile(os.path.join(data_dir, "manifest.json")):
        task_dirs = [data_dir]
    elif os.path.isdir(data_dir):
        task_dirs = [
            os.path.join(data_dir, name)
            for name in sorted(os.listdir(data_dir))
            if os.path.isfile(os.path.join(data_dir, name, "manifest.json"))
        ]
    else:
        raise ValueError(f"--synth-eval-data {data_dir} does not exist")
    wanted = None if tasks is None else set(tasks)
    sets: List[EvalSet] = []
    seen = set()
    for task_dir in task_dirs:
        with open(os.path.join(task_dir, "manifest.json"), encoding="utf-8") as handle:
            manifest = json.load(handle)
        task = manifest["task"]
        seen.add(task)
        if wanted is not None and task not in wanted:
            continue
        low, high = manifest["steps_eval"]
        per_steps = manifest.get("splits", {}).get("eval", {}).get("per_steps", {})
        for steps in range(int(low), int(high) + 1):
            path = os.path.join(task_dir, f"eval_s{steps}.npz")
            if not os.path.isfile(path):
                raise ValueError(f"{path} is missing (listed by {task_dir}/manifest.json)")
            meta = per_steps.get(str(steps), {})
            sets.append(
                EvalSet(
                    task=task,
                    steps=steps,
                    path=path,
                    train_steps=tuple(int(s) for s in manifest["steps_train"]),
                    chance=manifest.get("chance_accuracy"),
                    baseline=meta.get("baseline_accuracy"),
                    train_overlap=meta.get("train_overlap"),
                )
            )
    if wanted is not None and wanted - seen:
        raise ValueError(f"--synth-eval-tasks: no task directory for {sorted(wanted - seen)}")
    if not sets:
        raise ValueError(f"no evaluation sets under {data_dir}")
    return sets


def load_eval_arrays(path: str, seq_length: int, max_rows: Optional[int] = None) -> dict:
    """An evaluation set's arrays; rows must be ``seq_length`` long. With ``max_rows`` only the
    first rows and their problems are kept."""
    with numpy.load(path) as data:
        missing = [name for name in EVAL_FIELDS if name not in data.files]
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        arrays = {name: numpy.asarray(data[name]) for name in EVAL_FIELDS}
    rows, width = arrays["tokens"].shape
    if width != seq_length:
        raise ValueError(
            f"{path}: rows of {width} tokens, but --seq-length is {seq_length} (regenerate the "
            f"data with synth-tasks generate --seq-length {seq_length})"
        )
    if max_rows is not None and max_rows < rows:
        keep = arrays["answer_row"] < max_rows
        arrays = {
            name: (value[:max_rows] if name in ("tokens", "row_length") else value[keep])
            for name, value in arrays.items()
        }
    return arrays


def forwards_per_rank(num_rows: int, dp_size: int, micro_batch_size: int) -> int:
    """The forwards every data-parallel rank runs for ``num_rows`` rows (equal on all ranks)."""
    per_forward = dp_size * micro_batch_size
    return (num_rows + per_forward - 1) // per_forward


def rank_rows(num_rows: int, dp_rank: int, dp_size: int, micro_batch_size: int) -> List[int]:
    """The rows of data-parallel rank ``dp_rank``: one contiguous block of
    ``forwards_per_rank * micro_batch_size`` slots; ``-1`` marks a padding slot."""
    slots = forwards_per_rank(num_rows, dp_size, micro_batch_size) * micro_batch_size
    start = dp_rank * slots
    return [row if row < num_rows else -1 for row in range(start, start + slots)]


def build_microbatch(
    arrays: dict, row_ids: Sequence[int], args, eod: int
) -> Dict[str, torch.Tensor]:
    """The CPU batch dict of ``row_ids`` exactly as ``GPTDataset`` with SBHD document packing
    builds a row (``_get_ltor_masks_and_position_ids`` with the run's flags, row-tail
    ``padding_mask``, loss mask 0 from ``content - 1`` on); a padding slot (``-1``) is row 0
    with a zero loss mask."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids

    seq_length = arrays["tokens"].shape[1]
    out: Dict[str, List[torch.Tensor]] = {}
    for row in row_ids:
        source = max(row, 0)
        tokens = torch.from_numpy(arrays["tokens"][source].astype(numpy.int64))
        content = int(arrays["row_length"][source])
        labels = torch.zeros_like(tokens)
        labels[:-1] = tokens[1:]
        attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
            tokens,
            eod,
            bool(getattr(args, "reset_position_ids", False)),
            bool(getattr(args, "reset_attention_mask", False)),
            bool(getattr(args, "eod_mask_loss", False)),
            bool(getattr(args, "create_attention_mask_in_dataloader", False)),
        )
        loss_mask[max(content - 1, 0) :] = 0.0
        if row < 0:
            loss_mask.zero_()
        fields = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "padding_mask": torch.arange(seq_length) >= content,
        }
        if attention_mask is not None:
            fields["attention_mask"] = attention_mask
        for name, value in fields.items():
            out.setdefault(name, []).append(value)
    return {name: torch.stack(values) for name, values in out.items()}


def score(
    arrays: dict, row_ids: Sequence[int], predictions: numpy.ndarray, counts: List[int]
) -> None:
    """Add this microbatch's (correct, total) to ``counts``: a problem in a real row is correct
    when ``predictions[b, answer_pos - 1]`` (the argmax at its ANS token) is its answer."""
    answer_row = arrays["answer_row"]
    for b, row in enumerate(row_ids):
        problems = numpy.flatnonzero(answer_row == row)  # none for a padding slot (-1)
        if problems.size == 0:
            continue
        guesses = predictions[b, arrays["answer_pos"][problems] - 1]
        counts[0] += int((guesses == arrays["answer"][problems]).sum())
        counts[1] += int(problems.size)


def make_record(
    iteration: int,
    depth: Optional[int],
    sets: Sequence[EvalSet],
    totals: Sequence[Sequence[int]],
    *,
    load: str,
    data: str,
) -> dict:
    """The JSON record of one (iteration, depth): per task the training step range, chance,
    ``[correct, total]`` per step count, and from the manifest the best trivial heuristic's
    accuracy (``baseline``) and the problems that also occur in training (``train_overlap``)
    per step count."""
    tasks: Dict[str, dict] = {}
    for eval_set, (correct, total) in zip(sets, totals):
        body = tasks.setdefault(
            eval_set.task,
            {
                "train_steps": list(eval_set.train_steps),
                "chance": eval_set.chance,
                "steps": {},
                "baseline": {},
                "train_overlap": {},
            },
        )
        body["steps"][str(eval_set.steps)] = [int(correct), int(total)]
        body["baseline"][str(eval_set.steps)] = eval_set.baseline
        body["train_overlap"][str(eval_set.steps)] = eval_set.train_overlap
    return {
        "format": RESULT_FORMAT,
        "iteration": int(iteration),
        "depth": None if depth is None else int(depth),
        "load": load,
        "data": data,
        "tasks": tasks,
    }


def check_depths(depths: Optional[Sequence[int]], decoder, loop_iterations: int) -> List:
    """The depths to evaluate (``[None]`` = the model's own depth only, first). Refuses up front,
    before any checkpoint is loaded, the configurations ``set_loop_depths`` would refuse at the
    first depth forward (same conditions as ``LoopHybridStack``)."""
    if not depths:
        return [None]
    if not hasattr(decoder, "set_loop_depths"):
        raise ValueError("--synth-eval-depths needs a hybrid loop decoder (set_loop_depths)")
    config = getattr(decoder, "config", None)
    if getattr(config, "loop_untie_core", False):
        raise ValueError(
            "--synth-eval-depths: an untied loop core (--loop-untie-core, plain-inject) is a "
            "fixed-depth baseline; per-row loop depths need a tied Huginn core"
        )
    if getattr(config, "loop_architecture", "huginn") != "huginn":
        raise ValueError(
            "--synth-eval-depths: per-row loop depths need --loop-architecture huginn, not "
            f"{config.loop_architecture!r}"
        )
    if getattr(config, "loop_bptt_iterations", 0):
        raise ValueError(
            "--synth-eval-depths: per-row loop depths need full BPTT "
            f"(--loop-bptt-iterations 0, not {config.loop_bptt_iterations})"
        )
    bad = [d for d in depths if not 1 <= d <= loop_iterations]
    if bad:
        raise ValueError(f"--synth-eval-depths {bad}: depths must be in 1..{loop_iterations}")
    return [None] + sorted(set(depths))


# ---------------------------------------------------------------------------------------------
# Distributed evaluation
# ---------------------------------------------------------------------------------------------


def _evaluate(model, decoder, sets, depth, args, get_batch) -> List[List[int]]:
    # pylint: disable=import-outside-toplevel
    from megatron.core import mpu
    from megatron.training import get_tokenizer

    dp_rank = mpu.get_data_parallel_rank()
    dp_size = mpu.get_data_parallel_world_size()
    mbs = args.micro_batch_size
    eod = get_tokenizer().eod
    totals = []
    for eval_set in sets:
        arrays = load_eval_arrays(eval_set.path, args.seq_length, args.synth_eval_max_rows)
        slots = rank_rows(arrays["tokens"].shape[0], dp_rank, dp_size, mbs)
        counts = [0, 0]
        for start in range(0, len(slots), mbs):
            row_ids = slots[start : start + mbs]
            batch = build_microbatch(arrays, row_ids, args, eod)
            cp_batch = get_batch(iter([batch]), None, pg_collection=None)
            fields = cp_batch.get_batch()
            if depth is not None:
                decoder.set_loop_depths(
                    torch.full((mbs,), depth, dtype=torch.long, device=fields["tokens"].device)
                )
            logits = model(
                fields["tokens"],
                fields["position_ids"],
                fields.get("attention_mask"),
                labels=None,
                runtime_gather_output=True,
                packed_seq_params=cp_batch.get_packed_seq_params(),
                loss_mask=fields.get("loss_mask"),
                padding_mask=fields.get("padding_mask"),
                cp_batch=cp_batch,
            )
            predictions = logits.argmax(dim=-1).cpu().numpy()
            score(arrays, row_ids, predictions, counts)
        totals.append(counts)
    reduced = torch.tensor(totals, dtype=torch.long, device=torch.cuda.current_device())
    torch.distributed.all_reduce(reduced, group=mpu.get_data_parallel_group())
    return reduced.cpu().tolist()


def main() -> None:
    """Evaluate every requested checkpoint and append the records."""
    # pylint: disable=import-outside-toplevel
    from functools import partial

    from hybrid_builders import hybrid_builder
    from megatron.core.utils import unwrap_model
    from megatron.training import get_args, get_model, initialize_megatron, print_rank_0
    from megatron.training.argument_utils import inference_cfg_container_from_args
    from megatron.training.arguments import parse_and_validate_args
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.global_vars import set_run_config
    from model_provider import model_provider
    from pretrain_hybrid import get_batch

    args = parse_and_validate_args(
        extra_args_provider=add_synthetic_eval_args,
        args_defaults={"no_load_rng": True, "no_load_optim": True},
    )
    if not args.load:
        raise ValueError("synthetic_eval needs --load (the run's checkpoint directory)")
    if args.pipeline_model_parallel_size != 1 or args.context_parallel_size != 1:
        raise ValueError("synthetic_eval supports neither pipeline nor context parallelism")
    if getattr(args, "sequence_packing_scheduler", None) is not None:
        raise ValueError("synthetic_eval reads SBHD rows; drop --sequence-packing-scheduler")
    # The carrier diagnostics are a training-step quantity; an eval forward would only add
    # their reductions.
    args.loop_carrier_diagnostics = "off"
    tasks = None
    if args.synth_eval_tasks:
        tasks = [name.strip() for name in args.synth_eval_tasks.split(",") if name.strip()]
    sets = discover_eval_sets(args.synth_eval_data, tasks)
    iterations = select_iterations(args.synth_eval_iterations, args.load)
    output = args.synth_eval_output or os.path.join(args.load, "synth_eval.jsonl")

    set_run_config(inference_cfg_container_from_args(args, build_model_config=False))
    initialize_megatron(training=False)
    args = get_args()
    model = get_model(partial(model_provider, hybrid_builder), wrap_with_ddp=False)
    for module in model:
        module.eval()
    hybrid = unwrap_model(model[0])
    decoder = hybrid.decoder
    depths = check_depths(
        args.synth_eval_depths, decoder, int(getattr(args, "loop_iterations", 1) or 1)
    )
    print_rank_0(
        f"synthetic eval: {len(sets)} sets, iterations {iterations}, depths {depths} -> {output}"
    )
    for iteration in iterations:
        args.ckpt_step = iteration
        load_checkpoint(model, None, None, strict=True)
        for depth in depths:
            with torch.no_grad():
                totals = _evaluate(model[0], decoder, sets, depth, args, get_batch)
            record = make_record(
                iteration, depth, sets, totals, load=args.load, data=args.synth_eval_data
            )
            if torch.distributed.get_rank() == 0:
                with open(output, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                for task, body in record["tasks"].items():
                    correct = sum(c for c, _ in body["steps"].values())
                    total = sum(t for _, t in body["steps"].values())
                    print_rank_0(
                        f"synthetic eval: iteration {iteration} depth {depth} {task}: "
                        f"{correct}/{total} correct"
                    )
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
