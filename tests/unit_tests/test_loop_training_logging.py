# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training-side logging of the loop RSBF exit-depth histogram (megatron/training/
loop_rsbf_planning.py), with a stub schedule (no model, no process group):

* the post-``train_step`` hook: ``args._loop_exit_counts`` / ``_loop_padding``, the interval
  sums and their reset, the sum check against the batch the step ran
  (``get_current_running_global_batch_size``: a batch-size schedule passes, a wrong count
  raises), the JSONL line of the last rank (read back by ``DepthDistribution``);
* the TensorBoard / W&B values of a step and the console fragment of an interval;
* ``--loop-exit-histogram-path`` and its validator.
"""

import json
import sys
from argparse import ArgumentParser, Namespace
from unittest.mock import Mock

import pytest

from megatron.core.transformer.loop.planner import EXIT_HISTOGRAM_FORMAT, DepthDistribution
from megatron.training import loop_rsbf_planning as planning
from megatron.training.arguments import _validate_loop_exit_histogram, parse_args


class _StubSchedule:
    """The ``LoopRSBFSchedule`` accessors the hook reads, fed one step at a time."""

    def __init__(self, waves=1):
        self.counts = None
        self.pad = None
        self.executor = Namespace(waves=waves)
        self.models = []

    def exit_counts(self, model):
        self.models.append(model)
        return None if self.counts is None else list(self.counts)

    def padding(self, model):
        return self.pad

    def cached_executor(self, model):
        return self.executor


class _Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, name, value, iteration):
        self.scalars.append((name, value, iteration))


class _WandB:
    def __init__(self):
        self.logs = []

    def log(self, values, iteration):
        self.logs.append((dict(values), iteration))


def _args(path=None, **overrides):
    values = dict(
        loop_exit_histogram_path=path,
        seq_length=4096,
        context_parallel_size=1,
        world_size=8,
        data_parallel_size=8,
        expert_model_parallel_size=8,
        global_batch_size=64,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.fixture
def step_batch(monkeypatch):
    """The running global batch of the step (settable) and the last-rank flag; a fresh
    process-wide histogram log."""
    state = Namespace(samples=64, last_rank=True)
    monkeypatch.setattr(planning, "get_current_running_global_batch_size", lambda: state.samples)
    monkeypatch.setattr(planning, "is_last_rank", lambda: state.last_rank)
    planning.reset_loop_exit_histogram_log()
    try:
        yield state
    finally:
        planning.reset_loop_exit_histogram_log()


def _step(args, schedule, iteration, counts, pad=None):
    schedule.counts, schedule.pad = counts, pad
    return planning.record_loop_exit_histogram(args, schedule, ["model chunk"], iteration)


def test_hook_records_the_step_and_the_interval(tmp_path, step_batch):
    path = tmp_path / "run" / "exit_hist.jsonl"
    args = _args(str(path))
    schedule = _StubSchedule(waves=2)
    assert _step(args, schedule, 1, None) is None
    assert not hasattr(args, "_loop_exit_counts") and not hasattr(
        args, "_loop_exit_counts_interval"
    )

    assert _step(args, schedule, 1, [16, 16, 16, 16], (512, 8 * 4096)) == [16, 16, 16, 16]
    assert schedule.models[-1] == "model chunk"
    assert args._loop_exit_counts == [16, 16, 16, 16] and args._loop_padding == [512, 32768]
    _step(args, schedule, 2, [30, 20, 10, 4], (0, 32768))
    assert args._loop_exit_counts == [30, 20, 10, 4]
    assert args._loop_exit_counts_interval == [46, 36, 26, 20]
    assert planning.get_loop_exit_histogram_log(args).interval_padding() == (512, 65536)

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0] == {
        "format": EXIT_HISTOGRAM_FORMAT,
        "iteration": 1,
        "R_max": 4,
        "counts": [16, 16, 16, 16],
        "samples": 64,
        "padding_fraction": 512 / 32768,
        "phase": {"seq_length": 4096, "cp": 1, "gpus": 8, "ep": 8, "samples_per_eu": 8, "waves": 2},
    }
    assert [line["iteration"] for line in lines] == [1, 2]
    assert lines[1]["padding_fraction"] == 0.0
    measured = DepthDistribution.from_histogram_file(str(path), max_rounds=4)
    assert measured.params == (46 / 128, 36 / 128, 26 / 128, 20 / 128)


def test_args_keep_plain_data_for_checkpoints(tmp_path, step_batch):
    """Checkpoints pickle ``args`` and load them with ``weights_only=True``: the hook leaves
    only plain lists there (the log object stays in the module)."""
    import io

    import torch

    args = _args(str(tmp_path / "exit_hist.jsonl"))
    _step(args, _StubSchedule(), 1, [16, 16, 16, 16], (4, 64))
    buffer = io.BytesIO()
    torch.save({"args": args}, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)["args"]
    assert loaded._loop_exit_counts == [16, 16, 16, 16] and loaded._loop_padding == [4, 64]
    assert loaded._loop_exit_counts_interval == [16, 16, 16, 16]


def test_console_fragment_and_interval_reset(step_batch):
    args = _args()
    schedule = _StubSchedule()
    assert planning.loop_exit_histogram_log_string(args, reset=True) == ""
    _step(args, schedule, 1, [16, 16, 16, 16], (1024, 32768))
    _step(args, schedule, 2, [10, 22, 0, 32], (0, 32768))
    text = planning.loop_exit_histogram_log_string(args, reset=False)
    assert text == (
        " loop exit depths 1..4: 26 / 38 / 16 / 48 (mean 2.67, 128 samples) |"
        " loop padding fraction (logging rank's EU): 0.0156 |"
    )
    # The first iteration's extra line does not reset; the interval's line does.
    assert planning.loop_exit_histogram_log_string(args, reset=True) == text
    assert args._loop_exit_counts_interval == []
    assert planning.loop_exit_histogram_log_string(args, reset=True) == ""
    _step(args, schedule, 3, [64, 0, 0, 0], None)
    assert planning.loop_exit_histogram_log_string(args, reset=True) == (
        " loop exit depths 1..4: 64 / 0 / 0 / 0 (mean 1.00, 64 samples) |"
        " loop padding fraction (logging rank's EU): 0.0000 |"
    )


def test_tensorboard_and_wandb_values_of_the_step(step_batch):
    args = _args()
    writer, wandb = _Writer(), _WandB()
    planning.log_loop_exit_histogram(args, 5, writer, wandb)  # no step recorded: nothing
    assert writer.scalars == [] and wandb.logs == []
    _step(args, _StubSchedule(), 4, [16, 16, 16, 16])
    _step(args, _StubSchedule(), 5, [32, 0, 16, 16], (2048, 32768))  # this step's values
    planning.log_loop_exit_histogram(args, 5, writer, wandb)
    expected = {
        "loop exit depth count/d1": 32,
        "loop exit depth count/d2": 0,
        "loop exit depth count/d3": 16,
        "loop exit depth count/d4": 16,
        "loop exit depth fraction/d1": 0.5,
        "loop exit depth fraction/d2": 0.0,
        "loop exit depth fraction/d3": 0.25,
        "loop exit depth fraction/d4": 0.25,
        "loop exit depth mean": 2.25,
        "loop padding fraction": 0.0625,
    }
    assert writer.scalars == [(name, value, 5) for name, value in expected.items()]
    assert wandb.logs == [(expected, 5)]
    planning.log_loop_exit_histogram(args, 6, None, None)  # no writers on this rank: no error


def test_sum_check_uses_the_batch_the_step_ran(step_batch):
    """A batch-size schedule (or --decrease-batch-size-if-needed) runs another batch than
    --global-batch-size: the running batch is the reference; a wrong count raises."""
    args = _args(global_batch_size=64)
    step_batch.samples = 24
    _step(args, _StubSchedule(), 1, [6, 6, 6, 6])
    assert args._loop_exit_counts == [6, 6, 6, 6]
    with pytest.raises(RuntimeError, match=r"sums to 23 samples, the step ran 24"):
        _step(args, _StubSchedule(), 2, [6, 6, 6, 5])


def test_only_the_last_rank_writes_the_file(tmp_path, step_batch):
    path = tmp_path / "exit_hist.jsonl"
    step_batch.last_rank = False
    args = _args(str(path))
    _step(args, _StubSchedule(), 1, [64, 0, 0, 0])
    assert not path.exists() and args._loop_exit_counts_interval == [64, 0, 0, 0]


def test_interval_refuses_a_changed_r_max(step_batch):
    args = _args()
    _step(args, _StubSchedule(), 1, [32, 32])
    with pytest.raises(ValueError, match="R_max 3 in an interval of R_max 2"):
        _step(args, _StubSchedule(), 2, [32, 16, 16])


def test_cli_flag_and_validator(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test"])
    assert parse_args().loop_exit_histogram_path is None
    monkeypatch.setattr(sys, "argv", ["test", "--loop-exit-histogram-path", "/tmp/h.jsonl"])
    assert parse_args().loop_exit_histogram_path == "/tmp/h.jsonl"
    _validate_loop_exit_histogram(Namespace())
    _validate_loop_exit_histogram(Namespace(loop_exit_histogram_path=None))
    _validate_loop_exit_histogram(Namespace(loop_exit_histogram_path="h", loop_schedule="rsbf"))
    for args in (
        Namespace(loop_exit_histogram_path="h"),
        Namespace(loop_exit_histogram_path="h", loop_schedule="standard"),
    ):
        with pytest.raises(ValueError, match="needs --loop-schedule rsbf"):
            _validate_loop_exit_histogram(args)


@pytest.mark.parametrize("iteration,console", [(2, True), (3, False)])
def test_training_log_writes_the_histogram(monkeypatch, step_batch, run_config, iteration, console):
    """``training_log`` (log interval 2, TensorBoard interval 1): this step's values reach the
    TensorBoard and W&B writers every step; at the log interval the console line carries the
    interval's histogram (two steps) and a new interval starts."""
    from megatron.training import arguments, training
    from megatron.training.config import LoggerConfig

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args(["--micro-batch-size", "1"])
    args.data_parallel_size = args.world_size = args.gtp_weight_remat_size = 1
    args.consumed_train_samples = args.skipped_train_samples = 0
    args.dsa_indexer_loss_coeff = None
    args.loop_architecture = "huginn"  # the loop-FLOPs throughput path
    args.train_iters, args.seq_length = 10, 16
    step_batch.samples = 4
    run_config.logger = LoggerConfig(log_interval=2, tensorboard_log_interval=1)
    _step(args, _StubSchedule(), iteration - 1, [1, 1, 1, 1])
    _step(args, _StubSchedule(), iteration, [4, 0, 0, 0], (4, 64))
    writer, wandb, lines = Mock(), Mock(), []
    timers = Mock()
    timers.return_value.elapsed.return_value = 1.0
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: timers)
    monkeypatch.setattr(training, "get_tensorboard_writer", lambda: writer)
    monkeypatch.setattr(training, "get_wandb_writer", lambda: wandb)
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 4)
    monkeypatch.setattr(training, "one_logger_utils", Mock())
    monkeypatch.setattr(training, "print_rank_last", lines.append)
    monkeypatch.setattr(training, "reduce_max_stat_across_model_parallel_group", lambda x, **kw: x)
    for getter in ("get_one_logger", "get_energy_monitor", "get_telemetry"):
        monkeypatch.setattr(training, getter, lambda: None)
    training.training_log({}, {}, 0.01, iteration, 1.0, False, 0, None, None, None, None)

    scalars = {c.args[0]: c.args[1:] for c in writer.add_scalar.call_args_list}
    assert scalars["loop exit depth count/d1"] == (4, iteration)
    assert scalars["loop exit depth mean"] == (1.0, iteration)
    assert scalars["loop padding fraction"] == (4 / 64, iteration)
    logged = [c.args for c in wandb.log.call_args_list if "loop exit depth mean" in c.args[0]]
    assert logged == [(planning.loop_exit_histogram_scalars([4, 0, 0, 0], (4, 64)), iteration)]
    if console:
        (line,) = lines
        fragment = (
            " loop exit depths 1..4: 5 / 1 / 1 / 1 (mean 1.75, 8 samples) |"
            " loop padding fraction (logging rank's EU): 0.0625 |"
        )
        assert fragment + " loss scale:" in line, line
        assert planning.get_loop_exit_histogram_log(args).interval() == []
    else:
        assert lines == [] and args._loop_exit_counts_interval == [5, 1, 1, 1]
