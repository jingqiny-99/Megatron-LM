# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""report_memory's max-over-ranks line (LoggerConfig.log_memory_max_over_ranks).

A memory report prints data-parallel rank 0's values only. With the option on, every rank also
takes part in one all-gather and global rank 0 prints the elementwise maximum over the ranks,
with the rank of each maximum. With it off, the output is the legacy line and no collective runs.

CPU tests: fake CUDA memory values per rank; single-process tests stub the all-gather, and one
test runs two real gloo processes.
"""

import io
import json
import os
import socket
from argparse import ArgumentParser
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.training import arguments
from megatron.training.argument_utils import _default_config_from_args
from megatron.training.config import LoggerConfig
from megatron.training.utils import common_utils

MB = 1024 * 1024
# Per rank (bytes): allocated, max allocated, reserved, max reserved, device memory used.
VALUES = {
    0: (10 * MB, 50 * MB, 60 * MB, 70 * MB, 90 * MB),
    1: (20 * MB, 40 * MB, 80 * MB, 75 * MB, 85 * MB),
}
NAME = "(after 1 iterations)"
LEGACY_RANK0 = (
    "[Rank 0] (after 1 iterations) memory (MB) | allocated: 10.00 | max allocated: 50.00"
    " | reserved: 60.00 | max reserved: 70.00"
)
MAX_LINE = (
    "[Max over 2 ranks] (after 1 iterations) memory (MB) | allocated: 20.00"
    " | max allocated: 50.00 | reserved: 80.00 | max reserved: 75.00"
    " | rank of each maximum: allocated 1, max allocated 0, reserved 1, max reserved 1"
)
MAX_LINE_DEVICE = (
    "[Max over 2 ranks] (after 1 iterations) memory (MB) | allocated: 20.00"
    " | max allocated: 50.00 | reserved: 80.00 | max reserved: 75.00"
    " | total device memory used: 90.00 | rank of each maximum: allocated 1, max allocated 0,"
    " reserved 1, max reserved 1, total device memory used 0"
)


def _fake_cuda(target, rank):
    """Rank ``rank``'s fake memory counters on ``target`` (a monkeypatch or a setattr shim)."""
    allocated, max_allocated, reserved, max_reserved, used = VALUES[rank]
    target.setattr(torch.cuda, "is_available", lambda: False)
    target.setattr(torch.cuda, "memory_allocated", lambda *a, **k: allocated)
    target.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: max_allocated)
    target.setattr(torch.cuda, "memory_reserved", lambda *a, **k: reserved)
    target.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: max_reserved)
    target.setattr(torch.cuda, "device_memory_used", lambda *a, **k: used)


def _run_config(enabled, device_used=False):
    logger = SimpleNamespace(log_memory_max_over_ranks=enabled, log_device_memory_used=device_used)
    return SimpleNamespace(logger=logger)


class _SetattrShim:
    """``monkeypatch.setattr`` for a spawned worker (no pytest fixture there)."""

    def __init__(self):
        self.undo = []

    def setattr(self, obj, name, value):
        self.undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def restore(self):
        for obj, name, value in reversed(self.undo):
            setattr(obj, name, value)


def _stub_dist(monkeypatch, rank, gathers):
    """Two-rank distributed stubs: rank ``rank``'s view, an all-gather of VALUES, a call log."""

    def all_gather(output, tensor, group=None):
        assert group is None and tensor.dtype == torch.int64 and len(output) == 2
        gathers.append(tensor.tolist())
        for peer, out in enumerate(output):
            out.copy_(torch.tensor(VALUES[peer][: tensor.numel()], dtype=torch.int64))

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(dist, "all_gather", all_gather)
    monkeypatch.setattr(common_utils, "get_pg_rank", lambda group=None: rank)


def _report(monkeypatch, rank, enabled, device_used=False):
    gathers = []
    _fake_cuda(monkeypatch, rank)
    _stub_dist(monkeypatch, rank, gathers)
    monkeypatch.setattr(common_utils, "get_run_config", lambda: _run_config(enabled, device_used))
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        common_utils.report_memory(NAME, process_group=object())
    return buffer.getvalue().splitlines(), gathers


@pytest.mark.parametrize("device_used", [False, True])
def test_rank0_prints_max_and_ranks_of_maxima(monkeypatch, device_used):
    lines, gathers = _report(monkeypatch, rank=0, enabled=True, device_used=device_used)
    legacy = LEGACY_RANK0 + (" | total device memory used: 90.00" if device_used else "")
    assert lines == [legacy, MAX_LINE_DEVICE if device_used else MAX_LINE]
    assert gathers == [list(VALUES[0][: 5 if device_used else 4])]


def test_other_ranks_gather_but_print_nothing(monkeypatch):
    lines, gathers = _report(monkeypatch, rank=1, enabled=True)
    assert lines == []
    assert gathers == [list(VALUES[1][:4])]


@pytest.mark.parametrize("rank", [0, 1])
def test_off_keeps_legacy_output_and_runs_no_collective(monkeypatch, rank):
    lines, gathers = _report(monkeypatch, rank=rank, enabled=False)
    assert lines == ([LEGACY_RANK0] if rank == 0 else [])
    assert gathers == []


def test_without_a_process_group_the_helper_does_nothing(monkeypatch):
    _fake_cuda(monkeypatch, 0)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        common_utils._report_memory_max_over_ranks(NAME, False)
    assert buffer.getvalue() == ""


@pytest.mark.parametrize("argv, expected", [([], False), (["--log-memory-max-over-ranks"], True)])
def test_cli_flag_reaches_logger_config(argv, expected):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args(argv)
    assert args.log_memory_max_over_ranks is expected
    assert _default_config_from_args(LoggerConfig, args).log_memory_max_over_ranks is expected
    assert LoggerConfig().log_memory_max_over_ranks is False


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _gloo_worker(rank, port, enabled, out_dir):
    """One rank of the two-process test: report fake memory, record stdout and the gathers."""
    shim = _SetattrShim()
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=f"tcp://127.0.0.1:{port}")
    try:
        _fake_cuda(shim, rank)
        gathers = []
        real_all_gather = dist.all_gather

        def counting_all_gather(*args, **kwargs):
            gathers.append(1)
            return real_all_gather(*args, **kwargs)

        shim.setattr(dist, "all_gather", counting_all_gather)
        shim.setattr(common_utils, "get_run_config", lambda: _run_config(enabled))
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            common_utils.report_memory(NAME, process_group=dist.group.WORLD)
        dist.barrier()
        with open(os.path.join(out_dir, f"rank{rank}.json"), "w", encoding="utf-8") as handle:
            json.dump({"lines": buffer.getvalue().splitlines(), "gathers": len(gathers)}, handle)
    finally:
        shim.restore()
        dist.destroy_process_group()


@pytest.mark.parametrize("enabled", [False, True])
def test_two_gloo_ranks(tmp_path, enabled):
    mp.spawn(_gloo_worker, args=(_free_port(), enabled, str(tmp_path)), nprocs=2, join=True)
    results = [
        json.loads((tmp_path / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in (0, 1)
    ]
    assert results[0]["lines"] == ([LEGACY_RANK0, MAX_LINE] if enabled else [LEGACY_RANK0])
    assert results[1]["lines"] == []
    assert [r["gathers"] for r in results] == ([1, 1] if enabled else [0, 0])
