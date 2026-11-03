# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Soak of the loop ring (GPU): ``LOOP_RING_SOAK_STEPS`` (default 1,000) training steps at P = 4.

Runs under torchrun with CUDA at W = 4 or 8 ranks (P = 4, DP = W / 4, k = 1) on
``ring_gpu_utils``' tiny DSv4 loop in bf16 (EP 1), with ``loop_recompute`` (plan all 'C', one
routing tape per checkpoint call), DDP + the distributed optimizer (Adam, lr 1e-4) and one
``LoopRingSchedule`` (as ``train()`` reuses the module-level one): every step runs P microbatches
of new data per pipeline (sequence 256, MBS 1) through Megatron's interleaved schedule over R = 4
round views per stage, ``RingFinalize`` (grad reduce-scatter, per-token scaling, the router
expert-bias update) and an optimizer step.

Hang detection: NCCL's asynchronous error handling (``TORCH_NCCL_ASYNC_ERROR_HANDLING=1``, set
before the process group exists) and a per-step watchdog (``faulthandler.dump_traceback_later``,
``LOOP_RING_SOAK_STEP_TIMEOUT`` seconds, default 300): a stuck step dumps every thread's stack
and kills the process. Gates (every rank, gathered):

* every step finishes and its ``LoopRingStepRecord`` reads rounds 4 / 4, prelude / coda / MTP
  1 / 1 / 1, and every loss is finite;
* memory is flat after step 10: the per-step peak (``max_memory_allocated`` after resetting the
  peak at the step's start) and the allocated bytes at the step's end stay within +-1 % of their
  step-10 values on every rank (a leaked injection entry, routing tape or saved activation would
  grow them step by step).
"""

import faulthandler
import math
import os
import sys
import time

import pytest
import torch

# Before any process group exists (the NCCL watchdog reads it when a communicator is made).
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

from tests.unit_tests.transformer.loop.hybrid_loop_utils import determinism_env  # noqa: E402
from tests.unit_tests.transformer.loop.ring_gpu_utils import (  # noqa: E402
    CUDA,
    ROUNDS,
    WORLD,
    Layout,
    build_chunks,
    deterministic_algorithms,
    gather,
    hadamard_fallback,
    parallel_layout,
    rank_batches,
    run_step,
    tiny_config,
    wrap_ddp,
)

STEPS = int(os.environ.get("LOOP_RING_SOAK_STEPS", "1000"))
STEP_TIMEOUT = float(os.environ.get("LOOP_RING_SOAK_STEP_TIMEOUT", "300"))
STAGES = 4
BASELINE_STEP = 10

pytestmark = pytest.mark.skipif(
    not CUDA or WORLD < STAGES or WORLD % STAGES, reason="the P = 4 soak needs 4 or 8 GPUs"
)


def _optimizer(ddp_chunks):
    # pylint: disable=import-outside-toplevel
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    return get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=1e-4,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
            clip_grad=1.0,
        ),
        list(ddp_chunks),
        use_gloo_process_groups=False,
    )


def _soak():
    """Run the soak; returns this rank's per-step records."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer.loop.ring_pp import LoopRingSchedule

    layout = Layout(stages=STAGES, chunks=1, recompute=True)
    dtype = torch.bfloat16
    rows = []
    with parallel_layout(layout), hadamard_fallback(dtype):
        config = tiny_config(dtype, layout)
        models = build_chunks(config, layout)
        ddp = wrap_ddp(models, config, distributed_optimizer=True)
        optimizer = _optimizer(ddp)
        schedule = LoopRingSchedule()
        size = STAGES * layout.data_parallel  # P microbatches per pipeline
        try:
            for step in range(1, STEPS + 1):
                faulthandler.dump_traceback_later(STEP_TIMEOUT, exit=True, file=sys.stderr)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                batches = rank_batches(seed=1000 + step, size=size)
                optimizer.zero_grad()
                result = run_step(ddp, layout, batches, schedule=schedule, collect=False)
                success, grad_norm, _ = optimizer.step()
                torch.cuda.synchronize()
                record = result.record
                rows.append(
                    dict(
                        step=step,
                        seconds=time.perf_counter() - start,
                        peak=torch.cuda.max_memory_allocated(),
                        allocated=torch.cuda.memory_allocated(),
                        success=bool(success),
                        grad_norm=float(grad_norm) if grad_norm is not None else float("nan"),
                        rounds=record.rounds,
                        once=(record.prelude, record.coda, record.mtp),
                        finite=all(math.isfinite(loss) for loss, _ in result.reports.values()),
                    )
                )
        finally:
            faulthandler.cancel_dump_traceback_later()
        del optimizer, ddp, models
    return rows


@pytest.fixture(scope="module")
def soak():
    """The soak's per-step records of every rank."""
    with determinism_env(), deterministic_algorithms():
        rows = _soak()
    yield gather(rows)


def test_soak_completes_every_step(soak):
    """Every rank ran every step; every step ran R rounds and the edges once per sample, with
    finite losses and a successful optimizer step."""
    for rank, rows in enumerate(soak):
        assert len(rows) == STEPS, (rank, len(rows))
        for row in rows:
            assert row["rounds"] == (ROUNDS, ROUNDS), (rank, row)
            assert row["once"] == ((1, 1), (1, 1), (1, 1)), (rank, row)
            assert row["finite"] and row["success"], (rank, row)
    seconds = sorted(row["seconds"] for rows in soak for row in rows[BASELINE_STEP:])
    print(
        f"loop ring soak: {STEPS} steps at P = {STAGES} on {WORLD} ranks completed; step time "
        f"median {seconds[len(seconds) // 2]:.3f} s, max {seconds[-1]:.3f} s (after step "
        f"{BASELINE_STEP})"
    )


def test_soak_memory_is_flat(soak):
    """After step 10 the per-step peak and the end-of-step allocated bytes stay within +-1 % of
    their step-10 values on every rank."""
    lines, failures = [], []
    for rank, rows in enumerate(soak):
        base = rows[BASELINE_STEP - 1]
        tail = rows[BASELINE_STEP - 1 :]
        for field in ("peak", "allocated"):
            low = min(row[field] for row in tail)
            high = max(row[field] for row in tail)
            spread = (high - low) / base[field]
            lines.append(
                f"rank {rank} {field}: step {BASELINE_STEP} {base[field] / 2**20:.2f} MB, "
                f"min {low / 2**20:.2f} MB, max {high / 2**20:.2f} MB, spread {spread:.4%}"
            )
            if high > 1.01 * base[field] or low < 0.99 * base[field]:
                failures.append(lines[-1])
    print("loop ring soak memory:\n  " + "\n  ".join(lines))
    assert not failures, failures
