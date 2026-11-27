# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""One reporting rule for loop metrics, shared by the eager and the RSBF schedules.

Every loop metric of a microbatch (eager) or a sample (RSBF) is reported as
``[value summed over the metric's population, population size]``. ``train_step`` sums these
pairs over microbatches and data-parallel ranks and logs ``sum / weight``, so the logged value
is the population mean in both schedules:

* the HybridStack / RSBF metrics (``loop depth`` per row, ``loop ... flops per token``, the
  token-layer counts) are defined over the **issued** tokens of the microbatch;
* the MoR metrics (``loop depth`` = executed rounds per valid token, ``depth fraction i``) are
  defined over the **valid** (non-padding) tokens; the loop decoders expose those population
  sizes as ``loop_metric_populations``.

The token-layer counts (``LOOP_TOKEN_COUNT_METRICS``) are already totals over the microbatch
(``loop_metrics`` of ``LoopHybridStack`` / ``LoopTransformerBlock``); every other loop metric
is a mean over its population and is multiplied by the population size. Loop **losses** (MoR
router losses) are loss terms, not metrics: they stay weighted by the loss-token count.
"""

from contextlib import contextmanager
from typing import Mapping, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core.utils import nvtx_profiling_enabled

# Loop metrics whose module value is a microbatch total (token x sublayer counts), not a mean.
LOOP_TOKEN_COUNT_METRICS = frozenset({"loop active token layers", "loop backward token layers"})


@contextmanager
def loop_range(name: str):
    """``torch.profiler.record_function(name)`` that is also the NVTX range ``name`` while
    Megatron's NVTX ranges are on (``--nvtx-ranges``: the nsys-profiled steps only;
    ``configure_nvtx_profiling``), and nothing more otherwise. No tensor op.

    RSBF names cell phases and backward items by sample and round.
    The hybrid loop names its ranges ``loop_hybrid/mb={m}`` (one forward of a microbatch),
    ``.../prelude`` and ``.../coda`` (block = the edge checkpoints), ``.../loop={r}`` (round r),
    ``.../block={k}`` (one block checkpoint; inside the checkpointed function, so its recompute
    in the backward opens it again on the autograd thread) and ``.../layer={i}`` (sublayer i).

    The NVTX range is a start / end range, not a push / pop range: a non-reentrant checkpoint's
    recompute stops early by raising through the checkpointed function
    (``torch.utils.checkpoint`` early stop), which skips the pops of Megatron's push / pop ranges
    inside it (``nvtx_range_push``; their stack keeps the stale entries). A start / end range
    ends where it was opened whatever is left on that stack, and touches neither stack.
    """
    handle = torch.cuda.nvtx.range_start(name) if nvtx_profiling_enabled() else None
    try:
        with torch.profiler.record_function(name):
            yield
    finally:
        if handle is not None:
            torch.cuda.nvtx.range_end(handle)


def loop_metric_entry(name: str, metric: Tensor, *, population: Union[int, Tensor]) -> Tensor:
    """The ``[total, population]`` report entry (fp32) of one loop metric of one microbatch.

    Args:
        name: The metric's report key.
        metric: The module's value: a microbatch total for ``LOOP_TOKEN_COUNT_METRICS``, a
            mean over the population for every other loop metric.
        population: The number of tokens the metric is defined over (issued or valid tokens,
            see the module docstring); a host int or a device scalar.
    """
    metric = metric.detach().to(torch.float32)
    if isinstance(population, Tensor):
        population = population.to(device=metric.device, dtype=torch.float32)
    else:
        # A device fill: ``torch.as_tensor(<host number>, device=cuda)`` is a pageable copy that
        # synchronizes the stream (once per metric and microbatch).
        population = metric.new_full((), float(population))
    total = metric if name in LOOP_TOKEN_COUNT_METRICS else metric * population
    return torch.stack((total.reshape(()), population.reshape(())))


def loop_metric_report(
    loop_metrics: Optional[Mapping[str, Tensor]],
    *,
    issued_tokens: int,
    populations: Optional[Mapping[str, Union[int, Tensor]]] = None,
) -> dict:
    """The report entries of a loop decoder's ``loop_metrics`` for one microbatch.

    Args:
        loop_metrics: The decoder's ``loop_metrics`` captured for this microbatch (or None).
        issued_tokens: The microbatch's issued token count (the per-token loss tensor's numel).
        populations: The decoder's ``loop_metric_populations`` captured for this microbatch:
            the population of each metric not defined over the issued tokens.
    """
    populations = populations or {}
    return {
        name: loop_metric_entry(name, metric, population=populations.get(name, issued_tokens))
        for name, metric in (loop_metrics or {}).items()
    }


def stream_mean_rms(
    x: Tensor, num_streams: int, valid: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """The carrier diagnostics' hidden-state norm: ``(rms, tokens)``.

    ``rms`` is the root mean square, over the ``valid`` tokens and the hidden dimension, of the
    **stream mean** ``(1 / N) sum_s x_s`` of ``x`` (``[S, B, N * h]``, the streams as contiguous
    ``h`` chunks as ``HyperConnectionModule.input_expand`` lays them out; ``N = 1`` is the plain
    residual stream). ``valid``: ``[S, B]`` bool (None: every token). ``tokens`` is the valid
    token count (a device scalar), the metric's population for ``loop_metric_report``; with no
    valid token ``rms`` is 0. FP32, no host sync.

    The logged ``loop state rms ...`` value is ``loop_metric_report``'s population-weighted mean
    of these per-microbatch RMS values over the microbatches, data-parallel ranks and log
    interval, not the RMS over all their tokens (at most that, by Jensen; equal when the
    microbatch values agree). Every variant logs the same quantity, so the variants compare.
    """
    x = x.detach().float()
    if num_streams > 1:
        x = x.reshape(*x.shape[:-1], num_streams, x.shape[-1] // num_streams).mean(-2)
    mean_square = x.square().mean(-1)  # [S, B]
    if valid is None:
        # A device fill (``torch.tensor(..., device=cuda)`` is a pageable copy that syncs).
        tokens = mean_square.new_full((), float(mean_square.numel()))
        total = mean_square.sum()
    else:
        weight = valid.to(torch.float32)
        tokens = weight.sum()
        total = (mean_square * weight).sum()
    return (total / tokens.clamp_min(1.0)).sqrt(), tokens
