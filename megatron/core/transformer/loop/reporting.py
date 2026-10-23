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

from typing import Mapping, Optional, Union

import torch
from torch import Tensor

# Loop metrics whose module value is a microbatch total (token x sublayer counts), not a mean.
LOOP_TOKEN_COUNT_METRICS = frozenset({"loop active token layers", "loop backward token layers"})


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
    population = torch.as_tensor(population, device=metric.device).to(torch.float32)
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
