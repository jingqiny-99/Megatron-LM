# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Step-level accounting of actual ordinary-DP online exits."""

import json

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.training.loop_checkpoint import loop_hybrid_model
from megatron.training.loop_rsbf_planning import (
    get_loop_exit_histogram_log,
    loop_exit_histogram_phase,
)
from megatron.training.utils import is_last_rank


def record_online_step(args, model, iteration, first_sample, samples):
    """Gather one copy per DP replica along each CP lane, preserving sampler order."""
    decoder = loop_hybrid_model(model).decoder
    records = decoder.take_online_exit_records()
    group = parallel_state.get_data_parallel_group(with_context_parallel=False)
    replicas = group.size()
    if len(records) * replicas != samples:
        raise RuntimeError(f"{len(records)} local online rows x {replicas} DP != {samples} samples")

    rounds = args.loop_iterations
    device = next(decoder.parameters()).device
    local = torch.full((len(records), rounds), torch.nan, device=device)
    local[:, 0] = torch.tensor([depth for depth, _ in records], device=device)
    for row, (_, scores) in enumerate(records):
        if scores:
            local[row, 1 : 1 + len(scores)] = torch.stack(scores)
    gathered = local.new_empty((samples, rounds))
    dist.all_gather_into_tensor(gathered, local.contiguous(), group=group)
    # The sequential MBS1 sampler distributes each consecutive group of DP rows across replicas.
    rows = gathered.view(replicas, len(records), rounds).transpose(0, 1).reshape(samples, rounds)
    rows = rows.tolist()
    counts = [0] * rounds
    decisions = []
    report = is_last_rank()
    for offset, (depth, *scores) in enumerate(rows):
        depth = int(depth)
        counts[depth - 1] += 1
        if report:
            decisions.append(
                {
                    "sample_id": first_sample + offset,
                    "rounds": [
                        {
                            "round": visit,
                            "stop": visit == depth,
                            "score": scores[visit - 1] if visit < rounds else None,
                        }
                        for visit in range(1, depth + 1)
                    ],
                }
            )
    log = get_loop_exit_histogram_log(args)
    log.record(iteration, counts, samples, None, loop_exit_histogram_phase(args, samples, waves=1))
    args._loop_exit_counts = counts
    args._loop_padding = None
    args._loop_exit_counts_interval = log.interval()
    if report:
        print(
            "loop online decisions "
            + json.dumps(
                {
                    "iteration": iteration,
                    "sample_id_kind": "global_sampler_position",
                    "decisions": decisions,
                },
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )
    return counts
