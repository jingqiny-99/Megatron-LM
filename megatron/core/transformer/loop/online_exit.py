# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sequence-level halting from the gated output of a completed recurrent pass."""

import math

import torch
import torch.distributed as dist


def _project(token):
    """Fixed unit-norm projection of an RMS-normalized token, in FP32."""
    token = token.float()
    width = token.shape[-1]
    # Sample up to 64 coordinates across the residual width. Alternating signs
    # define a reproducible probe without a learned head or an RNG stream.
    stride = max(1, math.ceil(width / 64))
    features = token[..., ::stride]
    projection = features[..., ::2].sum(-1) - features[..., 1::2].sum(-1)
    rms = token.square().mean(-1).clamp_min(1e-12).sqrt()
    return projection / (math.sqrt(features.shape[-1]) * rms)


@torch.no_grad()
def round_exit_score(state, valid=None, cp_group=None):
    """Score the last valid global token of each SBHD row on zigzag CP shards.

    The return value is identical within a sequence's CP group. A wholly
    padded row scores zero. Only the selected token is cast to FP32.
    """
    if state.ndim != 3 or min(state.shape) == 0:
        raise ValueError("online exit expects a nonempty [S_local, B, H] residual")
    sequence, batch, _ = state.shape
    size = cp_group.size() if cp_group is not None else 1
    if size > 1 and sequence % 2:
        raise ValueError("online exit needs equal front and back chunks for zigzag CP")

    if valid is None:
        # CP rank zero owns global tokens [0, chunk) and [S-chunk, S).
        score = _project(state[-1])
        if size > 1:
            dist.broadcast(score, src=dist.get_global_rank(cp_group, 0), group=cp_group)
        return score
    if valid.shape != (batch, sequence) or valid.dtype != torch.bool:
        raise ValueError("online exit valid must be boolean [B, S_local]")
    positions = torch.arange(sequence, device=state.device)
    last = torch.where(valid, positions, -1).amax(dim=-1)
    rows = torch.arange(batch, device=state.device)
    score = _project(state[last.clamp_min(0), rows])
    if size == 1:
        return torch.where(last >= 0, score, 0.0)

    rank = cp_group.rank()
    half = sequence // 2
    global_last = torch.where(last < half, last + rank * half, last + (2 * size - 2 - rank) * half)
    global_last = torch.where(last >= 0, global_last, -1)
    selected = global_last.clone()
    dist.all_reduce(selected, op=dist.ReduceOp.MAX, group=cp_group)
    score = torch.where((selected >= 0) & (global_last == selected), score, 0.0)
    dist.all_reduce(score, group=cp_group)
    return score


@torch.no_grad()
def online_exit_decision(config, state, round_number, *, valid=None, cp_group=None, return_score=False):
    """Stop at/below the round's threshold, or unconditionally at the cap."""
    if isinstance(round_number, int):
        if not 1 <= round_number <= config.loop_iterations:
            raise ValueError("online exit round lies outside the configured loop")
        if round_number == config.loop_iterations and not return_score:
            return torch.ones(state.shape[1], dtype=torch.bool, device=state.device)
    score = round_exit_score(state, valid=valid, cp_group=cp_group)
    if isinstance(round_number, int):
        stop = (torch.ones_like(score, dtype=torch.bool) if round_number == config.loop_iterations
                else score <= config.loop_exit_thresholds[round_number - 1])
    else:
        # Cell programs may present the round as a device scalar.
        stop = (round_number == config.loop_iterations).expand_as(score)
        for visit, threshold in enumerate(config.loop_exit_thresholds, start=1):
            stop = stop | ((round_number == visit) & (score <= threshold))
    return (stop, score) if return_score else stop
