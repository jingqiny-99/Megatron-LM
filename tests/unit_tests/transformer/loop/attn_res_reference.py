# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Naive references for the loop's attention-residual carrier (``loop/attn_res.py``,
``loop/attn_res_stack.py``): plain autograd, no custom Function, no chunking.

``native_attn_res`` is ported from NVIDIA/Megatron-LM PR #6840
(https://github.com/NVIDIA/Megatron-LM/pull/6840; source: https://github.com/jingqiny-99/Megatron-LM
branch jingqiny/attention-residuals-dev @ dc619f559, Apache-2.0,
tests/unit_tests/transformer/test_kimi_k3_attnres_mtp.py ``_native_attn_res``), extended with a
per-row source mask and a float64 path. ``reference_k3_loop`` composes the K3 loop by hand from
the stack's phases: every sublayer is called normally and its delta taken as
``layer(h) - h`` (exact in float64; in fp32 within rounding).
"""

import dataclasses
from typing import List, Optional, Sequence

import torch
from torch import Tensor


def native_attn_res(
    values: Sequence[Tensor],
    query: Tensor,
    norm_weight: Tensor,
    eps: float,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """``sum_j softmax_j(keys_j . (g * w)) V_j`` with ``keys = V rsqrt(mean(V^2) + eps)``.

    ``mask``: ``[n, B]`` bool (True = valid) or None; masked scores are ``-inf``. Computed in
    ``promote(values, fp32)``; returns the values' dtype.
    """
    dtype = torch.promote_types(values[0].dtype, torch.float32)
    stacked = torch.stack([value.to(dtype) for value in values])  # [n, S, B, H]
    keys = stacked * torch.rsqrt(stacked.square().mean(-1, keepdim=True) + eps)
    scores = (keys * (norm_weight.to(dtype) * query.to(dtype))).sum(-1)  # [n, S, B]
    if mask is not None:
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
    weights = scores.softmax(dim=0)
    return (stacked * weights.unsqueeze(-1)).sum(0).to(values[0].dtype)


def _mask(n_hist: int, lengths, has_partial: bool, device) -> Optional[Tensor]:
    if lengths is None or min(lengths) >= n_hist:
        return None
    rows = torch.tensor(lengths, device=device)
    mask = torch.arange(n_hist, device=device)[:, None] < rows[None, :]
    if has_partial:
        mask = torch.cat((mask, torch.ones(1, len(lengths), dtype=torch.bool, device=device)))
    return mask


def reference_k3_loop(stack, x: Tensor, rotary_pos_emb, depths: Optional[Sequence[int]] = None):
    """The K3 carrier of ``stack`` (a ``LoopAttnResHybridStack``) composed by hand.

    Uniform depth ``loop_iterations`` (``depths`` None) or per-row ``depths``. Returns
    ``(output after the final norm, [n sources of every aggregation call, in call order])``.
    """
    hidden, ctx = stack._prepare_forward(
        x,
        None,
        None,
        rotary_pos_emb,
        packed_seq_params=None,
        padding_mask=None,
        input_ids=None,
        packed_seq_params_by_layout=None,
        cp_layout_plan=None,
    )
    roles = stack.loop_roles
    k = stack.config.loop_attn_res_blocks_per_round
    eps = stack.config.layernorm_epsilon
    for index in range(roles.core_start):
        hidden = stack._run_layer(index, hidden, ctx)
    sequence, batch = hidden.shape[:2]
    rounds = stack.config.loop_iterations if depths is None else max(depths)
    lengths = None if depths is None else [1 + k * d for d in depths]
    history: List[Tensor] = [hidden]
    counts: List[int] = []
    block = roles.core // k

    def entry(index, module, partial, call_ctx):
        values = history if partial is None else [*history, partial]
        counts.append(len(values))
        mask = _mask(len(history), lengths, partial is not None, x.device)
        aggregated = native_attn_res(
            values, module.pseudo_query, module.key_norm_weight, module.eps, mask
        )
        delta = stack._run_layer(index, aggregated, call_ctx) - aggregated
        return delta if partial is None else partial + delta

    for step in range(rounds):
        round_ctx = ctx
        if depths is not None and step + 1 > min(depths):
            exited = torch.tensor([d < step + 1 for d in depths], device=x.device)[:, None]
            round_ctx = dataclasses.replace(ctx, padding_mask=exited.repeat(1, sequence))
        partial = None
        for offset in range(roles.core):
            if offset and offset % block == 0:
                history.append(partial)
                partial = None
            partial = entry(
                roles.core_start + offset, stack.loop_attn_res[offset], partial, round_ctx
            )
        history.append(partial)
    partial = None
    for offset in range(roles.coda):
        partial = entry(
            roles.core_end + offset, stack.loop_attn_res[roles.core + offset], partial, ctx
        )
    values = history if partial is None else [*history, partial]
    counts.append(len(values))
    head = stack.loop_attn_res_head
    output = native_attn_res(
        values,
        head.pseudo_query,
        head.key_norm_weight,
        eps,
        _mask(len(history), lengths, partial is not None, x.device),
    )
    return stack._finalize_forward(output), counts
