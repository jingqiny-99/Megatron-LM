# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Kimi-K3-style attention residuals for the hybrid loop (``loop_carrier='attn_res'``).

Every sublayer input is a per-token softmax over depth sources::

    h = sum_j alpha_j V_j,   alpha = softmax_j( (w * g) . V_j * rsqrt(mean(V_j^2) + eps) )

with ``w`` a zero-initialized pseudo-query and ``g`` a key RMSNorm weight (one pair per
sublayer, fp32). The sources are the prelude output ``e``, the sealed block sums of earlier
blocks and the running partial sum of the current block (``LoopAttnResHybridStack``). With
``w = 0`` the softmax is uniform and ``h`` is the mean of the sources.

Ported from NVIDIA/Megatron-LM PR #6840 ("Attention Residuals",
https://github.com/NVIDIA/Megatron-LM/pull/6840; source: https://github.com/jingqiny-99/Megatron-LM
branch jingqiny/attention-residuals-dev @ dc619f559, Apache-2.0, same copyright holder):
megatron/core/transformer/attention_residual.py (``_attn_res_fwd_math``,
``_attn_res_bwd_math``, ``_AttnResAggregation``, ``AttentionResidual``) and
megatron/core/models/hybrid/hybrid_block.py (``AttnResHybridLayer.forward``, the
``final_attn_res`` head; ported into ``loop/attn_res_stack.py``). Reference math: Kimi K3
(https://huggingface.co/moonshotai/Kimi-K3 @ f831ab66, modeling_kimi_linear.py,
``_apply_attn_res``).

Loop changes against the PR: a per-row source mask (per-sample exit: a masked source has weight
exactly 0 and receives an exactly-zero gradient), evaluation on one stacked tensor per token
chunk (a bounded transient; the RMS keys, logits and value gradients are a few kernels per
chunk, the source mix and ``<g, V_j>`` one multiply-add per source), a statistics
hook for the carrier diagnostics, eager only (no FLA / compile backend, no PP / VPP / MTP
bookkeeping), and a compute dtype of ``promote(values, fp32)`` (fp32 for bf16 / fp32 values,
float64 for float64 values, so the reference tests can run in float64). The products over the
hidden size are GEMVs (``mv``) or per-source multiply-adds, never GEMM / batched-GEMM calls:
cuBLAS may run fp32 GEMMs in TF32 (NGC containers start torch with
``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1``), which would round every source to a 10-bit mantissa
(about 5e-4 relative); GEMV and elementwise kernels have no TF32 mode, so the aggregation is
fp32 whatever the process-wide TF32 setting, as the PR's elementwise math was.
"""

from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import torch
from torch import Tensor, nn

from megatron.core.transformer.forward_replay import in_forward_replay
from megatron.core.transformer.loop.guards import is_block_boundary
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig

# Byte budget of the stacked compute-dtype values of one chunk (``[s_c, B, n, H]``). The
# backward holds about 1.5x this (the stacked activation-dtype copy is freed after the upcast)
# plus a few compute-dtype ``[s_c, B, H]`` buffers;
# a default that keeps one chunk at n <= 4, h <= 2048, B = 1, S = 4096.
ATTN_RES_CHUNK_BYTES: int = 256 * 2**20

StatsKey = Tuple[str, int, int]


def attn_res_block_offsets(core_symbols: Sequence[str], blocks_per_round: int) -> Tuple[int, ...]:
    """The core-sublayer offsets at which the ``blocks_per_round`` blocks of a round start.

    ``(0, C/k, 2C/k, ...)`` for a core of ``C`` sublayers. Raises ``ValueError`` (naming the
    offsets) unless ``k >= 1``, ``k`` divides ``C`` and every offset is an attention + MLP block
    boundary of the core (``guards.is_block_boundary``).
    """
    core = len(core_symbols)
    k = blocks_per_round
    if k < 1 or core % k:
        raise ValueError(
            f"loop_attn_res_blocks_per_round={k} does not cut the core of {core} sublayers into "
            "equal blocks"
        )
    offsets = tuple(index * (core // k) for index in range(k))
    off_boundary = [offset for offset in offsets if not is_block_boundary(core_symbols, offset)]
    if off_boundary:
        raise ValueError(
            f"loop_attn_res_blocks_per_round={k}: block offsets {list(offsets)} of the core "
            f"{''.join(core_symbols)!r}; {off_boundary} are not attention + MLP block boundaries"
        )
    return offsets


def history_lengths(depths_host: Sequence[int], blocks_per_round: int) -> Tuple[int, ...]:
    """Per-row history length ``1 + k * d_b`` (the prelude output plus the row's block sums)."""
    return tuple(1 + blocks_per_round * int(depth) for depth in depths_host)


def source_mask_for_lengths(
    n_hist: int, lengths_host: Optional[Sequence[int]], has_partial: bool, device: torch.device
) -> Optional[Tensor]:
    """The ``[n_hist + has_partial, B]`` bool source mask (True = the row reads the source).

    History entry ``i`` is valid for row ``b`` iff ``i < lengths_host[b]``; the partial (last
    row of the mask, when present) is always valid. Returns None (no mask: every source valid)
    when ``lengths_host`` is None or no row is shorter than ``n_hist``; the decision is made on
    host integers only.
    """
    if lengths_host is None or min(lengths_host) >= n_hist:
        return None
    lengths = torch.tensor(lengths_host, dtype=torch.int64)
    mask = torch.arange(n_hist, dtype=torch.int64)[:, None] < lengths[None, :]
    if has_partial:
        mask = torch.cat((mask, torch.ones(1, len(lengths_host), dtype=torch.bool)))
    return mask.to(device)


def attn_res_chunk_tokens(
    n: int, rows: int, hidden: int, itemsize: int = 4, budget: Optional[int] = None
) -> int:
    """Sequence positions per chunk: ``max(1, budget // (itemsize * n * rows * hidden))``
    (``budget`` defaults to ``ATTN_RES_CHUNK_BYTES``, read at call time)."""
    budget = ATTN_RES_CHUNK_BYTES if budget is None else budget
    return max(1, budget // max(1, itemsize * n * rows * hidden))


class AttnResStatsSink(Protocol):
    """Receives the masked softmax weights of one aggregation call (carrier diagnostics)."""

    def add(self, key: StatsKey, alpha: Tensor, row_weight: Optional[Tensor]) -> None:
        """``alpha``: ``[n, S, B]`` compute-dtype weights (exactly 0 on masked sources; a
        view, do not modify); ``row_weight``: ``[S, B]`` token weights (1 for the tokens the
        diagnostics count) or None (every token counts)."""


class AttnResStats:
    """The production sink: per key, the token-weighted sum of the weights ``[n]`` and the
    weighted token count, accumulated on device (no host sync)."""

    def __init__(self) -> None:
        self.sums: Dict[StatsKey, List[Tensor]] = {}

    def add(self, key: StatsKey, alpha: Tensor, row_weight: Optional[Tensor]) -> None:
        """Accumulate one call's weights under ``key``."""
        alpha = alpha.detach().float()
        if row_weight is None:
            weights = alpha.sum(dim=(1, 2))
            # A device fill (``torch.tensor(..., device=cuda)`` is a pageable copy that syncs).
            tokens = alpha.new_full((), float(alpha.shape[1] * alpha.shape[2]))
        else:
            row_weight = row_weight.detach().float()
            weights = (alpha * row_weight).sum(dim=(1, 2))
            tokens = row_weight.sum()
        entry = self.sums.get(key)
        if entry is None:
            self.sums[key] = [weights, tokens]
        else:
            entry[0] = entry[0] + weights
            entry[1] = entry[1] + tokens

    def merge(self, other: "AttnResStats") -> None:
        """Add another sink's sums into this one."""
        for key, (weights, tokens) in other.sums.items():
            entry = self.sums.get(key)
            if entry is None:
                self.sums[key] = [weights, tokens]
            else:
                entry[0] = entry[0] + weights
                entry[1] = entry[1] + tokens

    def take(self) -> Dict[StatsKey, Tuple[Tensor, Tensor]]:
        """Return ``{key: (weight sums [n], tokens)}`` and reset."""
        sums, self.sums = self.sums, {}
        return {key: (weights, tokens) for key, (weights, tokens) in sums.items()}


class _FanOut:
    """A sink forwarding every call to several sinks."""

    def __init__(self, sinks: Sequence[AttnResStatsSink]) -> None:
        self.sinks = tuple(sinks)

    def add(self, key: StatsKey, alpha: Tensor, row_weight: Optional[Tensor]) -> None:
        """Forward one call to every sink."""
        for sink in self.sinks:
            sink.add(key, alpha, row_weight)


def fan_out(*sinks: Optional[AttnResStatsSink]) -> Optional[AttnResStatsSink]:
    """One sink for the non-None ``sinks`` (None if there are none)."""
    present = [sink for sink in sinks if sink is not None]
    if not present:
        return None
    return present[0] if len(present) == 1 else _FanOut(present)


def _compute_dtype(values: Sequence[Tensor]) -> torch.dtype:
    return torch.promote_types(values[0].dtype, torch.float32)


def _stacked_chunk(
    values: Sequence[Tensor], start: int, end: int, dtype: torch.dtype, mask_t: Optional[Tensor]
) -> Tensor:
    """``[s_c, B, n, H]`` compute-dtype copy of the chunk ``[start, end)`` of every source,
    masked sources zeroed in place (the copy is owned)."""
    stacked = torch.stack([value[start:end] for value in values], dim=-2)
    if stacked.dtype != dtype:
        stacked = stacked.to(dtype)
    if mask_t is not None:
        stacked.masked_fill_(~mask_t[None, :, :, None], 0)
    return stacked


def _attn_res_fwd_math(
    pseudo_query: Tensor,
    key_norm_weight: Tensor,
    eps: float,
    values: Sequence[Tensor],
    source_mask: Optional[Tensor],
    chunk_tokens: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Forward math (PR #6840 ``_attn_res_fwd_math``, stacked and chunked).

    Returns ``(out [S, B, H] in the values' dtype, alpha, dots, rstds)``; the three statistics
    are ``[S, B, n]`` in the compute dtype (alpha exactly 0 on masked sources).
    """
    dtype = _compute_dtype(values)
    first = values[0]
    sequence, rows, hidden = first.shape
    n = len(values)
    q = (pseudo_query.to(dtype) * key_norm_weight.to(dtype)).contiguous()  # [H]
    mask_t = None if source_mask is None else source_mask.t()  # [B, n]
    out = torch.empty_like(first)
    alpha = torch.empty(sequence, rows, n, dtype=dtype, device=first.device)
    dots = torch.empty_like(alpha)
    rstds = torch.empty_like(alpha)
    if chunk_tokens is None:
        chunk_tokens = attn_res_chunk_tokens(n, rows, hidden, torch.finfo(dtype).bits // 8)
    for start in range(0, sequence, chunk_tokens):
        end = min(sequence, start + chunk_tokens)
        stacked = _stacked_chunk(values, start, end, dtype, mask_t)  # [s_c, B, n, H]
        # [s_c, B, n]: one GEMV (no TF32 mode).
        chunk_dots = stacked.reshape(-1, hidden).mv(q).reshape(stacked.shape[:-1])
        mean_sq = torch.linalg.vector_norm(stacked, dim=-1).square_() / hidden
        chunk_rstds = torch.rsqrt(mean_sq + eps)
        logits = chunk_dots * chunk_rstds
        if mask_t is not None:
            logits.masked_fill_(~mask_t[None], float("-inf"))
        chunk_alpha = torch.softmax(logits, dim=-1)
        # sum_j alpha_j V_j as n multiply-adds into one [s_c, B, H] accumulator (a batched
        # GEMM here could run in TF32).
        mixed = stacked[:, :, 0, :] * chunk_alpha[..., 0:1]
        for index in range(1, n):
            mixed.addcmul_(stacked[:, :, index, :], chunk_alpha[..., index : index + 1])
        out[start:end] = mixed
        alpha[start:end] = chunk_alpha
        dots[start:end] = chunk_dots
        rstds[start:end] = chunk_rstds
    return out, alpha, dots, rstds


def _attn_res_bwd_math(
    pseudo_query: Tensor,
    key_norm_weight: Tensor,
    alpha: Tensor,
    dots: Tensor,
    rstds: Tensor,
    grad_output: Tensor,
    values: Sequence[Tensor],
    source_mask: Optional[Tensor],
    chunk_tokens: Optional[int] = None,
) -> Tuple[Tensor, Tensor, List[Tensor]]:
    """Backward math (PR #6840 ``_attn_res_bwd_math``, stacked and chunked).

    Recomputes the compute-dtype values per chunk from the saved references. Per token,
    ``u_j = <g, V_j>``, ``dlogits = alpha (u - sum alpha u)``, ``ddots = dlogits rstd``,
    ``dmean_sq = -dlogits dots rstd^3 / 2`` and
    ``dV_j = alpha_j g + ddots_j q + dmean_sq_j (2 / H) V_j``; ``dq = sum_tokens ddots_j V_j``
    (chunks summed in order: deterministic), ``grad w = dq * g``, ``grad g = dq * w``. A masked
    source has ``alpha = 0`` and a zeroed recomputed value, so its gradient is exactly 0.

    Returns ``(grad pseudo_query, grad key_norm_weight, [grad V_j])``; every value gradient is
    its own tensor (no shared buffer).
    """
    dtype = alpha.dtype
    sequence, rows, hidden = values[0].shape
    n = len(values)
    q = (pseudo_query.to(dtype) * key_norm_weight.to(dtype)).contiguous()
    mask_t = None if source_mask is None else source_mask.t()
    grads = [torch.empty_like(value) for value in values]
    dq = torch.zeros(hidden, dtype=dtype, device=values[0].device)
    if chunk_tokens is None:
        chunk_tokens = attn_res_chunk_tokens(n, rows, hidden, torch.finfo(dtype).bits // 8)
    for start in range(0, sequence, chunk_tokens):
        end = min(sequence, start + chunk_tokens)
        stacked = _stacked_chunk(values, start, end, dtype, mask_t)  # [s_c, B, n, H]
        g = grad_output[start:end].to(dtype)  # [s_c, B, H]
        chunk_alpha = alpha[start:end]
        chunk_rstds = rstds[start:end]
        # u_j = <g, V_j> per source (multiply + reduce; a batched GEMM here could run in TF32).
        u = torch.stack(
            [(stacked[:, :, index, :] * g).sum(dim=-1) for index in range(n)], dim=-1
        )  # [s_c, B, n]
        dlogits = chunk_alpha * (u - (chunk_alpha * u).sum(dim=-1, keepdim=True))
        ddots = dlogits * chunk_rstds
        dmean_sq = dlogits * dots[start:end] * (-0.5) * chunk_rstds.pow(3)
        # dq += sum over the chunk's tokens of ddots_j V_j: one GEMV on the transposed stack
        # (no TF32 mode), before V is overwritten.
        dq = dq + stacked.reshape(-1, hidden).t().mv(ddots.reshape(-1))
        # dV in place of the owned stacked copy: (2/H) dmean_sq V + alpha g + ddots q.
        stacked.mul_((dmean_sq * (2.0 / hidden)).unsqueeze(-1))
        stacked.addcmul_(chunk_alpha.unsqueeze(-1), g.unsqueeze(-2))
        stacked.addcmul_(ddots.unsqueeze(-1), q)
        for index, grad in enumerate(grads):
            grad[start:end] = stacked[:, :, index, :]
    grad_query = (dq * key_norm_weight.to(dtype)).to(pseudo_query.dtype)
    grad_norm_weight = (dq * pseudo_query.to(dtype)).to(key_norm_weight.dtype)
    return grad_query, grad_norm_weight, grads


class _AttnResAggregation(torch.autograd.Function):
    """Eager depth-softmax aggregation with a memory-lean, recomputing backward (PR #6840).

    Saves references to the incoming values plus three ``[S, B, n]`` compute-dtype statistics
    (no hidden-size factor) and recomputes everything else in backward; a naive autograd
    implementation would keep about ``2 (n + 1)`` fp32 ``[S, B, H]`` intermediates per call.
    """

    @staticmethod
    def forward(ctx, pseudo_query, key_norm_weight, eps, source_mask, stats, *values):
        """Aggregate the sources. ``stats``: None or ``(sink, key, row_weight)``; the sink
        receives the masked weights once per forward (never in an activation replay)."""
        chunk_tokens = attn_res_chunk_tokens(
            len(values),
            values[0].shape[1],
            values[0].shape[-1],
            torch.finfo(_compute_dtype(values)).bits // 8,
        )
        out, alpha, dots, rstds = _attn_res_fwd_math(
            pseudo_query, key_norm_weight, eps, values, source_mask, chunk_tokens
        )
        if stats is not None and not in_forward_replay():
            sink, key, row_weight = stats
            sink.add(key, alpha.permute(2, 0, 1), row_weight)
        ctx.chunk_tokens = chunk_tokens
        ctx.has_mask = source_mask is not None
        saved = (pseudo_query, key_norm_weight, alpha, dots, rstds)
        if source_mask is not None:
            saved = saved + (source_mask,)
        ctx.save_for_backward(*saved, *values)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """Recompute the aggregation intermediates and return the input gradients."""
        saved = ctx.saved_tensors
        pseudo_query, key_norm_weight, alpha, dots, rstds = saved[:5]
        rest = saved[5:]
        source_mask = rest[0] if ctx.has_mask else None
        values = rest[1:] if ctx.has_mask else rest
        grad_query, grad_norm_weight, grad_values = _attn_res_bwd_math(
            pseudo_query,
            key_norm_weight,
            alpha,
            dots,
            rstds,
            grad_output,
            values,
            source_mask,
            ctx.chunk_tokens,
        )
        return (grad_query, grad_norm_weight, None, None, None, *grad_values)


def attn_res_aggregate(
    pseudo_query: Tensor,
    key_norm_weight: Tensor,
    eps: float,
    values: Sequence[Tensor],
    source_mask: Optional[Tensor] = None,
    stats: Optional[Tuple[AttnResStatsSink, StatsKey, Optional[Tensor]]] = None,
) -> Tensor:
    """The aggregation of ``values`` (``n >= 2`` sources of one shape and dtype) through
    ``_AttnResAggregation``; ``source_mask``: ``[n, B]`` bool (True = valid; every row needs
    at least one valid source)."""
    if len(values) < 2:
        raise ValueError("attn_res_aggregate needs at least two sources")
    shape, dtype = values[0].shape, values[0].dtype
    if any(value.shape != shape or value.dtype != dtype for value in values):
        raise ValueError("attention-residual sources must share one shape and dtype")
    if source_mask is not None and tuple(source_mask.shape) != (len(values), shape[1]):
        raise ValueError(
            f"source mask {tuple(source_mask.shape)} does not match {len(values)} sources of "
            f"batch {shape[1]}"
        )
    return _AttnResAggregation.apply(
        pseudo_query, key_norm_weight, eps, source_mask, stats, *values
    )


class LoopAttentionResidual(MegatronModule):
    """One sublayer's aggregation (PR #6840 ``AttentionResidual``, eager only).

    A zero-initialized pseudo-query ``[H]`` and a ones-initialized key-norm weight ``[H]``,
    both fp32 and marked ``keep_in_fp32``; ``eps = config.layernorm_epsilon``. One source
    returns it unchanged, plus a zero parameter term so data-parallel gradient accounting sees
    every parameter. The math is per token, so TP / EP need nothing (replicated parameters,
    identical inputs on every TP rank); sequence parallelism is refused by the loop.
    """

    def __init__(self, config: TransformerConfig, device: Optional[torch.device] = None) -> None:
        super().__init__(config)
        self.eps = config.layernorm_epsilon
        hidden = config.hidden_size
        # Zero init: uniform initial weights (the w = 0 oracle). No RNG draw.
        self.pseudo_query = mark_keep_in_fp32(
            nn.Parameter(torch.zeros(hidden, dtype=torch.float32, device=device))
        )
        self.key_norm_weight = mark_keep_in_fp32(
            nn.Parameter(torch.ones(hidden, dtype=torch.float32, device=device))
        )

    def forward(
        self,
        values: Sequence[Tensor],
        source_mask: Optional[Tensor] = None,
        stats: Optional[Tuple[AttnResStatsSink, StatsKey, Optional[Tensor]]] = None,
    ) -> Tensor:
        """Aggregate the sources into the sublayer input (``[S, B, H]``, the values' dtype).

        Args:
            values: The sources, oldest first (the partial last when present).
            source_mask: ``[n, B]`` bool (True = valid) or None (every source valid).
            stats: ``(sink, key, row_weight)`` for the carrier diagnostics, or None.
        """
        if not values:
            raise ValueError("LoopAttentionResidual needs at least one source")
        if len(values) == 1:
            if stats is not None and not in_forward_replay():
                sink, key, row_weight = stats
                ones = torch.ones(1, *values[0].shape[:2], device=values[0].device)
                sink.add(key, ones, row_weight)
            zero = (self.pseudo_query.sum() + self.key_norm_weight.sum()) * 0.0
            return values[0] + zero.to(values[0].dtype)
        return attn_res_aggregate(
            self.pseudo_query, self.key_norm_weight, self.eps, values, source_mask, stats
        )
