# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# The online-softmax forward below is adapted from flash-linear-attention.
# Its MIT license is reproduced in docs/developer/attnres_forward_projection_primitive.md.

"""Experimental forward caches with the complete, unmodified FLA consumer VJP.

This standalone primitive deliberately has no model, pipeline or optimizer integration.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl
from fla.ops.attnres.fused import (
    _build_ptr_table,
    attnres_fwd_kernel,
    fused_attnres_bwd,
    fused_attnres_fwd,
)
from fla.ops.utils.op import exp


@dataclass(frozen=True)
class ForwardSourceCache:
    """One consumer's dot product and a source's shared inverse RMS.

    Bindings describe actual local tensors, not a transferable PP identity. Graph
    replay must execute producer and consumer on their recorded static surfaces.
    """

    dot: torch.Tensor
    rstd: torch.Tensor
    source: torch.Tensor
    query: torch.Tensor
    norm_weight: torch.Tensor
    source_binding: tuple
    query_binding: tuple
    norm_binding: tuple
    dot_binding: tuple
    rstd_binding: tuple
    eps: float


def _binding(tensor):
    return (
        tensor.data_ptr(),
        tensor._version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _validate_value(value):
    if not value.is_cuda or value.dtype != torch.bfloat16:
        raise ValueError("forward projection requires CUDA BF16 sources")
    if value.ndim < 2 or value.numel() == 0 or not value.is_contiguous():
        raise ValueError("sources must be nonempty contiguous [..., hidden] tensors")
    if value.data_ptr() % 16:
        raise ValueError("sources must have 16-byte aligned storage")


def _validate_eps(eps):
    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")


@triton.jit
def _project_source_kernel(
    V,
    Q,
    W,
    Dot,
    Rstd,
    N: tl.constexpr,
    D: tl.constexpr,
    QCOUNT: tl.constexpr,
    EPS: tl.constexpr,
    BD: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BD)
    v = tl.load(V + token * D + d, d < D, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(v * v, axis=0) / D + EPS)
    tl.store(Rstd + token, r)
    for row in range(QCOUNT):
        qw = tl.load(Q + row * D + d, d < D, other=0.0).to(tl.float32) * tl.load(
            W + row * D + d, d < D, other=0.0
        ).to(tl.float32)
        dot = tl.sum(v * qw, axis=0)
        tl.store(Dot + row * N + token, dot)


@torch.no_grad()
def project_forward_source(value, queries, norm_weights, *, eps=1e-6):
    """Project one source for raw FP32 query/norm banks shaped [consumers, H].

    Returned tensors have no autograd edge. The paired consumer function supplies
    the complete derivatives with respect to canonical queries, norms and values.
    """
    _validate_value(value)
    _validate_eps(eps)
    expected = (queries.shape[0], value.shape[-1]) if queries.ndim == 2 else None
    for bank in (queries, norm_weights):
        if (
            bank.shape != expected
            or bank.dtype != torch.float32
            or bank.device != value.device
            or not bank.is_contiguous()
        ):
            raise ValueError("query and norm banks must be contiguous CUDA FP32 [Q, H]")
    count = queries.shape[0]
    if count == 0:
        return ()
    shape = value.shape[:-1]
    dot = torch.empty((count, *shape), dtype=torch.float32, device=value.device)
    rstd = torch.empty(shape, dtype=torch.float32, device=value.device)
    _project_source_kernel[(value.numel() // value.shape[-1],)](
        value,
        queries,
        norm_weights,
        dot,
        rstd,
        N=value.numel() // value.shape[-1],
        D=value.shape[-1],
        QCOUNT=count,
        EPS=eps,
        BD=triton.next_power_of_2(value.shape[-1]),
        num_warps=4,
    )
    result = []
    for row in range(count):
        query, norm, row_dot = queries[row], norm_weights[row], dot[row]
        result.append(
            ForwardSourceCache(
                row_dot,
                rstd,
                value,
                query,
                norm,
                _binding(value),
                _binding(query),
                _binding(norm),
                _binding(row_dot),
                _binding(rstd),
                eps,
            )
        )
    return tuple(result)


@triton.jit(do_not_specialize=["L"])
def _cached_attnres_fwd_kernel(
    q,
    res,
    w,
    dots,
    norms,
    o,
    rstd,
    logit,
    lse,
    N,
    L,
    L2: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    CACHE_MASK: tl.constexpr,
    LOCAL_INDEX: tl.constexpr,
    BL: tl.constexpr,
    BD: tl.constexpr,
):
    # Preserve FLA's source order, BL, reduction shapes and online-softmax expressions.
    i_n = tl.program_id(0).to(tl.int64)
    o_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, BD), BD), BD)
    m_d = o_d < D
    b_qw = tl.load(q + o_d, mask=m_d, other=0.0).to(tl.float32) * tl.load(
        w + o_d, mask=m_d, other=0.0
    ).to(tl.float32)
    if LOCAL_INDEX >= 0:
        local_v = tl.load(res[LOCAL_INDEX] + i_n * D + o_d, m_d, other=0.0).to(tl.float32)
        local_rstd = tl.rsqrt(tl.sum(local_v * local_v, axis=0) / D + eps)
        local_dot = tl.sum(local_v * b_qw, axis=0)
    b_m = tl.full([], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([], dtype=tl.float32)
    b_o = tl.zeros([BD], dtype=tl.float32)
    for i_l in range(tl.cdiv(L, BL)):
        o_l = i_l * BL + tl.arange(0, BL)
        m_l = o_l < L
        p_v = res[0] + o_l * 0
        p_dot = dots[0] + o_l * 0
        p_norm = norms[0] + o_l * 0
        cached = tl.full([BL], False, tl.int1)
        for i in tl.static_range(L2):
            if i > 0:
                p_v = tl.where(o_l == i, res[i], p_v)
                p_dot = tl.where(o_l == i, dots[i], p_dot)
                p_norm = tl.where(o_l == i, norms[i], p_norm)
            if CACHE_MASK[i]:
                cached = cached | (o_l == i)
        p_v = tl.multiple_of(p_v, 16)
        b_v = tl.load(
            tl.multiple_of(p_v[:, None] + (i_n * D + o_d[None, :]), (1, 16)),
            mask=m_l[:, None] & m_d[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        # A mixed BL tile retains the original reduction layout. Fully cached
        # tiles skip both reductions. This intentionally favors parity first.
        if LOCAL_INDEX >= 0:
            b_rstd = tl.full([BL], 0.0, tl.float32) + local_rstd
            b_dot = tl.full([BL], 0.0, tl.float32) + local_dot
        elif tl.sum((m_l & ~cached).to(tl.int32), axis=0) > 0:
            b_rstd = tl.rsqrt(tl.sum(b_v * b_v, axis=1) / D + eps)
            b_dot = tl.sum(b_v * b_qw[None, :], axis=1)
        else:
            b_rstd = tl.full([BL], 0.0, tl.float32)
            b_dot = tl.full([BL], 0.0, tl.float32)
        b_cached_dot = tl.load(p_dot + i_n, mask=m_l & cached, other=0.0)
        b_cached_rstd = tl.load(p_norm + i_n, mask=m_l & cached, other=0.0)
        b_rstd = tl.where(cached, b_cached_rstd, b_rstd)
        b_dot = tl.where(cached, b_cached_dot, b_dot)
        b_logit = b_dot * b_rstd
        b_s = tl.where(m_l, b_logit * 1.0, float('-inf'))
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, axis=0)), b_m
        b_r = exp(b_mp - b_m)
        b_p = exp(b_s - b_m)
        b_acc = b_acc * b_r + tl.sum(b_p, axis=0)
        b_o = b_o * b_r + tl.sum(b_p[:, None] * b_v, axis=0)
        p_rstd = tl.make_block_ptr(rstd + i_n, (L,), (N,), (i_l * BL,), (BL,), (0,))
        p_logit = tl.make_block_ptr(logit + i_n, (L,), (N,), (i_l * BL,), (BL,), (0,))
        tl.store(p_rstd, b_rstd.to(rstd.dtype.element_ty), boundary_check=(0,))
        tl.store(p_logit, b_logit.to(logit.dtype.element_ty), boundary_check=(0,))
    tl.store(lse + i_n, b_m + tl.log(b_acc))
    b_o = b_o / b_acc
    p_o = tl.make_block_ptr(o + i_n * D, (D,), (1,), (0,), (BD,), (0,))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


# Filled only by a real original-FLA launch, never by an independent tuner.
_FLA_LAUNCH_CONFIGS = {}


def _launch_config(query, norm_weight, values, eps):
    key = (
        values[0].device,
        query.dtype,
        norm_weight.dtype,
        values[0].dtype,
        max(8, triton.next_power_of_2(len(values))),
        values[0].shape[-1],
    )
    if key not in _FLA_LAUNCH_CONFIGS:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm original FLA and the paired primitive before graph capture")
        with torch.no_grad():
            fused_attnres_fwd(
                query, values, _build_ptr_table(values), norm_weight, None, eps, 1.0, 1
            )
        config = attnres_fwd_kernel.best_config
        if set(config.kwargs) != {"BL"}:
            raise RuntimeError("unsupported installed FLA forward launch configuration")
        _FLA_LAUNCH_CONFIGS[key] = (config.kwargs["BL"], config.num_warps, config.num_stages)
    return _FLA_LAUNCH_CONFIGS[key]


def _validate_consumer(query, norm_weight, values, caches, eps):
    _validate_eps(eps)
    if not values:
        raise ValueError("values must contain at least one source")
    first = values[0]
    _validate_value(first)
    for param in (query, norm_weight):
        if (
            param.shape != (first.shape[-1],)
            or param.dtype != torch.float32
            or param.device != first.device
            or not param.is_contiguous()
        ):
            raise ValueError("consumer query and norm must be contiguous CUDA FP32 [H]")
    if len(caches) != len(values):
        raise ValueError("caches must have one entry per source, preserving source order")
    for value, cache in zip(values, caches):
        _validate_value(value)
        if value.shape != first.shape or value.device != first.device:
            raise ValueError("all source shapes and devices must match")
        if cache is None:
            continue
        if not isinstance(cache, ForwardSourceCache) or cache.eps != eps:
            raise ValueError("invalid cache or mismatched eps")
        for tensor, binding in (
            (value, cache.source_binding),
            (query, cache.query_binding),
            (norm_weight, cache.norm_binding),
            (cache.dot, cache.dot_binding),
            (cache.rstd, cache.rstd_binding),
        ):
            if _binding(tensor) != binding:
                raise ValueError("stale or mismatched forward cache tensor binding")
        for stat in (cache.dot, cache.rstd):
            if (
                stat.shape != first.shape[:-1]
                or stat.dtype != torch.float32
                or stat.device != first.device
                or not stat.is_contiguous()
                or stat.requires_grad
            ):
                raise ValueError("cache statistics must be contiguous nondifferentiable FP32")


def _cached_forward(query, norm_weight, values, caches, eps, config):
    first = values[0]
    length = len(values)
    l2 = max(8, triton.next_power_of_2(length))
    # Local rows use a never-dereferenced pointer; no CPU/H2D pointer table exists.
    dots = tuple(query if cache is None else cache.dot for cache in caches)
    norms = tuple(query if cache is None else cache.rstd for cache in caches)
    dots += (dots[0],) * (l2 - length)
    norms += (norms[0],) * (l2 - length)
    mask = tuple(cache is not None for cache in caches) + (False,) * (l2 - length)
    output = torch.empty_like(first)
    rstd = torch.empty((length, *first.shape[:-1]), device=first.device, dtype=torch.float32)
    logit = torch.empty_like(rstd)
    lse = torch.empty(first.shape[:-1], device=first.device, dtype=torch.float32)
    bl, warps, stages = config
    local_rows = [i for i, cache in enumerate(caches) if cache is None]
    local_index = local_rows[0] if len(local_rows) == 1 and length > 1 else -1
    _cached_attnres_fwd_kernel[(first.numel() // first.shape[-1],)](
        query,
        _build_ptr_table(values),
        norm_weight,
        dots,
        norms,
        output,
        rstd,
        logit,
        lse,
        first.numel() // first.shape[-1],
        length,
        L2=l2,
        D=first.shape[-1],
        eps=eps,
        CACHE_MASK=mask,
        LOCAL_INDEX=local_index,
        BL=bl,
        BD=triton.next_power_of_2(first.shape[-1]),
        num_warps=warps,
        num_stages=stages,
    )
    return output, rstd, logit, lse


def forward_cache_launch_info(query, norm_weight, values, caches=None, *, eps=1e-6):
    """Return the original FLA launch choice and the primitive's reduction path."""
    values = tuple(values)
    caches = tuple(caches) if caches is not None else (None,) * len(values)
    _validate_consumer(query, norm_weight, values, caches, eps)
    bl, warps, stages = _launch_config(query, norm_weight, values, eps)
    local_count = sum(cache is None for cache in caches)
    return {
        "BL": bl,
        "num_warps": warps,
        "num_stages": stages,
        "local_sources": local_count,
        "single_local_reduction": local_count == 1 and len(values) > 1,
    }


class _ForwardCachedAttnRes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, norm_weight, caches, eps, config, *values):
        """Save original values and FLA-compatible statistics for the complete VJP."""
        output, rstd, logit, lse = _cached_forward(query, norm_weight, values, caches, eps, config)
        ctx.save_for_backward(query, norm_weight, rstd, logit, lse, *values)
        ctx.res = _build_ptr_table(values)
        ctx.eps = eps
        ctx.mark_non_differentiable(rstd, logit, lse)
        return output, rstd, logit, lse

    @staticmethod
    def backward(ctx, grad_output, _grad_rstd, _grad_logit, _grad_lse):
        """Delegate all input derivatives to the original installed FLA backward."""
        query, norm_weight, rstd, logit, lse, *values = ctx.saved_tensors
        dvs, dq, dw, _ = fused_attnres_bwd(
            do=grad_output.contiguous(),
            q=query,
            residuals=values,
            res=ctx.res,
            w=norm_weight,
            ow=None,
            o_pre=None,
            rstd=rstd,
            logit=logit,
            lse=lse,
            eps=ctx.eps,
            scale=1.0,
            checkpoint_level=1,
        )
        return dq, dw, None, None, None, *dvs


def aggregate_with_forward_cache(
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    values: Sequence[torch.Tensor],
    caches: Optional[Sequence[Optional[ForwardSourceCache]]] = None,
    *,
    eps: float = 1e-6,
    return_stats: bool = False,
):
    """Aggregate cached and local sources with complete original consumer gradients.

    Optional statistics are (rstd, logit, lse), all FP32 and nondifferentiable.
    Only scale=1, no output norm, and FLA checkpoint level 1 are supported.
    """
    values = tuple(values)
    caches = tuple(caches) if caches is not None else (None,) * len(values)
    _validate_consumer(query, norm_weight, values, caches, eps)
    config = _launch_config(query, norm_weight, values, eps)
    outputs = _ForwardCachedAttnRes.apply(query, norm_weight, caches, eps, config, *values)
    return outputs if return_stats else outputs[0]
