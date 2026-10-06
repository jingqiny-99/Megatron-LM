# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Source-owned AttnRes projections and consumers, with explicit FP32 math.

The effective query is the caller-owned FP32 product of pseudo-query and
RMSNorm weight. The source mask stops only source derivatives, never query
derivatives. CUDA execution requires Triton; CPU execution uses the explicit
Torch implementation. Neither path retries a failed kernel with another backend.
"""

import math
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def _backend(value: Tensor, backend: str | None) -> str:
    selected = backend or ("triton" if value.is_cuda else "torch")
    if selected not in ("torch", "triton"):
        raise ValueError("backend must be 'torch' or 'triton'")
    if selected == "triton" and (not value.is_cuda or triton is None):
        raise RuntimeError("The Triton AttnRes backend requires CUDA and Triton")
    return selected


def _validate_value(value: Tensor, eps: float) -> None:
    if value.ndim < 2 or any(size == 0 for size in value.shape):
        raise ValueError("value must have nonempty token dimensions and a hidden dimension")
    if value.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("AttnRes source values must be BF16 or FP32")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("RMSNorm epsilon must be positive and finite")


def _validate_query(query: Tensor, value: Tensor, *, bank: bool) -> None:
    if query.ndim != (2 if bank else 1) or query.shape[-1] != value.shape[-1]:
        raise ValueError("effective query shape must match the source hidden dimension")
    if query.dtype != torch.float32 or query.device != value.device:
        raise TypeError("effective queries must be FP32 on the source device")


def _source_inputs(value, queries, mask, eps):
    _validate_value(value, eps)
    _validate_query(queries, value, bank=True)
    if mask is None:
        mask = torch.ones(queries.shape[0], dtype=torch.bool, device=value.device)
    if mask.shape != (queries.shape[0],) or mask.dtype != torch.bool:
        raise ValueError("live_source_mask must be a Boolean vector with one entry per query")
    if mask.device != value.device:
        raise ValueError("live_source_mask must be on the source device")
    return value.contiguous(), queries.contiguous(), mask.contiguous()


def _consumer_inputs(values, query, logits, eps):
    values = tuple(values)
    if not values:
        raise ValueError("AttnRes requires at least one source")
    _validate_value(values[0], eps)
    _validate_query(query, values[0], bank=False)
    for value in values:
        if (
            value.shape != values[0].shape
            or value.dtype != values[0].dtype
            or value.device != values[0].device
        ):
            raise ValueError("all source values must have identical shape, dtype and device")
    logits = (None,) * len(values) if logits is None else tuple(logits)
    if len(logits) != len(values):
        raise ValueError("logits must contain one tensor or None per source")
    for logit in logits:
        if logit is not None and (
            logit.shape != values[0].shape[:-1]
            or logit.dtype != torch.float32
            or logit.device != values[0].device
        ):
            raise ValueError("preprojected logits must be FP32 with the source token shape/device")
    return (
        tuple(value.contiguous() for value in values),
        query.contiguous(),
        tuple(logit.contiguous() if logit is not None else None for logit in logits),
    )


def project_source_reference(
    value: Tensor,
    effective_queries: Tensor,
    live_source_mask: Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Independent ordinary-autograd FP64 reference with the FP32 score interface.

    A detached normalized activation is used for masked query columns. Query
    gradients remain live. A shared wide source and a source-dtype proxy merge
    direct and score gradients before the single cast to the source leaf.
    """
    value, queries, mask = _source_inputs(value, effective_queries, live_source_mask, eps)
    v = value.double()
    normalized = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)
    live = torch.where(mask[:, None], normalized.unsqueeze(-2), normalized.detach().unsqueeze(-2))
    logits = (live * queries.double()).sum(-1).float()
    return v.to(value.dtype), logits


def aggregate_preprojected_reference(
    values: Sequence[Tensor],
    local_query: Tensor,
    logits: Sequence[Tensor | None] | None = None,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Independent ordinary-autograd softmax/mixing reference using FP64 math."""
    values, query, logits = _consumer_inputs(values, local_query, logits, eps)
    if len(values) == 1:
        zero = query.sum() * 0
        if logits[0] is not None:
            zero = zero + logits[0].sum() * 0
        return values[0] + zero.to(values[0].dtype)
    expanded = [value.double() for value in values]
    scores = [
        (
            logit.double()
            if logit is not None
            else (value * query.double()).sum(-1) * torch.rsqrt(value.square().mean(-1) + eps)
        )
        for value, logit in zip(expanded, logits)
    ]
    probabilities = torch.softmax(torch.stack(scores), dim=0)
    anchor = expanded[0]
    mixed = anchor + sum(
        (value - anchor) * probabilities[index].unsqueeze(-1)
        for index, value in enumerate(expanded)
    )
    # Preserve an explicit zero gradient when every score was precomputed.
    return (mixed + query.double().sum() * 0).to(values[0].dtype)


if triton is not None:

    @triton.jit
    def _source_forward(
        V, Q, L, RMS, H: tl.constexpr, R: tl.constexpr, EPS: tl.constexpr, BH: tl.constexpr
    ):
        token = tl.program_id(0)
        h = tl.arange(0, BH)
        value = tl.load(V + token * H + h, h < H, other=0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(value * value, 0) / H + EPS)
        tl.store(RMS + token, rstd)
        # Runtime loop bounds source-vector liveness independently of bank size.
        for row in range(R):
            query = tl.load(Q + row * H + h, h < H, other=0)
            score = tl.sum(value * query, 0) * rstd
            tl.store(L + token * R + row, score)

    @triton.jit
    def _source_backward_value(
        V,
        Q,
        MASK,
        L,
        RMS,
        GV,
        GL,
        DV,
        H: tl.constexpr,
        R: tl.constexpr,
        HAS_GV: tl.constexpr,
        BH: tl.constexpr,
    ):
        token = tl.program_id(0)
        h = tl.arange(0, BH)
        value = tl.load(V + token * H + h, h < H, other=0).to(tl.float32)
        rstd = tl.load(RMS + token)
        du = tl.full((BH,), 0, tl.float32)
        radial = tl.full((), 0, tl.float32)
        for row in range(R):
            live = tl.load(MASK + row)
            grad = tl.where(live, tl.load(GL + token * R + row), 0.0)
            query = tl.load(Q + row * H + h, h < H, other=0)
            du += grad * query
            radial += grad * tl.load(L + token * R + row)
        direct = tl.full((BH,), 0, tl.float32)
        if HAS_GV:
            direct = tl.load(GV + token * H + h, h < H, other=0).to(tl.float32)
        result = direct + rstd * du - value * (rstd * rstd * radial / H)
        tl.store(DV + token * H + h, result, h < H)

    @triton.jit
    def _source_query_partials(
        V,
        RMS,
        GL,
        PARTIAL,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        BT: tl.constexpr,
        BH: tl.constexpr,
    ):
        split, row, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        tokens = split * BT + tl.arange(0, BT)
        h = tile * BH + tl.arange(0, BH)
        value = tl.load(
            V + tokens[:, None] * H + h[None, :], (tokens[:, None] < T) & (h[None, :] < H), other=0
        ).to(tl.float32)
        factor = tl.load(GL + tokens * R + row, tokens < T, other=0)
        factor *= tl.load(RMS + tokens, tokens < T, other=0)
        partial = tl.sum(value * factor[:, None], 0)
        tl.store(PARTIAL + (split * R + row) * H + h, partial, h < H)

    @triton.jit
    def _sum_query_partials(
        PARTIAL,
        DQ,
        N: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        BT: tl.constexpr,
        BH: tl.constexpr,
    ):
        row, tile = tl.program_id(0), tl.program_id(1)
        h = tile * BH + tl.arange(0, BH)
        accumulator = tl.full((BH,), 0, tl.float32)
        for start in range(tl.cdiv(N, BT)):
            tokens = start * BT + tl.arange(0, BT)
            values = tl.load(
                PARTIAL + (tokens[:, None] * R + row) * H + h[None, :],
                (tokens[:, None] < N) & (h[None, :] < H),
                other=0,
            )
            accumulator += tl.sum(values, 0)
        tl.store(DQ + row * H + h, accumulator, h < H)

    @triton.jit
    def _consumer_forward(
        VALUES,
        Q,
        PRE,
        O,
        P,
        RMS,
        L,
        T: tl.constexpr,
        H: tl.constexpr,
        S: tl.constexpr,
        PRESENT: tl.constexpr,
        EPS: tl.constexpr,
        BH: tl.constexpr,
    ):
        token = tl.program_id(0)
        h = tl.arange(0, BH)
        query = tl.load(Q + h, h < H, other=0)
        maximum = tl.full((), float("-inf"), tl.float32)
        scores = ()
        for source in tl.static_range(S):
            if PRESENT[source]:
                score = tl.load(PRE[source] + token)
                rstd = tl.full((), 0, tl.float32)
            else:
                value = tl.load(VALUES[source] + token * H + h, h < H, other=0).to(tl.float32)
                rstd = tl.rsqrt(tl.sum(value * value, 0) / H + EPS)
                score = tl.sum(value * query, 0) * rstd
            scores += (score,)
            maximum = tl.maximum(maximum, score)
            tl.store(RMS + source * T + token, rstd)
            tl.store(L + source * T + token, score)
        denominator = tl.full((), 0, tl.float32)
        for source in tl.static_range(S):
            denominator += tl.exp(scores[source] - maximum)
        anchor = tl.load(VALUES[0] + token * H + h, h < H, other=0).to(tl.float32)
        output = tl.full((BH,), 0, tl.float32)
        for source in tl.static_range(S):
            probability = tl.exp(scores[source] - maximum) / denominator
            tl.store(P + source * T + token, probability)
            value = tl.load(VALUES[source] + token * H + h, h < H, other=0).to(tl.float32)
            output += probability * (value - anchor)
        tl.store(O + token * H + h, output + anchor, h < H)

    @triton.jit
    def _consumer_backward(
        VALUES,
        Q,
        P,
        RMS,
        L,
        GO,
        DVALUES,
        DLOGITS,
        DQ_ROWS,
        T: tl.constexpr,
        H: tl.constexpr,
        S: tl.constexpr,
        PRESENT: tl.constexpr,
        HAS_LOCAL: tl.constexpr,
        BH: tl.constexpr,
    ):
        token = tl.program_id(0)
        h = tl.arange(0, BH)
        gradient = tl.load(GO + token * H + h, h < H, other=0).to(tl.float32)
        anchor = tl.load(VALUES[0] + token * H + h, h < H, other=0).to(tl.float32)
        query = tl.load(Q + h, h < H, other=0)
        projections = ()
        probabilities = ()
        denominator = tl.full((), 0, tl.float32)
        delta = tl.full((), 0, tl.float32)
        for source in tl.static_range(S):
            value = tl.load(VALUES[source] + token * H + h, h < H, other=0).to(tl.float32)
            projection = tl.sum(gradient * (value - anchor), 0)
            probability = tl.load(P + source * T + token)
            projections += (projection,)
            probabilities += (probability,)
            denominator += probability
            delta += probability * projection
        delta /= denominator
        dq = tl.full((BH,), 0, tl.float32)
        for source in tl.static_range(S):
            probability = probabilities[source] / denominator
            ds = probability * (projections[source] - delta)
            result = probability * gradient
            if PRESENT[source]:
                tl.store(DLOGITS[source] + token, ds)
            else:
                value = tl.load(VALUES[source] + token * H + h, h < H, other=0).to(tl.float32)
                rstd = tl.load(RMS + source * T + token)
                score = tl.load(L + source * T + token)
                factor = ds * rstd
                result += factor * query - value * (factor * rstd * score / H)
                dq += factor * value
            tl.store(DVALUES[source] + token * H + h, result, h < H)
        if HAS_LOCAL:
            tl.store(DQ_ROWS + token * H + h, dq, h < H)


def _torch_source_forward(value, queries, eps):
    flat = value.reshape(-1, value.shape[-1]).float()
    rstd = torch.rsqrt(flat.square().mean(-1) + eps)
    logits = torch.stack([(flat * query).sum(-1) * rstd for query in queries], dim=-1)
    return rstd, logits


def _torch_source_backward(value, queries, mask, logits, rstd, grad_value, grad_logits):
    flat = value.reshape(-1, value.shape[-1]).float()
    du = torch.zeros_like(flat)
    radial = torch.zeros_like(rstd)
    dqueries = torch.empty_like(queries)
    for index, query in enumerate(queries):
        grad = grad_logits[:, index]
        dqueries[index] = (flat * (grad * rstd)[:, None]).sum(0)
        live_grad = grad * mask[index]
        du += live_grad[:, None] * query
        radial += live_grad * logits[:, index]
    derivative = rstd[:, None] * du - flat * (rstd.square() * radial / flat.shape[-1])[:, None]
    if grad_value is not None:
        derivative += grad_value.reshape_as(flat).float()
    return derivative.reshape_as(value).to(value.dtype), dqueries


class _ProjectSource(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, queries, mask, eps, backend):
        """Project every selected query using one source normalization."""
        ctx.set_materialize_grads(False)
        hidden, rows = value.shape[-1], queries.shape[0]
        tokens = value.numel() // hidden
        if rows == 0:
            logits = torch.empty((tokens, 0), device=value.device, dtype=torch.float32)
            rstd = torch.empty((0,), device=value.device, dtype=torch.float32)
        elif backend == "torch":
            rstd, logits = _torch_source_forward(value, queries, eps)
        else:
            logits = torch.empty((tokens, rows), device=value.device, dtype=torch.float32)
            rstd = torch.empty((tokens,), device=value.device, dtype=torch.float32)
            _source_forward[(tokens,)](
                value,
                queries,
                logits,
                rstd,
                hidden,
                rows,
                eps,
                triton.next_power_of_2(hidden),
                num_warps=4 if hidden <= 2048 else 8,
            )
        ctx.backend = backend
        ctx.save_for_backward(value, queries, mask, rstd, logits)
        return value, logits.reshape(*value.shape[:-1], rows)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_value, grad_logits):
        """Combine value and score derivatives before casting the source gradient."""
        value, queries, mask, rstd, logits = ctx.saved_tensors
        hidden, rows = value.shape[-1], queries.shape[0]
        tokens = value.numel() // hidden
        if grad_logits is None or rows == 0:
            dv = torch.zeros_like(value) if grad_value is None else grad_value
            return dv, torch.zeros_like(queries), None, None, None
        grad_logits = grad_logits.reshape(tokens, rows).contiguous()
        if ctx.backend == "torch":
            dv, dq = _torch_source_backward(
                value, queries, mask, logits, rstd, grad_value, grad_logits
            )
        else:
            dv, dq = torch.empty_like(value), torch.empty_like(queries)
            direct = value if grad_value is None else grad_value.contiguous()
            _source_backward_value[(tokens,)](
                value,
                queries,
                mask,
                logits,
                rstd,
                direct,
                grad_logits,
                dv,
                hidden,
                rows,
                grad_value is not None,
                triton.next_power_of_2(hidden),
                num_warps=4 if hidden <= 2048 else 8,
            )
            splits = triton.cdiv(tokens, 128)
            partial = torch.empty((splits, rows, hidden), device=value.device, dtype=torch.float32)
            _source_query_partials[(splits, rows, triton.cdiv(hidden, 32))](
                value, rstd, grad_logits, partial, tokens, hidden, rows, 128, 32, num_warps=4
            )
            _sum_query_partials[(rows, triton.cdiv(hidden, 64))](
                partial, dq, splits, hidden, rows, 32, 64, num_warps=4
            )
        return dv, dq, None, None, None


def _torch_consumer_forward(values, query, precomputed, eps):
    flattened = [value.reshape(-1, value.shape[-1]).float() for value in values]
    scores, rstds = [], []
    for value, logit in zip(flattened, precomputed):
        rstd = (
            torch.rsqrt(value.square().mean(-1) + eps)
            if logit is None
            else torch.zeros(value.shape[0], device=value.device, dtype=torch.float32)
        )
        scores.append((value * query).sum(-1) * rstd if logit is None else logit.reshape(-1))
        rstds.append(rstd)
    scores, rstds = torch.stack(scores), torch.stack(rstds)
    probabilities = scores.softmax(0)
    anchor = flattened[0]
    mixed = anchor + sum(
        probability[:, None] * (value - anchor)
        for probability, value in zip(probabilities, flattened)
    )
    return mixed.reshape_as(values[0]).to(values[0].dtype), probabilities, rstds, scores


def _torch_consumer_backward(values, query, present, probabilities, rstds, scores, grad):
    flat = [value.reshape(-1, value.shape[-1]).float() for value in values]
    gradient = grad.reshape_as(flat[0]).float()
    probabilities = probabilities / probabilities.sum(0, keepdim=True)
    projections = torch.stack([(gradient * (value - flat[0])).sum(-1) for value in flat])
    ds = probabilities * (projections - (probabilities * projections).sum(0, keepdim=True))
    dq = torch.zeros_like(query)
    dvs, dlogits = [], []
    for index, value in enumerate(flat):
        derivative = probabilities[index, :, None] * gradient
        if present[index]:
            dlogits.append(ds[index].reshape(values[index].shape[:-1]))
        else:
            factor = ds[index] * rstds[index]
            derivative += (
                factor[:, None] * query
                - value * (factor * rstds[index] * scores[index] / query.numel())[:, None]
            )
            dq += (factor[:, None] * value).sum(0)
            dlogits.append(None)
        dvs.append(derivative.reshape_as(values[index]).to(values[index].dtype))
    return dq, dvs, dlogits


class _AggregatePreprojected(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, eps, backend, count, *inputs):
        """Mix source values using precomputed or locally evaluated logits."""
        values, logits = inputs[:count], inputs[count:]
        ctx.present = tuple(logit is not None for logit in logits)
        ctx.count, ctx.backend = count, backend
        ctx.set_materialize_grads(False)
        if count == 1:
            ctx.save_for_backward(query, *values, *[logit for logit in logits if logit is not None])
            return values[0]
        hidden = values[0].shape[-1]
        tokens = values[0].numel() // hidden
        if backend == "torch":
            output, probabilities, rstd, scores = _torch_consumer_forward(
                values, query, logits, eps
            )
        else:
            output = torch.empty_like(values[0])
            probabilities = torch.empty((count, tokens), device=query.device, dtype=torch.float32)
            rstd, scores = torch.empty_like(probabilities), torch.empty_like(probabilities)
            _consumer_forward[(tokens,)](
                values,
                query,
                tuple(logit if logit is not None else query for logit in logits),
                output,
                probabilities,
                rstd,
                scores,
                tokens,
                hidden,
                count,
                ctx.present,
                eps,
                triton.next_power_of_2(hidden),
                num_warps=4 if hidden <= 2048 else 8,
            )
        ctx.save_for_backward(
            query,
            *values,
            probabilities,
            rstd,
            scores,
            *[logit for logit in logits if logit is not None],
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        """Return direct source, local-query and precomputed-score gradients."""
        query, *saved = ctx.saved_tensors
        values = saved[: ctx.count]
        if grad is None:
            grad = torch.zeros_like(values[0])
        if ctx.count == 1:
            dlogit = torch.zeros(values[0].shape[:-1], device=query.device, dtype=torch.float32)
            return (
                torch.zeros_like(query),
                None,
                None,
                None,
                grad,
                dlogit if ctx.present[0] else None,
            )
        probabilities, rstd, scores = saved[ctx.count : ctx.count + 3]
        if ctx.backend == "torch":
            dq, dvs, dlogits = _torch_consumer_backward(
                values, query, ctx.present, probabilities, rstd, scores, grad
            )
        else:
            hidden = values[0].shape[-1]
            tokens = values[0].numel() // hidden
            dvs = [torch.empty_like(value) for value in values]
            dlogits = [
                (
                    torch.empty(value.shape[:-1], device=query.device, dtype=torch.float32)
                    if present
                    else None
                )
                for value, present in zip(values, ctx.present)
            ]
            has_local = not all(ctx.present)
            dq = torch.empty_like(query) if has_local else torch.zeros_like(query)
            dq_rows = (
                torch.empty((tokens, hidden), device=query.device, dtype=torch.float32)
                if has_local
                else query
            )
            _consumer_backward[(tokens,)](
                tuple(values),
                query,
                probabilities,
                rstd,
                scores,
                grad.contiguous(),
                tuple(dvs),
                tuple(logit if logit is not None else query for logit in dlogits),
                dq_rows,
                tokens,
                hidden,
                ctx.count,
                ctx.present,
                has_local,
                triton.next_power_of_2(hidden),
                num_warps=4 if hidden <= 2048 else 8,
            )
            if has_local:
                _sum_query_partials[(1, triton.cdiv(hidden, 64))](
                    dq_rows, dq, tokens, hidden, 1, 128, 64, num_warps=4
                )
        return dq, None, None, None, *dvs, *dlogits


def project_source(
    value: Tensor,
    effective_queries: Tensor,
    live_source_mask: Tensor | None = None,
    *,
    eps: float = 1e-6,
    backend: str | None = None,
) -> tuple[Tensor, Tensor]:
    """Project one immutable source against a bank of future effective queries.

    Args:
        value: BF16/FP32 tensor shaped ``[*token_dimensions, hidden]``.
        effective_queries: FP32 tensor shaped ``[queries, hidden]``.
        live_source_mask: Boolean query mask gating source-score gradients only.
        eps: Positive RMSNorm epsilon.
        backend: Explicit ``torch``/``triton``, or CPU/CUDA device selection.

    Returns:
        A same-dtype value proxy and FP32 ``[*token_dimensions, queries]`` logits.
        Both must participate in the same backward invocation when both are used.
    """
    value, queries, mask = _source_inputs(value, effective_queries, live_source_mask, eps)
    return _ProjectSource.apply(value, queries, mask, eps, _backend(value, backend))


def aggregate_preprojected(
    values: Sequence[Tensor],
    local_query: Tensor,
    logits: Sequence[Tensor | None] | None = None,
    *,
    eps: float = 1e-6,
    backend: str | None = None,
) -> Tensor:
    """Mix sources using producer logits and locally scored partial sources.

    Each ``logits`` entry is an FP32 tensor with the source token shape or
    ``None`` for a locally scored source. Logical source aliases remain separate
    autograd inputs. One source is an exact identity with explicit zero query
    and preprojected-score gradients. Higher-order derivatives are unsupported.
    """
    values, query, logits = _consumer_inputs(values, local_query, logits, eps)
    return _AggregatePreprojected.apply(
        query, eps, _backend(values[0], backend), len(values), *values, *logits
    )
