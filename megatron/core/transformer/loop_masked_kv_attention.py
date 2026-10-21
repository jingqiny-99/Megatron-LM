# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Causal attention in which only *active* tokens provide keys and values.

This is the attention of the loop transformer's token-choice ("MoR") masked
execution: every recursion keeps the full ``[B, S]`` shape, but a token that has
exited no longer contributes K/V, and its own output is discarded downstream.
The dense equivalent is a boolean ``[B, S, S]`` mask, which forces an unfused
O(S^2) attention. This module provides the same math with work proportional to
the active (query, key) pairs and with static shapes, so that it can be captured
in a CUDA graph and replayed with different masks.

Semantics (``allowed[b, i, j]``)
--------------------------------
``allowed[b, i, j] = query_active[b, i] & key_active[b, j] & (j <= i if causal)``

* For an allowed row, ``out[b, i] = softmax_j(scale * q_i . k_j) @ v`` over the
  allowed keys only, and ``lse[b, h, i]`` is the natural-log logsumexp of the
  scaled scores of those keys.
* A row with no allowed key (an inactive query, or an active query with no
  active key at or before it) yields ``out = 0`` and ``lse = -inf``. Its
  gradient contribution is exactly zero, as are ``dq`` for inactive queries and
  ``dk``/``dv`` for inactive keys. No NaN is produced for any such row, whatever
  values the inactive rows of q/k/v/dout hold.
* ``query_active`` defaults to ``key_active`` (the MoR masked mode). With
  ``all_queries=True`` every query is computed and only keys are masked.
* If a query is active but its own key is not (possible only when the two masks
  differ), the query attends to the active keys strictly before it, excluding
  itself; if there are none, its row is empty (out 0, lse -inf).

This differs from ``active_causal_mask`` in ``loop_transformer_block.py`` only on
inactive query rows: that mask lets an inactive query attend to itself (to avoid
NaN in the unfused softmax) whereas here its output is zero. MoR discards those
rows (``state = where(active, new, old)``), so model outputs and gradients match.

Algorithm
---------
Tile skipping alone does not help this workload: with tokens exiting at random
positions, a 64-token tile has no active token with probability ``(1 - f)^64``,
which is about 0.1% for an active fraction ``f`` of 10%. (Measured with
FlexAttention 128x128 block masks on scattered masks at 10-50% active: every
causal tile stays live.) Instead, the kernels work in a
*compacted* index space. ``build_masked_kv_metadata`` computes, on device and
with static shapes, a per-batch stable permutation that lists the active tokens
first (in sequence order) followed by the inactive ones, plus prefix counts. A
Q tile of compacted rows loads its queries by gathering rows at their original
positions, and iterates only over the compacted active keys that are causally
visible to it. Within the active subsequence the original order is preserved,
so causal visibility reduces to a prefix of the compacted keys, and only the
tile on the diagonal needs a mask. Q tiles past the active count write zeros and
exit. The work is therefore ``O(n_q * n_k / 2)`` per sequence, instead of
``O(S^2 / 2)``. The gather costs one int32 index load per row, and each gathered
row is a contiguous ``D``-element vector.

Static shapes and graph capture
-------------------------------
The kernel grids depend only on ``(B, S, H, D)``. Active counts and loop bounds
are read on device, so the same captured graph replays correctly for any mask.
There is no host synchronization, no data-dependent Python control flow, and
no allocation whose size depends on the mask.

Backends
--------
``"triton"``: FlashAttention-2 style forward, dK/dV and dQ kernels in Triton
(CUDA, fp16/bf16, head_dim <= 256; D that is not a power of two is padded).
Deterministic: no atomics. A dK/dV program owns a tile of keys of one KV head
and loops over the query heads of its GQA group; when that grid would leave SMs
idle (short sequences), the group's query heads are spread over programs and
their fp32 partial dK/dV are summed afterwards.
``"flex"``: the same compaction around ``torch.nn.attention.flex_attention``,
with a BlockMask built on device. On CPU it runs eagerly (forward only, O(S^2)
memory); on CUDA it is compiled and has about 1.3 ms of host overhead per
eager call, which CUDA graphs remove.
``"reference"``: pure PyTorch, O(S^2) memory, for tests.

Performance (H100, bf16, Hq=16, Hkv=4, D=128, S=2K..16K)
--------------------------------------------------------
With every token active, the Triton kernels take 1.4-1.7x (forward) and
1.5-1.9x (backward) the time of cuDNN's dense causal attention (S >= 4K). At 50%
active they are 1.2-2.7x (forward) and 1.0-2.4x (backward) faster than it, and
at 10% active 2.6-17x and 1.6-21x faster. Against TE's arbitrary-mask (unfused)
path they are 5-460x faster in forward and 1.5-230x in backward, and they run
at S=16K, B>=2, where that path runs out of memory. A caller should therefore
keep dense causal attention when every token is known to be active (for example
MoR's first recursion without padding).

Layout: q ``[B, S, Hq, D]``, k/v ``[B, S, Hkv, D]`` with ``Hq % Hkv == 0``. Any
strides are accepted as long as the head dimension is contiguous, so a
Megatron ``[S, B, H, D]`` tensor can be passed as ``x.transpose(0, 1)`` without
a copy; the output keeps the input's memory layout.
"""

import math
from typing import NamedTuple

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised on hosts without Triton
    HAVE_TRITON = False


_LOG2E = 1.4426950408889634


class MaskedKVMetadata(NamedTuple):
    """Per-batch compaction metadata, all on device, all with static shapes.

    ``q_perm[b, c]`` is the original position of the ``c``-th compacted query:
    active queries in increasing position, followed by inactive ones.
    ``q_count[b]`` is the number of active queries. ``q_excl[b, p]`` counts the
    active queries at positions ``< p``. ``k_perm``/``k_count`` are the same for
    keys, and ``k_incl[b, p]`` counts the active keys at positions ``<= p``.
    All tensors are int32 and contiguous.
    """

    q_perm: torch.Tensor
    q_count: torch.Tensor
    q_excl: torch.Tensor
    k_perm: torch.Tensor
    k_count: torch.Tensor
    k_incl: torch.Tensor


def _compaction(active):
    """Stable active-first permutation, active count, and inclusive prefix count."""
    batch, sequence = active.shape
    flags = active.to(torch.int32)
    inclusive = flags.cumsum(1, dtype=torch.int32)
    count = inclusive[:, -1].contiguous()
    exclusive = inclusive - flags
    positions = torch.arange(sequence, dtype=torch.int32, device=active.device).expand(
        batch, sequence
    )
    # Active tokens go to their exclusive rank; inactive ones follow, in order.
    destination = torch.where(active, exclusive, count[:, None] + positions - exclusive)
    permutation = torch.empty_like(flags).scatter_(1, destination.long(), positions)
    return permutation, count, inclusive.contiguous(), exclusive.contiguous()


def build_masked_kv_metadata(key_active, query_active=None, *, all_queries=False):
    """Build compaction metadata from ``[B, S]`` boolean masks.

    Args:
        key_active: ``[B, S]`` bool, True where the token provides K/V.
        query_active: optional ``[B, S]`` bool, True where the query is computed.
            Defaults to ``key_active``.
        all_queries: compute every query (only keys are masked). Mutually
            exclusive with ``query_active``.

    The result depends only on the masks, so one metadata object can be shared
    by every attention layer of a recursion step, in forward and backward.
    """
    if key_active.dim() != 2:
        raise ValueError(f"key_active must be [B, S], got {tuple(key_active.shape)}")
    if all_queries and query_active is not None:
        raise ValueError("Pass either query_active or all_queries=True, not both")
    key_active = key_active.to(torch.bool)
    k_perm, k_count, k_incl, k_excl = _compaction(key_active)
    if all_queries:
        query_active = torch.ones_like(key_active)
    if query_active is None:
        q_perm, q_count, q_excl = k_perm, k_count, k_excl
    else:
        if query_active.shape != key_active.shape:
            raise ValueError("query_active and key_active must have the same [B, S] shape")
        q_perm, q_count, _, q_excl = _compaction(query_active.to(torch.bool))
    return MaskedKVMetadata(q_perm, q_count, q_excl, k_perm, k_count, k_incl)


def _query_mask(key_active, query_active, all_queries):
    if all_queries:
        if query_active is not None:
            raise ValueError("Pass either query_active or all_queries=True, not both")
        return torch.ones_like(key_active, dtype=torch.bool)
    return key_active if query_active is None else query_active


def _check_inputs(q, k, v):
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k and v must be [B, S, H, D]")
    if k.shape != v.shape:
        raise ValueError(f"k and v must have the same shape, got {k.shape} and {v.shape}")
    if q.shape[:2] != k.shape[:2] or q.shape[3] != k.shape[3]:
        raise ValueError(f"Incompatible q {tuple(q.shape)} and k {tuple(k.shape)}")
    if q.shape[2] % k.shape[2]:
        raise ValueError("The number of query heads must be a multiple of the KV heads")


# ---------------------------------------------------------------------------
# Pure PyTorch reference
# ---------------------------------------------------------------------------


def masked_kv_attention_reference(
    q,
    k,
    v,
    key_active,
    query_active=None,
    *,
    all_queries=False,
    causal=True,
    softmax_scale=None,
    return_lse=False,
):
    """Dense O(S^2) reference with the module's semantics, differentiable.

    Computes in at least float32 and returns the output in ``q.dtype``.
    """
    _check_inputs(q, k, v)
    batch, sequence, heads, dim = q.shape
    group = heads // k.shape[2]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(dim)
    compute = torch.promote_types(q.dtype, torch.float32)
    query_mask = _query_mask(key_active, query_active, all_queries).to(torch.bool)
    qf = q.to(compute).transpose(1, 2)
    kf = k.to(compute).transpose(1, 2).repeat_interleave(group, dim=1)
    vf = v.to(compute).transpose(1, 2).repeat_interleave(group, dim=1)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    allowed = query_mask[:, None, :, None] & key_active.to(torch.bool)[:, None, None, :]
    if causal:
        allowed = allowed & torch.ones(sequence, sequence, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~allowed, float("-inf"))
    # A detached, finite shift keeps empty rows at exactly zero without the NaN
    # that logsumexp's backward produces for all -inf rows.
    shift = scores.detach().amax(-1, keepdim=True)
    shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
    weights = torch.exp(scores - shift)
    total = weights.sum(-1, keepdim=True)
    nonempty = total > 0
    out = torch.matmul(weights, vf) / torch.where(nonempty, total, torch.ones_like(total))
    out = out.transpose(1, 2).to(q.dtype)
    if not return_lse:
        return out
    lse = torch.where(
        nonempty, torch.log(torch.where(nonempty, total, torch.ones_like(total))) + shift, -math.inf
    ).squeeze(-1)
    return out, lse.detach().float()


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _load_rows(ptrs, row_mask, d_mask, ROW_MASK: tl.constexpr, EVEN_D: tl.constexpr):
        if ROW_MASK:
            if EVEN_D:
                tile = tl.load(ptrs, mask=row_mask[:, None], other=0.0)
            else:
                tile = tl.load(ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0)
        else:
            if EVEN_D:
                tile = tl.load(ptrs)
            else:
                tile = tl.load(ptrs, mask=d_mask[None, :], other=0.0)
        return tile

    @triton.jit
    def _q_tile_kv_bounds(
        Q_PERM, K_INCL, meta, m_start, nq, nk, S, BLOCK_M: tl.constexpr, CAUSAL: tl.constexpr
    ):
        """Compacted key count fully visible to the tile, and visible to any row."""
        live = m_start < nq
        if CAUSAL:
            first = tl.minimum(m_start, S - 1)
            last = tl.maximum(tl.minimum(m_start + BLOCK_M, nq) - 1, 0)
            kv_full = tl.load(K_INCL + meta + tl.load(Q_PERM + meta + first))
            kv_high = tl.load(K_INCL + meta + tl.load(Q_PERM + meta + last))
        else:
            kv_full = nk
            kv_high = nk
        kv_full = tl.where(live, kv_full, 0)
        kv_high = tl.where(live, kv_high, 0)
        return kv_full, kv_high

    @triton.jit
    def _fwd_kernel(
        Q,
        K,
        V,
        O,
        LSE,
        Q_PERM,
        Q_COUNT,
        K_PERM,
        K_COUNT,
        K_INCL,
        qk_scale,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_ob,
        stride_os,
        stride_oh,
        S,
        H,
        GROUP,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr,
        EVEN_D: tl.constexpr,
    ):
        # Later compacted tiles see more keys; start them first.
        pid_m = tl.num_programs(0) - 1 - tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        b64 = b.to(tl.int64)
        meta = b64 * S
        nq = tl.load(Q_COUNT + b)
        nk = tl.load(K_COUNT + b)

        m_start = pid_m * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_n0 = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        d_ok = offs_d < HEAD_DIM
        row_in = offs_m < S
        row_ok = offs_m < nq
        pos_q = tl.load(Q_PERM + meta + offs_m, mask=row_in, other=0).to(tl.int64)
        q_ptrs = Q + b64 * stride_qb + h * stride_qh + pos_q[:, None] * stride_qs + offs_d[None, :]
        q = _load_rows(q_ptrs, row_ok, d_ok, True, EVEN_D)

        kv_full, kv_high = _q_tile_kv_bounds(
            Q_PERM, K_INCL, meta, m_start, nq, nk, S, BLOCK_M, CAUSAL
        )
        full_end = (kv_full // BLOCK_N) * BLOCK_N
        high_end = tl.cdiv(kv_high, BLOCK_N) * BLOCK_N

        k_base = K + b64 * stride_kb + (h // GROUP) * stride_kh
        v_base = V + b64 * stride_vb + (h // GROUP) * stride_vh
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

        # Keys visible to every row of the tile: no mask.
        for start_n in range(0, full_end, BLOCK_N):
            offs_n = start_n + offs_n0
            pos_k = tl.load(K_PERM + meta + offs_n).to(tl.int64)
            k = _load_rows(
                k_base + pos_k[:, None] * stride_ks + offs_d[None, :], offs_n, d_ok, False, EVEN_D
            )
            v = _load_rows(
                v_base + pos_k[:, None] * stride_vs + offs_d[None, :], offs_n, d_ok, False, EVEN_D
            )
            qk = tl.dot(q, tl.trans(k)) * qk_scale
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        # Diagonal tiles and the ragged end of the active keys: masked.
        for start_n in range(full_end, high_end, BLOCK_N):
            offs_n = start_n + offs_n0
            col_ok = offs_n < nk
            pos_k = tl.load(K_PERM + meta + offs_n, mask=col_ok, other=0).to(tl.int64)
            k = _load_rows(
                k_base + pos_k[:, None] * stride_ks + offs_d[None, :], col_ok, d_ok, True, EVEN_D
            )
            v = _load_rows(
                v_base + pos_k[:, None] * stride_vs + offs_d[None, :], col_ok, d_ok, True, EVEN_D
            )
            qk = tl.dot(q, tl.trans(k)) * qk_scale
            allowed = col_ok[None, :]
            if CAUSAL:
                allowed = allowed & (pos_k[None, :] <= pos_q[:, None])
            qk = tl.where(allowed, qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            # A row with no visible key yet keeps m = -inf; shift it by 0.
            m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.math.exp2(qk - m_use[:, None])
            alpha = tl.math.exp2(m_i - m_use)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        nonempty = l_i > 0.0
        l_safe = tl.where(nonempty, l_i, 1.0)
        out = acc / l_safe[:, None]
        out = tl.where(row_ok[:, None], out, 0.0)
        lse = tl.where(
            row_ok & nonempty, (m_i + tl.math.log2(l_safe)) / 1.4426950408889634, float("-inf")
        )
        o_ptrs = O + b64 * stride_ob + h * stride_oh + pos_q[:, None] * stride_os + offs_d[None, :]
        if EVEN_D:
            tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=row_in[:, None])
        else:
            tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=row_in[:, None] & d_ok[None, :])
        tl.store(LSE + (b64 * H + h) * S + pos_q, lse, mask=row_in)

    @triton.jit
    def _bwd_preprocess_kernel(
        O,
        DO,
        DELTA,
        Q_PERM,
        Q_COUNT,
        stride_ob,
        stride_os,
        stride_oh,
        stride_dob,
        stride_dos,
        stride_doh,
        S,
        H,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        EVEN_D: tl.constexpr,
    ):
        """delta = rowsum(out * dout) for active queries, 0 elsewhere."""
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        b64 = b.to(tl.int64)
        meta = b64 * S
        nq = tl.load(Q_COUNT + b)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        d_ok = offs_d < HEAD_DIM
        row_in = offs_m < S
        row_ok = offs_m < nq
        pos_q = tl.load(Q_PERM + meta + offs_m, mask=row_in, other=0).to(tl.int64)
        o = _load_rows(
            O + b64 * stride_ob + h * stride_oh + pos_q[:, None] * stride_os + offs_d[None, :],
            row_ok,
            d_ok,
            True,
            EVEN_D,
        )
        do = _load_rows(
            DO + b64 * stride_dob + h * stride_doh + pos_q[:, None] * stride_dos + offs_d[None, :],
            row_ok,
            d_ok,
            True,
            EVEN_D,
        )
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
        tl.store(DELTA + (b64 * H + h) * S + pos_q, delta, mask=row_in)

    @triton.jit
    def _dkdv_inner(
        dk,
        dv,
        k,
        v,
        pos_k,
        q_base,
        do_base,
        LSE_ROW,
        DELTA_ROW,
        Q_PERM_ROW,
        nq,
        stride_qs,
        stride_dos,
        qk_scale,
        offs_m0,
        offs_d,
        d_ok,
        start,
        end,
        BLOCK_M: tl.constexpr,
        MASK: tl.constexpr,
        EVEN_D: tl.constexpr,
    ):
        for start_m in range(start, end, BLOCK_M):
            offs_m = start_m + offs_m0
            row_ok = offs_m < nq
            pos_q = tl.load(Q_PERM_ROW + offs_m, mask=row_ok, other=0).to(tl.int64)
            q = _load_rows(
                q_base + pos_q[:, None] * stride_qs + offs_d[None, :], row_ok, d_ok, True, EVEN_D
            )
            do = _load_rows(
                do_base + pos_q[:, None] * stride_dos + offs_d[None, :], row_ok, d_ok, True, EVEN_D
            )
            lse = tl.load(LSE_ROW + pos_q, mask=row_ok, other=float("inf"))
            # Empty rows store -inf; +inf makes their probabilities 0.
            lse = tl.where(lse == float("-inf"), float("inf"), lse) * 1.4426950408889634
            delta = tl.load(DELTA_ROW + pos_q, mask=row_ok, other=0.0)
            qk_t = tl.dot(k, tl.trans(q)) * qk_scale
            p_t = tl.math.exp2(qk_t - lse[None, :])
            if MASK:
                p_t = tl.where(pos_k[:, None] <= pos_q[None, :], p_t, 0.0)
            dv += tl.dot(p_t.to(do.dtype), do)
            dp_t = tl.dot(v, tl.trans(do))
            ds_t = p_t * (dp_t - delta[None, :])
            dk += tl.dot(ds_t.to(q.dtype), q)
        return dk, dv

    @triton.jit
    def _bwd_dkdv_kernel(
        Q,
        K,
        V,
        DO,
        DK,
        DV,
        LSE,
        DELTA,
        Q_PERM,
        Q_COUNT,
        Q_EXCL,
        K_PERM,
        K_COUNT,
        qk_scale,
        sm_scale,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_dob,
        stride_dos,
        stride_doh,
        stride_dkb,
        stride_dks,
        stride_dkh,
        stride_dvb,
        stride_dvs,
        stride_dvh,
        S,
        H,
        HKV,
        GROUP,
        SPLIT,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr,
        EVEN_D: tl.constexpr,
    ):
        """One program owns a tile of compacted keys of one KV head and walks the
        compacted queries that can see it, for GROUP // SPLIT of its query heads.
        With SPLIT > 1, DK/DV are per-split partial sums with HKV * SPLIT heads."""
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // (HKV * SPLIT)
        out_head = pid_bh % (HKV * SPLIT)
        hk = out_head // SPLIT
        heads_per_split = GROUP // SPLIT
        first_head = hk * GROUP + (out_head % SPLIT) * heads_per_split
        b64 = b.to(tl.int64)
        meta = b64 * S
        nq = tl.load(Q_COUNT + b)
        nk = tl.load(K_COUNT + b)

        n_start = pid_n * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        offs_m0 = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        d_ok = offs_d < HEAD_DIM
        col_in = offs_n < S
        col_ok = offs_n < nk
        pos_k = tl.load(K_PERM + meta + offs_n, mask=col_in, other=0).to(tl.int64)
        k = _load_rows(
            K + b64 * stride_kb + hk * stride_kh + pos_k[:, None] * stride_ks + offs_d[None, :],
            col_ok,
            d_ok,
            True,
            EVEN_D,
        )
        v = _load_rows(
            V + b64 * stride_vb + hk * stride_vh + pos_k[:, None] * stride_vs + offs_d[None, :],
            col_ok,
            d_ok,
            True,
            EVEN_D,
        )

        live = n_start < nk
        if CAUSAL:
            first = tl.minimum(n_start, S - 1)
            last = tl.maximum(tl.minimum(n_start + BLOCK_N, nk) - 1, 0)
            # Queries before the tile's first key see none of it; queries at or
            # after its last key see all of it.
            q_low = tl.load(Q_EXCL + meta + tl.load(K_PERM + meta + first))
            q_full = tl.load(Q_EXCL + meta + tl.load(K_PERM + meta + last))
        else:
            q_low = 0
            q_full = 0
        q_high = tl.where(live, nq, 0)
        low = tl.where(live, (q_low // BLOCK_M) * BLOCK_M, 0)
        high = tl.cdiv(q_high, BLOCK_M) * BLOCK_M
        mid = tl.minimum(tl.maximum(tl.cdiv(q_full, BLOCK_M) * BLOCK_M, low), high)

        dk = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
        for g in range(0, heads_per_split):
            h = first_head + g
            q_base = Q + b64 * stride_qb + h * stride_qh
            do_base = DO + b64 * stride_dob + h * stride_doh
            row_base = (b64 * H + h) * S
            # Queries on the tile's causal diagonal: masked.
            dk, dv = _dkdv_inner(
                dk,
                dv,
                k,
                v,
                pos_k,
                q_base,
                do_base,
                LSE + row_base,
                DELTA + row_base,
                Q_PERM + meta,
                nq,
                stride_qs,
                stride_dos,
                qk_scale,
                offs_m0,
                offs_d,
                d_ok,
                low,
                mid,
                BLOCK_M,
                CAUSAL,
                EVEN_D,
            )
            # Queries at or after the tile's last key: every key visible.
            dk, dv = _dkdv_inner(
                dk,
                dv,
                k,
                v,
                pos_k,
                q_base,
                do_base,
                LSE + row_base,
                DELTA + row_base,
                Q_PERM + meta,
                nq,
                stride_qs,
                stride_dos,
                qk_scale,
                offs_m0,
                offs_d,
                d_ok,
                mid,
                high,
                BLOCK_M,
                False,
                EVEN_D,
            )

        dk = tl.where(col_ok[:, None], dk * sm_scale, 0.0)
        dv = tl.where(col_ok[:, None], dv, 0.0)
        dk_ptrs = (
            DK
            + b64 * stride_dkb
            + out_head * stride_dkh
            + pos_k[:, None] * stride_dks
            + offs_d[None, :]
        )
        dv_ptrs = (
            DV
            + b64 * stride_dvb
            + out_head * stride_dvh
            + pos_k[:, None] * stride_dvs
            + offs_d[None, :]
        )
        if EVEN_D:
            tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=col_in[:, None])
            tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=col_in[:, None])
        else:
            tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=col_in[:, None] & d_ok[None, :])
            tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=col_in[:, None] & d_ok[None, :])

    @triton.jit
    def _bwd_dq_kernel(
        Q,
        K,
        V,
        DO,
        DQ,
        LSE,
        DELTA,
        Q_PERM,
        Q_COUNT,
        K_PERM,
        K_COUNT,
        K_INCL,
        qk_scale,
        sm_scale,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_dob,
        stride_dos,
        stride_doh,
        stride_dqb,
        stride_dqs,
        stride_dqh,
        S,
        H,
        GROUP,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr,
        EVEN_D: tl.constexpr,
    ):
        pid_m = tl.num_programs(0) - 1 - tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        b64 = b.to(tl.int64)
        meta = b64 * S
        nq = tl.load(Q_COUNT + b)
        nk = tl.load(K_COUNT + b)

        m_start = pid_m * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_n0 = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        d_ok = offs_d < HEAD_DIM
        row_in = offs_m < S
        row_ok = offs_m < nq
        pos_q = tl.load(Q_PERM + meta + offs_m, mask=row_in, other=0).to(tl.int64)
        q = _load_rows(
            Q + b64 * stride_qb + h * stride_qh + pos_q[:, None] * stride_qs + offs_d[None, :],
            row_ok,
            d_ok,
            True,
            EVEN_D,
        )
        do = _load_rows(
            DO + b64 * stride_dob + h * stride_doh + pos_q[:, None] * stride_dos + offs_d[None, :],
            row_ok,
            d_ok,
            True,
            EVEN_D,
        )
        row_base = (b64 * H + h) * S
        lse = tl.load(LSE + row_base + pos_q, mask=row_ok, other=float("inf"))
        lse = tl.where(lse == float("-inf"), float("inf"), lse) * 1.4426950408889634
        delta = tl.load(DELTA + row_base + pos_q, mask=row_ok, other=0.0)

        kv_full, kv_high = _q_tile_kv_bounds(
            Q_PERM, K_INCL, meta, m_start, nq, nk, S, BLOCK_M, CAUSAL
        )
        full_end = (kv_full // BLOCK_N) * BLOCK_N
        high_end = tl.cdiv(kv_high, BLOCK_N) * BLOCK_N
        k_base = K + b64 * stride_kb + (h // GROUP) * stride_kh
        v_base = V + b64 * stride_vb + (h // GROUP) * stride_vh

        dq = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
        for start_n in range(0, full_end, BLOCK_N):
            offs_n = start_n + offs_n0
            pos_k = tl.load(K_PERM + meta + offs_n).to(tl.int64)
            k = _load_rows(
                k_base + pos_k[:, None] * stride_ks + offs_d[None, :], offs_n, d_ok, False, EVEN_D
            )
            v = _load_rows(
                v_base + pos_k[:, None] * stride_vs + offs_d[None, :], offs_n, d_ok, False, EVEN_D
            )
            qk = tl.dot(q, tl.trans(k)) * qk_scale
            p = tl.math.exp2(qk - lse[:, None])
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k)
        for start_n in range(full_end, high_end, BLOCK_N):
            offs_n = start_n + offs_n0
            col_ok = offs_n < nk
            pos_k = tl.load(K_PERM + meta + offs_n, mask=col_ok, other=0).to(tl.int64)
            k = _load_rows(
                k_base + pos_k[:, None] * stride_ks + offs_d[None, :], col_ok, d_ok, True, EVEN_D
            )
            v = _load_rows(
                v_base + pos_k[:, None] * stride_vs + offs_d[None, :], col_ok, d_ok, True, EVEN_D
            )
            qk = tl.dot(q, tl.trans(k)) * qk_scale
            allowed = col_ok[None, :]
            if CAUSAL:
                allowed = allowed & (pos_k[None, :] <= pos_q[:, None])
            p = tl.where(allowed, tl.math.exp2(qk - lse[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k)

        dq = tl.where(row_ok[:, None], dq * sm_scale, 0.0)
        dq_ptrs = (
            DQ + b64 * stride_dqb + h * stride_dqh + pos_q[:, None] * stride_dqs + offs_d[None, :]
        )
        if EVEN_D:
            tl.store(dq_ptrs, dq.to(DQ.dtype.element_ty), mask=row_in[:, None])
        else:
            tl.store(dq_ptrs, dq.to(DQ.dtype.element_ty), mask=row_in[:, None] & d_ok[None, :])


class KernelConfig(NamedTuple):
    """Tile shape and launch parameters of one Triton kernel."""

    block_m: int
    block_n: int
    num_warps: int
    num_stages: int


# Keyed by padded head dim. D=128 was swept on H100 at (S, B) = (8192, 2) and
# (16384, 1), Hq=16, Hkv=4, 100% and 50%/25% active (bench_loop_masked_kv_attention.py
# --tune); the other head dims are untuned.
_DEFAULT_CONFIGS = {
    "fwd": {
        64: KernelConfig(128, 64, 4, 3),
        128: KernelConfig(128, 128, 8, 3),
        256: KernelConfig(64, 32, 4, 2),
    },
    "dkdv": {
        64: KernelConfig(32, 64, 4, 3),
        128: KernelConfig(32, 64, 4, 4),
        256: KernelConfig(32, 64, 8, 1),
    },
    "dq": {
        64: KernelConfig(64, 64, 4, 3),
        128: KernelConfig(128, 128, 8, 3),
        256: KernelConfig(64, 32, 4, 1),
    },
}
# Benchmarks may override entries, e.g. _CONFIG_OVERRIDES[("fwd", 128)] = KernelConfig(...).
_CONFIG_OVERRIDES = {}


def _kernel_config(kind, block_d):
    key = max(64, block_d)
    override = _CONFIG_OVERRIDES.get((kind, key))
    return override if override is not None else _DEFAULT_CONFIGS[kind][key]


def _block_d(dim):
    return max(16, triton.next_power_of_2(dim))


def _triton_forward(q, k, v, meta, causal, sm_scale):
    batch, sequence, heads, dim = q.shape
    kv_heads = k.shape[2]
    out = torch.empty_like(q)
    if out.stride(-1) != 1:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    lse = torch.empty((batch, heads, sequence), dtype=torch.float32, device=q.device)
    block_d = _block_d(dim)
    cfg = _kernel_config("fwd", block_d)
    grid = (triton.cdiv(sequence, cfg.block_m), batch * heads)
    _fwd_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        meta.q_perm,
        meta.q_count,
        meta.k_perm,
        meta.k_count,
        meta.k_incl,
        sm_scale * _LOG2E,
        *q.stride()[:3],
        *k.stride()[:3],
        *v.stride()[:3],
        *out.stride()[:3],
        sequence,
        heads,
        heads // kv_heads,
        HEAD_DIM=dim,
        BLOCK_D=block_d,
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.block_n,
        CAUSAL=causal,
        EVEN_D=block_d == dim,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return out, lse


_SM_COUNT = {}
# The dK/dV grid aims for at least this many programs per SM before splitting
# the query heads of a GQA group across programs.
_DKDV_PROGRAMS_PER_SM = 4


def _dkdv_split(programs, group, device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SM_COUNT:
        _SM_COUNT[index] = torch.cuda.get_device_properties(index).multi_processor_count
    target = _DKDV_PROGRAMS_PER_SM * _SM_COUNT[index]
    split = 1
    while group % (2 * split) == 0 and programs * split < target:
        split *= 2
    return split


def _like(t):
    out = torch.empty_like(t)
    if out.stride(-1) != 1:
        out = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    return out


def _triton_backward(dout, q, k, v, out, lse, meta, causal, sm_scale):
    batch, sequence, heads, dim = q.shape
    kv_heads = k.shape[2]
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    dq, dk, dv = _like(q), _like(k), _like(v)
    delta = torch.empty_like(lse)
    block_d = _block_d(dim)
    even_d = block_d == dim
    pre_block = 64
    _bwd_preprocess_kernel[(triton.cdiv(sequence, pre_block), batch * heads)](
        out,
        dout,
        delta,
        meta.q_perm,
        meta.q_count,
        *out.stride()[:3],
        *dout.stride()[:3],
        sequence,
        heads,
        HEAD_DIM=dim,
        BLOCK_D=block_d,
        BLOCK_M=pre_block,
        EVEN_D=even_d,
        num_warps=4,
    )
    cfg = _kernel_config("dkdv", block_d)
    group = heads // kv_heads
    split = _dkdv_split(triton.cdiv(sequence, cfg.block_n) * batch * kv_heads, group, q.device)
    if split > 1:
        # Few KV tiles: spread the query heads of a group over programs and sum
        # their fp32 partial dK/dV afterwards.
        part_shape = (batch, sequence, kv_heads * split, dim)
        dk_out = torch.empty(part_shape, dtype=torch.float32, device=q.device)
        dv_out = torch.empty(part_shape, dtype=torch.float32, device=q.device)
    else:
        dk_out, dv_out = dk, dv
    _bwd_dkdv_kernel[(triton.cdiv(sequence, cfg.block_n), batch * kv_heads * split)](
        q,
        k,
        v,
        dout,
        dk_out,
        dv_out,
        lse,
        delta,
        meta.q_perm,
        meta.q_count,
        meta.q_excl,
        meta.k_perm,
        meta.k_count,
        sm_scale * _LOG2E,
        sm_scale,
        *q.stride()[:3],
        *k.stride()[:3],
        *v.stride()[:3],
        *dout.stride()[:3],
        *dk_out.stride()[:3],
        *dv_out.stride()[:3],
        sequence,
        heads,
        kv_heads,
        group,
        split,
        HEAD_DIM=dim,
        BLOCK_D=block_d,
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.block_n,
        CAUSAL=causal,
        EVEN_D=even_d,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    if split > 1:
        dk.copy_(dk_out.view(batch, sequence, kv_heads, split, dim).sum(3))
        dv.copy_(dv_out.view(batch, sequence, kv_heads, split, dim).sum(3))
    cfg = _kernel_config("dq", block_d)
    _bwd_dq_kernel[(triton.cdiv(sequence, cfg.block_m), batch * heads)](
        q,
        k,
        v,
        dout,
        dq,
        lse,
        delta,
        meta.q_perm,
        meta.q_count,
        meta.k_perm,
        meta.k_count,
        meta.k_incl,
        sm_scale * _LOG2E,
        sm_scale,
        *q.stride()[:3],
        *k.stride()[:3],
        *v.stride()[:3],
        *dout.stride()[:3],
        *dq.stride()[:3],
        sequence,
        heads,
        heads // kv_heads,
        HEAD_DIM=dim,
        BLOCK_D=block_d,
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.block_n,
        CAUSAL=causal,
        EVEN_D=even_d,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return dq, dk, dv


class _TritonMaskedKVAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q_perm, q_count, q_excl, k_perm, k_count, k_incl, causal, scale):
        meta = MaskedKVMetadata(q_perm, q_count, q_excl, k_perm, k_count, k_incl)
        out, lse = _triton_forward(q, k, v, meta, causal, scale)
        ctx.save_for_backward(q, k, v, out, lse, *meta)
        ctx.causal = causal
        ctx.scale = scale
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        q, k, v, out, lse, *meta = ctx.saved_tensors
        dq, dk, dv = _triton_backward(
            dout, q, k, v, out, lse, MaskedKVMetadata(*meta), ctx.causal, ctx.scale
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# FlexAttention backend: compaction outside, block-sparse causal mask inside
# ---------------------------------------------------------------------------

_FLEX_BLOCK = 128
_COMPILED_FLEX = None


def _flex_attention_fn(compile_flex):
    from torch.nn.attention.flex_attention import flex_attention

    global _COMPILED_FLEX
    if not compile_flex:
        return flex_attention
    if _COMPILED_FLEX is None:
        _COMPILED_FLEX = torch.compile(flex_attention, dynamic=False)
    return _COMPILED_FLEX


def _flex_block_mask(meta, causal, sequence, block=_FLEX_BLOCK):
    """BlockMask over compacted indices, built on device without host syncs."""
    from torch.nn.attention.flex_attention import BlockMask

    device = meta.q_perm.device
    batch = meta.q_perm.shape[0]
    blocks = (sequence + block - 1) // block
    starts = torch.arange(blocks, dtype=torch.int32, device=device) * block
    nq = meta.q_count[:, None]
    nk = meta.k_count[:, None]
    live = starts[None] < nq
    if causal:
        first = starts.clamp(max=sequence - 1)[None].expand(batch, blocks)
        last = (torch.minimum(starts[None] + block, nq) - 1).clamp(min=0)
        kv_full = meta.k_incl.gather(1, meta.q_perm.gather(1, first.long()).long())
        kv_high = meta.k_incl.gather(1, meta.q_perm.gather(1, last.long()).long())
    else:
        kv_full = nk.expand(batch, blocks)
        kv_high = nk.expand(batch, blocks)
    zero = torch.zeros_like(kv_full)
    full_blocks = torch.where(live, kv_full // block, zero)
    total_blocks = torch.where(live, (kv_high + block - 1) // block, zero)
    columns = torch.arange(blocks, dtype=torch.int32, device=device)
    full_indices = columns.expand(batch, 1, blocks, blocks).contiguous()
    partial_indices = (full_blocks[:, None, :, None] + columns).clamp(max=blocks - 1)
    q_perm, k_perm = meta.q_perm, meta.k_perm
    q_count, k_count = meta.q_count, meta.k_count

    def mask_mod(b, h, q_idx, kv_idx):
        allowed = (kv_idx < k_count[b]) & (q_idx < q_count[b])
        if causal:
            allowed = allowed & (k_perm[b, kv_idx] <= q_perm[b, q_idx])
        return allowed

    return BlockMask.from_kv_blocks(
        (total_blocks - full_blocks)[:, None].contiguous(),
        partial_indices.to(torch.int32).contiguous(),
        full_blocks[:, None].contiguous(),
        full_indices,
        BLOCK_SIZE=block,
        mask_mod=mask_mod,
        seq_lengths=(sequence, sequence),
    )


def _gather_rows(x, perm):
    index = perm.long()[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
    return x.gather(1, index)


def flex_masked_kv_attention(
    q, k, v, meta, *, causal=True, softmax_scale=None, compile_flex=True, return_lse=False
):
    """Compact (gather), run FlexAttention causally over the active prefix, scatter back.

    Differentiable through gather/flex/gather. On CPU pass ``compile_flex=False``
    to use the eager (O(S^2) memory) implementation.
    """
    batch, sequence, heads, dim = q.shape
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(dim)
    block_mask = _flex_block_mask(meta, causal, sequence)
    qc = _gather_rows(q, meta.q_perm).transpose(1, 2)
    kc = _gather_rows(k, meta.k_perm).transpose(1, 2)
    vc = _gather_rows(v, meta.k_perm).transpose(1, 2)
    flex = _flex_attention_fn(compile_flex)
    kwargs = dict(block_mask=block_mask, scale=scale, enable_gqa=heads != k.shape[2])
    try:
        from torch.nn.attention.flex_attention import AuxRequest
    except ImportError:  # torch < 2.9
        out_c, lse_c = flex(qc, kc, vc, return_lse=True, **kwargs)
    else:
        out_c, aux = flex(qc, kc, vc, return_aux=AuxRequest(lse=True), **kwargs)
        lse_c = aux.lse
    rows = torch.arange(sequence, device=q.device)
    valid = rows[None] < meta.q_count[:, None]  # [B, S] over compacted rows
    out_c = out_c.transpose(1, 2) * valid[:, :, None, None].to(out_c.dtype)
    # Inverse permutation: compacted row of each original position.
    inverse = torch.empty_like(meta.q_perm).scatter_(
        1,
        meta.q_perm.long(),
        torch.arange(sequence, dtype=torch.int32, device=q.device).expand(batch, sequence),
    )
    out = _gather_rows(out_c, inverse)
    if not return_lse:
        return out
    lse_c = torch.where(valid[:, None, :], lse_c.float(), float("-inf"))
    lse = lse_c.gather(2, inverse.long()[:, None, :].expand(-1, heads, -1))
    return out, lse.detach()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def triton_supported(q):
    """Whether the Triton backend can run ``q``."""
    return (
        HAVE_TRITON
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] <= 256
    )


def loop_masked_kv_attention(
    q,
    k,
    v,
    key_active=None,
    query_active=None,
    *,
    all_queries=False,
    causal=True,
    softmax_scale=None,
    metadata=None,
    backend="auto",
    return_lse=False,
):
    """Attention in which only active tokens provide K/V (see the module docstring).

    Args:
        q: ``[B, S, Hq, D]``; k, v: ``[B, S, Hkv, D]``; the last dim contiguous.
        key_active: ``[B, S]`` bool. Not needed when ``metadata`` is given.
        query_active: ``[B, S]`` bool; defaults to ``key_active``.
        all_queries: compute every query; only keys are masked.
        causal: restrict query i to keys j <= i.
        softmax_scale: defaults to ``1 / sqrt(D)``.
        metadata: precomputed ``build_masked_kv_metadata`` result, shareable by
            all layers of a recursion step (it then defines both masks).
        backend: ``"auto"`` (Triton when supported, else Flex), ``"triton"``,
            ``"flex"`` or ``"reference"``.
        return_lse: also return the ``[B, Hq, S]`` float32 logsumexp (no grad).

    Returns:
        ``out`` in q's dtype and memory layout, and optionally ``lse``.
    """
    _check_inputs(q, k, v)
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    if backend == "auto":
        backend = "triton" if triton_supported(q) else "flex"
    if backend == "reference":
        if metadata is not None:
            raise ValueError("The reference backend takes masks, not metadata")
        return masked_kv_attention_reference(
            q,
            k,
            v,
            key_active,
            query_active,
            all_queries=all_queries,
            causal=causal,
            softmax_scale=scale,
            return_lse=return_lse,
        )
    if metadata is None:
        if key_active is None:
            raise ValueError("Either key_active or metadata is required")
        metadata = build_masked_kv_metadata(key_active, query_active, all_queries=all_queries)
    elif key_active is not None or query_active is not None or all_queries:
        raise ValueError("Pass either masks or metadata, not both")
    if backend == "flex":
        return flex_masked_kv_attention(
            q,
            k,
            v,
            metadata,
            causal=causal,
            softmax_scale=scale,
            compile_flex=q.is_cuda,
            return_lse=return_lse,
        )
    if backend != "triton":
        raise ValueError(f"Unknown backend {backend!r}")
    if not triton_supported(q):
        raise ValueError("The Triton backend needs CUDA fp16/bf16 inputs with head_dim <= 256")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("The head dimension of q, k and v must be contiguous")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k and v must have the same dtype")
    out, lse = _TritonMaskedKVAttention.apply(q, k, v, *metadata, bool(causal), float(scale))
    return (out, lse) if return_lse else out
