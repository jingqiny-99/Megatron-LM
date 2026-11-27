# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Masked-KV causal attention for loop-transformer MoR recursion.

CPU tests check the compaction metadata and the reference semantics (and the
FlexAttention backend in eager mode). GPU tests check the Triton kernels against
the reference, forward and backward, and CUDA-graph capture/replay with masks
that change between replays.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer import loop_masked_kv_attention as lmkv
from megatron.core.transformer.loop_masked_kv_attention import (
    HAVE_TRITON,
    build_masked_kv_metadata,
    flex_masked_kv_attention,
    loop_masked_kv_attention,
    masked_kv_attention_reference,
)

pytestmark = pytest.mark.launch_on_gb200

CUDA = torch.cuda.is_available()
needs_triton = pytest.mark.skipif(not (CUDA and HAVE_TRITON), reason="needs CUDA and Triton")
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")

PATTERNS = ["random", "all", "none", "first", "last", "clustered", "mixed"]


def make_mask(pattern, batch, sequence, fraction=0.3, seed=0, device="cpu"):
    """[B, S] bool masks covering scattered, empty, full, edge and clustered cases."""
    generator = torch.Generator().manual_seed(seed)
    if pattern == "random":
        mask = torch.rand(batch, sequence, generator=generator) < fraction
    elif pattern == "all":
        mask = torch.ones(batch, sequence, dtype=torch.bool)
    elif pattern == "none":
        mask = torch.zeros(batch, sequence, dtype=torch.bool)
    elif pattern == "first":
        mask = torch.zeros(batch, sequence, dtype=torch.bool)
        mask[:, 0] = True
    elif pattern == "last":
        mask = torch.zeros(batch, sequence, dtype=torch.bool)
        mask[:, -1] = True
    elif pattern == "clustered":
        mask = torch.zeros(batch, sequence, dtype=torch.bool)
        length = max(1, int(round(fraction * sequence)))
        for b in range(batch):
            start = int(torch.randint(0, sequence - length + 1, (1,), generator=generator))
            mask[b, start : start + length] = True
    elif pattern == "mixed":
        # A different regime per sequence: random, all, none, late half.
        mask = torch.rand(batch, sequence, generator=generator) < fraction
        if batch > 1:
            mask[1] = True
        if batch > 2:
            mask[2] = False
        if batch > 3:
            mask[3] = torch.arange(sequence) >= sequence // 2
    else:
        raise ValueError(pattern)
    return mask.to(device)


def make_qkv(batch, sequence, heads, kv_heads, dim, dtype, device, seed=0):
    generator = torch.Generator(device=device).manual_seed(seed)
    shape_q = (batch, sequence, heads, dim)
    shape_kv = (batch, sequence, kv_heads, dim)
    q = torch.randn(shape_q, generator=generator, device=device, dtype=torch.float32)
    k = torch.randn(shape_kv, generator=generator, device=device, dtype=torch.float32)
    v = torch.randn(shape_kv, generator=generator, device=device, dtype=torch.float32)
    return [t.to(dtype).requires_grad_() for t in (q, k, v)]


def query_mask_for(mode, key_active, seed=1):
    """(query_active, all_queries) for the three query modes."""
    if mode == "same":
        return None, False
    if mode == "all":
        return None, True
    generator = torch.Generator().manual_seed(seed)
    query = torch.rand(key_active.shape, generator=generator) < 0.6
    return query.to(key_active.device), False


def effective_query_mask(key_active, query_active, all_queries):
    if all_queries:
        return torch.ones_like(key_active)
    return key_active if query_active is None else query_active


def allowed_pairs(key_active, query_mask, causal):
    sequence = key_active.shape[1]
    allowed = query_mask[:, :, None] & key_active[:, None, :]
    if causal:
        allowed &= torch.ones(sequence, sequence, dtype=torch.bool, device=key_active.device).tril()
    return allowed  # [B, Sq, Sk]


# ---------------------------------------------------------------------------
# CPU: metadata and reference semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("sequence", [1, 7, 130])
def test_metadata_is_a_stable_active_first_permutation(pattern, sequence):
    key_active = make_mask(pattern, 4, sequence, seed=sequence)
    query_active = make_mask("random", 4, sequence, fraction=0.5, seed=sequence + 1)
    for meta_q, mask_q, meta in (
        ("k", key_active, build_masked_kv_metadata(key_active)),
        ("q", query_active, build_masked_kv_metadata(key_active, query_active)),
        (
            "all",
            torch.ones_like(key_active),
            build_masked_kv_metadata(key_active, all_queries=True),
        ),
    ):
        for perm, count, mask in (
            (meta.q_perm, meta.q_count, mask_q),
            (meta.k_perm, meta.k_count, key_active),
        ):
            assert perm.dtype == count.dtype == torch.int32
            for b in range(4):
                active = mask[b].nonzero().flatten().int()
                inactive = (~mask[b]).nonzero().flatten().int()
                assert int(count[b]) == active.numel(), meta_q
                torch.testing.assert_close(perm[b], torch.cat([active, inactive]))
        positions = torch.arange(sequence)
        torch.testing.assert_close(meta.k_incl, key_active.int().cumsum(1, dtype=torch.int32))
        expected_excl = mask_q[:, None, :] & (positions[None, None, :] < positions[None, :, None])
        torch.testing.assert_close(meta.q_excl, expected_excl.sum(-1, dtype=torch.int32))


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("query_mode", ["same", "separate", "all"])
def test_reference_matches_per_row_softmax(causal, query_mode):
    batch, sequence, heads, kv_heads, dim = 3, 11, 4, 2, 8
    key_active = make_mask("mixed", batch, sequence, fraction=0.5, seed=3)
    key_active[0, 0] = False  # a query whose own key is inactive
    query_active, all_queries = query_mask_for(query_mode, key_active)
    q, k, v = make_qkv(batch, sequence, heads, kv_heads, dim, torch.float64, "cpu")
    scale = 0.37
    out, lse = masked_kv_attention_reference(
        q,
        k,
        v,
        key_active,
        query_active,
        all_queries=all_queries,
        causal=causal,
        softmax_scale=scale,
        return_lse=True,
    )
    query_mask = effective_query_mask(key_active, query_active, all_queries)
    allowed = allowed_pairs(key_active, query_mask, causal)
    group = heads // kv_heads
    for b in range(batch):
        for h in range(heads):
            for i in range(sequence):
                keys = allowed[b, i].nonzero().flatten()
                if keys.numel() == 0:
                    assert torch.all(out[b, i, h] == 0)
                    assert lse[b, h, i] == -math.inf
                    continue
                scores = (k[b, keys, h // group] @ q[b, i, h]) * scale
                expected = torch.softmax(scores, 0) @ v[b, keys, h // group]
                torch.testing.assert_close(out[b, i, h], expected)
                torch.testing.assert_close(lse[b, h, i].double(), torch.logsumexp(scores, 0))


def test_reference_gradients_are_zero_on_masked_rows_without_nan():
    batch, sequence, heads, kv_heads, dim = 2, 33, 4, 1, 16
    key_active = make_mask("random", batch, sequence, fraction=0.2, seed=5)
    key_active[:, :4] = False  # leading queries have no visible keys
    query_active = torch.ones_like(key_active)
    q, k, v = make_qkv(batch, sequence, heads, kv_heads, dim, torch.float32, "cpu")
    out = masked_kv_attention_reference(q, k, v, key_active, query_active)
    out.backward(torch.randn_like(out))
    for grad in (q.grad, k.grad, v.grad):
        assert torch.isfinite(grad).all()
    assert torch.all(q.grad[:, :4] == 0)
    assert torch.all(k.grad[~key_active] == 0) and torch.all(v.grad[~key_active] == 0)


def test_reference_matches_loop_block_mask_on_active_rows():
    """Same math as the arbitrary-mask path LoopTransformerBlock uses today."""
    block = pytest.importorskip("megatron.core.transformer.loop_transformer_block")
    batch, sequence, heads, kv_heads, dim = 2, 40, 4, 2, 16
    key_active = make_mask("random", batch, sequence, fraction=0.4, seed=7)
    q, k, v = make_qkv(batch, sequence, heads, kv_heads, dim, torch.float64, "cpu")
    # active_causal_mask takes [S, B] and returns [B, 1, S, S], True = masked out.
    masked_out = block.active_causal_mask(key_active.transpose(0, 1))
    group = heads // kv_heads
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2).repeat_interleave(group, 1),
        v.transpose(1, 2).repeat_interleave(group, 1),
        attn_mask=~masked_out,
    ).transpose(1, 2)
    out = masked_kv_attention_reference(q, k, v, key_active)
    torch.testing.assert_close(out[key_active], expected[key_active])
    assert torch.all(out[~key_active] == 0)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("query_mode", ["same", "separate", "all"])
def test_flex_backend_eager_forward_matches_reference(causal, query_mode):
    """FlexAttention has no CPU backward; the CUDA test below covers gradients."""
    pytest.importorskip("torch.nn.attention.flex_attention")
    batch, sequence, heads, kv_heads, dim = 4, 300, 4, 2, 16
    key_active = make_mask("mixed", batch, sequence, fraction=0.3, seed=11)
    query_active, all_queries = query_mask_for(query_mode, key_active)
    q, k, v = (
        t.detach() for t in make_qkv(batch, sequence, heads, kv_heads, dim, torch.float32, "cpu")
    )
    meta = build_masked_kv_metadata(key_active, query_active, all_queries=all_queries)
    out, lse = flex_masked_kv_attention(
        q, k, v, meta, causal=causal, compile_flex=False, return_lse=True
    )
    ref_out, ref_lse = masked_kv_attention_reference(
        q, k, v, key_active, query_active, all_queries=all_queries, causal=causal, return_lse=True
    )
    torch.testing.assert_close(out, ref_out, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(lse, ref_lse, atol=2e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# GPU: Triton kernels
# ---------------------------------------------------------------------------


def assert_close_relative(actual, expected, tolerance, name, floor=0.05):
    """Max error relative to the reference's max magnitude, floored.

    The floor matters when the exact result is zero, e.g. dq of a row that sees
    a single key (softmax == 1), where the fp32 reference is rounding noise.
    Inputs are unit normal, so typical outputs and gradients are O(0.1..1).
    """
    assert torch.isfinite(actual).all(), f"{name} has non-finite values"
    scale = expected.abs().max().clamp_min(floor)
    error = (actual.float() - expected.float()).abs().max() / scale
    assert error <= tolerance, f"{name}: relative max error {error:.3e} > {tolerance}"


def check_against_reference(
    q, k, v, key_active, query_active=None, all_queries=False, causal=True, backend="triton"
):
    """Forward/backward parity with the fp32 reference, plus exact zeros."""
    out, lse = loop_masked_kv_attention(
        q,
        k,
        v,
        key_active,
        query_active,
        all_queries=all_queries,
        causal=causal,
        backend=backend,
        return_lse=True,
    )
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout)
    leaves = [t.detach().float().requires_grad_() for t in (q, k, v)]
    ref_out, ref_lse = masked_kv_attention_reference(
        *leaves, key_active, query_active, all_queries=all_queries, causal=causal, return_lse=True
    )
    ref_grads = torch.autograd.grad(ref_out, leaves, dout.float())
    tolerance = 2e-2 if q.dtype == torch.bfloat16 else 5e-3
    assert_close_relative(out, ref_out, tolerance, "out")
    finite = torch.isfinite(ref_lse)
    assert torch.equal(torch.isfinite(lse), finite)
    if finite.any():
        assert_close_relative(lse[finite], ref_lse[finite], 1e-3, "lse")
    for name, grad, ref in zip(("dq", "dk", "dv"), grads, ref_grads):
        assert_close_relative(grad, ref, 2 * tolerance, name)
    # Exact zeros where nothing is attended and for keys that are never seen.
    query_mask = effective_query_mask(key_active, query_active, all_queries)
    empty = ~allowed_pairs(key_active, query_mask, causal).any(-1)
    assert torch.all(out[empty] == 0)
    assert torch.all(grads[0][empty] == 0)
    assert torch.all(grads[1][~key_active] == 0) and torch.all(grads[2][~key_active] == 0)


@needs_triton
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("sequence", [1, 77, 256, 1000])
def test_triton_patterns(pattern, sequence):
    key_active = make_mask(pattern, 4, sequence, fraction=0.3, seed=sequence, device="cuda")
    q, k, v = make_qkv(4, sequence, 4, 2, 128, torch.bfloat16, "cuda", seed=sequence)
    check_against_reference(q, k, v, key_active)


@needs_triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dim", [64, 128, 192])
@pytest.mark.parametrize("heads,kv_heads", [(4, 4), (8, 2), (16, 4)])
def test_triton_dtypes_head_dims_gqa(dtype, dim, heads, kv_heads):
    key_active = make_mask("random", 2, 600, fraction=0.5, seed=dim, device="cuda")
    q, k, v = make_qkv(2, 600, heads, kv_heads, dim, dtype, "cuda", seed=heads)
    check_against_reference(q, k, v, key_active)


@needs_triton
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("query_mode", ["same", "separate", "all"])
@pytest.mark.parametrize("pattern", ["random", "mixed", "clustered"])
def test_triton_query_modes(causal, query_mode, pattern):
    key_active = make_mask(pattern, 4, 513, fraction=0.25, seed=2, device="cuda")
    query_active, all_queries = query_mask_for(query_mode, key_active)
    q, k, v = make_qkv(4, 513, 8, 2, 128, torch.bfloat16, "cuda", seed=3)
    check_against_reference(q, k, v, key_active, query_active, all_queries, causal)


@needs_triton
def test_triton_active_query_without_any_visible_key():
    key_active = torch.zeros(2, 300, dtype=torch.bool, device="cuda")
    key_active[:, 150:] = torch.rand(2, 150, device="cuda") < 0.5
    q, k, v = make_qkv(2, 300, 4, 1, 128, torch.bfloat16, "cuda")
    check_against_reference(q, k, v, key_active, all_queries=True)


@needs_triton
def test_triton_ignores_nonfinite_values_in_masked_rows():
    """Inactive rows may hold anything: their outputs and gradients stay exact zeros."""
    key_active = make_mask("random", 2, 400, fraction=0.3, seed=9, device="cuda")
    q, k, v = make_qkv(2, 400, 4, 2, 128, torch.bfloat16, "cuda")
    with torch.no_grad():
        q[~key_active] = float("nan")
        k[~key_active] = float("inf")
        v[~key_active] = float("nan")
    out = loop_masked_kv_attention(q, k, v, key_active)
    dout = torch.randn_like(out)
    dout[~key_active] = float("nan")
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    for name, t in (("out", out), ("dq", dq), ("dk", dk), ("dv", dv)):
        assert torch.isfinite(t).all(), name
        assert torch.all(t[~key_active] == 0), name


@needs_triton
def test_triton_accepts_megatron_sbhd_views_and_keeps_their_layout():
    sequence, batch, heads, kv_heads, dim = 700, 2, 8, 2, 128
    key_active = make_mask("random", batch, sequence, fraction=0.4, seed=4, device="cuda")
    q_sbhd = torch.randn(sequence, batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    k_sbhd = torch.randn(sequence, batch, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
    v_sbhd = torch.randn(sequence, batch, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
    leaves = [t.requires_grad_() for t in (q_sbhd, k_sbhd, v_sbhd)]
    q, k, v = (t.transpose(0, 1) for t in leaves)
    out = loop_masked_kv_attention(q, k, v, key_active)
    assert out.transpose(0, 1).is_contiguous()
    check_against_reference(q, k, v, key_active)


@needs_triton
@pytest.mark.parametrize("programs_per_sm", [0, 1000])
@pytest.mark.parametrize("heads,kv_heads", [(8, 2), (16, 4)])
def test_triton_dkdv_gqa_loop_and_head_split(monkeypatch, programs_per_sm, heads, kv_heads):
    """dK/dV either loops all query heads of a group in one program (split 1)
    or spreads them over programs and sums fp32 partials (split = group)."""
    monkeypatch.setattr(lmkv, "_DKDV_PROGRAMS_PER_SM", programs_per_sm)
    expected_split = 1 if programs_per_sm == 0 else heads // kv_heads
    assert lmkv._dkdv_split(8, heads // kv_heads, torch.device("cuda")) == expected_split
    key_active = make_mask("mixed", 4, 900, fraction=0.4, seed=13, device="cuda")
    q, k, v = make_qkv(4, 900, heads, kv_heads, 128, torch.bfloat16, "cuda", seed=13)
    check_against_reference(q, k, v, key_active)


@needs_triton
def test_triton_metadata_shared_across_calls():
    key_active = make_mask("random", 2, 512, fraction=0.3, seed=6, device="cuda")
    meta = build_masked_kv_metadata(key_active)
    q, k, v = make_qkv(2, 512, 4, 2, 128, torch.bfloat16, "cuda")
    shared = loop_masked_kv_attention(q, k, v, metadata=meta)
    direct = loop_masked_kv_attention(q, k, v, key_active)
    assert torch.equal(shared, direct)


@needs_triton
def test_triton_has_no_host_sync():
    key_active = make_mask("random", 2, 1024, fraction=0.3, seed=8, device="cuda")
    q, k, v = make_qkv(2, 1024, 4, 2, 128, torch.bfloat16, "cuda")
    loop_masked_kv_attention(q, k, v, key_active).sum().backward()  # compile first
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        meta = build_masked_kv_metadata(key_active)
        out = loop_masked_kv_attention(q, k, v, metadata=meta)
        torch.autograd.backward(out, torch.ones_like(out))
    finally:
        torch.cuda.set_sync_debug_mode(previous)


GRAPH_MASKS = ["random", "all", "none", "clustered", "first", "mixed"]


@needs_triton
@pytest.mark.parametrize("query_mode", ["same", "all"])
def test_triton_cuda_graph_replays_with_changing_masks(query_mode):
    batch, sequence, heads, kv_heads, dim = 4, 640, 8, 2, 128
    all_queries = query_mode == "all"
    q, k, v = make_qkv(batch, sequence, heads, kv_heads, dim, torch.bfloat16, "cuda")
    static_mask = torch.ones(batch, sequence, dtype=torch.bool, device="cuda")
    static_dout = torch.randn(q.shape, device="cuda", dtype=q.dtype)

    def step():
        meta = build_masked_kv_metadata(static_mask, all_queries=all_queries)
        out = loop_masked_kv_attention(q, k, v, metadata=meta)
        grads = torch.autograd.grad(out, (q, k, v), static_dout)
        return out, grads

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out, static_grads = step()

    for index, pattern in enumerate(GRAPH_MASKS):
        mask = make_mask(pattern, batch, sequence, fraction=0.2 + 0.1 * index, seed=index)
        static_mask.copy_(mask)
        static_dout.normal_()
        graph.replay()
        leaves = [t.detach().float().requires_grad_() for t in (q, k, v)]
        ref_out = masked_kv_attention_reference(*leaves, static_mask, all_queries=all_queries)
        ref_grads = torch.autograd.grad(ref_out, leaves, static_dout.float())
        assert_close_relative(static_out, ref_out, 2e-2, f"{pattern} out")
        for name, grad, ref in zip(("dq", "dk", "dv"), static_grads, ref_grads):
            assert_close_relative(grad, ref, 4e-2, f"{pattern} {name}")


@needs_triton
def test_triton_make_graphed_callables():
    batch, sequence, heads, kv_heads, dim = 2, 512, 4, 2, 128

    class Attention(torch.nn.Module):
        def forward(self, q, k, v, key_active):
            return loop_masked_kv_attention(q, k, v, key_active)

    q, k, v = make_qkv(batch, sequence, heads, kv_heads, dim, torch.bfloat16, "cuda")
    sample_mask = torch.ones(batch, sequence, dtype=torch.bool, device="cuda")
    graphed = torch.cuda.make_graphed_callables(
        Attention(),
        (
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            sample_mask,
        ),
    )
    for seed in range(3):
        mask = make_mask(
            "random", batch, sequence, fraction=0.3 + 0.2 * seed, seed=seed, device="cuda"
        )
        out = graphed(q, k, v, mask)
        dout = torch.randn_like(out)
        grads = torch.autograd.grad(out, (q, k, v), dout)
        leaves = [t.detach().float().requires_grad_() for t in (q, k, v)]
        ref_out = masked_kv_attention_reference(*leaves, mask)
        ref_grads = torch.autograd.grad(ref_out, leaves, dout.float())
        assert_close_relative(out, ref_out, 2e-2, "out")
        for name, grad, ref in zip(("dq", "dk", "dv"), grads, ref_grads):
            assert_close_relative(grad, ref, 4e-2, name)


# ---------------------------------------------------------------------------
# GPU: FlexAttention backend (compiled)
# ---------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize("query_mode", ["same", "all"])
def test_flex_backend_cuda_matches_reference(query_mode):
    pytest.importorskip("torch.nn.attention.flex_attention")
    key_active = make_mask("mixed", 4, 1000, fraction=0.3, seed=12, device="cuda")
    query_active, all_queries = query_mask_for(query_mode, key_active)
    q, k, v = make_qkv(4, 1000, 8, 2, 128, torch.bfloat16, "cuda")
    check_against_reference(q, k, v, key_active, query_active, all_queries, backend="flex")
