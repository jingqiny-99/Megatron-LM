# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The attention-residual aggregation of the hybrid loop (``loop/attn_res.py``).

No process group. The custom Function (forward and backward) against the plain-autograd
reference (``attn_res_reference.native_attn_res``) with and without source masks, in fp32 and
float64; ``gradcheck``; the saved-tensor budget (references plus per-token statistics only, one
gradient tensor per source); chunking; zero init; masked sources (exactly zero weight and
gradient, also for a non-finite masked source); the block offsets and the source mask helper.
"""

import pytest
import torch

from megatron.core.transformer.loop import attn_res
from megatron.core.transformer.loop.attn_res import (
    AttnResStats,
    LoopAttentionResidual,
    attn_res_aggregate,
    attn_res_block_offsets,
    attn_res_chunk_tokens,
    history_lengths,
    source_mask_for_lengths,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.transformer.loop.attn_res_reference import native_attn_res

S, B, H = 6, 3, 16
EPS = 1e-6


def _values(n, dtype=torch.float32, seed=0, scale=None):
    generator = torch.Generator().manual_seed(seed)
    scales = scale or [0.5 + i for i in range(n)]
    return [
        (torch.randn(S, B, H, generator=generator, dtype=torch.float64) * scales[i]).to(dtype)
        for i in range(n)
    ]


def _params(dtype=torch.float32, seed=1):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(H, generator=generator, dtype=torch.float64).to(dtype)
    norm = (1.0 + 0.3 * torch.randn(H, generator=generator, dtype=torch.float64)).to(dtype)
    return query, norm


def _random_mask(n, seed=2, all_but_one=False):
    """``[n, B]`` valid mask: per row a random prefix length (or only source 0)."""
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, n + 1, (B,), generator=generator)
    if all_but_one:
        lengths = torch.ones(B, dtype=torch.int64)
    return torch.arange(n)[:, None] < lengths[None, :]


def _run(fn, values, query, norm, upstream):
    leaves = [v.detach().clone().requires_grad_(True) for v in values]
    q = query.detach().clone().requires_grad_(True)
    g = norm.detach().clone().requires_grad_(True)
    out = fn(leaves, q, g)
    (out * upstream).sum().backward()
    return out.detach(), [v.grad for v in leaves], q.grad, g.grad


def _custom(mask):
    return lambda v, q, g: attn_res_aggregate(q, g, EPS, v, mask)


def _native(mask):
    return lambda v, q, g: native_attn_res(v, q, g, EPS, mask)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("masked", ["none", "random", "all_but_one"])
@pytest.mark.parametrize("n", [2, 3, 6, 10])
def test_function_matches_the_naive_reference(n, masked, dtype):
    """T-A1."""
    values = _values(n, dtype)
    query, norm = _params(dtype)
    upstream = torch.randn(S, B, H, generator=torch.Generator().manual_seed(5)).to(dtype)
    mask = None if masked == "none" else _random_mask(n, all_but_one=masked == "all_but_one")
    actual = _run(_custom(mask), values, query, norm, upstream)
    expected = _run(_native(mask), values, query, norm, upstream)
    tol = dict(rtol=1e-10, atol=1e-10) if dtype == torch.float64 else dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[0], expected[0], **tol)
    for got, want in zip(actual[1], expected[1]):
        torch.testing.assert_close(got, want, **tol)
    torch.testing.assert_close(actual[2], expected[2], **tol)
    torch.testing.assert_close(actual[3], expected[3], **tol)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("n", [2, 3, 6])
def test_gradcheck(n, masked):
    """T-A2 (float64)."""
    values = [v.requires_grad_(True) for v in _values(n, torch.float64)]
    query, norm = (p.requires_grad_(True) for p in _params(torch.float64))
    mask = _random_mask(n) if masked else None
    assert torch.autograd.gradcheck(
        lambda q, g, *v: attn_res_aggregate(q, g, EPS, v, mask), (query, norm, *values)
    )


def test_one_source_is_the_identity_with_zero_parameter_gradients():
    """T-A4 (one source) and the module's zero init."""
    config = TransformerConfig(num_layers=1, hidden_size=H, num_attention_heads=1)
    module = LoopAttentionResidual(config)
    assert module.pseudo_query.dtype == torch.float32 and module.pseudo_query.keep_in_fp32
    assert torch.equal(module.pseudo_query, torch.zeros(H))
    assert torch.equal(module.key_norm_weight, torch.ones(H))
    value = _values(1)[0].requires_grad_(True)
    out = module([value])
    assert torch.equal(out, value)
    out.sum().backward()
    assert torch.equal(module.pseudo_query.grad, torch.zeros(H))
    assert torch.equal(module.key_norm_weight.grad, torch.zeros(H))
    assert torch.equal(value.grad, torch.ones_like(value))


def test_zero_init_is_the_mean_and_trains_the_query():
    """T-A4: w = 0 gives the mean of the sources; the pseudo-query gets a gradient."""
    config = TransformerConfig(num_layers=1, hidden_size=H, num_attention_heads=1)
    module = LoopAttentionResidual(config)
    values = _values(4)
    out = module(values)
    torch.testing.assert_close(out, torch.stack(values).mean(0), rtol=1e-6, atol=1e-6)
    upstream = torch.randn(S, B, H, generator=torch.Generator().manual_seed(9))
    (out * upstream).sum().backward()
    assert module.pseudo_query.grad.abs().sum() > 0
    # w = 0: the key-norm gradient is dq * w = 0.
    assert torch.equal(module.key_norm_weight.grad, torch.zeros(H))


def test_masked_sources_have_exactly_zero_weight_and_gradient():
    """T-A5, including a non-finite masked source."""
    n = 5
    values = _values(n)
    values[3][:, 1] = float("nan")  # row 1 does not read source 3
    values[4][:, 1] = float("inf")
    mask = torch.ones(n, B, dtype=torch.bool)
    mask[3:, 1] = False
    query, norm = _params()
    recorded = []

    class Sink:
        def add(self, key, alpha, row_weight):
            recorded.append(alpha.clone())

    leaves = [v.clone().requires_grad_(True) for v in values]
    out = attn_res_aggregate(query, norm, EPS, leaves, mask, (Sink(), ("core", 0, 1), None))
    assert torch.isfinite(out[:, 1]).all()
    alpha = recorded[0]  # [n, S, B]
    assert torch.count_nonzero(alpha[3:, :, 1]) == 0
    torch.testing.assert_close(alpha.sum(0), torch.ones(S, B))
    out[:, 1].sum().backward()
    for index in (3, 4):
        assert torch.count_nonzero(leaves[index].grad[:, 1]) == 0
    # The other rows read every source.
    assert torch.count_nonzero(alpha[:, :, 0] == 0) == 0


def test_bf16_values_keep_fp32_statistics():
    """T-A5: bf16 sources give a bf16 output with fp32 weights."""
    values = _values(3, torch.bfloat16)
    query, norm = _params()
    recorded = []

    class Sink:
        def add(self, key, alpha, row_weight):
            recorded.append(alpha)

    out = attn_res_aggregate(query, norm, EPS, values, None, (Sink(), ("core", 0, 1), None))
    assert out.dtype == torch.bfloat16 and recorded[0].dtype == torch.float32
    expected = native_attn_res(values, query, norm, EPS)
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


def test_saved_tensors_are_references_and_per_token_statistics():
    """T-A3: no fp32 ``[S, B, H]`` saved; the value gradients are distinct storages."""
    n = 6
    values = [v.requires_grad_(True) for v in _values(n, torch.bfloat16)]
    query, norm = (p.requires_grad_(True) for p in _params())
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = attn_res_aggregate(query, norm, EPS, values, _random_mask(n))
    pointers = {v.untyped_storage().data_ptr() for v in values}
    for tensor in saved:
        if tensor.shape == (S, B, H):
            assert tensor.untyped_storage().data_ptr() in pointers
            assert tensor.dtype == torch.bfloat16
        else:
            assert tensor.numel() <= n * S * B or tensor.numel() == H, tuple(tensor.shape)
    grads = torch.autograd.grad(out.float().sum(), values)
    storages = {grad.untyped_storage().data_ptr() for grad in grads}
    assert len(storages) == n
    assert all(grad.is_contiguous() and grad.dtype == torch.bfloat16 for grad in grads)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("masked", [False, True])
def test_chunking_matches_one_chunk(monkeypatch, masked, dtype):
    """T-A7: budgets forcing 1, 2, 3 and S chunks (uneven last chunk); dq is deterministic."""
    n = 4
    values = _values(n, dtype)
    query, norm = _params(dtype)
    upstream = torch.randn(S, B, H, generator=torch.Generator().manual_seed(4)).to(dtype)
    mask = _random_mask(n) if masked else None
    item = torch.finfo(dtype).bits // 8
    per_token = item * n * B * H
    results = {}
    for chunks, tokens in ((1, S), (2, 3), (3, 2), (S, 1), ("uneven", 4)):
        monkeypatch.setattr(attn_res, "ATTN_RES_CHUNK_BYTES", tokens * per_token)
        assert attn_res_chunk_tokens(n, B, H, item) == tokens
        results[chunks] = _run(_custom(mask), values, query, norm, upstream)
    reference = results.pop(1)
    tol = dict(rtol=1e-12, atol=1e-12) if dtype == torch.float64 else dict(rtol=1e-6, atol=1e-6)
    for result in results.values():
        torch.testing.assert_close(result[0], reference[0], **tol)
        for got, want in zip(result[1], reference[1]):
            torch.testing.assert_close(got, want, **tol)
        torch.testing.assert_close(result[2], reference[2], **tol)
        torch.testing.assert_close(result[3], reference[3], **tol)
    monkeypatch.setattr(attn_res, "ATTN_RES_CHUNK_BYTES", 2 * per_token)
    again = [_run(_custom(mask), values, query, norm, upstream) for _ in range(2)]
    assert torch.equal(again[0][2], again[1][2]) and torch.equal(again[0][3], again[1][3])


def test_block_offsets():
    """T-A6."""
    assert attn_res_block_offsets("*-*-*-*-", 1) == (0,)
    assert attn_res_block_offsets("*-*-*-*-", 2) == (0, 4)
    assert attn_res_block_offsets("*-*-*-*-", 4) == (0, 2, 4, 6)
    with pytest.raises(ValueError, match="equal blocks"):
        attn_res_block_offsets("*-*-*-", 4)
    with pytest.raises(ValueError, match="equal blocks"):
        attn_res_block_offsets("*-*-", 0)
    # 6 sublayers, k = 2: the cut at 3 falls inside a block.
    with pytest.raises(ValueError, match=r"\[3\] are not attention \+ MLP block boundaries"):
        attn_res_block_offsets("*-*-*-", 2)
    with pytest.raises(ValueError, match=r"\[3\] are not attention \+ MLP block boundaries"):
        attn_res_block_offsets("**--*-", 2)


def test_source_masks():
    """T-A6: history lengths and the host-decided source mask."""
    assert history_lengths([1, 3], 1) == (2, 4)
    assert history_lengths([1, 3], 2) == (3, 7)
    cpu = torch.device("cpu")
    assert source_mask_for_lengths(3, None, True, cpu) is None
    assert source_mask_for_lengths(3, (3, 5), True, cpu) is None
    mask = source_mask_for_lengths(4, (2, 5), False, cpu)
    assert mask.tolist() == [[True, True], [True, True], [False, True], [False, True]]
    mask = source_mask_for_lengths(3, (1, 3), True, cpu)
    assert mask.tolist() == [[True, True], [False, True], [False, True], [True, True]]


def test_stats_sink_weights_tokens():
    stats = AttnResStats()
    alpha = torch.full((2, S, B), 0.5)
    stats.add(("core", 0, 1), alpha, None)
    row_weight = torch.zeros(S, B)
    row_weight[:, 0] = 1
    stats.add(("core", 0, 1), alpha, row_weight)
    sums = stats.take()
    weights, tokens = sums[("core", 0, 1)]
    assert tokens.item() == S * B + S
    torch.testing.assert_close(weights, torch.full((2,), 0.5 * (S * B + S)))
    assert stats.take() == {}
