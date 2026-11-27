# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SBHD loop reuse versus an independent dense, masked attention oracle."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.core.transformer.experimental_attention_variant import csa
from megatron.core.transformer.loop.cross_loop_kv import CrossLoopPass


class _Compressor(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio
        self.scale = nn.Parameter(torch.tensor(0.7))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x[self.ratio - 1 :: self.ratio] * self.scale


class _Indexer(nn.Module):
    index_topk = 1
    softmax_scale = 1.0

    def __init__(self, pg):
        super().__init__()
        self.pg_collection = pg
        self.calls = 0

    def forward_before_topk(self, x, qr, packed_seq_params):
        self.calls += 1
        q = torch.ones_like(qr[..., :1]).unsqueeze(-2)
        # The highest positive first channel wins among completed groups.
        return q, x[3::4, :, :1], q.squeeze(-1)


def _dense_reference(q, window_kv, sparse_x, index_x, scale, ratio, sink, window):
    """Compute all logits and apply causal window and compressed-bank masks directly."""
    sq, batch, heads, dim = q.shape
    row = torch.arange(sq)[:, None]
    pos = torch.arange(sq)[None, :]
    allowed = ((pos <= row) & (pos > row - window))[None].expand(batch, -1, -1)
    kv = window_kv
    if ratio and sq >= ratio:
        sparse = sparse_x[ratio - 1 :: ratio] * scale
        kv = torch.cat((kv, sparse))
        visible = (row + 1 >= (torch.arange(sparse.shape[0]) + 1) * ratio)[None]
        if index_x is not None:
            scores = index_x[3::4, :, 0].T[:, None, :].expand(-1, sq, -1)
            best = scores.masked_fill(~visible, -torch.inf).argmax(dim=-1, keepdim=True)
            visible = visible & (torch.arange(sparse.shape[0]) == best)
        allowed = torch.cat((allowed, visible.expand(batch, -1, -1)), dim=-1)

    logits = torch.einsum("sbhd,kbd->bshk", q, kv) * dim**-0.5
    logits = logits.masked_fill(~allowed[:, :, None], -torch.inf)
    with_sink = torch.cat((logits, sink[None, None, :, None].expand(batch, sq, -1, 1)), dim=-1)
    probs = with_sink.softmax(dim=-1)[..., :-1]
    return torch.einsum("bshk,kbd->sbhd", probs, kv).reshape(sq, batch, heads * dim)


@pytest.mark.parametrize(
    "ratio,sq,reuse_kv,reuse_indices,dense,compressions,selections",
    [
        (4, 13, True, True, False, 1, 1),
        (4, 13, True, False, False, 1, 1),
        (4, 13, False, True, False, 3, 1),
        (4, 13, False, False, False, 3, 3),
        (4, 13, True, False, True, 1, 0),
        (128, 130, True, False, False, 1, 0),
        (4, 3, True, True, False, 0, 0),
        (0, 13, True, True, False, 0, 0),
    ],
)
def test_consumer_loss_reaches_expected_kv_and_query(
    monkeypatch, ratio, sq, reuse_kv, reuse_indices, dense, compressions, selections
):
    torch.manual_seed(2341)
    batch, heads, dim, window = 2, 2, 4, 2
    pg = SimpleNamespace(tp=SimpleNamespace(size=lambda: 1), cp=None)
    compressor, indexer = _Compressor(ratio), _Indexer(pg)
    config = SimpleNamespace(
        csa_window_size=window,
        v_head_dim=dim,
        qk_pos_emb_head_dim=2,
        num_attention_heads=heads,
        csa_dense_mode=dense,
        attention_backend="unfused",
        dsa_cp_balance_indexer=False,
        apply_rope_fusion=False,
        loop_reuse_sparse_kv=reuse_kv,
        loop_reuse_indexer=reuse_indices,
        calculate_per_token_loss=False,
    )
    core = csa.CompressedSparseAttention(
        config,
        csa.CompressedSparseAttentionSubmodules(
            compressor=lambda **_: compressor, indexer=lambda **_: indexer
        ),
        layer_number=1,
        attn_mask_type=None,
        attention_type="self",
        pg_collection=pg,
        rotary_pos_emb=lambda *args, **kwargs: None,
        compress_ratio=ratio,
    )
    # The fixture uses zero-angle RoPE so the oracle can compare the attention output directly.
    monkeypatch.setattr(csa, "apply_rotary_pos_emb", lambda tensor, *args, **kwargs: tensor)
    states = [torch.randn(sq, batch, dim) for _ in range(3)]
    if ratio == 4:
        # Batch rows prefer different compressed positions and consumers reverse that order.
        ranks = torch.arange(1, sq // ratio + 1, dtype=torch.float32)
        scores = torch.stack([ranks, ranks.flip((0,))], dim=1)
        for step, x in enumerate(states):
            x[3::4, :, 0] = scores if step == 0 else scores.flip((0,))
    states = [x.requires_grad_() for x in states]
    windows = [torch.randn(sq, batch, dim, requires_grad=True) for _ in states]
    queries = [torch.randn(sq, batch, heads, dim, requires_grad=True) for _ in states]

    actual, expected, saved = [], [], {}
    for step, (q, local, x) in enumerate(zip(queries, windows, states)):
        with CrossLoopPass(step, saved).activate():
            actual.append(core(q, local.unsqueeze(2), None, None, x=x, qr=x))
        index_x = None
        if ratio == 4 and not dense:
            index_x = states[0] if reuse_kv or reuse_indices else x
        expected.append(
            _dense_reference(
                q,
                local,
                states[0] if reuse_kv else x,
                index_x,
                compressor.scale,
                ratio,
                core.attn_sink,
                window,
            )
        )
        torch.testing.assert_close(actual[-1], expected[-1])
    assert compressor.calls == compressions
    assert indexer.calls == selections

    # Consumer-only losses expose both a missing producer gradient and a stale window.
    leaves = [*states, *windows, *queries, compressor.scale, core.attn_sink]

    def consumer_grads(outputs):
        loss = sum(out.square().sum() for out in outputs[1:])
        return torch.autograd.grad(loss, leaves, allow_unused=True)

    for grad, ref in zip(consumer_grads(actual), consumer_grads(expected)):
        if ref is None:
            assert grad is None
        else:
            torch.testing.assert_close(grad, ref, atol=2e-5, rtol=1e-5)
