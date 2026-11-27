# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.experimental_attention_variant.dsa_loop_kv import (
    compact_source_topk,
    two_bank_topk,
)
from megatron.core.transformer.loop.cross_loop_kv import CrossLoopPass


def test_two_bank_padding():
    # A sentinel-based producer may interleave -1 with valid entries.
    raw = torch.tensor([[[-1, 0], [1, -1], [2, 0], [0, 2]]])
    ends = torch.arange(1, 5)
    indices, lengths = compact_source_topk(raw, None, causal_ends=ends)
    # A length-based producer may leave arbitrary values beyond the valid prefix.
    indices = indices.masked_fill(indices == -1, 3)
    selected, total = two_bank_topk(indices, lengths, window_size=2, causal_ends=ends, bank_size=4)
    assert selected.tolist() == [[[0, 4, -1, -1], [1, 4, 5, -1], [0, 2, 5, 6], [0, 2, 6, 7]]]
    assert total.tolist() == [[2, 3, 4, 4]]


def _dense_reference(query, key1, key2, up, topk, window):
    # This indexer scores each key by token position, so its causal top-k is exactly
    # the most recent topk positions. Each bank participates in the same softmax.
    seq, _, _, dim = query.shape
    row = torch.arange(seq, device=query.device)
    distance = row[:, None] - row
    banks = [key1] if key2 is None else [key1, key2]
    supports = [(distance >= 0) & (distance < span) for span in (topk, window)[: len(banks)]]
    scores = torch.einsum("sbhd,tbd->sbht", query, torch.cat(banks)[:, :, 0]) * dim**-0.5
    scores = scores.masked_fill(~torch.cat(supports, dim=-1)[:, None, None], -torch.inf)
    latent = torch.einsum(
        "sbht,tbd->sbhd", scores.softmax(dim=-1), torch.cat(banks)[:, :, 0, : up.size(2)]
    )
    return torch.einsum("sbhd,hvd->sbhv", latent, up).flatten(2)


@pytest.mark.parametrize("fused", [False, True], ids=["torch", "cudnn_flashmla"])
@pytest.mark.parametrize(
    "packed", [None, "mask", "lengths"], ids=["sbhd", "padded_thd", "padded_thd_lengths"]
)
def test_native_dsa_rounds_and_gradients(fused, packed, monkeypatch):
    if fused and (not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9):
        pytest.skip("cuDNN/FlashMLA needs SM90+")
    device = "cuda" if fused else "cpu"
    dtype = torch.bfloat16 if fused else torch.float32
    seq, batch, heads, k, window = 17, 2, 2, 4, 3
    documents = [slice(0, 7), slice(7, seq)] if packed else [slice(0, seq)]
    real = torch.ones(seq, dtype=torch.bool, device=device)
    packed_params = None
    if packed:
        batch = 1
        real[[6, 15, 16]] = False
        cu = torch.tensor([0, 6, 14], device=device, dtype=torch.int32)
        padded = torch.tensor([0, 7, 17], device=device, dtype=torch.int32)
        packed_params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                                        cu_seqlens_q_padded=padded, cu_seqlens_kv_padded=padded,
                                        max_seqlen_q=10, max_seqlen_kv=10)
        if packed == "mask":
            packed_params.real_token_mask_q = real
    rank, rope = (512, 64) if fused else (4, 2)
    torch.manual_seed(444)
    config = SimpleNamespace(
        num_layers=1,
        mtp_num_layers=0,
        dsa_indexer_topk=k,
        dsa_indexer_topk_freq=1,
        dsa_indexer_skip_topk_offset=0,
        loop_reuse_sparse_kv=True,
        loop_sparse_kv_window_size=window,
        kv_lora_rank=rank,
        qk_pos_emb_head_dim=rope,
        sequence_parallel=False,
        attention_backend="unfused",
        dsa_indexer_loss_coeff=0,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_scoring_relu=True,
        dsa_indexer_reuse_checkpoint_topk=False,
        calculate_per_token_loss=False,
        dsa_kernel_backend="cudnn" if fused else "none",
    )
    calls = []

    class PositionIndexer(torch.nn.Module):
        def forward_before_topk(self, x, qr, packed):
            calls.append("indexer")
            keys = torch.arange(seq, device=device).float()[:, None, None].expand(-1, batch, 1)
            q = torch.ones(seq, batch, 1, 1, device=device)
            return q, keys, q[..., 0]

    monkeypatch.setattr(dsa, "build_module", lambda *args, **kwargs: PositionIndexer())
    module = dsa.DSAttention(
        config,
        dsa.DSAttentionSubmodules(indexer=None),
        1,
        AttnMaskType.causal,
        "self",
        softmax_scale=(rank + rope) ** -0.5,
        pg_collection=SimpleNamespace(),
    )

    def tensor(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)

    q1, q2 = (tensor(seq, batch, heads, rank + rope) for _ in range(2))
    key1, key2 = (tensor(seq, batch, 1, rank + rope) for _ in range(2))
    up = tensor(heads, 4, rank)
    x = torch.zeros(seq, batch, 2, device=device)

    def run(query, key):
        return module(
            query, key, None, None, x, x, attn_mask_type=AttnMaskType.causal, up_v_weight=up, packed_seq_params=packed_params
        )

    bank = {}
    source = CrossLoopPass(0, bank)
    with source.activate():
        out1 = run(q1, key1)
    source.finish_source()
    assert calls == ["indexer"]

    # Turn on the indexer loss here: the consumer must neither score keys nor attach a loss.
    config.dsa_indexer_loss_coeff = 0.01
    if fused:
        config.attention_backend = "auto"
        # A silently declined external kernel would make this an unfused test.
        monkeypatch.setattr(
            dsa, "_unfused_absorbed_dsa_fn", lambda *a, **kw: pytest.fail("fallback")
        )
    with CrossLoopPass(1, bank).activate():
        out2 = run(q2, key2)
    assert calls == ["indexer"]

    inputs = (q1, key1, q2, key2, up)
    r_q1, r_k1, r_q2, r_k2, r_up = refs = [t.detach().float().requires_grad_() for t in inputs]
    ref1 = torch.cat([_dense_reference(r_q1[t], r_k1[t], None, r_up, k, window) for t in documents])
    ref2 = torch.cat([_dense_reference(r_q2[t], r_k1[t], r_k2[t], r_up, k, window) for t in documents])
    ref2 = ref2.masked_fill(~real[:, None, None], 0)
    gradients = tuple(torch.randn_like(t) * real[:, None, None] for t in (out1, out2))
    actual = torch.autograd.grad((out1, out2), inputs, gradients)
    expected = torch.autograd.grad((ref1, ref2), refs, tuple(g.float() for g in gradients))
    for observed, desired in zip((out1, out2, *actual), (ref1, ref2, *expected)):
        if fused:
            # BF16 projection and sparse attention round independently in the production path.
            assert (observed.float() - desired).norm() <= 0.03 * desired.norm()
        else:
            torch.testing.assert_close(observed, desired, atol=2e-5, rtol=2e-5)
