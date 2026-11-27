# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run under torchrun on two or more GPUs. Compare real CP collectives with dense MLA."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.loop.cross_loop_kv import CrossLoopPass


@pytest.fixture(scope="module")
def cp_group():
    if int(os.environ.get("WORLD_SIZE", 1)) < 2 or not torch.cuda.is_available():
        pytest.skip("run with torchrun and at least two GPUs")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    own_group = not dist.is_initialized()
    if own_group:
        dist.init_process_group("nccl")
    yield dist.group.WORLD
    if own_group:
        dist.destroy_process_group()


@pytest.mark.parametrize("fused", [False, True], ids=["fp32", "cudnn_flashmla"])
@pytest.mark.parametrize("packed", [False, True], ids=["sbhd", "padded_thd"])
@pytest.mark.parametrize("window", [1, 4, 6], ids=["local", "halo", "multi_chunk"])
def test_global_indices_window_and_remote_kv_gradients(
    cp_group, fused, window, packed, monkeypatch
):
    # The dense reference and the sharded attention multiply different GEMM
    # shapes. Use FP32 accumulators and operands for the FP32 comparison.
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    size, rank = cp_group.size(), cp_group.rank()
    # A window straddles a rank boundary; remote positions also dominate the
    # first-pass top-k (the indexer favors the earliest keys).
    chunk, k, heads = 4, 5, 2
    seq = 2 * size * chunk
    all_pos = torch.arange(seq, device="cuda")
    pos = torch.cat((all_pos.chunk(2 * size)[rank], all_pos.chunk(2 * size)[2 * size - 1 - rank]))
    document = torch.zeros_like(all_pos)
    real = torch.ones(seq, dtype=torch.bool, device="cuda")
    packed_params = None
    if packed:
        half = seq // 2
        document = all_pos // half
        real = (all_pos % half) < (half - 1 - document)
        chunks = [doc.chunk(2 * size) for doc in all_pos.split(half)]
        pos = torch.cat([part for doc in chunks for part in (doc[rank], doc[2 * size - rank - 1])])
        cu = torch.tensor([0, half - 1, seq - 3], device="cuda", dtype=torch.int32)
        padded = torch.tensor([0, half, seq], device="cuda", dtype=torch.int32)
        packed_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=padded,
            cu_seqlens_kv_padded=padded,
            max_seqlen_q=half,
            max_seqlen_kv=half,
        )
        packed_params.real_token_mask_q = real[pos]
    latent, rope = (512, 64) if fused else (4, 2)
    dtype = torch.bfloat16 if fused else torch.float32
    cfg = SimpleNamespace(
        num_layers=1,
        mtp_num_layers=0,
        dsa_indexer_topk=k,
        dsa_indexer_topk_freq=1,
        dsa_indexer_skip_topk_offset=0,
        loop_reuse_sparse_kv=True,
        loop_sparse_kv_window_size=window,
        kv_lora_rank=latent,
        qk_pos_emb_head_dim=rope,
        sequence_parallel=False,
        attention_backend="auto" if fused else "unfused",
        dsa_indexer_loss_coeff=0,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_scoring_relu=True,
        dsa_indexer_reuse_checkpoint_topk=False,
        calculate_per_token_loss=False,
        dsa_kernel_backend="cudnn" if fused else "none",
    )
    calls = []

    class PositionIndexer(torch.nn.Module):
        _kpool_gate_score = None

        def forward_before_topk(self, x, qr, packed):
            calls.append(1)
            q = torch.zeros((len(pos), 1, 32, 128), device="cuda", dtype=dtype)
            q[..., 0] = 1
            key = torch.zeros((len(pos), 1, 128), device="cuda", dtype=dtype)
            key[:, 0, 0] = seq - pos
            return q, key, torch.full_like(q[..., 0], 1 / 32, dtype=torch.float32)

    monkeypatch.setattr(dsa, "build_module", lambda *a, **kw: PositionIndexer())
    if fused:
        monkeypatch.setattr(
            dsa, "_unfused_absorbed_dsa_fn", lambda *a, **kw: pytest.fail("unfused MLA")
        )
    scale = (latent + rope) ** -0.5
    attn = dsa.DSAttention(
        cfg,
        dsa.DSAttentionSubmodules(indexer=None),
        1,
        AttnMaskType.causal,
        "self",
        softmax_scale=scale,
        cp_comm_type="allgather",
        pg_collection=SimpleNamespace(cp=cp_group),
    )
    gen = torch.Generator().manual_seed(54)

    def data(*shape):
        return torch.randn(shape, generator=gen).to(device="cuda", dtype=dtype)

    q1, k1, q2, k2 = complete = [
        data(seq, 1, h, latent + rope).float().requires_grad_() for h in (heads, 1, heads, 1)
    ]
    projection = data(heads, 4, latent).float().requires_grad_()
    shards = [t.detach()[pos].to(dtype).requires_grad_() for t in complete]
    up = projection.detach().to(dtype).requires_grad_()
    x = torch.zeros(len(pos), 1, 2, device="cuda", dtype=dtype)
    inputs = dict(
        value=None,
        attention_mask=None,
        x=x,
        qr=x,
        up_v_weight=up,
        attn_mask_type=AttnMaskType.causal,
        packed_seq_params=packed_params,
    )
    compact_calls = []
    gather_window = dsa.gather_zigzag_window_kv

    def record_window(key, width, group):
        compact_calls.append(width)
        return gather_window(key, width, group)

    monkeypatch.setattr(dsa, "gather_zigzag_window_kv", record_window)
    saved = {}
    first = CrossLoopPass(0, saved)
    with first.activate():
        out1 = attn(shards[0], shards[1], **inputs)
    first.finish_source()
    second = CrossLoopPass(1, saved)
    with second.activate():
        out2 = attn(shards[2], shards[3], **inputs)
        saved_ids = second.read(attn, "dsa_topk")
        lengths = second.read(attn, "dsa_topk_length")

    assert compact_calls == ([window] if not packed and window <= chunk + 1 else [])
    causal = all_pos[:, None] >= all_pos
    causal &= document[:, None] == document
    starts = document * (seq // 2)
    sparse = causal & ((all_pos - starts) < k)
    local = causal & (all_pos[:, None] - all_pos < window)

    def dense(query, keys, support):
        scores = torch.einsum("sbhd,tbd->sbht", query, keys[:, :, 0]) * scale
        prob = scores.masked_fill(~support[:, None, None], -torch.inf).softmax(-1)
        context = torch.einsum("sbht,tbd->sbhd", prob, keys[:, :, 0, :latent])
        return torch.einsum("sbhd,hvd->sbhv", context, projection).flatten(2)

    ref1 = dense(q1, k1, sparse)
    ref2 = dense(q2, torch.cat((k1, k2)), torch.cat((sparse, local), 1))
    if packed:
        ref2 = ref2.masked_fill(~real[:, None, None], 0)
    adjoints = [data(*t.shape) * real[:, None, None] for t in (ref1, ref2)]
    grads = torch.autograd.grad((out1, out2), [*shards, up], [g[pos] for g in adjoints])
    # up is replicated; the training wrapper sums this weight gradient over CP.
    projection_grad = grads[-1].contiguous()
    dist.all_reduce(projection_grad, group=cp_group)
    ref_grads = torch.autograd.grad(
        (ref1, ref2), [*complete, projection], [g.float() for g in adjoints]
    )
    assert calls == [1]
    torch.testing.assert_close(
        lengths.long(), ((pos - starts[pos] + 1).clamp_max(k) * real[pos])[None, :]
    )
    columns = torch.arange(k, device="cuda")
    valid = columns < lengths[..., None]
    observed_ids = saved_ids.long().masked_fill(~valid, seq).sort(-1).values
    expected_ids = (columns + starts[pos, None]).expand_as(observed_ids).masked_fill(~valid, seq)
    torch.testing.assert_close(observed_ids, expected_ids)
    for name, actual, expected in zip(
        ("round1", "round2", "dq1", "dk1", "dq2", "dk2", "projection gradient"),
        (out1, out2, *grads[:-1], projection_grad),
        (ref1[pos], ref2[pos], *(g[pos] for g in ref_grads[:-1]), ref_grads[-1]),
    ):
        if fused:
            assert (actual.float() - expected).norm() <= 0.04 * expected.norm(), name
        else:
            torch.testing.assert_close(
                actual, expected, atol=4e-5, rtol=4e-5, msg=lambda msg: f"{name}: {msg}"
            )
