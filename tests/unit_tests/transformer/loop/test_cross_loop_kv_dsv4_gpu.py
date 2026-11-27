# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise real BF16 DSv4 projections, compressors and learned selection in a fixed R4 loop."""

import os
from collections import Counter, defaultdict
from contextlib import ExitStack

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa import CompressedSparseAttention
from megatron.core.transformer.forward_replay import in_forward_replay
from megatron.core.transformer.loop.cross_loop_kv import current_cross_loop_pass
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA, VOCAB, model_parallel_once
from tests.unit_tests.transformer.loop.test_loop_attn_res_dsv4_gpu import _model

pytestmark = pytest.mark.skipif(
    not CUDA or int(os.environ.get("WORLD_SIZE", 1)) != 1, reason="single-GPU training smoke"
)


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


@pytest.mark.parametrize("checkpoint", ["eager", "round", "block"])
def test_r4_hybrid_language_model_shares_sparse_bank_and_keeps_four_windows(
    monkeypatch, checkpoint
):
    _check_model(monkeypatch, checkpoint, "injection", 4)


@pytest.mark.parametrize("checkpoint", ["eager", "round", "block"])
def test_gated_r2_shares_sparse_bank_with_fresh_windows(monkeypatch, checkpoint):
    _check_model(monkeypatch, checkpoint, "gated_residual", 2)


def _check_model(monkeypatch, checkpoint, carrier, rounds):
    from megatron.core.transformer.experimental_attention_variant import dsa

    with ExitStack() as hooks:
        model = _model(
            carrier,
            torch.bfloat16,
            loop_iterations=rounds,
            loop_recompute=checkpoint != "eager",
            loop_recompute_block_stride=2 if checkpoint == "block" else None,
            loop_recompute_edges=checkpoint == "block",
            loop_reuse_sparse_kv=True,
            dsa_indexer_loss_coeff=1e-2,
            dsa_indexer_topk=8,
            gradient_accumulation_fusion=False,
            deterministic_mode=False,
        )
        roles = model.decoder.loop_roles
        core = model.decoder.layers[roles.core_start : roles.core_end]
        attentions = [m for m in core.modules() if isinstance(m, CompressedSparseAttention)]
        assert sorted(attn.compress_ratio for attn in attentions) == [4, 128]

        local_kv, sparse_kv = defaultdict(list), defaultdict(list)
        indexer_calls = Counter()
        replayed_windows = Counter()
        real_select = dsa.fused_qk_topk_naive
        topk_calls = []

        def count_topk(q, k, *args, **kwargs):
            # This runs during ordinary forward and any checkpoint replay that reselects.
            assert current_cross_loop_pass().is_source, "a consumer reran learned top-k"
            if not in_forward_replay():
                topk_calls.append((q.size(0), k.size(0)))
            return real_select(q, k, *args, **kwargs)

        monkeypatch.setattr(dsa, "fused_qk_topk_naive", count_topk)

        def record_local(module, args):
            is_source = current_cross_loop_pass().is_source
            if in_forward_replay():
                replayed_windows[is_source] += 1
                return
            key = args[1]
            key.retain_grad()
            local_kv[module].append((is_source, key))

        def record_sparse(module, args, output):
            assert current_cross_loop_pass().is_source, "a consumer recompressed sparse KV"
            if in_forward_replay():
                return
            output.retain_grad()
            sparse_kv[module].append(output)

        def record_indexer(module, args):
            assert current_cross_loop_pass().is_source, "a consumer ran the indexer projection"
            if not in_forward_replay():
                indexer_calls[module] += 1

        for attn in attentions:
            hooks.callback(attn.register_forward_pre_hook(record_local).remove)
            hooks.callback(attn.compressor.register_forward_hook(record_sparse).remove)
            if attn.indexer is not None:
                query_proj = attn.indexer.linear_wq_b
                hooks.callback(query_proj.register_forward_pre_hook(record_indexer).remove)

        # At S=256 the HCA bank has two entries and the CSA indexer selects 8 of 64.
        tokens = torch.randint(VOCAB, (1, 257), device="cuda")
        position_ids = torch.arange(256, device="cuda").unsqueeze(0)
        loss = model(
            tokens[:, :-1].contiguous(), position_ids, None, labels=tokens[:, 1:].contiguous()
        )
        assert torch.isfinite(loss).all()
        # The pattern has exactly one learned-indexer layer and it is in the repeated core.
        assert topk_calls == [(256, 64)]
        for attn in attentions:
            assert [source for source, _ in local_kv[attn]] == [True] + [False] * (rounds - 1)
            assert len(sparse_kv[attn.compressor]) == 1
            if attn.indexer is not None:
                assert indexer_calls[attn.indexer.linear_wq_b] == 1
        loss.float().mean().backward()
        if checkpoint != "eager":
            assert replayed_windows[False] > 0, "the test did not exercise a consumer replay"

        def assert_trained(tensor):
            grad = tensor.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.count_nonzero() > 0

        for attn in attentions:
            windows = [key for _, key in local_kv[attn]]
            assert len({kv.data_ptr() for kv in windows}) == rounds
            for kv in windows:
                assert_trained(kv)
            (compressed,) = sparse_kv[attn.compressor]
            assert_trained(compressed)
            assert_trained(attn.compressor.linear_wkv.weight)
            if attn.indexer is not None:
                # This real indexer weight receives gradients only through its auxiliary loss.
                assert_trained(attn.indexer.linear_wq_b.weight)

        if carrier == "gated_residual":
            assert len(model.decoder.loop_gates) == rounds
            for gate in model.decoder.loop_gates:
                assert_trained(gate.proj.weight)
                assert_trained(gate.norm.weight)
