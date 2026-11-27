# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exercise the native token-level GLM loop, its grouped selectors and cross-pass gradients."""

import os
from collections import Counter
from contextlib import ExitStack

import pytest
import torch

from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.forward_replay import in_forward_replay, statistics_round
from megatron.core.transformer.loop.cross_loop_kv import current_cross_loop_pass
from megatron.core.transformer.loop.glm_dsa import loop_hybrid_glm_dsa_stack_spec
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA, VOCAB, model_parallel_once

pytestmark = pytest.mark.skipif(
    not CUDA or int(os.environ.get("WORLD_SIZE", "1")) != 1, reason="single-GPU native DSA model"
)

# 5 prelude, 5 shared, 3 coda. The stride of 4 sublayers = 2 decoder blocks
# starts the second core checkpoint at layer 8, inside layer 6's selector group.
PATTERN = "D-" * 13
SEQUENCE = 48
CORE = range(6, 11)
OWNERS = [1, 2, 3, 3, 3, 6, 6, 6, 6, 10, 11, 11, 11]


@pytest.fixture(scope="module", autouse=True)
def _parallel():
    with model_parallel_once():
        yield


def _config(
    reuse,
    checkpoint,
    cp=1,
    edge_checkpoint_stride=None,
    *,
    num_layers=len(PATTERN),
    loop_iterations=2,
    prelude=10,
    coda=6,
    **loop_options
):
    return MLATransformerConfig(
        num_layers=num_layers,
        context_parallel_size=cp,
        cp_comm_type="allgather",
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=256,
        normalization="RMSNorm",
        hidden_dropout=0,
        attention_dropout=0,
        add_bias_linear=False,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        multi_latent_attention=True,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_head_dim=16,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        qk_layernorm=True,
        rope_type="rope",
        apply_rope_fusion=False,
        attention_backend=AttnBackend.unfused,
        experimental_attention_variant="dsa",
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=32,
        dsa_indexer_topk=8,
        dsa_indexer_topk_freq=4,
        dsa_indexer_skip_topk_offset=3,
        dsa_indexer_loss_coeff=0.01,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_rotate_activation=False,  # FP32 reference without the BF16 Hadamard kernel
        calculate_per_token_loss=True,
        loop_architecture="huginn",
        loop_carrier="gated_residual",
        loop_iterations=loop_iterations,
        loop_prelude_sublayers=prelude,
        loop_coda_sublayers=coda,
        loop_reuse_sparse_kv=reuse,
        loop_sparse_kv_window_size=4,
        loop_recompute=checkpoint,
        loop_recompute_block_stride=4 if checkpoint else edge_checkpoint_stride,
        loop_recompute_edges=checkpoint or edge_checkpoint_stride is not None,
        **loop_options,
    )


def _model(
    reuse,
    checkpoint,
    cp=1,
    edge_checkpoint_stride=None,
    mtp_pattern=None,
    *,
    sequence_length=SEQUENCE,
    **loop_options,
):
    cfg = _config(reuse, checkpoint, cp, edge_checkpoint_stride, **loop_options)
    return HybridModel(
        config=cfg,
        hybrid_stack_spec=loop_hybrid_glm_dsa_stack_spec(cfg),
        vocab_size=VOCAB,
        max_sequence_length=sequence_length,
        hybrid_layer_pattern=PATTERN + ("/" + mtp_pattern if mtp_pattern else ""),
        position_embedding_type="rope",
    ).cuda()


def _nonzero(grad):
    assert grad is not None and torch.isfinite(grad).all() and grad.count_nonzero() > 0


def _run(model, rows, reuse, monkeypatch):
    # Counts address the actual selector kernel (including the auxiliary-loss path).
    # The read of the first bank is checked at the attention kernel's joint softmax input.
    attentions = [layer.self_attention.core_attention for layer in model.decoder.layers[::2]]
    assert [a.source_layer for a in attentions] == OWNERS
    assert [a.indexer is not None for a in attentions] == [
        n == owner for n, owner in enumerate(OWNERS, 1)
    ]
    assert len(model.decoder.loop_gates) == 2
    assert model.decoder.loop_gates[0].proj.weight is not model.decoder.loop_gates[1].proj.weight
    current_layer = 0
    microbatch = 0
    selections, calls = Counter(), Counter()
    projected, joined, first_indices = {}, {}, {}
    replayed = Counter()
    select = dsa.fused_qk_topk_naive
    attend = dsa._run_sparse_attention

    def before(module, args):
        nonlocal current_layer
        current_layer = module.layer_number
        if in_forward_replay():
            return
        round_id = statistics_round()
        round_id = round_id[0] if round_id else 0
        calls[microbatch, round_id, current_layer] += 1
        if current_layer in CORE:
            raw = args[1]  # absorbed key, S x B x 1 x (latent + RoPE)
            raw.retain_grad()
            projected[microbatch, round_id, current_layer] = raw

    def select_topk(*args, **kwargs):
        visit = current_cross_loop_pass()
        assert visit is None or visit.is_source, "a consumer pass reran top-k"
        if not in_forward_replay():
            round_id = statistics_round()
            selections[microbatch, round_id[0] if round_id else 0, current_layer] += 1
        return select(*args, **kwargs)

    def observe_attention(**kwargs):
        visit = current_cross_loop_pass()
        if in_forward_replay():
            if visit is not None:
                replayed[visit.is_source] += 1
            return attend(**kwargs)
        round_id = statistics_round()
        round_id = round_id[0] if round_id else 0
        indices = kwargs["topk_indices"]
        if visit is None or visit.is_source:
            first_indices[microbatch, round_id, current_layer] = indices.detach()
            owner = OWNERS[current_layer - 1]
            torch.testing.assert_close(indices, first_indices[microbatch, round_id, owner])
        else:
            dual = kwargs["key"]
            assert dual.size(0) == 2 * SEQUENCE
            # An observer must not register extra saved tensors in a checkpoint's
            # forward. The model's dual-bank tensor below retains its actual gradient.
            with torch.no_grad():
                for half, r in zip(dual.chunk(2, dim=0), (1, 2)):
                    torch.testing.assert_close(half, projected[microbatch, r, current_layer])
            dual.retain_grad()
            joined[microbatch, current_layer] = dual
        return attend(**kwargs)

    with ExitStack() as hooks, monkeypatch.context() as patches:
        for attention in attentions:
            hooks.callback(attention.register_forward_pre_hook(before).remove)
        patches.setattr(dsa, "fused_qk_topk_naive", select_topk)
        patches.setattr(dsa, "_run_sparse_attention", observe_attention)
        positions = torch.arange(SEQUENCE, device="cuda")[None, :]
        losses = []
        for microbatch, row in enumerate(rows):
            losses.append(model(row[:, :-1], positions, None, labels=row[:, 1:].contiguous()))
        expected_select = Counter()
        expected_calls = Counter()
        for mb in range(len(rows)):
            for n, owner in enumerate(OWNERS, 1):
                for r in ((1, 2) if n in CORE else (0,)):
                    expected_calls[mb, r, n] = 1
                    if n == owner and (not reuse or r != 2):
                        expected_select[mb, r, n] = 1
        assert calls == expected_calls
        assert selections == expected_select
        # Both first-pass graphs are alive together; replay in the opposite order.
        for loss in reversed(losses):
            assert torch.isfinite(loss).all()
            loss.mean().backward()

    for latent in projected.values():
        _nonzero(latent.grad)
    assert len(joined) == (len(rows) * len(CORE) if reuse else 0)
    for dual in joined.values():
        prefix_grad, local_grad = dual.grad.chunk(2, dim=0)
        _nonzero(prefix_grad)  # isolates R2's contribution to the R1 latent
        _nonzero(local_grad)
    for gate in model.decoder.loop_gates:
        _nonzero(gate.proj.weight.grad)
        _nonzero(gate.norm.weight.grad)
    for a in attentions:
        if a.indexer is not None:
            _nonzero(a.indexer.linear_wq_b.weight.grad)
    if model.config.loop_recompute and reuse:
        assert replayed[True] and replayed[False]
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    return [loss.detach() for loss in losses], grads


@pytest.mark.parametrize("reuse", [False, True])
def test_native_two_pass_training_with_checkpointed_groups(reuse, monkeypatch):
    torch.manual_seed(54)
    reference = _model(reuse, False)
    checkpointed = _model(reuse, True)
    checkpointed.load_state_dict(reference.state_dict())
    rows = [torch.randint(VOCAB, (1, SEQUENCE + 1), device="cuda") for _ in range(2)]
    losses, grads = _run(reference, rows, reuse, monkeypatch)
    other_losses, other_grads = _run(checkpointed, rows, reuse, monkeypatch)
    torch.testing.assert_close(losses, other_losses)
    assert grads.keys() == other_grads.keys()
    for name in grads:
        torch.testing.assert_close(other_grads[name], grads[name], rtol=1e-4, atol=2e-6, msg=name)
