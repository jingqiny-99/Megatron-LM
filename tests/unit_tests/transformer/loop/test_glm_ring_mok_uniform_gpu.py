# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Full uniform recompute groups on a tied, two-pass PP2/EP4 DSA and MoK ring."""

from collections import Counter
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant import dsa, dsa_cudnn_kernels
from megatron.core.transformer.forward_replay import in_forward_replay, routing_tape_active
from megatron.core.transformer.loop.glm_dsa import GLMDSALoopStack
from tests.unit_tests.transformer.loop import glm_ring_mok_utils as moe
from tests.unit_tests.transformer.loop import test_glm_ring_mok_recompute_gpu as checkpoint
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA


@pytest.mark.skipif(not CUDA or checkpoint.ring.WORLD != 8, reason="requires eight CUDA ranks")
def test_full_uniform_four_sublayers_matches_one_with_trained_indexer(monkeypatch):
    """Compare real block boundaries as well as losses, routes, shared KV and all weight grads."""
    strides = iter((1, 4))
    ranges = Counter()
    indexer_grads = Counter()
    selector_calls = Counter()
    run_range = GLMDSALoopStack._run_range
    run = checkpoint._run

    def models(plans, *, layernorm=False):
        size = next(strides)
        # The actual fused DSA path records a source selection and score payload for its
        # checkpoint replay. A four-sublayer group spans D/E/D/E on the shared core.
        config = replace(
            moe.config(2),
            num_attention_heads=64,
            num_query_groups=64,
            q_lora_rank=512,
            kv_lora_rank=512,
            qk_head_dim=128,
            qk_pos_emb_head_dim=64,
            v_head_dim=128,
            kv_channels=128,
            attention_backend=AttnBackend.auto,
            dsa_kernel_backend="cudnn",
            dsa_indexer_head_dim=128,
            dsa_indexer_n_heads=32,
            dsa_indexer_topk=64,
            dsa_indexer_reuse_checkpoint_topk=True,
            dsa_indexer_weights_proj_use_quantization=False,
            dsa_indexer_weights_proj_output_dtype="bf16",
            linear_cp_mode=None,
            linear_cp_layout=None,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=size,
            loop_recompute=True,
            loop_recompute_edges=True,
            loop_recompute_block_stride=4,
            loop_ring_recompute_plan="/".join(plans),
        )
        with patch.object(moe, "config", return_value=config):
            return moe.models(2)

    def record_ranges(stack, x, start, end, ctx, label):
        if label.startswith("loop_hybrid/mb="):
            assert routing_tape_active()
            ranges[stack.config.recompute_num_layers, label, start, end, in_forward_replay()] += 1
        return run_range(stack, x, start, end, ctx, label)

    def record_gradients(models, *args, **kwargs):
        size = models[0].config.recompute_num_layers
        result = run(models, *args, **kwargs)
        for name, grad in result.grads.items():
            if ".indexer." in name and grad.norm() > 0:
                indexer_grads[size] += 1
        return result

    # Count selection itself (layer projections also run to train the indexer during replay).
    dense = dsa_cudnn_kernels._indexer_topk_bshd

    def record_selection(*args, **kwargs):
        scope = dsa.current_cross_loop_pass()
        selector_calls[scope.is_source if scope is not None else None, in_forward_replay()] += 1
        return dense(*args, **kwargs)

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", record_selection)
    # Trunk checkpoints mark their own non-reentrant replays. The MTP /E also uses
    # reentrant tensor-parallel checkpoints with full recompute. Its nested original
    # calls during autograd backward belong to the same sample as the first MTP call.
    def observing_backward():
        return in_forward_replay() or torch._C._current_graph_task_id() >= 0

    monkeypatch.setattr(checkpoint, "in_forward_replay", observing_backward)
    monkeypatch.setattr(checkpoint, "_models", models)
    monkeypatch.setattr(checkpoint, "_run", record_gradients)
    monkeypatch.setattr(GLMDSALoopStack, "_run_range", record_ranges)
    plans = ("CCCC", "CCCC")
    checkpoint._compare_plans(plans, plans)
    assert indexer_grads[1] == indexer_grads[4] > 0
    assert selector_calls[True, False] > 0
    assert not any(
        count for (source, replay), count in selector_calls.items() if source is False or replay
    )

    # The same logical visits ran forward and backward exactly once per local sublayer.
    visits = {}
    for size in (1, 4):
        covered = Counter()
        for (stride, label, first, end, replay), count in ranges.items():
            if stride == size:
                assert first % size == 0 and 0 < end - first <= size and count == 1
                for sublayer in range(first, end):
                    covered[label, sublayer, replay] += count
        assert covered
        visits[size] = covered
    assert visits[1] == visits[4]
    assert all(
        count == 1 and visits[4][label, layer, not replay] == 1
        for (label, layer, replay), count in visits[4].items()
    )
    assert any(stride == 4 and end - first >= 2 for stride, label, first, end, replay in ranges)
    print(
        f"verified {len(visits[4])} original/replay sublayer visits, "
        f"{indexer_grads[4]} nonzero indexer gradients and {sum(selector_calls.values())} selections"
    )
