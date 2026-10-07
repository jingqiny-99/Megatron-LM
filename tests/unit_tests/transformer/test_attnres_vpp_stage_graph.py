# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Final-local-chunk graph source/cache and deferred-gradient ownership contracts."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.attention_residual import (
    AttnResStageSources,
    attn_res_tap_source,
    get_attn_res_source_cache,
)
from megatron.core.transformer.transformer_block import (
    TransformerBlock,
    _AttnResGraphInputGradientOwner,
)
from tests.unit_tests.transformer.test_attnres_stage_graph import stage_config


def vpp_config(**overrides):
    kwargs = dict(
        num_layers=8,
        virtual_pipeline_model_parallel_size=2,
        attn_res_stage_cuda_graph=False,
        attn_res_vpp_final_chunk_cuda_graph=True,
    )
    kwargs.update(overrides)
    return stage_config(**kwargs)


def test_explicit_final_vp_mode_is_legal():
    config = vpp_config()
    assert config.attn_res_vpp_final_chunk_cuda_graph
    assert not config.attn_res_stage_cuda_graph


@pytest.mark.parametrize(
    "override",
    [
        {"virtual_pipeline_model_parallel_size": None},
        {"virtual_pipeline_model_parallel_size": 4},
        {"pipeline_model_parallel_size": 4},
        {"attn_res_stage_cuda_graph": True},
        {"hidden_dropout": 0.1},
        {"recompute_granularity": "selective"},
        {"fine_grained_activation_offloading": True},
        {"cuda_graph_impl": "none"},
    ],
)
def test_final_vp_mode_rejects_unsupported_scope(override):
    with pytest.raises((ValueError, AssertionError)):
        vpp_config(**override)


def test_pure_captured_state_does_not_mutate_real_cache():
    config = vpp_config()
    cache = get_attn_res_source_cache().sources
    source = torch.randn(2, 1, 4, requires_grad=True)
    assert 37 not in cache
    cache[37] = [source]
    try:
        captured = AttnResStageSources(
            config, pp_rank=0, vp_stage=1, microbatch_id=37, pre_process=False, manage_cache=False
        )
        captured._update_cache()
        assert cache[37][0] is source
        host = AttnResStageSources(
            config, pp_rank=0, vp_stage=1, microbatch_id=37, pre_process=False
        )
        host._update_cache()
        assert 37 not in cache
    finally:
        cache.pop(37, None)


def test_earlier_chunks_cannot_disable_original_cache_lifecycle():
    with pytest.raises(ValueError, match="restricted to final-VP"):
        AttnResStageSources(
            vpp_config(),
            pp_rank=0,
            vp_stage=0,
            microbatch_id=0,
            pre_process=True,
            manage_cache=False,
        )


def test_graph_gradient_copy_survives_recycled_static_gradient_storage():
    static_gradient = torch.randn(3, 1, 8, dtype=torch.bfloat16)
    expected = static_gradient.clone()
    owned = _AttnResGraphInputGradientOwner.backward(None, static_gradient)
    assert owned.data_ptr() != static_gradient.data_ptr()
    static_gradient.zero_()
    torch.testing.assert_close(owned, expected, rtol=0, atol=0)


def test_original_cache_leaf_drains_once_after_final_chunk_gradient_copy():
    source = torch.randn(3, 1, 8, dtype=torch.bfloat16, requires_grad=True)
    producer_edge, cache_leaf = attn_res_tap_source(source)
    final_chunk_input = _AttnResGraphInputGradientOwner.apply(cache_leaf)
    # The final chunk finishes before the original producing chunk's backward.
    (final_chunk_input * 2).sum().backward()
    torch.testing.assert_close(cache_leaf.grad, torch.full_like(source, 2), rtol=0, atol=0)
    assert source.grad is None
    (producer_edge * 3).sum().backward()
    assert cache_leaf.grad is None
    torch.testing.assert_close(source.grad, torch.full_like(source, 5), rtol=0, atol=0)


def test_final_chunk_body_owns_every_saved_source_and_partial():
    block = TransformerBlock.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.config = vpp_config()
    block.vp_stage = 1
    block.pg_collection = SimpleNamespace(pp=None)
    block._attn_res_graph_entry_source_count = 2
    inputs = [torch.randn(3, 1, 8, requires_grad=True) for _ in range(3)]
    original = [tensor.detach().clone() for tensor in inputs]
    captured = []

    def body(hidden_states, _attn_res_stage_state, **kwargs):
        assert not _attn_res_stage_state.manage_cache
        values = [hidden_states, *_attn_res_stage_state.graph_sources]
        captured.extend(values)
        return sum(value.square().sum() for value in values)

    block.forward = body
    output = block._forward_attn_res_final_vp_graph(
        inputs[0], None, None, 5, source_0=inputs[1], source_1=inputs[2]
    )
    for surface, saved, expected in zip(inputs, captured, original):
        assert saved.data_ptr() != surface.data_ptr()
        surface.data = torch.zeros_like(surface)
        torch.testing.assert_close(saved, expected, rtol=0, atol=0)
    output.backward()
    for surface, expected in zip(inputs, original):
        torch.testing.assert_close(surface.grad, 2 * expected, rtol=0, atol=0)
