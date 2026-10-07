# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Source export, gradient order and ownership contracts for full VPP stage graphs."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.attention_residual import (
    AttnResStageSources,
    _AttnResGraphSourceTap,
    attn_res_tap_source,
    get_attn_res_source_cache,
)
from megatron.core.transformer.transformer_block import (
    TransformerBlock,
    _AttnResGraphInputGradientOwner,
    _AttnResGraphOutputBridge,
)
from tests.unit_tests.transformer.test_attnres_vpp_stage_graph import vpp_config


def full_vpp_config(**overrides):
    kwargs = dict(attn_res_vpp_final_chunk_cuda_graph=False, attn_res_vpp_cuda_graph=True)
    kwargs.update(overrides)
    return vpp_config(**kwargs)


def test_full_vpp_is_explicit_and_mutually_exclusive():
    config = full_vpp_config()
    assert config.attn_res_vpp_cuda_graph
    assert not config.attn_res_stage_cuda_graph
    assert not config.attn_res_vpp_final_chunk_cuda_graph


@pytest.mark.parametrize(
    "override",
    [
        {"attn_res_vpp_final_chunk_cuda_graph": True},
        {"attn_res_stage_cuda_graph": True},
        {"virtual_pipeline_model_parallel_size": None},
        {"virtual_pipeline_model_parallel_size": 4},
        {"pipeline_model_parallel_size": 4},
        {"cuda_graph_impl": "none"},
        {"hidden_dropout": 0.1},
        {"fine_grained_activation_offloading": True},
    ],
)
def test_full_vpp_rejects_unsupported_scope(override):
    with pytest.raises((ValueError, AssertionError)):
        full_vpp_config(**override)


def test_exported_source_matches_original_bf16_addition_order():
    original = torch.ones(4, dtype=torch.bfloat16, requires_grad=True)
    captured = original.detach().clone().requires_grad_(True)
    local, leaf = attn_res_tap_source(original)
    graph_local, graph_export = _AttnResGraphSourceTap.apply(captured)
    # BF16 (512 + -512) + 1 is 1, but 512 + (-512 + 1) is 0. External fan-in
    # must occur after all local consumer contributions have accumulated.
    leaf.backward(torch.ones_like(leaf))
    torch.autograd.backward(
        (local * 512, local * -512), (torch.ones_like(local), torch.ones_like(local))
    )
    torch.autograd.backward(
        (graph_local * 512, graph_local * -512, graph_export),
        (torch.ones_like(local), torch.ones_like(local), torch.ones_like(local)),
    )
    torch.testing.assert_close(captured.grad, original.grad, rtol=0, atol=0)
    torch.testing.assert_close(captured.grad, torch.ones_like(captured), rtol=0, atol=0)
    assert leaf.grad is None


@pytest.mark.parametrize("omit_external", [False, True])
def test_source_tap_rejects_missing_gradient(omit_external):
    source = torch.ones(3, dtype=torch.bfloat16, requires_grad=True)
    local, exported = _AttnResGraphSourceTap.apply(source)
    with pytest.raises(RuntimeError, match="requires local and external gradients"):
        (local if omit_external else exported).sum().backward()


def bridge_context(leaves):
    return SimpleNamespace(
        cache_leaves=tuple(leaves),
        source_ids=tuple(range(len(leaves))),
        microbatch_id=7,
        vp_stage=0,
        drained=False,
    )


def test_bridge_drains_once_and_returns_original_owned_gradients_in_order():
    leaves = [torch.ones(3, dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
    for index, leaf in enumerate(leaves):
        leaf.grad = torch.full_like(leaf, index + 2)
    gradients = [leaf.grad for leaf in leaves]
    ctx = bridge_context(leaves)
    payload_gradient = torch.ones(3, dtype=torch.bfloat16)
    result = _AttnResGraphOutputBridge.backward(ctx, payload_gradient)
    assert result[:4] == (None, None, None, None)
    assert result[4] is payload_gradient
    assert all(actual is expected for actual, expected in zip(result[5:], gradients))
    assert all(leaf.grad is None for leaf in leaves)
    assert ctx.drained
    with pytest.raises(RuntimeError, match="already drained"):
        _AttnResGraphOutputBridge.backward(ctx, payload_gradient)


def test_missing_external_gradient_does_not_partially_drain_other_leaves():
    leaves = [torch.ones(3, requires_grad=True) for _ in range(2)]
    leaves[0].grad = torch.full_like(leaves[0], 2)
    retained = leaves[0].grad
    ctx = bridge_context(leaves)
    with pytest.raises(RuntimeError, match="requires every later-chunk source gradient"):
        _AttnResGraphOutputBridge.backward(ctx, torch.ones(3))
    assert leaves[0].grad is retained
    assert not ctx.drained


def test_bridge_autograd_keeps_source_order_and_original_fanin_after_cache_eviction():
    sources = [torch.randn(2, 1, 4, dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
    edges = [_AttnResGraphSourceTap.apply(source) for source in sources]
    payload = edges[0][0] * 3 + edges[1][0] * 5
    exports = tuple(edge[1] for edge in edges)
    leaves = tuple(export.detach().clone().requires_grad_(True) for export in exports)
    output = _AttnResGraphOutputBridge.apply(leaves, (0, 1), 11, 0, payload, *exports)
    cache = get_attn_res_source_cache().sources
    assert 11 not in cache
    cache[11] = list(leaves)
    try:
        later = [_AttnResGraphInputGradientOwner.apply(leaf) for leaf in cache[11]]
        cache.pop(11)
        # Later local chunk backward precedes the producing chunk backward.
        (later[0] * 7 + later[1] * 11).sum().backward()
        output.sum().backward()
        assert all(leaf.grad is None for leaf in leaves)
        for source, expected in zip(sources, (10, 16)):
            torch.testing.assert_close(
                source.grad, torch.full_like(source, expected), rtol=0, atol=0
            )
    finally:
        cache.pop(11, None)


@pytest.mark.parametrize("vp_stage", [0, 1])
def test_full_vpp_pure_state_has_no_cache_side_effects(vp_stage):
    cache = get_attn_res_source_cache().sources
    source = torch.randn(2, 1, 4, requires_grad=True)
    assert 37 not in cache
    cache[37] = [source]
    try:
        state = AttnResStageSources(
            full_vpp_config(),
            pp_rank=0,
            vp_stage=vp_stage,
            microbatch_id=37,
            pre_process=vp_stage == 0,
            manage_cache=False,
        )
        state.append_block_start(source)
        state._update_cache()
        assert cache[37][0] is source
        assert len(state.graph_source_exports) == (1 if vp_stage == 0 else 0)
    finally:
        cache.pop(37, None)


def test_earlier_body_owns_inputs_and_exports_original_source_order():
    block = TransformerBlock.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.config = full_vpp_config()
    block.pre_process = True
    block.vp_stage = 0
    block.pg_collection = SimpleNamespace(pp=None)
    block._attn_res_graph_entry_source_count = 0
    block._attn_res_graph_export_source_ids = (0, 1)
    source = torch.randn(2, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    original = source.detach().clone()
    saved = []

    def body(hidden_states, _attn_res_stage_state, **kwargs):
        state = _attn_res_stage_state
        saved.append(hidden_states)
        state.append_block_start(hidden_states)
        state.append_block_start(hidden_states * 2)
        return state.graph_sources[0] * 3 + state.graph_sources[1] * 5

    block.forward = body
    payload, first, second = block._forward_attn_res_earlier_vp_graph(source, None, None, 5)
    assert saved[0].data_ptr() != source.data_ptr()
    source.data = torch.zeros_like(source)
    torch.testing.assert_close(first, original, rtol=0, atol=0)
    torch.testing.assert_close(second, original * 2, rtol=0, atol=0)
    torch.autograd.backward(
        (payload, first, second), tuple(torch.ones_like(source) for _ in range(3))
    )
    torch.testing.assert_close(source.grad, torch.full_like(source, 16), rtol=0, atol=0)


def test_host_publishes_owning_exports_and_returns_viewless_payload():
    block = TransformerBlock.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.config = full_vpp_config()
    block.pre_process = True
    block.vp_stage = 0
    block.pg_collection = SimpleNamespace(pp=None)
    block.layers = [SimpleNamespace(layer_number=1, current_microbatch=29)]
    block._attn_res_graph_entry_source_count = 0
    block._attn_res_graph_export_source_ids = (0, 1)
    block.preprocess_for_layer_schedule = lambda tensor: tensor
    source = torch.randn(2, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    exports = [torch.randn_like(source, requires_grad=True) for _ in range(2)]
    original = [value.detach().clone() for value in exports]

    def manager(module, args, kwargs):
        assert kwargs["microbatch_id"] == 29
        assert set(kwargs) == {"hidden_states", "attention_mask", "rotary_pos_emb", "microbatch_id"}
        return (kwargs["hidden_states"] * 2, *exports)

    block.cudagraph_manager = manager
    cache = get_attn_res_source_cache().sources
    assert 29 not in cache
    try:
        output = block._call_attn_res_final_vp_graph(source)
        assert output._base is None
        leaves = cache.pop(29)
        assert len(leaves) == 2
        for leaf, exported, expected in zip(leaves, exports, original):
            assert leaf.is_leaf and leaf.requires_grad
            assert leaf.data_ptr() != exported.data_ptr()
            exported.data.zero_()
            torch.testing.assert_close(leaf, expected, rtol=0, atol=0)
        (leaves[0] * 3 + leaves[1] * 7).sum().backward()
        output.sum().backward()
        assert all(leaf.grad is None for leaf in leaves)
        for exported, expected in zip(exports, (3, 7)):
            torch.testing.assert_close(
                exported.grad, torch.full_like(exported, expected), rtol=0, atol=0
            )
    finally:
        cache.pop(29, None)
