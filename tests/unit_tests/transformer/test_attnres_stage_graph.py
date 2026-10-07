# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Scope and stage-surface contracts; evolving Adam parity lives in the GPU harness."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


def stage_config(**overrides):
    kwargs = dict(
        num_layers=4,
        hidden_size=32,
        num_attention_heads=4,
        pipeline_model_parallel_size=2,
        enable_attention_residuals=True,
        attn_res_block_layers=3,
        attn_res_impl="fla",
        cuda_graph_impl="local",
        attn_res_stage_cuda_graph=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        gradient_accumulation_fusion=False,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_explicit_stage_mode_is_legal():
    config = stage_config()
    assert config.attn_res_stage_cuda_graph
    assert config.cuda_graph_modules == []


@pytest.mark.parametrize(
    "override",
    [
        {"virtual_pipeline_model_parallel_size": 2},
        {"pipeline_model_parallel_size": 1},
        {"tensor_model_parallel_size": 2},
        {"context_parallel_size": 2},
        {"bf16": False, "params_dtype": torch.float32},
        {"attn_res_impl": "eager"},
        {"hidden_dropout": 0.1},
        {"attention_dropout": 0.1},
        {"fine_grained_activation_offloading": True},
        {"recompute_granularity": "selective"},
        {"gradient_accumulation_fusion": True},
        {"cuda_graph_impl": "none"},
        {"cuda_graph_modules": ["attn"]},
    ],
)
def test_reject_unsupported_stage_modes(override):
    with pytest.raises((ValueError, AssertionError)):
        stage_config(**override)


def test_attnres_local_graph_still_requires_opt_in():
    with pytest.raises(ValueError, match="cuda_graph_impl"):
        stage_config(attn_res_stage_cuda_graph=False)


def test_stage_mode_requires_attnres():
    with pytest.raises(ValueError, match="requires enable_attention_residuals"):
        stage_config(enable_attention_residuals=False, attn_res_block_layers=None)


def surface_block(pre_process=False):
    # No distributed/CUDA constructor is needed to test the graph surface adapter.
    block = TransformerBlock.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.config = SimpleNamespace(
        attn_res_stage_cuda_graph=True, attn_res_vpp_final_chunk_cuda_graph=False
    )
    block.pre_process = pre_process
    block.input_tensor = torch.randn(12, 1, 4, requires_grad=True)
    return block


def test_nonfirst_stage_binds_static_capture_payload_and_restores_live_input():
    block = surface_block()
    live = block.input_tensor
    static = torch.randn_like(live, requires_grad=True)

    def decoder_body(hidden_states, attention_mask, rotary_pos_emb):
        assert block.input_tensor is static
        assert hidden_states is static
        return block.input_tensor * 2

    block.forward = decoder_body
    output = block._forward_attn_res_stage_graph(static, None, None)
    assert block.input_tensor is live
    output.sum().backward()
    torch.testing.assert_close(static.grad, torch.full_like(static, 2))
    assert live.grad is None


def test_nonfirst_stage_restores_live_input_when_body_raises():
    block = surface_block()
    live = block.input_tensor

    def failed_body(**kwargs):
        raise RuntimeError("capture error")

    block.forward = failed_body
    with pytest.raises(RuntimeError, match="capture error"):
        block._forward_attn_res_stage_graph(torch.ones_like(live), None, None)
    assert block.input_tensor is live


def test_first_stage_source_owns_storage_after_input_surface_is_replaced():
    """A direct b_0 reference must not follow the runner's input .data mutation."""
    block = surface_block(pre_process=True)
    surface = torch.randn(6, 1, 4, requires_grad=True)
    original = surface.detach().clone()
    source = []

    def decoder_body(hidden_states, **kwargs):
        source.append(hidden_states)
        return hidden_states.square().sum()

    block.forward = decoder_body
    output = block._forward_attn_res_stage_graph(surface, None, None)
    assert source[0].data_ptr() != surface.data_ptr()
    # The native graph runner mutates .data to weaken storage ownership. Replacing
    # it here detects the same forbidden dependence on the input TensorImpl.
    surface.data = torch.zeros_like(surface)
    torch.testing.assert_close(source[0], original, rtol=0, atol=0)
    output.backward()
    torch.testing.assert_close(surface.grad, 2 * original, rtol=0, atol=0)


@pytest.mark.parametrize("pre_process", [False, True])
def test_stage_surface_selects_correct_differentiable_input(pre_process):
    block = surface_block(pre_process)
    embedding = torch.randn_like(block.input_tensor, requires_grad=True)
    payload = embedding if pre_process else block.input_tensor

    def manager(module, args, kwargs):
        assert module is block and args == ()
        assert kwargs["hidden_states"] is payload
        # Simulate a recording node's aliased output. The schedule needs a viewless handle.
        return payload.view_as(payload)

    block.cudagraph_manager = manager
    output = block(embedding, None)
    assert output._base is None
    output.sum().backward()
    torch.testing.assert_close(payload.grad, torch.ones_like(payload))
    if not pre_process:
        assert embedding.grad is None


def test_stage_surface_rejects_unimplemented_inference_and_packed_inputs():
    block = surface_block()
    with torch.no_grad(), pytest.raises(ValueError, match="training with gradients"):
        block(None, None)
    with pytest.raises(ValueError, match="packed_seq_params"):
        block(None, None, packed_seq_params=object())
