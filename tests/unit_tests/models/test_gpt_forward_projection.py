# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPT host ownership of forward-only PP metadata and bounded feature configuration."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig


class RecordingDecoder(torch.nn.Module):
    """Minimal decoder surface; exercises the real GPT boundary methods."""

    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.calls = []
        self.output = (torch.zeros(2, 1, 4, dtype=torch.bfloat16), torch.zeros(2, 1, 1))

    def set_input_tensor(self, value):
        self.input_tensor = value

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


def model(enabled=True, pre_process=False, post_process=False):
    result = GPTModel.__new__(GPTModel)
    torch.nn.Module.__init__(result)
    result.config = SimpleNamespace(
        attn_res_forward_projection=enabled,
        fine_grained_activation_offloading=False,
        moe_paged_stash=False,
        moe_n_hash_layers=0,
        enable_attention_residuals=True,
    )
    result.pre_process = pre_process
    result.post_process = post_process
    result.mtp_process = False
    result._attn_res_forward_snapshot = None
    result._attn_res_forward_payload = None
    result.decoder = RecordingDecoder()
    result._preprocess = lambda **kwargs: (None, None, None, None, None, None)
    result._postprocess = lambda **kwargs: kwargs["hidden_states"]
    return result


def snapshot():
    calls = []
    result = SimpleNamespace(epoch=3, plan="plan")
    result.validate = lambda **kwargs: calls.append(kwargs)
    return result, calls


def projection_config(vp=1, **overrides):
    kwargs = dict(
        num_layers=16,
        hidden_size=1024,
        num_attention_heads=8,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2 if vp == 2 else None,
        enable_attention_residuals=True,
        attn_res_block_layers=3,
        attn_res_impl="fla",
        cuda_graph_impl="local",
        attn_res_stage_cuda_graph=vp == 1,
        attn_res_vpp_cuda_graph=vp == 2,
        attn_res_forward_projection=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        gradient_accumulation_fusion=False,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@pytest.mark.parametrize("vp", [1, 2])
@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_dense_bf16_projection_config(vp, fraction):
    config = projection_config(vp, attn_res_forward_projection_fraction=fraction)
    assert config.attn_res_forward_projection
    assert config.attn_res_forward_projection_fraction == fraction


@pytest.mark.parametrize(
    "override",
    [
        {"attn_res_stage_cuda_graph": False, "cuda_graph_impl": "none"},
        {"bf16": False, "params_dtype": torch.float32},
        {"pipeline_dtype": torch.float32},
        {"pipeline_model_parallel_size": 4},
        {"context_parallel_size": 2},
        {"tensor_model_parallel_size": 2},
        {"hidden_dropout": 0.1},
        {"attention_dropout": 0.1},
        {"variable_seq_lengths": True},
        {"recompute_granularity": "selective"},
        {"gradient_accumulation_fusion": True},
        {"attn_res_impl": "eager"},
        {"fine_grained_activation_offloading": True},
        {"enable_attention_residuals": False, "attn_res_block_layers": None},
        {"attn_res_forward_projection_fraction": -0.01},
        {"attn_res_forward_projection_fraction": 1.01},
        {"attn_res_forward_projection_fraction": float("nan")},
    ],
)
def test_reject_unsupported_projection_configs(override):
    with pytest.raises((ValueError, AssertionError)):
        projection_config(**override)


def test_final_chunk_only_graph_is_not_supported():
    with pytest.raises((ValueError, AssertionError)):
        projection_config(
            vp=2, attn_res_vpp_cuda_graph=False, attn_res_vpp_final_chunk_cuda_graph=True
        )


@pytest.mark.parametrize("input_value", [None, [None], [None, None]])
def test_first_stage_clears_old_metadata(input_value):
    instance = model(pre_process=True)
    instance._attn_res_forward_payload = torch.ones(1)
    instance.decoder.input_tensor = torch.ones(1)
    instance.set_input_tensor(input_value)
    assert instance._attn_res_forward_payload is None
    assert instance.decoder.input_tensor is None


def test_first_stage_rejects_an_incoming_payload():
    instance = model(pre_process=True)
    with pytest.raises(ValueError, match="first projection stage"):
        instance.set_input_tensor([torch.zeros(1), torch.zeros(1)])


def test_nonfirst_stage_replaces_value_and_metadata_together():
    instance = model()
    first = torch.zeros(2, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    metadata = torch.zeros(2, 1, 1)
    instance.set_input_tensor([first, metadata])
    assert instance.decoder.input_tensor is first
    assert instance._attn_res_forward_payload is metadata
    replacement = torch.ones_like(metadata)
    instance.set_input_tensor([first, replacement])
    assert instance._attn_res_forward_payload is replacement
    with pytest.raises(ValueError, match="value and metadata"):
        instance.set_input_tensor(first)
    assert instance._attn_res_forward_payload is None


@pytest.mark.parametrize(
    "bad", ["value_dtype", "metadata_dtype", "metadata_grad", "metadata_device", "none"]
)
def test_reject_malformed_metadata_before_decoder_binding(bad):
    instance = model()
    value = torch.zeros(2, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    metadata = torch.zeros(2, 1, 1)
    if bad == "value_dtype":
        value = value.float()
    elif bad == "metadata_dtype":
        metadata = metadata.bfloat16()
    elif bad == "metadata_grad":
        metadata.requires_grad_()
    elif bad == "metadata_device":
        metadata = torch.empty(2, 1, 1, device="meta")
    else:
        metadata = None
    with pytest.raises(ValueError, match="BF16 values"):
        instance.set_input_tensor([value, metadata])
    assert instance.decoder.input_tensor is None
    assert instance._attn_res_forward_payload is None


def test_snapshot_setter_rejects_missing_stale_and_disabled():
    instance = model()
    with pytest.raises(ValueError, match="current snapshot"):
        instance.set_attn_res_forward_snapshot(None)
    stale, _ = snapshot()

    def reject(**kwargs):
        raise RuntimeError("stale snapshot")

    stale.validate = reject
    with pytest.raises(RuntimeError, match="stale snapshot"):
        instance.set_attn_res_forward_snapshot(stale)
    assert instance._attn_res_forward_snapshot is None
    with pytest.raises(RuntimeError, match="configured feature"):
        model(enabled=False).set_attn_res_forward_snapshot(stale)


def test_forward_requires_snapshot_before_decoder_execution():
    instance = model()
    with pytest.raises(RuntimeError, match="publish projections"):
        instance(None, None, None)
    assert not instance.decoder.calls


def test_forward_threads_explicit_snapshot_payload_and_preserves_extra_kwargs():
    instance = model()
    current, calls = snapshot()
    instance.set_attn_res_forward_snapshot(current)
    value = torch.zeros(2, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    metadata = torch.zeros(2, 1, 1)
    instance.set_input_tensor([value, metadata])
    extra = {"example": "unchanged"}
    output = instance(None, None, None, extra_block_kwargs=extra)
    assert isinstance(output, list) and len(output) == 2
    assert output[0] is instance.decoder.output[0]
    assert output[1] is instance.decoder.output[1]
    assert calls == [{"epoch": 3, "plan": "plan"}]
    forwarded = instance.decoder.calls[0]
    assert forwarded["forward_projection_snapshot"] is current
    assert forwarded["forward_projection_payload"] is metadata
    assert extra == {"example": "unchanged"}


@pytest.mark.parametrize("key", ["forward_projection_snapshot", "forward_projection_payload"])
def test_caller_cannot_override_schedule_owned_projection_inputs(key):
    instance = model()
    instance.set_attn_res_forward_snapshot(snapshot()[0])
    with pytest.raises(ValueError, match="owned by"):
        instance(None, None, None, extra_block_kwargs={key: object()})
    assert not instance.decoder.calls


def test_nonfinal_decoder_must_return_both_channels():
    instance = model()
    instance.set_attn_res_forward_snapshot(snapshot()[0])
    instance.decoder.output = torch.zeros(2, 1, 4, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="values and metadata"):
        instance(None, None, None)


def test_disabled_gpt_retains_single_tensor_boundary():
    instance = model(enabled=False)
    value = torch.zeros(2, 1, 4)
    instance.set_input_tensor(value)
    assert instance.decoder.input_tensor is value
    with pytest.raises(AssertionError, match="length 1"):
        instance.set_input_tensor([value, value])
    instance.decoder.output = value
    assert instance(None, None, None) is value
    assert "forward_projection_snapshot" not in instance.decoder.calls[0]
    assert "forward_projection_payload" not in instance.decoder.calls[0]


def test_final_stage_retains_usual_postprocess_output():
    instance = model(post_process=True)
    instance.set_attn_res_forward_snapshot(snapshot()[0])
    value = torch.zeros(2, 1, 4, dtype=torch.bfloat16)
    instance.decoder.output = value
    assert instance(None, None, None) is value
