# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Forward-only metadata at real schedule helpers and loss/backward boundaries."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.pipeline_parallel import schedules
from megatron.core.pipeline_parallel.p2p_communication import PipelineTensorSpec
from megatron.core.transformer import attention_residual, attention_residual_forward_state


def _config(**overrides):
    values = dict(
        num_layers=16,
        hidden_size=8,
        attn_res_block_layers=3,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=None,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        enable_attention_residuals=True,
        attn_res_impl='fla',
        attn_res_forward_projection=True,
        attn_res_forward_projection_fraction=1.0,
        layernorm_epsilon=1e-6,
        pipeline_dtype=torch.bfloat16,
        variable_seq_lengths=False,
        sequence_parallel=False,
        pipeline_model_parallel_layout=None,
        num_layers_in_first_pipeline_stage=None,
        num_layers_in_last_pipeline_stage=None,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
        timers=None,
        enable_autocast=False,
        calculate_per_token_loss=False,
        grad_scale_func=None,
        deallocate_pipeline_outputs=False,
        num_moe_experts=None,
        mtp_num_layers=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('enabled', [False, True])
def test_noninterleaved_peer_shapes_match_and_preserve_local_token_count(enabled):
    config = _config(attn_res_forward_projection=enabled)
    unit = SimpleNamespace(size=lambda: 1)
    results = []
    for rank, receive in ((0, False), (1, True)):
        results.append(
            schedules.get_tensor_shapes(
                seq_length=128,
                micro_batch_size=2,
                decoder_seq_length=None,
                config=config,
                tp_group=unit,
                cp_group=unit,
                pp_group=SimpleNamespace(size=lambda: 2, rank=lambda: rank),
                is_recv=receive,
            )
        )
    assert results[0] == results[1]
    if enabled:
        value, auxiliary = results[0]
        assert value == PipelineTensorSpec((4 * 128, 2, 8), torch.bfloat16)
        # Auxiliary rows multiply local S, not the four-slice value payload length.
        assert auxiliary == PipelineTensorSpec((54 * 128, 2, 1), torch.float32, False)
    else:
        assert results[0] == [(4 * 128, 2, 8)]


@pytest.mark.parametrize('fraction,rows', [(0, 1), (0.5, 45), (1, 54)])
def test_vpp_uniform_typed_shape(fraction, rows):
    config = _config(
        virtual_pipeline_model_parallel_size=2, attn_res_forward_projection_fraction=fraction
    )
    specs = schedules._attn_res_forward_tensor_specs(config, (3 * 128, 2, 8), 128, 2)
    assert specs[0].shape == (3 * 128, 2, 8)
    assert specs[1].shape == (rows * 128, 2, 1)
    assert specs[0].requires_grad and not specs[1].requires_grad


class _Model(torch.nn.Module):
    def set_input_tensor(self, inputs):
        self.received = inputs


@pytest.mark.parametrize('inputs', [None, [None, None], 'received'])
def test_nonfinal_forward_preserves_flat_channels(inputs):
    model = _Model()
    if inputs == 'received':
        inputs = [torch.ones(2, 1, 8, requires_grad=True), torch.ones(3, 1, 1)]
    payload = [torch.ones(2, 1, 8, requires_grad=True), torch.ones(3, 1, 1)]
    result, tokens = schedules.forward_step(
        lambda iterator, chunk: (payload, None),
        None,
        model,
        8,
        inputs,
        [],
        _config(),
        cp_group_size=1,
        is_last_stage=False,
    )
    assert result is payload and len(result) == 2
    assert int(tokens) == 0
    assert model.received == ([None] if inputs is None else inputs)


def test_final_stage_keeps_loss_normalization_and_none_input_gradient():
    model = _Model()
    value = torch.ones(2, 1, 8, requires_grad=True)
    metadata = torch.ones(3, 1, 1)
    losses = []
    config = _config()
    output, _ = schedules.forward_step(
        lambda iterator, chunk: (value * 2, lambda tensor: (tensor.sum(), {'loss': 32.0})),
        None,
        model,
        8,
        [value, metadata],
        losses,
        config,
        cp_group_size=1,
        is_last_stage=True,
    )
    assert len(output) == 1 and output[0].item() == 4.0
    grads = schedules.backward_step([value, metadata], output, None, config)
    torch.testing.assert_close(grads[0], torch.full_like(value, 0.25), rtol=0, atol=0)
    assert grads[1] is None and metadata.grad is None
    assert losses == [{'loss': 32.0}]


@pytest.mark.parametrize('deallocate', [False, True])
def test_nonfinal_backward_and_pseudo_deallocation_keep_auxiliary_absent(deallocate):
    value = torch.ones(2, 1, 8, requires_grad=True)
    received_aux = torch.ones(3, 1, 1)
    output = [value * 3, received_aux.clone()]
    gradient = torch.full_like(value, 2)
    schedules.deallocate_output_tensor(output, deallocate)
    grads = schedules.backward_step(
        [value, received_aux],
        output,
        [gradient, None],
        _config(deallocate_pipeline_outputs=deallocate),
    )
    torch.testing.assert_close(grads[0], torch.full_like(value, 6), rtol=0, atol=0)
    assert grads[1] is None
    assert received_aux.shape == (3, 1, 1)
    if deallocate:
        assert output[0].numel() == output[1].numel() == 1


def test_nonzero_auxiliary_gradient_rejected_before_backward():
    value = torch.ones(2, 1, 8, requires_grad=True)
    output = [value * 3, torch.ones(3, 1, 1)]
    with pytest.raises(ValueError, match='structural None'):
        schedules.backward_step(
            [value, output[1]], output, [torch.ones_like(value), output[1]], _config()
        )
    assert value.grad is None


def test_snapshot_entry_prepares_once_and_sets_each_wrapped_model(monkeypatch):
    events = []
    chunks = [_Model(), _Model()]
    config = _config(virtual_pipeline_model_parallel_size=2)
    pp = object()
    snapshot = object()
    groups = SimpleNamespace(pp=pp, dp_cp=SimpleNamespace(size=lambda: 1))

    def prepare(model, group, *, data_parallel_size):
        assert model is chunks and group is pp and data_parallel_size == 1
        events.append('prepare')
        return snapshot

    def setter(model, attr):
        assert attr == 'set_attn_res_forward_snapshot'

        def set_snapshot(value):
            assert value is snapshot
            events.append(chunks.index(model))

        return set_snapshot

    monkeypatch.setattr(
        attention_residual, 'attn_res_source_cache_reset', lambda: events.append('reset')
    )
    monkeypatch.setattr(
        attention_residual_forward_state, 'prepare_attn_res_forward_projection', prepare
    )
    monkeypatch.setattr(schedules, 'get_attr_wrapped_model', setter)
    assert schedules._prepare_attn_res_forward_step(chunks, config, groups) is snapshot
    assert events == ['reset', 'prepare', 0, 1]


def test_disabled_schedule_hook_does_not_touch_model_groups_or_cache(monkeypatch):
    def forbidden():
        raise AssertionError('disabled projection path touched source cache')

    monkeypatch.setattr(attention_residual, 'attn_res_source_cache_reset', forbidden)
    assert (
        schedules._prepare_attn_res_forward_step(
            None, _config(attn_res_forward_projection=False), None
        )
        is None
    )


def test_variable_sequence_mode_rejected_before_preparation():
    with pytest.raises(ValueError, match='static pipeline'):
        schedules._prepare_attn_res_forward_step(None, _config(variable_seq_lengths=True), None)
