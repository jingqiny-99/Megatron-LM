# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Static layout and model ownership tests; mocked collectives are not PP qualification."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer import attention_residual_forward_state as state
from megatron.core.transformer.attention_residual import attn_res_num_sources


def _config(**overrides):
    values = dict(
        num_layers=16,
        hidden_size=4,
        attn_res_block_layers=3,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        enable_attention_residuals=True,
        attn_res_impl='fla',
        attn_res_forward_projection=True,
        attn_res_forward_projection_fraction=1.0,
        layernorm_epsilon=1e-6,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_source_completion_precedes_value_registration():
    plan = state.ForwardProjectionPlan.from_config(_config())
    assert plan.completed_sources(0) == (0,)
    assert plan.completed_sources(12) == (0, 1, 2, 3, 4)
    assert attn_res_num_sources(12, 3) == 4
    assert attn_res_num_sources(13, 3) == 5
    assert state.ForwardProjectionRow(4, None) in plan.entry_rows(3)
    assert plan.source_completion_layer(4) == 12
    assert plan.source_completion_layer(0) == 0
    assert plan.entry_rows(0) == plan.exit_rows(3) == ()


@pytest.mark.parametrize('layers,block', [(16, 3), (12, 3), (8, 8), (8, 16)])
def test_trailing_partial_never_projected(layers, block):
    plan = state.ForwardProjectionPlan.from_config(
        _config(num_layers=layers, attn_res_block_layers=block)
    )
    sources = plan.completed_sources(layers)
    assert all(plan.source_completion_layer(sid) < layers for sid in sources)
    with pytest.raises(ValueError, match='trailing partial'):
        plan.source_completion_layer(sources[-1] + 1)


@pytest.mark.parametrize('fraction', [0.0, 0.01, 0.5, 1.0])
def test_latest_fraction_and_exact_full_prefix_rows(fraction):
    import math

    plan = state.ForwardProjectionPlan.from_config(
        _config(attn_res_forward_projection_fraction=fraction)
    )
    for source in plan.completed_sources(16):
        eligible = [cid for cid in range(33) if plan.consumer_layer(cid) > source * 3]
        count = math.ceil(len(eligible) * fraction)
        assert plan.selected_consumers(source) == tuple(eligible[-count:] if count else [])
    for stage, offset in ((1, 4), (2, 8), (3, 12)):
        rows = plan.entry_rows(stage)
        assert plan.exit_rows(stage - 1) == rows
        expected = []
        for source in range(offset // 3 + 1):
            consumers = [cid for cid in plan.selected_consumers(source) if cid // 2 + 1 > offset]
            if consumers:
                expected.append(state.ForwardProjectionRow(source, None))
                expected.extend(state.ForwardProjectionRow(source, cid) for cid in consumers)
        assert rows == tuple(expected)
        assert len(set(rows)) == len(rows)
    assert plan.padded_rows == max(1, *(len(plan.entry_rows(s)) for s in (1, 2, 3)))
    if fraction == 1:
        assert [len(plan.entry_rows(s)) for s in (1, 2, 3)] == [52, 54, 50]
    if fraction == 0:
        assert plan.payload_shape(128, 2) == (128, 2, 1)


@pytest.mark.parametrize('vp', [None, 1, 2])
def test_uniform_consumer_owner_and_no_interleaved_score_cache(vp):
    plan = state.ForwardProjectionPlan.from_config(_config(virtual_pipeline_model_parallel_size=vp))
    assert plan.num_consumers == 33
    assert plan.consumer_layer(32) == 17
    assert plan.consumer_owner(32) == 1
    for cid in range(32):
        expected = ((cid // 2) // (16 // plan.num_stages)) % 2
        assert plan.consumer_owner(cid) == expected
    if vp == 2:
        # Rank 0's second virtual chunk receives the complete statistic prefix.
        assert {row.source_id for row in plan.entry_rows(2)} == {0, 1, 2}


@pytest.mark.parametrize(
    'overrides',
    [
        {'pipeline_model_parallel_size': 1},
        {'virtual_pipeline_model_parallel_size': 3},
        {'num_layers': 15},
        {'attn_res_block_layers': 0},
        {'attn_res_forward_projection_fraction': float('nan')},
        {'attn_res_forward_projection_fraction': -0.1},
        {'attn_res_forward_projection_fraction': 1.1},
        {'layernorm_epsilon': 0},
        {'tensor_model_parallel_size': 2},
        {'context_parallel_size': 2},
        {'expert_model_parallel_size': 2},
        {'num_moe_experts': 2},
        {'is_hybrid_model': True},
        {'mtp_num_layers': 1},
        {'cpu_offloading': True},
        {'num_layers_in_first_pipeline_stage': 3},
        {'attn_res_impl': 'compile'},
    ],
)
def test_unsupported_plan_rejected(overrides):
    with pytest.raises(ValueError):
        state.ForwardProjectionPlan.from_config(_config(**overrides))


@pytest.mark.parametrize('bad_id', [-1, 33, True, 1.0])
def test_invalid_consumer_identity_rejected(bad_id):
    plan = state.ForwardProjectionPlan.from_config(_config())
    with pytest.raises(ValueError):
        plan.consumer_layer(bad_id)


@pytest.mark.parametrize('fraction', [0.0, 0.5, 1.0])
def test_pack_unpack_owning_layout_and_padding(fraction):
    plan = state.ForwardProjectionPlan.from_config(
        _config(attn_res_forward_projection_fraction=fraction)
    )
    for stage in (1, 2, 3):
        rows = {
            key: torch.full((5, 2), index + 0.5) for index, key in enumerate(plan.entry_rows(stage))
        }
        payload = state.pack_forward_rows(plan, stage, rows, 5, 2, torch.device('cpu'))
        unpacked = state.unpack_forward_rows(plan, stage, payload, 5, 2)
        assert list(unpacked) == list(rows)
        assert payload.shape == plan.payload_shape(5, 2)
        assert not payload.requires_grad
        for key, tensor in unpacked.items():
            torch.testing.assert_close(tensor, rows[key], rtol=0, atol=0)
            assert tensor.is_contiguous()
            assert tensor.data_ptr() != rows[key].data_ptr()
        assert torch.count_nonzero(payload.view(plan.padded_rows, 5, 2)[len(rows) :]) == 0
        for value in rows.values():
            value.add_(100)
        for key, tensor in unpacked.items():
            assert not torch.equal(tensor, rows[key])


@pytest.mark.parametrize('invalid', ['missing', 'dtype', 'shape', 'grad', 'strided'])
def test_invalid_auxiliary_rows_rejected(invalid):
    plan = state.ForwardProjectionPlan.from_config(_config())
    keys = plan.entry_rows(1)
    rows = {key: torch.ones(5, 2) for key in keys}
    if invalid == 'missing':
        rows.pop(keys[0])
    elif invalid == 'dtype':
        rows[keys[0]] = rows[keys[0]].bfloat16()
    elif invalid == 'shape':
        rows[keys[0]] = torch.ones(2, 5)
    elif invalid == 'grad':
        rows[keys[0]].requires_grad_()
    else:
        rows[keys[0]] = torch.ones(5, 4)[:, ::2]
    with pytest.raises(ValueError):
        state.pack_forward_rows(plan, 1, rows, 5, 2, torch.device('cpu'))


class _Consumer(torch.nn.Module):
    def __init__(self, cid, config):
        super().__init__()
        self.forward_projection_consumer_id = cid
        self.eps = config.layernorm_epsilon
        self.pseudo_query = torch.nn.Parameter(torch.full((config.hidden_size,), cid + 0.125))
        self.key_norm_weight = torch.nn.Parameter(torch.full((config.hidden_size,), cid + 1.25))


def _model(config, rank):
    plan = state.ForwardProjectionPlan.from_config(config)
    chunks = []
    for vp in range(plan.vp_size):
        chunk = torch.nn.Module()
        chunk.config = config
        stage = vp * 2 + rank
        consumers = range(2 * plan.layer_offset(stage), 2 * plan.layer_offset(stage + 1))
        if stage == plan.num_stages - 1:
            consumers = (*consumers, 2 * plan.num_layers)
        chunk.consumers = torch.nn.ModuleList([_Consumer(cid, config) for cid in consumers])
        chunks.append(chunk)
    return chunks


def _mock_world(monkeypatch, config=None, rank=0, mutate_registry=None):
    config = config or _config()
    plan = state.ForwardProjectionPlan.from_config(config)
    models = [_model(config, owner) for owner in range(2)]
    modules = [[module for chunk in model for module in chunk.consumers] for model in models]
    group = SimpleNamespace(size=lambda: 2, rank=lambda: rank)
    calls = {'gather': 0, 'sum': 0, 'ready': []}

    def gather(destination, local, group=None):
        calls['gather'] += 1
        for owner in range(2):
            destination[owner] = dict(
                plan=plan,
                records=[
                    (module.forward_projection_consumer_id, owner, index, module.eps)
                    for index, module in enumerate(modules[owner])
                ],
                errors=[],
            )
        destination[rank] = local
        if mutate_registry:
            mutate_registry(destination)

    def reduce(bank, op=None, group=None):
        assert calls['ready']
        assert op is torch.distributed.ReduceOp.SUM
        attr = 'pseudo_query' if calls['sum'] % 2 == 0 else 'key_norm_weight'
        calls['sum'] += 1
        for module in modules[1 - rank]:
            bank[module.forward_projection_consumer_id].copy_(getattr(module, attr))

    original_ensure = state.ensure_params_ready

    def ready(params):
        calls['ready'].append(tuple(id(param) for param in params))
        original_ensure(params)

    monkeypatch.setattr(torch.distributed, 'all_gather_object', gather)
    monkeypatch.setattr(torch.distributed, 'all_reduce', reduce)
    monkeypatch.setattr(state, 'ensure_params_ready', ready)
    return models, modules, group, calls


@pytest.mark.parametrize('rank', [0, 1])
def test_fresh_raw_snapshots_and_no_gradient_ownership_changes(monkeypatch, rank):
    models, modules, group, calls = _mock_world(monkeypatch, rank=rank)
    canonical = modules[rank][0].pseudo_query
    sentinel_grad = torch.full_like(canonical, 7)
    canonical.grad = sentinel_grad
    first = state.prepare_attn_res_forward_projection(models[rank], group)
    first.validate(1, first.plan)
    first.validate_module(modules[rank][0].forward_projection_consumer_id, modules[rank][0])
    for owner in modules:
        for module in owner:
            cid = module.forward_projection_consumer_id
            torch.testing.assert_close(first.query_bank[cid], module.pseudo_query, rtol=0, atol=0)
            torch.testing.assert_close(
                first.gamma_bank[cid], module.key_norm_weight, rtol=0, atol=0
            )
    assert first.query_bank.grad_fn is first.gamma_bank.grad_fn is None
    assert not first.query_bank.requires_grad and not first.gamma_bank.requires_grad
    previous = first.query_bank.clone()
    with torch.no_grad():
        canonical.add_(0.25)
    with pytest.raises(RuntimeError, match='Canonical parameters changed'):
        first.validate(first.epoch, first.plan)
    second = state.prepare_attn_res_forward_projection(models[rank], group)
    assert second.epoch == 2
    assert first.query_bank.data_ptr() != second.query_bank.data_ptr()
    torch.testing.assert_close(first.query_bank, previous, rtol=0, atol=0)
    assert calls['gather'] == 1 and calls['sum'] == 4
    assert canonical.grad is sentinel_grad
    assert not hasattr(canonical, '_externally_managed_grad')
    assert not hasattr(models[rank][0].config, '_attn_res_forward_projection_runtime')
    with pytest.raises(RuntimeError, match='Stale'):
        first.validate(first.epoch, first.plan)
    with pytest.raises(RuntimeError, match='monotonically'):
        state.prepare_attn_res_forward_projection(models[rank], group, epoch=2)


def test_parameter_aliases_keep_distinct_consumer_rows(monkeypatch):
    models, modules, group, calls = _mock_world(monkeypatch)
    first, second = modules[0][:2]
    second.pseudo_query = first.pseudo_query
    second.key_norm_weight = first.key_norm_weight
    snapshot = state.prepare_attn_res_forward_projection(models[0], group)
    assert len(calls['ready'][0]) == len(set(calls['ready'][0]))
    assert (
        snapshot.query_bank[first.forward_projection_consumer_id].data_ptr()
        != snapshot.query_bank[second.forward_projection_consumer_id].data_ptr()
    )
    torch.testing.assert_close(snapshot.query_bank[0], snapshot.query_bank[1], rtol=0, atol=0)


def test_parameter_ready_callback_precedes_snapshot_copy(monkeypatch):
    models, modules, group, _ = _mock_world(monkeypatch)
    module = modules[0][0]
    callback_calls = []

    def ready():
        callback_calls.append(True)
        with torch.no_grad():
            module.pseudo_query.fill_(7)
            module.key_norm_weight.fill_(11)

    module.pseudo_query._ensure_param_ready_callback = ready
    module.key_norm_weight._ensure_param_ready_callback = ready
    snapshot = state.prepare_attn_res_forward_projection(models[0], group)
    assert callback_calls == [True]
    assert torch.equal(snapshot.query_bank[0], torch.full((4,), 7.0))
    assert torch.equal(snapshot.gamma_bank[0], torch.full((4,), 11.0))


def test_shared_config_does_not_share_runtime(monkeypatch):
    models, _, group, calls = _mock_world(monkeypatch)
    first = state.prepare_attn_res_forward_projection(models[0], group)
    fresh_model = _model(models[0][0].config, 0)
    second = state.prepare_attn_res_forward_projection(fresh_model, group)
    assert first.epoch == second.epoch == 1
    assert state.get_attn_res_forward_runtime(models[0]) is not state.get_attn_res_forward_runtime(
        fresh_model
    )
    first.validate(1, first.plan)
    assert calls['gather'] == 2
    with pytest.raises(RuntimeError, match='does not belong'):
        first.validate_module(0, fresh_model[0].consumers[0])


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'owner', 'eps', 'plan'])
def test_global_registry_rejects_invalid_ownership(monkeypatch, mutation):
    def mutate(gathered):
        records = gathered[1]['records']
        if mutation == 'duplicate':
            records.append(records[0])
        elif mutation == 'missing':
            records.pop()
        elif mutation == 'owner':
            cid, _, index, eps = records[0]
            records[0] = (cid, 0, index, eps)
        elif mutation == 'eps':
            cid, owner, index, _ = records[0]
            records[0] = (cid, owner, index, 1e-5)
        else:
            gathered[1]['plan'] = replace(gathered[1]['plan'], fraction=0.5)

    models, _, group, calls = _mock_world(monkeypatch, mutate_registry=mutate)
    with pytest.raises(ValueError):
        state.prepare_attn_res_forward_projection(models[0], group)
    assert calls['gather'] == 1 and calls['sum'] == 0


@pytest.mark.parametrize('mutation', ['query', 'module', 'bank', 'config'])
def test_host_validation_rejects_stale_surfaces(monkeypatch, mutation):
    models, modules, group, _ = _mock_world(monkeypatch)
    snapshot = state.prepare_attn_res_forward_projection(models[0], group)
    if mutation == 'query':
        modules[0][0].pseudo_query = torch.nn.Parameter(modules[0][0].pseudo_query.detach().clone())
    elif mutation == 'module':
        models[0][0].consumers[0] = _Consumer(0, models[0][0].config)
    elif mutation == 'bank':
        snapshot.query_bank.add_(1)
    else:
        models[0][0].config.attn_res_forward_projection_fraction = 0.5
    with pytest.raises(RuntimeError):
        snapshot.validate(snapshot.epoch, snapshot.plan)


def test_disabled_path_has_no_registry_or_group_access(monkeypatch):
    model = torch.nn.Module()
    model.config = SimpleNamespace(attn_res_forward_projection=False)

    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled path touched projection machinery')

    monkeypatch.setattr(torch.distributed, 'all_gather_object', forbidden)
    monkeypatch.setattr(torch.distributed, 'all_reduce', forbidden)
    monkeypatch.setattr(state, 'ensure_params_ready', forbidden)
    assert state.prepare_attn_res_forward_projection(model, None) is None
    assert state.get_attn_res_forward_runtime(model) is None


def test_explicit_group_and_data_parallel_scope(monkeypatch):
    models, _, group, _ = _mock_world(monkeypatch)
    with pytest.raises(ValueError, match='explicit two-rank'):
        state.prepare_attn_res_forward_projection(models[0], None)
    with pytest.raises(ValueError, match='TP=CP=EP=DP=1'):
        state.prepare_attn_res_forward_projection(models[0], group, data_parallel_size=2)
