# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Query publication, alias scatter and deferred DDP gradient contracts."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.distributed.param_and_grad_buffer import (
    _ParamAndGradBucketGroup,
    group_params_for_buffers,
    partition_buckets,
)
from megatron.core.transformer.attention_residual_projection_runtime import (
    AttnResProjectionRuntime,
    finalize_attn_res_projection,
    get_attn_res_projection_runtime,
    mark_attn_res_projection_parameters,
    prepare_attn_res_projection,
)
from megatron.core.transformer.attention_residual_source_state import ProjectionSourceState
from megatron.core.utils import PARAM_READY_CALLBACK_ATTR


class _Consumer(torch.nn.Module):
    def __init__(self, consumer_id, q=None, gamma=None, stop_source_grad=False):
        super().__init__()
        self.pseudo_query = q if q is not None else torch.nn.Parameter(torch.randn(4))
        self.key_norm_weight = gamma if gamma is not None else torch.nn.Parameter(torch.randn(4))
        self.projection_consumer_id = consumer_id
        self.projection_stop_source_grad = stop_source_grad
        self.eps = 1e-6
        mark_attn_res_projection_parameters(self)


def _runtime(modules, pp_group=None):
    return AttnResProjectionRuntime(
        modules, SimpleNamespace(hidden_size=4, layernorm_epsilon=1e-6), pp_group
    )


def test_alias_consumers_scatter_matches_independent_autograd():
    torch.manual_seed(52)
    first = _Consumer((0, 0, 1, 0))
    second = _Consumer((1, 1, 1, 0), first.pseudo_query, first.key_norm_weight, True)
    runtime = _runtime([second, first])
    ready = []
    published = []
    for param in first.parameters():
        param.main_grad = torch.zeros_like(param)
        param._external_grad_ready_callback = lambda p=param: ready.append(p)
        setattr(param, PARAM_READY_CALLBACK_ATTR, lambda: published.append(True))

    runtime.prepare()
    assert runtime.column(first) == 0
    assert runtime.column(second) == 1
    assert runtime.consumer_metadata[1]['stop_source_grad']
    assert runtime.bank.requires_grad
    assert runtime.bank._do_not_offload
    assert len(published) == 2

    source = torch.randn(3, 4)  # Detached sources must still train queries.
    loss = (source @ runtime.bank[0]).square().sum() + (source @ runtime.bank[1]).sum()
    loss.backward()
    runtime.finalize()

    q = first.pseudo_query.detach().clone().requires_grad_()
    gamma = first.key_norm_weight.detach().clone().requires_grad_()
    reference = (source @ (q * gamma)).square().sum() + (source @ (q * gamma)).sum()
    reference.backward()
    torch.testing.assert_close(first.pseudo_query.main_grad, q.grad)
    torch.testing.assert_close(first.key_norm_weight.main_grad, gamma.grad)
    assert {id(param) for param in ready} == {id(param) for param in first.parameters()}
    assert len(ready) == 2
    assert first.pseudo_query.grad is None
    assert first.key_norm_weight.grad is None
    assert runtime.bank.grad is None


def test_zero_gradients_are_ready_once_and_bank_refreshes_each_step():
    module = _Consumer((0, 0, 1, 0))
    runtime = _runtime([module])
    ready = []
    for param in module.parameters():
        param.main_grad = torch.zeros_like(param)
        param._external_grad_ready_callback = lambda: ready.append(True)
    runtime.prepare()
    with pytest.raises(RuntimeError, match='previous projection step'):
        runtime.prepare()
    runtime.finalize()  # One-source identity need not touch the query bank.
    assert len(ready) == 2
    assert torch.count_nonzero(module.pseudo_query.main_grad) == 0
    with pytest.raises(RuntimeError, match='already finalized'):
        runtime.finalize()
    with torch.no_grad():
        module.pseudo_query.add_(1)
    runtime.prepare()
    torch.testing.assert_close(runtime.bank[0], module.pseudo_query * module.key_norm_weight)
    runtime.finalize()
    assert len(ready) == 4


def test_source_eligibility_uses_consumer_semantics():
    modules = [
        _Consumer((0, 0, 2, 0)),
        _Consumer((0, 0, 3, 0)),
        _Consumer((0, 0, 3, 1)),
        _Consumer((0, 0, 5, 0)),
        _Consumer((1, 1, 1, 0), stop_source_grad=True),
    ]
    runtime = _runtime(modules)
    assert runtime.columns_for_source(2) == [1, 2, 3, 4]
    assert runtime.columns_for_source(2, 0.5) == [3, 4]
    assert runtime.columns_for_source(2, 0) == []
    assert runtime.columns_for_source(4) == [3, 4]
    with pytest.raises(ValueError, match='globally unique'):
        _runtime([modules[0], _Consumer((0, 0, 2, 0))])


def test_forward_only_does_not_publish_gradients():
    module = _Consumer((0, 0, 1, 0))
    runtime = _runtime([module])
    runtime.prepare(forward_only=True)
    assert not runtime.bank.requires_grad
    runtime.finalize()
    assert all(param.grad is None for param in module.parameters())
    runtime.prepare()  # An evaluation call must not prevent the next training step.
    runtime.finalize()


def test_runtime_rejects_unmanaged_parameters_and_unsupported_wrappers():
    module = _Consumer((0, 0, 1, 0))
    del module.pseudo_query._externally_managed_grad
    with pytest.raises(ValueError, match='before building optimizer/DDP'):
        _runtime([module])
    mark_attn_res_projection_parameters(module)
    module.pseudo_query.main_grad = torch.zeros_like(module.pseudo_query)
    with pytest.raises(ValueError, match='external gradient publication'):
        _runtime([module])


def test_schedule_api_attaches_without_changing_model_state(monkeypatch):
    config = SimpleNamespace(
        hidden_size=4,
        layernorm_epsilon=1e-6,
        enable_attention_residuals=True,
        attn_res_impl='source',
        num_layers=1,
        attn_res_block_layers=1,
        attn_res_source_projection_fraction=1.0,
    )
    models = [torch.nn.Module(), torch.nn.Module()]
    for index, model in enumerate(models):
        model.config = config  # Independent models can reuse a configuration.
        model.consumer = _Consumer((0, 0, 1, 0))
        with torch.no_grad():
            model.consumer.pseudo_query.fill_(index + 1)
            model.consumer.key_norm_weight.fill_(1)
    monkeypatch.setattr(
        'megatron.core.transformer.attention_residual_projection_runtime._projection_modules',
        lambda chunks: [chunk.consumer for chunk in chunks],
    )
    keys = tuple(models[0].state_dict())
    first = prepare_attn_res_projection(models[0], None)
    second = prepare_attn_res_projection([models[1]], None)
    assert first is not second
    assert get_attn_res_projection_runtime(models[0].consumer) is first
    assert get_attn_res_projection_runtime(config) is second
    assert tuple(models[0].state_dict()) == keys
    finalize_attn_res_projection(models[0], None)
    finalize_attn_res_projection(models[1], None)
    assert first.finalized and second.finalized
    assert prepare_attn_res_projection(models[0], None) is first
    assert get_attn_res_projection_runtime(config) is first

    # The source path reads the config while the consumer reads its module.
    # Exercise actual source projection after alternating model schedules: a
    # stale config binding would use the finalized second model's query bank
    # and silently send the score gradients to that bank as well.
    state = SimpleNamespace(config=config, interleaved=False, pre_process=True)
    source_state = ProjectionSourceState(state, layers_before=0)
    value = torch.tensor([[1.0, -2.0, 3.0, -4.0]], requires_grad=True)
    projected = source_state.enter(value, score_payload=None)
    scores = projected._attn_res_source_projection.scores
    normalized = value.detach() * torch.rsqrt(value.detach().square().mean(-1, keepdim=True) + 1e-6)
    query = models[0].consumer.pseudo_query.detach()
    torch.testing.assert_close(scores, (normalized @ query).unsqueeze(-1))
    scores.sum().backward()
    finalize_attn_res_projection(models[0], None)
    torch.testing.assert_close(models[0].consumer.pseudo_query.grad, normalized.sum(0))
    assert second.bank.grad is None
    assert torch.count_nonzero(models[1].consumer.pseudo_query.grad) == 0


def test_pp_registry_and_gradient_sum(monkeypatch):
    module = _Consumer((0, 0, 1, 0))
    remote = {
        'id': (0, 0, 2, 0),
        'stop_source_grad': False,
        'owner': 1,
        'local_index': 0,
        'hidden_size': 4,
        'eps': 1e-6,
    }

    def gather(records, local, group):
        records[:] = [local, [remote]]

    reductions = []

    def reduce(tensor, group):
        reductions.append(tensor.clone())
        if len(reductions) == 1:
            tensor[1].fill_(2)  # The other rank owns the second query.
        else:
            tensor[0].add_(3)  # The other rank projected our query at its source.

    monkeypatch.setattr(torch.distributed, 'all_gather_object', gather)
    monkeypatch.setattr(torch.distributed, 'all_reduce', reduce)
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    runtime = _runtime([module], group)
    runtime.prepare()
    assert torch.equal(runtime.bank[1], torch.full((4,), 2.0))
    runtime.bank[0].sum().backward()
    runtime.finalize()
    torch.testing.assert_close(module.pseudo_query.grad, 4 * module.key_norm_weight)
    torch.testing.assert_close(module.key_norm_weight.grad, 4 * module.pseudo_query)
    assert len(reductions) == 2


def test_external_parameters_have_separate_buffer_identity():
    ordinary = torch.nn.Parameter(torch.ones(4))
    external = torch.nn.Parameter(torch.ones(4))
    external._externally_managed_grad = True
    groups = group_params_for_buffers([ordinary, external], grad_reduce_in_fp32=True)
    assert len(groups) == 2
    assert {key.is_external_grad_managed for key in groups} == {False, True}
    # Same-dtype checkpoint indices retain their original global ordering.
    assert [indices for _, indices in groups.values()] == [[0], [1]]


@pytest.mark.parametrize('layer_wise', [False, True])
def test_optimizer_layouts_agree_with_ddp_buffer_identity(layer_wise):
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

    ordinary = torch.nn.Parameter(torch.ones(4, 4))
    external = torch.nn.Parameter(torch.ones(4))
    external._externally_managed_grad = True
    ordinary.is_managed_by_layer_wise_optimizer = layer_wise
    params = [ordinary, external]
    config = DistributedDataParallelConfig(use_distributed_optimizer=True, grad_reduce_in_fp32=True)
    optimizer = LayerWiseDistributedOptimizer if layer_wise else DistributedOptimizer
    layout = optimizer.compute_full_param_layout(params, None, 2, config)
    groups = group_params_for_buffers(params, grad_reduce_in_fp32=True)
    assert layout.layouts.keys() == groups.keys()
    for key, (group_params, indices) in groups.items():
        assert set(layout.layouts[key].param_index_map) == set(group_params)
        assert layout.layouts[key].param_indices == indices
    assert len(layout.layouts) == 2


@pytest.mark.parametrize('force_single', [False, True])
def test_external_groups_remain_separate_from_vpp_and_fp8_merges(monkeypatch, force_single):
    ordinary = torch.nn.Parameter(torch.ones(4))
    external = torch.nn.Parameter(torch.ones(4))
    external._externally_managed_grad = True
    config = SimpleNamespace(use_distributed_optimizer=False)
    group = object()

    def buffer(param, dtype):
        bucket = SimpleNamespace(params_list=[param])
        return SimpleNamespace(
            params=[param],
            buckets=[bucket],
            ddp_config=config,
            param_dtype=dtype,
            data_parallel_group=group,
            data_parallel_world_size=1,
        )

    # Test grouping itself without allocating CUDA communication buffers.
    monkeypatch.setattr(
        'megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup',
        lambda buckets, *_args: buckets,
    )
    ordinary_buffer = buffer(ordinary, torch.uint8)
    external_buffer = buffer(external, torch.float32)
    groups = partition_buckets(
        [ordinary_buffer, external_buffer], force_single_bucket_group=force_single
    )
    assert groups == [ordinary_buffer.buckets, external_buffer.buckets]


def _deferred_group(*, overlap=True, first_batch=True):
    group = object.__new__(_ParamAndGradBucketGroup)
    params = [torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
    group.params = set(params)
    group.externally_managed_params = set(params)
    group.pending_external_grads = set(params)
    group.param_to_bucket = dict.fromkeys(params)
    group.ddp_config = SimpleNamespace(overlap_grad_reduce=overlap)
    group.is_first_batch = first_batch
    group.is_last_microbatch = True
    group.per_param_grad_ready_counts = {}
    group.golden_per_param_grad_ready_counts = dict.fromkeys(params, 1)
    group.deferred_grad_sync_requested = False
    group.deferred_grad_sync_force_all_reduce = False
    group.external_grad_sync_dispatched = False
    group.grad_reduce_finished = False
    group.grad_reduce_handle = None
    return group, params


@pytest.mark.parametrize('first_batch', [False, True])
def test_early_sync_defers_until_all_external_gradients_are_ready(first_batch):
    group, params = _deferred_group(first_batch=first_batch)
    group.start_grad_sync(force_all_reduce=True)
    assert group.deferred_grad_sync_requested
    assert group.deferred_grad_sync_force_all_reduce
    with pytest.raises(AssertionError, match='Finalize externally managed'):
        group.finish_grad_sync()

    def dispatch(force_all_reduce=False):
        assert not group.pending_external_grads
        assert force_all_reduce
        group.grad_reduce_handle = object()

    with mock.patch.object(group, 'start_grad_sync', side_effect=dispatch) as start:
        group.register_external_grad_ready(params[0])
        assert start.call_count == 0
        group.register_external_grad_ready(params[1])
        assert start.call_count == 1
        with pytest.raises(AssertionError, match='exactly once'):
            group.register_external_grad_ready(params[1])
    assert group.per_param_grad_ready_counts == dict.fromkeys(params, 1)
    group.reset()
    assert group.pending_external_grads == set(params)
    assert not group.deferred_grad_sync_requested
    assert not group.external_grad_sync_dispatched


def test_external_completion_requires_leaving_no_sync():
    group, params = _deferred_group()
    group.is_last_microbatch = False
    with pytest.raises(AssertionError, match='leaving no_sync'):
        group.register_external_grad_ready(params[0])


def test_external_readiness_preserves_force_all_reduce():
    group, params = _deferred_group(first_batch=False)
    with mock.patch.object(group, 'start_grad_sync') as start:
        group.register_external_grad_ready(params[0], force_all_reduce=True)
        group.register_external_grad_ready(params[1])
    start.assert_called_once_with(force_all_reduce=True)


@pytest.mark.parametrize('overlap', [False, True])
def test_real_pp_and_dp_reductions_across_repeated_steps(overlap):
    """Exercise first-batch readiness, steady state and explicit early DDP sync."""
    if not torch.cuda.is_available():
        pytest.skip('Real NCCL PP/DDP integration requires the distributed GPU runner')
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
    try:
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        rank = pp_group.rank()
        module = _Consumer((0, 0, rank + 1, 0)).cuda()
        with torch.no_grad():
            module.pseudo_query.fill_(rank + 1)
            module.key_norm_weight.fill_(2)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=4,
            num_attention_heads=1,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
        )
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(overlap_grad_reduce=overlap),
            module,
            disable_bucketing=True,
        )
        runtime = _runtime([module], pp_group)
        for _ in range(2):
            ddp.zero_grad_buffer()
            runtime.prepare()
            ddp.start_grad_sync()  # Must defer, even though gradients are still zero.
            (runtime.bank.sum() * (rank + 1)).backward()
            runtime.finalize()
            ddp.finish_grad_sync()
            torch.testing.assert_close(
                module.pseudo_query.main_grad, torch.full((4,), 6.0, device='cuda')
            )
            torch.testing.assert_close(
                module.key_norm_weight.main_grad, torch.full((4,), 3.0 * (rank + 1), device='cuda')
            )
            assert not any(group.pending_external_grads for group in ddp.bucket_groups)
    finally:
        Utils.destroy_model_parallel()
