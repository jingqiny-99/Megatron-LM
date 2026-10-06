# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real DistOpt checkpoint migration across the source-query buffer split.

Run on one node with four GPUs::

    torchrun --standalone --nproc-per-node=4 -m pytest -o addopts= -q \
        tests/unit_tests/distributed/test_attn_res_projection_checkpoint.py

The small PP-agnostic model is deliberately replicated across PP ranks, as in
MCore's optimizer resharding tests. Changing PP1/DP4 to PP2/DP2 exercises real
optimizer DP resharding without duplicating the full-model pipeline tests.
Set ATTNRES_CHECKPOINT_DEST_PP=1 to retain PP1/DP4 throughout instead.

Use the supported fully_reshardable model-space format selected in training by
--dist-ckpt-optim-fully-reshardable. The deprecated fully_sharded_model_space
writer still requests flattened_range, which current ShardedTensor rejects;
it is not a usable qualification format even with the source feature disabled.
The training default, dp_reshardable, addresses optimizer tensors by buffer
index and cannot cross a changed buffer layout. It must first be converted
with the feature disabled and --dist-ckpt-optim-fully-reshardable. This test
does not disguise that restriction by loading only model weights or
reinitializing optimizer moments.
"""

import gc
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.transformer.attention_residual import AttentionResidual
from megatron.core.transformer.attention_residual_projection_kernels import project_source
from megatron.core.transformer.attention_residual_projection_runtime import (
    finalize_attn_res_projection,
    get_attn_res_projection_runtime,
    prepare_attn_res_projection,
)
from megatron.core.transformer.attention_residual_source_state import SourceProjection
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_HIDDEN = 256
_STEPS = 10
_RESUME_STEPS = 3


class _CheckpointModel(MegatronModule):
    """Use a real AttnRes query beside ordinary FP32 and optional BF16 params."""

    def __init__(self, impl, bf16):
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        config = TransformerConfig(
            num_layers=4,
            hidden_size=_HIDDEN,
            num_attention_heads=4,
            pipeline_model_parallel_size=pp_size,
            enable_attention_residuals=True,
            attn_res_block_layers=2,
            attn_res_impl=impl,
            normalization='RMSNorm',
            bf16=bf16,
            params_dtype=torch.bfloat16 if bf16 else torch.float32,
            pipeline_dtype=torch.bfloat16 if bf16 else torch.float32,
        )
        super().__init__(config)
        self.pre = torch.nn.Linear(_HIDDEN, _HIDDEN, bias=False, dtype=config.params_dtype)
        # An ordinary FP32 parameter makes the changed grouping observable even
        # under BF16: q/gamma must move out of a shared FP32 buffer.
        self.gain = mark_keep_in_fp32(torch.nn.Parameter(torch.linspace(0.7, 1.3, _HIDDEN)))
        self.attention = AttentionResidual(
            config, layer_number=parallel_state.get_pipeline_model_parallel_rank() + 1
        )
        with torch.no_grad():
            self.attention.pseudo_query.normal_(std=0.01)
            self.attention.key_norm_weight.uniform_(0.8, 1.2)

    def forward(self, inputs):
        projected = self.pre(inputs)
        sources = [inputs, projected, (projected.float() * self.gain.tanh()).to(inputs.dtype)]
        if self.config.attn_res_impl == 'source':
            runtime = get_attn_res_projection_runtime(self)
            columns = tuple(range(runtime.bank.shape[0]))
            for source_id in range(2):
                proxy, logits = project_source(
                    sources[source_id], runtime.bank, eps=self.config.layernorm_epsilon
                )
                proxy._attn_res_source_projection = SourceProjection(source_id, columns, logits)
                sources[source_id] = proxy
            # The final partial remains consumer-scored, covering both paths.
        return self.attention(sources)

    def sharded_state_dict(self, *args, **kwargs):
        state = super().sharded_state_dict(*args, **kwargs)
        # This fixture intentionally replicates its full model across PP.
        # Identify those replicas so the writer has exactly one primary copy.
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        for tensor in state.values():
            tensor.replica_id = (pp_rank, *tensor.replica_id[1:])
        return state


def _build(impl, bf16):
    torch.manual_seed(9182)
    model = _CheckpointModel(impl, bf16).cuda()
    ddp = DistributedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            # Keep actual reduce-scatter enabled; synchronous gather simplifies
            # exact checkpoint snapshots without replacing optimizer behavior.
            overlap_param_gather=False,
            bucket_size=8192,
        ),
        model,
        disable_bucketing=True,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer='adam',
            lr=1e-4,
            weight_decay=0.01,
            clip_grad=0.0,
            bf16=bf16,
            params_dtype=model.config.params_dtype,
            use_distributed_optimizer=True,
        ),
        [ddp],
    )
    inners = getattr(optimizer, 'chained_optimizers', [optimizer])
    assert len(inners) == 1 and isinstance(inners[0], DistributedOptimizer)
    if impl == 'source':
        assert any(group.externally_managed_params for group in ddp.bucket_groups)
        assert all(
            not group.externally_managed_params or group.externally_managed_params == group.params
            for group in ddp.bucket_groups
        )
    return ddp, optimizer


def _train_step(model, optimizer, step):
    model.zero_grad_buffer()
    optimizer.zero_grad()
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    prepare_attn_res_projection(model, pp_group)
    dp_size = parallel_state.get_data_parallel_world_size()
    dp_rank = parallel_state.get_data_parallel_rank()
    # Preserve the same global batch when DP size changes. Every PP replica
    # independently computes the same optimizer update, with distinct bank IDs.
    global_batch = Utils.world_size
    losses = torch.zeros((), device='cuda', dtype=torch.float32)
    with model.no_sync():
        for batch_id in range(dp_rank, global_batch, dp_size):
            generator = torch.Generator().manual_seed(23000 + step * global_batch + batch_id)
            inputs = torch.randn(8, 1, _HIDDEN, generator=generator).cuda()
            target = torch.randn(8, 1, _HIDDEN, generator=generator).cuda()
            output = model(inputs.to(model.config.params_dtype))
            loss = (output.float() - target).square().mean() * (dp_size / global_batch)
            optimizer.scale_loss(loss).backward()
            losses += loss.detach()
    # An explicit early sync must defer managed query buckets until PP SUM.
    model.start_grad_sync()
    finalize_attn_res_projection(model, pp_group)
    model.finish_grad_sync()
    result = optimizer.step()
    assert result[0], 'Optimizer skipped the checkpoint qualification update'
    torch.distributed.all_reduce(losses, group=parallel_state.get_data_parallel_group())
    return float(losses / dp_size)


def _canonical_snapshot(model, optimizer):
    """Reassemble actual master params and Adam moments by original model name."""
    inner = getattr(optimizer, 'chained_optimizers', [optimizer])[0]
    dp_group = parallel_state.get_data_parallel_group()
    snapshot = {}
    for name, param in model.module.named_parameters():
        states = {
            key: torch.zeros(param.numel(), device=param.device, dtype=torch.float32)
            for key in ('param', 'exp_avg', 'exp_avg_sq')
        }
        if param in inner.model_param_group_index_map:
            local_range = inner._get_model_param_range_map(param)['param']
            local = inner._get_main_param_and_optimizer_states(param)
            for key, tensor in states.items():
                tensor[local_range.start : local_range.end].copy_(local[key])
        for key, tensor in states.items():
            torch.distributed.all_reduce(tensor, group=dp_group)
            assert torch.isfinite(tensor).all()
            snapshot[f'{key}.{name}'] = tensor.cpu()
    snapshot.update(
        {
            f'model.{name}': value.detach().cpu().clone()
            for name, value in model.module.state_dict().items()
        }
    )
    # FusedAdam stores step on groups; native Adam stores it on state entries.
    state = inner.optimizer.state_dict()
    steps = [group['step'] for group in state['param_groups'] if 'step' in group]
    steps.extend(item['step'] for item in state['state'].values() if 'step' in item)
    snapshot['adam_steps'] = sorted({int(step) for step in steps})
    assert snapshot['adam_steps'], 'The optimizer checkpoint must include Adam step counts'
    return snapshot


def _assert_snapshot(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            torch.testing.assert_close(
                actual[key], expected[key], atol=0, rtol=0, msg=lambda msg: f'{key}: {msg}'
            )
        else:
            assert actual[key] == expected[key], key


def _checkpoint_state(model, optimizer, sharding_type, *, loading=False):
    model_state = model.module.sharded_state_dict()
    assert 'attention.pseudo_query' in model_state
    assert 'attention.key_norm_weight' in model_state
    assert not any('projection' in name or 'bank' in name for name in model_state)
    return {
        'model': model_state,
        'optimizer': optimizer.sharded_state_dict(
            model_state, is_loading=loading, metadata={'distrib_optim_sharding_type': sharding_type}
        ),
    }


def _load(model, optimizer, directory, sharding_type):
    template = _checkpoint_state(model, optimizer, sharding_type, loading=True)
    state = dist_checkpointing.load(template, directory)
    model.module.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    # Loaded master shards are already authoritative. Do not call
    # reload_model_params(), which would overwrite them from rounded BF16.


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires real NCCL reduce-scatter')
@pytest.mark.parametrize('bf16', [False, True], ids=['fp32', 'bf16'])
@pytest.mark.parametrize('sharding_type', ['fully_reshardable'])
def test_source_projection_distopt_checkpoint_migration(sharding_type, bf16):
    if Utils.world_size < 4 or Utils.world_size % 2:
        pytest.skip('Use torchrun with an even world size of at least four')
    destination_pp = int(os.environ.get('ATTNRES_CHECKPOINT_DEST_PP', '2'))
    assert destination_pp in (1, 2)
    Utils.initialize_model_parallel(1, 1)
    directory = [tempfile.mkdtemp(prefix='attnres-distopt-') if Utils.rank == 0 else None]
    torch.distributed.broadcast_object_list(directory, src=0)
    directory = Path(directory[0])
    if Utils.rank == 0:
        (directory / 'off').mkdir()
        (directory / 'converted').mkdir()
        (directory / 'on').mkdir()
    torch.distributed.barrier()
    try:
        baseline, baseline_optimizer = _build('eager', bf16)
        for step in range(_STEPS):
            _train_step(baseline, baseline_optimizer, step)
        saved = _canonical_snapshot(baseline, baseline_optimizer)
        assert saved['adam_steps'] == [_STEPS]
        for name in ('pseudo_query', 'key_norm_weight'):
            assert torch.count_nonzero(saved[f'exp_avg.attention.{name}'])
        original_buffers = len(baseline.buffers)
        dist_checkpointing.save(
            _checkpoint_state(baseline, baseline_optimizer, sharding_type), directory / 'off'
        )
        del baseline, baseline_optimizer
        gc.collect()
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(1, destination_pp)
        source, source_optimizer = _build('source', bf16)
        assert len(source.buffers) > original_buffers, 'Qualification must change the buffer layout'
        _load(source, source_optimizer, directory / 'off', sharding_type)
        _assert_snapshot(_canonical_snapshot(source, source_optimizer), saved)
        dist_checkpointing.save(
            _checkpoint_state(source, source_optimizer, sharding_type), directory / 'converted'
        )
        source_losses = [
            _train_step(source, source_optimizer, step)
            for step in range(_STEPS, _STEPS + _RESUME_STEPS)
        ]
        resumed = _canonical_snapshot(source, source_optimizer)
        assert resumed['adam_steps'] == [_STEPS + _RESUME_STEPS]
        assert not torch.equal(
            resumed['param.attention.pseudo_query'], saved['param.attention.pseudo_query']
        )
        dist_checkpointing.save(
            _checkpoint_state(source, source_optimizer, sharding_type), directory / 'on'
        )
        del source, source_optimizer
        gc.collect()

        # Replay three source-backend updates from the converted step-10
        # checkpoint, requiring exact results. Cross-backend BF16 interface
        # rounding is qualified separately by the model/kernel parity tests.
        replay, replay_optimizer = _build('source', bf16)
        _load(replay, replay_optimizer, directory / 'converted', sharding_type)
        _assert_snapshot(_canonical_snapshot(replay, replay_optimizer), saved)
        replay_losses = [
            _train_step(replay, replay_optimizer, step)
            for step in range(_STEPS, _STEPS + _RESUME_STEPS)
        ]
        _assert_snapshot(_canonical_snapshot(replay, replay_optimizer), resumed)
        assert replay_losses == source_losses
        del replay, replay_optimizer
        gc.collect()
        Utils.destroy_model_parallel()

        # Read the new managed-buffer checkpoint back with the original DP
        # degree as well; master parameters/moments must remain bitwise intact.
        Utils.initialize_model_parallel(1, 1)
        restored, restored_optimizer = _build('source', bf16)
        _load(restored, restored_optimizer, directory / 'on', sharding_type)
        _assert_snapshot(_canonical_snapshot(restored, restored_optimizer), resumed)
        if Utils.rank == 0:
            print(
                json.dumps(
                    {
                        'qualification': 'attnres_source_distopt_checkpoint',
                        'format': sharding_type,
                        'dtype': 'bf16' if bf16 else 'fp32',
                        'initial_steps': _STEPS,
                        'resumed_steps': _RESUME_STEPS,
                        'source_pp': 1,
                        'destination_pp': destination_pp,
                        'initial_dp': Utils.world_size,
                        'resumed_dp': Utils.world_size // destination_pp,
                        'exact_master_and_moment_restore': True,
                        'exact_three_step_resume_replay': True,
                    },
                    sort_keys=True,
                )
            )
        del restored, restored_optimizer
    finally:
        Utils.destroy_model_parallel()
        if Utils.rank == 0:
            shutil.rmtree(directory)
