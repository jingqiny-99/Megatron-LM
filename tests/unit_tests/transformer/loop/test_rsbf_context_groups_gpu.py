# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Check the CP2 x DP2 sample domain, EP4 collective domain, and DDP token divisor."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.loop.dispatcher import LoopSampleDispatcher, RSBFGroups
from megatron.core.transformer.loop.program import global_sample_index
from megatron.core.transformer.loop.rsbf_executor import LoopRSBFExecutor
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA, model_parallel_once


@pytest.mark.skipif(not CUDA or int(os.environ.get("WORLD_SIZE", "1")) != 4, reason="needs 4 GPUs")
def test_fixed_context_group_and_masked_token_normalization():
    with model_parallel_once(context_parallel_size=2, expert_model_parallel_size=4):
        groups = ProcessGroupCollection.use_mpu_process_groups()
        dispatcher = LoopSampleDispatcher(RSBFGroups(domain=groups.dp), device=torch.device("cuda"))
        # A lockstep MoE layer issues calls on all four expert ranks. Each fixed-depth
        # per-sample plan is still built over just the two data replicas.
        ep_entry = SimpleNamespace(
            name="ep4", module=None, adapter=SimpleNamespace(groups=lambda module: [groups.ep])
        )
        context = SimpleNamespace(
            config=SimpleNamespace(context_parallel_size=2),
            pg_collection=groups,
            lockstep=SimpleNamespace(entries=lambda: [ep_entry]),
        )
        LoopRSBFExecutor._setup_lockstep(context, dispatcher)
        assert dispatcher.num_eus == 2 and groups.cp.size() == 2 and groups.ep.size() == 4
        context.pg_collection = SimpleNamespace(cp=groups.cp, dp_cp=groups.dp)
        with pytest.raises(ValueError, match="rectangular"):
            LoopRSBFExecutor._setup_lockstep(context, dispatcher)

        config = TransformerConfig(
            num_layers=1, hidden_size=8, num_attention_heads=1,
            context_parallel_size=2, calculate_per_token_loss=True,
        )
        module = torch.nn.Linear(1, 1, bias=False, device="cuda")
        module.config = config
        model = DistributedDataParallel(
            config, DistributedDataParallelConfig(overlap_grad_reduce=False), module,
            pg_collection=groups,
        )
        model.zero_grad_buffer()
        # Four samples of eight tokens. Each context shard gets different mask
        # counts, including zero. Sum only its supervised token losses into DDP.
        values = torch.arange(32, dtype=torch.float32, device="cuda").reshape(4, 8) + 1
        mask = (values.remainder(5) == 0) | (values == 32)
        local_tokens = torch.zeros((), device="cuda", dtype=torch.int32)
        cp_rank = groups.cp.rank()
        chunks = torch.arange(8, device="cuda").chunk(4)
        positions = torch.cat((chunks[cp_rank], chunks[-cp_rank - 1]))
        for mb in range(2):
            sample = global_sample_index(
                mb, 0, micro_batch_size=1, dp_rank=dispatcher.eu, dp_size=dispatcher.num_eus
            )
            assert sample == 2 * mb + groups.dp.rank()
            selected = mask[sample, positions]
            (model(values[sample, positions, None]).flatten() * selected).sum().backward()
            local_tokens.add_(selected.sum())
        finalize_model_grads([model], local_tokens, pg_collection=groups)
        assert local_tokens.item() == mask.sum().item()
        torch.testing.assert_close(module.weight.main_grad.flatten(), values[mask].mean().view(1))
