# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``train()`` wraps every microbatch but the last in the model's ``no_sync()`` exactly when the
wrapper reduces gradients from its backward hooks (``reduces_grads_during_backward``). CPU only:
the wrappers are built without their ``__init__``, carrying just the config the check reads."""

import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallelV1,
    FullyShardedDataParallelV2,
)
from megatron.training.training import reduces_grads_during_backward


def _wrapper(cls, **ddp_fields):
    chunk = object.__new__(cls)
    chunk.ddp_config = DistributedDataParallelConfig(**ddp_fields)
    return chunk


def test_megatron_fsdp_v1_follows_its_own_overlap_flag_not_the_argument():
    # MegatronFSDP.__init__ sets ddp_config.overlap_grad_reduce for gradient-sharding
    # strategies while args.overlap_grad_reduce stays False (the Pro FSDP recipes): its HSDP
    # DP-Outer reduction then runs only in the backward that no_sync() marks as the last.
    sharded = _wrapper(
        FullyShardedDataParallelV1,
        use_megatron_fsdp=True,
        data_parallel_sharding_strategy="optim_grads_params",
        num_distributed_optimizer_instances=2,
        overlap_grad_reduce=True,
    )
    assert reduces_grads_during_backward(sharded)
    unsharded = _wrapper(
        FullyShardedDataParallelV1,
        use_megatron_fsdp=True,
        data_parallel_sharding_strategy="optim",
        overlap_grad_reduce=False,
    )
    assert not reduces_grads_during_backward(unsharded)


def test_ddp_follows_overlap_grad_reduce():
    assert reduces_grads_during_backward(
        _wrapper(DistributedDataParallel, overlap_grad_reduce=True)
    )
    assert not reduces_grads_during_backward(
        _wrapper(DistributedDataParallel, overlap_grad_reduce=False)
    )


def test_megatron_fsdp_v2_always_and_an_unwrapped_module_never():
    assert reduces_grads_during_backward(object.__new__(FullyShardedDataParallelV2))
    assert not reduces_grads_during_backward(torch.nn.Linear(2, 2))
