# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-FSDP buffer index alignment, on CPU (``build_data_parallel_buffer_index``).

Parameters and gradients are views of buckets. Every item starts at a multiple of
``_ITEM_ALIGNMENT`` elements (an item of at least one row also at a multiple of the row size,
so a row never straddles a shard), the bucket is a multiple of it long (sharded: of the world
size times the common multiple of the row size and the alignment), and no two items overlap.
Without it small parameters packed after one another (the mHC scalars of a hybrid layer, next
to its 2048-wide projection rows) started at arbitrary offsets, and the Triton kernels
torch.compile generates for them failed with a CUDA "misaligned address". A layout that was
already aligned (rows a multiple of the alignment wide, no small items) is unchanged.
"""

import math

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.distributed_data_parallel_config import (
    DistributedDataParallelConfig as FSDPConfig,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    _ITEM_ALIGNMENT,
    build_data_parallel_buffer_index,
)

# (shapes, row size = chunk_size_factor) as ParamAndGradBuffer groups them.
CASES = {
    # The mHC hybrid wrapper's fp32 group: 2048-wide projection rows and the mHC scalars.
    "mhc_wrapper": ([(24, 2048), (4,), (1,), (1,), (1,), (24,)], 2048),
    # Square weights with norms, a sink and odd-sized small parameters.
    "weights_and_small": ([(512, 512), (512,), (8,), (3,), (512, 512), (16,)], 512),
    # A group of 1-D parameters only.
    "one_dimensional": ([(7,), (5,), (3,), (64,), (100,)], 1),
    # A row size below the alignment.
    "narrow_rows": ([(6, 4), (5,), (10, 4), (2,)], 4),
}


def _index(shapes, chunk, world, strategy="optim_grads_params"):
    return build_data_parallel_buffer_index(
        [torch.Size(shape) for shape in shapes],
        0,
        world,
        True,
        FSDPConfig(data_parallel_sharding_strategy=strategy),
        bucket_id=0,
        chunk_size_factor=chunk,
    )


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("case", sorted(CASES))
def test_every_item_is_aligned_and_inside_its_bucket(case, world):
    shapes, chunk = CASES[case]
    items, bucket, shard = _index(shapes, chunk, world)
    assert sorted(items) == list(range(len(shapes)))
    spans = []
    for item_id, item in items.items():
        size = torch.Size(shapes[item_id]).numel()
        assert item.size == size
        assert item.global_data_index % _ITEM_ALIGNMENT == 0, (item_id, item)
        if size >= chunk:
            assert item.global_data_index % chunk == 0, (item_id, item)
        spans.append((item.global_data_index, item.global_data_index + size))
    spans.sort()
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end <= start, spans
    assert spans[-1][1] <= bucket.size
    assert bucket.size % (world * math.lcm(chunk, _ITEM_ALIGNMENT)) == 0
    assert shard.size * world == bucket.size and shard.size % _ITEM_ALIGNMENT == 0


@pytest.mark.parametrize("case", sorted(CASES))
def test_unsharded_buckets_are_aligned_too(case):
    shapes, chunk = CASES[case]
    items, bucket, _ = _index(shapes, chunk, 4, strategy="no_shard")
    assert all(item.global_data_index % _ITEM_ALIGNMENT == 0 for item in items.values())
    assert bucket.size % _ITEM_ALIGNMENT == 0


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_aligned_layouts_are_unchanged(world):
    """Rows a multiple of the alignment wide and no small items: items back to back from 0 and
    the bucket padded to the world size times the row size, as before the alignment."""
    shapes = [(128, 512), (64, 512), (512,), (32, 512)]
    items, bucket, _ = _index(shapes, 512, world)
    starts = [items[i].global_data_index for i in range(len(shapes))]
    assert starts == [0, 128 * 512, 192 * 512, 193 * 512]
    end = 225 * 512
    assert bucket.size == math.ceil(end / (world * 512)) * world * 512
