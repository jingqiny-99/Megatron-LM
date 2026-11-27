# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint keys of the ring schedule (``loop_schedule='ring'``): the stage models of a ring
loop save the PP = 1 model's keys.

Local-layer ``HybridModel`` loops with mHC 4 and one MTP depth (the parity mini's role
structure in dense layers: prelude 3 blocks, core 8, coda 2), at P in {2, 4} and k in {1, 2}:
every stage chunk ``(p, j)`` is built in one process (a P-rank stub pipeline group, ``vp_stage``
j at k > 1, no embedding group), as ``get_model`` builds it on pipeline rank p. The union of the
chunks' checkpoint keys (``ShardedTensor.key`` and global shape) equals the PP = 1 model's;
``decoder.loop_adapter.weight`` and ``decoder.loop_norm.weight`` appear exactly once, on the
round-start and the round-end chunk; the only key two chunks share is the word embedding, which
the MTP chunk ties to the first stage's (dev's pipeline placement of the MTP embedding copy).
On CPU the mHC layers need ``TORCHDYNAMO_DISABLE=1``.
"""

import copy

import pytest
import torch

from megatron.core.dist_checkpointing.mapping import (
    ShardedObject,
    ShardedTensor,
    ShardedTensorFactory,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from tests.unit_tests.transformer.loop import hybrid_loop_utils
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    SEQ,
    VOCAB,
    device,
    hybrid_loop_config,
    local_model_spec,
    model_parallel_once,
)
from tests.unit_tests.transformer.loop.ring_utils import (
    StageGroup,
    ring_fields,
    ring_layout,
    stage_group_ranks,
)

mhc_on_this_device = hybrid_loop_utils.mhc_on_this_device
cpu_mtp = hybrid_loop_utils.cpu_mtp

MAIN = "*-" * 13
MTP = "*-"
PRELUDE, CODA = 6, 4
MHC = dict(enable_mhc_connections=True, mhc_num_residual_streams=4, mhc_sinkhorn_iterations=5)
LOOP = dict(loop_iterations=4, mtp_num_layers=1, mtp_loss_scaling_factor=0.1, **MHC)


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


def _keys(sharded_state_dict):
    """``{checkpoint key: global shape}`` of a sharded state dict."""
    keys = {}
    for value in sharded_state_dict.values():
        if isinstance(value, (ShardedTensor, ShardedObject)):
            keys[value.key] = tuple(value.global_shape)
        elif isinstance(value, ShardedTensorFactory):
            keys[value.key] = tuple(value.data.shape)
    return keys


def _model(
    config, pattern, *, pre_process=True, post_process=True, pg_collection=None, vp_stage=None
):
    torch.manual_seed(1234)
    model = HybridModel(
        config=config,
        hybrid_stack_spec=local_model_spec(LoopHybridStack, hyper_connections=True),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
        pre_process=pre_process,
        post_process=post_process,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )
    return model.to(device())


def _stage_pg_collection(stage, stages):
    """The model's process groups as pipeline rank ``stage`` of ``stages`` sees them (a stub
    pipeline group; no embedding group, so no tied-copy all-reduce at init)."""
    collection = copy.copy(ProcessGroupCollection.use_mpu_process_groups())
    collection.pp = StageGroup(stage, stages)
    collection.embd = None
    return collection


@pytest.mark.parametrize("stages,chunks", [(2, 1), (2, 2), (4, 1), (4, 2)])
def test_ring_chunk_keys_are_the_pp1_keys(mhc_on_this_device, cpu_mtp, stages, chunks):
    plain = _model(hybrid_loop_config(MAIN, PRELUDE, CODA, **LOOP), MAIN + "/" + MTP)
    plain_keys = _keys(plain.sharded_state_dict())
    assert "decoder.loop_adapter.weight" in plain_keys
    assert "decoder.loop_norm.weight" in plain_keys

    config = hybrid_loop_config(MAIN, PRELUDE, CODA, **LOOP, **ring_fields(stages, chunks))
    piped, segments = ring_layout(MAIN, PRELUDE, CODA, stages, chunks)
    last = stages * chunks - 1
    union, owners = {}, {}
    with stage_group_ranks():
        for segment in segments:
            model = _model(
                config,
                piped + "/" + MTP,
                pre_process=segment["index"] == 0,
                post_process=segment["index"] == last,
                pg_collection=_stage_pg_collection(segment["stage"], stages),
                vp_stage=segment["chunk"] if chunks > 1 else None,
            )
            decoder = model.decoder
            assert decoder.loop_ring.offset == segment["offset"]
            assert model.mtp_process == (segment["index"] == last)
            for key, shape in _keys(model.sharded_state_dict()).items():
                assert union.setdefault(key, shape) == shape, key
                owners.setdefault(key, []).append(segment["index"])
    assert union == plain_keys
    assert owners["decoder.loop_adapter.weight"] == [0]
    assert owners["decoder.loop_norm.weight"] == [last]
    shared = {key: chunk_ids for key, chunk_ids in owners.items() if len(chunk_ids) > 1}
    assert shared == {"embedding.word_embeddings.weight": [0, last]}
    # Every decoder sublayer's keys come from the chunk that holds that global sublayer.
    for segment in segments:
        for index in range(segment["offset"], segment["offset"] + len(segment["pattern"])):
            prefix = f"decoder.layers.{index}."
            assert {tuple(o) for k, o in owners.items() if k.startswith(prefix)} == {
                (segment["index"],)
            }, index
