# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The last activation and first gradient at the batched interleaved warmup boundary."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_pipelining_with_interleaving,
    get_pp_rank_microbatches,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("stages,batch,group", [(4, 4, 4), (7, 8, 8), (7, 56, 8), (7, 64, 8)])
def test_only_exact_boundary_pairs_last_forward_and_first_backward(stages, batch, group):
    for pp_rank in range(stages):
        comm = SimpleNamespace(
            pp_group=SimpleNamespace(size=lambda: stages, rank=lambda: pp_rank),
            virtual_pipeline_model_parallel_size=4,
        )
        original = get_pp_rank_microbatches(batch, 4, group, p2p_communicator=comm)
        batched = get_pp_rank_microbatches(
            batch, 4, group, p2p_communicator=comm, batch_p2p_comm=True
        )
        natural = 2 * (stages - pp_rank - 1) + 3 * group
        assert original[1] == (natural >= batch * 4)
        assert batched[1] == (natural > batch * 4)
        assert original[0] == batched[0] == batch * 4
        assert original[2:] == batched[2:]


class _Chunk(torch.nn.Module):
    def __init__(self, config, view):
        super().__init__()
        self.config = config
        self.vp_stage = view
        self.model_type = ModelType.encoder_or_decoder
        self.scale = torch.nn.Parameter(torch.ones((), device="cuda"))

    def set_input_tensor(self, inputs):
        self.pipeline_input = inputs[0]


@pytest.mark.parametrize("stages,batch", [(4, 4), (7, 8)])
def test_short_batched_interleaved_values_and_all_chunk_gradients(stages, batch):
    if not torch.cuda.is_available() or Utils.world_size % stages:
        pytest.skip(f"requires a multiple of {stages} CUDA ranks")
    views = 4
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=stages, virtual_pipeline_model_parallel_size=views
    )
    try:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        config = TransformerConfig(
            num_layers=stages * views,
            hidden_size=8,
            num_attention_heads=1,
            pipeline_model_parallel_size=stages,
            virtual_pipeline_model_parallel_size=views,
            pipeline_dtype=torch.float32,
            batch_p2p_comm=True,
            overlap_p2p_comm=False,
            microbatch_group_size_per_vp_stage=batch,
        )
        chunks = [_Chunk(config, view) for view in range(views)]

        def step(data_iterator, model):
            sample = next(data_iterator)
            x = model.pipeline_input
            if x is None:
                x = torch.full((1, 1, 8), sample + 1.0, device="cuda")
            output = x * model.scale + 1

            def loss(value):
                return value.mean(), {"value": value.detach().mean().item()}

            return output, loss

        reports = forward_backward_pipelining_with_interleaving(
            forward_step_func=step,
            data_iterator=[iter(range(batch)) for _ in chunks],
            model=chunks,
            num_microbatches=batch,
            seq_length=1,
            micro_batch_size=1,
        )
        if pp_rank == stages - 1:
            assert [record["value"] for record in reports] == [
                i + 1 + stages * views for i in range(batch)
            ]
        for view, chunk in enumerate(chunks):
            assert chunk.scale.grad is not None
            assert chunk.scale.grad.item() == pytest.approx(
                (batch + 1) / 2 + stages * view + pp_rank
            )
    finally:
        Utils.destroy_model_parallel()
