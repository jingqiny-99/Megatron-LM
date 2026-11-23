# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import logging

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor, load, load_plain_tensors, save
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLP, MLPSubmodules, apply_swiglu_sharded_factory
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def initialize_mlp(glu=True):
    model_parallel_cuda_manual_seed(123)
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    transformer_config = TransformerConfig(
        num_layers=pp_size,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
        gated_linear_unit=glu,
    )
    mlp_submodules = get_submodules(get_gpt_layer_with_transformer_engine_submodules().mlp)
    assert isinstance(mlp_submodules, MLPSubmodules)
    return MLP(transformer_config, mlp_submodules)


class TestParallelMLPWithGLU:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "src_tp_pp,dest_tp_pp",
        [
            # changing PP is impossible because the number of layers must be the same
            ((2, 2), (4, 2)),
            ((1, 1), (8, 1)),
            ((1, 8), (1, 8)),
            ((1, 1), (2, 1)),
        ],
    )
    @pytest.mark.parametrize('singleton_local_shards', [True, False])
    def test_parallel_reconfiguration_e2e(
        self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp, singleton_local_shards
    ):
        """Test module saving and loading with different TP/PP"""
        Utils.initialize_model_parallel(*src_tp_pp)
        metadata = {'singleton_local_shards': singleton_local_shards}

        with (
            TempNamedDir(tmp_path_dist_ckpt / 'test_mlp_glu_reconfiguration_model_A') as ckpt_dir_A,
            TempNamedDir(tmp_path_dist_ckpt / 'test_mlp_glu_reconfiguration_model_B') as ckpt_dir_B,
        ):
            # Save checkpoint A
            layer_prefix = f'{parallel_state.get_pipeline_model_parallel_rank()}.'
            mlp_A = initialize_mlp()
            save(mlp_A.sharded_state_dict(prefix=layer_prefix, metadata=metadata), ckpt_dir_A)
            Utils.destroy_model_parallel()

            if "dp_cp_group" in metadata.keys():
                del metadata["dp_cp_group"]

            # Load checkpoint A with different TP/PP and save as checkpoint B
            Utils.initialize_model_parallel(*dest_tp_pp)
            mlp_B = initialize_mlp()
            state_dict = load(
                mlp_B.sharded_state_dict(prefix=layer_prefix, metadata=metadata), ckpt_dir_A
            )
            mlp_B.load_state_dict({k.removeprefix(layer_prefix): v for k, v in state_dict.items()})
            save(mlp_B.sharded_state_dict(prefix=layer_prefix, metadata=metadata), ckpt_dir_B)
            Utils.destroy_model_parallel()

            # Test both checkpoints are equal
            Utils.initialize_model_parallel(1, 1)
            state_dict_A = load_plain_tensors(ckpt_dir_A)
            state_dict_B = load_plain_tensors(ckpt_dir_B)
            diffs = diff(state_dict_A, state_dict_B)
            assert not any(map(bool, diffs)), diffs

    def test_oom_is_handled(self, caplog):
        Utils.initialize_model_parallel(Utils.world_size, 1)
        dtype = torch.bfloat16

        # Compute free memory in bytes
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device)
        total = torch.cuda.get_device_properties(device).total_memory
        free = total - allocated

        # We should create two tensor which take up between 50% and 100% of free memory,
        # so that the torch.cat tries to allocate twice as many and OOMs.
        expected_local_num_bytes = free * 0.6

        local_num_elems = expected_local_num_bytes // torch._utils._element_size(dtype)
        local_num_elems = int(local_num_elems // 1024 * 1024)
        assert local_num_elems % 1024 == 0

        local_w_plus_v_shape = (local_num_elems // 512, 512)
        local_w_or_v_shape = (local_num_elems // 1024, 512)

        fc1_weight_sh_ten = ShardedTensor.from_rank_offsets(
            'a',
            torch.ones(local_w_plus_v_shape, device='cuda', dtype=dtype),
            (0, Utils.rank, Utils.world_size),
        )
        fc1_factory = apply_swiglu_sharded_factory(fc1_weight_sh_ten, ())
        sharded_state_dict = fc1_factory.build()
        assert len(sharded_state_dict) == 2
        assert sharded_state_dict[0].data.shape == local_w_or_v_shape
        # NOTE: with singleton_local_shards=True this assert would fail - global shape is
        #  `(Utils.world_size * local_w_or_v_shape[0], local_w_or_v_shape[1])`
        assert sharded_state_dict[0].global_shape[-2:] == (
            Utils.world_size * local_w_plus_v_shape[0],
            local_w_or_v_shape[1],
        )

        # Checkpoint load replaces ShardedTensors with tensors.
        # Load happens in-place, so we can just use the same tensors
        loaded_state_dict = [sh_ten.data for sh_ten in sharded_state_dict]

        # The critical part that should OOM:
        with caplog.at_level(logging.WARNING):
            fc1_factory.merge_fn(loaded_state_dict)
            assert "CUDA OutOfMemoryError encountered during tensors merging" in caplog.text


@pytest.mark.parametrize("singleton", [False, True])
@pytest.mark.parametrize("tp_size,gtp_size,half,rows", [(2, 4, 8, 4), (2, 3, 6, 4), (2, 4, 3, 4)])
def test_gtp_swiglu_checkpoint_rows(tp_size, gtp_size, half, rows, singleton):
    """Every canonical row is written once; each physical shard restores only its own W/V."""
    width = 3
    full = torch.arange(2 * tp_size * half * width).view(-1, width)
    gate, up = full.chunk(2)
    expected = {"fc1_w": gate, "fc1_v": up} if singleton else {"fc1": full}
    restored = {k: torch.full_like(v, -1) for k, v in expected.items()}
    counts = {k: torch.zeros_like(v) for k, v in expected.items()}
    for tp in range(tp_size):
        logical = torch.cat([gate.chunk(tp_size)[tp], up.chunk(tp_size)[tp]])
        physical = torch.zeros(gtp_size * rows, width, dtype=full.dtype)
        physical[: logical.shape[0]] = logical
        for gtp, local in enumerate(physical.chunk(gtp_size)):
            offsets = ((0, 1, 3),)
            entry = ShardedTensor.from_rank_offsets(
                "fc1",
                local,
                *offsets,
                (1, tp * gtp_size + gtp, tp_size * gtp_size),
                prepend_axis_num=1,
            )
            entry.gtp_size = gtp_size
            entry.gtp_pad_length = gtp_size * rows - 2 * half
            factory = apply_swiglu_sharded_factory(entry, offsets, singleton)
            loaded = []
            for piece in factory.build():
                start = piece.global_offset[1]
                end = start + piece.data.shape[0]
                assert piece.global_offset[0] == 1
                assert piece.global_shape == (3, *expected[piece.key].shape)
                restored[piece.key][start:end].copy_(piece.data)
                counts[piece.key][start:end] += 1
                loaded.append(expected[piece.key][start:end].clone())
            torch.testing.assert_close(factory.merge_fn(loaded), local, rtol=0, atol=0)
    for key in expected:
        assert counts[key].eq(1).all()
        torch.testing.assert_close(restored[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("half", [4, 3, 1])
def test_gtp_swiglu_torch_dist_roundtrip(tmp_path_dist_ckpt, half):
    """DP -> contiguous GTP weight -> DP using the real distributed checkpoint backend."""
    Utils.initialize_model_parallel(1, 1)
    try:
        rank, size = Utils.rank, Utils.world_size
        dense = torch.arange(4 * half, device="cuda").reshape(2 * half, 2)
        rows = (2 * half + size - 1) // size
        pad = size * rows - 2 * half
        physical = torch.cat([dense, dense.new_zeros((pad, 2))])
        local = physical.chunk(size)[rank]
        with (
            TempNamedDir(tmp_path_dist_ckpt / "swiglu_dp") as dp_dir,
            TempNamedDir(tmp_path_dist_ckpt / "swiglu_gtp") as gtp_dir,
        ):

            def plain(tensor):
                return {"fc1": ShardedTensor.from_rank_offsets("fc1", tensor, replica_id=rank)}

            def sharded(tensor):
                entry = ShardedTensor.from_rank_offsets("fc1", tensor, (0, rank, size))
                entry.gtp_size = size
                entry.gtp_pad_length = pad
                return {"fc1": apply_swiglu_sharded_factory(entry, ())}

            save(plain(dense), dp_dir)
            from_dp = load(sharded(torch.empty_like(local)), dp_dir)["fc1"]
            torch.testing.assert_close(from_dp, local, rtol=0, atol=0)
            save(sharded(from_dp), gtp_dir)
            from_gtp = load(plain(torch.empty_like(dense)), gtp_dir)["fc1"]
            torch.testing.assert_close(from_gtp, dense, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()
