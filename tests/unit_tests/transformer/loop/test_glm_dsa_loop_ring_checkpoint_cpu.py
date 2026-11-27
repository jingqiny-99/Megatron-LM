# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Check PP1 checkpoint ownership and weights on all physical GLM decoder chunks."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.backends import LocalSpecProvider
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsa_module_spec_for_backend,
)
from megatron.core.models.hybrid import hybrid_block
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.loop.glm_dsa import GLMDSALoopStack, GLMLoopDSAttention
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    CUDA,
    local_hybrid_submodules,
    model_parallel_once,
)
from tests.unit_tests.transformer.loop.test_glm_dsa_loop_model_gpu import _config

# Decoder blocks per PP rank for each physical chunk. The two core visits share its weights.
BLOCKS = ((2, 4, 4, 4, 4, 4, 4), (4, 4, 4, 4, 4, 4, 4), (4, 4, 4, 4, 4, 4, 0))


class _ReplicatedLinear(torch.nn.Linear):
    """CPU equivalent for the replicated DSA indexer projections."""

    def __init__(self, input_size, output_size, *, config, init_method, bias=False, **kwargs):
        super().__init__(
            input_size, output_size, bias=bias, device="cpu", dtype=config.params_dtype
        )
        init_method(self.weight)

    def forward(self, x):
        return super().forward(x), None


class _LocalDSAProvider(LocalSpecProvider):
    def linear(self):
        return _ReplicatedLinear


def _local_dsa_submodules(config):
    # Keep native absorbed MLA, the actual selector and the existing local dense FFN.
    backend = _LocalDSAProvider()
    attention = get_dsa_module_spec_for_backend(config, backend)
    attention.submodules.linear_q_down_proj = backend.column_parallel_linear()
    attention.submodules.linear_kv_down_proj = backend.column_parallel_linear()
    attention.submodules.core_attention.module = GLMLoopDSAttention
    local = local_hybrid_submodules()
    layer = local.attention_layer
    dsa = replace(layer, submodules=replace(layer.submodules, self_attention=attention))
    return replace(local, dsa_layer=dsa)


def _stack(config, submodules, symbols, offset, rank, pre, post):
    groups = ProcessGroupCollection.use_mpu_process_groups()
    # Only rank/size are consulted for model construction; TP and DP use the real Gloo groups.
    groups.pp = SimpleNamespace(rank=lambda: rank, size=lambda: config.pipeline_model_parallel_size)
    return GLMDSALoopStack(
        config,
        submodules=submodules,
        layer_config_list=validate_segment_layers(symbols, config),
        pp_layer_offset=offset,
        pre_process=pre,
        post_process=post,
        pg_collection=groups,
    )


def _shards(stack):
    return stack.sharded_state_dict("decoder.", metadata={"dp_cp_group": stack.pg_collection.dp_cp})


def _selectors(stack, offset=0):
    result = []
    for i, layer in enumerate(stack.layers, offset):
        if i % 2:
            continue
        attention = layer.self_attention.core_attention
        result.append(
            (attention.layer_number, attention.source_layer, attention.indexer is not None)
        )
    return result


def _pp1_key(local_key, offset):
    if local_key.startswith("layers."):
        _, local_index, name = local_key.split(".", 2)
        return f"layers.{offset + int(local_index)}.{name}"
    return local_key


@pytest.mark.skipif(CUDA, reason="CPU decoder construction with local dense FFNs and Gloo")
@pytest.mark.parametrize(
    ("stages", "layout"),
    ((7, "reference"), (8, "reference"), (8, "balanced"), (8, "critical_path")),
)
def test_glm_physical_chunks_restore_the_full_pp1_checkpoint(monkeypatch, stages, layout):
    sublayers = tuple(tuple(2 * n for n in section) for section in BLOCKS)
    if stages == 8:
        sublayers = ((2, 2, 8, 8, 8, 8, 8, 8), (8, 8, 8, 8, 8, 8, 7, 1), (8, 8, 0, 0, 8, 8, 0, 16))
    if layout == "balanced":
        sublayers = ((2, 2, 8, 8, 8, 8, 8, 8), (8, 8, 7, 9, 7, 9, 7, 1), (8, 8, 8, 0, 8, 0, 8, 8))
    if layout == "critical_path":
        sublayers = ((2, 2, 8, 8, 8, 8, 8, 8), (8, 8, 8, 8, 8, 7, 8, 1), (8, 8, 8, 8, 8, 8, 0, 0))
    symbols = "D-" * 78
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(hybrid_block, "TENorm", WrappedTorchNorm)
    with model_parallel_once(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(54)
        architecture = dict(num_layers=156, prelude=52, coda=48)
        pp1 = _config(True, False, **architecture)
        ring = _config(
            True,
            False,
            **architecture,
            loop_schedule="ring",
            loop_ring_separate_edges=True,
            pipeline_model_parallel_size=stages,
            virtual_pipeline_model_parallel_size=3,
            pipeline_dtype=torch.float32,
            overlap_p2p_comm=True,
            batch_p2p_comm=False,
        )
        submodules = _local_dsa_submodules(pp1)
        reference = _stack(pp1, submodules, symbols, 0, 0, pre=True, post=True)
        # Identically initialized parameters (e.g. norm scales) must distinguish wrong owners.
        with torch.no_grad():
            for marker, weight in enumerate(reference.parameters(), start=1):
                weight[(0,) * weight.ndim] = marker
        reference_state = reference.state_dict()
        pp1_shards = {shard.key: shard.data for shard in _shards(reference).values()}
        assert len(pp1_shards) == len(reference_state) == 1514

        selector_layers = (1, 2, *range(3, 27, 4), *range(27, 55, 4), *range(55, 79, 4))
        expected = [
            (n, max(i for i in selector_layers if i <= n), n in selector_layers)
            for n in range(1, 79)
        ]
        assert _selectors(reference) == expected
        seen = set()
        ring_selectors = []
        offset = 0
        for section, counts in enumerate(sublayers):
            for rank, count in enumerate(counts):
                first = (section, rank) == (0, 0)
                last = (section, rank) == (2, stages - 1)
                chunk = _stack(
                    ring, submodules, symbols[offset : offset + count], offset, rank, first, last
                )
                assert len(chunk.layers) == count
                assert hasattr(chunk, "final_norm") == last
                if (section, rank) == (1, stages - 1):
                    assert len(chunk.loop_gates) == 2
                    assert chunk.loop_gates[0].proj.weight is not chunk.loop_gates[1].proj.weight
                else:
                    assert len(chunk.loop_gates) == 0
                ring_selectors.extend(_selectors(chunk, offset))

                # Load via canonical DCP names, then verify the ordinary module state using
                # the independent physical layer offset, including both gates on the core owner.
                local_state = {}
                for local_name, shard in _shards(chunk).items():
                    assert shard.key not in seen, shard.key
                    seen.add(shard.key)
                    local_state[local_name.removeprefix("decoder.")] = pp1_shards[shard.key]
                chunk.load_state_dict(local_state, strict=True)
                for local_key, actual in chunk.state_dict().items():
                    key = _pp1_key(local_key, offset)
                    wanted = reference_state[key]
                    if isinstance(wanted, torch.Tensor):
                        assert torch.equal(actual, wanted), (section, rank, key)
                    else:
                        assert actual == wanted, (section, rank, key)
                offset += count

        assert offset == 156
        assert seen == pp1_shards.keys()
        assert ring_selectors == expected
