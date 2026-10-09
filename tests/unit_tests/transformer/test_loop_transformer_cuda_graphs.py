# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared-depth graph replay against independent native decoder math."""

from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import (
    annotate_first_last_layer,
    create_cudagraphs,
    delete_cuda_graphs,
)
from megatron.core.transformer.loop_transformer_block import LoopTransformerBlock
from tests.unit_tests.transformer.test_loop_transformer import (
    NativeDecoder,
    compare,
    make_block,
    parallel_environment,
)

pytestmark = pytest.mark.launch_on_gb200


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("production_shape", [False, True])
@pytest.mark.parametrize("fusion", [False, True])
def test_shared_graph_native_updates(dtype, production_shape, fusion):
    """Check live activation ownership and DDP accumulation across graph replays."""
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    model_parallel_cuda_manual_seed(123)
    template = make_block(depth=4, dtype=dtype, production_shape=production_shape)
    config = replace(
        template.config,
        cuda_graph_impl="local",
        cuda_graph_warmup_steps=3,
        gradient_accumulation_fusion=fusion,
        inference_cuda_graph_scope=None,
    )
    block = LoopTransformerBlock(
        config=config,
        spec=get_gpt_layer_with_transformer_engine_spec(),
        pre_process=True,
        post_process=True,
    ).cuda()
    block.load_state_dict(template.state_dict())
    del template
    annotate_first_last_layer(block.layers)
    ddp = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(overlap_grad_reduce=False, grad_reduce_in_fp32=True),
        block,
        pg_collection=block.pg_collection,
    )
    ddp.broadcast_params()
    native = NativeDecoder(block).cuda()
    rope = RotaryEmbedding(config.kv_channels, rotary_percent=1.0)(16).cuda()
    scale = config.init_method_std if dtype == torch.bfloat16 else 1.0

    def inputs():
        return (
            torch.randn(16, 2, config.hidden_size, device="cuda", dtype=dtype) * scale
        ).requires_grad_(True)

    try:
        warm_x = inputs()
        warm_y = ddp(warm_x, attention_mask=None, rotary_pos_emb=rope)
        compare(warm_y, native(warm_x.detach().clone()), dtype)
        warm_y.float().sum().backward()
        ddp.finish_grad_sync()
        create_cudagraphs()
        for name, parameter in block.named_parameters():
            torch.testing.assert_close(parameter, native.w(name), rtol=0, atol=0)
        assert [len(layer.cudagraph_manager.cudagraph_runners) for layer in block.layers] == [
            1,
            4,
            4,
            1,
        ]
        optimizer = torch.optim.SGD(block.parameters(), lr=0.003)
        reference_optimizer = torch.optim.SGD(native.parameters(), lr=0.003)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            ddp.zero_grad_buffer()
            native.zero_grad(set_to_none=True)
            for microbatch in range(2):
                x = inputs()
                ref_x = x.detach().clone().requires_grad_(True)
                with ddp.no_sync() if microbatch == 0 else nullcontext():
                    y = ddp(x, attention_mask=None, rotary_pos_emb=rope)
                    ref_y = native(ref_x)
                    compare(y, ref_y, dtype)
                    # Mean-reduced probe loss, like the training objective.
                    upstream = torch.randn_like(y) / y.numel()
                    (y * upstream).float().sum().backward()
                (ref_y * upstream).float().sum().backward()
                compare(x.grad, ref_x.grad, dtype)
            ddp.finish_grad_sync()
            for name, parameter in block.named_parameters():
                expected = native.w(name).grad
                torch.distributed.all_reduce(expected, group=block.pg_collection.dp_cp)
                expected /= torch.distributed.get_world_size(block.pg_collection.dp_cp)
                try:
                    compare(parameter.main_grad, expected, dtype)
                except AssertionError as error:
                    raise AssertionError(f"Step {step}, gradient {name}") from error
                parameter.grad = parameter.main_grad.to(parameter.dtype).clone()
            adapter_before = block.loop_adapter.weight.detach().clone()
            optimizer.step()
            reference_optimizer.step()
            assert torch.any(block.loop_adapter.weight != adapter_before)
            for name, parameter in block.named_parameters():
                compare(parameter, native.w(name), dtype)
    finally:
        delete_cuda_graphs()
        initialize_rng_tracker(force_reset=True)
