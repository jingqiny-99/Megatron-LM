# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``DSv4HybridAttention`` under a scoped ``mla_up_proj`` recompute request (D42).

Inside ``request_selective_recompute({"mla_up_proj"})`` the attention checkpoints its query /
key-value up projections and RoPE (``qkv_up_proj_and_rope_apply``) exactly as the config-level
``recompute_granularity='selective'`` with ``recompute_modules=['mla_up_proj']`` does: the
first pass keeps no graph, its outputs are released after the core attention, and the
backward replays it. The forward output and every gradient equal the plain attention's
bitwise (window-only ratio 0 and CSA ratio 4 layers, indexer loss on); outside a request the
attention runs the dev path. Bitwise needs deterministic kernels: without them the plain
attention's backward already differs from itself run to run.
"""

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.selective_recompute import request_selective_recompute
from tests.unit_tests.transformer.loop import test_loop_rsbf_executor as loop_tests
from tests.unit_tests.transformer.loop.hybrid_loop_utils import model_parallel_once

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available"),
]

_SEED = 42

# Deterministic kernels (``NVTE_ALLOW_NONDETERMINISTIC_ALGO=0``, torch deterministic algorithms).
deterministic_algorithms = loop_tests.deterministic_algorithms


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


def _attention(layer_number, **config_fields):
    from tests.unit_tests.transformer.experimental_attention_variant import (
        test_dsv4_hybrid_attention as dsv4_tests,
    )

    config = dsv4_tests._make_config(dsa_indexer_loss_coeff=1.0, **config_fields)
    torch.manual_seed(_SEED)
    model_parallel_cuda_manual_seed(_SEED)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    attention = dsv4_tests._build_attention(config, layer_number=layer_number, pg_collection=pg)
    attention = attention.cuda()
    attention.train()
    return attention


def _step(attention, hidden, grad, scope=False):
    """Forward and backward; the output, the input gradient, the parameter gradients and the
    number of up-projection checkpoints the call recorded."""
    calls = []
    real = tensor_parallel.CheckpointWithoutOutput.checkpoint

    def counting(checkpoint, run_function, *args):
        calls.append(run_function.__name__)
        return real(checkpoint, run_function, *args)

    attention.zero_grad(set_to_none=True)
    hidden = hidden.detach().clone().requires_grad_()
    tensor_parallel.CheckpointWithoutOutput.checkpoint = counting
    try:
        with request_selective_recompute({"mla_up_proj"} if scope else ()):
            output, _ = attention(hidden_states=hidden, attention_mask=None)
    finally:
        tensor_parallel.CheckpointWithoutOutput.checkpoint = real
    # The checkpoint is consumed by the forward (released outputs, a backward hook).
    assert attention.qkv_up_checkpoint is None
    output.backward(grad)
    grads = {
        name: param.grad.detach().clone()
        for name, param in attention.named_parameters()
        if param.grad is not None
    }
    checkpoints = calls.count("qkv_up_proj_and_rope_apply")
    return output.detach(), hidden.grad.detach(), grads, checkpoints


@pytest.mark.parametrize("layer_number", [1, 2], ids=["window", "csa4"])
def test_scoped_up_proj_recompute_changes_no_value(layer_number, deterministic_algorithms):
    """Plain (no request), scoped request and the config switch: the same output, input and
    parameter gradients bitwise; only the request and the switch checkpoint the up
    projection; a request ends with its scope."""
    seq, batch = 256, 2
    with loop_tests._hadamard_fallback():
        plain = _attention(layer_number)
        switched = _attention(
            layer_number, recompute_granularity="selective", recompute_modules=["mla_up_proj"]
        )
        assert not plain.recompute_up_proj and switched.recompute_up_proj
        generator = torch.Generator(device="cuda").manual_seed(3)
        hidden = torch.randn(
            seq, batch, plain.config.hidden_size, generator=generator, device="cuda"
        ).to(torch.bfloat16)
        grad = torch.randn(
            seq, batch, plain.config.hidden_size, generator=generator, device="cuda"
        ).to(torch.bfloat16)
        expected = _step(plain, hidden, grad)
        scoped = _step(plain, hidden, grad, scope=True)
        config_level = _step(switched, hidden, grad)
        after = _step(plain, hidden, grad)
    assert expected[3] == 0 and after[3] == 0
    assert scoped[3] == 1 and config_level[3] == 1
    for actual in (scoped, config_level, after):
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        assert actual[2].keys() == expected[2].keys()
        for name, value in expected[2].items():
            assert torch.equal(actual[2][name], value), name
