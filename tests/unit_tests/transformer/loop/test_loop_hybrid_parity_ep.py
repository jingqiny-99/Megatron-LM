# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""H.5 parity gate, EP = 2: LoopHybridStack against the GPT loop with an empty EP rank.

Its own module so that model parallel (with ``expert_model_parallel_size=2``) is initialized
once per module, not re-initialized inside the tests (the CI process-group budget). The
helpers are in ``hybrid_parity_utils.py``.
"""

import os

import pytest
import torch

from tests.unit_tests.transformer.loop import hybrid_parity_utils
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA, model_parallel_once
from tests.unit_tests.transformer.loop.hybrid_parity_utils import (
    assert_parity,
    loop_fields,
    make_inputs,
    te_base,
    te_pair,
    te_rope,
)

pytestmark = pytest.mark.skipif(
    not CUDA or int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="EP = 2 needs CUDA and at least 2 ranks",
)

# The TE comparison fixture, made visible to pytest in this module.
full_precision_gemms = hybrid_parity_utils.full_precision_gemms


@pytest.fixture(scope="module", autouse=True)
def _expert_parallel():
    with model_parallel_once(expert_model_parallel_size=2):
        yield


@pytest.mark.parametrize("recompute", [False, True])
def test_te_moe_collective_order_with_an_empty_ep_rank(full_precision_gemms, recompute):
    """EP = 2, alltoall, compact MoR: one EP rank's tokens all exit after round 1 (prescribed
    router) while its peer runs round 2, so the first rank runs round 2's MoE dispatch /
    combine on an empty input. No hang, and equal to the GPT loop on every rank."""
    from tests.unit_tests.transformer.test_loop_transformer import PrescribedDepthRouter

    base = te_base(
        torch.bfloat16,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=256,
        moe_layer_freq=[0, 1, 1, 0],
        expert_model_parallel_size=2,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_aux_loss_coeff=0.01,
    )
    gpt, hybrid = te_pair(base, loop_fields("mor", 2, 0, recompute, "compact"))
    depth = 1 if torch.distributed.get_rank(gpt.pg_collection.ep) == 0 else 2
    gpt.depth_router = PrescribedDepthRouter(gpt.depth_router, depth)
    hybrid.depth_router = PrescribedDepthRouter(hybrid.depth_router, depth)
    x, upstream = make_inputs(
        torch.bfloat16,
        hidden=base.hidden_size,
        batch=1,
        scale=0.02,
        seed=17 + torch.distributed.get_rank(),
    )
    assert_parity(gpt, hybrid, x, te_rope(base.kv_channels), upstream, flop_rtol=None)
