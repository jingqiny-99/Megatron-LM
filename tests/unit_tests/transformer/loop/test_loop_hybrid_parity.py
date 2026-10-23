# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""H.5 parity gate: LoopHybridStack against the GPT loop (LoopTransformerBlock).

A dense GPT loop of ``P_g + C_g + K_g`` layers equals a hybrid loop over ``('*-')^n`` with
roles ``2 P_g`` / ``2 K_g`` under the weight map ``gpt_to_hybrid_key`` (GPT layer ``i`` is
hybrid sublayers ``2i`` and ``2i + 1``). The gate: activations bitwise (output and input
grad), parameter grads at ``rtol=1e-5``, FLOP-per-token metrics equal, and token-layer metrics
exactly twice the GPT values (the hybrid stack counts sublayers).

* Local class (device-agnostic, CPU locally): local layers with SDPA core attention, fp32,
  Huginn (depths, truncated BPTT, recompute) and masked MoR.
* TE class (CUDA): TE specs, ``loop_hybrid_stack_spec``, final norm, fp32 and bf16, Huginn and
  MoR (masked and compact, which needs THD attention).
* Empty compact-MoR rounds (CUDA): a zeroed router sends every token to depth 1, so later
  rounds have no rows; attention / dense MLP are skipped there, MoE MLPs are not.

The EP = 2 case is in ``test_loop_hybrid_parity_ep.py``; the helpers in
``hybrid_parity_utils.py``.
"""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_observation import capture_tensor_observations
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.loop_transformer_block import LoopTransformerBlock
from tests.unit_tests.transformer.loop import hybrid_parity_utils
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    BATCH,
    CUDA,
    SEQ,
    copy_gpt_weights,
    dense_config,
    device,
    gpt_to_hybrid_key,
    local_gpt_layer_spec,
    local_hybrid_submodules,
    model_parallel_once,
    rope,
)
from tests.unit_tests.transformer.loop.hybrid_parity_utils import (
    NUM_GPT_LAYERS,
    PATTERN,
    assert_parity,
    loop_fields,
    make_gpt_config,
    make_hybrid_config,
    make_inputs,
    run_and_collect,
    te_base,
    te_pair,
    te_rope,
)

# The TE comparison fixture, made visible to pytest in this module.
full_precision_gemms = hybrid_parity_utils.full_precision_gemms

gpu = pytest.mark.skipif(not CUDA, reason="TE specs and THD attention need CUDA")


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


# ---------------------------------------------------------------------------
# Local layers (CPU locally, CUDA in CI)
# ---------------------------------------------------------------------------


def _local_pair(loop):
    base = dense_config(NUM_GPT_LAYERS)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    torch.manual_seed(1234)
    gpt = LoopTransformerBlock(
        config=make_gpt_config(base, loop),
        spec=local_gpt_layer_spec(),
        pre_process=True,
        post_process=True,
        post_layer_norm=False,
        pg_collection=pg_collection,
    ).to(device())
    hybrid_config = make_hybrid_config(base, loop)
    hybrid = LoopHybridStack(
        config=hybrid_config,
        submodules=local_hybrid_submodules(),
        layer_config_list=validate_segment_layers(PATTERN, hybrid_config),
        post_layer_norm=False,
        pg_collection=pg_collection,
    ).to(device())
    copy_gpt_weights(gpt, hybrid)
    return gpt, hybrid


@pytest.mark.parametrize("depth", [1, 2, 4])
@pytest.mark.parametrize("bptt,recompute", [(0, False), (2, False), (0, True)])
def test_local_huginn_parity(depth, bptt, recompute):
    gpt, hybrid = _local_pair(loop_fields("huginn", depth, bptt, recompute, "compact"))
    x, upstream = make_inputs(torch.float32)
    metrics = assert_parity(gpt, hybrid, x, rope(), upstream)
    assert metrics["loop depth"] == depth
    if recompute:
        assert metrics["loop recompute flops per token"] > 0


@pytest.mark.parametrize("recompute", [False, True])
def test_local_masked_mor_parity(recompute):
    gpt, hybrid = _local_pair(loop_fields("mor", 4, 0, recompute, "masked"))
    x, upstream = make_inputs(torch.float32)
    metrics = assert_parity(gpt, hybrid, x, rope(), upstream)
    assert set(f"depth fraction {index}" for index in range(1, 5)) <= set(metrics)


def test_local_masked_mor_parity_with_padding():
    """Padded tokens (``padding_mask`` True) are excluded from the router statistics."""
    gpt, hybrid = _local_pair(loop_fields("mor", 4, 0, False, "masked"))
    x, upstream = make_inputs(torch.float32)
    padding_mask = torch.zeros(BATCH, SEQ, dtype=torch.bool, device=device())
    padding_mask[0, SEQ - 5 :] = True
    metrics = assert_parity(gpt, hybrid, x, rope(), upstream, padding_mask=padding_mask)
    assert metrics["loop active token layers"] <= (BATCH * SEQ - 5) * 4 * 4


def test_weight_map_covers_both_stacks():
    """Every GPT key maps to a distinct hybrid key, and the loop parameters keep their names."""
    gpt, hybrid = _local_pair(loop_fields("huginn", 2, 0, False, "compact"))
    mapped = [gpt_to_hybrid_key(key) for key in gpt.state_dict()]
    assert len(mapped) == len(set(mapped)) == len(hybrid.state_dict())
    assert {"loop_adapter.weight", "loop_norm.weight"} <= set(mapped)


# ---------------------------------------------------------------------------
# TE layers (CUDA)
# ---------------------------------------------------------------------------


@gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_te_huginn_parity(full_precision_gemms, dtype, depth):
    base = te_base(dtype)
    gpt, hybrid = te_pair(base, loop_fields("huginn", depth, 0, False, "compact"))
    # BF16 inputs at the embedding scale (see test_loop_transformer.check_native).
    scale = base.init_method_std if dtype == torch.bfloat16 else 1.0
    x, upstream = make_inputs(dtype, hidden=base.hidden_size, scale=scale)
    assert_parity(gpt, hybrid, x, te_rope(base.kv_channels), upstream)


@gpu
@pytest.mark.parametrize(
    "execution,dtype",
    [("masked", torch.float32), ("masked", torch.bfloat16), ("compact", torch.bfloat16)],
)
@pytest.mark.parametrize("recompute", [False, True])
def test_te_mor_parity(full_precision_gemms, execution, dtype, recompute):
    base = te_base(dtype)
    gpt, hybrid = te_pair(base, loop_fields("mor", 4, 0, recompute, execution))
    scale = base.init_method_std if dtype == torch.bfloat16 else 1.0
    x, upstream = make_inputs(dtype, hidden=base.hidden_size, scale=scale, seed=11)
    # Compact MoR's FLOPs carry an FP32 tensor sum of squared lengths, accumulated in a
    # different order on the two stacks.
    assert_parity(
        gpt,
        hybrid,
        x,
        te_rope(base.kv_channels),
        upstream,
        flop_rtol=1e-6 if execution == "compact" else 1e-12,
    )


def _record_calls(hybrid):
    """Count forward calls of each sublayer's attention and MLP / MoE module."""
    calls = {}

    def hook(name):
        def record(module, args, kwargs):
            calls[name] = calls.get(name, 0) + 1

        return record

    handles = []
    for index, (symbol, layer) in enumerate(zip(hybrid._loop_symbols, hybrid.layers)):
        if symbol == "*":
            module, name = layer.self_attention, f"{index}.attention"
        else:
            module, name = layer.mlp, f"{index}.mlp"
        handles.append(module.register_forward_pre_hook(hook(name), with_kwargs=True))
    return calls, handles


@gpu
@pytest.mark.parametrize("moe_core", [False, True], ids=["dense", "moe"])
def test_te_empty_compact_rounds(full_precision_gemms, moe_core):
    """Zero router logits pick depth 1 for every token (first-occurrence argmax), so compact
    rounds 2..R have no rows: attention and dense MLPs are not called there, MoE MLPs are, and
    every core sublayer is still observed once per round."""
    from tests.unit_tests.transformer.test_loop_transformer import compare

    overrides = {}
    if moe_core:
        overrides = dict(
            num_moe_experts=4,
            moe_router_topk=2,
            moe_ffn_hidden_size=256,
            moe_layer_freq=[0, 1, 1, 0],
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.0,
        )
    base = te_base(torch.bfloat16, **overrides)
    depth = 3
    x, upstream = make_inputs(torch.bfloat16, hidden=base.hidden_size, scale=0.02, seed=13)
    results = {}
    for execution in ("masked", "compact"):
        gpt, hybrid = te_pair(base, loop_fields("mor", depth, 0, False, execution))
        with torch.no_grad():
            gpt.depth_router[2].weight.zero_()
            hybrid.depth_router[2].weight.zero_()
        # Each stack equals its GPT-loop counterpart.
        metrics = assert_parity(
            gpt,
            hybrid,
            x,
            te_rope(base.kv_channels),
            upstream,
            flop_rtol=None if moe_core else (1e-6 if execution == "compact" else 1e-12),
        )
        assert metrics["loop depth"] == 1

        # One more hybrid pass, recording layer calls and residual observations.
        calls, handles = _record_calls(hybrid)
        seen = []

        def observer(owner, name, kind, tensor, *dims):
            seen.append(owner.layer_number)

        try:
            with capture_tensor_observations(observer, frozenset({"residual_accumulator"})):
                results[execution] = run_and_collect(hybrid, x, te_rope(base.kv_channels), upstream)
        finally:
            for handle in handles:
                handle.remove()
        roles = hybrid.loop_roles
        core = range(roles.core_start, roles.core_end)
        numbers = [layer.layer_number for layer in hybrid.layers]
        assert seen == (
            numbers[: roles.core_start]
            + [numbers[index] for _ in range(depth) for index in core]
            + numbers[roles.core_end :]
        )
        for index in core:
            symbol = hybrid._loop_symbols[index]
            name = f"{index}.attention" if symbol == "*" else f"{index}.mlp"
            if execution == "compact" and symbol != "E":
                expected_calls = 1  # round 1 only; rounds 2..R are empty
            else:
                expected_calls = depth  # masked runs every row; MoE MLPs run on empty rounds
            assert calls.get(name) == expected_calls, (execution, name, calls)
    # Compact equals masked at the GPT loop's BF16 tolerance.
    compare(results["compact"][0], results["masked"][0], torch.bfloat16)
    compare(results["compact"][1], results["masked"][1], torch.bfloat16)
