# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""H.5 parity helpers: GPT-loop / hybrid-loop pairs under one weight map, and the gate.

Shared by ``test_loop_hybrid_parity.py`` and ``test_loop_hybrid_parity_ep.py`` (the EP = 2
case lives in its own module so each module initializes model parallel once).
"""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.loop_transformer_block import LoopTransformerBlock
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    BATCH,
    HIDDEN,
    SEQ,
    copy_gpt_weights,
    device,
    gpt_to_hybrid_key,
)

# GPT roles (layers) and the hybrid roles (sublayers) they map to.
P_G, C_G, K_G = 1, 2, 1
NUM_GPT_LAYERS = P_G + C_G + K_G
PATTERN = "*-" * NUM_GPT_LAYERS

_FLOP_KEYS = tuple(
    f"loop {phase} flops per token" for phase in ("forward", "backward", "recompute")
)
_TOKEN_LAYER_KEYS = ("loop active token layers", "loop backward token layers")


def loop_fields(architecture, depth, bptt, recompute, execution):
    """The loop fields shared by the GPT and hybrid configs of one pair."""
    return dict(
        loop_architecture=architecture,
        loop_iterations=depth,
        loop_bptt_iterations=bptt,
        loop_recompute=recompute,
        loop_token_execution=execution,
    )


def make_gpt_config(base, loop):
    """The GPT loop config: ``P_G`` / ``C_G`` / ``K_G`` layers."""
    return replace(
        base,
        num_layers=NUM_GPT_LAYERS,
        loop_prelude_layers=P_G,
        loop_core_layers=C_G,
        loop_prelude_sublayers=None,
        loop_coda_sublayers=None,
        is_hybrid_model=False,
        **loop,
    )


def make_hybrid_config(base, loop):
    """The hybrid loop config over ``PATTERN``: roles in sublayers (two per GPT layer)."""
    return replace(
        base,
        num_layers=len(PATTERN),
        loop_prelude_sublayers=2 * P_G,
        loop_coda_sublayers=2 * K_G,
        is_hybrid_model=True,
        **loop,
    )


def run_and_collect(module, x, rotary, upstream, padding_mask=None):
    """Forward + backward of ``sum(y * upstream) + loop losses``; returns detached results."""
    # The process-global MoE metrics tracker sizes each entry by the num_layers of its first
    # recorder; the GPT loop (num_layers = GPT layers) and the hybrid loop (sublayers) differ.
    set_moe_metrics_tracker(MoEMetricsTracker())
    x = x.detach().clone().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    y = module(x, attention_mask=None, rotary_pos_emb=rotary, padding_mask=padding_mask)
    ((y * upstream).float().sum() + sum(module.loop_losses.values())).backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in module.named_parameters()
        if param.grad is not None
    }
    metrics = {name: value.detach().clone() for name, value in module.loop_metrics.items()}
    return y.detach().clone(), x.grad.detach().clone(), grads, metrics


def _assert_metrics(hybrid, gpt, flop_rtol):
    assert hybrid.keys() == gpt.keys()
    for name, expected in gpt.items():
        actual = hybrid[name]
        if name in _FLOP_KEYS:
            # flop_rtol=None: MoE patterns, where the two loops' FLOP conventions differ by
            # the router GEMM (test_loop_flops.py pins the difference).
            if flop_rtol is not None:
                torch.testing.assert_close(actual, expected, rtol=flop_rtol, atol=0, msg=name)
        elif name in _TOKEN_LAYER_KEYS:
            # The hybrid stack counts sublayers: one GPT layer is '*' + '-'.
            torch.testing.assert_close(actual, 2 * expected, rtol=0, atol=0, msg=name)
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)


def assert_parity(gpt, hybrid, x, rotary, upstream, *, flop_rtol=1e-12, padding_mask=None):
    """The H.5 gate, with a self-determinism diagnosis when activations differ."""
    y_g, dx_g, grads_g, metrics_g = run_and_collect(gpt, x, rotary, upstream, padding_mask)
    y_h, dx_h, grads_h, metrics_h = run_and_collect(hybrid, x, rotary, upstream, padding_mask)
    for label, actual, expected in (("output", y_h, y_g), ("input grad", dx_h, dx_g)):
        if not torch.equal(actual, expected):
            rerun = run_and_collect(gpt, x, rotary, upstream, padding_mask)
            if not (torch.equal(rerun[0], y_g) and torch.equal(rerun[1], dx_g)):
                pytest.fail(f"NONDETERMINISTIC: the GPT loop does not reproduce its {label}")
            max_diff = (actual.float() - expected.float()).abs().max().item()
            pytest.fail(f"{label} differs from the GPT loop (max |diff| {max_diff})")
    assert {gpt_to_hybrid_key(name) for name in grads_g} == set(grads_h)
    for name, expected in grads_g.items():
        torch.testing.assert_close(
            grads_h[gpt_to_hybrid_key(name)], expected, rtol=1e-5, atol=0, msg=name
        )
    _assert_metrics(metrics_h, metrics_g, flop_rtol)
    return metrics_h


def make_inputs(dtype, *, hidden=HIDDEN, batch=BATCH, scale=1.0, seed=7):
    """``[S, B, h]`` inputs and an upstream gradient, seeded on CPU."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(SEQ, batch, hidden, generator=generator) * scale
    upstream = torch.randn(SEQ, batch, hidden, generator=generator)
    return x.to(device=device(), dtype=dtype), upstream.to(device=device(), dtype=dtype)


@pytest.fixture
def full_precision_gemms(monkeypatch):
    """Compare full-precision GEMMs (no TF32), deterministic attention, as the GPT tests do."""
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old_tf32


def te_base(dtype, **overrides):
    """The GPT loop test's small config (test_loop_transformer.make_config), loop fields unset."""
    fields = dict(
        num_layers=NUM_GPT_LAYERS,
        hidden_size=128,
        num_attention_heads=2,
        num_query_groups=1,
        kv_channels=64,
        ffn_hidden_size=256,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        apply_rope_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        attention_backend=AttnBackend.auto if dtype == torch.bfloat16 else AttnBackend.unfused,
    )
    fields.update(overrides)
    return TransformerConfig(**fields)


def te_pair(base, loop):
    """A TE GPT loop and a ``loop_hybrid_stack_spec`` stack with the GPT weights copied."""
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_decoder_block_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_stack_spec

    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    gpt_config = make_gpt_config(base, loop)
    gpt_spec = (
        get_gpt_decoder_block_spec(gpt_config, use_transformer_engine=True)
        if gpt_config.num_moe_experts
        else get_gpt_layer_with_transformer_engine_spec()
    )
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    gpt = LoopTransformerBlock(
        config=gpt_config,
        spec=gpt_spec,
        pre_process=True,
        post_process=True,
        pg_collection=pg_collection,
    ).cuda()
    hybrid_config = make_hybrid_config(base, loop)
    pattern = _hybrid_pattern(gpt_config)
    hybrid = build_module(
        loop_hybrid_stack_spec,
        config=hybrid_config,
        pre_process=True,
        layer_config_list=validate_segment_layers(pattern, hybrid_config),
        post_process=True,
        pg_collection=pg_collection,
    ).cuda()
    assert type(hybrid) is LoopHybridStack
    copy_gpt_weights(gpt, hybrid)
    return gpt, hybrid


def _hybrid_pattern(gpt_config):
    """'*-' per dense GPT layer, '*E' per MoE GPT layer (moe_layer_freq list)."""
    freq = gpt_config.moe_layer_freq if gpt_config.num_moe_experts else 0
    moe = freq if isinstance(freq, list) else [0] * NUM_GPT_LAYERS
    return "".join("*E" if is_moe else "*-" for is_moe in moe)


def te_rope(kv_channels):
    """Rotary embeddings for ``SEQ`` on CUDA."""
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    return RotaryEmbedding(kv_channels, rotary_percent=1.0)(SEQ).cuda()
