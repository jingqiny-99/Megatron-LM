# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the hybrid loop's per-sublayer FLOP table (megatron/core/transformer/loop/flops.py).

Pure arithmetic on configs; no process groups. The rows are pinned against hand formulas, the
DSv4 coefficients against the training FLOP counter they copy, and the MoE row against the GPT
loop's ``_record_work`` (which also counts the router GEMM).
"""

from types import SimpleNamespace

import pytest

from megatron.core.transformer.loop import flops
from megatron.core.transformer.loop_transformer_block import LoopTransformerBlock

T, Q, S = 96, 96 * 48, 48
H = 64


def _config(**overrides):
    fields = dict(
        hidden_size=H,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=16,
        attention_output_gate=False,
        gated_attention_proj_granularity="elementwise",
        ffn_hidden_size=256,
        gated_linear_unit=True,
        moe_ffn_hidden_size=32,
        moe_router_topk=2,
        moe_shared_expert_intermediate_size=None,
        moe_latent_size=None,
        num_moe_experts=8,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


_DSV4 = dict(
    hidden_size=512,
    num_attention_heads=8,
    v_head_dim=16,
    q_lora_rank=192,
    o_groups=4,
    o_lora_rank=64,
    csa_window_size=32,
    dsa_indexer_n_heads=8,
    dsa_indexer_head_dim=64,
    dsa_indexer_topk=32,
)


def test_attention_row():
    q, kv = 4 * 16, 2 * 16
    config = _config()
    assert flops.sublayer_forward_flops("*", config, T, Q, S) == 4 * T * H * (q + kv) + 2 * Q * q
    assert (
        flops.sublayer_forward_flops("*", config, T, Q, S, arbitrary_mask=True)
        == 4 * T * H * (q + kv) + 4 * Q * q
    )
    gated = _config(attention_output_gate=True)
    assert flops.sublayer_forward_flops("*", gated, T, Q, S) == (
        4 * T * H * (q + kv) + 2 * T * H * q + 2 * Q * q
    )
    headwise = _config(attention_output_gate=True, gated_attention_proj_granularity="headwise")
    assert flops.sublayer_forward_flops("*", headwise, T, Q, S) == (
        4 * T * H * (q + kv) + 2 * T * H * 4 + 2 * Q * q
    )


def test_mlp_row():
    assert flops.sublayer_forward_flops("-", _config(), T, Q, S) == 6 * T * H * 256
    assert flops.sublayer_forward_flops("-", _config(gated_linear_unit=False), T, Q, S) == (
        4 * T * H * 256
    )


def test_moe_rows():
    assert flops.sublayer_forward_flops("E", _config(), T, Q, S) == 6 * T * H * (32 * 2)
    shared = _config(moe_shared_expert_intermediate_size=48)
    assert flops.sublayer_forward_flops("E", shared, T, Q, S) == 6 * T * H * (32 * 2 + 48)
    latent = _config(moe_latent_size=16, moe_shared_expert_intermediate_size=48)
    assert flops.sublayer_forward_flops("E", latent, T, Q, S) == (
        6 * T * 16 * 32 * 2 + 4 * T * H * 16 + 6 * T * H * 48
    )
    fallback = _config(moe_ffn_hidden_size=None)
    assert flops.sublayer_forward_flops("E", fallback, T, Q, S) == 6 * T * H * (256 * 2)


def test_moe_row_is_the_gpt_loop_moe_term_without_the_router_gemm():
    """The GPT loop counts the router GEMM (2 T h E); the hybrid table, like the training
    FLOP counter, does not. Pinned so neither side changes silently."""
    config = _config(moe_shared_expert_intermediate_size=48)
    gpt = SimpleNamespace(
        config=config, layers=[SimpleNamespace(is_moe_layer=True)], _work=[0.0, 0.0, 0.0]
    )
    LoopTransformerBlock._record_work(gpt, T, Q, 0, 1, backward=False)
    hybrid = flops.sublayer_forward_flops("*", config, T, Q, S) + flops.sublayer_forward_flops(
        "E", config, T, Q, S
    )
    assert gpt._work[0] == hybrid + 2 * T * H * config.num_moe_experts


def test_dense_rows_equal_the_gpt_loop_layer():
    config = _config()
    gpt = SimpleNamespace(
        config=config, layers=[SimpleNamespace(is_moe_layer=False)], _work=[0.0, 0.0, 0.0]
    )
    LoopTransformerBlock._record_work(gpt, T, Q, 0, 1, backward=False, arbitrary_mask=True)
    hybrid = flops.sublayer_forward_flops(
        "*", config, T, Q, S, arbitrary_mask=True
    ) + flops.sublayer_forward_flops("-", config, T, Q, S)
    assert gpt._work[0] == hybrid


@pytest.mark.parametrize("seq_length", [64, 4096])
@pytest.mark.parametrize("counts", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 3, 1)])
def test_dsv4_coefficients_equal_the_training_counter(seq_length, counts):
    training = pytest.importorskip("megatron.training.training")
    kwargs = dict(
        _DSV4,
        seq_length=seq_length,
        n_layers_r0=counts[0],
        n_layers_r4=counts[1],
        n_layers_r128=counts[2],
    )
    assert flops.dsv4_hybrid_self_attention_coefficients(
        **kwargs
    ) == training._dsv4_hybrid_self_attention_flops(**kwargs)


@pytest.mark.parametrize("symbol,ratio", [("W", 0), ("C", 4), ("H", 128)])
def test_dsv4_rows(symbol, ratio):
    config = SimpleNamespace(**_DSV4)
    token_linear, core = flops.dsv4_hybrid_self_attention_coefficients(
        **_DSV4,
        seq_length=S,
        n_layers_r0=int(ratio == 0),
        n_layers_r4=int(ratio == 4),
        n_layers_r128=int(ratio == 128),
    )
    expected = 2 * (token_linear * T + core * Q)
    assert flops.sublayer_forward_flops(symbol, config, T, Q, S) == expected
    # 'D' (dsv4_hybrid DSA) takes its ratio from csa_compress_ratios.
    assert flops.sublayer_forward_flops("D", config, T, Q, S, compress_ratio=ratio) == expected


def test_dsv4_rows_with_a_real_config():
    """The fields the DSv4 row reads exist on MLATransformerConfig (o_groups defaults)."""
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    config = MLATransformerConfig(
        num_layers=2,
        hidden_size=512,
        num_attention_heads=8,
        multi_latent_attention=True,
        q_lora_rank=192,
        kv_lora_rank=64,
        qk_head_dim=16,
        qk_pos_emb_head_dim=8,
        v_head_dim=16,
        experimental_attention_variant="dsv4_hybrid",
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=32,
        csa_window_size=32,
        csa_compress_ratios=[4, 128],
        use_cpu_initialization=True,
    )
    assert flops.sublayer_forward_flops("C", config, T, Q, S) > 0


def test_rows_reject_unknown_symbols_and_ratios():
    with pytest.raises(ValueError, match="no row for sublayer 'M'"):
        flops.sublayer_forward_flops("M", _config(), T, Q, S)
    with pytest.raises(ValueError, match="needs its csa_compress_ratios entry"):
        flops.sublayer_forward_flops("D", SimpleNamespace(**_DSV4), T, Q, S)
    with pytest.raises(ValueError, match="compress ratio 8"):
        flops.sublayer_forward_flops("D", SimpleNamespace(**_DSV4), T, Q, S, compress_ratio=8)


def test_loop_and_head_rows():
    assert flops.huginn_adapter_forward_flops(T, H) == 4 * T * H**2
    assert flops.huginn_adapter_forward_flops(T, H, 4) == 4 * 4 * T * H**2
    assert flops.mor_router_forward_flops(T, H, 4) == 2 * T * H * (H + 4)
    assert flops.logits_forward_flops(T, H, 1000) == 2 * T * H * 1000
    assert flops.mtp_depth_extra_forward_flops(T, H) == 2 * (3 * H + 2 * H * H) * T
