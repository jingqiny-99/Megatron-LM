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


def test_dsv4_up_proj_row_is_part_of_the_attention_row():
    """D42's coda recompute FLOPs: the query up projection and the key-value projection of one
    DSv4 attention, the weight terms the attention row counts for them (its ``q_term`` without
    the down projection and the norm, its ``kv_term`` without the norm)."""
    c = SimpleNamespace(**_DSV4)
    up = flops.dsv4_up_proj_forward_flops(c, T)
    weights = c.q_lora_rank * c.num_attention_heads * c.v_head_dim + c.hidden_size * c.v_head_dim
    assert up == 2 * T * weights
    token_linear, _ = flops.dsv4_hybrid_self_attention_coefficients(
        **_DSV4, seq_length=S, n_layers_r0=1, n_layers_r4=0, n_layers_r128=0
    )
    q_term = c.q_lora_rank * (c.hidden_size + c.num_attention_heads * c.v_head_dim + 1)
    kv_term = c.hidden_size * c.v_head_dim + c.v_head_dim
    assert weights == q_term + kv_term - c.q_lora_rank * (c.hidden_size + 1) - c.v_head_dim
    assert up < 2 * token_linear * T


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


@pytest.mark.parametrize(
    "reuse_kv,reuse_indexer", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("block_checkpoint", [False, True])
def test_four_round_work_counts_only_issued_sparse_producers(
    reuse_kv, reuse_indexer, block_checkpoint
):
    import torch
    from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack

    cfg = _config(
        **_DSV4,
        loop_reuse_sparse_kv=reuse_kv,
        loop_reuse_indexer=reuse_indexer,
        loop_iterations=4,
        loop_sample_depth=False,
        loop_bptt_iterations=0,
        loop_carrier="residual",
        loop_untie_core=False,
        loop_recompute=block_checkpoint,
        loop_recompute_block_stride=1 if block_checkpoint else None,
    )
    # Exercise the actual round dispatcher, block checkpoints and work recorder without GPU kernels.
    stack = LoopHybridStack.__new__(LoopHybridStack)
    torch.nn.Module.__init__(stack)
    stack.config = cfg
    stack._round_core = 2
    stack.loop_roles = SimpleNamespace(core_start=1, core_end=3)
    stack._loop_flop_rows = [("C", cfg, None), ("D", cfg, 4), ("H", cfg, None)]
    stack._cross_loop_attention_layers = {1: (4, True), 2: (128, False)}
    stack._run_range = lambda x, *args: x.sin()
    stack._diagnostics_active = lambda: False
    stack._work = [0, 0, 0]
    seq, batch = 256, 2
    tok, sq = seq * batch, seq**2 * batch
    embedding = torch.randn(seq, batch, 1, requires_grad=True)
    stack._record_range(0, 1, tok, sq, seq)
    stack._huginn(embedding, None, "work").sum().backward()

    csa = flops.sublayer_forward_flops("C", cfg, tok, sq, seq)
    hca = flops.sublayer_forward_flops("H", cfg, tok, sq, seq)
    # Input-to-value and input-to-gate projections; the overlap compressor doubles its width.
    compressors = 2 * tok * cfg.hidden_size * cfg.v_head_dim * (4 + 2)
    index_kv = 4 * cfg.hidden_size * cfg.dsa_indexer_head_dim
    index_q = cfg.q_lora_rank * cfg.dsa_indexer_n_heads * cfg.dsa_indexer_head_dim
    index_weights = cfg.hidden_size * cfg.dsa_indexer_n_heads
    score = sq / 4 * cfg.dsa_indexer_n_heads * cfg.dsa_indexer_head_dim
    indexer = 2 * (tok * (index_kv + index_q + index_weights) + score)
    saved = reuse_kv * compressors + (reuse_kv or reuse_indexer) * indexer
    core = 4 * (csa + hca) - 3 * saved
    assert stack._work == [csa + core, 2 * (csa + core), core if block_checkpoint else 0]


def test_nonindexed_ratio4_consumer_keeps_attention_to_all_compressed_rows():
    cfg = SimpleNamespace(**_DSV4)
    seq = 1024
    window = flops.dsv4_attention_forward_flops(cfg, 0, seq, seq**2, seq)
    actual = flops.dsv4_attention_forward_flops(
        cfg, 4, seq, seq**2, seq, project_compressed_kv=False, has_indexer=False, run_indexer=False
    )
    avg_causal_rows = seq / (2 * 4)
    assert actual == window + 4 * seq * avg_causal_rows * cfg.num_attention_heads * cfg.v_head_dim


@pytest.mark.parametrize("source", [True, False])
@pytest.mark.parametrize("cp_size", [1, 2, 16])
def test_glm_dsa_cp_counts_global_pairs_for_local_query_rows(cp_size, source):
    from megatron.core.transformer.loop.glm_dsa import GLMDSALoopStack

    sequence, batch = 64, 2
    local_sequence = sequence // cp_size
    cfg = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        q_lora_rank=3,
        kv_lora_rank=4,
        qk_pos_emb_head_dim=2,
        qk_head_dim=3,
        v_head_dim=4,
        dsa_indexer_topk=16,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=3,
        loop_sparse_kv_window_size=6,
        context_parallel_size=cp_size,
    )
    attention = SimpleNamespace(core_attention=SimpleNamespace(indexer=object()))
    from torch import nn

    stack = GLMDSALoopStack.__new__(GLMDSALoopStack)
    nn.Module.__init__(stack)
    stack.config = cfg
    stack._loop_flop_rows = [("D", cfg, None)]
    stack.layers = [SimpleNamespace(self_attention=attention)]
    flops_per_rank = GLMDSALoopStack._layer_forward_flops(
        stack,
        0,
        batch * local_sequence,
        batch * local_sequence**2,
        local_sequence,
        False,
        SimpleNamespace(is_source=source),
    )
    projections = 8 * 3 + 3 * 2 * (3 + 2) + 8 * (4 + 2) + 2 * (3 * 4 + 4 * 4 + 4 * 8)
    expected_macs = 0
    for position in range(1, sequence + 1):
        pairs = min(position, 16) + (0 if source else min(position, 6))
        expected_macs += projections + 2 * (4 + 2 + 4) * pairs
        if source:
            expected_macs += 3 * 2 * 3 + 8 * 3 + 8 * 2 + 2 * 3 * position
    assert cp_size * flops_per_rank == 2 * batch * expected_macs


def test_e2e_tv_teacher_projects_once_without_weight_or_input_gradient():
    from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack

    vocab = 256
    cfg = _config(mtp_loss_type="e2e_tv", mtp_detach_heads=True)
    stack = SimpleNamespace(config=cfg, _loop_head=(vocab, "", 7))
    terms = LoopHybridStack.head_work_terms(stack, T, Q, S)
    logits = 2 * T * H * vocab
    assert terms.count((logits, 0)) == 1
    cfg.mtp_loss_type = "cross_entropy"
    assert (logits, 0) not in LoopHybridStack.head_work_terms(stack, T, Q, S)
