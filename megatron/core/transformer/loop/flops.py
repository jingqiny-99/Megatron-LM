# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-sublayer GEMM FLOP table of the hybrid loop.

The convention is the GPT loop's (``LoopTransformerBlock._record_work``) and the training
FLOP counter's: GEMM work only, with FMA = 2; norms, softmax, mHC mixing, MoE dispatch and the
optimizer are excluded. The functions return **forward** FLOPs of one pass; the caller adds
backward (2 x forward) and recompute (1 x forward) per pass.

Arguments:

* ``tokens`` (T): rows issued to the pass (``S * B`` for SBHD, the compacted row count for
  compact MoR). May be a Python number or a tensor.
* ``squared_lengths`` (Q): ``sum_rows L**2`` (``B * S**2`` for full SBHD rows).
* ``seq_length``: the sequence length of the call; the DSv4 top-k term depends on it.

One documented difference from the GPT loop: the MoE row excludes the router GEMM
(``2 * T * h * num_experts``), as the training FLOP counter does; ``_record_work`` includes it.
Dense patterns are unaffected.
"""

from typing import Any, Optional

from megatron.core.models.hybrid.layers.utils import Symbols

# The DSv4 attention compress ratios (window-only, CSA, HCA).
DSV4_COMPRESS_RATIOS = (0, 4, 128)


def dsv4_hybrid_self_attention_coefficients(
    *,
    hidden_size,
    num_attention_heads,
    v_head_dim,
    q_lora_rank,
    o_groups,
    o_lora_rank,
    csa_window_size,
    seq_length,
    n_layers_r0,
    n_layers_r4,
    n_layers_r128,
    dsa_indexer_n_heads,
    dsa_indexer_head_dim,
    dsa_indexer_topk,
):
    """DSv4-hybrid self-attention FLOP coefficients ``(token_linear, core)``.

    A copy of ``megatron.training.training._dsv4_hybrid_self_attention_flops`` (core code may
    not import training; a unit test pins the two equal). Multiply ``token_linear`` by the
    token count and ``core`` by ``sum_i(L_i ** 2)``; neither includes the FMA factor.
    """
    n_attn_layers = n_layers_r0 + n_layers_r4 + n_layers_r128

    # MLA projections (token-linear, per attention layer).
    q_term = q_lora_rank * (hidden_size + num_attention_heads * v_head_dim + 1)
    kv_term = hidden_size * v_head_dim + v_head_dim
    o_term = num_attention_heads * v_head_dim * o_lora_rank + o_groups * o_lora_rank * hidden_size
    mla_proj_term = (q_term + kv_term + o_term) * n_attn_layers

    # Sparse attention: window (token-linear) and compressed KV (L^2).
    sparse_attn_r0 = n_layers_r0 * num_attention_heads * csa_window_size * v_head_dim * 2
    sparse_attn_r128_window = n_layers_r128 * num_attention_heads * csa_window_size * v_head_dim * 2
    sparse_attn_r128_core = n_layers_r128 * num_attention_heads * v_head_dim / 128

    # Main compressor (ratio > 0 layers).
    main_compressor_term = (
        n_layers_r4 * hidden_size * (2 * v_head_dim) * 2
        + n_layers_r128 * hidden_size * (1 * v_head_dim) * 2
    )

    # ratio == 4 layers: sparse attention + indexer.
    if n_layers_r4 > 0:
        assert (
            dsa_indexer_n_heads is not None
        ), "dsa_indexer_n_heads must be set for dsv4_hybrid with ratio==4 layers."
        assert (
            dsa_indexer_head_dim is not None
        ), "dsa_indexer_head_dim must be set for dsv4_hybrid with ratio==4 layers."
        assert (
            dsa_indexer_topk is not None
        ), "dsa_indexer_topk must be set for dsv4_hybrid with ratio==4 layers."
        effective_topk_4 = min(dsa_indexer_topk, seq_length // 4)
        avg_comp_4 = effective_topk_4 * (1 - effective_topk_4 * 4 / (2 * seq_length))
        sparse_attn_r4 = (
            n_layers_r4 * num_attention_heads * (csa_window_size + avg_comp_4) * v_head_dim * 2
        )
        indexer_token_term = (
            n_layers_r4 * hidden_size * (2 * dsa_indexer_head_dim) * 2
            + n_layers_r4 * q_lora_rank * dsa_indexer_n_heads * dsa_indexer_head_dim
            + n_layers_r4 * hidden_size * dsa_indexer_n_heads
        )
        indexer_scoring_core = n_layers_r4 * dsa_indexer_n_heads * dsa_indexer_head_dim / 4
    else:
        sparse_attn_r4 = 0
        indexer_token_term = 0
        indexer_scoring_core = 0

    token_linear = (
        mla_proj_term
        + sparse_attn_r0
        + sparse_attn_r4
        + sparse_attn_r128_window
        + main_compressor_term
        + indexer_token_term
    )
    core = sparse_attn_r128_core + indexer_scoring_core
    return token_linear, core


def attention_forward_flops(config: Any, tokens, squared_lengths, *, arbitrary_mask=False):
    """Plain attention (``*``): QKV + output projections, output gate, core attention.

    The core term is ``2 * Q * q_width`` with a causal mask and ``4 * Q * q_width`` with an
    arbitrary mask (masked MoR runs the full square).
    """
    hidden = config.hidden_size
    q_width = config.num_attention_heads * config.kv_channels
    kv_width = config.num_query_groups * config.kv_channels
    gate_width = 0
    if config.attention_output_gate:
        gate_width = (
            q_width
            if config.gated_attention_proj_granularity == "elementwise"
            else config.num_attention_heads
        )
    return (
        4 * tokens * hidden * (q_width + kv_width)
        + 2 * tokens * hidden * gate_width
        + (4 if arbitrary_mask else 2) * squared_lengths * q_width
    )


def mlp_forward_flops(config: Any, tokens):
    """Dense MLP (``-``)."""
    return (
        (6 if config.gated_linear_unit else 4)
        * tokens
        * config.hidden_size
        * (config.ffn_hidden_size)
    )


def moe_forward_flops(config: Any, tokens):
    """MoE (``E``): routed experts (top-k) and shared experts; router GEMM excluded."""
    glu = 6 if config.gated_linear_unit else 4
    hidden = config.hidden_size
    ffn = (
        config.moe_ffn_hidden_size
        if config.moe_ffn_hidden_size is not None
        else config.ffn_hidden_size
    )
    shared = config.moe_shared_expert_intermediate_size or 0
    topk = config.moe_router_topk
    latent = config.moe_latent_size
    if latent:
        return (
            glu * tokens * latent * ffn * topk
            + 4 * tokens * hidden * latent
            + glu * tokens * hidden * shared
        )
    return glu * tokens * hidden * (ffn * topk + shared)


def dsv4_attention_forward_flops(
    config: Any, compress_ratio: int, tokens, squared_lengths, seq_length
):
    """DSv4 attention (``W`` / ``C`` / ``H``, and ``D`` under ``dsv4_hybrid``) of one layer."""
    if compress_ratio not in DSV4_COMPRESS_RATIOS:
        raise ValueError(
            f"DSv4 compress ratio {compress_ratio} has no FLOP row; "
            f"expected one of {DSV4_COMPRESS_RATIOS}"
        )
    token_linear, core = dsv4_hybrid_self_attention_coefficients(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        v_head_dim=config.v_head_dim,
        q_lora_rank=config.q_lora_rank,
        o_groups=config.o_groups,
        o_lora_rank=config.o_lora_rank,
        csa_window_size=config.csa_window_size,
        seq_length=seq_length,
        n_layers_r0=int(compress_ratio == 0),
        n_layers_r4=int(compress_ratio == 4),
        n_layers_r128=int(compress_ratio == 128),
        dsa_indexer_n_heads=config.dsa_indexer_n_heads,
        dsa_indexer_head_dim=config.dsa_indexer_head_dim,
        dsa_indexer_topk=config.dsa_indexer_topk,
    )
    return 2 * (token_linear * tokens + core * squared_lengths)


def dsv4_up_proj_forward_flops(config: Any, tokens):
    """The up projections inside one DSv4 attention's ``qkv_up_proj_and_rope_apply`` (the part
    a ``mla_up_proj`` selective recompute replays, D42): the query up projection
    (``q_lora_rank -> heads * v_head_dim``) and the single-head key-value projection
    (``hidden -> v_head_dim``). The same weight terms as ``q_term`` / ``kv_term`` of
    ``dsv4_hybrid_self_attention_coefficients``; norms and RoPE are excluded."""
    weights = (
        config.q_lora_rank * config.num_attention_heads * config.v_head_dim
        + config.hidden_size * config.v_head_dim
    )
    return 2 * tokens * weights


def sublayer_forward_flops(
    symbol: str,
    config: Any,
    tokens,
    squared_lengths,
    seq_length,
    *,
    arbitrary_mask: bool = False,
    compress_ratio: Optional[int] = None,
):
    """Forward FLOPs of one pass through one hybrid sublayer.

    Args:
        symbol: The sublayer's pattern symbol.
        config: The sublayer's config (its per-layer config, or the stack config).
        tokens: Issued rows (T).
        squared_lengths: ``sum_rows L**2`` (Q).
        seq_length: Sequence length of the call (DSv4 top-k term).
        arbitrary_mask: Plain attention runs an arbitrary (full-square) mask.
        compress_ratio: The DSv4 ratio of a ``D`` sublayer (``csa_compress_ratios`` entry).
    """
    if symbol == Symbols.ATTENTION:
        return attention_forward_flops(
            config, tokens, squared_lengths, arbitrary_mask=arbitrary_mask
        )
    if symbol == Symbols.MLP:
        return mlp_forward_flops(config, tokens)
    if symbol == Symbols.MOE:
        return moe_forward_flops(config, tokens)
    if symbol in Symbols.DSV4_COMPRESS_RATIO_MAP:
        return dsv4_attention_forward_flops(
            config, Symbols.DSV4_COMPRESS_RATIO_MAP[symbol], tokens, squared_lengths, seq_length
        )
    if symbol == Symbols.DS_ATTENTION:
        if compress_ratio is None:
            raise ValueError("A 'D' sublayer needs its csa_compress_ratios entry for FLOPs")
        return dsv4_attention_forward_flops(
            config, compress_ratio, tokens, squared_lengths, seq_length
        )
    raise ValueError(f"The hybrid loop FLOP table has no row for sublayer {symbol!r}")


def huginn_adapter_forward_flops(tokens, hidden_size, num_streams=1):
    """The shared ``[h, 2h]`` Huginn adapter on each of ``num_streams`` streams, one round."""
    return 4 * num_streams * tokens * hidden_size**2


def mor_router_forward_flops(tokens, hidden_size, loop_iterations):
    """The MoR depth router (``h -> h -> R``), once per microbatch."""
    return 2 * tokens * hidden_size * (hidden_size + loop_iterations)


def logits_forward_flops(tokens, hidden_size, vocab_size):
    """One output-layer (logit) projection."""
    return 2 * tokens * hidden_size * vocab_size


def mtp_depth_extra_forward_flops(tokens, hidden_size):
    """One MTP depth's norms and ``eh`` projection, as the training FLOP counter counts them."""
    return 2 * (3 * hidden_size + 2 * hidden_size * hidden_size) * tokens
