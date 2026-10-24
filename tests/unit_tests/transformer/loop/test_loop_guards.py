# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the hybrid loop guards (megatron/core/transformer/loop/guards.py) and the loop
role fields of TransformerConfig.

Config, pattern, role and input rules run without process groups. The wiring tests at the end
(DistributedDataParallel and HybridModel raise through the guards) share one model-parallel
initialization for the module: gloo on CPU, NCCL on CUDA.
"""

import copy
import os
import socket

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.quantization.quant_config import RecipeConfig
from megatron.core.transformer.loop import guards
from megatron.core.transformer.loop.guards import (
    LoopRoles,
    has_dsv4_attention,
    is_block_boundary,
    is_hybrid_loop_config,
    reject_pipe_in_loop_pattern,
    resolve_hybrid_loop_roles,
    rsbf_item_form_violations,
    validate_hybrid_loop_config,
    validate_hybrid_loop_inputs,
    validate_hybrid_loop_pattern,
    validate_loop_ddp_config,
)
from megatron.core.transformer.loop_transformer_block import LoopTransformerBlock
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from tests.unit_tests.test_utilities import Utils

_PREFIX = "Hybrid loop does not support: "
_DENSE = "*-*-*-*-"
_DSV4 = "HEHECEHECEHECE"


def _dense_config(**overrides):
    """Hybrid loop config of the dense '*-*-*-*-' stack: prelude 2, core 4, coda 2."""
    fields = dict(
        num_layers=len(_DENSE),
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        loop_architecture="huginn",
        loop_prelude_sublayers=2,
        loop_coda_sublayers=2,
    )
    fields.update(overrides)
    return TransformerConfig(**fields)


def _gpt_loop_config(**overrides):
    """GPT loop config (no role fields), as test_loop_transformer.py builds it."""
    fields = dict(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        loop_architecture="huginn",
        loop_prelude_layers=1,
        loop_core_layers=2,
    )
    fields.update(overrides)
    return TransformerConfig(**fields)


def _dsv4_config(**overrides):
    """The DSV4-TINY loop config: prelude 'HE', core 'HECEHECE', coda 'HECE', mHC 4, hash 1."""
    ratios = {"H": 128, "C": 4, "W": 0, "E": 0}
    fields = dict(
        num_layers=len(_DSV4),
        hidden_size=512,
        num_attention_heads=8,
        multi_latent_attention=True,
        q_lora_rank=192,
        kv_lora_rank=64,
        qk_head_dim=16,
        qk_pos_emb_head_dim=8,
        v_head_dim=16,
        rope_type="rope",
        experimental_attention_variant="dsv4_hybrid",
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=32,
        csa_window_size=32,
        csa_compress_ratios=[ratios[symbol] for symbol in _DSV4],
        num_moe_experts=8,
        moe_ffn_hidden_size=256,
        moe_router_topk=2,
        moe_token_dispatcher_type="alltoall",
        moe_num_hash_layers=1,
        hash_moe_vocab_size=128,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        is_hybrid_model=True,
        loop_architecture="huginn",
        loop_prelude_sublayers=2,
        loop_coda_sublayers=4,
    )
    fields.update(overrides)
    return MLATransformerConfig(**fields)


def _violating(config, **fields):
    """A copy of a valid config with ``fields`` set directly (no ``__post_init__``).

    Each rule is tested on a config that differs from a valid one in that rule's fields only,
    so the rule is isolated from unrelated ``__post_init__`` prerequisites of the feature.
    """
    config = copy.copy(config)
    for name, value in fields.items():
        assert hasattr(config, name), name
        setattr(config, name, value)
    return config


def _reported(excinfo):
    """Rule names listed by a guard's ValueError."""
    message = str(excinfo.value)
    assert message.startswith(_PREFIX), message
    return message[len(_PREFIX) :].split(", ")


# ---------------------------------------------------------------------------------------------
# Config rules (validate_hybrid_loop_config)
# ---------------------------------------------------------------------------------------------

_CONFIG_RULES = [
    (guards.DELAYED_FP8_SCALING, dict(fp8="hybrid", fp8_recipe="delayed")),
    (guards.CUDA_GRAPHS, dict(cuda_graph_impl="transformer_engine")),
    (guards.CUDA_GRAPHS, dict(cuda_graph_impl="local")),
    (guards.MHC_RECOMPUTE, dict(recompute_granularity="selective", recompute_modules=["mhc"])),
    (guards.MOE_EP_OVERLAP, dict(overlap_moe_expert_parallel_comm=True)),
    (guards.PAGED_STASH, dict(moe_paged_stash=True)),
    (guards.HYBRIDEP_PADDING, dict(moe_hybridep_pad_variable_tokens=True)),
    (guards.HYBRIDEP_PADDING, dict(moe_hybridep_pad_uneven_dispatch_inputs=True)),
    (guards.SHORTCUT_MOE, dict(moe_shortcut_connection=True)),
    (
        guards.RESIDUAL_STREAM_RECOMPUTE,
        dict(recompute_granularity="selective", recompute_modules=["residual_stream"]),
    ),
    (guards.PIPELINE_PARALLELISM, dict(pipeline_model_parallel_size=2)),
    (guards.PIPELINE_PARALLELISM, dict(virtual_pipeline_model_parallel_size=2)),
    (guards.CONTEXT_PARALLELISM, dict(context_parallel_size=2)),
    (guards.CONTEXT_PARALLELISM, dict(dynamic_context_parallel=True)),
    (guards.SEQUENCE_PARALLELISM, dict(sequence_parallel=True)),
    (guards.DROPOUT, dict(hidden_dropout=0.1)),
    (guards.DROPOUT, dict(attention_dropout=0.1)),
    (guards.DROPOUT, dict(moe_input_jitter_eps=0.01)),
    (guards.FP4, dict(fp4="e2m1")),
    (guards.ACTIVATION_OFFLOAD, dict(cpu_offloading=True)),
    (guards.ACTIVATION_OFFLOAD, dict(fine_grained_activation_offloading=True)),
    (guards.OTHER_RECOMPUTE, dict(recompute_granularity="full")),
    (
        guards.OTHER_RECOMPUTE,
        dict(recompute_granularity="selective", recompute_modules=["core_attn"]),
    ),
    (
        guards.OTHER_RECOMPUTE,
        dict(recompute_granularity="selective", recompute_modules=["moe_act", "layernorm"]),
    ),
    (guards.GTP_WEIGHT_REMAT, dict(gtp_weight_remat_size=2)),
    (guards.GTP_WEIGHT_REMAT, dict(expert_gtp_weight_remat_size=2)),
    (guards.WIDE_RESIDUAL, dict(wide_residual=object())),
]


@pytest.mark.parametrize(
    "rule,fields",
    _CONFIG_RULES,
    ids=[f"{rule}-{'-'.join(fields)}" for rule, fields in _CONFIG_RULES],
)
def test_config_rule_is_reported_alone(rule, fields):
    config = _violating(_dense_config(), **fields)
    with pytest.raises(ValueError, match="Hybrid loop does not support") as excinfo:
        validate_hybrid_loop_config(config)
    assert _reported(excinfo) == [rule]


def test_valid_configs_pass_config_rules():
    validate_hybrid_loop_config(_dense_config())
    validate_hybrid_loop_config(_dsv4_config())
    # Non-delayed FP8 recipes are allowed (with the weight-cache rule of __post_init__).
    validate_hybrid_loop_config(
        _violating(
            _dense_config(), fp8="e4m3", fp8_recipe="mxfp8", disable_parameter_transpose_cache=True
        )
    )
    # mHC and MTP are allowed; recompute modules without selective granularity are ignored.
    validate_hybrid_loop_config(
        _violating(
            _dense_config(),
            enable_mhc_connections=True,
            mtp_num_layers=1,
            recompute_modules=["mhc"],
        )
    )


def test_config_rules_collect_every_violation():
    config = _violating(_dense_config(), sequence_parallel=True, moe_paged_stash=True)
    with pytest.raises(ValueError) as excinfo:
        validate_hybrid_loop_config(config)
    assert set(_reported(excinfo)) == {guards.PAGED_STASH, guards.SEQUENCE_PARALLELISM}


def test_mhc_and_residual_stream_recompute_have_their_own_rules():
    # The two loop-ruled modules never also count as "other" recompute.
    config = _violating(
        _dense_config(),
        recompute_granularity="selective",
        recompute_modules=["mhc", "residual_stream"],
    )
    with pytest.raises(ValueError) as excinfo:
        validate_hybrid_loop_config(config)
    assert set(_reported(excinfo)) == {guards.MHC_RECOMPUTE, guards.RESIDUAL_STREAM_RECOMPUTE}


def test_recompute_rules_accept_unset_module_list():
    # recompute_modules=None (before __post_init__ fills the default) must not raise a
    # TypeError; None means the default list ['core_attn'], which only the generic rule covers.
    config = _violating(_dense_config(), recompute_granularity="selective", recompute_modules=None)
    with pytest.raises(ValueError) as excinfo:
        validate_hybrid_loop_config(config)
    assert _reported(excinfo) == [guards.OTHER_RECOMPUTE]
    validate_hybrid_loop_config(_violating(_dense_config(), recompute_modules=None))


# ---------------------------------------------------------------------------------------------
# Pattern rules (validate_hybrid_loop_pattern, reject_pipe_in_loop_pattern)
# ---------------------------------------------------------------------------------------------


def test_pipe_in_pattern_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        validate_hybrid_loop_pattern(_dense_config(), "*-*-|*-*-", None)
    assert _reported(excinfo) == [guards.PIPE_IN_PATTERN]


def test_pipe_rejected_at_argument_validation():
    with pytest.raises(ValueError) as excinfo:
        reject_pipe_in_loop_pattern("huginn", "*-*-|*-*-/*-")
    assert _reported(excinfo) == [guards.PIPE_IN_PATTERN]
    with pytest.raises(ValueError, match="'|' in hybrid_layer_pattern"):
        reject_pipe_in_loop_pattern("mor", "*-|*-")
    # Not a loop, no pattern, or a pipe-free pattern: nothing to reject.
    reject_pipe_in_loop_pattern("none", "*-*-|*-*-")
    reject_pipe_in_loop_pattern("huginn", None)
    reject_pipe_in_loop_pattern("huginn", "*-*-*-*-/*-/*-")


def test_mla_without_dsv4_hybrid_is_rejected():
    # Isolated: multi_latent_attention on a pattern without MLA symbols.
    config = _violating(_dense_config(), multi_latent_attention=True)
    with pytest.raises(ValueError) as excinfo:
        validate_hybrid_loop_pattern(config, _DENSE, None)
    assert _reported(excinfo) == [guards.MLA_WITHOUT_DSV4]


@pytest.mark.parametrize("main,mtp", [("C-C-C-C-", None), ("-E-E", "W-"), ("+-+-+-+-", None)])
def test_mla_symbols_without_dsv4_hybrid_are_rejected(main, mtp):
    with pytest.raises(ValueError) as excinfo:
        validate_hybrid_loop_pattern(_dense_config(), main, mtp)
    assert _reported(excinfo) == [guards.MLA_WITHOUT_DSV4]


def test_valid_patterns_pass():
    validate_hybrid_loop_pattern(_dense_config(), _DENSE, None)
    validate_hybrid_loop_pattern(_dense_config(), _DENSE, "*-")
    validate_hybrid_loop_pattern(_dsv4_config(), _DSV4, "WE")


# ---------------------------------------------------------------------------------------------
# Role rules (resolve_hybrid_loop_roles)
# ---------------------------------------------------------------------------------------------


def test_dense_roles_resolve():
    roles = resolve_hybrid_loop_roles(_dense_config(), list(_DENSE), None)
    assert roles == LoopRoles(prelude=2, core=4, coda=2)
    assert (roles.core_start, roles.core_end) == (2, 6)


def test_dsv4_roles_resolve():
    # The hash layer is the first E (global layer number 2 = the prelude end).
    config = _dsv4_config()
    roles = resolve_hybrid_loop_roles(config, list(_DSV4), hash_moe_layer_threshold=2)
    assert roles == LoopRoles(prelude=2, core=8, coda=4)
    assert _DSV4[roles.core_start : roles.core_end] == "HECEHECE"
    assert has_dsv4_attention(_DSV4) and not has_dsv4_attention(_DENSE)


@pytest.mark.parametrize("prelude,coda", [(4, 4), (6, 2), (8, 0), (0, 9)])
def test_roles_without_core_are_rejected(prelude, coda):
    config = _violating(_dense_config(), loop_prelude_sublayers=prelude, loop_coda_sublayers=coda)
    with pytest.raises(ValueError) as excinfo:
        resolve_hybrid_loop_roles(config, list(_DENSE), None)
    assert _reported(excinfo) == [guards.ROLES_DO_NOT_MATCH]


def test_unset_roles_are_rejected():
    # A loop stack built from a config without role fields (e.g. a GPT loop config).
    config = _violating(_dense_config(), loop_prelude_sublayers=None, loop_coda_sublayers=None)
    with pytest.raises(ValueError) as excinfo:
        resolve_hybrid_loop_roles(config, list(_DENSE), None)
    assert _reported(excinfo) == [guards.ROLES_DO_NOT_MATCH]


@pytest.mark.parametrize("prelude,coda", [(1, 2), (2, 1), (3, 3)])
def test_cut_inside_a_block_is_rejected(prelude, coda):
    config = _violating(_dense_config(), loop_prelude_sublayers=prelude, loop_coda_sublayers=coda)
    with pytest.raises(ValueError) as excinfo:
        resolve_hybrid_loop_roles(config, list(_DENSE), None)
    assert _reported(excinfo) == [guards.CUT_NOT_ON_BLOCK]


def test_empty_prelude_and_coda_are_block_boundaries():
    config = _violating(_dense_config(), loop_prelude_sublayers=0, loop_coda_sublayers=0)
    assert resolve_hybrid_loop_roles(config, list(_DENSE), None) == LoopRoles(0, 8, 0)


def test_hash_layer_beyond_the_prelude_is_rejected():
    config = _dsv4_config()
    with pytest.raises(ValueError) as excinfo:
        resolve_hybrid_loop_roles(config, list(_DSV4), hash_moe_layer_threshold=4)
    assert _reported(excinfo) == [guards.HASH_BEYOND_PRELUDE]
    # No hash routing (None or 0) and a threshold inside the prelude pass.
    for threshold in (None, 0, 1, 2):
        resolve_hybrid_loop_roles(config, list(_DSV4), hash_moe_layer_threshold=threshold)


@pytest.mark.parametrize("symbol", ["M", "G", "K"])
def test_unsupported_sublayers_are_rejected(symbol):
    pattern = (symbol + "-") * 4
    with pytest.raises(ValueError) as excinfo:
        resolve_hybrid_loop_roles(_dense_config(), list(pattern), None)
    assert _reported(excinfo) == [guards.UNSUPPORTED_SUBLAYERS]


def test_plain_mla_sublayer_trips_two_rules():
    # '+' is a plain MLA sublayer (G27) and an MLA symbol without dsv4_hybrid (G15).
    pattern = "+-+-+-+-"
    with pytest.raises(ValueError) as excinfo:
        resolve_hybrid_loop_roles(_dense_config(), list(pattern), None)
    assert _reported(excinfo) == [guards.UNSUPPORTED_SUBLAYERS]
    with pytest.raises(ValueError) as excinfo:
        validate_hybrid_loop_pattern(_dense_config(), pattern, None)
    assert _reported(excinfo) == [guards.MLA_WITHOUT_DSV4]


def test_mor_on_a_dense_or_moe_core_passes():
    config = _dense_config(loop_architecture="mor")
    resolve_hybrid_loop_roles(config, list(_DENSE), None)
    resolve_hybrid_loop_roles(config, list("*-*E*E*-"), None)
    masked = _dense_config(loop_architecture="mor", loop_token_execution="masked")
    resolve_hybrid_loop_roles(masked, list(_DENSE), None)


def test_mor_with_mhc_is_rejected():
    config = _violating(_dense_config(loop_architecture="mor"), enable_mhc_connections=True)
    with pytest.raises(ValueError) as excinfo:
        resolve_hybrid_loop_roles(config, list(_DENSE), None)
    assert _reported(excinfo) == [guards.MOR_NON_DENSE_CORE]
    # Huginn runs on an mHC core.
    resolve_hybrid_loop_roles(
        _violating(_dense_config(), enable_mhc_connections=True), list(_DENSE), None
    )


def test_compact_mor_on_a_dsv4_core_is_rejected():
    config = _dense_config(loop_architecture="mor", loop_token_execution="compact")
    with pytest.raises(ValueError) as excinfo:
        resolve_hybrid_loop_roles(config, list("C-C-C-C-"), None)
    assert _reported(excinfo) == [guards.MOR_NON_DENSE_CORE]


@pytest.mark.parametrize("symbol", ["C", "H", "W", "D"])
def test_masked_mor_on_dsv4_attention_is_rejected(symbol):
    config = _dense_config(loop_architecture="mor", loop_token_execution="masked")
    with pytest.raises(ValueError) as excinfo:
        resolve_hybrid_loop_roles(config, list((symbol + "-") * 4), None)
    assert _reported(excinfo) == [f"{guards.MOR_NON_DENSE_CORE} ({guards.MASKED_MOR_ON_DSV4})"]


def test_masked_mor_on_a_plain_mla_core_trips_two_rules():
    config = _dense_config(loop_architecture="mor", loop_token_execution="masked")
    with pytest.raises(ValueError) as excinfo:
        resolve_hybrid_loop_roles(config, list("+-+-+-+-"), None)
    assert set(_reported(excinfo)) == {
        guards.UNSUPPORTED_SUBLAYERS,
        f"{guards.MOR_NON_DENSE_CORE} ({guards.MASKED_MOR_ON_DSV4})",
    }


# ---------------------------------------------------------------------------------------------
# Input rules (validate_hybrid_loop_inputs)
# ---------------------------------------------------------------------------------------------


def _thd(num_sequences, qkv_format="thd"):
    cu_seqlens = torch.arange(num_sequences + 1, dtype=torch.int32) * 8
    return PackedSeqParams(
        qkv_format=qkv_format,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens.clone(),
        max_seqlen_q=8,
        max_seqlen_kv=8,
    )


def test_inputs_with_one_sequence_pass():
    validate_hybrid_loop_inputs(packed_seq_params=None, inference_context=None)
    validate_hybrid_loop_inputs(packed_seq_params=_thd(1), inference_context=None)
    # A dense mask without DSv4 attention, or DSv4 attention without a mask, is fine.
    mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    validate_hybrid_loop_inputs(packed_seq_params=None, inference_context=None, attention_mask=mask)
    validate_hybrid_loop_inputs(
        packed_seq_params=None, inference_context=None, has_dsv4_attention=True
    )


@pytest.mark.parametrize("packed", [_thd(3), _thd(1, qkv_format="sbhd"), PackedSeqParams()])
def test_packed_multi_sequence_is_rejected(packed):
    with pytest.raises(ValueError) as excinfo:
        validate_hybrid_loop_inputs(packed_seq_params=packed, inference_context=None)
    assert _reported(excinfo) == [guards.PACKED_MULTI_SEQUENCE]


def test_inference_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        validate_hybrid_loop_inputs(packed_seq_params=None, inference_context=object())
    assert _reported(excinfo) == [guards.INFERENCE]


def test_dense_mask_with_dsv4_attention_on_sbhd_is_rejected():
    mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    with pytest.raises(ValueError, match="no_create_attention_mask_in_dataloader") as excinfo:
        validate_hybrid_loop_inputs(
            packed_seq_params=None,
            inference_context=None,
            attention_mask=mask,
            has_dsv4_attention=True,
        )
    assert _reported(excinfo) == [guards.SBHD_MASK_WITH_DSV4]
    # THD: DSv4 attention does not take the dense mask, so the input check does not apply.
    validate_hybrid_loop_inputs(
        packed_seq_params=_thd(1),
        inference_context=None,
        attention_mask=mask,
        has_dsv4_attention=True,
    )


# ---------------------------------------------------------------------------------------------
# DDP rules (validate_loop_ddp_config)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule,ddp_fields",
    [
        (guards.OVERLAP_GRAD_REDUCE, dict(overlap_grad_reduce=True)),
        (
            guards.OVERLAP_PARAM_GATHER,
            dict(use_distributed_optimizer=True, overlap_param_gather=True),
        ),
    ],
)
def test_ddp_overlap_is_rejected_for_hybrid_loops(rule, ddp_fields):
    ddp_config = DistributedDataParallelConfig(**ddp_fields)
    with pytest.raises(ValueError) as excinfo:
        validate_loop_ddp_config(_dense_config(), ddp_config)
    assert _reported(excinfo) == [rule]


def test_ddp_overlap_is_allowed_outside_hybrid_loops():
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=True, use_distributed_optimizer=True, overlap_param_gather=True
    )
    # The eager GPT loop keeps overlap (its DDP readiness test runs with it); non-loop too.
    validate_loop_ddp_config(_gpt_loop_config(), ddp_config)
    validate_loop_ddp_config(
        TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4), ddp_config
    )
    validate_loop_ddp_config(_dense_config(), DistributedDataParallelConfig())


# ---------------------------------------------------------------------------------------------
# TransformerConfig role fields and loop rules (__post_init__)
# ---------------------------------------------------------------------------------------------


def test_role_fields_select_hybrid_role_mode():
    assert is_hybrid_loop_config(_dense_config())
    assert not is_hybrid_loop_config(_gpt_loop_config())
    assert not is_hybrid_loop_config(
        TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4)
    )


@pytest.mark.parametrize(
    "fields", [dict(loop_prelude_sublayers=2), dict(loop_coda_sublayers=0)], ids=str
)
def test_role_fields_require_a_loop(fields):
    with pytest.raises(ValueError, match="require loop_architecture"):
        TransformerConfig(num_layers=8, hidden_size=64, num_attention_heads=4, **fields)


@pytest.mark.parametrize(
    "fields", [dict(loop_coda_sublayers=None), dict(loop_prelude_sublayers=None)], ids=str
)
def test_role_fields_are_set_together(fields):
    with pytest.raises(ValueError, match="must be set together"):
        _dense_config(**fields)


@pytest.mark.parametrize(
    "fields", [dict(loop_prelude_sublayers=-2), dict(loop_coda_sublayers=-1)], ids=str
)
def test_role_fields_are_nonnegative(fields):
    with pytest.raises(ValueError, match="must be nonnegative"):
        _dense_config(**fields)


@pytest.mark.parametrize("prelude,coda", [(4, 4), (8, 0), (6, 3)])
def test_role_fields_leave_a_core(prelude, coda):
    with pytest.raises(ValueError, match="empty core"):
        _dense_config(loop_prelude_sublayers=prelude, loop_coda_sublayers=coda)


def test_untied_core_is_rejected_in_role_mode():
    with pytest.raises(ValueError, match="loop_untie_core is a GPT-loop baseline"):
        _dense_config(loop_untie_core=True)


@pytest.mark.parametrize("fields", [dict(loop_core_layers=3), dict(loop_prelude_layers=1)], ids=str)
def test_gpt_layer_counts_are_rejected_in_role_mode(fields):
    with pytest.raises(ValueError, match="GPT loop layer counts in role mode"):
        _dense_config(**fields)


def test_gpt_layer_count_defaults_are_accepted_in_role_mode():
    config = _dense_config(loop_prelude_layers=2, loop_core_layers=4)
    assert is_hybrid_loop_config(config)


def test_role_mode_skips_the_gpt_layer_count_checks():
    # num_layers=4 does not fit the GPT defaults (prelude 2 + core 4); role mode ignores them.
    with pytest.raises(ValueError, match="Loop prelude/core must fit in physical num_layers"):
        TransformerConfig(
            num_layers=4, hidden_size=64, num_attention_heads=4, loop_architecture="huginn"
        )
    config = TransformerConfig(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        loop_architecture="huginn",
        loop_prelude_sublayers=2,
        loop_coda_sublayers=0,
    )
    assert is_hybrid_loop_config(config)


@pytest.mark.parametrize("roles", [False, True])
def test_shared_loop_messages_are_unchanged(roles):
    build = _dense_config if roles else _gpt_loop_config
    with pytest.raises(ValueError, match="Loop iterations and core layers must be positive"):
        build(loop_iterations=0)
    with pytest.raises(ValueError, match="loop_bptt_iterations must be nonnegative"):
        build(loop_bptt_iterations=-1)
    with pytest.raises(ValueError, match="MoR uses token-choice depth and full BPTT"):
        build(loop_architecture="mor", loop_sample_depth=True)


def test_gpt_loop_messages_are_unchanged():
    with pytest.raises(ValueError, match="Loop iterations and core layers must be positive"):
        _gpt_loop_config(loop_core_layers=0)
    with pytest.raises(ValueError, match="Loop prelude/core must fit in physical num_layers"):
        _gpt_loop_config(loop_prelude_layers=3)
    with pytest.raises(ValueError, match="Untied loop baseline requires fixed-depth Huginn"):
        _gpt_loop_config(loop_untie_core=True, loop_sample_depth=True)
    with pytest.raises(ValueError, match="Untied core banks must fit in physical num_layers"):
        _gpt_loop_config(loop_untie_core=True)
    # The GPT loop config itself still constructs.
    assert _gpt_loop_config().loop_core_layers == 2


def test_gpt_loop_block_rejects_role_fields():
    with pytest.raises(ValueError, match="hybrid loop role fields"):
        LoopTransformerBlock._validate_config(_dense_config())
    LoopTransformerBlock._validate_config(_gpt_loop_config())


_WEIGHT_CACHE_RULE = "quantized loop without disable_parameter_transpose_cache"


def test_quantized_hybrid_loop_requires_disabled_weight_cache():
    with pytest.raises(ValueError, match=_WEIGHT_CACHE_RULE):
        _dense_config(fp8="e4m3", fp8_recipe="mxfp8")
    with pytest.raises(ValueError, match=_WEIGHT_CACHE_RULE):
        _dense_config(fp4="e2m1")


def test_quantized_hybrid_loop_with_disabled_weight_cache_passes_the_rule():
    # The weight-cache rule runs before the generic FP8 checks; anything raised after it
    # comes from those checks (e.g. a missing Transformer Engine), not from the loop.
    try:
        _dense_config(fp8="e4m3", fp8_recipe="mxfp8", disable_parameter_transpose_cache=True)
    except Exception as error:  # pylint: disable=broad-except
        assert _WEIGHT_CACHE_RULE not in str(error)
        assert "Hybrid loop" not in str(error) and "Loop does not support" not in str(error)


def test_quantized_gpt_loop_requires_disabled_weight_cache():
    # New on the GPT path: its block guard rejects FP8 / FP4, so a quant_recipe (or kitchen)
    # loop config is the one that reaches this rule.
    with pytest.raises(ValueError, match=_WEIGHT_CACHE_RULE):
        _gpt_loop_config(quant_recipe=RecipeConfig([], {}))
    config = _gpt_loop_config(
        quant_recipe=RecipeConfig([], {}), disable_parameter_transpose_cache=True
    )
    assert config.disable_parameter_transpose_cache
    # Unquantized loops and quantized non-loop configs are unaffected.
    assert not _gpt_loop_config().disable_parameter_transpose_cache
    TransformerConfig(
        num_layers=2, hidden_size=64, num_attention_heads=4, quant_recipe=RecipeConfig([], {})
    )


# ---------------------------------------------------------------------------------------------
# Wiring: DistributedDataParallel and HybridModel raise through the guards
# ---------------------------------------------------------------------------------------------


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def model_parallel():
    """Initialize model parallelism once for the module (NCCL on CUDA, gloo on CPU)."""
    if torch.cuda.is_available():
        Utils.initialize_model_parallel(1, 1)
        yield
        Utils.destroy_model_parallel()
    else:
        created_group = False
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(_free_port()))
            dist.init_process_group("gloo", rank=0, world_size=1)
            created_group = True
        parallel_state.initialize_model_parallel(1, 1)
        yield
        parallel_state.destroy_model_parallel()
        if created_group:
            dist.destroy_process_group()


def test_ddp_wrap_rejects_overlap_for_a_hybrid_loop(model_parallel):
    module = torch.nn.Linear(8, 8, bias=False)
    if torch.cuda.is_available():
        module = module.cuda()
    with pytest.raises(ValueError) as excinfo:
        DistributedDataParallel(
            _dense_config(), DistributedDataParallelConfig(overlap_grad_reduce=True), module
        )
    assert _reported(excinfo) == [guards.OVERLAP_GRAD_REDUCE]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DDP grad buffers live on CUDA")
def test_ddp_wrap_keeps_overlap_for_a_gpt_loop(model_parallel):
    module = torch.nn.Linear(8, 8, bias=False).cuda()
    ddp = DistributedDataParallel(
        _gpt_loop_config(), DistributedDataParallelConfig(overlap_grad_reduce=True), module
    )
    assert ddp.ddp_config.overlap_grad_reduce


@pytest.mark.parametrize(
    "rule,pattern,overrides",
    [
        (guards.PIPE_IN_PATTERN, "*-*-|*-*-", {}),
        (guards.MLA_WITHOUT_DSV4, _DENSE, dict(multi_latent_attention=True)),
        (guards.MLA_WITHOUT_DSV4, "--------/W-", {}),
    ],
)
def test_hybrid_model_checks_the_loop_pattern(model_parallel, rule, pattern, overrides):
    # The pattern check runs right after the pattern is parsed, before any layer is built.
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.transformer.enums import AttnBackend

    # The default 'auto' backend conflicts with the NVTE_FLASH_ATTN=0 / NVTE_FUSED_ATTN=0 that
    # tests/unit_tests/conftest.py sets for every test (LanguageModule checks it on CUDA).
    config = _violating(_dense_config(attention_backend=AttnBackend.unfused), **overrides)
    with pytest.raises(ValueError) as excinfo:
        HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=16,
            hybrid_layer_pattern=pattern,
        )
    assert _reported(excinfo) == [rule]


# ---------------------------------------------------------------------------------------------
# RSBF rules (validate_rsbf_config, loop_schedule='rsbf')
# ---------------------------------------------------------------------------------------------

_RSBF_PREFIX = "Loop RSBF does not support: "
_RSBF = dict(loop_schedule="rsbf", calculate_per_token_loss=True)


def _rsbf_reported(excinfo):
    message = str(excinfo.value)
    assert message.startswith(_RSBF_PREFIX), message
    return message[len(_RSBF_PREFIX) :].split(", ")


_RSBF_RULES = [
    (guards.RSBF_WITHOUT_HYBRID_ROLES, dict(loop_architecture="none")),
    (guards.RSBF_WITHOUT_HYBRID_ROLES, dict(loop_prelude_sublayers=None, loop_coda_sublayers=None)),
    (guards.RSBF_MOR, dict(loop_architecture="mor")),
    (guards.RSBF_SUM_LOSS, dict(calculate_per_token_loss=False)),
    (guards.RSBF_TRUNCATED_BPTT, dict(loop_bptt_iterations=1)),
    (guards.RSBF_SAMPLE_DEPTH, dict(loop_sample_depth=True)),
    (guards.RSBF_LOOP_RECOMPUTE, dict(loop_recompute=True)),
    (guards.RSBF_TP, dict(tensor_model_parallel_size=2, expert_tensor_parallel_size=1)),
    (guards.RSBF_ETP, dict(expert_tensor_parallel_size=2)),
    (
        guards.RSBF_STATIC_BUDGET_MOE,
        dict(
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="ncclep",
            moe_expert_rank_capacity_factor=1.0,
        ),
    ),
    # The all-to-all dispatcher ignores the static budget (the backend field alone is not
    # enough).
    (
        guards.RSBF_STATIC_BUDGET_MOE,
        dict(
            moe_token_dispatcher_type="alltoall",
            moe_flex_dispatcher_backend="hybridep",
            moe_expert_rank_capacity_factor=1.0,
            use_transformer_engine_op_fuser=True,
        ),
    ),
    (guards.RSBF_MOE_RECOMPUTE, dict(recompute_granularity="selective", recompute_modules=["moe"])),
    (guards.RSBF_CAPACITY_MOE, dict(moe_expert_capacity_factor=1.0)),
    (
        guards.RSBF_CAPACITY_MOE,
        dict(moe_expert_capacity_factor=1.0, moe_pad_expert_input_to_capacity=True),
    ),
    (guards.RSBF_FREEZE_FOR_MTP, dict(freeze_base_model_for_mtp=True)),
    (guards.RSBF_QUANTILE_BALANCING, dict(moe_router_load_balancing_type="quantile_balancing")),
    (
        guards.RSBF_QUANTILE_BALANCING,
        dict(moe_router_load_balancing_type=["aux_loss", "quantile_balancing"]),
    ),
    (guards.RSBF_DEPTH_PROBS, dict(loop_exit_criterion="sampled")),
    (guards.RSBF_DEPTH_PROBS, dict(loop_exit_criterion="sampled", loop_depth_probs=[0.5, 0.5])),
    (
        guards.RSBF_DEPTH_PROBS,
        dict(loop_exit_criterion="sampled", loop_depth_probs=[0.5, 0.5, 0.5, -0.5]),
    ),
    (guards.RSBF_DEPTH_PROBS, dict(loop_depth_probs=[0.25, 0.25, 0.25, 0.25])),
    (guards.RSBF_MIN_ROUNDS, dict(loop_min_rounds=0)),
    (guards.RSBF_MIN_ROUNDS, dict(loop_min_rounds=5)),
    (guards.RSBF_WAVES, dict(loop_rsbf_waves=0)),
    (guards.RSBF_GLOBAL_AUX_LOSS, dict(moe_router_load_balancing_type="global_aux_loss")),
    (
        guards.RSBF_GLOBAL_AUX_LOSS,
        dict(
            moe_router_load_balancing_type=["aux_loss", "global_aux_loss"],
            moe_aux_loss_coeff=[1e-3, 1e-3],
        ),
    ),
    (guards.RSBF_AUTOCAST, dict(enable_autocast=True)),
    # Item forms on the dense core of 4 sublayers (R19 - R21).
    (guards.RSBF_ITEM_FORM, dict(loop_rsbf_item_form="bogus")),
    (guards.RSBF_BLOCK_STRIDE_MISMATCH, dict(loop_rsbf_item_form="block")),
    (guards.RSBF_BLOCK_STRIDE_MISMATCH, dict(loop_rsbf_block_stride=2)),
    (
        guards.RSBF_BLOCK_STRIDE_MISMATCH,
        dict(loop_rsbf_item_form="block", loop_rsbf_block_stride=0),
    ),
    (
        guards.RSBF_BLOCK_STRIDE_MISMATCH,
        dict(loop_rsbf_item_form="nested", loop_rsbf_block_stride=-1),
    ),
    (
        guards.RSBF_BLOCK_STRIDE_BEYOND_CORE,
        dict(loop_rsbf_item_form="block", loop_rsbf_block_stride=5),
    ),
    (
        guards.RSBF_BLOCK_STRIDE_BEYOND_CORE,
        dict(loop_rsbf_item_form="nested", loop_rsbf_block_stride=4),
    ),
    (guards.RSBF_CODA_RECOMPUTE_WITHOUT_DSV4, dict(loop_rsbf_coda_recompute_mla_up_proj=True)),
]


@pytest.mark.parametrize(
    "rule,fields", _RSBF_RULES, ids=[f"{rule}-{'-'.join(fields)}" for rule, fields in _RSBF_RULES]
)
def test_rsbf_rule_is_reported_alone(rule, fields):
    """Each rule on a copy of a passing RSBF config (dense, or DSv4-like for the MoE rules)
    that differs in that rule's fields only."""
    moe = rule == guards.RSBF_MOE_RECOMPUTE or any(name.startswith("moe_") for name in fields)
    base = _dsv4_config(**_RSBF) if moe else _dense_config(**_RSBF)
    config = _violating(base, **fields)
    with pytest.raises(ValueError) as excinfo:
        guards.validate_rsbf_config(config)
    assert _rsbf_reported(excinfo) == [rule]


# R16: each option alone, at EP 8 (the rule is reported with the offending option).
_LOCKSTEP_OPTIONS = [
    ("moe_token_dispatcher_type='allgather'", dict(moe_token_dispatcher_type="allgather")),
    (
        "moe_flex_dispatcher_backend='deepep'",
        dict(moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="deepep"),
    ),
    (
        "moe_flex_dispatcher_backend='deepepv2'",
        dict(moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="deepepv2"),
    ),
    (
        "moe_flex_dispatcher_backend='ncclep'",
        dict(moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="ncclep"),
    ),
    ("moe_shared_expert_overlap", dict(moe_shared_expert_overlap=True)),
    (
        "overlap_dispatch_backward_with_experts_wgrad",
        dict(overlap_dispatch_backward_with_experts_wgrad=True),
    ),
    ("moe_megakernel_backend", dict(moe_megakernel_backend="any")),
    ("moe_latent_size", dict(moe_latent_size=128)),
    ("log_moe_overload_factor", dict(log_moe_overload_factor=True)),
]


@pytest.mark.parametrize(
    "option,fields", _LOCKSTEP_OPTIONS, ids=[option for option, _ in _LOCKSTEP_OPTIONS]
)
def test_rsbf_lockstep_moe_option_is_reported_alone(option, fields):
    """R16: with expert parallelism, each option a lockstep null cannot replay is reported
    alone (named); without expert parallelism the same config passes."""
    config = _violating(_dsv4_config(**_RSBF, expert_model_parallel_size=8), **fields)
    with pytest.raises(ValueError) as excinfo:
        guards.validate_rsbf_config(config)
    assert _rsbf_reported(excinfo) == [f"{guards.RSBF_LOCKSTEP_MOE} ({option})"]
    guards.validate_rsbf_config(_violating(_dsv4_config(**_RSBF), **fields))


def test_rsbf_lockstep_moe_options_are_listed_together():
    config = _violating(
        _dsv4_config(**_RSBF, expert_model_parallel_size=8),
        moe_token_dispatcher_type="allgather",
        log_moe_overload_factor=True,
    )
    with pytest.raises(ValueError) as excinfo:
        guards.validate_rsbf_config(config)
    assert _rsbf_reported(excinfo) == [
        f"{guards.RSBF_LOCKSTEP_MOE} (moe_token_dispatcher_type='allgather' / "
        "log_moe_overload_factor)"
    ]


def test_rsbf_configs_pass_and_rules_collect():
    guards.validate_rsbf_config(_dense_config(**_RSBF))
    guards.validate_rsbf_config(_dsv4_config(**_RSBF))
    # Lockstep MoE: DSv4 at EP 8 with the all-to-all dispatcher, and with flex + HybridEP under
    # a static per-rank budget (with the TE op fuser).
    guards.validate_rsbf_config(_dsv4_config(**_RSBF, expert_model_parallel_size=8))
    guards.validate_rsbf_config(
        _violating(
            _dsv4_config(**_RSBF, expert_model_parallel_size=8),
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            moe_expert_rank_capacity_factor=2.0,
            use_transformer_engine_op_fuser=True,
            loop_rsbf_over_budget="warn",
        )
    )
    guards.validate_rsbf_config(
        _dense_config(
            **_RSBF,
            loop_exit_criterion="sampled",
            loop_depth_probs=[0.25, 0.25, 0.25, 0.25],
            loop_min_rounds=2,
            loop_rsbf_waves=2,
            loop_rsbf_fill="sample_order",
            loop_rsbf_rebalance=False,
            loop_rsbf_move_transport="all_to_all",
            loop_rsbf_tail_item_moves=False,
        )
    )
    # Quantile balancing and global aux loss on a dense model: no router, nothing rejected.
    guards.validate_rsbf_config(
        _violating(_dense_config(**_RSBF), moe_router_load_balancing_type="global_aux_loss")
    )
    config = _violating(_dense_config(**_RSBF), calculate_per_token_loss=False, loop_recompute=True)
    with pytest.raises(ValueError) as excinfo:
        guards.validate_rsbf_config(config)
    assert _rsbf_reported(excinfo) == [guards.RSBF_SUM_LOSS, guards.RSBF_LOOP_RECOMPUTE]


def test_rsbf_item_forms_pass():
    """Block strides 1 .. L (dividing or not; L is the whole round), nested strides 1 .. L - 1
    or none, on the dense core (L = 4) and the DSV4-TINY core (L = 8, D42 with dsv4_hybrid);
    built through ``__post_init__``."""
    for stride in range(1, 5):
        _dense_config(**_RSBF, loop_rsbf_item_form="block", loop_rsbf_block_stride=stride)
    for stride in (None, 1, 2, 3):
        _dense_config(**_RSBF, loop_rsbf_item_form="nested", loop_rsbf_block_stride=stride)
    _dense_config(**_RSBF, loop_rsbf_item_form="whole_round")
    _dsv4_config(**_RSBF, loop_rsbf_item_form="block", loop_rsbf_block_stride=2)
    _dsv4_config(**_RSBF, loop_rsbf_item_form="block", loop_rsbf_block_stride=8)
    _dsv4_config(**_RSBF, loop_rsbf_item_form="nested", loop_rsbf_block_stride=7)
    _dsv4_config(
        **_RSBF,
        loop_rsbf_item_form="nested",
        loop_rsbf_block_stride=2,
        loop_rsbf_coda_recompute_mla_up_proj=True,
    )
    with pytest.raises(ValueError) as excinfo:
        _dsv4_config(**_RSBF, loop_rsbf_item_form="block", loop_rsbf_block_stride=9)
    assert _rsbf_reported(excinfo) == [guards.RSBF_BLOCK_STRIDE_BEYOND_CORE]


@pytest.mark.parametrize(
    "form, stride, core, expected",
    [
        ("whole_round", None, 8, []),
        ("whole_round", None, None, []),
        ("whole_round", 8, 8, [guards.RSBF_BLOCK_STRIDE_MISMATCH]),
        ("block", None, 8, [guards.RSBF_BLOCK_STRIDE_MISMATCH]),
        ("block", 0, None, [guards.RSBF_BLOCK_STRIDE_MISMATCH]),
        ("block", 1, 8, []),
        ("block", 3, 8, []),
        ("block", 8, 8, []),
        ("block", 9, 8, [guards.RSBF_BLOCK_STRIDE_BEYOND_CORE]),
        ("block", 9, None, []),  # core unknown at config time: checked by LoopProgram
        ("nested", None, 8, []),
        ("nested", 7, 8, []),
        ("nested", 8, 8, [guards.RSBF_BLOCK_STRIDE_BEYOND_CORE]),
        ("nested", None, 1, [guards.RSBF_BLOCK_STRIDE_BEYOND_CORE]),
        ("nested", 0, 8, [guards.RSBF_BLOCK_STRIDE_MISMATCH]),
        ("layer", 2, 8, [guards.RSBF_ITEM_FORM]),
    ],
)
def test_rsbf_item_form_rules(form, stride, core, expected):
    assert rsbf_item_form_violations(form, stride, core) == expected


def test_rsbf_item_form_range_rule_needs_the_layer_count():
    """Without num_layers the config cannot know the core; the program checks it later."""
    config = _violating(
        _dense_config(**_RSBF), num_layers=0, loop_rsbf_item_form="block", loop_rsbf_block_stride=99
    )
    guards.validate_rsbf_config(config)


def test_block_boundaries_of_a_pattern():
    """A cut is on a block boundary when an MLP / MoE sublayer ends before it and a mixer starts
    after it; the pattern's ends always are."""
    symbols = list("*-*-EH")
    assert [is_block_boundary(symbols, cut) for cut in range(len(symbols) + 1)] == [
        True,  # start
        False,  # '*' | '-'
        True,  # '-' | '*'
        False,  # '*' | '-'
        False,  # '-' | 'E': no mixer after the cut
        True,  # 'E' | 'H'
        True,  # end
    ]


def test_rsbf_config_is_validated_at_construction():
    with pytest.raises(ValueError) as excinfo:
        _dense_config(loop_schedule="rsbf")
    assert _rsbf_reported(excinfo) == [guards.RSBF_SUM_LOSS]
    with pytest.raises(ValueError) as excinfo:
        TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            loop_schedule="rsbf",
            calculate_per_token_loss=True,
        )
    assert _rsbf_reported(excinfo) == [guards.RSBF_WITHOUT_HYBRID_ROLES]


@pytest.mark.parametrize(
    "fields",
    [
        dict(loop_exit_criterion="sampled", loop_depth_probs=[0.25, 0.25, 0.25, 0.25]),
        dict(loop_depth_probs=[0.25, 0.25, 0.25, 0.25]),
        dict(loop_min_rounds=2),
        dict(loop_rsbf_waves=2),
        dict(loop_rsbf_fill="sample_order"),
        dict(loop_rsbf_rebalance=False),
        dict(loop_rsbf_move_transport="all_to_all"),
        dict(loop_rsbf_tail_item_moves=False),
        dict(loop_rsbf_over_budget="warn"),
        dict(loop_rsbf_item_form="nested"),
        dict(loop_rsbf_item_form="block", loop_rsbf_block_stride=2),
        dict(loop_rsbf_coda_recompute_mla_up_proj=True),
    ],
)
def test_rsbf_fields_require_the_rsbf_schedule(fields):
    with pytest.raises(ValueError, match="require loop_schedule='rsbf'") as excinfo:
        _dense_config(**fields)
    assert set(str(excinfo.value).split(" require")[0].split(", ")) == set(fields)
