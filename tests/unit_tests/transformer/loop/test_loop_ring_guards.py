# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Guards of the ring schedule (``loop_schedule='ring'``: the hybrid Huginn loop over pipeline
stages, megatron/core/transformer/loop/guards.py).

* Accepted: PP 2 / 4 / 8 with k = 1 / 2 / 4 core slices per stage, dense and DSv4 Flash-like
  (MLA, MoE with a hash-routed prelude, mHC, MTP), with P2P overlap; the recompute plan.
* Refused by rule name, each rule alone: every config rule of ``validate_loop_ring_config``
  (the ring without PP, at PP 2 without overlapped P2P, ...), every layout rule of
  ``validate_ring_layout``; the eager hybrid-loop rules, the DDP rules, the RSBF-only fields and
  the parameter store stay refused under the ring.
* Every other loop keeps its pipeline rules: '|' without the ring (``PIPE_IN_PATTERN``, also at
  argument time) and pipeline / virtual-pipeline parallelism without the ring
  (``PIPELINE_PARALLELISM``).
* The argument path: ``--loop-schedule ring`` accepts '|', derives the virtual pipeline size k
  from the segments (None at k = 1) and keeps ``overlap_p2p_comm``; the TransformerConfig built
  from the arguments refuses PP 2 without it.

No process groups: config, layout and argument rules only.
"""

import copy
import sys

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.transformer.loop import guards
from megatron.core.transformer.loop.guards import LoopRingChunk, LoopRoles
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.transformer.loop.ring_utils import (
    FLASH_MAIN,
    MINI_MAIN,
    dsv4_loop_config,
    ring_fields,
    ring_layout,
)

_RING_PREFIX = "Loop ring does not support: "
_HYBRID_PREFIX = "Hybrid loop does not support: "
# Dense: prelude '*-', a core of `blocks` '*-' blocks, coda '*-'.
_DENSE_CORE_BLOCKS = 32


def _dense_main(core_blocks=_DENSE_CORE_BLOCKS):
    return "*-" * (core_blocks + 2)


def _dense_ring(stages=2, chunks=1, core_blocks=_DENSE_CORE_BLOCKS, **overrides):
    """A dense hybrid loop config on the ring (prelude 2, coda 2, R = 4)."""
    fields = dict(
        num_layers=len(_dense_main(core_blocks)),
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        loop_architecture="huginn",
        loop_iterations=4,
        loop_prelude_sublayers=2,
        loop_coda_sublayers=2,
        **ring_fields(stages, chunks),
    )
    fields.update(overrides)
    return TransformerConfig(**fields)


def _violating(config, **fields):
    """A copy of a valid config with ``fields`` set directly (no ``__post_init__``)."""
    config = copy.copy(config)
    for name, value in fields.items():
        assert hasattr(config, name), name
        setattr(config, name, value)
    return config


def _reported(excinfo, prefix):
    message = str(excinfo.value)
    assert message.startswith(prefix), message
    return message[len(prefix) :].split(", ")


# ---------------------------------------------------------------------------------------------
# Accepted rings
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("chunks", [1, 2, 4])
@pytest.mark.parametrize("stages", [2, 4, 8])
def test_dense_ring_is_accepted(stages, chunks):
    config = _dense_ring(stages, chunks)
    assert guards.is_ring_loop_config(config)
    guards.validate_loop_ring_config(config)
    guards.validate_hybrid_loop_config(config)  # no PIPELINE_PARALLELISM under the ring
    guards.validate_loop_ddp_config(config, DistributedDataParallelConfig())
    piped, segments = ring_layout(_dense_main(), 2, 2, stages, chunks)
    guards.reject_pipe_in_loop_pattern("huginn", piped, "ring")
    guards.validate_hybrid_loop_pattern(config, piped, None)
    chunks_found = guards.validate_ring_layout(config, piped, None)
    assert [c.offset for c in chunks_found] == [s["offset"] for s in segments]
    assert len(chunks_found) == stages * chunks


@pytest.mark.parametrize("chunks", [1, 2, 4])
@pytest.mark.parametrize("stages", [2, 4, 8])
def test_dsv4_flash_ring_is_accepted(stages, chunks):
    """The Flash pattern (MLA / DSv4, MoE with the hash-routed prelude, mHC 4, MTP)."""
    config = dsv4_loop_config(FLASH_MAIN, **ring_fields(stages, chunks))
    guards.validate_loop_ring_config(config)
    guards.validate_hybrid_loop_config(config)
    piped, _ = ring_layout(FLASH_MAIN, 6, 4, stages, chunks)
    guards.validate_hybrid_loop_pattern(config, piped, "WE")
    chunks_found = guards.validate_ring_layout(config, piped, 6)
    assert len(chunks_found) == stages * chunks


def test_ring_config_is_validated_at_construction():
    with pytest.raises(ValueError) as excinfo:
        _dense_ring(pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None)
    assert _reported(excinfo, _RING_PREFIX) == [guards.RING_WITHOUT_PP]
    with pytest.raises(ValueError) as excinfo:
        _dense_ring(2, 1, overlap_p2p_comm=False, batch_p2p_comm=True)
    assert _reported(excinfo, _RING_PREFIX) == [guards.RING_P2_WITHOUT_OVERLAP]
    # P > 2 runs with batched P2P (the interleaved schedule's rule).
    _dense_ring(4, 1, overlap_p2p_comm=False, batch_p2p_comm=True)
    with pytest.raises(ValueError) as excinfo:
        TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            loop_schedule="ring",
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            overlap_p2p_comm=True,
            batch_p2p_comm=False,
        )
    assert _reported(excinfo, _RING_PREFIX) == [guards.RING_WITHOUT_HYBRID_ROLES]


# ---------------------------------------------------------------------------------------------
# Config rules (validate_loop_ring_config), each alone
# ---------------------------------------------------------------------------------------------

_RING_RULES = [
    (guards.RING_WITHOUT_HYBRID_ROLES, dict(loop_prelude_sublayers=None, loop_coda_sublayers=None)),
    (guards.RING_WITHOUT_HYBRID_ROLES, dict(loop_architecture="none")),
    (
        guards.RING_WITHOUT_PP,
        dict(pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None),
    ),
    (guards.RING_MOR, dict(loop_architecture="mor")),
    (guards.RING_CARRIER, dict(loop_carrier="residual")),
    (guards.RING_CARRIER, dict(loop_carrier="attn_res")),
    (guards.RING_UNTIED_CORE, dict(loop_untie_core=True)),
    (guards.RING_SAMPLE_DEPTH, dict(loop_sample_depth=True)),
    (guards.RING_TRUNCATED_BPTT, dict(loop_bptt_iterations=1)),
    (guards.RING_TP, dict(tensor_model_parallel_size=2)),
    (guards.RING_STATIC_BUDGET_MOE, dict(moe_expert_rank_capacity_factor=1.0)),
    (guards.RING_DEFER_EMBEDDING_WGRAD, dict(defer_embedding_wgrad_compute=True)),
    (guards.RING_PIPELINE_LAYOUT, dict(pipeline_model_parallel_layout=object())),
    (guards.RING_P2_WITHOUT_OVERLAP, dict(overlap_p2p_comm=False, batch_p2p_comm=True)),
    # P = 2, k = 1, R = 4: two groups of four codes.
    (guards.RING_RECOMPUTE_PLAN, dict(loop_recompute=True, loop_ring_recompute_plan="CCCC")),
    (guards.RING_RECOMPUTE_PLAN, dict(loop_recompute=True, loop_ring_recompute_plan="CCCC/CCC")),
    (guards.RING_RECOMPUTE_PLAN, dict(loop_recompute=True, loop_ring_recompute_plan="CCCC/CCCx")),
    (guards.RING_RECOMPUTE_PLAN, dict(loop_ring_recompute_plan="nnnn/nnnn/nnnn")),
    (guards.RING_PLAN_WITHOUT_RECOMPUTE, dict(loop_ring_recompute_plan="nnnn/nnnC")),
]


@pytest.mark.parametrize(
    "rule,fields", _RING_RULES, ids=[f"{rule}-{'-'.join(fields)}" for rule, fields in _RING_RULES]
)
def test_ring_rule_is_reported_alone(rule, fields):
    """Each rule on a copy of an accepted ring (dense at P 2, k 1; DSv4 for the MoE rule) that
    differs in that rule's fields only."""
    moe = any(name.startswith("moe_") for name in fields)
    base = dsv4_loop_config(FLASH_MAIN, **ring_fields(2, 1)) if moe else _dense_ring(2, 1)
    config = _violating(base, **fields)
    with pytest.raises(ValueError) as excinfo:
        guards.validate_loop_ring_config(config)
    assert _reported(excinfo, _RING_PREFIX) == [rule]


def test_ring_rules_collect_every_violation():
    config = _violating(_dense_ring(2, 1), loop_sample_depth=True, tensor_model_parallel_size=2)
    with pytest.raises(ValueError) as excinfo:
        guards.validate_loop_ring_config(config)
    assert set(_reported(excinfo, _RING_PREFIX)) == {guards.RING_SAMPLE_DEPTH, guards.RING_TP}


def test_recompute_plans():
    """The plan: P groups of R x k codes ('C' needs loop_recompute); unset, every view 'C' with
    loop_recompute and 'n' without. Built through __post_init__."""
    assert guards.resolve_loop_ring_recompute_plan(_dense_ring(2, 1)) == ("nnnn", "nnnn")
    assert guards.resolve_loop_ring_recompute_plan(_dense_ring(4, 2, loop_recompute=True)) == (
        ("C" * 8,) * 4
    )
    plan = "CCCCCCCC/CCCCCCCn"
    config = _dense_ring(2, 2, loop_recompute=True, loop_ring_recompute_plan=plan)
    assert guards.resolve_loop_ring_recompute_plan(config) == ("CCCCCCCC", "CCCCCCCn")
    # An all-'n' plan without loop_recompute is the default made explicit.
    _dense_ring(2, 1, loop_ring_recompute_plan="nnnn/nnnn")
    with pytest.raises(ValueError) as excinfo:
        _dense_ring(2, 2, loop_recompute=True, loop_ring_recompute_plan="CCCC/CCCC")
    assert _reported(excinfo, _RING_PREFIX) == [guards.RING_RECOMPUTE_PLAN]
    with pytest.raises(ValueError, match="loop_ring_recompute_plan requires loop_schedule='ring'"):
        _dense_ring(
            loop_schedule="standard",
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            loop_ring_recompute_plan="nnnn",
        )


# ---------------------------------------------------------------------------------------------
# What stays refused under the ring (the eager loop's, DDP, RSBF and parameter-store guards)
# ---------------------------------------------------------------------------------------------

_HYBRID_RULES_UNDER_RING = [
    (guards.CONTEXT_PARALLELISM, dict(context_parallel_size=2)),
    (guards.SEQUENCE_PARALLELISM, dict(sequence_parallel=True)),
    (guards.CUDA_GRAPHS, dict(cuda_graph_impl="transformer_engine")),
    (guards.MOE_EP_OVERLAP, dict(overlap_moe_expert_parallel_comm=True)),
    (guards.ACTIVATION_OFFLOAD, dict(cpu_offloading=True)),
    (guards.ACTIVATION_OFFLOAD, dict(fine_grained_activation_offloading=True)),
    (guards.PAGED_STASH, dict(moe_paged_stash=True)),
    (guards.OTHER_RECOMPUTE, dict(recompute_granularity="full")),
    (guards.DROPOUT, dict(hidden_dropout=0.1)),
]


@pytest.mark.parametrize(
    "rule,fields",
    _HYBRID_RULES_UNDER_RING,
    ids=[f"{rule}-{'-'.join(fields)}" for rule, fields in _HYBRID_RULES_UNDER_RING],
)
def test_eager_loop_rules_still_apply_under_the_ring(rule, fields):
    """The ring lifts only the pipeline rule: each eager-loop rule is still reported, alone."""
    config = _violating(_dense_ring(4, 2), **fields)
    with pytest.raises(ValueError) as excinfo:
        guards.validate_hybrid_loop_config(config)
    assert _reported(excinfo, _HYBRID_PREFIX) == [rule]


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
def test_ddp_overlap_stays_refused_under_the_ring(rule, ddp_fields):
    with pytest.raises(ValueError) as excinfo:
        guards.validate_loop_ddp_config(
            _dense_ring(2, 1), DistributedDataParallelConfig(**ddp_fields)
        )
    assert _reported(excinfo, _HYBRID_PREFIX) == [rule]


def test_rsbf_fields_and_the_parameter_store_stay_refused_under_the_ring():
    with pytest.raises(ValueError, match="require loop_schedule='rsbf'"):
        _dense_ring(2, 1, loop_rsbf_waves=2)
    with pytest.raises(ValueError, match="require loop_schedule='rsbf'"):
        _dense_ring(2, 1, loop_exit_criterion="sampled", loop_depth_probs=[0.25] * 4)
    with pytest.raises(ValueError) as excinfo:
        guards.validate_param_store_config(_dense_ring(2, 1, calculate_per_token_loss=True))
    message = str(excinfo.value)
    assert guards.STORE_WITHOUT_RSBF in message and guards.STORE_PIPELINE in message


def test_carrier_diagnostics_arguments_refuse_the_ring():
    from argparse import Namespace

    from megatron.training.arguments import _validate_loop_carrier_diagnostics

    args = Namespace(
        loop_carrier_diagnostics="summary",
        loop_carrier_diagnostics_path=None,
        loop_carrier_diagnostics_segments=None,
        loop_architecture="huginn",
        loop_prelude_sublayers=2,
        loop_coda_sublayers=2,
        loop_schedule="ring",
        loop_sample_depth=False,
    )
    with pytest.raises(ValueError, match="needs an eager hybrid Huginn loop"):
        _validate_loop_carrier_diagnostics(args)


# ---------------------------------------------------------------------------------------------
# Every other loop keeps its pipeline rules
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("schedule", ["standard", "rsbf"])
def test_pipeline_parallelism_needs_the_ring(schedule):
    base = _dense_ring(2, 2)
    for fields in (
        dict(pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=None),
        dict(pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=2),
    ):
        config = _violating(base, loop_schedule=schedule, **fields)
        with pytest.raises(ValueError) as excinfo:
            guards.validate_hybrid_loop_config(config)
        assert _reported(excinfo, _HYBRID_PREFIX) == [guards.PIPELINE_PARALLELISM]
        # The same parallel layout under the ring passes the eager rules.
        guards.validate_hybrid_loop_config(_violating(config, loop_schedule="ring"))


@pytest.mark.parametrize("schedule", ["standard", "rsbf"])
def test_pipe_needs_the_ring(schedule):
    piped = "*-*-|*-*-"
    config = _violating(_dense_ring(2, 1, core_blocks=2), loop_schedule=schedule)
    with pytest.raises(ValueError) as excinfo:
        guards.validate_hybrid_loop_pattern(config, piped, None)
    assert _reported(excinfo, _HYBRID_PREFIX) == [guards.PIPE_IN_PATTERN]
    with pytest.raises(ValueError) as excinfo:
        guards.reject_pipe_in_loop_pattern("huginn", piped + "/*-", schedule)
    assert _reported(excinfo, _HYBRID_PREFIX) == [guards.PIPE_IN_PATTERN]
    # The ring accepts it (its layout rules are validate_ring_layout's).
    guards.validate_hybrid_loop_pattern(_dense_ring(2, 1, core_blocks=2), piped, None)
    guards.reject_pipe_in_loop_pattern("huginn", piped + "/*-", "ring")
    # The default schedule argument is the standard one.
    with pytest.raises(ValueError):
        guards.reject_pipe_in_loop_pattern("huginn", piped)


# ---------------------------------------------------------------------------------------------
# Layout rules (validate_ring_layout), each alone
# ---------------------------------------------------------------------------------------------

# A dense ring with a core of 4 blocks: '*-' | '*-*-*-*-' | '*-' (prelude 2, coda 2), 12
# sublayers.
_SMALL = dict(core_blocks=4)

_LAYOUT_RULES = [
    # (rules, pattern, P, k, config overrides)
    ({guards.RING_SEGMENTS}, "*-*-*-*-*-*-", 2, 1, {}),  # the ring without '|'
    ({guards.RING_SEGMENTS}, "*-*-|*-*-|*-*-", 2, 1, {}),  # 3 segments at P 2, k 1
    ({guards.RING_SEGMENTS}, "*-*-*-|*-*-*-", 2, 2, {}),  # VPP 2 needs 4 segments
    ({guards.RING_SEGMENTS}, "*-*-|*-|*-|*-*-", 3, 1, {}),  # 4 segments at P 3
    ({guards.RING_NUM_LAYERS}, "*-*-*-|*-*-*-", 2, 1, dict(num_layers=14)),
    ({guards.RING_CUT_NOT_ON_BLOCK}, "*-*-*|-*-*-*-", 2, 1, {}),
    ({guards.RING_SEGMENT_WITHOUT_CORE}, "*-|*-*-*-*-*-", 2, 1, {}),  # segment 0 = prelude
    ({guards.RING_SEGMENT_WITHOUT_CORE}, "*-*-*-*-*-|*-", 2, 1, {}),  # last = coda
    ({guards.RING_SEGMENT_WITHOUT_CORE}, "*-*-*-||*-*-*-", 3, 1, {}),  # an empty segment
    # Prelude 4: segment 0 holds half of it (and no core).
    (
        {guards.RING_PRELUDE_SPLIT, guards.RING_SEGMENT_WITHOUT_CORE},
        "*-|*-*-*-*-*-",
        2,
        1,
        dict(loop_prelude_sublayers=4),
    ),
    # Coda 4: the last segment holds half of it (and no core).
    (
        {guards.RING_CODA_SPLIT, guards.RING_SEGMENT_WITHOUT_CORE},
        "*-*-*-*-*-|*-",
        2,
        1,
        dict(loop_coda_sublayers=4),
    ),
]


@pytest.mark.parametrize(
    "rules,pattern,stages,chunks,overrides",
    _LAYOUT_RULES,
    ids=[f"{pattern}-P{p}k{k}" for _, pattern, p, k, _ in _LAYOUT_RULES],
)
def test_ring_layout_rule_is_reported(rules, pattern, stages, chunks, overrides):
    config = _violating(_dense_ring(stages, chunks, **_SMALL), **overrides)
    with pytest.raises(ValueError) as excinfo:
        guards.validate_ring_layout(config, pattern, None)
    assert set(_reported(excinfo, _RING_PREFIX)) == rules


def test_ring_layout_applies_the_loop_role_rules_first():
    """The pipe-free pattern passes resolve_hybrid_loop_roles first: a role cut inside a block,
    a hash-routed MoE layer beyond the prelude (which would sit on a stage without the token
    ids), an unsupported sublayer."""
    config = _violating(_dense_ring(2, 1, **_SMALL), loop_prelude_sublayers=3)
    with pytest.raises(ValueError) as excinfo:
        guards.validate_ring_layout(config, "*-*-*-|*-*-*-", None)
    assert _reported(excinfo, _HYBRID_PREFIX) == [guards.CUT_NOT_ON_BLOCK]
    moe = "*E*E*E|*E*E*E"
    with pytest.raises(ValueError) as excinfo:
        guards.validate_ring_layout(_dense_ring(2, 1, **_SMALL), moe, 4)
    assert _reported(excinfo, _HYBRID_PREFIX) == [guards.HASH_BEYOND_PRELUDE]
    guards.validate_ring_layout(_dense_ring(2, 1, **_SMALL), moe, 2)
    with pytest.raises(ValueError) as excinfo:
        guards.validate_ring_layout(_dense_ring(2, 1, **_SMALL), "*-*-*-|M-*-*-", None)
    assert _reported(excinfo, _HYBRID_PREFIX) == [guards.UNSUPPORTED_SUBLAYERS]


def test_ring_layout_chunks():
    """The chunks of a valid layout: the model roles split by segment."""
    config = _dense_ring(2, 2, **_SMALL)
    chunks = guards.validate_ring_layout(config, "*-*-|*-|*-|*-*-", None)
    assert chunks == [
        LoopRingChunk(0, LoopRoles(2, 2, 0), round_start=True, round_end=False),
        LoopRingChunk(4, LoopRoles(0, 2, 0), round_start=False, round_end=False),
        LoopRingChunk(6, LoopRoles(0, 2, 0), round_start=False, round_end=False),
        LoopRingChunk(8, LoopRoles(0, 2, 2), round_start=False, round_end=True),
    ]


# ---------------------------------------------------------------------------------------------
# The argument path (parse_args / validate_args, no process groups)
# ---------------------------------------------------------------------------------------------


def _parse(monkeypatch, pattern, stages, *extra):
    """``parse_args`` + ``validate_args`` of a small dense hybrid loop on ``stages`` ranks."""
    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.global_vars import destroy_global_vars

    destroy_global_vars()
    argv = [
        "pretrain_hybrid.py",
        "--hybrid-layer-pattern", pattern,
        "--loop-architecture", "huginn",
        "--loop-prelude-sublayers", "2",
        "--loop-coda-sublayers", "2",
        "--pipeline-model-parallel-size", str(stages),
        "--hidden-size", "64",
        "--num-attention-heads", "4",
        "--seq-length", "16",
        "--max-position-embeddings", "16",
        "--micro-batch-size", "1",
        "--global-batch-size", "8",
        "--tokenizer-type", "NullTokenizer",
        "--vocab-size", "128",
        "--attention-dropout", "0",
        "--hidden-dropout", "0",
        "--bf16",
        *extra,
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    for name, value in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", str(stages))):
        monkeypatch.setenv(name, value)
    args = parse_args()
    validate_args(args, {})
    return args


def test_arguments_take_the_ring_layout(monkeypatch):
    from megatron.training.arguments import core_transformer_config_from_args

    pattern = "*-*-*-|*-*-*-"
    # k = 1: no virtual pipeline in the arguments; the ring keeps overlap_p2p_comm (on).
    args = _parse(monkeypatch, pattern, 2, "--loop-schedule", "ring")
    assert args.virtual_pipeline_model_parallel_size is None
    assert args.overlap_p2p_comm is True and args.loop_schedule == "ring"
    config = core_transformer_config_from_args(args)
    assert config.loop_schedule == "ring" and config.overlap_p2p_comm and not config.batch_p2p_comm
    assert config.loop_ring_recompute_plan is None
    # k = 2: the segments give virtual pipeline size 2.
    args = _parse(monkeypatch, "*-*-|*-|*-|*-*-", 2, "--loop-schedule", "ring")
    assert args.virtual_pipeline_model_parallel_size == 2 and args.overlap_p2p_comm
    # A plan reaches the config.
    args = _parse(
        monkeypatch,
        pattern,
        2,
        "--loop-schedule",
        "ring",
        "--loop-recompute",
        "--loop-ring-recompute-plan",
        "CCCC/CCCn",
    )
    assert core_transformer_config_from_args(args).loop_ring_recompute_plan == "CCCC/CCCn"
    # PP 2 without overlapped P2P: the arguments pass it through, the config refuses it.
    args = _parse(
        monkeypatch, pattern, 2, "--loop-schedule", "ring", "--no-overlap-p2p-communication"
    )
    assert args.overlap_p2p_comm is False
    with pytest.raises(ValueError) as excinfo:
        core_transformer_config_from_args(args)
    assert _reported(excinfo, _RING_PREFIX) == [guards.RING_P2_WITHOUT_OVERLAP]


def test_arguments_refuse_pipes_without_the_ring(monkeypatch):
    with pytest.raises(ValueError) as excinfo:
        _parse(monkeypatch, "*-*-*-|*-*-*-", 2)
    assert _reported(excinfo, _HYBRID_PREFIX) == [guards.PIPE_IN_PATTERN]
    # Without the ring and without '|', a non-interleaved PP run still loses P2P overlap (the
    # loop guards then refuse its pipeline parallelism when the model is built).
    args = _parse(monkeypatch, "*-*-*-*-*-*-", 2)
    assert args.overlap_p2p_comm is False and args.virtual_pipeline_model_parallel_size is None


def test_mini_pattern_layouts_are_valid():
    """The parity mini's ring layouts of the design (VERIFY.md) pass validate_ring_layout."""
    for stages, chunks, expected in (
        (2, 1, "WEWECEHECEHECE|HECEHECEHECE"),
        (2, 2, "WEWECEHECE|HECE|HECE|HECEHECE"),
        (4, 1, "WEWECEHECE|HECE|HECE|HECEHECE"),
    ):
        piped, _ = ring_layout(MINI_MAIN, 6, 4, stages, chunks)
        assert piped == expected
        config = dsv4_loop_config(MINI_MAIN, **ring_fields(stages, chunks))
        guards.validate_ring_layout(config, piped, 6)


def test_ring_rule_names_never_contain_the_separator():
    """A ring refusal joins its rule names with ", " (and the ring tests split on it): no ring
    rule name holds it."""
    rules = {
        name: value
        for name, value in vars(guards).items()
        if name.startswith("RING_") and isinstance(value, str)
    }
    assert {"RING_EAGER_CALL", "RING_CALL_ROLES", "RING_WITHOUT_PP"} <= set(rules)
    assert [name for name, value in rules.items() if ", " in value] == []
