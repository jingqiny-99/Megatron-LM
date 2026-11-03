# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Configuration and input guards for the hybrid (HybridStack) loop.

A hybrid loop splits the main hybrid layer pattern into three roles, counted in sublayers
(pattern symbols): a prelude of ``loop_prelude_sublayers`` sublayers that runs once, a core
that runs once per round, and a coda of ``loop_coda_sublayers`` sublayers that runs once.
The core is whatever remains in between.

Each ``validate_*`` / ``resolve_*`` function checks one group of rules, collects every
violation of that group, and raises a single ``ValueError`` that lists the rule names
(``"Hybrid loop does not support: <name>, <name>"``). The rule names are stable; tests and
users can match on them.

Where each group runs:

* ``validate_hybrid_loop_pattern``: ``HybridModel.__init__``, right after the pattern is parsed.
* ``reject_pipe_in_loop_pattern``: argument validation, before ``|`` segments set VPP.
* ``validate_loop_stack_construction``, ``validate_hybrid_loop_config`` and
  ``resolve_hybrid_loop_roles``: the loop stack's constructor (the stack owns the layer list
  the roles index into).
* ``validate_loop_carrier``: ``TransformerConfig.__post_init__`` and the loop stack
  constructors; ``validate_loop_carrier_stack``: the loop stack constructors.
* ``validate_loop_decoder``: ``HybridModel.__init__``, after the decoder is built.
* ``validate_hybrid_loop_inputs``: the start of the loop stack's layer loop, every forward.
* ``validate_loop_ddp_config``: ``DistributedDataParallel.__init__``.
* ``validate_loop_fsdp_config`` (``"Loop with Megatron-FSDP does not support: ..."``): the
  Megatron-FSDP v1 wrapper's constructor (``FullyShardedDataParallel``) and argument validation;
  ``validate_loop_row_depths``: ``LoopHybridStack.set_loop_depths``.
* ``validate_rsbf_config`` (``"Loop RSBF does not support: ..."``):
  ``TransformerConfig.__post_init__`` with ``loop_schedule='rsbf'``, and the RSBF schedule.
* ``validate_rsbf_data_parallel``: the RSBF executor, every step, on the model it runs.

The role-field checks themselves (both set or neither, nonnegative, non-empty core) and the
quantized-loop weight-cache rule are in ``TransformerConfig.__post_init__``.

Block and edge ``loop_recompute`` (``loop_recompute_block_stride`` / ``loop_recompute_edges``):
the prerequisites and the MoR / ``attn_res`` refusals are config rules
(``validate_hybrid_loop_config``), the stride's range is a role rule
(``resolve_hybrid_loop_roles``: it needs the round's sublayer count), and
``RECOMPUTE_BLOCKS_WITHOUT_HYBRID_LOOP`` (the fields on a GPT loop or without a loop) is raised
by ``TransformerConfig.__post_init__``.

This module is imported lazily by its callers; keep its imports light.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

from megatron.core.models.hybrid.layers.utils import Symbols

if TYPE_CHECKING:
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.transformer_config import TransformerConfig

# Rule names, as they appear in the error message.
PIPE_IN_PATTERN = "'|' in hybrid_layer_pattern"
OVERLAP_GRAD_REDUCE = "overlap_grad_reduce"
OVERLAP_PARAM_GATHER = "overlap_param_gather"
DELAYED_FP8_SCALING = "delayed FP8 scaling"
CUDA_GRAPHS = "cuda_graph_impl"
MHC_RECOMPUTE = "mHC recompute"
MOE_EP_OVERLAP = "MoE EP overlap"
PAGED_STASH = "paged stash"
HYBRIDEP_PADDING = "moe_hybridep_pad_variable_tokens"
SHORTCUT_MOE = "Shortcut MoE"
RESIDUAL_STREAM_RECOMPUTE = "residual-stream recompute"
PACKED_MULTI_SEQUENCE = "packed THD with more than one sequence per row"
HASH_BEYOND_PRELUDE = "hash MoE layer beyond the prelude"
MLA_WITHOUT_DSV4 = "MLA without experimental_attention_variant=dsv4_hybrid"
CUT_NOT_ON_BLOCK = "role cut not on a block boundary"
ROLES_DO_NOT_MATCH = "role fields do not match the pattern"
PIPELINE_PARALLELISM = "pipeline parallelism"
CONTEXT_PARALLELISM = "context parallelism"
SEQUENCE_PARALLELISM = "sequence parallelism"
DROPOUT = "dropout / MoE input jitter"
UNTAPED_RECOMPUTE_ROUTING = (
    "loop_recompute with MoE routing its recompute does not replay (forced load balancing / "
    "moe_router_force_biased / routing replay / Sinkhorn / quantile balancing)"
)
FP4 = "FP4"
ACTIVATION_OFFLOAD = "activation offload"
OTHER_RECOMPUTE = "activation recompute other than loop_recompute"
GTP_WEIGHT_REMAT = "GTP weight remat"
WIDE_RESIDUAL = "wide residual"
DSA_OVER_GQA = (
    "DSA over GQA (experimental_attention_variant='dsa' without MLA: its indexer has no "
    "padding / exit row mask)"
)
UNSUPPORTED_SUBLAYERS = "Mamba / GDN / KDA / plain MLA sublayers"
MOR_NON_DENSE_CORE = "MoR on a non-dense core"
MASKED_MOR_ON_DSV4 = "masked MoR on DSv4 / MLA attention"
INFERENCE = "inference"
MTP_LOOP_STACK = "MTP stack built as a loop stack"
LOOP_STACK_WITHOUT_LOOP = "LoopHybridStack without loop_architecture"
NON_LOOP_DECODER = "loop config with a non-loop stack"
MTP_DS_ATTENTION = "'D' sublayer in the MTP pattern (no FLOP row; use W / C / H)"
SBHD_MASK_WITH_DSV4 = (
    "explicit attention mask with DSv4 attention on SBHD "
    "(set no_create_attention_mask_in_dataloader)"
)
# Residual carriers (``loop_carrier``; ``validate_loop_carrier``, the stack constructors).
CUDA_GRAPHS_ATTN_RES = (
    "cuda_graph_impl with loop_carrier='attn_res' (the source count grows every round; no "
    "graph path)"
)
ATTN_RES_WITH_MHC = "enable_mhc_connections with loop_carrier='attn_res' (two residual carriers)"
ATTN_RES_WITH_MTP = "MTP with loop_carrier='attn_res' (MTP trunk sources not ported)"
ATTN_RES_TRUNCATED_BPTT = "loop_bptt_iterations with loop_carrier='attn_res'"
ATTN_RES_FP32_RESIDUAL = (
    "fp32_residual_connection with loop_carrier='attn_res' (MLP deltas would be fp32, "
    "attention deltas params_dtype: the sources must share one dtype)"
)
ATTN_RES_BLOCKS = "loop_attn_res_blocks_per_round does not cut the core into equal blocks"
RESIDUAL_WITH_MHC = (
    "enable_mhc_connections with loop_carrier='residual' (the residual carrier is the "
    "attn_res w = 0 reference only)"
)
# The hybrid untied core (``loop_untie_core`` in role mode, the ablation's 'plain-inject').
UNTIED_CORE_CARRIER = "loop_untie_core with loop_carrier other than 'injection'"
UNTIED_CORE_SEGMENTS = "loop_untie_core core not R equal segments cut on block boundaries"
CARRIER_STACK_MISMATCH = (
    "loop_carrier does not match the loop stack class (use "
    "loop_attn_res_hybrid_*_stack_spec for 'attn_res', loop_hybrid_*_stack_spec otherwise)"
)
# Block and edge checkpoints of loop_recompute (``loop_recompute_block_stride`` /
# ``loop_recompute_edges``; ``LoopHybridStack._checkpointed_blocks``).
RECOMPUTE_BLOCKS_WITHOUT_RECOMPUTE = (
    "loop_recompute_block_stride / loop_recompute_edges without loop_recompute"
)
RECOMPUTE_EDGES_WITHOUT_STRIDE = "loop_recompute_edges without loop_recompute_block_stride"
RECOMPUTE_STRIDE_RANGE = (
    "loop_recompute_block_stride outside 1..the sublayers of one round (the core or one untied "
    "segment)"
)
RECOMPUTE_BLOCKS_MOR = (
    "loop_recompute_block_stride with MoR (its recurrence keeps one checkpoint per round)"
)
RECOMPUTE_BLOCKS_ATTN_RES = (
    "loop_recompute_block_stride with loop_carrier='attn_res' (LoopAttnResHybridStack "
    "checkpoints whole rounds)"
)
# Raised by TransformerConfig.__post_init__ ("Loop does not support: ..."): the fields would be
# ignored silently by a GPT loop or a model without a loop.
RECOMPUTE_BLOCKS_WITHOUT_HYBRID_LOOP = (
    "loop_recompute_block_stride / loop_recompute_edges without a hybrid loop "
    "(loop_architecture with loop_prelude_sublayers / loop_coda_sublayers)"
)

# Sublayer symbols that begin a block (a sequence mixer) or end one (an MLP / MoE). The
# mixers rejected by UNSUPPORTED_SUBLAYERS are included so that rule alone reports them.
_BLOCK_START = frozenset(
    (
        Symbols.ATTENTION,
        Symbols.DS_ATTENTION,
        Symbols.CSA,
        Symbols.HCA,
        Symbols.WINDOW,
        Symbols.MLA,
        Symbols.MAMBA,
        Symbols.GDN,
        Symbols.KDA,
    )
)
_BLOCK_END = frozenset((Symbols.MLP, Symbols.MOE))
# Sublayers the stage-1 loop has no FLOP row or plan use for.
_UNSUPPORTED_SYMBOLS = frozenset((Symbols.MAMBA, Symbols.GDN, Symbols.KDA, Symbols.MLA))
# Symbols MoR may revisit: plain attention, dense MLP and MoE.
_MOR_CORE_SYMBOLS = frozenset((Symbols.ATTENTION, Symbols.MLP, Symbols.MOE))
# Symbols built with a DSv4 attention (CSA / HCA / window, and DSA under dsv4_hybrid).
_DSV4_ATTENTION_SYMBOLS = frozenset(
    (Symbols.DS_ATTENTION, Symbols.CSA, Symbols.HCA, Symbols.WINDOW)
)
# The recompute modules that have their own loop rule (MHC_RECOMPUTE, RESIDUAL_STREAM_RECOMPUTE).
_LOOP_RULED_RECOMPUTE_MODULES = frozenset(("mhc", "residual_stream"))
# TransformerConfig.__post_init__ replaces recompute_modules=None by this list.
_DEFAULT_RECOMPUTE_MODULES = ("core_attn",)


@dataclass(frozen=True)
class LoopRoles:
    """Prelude, core and coda sizes of a hybrid loop, in sublayers of the main pattern."""

    prelude: int
    core: int
    coda: int

    @property
    def core_start(self) -> int:
        """Index of the first core sublayer."""
        return self.prelude

    @property
    def core_end(self) -> int:
        """Index one past the last core sublayer."""
        return self.prelude + self.core


def _raise_if_any(violations: List[str]) -> None:
    if violations:
        raise ValueError("Hybrid loop does not support: " + ", ".join(violations))


def _main_pattern(pattern: Optional[str]) -> str:
    return (pattern or "").split(Symbols.MTP_SEPARATOR)[0]


def is_hybrid_loop_config(config: Any) -> bool:
    """True for a loop config with hybrid role fields (the hybrid loop's role mode)."""
    return (
        getattr(config, "loop_architecture", "none") != "none"
        and getattr(config, "loop_prelude_sublayers", None) is not None
        and getattr(config, "loop_coda_sublayers", None) is not None
    )


def has_dsv4_attention(layer_symbols: Sequence[str]) -> bool:
    """True if any sublayer is built with a DSv4 attention (CSA / HCA / window / DSA)."""
    return any(symbol in _DSV4_ATTENTION_SYMBOLS for symbol in layer_symbols)


def is_block_boundary(layer_symbols: Sequence[str], cut: int) -> bool:
    """True if a cut before sublayer ``cut`` separates whole attention + MLP blocks: the
    sublayer before it ends a block (MLP / MoE) and the one after it starts one (a sequence
    mixer). The ends of the pattern (``cut`` 0 or ``len(layer_symbols)``) are boundaries."""
    if not 0 < cut < len(layer_symbols):
        return True
    return layer_symbols[cut - 1] in _BLOCK_END and layer_symbols[cut] in _BLOCK_START


def _recompute_modules(config: Any) -> Sequence[str]:
    modules = getattr(config, "recompute_modules", None)
    return _DEFAULT_RECOMPUTE_MODULES if modules is None else modules


def validate_loop_stack_construction(config: "TransformerConfig", is_mtp_layer: bool) -> None:
    """Reject building a loop stack without a loop, or as an MTP depth's nested stack."""
    unsupported = {
        # A loop stack that silently ran its layers once would hide a config error.
        LOOP_STACK_WITHOUT_LOOP: config.loop_architecture == "none",
        # MTP depths build their own plain HybridStack; only the main decoder loops.
        MTP_LOOP_STACK: is_mtp_layer,
    }
    _raise_if_any([name for name, value in unsupported.items() if value])


def validate_loop_decoder(config: "TransformerConfig", decoder: Any) -> None:
    """Reject a loop config whose decoder is not a loop stack (it would run no loop)."""
    from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack

    if config.loop_architecture != "none" and not isinstance(decoder, LoopHybridStack):
        _raise_if_any([NON_LOOP_DECODER])


def validate_loop_carrier(config: "TransformerConfig") -> None:
    """Reject features the non-default residual carriers do not support (``loop_carrier``).

    ``'attn_res'`` (Kimi-K3-style attention residuals, ``LoopAttnResHybridStack``): no mHC (a
    second residual carrier), no MTP (the PR's MTP trunk sources are not ported), full BPTT
    only, no ``fp32_residual_connection`` (the layer-delta path would return fp32 MLP deltas
    and ``params_dtype`` attention deltas; the aggregation needs one dtype). ``'residual'``
    (the ``w = 0`` reference of ``'attn_res'``): no mHC. Called by
    ``TransformerConfig.__post_init__`` and the loop stack constructors; the carriers' role
    and architecture rules (hybrid Huginn only) are in the config.
    """
    carrier = getattr(config, "loop_carrier", "injection")
    unsupported = {
        ATTN_RES_WITH_MHC: carrier == "attn_res" and bool(config.enable_mhc_connections),
        ATTN_RES_WITH_MTP: carrier == "attn_res" and bool(config.mtp_num_layers),
        ATTN_RES_TRUNCATED_BPTT: carrier == "attn_res" and config.loop_bptt_iterations > 0,
        ATTN_RES_FP32_RESIDUAL: carrier == "attn_res" and bool(config.fp32_residual_connection),
        RESIDUAL_WITH_MHC: carrier == "residual" and bool(config.enable_mhc_connections),
    }
    _raise_if_any([name for name, value in unsupported.items() if value])


def validate_loop_carrier_stack(config: "TransformerConfig", accepted: Sequence[str]) -> None:
    """Reject a loop stack class built for another ``loop_carrier`` (``accepted``: the
    carriers the class runs)."""
    if getattr(config, "loop_carrier", "injection") not in accepted:
        _raise_if_any([CARRIER_STACK_MISMATCH])


def untaped_recompute_routing(config: "TransformerConfig") -> List[str]:
    """The MoE routing features of a ``loop_recompute`` config whose recompute the routing tape
    (``forward_replay.RoutingTape``) cannot make route as its forward did; empty without
    ``loop_recompute`` or MoE. Forced load balancing and force-biased logits draw new random
    logits from Megatron's CUDA RNG tracker, which ``torch.utils.checkpoint`` does not rewind
    (the tape would replay the experts with other probabilities); routing replay, Sinkhorn and
    quantile balancing choose their experts outside the tape (and the quantile-balancing state
    would update twice), so a recompute that is not bitwise its forward changes the recomputed
    tensors' shapes (``CheckpointError``). Shared by the hybrid and GPT loop validators."""
    if not (getattr(config, "loop_recompute", False) and getattr(config, "num_moe_experts", None)):
        return []
    balancing = config.moe_router_load_balancing_type
    balancing = set(balancing) if isinstance(balancing, (list, tuple)) else {balancing}
    features = {
        "moe_router_force_load_balancing": config.moe_router_force_load_balancing,
        "moe_router_force_biased": config.moe_router_force_biased is not None,
        "moe_enable_routing_replay": config.moe_enable_routing_replay,
        "sinkhorn": "sinkhorn" in balancing,
        "quantile_balancing": "quantile_balancing" in balancing,
    }
    return [name for name, value in features.items() if value]


def validate_hybrid_loop_config(config: "TransformerConfig") -> None:
    """Reject config features the eager hybrid loop does not support.

    Pattern-independent rules only; the role rules need the stack's layer list
    (``resolve_hybrid_loop_roles``).
    """
    selective = config.recompute_granularity == "selective"
    explicit_modules = config.recompute_modules if config.recompute_modules is not None else ()
    attn_res = getattr(config, "loop_carrier", "injection") == "attn_res"
    blocks = getattr(config, "loop_recompute_block_stride", None) is not None
    edges = bool(getattr(config, "loop_recompute_edges", False))
    unsupported = {
        # Delayed scaling keeps an amax history per module call; a core module is called
        # R times per microbatch, so the history would mix rounds.
        DELAYED_FP8_SCALING: bool(config.fp8) and config.fp8_recipe == "delayed",
        # The attention-residual carrier names its own reason.
        (CUDA_GRAPHS_ATTN_RES if attn_res else CUDA_GRAPHS): config.cuda_graph_impl != "none",
        MHC_RECOMPUTE: selective and "mhc" in explicit_modules,
        MOE_EP_OVERLAP: config.overlap_moe_expert_parallel_comm,
        PAGED_STASH: config.moe_paged_stash,
        HYBRIDEP_PADDING: config.moe_hybridep_pad_variable_tokens
        or config.moe_hybridep_pad_uneven_dispatch_inputs,
        SHORTCUT_MOE: config.moe_shortcut_connection,
        RESIDUAL_STREAM_RECOMPUTE: selective and "residual_stream" in explicit_modules,
        PIPELINE_PARALLELISM: config.pipeline_model_parallel_size != 1
        or bool(config.virtual_pipeline_model_parallel_size),
        CONTEXT_PARALLELISM: config.context_parallel_size != 1 or config.dynamic_context_parallel,
        SEQUENCE_PARALLELISM: config.sequence_parallel,
        # Rounds must see the same function; per-call randomness breaks the recurrence and
        # the recompute replays.
        DROPOUT: bool(
            config.hidden_dropout or config.attention_dropout or config.moe_input_jitter_eps
        ),
        # A loop_recompute round's recompute must route as its forward (the routing tape).
        UNTAPED_RECOMPUTE_ROUTING: bool(untaped_recompute_routing(config)),
        FP4: bool(config.fp4),
        ACTIVATION_OFFLOAD: config.cpu_offloading or config.fine_grained_activation_offloading,
        # Standard recompute hooks would re-enter once per round and nest inside the
        # loop_recompute checkpoints. mhc / residual_stream have their own rules above.
        OTHER_RECOMPUTE: config.recompute_granularity == "full"
        or (
            selective
            and any(
                module not in _LOOP_RULED_RECOMPUTE_MODULES for module in _recompute_modules(config)
            )
        ),
        # GTP prefetches weights along a static next-weight chain, which a core revisited
        # R times per microbatch breaks.
        GTP_WEIGHT_REMAT: config.gtp_weight_remat_size > 1
        or config.expert_gtp_weight_remat_size > 1,
        # A second multi-stream residual scheme; the loop adapter is defined for mHC only.
        WIDE_RESIDUAL: config.wide_residual is not None,
        # DSGQACoreAttention takes no padding mask (padded / exited rows would train its
        # indexer); the loop specs build plain attention for '*', so the variant would be
        # silently dense there. DSA over MLA is MLA_WITHOUT_DSV4 (pattern rule).
        DSA_OVER_GQA: getattr(config, "experimental_attention_variant", None) == "dsa"
        and not config.multi_latent_attention,
        # Block / edge checkpoints refine loop_recompute; without it nothing is checkpointed,
        # and edges reuse the block stride. The stride's range needs the pattern
        # (resolve_hybrid_loop_roles).
        RECOMPUTE_BLOCKS_WITHOUT_RECOMPUTE: (blocks or edges) and not config.loop_recompute,
        RECOMPUTE_EDGES_WITHOUT_STRIDE: edges and not blocks,
        # Only the Huginn rounds of LoopHybridStack run block checkpoints.
        RECOMPUTE_BLOCKS_MOR: blocks and config.loop_architecture == "mor",
        RECOMPUTE_BLOCKS_ATTN_RES: blocks and attn_res,
    }
    _raise_if_any([name for name, value in unsupported.items() if value])


def validate_hybrid_loop_pattern(
    config: "TransformerConfig", main_pattern: str, mtp_pattern: Optional[str]
) -> None:
    """Reject hybrid patterns the loop cannot run.

    Args:
        config: The model config (``loop_architecture`` set).
        main_pattern: The main decoder pattern, possibly with ``|`` pipe symbols.
        mtp_pattern: The per-depth MTP pattern, or None.
    """
    symbols = set(main_pattern) | set(mtp_pattern or "")
    unsupported = {
        # Roles are counted over the whole main pattern; pipeline stages are not supported.
        PIPE_IN_PATTERN: Symbols.PIPE in main_pattern,
        # The loop runs MLA-family attention only as the DSv4 hybrid attention.
        MLA_WITHOUT_DSV4: (config.multi_latent_attention or bool(symbols & Symbols.MLA_ATTENTION))
        and config.experimental_attention_variant != "dsv4_hybrid",
    }
    _raise_if_any([name for name, value in unsupported.items() if value])


def reject_pipe_in_loop_pattern(loop_architecture: str, pattern: Optional[str]) -> None:
    """Reject ``|`` in the main pattern of a loop model (argument-validation time)."""
    if loop_architecture != "none" and Symbols.PIPE in _main_pattern(pattern):
        _raise_if_any([PIPE_IN_PATTERN])


def round_sublayers(config: Any, roles: LoopRoles) -> int:
    """The core sublayers one round runs: the whole core, or one of the ``loop_iterations``
    segments of an untied core (``loop_untie_core``)."""
    if getattr(config, "loop_untie_core", False):
        return roles.core // config.loop_iterations
    return roles.core


def _untied_segments_valid(symbols: Sequence[str], roles: LoopRoles, rounds: int) -> bool:
    """True if the core splits into ``rounds`` equal segments whose cuts lie on block
    boundaries (UNTIED_CORE_SEGMENTS)."""
    if rounds < 1 or roles.core % rounds:
        return False
    segment = roles.core // rounds
    return all(
        is_block_boundary(symbols, roles.core_start + step * segment) for step in range(1, rounds)
    )


def resolve_hybrid_loop_roles(
    config: "TransformerConfig",
    layer_symbols: Sequence[str],
    hash_moe_layer_threshold: Optional[int],
) -> LoopRoles:
    """Check the role rules against the stack's sublayers and return the roles.

    Args:
        config: The loop config (role fields set).
        layer_symbols: Symbols of the stack's sublayers, in order (the whole main pattern,
            without ``|``).
        hash_moe_layer_threshold: Global (1-based) layer number of the last hash-routed MoE
            layer, or None / 0 without hash routing.
    """
    symbols = list(layer_symbols)
    num_sublayers = len(symbols)
    prelude = config.loop_prelude_sublayers
    coda = config.loop_coda_sublayers
    violations = []
    roles = None
    if (
        prelude is None
        or coda is None
        or prelude < 0
        or coda < 0
        or prelude + coda >= num_sublayers
    ):
        violations.append(ROLES_DO_NOT_MATCH)
    else:
        roles = LoopRoles(prelude=prelude, core=num_sublayers - prelude - coda, coda=coda)

    if roles is not None:
        # A cut must separate whole attention + MLP blocks: the sublayer before it ends a
        # block, the one after it starts one.
        for cut in (roles.core_start, roles.core_end):
            if not is_block_boundary(symbols, cut):
                violations.append(CUT_NOT_ON_BLOCK)
                break
        # Hash layers route by token id; in the prelude each runs once per sample, while a
        # core round may run on compacted tokens without ids.
        if hash_moe_layer_threshold and hash_moe_layer_threshold > roles.prelude:
            violations.append(HASH_BEYOND_PRELUDE)
        # An untied core holds loop_iterations physical segments, one per round; each runs
        # as a round body, so every segment cut must separate whole blocks.
        if getattr(config, "loop_untie_core", False) and not _untied_segments_valid(
            symbols, roles, config.loop_iterations
        ):
            violations.append(UNTIED_CORE_SEGMENTS)
        # Block checkpoints cut the sublayers one round runs: the core, or one untied segment.
        stride = getattr(config, "loop_recompute_block_stride", None)
        if stride is not None and not 1 <= stride <= round_sublayers(config, roles):
            violations.append(RECOMPUTE_STRIDE_RANGE)

    if set(symbols) & _UNSUPPORTED_SYMBOLS:
        violations.append(UNSUPPORTED_SUBLAYERS)

    if config.loop_architecture == "mor":
        core = symbols[roles.core_start : roles.core_end] if roles is not None else symbols
        if config.enable_mhc_connections or not set(core) <= _MOR_CORE_SYMBOLS:
            name = MOR_NON_DENSE_CORE
            if config.loop_token_execution == "masked" and set(core) & (
                _DSV4_ATTENTION_SYMBOLS | {Symbols.MLA}
            ):
                name = f"{MOR_NON_DENSE_CORE} ({MASKED_MOR_ON_DSV4})"
            violations.append(name)

    _raise_if_any(violations)
    return roles


def validate_loop_mtp_pattern(mtp_pattern: Optional[str]) -> None:
    """Reject MTP sublayers the loop's per-symbol FLOP table has no row for.

    The MTP depths run once per microbatch outside the loop, but the loop stack reports the
    whole model's FLOPs, so every MTP symbol needs a row (``D`` would need a compress ratio
    for a layer number outside the main pattern).
    """
    symbols = set(mtp_pattern or "")
    unsupported = {
        UNSUPPORTED_SUBLAYERS: bool(symbols & _UNSUPPORTED_SYMBOLS),
        MTP_DS_ATTENTION: Symbols.DS_ATTENTION in symbols,
    }
    _raise_if_any([name for name, value in unsupported.items() if value])


def validate_hybrid_loop_inputs(
    *,
    packed_seq_params: Optional["PackedSeqParams"],
    inference_context: Optional[Any],
    attention_mask: Optional[Any] = None,
    has_dsv4_attention: bool = False,
) -> None:
    """Reject forward inputs the hybrid loop does not support (checked every forward).

    Packed THD is accepted only with a single sequence per microbatch (two ``cu_seqlens``
    entries). This is a stage-1 restriction, stricter than one sequence per row: checking
    that sequences are row-aligned needs the ``cu_seqlens`` values on the host, and this
    check reads shapes only (no device sync). Context parallelism, the main producer of
    packed inputs, is rejected separately.
    """
    violations = []
    if packed_seq_params is not None:
        cu_q = packed_seq_params.cu_seqlens_q
        cu_kv = packed_seq_params.cu_seqlens_kv
        if not (
            packed_seq_params.qkv_format == "thd"
            and cu_q is not None
            and cu_kv is not None
            and cu_q.numel() == 2
            and cu_kv.numel() == 2
        ):
            violations.append(PACKED_MULTI_SEQUENCE)
    if inference_context is not None:
        violations.append(INFERENCE)
    # On SBHD, DSv4 attention owns causal masking and raises on a dense mask; fail before
    # the first layer with the remedy instead.
    if packed_seq_params is None and attention_mask is not None and has_dsv4_attention:
        violations.append(SBHD_MASK_WITH_DSV4)
    _raise_if_any(violations)


def validate_loop_ddp_config(
    config: "TransformerConfig", ddp_config: "DistributedDataParallelConfig"
) -> None:
    """Reject DDP communication overlap for a hybrid loop (no-op for other configs).

    Overlapped grad reduce starts a bucket's sync when the per-parameter grad-ready counts
    reach those of the first batch; per-sample exit changes the number of core rounds per
    step, so a sync would never start or would start before later rounds' gradients land.
    Overlapped param gather waits on a bucket's all-gather at a module's first forward, which
    a core module re-enters once per round. GPT loop configs (no role fields) are not
    affected; their eager loop is tested with overlapped grad reduce.
    """
    if not is_hybrid_loop_config(config):
        return
    unsupported = {
        OVERLAP_GRAD_REDUCE: ddp_config.overlap_grad_reduce,
        OVERLAP_PARAM_GATHER: ddp_config.overlap_param_gather,
    }
    _raise_if_any([name for name, value in unsupported.items() if value])


# Megatron-FSDP rule names (``validate_loop_fsdp_config``, ``validate_loop_row_depths``), as
# they appear in the error message ("Loop with Megatron-FSDP does not support: <name>, <name>").
FSDP_PREFIX = "Loop with Megatron-FSDP does not support: "
FSDP_GPT_LOOP = "the GPT loop (no loop_prelude_sublayers / loop_coda_sublayers)"
FSDP_VERSION = "megatron_fsdp_version other than 1"
FSDP_OVERWRITE_MAIN_GRAD = (
    "megatron_fsdp_accumulate_main_grad off (each backward use of a core weight would overwrite "
    "the gradients of its earlier uses)"
)
FSDP_GRAD_DTYPE = (
    "megatron_fsdp_main_grads_dtype or megatron_fsdp_grad_comm_dtype other than fp32 (the "
    "rounds accumulate in the gradient bucket's dtype)"
)
FSDP_PERSISTENT_BUFFERS = (
    "persistent communication buffers (fsdp_double_buffer / nccl_ub / "
    "megatron_fsdp_max_pool_double_buffer)"
)
FSDP_FINE_GRAINED_GATHER = (
    "fine-grained parameter gather (megatron_fsdp_enable_fine_grained_param_gather or MXFP8 "
    "fp8_param_gather)"
)
FSDP_TE_OP_FUSER = "use_transformer_engine_op_fuser (its expert path forces overwrite_main_grad)"
FSDP_DELAYED_WGRAD = "delay_wgrad_compute"
FSDP_SAMPLED_DEPTHS = "sampled loop depths (loop_sample_depth / loop_exit_criterion='sampled')"
FSDP_ROW_DEPTHS = "per-row loop depths (set_loop_depths)"
FSDP_MOR = "MoR (loop_architecture='mor')"
FSDP_ATTN_RES = "loop_carrier='attn_res'"
FSDP_HSDP = "HSDP (num_distributed_optimizer_instances > 1)"
FSDP_CHECKPOINTING = (
    "checkpoint save / load (--save / --load / --pretrained-checkpoint; the fsdp_dtensor format "
    "is untested with the loop)"
)


def _raise_fsdp_if_any(violations: List[str]) -> None:
    if violations:
        raise ValueError(FSDP_PREFIX + ", ".join(violations))


def _is_fp32(dtype: Any) -> bool:
    """True for fp32 as a ``torch.dtype`` or as the CLI string (before argument mapping)."""
    import torch

    return dtype is torch.float32 or dtype == "fp32"


def validate_loop_fsdp_config(config: Any, ddp_config: Any) -> None:
    """Reject Megatron-FSDP settings a loop cannot train with (no-op without a loop).

    Called by the Megatron-FSDP v1 wrapper (``FullyShardedDataParallel``) with the model and
    DDP configs, and by argument validation with the argument namespace as both (duck-typed;
    the checkpoint rule needs the namespace's ``save`` / ``load`` / ``pretrained_checkpoint``).

    The loop runs every core weight several times per microbatch. Megatron-FSDP reduce-scatters
    a sharded gradient once per microbatch, after the parameter's last backward use, from an
    unsharded bucket allocated for that microbatch; the uses must therefore add into the
    bucket (accumulate mode, ``megatron_fsdp_accumulate_main_grad``), in fp32 (the bucket takes
    the gradient communication dtype). Refused besides: persistent communication buffers (every
    core unit's bucket stays alive across the rounds, more than a fixed pool holds),
    fine-grained gather, the TE op fuser and delayed weight gradients (untested with repeated
    module calls; the op fuser forces overwrites), sampled depths and per-row depths (ranks
    could run different numbers of rounds, i.e. different gather sequences, while the round
    count is agreed over TP x EP only; per-row depths are refused at runtime by
    ``validate_loop_row_depths``), MoR and the attention-residual carrier, HSDP, and
    checkpointing (Megatron-FSDP saves and loads ``fsdp_dtensor``, untested with the loop). The
    GPT loop and Megatron-FSDP v2 are not supported. Collects every violation and raises one
    ``ValueError`` (``FSDP_PREFIX``).
    """
    architecture = getattr(config, "loop_architecture", "none")
    if architecture == "none":
        return
    double_buffer = (
        bool(getattr(ddp_config, "fsdp_double_buffer", False))
        or bool(getattr(ddp_config, "nccl_ub", False))
        or bool(getattr(ddp_config, "megatron_fsdp_max_pool_double_buffer", False))
    )
    fine_grained_gather = bool(
        getattr(ddp_config, "megatron_fsdp_enable_fine_grained_param_gather", False)
    ) or (
        getattr(config, "fp8_recipe", None) == "mxfp8"
        and bool(getattr(ddp_config, "fp8_param_gather", False))
    )
    unsupported = {
        FSDP_GPT_LOOP: not is_hybrid_loop_config(config),
        FSDP_VERSION: getattr(ddp_config, "megatron_fsdp_version", 1) != 1,
        FSDP_OVERWRITE_MAIN_GRAD: not getattr(
            ddp_config, "megatron_fsdp_accumulate_main_grad", False
        ),
        FSDP_GRAD_DTYPE: not (
            _is_fp32(getattr(ddp_config, "megatron_fsdp_main_grads_dtype", None))
            and _is_fp32(getattr(ddp_config, "megatron_fsdp_grad_comm_dtype", None))
        ),
        FSDP_PERSISTENT_BUFFERS: double_buffer,
        FSDP_FINE_GRAINED_GATHER: fine_grained_gather,
        FSDP_TE_OP_FUSER: bool(getattr(config, "use_transformer_engine_op_fuser", False)),
        FSDP_DELAYED_WGRAD: bool(getattr(config, "delay_wgrad_compute", False))
        or bool(getattr(ddp_config, "delay_wgrad_compute", False)),
        FSDP_SAMPLED_DEPTHS: bool(getattr(config, "loop_sample_depth", False))
        or getattr(config, "loop_exit_criterion", "fixed") == "sampled",
        FSDP_MOR: architecture == "mor",
        FSDP_ATTN_RES: getattr(config, "loop_carrier", "injection") == "attn_res",
        FSDP_HSDP: getattr(ddp_config, "num_distributed_optimizer_instances", 1) > 1,
        FSDP_CHECKPOINTING: any(
            getattr(ddp_config, name, None) for name in ("save", "load", "pretrained_checkpoint")
        ),
    }
    _raise_fsdp_if_any([name for name, value in unsupported.items() if value])


def validate_loop_row_depths(stack: Any) -> None:
    """Reject per-row loop depths (``LoopHybridStack.set_loop_depths``) on a stack whose
    parameters Megatron-FSDP manages (``FSDP_ROW_DEPTHS``): the round count is agreed over
    TP x EP only, so data-parallel ranks could run different numbers of rounds and issue
    different parameter gathers. Megatron-FSDP marks every parameter it wraps
    ``__fsdp_param__``."""
    param = next(stack.parameters(), None)
    if param is not None and getattr(param, "__fsdp_param__", False):
        _raise_fsdp_if_any([FSDP_ROW_DEPTHS])


# RSBF rule names (``validate_rsbf_config``), as they appear in the error message.
RSBF_WITHOUT_HYBRID_ROLES = "RSBF without hybrid loop roles"
RSBF_MOR = "MoR under the RSBF schedule"
RSBF_SUM_LOSS = "calculate_per_token_loss=False"
RSBF_TRUNCATED_BPTT = "truncated BPTT under RSBF"
RSBF_SAMPLE_DEPTH = "loop_sample_depth under RSBF"
RSBF_LOOP_RECOMPUTE = "loop_recompute under RSBF"
RSBF_TP = "tensor parallelism under RSBF"
RSBF_ETP = "expert tensor parallelism under RSBF"
RSBF_STATIC_BUDGET_MOE = "static-budget MoE other than flex + HybridEP under RSBF"
# Expert parallelism runs lockstep across the EUs (null items replay a sub-round's MoE
# collectives on EUs without an item of its type); options whose collectives or statistics a
# null cannot replay are rejected (with EP > 1 only). moe_hybridep_pad_variable_tokens /
# moe_hybridep_pad_uneven_dispatch_inputs (an EP all-reduce inside every call) stay rejected by
# HYBRIDEP_PADDING, MoE recompute by OTHER_RECOMPUTE (and RSBF_MOE_RECOMPUTE).
RSBF_LOCKSTEP_MOE = "MoE options that break lockstep nulls under RSBF"
RSBF_MOE_RECOMPUTE = "MoE layer recompute under RSBF"
RSBF_CAPACITY_MOE = "capacity MoE under RSBF"
RSBF_FREEZE_FOR_MTP = "freeze_base_model_for_mtp under RSBF"
RSBF_QUANTILE_BALANCING = "quantile-balancing router under RSBF"
RSBF_DEPTH_PROBS = "loop_depth_probs mismatch"
RSBF_MIN_ROUNDS = "loop_min_rounds out of range"
RSBF_WAVES = "loop_rsbf_waves < 1"
RSBF_GLOBAL_AUX_LOSS = "global aux loss under RSBF"
RSBF_AUTOCAST = "autocast under RSBF"
RSBF_SHARDED_DP = "sharded data parallelism (Megatron FSDP / torch FSDP2) under RSBF"
# Backward item forms (``loop_rsbf_item_form``; ``program.build_core_plan``).
RSBF_ITEM_FORM = "unknown loop_rsbf_item_form"
RSBF_BLOCK_STRIDE_MISMATCH = "loop_rsbf_block_stride mismatch"
RSBF_BLOCK_STRIDE_BEYOND_CORE = "loop_rsbf_block_stride beyond the core"
RSBF_CODA_RECOMPUTE_WITHOUT_DSV4 = "coda mla_up_proj recompute without DSv4 attention"
RSBF_UNTIED_CORE = "loop_untie_core (a non-loop baseline; eager only)"
RSBF_CARRIER = (
    "loop_carrier other than 'injection' (the K3 attention-residual and residual carriers run "
    "in the eager LoopHybridStack only; RSBF would have to move the history with samples)"
)

# The CORE item forms: the whole round as one block (stage 3), the block-checkpoint item with
# stride ``loop_rsbf_block_stride`` (block inputs stashed by the sweep) and the nested item
# (rebuild, then per-block recompute + backward).
RSBF_ITEM_FORMS = ("whole_round", "block", "nested")


def _load_balancing_types(config: Any) -> Sequence[str]:
    types = getattr(config, "moe_router_load_balancing_type", None)
    if types is None:
        return ()
    return (types,) if isinstance(types, str) else tuple(types)


def _depth_probs_valid(probs: Optional[Sequence[float]], max_rounds: int) -> bool:
    return (
        probs is not None
        and len(probs) == max_rounds
        and all(p >= 0 for p in probs)
        and abs(sum(probs) - 1.0) <= 1e-6
    )


def rsbf_item_form_violations(
    form: str, stride: Optional[int], core_sublayers: Optional[int]
) -> List[str]:
    """The item-form rules (``loop_rsbf_item_form`` / ``loop_rsbf_block_stride``) broken by
    ``form`` and ``stride`` on a core of ``core_sublayers`` sublayers (None: unknown, the
    range rule is skipped).

    ``stride`` counts core sublayers. ``block`` needs ``1 <= stride <= core_sublayers``
    (``stride == core_sublayers`` is the whole round; the last block may be shorter);
    ``nested`` takes an optional stride (its granularity, default 1) with
    ``1 <= stride < core_sublayers`` (a nested item has at least two blocks); ``whole_round``
    takes none.
    """
    if form not in RSBF_ITEM_FORMS:
        return [RSBF_ITEM_FORM]
    if (
        (form == "block" and stride is None)
        or (form == "whole_round" and stride is not None)
        or (stride is not None and stride < 1)
    ):
        return [RSBF_BLOCK_STRIDE_MISMATCH]
    if core_sublayers is None or form == "whole_round":
        return []
    if form == "block":
        beyond = stride > core_sublayers
    else:
        beyond = (1 if stride is None else stride) >= core_sublayers
    return [RSBF_BLOCK_STRIDE_BEYOND_CORE] if beyond else []


def _config_core_sublayers(config: Any) -> Optional[int]:
    """``L_core = num_layers - prelude - coda`` at config time (``num_layers`` counts the
    sublayers of the main hybrid pattern; the pattern itself is a model argument), or None
    when the role fields or ``num_layers`` are unset. ``LoopProgram`` re-checks the item form
    against the built pattern."""
    prelude = getattr(config, "loop_prelude_sublayers", None)
    coda = getattr(config, "loop_coda_sublayers", None)
    num_layers = getattr(config, "num_layers", 0) or 0
    if prelude is None or coda is None or num_layers <= 0:
        return None
    return num_layers - prelude - coda


def _lockstep_moe_options(config: Any) -> List[str]:
    """The MoE options a lockstep null item cannot replay (checked with expert parallelism)."""
    dispatcher = getattr(config, "moe_token_dispatcher_type", "allgather")
    backend = getattr(config, "moe_flex_dispatcher_backend", None)
    options = {
        # Every rank all-gathers every rank's tokens: no null form.
        "moe_token_dispatcher_type='allgather'": dispatcher == "allgather",
        # Only HybridEP has a null form (the other backends' metadata and budgets are not
        # defined for an all-excluded routing).
        f"moe_flex_dispatcher_backend={backend!r}": dispatcher == "flex" and backend != "hybridep",
        # The dispatcher runs the shared experts inside token_dispatch.
        "moe_shared_expert_overlap": bool(getattr(config, "moe_shared_expert_overlap", False)),
        "overlap_dispatch_backward_with_experts_wgrad": bool(
            getattr(config, "overlap_dispatch_backward_with_experts_wgrad", False)
        ),
        "moe_megakernel_backend": getattr(config, "moe_megakernel_backend", None) is not None,
        # The null path through the latent projections is untested.
        "moe_latent_size": bool(getattr(config, "moe_latent_size", None)),
        # MoonEP: the null skips route() (so the dispatcher's wrap_layer_input and its
        # grad-reduce hook never run, while postprocess still finalizes the planner) and feeds
        # full-width probabilities to a planner that needs [tokens, topk].
        "moe_virtual_expert_load_balance": bool(
            getattr(config, "moe_virtual_expert_load_balance", False)
        ),
        # Its metric divides by the local token count, 0 on a null.
        "log_moe_overload_factor": bool(getattr(config, "log_moe_overload_factor", False)),
    }
    return [name for name, applies in options.items() if applies]


def _lockstep_moe_rule(config: Any, moe: bool) -> dict:
    """``{"<rule> (<options>)": True}`` when expert parallelism meets lockstep-breaking
    options, else ``{}``."""
    if not moe or config.expert_model_parallel_size <= 1:
        return {}
    options = _lockstep_moe_options(config)
    if not options:
        return {}
    return {f"{RSBF_LOCKSTEP_MOE} ({' / '.join(options)})": True}


def validate_rsbf_config(config: "TransformerConfig") -> None:
    """Reject configs the RSBF executor (``loop_schedule='rsbf'``) cannot run.

    Called by ``TransformerConfig.__post_init__`` when ``loop_schedule == 'rsbf'`` and again by
    the RSBF schedule. Collects every violation and raises one ``ValueError``
    (``"Loop RSBF does not support: <name>, <name>"``). The eager hybrid-loop guards
    (``validate_hybrid_loop_config``, the DDP guards) still apply.
    """
    moe = getattr(config, "num_moe_experts", None) is not None
    balancing = _load_balancing_types(config)
    sampled = config.loop_exit_criterion == "sampled"
    item_form_rules = rsbf_item_form_violations(
        getattr(config, "loop_rsbf_item_form", "whole_round"),
        getattr(config, "loop_rsbf_block_stride", None),
        _config_core_sublayers(config),
    )
    unsupported = {
        # The executor drives LoopHybridStack cells: a hybrid Huginn loop with role fields.
        RSBF_WITHOUT_HYBRID_ROLES: not is_hybrid_loop_config(config),
        # MoR (token-level exit) stays on the eager path.
        RSBF_MOR: config.loop_architecture == "mor",
        # The cells carry the injection state only.
        RSBF_CARRIER: getattr(config, "loop_carrier", "injection") != "injection",
        # The cells run one tied core with one adapter.
        RSBF_UNTIED_CORE: bool(getattr(config, "loop_untie_core", False)),
        # Per-sample backward seeds are loss sums; finalize divides by the token count.
        RSBF_SUM_LOSS: not config.calculate_per_token_loss,
        RSBF_TRUNCATED_BPTT: config.loop_bptt_iterations > 0,
        RSBF_SAMPLE_DEPTH: config.loop_sample_depth,
        # RSBF always checkpoints round inputs; a set flag would be silently ignored.
        RSBF_LOOP_RECOMPUTE: config.loop_recompute,
        # EUs run different items: a collective inside a cell call must be one of the
        # lockstep MoE calls (expert parallelism, replayed by null items). TP groups would
        # issue collectives in every attention / MLP call.
        RSBF_TP: config.tensor_model_parallel_size > 1,
        # With TP = 1, ETP > 1 builds expert-TP groups across DP ranks (untested with nulls).
        RSBF_ETP: (config.expert_tensor_parallel_size or 1) > 1,
        # NCCL EP has no null form (and its static budget has no padding exclusion); only the
        # flex dispatcher reads the budget (the all-to-all dispatcher ignores it, so a step would
        # run with a silently unused budget and no over-budget flag).
        RSBF_STATIC_BUDGET_MOE: getattr(config, "moe_expert_rank_capacity_factor", None) is not None
        and (
            getattr(config, "moe_token_dispatcher_type", None) != "flex"
            or getattr(config, "moe_flex_dispatcher_backend", None) != "hybridep"
        ),
        # Options a null item cannot replay (named in the message), with expert parallelism.
        **_lockstep_moe_rule(config, moe),
        # The checkpointed MoE layer would replay its dispatch collectives inside a real item's
        # backward, which no null replays.
        RSBF_MOE_RECOMPUTE: moe
        and config.recompute_granularity == "selective"
        and "moe" in _recompute_modules(config),
        # Token drop / drop-and-pad: the capacity comes from the local rows (a zero-row null
        # would send an empty equal-split all-to-all) and drops depend on the co-scheduled
        # samples. The main plan rejects capacity mode under RSBF.
        RSBF_CAPACITY_MOE: moe and getattr(config, "moe_expert_capacity_factor", None) is not None,
        RSBF_FREEZE_FOR_MTP: config.freeze_base_model_for_mtp,
        # Its histograms are not replay-gated.
        RSBF_QUANTILE_BALANCING: moe and "quantile_balancing" in balancing,
        RSBF_DEPTH_PROBS: (
            not _depth_probs_valid(config.loop_depth_probs, config.loop_iterations)
            if sampled
            else config.loop_depth_probs is not None
        ),
        RSBF_MIN_ROUNDS: not 1 <= config.loop_min_rounds <= config.loop_iterations,
        RSBF_WAVES: config.loop_rsbf_waves < 1,
        # A tp_dp_cp all-reduce on every grad-mode router call (ranks with different depths
        # would issue different counts) and a running average advanced by every forward.
        RSBF_GLOBAL_AUX_LOSS: moe and "global_aux_loss" in balancing,
        # forward_step's autocast region; the cells do not replicate it.
        RSBF_AUTOCAST: config.enable_autocast,
        # The CORE item form and its stride (no silently ignored stride, none outside the
        # core).
        **{rule: True for rule in item_form_rules},
        # D42: the scoped up-projection recompute exists only in DSv4HybridAttention.
        RSBF_CODA_RECOMPUTE_WITHOUT_DSV4: bool(
            getattr(config, "loop_rsbf_coda_recompute_mla_up_proj", False)
        )
        and getattr(config, "experimental_attention_variant", None) != "dsv4_hybrid",
    }
    violations = [name for name, value in unsupported.items() if value]
    if violations:
        raise ValueError("Loop RSBF does not support: " + ", ".join(violations))


# Loop parameter store rule names (``validate_param_store_config``), as they appear in the
# error message ("Loop parameter store does not support: <name>, <name>").
STORE_WITHOUT_RSBF = "the loop parameter store without loop_schedule='rsbf'"
STORE_PIPELINE = "pipeline parallelism with the loop parameter store"
STORE_TP = "tensor parallelism with the loop parameter store"
STORE_CP = "context parallelism with the loop parameter store"
STORE_SHARED_EXPERT_OVERLAP = "moe_shared_expert_overlap with the loop parameter store"
STORE_SUM_LOSS = "calculate_per_token_loss=False with the loop parameter store"
STORE_DELAYED_WGRAD = "delay_wgrad_compute with the loop parameter store"
STORE_FP8_PARAMS = (
    "FP8 primary weights (fp8_param / fp8_param_gather) with the loop parameter store"
)
STORE_TRANSPOSE_CACHE = (
    "FP8 / FP4 without disable_parameter_transpose_cache with the loop parameter store"
)
STORE_GRAD_REDUCE_DTYPE = "grad_reduce_in_fp32=False with the loop parameter store"
STORE_SHARDED_DP = "Megatron FSDP with the loop parameter store"
STORE_OPTIMIZER_INSTANCES = "num_distributed_optimizer_instances > 1 with the loop parameter store"
STORE_AVERAGE_IN_COLLECTIVE = "average_in_collective with the loop parameter store"
STORE_MODULE_GTP = "module GTP (gtp_weight_remat_size > 1) with the loop parameter store"
STORE_SHARED_EXPERT_GLU_INTERLEAVE = (
    "shared-expert GLU interleave (moe_shared_expert_glu_interleave_size) with the loop "
    "parameter store"
)


def validate_param_store_config(
    config: "TransformerConfig", store_config: Any = None, ddp_config: Any = None
) -> None:
    """Reject configs the loop parameter store (``loop/param_store.py``) cannot run.

    ``config``: the model's ``TransformerConfig``; ``store_config``: the store's
    ``LoopParamStoreConfig`` (its own settings are validated by the store; accepted for the
    guard's call signature); ``ddp_config``: the DDP wrap's ``DistributedDataParallelConfig``
    (None: the DDP rules are skipped, e.g. a bare model in tests). A store trains only through
    the RSBF executor (an eager training step would leave the shard gradients unreduced), on
    one stage without TP / CP; the shard-axis SUM of finalize needs the per-token loss; TE's
    delayed wgrad would run after the segment's reduce-scatter; the store gathers bf16 (no FP8
    primary weights) and a cached FP8 transpose would pin a quantized copy per layer; the shard
    gradients are DDP's fp32 ``main_grad``; one distributed-optimizer instance (the grid's);
    no pre-scaling inside the collective (per-token loss); no module GTP (only the grid); no
    shared-expert GLU interleave (the checkpoint's GLU row conversion would permute a stored
    row shard as if it were the whole weight).
    Collects every violation and raises one ``ValueError``.
    """
    del store_config
    quantized = getattr(config, "fp8", None) is not None or getattr(config, "fp4", None) is not None
    unsupported = {
        STORE_WITHOUT_RSBF: getattr(config, "loop_schedule", "standard") != "rsbf",
        STORE_PIPELINE: getattr(config, "pipeline_model_parallel_size", 1) > 1,
        STORE_TP: getattr(config, "tensor_model_parallel_size", 1) > 1,
        STORE_CP: getattr(config, "context_parallel_size", 1) > 1,
        # Its side stream reads the shared-expert weights outside the store's events.
        STORE_SHARED_EXPERT_OVERLAP: bool(getattr(config, "moe_shared_expert_overlap", False)),
        STORE_SUM_LOSS: not getattr(config, "calculate_per_token_loss", False),
        STORE_DELAYED_WGRAD: bool(getattr(config, "delay_wgrad_compute", False)),
        STORE_FP8_PARAMS: bool(getattr(config, "fp8_param", False))
        or bool(getattr(ddp_config, "fp8_param_gather", False)),
        STORE_TRANSPOSE_CACHE: quantized
        and not getattr(config, "disable_parameter_transpose_cache", False),
        STORE_MODULE_GTP: getattr(config, "gtp_weight_remat_size", 1) > 1,
        # The checkpoint's GLU conversion permutes a stored fc1's row shard as if it were the
        # whole [2F, H] weight (mixing gate / up rows across shards and moving pad rows into
        # real rows). Routed experts are EP-resident (not stored), so their interleave is fine.
        STORE_SHARED_EXPERT_GLU_INTERLEAVE: bool(getattr(config, "gated_linear_unit", False))
        and bool(getattr(config, "use_grouped_gemm_for_shared_expert", False))
        and getattr(config, "moe_shared_expert_glu_interleave_size", None) is not None,
    }
    if ddp_config is not None:
        unsupported.update(
            {
                STORE_GRAD_REDUCE_DTYPE: not getattr(ddp_config, "grad_reduce_in_fp32", False),
                STORE_SHARDED_DP: bool(getattr(ddp_config, "use_megatron_fsdp", False))
                or bool(getattr(ddp_config, "use_custom_fsdp", False)),
                STORE_OPTIMIZER_INSTANCES: getattr(
                    ddp_config, "num_distributed_optimizer_instances", 1
                )
                > 1,
                STORE_AVERAGE_IN_COLLECTIVE: bool(
                    getattr(ddp_config, "average_in_collective", False)
                ),
            }
        )
    violations = [name for name, value in unsupported.items() if value]
    if violations:
        raise ValueError("Loop parameter store does not support: " + ", ".join(violations))


def validate_rsbf_data_parallel(model: Any) -> None:
    """Reject a sharded data-parallel wrapper around a model run by the RSBF executor.

    Inside a step the executor issues collectives only in its plan's fixed order (status
    exchanges and sample / chain / item moves, identical on every rank), while the ranks run
    different numbers of cell calls and items (and nulls). Megatron FSDP and torch FSDP2
    all-gather parameters and reduce-scatter gradients in per-module hooks, which fire on every
    cell call and item, so their collective count would depend on each rank's items and the
    ranks would issue mismatched collectives. Megatron ``DistributedDataParallel`` (without
    ``use_megatron_fsdp``) and a bare module pass. Walks the ``.module`` chain of the wrappers.
    """
    import torch

    from megatron.core.distributed.data_parallel_base import _BaseDataParallel
    from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel

    try:
        from torch.distributed.fsdp import FSDPModule as fsdp_module
    except ImportError:  # torch < 2.6 or no distributed build
        fsdp_module = None
    module = model
    while isinstance(module, torch.nn.Module):
        sharded = isinstance(module, _BaseDataParallel) and (
            not isinstance(module, DistributedDataParallel)
            or getattr(getattr(module, "ddp_config", None), "use_megatron_fsdp", False)
        )
        if sharded or (fsdp_module is not None and isinstance(module, fsdp_module)):
            raise ValueError("Loop RSBF does not support: " + RSBF_SHARDED_DP)
        module = getattr(module, "module", None)
