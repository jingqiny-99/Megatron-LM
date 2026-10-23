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
* ``validate_loop_decoder``: ``HybridModel.__init__``, after the decoder is built.
* ``validate_hybrid_loop_inputs``: the start of the loop stack's layer loop, every forward.
* ``validate_loop_ddp_config``: ``DistributedDataParallel.__init__``.

The role-field checks themselves (both set or neither, nonnegative, non-empty core) and the
quantized-loop weight-cache rule are in ``TransformerConfig.__post_init__``.

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
FP4 = "FP4"
ACTIVATION_OFFLOAD = "activation offload"
OTHER_RECOMPUTE = "activation recompute other than loop_recompute"
GTP_WEIGHT_REMAT = "GTP weight remat"
WIDE_RESIDUAL = "wide residual"
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


def validate_hybrid_loop_config(config: "TransformerConfig") -> None:
    """Reject config features the eager hybrid loop does not support.

    Pattern-independent rules only; the role rules need the stack's layer list
    (``resolve_hybrid_loop_roles``).
    """
    selective = config.recompute_granularity == "selective"
    explicit_modules = config.recompute_modules if config.recompute_modules is not None else ()
    unsupported = {
        # Delayed scaling keeps an amax history per module call; a core module is called
        # R times per microbatch, so the history would mix rounds.
        DELAYED_FP8_SCALING: bool(config.fp8) and config.fp8_recipe == "delayed",
        CUDA_GRAPHS: config.cuda_graph_impl != "none",
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
            if 0 < cut < num_sublayers and not (
                symbols[cut - 1] in _BLOCK_END and symbols[cut] in _BLOCK_START
            ):
                violations.append(CUT_NOT_ON_BLOCK)
                break
        # Hash layers route by token id; in the prelude each runs once per sample, while a
        # core round may run on compacted tokens without ids.
        if hash_moe_layer_threshold and hash_moe_layer_threshold > roles.prelude:
            violations.append(HASH_BEYOND_PRELUDE)

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
