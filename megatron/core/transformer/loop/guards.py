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
* ``validate_loop_gtp_shards``: the loop stack's constructor, after its layers are built.
* ``validate_loop_ddp_config``: ``DistributedDataParallel.__init__``.
* ``validate_loop_fsdp_config`` (``"Loop with Megatron-FSDP does not support: ..."``): the
  Megatron-FSDP v1 wrapper's constructor (``FullyShardedDataParallel``) and argument validation;
  ``validate_loop_row_depths``: ``LoopHybridStack.set_loop_depths``.
* ``validate_rsbf_config`` (``"Loop RSBF does not support: ..."``):
  ``TransformerConfig.__post_init__`` with ``loop_schedule='rsbf'``, and the RSBF schedule.
* ``validate_rsbf_data_parallel``: the RSBF executor, every step, on the model it runs.
* The ring (``loop_schedule='ring'``: the loop over pipeline stages, ``"Loop ring does not
  support: ..."``): ``validate_loop_ring_config`` in ``TransformerConfig.__post_init__`` and the
  loop stack's constructor; ``validate_ring_layout`` (the ``'|'`` layout) in
  ``HybridModel.__init__``; ``resolve_hybrid_loop_ring_roles`` (one chunk's roles) in the loop
  stack's constructor. Under the ring, ``'|'`` and pipeline / virtual-pipeline parallelism are
  allowed (``PIPE_IN_PATTERN`` and ``PIPELINE_PARALLELISM`` apply to every other loop).

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
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

from megatron.core.models.hybrid.layers.utils import Symbols

if TYPE_CHECKING:
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.transformer_config import TransformerConfig

# Rule names, as they appear in the error message.
PIPE_IN_PATTERN = "'|' in hybrid_layer_pattern"
OVERLAP_GRAD_REDUCE = "overlap_grad_reduce without a fixed depth"
OVERLAP_PARAM_GATHER = "overlap_param_gather without a fixed depth"
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
GTP_WEIGHT_REMAT = "GTP weight remat without a fixed depth"
# Per-row depths (``LoopHybridStack.set_loop_depths``) on a stack holding GTP shards.
GTP_ROW_DEPTHS = "per-row loop depths (set_loop_depths) with GTP weight remat"
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


def fixed_depth_loop(config: Any) -> bool:
    """True for a Huginn loop that runs the same rounds on every sample of every step: the
    standard schedule, ``loop_exit_criterion='fixed'`` and no ``loop_sample_depth`` (per-row
    depths, ``LoopHybridStack.set_loop_depths``, are refused at runtime where they matter).

    Every core weight is then used the same number of times in every microbatch (R times, or
    the differentiated rounds under truncated BPTT), so a per-use count learned once holds for
    every later step. MoR (a round with no token skips its sublayers), sampled depths and the
    RSBF schedule (per-sample exit) are not fixed."""
    return (
        getattr(config, "loop_architecture", "none") == "huginn"
        and getattr(config, "loop_exit_criterion", "fixed") == "fixed"
        and not getattr(config, "loop_sample_depth", False)
        and getattr(config, "loop_schedule", "standard") == "standard"
    )


def is_ring_loop_config(config: Any) -> bool:
    """True for a loop config on the ring schedule (``loop_schedule='ring'``: the hybrid loop over
    pipeline stages)."""
    return (
        getattr(config, "loop_architecture", "none") != "none"
        and getattr(config, "loop_schedule", "standard") == "ring"
    )


def fixed_glm_ring(config: Any) -> bool:
    """A distinct prelude and coda around data-independent native DSA core visits."""
    return (
        is_ring_loop_config(config)
        and getattr(config, "loop_ring_separate_edges", False)
        and getattr(config, "experimental_attention_variant", None) == "dsa"
        and getattr(config, "loop_carrier", None) == "gated_residual"
        and getattr(config, "loop_exit_criterion", "fixed") == "fixed"
        and not getattr(config, "loop_sample_depth", False)
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
        "gated_residual requires supported exit depths and full BPTT": carrier == "gated_residual"
        and (
            config.loop_sample_depth
            or config.loop_bptt_iterations
            or config.loop_exit_criterion not in ("fixed", "stratified", "online")
        ),
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
        "moe_router_force_load_balancing": config.moe_router_force_load_balancing
        and not fixed_glm_ring(config),
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
        (CUDA_GRAPHS_ATTN_RES if attn_res else CUDA_GRAPHS): config.cuda_graph_impl != "none"
        and not (config.cuda_graph_impl == "full_iteration" and fixed_glm_ring(config)),
        MHC_RECOMPUTE: selective and "mhc" in explicit_modules,
        MOE_EP_OVERLAP: config.overlap_moe_expert_parallel_comm,
        PAGED_STASH: config.moe_paged_stash,
        HYBRIDEP_PADDING: config.moe_hybridep_pad_variable_tokens
        or config.moe_hybridep_pad_uneven_dispatch_inputs,
        SHORTCUT_MOE: config.moe_shortcut_connection,
        RESIDUAL_STREAM_RECOMPUTE: selective and "residual_stream" in explicit_modules,
        # The ring schedule runs the loop over pipeline stages (validate_loop_ring_config,
        # validate_ring_layout); every other loop holds the whole main pattern on one stage.
        PIPELINE_PARALLELISM: (
            config.pipeline_model_parallel_size != 1
            or bool(config.virtual_pipeline_model_parallel_size)
        )
        and not is_ring_loop_config(config),
        # Native GLM DSA gathers token latents and indexer keys over the CP group.
        # The other loop attention implementations still assume local sequence ownership.
        CONTEXT_PARALLELISM: (config.dynamic_context_parallel and not fixed_glm_ring(config))
        or (config.context_parallel_size != 1 and config.experimental_attention_variant != "dsa"),
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
        OTHER_RECOMPUTE: (
            config.recompute_granularity == "full"
            and not (
                fixed_glm_ring(config)
                and config.recompute_method == "uniform"
                and (config.recompute_num_layers or 0) > 0
            )
        )
        or (
            selective
            and any(
                module not in _LOOP_RULED_RECOMPUTE_MODULES
                and not (module == "layernorm" and fixed_glm_ring(config))
                for module in _recompute_modules(config)
            )
        ),
        # GTP gathers a weight before every use and reduce-scatters its gradient after every
        # use. A core weight is used R times per microbatch: every reduce-scatter adds into the
        # shard's main_grad (a weight's next reduce-scatter first finalizes its previous one),
        # and the static prefetch chain is only a hint under the loop (the stack turns on
        # GTP_CONFIG.verify_prefetched_gathers: a consume reads a prefetched buffer only if
        # that buffer still holds this weight's last gather). DDP's grad-ready fires once per
        # use, so the per-use count must be the same every step: fixed depth only.
        GTP_WEIGHT_REMAT: (
            config.gtp_weight_remat_size > 1 or config.expert_gtp_weight_remat_size > 1
        )
        and not fixed_depth_loop(config),
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
        RECOMPUTE_BLOCKS_WITHOUT_RECOMPUTE: (blocks or edges)
        and not config.loop_recompute
        and not (
            edges
            and getattr(config, "loop_schedule", "standard") == "rsbf"
            and getattr(config, "experimental_attention_variant", None) == "dsa"
            and config.loop_carrier == "gated_residual"
        ),
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
    symbols = set(_main_pattern(main_pattern).replace(Symbols.PIPE, "")) | set(mtp_pattern or "")
    native_dsa = config.experimental_attention_variant == "dsa" and config.multi_latent_attention
    unsupported = {
        # Roles are counted over the whole main pattern; only the ring schedule runs them over
        # pipeline stages (its layout: validate_ring_layout).
        PIPE_IN_PATTERN: Symbols.PIPE in main_pattern and not is_ring_loop_config(config),
        # The native DSA stack builds D, while DSv4 builds the full W/C/H/D family.
        MLA_WITHOUT_DSV4: (config.multi_latent_attention or bool(symbols & Symbols.MLA_ATTENTION))
        and not (native_dsa or config.experimental_attention_variant == "dsv4_hybrid"),
        "native DSA loop requires D attention with dense or MoE feed-forward blocks": native_dsa
        and (
            Symbols.DS_ATTENTION not in symbols
            or bool(symbols - {Symbols.DS_ATTENTION, Symbols.MLP, Symbols.MOE})
        ),
    }
    _raise_if_any([name for name, value in unsupported.items() if value])


def reject_pipe_in_loop_pattern(
    loop_architecture: str, pattern: Optional[str], loop_schedule: str = "standard"
) -> None:
    """Reject ``|`` in the main pattern of a loop model (argument-validation time), unless the
    ring schedule (``loop_schedule='ring'``) runs the loop over those pipeline stages."""
    if (
        loop_architecture != "none"
        and loop_schedule != "ring"
        and Symbols.PIPE in _main_pattern(pattern)
    ):
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


def validate_loop_mtp_pattern(mtp_pattern: Optional[str], *, native_dsa: bool = False) -> None:
    """Reject MTP sublayers the loop's per-symbol FLOP table has no row for.

    The MTP depths run once per microbatch outside the loop, but the loop stack reports the
    whole model's FLOPs, so every MTP symbol needs a row (``D`` would need a compress ratio
    for a layer number outside the main pattern).
    """
    symbols = set(mtp_pattern or "")
    unsupported = {
        UNSUPPORTED_SUBLAYERS: bool(symbols & _UNSUPPORTED_SYMBOLS),
        MTP_DS_ATTENTION: Symbols.DS_ATTENTION in symbols and not native_dsa,
    }
    _raise_if_any([name for name, value in unsupported.items() if value])


def validate_hybrid_loop_inputs(
    *,
    packed_seq_params: Optional["PackedSeqParams"],
    inference_context: Optional[Any],
    attention_mask: Optional[Any] = None,
    has_dsv4_attention: bool = False,
    allow_packed_documents: bool = False,
) -> None:
    """Reject forward inputs the hybrid loop does not support (checked every forward).

    The fixed native-DSA ring carries document boundaries through both attention
    banks and accepts multiple THD documents. Other schedules require a single
    sequence per microbatch. All boundary checks here use shapes on the host.
    """
    violations = []
    if packed_seq_params is not None:
        cu_q = packed_seq_params.cu_seqlens_q
        cu_kv = packed_seq_params.cu_seqlens_kv
        if not (
            packed_seq_params.qkv_format == "thd"
            and cu_q is not None
            and cu_kv is not None
            and cu_q.ndim == cu_kv.ndim == 1
            and cu_q.numel() == cu_kv.numel()
            and (cu_q.numel() >= 2 if allow_packed_documents else cu_q.numel() == 2)
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
    """Reject DDP communication overlap for a hybrid loop without a fixed depth (no-op for
    other configs).

    Overlapped grad reduce starts a bucket's sync when the per-parameter grad-ready counts of
    the last microbatch reach those of the first step (``_ParamAndGradBucketGroup.
    register_grad_ready``). An autograd-hooked parameter is ready once per backward, after its
    last use (its ``AccumulateGrad`` node waits for every use's gradient, also through the
    loop's non-reentrant checkpoints); a GTP shard is ready once per use
    (``_handle_megatron_grad_accum`` after each use's reduce-scatter). At a fixed depth
    (``fixed_depth_loop``) both counts are the same every step, so a bucket's sync starts after
    the last use of every parameter in it, and a count above the first step's raises in
    ``register_grad_ready``. Per-sample exit changes the number of rounds per step, so a GTP
    shard's sync would never start or would start before later rounds' gradients land.

    Overlapped param gather waits on a bucket's all-gather in the first forward pre-hook of a
    module owning one of its parameters; a core module re-entered in later rounds and in the
    recompute finds the bucket published (``finish_param_sync`` is a no-op until the step's
    ``finish_grad_sync``). Allowed with the grad-reduce rule.

    GPT loop configs (no role fields) are not affected; their eager loop is tested with
    overlapped grad reduce.
    """
    if not is_hybrid_loop_config(config):
        return
    fixed = fixed_depth_loop(config)
    unsupported = {
        OVERLAP_GRAD_REDUCE: ddp_config.overlap_grad_reduce and not fixed,
        OVERLAP_PARAM_GATHER: ddp_config.overlap_param_gather and not fixed,
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
FSDP_HSDP = "HSDP (num_distributed_optimizer_instances > 1) without a fixed depth"
FSDP_REDUCE_SCATTER_PER_USE = (
    "megatron_fsdp_reduce_scatter_per_use without a fixed depth (a reduce-scatter per use: "
    "every data-parallel rank must run the same rounds)"
)
FSDP_CHECKPOINTING = (
    "checkpoint save or resume (--save / --pretrained-checkpoint / --load without --finetune "
    "--no-load-optim or with init_model_with_meta_device; only a model-only torch_dist load is "
    "supported: it loads the built model before the Megatron-FSDP wrap)"
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
    the checkpoint rule needs the namespace's ``save`` / ``load`` / ``pretrained_checkpoint`` /
    ``finetune`` / ``no_load_optim`` / ``init_model_with_meta_device``).

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
    ``validate_loop_row_depths``), MoR and the attention-residual carrier, HSDP without a fixed
    depth, and checkpointing except a model-only load (Megatron-FSDP saves and loads ``fsdp_dtensor``,
    untested with the loop; a ``--load`` with ``--finetune --no-load-optim`` loads a torch_dist
    checkpoint's weights into the built model before the Megatron-FSDP wrap, in training's
    ``_build_loop_megatron_fsdp_models``, which needs materialized weights: no
    ``init_model_with_meta_device``). The GPT loop and Megatron-FSDP v2 are not supported.
    Collects every violation and raises one ``ValueError`` (``FSDP_PREFIX``).

    HSDP / HFSDP (``num_distributed_optimizer_instances > 1``) at a fixed depth
    (``fixed_depth_loop``): a gradient is reduced when the parameter's
    ``register_post_accumulate_grad_hook`` fires, once per backward and after its last use
    (autograd's ``AccumulateGrad`` waits for every use's gradient); the DP-Outer reduction
    (all-reduce, or reduce-scatter for HFSDP) of a bucket group runs only in that reduction on
    the last microbatch (``outer_fsdp_group_grad_reduce``), over the DP-Shard gradient shard
    that has accumulated every microbatch's reduce-scatter: once per step, after the last use.

    Per-use reduce-scatter mode (``megatron_fsdp_reduce_scatter_per_use``) at a fixed depth:
    the unit's post-backward hook reduce-scatters its sharded buckets after every use (each
    adds into the sharded main gradient) and frees them, so a core unit's unsharded bucket no
    longer lives across the rounds; the hook-driven reduction after the last use stays the only
    one with the DP-Outer reduction. Each use issues a collective, so every data-parallel rank
    must run the same rounds: refused without a fixed depth (``FSDP_REDUCE_SCATTER_PER_USE``).
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
        FSDP_HSDP: getattr(ddp_config, "num_distributed_optimizer_instances", 1) > 1
        and not fixed_depth_loop(config),
        FSDP_REDUCE_SCATTER_PER_USE: bool(
            getattr(ddp_config, "megatron_fsdp_reduce_scatter_per_use", False)
        )
        and not fixed_depth_loop(config),
        FSDP_CHECKPOINTING: any(
            getattr(ddp_config, name, None) for name in ("save", "pretrained_checkpoint")
        )
        or (
            bool(getattr(ddp_config, "load", None))
            and not (
                getattr(ddp_config, "finetune", False)
                and getattr(ddp_config, "no_load_optim", False)
                and not getattr(ddp_config, "init_model_with_meta_device", False)
            )
        ),
    }
    _raise_fsdp_if_any([name for name, value in unsupported.items() if value])


def validate_loop_row_depths(stack: Any) -> None:
    """Reject per-row loop depths (``LoopHybridStack.set_loop_depths``) on a stack whose
    parameters Megatron-FSDP manages (``FSDP_ROW_DEPTHS``): the round count is agreed over
    TP x EP only, so data-parallel ranks could run different numbers of rounds and issue
    different parameter gathers. Megatron-FSDP marks every parameter it wraps
    ``__fsdp_param__``.

    Also rejects them on a stack holding GTP weight-remat shards (``GTP_ROW_DEPTHS``,
    ``is_gtp_weight_remat``): the stack runs ``max(depths)`` rounds, which changes from step to
    step, and a GTP shard is reduce-scattered and marked grad-ready once per use, so DDP's
    per-use count would differ from the one learned in the first step."""
    param = next(stack.parameters(), None)
    if param is not None and getattr(param, "__fsdp_param__", False):
        _raise_fsdp_if_any([FSDP_ROW_DEPTHS])
    if has_module_gtp_shards(stack):
        _raise_if_any([GTP_ROW_DEPTHS])


def has_module_gtp_shards(module: Any) -> bool:
    """True if ``module`` holds module-GTP weight-remat shards: the ``GTPShardedParam`` weights
    of GTP-wrapped layers (``is_gtp_weight_remat`` and TE's ``is_distributed_weight``). The
    loop parameter store's grid shards carry only the first tag and are not counted."""
    return any(
        getattr(p, "is_gtp_weight_remat", False) and getattr(p, "is_distributed_weight", False)
        for p in module.parameters()
    )


def validate_loop_gtp_shards(config: Any, stack: Any) -> None:
    """Reject a loop stack built with module-GTP shards without a fixed depth
    (``GTP_WEIGHT_REMAT``). Layers are GTP-wrapped by their process groups, so the shards can
    exist while the config's ``gtp_weight_remat_size`` / ``expert_gtp_weight_remat_size`` (read
    by ``validate_hybrid_loop_config``) are 1."""
    if has_module_gtp_shards(stack) and not fixed_depth_loop(config):
        _raise_if_any([GTP_WEIGHT_REMAT])


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
RSBF_CODA_RETAIN_RANGE = "loop_rsbf_coda_retain_sublayers outside the coda"
RSBF_CODA_RETAIN_WITHOUT_EDGES = "loop_rsbf_coda_retain_sublayers needs loop_recompute_edges"
RSBF_CODA_RECOMPUTE_WITHOUT_DSV4 = "coda mla_up_proj recompute without DSv4 attention"
RSBF_UNTIED_CORE = "loop_untie_core (a non-loop baseline; eager only)"
RSBF_CARRIER = "RSBF needs the injection carrier or known-depth native DSA gated_residual without a parameter store"

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
    # The executor checks equal per-depth home counts before dispatch or replay.
    home_lockstep_mok = (
        getattr(config, "moe_megakernel_backend", None) == "mok"
        and (getattr(config, "loop_exit_criterion", None) in ("fixed", "stratified")
             or getattr(config, "loop_rsbf_fill", "drain") == "refill")
        and not getattr(config, "loop_sample_depth", False)
    )
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
        "moe_megakernel_backend": getattr(config, "moe_megakernel_backend", None) is not None
        and not home_lockstep_mok,
        # The null path through the latent projections is untested.
        "moe_latent_size": bool(getattr(config, "moe_latent_size", None)),
        # MoonEP: the null skips route() (so the dispatcher's wrap_layer_input and its
        # grad-reduce hook never run, while postprocess still finalizes the planner) and feeds
        # full-width probabilities to a planner that needs [tokens, topk].
        "moe_virtual_expert_load_balance": bool(
            getattr(config, "moe_virtual_expert_load_balance", False)
        )
        and not home_lockstep_mok,
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
    retained = getattr(config, "loop_rsbf_coda_retain_sublayers", 0)
    item_form_rules = rsbf_item_form_violations(
        getattr(config, "loop_rsbf_item_form", "whole_round"),
        getattr(config, "loop_rsbf_block_stride", None),
        _config_core_sublayers(config),
    )
    unsupported = {
        # The executor drives LoopHybridStack cells: a hybrid Huginn loop with role fields.
        "refill requires one optimizer-wide wave": getattr(config, "loop_rsbf_fill", "drain") == "refill"
        and config.loop_rsbf_waves != 1,
        "refill with MoonEP replica planning": getattr(config, "loop_rsbf_fill", "drain") == "refill"
        and bool(getattr(config, "moe_virtual_expert_load_balance", False)),
        "refill with a sharded parameter store or Megatron-FSDP": getattr(config, "loop_rsbf_fill", "drain") == "refill"
        and (getattr(config, "loop_param_store_shard_size", 1) > 1 or getattr(config, "use_megatron_fsdp", False)),
        RSBF_WITHOUT_HYBRID_ROLES: not is_hybrid_loop_config(config),
        # MoR (token-level exit) stays on the eager path.
        RSBF_MOR: config.loop_architecture == "mor",
        # The gated core uses its original round input in the last backward block.
        # Store visits for the gate have no segment mapping.
        RSBF_CARRIER: getattr(config, "loop_carrier", "injection") != "injection"
        and not (
            config.loop_carrier == "gated_residual"
            and getattr(config, "experimental_attention_variant", None) == "dsa"
            and (config.loop_exit_criterion in ("fixed", "stratified")
                 or getattr(config, "loop_rsbf_fill", "drain") == "refill")
        ),
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
        RSBF_CODA_RETAIN_RANGE: retained < 0
        or retained > (getattr(config, "loop_coda_sublayers", None) or 0),
        RSBF_CODA_RETAIN_WITHOUT_EDGES: retained > 0 and not config.loop_recompute_edges,
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
        # Store-driven coda segments have their own activation and reduction lifecycle.
        "coda activation retention with the loop parameter store": bool(
            getattr(config, "loop_rsbf_coda_retain_sublayers", 0)
        ),
        STORE_PIPELINE: getattr(config, "pipeline_model_parallel_size", 1) > 1,
        STORE_TP: getattr(config, "tensor_model_parallel_size", 1) > 1,
        STORE_CP: getattr(config, "context_parallel_size", 1) > 1,
        RSBF_CARRIER: getattr(config, "loop_carrier", "injection") == "gated_residual",
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


def lockstep_rsbf_mok_zero2(config, ddp_config):
    """The fixed, equal-size RSBF plan supported by the Megatron-FSDP v1 adapter."""
    dense = getattr(ddp_config, "data_parallel_sharding_strategy", None)
    expert = getattr(ddp_config, "expert_data_parallel_sharding_strategy", None) or dense
    return (
        getattr(config, "loop_schedule", None) == "rsbf"
        and getattr(config, "loop_exit_criterion", None) == "fixed"
        and not getattr(config, "loop_sample_depth", False)
        and getattr(config, "moe_megakernel_backend", None) == "mok"
        and getattr(config, "bf16", False)
        and not getattr(config, "fp8", None)
        and not getattr(config, "fp4", None)
        and getattr(ddp_config, "megatron_fsdp_version", None) == 1
        and dense == expert == "optim_grads"
        and getattr(ddp_config, "num_distributed_optimizer_instances", 1) == 1
        and getattr(ddp_config, "megatron_fsdp_accumulate_main_grad", False)
        and getattr(ddp_config, "overlap_grad_reduce", False)
        and not getattr(ddp_config, "megatron_fsdp_reduce_scatter_per_use", False)
    )


def validate_rsbf_data_parallel(model: Any, config=None):
    """Find the v1 ZeRO-2 wrapper for a fixed native MoK plan and reject other sharded modes.

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
        if (
            sharded
            and config is not None
            and lockstep_rsbf_mok_zero2(config, getattr(module, "ddp_config", None))
        ):
            from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV1

            if isinstance(module, FullyShardedDataParallelV1):
                return module.module
        if sharded or (fsdp_module is not None and isinstance(module, fsdp_module)):
            raise ValueError("Loop RSBF does not support: " + RSBF_SHARDED_DP)
        module = getattr(module, "module", None)


# ---------------------------------------------------------------------------------------------
# The ring schedule (loop_schedule='ring'): the hybrid Huginn loop over pipeline stages
# ---------------------------------------------------------------------------------------------
#
# The main pattern carries P x k '|' segments of the unchanged layer sequence (P pipeline stages,
# k core slices per stage: k = 1 one chunk per rank, k > 1 Megatron's standard virtual-pipeline
# chunks). Segment s is the physical chunk (p, j) = (s % P, s // P), as select_pipeline_segment
# indexes it. The first segment holds the prelude before its core slice (the round-start chunk:
# the embedding and the loop adapter), the last one the coda after its core slice (the
# round-end chunk: the loop norm, MTP and the heads). Every microbatch passes the stage ring once
# per round; the shared core weights live once, on the chunk that holds them.

# Ring rule names, as they appear in the error message ("Loop ring does not support: <name>").
# Config rules (validate_loop_ring_config).
RING_WITHOUT_HYBRID_ROLES = "the ring schedule without hybrid loop roles"
RING_WITHOUT_PP = "loop_schedule='ring' without pipeline parallelism"
RING_MOR = "MoR under the ring schedule"
RING_CARRIER = "loop_carrier other than 'injection' under the ring schedule"
RING_SAMPLE_DEPTH = "loop_sample_depth under the ring schedule"
RING_TRUNCATED_BPTT = "truncated BPTT under the ring schedule"
RING_UNTIED_CORE = "loop_untie_core under the ring schedule"
RING_TP = "tensor parallelism under the ring schedule"
RING_STATIC_BUDGET_MOE = (
    "a static MoE rank budget (moe_expert_rank_capacity_factor) under the ring schedule"
)
RING_DEFER_EMBEDDING_WGRAD = "defer_embedding_wgrad_compute under the ring schedule"
RING_PIPELINE_LAYOUT = (
    "pipeline_model_parallel_layout under the ring schedule (the '|' segments are the layout)"
)
RING_P2_WITHOUT_OVERLAP = "the ring at pipeline_model_parallel_size 2 without overlap_p2p_comm"
RING_RECOMPUTE_PLAN = (
    "loop_ring_recompute_plan requires one '/'-separated group per pipeline rank "
    "with one 'C' / 'n' code for each logical view"
)
RING_PLAN_WITHOUT_RECOMPUTE = "'C' in loop_ring_recompute_plan without loop_recompute"
# Layout rules (validate_ring_layout on the main pattern; resolve_hybrid_loop_ring_roles on one
# chunk).
RING_SEGMENTS = "'|' segments other than pipeline_model_parallel_size x the virtual pipeline size"
RING_NUM_LAYERS = "num_layers other than the main pattern's sublayer count"
RING_CUT_NOT_ON_BLOCK = "'|' cut not on a block boundary"
RING_PRELUDE_SPLIT = "prelude outside the first '|' segment"
RING_CODA_SPLIT = "coda outside the last '|' segment"
RING_SEGMENT_WITHOUT_CORE = "a '|' segment without a core block"
RING_CHUNK_RANGE = "a ring chunk outside the main pattern"
# Runtime rules (the loop stack).
# (Rule names never contain ", ": the error message joins them with it.)
RING_EAGER_CALL = (
    "a ring chunk called without the ring's call roles (it runs only as the ring schedule's "
    "round views: ring_pp.LoopRingRoundView)"
)
RING_CALL_ROLES = (
    "ring call roles that do not fit the chunk and round (the embedding only in round 1 of the "
    "round-start chunk; the head only in round R of the round-end chunk)"
)
RING_ROW_DEPTHS = "per-row loop depths (set_loop_depths) under the ring schedule"
RING_CARRIER_DIAGNOSTICS = "carrier diagnostics under the ring schedule"

# The recompute plan's codes, one per (stage, round view): 'C' checkpoints the round body (keeps
# only its inputs), 'n' keeps its activations.
RING_RECOMPUTE_CODES = ("C", "n")


def refuse_ring(*rules: str) -> None:
    """Raise ``ValueError("Loop ring does not support: <rule>, ...")`` for ``rules`` (if any)."""
    if rules:
        raise ValueError("Loop ring does not support: " + ", ".join(rules))


@dataclass(frozen=True)
class LoopRingChunk:
    """One chunk of a ring loop: the ``'|'`` segment a pipeline rank builds as one module (one of
    its k virtual-pipeline modules at k > 1).

    ``offset`` is the global index of the chunk's first sublayer (the stack's
    ``pp_layer_offset``). ``local`` counts the chunk's own prelude, core and coda sublayers, which
    lie in that order, so ``local.core_start`` / ``local.core_end`` index the chunk's layer list.
    ``round_start``: the chunk holds global sublayer 0 (the prelude and the loop adapter: every
    round starts here); ``round_end``: it holds the last sublayer (the coda and the loop norm:
    every round ends here). For separate edge laps these are the first and last ranks of
    the chunk's section. The role identifies which of the three sections it holds.
    """

    offset: int
    local: LoopRoles
    round_start: bool
    round_end: bool
    # Separate laps partition each of the three decoder sections across all PP ranks.
    role: Optional[str] = None


def ring_virtual_pipeline_size(config: Any) -> int:
    """k: the core slices per pipeline stage of a ring loop (the virtual pipeline size, 1
    without one)."""
    return getattr(config, "virtual_pipeline_model_parallel_size", None) or 1


def _ring_views(config: Any) -> int:
    """Logical stage count: R passes of k chunks, or R core passes and two edge laps."""
    if getattr(config, "loop_ring_separate_edges", False):
        return config.loop_iterations + 2
    return config.loop_iterations * ring_virtual_pipeline_size(config)


def ring_recompute_plan_violations(config: Any) -> List[str]:
    """The ``loop_ring_recompute_plan`` rules ``config`` breaks (empty without a plan)."""
    plan = getattr(config, "loop_ring_recompute_plan", None)
    if plan is None:
        return []
    groups = plan.split("/")
    views = _ring_views(config)
    violations = []
    if len(groups) != config.pipeline_model_parallel_size or any(
        len(group) != views or not set(group) <= set(RING_RECOMPUTE_CODES) for group in groups
    ):
        violations.append(RING_RECOMPUTE_PLAN)
    if "C" in plan and not config.loop_recompute:
        violations.append(RING_PLAN_WITHOUT_RECOMPUTE)
    return violations


def resolve_loop_ring_recompute_plan(config: Any) -> Tuple[str, ...]:
    """Return a recompute code per logical view for each pipeline stage in rank order.

    Without an explicit plan every view is 'C' with ``loop_recompute`` and 'n' without.
    """
    refuse_ring(*ring_recompute_plan_violations(config))
    plan = getattr(config, "loop_ring_recompute_plan", None)
    if plan is not None:
        return tuple(plan.split("/"))
    code = "C" if config.loop_recompute else "n"
    return tuple(code * _ring_views(config) for _ in range(config.pipeline_model_parallel_size))


def validate_loop_ring_config(config: "TransformerConfig") -> None:
    """Reject configs the ring schedule (``loop_schedule='ring'``) cannot run.

    Called by ``TransformerConfig.__post_init__`` with ``loop_schedule='ring'`` and by the loop
    stack's constructor. Collects every violation and raises one ``ValueError`` (``"Loop ring
    does not support: <name>, <name>"``). The ring runs a fixed-depth Huginn loop with the
    injection carrier over P >= 2 pipeline stages; each microbatch runs exactly
    ``loop_iterations`` rounds, the shared core once per round on every stage. The eager
    hybrid-loop guards still apply (``validate_hybrid_loop_config`` except its pipeline rule, the
    DDP guards): context / sequence parallelism, CUDA graphs, EP overlap, activation offload and
    paged stash, recompute other than ``loop_recompute`` and overlapped grad reduce / param gather
    stay refused there; the RSBF-only fields by ``TransformerConfig`` ("require
    loop_schedule='rsbf'") and the loop parameter store by ``validate_param_store_config``. The
    ``'|'`` layout is ``validate_ring_layout``'s.
    """
    stages = config.pipeline_model_parallel_size
    separate_edges = getattr(config, "loop_ring_separate_edges", False)
    unsupported = {
        RING_WITHOUT_HYBRID_ROLES: not is_hybrid_loop_config(config),
        RING_WITHOUT_PP: stages < 2,
        # The round views run Huginn rounds of one tied core with one adapter and one norm.
        RING_MOR: config.loop_architecture == "mor",
        RING_CARRIER: not separate_edges
        and getattr(config, "loop_carrier", "injection") != "injection",
        "separate edge laps require native GLM gated-residual DSA and three physical chunks": separate_edges
        and not (
            config.experimental_attention_variant == "dsa"
            and config.loop_carrier == "gated_residual"
            and ring_virtual_pipeline_size(config) == 3
            and config.loop_iterations >= 1
        ),
        RING_UNTIED_CORE: bool(getattr(config, "loop_untie_core", False)),
        "online stopping requires separate GLM edge laps": (
            getattr(config, "loop_exit_criterion", None) == "online" and not separate_edges
        ),
        # Every microbatch runs exactly loop_iterations rounds (the schedule's views), each with
        # its full backward.
        RING_SAMPLE_DEPTH: config.loop_sample_depth,
        RING_TRUNCATED_BPTT: config.loop_bptt_iterations > 0,
        # Untested: TP collectives inside every round view.
        RING_TP: config.tensor_model_parallel_size > 1,
        # Dropless routing: a static per-rank budget drops the tokens that overflow it.
        RING_STATIC_BUDGET_MOE: getattr(config, "moe_expert_rank_capacity_factor", None)
        is not None,
        # The deferred output-layer wgrad is finished on the schedule's model chunks, which are
        # the parameterless round views.
        RING_DEFER_EMBEDDING_WGRAD: bool(config.defer_embedding_wgrad_compute),
        RING_PIPELINE_LAYOUT: getattr(config, "pipeline_model_parallel_layout", None) is not None,
        # The ring always runs the interleaved schedule; at PP 2 both directions use one peer
        # pair, which batched P2P cannot post in one batch (the virtual-pipeline rule).
        RING_P2_WITHOUT_OVERLAP: stages == 2 and not config.overlap_p2p_comm,
        **{rule: True for rule in ring_recompute_plan_violations(config)},
    }
    refuse_ring(*[name for name, value in unsupported.items() if value])


def _ring_chunk(
    roles: LoopRoles, offset: int, length: int
) -> Tuple[Optional[LoopRingChunk], List[str]]:
    """The chunk holding sublayers ``[offset, offset + length)`` of a ring loop with the model
    roles ``roles`` (None when it breaks a rule) and the ring rules it breaks."""
    total = roles.prelude + roles.core + roles.coda
    end = offset + length
    if length < 1:
        return None, [RING_SEGMENT_WITHOUT_CORE]
    if offset < 0 or end > total:
        return None, [RING_CHUNK_RANGE]
    prelude = max(0, min(end, roles.core_start) - offset)
    core = max(0, min(end, roles.core_end) - max(offset, roles.core_start))
    coda = max(0, end - max(offset, roles.core_end))
    violations = []
    # The prelude and the coda run once per microbatch: each lies whole in one chunk.
    if 0 < prelude < roles.prelude:
        violations.append(RING_PRELUDE_SPLIT)
    if 0 < coda < roles.coda:
        violations.append(RING_CODA_SPLIT)
    # Every chunk runs a slice of every round.
    if core == 0:
        violations.append(RING_SEGMENT_WITHOUT_CORE)
    if violations:
        return None, violations
    local = LoopRoles(prelude=prelude, core=core, coda=coda)
    return LoopRingChunk(offset, local, round_start=offset == 0, round_end=end == total), []


def _separate_ring_chunk(config, roles, offset, length, rank):
    """Identify a physical GLM slice and keep each cross-layer selector group local."""
    stages = config.pipeline_model_parallel_size
    if rank is None or not 0 <= rank < stages:
        raise ValueError("Separate ring chunks require their physical pipeline rank")
    if offset < roles.core_start:
        role, start, stop = "prelude", 0, roles.core_start
    elif offset < roles.core_end:
        role, start, stop = "core", roles.core_start, roles.core_end
    else:
        role, start, stop = "coda", roles.core_end, config.num_layers
    end = offset + length
    if not (start <= offset <= end <= stop):
        raise ValueError("A separate ring chunk cannot cross a prelude/core/coda boundary")
    if length == 0 and role != "coda":
        raise ValueError("Only coda stages may pass through an empty decoder chunk")
    from megatron.core.transformer.experimental_attention_variant.dsa import is_dsa_skip_topk_layer

    selector_offset = config.dsa_indexer_skip_topk_offset if role == "prelude" else start // 2 + 1
    for cut in (offset, end):
        if cut % 2:
            # Leave the last DSA of a selector group with its producer. Only its
            # standalone FFN may move across the pipeline boundary. The next DSA
            # must produce its own indices, unless this is the final core FFN.
            next_dsa = (cut + 1) // 2 + 1
            if role != "core" or (
                next_dsa <= stop // 2
                and is_dsa_skip_topk_layer(next_dsa, selector_offset, config.dsa_indexer_topk_freq)
            ):
                refuse_ring(RING_CUT_NOT_ON_BLOCK)
    if length:
        first_dsa = (offset + 1) // 2 + 1
        if 2 * (first_dsa - 1) < end and is_dsa_skip_topk_layer(
            first_dsa, selector_offset, config.dsa_indexer_topk_freq
        ):
            raise ValueError(
                f"{role} chunk starts inside a DSA top-k sharing group at layer {first_dsa}"
            )
    local = LoopRoles(
        length if role == "prelude" else 0,
        length if role == "core" else 0,
        length if role == "coda" else 0,
    )
    return LoopRingChunk(offset, local, rank == 0, rank == stages - 1, role=role)


def validate_ring_layout(
    config: "TransformerConfig", main_pattern: str, hash_moe_layer_threshold: Optional[int]
) -> List[LoopRingChunk]:
    """Check the ``'|'`` layout of a ring loop's main pattern; return its chunks in segment order.

    The pipe-free pattern must pass the hybrid loop's role rules (``resolve_hybrid_loop_roles``:
    role cuts on block boundaries, the hash-routed MoE layers inside the prelude, supported
    sublayers; raised as "Hybrid loop does not support: ..."). Then the ring rules, collected
    into one "Loop ring does not support: ..." error: P x k segments (``RING_SEGMENTS``; k, the
    virtual pipeline size, exists only as ring sub-chunks), ``num_layers`` equal to the pipe-free
    sublayer count (the stacks derive the model roles from it), every cut on a block boundary,
    the prelude inside the first segment and the coda inside the last one, and a core block in
    every segment. With the prelude in the first segment, the hash-routed layers lie on the stage
    that owns the token ids. Called by ``HybridModel.__init__`` with ``loop_schedule='ring'``
    on every rank, before any layer is built.
    """
    segments = main_pattern.split(Symbols.PIPE)
    symbols = list(main_pattern.replace(Symbols.PIPE, ""))
    roles = resolve_hybrid_loop_roles(config, symbols, hash_moe_layer_threshold)
    violations = []
    if len(segments) != config.pipeline_model_parallel_size * ring_virtual_pipeline_size(config):
        violations.append(RING_SEGMENTS)
    if config.num_layers != len(symbols):
        violations.append(RING_NUM_LAYERS)
    chunks = []
    offset = 0
    separate_edges = getattr(config, "loop_ring_separate_edges", False)
    for index, segment in enumerate(segments):
        if offset and not is_block_boundary(symbols, offset):
            # The GLM core may send hidden states from the final DSA of a
            # selector group to its FFN. All other cuts retain the block rule.
            tail_ffn = (
                separate_edges
                and roles.core_start < offset < roles.core_end
                and offset % 2 == 1
                and symbols[offset - 1] == Symbols.DS_ATTENTION
                and symbols[offset] in _BLOCK_END
            )
            if not tail_ffn:
                violations.append(RING_CUT_NOT_ON_BLOCK)
        if separate_edges:
            rank = index % config.pipeline_model_parallel_size
            chunk = _separate_ring_chunk(config, roles, offset, len(segment), rank)
            section = index // config.pipeline_model_parallel_size
            if section >= 3 or chunk.role != ("prelude", "core", "coda")[section]:
                raise ValueError(
                    "Separate ring patterns must contain P prelude, P core and P coda chunks"
                )
        else:
            chunk, broken = _ring_chunk(roles, offset, len(segment))
            violations += broken
        chunks.append(chunk)
        offset += len(segment)
    refuse_ring(*dict.fromkeys(violations))
    return chunks


def resolve_hybrid_loop_ring_roles(
    config: "TransformerConfig",
    layer_symbols: Sequence[str],
    pp_layer_offset: int,
    hash_moe_layer_threshold: Optional[int],
    *,
    pp_rank: Optional[int] = None,
) -> Tuple[LoopRoles, LoopRingChunk]:
    """The model roles and this chunk's roles of a ring loop stack (``loop_schedule='ring'``).

    The stack holds sublayers ``[pp_layer_offset, pp_layer_offset + len(layer_symbols))`` of the
    main pattern. The model roles come from the global counts (``num_layers`` sublayers in the
    main pattern, ``loop_prelude_sublayers``, ``loop_coda_sublayers``) and the chunk's roles from
    its offset: no pipeline rank is read, so the same code builds every chunk, on its rank or all
    in one process. Checks what one chunk can: its sublayer kinds (``UNSUPPORTED_SUBLAYERS``),
    the hash-routed layers inside the prelude (``HASH_BEYOND_PRELUDE``), its role cuts on block
    boundaries (``CUT_NOT_ON_BLOCK``), then the ring rules (its ends on block boundaries, the
    prelude / coda not split, a core block, inside the pattern). The whole layout is
    ``validate_ring_layout``'s.
    """
    symbols = list(layer_symbols)
    num_layers = config.num_layers
    prelude, coda = config.loop_prelude_sublayers, config.loop_coda_sublayers
    violations = []
    roles = None
    if prelude is None or coda is None or prelude < 0 or coda < 0 or prelude + coda >= num_layers:
        violations.append(ROLES_DO_NOT_MATCH)
    else:
        roles = LoopRoles(prelude=prelude, core=num_layers - prelude - coda, coda=coda)
    if set(symbols) & _UNSUPPORTED_SYMBOLS:
        violations.append(UNSUPPORTED_SUBLAYERS)
    if roles is not None and hash_moe_layer_threshold and hash_moe_layer_threshold > roles.prelude:
        violations.append(HASH_BEYOND_PRELUDE)
    if roles is not None and getattr(config, "loop_ring_separate_edges", False):
        _raise_if_any(violations)
        return roles, _separate_ring_chunk(config, roles, pp_layer_offset, len(symbols), pp_rank)
    chunk, ring = (None, []) if roles is None else _ring_chunk(roles, pp_layer_offset, len(symbols))
    if chunk is not None:
        local = chunk.local
        if not (
            is_block_boundary(symbols, local.core_start)
            and is_block_boundary(symbols, local.core_end)
        ):
            violations.append(CUT_NOT_ON_BLOCK)
        # The chunk's own ends are '|' cuts (except at the ends of the main pattern).
        if (not chunk.round_start and symbols[0] not in _BLOCK_START) or (
            not chunk.round_end and symbols[-1] not in _BLOCK_END
        ):
            ring.append(RING_CUT_NOT_ON_BLOCK)
    _raise_if_any(violations)
    refuse_ring(*ring)
    return roles, chunk
