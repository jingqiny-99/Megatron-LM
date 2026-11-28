# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Depth recurrence over a HybridStack: the eager reference of the hybrid loop.

``LoopHybridStack`` is a ``HybridStack`` whose layer loop runs the main pattern in three roles
(counted in sublayers, see ``guards.py``): a prelude ``[0, P)`` once, a core ``[P, P + C)``
once per round, and a coda ``[P + C, L)`` once. Everything else is the stack's own code: the
mHC stream expansion and contraction, the final norm, the MTP multistream output, and every
layer call (``HybridStack._run_layer``). Two recurrences are supported, as on the GPT loop
(``loop_transformer_block.py``, whose round bodies are re-hosted here):

* Huginn: ``state_0 = 0``; each round runs ``loop_norm(core(loop_adapter([state, e])))`` with
  ``e`` the prelude output. Uniform depth, or per-row depth (``set_loop_depths``): a row that
  has exited keeps its state (self-carry) while the others continue. ``loop_carrier``
  selects what carries the state: ``'injection'`` (that adapter and norm, the default) or
  ``'residual'`` (``state_0 = e``, each round ``core(state)`` on one continuing residual
  stream, no adapter or norm: the ``w = 0`` reference of the attention-residual carrier, which
  is ``attn_res_stack.LoopAttnResHybridStack``). With ``'gated_residual'`` each round
  interpolates between its input and its candidate output using its own token-wise gate.
* MoR (token-choice mixture of recursions, dense cores only): a router picks a depth per token;
  round ``r`` runs the core on the tokens with depth ``>= r`` (masked or compacted).

Untied core (``loop_untie_core``, the carrier ablation's ``plain-inject``: a non-loop baseline
with the same injection operator): the physical core holds ``R = loop_iterations`` equal
segments cut on block boundaries (``UNTIED_CORE_SEGMENTS``); round ``r`` runs segment ``r``
with its own adapter ``loop_adapters[r]`` and norm ``loop_norms[r]``, so the forward is the
injection carrier's Huginn recurrence with untied weights. Fixed depth only (the config's
"Untied loop baseline requires fixed-depth Huginn" rule; ``set_loop_depths`` raises).

Eager only: no CUDA graphs, no context / sequence parallelism, and pipeline parallelism only as
the ring (see the guards).

Ring (``loop_schedule='ring'``, PP > 1; ``guards.validate_ring_layout``): the main pattern carries
P x k ``'|'`` segments and each stack is one of them (a chunk). ``loop_roles`` stay the model's
roles (from ``num_layers`` and the role fields); ``loop_ring`` (``guards.LoopRingChunk``) holds
the chunk's offset, its own prelude / core / coda sublayers and whether it starts a round (it
holds the prelude and builds ``loop_adapter``) or ends one (it holds the coda and builds
``loop_norm``); the other chunks set those two to None. Core sublayer modules that track
``is_first_microbatch`` are marked ``is_repeated_layer`` (a core runs once per round). A ring chunk
runs only through the ring schedule's round views (``ring_pp.LoopRingRoundView``): each call
carries a ``ring_pp.LoopRingCall`` (``HybridStack.call_roles``) and runs the chunk's slice of one
round of one microbatch (``_ring_layer_loop``: the prelude and the injection store in round 1 on
the round-start chunk, the round body, the coda in round R on the round-end chunk). Called
without those roles, with per-row depths or with carrier diagnostics it raises.

Activation recompute (``loop_recompute``): one non-reentrant checkpoint per round (Huginn) or
recurrence (MoR) by default. With ``loop_recompute_block_stride`` a Huginn round is cut into
checkpointed blocks of that many core sublayers (the adapter in the first block, the loop norm
in the last), and with ``loop_recompute_edges`` the prelude and the coda are checkpointed in
blocks of the same stride (``_checkpointed_blocks``). Each checkpoint keeps only its inputs and
has its own routing tape; the math is unchanged (bitwise with deterministic kernels).

Checkpoint keys (M3c). At PP = 1 (guarded) a sharded key equals the module path; on a ring chunk
the sublayer keys carry the global sublayer index (``HybridStack``'s ``layer_number``), so the
chunks' keys together are the PP = 1 keys, ``loop_adapter`` and ``loop_norm`` once each. The
loop adds no key code of its own: ``HybridStack.sharded_state_dict`` writes sublayer ``i`` as
``decoder.layers.i`` (mHC-wrapped sublayers nest their layer under ``inner_layer.`` next to
``hyper_connection.*``; attention and MoE are separate sublayer indices; prelude / core / coda
are roles in the config, not part of any key) and gives every other child and the stack's own
tensors a replicated entry through the default walk:

* ``decoder.loop_adapter.weight``, ``[h, 2h]``: the Huginn adapter (D34: shared by the mHC
  streams);
* ``decoder.loop_norm.weight``, ``[h]`` (``LoopRMSNorm``, one stream) or ``[n, h]``
  (``LoopStreamRMSNorm``, D34's per-stream norm);
* ``decoder.loop_adapters.{r}.weight`` ``[h, 2h]`` and ``decoder.loop_norms.{r}.weight``
  instead of the two above for an untied core (``loop_untie_core``), ``r`` over the rounds;
* ``decoder.depth_router.0.weight`` ``[h, h]`` and ``decoder.depth_router.2.weight``
  ``[R_max, h]``: the MoR router;
* ``decoder.loop_attn_res.{j}.pseudo_query`` / ``.key_norm_weight`` (``j`` over the core and
  coda sublayers) and ``decoder.loop_attn_res_head.*``, ``[h]`` fp32: the attention-residual
  carrier (``LoopAttnResHybridStack``; no ``loop_adapter`` / ``loop_norm``; the residual
  carrier has no loop parameter at all);
* ``decoder.<name>`` / ``decoder.<child>.<...>``: anything registered on the stack later.

A parameter or persistent buffer registered on the stack (a direct ``Parameter``, a child
module, an ``nn.ParameterList``) or on a module inside a layer gets its entry without new code;
``param_store_checkpoint.uncovered_state_names`` (raised on by ``megatron/training/
loop_checkpoint.py`` at setup, the first save and every load) catches a module whose custom
``sharded_state_dict`` drops one. The exit contraction is the stack's own ``hc_head_*`` (dev's
keys); the cells own no parameters. With the loop parameter store installed, every stored weight
keeps the same key and global shape: it is saved as row shards of the logical tensor
(``param_store_checkpoint``; ``LoopHybridStack.sharded_state_dict`` / ``HybridModel.
sharded_state_dict`` build in the store's logical view), so the keys do not depend on z.
"""

import dataclasses
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackLayerContext
from megatron.core.models.hybrid.layers.utils import Symbols
from megatron.core.tensor_observation import observe_layer_residuals
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
    indexer_loss_padding_mask,
)
from megatron.core.transformer.forward_replay import forward_statistics_round, in_forward_replay
from megatron.core.transformer.loop import flops
from megatron.core.transformer.loop.cross_loop_kv import CrossLoopPass
from megatron.core.transformer.loop.guards import (
    RING_CALL_ROLES,
    RING_CARRIER_DIAGNOSTICS,
    RING_EAGER_CALL,
    RING_RECOMPUTE_CODES,
    RING_ROW_DEPTHS,
    LoopRingChunk,
    LoopRoles,
    has_dsv4_attention,
    has_module_gtp_shards,
    is_ring_loop_config,
    refuse_ring,
    resolve_hybrid_loop_ring_roles,
    resolve_hybrid_loop_roles,
    round_sublayers,
    validate_hybrid_loop_config,
    validate_hybrid_loop_inputs,
    validate_loop_carrier,
    validate_loop_carrier_stack,
    validate_loop_gtp_shards,
    validate_loop_mtp_pattern,
    fixed_glm_loop,
    fixed_glm_standard,
    validate_loop_ring_config,
    validate_loop_row_depths,
    validate_loop_stack_construction,
)
from megatron.core.transformer.loop.online_exit import online_exit_decision
from megatron.core.transformer.loop.reporting import loop_range, stream_mean_rms
from megatron.core.transformer.loop.ring_pp import (
    LoopRingCall,
    RingCrossLoopPass,
    RingInjection,
    ring_injection_source,
    ring_injection_use,
)
from megatron.core.transformer.loop_transformer_block import (
    LoopRMSNorm,
    active_causal_mask,
    compact_metadata,
    loop_checkpoint_context_fn,
    loop_round_checkpoint,
    token_depth_distribution,
)


class LoopStreamRMSNorm(nn.Module):
    """``LoopRMSNorm`` applied per mHC stream: FP32 statistics and a ``[n, h]`` scale.

    The input is ``[..., n * h]`` with the streams as contiguous ``h`` chunks, the layout of
    ``HyperConnectionModule.input_expand``.
    """

    def __init__(self, width, num_streams, eps, dtype, device):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_streams, width, dtype=dtype, device=device))
        self.width = width
        self.num_streams = num_streams
        self.eps = eps

    def forward(self, x):
        """Normalize each stream's last dimension in FP32 and apply its learned scale."""
        streams = x.reshape(*x.shape[:-1], self.num_streams, self.width)
        y = streams.float() * torch.rsqrt(
            streams.float().square().mean(-1, keepdim=True) + self.eps
        )
        return (y.to(x.dtype) * self.weight).reshape(x.shape)


class LoopResidualGate(nn.Module):
    """Write a core pass's candidate back to its input with one learned scalar per token.

    GLM-5.4's diagram uses sigmoid(linear(RMSNorm(pass input))). For a DSv4 mHC
    proxy, read the mean of its streams and broadcast the token's gate to every stream.
    """

    def __init__(self, config, streams, device):
        super().__init__()
        self.streams = streams
        h, dtype = config.hidden_size, config.params_dtype
        self.norm = LoopRMSNorm(h, config.layernorm_epsilon, dtype, device)
        self.proj = nn.Linear(h, 1, bias=False, device=device, dtype=dtype)
        config.init_method(self.proj.weight)

    def forward(self, previous, candidate):
        """The gate reads the incoming state, before this pass's shared core."""
        gate_input = previous
        if self.streams > 1:
            gate_input = previous.unflatten(-1, (self.streams, -1)).mean(dim=-2)
        normalized = self.norm(gate_input).to(self.proj.weight.dtype)
        beta = self.proj(normalized).sigmoid().to(previous.dtype)
        return torch.lerp(previous, candidate, beta)


def _local_glm_full_recompute(config):
    return getattr(config, "recompute_granularity", None) == "full" and fixed_glm_standard(config)


def _local_glm_visit_stride(config, lap):
    sizes = getattr(config, "loop_ring_recompute_group_sizes", None)
    return config.recompute_num_layers if sizes is None else sizes[lap]


class LoopHybridStack(HybridStack):
    """HybridStack whose layer loop runs prelude, core x rounds and coda (eager reference).

    Same constructor as ``HybridStack``. The loop parameters are children of the stack, named
    as on the GPT loop block: ``loop_adapter`` (``[h, 2h]``, shared by every mHC stream) and
    ``loop_norm`` for Huginn, ``depth_router`` for MoR.

    Per forward, ``loop_losses`` (MoR router losses) and ``loop_metrics`` are rebound to new
    dicts; ``loop_metrics`` carries ``loop {forward,backward,recompute} flops per token``
    (per-symbol FLOP table, ``flops.py``) for the training FLOP counter. The token-layer
    metrics count **sublayers**: a dense ``*-`` model reports twice the GPT loop's values.
    ``loop_metric_populations`` (also rebound per forward) holds the population size of each
    metric not defined over the issued tokens (the MoR metrics: valid tokens); the loss
    function reports every loop metric by ``loop/reporting.py``'s rule.
    """

    # The loop parameter store's visit hook (``LoopParamStore.install`` sets it on the instance):
    # every sublayer call of ``_run_range`` and the adapter call of ``_adapt`` run inside its
    # segment visits. None without a store (the stage-5 path, op for op).
    _loop_store_hook = None
    # The ``loop_carrier`` values this class runs (``LoopAttnResHybridStack``: 'attn_res').
    _LOOP_CARRIERS = ("injection", "residual", "gated_residual")

    def __init__(
        self,
        config,
        submodules,
        pre_process=True,
        layer_type_list=None,
        pp_layer_offset=0,
        post_layer_norm=True,
        post_process=True,
        device=None,
        dtype=None,
        pg_collection=None,
        is_mtp_layer=False,
        mtp_layer_number=None,
        hash_moe_layer_threshold=None,
        name=None,
        layer_config_list=None,
        boundary_layout=None,
        vp_stage=None,
    ):
        validate_loop_stack_construction(config, is_mtp_layer)
        validate_loop_carrier_stack(config, type(self)._LOOP_CARRIERS)
        validate_loop_carrier(config)
        validate_hybrid_loop_config(config)
        ring = is_ring_loop_config(config)
        if ring:
            validate_loop_ring_config(config)
        super().__init__(
            config=config,
            submodules=submodules,
            pre_process=pre_process,
            layer_type_list=layer_type_list,
            pp_layer_offset=pp_layer_offset,
            vp_stage=vp_stage,
            post_layer_norm=post_layer_norm,
            post_process=post_process,
            device=device,
            dtype=dtype,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            mtp_layer_number=mtp_layer_number,
            hash_moe_layer_threshold=hash_moe_layer_threshold,
            name=name,
            layer_config_list=layer_config_list,
            boundary_layout=boundary_layout,
        )
        self._online_exit_records = []
        # Without Shortcut MoE (guarded) self.layers is indexed by the physical sublayer.
        assert self._execution_layer_indices == list(range(len(self.layers)))
        validate_loop_gtp_shards(config, self)
        if has_module_gtp_shards(self):
            # Module GTP at a fixed depth: a core weight is consumed once per round, in the
            # forward and again in the recompute, which the static prefetch chains (built for
            # one consume per pass) do not follow. Consumes check their prefetched buffers and
            # gather on demand otherwise (process-global GTP setting, read at every consume).
            from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTP_CONFIG

            GTP_CONFIG.verify_prefetched_gathers = True
        # PP = 1 (guarded outside the ring): the stack holds the whole main pattern, layer numbers
        # are i + 1. The ring (loop_schedule='ring', PP > 1): the stack is one '|' segment, layer
        # numbers i + 1 + pp_layer_offset; loop_roles are the model's (from the global counts)
        # and loop_ring this chunk's own prelude / core / coda and round flags.
        self.loop_ring: Optional[LoopRingChunk] = None
        if ring:
            self.loop_roles, self.loop_ring = resolve_hybrid_loop_ring_roles(
                config,
                self.layer_type_list,
                pp_layer_offset,
                hash_moe_layer_threshold,
                pp_rank=(
                    self.pg_collection.pp.rank()
                    if getattr(config, "loop_ring_separate_edges", False)
                    else None
                ),
            )
        else:
            self.loop_roles: LoopRoles = resolve_hybrid_loop_roles(
                config, self.layer_type_list, hash_moe_layer_threshold
            )
        self._cross_loop_attention_layers = (
            self._validate_cross_loop_attention_targets()
            if config.loop_reuse_sparse_kv or config.loop_reuse_indexer
            else {}
        )
        self._loop_symbols = tuple(self.layer_type_list)
        self._has_dsv4_attention = has_dsv4_attention(self._loop_symbols)
        self._loop_flop_rows = [
            (symbol, layer_config, self._compress_ratio(symbol, index))
            for index, (symbol, layer_config) in enumerate(
                zip(self._loop_symbols, self.layer_config_list, strict=True)
            )
        ]
        self.loop_num_streams = (
            config.mhc_num_residual_streams if config.enable_mhc_connections else 1
        )

        if device is not None:
            loop_device = device  # e.g. meta-device initialization
        elif config.use_cpu_initialization:
            loop_device = "cpu"
        else:
            loop_device = torch.cuda.current_device()
        self._loop_device = loop_device
        loop_dtype = config.params_dtype
        hidden = config.hidden_size
        if config.loop_carrier == "gated_residual":
            # Only the last physical rank of the shared core mixes the two full-pass outputs.
            # Its two gates are distinct and keep the same checkpoint names as the PP1 model.
            owns_gates = not getattr(config, "loop_ring_separate_edges", False) or (
                self.loop_ring.role == "core" and self.loop_ring.round_end
            )
            self.loop_gates = nn.ModuleList(
                LoopResidualGate(config, self.loop_num_streams, loop_device)
                for _ in range(config.loop_iterations if owns_gates else 0)
            )
        elif config.loop_architecture == "huginn" and config.loop_carrier != "injection":
            pass  # the residual / attention-residual carriers have no adapter and no norm
        elif config.loop_architecture == "huginn":

            def adapter():
                linear = nn.Linear(
                    2 * hidden, hidden, bias=False, dtype=loop_dtype, device=loop_device
                )
                config.init_method(linear.weight)
                return linear

            def norm():
                if self.loop_num_streams == 1:
                    return LoopRMSNorm(hidden, config.layernorm_epsilon, loop_dtype, loop_device)
                return LoopStreamRMSNorm(
                    hidden, self.loop_num_streams, config.layernorm_epsilon, loop_dtype, loop_device
                )

            if config.loop_untie_core:
                # One adapter and one norm per untied segment (round).
                rounds = range(config.loop_iterations)
                self.loop_adapters = nn.ModuleList([adapter() for _ in rounds])
                self.loop_norms = nn.ModuleList([norm() for _ in rounds])
            else:
                # Ring: the adapter opens every round, on the round-start chunk only; the norm
                # closes it, on the round-end chunk only (one copy of each in the model; None on
                # the other chunks).
                chunk = self.loop_ring
                self.loop_adapter = adapter() if chunk is None or chunk.round_start else None
                self.loop_norm = norm() if chunk is None or chunk.round_end else None
        else:
            self.depth_router = nn.Sequential(
                nn.Linear(hidden, hidden, bias=False, dtype=loop_dtype, device=loop_device),
                nn.SiLU(),
                nn.Linear(
                    hidden, config.loop_iterations, bias=False, dtype=loop_dtype, device=loop_device
                ),
            )
            for module in self.depth_router:
                if isinstance(module, nn.Linear):
                    config.init_method(module.weight)
            if config.loop_token_execution == "masked":
                for layer in self.layers[self.loop_roles.core_start : self.loop_roles.core_end]:
                    attention = getattr(layer, "self_attention", None)
                    if hasattr(attention, "attn_mask_type"):  # '*' sublayers only
                        attention.attn_mask_type = AttnMaskType.arbitrary
        if self.loop_ring is not None or fixed_glm_standard(config):
            # A shared core is called on every visit. Its first forward can run last in
            # backward, so the TE first-microbatch fast path must not overwrite later visits.
            local = self.loop_ring.local if self.loop_ring is not None else self.loop_roles
            for layer in self.layers[local.core_start : local.core_end]:
                for module in layer.modules():
                    if hasattr(module, "is_first_microbatch"):
                        module.is_repeated_layer = True
        # Sublayers one round runs: the whole core, or one of the R untied segments.
        self._round_core = round_sublayers(config, self.loop_roles)
        self.loop_losses = {}
        self.loop_metrics = {}
        self.loop_metric_populations = {}
        self._microbatch_id = 0
        self._loop_depths: Optional[Tensor] = None
        self._loop_head = None
        self._work = [0.0, 0.0, 0.0]
        # Carrier diagnostics (``enable_carrier_diagnostics``): off by default.
        self._carrier_diagnostics = False
        self._diagnostic_metrics = {}
        # Ring chunk state (``_ring_layer_loop``): the injection store of the round-start chunk
        # (microbatch -> ring_pp.RingInjection), the G1 counters (microbatch -> [core slice runs,
        # prelude runs, coda runs, MTP runs]), the FLOPs the calls recorded (forward, backward,
        # recompute) and the whole model's per-microbatch FLOPs by (sequence, batch), which the
        # head call reports.
        self._ring_injections = {}
        self._ring_counts = {}
        self._ring_work = [0.0, 0.0, 0.0]
        self._ring_model_work = {}
        self._ring_mtp_hook = None

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: Optional[tuple] = None, metadata=None
    ):
        """``HybridStack.sharded_state_dict``; with a loop parameter store installed, in its
        logical view (stored weights saved as row shards of the logical tensor under their own
        keys; a nested call from ``HybridModel.sharded_state_dict`` converts once, there)."""
        store = self.__dict__.get("_loop_store_hook")
        if store is None:
            return super().sharded_state_dict(prefix, sharded_offsets, metadata)
        return store.logical_sharded_state_dict(
            self,
            lambda: super(LoopHybridStack, self).sharded_state_dict(
                prefix, sharded_offsets, metadata
            ),
            prefix=prefix,
        )

    def _validate_cross_loop_attention_targets(self) -> dict[int, tuple[int, bool]]:
        """Check built modules because array-driven 'D' and per-layer specs choose their banks."""
        from megatron.core.transformer.experimental_attention_variant.csa import (
            CompressedSparseAttention,
        )

        ring = getattr(self, "loop_ring", None)
        roles = ring.local if ring is not None else self.loop_roles
        compressed = {
            index: (module.compress_ratio, module.indexer is not None)
            for index in range(roles.core_start, roles.core_end)
            for module in self.layers[index].modules()
            if isinstance(module, CompressedSparseAttention) and module.compressor is not None
        }
        if ring is not None:
            # The schedule checks the union of physical targets over all chunks and PP ranks.
            return compressed
        if self.config.loop_reuse_sparse_kv:
            if not compressed:
                raise ValueError("loop_reuse_sparse_kv requires a compressor in the recurrent core")
        elif not any(has_indexer for _, has_indexer in compressed.values()):
            raise ValueError("loop_reuse_indexer requires a learned indexer in the recurrent core")
        return compressed

    def _compress_ratio(self, symbol, index):
        """The DSv4 compress ratio of a ``D`` sublayer, read from its built attention.

        The attention resolves the ratio itself (an explicit spec parameter, then the layer
        config's ``compress_ratio``, then ``csa_compress_ratios``); reading it back keeps the
        FLOP row on the module's own ratio. A ``D`` sublayer not built as the DSv4 hybrid
        attention (e.g. ``loop_hybrid_stack_spec`` wires the legacy ``DSAttention``) has no
        row in the FLOP table and is rejected.
        """
        if symbol != Symbols.DS_ATTENTION:
            return None
        layer = self.layers[index]
        attention = getattr(getattr(layer, "inner_layer", layer), "self_attention", None)
        ratio = getattr(attention, "_dsv4_compress_ratio", None)
        if ratio is None:
            raise ValueError(
                "A 'D' sublayer in a hybrid loop must be built as the DSv4 hybrid attention "
                "(loop_hybrid_dsv4_stack_spec, experimental_attention_variant='dsv4_hybrid'); "
                f"sublayer {index} is {type(attention).__name__}"
            )
        return ratio

    def set_loop_depths(self, depths: Optional[Tensor]) -> None:
        """Set per-row depths for the next forward only (Huginn row exit).

        Args:
            depths: ``[B]`` integer tensor with values in ``1..loop_iterations``, on the
                activations' device, or None for uniform depth. Row ``b`` runs ``depths[b]``
                rounds and then keeps its state; the stack runs ``max(depths)`` rounds (agreed
                across the TP x EP group). Requires full BPTT (``loop_bptt_iterations == 0``).
                Depths must be valid on every rank: a rank that rejects its depths raises
                before the round-count all-reduce, which its TP x EP peers then wait on.

        This is the eager oracle of row exit, not the executor's: exited rows still run every
        later round and are discarded by the self-carry. The MoE router and the SBHD indexer
        loss of DSv4 attentions see them as padding (``padding_mask``, through
        ``_run_range``'s ``indexer_loss_padding_mask``; the fused SBHD indexer kernel has no
        row mask and refuses it); with the alltoall dispatcher they are still dispatched to
        experts. While every row of the TP x EP group is alive the round runs exactly as
        uniform depth (no padding mask is added), so equal row depths equal uniform depth
        bitwise.

        Raises ``ValueError`` on an untied core (``loop_untie_core``: a fixed-depth baseline),
        on a ring chunk, on a stack wrapped by Megatron-FSDP (``guards.FSDP_ROW_DEPTHS``: data-parallel ranks
        would issue different parameter gathers) and on a stack holding module-GTP shards
        (``guards.GTP_ROW_DEPTHS``: their per-use grad-ready count would change per step).
        """
        if depths is not None and self.config.loop_untie_core:
            raise ValueError("Untied loop baseline requires fixed-depth Huginn (set_loop_depths)")
        if depths is not None:
            if self.config.loop_reuse_sparse_kv or self.config.loop_reuse_indexer:
                raise ValueError("Cross-loop attention reuse requires uniform fixed depth")
            if self.config.loop_carrier == "gated_residual":
                raise ValueError("gated_residual requires uniform fixed depth")
            if self.config.context_parallel_size > 1:
                raise ValueError("Context-parallel loop ranks require uniform fixed depth")
            if self.loop_ring is not None:
                refuse_ring(RING_ROW_DEPTHS)
            validate_loop_row_depths(self)
        self._loop_depths = depths

    def set_loop_head_flops(
        self, vocab_size: int, mtp_pattern: Optional[str], mtp_num_depths: int
    ) -> None:
        """Include the output layer and the MTP depths in the reported FLOPs.

        ``HybridModel`` calls this after building the decoder; a stack used on its own reports
        its layers only.
        """
        validate_loop_mtp_pattern(
            mtp_pattern if mtp_num_depths else None,
            native_dsa=self.config.experimental_attention_variant == "dsa",
        )
        self._loop_head = (
            vocab_size,
            tuple(mtp_pattern or "") if mtp_num_depths else (),
            mtp_num_depths,
        )

    def _layer_loop(self, hidden_states: Tensor, ctx: HybridStackLayerContext) -> Tensor:
        """Prelude once, core per round (Huginn or MoR), coda once.

        A ring chunk holds one '|' segment, not the whole loop: each call runs its slice of one
        round (``_ring_layer_loop``), and only with the ring's call roles (``RING_EAGER_CALL``).
        """
        if (
            self.config.experimental_attention_variant == "dsa"
            and ctx.packed_seq_params is not None
            and not fixed_glm_loop(self.config)
        ):
            raise ValueError("The native DSA loop requires unpacked SBHD training inputs")
        if self.loop_ring is not None:
            return self._ring_layer_loop(hidden_states, ctx)
        validate_hybrid_loop_inputs(
            packed_seq_params=ctx.packed_seq_params,
            inference_context=ctx.inference_context,
            attention_mask=ctx.attention_mask,
            has_dsv4_attention=self._has_dsv4_attention,
            allow_packed_documents=fixed_glm_standard(self.config),
        )
        ctx.grouped_tail_to_skip = None
        depths, self._loop_depths = self._loop_depths, None
        if depths is not None and getattr(self.config, "cuda_graph_impl", None) == "full_iteration":
            raise ValueError("Full-iteration GLM loop capture requires fixed depth for all rows")
        if depths is not None and self.config.loop_architecture != "huginn":
            raise ValueError("Per-row loop depths need loop_architecture='huginn'")
        # Fresh dicts every forward: the training loss closure captures them, so a later
        # microbatch must not mutate an earlier one's.
        self.loop_losses = {}
        self.loop_metrics = {}
        self.loop_metric_populations = {}
        self._diagnostic_metrics = {}
        self._work = [0.0, 0.0, 0.0]
        label = f"loop_hybrid/mb={self._microbatch_id}"
        self._microbatch_id += 1
        roles = self.loop_roles
        sequence = hidden_states.shape[0]
        tokens = sequence * hidden_states.shape[1]
        squared_lengths = tokens * sequence
        with loop_range(label):
            self._record_range(
                0,
                roles.core_start,
                tokens,
                squared_lengths,
                sequence,
                recompute=self.config.loop_recompute_edges or _local_glm_full_recompute(self.config),
            )
            embedding = self._run_edge(hidden_states, 0, roles.core_start, ctx, label + "/prelude")
            if self._diagnostics_active():
                self._note_state_rms("loop state rms prelude", embedding, ctx.padding_mask)
            state = self._rounds_and_coda(
                embedding, ctx, label, depths, tokens, squared_lengths, sequence
            )
            for phase, value in zip(("forward", "backward", "recompute"), self._work):
                self.loop_metrics[f"loop {phase} flops per token"] = (
                    torch.as_tensor(value, dtype=torch.float32, device=hidden_states.device)
                    / tokens
                )
            # Carrier diagnostics (enable_carrier_diagnostics): merged, never rebound. Every
            # round 1..loop_iterations has its metric (the loss-dict reduction needs one key set
            # on every microbatch and rank); a round this forward did not run has population 0.
            if self._diagnostic_metrics:
                zero = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
                for step in range(self.config.loop_iterations):
                    self._diagnostic_metrics.setdefault(f"loop state rms r{step + 1}", (zero, zero))
            for name, (value, population) in self._diagnostic_metrics.items():
                self.loop_metrics[name] = value
                self.loop_metric_populations[name] = population
        return state

    def _rounds_and_coda(
        self,
        embedding: Tensor,
        ctx: HybridStackLayerContext,
        label: str,
        depths: Optional[Tensor],
        tokens: int,
        squared_lengths: int,
        sequence: int,
    ) -> Tensor:
        """The rounds (Huginn or MoR) and the coda after the prelude; returns the coda output.

        Overridden by ``LoopAttnResHybridStack`` (the attention-residual carrier).
        """
        roles = self.loop_roles
        if self.config.loop_architecture == "huginn":
            state = self._huginn(embedding, ctx, label, depths)
        else:
            state = self._mor(embedding, ctx, label)
        self._record_range(
            roles.core_end,
            len(self.layers),
            tokens,
            squared_lengths,
            sequence,
            recompute=self.config.loop_recompute_edges or _local_glm_full_recompute(self.config),
        )
        self._record_head(tokens, squared_lengths, sequence)
        return self._run_edge(state, roles.core_end, len(self.layers), ctx, label + "/coda")

    # ---------------------------------------------------------------------------------------
    # Ring chunk calls (loop_schedule='ring'; ring_pp.py)
    # ---------------------------------------------------------------------------------------

    def _ring_layer_loop(self, hidden_states: Tensor, ctx: HybridStackLayerContext) -> Tensor:
        """One ring call: this chunk's slice of round ``r`` of one microbatch (``LoopRingCall``).

        The ops are those of ``_huginn`` followed by the coda, cut at the chunk's ends, in the
        same order:

        * round 1 on the round-start chunk: the prelude on the (stream-expanded) embedding, once
          per microbatch (checkpointed according to the view and edge settings); its output
          ``e`` goes to the injection store; then the round body on ``(zeros, e)``;
        * a later round on the round-start chunk: the round body on ``(received state, e)``;
        * any other chunk: the round body on the received state;
        * round R on the round-end chunk: after the round body the coda (block-checkpointed with edge recompute),
          then the loop metrics (``_ring_head_metrics``).

        The round body is the Huginn round ``loop_norm(core(adapter([state, e])))`` cut to this
        chunk: the adapter on the round-start chunk, the core slice, the loop norm on the
        round-end chunk. It runs inside ``forward_statistics_round(r + 1, R)``, as one activation
        checkpoint over its tensors (state, injection), or as blocks when a stride is set,
        when the call's code is 'C'. Everything
        else it uses (round, label, core range, adapter / norm presence, the layer context) is
        bound when the call runs (``_ring_round_function``): its recompute runs inside the
        backward, after other calls (other microbatches and rounds) and after the call roles
        were cleared, and must repeat this call's forward. The routing tape of each checkpoint
        call replays its MoE choices.

        Records the FLOPs the call runs (``take_ring_work``) and the G1 counters
        (``take_ring_counts``).
        """
        call = self.call_roles
        if not isinstance(call, LoopRingCall):
            refuse_ring(RING_EAGER_CALL)
        chunk = self.loop_ring
        rounds = self.config.loop_iterations
        step = call.round
        embeds = chunk.round_start and step == 0
        heads = chunk.round_end and step == rounds - 1
        if (
            not 0 <= step < rounds
            or bool(call.pre_process) != embeds
            or bool(call.post_process) != heads
            or (call.mtp_process and not heads)
        ):
            refuse_ring(RING_CALL_ROLES)
        validate_hybrid_loop_inputs(
            packed_seq_params=ctx.packed_seq_params,
            inference_context=ctx.inference_context,
            attention_mask=ctx.attention_mask,
            has_dsv4_attention=self._has_dsv4_attention,
        )
        depths, self._loop_depths = self._loop_depths, None
        if depths is not None:
            refuse_ring(RING_ROW_DEPTHS)
        ctx.grouped_tail_to_skip = None
        # Fresh dicts every call: the training loss closure of the head call captures them.
        self.loop_losses = {}
        self.loop_metrics = {}
        self.loop_metric_populations = {}
        self._diagnostic_metrics = {}
        local = chunk.local
        label = f"loop_hybrid/mb={call.microbatch}"
        sequence, batch = hidden_states.shape[:2]
        tokens = sequence * batch
        squared_lengths = tokens * sequence
        scope = self._ring_sparse_pass(call)
        counts = self._ring_counts.setdefault(call.microbatch, [0, 0, 0, 0])
        with torch.profiler.record_function(f"{label}/ring_chunk={chunk.offset}"):
            if embeds:
                embedding = self._ring_prelude(
                    hidden_states, ctx, call, label, tokens, squared_lengths, sequence
                )
                counts[1] += 1
                state = torch.zeros_like(embedding)
                entry = RingInjection(
                    self._ring_injections, self._ring_injection_key(call), embedding
                )
                injection = ring_injection_source(embedding, entry)
                if not entry.grad_mode and rounds == 1:
                    entry.release()
            else:
                state = hidden_states
                injection = self._ring_injection(call) if chunk.round_start else None
            extra = self.round_extra_forward_flops(tokens) if chunk.round_start else 0
            self._ring_add_work(
                self.range_forward_flops(
                    local.core_start,
                    local.core_end,
                    tokens,
                    squared_lengths,
                    sequence,
                    extra=extra,
                    cross_loop_pass=scope,
                ),
                recompute=call.checkpoint,
            )
            round_label = f"{label}/loop={step + 1}"
            recompute = call.checkpoint and torch.is_grad_enabled()
            with forward_statistics_round(step + 1, rounds):
                if recompute and self.config.loop_recompute_block_stride is not None:
                    state = self._checkpointed_blocks(
                        state,
                        injection,
                        local.core_start,
                        local.core_end,
                        ctx,
                        round_label,
                        step,
                        round_start=chunk.round_start,
                        round_end=chunk.round_end,
                        cross_loop_pass=scope,
                    )
                else:
                    body = self._ring_round_function(
                        step,
                        ctx,
                        label,
                        adapt=chunk.round_start,
                        norm=chunk.round_end,
                        cross_loop_pass=scope,
                    )
                    if recompute:
                        state = loop_round_checkpoint(
                            body,
                            state,
                            injection,
                            label=round_label,
                            round_ends_in_layer=not chunk.round_end,
                        )
                    else:
                        state = body(state, injection)
            if scope is not None and scope.is_source:
                scope.finish_source()
            counts[0] += 1
            if heads:
                state = self._ring_coda(state, ctx, call, label, tokens, squared_lengths, sequence)
                counts[2] += 1
                self._ring_head_metrics(state.device, sequence, batch, depth=counts[0])
        return state

    def _ring_sparse_pass(self, call: LoopRingCall):
        if not self._cross_loop_attention_layers:
            return None
        banks, mb = call.sparse_banks, call.microbatch
        if banks is None:
            raise RuntimeError("Ring sparse reuse needs the physical chunk's step-local banks")
        if call.round == 0:
            if mb in banks:
                raise RuntimeError(f"Ring sparse bank already holds microbatch {mb}")
            banks[mb] = {}
        if mb not in banks:
            raise RuntimeError(
                f"Ring round {call.round + 1} has no first-round sparse bank for microbatch {mb}"
            )
        # After the last forward lookup, consumer checkpoint closures maintain the
        # only bank references they will need for their backwards.
        depth = self.config.loop_iterations if call.depth is None else call.depth
        bank = banks.pop(mb) if call.round + 1 == depth else banks[mb]
        return RingCrossLoopPass(call.round, bank)

    def _ring_injection_key(self, call: LoopRingCall) -> int:
        """The injection store's key of a call: its microbatch."""
        return call.microbatch

    def _ring_injection(self, call: LoopRingCall) -> Tensor:
        """The injection of a later round on the round-start chunk (``ring_pp``): through
        ``_RingInjectionUse`` in training; without autograd the round-R call releases the
        entry."""
        entry = self._ring_injections.get(self._ring_injection_key(call))
        if entry is None:
            raise RuntimeError(
                f"Ring round {call.round + 1} of microbatch {call.microbatch} has no injection: "
                "the microbatch's round-1 call did not run on this chunk"
            )
        injection = ring_injection_use(entry)
        if not entry.grad_mode and call.round == self.config.loop_iterations - 1:
            entry.release()
        return injection

    def _ring_prelude(self, hidden_states, ctx, call, label, tokens, squared_lengths, sequence):
        """Run the sole prelude of a microbatch under its round-start view's checkpoint policy."""
        end = self.loop_ring.local.core_start
        recompute = self._ring_edge_recomputed(call.checkpoint, prelude=True)
        self._ring_add_work(
            self.range_forward_flops(0, end, tokens, squared_lengths, sequence), recompute=recompute
        )
        if self.config.loop_recompute_block_stride is not None and call.checkpoint:
            return self._run_edge(hidden_states, 0, end, ctx, f"{label}/prelude")

        def prelude(x, ctx=ctx, end=end, label=label):
            return self._run_range(x, 0, end, ctx, f"{label}/prelude")

        if recompute and torch.is_grad_enabled():
            return loop_round_checkpoint(
                prelude, hidden_states, label=f"{label}/prelude", round_ends_in_layer=True
            )
        return prelude(hidden_states)

    def _ring_round_function(self, step, ctx, label, *, adapt, norm, cross_loop_pass=None):
        """The round body of one ring call as a function of its tensors ``(previous,
        injection)``: the adapter (``adapt``), the chunk's core slice, the loop norm (``norm``).

        Every other value is bound now as a default argument (the round ``step``, the layer
        context ``ctx``, the core range, ``adapt`` / ``norm``, the profiler ``label``), as
        ``_huginn`` binds its round: the function never reads the call roles, the injection
        store or the round view, so a recompute inside the backward repeats this call."""
        local = self.loop_ring.local

        def round_body(
            previous,
            injection,
            step=step,
            ctx=ctx,
            start=local.core_start,
            end=local.core_end,
            adapt=adapt,
            norm=norm,
            label=label,
        ):
            name = f"{label}/loop={step + 1}"
            with (
                cross_loop_pass.activate() if cross_loop_pass is not None else nullcontext(),
                torch.profiler.record_function(name),
            ):
                x = self._adapt(previous, injection, step) if adapt else previous
                x = self._run_range(x, start, end, ctx, name)
                return self._loop_norm(step)(x) if norm else x

        return round_body

    def _ring_edge_recomputed(self, checkpoint_view: bool, *, prelude: bool) -> bool:
        """Whole-view checkpointing includes the prelude. Block mode uses the Pro edge setting."""
        if self.config.loop_recompute_block_stride is None:
            return checkpoint_view and prelude
        return checkpoint_view and self.config.loop_recompute_edges

    def _ring_coda(self, state, ctx, call, label, tokens, squared_lengths, sequence):
        """Run the coda once, optionally checkpointing it in blocks on the final view."""
        local = self.loop_ring.local
        self._ring_add_work(
            self.range_forward_flops(
                local.core_end, len(self.layers), tokens, squared_lengths, sequence
            ),
            recompute=self._ring_edge_recomputed(call.checkpoint, prelude=False),
        )
        for work, backward_factor, replay in flops.head_schedule_work(
            self, tokens, squared_lengths, sequence
        ):
            self._ring_add_work(work, recompute=replay, backward_factor=backward_factor)
        if call.checkpoint:
            return self._run_edge(state, local.core_end, len(self.layers), ctx, f"{label}/coda")
        return self._run_range(state, local.core_end, len(self.layers), ctx, f"{label}/coda")

    def _ring_head_metrics(self, device, sequence, batch, *, depth):
        """The loop metrics of the head call, for the whole model (as ``_layer_loop`` reports
        them at PP = 1): ``loop depth`` = this chunk's core slice runs of the microbatch (R),
        the token-layer counts over the model's core, and the model's FLOPs per token
        (``set_ring_model_work``; backward and recompute 0 without autograd)."""
        work = self._ring_model_work.get((sequence, batch))
        if work is None:
            raise RuntimeError(
                f"The ring's model FLOPs for [sequence {sequence}, batch {batch}] are not set "
                "(LoopHybridStack.set_ring_model_work; the ring schedule sets them)"
            )
        forward, backward, recompute = work
        grad = torch.is_grad_enabled()
        if not grad:
            backward = recompute = 0.0
        tokens = sequence * batch
        rows = float(tokens * depth)
        core = self.loop_roles.core
        # Scalar fills execute on the capture stream. A tensor constructor
        # from a Python scalar can copy via pageable host memory during capture.
        self.loop_metrics = {
            "loop depth": torch.full((), float(depth), dtype=torch.float32, device=device),
            "loop active token layers": torch.full((), rows * core, dtype=torch.float32, device=device),
            "loop backward token layers": torch.full(
                (), rows * core if grad else 0.0, dtype=torch.float32, device=device
            ),
        }
        for phase, value in zip(
            ("forward", "backward", "recompute"), (forward, backward, recompute)
        ):
            self.loop_metrics[f"loop {phase} flops per token"] = torch.full(
                (), value / tokens, dtype=torch.float32, device=device
            )

    def _ring_add_work(self, work, *, recompute, backward_factor=2):
        """Add one pass to the ring's recorded FLOPs, as ``_add_work`` does: backward only with
        autograd on, recompute only for a checkpointed pass."""
        self._ring_work[0] = self._ring_work[0] + work
        if torch.is_grad_enabled():
            self._ring_work[1] = self._ring_work[1] + backward_factor * work
            if recompute:
                self._ring_work[2] = self._ring_work[2] + work

    def ring_microbatch_work(self, sequence: int, batch: int, codes) -> Tuple[float, float, float]:
        """The ``(forward, backward, recompute)`` FLOPs one training microbatch of
        ``[sequence, batch]`` costs this ring chunk, as its calls record them: the prelude once
        (round-start chunk), R round bodies (with the adapter on the round-start chunk), the coda
        and the head once (round-end chunk). ``codes``: the chunk's recompute code per round
        ('C' / 'n'; the first and last codes also select the corresponding edges). The model's
        FLOPs are the sum over
        every chunk of every stage (``set_ring_model_work``); a step of M microbatches records
        M times this (``take_ring_work``)."""
        chunk = self.loop_ring
        if chunk is None:
            raise ValueError("ring_microbatch_work needs a ring chunk (loop_schedule='ring')")
        codes = tuple(codes)
        if len(codes) != self.config.loop_iterations or not set(codes) <= set(RING_RECOMPUTE_CODES):
            raise ValueError(
                f"Ring chunk recompute codes must be {self.config.loop_iterations} of "
                f"{RING_RECOMPUTE_CODES}, got {codes}"
            )
        local = chunk.local
        tokens = sequence * batch
        squared_lengths = tokens * sequence
        work = [0.0, 0.0, 0.0]

        def add(value, recompute, backward_factor=2):
            work[0] = work[0] + value
            work[1] = work[1] + backward_factor * value
            if recompute:
                work[2] = work[2] + value

        if chunk.round_start:
            add(
                self.range_forward_flops(0, local.core_start, tokens, squared_lengths, sequence),
                self._ring_edge_recomputed(codes[0] == "C", prelude=True),
            )
        extra = self.round_extra_forward_flops(tokens) if chunk.round_start else 0
        for step, code in enumerate(codes):
            scope = RingCrossLoopPass(step, {}) if self._cross_loop_attention_layers else None
            add(
                self.range_forward_flops(
                    local.core_start,
                    local.core_end,
                    tokens,
                    squared_lengths,
                    sequence,
                    extra=extra,
                    cross_loop_pass=scope,
                ),
                code == "C",
            )
        if chunk.round_end:
            add(
                self.range_forward_flops(
                    local.core_end, len(self.layers), tokens, squared_lengths, sequence
                ),
                self._ring_edge_recomputed(codes[-1] == "C", prelude=False),
            )
            for value, backward_factor, replay in flops.head_schedule_work(
                self, tokens, squared_lengths, sequence
            ):
                add(value, replay, backward_factor)
        return tuple(work)

    def set_ring_model_work(self, sequence: int, batch: int, work) -> None:
        """Set the whole model's ``(forward, backward, recompute)`` FLOPs of one training
        microbatch of ``[sequence, batch]`` (the sum of ``ring_microbatch_work`` over every chunk
        of every stage), which this chunk's head call reports per token."""
        self._ring_model_work[(int(sequence), int(batch))] = tuple(float(w) for w in work)

    def take_ring_work(self) -> Tuple[float, float, float]:
        """The ``(forward, backward, recompute)`` FLOPs the ring calls recorded since the last
        call; resets them."""
        work, self._ring_work = tuple(self._ring_work), [0.0, 0.0, 0.0]
        return work

    def take_ring_counts(self) -> dict:
        """The G1 counters since the last call, per microbatch: ``[core slice runs, prelude
        runs, coda runs, MTP runs]`` of this chunk; resets them."""
        counts, self._ring_counts = self._ring_counts, {}
        return counts

    def ring_injection_keys(self) -> Tuple[int, ...]:
        """The microbatches the injection store holds (empty after a complete step)."""
        return tuple(sorted(self._ring_injections))

    def watch_ring_mtp(self, mtp: nn.Module) -> None:
        """Count the runs of ``mtp`` (the model's MTP block, on the round-end chunk) in the G1
        counters of the running call's microbatch (a forward hook, registered once)."""
        if self._ring_mtp_hook is None:
            self._ring_mtp_hook = mtp.register_forward_hook(self._count_ring_mtp_run)

    def _count_ring_mtp_run(self, *_):
        """The MTP block's forward hook (``watch_ring_mtp``): one run of the call's microbatch
        (an activation replay is not a run)."""
        call = self.call_roles
        if in_forward_replay():
            return
        if not isinstance(call, LoopRingCall):
            raise RuntimeError("The MTP block of a ring chunk ran outside a ring call")
        self._ring_counts.setdefault(call.microbatch, [0, 0, 0, 0])[3] += 1

    # ---------------------------------------------------------------------------------------
    # Carrier diagnostics
    # ---------------------------------------------------------------------------------------

    def enable_carrier_diagnostics(self, enabled: bool, sink=None, per_sublayer=False) -> None:
        """Turn the carrier diagnostics on or off (off by default).

        When on, every training forward (never an activation replay) adds to ``loop_metrics``
        (with its population in ``loop_metric_populations``, the loop reporting rule):
        ``loop state rms prelude`` (the prelude output) and ``loop state rms r{r}`` (the hidden
        state at the end of round ``r``: the injection carrier's round output before the loop
        norm, the residual carrier's stream; the RMS of the stream mean with mHC), each over the
        non-padding tokens of the rows alive in that round. ``LoopAttnResHybridStack`` adds the
        attention-residual weights and accepts ``sink`` (an ``attn_res.AttnResStatsSink`` that
        also receives every call's weights) and ``per_sublayer`` (accumulate the per-sublayer
        weight sums for ``take_carrier_diagnostics``); this class takes no sink and has no
        per-sublayer statistics, so ``per_sublayer`` changes nothing here. A ring chunk refuses
        them (``RING_CARRIER_DIAGNOSTICS``: a round's state is spread over the stages).
        """
        del per_sublayer  # No per-sublayer statistics on this carrier.
        if sink is not None:
            raise ValueError("Only the attention-residual carrier takes a diagnostics sink")
        if enabled and self.loop_ring is not None:
            refuse_ring(RING_CARRIER_DIAGNOSTICS)
        self._carrier_diagnostics = bool(enabled)

    def take_carrier_diagnostics(self) -> Optional[dict]:
        """The per-sublayer carrier statistics accumulated since the last call (None: this
        carrier has none; see ``LoopAttnResHybridStack``)."""
        return None

    def _diagnostics_active(self) -> bool:
        return self._carrier_diagnostics and self.training and not in_forward_replay()

    def _note_state_rms(self, name, x, padding_mask) -> None:
        """Record ``name`` = the RMS of the stream mean of ``x`` over the tokens that are not
        ``padding_mask`` (``[B, S]``, True = padding; None: every token)."""
        with torch.no_grad():
            valid = None if padding_mask is None else ~padding_mask.transpose(0, 1)
            self._diagnostic_metrics[name] = stream_mean_rms(x, self.loop_num_streams, valid)

    def _run_range(self, x, start, end, ctx, label):
        """Run sublayers ``[start, end)`` once, each through ``HybridStack._run_layer``.

        An empty input (a compact-MoR round no local token reached) skips every non-MoE
        sublayer and runs only the MoE MLP, so the EP dispatch / combine and its backward keep
        the collective order across ranks (the GPT loop's rule).

        The context's padding mask (row-tail padding, and the rows of exited samples in a
        per-row-depth round) also masks the SBHD indexer loss of DSv4 attentions
        (``indexer_loss_padding_mask``): padded and frozen rows do not train the indexer. The
        scope is entered here, inside every replay of the range, so activation recomputes
        see it too.

        With a loop parameter store each sublayer call runs inside the store's visit of its
        segment (``SegmentKey("sub", index)``): the sublayer's weights are bound there (an
        RSBF sub-round's schedule, or an eager no-grad visit).
        """
        with indexer_loss_padding_mask(ctx.padding_mask):
            for index in range(start, end):
                x = self._run_sublayer(index, x, ctx, label)
        return x

    def _run_sublayer(self, index, x, ctx, label):
        """One sublayer call of ``_run_range`` (the caller enters the indexer-loss scope).

        Inside the loop parameter store's visit of the sublayer and a ``{label}/layer={index}``
        profiler range. An empty input runs only a MoE sublayer's MLP (``_run_range``); with
        ``ctx.return_layer_delta`` the call returns the layer's delta (the MoE update itself on
        an empty input) instead of the residual sum.
        """
        hook = self._loop_store_hook
        layer = self.layers[index]
        visit = nullcontext() if hook is None else hook.visit("sub", index)
        with visit, loop_range(f"{label}/layer={index}"):
            if x.shape[0] == 0:
                residual_accumulator = x
                if ctx.return_layer_delta:
                    if getattr(layer, "is_moe_layer", False):
                        update, _ = layer.mlp(x)
                        return update
                    return torch.zeros_like(x)
                if getattr(layer, "is_moe_layer", False):
                    update, _ = layer.mlp(x)
                    x = x + update
                observe_layer_residuals(layer, residual_accumulator, x)
            else:
                # Observes the layer residuals itself, once per call (not for a delta).
                x = self._run_layer(index, x, ctx)
                # A grouped CUDA-graph replay covering the next sublayer would make
                # this loop skip it silently; eager only (graphs are guarded off).
                assert (
                    ctx.grouped_tail_to_skip is None
                ), f"sublayer {index} replayed a grouped CUDA graph inside the hybrid loop"
        return x

    def _run_edge(self, x, start, end, ctx, label, *, retain_sublayers=0):
        """The prelude or the coda (sublayers ``[start, end)``): block checkpoints with
        ``loop_recompute_edges`` in grad mode (``_checkpointed_blocks``), else ``_run_range``.
        A retained suffix executes with its ordinary autograd graph until this item's backward.
        Inside the profiler / NVTX range ``label`` (``loop_range``)."""
        with loop_range(label):
            full = _local_glm_full_recompute(self.config)
            checkpoint_edges = (
                self.config.loop_recompute or self.config.loop_schedule == "rsbf"
            ) and self.config.loop_recompute_edges
            if (full or checkpoint_edges) and torch.is_grad_enabled():
                split = end - retain_sublayers
                if start < split:
                    stride = None
                    if full:
                        lap = 0 if start == 0 else self.config.loop_iterations + 1
                        stride = _local_glm_visit_stride(self.config, lap)
                    x = self._checkpointed_blocks(
                        x, None, start, split, ctx, label,
                        block_stride=stride,
                    )
                if retain_sublayers:
                    x = self._run_range(x, split, end, ctx, label)
                return x
            return self._run_range(x, start, end, ctx, label)

    def _checkpointed_blocks(
        self,
        x,
        injection,
        start,
        end,
        ctx,
        label,
        step=None,
        *,
        round_start=True,
        round_end=True,
        cross_loop_pass=None,
        null_owner=False,
        block_stride=None,
    ):
        """Run sublayers ``[start, end)`` as non-reentrant activation checkpoints of
        ``loop_recompute_block_stride`` sublayers each (block and edge ``loop_recompute``).

        Block ``k`` (1-based, the last one possibly shorter) is one ``torch.utils.checkpoint``
        whose ``context_fn`` is ``loop_checkpoint_context_fn(f"{label}/block={k}")``: its own
        routing tape (the MoE choices of its forward, replayed by its recompute) and a replay
        marked as a forward replay (statistics count each forward once). A block keeps only
        its inputs for the backward. Every block runs its sublayers through ``_run_range`` with
        ``ctx`` captured by the closure, so each replay enters ``ctx``'s indexer-loss padding
        scope and gives hash-routed sublayers their token ids again. ``label`` is the range's
        ``_run_range`` label, so the sublayer profiler ranges keep their names.

        For a pipeline chunk only the round-start chunk applies the adapter and only the
        round-end chunk applies the norm. The other chunks retain layer-ending replay semantics.

        With ``step`` (a Huginn round, 0-based) the blocks form the round body, in the round
        checkpoint's op order: block 1 takes ``(x, injection)`` (the previous state and the
        prelude output) and runs the adapter first (the residual carrier has no adapter and
        takes ``x`` only); the last block records the round's carrier diagnostics (never in a
        replay, ``_diagnostics_active``) and then applies the loop norm (none on the residual
        carrier). One block (stride = the round's sublayers) is the round checkpoint under a
        block label. Without ``step`` (the prelude and coda edges) blocks run sublayers only.

        The checkpointed function opens the profiler / NVTX range ``{label}/block={k}``
        (``loop_range``), so the range appears in the forward and again in the block's
        recompute during the backward.

        A block that ends inside a layer (every block but the last of an injection round, which
        ends in the loop norm, and every prelude / coda block) is checkpointed with early stop
        off, as ``loop_round_checkpoint`` does for such rounds: its replay runs the block's last
        sublayer to its end instead of raising inside it, so no cut-off replay stays referenced
        by the layer's per-call state (``mlp_norm_manager``, the MoE dispatcher's routing state),
        every module of the block finishes its forward hooks in the replay (Megatron-FSDP's
        post-forward release, DDP's and GTP's gathers), and the routing tape's end-of-replay
        count check runs. Forward ops and gradients are unchanged.
        """
        stride = self.config.loop_recompute_block_stride if block_stride is None else block_stride
        cuts = list(range(start, end, stride))
        if not cuts:  # an empty prelude or coda
            return self._run_range(x, start, end, ctx, label)
        bounds = list(zip(cuts, cuts[1:] + [end]))
        injection_carrier = self.config.loop_carrier == "injection"
        gated = self.config.loop_carrier == "gated_residual"
        round_input = x
        for k, (first, last) in enumerate(bounds, start=1):
            adapter = round_start and step is not None and k == 1 and injection_carrier
            tail = round_end and step is not None and k == len(bounds)

            # Bind the block's bounds in the closure: its replay runs in the backward.
            def block(
                h,
                *extra,
                first=first,
                last=last,
                adapter=adapter,
                tail=tail,
                name=f"{label}/block={k}",
            ):
                with (
                    cross_loop_pass.activate() if cross_loop_pass is not None else nullcontext()
                ), loop_range(name):
                    if null_owner:
                        return self._run_null_experts(h, first, last)
                    if adapter:
                        h = self._adapt(h, extra[0], step)
                    h = self._run_range(h, first, last, ctx, label)
                    if tail:
                        if self._diagnostics_active():
                            self._note_state_rms(f"loop state rms r{step + 1}", h, ctx.padding_mask)
                        if gated:
                            h = self.loop_gates[step](extra[0], h)
                        elif injection_carrier:
                            h = self._loop_norm(step)(h)
                return h

            ends_in_layer = not (tail and (injection_carrier or gated))
            inputs = (x,)
            if adapter:
                inputs = (x, injection)
            elif tail and gated:
                inputs = (x, round_input)
            # torch reads the flag when the checkpoint frame is built (here), so the frame's
            # backward replay runs with it too.
            # A peer may replay an owner-only block while we replay a real one. Both
            # sides must reach the last MoK collective before either starts backward.
            online = getattr(self.config, "loop_exit_criterion", None) == "online"
            with set_checkpoint_early_stop(False) if ends_in_layer or online else nullcontext():
                x = checkpoint(
                    block,
                    *inputs,
                    use_reentrant=False,
                    context_fn=loop_checkpoint_context_fn(f"{label}/block={k}"),
                )
        return x

    def _layer_forward_flops(
        self, index, tokens, squared_lengths, sequence, arbitrary_mask, cross_loop_pass=None
    ):
        symbol, layer_config, compress_ratio = self._loop_flop_rows[index]
        if cross_loop_pass is not None and index in self._cross_loop_attention_layers:
            ratio, has_indexer = self._cross_loop_attention_layers[index]
            # A sequence too short for one group runs only the window on every pass.
            return flops.dsv4_attention_forward_flops(
                layer_config,
                ratio if sequence >= ratio else 0,
                tokens,
                squared_lengths,
                sequence,
                project_compressed_kv=cross_loop_pass.is_source
                or not self.config.loop_reuse_sparse_kv,
                has_indexer=has_indexer,
                run_indexer=cross_loop_pass.is_source,
            )
        return flops.sublayer_forward_flops(
            symbol,
            layer_config,
            tokens,
            squared_lengths,
            sequence,
            arbitrary_mask=arbitrary_mask,
            compress_ratio=compress_ratio,
        )

    def _add_work(self, work, *, backward=True, recompute=False, backward_factor=2):
        self._work[0] = self._work[0] + work
        if backward and torch.is_grad_enabled():
            self._work[1] = self._work[1] + backward_factor * work
            if recompute:
                self._work[2] = self._work[2] + work

    def _record_range(
        self,
        start,
        end,
        tokens,
        squared_lengths,
        sequence,
        *,
        backward=True,
        recompute=False,
        extra=0,
        arbitrary_mask=False,
        cross_loop_pass=None,
    ):
        """Record one pass over sublayers ``[start, end)`` (plus ``extra``) in the FLOP sums.

        Called outside activation checkpoints (a replay must not record again), so backward
        and recompute are attributed when the pass is issued, as on the GPT loop.
        """
        work = self.range_forward_flops(
            start,
            end,
            tokens,
            squared_lengths,
            sequence,
            arbitrary_mask=arbitrary_mask,
            extra=extra,
            cross_loop_pass=cross_loop_pass,
        )
        self._add_work(work, backward=backward, recompute=recompute)

    def _record_head(self, tokens, squared_lengths, sequence):
        """Record the output layer and the MTP depths (set by ``set_loop_head_flops``)."""
        for work, backward_factor, replay in flops.head_schedule_work(
            self, tokens, squared_lengths, sequence
        ):
            self._add_work(work, backward_factor=backward_factor, recompute=replay)

    def range_forward_flops(
        self,
        start,
        end,
        tokens,
        squared_lengths,
        sequence,
        *,
        arbitrary_mask=False,
        extra=0,
        cross_loop_pass=None,
    ):
        """Forward FLOPs of one pass over sublayers ``[start, end)`` (the per-symbol table),
        accumulated onto ``extra`` (e.g. the round's adapter work) in sublayer order."""
        work = extra
        for index in range(start, end):
            work = work + self._layer_forward_flops(
                index, tokens, squared_lengths, sequence, arbitrary_mask, cross_loop_pass
            )
        return work

    def round_extra_forward_flops(self, tokens):
        """Forward FLOPs of one Huginn round outside the core sublayers: the adapter on every
        mHC stream (none for the residual and attention-residual carriers; the aggregation is
        not a GEMM and is not counted)."""
        if self.config.loop_carrier == "gated_residual":
            return 2 * tokens * self.config.hidden_size
        if self.config.loop_carrier != "injection":
            return 0
        return flops.huginn_adapter_forward_flops(
            tokens, self.config.hidden_size, self.loop_num_streams
        )

    def _ring_global_sequence(self, local_sequence):
        """Issued document extent for fixed-shape work counters, including packed subgroups."""
        return getattr(self, "ring_issued_sequence", local_sequence * self.config.context_parallel_size)

    def head_work_terms(self, tokens, squared_lengths, sequence):
        """``(forward work, backward factor)`` of the output layer and the MTP depths, in the
        order the FLOP sums add them: the logits, the MTP depth sublayers, the MTP logits
        (backward factor 1 under ``mtp_detach_heads``: no wgrad, the training FLOP
        convention). Empty when ``set_loop_head_flops`` was not called."""
        if self._loop_head is None:
            return []
        vocab_size, mtp_symbols, mtp_depths = self._loop_head
        hidden = self.config.hidden_size
        logits = flops.logits_forward_flops(tokens, hidden, vocab_size)
        terms = [(logits, 2)]
        if not mtp_depths:
            return terms
        if getattr(self.config, "mtp_loss_type", None) == "e2e_tv":
            # The prediction objective projects the trunk hidden state a second
            # time under no_grad to obtain its teacher probability distribution.
            terms.append((logits, 0))
        depth_work = flops.mtp_depth_extra_forward_flops(tokens, hidden)
        for symbol in mtp_symbols:
            if symbol != Symbols.DS_ATTENTION:
                depth_work += flops.sublayer_forward_flops(
                    symbol, self.config, tokens, squared_lengths, sequence
                )
                continue
            shared = set(self.config.mtp_repeated_layer_shared_components or ())
            for depth in range(mtp_depths):
                attention, indexer = flops.native_dsa_forward_work(
                    self.config,
                    tokens,
                    self._ring_global_sequence(sequence),
                    project_kv=depth == 0 or "latent_kv" not in shared,
                    select=depth == 0 or "sparse_attention_index" not in shared,
                )
                terms.append((attention, 2))
                if indexer:
                    terms.append((indexer, 2 if self.config.dsa_indexer_loss_coeff else 0))
        terms.append((mtp_depths * depth_work, 2))
        terms.append((mtp_depths * logits, 1 if self.config.mtp_detach_heads else 2))
        return terms

    @property
    def _loop_round_group(self):
        """The group whose ranks must run the same number of eager rounds: TP x EP, or, when a
        loop parameter store keeps units in a Layer window, the RSBF domain (every eager visit
        of a windowed unit all-gathers it over the shard axis inside the domain)."""
        hook = self._loop_store_hook
        if hook is not None and hook.windows:
            return hook.groups.domain
        return getattr(self.pg_collection, "tp_ep", None)

    def _depth(self, device):
        if not self.config.loop_sample_depth or not self.training:
            return self.config.loop_iterations
        depth = torch.tensor([1, 2, 4, 8])[torch.randint(4, (1,))].to(device)
        # PP=CP=1. TP+EP ranks must take the same number of collective calls.
        # DP-only domains are allowed different depths; their backward remains
        # one reduction per unique parameter (a windowing parameter store: the domain).
        group = self._loop_round_group
        if group is not None and dist.get_world_size(group) > 1:
            dist.broadcast(depth, src=dist.get_global_rank(group, 0), group=group)
        return int(depth.item())

    def _row_rounds(self, depths, batch):
        """Validate per-row depths; return the round count and the minimum depth, both agreed
        across TP x EP (rounds ``1..low`` have every row of the group alive, so every rank
        runs them unmasked and the MoE aux-loss normalization stays consistent)."""
        if depths.dim() != 1 or depths.shape[0] != batch or depths.is_floating_point():
            raise ValueError(f"Loop depths must be an integer [B={batch}] tensor")
        if self.config.loop_bptt_iterations:
            raise ValueError("Per-row loop depths need full BPTT (loop_bptt_iterations=0)")
        bounds = torch.stack((depths.min(), depths.max())).to(torch.int64)
        # Validate before the collective (see set_loop_depths).
        low, high = bounds.tolist()
        if low < 1 or high > self.config.loop_iterations:
            raise ValueError(
                f"Loop depths must be in 1..{self.config.loop_iterations}, got {low}..{high}"
            )
        # (max, -min) in one MAX all-reduce.
        agreed = bounds.flip(0) * torch.tensor([1, -1], dtype=torch.int64, device=bounds.device)
        group = self._loop_round_group
        if group is not None and dist.get_world_size(group) > 1:
            # EP ranks must run the same number of rounds (MoE collectives; a windowing
            # parameter store: the gathers of the whole domain).
            dist.all_reduce(agreed, op=dist.ReduceOp.MAX, group=group)
        rounds, negative_low = agreed.tolist()
        return rounds, -negative_low

    def _round_bounds(self, step):
        """``(start, end)`` of the core sublayers round ``step`` (0-based) runs: the whole core,
        or segment ``step`` of an untied core."""
        start = self.loop_roles.core_start
        if not self.config.loop_untie_core:
            return start, self.loop_roles.core_end
        start = start + step * self._round_core
        return start, start + self._round_core

    def _loop_norm(self, step):
        """The loop norm of round ``step`` (its own norm on an untied core)."""
        return self.loop_norms[step] if self.config.loop_untie_core else self.loop_norm

    def _adapt(self, state, injection, step=0):
        """The ``[h, 2h]`` adapter on ``[state, injection]``, stream by stream (inside the loop
        parameter store's adapter visit when a store is installed): the shared adapter, or
        round ``step``'s own on an untied core."""
        hook = self._loop_store_hook
        adapter = self.loop_adapters[step] if self.config.loop_untie_core else self.loop_adapter
        with nullcontext() if hook is None else hook.visit("adapter"):
            if self.loop_num_streams == 1:
                return adapter(torch.cat((state, injection), -1))
            split = (*state.shape[:-1], self.loop_num_streams, self.config.hidden_size)
            adapted = adapter(torch.cat((state.reshape(split), injection.reshape(split)), -1))
            return adapted.reshape(state.shape)

    def _run_null_experts(self, hidden, start, end):
        """Match the live peers' expert calls without passing a stopped row through a layer."""
        for index in range(start, end):
            layer = self.layers[index]
            inner = getattr(layer, "inner_layer", layer)
            if getattr(inner, "is_moe_layer", False):
                hidden = inner.mlp.megakernel_experts.forward_null(hidden)
        return hidden

    def _online_next_round(self, local, device):
        """Agree on whether any EP peer still has a live sample."""
        local = torch.as_tensor(local, dtype=torch.int32, device=device).reshape(())
        any_running = local.clone()
        group = self.pg_collection.ep
        if group is not None and dist.get_world_size(group) > 1:
            dist.all_reduce(any_running, op=dist.ReduceOp.MAX, group=group)
        return torch.stack((local, any_running)).tolist()

    def take_online_exit_records(self):
        """Drain (realized depth, nonterminal scores) in local microbatch order."""
        records, self._online_exit_records = self._online_exit_records, []
        return records

    def _huginn(self, embedding, ctx, label, depths=None):
        sequence, batch = embedding.shape[:2]
        full = _local_glm_full_recompute(self.config)
        separate_gate = fixed_glm_standard(self.config) and (
            full or getattr(self.config, "loop_recompute_ring_gates", False)
        )
        online = getattr(self.config, "loop_exit_criterion", None) == "online"
        if online and (batch != 1 or depths is not None):
            raise ValueError("standard online exit requires one live sample and no preset depths")
        if depths is None:
            depth = self._depth(embedding.device)
            all_alive = depth
        else:
            depth, all_alive = self._row_rounds(depths, batch)
        differentiated = self.config.loop_bptt_iterations or depth
        first_grad = max(0, depth - differentiated)
        tokens = sequence * batch
        injection_carrier = self.config.loop_carrier == "injection"
        gated = self.config.loop_carrier == "gated_residual"
        state = torch.zeros_like(embedding) if injection_carrier else embedding
        reuse_attention = self.config.loop_reuse_sparse_kv or self.config.loop_reuse_indexer
        saved_attention = {} if reuse_attention else None
        running = True
        realized = 0
        scores = []
        for step in range(depth):
            core_start, core_end = self._round_bounds(step)
            if online and not running:
                # EP participation remains mandatory while another home runs its core.
                # The empty reduction keeps its backward in order without changing the
                # accepted state or passing a stopped sample into a model sublayer.
                round_label = f"{label}/owner-only={step + 1}"
                with loop_range(round_label):
                    if self.config.loop_recompute and torch.is_grad_enabled():
                        if self.config.loop_recompute_block_stride is not None:
                            dummy = self._checkpointed_blocks(
                                state, None, core_start, core_end, ctx, round_label, null_owner=True
                            )
                        else:
                            def null_round(h, start=core_start, end=core_end):
                                return self._run_null_experts(h, start, end)

                            dummy = loop_round_checkpoint(
                                null_round, state, label=round_label, round_ends_in_layer=True
                            )
                    else:
                        dummy = self._run_null_experts(state, core_start, core_end)
                state = state + dummy.reshape(-1)[:0].sum()
                if step + 1 < depth and not self._online_next_round(False, state.device)[1]:
                    break
                continue
            pass_scope = CrossLoopPass(step, saved_attention) if reuse_attention else None
            self._record_range(
                core_start,
                core_end,
                tokens,
                tokens * sequence,
                sequence,
                backward=step >= first_grad,
                recompute=self.config.loop_recompute or full,
                extra=self.round_extra_forward_flops(tokens),
                cross_loop_pass=pass_scope,
            )
            alive = None
            round_ctx = ctx
            # While every row (of the TP x EP group) is alive the round is the uniform-depth
            # round (no padding mask, no select): equal row depths equal uniform depth bitwise.
            if step + 1 > all_alive:
                alive = depths >= step + 1
                # Exited rows still run (eager oracle) but are padding for MoE routing
                # statistics; with alltoall they are still dispatched to experts.
                exited = (~alive)[:, None]
                round_ctx = dataclasses.replace(
                    ctx,
                    padding_mask=(
                        exited.repeat(1, sequence)
                        if ctx.padding_mask is None
                        else ctx.padding_mask | exited
                    ),
                )

            # Bind the step in the checkpoint closure: backward runs after this
            # Python loop, so capturing its mutable variable would mislabel replay.
            def recurrence(
                previous,
                injection,
                step=step,
                round_ctx=round_ctx,
                core_start=core_start,
                core_end=core_end,
                pass_scope=pass_scope,
            ):
                with (
                    pass_scope.activate() if pass_scope is not None else nullcontext(),
                    loop_range(f"{label}/loop={step + 1}"),
                ):
                    adapted = (
                        self._adapt(previous, injection, step) if injection_carrier else previous
                    )
                    output = self._run_range(
                        adapted, core_start, core_end, round_ctx, f"{label}/loop={step + 1}"
                    )
                    if self._diagnostics_active():
                        self._note_state_rms(
                            f"loop state rms r{step + 1}", output, round_ctx.padding_mask
                        )
                    if gated and not separate_gate:
                        return self.loop_gates[step](previous, output)
                    return self._loop_norm(step)(output) if injection_carrier else output

            context = torch.no_grad() if step < first_grad else nullcontext()
            # Statistics recorded in this round's forward are also binned as round step + 1
            # (an activation replay records nothing, forward_replay). The bins cover rounds
            # 1..loop_iterations (loop_sample_depth may draw deeper rounds: not binned).
            rounds = self.config.loop_iterations
            binned = forward_statistics_round(step + 1, rounds) if step < rounds else nullcontext()
            with context, binned:
                if (
                    full or (self.config.loop_recompute and self.config.loop_recompute_block_stride)
                ) and torch.is_grad_enabled():
                    # Block checkpoints: the round body cut into blocks of the stride.
                    round_label = f"{label}/loop={step + 1}"
                    with loop_range(round_label):
                        updated = self._checkpointed_blocks(
                            state,
                            embedding,
                            core_start,
                            core_end,
                            round_ctx,
                            round_label,
                            step,
                            cross_loop_pass=pass_scope,
                            round_end=not separate_gate,
                            block_stride=(
                                _local_glm_visit_stride(self.config, step + 1) if full else None
                            ),
                        )
                elif self.config.loop_recompute and torch.is_grad_enabled():
                    # A residual round returns its last sublayer's output; an injection round
                    # ends in the loop norm, after its last layer.
                    updated = loop_round_checkpoint(
                        recurrence,
                        state,
                        embedding,
                        label=f"{label}/loop={step + 1}",
                        round_ends_in_layer=online or separate_gate or not (injection_carrier or gated),
                    )
                else:
                    updated = recurrence(state, embedding)
                if separate_gate:
                    gate = self.loop_gates[step]
                    if self.config.loop_recompute_ring_gates and torch.is_grad_enabled():
                        updated = checkpoint(
                            gate, state, updated, use_reentrant=False, preserve_rng_state=False
                        )
                    else:
                        updated = gate(state, updated)
            if pass_scope is not None and pass_scope.is_source:
                pass_scope.finish_source()
            # Row exit: an exited row keeps its state on every stream (self-carry).
            state = updated if alive is None else torch.where(alive[None, :, None], updated, state)
            if online:
                realized = step + 1
                if realized < depth:
                    stop, score = online_exit_decision(
                        self.config, state, realized,
                        valid=None if ctx.padding_mask is None else ~ctx.padding_mask,
                        cp_group=self.pg_collection.cp, return_score=True,
                    )
                    scores.append(score[0])
                    # Copy the home flag and the EP maximum together after the collective.
                    flags = self._online_next_round(~stop, state.device)
                    running, any_running = flags
                    if not any_running:
                        break
        if online:
            if self.training and torch.is_grad_enabled():
                self._online_exit_records.append((realized, scores))
            depth = realized
        device = embedding.device
        core = self._round_core
        if depths is None:
            row_rounds = float(tokens * depth)
            backward_rounds = float(tokens * min(depth, differentiated))
            mean_depth = torch.tensor(float(depth), device=device)
        else:
            row_rounds = depths.sum().float() * sequence
            backward_rounds = row_rounds
            mean_depth = depths.float().mean()
        self.loop_metrics = {
            "loop depth": mean_depth,
            "loop active token layers": torch.as_tensor(row_rounds * core, device=device),
            "loop backward token layers": torch.as_tensor(
                backward_rounds * core if torch.is_grad_enabled() else 0.0, device=device
            ),
        }
        return state

    def _mor(self, embedding, ctx, label):
        valid = torch.ones(embedding.shape[:2], dtype=torch.bool, device=embedding.device)
        if ctx.padding_mask is not None:
            valid &= ~ctx.padding_mask.transpose(0, 1)
        with torch.profiler.record_function(f"{label}/depth_router"):
            logits = self.depth_router(embedding)
            depth, selected, balance, z_loss, frequency = token_depth_distribution(
                logits,
                valid,
                self.config.loop_router_balance_coeff,
                self.config.loop_router_z_loss_coeff,
            )
        self.loop_losses = {"depth balance loss": balance, "depth z loss": z_loss}
        sequence = embedding.shape[0]
        tokens = sequence * embedding.shape[1]
        self._record_range(
            0,
            0,
            tokens,
            0,
            sequence,
            extra=flops.mor_router_forward_flops(
                tokens, self.config.hidden_size, self.config.loop_iterations
            ),
        )
        core_start, core_end = self.loop_roles.core_start, self.loop_roles.core_end
        state = embedding
        active_counts = []
        for step in range(1, self.config.loop_iterations + 1):
            active = (depth >= step) & valid
            active_counts.append(active.sum().detach())
            loop_label = f"{label}/loop={step}"
            if self.config.loop_token_execution == "masked":
                loop_ctx = dataclasses.replace(
                    ctx,
                    attention_mask=active_causal_mask(active),
                    padding_mask=~active.transpose(0, 1),
                )
                indices = None
                layer_input = state
            else:
                with torch.profiler.record_function(f"{loop_label}/compact"):
                    indices, packed = compact_metadata(active)
                    flat = state.transpose(0, 1).reshape(-1, state.shape[-1])
                    layer_input = flat.index_select(0, indices).unsqueeze(1)
                loop_ctx = dataclasses.replace(
                    ctx,
                    attention_mask=None,
                    padding_mask=None,
                    packed_seq_params=packed,
                    input_ids=None,
                )

            issued_tokens = tokens if indices is None else indices.numel()
            squared_lengths = (
                tokens * sequence
                if indices is None
                else active.sum(0).float().square().sum().detach()
            )
            self._record_range(
                core_start,
                core_end,
                issued_tokens,
                squared_lengths,
                sequence,
                recompute=self.config.loop_recompute,
                arbitrary_mask=indices is None,
            )

            def recurrence(x, loop_ctx=loop_ctx, loop_label=loop_label):
                with torch.profiler.record_function(loop_label):
                    return self._run_range(x, core_start, core_end, loop_ctx, loop_label)

            if self.config.loop_recompute and torch.is_grad_enabled():
                updated = loop_round_checkpoint(
                    recurrence, layer_input, label=loop_label, round_ends_in_layer=True
                )
            else:
                updated = recurrence(layer_input)
            if indices is None:
                state = torch.where(active[..., None], updated, state)
            else:
                with torch.profiler.record_function(f"{loop_label}/scatter"):
                    # index_copy is differentiable even for empty updates: this
                    # retains the empty rank's EP backward in the autograd graph.
                    flat = flat.index_copy(0, indices, updated.squeeze(1))
                    state = flat.view(embedding.shape[1], embedding.shape[0], -1).transpose(0, 1)
        counts = torch.stack(active_counts).float()
        core = self.loop_roles.core
        self.loop_metrics = {
            "loop depth": counts.sum() / valid.sum().clamp_min(1),
            "loop active token layers": counts.sum() * core,
            "loop backward token layers": counts.sum() * core * int(torch.is_grad_enabled()),
        }
        for index, fraction in enumerate(frequency.detach()):
            self.loop_metrics[f"depth fraction {index + 1}"] = fraction
        # The MoR metrics are means over the valid tokens (loop/reporting.py).
        valid_tokens = valid.sum()
        self.loop_metric_populations = {
            name: valid_tokens
            for name in self.loop_metrics
            if name == "loop depth" or name.startswith("depth fraction ")
        }
        # Gate gradients reduce over hidden channels and can nearly cancel
        # after the final RMSNorm. Accumulate the weighted residual in FP32,
        # rather than rounding the gate and its reduction to BF16.
        return (embedding.float() + selected[..., None] * state.float()).to(state.dtype)
