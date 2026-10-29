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
  has exited keeps its state (self-carry) while the others continue.
* MoR (token-choice mixture of recursions, dense cores only): a router picks a depth per token;
  round ``r`` runs the core on the tokens with depth ``>= r`` (masked or compacted).

Eager only: no CUDA graphs, no pipeline / context / sequence parallelism (see the guards).

Checkpoint keys (M3c). At PP = 1 (guarded) a sharded key equals the module path. The loop adds
no key code of its own: ``HybridStack.sharded_state_dict`` writes sublayer ``i`` as
``decoder.layers.i`` (mHC-wrapped sublayers nest their layer under ``inner_layer.`` next to
``hyper_connection.*``; attention and MoE are separate sublayer indices; prelude / core / coda
are roles in the config, not part of any key) and gives every other child and the stack's own
tensors a replicated entry through the default walk:

* ``decoder.loop_adapter.weight``, ``[h, 2h]``: the Huginn adapter (D34: shared by the mHC
  streams);
* ``decoder.loop_norm.weight``, ``[h]`` (``LoopRMSNorm``, one stream) or ``[n, h]``
  (``LoopStreamRMSNorm``, D34's per-stream norm);
* ``decoder.depth_router.0.weight`` ``[h, h]`` and ``decoder.depth_router.2.weight``
  ``[R_max, h]``: the MoR router;
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
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackLayerContext
from megatron.core.models.hybrid.layers.utils import Symbols
from megatron.core.tensor_observation import observe_layer_residuals
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
    indexer_loss_padding_mask,
)
from megatron.core.transformer.forward_replay import forward_statistics_round
from megatron.core.transformer.loop import flops
from megatron.core.transformer.loop.guards import (
    LoopRoles,
    has_dsv4_attention,
    resolve_hybrid_loop_roles,
    validate_hybrid_loop_config,
    validate_hybrid_loop_inputs,
    validate_loop_mtp_pattern,
    validate_loop_stack_construction,
)
from megatron.core.transformer.loop_transformer_block import (
    LoopRMSNorm,
    active_causal_mask,
    compact_metadata,
    recompute_context,
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
    ):
        validate_loop_stack_construction(config, is_mtp_layer)
        validate_hybrid_loop_config(config)
        super().__init__(
            config=config,
            submodules=submodules,
            pre_process=pre_process,
            layer_type_list=layer_type_list,
            pp_layer_offset=pp_layer_offset,
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
        # PP = 1 (guarded): the stack holds the whole main pattern, layer numbers are i + 1, and
        # without Shortcut MoE (guarded) self.layers is indexed by the physical sublayer.
        assert self._execution_layer_indices == list(range(len(self.layers)))
        self.loop_roles: LoopRoles = resolve_hybrid_loop_roles(
            config, self.layer_type_list, hash_moe_layer_threshold
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
        loop_dtype = config.params_dtype
        hidden = config.hidden_size
        if config.loop_architecture == "huginn":
            self.loop_adapter = nn.Linear(
                2 * hidden, hidden, bias=False, dtype=loop_dtype, device=loop_device
            )
            config.init_method(self.loop_adapter.weight)
            if self.loop_num_streams == 1:
                self.loop_norm = LoopRMSNorm(
                    hidden, config.layernorm_epsilon, loop_dtype, loop_device
                )
            else:
                self.loop_norm = LoopStreamRMSNorm(
                    hidden, self.loop_num_streams, config.layernorm_epsilon, loop_dtype, loop_device
                )
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
        self.loop_losses = {}
        self.loop_metrics = {}
        self.loop_metric_populations = {}
        self._microbatch_id = 0
        self._loop_depths: Optional[Tensor] = None
        self._loop_head = None
        self._work = [0.0, 0.0, 0.0]

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
        """
        self._loop_depths = depths

    def set_loop_head_flops(
        self, vocab_size: int, mtp_pattern: Optional[str], mtp_num_depths: int
    ) -> None:
        """Include the output layer and the MTP depths in the reported FLOPs.

        ``HybridModel`` calls this after building the decoder; a stack used on its own reports
        its layers only.
        """
        validate_loop_mtp_pattern(mtp_pattern if mtp_num_depths else None)
        self._loop_head = (
            vocab_size,
            tuple(mtp_pattern or "") if mtp_num_depths else (),
            mtp_num_depths,
        )

    def _layer_loop(self, hidden_states: Tensor, ctx: HybridStackLayerContext) -> Tensor:
        """Prelude once, core per round (Huginn or MoR), coda once."""
        validate_hybrid_loop_inputs(
            packed_seq_params=ctx.packed_seq_params,
            inference_context=ctx.inference_context,
            attention_mask=ctx.attention_mask,
            has_dsv4_attention=self._has_dsv4_attention,
        )
        ctx.grouped_tail_to_skip = None
        depths, self._loop_depths = self._loop_depths, None
        if depths is not None and self.config.loop_architecture != "huginn":
            raise ValueError("Per-row loop depths need loop_architecture='huginn'")
        # Fresh dicts every forward: the training loss closure captures them, so a later
        # microbatch must not mutate an earlier one's.
        self.loop_losses = {}
        self.loop_metrics = {}
        self.loop_metric_populations = {}
        self._work = [0.0, 0.0, 0.0]
        label = f"loop_hybrid/mb={self._microbatch_id}"
        self._microbatch_id += 1
        roles = self.loop_roles
        sequence = hidden_states.shape[0]
        tokens = sequence * hidden_states.shape[1]
        squared_lengths = tokens * sequence
        with torch.profiler.record_function(label):
            self._record_range(0, roles.core_start, tokens, squared_lengths, sequence)
            embedding = self._run_range(hidden_states, 0, roles.core_start, ctx, label + "/prelude")
            if self.config.loop_architecture == "huginn":
                state = self._huginn(embedding, ctx, label, depths)
            else:
                state = self._mor(embedding, ctx, label)
            self._record_range(roles.core_end, len(self.layers), tokens, squared_lengths, sequence)
            self._record_head(tokens, squared_lengths, sequence)
            state = self._run_range(state, roles.core_end, len(self.layers), ctx, label + "/coda")
            for phase, value in zip(("forward", "backward", "recompute"), self._work):
                self.loop_metrics[f"loop {phase} flops per token"] = (
                    torch.as_tensor(value, dtype=torch.float32, device=hidden_states.device)
                    / tokens
                )
        return state

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
        hook = self._loop_store_hook
        with indexer_loss_padding_mask(ctx.padding_mask):
            for index in range(start, end):
                layer = self.layers[index]
                visit = nullcontext() if hook is None else hook.visit("sub", index)
                with visit, torch.profiler.record_function(f"{label}/layer={index}"):
                    if x.shape[0] == 0:
                        residual_accumulator = x
                        if getattr(layer, "is_moe_layer", False):
                            update, _ = layer.mlp(x)
                            x = x + update
                        observe_layer_residuals(layer, residual_accumulator, x)
                    else:
                        # Observes the layer residuals itself, once per call.
                        x = self._run_layer(index, x, ctx)
                        # A grouped CUDA-graph replay covering the next sublayer would make
                        # this loop skip it silently; eager only (graphs are guarded off).
                        assert ctx.grouped_tail_to_skip is None, (
                            f"sublayer {index} replayed a grouped CUDA graph inside the hybrid "
                            "loop"
                        )
        return x

    def _layer_forward_flops(self, index, tokens, squared_lengths, sequence, arbitrary_mask):
        symbol, layer_config, compress_ratio = self._loop_flop_rows[index]
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
        )
        self._add_work(work, backward=backward, recompute=recompute)

    def _record_head(self, tokens, squared_lengths, sequence):
        """Record the output layer and the MTP depths (set by ``set_loop_head_flops``)."""
        for work, backward_factor in self.head_work_terms(tokens, squared_lengths, sequence):
            self._add_work(work, backward_factor=backward_factor)

    def range_forward_flops(
        self, start, end, tokens, squared_lengths, sequence, *, arbitrary_mask=False, extra=0
    ):
        """Forward FLOPs of one pass over sublayers ``[start, end)`` (the per-symbol table),
        accumulated onto ``extra`` (e.g. the round's adapter work) in sublayer order."""
        work = extra
        for index in range(start, end):
            work = work + self._layer_forward_flops(
                index, tokens, squared_lengths, sequence, arbitrary_mask
            )
        return work

    def round_extra_forward_flops(self, tokens):
        """Forward FLOPs of one Huginn round outside the core sublayers: the adapter on every
        mHC stream."""
        return flops.huginn_adapter_forward_flops(
            tokens, self.config.hidden_size, self.loop_num_streams
        )

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
        depth_work = flops.mtp_depth_extra_forward_flops(tokens, hidden)
        for symbol in mtp_symbols:
            depth_work = depth_work + flops.sublayer_forward_flops(
                symbol, self.config, tokens, squared_lengths, sequence
            )
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

    def _adapt(self, state, injection):
        """The shared ``[h, 2h]`` adapter on ``[state, injection]``, stream by stream (inside
        the loop parameter store's adapter visit when a store is installed)."""
        hook = self._loop_store_hook
        with nullcontext() if hook is None else hook.visit("adapter"):
            if self.loop_num_streams == 1:
                return self.loop_adapter(torch.cat((state, injection), -1))
            split = (*state.shape[:-1], self.loop_num_streams, self.config.hidden_size)
            adapted = self.loop_adapter(
                torch.cat((state.reshape(split), injection.reshape(split)), -1)
            )
            return adapted.reshape(state.shape)

    def _huginn(self, embedding, ctx, label, depths=None):
        sequence, batch = embedding.shape[:2]
        if depths is None:
            depth = self._depth(embedding.device)
            all_alive = depth
        else:
            depth, all_alive = self._row_rounds(depths, batch)
        differentiated = self.config.loop_bptt_iterations or depth
        first_grad = max(0, depth - differentiated)
        core_start, core_end = self.loop_roles.core_start, self.loop_roles.core_end
        tokens = sequence * batch
        state = torch.zeros_like(embedding)
        for step in range(depth):
            self._record_range(
                core_start,
                core_end,
                tokens,
                tokens * sequence,
                sequence,
                backward=step >= first_grad,
                recompute=self.config.loop_recompute,
                extra=self.round_extra_forward_flops(tokens),
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
            def recurrence(previous, injection, step=step, round_ctx=round_ctx):
                with torch.profiler.record_function(f"{label}/loop={step + 1}"):
                    adapted = self._adapt(previous, injection)
                    output = self._run_range(
                        adapted, core_start, core_end, round_ctx, f"{label}/loop={step + 1}"
                    )
                    return self.loop_norm(output)

            context = torch.no_grad() if step < first_grad else nullcontext()
            # Statistics recorded in this round's forward are also binned as round step + 1
            # (an activation replay records nothing, forward_replay). The bins cover rounds
            # 1..loop_iterations (loop_sample_depth may draw deeper rounds: not binned).
            rounds = self.config.loop_iterations
            binned = forward_statistics_round(step + 1, rounds) if step < rounds else nullcontext()
            with context, binned:
                if self.config.loop_recompute and torch.is_grad_enabled():
                    updated = checkpoint(
                        recurrence,
                        state,
                        embedding,
                        use_reentrant=False,
                        context_fn=lambda step=step: (
                            torch.profiler.record_function(
                                f"{label}/loop={step + 1}/phase=forward"
                            ),
                            recompute_context(f"{label}/loop={step + 1}/phase=recompute"),
                        ),
                    )
                else:
                    updated = recurrence(state, embedding)
            # Row exit: an exited row keeps its state on every stream (self-carry).
            state = updated if alive is None else torch.where(alive[None, :, None], updated, state)
        device = embedding.device
        core = self.loop_roles.core
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
                updated = checkpoint(
                    recurrence,
                    layer_input,
                    use_reentrant=False,
                    context_fn=lambda loop_label=loop_label: (
                        torch.profiler.record_function(f"{loop_label}/phase=forward"),
                        recompute_context(f"{loop_label}/phase=recompute"),
                    ),
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
