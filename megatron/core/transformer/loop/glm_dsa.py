# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GLM-style two-pass text decoder using native token-level sparse MLA."""

import copy
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import replace
from types import SimpleNamespace

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.layers.utils import Symbols
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAttention,
    dsa_nonreentrant_checkpoint_scope,
)
from megatron.core.transformer.forward_replay import in_forward_replay, routing_tape_active
from megatron.core.transformer.loop.flops import native_dsa_forward_work
from megatron.core.transformer.loop.glm_ring_stack import GLMRingStack

_INDEX_SHARE = ContextVar("glm_loop_dsa_index_share", default=None)


class GLMLoopDSAttention(DSAttention):
    """Number decoder blocks (D + FFN) and start each section with its own selector."""

    def __init__(self, config, submodules, layer_number, **kwargs):
        if layer_number % 2 != 1:
            raise ValueError("Expected native DSA in the first sublayer of each decoder block")
        ordinal = (layer_number + 1) // 2
        first_core = 1 + config.loop_prelude_sublayers // 2
        first_coda = 1 + (config.num_layers - config.loop_coda_sublayers) // 2
        config = copy.copy(config)
        if ordinal >= first_coda:
            config.dsa_indexer_skip_topk_offset = first_coda
        elif ordinal >= first_core:
            config.dsa_indexer_skip_topk_offset = first_core
        super().__init__(config, submodules, layer_number=ordinal, **kwargs)

    def _get_index_share_carrier(self, packed_seq_params, attention_mask):
        carrier = _INDEX_SHARE.get()
        if carrier is None:
            raise RuntimeError("A GLM loop DSA layer needs its microbatch's index-sharing scope")
        return carrier

    def _get_nonpacked_cp_layout_cache(self):
        carrier = _INDEX_SHARE.get()
        return None if carrier is None else carrier.cp_layout


class GLMDSALoopStack(GLMRingStack):
    """Use one cross-layer index holder per microbatch and pass, including block replay."""

    def __init__(self, config, **kwargs):
        if config.experimental_attention_variant != "dsa" or not config.multi_latent_attention:
            raise ValueError("GLM loop requires native DSA over absorbed MLA")
        splits = config.loop_prelude_sublayers, config.loop_coda_sublayers
        if any(n is None or n % 2 for n in splits):
            raise ValueError("GLM loop splits must be specified in complete DSA/FFN pairs")
        super().__init__(config, **kwargs)
        pattern = self.layer_type_list
        offset = self.loop_ring.offset if self.loop_ring is not None else 0
        for i, symbol in enumerate(pattern, offset):
            allowed = (Symbols.DS_ATTENTION,) if i % 2 == 0 else (Symbols.MOE, Symbols.MLP)
            if symbol not in allowed:
                raise ValueError(
                    "GLM loop requires alternating DSA and dense or expert FFN sublayers"
                )

    def _run_range(self, x, start, end, ctx, label):
        # The block checkpoint closures hold ctx and the original pass label. A
        # different microbatch or a second core visit must never replace their indices.
        if ctx.forward_adapter_context is None:
            ctx.forward_adapter_context = {}
        holders = ctx.forward_adapter_context
        if label not in holders:
            # All layers and replays see the same fixed SBHD shard geometry. Sparse
            # selections still have a separate holder per pass and sample.
            holders[label] = SimpleNamespace(cp_layout=holders.setdefault("sbhd_cp_layout", {}))
        carrier = holders[label]
        checkpoint = nullcontext()
        if self.config.dsa_indexer_reuse_checkpoint_topk and routing_tape_active():
            # A pass may have several disjoint block checkpoints, replayed in reverse order.
            # The holder is specific to this physical stack, sample and logical visit.
            owners = vars(carrier).setdefault("checkpoint_owners", {})
            key = (start, end)
            if not in_forward_replay():
                owners[key] = object()
            checkpoint = dsa_nonreentrant_checkpoint_scope(owners[key], replay=in_forward_replay())
        token = _INDEX_SHARE.set(carrier)
        try:
            with checkpoint:
                return super()._run_range(x, start, end, ctx, label)
        finally:
            _INDEX_SHARE.reset(token)

    def _validate_cross_loop_attention_targets(self):
        roles = self.loop_ring.local if self.loop_ring is not None else self.loop_roles
        return {
            index: (1, self.layers[index].self_attention.core_attention.indexer is not None)
            for index in range(roles.core_start, roles.core_end)
            if self.layer_type_list[index] == Symbols.DS_ATTENTION
        }

    def _compress_ratio(self, symbol, index):
        # All native attention layers use one KV entry per token.
        return None

    def _ring_lap_work(self, lap, checkpoint, sequence, batch):
        """Split the selector from the ordinary F/B and the small gate/head terms."""
        terms = super()._ring_lap_work(lap, checkpoint, sequence, batch)
        body, _, replay = next(terms)
        tokens = sequence * batch
        total_select = 0
        selector_terms = []
        reuse_first = 1 < lap <= self.config.loop_iterations and self._cross_loop_attention_layers
        if not reuse_first:
            for i, (symbol, config, _) in enumerate(self._loop_flop_rows):
                if (
                    symbol != Symbols.DS_ATTENTION
                    or self.layers[i].self_attention.core_attention.indexer is None
                ):
                    continue
                _, select = native_dsa_forward_work(
                    config, tokens, self._ring_global_sequence(sequence)
                )
                total_select += select
                train = bool(config.dsa_indexer_loss_coeff)
                compute_replay = replay and not config.dsa_indexer_reuse_checkpoint_topk
                if replay and config.dsa_indexer_reuse_checkpoint_topk and train:
                    # The KL still projects the BF16 Q/K/weights in replay, but reads the
                    # original dense selection and sparse-loss log probabilities.
                    dim, heads = config.dsa_indexer_head_dim, config.dsa_indexer_n_heads
                    projection = (
                        2
                        * tokens
                        * (config.q_lora_rank * heads * dim + config.hidden_size * (dim + heads))
                    )
                    selector_terms.append((projection, 2, True))
                    select -= projection
                selector_terms.append((select, 2 if train else 0, compute_replay))
        yield body - total_select, 2, replay
        yield from selector_terms
        yield from terms

    def _layer_forward_flops(
        self, index, tokens, squared_lengths, sequence, arbitrary_mask, cross_loop_pass=None
    ):
        symbol, config, _ = self._loop_flop_rows[index]
        if symbol != Symbols.DS_ATTENTION:
            return super()._layer_forward_flops(
                index, tokens, squared_lengths, sequence, arbitrary_mask
            )
        recurrent_read = cross_loop_pass is not None and not cross_loop_pass.is_source
        has_indexer = self.layers[index].self_attention.core_attention.indexer is not None
        attention, indexer = native_dsa_forward_work(
            config,
            tokens,
            self._ring_global_sequence(sequence),
            select=has_indexer and not recurrent_read,
            window=config.loop_sparse_kv_window_size if recurrent_read else 0,
        )
        return attention + indexer


def loop_hybrid_glm_dsa_stack_spec(config):
    """Keep the ordinary absorbed-MLA/TE training spec and specialize index selection."""
    submodules = hybrid_stack_spec.submodules
    layer = submodules.dsa_layer
    attention = layer.submodules.self_attention
    core = replace(attention.submodules.core_attention, module=GLMLoopDSAttention)
    attention = replace(attention, submodules=replace(attention.submodules, core_attention=core))
    layer = replace(layer, submodules=replace(layer.submodules, self_attention=attention))
    return replace(
        hybrid_stack_spec, module=GLMDSALoopStack, submodules=replace(submodules, dsa_layer=layer)
    )
