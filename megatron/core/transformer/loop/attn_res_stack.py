# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Kimi-K3-style attention-residual carrier of the hybrid loop (``loop_carrier='attn_res'``).

``LoopAttnResHybridStack`` is a ``LoopHybridStack`` (eager Huginn only) whose state across
sublayers and rounds is an attention-residual history instead of the Huginn adapter + norm:

* the prelude runs with a plain residual; its output ``e`` is source 0 of the history;
* round ``r`` (``1..R``) splits the core into ``k = loop_attn_res_blocks_per_round`` equal
  blocks (cuts on attention + MLP block boundaries). Every core sublayer reads the softmax mix
  (``attn_res.LoopAttentionResidual``) of the history plus the current block's partial sum
  (none at a block start), runs as a layer delta (``HybridStackLayerContext.return_layer_delta``:
  the branch output without the residual add) and adds the delta to the partial. At every
  block end the partial is appended to the history (a block sum ``b``) and reset;
* the coda is the last block: each coda sublayer reads the whole history plus the coda's
  partial, and the output head (``loop_attn_res_head``) aggregates ``history + [coda partial]``
  before the stack's final norm.

One ``LoopAttentionResidual`` (a pseudo-query ``w`` and a key-norm weight, fp32) per physical
core and coda sublayer, shared by every round (the core weights are tied; sources carry no
round identity), plus the head. With ``w = 0`` every input is the mean of its sources; the sum
of the sources is the plain residual stream of ``loop_carrier='residual'``, so after each
sublayer's RMSNorm the two carriers agree up to the norm's eps (the oracle tests).

Per-row exit (``set_loop_depths``): row ``b`` with depth ``d_b`` reads at most its own
``1 + k d_b`` history entries in every later round, in the coda and in the head (a source mask:
its later block sums get weight exactly 0 and an exactly-zero gradient); in rounds it has left
it is padding for the MoE router and the DSv4 indexer loss, as in ``LoopHybridStack``.

Ported from NVIDIA/Megatron-LM PR #6840 ("Attention Residuals",
https://github.com/NVIDIA/Megatron-LM/pull/6840; source: https://github.com/jingqiny-99/Megatron-LM
branch jingqiny/attention-residuals-dev @ dc619f559, Apache-2.0, same copyright holder):
``AttnResHybridLayer.forward`` (one hybrid entry = one sublayer: aggregate, delta,
``partial + delta``; block-start entries see no partial) and the stack's block-start append
(megatron/core/models/hybrid/hybrid_block.py), the ``final_attn_res`` output head. The wrapper
module is ported as the methods ``_attn_res_sublayer`` / ``_attn_res_range`` of this stack so
that layer keys stay those of the other loop variants. See ``loop/attn_res.py`` for the
aggregation itself; the reference math is Kimi K3's ``_apply_attn_res``
(https://huggingface.co/moonshotai/Kimi-K3 @ f831ab66, modeling_kimi_linear.py).

Eager only: RSBF, CUDA graphs, the loop parameter store, mHC, MTP, truncated BPTT and MoR are
refused (``guards.py``). Checkpoint keys: ``decoder.loop_attn_res.{j}.pseudo_query`` /
``.key_norm_weight`` (``j`` in ``[0, core + coda)``) and ``decoder.loop_attn_res_head.*``,
replicated, fp32, through the default walk; no ``loop_adapter`` / ``loop_norm``.
"""

import dataclasses
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from megatron.core.models.hybrid.hybrid_block import HybridStackLayerContext
from megatron.core.tensor_observation import is_observing_tensor, observe_layer_residuals
from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
    indexer_loss_padding_mask,
)
from megatron.core.transformer.forward_replay import forward_statistics_round
from megatron.core.transformer.loop.attn_res import (
    AttnResStats,
    AttnResStatsSink,
    LoopAttentionResidual,
    StatsKey,
    attn_res_block_offsets,
    fan_out,
    history_lengths,
    source_mask_for_lengths,
)
from megatron.core.transformer.loop.guards import ATTN_RES_BLOCKS
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.loop.reporting import stream_mean_rms
from megatron.core.transformer.loop_transformer_block import loop_round_checkpoint


class _SourceMasks:
    """The source masks of one forward, built on first use per ``(n_hist, has_partial)``
    (``attn_res.source_mask_for_lengths``; None while every row reads every source)."""

    def __init__(self, lengths: Optional[Tuple[int, ...]], device: torch.device) -> None:
        self.lengths = lengths
        self.device = device
        self.cache: Dict[Tuple[int, bool], Optional[Tensor]] = {}

    def get(self, n_hist: int, has_partial: bool) -> Optional[Tensor]:
        """The mask of a call with ``n_hist`` history entries and an optional partial."""
        key = (n_hist, has_partial)
        if key not in self.cache:
            self.cache[key] = source_mask_for_lengths(
                n_hist, self.lengths, has_partial, self.device
            )
        return self.cache[key]


def unreached_metric(device):
    """A diagnostics metric this forward did not reach: value 0, population 0."""
    zero = torch.zeros((), dtype=torch.float32, device=device)
    return zero, zero


def attn_res_source_tag(segment: str, round_index: int, source: int, blocks_per_round: int):
    """The diagnostics tag of source ``source`` of a call keyed ``(segment, *, round_index)``.

    Core round ``r``: ``e`` (source 0), ``lag{d}`` (a block sum of round ``r - d``) or
    ``current`` (the round's own sealed blocks and partial). Coda / head (``round_index`` = the
    forward's round count ``R``): ``e``, ``round{q}`` (a block sum of round ``q``) or
    ``current`` (the coda partial).
    """
    if source == 0:
        return "e"
    origin = (source - 1) // blocks_per_round + 1
    if segment == "core":
        return "current" if origin >= round_index else f"lag{round_index - origin}"
    return "current" if origin > round_index else f"round{origin}"


def summary_metric_names(max_rounds: int, has_coda: bool) -> List[Tuple[str, str, int]]:
    """``(metric name, segment, round)`` of every summary weight metric, a fixed set per
    model (the loss-dict reduction needs the same keys on every microbatch and rank)."""
    names = []
    for r in range(1, max_rounds + 1):
        for tag in ["e", *(f"lag{d}" for d in range(1, r)), "current"]:
            names.append((f"loop attnres weight r{r} {tag}", "core", r))
    for segment in ("coda", "head") if has_coda else ("head",):
        for tag in ["e", *(f"round{q}" for q in range(1, max_rounds + 1)), "current"]:
            names.append((f"loop attnres weight {segment} {tag}", segment, 0))
    return names


def summarize_attn_res_weights(
    sums: Dict[StatsKey, Tuple[Tensor, Tensor]],
    blocks_per_round: int,
    max_rounds: int,
    has_coda: bool,
    device: torch.device,
) -> Dict[str, Tuple[Tensor, Tensor]]:
    """The summary diagnostics of one forward: ``{metric name: (mean weight, tokens)}``.

    ``sums``: ``AttnResStats.take()`` of one forward. Names: ``summary_metric_names`` (core
    round ``r``: ``loop attnres weight r{r} e|lag{d}|current``; coda and head: ``loop attnres
    weight {coda|head} e|round{q}|current``). Each value is the token-weighted mean weight,
    averaged over the segment's sublayers (a sublayer without that source counts 0, so a
    segment's values sum to 1); ``tokens`` is the population (the counted tokens of one
    sublayer call). A metric this forward did not reach (a round it did not run, a round
    beyond its round count) is 0 with population 0.
    """
    groups: Dict[Tuple[str, int], List[Tuple[int, Tensor, Tensor]]] = {}
    for (segment, _, round_index), (weights, tokens) in sums.items():
        group = (segment, round_index if segment == "core" else 0)
        groups.setdefault(group, []).append((round_index, weights, tokens))
    values: Dict[str, Tensor] = {}
    populations: Dict[str, Tensor] = {}
    for (segment, round_index), calls in groups.items():
        prefix = (
            f"loop attnres weight r{round_index}"
            if segment == "core"
            else f"loop attnres weight {segment}"
        )
        for call_round, weights, call_tokens in calls:
            means = weights / call_tokens.clamp_min(1.0) / len(calls)
            for source in range(means.shape[0]):
                tag = attn_res_source_tag(segment, call_round, source, blocks_per_round)
                name = f"{prefix} {tag}"
                values[name] = values[name] + means[source] if name in values else means[source]
                populations[name] = calls[0][2]
    zero = torch.zeros((), dtype=torch.float32, device=device)
    out: Dict[str, Tuple[Tensor, Tensor]] = {}
    for name, segment, round_index in summary_metric_names(max_rounds, has_coda):
        key = (segment, round_index)
        if name in values:
            out[name] = (values[name], populations[name])
        else:
            population = groups[key][0][2] if key in groups else zero
            out[name] = (zero, population)
    return out


class LoopAttnResHybridStack(LoopHybridStack):
    """The K3 attention-residual carrier (``loop_carrier='attn_res'``), eager only.

    Same constructor as ``HybridStack``. Children: ``loop_attn_res`` (one
    ``LoopAttentionResidual`` per core and coda sublayer, in sublayer order) and
    ``loop_attn_res_head``; no ``loop_adapter`` / ``loop_norm``.
    """

    _LOOP_CARRIERS = ("attn_res",)

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
        roles = self.loop_roles
        core_symbols = self._loop_symbols[roles.core_start : roles.core_end]
        try:
            self._attn_res_offsets = attn_res_block_offsets(
                core_symbols, config.loop_attn_res_blocks_per_round
            )
        except ValueError as error:
            raise ValueError(f"Hybrid loop does not support: {ATTN_RES_BLOCKS} ({error})") from None
        # Built after the layers and without init_method: no RNG draw (zeros / ones).
        self.loop_attn_res = nn.ModuleList(
            LoopAttentionResidual(config, device=self._loop_device)
            for _ in range(roles.core + roles.coda)
        )
        self.loop_attn_res_head = LoopAttentionResidual(config, device=self._loop_device)
        self._carrier_sink: Optional[AttnResStatsSink] = None
        self._carrier_totals = AttnResStats()
        self._carrier_per_sublayer = False

    # ---------------------------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------------------------

    def enable_carrier_diagnostics(
        self, enabled: bool, sink: Optional[AttnResStatsSink] = None, per_sublayer: bool = False
    ) -> None:
        """``LoopHybridStack.enable_carrier_diagnostics`` plus the attention-residual weights.

        When on, each training forward adds the summary weights of
        ``summarize_attn_res_weights`` and the state norms ``loop state rms prelude``,
        ``loop state rms r{r}`` (the stream ``e + b_1 + ... + b_r``, every block of rounds
        ``<= r``; at ``w = 0`` this is the residual carrier's state) and ``loop attnres block
        rms r{r}`` (the sum of round ``r``'s block sums) to ``loop_metrics``. With
        ``per_sublayer`` (``--loop-carrier-diagnostics full``) it also accumulates the per-call
        weight sums for ``take_carrier_diagnostics``; otherwise nothing accumulates and the take
        is empty. ``sink`` (optional) also receives every call's masked weights
        (``attn_res.AttnResStatsSink``).
        """
        self._carrier_diagnostics = bool(enabled)
        self._carrier_sink = sink if enabled else None
        self._carrier_per_sublayer = bool(enabled) and bool(per_sublayer)
        if not self._carrier_per_sublayer:
            self._carrier_totals.take()

    def take_carrier_diagnostics(self) -> Optional[dict]:
        """``{"weights": {(segment, sublayer, round): (weight sums [n], tokens)}, "k": k}``
        accumulated over the training forwards since the last call (device tensors; only with
        ``enable_carrier_diagnostics(..., per_sublayer=True)``), then reset; ``segment`` is
        ``core`` (``round`` = the round, 1..R) or ``coda`` / ``head`` (``round`` = the
        forward's round count R, which fixes the call's source count); the tags of the sources
        are ``attn_res_source_tag``."""
        return {
            "weights": self._carrier_totals.take(),
            "k": self.config.loop_attn_res_blocks_per_round,
        }

    def _row_weight(self, ctx: HybridStackLayerContext) -> Optional[Tensor]:
        if ctx.padding_mask is None:
            return None
        return (~ctx.padding_mask).transpose(0, 1).to(torch.float32)

    # ---------------------------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------------------------

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
        """The attention-residual rounds, the coda and the output head (see the module)."""
        if self._loop_store_hook is not None:
            raise RuntimeError("The loop parameter store does not run loop_carrier='attn_res'")
        roles = self.loop_roles
        batch = embedding.shape[1]
        k = self.config.loop_attn_res_blocks_per_round
        if depths is None:
            rounds = self._depth(embedding.device)
            all_alive = rounds
            lengths = None
        else:
            rounds, all_alive = self._row_rounds(depths, batch)
            lengths = history_lengths(depths.tolist(), k)
        masks = _SourceMasks(lengths, embedding.device)
        diagnostics = self._diagnostics_active()
        forward_stats = AttnResStats() if diagnostics else None
        sink = fan_out(forward_stats, self._carrier_sink) if diagnostics else None
        stream = embedding.detach().float() if diagnostics else None

        history: List[Tensor] = [embedding]
        statistic_rounds = self.config.loop_iterations
        for step in range(rounds):
            self._record_range(
                roles.core_start,
                roles.core_end,
                tokens,
                tokens * sequence,
                sequence,
                recompute=self.config.loop_recompute,
            )
            round_ctx = ctx
            # While every row (of the TP x EP group) is alive the round is the uniform-depth
            # round (no padding mask, no source mask): equal depths equal uniform depth bitwise.
            if step + 1 > all_alive:
                exited = (depths < step + 1)[:, None]
                round_ctx = dataclasses.replace(
                    ctx,
                    padding_mask=(
                        exited.repeat(1, sequence)
                        if ctx.padding_mask is None
                        else ctx.padding_mask | exited
                    ),
                )

            # Bind the step in the closure: backward replays run after this Python loop.
            def round_fn(*sources, step=step, round_ctx=round_ctx):
                return self._attn_res_round(sources, step, round_ctx, masks, sink, label)

            binned = (
                forward_statistics_round(step + 1, statistic_rounds)
                if step < statistic_rounds
                else nullcontext()
            )
            with binned:
                if self.config.loop_recompute and torch.is_grad_enabled():
                    # The round returns its block sums: it ends in its last sublayer.
                    blocks = loop_round_checkpoint(
                        round_fn,
                        *history,
                        label=f"{label}/loop={step + 1}",
                        round_ends_in_layer=True,
                    )
                else:
                    blocks = round_fn(*history)
            history.extend(blocks)
            if diagnostics:
                with torch.no_grad():
                    block = sum(b.detach().float() for b in blocks)
                    stream = stream + block
                    valid = (
                        None
                        if round_ctx.padding_mask is None
                        else ~round_ctx.padding_mask.transpose(0, 1)
                    )
                    self._diagnostic_metrics[f"loop state rms r{step + 1}"] = stream_mean_rms(
                        stream, 1, valid
                    )
                    self._diagnostic_metrics[f"loop attnres block rms r{step + 1}"] = (
                        stream_mean_rms(block, 1, valid)
                    )

        self._record_range(roles.core_end, len(self.layers), tokens, squared_lengths, sequence)
        self._record_head(tokens, squared_lengths, sequence)
        with indexer_loss_padding_mask(ctx.padding_mask):
            _, partial = self._attn_res_range(
                history,
                roles.core_end,
                len(self.layers),
                roles.core,
                (0,),
                ctx,
                masks,
                sink,
                "coda",
                rounds,
                label + "/coda",
            )
            values = history if partial is None else [*history, partial]
            stats = None if sink is None else (sink, ("head", 0, rounds), self._row_weight(ctx))
            with torch.profiler.record_function(label + "/attn_res_head"):
                output = self.loop_attn_res_head(
                    values, masks.get(len(history), partial is not None), stats
                )

        device = embedding.device
        if depths is None:
            row_rounds = float(tokens * rounds)
            mean_depth = torch.tensor(float(rounds), device=device)
        else:
            row_rounds = depths.sum().float() * sequence
            mean_depth = depths.float().mean()
        core = roles.core
        self.loop_metrics.update(
            {
                "loop depth": mean_depth,
                "loop active token layers": torch.as_tensor(row_rounds * core, device=device),
                "loop backward token layers": torch.as_tensor(
                    row_rounds * core if torch.is_grad_enabled() else 0.0, device=device
                ),
            }
        )
        if diagnostics:
            if self._carrier_per_sublayer:
                self._carrier_totals.merge(forward_stats)
            self._diagnostic_metrics.update(
                summarize_attn_res_weights(
                    forward_stats.take(),
                    k,
                    self.config.loop_iterations,
                    roles.coda > 0,
                    embedding.device,
                )
            )
            for step in range(rounds, self.config.loop_iterations):
                for name in ("loop state rms", "loop attnres block rms"):
                    self._diagnostic_metrics[f"{name} r{step + 1}"] = unreached_metric(device)
        return output

    def _attn_res_round(
        self,
        sources: Sequence[Tensor],
        step: int,
        round_ctx: HybridStackLayerContext,
        masks: _SourceMasks,
        sink: Optional[AttnResStatsSink],
        label: str,
    ) -> Tuple[Tensor, ...]:
        """One core round on the history ``sources``; returns its ``k`` block sums. The
        indexer-loss scope is entered here, so an activation replay enters it too."""
        roles = self.loop_roles
        loop_label = f"{label}/loop={step + 1}"
        with (
            torch.profiler.record_function(loop_label),
            indexer_loss_padding_mask(round_ctx.padding_mask),
        ):
            history = list(sources)
            sealed, partial = self._attn_res_range(
                history,
                roles.core_start,
                roles.core_end,
                0,
                self._attn_res_offsets,
                round_ctx,
                masks,
                sink,
                "core",
                step + 1,
                loop_label,
            )
        return (*sealed, partial)

    def _attn_res_range(
        self,
        history: List[Tensor],
        start: int,
        end: int,
        module_base: int,
        offsets: Tuple[int, ...],
        ctx: HybridStackLayerContext,
        masks: _SourceMasks,
        sink: Optional[AttnResStatsSink],
        segment: str,
        round_index: int,
        label: str,
    ) -> Tuple[List[Tensor], Optional[Tensor]]:
        """Sublayers ``[start, end)`` as attention-residual entries (``AttnResHybridLayer``).

        A block starts at every offset in ``offsets`` (relative to ``start``); at a block start
        after the first the open partial is appended to ``history`` (mutated) and reset.
        Returns the sums sealed here and the open partial (None for an empty range).
        """
        delta_ctx = dataclasses.replace(ctx, return_layer_delta=True)
        row_weight = None if sink is None else self._row_weight(ctx)
        sealed: List[Tensor] = []
        partial: Optional[Tensor] = None
        for offset, index in enumerate(range(start, end)):
            if offset > 0 and offset in offsets:
                history.append(partial)
                sealed.append(partial)
                partial = None
            stats = None if sink is None else (sink, (segment, offset, round_index), row_weight)
            partial = self._attn_res_sublayer(
                index,
                self.loop_attn_res[module_base + offset],
                history,
                partial,
                delta_ctx,
                masks,
                stats,
                label,
            )
        return sealed, partial

    def _attn_res_sublayer(
        self,
        index: int,
        module: LoopAttentionResidual,
        history: Sequence[Tensor],
        partial: Optional[Tensor],
        ctx: HybridStackLayerContext,
        masks: _SourceMasks,
        stats,
        label: str,
    ) -> Tensor:
        """One hybrid entry: aggregate ``history [+ partial]``, run sublayer ``index`` as a
        delta (``_run_sublayer`` with ``ctx.return_layer_delta``) and return ``partial +
        delta`` (the delta itself at a block start)."""
        values = list(history) if partial is None else [*history, partial]
        aggregated = module(values, masks.get(len(history), partial is not None), stats)
        delta = self._run_sublayer(index, aggregated, ctx, label)
        output = delta if partial is None else partial + delta
        if is_observing_tensor("residual_accumulator") or is_observing_tensor(
            "residual_contribution"
        ):
            before = torch.zeros_like(output) if partial is None else partial
            observe_layer_residuals(self.layers[index], before, output)
        return output
