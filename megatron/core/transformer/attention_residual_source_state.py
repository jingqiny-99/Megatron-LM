# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Immutable source projections and their pipeline/cache lifecycle.

The value and score graphs travel together but retain their original dtypes.
Score layouts count completed sources, including a completed outgoing partial
which the next block has not yet registered in its historical value prefix.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class SourceProjection:
    """Token-shaped scores for selected globally ordered consumer columns."""

    source_id: int
    columns: Tuple[int, ...]
    scores: torch.Tensor


class _ScoreGradientAnchor(torch.autograd.Function):
    """Visit score taps even when a chunk forwards no scores for that source.

    A later local virtual chunk can accumulate into a cached score leaf even
    when this chunk consumes none of the selected columns. An explicit zero
    VJP keeps the original tap reachable so it drains those future gradients.
    No score values are read or saved, and no floating-point zero multiplication
    can introduce overflow or change the forward result.
    """

    @staticmethod
    def forward(ctx, payload, *scores):
        """Return the payload while retaining only score layout metadata."""
        ctx.score_specs = [(score.shape, score.dtype, score.device) for score in scores]
        return payload.view_as(payload)

    @staticmethod
    def backward(ctx, grad_payload):
        """Visit every score tap through an explicit zero gradient."""
        return (grad_payload,) + tuple(
            torch.zeros(shape, dtype=dtype, device=device)
            for shape, dtype, device in ctx.score_specs
        )


def copy_source_projection(source, target):
    """Preserve immutable metadata across an explicit view or detach boundary."""
    record = getattr(source, '_attn_res_source_projection', None)
    if record is not None:
        target._attn_res_source_projection = record
        target._do_not_offload = True
    return target


def detach_attn_res_source(source):
    """Detach a trunk value while retaining trainable, source-masked MTP scores."""
    return copy_source_projection(source, source.detach())


def source_logits_for_consumer(source, column):
    """Return a precomputed score, or None when this consumer scores locally."""
    record = getattr(source, '_attn_res_source_projection', None)
    if record is None or column not in record.columns:
        return None
    return record.scores[..., record.columns.index(column)]


def _runtime(config):
    from .attention_residual_projection_runtime import get_attn_res_projection_runtime

    runtime = get_attn_res_projection_runtime(config)
    if runtime is None:
        raise RuntimeError('Prepare the source-projection runtime before creating source state')
    return runtime


def _selected_columns(config, source_id):
    """Select the latest legal consumers of a completed trunk source."""
    completed_layer = min(source_id * config.attn_res_block_layers, config.num_layers)
    columns = [
        record['column']
        for record in _runtime(config).consumer_metadata
        if record['id'][0] > 0 or record['id'][2] > completed_layer
    ]
    count = math.ceil(config.attn_res_source_projection_fraction * len(columns))
    return tuple(columns[-count:]) if count else ()


def _boundary_layout(config, pp_rank, vp_stage=0):
    """Ordered (source, consumer) pairs entering a logical pipeline stage."""
    from .attention_residual import _stage_layers_before

    stage = vp_stage * config.pipeline_model_parallel_size + pp_rank
    if stage == 0:
        return ()
    layers_before = _stage_layers_before(config, stage)
    completed_count = layers_before // config.attn_res_block_layers + 1
    known_count = 0
    if vp_stage:
        previous_end = _stage_layers_before(config, stage - config.pipeline_model_parallel_size + 1)
        known_count = previous_end // config.attn_res_block_layers + 1
    metadata = _runtime(config).consumer_metadata
    return tuple(
        (source_id, column)
        for source_id in range(known_count, completed_count)
        for column in _selected_columns(config, source_id)
        if metadata[column]['id'][0] > 0 or metadata[column]['id'][2] > layers_before
    )


def attn_res_projection_payload_shape(config, seq_length, micro_batch_size, pp_rank, vp_stage=None):
    """FP32 score shape for a receiving boundary, padded uniformly for VPP."""
    if config.virtual_pipeline_model_parallel_size is not None:
        count = max(
            (
                len(
                    _boundary_layout(
                        config,
                        stage % config.pipeline_model_parallel_size,
                        stage // config.pipeline_model_parallel_size,
                    )
                )
                for stage in range(
                    1,
                    config.pipeline_model_parallel_size
                    * config.virtual_pipeline_model_parallel_size,
                )
            ),
            default=0,
        )
    else:
        count = len(_boundary_layout(config, pp_rank, vp_stage or 0))
    # A real zero-filled channel keeps all schedules' send/recv ordering identical.
    return (max(1, count) * seq_length, micro_batch_size, 1)


class ProjectionSourceState:
    """Per-chunk score records, paired with AttnResStageSources' value state."""

    def __init__(self, state, layers_before):
        from .attention_residual import get_attn_res_source_cache

        self.state = state
        self.config = state.config
        self.runtime = _runtime(self.config)
        self.layers_completed = layers_before
        self.records = {}
        self.cache_records = {}
        if state.interleaved and state.vp_stage:
            cached = get_attn_res_source_cache().projections.get(state.microbatch_id)
            if cached is None:
                raise RuntimeError(
                    'Missing source projection cache for the preceding virtual chunk'
                )
            self.records.update(cached)
            self.cache_records.update(cached)

    def _record(self, source_id, columns, scores):
        from .attention_residual import attn_res_tap_source

        scores._do_not_offload = True
        if self.state.interleaved:
            scores, cache_leaf = attn_res_tap_source(scores)
            scores._do_not_offload = True
            cache_leaf._do_not_offload = True
            self.cache_records[source_id] = SourceProjection(source_id, columns, cache_leaf)
        record = SourceProjection(source_id, columns, scores)
        self.records[source_id] = record
        return record

    @staticmethod
    def _attach(value, record):
        value._attn_res_source_projection = record
        value._do_not_offload = True
        return value

    def complete(self, value, source_id):
        """Project once at actual completion, preserving an already projected boundary value."""
        from .attention_residual_projection_kernels import project_source

        existing = getattr(value, '_attn_res_source_projection', None)
        if existing is not None and existing.source_id == source_id:
            return value
        columns = _selected_columns(self.config, source_id)
        value._do_not_offload = True
        if columns:
            indices = torch.tensor(columns, device=self.runtime.bank.device, dtype=torch.long)
            queries = self.runtime.bank.index_select(0, indices)
            queries._do_not_offload = True
            live_mask = torch.tensor(
                [
                    not self.runtime.consumer_metadata[column]['stop_source_grad']
                    for column in columns
                ],
                device=value.device,
                dtype=torch.bool,
            )
            proxy, scores = project_source(
                value, queries, live_mask, eps=self.config.layernorm_epsilon
            )
        else:
            proxy = value.view_as(value)
            scores = value.new_empty((*value.shape[:-1], 0), dtype=torch.float32)
        return self._attach(proxy, self._record(source_id, columns, scores))

    def enter(self, partial, score_payload):
        """Decode score deltas and reconnect them to received/cached value slots."""
        if self.state.pre_process:
            return self.complete(partial, 0)
        if score_payload is None:
            raise RuntimeError('Source projection requires both value and FP32 score payloads')
        layout = _boundary_layout(self.config, self.state.pp_rank, self.state.vp_stage)
        tokens, batch = partial.shape[:2]
        if score_payload.dtype != torch.float32:
            raise TypeError('Attention Residual pipeline scores must remain FP32')
        rows = score_payload.reshape(-1, tokens, batch)
        by_source = {}
        for position, (source_id, column) in enumerate(layout):
            by_source.setdefault(source_id, []).append((column, position))
        for source_id, entries in by_source.items():
            columns, positions = zip(*entries)
            scores = torch.stack([rows[position] for position in positions], dim=-1)
            self._record(source_id, tuple(columns), scores)
        for source_id, value in enumerate(self.state.graph_sources):
            self._attach_received(value, source_id)
        if self.layers_completed % self.config.attn_res_block_layers == 0:
            self._attach_received(
                partial, self.layers_completed // self.config.attn_res_block_layers
            )
        return partial

    def _attach_received(self, value, source_id):
        record = self.records.get(source_id)
        if record is None:
            # With fraction zero, or after the last selected consumer, no scores
            # for this source need to cross this boundary.
            scores = value.new_empty((*value.shape[:-1], 0), dtype=torch.float32)
            record = self._record(source_id, (), scores)
        self._attach(value, record)

    def after_layer(self, partial, layer_number):
        """Finalize a completed block before leaving its producing chunk."""
        self.layers_completed = layer_number
        if (
            layer_number % self.config.attn_res_block_layers == 0
            or layer_number == self.config.num_layers
        ):
            source_id = math.ceil(layer_number / self.config.attn_res_block_layers)
            return self.complete(partial, source_id)
        return partial

    def finish(self):
        """Mirror value-cache lifetime; autograd owns backward references."""
        from .attention_residual import get_attn_res_source_cache

        if not self.state.interleaved:
            return
        cache = get_attn_res_source_cache().projections
        if self.state.vp_stage == self.config.virtual_pipeline_model_parallel_size - 1:
            cache.pop(self.state.microbatch_id, None)
        else:
            cache[self.state.microbatch_id] = dict(self.cache_records)

    def pack(self, value_payload, partial):
        """Pack selected score pairs into a viewless FP32 companion tensor."""
        pp_size = self.config.pipeline_model_parallel_size
        recv_rank = (self.state.pp_rank + 1) % pp_size
        recv_vp = self.state.vp_stage + (self.state.pp_rank == pp_size - 1)
        layout = _boundary_layout(self.config, recv_rank, recv_vp)
        shape = attn_res_projection_payload_shape(
            self.config, partial.shape[0], partial.shape[1], recv_rank, recv_vp
        )
        rows = []
        for source_id, column in layout:
            record = self.records[source_id]
            rows.append(record.scores[..., record.columns.index(column)])
        padded_rows = shape[0] // partial.shape[0]
        if len(rows) < padded_rows:
            zero = partial.new_zeros(partial.shape[:-1], dtype=torch.float32)
            # Preserve a differentiable channel even when there are no scores.
            zero = _ScoreGradientAnchor.apply(zero, self.runtime.bank)
            rows.extend([zero] * (padded_rows - len(rows)))
        scores = torch.stack(rows).reshape(shape)
        if self.state.interleaved:
            anchors = [
                record.scores for record in self.records.values() if record.scores.requires_grad
            ]
            scores = _ScoreGradientAnchor.apply(scores, *anchors)
        scores = scores.clone()
        self.finish()
        return [value_payload, scores]
