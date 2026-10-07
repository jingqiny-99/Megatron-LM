# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Static forward-only projection layouts and model-owned query snapshots.

There is no score-gradient channel, gradient finalization, or graph-time runtime
lookup here. Callers pass the prepared tensor banks and auxiliary payloads as
explicit graph surfaces; canonical AttentionResidual parameters retain their
ordinary complete consumer backward path.
"""

import math
import weakref
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from megatron.core.utils import ensure_params_ready

_RUNTIME_ATTR = '_attn_res_forward_projection_runtime'


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def _parameter_binding(param: torch.Tensor) -> tuple[Any, ...]:
    return (
        id(param),
        param.data_ptr(),
        param._version,
        tuple(param.shape),
        tuple(param.stride()),
        param.dtype,
        param.device,
    )


@dataclass(frozen=True)
class ForwardProjectionRow:
    """One token-shaped statistic; ``None`` denotes a source's inverse RMS."""

    source_id: int
    consumer_id: int | None


@dataclass(frozen=True)
class ForwardProjectionPlan:
    """Immutable dense, uniformly partitioned PP2/VP1-or-2 projection plan."""

    num_layers: int
    hidden_size: int
    block_layers: int
    pp_size: int
    vp_size: int
    fraction: float
    eps: float

    def __post_init__(self) -> None:
        for name in ('num_layers', 'hidden_size', 'block_layers', 'pp_size', 'vp_size'):
            _positive_integer(getattr(self, name), name)
        if self.pp_size != 2 or self.vp_size not in (1, 2):
            raise ValueError('Forward projection supports only PP2/VP1 or PP2/VP2')
        if self.num_layers % self.num_stages:
            raise ValueError('Forward projection requires a uniform, nonempty layer partition')
        if not math.isfinite(self.fraction) or not 0 <= self.fraction <= 1:
            raise ValueError('Forward projection fraction must be finite and in [0, 1]')
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('Forward projection epsilon must be finite and positive')

    @classmethod
    def from_config(cls, config: Any, *, data_parallel_size: int = 1) -> 'ForwardProjectionPlan':
        """Validate the bounded experiment and freeze its configuration values."""
        if (
            type(data_parallel_size) is not int
            or data_parallel_size != 1
            or any(
                getattr(config, name, 1) != 1
                for name in (
                    'tensor_model_parallel_size',
                    'context_parallel_size',
                    'expert_model_parallel_size',
                )
            )
        ):
            raise ValueError('Forward projection requires TP=CP=EP=DP=1')
        unsupported = (
            'is_hybrid_model',
            'num_moe_experts',
            'mtp_num_layers',
            'sequence_parallel',
            'cpu_offloading',
            'fine_grained_activation_offloading',
            'pipeline_model_parallel_layout',
            'num_layers_in_first_pipeline_stage',
            'num_layers_in_last_pipeline_stage',
            'account_for_embedding_in_pipeline_split',
            'account_for_loss_in_pipeline_split',
            'standalone_embedding_stage',
        )
        if any(getattr(config, name, None) for name in unsupported):
            raise ValueError('Forward projection requires dense uniform stages without MTP/offload')
        if not getattr(config, 'enable_attention_residuals', False):
            raise ValueError('Forward projection requires Attention Residuals')
        if getattr(config, 'attn_res_impl', 'fla') != 'fla':
            raise ValueError('Forward projection preserves the original FLA consumer backward')
        return cls(
            config.num_layers,
            config.hidden_size,
            config.attn_res_block_layers,
            config.pipeline_model_parallel_size,
            config.virtual_pipeline_model_parallel_size or 1,
            config.attn_res_forward_projection_fraction,
            config.layernorm_epsilon,
        )

    @property
    def num_stages(self) -> int:
        """Number of chunk-major logical pipeline stages."""
        return self.pp_size * self.vp_size

    @property
    def num_consumers(self) -> int:
        """Attention, MLP and final-aggregation query rows."""
        return 2 * self.num_layers + 1

    def layer_offset(self, global_stage: int) -> int:
        """Return completed layers before a chunk-major stage, including the endpoint."""
        if type(global_stage) is not int or not 0 <= global_stage <= self.num_stages:
            raise ValueError('Invalid logical pipeline stage')
        return global_stage * (self.num_layers // self.num_stages)

    def consumer_layer(self, consumer_id: int) -> int:
        """Map a consumer to its layer; final aggregation has logical layer L+1."""
        if type(consumer_id) is not int or not 0 <= consumer_id < self.num_consumers:
            raise ValueError('Invalid forward projection consumer ID')
        return consumer_id // 2 + 1

    def consumer_owner(self, consumer_id: int) -> int:
        """Physical pipeline rank owning the canonical consumer parameters."""
        layer = min(self.consumer_layer(consumer_id), self.num_layers)
        stage = (layer - 1) // (self.num_layers // self.num_stages)
        return stage % self.pp_size

    def source_completion_layer(self, source_id: int) -> int:
        """Completion precedes registration; the trailing final partial is never cached."""
        if (
            type(source_id) is not int
            or source_id < 0
            or source_id * self.block_layers >= self.num_layers
        ):
            raise ValueError('Invalid historical source ID; trailing partial must remain local')
        return source_id * self.block_layers

    def completed_sources(self, layers_before: int) -> tuple[int, ...]:
        """Completed immutable sources, including an unregistered outgoing full block."""
        if type(layers_before) is not int or not 0 <= layers_before <= self.num_layers:
            raise ValueError('Invalid completed layer count')
        last = min(layers_before // self.block_layers, (self.num_layers - 1) // self.block_layers)
        return tuple(range(last + 1))

    def selected_consumers(self, source_id: int) -> tuple[int, ...]:
        """Choose the latest ceil(fraction * eligible) consumers, preserving order."""
        completed = self.source_completion_layer(source_id)
        eligible = tuple(range(2 * completed, self.num_consumers))
        count = math.ceil(self.fraction * len(eligible))
        return eligible[-count:] if count else ()

    def entry_rows(self, global_stage: int) -> tuple[ForwardProjectionRow, ...]:
        """Full prefix of still-needed FP32 statistics entering a logical stage."""
        offset = self.layer_offset(global_stage)
        if global_stage in (0, self.num_stages):
            return ()
        rows = []
        for source in self.completed_sources(offset):
            consumers = tuple(
                cid for cid in self.selected_consumers(source) if self.consumer_layer(cid) > offset
            )
            if consumers:
                rows.append(ForwardProjectionRow(source, None))
                rows.extend(ForwardProjectionRow(source, cid) for cid in consumers)
        return tuple(rows)

    def exit_rows(self, global_stage: int) -> tuple[ForwardProjectionRow, ...]:
        """Logical rows sent after this stage; no auxiliary output follows the final stage."""
        if type(global_stage) is not int or not 0 <= global_stage < self.num_stages:
            raise ValueError('Invalid logical pipeline stage')
        return self.entry_rows(global_stage + 1)

    @property
    def padded_rows(self) -> int:
        """Uniform VPP wire extent, including a real zero row for fraction zero."""
        return max(1, *(len(self.entry_rows(stage)) for stage in range(1, self.num_stages)))

    def payload_shape(self, seq_length: int, micro_batch_size: int) -> tuple[int, int, int]:
        """Static FP32 auxiliary payload shape shared by all pipeline boundaries."""
        _positive_integer(seq_length, 'seq_length')
        _positive_integer(micro_batch_size, 'micro_batch_size')
        return self.padded_rows * seq_length, micro_batch_size, 1


def unpack_forward_rows(
    plan: ForwardProjectionPlan,
    boundary_stage: int,
    payload: torch.Tensor,
    seq_length: int,
    micro_batch_size: int,
) -> dict[ForwardProjectionRow, torch.Tensor]:
    """Decode explicit row views without attributes, caches or gradient edges."""
    layout = plan.entry_rows(boundary_stage)
    if (
        payload.shape != plan.payload_shape(seq_length, micro_batch_size)
        or payload.dtype != torch.float32
        or payload.requires_grad
        or not payload.is_contiguous()
    ):
        raise ValueError('Forward auxiliary payload must be contiguous nondifferentiable FP32')
    rows = payload.view(plan.padded_rows, seq_length, micro_batch_size)
    return {key: rows[index] for index, key in enumerate(layout)}


def pack_forward_rows(
    plan: ForwardProjectionPlan,
    boundary_stage: int,
    rows: Mapping[ForwardProjectionRow, torch.Tensor],
    seq_length: int,
    micro_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Pack the required full prefix into owning, uniformly padded FP32 storage."""
    layout = plan.entry_rows(boundary_stage)
    shape = plan.payload_shape(seq_length, micro_batch_size)
    device = torch.device(device)
    selected = []
    for key in layout:
        if key not in rows:
            raise ValueError(f'Missing completed-source statistic {key}')
        value = rows[key]
        if (
            value.shape != (seq_length, micro_batch_size)
            or value.dtype != torch.float32
            or value.device != device
            or value.requires_grad
            or not value.is_contiguous()
        ):
            raise ValueError('Forward statistics must be contiguous nondifferentiable FP32 [S,B]')
        selected.append(value)
    payload = torch.zeros(shape, dtype=torch.float32, device=device)
    output_rows = payload.view(plan.padded_rows, seq_length, micro_batch_size)
    for index, value in enumerate(selected):
        output_rows[index].copy_(value)
    return payload


@dataclass(frozen=True)
class ForwardProjectionSnapshot:
    """One prepared step's raw, nondifferentiable query and norm tensor banks."""

    epoch: int
    plan: ForwardProjectionPlan
    query_bank: torch.Tensor
    gamma_bank: torch.Tensor
    _runtime: Any = field(repr=False, compare=False)
    _bank_versions: tuple[int, int] = field(repr=False, compare=False)

    def _validate_current(self, epoch: int, plan: ForwardProjectionPlan) -> Any:
        runtime = self._runtime()
        if (
            runtime is None
            or runtime.snapshot is not self
            or epoch != self.epoch
            or plan != self.plan
        ):
            raise RuntimeError('Stale or foreign forward projection snapshot')
        for tensor, version in zip((self.query_bank, self.gamma_bank), self._bank_versions):
            if (
                tensor.shape != (plan.num_consumers, plan.hidden_size)
                or tensor.dtype != torch.float32
                or tensor.requires_grad
                or not tensor.is_contiguous()
                or tensor._version != version
            ):
                raise RuntimeError('Prepared forward projection bank was modified')
        return runtime

    def validate(self, epoch: int, plan: ForwardProjectionPlan) -> None:
        """Reject stale/mutated host snapshots before dispatching graph surfaces."""
        runtime = self._validate_current(epoch, plan)
        runtime.validate_model()
        if runtime.parameter_versions() != runtime._prepared_parameter_versions:
            raise RuntimeError('Canonical parameters changed after forward projection preparation')

    def validate_module(self, consumer_id: int, module: torch.nn.Module) -> None:
        """Authorize this snapshot's local canonical consumer at the host boundary."""
        runtime = self._validate_current(self.epoch, self.plan)
        if (
            runtime.local_modules.get(consumer_id) is not module
            or module.forward_projection_consumer_id != consumer_id
        ):
            raise RuntimeError(
                'Consumer module does not belong to this forward projection snapshot'
            )
        index = runtime._local_module_indices[consumer_id] * 2
        current = tuple(
            _parameter_binding(param) for param in (module.pseudo_query, module.key_norm_weight)
        )
        if (
            module.eps != self.plan.eps
            or current != runtime._prepared_parameter_versions[index : index + 2]
        ):
            raise RuntimeError('Canonical parameters changed after forward projection preparation')


def _model_chunks(model: torch.nn.Module | list[torch.nn.Module]) -> tuple[torch.nn.Module, ...]:
    chunks = tuple(model) if isinstance(model, (list, tuple)) else (model,)
    if not chunks or any(not isinstance(chunk, torch.nn.Module) for chunk in chunks):
        raise ValueError('Expected nonempty model chunks')
    return chunks


class AttnResForwardProjectionRuntime:
    """A model-owned registry; only snapshot publication communicates each step."""

    def __init__(
        self,
        chunks: tuple[torch.nn.Module, ...],
        pp_group: torch.distributed.ProcessGroup,
        *,
        data_parallel_size: int = 1,
    ) -> None:
        self.chunks = chunks
        self.pp_group = pp_group
        self.data_parallel_size = data_parallel_size
        self.plan = ForwardProjectionPlan.from_config(
            chunks[0].config, data_parallel_size=data_parallel_size
        )
        if pp_group is None or pp_group.size() != self.plan.pp_size:
            raise ValueError('Pass the explicit two-rank pipeline process group')
        self.pp_rank = pp_group.rank()
        self.modules = tuple(
            dict.fromkeys(
                module
                for chunk in chunks
                for module in chunk.modules()
                if hasattr(module, 'forward_projection_consumer_id')
            )
        )
        errors, records = [], []
        if len(chunks) != self.plan.vp_size:
            errors.append('Number of local model chunks differs from the static plan')
        for index, module in enumerate(self.modules):
            cid = module.forward_projection_consumer_id
            try:
                owner = self.plan.consumer_owner(cid)
                if owner != self.pp_rank:
                    raise ValueError('Consumer assigned to a different physical pipeline owner')
                if module.eps != self.plan.eps:
                    raise ValueError('All consumer RMS epsilons must match')
                for param in (module.pseudo_query, module.key_norm_weight):
                    if not isinstance(param, torch.nn.Parameter) or param.shape != (
                        self.plan.hidden_size,
                    ):
                        raise ValueError('Canonical query/norm must be Parameters with shape [H]')
                    if param.dtype != torch.float32 or not param.is_contiguous():
                        raise ValueError('Canonical query/norm must retain contiguous FP32 storage')
                    if getattr(param, '_externally_managed_grad', False):
                        raise ValueError('Forward projection must not change gradient ownership')
                records.append((cid, self.pp_rank, index, module.eps))
            except (AttributeError, TypeError, ValueError) as error:
                errors.append(str(error))
        gathered = [None] * self.plan.pp_size
        torch.distributed.all_gather_object(
            gathered, {'plan': self.plan, 'records': records, 'errors': errors}, group=pp_group
        )
        all_records = []
        for owner, entry in enumerate(gathered):
            errors.extend(entry['errors'])
            if entry['plan'] != self.plan:
                errors.append('Pipeline ranks disagree on the immutable projection plan')
            for record in entry['records']:
                if record[1] != owner or self.plan.consumer_owner(record[0]) != owner:
                    errors.append('Invalid registry owner')
                if record[3] != self.plan.eps:
                    errors.append('Pipeline ranks disagree on RMS epsilon')
                all_records.append(record)
        if sorted(record[0] for record in all_records) != list(range(self.plan.num_consumers)):
            errors.append(
                'Consumer IDs must occur exactly once globally, including final aggregation'
            )
        if errors:
            raise ValueError('; '.join(sorted(set(errors))))
        self.consumer_metadata = tuple(sorted(all_records))
        self.local_modules = {
            module.forward_projection_consumer_id: module for module in self.modules
        }
        self._local_module_indices = {
            module.forward_projection_consumer_id: index
            for index, module in enumerate(self.modules)
        }
        self._module_identity = tuple(
            (
                id(module),
                module.forward_projection_consumer_id,
                id(module.pseudo_query),
                id(module.key_norm_weight),
            )
            for module in self.modules
        )
        self.snapshot = None
        self.epoch = 0
        self._prepared_parameter_versions = ()
        self.validate_model()

    def validate_model(self) -> None:
        """Detect configuration or canonical Parameter replacement after registration."""
        if any(
            not getattr(chunk.config, 'attn_res_forward_projection', False)
            or ForwardProjectionPlan.from_config(
                chunk.config, data_parallel_size=self.data_parallel_size
            )
            != self.plan
            for chunk in self.chunks
        ):
            raise RuntimeError('Model configuration changed after forward projection registration')
        identity = tuple(
            (
                id(module),
                module.forward_projection_consumer_id,
                id(module.pseudo_query),
                id(module.key_norm_weight),
            )
            for module in self.modules
        )
        current_modules = tuple(
            dict.fromkeys(
                module
                for chunk in self.chunks
                for module in chunk.modules()
                if hasattr(module, 'forward_projection_consumer_id')
            )
        )
        if identity != self._module_identity or current_modules != self.modules:
            raise RuntimeError('Canonical consumer identity changed after registration')
        if any(
            module.eps != self.plan.eps
            or any(
                param.shape != (self.plan.hidden_size,)
                or param.dtype != torch.float32
                or not param.is_contiguous()
                or getattr(param, '_externally_managed_grad', False)
                for param in (module.pseudo_query, module.key_norm_weight)
            )
            for module in self.modules
        ):
            raise RuntimeError('Canonical consumer math or gradient-ownership contract changed')

    def parameter_versions(self) -> tuple[tuple[Any, ...], ...]:
        """Read local canonical identity/version metadata without inspecting tensor values."""
        return tuple(
            _parameter_binding(param)
            for module in self.modules
            for param in (module.pseudo_query, module.key_norm_weight)
        )

    def prepare(self, *, epoch: int | None = None) -> ForwardProjectionSnapshot:
        """Publish raw FP32 banks before any blocking receive or graph execution."""
        self.validate_model()
        epoch = self.epoch + 1 if epoch is None else epoch
        _positive_integer(epoch, 'epoch')
        if epoch <= self.epoch:
            raise RuntimeError('Forward projection epochs must increase monotonically')
        params = tuple(
            dict.fromkeys(
                param
                for module in self.modules
                for param in (module.pseudo_query, module.key_norm_weight)
            )
        )
        ensure_params_ready(params)
        device = params[0].device
        if any(param.device != device for param in params):
            raise ValueError('Local query/norm parameters must share one device')
        if device.type == 'cuda' and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare forward projection banks outside CUDA graph capture')
        with torch.no_grad():
            query_bank = torch.zeros(
                (self.plan.num_consumers, self.plan.hidden_size), dtype=torch.float32, device=device
            )
            gamma_bank = torch.zeros_like(query_bank)
            for module in self.modules:
                cid = module.forward_projection_consumer_id
                query_bank[cid].copy_(module.pseudo_query)
                gamma_bank[cid].copy_(module.key_norm_weight)
            # Every row has one registered owner. These forward-only publications
            # introduce no additional autograd path or query-gradient reduction.
            torch.distributed.all_reduce(
                query_bank, op=torch.distributed.ReduceOp.SUM, group=self.pp_group
            )
            torch.distributed.all_reduce(
                gamma_bank, op=torch.distributed.ReduceOp.SUM, group=self.pp_group
            )
        self.epoch = epoch
        self._prepared_parameter_versions = self.parameter_versions()
        self.snapshot = ForwardProjectionSnapshot(
            epoch,
            self.plan,
            query_bank,
            gamma_bank,
            weakref.ref(self),
            (query_bank._version, gamma_bank._version),
        )
        return self.snapshot


def get_attn_res_forward_runtime(
    model: torch.nn.Module | list[torch.nn.Module],
) -> AttnResForwardProjectionRuntime | None:
    """Read only this model's runtime, never an alias on a shared configuration."""
    chunks = _model_chunks(model)
    runtime = getattr(chunks[0], _RUNTIME_ATTR, None)
    if runtime is not None and (
        runtime.chunks != chunks
        or any(getattr(chunk, _RUNTIME_ATTR, None) is not runtime for chunk in chunks)
    ):
        raise RuntimeError('Forward projection runtime belongs to different model chunks')
    return runtime


def prepare_attn_res_forward_projection(
    model: torch.nn.Module | list[torch.nn.Module],
    pp_group: torch.distributed.ProcessGroup,
    *,
    epoch: int | None = None,
    data_parallel_size: int = 1,
) -> ForwardProjectionSnapshot | None:
    """No-op when disabled; otherwise register once and publish a fresh step snapshot."""
    chunks = _model_chunks(model)
    enabled = [
        bool(getattr(chunk.config, 'attn_res_forward_projection', False)) for chunk in chunks
    ]
    if not any(enabled):
        return None
    if not all(enabled):
        raise ValueError('All model chunks must agree on forward projection enablement')
    runtime = get_attn_res_forward_runtime(chunks)
    if runtime is None:
        runtime = AttnResForwardProjectionRuntime(
            chunks, pp_group, data_parallel_size=data_parallel_size
        )
        for chunk in chunks:
            setattr(chunk, _RUNTIME_ATTR, runtime)
    elif runtime.pp_group is not pp_group or runtime.data_parallel_size != data_parallel_size:
        raise RuntimeError('Forward projection process-group contract changed')
    return runtime.prepare(epoch=epoch)
