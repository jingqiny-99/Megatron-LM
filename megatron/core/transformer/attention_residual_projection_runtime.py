# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Step-scoped effective queries for source-owned Attention Residual projections.

Canonical q/gamma Parameters remain in their original modules. The published
effective bank is an ordinary autograd leaf, not a Parameter or persistent
buffer. Its gradients are combined across PP before canonical gradients enter
the usual DDP/TP synchronization and loss-normalization path.
"""

import math
from typing import Any

import torch

from megatron.core.utils import ensure_params_ready

_RUNTIME_ATTR = '_attn_res_projection_runtime'


def mark_attn_res_projection_parameters(module: torch.nn.Module) -> None:
    """Mark source-query parameters before optimizer layouts and DDP are built."""
    for param in (module.pseudo_query, module.key_norm_weight):
        if param.requires_grad:
            param._externally_managed_grad = True


def get_attn_res_projection_runtime(module_or_config: Any) -> 'AttnResProjectionRuntime | None':
    """Return the active runtime attached to a module or its configuration."""
    runtime = getattr(module_or_config, _RUNTIME_ATTR, None)
    if runtime is None:
        runtime = getattr(getattr(module_or_config, 'config', None), _RUNTIME_ATTR, None)
    return runtime


def _model_chunks(model: torch.nn.Module | list[torch.nn.Module]) -> list[torch.nn.Module]:
    return list(model) if isinstance(model, (tuple, list)) else [model]


def _projection_modules(chunks: list[torch.nn.Module]) -> list[torch.nn.Module]:
    # Import here to avoid a cycle with AttentionResidual's construction helper.
    from .attention_residual import AttentionResidual

    seen = set()
    modules = []
    for chunk in chunks:
        for module in chunk.modules():
            if isinstance(module, AttentionResidual) and id(module) not in seen:
                seen.add(id(module))
                modules.append(module)
    return modules


class AttnResProjectionRuntime:
    """Own a static consumer registry and one differentiable bank per schedule step.

    ``consumer_metadata`` is ordered globally by semantic consumer ID. Distinct
    modules retain distinct rows even if their Parameters alias: eligibility
    and stop-source-gradient semantics belong to consumers, not Parameters.
    """

    def __init__(
        self,
        modules: list[torch.nn.Module],
        config: Any,
        pp_group: torch.distributed.ProcessGroup | None,
    ) -> None:
        self.modules = modules
        self.config = config
        self._model_configs = (config,)
        self.pp_group = pp_group
        self.pp_size = pp_group.size() if pp_group is not None else 1
        self.pp_rank = pp_group.rank() if pp_group is not None else 0
        self.bank = None
        self.forward_only = True
        self.finalized = True
        self.step = 0
        local_records = []
        for local_index, module in enumerate(modules):
            consumer_id = getattr(module, 'projection_consumer_id', None)
            if consumer_id is None:
                raise ValueError('Every source-projection AttentionResidual needs a consumer ID')
            consumer_id = tuple(consumer_id)
            if len(consumer_id) != 4 or not all(isinstance(item, int) for item in consumer_id):
                raise ValueError('AttentionResidual consumer IDs must be four-integer tuples')
            if module.pseudo_query.numel() != config.hidden_size:
                raise ValueError(
                    'All AttentionResidual queries in a pipeline must share hidden size'
                )
            for param in (module.pseudo_query, module.key_norm_weight):
                if param.requires_grad and not getattr(param, '_externally_managed_grad', False):
                    raise ValueError(
                        'Mark source-query parameters before building optimizer/DDP layouts'
                    )
                if (
                    param.requires_grad
                    and hasattr(param, 'main_grad')
                    and not hasattr(param, '_external_grad_ready_callback')
                ):
                    raise ValueError(
                        'The parameter wrapper must support external gradient publication for '
                        'AttentionResidual source projections'
                    )
            local_records.append(
                {
                    'id': consumer_id,
                    'stop_source_grad': bool(getattr(module, 'projection_stop_source_grad', False)),
                    'owner': self.pp_rank,
                    'local_index': local_index,
                    'hidden_size': config.hidden_size,
                    'eps': module.eps,
                }
            )

        if self.pp_size > 1:
            gathered = [None] * self.pp_size
            torch.distributed.all_gather_object(gathered, local_records, group=pp_group)
            records = [record for rank_records in gathered for record in rank_records]
        else:
            records = local_records
        records.sort(key=lambda record: record['id'])
        if len({record['id'] for record in records}) != len(records):
            raise ValueError('AttentionResidual consumer IDs must be globally unique')
        if any(record['hidden_size'] != config.hidden_size for record in records):
            raise ValueError('Pipeline ranks disagree on AttentionResidual hidden size')
        if any(record['eps'] != config.layernorm_epsilon for record in records):
            raise ValueError('Source projections require a shared RMS-normalization epsilon')
        self.consumer_metadata = tuple(
            {
                'id': record['id'],
                'stop_source_grad': record['stop_source_grad'],
                'owner': record['owner'],
                'column': column,
            }
            for column, record in enumerate(records)
        )
        self._module_columns = {
            id(modules[record['local_index']]): column
            for column, record in enumerate(records)
            if record['owner'] == self.pp_rank
        }

    def column(self, module: torch.nn.Module) -> int:
        """Return the globally stable bank row belonging to a local consumer."""
        return self._module_columns[id(module)]

    def effective_weight(self, module: torch.nn.Module) -> torch.Tensor:
        """Read a consumer's differentiable effective query from this step's bank."""
        if self.bank is None:
            raise RuntimeError('Prepare source projections before reading effective queries')
        weight = self.bank[self.column(module)]
        weight._do_not_offload = True
        return weight

    def columns_for_source(self, completed_layer: int, fraction: float = 1.0) -> list[int]:
        """Select the last fraction of consumers eligible for a completed trunk source."""
        if not 0.0 <= fraction <= 1.0:
            raise ValueError('Source projection fraction must be in [0, 1]')
        eligible = [
            record['column']
            for record in self.consumer_metadata
            if record['id'][0] == 1 or record['id'][2] > completed_layer
        ]
        count = math.ceil(len(eligible) * fraction)
        return eligible[-count:] if count else []

    def prepare(self, *, forward_only: bool = False) -> None:
        """Publish a fresh query snapshot before any blocking pipeline receive."""
        if not self.finalized and not self.forward_only:
            raise RuntimeError('Finalize the previous projection step before preparing another')
        params = list(
            dict.fromkeys(
                param
                for module in self.modules
                for param in (module.pseudo_query, module.key_norm_weight)
            )
        )
        ensure_params_ready(params)
        device = (
            params[0].device
            if params
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        with torch.no_grad():
            bank = torch.zeros(
                (len(self.consumer_metadata), self.config.hidden_size),
                device=device,
                dtype=torch.float32,
            )
            for module in self.modules:
                bank[self.column(module)].copy_(
                    module.pseudo_query.float() * module.key_norm_weight.float()
                )
            if self.pp_size > 1:
                # Exactly one owner writes each row; SUM publishes the complete
                # bank without introducing another persistent parameter copy.
                torch.distributed.all_reduce(bank, group=self.pp_group)
        self.bank = bank.detach().requires_grad_(not forward_only)
        self.bank._do_not_offload = True
        self.forward_only = forward_only
        self.finalized = forward_only
        self.step += 1

    def finalize(self) -> None:
        """PP-sum effective gradients and publish canonical gradients exactly once."""
        if self.forward_only:
            return
        if self.finalized:
            raise RuntimeError('AttentionResidual projection gradients were already finalized')
        if self.bank is None:
            raise RuntimeError('Prepare source projections before finalizing gradients')
        with torch.no_grad():
            grad = self.bank.grad
            if grad is None:
                grad = torch.zeros_like(self.bank)
            else:
                grad = grad.detach().contiguous()
            if self.pp_size > 1:
                torch.distributed.all_reduce(grad, group=self.pp_group)

            # First merge every consumer alias. A ready callback may immediately
            # launch a reduction, so all canonical buffers must be complete
            # before invoking even the first callback.
            canonical_grads = {}
            for module in self.modules:
                dw = grad[self.column(module)]
                for param, contribution in (
                    (module.pseudo_query, dw * module.key_norm_weight.float()),
                    (module.key_norm_weight, dw * module.pseudo_query.float()),
                ):
                    if not param.requires_grad:
                        continue
                    if param in canonical_grads:
                        canonical_grads[param].add_(contribution)
                    else:
                        canonical_grads[param] = contribution
            for param, contribution in canonical_grads.items():
                if hasattr(param, 'main_grad'):
                    param.main_grad.add_(contribution)
                    param.grad_added_to_main_grad = True
                elif param.grad is None:
                    param.grad = contribution.to(param.dtype)
                else:
                    param.grad.add_(contribution)
            for param in canonical_grads:
                callback = getattr(param, '_external_grad_ready_callback', None)
                if callback is not None:
                    callback()
            self.bank.grad = None
        self.finalized = True


def prepare_attn_res_projection(
    model: torch.nn.Module | list[torch.nn.Module],
    pp_group: torch.distributed.ProcessGroup | None,
    *,
    forward_only: bool = False,
) -> AttnResProjectionRuntime | None:
    """Prepare source projections collectively at entry to a pipeline schedule."""
    chunks = _model_chunks(model)
    config = getattr(chunks[0], 'config', None)
    if (
        config is None
        or not getattr(config, 'enable_attention_residuals', False)
        or getattr(config, 'attn_res_impl', None) != 'source'
    ):
        return None
    # A config may be reused by separately constructed models. Only the model
    # attachment identifies this particular registry; config lookup is for its
    # source-state consumers, not registry ownership.
    runtime = getattr(chunks[0], _RUNTIME_ATTR, None)
    if runtime is None:
        modules = _projection_modules(chunks)
        runtime = AttnResProjectionRuntime(modules, config, pp_group)
        model_configs = {id(config): config}
        for chunk in chunks:
            for module in chunk.modules():
                setattr(module, _RUNTIME_ATTR, runtime)
                module_config = getattr(module, 'config', None)
                if module_config is not None:
                    model_configs[id(module_config)] = module_config
        runtime._model_configs = tuple(model_configs.values())
    elif runtime.pp_group is not pp_group:
        raise ValueError('The projection runtime cannot change pipeline process groups')
    # Independent models may share configuration objects and alternate schedule
    # calls. Source state resolves its runtime through those configs, so restore
    # their active owner even when this model already has a cached registry.
    for model_config in runtime._model_configs:
        setattr(model_config, _RUNTIME_ATTR, runtime)
    runtime.prepare(forward_only=forward_only)
    return runtime


def finalize_attn_res_projection(
    model: torch.nn.Module | list[torch.nn.Module], pp_group: torch.distributed.ProcessGroup | None
) -> None:
    """Finalize after backward P2P completion, before ordinary gradient finalization."""
    runtime = get_attn_res_projection_runtime(_model_chunks(model)[0])
    if runtime is not None:
        if runtime.pp_group is not pp_group:
            raise ValueError('The projection runtime cannot change pipeline process groups')
        runtime.finalize()
