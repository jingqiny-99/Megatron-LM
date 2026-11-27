# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Measure native MoK per-call storage lifetime inside a group-eight PP2/EP4 loop."""

from collections import Counter
from contextlib import nullcontext
from dataclasses import fields, is_dataclass, replace
from unittest.mock import patch

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from megatron.core import parallel_state as ps
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.forward_replay import in_forward_replay
from megatron.core.transformer.loop.glm_ring_stack import GLMRingStack
from tests.unit_tests.transformer.loop import glm_ring_mok_utils as moe
from tests.unit_tests.transformer.loop import test_glm_ring_mok_recompute_gpu as oracle
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA


def _storages(value, name=""):
    """Native dataclasses can contain BF16 tensors, MXFP8 data/scale tuples, or descriptors."""
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        if value.is_cuda and storage.nbytes():
            yield name, StorageWeakRef(storage), storage.nbytes()
    elif is_dataclass(value):
        for field in fields(value):
            yield from _storages(getattr(value, field.name), name + "." + field.name)
    elif isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            yield from _storages(child, name + f"[{i}]")


def _copy_native_record(record):
    # Keep independent storage so the diagnostic leaves the native lifetime unchanged.
    if isinstance(record, torch.Tensor):
        return record.detach().clone()
    if isinstance(record, tuple):
        return tuple(_copy_native_record(item) for item in record)
    return type(record)(
        **{
            field.name: _copy_native_record(getattr(record, field.name))
            for field in fields(record)
        }
    )


def _assert_bytes(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def _check_mxfp8_native_backward(monkeypatch, functional):
    """Compare saved and replayed native state at the actual upstream gradient of each visit."""
    from megatron.core.transformer.moe.megakernel.mok.runtime import _MoKAutograd
    from megatron.core.transformer.moe.megakernel.parameter_bridge import (
        main_grad_buffer,
    )

    run_layer = GLMRingStack._run_sublayer
    bridge_forward, bridge_backward = _MoKAutograd.forward, _MoKAutograd.backward
    native_forward, native_backward = functional.forward, functional.backward
    visits = {f"loop_hybrid/mb=0/core/lap={lap}": lap for lap in (1, 2)}
    current_lap = forward_key = backward_key = None
    originals, first_moe, checked = {}, {}, Counter()

    def layer(stack, index, x, ctx, label):
        nonlocal current_lap
        previous, current_lap = current_lap, visits.get(label)
        try:
            return run_layer(stack, index, x, ctx, label)
        finally:
            current_lap = previous

    def forward_bridge(ctx, module, *args):
        nonlocal forward_key
        previous = forward_key
        key = (module.checkpoint_forward_context, current_lap)
        if current_lap is not None and first_moe.setdefault(key, module) is module:
            forward_key = key
        else:
            forward_key = None
        ctx.audit_key = forward_key
        try:
            return bridge_forward(ctx, module, *args)
        finally:
            forward_key = previous

    def forward(config, workspace, schedule, x, router_weights, *weights, **options):
        output, context = native_forward(
            config, workspace, schedule, x, router_weights, *weights, **options
        )
        if forward_key is not None and torch._C._current_graph_task_id() < 0:
            assert forward_key not in originals
            originals[forward_key] = (
                _copy_native_record(schedule),
                _copy_native_record(context),
                x.detach().clone(),
                router_weights.detach().clone(),
            )
        return output, context

    def backward_bridge(ctx, dy):
        nonlocal backward_key
        previous, backward_key = backward_key, ctx.audit_key
        try:
            return bridge_backward(ctx, dy)
        finally:
            backward_key = previous

    def backward(
        config, workspace, schedule, context, dy, x, router_weights, *weights, **options
    ):
        if backward_key is None:
            return native_backward(
                config,
                workspace,
                schedule,
                context,
                dy,
                x,
                router_weights,
                *weights,
                **options,
            )
        old_schedule, old_context, old_x, old_probs = originals.pop(backward_key)
        _assert_bytes(x, old_x)
        _assert_bytes(router_weights, old_probs)
        for attr in ("num_tokens", "tokens_per_expert", "top_experts"):
            _assert_bytes(getattr(schedule, attr), getattr(old_schedule, attr))
        active = int(schedule.num_tokens.item())
        assert active > 0
        for attr in ("peer_rank", "peer_token_idx"):
            _assert_bytes(
                getattr(schedule, attr)[:active], getattr(old_schedule, attr)[:active]
            )

        # Native MoK writes directly to the parameters' FP32 accumulation buffers.
        # Restore them before the second call so both see identical initial accumulators.
        module = first_moe[backward_key]
        grad_buffers = {
            name: main_grad_buffer(param)
            for name, param in module.named_parameters()
            if param.requires_grad
        }
        assert len(grad_buffers) == 2 * (moe.EXPERTS // moe.EP) + 2
        before = {name: buf.clone() for name, buf in grad_buffers.items()}
        reference = native_backward(
            config,
            workspace,
            old_schedule,
            old_context,
            dy,
            x,
            router_weights,
            *weights,
            **options,
        )
        expected_dx, expected_router = (g.clone() for g in reference[:2])
        expected_weights = {name: buf.clone() for name, buf in grad_buffers.items()}
        for name, buf in grad_buffers.items():
            buf.copy_(before[name])
        actual = native_backward(
            config,
            workspace,
            schedule,
            context,
            dy,
            x,
            router_weights,
            *weights,
            **options,
        )
        for grad, expected in zip(
            actual[:2], (expected_dx, expected_router), strict=True
        ):
            assert torch.isfinite(grad).all() and grad.count_nonzero()
            _assert_bytes(grad, expected)
        assert any(
            not torch.equal(expected_weights[n], before[n]) for n in grad_buffers
        )
        for name, buf in grad_buffers.items():
            assert torch.isfinite(buf).all()
            _assert_bytes(buf, expected_weights[name])
        checked[backward_key] += 1
        return actual

    monkeypatch.setattr(GLMRingStack, "_run_sublayer", layer)
    monkeypatch.setattr(_MoKAutograd, "forward", staticmethod(forward_bridge))
    monkeypatch.setattr(_MoKAutograd, "backward", staticmethod(backward_bridge))
    monkeypatch.setattr(functional, "forward", forward)
    monkeypatch.setattr(functional, "backward", backward)
    return checked


@pytest.mark.skipif(
    not CUDA or oracle.ring.WORLD != 8, reason="requires eight CUDA ranks"
)
@pytest.mark.parametrize("mxfp8", [False, True], ids=["bf16", "mxfp8"])
def test_mok_context_lifetime_and_gradients_on_two_pass_ring(monkeypatch, mxfp8):
    from megatron.core.fp8_utils import get_fp8_context
    from mok import functional

    modes = iter((False, True))
    phase = False
    current_label = None
    workspace_ids = set()
    pending = []
    visits = {}
    native_forward = functional.forward
    build_schedule = functional.build_schedule
    get_workspace = functional.get_workspace
    run_layers = GLMRingStack._ring_local_layers
    stage_sources = {}

    def models(plans, *, layernorm=False):
        nonlocal phase
        phase = next(modes)
        config = replace(
            moe.config(2),
            **(
                {
                    "fp8": "e4m3",
                    "fp8_recipe": "mxfp8",
                    "fp8_param": True,
                    "disable_parameter_transpose_cache": True,
                }
                if mxfp8
                else {}
            ),
            num_attention_heads=64,
            num_query_groups=64,
            q_lora_rank=512,
            kv_lora_rank=512,
            qk_head_dim=128,
            qk_pos_emb_head_dim=64,
            v_head_dim=128,
            kv_channels=128,
            attention_backend=AttnBackend.auto,
            dsa_kernel_backend="cudnn",
            dsa_indexer_head_dim=128,
            dsa_indexer_n_heads=32,
            dsa_indexer_topk=64,
            dsa_indexer_reuse_checkpoint_topk=True,
            dsa_indexer_weights_proj_use_quantization=False,
            dsa_indexer_weights_proj_output_dtype="bf16",
            linear_cp_mode=None,
            linear_cp_layout=None,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=8,
            loop_recompute=True,
            loop_recompute_edges=True,
            loop_recompute_block_stride=4,
            loop_ring_recompute_plan="/".join(plans),
            moe_mok_checkpoint_forward_context=phase,
        )
        with (
            patch.object(moe, "config", return_value=config),
            get_fp8_context(config, is_init=True) if mxfp8 else nullcontext(),
        ):
            chunks = moe.models(2)
        return chunks

    def workspace(*args, **kwargs):
        value = get_workspace(*args, **kwargs)
        workspace_ids.update(ref for _, ref, _ in _storages(value))
        return value

    def schedule(workspace, config, top_experts, **kwargs):
        result = build_schedule(workspace, config, top_experts, **kwargs)
        if current_label is not None:
            # Raw routing input can be a view of schedule.top_experts. It has another owner.
            stage_sources[id(result)] = {ref for _, ref, _ in _storages(top_experts)}
        return result

    def native(config, workspace, schedule, x, router_weights, *weights, **options):
        output, context = native_forward(
            config, workspace, schedule, x, router_weights, *weights, **options
        )
        if current_label is not None:
            assert all(
                isinstance(getattr(context, k), tuple) == mxfp8
                for k in ("x_routed", "gate_routed", "up_routed", "hidden_routed")
            )
            excluded = workspace_ids | stage_sources.pop(id(schedule))
            excluded.update(
                ref for _, ref, _ in _storages((x, router_weights, weights, output))
            )
            unique = {}
            for kind, obj in (("schedule", schedule), ("context", context)):
                for name, ref, size in _storages(obj, kind):
                    if ref not in excluded:
                        unique.setdefault(ref, (name, size))
            assert any(name.startswith("context.") for name, _ in unique.values())
            pending.append(unique)
        return output, context

    def visit(stack, state, ctx, label, scope, recompute):
        nonlocal current_label
        current_label = label
        try:
            output = run_layers(stack, state, ctx, label, scope, recompute)
        finally:
            current_label = None
        if pending:
            per_call = []
            for buffers in pending:
                issued = sum(size for name, size in buffers.values())
                retained = sum(
                    size for ref, (name, size) in buffers.items() if not ref.expired()
                )
                assert issued > 0
                per_call.append((issued, retained))
            visits[phase, label] = per_call
            pending.clear()
        return output

    monkeypatch.setattr(oracle, "_models", models)
    monkeypatch.setattr(
        oracle,
        "in_forward_replay",
        lambda: in_forward_replay() or torch._C._current_graph_task_id() >= 0,
    )
    monkeypatch.setattr(functional, "get_workspace", workspace)
    monkeypatch.setattr(functional, "build_schedule", schedule)
    monkeypatch.setattr(functional, "forward", native)
    monkeypatch.setattr(GLMRingStack, "_ring_local_layers", visit)
    if mxfp8:
        checked = _check_mxfp8_native_backward(monkeypatch, functional)
        for enabled in (False, True):
            with moe.parallel(2):
                chunks = models(("CCCC", "CCCC"))
                stage = ps.get_pipeline_model_parallel_rank()
                offsets = [
                    sum(map(len, moe.CHUNKS[: vp * 2 + stage])) for vp in range(3)
                ]
                batches = oracle.layout._batches(moe.sample_rows(), 2)
                oracle._run(chunks, offsets, batches, "CCCC")
                assert checked == Counter({(enabled, 1): 1, (enabled, 2): 1})
                checked.clear()
                del chunks
    else:
        # BF16 also checks the entire graph against the original per-parameter gradient tolerance.
        oracle._compare_plans(("CCCC", "CCCC"), ("CCCC", "CCCC"))
    labels = {label for mode, label in visits if not mode}
    assert labels and labels == {label for mode, label in visits if mode}
    bytes_by_case = Counter()
    total = 0
    for label in labels:
        for baseline, candidate in zip(
            visits[False, label], visits[True, label], strict=True
        ):
            allocated, retained = baseline
            assert allocated == candidate[0] and candidate[1] == 0 < retained, (
                label,
                baseline,
                candidate,
            )
            bytes_by_case[(retained, candidate[1])] += 1
            total += 1
    print(
        f"real MoK group-8: {total} original checkpointed MoE calls, unique retained (baseline, candidate, count) "
        f"{sorted((a, b, n) for (a, b), n in bytes_by_case.items())}"
    )
