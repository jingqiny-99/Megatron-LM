# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Same-call MXFP8 VJP on a ring checkpoint's original and restored native MoK records."""

import json
import os
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from megatron.core.fp8_utils import get_fp8_context
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.forward_replay import in_forward_replay
from megatron.core.transformer.loop_transformer_block import loop_round_checkpoint
from megatron.core.transformer.moe.megakernel.mok.runtime import _gate_up_main_grad_arguments
from tests.unit_tests.transformer.loop import glm_ring_mok_utils as moe


def _snapshot(record):
    def copy(value):
        if isinstance(value, tuple):
            return tuple(copy(item) for item in value)
        return value.detach().clone()

    return type(record)(**{f.name: copy(getattr(record, f.name)) for f in fields(record)})


def _scratch(module):
    """Own both the target tensors and the indirect pointer tables until the kernel finishes."""
    from mok import ops

    params = [
        module.shared_fc1_weight,
        module.shared_fc2_weight,
        *module.routed_fc1_parameters,
        *module.routed_fc2_parameters,
    ]
    buffers = [torch.zeros(p.shape, device=p.device, dtype=torch.float32) for p in params]
    shared1, shared2 = buffers[:2]
    n = len(module.routed_fc1_parameters)
    routed1, routed2 = buffers[2 : 2 + n], buffers[2 + n :]
    tables = (
        None
        if module.native_single_grouped_weights
        else (
            ops.make_routed_d_weight_storage_table(routed1),
            ops.make_routed_d_weight_storage_table(routed2),
        )
    )
    arguments = _gate_up_main_grad_arguments(
        (shared1, routed1[0], shared2, routed2[0]), tables, module.intermediate_size
    )
    names = [
        "shared_fc1",
        "shared_fc2",
        *[f"routed_fc1_{i}" for i in range(n)],
        *[f"routed_fc2_{i}" for i in range(len(routed2))],
    ]
    return arguments, dict(zip(names, buffers, strict=True))


def _error(actual, baseline):
    actual, baseline = actual.float(), baseline.float()
    delta = (actual - baseline).norm().item()
    norm = baseline.norm().item()
    return dict(
        rel_l2=delta / max(norm, 1e-30),
        delta_l2=delta,
        ref_l2=norm,
        max_abs=(actual - baseline).abs().max().item(),
        native_bound=0.02 * norm + 2e-8 * baseline.numel() ** 0.5,
        bitwise=torch.equal(actual, baseline),
    )


def _raw_counts(left, right, prefix=""):
    if isinstance(left, tuple):
        result = {}
        for i, (x, y) in enumerate(zip(left, right, strict=True)):
            result.update(_raw_counts(x, y, f"{prefix}.{i}"))
        return result
    if isinstance(left, torch.Tensor):
        x, y = (v.contiguous().view(torch.uint8) for v in (left, right))
        return {
            prefix: {
                "shape": list(left.shape),
                "dtype": str(left.dtype),
                "bytes": x.numel(),
                "different_bytes": (x != y).sum().item(),
            }
        }
    result = {}
    for f in fields(left):
        result.update(_raw_counts(getattr(left, f.name), getattr(right, f.name), f.name))
    return result


@pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.getenv("WORLD_SIZE", "1")) != 8,
    reason="requires two independent EP4 groups on eight GB300s",
)
def test_mxfp8_native_vjp_against_same_call_original_context(monkeypatch):
    from mok import functional

    with moe.parallel(2):
        cfg = replace(
            moe.config(2),
            fp8="e4m3",
            fp8_recipe="mxfp8",
            fp8_param=True,
            disable_parameter_transpose_cache=True,
            moe_mok_checkpoint_forward_context=True,
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
            dsa_indexer_weights_proj_use_quantization=False,
            dsa_indexer_weights_proj_output_dtype="bf16",
            linear_cp_mode=None,
            linear_cp_layout=None,
        )
        with patch.object(moe, "config", return_value=cfg), get_fp8_context(cfg, is_init=True):
            chunks = moe.models(2)
        module = next(
            m.megakernel_experts
            for m in chunks[1].modules()
            if getattr(m, "megakernel_experts", None) is not None
        )
        assert module.use_mxfp8_weights and module.checkpoint_forward_context
        assert module.virtual_layout is None and module.shared_output_gate_weight is None
        params = list(module.autograd_routed_parameters) + [
            module.shared_fc1_weight,
            module.shared_fc2_weight,
        ]
        for parameter in params:
            parameter.main_grad = torch.zeros(
                parameter.shape, dtype=torch.float32, device=parameter.device
            )
        rank = dist.get_rank()
        torch.manual_seed(719 + rank)
        rows, k, experts = moe.SEQUENCE, moe.TOPK, moe.EXPERTS
        x = torch.randn(
            rows, cfg.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        # Each EP group assigns exactly 1,024 rows per expert, eliminating inter-expert padding.
        ids = (
            (
                torch.arange(rows, device=x.device)[:, None]
                + 3 * rank
                + torch.arange(k, device=x.device)[None, :]
            )
            .remainder(experts)
            .int()
        )
        probs = torch.rand(rows, k, device=x.device).softmax(dim=-1).detach().requires_grad_()
        dy = torch.randn_like(x)
        forward, backward = functional.forward, functional.backward
        records, measures = {}, {}

        def capture(config, workspace, schedule, local_x, weights, *native_weights, **opts):
            out, ctx = forward(
                config, workspace, schedule, local_x, weights, *native_weights, **opts
            )
            mode = "replay" if in_forward_replay() else "original"
            assert mode not in records
            records[mode] = (_snapshot(schedule), _snapshot(ctx), out.detach().clone())
            assert (
                torch.equal(local_x, x)
                and torch.equal(weights, probs)
                and torch.equal(schedule.top_experts, ids)
            )
            return out, ctx

        def compare(
            config, workspace, real_schedule, real_ctx, adjoint, local_x, weights, *args, **kwargs
        ):
            assert set(records) == {"original", "replay"} and torch.equal(adjoint, dy)
            orig, replay = records["original"], records["replay"]
            count = int(orig[0].num_tokens.item())
            real_count = rows * k
            assert count == real_count and real_count <= workspace.schedule_capacity
            assert torch.equal(orig[0].tokens_per_expert, replay[0].tokens_per_expert)
            equal_positions = all(
                torch.equal(getattr(orig[0], name)[:count], getattr(replay[0], name)[:count])
                for name in ("peer_rank", "peer_token_idx")
            )
            orig_pairs = (
                orig[0].peer_rank[:count].long() * (rows * k)
                + orig[0].peer_token_idx[:count].long()
            )
            repl_pairs = (
                replay[0].peer_rank[:count].long() * (rows * k)
                + replay[0].peer_token_idx[:count].long()
            )
            assert torch.equal(orig_pairs.sort().values, repl_pairs.sort().values)
            measures["schedule"] = {
                "active_rows": count,
                "capacity": workspace.schedule_capacity,
                "tokens_per_expert": orig[0].tokens_per_expert.tolist(),
                "active_physical_order_equal": equal_positions,
                "raw_including_unused": _raw_counts(orig[0], replay[0]),
            }
            measures["forward"] = _error(replay[2], orig[2])
            for name in ("x_routed", "gate_routed", "up_routed", "hidden_routed"):
                assert (
                    isinstance(getattr(orig[1], name), tuple) and len(getattr(orig[1], name)) == 2
                )
            measures["context_including_unused"] = _raw_counts(orig[1], replay[1])
            # All experts receive the same multiple of 128 rows. No internal expert padding
            # is part of this prefix. The transposed inputs/hidden use column tile indices.
            per_expert = torch.full_like(
                orig[0].tokens_per_expert, real_count // module.num_local_experts
            )
            assert torch.equal(orig[0].tokens_per_expert, per_expert)
            active = {}
            for field in ("x_routed", "gate_routed", "up_routed", "hidden_routed"):
                axis = 1 if field in ("x_routed", "hidden_routed") else 0
                for index, (old, new) in enumerate(
                    zip(getattr(orig[1], field), getattr(replay[1], field), strict=True)
                ):
                    width = count if index == 0 else count // 128
                    if index:
                        assert old.ndim == 4 and tuple(old.shape[-2:]) == (32, 16)
                    else:
                        assert old.ndim == 2
                    active.update(
                        _raw_counts(
                            old.narrow(axis, 0, width),
                            new.narrow(axis, 0, width),
                            field + (".scale" if index else ".payload"),
                        )
                    )
            measures["active_context"] = active
            # Check that the actual unpacked backward record is the saved replay record.
            assert all(v["different_bytes"] == 0 for v in _raw_counts(real_ctx, replay[1]).values())
            assert all(
                v["different_bytes"] == 0 for v in _raw_counts(real_schedule, replay[0]).values()
            )
            calls = []
            for key in ("original", "replay", "original", "replay", "original", "replay"):
                schedule, context, _ = records[key]
                (mg, tables), scratch = _scratch(module)
                result = backward(
                    config,
                    workspace,
                    schedule,
                    context,
                    adjoint,
                    local_x,
                    weights,
                    *args,
                    **{**kwargs, "main_grads": mg, "main_grad_storage_tables": tables},
                )
                scratch.update(dX=result[0], dRoutes=result[1])
                calls.append({name: val.detach().cpu().clone() for name, val in scratch.items()})
            measures["vjp"] = {}
            for i in range(1, len(calls)):
                measures["vjp"][f"{i}-{'replay' if i % 2 else 'original'}_vs_original"] = {
                    name: _error(v, calls[0][name]) for name, v in calls[i].items()
                }
            # Only the normal training call writes to the real main gradients.
            assert all(torch.count_nonzero(p.main_grad).item() == 0 for p in params)
            return backward(
                config,
                workspace,
                real_schedule,
                real_ctx,
                adjoint,
                local_x,
                weights,
                *args,
                **kwargs,
            )

        monkeypatch.setattr(functional, "forward", capture)
        monkeypatch.setattr(functional, "backward", compare)
        output = loop_round_checkpoint(
            lambda h, p: module._apply_compact(h, p, ids),
            x,
            probs,
            label="direct_mok_vjp",
            round_ends_in_layer=True,
        )
        torch.autograd.grad(output, (x, probs), grad_outputs=dy)
        assert len(measures["vjp"]) == 5
        if directory := os.environ.get("MOK_VJP_RESULTS"):
            Path(directory, f"rank{rank}.json").write_text(json.dumps(measures, indent=2) + "\n")
        failures = [
            f"{call}/{name}"
            for call, row in measures["vjp"].items()
            for name, m in row.items()
            if not (0 < m["ref_l2"] and m["delta_l2"] <= m["native_bound"])
        ]
        assert not failures, failures
        assert all(
            field["different_bytes"] == 0 for field in measures["active_context"].values()
        ), measures["active_context"]
