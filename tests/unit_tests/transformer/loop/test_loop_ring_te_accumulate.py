# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The loop ring with TE's fused weight-gradient accumulation (GPU).

A ring core sublayer runs once per round: R calls per microbatch, each with its own backward. With
``gradient_accumulation_fusion`` TE writes each call's weight gradient straight into
``main_grad``; it must add every call's, never overwrite (compat-doc item B1: TE overwrites when
told ``is_first_microbatch=True``). Runs under torchrun with CUDA at W = 2, 4 or 8 ranks, on
``ring_gpu_utils``' tiny DSv4 loop at P = 2 (k = 1 and k = 2, no recompute, EP 1), one step
without ``finalize_model_grads`` (so ``main_grad`` is this rank's raw sum), in deterministic mode:

* **fusion off** (the reference): a hook on every core TE weight records the gradient of every
  backward call; there are R x M per weight (R rounds of M microbatches; a routed expert's weight
  may miss a call that routed it no token), and their sum in arrival order is the run's
  ``main_grad`` (bitwise: DDP's hook adds each call's gradient in that order);
* **fusion on**, from the same weights and batches: TE accumulated every core weight
  (``grad_added_to_main_grad``), and ``main_grad`` equals the fusion-off sum of the per-call
  gradients: relative difference <= 1e-6 in fp32 (the GEMM accumulates into ``main_grad`` instead
  of a separate add), and at least 10x below the share of the smallest non-zero call, so a lost or
  overwritten round could not pass; bf16 (only if the fp32 run raises): <= 1e-2, the shares
  printed;
* ``_resolve_is_first_microbatch``: every core TE module of a ring chunk is marked
  ``is_repeated_layer`` and resolves to None (accumulate) also under a config stub that tracks
  ``is_first_microbatch`` (FP8); a prelude / coda TE module under the same stub resolves to its
  flag (the stub does track).
"""

import types
from unittest.mock import patch

import pytest
import torch

from tests.unit_tests.transformer.loop.hybrid_loop_utils import determinism_env
from tests.unit_tests.transformer.loop.ring_gpu_utils import (
    CUDA,
    ROUNDS,
    WORLD,
    Layout,
    build_chunks,
    chunk_offset,
    core_te_modules,
    deterministic_algorithms,
    edge_te_modules,
    full_precision_matmuls,
    gather,
    global_name,
    load_state,
    parallel_layout,
    rank_batches,
    run_step,
    run_with_precision,
    state_snapshot,
    tiny_config,
    wrap_ddp,
)

pytestmark = pytest.mark.skipif(
    not CUDA or WORLD < 2 or WORLD % 2, reason="the ring needs CUDA and an even number of ranks"
)

CHUNKS = (1, 2)


def _core_weights(models):
    """``{global name: parameter}`` of every core TE module's own parameters."""
    weights = {}
    for model in models:
        offset = chunk_offset(model)
        for module_name, module in core_te_modules(model):
            for name, param in module.named_parameters(recurse=False):
                weights[global_name(f"{module_name}.{name}", offset)] = param
    return weights


def _one(dtype, chunks):
    """Fusion off with per-call gradient hooks, then fusion on, at P = 2 with ``chunks`` core
    slices per stage; returns this rank's comparisons and the is_first_microbatch checks."""
    layout_off = Layout(stages=2, chunks=chunks, fusion=False)
    layout_on = Layout(stages=2, chunks=chunks, fusion=True)
    with parallel_layout(layout_on):
        batches = rank_batches()
        microbatches = len(batches)

        config = tiny_config(dtype, layout_off)
        models = build_chunks(config, layout_off)
        state = state_snapshot(models)
        ddp = wrap_ddp(models, config)
        config.finalize_model_grads_func = None  # main_grad = this rank's raw sum
        calls = {}
        handles = []
        for key, param in _core_weights(models).items():

            def hook(grad, key=key):
                calls.setdefault(key, []).append(grad.detach().float().clone())

            handles.append(param.register_hook(hook))
        off = run_step(ddp, layout_off, batches)
        for handle in handles:
            handle.remove()
        del ddp, models

        config = tiny_config(dtype, layout_on)
        models = build_chunks(config, layout_on)
        load_state(models, state)
        ddp = wrap_ddp(models, config)
        config.finalize_model_grads_func = None
        on = run_step(ddp, layout_on, batches)
        weights = sorted(_core_weights(models))

        rows = {}
        for key in weights:
            contributions = calls.get(key, [])
            total = torch.zeros_like(off.grads[key])
            for grad in contributions:
                total += grad
            total_norm = total.double().norm().item()
            shares = [g.double().norm().item() / total_norm for g in contributions if total_norm]
            nonzero = [share for share in shares if share > 0]
            diff = (on.grads[key].double() - total.double()).norm().item()
            rows[key] = dict(
                calls=len(contributions),
                sum_is_main_grad=bool(torch.equal(total, off.grads[key])),
                fused=key in on.fused,
                relative=diff / total_norm if total_norm else (0.0 if diff == 0 else float("inf")),
                min_share=min(nonzero) if nonzero else 0.0,
                bitwise=bool(torch.equal(on.grads[key], total)),
            )

        stub = types.SimpleNamespace(fp8="hybrid", fp4=None)
        # pylint: disable=import-outside-toplevel
        from megatron.core.extensions.transformer_engine import _resolve_is_first_microbatch

        core = []
        for model in models:
            for name, module in core_te_modules(model):
                with patch.object(module, "config", stub):
                    resolved = _resolve_is_first_microbatch(module)
                core.append((name, bool(getattr(module, "is_repeated_layer", False)), resolved))
        edge = []
        for model in models:
            for name, module in edge_te_modules(model):
                with patch.object(module, "config", stub):
                    resolved = _resolve_is_first_microbatch(module)
                edge.append((name, bool(getattr(module, "is_repeated_layer", False)), resolved))
        del ddp, models
    return dict(rows=rows, microbatches=microbatches, core=core, edge=edge)


@pytest.fixture(scope="module")
def accumulate():
    """Both runs for k = 1 and k = 2 (fp32 unless the fp32 run raises)."""

    def compute(dtype):
        return {chunks: _one(dtype, chunks) for chunks in CHUNKS}

    with determinism_env(), deterministic_algorithms(), full_precision_matmuls():
        dtype, probe, results = run_with_precision(compute)
    yield dict(dtype=dtype, probe=probe, results=results)


@pytest.mark.parametrize("chunks", CHUNKS)
def test_fused_main_grad_is_the_sum_of_every_round(accumulate, chunks):
    """Fusion on: every core TE weight's main_grad is the sum of its R x M per-call gradients."""
    dtype = accumulate["dtype"]
    result = accumulate["results"][chunks]
    rows = result["rows"]
    assert rows, "no core TE weight on this rank"
    expected_calls = ROUNDS * result["microbatches"]
    tolerance = 1e-6 if dtype == torch.float32 else 1e-2
    failures = []
    for key, row in rows.items():
        problems = []
        # A routed expert's weight may get no gradient from a call that routed it no token (TE
        # may skip an empty group): between 1 and R x M calls; every other weight exactly R x M.
        expert = ".experts." in key
        if (not expert and row["calls"] != expected_calls) or not (
            1 <= row["calls"] <= expected_calls
        ):
            problems.append(f"{row['calls']} calls, {expected_calls} expected")
        if not row["sum_is_main_grad"]:
            problems.append("fusion-off main_grad is not the sum of its calls")
        if not row["fused"]:
            problems.append("TE did not accumulate (grad_added_to_main_grad unset)")
        if row["relative"] > tolerance:
            problems.append(f"fusion on vs the sum: {row['relative']:.3e}")
        if dtype == torch.float32 and row["min_share"] and row["relative"] > 0.1 * row["min_share"]:
            problems.append(
                f"agreement {row['relative']:.3e} not below 0.1 x {row['min_share']:.3e}"
            )
        if problems:
            failures.append((key, problems))
    worst = max(row["relative"] for row in rows.values())
    shares = [row["min_share"] for row in rows.values() if row["min_share"]]
    summary = (
        f"loop ring TE accumulate k={chunks} ({dtype}): {len(rows)} core TE weights, "
        f"{expected_calls} calls each, max |on - sum| / |sum| {worst:.3e}, "
        f"bitwise {sum(row['bitwise'] for row in rows.values())}, "
        f"smallest call share {min(shares) if shares else 0.0:.3e}"
    )
    print(summary)
    gathered = gather(failures)
    assert not any(gathered), [item for rank in gathered for item in rank][:10]


@pytest.mark.parametrize("chunks", CHUNKS)
def test_core_modules_never_take_the_first_microbatch_flag(accumulate, chunks):
    """Core TE modules are repeated layers: None under a tracking config; prelude / coda TE
    modules get their flag under the same config."""
    result = accumulate["results"][chunks]
    assert result["core"], "no core TE module on this rank"
    for name, repeated, resolved in result["core"]:
        assert repeated and resolved is None, (name, repeated, resolved)
    for name, repeated, resolved in result["edge"]:
        assert not repeated and isinstance(resolved, bool), (name, repeated, resolved)
    print(
        f"loop ring is_first_microbatch k={chunks}: {len(result['core'])} core TE modules -> "
        f"None; {len(result['edge'])} prelude / coda TE modules -> their flag"
    )
