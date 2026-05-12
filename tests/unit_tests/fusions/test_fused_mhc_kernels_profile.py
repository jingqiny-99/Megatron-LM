# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in full mHC fwd/bwd profile test.

This test intentionally runs the public fused mHC path at the large target
shape, primes autotune caches first, then optionally brackets only measured
iterations with cudaProfilerStart/Stop for nsys capture.
"""

import os
from contextlib import contextmanager

import pytest


if os.getenv("MHC_RUN_MHC_PROFILE") != "1" and os.getenv("MHC_RUN_BENCHMARKS") != "1":
    pytest.skip(
        "mHC profile UT is opt-in; set MHC_RUN_MHC_PROFILE=1 to run",
        allow_module_level=True,
    )

import torch
from torch import Tensor

from megatron.core.fusions import fused_mhc_kernels as mhc_kernels
from megatron.core.fusions.fused_mhc_kernels import (
    fused_h_aggregate,
    fused_h_post_bda,
    fused_proj_rms_compute_h,
    fused_sinkhorn,
    is_cutile_available,
    is_triton_available,
)


DTYPE = torch.bfloat16
DEVICE = "cuda"
EPS = 1e-6
RAND_LO, RAND_HI = -0.1, 0.1


def _env_flag(name: str) -> bool:
    return os.getenv(name, "0").lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


@contextmanager
def _nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


@contextmanager
def _nsys_capture(name: str):
    with _nvtx_range(name):
        if _env_flag("MHC_NSYS_CAPTURE"):
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
        try:
            yield
        finally:
            if _env_flag("MHC_NSYS_CAPTURE"):
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()


def _rand(*shape, **kwargs) -> Tensor:
    return torch.empty(*shape, dtype=DTYPE, device=DEVICE, **kwargs).uniform_(RAND_LO, RAND_HI)


def _clear_grads(tensors) -> None:
    for tensor in tensors:
        tensor.grad = None


def _print_cutile_autotune_caches() -> None:
    cache_names = (
        "_proj_rms_fwd_best_cfg",
        "_reduce_compute_h_best_cfg",
        "_fused_grad_x_weight_best_cfg",
    )
    for name in cache_names:
        print(f"\n  [mHC profile] {name}: {getattr(mhc_kernels, name, {})}")


def _make_inputs(s: int, b: int, n: int, c: int):
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    out_features = n * n + 2 * n
    hidden_states = _rand(s, b, n * c).requires_grad_(True)
    weight = _rand(out_features, n * c).requires_grad_(True)
    alpha_pre = _rand(1).requires_grad_(True)
    alpha_post = _rand(1).requires_grad_(True)
    alpha_res = _rand(1).requires_grad_(True)
    compute_h_bias = _rand(out_features).requires_grad_(True)
    layer_out = _rand(s, b, c).requires_grad_(True)
    layer_bias = _rand(c).requires_grad_(True)

    grad_tensors = (
        hidden_states,
        weight,
        alpha_pre,
        alpha_post,
        alpha_res,
        compute_h_bias,
        layer_out,
        layer_bias,
    )
    return grad_tensors


def _run_mhc_step(grad_tensors, s: int, b: int, n: int, c: int, sinkhorn_iters: int, tag: str):
    (
        hidden_states,
        weight,
        alpha_pre,
        alpha_post,
        alpha_res,
        compute_h_bias,
        layer_out,
        layer_bias,
    ) = grad_tensors
    _clear_grads(grad_tensors)

    hidden_2d = hidden_states.reshape(s * b, n * c)
    hidden_n = hidden_states.view(s, b, n, c)

    with _nvtx_range(f"{tag}:proj_rms_compute_h"):
        h_pre, h_post, h_res_logits, _ = fused_proj_rms_compute_h(
            hidden_2d,
            weight,
            alpha_pre,
            alpha_post,
            alpha_res,
            compute_h_bias,
            n,
            EPS,
        )

    with _nvtx_range(f"{tag}:sinkhorn"):
        h_res = fused_sinkhorn(h_res_logits.view(s, b, n, n), sinkhorn_iters, EPS)

    with _nvtx_range(f"{tag}:h_aggregate"):
        aggregated = fused_h_aggregate(hidden_n, h_pre.view(s, b, n))

    with _nvtx_range(f"{tag}:h_post_bda"):
        output = fused_h_post_bda(
            h_res,
            hidden_n,
            h_post.view(s, b, n),
            layer_out,
            layer_bias,
        )

    with _nvtx_range(f"{tag}:backward"):
        loss = output.sum() + aggregated.sum()
        loss.backward()

    return loss.detach()


def test_mhc_full_fwd_bwd_autotuned_profile():
    """Run one full public mHC fwd/bwd step after autotune cache warmup."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 not supported")
    if not is_cutile_available():
        pytest.skip("cuTile not installed; profile target requires public cuTile kernels")
    if not is_triton_available():
        pytest.skip("Triton not installed; profile target is the current public mixed path")
    if not getattr(mhc_kernels, "_CUTILE_EXPERIMENTAL_AVAILABLE", False):
        pytest.skip("cuda.tile_experimental not installed; cuTile autotune is unavailable")

    s = _env_int("MHC_PROFILE_S", 4096)
    b = _env_int("MHC_PROFILE_B", 1)
    n = _env_int("MHC_PROFILE_N", 4)
    c = _env_int("MHC_PROFILE_C", 7168)
    sinkhorn_iters = _env_int("MHC_PROFILE_SINKHORN_ITERS", 5)
    warmup = _env_int("MHC_PROFILE_WARMUP", 1)
    reps = _env_int("MHC_PROFILE_REPS", 1)

    grad_tensors = _make_inputs(s, b, n, c)

    with _nvtx_range("mhc_full_fwd_bwd:autotune"):
        loss = _run_mhc_step(grad_tensors, s, b, n, c, sinkhorn_iters, "mhc_autotune")
        torch.cuda.synchronize()
    _print_cutile_autotune_caches()
    assert torch.isfinite(loss).item()
    for tensor in grad_tensors:
        assert tensor.grad is not None

    with _nvtx_range("mhc_full_fwd_bwd:warmup"):
        for idx in range(warmup):
            _run_mhc_step(grad_tensors, s, b, n, c, sinkhorn_iters, f"mhc_warmup_{idx}")
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with _nsys_capture("mhc_full_fwd_bwd:measure_autotuned"):
        start.record()
        for idx in range(reps):
            _run_mhc_step(grad_tensors, s, b, n, c, sinkhorn_iters, f"mhc_measure_{idx}")
        end.record()
        end.synchronize()

    ms = start.elapsed_time(end) / reps
    print(
        f"\n  [mHC profile] full public fwd+bwd: {ms:.3f} ms/iter, "
        f"shape=(s={s}, b={b}, n={n}, C={c}), warmup={warmup}, reps={reps}"
    )
    assert ms > 0
