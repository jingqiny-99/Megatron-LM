# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Fused H_pre Aggregate kernel for mHC.

This module provides fused GPU kernels for the H_pre aggregation operation
in mHC (Manifold-Constrained Hyper-Connections).

The computation is:
    aggregated[i, c] = sum_j(h_pre[i, j] * x[i, j, c])

where i is the flattened (s*b) batch dimension, j is the stream index,
and c is the hidden dimension.

Gradient formulas:
    grad_x[i, j, c] = grad_output[i, c] * h_pre[i, j]
    grad_h_pre[i, j] = sum_c(grad_output[i, c] * x[i, j, c])
"""

import torch
from torch import Tensor
from typing import Tuple
import math
# from megatron.core.fusions.fused_sinkhorn import is_tilelang_available
import tilelang
import tilelang.language as T
_TILELANG_AVAILABLE = True

# try:
#     import tilelang
#     import tilelang.language as T
#     _TILELANG_AVAILABLE = True
# except ImportError:
#     pass


if _TILELANG_AVAILABLE:
    tilelang.set_log_level("WARNING")

    FP32 = "float32"

    _TORCH_DTYPE_TO_TL = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }

    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }

    # ==================== Forward Kernel ====================

    # Input shapes (flattened s*b):
    #   - x: [sb, n, C] - n-stream hidden states
    #   - h_pre: [sb, n] - aggregation weights
    # Output shapes:
    #   - output: [sb, C] - aggregated hidden states
    #
    # Computation:
    #   output[i, c] = sum_j(h_pre[i, j] * x[i, j, c])

    @tilelang.jit(pass_configs=pass_configs)
    def _h_aggregate_forward_kernel_generator(sb: int, n: int, C: int, dtype: str = FP32):
        """Generate forward H_pre aggregate kernel."""
        EXPECTED_BLOCK_C = 1024
        BLOCK_C = math.gcd(EXPECTED_BLOCK_C, C)
        threads = 128

        @T.prim_func
        def h_aggregate_forward_kernel_(
            x: T.Tensor[(sb, n, C), dtype],
            h_pre: T.Tensor[(sb, n), dtype],
            output: T.Tensor[(sb, C), dtype],
        ):
            with T.Kernel(sb, threads=threads) as i:
                x_local = T.alloc_fragment((n, BLOCK_C), dtype)
                h_pre_local = T.alloc_fragment((n,), dtype)
                output_local = T.alloc_fragment((BLOCK_C,), dtype)
                temp = T.alloc_fragment((n, BLOCK_C), dtype)
                T.copy(h_pre[i, 0], h_pre_local)
                for pid_c in T.Pipelined(C // BLOCK_C, num_stages=3):
                    c_idx = pid_c * BLOCK_C
                    T.copy(x[i, 0, c_idx], x_local)
                    T.clear(output_local)

                    for j, k in T.Parallel(n, BLOCK_C):
                        temp[j, k] = h_pre_local[j] * x_local[j, k]

                    T.reduce_sum(temp, output_local, dim=0, clear=False)
                    T.copy(output_local, output[i, c_idx])

        return h_aggregate_forward_kernel_

    # ==================== Backward Kernel ====================

    # Input shapes (flattened s*b):
    #   - grad_output: [sb, C] - gradient w.r.t. output
    #   - x: [sb, n, C] - n-stream hidden states
    #   - h_pre: [sb, n] - aggregation weights
    # Output shapes:
    #   - grad_x: [sb, n, C] - gradient w.r.t. x
    #   - grad_h_pre: [sb, n] - gradient w.r.t. h_pre
    #
    # Gradient formulas:
    #   grad_x[i, j, c] = grad_output[i, c] * h_pre[i, j]
    #   grad_h_pre[i, j] = sum_c(grad_output[i, c] * x[i, j, c])

    @tilelang.jit(pass_configs=pass_configs)
    def _h_aggregate_backward_kernel_generator(sb: int, n: int, C: int, dtype: str = FP32):
        """Generate backward H_pre aggregate kernel."""
        EXPECTED_BLOCK_C = 1024
        BLOCK_C = math.gcd(EXPECTED_BLOCK_C, C)
        threads = 128

        @T.prim_func
        def h_aggregate_backward_kernel_(
            grad_output: T.Tensor[(sb, C), dtype],
            x: T.Tensor[(sb, n, C), dtype],
            h_pre: T.Tensor[(sb, n), dtype],
            grad_x: T.Tensor[(sb, n, C), dtype],
            grad_h_pre: T.Tensor[(sb, n), dtype],
        ):
            with T.Kernel(sb, threads=threads) as i:
                grad_output_local = T.alloc_fragment((BLOCK_C,), dtype)
                x_local = T.alloc_fragment((n, BLOCK_C), dtype)
                h_pre_local = T.alloc_fragment((n,), dtype)
                grad_x_local = T.alloc_fragment((n, BLOCK_C), dtype)
                grad_h_pre_local = T.alloc_fragment((n,), dtype)
                temp_h_pre = T.alloc_fragment((n, BLOCK_C), dtype)
                T.copy(h_pre[i, 0], h_pre_local)

                T.clear(grad_h_pre_local)
                for pid_c in T.Pipelined(C // BLOCK_C, num_stages=3):
                    c_idx = pid_c * BLOCK_C

                    T.copy(grad_output[i, c_idx], grad_output_local)
                    T.copy(x[i, 0, c_idx], x_local)

                    T.clear(grad_x_local)
                    for j, k in T.Parallel(n, BLOCK_C):
                        grad_x_local[j, k] = grad_output_local[k] * h_pre_local[j]
                        temp_h_pre[j, k] = grad_output_local[k] * x_local[j, k] 

                    T.reduce_sum(temp_h_pre, grad_h_pre_local, dim=1, clear=False)
                    T.copy(grad_x_local, grad_x[i, 0, c_idx])
                T.copy(grad_h_pre_local, grad_h_pre[i, 0])
        return h_aggregate_backward_kernel_

    _FORWARD_KERNEL_CACHE = {}
    _BACKWARD_KERNEL_CACHE = {}

    def _get_forward_kernel(sb: int, n: int, C: int, dtype: str = FP32):
        """Get cached forward kernel or create a new one."""
        key = (sb, n, C, dtype)
        if key not in _FORWARD_KERNEL_CACHE:
            _FORWARD_KERNEL_CACHE[key] = _h_aggregate_forward_kernel_generator(sb, n, C, dtype)
        return _FORWARD_KERNEL_CACHE[key]

    def _get_backward_kernel(sb: int, n: int, C: int, dtype: str = FP32):
        """Get cached backward kernel or create a new one."""
        key = (sb, n, C, dtype)
        if key not in _BACKWARD_KERNEL_CACHE:
            _BACKWARD_KERNEL_CACHE[key] = _h_aggregate_backward_kernel_generator(sb, n, C, dtype)
        return _BACKWARD_KERNEL_CACHE[key]



def h_aggregate_tilelang_forward(
    x: Tensor,
    h_pre: Tensor,
) -> Tensor:
    """
    TileLang implementation of H_pre aggregate forward pass.

    Computes:
        output = sum_j(h_pre_j * x_j)  (weighted sum over n streams)

    Args:
        x: [s, b, n, C] - n-stream hidden states
        h_pre: [s, b, n] - aggregation weights

    Returns:
        output: [s, b, C] - aggregated hidden states
    """
    s, b, n, C = x.shape
    sb = s * b
    dtype_str = _TORCH_DTYPE_TO_TL[x.dtype]

    # Reshape inputs to flattened batch dimension
    x_flat = x.view(sb, n, C).contiguous()
    h_pre_flat = h_pre.view(sb, n).contiguous()

    # Allocate output in same dtype
    output_flat = torch.empty(sb, C, dtype=x.dtype, device=x.device)

    # Get cached kernel
    kernel = _get_forward_kernel(sb, n, C, dtype_str)

    # Launch kernel
    kernel(x_flat, h_pre_flat, output_flat)

    # Reshape output back
    return output_flat.view(s, b, C)


def h_aggregate_tilelang_backward(
    grad_output: Tensor,
    x: Tensor,
    h_pre: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    TileLang implementation of H_pre aggregate backward pass.

    Computes gradients:
        grad_x[i, j, c] = grad_output[i, c] * h_pre[i, j]
        grad_h_pre[i, j] = sum_c(grad_output[i, c] * x[i, j, c])

    Args:
        grad_output: [s, b, C] - gradient w.r.t. output
        x: [s, b, n, C] - n-stream hidden states
        h_pre: [s, b, n] - aggregation weights

    Returns:
        grad_x: [s, b, n, C] - gradient w.r.t. x
        grad_h_pre: [s, b, n] - gradient w.r.t. h_pre
    """
    s, b, n, C = x.shape
    sb = s * b
    dtype_str = _TORCH_DTYPE_TO_TL[x.dtype]

    # Reshape inputs to flattened batch dimension
    grad_output_flat = grad_output.view(sb, C).contiguous()
    x_flat = x.view(sb, n, C).contiguous()
    h_pre_flat = h_pre.view(sb, n).contiguous()

    # Allocate outputs in same dtype
    grad_x_flat = torch.empty(sb, n, C, dtype=x.dtype, device=x.device)
    grad_h_pre_flat = torch.empty(sb, n, dtype=x.dtype, device=x.device)

    # Get cached kernel
    kernel = _get_backward_kernel(sb, n, C, dtype_str)

    # Launch kernel
    kernel(grad_output_flat, x_flat, h_pre_flat, grad_x_flat, grad_h_pre_flat)

    # Reshape outputs back
    return grad_x_flat.view(s, b, n, C), grad_h_pre_flat.view(s, b, n)


def h_aggregate_native_forward(
    x: Tensor,
    h_pre: Tensor,
) -> Tensor:
    """
    Native PyTorch implementation of H_pre aggregate forward pass.

    This is the reference implementation for testing and fallback.

    Args:
        x: [s, b, n, C] - n-stream hidden states
        h_pre: [s, b, n] - aggregation weights

    Returns:
        output: [s, b, C] - aggregated hidden states
    """
    # Weighted sum: [s, b, n, C] * [s, b, n, 1] -> sum over n -> [s, b, C]
    aggregated = (x * h_pre.unsqueeze(-1)).sum(dim=2)
    return aggregated


def h_aggregate_native_backward(
    grad_output: Tensor,
    x: Tensor,
    h_pre: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Native PyTorch implementation of H_pre aggregate backward pass.

    Uses autograd for gradient computation via recomputation.

    Args:
        grad_output: [s, b, C] - gradient w.r.t. output
        x: [s, b, n, C] - n-stream hidden states
        h_pre: [s, b, n] - aggregation weights

    Returns:
        grad_x: [s, b, n, C] - gradient w.r.t. x
        grad_h_pre: [s, b, n] - gradient w.r.t. h_pre
    """
    with torch.enable_grad():
        x_input = x.detach().requires_grad_(True)
        h_pre_input = h_pre.detach().requires_grad_(True)

        # Recompute forward
        output = (x_input * h_pre_input.unsqueeze(-1)).sum(dim=2)

        # Compute gradients
        grad_x, grad_h_pre = torch.autograd.grad(
            outputs=output,
            inputs=[x_input, h_pre_input],
            grad_outputs=grad_output,
            create_graph=False,
            retain_graph=False,
        )

    return grad_x, grad_h_pre


def _benchmark_native(
    s: int,
    b: int,
    n: int,
    C: int,
    dtype: torch.dtype = torch.float32,
    warmup_iters: int = 20,
    bench_iters: int = 100,
):
    """
    Benchmark native H_pre aggregate forward + backward using CUDA events.

    Args:
        s: sequence length
        b: batch size
        n: number of hyper-connection streams
        C: hidden dimension
        dtype: data type
        warmup_iters: number of warmup iterations
        bench_iters: number of benchmark iterations
    """
    device = "cuda"
    torch.manual_seed(42)

    # --- Allocate tensors ---
    x = torch.randn(s, b, n, C, dtype=dtype, device=device, requires_grad=True)
    h_pre = torch.randn(s, b, n, dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn(s, b, C, dtype=dtype, device=device)

    # --- Helper: single fwd+bwd iteration ---
    def _run_once():
        for t in [x, h_pre]:
            if t.grad is not None:
                t.grad.zero_()

        output = h_aggregate_native_forward(x, h_pre)
        grads = h_aggregate_native_backward(grad_output, x, h_pre)
        return output, grads

    # --- Warmup ---
    for _ in range(warmup_iters):
        _run_once()
    torch.cuda.synchronize()

    # --- Benchmark with CUDA events ---
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]

    for i in range(bench_iters):
        start_events[i].record()
        _run_once()
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [s_evt.elapsed_time(e_evt) for s_evt, e_evt in zip(start_events, end_events)]
    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    return avg_ms, min_ms, max_ms


def _benchmark_tilelang_forward(
    s: int,
    b: int,
    n: int,
    C: int,
    dtype: torch.dtype = torch.float32,
    warmup_iters: int = 20,
    bench_iters: int = 100,
):
    """
    Benchmark TileLang H_pre aggregate forward using CUDA events.
    """
    device = "cuda"
    torch.manual_seed(42)

    x = torch.randn(s, b, n, C, dtype=dtype, device=device)
    h_pre = torch.randn(s, b, n, dtype=dtype, device=device)

    def _run_once():
        return h_aggregate_tilelang_forward(x, h_pre)

    # --- Warmup ---
    for _ in range(warmup_iters):
        _run_once()
    torch.cuda.synchronize()

    # --- Benchmark with CUDA events ---
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]

    for i in range(bench_iters):
        start_events[i].record()
        _run_once()
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [s_evt.elapsed_time(e_evt) for s_evt, e_evt in zip(start_events, end_events)]
    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    return avg_ms, min_ms, max_ms


def _benchmark_tilelang_backward(
    s: int,
    b: int,
    n: int,
    C: int,
    dtype: torch.dtype = torch.float32,
    warmup_iters: int = 20,
    bench_iters: int = 100,
):
    """
    Benchmark TileLang H_pre aggregate backward using CUDA events.
    """
    device = "cuda"
    torch.manual_seed(42)

    grad_output = torch.randn(s, b, C, dtype=dtype, device=device)
    x = torch.randn(s, b, n, C, dtype=dtype, device=device)
    h_pre = torch.randn(s, b, n, dtype=dtype, device=device)

    def _run_once():
        return h_aggregate_tilelang_backward(grad_output, x, h_pre)

    # --- Warmup ---
    for _ in range(warmup_iters):
        _run_once()
    torch.cuda.synchronize()

    # --- Benchmark with CUDA events ---
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]

    for i in range(bench_iters):
        start_events[i].record()
        _run_once()
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [s_evt.elapsed_time(e_evt) for s_evt, e_evt in zip(start_events, end_events)]
    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    return avg_ms, min_ms, max_ms


def _benchmark_tilelang_fwd_bwd(
    s: int,
    b: int,
    n: int,
    C: int,
    dtype: torch.dtype = torch.float32,
    warmup_iters: int = 20,
    bench_iters: int = 100,
):
    """
    Benchmark TileLang H_pre aggregate forward + backward using CUDA events.
    """
    device = "cuda"
    torch.manual_seed(42)

    x = torch.randn(s, b, n, C, dtype=dtype, device=device)
    h_pre = torch.randn(s, b, n, dtype=dtype, device=device)
    grad_output = torch.randn(s, b, C, dtype=dtype, device=device)

    def _run_once():
        output = h_aggregate_tilelang_forward(x, h_pre)
        grads = h_aggregate_tilelang_backward(grad_output, x, h_pre)
        return output, grads

    # --- Warmup ---
    for _ in range(warmup_iters):
        _run_once()
    torch.cuda.synchronize()

    # --- Benchmark with CUDA events ---
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]

    for i in range(bench_iters):
        start_events[i].record()
        _run_once()
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [s_evt.elapsed_time(e_evt) for s_evt, e_evt in zip(start_events, end_events)]
    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    return avg_ms, min_ms, max_ms


def _benchmark_native_forward(
    s: int,
    b: int,
    n: int,
    C: int,
    dtype: torch.dtype = torch.float32,
    warmup_iters: int = 20,
    bench_iters: int = 100,
):
    """
    Benchmark native H_pre aggregate forward only (for fair comparison with tilelang).
    """
    device = "cuda"
    torch.manual_seed(42)

    x = torch.randn(s, b, n, C, dtype=dtype, device=device)
    h_pre = torch.randn(s, b, n, dtype=dtype, device=device)

    def _run_once():
        return h_aggregate_native_forward(x, h_pre)

    # --- Warmup ---
    for _ in range(warmup_iters):
        _run_once()
    torch.cuda.synchronize()

    # --- Benchmark with CUDA events ---
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]

    for i in range(bench_iters):
        start_events[i].record()
        _run_once()
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [s_evt.elapsed_time(e_evt) for s_evt, e_evt in zip(start_events, end_events)]
    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    return avg_ms, min_ms, max_ms


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark H_pre Aggregate implementation")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "native", "tilelang", "compare"],
                        help="Benchmark mode: all, native, tilelang, or compare (forward only comparison)")
    args = parser.parse_args()

    print("=" * 70)
    print("H_pre Aggregate Benchmark")
    print("=" * 70)

    # ---- Correctness check: TileLang vs Native ----
    print("\n[Correctness Check] TileLang vs Native Forward")
    print("-" * 50)
    torch.manual_seed(42)
    s, b, n, C = 1, 64, 4, 128
    x = torch.randn(s, b, n, C, dtype=torch.float32).cuda()
    h_pre = torch.randn(s, b, n, dtype=torch.float32).cuda()

    # Native forward
    output_native = h_aggregate_native_forward(x, h_pre)
    print(f"  Native output shape: {output_native.shape}")

    # TileLang forward
    output_tilelang = h_aggregate_tilelang_forward(x, h_pre)
    print(f"  TileLang output shape: {output_tilelang.shape}")

    # Compare results
    max_diff = (output_native - output_tilelang).abs().max().item()
    mean_diff = (output_native - output_tilelang).abs().mean().item()
    print(f"  Max absolute diff: {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")

    if max_diff < 1e-4:
        print("  [PASS] TileLang Forward matches Native!")
    else:
        print("  [FAIL] TileLang Forward differs from Native!")
        print(f"  Native output sample:\n{output_native[0, 0, :8]}")
        print(f"  TileLang output sample:\n{output_tilelang[0, 0, :8]}")

    # ---- Correctness check: TileLang Backward vs Native Backward ----
    print("\n[Correctness Check] TileLang vs Native Backward")
    print("-" * 50)
    grad_output = torch.randn(s, b, C, dtype=torch.float32).cuda()

    # Native backward
    grads_native = h_aggregate_native_backward(grad_output, x, h_pre)
    print(f"  Native grad shapes: x={grads_native[0].shape}, h_pre={grads_native[1].shape}")

    # TileLang backward
    grads_tilelang = h_aggregate_tilelang_backward(grad_output, x, h_pre)
    print(f"  TileLang grad shapes: x={grads_tilelang[0].shape}, h_pre={grads_tilelang[1].shape}")

    # Compare each gradient
    grad_names = ["grad_x", "grad_h_pre"]
    all_pass = True
    for name, g_native, g_tilelang in zip(grad_names, grads_native, grads_tilelang):
        max_diff = (g_native - g_tilelang).abs().max().item()
        mean_diff = (g_native - g_tilelang).abs().mean().item()
        status = "[PASS]" if max_diff < 1e-3 else "[FAIL]"
        if max_diff >= 1e-3:
            all_pass = False
        print(f"  {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e} {status}")

    if all_pass:
        print("  [PASS] TileLang Backward matches Native!")
    else:
        print("  [FAIL] Some gradients differ!")

    # ---- Benchmark configs ----
    configs = [
        # Small (debugging)
        (64, 1, 4, 1024),
        (64, 1, 4, 2048),
        # Medium (e.g. GPT-like)
        (64, 1, 4, 4096),
        (128, 1, 4, 4096),
        # Larger hidden dim
        (64, 1, 4, 8192),
        (128, 1, 4, 8192),
        # Different stream counts
        (64, 1, 2, 4096),
        (64, 1, 8, 4096),
        # Larger batch
        (64, 2, 4, 4096),
        (64, 4, 4, 4096),
    ]

    if args.mode in ["all", "native"]:
        print(f"\n{'=' * 70}")
        print(f"  Native (fwd+bwd): dtype=float32")
        print(f"  warmup={args.warmup}, iters={args.iters}")
        print(f"{'=' * 70}")
        print(f"  {'(s, b, n, C)':<25s} {'avg (ms)':>10s} {'min (ms)':>10s} {'max (ms)':>10s}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

        for (s, b, n, C) in configs:
            avg_ms, min_ms, max_ms = _benchmark_native(
                s, b, n, C,
                dtype=torch.float32,
                warmup_iters=args.warmup,
                bench_iters=args.iters,
            )
            label = f"({s}, {b}, {n}, {C})"
            print(f"  {label:<25s} {avg_ms:10.3f} {min_ms:10.3f} {max_ms:10.3f}")

    if args.mode in ["all", "tilelang"]:
        # ---- TileLang forward benchmark ----
        print(f"\n{'=' * 70}")
        print(f"  TileLang (fwd only): dtype=float32")
        print(f"  warmup={args.warmup}, iters={args.iters}")
        print(f"{'=' * 70}")
        print(f"  {'(s, b, n, C)':<25s} {'avg (ms)':>10s} {'min (ms)':>10s} {'max (ms)':>10s}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

        for (s, b, n, C) in configs:
            avg_ms, min_ms, max_ms = _benchmark_tilelang_forward(
                s, b, n, C,
                dtype=torch.float32,
                warmup_iters=args.warmup,
                bench_iters=args.iters,
            )
            label = f"({s}, {b}, {n}, {C})"
            print(f"  {label:<25s} {avg_ms:10.3f} {min_ms:10.3f} {max_ms:10.3f}")

        # ---- TileLang backward benchmark ----
        print(f"\n{'=' * 70}")
        print(f"  TileLang (bwd only): dtype=float32")
        print(f"  warmup={args.warmup}, iters={args.iters}")
        print(f"{'=' * 70}")
        print(f"  {'(s, b, n, C)':<25s} {'avg (ms)':>10s} {'min (ms)':>10s} {'max (ms)':>10s}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

        for (s, b, n, C) in configs:
            avg_ms, min_ms, max_ms = _benchmark_tilelang_backward(
                s, b, n, C,
                dtype=torch.float32,
                warmup_iters=args.warmup,
                bench_iters=args.iters,
            )
            label = f"({s}, {b}, {n}, {C})"
            print(f"  {label:<25s} {avg_ms:10.3f} {min_ms:10.3f} {max_ms:10.3f}")

        # ---- TileLang fwd+bwd benchmark ----
        print(f"\n{'=' * 70}")
        print(f"  TileLang (fwd+bwd): dtype=float32")
        print(f"  warmup={args.warmup}, iters={args.iters}")
        print(f"{'=' * 70}")
        print(f"  {'(s, b, n, C)':<25s} {'avg (ms)':>10s} {'min (ms)':>10s} {'max (ms)':>10s}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

        for (s, b, n, C) in configs:
            avg_ms, min_ms, max_ms = _benchmark_tilelang_fwd_bwd(
                s, b, n, C,
                dtype=torch.float32,
                warmup_iters=args.warmup,
                bench_iters=args.iters,
            )
            label = f"({s}, {b}, {n}, {C})"
            print(f"  {label:<25s} {avg_ms:10.3f} {min_ms:10.3f} {max_ms:10.3f}")

    if args.mode in ["all", "compare"]:
        # ---- Forward comparison: Native vs TileLang ----
        print(f"\n{'=' * 70}")
        print(f"  Forward Comparison: Native vs TileLang")
        print(f"  warmup={args.warmup}, iters={args.iters}")
        print(f"{'=' * 70}")
        print(f"  {'(s, b, n, C)':<20s} {'Native(ms)':>12s} {'TileLang(ms)':>12s} {'Speedup':>10s}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")

        for (s, b, n, C) in configs:
            native_avg, _, _ = _benchmark_native_forward(
                s, b, n, C,
                dtype=torch.float32,
                warmup_iters=args.warmup,
                bench_iters=args.iters,
            )
            tilelang_avg, _, _ = _benchmark_tilelang_forward(
                s, b, n, C,
                dtype=torch.float32,
                warmup_iters=args.warmup,
                bench_iters=args.iters,
            )
            speedup = native_avg / tilelang_avg if tilelang_avg > 0 else float('inf')
            label = f"({s}, {b}, {n}, {C})"
            print(f"  {label:<20s} {native_avg:12.3f} {tilelang_avg:12.3f} {speedup:10.2f}x")

        # ---- Fwd+Bwd comparison: Native vs TileLang ----
        print(f"\n{'=' * 70}")
        print(f"  Fwd+Bwd Comparison: Native vs TileLang")
        print(f"  warmup={args.warmup}, iters={args.iters}")
        print(f"{'=' * 70}")
        print(f"  {'(s, b, n, C)':<20s} {'Native(ms)':>12s} {'TileLang(ms)':>12s} {'Speedup':>10s}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")

        for (s, b, n, C) in configs:
            native_avg, _, _ = _benchmark_native(
                s, b, n, C,
                dtype=torch.float32,
                warmup_iters=args.warmup,
                bench_iters=args.iters,
            )
            tilelang_avg, _, _ = _benchmark_tilelang_fwd_bwd(
                s, b, n, C,
                dtype=torch.float32,
                warmup_iters=args.warmup,
                bench_iters=args.iters,
            )
            speedup = native_avg / tilelang_avg if tilelang_avg > 0 else float('inf')
            label = f"({s}, {b}, {n}, {C})"
            print(f"  {label:<20s} {native_avg:12.3f} {tilelang_avg:12.3f} {speedup:10.2f}x")

    print(f"\n{'=' * 70}")
    print("Done.")
