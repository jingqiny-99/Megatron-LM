# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Memory usage comparison test for HyperConnection fused kernels vs native PyTorch.

Measures peak GPU memory during forward and backward passes for:
1. Native PyTorch path (no fused kernels)
2. Fused TileLang kernel path

Reports per-operation and total peak memory for each configuration.

Usage:
    python tests/unit_tests/transformer/test_hyper_connection_memory.py
    pytest tests/unit_tests/transformer/test_hyper_connection_memory.py -s
"""

import argparse
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.fusions.fused_sinkhorn import is_tilelang_available

try:
    from tests.unit_tests.test_utilities import Utils
    _HAS_UTILS = True
except ImportError:
    _HAS_UTILS = False


@dataclass
class MemorySnapshot:
    """Memory measurements for a single run."""
    fwd_peak_mb: float          # Peak memory during forward (above baseline)
    bwd_peak_mb: float          # Peak memory during backward (above post-fwd)
    total_peak_mb: float        # Overall peak memory (above baseline)
    fwd_alloc_mb: float         # Memory allocated after forward
    total_alloc_mb: float       # Memory allocated after backward


def _create_module(
    hidden_size: int,
    num_residual_streams: int,
    use_fused_kernel: bool,
) -> HyperConnectionModule:
    """Create a HyperConnectionModule for testing."""
    config = TransformerConfig(
        num_layers=2,
        hidden_size=hidden_size,
        num_attention_heads=max(4, hidden_size // 64),
        use_cpu_initialization=True,
        enable_hyper_connections=True,
        num_residual_streams=num_residual_streams,
        mhc_sinkhorn_iterations=5,
        mhc_init_gating_factor=0.01,
        mhc_use_fused_kernel=use_fused_kernel,
        bias_dropout_fusion=False,
        hidden_dropout=0.0,
    )
    module = HyperConnectionModule(config=config, layer_number=1)
    module.cuda()
    return module


def _measure_memory(
    module: HyperConnectionModule,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    layer_output: torch.Tensor,
    bias: Optional[torch.Tensor],
    num_warmup: int = 3,
) -> MemorySnapshot:
    """
    Measure peak GPU memory for forward and backward passes.

    Runs warmup iterations first, then measures a single clean iteration.
    Uses torch.cuda.max_memory_allocated() with reset for accurate peak tracking.

    Args:
        module: HyperConnectionModule to test
        hidden_states: [s, b, n*C] input tensor
        residual: [s, b, n*C] residual tensor
        layer_output: [s, b, C] simulated layer output
        bias: [C] or None
        num_warmup: Number of warmup iterations

    Returns:
        MemorySnapshot with peak memory measurements
    """
    n = module.n
    C = module.hidden_size

    # --- Warmup (let torch.compile / tilelang JIT settle) ---
    for _ in range(num_warmup):
        hs = hidden_states.detach().clone().requires_grad_(True)
        res = residual.detach().clone().requires_grad_(True)
        lo = layer_output.detach().clone().requires_grad_(True)

        aggregated, h_res, h_post = module.forward(hs, res)
        output = module.fused_h_res_h_post_bda(
            h_res, res, h_post, (lo, bias), dropout_prob=0.0, training=True
        )
        loss = output.sum()
        loss.backward()

        del hs, res, lo, aggregated, h_res, h_post, output, loss
        torch.cuda.empty_cache()

    # --- Measurement run ---
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    baseline = torch.cuda.memory_allocated()

    # Prepare inputs
    hs = hidden_states.detach().clone().requires_grad_(True)
    res = residual.detach().clone().requires_grad_(True)
    lo = layer_output.detach().clone().requires_grad_(True)

    # === Forward pass ===
    torch.cuda.reset_peak_memory_stats()
    fwd_start = torch.cuda.memory_allocated()

    aggregated, h_res, h_post = module.forward(hs, res)
    output = module.fused_h_res_h_post_bda(
        h_res, res, h_post, (lo, bias), dropout_prob=0.0, training=True
    )
    torch.cuda.synchronize()

    fwd_peak = torch.cuda.max_memory_allocated()
    fwd_end = torch.cuda.memory_allocated()

    fwd_peak_mb = (fwd_peak - fwd_start) / (1024 ** 2)
    fwd_alloc_mb = (fwd_end - fwd_start) / (1024 ** 2)

    # === Backward pass ===
    torch.cuda.reset_peak_memory_stats()
    bwd_start = torch.cuda.memory_allocated()

    loss = output.sum()
    loss.backward()
    torch.cuda.synchronize()

    bwd_peak = torch.cuda.max_memory_allocated()
    bwd_end = torch.cuda.memory_allocated()

    bwd_peak_mb = (bwd_peak - bwd_start) / (1024 ** 2)

    # Total peak (from baseline)
    total_peak_mb = max(fwd_peak, bwd_peak) / (1024 ** 2) - baseline / (1024 ** 2)
    total_alloc_mb = (bwd_end - fwd_start) / (1024 ** 2)

    # Cleanup
    del hs, res, lo, aggregated, h_res, h_post, output, loss
    torch.cuda.empty_cache()

    return MemorySnapshot(
        fwd_peak_mb=fwd_peak_mb,
        bwd_peak_mb=bwd_peak_mb,
        total_peak_mb=total_peak_mb,
        fwd_alloc_mb=fwd_alloc_mb,
        total_alloc_mb=total_alloc_mb,
    )


def run_memory_comparison(
    s: int = 1,
    b: int = 4,
    hidden_size: int = 1024,
    n: int = 4,
    dtype: torch.dtype = torch.bfloat16,
    num_warmup: int = 3,
) -> Tuple[MemorySnapshot, MemorySnapshot]:
    """
    Run memory comparison between native and fused kernel paths.

    Args:
        s: Sequence length
        b: Batch size
        hidden_size: Hidden dimension (C)
        n: Number of residual streams
        dtype: Data type for tensors
        num_warmup: Number of warmup iterations

    Returns:
        Tuple of (native_snapshot, fused_snapshot)
    """
    C = hidden_size

    # Create shared input tensors (on GPU, same data for fair comparison)
    hidden_states_data = torch.randn(s, b, n * C, dtype=dtype, device='cuda')
    residual_data = torch.randn(s, b, n * C, dtype=dtype, device='cuda')
    layer_output_data = torch.randn(s, b, C, dtype=dtype, device='cuda')
    bias_data = torch.randn(C, dtype=dtype, device='cuda')

    results = {}

    for label, use_fused in [("native", False), ("fused", True)]:
        if use_fused and not is_tilelang_available():
            print(f"  [SKIP] TileLang not available, skipping fused kernel test")
            results[label] = None
            continue

        # Fresh module for each test
        module = _create_module(hidden_size, n, use_fused)

        # Clone inputs for each run
        hs = hidden_states_data.clone()
        res = residual_data.clone()
        lo = layer_output_data.clone()
        bias = bias_data.clone()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        snapshot = _measure_memory(
            module, hs, res, lo, bias, num_warmup=num_warmup,
        )
        results[label] = snapshot

        # Cleanup module
        del module, hs, res, lo, bias
        torch.cuda.empty_cache()

    return results.get("native"), results.get("fused")


def print_report(
    native: Optional[MemorySnapshot],
    fused: Optional[MemorySnapshot],
    config_str: str,
):
    """Pretty-print a comparison report."""
    print(f"\n{'=' * 70}")
    print(f"  Memory Comparison: {config_str}")
    print(f"{'=' * 70}")
    print(f"{'Metric':<35} {'Native (MB)':>12} {'Fused (MB)':>12} {'Δ (MB)':>10}")
    print(f"{'-' * 70}")

    def row(name, native_val, fused_val):
        if native_val is not None and fused_val is not None:
            delta = fused_val - native_val
            sign = "+" if delta >= 0 else ""
            print(f"{name:<35} {native_val:>12.2f} {fused_val:>12.2f} {sign}{delta:>9.2f}")
        elif native_val is not None:
            print(f"{name:<35} {native_val:>12.2f} {'N/A':>12} {'N/A':>10}")
        else:
            print(f"{name:<35} {'N/A':>12} {'N/A':>12} {'N/A':>10}")

    n_fwd = native.fwd_peak_mb if native else None
    f_fwd = fused.fwd_peak_mb if fused else None
    row("Forward peak memory", n_fwd, f_fwd)

    n_bwd = native.bwd_peak_mb if native else None
    f_bwd = fused.bwd_peak_mb if fused else None
    row("Backward peak memory", n_bwd, f_bwd)

    n_tot = native.total_peak_mb if native else None
    f_tot = fused.total_peak_mb if fused else None
    row("Total peak memory", n_tot, f_tot)

    n_fa = native.fwd_alloc_mb if native else None
    f_fa = fused.fwd_alloc_mb if fused else None
    row("Forward retained (activations)", n_fa, f_fa)

    n_ta = native.total_alloc_mb if native else None
    f_ta = fused.total_alloc_mb if fused else None
    row("Total retained after backward", n_ta, f_ta)

    print(f"{'=' * 70}")


# ==================== Pytest interface ====================

class TestHyperConnectionMemory:
    """Memory comparison tests for HyperConnection fused vs native."""

    def setup_method(self, method):
        if _HAS_UTILS:
            Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        if _HAS_UTILS:
            Utils.destroy_model_parallel()

    def test_memory_fused_vs_native_small(self):
        """Compare memory for small config: s=1, b=4, C=1024, n=4."""
        native, fused = run_memory_comparison(
            s=1, b=4, hidden_size=1024, n=4, dtype=torch.bfloat16,
        )
        print_report(native, fused, "s=1, b=4, C=1024, n=4, bf16")

        if native is not None and fused is not None:
            # Fused forward peak should be <= native (fewer intermediates)
            # Allow some tolerance for measurement noise
            print(f"\n  Forward peak: native={native.fwd_peak_mb:.2f} MB, "
                  f"fused={fused.fwd_peak_mb:.2f} MB")
            print(f"  Backward peak: native={native.bwd_peak_mb:.2f} MB, "
                  f"fused={fused.bwd_peak_mb:.2f} MB")

    def test_memory_fused_vs_native_large(self):
        """Compare memory for larger config: s=4, b=8, C=4096, n=4."""
        native, fused = run_memory_comparison(
            s=4, b=8, hidden_size=4096, n=4, dtype=torch.bfloat16,
        )
        print_report(native, fused, "s=4, b=8, C=4096, n=4, bf16")

    def test_memory_fused_vs_native_fp32(self):
        """Compare memory with fp32 to verify dtype parameterization benefit."""
        native_fp32, fused_fp32 = run_memory_comparison(
            s=1, b=4, hidden_size=1024, n=4, dtype=torch.float32,
        )
        print_report(native_fp32, fused_fp32, "s=1, b=4, C=1024, n=4, fp32")

        native_bf16, fused_bf16 = run_memory_comparison(
            s=1, b=4, hidden_size=1024, n=4, dtype=torch.bfloat16,
        )
        print_report(native_bf16, fused_bf16, "s=1, b=4, C=1024, n=4, bf16")

        if fused_fp32 is not None and fused_bf16 is not None:
            print(f"\n  [dtype parameterization] "
                  f"Fused fp32 fwd peak: {fused_fp32.fwd_peak_mb:.2f} MB, "
                  f"Fused bf16 fwd peak: {fused_bf16.fwd_peak_mb:.2f} MB")
            ratio = fused_fp32.fwd_peak_mb / max(fused_bf16.fwd_peak_mb, 0.01)
            print(f"  fp32/bf16 ratio: {ratio:.2f}x "
                  f"(expected ~2x if dtype parameterization works)")


# ==================== Standalone runner ====================

def main():
    import sys

    parser = argparse.ArgumentParser(description="Memory comparison: fused vs native HyperConnection")
    parser.add_argument("--seq-len", "-s", type=int, default=1, help="Sequence length")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument("--hidden-size", "-C", type=int, default=4096, help="Hidden size")
    parser.add_argument("--num-streams", "-n", type=int, default=4, help="Number of residual streams")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--sweep", action="store_true", help="Run a sweep of configurations")
    args = parser.parse_args()

    # Initialize model parallel (required for TransformerConfig)
    if _HAS_UTILS:
        Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    if args.sweep:
        configs = [
            # (s, b, C, n, dtype_str)
            (1, 4, 1024, 4, torch.bfloat16),
            (1, 4, 2048, 4, torch.bfloat16),
            (1, 4, 4096, 4, torch.bfloat16),
            (4, 4, 4096, 4, torch.bfloat16),
            (4, 8, 4096, 4, torch.bfloat16),
            (4, 8, 4096, 6, torch.bfloat16),
            # fp32 comparison
            (1, 4, 4096, 4, torch.float32),
        ]
        for s, b, C, n, dt in configs:
            dt_name = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}[dt]
            try:
                native, fused = run_memory_comparison(
                    s=s, b=b, hidden_size=C, n=n, dtype=dt, num_warmup=args.warmup,
                )
                print_report(native, fused, f"s={s}, b={b}, C={C}, n={n}, {dt_name}")
            except Exception as e:
                print(f"\n[ERROR] s={s}, b={b}, C={C}, n={n}, {dt_name}: {e}")
    else:
        native, fused = run_memory_comparison(
            s=args.seq_len, b=args.batch_size, hidden_size=args.hidden_size,
            n=args.num_streams, dtype=dtype, num_warmup=args.warmup,
        )
        dt_name = args.dtype
        print_report(
            native, fused,
            f"s={args.seq_len}, b={args.batch_size}, C={args.hidden_size}, "
            f"n={args.num_streams}, {dt_name}"
        )

    if _HAS_UTILS:
        Utils.destroy_model_parallel()


if __name__ == "__main__":
    main()
