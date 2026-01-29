# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unit tests for fused Sinkhorn-Knopp kernel.

Tests include:
1. Correctness: Compare fused TileLang kernel output with PyTorch native implementation
2. Performance: Benchmark latency comparison between fused and native implementations
"""

import pytest
import torch
from typing import List, Dict, Tuple

from megatron.core.fusions.fused_sinkhorn import (
    sinkhorn_fused_forward,
    sinkhorn_native_forward,
    is_tilelang_available,
)


# Skip all tests if tilelang is not available
pytestmark = pytest.mark.skipif(
    not is_tilelang_available(),
    reason="TileLang is not available"
)


class TestSinkhornCorrectness:
    """Test correctness of fused Sinkhorn kernel against PyTorch native implementation."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    @pytest.mark.parametrize("seq_len", [128, 512, 2048])
    @pytest.mark.parametrize("hc", [2, 4, 8])
    @pytest.mark.parametrize("num_iterations", [5, 10, 20])
    def test_correctness_various_shapes(
        self, batch_size: int, seq_len: int, hc: int, num_iterations: int
    ):
        """Test that fused kernel produces same results as native implementation."""
        input_logits = torch.randn(
            seq_len, batch_size, hc, hc,
            device='cuda', dtype=torch.float32
        )

        output_native = sinkhorn_native_forward(input_logits, num_iterations, eps=1e-8)
        output_fused = sinkhorn_fused_forward(input_logits, num_iterations, eps=1e-8)

        assert output_fused.shape == output_native.shape
        torch.testing.assert_close(
            output_fused, output_native,
            rtol=1e-3, atol=1e-4,
            msg=f"Output mismatch for shape [{seq_len}, {batch_size}, {hc}, {hc}], "
                f"num_iterations={num_iterations}"
        )

    def test_correctness_flat_input(self):
        """Test with flattened input shape (num_tokens, hc, hc)."""
        input_logits = torch.randn(1024, 4, 4, device='cuda', dtype=torch.float32)

        output_native = sinkhorn_native_forward(input_logits, 20)
        output_fused = sinkhorn_fused_forward(input_logits, 20)

        torch.testing.assert_close(output_fused, output_native, rtol=1e-3, atol=1e-4)

    def test_correctness_edge_cases(self):
        """Test edge cases: small matrices, single token, etc."""
        # Single token
        input_single = torch.randn(1, 4, 4, device='cuda', dtype=torch.float32)
        output_native = sinkhorn_native_forward(input_single, 20)
        output_fused = sinkhorn_fused_forward(input_single, 20)
        torch.testing.assert_close(output_fused, output_native, rtol=1e-3, atol=1e-4)

        # 2x2 matrix
        input_2x2 = torch.randn(100, 2, 2, device='cuda', dtype=torch.float32)
        output_native = sinkhorn_native_forward(input_2x2, 20)
        output_fused = sinkhorn_fused_forward(input_2x2, 20)
        torch.testing.assert_close(output_fused, output_native, rtol=1e-3, atol=1e-4)

    def test_numerical_stability_large_values(self):
        """Test numerical stability with large input values."""
        input_logits = torch.randn(512, 4, 4, device='cuda', dtype=torch.float32) * 10

        output_native = sinkhorn_native_forward(input_logits, 20)
        output_fused = sinkhorn_fused_forward(input_logits, 20)

        assert not torch.isnan(output_fused).any(), "Fused output contains NaN"
        assert not torch.isinf(output_fused).any(), "Fused output contains Inf"

        torch.testing.assert_close(output_fused, output_native, rtol=1e-2, atol=1e-3)


class TestSinkhornPerformance:
    """Benchmark performance of fused Sinkhorn kernel vs PyTorch native."""

    def _benchmark_function(
        self,
        func,
        inputs: Tuple,
        num_warmup: int = 10,
        num_iterations: int = 100,
    ) -> float:
        """Benchmark a function using CUDA events. Returns average latency in ms."""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for _ in range(num_warmup):
            _ = func(*inputs)
        torch.cuda.synchronize()

        latencies = []
        for _ in range(num_iterations):
            start_event.record()
            _ = func(*inputs)
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))

        return sum(latencies) / len(latencies)

    def _get_test_shapes(self) -> List[Dict]:
        """Get test shapes for benchmarking."""
        shapes = []
        for seq_len in [512, 1024, 2048, 4096]:
            for batch_size in [1, 2, 4]:
                for hc in [4, 8]:
                    shapes.append({
                        'seq_len': seq_len,
                        'batch_size': batch_size,
                        'hc': hc,
                    })
        return shapes

    def test_performance_comparison(self):
        """Benchmark and compare fused vs native Sinkhorn implementations."""
        num_sinkhorn_iterations = 20
        results = []

        for shape in self._get_test_shapes():
            seq_len = shape['seq_len']
            batch_size = shape['batch_size']
            hc = shape['hc']

            input_logits = torch.randn(
                seq_len, batch_size, hc, hc,
                device='cuda', dtype=torch.float32
            )

            native_latency = self._benchmark_function(
                lambda x: sinkhorn_native_forward(x, num_sinkhorn_iterations),
                (input_logits,),
            )

            fused_latency = self._benchmark_function(
                lambda x: sinkhorn_fused_forward(x, num_sinkhorn_iterations),
                (input_logits,),
            )

            speedup = native_latency / fused_latency if fused_latency > 0 else float('inf')

            results.append({
                'shape': f"s={seq_len}, b={batch_size}, hc={hc}",
                'num_tokens': seq_len * batch_size,
                'native_ms': native_latency,
                'fused_ms': fused_latency,
                'speedup': speedup,
            })

            del input_logits
            torch.cuda.empty_cache()

        self._print_performance_table(results, num_sinkhorn_iterations)

    def test_performance_varying_iterations(self):
        """Benchmark performance with varying number of Sinkhorn iterations."""
        seq_len, batch_size, hc = 1024, 2, 4
        results = []

        input_logits = torch.randn(
            seq_len, batch_size, hc, hc,
            device='cuda', dtype=torch.float32
        )

        for num_iterations in [5, 10, 20, 50]:
            native_latency = self._benchmark_function(
                lambda x, n=num_iterations: sinkhorn_native_forward(x, n),
                (input_logits,),
            )

            fused_latency = self._benchmark_function(
                lambda x, n=num_iterations: sinkhorn_fused_forward(x, n),
                (input_logits,),
            )

            speedup = native_latency / fused_latency if fused_latency > 0 else float('inf')

            results.append({
                'num_iterations': num_iterations,
                'native_ms': native_latency,
                'fused_ms': fused_latency,
                'speedup': speedup,
            })

        self._print_iteration_comparison_table(results, seq_len, batch_size, hc)

    def test_performance_large_batch(self):
        """Benchmark performance with large batch/sequence combinations."""
        num_sinkhorn_iterations = 20
        results = []

        configs = [
            (4096, 4, 4),
            (2048, 8, 4),
            (8192, 1, 4),
            (1024, 4, 8),
        ]

        for seq_len, batch_size, hc in configs:
            input_logits = torch.randn(
                seq_len, batch_size, hc, hc,
                device='cuda', dtype=torch.float32
            )

            native_latency = self._benchmark_function(
                lambda x: sinkhorn_native_forward(x, num_sinkhorn_iterations),
                (input_logits,),
                num_iterations=50,
            )

            fused_latency = self._benchmark_function(
                lambda x: sinkhorn_fused_forward(x, num_sinkhorn_iterations),
                (input_logits,),
                num_iterations=50,
            )

            speedup = native_latency / fused_latency if fused_latency > 0 else float('inf')

            results.append({
                'shape': f"s={seq_len}, b={batch_size}, hc={hc}",
                'num_tokens': seq_len * batch_size,
                'native_ms': native_latency,
                'fused_ms': fused_latency,
                'speedup': speedup,
            })

            del input_logits
            torch.cuda.empty_cache()

        print(f"\n{'=' * 90}")
        print(f"LARGE BATCH PERFORMANCE (sinkhorn_iterations={num_sinkhorn_iterations})")
        print(f"{'=' * 90}")
        print(f"{'Shape':<30} | {'Tokens':>10} | {'Native':>12} | {'Fused':>12} | {'Speedup':>10}")
        print(f"{'-' * 30}-+-{'-' * 10}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}")

        for r in results:
            print(
                f"{r['shape']:<30} | "
                f"{r['num_tokens']:>10} | "
                f"{r['native_ms']:>9.4f} ms | "
                f"{r['fused_ms']:>9.4f} ms | "
                f"{r['speedup']:>9.2f}x"
            )
        print(f"{'=' * 90}\n")

    def _print_performance_table(self, results: List[Dict], num_iterations: int):
        """Print formatted performance comparison table."""
        print(f"\n{'=' * 100}")
        print(f"SINKHORN PERFORMANCE (sinkhorn_iterations={num_iterations})")
        print(f"{'=' * 100}")
        print(f"{'Shape':<30} | {'Tokens':>10} | {'Native':>12} | {'Fused':>12} | {'Speedup':>10}")
        print(f"{'-' * 30}-+-{'-' * 10}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}")

        for r in results:
            print(
                f"{r['shape']:<30} | "
                f"{r['num_tokens']:>10} | "
                f"{r['native_ms']:>9.4f} ms | "
                f"{r['fused_ms']:>9.4f} ms | "
                f"{r['speedup']:>9.2f}x"
            )

        print(f"{'=' * 100}")

        native_times = [r['native_ms'] for r in results]
        fused_times = [r['fused_ms'] for r in results]
        speedups = [r['speedup'] for r in results]

        print(f"\nSUMMARY:")
        print(f"  Native:  min={min(native_times):.4f}ms, max={max(native_times):.4f}ms, avg={sum(native_times)/len(native_times):.4f}ms")
        print(f"  Fused:   min={min(fused_times):.4f}ms, max={max(fused_times):.4f}ms, avg={sum(fused_times)/len(fused_times):.4f}ms")
        print(f"  Speedup: min={min(speedups):.2f}x, max={max(speedups):.2f}x, avg={sum(speedups)/len(speedups):.2f}x")
        print()

    def _print_iteration_comparison_table(
        self, results: List[Dict], seq_len: int, batch_size: int, hc: int
    ):
        """Print performance comparison with varying iterations."""
        print(f"\n{'=' * 70}")
        print(f"ITERATION SCALING (s={seq_len}, b={batch_size}, hc={hc})")
        print(f"{'=' * 70}")
        print(f"{'Iterations':>12} | {'Native':>12} | {'Fused':>12} | {'Speedup':>10}")
        print(f"{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}")

        for r in results:
            print(
                f"{r['num_iterations']:>12} | "
                f"{r['native_ms']:>9.4f} ms | "
                f"{r['fused_ms']:>9.4f} ms | "
                f"{r['speedup']:>9.2f}x"
            )
        print(f"{'=' * 70}\n")


class TestSinkhornIntegration:
    """Integration tests with HyperConnectionModule."""

    def setup_method(self, method):
        from tests.unit_tests.test_utilities import Utils
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        from tests.unit_tests.test_utilities import Utils
        Utils.destroy_model_parallel()

    def test_hyper_connection_with_fused_sinkhorn(self):
        """Test HyperConnectionModule with fused Sinkhorn kernel enabled."""
        from megatron.core.transformer.hyper_connection import HyperConnectionModule
        from megatron.core.transformer.transformer_config import TransformerConfig

        hidden_size = 256
        num_streams = 4
        seq_len = 128
        batch_size = 2

        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=num_streams,
            mhc_sinkhorn_iterations=20,
            mhc_init_gating_factor=0.01,
            mhc_use_fused_kernel=True,
        )

        module_fused = HyperConnectionModule(config=config, layer_number=1)
        module_fused.cuda()

        config.mhc_use_fused_kernel = False
        module_native = HyperConnectionModule(config=config, layer_number=1)
        module_native.cuda()
        module_native.load_state_dict(module_fused.state_dict())

        x = torch.randn(
            seq_len, batch_size, num_streams * hidden_size,
            device='cuda', dtype=torch.float32
        )

        h_pre_fused, h_post_fused, h_res_fused = module_fused.compute_mappings(x)
        h_pre_native, h_post_native, h_res_native = module_native.compute_mappings(x)

        torch.testing.assert_close(h_pre_fused, h_pre_native, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(h_post_fused, h_post_native, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(h_res_fused, h_res_native, rtol=1e-3, atol=1e-4)


def run_benchmark():
    """
    Standalone benchmark function.

    Usage:
        python -c "from tests.unit_tests.fusions.test_fused_sinkhorn import run_benchmark; run_benchmark()"
    """
    print("=" * 60)
    print("FUSED SINKHORN KERNEL BENCHMARK")
    print("=" * 60)

    if not is_tilelang_available():
        print("ERROR: TileLang is not available.")
        return

    test = TestSinkhornPerformance()

    print("\n[1/3] Performance comparison across shapes...")
    test.test_performance_comparison()

    print("\n[2/3] Iteration scaling test...")
    test.test_performance_varying_iterations()

    print("\n[3/3] Large batch test...")
    test.test_performance_large_batch()

    print("\nBenchmark complete!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
