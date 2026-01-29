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
        """
        Test that fused kernel produces same results as native implementation
        for various input shapes.
        """
        # Create random input logits
        # Shape: [seq_len, batch_size, hc, hc] to match HyperConnection usage
        input_logits = torch.randn(
            seq_len, batch_size, hc, hc,
            device='cuda', dtype=torch.float32
        )
        
        # Run native implementation
        output_native = sinkhorn_native_forward(
            input_logits, num_iterations, eps=1e-8
        )
        
        # Run fused implementation
        output_fused = sinkhorn_fused_forward(
            input_logits, num_iterations, eps=1e-8
        )
        
        # Check shapes match
        assert output_fused.shape == output_native.shape, (
            f"Shape mismatch: fused={output_fused.shape}, native={output_native.shape}"
        )
        
        # Check numerical correctness with tolerance
        # Sinkhorn iterations can amplify small numerical differences,
        # so we use a relaxed tolerance
        torch.testing.assert_close(
            output_fused, output_native,
            rtol=1e-3, atol=1e-4,
            msg=f"Output mismatch for shape [{seq_len}, {batch_size}, {hc}, {hc}], "
                f"num_iterations={num_iterations}"
        )
        
        # Verify doubly stochastic property
        self._verify_doubly_stochastic(output_fused, rtol=1e-2, atol=1e-3)
        self._verify_doubly_stochastic(output_native, rtol=1e-2, atol=1e-3)

    def test_correctness_flat_input(self):
        """Test with flattened input shape (num_tokens, hc, hc)."""
        num_tokens = 1024
        hc = 4
        num_iterations = 20
        
        input_logits = torch.randn(
            num_tokens, hc, hc,
            device='cuda', dtype=torch.float32
        )
        
        output_native = sinkhorn_native_forward(input_logits, num_iterations)
        output_fused = sinkhorn_fused_forward(input_logits, num_iterations)
        
        torch.testing.assert_close(
            output_fused, output_native,
            rtol=1e-3, atol=1e-4
        )

    def test_correctness_edge_cases(self):
        """Test edge cases: small matrices, single token, etc."""
        num_iterations = 20
        
        # Single token
        input_single = torch.randn(1, 4, 4, device='cuda', dtype=torch.float32)
        output_native = sinkhorn_native_forward(input_single, num_iterations)
        output_fused = sinkhorn_fused_forward(input_single, num_iterations)
        torch.testing.assert_close(output_fused, output_native, rtol=1e-3, atol=1e-4)
        
        # 2x2 matrix
        input_2x2 = torch.randn(100, 2, 2, device='cuda', dtype=torch.float32)
        output_native = sinkhorn_native_forward(input_2x2, num_iterations)
        output_fused = sinkhorn_fused_forward(input_2x2, num_iterations)
        torch.testing.assert_close(output_fused, output_native, rtol=1e-3, atol=1e-4)

    def test_numerical_stability_large_values(self):
        """Test numerical stability with large input values."""
        input_logits = torch.randn(512, 4, 4, device='cuda', dtype=torch.float32) * 10
        num_iterations = 20
        
        output_native = sinkhorn_native_forward(input_logits, num_iterations)
        output_fused = sinkhorn_fused_forward(input_logits, num_iterations)
        
        # Check no NaN or Inf
        assert not torch.isnan(output_fused).any(), "Fused output contains NaN"
        assert not torch.isinf(output_fused).any(), "Fused output contains Inf"
        assert not torch.isnan(output_native).any(), "Native output contains NaN"
        assert not torch.isinf(output_native).any(), "Native output contains Inf"
        
        torch.testing.assert_close(output_fused, output_native, rtol=1e-2, atol=1e-3)

    def _verify_doubly_stochastic(
        self, M: torch.Tensor, rtol: float = 1e-2, atol: float = 1e-3
    ):
        """
        Verify that output matrix is doubly stochastic:
        - All elements are non-negative
        - Each row sums to 1
        - Each column sums to 1
        """
        # Non-negative
        assert (M >= 0).all(), "Matrix contains negative values"
        
        # Row sums should be close to 1
        row_sums = M.sum(dim=-1)
        expected_ones = torch.ones_like(row_sums)
        torch.testing.assert_close(
            row_sums, expected_ones,
            rtol=rtol, atol=atol,
            msg="Row sums not equal to 1"
        )
        
        # Column sums should be close to 1
        col_sums = M.sum(dim=-2)
        torch.testing.assert_close(
            col_sums, expected_ones,
            rtol=rtol, atol=atol,
            msg="Column sums not equal to 1"
        )


class TestSinkhornPerformance:
    """Benchmark performance of fused Sinkhorn kernel vs PyTorch native."""

    def _benchmark_function(
        self,
        func,
        inputs: Tuple,
        num_warmup: int = 10,
        num_iterations: int = 100,
    ) -> float:
        """
        Benchmark a function using CUDA events for accurate GPU timing.
        
        Returns:
            Average latency in milliseconds
        """
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Warmup
        for _ in range(num_warmup):
            _ = func(*inputs)
        torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start_event.record()
            _ = func(*inputs)
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
        
        return sum(latencies) / len(latencies)

    def _get_test_shapes(self) -> List[Dict]:
        """Get a list of test shapes for benchmarking."""
        shapes = []
        
        # Typical mHC usage patterns
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
        """
        Benchmark and compare fused vs native Sinkhorn implementations.
        Prints a detailed performance comparison table.
        """
        num_sinkhorn_iterations = 20
        results = []
        
        test_shapes = self._get_test_shapes()
        
        for shape in test_shapes:
            seq_len = shape['seq_len']
            batch_size = shape['batch_size']
            hc = shape['hc']
            
            # Create input: [seq_len, batch_size, hc, hc]
            input_logits = torch.randn(
                seq_len, batch_size, hc, hc,
                device='cuda', dtype=torch.float32
            )
            
            # Benchmark native implementation
            native_latency = self._benchmark_function(
                lambda x: sinkhorn_native_forward(x, num_sinkhorn_iterations),
                (input_logits,),
                num_warmup=10,
                num_iterations=100,
            )
            
            # Benchmark fused implementation
            fused_latency = self._benchmark_function(
                lambda x: sinkhorn_fused_forward(x, num_sinkhorn_iterations),
                (input_logits,),
                num_warmup=10,
                num_iterations=100,
            )
            
            # Calculate speedup
            speedup = native_latency / fused_latency if fused_latency > 0 else float('inf')
            
            results.append({
                'shape': f"s={seq_len}, b={batch_size}, hc={hc}",
                'num_tokens': seq_len * batch_size,
                'native_ms': native_latency,
                'fused_ms': fused_latency,
                'speedup': speedup,
            })
            
            # Clean up
            del input_logits
            torch.cuda.empty_cache()
        
        # Print results table
        self._print_performance_table(results, num_sinkhorn_iterations)

    def test_performance_varying_iterations(self):
        """
        Benchmark performance with varying number of Sinkhorn iterations.
        """
        seq_len, batch_size, hc = 1024, 2, 4
        results = []
        
        input_logits = torch.randn(
            seq_len, batch_size, hc, hc,
            device='cuda', dtype=torch.float32
        )
        
        for num_iterations in [5, 10, 20, 50]:
            # Benchmark native
            native_latency = self._benchmark_function(
                lambda x, n=num_iterations: sinkhorn_native_forward(x, n),
                (input_logits,),
                num_warmup=10,
                num_iterations=100,
            )
            
            # Benchmark fused
            fused_latency = self._benchmark_function(
                lambda x, n=num_iterations: sinkhorn_fused_forward(x, n),
                (input_logits,),
                num_warmup=10,
                num_iterations=100,
            )
            
            speedup = native_latency / fused_latency if fused_latency > 0 else float('inf')
            
            results.append({
                'num_iterations': num_iterations,
                'native_ms': native_latency,
                'fused_ms': fused_latency,
                'speedup': speedup,
            })
        
        # Print results
        self._print_iteration_comparison_table(results, seq_len, batch_size, hc)

    def test_performance_large_batch(self):
        """
        Benchmark performance with large batch/sequence combinations.
        """
        num_sinkhorn_iterations = 20
        results = []
        
        # Large configurations
        configs = [
            (4096, 4, 4),   # Large sequence
            (2048, 8, 4),   # Large batch
            (8192, 1, 4),   # Very large sequence
            (1024, 4, 8),   # Larger hc dimension
        ]
        
        for seq_len, batch_size, hc in configs:
            input_logits = torch.randn(
                seq_len, batch_size, hc, hc,
                device='cuda', dtype=torch.float32
            )
            
            # Benchmark native
            native_latency = self._benchmark_function(
                lambda x: sinkhorn_native_forward(x, num_sinkhorn_iterations),
                (input_logits,),
                num_warmup=10,
                num_iterations=50,  # Fewer iterations for large configs
            )
            
            # Benchmark fused
            fused_latency = self._benchmark_function(
                lambda x: sinkhorn_fused_forward(x, num_sinkhorn_iterations),
                (input_logits,),
                num_warmup=10,
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
        
        # Print results
        print(f"\n{'=' * 90}")
        print(f"LARGE BATCH PERFORMANCE COMPARISON (sinkhorn_iterations={num_sinkhorn_iterations})")
        print(f"{'=' * 90}")
        print(f"{'Shape':<30} | {'Num Tokens':>12} | {'Native (ms)':>12} | {'Fused (ms)':>12} | {'Speedup':>10}")
        print(f"{'-' * 30}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}")
        
        for r in results:
            print(
                f"{r['shape']:<30} | "
                f"{r['num_tokens']:>12} | "
                f"{r['native_ms']:>9.4f} ms | "
                f"{r['fused_ms']:>9.4f} ms | "
                f"{r['speedup']:>9.2f}x"
            )
        
        print(f"{'=' * 90}\n")

    def _print_performance_table(self, results: List[Dict], num_iterations: int):
        """Print formatted performance comparison table."""
        print(f"\n{'=' * 100}")
        print(f"SINKHORN PERFORMANCE COMPARISON (sinkhorn_iterations={num_iterations})")
        print(f"{'=' * 100}")
        print(f"{'Shape':<30} | {'Num Tokens':>12} | {'Native (ms)':>12} | {'Fused (ms)':>12} | {'Speedup':>10}")
        print(f"{'-' * 30}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}")
        
        for r in results:
            print(
                f"{r['shape']:<30} | "
                f"{r['num_tokens']:>12} | "
                f"{r['native_ms']:>9.4f} ms | "
                f"{r['fused_ms']:>9.4f} ms | "
                f"{r['speedup']:>9.2f}x"
            )
        
        print(f"{'=' * 100}")
        
        # Print summary statistics
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
        print(f"\n{'=' * 80}")
        print(f"SINKHORN ITERATION SCALING (shape: s={seq_len}, b={batch_size}, hc={hc})")
        print(f"{'=' * 80}")
        print(f"{'Iterations':>12} | {'Native (ms)':>12} | {'Fused (ms)':>12} | {'Speedup':>10}")
        print(f"{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}")
        
        for r in results:
            print(
                f"{r['num_iterations']:>12} | "
                f"{r['native_ms']:>9.4f} ms | "
                f"{r['fused_ms']:>9.4f} ms | "
                f"{r['speedup']:>9.2f}x"
            )
        
        print(f"{'=' * 80}\n")


class TestSinkhornIntegration:
    """Integration tests with HyperConnectionModule."""

    def setup_method(self, method):
        """Set up test fixtures."""
        from tests.unit_tests.test_utilities import Utils
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        """Tear down test fixtures."""
        from tests.unit_tests.test_utilities import Utils
        Utils.destroy_model_parallel()

    def test_hyper_connection_with_fused_sinkhorn(self):
        """
        Test HyperConnectionModule with fused Sinkhorn kernel enabled.
        """
        from megatron.core.transformer.hyper_connection import HyperConnectionModule
        from megatron.core.transformer.transformer_config import TransformerConfig
        
        hidden_size = 256
        num_streams = 4
        seq_len = 128
        batch_size = 2
        
        # Create config with fused kernel enabled
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
        
        # Create module with native implementation
        config.mhc_use_fused_kernel = False
        module_native = HyperConnectionModule(config=config, layer_number=1)
        module_native.cuda()
        
        # Copy weights
        module_native.load_state_dict(module_fused.state_dict())
        
        # Create input
        x = torch.randn(
            seq_len, batch_size, num_streams * hidden_size,
            device='cuda', dtype=torch.float32
        )
        
        # Run forward with fused sinkhorn
        h_pre_fused, h_post_fused, h_res_fused = module_fused.compute_mappings(x)
        
        # Run forward with native sinkhorn
        h_pre_native, h_post_native, h_res_native = module_native.compute_mappings(x)
        
        # Check h_pre and h_post are identical (Sinkhorn doesn't affect them)
        torch.testing.assert_close(h_pre_fused, h_pre_native, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(h_post_fused, h_post_native, rtol=1e-5, atol=1e-5)
        
        # Check h_res is close (some numerical difference expected due to different implementations)
        torch.testing.assert_close(h_res_fused, h_res_native, rtol=1e-3, atol=1e-4)


def run_benchmark():
    """
    Standalone function to run the performance benchmark.
    
    Usage:
        python -c "from tests.unit_tests.fusions.test_fused_sinkhorn import run_benchmark; run_benchmark()"
    """
    print("=" * 60)
    print("FUSED SINKHORN KERNEL BENCHMARK")
    print("=" * 60)
    
    if not is_tilelang_available():
        print("ERROR: TileLang is not available. Cannot run benchmark.")
        return
    
    test = TestSinkhornPerformance()
    
    print("\n[1/3] Running performance comparison across shapes...")
    test.test_performance_comparison()
    
    print("\n[2/3] Running iteration scaling test...")
    test.test_performance_varying_iterations()
    
    print("\n[3/3] Running large batch test...")
    test.test_performance_large_batch()
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    # Run with: python tests/unit_tests/fusions/test_fused_sinkhorn.py
    pytest.main([__file__, "-v", "-s"])
