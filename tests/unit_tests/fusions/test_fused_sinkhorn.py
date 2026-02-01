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
    sinkhorn_fused_backward,
    sinkhorn_native_forward,
    sinkhorn_native_backward,
    is_tilelang_available,
)


pytestmark = pytest.mark.skipif(
    not is_tilelang_available(),
    reason="TileLang is not available"
)


class TestSinkhornForwardCorrectness:
    """Test correctness of fused Sinkhorn forward kernel."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    @pytest.mark.parametrize("seq_len", [128, 512, 2048])
    @pytest.mark.parametrize("hc", [2, 4, 8])
    @pytest.mark.parametrize("num_iterations", [5, 10, 20])
    def test_forward_various_shapes(
        self, batch_size: int, seq_len: int, hc: int, num_iterations: int
    ):
        """Test that fused forward produces same results as native implementation."""
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
            msg=f"Forward mismatch for shape [{seq_len}, {batch_size}, {hc}, {hc}]"
        )

    def test_forward_flat_input(self):
        """Test forward with flattened input shape."""
        input_logits = torch.randn(1024, 4, 4, device='cuda', dtype=torch.float32)

        output_native = sinkhorn_native_forward(input_logits, 20)
        output_fused = sinkhorn_fused_forward(input_logits, 20)

        torch.testing.assert_close(output_fused, output_native, rtol=1e-3, atol=1e-4)

    def test_forward_numerical_stability(self):
        """Test numerical stability with large input values."""
        input_logits = torch.randn(512, 4, 4, device='cuda', dtype=torch.float32) * 10

        output_native = sinkhorn_native_forward(input_logits, 20)
        output_fused = sinkhorn_fused_forward(input_logits, 20)

        assert not torch.isnan(output_fused).any(), "Fused output contains NaN"
        assert not torch.isinf(output_fused).any(), "Fused output contains Inf"

        torch.testing.assert_close(output_fused, output_native, rtol=1e-2, atol=1e-3)


class TestSinkhornBackwardCorrectness:
    """Test correctness of fused Sinkhorn backward kernel."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [128, 512])
    @pytest.mark.parametrize("hc", [2, 4, 8])
    @pytest.mark.parametrize("num_iterations", [5, 10, 20])
    def test_backward_various_shapes(
        self, batch_size: int, seq_len: int, hc: int, num_iterations: int
    ):
        """Test that fused backward produces same results as native implementation."""
        input_logits = torch.randn(
            seq_len, batch_size, hc, hc,
            device='cuda', dtype=torch.float32
        )
        
        # M_init = exp(input_logits)
        M_init = torch.exp(input_logits)
        
        # Random gradient from downstream
        grad_output = torch.randn_like(M_init)

        grad_native = sinkhorn_native_backward(grad_output, M_init, num_iterations, eps=1e-8)
        grad_fused = sinkhorn_fused_backward(grad_output, M_init, num_iterations, eps=1e-8)

        assert grad_fused.shape == grad_native.shape
        torch.testing.assert_close(
            grad_fused, grad_native,
            rtol=1e-2, atol=1e-3,
            msg=f"Backward mismatch for shape [{seq_len}, {batch_size}, {hc}, {hc}], expected={grad_native}, actual={grad_fused}"
        )

    def test_backward_flat_input(self):
        """Test backward with flattened input shape."""
        input_logits = torch.randn(1024, 4, 4, device='cuda', dtype=torch.float32)
        M_init = torch.exp(input_logits)
        grad_output = torch.randn_like(M_init)

        grad_native = sinkhorn_native_backward(grad_output, M_init, 20)
        grad_fused = sinkhorn_fused_backward(grad_output, M_init, 20)

        torch.testing.assert_close(grad_fused, grad_native, rtol=1e-2, atol=1e-3)

    def test_backward_numerical_stability(self):
        """Test backward numerical stability."""
        input_logits = torch.randn(256, 4, 4, device='cuda', dtype=torch.float32) * 5
        M_init = torch.exp(input_logits)
        grad_output = torch.randn_like(M_init)

        grad_native = sinkhorn_native_backward(grad_output, M_init, 20)
        grad_fused = sinkhorn_fused_backward(grad_output, M_init, 20)

        assert not torch.isnan(grad_fused).any(), "Fused grad contains NaN"
        assert not torch.isinf(grad_fused).any(), "Fused grad contains Inf"

        torch.testing.assert_close(grad_fused, grad_native, rtol=1e-2, atol=1e-3)

    def test_end_to_end_gradient(self):
        """Test end-to-end gradient computation matches autograd."""
        input_logits = torch.randn(256, 2, 4, 4, device='cuda', dtype=torch.float32, requires_grad=True)
        
        # Native: use autograd to get ground truth gradient
        M_init_native = torch.exp(input_logits)
        M_native = M_init_native.clone()
        for _ in range(20):
            M_native = M_native / M_native.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            M_native = M_native / M_native.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        
        loss_native = M_native.sum()
        loss_native.backward()
        grad_autograd = input_logits.grad.clone()
        input_logits.grad = None
        
        # Fused: forward + backward (backward now includes chain rule)
        with torch.no_grad():
            M_init_fused = torch.exp(input_logits)
        
        grad_output = torch.ones_like(M_init_fused)
        # sinkhorn_fused_backward now returns grad_input directly (includes chain rule)
        grad_fused = sinkhorn_fused_backward(grad_output, M_init_fused, 20, eps=1e-8)
        
        torch.testing.assert_close(grad_fused, grad_autograd, rtol=1e-2, atol=1e-3)


class TestSinkhornPerformance:
    """Benchmark performance of fused Sinkhorn kernels."""

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

    def test_forward_performance(self):
        """Benchmark forward pass."""
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
                'tokens': seq_len * batch_size,
                'native_ms': native_latency,
                'fused_ms': fused_latency,
                'speedup': speedup,
            })

            del input_logits
            torch.cuda.empty_cache()

        self._print_table("FORWARD", results, num_sinkhorn_iterations)

    def test_backward_performance(self):
        """Benchmark backward pass."""
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
            M_init = torch.exp(input_logits)
            grad_output = torch.randn_like(M_init)

            native_latency = self._benchmark_function(
                lambda g, m: sinkhorn_native_backward(g, m, num_sinkhorn_iterations),
                (grad_output, M_init),
            )

            fused_latency = self._benchmark_function(
                lambda g, m: sinkhorn_fused_backward(g, m, num_sinkhorn_iterations),
                (grad_output, M_init),
            )

            speedup = native_latency / fused_latency if fused_latency > 0 else float('inf')

            results.append({
                'shape': f"s={seq_len}, b={batch_size}, hc={hc}",
                'tokens': seq_len * batch_size,
                'native_ms': native_latency,
                'fused_ms': fused_latency,
                'speedup': speedup,
            })

            del input_logits, M_init, grad_output
            torch.cuda.empty_cache()

        self._print_table("BACKWARD", results, num_sinkhorn_iterations)

    def test_combined_fwd_bwd_performance(self):
        """Benchmark combined forward + backward pass."""
        num_sinkhorn_iterations = 20
        results = []

        configs = [
            (1024, 2, 4),
            (2048, 2, 4),
            (4096, 2, 4),
            (1024, 4, 8),
        ]

        for seq_len, batch_size, hc in configs:
            input_logits = torch.randn(
                seq_len, batch_size, hc, hc,
                device='cuda', dtype=torch.float32
            )

            def native_fwd_bwd(x):
                M_init = torch.exp(x)
                M = sinkhorn_native_forward(x, num_sinkhorn_iterations)
                grad_output = torch.ones_like(M)
                return sinkhorn_native_backward(grad_output, M_init, num_sinkhorn_iterations)

            def fused_fwd_bwd(x):
                M_init = torch.exp(x)
                M = sinkhorn_fused_forward(x, num_sinkhorn_iterations)
                grad_output = torch.ones_like(M)
                return sinkhorn_fused_backward(grad_output, M_init, num_sinkhorn_iterations)

            native_latency = self._benchmark_function(native_fwd_bwd, (input_logits,))
            fused_latency = self._benchmark_function(fused_fwd_bwd, (input_logits,))

            speedup = native_latency / fused_latency if fused_latency > 0 else float('inf')

            results.append({
                'shape': f"s={seq_len}, b={batch_size}, hc={hc}",
                'tokens': seq_len * batch_size,
                'native_ms': native_latency,
                'fused_ms': fused_latency,
                'speedup': speedup,
            })

            del input_logits
            torch.cuda.empty_cache()

        self._print_table("FWD+BWD", results, num_sinkhorn_iterations)

    def _print_table(self, title: str, results: List[Dict], num_iters: int):
        """Print formatted performance table."""
        print(f"\n{'=' * 90}")
        print(f"SINKHORN {title} PERFORMANCE (iterations={num_iters})")
        print(f"{'=' * 90}")
        print(f"{'Shape':<25} | {'Tokens':>10} | {'Native':>12} | {'Fused':>12} | {'Speedup':>10}")
        print(f"{'-' * 25}-+-{'-' * 10}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}")

        for r in results:
            print(
                f"{r['shape']:<25} | "
                f"{r['tokens']:>10} | "
                f"{r['native_ms']:>9.4f} ms | "
                f"{r['fused_ms']:>9.4f} ms | "
                f"{r['speedup']:>9.2f}x"
            )

        print(f"{'=' * 90}")

        speedups = [r['speedup'] for r in results]
        print(f"Speedup: min={min(speedups):.2f}x, max={max(speedups):.2f}x, avg={sum(speedups)/len(speedups):.2f}x\n")


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

    def test_hyper_connection_forward(self):
        """Test HyperConnectionModule forward with fused kernel."""
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

    def test_hyper_connection_backward(self):
        """Test HyperConnectionModule backward with fused kernel."""
        from megatron.core.transformer.hyper_connection import HyperConnectionModule
        from megatron.core.transformer.transformer_config import TransformerConfig

        hidden_size = 256
        num_streams = 4
        seq_len = 64
        batch_size = 2

        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=num_streams,
            mhc_sinkhorn_iterations=10,
            mhc_init_gating_factor=0.01,
            mhc_use_fused_kernel=True,
        )

        module_fused = HyperConnectionModule(config=config, layer_number=1)
        module_fused.cuda()

        config.mhc_use_fused_kernel = False
        module_native = HyperConnectionModule(config=config, layer_number=1)
        module_native.cuda()
        module_native.load_state_dict(module_fused.state_dict())

        # Forward + backward with fused
        x_fused = torch.randn(
            seq_len, batch_size, num_streams * hidden_size,
            device='cuda', dtype=torch.float32, requires_grad=True
        )
        h_pre_f, h_post_f, h_res_f = module_fused.compute_mappings(x_fused)
        loss_fused = h_res_f.sum() + h_pre_f.sum() + h_post_f.sum()
        loss_fused.backward()
        grad_fused = x_fused.grad.clone()

        # Forward + backward with native
        x_native = x_fused.detach().clone().requires_grad_(True)
        h_pre_n, h_post_n, h_res_n = module_native.compute_mappings(x_native)
        loss_native = h_res_n.sum() + h_pre_n.sum() + h_post_n.sum()
        loss_native.backward()
        grad_native = x_native.grad.clone()

        # Gradients should match
        torch.testing.assert_close(grad_fused, grad_native, rtol=1e-2, atol=1e-3)


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

    print("\n[1/3] Forward performance...")
    test.test_forward_performance()

    print("\n[2/3] Backward performance...")
    test.test_backward_performance()

    print("\n[3/3] Combined fwd+bwd performance...")
    test.test_combined_fwd_bwd_performance()

    print("\nBenchmark complete!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
