# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Fused Sinkhorn-Knopp kernel using TileLang.

This module provides fused GPU kernels for the Sinkhorn-Knopp algorithm,
which projects a positive matrix onto the Birkhoff polytope (doubly stochastic matrices)
via iterative row and column normalization.

Reference: Eq. (9) in mHC paper - M^{(t)} = T_c(T_r(M^{(t-1)}))
"""

import torch
from torch import Tensor

_TILELANG_AVAILABLE = False

try:
    import tilelang
    import tilelang.language as T
    _TILELANG_AVAILABLE = True
except ImportError:
    pass


def is_tilelang_available() -> bool:
    """Check if tilelang is available."""
    return _TILELANG_AVAILABLE


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

    @tilelang.jit(pass_configs=pass_configs)
    def _sinkhorn_forward_kernel_generator(hc: int, sinkhorn_iters: int, eps: float, dtype: str = FP32):
        """Generate forward Sinkhorn kernel."""
        n = T.symbolic("n")
        threads = 64

        @T.prim_func
        def sinkhorn_forward_kernel_(
            input_logits: T.Tensor[(n, hc, hc), dtype],
            output: T.Tensor[(n, hc, hc), dtype],
        ):
            with T.Kernel(n, threads=threads) as i:
                M_frag = T.alloc_fragment((hc, hc), dtype)
                row_sum = T.alloc_fragment(hc, dtype)
                col_sum = T.alloc_fragment(hc, dtype)
                row_max = T.alloc_fragment(hc, dtype)

                T.copy(input_logits[i, :, :], M_frag)

                # exp with numerical stability (row max trick)
                T.reduce_max(M_frag, row_max, dim=1)
                for j, k in T.Parallel(hc, hc):
                    M_frag[j, k] = T.exp(M_frag[j, k] - row_max[j])

                # First row normalization
                T.reduce_sum(M_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    M_frag[j, k] = M_frag[j, k] / (row_sum[j] + eps)

                # First col normalization
                T.reduce_sum(M_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    M_frag[j, k] = M_frag[j, k] / (col_sum[k] + eps)

                # Remaining iterations
                for _ in T.serial(sinkhorn_iters - 1):
                    T.reduce_sum(M_frag, row_sum, dim=1)
                    for j, k in T.Parallel(hc, hc):
                        M_frag[j, k] = M_frag[j, k] / (row_sum[j] + eps)

                    T.reduce_sum(M_frag, col_sum, dim=0)
                    for j, k in T.Parallel(hc, hc):
                        M_frag[j, k] = M_frag[j, k] / (col_sum[k] + eps)

                T.copy(M_frag, output[i, :, :])

        return sinkhorn_forward_kernel_

    @tilelang.jit(pass_configs=pass_configs)
    def _sinkhorn_backward_bwd_kernel_generator(hc: int, sinkhorn_iters: int, eps: float, dtype: str = FP32):
        """
        Generate backward kernel that computes gradients.
        
        Reconstructs intermediate states from saved row_sum/col_sum:
        - M_row[t] = M_col[t] * col_sum[t]  (reverse col normalization)
        - M_in[t] = M_row[t] * row_sum[t]   (reverse row normalization)
        
        Backward formulas (optimized to divide first, then subtract):
        - For col normalization y = x / col_sum(x):
            grad_x = (grad_y - (grad_y * y).sum(dim=0)) / col_sum
            Optimized: grad_y' = grad_y / col_sum, then grad_x = grad_y' - (grad_y' * y).sum(dim=0)
        - For row normalization y = x / row_sum(x):
            grad_x = (grad_y - (grad_y * y).sum(dim=1)) / row_sum
            Optimized: grad_y' = grad_y / row_sum, then grad_x = grad_y' - (grad_y' * y).sum(dim=1)
        """
        n = T.symbolic("n")
        threads = 64

        @T.prim_func
        def sinkhorn_backward_bwd_kernel(
            grad_output: T.Tensor[(n, hc, hc), dtype],
            M_init: T.Tensor[(n, hc, hc), dtype],
            grad_input: T.Tensor[(n, hc, hc), dtype],
        ):
            with T.Kernel(n, threads=128) as bn:
                M_median = T.alloc_shared((sinkhorn_iters * 2, hc, hc), dtype)
                row_sums = T.alloc_shared((sinkhorn_iters, hc), dtype)
                col_sums = T.alloc_shared((sinkhorn_iters, hc), dtype)

                M = T.alloc_fragment((hc, hc), dtype)
                grad =  T.alloc_fragment((hc, hc), dtype)
                row_sum = T.alloc_fragment(hc, dtype)
                col_sum = T.alloc_fragment(hc, dtype)
                temp = T.alloc_fragment((hc, hc), dtype)
                T.copy(M_init[bn, :, :], M)
                
                for t in T.Serial(sinkhorn_iters):
                    T.copy(M, M_median[2 * t, :, :])
                    T.reduce_sum(M, row_sum, dim=1)
                    T.copy(row_sum, row_sums[t, :])
                    for i, j in T.Parallel(hc, hc):
                        M[i, j] = M[i, j] / (row_sum[i] + eps)
                    T.copy(M, M_median[2 * t + 1, :, :])
                    T.reduce_sum(M, col_sum, dim=0)
                    T.copy(col_sum, col_sums[t, :])
                    for i, j in T.Parallel(hc, hc):
                        M[i, j] = M[i, j] / (col_sum[j] + eps)
                T.copy(grad_output[bn, :, :], grad)

                for t_rev in T.Serial(sinkhorn_iters):
                    t = sinkhorn_iters - t_rev - 1
                    for i, j in T.Parallel(hc, hc):
                        grad[i, j] = grad[i, j] / (col_sums[t, j] + eps)
                        temp[i, j] = grad[i, j] * M[i, j]
                    T.reduce_sum(temp, col_sum, dim=0)
                    for i, j in T.Parallel(hc, hc):
                        grad[i, j] = grad[i, j] - col_sum[j]
                    T.copy(M_median[2 * t + 1, :, :], M)

                    for i, j in T.Parallel(hc, hc): 
                        grad[i, j] = grad[i, j] / (row_sums[t, i] + eps)
                        temp[i, j] = grad[i, j] * M[i, j]
                    
                    T.reduce_sum(temp, row_sum, dim=1)
                    for i, j in T.Parallel(hc, hc):
                        grad[i, j] = grad[i, j] - row_sum[i]
                    
                    T.copy(M_median[2 * t, :, :], M)

                for i, j in T.Parallel(hc, hc):
                    grad[i, j] = grad[i, j] * M[i, j]
                
                T.copy(grad, grad_input[bn, :, :])

        return sinkhorn_backward_bwd_kernel

    # ==================== Kernel Cache ====================

    _FORWARD_KERNEL_CACHE = {}
    _BACKWARD_FWD_KERNEL_CACHE = {}
    _BACKWARD_BWD_KERNEL_CACHE = {}

    def _get_forward_kernel(hc: int, sinkhorn_iters: int, eps: float, dtype: str = FP32):
        """Get or create a cached forward kernel."""
        key = (hc, sinkhorn_iters, eps, dtype)
        if key not in _FORWARD_KERNEL_CACHE:
            _FORWARD_KERNEL_CACHE[key] = _sinkhorn_forward_kernel_generator(hc, sinkhorn_iters, eps, dtype)
        return _FORWARD_KERNEL_CACHE[key]

    def _get_backward_bwd_kernel(hc: int, sinkhorn_iters: int, eps: float, dtype: str = FP32):
        """Get or create a cached backward kernel."""
        key = (hc, sinkhorn_iters, eps, dtype)
        if key not in _BACKWARD_BWD_KERNEL_CACHE:
            _BACKWARD_BWD_KERNEL_CACHE[key] = _sinkhorn_backward_bwd_kernel_generator(hc, sinkhorn_iters, eps, dtype)
        return _BACKWARD_BWD_KERNEL_CACHE[key]


def sinkhorn_fused_forward(
    input_logits: Tensor,
    num_iterations: int,
    eps: float = 1e-8,
) -> Tensor:
    """
    Fused Sinkhorn-Knopp forward pass using TileLang kernel.

    Args:
        input_logits: [..., n, n] - raw logits for residual mixing matrix
        num_iterations: Number of Sinkhorn iterations (paper uses 20)
        eps: Small epsilon for numerical stability (default: 1e-8)

    Returns:
        output: [..., n, n] - doubly stochastic matrix (same shape as input)
    """
    if not _TILELANG_AVAILABLE:
        raise RuntimeError("TileLang is not available.")

    original_shape = input_logits.shape
    hc = original_shape[-1]
    dtype_str = _TORCH_DTYPE_TO_TL[input_logits.dtype]

    input_flat = input_logits.reshape(-1, hc, hc).contiguous()

    output = torch.empty_like(input_flat)

    kernel = _get_forward_kernel(hc, num_iterations, eps, dtype_str)
    kernel(input_flat, output)

    return output.reshape(original_shape)


def sinkhorn_fused_backward(
    grad_output: Tensor,
    M_init: Tensor,
    num_iterations: int,
    eps: float = 1e-8,
) -> Tensor:
    """
    Fused Sinkhorn-Knopp backward pass using TileLang kernel.
    
    Computes dL/dH where H is the input logits to Sinkhorn (H_res_logits).
    The kernel fuses the entire backward pass including the chain rule
    (grad_input = grad_M_init * M_init) for better efficiency.
    
    Memory efficient: only stores row_sum and col_sum vectors (O(n*iters*hc))
    instead of full intermediate matrices (O(n*iters*hc*hc)).

    Args:
        grad_output: [..., n, n] - gradient w.r.t. output (dL/dM_final)
        M_init: [..., n, n] - exp(H_res_logits), the initial M before Sinkhorn iterations
        num_iterations: Number of Sinkhorn iterations
        eps: Small epsilon for numerical stability (default: 1e-8)

    Returns:
        grad_input: [..., n, n] - gradient w.r.t. H_res_logits (dL/dH)
    """
    if not _TILELANG_AVAILABLE:
        raise RuntimeError("TileLang is not available.")

    original_shape = grad_output.shape
    hc = original_shape[-1]
    dtype_str = _TORCH_DTYPE_TO_TL[grad_output.dtype]
    
    grad_output_flat = grad_output.reshape(-1, hc, hc).contiguous()
    M_init_flat = M_init.reshape(-1, hc, hc).contiguous()

    grad_input = torch.empty_like(M_init_flat)
    bwd_kernel = _get_backward_bwd_kernel(hc, num_iterations, eps, dtype_str)
    bwd_kernel(grad_output_flat, M_init_flat, grad_input)

    return grad_input.reshape(original_shape)


def sinkhorn_native_forward(
    input_logits: Tensor,
    num_iterations: int,
    eps: float = 1e-8,
) -> Tensor:
    """
    Native PyTorch implementation of Sinkhorn-Knopp forward pass.

    Args:
        input_logits: [..., n, n] - raw logits for residual mixing matrix
        num_iterations: Number of Sinkhorn iterations
        eps: Small epsilon for numerical stability

    Returns:
        output: [..., n, n] - doubly stochastic matrix
    """
    M = torch.exp(input_logits)

    for _ in range(num_iterations):
        M = M / M.sum(dim=-1, keepdim=True).clamp(min=eps)
        M = M / M.sum(dim=-2, keepdim=True).clamp(min=eps)

    return M


def sinkhorn_native_backward(
    grad_output: Tensor,
    M_init: Tensor,
    num_iterations: int,
    eps: float = 1e-8,
) -> Tensor:
    """
    Native PyTorch implementation of Sinkhorn-Knopp backward pass.
    Uses autograd for gradient computation via recomputation.
    
    Computes dL/dH where H is the input logits to Sinkhorn (H_res_logits).
    Includes the chain rule (grad_input = grad_M_init * M_init).

    Args:
        grad_output: [..., n, n] - gradient w.r.t. output (dL/dM_final)
        M_init: [..., n, n] - exp(H_res_logits), the initial M before Sinkhorn iterations
        num_iterations: Number of Sinkhorn iterations
        eps: Small epsilon for numerical stability

    Returns:
        grad_input: [..., n, n] - gradient w.r.t. H_res_logits (dL/dH)
    """
    with torch.enable_grad():
        M_input = M_init.detach().requires_grad_(True)

        M_current = M_input
        for _ in range(num_iterations):
            M_current = M_current / M_current.sum(dim=-1, keepdim=True).clamp(min=eps)
            M_current = M_current / M_current.sum(dim=-2, keepdim=True).clamp(min=eps)

        grad_M_init, = torch.autograd.grad(
            outputs=M_current,
            inputs=M_input,
            grad_outputs=grad_output,
            create_graph=False,
            retain_graph=False,
        )

    # Apply chain rule: grad_input = grad_M_init * M_init
    grad_input = grad_M_init * M_init

    return grad_input

if __name__ == "__main__":
    
    input_logits = torch.randn(2, 2, 4, 4, dtype=torch.float32, requires_grad=True).cuda()
    M_init = torch.exp(input_logits)
    grad_output = torch.randn_like(M_init)

    grad_native = sinkhorn_native_backward(grad_output, M_init, 20)
    grad_fused = sinkhorn_fused_backward(grad_output, M_init, 20)

    print(grad_native)  
    print(grad_fused)