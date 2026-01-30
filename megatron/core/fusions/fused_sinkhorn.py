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

    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }

    # ==================== Forward Kernel ====================

    @tilelang.jit(pass_configs=pass_configs)
    def _sinkhorn_forward_kernel_generator(hc: int, sinkhorn_iters: int, eps: float):
        """Generate forward Sinkhorn kernel."""
        n = T.symbolic("n")
        threads = 64

        @T.prim_func
        def sinkhorn_forward_kernel_(
            input_logits: T.Tensor[(n, hc, hc), FP32],
            output: T.Tensor[(n, hc, hc), FP32],
        ):
            with T.Kernel(n, threads=threads) as i:
                M_frag = T.alloc_fragment((hc, hc), FP32)
                row_sum = T.alloc_fragment(hc, FP32)
                col_sum = T.alloc_fragment(hc, FP32)
                row_max = T.alloc_fragment(hc, FP32)

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

    # ==================== Backward Kernels ====================
    # Split into two kernels:
    # 1. Forward kernel that saves row_sum/col_sum for each iteration
    # 2. Backward kernel that reconstructs states and computes gradients

    @tilelang.jit(pass_configs=pass_configs)
    def _sinkhorn_backward_fwd_kernel_generator(hc: int, sinkhorn_iters: int, eps: float):
        """
        Generate kernel for forward pass that saves row_sum and col_sum.
        Memory efficient: only stores 1D vectors instead of full matrices.
        
        Storage format:
        - row_sums: (n, sinkhorn_iters, hc) - row sums for each iteration
        - col_sums: (n, sinkhorn_iters, hc) - col sums for each iteration
        - M_final: (n, hc, hc) - final output after all iterations
        """
        n = T.symbolic("n")
        threads = 64

        @T.prim_func
        def sinkhorn_backward_fwd_kernel_(
            M_init: T.Tensor[(n, hc, hc), FP32],
            row_sums: T.Tensor[(n, sinkhorn_iters, hc), FP32],
            col_sums: T.Tensor[(n, sinkhorn_iters, hc), FP32],
            M_final: T.Tensor[(n, hc, hc), FP32],
        ):
            with T.Kernel(n, threads=threads) as i:
                M_frag = T.alloc_fragment((hc, hc), FP32)
                row_sum = T.alloc_fragment(hc, FP32)
                col_sum = T.alloc_fragment(hc, FP32)

                T.copy(M_init[i, :, :], M_frag)

                for t in T.serial(sinkhorn_iters):
                    # Row normalization: M_row = M_in / row_sum
                    T.reduce_sum(M_frag, row_sum, dim=1)
                    # Save row_sum to global memory (with eps already added)
                    for j in T.Parallel(hc):
                        row_sums[i, t, j] = row_sum[j] + eps
                    for j, k in T.Parallel(hc, hc):
                        M_frag[j, k] = M_frag[j, k] / (row_sum[j] + eps)

                    # Col normalization: M_col = M_row / col_sum
                    T.reduce_sum(M_frag, col_sum, dim=0)
                    # Save col_sum to global memory (with eps already added)
                    for j in T.Parallel(hc):
                        col_sums[i, t, j] = col_sum[j] + eps
                    for j, k in T.Parallel(hc, hc):
                        M_frag[j, k] = M_frag[j, k] / (col_sum[k] + eps)

                # Save final output
                T.copy(M_frag, M_final[i, :, :])

        return sinkhorn_backward_fwd_kernel_

    @tilelang.jit(pass_configs=pass_configs)
    def _sinkhorn_backward_bwd_kernel_generator(hc: int, sinkhorn_iters: int, eps: float, BLOCK_N: int):
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
        def sinkhorn_backward_bwd_kernel_(
            grad_output: T.Tensor[(n, hc, hc), FP32],
            M_init: T.Tensor[(n, hc, hc), FP32],
            M_final: T.Tensor[(n, hc, hc), FP32],
            row_sums: T.Tensor[(n, sinkhorn_iters, hc), FP32],
            col_sums: T.Tensor[(n, sinkhorn_iters, hc), FP32],
            grad_input: T.Tensor[(n, hc, hc), FP32],
        ):
            with T.kernel(T.ceildiv(n, BLOCK_N), threads=BLOCK_N) as bn:
                tn = T.get_thread_binding(0)  
                grad_frag = T.alloc_fragment((hc, hc), FP32)
                T.copy(grad_output[bn, :, :], grad_frag)
                M = T.alloc_shared((BLOCK_N, hc, hc), FP32)
                M_init_shared = T.alloc_shared((BLOCK_N, hc, hc), FP32)
                row_sums_shared = T.alloc_shared((BLOCK_N, sinkhorn_iters, hc), FP32)
                col_sums_shared = T.alloc_shared((BLOCK_N, sinkhorn_iters, hc), FP32)
                T.copy(M_final[bn * BLOCK_N + tn, :, :], M[tn, :, :])
                T.copy(row_sums[bn * BLOCK_N + tn, :, :], row_sums_shared[tn, :, :])
                T.copy(col_sums[bn * BLOCK_N + tn, :, :], col_sums_shared[tn, :, :])
                T.copy(M_init[bn * BLOCK_N + tn, :, :], M_init_shared[tn, :, :])
                dot_prod = T.alloc_fragment(hc, FP32)
                temp = T.alloc_fragment((hc, hc), FP32)

                # Backward pass through Sinkhorn iterations
                for t_rev in T.serial(sinkhorn_iters):
                    t = sinkhorn_iters - 1 - t_rev
                    
                    for j, k in T.Parallel(hc, hc):
                        grad_frag[j, k] = grad_frag[j, k] / col_sums_shared[tn, t, k]
                    # Step 2: dot_prod = (grad_frag * M_col).sum(dim=0)
                    for j, k in T.Parallel(hc, hc):
                        temp[j, k] = grad_frag[j, k] * M[tn, j, k]

                    T.reduce_sum(temp, dot_prod, dim=1)
                    # Step 3: grad_frag = grad_frag - dot_prod
                    for j, k in T.Parallel(hc, hc):
                        grad_frag[j, k] = grad_frag[j, k] - dot_prod[k]

                    for j, k in T.Parallel(hc, hc):
                        M[tn, j, k] = M[tn, j, k] * col_sums_shared[tn, t, k]
                    # ---- Backward through row normalization (optimized) ----
                    # Step 1: grad_frag = grad_frag / row_sum
                    for j, k in T.Parallel(hc, hc):
                        grad_frag[j, k] = grad_frag[j, k] / row_sums_shared[tn, t, j]
                    # Step 2: dot_prod = (grad_frag * M_row).sum(dim=1)
                    for j, k in T.Parallel(hc, hc):
                        temp[j, k] = grad_frag[j, k] * M[tn, j, k]

                    T.reduce_sum(temp, dot_prod, dim=1)
                    # Step 3: grad_frag = grad_frag - dot_prod
                    for j, k in T.Parallel(hc, hc):
                        grad_frag[j, k] = grad_frag[j, k] - dot_prod[j]

                    for j, k in T.Parallel(hc, hc):
                        M[tn, j, k] = M[tn, j, k] * row_sums_shared[tn, t, j]

                for j in T.serial(hc):
                    for k in T.serial(hc):
                        grad_frag[j, k] = grad_frag[j, k] * M_init_shared[tn, j, k]

                T.copy(grad_frag, grad_input[bn * BLOCK_N + tn, :, :])

        return sinkhorn_backward_bwd_kernel_

    # ==================== Kernel Cache ====================

    _FORWARD_KERNEL_CACHE = {}
    _BACKWARD_FWD_KERNEL_CACHE = {}
    _BACKWARD_BWD_KERNEL_CACHE = {}

    def _get_forward_kernel(hc: int, sinkhorn_iters: int, eps: float):
        """Get or create a cached forward kernel."""
        key = (hc, sinkhorn_iters, eps)
        if key not in _FORWARD_KERNEL_CACHE:
            _FORWARD_KERNEL_CACHE[key] = _sinkhorn_forward_kernel_generator(hc, sinkhorn_iters, eps)
        return _FORWARD_KERNEL_CACHE[key]

    def _get_backward_fwd_kernel(hc: int, sinkhorn_iters: int, eps: float):
        """Get or create a cached backward forward kernel."""
        key = (hc, sinkhorn_iters, eps)
        if key not in _BACKWARD_FWD_KERNEL_CACHE:
            _BACKWARD_FWD_KERNEL_CACHE[key] = _sinkhorn_backward_fwd_kernel_generator(hc, sinkhorn_iters, eps)
        return _BACKWARD_FWD_KERNEL_CACHE[key]

    def _get_backward_bwd_kernel(hc: int, sinkhorn_iters: int, eps: float):
        """Get or create a cached backward kernel."""
        key = (hc, sinkhorn_iters, eps)
        if key not in _BACKWARD_BWD_KERNEL_CACHE:
            _BACKWARD_BWD_KERNEL_CACHE[key] = _sinkhorn_backward_bwd_kernel_generator(hc, sinkhorn_iters, eps)
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

    input_flat = input_logits.reshape(-1, hc, hc).contiguous()
    input_fp32 = input_flat.float()

    output = torch.empty_like(input_fp32)

    kernel = _get_forward_kernel(hc, num_iterations, eps)
    kernel(input_fp32, output)

    output = output.reshape(original_shape)
    if input_logits.dtype != torch.float32:
        output = output.to(input_logits.dtype)

    return output


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
    
    grad_output_flat = grad_output.reshape(-1, hc, hc).contiguous().float()
    M_init_flat = M_init.reshape(-1, hc, hc).contiguous().float()
    num_tokens = grad_output_flat.shape[0]

    # Allocate memory for row_sum and col_sum (memory efficient)
    row_sums = torch.empty(num_tokens, num_iterations, hc,
                           device=grad_output.device, dtype=torch.float32)
    col_sums = torch.empty(num_tokens, num_iterations, hc,
                           device=grad_output.device, dtype=torch.float32)
    M_final = torch.empty_like(M_init_flat)

    # Step 1: Forward pass to save row_sums, col_sums, and M_final
    fwd_kernel = _get_backward_fwd_kernel(hc, num_iterations, eps)
    fwd_kernel(M_init_flat, row_sums, col_sums, M_final)

    # Step 2: Backward pass using saved sums
    grad_input = torch.empty_like(grad_output_flat)
    bwd_kernel = _get_backward_bwd_kernel(hc, num_iterations, eps)
    bwd_kernel(grad_output_flat, M_init_flat, M_final, row_sums, col_sums, grad_input)

    grad_input = grad_input.reshape(original_shape)
    if grad_output.dtype != torch.float32:
        grad_input = grad_input.to(grad_output.dtype)

    return grad_input


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
