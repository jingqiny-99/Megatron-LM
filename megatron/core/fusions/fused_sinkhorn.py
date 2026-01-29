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

    # ==================== Backward Kernel ====================

    @tilelang.jit(pass_configs=pass_configs)
    def _sinkhorn_backward_kernel_generator(hc: int, sinkhorn_iters: int, eps: float):
        """
        Generate backward Sinkhorn kernel.
        
        The backward pass:
        1. Recomputes forward pass to get all intermediate states
        2. Backpropagates gradients through each normalization step
        3. Applies chain rule: grad_input = grad_M_init * M_init
        
        For row normalization y = x / row_sum(x):
            grad_x = (grad_y - (grad_y * y).sum(-1)) / row_sum
        
        For col normalization y = x / col_sum(x):
            grad_x = (grad_y - (grad_y * y).sum(-2)) / col_sum
        """
        n = T.symbolic("n")
        threads = 64

        @T.prim_func
        def sinkhorn_backward_kernel_(
            grad_output: T.Tensor[(n, hc, hc), FP32],  # dL/dM_final
            M_init: T.Tensor[(n, hc, hc), FP32],       # exp(H_res_logits)
            grad_input: T.Tensor[(n, hc, hc), FP32],   # output: dL/dH (includes chain rule)
        ):
            with T.Kernel(n, threads=threads) as i:
                # Allocate storage for intermediate states
                M_row_states = T.alloc_fragment((sinkhorn_iters, hc, hc), FP32)
                M_col_states = T.alloc_fragment((sinkhorn_iters, hc, hc), FP32)
                
                M_frag = T.alloc_fragment((hc, hc), FP32)
                M_init_frag = T.alloc_fragment((hc, hc), FP32)
                grad_frag = T.alloc_fragment((hc, hc), FP32)
                row_sum = T.alloc_fragment(hc, FP32)
                col_sum = T.alloc_fragment(hc, FP32)
                dot_prod = T.alloc_fragment(hc, FP32)
                temp = T.alloc_fragment((hc, hc), FP32)

                # Load M_init for later use in chain rule
                T.copy(M_init[i, :, :], M_init_frag)
                
                # ============ Forward pass to save intermediate states ============
                T.copy(M_init_frag, M_frag)

                for t in T.serial(sinkhorn_iters):
                    # Row normalization
                    T.reduce_sum(M_frag, row_sum, dim=1)
                    for j, k in T.Parallel(hc, hc):
                        M_frag[j, k] = M_frag[j, k] / (row_sum[j] + eps)
                    
                    # Save row_norm result
                    for j, k in T.Parallel(hc, hc):
                        M_row_states[t, j, k] = M_frag[j, k]
                    
                    # Col normalization
                    T.reduce_sum(M_frag, col_sum, dim=0)
                    for j, k in T.Parallel(hc, hc):
                        M_frag[j, k] = M_frag[j, k] / (col_sum[k] + eps)
                    
                    # Save col_norm result
                    for j, k in T.Parallel(hc, hc):
                        M_col_states[t, j, k] = M_frag[j, k]

                # ============ Backward pass ============
                T.copy(grad_output[i, :, :], grad_frag)

                # Iterate backwards through Sinkhorn iterations
                for t_rev in T.serial(sinkhorn_iters):
                    t = sinkhorn_iters - 1 - t_rev
                    
                    # ---- Backward through col normalization ----
                    for j, k in T.Parallel(hc, hc):
                        M_frag[j, k] = M_col_states[t, j, k]
                    
                    for j, k in T.Parallel(hc, hc):
                        temp[j, k] = grad_frag[j, k] * M_frag[j, k]
                    T.reduce_sum(temp, dot_prod, dim=0)
                    
                    for j, k in T.Parallel(hc, hc):
                        M_frag[j, k] = M_row_states[t, j, k]
                    T.reduce_sum(M_frag, col_sum, dim=0)
                    
                    for j, k in T.Parallel(hc, hc):
                        grad_frag[j, k] = (grad_frag[j, k] - dot_prod[k]) / (col_sum[k] + eps)
                    
                    # ---- Backward through row normalization ----
                    for j, k in T.Parallel(hc, hc):
                        M_frag[j, k] = M_row_states[t, j, k]
                    
                    for j, k in T.Parallel(hc, hc):
                        temp[j, k] = grad_frag[j, k] * M_frag[j, k]
                    T.reduce_sum(temp, dot_prod, dim=1)
                    
                    # Get M_in (input to row_norm): M_col_states[t-1] if t > 0, else M_init
                    for j, k in T.Parallel(hc, hc):
                        if t > 0:
                            temp[j, k] = M_col_states[t - 1, j, k]
                        else:
                            temp[j, k] = M_init_frag[j, k]
                    T.reduce_sum(temp, row_sum, dim=1)
                    
                    for j, k in T.Parallel(hc, hc):
                        grad_frag[j, k] = (grad_frag[j, k] - dot_prod[j]) / (row_sum[j] + eps)

                # Apply chain rule: grad_input = grad_M_init * M_init
                # Since M_init = exp(H), d(exp(x))/dx = exp(x) = M_init
                for j, k in T.Parallel(hc, hc):
                    grad_frag[j, k] = grad_frag[j, k] * M_init_frag[j, k]

                # Output grad_input (dL/dH)
                T.copy(grad_frag, grad_input[i, :, :])

        return sinkhorn_backward_kernel_

    # ==================== Kernel Cache ====================

    _FORWARD_KERNEL_CACHE = {}
    _BACKWARD_KERNEL_CACHE = {}

    def _get_forward_kernel(hc: int, sinkhorn_iters: int, eps: float):
        """Get or create a cached forward kernel."""
        key = (hc, sinkhorn_iters, eps)
        if key not in _FORWARD_KERNEL_CACHE:
            _FORWARD_KERNEL_CACHE[key] = _sinkhorn_forward_kernel_generator(hc, sinkhorn_iters, eps)
        return _FORWARD_KERNEL_CACHE[key]

    def _get_backward_kernel(hc: int, sinkhorn_iters: int, eps: float):
        """Get or create a cached backward kernel."""
        key = (hc, sinkhorn_iters, eps)
        if key not in _BACKWARD_KERNEL_CACHE:
            _BACKWARD_KERNEL_CACHE[key] = _sinkhorn_backward_kernel_generator(hc, sinkhorn_iters, eps)
        return _BACKWARD_KERNEL_CACHE[key]


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

    grad_input = torch.empty_like(grad_output_flat)

    kernel = _get_backward_kernel(hc, num_iterations, eps)
    kernel(grad_output_flat, M_init_flat, grad_input)

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
    # Since M_init = exp(H), d(exp(x))/dx = exp(x) = M_init
    grad_input = grad_M_init * M_init

    return grad_input
