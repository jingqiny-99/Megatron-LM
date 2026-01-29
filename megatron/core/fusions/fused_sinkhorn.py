# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Fused Sinkhorn-Knopp kernel using TileLang.

This module provides a fused GPU kernel for the Sinkhorn-Knopp algorithm,
which projects a positive matrix onto the Birkhoff polytope (doubly stochastic matrices)
via iterative row and column normalization.

Reference: Eq. (9) in mHC paper - M^{(t)} = T_c(T_r(M^{(t-1)}))
"""

from typing import Optional
import torch
from torch import Tensor

# Flag to check if tilelang is available
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
    FP32 = T.float32

    # TileLang pass configurations for optimization
    pass_configs = tilelang.PassConfigContext(
        [
            tilelang.PassConfig(
                "tl.ConvertForLoopsToSerial", 
                {"check_race": False}
            ),
        ]
    )

    @tilelang.jit(pass_configs=pass_configs)
    def _sinkhorn_kernel_generator(hc: int, sinkhorn_iters: int, eps: float):
        """
        Generate a TileLang Sinkhorn-Knopp kernel.
        
        Args:
            hc: Number of hyper-connection streams (matrix dimension)
            sinkhorn_iters: Number of Sinkhorn iterations
            eps: Small epsilon for numerical stability
        
        Returns:
            TileLang kernel function
        """
        n = T.symbolic("n")  # Number of tokens (batch dimension)
        threads = 64  # Number of threads per block

        @T.prim_func
        def sinkhorn_kernel_(
            input_logits: T.Tensor[(n, hc, hc), FP32],  # Input: raw logits
            output: T.Tensor[(n, hc, hc), FP32],        # Output: doubly stochastic matrix
        ):
            with T.Kernel(n, threads=threads) as i:
                # Allocate shared memory and fragments
                M_frag = T.alloc_fragment((hc, hc), FP32)
                row_sum = T.alloc_fragment(hc, FP32)
                col_sum = T.alloc_fragment(hc, FP32)
                row_max = T.alloc_fragment(hc, FP32)

                # Copy input logits to fragment
                T.copy(input_logits[i, :, :], M_frag)

                # Step 1: Compute M = exp(logits) with numerical stability (row max trick)
                # This implements M^{(0)} = exp(H_res_logits) followed by first row normalization
                # Together, they are equivalent to softmax(-1) + eps
                T.reduce_max(M_frag, row_max, dim=1)
                for j, k in T.Parallel(hc, hc):
                    M_frag[j, k] = T.exp(M_frag[j, k] - row_max[j])
                
                # First row normalization (completes the softmax)
                T.reduce_sum(M_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    M_frag[j, k] = M_frag[j, k] / (row_sum[j] + eps)

                # First column normalization
                T.reduce_sum(M_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    M_frag[j, k] = M_frag[j, k] / (col_sum[k] + eps)

                # Remaining sinkhorn iterations (sinkhorn_iters - 1)
                for _ in T.serial(sinkhorn_iters - 1):
                    # Row normalization: M = M / M.sum(dim=-1)
                    T.reduce_sum(M_frag, row_sum, dim=1)
                    for j, k in T.Parallel(hc, hc):
                        M_frag[j, k] = M_frag[j, k] / (row_sum[j] + eps)
                    
                    # Column normalization: M = M / M.sum(dim=-2)
                    T.reduce_sum(M_frag, col_sum, dim=0)
                    for j, k in T.Parallel(hc, hc):
                        M_frag[j, k] = M_frag[j, k] / (col_sum[k] + eps)

                # Copy result to output
                T.copy(M_frag, output[i, :, :])

        return sinkhorn_kernel_

    # Cache for compiled kernels
    _KERNEL_CACHE = {}

    def _get_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
        """
        Get or create a cached Sinkhorn kernel.
        
        Args:
            hc: Matrix dimension (number of hyper-connection streams)
            sinkhorn_iters: Number of Sinkhorn iterations
            eps: Epsilon for numerical stability
        
        Returns:
            Compiled TileLang kernel
        """
        key = (hc, sinkhorn_iters, eps)
        if key not in _KERNEL_CACHE:
            _KERNEL_CACHE[key] = _sinkhorn_kernel_generator(hc, sinkhorn_iters, eps)
        return _KERNEL_CACHE[key]


def sinkhorn_fused_forward(
    input_logits: Tensor,
    num_iterations: int,
    eps: float = 1e-8,
) -> Tensor:
    """
    Fused Sinkhorn-Knopp forward pass using TileLang kernel.
    
    Projects a positive matrix onto the Birkhoff polytope (doubly stochastic matrices)
    via iterative row and column normalization.
    
    Args:
        input_logits: [..., n, n] - raw logits for residual mixing matrix
            The tensor will be reshaped to (-1, n, n) for kernel execution.
        num_iterations: Number of Sinkhorn iterations (paper uses 20)
        eps: Small epsilon for numerical stability (default: 1e-8)
    
    Returns:
        output: [..., n, n] - doubly stochastic matrix (same shape as input)
    
    Raises:
        RuntimeError: If tilelang is not available
    """
    if not _TILELANG_AVAILABLE:
        raise RuntimeError(
            "TileLang is not available. Please install tilelang to use fused Sinkhorn kernel."
        )
    
    # Save original shape
    original_shape = input_logits.shape
    hc = original_shape[-1]  # Matrix dimension
    
    # Reshape to (num_tokens, hc, hc)
    input_flat = input_logits.reshape(-1, hc, hc).contiguous()
    num_tokens = input_flat.shape[0]
    
    # Ensure float32 for kernel
    input_fp32 = input_flat.float()
    
    # Allocate output tensor
    output = torch.empty_like(input_fp32)
    
    # Get compiled kernel
    kernel = _get_sinkhorn_kernel(hc, num_iterations, eps)
    
    # Execute kernel
    kernel(input_fp32, output)
    
    # Reshape output back to original shape and dtype
    output = output.reshape(original_shape)
    if input_logits.dtype != torch.float32:
        output = output.to(input_logits.dtype)
    
    return output


def sinkhorn_native_forward(
    input_logits: Tensor,
    num_iterations: int,
    eps: float = 1e-8,
) -> Tensor:
    """
    Native PyTorch implementation of Sinkhorn-Knopp forward pass.
    
    This is the reference implementation for comparison and fallback.
    
    Args:
        input_logits: [..., n, n] - raw logits for residual mixing matrix
        num_iterations: Number of Sinkhorn iterations
        eps: Small epsilon for numerical stability
    
    Returns:
        output: [..., n, n] - doubly stochastic matrix
    """
    # M^{(0)} = exp(H_res_logits)
    M = torch.exp(input_logits)
    
    for _ in range(num_iterations):
        # T_r: Row normalization
        M = M / M.sum(dim=-1, keepdim=True).clamp(min=eps)
        # T_c: Column normalization
        M = M / M.sum(dim=-2, keepdim=True).clamp(min=eps)
    
    return M
