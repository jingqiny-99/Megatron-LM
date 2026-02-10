# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Fused H_post BDA (Bias-Dropout-Add) kernel for mHC.

This module provides fused GPU kernels for the H_post expansion and bias-dropout-add
operations in mHC (Manifold-Constrained Hyper-Connections).

The computation flow is:
    1. mixed = H_res @ original_residual (apply_h_res)
    2. x_expanded = H_post^T @ layer_output (apply_h_post)
    3. bias_expanded = H_post^T @ bias (if bias is not None)
    4. output = dropout(x_expanded + bias_expanded) + mixed (bias-dropout-add)

This kernel fuses all operations into a single kernel for better efficiency.
"""

import torch
from torch import Tensor
from typing import Optional, Tuple
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

    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }

    # ==================== Forward Kernel ====================

    # TODO: Implement forward kernel
    # Input shapes:
    #   - h_res: [sb, n, n] - residual mixing matrix (flattened s*b)
    #   - original_residual: [sb, n, C] - n-stream hidden states
    #   - h_post: [sb, n] - expansion weights
    #   - x: [sb, C] - layer output
    #   - bias: [C] or None - optional bias tensor
    # Output shapes:
    #   - output: [sb, n, C] - final output after all operations

    @tilelang.jit(pass_configs=pass_configs)
    def _h_post_bda_forward_kernel_generator(sb: int, n: int, C: int):
        """Generate forward H_post BDA kernel."""
        threads = 128
        BLOCK_SB = 16
        EXPECTED_BLOCK_C = 1024
        BLOCK_C = math.gcd(EXPECTED_BLOCK_C, C)
        @T.prim_func
        def h_post_bda_forward_kernel_(
            h_res: T.Tensor[(sb, n, n), FP32],
            original_residual: T.Tensor[(sb, n, C), FP32],
            h_post: T.Tensor[(sb, n), FP32],
            x: T.Tensor[(sb, C), FP32],
            bias: T.Tensor[(C,), FP32],
            output: T.Tensor[(sb, n, C), FP32],
        ):
            # no need for block div n, but need to block div sb and C. Notice that the output is in shape of (sb, n, C) 
            with T.Kernel(sb, threads=threads) as i:
                h_res_local = T.alloc_fragment((n, n), FP32)
                original_residual_local = T.alloc_fragment((n, BLOCK_C), FP32)
                temp = T.alloc_fragment((n, BLOCK_C), FP32)
                h_post_local = T.alloc_fragment((n), FP32)
                x_local = T.alloc_fragment((BLOCK_C), FP32)
                bias_local = T.alloc_fragment((BLOCK_C), FP32)
                output_local = T.alloc_fragment((n, BLOCK_C), FP32)
                temp = T.alloc_fragment((n, BLOCK_C, n ), FP32)
                T.copy(h_res[i, :, :], h_res_local)
                T.copy(h_post[i, :], h_post_local)

                for pid_c in T.Pipelined(C // BLOCK_C, num_stages=3): 
                    c_idx = pid_c * BLOCK_C
                    T.copy(original_residual[i, 0, c_idx], original_residual_local)
                    T.copy(x[i, c_idx], x_local)
                    T.copy(bias[c_idx], bias_local)
                    for j, k, l in T.Parallel(n, n, BLOCK_C):
                        temp[j, l, k] = h_res_local[j, k] * original_residual_local[k, l]
                    T.reduce_sum(temp, output_local, dim=2, clear=True)
                    for j, k in T.Parallel(n, BLOCK_C):
                        output_local[j, k] += h_post_local[j] * (x_local[k] + bias_local[k])
                    T.copy(output_local, output[i, 0, c_idx])


        return h_post_bda_forward_kernel_

    # ==================== Backward Kernel ====================

    # Backward gradient formulas:
    #   grad_h_res[i, j, m] = Σ_k (grad_output[i, j, k] × original_residual[i, m, k])
    #   grad_original_residual[i, m, k] = Σ_j (grad_output[i, j, k] × h_res[i, j, m])
    #   grad_h_post[i, j] = Σ_k (grad_output[i, j, k] × (x[i, k] + bias[k]))
    #   grad_x[i, k] = Σ_j (grad_output[i, j, k] × h_post[i, j])
    #   grad_bias[k] = Σ_{i,j} (grad_output[i, j, k] × h_post[i, j])  # accumulated outside kernel

    @tilelang.jit(pass_configs=pass_configs)
    def _h_post_bda_backward_kernel_generator(sb: int, n: int, C: int):
        """Generate backward H_post BDA kernel.
        
        Computes gradients for h_res, original_residual, h_post, and x.
        grad_bias is computed by summing grad_x across batch dimension outside kernel.
        """
        threads = 128
        BLOCK_SB = 16
        EXPECTED_BLOCK_C = 1024
        BLOCK_C = math.gcd(EXPECTED_BLOCK_C, C)

        @T.prim_func
        def h_post_bda_backward_kernel_(
            grad_output: T.Tensor[(sb, n, C), FP32],
            h_res: T.Tensor[(sb, n, n), FP32],
            original_residual: T.Tensor[(sb, n, C), FP32],
            h_post: T.Tensor[(sb, n), FP32],
            x: T.Tensor[(sb, C), FP32],
            bias: T.Tensor[(C,), FP32],
            grad_h_res: T.Tensor[(sb, n, n), FP32],
            grad_original_residual: T.Tensor[(sb, n, C), FP32],
            grad_h_post: T.Tensor[(sb, n), FP32],
            grad_x: T.Tensor[(sb, C), FP32],
        ):
            with T.Kernel(sb, threads=threads) as i:
                # Allocate local fragments
                grad_output_local = T.alloc_fragment((n, BLOCK_C), FP32)
                h_res_local = T.alloc_fragment((n, n), FP32)
                original_residual_local = T.alloc_fragment((n, BLOCK_C), FP32)
                h_post_local = T.alloc_fragment((n,), FP32)
                x_local = T.alloc_fragment((BLOCK_C,), FP32)
                bias_local = T.alloc_fragment((BLOCK_C,), FP32)
                
                # Output fragments
                grad_h_res_local = T.alloc_fragment((n, n), FP32)
                grad_original_residual_local = T.alloc_fragment((n, BLOCK_C), FP32)
                grad_h_post_local = T.alloc_fragment((n,), FP32)
                grad_x_local = T.alloc_fragment((BLOCK_C,), FP32)
                
                # Temp fragments for matmul
                temp_h_res = T.alloc_fragment((n, n, BLOCK_C), FP32)
                temp_orig_res = T.alloc_fragment((n, BLOCK_C, n), FP32)
                temp_h_post = T.alloc_fragment((n, BLOCK_C), FP32)
                temp_x = T.alloc_fragment((n, BLOCK_C), FP32)
                
                # Load h_res and h_post (constant across C blocks)
                T.copy(h_res[i, 0, 0], h_res_local)
                T.copy(h_post[i, 0], h_post_local)
                
                # Initialize grad_h_res_local and grad_h_post_local to zero
                T.clear(grad_h_res_local)
                T.clear(grad_h_post_local)
                
                # Process C dimension in blocks
                for pid_c in T.Pipelined(C // BLOCK_C, num_stages=3):
                    c_idx = pid_c * BLOCK_C
                    
                    # Load inputs for this C block
                    T.copy(grad_output[i, 0, c_idx], grad_output_local)
                    T.copy(original_residual[i, 0, c_idx], original_residual_local)
                    T.copy(x[i, c_idx], x_local)
                    T.copy(bias[c_idx], bias_local)
                    
                    # grad_h_res[i, j, m] += Σ_k (grad_output[i, j, k] × original_residual[i, m, k])
                    # temp_h_res[j, m, k] = grad_output[j, k] * original_residual[m, k]

                    for j, k, l in T.Parallel(n, n, BLOCK_C): 
                        temp_h_res[k, j, l] = grad_output_local[k, l] * original_residual_local[j, l]
                        temp_orig_res[j, l, k] = grad_output_local[k, l] * h_res_local[k, j]
                    for k, l  in T.Parallel(n, BLOCK_C):
                        temp_h_post[k, l] = grad_output_local[k, l] * (x_local[l] + bias_local[l])
                        temp_x[k, l] = grad_output_local[k, l] * h_post_local[k]

                    T.reduce_sum(temp_h_res, grad_h_res_local, dim=2, clear=False)
                    T.reduce_sum(temp_orig_res, grad_original_residual_local, dim=2, clear=True)

                    T.reduce_sum(temp_h_post, grad_h_post_local, dim=1, clear=False)
                    T.reduce_sum(temp_x, grad_x_local, dim=0, clear=True)
                    T.copy(grad_original_residual_local, grad_original_residual[i, 0, c_idx])
                    T.copy(grad_x_local, grad_x[i, c_idx])
                
                # Write accumulated results
                T.copy(grad_h_res_local, grad_h_res[i, 0, 0])
                T.copy(grad_h_post_local, grad_h_post[i, 0])
        
        return h_post_bda_backward_kernel_

    # ==================== Kernel Cache ====================

    _FORWARD_KERNEL_CACHE = {}
    _BACKWARD_KERNEL_CACHE = {}

    def _get_forward_kernel(sb: int, n: int, C: int):
        """Get cached forward kernel or create a new one."""
        key = (sb, n, C)
        if key not in _FORWARD_KERNEL_CACHE:
            _FORWARD_KERNEL_CACHE[key] = _h_post_bda_forward_kernel_generator(sb, n, C)
        return _FORWARD_KERNEL_CACHE[key]

    def _get_backward_kernel(sb: int, n: int, C: int):
        """Get cached backward kernel or create a new one."""
        key = (sb, n, C)
        if key not in _BACKWARD_KERNEL_CACHE:
            _BACKWARD_KERNEL_CACHE[key] = _h_post_bda_backward_kernel_generator(sb, n, C)
        return _BACKWARD_KERNEL_CACHE[key]


def h_post_bda_tilelang_forward(
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Tensor,
) -> Tensor:
    """
    TileLang implementation of H_post BDA forward pass (no dropout).

    Computes:
        output = H_res @ original_residual + H_post^T @ (x + bias)

    Args:
        h_res: [s, b, n, n] - residual mixing matrix
        original_residual: [s, b, n, C] - n-stream hidden states
        h_post: [s, b, n] - expansion weights
        x: [s, b, C] - layer output
        bias: [C] - bias tensor

    Returns:
        output: [s, b, n, C] - final output
    """
    s, b, n, C = original_residual.shape
    sb = s * b

    # Reshape inputs to flattened batch dimension
    h_res_flat = h_res.view(sb, n, n).contiguous()
    original_residual_flat = original_residual.view(sb, n, C).contiguous()
    h_post_flat = h_post.view(sb, n).contiguous()
    x_flat = x.view(sb, C).contiguous()
    bias_flat = bias.contiguous()

    # Allocate output
    output_flat = torch.empty(sb, n, C, dtype=h_res.dtype, device=h_res.device)

    # Get cached kernel
    kernel = _get_forward_kernel(sb, n, C)

    # Launch kernel
    kernel(h_res_flat, original_residual_flat, h_post_flat, x_flat, bias_flat, output_flat)

    # Reshape output back
    return output_flat.view(s, b, n, C)


def h_post_bda_tilelang_backward(
    grad_output: Tensor,
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    TileLang implementation of H_post BDA backward pass.

    Computes gradients:
        grad_h_res[i, j, m] = Σ_k (grad_output[i, j, k] × original_residual[i, m, k])
        grad_original_residual[i, m, k] = Σ_j (grad_output[i, j, k] × h_res[i, j, m])
        grad_h_post[i, j] = Σ_k (grad_output[i, j, k] × (x[i, k] + bias[k]))
        grad_x[i, k] = Σ_j (grad_output[i, j, k] × h_post[i, j])
        grad_bias[k] = Σ_{i,j} (grad_output[i, j, k] × h_post[i, j])

    Args:
        grad_output: [s, b, n, C] - gradient w.r.t. output
        h_res: [s, b, n, n] - residual mixing matrix
        original_residual: [s, b, n, C] - n-stream hidden states
        h_post: [s, b, n] - expansion weights
        x: [s, b, C] - layer output
        bias: [C] - bias tensor

    Returns:
        grad_h_res: [s, b, n, n] - gradient w.r.t. h_res
        grad_original_residual: [s, b, n, C] - gradient w.r.t. original_residual
        grad_h_post: [s, b, n] - gradient w.r.t. h_post
        grad_x: [s, b, C] - gradient w.r.t. x
        grad_bias: [C] - gradient w.r.t. bias
    """
    s, b, n, C = original_residual.shape
    sb = s * b

    # Reshape inputs to flattened batch dimension
    grad_output_flat = grad_output.view(sb, n, C).contiguous()
    h_res_flat = h_res.view(sb, n, n).contiguous()
    original_residual_flat = original_residual.view(sb, n, C).contiguous()
    h_post_flat = h_post.view(sb, n).contiguous()
    x_flat = x.view(sb, C).contiguous()
    bias_flat = bias.contiguous()

    # Allocate outputs
    grad_h_res_flat = torch.empty(sb, n, n, dtype=h_res.dtype, device=h_res.device)
    grad_original_residual_flat = torch.empty(sb, n, C, dtype=h_res.dtype, device=h_res.device)
    grad_h_post_flat = torch.empty(sb, n, dtype=h_res.dtype, device=h_res.device)
    grad_x_flat = torch.empty(sb, C, dtype=h_res.dtype, device=h_res.device)

    # Get cached kernel
    kernel = _get_backward_kernel(sb, n, C)

    # Launch kernel
    kernel(
        grad_output_flat, h_res_flat, original_residual_flat,
        h_post_flat, x_flat, bias_flat,
        grad_h_res_flat, grad_original_residual_flat, grad_h_post_flat, grad_x_flat
    )

    # Reshape outputs back
    grad_h_res = grad_h_res_flat.view(s, b, n, n)
    grad_original_residual = grad_original_residual_flat.view(s, b, n, C)
    grad_h_post = grad_h_post_flat.view(s, b, n)
    grad_x = grad_x_flat.view(s, b, C)

    # grad_bias[k] = Σ_i grad_x[i, k] (sum over batch dimension)
    grad_bias = grad_x_flat.sum(dim=0)

    return grad_h_res, grad_original_residual, grad_h_post, grad_x, grad_bias


def h_post_bda_fused_forward(
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor],
    dropout_prob: float,
    training: bool,
    dropout_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Fused H_post BDA forward pass using TileLang kernel.

    Computes:
        1. mixed = H_res @ original_residual
        2. x_expanded = H_post^T @ x
        3. bias_expanded = H_post^T @ bias (if bias is not None)
        4. output = dropout(x_expanded + bias_expanded) + mixed

    Args:
        h_res: [s, b, n, n] - residual mixing matrix
        original_residual: [s, b, n, C] - n-stream hidden states (reshaped from [s, b, n*C])
        h_post: [s, b, n] - expansion weights
        x: [s, b, C] - layer output
        bias: [C] or None - optional bias tensor
        dropout_prob: Dropout probability
        training: Whether in training mode
        dropout_mask: Optional pre-generated dropout mask for deterministic dropout.
            If None and training=True, generates mask internally.

    Returns:
        output: [s, b, n, C] - final output (to be reshaped to [s, b, n*C])
        dropout_mask: [s, b, n, C] or None - dropout mask used (for backward)
    """
    if not _TILELANG_AVAILABLE:
        raise RuntimeError("TileLang is not available.")

    # TODO: Implement fused forward kernel
    # Placeholder: raise NotImplementedError for now
    raise NotImplementedError(
        "h_post_bda_fused_forward kernel not yet implemented. "
        "Please implement the TileLang kernel."
    )


def h_post_bda_fused_backward(
    grad_output: Tensor,
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor],
    dropout_mask: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """
    Fused H_post BDA backward pass using TileLang kernel.

    Computes gradients for all inputs:
        - grad_h_res: gradient w.r.t. h_res
        - grad_original_residual: gradient w.r.t. original_residual
        - grad_h_post: gradient w.r.t. h_post
        - grad_x: gradient w.r.t. x
        - grad_bias: gradient w.r.t. bias (if bias was provided)

    Args:
        grad_output: [s, b, n, C] - gradient w.r.t. output
        h_res: [s, b, n, n] - residual mixing matrix
        original_residual: [s, b, n, C] - n-stream hidden states
        h_post: [s, b, n] - expansion weights
        x: [s, b, C] - layer output
        bias: [C] or None - optional bias tensor
        dropout_mask: [s, b, n, C] or None - dropout mask from forward

    Returns:
        grad_h_res: [s, b, n, n] - gradient w.r.t. h_res
        grad_original_residual: [s, b, n, C] - gradient w.r.t. original_residual
        grad_h_post: [s, b, n] - gradient w.r.t. h_post
        grad_x: [s, b, C] - gradient w.r.t. x
        grad_bias: [C] or None - gradient w.r.t. bias
    """
    if not _TILELANG_AVAILABLE:
        raise RuntimeError("TileLang is not available.")

    # TODO: Implement fused backward kernel
    # Placeholder: raise NotImplementedError for now
    raise NotImplementedError(
        "h_post_bda_fused_backward kernel not yet implemented. "
        "Please implement the TileLang kernel."
    )


def h_post_bda_native_forward(
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor],
    dropout_prob: float,
    training: bool,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Native PyTorch implementation of H_post BDA forward pass.

    This is the reference implementation for testing and fallback.

    Args:
        h_res: [s, b, n, n] - residual mixing matrix
        original_residual: [s, b, n, C] - n-stream hidden states
        h_post: [s, b, n] - expansion weights
        x: [s, b, C] - layer output
        bias: [C] or None - optional bias tensor
        dropout_prob: Dropout probability
        training: Whether in training mode

    Returns:
        output: [s, b, n, C] - final output
        dropout_mask: [s, b, n, C] or None - dropout mask used (for backward)
    """
    s, b, n, C = original_residual.shape

    # Step 1: Apply H_res to original residual
    # [s, b, n, n] @ [s, b, n, C] -> [s, b, n, C]
    h_res_batched = h_res.view(s * b, n, n)
    residual_batched = original_residual.view(s * b, n, C)
    mixed = torch.bmm(h_res_batched, residual_batched)  # [s*b, n, C]
    mixed = mixed.view(s, b, n, C)

    # Step 2: Apply H_post to x
    # x: [s, b, C] -> [s, b, 1, C]
    # h_post: [s, b, n] -> [s, b, n, 1]
    # x_expanded: [s, b, n, C]
    x_expanded = h_post.unsqueeze(-1) * x.unsqueeze(2)  # [s, b, n, C]

    # Step 3: Apply H_post to bias (if present)
    if bias is not None:
        # bias: [C] -> [1, 1, 1, C]
        bias_expanded = h_post.unsqueeze(-1) * bias.view(1, 1, 1, C)  # [s, b, n, C]
        pre_dropout = x_expanded + bias_expanded
    else:
        pre_dropout = x_expanded

    # Step 4: Dropout and add mixed
    if training and dropout_prob > 0:
        dropout_mask = torch.bernoulli(
            torch.full_like(pre_dropout, 1.0 - dropout_prob)
        ) / (1.0 - dropout_prob)
        output = pre_dropout * dropout_mask + mixed
    else:
        dropout_mask = None
        output = pre_dropout + mixed

    return output, dropout_mask


def h_post_bda_native_backward(
    grad_output: Tensor,
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor],
    dropout_mask: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """
    Native PyTorch implementation of H_post BDA backward pass.

    Uses autograd for gradient computation via recomputation.

    Args:
        grad_output: [s, b, n, C] - gradient w.r.t. output
        h_res: [s, b, n, n] - residual mixing matrix
        original_residual: [s, b, n, C] - n-stream hidden states
        h_post: [s, b, n] - expansion weights
        x: [s, b, C] - layer output
        bias: [C] or None - optional bias tensor
        dropout_mask: [s, b, n, C] or None - dropout mask from forward

    Returns:
        grad_h_res: [s, b, n, n] - gradient w.r.t. h_res
        grad_original_residual: [s, b, n, C] - gradient w.r.t. original_residual
        grad_h_post: [s, b, n] - gradient w.r.t. h_post
        grad_x: [s, b, C] - gradient w.r.t. x
        grad_bias: [C] or None - gradient w.r.t. bias
    """
    s, b, n, C = original_residual.shape

    with torch.enable_grad():
        # Make inputs require grad for autograd
        h_res_input = h_res.detach().requires_grad_(True)
        original_residual_input = original_residual.detach().requires_grad_(True)
        h_post_input = h_post.detach().requires_grad_(True)
        x_input = x.detach().requires_grad_(True)
        if bias is not None:
            bias_input = bias.detach().requires_grad_(True)
        else:
            bias_input = None

        # Recompute forward
        # Step 1: Apply H_res
        h_res_batched = h_res_input.view(s * b, n, n)
        residual_batched = original_residual_input.view(s * b, n, C)
        mixed = torch.bmm(h_res_batched, residual_batched).view(s, b, n, C)

        # Step 2: Apply H_post to x
        x_expanded = h_post_input.unsqueeze(-1) * x_input.unsqueeze(2)

        # Step 3: Apply H_post to bias
        if bias_input is not None:
            bias_expanded = h_post_input.unsqueeze(-1) * bias_input.view(1, 1, 1, C)
            pre_dropout = x_expanded + bias_expanded
        else:
            pre_dropout = x_expanded

        # Step 4: Dropout and add
        if dropout_mask is not None:
            output = pre_dropout * dropout_mask + mixed
        else:
            output = pre_dropout + mixed

        # Compute gradients
        inputs = [h_res_input, original_residual_input, h_post_input, x_input]
        if bias_input is not None:
            inputs.append(bias_input)

        grads = torch.autograd.grad(
            outputs=output,
            inputs=inputs,
            grad_outputs=grad_output,
            create_graph=False,
            retain_graph=False,
        )

        grad_h_res = grads[0]
        grad_original_residual = grads[1]
        grad_h_post = grads[2]
        grad_x = grads[3]
        grad_bias = grads[4] if bias_input is not None else None

    return grad_h_res, grad_original_residual, grad_h_post, grad_x, grad_bias


def _benchmark_native(
    s: int,
    b: int,
    n: int,
    C: int,
    dtype: torch.dtype = torch.float32,
    dropout_prob: float = 0.0,
    use_bias: bool = False,
    warmup_iters: int = 20,
    bench_iters: int = 100,
):
    """
    Benchmark native H_post BDA forward + backward using CUDA events.

    Args:
        s: sequence length
        b: batch size
        n: number of hyper-connection streams
        C: hidden dimension
        dtype: data type
        dropout_prob: dropout probability (0.0 = no dropout)
        use_bias: whether to include bias tensor
        warmup_iters: number of warmup iterations
        bench_iters: number of benchmark iterations
    """
    device = "cuda"
    torch.manual_seed(42)

    # --- Allocate tensors ---
    h_res = torch.randn(s, b, n, n, dtype=dtype, device=device, requires_grad=True)
    original_residual = torch.randn(s, b, n, C, dtype=dtype, device=device, requires_grad=True)
    h_post = torch.randn(s, b, n, dtype=dtype, device=device, requires_grad=True)
    x = torch.randn(s, b, C, dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn(C, dtype=dtype, device=device, requires_grad=True) if use_bias else None
    grad_output = torch.randn(s, b, n, C, dtype=dtype, device=device)

    training = dropout_prob > 0.0

    # --- Helper: single fwd+bwd iteration ---
    def _run_once():
        # Zero grads
        for t in [h_res, original_residual, h_post, x]:
            if t.grad is not None:
                t.grad.zero_()
        if bias is not None and bias.grad is not None:
            bias.grad.zero_()

        output, dropout_mask = h_post_bda_native_forward(
            h_res, original_residual, h_post, x, bias,
            dropout_prob=dropout_prob, training=training,
        )
        grads = h_post_bda_native_backward(
            grad_output, h_res, original_residual, h_post, x, bias, dropout_mask,
        )
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
    Benchmark TileLang H_post BDA forward using CUDA events.

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
    h_res = torch.randn(s, b, n, n, dtype=dtype, device=device)
    original_residual = torch.randn(s, b, n, C, dtype=dtype, device=device)
    h_post = torch.randn(s, b, n, dtype=dtype, device=device)
    x = torch.randn(s, b, C, dtype=dtype, device=device)
    bias = torch.randn(C, dtype=dtype, device=device)

    # --- Helper: single forward iteration ---
    def _run_once():
        output = h_post_bda_tilelang_forward(
            h_res, original_residual, h_post, x, bias,
        )
        return output

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
    Benchmark native H_post BDA forward only (for fair comparison with tilelang).
    """
    device = "cuda"
    torch.manual_seed(42)

    # --- Allocate tensors ---
    h_res = torch.randn(s, b, n, n, dtype=dtype, device=device)
    original_residual = torch.randn(s, b, n, C, dtype=dtype, device=device)
    h_post = torch.randn(s, b, n, dtype=dtype, device=device)
    x = torch.randn(s, b, C, dtype=dtype, device=device)
    bias = torch.randn(C, dtype=dtype, device=device)

    # --- Helper: single forward iteration ---
    def _run_once():
        output, _ = h_post_bda_native_forward(
            h_res, original_residual, h_post, x, bias,
            dropout_prob=0.0, training=False,
        )
        return output

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
    Benchmark TileLang H_post BDA backward using CUDA events.
    """
    device = "cuda"
    torch.manual_seed(42)

    # --- Allocate tensors ---
    grad_output = torch.randn(s, b, n, C, dtype=dtype, device=device)
    h_res = torch.randn(s, b, n, n, dtype=dtype, device=device)
    original_residual = torch.randn(s, b, n, C, dtype=dtype, device=device)
    h_post = torch.randn(s, b, n, dtype=dtype, device=device)
    x = torch.randn(s, b, C, dtype=dtype, device=device)
    bias = torch.randn(C, dtype=dtype, device=device)

    # --- Helper: single backward iteration ---
    def _run_once():
        grads = h_post_bda_tilelang_backward(
            grad_output, h_res, original_residual, h_post, x, bias,
        )
        return grads

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
    Benchmark TileLang H_post BDA forward + backward using CUDA events.
    """
    device = "cuda"
    torch.manual_seed(42)

    # --- Allocate tensors ---
    h_res = torch.randn(s, b, n, n, dtype=dtype, device=device)
    original_residual = torch.randn(s, b, n, C, dtype=dtype, device=device)
    h_post = torch.randn(s, b, n, dtype=dtype, device=device)
    x = torch.randn(s, b, C, dtype=dtype, device=device)
    bias = torch.randn(C, dtype=dtype, device=device)
    grad_output = torch.randn(s, b, n, C, dtype=dtype, device=device)

    # --- Helper: single fwd+bwd iteration ---
    def _run_once():
        output = h_post_bda_tilelang_forward(
            h_res, original_residual, h_post, x, bias,
        )
        grads = h_post_bda_tilelang_backward(
            grad_output, h_res, original_residual, h_post, x, bias,
        )
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark H_post BDA implementation")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "native", "tilelang", "compare"],
                        help="Benchmark mode: all, native, tilelang, or compare (forward only comparison)")
    args = parser.parse_args()

    print("=" * 70)
    print("H_post BDA Benchmark")
    print("=" * 70)

    # ---- Correctness check: TileLang vs Native ----
    print("\n[Correctness Check] TileLang vs Native Forward")
    print("-" * 50)
    torch.manual_seed(42)
    # 使用能被 BLOCK_SB=64 和 BLOCK_C=128 整除的尺寸
    s, b, n, C = 1, 64, 4, 128
    h_res = torch.randn(s, b, n, n, dtype=torch.float32).cuda()
    original_residual = torch.randn(s, b, n, C, dtype=torch.float32).cuda()
    h_post = torch.randn(s, b, n, dtype=torch.float32).cuda()
    x = torch.randn(s, b, C, dtype=torch.float32).cuda()
    bias_tensor = torch.randn(C, dtype=torch.float32).cuda()

    # Native forward (no dropout for comparison)
    output_native, _ = h_post_bda_native_forward(
        h_res, original_residual, h_post, x, bias_tensor,
        dropout_prob=0.0, training=False,
    )
    print(f"  Native output shape: {output_native.shape}")

    # TileLang forward
    output_tilelang = h_post_bda_tilelang_forward(
        h_res, original_residual, h_post, x, bias_tensor,
    )
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
        print(f"  Native output sample:\n{output_native[0, 0, 0, :8]}")
        print(f"  TileLang output sample:\n{output_tilelang[0, 0, 0, :8]}")

    # ---- Correctness check: TileLang Backward vs Native Backward ----
    print("\n[Correctness Check] TileLang vs Native Backward")
    print("-" * 50)
    grad_output = torch.randn(s, b, n, C, dtype=torch.float32).cuda()

    # Native backward
    grads_native = h_post_bda_native_backward(
        grad_output, h_res, original_residual, h_post, x, bias_tensor, dropout_mask=None,
    )
    print(f"  Native grad shapes: h_res={grads_native[0].shape}, orig_res={grads_native[1].shape}, "
          f"h_post={grads_native[2].shape}, x={grads_native[3].shape}, bias={grads_native[4].shape}")

    # TileLang backward
    grads_tilelang = h_post_bda_tilelang_backward(
        grad_output, h_res, original_residual, h_post, x, bias_tensor,
    )
    print(f"  TileLang grad shapes: h_res={grads_tilelang[0].shape}, orig_res={grads_tilelang[1].shape}, "
          f"h_post={grads_tilelang[2].shape}, x={grads_tilelang[3].shape}, bias={grads_tilelang[4].shape}")

    # Compare each gradient
    grad_names = ["grad_h_res", "grad_original_residual", "grad_h_post", "grad_x", "grad_bias"]
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
    # Realistic sizes for GPT-style models with mHC
    # (s, b, n, C): seq_len, batch_size, num_streams, hidden_dim
    # 需要保证 s*b 能被 BLOCK_SB=64 整除, C 能被 BLOCK_C=128 整除
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
        # ---- Native forward + backward benchmark ----
        for use_bias in [False, True]:
            bias_label = "bias=yes" if use_bias else "bias=None"
            dropout_prob = 0.0

            print(f"\n{'=' * 70}")
            print(f"  Native (fwd+bwd): dropout_prob={dropout_prob}, {bias_label}, dtype=float32")
            print(f"  warmup={args.warmup}, iters={args.iters}")
            print(f"{'=' * 70}")
            print(f"  {'(s, b, n, C)':<25s} {'avg (ms)':>10s} {'min (ms)':>10s} {'max (ms)':>10s}")
            print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

            for (s, b, n, C) in configs:
                avg_ms, min_ms, max_ms = _benchmark_native(
                    s, b, n, C,
                    dtype=torch.float32,
                    dropout_prob=dropout_prob,
                    use_bias=use_bias,
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
                dropout_prob=0.0,
                use_bias=True,
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
