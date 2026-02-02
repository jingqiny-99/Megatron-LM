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

from megatron.core.fusions.fused_sinkhorn import is_tilelang_available

_TILELANG_AVAILABLE = False

try:
    import tilelang
    import tilelang.language as T
    _TILELANG_AVAILABLE = True
except ImportError:
    pass


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
    #   - dropout_prob: float - dropout probability
    #   - dropout_mask: [sb, n, C] or None - pre-generated dropout mask (for training)
    # Output shapes:
    #   - output: [sb, n, C] - final output after all operations

    # ==================== Backward Kernel ====================

    # TODO: Implement backward kernel
    # Input shapes:
    #   - grad_output: [sb, n, C] - gradient w.r.t. output
    #   - h_res: [sb, n, n] - residual mixing matrix
    #   - original_residual: [sb, n, C] - n-stream hidden states
    #   - h_post: [sb, n] - expansion weights
    #   - x: [sb, C] - layer output
    #   - bias: [C] or None - optional bias tensor
    #   - dropout_mask: [sb, n, C] or None - dropout mask from forward
    # Output shapes:
    #   - grad_h_res: [sb, n, n] - gradient w.r.t. h_res
    #   - grad_original_residual: [sb, n, C] - gradient w.r.t. original_residual
    #   - grad_h_post: [sb, n] - gradient w.r.t. h_post
    #   - grad_x: [sb, C] - gradient w.r.t. x
    #   - grad_bias: [C] or None - gradient w.r.t. bias

    # ==================== Kernel Cache ====================

    _FORWARD_KERNEL_CACHE = {}
    _BACKWARD_KERNEL_CACHE = {}


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


if __name__ == "__main__":
    # Test native implementation
    torch.manual_seed(42)

    s, b, n, C = 2, 2, 4, 64
    h_res = torch.randn(s, b, n, n, dtype=torch.float32, requires_grad=True).cuda()
    original_residual = torch.randn(s, b, n, C, dtype=torch.float32, requires_grad=True).cuda()
    h_post = torch.randn(s, b, n, dtype=torch.float32, requires_grad=True).cuda()
    x = torch.randn(s, b, C, dtype=torch.float32, requires_grad=True).cuda()
    bias = torch.randn(C, dtype=torch.float32, requires_grad=True).cuda()

    # Forward
    output, dropout_mask = h_post_bda_native_forward(
        h_res, original_residual, h_post, x, bias,
        dropout_prob=0.1, training=True
    )
    print(f"Output shape: {output.shape}")

    # Backward
    grad_output = torch.randn_like(output)
    grads = h_post_bda_native_backward(
        grad_output, h_res, original_residual, h_post, x, bias, dropout_mask
    )
    print(f"grad_h_res shape: {grads[0].shape}")
    print(f"grad_original_residual shape: {grads[1].shape}")
    print(f"grad_h_post shape: {grads[2].shape}")
    print(f"grad_x shape: {grads[3].shape}")
    print(f"grad_bias shape: {grads[4].shape}")
