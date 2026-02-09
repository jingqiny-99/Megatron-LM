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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark H_post BDA native implementation")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Number of benchmark iterations")
    args = parser.parse_args()

    print("=" * 70)
    print("H_post BDA Native Benchmark (forward + backward)")
    print("=" * 70)

    # ---- Correctness check first ----
    torch.manual_seed(42)
    s, b, n, C = 2, 2, 4, 64
    h_res = torch.randn(s, b, n, n, dtype=torch.float32, requires_grad=True).cuda()
    original_residual = torch.randn(s, b, n, C, dtype=torch.float32, requires_grad=True).cuda()
    h_post = torch.randn(s, b, n, dtype=torch.float32, requires_grad=True).cuda()
    x = torch.randn(s, b, C, dtype=torch.float32, requires_grad=True).cuda()
    bias_tensor = torch.randn(C, dtype=torch.float32, requires_grad=True).cuda()

    output, dropout_mask = h_post_bda_native_forward(
        h_res, original_residual, h_post, x, bias_tensor,
        dropout_prob=0.0, training=False,
    )
    print(f"\n[Correctness] Output shape: {output.shape}")

    grad_output = torch.randn_like(output)
    grads = h_post_bda_native_backward(
        grad_output, h_res, original_residual, h_post, x, bias_tensor, dropout_mask,
    )
    print(f"[Correctness] grad_h_res shape:            {grads[0].shape}")
    print(f"[Correctness] grad_original_residual shape: {grads[1].shape}")
    print(f"[Correctness] grad_h_post shape:            {grads[2].shape}")
    print(f"[Correctness] grad_x shape:                 {grads[3].shape}")
    print(f"[Correctness] grad_bias shape:              {grads[4].shape}")

    # ---- Benchmark configs ----
    # Realistic sizes for GPT-style models with mHC
    # (s, b, n, C): seq_len, batch_size, num_streams, hidden_dim
    configs = [
        # Small (debugging)
        (128, 1, 4, 1024),
        (128, 1, 4, 2048),
        # Medium (e.g. GPT-like)
        (512, 1, 4, 4096),
        (1024, 1, 4, 4096),
        # Larger hidden dim
        (512, 1, 4, 8192),
        (1024, 1, 4, 8192),
        # Different stream counts
        (512, 1, 2, 4096),
        (512, 1, 8, 4096),
    ]

    # Note: For mHC, dropout is typically 0 and bias is typically None
    # (modern LLMs use --disable-bias-linear).
    # We benchmark both cases for reference.

    for use_bias in [False, True]:
        bias_label = "bias=yes" if use_bias else "bias=None"
        dropout_prob = 0.0  # mHC typically uses dropout=0

        print(f"\n{'=' * 70}")
        print(f"  dropout_prob={dropout_prob}, {bias_label}, dtype=float32")
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

    print(f"\n{'=' * 70}")
    print("Done.")
