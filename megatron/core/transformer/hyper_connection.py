from typing import Tuple, Optional, TYPE_CHECKING
import math
import torch
import torch.nn as nn
from torch import Tensor

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_decorator, nvtx_range_push, nvtx_range_pop
from megatron.core.fusions.fused_sinkhorn import (
    sinkhorn_fused_forward,
    sinkhorn_fused_backward,
    is_tilelang_available,
)
from megatron.core.fusions.fused_h_post_bda import (
    h_post_bda_tilelang_forward,
    h_post_bda_tilelang_backward,
)
from megatron.core.fusions.fused_h_aggregate import (
    h_aggregate_tilelang_forward,
    h_aggregate_tilelang_backward,
)

if TYPE_CHECKING:
    from megatron.core.tensor_parallel.random import MHCBlockRecomputeManager


class SinkhornKnopp(torch.autograd.Function):
    """
    Differentiable Sinkhorn-Knopp algorithm for doubly stochastic projection.
    
    Projects a positive matrix onto the Birkhoff polytope (doubly stochastic matrices)
    via iterative row and column normalization.
    
    Reference: Eq. (9) in mHC paper - M^{(t)} = T_c(T_r(M^{(t-1)}))
    """
    
    @staticmethod
    def forward(
        ctx, H_res_logits: Tensor, num_iterations: int, use_fused_kernel: bool = False
    ) -> Tensor:
        """
        Project to doubly stochastic matrix via iterative row/col normalization.
        
        Args:
            H_res_logits: [s, b, n, n] - raw logits for residual mixing matrix
            num_iterations: Number of Sinkhorn iterations (paper uses 20)
            use_fused_kernel: Whether to use fused TileLang kernel.
                If True and TileLang is available, uses fused kernels for both forward and backward.
                Falls back to native PyTorch implementation otherwise.
        
        Returns:
            H_res: [s, b, n, n] - doubly stochastic matrix
        """
        # M^{(0)} = exp(H_res_logits) - save initial M for backward recomputation
        M_init = torch.exp(H_res_logits)
        
        _use_fused = use_fused_kernel and is_tilelang_available()
        
        if _use_fused:
            # Use fused TileLang kernel for forward computation
            with torch.no_grad():
                M = sinkhorn_fused_forward(H_res_logits, num_iterations, eps=1e-8)
        else:
            # Native PyTorch implementation
            M = M_init.clone()
            with torch.no_grad():
                for _ in range(num_iterations):
                    # T_r: Row normalization
                    M = M / M.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    # T_c: Column normalization
                    M = M / M.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        
        # Save initial M for backward recomputation
        ctx.save_for_backward(M_init)
        ctx.num_iterations = num_iterations
        ctx.use_fused_kernel = _use_fused
        return M
    
    @staticmethod  
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None]:
        """
        Backward through Sinkhorn-Knopp iterations.
        
        Uses fused TileLang kernel if enabled in forward, otherwise uses
        recomputation with autograd for gradient computation.
        
        Both paths return dL/dH (gradient w.r.t. input logits), including
        the chain rule (grad_input = grad_M_init * M_init).
        """
        M_init, = ctx.saved_tensors
        num_iterations = ctx.num_iterations
        use_fused_kernel = ctx.use_fused_kernel
        
        if use_fused_kernel:
            # Fused kernel computes grad_input = grad_M_init * M_init internally
            grad_input = sinkhorn_fused_backward(
                grad_output, M_init, num_iterations, eps=1e-8
            )
        else:
            # Recompute forward with autograd enabled
            with torch.enable_grad():
                M_input = M_init.detach().requires_grad_(True)

                M_current = M_input
                for _ in range(num_iterations):
                    M_current = M_current / M_current.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    M_current = M_current / M_current.sum(dim=-2, keepdim=True).clamp(min=1e-8)

                grad_M_init, = torch.autograd.grad(
                    outputs=M_current,
                    inputs=M_input,
                    grad_outputs=grad_output,
                    create_graph=False,
                    retain_graph=False,
                )
            
            # Apply chain rule: dL/dH = dL/dM_init * M_init
            # Since M_init = exp(H), d(exp(x))/dx = exp(x) = M_init
            grad_input = grad_M_init * M_init
        
        return grad_input, None, None


class FusedHAggregate(torch.autograd.Function):
    """
    Differentiable fused H_pre aggregation operation.
    
    This autograd function wraps the fused kernel for aggregating n-stream
    hidden states into a single stream using H_pre weights:
        aggregated[i, c] = sum_j(h_pre[i, j] * x[i, j, c])
    
    When use_fused_kernel=True and TileLang is available, uses fused kernels.
    Otherwise falls back to native PyTorch implementation.
    """
    
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        h_pre: Tensor,
        n: int,
        C: int,
        use_fused_kernel: bool = False,
    ) -> Tensor:
        """
        Fused forward pass for H_pre aggregation.
        
        Args:
            x: [s, b, n*C] - n-stream hidden states
            h_pre: [s, b, n] - aggregation weights
            n: Number of residual streams
            C: Hidden size per stream
            use_fused_kernel: Whether to use fused TileLang kernel.
                If True and TileLang is available, uses fused kernels.
                Falls back to native PyTorch implementation otherwise.
        
        Returns:
            aggregated: [s, b, C] - aggregated hidden states
        """
        s, b, _ = x.shape
        x_4d = x.view(s, b, n, C)
        
        _use_fused = use_fused_kernel and is_tilelang_available()
        
        if _use_fused:
            with torch.no_grad():
                aggregated = h_aggregate_tilelang_forward(x_4d, h_pre)
        else:
            with torch.no_grad():
                aggregated = (x_4d * h_pre.unsqueeze(-1)).sum(dim=2)
        
        ctx.save_for_backward(x_4d, h_pre)
        ctx.use_fused_kernel = _use_fused
        ctx.n = n
        ctx.C = C
        return aggregated
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor, None, None, None]:
        """
        Backward through fused H_pre aggregation.
        
        Gradient formulas:
            grad_x[i, j, c] = grad_output[i, c] * h_pre[i, j]
            grad_h_pre[i, j] = sum_c(grad_output[i, c] * x[i, j, c])
        
        Returns:
            grad_x: gradient w.r.t. x (reshaped to [s, b, n*C])
            grad_h_pre: gradient w.r.t. h_pre
            None: for n
            None: for C
            None: for use_fused_kernel
        """
        x_4d, h_pre = ctx.saved_tensors
        use_fused_kernel = ctx.use_fused_kernel
        n = ctx.n
        C = ctx.C
        s, b, _ = grad_output.shape
        
        if use_fused_kernel:
            grad_x_4d, grad_h_pre = h_aggregate_tilelang_backward(
                grad_output, x_4d, h_pre
            )
        else:
            with torch.enable_grad():
                x_input = x_4d.detach().requires_grad_(True)
                h_pre_input = h_pre.detach().requires_grad_(True)
                
                output = (x_input * h_pre_input.unsqueeze(-1)).sum(dim=2)
                
                grad_x_4d, grad_h_pre = torch.autograd.grad(
                    outputs=output,
                    inputs=[x_input, h_pre_input],
                    grad_outputs=grad_output,
                    create_graph=False,
                    retain_graph=False,
                )
        
        # Reshape grad_x from [s, b, n, C] to [s, b, n*C]
        grad_x = grad_x_4d.view(s, b, n * C)
        return grad_x, grad_h_pre, None, None, None


class FusedHPostBDA(torch.autograd.Function):
    """
    Differentiable fused H_post expansion and bias-dropout-add operation.
    
    This autograd function wraps the fused kernel for memory-efficient computation of:
        1. mixed = H_res @ original_residual (apply_h_res)
        2. x_expanded = H_post^T @ layer_output (apply_h_post)
        3. bias_expanded = H_post^T @ bias (if bias is not None)
        4. output = dropout(x_expanded + bias_expanded) + mixed (bias-dropout-add)
    
    When use_fused_kernel=True and TileLang is available, uses fused kernels.
    Otherwise falls back to native PyTorch implementation.
    """
    
    @staticmethod
    def forward(
        ctx,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        x: Tensor,
        bias: Optional[Tensor],
        dropout_prob: float,
        training: bool,
        use_fused_kernel: bool = False,
    ) -> Tensor:
        """
        Fused forward pass for H_post expansion and bias-dropout-add.
        
        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            original_residual: [s, b, n*C] - n-stream hidden states
            h_post: [s, b, n] - expansion weights
            x: [s, b, C] - layer output
            bias: [C] or None - optional bias tensor
            dropout_prob: Dropout probability
            training: Whether in training mode
            use_fused_kernel: Whether to use fused TileLang kernel.
                If True and TileLang is available, uses fused kernels.
                Falls back to native PyTorch implementation otherwise.
        
        Returns:
            output: [s, b, n*C] - final output after all operations
        """
        s, b, n, _ = h_res.shape
        C = x.shape[-1]
        
        # Reshape original_residual from [s, b, n*C] to [s, b, n, C]
        original_residual_4d = original_residual.view(s, b, n, C)
        
        _use_fused = use_fused_kernel and is_tilelang_available()
        
        if _use_fused:
            assert bias is not None, (
                "Fused H_post BDA kernel requires bias to be non-None. "
                "Please provide a bias tensor when using fused kernels."
            )
            assert dropout_prob == 0.0, (
                f"Fused H_post BDA kernel requires dropout_prob to be 0, "
                f"got {dropout_prob}. Dropout is not supported in fused kernels."
            )
            # Use fused TileLang kernel for forward computation (no dropout support)
            with torch.no_grad():
                output_4d = h_post_bda_tilelang_forward(
                    h_res, original_residual_4d, h_post, x, bias
                )
                dropout_mask = None
        else:
            # Native PyTorch implementation
            with torch.no_grad():
                # Step 1: Apply H_res to original residual
                # [s*b, n, n] @ [s*b, n, C] -> [s*b, n, C]
                h_res_batched = h_res.view(s * b, n, n)
                residual_batched = original_residual_4d.view(s * b, n, C)
                mixed = torch.bmm(h_res_batched, residual_batched).view(s, b, n, C)
                
                # Step 2: Apply H_post to x
                # x: [s, b, C] -> [s, b, 1, C]
                # h_post: [s, b, n] -> [s, b, n, 1]
                x_expanded = h_post.unsqueeze(-1) * x.unsqueeze(2)  # [s, b, n, C]
                
                # Step 3: Apply H_post to bias (if present)
                if bias is not None:
                    bias_expanded = h_post.unsqueeze(-1) * bias.view(1, 1, 1, C)
                    pre_dropout = x_expanded + bias_expanded
                else:
                    pre_dropout = x_expanded
                
                # Step 4: Dropout and add mixed
                if training and dropout_prob > 0:
                    dropout_mask = torch.bernoulli(
                        torch.full_like(pre_dropout, 1.0 - dropout_prob)
                    ) / (1.0 - dropout_prob)
                    output_4d = pre_dropout * dropout_mask + mixed
                else:
                    dropout_mask = None
                    output_4d = pre_dropout + mixed
        
        # Save tensors for backward
        ctx.save_for_backward(
            h_res, original_residual_4d, h_post, x, bias, dropout_mask
        )
        ctx.use_fused_kernel = _use_fused
        ctx.n = n
        ctx.C = C
        
        # Reshape output from [s, b, n, C] to [s, b, n*C]
        return output_4d.view(s, b, n * C)
    
    @staticmethod
    def backward(
        ctx, grad_output: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor], None, None, None]:
        """
        Backward through fused H_post BDA operations.
        
        Uses fused TileLang kernel if enabled in forward, otherwise uses
        recomputation with autograd for gradient computation.
        
        Returns:
            grad_h_res: gradient w.r.t. h_res
            grad_original_residual: gradient w.r.t. original_residual
            grad_h_post: gradient w.r.t. h_post
            grad_x: gradient w.r.t. x
            grad_bias: gradient w.r.t. bias (or None)
            None: for dropout_prob
            None: for training
            None: for use_fused_kernel
        """
        h_res, original_residual_4d, h_post, x, bias, dropout_mask = ctx.saved_tensors
        use_fused_kernel = ctx.use_fused_kernel
        n = ctx.n
        C = ctx.C
        
        s, b, _ = grad_output.shape
        grad_output_4d = grad_output.view(s, b, n, C)
        
        if use_fused_kernel:
            # Fused TileLang kernel computes all gradients (no dropout)
            (grad_h_res, grad_original_residual_4d, grad_h_post, 
             grad_x, grad_bias) = h_post_bda_tilelang_backward(
                grad_output_4d, h_res, original_residual_4d, h_post, x, bias
            )
        else:
            # Recompute forward with autograd enabled
            with torch.enable_grad():
                # Make inputs require grad
                h_res_input = h_res.detach().requires_grad_(True)
                original_residual_input = original_residual_4d.detach().requires_grad_(True)
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
                    grad_outputs=grad_output_4d,
                    create_graph=False,
                    retain_graph=False,
                )
                
                grad_h_res = grads[0]
                grad_original_residual_4d = grads[1]
                grad_h_post = grads[2]
                grad_x = grads[3]
                grad_bias = grads[4] if bias_input is not None else None
        
        # Reshape grad_original_residual from [s, b, n, C] to [s, b, n*C]
        grad_original_residual = grad_original_residual_4d.view(s, b, n * C)
        
        return grad_h_res, grad_original_residual, grad_h_post, grad_x, grad_bias, None, None, None


class HyperConnectionModule(MegatronModule):
    """
    Unified mHC (Manifold-Constrained Hyper-Connections) module.
    
    Implements the complete mHC propagation:
        x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)
    
    This module handles:
    1. Computing learnable mappings: H_pre, H_post, H_res (with Sinkhorn-Knopp projection)
    2. Aggregation: n-stream → 1-stream (H_pre @ x)
    3. Expansion: 1-stream → n-stream (H_post^T @ output)
    4. Residual merge: H_res @ x + expanded_output
    5. Block-level expand/contract for TransformerBlock boundaries
    
    Args:
        config: TransformerConfig with hyper-connection fields
        layer_number: Current layer index for initialization
    """
    
    def __init__(self, config: TransformerConfig, layer_number: int):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.n = config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.sinkhorn_iterations = config.mhc_sinkhorn_iterations
        
        # Whether to use fused TileLang kernels for mHC operations
        # Controlled via config.mhc_use_fused_kernel
        self.use_fused_kernel = getattr(config, 'mhc_use_fused_kernel', False)
        
        # Projection weights for dynamic mappings
        # Input: [s, b, n*C] -> Output: n^2 + 2n values per token
        # - H_pre: n values
        # - H_post: n values  
        # - H_res: n^2 values (before Sinkhorn projection)
        self.norm = nn.RMSNorm(self.hidden_size * self.n)
        
        self.mapping_proj = nn.Linear(
            self.n * self.hidden_size, 
            self.n * self.n + 2 * self.n,
            bias=False
        )
        
        init_alpha = config.mhc_init_gating_factor
        # Learnable scaling factors (Eq. 5 in paper)
        self.alpha_pre = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_post = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_res = nn.Parameter(torch.full((1,), init_alpha))
        
        # Static bias terms
        self.bias = nn.Parameter(torch.zeros(self.n * self.n + 2 * self.n))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.mapping_proj.weight)
        
        # Set sequence_parallel attribute on parameters for gradient synchronization
        # across TP ranks when sequence_parallel is enabled.
        # This is required because HyperConnectionModule uses non-TP-aware layers
        # (nn.Linear, nn.RMSNorm) whose gradients need to be all-reduced.
        if self.config.sequence_parallel:
            setattr(self.mapping_proj.weight, 'sequence_parallel', True)
            setattr(self.norm.weight, 'sequence_parallel', True)
            setattr(self.alpha_pre, 'sequence_parallel', True)
            setattr(self.alpha_post, 'sequence_parallel', True)
            setattr(self.alpha_res, 'sequence_parallel', True)
            setattr(self.bias, 'sequence_parallel', True)
    

    # TODO: Kernel fusion
    @torch.compile
    @nvtx_decorator(message="HyperConnection::projection_and_rms")
    def _projection_and_rms(self, x : Tensor) -> Tuple[Tensor, Tensor]:
        """
        Project input hidden states to mapping space and apply RMS normalization.
        
        Args:
            x: [s, b, n*C] - n-stream hidden states
        """
        s, b, nC = x.shape
        n = self.n
        r = x.norm(dim=-1, keepdim=True) / math.sqrt(nC) # shape: [s, b, 1]
        r = 1.0 / (r + 1e-8) # shape: [s, b, 1]
        proj = self.mapping_proj(x)  # [s, b, n^2 + 2n]
        return proj, r
    
    #TODO: kernel fusion
    @torch.compile
    @nvtx_decorator(message="HyperConnection::compute_h")
    def _compute_h(self, proj: Tensor, r: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute h from projected hidden states and scaling factors.
        
        Args:
            proj: [s, b, n^2 + 2n] - projected hidden states
            r: [s, b, 1] - scaling factors
        
        Returns:
            h_pre: [s, b, n] - aggregation weights
            h_post: [s, b, n] - expansion weights
            h_res: [s, b, n^2] - residual mixing logits
        """
        s, b, _ = proj.shape
        alpha_ = torch.cat([self.alpha_pre.expand(self.n), self.alpha_post.expand(self.n), self.alpha_res.expand(self.n * self.n)], dim = -1)
        h = r * proj * alpha_ + self.bias
        # H_pre = σ(α_pre * (θ_pre @ x̃) + b_pre)
        h_pre = h[..., :self.n].sigmoid()  # [s, b, n]

        # H_post = 2σ(α_post * (θ_post @ x̃) + b_post)
        h_post = h[..., self.n:2*self.n].sigmoid() * 2 # [s, b, n]
        h_res = h[..., 2*self.n:]
        return h_pre, h_post, h_res
    
    @nvtx_decorator(message="HyperConnection::compute_mappings")
    def compute_mappings(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute mHC mappings from input hidden states.
        
        Reference: Eq. (5) and (8) in mHC paper
        
        Args:
            x: [s, b, n*C] - n-stream hidden states
        
        Returns:
            h_pre: [s, b, n] - aggregation weights (sigmoid activated)
            h_post: [s, b, n] - expansion weights (2*sigmoid activated)
            h_res: [s, b, n, n] - residual mixing matrix (doubly stochastic)
        """
        s, b, _ = x.shape
        proj, r = self._projection_and_rms(x)
        h_pre, h_post, h_res = self._compute_h(proj, r)
        h_res = SinkhornKnopp.apply(
            h_res.view(s, b, self.n, self.n), 
            self.sinkhorn_iterations,
            self.use_fused_kernel
        )  # [s, b, n, n] 
        
        return h_pre, h_post, h_res
    
    @torch.compile
    @nvtx_decorator(message="HyperConnection::apply_h_post_inner")
    def _apply_h_post(self, x: Tensor, h_post: Tensor) -> Tensor:
        """
        Core implementation of H_post application to a single tensor.
        
        Computes: H_post^T @ x
        
        Args:
            x: Input tensor, can be either:
               - [s, b, C] - standard hidden states
               - [C] - bias tensor (will be broadcast)
            h_post: [s, b, n] - expansion weights
        
        Returns:
            output: [s, b, n*C] - expanded tensor
        """
        n = self.n
        s, b, _ = h_post.shape
        
        if x.dim() == 1:
            # x is bias with shape [C], need to broadcast to [s, b, 1, C]
            C = x.shape[0]
            x_expanded = x.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(s, b, 1, C)
        else:
            # x is [s, b, C]
            C = x.shape[-1]
            x_expanded = x.unsqueeze(2)  # [s, b, 1, C]
        
        # h_post^T @ x : [s, b, n, 1] * [s, b, 1, C] -> [s, b, n, C]
        # Using broadcast multiply instead of einsum
        result = h_post.unsqueeze(-1) * x_expanded
        return result.view(s, b, n * C)
    
    @nvtx_decorator(message="HyperConnection::apply_h_post")
    def apply_h_post(
        self,
        x_with_bias: Tuple[Tensor, Optional[Tensor]],
        h_post: Tensor,
        manager: Optional['MHCBlockRecomputeManager'] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Apply H_post to x and optionally bias, with optional checkpointing.
        
        This is the unified entry point that handles both normal execution
        and checkpoint-based execution for memory efficiency.
        
        Args:
            x_with_bias: Tuple of (x, bias) where:
                - x: [s, b, C] - hidden states
                - bias: [C] or None - optional bias tensor
            h_post: [s, b, n] - expansion weights
            manager: Optional MHCBlockRecomputeManager for checkpoint management.
                When provided, wraps _apply_h_post with CheckpointWithoutOutput.
        
        Returns:
            Tuple of (x_out, bias_out) where:
                - x_out: [s, b, n*C] - expanded hidden states
                - bias_out: [s, b, n*C] or None - expanded bias if input bias was not None
        """
        x, bias = x_with_bias
        
        if manager is not None:
            from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
            
            # Checkpoint _apply_h_post for x
            x_out = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
                self._apply_h_post, x, h_post
            )
            
            # Checkpoint _apply_h_post for bias if not None
            if bias is not None:
                bias_out = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
                    self._apply_h_post, bias, h_post
                )
            else:
                bias_out = None
        else:
            # Normal execution without checkpoint
            x_out = self._apply_h_post(x, h_post)
            bias_out = self._apply_h_post(bias, h_post) if bias is not None else None
        
        return x_out, bias_out

    
    @nvtx_decorator(message="HyperConnection::aggregate")
    def aggregate(self, x: Tensor, h_pre: Tensor) -> Tensor:
        """
        Aggregate n-stream to 1-stream using H_pre weights.
        
        Computes: sum_i(h_pre_i * x_stream_i)
        
        When use_fused_kernel is True and TileLang is available, uses the
        FusedHAggregate autograd function with fused TileLang kernels.
        Otherwise falls back to the torch.compile'd native implementation.
        
        Args:
            x: [s, b, n*C] - n-stream hidden states
            h_pre: [s, b, n] - aggregation weights
        
        Returns:
            aggregated: [s, b, C] - single stream hidden states
        """
        if self.use_fused_kernel and is_tilelang_available():
            return FusedHAggregate.apply(
                x, h_pre, self.n, self.hidden_size, True
            )
        return self._aggregate_native(x, h_pre)
    
    @torch.compile
    @nvtx_decorator(message="HyperConnection::aggregate_native")
    def _aggregate_native(self, x: Tensor, h_pre: Tensor) -> Tensor:
        """
        Native PyTorch implementation of aggregate (torch.compile'd).
        
        Args:
            x: [s, b, n*C] - n-stream hidden states
            h_pre: [s, b, n] - aggregation weights
        
        Returns:
            aggregated: [s, b, C] - single stream hidden states
        """
        s, b, _ = x.shape
        C = self.hidden_size
        
        # Reshape to [s, b, n, C]
        x_streams = x.view(s, b, self.n, C)
        
        # Weighted sum: [s, b, n, C] * [s, b, n, 1] -> sum over n -> [s, b, C]
        aggregated = (x_streams * h_pre.unsqueeze(-1)).sum(dim=2)
        
        return aggregated

    @torch.compile
    @nvtx_decorator(message="HyperConnection::apply_h_res")
    def apply_h_res(self, h_res: Tensor, residual: Tensor) -> Tensor:
        """
        Apply H_res to residual using H_res weights.
        
        Computes: H_res @ residual
        
        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            residual: [s, b, n*C] - n-stream hidden states
        """
        s, b, _ = residual.shape 
        n = self.n
        C = self.hidden_size
        
        # Reshape for bmm: [s, b, n, n] -> [s*b, n, n]
        h_res_batched = h_res.view(s * b, n, n)
        # [s, b, n*C] -> [s, b, n, C] -> [s*b, n, C]
        residual_batched = residual.view(s, b, n, C).view(s * b, n, C)
        
        # Batch matrix multiply: [s*b, n, n] @ [s*b, n, C] -> [s*b, n, C]
        mixed = torch.bmm(h_res_batched, residual_batched)
        
        return mixed.view(s, b, n * C)
    
    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor, 
        training: bool = True,
        mhc_recompute_manager: Optional['MHCBlockRecomputeManager'] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full mHC forward pass.
        
        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            residual: [s, b, n*C] - n-stream hidden states (x_l)
            training: Whether in training mode
            mhc_recompute_manager: Optional MHCBlockRecomputeManager for checkpoint management.
                When provided, uses _forward_with_checkpoint for memory-efficient execution.
        
        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
            
        """
        if mhc_recompute_manager is not None:
            return self._forward_with_checkpoint(
                hidden_states, residual, mhc_recompute_manager
            )
        else:
            return self._forward_normal(hidden_states, residual)
    
    def _forward_normal(
        self, hidden_states: Tensor, residual: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Normal forward pass without checkpointing.
        
        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            residual: [s, b, n*C] - n-stream hidden states (x_l)
        
        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
        """
        # Compute mappings
        h_pre, h_post, h_res = self.compute_mappings(hidden_states)
        
        # Aggregate for layer input
        aggregated = self.aggregate(hidden_states, h_pre)

        return aggregated, h_res, h_post
    
    def _forward_with_checkpoint(
        self,
        hidden_states: Tensor,
        residual: Tensor,
        manager: 'MHCBlockRecomputeManager',
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with checkpointing for memory efficiency.
        
        Operations (compute_mappings, aggregate) are wrapped with
        CheckpointWithoutOutput and auto-registered to the manager.
        apply_h_res is deferred to fused_h_res_h_post_bda for kernel fusion.
        
        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            residual: [s, b, n*C] - n-stream hidden states (x_l)
            manager: MHCBlockRecomputeManager for unified recomputation
        
        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
        """
        from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
        
        nvtx_range_push("HyperConnection::compute_mappings")
        # Checkpoint compute_mappings - auto-registers to manager via ckpt_manager parameter
        h_pre, h_post, h_res = self.compute_mappings(hidden_states)
        
        nvtx_range_pop("HyperConnection::compute_mappings")
        # Checkpoint aggregate - auto-registers to manager
        nvtx_range_push("HyperConnection::aggregate")
        aggregated = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
            self.aggregate, hidden_states, h_pre
        )
        nvtx_range_pop("HyperConnection::aggregate")
        return aggregated, h_res, h_post
    
    # ==================== Block-level utilities ====================
    
    @staticmethod
    def input_expand(x: Tensor, n: int) -> Tensor:
        """
        Expand 1-stream to n-stream at TransformerBlock entry.
        
        Simple replication strategy: each stream initialized as a copy of input.
        
        Args:
            x: [s, b, C] - single stream hidden states
            n: Number of residual streams
        
        Returns:
            expanded: [s, b, n*C] - n-stream hidden states
        """
        s, b, C = x.shape
        # Replicate input to n streams
        expanded = x.unsqueeze(2).expand(s, b, n, C).contiguous()
        return expanded.view(s, b, n * C)
    
    @staticmethod
    def output_contract(x: Tensor, n: int) -> Tensor:
        """
        Contract n-stream to 1-stream at TransformerBlock exit.
        
        Simple averaging strategy: average all streams.
        
        Args:
            x: [s, b, n*C] - n-stream hidden states
            n: Number of residual streams
        
        Returns:
            contracted: [s, b, C] - single stream hidden states
        """
        s, b, nC = x.shape
        C = nC // n
        # Average all streams
        x_streams = x.view(s, b, n, C)
        contracted = x_streams.mean(dim=2)
        return contracted

    # ==================== Fused kernel placeholder ====================
    
    @nvtx_decorator(message="HyperConnection::fused_h_res_h_post_bda")
    def fused_h_res_h_post_bda(
        self,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager: Optional['MHCBlockRecomputeManager'] = None,
    ) -> Tensor:
        """
        Fused kernel combining apply_h_res, apply_h_post and bias-dropout-add.
        
        This is a placeholder for future kernel fusion optimization.
        Currently implements the operations sequentially using native PyTorch.
        
        The computation flow is:
            1. mixed = H_res @ original_residual (apply_h_res)
            2. expanded = H_post^T @ layer_output (apply_h_post)
            3. output = dropout(expanded + bias) + mixed (bias-dropout-add)
        
        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            original_residual: [s, b, n*C] - n-stream hidden states (before H_res applied)
            h_post: [s, b, n] - expansion weights
            layer_output_with_bias: Tuple of (x, bias) where:
                - x: [s, b, C] - layer output (attention or MLP output)
                - bias: [C] or None - optional bias tensor
            dropout_prob: Dropout probability
            training: Whether in training mode
            fused: Whether to use fused BDA implementation
            manager: Optional MHCBlockRecomputeManager for checkpoint management.
                When provided, each operation is wrapped with CheckpointWithoutOutput.
        
        Returns:
            output: [s, b, n*C] - final output after all operations
        """
        if manager is not None:
            return self._fused_h_res_h_post_bda_with_checkpoint(
                h_res, original_residual, h_post, layer_output_with_bias,
                dropout_prob, training, fused, manager
            )
        else:
            return self._fused_h_res_h_post_bda_native(
                h_res, original_residual, h_post, layer_output_with_bias,
                dropout_prob, training, fused
            )
    
    @nvtx_decorator(message="HyperConnection::fused_h_res_h_post_bda_native")
    def _fused_h_res_h_post_bda_native(
        self,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
    ) -> Tensor:
        """
        Native implementation of fused h_res, h_post and bda operations.
        
        When use_fused_kernel is True and TileLang is available, uses the
        FusedHPostBDA autograd function with fused TileLang kernels.
        Otherwise falls back to the sequential PyTorch implementation.
        
        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            original_residual: [s, b, n*C] - n-stream hidden states
            h_post: [s, b, n] - expansion weights
            layer_output_with_bias: Tuple of (x, bias)
            dropout_prob: Dropout probability
            training: Whether in training mode
            fused: Whether to use fused BDA implementation (for non-kernel path)
        
        Returns:
            output: [s, b, n*C] - final output
        """
        x, bias = layer_output_with_bias
        
        # Use fused kernel path when enabled
        if self.use_fused_kernel and is_tilelang_available():
            assert bias is not None, (
                "Fused H_post BDA kernel requires bias to be non-None. "
                "Please provide a bias tensor when using fused kernels."
            )
            assert dropout_prob == 0.0, (
                f"Fused H_post BDA kernel requires dropout_prob to be 0, "
                f"got {dropout_prob}. Dropout is not supported in fused kernels."
            )
            return FusedHPostBDA.apply(
                h_res, original_residual, h_post, x, bias,
                dropout_prob, training, True  # use_fused_kernel=True
            )
        
        # Fallback to sequential PyTorch implementation
        from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
        
        # Step 1: Apply H_res to original residual
        mixed = self.apply_h_res(h_res, original_residual)
        
        # Step 2: Apply H_post to layer output
        x_expanded = self._apply_h_post(x, h_post)
        bias_expanded = self._apply_h_post(bias, h_post) if bias is not None else None
        
        # Step 3: Bias-dropout-add
        bda_func = get_bias_dropout_add(training, fused)
        output = bda_func((x_expanded, bias_expanded), mixed, dropout_prob)
        
        return output
    
    @nvtx_decorator(message="HyperConnection::fused_h_res_h_post_bda_with_checkpoint")
    def _fused_h_res_h_post_bda_with_checkpoint(
        self,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager: 'MHCBlockRecomputeManager',
    ) -> Tensor:
        """
        Checkpointed implementation of fused h_res, h_post and bda operations.
        
        When use_fused_kernel is True and TileLang is available, uses the
        FusedHPostBDA autograd function directly (which has its own backward).
        Otherwise uses a single checkpoint wrapper around all operations.
        
        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            original_residual: [s, b, n*C] - n-stream hidden states
            h_post: [s, b, n] - expansion weights
            layer_output_with_bias: Tuple of (x, bias)
            dropout_prob: Dropout probability
            training: Whether in training mode
            fused: Whether to use fused BDA implementation (for non-kernel path)
            manager: MHCBlockRecomputeManager for checkpoint management
        
        Returns:
            output: [s, b, n*C] - final output
        """
        x, bias = layer_output_with_bias
        
        # Use fused kernel path when enabled - FusedHPostBDA has its own backward
        if self.use_fused_kernel and is_tilelang_available():
            assert bias is not None, (
                "Fused H_post BDA kernel requires bias to be non-None. "
                "Please provide a bias tensor when using fused kernels."
            )
            assert dropout_prob == 0.0, (
                f"Fused H_post BDA kernel requires dropout_prob to be 0, "
                f"got {dropout_prob}. Dropout is not supported in fused kernels."
            )
            return FusedHPostBDA.apply(
                h_res, original_residual, h_post, x, bias,
                dropout_prob, training, True  # use_fused_kernel=True
            )
        
        # Fallback to checkpointed sequential PyTorch implementation
        from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
        from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
        
        # Get BDA function (captured via closure)
        bda_func = get_bias_dropout_add(training, fused)
        
        has_bias = bias is not None
        
        # Native wrapper that combines all operations without internal checkpointing.
        # Non-tensor args (dropout_prob, has_bias) are captured via closure.
        def _native_wrapper(h_res, original_residual, h_post, x, *optional_bias):
            # Step 1: Apply H_res to original residual
            nvtx_range_push("HyperConnection::apply_h_res")
            mixed = self.apply_h_res(h_res, original_residual)
            nvtx_range_pop("HyperConnection::apply_h_res")
            
            # Step 2: Apply H_post to x and bias
            nvtx_range_push("HyperConnection::apply_h_post")
            x_expanded = self._apply_h_post(x, h_post)
            if has_bias:
                bias_expanded = self._apply_h_post(optional_bias[0], h_post)
            else:
                bias_expanded = None
            nvtx_range_pop("HyperConnection::apply_h_post")
            
            # Step 3: Bias-dropout-add
            nvtx_range_push("HyperConnection::bda")
            output = bda_func((x_expanded, bias_expanded), mixed, dropout_prob)
            nvtx_range_pop("HyperConnection::bda")
            
            return output
        
        # Use a single checkpoint wrapper for all operations
        ckpt = CheckpointWithoutOutput(ckpt_manager=manager)
        if has_bias:
            output = ckpt.checkpoint(_native_wrapper, h_res, original_residual, h_post, x, bias)
        else:
            output = ckpt.checkpoint(_native_wrapper, h_res, original_residual, h_post, x)
        
        return output


# ==================== Checkpoint utilities for mHC ====================

class HyperConnectionCheckpoint:
    """
    Checkpoint utility for mHC intermediate activations.
    
    Implements the paper's "recomputing strategy" to reduce memory footprint
    by discarding intermediate n-stream activations and recomputing on-the-fly.
    """
    
    @staticmethod
    def compute_optimal_block_size(num_layers: int, num_streams: int) -> int:
        """
        Compute optimal recomputation block size.
        
        From paper Eq. (20): L_r^* ≈ sqrt(nL/(n+2))
        
        Args:
            num_layers: Total number of transformer layers
            num_streams: Number of residual streams (n)
        
        Returns:
            block_size: Optimal block size for checkpointing
        """
        block_size = int(math.sqrt(num_streams * num_layers / (num_streams + 2)))
        return max(1, block_size)
