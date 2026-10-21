# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Experimental depth recurrence over one registered bank of GPT layers.

See specs/loop_transformer.md in the development toolkit for model provenance.
No inference cache, pipeline schedule, or activation sharing is implied by
parameter sharing.
"""

from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_observation import observe_layer_residuals, suspend_tensor_observations
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_block import TransformerBlock


class LoopRMSNorm(nn.Module):
    """FP32 statistics, with a replicated learned scale."""

    def __init__(self, width, eps, dtype, device):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x):
        """Normalize the last dimension in FP32 and apply the learned scale."""
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def token_depth_distribution(logits, valid, balance_coeff, z_coeff):
    """Token-local top-1 selection and separately differentiable router losses."""
    probabilities = logits.float().softmax(-1)
    depth = probabilities.argmax(-1)
    selected = probabilities.gather(-1, depth.unsqueeze(-1)).squeeze(-1)
    valid_f = valid.float()
    denominator = valid_f.sum().clamp_min(1)
    frequency = (F.one_hot(depth, logits.shape[-1]).float() * valid_f[..., None]).sum(
        dim=(0, 1)
    ) / denominator
    mean_probability = (probabilities * valid_f[..., None]).sum(dim=(0, 1)) / denominator
    balance = balance_coeff * logits.shape[-1] * (frequency * mean_probability).sum()
    z_loss = z_coeff * (logits.float().logsumexp(-1).square() * valid_f).sum() / denominator
    return depth + 1, selected, balance, z_loss, frequency


def active_causal_mask(active):
    """Boolean [B,1,S,S] mask. Inactive queries attend themselves to avoid NaNs."""
    sequence = active.shape[0]
    causal = torch.ones(sequence, sequence, dtype=torch.bool, device=active.device).tril()
    active_bs = active.transpose(0, 1)
    allowed = causal[None] & active_bs[:, None, :] & active_bs[:, :, None]
    allowed |= torch.eye(sequence, dtype=torch.bool, device=active.device)[None] & (
        ~active_bs[:, :, None]
    )
    return ~allowed[:, None]


@contextmanager
def recompute_context(name):
    """Profile an activation-checkpoint replay without re-observing its tensors.

    The original forward already notified the tensor observer, so the replay
    is suppressed the same way MCore's own checkpoint implementations do.
    """
    with torch.profiler.record_function(name), suspend_tensor_observations():
        yield


def compact_metadata(active):
    """Sequence-major indices, cumulative lengths, and original RoPE positions."""
    sequence = active.shape[0]
    indices = active.transpose(0, 1).reshape(-1).nonzero(as_tuple=True)[0]
    lengths = active.sum(0, dtype=torch.int32)
    lengths = lengths[lengths > 0]
    cumulative = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    # Shape discovery necessarily synchronizes eager dynamic compaction. Keep
    # this visible in profiler ranges; don't hide it in reported attention time.
    maximum = int(lengths.max().item()) if lengths.numel() else 0
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cumulative,
        cu_seqlens_kv=cumulative,
        max_seqlen_q=maximum,
        max_seqlen_kv=maximum,
        rotary_position_ids=indices.remainder(sequence),
    )
    return indices, params


class LoopTransformerBlock(TransformerBlock):
    """Shared physical layer bank with Huginn or token-choice MoR recurrence."""

    def __init__(self, *args, **kwargs):
        config = kwargs["config"]
        self._validate_config(config)
        super().__init__(*args, **kwargs)
        device = "cpu" if config.use_cpu_initialization else torch.cuda.current_device()
        dtype = config.params_dtype
        if config.loop_architecture == "huginn":
            self.loop_adapter = nn.Linear(
                2 * config.hidden_size, config.hidden_size, bias=False, dtype=dtype, device=device
            )
            config.init_method(self.loop_adapter.weight)
            self.loop_norm = LoopRMSNorm(
                config.hidden_size, config.layernorm_epsilon, dtype, device
            )
        else:
            self.depth_router = nn.Sequential(
                nn.Linear(
                    config.hidden_size, config.hidden_size, bias=False, dtype=dtype, device=device
                ),
                nn.SiLU(),
                nn.Linear(
                    config.hidden_size,
                    config.loop_iterations,
                    bias=False,
                    dtype=dtype,
                    device=device,
                ),
            )
            for module in self.depth_router:
                if isinstance(module, nn.Linear):
                    config.init_method(module.weight)
            if config.loop_token_execution == "masked":
                for layer in self.layers[self.core_start : self.core_end]:
                    layer.self_attention.attn_mask_type = AttnMaskType.arbitrary
        self.loop_losses = {}
        self.loop_metrics = {}
        self._microbatch_id = 0

    @staticmethod
    def _validate_config(config):
        unsupported = {
            "pipeline parallelism": config.pipeline_model_parallel_size != 1,
            "context parallelism": config.context_parallel_size != 1
            or config.dynamic_context_parallel,
            "sequence parallelism": config.sequence_parallel,
            "FP8": bool(config.fp8),
            "FP4": bool(config.fp4),
            "MTP": bool(config.mtp_num_layers),
            "CPU offload": config.cpu_offloading,
            "mHC": config.enable_hyper_connections,
            "standard recompute (use loop_recompute)": config.recompute_granularity is not None,
            "MoE communication overlap": config.overlap_moe_expert_parallel_comm,
            "dropout": bool(config.hidden_dropout or config.attention_dropout),
            "multi-latent attention": config.multi_latent_attention,
            # Hash routing needs per-token input ids in every MoE layer, and compact
            # MoR would also have to gather them. Not supported yet.
            "hash-routed MoE": config.moe_num_hash_layers > 0,
            # GTP prefetches weights along a static next-weight chain, which a
            # shared core revisited R times per microbatch breaks.
            "GTP weight remat": config.gtp_weight_remat_size > 1
            or config.expert_gtp_weight_remat_size > 1,
        }
        enabled = [name for name, value in unsupported.items() if value]
        if enabled:
            raise ValueError("Loop decoder does not support: " + ", ".join(enabled))
        if config.cuda_graph_impl not in ("none", "local"):
            raise ValueError("Loop decoder supports only local CUDA graphs")
        if config.loop_architecture == "mor" and config.cuda_graph_impl != "none":
            raise ValueError("Token-choice loops currently require eager execution")
        if config.cuda_graph_impl != "none" and (
            config.loop_sample_depth
            or config.loop_recompute
            or config.loop_bptt_iterations
            or config.num_moe_experts
            or config.loop_iterations != 4
        ):
            raise ValueError("Loop CUDA graphs require fixed R=4 dense full BPTT without recompute")

    @property
    def core_start(self):
        """Index of the first shared core layer."""
        return self.config.loop_prelude_layers

    @property
    def core_end(self):
        """Index one past the last core layer (all untied banks)."""
        banks = self.config.loop_iterations if self.config.loop_untie_core else 1
        return self.core_start + self.config.loop_core_layers * banks

    def _layers(self, x, start, end, label, **kwargs):
        for physical in range(start, end):
            with torch.profiler.record_function(f"{label}/layer={physical}"):
                residual_accumulator = x
                if x.shape[0] == 0:
                    # A rank with no queries still receives EP tokens. Keep its
                    # MLP dispatch/combine and backward in the collective order.
                    if self.layers[physical].is_moe_layer:
                        update, _ = self.layers[physical].mlp(x)
                        x = x + update
                else:
                    x, _ = self.layers[physical](hidden_states=x, **kwargs)
                # Same per-layer-call site as TransformerBlock.forward: outside
                # any per-layer CUDA graph, once per layer call (R times for a
                # shared core layer).
                observe_layer_residuals(self.layers[physical], residual_accumulator, x)
        if self.config.cuda_graph_impl == "local" and end > start:
            # The prelude injection and recurrence norm are eager consumers.
            # Local layer graphs return graph-pool views whose lifetime only
            # covers captured consumers. Own the segment output outside that
            # pool before a later graph reuses its storage.
            x = x.clone()
        return x

    def _record_work(
        self,
        tokens,
        squared_lengths,
        start,
        end,
        *,
        backward=True,
        recompute=False,
        extra=0,
        arbitrary_mask=False,
    ):
        """Estimate GEMM work with FMA=2; count backward and replay separately.

        This excludes norms, softmax, optimizer, dispatch, and indexing. The
        arbitrary masked backend executes a full attention square. Backward
        is estimated as twice forward, following MCore's reporting convention.
        """
        c = self.config
        h = c.hidden_size
        q_width = c.num_attention_heads * c.kv_channels
        kv_width = c.num_query_groups * c.kv_channels
        work = extra
        for physical in range(start, end):
            work = work + 4 * tokens * h * (q_width + kv_width)
            work = work + (4 if arbitrary_mask else 2) * squared_lengths * q_width
            if self.layers[physical].is_moe_layer:
                ffn = c.moe_ffn_hidden_size * c.moe_router_topk
                ffn += c.moe_shared_expert_intermediate_size or 0
                work = work + 2 * tokens * h * c.num_moe_experts
            else:
                ffn = c.ffn_hidden_size
            work = work + (6 if c.gated_linear_unit else 4) * tokens * h * ffn
        self._work[0] = self._work[0] + work
        if backward and torch.is_grad_enabled():
            self._work[1] = self._work[1] + 2 * work
            if recompute:
                self._work[2] = self._work[2] + work

    def _depth(self, device):
        if not self.config.loop_sample_depth or not self.training:
            return self.config.loop_iterations
        depth = torch.tensor([1, 2, 4, 8])[torch.randint(4, (1,))].to(device)
        # PP=CP=1. TP+EP ranks must take the same number of collective calls.
        # DP-only domains are allowed different depths; their backward remains
        # one reduction per unique parameter.
        group = self.pg_collection.tp_ep
        if group is not None and dist.get_world_size(group) > 1:
            dist.broadcast(depth, src=dist.get_global_rank(group, 0), group=group)
        return int(depth.item())

    def _huginn(self, embedding, kwargs, label):
        depth = self._depth(embedding.device)
        differentiated = self.config.loop_bptt_iterations or depth
        first_grad = max(0, depth - differentiated)
        state = torch.zeros_like(embedding)
        for step in range(depth):
            tokens = embedding.shape[0] * embedding.shape[1]
            core_start = self.core_start + (
                step * self.config.loop_core_layers if self.config.loop_untie_core else 0
            )
            core_end = core_start + self.config.loop_core_layers
            self._record_work(
                tokens,
                tokens * embedding.shape[0],
                core_start,
                core_end,
                backward=step >= first_grad,
                recompute=self.config.loop_recompute,
                extra=4 * tokens * self.config.hidden_size**2,
            )

            # Bind the step in the checkpoint closure: backward runs after this
            # Python loop, so capturing its mutable variable would mislabel replay.
            def recurrence(
                previous, injection, step=step, core_start=core_start, core_end=core_end
            ):
                with torch.profiler.record_function(f"{label}/loop={step + 1}"):
                    adapted = self.loop_adapter(torch.cat((previous, injection), -1))
                    output = self._layers(
                        adapted, core_start, core_end, f"{label}/loop={step + 1}", **kwargs
                    )
                    return self.loop_norm(output)

            context = torch.no_grad() if step < first_grad else nullcontext()
            with context:
                if self.config.loop_recompute and torch.is_grad_enabled():
                    state = checkpoint(
                        recurrence,
                        state,
                        embedding,
                        use_reentrant=False,
                        context_fn=lambda step=step: (
                            torch.profiler.record_function(
                                f"{label}/loop={step + 1}/phase=forward"
                            ),
                            recompute_context(f"{label}/loop={step + 1}/phase=recompute"),
                        ),
                    )
                else:
                    state = recurrence(state, embedding)
        tokens = embedding.shape[0] * embedding.shape[1]
        device = embedding.device
        self.loop_metrics = {
            "loop depth": torch.tensor(float(depth), device=device),
            "loop active token layers": torch.tensor(
                float(tokens * depth * self.config.loop_core_layers), device=device
            ),
            "loop backward token layers": torch.tensor(
                (
                    float(tokens * min(depth, differentiated) * self.config.loop_core_layers)
                    if torch.is_grad_enabled()
                    else 0.0
                ),
                device=device,
            ),
        }
        return state

    def _mor(self, embedding, kwargs, label):
        valid = torch.ones(embedding.shape[:2], dtype=torch.bool, device=embedding.device)
        if kwargs["padding_mask"] is not None:
            valid &= ~kwargs["padding_mask"].transpose(0, 1)
        with torch.profiler.record_function(f"{label}/depth_router"):
            logits = self.depth_router(embedding)
            depth, selected, balance, z_loss, frequency = token_depth_distribution(
                logits,
                valid,
                self.config.loop_router_balance_coeff,
                self.config.loop_router_z_loss_coeff,
            )
        self.loop_losses = {"depth balance loss": balance, "depth z loss": z_loss}
        tokens = embedding.shape[0] * embedding.shape[1]
        self._record_work(
            tokens,
            0,
            0,
            0,
            extra=2
            * tokens
            * self.config.hidden_size
            * (self.config.hidden_size + self.config.loop_iterations),
        )
        state = embedding
        active_counts = []
        for step in range(1, self.config.loop_iterations + 1):
            active = (depth >= step) & valid
            active_counts.append(active.sum().detach())
            loop_label = f"{label}/loop={step}"
            if self.config.loop_token_execution == "masked":
                loop_kwargs = dict(
                    kwargs,
                    attention_mask=active_causal_mask(active),
                    padding_mask=~active.transpose(0, 1),
                )
                indices = None
                layer_input = state
            else:
                with torch.profiler.record_function(f"{loop_label}/compact"):
                    indices, packed = compact_metadata(active)
                    flat = state.transpose(0, 1).reshape(-1, state.shape[-1])
                    layer_input = flat.index_select(0, indices).unsqueeze(1)
                loop_kwargs = dict(
                    kwargs, attention_mask=None, padding_mask=None, packed_seq_params=packed
                )

            issued_tokens = tokens if indices is None else indices.numel()
            squared_lengths = (
                tokens * embedding.shape[0]
                if indices is None
                else active.sum(0).float().square().sum().detach()
            )
            self._record_work(
                issued_tokens,
                squared_lengths,
                self.core_start,
                self.core_end,
                recompute=self.config.loop_recompute,
                arbitrary_mask=indices is None,
            )

            def recurrence(x, loop_kwargs=loop_kwargs, loop_label=loop_label):
                with torch.profiler.record_function(loop_label):
                    return self._layers(
                        x, self.core_start, self.core_end, loop_label, **loop_kwargs
                    )

            if self.config.loop_recompute and torch.is_grad_enabled():
                updated = checkpoint(
                    recurrence,
                    layer_input,
                    use_reentrant=False,
                    context_fn=lambda loop_label=loop_label: (
                        torch.profiler.record_function(f"{loop_label}/phase=forward"),
                        recompute_context(f"{loop_label}/phase=recompute"),
                    ),
                )
            else:
                updated = recurrence(layer_input)
            if indices is None:
                state = torch.where(active[..., None], updated, state)
            else:
                with torch.profiler.record_function(f"{loop_label}/scatter"):
                    # index_copy is differentiable even for empty updates: this
                    # retains the empty rank's EP backward in the autograd graph.
                    flat = flat.index_copy(0, indices, updated.squeeze(1))
                    state = flat.view(embedding.shape[1], embedding.shape[0], -1).transpose(0, 1)
        counts = torch.stack(active_counts).float()
        self.loop_metrics = {
            "loop depth": counts.sum() / valid.sum().clamp_min(1),
            "loop active token layers": counts.sum() * self.config.loop_core_layers,
            "loop backward token layers": counts.sum()
            * self.config.loop_core_layers
            * int(torch.is_grad_enabled()),
        }
        for index, fraction in enumerate(frequency.detach()):
            self.loop_metrics[f"depth fraction {index + 1}"] = fraction
        # Gate gradients reduce over hidden channels and can nearly cancel
        # after the final RMSNorm. Accumulate the weighted residual in FP32,
        # rather than rounding the gate and its reduction to BF16.
        return (embedding.float() + selected[..., None] * state.float()).to(state.dtype)

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        padding_mask=None,
        extract_layer_indices=None,
        input_ids=None,
        *,
        inference_params=None,
        dynamic_inference_decode_only=None,
    ):
        """Run prelude, recurrent core, and coda; same signature as TransformerBlock."""
        if inference_context is not None or inference_params is not None:
            raise ValueError("Loop decoder implements training, without an inference KV cache")
        if packed_seq_params is not None or context is not None or extract_layer_indices:
            raise ValueError("Loop decoder currently requires ordinary SBHD causal GPT input")
        if attention_bias is not None:
            raise ValueError("Loop decoder does not support external attention bias")
        self.loop_losses = {}
        hidden_states = self.preprocess_for_layer_schedule(hidden_states)
        self._work = [0.0, 0.0, 0.0]
        kwargs = dict(
            attention_mask=attention_mask,
            context_mask=context_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            padding_mask=padding_mask,
            packed_seq_params=None,
        )
        label = f"loop_transformer/mb={self._microbatch_id}"
        self._microbatch_id += 1
        with torch.profiler.record_function(label):
            tokens = hidden_states.shape[0] * hidden_states.shape[1]
            square_lengths = tokens * hidden_states.shape[0]
            self._record_work(tokens, square_lengths, 0, self.core_start)
            embedding = self._layers(
                hidden_states, 0, self.core_start, label + "/prelude", **kwargs
            )
            if self.config.loop_architecture == "huginn":
                state = self._huginn(embedding, kwargs, label)
            else:
                state = self._mor(embedding, kwargs, label)
            self._record_work(
                tokens,
                square_lengths,
                self.core_end,
                len(self.layers),
                extra=2 * tokens * self.config.hidden_size * getattr(self, "loop_vocab_size", 0),
            )
            state = self._layers(state, self.core_end, len(self.layers), label + "/coda", **kwargs)
            for phase, value in zip(("forward", "backward", "recompute"), self._work):
                self.loop_metrics[f"loop {phase} flops per token"] = (
                    torch.as_tensor(value, dtype=torch.float32, device=hidden_states.device)
                    / tokens
                )
            return self.postprocess_for_layer_schedule(state)
