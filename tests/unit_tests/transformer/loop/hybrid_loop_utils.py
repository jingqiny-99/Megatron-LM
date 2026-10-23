# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Test helpers for the hybrid loop (LoopHybridStack): device-agnostic local layer specs,
configs, the model-parallel fixture, and the GPT-loop -> hybrid-loop weight map.

The local (non-TE) specs replace core attention with a deterministic SDPA module, because the
local ``DotProductAttention`` needs CUDA buffers. With them, dense and mHC stacks run on CPU
(gloo) and on CUDA alike.
"""

import contextlib
import os
import socket
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mlp_layer import MLPLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.unit_tests.test_utilities import Utils

HIDDEN = 64
SEQ = 16
BATCH = 2
KV_CHANNELS = 32

CUDA = torch.cuda.is_available()


def device():
    """The test device: the current CUDA device, else CPU."""
    return torch.device("cuda", torch.cuda.current_device()) if CUDA else torch.device("cpu")


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def model_parallel_once(**parallel_sizes):
    """Initialize model parallelism (NCCL on CUDA, gloo on CPU) for one test module.

    ``parallel_sizes`` (e.g. ``expert_model_parallel_size=2``) go to
    ``initialize_model_parallel``; a module needing them initializes once with them instead of
    re-initializing inside its tests (the CI process-group budget).
    """
    if CUDA:
        Utils.initialize_model_parallel(1, 1, **parallel_sizes)
        model_parallel_cuda_manual_seed(123)
        try:
            yield
        finally:
            Utils.destroy_model_parallel()
    else:
        created_group = False
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(_free_port()))
            dist.init_process_group("gloo", rank=0, world_size=1)
            created_group = True
        parallel_state.initialize_model_parallel(1, 1, **parallel_sizes)
        try:
            yield
        finally:
            parallel_state.destroy_model_parallel()
            if created_group:
                dist.destroy_process_group()


@pytest.fixture
def mhc_on_this_device(monkeypatch):
    """Let mHC layers run eagerly on CPU (no-op NVTX ranges); CUDA runs unchanged."""
    if CUDA:
        return
    if os.environ.get("TORCHDYNAMO_DISABLE") != "1" and not torch._dynamo.config.disable:
        pytest.skip("mHC on CPU needs TORCHDYNAMO_DISABLE=1 (its fused ops use torch.compile)")
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda *a, **k: None)


class SdpaCoreAttention(MegatronModule):
    """SDPA core attention with ``DotProductAttention``'s call signature.

    Causal without a mask; with a boolean mask (``True`` = masked, the MCore convention, as
    masked MoR passes it) it attends where the mask is ``False``. The math backend keeps CUDA
    runs deterministic.
    """

    def __init__(
        self,
        config,
        layer_number,
        attn_mask_type,
        attention_type,
        attention_dropout=None,
        softmax_scale=None,
        cp_comm_type=None,
        pg_collection=None,
    ):
        super().__init__(config=config)

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
    ):
        repeats = query.shape[2] // key.shape[2]
        key = key.repeat_interleave(repeats, dim=2)
        value = value.repeat_interleave(repeats, dim=2)
        # [s, b, h, d] -> [b, h, s, d]
        q, k, v = (t.permute(1, 2, 0, 3) for t in (query, key, value))
        with sdpa_kernel(SDPBackend.MATH):
            if attention_mask is None:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=~attention_mask)
        return out.permute(2, 0, 1, 3).reshape(query.shape[0], query.shape[1], -1)


def _local_submodules():
    local = get_gpt_layer_local_submodules(normalization="RMSNorm")
    self_attention = replace(
        local.self_attention,
        submodules=replace(local.self_attention.submodules, core_attention=SdpaCoreAttention),
    )
    return local, self_attention


def local_gpt_layer_spec():
    """GPT TransformerLayer (attention + MLP) with local modules and SDPA core attention."""
    local, self_attention = _local_submodules()
    return ModuleSpec(
        module=TransformerLayer, submodules=replace(local, self_attention=self_attention)
    )


def local_hybrid_submodules():
    """Local '*' and '-' hybrid layer specs, the halves of ``local_gpt_layer_spec``."""
    local, self_attention = _local_submodules()
    attention_layer = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=local.input_layernorm,
            self_attention=self_attention,
            self_attn_bda=local.self_attn_bda,
        ),
    )
    mlp_layer = ModuleSpec(
        module=MLPLayer,
        submodules=TransformerLayerSubmodules(
            pre_mlp_layernorm=local.pre_mlp_layernorm, mlp=local.mlp, mlp_bda=local.mlp_bda
        ),
    )
    return HybridStackSubmodules(attention_layer=attention_layer, mlp_layer=mlp_layer)


def dense_config(num_layers, **overrides):
    """Small fp32 dense config with the fusions off (CPU-runnable)."""
    fields = dict(
        num_layers=num_layers,
        hidden_size=HIDDEN,
        num_attention_heads=2,
        num_query_groups=1,
        kv_channels=KV_CHANNELS,
        ffn_hidden_size=128,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        apply_rope_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        masked_softmax_fusion=False,
        attention_backend=AttnBackend.local,
    )
    fields.update(overrides)
    return TransformerConfig(**fields)


def hybrid_loop_config(pattern, prelude, coda, **overrides):
    """Dense hybrid loop config for ``pattern`` (Huginn depth 2 unless overridden)."""
    fields = dict(
        is_hybrid_model=True,
        loop_architecture="huginn",
        loop_iterations=2,
        loop_prelude_sublayers=prelude,
        loop_coda_sublayers=coda,
    )
    fields.update(overrides)
    return dense_config(len(pattern), **fields)


def build_local_loop_stack(pattern, prelude, coda, *, seed=1234, **overrides):
    """A LoopHybridStack of local layers, without the TE final norm, on the test device."""
    torch.manual_seed(seed)
    config = hybrid_loop_config(pattern, prelude, coda, **overrides)
    stack = LoopHybridStack(
        config=config,
        submodules=local_hybrid_submodules(),
        layer_config_list=validate_segment_layers(pattern, config),
        post_layer_norm=False,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    return stack.to(device())


def rope(seq=SEQ, dim=KV_CHANNELS, dev=None):
    """A ``[S, 1, 1, dim]`` RoPE frequency tensor (``RotaryEmbedding.forward`` needs CUDA)."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = torch.outer(torch.arange(seq, dtype=torch.float32), inv_freq)
    return torch.cat((freqs, freqs), dim=-1)[:, None, None, :].to(dev or device())


_ATTENTION_HALF = ("input_layernorm", "self_attention", "self_attn_bda")
_MLP_HALF = ("pre_mlp_layernorm", "mlp", "mlp_bda")


def gpt_to_hybrid_key(key):
    """Map a GPT-loop block state key to the dense ``('*-')^n`` hybrid loop stack key.

    GPT layer ``i`` becomes hybrid sublayers ``2i`` ('*': input norm + attention) and ``2i + 1``
    ('-': pre-MLP norm + MLP); the final norm is ``final_norm``; the loop parameters keep their
    names.
    """
    if key.startswith("layers."):
        _, index, rest = key.split(".", 2)
        index = int(index)
        head = rest.split(".", 1)[0]
        if head in _ATTENTION_HALF:
            return f"layers.{2 * index}.{rest}"
        if head in _MLP_HALF:
            return f"layers.{2 * index + 1}.{rest}"
        raise KeyError(f"Unmapped GPT layer key {key}")
    if key.startswith("final_layernorm."):
        return "final_norm." + key[len("final_layernorm.") :]
    return key


def copy_gpt_weights(gpt_block, hybrid_stack):
    """Load the GPT block's state into the hybrid stack through ``gpt_to_hybrid_key``."""
    mapped = {gpt_to_hybrid_key(key): value for key, value in gpt_block.state_dict().items()}
    assert set(mapped) == set(hybrid_stack.state_dict()), sorted(
        set(mapped) ^ set(hybrid_stack.state_dict())
    )
    hybrid_stack.load_state_dict(mapped, strict=True)
