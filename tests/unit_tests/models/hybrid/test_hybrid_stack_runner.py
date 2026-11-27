# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the HybridStack layer runner.

``HybridStack.forward`` runs as ``_prepare_forward`` -> ``_layer_loop`` (one ``_run_layer``
per layer call) -> ``_finalize_forward``. These tests pin that composition, the hooks a
subclass overrides, and the per-call behaviour the runner carries (grouped CUDA-graph tail
skip, per-layer quantization context, hash-routing token IDs). They run on CPU (gloo) and
on CUDA.
"""

import contextlib
import os
import socket
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from megatron.core import parallel_state
from megatron.core import recompute as recompute_module
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.models.hybrid.hybrid_block import (
    HybridStack,
    HybridStackLayerContext,
    HybridStackSubmodules,
)
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, validate_segment_layers
from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mlp_layer import MLPLayer
from megatron.core.tensor_observation import capture_tensor_observations
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.unit_tests.test_utilities import Utils

_HIDDEN = 64
_SEQ = 16
_BATCH = 2


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    """Initialize model parallelism once for the module (NCCL on CUDA, gloo on CPU)."""
    if torch.cuda.is_available():
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()
    else:
        created_group = False
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(_free_port()))
            dist.init_process_group("gloo", rank=0, world_size=1)
            created_group = True
        parallel_state.initialize_model_parallel(1, 1)
        yield
        parallel_state.destroy_model_parallel()
        if created_group:
            dist.destroy_process_group()


@pytest.fixture
def mhc_on_this_device(monkeypatch):
    """Let mHC layers run eagerly on CPU (no-op NVTX ranges); CUDA runs unchanged."""
    if torch.cuda.is_available():
        return
    if os.environ.get("TORCHDYNAMO_DISABLE") != "1" and not torch._dynamo.config.disable:
        pytest.skip("mHC on CPU needs TORCHDYNAMO_DISABLE=1 (its fused ops use torch.compile)")
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda *a, **k: None)


def _device():
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"


class _SdpaCoreAttention(MegatronModule):
    """Causal SDPA core attention with DotProductAttention's call signature.

    The local ``DotProductAttention`` needs CUDA buffers; this keeps the dense stack
    runnable on CPU and on CUDA alike.
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
        # The math backend is deterministic on CUDA too (fused backward kernels may not be).
        with sdpa_kernel(SDPBackend.MATH):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.permute(2, 0, 1, 3).reshape(query.shape[0], query.shape[1], -1)


def _local_stack_submodules():
    """Local (non-TE) '*' and '-' hybrid layer specs."""
    local = get_gpt_layer_local_submodules(normalization="RMSNorm")
    self_attention = replace(
        local.self_attention,
        submodules=replace(local.self_attention.submodules, core_attention=_SdpaCoreAttention),
    )
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


def _config(num_layers, **overrides):
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=_HIDDEN,
        num_attention_heads=2,
        num_query_groups=1,
        kv_channels=32,
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
        is_hybrid_model=True,
        **overrides,
    )


def _build_stack(pattern, **overrides):
    torch.manual_seed(1234)
    config = _config(len(pattern), **overrides)
    stack = HybridStack(
        config=config,
        submodules=_local_stack_submodules(),
        layer_config_list=validate_segment_layers(pattern, config),
        post_layer_norm=False,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    return stack.to(_device())


def _rope(device):
    dim = 32
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = torch.outer(torch.arange(_SEQ, dtype=torch.float32), inv_freq)
    return torch.cat((freqs, freqs), dim=-1)[:, None, None, :].to(device)


def _inputs(device):
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(_SEQ, _BATCH, _HIDDEN, generator=generator).to(device)
    grad_out = torch.randn(_SEQ, _BATCH, _HIDDEN, generator=generator).to(device)
    return hidden, grad_out


def _forward_backward(stack, run):
    """Run ``run(stack, hidden, rope)`` and backward a fixed cotangent; return everything."""
    device = _device()
    hidden, grad_out = _inputs(device)
    hidden.requires_grad_(True)
    stack.zero_grad(set_to_none=True)
    output = run(stack, hidden, _rope(device))
    output.backward(grad_out)
    grads = {name: param.grad.detach().clone() for name, param in stack.named_parameters()}
    return output.detach().clone(), hidden.grad.detach().clone(), grads


def _run_forward(stack, hidden, rope):
    return stack(hidden_states=hidden, attention_mask=None, rotary_pos_emb=rope)


def _run_phases(stack, hidden, rope):
    """The composition ``forward`` uses, spelled out."""
    hidden, ctx = stack._prepare_forward(
        hidden,
        None,
        None,
        rope,
        packed_seq_params=None,
        padding_mask=None,
        input_ids=None,
        packed_seq_params_by_layout=None,
        cp_layout_plan=None,
    )
    assert isinstance(ctx, HybridStackLayerContext)
    with ctx.outer_quant_context:
        for l_no in range(len(stack.layers)):
            hidden = stack._run_layer(l_no, hidden, ctx)
    assert ctx.grouped_tail_to_skip is None
    assert ctx.forward_adapter_context is None
    return stack._finalize_forward(hidden)


def _assert_bitwise(reference, candidate):
    ref_out, ref_dx, ref_grads = reference
    out, dx, grads = candidate
    assert torch.equal(out, ref_out)
    assert torch.equal(dx, ref_dx)
    assert grads.keys() == ref_grads.keys()
    for name, grad in ref_grads.items():
        assert torch.equal(grads[name], grad), name


def test_forward_equals_phased_runner_dense():
    stack = _build_stack("*-*-*-*-")
    _assert_bitwise(_forward_backward(stack, _run_forward), _forward_backward(stack, _run_phases))


def test_forward_equals_phased_runner_mhc(mhc_on_this_device):
    stack = _build_stack("*-*-", enable_mhc_connections=True, mhc_num_residual_streams=4)
    assert all(isinstance(layer, HyperConnectionHybridLayer) for layer in stack.layers)
    _assert_bitwise(_forward_backward(stack, _run_forward), _forward_backward(stack, _run_phases))


def test_observed_mhc_forward_takes_module_call_path_per_layer(mhc_on_this_device):
    """With residual observation on, every physical layer is observed once, in order,
    and the observed output equals the unobserved one."""
    stack = _build_stack("*-*-", enable_mhc_connections=True, mhc_num_residual_streams=4)
    reference = _forward_backward(stack, _run_forward)
    owners = []

    def observer(owner, name, source_kind, tensor, *_dims):
        owners.append((owner, source_kind))

    with capture_tensor_observations(observer, frozenset({"residual_accumulator"})):
        observed = _forward_backward(stack, _run_forward)
    assert owners == [(layer, "residual_accumulator") for layer in stack.layers]
    _assert_bitwise(reference, observed)


# ---------------------------------------------------------------------------
# Runner hooks and per-call behaviour, with stand-in layers (no process group needed).
# ---------------------------------------------------------------------------


class _SizeOneGroup:
    @staticmethod
    def size():
        return 1

    @staticmethod
    def rank():
        return 0


class _RecordingTransformerLayer(TransformerLayer):
    """TransformerLayer stand-in that adds ``offset`` and records its keyword arguments."""

    def __init__(self, layer_number, offset=1.0, is_hash_layer=False):
        torch.nn.Module.__init__(self)
        self.layer_number = layer_number
        self.offset = offset
        self.config = SimpleNamespace(cuda_graph_impl="none")
        self.mlp = SimpleNamespace(router=SimpleNamespace(is_hash_layer=is_hash_layer))
        self.calls = []

    def forward(self, hidden_states, **kwargs):
        self.calls.append(kwargs)
        return hidden_states + self.offset, None


class _GroupedHeadLayer(HyperConnectionHybridLayer):
    """mHC-wrapper stand-in whose (simulated) graph replay also ran ``tail``."""

    def __init__(self, layer_number, offset):
        torch.nn.Module.__init__(self)
        self.layer_number = layer_number
        self.offset = offset
        self.tail = None
        self.calls = 0

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, hidden_states, **kwargs):
        self.calls += 1
        offset = self.offset + (self.tail.offset if self.tail is not None else 0.0)
        return hidden_states + offset, None

    def _get_active_te_cuda_graph_group_tail(self):
        return self.tail


def _make_stub_stack(layers, stack_cls=HybridStack, **config_overrides):
    """Production stack/config initialization with the built layers replaced."""
    config = TransformerConfig(
        num_layers=len(layers),
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        **config_overrides,
    )
    group = _SizeOneGroup()
    with mock.patch("megatron.core.models.hybrid.hybrid_block.build_module", side_effect=layers):
        return stack_cls(
            config=config,
            submodules=HybridStackSubmodules(),
            layer_config_list=validate_segment_layers(Symbols.ATTENTION * len(layers), config),
            post_process=False,
            post_layer_norm=False,
            pg_collection=SimpleNamespace(pp=group, tp=group, cp=group, tp_cp=group),
        )


class _LoopRecordingStack(HybridStack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_loop_calls = 0

    def _layer_loop(self, hidden_states, ctx):
        self.layer_loop_calls += 1
        return super()._layer_loop(hidden_states, ctx)


def test_layer_loop_is_the_eager_hook_and_full_recompute_bypasses_it(monkeypatch):
    # checkpointed_forward checkpoints through precision_aware_checkpoint (upstream dev).
    monkeypatch.setattr(
        recompute_module,
        "precision_aware_checkpoint",
        lambda function, _config, _tp_group, *args, distribute_saved_activations=False: function(
            *args
        ),
    )
    hidden = torch.zeros(4, 2, 8)

    layers = [_RecordingTransformerLayer(1), _RecordingTransformerLayer(2)]
    eager = _make_stub_stack(layers, stack_cls=_LoopRecordingStack)
    output = eager(hidden_states=hidden, attention_mask=None)
    assert eager.layer_loop_calls == 1
    assert torch.equal(output, hidden + 2.0)
    assert [len(layer.calls) for layer in layers] == [1, 1]

    layers = [_RecordingTransformerLayer(1), _RecordingTransformerLayer(2)]
    full = _make_stub_stack(
        layers,
        stack_cls=_LoopRecordingStack,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=2,
    )
    output = full(hidden_states=hidden, attention_mask=None)
    assert full.layer_loop_calls == 0
    assert torch.equal(output, hidden + 2.0)
    assert [len(layer.calls) for layer in layers] == [1, 1]


@pytest.mark.parametrize("with_cp_layout_state", [False, True])
def test_grouped_tail_is_skipped_and_cleared(with_cp_layout_state):
    head = _GroupedHeadLayer(1, offset=1.0)
    tail = _RecordingTransformerLayer(2, offset=10.0)
    last = _RecordingTransformerLayer(3, offset=100.0)
    head.tail = tail
    stack = _make_stub_stack([head, tail, last])
    finalized = []
    if with_cp_layout_state:
        manager = SimpleNamespace(layer_layouts=("zigzag", "zigzag", "zigzag"))
        state = SimpleNamespace(
            manager=manager,
            prepare_layer=lambda _index, hidden: (hidden, None),
            get_layer_padding_mask=lambda _index, default, _masks: default,
            finalize_layer=lambda index, hidden: finalized.append(index) or hidden,
        )
        manager.build_forward_state = lambda *_args, **_kwargs: state
        stack._cp_layout_manager = manager

    hidden = torch.zeros(4, 2, 8)
    hidden_in, ctx = stack._prepare_forward(
        hidden,
        None,
        None,
        None,
        packed_seq_params=None,
        padding_mask=None,
        input_ids=None,
        packed_seq_params_by_layout=None,
        cp_layout_plan=None,
    )
    output = stack._layer_loop(hidden_in, ctx)

    assert torch.equal(output, hidden + 111.0)
    assert head.calls == 1
    assert tail.calls == []
    assert len(last.calls) == 1
    assert ctx.grouped_tail_to_skip is None
    if with_cp_layout_state:
        # The grouped head finalizes in its tail's layout; the skipped tail does not run.
        assert finalized == [1, 2]


def test_run_layer_sets_grouped_tail_for_the_loop():
    head = _GroupedHeadLayer(1, offset=1.0)
    tail = _RecordingTransformerLayer(2)
    head.tail = tail
    stack = _make_stub_stack([head, tail])
    hidden, ctx = stack._prepare_forward(
        torch.zeros(4, 2, 8),
        None,
        None,
        None,
        packed_seq_params=None,
        padding_mask=None,
        input_ids=None,
        packed_seq_params_by_layout=None,
        cp_layout_plan=None,
    )
    stack._run_layer(0, hidden, ctx)
    assert ctx.grouped_tail_to_skip is tail


def test_quant_context_factory_gets_layer_config_and_zero_based_layer_number():
    layers = [_RecordingTransformerLayer(5), _RecordingTransformerLayer(6)]
    stack = _make_stub_stack(layers)
    hidden, ctx = stack._prepare_forward(
        torch.zeros(4, 2, 8),
        None,
        None,
        None,
        packed_seq_params=None,
        padding_mask=None,
        input_ids=None,
        packed_seq_params_by_layout=None,
        cp_layout_plan=None,
    )
    assert not ctx.use_inner_quantization_context
    assert isinstance(
        ctx.quant_context_factory(stack.layer_config_list[0], 0), type(contextlib.nullcontext())
    )
    calls = []

    def recording_factory(config, layer_number):
        calls.append((config, layer_number))
        return contextlib.nullcontext()

    ctx.quant_context_factory = recording_factory
    stack._layer_loop(hidden, ctx)
    assert len(calls) == 2
    assert calls[0][0] is stack.layer_config_list[0] and calls[0][1] == 4
    assert calls[1][0] is stack.layer_config_list[1] and calls[1][1] == 5


def test_input_ids_reach_only_hash_routed_layers():
    hashed = _RecordingTransformerLayer(1, is_hash_layer=True)
    learned = _RecordingTransformerLayer(2, is_hash_layer=False)
    stack = _make_stub_stack([hashed, learned])
    input_ids = torch.arange(8).reshape(2, 4)
    hidden, ctx = stack._prepare_forward(
        torch.zeros(4, 2, 8),
        None,
        None,
        None,
        packed_seq_params=None,
        padding_mask=None,
        input_ids=input_ids,
        packed_seq_params_by_layout=None,
        cp_layout_plan=None,
    )
    for l_no in range(len(stack.layers)):
        hidden = stack._run_layer(l_no, hidden, ctx)
    assert hashed.calls[0]["input_ids"] is input_ids
    assert "input_ids" not in learned.calls[0]
