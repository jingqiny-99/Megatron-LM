# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttention,
    CompressedSparseAttentionSubmodules,
)
from megatron.core.transformer.loop.cross_loop_kv import current_cross_loop_pass
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack, LoopResidualGate


class _CoreLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.sparse = nn.Linear(4, 4, bias=False)
        self.local = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        scope = current_cross_loop_pass()
        if scope.is_source:
            kv = self.sparse(x)
            scope.publish(self, "kv", kv)
        else:
            kv = scope.read(self, "kv")
        return (kv + self.local(x)).tanh()


class _Stack(nn.Module):
    """Run actual loop and checkpoint boundaries with two small differentiable KV producers."""

    _huginn = LoopHybridStack._huginn
    _checkpointed_blocks = LoopHybridStack._checkpointed_blocks

    def __init__(self, depth, mode, carrier, streams=1):
        super().__init__()
        self.layers = nn.ModuleList([_CoreLayer(), _CoreLayer()])
        self._round_core = len(self.layers)
        if carrier == "injection":
            self.adapter = nn.Linear(8, 4, bias=False)
            self.norm = nn.LayerNorm(4)
        if carrier == "gated_residual":
            gate_config = SimpleNamespace(
                hidden_size=4 // streams,
                params_dtype=torch.float32,
                layernorm_epsilon=1e-5,
                init_method=nn.init.normal_,
            )
            self.loop_gates = nn.ModuleList(
                LoopResidualGate(gate_config, streams, "cpu") for _ in range(depth)
            )
        self.config = SimpleNamespace(
            loop_iterations=depth,
            loop_bptt_iterations=0,
            loop_carrier=carrier,
            loop_recompute=mode != "eager",
            loop_recompute_block_stride=1 if mode == "block" else None,
            loop_reuse_sparse_kv=True,
            loop_reuse_indexer=False,
        )

    def _run_range(self, x, start, end, ctx, label):
        for layer in self.layers[start:end]:
            x = layer(x)
        return x

    def _depth(self, device):
        return self.config.loop_iterations

    def _round_bounds(self, step):
        return 0, self._round_core

    def _diagnostics_active(self):
        return False

    def _adapt(self, state, injection, step):
        return self.adapter(torch.cat((state, injection), dim=-1))

    def _loop_norm(self, step):
        return self.norm

    def _record_range(self, *args, **kwargs):
        pass

    def round_extra_forward_flops(self, tokens):
        return 0

    def explicit_recurrence(self, embedding):
        first_pass = []
        injection = self.config.loop_carrier == "injection"
        x = torch.zeros_like(embedding) if injection else embedding
        for step in range(self.config.loop_iterations):
            previous = x
            if injection:
                x = self.adapter(torch.cat((x, embedding), dim=-1))
            for i, layer in enumerate(self.layers):
                if step == 0:
                    first_pass.append(layer.sparse(x))
                x = (first_pass[i] + layer.local(x)).tanh()
            if injection:
                x = self.norm(x)
            if self.config.loop_carrier == "gated_residual":
                gate = self.loop_gates[step]
                read = previous.unflatten(-1, (gate.streams, -1)).mean(-2)
                read = read.float()
                read = read * (read.square().mean(-1, keepdim=True) + gate.norm.eps).rsqrt()
                beta = (read.to(x.dtype) * gate.norm.weight * gate.proj.weight[0]).sum(-1).sigmoid()
                x = previous + beta.unsqueeze(-1) * (x - previous)
        return x


@pytest.mark.parametrize(
    "carrier,streams",
    [("residual", 1), ("injection", 1), ("gated_residual", 1), ("gated_residual", 2)],
)
@pytest.mark.parametrize("mode", ["eager", "round", "block"])
@pytest.mark.parametrize("depth", [2, 4])
def test_shared_source_graph_survives_checkpoint_replay_and_interleaved_microbatches(
    mode, depth, carrier, streams
):
    torch.manual_seed(71)
    actual = _Stack(depth, mode, carrier, streams).double()
    reference = copy.deepcopy(actual)
    inputs = [torch.randn(3, 2, 4, dtype=torch.double).requires_grad_() for _ in range(2)]
    ref_inputs = [x.detach().clone().requires_grad_() for x in inputs]

    # Both forwards complete before either backward. A module-local cache leaks across them.
    outputs = [actual._huginn(x, None, f"microbatch={i}") for i, x in enumerate(inputs)]
    ref_outputs = [reference.explicit_recurrence(x) for x in ref_inputs]
    for i in (1, 0):
        if carrier == "gated_residual":
            torch.testing.assert_close(outputs[i], ref_outputs[i])
        else:
            torch.testing.assert_close(outputs[i], ref_outputs[i], rtol=0, atol=0)
        upstream = torch.randn_like(outputs[i])
        outputs[i].backward(upstream)
        ref_outputs[i].backward(upstream)
        torch.testing.assert_close(inputs[i].grad, ref_inputs[i].grad)
        for param, ref_param in zip(actual.parameters(), reference.parameters()):
            assert param.grad is not None
            torch.testing.assert_close(param.grad, ref_param.grad)
        assert current_cross_loop_pass() is None


def _csa(ratio, dense):
    config = SimpleNamespace(
        csa_window_size=4,
        v_head_dim=4,
        qk_pos_emb_head_dim=2,
        num_attention_heads=1,
        attention_backend="unfused",
        csa_dense_mode=dense,
        dsa_cp_balance_indexer=False,
    )
    return CompressedSparseAttention(
        config=config,
        submodules=CompressedSparseAttentionSubmodules(compressor=nn.Identity, indexer=nn.Identity),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        compress_ratio=ratio,
        pg_collection=SimpleNamespace(tp=SimpleNamespace(size=lambda: 1)),
    )


@pytest.mark.parametrize(
    "ratio,dense,reuse_kv,reuse_index,expected_error",
    [
        (0, False, True, False, "requires a compressor"),
        (128, False, False, True, "requires a learned indexer"),
        (128, False, True, False, None),
        (128, False, True, True, None),
        (4, False, True, False, None),
        (4, False, True, True, None),
        (4, False, False, True, None),
        (4, True, True, False, None),
        (4, True, True, True, None),
        (4, True, False, True, "requires a learned indexer"),
    ],
)
def test_targets_follow_built_core_compressor_and_learned_indexer(
    ratio, dense, reuse_kv, reuse_index, expected_error
):
    # Both edges have indexers. Only the middle layer, nested as in an mHC wrapper, counts.
    stack = SimpleNamespace(
        config=SimpleNamespace(loop_reuse_sparse_kv=reuse_kv, loop_reuse_indexer=reuse_index),
        loop_roles=SimpleNamespace(core_start=1, core_end=2),
        layers=nn.ModuleList([_csa(4, False), nn.Sequential(_csa(ratio, dense)), _csa(4, False)]),
    )
    if expected_error is None:
        targets = LoopHybridStack._validate_cross_loop_attention_targets(stack)
        assert targets == {1: (ratio, ratio == 4 and not dense)}
    else:
        with pytest.raises(ValueError, match=expected_error):
            LoopHybridStack._validate_cross_loop_attention_targets(stack)


@pytest.mark.parametrize("reuse_kv,reuse_index", [(True, False), (False, True)])
def test_row_depths_refused_for_cross_loop_attention(reuse_kv, reuse_index):
    stack = SimpleNamespace(
        config=SimpleNamespace(
            loop_untie_core=False, loop_reuse_sparse_kv=reuse_kv, loop_reuse_indexer=reuse_index
        )
    )
    LoopHybridStack.set_loop_depths(stack, None)
    with pytest.raises(ValueError, match="uniform fixed depth"):
        LoopHybridStack.set_loop_depths(stack, torch.tensor([2, 4]))
