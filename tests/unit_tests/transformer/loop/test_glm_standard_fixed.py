# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fixed-depth standard PP1 uses local autograd through both GLM visits."""

from collections import defaultdict
from functools import partial
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_no_pipelining,
    get_forward_backward_func,
)
from megatron.core.transformer.forward_replay import in_forward_replay
from megatron.core.transformer.loop.guards import (
    fixed_glm_standard,
    validate_hybrid_loop_config,
    validate_hybrid_loop_inputs,
)
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from tests.unit_tests.transformer.loop.test_cross_loop_kv_recompute import _Stack
from tests.unit_tests.transformer.loop.test_glm_dsa_loop_model_gpu import _config


def test_two_document_standard_dcp_full_graph(monkeypatch):
    # Only the configuration/shape contract is exercised on CPU.
    from megatron.core.transformer import transformer_config

    monkeypatch.setattr(transformer_config, "is_te_min_version", lambda *args, **kwargs: True)
    cfg = _config(
        True,
        True,
        cp=2,
        dynamic_context_parallel=True,
        sequence_packing_scheduler="default_dynamic_cp",
        max_seqlen_per_dp_cp_rank=16384,
        pad_packed_seq_alignment="max",
        thd_max_packed_sequences=2,
        cuda_graph_impl="full_iteration",
        cuda_graph_static_dynamic_cp=True,
        cuda_graph_warmup_steps=3,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=8,
        loop_ring_recompute_group_sizes=[8, 4, 4, 4],
        loop_reuse_indexer=True,
        loop_recompute_ring_gates=True,
        loop_ring_trim_cache_before_first_grad_sync=True,
    )
    validate_hybrid_loop_config(cfg)
    assert fixed_glm_standard(cfg)
    assert get_forward_backward_func(pp_size=1, vp_size=None) is forward_backward_no_pipelining
    cu = torch.tensor([0, 24000, 32768], dtype=torch.int32)
    validate_hybrid_loop_inputs(
        packed_seq_params=PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu),
        inference_context=None,
        allow_packed_documents=fixed_glm_standard(cfg),
    )
    with pytest.raises(ValueError, match="eager warmup"):
        replace(cfg, cuda_graph_warmup_steps=0)


@pytest.mark.parametrize("recompute_gate", [False, True])
def test_shared_source_and_gate_vjp_with_distinct_visit_groups(recompute_gate):
    torch.manual_seed(137)
    model = _Stack(2, "eager", "gated_residual").double()
    reference = deepcopy(model)
    config = model.config
    config.loop_architecture = "huginn"
    config.experimental_attention_variant = "dsa"
    config.recompute_granularity = "full"
    config.recompute_num_layers = 8
    config.loop_ring_recompute_group_sizes = [8, 1, 2, 4]
    config.loop_recompute_ring_gates = recompute_gate
    trace = defaultdict(list)
    run_range = model._run_range

    def traced(x, start, end, ctx, label):
        trace[label, in_forward_replay()].append((start, end))
        return run_range(x, start, end, ctx, label)

    model._run_range = traced
    inputs = [torch.randn(3, 2, 4, dtype=torch.float64).requires_grad_() for _ in range(2)]
    expected_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    outputs = [
        model._huginn(x, SimpleNamespace(padding_mask=None), str(i)) for i, x in enumerate(inputs)
    ]
    expected = [reference.explicit_recurrence(x) for x in expected_inputs]
    for i in (1, 0):
        torch.testing.assert_close(outputs[i], expected[i])
        dy = torch.randn_like(outputs[i])
        expected[i].backward(dy)
        outputs[i].backward(dy)
        torch.testing.assert_close(inputs[i].grad, expected_inputs[i].grad)
        for name, tensor in model.named_parameters():
            grad = reference.get_parameter(name).grad
            assert grad.abs().max() > 0
            torch.testing.assert_close(tensor.grad, grad, msg=name)
        for visit, spans in enumerate(([(0, 1), (1, 2)], [(0, 2)]), start=1):
            assert trace[f"{i}/loop={visit}", False] == spans
            assert trace[f"{i}/loop={visit}", True] == spans[::-1]


def test_prelude_and_coda_apply_their_visit_width():
    config = SimpleNamespace(
        loop_architecture="huginn",
        loop_schedule="standard",
        loop_iterations=2,
        loop_carrier="gated_residual",
        experimental_attention_variant="dsa",
        recompute_granularity="full",
        recompute_num_layers=8,
        loop_recompute=False,
        loop_recompute_edges=False,
        loop_recompute_block_stride=None,
        loop_ring_recompute_group_sizes=[2, 1, 2, 3],
    )
    events = defaultdict(list)
    weights = torch.linspace(0.6, 1.3, 12, dtype=torch.float64).requires_grad_()
    ref_weights = weights.detach().clone().requires_grad_()

    def run(x, start, end, params):
        for i in range(start, end):
            x = torch.sin(x * params[i] + 0.11)
        return x

    def trace(x, start, end, ctx, label):
        events[label, in_forward_replay()].append((start, end))
        return run(x, start, end, weights)

    stack = SimpleNamespace(config=config, _run_range=trace)
    stack._checkpointed_blocks = partial(LoopHybridStack._checkpointed_blocks, stack)
    x = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
    pre = LoopHybridStack._run_edge(stack, x, 0, 5, None, "prelude")
    out = LoopHybridStack._run_edge(stack, pre, 7, 12, None, "coda")
    ref_x = x.detach().clone().requires_grad_()
    expected = run(run(ref_x, 0, 5, ref_weights), 7, 12, ref_weights)
    out.backward()
    expected.backward()
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(x.grad, ref_x.grad)
    torch.testing.assert_close(weights.grad, ref_weights.grad)
    for role, groups in (("prelude", [(0, 2), (2, 4), (4, 5)]), ("coda", [(7, 10), (10, 12)])):
        assert events[role, False] == groups
        assert events[role, True] == groups[::-1]
