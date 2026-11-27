# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.forward_replay import in_forward_replay
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.loop.rsbf_executor import LoopRSBFExecutor


class Edge(torch.nn.Module):
    """Use the production checkpoint implementation and count each real/replay sublayer."""

    _run_edge = LoopHybridStack._run_edge
    _checkpointed_blocks = LoopHybridStack._checkpointed_blocks

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            loop_recompute=False,
            loop_schedule="rsbf",
            loop_recompute_edges=True,
            loop_recompute_block_stride=3,
            loop_carrier="gated_residual",
            loop_reuse_sparse_kv=False,
            loop_reuse_indexer=False,
        )
        self.layers = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.2 + i / 10)) for i in range(8)]
        )
        self.calls = Counter()

    def _run_range(self, x, start, end, ctx, label):
        for i in range(start, end):
            self.calls[in_forward_replay(), i] += 1
            x = (self.layers[i] * x).tanh()
        return x

    def range_forward_flops(self, start, end, tokens, *args, extra=0, **kwargs):
        return tokens * sum(1 + i for i in range(start, end)) + extra

    def round_extra_forward_flops(self, tokens):
        return 0

    def head_work_terms(self, *args):
        return [(20, 1.5)]


@pytest.mark.parametrize("retain", [0, 1, 2, 4])
def test_coda_suffix_replays_and_work_match_the_actual_layers(retain):
    start, end, tokens = 4, 8, 16
    edge = Edge()
    expected = Edge()
    inp = torch.tensor([0.2, 1.3], requires_grad=True)
    ref = inp.detach().clone().requires_grad_()
    out = edge._run_edge(inp, start, end, None, "test/coda", retain_sublayers=retain)
    eager = expected._run_range(ref, start, end, None, "test/coda")
    out.sum().backward()
    eager.sum().backward()
    assert torch.equal(out, eager) and torch.equal(inp.grad, ref.grad)
    for i in range(start, end):
        assert torch.equal(edge.layers[i].grad, expected.layers[i].grad)
        assert edge.calls[False, i] == 1
        assert edge.calls[True, i] == (i < end - retain)

    edge.config.loop_rsbf_coda_retain_sublayers = retain
    program = SimpleNamespace(
        stack=edge,
        roles=SimpleNamespace(core_start=2, core_end=start),
        uniform_core_plan=SimpleNamespace(rebuild=False),
        coda_recompute_attentions=lambda: (),
    )
    executor = SimpleNamespace(config=edge.config, program=program, _flops={})
    work = LoopRSBFExecutor._cell_flops(executor, tokens)
    coda = tokens * sum(1 + i for i in range(start, end))
    assert work["coda"] == coda + 20
    assert work["coda_backward"] == 2 * coda + 30
    assert work["coda_recompute"] == tokens * sum(
        (1 + i) * edge.calls[True, i] for i in range(start, end)
    )
