# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Detached bank gradients across separately scheduled PP backwards and checkpoint replay."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.transformer.loop.cross_loop_kv import current_cross_loop_pass
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.loop.ring_pp import LoopRingCall


def _stage():
    return SimpleNamespace(
        _cross_loop_attention_layers={0: (4, True)}, config=SimpleNamespace(loop_iterations=4)
    )


def _pass(stage, banks, mb, step):
    call = LoopRingCall(
        pre_process=False,
        post_process=False,
        mtp_process=False,
        round=step,
        microbatch=mb,
        sparse_banks=banks,
    )
    return LoopHybridStack._ring_sparse_pass(stage, call)


@pytest.mark.parametrize("recompute", [False, True])
def test_interleaved_ring_microbatches_stop_at_bank_then_merge_into_source(recompute):
    torch.manual_seed(11)
    layer = torch.nn.Linear(4, 4, bias=False).double()
    reference = torch.nn.Linear(4, 4, bias=False).double()
    reference.load_state_dict(layer.state_dict())
    inputs = torch.randn(2, 4, 3, 4, dtype=torch.double, requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    ref_losses = []
    for mb_inputs in ref_inputs:
        bank = reference(mb_inputs[0]).sin()
        indices = bank.topk(2, dim=-1).indices
        ref_losses.extend((x * bank).gather(-1, indices).tanh().sum() for x in mb_inputs)
    sum(ref_losses).backward()

    stage, banks, losses = _stage(), {}, {}
    for step in range(4):
        for mb in range(2):
            scope = _pass(stage, banks, mb, step)

            def body(x, scope=scope):
                with scope.activate():
                    active = current_cross_loop_pass()
                    if active.is_source:
                        bank = layer(x).sin()
                        indices = bank.topk(2, dim=-1).indices
                        active.publish(layer, "compressed_kv", bank)
                        active.publish(layer, "indices", indices)
                    else:
                        bank = active.read(layer, "compressed_kv")
                        indices = active.read(layer, "indices")
                    return (x * bank).gather(-1, indices).tanh().sum()

            x = inputs[mb, step]
            losses[mb, step] = checkpoint(body, x, use_reentrant=False) if recompute else body(x)
            if scope.is_source:
                scope.finish_source()
    assert banks == {}

    # Independent backwards per view in reverse round order, after both original forwards.
    for step in reversed(range(4)):
        for mb in reversed(range(2)):
            losses[mb, step].backward()
        if step > 0:
            assert layer.weight.grad is None
    torch.testing.assert_close(layer.weight.grad, reference.weight.grad)
    torch.testing.assert_close(inputs.grad, ref_inputs.grad)


def test_ring_rejects_missing_or_duplicate_microbatch_source():
    stage, banks = _stage(), {}
    with pytest.raises(RuntimeError, match="no first-round sparse bank"):
        _pass(stage, banks, 9, 2)
    _pass(stage, banks, 9, 0)
    with pytest.raises(RuntimeError, match="already holds microbatch 9"):
        _pass(stage, banks, 9, 0)
