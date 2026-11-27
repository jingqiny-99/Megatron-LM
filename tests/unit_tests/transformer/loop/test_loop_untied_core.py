# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the hybrid untied core (``loop_untie_core`` in role mode), the carrier ablation's
``plain-inject`` variant: the physical core holds R equal segments; round r runs segment r with
its own adapter and loop norm (megatron/core/transformer/loop/hybrid_stack.py).

The oracle (T-P1): a ``plain-inject`` stack whose segments, adapters and norms are copies of a
``loop-entry`` stack's core, adapter and norm computes the ``loop-entry`` output and input
gradient bitwise (same ops in the same order), and each tied parameter's gradient equals the
sum of its R copies' gradients up to fp32 summation order. Local layers, CPU (gloo) and CUDA.
"""

import pytest
import torch

from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack, LoopStreamRMSNorm
from megatron.core.transformer.loop_transformer_block import LoopRMSNorm
from tests.unit_tests.transformer.loop import hybrid_loop_utils
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    BATCH,
    HIDDEN,
    SEQ,
    build_local_loop_stack,
    device,
    model_parallel_once,
    rope,
)

mhc_on_this_device = hybrid_loop_utils.mhc_on_this_device

ROUNDS = 2
PRELUDE = "*-"
CORE = "*-*-"
CODA = "*-"
TIED = PRELUDE + CORE + CODA
UNTIED = PRELUDE + CORE * ROUNDS + CODA
MHC = dict(enable_mhc_connections=True, mhc_num_residual_streams=4, mhc_sinkhorn_iterations=5)


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


def _tied(**overrides):
    return build_local_loop_stack(TIED, 2, 2, loop_iterations=ROUNDS, **overrides)


def _untied(**overrides):
    return build_local_loop_stack(
        UNTIED, 2, 2, loop_iterations=ROUNDS, loop_untie_core=True, seed=4321, **overrides
    )


def _copy_tied_into_untied(tied, untied):
    """Load the tied stack's weights into every segment of the untied stack."""
    core = len(CORE)
    prelude = len(PRELUDE)
    mapped = {}
    for key, value in untied.state_dict().items():
        if key.startswith("layers."):
            _, index, rest = key.split(".", 2)
            index = int(index)
            if index < prelude:
                source = index
            elif index < prelude + ROUNDS * core:
                source = prelude + (index - prelude) % core
            else:
                source = index - (ROUNDS - 1) * core
            mapped[key] = tied.state_dict()[f"layers.{source}.{rest}"]
        elif key.startswith(("loop_adapters.", "loop_norms.")):
            group, _, rest = key.split(".", 2)
            mapped[key] = tied.state_dict()[f"{group[:-1]}.{rest}"]
        else:
            mapped[key] = tied.state_dict()[key]
    untied.load_state_dict(mapped, strict=True)


def _inputs(seed=7):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(SEQ, BATCH, HIDDEN, generator=generator).to(device())
    upstream = torch.randn(SEQ, BATCH, HIDDEN, generator=generator).to(device())
    return x, upstream


def _forward_backward(stack, x, upstream):
    x = x.detach().clone().requires_grad_(True)
    stack.zero_grad(set_to_none=True)
    y = stack(x, attention_mask=None, rotary_pos_emb=rope())
    (y * upstream).sum().backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in stack.named_parameters()
        if param.grad is not None
    }
    return y.detach().clone(), x.grad.detach().clone(), grads


def test_untied_stack_owns_one_adapter_and_norm_per_round():
    stack = _untied()
    assert isinstance(stack, LoopHybridStack)
    assert stack.loop_roles.core == ROUNDS * len(CORE)
    assert stack._round_core == len(CORE)
    assert [stack._round_bounds(step) for step in range(ROUNDS)] == [(2, 6), (6, 10)]
    assert len(stack.loop_adapters) == ROUNDS and len(stack.loop_norms) == ROUNDS
    assert all(isinstance(norm, LoopRMSNorm) for norm in stack.loop_norms)
    assert all(a.weight.shape == (HIDDEN, 2 * HIDDEN) for a in stack.loop_adapters)
    # Distinct parameters, not one module registered R times.
    assert stack.loop_adapters[0].weight is not stack.loop_adapters[1].weight
    names = {name for name, _ in stack.named_parameters()}
    assert not {"loop_adapter.weight", "loop_norm.weight"} & names
    assert {f"loop_adapters.{r}.weight" for r in range(ROUNDS)} <= names
    # A tied stack keeps its keys.
    tied_names = {name for name, _ in _tied().named_parameters()}
    assert {"loop_adapter.weight", "loop_norm.weight"} <= tied_names
    assert not any(name.startswith(("loop_adapters.", "loop_norms.")) for name in tied_names)


def test_untied_mhc_stack_gets_a_stream_norm_per_round(mhc_on_this_device):
    stack = _untied(**MHC)
    assert all(isinstance(norm, LoopStreamRMSNorm) for norm in stack.loop_norms)
    x, upstream = _inputs()
    y, dx, grads = _forward_backward(stack, x, upstream)
    assert torch.isfinite(y).all() and torch.isfinite(dx).all()
    assert all(f"loop_norms.{r}.weight" in grads for r in range(ROUNDS))


@pytest.mark.parametrize("recompute", [False, True])
def test_copied_segments_reproduce_the_tied_loop(recompute):
    """T-P1: output and input gradient bitwise; tied gradients = summed segment gradients."""
    tied = _tied(loop_recompute=recompute)
    untied = _untied(loop_recompute=recompute)
    _copy_tied_into_untied(tied, untied)
    x, upstream = _inputs()
    y_ref, dx_ref, grads_ref = _forward_backward(tied, x, upstream)
    y, dx, grads = _forward_backward(untied, x, upstream)
    assert torch.equal(y, y_ref)
    assert torch.equal(dx, dx_ref)

    core = len(CORE)
    prelude = len(PRELUDE)
    for name, reference in grads_ref.items():
        if name.startswith("layers."):
            _, index, rest = name.split(".", 2)
            index = int(index)
            if prelude <= index < prelude + core:
                copies = [f"layers.{index + step * core}.{rest}" for step in range(ROUNDS)]
            elif index < prelude:
                copies = [name]
            else:
                copies = [f"layers.{index + (ROUNDS - 1) * core}.{rest}"]
        else:
            group, rest = name.split(".", 1)
            copies = [f"{group}s.{step}.{rest}" for step in range(ROUNDS)]
        summed = sum(grads[copy] for copy in copies)
        if len(copies) == 1:
            # Prelude and coda are used once in both stacks: same ops, same order.
            assert torch.equal(summed, reference), name
        else:
            error = (summed - reference).norm() / reference.norm().clamp_min(1e-30)
            assert error <= 1e-6, (name, error.item())
    # Each segment's copy gets its own round's gradient: the two copies differ.
    assert not torch.equal(
        grads["layers.2.self_attention.linear_qkv.weight"],
        grads["layers.6.self_attention.linear_qkv.weight"],
    )


def test_untied_recompute_replays_each_round_on_its_own_segment():
    """``loop_recompute`` with distinct untied segments (their own random init, not copies):
    output, input gradient and every parameter gradient equal the non-recompute run bitwise,
    so each round's replay binds its own segment bounds (a replay of the wrong segment would
    be invisible in T-P1, whose segments are identical copies)."""
    plain = _untied(loop_recompute=False)
    recomputed = _untied(loop_recompute=True)
    recomputed.load_state_dict(plain.state_dict())
    assert not torch.equal(
        plain.state_dict()["layers.2.self_attention.linear_qkv.weight"],
        plain.state_dict()["layers.6.self_attention.linear_qkv.weight"],
    )
    x, upstream = _inputs()
    y_ref, dx_ref, grads_ref = _forward_backward(plain, x, upstream)
    y, dx, grads = _forward_backward(recomputed, x, upstream)
    assert torch.equal(y, y_ref)
    assert torch.equal(dx, dx_ref)
    assert grads.keys() == grads_ref.keys()
    for name, reference in grads_ref.items():
        assert torch.equal(grads[name], reference), name


def test_untied_flops_equal_the_tied_loop():
    tied = _tied(loop_recompute=True)
    untied = _untied(loop_recompute=True)
    x, upstream = _inputs()
    _forward_backward(tied, x, upstream)
    _forward_backward(untied, x, upstream)
    for name in (
        "loop forward flops per token",
        "loop backward flops per token",
        "loop recompute flops per token",
        "loop depth",
        "loop active token layers",
        "loop backward token layers",
    ):
        assert torch.equal(untied.loop_metrics[name], tied.loop_metrics[name]), name
    assert untied.loop_metrics["loop active token layers"] == SEQ * BATCH * ROUNDS * len(CORE)


@pytest.mark.parametrize("training", [True, False])
def test_untied_rounds_call_their_own_segment_adapter_and_norm(training):
    """Call order: prelude, then per round r its adapter, segment r and norm, then the coda.

    The copied-weights oracle cannot tell the R adapters / norms / segments apart; this pins
    which module each round calls."""
    stack = _untied()
    calls = []
    modules = [(f"layer{index}", layer) for index, layer in enumerate(stack.layers)]
    modules += [(f"adapter{r}", m) for r, m in enumerate(stack.loop_adapters)]
    modules += [(f"norm{r}", m) for r, m in enumerate(stack.loop_norms)]
    for name, module in modules:
        module.register_forward_hook(lambda m, args, output, name=name: calls.append(name))
    stack.train(training)
    with torch.set_grad_enabled(training):
        stack(_inputs()[0], attention_mask=None, rotary_pos_emb=rope())
    expected = ["layer0", "layer1"]
    for step in range(ROUNDS):
        start, end = stack._round_bounds(step)
        expected += [f"adapter{step}"] + [f"layer{i}" for i in range(start, end)]
        expected += [f"norm{step}"]
    expected += [f"layer{i}" for i in range(len(UNTIED) - 2, len(UNTIED))]
    assert calls == expected


def test_untied_stack_refuses_row_depths():
    stack = _untied()
    with pytest.raises(ValueError, match=r"fixed-depth Huginn \(set_loop_depths\)"):
        stack.set_loop_depths(torch.tensor([1, 2], device=device()))
    stack.set_loop_depths(None)  # clearing stays allowed


def test_untied_stack_checks_the_segments():
    with pytest.raises(ValueError, match="loop_untie_core core not R equal segments"):
        build_local_loop_stack(
            PRELUDE + "*-*-*-" + CODA, 2, 2, loop_iterations=2, loop_untie_core=True
        )
