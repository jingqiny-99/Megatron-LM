# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from itertools import product

import pytest

from megatron.core.transformer.loop.rsbf_plan import SubRoundType
from megatron.core.transformer.loop.rsbf_refill import RefillPlanner


def run_online(depths, window):
    home = [g % 2 for g in range(len(depths))]
    planner = RefillPlanner(home, 2, 3, window)
    visits = [[] for _ in home]
    while (tick := planner.next_tick()) is not None:
        exited = []
        for eu, item in enumerate(tick.items):
            if item is None:
                assert tick.type in (SubRoundType.CORE_F, SubRoundType.CORE_B)
                continue
            assert home[item.sample] == eu
            visits[item.sample].append((item.kind, item.round))
            if tick.type == SubRoundType.CORE_F and item.round == depths[item.sample]:
                exited.append(item.sample)
        if tick.type == SubRoundType.CORE_F:
            planner.observe_exits(exited)
    for g, d in enumerate(depths):
        assert planner.exit_round(g) == d
        assert visits[g] == (
            [("prelude_sweep", 0)] + [("core_sweep", r) for r in range(1, d + 1)]
            + [("coda", d)] + [("core", r) for r in range(d, 0, -1)] + [("prelude", 0)]
        )
    assert planner.peak_live <= window
    return planner


@pytest.mark.parametrize("window", [1, 2, 3])
def test_online_complete_uneven_homes(window):
    # All 729 orderings include unequal per-home work and both directions of asymmetry.
    for depths in product((1, 2, 3), repeat=6):
        run_online(depths, window)


def test_admits_fresh_input_and_starts_first_round_while_another_home_continues():
    planner = run_online([1, 3, 3, 1], window=2)
    ticks = [[(x.kind, x.sample, x.round) if x else None for x in t.items] for t in planner.ticks]
    assert ticks[:4] == [
        [("prelude_sweep", 0, 0), ("prelude_sweep", 1, 0)],
        [("core_sweep", 0, 1), ("core_sweep", 1, 1)],
        [("prelude_sweep", 2, 0), ("prelude_sweep", 3, 0)],
        [("core_sweep", 2, 1), ("core_sweep", 1, 2)],
    ]


@pytest.mark.parametrize("keep_backward", [False, True])
def test_mok_null_buffer_lifetime(keep_backward):
    import weakref
    from contextlib import nullcontext
    from types import SimpleNamespace
    import torch
    from megatron.core.transformer.loop.null_items import FORM_MOK, _null_forward_chain

    saved = []
    gradients = []

    class Capture(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, layer):
            # A native MoK forward holds dispatch, intermediate and shared tensors.
            buffer = x.new_zeros(16)
            saved.append(weakref.ref(buffer))
            ctx.save_for_backward(buffer)
            ctx.layer = layer
            return x * 2

        @staticmethod
        def backward(ctx, grad):
            assert ctx.saved_tensors[0].shape == (16,)
            gradients.append(ctx.layer)
            return grad * 2, None

    class Adapter:
        @staticmethod
        def null_forward(entry, hidden, signature):
            assert torch.is_grad_enabled() and hidden.requires_grad
            assert sum(ref() is not None for ref in saved) == (entry.layer_number if keep_backward else 0)
            return Capture.apply(hidden, entry.layer_number)

    entries = [SimpleNamespace(module=i, layer_number=i, form=FORM_MOK, wrap=None, adapter=Adapter(),
                               layer_config=SimpleNamespace(hidden_size=2)) for i in range(4)]
    sig = SimpleNamespace(hidden_dtype=torch.float32, hidden_requires_grad=True)
    result = _null_forward_chain(
        SimpleNamespace(signature={i: sig for i in range(4)}), entries, sequence=4,
        device=torch.device("cpu"), inner_context=lambda *args: nullcontext(), keep_backward=keep_backward,
    )
    if keep_backward:
        result.sum().backward()
        assert gradients == [3, 2, 1, 0]
    else:
        assert result.grad_fn is None
    assert all(ref() is None for ref in saved)
