# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-visit full uniform checkpoints on the separate-edge GLM ring."""

from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.forward_replay import in_forward_replay
from megatron.core.transformer.loop.glm_ring_stack import GLMRingStack
from megatron.core.transformer.loop.ring_pp import LoopRingCall, loop_ring_call
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.transformer.loop.test_glm_dsa_loop_ring_layout import (
    CORE,
    POST,
    PRE,
    _config_and_pattern,
)


def test_group_size_validation():
    cfg, _ = _config_and_pattern((PRE, CORE, POST))
    assert cfg.loop_ring_recompute_group_sizes is None
    widths = [8, 8, 2, 4]
    with pytest.raises(ValueError, match="separate-edge GLM ring with full uniform"):
        replace(cfg, loop_ring_recompute_group_sizes=widths)
    full = replace(
        cfg,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=8,
        loop_ring_recompute_group_sizes=widths,
    )
    assert full.loop_ring_recompute_group_sizes == widths
    for wrong in (
        [],
        [8, 2, 4],
        [8, 8, 2, 4, 1],
        [0, 8, 2, 4],
        [-1, 8, 2, 4],
        [True, 8, 2, 4],
        [2.5, 8, 2, 4],
    ):
        with pytest.raises(ValueError, match="one positive integer per visit"):
            replace(full, loop_ring_recompute_group_sizes=wrong)
    with pytest.raises(ValueError, match="separate-edge GLM ring with full uniform"):
        TransformerConfig(
            num_layers=4,
            hidden_size=8,
            num_attention_heads=1,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=2,
            loop_ring_recompute_group_sizes=widths,
        )


def test_visit_boundaries_replay_tied_weights_and_work():
    # One middle physical PP rank. The R1/R2 calls reuse exactly the same core layers.
    lens = (10, 9, 7)
    roles = ("prelude", "core", "coda")
    visits = (0, 1, 1, 2)
    ctx = SimpleNamespace(attention_mask=None, packed_seq_params=None, inference_context=None)
    base = torch.linspace(-0.6, 0.7, 12, dtype=torch.float64).view(3, 1, 4)

    def run(sizes):
        cfg = SimpleNamespace(
            loop_ring_separate_edges=True,
            loop_iterations=2,
            loop_exit_criterion="fixed",
            loop_carrier="gated_residual",
            recompute_granularity="full",
            recompute_num_layers=8,
            loop_ring_recompute_group_sizes=sizes,
        )
        traces = {}
        weights = [torch.linspace(0.7, 1.1, n, dtype=base.dtype).requires_grad_() for n in lens]
        stacks = []
        for role, params in zip(roles, weights):
            stack = SimpleNamespace(
                call_roles=None,
                config=cfg,
                layers=params,
                _loop_depths=None,
                _ring_counts={},
                _ring_work=[0, 0, 0],
                _cross_loop_attention_layers={},
                loop_ring=SimpleNamespace(role=role, round_start=False, round_end=False),
                range_forward_flops=lambda first, last, tokens, *_args, **_kw: (last - first)
                * tokens,
            )

            def run_range(state, first, last, ctx, label, params=params):
                traces.setdefault((label, in_forward_replay()), []).append((first, last))
                for index in range(first, last):
                    state = (state * params[index] + 0.03).sin()
                return state

            stack._run_range = run_range
            for name in (
                "_ring_lap_work",
                "_lap_uses_checkpoint",
                "_ring_add_work",
                "_ring_sparse_pass",
                "_ring_local_layers",
                "_checkpointed_blocks",
            ):
                setattr(stack, name, partial(getattr(GLMRingStack, name), stack))
            stacks.append(stack)

        x = base.clone().requires_grad_()
        y = x
        # Full/uniform has its existing precedence over both plan codes.
        for lap, physical in enumerate(visits):
            stack = stacks[physical]
            call = LoopRingCall(
                False,
                False,
                False,
                round=max(0, lap - 1),
                lap=lap,
                microbatch=0,
                checkpoint=lap in (0, 2),
            )
            with loop_ring_call(stack, call):
                y = GLMRingStack._ring_layer_loop(stack, y, ctx)
        # Every call has left the physical stack before any shared core checkpoint replays.
        assert all(stack.call_roles is None for stack in stacks)
        y.square().sum().backward()
        for lap, physical in enumerate(visits):
            label = f"loop_hybrid/mb=0/{roles[physical]}/lap={lap}"
            width = 8 if sizes is None else sizes[lap]
            bounds = [(i, min(i + width, lens[physical])) for i in range(0, lens[physical], width)]
            assert traces[label, False] == bounds
            assert traces[label, True] == bounds[::-1]
        for physical, (stack, codes) in enumerate(zip(stacks, [("C",), ("n", "C"), ("n",)])):
            analytic = GLMRingStack.ring_microbatch_work(stack, len(base), 1, codes)
            f = len(base) * lens[physical] * visits.count(physical)
            assert analytic == tuple(stack._ring_work) == (f, 2 * f, f)
        return y.detach(), x.grad, *[w.grad for w in weights]

    expected = run(None)
    for sizes in ([8, 8, 8, 8], [8, 8, 2, 4]):
        for got, ref in zip(run(sizes), expected, strict=True):
            torch.testing.assert_close(got, ref, atol=0, rtol=0)
