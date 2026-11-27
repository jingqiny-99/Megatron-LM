# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Blackwell LinearCE reconstructs each vocabulary projection once during backward."""

from types import MethodType, SimpleNamespace

import pytest

from megatron.core.transformer.loop import flops
from megatron.core.transformer.loop.glm_ring_stack import GLMRingStack
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.loop.rsbf_executor import LoopRSBFExecutor
from tests.unit_tests.transformer.loop.test_rsbf_coda_retention import Edge

HIDDEN, VOCAB, SEQUENCE = 6144, 154880, 24
PROJECTION = 2 * HIDDEN * VOCAB * SEQUENCE


def _measure(edge):
    """Read the PP1 head, a checkpointed PP coda and the RSBF coda training counters."""
    edge._work = [0, 0, 0]
    LoopHybridStack._record_head(edge, SEQUENCE, SEQUENCE**2, SEQUENCE)
    ring = GLMRingStack.ring_microbatch_work(edge, SEQUENCE, 1, ["C"])
    program = SimpleNamespace(
        stack=edge,
        roles=SimpleNamespace(core_start=2, core_end=4),
        uniform_core_plan=SimpleNamespace(rebuild=False),
        coda_recompute_attentions=lambda: (),
    )
    executor = SimpleNamespace(config=edge.config, program=program, _flops={})
    cells = LoopRSBFExecutor._cell_flops(executor, SEQUENCE)
    return [
        tuple(edge._work),
        ring,
        tuple(cells[k] for k in ("coda", "coda_backward", "coda_recompute")),
    ]


@pytest.mark.parametrize(
    "depths,loss,ce,kd,checkpoint_tv,linear_heads,other_replays",
    [
        (0, "cross_entropy", 1.0, 0.0, False, 1, 0),
        (7, "cross_entropy", 1.0, 0.0, False, 8, 0),
        (7, "cross_entropy", 0.3, 0.7, False, 1, 0),
        (7, "cross_entropy", 0.0, 1.0, False, 1, 0),
        (7, "cross_entropy", 0.0, 0.0, False, 1, 0),
        (7, "e2e_tv", 1.0, 0.0, False, 1, 0),
        (7, "e2e_tv", 1.0, 0.0, True, 1, 7),
    ],
    ids=["lm-only", "ce", "ce-and-kd", "kd-only", "no-draft-loss", "tv", "tv-checkpoint"],
)
def test_only_executed_linear_ce_projections_add_replay_to_all_schedules(
    depths, loss, ce, kd, checkpoint_tv, linear_heads, other_replays
):
    edge = Edge()
    config = edge.config
    config.hidden_size = HIDDEN
    config.loop_ring_separate_edges = True
    config.loop_iterations = 2
    config.loop_rsbf_coda_retain_sublayers = 0
    config.mtp_loss_type = loss
    config.mtp_ce_loss_weight = ce
    config.mtp_kl_loss_weight = kd
    config.mtp_detach_heads = loss == "e2e_tv"
    config.mtp_recompute_tv_heads = checkpoint_tv
    edge._loop_head = (VOCAB, "", depths)
    edge.head_work_terms = MethodType(LoopHybridStack.head_work_terms, edge)
    edge._add_work = MethodType(LoopHybridStack._add_work, edge)
    edge._ring_lap_work = MethodType(GLMRingStack._ring_lap_work, edge)
    edge._lap_uses_checkpoint = MethodType(GLMRingStack._lap_uses_checkpoint, edge)
    edge.loop_ring = SimpleNamespace(role="coda", round_end=True)

    config.cross_entropy_loss_fusion = True
    config.cross_entropy_fusion_impl = "native"
    native = _measure(edge)
    assert native[0][2] == other_replays * PROJECTION
    config.cross_entropy_fusion_impl = "linear"
    config.cross_entropy_loss_fusion = False
    assert _measure(edge) == native
    config.cross_entropy_loss_fusion = True
    linear = _measure(edge)
    for before, after in zip(native, linear, strict=True):
        assert after[:2] == before[:2]
        assert after[2] - before[2] == linear_heads * PROJECTION
    if loss == "e2e_tv":
        terms = list(flops.head_schedule_work(edge, SEQUENCE, SEQUENCE**2, SEQUENCE))
        assert terms[1] == (PROJECTION, 0, False)  # detached TV teacher


def test_unregistered_head_does_not_infer_a_projection_from_the_first_work_term():
    stack = Edge()
    stack.config.cross_entropy_loss_fusion = True
    stack.config.cross_entropy_fusion_impl = "linear"
    assert list(flops.head_schedule_work(stack, 24, 24**2, 24)) == [(20, 1.5, False)]
