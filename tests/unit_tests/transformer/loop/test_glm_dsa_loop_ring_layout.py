# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks for GLM physical ownership and its four virtual pipeline passes."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.pipeline_parallel.schedules import get_schedule_table
from megatron.core.transformer.loop.guards import (
    resolve_hybrid_loop_ring_roles,
    resolve_loop_ring_recompute_plan,
    validate_ring_layout,
)
from megatron.core.transformer.loop.ring_pp import (
    LoopRingRoundView,
    _chunk_codes,
    _ring_view_layout,
    ring_physical_chunks,
    ring_sample_counts,
)
from tests.unit_tests.transformer.loop.test_glm_dsa_loop_model_gpu import _config

# Offsets below are in physical decoder blocks; guards take D/FFN sublayer offsets.
PRE, CORE, POST = (2, 4, 4, 4, 4, 4, 4), (4,) * 7, (4, 4, 4, 4, 4, 4, 0)


def _config_and_pattern(sections):
    cfg = _config(
        True,
        False,
        num_layers=156,
        prelude=52,
        coda=48,
        pipeline_model_parallel_size=len(sections[0]),
        virtual_pipeline_model_parallel_size=3,
        loop_ring_separate_edges=True,
        loop_schedule="ring",
        pipeline_dtype=torch.float32,
        overlap_p2p_comm=True,
        batch_p2p_comm=False,
        is_hybrid_model=True,
    )
    return cfg, "|".join("D-" * n for section in sections for n in section)


def test_78_layer_global_and_local_ownership_match():
    cfg, main = _config_and_pattern((PRE, CORE, POST))
    chunks = validate_ring_layout(cfg, main, None)
    starts = [0, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78]
    assert [c.offset for c in chunks] == [2 * start for start in starts]
    assert [c.role for c in chunks] == ["prelude"] * 7 + ["core"] * 7 + ["coda"] * 7
    for i, (chunk, symbols) in enumerate(zip(chunks, main.split("|"), strict=True)):
        roles, local = resolve_hybrid_loop_ring_roles(
            cfg, symbols, 2 * starts[i], None, pp_rank=i % 7
        )
        assert local == chunk
        assert (roles.prelude, roles.core, roles.coda) == (52, 56, 48)
        assert chunk.round_start == (i % 7 == 0)
        assert chunk.round_end == (i % 7 == 6)
        assert (chunk.local.prelude, chunk.local.core, chunk.local.coda) == tuple(
            len(symbols) if j == i // 7 else 0 for j in range(3)
        )


@pytest.mark.parametrize(
    "section,replacement,reason",
    [
        (0, (3, 3, 4, 4, 4, 4, 4), "prelude chunk starts inside a DSA top-k sharing group"),
        (1, (3, 5, 4, 4, 4, 4, 4), "core chunk starts inside a DSA top-k sharing group"),
        (2, (3, 5, 4, 4, 4, 4, 0), "coda chunk starts inside a DSA top-k sharing group"),
        (0, (0, 6, 4, 4, 4, 4, 4), "Only coda stages may pass through"),
    ],
)
def test_layout_rejects_illegal_boundary(section, replacement, reason):
    sections = [PRE, CORE, POST]
    sections[section] = replacement
    cfg, main = _config_and_pattern(sections)
    with pytest.raises(ValueError, match=reason):
        validate_ring_layout(cfg, main, None)


def test_pp8_places_the_final_core_ffn_on_the_gate_rank():
    sections = ((1, 1, 4, 4, 4, 4, 4, 4), (4,) * 7 + (0,), (4, 4, 0, 0, 4, 4, 0, 8))
    config, main = _config_and_pattern(sections)
    chunks = main.split("|")
    chunks[15], chunks[14] = chunks[14][-1], chunks[14][:-1]
    result = validate_ring_layout(config, "|".join(chunks), None)
    assert [item.role for item in result] == ["prelude"] * 8 + ["core"] * 8 + ["coda"] * 8
    assert [item.local.core for item in result[8:16]] == [8] * 6 + [7, 1]
    assert result[14].offset == 100 and result[15].offset == 107
    assert result[15].round_end and result[23].local.coda == 16
    for i, (symbols, item) in enumerate(zip(chunks, result, strict=True)):
        _, actual = resolve_hybrid_loop_ring_roles(
            config, symbols, item.offset, None, pp_rank=i % 8
        )
        assert actual == item
    # A cut before the FFN inside a group would strand following DSA consumers.
    chunks[14], chunks[15] = chunks[14][:-2], chunks[14][-2:] + chunks[15]
    with pytest.raises(ValueError, match="block boundar"):
        validate_ring_layout(config, "|".join(chunks), None)


@pytest.mark.parametrize("stage", [0, 3, 6])
def test_logical_schedule_reuses_one_core_and_restricts_endpoints(stage):
    cfg, _ = _config_and_pattern((PRE, CORE, POST))
    physical = []
    for i in range(3):
        chunk = torch.nn.Module()
        chunk.decoder = SimpleNamespace(
            config=cfg, loop_ring=True, watch_ring_mtp=lambda module: None
        )
        chunk.pre_process = stage == 0 and i == 0
        chunk.post_process = chunk.mtp_process = stage == 6 and i == 2
        chunk.mtp = torch.nn.Identity() if chunk.mtp_process else None
        physical.append(chunk)

    layout = _ring_view_layout(cfg)
    assert layout == ((0, 0), (1, 0), (1, 1), (2, 1))
    # The physical VPP3 config describes four recompute choices on each pipeline rank.
    assert resolve_loop_ring_recompute_plan(cfg) == ("nnnn",) * 7
    assert [_chunk_codes(layout, "nCnn", i) for i in range(3)] == [("n",), ("C", "n"), ("n",)]
    views = [
        LoopRingRoundView(
            physical[owner],
            cfg,
            round_index=r,
            chunk_index=owner,
            chunks=3,
            checkpoint=False,
            lap=lap,
            gate_banks={},
        )
        for lap, (owner, r) in enumerate(layout)
    ]
    assert ring_physical_chunks(views) == physical
    assert [v.view_index for v in views] == [0, 1, 2, 3]
    assert all(v.views == 4 for v in views)
    table = get_schedule_table(
        num_microbatches=7, num_model_chunks=4, microbatch_group_size_per_vp_stage=7
    )
    for mb in range(7):
        visits = [views[v] for batch, v in table if batch == mb]
        assert [v.module for v in visits] == [physical[0], physical[1], physical[1], physical[2]]
        calls = [view.roles_for(mb) for view in visits]
        assert [c.lap for c in calls] == [0, 1, 2, 3]
        assert all(c.microbatch == mb for c in calls)
        assert [c.pre_process for c in calls] == [stage == 0, False, False, False]
        assert [c.post_process for c in calls] == [False, False, False, stage == 6]
        assert [c.mtp_process for c in calls] == [False, False, False, stage == 6]

    # In a separate-edge ring only the shared physical core is expected to run twice.
    chunk_counts = [
        {mb: counts for mb in range(7)}
        for counts in [
            (0, int(stage == 0), 0, 0),
            (2, 0, 0, 0),
            (0, 0, int(stage == 6), int(stage == 6)),
        ]
    ]
    expected = (2, 2, int(stage == 0), int(stage == 6), int(stage == 6))
    assert ring_sample_counts(chunk_counts, core_indices=(1,)) == {mb: expected for mb in range(7)}


@pytest.mark.parametrize(
    "depths,logical",
    [([1] * 56, (0, 1, 4)), ([2] * 56, (0, 1, 2, 4)), ([1, 2, 3, 2] * 14, (0, 1, 2, 3, 4))],
)
def test_empty_core_laps_remove_p2p_slots_keep_coda_and_sample_order(depths, logical):
    from collections import Counter
    from megatron.core.pipeline_parallel.schedules import convert_schedule_table_to_order
    from megatron.core.transformer.loop.ring_pp import _scheduled_laps

    cfg = SimpleNamespace(loop_iterations=3, loop_ring_separate_edges=True)
    chosen = _scheduled_laps(_ring_view_layout(cfg), depths)
    assert tuple(lap for lap, _, _ in chosen) == logical
    assert chosen[0] == (0, 0, 0) and chosen[-1] == (4, 2, 2)
    assert {chunk for _, chunk, _ in chosen} == {0, 1, 2}
    for rank in (0, 3, 6):
        table = get_schedule_table(56, len(chosen), 8)
        order = convert_schedule_table_to_order(
            2 * (6 - rank) + (len(chosen) - 1) * 8, len(chosen), table
        )
        since = Counter()
        fwd = [[] for _ in depths]
        bwd = [[] for _ in depths]
        for slot in order:
            micro = since[slot]
            since[slot] += 1
            lap, chunk, r = chosen[abs(slot) - 1]
            (fwd if slot > 0 else bwd)[micro].append(lap)
            if chunk == 2:
                assert r == cfg.loop_iterations - 1
                assert "CnnCn"[lap] == "n"
        assert fwd == [list(logical)] * 56 and bwd == [list(reversed(logical))] * 56
