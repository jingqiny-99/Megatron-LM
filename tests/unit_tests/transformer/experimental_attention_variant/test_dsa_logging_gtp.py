# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU coverage for the GTP-aware indexer-loss reduction (DSAIndexerLossLoggingHelper).

GTP remat ranks each run their own micro-batches, so the indexer loss is also averaged over
the GTP remat group, unless the data-parallel group passed in already spans it. The
collectives are recorded instead of run: every group is a sentinel with fixed ranks."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import (
    dsa_logging as dsa_logging_module,
)
from megatron.core.transformer.experimental_attention_variant.dsa_logging import (
    DSAIndexerLossLoggingHelper,
)


class _Group:
    def __init__(self, name, ranks):
        self.name = name
        self.ranks = tuple(ranks)

    def __repr__(self):
        return self.name


# Four ranks: replicate DP pairs (0, 2) / (1, 3), GTP remat pairs (0, 1) / (2, 3).
PP = _Group("pp", [0])
DP = _Group("dp", [0, 2])
DP_GTP = _Group("dp_gtp_remat", [0, 1, 2, 3])
GTP = _Group("gtp_remat", [0, 1])
GTP_OFF = _Group("gtp_remat_off", [0])


@pytest.fixture
def recorded(monkeypatch):
    calls = []
    monkeypatch.setattr(dsa_logging_module, "_get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(dsa_logging_module, "get_pg_size", lambda group: len(group.ranks))
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op=None, group=None: calls.append((group.name, op)),
    )
    DSAIndexerLossLoggingHelper.tracker = {
        "values": torch.ones(2),
        "agreed_size": 2,
        "agreed_size_pp_group": PP,
        "raw_values_enabled": False,
    }
    yield calls
    DSAIndexerLossLoggingHelper.tracker = {}


AVG = torch.distributed.ReduceOp.AVG


def test_gtp_group_is_averaged_beside_replicate_dp(recorded):
    DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(
        pp_group=PP, dp_group=DP, gtp_remat_group=GTP
    )
    assert recorded == [("pp", None), ("dp", AVG), ("gtp_remat", AVG)]


def test_gtp_group_is_skipped_when_dp_spans_it(recorded):
    # Training passes dp_gtp_remat (metric_data_parallel_group) under the GTP-remat grid.
    DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(
        pp_group=PP, dp_group=DP_GTP, gtp_remat_group=GTP
    )
    assert recorded == [("pp", None), ("dp_gtp_remat", AVG)]


def test_size_one_gtp_group_adds_no_collective(recorded):
    DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(
        pp_group=PP, dp_group=DP, gtp_remat_group=GTP_OFF
    )
    assert recorded == [("pp", None), ("dp", AVG)]


def test_gtp_group_is_read_from_pg_collection(recorded):
    pg_collection = SimpleNamespace(pp=PP, dp=DP, gtp_remat=GTP)
    DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(pg_collection=pg_collection)
    assert recorded == [("pp", None), ("dp", AVG), ("gtp_remat", AVG)]


def test_dynamic_cp_parent_is_extended_by_gtp(recorded):
    DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(
        pp_group=PP,
        dp_group=_Group("dp_dynamic", [0]),
        dynamic_cp_parent_group=_Group("dp_cp", [0, 2]),
        configured_cp_size=2,
        gtp_remat_group=GTP,
    )
    assert recorded == [("pp", None), ("dp_cp", AVG), ("gtp_remat", AVG)]


def test_track_indexer_metrics_passes_gtp_group(monkeypatch):
    seen = {}

    def reduce_loss(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "reduce_loss_in_tracker", reduce_loss)
    DSAIndexerLossLoggingHelper.tracker = {}
    try:
        DSAIndexerLossLoggingHelper.track_indexer_metrics(
            loss_scale=1.0,
            iteration=1,
            writer=None,
            pg_collection=SimpleNamespace(pp=PP, dp=DP, gtp_remat=GTP),
        )
        assert seen["gtp_remat_group"] is GTP
        seen.clear()
        # Without a GTP group the call keeps its pre-GTP shape.
        DSAIndexerLossLoggingHelper.track_indexer_metrics(
            loss_scale=1.0, iteration=1, writer=None, pg_collection=SimpleNamespace(pp=PP, dp=DP)
        )
        assert "gtp_remat_group" not in seen
    finally:
        DSAIndexerLossLoggingHelper.tracker = {}
