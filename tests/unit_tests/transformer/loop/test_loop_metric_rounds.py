# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-(layer, round) metric bins of depth-recurrent loops (M8 glue; plan: MoE metrics and the
indexer loss "summed over recursions" get a round dimension).

``forward_statistics_round(r, rounds)`` marks the forwards of loop round ``r``: the MoE aux /
z-loss tracker (``MoEMetricsTracker``) and the DSA indexer-loss tracker also add every value
they record into a ``[rounds, layers]`` bin; ``report(loop_rounds=)`` /
``track_indexer_metrics(loop_rounds=)`` reduce the bins like the per-layer values and log
``"{name} round {r}"`` to the writers, leaving every existing key unchanged. The eager loop
stack enters the scope per round, the RSBF executor per sweep (the counting forward; replays
record nothing), so both bin a round's records alike.

The trackers here hold CPU tensors, so their report collectives run on a one-rank Gloo group of
each rank (the world group is NCCL on the GPU CI); the model cases run on CPU or GPU.
"""

import copy
import functools
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.experimental_attention_variant.dsa_logging import (
    DSAIndexerLossLoggingHelper,
    _record_tracker_pp_group,
)
from megatron.core.transformer.forward_replay import (
    forward_replay,
    forward_statistics_round,
    forward_statistics_weight,
    statistics_round,
)
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker
from tests.unit_tests.transformer.loop import hybrid_loop_utils
from tests.unit_tests.transformer.loop import test_loop_rsbf_executor as base
from tests.unit_tests.transformer.loop.hybrid_loop_utils import model_parallel_once

cpu_mtp = hybrid_loop_utils.cpu_mtp
hybrid = base.hybrid
DEPTHS = base.DEPTHS  # (1, 3, 2, 4) at R_max 4


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        try:
            yield
        finally:
            base.get_loop_rsbf_schedule().reset()


class _Writer:
    """A TensorBoard-writer stub recording ``add_scalar`` calls."""

    def __init__(self):
        self.scalars = {}

    def add_scalar(self, name, value, iteration):
        del iteration
        self.scalars[name] = float(value)


@functools.cache
def _local_group():
    """This rank's one-rank Gloo group (created once, by every rank: ``new_subgroups`` is
    collective)."""
    return dist.new_subgroups(group_size=1, backend="gloo")[0]


def _groups():
    local = _local_group()
    return SimpleNamespace(pp=local, dp=local, dp_cp_gtp_remat=local)


# ---------------------------------------------------------------------------
# The scope and the trackers' bins
# ---------------------------------------------------------------------------


def test_round_scope_nests_validates_and_resets():
    assert statistics_round() is None
    with forward_statistics_round(2, 4):
        assert statistics_round() == (2, 4)
        with forward_statistics_round(3, 4):
            assert statistics_round() == (3, 4)
        assert statistics_round() == (2, 4)
    with pytest.raises(RuntimeError, match="boom"):
        with forward_statistics_round(1, 4):
            raise RuntimeError("boom")
    assert statistics_round() is None
    for r, rounds in ((0, 4), (5, 4)):
        with pytest.raises(ValueError, match="outside"):
            with forward_statistics_round(r, rounds):
                pass


def test_moe_tracker_bins_records_by_round():
    """Records inside a round scope also land in ``round_values[r - 1, layer - 1]`` (weighted
    like ``values``); records outside one, and replays, add nothing to the bins."""
    tracker = MoEMetricsTracker()
    value = torch.tensor(0.25)
    tracker.record("load_balancing_loss", value, layer_number=1, num_layers=3)
    entry = tracker.metrics["load_balancing_loss"]
    assert entry.round_values is None
    with forward_statistics_round(2, 3):
        tracker.record("load_balancing_loss", value, layer_number=2, num_layers=3)
        with forward_statistics_weight(0.5):
            tracker.record("load_balancing_loss", value, layer_number=2, num_layers=3)
        with forward_replay():
            tracker.record("load_balancing_loss", value, layer_number=3, num_layers=3)
    with forward_statistics_round(1, 3):
        tracker.record("load_balancing_loss", value, layer_number=2, num_layers=3)
    assert torch.equal(entry.values, torch.tensor([0.25, 0.625, 0.0]))
    expected = torch.zeros(3, 3)
    expected[1, 1] = 0.375
    expected[0, 1] = 0.25
    assert torch.equal(entry.round_values, expected)
    # The bins over rounds sum to the round-scoped part of the per-layer values.
    assert torch.equal(entry.round_values.sum(0), torch.tensor([0.0, 0.625, 0.0]))
    tracker.clear()
    assert not entry.values.any() and not entry.round_values.any()


def _report(tracker, loop_rounds, per_layer=False):
    writer = _Writer()
    log = tracker.report(
        loss_scale=0.5,
        iteration=7,
        writer=writer,
        per_layer_logging=per_layer,
        num_moe_layers=2,
        pg_collection=_groups(),
        loop_rounds=loop_rounds,
    )
    return log, writer.scalars


def test_moe_report_logs_round_keys_and_keeps_the_others():
    """``report(loop_rounds=)`` adds ``"{name} round {r}"`` for every reported metric (zeros
    where nothing was binned, so every rank reduces the same tensors), normalized like
    ``"{name}"``; the console string, the other scalars and the per-layer keys are those of a
    report without rounds."""
    tracker = MoEMetricsTracker()
    with forward_statistics_round(1, 2):
        tracker.record("load_balancing_loss", torch.tensor(0.2), layer_number=2, num_layers=3)
    with forward_statistics_round(2, 2):
        tracker.record("load_balancing_loss", torch.tensor(0.6), layer_number=2, num_layers=3)
    tracker.record("load_balancing_loss", torch.tensor(1.0), layer_number=3, num_layers=3)
    tracker.record("z_loss", torch.tensor(0.4), layer_number=1, num_layers=3)
    plain = copy.deepcopy(tracker)
    log, scalars = _report(tracker, loop_rounds=2, per_layer=True)
    plain_log, plain_scalars = _report(plain, loop_rounds=None, per_layer=True)
    assert log == plain_log
    rounds = {key: value for key, value in scalars.items() if "round" in key}
    assert {key: value for key, value in scalars.items() if "round" not in key} == plain_scalars
    # Normalized as the main key: values * loss_scale summed over layers / num_moe_layers.
    assert rounds["load_balancing_loss round 1"] == pytest.approx(0.2 * 0.5 / 2)
    assert rounds["load_balancing_loss round 2"] == pytest.approx(0.6 * 0.5 / 2)
    assert rounds["z_loss round 1"] == rounds["z_loss round 2"] == 0.0
    assert scalars["load_balancing_loss"] == pytest.approx((0.2 + 0.6 + 1.0) * 0.5 / 2)
    assert rounds["moe/load_balancing_loss/round_2/layer_1"] == pytest.approx(0.3)
    assert len(rounds) == 2 * 2 + 2 * 2 * 3
    assert not tracker.metrics["load_balancing_loss"].round_values.any()


def test_dsa_tracker_bins_and_reports_rounds(monkeypatch):
    """The indexer-loss tracker: the same bins, reduced with the values and logged as
    ``"indexer loss round {r}"``; ``"indexer loss"`` and ``total_loss_dict`` unchanged."""
    results = {}
    for loop_rounds in (None, 2):
        tracker = {"values": torch.zeros(3), "agreed_size": 3}
        _record_tracker_pp_group(tracker, "agreed_size", _local_group())
        monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", tracker)
        save = DSAIndexerLossLoggingHelper.save_loss_to_tracker
        with forward_statistics_round(2, 2):
            save(torch.tensor(0.8), layer_number=2, num_layers=3)
        with forward_replay(), forward_statistics_round(1, 2):
            save(torch.tensor(5.0), layer_number=2, num_layers=3)
        save(torch.tensor(0.4), layer_number=3, num_layers=3)
        expected = torch.zeros(2, 3)
        expected[1, 1] = 0.8
        assert torch.equal(tracker["round_values"], expected)
        writer, totals = _Writer(), {}
        DSAIndexerLossLoggingHelper.track_indexer_metrics(
            loss_scale=0.5,
            iteration=3,
            writer=writer,
            total_loss_dict=totals,
            num_indexer_layers=2,
            pp_group=_local_group(),
            dp_group=_local_group(),
            loop_rounds=loop_rounds,
        )
        assert not tracker["values"].any() and not tracker["round_values"].any()
        results[loop_rounds] = (writer.scalars, totals)
    (plain, plain_totals), (scalars, totals) = results[None], results[2]
    assert totals.keys() == plain_totals.keys() == {"indexer loss"}
    assert torch.equal(totals["indexer loss"], plain_totals["indexer loss"])
    assert scalars["indexer loss"] == plain["indexer loss"] == pytest.approx(1.2 * 0.5 / 2)
    assert scalars["indexer loss round 1"] == 0.0
    assert scalars["indexer loss round 2"] == pytest.approx(0.8 * 0.5 / 2)
    assert set(scalars) == {"indexer loss", "indexer loss round 1", "indexer loss round 2"}


def test_training_log_asks_for_round_bins_of_hybrid_loops_only():
    """``training_log`` passes ``loop_rounds=loop_metric_rounds(args)`` to both trackers:
    ``loop_iterations`` for a Huginn hybrid loop (role fields set), None for a MoR hybrid loop
    (its stack does not bin rounds), the GPT-path loop and every other model (a config-only
    condition: every rank issues the same collectives)."""
    import ast
    import inspect

    from megatron.training import training

    hybrid_loop = SimpleNamespace(
        loop_architecture="huginn",
        loop_prelude_sublayers=2,
        loop_coda_sublayers=2,
        loop_iterations=4,
    )
    gpt_loop = SimpleNamespace(
        loop_architecture="huginn",
        loop_prelude_sublayers=None,
        loop_coda_sublayers=None,
        loop_iterations=4,
    )
    mor_hybrid_loop = SimpleNamespace(**{**vars(hybrid_loop), "loop_architecture": "mor"})
    assert training.loop_metric_rounds(hybrid_loop) == 4
    assert training.loop_metric_rounds(mor_hybrid_loop) is None
    assert training.loop_metric_rounds(gpt_loop) is None
    assert training.loop_metric_rounds(SimpleNamespace(loop_architecture="none")) is None
    tree = ast.parse(inspect.getsource(training.training_log))
    calls = {
        ast.unparse(node.func): node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func)
        in ("get_moe_metrics_tracker().report", "DSAIndexerLossLoggingHelper.track_indexer_metrics")
    }
    assert len(calls) == 2
    for node in calls.values():
        keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in node.keywords}
        assert keywords["loop_rounds"] == "loop_metric_rounds(args)"


# ---------------------------------------------------------------------------
# The eager loop stack and the RSBF sweep bin alike
# ---------------------------------------------------------------------------


def _binning_spy(model, tracker):
    """A forward hook on every decoder sublayer recording one unit per call into ``tracker``
    (as the MoE router records its losses: gated by replays, binned by round)."""
    layers = len(model.decoder.layers)
    return [
        layer.register_forward_hook(
            lambda module, args, output, index=index: tracker.record(
                "calls", torch.ones(()), layer_number=index + 1, num_layers=layers
            )
        )
        for index, layer in enumerate(model.decoder.layers)
    ]


@pytest.mark.parametrize("form", ["whole_round", "nested"])
def test_eager_and_rsbf_bin_rounds_alike(request, hybrid, form):
    """At prescribed depths (1, 3, 2, 4): the eager loop (``set_loop_depths`` per sample) and
    RSBF (sweeps count; recomputes and rebuilds replay) record each core sublayer once per
    sample-round, binned by its round; prelude and coda records have no round."""
    overrides = {} if form == "whole_round" else dict(loop_rsbf_item_form="nested")
    model = base._model("mtp", request, **overrides)
    roles = model.decoder.loop_roles
    batches = base._batches(len(DEPTHS))
    trackers = {}
    for arm in ("eager", "rsbf"):
        tracker = MoEMetricsTracker()
        handles = _binning_spy(model, tracker)
        try:
            if arm == "eager":
                base._eager(model, hybrid, batches, DEPTHS)
            else:
                base._rsbf(model, hybrid, batches, DEPTHS)
        finally:
            for handle in handles:
                handle.remove()
        trackers[arm] = tracker.metrics["calls"]
    eager, rsbf = trackers["eager"], trackers["rsbf"]
    assert torch.equal(eager.values, rsbf.values)
    assert torch.equal(eager.round_values, rsbf.round_values)
    layers = len(model.decoder.layers)
    expected = torch.zeros(base.R_MAX, layers)
    for depth in DEPTHS:
        expected[:depth, roles.core_start : roles.core_end] += 1
    assert torch.equal(rsbf.round_values, expected)
    per_layer = torch.full((layers,), float(len(DEPTHS)))
    per_layer[roles.core_start : roles.core_end] = float(sum(DEPTHS))
    assert torch.equal(rsbf.values, per_layer)
    assert statistics_round() is None
