# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the forward-replay gate and the statistics record weight
(``megatron/core/transformer/forward_replay.py``) and their call sites: the MoE metrics
tracker, the DSA indexer-loss tracker, the overload-factor forward record, the router's
expert-bias counts and the loop recompute context.

CPU: the context managers and the tracker / overload gates. GPU: a MoE-core hybrid loop with
expert bias and overload logging counts each round once with ``loop_recompute`` on, as with it
off (the non-reentrant checkpoint replays with grad enabled; before the gate it counted twice).
"""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa_logging import (
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.forward_replay import (
    forward_replay,
    forward_statistics_weight,
    in_forward_replay,
    statistics_weight,
)
from megatron.core.transformer.loop_transformer_block import recompute_context
from megatron.core.transformer.moe.moe_logging import (
    MoEMetricsTracker,
    MoEOverloadFactorTracker,
    set_moe_metrics_tracker,
    set_moe_overload_factor_tracker,
)
from megatron.core.transformer.moe.moe_utils import record_dispatch_token_counts

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="TE MoE layers need CUDA")


def test_replay_marker_nests_and_resets_on_exceptions():
    assert not in_forward_replay()
    with forward_replay():
        assert in_forward_replay()
        with forward_replay():
            assert in_forward_replay()
        assert in_forward_replay()
    assert not in_forward_replay()
    with pytest.raises(RuntimeError):
        with forward_replay():
            raise RuntimeError("boom")
    assert not in_forward_replay()
    with recompute_context("loop=1/phase=recompute"):
        assert in_forward_replay()
    assert not in_forward_replay()


def test_statistics_weight_multiplies_and_resets():
    assert statistics_weight() == 1.0
    with forward_statistics_weight(0.5):
        assert statistics_weight() == 0.5
        with forward_statistics_weight(0.5):
            assert statistics_weight() == 0.25
        assert statistics_weight() == 0.5
    with pytest.raises(RuntimeError):
        with forward_statistics_weight(0.25):
            raise RuntimeError("boom")
    assert statistics_weight() == 1.0
    for weight in (0.0, -1.0):
        with pytest.raises(ValueError, match="positive"):
            with forward_statistics_weight(weight):
                pass


def test_moe_metrics_tracker_gate_and_weight():
    tracker = MoEMetricsTracker()
    value = torch.tensor(0.3)
    tracker.record("load_balancing_loss", value, layer_number=2, num_layers=3)
    reference = tracker.metrics["load_balancing_loss"].values.clone()
    with forward_replay():
        tracker.record("load_balancing_loss", value, layer_number=2, num_layers=3)
        tracker.record("z_loss", value, layer_number=1, num_layers=3)
    assert torch.equal(tracker.metrics["load_balancing_loss"].values, reference)
    assert "z_loss" not in tracker.metrics
    with forward_statistics_weight(0.5):
        tracker.record("load_balancing_loss", value, layer_number=2, num_layers=3)
    assert torch.equal(
        tracker.metrics["load_balancing_loss"].values, reference + torch.tensor([0, 0.15, 0])
    )
    # Weight 1.0 records exactly the unweighted value (dev behaviour unchanged).
    plain = MoEMetricsTracker()
    plain.record("load_balancing_loss", value, layer_number=2, num_layers=3)
    assert torch.equal(plain.metrics["load_balancing_loss"].values, reference)


def test_dsa_indexer_tracker_gate_and_weight(monkeypatch):
    tracker = {"values": torch.zeros(3)}
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", tracker)
    loss = torch.tensor(0.8)
    DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_number=1, num_layers=3)
    with forward_replay():
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_number=1, num_layers=3)
    assert torch.equal(tracker["values"], torch.tensor([0.8, 0.0, 0.0]))
    with forward_statistics_weight(0.5):
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_number=3, num_layers=3)
    assert torch.equal(tracker["values"], torch.tensor([0.8, 0.0, 0.4]))


def test_overload_forward_record_is_gated_backward_is_not():
    tracker = MoEOverloadFactorTracker()
    set_moe_overload_factor_tracker(tracker)
    try:
        counts = torch.tensor([3.0, 5.0])
        balanced = torch.tensor(8.0)
        x = torch.ones(2, requires_grad=True)
        record_dispatch_token_counts(x * 1.0, counts, balanced, 1).sum().backward()
        assert len(tracker._cumulative_tokens_timeline) == 2  # one forward, one backward
        with forward_replay():
            y = record_dispatch_token_counts(x * 1.0, counts, balanced, 1)
        assert len(tracker._cumulative_tokens_timeline) == 2  # the replay records no forward
        y.sum().backward()
        timeline = [value.item() for value in tracker._cumulative_tokens_timeline]
        assert timeline == [8.0, -8.0, -8.0]
        assert [len(values) for values in tracker._layer_fwd_tokens.values()] == [1]
    finally:
        set_moe_overload_factor_tracker(None)


@gpu
def test_loop_recompute_counts_each_round_once():
    """A '*-*E*E*-' hybrid loop (MoE core, expert bias, overload logging) at depth 3: the
    expert-bias token counts, the logged aux loss and the overload forward / backward records
    with loop_recompute on equal those with it off."""
    from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_stack_spec
    from megatron.core.transformer.spec_utils import build_module
    from tests.unit_tests.test_utilities import Utils
    from tests.unit_tests.transformer.loop.hybrid_parity_utils import (
        loop_fields,
        make_hybrid_config,
        make_inputs,
        te_base,
        te_rope,
    )

    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    try:
        pattern = "*-*E*E*-"
        base = te_base(
            torch.bfloat16,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_ffn_hidden_size=256,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_aux_loss_coeff=0.01,
            log_moe_overload_factor=True,
        )
        results = {}
        for recompute in (False, True):
            config = make_hybrid_config(base, loop_fields("huginn", 3, 0, recompute, "compact"))
            torch.manual_seed(1234)
            stack = build_module(
                loop_hybrid_stack_spec,
                config=config,
                pre_process=True,
                layer_config_list=validate_segment_layers(pattern, config),
                post_process=True,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            ).cuda()
            x, upstream = make_inputs(
                torch.bfloat16, hidden=config.hidden_size, scale=config.init_method_std, seed=3
            )
            metrics = MoEMetricsTracker()
            set_moe_metrics_tracker(metrics)
            overload = MoEOverloadFactorTracker()
            set_moe_overload_factor_tracker(overload)
            x = x.detach().requires_grad_()
            y = stack(x, attention_mask=None, rotary_pos_emb=te_rope(config.kv_channels))
            (y * upstream).float().sum().backward()
            counts = [
                module.local_tokens_per_expert.clone()
                for module in stack.modules()
                if getattr(module, "local_tokens_per_expert", None) is not None
            ]
            logged = {name: entry.values.clone() for name, entry in metrics.metrics.items()}
            records = (
                sum(len(values) for values in overload._layer_fwd_tokens.values()),
                len(overload._cumulative_tokens_timeline),
            )
            results[recompute] = (counts, logged, records)
        set_moe_overload_factor_tracker(None)
        set_moe_metrics_tracker(MoEMetricsTracker())
        (counts_off, logged_off, records_off), (counts_on, logged_on, records_on) = (
            results[False],
            results[True],
        )
        assert counts_off and len(counts_off) == len(counts_on)
        for off, on in zip(counts_off, counts_on):
            assert torch.equal(off, on)
        assert logged_off.keys() == logged_on.keys() and logged_off
        for name in logged_off:
            torch.testing.assert_close(logged_on[name], logged_off[name], rtol=1e-2, atol=1e-6)
        # 2 MoE sublayers x 3 rounds forward; every forward record paired by one backward.
        assert records_off == records_on == (6, 12)
    finally:
        Utils.destroy_model_parallel()
