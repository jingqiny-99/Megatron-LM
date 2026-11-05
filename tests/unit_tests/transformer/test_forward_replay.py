# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the forward-replay gate and the statistics record weight
(``megatron/core/transformer/forward_replay.py``) and their call sites: the MoE metrics
tracker, the DSA indexer-loss tracker, the overload-factor forward record, the router's
expert-bias counts and the loop recompute context.

CPU: the context managers and the tracker / overload gates. GPU: a MoE-core hybrid loop with
expert bias and overload logging counts each round once with ``loop_recompute`` on, as with it
off (the non-reentrant checkpoint replays with grad enabled; before the gate it counted twice).

The routing tape (``RoutingTape``, ``loop_checkpoint_context_fn``): CPU, its record / replay
rules, a checkpointed MoE-like block whose recompute routes as its forward did, and the fused
router request under a tape (routed unfused, with the unfused output contract of
``TopKRouter.routing``); GPU, a MoE-core hybrid loop (injection and attention-residual
carriers) under ``loop_recompute`` whose routers see a different recompute input (the
learned-routing ``CheckpointError`` of the deep probe, with a control without the tape). Every
loop checkpoint site on CPU / CUDA: ``loop/test_loop_routing_tape_sites.py``.
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


@pytest.fixture(scope="module")
def model_parallel():
    """Model parallelism for this module's GPU tests, initialized once per module (the CI
    process-group budget); each test reseeds."""
    from tests.unit_tests.transformer.loop.hybrid_loop_utils import model_parallel_once

    with model_parallel_once():
        yield


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
@pytest.mark.usefixtures("model_parallel")
def test_loop_recompute_counts_each_round_once():
    """A '*-*E*E*-' hybrid loop (MoE core, expert bias, overload logging) at depth 3: the
    expert-bias token counts, the logged aux loss and the overload forward / backward records
    with loop_recompute on equal those with it off."""
    from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_stack_spec
    from megatron.core.transformer.spec_utils import build_module
    from tests.unit_tests.transformer.loop.hybrid_parity_utils import (
        loop_fields,
        make_hybrid_config,
        make_inputs,
        te_base,
        te_rope,
    )

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
        set_moe_overload_factor_tracker(None)
        set_moe_metrics_tracker(MoEMetricsTracker())


def test_routing_tape_records_and_replays_in_order():
    from megatron.core.transformer.forward_replay import (
        RoutingTape,
        record_routing,
        replay_routing,
        routing_tape_active,
        taped_topk,
    )

    def topk(scores):
        return torch.topk(scores, k=2, dim=1)

    first, second = torch.randn(5, 4), torch.randn(5, 4)
    tape = RoutingTape()
    assert not routing_tape_active()
    assert torch.equal(taped_topk(first, topk)[1], topk(first)[1])  # no tape: unchanged
    with record_routing(tape):
        assert routing_tape_active()
        recorded = [taped_topk(first, topk)[1], taped_topk(second, topk)[1]]
    assert not routing_tape_active() and len(tape.indices) == 2
    other = torch.randn(5, 4)
    for _ in range(2):  # every recompute replays from the start
        with replay_routing(tape):
            values, indices = taped_topk(other, topk)
            assert torch.equal(indices, recorded[0])
            assert torch.equal(values, other.gather(1, recorded[0]))
            assert torch.equal(taped_topk(other, topk)[1], recorded[1])
    # A recompute that routes fewer times than its forward is an error; an early stop (an
    # exception through the scope) is not.
    with pytest.raises(RuntimeError, match="routed 1 times, its forward 2 times"):
        with replay_routing(tape):
            taped_topk(other, topk)
    with pytest.raises(KeyError):
        with replay_routing(tape):
            taped_topk(other, topk)
            raise KeyError("early stop")
    with pytest.raises(RuntimeError, match="more often than its forward"):
        with replay_routing(tape):
            for _ in range(3):
                taped_topk(other, topk)
    with pytest.raises(RuntimeError, match="has scores"):
        with replay_routing(tape):
            taped_topk(torch.randn(6, 4), topk)
    assert not routing_tape_active()


def _fake_fused_topk(calls):
    """A stand-in for TE's ``fused_topk_with_score_function`` that logs its calls (the fused
    path's output: full-width probs and the dense ``topk_indices`` buffer when given, else a
    bool map)."""

    def fused_topk_with_score_function(**kwargs):
        calls.append(kwargs)
        logits = kwargs["logits"]
        values, indices = torch.topk(logits, kwargs["topk"], dim=1)
        probs = torch.zeros_like(logits).scatter(1, indices, values.softmax(-1))
        if kwargs.get("topk_indices") is not None:
            kwargs["topk_indices"].copy_(indices)
            return probs, kwargs["topk_indices"]
        return probs, probs > 0

    return fused_topk_with_score_function


def test_fused_routing_under_a_tape_takes_the_taped_unfused_path(monkeypatch):
    """The fused TE top-k has no tape hook: a fused request under a routing tape routes on the
    unfused path, whose top-k the tape records; without a tape the fused kernel runs."""
    from megatron.core.transformer.forward_replay import RoutingTape, record_routing
    from megatron.core.transformer.moe import moe_utils

    calls = []
    monkeypatch.setattr(moe_utils, "HAVE_TE", True)
    monkeypatch.setattr(moe_utils, "fused_topk_with_score_function", _fake_fused_topk(calls))
    logits = torch.randn(6, 8, generator=torch.Generator().manual_seed(1))
    tape = RoutingTape()
    with record_routing(tape):
        probs, indices = moe_utils.topk_routing_with_score_function(
            logits, 2, fused=True, dense_output=True
        )
    assert not calls
    assert len(tape.indices) == 1 and torch.equal(indices, tape.indices[0])
    assert torch.equal(indices, torch.topk(logits, 2, dim=1).indices) and probs.shape == (6, 2)
    moe_utils.topk_routing_with_score_function(logits, 2, fused=True)
    assert len(calls) == 1


def test_taped_router_keeps_the_unfused_output_contract(monkeypatch):
    """``TopKRouter.routing`` picks fused / unfused before its output format: under a routing
    tape a router with ``moe_router_fusion`` that wants dense expert ids (the flex / MoonEP
    dispatchers) routes unfused and returns the ``[tokens, topk]`` ids the tape recorded. (The
    fused choice made first allocated a ``topk_indices`` buffer the unfused fallback never
    filled and returned a bool ``[tokens, experts]`` map.) Without a tape it stays fused."""
    from types import SimpleNamespace

    from megatron.core.transformer.forward_replay import RoutingTape, record_routing
    from megatron.core.transformer.moe import moe_utils
    from megatron.core.transformer.moe import router as router_module
    from megatron.core.transformer.transformer_config import TransformerConfig

    calls = []
    monkeypatch.setattr(moe_utils, "HAVE_TE", True)
    monkeypatch.setattr(moe_utils, "fused_topk_with_score_function", _fake_fused_topk(calls))
    monkeypatch.setattr(moe_utils, "fused_topk_with_score_function_supports_topk_indices", True)
    monkeypatch.setattr(router_module, "fused_topk_with_score_function_supports_topk_indices", True)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=1,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_fusion=True,
        use_cpu_initialization=True,
    )
    groups = SimpleNamespace(tp=None, expt_tp=None, cp=None, tp_cp=None, tp_dp_cp=None)
    router = router_module.TopKRouter(config=config, pg_collection=groups)
    monkeypatch.setattr(router, "_dense_route_indices_dtype", lambda: torch.int64)
    logits = torch.randn(3, 2, 8, generator=torch.Generator().manual_seed(2))  # [s, b, experts]
    tape = RoutingTape()
    with record_routing(tape):
        probs, routing_map = router.routing(logits)
    assert not calls
    assert routing_map.dtype == torch.int64 and routing_map.shape == (6, 2)
    assert len(tape.indices) == 1 and torch.equal(routing_map, tape.indices[0])
    assert probs.shape == (6, 2)
    _, untaped_map = router.routing(logits)
    assert len(calls) == 1 and calls[0]["topk_indices"] is untaped_map
    assert torch.equal(untaped_map, routing_map)


def _routed_block(x, router, experts, perturb):
    """A checkpointed MoE-like block whose saved tensors' shapes follow the routing: the
    rows of ``x`` each expert receives (``moe_utils.topk_routing_with_score_function``).
    ``perturb`` changes the recompute's input (a recompute that is not bitwise its forward)."""
    from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function

    if perturb and in_forward_replay():
        noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(9))
        x = x + noise * x.detach().std()
    probs, indices = topk_routing_with_score_function(
        x @ router, 2, score_function="softmax", dense_output=True
    )
    out = torch.zeros_like(x)
    for expert, weight in enumerate(experts):
        rows, slot = (indices == expert).nonzero(as_tuple=True)
        out = out.index_add(0, rows, (x[rows] @ weight) * probs[rows, slot, None])
    return out


@pytest.mark.parametrize("perturb", [False, True], ids=["exact", "perturbed"])
def test_loop_checkpoint_replays_the_forward_routing(perturb):
    """``loop_checkpoint_context_fn``: a recompute routes as its forward did. Exact recompute:
    gradients bitwise those without a checkpoint. Perturbed recompute (the non-bitwise
    recompute of a learned-routing MoE round): the backward runs; the control without the
    routing tape fails with the CheckpointError of the original bug."""
    from torch.utils.checkpoint import CheckpointError, checkpoint

    from megatron.core.transformer.loop_transformer_block import loop_checkpoint_context_fn

    generator = torch.Generator().manual_seed(4)
    x = torch.randn(64, 16, generator=generator)
    router = torch.randn(16, 4, generator=generator)
    experts = [torch.randn(16, 16, generator=generator) for _ in range(4)]

    def run(context_fn):
        leaves = [x.clone().requires_grad_(True)]
        leaves += [weight.clone().requires_grad_(True) for weight in experts]
        if context_fn is None:
            y = _routed_block(leaves[0], router, leaves[1:], perturb)
        else:
            y = checkpoint(
                _routed_block,
                leaves[0],
                router,
                leaves[1:],
                perturb,
                use_reentrant=False,
                context_fn=context_fn,
            )
        y.square().sum().backward()
        return y.detach(), [leaf.grad for leaf in leaves]

    y_ref, grads_ref = run(None)
    y, grads = run(loop_checkpoint_context_fn("t"))
    assert torch.equal(y, y_ref)
    if not perturb:
        for grad, ref in zip(grads, grads_ref):
            assert torch.equal(grad, ref)
        return
    assert all(torch.isfinite(grad).all() for grad in grads)
    with pytest.raises(CheckpointError):
        run(
            lambda: (
                torch.profiler.record_function("t/phase=forward"),
                recompute_context("t/phase=recompute"),
            )
        )


@gpu
@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("taped", [True, False], ids=["taped", "control"])
@pytest.mark.parametrize("carrier", ["injection", "attn_res"])
def test_loop_recompute_with_a_non_bitwise_replay_keeps_the_routing(monkeypatch, carrier, taped):
    """The deep-probe failure (h1024_c54 loop-mhc, learned routing, ``loop_recompute``:
    ``CheckpointError: Recomputed values ... have different metadata`` in the first backward):
    a '*-*E*E*-' MoE-core hybrid loop (TE layers, ``TopKRouter``, alltoall, grouped GEMM) whose
    routers see a different input in the recompute (a stand-in for a non-bitwise recompute,
    e.g. the unpermute's ``scatter_add_``), on the injection carrier (``LoopHybridStack``) and
    on the attention-residual carrier (``LoopAttnResHybridStack``, the loop-attnres variants'
    round checkpoint). With the routing tape the backward runs and the gradients are finite;
    the control (the round checkpoint's context without the tape, patched where
    ``loop_round_checkpoint`` reads it) raises the original CheckpointError."""
    from torch.utils.checkpoint import CheckpointError

    from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import loop_transformer_block
    from megatron.core.transformer.loop.attn_res_stack import LoopAttnResHybridStack
    from megatron.core.transformer.loop.hybrid_specs import (
        loop_attn_res_hybrid_stack_spec,
        loop_hybrid_stack_spec,
    )
    from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.spec_utils import build_module
    from tests.unit_tests.transformer.loop.hybrid_parity_utils import (
        loop_fields,
        make_hybrid_config,
        make_inputs,
        te_base,
        te_rope,
    )

    if not taped:

        def untaped(prefix):
            return lambda: (
                torch.profiler.record_function(f"{prefix}/phase=forward"),
                recompute_context(f"{prefix}/phase=recompute"),
            )

        # Every loop round checkpoint takes its context from loop_round_checkpoint.
        monkeypatch.setattr(loop_transformer_block, "loop_checkpoint_context_fn", untaped)

    def perturb_replay(module, args, kwargs):
        if not in_forward_replay():
            return None
        x = args[0]
        generator = torch.Generator(device=x.device).manual_seed(7)
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        return (x + noise * x.detach().float().std().to(x.dtype), *args[1:]), kwargs

    model_parallel_cuda_manual_seed(123)
    try:
        base = te_base(
            torch.bfloat16,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_ffn_hidden_size=256,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_aux_loss_coeff=0.01,
        )
        loop = loop_fields("huginn", 3, 0, True, "compact")
        if carrier == "attn_res":
            loop.update(loop_carrier="attn_res", loop_attn_res_blocks_per_round=1)
        config = make_hybrid_config(base, loop)
        torch.manual_seed(1234)
        stack = build_module(
            loop_attn_res_hybrid_stack_spec if carrier == "attn_res" else loop_hybrid_stack_spec,
            config=config,
            pre_process=True,
            layer_config_list=validate_segment_layers("*-*E*E*-", config),
            post_process=True,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        ).cuda()
        assert type(stack) is (LoopAttnResHybridStack if carrier == "attn_res" else LoopHybridStack)
        routers = [module for module in stack.modules() if isinstance(module, TopKRouter)]
        assert routers
        handles = [
            router.register_forward_pre_hook(perturb_replay, with_kwargs=True) for router in routers
        ]
        set_moe_metrics_tracker(MoEMetricsTracker())
        x, upstream = make_inputs(
            torch.bfloat16, hidden=config.hidden_size, scale=config.init_method_std, seed=3
        )
        x = x.detach().requires_grad_()
        try:
            y = stack(x, attention_mask=None, rotary_pos_emb=te_rope(config.kv_channels))
            loss = (y * upstream).float().sum()
            if taped:
                loss.backward()
                assert torch.isfinite(x.grad).all()
                grads = [p.grad for p in stack.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
            else:
                with pytest.raises(CheckpointError):
                    loss.backward()
        finally:
            for handle in handles:
                handle.remove()
    finally:
        set_moe_metrics_tracker(MoEMetricsTracker())
