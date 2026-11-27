# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GLM gate residual across independently scheduled pipeline backward calls."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from megatron.core.pipeline_parallel.schedules import custom_backward, deallocate_output_tensor
from megatron.core.transformer.loop.glm_ring_stack import GLMRingStack, _publish_gate_input
from megatron.core.transformer.loop.hybrid_stack import LoopResidualGate
from megatron.core.transformer.loop.ring_pp import ring_injection_use


def test_realized_online_ring_depths_reach_native_wandb_and_tensorboard(monkeypatch):
    from megatron.training import loop_ring_report, loop_rsbf_planning

    args = SimpleNamespace(seq_length=2048, data_parallel_size=1, log_interval=0)
    record = SimpleNamespace(
        forward_only=False,
        microbatches=8,
        views=5,
        exit_counts=(2, 4, 2),
        online_decisions=(),
        summary=lambda: "ring",
    )
    monkeypatch.setattr(loop_ring_report, "print_rank_last", lambda line: None)
    monkeypatch.setattr(loop_rsbf_planning, "is_last_rank", lambda: True)
    monkeypatch.setattr(loop_rsbf_planning._EXIT_HISTOGRAM, "log", None)
    loop_ring_report.log_loop_ring_step(args, SimpleNamespace(last_step=record), 7)
    tb, wb = Mock(), Mock()
    loop_rsbf_planning.log_loop_exit_histogram(args, 7, tb, wb)
    assert args._loop_exit_counts_interval == [2, 4, 2]
    assert wb.log.call_args.args[0]["loop exit depth count/d2"] == 4
    assert wb.log.call_args.args[0]["loop exit depth mean"] == 2.0
    tb.add_scalar.assert_any_call("loop exit depth fraction/d3", 0.25, 7)


def test_ring_declines_single_sequence_thd_before_an_edge_runs():
    stack = SimpleNamespace(
        config=SimpleNamespace(experimental_attention_variant="dsa"), loop_ring=True
    )
    cu = torch.tensor([0, 12], dtype=torch.int32)
    packed = SimpleNamespace(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
    with pytest.raises(ValueError, match="unpacked SBHD"):
        GLMRingStack._layer_loop(
            stack, torch.empty(12, 1, 8), SimpleNamespace(packed_seq_params=packed)
        )


@pytest.mark.parametrize("training", [False, True])
def test_pass_input_survives_send_and_gate_gradients_join_the_source(training):
    torch.manual_seed(57)
    cfg = SimpleNamespace(
        hidden_size=8,
        params_dtype=torch.float64,
        layernorm_epsilon=1e-5,
        init_method=nn.init.xavier_uniform_,
    )
    core = nn.Linear(8, 8, dtype=torch.float64)
    gates = nn.ModuleList(LoopResidualGate(cfg, 1, "cpu") for _ in range(2))
    ref_core, ref_gates = copy.deepcopy((core, gates))
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=training)
    ref_x = x.detach().clone().requires_grad_(training)
    bank = {}
    with torch.set_grad_enabled(training):
        oracle = ref_x.sin()
        for gate in ref_gates:
            oracle = gate(oracle, ref_core(oracle).tanh())
        sent = []
        received = []
        stage_output = x.sin()
        for lap in range(2):
            outgoing = _publish_gate_input(stage_output, bank, (3, lap))
            assert outgoing._base is None
            recv = outgoing.detach().clone().requires_grad_(training)
            sent.append(outgoing)
            received.append(recv)
            deallocate_output_tensor(outgoing, deallocate_pipeline_outputs=True)
            entry = bank[(3, lap)]
            assert entry.tensor.shape == x.shape
            stage_output = gates[lap](ring_injection_use(entry), core(recv).tanh())
            if not training:
                entry.release()
        torch.testing.assert_close(stage_output, oracle, rtol=0, atol=0)
        if training:
            gradient = torch.randn_like(oracle)
            oracle.backward(gradient)
            stage_output.backward(gradient)
            for outgoing, recv in reversed(list(zip(sent, received))):
                custom_backward(outgoing, recv.grad)
            torch.testing.assert_close(x.grad, ref_x.grad)
            for actual, expected in zip(
                [*core.parameters(), *gates.parameters()],
                [*ref_core.parameters(), *ref_gates.parameters()],
                strict=True,
            ):
                torch.testing.assert_close(actual.grad, expected.grad)
    assert not bank


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("online", [False, True], ids=["known", "online"])
def test_stratified_laps_skip_exited_core_and_backpropagate_two_sparse_reads(depth, online):
    from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
    from megatron.core.transformer.loop.ring_online import RingOnlineExit
    from megatron.core.transformer.loop.ring_pp import LoopRingCall

    torch.manual_seed(41)
    cfg = SimpleNamespace(
        hidden_size=8,
        params_dtype=torch.float64,
        layernorm_epsilon=1e-5,
        init_method=nn.init.xavier_uniform_,
        loop_iterations=3,
        loop_exit_criterion="online" if online else "stratified",
        loop_exit_thresholds=[1e6 if r == depth else -1e6 for r in (1, 2)],
        loop_recompute_edges=False,
        loop_ring_separate_edges=True,
    )
    gates = nn.ModuleList(LoopResidualGate(cfg, 1, "cpu") for _ in range(3))
    kernel = nn.Linear(8, 8, dtype=torch.float64)
    ref_gates, ref_kernel = copy.deepcopy((gates, kernel))
    x = torch.randn(5, 1, 8, dtype=torch.float64, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    expected = ref_x.sin()
    source = expected.square()
    for i in range(depth):
        expected = ref_gates[i](expected, ref_kernel(expected + source).tanh())
    gradient = torch.randn_like(expected)
    (expected * 0.9).backward(gradient)

    ctx = SimpleNamespace(
        packed_seq_params=None, inference_context=None, attention_mask=None, padding_mask=None
    )
    gate_bank, sparse_bank, visits, observed_depths = {}, {}, [], []
    fake = SimpleNamespace(
        config=cfg,
        _loop_depths=None,
        loop_gates=gates,
        _cross_loop_attention_layers={0: True},
        _ring_counts={},
        _ring_lap_work=lambda *_: (),
        _ring_head_metrics=lambda *a, depth: observed_depths.append(depth),
    )
    fake._lap_uses_checkpoint = lambda checkpoint, core: GLMRingStack._lap_uses_checkpoint(
        fake, checkpoint, core
    )
    fake._ring_sparse_pass = lambda call: LoopHybridStack._ring_sparse_pass(fake, call)

    def local(state, ctx, label, scope, recompute):
        visits.append(fake.call_roles.lap)
        if scope is None:
            return state.sin() if fake.call_roles.lap == 0 else state * 0.9
        if scope.is_source:
            value = state.square()
            scope.publish(kernel, "latent", value)
        else:
            value = scope.read(kernel, "latent")
        return kernel(state + value).tanh()

    fake._ring_local_layers = local
    sends, recvs = [], []
    state = x
    control = RingOnlineExit(cfg, None) if online else None
    for lap, role in enumerate(("prelude", "core", "core", "core", "coda")):
        if online and lap:
            state = control.receive(0, lap, state)
        fake.loop_ring = SimpleNamespace(role=role, round_start=True, round_end=True)
        fake.call_roles = LoopRingCall(
            pre_process=lap == 0,
            post_process=lap == 4,
            mtp_process=False,
            lap=lap,
            round=max(0, min(2, lap - 1)),
            microbatch=0,
            depth=control.stopped.get(0, 3) if online else depth,
            online_exit=control,
            sparse_banks=sparse_bank,
            gate_banks=gate_bank,
        )
        outgoing = GLMRingStack._ring_layer_loop(fake, state, ctx)
        if lap < 4:
            if online:
                outgoing = control.send(0, outgoing)
            state = outgoing.detach().clone().requires_grad_()
            sends.append(outgoing)
            recvs.append(state)
            deallocate_output_tensor(outgoing, deallocate_pipeline_outputs=True)
    torch.testing.assert_close(outgoing, expected * 0.9)
    outgoing.backward(gradient)
    for send, recv in reversed(list(zip(sends, recvs, strict=True))):
        custom_backward(send, recv.grad)
    assert visits == [0, *range(1, depth + 1), 4]
    assert observed_depths == [depth] and fake._ring_counts[0] == [depth, 1, 1, 0]
    assert not gate_bank and not sparse_bank
    if online:
        assert control.finish(1, owner=True) == (depth,)
        assert [d[:3] for d in control.records()] == [
            (0, r, r == depth) for r in range(1, depth + 1)
        ]
    torch.testing.assert_close(x.grad, ref_x.grad)
    for check, ref in zip(
        [*kernel.parameters(), *gates.parameters()],
        [*ref_kernel.parameters(), *ref_gates.parameters()],
        strict=True,
    ):
        torch.testing.assert_close(check.grad, ref.grad)
