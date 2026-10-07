# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Mixed differentiable and forward-only native graph surfaces."""

from types import SimpleNamespace

import pytest
import torch

import megatron.core.tensor_parallel.random as rng
import megatron.core.transformer.cuda_graphs as cg
from tests.unit_tests.test_utilities import Utils


class _MixedOutputs(torch.nn.Module):
    def __init__(self, device, dtype, is_last_layer):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((8,), 0.5, device=device, dtype=dtype))
        self.weight.main_grad = torch.zeros(8, device=device, dtype=torch.float32)
        self.is_last_layer = is_last_layer

    def forward(self, hidden_states, snapshot, residual):
        # Own saved inputs inside the graph, independently of native pool reuse.
        x, bank, skip = hidden_states.clone(), snapshot.clone(), residual.clone()
        return (
            x * self.weight + bank.to(x.dtype),
            (x.detach().float() + bank).detach(),
            skip * self.weight - bank.to(skip.dtype),
        )


def _inputs(step, device, dtype):
    x = torch.arange(8, device=device, dtype=dtype) * 0.25 + step * 0.5
    bank = torch.full_like(x, 0.25 * (step + 1), dtype=torch.float32)
    skip = torch.full_like(x, 0.5 * (step + 1))
    return x.requires_grad_(), bank, skip.requires_grad_()


def _check_replays(runner, model, device, dtype):
    expected_main_grad = torch.zeros_like(model.weight.main_grad)
    for step in range(3):
        inputs = _inputs(step, device, dtype)
        # Change the parameter and metadata between replays without replacing storage.
        with torch.no_grad():
            model.weight.add_(0.125)
        outputs = runner.replay_graph_capture(step == 0, inputs, {})
        assert tuple(t.requires_grad for t in outputs) == (True, False, True)
        assert outputs[1].grad_fn is None
        assert tuple(t is None for t in runner.static_grad_outputs) == (False, True, False)
        assert tuple(t is None for t in runner.static_grad_inputs) == (False, True, False)
        with torch.no_grad():
            expected_outputs = model(*inputs)
        for actual, expected, static in zip(
            outputs, expected_outputs, runner.fwd_graph_output_surface
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert (actual.data_ptr() != static.data_ptr()) == runner.is_last_layer
            if runner.is_last_layer:
                assert actual.can_skip_replay_copy is False

        dy = torch.full_like(inputs[0], (step + 1) * 0.5)
        # An unused differentiable output must retain the native zero-gradient behavior.
        dz = torch.full_like(inputs[2], 0.0 if step == 1 else 0.25)
        loss = (outputs[0] * dy).sum() + outputs[1].sum() * 100
        if step != 1:
            loss = loss + (outputs[2] * dz).sum()
        runner.status = cg._GraphStatus.BWD_READY
        loss.backward()
        assert runner.status == cg._GraphStatus.FWD_READY
        assert inputs[1].grad is None
        torch.testing.assert_close(inputs[0].grad, dy * model.weight.detach(), rtol=0, atol=0)
        torch.testing.assert_close(inputs[2].grad, dz * model.weight.detach(), rtol=0, atol=0)
        with torch.no_grad():
            expected_main_grad.add_((dy * inputs[0] + dz * inputs[2]).float())
        torch.testing.assert_close(model.weight.main_grad, expected_main_grad, rtol=0, atol=0)
        assert model.weight.grad is None
        assert model.weight._cudagraph_wgrad_ready_event is runner.bwd_graph_replay_complete_event


@pytest.mark.parametrize("is_last_layer", [False, True])
def test_mixed_output_replay_autograd_cpu(monkeypatch, is_last_layer):
    """Exercise the real replay Function on CPU; this does not simulate CUDA capture."""
    model = _MixedOutputs("cpu", torch.float32, is_last_layer)
    inputs = _inputs(0, "cpu", torch.float32)
    surfaces = tuple(torch.empty_like(t) for t in inputs)
    outputs = tuple(torch.empty_like(t, requires_grad=i != 1) for i, t in enumerate(inputs))
    grad_outputs = (torch.zeros_like(inputs[0]), None, torch.zeros_like(inputs[2]))
    grad_inputs = (torch.zeros_like(inputs[0]), None, torch.zeros_like(inputs[2]))

    def forward():
        for dst, src in zip(outputs, model(*surfaces)):
            dst.copy_(src)

    def backward():
        grad_inputs[0].copy_(grad_outputs[0] * model.weight)
        grad_inputs[2].copy_(grad_outputs[2] * model.weight)
        model.weight.main_grad.add_(grad_outputs[0] * surfaces[0] + grad_outputs[2] * surfaces[2])

    runner = SimpleNamespace(
        fwd_graph=SimpleNamespace(replay=forward),
        bwd_graph=SimpleNamespace(replay=backward),
        fwd_graph_input_surface=surfaces + (model.weight,),
        fwd_graph_output_surface=outputs,
        fwd_graph_output_requires_grad=(True, False, True),
        static_grad_outputs=grad_outputs,
        static_grad_inputs=grad_inputs,
        params_to_backprop=(model.weight,),
        bwd_graph_replay_complete_event=SimpleNamespace(record=lambda stream: None),
        is_last_layer=is_last_layer,
        fp8_enabled=False,
        fp4_enabled=False,
        status=cg._GraphStatus.FWD_READY,
    )
    runner.replay_graph_capture = lambda first, args, kwargs: cg._CudagraphReplayNode.apply(
        runner, first, *args, model.weight
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    _check_replays(runner, model, "cpu", torch.float32)


@pytest.fixture
def native_capture(monkeypatch):
    """Isolate native graph globals while retaining real allocation and weak references."""
    Utils.initialize_model_parallel()
    monkeypatch.setattr(cg._CudagraphGlobalRecord, "cudagraph_created", False)
    monkeypatch.setattr(cg._CudagraphGlobalRecord, "cudagraph_record", [])
    monkeypatch.setattr(cg, "fwd_buffer_reuse_ref_count", 0)
    monkeypatch.setattr(cg, "bwd_buffer_reuse_ref_count", 0)
    monkeypatch.setattr(cg, "_IS_GRAPH_CAPTURING", False)
    monkeypatch.setattr(cg, "_IS_GRAPH_WARMUP", False)
    monkeypatch.setattr(cg.CudaGraphManager, "global_mempool", torch.cuda.graph_pool_handle())
    monkeypatch.setattr(rng, "_CUDA_RNG_STATE_TRACKER", None)
    monkeypatch.setattr(rng, "_CUDA_RNG_STATE_TRACKER_INITIALIZED", False)
    rng.initialize_rng_tracker(use_cudagraphable_rng=True)
    # Native managers select their side stream before model execution. Keep
    # parameter AccumulateGrad creation, eager recording and capture on that
    # lifecycle: moving only capture would retain default-stream grad nodes.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    try:
        yield
    finally:
        torch.cuda.synchronize()
        torch.cuda.set_stream(torch.cuda.default_stream())
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA graph capture")
@pytest.mark.parametrize("is_last_layer", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_record_capture_mixed_outputs(native_capture, is_last_layer, dtype):
    """Record and capture real F/B graphs, then check three evolving replay cycles."""
    model = _MixedOutputs("cuda", dtype, is_last_layer)
    inputs = _inputs(0, "cuda", dtype)
    runner = cg._CudaGraphRunner(
        model, cg.CudaGraphManager.global_mempool, inputs, {}, model.forward, need_backward=True
    )
    runner.num_warmup_steps = 2
    outputs = runner.record_graph_capture(inputs, {})
    assert tuple(t.requires_grad for t in outputs) == (True, False, True)
    runner.status = cg._GraphStatus.BWD_READY
    (outputs[0].sum() + outputs[2].sum()).backward()
    assert [entry[1] for entry in cg._CudagraphGlobalRecord.cudagraph_record] == ["fwd", "bwd"]
    model.weight.grad = None

    # No capture path is mocked; the fixture selected the stream before record.
    cg._CudagraphGlobalRecord.create_cudagraphs()
    torch.cuda.synchronize()
    assert runner.cudagraph_created
    assert runner.fwd_graph_output_requires_grad == (True, False, True)
    model.weight.main_grad.zero_()
    _check_replays(runner, model, "cuda", dtype)
