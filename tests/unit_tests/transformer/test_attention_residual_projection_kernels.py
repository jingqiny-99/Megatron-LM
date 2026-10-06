# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Source/consumer parity against ordinary PyTorch autograd.

The independent reference is the RMSNorm + depth softmax equation in
docs/attention_residual_source_projection.md and attention_residual.py, not a
copy of the custom backward. CPU cases use the Torch custom-autograd path;
CUDA cases execute Triton and keep the same accuracy gates.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# The kernel module is deliberately independent of MCore's distributed setup.
# Loading its file also permits real CPU math tests without importing optional
# CUDA-only modules from megatron.core.__init__.
_KERNEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "megatron/core/transformer/attention_residual_projection_kernels.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "attnres_projection_kernel_test_module", _KERNEL_PATH
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
project_source = _MODULE.project_source
project_source_reference = _MODULE.project_source_reference
aggregate_preprojected = _MODULE.aggregate_preprojected
aggregate_preprojected_reference = _MODULE.aggregate_preprojected_reference


def _leaf(tensor):
    return tensor.detach().clone().requires_grad_()


def _check(actual, expected, kind="value"):
    assert actual is not None
    assert torch.isfinite(actual).all()
    if kind == "norm":
        atol, rtol, limit = 5e-5, 5e-4, 1e-4
    elif kind == "query":
        atol, rtol, limit = 1e-3, 5e-4, 1e-4
    elif kind == "state":
        atol, rtol, limit = 2e-5, 2e-4, 2e-5
    elif actual.dtype == torch.bfloat16:
        atol, rtol, limit = 0.016, 0.01, 0.003
    else:
        atol, rtol, limit = 2e-5, 2e-4, 2e-5
    torch.testing.assert_close(actual.double(), expected.double(), atol=atol, rtol=rtol)
    error = (actual.double() - expected.double()).norm()
    relative = error / expected.double().norm().clamp_min(1e-30)
    assert relative < limit, (kind, relative.item(), limit)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape,rows", [((7, 1024), 5), ((3, 2, 33), 3), ((1, 1024), 0)])
def test_source_forward_and_all_gradients(dtype, shape, rows):
    torch.manual_seed(41)
    value = torch.randn(shape, dtype=dtype, requires_grad=True)
    query = (torch.randn(rows, shape[-1]) * 0.02).requires_grad_()
    mask = torch.arange(rows) % 2 == 0
    ref_value, ref_query = _leaf(value), _leaf(query)
    proxy, logits = project_source(value, query, mask, backend="torch")
    ref_proxy, ref_logits = project_source_reference(ref_value, ref_query, mask)
    assert proxy.data_ptr() == value.data_ptr()
    assert logits.dtype == torch.float32 and logits.shape == (*shape[:-1], rows)
    if rows:
        _check(logits, ref_logits, "state")
    direct = torch.randn_like(proxy)
    upstream = torch.randn_like(logits)
    grads = torch.autograd.grad((proxy, logits), (value, query), (direct, upstream))
    refs = torch.autograd.grad((ref_proxy, ref_logits), (ref_value, ref_query), (direct, upstream))
    _check(grads[0], refs[0])
    if rows:
        _check(grads[1], refs[1], "query")
    else:
        assert grads[1].shape == refs[1].shape == (0, shape[-1])


def test_stopped_source_keeps_query_gradient_and_value_proxy():
    torch.manual_seed(4)
    value = torch.randn(4, 1024, requires_grad=True)
    query = torch.randn(3, 1024, requires_grad=True)
    proxy, logits = project_source(value, query, torch.zeros(3, dtype=torch.bool))
    logits.sum().backward(retain_graph=True)
    assert torch.count_nonzero(value.grad) == 0
    assert query.grad.abs().sum() > 0
    value.grad = query.grad = None
    proxy.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))
    assert torch.count_nonzero(query.grad) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("detach_last", [False, True])
def test_paired_consumer_value_gradient(device, dtype, fraction, detach_last):
    """Detached producer inputs keep dW live while each consumer owns complete dV."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA/Triton")
    torch.manual_seed(724)
    sources = [torch.randn(7, 1024, device=device, dtype=dtype) for _ in range(2)]
    partials = [torch.randn_like(sources[0]) for _ in range(2)]
    query = torch.randn(2, 1024, device=device) * 0.02
    norm = 1 + torch.randn_like(query) * 0.1
    upstream = [torch.randn_like(sources[0]) for _ in range(2)]
    count = math_ceil_fraction(fraction, len(sources))

    def run(reference, projected_count=count):
        values, changing = [_leaf(value) for value in sources], [_leaf(value) for value in partials]
        q, gamma = _leaf(query), _leaf(norm)
        effective = q * gamma
        project = project_source_reference if reference else project_source
        aggregate = aggregate_preprojected_reference if reference else aggregate_preprojected
        scores = [
            project(value.detach(), effective)[1] if index < projected_count else None
            for index, value in enumerate(values)
        ]
        outputs = []
        for row in range(2):
            history = [value.detach() if detach_last and row == 1 else value for value in values]
            outputs.append(
                aggregate(
                    [*history, changing[row]],
                    effective[row],
                    [*[score[..., row] if score is not None else None for score in scores], None],
                    precomputed_value_grad=True,
                )
            )
        gradients = torch.autograd.grad(outputs, [*values, *changing, q, gamma], upstream)
        return outputs, gradients

    actual_outputs, actual_gradients = run(False)
    expected_outputs, expected_gradients = run(True)
    for actual, expected in zip(actual_outputs, expected_outputs):
        _check(actual, expected)
    for index, (actual, expected) in enumerate(zip(actual_gradients, expected_gradients)):
        _check(actual, expected, "query" if index == 4 else "norm" if index == 5 else "value")
    if fraction == 1.0:
        local_outputs, local_gradients = run(False, projected_count=0)
        for actual, local in zip(actual_outputs, local_outputs):
            _check(actual, local)
        for index, (actual, local) in enumerate(zip(actual_gradients, local_gradients)):
            _check(actual, local, "query" if index == 4 else "norm" if index == 5 else "value")
    assert torch.count_nonzero(actual_gradients[-2][1]) > 0
    assert torch.count_nonzero(actual_gradients[-1][1]) > 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_paired_detached_history_trains_query_only(device, dtype):
    """Detached MTP history has no value VJP but still trains its projected query."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA/Triton")
    torch.manual_seed(725)
    source = torch.randn(7, 1024, device=device, dtype=dtype, requires_grad=True)
    partial = torch.randn_like(source, requires_grad=True)
    query = (torch.randn(1, 1024, device=device) * 0.02).requires_grad_()
    _, scores = project_source(source.detach(), query)
    output = aggregate_preprojected(
        [source.detach(), partial], query[0], [scores[..., 0], None], precomputed_value_grad=True
    )
    output.backward(torch.randn_like(output))
    assert source.grad is None
    assert torch.count_nonzero(query.grad) > 0
    assert torch.count_nonzero(partial.grad) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_detached_projection_skips_value_backward_kernel(monkeypatch):
    """The producer needs only dW when the consumer owns the source-value path."""

    class UnexpectedValueBackward:
        def __getitem__(self, grid):
            raise AssertionError("Detached source must not launch its value-backward kernel")

    monkeypatch.setattr(_MODULE, "_source_backward_value", UnexpectedValueBackward())
    value = torch.randn(7, 1024, device="cuda", dtype=torch.bfloat16)
    query = (torch.randn(3, 1024, device="cuda") * 0.02).requires_grad_()
    ref_query = _leaf(query)
    _, logits = project_source(value, query)
    _, reference = project_source_reference(value, ref_query)
    upstream = torch.randn_like(logits)
    logits.backward(upstream)
    reference.backward(upstream)
    _check(query.grad, ref_query.grad, "query")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("projected_count", [1, 2, 3])
def test_projection_placement_preserves_probabilities(dtype, projected_count):
    """Softmax observes the same FP32 score boundary for local and cached scores."""
    generator = torch.Generator(device="cpu").manual_seed(271828)
    initial = [torch.randn(128, 1024, generator=generator).to(dtype).cuda() for _ in range(3)]
    initial_query = (torch.randn(1024, generator=generator) * 0.02).cuda()
    upstream = torch.randn(128, 1024, generator=generator).to(dtype).cuda()

    def run(count):
        values = [_leaf(value) for value in initial]
        query = _leaf(initial_query)
        scores = [
            project_source(value.detach(), query[None])[1][..., 0] if index < count else None
            for index, value in enumerate(values)
        ]
        output = aggregate_preprojected(values, query, scores, precomputed_value_grad=True)
        # Equal stored logits alone do not cover contraction into softmax:
        # inspect the actual saved probabilities used by the custom backward.
        saved = output.grad_fn.saved_tensors
        probabilities, _, logits = saved[len(values) + 1 : len(values) + 4]
        gradients = torch.autograd.grad(output, [*values, query], upstream)
        return output, probabilities, logits, gradients

    local, cached = run(0), run(projected_count)
    for actual, expected in zip(cached[:3], local[:3]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual, expected in zip(cached[3][:-1], local[3][:-1]):
        _check(actual, expected)
    # Source-owned query reductions have a different summation order.
    _check(cached[3][-1], local[3][-1], "query")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("projected", [False, True])
def test_one_source_identity_and_explicit_zero_gradients(dtype, projected):
    value = torch.randn(3, 2, 1024, dtype=dtype, requires_grad=True)
    bank = torch.randn(2, 1024, requires_grad=True)
    query = torch.randn(1024, requires_grad=True)
    if projected:
        proxy, scores = project_source(value, bank)
        output = aggregate_preprojected([proxy], query, [scores[..., 1]])
    else:
        output = aggregate_preprojected([value], query)
    assert output.data_ptr() == value.data_ptr()
    torch.testing.assert_close(output, value, atol=0, rtol=0)
    gradient = torch.randn_like(value)
    output.backward(gradient)
    torch.testing.assert_close(value.grad, gradient, atol=0, rtol=0)
    assert torch.count_nonzero(query.grad) == 0
    if projected:
        assert bank.grad is not None and torch.count_nonzero(bank.grad) == 0


def _composed_case(device, dtype, hidden, source_count, fraction, alias, backend, tokens=7):
    torch.manual_seed(22)
    sources = [
        torch.randn(tokens, hidden, device=device, dtype=dtype, requires_grad=True)
        for _ in range(source_count)
    ]
    query = (torch.randn(3, hidden, device=device) * 0.02).requires_grad_()
    norm = (1 + torch.randn(3, hidden, device=device) * 0.1).requires_grad_()
    reference_sources = [_leaf(source) for source in sources]
    reference_query, reference_norm = _leaf(query), _leaf(norm)
    stop_mask = torch.tensor([True, False, True], device=device)
    effective = query * norm
    reference_effective = reference_query * reference_norm
    projected_count = math_ceil_fraction(fraction, source_count - 1)
    values, reference_values, scores, reference_scores = [], [], [], []
    for index, (source, reference_source) in enumerate(zip(sources, reference_sources)):
        if index < projected_count:
            value, score = project_source(source, effective, stop_mask, backend=backend)
            ref_value, ref_score = project_source_reference(
                reference_source, reference_effective, stop_mask
            )
        else:
            value, score = source, None
            ref_value, ref_score = reference_source, None
        values.append(value)
        reference_values.append(ref_value)
        scores.append(score)
        reference_scores.append(ref_score)
    if alias and source_count > 1:
        values[-1] = values[0]
        reference_values[-1] = reference_values[0]
        scores[-1], reference_scores[-1] = scores[0], reference_scores[0]
    outputs, expected_outputs = [], []
    for row in (0, 2):
        outputs.append(
            aggregate_preprojected(
                values,
                effective[row],
                [score[..., row] if score is not None else None for score in scores],
                backend=backend,
            )
        )
        expected_outputs.append(
            aggregate_preprojected_reference(
                reference_values,
                reference_effective[row],
                [score[..., row] if score is not None else None for score in reference_scores],
            )
        )
    for actual, expected in zip(outputs, expected_outputs):
        _check(actual, expected)
    upstream = [torch.randn_like(output) for output in outputs]
    torch.autograd.backward(outputs, upstream)
    torch.autograd.backward(expected_outputs, upstream)
    _check(query.grad, reference_query.grad, "query")
    _check(norm.grad, reference_norm.grad, "norm")
    for actual, expected in zip(sources, reference_sources):
        if expected.grad is None:
            assert actual.grad is None
        else:
            _check(actual.grad, expected.grad)


def math_ceil_fraction(fraction, count):
    # Fixed test fractions avoid reproducing production source-selection logic.
    return int(torch.ceil(torch.tensor(fraction * count)).item())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fraction", [0, 0.5, 1])
@pytest.mark.parametrize("alias", [False, True])
def test_complete_source_consumer_graph(dtype, fraction, alias):
    _composed_case("cpu", dtype, 1024, 3, fraction, alias, "torch")


def test_zero_query_trains_and_identical_sources_have_zero_score_derivative():
    torch.manual_seed(30)
    values = [torch.randn(3, 1024, requires_grad=True) for _ in range(3)]
    query = torch.zeros(1024, requires_grad=True)
    output = aggregate_preprojected(values, query)
    output.backward(torch.randn_like(output))
    assert query.grad.abs().sum() > 0
    bank = (torch.randn(2, 1024) * 0.02).requires_grad_()
    proxy, scores = project_source(values[0], bank)
    local = torch.randn(1024, requires_grad=True)
    output = aggregate_preprojected([proxy] * 3, local, [scores[:, 0]] * 3)
    output.sum().backward()
    assert torch.count_nonzero(bank.grad) == 0
    assert torch.count_nonzero(local.grad) == 0


def test_zero_query_bank_trains_through_multiple_changing_partials():
    torch.manual_seed(55)
    source = torch.randn(5, 1024, requires_grad=True)
    partials = [torch.randn(5, 1024, requires_grad=True) for _ in range(2)]
    queries = torch.zeros(2, 1024, requires_grad=True)
    ref_source, ref_queries = _leaf(source), _leaf(queries)
    ref_partials = [_leaf(partial) for partial in partials]
    value, scores = project_source(source, queries)
    ref_value, ref_scores = project_source_reference(ref_source, ref_queries)
    outputs, references = [], []
    for row in range(2):
        outputs.append(
            aggregate_preprojected([value, partials[row]], queries[row], [scores[:, row], None])
        )
        references.append(
            aggregate_preprojected_reference(
                [ref_value, ref_partials[row]], ref_queries[row], [ref_scores[:, row], None]
            )
        )
    upstream = [torch.randn_like(output) for output in outputs]
    torch.autograd.backward(outputs, upstream)
    torch.autograd.backward(references, upstream)
    assert queries.grad.abs().sum() > 0
    _check(queries.grad, ref_queries.grad, "query")
    _check(source.grad, ref_source.grad)
    for partial, reference in zip(partials, ref_partials):
        _check(partial.grad, reference.grad)


def test_noncontiguous_inputs_and_output_gradient():
    torch.manual_seed(12)
    value = torch.randn(2, 1024, 3, requires_grad=True)
    query = torch.randn(5, 1024, requires_grad=True)
    ref_value, ref_query = _leaf(value), _leaf(query)
    _, logits = project_source(value.transpose(1, 2), query)
    _, expected = project_source_reference(ref_value.transpose(1, 2), ref_query)
    grad = torch.randn(2, 5, 3).transpose(1, 2)
    logits.backward(grad)
    expected.backward(grad)
    _check(value.grad, ref_value.grad)
    _check(query.grad, ref_query.grad, "query")


def test_invalid_input_contracts():
    value, query = torch.ones(2, 32), torch.ones(3, 32)
    with pytest.raises(TypeError, match="FP32"):
        project_source(value, query.bfloat16())
    with pytest.raises(ValueError, match="Boolean"):
        project_source(value, query, torch.ones(3))
    with pytest.raises(ValueError, match="epsilon"):
        project_source(value, query, eps=0)
    with pytest.raises(ValueError, match="source"):
        aggregate_preprojected([], query[0])
    with pytest.raises(ValueError, match="token shape"):
        aggregate_preprojected([value], query[0], [torch.ones(3)])
    with pytest.raises(RuntimeError, match="CUDA and Triton"):
        project_source(value, query, backend="triton")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden", [1024, 7168])
@pytest.mark.parametrize("source_count", [1, 3, 9])
def test_cuda_complete_source_consumer_graph(dtype, hidden, source_count):
    _composed_case("cuda", dtype, hidden, source_count, 0.75, False, "triton", tokens=257)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden", [33, 1024, 7168])
def test_cuda_source_mask_and_statistics(dtype, hidden):
    torch.manual_seed(84)
    value = torch.randn(257, hidden, device="cuda", dtype=dtype, requires_grad=True)
    query = (torch.randn(5, hidden, device="cuda") * 0.02).requires_grad_()
    ref_value, ref_query = _leaf(value), _leaf(query)
    mask = torch.tensor([False, True, False, True, True], device="cuda")
    proxy, logits = project_source(value, query, mask)
    ref_proxy, ref_logits = project_source_reference(ref_value, ref_query, mask)
    _check(logits, ref_logits, "state")
    direct, upstream = torch.randn_like(proxy), torch.randn_like(logits)
    actual = torch.autograd.grad((proxy, logits), (value, query), (direct, upstream))
    expected = torch.autograd.grad(
        (ref_proxy, ref_logits), (ref_value, ref_query), (direct, upstream)
    )
    _check(actual[0], expected[0])
    _check(actual[1], expected[1], "query")
    assert actual[1][~mask].abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require CUDA")
def test_cuda_graph_capture_and_replay():
    torch.manual_seed(15)
    values = [
        torch.randn(5, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    query = (torch.randn(2, 1024, device="cuda") * 0.02).requires_grad_()
    gradient = torch.randn_like(values[0])

    def run():
        projected = [project_source(value, query) for value in values]
        output = aggregate_preprojected(
            [item[0] for item in projected], query[0], [item[1][:, 0] for item in projected]
        )
        grads = torch.autograd.grad(output, (*values, query), gradient)
        return output, grads

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        eager_output, eager_grads = run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output, grads = run()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, eager_output, atol=0, rtol=0)
        for actual, expected in zip(grads, eager_grads):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
