# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CUDA primitive gates against unchanged FLA and independent FP64 autograd."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("fla.ops.attnres.fused")
from fla.ops.attnres.fused import FusedAttnresFunction, _build_ptr_table, fused_attnres_fwd

_PATH = (
    Path(__file__).resolve().parents[3]
    / "megatron/core/transformer/attention_residual_forward_projection.py"
)
_SPEC = importlib.util.spec_from_file_location("attnres_forward_projection_test_module", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
project_forward_source = _MODULE.project_forward_source
aggregate_with_forward_cache = _MODULE.aggregate_with_forward_cache
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA primitive")


def _check(actual, expected, kind="value"):
    assert actual is not None and torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    if kind == "norm":
        atol, rtol, limit = 5e-5, 5e-4, 1e-4
    elif kind == "query":
        atol, rtol, limit = 1e-3, 5e-4, 1e-4
    elif kind == "state":
        atol, rtol, limit = 2e-5, 2e-4, 2e-5
    else:
        atol, rtol, limit = 0.016, 0.01, 0.003
    actual, expected = actual.double(), expected.double()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    norm = expected.norm()
    if norm == 0:
        assert torch.count_nonzero(actual) == 0
        return
    relative = (actual - expected).norm() / norm
    assert relative < limit, (kind, relative.item(), limit)
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    similarity = 1 - (actual - expected).square().sum() / (
        actual.square().sum() + expected.square().sum()
    ).clamp_min(1e-30)
    assert cosine > 0.999 and similarity > 0.999


def _inputs(sources=5, queries=1, tokens=128):
    torch.manual_seed(381)
    values = tuple(
        torch.randn(tokens, 1, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(sources)
    )
    q = (torch.randn(queries, 1024, device="cuda") * 0.02).requires_grad_()
    w = (torch.randn(queries, 1024, device="cuda") * 0.1 + 1).requires_grad_()
    return values, q, w


def _native(query, norm, values):
    # Direct original Function preserves source shape and the actual FLA backward.
    return FusedAttnresFunction.apply(query, norm, None, 1e-6, 1.0, False, 1, *values)[0]


def _reference(query, norm, values):
    """Independent RMS normalization, depth softmax, and value mixing in FP64."""
    stacked = torch.stack(values).double()
    inverse_rms = torch.rsqrt(stacked.square().mean(dim=-1) + 1e-6)
    logits = (stacked * (query.double() * norm.double())).sum(dim=-1) * inverse_rms
    probabilities = logits.softmax(dim=0)
    output = (probabilities.unsqueeze(-1) * stacked).sum(dim=0)
    return output.to(values[0].dtype), inverse_rms, logits, logits.logsumexp(dim=0)


def _caches(values, queries, norms, selected):
    return tuple(
        project_forward_source(value, queries, norms) if i in selected else None
        for i, value in enumerate(values)
    )


def _row(caches, row):
    return tuple(None if cache is None else cache[row] for cache in caches)


def _grads(output, query, norm, values, upstream):
    return torch.autograd.grad(output, (query, norm, *values), upstream)


def _check_grads(actual, reference):
    for i, (got, expected) in enumerate(zip(actual, reference)):
        _check(got, expected, "query" if i == 0 else "norm" if i == 1 else "value")


@pytest.mark.parametrize(
    "sources,selected",
    [(5, ()), (5, (0, 2)), (5, (0, 1, 2, 3)), (5, (0, 1, 2, 3, 4)), (9, (0, 1, 2, 3, 4, 5, 6, 7))],
)
def test_mixed_forward_stats_and_complete_vjp(sources, selected):
    values, queries, norms = _inputs(sources=sources)
    query, norm = queries[0], norms[0]
    caches = _caches(values, queries, norms, selected)
    output, rstd, logits, lse = aggregate_with_forward_cache(
        query, norm, values, _row(caches, 0), return_stats=True
    )
    native = _native(query, norm, values)
    reference, rr, rl, rs = _reference(query, norm, values)
    native_stats = fused_attnres_fwd(
        query, values, _build_ptr_table(values), norm, None, 1e-6, 1.0, 1
    )[2:]
    for expected in (native, reference):
        _check(output, expected)
    for got, nref, independent in zip((rstd, logits, lse), native_stats, (rr, rl, rs)):
        assert not got.requires_grad
        _check(got, nref, "state")
        _check(got, independent, "state")
    upstream = torch.randn_like(output)
    actual_grads = _grads(output, queries, norms, values, upstream)
    _check_grads(actual_grads, _grads(native, queries, norms, values, upstream))
    _check_grads(actual_grads, _grads(reference, queries, norms, values, upstream))
    info = _MODULE.forward_cache_launch_info(query, norm, values, _row(caches, 0))
    assert info["single_local_reduction"] == (len(selected) == sources - 1)


@pytest.mark.parametrize("query_count", [8, 33])
def test_multiple_consumer_fanout(query_count):
    values, queries, norms = _inputs(queries=query_count)
    caches = _caches(values, queries, norms, (0, 1, 2, 3))
    outputs, natives, references, upstream = [], [], [], []
    for row in range(query_count):
        outputs.append(
            aggregate_with_forward_cache(queries[row], norms[row], values, _row(caches, row))
        )
        natives.append(_native(queries[row], norms[row], values))
        references.append(_reference(queries[row], norms[row], values)[0])
        upstream.append(torch.randn_like(outputs[-1]))
    actual = _grads(outputs, queries, norms, values, upstream)
    _check_grads(actual, _grads(natives, queries, norms, values, upstream))
    _check_grads(actual, _grads(references, queries, norms, values, upstream))


@pytest.mark.parametrize("sources,zero_query", [(1, False), (5, True)])
def test_single_source_and_zero_query(sources, zero_query):
    values, queries, norms = _inputs(sources=sources)
    if zero_query:
        with torch.no_grad():
            queries.zero_()
    caches = _caches(values, queries, norms, tuple(range(sources)))
    output = aggregate_with_forward_cache(queries[0], norms[0], values, _row(caches, 0))
    native = _native(queries[0], norms[0], values)
    reference = _reference(queries[0], norms[0], values)[0]
    _check(output, native)
    _check(output, reference)
    upstream = torch.randn_like(output)
    actual = _grads(output, queries, norms, values, upstream)
    _check_grads(actual, _grads(native, queries, norms, values, upstream))
    _check_grads(actual, _grads(reference, queries, norms, values, upstream))
    if zero_query:
        assert torch.count_nonzero(actual[1]) == 0
    if sources == 1:
        assert torch.equal(output, values[0])
        assert torch.equal(actual[2], upstream)
        assert torch.count_nonzero(actual[0]) == torch.count_nonzero(actual[1]) == 0


@pytest.mark.parametrize(
    "mutate", ["source", "query", "norm", "dot", "rstd", "row", "eps", "source_order"]
)
def test_cache_binding_rejects_stale_and_mismatched_inputs(mutate):
    values, queries, norms = _inputs(queries=2)
    caches = _caches(values, queries, norms, (0, 1, 2, 3))
    row = list(_row(caches, 0))
    query, norm = queries[0], norms[0]
    eps = 1e-6
    tensors = {
        "source": values[0],
        "query": queries,
        "norm": norms,
        "dot": row[0].dot,
        "rstd": row[0].rstd,
    }
    if mutate in tensors:
        with torch.no_grad():
            tensors[mutate].add_(0.25)
    elif mutate == "row":
        query = queries[1]
    elif mutate == "eps":
        eps = 1e-5
    else:
        row[0], row[1] = row[1], row[0]
    with pytest.raises(ValueError, match="binding|eps"):
        aggregate_with_forward_cache(query, norm, values, row, eps=eps)


def test_producer_no_autograd_and_cache_contract():
    values, queries, norms = _inputs(queries=3)
    cache = project_forward_source(values[0], queries, norms)
    assert len(cache) == 3
    assert all(not row.dot.requires_grad and not row.rstd.requires_grad for row in cache)
    assert all(row.rstd is cache[0].rstd for row in cache)
    assert project_forward_source(values[0], queries[:0], norms[:0]) == ()
    with pytest.raises(ValueError, match="contiguous CUDA FP32"):
        project_forward_source(values[0], queries, norms[:2])
    bad = replace(cache[0], dot=cache[0].dot.clone())
    with pytest.raises(ValueError, match="binding"):
        aggregate_with_forward_cache(queries[0], norms[0], (values[0],), (bad,))


def test_capture_replays_fresh_values_and_query_bank():
    values, queries, norms = _inputs()
    upstream = torch.randn_like(values[0])

    def execute():
        caches = _caches(values, queries, norms, (0, 1, 2, 3))
        output = aggregate_with_forward_cache(queries[0], norms[0], values, _row(caches, 0))
        grads = _grads(output, queries, norms, values, upstream)
        return output, grads

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            execute()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual, actual_grads = execute()
    for step in range(3):
        with torch.no_grad():
            queries.add_(0.001 * (step + 1))
            norms.mul_(1.001)
            values[0].mul_(0.98)
        graph.replay()
        expected = _native(queries[0], norms[0], values)
        expected_grads = _grads(expected, queries, norms, values, upstream)
        _check(actual, expected)
        _check_grads(actual_grads, expected_grads)
