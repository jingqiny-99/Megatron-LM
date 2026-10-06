# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Standalone source-projection operator qualification and CUDA microbenchmark.

Examples (run from the Megatron-LM checkout)::

    python tests/functional_tests/python_test_utils/attn_res_source_projection_benchmark.py \
        --output /tmp/attnres-quick.json
    python tests/functional_tests/python_test_utils/attn_res_source_projection_benchmark.py \
        --mode full --tokens 256 --output /tmp/attnres-full.json
    python tests/functional_tests/python_test_utils/attn_res_source_projection_benchmark.py \
        --hidden 1024 --sources 3 --queries 8 --fractions 0 .5 1 --dtype float32 \
        --device cpu --qualify-only --output /tmp/attnres-cpu-check.json

A workload has S-1 immutable sources and one changing partial for each of Q
consumers. Each immutable source projects the final ceil(fraction*Q) query rows.
Producer projections read detached values and train their queries; consumers
compute the complete direct-plus-score value VJP before the activation-dtype
cast. The original immutable value graph stays connected to every consumer.
The optional last MTP-like consumer detaches immutable values while training its
query; its changing partial stays live. S=1 is the exact-identity edge case.

Qualification uses independent ordinary-autograd FP64 RMS/softmax equations,
with the real FP32 logits and activation-dtype interfaces. Every input gradient,
projected score and score gradient is checked at the unchanged design thresholds.
Timing compares the production project+aggregate F+B graph to ordinary PyTorch
FP32 consumer-owned RMS/softmax F+B, NOT FLA or a full training baseline. CUDA
compiler warmup is excluded. No distributed communication, offload, optimizer,
or model layers are represented. CPU mode is qualification-only and exercises
the production Torch custom-autograd backend, not Triton.
"""

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import platform
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch


@dataclass(frozen=True)
class Case:
    hidden: int
    sources: int
    queries: int
    fraction: float
    dtype: str
    stop_last_consumer: bool

    @property
    def selected_count(self):
        return math.ceil(self.fraction * self.queries)

    @property
    def selected_columns(self):
        return tuple(range(self.queries - self.selected_count, self.queries))


@dataclass
class Inputs:
    sources: list
    partials: list
    query: torch.Tensor
    norm: torch.Tensor
    upstream: list

    @property
    def leaves(self):
        return [*self.sources, *self.partials, self.query, self.norm]

    @property
    def names(self):
        return [
            *(f'source_{index}' for index in range(len(self.sources))),
            *(f'partial_{index}' for index in range(len(self.partials))),
            'query',
            'norm',
        ]


def _load_kernels():
    root = Path(__file__).resolve().parents[3]
    path = root / 'megatron/core/transformer/attention_residual_projection_kernels.py'
    spec = importlib.util.spec_from_file_location('_attnres_benchmark_kernels', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def _inputs(case, tokens, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    dtype = getattr(torch, case.dtype)

    def value():
        return torch.randn(
            tokens, case.hidden, device=device, dtype=dtype, generator=generator
        ).requires_grad_()

    return Inputs(
        sources=[value() for _ in range(case.sources - 1)],
        partials=[value() for _ in range(case.queries)],
        query=(
            torch.randn(case.queries, case.hidden, device=device, generator=generator) * 0.02
        ).requires_grad_(),
        norm=(
            1 + torch.randn(case.queries, case.hidden, device=device, generator=generator) * 0.1
        ).requires_grad_(),
        upstream=[
            torch.randn(tokens, case.hidden, device=device, dtype=dtype, generator=generator)
            for _ in range(case.queries)
        ],
    )


def _clone_inputs(inputs):
    def leaf(tensor):
        return tensor.detach().clone().requires_grad_()

    return Inputs(
        [leaf(value) for value in inputs.sources],
        [leaf(value) for value in inputs.partials],
        leaf(inputs.query),
        leaf(inputs.norm),
        inputs.upstream,
    )


def _production_graph(inputs, case, kernels, eps, backend, *, capture=None):
    effective = inputs.query * inputs.norm
    columns = case.selected_columns
    query_bank = effective[list(columns)]
    values, scores = [], []
    for source in inputs.sources:
        if columns:
            _, score = kernels.project_source(source.detach(), query_bank, eps=eps, backend=backend)
        else:
            score = None
        values.append(source)
        scores.append(score)
    if capture is not None:
        capture.update(
            scores=[score for score in scores if score is not None],
            query_bank=query_bank,
            producer_source_values_detached=True,
        )
    outputs = []
    for column, partial in enumerate(inputs.partials):
        detached = case.stop_last_consumer and column == case.queries - 1
        consumer_values = [value.detach() if detached else value for value in values] + [partial]
        row = column - (case.queries - len(columns))
        logits = [score[:, row] if score is not None and row >= 0 else None for score in scores]
        outputs.append(
            kernels.aggregate_preprojected(
                consumer_values,
                effective[column],
                [*logits, None],
                eps=eps,
                backend=backend,
                precomputed_value_grad=True,
            )
        )
    return outputs, [score for score in scores if score is not None]


def _reference_graph(inputs, case, eps, *, projected, precision):
    """Independent equations only: no production reference or backward imports."""
    effective = inputs.query * inputs.norm
    columns = case.selected_columns if projected else ()
    score_banks = []
    for source in inputs.sources:
        if columns:
            # Producer scores own only query gradients. In particular, detached
            # MTP consumers must still train their query through this bank.
            wide = source.detach().to(precision)
            normalized = wide * (wide.square().mean(-1, keepdim=True) + eps).rsqrt()
            rows = [(normalized * effective[column].to(precision)).sum(-1) for column in columns]
            score_banks.append(torch.stack(rows, dim=-1).float())
        else:
            score_banks.append(None)
    outputs = []
    for column, partial in enumerate(inputs.partials):
        detached = case.stop_last_consumer and column == case.queries - 1
        values = [value.detach() if detached else value for value in inputs.sources] + [partial]
        values = [value.to(precision) for value in values]
        query = effective[column].to(precision)
        scores = []
        for source_index, value in enumerate(values):
            bank = score_banks[source_index] if source_index < len(score_banks) else None
            if bank is not None and column in columns:
                score = bank[:, columns.index(column)].to(precision)
                # Preserve the producer's FP32 score in forward while adding
                # only its value derivative at this consumer. Sharing `value`
                # with the weighted sum combines both paths before one cast
                # into the original source's activation-dtype gradient edge.
                value_score = (value * query.detach()).sum(-1) * (
                    value.square().mean(-1) + eps
                ).rsqrt()
                score = score + (value_score - value_score.detach())
            else:
                score = (value * query).sum(-1) * (value.square().mean(-1) + eps).rsqrt()
            scores.append(score)
        probabilities = torch.stack(scores).softmax(dim=0)
        # Centering makes identity and identical-source derivatives exactly zero.
        anchor = values[0]
        mixed = anchor + sum(
            probabilities[index, :, None] * (value - anchor) for index, value in enumerate(values)
        )
        outputs.append((mixed + query.sum() * 0).to(partial.dtype))
    return outputs, [score for score in score_banks if score is not None]


def _gradients(outputs, inputs, scores=()):
    targets = [*inputs.leaves, *scores]
    gradients = torch.autograd.grad(outputs, targets, inputs.upstream, allow_unused=True)
    return [
        torch.zeros_like(tensor) if gradient is None else gradient
        for tensor, gradient in zip(targets, gradients)
    ]


def _thresholds(kind, dtype):
    if kind == 'query':
        return 1e-3, 5e-4, 1e-4
    if kind == 'norm':
        return 5e-5, 5e-4, 1e-4
    if kind == 'state' or dtype == torch.float32:
        return 2e-5, 2e-4, 2e-5
    return 0.016, 0.01, 0.003


def _error(actual, expected, kind):
    atol, rtol, l2_limit = _thresholds(kind, actual.dtype)
    actual, expected = actual.double(), expected.double()
    difference = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    reference_norm = expected.norm().item()
    relative_l2 = difference.norm().item() / max(reference_norm, 1e-30)
    exact_zero = bool(torch.count_nonzero(expected) == 0)
    zeros_preserved = not exact_zero or bool(torch.count_nonzero(actual) == 0)
    close = bool((difference <= atol + rtol * expected.abs()).all())
    return {
        'passed': finite and close and relative_l2 < l2_limit and zeros_preserved,
        'finite': finite,
        # JSON has no NaN/Infinity literals; finite=false retains the failure.
        'max_abs': (difference.max().item() if difference.numel() else 0.0) if finite else None,
        'relative_l2': relative_l2 if math.isfinite(relative_l2) else None,
        'reference_norm': reference_norm if math.isfinite(reference_norm) else None,
        'exact_zero_reference': exact_zero,
        'zeros_preserved': zeros_preserved,
        'atol': atol,
        'rtol': rtol,
        'relative_l2_limit': l2_limit,
    }


def _qualify(case, tokens, device, kernels, eps, seed, *, inputs=None):
    actual_inputs = inputs if inputs is not None else _inputs(case, tokens, device, seed)
    reference_inputs = _clone_inputs(actual_inputs)
    backend = 'triton' if device.type == 'cuda' else 'torch'
    outputs, scores = _production_graph(actual_inputs, case, kernels, eps, backend)
    expected, reference_scores = _reference_graph(
        reference_inputs, case, eps, projected=True, precision=torch.float64
    )
    actual_gradients = _gradients(outputs, actual_inputs, scores)
    expected_gradients = _gradients(expected, reference_inputs, reference_scores)
    errors = {}
    for index, (actual, reference) in enumerate(zip(outputs, expected)):
        errors[f'output_{index}'] = _error(actual, reference, 'value')
    for index, (actual, reference) in enumerate(zip(scores, reference_scores)):
        errors[f'scores_{index}'] = _error(actual, reference, 'state')
    names = actual_inputs.names + [f'scores_{index}' for index in range(len(scores))]
    for name, actual, reference in zip(names, actual_gradients, expected_gradients):
        kind = (
            name
            if name in ('query', 'norm')
            else ('state' if name.startswith('scores_') else 'value')
        )
        errors['grad_' + name] = _error(actual, reference, kind)
    return {
        'passed': all(record['passed'] for record in errors.values()),
        'tokens': tokens,
        'oracle': 'independent_fp64_autograd_with_activation_and_fp32_score_interfaces',
        'backend': backend,
        'errors': errors,
        'failed_checks': [name for name, record in errors.items() if not record['passed']],
    }


def _rounding_diagnostics(case, tokens, device, kernels, eps, seed):
    """Diagnose failed gates without changing their status or objective scale.

    Float32 inputs preserve the original quantized BF16 values. The ideal
    FP64 graph casts only the final accumulated activation gradient; it removes
    the per-consumer BF16 autograd edges, so it is a diagnostic, not an oracle
    for a different production interface. For every placement fraction, inspect
    each complete consumer VJP and forward/reverse BF16 accumulation orders.
    Detached producer projections are checked only for their query-bank VJP.
    """
    inputs = _inputs(case, tokens, device, seed)

    def convert(precision):
        def value(tensor):
            return tensor.detach().to(precision).clone().requires_grad_()

        return Inputs(
            [value(tensor) for tensor in inputs.sources],
            [value(tensor) for tensor in inputs.partials],
            inputs.query.detach().clone().requires_grad_(),
            inputs.norm.detach().clone().requires_grad_(),
            [tensor.to(precision) for tensor in inputs.upstream],
        )

    result = {
        'affects_qualification_status': False,
        'objective_rescaled': False,
        'backward_contract': 'consumer_complete_dv_detached_producer_dw',
        'fp32_inputs_preserving_bf16_values': _qualify(
            replace(case, dtype='float32'),
            tokens,
            device,
            kernels,
            eps,
            seed,
            inputs=convert(torch.float32),
        ),
    }
    backend = 'triton' if device.type == 'cuda' else 'torch'
    capture = {}
    outputs, _ = _production_graph(inputs, case, kernels, eps, backend, capture=capture)
    intermediates = (
        [*capture['scores'], capture['query_bank']]
        if case.selected_count and inputs.sources
        else []
    )
    gradients = _gradients(outputs, inputs, intermediates)
    actual_gradients = gradients[: len(inputs.leaves)]
    ideal_inputs = convert(torch.float64)
    ideal_outputs, _ = _reference_graph(
        ideal_inputs, case, eps, projected=False, precision=torch.float64
    )
    ideal_gradients = _gradients(ideal_outputs, ideal_inputs)
    result['production_vs_ideal_single_cast_source_gradients'] = {
        f'source_{index}': _error(actual, ideal.to(actual.dtype), 'value')
        for index, (actual, ideal) in enumerate(
            zip(actual_gradients[: len(inputs.sources)], ideal_gradients)
        )
    }
    if case.selected_count and inputs.sources:
        count = len(inputs.sources)
        score_gradients = gradients[len(inputs.leaves) : len(inputs.leaves) + count]
        bank_gradient = gradients[-1]
        producer_errors = {}
        expected_bank_gradients = []
        producer_value_gradients_absent = []
        for index, (source, score_gradient) in enumerate(zip(inputs.sources, score_gradients)):
            # Isolate producer dW using identical incoming dscore. No producer
            # dV exists in this paired graph; every source dV comes from a
            # complete consumer VJP and its activation-dtype accumulation.
            query = capture['query_bank'].detach().clone().requires_grad_()
            _, score = kernels.project_source(source.detach(), query, eps=eps, backend=backend)
            actual, value_gradient = torch.autograd.grad(
                score, (query, source), score_gradient, allow_unused=True
            )
            producer_value_gradients_absent.append(value_gradient is None)
            reference_query = capture['query_bank'].detach().clone().requires_grad_()
            wide = source.detach().double()
            normalized = wide * (wide.square().mean(-1, keepdim=True) + eps).rsqrt()
            reference_score = (
                (normalized[:, None, :] * reference_query.double()[None, :, :]).sum(-1).float()
            )
            (expected,) = torch.autograd.grad(reference_score, reference_query, score_gradient)
            expected_bank_gradients.append(expected)
            producer_errors[f'source_{index}'] = _error(actual, expected, 'state')
        result['producer_query_vjp_with_identical_incoming_scores'] = producer_errors
        result['detached_producer_value_gradients_absent'] = all(producer_value_gradients_absent)
        result['combined_producer_query_bank_gradient'] = _error(
            bank_gradient, torch.stack(expected_bank_gradients).sum(0), 'state'
        )
    if not inputs.sources:
        return result

    # Every source is now the sum of complete consumer VJPs, at every fraction.
    # Computing those VJPs independently separates kernel arithmetic from the
    # graph engine's BF16 accumulation order and its amplification of one ULP.
    reference_inputs = _clone_inputs(inputs)
    outputs, _ = _production_graph(inputs, case, kernels, eps, backend)
    expected, _ = _reference_graph(
        reference_inputs, case, eps, projected=True, precision=torch.float64
    )
    actual_edges, reference_edges = [], []
    for column in range(case.queries):
        gradients = torch.autograd.grad(
            outputs[column],
            inputs.sources,
            inputs.upstream[column],
            retain_graph=True,
            allow_unused=True,
        )
        references = torch.autograd.grad(
            expected[column],
            reference_inputs.sources,
            inputs.upstream[column],
            retain_graph=True,
            allow_unused=True,
        )
        actual_edges.append(
            [
                torch.zeros_like(value) if gradient is None else gradient
                for value, gradient in zip(inputs.sources, gradients)
            ]
        )
        reference_edges.append(
            [
                torch.zeros_like(value) if gradient is None else gradient
                for value, gradient in zip(reference_inputs.sources, references)
            ]
        )
    edge_errors = {
        f'consumer_{column}_source_{source}': _error(actual, reference, 'value')
        for column, (actuals, references) in enumerate(zip(actual_edges, reference_edges))
        for source, (actual, reference) in enumerate(zip(actuals, references))
    }
    result['consumer_edge_source_gradient_checks'] = edge_errors
    result['all_consumer_edge_source_gradients_pass'] = all(
        record['passed'] for record in edge_errors.values()
    )
    result['source_accumulation'] = {}
    for source, actual_full in enumerate(actual_gradients[: len(inputs.sources)]):
        actual = [row[source] for row in actual_edges]
        reference = [row[source] for row in reference_edges]
        wide_actual = torch.stack(actual).double().sum(0).to(actual_full.dtype)
        wide_reference = torch.stack(reference).double().sum(0).to(actual_full.dtype)
        details = {
            'single_cast_sum_of_edges_comparison': _error(wide_actual, wide_reference, 'value')
        }
        for order, edges in [('forward', actual), ('reverse', actual[::-1])]:
            accumulated = torch.zeros_like(actual_full)
            for gradient in edges:
                accumulated = accumulated + gradient
            details[f'production_matches_{order}_bf16_fold_exactly'] = torch.equal(
                actual_full, accumulated
            )
            details[f'production_vs_{order}_bf16_fold'] = _error(actual_full, accumulated, 'value')
        result['source_accumulation'][f'source_{source}'] = details
    return result


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _time_graph(build, inputs, device, warmup, iterations, repetitions):
    def step():
        outputs, _ = build()
        return _gradients(outputs, inputs)

    # Isolate allocator telemetry while retaining compiled-kernel caches.
    gc.collect()
    torch.cuda.empty_cache()
    for _ in range(warmup):
        step()
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            step()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return {
        'fwd_bwd_ms_median': statistics.median(samples),
        'fwd_bwd_ms_p05': _percentile(samples, 0.05),
        'fwd_bwd_ms_p95': _percentile(samples, 0.95),
        'repetition_ms': samples,
        'warmup_iterations': warmup,
        'measured_iterations_per_repetition': iterations,
        'repetitions': repetitions,
        'memory': {
            'allocated_before_bytes': baseline,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'incremental_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) - baseline,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        },
    }


def _benchmark(case, args, device, kernels):
    inputs = _inputs(case, args.tokens, device, args.seed)
    source = _time_graph(
        lambda: _production_graph(inputs, case, kernels, args.eps, 'triton'),
        inputs,
        device,
        args.warmup,
        args.iterations,
        args.repetitions,
    )
    reference = _time_graph(
        lambda: _reference_graph(inputs, case, args.eps, projected=False, precision=torch.float32),
        inputs,
        device,
        args.warmup,
        args.iterations,
        args.repetitions,
    )
    return {
        'tokens': args.tokens,
        'production_project_aggregate': source,
        'torch_consumer_owned_fp32_autograd': reference,
        'reference_over_production_time': (
            reference['fwd_bwd_ms_median'] / source['fwd_bwd_ms_median']
        ),
        'timing_method': 'CUDA events around eager Python forward plus autograd.grad; includes launch gaps',
        'reference_is_fla': False,
    }


def _cases(args):
    overridden = any(
        option is not None for option in (args.hidden, args.sources, args.queries, args.fractions)
    )
    if args.mode == 'quick' and not overridden:
        shapes = [
            (1024, 1, 8, 1.0),
            (1024, 3, 8, 0.0),
            (1024, 3, 8, 0.5),
            (1024, 3, 8, 1.0),
            (7168, 9, 8, 0.0),
            (7168, 9, 8, 1.0),
        ]
    else:
        shapes = itertools.product(
            args.hidden or [1024, 7168],
            args.sources or [1, 3, 9],
            args.queries or ([8] if args.mode == 'quick' else [8, 32, 64]),
            args.fractions or [0, 0.25, 0.5, 0.75, 1],
        )
    return [
        Case(hidden, sources, queries, fraction, args.dtype, args.stop_last_consumer)
        for hidden, sources, queries, fraction in shapes
    ]


def _arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('quick', 'full'), default='quick')
    parser.add_argument('--hidden', type=int, nargs='+')
    parser.add_argument('--sources', type=int, nargs='+')
    parser.add_argument('--queries', type=int, nargs='+')
    parser.add_argument('--fractions', type=float, nargs='+')
    parser.add_argument('--dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument(
        '--qualify-tokens',
        type=int,
        default=None,
        help='Independent FP64 qualification token count; defaults to the timing --tokens value.',
    )
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260923)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--stop-last-consumer', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--qualify-only', action='store_true')
    parser.add_argument(
        '--diagnose-rounding',
        action='store_true',
        help='Report FP32/ideal/edge diagnostics for failed BF16 cases; gates stay unchanged.',
    )
    args = parser.parse_args(argv)
    if args.qualify_tokens is None:
        args.qualify_tokens = args.tokens
    for name in ('tokens', 'qualify_tokens', 'warmup', 'iterations', 'repetitions'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if any(
        value < 1
        for values in (args.hidden, args.sources, args.queries)
        if values
        for value in values
    ):
        parser.error('hidden dimensions, source counts and query counts must be positive')
    if args.fractions and any(
        not math.isfinite(value) or not 0 <= value <= 1 for value in args.fractions
    ):
        parser.error('fractions must be finite and in [0, 1]')
    if not math.isfinite(args.eps) or args.eps <= 0:
        parser.error('--eps must be positive and finite')
    if torch.device(args.device).type != 'cuda' and not args.qualify_only:
        parser.error('non-CUDA devices require --qualify-only')
    return args


def _environment(device, kernels, path):
    result = {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'triton': getattr(kernels.triton, '__version__', None),
        'device': str(device),
        'kernel_path': str(path),
        'kernel_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32,
    }
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        result.update(
            gpu_name=properties.name,
            compute_capability=list(torch.cuda.get_device_capability(device)),
            device_total_memory_bytes=properties.total_memory,
            cuda_device_count=torch.cuda.device_count(),
        )
    return result


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep all completed qualification failures/results if a later workload fails.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main(argv=None):
    args = _arguments(argv)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    kernels, path = _load_kernels()
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA qualification requires a visible CUDA device')
        if kernels.triton is None:
            raise RuntimeError('Production CUDA qualification requires Triton')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
    cases = _cases(args)
    report = {
        'schema_version': 1,
        'experiment': 'standalone_attnres_source_projection_operator',
        'environment': _environment(device, kernels, path),
        'arguments': {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        'scope': {
            'source_count_includes_changing_partial': True,
            'fractions_select_latest_consumer_columns': True,
            'backward_contract': 'consumer_complete_dv_detached_producer_dw',
            'public_primitive_defaults_changed': False,
            'full_training_speed_claim': False,
            'distributed_communication_included': False,
            'oracle_precision': 'float64',
            'timed_reference_precision': 'float32',
            'qualification_and_timing_token_counts_may_differ': args.qualify_tokens != args.tokens,
        },
        'results': [],
    }
    _write_report(args.output, report)
    failed = False
    for index, case in enumerate(cases):
        result = {'case': asdict(case), 'selected_query_rows': case.selected_count}
        started = time.time()
        try:
            qualification = _qualify(
                case, args.qualify_tokens, device, kernels, args.eps, args.seed
            )
            result['qualification'] = qualification
            if not qualification['passed']:
                result['status'] = 'qualification_failed'
                failed = True
                if args.diagnose_rounding and case.dtype == 'bfloat16':
                    result['rounding_diagnostics'] = _rounding_diagnostics(
                        case, args.qualify_tokens, device, kernels, args.eps, args.seed
                    )
            elif args.qualify_only:
                result['status'] = 'qualified_only'
            else:
                result['benchmark'] = _benchmark(case, args, device, kernels)
                result['status'] = 'qualified_and_benchmarked'
        except Exception as error:
            failed = True
            result['status'] = 'error'
            result['error'] = {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc(),
            }
        result['elapsed_seconds'] = time.time() - started
        report['results'].append(result)
        report['passed'] = not failed
        _write_report(args.output, report)
        print(
            f'[{index + 1}/{len(cases)}] H={case.hidden} S={case.sources} Q={case.queries} '
            f'fraction={case.fraction:g}: {result["status"]}',
            flush=True,
        )
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
