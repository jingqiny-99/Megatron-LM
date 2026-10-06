# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real-model pipeline training benchmark for FLA and source-owned AttnRes.

Run with torchrun from a Megatron-LM checkout::

    torchrun --standalone --nproc-per-node=4 \
        tests/functional_tests/python_test_utils/attn_res_source_pipeline_benchmark.py \
        --pp 2 --vp 2 --output /tmp/attnres-pipeline.json

Measure an unchanged offload checkout with this driver and the shared model
helper (both files may live outside that checkout)::

    torchrun --standalone --nproc-per-node=4 /feature/tests/functional_tests/\
python_test_utils/attn_res_source_pipeline_benchmark.py \
        --megatron-root /offload-base --model-helper /feature/tests/unit_tests/\
transformer/test_attention_residual_source_model.py \
        --implementations fla --pp 2 --vp 2 --output /tmp/offload-base-fla.json

The actual integration-test builder creates GPT/hybrid chunks, overlapped MCore
DDP, BF16 Adam and FP32 master parameters. Each variant starts with the same
initialization seed and follows the same fixed synthetic batches. Repetitions
continue the model/optimizer state; each has ten warmup and fifty measured
updates by default. Each measured iteration includes zeroing, the real PP/VPP
schedule, query preparation/finalization, DP gradient synchronization and the
optimizer update. Data generation, barriers before iterations, report gathering
and telemetry resolution are outside the clock. A device synchronization at the
end is included in host latency, ensuring all local streams have completed.

CUDA events around schedule forward/backward calls are elapsed stream intervals,
not kernel-active time or disjoint communication exposure. They can include
stream waits and host launch gaps. Nested phase intervals must not be added to
whole-iteration time. Instrumentation adds overhead and is the same for every
variant. Wire counters observe actual padded tensors submitted to P2P; they do
not count NCCL protocol overhead or query-bank collectives.

This is a full synthetic training-iteration experiment, not an accuracy test.
Run model and primitive qualification separately; this driver never relaxes
those thresholds. Results apply only to the recorded model, layout and hardware.
"""

import argparse
import functools
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
_CHANNELS = ('value', 'score')
_DIRECTIONS = ('forward_send', 'backward_send', 'forward_recv', 'backward_recv')


def _arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--megatron-root', type=Path, default=_ROOT)
    parser.add_argument('--model-helper', type=Path)
    parser.add_argument(
        '--implementations', choices=('fla', 'source'), nargs='+', default=['fla', 'source']
    )
    parser.add_argument('--fractions', type=float, nargs='+', default=[0, 0.25, 0.5, 0.75, 1])
    parser.add_argument('--pp', type=int, default=2)
    parser.add_argument('--vp', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--layers', type=int, default=16)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--ffn-hidden', type=int)
    parser.add_argument('--seq-length', type=int, default=256)
    parser.add_argument('--vocab-size', type=int, default=4096)
    parser.add_argument('--micro-batch-size', type=int, default=1)
    parser.add_argument('--microbatches', type=int)
    parser.add_argument('--block-layers', type=int, default=3)
    parser.add_argument('--model', choices=('gpt', 'hybrid'), default='gpt')
    parser.add_argument('--mtp', type=int, default=0)
    parser.add_argument('--detach-mtp', action='store_true')
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--selective-recompute', action='store_true')
    parser.add_argument('--overlap-p2p', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        '--attention-backend', choices=('auto', 'flash', 'fused', 'unfused'), default='unfused'
    )
    parser.add_argument('--gradient-accumulation-fusion', action='store_true')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args(argv)
    args.megatron_root = args.megatron_root.resolve()
    args.model_helper = (
        args.model_helper
        or args.megatron_root
        / 'tests/unit_tests/transformer/test_attention_residual_source_model.py'
    ).resolve()
    args.ffn_hidden = args.ffn_hidden or 2 * args.hidden
    args.microbatches = args.microbatches or max(4, 2 * args.pp)
    for name in (
        'pp',
        'vp',
        'hidden',
        'layers',
        'heads',
        'ffn_hidden',
        'seq_length',
        'vocab_size',
        'micro_batch_size',
        'microbatches',
        'block_layers',
        'warmup',
        'iterations',
        'repetitions',
    ):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.mtp < 0:
        parser.error('--mtp must be nonnegative')
    if args.layers % (args.pp * args.vp):
        parser.error('--layers must divide evenly across PP*VP chunks')
    if args.model == 'hybrid' and args.layers // (args.pp * args.vp) % 2:
        parser.error('hybrid chunks require an even number of entries for the *- pattern')
    if args.hidden % args.heads:
        parser.error('--hidden must be divisible by --heads')
    if args.vp > 1 and args.microbatches % args.pp:
        parser.error('interleaved microbatch count must be divisible by PP size')
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in args.fractions):
        parser.error('fractions must be finite and in [0, 1]')
    if len(set(args.implementations)) != len(args.implementations):
        parser.error('implementations must not be duplicated')
    if len(set(args.fractions)) != len(args.fractions):
        parser.error('fractions must not be duplicated')
    return args


def _load_fixture(args):
    """Load real model setup against the requested checkout, including baseline."""
    if not args.model_helper.is_file():
        raise FileNotFoundError(f'Model helper is missing: {args.model_helper}')
    sys.path.insert(0, str(args.megatron_root))
    spec = importlib.util.spec_from_file_location(
        '_attnres_pipeline_model_helper', args.model_helper
    )
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    spec.loader.exec_module(fixture)
    fixture._SEQ_LENGTH = args.seq_length
    fixture._HIDDEN_SIZE = args.hidden
    fixture._VOCAB_SIZE = args.vocab_size
    fixture._NUM_LAYERS = args.layers
    original_factory = fixture._config
    config_type = fixture.TransformerConfig
    field_names = {field.name for field in fields(config_type)}
    if (
        'source' in args.implementations
        and 'attn_res_source_projection_fraction' not in field_names
    ):
        raise RuntimeError('The requested checkout does not support source projection')

    def benchmark_config(case, impl):
        # Adapt arguments BEFORE __post_init__, preserving every other real test
        # setting and inheriting test-helper API fixes. This replaces a symbol
        # only in our private helper module for this synchronous factory call.
        def construct(**kwargs):
            kwargs.update(
                num_layers=args.layers,
                hidden_size=args.hidden,
                num_attention_heads=args.heads,
                num_query_groups=args.heads,
                kv_channels=args.hidden // args.heads,
                ffn_hidden_size=args.ffn_hidden,
                attn_res_block_layers=args.block_layers,
                attention_backend=getattr(fixture.AttnBackend, args.attention_backend),
                gradient_accumulation_fusion=args.gradient_accumulation_fusion,
            )
            if 'attn_res_source_projection_fraction' in field_names:
                kwargs['attn_res_source_projection_fraction'] = case.fraction
            elif impl != 'source':
                # The unchanged offload baseline does not know this source-only
                # field. Its FLA construction otherwise follows its own guards.
                kwargs.pop('attn_res_source_projection_fraction', None)
            return config_type(**kwargs)

        fixture.TransformerConfig = construct
        try:
            return original_factory(case, impl)
        finally:
            fixture.TransformerConfig = config_type

    fixture._config = benchmark_config
    return fixture


def _git_identity(root):
    def git(*arguments):
        completed = subprocess.run(
            ['git', '-C', str(root), *arguments], check=False, capture_output=True, text=True
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        'root': str(root),
        'commit': git('rev-parse', 'HEAD'),
        'status': git('status', '--short'),
    }


def _summary(values):
    values = list(values)
    if not values:
        return {'count': 0}
    ordered = sorted(values)
    return {
        'count': len(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'p05': ordered[round((len(ordered) - 1) * 0.05)],
        'p95': ordered[round((len(ordered) - 1) * 0.95)],
        'min': ordered[0],
        'max': ordered[-1],
    }


class _Telemetry:
    """Asynchronous event brackets plus Python enqueue time, resolved per step."""

    def __init__(self, source_mode):
        self.source_mode = source_mode
        self.enabled = False
        self.events = []
        self.wire = {}

    def begin(self):
        self.events = []
        self.wire = {
            f'{direction}_{channel}_bytes': 0 for direction in _DIRECTIONS for channel in _CHANNELS
        }
        self.enabled = True

    @contextmanager
    def measure(self, phase):
        if not self.enabled:
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        before = time.perf_counter()
        try:
            yield
        finally:
            host_ms = (time.perf_counter() - before) * 1000
            end.record()
            self.events.append((phase, start, end, host_ms))

    def observe(self, direction, payload):
        if not self.enabled or payload is None:
            return
        tensors = payload if isinstance(payload, (list, tuple)) else [payload]
        for index, tensor in enumerate(tensors):
            if tensor is not None:
                channel = 'score' if self.source_mode and index == 1 else 'value'
                self.wire[f'{direction}_{channel}_bytes'] += tensor.numel() * tensor.element_size()

    def resolve(self):
        self.enabled = False
        phases = defaultdict(lambda: {'calls': 0, 'host_enqueue_ms': 0.0, 'cuda_elapsed_ms': 0.0})
        for phase, start, end, host_ms in self.events:
            phases[phase]['calls'] += 1
            phases[phase]['host_enqueue_ms'] += host_ms
            phases[phase]['cuda_elapsed_ms'] += start.elapsed_time(end)
        return {'phases': dict(phases), 'wire': dict(self.wire)}


@contextmanager
def _instrument(schedules, p2p, telemetry):
    originals = []

    def wrap(module, name, phase):
        if not hasattr(module, name):
            return
        original = getattr(module, name)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            with telemetry.measure(phase):
                return original(*args, **kwargs)

        originals.append((module, name, original))
        setattr(module, name, measured)

    wrap(schedules, 'forward_step', 'forward_step')
    wrap(schedules, 'backward_step', 'backward_step')
    wrap(schedules, '_prepare_attn_res_projection', 'source_prepare')
    wrap(schedules, '_finalize_attn_res_projection', 'source_finalize')
    cls = p2p.P2PCommunicator
    original = cls._communicate
    signature = inspect.signature(original)

    @functools.wraps(original)
    def communicate(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        telemetry.observe('forward_send', bound.arguments.get('tensor_send_next'))
        telemetry.observe('backward_send', bound.arguments.get('tensor_send_prev'))
        result = original(*args, **kwargs)
        telemetry.observe('forward_recv', result[0])
        telemetry.observe('backward_recv', result[1])
        return result

    originals.append((cls, '_communicate', original))
    cls._communicate = communicate
    try:
        yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)


def _batches(args, dp_rank):
    generator = torch.Generator().manual_seed(9000 + dp_rank)
    batches = []
    for _ in range(args.microbatches):
        tokens = torch.randint(
            args.vocab_size,
            (args.micro_batch_size, args.seq_length + 1),
            generator=generator,
            device='cpu',
        ).cuda()
        batches.append((tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()))
    return batches


def _one_step(args, fixture, case, chunks, optimizer, batches, groups, telemetry, communicator):
    # Barriers are deliberately outside the synchronized iteration latency.
    torch.distributed.barrier()
    torch.cuda.synchronize()
    telemetry.begin()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    before = time.perf_counter()
    for chunk in chunks:
        chunk.zero_grad_buffer()
    optimizer.zero_grad()
    with telemetry.measure('schedule'):
        losses = fixture.get_forward_backward_func()(
            forward_step_func=fixture._forward_step,
            data_iterator=[iter(batches) for _ in range(case.vp)],
            model=chunks,
            num_microbatches=args.microbatches,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=False,
            pg_collection=groups,
            p2p_communicator=communicator,
        )
    with telemetry.measure('optimizer'):
        success, grad_norm, _ = optimizer.step()
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - before) * 1000
    record = telemetry.resolve()
    record.update(host_iteration_ms=wall_ms, cuda_iteration_elapsed_ms=start.elapsed_time(end))
    # Health checks and their collective are outside every measured interval;
    # they are not a replacement for model/gradient numerical qualification.
    norm = float(grad_norm) if grad_norm is not None else None
    loss_values = [float(item['loss']) for item in losses if 'loss' in item]
    local_valid = bool(success) and norm is not None and math.isfinite(norm)
    local_valid = local_valid and all(math.isfinite(value) for value in loss_values)
    healthy = torch.tensor(int(local_valid), dtype=torch.int32, device='cuda')
    torch.distributed.all_reduce(healthy, op=torch.distributed.ReduceOp.MIN)
    if not healthy.item():
        raise RuntimeError(
            f'Nonfinite loss/gradient norm or failed optimizer update: '
            f'{success=}, {norm=}, {loss_values=}'
        )
    record.update(losses=loss_values, grad_norm=norm)
    return record


def _rank_summary(records, rank_info, memory):
    phases = sorted({phase for record in records for phase in record['phases']})
    return {
        **rank_info,
        'iterations': records,
        'host_iteration_ms': _summary(record['host_iteration_ms'] for record in records),
        'cuda_iteration_elapsed_ms': _summary(
            record['cuda_iteration_elapsed_ms'] for record in records
        ),
        'phases': {
            phase: {
                metric: _summary(
                    record['phases'].get(phase, {}).get(metric, 0) for record in records
                )
                for metric in ('host_enqueue_ms', 'cuda_elapsed_ms', 'calls')
            }
            for phase in phases
        },
        'wire_bytes_per_iteration': {
            key: _summary(record['wire'][key] for record in records) for key in records[0]['wire']
        },
        'memory': memory,
    }


def _global_summary(ranks, tokens_per_iteration):
    count = len(ranks[0]['iterations'])
    if any(len(rank['iterations']) != count for rank in ranks):
        raise RuntimeError('Ranks reported different measured iteration counts')
    for step in range(count):
        for direction in ('forward', 'backward'):
            for channel in _CHANNELS:
                sent = sum(
                    rank['iterations'][step]['wire'][f'{direction}_send_{channel}_bytes']
                    for rank in ranks
                )
                received = sum(
                    rank['iterations'][step]['wire'][f'{direction}_recv_{channel}_bytes']
                    for rank in ranks
                )
                if sent != received:
                    raise RuntimeError(
                        f'Unbalanced P2P byte telemetry at iteration {step}: '
                        f'{direction}/{channel} sent={sent} received={received}'
                    )
    latencies = [
        max(rank['iterations'][step]['host_iteration_ms'] for rank in ranks)
        for step in range(count)
    ]
    wire = {
        key: _summary(
            sum(rank['iterations'][step]['wire'][key] for rank in ranks) for step in range(count)
        )
        for key in ranks[0]['iterations'][0]['wire']
    }
    return {
        'max_rank_host_iteration_ms': _summary(latencies),
        'tokens_per_second': _summary(tokens_per_iteration * 1000 / value for value in latencies),
        'tokens_per_iteration': tokens_per_iteration,
        'global_wire_bytes_per_iteration': wire,
        'max_rank_peak_allocated_bytes': max(
            rank['memory']['peak_allocated_bytes'] for rank in ranks
        ),
        'all_reported_losses_and_grad_norms_finite': True,
        'global_p2p_send_receive_bytes_balance': True,
    }


def _package_versions():
    versions = {}
    for name in ('triton', 'transformer-engine', 'flash-linear-attention'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _source_hashes(root):
    paths = (
        'megatron/core/transformer/attention_residual.py',
        'megatron/core/transformer/attention_residual_projection_kernels.py',
        'megatron/core/transformer/attention_residual_projection_runtime.py',
        'megatron/core/transformer/attention_residual_source_state.py',
        'megatron/core/pipeline_parallel/schedules.py',
        'megatron/core/pipeline_parallel/p2p_communication.py',
    )
    return {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in paths
        if (root / path).is_file()
    }


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _config_metadata(chunks):
    config = chunks[0].module.config
    names = (
        'num_layers',
        'hidden_size',
        'num_attention_heads',
        'num_query_groups',
        'kv_channels',
        'ffn_hidden_size',
        'pipeline_model_parallel_size',
        'virtual_pipeline_model_parallel_size',
        'attn_res_impl',
        'attn_res_block_layers',
        'attn_res_source_projection_fraction',
        'attention_backend',
        'pipeline_dtype',
        'params_dtype',
        'gradient_accumulation_fusion',
        'overlap_p2p_comm',
        'batch_p2p_comm',
        'batch_p2p_sync',
        'mtp_num_layers',
        'mtp_detach_heads',
        'fine_grained_activation_offloading',
        'offload_modules',
        'recompute_granularity',
        'recompute_modules',
    )
    result = {}
    for name in names:
        value = getattr(config, name, None)
        result[name] = (
            value if isinstance(value, (type(None), str, int, float, bool, list)) else str(value)
        )
    return result


def main(argv=None):
    args = _arguments(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires real CUDA devices and torchrun/NCCL')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    fixture = _load_fixture(args)
    if fixture.Utils.world_size % args.pp:
        raise ValueError('WORLD_SIZE must be divisible by PP (this driver uses TP1/CP1/EP1)')
    fixture.Utils.initialize_model_parallel(
        1, args.pp, virtual_pipeline_model_parallel_size=args.vp if args.vp > 1 else None
    )
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    groups = fixture.ProcessGroupCollection.use_mpu_process_groups()
    schedules = importlib.import_module('megatron.core.pipeline_parallel.schedules')
    p2p = importlib.import_module('megatron.core.pipeline_parallel.p2p_communication')
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    rank_info = {
        'rank': rank,
        'pp_rank': fixture.parallel_state.get_pipeline_model_parallel_rank(),
        'dp_rank': fixture.parallel_state.get_data_parallel_rank(),
        'hostname': platform.node(),
        'cuda_device': torch.cuda.current_device(),
        'gpu_name': properties.name,
        'gpu_memory_bytes': properties.total_memory,
        'compute_capability': list(torch.cuda.get_device_capability()),
    }
    report = {
        'schema_version': 1,
        'completed': False,
        'experiment': 'actual_attnres_pipeline_training_iteration',
        'checkout': _git_identity(args.megatron_root),
        'source_file_sha256': _source_hashes(args.megatron_root),
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_helper': {
            'path': str(args.model_helper),
            'sha256': hashlib.sha256(args.model_helper.read_bytes()).hexdigest(),
        },
        'arguments': {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        'environment': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'world_size': world,
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'packages': _package_versions(),
        },
        'methodology': {
            'real_model_ddp_optimizer_and_schedule': True,
            'numerical_qualification_included': False,
            'initialization_seed_per_variant': 1234,
            'synthetic_batch_seed_per_dp_rank': '9000 + dp_rank',
            'batches_gpu_resident_and_reused': True,
            'data_generation_and_io_timed': False,
            'optimizer_update_timed': True,
            'source_prepare_and_finalize_timed': True,
            'ddp_gradient_synchronization_timed': True,
            'pre_iteration_barrier_timed': False,
            'post_iteration_device_synchronize_in_host_time': True,
            'repetitions_continue_model_and_optimizer_state': True,
            'phase_cuda_times_are_elapsed_stream_intervals_not_active_kernel_time': True,
            'phase_intervals_are_nested_not_additive': True,
            'p2p_bytes_include_value_and_score_padding': True,
            'p2p_bytes_exclude_nccl_protocol_and_query_bank_collectives': True,
            'distributed_optimizer': False,
            'tp': 1,
            'cp': 1,
            'ep': 1,
            'dp': world // args.pp,
        },
        'variants': [],
    }
    if rank == 0:
        _write_report(args.output, report)
    variants = [
        (impl, fraction)
        for impl in args.implementations
        for fraction in (args.fractions if impl == 'source' else [0.0])
    ]
    try:
        for impl, fraction in variants:
            label = f'source_fraction_{fraction:g}' if impl == 'source' else 'fla'
            case = fixture._Case(
                label,
                args.pp,
                args.vp,
                hybrid=args.model == 'hybrid',
                mtp=args.mtp,
                detach=args.detach_mtp,
                offload=args.offload,
                recompute=args.selective_recompute,
                fraction=fraction,
                overlap_p2p=args.overlap_p2p,
            )
            chunks, optimizer = fixture._build(case, impl, groups)
            communicator = (
                p2p.P2PCommunicator(pp_group=groups.pp, config=chunks[0].config)
                if case.pp > 1
                else None
            )
            batches = _batches(args, rank_info['dp_rank'])
            telemetry = _Telemetry(impl == 'source')
            variant = {
                'label': label,
                'implementation': impl,
                'fraction': fraction if impl == 'source' else None,
                'config': _config_metadata(chunks),
                'repetitions': [],
            }
            if rank == 0:
                report['variants'].append(variant)
                _write_report(args.output, report)
            with _instrument(schedules, p2p, telemetry):
                for repetition in range(args.repetitions):
                    for _ in range(args.warmup):
                        _one_step(
                            args,
                            fixture,
                            case,
                            chunks,
                            optimizer,
                            batches,
                            groups,
                            telemetry,
                            communicator,
                        )
                    torch.cuda.synchronize()
                    gc.collect()
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    records = [
                        _one_step(
                            args,
                            fixture,
                            case,
                            chunks,
                            optimizer,
                            batches,
                            groups,
                            telemetry,
                            communicator,
                        )
                        for _ in range(args.iterations)
                    ]
                    for index, record in enumerate(records):
                        record['optimizer_update'] = (
                            repetition * (args.warmup + args.iterations) + args.warmup + index + 1
                        )
                    memory = {
                        'allocated_before_bytes': baseline,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                        'incremental_peak_allocated_bytes': torch.cuda.max_memory_allocated()
                        - baseline,
                        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                    }
                    local = _rank_summary(records, rank_info, memory)
                    gathered = [None] * world
                    torch.distributed.all_gather_object(gathered, local)
                    if rank == 0:
                        tokens = (
                            args.seq_length
                            * args.micro_batch_size
                            * args.microbatches
                            * (world // args.pp)
                        )
                        summary = _global_summary(gathered, tokens)
                        variant['repetitions'].append(
                            {'index': repetition, 'ranks': gathered, 'global': summary}
                        )
                        _write_report(args.output, report)
                        latency = summary['max_rank_host_iteration_ms']['median']
                        print(
                            f'{label} repetition={repetition + 1}/{args.repetitions} '
                            f'max-rank iteration median={latency:.3f}ms',
                            flush=True,
                        )
            del chunks, optimizer, batches, records, gathered, local, communicator
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.barrier()
        if rank == 0:
            report['completed'] = True
            _write_report(args.output, report)
    finally:
        fixture.Utils.destroy_model_parallel()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
