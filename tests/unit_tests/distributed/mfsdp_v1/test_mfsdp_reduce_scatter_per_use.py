# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-FSDP per-use reduce-scatter mode (``megatron_fsdp_reduce_scatter_per_use``), on CPU.

In accumulate mode a parameter used several times per microbatch (the core of a loop model)
keeps its unsharded gradient bucket from its first backward use to its last, where its
gradient hook reduce-scatters it once. In per-use mode the FSDP unit's post-backward hook
reduce-scatters the unit's sharded buckets after every use, adding each use's reduced shard
into the sharded main gradient, and the next use's main-gradient fetch finishes that reduction
and starts from a fresh, zeroed bucket.

The tests run Megatron-FSDP's own reduction code (``GradReducePipeline``: ``reduce_gradients``,
``reduce_scatter_use``, ``before_main_grad_fetch``, ``_bucket_group_gradient_reduce`` with its
queue and bucket release, ``reset``; ``MegatronFSDP._reduce_scatter_use`` /
``_accumulate_grad`` / ``_reduces_grad_every_backward``; ``DataParallelBuffer``;
``RegisterFSDPBackwardFunction``) without a device or process groups: CUDA streams and events
are no-ops, the coalescing manager is a null context and the reduce-scatter is emulated on a
2-rank group whose peer holds the same bucket (rank 0's shard becomes twice its slice of the
bucket). The hooks are wired as ``MegatronFSDP._register_fsdp_hooks`` wires them: the unit's
post-backward function on the inputs of every call, the post-accumulate-grad hook per
parameter (``_accumulate_grad``, then ``reduce_gradients`` with the DP-Outer flag on the last
microbatch), the pipeline reset after each backward, and the main-gradient getter of
``ParamAndGradBuffer`` (``before_main_grad_fetch``, then ``fetch_bucket``).

The model: a core of ``UNITS`` units run ``ROUNDS`` times per microbatch, each unit
``tanh(gain * (x W^T)) + bias`` with a TE-style fused weight-gradient ``W`` (the contract of TE
``linear.py``), and ``gain`` and ``bias`` reduced from their hooks (``bias`` does not lie on the
path to the unit's input, so its last hook may fire after the unit's post-backward). All three
share the unit's bucket.

* Every parameter's reduced gradient equals plain fp32 autograd summed over the microbatches
  (relative <= 1e-6), in accumulate mode and in per-use mode; a single use gives bitwise the
  same shard in both modes.
* Per-use mode reduces every bucket once per use (the last use's reduction may come from the
  gradient hooks, plus at most one reduction of a fresh bucket), carries the DP-Outer flag only
  in the hook-driven reduction of the last microbatch, and keeps at most a few unsharded buckets
  alive (accumulate mode: every core unit's, across the rounds).
* Teeth, each must fail the gradient check: ``overwrite`` (a reduction writes over the sharded
  main gradient instead of adding), ``skip_use`` (one use's reduction dropped, its bucket freed)
  and ``no_drain`` (the next use writes into the bucket a queued reduction still holds).
"""

import collections
import functools
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp import param_and_grad_buffer as pgb
from megatron.core.distributed.fsdp.src.megatron_fsdp.distributed_data_parallel_config import (
    DistributedDataParallelConfig as FSDPConfig,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import (
    MegatronFSDP,
    RegisterFSDPBackwardFunction,
)

ROUNDS = 4
UNITS = 6
MICROBATCHES = 3
HIDDEN = 8
TOKENS = 4
WORLD = 2
MAX_RELATIVE = 1e-6

# The methods the tests wrap, before any test patches them.
_REDUCE = pgb.GradReducePipeline._bucket_group_gradient_reduce
_ENFORCE_DOUBLE_BUFFER_LIMIT = pgb.GradReducePipeline._enforce_double_buffer_limit


class _Group:
    """A 2-rank data-parallel group seen from rank 0."""

    def size(self):
        return WORLD

    def rank(self):
        return 0


class _Event:
    def wait(self, stream=None):
        del stream


class _Stream:
    def wait_stream(self, stream):
        del stream

    def record_event(self, event=None):
        del event
        return _Event()


@pytest.fixture
def cpu_collectives(monkeypatch):
    """No-op streams and coalescing; a reduce-scatter over two ranks holding the same input."""
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: WORLD)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(pgb, "_coalescing_manager", lambda *args, **kwargs: nullcontext())

    def reduce_scatter_tensor(output, input, op=None, group=None, async_op=False):
        del op, async_op
        output.copy_(input.view(group.size(), -1)[group.rank()] * group.size())

    monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", reduce_scatter_tensor)


class _CountingAllocator(pgb.TemporaryBucketAllocator):
    """The default temporary allocator, filling every new allocation with NaN (a missing zero
    shows) and recording the peak number of live buckets."""

    def __init__(self):
        super().__init__()
        self.peak = 0

    def allocate(
        self, bucket_id, size, dtype, device, mem_alloc_context=None, strict_assignments=True
    ):
        if bucket_id not in self.buckets:
            data = torch.full((size,), float("nan"), dtype=dtype, device=device)
            self.buckets[bucket_id] = pgb.Bucket(data=data)
            self.peak = max(self.peak, len(self.buckets))
        return self.buckets[bucket_id]


class _FusedWgradLinear(torch.autograd.Function):
    """``y = x W^T`` with Transformer Engine's fused weight-gradient contract for a
    Megatron-FSDP parameter: ``overwrite_main_grad`` read at forward time, ``get_main_grad()``
    in the backward, copy or add, ``grad_added_to_main_grad`` set, a dummy gradient to
    autograd."""

    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x)
        ctx.weight = weight
        ctx.overwrite = bool(getattr(weight, "overwrite_main_grad", False))
        ctx.main_grad_func = weight.get_main_grad
        return x @ weight.t()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        weight = ctx.weight
        main_grad = ctx.main_grad_func()
        wgrad = grad_output.t() @ x
        if ctx.overwrite:
            main_grad.copy_(wgrad)
        else:
            main_grad.add_(wgrad)
        weight.grad_added_to_main_grad = True
        return grad_output @ weight, torch.zeros_like(weight)


class _Unit(torch.nn.Module):
    """``tanh(gain * (x W^T)) + bias``; ``fused``: TE-style fused weight gradient of ``W``."""

    def __init__(self, generator, fused):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(HIDDEN, HIDDEN, generator=generator) / 3)
        self.gain = torch.nn.Parameter(1.0 + 0.1 * torch.randn(HIDDEN, generator=generator))
        self.bias = torch.nn.Parameter(0.1 * torch.randn(HIDDEN, generator=generator))
        self.fused = fused

    def forward(self, x):
        y = _FusedWgradLinear.apply(x, self.weight) if self.fused else x @ self.weight.t()
        return torch.tanh(self.gain * y) + self.bias


class _Core(torch.nn.Module):
    def __init__(self, seed=1234, fused=True):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.units = torch.nn.ModuleList(_Unit(generator, fused) for _ in range(UNITS))

    def forward(self, x, rounds, before_unit=None):
        for _ in range(rounds):
            for unit in self.units:
                if before_unit is not None:
                    x = before_unit(unit, x)
                x = unit(x)
        return (x * torch.linspace(0.5, 1.5, HIDDEN)).sum()


def _inputs(microbatches):
    generator = torch.Generator().manual_seed(77)
    return [torch.randn(TOKENS, HIDDEN, generator=generator) for _ in range(microbatches)]


class _FakeFsdp:
    """The MegatronFSDP state the per-use methods read, with Megatron-FSDP's own methods: one
    sharded fp32 gradient bucket per unit (``optim_grads_params``) on a 2-rank group, the real
    ``GradReducePipeline``, and the hooks wired as ``_register_fsdp_hooks`` wires them."""

    _reduce_scatter_use = MegatronFSDP._reduce_scatter_use
    _accumulate_grad = MegatronFSDP._accumulate_grad
    _reduces_grad_every_backward = MegatronFSDP._reduces_grad_every_backward
    _sharding_strategy_for_param = MegatronFSDP._sharding_strategy_for_param
    _overwrites_main_grad = MegatronFSDP._overwrites_main_grad
    all_gather_and_wait_parameters_ready = MegatronFSDP.all_gather_and_wait_parameters_ready

    def __init__(self, core, *, per_use, queue_buckets=1, late=None):
        self.ddp_config = FSDPConfig(
            data_parallel_sharding_strategy="optim_grads_params",
            megatron_fsdp_accumulate_main_grad=True,
            megatron_fsdp_reduce_scatter_per_use=per_use,
        )
        self.accumulate_main_grad = True
        self.reduce_scatter_per_use = per_use
        self.report_nan_in_param_grad = False
        self.allocator = _CountingAllocator()
        group = _Group()
        parameter_groups, param_to_group, names = [], {}, {}
        for unit_id, unit in enumerate(core.units):
            params = list(unit.parameters())
            gbuf = pgb.DataParallelBuffer(
                self.ddp_config,
                params,
                is_data_distributed=True,
                bucket_id=unit_id,
                dtype=torch.float32,
                device=torch.device("cpu"),
                data_parallel_group=group,
                dp_rank=0,
                temporary_bucket_allocator=self.allocator,
                zero_fresh_bucket=True,
            )
            gbuf.init_data(torch.zeros(gbuf.data_size))
            parameter_groups.append(
                SimpleNamespace(
                    params=params,
                    requires_grad=True,
                    main_grad_buffer=gbuf,
                    hfsdp_helper_gbuf=None,
                    hsdp_comm_gbuf=None,
                    sharding_strategy="optim_grads_params",
                    is_expert_param=False,
                    fsdp_unit_id=unit_id,
                )
            )
            for name, param in unit.named_parameters():
                param_to_group[param] = unit_id
                names[param] = f"units.{unit_id}.{name}"
                self._attach_main_grad(param, gbuf, gbuf.param_idx[param])
        self.param_and_grad_buffer = SimpleNamespace(
            ddp_config=self.ddp_config,
            mp_policy=SimpleNamespace(grad_comm_dtype=torch.float32),
            parameter_groups=parameter_groups,
            param_to_param_group=param_to_group,
            param_to_name=names,
            num_buckets=len(parameter_groups),
            bucket_to_bucket_group={i: [i] for i in range(len(parameter_groups))},
            dist_index=SimpleNamespace(use_hybrid_fsdp=False),
        )
        self.grad_reduce_pipeline = pgb.GradReducePipeline(self.param_and_grad_buffer)
        bucket_size = parameter_groups[0].main_grad_buffer.bucket_index.size
        self.suggested_RS_queue_capacity = queue_buckets * bucket_size
        self.all_gather_pipeline = SimpleNamespace(all_gather_params=lambda **kwargs: None)
        self.dist_index = SimpleNamespace(use_hybrid_fsdp=False)
        self.microbatch_count = 0
        self.model_auto_sync = False
        self.suggested_AG_prefetch_size = None
        self.is_last_microbatch = False
        # ``late``: the name ("weight" / "bias") of the parameter of every unit whose gradient
        # hook runs at the end of the backward instead (``run_late_hooks``), after its unit's
        # last post-backward hook, as a delayed weight-gradient hook does
        # (``post_wgrad_grad_acc_hook``). Autograd alone runs every hook of this model before
        # the unit's post-backward hook of the same use.
        self.late_params = {
            param for unit in core.units for name, param in unit.named_parameters() if name == late
        }
        self.pending = []
        for param in param_to_group:
            param.register_post_accumulate_grad_hook(self._post_accumulate_grad)

    def _attach_main_grad(self, param, gbuf, item_id):
        """``ParamAndGradBuffer``'s main-gradient getter."""
        pipeline = lambda: self.grad_reduce_pipeline  # noqa: E731 (built after the buffers)

        def get_main_grad():
            pipeline().before_main_grad_fetch(gbuf.bucket_id)
            bucket = gbuf.fetch_bucket(dtype=torch.float32)
            return gbuf.get_item_from_bucket(bucket, item_id).view(param.shape)

        param.get_main_grad = get_main_grad
        param.grad_added_to_main_grad = False

    def _post_accumulate_grad(self, param, late=False):
        """``_process_post_backward_gradients([param])`` of a gradient-sharding strategy."""
        if param in self.late_params and not late:
            self.pending.append(param)
            return
        self._accumulate_grad(param)
        self.grad_reduce_pipeline.reduce_gradients(
            [param],
            suggested_queue_capacity=self.suggested_RS_queue_capacity,
            outer_fsdp_group_grad_reduce=self.is_last_microbatch,
        )

    def run_late_hooks(self):
        for param in sorted(self.pending, key=self.param_and_grad_buffer.param_to_name.get):
            self._post_accumulate_grad(param, late=True)
        self.pending = []

    def before_unit(self, unit, x):
        """The unit's forward pre-hooks: unshard (sets ``overwrite_main_grad``), then the
        per-use post-backward function on its input (``_register_post_backward_hook``)."""
        self.all_gather_and_wait_parameters_ready(list(unit.parameters()), wait_bucket_ready=False)
        if self.reduce_scatter_per_use:
            (x,) = RegisterFSDPBackwardFunction.apply(
                functools.partial(self._reduce_scatter_use, unit), x
            )
        return x

    def shards(self):
        """Every parameter's reduced gradient on rank 0 (its slice of the shard), by name."""
        out = {}
        for group in self.param_and_grad_buffer.parameter_groups:
            gbuf = group.main_grad_buffer
            full = torch.zeros(gbuf.bucket_index.size)
            full[: gbuf.data_size] = gbuf.data
            for param in group.params:
                item = gbuf.get_item_from_bucket(pgb.Bucket(data=full), gbuf.param_idx[param])
                out[self.param_and_grad_buffer.param_to_name[param]] = item.view(param.shape)
        return out, {
            self.param_and_grad_buffer.param_to_name[param]: self._in_shard(group, param)
            for group in self.param_and_grad_buffer.parameter_groups
            for param in group.params
        }

    @staticmethod
    def _in_shard(group, param):
        """The mask of ``param``'s elements that lie in rank 0's shard."""
        gbuf = group.main_grad_buffer
        mask = torch.zeros(gbuf.bucket_index.size, dtype=torch.bool)
        mask[: gbuf.data_size] = True
        item = gbuf.get_item_from_bucket(pgb.Bucket(data=mask), gbuf.param_idx[param])
        return item.view(param.shape)


class _Recorder:
    """Wraps ``GradReducePipeline._bucket_group_gradient_reduce``: records ``(microbatch,
    bucket ids, outer, per_use)`` and calls the original without the DP-Outer reduction (no
    DP-Outer group here). ``mutation``: ``overwrite`` zeroes the target shard before each
    reduction (the reduction then writes over it); ``skip_use`` drops the per-use reduction of
    round 2 of microbatch 0 (its buckets are freed unreduced)."""

    def __init__(self, monkeypatch, mutation=None):
        self.calls = []
        self.microbatch = 0
        recorder = self

        def recording(
            pipeline, bucket_group, async_op=False, outer_fsdp_group_grad_reduce=False, **kw
        ):
            per_use = kw.get("per_use", False)
            call = (recorder.microbatch, tuple(bucket_group), outer_fsdp_group_grad_reduce, per_use)
            recorder.calls.append(call)
            if mutation == "overwrite":
                for bucket_id in bucket_group:
                    pipeline.get_fsdp_buffer(bucket_id).data.zero_()
            if mutation == "skip_use" and per_use and recorder.microbatch == 0:
                if recorder.calls.count(call) == 2:
                    for bucket_id in bucket_group:
                        gbuf = pipeline.get_fsdp_buffer(bucket_id)
                        gbuf.free_bucket_storage()
                        gbuf.reset_param_main_grad()
                    return True
            return _REDUCE(pipeline, bucket_group, async_op=async_op, **kw)

        monkeypatch.setattr(pgb.GradReducePipeline, "_bucket_group_gradient_reduce", recording)
        if mutation == "no_drain":
            monkeypatch.setattr(
                pgb.GradReducePipeline,
                "before_main_grad_fetch",
                lambda pipeline, bucket_id: _ENFORCE_DOUBLE_BUFFER_LIMIT(pipeline, [bucket_id]),
            )


def _run(
    monkeypatch,
    *,
    per_use,
    rounds=ROUNDS,
    mutation=None,
    microbatches=MICROBATCHES,
    queue_buckets=1,
    late=None,
):
    """``queue_buckets``: the reduction queue's capacity in buckets. With 1, a queued per-use
    reduction is mostly drained by later units' reductions; with ``2 * UNITS``, only by the
    bucket's next fetch (``before_main_grad_fetch``) or reduction."""
    core = _Core()
    fake = _FakeFsdp(core, per_use=per_use, queue_buckets=queue_buckets, late=late)
    recorder = _Recorder(monkeypatch, mutation=mutation)
    for index, x in enumerate(_inputs(microbatches)):
        recorder.microbatch = index
        fake.is_last_microbatch = index == microbatches - 1
        # The input requires a gradient (as an embedding's output does), so that the first
        # unit's post-backward hook fires too.
        loss = core(x.clone().requires_grad_(), rounds, before_unit=fake.before_unit)
        loss.backward()
        fake.run_late_hooks()
        # The root post-backward hook (_root_post_backward): drain and reset the pipeline.
        fake.grad_reduce_pipeline.reset()
    grads, masks = fake.shards()
    return grads, masks, recorder.calls, fake.allocator.peak


def _reference(rounds=ROUNDS, microbatches=MICROBATCHES):
    """Plain fp32 autograd, summed over microbatches, times 2 (the emulated peer)."""
    core = _Core(fused=False)
    for x in _inputs(microbatches):
        core(x, rounds).backward()
    return {name: WORLD * param.grad for name, param in core.named_parameters()}


def _relative(grads, masks, reference):
    worst = 0.0
    for name, expected in reference.items():
        mask = masks[name]
        if not mask.any():
            continue
        got, want = grads[name][mask].double(), expected[mask].double()
        worst = max(worst, ((got - want).norm() / want.norm()).item())
    return worst


LATE = [None, "weight", "bias"]
LATE_IDS = ["hooks_in_order", "late_fused_weight", "late_bias"]


@pytest.mark.parametrize("late", LATE, ids=LATE_IDS)
@pytest.mark.parametrize("queue_buckets", [1, 2 * UNITS], ids=["short_queue", "long_queue"])
@pytest.mark.parametrize("per_use", [False, True], ids=["accumulate", "per_use"])
def test_gradients_match_autograd(cpu_collectives, monkeypatch, per_use, queue_buckets, late):
    grads, masks, _, _ = _run(monkeypatch, per_use=per_use, queue_buckets=queue_buckets, late=late)
    assert all(torch.isfinite(grad).all() for grad in grads.values())
    assert sum(int(mask.sum()) for mask in masks.values()) > 0
    assert _relative(grads, masks, _reference()) <= MAX_RELATIVE


@pytest.mark.parametrize("late", LATE, ids=LATE_IDS)
def test_reductions_per_use_and_outer_once_per_step(cpu_collectives, monkeypatch, late):
    """Per-use mode: the hooks reduce every bucket exactly once per microbatch, after the last
    use, and only that reduction in the last microbatch carries the DP-Outer flag. The unit's
    post-backward hook reduces it after each earlier use; after the last use too when a hook
    of the bucket runs after it (``late``: the bucket is not yet reduced then), and not when
    the hooks already reduced it. Accumulate mode: one reduction per bucket per microbatch."""
    for per_use in (False, True):
        _, _, calls, _ = _run(monkeypatch, per_use=per_use, late=late)
        uses, finals, outer = (collections.Counter() for _ in range(3))
        for microbatch, bucket_ids, with_outer, is_per_use in calls:
            for bucket_id in bucket_ids:
                (uses if is_per_use else finals)[(microbatch, bucket_id)] += 1
                if with_outer:
                    assert not is_per_use, calls
                    outer[(microbatch, bucket_id)] += 1
        for microbatch in range(MICROBATCHES):
            for bucket_id in range(UNITS):
                key = (microbatch, bucket_id)
                assert finals[key] == 1, (per_use, key, calls)
                expected_uses = (ROUNDS if late else ROUNDS - 1) if per_use else 0
                assert uses[key] == expected_uses, (per_use, late, key, uses[key])
                last = microbatch == MICROBATCHES - 1
                assert outer[key] == (1 if last else 0), (per_use, key)


def test_per_use_mode_frees_buckets_between_uses(cpu_collectives, monkeypatch):
    """Accumulate mode keeps every core unit's bucket alive across the rounds; per-use mode
    keeps the bucket being written plus the queued reductions (capacity: one bucket)."""
    _, _, _, peak_accumulate = _run(monkeypatch, per_use=False)
    _, _, _, peak_per_use = _run(monkeypatch, per_use=True)
    assert peak_accumulate == UNITS
    assert peak_per_use <= 3, peak_per_use


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "autograd"])
def test_single_use_shards_do_not_depend_on_the_mode(cpu_collectives, monkeypatch, fused):
    shards = {}
    for per_use in (False, True):
        core = _Core(fused=fused)
        fake = _FakeFsdp(core, per_use=per_use)
        _Recorder(monkeypatch)
        fake.is_last_microbatch = True
        core(_inputs(1)[0], 1, before_unit=fake.before_unit).backward()
        fake.grad_reduce_pipeline.reset()
        groups = fake.param_and_grad_buffer.parameter_groups
        shards[per_use] = [group.main_grad_buffer.data.clone() for group in groups]
    for accumulate_shard, per_use_shard in zip(shards[False], shards[True]):
        assert torch.equal(accumulate_shard.view(torch.int32), per_use_shard.view(torch.int32))


@pytest.mark.parametrize("mutation", ["overwrite", "skip_use", "no_drain"])
def test_mutations_fail_the_gradient_check(cpu_collectives, monkeypatch, mutation):
    # The long queue keeps a per-use reduction queued until the bucket's next use, where
    # no_drain matters (with the short queue later units' reductions drain it first).
    grads, masks, _, _ = _run(monkeypatch, per_use=True, mutation=mutation, queue_buckets=2 * UNITS)
    assert _relative(grads, masks, _reference()) > 1e-2, mutation


def test_a_queued_bucket_is_never_reduced_twice(cpu_collectives, monkeypatch):
    """Two per-use reductions of a bucket with no fetch in between: the second drains the
    first (which frees the bucket) and reduces a fresh, zeroed bucket; the same for the
    reduction after the last use. The shard holds the written gradient once."""
    core = _Core()
    fake = _FakeFsdp(core, per_use=True, queue_buckets=2 * UNITS)
    _Recorder(monkeypatch)
    pipeline = fake.grad_reduce_pipeline
    weight = core.units[0].weight
    weight.get_main_grad().fill_(1.5)
    fake._reduce_scatter_use(core.units[0])
    assert pipeline.bucket_status[0] == pgb.BucketStatus.COMMUNICATING
    fake._reduce_scatter_use(core.units[0])
    params = list(core.units[0].parameters())
    for param in params:
        param.grad_added_to_main_grad = True  # the fused path: no fetch at the hook
        fake._post_accumulate_grad(param)
    pipeline.reset()
    grads, masks = fake.shards()
    mask = masks["units.0.weight"]
    assert torch.all(grads["units.0.weight"][mask] == WORLD * 1.5)
    for name in ("units.0.gain", "units.0.bias"):
        assert not grads[name][masks[name]].any(), name


def test_partial_backward_flushes_each_live_subset_before_its_next_use(cpu_collectives):
    core = _Core()
    fake = _FakeFsdp(core, per_use=False, queue_buckets=2 * UNITS)
    pipeline = fake.grad_reduce_pipeline
    expected = collections.defaultdict(float)
    # R2 skips the indexer and R1 revisits some of the same weights. Both VJPs may
    # finish with only a subset of this unit's bucket ready, in consecutive waves.
    unit = core.units[0]
    for writes in (
        (("weight", 1.0),),
        (("gain", 2.0), ("weight", 3.0)),
        (("weight", 4.0), ("gain", 5.0), ("bias", 6.0)),
    ):
        with pipeline.partial_backward(fake.suggested_RS_queue_capacity):
            for name, grad in writes:
                param = getattr(unit, name)
                param.get_main_grad().add_(grad)
                param.grad_added_to_main_grad = True
                fake._post_accumulate_grad(param)
                expected[name] += WORLD * grad
            # A complete bucket starts reducing before this VJP finishes.
            if len(writes) == 3:
                assert pipeline.bucket_status[0] == pgb.BucketStatus.COMMUNICATING
        assert pipeline.bucket_status[0] == pgb.BucketStatus.COMMUNICATING
    pipeline.reset()
    grads, masks = fake.shards()
    for name, want in expected.items():
        key = f"units.0.{name}"
        torch.testing.assert_close(grads[key][masks[key]], torch.full_like(grads[key][masks[key]], want))


def test_coalesced_partial_backward_clears_ready_siblings(cpu_collectives):
    core = _Core()
    fake = _FakeFsdp(core, per_use=False, queue_buckets=2 * UNITS)
    pipeline = fake.grad_reduce_pipeline
    for i in range(2):
        fake.param_and_grad_buffer.bucket_to_bucket_group[i] = [0, 1]
    for current in ((0, 1), (0,)):
        with pipeline.partial_backward(fake.suggested_RS_queue_capacity):
            for idx in current:
                for p in core.units[idx].parameters():
                    p.get_main_grad().add_(1.)
                    p.grad_added_to_main_grad = True
                    fake._post_accumulate_grad(p)
    pipeline.reset()
    grads, masks = fake.shards()
    for idx, uses in ((0, 2), (1, 1)):
        for name, _ in core.units[idx].named_parameters():
            key = f"units.{idx}.{name}"
            actual = grads[key][masks[key]]
            torch.testing.assert_close(actual, torch.full_like(actual, WORLD * uses))
