# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-FSDP per-use reduce-scatter mode (``megatron_fsdp_reduce_scatter_per_use``) on a
fixed-depth DSv4 loop (GPU).

Two training steps with an SGD step between them (no momentum) of the model of
``strategy_grad_utils`` (a core of 8 sublayers run exactly 4 times, block + edge
``loop_recompute`` with stride 4, three microbatches of one row per rank, per-token loss,
``finalize_model_grads``) on 4 GPUs (or 8) as two distributed-optimizer instances (DP-Shard x
DP-Outer 2), from the same weights and rows in every arm (the mutations run step 1 only):

* ``ref``: Megatron DDP over the whole data-parallel group (fp32 gradient all-reduce);
* ``accumulate``: Megatron-FSDP v1 HSDP (ZeRO-3 inside each instance, outer ``no_shard``),
  accumulate mode, fp32 main and communication gradients (``test_loop_hsdp_gpu.py``'s arm; here
  the memory baseline);
* ``per_use``: the same with per-use reduce-scatter mode;
* ``per_use_hfsdp``: per-use mode with the optimizer state sharded across the instances too
  (outer ``optim``);
* ``per_use_zero2``: per-use mode with ZeRO-2 inside each instance (``optim_grads``: no weight
  release, so the per-use post-backward hook is the unit's only one);
* ``per_use_recipe``: the layout of ``pro_loop_r4_full_fsdp_per_use``: dense ZeRO-3, experts
  ZeRO-2 (``expert_data_parallel_sharding_strategy='optim_grads'``), outer ``optim``. At the
  default EP (``W / 2``) the expert DP-Shard group is one rank, as at EP 64 on 256 GPUs with
  four instances: no per-use reduction of the expert buckets, one reduction per step.

In per-use mode the unit's post-backward hook reduce-scatters the unit's sharded gradient
buckets after every use (each reduced shard is added into the sharded fp32 main gradient) and
the next use starts from a fresh, zeroed bucket; the gradient hooks still reduce each bucket
after its parameters' last use, and only that reduction, on the last microbatch, also reduces
over DP-Outer.

* Every parameter's reduced fp32 gradient of both steps matches ``ref``'s: cosine >= 0.9999 and
  relative difference <= 1e-3 (a gradient zero to fp32 precision: absolute, at most 1e-6 of the
  whole gradient's norm); step 1's losses and every row's LM loss equal ``ref``'s, step 2's
  losses are within 1e-5.
* The recorded reductions, every step: every sharded bucket is reduced by the gradient hooks exactly once
  per microbatch; a core bucket also once after each earlier use (3, or 4 when a hook of the
  bucket runs after the unit's last post-backward hook), a single-use bucket at most once; only
  the hooks' reduction in the last microbatch reduces over DP-Outer, once per step.
* Memory: the peak of the live unsharded gradient buckets (temporary allocations of the gradient
  buffers) is at most 0.75 of accumulate mode's, where every core unit's bucket stays alive
  across the rounds (printed; the bound is an estimate with margin for this model).
* Teeth, each must fail on every core weight whose weight gradient TE accumulates in place:
  ``overwrite`` (every reduction writes over the sharded main gradient instead of adding into
  it: only the last reduction survives) and ``skip_use`` (the second per-use reduction of every
  bucket in the first microbatch is dropped and its bucket freed unreduced).

Launch: ``torchrun --nproc-per-node 4 -m pytest -q tests/unit_tests/transformer/loop/
test_loop_fsdp_per_use_gpu.py`` (also 8). ``LOOP_PER_USE_EP`` sets the expert-parallel size
(default ``W / 2``, the recipes' expert layout; run it also with ``LOOP_PER_USE_EP=1``, where the
experts are sharded over DP-Shard as well). ``MBS_REPORT_DIR`` dumps every per-parameter report.
"""

import collections
import os

import pytest
import torch

from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    CUDA,
    determinism_env,
    model_parallel_once,
)

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
INSTANCES = 2
EP = int(os.environ.get("LOOP_PER_USE_EP", str(max(WORLD // INSTANCES, 1))))
STEPS = 2
MAX_PEAK_RATIO = 0.75
pytestmark = pytest.mark.skipif(
    not CUDA
    or WORLD < 4
    or WORLD % (2 * INSTANCES) != 0
    or WORLD % EP != 0
    or (WORLD // EP) % INSTANCES != 0
    or sgu.NUM_EXPERTS % EP != 0,
    reason="HSDP needs CUDA, DP-Shard >= 2 in each of the 2 instances (W = 4 or 8) and an "
    "expert data-parallel size divisible by the instances",
)


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    if not CUDA:
        yield
        return
    with (
        determinism_env(),
        model_parallel_once(
            expert_model_parallel_size=EP, num_distributed_optimizer_instances=INSTANCES
        ),
    ):
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(False)


def _ddp(model):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.process_groups_config import ProcessGroupCollection

    return DistributedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            overlap_grad_reduce=False, use_distributed_optimizer=False, grad_reduce_in_fp32=True
        ),
        model,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )


def _fsdp(model, *, per_use, outer="no_shard", strategy="optim_grads_params", expert=None):
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel

    return FullyShardedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            use_distributed_optimizer=True,
            data_parallel_sharding_strategy=strategy,
            expert_data_parallel_sharding_strategy=expert,
            num_distributed_optimizer_instances=INSTANCES,
            outer_dp_sharding_strategy=outer,
            megatron_fsdp_accumulate_main_grad=True,
            megatron_fsdp_reduce_scatter_per_use=per_use,
            megatron_fsdp_main_grads_dtype=torch.float32,
            megatron_fsdp_grad_comm_dtype=torch.float32,
        ),
        model,
    )


class _Instrumentation:
    """Records every bucket-group reduction of Megatron-FSDP (``GradReducePipeline.
    _bucket_group_gradient_reduce``) as ``(step, microbatch, bucket ids, outer, per_use)`` and the
    live bytes of the unsharded gradient buckets (``TemporaryBucketAllocator``: the gradient
    buffers' allocator; weight buckets use ``StorageResizeBasedBucketAllocator``, which
    overrides both methods). ``mutation``: ``overwrite`` zeroes the target shard on the
    reduction stream before each reduction (which then writes over it); ``skip_use`` drops the
    second per-use reduction of every bucket in microbatch 0 (its bucket is freed unreduced,
    after the compute stream's writes)."""

    def __init__(self, mutation=None):
        from megatron.core.distributed.fsdp.src.megatron_fsdp import param_and_grad_buffer as pgb

        self.pgb = pgb
        self.original_reduce = pgb.GradReducePipeline._bucket_group_gradient_reduce
        self.original_allocate = pgb.TemporaryBucketAllocator.allocate
        self.original_free = pgb.TemporaryBucketAllocator.free
        self.step = 0
        self.microbatch = -1
        self.calls = []
        self.live = {}
        self.peak = 0
        instrumentation = self

        def reduce(
            pipeline, bucket_group, async_op=False, outer_fsdp_group_grad_reduce=False, **kw
        ):
            per_use = kw.get("per_use", False)
            call = (
                instrumentation.step,
                instrumentation.microbatch,
                tuple(bucket_group),
                bool(outer_fsdp_group_grad_reduce),
                per_use,
            )
            instrumentation.calls.append(call)
            if mutation == "overwrite":
                stream = pipeline.rs_stream or torch.cuda.current_stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for bucket_id in bucket_group:
                        pipeline.get_fsdp_buffer(bucket_id).get_shard_from_local_buffer().zero_()
            if (
                mutation == "skip_use"
                and per_use
                and instrumentation.step == 0
                and instrumentation.microbatch == 0
                and instrumentation.calls.count(call) == 2
            ):
                for bucket_id in bucket_group:
                    gbuf = pipeline.get_fsdp_buffer(bucket_id)
                    gbuf.free_bucket_storage()
                    gbuf.reset_param_main_grad()
                return True
            return instrumentation.original_reduce(
                pipeline,
                bucket_group,
                async_op=async_op,
                outer_fsdp_group_grad_reduce=outer_fsdp_group_grad_reduce,
                **kw,
            )

        def allocate(allocator, bucket_id, size, dtype, device, *args, **kwargs):
            bucket = instrumentation.original_allocate(
                allocator, bucket_id, size, dtype, device, *args, **kwargs
            )
            instrumentation.live[(id(allocator), bucket_id)] = bucket.data.numel() * (
                bucket.data.element_size()
            )
            instrumentation.peak = max(instrumentation.peak, sum(instrumentation.live.values()))
            return bucket

        def free(allocator, bucket_id):
            instrumentation.live.pop((id(allocator), bucket_id), None)
            return instrumentation.original_free(allocator, bucket_id)

        self.patched = dict(_bucket_group_gradient_reduce=reduce)
        self.allocator_patches = dict(allocate=allocate, free=free)

    def on_microbatch(self, index):
        self.microbatch = index

    def __enter__(self):
        for name, value in self.patched.items():
            setattr(self.pgb.GradReducePipeline, name, value)
        for name, value in self.allocator_patches.items():
            setattr(self.pgb.TemporaryBucketAllocator, name, value)
        return self

    def __exit__(self, *exc):
        self.pgb.GradReducePipeline._bucket_group_gradient_reduce = self.original_reduce
        self.pgb.TemporaryBucketAllocator.allocate = self.original_allocate
        self.pgb.TemporaryBucketAllocator.free = self.original_free


def _run_ddp(state):
    """``STEPS`` steps of the DDP reference; returns ``[(step result, grads)]`` and the core's
    fused-wgrad weight names."""
    model = sgu.build_model(state, expert_model_parallel_size=EP)
    ddp = _ddp(model)
    opt = sgu.optimizer(ddp)
    out = []
    for step in range(STEPS):
        opt.zero_grad()
        result = sgu.run_step(ddp, model, sgu.make_rows(step))
        out.append((result, sgu.ddp_grads(ddp, model)))
        if step + 1 < STEPS:
            sgu.step_optimizer(opt)
    fused = sgu.fused_core_weights(model)
    del opt, ddp, model
    torch.cuda.empty_cache()
    return out, fused


def _run_fsdp(
    state,
    *,
    per_use,
    outer="no_shard",
    strategy="optim_grads_params",
    expert=None,
    mutation=None,
    steps=STEPS,
):
    """``steps`` steps of one arm; returns ``[(step result, grads)]`` and the record."""
    model = sgu.build_model(state, expert_model_parallel_size=EP)
    fsdp = _fsdp(model, per_use=per_use, outer=outer, strategy=strategy, expert=expert)
    megatron_fsdp = fsdp.module
    opt = sgu.optimizer(fsdp)
    out = []
    with _Instrumentation(mutation=mutation) as instrumentation:
        for step in range(steps):
            instrumentation.step = step
            opt.zero_grad()
            result = sgu.run_step(
                fsdp, model, sgu.make_rows(step), on_microbatch=instrumentation.on_microbatch
            )
            out.append((result, sgu.fsdp_grads(fsdp)))
            if step + 1 < steps:
                sgu.step_optimizer(opt)
    buffer = megatron_fsdp.param_and_grad_buffer
    # {bucket id: (DP-Shard size, roles of its parameters)}: a bucket whose DP-Shard group is
    # one rank shards nothing and is reduced once per step.
    buckets = {}
    for bucket_id, group in enumerate(buffer.parameter_groups):
        if group.main_grad_buffer is None or not group.requires_grad:
            continue
        shard_size = megatron_fsdp.grad_reduce_pipeline.get_fsdp_buffer(
            bucket_id
        ).data_parallel_group.size()
        roles = {sgu.role(buffer.param_to_name[param])[0] for param in group.params}
        buckets[bucket_id] = (shard_size, roles)
    record = dict(calls=instrumentation.calls, buckets=buckets, peak=instrumentation.peak)
    fsdp.stop_communication()
    del opt, fsdp, model, megatron_fsdp
    torch.cuda.empty_cache()
    return out, record


ARMS = {
    "accumulate": dict(per_use=False),
    "per_use": dict(per_use=True),
    "per_use_hfsdp": dict(per_use=True, outer="optim"),
    "per_use_zero2": dict(per_use=True, strategy="optim_grads"),
    "per_use_recipe": dict(per_use=True, outer="optim", expert="optim_grads"),
}
PER_USE_ARMS = [arm for arm in ARMS if arm != "accumulate"]


@pytest.fixture(scope="module")
def results():
    """Every arm, run once from the same weights and rows."""
    state = sgu.clone_state(sgu.build_model(expert_model_parallel_size=EP))
    out = {}
    out["ref"], out["fused_core_weights"] = _run_ddp(state)
    for arm, kwargs in ARMS.items():
        out[arm] = _run_fsdp(state, **kwargs)
    out["overwrite"] = _run_fsdp(state, per_use=True, mutation="overwrite", steps=1)
    out["skip_use"] = _run_fsdp(state, per_use=True, mutation="skip_use", steps=1)
    return out


def _name(text):
    return sgu.tag(f"FSDP per-use EP{EP}", text)


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("step", range(STEPS))
def test_gradients_match_the_reference(results, arm, step):
    _, reference = results["ref"][step]
    _, grads = results[arm][0][step]
    report, failing = sgu.report_and_fail(_name(f"{arm} step {step + 1} vs ref"), grads, reference)
    assert not failing, sgu.mbs.worst(report, failing)


@pytest.mark.parametrize("arm", PER_USE_ARMS)
def test_losses_match_the_reference(results, arm):
    for step in range(STEPS):
        reference, _ = results["ref"][step]
        result, _ = results[arm][0][step]
        sgu.assert_step_losses_match(step, result, reference, _name(f"{arm} step {step + 1}"))


@pytest.mark.parametrize("arm", PER_USE_ARMS)
def test_reductions_per_use_and_outer_once_per_step(results, arm):
    _, record = results[arm]
    assert record["buckets"], record
    for step in range(STEPS):
        uses, finals, outer = (collections.Counter() for _ in range(3))
        for call_step, microbatch, bucket_ids, with_outer, per_use in record["calls"]:
            if call_step != step:
                continue
            for bucket_id in bucket_ids:
                (uses if per_use else finals)[(microbatch, bucket_id)] += 1
                if with_outer:
                    assert not per_use, (arm, step, microbatch, bucket_ids)
                    outer[(microbatch, bucket_id)] += 1
        print(
            f"[{_name(f'{arm} step {step + 1}')}] per-use {dict(uses)} hooks {dict(finals)} "
            f"outer {dict(outer)}"
        )
        core_buckets = 0
        for bucket_id, (shard_size, roles) in record["buckets"].items():
            for microbatch in range(sgu.ROWS):
                key = (microbatch, bucket_id)
                if shard_size == 1:
                    # Reduced once per step (MegatronFSDP._reduces_grad_every_backward).
                    assert uses[key] == 0, (arm, step, bucket_id, roles)
                    expected = 1 if microbatch == sgu.ROWS - 1 else 0
                    assert finals[key] == expected, (arm, step, key, roles, finals[key])
                    continue
                assert finals[key] == 1, (arm, step, key, roles)
                if roles == {"core"}:
                    assert uses[key] in (sgu.ROUNDS - 1, sgu.ROUNDS), (arm, step, key, uses[key])
                elif "core" not in roles:
                    assert uses[key] <= 1, (arm, step, key, roles, uses[key])
            if shard_size > 1 and roles == {"core"}:
                core_buckets += 1
            earlier = sum(outer[(m, bucket_id)] for m in range(sgu.ROWS - 1))
            assert earlier == 0, (arm, step, bucket_id, earlier)
            assert outer[(sgu.ROWS - 1, bucket_id)] == 1, (arm, step, bucket_id, roles)
        assert core_buckets > 0, record["buckets"]


def test_recipe_arm_has_one_rank_expert_shards_at_the_default_ep(results):
    """At EP = W / 2 the recipe's expert layout holds: some buckets (the experts') have a
    one-rank DP-Shard group."""
    _, record = results["per_use_recipe"]
    sizes = sorted({shard_size for shard_size, _ in record["buckets"].values()})
    print(f"[{_name('per_use_recipe')}] DP-Shard sizes {sizes}", flush=True)
    if EP == WORLD // INSTANCES:
        assert 1 in sizes and max(sizes) > 1, sizes


@pytest.mark.parametrize("arm", PER_USE_ARMS)
def test_per_use_mode_lowers_the_gradient_bucket_peak(results, arm):
    accumulate = results["accumulate"][1]["peak"]
    per_use = results[arm][1]["peak"]
    ratio = per_use / accumulate
    print(
        f"[{_name(arm)}] peak live unsharded gradient buckets: accumulate {accumulate / 2**20:.2f}"
        f" MiB, {arm} {per_use / 2**20:.2f} MiB (ratio {ratio:.3f})",
        flush=True,
    )
    assert ratio <= MAX_PEAK_RATIO, (accumulate, per_use)


@pytest.mark.parametrize("mutation", ["overwrite", "skip_use"])
def test_mutations_fail_on_the_core_fused_weights(results, mutation):
    _, reference = results["ref"][0]
    _, grads = results[mutation][0][0]
    report, _ = sgu.report_and_fail(_name(f"{mutation} (mutation) vs ref"), grads, reference)
    fused = results["fused_core_weights"]
    assert fused
    passing = [name for name in fused if report[name]["ok"]]
    assert not passing, sgu.mbs.worst(report, passing)
