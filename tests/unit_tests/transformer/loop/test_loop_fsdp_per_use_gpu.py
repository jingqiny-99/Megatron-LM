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
over DP-Outer. A bucket whose DP-Shard group is one rank (the experts at the default EP) is not
sharded: it accumulates in its persistent buffer, has no per-use reduction and is reduced once
per step.

Step 2 is teacher-forced (``strategy_grad_utils``): after its step-1 gradient is read, every
arm's reduced gradient storage takes ``ref``'s step-1 gradient, and the arm runs its own
optimizer step, weight install, gradient-buffer reset, gathers and step 2.

* Every parameter's reduced fp32 gradient of both steps matches ``ref``'s: cosine >= 0.9999 and
  relative difference <= 1e-3 (a gradient zero to fp32 precision: absolute, at most 1e-6 of the
  whole gradient's norm); the losses of both steps and every row's LM loss equal ``ref``'s
  within 1e-6. Before step 2 is gated, the precondition: the forced gradient reads back on every
  rank, and the fp32 main weights the arm's optimizer produced (this rank's slices) and the
  persistent buffers equal ``ref``'s masters and buffers exactly. The compute weights the
  optimizer step installed (this rank's shard of every model weight buffer) equal ``ref``'s bf16
  weights after that step exactly (a test of its own).
* Given the same step-1 gradient the un-forced ``per_use`` path agrees at step 2:
  ``per_use_unforced`` against ``ref_from_per_use`` (``ref`` stepping on its step-1 gradient),
  from exactly equal main weights. The un-forced step 2 against ``ref`` is printed (diagnostic).
* The recorded reductions, every step: every bucket's DP-Shard size is the layout's; a bucket
  whose strategy shards gradients over more than one rank is reduced by the gradient hooks
  exactly once per microbatch, a core bucket also once after each earlier use (3, or 4 when a
  hook of the bucket runs after the unit's last post-backward hook), a single-use bucket at
  most once; any other bucket only by the hook of the last microbatch, never per use; only the
  hooks' reduction in the last microbatch reduces over DP-Outer, once per step.
* Memory: the peak of the live unsharded gradient buckets (temporary allocations of the gradient
  buffers) is at most 0.75 of accumulate mode's, where every core unit's bucket stays alive
  across the rounds (printed; the bound is an estimate with margin for this model).
* Teeth (step 1): ``overwrite`` (every reduction writes over the sharded main gradient instead
  of adding into it: only the last reduction survives) must fail every core weight whose weight
  gradient TE accumulates in place. ``skip_use`` (the second per-use reduction of every bucket
  in the first microbatch is dropped and its bucket freed unreduced) must fail every such weight
  whose bucket is reduced per use, and leave those reduced once per step (derived from the
  layout, equal to those observed on the ``per_use`` arm) exactly equal to the ``per_use``
  arm's. ``outer_never`` (no DP-Outer reduction) must fail every fused core weight, the
  once-per-step ones included (some at ``EP = W / 2``, none at ``EP = 1``).

Launch: ``torchrun --nproc-per-node 4 -m pytest -q tests/unit_tests/transformer/loop/
test_loop_fsdp_per_use_gpu.py`` (also 8). ``LOOP_PER_USE_EP`` sets the expert-parallel size
(default ``W / 2``, the recipes' expert layout; run it also with ``LOOP_PER_USE_EP=1``, where the
experts are sharded over DP-Shard as well). ``MBS_REPORT_DIR`` dumps every per-parameter report.
"""

import os
import traceback

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
DENSE = "optim_grads_params"  # the dense sharding strategy unless an arm sets another
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


def _fsdp(model, *, per_use, outer="no_shard", strategy=DENSE, expert=None):
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
    after the compute stream's writes); ``outer_never`` clears every reduction's DP-Outer flag."""

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
            if mutation == "outer_never":
                outer_fsdp_group_grad_reduce = False
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


def _run_ddp(state, forced=None):
    """``STEPS`` steps of the DDP reference; returns ``[(step result, grads)]`` and a record:
    the core's fused-wgrad weight names (``fused``), the fp32 masters and the model weights
    (bf16: the masters rounded) after the step-1 update (``masters``, ``weights``), the
    persistent buffers step 2 starts with (``buffers``) and, with ``forced`` (step-1 gradients
    to step on: the teacher forcing), the read-back count (``readback``)."""
    model = sgu.build_model(state, expert_model_parallel_size=EP)
    ddp = _ddp(model)
    opt = sgu.optimizer(ddp)
    out = []
    info = dict(masters=None, weights=None, buffers=None, readback=None)
    for step in range(STEPS):
        opt.zero_grad()
        result = sgu.run_step(ddp, model, sgu.make_rows(step))
        out.append((result, sgu.ddp_grads(ddp, model)))
        if step + 1 < STEPS:
            if forced is not None:
                sgu.force_main_grads(model, forced)
                info["readback"] = sgu.agreed(
                    len(sgu.mismatches(sgu.ddp_grads(ddp, model), forced))
                )
            sgu.step_optimizer(opt)
            info["masters"] = sgu.float16_optimizer_masters(opt, model)
            info["weights"] = {name: p.detach().clone() for name, p in model.named_parameters()}
            info["buffers"] = sgu.persistent_buffers(model)
    info["fused"] = sgu.fused_core_weights(model)
    del opt, ddp, model
    torch.cuda.empty_cache()
    return out, info


def _run_fsdp(
    state,
    *,
    per_use,
    outer="no_shard",
    strategy=DENSE,
    expert=None,
    mutation=None,
    steps=STEPS,
    forced=None,
):
    """``steps`` steps of one arm; returns ``[(step result, grads)]`` and the record: the
    reductions (``calls``), the gradient-bucket peak, the bucket layout (``buckets``: DP-Shard
    size and roles; names and expert flags), after the step-1 update this rank's fp32 main
    weight slices (``main``), its installed compute weight slices (``installed``, or the error
    reading them under ``installed_error``) and the persistent buffers (``buffers``), and with
    ``forced`` (step-1 gradients to step on: the teacher forcing) the read-back count
    (``readback``) and the optimizer parameters' gradients against ``forced``
    (``optimizer_grads``, local)."""
    model = sgu.build_model(state, expert_model_parallel_size=EP)
    fsdp = _fsdp(model, per_use=per_use, outer=outer, strategy=strategy, expert=expert)
    opt = sgu.optimizer(fsdp)
    out = []
    record = dict(main=None, installed=None, buffers=None, readback=None, optimizer_grads=None)
    with _Instrumentation(mutation=mutation) as instrumentation:
        for step in range(steps):
            instrumentation.step = step
            opt.zero_grad()
            result = sgu.run_step(
                fsdp, model, sgu.make_rows(step), on_microbatch=instrumentation.on_microbatch
            )
            out.append((result, sgu.fsdp_grads(fsdp)))
            if step + 1 < steps:
                if forced is not None:
                    sgu.force_fsdp_grads(fsdp, forced)
                    record["readback"] = sgu.agreed(
                        len(sgu.mismatches(sgu.fsdp_grads(fsdp), forced))
                    )
                    record["optimizer_grads"] = sgu.fsdp_optimizer_grad_mismatches(fsdp, forced)
                sgu.step_optimizer(opt)
                record["main"] = sgu.fsdp_main_weight_slices(fsdp)
                try:  # local reads only: an error is kept for the install test to report
                    record["installed"] = sgu.fsdp_model_weight_slices(fsdp)
                except Exception:  # pylint: disable=broad-except
                    record["installed_error"] = traceback.format_exc()
                record["buffers"] = sgu.persistent_buffers(model)
    layout = sgu.fsdp_bucket_layout(fsdp)
    # {bucket id: (DP-Shard size, roles of its parameters)}.
    buckets = {
        bucket_id: (size, {sgu.role(name)[0] for name in layout["names"][bucket_id]})
        for bucket_id, size in layout["sizes"].items()
    }
    record.update(
        calls=instrumentation.calls,
        buckets=buckets,
        names=layout["names"],
        expert_params=layout["expert_params"],
        peak=instrumentation.peak,
    )
    fsdp.stop_communication()
    del opt, fsdp, model
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


def _strategies(arm):
    """``(dense, expert)`` sharding strategies of an arm (an unset expert strategy is the dense
    one)."""
    dense = ARMS[arm].get("strategy", DENSE)
    return dense, ARMS[arm].get("expert") or dense


@pytest.fixture(scope="module")
def results():
    """Every arm, run once from the same weights and rows; the arms take ``ref``'s step-1
    gradient before their optimizer step (teacher forcing). The preconditions are agreed over
    the ranks at the end (in a fixed order) and asserted by the tests."""
    state = sgu.clone_state(sgu.build_model(expert_model_parallel_size=EP))
    out = {}
    out["ref"], out["ref_info"] = _run_ddp(state)
    out["fused_core_weights"] = out["ref_info"]["fused"]
    forced = out["ref"][0][1]
    for arm, kwargs in ARMS.items():
        out[arm] = _run_fsdp(state, **kwargs, forced=forced)
    out["overwrite"] = _run_fsdp(state, per_use=True, mutation="overwrite", steps=1)
    out["skip_use"] = _run_fsdp(state, per_use=True, mutation="skip_use", steps=1)
    out["outer_never"] = _run_fsdp(state, per_use=True, mutation="outer_never", steps=1)
    # Un-forced: per_use on its own step-1 gradient, and ref stepping on that same gradient.
    out["per_use_unforced"] = _run_fsdp(state, **ARMS["per_use"])
    out["ref_from_per_use"], out["ref_from_per_use_info"] = _run_ddp(
        state, forced=out["per_use_unforced"][0][0][1]
    )
    for arm in ARMS:
        record = out[arm][1]
        start = sgu.fsdp_start_mismatches(record, out["ref_info"])
        record["forcing"] = sgu.forcing_record(record["readback"], start)
        record["install"] = sgu.install_record(record, out["ref_info"]["weights"])
    record, info = out["ref_from_per_use_info"], out["per_use_unforced"][1]
    record["forcing"] = sgu.forcing_record(
        record["readback"], sgu.fsdp_start_mismatches(info, record)
    )
    return out


def _name(text):
    return sgu.tag(f"FSDP per-use EP{EP}", text)


@pytest.mark.parametrize("arm", list(ARMS))
def test_step_two_starts_from_the_reference_state(results, arm):
    """The teacher-forcing precondition: the forced gradient read back on every rank, and the
    arm's own optimizer step on it produced ``ref``'s fp32 masters (this rank's slices) and
    buffers exactly. On failure, ``optimizer_grads`` names gradients the forcing missed."""
    record = results[arm][1]
    print(f"[{_name(arm)}] optimizer grads off the forced ones {record['optimizer_grads']}")
    sgu.assert_teacher_forced(record["forcing"], arm)


@pytest.mark.parametrize("arm", list(ARMS))
def test_step_two_runs_on_the_installed_reference_weights(results, arm):
    """The compute weights the arm's step-1 optimizer step installed (this rank's shard of
    every model weight buffer, read right after the step) equal ``ref``'s bf16 weights after
    the same step exactly, on every rank (the check's teeth: ``test_loop_hsdp_gpu.py``'s
    ``no_install``)."""
    install = results[arm][1]["install"]
    assert install["agreed"] == 0, (
        f"{arm}: the installed compute weights differ from ref's in {len(install['local'])} "
        f"tensors on this rank ({install['agreed']} on the worst rank)",
        sorted(install["local"].items())[:8],
    )


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("step", range(STEPS))
def test_gradients_match_the_reference(results, arm, step):
    if step > 0:
        sgu.assert_teacher_forced(results[arm][1]["forcing"], arm)
    _, reference = results["ref"][step]
    _, grads = results[arm][0][step]
    report, failing = sgu.report_and_fail(_name(f"{arm} step {step + 1} vs ref"), grads, reference)
    assert not failing, sgu.mbs.worst(report, failing)


@pytest.mark.parametrize("arm", PER_USE_ARMS)
def test_losses_match_the_reference(results, arm):
    for step in range(STEPS):
        if step > 0:
            sgu.assert_teacher_forced(results[arm][1]["forcing"], arm)
        reference, _ = results["ref"][step]
        result, _ = results[arm][0][step]
        sgu.assert_losses_match(result, reference, _name(f"{arm} step {step + 1}"))


def test_step_two_agrees_given_the_same_step_one_gradient(results):
    """The un-forced path: ``per_use_unforced`` steps on its own step-1 gradient and
    ``ref_from_per_use`` (``ref``) on that same gradient. From exactly equal fp32 main weights
    and buffers, their step-2 gradients pass the gate and their losses are equal."""
    sgu.assert_teacher_forced(
        results["ref_from_per_use_info"]["forcing"], "ref_from_per_use", "per_use_unforced's"
    )
    reference_result, reference = results["ref_from_per_use"][1]
    result, grads = results["per_use_unforced"][0][1]
    report, failing = sgu.report_and_fail(
        _name("per_use_unforced step 2 vs ref_from_per_use"), grads, reference
    )
    assert not failing, sgu.mbs.worst(report, failing)
    sgu.assert_losses_match(result, reference_result, _name("per_use_unforced step 2"))


def test_report_unforced_step_two(results):
    """Diagnostic only (prints, asserts nothing): step 2 without teacher forcing against
    ``ref``: ``per_use_unforced`` (its own step-1 gradient), and ``ref_from_per_use`` (``ref`` on
    ``per_use_unforced``'s step-1 gradient, which differs from its own by fp32 reassociation)."""
    _, reference = results["ref"][1]
    for arm in ("per_use_unforced", "ref_from_per_use"):
        steps = results[arm] if arm == "ref_from_per_use" else results[arm][0]
        report, failing = sgu.report_and_fail(
            _name(f"{arm} step 2 vs ref (diagnostic)"), steps[1][1], reference
        )
        print(f"[{_name(arm)}] step 2 failing {len(failing)} of {len(report)}", flush=True)
        rel = sgu.mbs.relative_loss_differences(steps[1][0].losses, results["ref"][1][0].losses)
        print(f"[{_name(f'{arm} step 2')}] losses rel {rel} (diagnostic)", flush=True)


@pytest.mark.parametrize("arm", PER_USE_ARMS)
def test_reductions_per_use_and_outer_once_per_step(results, arm):
    """Every bucket, every step: the DP-Shard size of the layout; a bucket whose strategy
    shards gradients over more than one rank reduced by the hooks once per microbatch and per
    use as its roles imply, any other bucket only by the last microbatch's hook (the
    expectation comes from the configured strategies, not from a production predicate); the
    DP-Outer reduction once per step, in the last microbatch."""
    _, record = results[arm]
    assert record["buckets"], record
    dense, expert = _strategies(arm)
    for step in range(STEPS):
        finals, uses, outer = sgu.reduction_counts(record["calls"], step)
        for call in record["calls"]:
            assert not (call[0] == step and call[3] and call[4]), (arm, call)  # outer per use
        print(
            f"[{_name(f'{arm} step {step + 1}')}] per-use {dict(uses)} hooks {dict(finals)} "
            f"outer {dict(outer)}"
        )
        core_buckets = 0
        for bucket_id, (shard_size, roles) in record["buckets"].items():
            expert_bucket = sgu.bucket_is_expert(record, bucket_id)
            layout_size = sgu.expected_dp_shard(WORLD, EP, INSTANCES, expert_bucket)
            assert shard_size == layout_size, (arm, bucket_id, roles, shard_size, layout_size)
            strategy = expert if expert_bucket else dense
            every = sgu.reduced_every_microbatch(strategy, layout_size)
            for microbatch in range(sgu.ROWS):
                key = (microbatch, bucket_id)
                if not every:
                    assert uses[key] == 0, (arm, step, bucket_id, strategy, roles)
                    expected = 1 if microbatch == sgu.ROWS - 1 else 0
                    assert finals[key] == expected, (arm, step, key, strategy, finals[key])
                    continue
                assert finals[key] == 1, (arm, step, key, roles)
                if roles == {"core"}:
                    assert uses[key] in (sgu.ROUNDS - 1, sgu.ROUNDS), (arm, step, key, uses[key])
                elif "core" not in roles:
                    assert uses[key] <= 1, (arm, step, key, roles, uses[key])
            if every and roles == {"core"}:
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
    if EP != WORLD // INSTANCES:
        pytest.skip(f"EP {EP} is not the recipe's expert layout (W / {INSTANCES})")
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


def _once_per_step(results, mutation):
    """The parameters whose buckets are reduced once per step in the mutants' configuration
    (the ``per_use`` arm's): derived from the layout, and equal to those observed on the
    ``per_use`` arm (both steps) and, except under ``outer_never``, on ``mutation``'s own
    step."""
    dense, expert = _strategies("per_use")
    names = list(results["ref"][0][1])
    once = sgu.expected_once_per_step(
        names, dense=dense, expert=expert, world=WORLD, ep=EP, instances=INSTANCES
    )
    observed = sgu.observed_once_per_step(results["per_use"][1], STEPS)
    assert observed == once, sorted(observed ^ once)[:8]
    if mutation != "outer_never":  # it drops the DP-Outer reductions the observation needs
        mutant = sgu.observed_once_per_step(results[mutation][1], 1)
        assert mutant == once, sorted(mutant ^ once)[:8]
    print(f"[{_name(mutation)}] {len(once)} parameters reduced once per step", flush=True)
    return once


@pytest.mark.parametrize("mutation", ["overwrite", "skip_use", "outer_never"])
def test_mutations_fail_on_the_core_fused_weights(results, mutation):
    """``overwrite`` and ``outer_never`` reach every fused core weight. ``overwrite`` zeroes the
    target shard before every reduction; for a once-per-step bucket (a one-rank DP-Shard group)
    that is its whole persistent buffer before its only reduction, so those weights fail with an
    all-zero gradient: a degenerate detection, not overwrite-instead-of-add, and ``outer_never``
    is the check of that path. ``skip_use`` drops a per-use reduction, so it must fail every
    fused core weight whose bucket is reduced per use and leave those reduced once per step
    exactly equal to the ``per_use`` arm's. Once-per-step fused weights exist exactly when the
    experts' DP-Shard group is one rank."""
    _, reference = results["ref"][0]
    _, grads = results[mutation][0][0]
    report, _ = sgu.report_and_fail(_name(f"{mutation} (mutation) vs ref"), grads, reference)
    fused = results["fused_core_weights"]
    assert fused
    once = _once_per_step(results, mutation)
    excluded = [name for name in fused if name in once]
    if sgu.expected_dp_shard(WORLD, EP, INSTANCES, True) == 1:
        assert excluded, "no once-per-step fused weight at a one-rank expert DP-Shard"
    else:
        assert not excluded, excluded[:8]
    if mutation == "skip_use":
        sgu.assert_per_backward_mutation(report, grads, results["per_use"][0][0][1], fused, once)
        return
    passing = [name for name in fused if report[name]["ok"]]
    assert not passing, sgu.mbs.worst(report, passing)
