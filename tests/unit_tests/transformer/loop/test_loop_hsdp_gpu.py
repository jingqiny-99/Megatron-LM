# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-FSDP HSDP on a fixed-depth DSv4 loop (GPU; the lifted guard ``FSDP_HSDP`` of
``validate_loop_fsdp_config``).

Two training steps with an SGD step between them (no momentum: the step-2 weights are a plain
function of step 1's gradients) of the model of ``strategy_grad_utils`` (a core of 8 sublayers
run exactly 4 times, block + edge ``loop_recompute`` with stride 4, three microbatches of one
row per rank, per-token loss, ``finalize_model_grads``) on 4 GPUs (or 8) as two
distributed-optimizer instances (``num_distributed_optimizer_instances = 2``: DP-Shard 2 x
DP-Outer 2), from the same weights and rows in every arm:

* ``ref``: Megatron DDP over the whole data-parallel group (fp32 gradient all-reduce);
* ``hsdp``: Megatron-FSDP v1 ZeRO-3 (``optim_grads_params``) inside each instance, replicated
  across them (``outer_dp_sharding_strategy='no_shard'``), accumulate mode, fp32 main and
  communication gradients;
* ``hfsdp``: the same with the optimizer state sharded across the instances too
  (``outer_dp_sharding_strategy='optim'``);
* ``hfsdp_recipe``: the Pro FSDP recipes' layout: dense ZeRO-3, experts ZeRO-2
  (``expert_data_parallel_sharding_strategy='optim_grads'``), outer ``optim``. At the default
  EP (``W / 2``) the expert DP-Shard group is one rank, as at EP 128 on 256 GPUs with two
  instances: the experts' buckets are reduced once per step, over DP-Outer;
* ``hsdp_expert_optim``: outer ``no_shard`` with experts ZeRO-1 (``optim``: an unsharded
  gradient buffer of which only the rank's shard is reduced).

Megatron-FSDP reduces a parameter's gradient from its ``register_post_accumulate_grad_hook``,
which fires once per backward, after the last of the 4 uses (autograd's ``AccumulateGrad``):
every microbatch reduce-scatters a bucket once over DP-Shard into the accumulating shard, and
the last microbatch's reduction also reduces that shard over DP-Outer (``outer_fsdp_group_grad_
reduce``): once per step, after the last use. Step 2 also runs the parameter all-gathers of the
updated weights (under HFSDP: the DP-Outer and DP-Shard gathers).

* Every parameter's reduced fp32 gradient of both steps matches ``ref``'s: cosine >= 0.9999 and
  relative difference <= 1e-3 (a gradient zero to fp32 precision: absolute, at most 1e-6 of the
  whole gradient's norm); step 1's losses and every row's LM loss equal ``ref``'s, step 2's
  losses are within 1e-5.
* The recorded reductions, every step: every gradient bucket is reduced over DP-Shard exactly
  once per microbatch (not once per use; a bucket whose DP-Shard group is one rank once per
  step), and over DP-Outer exactly once per step, in the last microbatch.
* ``cpu_adam`` (smoke test of the recipes' optimizer): ``hfsdp_recipe`` with Adam offloaded to
  the host (``optimizer_cpu_offload``, fraction 1.0, precision aware) builds and steps; step 1's
  gradients pass the gate against a DDP run, and its step-2 losses are within 1e-3 of that run
  stepped with the GPU Adam (a different update kernel) and closer to it than to the same run
  with a learning rate of zero (the update happened).
* Teeth, each must fail (step 1): ``overwrite`` (accumulate mode off, the loop guard bypassed:
  each use overwrites the bucket, so the core's fused-wgrad weights keep one round);
  ``outer_each`` (the DP-Outer reduction on every microbatch: earlier microbatches' sums are
  reduced again); ``outer_never`` (no DP-Outer reduction: each instance keeps its own half).

Launch: ``torchrun --nproc-per-node 4 -m pytest -q tests/unit_tests/transformer/loop/
test_loop_hsdp_gpu.py``. ``LOOP_HSDP_EP`` sets the expert-parallel size (default ``W / 2``, the
recipes' expert layout; run it also with ``LOOP_HSDP_EP=1``, where the experts are sharded over
DP-Shard as well). ``MBS_REPORT_DIR`` dumps every per-parameter report.
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
EP = int(os.environ.get("LOOP_HSDP_EP", str(max(WORLD // INSTANCES, 1))))
STEPS = 2
ADAM_LR = 1e-3
CPU_ADAM_LOSS_RELATIVE = 1e-3
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


def _hsdp(model, outer, accumulate=True, expert=None):
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel

    return FullyShardedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            use_distributed_optimizer=True,
            data_parallel_sharding_strategy="optim_grads_params",
            expert_data_parallel_sharding_strategy=expert,
            num_distributed_optimizer_instances=INSTANCES,
            outer_dp_sharding_strategy=outer,
            megatron_fsdp_accumulate_main_grad=accumulate,
            megatron_fsdp_main_grads_dtype=torch.float32,
            megatron_fsdp_grad_comm_dtype=torch.float32,
        ),
        model,
    )


def _adam(lr=ADAM_LR, cpu_offload=False):
    """Adam as the recipes set it; ``cpu_offload``: on the host (HybridDeviceOptimizer, the
    recipes' --optimizer-cpu-offload with fraction 1.0 and the precision-aware optimizer)."""
    fields = dict(optimizer="adam", lr=lr, adam_beta1=0.9, adam_beta2=0.95)
    if cpu_offload:
        fields.update(
            optimizer_cpu_offload=True,
            optimizer_offload_fraction=1.0,
            use_precision_aware_optimizer=True,
        )
    return fields


class _ReductionRecorder:
    """Records every bucket-group reduction of Megatron-FSDP (``GradReducePipeline.
    _bucket_group_gradient_reduce``) with the step, the microbatch and whether it reduced over
    DP-Outer; ``outer`` (``"each"`` / ``"never"``) overrides that flag for the mutations."""

    def __init__(self, outer=None):
        from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
            GradReducePipeline,
        )

        self.cls = GradReducePipeline
        self.original = GradReducePipeline._bucket_group_gradient_reduce
        self.step = 0
        self.microbatch = -1
        self.calls = []  # (step, microbatch, bucket ids, outer)
        recorder = self

        def recording(
            pipeline, bucket_group, async_op=False, outer_fsdp_group_grad_reduce=False, **kw
        ):
            if outer == "each":
                outer_fsdp_group_grad_reduce = True
            elif outer == "never":
                outer_fsdp_group_grad_reduce = False
            recorder.calls.append(
                (
                    recorder.step,
                    recorder.microbatch,
                    tuple(bucket_group),
                    bool(outer_fsdp_group_grad_reduce),
                )
            )
            return recorder.original(
                pipeline,
                bucket_group,
                async_op=async_op,
                outer_fsdp_group_grad_reduce=outer_fsdp_group_grad_reduce,
                **kw,
            )

        self.recording = recording

    def on_microbatch(self, index):
        self.microbatch = index

    def __enter__(self):
        self.cls._bucket_group_gradient_reduce = self.recording
        return self

    def __exit__(self, *exc):
        self.cls._bucket_group_gradient_reduce = self.original


def _run_ddp(state, steps=STEPS, **optimizer_fields):
    """``steps`` steps of the DDP reference; returns ``[(step result, grads)]`` and the core's
    fused-wgrad weight names."""
    model = sgu.build_model(state, expert_model_parallel_size=EP)
    ddp = _ddp(model)
    opt = sgu.optimizer(ddp, **optimizer_fields)
    out = []
    for step in range(steps):
        opt.zero_grad()
        result = sgu.run_step(ddp, model, sgu.make_rows(step))
        out.append((result, sgu.ddp_grads(ddp, model)))
        if step + 1 < steps:
            sgu.step_optimizer(opt)
    fused = sgu.fused_core_weights(model)
    del opt, ddp, model
    torch.cuda.empty_cache()
    return out, fused


def _run_fsdp(
    state,
    outer,
    *,
    accumulate=True,
    expert=None,
    outer_override=None,
    steps=STEPS,
    optimizer_fields=None,
):
    """``steps`` steps of one Megatron-FSDP arm; returns ``[(step result, grads)]`` and the
    reduction record."""
    from megatron.core.transformer.loop import guards

    original_guard = guards.validate_loop_fsdp_config
    if not accumulate:
        # The mutation: the loop guard would refuse it (FSDP_OVERWRITE_MAIN_GRAD).
        guards.validate_loop_fsdp_config = lambda config, ddp_config: None
    try:
        model = sgu.build_model(state, expert_model_parallel_size=EP)
        fsdp = _hsdp(model, outer, accumulate=accumulate, expert=expert)
    finally:
        guards.validate_loop_fsdp_config = original_guard
    opt = sgu.optimizer(fsdp, **(optimizer_fields or {}))
    out = []
    with _ReductionRecorder(outer=outer_override) as recorder:
        for step in range(steps):
            recorder.step = step
            opt.zero_grad()
            result = sgu.run_step(
                fsdp, model, sgu.make_rows(step), on_microbatch=recorder.on_microbatch
            )
            out.append((result, sgu.fsdp_grads(fsdp)))
            if step + 1 < steps:
                sgu.step_optimizer(opt)
    # {bucket id: DP-Shard size}: a bucket whose DP-Shard group is one rank shards nothing and
    # is reduced once per step (MegatronFSDP._reduces_grad_every_backward).
    buckets = {
        bucket_id: fsdp.module.grad_reduce_pipeline.get_fsdp_buffer(
            bucket_id
        ).data_parallel_group.size()
        for bucket_id, group in enumerate(fsdp.param_and_grad_buffer.parameter_groups)
        if group.main_grad_buffer is not None and group.requires_grad
    }
    fsdp.stop_communication()
    del opt, fsdp, model
    torch.cuda.empty_cache()
    return out, dict(calls=recorder.calls, buckets=buckets)


ARMS = {
    "hsdp": dict(outer="no_shard"),
    "hfsdp": dict(outer="optim"),
    "hfsdp_recipe": dict(outer="optim", expert="optim_grads"),
    "hsdp_expert_optim": dict(outer="no_shard", expert="optim"),
}


@pytest.fixture(scope="module")
def results():
    """Every arm, run once from the same weights and rows."""
    state = sgu.clone_state(sgu.build_model(expert_model_parallel_size=EP))
    out = {}
    out["ref"], out["fused_core_weights"] = _run_ddp(state)
    for arm, kwargs in ARMS.items():
        out[arm] = _run_fsdp(state, **kwargs)
    out["ref_adam"], _ = _run_ddp(state, **_adam())
    out["ref_frozen"], _ = _run_ddp(state, **_adam(lr=0.0))
    out["cpu_adam"] = _run_fsdp(
        state, **ARMS["hfsdp_recipe"], optimizer_fields=_adam(cpu_offload=True)
    )
    out["overwrite"] = _run_fsdp(state, "no_shard", accumulate=False, steps=1)
    out["outer_each"] = _run_fsdp(state, "no_shard", outer_override="each", steps=1)
    out["outer_never"] = _run_fsdp(state, "no_shard", outer_override="never", steps=1)
    return out


def _name(text):
    return sgu.tag(f"HSDP EP{EP}", text)


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("step", range(STEPS))
def test_gradients_match_the_reference(results, arm, step):
    _, reference = results["ref"][step]
    _, grads = results[arm][0][step]
    report, failing = sgu.report_and_fail(_name(f"{arm} step {step + 1} vs ref"), grads, reference)
    assert not failing, sgu.mbs.worst(report, failing)


@pytest.mark.parametrize("arm", list(ARMS))
def test_losses_match_the_reference(results, arm):
    for step in range(STEPS):
        reference, _ = results["ref"][step]
        result, _ = results[arm][0][step]
        sgu.assert_step_losses_match(step, result, reference, _name(f"{arm} step {step + 1}"))


@pytest.mark.parametrize("arm", list(ARMS))
def test_reductions_once_per_microbatch_and_outer_once_per_step(results, arm):
    """Every gradient bucket, every step: one DP-Shard reduction per microbatch (after its
    parameters' last use, not per use), and one DP-Outer reduction per step, in the last
    microbatch."""
    _, record = results[arm]
    assert record["buckets"], record
    for step in range(STEPS):
        per_microbatch = collections.Counter()
        outer = collections.Counter()
        for call_step, microbatch, bucket_ids, with_outer in record["calls"]:
            if call_step != step:
                continue
            for bucket_id in bucket_ids:
                per_microbatch[(microbatch, bucket_id)] += 1
                if with_outer:
                    outer[(microbatch, bucket_id)] += 1
        print(
            f"[{_name(f'{arm} step {step + 1}')}] reductions {dict(per_microbatch)} outer "
            f"{dict(outer)}",
            flush=True,
        )
        for bucket_id, shard_size in record["buckets"].items():
            expected = [1] * sgu.ROWS if shard_size > 1 else [0] * (sgu.ROWS - 1) + [1]
            got = [per_microbatch[(microbatch, bucket_id)] for microbatch in range(sgu.ROWS)]
            assert got == expected, (arm, step, bucket_id, shard_size, got)
            assert outer[(sgu.ROWS - 1, bucket_id)] == 1, (arm, step, bucket_id)
            earlier = sum(outer[(m, bucket_id)] for m in range(sgu.ROWS - 1))
            assert earlier == 0, (arm, step, bucket_id, earlier)


def test_recipe_arm_has_one_rank_expert_shards_at_the_default_ep(results):
    """At EP = W / 2 the recipes' expert layout holds: some buckets (the experts') have a
    one-rank DP-Shard group, so the gate checked the once-per-step expert path."""
    _, record = results["hfsdp_recipe"]
    sizes = sorted(set(record["buckets"].values()))
    print(f"[{_name('hfsdp_recipe')}] DP-Shard sizes {sizes}", flush=True)
    if EP == WORLD // INSTANCES:
        assert 1 in sizes and max(sizes) > 1, sizes


def test_cpu_offloaded_adam_steps_on_the_recipe_layout(results):
    """Smoke test of the recipes' optimizer on the recipes' layout (see the module docstring)."""
    _, reference = results["ref_adam"][0]
    _, grads = results["cpu_adam"][0][0]
    report, failing = sgu.report_and_fail(_name("cpu_adam step 1 vs ref"), grads, reference)
    assert not failing, sgu.mbs.worst(report, failing)
    sgu.assert_losses_match(results["cpu_adam"][0][0][0], results["ref_adam"][0][0], _name("cpu"))
    step2 = results["cpu_adam"][0][1][0]
    to_adam = sgu.mbs.relative_loss_differences(step2.losses, results["ref_adam"][1][0].losses)
    to_frozen = sgu.mbs.relative_loss_differences(step2.losses, results["ref_frozen"][1][0].losses)
    print(f"[{_name('cpu_adam step 2')}] rel to GPU Adam {to_adam}, to lr 0 {to_frozen}")
    assert max(to_adam.values()) <= CPU_ADAM_LOSS_RELATIVE, to_adam
    assert max(to_adam.values()) < max(to_frozen.values()), (to_adam, to_frozen)


def test_overwrite_mutation_fails_on_the_core_fused_weights(results):
    """Without accumulate mode each use overwrites the gradient bucket: every core weight whose
    weight gradient TE accumulates in place keeps one round's gradient."""
    _, reference = results["ref"][0]
    _, grads = results["overwrite"][0][0]
    report, failing = sgu.report_and_fail(_name("overwrite (mutation) vs ref"), grads, reference)
    fused = results["fused_core_weights"]
    assert fused
    passing = [name for name in fused if report[name]["ok"]]
    assert not passing, sgu.mbs.worst(report, passing)


@pytest.mark.parametrize("mutation", ["outer_each", "outer_never"])
def test_outer_reduction_mutations_fail_the_gate(results, mutation):
    _, reference = results["ref"][0]
    _, grads = results[mutation][0][0]
    report, failing = sgu.report_and_fail(_name(f"{mutation} (mutation) vs ref"), grads, reference)
    assert failing, f"the {mutation} mutation passed the gate: the test has no teeth"
    print(f"[{_name(mutation)}] failing {len(failing)} of {len(report)}", flush=True)
