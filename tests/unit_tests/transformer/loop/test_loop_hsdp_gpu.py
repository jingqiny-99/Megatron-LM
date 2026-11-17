# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-FSDP HSDP on a fixed-depth DSv4 loop (GPU; the lifted guard ``FSDP_HSDP`` of
``validate_loop_fsdp_config``).

One training step of the model of ``strategy_grad_utils`` (a core of 8 sublayers run exactly 4
times, block + edge ``loop_recompute`` with stride 4, three microbatches of one row per rank,
per-token loss, ``finalize_model_grads``) on 4 GPUs (or 8) as two distributed-optimizer
instances (``num_distributed_optimizer_instances = 2``: DP-Shard 2 x DP-Outer 2), from the same
weights and rows in every arm:

* ``ref``: Megatron DDP over the whole data-parallel group (fp32 gradient all-reduce);
* ``hsdp``: Megatron-FSDP v1 ZeRO-3 (``optim_grads_params``) inside each instance, replicated
  across them (``outer_dp_sharding_strategy='no_shard'``), accumulate mode, fp32 main and
  communication gradients;
* ``hfsdp``: the same with the optimizer state sharded across the instances too
  (``outer_dp_sharding_strategy='optim'``).

Megatron-FSDP reduces a parameter's gradient from its ``register_post_accumulate_grad_hook``,
which fires once per backward, after the last of the 4 uses (autograd's ``AccumulateGrad``):
every microbatch reduce-scatters a bucket once over DP-Shard into the accumulating shard, and
the last microbatch's reduction also reduces that shard over DP-Outer (``outer_fsdp_group_grad_
reduce``): once per step, after the last use.

* Every parameter's reduced fp32 gradient matches ``ref``'s: cosine >= 0.9999 and relative
  difference <= 1e-3 (a gradient zero to fp32 precision: absolute, at most 1e-6 of the whole
  gradient's norm); the losses and every row's LM loss equal ``ref``'s.
* The recorded reductions: every gradient bucket is reduced over DP-Shard exactly once per
  microbatch (not once per use), and over DP-Outer exactly once per step, in the last
  microbatch.
* Teeth, each must fail: ``overwrite`` (accumulate mode off, the loop guard bypassed: each use
  overwrites the bucket, so the core's fused-wgrad weights keep one round); ``outer_each``
  (the DP-Outer reduction on every microbatch: earlier microbatches' sums are reduced again);
  ``outer_never`` (no DP-Outer reduction: each instance keeps its own half).

Launch: ``torchrun --nproc-per-node 4 -m pytest -q tests/unit_tests/transformer/loop/
test_loop_hsdp_gpu.py``. ``LOOP_HSDP_EP`` sets the expert-parallel size (default 1).
``MBS_REPORT_DIR`` dumps every per-parameter report.
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
EP = int(os.environ.get("LOOP_HSDP_EP", "1"))
pytestmark = pytest.mark.skipif(
    not CUDA
    or WORLD < 4
    or WORLD % (2 * INSTANCES) != 0
    or WORLD % EP != 0
    or sgu.NUM_EXPERTS % EP != 0,
    reason="HSDP needs CUDA and DP-Shard >= 2 in each of the 2 instances (W = 4 or 8)",
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


def _hsdp(model, outer, accumulate=True):
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel

    return FullyShardedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            use_distributed_optimizer=True,
            data_parallel_sharding_strategy="optim_grads_params",
            num_distributed_optimizer_instances=INSTANCES,
            outer_dp_sharding_strategy=outer,
            megatron_fsdp_accumulate_main_grad=accumulate,
            megatron_fsdp_main_grads_dtype=torch.float32,
            megatron_fsdp_grad_comm_dtype=torch.float32,
        ),
        model,
    )


class _ReductionRecorder:
    """Records every bucket-group reduction of Megatron-FSDP (``GradReducePipeline.
    _bucket_group_gradient_reduce``) with the microbatch and whether it reduced over DP-Outer;
    ``outer`` (``"each"`` / ``"never"``) overrides that flag for the mutations."""

    def __init__(self, outer=None):
        from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
            GradReducePipeline,
        )

        self.cls = GradReducePipeline
        self.original = GradReducePipeline._bucket_group_gradient_reduce
        self.microbatch = -1
        self.calls = []  # (microbatch, bucket ids, outer)
        recorder = self

        def recording(pipeline, bucket_group, async_op=False, outer_fsdp_group_grad_reduce=False):
            if outer == "each":
                outer_fsdp_group_grad_reduce = True
            elif outer == "never":
                outer_fsdp_group_grad_reduce = False
            recorder.calls.append(
                (recorder.microbatch, tuple(bucket_group), bool(outer_fsdp_group_grad_reduce))
            )
            return recorder.original(
                pipeline,
                bucket_group,
                async_op=async_op,
                outer_fsdp_group_grad_reduce=outer_fsdp_group_grad_reduce,
            )

        self.recording = recording

    def on_microbatch(self, index):
        self.microbatch = index

    def __enter__(self):
        self.cls._bucket_group_gradient_reduce = self.recording
        return self

    def __exit__(self, *exc):
        self.cls._bucket_group_gradient_reduce = self.original


def _run_fsdp(state, outer, *, accumulate=True, outer_override=None):
    from megatron.core.transformer.loop import guards

    original_guard = guards.validate_loop_fsdp_config
    if not accumulate:
        # The mutation: the loop guard would refuse it (FSDP_OVERWRITE_MAIN_GRAD).
        guards.validate_loop_fsdp_config = lambda config, ddp_config: None
    try:
        model = sgu.build_model(state, expert_model_parallel_size=EP)
        fsdp = _hsdp(model, outer, accumulate=accumulate)
    finally:
        guards.validate_loop_fsdp_config = original_guard
    with _ReductionRecorder(outer=outer_override) as recorder:
        step = sgu.run_step(fsdp, model, sgu.make_rows(), on_microbatch=recorder.on_microbatch)
    grads = sgu.fsdp_grads(fsdp)
    # {bucket id: DP-Shard size}: a bucket whose DP-Shard group is one rank shards nothing and
    # is reduced once per step (MegatronFSDP._reduces_grad_every_backward).
    buckets = {
        bucket_id: fsdp.grad_reduce_pipeline.get_fsdp_buffer(bucket_id).data_parallel_group.size()
        for bucket_id, group in enumerate(fsdp.param_and_grad_buffer.parameter_groups)
        if group.main_grad_buffer is not None and group.requires_grad
    }
    fsdp.stop_communication()
    del fsdp, model
    torch.cuda.empty_cache()
    return step, grads, dict(calls=recorder.calls, buckets=buckets)


@pytest.fixture(scope="module")
def results():
    """Every arm, run once from the same weights and rows."""
    state = sgu.clone_state(sgu.build_model(expert_model_parallel_size=EP))
    out = {}
    model = sgu.build_model(state, expert_model_parallel_size=EP)
    ddp = _ddp(model)
    step = sgu.run_step(ddp, model, sgu.make_rows())
    out["ref"] = (step, sgu.ddp_grads(ddp, model), None)
    out["fused_core_weights"] = sgu.fused_core_weights(model)
    del ddp, model
    out["hsdp"] = _run_fsdp(state, "no_shard")
    out["hfsdp"] = _run_fsdp(state, "optim")
    out["overwrite"] = _run_fsdp(state, "no_shard", accumulate=False)
    out["outer_each"] = _run_fsdp(state, "no_shard", outer_override="each")
    out["outer_never"] = _run_fsdp(state, "no_shard", outer_override="never")
    return out


def _name(text):
    return sgu.tag(f"HSDP EP{EP}", text)


@pytest.mark.parametrize("arm", ["hsdp", "hfsdp"])
def test_gradients_match_the_reference(results, arm):
    _, reference, _ = results["ref"]
    _, grads, _ = results[arm]
    report, failing = sgu.report_and_fail(_name(f"{arm} vs ref"), grads, reference)
    assert not failing, sgu.mbs.worst(report, failing)


@pytest.mark.parametrize("arm", ["hsdp", "hfsdp"])
def test_losses_match_the_reference(results, arm):
    reference, _, _ = results["ref"]
    step, _, _ = results[arm]
    sgu.assert_losses_match(step, reference, _name(arm))


@pytest.mark.parametrize("arm", ["hsdp", "hfsdp"])
def test_reductions_once_per_microbatch_and_outer_once_per_step(results, arm):
    """Every gradient bucket: one DP-Shard reduction per microbatch (after its parameters' last
    use, not per use), and one DP-Outer reduction per step, in the last microbatch."""
    _, _, record = results[arm]
    per_microbatch = collections.Counter()
    outer = collections.Counter()
    for microbatch, bucket_ids, with_outer in record["calls"]:
        for bucket_id in bucket_ids:
            per_microbatch[(microbatch, bucket_id)] += 1
            if with_outer:
                outer[(microbatch, bucket_id)] += 1
    print(f"[{_name(arm)}] reductions {dict(per_microbatch)} outer {dict(outer)}", flush=True)
    assert record["buckets"], record
    for bucket_id, shard_size in record["buckets"].items():
        expected = [1] * sgu.ROWS if shard_size > 1 else [0] * (sgu.ROWS - 1) + [1]
        got = [per_microbatch[(microbatch, bucket_id)] for microbatch in range(sgu.ROWS)]
        assert got == expected, (arm, bucket_id, shard_size, got)
        assert outer[(sgu.ROWS - 1, bucket_id)] == 1, (arm, bucket_id)
        assert sum(outer[(m, bucket_id)] for m in range(sgu.ROWS - 1)) == 0, (arm, bucket_id)


def test_overwrite_mutation_fails_on_the_core_fused_weights(results):
    """Without accumulate mode each use overwrites the gradient bucket: every core weight whose
    weight gradient TE accumulates in place keeps one round's gradient."""
    _, reference, _ = results["ref"]
    _, grads, _ = results["overwrite"]
    report, failing = sgu.report_and_fail(_name("overwrite (mutation) vs ref"), grads, reference)
    fused = results["fused_core_weights"]
    assert fused
    passing = [name for name in fused if report[name]["ok"]]
    assert not passing, sgu.mbs.worst(report, passing)


@pytest.mark.parametrize("mutation", ["outer_each", "outer_never"])
def test_outer_reduction_mutations_fail_the_gate(results, mutation):
    _, reference, _ = results["ref"]
    _, grads, _ = results[mutation]
    report, failing = sgu.report_and_fail(_name(f"{mutation} (mutation) vs ref"), grads, reference)
    assert failing, f"the {mutation} mutation passed the gate: the test has no teeth"
    print(f"[{_name(mutation)}] failing {len(failing)} of {len(report)}", flush=True)
