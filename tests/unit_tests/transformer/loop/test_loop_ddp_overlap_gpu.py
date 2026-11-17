# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DDP communication overlap on a fixed-depth DSv4 loop (GPU; the lifted guard
``validate_loop_ddp_config``).

Two training steps (SGD without momentum, run as Adam: ``strategy_grad_utils.SGD_AS_ADAM``; so
the second step's weights are a plain function of the first step's gradients) of the model of ``strategy_grad_utils`` (a core of 8 sublayers run exactly 4 times,
block + edge ``loop_recompute`` with stride 4, three microbatches of one row per rank, per-token
loss, ``finalize_model_grads``), from the same weights and rows in every arm:

* ``ref``: DDP with the distributed optimizer, no overlap (one bucket, synchronous
  reduce-scatter after the backward, synchronous parameter all-gather in the optimizer step);
* ``ovl``: DDP with the distributed optimizer, ``overlap_grad_reduce`` and
  ``overlap_param_gather``, small buckets (several bucket groups);
* ``ovl_ddp``: plain DDP (all-reduce), ``overlap_grad_reduce``, small buckets.

Overlapped grad reduce starts a bucket group's reduction from the gradient hook once every
parameter of the group reached the grad-ready count recorded in the first step (step 1 records,
step 2 overlaps). A parameter hooked through autograd is ready once per backward, after its
last use (``AccumulateGrad`` waits for all 4 rounds' gradients, through the non-reentrant block
checkpoints). Overlapped parameter gather waits on a bucket's all-gather in the first forward
pre-hook of a module that owns one of its parameters; later rounds and the recompute find the
bucket published.

* Every parameter's reduced fp32 gradient of steps 1 and 2 matches ``ref``'s: cosine >= 0.9999
  and relative difference <= 1e-3 (a gradient that is zero to fp32 precision: absolute, at most
  1e-6 of the whole gradient's norm). Both arms run the same bf16 kernels on the same weights;
  they differ in bucket boundaries (the fp32 association of the reduction).
* The losses of both steps (and every row's LM loss) equal ``ref``'s.
* Step 2 of ``ovl`` / ``ovl_ddp``: every bucket group's reduction is started from the gradient
  hook (overlapped), exactly once; the recorded grad-ready counts are printed (1 per backward
  for every parameter: no module GTP here).
* Teeth, each must fail on step 2: ``early``, a bucket group reduced as soon as its first
  parameter is ready (before the other parameters' last uses); ``stale``, step 2's parameter
  all-gathers skipped (the forward reads last step's weights outside the local shard).
* Diagnostics, not gated: ``ref_buckets`` (``ref`` with the overlap arms' buckets, no overlap)
  and ``ovl_rs`` (overlapped reduce-scatter, no overlapped parameter gather) gated at step 2
  against ``ref`` and printed, with every arm's step-2 weights compared to ``ref``'s.

Launch: ``torchrun --nproc-per-node 2 -m pytest -q tests/unit_tests/transformer/loop/
test_loop_ddp_overlap_gpu.py`` (2 or 4 GPUs; ``LOOP_DDP_EP`` sets the expert-parallel size,
default the world size). ``MBS_REPORT_DIR`` dumps every per-parameter report.
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
EP = int(os.environ.get("LOOP_DDP_EP", str(WORLD)))
pytestmark = pytest.mark.skipif(
    not CUDA or WORLD < 2 or WORLD % EP != 0 or sgu.NUM_EXPERTS % EP != 0,
    reason="DSv4 / MoE layers need CUDA; overlap needs data parallelism (W >= 2)",
)

BUCKET_SIZE = 200_000  # elements: several bucket groups for the tiny model
STEPS = 2
STEP2_LOSS_RELATIVE = 1e-5  # step 2 runs on weights updated from fp32-reassociated gradients


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    if not CUDA:
        yield
        return
    with determinism_env(), model_parallel_once(expert_model_parallel_size=EP):
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(False)


def _ddp(model, *, overlap, distributed_optimizer, param_gather=None, small_buckets=None):
    """``param_gather`` and ``small_buckets`` (both default ``overlap``) split the overlap arms'
    settings for the diagnostic arms."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.process_groups_config import ProcessGroupCollection

    param_gather = overlap if param_gather is None else param_gather
    small_buckets = overlap if small_buckets is None else small_buckets
    return DistributedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            overlap_grad_reduce=overlap,
            overlap_param_gather=param_gather and distributed_optimizer,
            use_distributed_optimizer=distributed_optimizer,
            grad_reduce_in_fp32=True,
            bucket_size=BUCKET_SIZE if small_buckets else None,
        ),
        model,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )


class _DispatchRecorder:
    """Records, per step, which bucket groups started their reduction from the gradient hook
    (``register_grad_ready``), and the grad-ready counts of the last microbatch."""

    def __init__(self):
        from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup

        self.cls = _ParamAndGradBucketGroup
        self.original = _ParamAndGradBucketGroup.register_grad_ready
        self.step = 0
        self.hook_dispatches = collections.Counter()  # (step, id(group)) -> count
        recorder = self

        def recording(group, param, force_all_reduce=False):
            before = group.grad_reduce_handle
            recorder.original(group, param, force_all_reduce)
            if before is None and group.grad_reduce_handle is not None:
                recorder.hook_dispatches[(recorder.step, id(group))] += 1

        self.recording = recording

    def __enter__(self):
        self.cls.register_grad_ready = self.recording
        return self

    def __exit__(self, *exc):
        self.cls.register_grad_ready = self.original


def _early_dispatch():
    """The mutation: a bucket group's reduction starts when its first parameter becomes ready
    in the last microbatch (after step 1), not when all of them reached their counts."""

    def early(group, param, force_all_reduce=False):
        if not group.is_last_microbatch:
            return
        counts = group.per_param_grad_ready_counts
        counts[param] = counts.get(param, 0) + 1
        if (
            not group.is_first_batch
            and group.grad_reduce_handle is None
            and not group.grad_reduce_finished
            and len(counts) == 1
            and counts[param] == 1
        ):
            group.start_grad_sync(force_all_reduce=force_all_reduce)

    return early


def _run(state, *, overlap, distributed_optimizer, mutation=None, **ddp_options):
    """``STEPS`` steps of one arm; returns ``[(step result, grads)]`` per step and the dispatch
    record, with the bf16 weights the last step ran on (``weights``)."""
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup

    model = sgu.build_model(state, expert_model_parallel_size=EP)
    ddp = _ddp(model, overlap=overlap, distributed_optimizer=distributed_optimizer, **ddp_options)
    optimizer = sgu.optimizer(ddp)
    out = []
    original_start_param_sync = _ParamAndGradBucketGroup.start_param_sync
    with _DispatchRecorder() as recorder:
        if mutation == "early":
            _ParamAndGradBucketGroup.register_grad_ready = _early_dispatch()
        try:
            for step in range(STEPS):
                recorder.step = step
                if mutation == "stale" and step == 1:

                    def skipped(group, force_sync=False):
                        group.param_gather_dispatched = True

                    _ParamAndGradBucketGroup.start_param_sync = skipped
                optimizer.zero_grad()
                result = sgu.run_step(ddp, model, sgu.make_rows(step))
                grads = sgu.ddp_grads(ddp, model)
                out.append((result, grads))
                if step + 1 < STEPS:
                    update_successful, _, _ = optimizer.step()
                    assert update_successful
        finally:
            _ParamAndGradBucketGroup.start_param_sync = original_start_param_sync
    groups = ddp.bucket_groups + ddp.expert_parallel_bucket_groups
    golden = [sorted(set(group.golden_per_param_grad_ready_counts.values())) for group in groups]
    dispatches = {
        step: [recorder.hook_dispatches.get((step, id(group)), 0) for group in groups]
        for step in range(STEPS)
    }
    # Every bucket's all-gather has been waited on by the end of the last step's forward, so
    # these are the weights that step ran on (without overlap: gathered in the optimizer step).
    weights = {name: param.detach().clone() for name, param in model.named_parameters()}
    del optimizer, ddp, model
    torch.cuda.empty_cache()
    return out, dict(dispatches=dispatches, golden=golden, groups=len(groups), weights=weights)


@pytest.fixture(scope="module")
def results():
    """Every arm, run once from the same weights and rows."""
    state = sgu.clone_state(sgu.build_model(expert_model_parallel_size=EP))
    out = {}
    out["ref"] = _run(state, overlap=False, distributed_optimizer=True)
    out["ovl"] = _run(state, overlap=True, distributed_optimizer=True)
    out["ovl_ddp"] = _run(state, overlap=True, distributed_optimizer=False)
    out["early"] = _run(state, overlap=True, distributed_optimizer=True, mutation="early")
    out["stale"] = _run(state, overlap=True, distributed_optimizer=True, mutation="stale")
    # Diagnostics (not gated): the overlap arms' bucket boundaries without any overlap, and the
    # overlapped reduce-scatter without the overlapped parameter gather.
    out["ref_buckets"] = _run(state, overlap=False, distributed_optimizer=True, small_buckets=True)
    out["ovl_rs"] = _run(state, overlap=True, distributed_optimizer=True, param_gather=False)
    return out


def _name(text):
    return sgu.tag(f"DDP-overlap EP{EP}", text)


@pytest.mark.parametrize("arm", ["ovl", "ovl_ddp"])
@pytest.mark.parametrize("step", range(STEPS))
def test_gradients_match_the_reference(results, arm, step):
    _, reference = results["ref"][0][step]
    _, grads = results[arm][0][step]
    report, failing = sgu.report_and_fail(_name(f"{arm} step {step + 1} vs ref"), grads, reference)
    assert not failing, sgu.mbs.worst(report, failing)


@pytest.mark.parametrize("arm", ["ovl", "ovl_ddp"])
def test_losses_match_the_reference(results, arm):
    for step in range(STEPS):
        reference, _ = results["ref"][0][step]
        result, _ = results[arm][0][step]
        name = _name(f"{arm} step {step + 1}")
        if step == 0:
            sgu.assert_losses_match(result, reference, name)
        else:
            rel = sgu.mbs.relative_loss_differences(result.losses, reference.losses)
            print(f"[{name}] losses {result.losses} rel {rel}", flush=True)
            assert all(value <= STEP2_LOSS_RELATIVE for value in rel.values()), rel


@pytest.mark.parametrize("arm", ["ovl", "ovl_ddp"])
def test_every_bucket_group_reduces_once_from_its_hook(results, arm):
    """Step 1 records the counts (no overlapped dispatch); in step 2 every bucket group starts
    its reduction from the gradient hook exactly once. The recorded counts are printed (an
    autograd-hooked parameter is expected to be ready once per backward: no module GTP in this
    model)."""
    _, record = results[arm]
    print(f"[{_name(arm)}] {record}", flush=True)
    assert record["groups"] > 1, record
    assert record["dispatches"][0] == [0] * record["groups"], record
    assert record["dispatches"][1] == [1] * record["groups"], record
    assert all(values and values[0] >= 1 for values in record["golden"]), record["golden"]


@pytest.mark.parametrize("arm", ["ref_buckets", "ovl_rs", "ovl", "ovl_ddp"])
def test_report_step_two_diagnostics(results, arm):
    """Diagnostic only (prints, asserts nothing): which part of an overlap arm moves step 2.

    * ``ref_buckets``: ``ref`` with the overlap arms' bucket boundaries and no overlap, i.e.
      how far step 2 moves from a step-1 gradient that differs only by fp32 reassociation;
    * ``ovl_rs``: the overlapped reduce-scatter without the overlapped parameter gather;
    * every arm: the bf16 weights its last step ran on against ``ref``'s (equal weights with
      different losses on some ranks would mean a forward read weights other than these).
    """
    _, reference = results["ref"][0][1]
    _, grads = results[arm][0][1]
    report, failing = sgu.report_and_fail(
        _name(f"{arm} step 2 vs ref (diagnostic)"), grads, reference
    )
    print(f"[{_name(arm)}] step 2 failing {len(failing)} of {len(report)}", flush=True)
    ref_weights = results["ref"][1]["weights"]
    weights = results[arm][1]["weights"]
    differing = {
        name: int((weights[name] != ref_weights[name]).sum().item())
        for name in ref_weights
        if not torch.equal(weights[name], ref_weights[name])
    }
    worst = sorted(differing.items(), key=lambda item: -item[1])[:8]
    print(
        f"[{_name(arm)}] step-2 weights differ from ref in {len(differing)} of "
        f"{len(ref_weights)} params ({sum(differing.values())} elements): {worst}",
        flush=True,
    )
    for step in range(STEPS):
        rel = sgu.mbs.relative_loss_differences(
            results[arm][0][step][0].losses, results["ref"][0][step][0].losses
        )
        print(f"[{_name(f'{arm} step {step + 1}')}] losses rel {rel} (diagnostic)", flush=True)


@pytest.mark.parametrize("mutation", ["early", "stale"])
def test_mutations_fail_the_gate(results, mutation):
    """A reduction started before the last use of every parameter in its bucket group, or a
    forward on stale gathered weights, must fail the step-2 gradient gate."""
    _, reference = results["ref"][0][1]
    _, grads = results[mutation][0][1]
    report, failing = sgu.report_and_fail(
        _name(f"{mutation} (mutation) step 2 vs ref"), grads, reference
    )
    assert failing, f"the {mutation} mutation passed the gate: the test has no teeth"
    print(f"[{_name(mutation)}] failing {len(failing)} of {len(report)}", flush=True)
