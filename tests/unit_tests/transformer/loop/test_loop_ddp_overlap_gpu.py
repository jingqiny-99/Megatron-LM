# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DDP communication overlap on a fixed-depth DSv4 loop (GPU; the lifted guard
``validate_loop_ddp_config``).

Two training steps (SGD without momentum, run as Adam: ``strategy_grad_utils.SGD_AS_ADAM``) of
the model of ``strategy_grad_utils`` (a core of 8 sublayers run exactly 4 times, block + edge
``loop_recompute`` with stride 4, three microbatches of one row per rank, per-token loss,
``finalize_model_grads``), from the same weights and rows in every arm:

* ``ref``: DDP with the distributed optimizer, no overlap (one bucket, synchronous
  reduce-scatter after the backward, synchronous parameter all-gather in the optimizer step);
* ``ovl``: DDP with the distributed optimizer, ``overlap_grad_reduce`` and
  ``overlap_param_gather``, small buckets (several bucket groups);
* ``ovl_ddp``: plain DDP (all-reduce), ``overlap_grad_reduce``, small buckets;
* ``ovl_rs``: ``ovl`` without ``overlap_param_gather`` (the overlapped reduce-scatter, the
  parameter all-gather synchronous in the optimizer step).

Overlapped grad reduce starts a bucket group's reduction from the gradient hook once every
parameter of the group reached the grad-ready count recorded in the first step (step 1 records,
step 2 overlaps). A parameter hooked through autograd is ready once per backward, after its
last use (``AccumulateGrad`` waits for all 4 rounds' gradients, through the non-reentrant block
checkpoints). Overlapped parameter gather waits on a bucket's all-gather in the first forward
pre-hook of a module that owns one of its parameters; later rounds and the recompute find the
bucket published.

Step 2 is teacher-forced (``strategy_grad_utils``): after its step-1 gradient is read, every arm
and mutant takes ``ref``'s reduced step-1 gradient into its ``main_grad`` and then runs its own
optimizer step, gradient-buffer reset, parameter gathers and overlapped step 2.

* Every parameter's reduced fp32 gradient of steps 1 and 2 matches ``ref``'s: cosine >= 0.9999
  and relative difference <= 1e-3 (a gradient that is zero to fp32 precision: absolute, at most
  1e-6 of the whole gradient's norm). Before step 2 is gated, the precondition: the forced
  gradient reads back on every rank, and the weights at the end of step 2 (the step-1 update,
  installed and gathered) and the persistent buffers step 2 started with equal ``ref``'s
  exactly.
* The losses of both steps (and every row's LM loss) equal ``ref``'s within 1e-6.
* Step 2 of ``ovl`` / ``ovl_ddp`` / ``ovl_rs``: every bucket group's reduction is started from
  the gradient hook (overlapped), exactly once; the recorded grad-ready counts are printed (1
  per backward for every parameter: no module GTP here).
* Given the same step-1 gradient, the un-forced overlap path agrees at step 2:
  ``ovl_unforced`` (``ovl`` stepping on its own step-1 gradient) against ``ref_from_ovl``
  (``ref`` stepping on ``ovl_unforced``'s step-1 gradient), from an exactly equal state.
* Teeth, each must fail on step 2: ``early``, a bucket group reduced as soon as its first
  parameter is ready in the last microbatch (before the other parameters' last uses; it acts
  from step 2, the first overlapped step), from ``ref``'s state; ``stale``, step 2's parameter
  all-gathers skipped (the forward reads step 1's weights outside the local shard: every weight
  element is ``ref``'s or the initial one, and both occur).
* Diagnostics, not gated: the un-forced ``ovl_unforced`` against ``ref`` at step 2 (the
  weights that differ, with dtype and size of the difference, and the losses); ``ref_from_ovl``
  against ``ref`` (``ref``'s own step 2 moved by a step-1 gradient that differs only by fp32
  reassociation).

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
GATED = ["ovl", "ovl_ddp", "ovl_rs"]  # gated against ref at both steps
FORCED = GATED + ["early"]  # teacher-forced, held to ref's step-2 state


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


def _ddp(model, *, overlap, distributed_optimizer, param_gather=None):
    """``param_gather`` (default ``overlap``) splits the overlap arms' settings for the
    diagnostic arm. Without ``overlap_grad_reduce`` DDP uses one bucket whatever the size."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.process_groups_config import ProcessGroupCollection

    param_gather = overlap if param_gather is None else param_gather
    return DistributedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            overlap_grad_reduce=overlap,
            overlap_param_gather=param_gather and distributed_optimizer,
            use_distributed_optimizer=distributed_optimizer,
            grad_reduce_in_fp32=True,
            bucket_size=BUCKET_SIZE if overlap else None,
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


def _run(state, *, overlap, distributed_optimizer, mutation=None, forced=None, **ddp_options):
    """``STEPS`` steps of one arm; returns ``[(step result, grads)]`` per step and the record:
    the dispatches, the weights at the end of the last step (``weights``), the persistent buffers
    step 2 started with (``buffers``) and, with ``forced`` (a reference's step-1 gradients, the
    teacher forcing), the number of tensors whose reduced gradient read back after the forcing
    differs from the forced one (``readback``, agreed over the ranks)."""
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup

    model = sgu.build_model(state, expert_model_parallel_size=EP)
    ddp = _ddp(model, overlap=overlap, distributed_optimizer=distributed_optimizer, **ddp_options)
    optimizer = sgu.optimizer(ddp)
    out = []
    readback = buffers = None
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
                    if forced is not None:
                        sgu.force_main_grads(model, forced)
                        # The read-back gathers each rank's shard of grad_data: it holds when
                        # main_grad is a view of grad_data, the storage the distributed
                        # optimizer reads; the weights below carry the rest of the precondition.
                        readback = sgu.agreed(
                            len(sgu.mismatches(sgu.ddp_grads(ddp, model), forced))
                        )
                    update_successful, _, _ = optimizer.step()
                    assert update_successful
                    buffers = sgu.persistent_buffers(model)
        finally:
            _ParamAndGradBucketGroup.start_param_sync = original_start_param_sync
    groups = ddp.bucket_groups + ddp.expert_parallel_bucket_groups
    golden = [sorted(set(group.golden_per_param_grad_ready_counts.values())) for group in groups]
    dispatches = {
        step: [recorder.hook_dispatches.get((step, id(group)), 0) for group in groups]
        for step in range(STEPS)
    }
    # The weights at the end of the last step: the last update, installed and gathered (every
    # bucket's all-gather has been waited on by the end of the forward; without
    # overlap_param_gather it ran in the optimizer step). A gather that completed only after a
    # bucket's first use is not visible here; the step-2 gate and ``stale`` cover that.
    weights = {name: param.detach().clone() for name, param in model.named_parameters()}
    del optimizer, ddp, model
    torch.cuda.empty_cache()
    record = dict(dispatches=dispatches, golden=golden, groups=len(groups), weights=weights)
    record.update(buffers=buffers, readback=readback)
    return out, record


def _start_mismatches(record, reference):
    """Where the state ``record``'s step 2 ran from differs from ``reference``'s (local): the
    weights at the end of step 2 and the persistent buffers step 2 started with."""
    found = sgu.mismatches(record["weights"], reference["weights"])
    for name, value in sgu.mismatches(record["buffers"], reference["buffers"]).items():
        found[f"buffer {name}"] = value
    return found


@pytest.fixture(scope="module")
def results():
    """Every arm, run once from the same weights and rows; the forced arms and mutants take
    ``ref``'s step-1 gradient before their optimizer step. The preconditions are agreed over the
    ranks here (in a fixed order) and asserted by the tests."""
    state = sgu.clone_state(sgu.build_model(expert_model_parallel_size=EP))
    out = {"initial": state}
    out["ref"] = _run(state, overlap=False, distributed_optimizer=True)
    forced = out["ref"][0][0][1]
    out["ovl"] = _run(state, overlap=True, distributed_optimizer=True, forced=forced)
    out["ovl_ddp"] = _run(state, overlap=True, distributed_optimizer=False, forced=forced)
    out["early"] = _run(
        state, overlap=True, distributed_optimizer=True, mutation="early", forced=forced
    )
    out["stale"] = _run(
        state, overlap=True, distributed_optimizer=True, mutation="stale", forced=forced
    )
    # The overlapped reduce-scatter without the overlapped parameter gather.
    out["ovl_rs"] = _run(
        state, overlap=True, distributed_optimizer=True, param_gather=False, forced=forced
    )
    # Un-forced: ovl on its own step-1 gradient, and ref stepping on that same gradient.
    out["ovl_unforced"] = _run(state, overlap=True, distributed_optimizer=True)
    out["ref_from_ovl"] = _run(
        state, overlap=False, distributed_optimizer=True, forced=out["ovl_unforced"][0][0][1]
    )
    reference = out["ref"][1]
    for arm in FORCED:
        record = out[arm][1]
        record["forcing"] = sgu.forcing_record(
            record["readback"], _start_mismatches(record, reference)
        )
    # stale's weights differ from ref's by design (checked by its test); its buffers do not.
    stale = out["stale"][1]
    stale["forcing"] = sgu.forcing_record(
        stale["readback"], sgu.mismatches(stale["buffers"], reference["buffers"])
    )
    record = out["ref_from_ovl"][1]
    record["forcing"] = sgu.forcing_record(
        record["readback"], _start_mismatches(record, out["ovl_unforced"][1])
    )
    return out


def _name(text):
    return sgu.tag(f"DDP-overlap EP{EP}", text)


@pytest.mark.parametrize("arm", FORCED)
def test_step_two_starts_from_the_reference_state(results, arm):
    """The teacher-forcing precondition: the forced gradient read back on every rank, and the
    arm's own optimizer step on it (and its parameter gather) reproduced ``ref``'s weights (at
    the end of step 2) and buffers exactly."""
    sgu.assert_teacher_forced(results[arm][1]["forcing"], arm)


@pytest.mark.parametrize("arm", GATED)
@pytest.mark.parametrize("step", range(STEPS))
def test_gradients_match_the_reference(results, arm, step):
    if step > 0:
        sgu.assert_teacher_forced(results[arm][1]["forcing"], arm)
    _, reference = results["ref"][0][step]
    _, grads = results[arm][0][step]
    report, failing = sgu.report_and_fail(_name(f"{arm} step {step + 1} vs ref"), grads, reference)
    assert not failing, sgu.mbs.worst(report, failing)


@pytest.mark.parametrize("arm", GATED)
def test_losses_match_the_reference(results, arm):
    for step in range(STEPS):
        if step > 0:
            sgu.assert_teacher_forced(results[arm][1]["forcing"], arm)
        reference, _ = results["ref"][0][step]
        result, _ = results[arm][0][step]
        sgu.assert_losses_match(result, reference, _name(f"{arm} step {step + 1}"))


@pytest.mark.parametrize("arm", GATED)
def test_every_bucket_group_reduces_once_from_its_hook(results, arm):
    """Step 1 records the counts (no overlapped dispatch); in step 2 every bucket group starts
    its reduction from the gradient hook exactly once. The recorded counts are printed (an
    autograd-hooked parameter is expected to be ready once per backward: no module GTP in this
    model)."""
    _, record = results[arm]
    print(f"[{_name(arm)}] dispatches {record['dispatches']} golden {record['golden']}", flush=True)
    assert record["groups"] > 1, record["groups"]
    assert record["dispatches"][0] == [0] * record["groups"], record["dispatches"]
    assert record["dispatches"][1] == [1] * record["groups"], record["dispatches"]
    assert all(values and values[0] >= 1 for values in record["golden"]), record["golden"]


def test_step_two_agrees_given_the_same_step_one_gradient(results):
    """The un-forced overlap path: ``ovl_unforced`` steps on its own step-1 gradient and
    ``ref_from_ovl`` (``ref``) on that same gradient. From an exactly equal state, their step-2
    gradients pass the gate and their losses are equal."""
    sgu.assert_teacher_forced(
        results["ref_from_ovl"][1]["forcing"], "ref_from_ovl", "ovl_unforced's"
    )
    reference_result, reference = results["ref_from_ovl"][0][1]
    result, grads = results["ovl_unforced"][0][1]
    report, failing = sgu.report_and_fail(
        _name("ovl_unforced step 2 vs ref_from_ovl"), grads, reference
    )
    assert not failing, sgu.mbs.worst(report, failing)
    sgu.assert_losses_match(result, reference_result, _name("ovl_unforced step 2"))


def test_report_unforced_step_two(results):
    """Diagnostic only (prints, asserts nothing): step 2 without teacher forcing, against
    ``ref``.

    * ``ovl_unforced``: the per-parameter report, every weight that differs at step 2 (dtype,
      differing elements, max abs difference) and the losses;
    * ``ref_from_ovl``: ``ref`` stepping on ``ovl``'s step-1 gradient, which differs from its own
      only by fp32 reassociation. If it fails like ``ovl_unforced`` while the forced ``ovl``
      passes, the un-forced step-2 gate fails ``ref`` against itself.
    """
    _, reference = results["ref"][0][1]
    for arm in ("ovl_unforced", "ref_from_ovl"):
        _, grads = results[arm][0][1]
        report, failing = sgu.report_and_fail(
            _name(f"{arm} step 2 vs ref (diagnostic)"), grads, reference
        )
        print(f"[{_name(arm)}] step 2 failing {len(failing)} of {len(report)}", flush=True)
        differing = sgu.mismatches(results[arm][1]["weights"], results["ref"][1]["weights"])
        print(
            f"[{_name(arm)}] step-2 weights differ from ref in {len(differing)} of "
            f"{len(reference)} params: {sorted(differing.items(), key=lambda item: -item[1][0])}",
            flush=True,
        )
        rel = sgu.mbs.relative_loss_differences(
            results[arm][0][1][0].losses, results["ref"][0][1][0].losses
        )
        print(f"[{_name(f'{arm} step 2')}] losses rel {rel} (diagnostic)", flush=True)


def _stale_weights(record, reference, initial):
    """The ``stale`` mutant's step-2 weights against ``ref``'s and the initial ones: the names
    with an element that is neither, whether some element differs from ``ref``'s (the mutation
    acted) and whether some element that ``ref``'s update moved equals ``ref``'s (the forced
    update landed in the local shard)."""
    foreign, acted, landed = [], False, False
    for name, want in reference.items():
        have = record["weights"][name]
        start = initial[name].to(have.device, have.dtype)
        if not bool(((have == want) | (have == start)).all()):
            foreign.append(name)
        acted |= bool((have != want).any())
        landed |= bool(((want != start) & (have == want)).any())
    return foreign, acted, landed


@pytest.mark.parametrize("mutation", ["early", "stale"])
def test_mutations_fail_the_gate(results, mutation):
    """A reduction started before the last use of every parameter in its bucket group, or a
    forward on stale gathered weights, must fail the step-2 gradient gate: ``early`` from
    ``ref``'s state, ``stale`` with ``ref``'s update in the local shard only."""
    record = results[mutation][1]
    if mutation == "early":
        sgu.assert_teacher_forced(record["forcing"], mutation)
    else:
        assert record["readback"] == 0 and record["forcing"]["agreed"] == 0, record["forcing"]
        foreign, acted, landed = _stale_weights(
            record, results["ref"][1]["weights"], results["initial"]
        )
        assert not foreign and acted and landed, (foreign[:8], acted, landed)
    _, reference = results["ref"][0][1]
    _, grads = results[mutation][0][1]
    report, failing = sgu.report_and_fail(
        _name(f"{mutation} (mutation) step 2 vs ref"), grads, reference
    )
    assert failing, f"the {mutation} mutation passed the gate: the test has no teeth"
    print(f"[{_name(mutation)}] failing {len(failing)} of {len(report)}", flush=True)
