# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The loop ring's schedule entry point (``ring_pp.LoopRingSchedule``) end to end on CPU.

One thread per pipeline rank runs a real ring step: the ring config and its communicator, the
round views, the replay iterators, Megatron's unmodified interleaved schedule and P2P code, the
finalize wrapper and the step-end checks (``ring_threads``: an NCCL-ordered fake transport and
tagging fake chunks). Every run checks:

* the step completes: no P2P deadlock under in-order streams with rendezvous, no send buffer
  freed before its send completed;
* FIFO pairing in the real P2P code: every forward input comes from the predecessor view of the
  same microbatch (the round wrap included) and every gradient from the successor view;
* each chunk ran each microbatch R times forward and backward, with the recompute code of its
  view in the plan (``loop_ring_recompute_plan``);
* replay: each data source pulled exactly M times, microbatch m's batch served to every round
  of the loading chunks;
* finalize ran once per step, on the k physical chunks in order (forward-only: never), after the
  offload prefetch when ``train_step``'s wrapper is installed;
* G1 (``LoopRingStepRecord``): rounds per sample R / R, prelude / coda once, MTP 0 (the fake
  chunks have none).
"""

import collections

import pytest
import torch

from tests.unit_tests.transformer.loop import hybrid_loop_utils as hlu
from tests.unit_tests.transformer.loop import ring_emulation as emu
from tests.unit_tests.transformer.loop.ring_threads import run_real_ring_threads, run_ring_threads
from tests.unit_tests.transformer.loop.ring_utils import ring_fields

ROUNDS = 4


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with hlu.model_parallel_once():
        yield


def _plan(stages, chunks):
    """A mixed recompute plan: 'C' where stage + view is even."""
    views = ROUNDS * chunks
    return "/".join(
        "".join("C" if (stage + view) % 2 == 0 else "n" for view in range(views))
        for stage in range(stages)
    )


def _config_factory(stages, chunks, group_size):
    plan = _plan(stages, chunks)

    def factory(stage):
        del stage
        config = emu.model_config(
            "dense",
            mtp_num_layers=None,
            loop_recompute=True,
            loop_ring_recompute_plan=plan,
            **ring_fields(stages, chunks),
        )
        config.microbatch_group_size_per_vp_stage = group_size
        config.deallocate_pipeline_outputs = True
        return config

    return factory, plan


def _check_run(results, *, stages, chunks, microbatches, plan, forward_only=False):
    views = ROUNDS * chunks
    codes = plan.split("/")
    for stage, result in enumerate(results):
        record = result.record
        assert record.rounds == (ROUNDS, ROUNDS), (stage, record)
        assert record.prelude == (1, 1) and record.coda == (1, 1) and record.mtp == (0, 0)
        assert record.views == views and record.microbatches == microbatches
        assert record.forward_only == forward_only
        assert record.summary() == (
            f"loop ring: rounds per sample min {ROUNDS} max {ROUNDS}; "
            "prelude/coda/mtp per sample 1/1/0"
        )
        forwards = collections.Counter(entry[2:] for entry in result.log if entry[0] == "F")
        backwards = collections.Counter(entry[2:] for entry in result.log if entry[0] == "B")
        expected = {
            (float(view), float(mb)): 1 for view in range(views) for mb in range(microbatches)
        }
        assert forwards == expected
        assert backwards == ({} if forward_only else expected)
        for chunk, source in zip(result.chunks, result.sources):
            assert chunk.codes == [
                (view, codes[stage][view] == "C")
                for view, _ in chunk.codes  # the schedule's call order
            ]
            assert sorted({view for view, _ in chunk.codes}) == list(
                range(chunk.chunk, views, chunks)
            )
            if source is None:
                assert chunk.served == []
                continue
            assert source.pulled == microbatches
            assert sorted(chunk.served) == sorted(
                (round_index, mb, mb) for round_index in range(ROUNDS) for mb in range(microbatches)
            )
        if forward_only:
            assert result.finalized == []
        else:
            assert result.finalized[-1] == (
                "finalize",
                tuple(id(chunk) for chunk in result.chunks),
                False,
            )
        if stage == stages - 1:
            assert len(result.output) == microbatches  # one loss report per microbatch
        else:
            assert result.output == []


def _grid():
    for stages in (2, 3, 4):
        for chunks in (1, 2):
            for group_size in (stages, 2 * stages):
                for microbatches in (group_size, 2 * group_size):
                    yield stages, chunks, group_size, microbatches
    yield 8, 1, 8, 8
    yield 8, 4, 8, 8
    yield 2, 4, 4, 8


@pytest.mark.parametrize("stages,chunks,group_size,microbatches", list(_grid()))
def test_ring_step_runs_the_interleaved_schedule_over_round_views(
    monkeypatch, stages, chunks, group_size, microbatches
):
    factory, plan = _config_factory(stages, chunks, group_size)
    results = run_ring_threads(
        monkeypatch, factory, stages=stages, chunks=chunks, microbatches=microbatches
    )
    _check_run(results, stages=stages, chunks=chunks, microbatches=microbatches, plan=plan)
    for result in results:
        assert result.finalized == [
            ("finalize", tuple(id(chunk) for chunk in result.chunks), False)
        ]


@pytest.mark.parametrize("stages,chunks", [(2, 1), (2, 2), (4, 1), (3, 2)])
def test_forward_only_ring_step(monkeypatch, stages, chunks):
    microbatches = 2 * stages
    factory, plan = _config_factory(stages, chunks, stages)
    results = run_ring_threads(
        monkeypatch,
        factory,
        stages=stages,
        chunks=chunks,
        microbatches=microbatches,
        forward_only=True,
    )
    _check_run(
        results,
        stages=stages,
        chunks=chunks,
        microbatches=microbatches,
        plan=plan,
        forward_only=True,
    )


def _train_step_offload_wrapper(config, record):
    """``train_step``'s chunked optimizer-state offload wrapper (training.py): the prefetch, then
    the base finalize function, installed on the model config before the step."""
    base = config.finalize_model_grads_func

    def finalize_model_grads_with_state_reload(*args, **kwargs):
        record.append(("prefetch",))
        return base(*args, **kwargs)

    config.finalize_model_grads_func = finalize_model_grads_with_state_reload


@pytest.mark.parametrize("stages,chunks", [(2, 1), (2, 2), (4, 2)])
def test_offload_prefetch_then_finalize_once(monkeypatch, stages, chunks):
    """With ``train_step``'s offload wrapper on the model config: prefetch, then finalize on the
    k physical chunks, once; ``force_all_reduce`` (the wgrad dump) reaches finalize."""
    microbatches = 2 * stages
    factory, plan = _config_factory(stages, chunks, stages)
    results = run_ring_threads(
        monkeypatch,
        factory,
        stages=stages,
        chunks=chunks,
        microbatches=microbatches,
        finalize_wrapper=_train_step_offload_wrapper,
        force_all_reduce=True,
    )
    for result in results:
        assert result.finalized == [
            ("prefetch",),
            ("finalize", tuple(id(chunk) for chunk in result.chunks), True),
        ]
        result.finalized[-1] = result.finalized[-1][:2] + (False,)  # _check_run's default
    _check_run(results, stages=stages, chunks=chunks, microbatches=microbatches, plan=plan)


def test_mutations_fail_the_ring_step(monkeypatch):
    """The harness detects a broken ring: a communicator without the ring's virtual pipeline
    size (the model config's at k = 1: the non-interleaved warmup) breaks the step, and a chunk
    expecting the round wrap from the wrong view rejects its input (FIFO pairing is checked)."""
    from megatron.core.pipeline_parallel import p2p_communication
    from tests.unit_tests.transformer.loop import ring_threads

    factory, _ = _config_factory(2, 1, 2)
    base = p2p_communication.P2PCommunicator

    class WithoutVirtualPipeline(base):
        def __init__(self, pp_group, config):
            super().__init__(pp_group, config)
            self.virtual_pipeline_model_parallel_size = None

    with monkeypatch.context() as patch:
        patch.setattr(p2p_communication, "P2PCommunicator", WithoutVirtualPipeline)
        with pytest.raises(RuntimeError, match="rank .* failed"):
            run_ring_threads(patch, factory, stages=2, chunks=1, microbatches=4, timeout=5.0)

    expect = ring_threads.FakeChunk._expect

    def wrap_from_the_same_view(self, view, microbatch, forward):
        tag = expect(self, view, microbatch, forward)
        if forward and self.stage == 0 and tag is not None:
            return tag[0], float(view), tag[2]
        return tag

    with monkeypatch.context() as patch:
        patch.setattr(ring_threads.FakeChunk, "_expect", wrap_from_the_same_view)
        with pytest.raises(RuntimeError, match="input tagged"):
            run_ring_threads(patch, factory, stages=2, chunks=1, microbatches=4, timeout=5.0)


mhc_on_this_device = hlu.mhc_on_this_device
cpu_mtp = hlu.cpu_mtp


@pytest.fixture
def cpu_ring(mhc_on_this_device, cpu_mtp, monkeypatch):
    """The emulation's CPU environment (router buffers on the CPU)."""
    del mhc_on_this_device, cpu_mtp
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="fp32 CPU ring chunks (the emulation's)")
@pytest.mark.parametrize(
    "kind,stages,chunks,microbatches,plan",
    [
        ("dense", 2, 1, 4, "all-n"),
        ("dense", 2, 2, 4, "all-C"),
        ("dense", 3, 2, 6, "mixed"),
        ("dense", 4, 2, 8, "mixed-inv"),
        ("moe", 2, 1, 4, "all-n"),
        ("moe", 2, 2, 4, "all-C"),
        ("moe", 4, 1, 8, "mixed"),
    ],
)
def test_real_ring_chunks_reproduce_pp1(
    cpu_ring, monkeypatch, kind, stages, chunks, microbatches, plan
):
    """The schedule entry point on the CPU emulation's real ring chunks (local layers, mHC 4,
    MTP, R = 4; MoE: expert bias, hash-routed prelude) equals the PP = 1 eager loop: per-token
    losses bitwise, every parameter's gradient inside the G2 gate (cosine >= 0.9999, relative
    norm difference <= 1e-3; measured about 1e-6), the head call's loop metrics (depth, token
    layers, forward and backward FLOPs per token) equal, recompute FLOPs per token the plan's,
    G1 R / 1 / 1 / 1 on every rank and finalize once per rank on its k chunks. The step-end
    invariants (FLOPs recorded = M x analytic, view calls, replay, stores, roles) are the
    schedule's own checks."""
    del cpu_ring
    batches = emu.make_batches(microbatches)
    reference, state, _ = emu.run_reference(kind, batches)
    ring = emu.build_ring(kind, stages, chunks, plan, state)
    results = run_real_ring_threads(
        monkeypatch, ring, batches, emu._loss_func, sequence=hlu.SEQ, micro_batch=hlu.BATCH
    )
    for stage in range(stages):
        record, output, finalized = results[stage]
        assert (record.rounds, record.prelude, record.coda, record.mtp) == (
            (ROUNDS, ROUNDS),
            (1, 1),
            (1, 1),
            (1, 1),
        )
        assert finalized == [chunks]
        assert len(output) == (microbatches if stage == stages - 1 else 0)
    reports = results[stages - 1][1]
    for want, got in zip(reference.losses, (report["per_token"] for report in reports)):
        assert torch.equal(want, got)
    tokens = hlu.SEQ * hlu.BATCH
    for want, got in zip(reference.head_metrics, (report["metrics"] for report in reports)):
        assert list(want) == list(got)
        for key, value in want.items():
            if key == "loop recompute flops per token":
                value = torch.as_tensor(record.model_work[2], dtype=torch.float32) / tokens
            assert torch.equal(got[key], value), key
    grads = {}
    for segment, model in zip(ring.segments, ring.models):
        for name, param in model.named_parameters():
            key = emu._global_name(name, segment["offset"])
            grad = None if param.grad is None else param.grad.detach().clone()
            previous = grads.get(key)
            grads[key] = (
                grad if previous is None else (previous if grad is None else previous + grad)
            )
    for name, (_, cosine, relative) in emu.compare_grads(reference.grads, grads).items():
        assert cosine >= emu.G2_COSINE and relative <= emu.G2_RELATIVE, (name, cosine, relative)


def test_ring_step_rejects_a_caller_communicator_and_shape_hooks():
    from megatron.core.transformer.loop.ring_pp import LoopRingSchedule

    schedule = LoopRingSchedule()
    common = dict(
        forward_step_func=None,
        data_iterator=None,
        model=[torch.nn.Linear(1, 1)],
        num_microbatches=1,
        seq_length=1,
        micro_batch_size=1,
    )
    with pytest.raises(ValueError, match="adjust_tensor_shapes_fn"):
        schedule(**common, adjust_tensor_shapes_fn=lambda *a: a)
    with pytest.raises(ValueError, match="builds its own P2P communicator"):
        schedule(**common, p2p_communicator=object())
    with pytest.raises(ValueError, match="one data iterator per model chunk"):
        schedule(**{**common, "model": [torch.nn.Linear(1, 1)] * 2, "data_iterator": None})
