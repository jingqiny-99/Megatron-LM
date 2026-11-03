# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The loop ring schedule's per-step pieces on real ring chunks (CPU): round views, replay
iterators and the finalize wrapper (``megatron/core/transformer/loop/ring_pp.py``).

Ring chunks: the local-layer loop ``HybridModel`` s of the CPU emulation (``ring_emulation``:
dense or MoE with expert bias, mHC 4, MTP, R = 4), each wrapped in a DDP stand-in (``_Chunk``:
the ``.module`` chain, ``finish_grad_sync`` / ``scale_gradients`` counters).

* Views: ``config`` is the ring config; ``decoder``, ``set_input_tensor``, ``pg_collection`` and
  ``model_type`` resolve through the ``.module`` walk; ``getattr(view, "vp_stage", None)`` is
  None while the wrapped ``vp_stage`` is the physical chunk index (None at k = 1); a view has no
  parameters; through a ``Float16Module`` only the final view asks for an fp32 output.
* Replay: each physical source is pulled exactly M times, in order, and microbatch m's batch is
  served to every round view of its chunk (a copy each); out-of-order or mismatched reads raise.
* Finalize: Megatron's ``finalize_model_grads`` through ``RingFinalize`` on the schedule's view
  list runs ``finish_grad_sync``, ``scale_gradients`` and the expert-bias update once per
  physical chunk (on the views directly the update would find no router); it looks the model
  config's function up when called, so ``train_step``'s offload wrapper runs prefetch, then
  finalize, once; it refuses a step that left call roles or changed a built flag.
"""

import pytest
import torch
from torch import nn

from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.hybrid.hybrid_block import HybridStackCallRoles
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.loop import ring_pp
from megatron.core.transformer.loop.ring_pp import (
    LoopRingRoundView,
    RingFinalize,
    RingReplayIterator,
    RingReplaySource,
    ring_physical_chunks,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
from megatron.core.transformer.moe.moe_utils import get_updated_expert_bias
from megatron.core.utils import get_attr_wrapped_model, get_model_config
from tests.unit_tests.transformer.loop import hybrid_loop_utils as hlu
from tests.unit_tests.transformer.loop import ring_emulation as emu

ROUNDS = emu.ROUNDS

pytestmark = pytest.mark.skipif(
    torch.cuda.is_available(), reason="CPU ring chunks (the GPU ring runs through job-launch)"
)

mhc_on_this_device = hlu.mhc_on_this_device
cpu_mtp = hlu.cpu_mtp


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with hlu.model_parallel_once():
        yield


@pytest.fixture
def cpu_ring(mhc_on_this_device, cpu_mtp, monkeypatch):
    """The emulation's CPU environment (router buffers on the CPU)."""
    del mhc_on_this_device, cpu_mtp
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")


class _Chunk(nn.Module):
    """A DDP stand-in around one ring chunk: the ``.module`` chain, the model config, and the
    calls finalize makes on a DDP chunk."""

    def __init__(self, module, config):
        super().__init__()
        self.module = module
        self.config = config
        self.ddp_config = None
        self.synced = []
        self.scaled = []

    def finish_grad_sync(self, force_all_reduce=False):
        self.synced.append(force_all_reduce)

    def scale_gradients(self, scaling):
        self.scaled.append(float(scaling))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _ring(kind, stages, chunks, plan="all-C"):
    """The emulation's ring of ``kind`` (weights of a fresh PP = 1 model)."""
    main, mtp = emu.patterns(kind)
    set_moe_metrics_tracker(MoEMetricsTracker())
    reference = emu.build_model(kind, emu.model_config(kind), main + "/" + mtp)
    state = {
        name: value.detach().clone() if torch.is_tensor(value) else value
        for name, value in reference.state_dict().items()
    }
    return emu.build_ring(kind, stages, chunks, plan, state)


def _stage_views(ring, stage, wrap=_Chunk):
    """Stage ``stage``'s wrapped physical chunks and their V round views (the ring config)."""
    chunks = [wrap(ring.chunk_model(stage, j), ring.config) for j in range(ring.chunks)]
    views = [
        LoopRingRoundView(
            chunks[view % ring.chunks],
            ring.step_config,
            round_index=view // ring.chunks,
            chunk_index=view % ring.chunks,
            chunks=ring.chunks,
            checkpoint=ring.codes[stage][view] == "C",
        )
        for view in range(ROUNDS * ring.chunks)
    ]
    return chunks, views


# ---------------------------------------------------------------------------------------------
# Round views
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("stages,chunks", [(2, 1), (2, 2), (4, 1)])
def test_views_resolve_through_the_wrapped_chunk(cpu_ring, stages, chunks):
    del cpu_ring
    ring = _ring("dense", stages, chunks)
    for stage in range(stages):
        physical, views = _stage_views(ring, stage)
        assert ring_physical_chunks(views) == physical
        for view in views:
            model = ring.chunk_model(stage, view.chunk_index)
            assert get_model_config(view) is ring.step_config
            assert ring.step_config.virtual_pipeline_model_parallel_size == ROUNDS * chunks
            assert get_attr_wrapped_model(view, "decoder") is model.decoder
            assert get_attr_wrapped_model(view, "set_input_tensor") == model.set_input_tensor
            assert get_attr_wrapped_model(view, "pg_collection") is model.pg_collection
            assert get_attr_wrapped_model(view, "model_type") == model.model_type
            assert getattr(view, "vp_stage", None) is None
            assert get_attr_wrapped_model(view, "vp_stage") == (
                view.chunk_index if chunks > 1 else None
            )
            assert list(view.parameters()) == [] and list(view.buffers()) == []
            assert view.view_index == view.round_index * chunks + view.chunk_index


class _RecordingFloat16(Float16Module):
    """A ``Float16Module`` in the view's chain that records ``fp32_output`` (no conversion)."""

    def __init__(self, module, config):
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        self.module = module
        self.config = config
        self.fp32_outputs = []

    def forward(self, *inputs, fp32_output=True, **kwargs):
        del inputs, kwargs
        self.fp32_outputs.append(fp32_output)


@pytest.mark.parametrize("stages,chunks", [(2, 1), (2, 2)])
def test_only_the_final_view_asks_for_an_fp32_output(cpu_ring, stages, chunks):
    del cpu_ring
    ring = _ring("dense", stages, chunks)
    views_count = ROUNDS * chunks
    for stage in range(stages):
        physical, views = _stage_views(ring, stage, wrap=_RecordingFloat16)
        for view in views:
            view.module.module.decoder.current_microbatch = 0
            view(None)
        for chunk in physical:
            index = physical.index(chunk)
            expected = [view == views_count - 1 for view in range(index, views_count, chunks)]
            assert chunk.fp32_outputs == expected
        for view in views:
            assert view.calls == 1


# ---------------------------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------------------------


class _Source:
    """A data source of dicts ``{"mb": i, "tokens": tensor}`` counting its pulls."""

    def __init__(self):
        self.pulled = 0

    def __iter__(self):
        return self

    def __next__(self):
        item = {"mb": self.pulled, "tokens": torch.full((2,), float(self.pulled))}
        self.pulled += 1
        return item


class _FakeView:
    """The two fields a replay iterator reads of its view."""

    def __init__(self, index):
        self.view_index = index
        self.calls = 0


@pytest.mark.parametrize("stages,chunks,microbatches", [(2, 1, 4), (2, 2, 8), (4, 2, 8)])
def test_replay_pulls_each_source_m_times_and_serves_every_view(stages, chunks, microbatches):
    """In each rank's interleaved order the chunk's R views read microbatch m from one pull."""
    from tests.unit_tests.transformer.loop.test_loop_ring_schedule_order import rank_ops

    views_count = ROUNDS * chunks
    for stage in range(stages):
        sources = [_Source() for _ in range(chunks)]
        replays = [RingReplaySource(source, ROUNDS) for source in sources]
        views = [_FakeView(v) for v in range(views_count)]
        iterators = [RingReplayIterator(replays[v % chunks], views[v]) for v in range(views_count)]
        ops, _ = rank_ops(stages, stage, views_count, microbatches, stages)
        served = {}
        for kind, view, microbatch in ops:
            if kind != "F":
                continue
            assert views[view].calls == microbatch
            batch = next(iterators[view])
            served.setdefault((view % chunks, microbatch), []).append(batch)
            views[view].calls += 1
        for chunk, (source, replay) in enumerate(zip(sources, replays)):
            assert source.pulled == microbatches and replay.pulled == microbatches
            assert replay.pending() == ()
            for microbatch in range(microbatches):
                batches = served[(chunk, microbatch)]
                assert len(batches) == ROUNDS
                assert all(batch["mb"] == microbatch for batch in batches)
                # One copy per view: rebinding an entry in one view's batch leaves the others.
                assert len({id(batch) for batch in batches}) == ROUNDS
                assert all(batch["tokens"] is batches[0]["tokens"] for batch in batches)


def test_replay_refuses_out_of_order_and_mismatched_reads():
    replay = RingReplaySource(_Source(), ROUNDS)
    with pytest.raises(RuntimeError, match="asked for after 0 pulled"):
        replay.get(1)
    view = _FakeView(0)
    iterator = RingReplayIterator(replay, view)
    next(iterator)
    with pytest.raises(RuntimeError, match="batch 1 asked for by call 0"):
        next(iterator)  # a second batch inside one call


# ---------------------------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------------------------


def _routers(chunks):
    """The routers whose expert bias finalize updates, in finalize's walk order."""
    return [
        module
        for chunk in chunks
        for module in chunk.modules()
        if getattr(module, "expert_bias", None) is not None
        and module.training
        and not getattr(module, "frozen_expert_bias", False)
    ]


@pytest.mark.parametrize("stages,chunks", [(2, 1), (2, 2), (4, 2)])
def test_finalize_runs_once_per_physical_chunk(cpu_ring, stages, chunks):
    del cpu_ring
    ring = _ring("moe", stages, chunks)
    collection = ProcessGroupCollection.use_mpu_process_groups()
    for stage in range(stages):
        physical, views = _stage_views(ring, stage)
        ring.config.finalize_model_grads_func = finalize_model_grads
        states = ring_pp.LoopRingSchedule._chunk_states(physical, ring.config)
        routers = _routers(physical)
        for index, router in enumerate(routers):
            router.local_tokens_per_expert.copy_(
                torch.arange(router.local_tokens_per_expert.numel()) * (index + 1)
            )
        expected = get_updated_expert_bias(
            torch.stack([router.local_tokens_per_expert.clone() for router in routers]),
            torch.stack([router.expert_bias.clone() for router in routers]),
            ring.config.moe_router_bias_update_rate,
            collection.tp_dp_cp,
        )
        finalize = RingFinalize(ring.config, states)
        finalize(
            views,
            torch.tensor(1000, dtype=torch.int),
            pg_collection=collection,
            force_all_reduce=True,
        )
        assert finalize.calls == 1
        for chunk in physical:
            assert chunk.synced == [True]
            assert chunk.scaled == [pytest.approx(1.0 / 1000, rel=1e-6)]
        assert len(routers) > 0
        for router, bias in zip(routers, expected):
            assert torch.equal(router.expert_bias, bias)
            assert not router.local_tokens_per_expert.any()


def test_finalize_on_the_views_directly_would_miss_the_routers(cpu_ring):
    """Why the wrapper reduces to physical chunks: a view has no submodules, so finalize's
    expert-bias walk over the views finds no router (and a view has no grad sync)."""
    del cpu_ring
    from megatron.core.distributed.finalize_model_grads import _update_router_expert_bias

    ring = _ring("moe", 2, 2)
    physical, views = _stage_views(ring, 0)
    routers = _routers(physical)
    for router in routers:
        router.local_tokens_per_expert.fill_(3)
    before = [router.expert_bias.clone() for router in routers]
    collection = ProcessGroupCollection.use_mpu_process_groups()
    _update_router_expert_bias(views, ring.config, tp_dp_cp_group=collection.tp_dp_cp)
    assert all(torch.equal(router.expert_bias, bias) for router, bias in zip(routers, before))
    assert not hasattr(views[0], "finish_grad_sync")


def test_finalize_delegates_at_call_time_and_after_the_offload_prefetch(cpu_ring):
    del cpu_ring
    ring = _ring("dense", 2, 2)
    physical, views = _stage_views(ring, 1)
    states = ring_pp.LoopRingSchedule._chunk_states(physical, ring.config)
    events = []

    def first(model, num_tokens, pg_collection=None, force_all_reduce=False):
        events.append(("first", len(model)))

    def second(model, num_tokens, pg_collection=None, force_all_reduce=False):
        del pg_collection
        events.append(("finalize", tuple(id(c) for c in model), num_tokens, force_all_reduce))

    ring.config.finalize_model_grads_func = first
    finalize = RingFinalize(ring.config, states)
    # Swapped after the wrapper exists (as train_step re-wraps every step): the call uses it.
    ring.config.finalize_model_grads_func = second
    base = ring.config.finalize_model_grads_func

    def finalize_model_grads_with_state_reload(*args, **kwargs):
        events.append(("prefetch",))
        return base(*args, **kwargs)

    ring.config.finalize_model_grads_func = finalize_model_grads_with_state_reload
    finalize(views, 7, force_all_reduce=False)
    assert events == [("prefetch",), ("finalize", tuple(id(c) for c in physical), 7, False)]


def test_finalize_refuses_roles_left_set_or_changed_flags(cpu_ring):
    del cpu_ring
    ring = _ring("dense", 2, 1)
    physical, views = _stage_views(ring, 1)
    ring.config.finalize_model_grads_func = lambda *args, **kwargs: None
    states = ring_pp.LoopRingSchedule._chunk_states(physical, ring.config)
    finalize = RingFinalize(ring.config, states)
    decoder = ring.chunk_model(1, 0).decoder
    decoder.call_roles = HybridStackCallRoles(False, False, False)
    with pytest.raises(RuntimeError, match="still has call roles"):
        finalize(views, None)
    decoder.call_roles = None
    decoder.post_process = not decoder.post_process
    with pytest.raises(RuntimeError, match="built pipeline flags changed"):
        finalize(views, None)
    decoder.post_process = not decoder.post_process
    with pytest.raises(RuntimeError, match="does not reduce to this step's"):
        finalize(views[:1] + [nn.Linear(1, 1)], None)
    finalize(views, None)
    assert finalize.calls == 1
