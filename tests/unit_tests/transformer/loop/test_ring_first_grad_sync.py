# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exercise the physical-chunk finalize boundary without allocating CUDA storage."""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.transformer.loop.ring_pp import LoopRingSchedule, RingFinalize


def _chunks():
    chunks = [torch.nn.Module(), torch.nn.Linear(3, 2)]
    for i, chunk in enumerate(chunks):
        chunk.vp_stage = i
        chunk.pre_process = chunk.post_process = False
        chunk.decoder = SimpleNamespace(
            call_roles=None, loop_ring=True, pre_process=False, post_process=False
        )
    return chunks


def test_trim_once_at_first_physical_training_finalize(monkeypatch):
    schedule = LoopRingSchedule()
    chunks = _chunks()
    events = []
    config = SimpleNamespace(
        virtual_pipeline_model_parallel_size=2,
        finalize_model_grads_func=lambda *a, **kw: events.append("old"),
    )

    def wrapper(forward_only=False):
        return RingFinalize(
            config,
            schedule._chunk_states(chunks, config),
            schedule._trimmed_before_grad_sync,
            forward_only=forward_only,
        )

    available = Mock(return_value=True)
    capture = Mock(return_value=False)
    trim = Mock(side_effect=lambda: events.append("trim"))
    monkeypatch.setattr(torch.cuda, "is_available", available)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", capture)
    monkeypatch.setattr(torch.cuda, "empty_cache", trim)

    def call(finalize):
        # Repeated round views resolve to the two physical chunks, in order.
        return finalize(
            [chunks[0], chunks[1], chunks[1]], 17, pg_collection="physical", force_all_reduce=True
        )

    call(wrapper())  # Default off, including config-like callers without the field.
    config.loop_ring_trim_cache_before_first_grad_sync = True
    call(wrapper(True))  # The external schedule normally omits finalize altogether for evaluation.
    assert available.call_count == 0
    call(wrapper())  # CPU model even if the machine has GPUs.
    available.return_value = False
    call(wrapper())
    available.return_value = True
    monkeypatch.setattr(chunks[1], "parameters", lambda: [SimpleNamespace(is_cuda=True)])
    capture.return_value = True
    call(wrapper())  # Keep a later eager training call eligible.
    assert trim.call_count == 0
    capture.return_value = False

    first = wrapper()
    chunks[0].decoder.call_roles = object()
    with pytest.raises(RuntimeError, match="still has call roles"):
        call(first)
    chunks[0].decoder.call_roles = None
    config.finalize_model_grads_func = None
    with pytest.raises(RuntimeError, match="was cleared"):
        call(first)
    assert trim.call_count == 0

    def finalize(physical, count, pg_collection, force_all_reduce):
        assert physical == chunks and (count, pg_collection, force_all_reduce) == (
            17,
            "physical",
            True,
        )
        events.extend(["prefetch", "sync"])
        return 73

    config.finalize_model_grads_func = finalize  # train_step supplies the live optimizer wrapper.
    trim.side_effect = RuntimeError("allocator trim failed")
    with pytest.raises(RuntimeError, match="allocator trim failed"):
        call(first)
    assert first.calls == 0 and not schedule._trimmed_before_grad_sync
    trim.side_effect = lambda: events.append("trim")
    assert call(wrapper()) == 73
    assert events[-3:] == ["trim", "prefetch", "sync"]

    queries = (available.call_count, capture.call_count)
    # Retry, a new training update, and clearing schedule metrics all reuse the physical model.
    schedule.reset()
    assert call(first) == call(wrapper()) == 73
    assert (available.call_count, capture.call_count) == queries
    assert events.count("trim") == 1
    first = None
    previous = weakref.ref(chunks[0])
    chunks = _chunks()
    gc.collect()
    assert previous() is None and not schedule._trimmed_before_grad_sync
    monkeypatch.setattr(chunks[1], "parameters", lambda: [SimpleNamespace(is_cuda=True)])
    assert call(wrapper()) == 73
    assert events.count("trim") == 2


def test_failed_finalize_does_not_repeat_a_successful_trim(monkeypatch):
    schedule = LoopRingSchedule()
    chunks = _chunks()
    config = SimpleNamespace(
        virtual_pipeline_model_parallel_size=2,
        loop_ring_trim_cache_before_first_grad_sync=True,
        finalize_model_grads_func=Mock(side_effect=RuntimeError("sync failed")),
    )
    monkeypatch.setattr(chunks[0], "parameters", lambda: [SimpleNamespace(is_cuda=True)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    trim = Mock()
    monkeypatch.setattr(torch.cuda, "empty_cache", trim)
    for _ in range(2):
        wrapper = RingFinalize(
            config, schedule._chunk_states(chunks, config), schedule._trimmed_before_grad_sync
        )
        with pytest.raises(RuntimeError, match="sync failed"):
            wrapper(chunks)
    assert trim.call_count == 1
    assert config.finalize_model_grads_func.call_count == 2
