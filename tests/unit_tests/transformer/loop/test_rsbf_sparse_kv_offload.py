# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the pinned RSBF KV page lifecycle across different autograd streams."""

import weakref

import pytest
import torch

from megatron.core.transformer.loop.rsbf_kv_offload import SparseKVOffload


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pinned transfers")
def test_alternating_streams_and_page_reuse():
    device = torch.device("cuda", torch.cuda.current_device())
    writer = torch.cuda.Stream(device=device)
    reader = torch.cuda.Stream(device=device)
    bank = SparseKVOffload(device, measure_time=True)
    key = (0, object())
    expected_grads = []
    for wave in range(3):
        expected = torch.arange(48, dtype=torch.float32) + wave
        with torch.cuda.stream(writer):
            original = expected.to(device)
            owner = weakref.ref(original)
            bank.publish(key, original)
            del original
            assert owner() is None
        with torch.cuda.stream(reader):
            retained = bank.read(key, backward=True)
            (retained.square() * 2).sum().backward()
            del retained
        with torch.cuda.stream(writer):
            replay = expected.to(device).requires_grad_()
            bank.connect_source(key, replay)
            replay.sum().backward()
            expected_grads.append((replay.grad, 1 + 4 * expected))
    bank.join()
    torch.cuda.synchronize(device)
    stats = bank.close()
    for actual, expected in expected_grads:
        torch.testing.assert_close(actual.cpu(), expected)
    assert stats["peak_pinned_bytes"] == 48 * 4
    for kind in ("kv_d2h", "kv_h2d", "grad_d2h", "grad_h2d"):
        assert stats[f"{kind}_bytes"] == 3 * stats["peak_pinned_bytes"]
        assert stats[f"{kind}_ms"] >= 0
    assert stats["h2d_consumer_wait_ms"] >= 0
