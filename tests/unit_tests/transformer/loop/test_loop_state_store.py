# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the RSBF round-input checkpoint store (``loop/state_store.py``).

Pure tensor bookkeeping: no model and no process group. Every case runs on CPU and, when a GPU
is present, on CUDA.
"""

import weakref

import pytest
import torch

from megatron.core.transformer.loop.state_store import (
    HBMOnlyPlacement,
    LoopStateStore,
    StateStoreError,
    StateTier,
    StoreKey,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

S, C, NH, K = 8, 1, 16, 2


@pytest.fixture(params=DEVICES)
def dev(request):
    return torch.device(request.param)


def _state(dev, value=1.0, dtype=torch.float32, rows=C):
    return torch.full((S, rows, NH), value, dtype=dtype, device=dev)


def _crit(dev, k=K, rows=C):
    return torch.arange(rows * k, dtype=torch.float32, device=dev).reshape(rows, k)


def _store(dev, **kwargs):
    store = LoopStateStore(device=dev, **kwargs)
    store.begin_wave(0)
    return store


def _nbytes(t):
    return t.untyped_storage().nbytes()


def test_put_get_free_round_trip(dev):
    store = _store(dev)
    e, s1, c0 = _state(dev, 0.5), _state(dev, 1.5), _crit(dev)
    store.put_injection(0, e)
    store.put_crit(0, 0, c0)
    store.put_state(0, 1, s1)
    assert store.get_state(0, 0) is None  # s_0 is built by the core cell
    assert torch.equal(store.get_injection(0), e)
    assert torch.equal(store.get_crit(0, 0), c0)
    assert torch.equal(store.get_state(0, 1), s1)
    # Reads do not release.
    assert torch.equal(store.get_state(0, 1), s1)
    stats = store.stats()
    assert stats.live_entries == 3
    assert stats.live_bytes == _nbytes(e) + _nbytes(s1) + _nbytes(c0)
    assert stats.wave == 0
    store.free_state(0, 1)
    store.free_crit(0, 0)
    store.free_injection(0)
    final = store.end_wave()
    assert final.live_entries == 0 and final.live_bytes == 0
    assert final.peak_bytes == _nbytes(e) + _nbytes(s1) + _nbytes(c0)
    assert store.wave is None


def test_keys_are_per_sample_and_round(dev):
    store = _store(dev)
    for sample in range(3):
        store.put_injection(sample, _state(dev, 10.0 + sample))
        for r in range(1, 4):
            store.put_state(sample, r, _state(dev, 100.0 * sample + r))
    for sample in range(3):
        assert store.get_injection(sample)[0, 0, 0].item() == 10.0 + sample
        for r in range(1, 4):
            assert store.get_state(sample, r)[0, 0, 0].item() == 100.0 * sample + r
    store.reset()


def test_duplicate_put_raises_with_key(dev):
    store = _store(dev)
    store.put_state(2, 3, _state(dev))
    with pytest.raises(StateStoreError, match=r"s\[sample 2, round 3\]: already stored"):
        store.put_state(2, 3, _state(dev))
    store.put_injection(2, _state(dev))
    with pytest.raises(StateStoreError, match=r"e\[sample 2\]: already stored"):
        store.put_injection(2, _state(dev))


def test_write_once_per_wave_even_after_free(dev):
    store = _store(dev)
    store.put_state(0, 1, _state(dev))
    store.free_state(0, 1)
    with pytest.raises(StateStoreError, match=r"already freed this wave"):
        store.put_state(0, 1, _state(dev))
    store.end_wave()
    # A new wave starts clean.
    store.begin_wave(1)
    store.put_state(0, 1, _state(dev))
    store.free_state(0, 1)
    store.end_wave()


def test_double_free_and_use_after_free(dev):
    store = _store(dev)
    store.put_state(1, 2, _state(dev))
    store.put_crit(1, 2, _crit(dev))
    store.put_injection(1, _state(dev))
    store.free_state(1, 2)
    store.free_crit(1, 2)
    store.free_injection(1)
    with pytest.raises(StateStoreError, match=r"sample 1, round 2\]: cannot free, already freed"):
        store.free_state(1, 2)
    with pytest.raises(StateStoreError, match=r"crit\[sample 1, round 2\]: cannot free"):
        store.free_crit(1, 2)
    with pytest.raises(StateStoreError, match=r"e\[sample 1\]: cannot free, already freed"):
        store.free_injection(1)
    with pytest.raises(StateStoreError, match=r"use after free"):
        store.get_state(1, 2)
    with pytest.raises(StateStoreError, match=r"use after free"):
        store.get_crit(1, 2)
    with pytest.raises(StateStoreError, match=r"use after free"):
        store.get_injection(1)
    with pytest.raises(StateStoreError, match=r"cannot take, already freed"):
        store.take_state(1, 2)


def test_missing_entries(dev):
    store = _store(dev)
    with pytest.raises(StateStoreError, match=r"s\[sample 0, round 1\]: never stored"):
        store.get_state(0, 1)
    with pytest.raises(StateStoreError, match=r"crit\[sample 0, round 0\]: never stored"):
        store.get_crit(0, 0)
    with pytest.raises(StateStoreError, match=r"never stored"):
        store.free_state(0, 1)
    with pytest.raises(StateStoreError, match=r"never stored"):
        store.take_state(0, 1)
    with pytest.raises(StateStoreError, match=r"never stored"):
        store.get_injection(5)


def test_round_ranges(dev):
    store = _store(dev)
    with pytest.raises(StateStoreError, match=r"state round must be >= 1"):
        store.put_state(0, 0, _state(dev))
    with pytest.raises(StateStoreError, match=r"state round must be >= 0"):
        store.get_state(0, -1)
    with pytest.raises(StateStoreError, match=r"state round must be >= 1"):
        store.free_state(0, 0)
    with pytest.raises(StateStoreError, match=r"state round must be >= 1"):
        store.take_state(0, 0)
    with pytest.raises(StateStoreError, match=r"crit round must be >= 0"):
        store.put_crit(0, -1, _crit(dev))
    store.put_crit(0, 0, _crit(dev))  # crit_0 is stored
    store.free_crit(0, 0)


def test_take_state_removes_and_never_poisons(dev):
    store = _store(dev, poison_on_free=True)
    s = _state(dev, 3.0)
    store.put_state(0, 1, s)
    taken = store.take_state(0, 1)
    assert torch.equal(taken, s)  # not poisoned
    assert taken.untyped_storage().data_ptr() == s.untyped_storage().data_ptr()
    assert store.stats().live_entries == 0 and store.stats().live_bytes == 0
    with pytest.raises(StateStoreError, match=r"already taken \(ownership moved"):
        store.get_state(0, 1)
    with pytest.raises(StateStoreError, match=r"cannot take, already taken"):
        store.take_state(0, 1)
    with pytest.raises(StateStoreError, match=r"cannot free, already taken"):
        store.free_state(0, 1)
    with pytest.raises(StateStoreError, match=r"already taken this wave"):
        store.put_state(0, 1, s)
    store.end_wave()
    assert torch.equal(taken, s)


def test_check_recompute_style_wave_ends_empty(dev):
    """The executor's check_recompute path (design 7.4) for two samples of depths 2 and 1:
    the consumer of s_r takes it into the chain record, CORE(i, r) compares and drops it."""
    store = _store(dev, poison_on_free=True)
    depths = {0: 2, 1: 1}
    sweep = {}
    for i, d in depths.items():  # prelude sweep, core sweeps
        store.put_injection(i, _state(dev, 0.25 + i))
        store.put_crit(i, 0, _crit(dev, k=0))
        for r in range(1, d + 1):
            sweep[(i, r)] = _state(dev, 10.0 * i + r)
            store.put_state(i, r, sweep[(i, r)].clone())
            store.put_crit(i, r, _crit(dev, k=0))
    for i, d in depths.items():
        # CODA: takes s_d into the chain record, frees crit_d.
        expected = store.take_state(i, d)
        store.free_crit(i, d)
        for r in range(d, 0, -1):
            # CORE(i, r): recompute s_r from s_{r-1}; compare with the record, drop it.
            assert torch.equal(sweep[(i, r)], expected)
            # Take + compare + drop leaves no reference: the store kept none.
            dropped = weakref.ref(expected)
            del expected
            assert dropped() is None
            expected = store.take_state(i, r - 1) if r > 1 else None
            store.get_injection(i)
            store.get_crit(i, r - 1)
            store.free_crit(i, r - 1)
        assert expected is None
        store.get_injection(i)  # PRELUDE item: compares e, then frees it
        store.free_injection(i)
    stats = store.end_wave()
    assert stats.live_entries == 0 and stats.live_bytes == 0
    with pytest.raises(StateStoreError, match=r"never stored"):
        store.get_state(0, 2)  # end_wave forgot everything; the next wave starts clean


def test_end_wave_with_live_entries(dev):
    store = _store(dev)
    store.put_injection(3, _state(dev))
    store.put_state(3, 1, _state(dev))
    with pytest.raises(StateStoreError, match=r"wave 0 ends with 2 live entries: .*sample 3"):
        store.end_wave()
    # The wave stays open; releasing the entries lets it end.
    store.free_state(3, 1)
    store.free_injection(3)
    store.end_wave()


def test_wave_protocol(dev):
    store = LoopStateStore(device=dev)
    with pytest.raises(StateStoreError, match=r"put outside a wave"):
        store.put_state(0, 1, _state(dev))
    with pytest.raises(StateStoreError, match=r"end_wave without an open wave"):
        store.end_wave()
    store.begin_wave(0)
    with pytest.raises(StateStoreError, match=r"wave 1 begins while wave 0 is open"):
        store.begin_wave(1)
    store.put_state(0, 1, _state(dev))
    store.reset()  # error recovery: drops the entry, closes the wave
    assert store.wave is None and store.stats().live_entries == 0
    assert store.stats().live_bytes == 0
    store.begin_wave(1)
    store.end_wave()


def test_crit_with_zero_width(dev):
    """k = 0: the entry holds no bytes but keeps its lifetime checks."""
    store = _store(dev)
    store.put_injection(0, _state(dev))
    before = store.stats().live_bytes
    crit = _crit(dev, k=0)
    store.put_crit(0, 0, crit)
    assert store.stats().live_bytes == before
    assert store.stats().live_entries == 2
    got = store.get_crit(0, 0)
    assert got.shape == (C, 0) and got.dtype == torch.float32
    store.free_crit(0, 0)
    with pytest.raises(StateStoreError, match=r"cannot free, already freed"):
        store.free_crit(0, 0)
    store.free_injection(0)
    store.end_wave()


def test_poison_on_free(dev):
    store = _store(dev, poison_on_free=True)
    s, crit = _state(dev, 2.0), _crit(dev)
    alias_s, alias_c = s.view(-1), crit.view(-1)  # a stale alias held past the free
    store.put_state(0, 1, s)
    store.put_crit(0, 1, crit)
    store.free_state(0, 1)
    store.free_crit(0, 1)
    assert torch.isnan(alias_s).all() and torch.isnan(alias_c).all()
    store.end_wave()

    # Without poison_on_free the freed tensor is untouched.
    store = _store(dev)
    s = _state(dev, 2.0)
    store.put_state(0, 1, s)
    store.free_state(0, 1)
    assert torch.equal(s, _state(dev, 2.0))
    store.end_wave()


def test_poison_spares_storage_of_live_entries(dev):
    """A criterion may pass crit_{r-1} through unchanged: crit_r then aliases crit_{r-1}. Freeing
    crit_{r-1} must not poison the still-live crit_r; the bytes count once."""
    store = _store(dev, poison_on_free=True)
    crit = _crit(dev)
    store.put_crit(0, 0, crit)
    store.put_crit(0, 1, crit)
    assert store.stats().live_bytes == _nbytes(crit)
    store.free_crit(0, 0)
    assert torch.equal(store.get_crit(0, 1), _crit(dev))
    assert store.stats().live_bytes == _nbytes(crit)
    store.free_crit(0, 1)  # the last reference: poisoned now
    assert torch.isnan(crit).all()
    assert store.stats().live_bytes == 0

    # A storage taken by a caller is never poisoned through another entry.
    s = _state(dev, 4.0)
    store.put_state(0, 1, s)
    store.put_state(0, 2, s)
    taken = store.take_state(0, 2)
    store.free_state(0, 1)
    assert torch.equal(taken, _state(dev, 4.0))
    store.end_wave()


def test_detached_storage(dev):
    store = _store(dev)
    leaf = torch.ones(S, C, NH, device=dev, requires_grad=True)
    state = leaf * 2.0  # non-leaf with a graph
    store.put_state(0, 1, state)
    stored = store.get_state(0, 1)
    assert not stored.requires_grad and stored.grad_fn is None
    assert stored.untyped_storage().data_ptr() == state.untyped_storage().data_ptr()
    # The executor's fresh leaf from the stored value does not reach the sweep's graph.
    fresh = stored.detach().requires_grad_()
    fresh.sum().backward()
    assert leaf.grad is None
    store.free_state(0, 1)
    store.end_wave()


def test_dtype_shape_and_device_checks(dev):
    store = _store(dev, act_dtype=torch.bfloat16)
    with pytest.raises(
        StateStoreError, match=r"s\[sample 0, round 1\]: dtype .*is not torch.bfloat16"
    ):
        store.put_state(0, 1, _state(dev, dtype=torch.float32))
    with pytest.raises(StateStoreError, match=r"e\[sample 0\]: dtype"):
        store.put_injection(0, _state(dev, dtype=torch.float16))
    with pytest.raises(StateStoreError, match=r"crit\[sample 0, round 0\]: dtype .* not fp32"):
        store.put_crit(0, 0, _crit(dev).to(torch.bfloat16))
    with pytest.raises(StateStoreError, match=r"is not \[S, c, nH\]"):
        store.put_state(0, 1, torch.zeros(S, NH, dtype=torch.bfloat16, device=dev))
    with pytest.raises(StateStoreError, match=r"is not \[c, k\]"):
        store.put_crit(0, 0, torch.zeros(K, dtype=torch.float32, device=dev))
    store.put_injection(0, _state(dev, dtype=torch.bfloat16))
    with pytest.raises(
        StateStoreError, match=r"shape \(9, 1, 16\) is not \(8, 1, 16\) of e\[sample 0\]"
    ):
        store.put_state(0, 1, torch.zeros(S + 1, C, NH, dtype=torch.bfloat16, device=dev))
    with pytest.raises(StateStoreError, match=r"2 rows, e\[sample 0\] has 1"):
        store.put_crit(0, 0, _crit(dev, rows=2))
    with pytest.raises(StateStoreError, match=r"expected a tensor"):
        store.put_state(0, 1, [1.0])
    # Rejected puts store nothing.
    assert store.stats().live_entries == 1
    store.put_state(0, 1, _state(dev, dtype=torch.bfloat16))
    store.put_crit(0, 0, _crit(dev))
    store.reset()

    # Without act_dtype any floating dtype is accepted, integers are not.
    store = _store(dev)
    store.put_state(0, 1, _state(dev, dtype=torch.bfloat16))
    with pytest.raises(StateStoreError, match=r"is not floating"):
        store.put_state(0, 2, torch.zeros(S, C, NH, dtype=torch.int64, device=dev))
    store.reset()

    other = torch.device("meta")
    store = _store(dev)
    with pytest.raises(StateStoreError, match=r"device meta is not"):
        store.put_state(0, 1, torch.zeros(S, C, NH, device=other))
    store.reset()


def test_hbm_placement_and_grace_interface(dev):
    assert HBMOnlyPlacement().tier(0, 1, "state") is StateTier.HBM
    store = _store(dev)
    store.put_state(0, 1, _state(dev))
    stats = store.stats()
    assert stats.peak_bytes_by_tier[StateTier.HBM] == stats.peak_bytes > 0
    assert stats.peak_bytes_by_tier[StateTier.GRACE] == 0
    store.prefetch(0, 1)  # a no-op for an HBM entry
    store.prefetch(0, 0)  # s_0 needs nothing
    with pytest.raises(StateStoreError, match=r"never stored"):
        store.prefetch(0, 2)
    # Injections and crit states have tiers too.
    store.put_injection(0, _state(dev))
    store.put_crit(0, 0, _crit(dev))
    store.prefetch(0, 0, kind="injection")
    store.prefetch(0, 0, kind="crit")
    with pytest.raises(StateStoreError, match=r"crit\[sample 0, round 1\]: never stored"):
        store.prefetch(0, 1, kind="crit")
    with pytest.raises(StateStoreError, match=r"e\[sample 1\]: never stored"):
        store.prefetch(1, 0, kind="injection")
    with pytest.raises(StateStoreError, match=r"unknown entry kind"):
        store.prefetch(0, 0, kind="other")
    store.reset()

    class GracePlacement:
        def tier(self, sample, r, kind):
            return StateTier.GRACE if kind == "state" else StateTier.HBM

    store = _store(dev, placement=GracePlacement())
    store.put_injection(0, _state(dev))
    with pytest.raises(NotImplementedError, match=r"interface only"):
        store.put_state(0, 1, _state(dev))
    assert store.stats().live_entries == 1
    store.reset()


def test_per_wave_peaks(dev):
    store = LoopStateStore(device=dev)
    unit = _nbytes(_state(dev))

    # Wave 0: three states live at once.
    store.begin_wave(0)
    for r in range(1, 4):
        store.put_state(0, r, _state(dev))
    assert store.stats().peak_bytes == 3 * unit
    for r in range(1, 4):
        store.free_state(0, r)
    assert store.stats().live_bytes == 0 and store.stats().peak_bytes == 3 * unit
    wave0 = store.end_wave()

    # Wave 1: at most two live at once; the peak is the wave's own.
    store.begin_wave(1)
    store.put_state(0, 1, _state(dev))
    store.put_state(0, 2, _state(dev))
    store.free_state(0, 1)
    store.put_state(0, 3, _state(dev))
    store.free_state(0, 2)
    store.free_state(0, 3)
    wave1 = store.end_wave()

    assert (wave0.wave, wave0.peak_bytes) == (0, 3 * unit)
    assert (wave1.wave, wave1.peak_bytes) == (1, 2 * unit)
    assert wave1.peak_bytes_by_tier == {StateTier.HBM: 2 * unit, StateTier.GRACE: 0}


def test_unindexed_device_accepts_indexed_tensors():
    """Tensors report an indexed device (``cuda:0``); a store built with ``cpu:0`` or ``cuda``
    must accept them."""
    store = LoopStateStore(device=torch.device("cpu", 0))
    assert store.device == torch.device("cpu")
    store.begin_wave(0)
    store.put_state(0, 1, _state("cpu"))
    store.free_state(0, 1)
    store.end_wave()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_unindexed_cuda_device_accepts_current_device_tensor():
    index = torch.cuda.current_device()
    for device in ("cuda", torch.device("cuda")):
        store = LoopStateStore(device=device)
        assert store.device == torch.device("cuda", index)
        store.begin_wave(0)
        store.put_state(0, 1, _state(torch.device("cuda", index)))
        store.put_crit(0, 1, _crit("cuda"))
        store.free_state(0, 1)
        store.free_crit(0, 1)
        store.end_wave()


def test_release_uses_storage_recorded_at_put(dev):
    """A storage resized or reallocated after the put (an offload path) still releases its
    bookkeeping; the changed storage is not poisoned."""
    store = _store(dev, poison_on_free=True)
    shrunk, moved = _state(dev), _state(dev)
    store.put_state(0, 1, shrunk)
    store.put_state(0, 2, moved)
    assert store.stats().live_bytes == 2 * _nbytes(_state(dev))
    shrunk.untyped_storage().resize_(0)
    moved.untyped_storage().resize_(4096)
    store.free_state(0, 1)
    store.free_state(0, 2)
    assert store.stats().live_bytes == 0 and store.stats().live_entries == 0
    store.end_wave()


def test_in_place_change_after_put_raises(dev):
    store = _store(dev)
    s, crit = _state(dev), _crit(dev)
    store.put_state(0, 1, s)
    store.put_crit(0, 1, crit)
    store.get_state(0, 1)  # unchanged: fine
    s.add_(1.0)
    with pytest.raises(StateStoreError, match=r"s\[sample 0, round 1\]: modified in place"):
        store.get_state(0, 1)
    with pytest.raises(StateStoreError, match=r"s\[sample 0, round 1\]: modified in place"):
        store.free_state(0, 1)
    with pytest.raises(StateStoreError, match=r"s\[sample 0, round 1\]: modified in place"):
        store.take_state(0, 1)
    crit.view(-1)[0] = 7.0  # through a view: same version counter
    with pytest.raises(StateStoreError, match=r"crit\[sample 0, round 1\]: modified in place"):
        store.get_crit(0, 1)
    assert store.stats().live_entries == 2  # a failed release keeps the entry
    store.reset()

    # Inference tensors have no version counter: stored and released without the check.
    store = _store(dev)
    with torch.inference_mode():
        state = _state(dev)
    store.put_state(0, 1, state)
    store.free_state(0, 1)
    store.end_wave()


def test_shape_checks_are_order_independent(dev):
    # crit before the injection: the injection's c must match the crit rows.
    store = _store(dev)
    store.put_crit(0, 0, _crit(dev, rows=3))
    with pytest.raises(
        StateStoreError, match=r"e\[sample 0\]: 1 rows, crit\[sample 0, round 0\] has 3"
    ):
        store.put_injection(0, _state(dev))
    with pytest.raises(
        StateStoreError, match=r"crit\[sample 0, round 1\]: 1 rows, crit\[sample 0, round 0\]"
    ):
        store.put_crit(0, 1, _crit(dev, rows=1))
    store.put_injection(0, _state(dev, rows=3))
    store.reset()

    # A state before the injection: the injection must match the state.
    store = _store(dev)
    store.put_state(0, 1, _state(dev))
    with pytest.raises(StateStoreError, match=r"e\[sample 0\]: shape .* of s\[sample 0, round 1\]"):
        store.put_injection(0, torch.zeros(S + 1, C, NH, device=dev))
    with pytest.raises(
        StateStoreError, match=r"crit\[sample 0, round 0\]: 2 rows, s\[sample 0, round 1\] has 1"
    ):
        store.put_crit(0, 0, _crit(dev, rows=2))
    store.put_injection(0, _state(dev))
    # Other samples are independent.
    store.put_state(1, 1, _state(dev, rows=2))
    store.reset()


def test_store_key_text():
    assert str(StoreKey("state", 4, 2)) == "s[sample 4, round 2]"
    assert str(StoreKey("crit", 4, 0)) == "crit[sample 4, round 0]"
    assert str(StoreKey("injection", 4, 0)) == "e[sample 4]"


# ---------------------------------------------------------------------------
# Multi-EU moves: send_* / put_received_* (S1-S3)
# ---------------------------------------------------------------------------


def test_send_then_receive_round_trip(dev):
    """S1: an entry that left by a move is released as "sent"; a copy that arrives later
    (an item move back to a former sample-move source) is installed; a received injection
    copy lives like an origin one; the wave still ends empty."""
    source = _store(dev)
    s1, c1 = _state(dev, 1.5), _crit(dev)
    source.put_state(4, 1, s1)
    source.put_crit(4, 1, c1)
    source.put_injection(4, _state(dev, 0.5))
    source.send_state(4, 1)
    source.send_crit(4, 1)
    assert source.stats().live_entries == 1
    with pytest.raises(StateStoreError, match=r"s\[sample 4, round 1\]: sent to another EU"):
        source.get_state(4, 1)
    with pytest.raises(StateStoreError, match=r"cannot send, sent to another EU"):
        source.send_state(4, 1)
    # The checkpoint comes back by a later move.
    source.put_received_state(4, 1, s1.clone())
    source.put_received_crit(4, 1, c1.clone())
    assert torch.equal(source.get_state(4, 1), s1)
    source.free_state(4, 1)
    source.free_crit(4, 1)
    assert source.holds_injection(4)
    source.free_injection(4)
    assert not source.holds_injection(4)
    source.end_wave()

    destination = _store(dev)
    destination.put_received_state(4, 1, s1)
    destination.put_received_crit(4, 1, c1)
    destination.put_received_injection(4, _state(dev, 0.5))
    assert destination.holds_injection(4)
    assert torch.equal(destination.get_crit(4, 1), c1)
    destination.take_state(4, 1)
    destination.free_crit(4, 1)
    destination.free_injection(4)
    destination.end_wave()


def test_receive_after_consumption_raises(dev):
    """S2: an entry consumed here ("taken") or evicted ("freed") can never come back."""
    store = _store(dev)
    store.put_state(0, 2, _state(dev))
    store.take_state(0, 2)
    with pytest.raises(StateStoreError, match=r"already taken this wave .*only an entry sent"):
        store.put_received_state(0, 2, _state(dev))
    store.put_crit(0, 2, _crit(dev))
    store.free_crit(0, 2)
    with pytest.raises(StateStoreError, match=r"already freed this wave"):
        store.put_received_crit(0, 2, _crit(dev))
    store.put_injection(0, _state(dev))
    store.free_injection(0)
    with pytest.raises(StateStoreError, match=r"e\[sample 0\]: already freed"):
        store.put_received_injection(0, _state(dev))
    store.put_received_state(1, 1, _state(dev))
    with pytest.raises(StateStoreError, match=r"already stored"):
        store.put_received_state(1, 1, _state(dev))
    store.free_state(1, 1)
    store.end_wave()


def test_origin_write_after_send_raises(dev):
    """S3: origin writes stay strictly write-once, also after a send."""
    store = _store(dev, poison_on_free=True)
    state = _state(dev, 3.0)
    alias = state.view(-1)
    store.put_state(0, 1, state)
    store.send_state(0, 1)
    assert torch.isnan(alias).all()  # released bytes are poisoned like a free
    with pytest.raises(
        StateStoreError, match=r"already sent this wave \(an entry is written once\)"
    ):
        store.put_state(0, 1, _state(dev))
    store.end_wave()


# ---------------------------------------------------------------------------
# Block inputs and sweep traces (stage 4)
# ---------------------------------------------------------------------------


def test_block_inputs_round_trip_and_free_rules(dev):
    """Block inputs ``x[g, r, l]`` and trace entries: put / get / free / take, write-once per
    wave, keys per (sample, round, sublayer, kind), and the wave ends empty."""
    store = _store(dev, core_sublayers=4)
    x1, x2, t0 = _state(dev, 1.0), _state(dev, 2.0), _state(dev, 3.0)
    store.put_block_input(0, 2, 1, x1)
    store.put_block_input(0, 2, 2, x2)
    store.put_block_input(0, 2, 0, t0, kind="trace")
    assert torch.equal(store.get_block_input(0, 2, 1), x1)
    assert torch.equal(store.get_block_input(0, 2, 0, kind="trace"), t0)
    with pytest.raises(StateStoreError, match=r"x\[sample 0, round 2, sublayer 3\]: never"):
        store.get_block_input(0, 2, 3)
    with pytest.raises(StateStoreError, match=r"x\[sample 1, round 2, sublayer 1\]: never"):
        store.get_block_input(1, 2, 1)
    with pytest.raises(StateStoreError, match=r"already stored"):
        store.put_block_input(0, 2, 1, x1)
    assert store.stats().live_entries == 3
    store.free_block_input(0, 2, 1)
    with pytest.raises(StateStoreError, match=r"cannot free, already freed"):
        store.free_block_input(0, 2, 1)
    with pytest.raises(StateStoreError, match=r"already freed this wave"):
        store.put_block_input(0, 2, 1, x1)
    assert store.take_block_input(0, 2, 2) is not None
    with pytest.raises(StateStoreError, match=r"already taken"):
        store.get_block_input(0, 2, 2)
    with pytest.raises(StateStoreError, match=r"trace\[sample 0, round 2, sublayer 0\]"):
        store.end_wave()
    store.take_block_input(0, 2, 0, kind="trace")
    assert store.end_wave().live_entries == 0


def test_block_input_ranges_and_checks(dev):
    """Rounds >= 1; block sublayers in 1 .. L-1 and trace sublayers in 0 .. L-1 (only the
    lower bound without ``core_sublayers``); the round state's shape and dtype; in-place
    changes after the put raise."""
    store = _store(dev, core_sublayers=4, act_dtype=torch.float32)
    with pytest.raises(StateStoreError, match=r"block round must be >= 1"):
        store.put_block_input(0, 0, 1, _state(dev))
    with pytest.raises(StateStoreError, match=r"block sublayer must be in 1 \.\. 4 - 1"):
        store.put_block_input(0, 1, 0, _state(dev))
    with pytest.raises(StateStoreError, match=r"block sublayer must be in 1 \.\. 4 - 1"):
        store.put_block_input(0, 1, 4, _state(dev))
    with pytest.raises(StateStoreError, match=r"trace sublayer must be in 0 \.\. 4 - 1"):
        store.put_block_input(0, 1, 4, _state(dev), kind="trace")
    with pytest.raises(StateStoreError, match=r"not a per-sublayer entry kind"):
        store.put_block_input(0, 1, 1, _state(dev), kind="state")
    with pytest.raises(StateStoreError, match=r"dtype torch.bfloat16"):
        store.put_block_input(0, 1, 1, _state(dev, dtype=torch.bfloat16))
    store.put_state(0, 1, _state(dev))
    with pytest.raises(StateStoreError, match=r"shape \(8, 2, 16\) is not \(8, 1, 16\)"):
        store.put_block_input(0, 1, 1, _state(dev, rows=2))
    x = _state(dev, 5.0)
    store.put_block_input(0, 1, 1, x)
    x.add_(1.0)
    with pytest.raises(StateStoreError, match=r"x\[sample 0, round 1, sublayer 1\]: modified"):
        store.get_block_input(0, 1, 1)
    store.reset()
    unbounded = _store(dev)
    unbounded.put_block_input(0, 1, 99, _state(dev))
    unbounded.prefetch(0, 1, kind="block", sublayer=99)
    with pytest.raises(StateStoreError, match=r"x\[sample 0, round 1, sublayer 98\]: never"):
        unbounded.prefetch(0, 1, kind="block", sublayer=98)
    unbounded.reset()


def test_block_inputs_move_between_stores(dev):
    """``send_block_input`` releases an entry as sent (poisoned like a free); the destination
    installs it with ``put_received_block_input``; an entry consumed here cannot come back,
    one sent away can."""
    src, dst = _store(dev, poison_on_free=True), _store(dev)
    x = _state(dev, 4.0)
    alias = x.view(-1)
    src.put_block_input(3, 2, 1, x)
    copy = src.get_block_input(3, 2, 1).clone()
    src.send_block_input(3, 2, 1)
    assert torch.isnan(alias).all()
    with pytest.raises(StateStoreError, match=r"sent to another EU"):
        src.get_block_input(3, 2, 1)
    dst.put_received_block_input(3, 2, 1, copy)
    assert torch.equal(dst.get_block_input(3, 2, 1), copy)
    src.put_received_block_input(3, 2, 1, copy.clone())  # it comes back
    src.free_block_input(3, 2, 1)
    with pytest.raises(StateStoreError, match=r"already freed this wave"):
        src.put_received_block_input(3, 2, 1, copy.clone())
    dst.take_block_input(3, 2, 1)
    with pytest.raises(StateStoreError, match=r"already taken this wave"):
        dst.put_received_block_input(3, 2, 1, copy.clone())
    src.end_wave()
    dst.end_wave()


class _SublayerPlacement:
    """Records the keyword the store passes for per-sublayer kinds."""

    def __init__(self):
        self.calls = []

    def tier(self, sample, r, kind, sublayer=-1):
        self.calls.append((kind, sample, r, sublayer))
        return StateTier.HBM


def test_placement_sees_the_sublayer_of_block_entries(dev):
    placement = _SublayerPlacement()
    store = _store(dev, placement=placement)
    store.put_state(1, 2, _state(dev))
    store.put_block_input(1, 2, 3, _state(dev))
    store.put_block_input(1, 2, 0, _state(dev), kind="trace")
    assert placement.calls == [("state", 1, 2, -1), ("block", 1, 2, 3), ("trace", 1, 2, 0)]
    assert HBMOnlyPlacement().tier(0, 1, "block", sublayer=2) is StateTier.HBM
    store.reset()


def test_bytes_per_kind_count_a_shared_storage_once(dev):
    """A trace entry that aliases a block input (the output of sublayer l - 1 is the input of
    sublayer l) shares its bytes: counted once, for the kind of the earliest-put live entry,
    and re-attributed to the other kind when that entry is released first."""
    store = _store(dev)
    unit = _nbytes(_state(dev))
    shared, other = _state(dev, 1.0), _state(dev, 2.0)
    store.put_state(0, 1, _state(dev))
    store.put_block_input(0, 1, 1, shared)
    store.put_block_input(0, 1, 0, shared, kind="trace")  # the same storage
    store.put_block_input(0, 1, 1, other, kind="trace")
    stats = store.stats()
    assert stats.live_bytes == 3 * unit and stats.live_entries == 4
    assert stats.live_bytes_by_kind == {"state": unit, "block": unit, "trace": unit}
    store.free_block_input(0, 1, 1)  # the trace entry keeps the storage
    stats = store.stats()
    assert stats.live_bytes == 3 * unit
    assert stats.live_bytes_by_kind == {"state": unit, "trace": 2 * unit}
    store.take_block_input(0, 1, 0, kind="trace")
    store.take_block_input(0, 1, 1, kind="trace")
    store.free_state(0, 1)
    final = store.end_wave()
    assert final.live_bytes_by_kind == {}
    assert final.peak_bytes_by_kind == {"state": unit, "block": unit, "trace": 2 * unit}
    assert final.peak_bytes == 3 * unit


def test_store_key_text_of_block_entries():
    assert str(StoreKey("block", 4, 2, 3)) == "x[sample 4, round 2, sublayer 3]"
    assert str(StoreKey("trace", 4, 2, 0)) == "trace[sample 4, round 2, sublayer 0]"
    assert StoreKey("state", 4, 2) == ("state", 4, 2, -1)
