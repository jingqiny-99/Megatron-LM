# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The host-side bookkeeping behind the lifted loop guards (CPU, no process group):

* GTP's prefetch ledger (``generalized_tensor_parallelism._GatherLedger``): a consume takes a
  prefetched buffer only if the buffer still holds the gather issued for that weight and slot
  since its last consume. Driven through the consume orders a fixed-depth loop produces: the
  next round's first weight, a prefetch issued for the other direction, a shared buffer
  rewritten by another weight's gather.

The GPU test of the same path is ``test_loop_gtp_gpu.py``.
"""

from types import SimpleNamespace

from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp

# ---------------------------------------------------------------------------------------------
# GTP prefetch ledger
# ---------------------------------------------------------------------------------------------


class _Weight:
    """A stand-in weight (the ledger keys weights and buffers by identity)."""


class _Buffer:
    """A stand-in gather buffer."""


def test_a_prefetch_is_current_until_its_buffer_is_rewritten():
    ledger = gtp._GatherLedger()
    a, b, shared = _Weight(), _Weight(), _Buffer()
    ledger.record(a, "fwd", shared, prefetch=True)
    assert ledger.is_current(a, "fwd", shared)
    assert ledger.is_current(a, "fwd", shared)  # looking does not consume
    # Another weight's gather into the shared buffer (an on-demand one, or a prefetch).
    ledger.record(b, "fwd", shared, prefetch=False)
    assert not ledger.take(a, "fwd", shared)
    # take() consumed a's pending entry: a later look finds none, even after b leaves.
    assert not ledger.is_current(a, "fwd", shared)


def test_take_consumes_the_pending_prefetch():
    ledger = gtp._GatherLedger()
    a, buffer = _Weight(), _Buffer()
    ledger.record(a, "fwd", buffer, prefetch=True)
    assert ledger.take(a, "fwd", buffer)
    # A second consume of the same weight (the next round) has no gather of its own.
    assert not ledger.take(a, "fwd", buffer)
    # An on-demand gather is consumed on the spot: nothing pending.
    ledger.record(a, "fwd", buffer, prefetch=False)
    assert not ledger.take(a, "fwd", buffer)


def test_the_slot_must_match():
    """A backward prefetch is not a forward consume's gather (FP8 gathers columnwise for the
    backward; BF16 tickets of one key may still be different buffers)."""
    ledger = gtp._GatherLedger()
    a, buffer = _Weight(), _Buffer()
    ledger.record(a, "bwd", buffer, prefetch=True)
    assert not ledger.take(a, "fwd", buffer)
    assert not ledger.take(a, "recompute", buffer)
    assert ledger.take(a, "bwd", buffer)


def test_the_buffer_must_match():
    """The consume reads its own ticket's buffer; a gather written elsewhere does not count."""
    ledger = gtp._GatherLedger()
    a, mine, other = _Weight(), _Buffer(), _Buffer()
    ledger.record(a, "fwd", other, prefetch=True)
    assert not ledger.take(a, "fwd", mine)


def test_a_newer_prefetch_of_the_same_weight_supersedes_the_older():
    """The last core weight of a round prefetches the coda's first weight every round; only the
    last of those prefetches is consumed, and it is current."""
    ledger = gtp._GatherLedger()
    coda_first, core_first = _Weight(), _Weight()
    shared = _Buffer()  # coda_first and core_first resolve to the same cache key and parity
    for _ in range(3):  # rounds 1-3: coda_first prefetched, then core_first gathered on demand
        ledger.record(coda_first, "fwd", shared, prefetch=True)
        ledger.record(core_first, "fwd", shared, prefetch=False)
    ledger.record(coda_first, "fwd", shared, prefetch=True)  # round 4
    assert ledger.take(coda_first, "fwd", shared)


def test_clear_forgets_everything():
    ledger = gtp._GatherLedger()
    a, buffer = _Weight(), _Buffer()
    ledger.record(a, "fwd", buffer, prefetch=True)
    ledger.clear()
    assert not ledger.is_current(a, "fwd", buffer)


def test_the_cache_clear_and_the_gtp_reset_clear_the_global_ledger():
    weight, buffer = _Weight(), _Buffer()
    gtp._GATHER_LEDGER.record(weight, "fwd", buffer, prefetch=True)
    gtp.GTPWeightCache().clear()
    assert not gtp._GATHER_LEDGER.is_current(weight, "fwd", buffer)
    gtp._GATHER_LEDGER.record(weight, "fwd", buffer, prefetch=True)
    gtp.reset_gtp_state()
    assert not gtp._GATHER_LEDGER.is_current(weight, "fwd", buffer)


def test_gather_slots():
    assert gtp._gather_slot(True) == "fwd"
    assert gtp._gather_slot(False) == "bwd"
    assert gtp._gather_slot(True, recompute=True) == "recompute"


def test_verification_is_off_by_default_and_on_eager_uncaptured_chains_only(monkeypatch):
    assert not gtp.GTPRematConfig().verify_prefetched_gathers
    eager = SimpleNamespace(chain_id=gtp.GTPChain.UNGRAPHED.value)
    graphed = SimpleNamespace(chain_id=gtp.GTPChain.GRAPHED.value)
    monkeypatch.setattr(gtp.GTP_CONFIG, "verify_prefetched_gathers", False)
    assert not gtp.GTPShardedParam._verify_gathers(eager)
    monkeypatch.setattr(gtp.GTP_CONFIG, "verify_prefetched_gathers", True)
    capturing = [False]
    monkeypatch.setattr(gtp.torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    assert gtp.GTPShardedParam._verify_gathers(eager)
    assert not gtp.GTPShardedParam._verify_gathers(graphed)
    capturing[0] = True  # a captured schedule replays without the host-side check
    assert not gtp.GTPShardedParam._verify_gathers(eager)
