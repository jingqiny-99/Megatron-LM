# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The host-side bookkeeping behind the lifted loop guards (CPU, no process group):

* GTP's prefetch ledger (``generalized_tensor_parallelism._GatherLedger``): a consume takes a
  prefetched buffer only if the buffer still holds the gather issued for that weight and slot
  since its last consume. Driven through the consume orders a fixed-depth loop produces: the
  next round's first weight, a prefetch issued for the other direction, a shared buffer
  rewritten by another weight's gather.
* DDP's overlapped grad-reduce trigger (``_ParamAndGradBucketGroup.register_grad_ready`` /
  ``reset``), on a stand-in bucket group: the first batch records how often each parameter is
  ready in the last microbatch (a GTP shard: once per use), later batches start the reduction
  when every parameter reached its count, and a count above it raises instead of losing a use.
* The GPU tests' Megatron-FSDP gradient reader (``strategy_grad_utils.fsdp_bucket_part``) on
  emulated buffers: every strategy, and DP-Outer sharding's logical shard order.
* The GPU tests' optimizer (``strategy_grad_utils.SGD_AS_ADAM``): Adam that steps as SGD without
  momentum, since the distributed optimizer refuses SGD.

The GPU tests of the same paths are ``test_loop_gtp_gpu.py`` and ``test_loop_ddp_overlap_gpu.py``.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
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


# ---------------------------------------------------------------------------------------------
# DDP grad-ready counts (overlap_grad_reduce)
# ---------------------------------------------------------------------------------------------


class _Param:
    shape = (4, 4)


def _bucket_group(params):
    """A stand-in for ``_ParamAndGradBucketGroup`` with the state ``register_grad_ready`` and
    ``reset`` use; ``starts`` counts the reductions started."""
    group = SimpleNamespace(
        ddp_config=SimpleNamespace(overlap_grad_reduce=True),
        param_to_bucket={param: object() for param in params},
        params=set(params),
        golden_per_param_grad_ready_counts={},
        per_param_grad_ready_counts={},
        is_last_microbatch=True,
        is_first_batch=True,
        grad_reduce_finished=False,
        starts=0,
    )

    def start_grad_sync(force_all_reduce=False):
        group.starts += 1

    group.start_grad_sync = start_grad_sync
    return group


def _ready(group, param, times=1):
    for _ in range(times):
        _ParamAndGradBucketGroup.register_grad_ready(group, param)


def _first_batch(group, counts, microbatches=2):
    """One batch of ``microbatches``: only the last one counts (``no_sync`` clears
    ``is_last_microbatch`` for the others)."""
    for microbatch in range(microbatches):
        group.is_last_microbatch = microbatch == microbatches - 1
        for param, count in counts.items():
            _ready(group, param, count)
    group.is_last_microbatch = True
    _ParamAndGradBucketGroup.reset(group)


def test_counts_are_recorded_from_the_first_batch_last_microbatch():
    core, coda = _Param(), _Param()
    group = _bucket_group([core, coda])
    _first_batch(group, {core: 4, coda: 1})
    assert group.golden_per_param_grad_ready_counts == {core: 4, coda: 1}
    assert not group.is_first_batch and group.starts == 0


def test_the_reduction_starts_after_the_last_use_of_every_parameter():
    """A GTP core shard is ready once per round (4 per microbatch): the bucket group's reduction
    starts only after its 4th use and the coda weight's one, never earlier."""
    core, coda = _Param(), _Param()
    group = _bucket_group([core, coda])
    _first_batch(group, {core: 4, coda: 1})
    group.is_last_microbatch = False  # an earlier microbatch of the batch: not counted
    _ready(group, core, 4)
    _ready(group, coda)
    assert group.starts == 0 and group.per_param_grad_ready_counts == {}
    group.is_last_microbatch = True
    _ready(group, coda)
    for use in range(1, 5):
        assert group.starts == 0, use
        _ready(group, core)
    assert group.starts == 1


def test_a_use_beyond_the_recorded_count_raises():
    """More uses than in the first batch (a depth change) would miss the started reduction."""
    core, coda = _Param(), _Param()
    group = _bucket_group([core, coda])
    _first_batch(group, {core: 4, coda: 1})
    _ready(group, coda)
    _ready(group, core, 4)
    assert group.starts == 1
    with pytest.raises(RuntimeError, match="more than the 4 times of the first batch"):
        _ready(group, core)


def test_a_use_beyond_the_count_raises_before_the_reduction_too():
    core, coda = _Param(), _Param()
    group = _bucket_group([core, coda])
    _first_batch(group, {core: 1, coda: 1})
    with pytest.raises(RuntimeError, match="became grad-ready 2 times"):
        _ready(group, core, 2)
    assert group.starts == 0


# ---------------------------------------------------------------------------------------------
# The GPU tests' Megatron-FSDP gradient reader (strategy_grad_utils.fsdp_bucket_part)
# ---------------------------------------------------------------------------------------------

BUCKET = 4 * 64  # elements; a multiple of every emulated group size


def _hfsdp_logical_rank(rank, outer, inner):
    """FSDPDistributedIndex.get_logical_hybrid_fsdp_rank: DP group rank (outer-major) to the
    shard index DP-Outer sharding uses (DP-Shard-major)."""
    return (rank % inner) * outer + rank // inner


def _emulated_buffers(strategy, world, logical_rank, reduced):
    """The main gradient buffer of every rank of a ``world``-rank group after the reduction:
    rank ``r`` owns the shard at ``logical_rank(r)``; an ``optim`` (unsharded) buffer keeps
    rank-local, unreduced values outside its shard."""
    from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
        BucketIndex,
        _get_dp_buffer_shard_bucket_index,
    )

    bucket = BucketIndex(bucket_id=0, global_data_index=0, size=BUCKET, items=[])
    distributed = strategy != "optim"
    buffers = []
    for rank in range(world):
        index = _get_dp_buffer_shard_bucket_index(bucket, distributed, world, logical_rank(rank))
        shard = slice(index.bucket_data_index, index.bucket_data_index + index.size)
        if distributed:
            data = reduced[shard].clone()
        else:
            data = torch.full((BUCKET,), 100.0 + rank)  # local, unreduced sums
            data[shard] = reduced[shard]
        buffers.append(
            SimpleNamespace(
                bucket_index=bucket, shard_bucket_index=index, dtype=torch.float32, data=data
            )
        )
    return buffers


@pytest.mark.parametrize(
    "layout", ["fsdp", "hfsdp", "hfsdp_expert_shard1"], ids=lambda layout: layout
)
@pytest.mark.parametrize("strategy", ["optim_grads_params", "optim_grads", "optim"])
def test_fsdp_bucket_parts_sum_to_the_reduced_bucket(layout, strategy):
    """Summing every rank's part gives the reduced bucket for every strategy and for DP-Outer
    sharding's logical shard order. The previous reader (all-gather in group order, the whole
    slice of an unsharded buffer) misplaced HFSDP shards and read unreduced values."""
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    if layout != "fsdp" and strategy == "optim":
        pytest.skip("DP-Outer sharding refuses a replicated-gradient strategy")
    outer, inner = dict(fsdp=(1, 4), hfsdp=(2, 2), hfsdp_expert_shard1=(2, 1))[layout]
    world = outer * inner
    logical = lambda rank: _hfsdp_logical_rank(rank, outer, inner)  # noqa: E731
    reduced = torch.randn(BUCKET, generator=torch.Generator().manual_seed(7))
    buffers = _emulated_buffers(strategy, world, logical, reduced)
    total = torch.zeros(BUCKET)
    for gbuf in buffers:
        part, partial = sgu.fsdp_bucket_part(gbuf, strategy)
        assert partial
        total += part
    assert torch.equal(total, reduced)

    # What the previous reader produced.
    if strategy == "optim":
        assert not torch.equal(buffers[0].data, reduced)
    elif world > 1:
        gathered = torch.cat([gbuf.data for gbuf in buffers])  # group-rank order
        assert torch.equal(gathered, reduced) == all(logical(r) == r for r in range(world))


def test_fsdp_bucket_part_of_a_replicated_bucket_is_the_whole_bucket():
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    data = torch.randn(BUCKET)
    gbuf = SimpleNamespace(
        bucket_index=SimpleNamespace(global_data_index=0, size=BUCKET), data=data
    )
    part, partial = sgu.fsdp_bucket_part(gbuf, "no_shard")
    assert not partial and torch.equal(part, data)


# ---------------------------------------------------------------------------------------------
# The GPU tests' optimizer (strategy_grad_utils.SGD_AS_ADAM)
# ---------------------------------------------------------------------------------------------


def test_sgd_as_adam_steps_as_sgd_without_momentum():
    """The distributed optimizer refuses SGD, so the GPU tests run SGD without momentum as Adam
    (beta1 = beta2 = 0, eps far above the gradients): every step's update equals SGD's to fp32
    rounding, also on a step after another (no momentum, no bias-correction drift)."""
    from megatron.core.optimizer import OptimizerConfig
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    config = OptimizerConfig(**sgu.SGD_AS_ADAM)
    assert config.optimizer == "adam"
    generator = torch.Generator().manual_seed(0)
    adam_param = torch.nn.Parameter(torch.zeros(4096))
    sgd_param = torch.nn.Parameter(torch.zeros(4096))
    adam = torch.optim.Adam(
        [adam_param],
        lr=config.lr,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=0.0,
    )
    sgd = torch.optim.SGD([sgd_param], lr=sgu.SGD_LR, momentum=0.0)
    # Each step starts from zero weights, so the weights after it are the update itself (no
    # cancellation hides a wrong update); entries reach ~12 at scale 3, past |g| < 4.
    for scale in (1e-6, 1.0, 3.0):
        grad = torch.randn(4096, generator=generator) * scale
        for param in (adam_param, sgd_param):
            param.data.zero_()
            param.grad = grad.clone()
        adam.step()
        sgd.step()
        torch.testing.assert_close(adam_param.detach(), sgd_param.detach(), rtol=1e-6, atol=0.0)
        torch.testing.assert_close(adam_param.detach(), -0.1 * grad, rtol=1e-6, atol=0.0)
