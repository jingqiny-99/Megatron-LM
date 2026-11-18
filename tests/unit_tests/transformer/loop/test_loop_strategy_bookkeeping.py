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
* The GPU tests' teacher forcing of step 2: the Megatron-FSDP writer
  (``fsdp_force_bucket_part``) is the reader's inverse on the same emulated buffers and keeps a
  ZeRO-1 buffer's unreduced sums outside the shard; a GTP shard's rows (``gtp_rows``)
  reassemble to the full tensor with zero padding; the optimizer windows (``item_window``) cover
  every item once over the ranks; the DDP writer and the exact comparison.
* The GPU tests' Megatron-FSDP reduction schedule expectation (``reduced_every_microbatch``,
  ``expected_once_per_step``, ``observed_once_per_step``) and the per-backward mutation check
  (``assert_per_backward_mutation``).
* The GPU tests' optimizer (``strategy_grad_utils.SGD_AS_ADAM``): Adam that steps as SGD without
  momentum, since the distributed optimizer refuses SGD.
* The GTP GPU test's precise embedding reference
  (``strategy_grad_utils.install_per_use_fp32_embedding``): each lookup's BF16 gradient is added
  into the FP32 ``main_grad`` on its own (bitwise the FP32 sum of the per-use gradients), DDP's
  hook then adds nothing, and the plain deterministic lookup it replaces sums the uses in BF16
  first (why the old reference is off); the installer refuses every other layout.

The GPU tests of the same paths are ``test_loop_gtp_gpu.py``, ``test_loop_ddp_overlap_gpu.py``,
``test_loop_hsdp_gpu.py`` and ``test_loop_fsdp_per_use_gpu.py``.
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
# The GPU tests' teacher forcing of step 2 (strategy_grad_utils)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layout", ["fsdp", "hfsdp", "hfsdp_expert_shard1"], ids=lambda layout: layout
)
@pytest.mark.parametrize("strategy", ["optim_grads_params", "optim_grads", "optim"])
def test_fsdp_force_bucket_part_is_the_readers_inverse(layout, strategy):
    """Forcing a bucket on every rank and reading it back gives the forced bucket, for every
    strategy and DP-Outer sharding's logical shard order; a ZeRO-1 buffer keeps its unreduced
    local sums outside the shard (the optimizer reads only its shard)."""
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    if layout != "fsdp" and strategy == "optim":
        pytest.skip("DP-Outer sharding refuses a replicated-gradient strategy")
    outer, inner = dict(fsdp=(1, 4), hfsdp=(2, 2), hfsdp_expert_shard1=(2, 1))[layout]
    world = outer * inner
    logical = lambda rank: _hfsdp_logical_rank(rank, outer, inner)  # noqa: E731
    reduced = torch.randn(BUCKET, generator=torch.Generator().manual_seed(1))
    buffers = _emulated_buffers(strategy, world, logical, reduced)
    before = [gbuf.data.clone() for gbuf in buffers]
    forced = torch.randn(BUCKET, generator=torch.Generator().manual_seed(2))
    for gbuf in buffers:
        sgu.fsdp_force_bucket_part(gbuf, strategy, forced)
    total = torch.zeros(BUCKET)
    for gbuf, old in zip(buffers, before):
        part, _ = sgu.fsdp_bucket_part(gbuf, strategy)
        total += part
        index = gbuf.shard_bucket_index
        outside = torch.ones(BUCKET, dtype=torch.bool)
        outside[index.bucket_data_index : index.bucket_data_index + index.size] = False
        if strategy == "optim":
            assert torch.equal(gbuf.data[outside], old[outside])
    assert torch.equal(total, forced)


def test_fsdp_force_bucket_part_of_a_replicated_bucket_writes_the_whole_bucket():
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    gbuf = SimpleNamespace(
        bucket_index=SimpleNamespace(global_data_index=0, size=BUCKET), data=torch.zeros(BUCKET)
    )
    forced = torch.randn(BUCKET)
    sgu.fsdp_force_bucket_part(gbuf, "no_shard", forced)
    assert torch.equal(gbuf.data, forced)
    with pytest.raises(AssertionError):
        sgu.fsdp_force_bucket_part(gbuf, "no_shard", forced[:-1])


@pytest.mark.parametrize("rows_full,size", [(7, 4), (8, 4), (3, 2), (1, 4)])
def test_gtp_rows_reassemble_to_the_full_tensor(rows_full, size):
    """Every rank's rows of the zero-padded tensor, all-gathered in rank order and stripped of
    the padding (``_full_grads``), give the full tensor back; the padding rows are zero."""
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    full = torch.randn(rows_full, 5)
    rows = -(-rows_full // size)
    shards = [sgu.gtp_rows(full, rows, size, rank) for rank in range(size)]
    assert all(shard.shape == (rows, 5) for shard in shards)
    gathered = torch.cat(shards)
    assert torch.equal(gathered[:rows_full], full)
    assert not gathered[rows_full:].any()


def _brute_window(item_start, item_size, shard_start, shard_size):
    inside = [
        offset
        for offset in range(item_size)
        if shard_start <= item_start + offset < shard_start + shard_size
    ]
    return (inside[0], inside[-1] + 1) if inside else None


def test_item_window_covers_every_item_once_over_the_ranks():
    """The optimizer window of an item on each rank is its intersection with the rank's shard
    (checked element by element), the windows of all ranks tile the item, and an unsharded
    optimizer state holds the whole item."""
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    bucket, world = 96, 4
    shard = bucket // world
    items = [(0, 10), (10, 30), (40, 8), (48, 48), (5, 0)]  # (start, size), one empty
    for start, size in items:
        covered = []
        for rank in range(world):
            lo, hi = sgu.item_window(start, size, rank * shard, shard, True)
            assert 0 <= lo <= hi <= size
            brute = _brute_window(start, size, rank * shard, shard)
            assert (lo, hi) == brute if brute else lo == hi, (start, size, rank, lo, hi)
            covered.extend(range(lo, hi))
        assert covered == list(range(size)), (start, size)
        assert sgu.item_window(start, size, shard, shard, False) == (0, size)


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(4, 6, bias=False)
        self.b = torch.nn.Linear(6, 2)
        self.register_buffer("bias_buffer", torch.zeros(3))
        self.register_buffer("scratch", torch.zeros(2), persistent=False)
        for param in self.parameters():
            param.main_grad = torch.zeros_like(param)


def test_force_main_grads_copies_in_place_and_shards():
    """The DDP writer copies into the existing ``main_grad`` (the optimizer's view stays
    valid), whole or through ``shard``; a shape mismatch is refused."""
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    net = _Net()
    storage = {name: param.main_grad for name, param in net.named_parameters()}
    forced = {name: torch.randn_like(param) for name, param in net.named_parameters()}
    sgu.force_main_grads(net, forced)
    for name, param in net.named_parameters():
        assert param.main_grad is storage[name] and torch.equal(param.main_grad, forced[name])
    full = {
        name: torch.randn(value.shape[0] * 2, *value.shape[1:]) for name, value in forced.items()
    }
    sgu.force_main_grads(net, full, shard=lambda name, param, value: value[: param.shape[0]])
    for name, param in net.named_parameters():
        assert torch.equal(param.main_grad, full[name][: param.shape[0]])
    with pytest.raises(AssertionError):
        sgu.force_main_grads(net, full)


def test_mismatches_are_exact_and_name_dtype_and_size():
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    want = {"x": torch.ones(4), "y": torch.zeros(2, dtype=torch.bfloat16), "z": torch.ones(3)}
    have = {"x": torch.ones(4), "y": torch.zeros(2, dtype=torch.bfloat16), "z": torch.ones(2)}
    have["y"][1] = 2**-10
    found = sgu.mismatches(have, want)
    assert set(found) == {"y", "z"}
    assert found["y"] == (1, "bfloat16", 2**-10)
    assert found["z"][0] == "shape"
    assert sgu.mismatches(want, want) == {}
    # Values, not bit patterns: a read-back through a sum with zeros turns -0.0 into +0.0.
    assert sgu.mismatches({"x": -torch.zeros(1)}, {"x": torch.zeros(1)}) == {}
    assert set(
        sgu.mismatches({"x": torch.full((1,), float("nan"))}, {"x": torch.full((1,), float("nan"))})
    ) == {"x"}
    # A different dtype is a mismatch even when the values are representable in both.
    found = sgu.mismatches({"x": torch.ones(4, dtype=torch.bfloat16)}, {"x": torch.ones(4)})
    assert found == {"x": ("dtype", "torch.bfloat16", "torch.float32")}


def test_persistent_buffers_and_float16_masters():
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    net = _Net()
    assert set(sgu.persistent_buffers(net)) == {"bias_buffer"}
    master_a = net.a.weight.detach().float() + 1
    leaf = SimpleNamespace(
        float16_groups=[[net.a.weight]],
        fp32_from_float16_groups=[[master_a]],
        fp32_from_fp32_groups=[[net.b.weight, net.b.bias]],
    )
    masters = sgu.float16_optimizer_masters(SimpleNamespace(chained_optimizers=[leaf]), net)
    assert set(masters) == {"a.weight", "b.weight", "b.bias"}
    assert torch.equal(masters["a.weight"], master_a)
    assert torch.equal(masters["b.bias"], net.b.bias.detach())
    with pytest.raises(AssertionError):  # a trainable parameter without a master
        sgu.float16_optimizer_masters(
            SimpleNamespace(
                float16_groups=[], fp32_from_float16_groups=[], fp32_from_fp32_groups=[]
            ),
            net,
        )


def test_slice_mismatches_compare_the_window():
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    full = {"w": torch.arange(12.0).reshape(3, 4)}
    slices = {"w": ((2, 7), torch.arange(2.0, 7.0))}
    assert sgu.slice_mismatches(slices, full) == {}
    slices = {"w": ((2, 7), torch.arange(3.0, 8.0))}
    assert set(sgu.slice_mismatches(slices, full)) == {"w"}
    assert sgu.slice_relative_difference(slices, full) > 0
    assert sgu.slice_relative_difference({"w": ((0, 0), torch.zeros(0))}, full) == 0.0
    # A bf16 slice is not compared with the fp32 masters after rounding them: a mismatch.
    slices = {"w": ((2, 7), torch.arange(2.0, 7.0).bfloat16())}
    assert sgu.slice_mismatches(slices, full)["w"][0] == "dtype"


def test_assert_teacher_forced_needs_readback_and_start():
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    sgu.assert_teacher_forced(dict(readback=0, start={}, agreed=0), "arm")
    for forcing in (
        None,
        dict(readback=1, start={}, agreed=0),
        dict(readback=0, start={}, agreed=2),  # another rank differs
        dict(readback=0, start={"w": (1, "float32", 1e-9)}, agreed=1),
    ):
        with pytest.raises(AssertionError):
            sgu.assert_teacher_forced(forcing, "arm")


_ITEMS = [(0, 10), (10, 70), (80, 100), (180, 76)]  # (start, size) in a BUCKET-sized bucket


def _emulated_weight_buffers(strategy, world, logical_rank, weights):
    """The model weight buffer of every rank of a ``world``-rank group right after the install
    (``weights``: the installed bucket): a sharded buffer stores its shard; an unsharded one
    (``optim`` / ``optim_grads``) the whole bucket, where only its (virtual) shard is fresh, the
    rest stale."""
    from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
        BucketIndex,
        _get_dp_buffer_shard_bucket_index,
    )

    bucket = BucketIndex(bucket_id=0, global_data_index=0, size=BUCKET, items=[])
    items = {
        item_id: SimpleNamespace(global_data_index=start, size=size)
        for item_id, (start, size) in enumerate(_ITEMS)
    }
    distributed = strategy == "optim_grads_params"
    buffers = []
    for rank in range(world):
        index = _get_dp_buffer_shard_bucket_index(bucket, distributed, world, logical_rank(rank))
        shard = slice(index.bucket_data_index, index.bucket_data_index + index.size)
        if distributed:
            data = weights[shard].clone()
        else:
            data = torch.full((BUCKET,), -7.0)  # stale outside the shard
            data[shard] = weights[shard]
        buffers.append(
            SimpleNamespace(
                bucket_index=bucket, shard_bucket_index=index, item_index_map=items, data=data
            )
        )
    return buffers


@pytest.mark.parametrize(
    "layout", ["fsdp", "hfsdp", "hfsdp_expert_shard1"], ids=lambda layout: layout
)
@pytest.mark.parametrize("strategy", ["optim_grads_params", "optim_grads", "optim"])
def test_buffer_item_shard_reads_what_each_rank_installs(layout, strategy):
    """Each rank's part of every item of the model weight buffer is its shard's intersection
    with the item (the optimizer window, ``item_window``), read at the right place of a sharded
    or an unsharded buffer (never its stale rest), and the parts of all ranks tile the item."""
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    outer, inner = dict(fsdp=(1, 4), hfsdp=(2, 2), hfsdp_expert_shard1=(2, 1))[layout]
    world = outer * inner
    logical = lambda rank: _hfsdp_logical_rank(rank, outer, inner)  # noqa: E731
    weights = torch.randn(BUCKET, generator=torch.Generator().manual_seed(3))
    buffers = _emulated_weight_buffers(strategy, world, logical, weights)
    for item_id, (start, size) in enumerate(_ITEMS):
        covered = torch.zeros(size, dtype=torch.int64)
        for buf in buffers:
            (lo, hi), values = sgu.buffer_item_shard(buf, item_id)
            index = buf.shard_bucket_index
            assert (lo, hi) == sgu.item_window(
                start, size, index.bucket_data_index, index.size, True
            )
            assert torch.equal(values, weights[start + lo : start + hi])
            covered[lo:hi] += 1
        assert bool((covered == 1).all()), (item_id, covered)


def _emulated_fsdp(wbuf, params, no_shard_param):
    """A Megatron-FSDP stand-in for ``fsdp_model_weight_slices``: one sharded group whose
    parameters live in ``wbuf``, one ``no_shard`` group (no model weight buffer) and a frozen
    group, which is skipped."""
    sharded = SimpleNamespace(
        requires_grad=True, main_grad_buffer=object(), model_weight_buffer=wbuf, params=params
    )
    replicated = SimpleNamespace(
        requires_grad=True,
        main_grad_buffer=object(),
        model_weight_buffer=None,
        params=[no_shard_param],
    )
    frozen = SimpleNamespace(requires_grad=False, main_grad_buffer=None, params=[])
    names = {param: f"p{i}" for i, param in enumerate(params)}
    names[no_shard_param] = "replicated"
    wbuf.param_idx = {param: i for i, param in enumerate(params)}
    return SimpleNamespace(
        param_and_grad_buffer=SimpleNamespace(
            parameter_groups=[sharded, replicated, frozen], param_to_name=names
        )
    )


def test_fsdp_model_weight_slices_and_install_mismatches():
    """The installed compute weights are read from the model weight buffer's shard (or, for a
    ``no_shard`` group, the whole module parameter) and compared with the reference weights in
    the optimizer's windows; a wrong element in this rank's window, a wrong dtype, a window that
    differs from the optimizer's and a parameter not installed are all reported."""
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    weights = torch.randn(BUCKET, generator=torch.Generator().manual_seed(4))
    wbuf = _emulated_weight_buffers("optim_grads_params", 4, lambda rank: rank, weights)[1]
    params = [torch.empty(size) for _, size in _ITEMS]
    replicated = torch.randn(5)
    installed = sgu.fsdp_model_weight_slices(_emulated_fsdp(wbuf, params, replicated))
    reference = {f"p{i}": weights[start : start + size] for i, (start, size) in enumerate(_ITEMS)}
    reference["replicated"] = replicated.clone()
    main = {name: (window, None) for name, (window, _) in installed.items()}
    assert set(installed) == set(reference)
    assert installed["replicated"][0] == (0, 5)
    assert sgu.install_mismatches(installed, reference, main) == {}

    # Item 1 (bucket 10..80) meets rank 1's shard (bucket 64..128) in its elements 54..70.
    assert installed["p1"][0] == (54, 70)
    stale = dict(reference, p1=reference["p1"].clone())
    stale["p1"][60] += 1.0  # inside rank 1's window: caught
    assert set(sgu.install_mismatches(installed, stale, main)) == {"p1"}
    stale["p1"] = reference["p1"].clone()
    stale["p1"][0] += 1.0  # rank 0's part: not this rank's to check
    assert sgu.install_mismatches(installed, stale, main) == {}
    rounded = dict(reference, p2=reference["p2"].bfloat16())
    assert sgu.install_mismatches(installed, rounded, main)["p2"][0] == "dtype"
    shifted = dict(main, p3=((0, 1), None))
    assert sgu.install_mismatches(installed, reference, shifted)["p3"][0] == "window"
    extra = dict(main, missing=((0, 1), None))
    assert sgu.install_mismatches(installed, reference, extra)["missing"] == ("not installed",)


def test_fsdp_optimizer_grads_are_looked_up_as_the_hybrid_optimizer_does():
    """``decoupled_grad`` when the attribute exists, even as None (the host Adam then consumes
    no gradient: a mismatch), else ``grad``."""
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    forced = torch.randn(BUCKET, generator=torch.Generator().manual_seed(5))
    gbuf = _emulated_weight_buffers("optim_grads_params", 4, lambda rank: rank, forced)[1]
    origs = [torch.empty(size) for _, size in _ITEMS]
    gbuf.param_idx = {orig: i for i, orig in enumerate(origs)}
    group = SimpleNamespace(
        requires_grad=True, main_grad_buffer=gbuf, sharding_strategy="optim_grads_params"
    )
    reference = {f"p{i}": forced[start : start + size] for i, (start, size) in enumerate(_ITEMS)}

    def fsdp_with(make_param):
        named = [
            (f"p{i}", make_param(orig, sgu.buffer_item_shard(gbuf, i)[1]))
            for i, orig in enumerate(origs)
        ]
        return SimpleNamespace(
            param_and_grad_buffer=SimpleNamespace(
                optimizer_named_parameters=named,
                parameter_groups=[group],
                param_to_param_group={orig: 0 for orig in origs},
            )
        )

    plain = fsdp_with(lambda orig, values: SimpleNamespace(orig_param=orig, grad=values))
    assert sgu.fsdp_optimizer_grad_mismatches(plain, reference) == {}
    decoupled = fsdp_with(
        lambda orig, values: SimpleNamespace(orig_param=orig, grad=None, decoupled_grad=values)
    )
    assert sgu.fsdp_optimizer_grad_mismatches(decoupled, reference) == {}
    unset = fsdp_with(
        lambda orig, values: SimpleNamespace(orig_param=orig, grad=values, decoupled_grad=None)
    )
    # Rank 1's shard (bucket 64..128) meets items 1 and 2: those have no consumed gradient.
    assert set(sgu.fsdp_optimizer_grad_mismatches(unset, reference)) == {"p1", "p2"}


# ---------------------------------------------------------------------------------------------
# The GPU tests' Megatron-FSDP reduction schedule expectation (strategy_grad_utils)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy,dp_shard,every",
    [
        ("optim_grads_params", 2, True),
        ("optim_grads", 2, True),
        ("optim_grads_params", 1, False),
        ("optim_grads", 1, False),
        ("optim", 2, False),  # ZeRO-1: unsharded buffer, reduced once per step
        ("no_shard", 2, False),
    ],
)
def test_reduced_every_microbatch(strategy, dp_shard, every):
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    assert sgu.reduced_every_microbatch(strategy, dp_shard) is every


_NAMES = [
    "decoder.layers.7.inner_layer.mlp.experts.linear_fc1.weight0",
    "decoder.layers.7.inner_layer.mlp.shared_experts.linear_fc1.weight",
    "decoder.layers.7.inner_layer.mlp.router.weight",
    "mtp.layers.0.mlp.experts.linear_fc2.weight1",
]


@pytest.mark.parametrize(
    "ep,expert,once",
    [
        (2, None, {0, 3}),  # W4 EP2: one-rank expert DP-Shard (the recipes' layout)
        (1, None, set()),  # W4 EP1: experts sharded over two ranks
        (1, "optim", {0, 3}),  # ZeRO-1 experts over two ranks (hsdp_expert_optim at EP1)
        (2, "optim_grads", {0, 3}),
    ],
)
def test_expected_once_per_step(ep, expert, once):
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    got = sgu.expected_once_per_step(
        _NAMES, dense="optim_grads_params", expert=expert, world=4, ep=ep, instances=2
    )
    assert got == {_NAMES[i] for i in once}
    assert sgu.expected_dp_shard(4, ep, 2, True) == 4 // (2 * ep)
    assert sgu.expected_dp_shard(4, ep, 2, False) == 2


def _record(calls):
    return dict(calls=calls, names={0: ["a"], 1: ["b", "c"], 2: ["d"]})


def test_observed_once_per_step():
    """A bucket counts as reduced once per step when, in every step, only the last
    microbatch's hook reduced it (over DP-Outer too) and no per-use reduction ran."""
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    calls = []
    for step in range(2):
        for microbatch in range(3):
            last = microbatch == 2
            calls.append((step, microbatch, (0,), last, False))  # every microbatch
            if last:
                calls.append((step, microbatch, (1,), True, False))  # once per step
                calls.append((step, microbatch, (2,), step == 0, False))  # outer lost in step 2
    assert sgu.observed_once_per_step(_record(calls), 2) == {"b", "c"}
    assert sgu.observed_once_per_step(_record(calls), 1) == {"b", "c", "d"}
    per_use = calls + [(0, 1, (1,), False, True)]
    assert sgu.observed_once_per_step(_record(per_use), 2) == set()
    hooks, uses, outer = sgu.reduction_counts(per_use, 0)
    assert hooks[(2, 1)] == 1 and uses[(1, 1)] == 1 and outer[(2, 0)] == 1 and not outer[(0, 0)]


def test_assert_per_backward_mutation():
    """Every fused weight reduced per backward must fail; the once-per-step ones must equal the
    arm's exactly (the mutation cannot reach them), and some must be per backward."""
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    fused = ["dense", "expert"]
    arm = {"dense": torch.ones(2), "expert": torch.ones(2)}
    mutant = {"dense": torch.zeros(2), "expert": torch.ones(2)}
    failing = {"dense": dict(ok=False), "expert": dict(ok=True)}
    assert sgu.assert_per_backward_mutation(failing, mutant, arm, fused, {"expert"}) == ["expert"]
    entry = dict(ok=True, cosine=1.0, relative=0.0, norm=1.0)
    passing = {"dense": entry, "expert": entry}
    with pytest.raises(AssertionError):  # a per-backward weight escaped
        sgu.assert_per_backward_mutation(passing, mutant, arm, fused, {"expert"})
    reached = {"dense": torch.zeros(2), "expert": torch.zeros(2)}
    with pytest.raises(AssertionError):  # the mutation changed a once-per-step weight
        sgu.assert_per_backward_mutation(failing, reached, arm, fused, {"expert"})
    with pytest.raises(AssertionError):  # nothing per backward left to check
        sgu.assert_per_backward_mutation(failing, mutant, arm, fused, {"dense", "expert"})


_BUCKET_NAMES = {
    0: ["decoder.layers.7.self_attention.linear_qkv.weight"],
    1: ["decoder.layers.7.mlp.experts.linear_fc1.weight0"],
}


def _reduction_record(
    expert_every, *, dense_every=True, outer_every=False, sizes=(2, 2), per_use=False
):
    """A two-step record of a dense bucket 0 and an expert bucket 1, each reduced by the hooks
    every microbatch (``*_every``) or only in the last one, over DP-Outer in the last microbatch
    (or every one), optionally with a per-use reduction of bucket 0."""
    calls = []
    for step in range(2):
        for microbatch in range(3):
            last = microbatch == 2
            for bucket_id, every in ((0, dense_every), (1, expert_every)):
                if every or last:
                    calls.append((step, microbatch, (bucket_id,), last or outer_every, False))
        if per_use:
            calls.append((step, 0, (0,), False, True))
    return dict(
        calls=calls,
        buckets=dict(enumerate(sizes)),
        names=_BUCKET_NAMES,
        expert_params={0: False, 1: True},
    )


@pytest.mark.parametrize(
    "ep,expert,expert_every,sizes,kinds",
    [
        # W4, two instances, EP1: experts over a two-rank DP-Shard group.
        (1, None, True, (2, 2), set()),
        (1, "optim", False, (2, 2), set()),  # ZeRO-1 experts: once per step
        (1, "optim", True, (2, 2), {("DP-Shard", 1)}),  # ZeRO-1 reduced every microbatch
        (1, None, False, (2, 2), {("DP-Shard", 1)}),  # ZeRO-3 over two ranks reduced once
        (1, "optim_grads", False, (2, 2), {("DP-Shard", 1)}),  # ZeRO-2 likewise
        # EP2: a one-rank expert DP-Shard group, reduced once per step whatever the strategy.
        (2, None, False, (2, 1), set()),
        (2, "optim_grads", True, (2, 1), {("DP-Shard", 1)}),
        (2, None, False, (2, 2), {("DP-Shard size", 1)}),  # not the layout's group
    ],
)
def test_reduction_problems_check_the_dp_shard_schedule(ep, expert, expert_every, sizes, kinds):
    """The checker of the HSDP GPU test's reduction records (``reduction_problems``) has teeth
    on its DP-Shard branch: it rejects a ZeRO-1 bucket reduced every microbatch, a ZeRO-2/3
    bucket over two ranks reduced once per step, and a DP-Shard size off the layout, in every
    step, and accepts the schedule the strategy and layout imply."""
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    record = _reduction_record(expert_every, sizes=sizes)
    problems = sgu.reduction_problems(
        record, dense="optim_grads_params", expert=expert, world=4, ep=ep, instances=2, steps=2
    )
    assert {(problem[0], problem[2]) for problem in problems} == kinds, problems
    for kind, _ in kinds:
        assert {problem[1] for problem in problems if problem[0] == kind} == {0, 1}, problems


def test_reduction_problems_check_the_dp_outer_and_per_use_schedule():
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    def kinds(record):
        problems = sgu.reduction_problems(
            record, dense="optim_grads_params", expert=None, world=4, ep=1, instances=2, steps=2
        )
        return {(problem[0], problem[2]) for problem in problems}

    assert kinds(_reduction_record(True, outer_every=True)) == {("DP-Outer", 0), ("DP-Outer", 1)}
    never = _reduction_record(True)
    never["calls"] = [call[:3] + (False,) + call[4:] for call in never["calls"]]
    assert kinds(never) == {("DP-Outer", 0), ("DP-Outer", 1)}
    assert kinds(_reduction_record(True, per_use=True)) == {("per use", 0)}
    # The dense bucket reduced once per step though its strategy shards over two ranks.
    assert kinds(_reduction_record(True, dense_every=False)) == {("DP-Shard", 0)}


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


# ---------------------------------------------------------------------------------------------
# The GTP GPU test's precise embedding reference (strategy_grad_utils.install_per_use_fp32_embedding)
# ---------------------------------------------------------------------------------------------


class _Group:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class _Embedding(torch.nn.Module):
    """A stand-in ``VocabParallelEmbedding``: its layout attributes and the deterministic TP=1
    branch of its forward (``weight[ids]``)."""

    def __init__(self, weight, *, tp=1, gtp=1, deterministic=True, reduce_scatter=False):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())
        self.tp_group = _Group(tp)
        self.gtp_remat_size = gtp
        self.deterministic_mode = deterministic
        self.reduce_scatter_embeddings = reduce_scatter

    def forward(self, input_):
        return self.weight[input_]


def _ddp_like(weight):
    """``main_grad`` and DDP's post-accumulate hook arithmetic (distributed_data_parallel.py):
    ``.grad`` is added unless ``grad_added_to_main_grad``, then dropped. Returns the list of
    ``(gradient, added)`` the hook saw."""
    weight.main_grad = torch.zeros(weight.shape, dtype=torch.float32)
    weight.grad_added_to_main_grad = False
    seen = []

    def hook(param):
        added = param.grad is not None and not param.grad_added_to_main_grad
        seen.append((param.grad, added))
        if added:
            param.main_grad.add_(param.grad.float())
        param.grad = None

    weight.register_post_accumulate_grad_hook(hook)
    return seen


def _two_use_main_grad(weight, ids, dys, *, precise):
    """Two lookups (the main and the MTP input) through a stand-in model, then backward; returns
    the FP32 ``main_grad``, the hook's gradients and the lookup counter (None if not precise)."""
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    embedding = _Embedding(weight)
    model = SimpleNamespace(embedding=SimpleNamespace(word_embeddings=embedding))
    counter = sgu.install_per_use_fp32_embedding(model) if precise else None
    seen = _ddp_like(embedding.weight)
    loss = sum((embedding(i).float() * dy).sum() for i, dy in zip(ids, dys))
    loss.backward()
    return embedding.weight.main_grad.clone(), seen, counter


def _per_use_fp32_sum(weight, ids, dys):
    """The expectation, computed without the helper: each use's own autograd gradient (BF16, the
    same index backward), widened, summed in FP32."""
    leaf = weight.clone().requires_grad_(True)
    total = torch.zeros(weight.shape, dtype=torch.float32)
    for i, dy in zip(ids, dys):
        (grad,) = torch.autograd.grad((leaf[i].float() * dy).sum(), leaf)
        assert grad.dtype == torch.bfloat16
        total += grad.float()
    return total


def _relative(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def test_precise_embedding_reference_sums_each_use_in_fp32():
    """Random rows, the MTP lookup on the shifted ids (the same rows, other positions): the
    precise reference's ``main_grad`` is bitwise the FP32 sum of the per-use BF16 gradients, and
    DDP's hook adds nothing. The plain lookup (autograd sums the uses in BF16 before the hook's
    FP32 add) is more than the 1e-3 gate off: the reason the reference changed."""
    generator = torch.Generator().manual_seed(0)
    vocab, hidden, seq = 512, 512, 256
    weight = torch.randn(vocab, hidden, generator=generator).to(torch.bfloat16)
    tokens = torch.randint(0, vocab, (1, seq), generator=generator)
    ids = [tokens, torch.roll(tokens, -1, dims=1)]
    dys = [torch.randn(1, seq, hidden, generator=generator) * s for s in (1.0, 0.1)]
    expected = _per_use_fp32_sum(weight, ids, dys)

    got, seen, counter = _two_use_main_grad(weight, ids, dys, precise=True)
    assert torch.equal(got, expected)
    assert counter.lookups == 2 and counter.checked
    # One hook call (both uses accumulate into one .grad): a zero tensor, flagged, not added.
    assert len(seen) == 1, seen
    grad, added = seen[0]
    assert not added and not torch.count_nonzero(grad)

    plain, _, _ = _two_use_main_grad(weight, ids, dys, precise=False)
    assert _relative(plain, expected) > 1e-3, _relative(plain, expected)


def test_precise_embedding_reference_keeps_a_bit_bf16_drops():
    """Constructed: one row used twice with gradients 1.0 and 2**-9. Per use in FP32 the sum is
    1.001953125; summed in BF16 first (7 fractional bits) it is 1.0."""
    weight = torch.zeros(4, 1, dtype=torch.bfloat16)
    ids = [torch.tensor([[1]]), torch.tensor([[1]])]
    dys = [torch.tensor([[[1.0]]]), torch.tensor([[[2.0**-9]]])]
    got, _, _ = _two_use_main_grad(weight, ids, dys, precise=True)
    plain, _, _ = _two_use_main_grad(weight, ids, dys, precise=False)
    assert got[1].item() == 1.0 + 2.0**-9
    assert plain[1].item() == 1.0
    assert torch.count_nonzero(got[[0, 2, 3]]) == 0


@pytest.mark.parametrize(
    "layout",
    [dict(tp=2), dict(gtp=2), dict(deterministic=False), dict(reduce_scatter=True)],
    ids=["tp2", "gtp2", "nondeterministic", "reduce_scatter"],
)
def test_precise_embedding_reference_refuses_other_layouts(layout):
    """The installer replaces the deterministic TP=1, non-GTP lookup only: any other layout is
    refused at install, and a layout changed after the install is refused at the call."""
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    weight = torch.zeros(4, 2, dtype=torch.bfloat16)
    model = SimpleNamespace(embedding=SimpleNamespace(word_embeddings=_Embedding(weight, **layout)))
    with pytest.raises(AssertionError, match="deterministic TP=1 lookup only"):
        sgu.install_per_use_fp32_embedding(model)
    embedding = _Embedding(weight)
    sgu.install_per_use_fp32_embedding(
        SimpleNamespace(embedding=SimpleNamespace(word_embeddings=embedding))
    )
    attribute = dict(
        tp="tp_group", gtp="gtp_remat_size", deterministic="deterministic_mode"
    ).get(next(iter(layout)), "reduce_scatter_embeddings")
    value = _Group(2) if attribute == "tp_group" else next(iter(layout.values()))
    setattr(embedding, attribute, value)
    with pytest.raises(AssertionError, match="deterministic TP=1 lookup only"):
        embedding(torch.tensor([[0]]))


def test_precise_embedding_reference_checks_the_original_forward():
    """The first call compares the replacement with the original forward exactly; a tied weight
    is refused (the output layer's ``.grad`` would be skipped by the flag)."""
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    class _Shifted(_Embedding):
        def forward(self, input_):
            return self.weight[input_] + 1.0

    weight = torch.zeros(4, 2, dtype=torch.bfloat16)
    model = SimpleNamespace(embedding=SimpleNamespace(word_embeddings=_Shifted(weight)))
    sgu.install_per_use_fp32_embedding(model)
    with pytest.raises(AssertionError, match="drifted from the original"):
        model.embedding.word_embeddings(torch.tensor([[0]]))
    tied = SimpleNamespace(
        embedding=SimpleNamespace(word_embeddings=_Embedding(weight)),
        share_embeddings_and_output_weights=True,
    )
    with pytest.raises(AssertionError, match="tied embedding"):
        sgu.install_per_use_fp32_embedding(tied)
