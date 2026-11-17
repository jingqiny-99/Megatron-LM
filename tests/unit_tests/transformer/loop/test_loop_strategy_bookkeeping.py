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
  every item once over the ranks; the DDP writer and the bitwise comparison.
* The GPU tests' Megatron-FSDP reduction schedule expectation (``reduced_every_microbatch``,
  ``expected_once_per_step``, ``observed_once_per_step``) and the per-backward mutation check
  (``assert_per_backward_mutation``).
* The GPU tests' optimizer (``strategy_grad_utils.SGD_AS_ADAM``): Adam that steps as SGD without
  momentum, since the distributed optimizer refuses SGD.

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


def test_mismatches_are_bitwise_and_name_dtype_and_size():
    from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu

    want = {"x": torch.ones(4), "y": torch.zeros(2, dtype=torch.bfloat16), "z": torch.ones(3)}
    have = {"x": torch.ones(4), "y": torch.zeros(2, dtype=torch.bfloat16), "z": torch.ones(2)}
    have["y"][1] = 2**-10
    found = sgu.mismatches(have, want)
    assert set(found) == {"y", "z"}
    assert found["y"] == (1, "bfloat16", 2**-10)
    assert found["z"][0] == "shape"
    assert sgu.mismatches(want, want) == {}
    assert sgu.mismatches({"x": -torch.zeros(1)}, {"x": torch.zeros(1)}) == {}  # -0 == 0


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
    arm's bitwise (the mutation cannot reach them), and some must be per backward."""
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
