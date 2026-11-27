# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests of the multi-EU RSBF sample dispatcher (``loop/dispatcher.py``).

The transfer cases run in spawned Gloo worlds of 2, 4 and 8 CPU processes (local gates; they
are skipped under torchrun, where the GPU multi-EU suite exercises the NCCL path):

* X1: a hand-built move phase with sample, chain and item transfers (bf16 / fp32 / int32 /
  int64 / bool slots, a zero-size criterion state, absent optional slots), EUs without
  transfers, both transports, over the whole world and over sub-group domains: every received
  tensor is bitwise the sent one.
* X2: ``exchange_status``: the MAX bitmap on every rank, one Gloo call per invocation; the plan
  digest check passes on equal digests and raises on every rank otherwise.
* X3: every move phase of random multi-EU plans (``WavePlanner``) through both transports:
  payloads exact, and for every pair of EUs the send sequence of one side equals the receive
  sequence of the other (the fixed collective order).

The one-EU behaviour needs no process group and runs everywhere.
"""

import functools
import random
import zlib

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.loop.dispatcher import LoopSampleDispatcher, RSBFGroups
from megatron.core.transformer.loop.rsbf_plan import (
    CRIT_STATE,
    EXIT_FLAGS,
    GRAD_CRIT,
    GRAD_INJECTION,
    GRAD_STATE,
    INJECTION,
    LOSS_RECORD,
    ROUND_INPUT,
    MovePhase,
    Transfer,
    WavePlanner,
    known_depth_plan,
)
from tests.unit_tests.transformer.loop.multi_eu_utils import (
    SPAWN_SKIP_REASON,
    run_spawned,
    spawn_tests_enabled,
)

S, NH = 5, 6

# Slot -> (shape, dtype): the stage-3 port table at c = 1 (k = 0: a zero-size criterion state).
SCHEMA = {
    ROUND_INPUT: ((S, 1, NH), torch.bfloat16),
    CRIT_STATE: ((1, 0), torch.float32),
    INJECTION: ((S, 1, NH), torch.bfloat16),
    EXIT_FLAGS: ((2,), torch.int32),
    "tokens": ((1, S), torch.int64),
    "position_ids": ((1, S), torch.int64),
    "labels": ((1, S), torch.int64),
    "loss_mask": ((1, S), torch.float32),
    "padding_mask": ((1, S), torch.bool),
    "attention_mask": ((1, 1, S, S), torch.bool),
    GRAD_STATE: ((S, 1, NH), torch.bfloat16),
    GRAD_INJECTION: ((S, 1, NH), torch.bfloat16),
    GRAD_CRIT: ((1, 3), torch.float32),
    LOSS_RECORD: ((4,), torch.int32),
    "expected_state": ((S, 1, NH), torch.bfloat16),
}
ALL_BATCH = ("tokens", "position_ids", "labels", "loss_mask", "padding_mask", "attention_mask")


def _payload(t: Transfer, slot: str) -> torch.Tensor:
    """The deterministic payload of ``slot`` of ``t`` (every rank can rebuild it)."""
    shape, dtype = SCHEMA[slot]
    seed = zlib.crc32(repr((t.order_key(), slot)).encode())
    gen = torch.Generator().manual_seed(seed)
    if dtype == torch.bool:
        return torch.rand(shape, generator=gen) > 0.5
    if dtype in (torch.int32, torch.int64):
        return torch.randint(-(2**30), 2**30, shape, generator=gen, dtype=dtype)
    value = torch.randn(shape, generator=gen).to(dtype)
    if value.numel():
        value.view(-1)[0] = float("nan")  # bit patterns, not values, must survive
    return value


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().reshape(-1).view(torch.uint8)


def _receive_buffer(t: Transfer, slot: str) -> torch.Tensor:
    shape, dtype = SCHEMA[slot]
    return torch.zeros(shape, dtype=dtype)


def _run_phase(dispatcher, phase):
    """Exchange ``phase`` and check every received payload bitwise; returns the count."""
    eu = dispatcher.eu
    received = dispatcher.exchange(
        phase,
        send=lambda t: [_payload(t, slot) for slot in t.slots],
        recv=lambda t: [_receive_buffer(t, slot) for slot in t.slots],
    )
    expected = [t for t in phase.transfers if t.dst == eu]
    assert sorted(received, key=Transfer.order_key) == sorted(expected, key=Transfer.order_key)
    for t, tensors in received.items():
        assert len(tensors) == len(t.slots)
        for slot, tensor in zip(t.slots, tensors):
            want = _payload(t, slot)
            assert tensor.shape == want.shape and tensor.dtype == want.dtype, (t, slot)
            assert torch.equal(_bits(tensor), _bits(want)), (t, slot)
    return len(received)


def _x1_phase(num_eus: int) -> MovePhase:
    """Sample, chain and item transfers; with 4+ EUs, EU 2 has nothing to do."""
    transfers = [
        Transfer(
            kind="sample",
            sample=0,
            src=0,
            dst=1,
            round=2,
            slots=(ROUND_INPUT, CRIT_STATE, INJECTION, EXIT_FLAGS) + ALL_BATCH,
            frees=(("state", 1), ("crit", 1)),
        ),
        Transfer(
            kind="sample",
            sample=3,
            src=0,
            dst=1,
            round=2,
            slots=(ROUND_INPUT, CRIT_STATE, EXIT_FLAGS),
            frees=(("state", 1), ("crit", 1)),
        ),
        Transfer(
            kind="chain",
            sample=1,
            src=1,
            dst=0,
            round=3,
            slots=(GRAD_STATE, GRAD_INJECTION, GRAD_CRIT, LOSS_RECORD),
            item="core",
        ),
        Transfer(
            kind="item",
            sample=2,
            src=num_eus - 1,
            dst=0,
            round=1,
            slots=(CRIT_STATE, INJECTION, EXIT_FLAGS, "tokens", "labels", "loss_mask"),
            frees=(("crit", 0),),
            item="core",
        ),
    ]
    if num_eus >= 4:
        transfers.append(
            Transfer(
                kind="chain",
                sample=5,
                src=3,
                dst=1,
                round=2,
                slots=(GRAD_STATE, LOSS_RECORD, "expected_state"),
                item="coda",
            )
        )
    return MovePhase(tuple(sorted(transfers, key=Transfer.order_key)))


def _random_phases(num_eus: int, seed: int) -> list:
    """The move phases of a random known-depth plan (every rank builds the same)."""
    rng = random.Random(seed)
    per_eu = rng.randint(2, 5)
    home = [g % num_eus for g in range(num_eus * per_eu)]
    depths = [rng.randint(1, 4) for _ in home]
    planner = WavePlanner(
        num_eus=num_eus,
        home=home,
        max_rounds=4,
        fill=rng.choice(["drain", "sample_order"]),
        rebalance=True,
        tail_item_moves=True,
        criterion_differentiable=rng.random() < 0.5,
        check_recompute=rng.random() < 0.5,
        batch_slots=ALL_BATCH[:4] + (("padding_mask",) if rng.random() < 0.5 else ()),
        report_keys=1,
    )
    return [s for s in known_depth_plan(planner, depths) if isinstance(s, MovePhase)]


def _worker(rank: int, world: int) -> dict:
    """Every case on one rank of a spawned Gloo world of ``world`` ranks."""
    out = {}
    status = dist.new_group(list(range(world)), backend="gloo")
    world_groups = RSBFGroups(domain=dist.group.WORLD, status=status)
    # Sub-group domains (DP groups of a 2-way "model parallel" layout): ranks {0, 2, ..} and
    # {1, 3, ..}; every rank creates every group, in the same order.
    halves = []
    if world >= 4:
        for start in (0, 1):
            ranks = list(range(start, world, 2))
            halves.append((ranks, dist.new_group(ranks), dist.new_group(ranks, backend="gloo")))

    # X1: hand-built phases, both transports, world and sub-group domains.
    for transport in ("p2p", "all_to_all"):
        dispatcher = LoopSampleDispatcher(world_groups, transport=transport, device="cpu")
        dispatcher.warm_up()
        received = _run_phase(dispatcher, _x1_phase(world))
        out[f"x1_{transport}_first_ops"] = dispatcher.stats.p2p_ops
        # A phase this rank takes no part in (and one only the last two EUs share).
        received += _run_phase(
            dispatcher,
            MovePhase(
                (
                    Transfer(
                        kind="sample",
                        sample=9,
                        src=world - 1,
                        dst=world - 2,
                        round=2,
                        slots=(ROUND_INPUT, CRIT_STATE, EXIT_FLAGS),
                    ),
                )
            ),
        )
        out[f"x1_{transport}"] = (received, dispatcher.stats)
        for ranks, domain, domain_status in halves:
            if rank not in ranks:
                continue
            sub = LoopSampleDispatcher(
                RSBFGroups(domain=domain, status=domain_status), transport=transport, device="cpu"
            )
            assert (sub.eu, sub.num_eus) == (ranks.index(rank), len(ranks))
            sub.warm_up()
            out[f"x1_sub_{transport}"] = _run_phase(sub, _x1_phase(len(ranks)))

    # X2: status exchange and digests.
    dispatcher = LoopSampleDispatcher(world_groups, device="cpu")
    size = 3 * world
    bitmap = torch.zeros(size + 1, dtype=torch.uint8)
    for g in range(size):
        if g % world == rank:
            bitmap[g] = 1
    bitmap[size] = 1 if rank == 1 else 0
    reduced = dispatcher.exchange_status(bitmap)
    out["x2_bitmap"] = reduced.tolist()
    out["x2_calls"] = dispatcher.stats.status_exchanges
    second = dispatcher.exchange_status(torch.zeros(size + 1, dtype=torch.uint8))
    out["x2_second"] = second.tolist()
    out["x2_calls_after"] = dispatcher.stats.status_exchanges
    dispatcher.check_identical(bytes(range(32)))
    try:
        dispatcher.check_identical(bytes([rank == world - 1] * 32))
        out["x2_mismatch"] = None
    except RuntimeError as error:
        out["x2_mismatch"] = str(error)
    try:
        LoopSampleDispatcher(RSBFGroups(domain=dist.group.WORLD, status=None), device="cpu")
        LoopSampleDispatcher(
            RSBFGroups(domain=dist.group.WORLD, status=None), device="cpu"
        ).exchange_status(torch.zeros(2, dtype=torch.uint8))
        out["x2_no_status"] = None
    except ValueError as error:
        out["x2_no_status"] = str(error)
    if halves:
        try:
            LoopSampleDispatcher(
                RSBFGroups(domain=dist.group.WORLD, status=halves[rank % 2][2]), device="cpu"
            )
            out["x2_stale"] = None
        except ValueError as error:
            out["x2_stale"] = str(error)

    # X3: random plans, both transports, op logs.
    logs = {}
    for transport in ("p2p", "all_to_all"):
        dispatcher = LoopSampleDispatcher(
            world_groups, transport=transport, device="cpu", record_ops=True
        )
        dispatcher.warm_up()
        phases = 0
        for seed in range(4):
            for phase in _random_phases(world, seed):
                _run_phase(dispatcher, phase)
                phases += 1
        logs[transport] = (dispatcher.op_log, phases)
    out["x3"] = logs
    return out


@functools.lru_cache(maxsize=None)
def _results(world: int) -> list:
    return run_spawned(_worker, world, timeout=600.0)


spawned = pytest.mark.skipif(not spawn_tests_enabled(), reason=SPAWN_SKIP_REASON)
WORLDS = [2, 4, 8]


@spawned
@pytest.mark.parametrize("world", WORLDS)
def test_x1_transfers_are_bitwise_over_both_transports(world):
    results = _results(world)
    phase = _x1_phase(world)
    for transport in ("p2p", "all_to_all"):
        for rank, out in enumerate(results):
            received, stats = out[f"x1_{transport}"]
            incoming = sum(t.dst == rank for t in phase.transfers) + (rank == world - 2)
            assert received == incoming
            assert stats.phases == 2
            if transport == "p2p":
                assert stats.all_to_alls == 0
                # Ranks without transfers post nothing (EU 2 at W >= 4 in the first phase).
                involved = {e for t in phase.transfers for e in (t.src, t.dst)}
                assert (out[f"x1_{transport}_first_ops"] > 0) == (rank in involved)
            else:
                assert stats.all_to_alls == 2 and stats.p2p_ops == 0
        if world >= 4:
            for rank, out in enumerate(results):
                half = list(range(rank % 2, world, 2))
                sub_phase = _x1_phase(len(half))
                assert out[f"x1_sub_{transport}"] == sum(
                    t.dst == half.index(rank) for t in sub_phase.transfers
                )


@spawned
@pytest.mark.parametrize("world", WORLDS)
def test_x2_status_exchange_and_digests(world):
    results = _results(world)
    size = 3 * world
    expected = [1] * size + [1]
    for rank, out in enumerate(results):
        assert out["x2_bitmap"] == expected
        assert out["x2_second"] == [0] * (size + 1)
        assert out["x2_calls"] == 1 and out["x2_calls_after"] == 2
        assert out["x2_mismatch"] is not None and f"EUs [{world - 1}]" in out["x2_mismatch"]
        assert out["x2_no_status"] is not None and "status group" in out["x2_no_status"]
        if world >= 4:
            assert out["x2_stale"] is not None and "stale" in out["x2_stale"]


@spawned
@pytest.mark.parametrize("world", WORLDS)
def test_x3_random_plans_keep_a_fixed_pairwise_order(world):
    results = _results(world)
    for transport in ("p2p", "all_to_all"):
        phases = {out["x3"][transport][1] for out in results}
        assert len(phases) == 1 and phases.pop() > 0
        logs = [out["x3"][transport][0] for out in results]
        total = 0
        for a in range(world):
            for b in range(world):
                if a == b:
                    continue
                sent = [
                    (p, key, slot) for p, key, slot, peer, d in logs[a] if peer == b and d == "send"
                ]
                got = [
                    (p, key, slot) for p, key, slot, peer, d in logs[b] if peer == a and d == "recv"
                ]
                assert sent == got, (transport, a, b)
                total += len(sent)
        assert total > 0


def test_one_eu_dispatcher_needs_no_group():
    dispatcher = LoopSampleDispatcher(RSBFGroups(domain=None), device="cpu")
    assert (dispatcher.eu, dispatcher.num_eus) == (0, 1)
    dispatcher.warm_up()
    assert dispatcher.exchange(MovePhase(()), send=None, recv=None) == {}
    bitmap = torch.tensor([0, 1, 0], dtype=torch.uint8)
    assert dispatcher.exchange_status(bitmap) is bitmap
    assert dispatcher.stats.status_exchanges == 0
    assert dispatcher.gather_digests(b"abc") == [b"abc"]
    dispatcher.check_identical(b"abc")
    with pytest.raises(ValueError, match="leaves the domain"):
        dispatcher.exchange(
            MovePhase((Transfer(kind="sample", sample=0, src=0, dst=1, round=2, slots=()),)),
            send=lambda t: [],
            recv=lambda t: [],
        )
    with pytest.raises(ValueError, match="1-D CPU uint8"):
        dispatcher.exchange_status(torch.zeros(3, dtype=torch.int32))
    with pytest.raises(ValueError, match="unknown move transport"):
        LoopSampleDispatcher(RSBFGroups(domain=None), transport="nccl", device="cpu")
