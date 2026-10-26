# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU memory of the loop parameter store per residency mode, against the planner's model (M3).

Runs under torchrun with CUDA at WORLD_SIZE = W in {2, 4, 8} (z = W, EP = W): the DSV4-TINY-like
loop model of ``test_loop_param_store_parity.py`` (TE linears, bf16, MTP 1, 8 experts), with a
store per mode set (Resident, Resident + sharded grads, core Layer window, Layer window with move
E, Layer window with moves E + M, re-gathered) and item form (whole round, block of one layer).

Gates (per mode set and item form):

* ``memory_report()`` is exact against the tensors the store holds: the canonical shards and
  their fp32 gradients, the window slots, the fp32 scratch pool, the compute copies and the
  accumulators;
* the persistent buffers ``begin_step`` allocates, measured with the caching allocator
  (record-stream blocks flushed), equal the report: the requested bytes
  (``requested_bytes.all.current``) within 2 MiB for the collectives' bookkeeping; the
  ``memory_allocated`` delta between the report rounded to 512-byte blocks and that plus the
  unsplit remainder of a reused cached block (at most 1 MiB per buffer above 1 MiB; it
  depends on each rank's allocator history: Resident copies and accumulators exceeded a flat
  2 MiB slack on some ranks). Every rank fails together;
* the planner's runtime terms (``memory_components(..., store_terms=store.memory_terms())`` on
  ``ModelShape.from_model`` of the built model) carry exactly the store's bytes: slots + scratch
  (``MEM_DENSE_BUF``), shards (``MEM_DENSE_SH``) and copies + accumulators (the stored part of
  ``MEM_DENSE_RES``); "memory per mode within the planner's model" for the store's buffers;
* report-only (printed per rank 0): the step's peak allocation above the pre-step baseline,
  next to the planner's runtime and cost-model (``store_terms=None``) dense terms, so the plan's
  tables can be corrected per component (design 10.4).
"""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.loop import planner as P
from megatron.core.transformer.loop import planner_costs as C
from megatron.core.transformer.loop.param_store import ResidencyMode, StoreModes
from megatron.core.transformer.loop.program import LoopProgram, PrescribedDepthCriterion
from megatron.core.transformer.loop.rsbf_executor import LoopMicrobatch, LoopRSBFExecutor
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.loop import param_store_utils as psu
from tests.unit_tests.transformer.loop.hybrid_loop_utils import VOCAB
from tests.unit_tests.transformer.loop.test_loop_param_store_parity import (
    FORMS,
    PER_RANK,
    _attach_main_grads,
    _dsv4_tiny,
    _global_batch,
    _hadamard_fallback,
    _install,
    _loss_func,
    _skew_depths,
)

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
EXPERTS = 8
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or WORLD < 2 or EXPERTS % WORLD,
    reason="the store memory tests need CUDA and 2, 4 or 8 ranks (EP = W over 8 experts)",
)

MODE_SETS = {
    "resident": "all=r",
    "rsg": "all=rsg",
    "core_lw": "core=lw",
    "lw_e": "core=lw,edges=lw",
    "lw_em": "core=lw,edges=lw,emb_head=rsg",
    "lw_em_regather": "core=lw,edges=lw,emb_head=rsg,core_regather=on",
}
MEASURED_FORMS = ("whole_round", "block2")
ROUND = 512  # the caching allocator's block rounding
SPLIT_REMAINDER = 1 << 20  # the largest large-pool remainder the allocator does not split
SLACK = 2 << 20


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=WORLD)
    model_parallel_cuda_manual_seed(123)
    try:
        yield
    finally:
        Utils.destroy_model_parallel()


@pytest.fixture(scope="module")
def store_groups():
    return psu.make_store_groups(WORLD, WORLD)


def _flush():
    """Settle the allocator: side-stream work done, record-stream frees processed (a freed
    block recorded on another stream stays allocated until the next allocation)."""
    torch.cuda.synchronize()
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()


def _rounded(tensors):
    return sum(-(-t.numel() * t.element_size() // ROUND) * ROUND for t in tensors)


def _requested_bytes():
    """The bytes the live tensors requested, without the allocator's block rounding."""
    return torch.cuda.memory_stats()["requested_bytes.all.current"]


def _remainders(tensors):
    """The most ``memory_allocated`` may count above ``_rounded``: the caching allocator hands
    out a cached large-pool block (requests above 1 MiB) without splitting it when the remainder
    is at most 1 MiB (``should_split``), and counts the whole block."""
    return sum(SPLIT_REMAINDER for t in tensors if t.numel() * t.element_size() > SPLIT_REMAINDER)


def _assert_on_every_rank(problems, what):
    """Fail on every rank when any rank failed: the allocator's history is rank-dependent, and a
    verdict on one rank only would desynchronize the module's later collectives."""
    failed = torch.tensor([1 if problems else 0], dtype=torch.int32, device="cuda")
    dist.all_reduce(failed)
    assert not problems, (what, dist.get_rank(), problems)
    assert int(failed.item()) == 0, (what, f"{int(failed.item())} rank(s) failed (see their logs)")


def _buffers(store):
    tensors = list(store._slots) + list(store._scratch)  # pylint: disable=protected-access
    for unit in store.units:
        tensors += [t for t in (unit.copy, unit.acc) if t is not None]
    return tensors


def _step(model, store):
    """One RSBF step of this rank's slice; returns the peak allocation above the baseline."""
    rank = dist.get_rank()
    total = PER_RANK * WORLD
    batch = _global_batch(total)
    set_moe_metrics_tracker(MoEMetricsTracker())
    program = LoopProgram(
        model, exit_criterion=PrescribedDepthCriterion(dict(enumerate(_skew_depths(total))))
    )
    executor = LoopRSBFExecutor(
        program, pg_collection=ProcessGroupCollection.use_mpu_process_groups()
    )
    microbatches = [
        LoopMicrobatch(
            tokens=sample["tokens"],
            position_ids=sample["position_ids"],
            labels=sample["labels"],
            loss_mask=sample["loss_mask"],
            attention_mask=None,
            padding_mask=None,
            loss_func=_loss_func,
        )
        for g, sample in enumerate(batch)
        if g % WORLD == rank
    ]
    _flush()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    executor.run_step(
        lambda iterator, _model: next(iterator), iter(microbatches), len(microbatches), model=model
    )
    _flush()
    return torch.cuda.max_memory_allocated() - base


@pytest.mark.parametrize("form", MEASURED_FORMS)
@pytest.mark.parametrize("name", list(MODE_SETS))
def test_store_memory_per_mode_against_the_planner(store_groups, name, form):
    with _hadamard_fallback():
        model = _dsv4_tiny(**FORMS[form])
        store = _install(model, store_groups, MODE_SETS[name])
        _attach_main_grads(model)
    modes = StoreModes.parse(MODE_SETS[name])
    report = store.memory_report()
    elem = 2  # bf16 store

    # The report against the tensors the store holds.
    shards = sum(
        t.shard.numel() * t.shard.element_size() + t.shard.main_grad.numel() * 4
        for t in store.tensors()
    )
    problems = []
    if shards != sum(report.shard_bytes.values()):
        problems.append(("shards", shards, sum(report.shard_bytes.values())))
    _flush()
    base, base_requested = torch.cuda.memory_allocated(), _requested_bytes()
    store.begin_step()
    _flush()
    persistent = torch.cuda.memory_allocated() - base
    requested = _requested_bytes() - base_requested
    store.end_step()
    buffers = _buffers(store)
    exact = sum(t.numel() * t.element_size() for t in buffers)
    expected = (
        report.slot_bytes
        + report.scratch_bytes
        + sum(report.resident_copy_bytes.values())
        + sum(report.accumulator_bytes.values())
    )
    if exact != expected:
        problems.append(("report != held tensors", exact, expected))
    # What begin_step allocated is exactly the report (requested bytes: no block rounding).
    if abs(requested - exact) > SLACK:
        problems.append(("requested", requested, exact))
    # The allocated delta also counts unsplit remainders of reused cached blocks: at most
    # SPLIT_REMAINDER per buffer (rank-dependent, it depends on the allocator's history).
    if not _rounded(buffers) - SLACK <= persistent <= _rounded(buffers) + _remainders(buffers):
        problems.append(("allocated", persistent, _rounded(buffers), _remainders(buffers)))
    _assert_on_every_rank(problems, (name, form))

    # The planner's runtime terms carry the store's bytes exactly.
    shape = C.ModelShape.from_model(model, vocab_size=VOCAB, expert_parallel_size=WORLD)
    terms = store.memory_terms()
    ph = C.SeqPhase.of(shape, 256, 1)
    hw = P.PlannerHardware.preset("h100")
    block = form != "whole_round"
    kwargs = dict(
        EP=WORLD,
        N=WORLD,
        prec="bf16",
        layout="layer_window" if modes.core is ResidencyMode.LAYER_WINDOW else "resident",
        nested=False,
        block_stride=(FORMS[form]["loop_rsbf_block_stride"] // 2) if block else shape.core_layers,
    )
    runtime, _ = C.memory_components(shape, ph, hw, 0.0, store_terms=terms, **kwargs)
    assert runtime[C.MEM_DENSE_BUF] == report.slot_bytes + report.scratch_bytes
    assert runtime[C.MEM_DENSE_SH] == sum(report.shard_bytes.values())
    stored = sum(terms.group_numel.values())
    replicated = (elem + 4) * (sum(shape.params.groups().values()) - stored)
    assert runtime[C.MEM_DENSE_RES] - replicated == sum(report.resident_copy_bytes.values()) + sum(
        report.accumulator_bytes.values()
    )
    assert requested <= runtime[C.MEM_DENSE_BUF] + (runtime[C.MEM_DENSE_RES] - replicated) + SLACK

    # Report-only: the step's peak against the runtime and cost-model terms.
    peak = _step(model, store)
    cost, _ = C.memory_components(
        shape,
        ph,
        hw,
        0.0,
        edges_mode="Layer window" if modes.prelude is ResidencyMode.LAYER_WINDOW else "Resident",
        emb_mode=(
            "Resident + sharded grads"
            if modes.emb_head is ResidencyMode.RESIDENT_SHARDED_GRADS
            else "Resident"
        ),
        z_lw=WORLD,
        **kwargs,
    )
    if dist.get_rank() == 0:
        dense = (C.MEM_DENSE_RES, C.MEM_DENSE_SH, C.MEM_DENSE_BUF)
        print(
            f"[store memory] {name} {form}: persistent {persistent / 2**20:.2f} MiB (requested "
            f"{requested / 2**20:.2f}, report "
            f"{expected / 2**20:.2f}), step peak above baseline {peak / 2**20:.2f} MiB; runtime "
            + ", ".join(f"{k.split(':')[1].strip()} {runtime[k] / 2**20:.2f}" for k in dense)
            + "; cost model "
            + ", ".join(f"{k.split(':')[1].strip()} {cost.get(k, 0.0) / 2**20:.2f}" for k in dense)
            + f" MiB; transients coda {terms.head_scratch_coda + terms.rsg_transient_coda:.0f} B,"
            f" hook .grad {terms.hook_grad_coda:.0f} B, slots {terms.slot_count}, scratch "
            f"{terms.scratch_entries}",
            flush=True,
        )
