# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The loop ring on real kernels and NCCL: ring = PP 1 eager loop (GPU; G2's gradient gate).

Runs under torchrun with CUDA at W = 4 or 8 ranks (2 runs the P = 2 layouts only). The model is
``ring_gpu_utils``' tiny DSv4 loop (DSv4 W / C / H attention with TE linears on the unfused path,
the indexer and its sparse loss, 8 experts + a shared expert with the hash prelude, expert bias
and the sequence aux loss, mHC 4, MTP, R = 4) at **EP 1 in every layout**, in deterministic mode.

The reference is the PP 1 x DP W eager ``LoopHybridStack`` through Megatron's standard schedule.
Every ring layout (P in {2, 4} x k in {1, 2} x recompute plan all 'C' / all 'n') starts from the
reference's initial weights and runs the same global batch (16 sequences of 256 tokens, MBS 1)
through the real ``LoopRingSchedule``: Megatron's interleaved schedule over R x k round views,
``P2PCommunicator`` over NCCL, one DDP per chunk, ``finalize_model_grads`` through
``RingFinalize``. Gates, per layout:

* every parameter's ``main_grad`` after finalize (data-parallel reduced, per-token scaled) against
  the reference's: cosine >= 0.9999 and ``|g - r| / |r| <= 1e-3`` in fp32 (CONTRACT G2's gate;
  only the fp32 summation order across rounds, microbatches and data ranks differs, about 1e-7,
  while one missing round changes a core gradient by O(1 / R)). In bf16 (only if the fp32 probe
  fails): <= 1e-3 for TE fused-wgrad and fp32 parameters, <= 1e-2 for the other bf16 parameters
  (PP 1 sums a parameter's four rounds in bf16 inside one backward, the ring adds each round into
  the fp32 main_grad); the measured spread is printed either way;
* every sequence's loss sum and token count equal the reference's (rel <= 1e-6: the forward ops
  are the same);
* G1: the step's ``LoopRingStepRecord`` reads rounds 4 / 4 and prelude / coda / MTP 1 / 1 / 1, and
  the head's loop forward / backward FLOPs per token equal the reference's;
* the router expert biases after finalize's update equal the reference's bitwise (the same token
  counts, one update per step);
* every reference parameter is held by exactly one pipeline stage, the word embedding by the
  first and (the MTP copy) the last.

Precision: the module first runs the reference in fp32 (TF32 off, ``NVIDIA_TF32_OVERRIDE=0``; the
indexer's bf16-only Hadamard rotation replaced by the same scaled identity in every layout). If
that raises, the grid runs in bf16 with the grouped tolerances above. ``LOOP_RING_GPU_PRECISION``
(``fp32`` / ``bf16``) forces one. The chosen precision and the measured maxima are printed.
"""

import pytest
import torch

from tests.unit_tests.transformer.loop.hybrid_loop_utils import determinism_env
from tests.unit_tests.transformer.loop.ring_gpu_utils import (
    CUDA,
    GLOBAL_BATCH,
    ROUNDS,
    WORLD,
    Layout,
    build_chunks,
    deterministic_algorithms,
    full_precision_matmuls,
    gather,
    grad_stats,
    load_state,
    merged,
    parallel_layout,
    rank_batches,
    ring_layouts,
    run_step,
    run_with_precision,
    state_snapshot,
    tiny_config,
    wrap_ddp,
)

pytestmark = pytest.mark.skipif(
    not CUDA or WORLD < 2 or WORLD % 2, reason="the ring needs CUDA and an even number of ranks"
)

LAYOUTS = ring_layouts() if CUDA and WORLD >= 2 and WORLD % 2 == 0 else []
NAMES = [layout.name for layout in LAYOUTS]
_FLOP_KEYS = ("loop forward flops per token", "loop backward flops per token")


def _reference(dtype):
    """The PP 1 eager reference: its initial state (by global name), main grads after finalize
    (identical on every rank), the per-sequence reports and metrics of all ranks, and the expert
    biases after the update."""
    layout = Layout()
    with parallel_layout(layout):
        config = tiny_config(dtype, layout)
        chunks = build_chunks(config, layout)
        state = state_snapshot(chunks)
        ddp = wrap_ddp(chunks, config)
        result = run_step(ddp, layout, rank_batches())
        del ddp, chunks
    return dict(
        state=state,
        grads=result.grads,
        reports=merged(gather(result.reports)),
        metrics=merged(gather(result.metrics)),
        expert_bias=result.expert_bias,
    )


def _ring(layout, dtype, reference):
    """One ring layout's step from the reference's initial state, reduced to what the tests check
    (gathered over every rank, keyed by ``<global name>@pp<p>``)."""
    # pylint: disable=import-outside-toplevel
    from megatron.core import parallel_state as ps

    with parallel_layout(layout):
        config = tiny_config(dtype, layout)
        chunks = build_chunks(config, layout)
        load_state(chunks, reference["state"])
        ddp = wrap_ddp(chunks, config)
        result = run_step(ddp, layout, rank_batches())
        stage = ps.get_pipeline_model_parallel_rank()
        stats = {
            f"{key}@pp{stage}": dict(
                value, fused=key in result.fused, fp32=result.dtypes[key] == torch.float32
            )
            for key, value in grad_stats(result.grads, reference["grads"]).items()
        }
        bias = {
            f"{name}@pp{stage}": bool(
                torch.equal(value, reference["expert_bias"][name].to(value.device))
            )
            for name, value in result.expert_bias.items()
        }
        record = result.record
        summary = dict(
            stats=merged(gather(stats)),
            bias=merged(gather(bias)),
            reports=merged(gather(result.reports)),
            metrics=merged(gather(result.metrics)),
            records=gather(record),
        )
        del ddp, chunks
    return summary


def _grid(dtype):
    reference = _reference(dtype)
    results = {layout.name: _ring(layout, dtype, reference) for layout in LAYOUTS}
    return reference, results


@pytest.fixture(scope="module")
def grid():
    """The reference and every ring layout, computed once for the module (each layout
    initializes model parallelism once), in fp32 unless the fp32 run raises."""
    with determinism_env(), deterministic_algorithms(), full_precision_matmuls():
        dtype, probe, (reference, results) = run_with_precision(_grid)
    yield dict(dtype=dtype, probe=probe, reference=reference, results=results)


def _tolerance(dtype, stat):
    if dtype == torch.float32 or stat["fused"] or stat["fp32"]:
        return 1e-3
    return 1e-2


def test_precision(grid):
    """The precision the fp32 probe chose (printed; fp32 unless the probe failed)."""
    print(
        f"loop ring gpu parity: precision {grid['dtype']}, world {WORLD}, layouts {NAMES}"
        + (f"; fp32 probe failed: {grid['probe']}" if grid["probe"] else "; fp32 probe passed")
    )
    assert grid["dtype"] in (torch.float32, torch.bfloat16)
    assert len(grid["reference"]["reports"]) == GLOBAL_BATCH


@pytest.mark.parametrize("name", NAMES)
def test_gradients_match_pp1(grid, name):
    """Every parameter's main_grad equals the PP 1 reference's within G2's gate."""
    dtype = grid["dtype"]
    stats = grid["results"][name]["stats"]
    failures = []
    worst_cos, worst_rel, worst_key, bitwise = 1.0, 0.0, None, 0
    for key, stat in sorted(stats.items()):
        worst_cos = min(worst_cos, stat["cosine"])
        if stat["relative"] > worst_rel:
            worst_rel, worst_key = stat["relative"], key
        bitwise += stat["bitwise"]
        cos_gate = 0.9999 if (dtype == torch.float32 or stat["fused"] or stat["fp32"]) else 0.999
        if stat["cosine"] < cos_gate or stat["relative"] > _tolerance(dtype, stat):
            failures.append((key, round(stat["cosine"], 8), stat["relative"], stat["ref_norm"]))
    print(
        f"loop ring gpu parity {name} ({dtype}): {len(stats)} parameter copies, min cosine "
        f"{worst_cos:.12f}, max |g - r| / |r| {worst_rel:.3e} ({worst_key}), bitwise {bitwise}"
    )
    assert not failures, failures[:10]


@pytest.mark.parametrize("name", NAMES)
def test_losses_match_pp1(grid, name):
    """Every sequence's loss sum and token count equal the reference's."""
    reports = grid["results"][name]["reports"]
    reference = grid["reference"]["reports"]
    assert sorted(reports) == sorted(reference) == list(range(GLOBAL_BATCH))
    worst, bitwise = 0.0, 0
    for index, (loss, tokens) in reports.items():
        ref_loss, ref_tokens = reference[index]
        assert tokens == ref_tokens, (index, tokens, ref_tokens)
        worst = max(worst, abs(loss - ref_loss) / abs(ref_loss))
        bitwise += loss == ref_loss
    print(
        f"loop ring gpu losses {name}: max relative {worst:.3e}, bitwise {bitwise}/{len(reports)}"
    )
    assert worst <= 1e-6


@pytest.mark.parametrize("name", NAMES)
def test_rounds_and_flops(grid, name):
    """G1: the step record reads R / R rounds and prelude / coda / MTP once per sample on every
    rank; the head's loop FLOPs per token equal the reference's."""
    result = grid["results"][name]
    for record in result["records"]:
        assert record is not None and not record.forward_only
        assert record.rounds == (ROUNDS, ROUNDS), record
        assert record.prelude == (1, 1) and record.coda == (1, 1) and record.mtp == (1, 1), record
    reference = grid["reference"]["metrics"]
    assert sorted(result["metrics"]) == sorted(reference)
    for index, metrics in result["metrics"].items():
        assert metrics["loop depth"] == ROUNDS
        for key in _FLOP_KEYS:
            want = reference[index][key]
            assert abs(metrics[key] - want) <= 1e-6 * abs(want), (index, key, metrics[key], want)
    record = result["records"][0]
    print(
        f"loop ring gpu G1 {name}: {record.summary()}; model work per microbatch {record.model_work}"
    )


@pytest.mark.parametrize("name", NAMES)
def test_expert_bias_update_matches_pp1(grid, name):
    """Finalize's router expert-bias update leaves the reference's biases, bitwise."""
    bias = grid["results"][name]["bias"]
    assert bias and all(bias.values()), sorted(key for key, same in bias.items() if not same)


@pytest.mark.parametrize("name", NAMES)
def test_every_parameter_on_one_stage(grid, name):
    """The stages together hold every reference parameter once (the word embedding twice: the
    first stage's and the last stage's MTP copy)."""
    holders = {}
    for key in grid["results"][name]["stats"]:
        param, stage = key.rsplit("@pp", 1)
        holders.setdefault(param.split("#", 1)[0], set()).add(int(stage))
    assert set(holders) == set(grid["reference"]["grads"])
    stages = next(layout.stages for layout in LAYOUTS if layout.name == name)
    for param, where in holders.items():
        if param == "embedding.word_embeddings.weight":
            assert where == {0, stages - 1}, (param, where)
        else:
            assert len(where) == 1, (param, where)
