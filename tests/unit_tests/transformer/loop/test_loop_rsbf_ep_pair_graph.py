"""A real differentiable core pair in all-active subrounds of the four-EU RSBF executor."""

import contextlib
from collections import Counter
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.dsa_logging import (
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.loop.rsbf_executor import LoopRSBFExecutor
from tests.unit_tests.transformer.loop import pair_capture, test_loop_rsbf_null_items as native
from tests.unit_tests.transformer.loop.test_loop_rsbf_null_items import (
    _model_parallel,
    deterministic,
)
from tests.unit_tests.transformer.loop.test_loop_rsbf_pair_graph import (
    _capture_core_pair,
    _statistics_buffers,
)


@contextlib.contextmanager
def _all_real_pairs(model, visit, counts):
    """Every EU sees the same planner items. Mixed subrounds preserve native real/null calls."""
    stack = model.decoder
    start, end = stack.loop_roles.core_start, stack.loop_roles.core_end
    assert end - start == 2
    all_real = None
    run_item, run_range = LoopRSBFExecutor._run_item, stack._run_range

    def mark(executor, subround, item):
        nonlocal all_real
        assert all_real is None
        all_real = all(peer is not None for peer in subround.items)
        try:
            return run_item(executor, subround, item)
        finally:
            all_real = None

    def run(hidden, first, last, ctx, label):
        if (first, last) == (start, end) and hidden.requires_grad and all_real is not None:
            counts["full" if all_real else "mixed"] += 1
            if all_real:
                output = visit(hidden, ctx)
                if output is not None:
                    return output
        return run_range(hidden, first, last, ctx, label)

    with patch.object(LoopRSBFExecutor, "_run_item", mark), patch.object(stack, "_run_range", run):
        yield


def _gather(record):
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, record)
    return rows


def _differences(left, right):
    names = set(left) | set(right)
    return sorted(
        k for k in names if k not in left or k not in right or not torch.equal(left[k], right[k])
    )


def _step(model, batches, criterion, pg):
    # Each arm starts its logging interval at zero and snapshots it before the other arm runs.
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    native._reset_expert_bias_counts(model)
    losses, grads, executor = native._run(model, batches, criterion, pg)
    stats = {k: v.clone() for k, v in _statistics_buffers(model).items()}
    return losses, grads, executor, stats


@pytest.mark.skipif(not native.EP_GPU or native.WORLD != 4, reason="four CUDA execution units")
def test_all_real_pair_with_mixed_eager_nulls(deterministic):
    if not (
        native._hybridep_available()
        and pair_capture.op_fuser_grouped_mlp_available()
        and pair_capture._te_min("2.19.0")
    ):
        pytest.skip("requires static HybridEP and TE grouped expert graph gradient copy-out")
    total = native.PER_RANK * native.WORLD
    criterion = native.PrescribedDepthCriterion(native._depths(total))
    pg = ProcessGroupCollection.use_mpu_process_groups()

    def batch(seed):
        return native._local(native._global_batch(total, seed=seed))

    with native._hadamard_fallback():
        eager, actual = [
            native._dsv4_model(**native.HYBRIDEP, moe_expert_rank_capacity_factor=4.0)
            for _ in range(2)
        ]
        native._sync_dense(eager)
        pair_capture.copy_model_state(eager, actual)
        _step(eager, batch(5), criterion, pg)
        sample = []

        def observe(hidden, ctx):
            if not sample:
                sample.append((hidden.detach().clone().requires_grad_(), ctx))

        with _all_real_pairs(actual, observe, Counter()):
            _step(actual, batch(5), criterion, pg)
        assert len(sample) == 1
        pair, replay, capture_buffers, initial = _capture_core_pair(actual, *sample[0])
        pair_capture.copy_model_state(eager, actual)

        def apply(hidden, ctx):
            return replay(*pair.args(hidden, ctx)).clone()

        optimizers = [torch.optim.SGD(model.parameters(), lr=0.05) for model in (eager, actual)]
        for iteration in (1, 2):
            counts = Counter()
            ref_loss, ref_grads, ref, ref_stats = _step(eager, batch(37 + iteration), criterion, pg)
            with _all_real_pairs(actual, apply, counts):
                loss, grads, test, stats = _step(actual, batch(37 + iteration), criterion, pg)
            result = dict(
                rank=dist.get_rank(),
                iteration=iteration,
                full=counts["full"],
                mixed=counts["mixed"],
                nulls=test.stats.null_moe_calls,
                gradients=len(grads),
                statistics=len(stats),
                losses=_differences(ref_loss, loss),
                grad_errors=_differences(ref_grads, grads),
                stat_errors=_differences(ref_stats, stats),
                capture_writes=[
                    k
                    for k, v in capture_buffers.items()
                    if k.startswith("moe/") and not torch.equal(v, initial[k])
                ],
                schedule=(
                    ref.stats.item_order == test.stats.item_order
                    and ref.stats.depths == test.stats.depths
                    and ref.stats.null_moe_calls == test.stats.null_moe_calls
                ),
                over_budget=test.stats.flagged_ranks["over_budget"],
            )
            rows = _gather(result)
            if dist.get_rank() == 0:
                print("RSBF_EP4:", rows, flush=True)
            # Assert collectively, so every EU reaches the same comparison before SGD.
            assert all(
                r["full"] == rows[0]["full"] > 0
                and r["gradients"] > 0
                and r["schedule"]
                and not (
                    r["losses"]
                    or r["grad_errors"]
                    or r["stat_errors"]
                    or r["over_budget"]
                    or r["capture_writes"]
                )
                for r in rows
            ), rows
            assert sum(r["mixed"] for r in rows) > 0 and sum(r["nulls"] for r in rows) > 0
            for model, local_grads, opt in zip((eager, actual), (ref_grads, grads), optimizers):
                reduced = native._dense_summed(model, local_grads)
                opt.zero_grad(set_to_none=True)
                for name, p in model.named_parameters():
                    if name in reduced:
                        p.grad = reduced[name].to(p.dtype)
                opt.step()
            errors = _gather(
                _differences(dict(eager.named_parameters()), dict(actual.named_parameters()))
            )
            assert not any(errors), errors
