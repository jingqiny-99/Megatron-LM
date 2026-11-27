# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare a BF16 MoK PP2/CP4/EP4 mixed step to fixed-depth PP1/CP4/EP4 references."""

from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state as ps
from megatron.core.transformer.loop.depth_schedule import stratified_depth
from megatron.core.utils import get_attr_wrapped_model
from tests.unit_tests.transformer.loop import glm_ring_mok_utils as moe
from tests.unit_tests.transformer.loop import test_glm_dsa_loop_ring_pp_gpu as ring
from tests.unit_tests.transformer.loop import test_glm_ring_mok_pp_gpu as oracle

CP, ITERATION, COUNTS = 4, 7, [2, 4, 2]
SEQUENCE = CP * moe.SEQUENCE


def _forward(data_iterator, model):
    hybrid = get_attr_wrapped_model(model, "decoder", return_model_obj=True)
    cp = ps.get_context_parallel_rank()
    if hybrid.pre_process or hybrid.post_process:
        sample, row = next(data_iterator)
        pieces = torch.arange(SEQUENCE, device="cuda").chunk(2 * CP)
        positions = torch.cat((pieces[cp], pieces[-1 - cp]))[None]
        tokens = row[:, :-1].index_select(1, positions[0])
        labels = row[:, 1:].index_select(1, positions[0]) if hybrid.post_process else None
    else:
        sample = positions = tokens = labels = None
    mask = torch.ones_like(labels, dtype=torch.float32) if labels is not None else None
    output = model(tokens, positions, None, labels=labels, loss_mask=mask)

    def loss(value):
        return value.sum(), mask.sum().int(), ((sample, cp), value.detach().cpu())

    return output, loss


def _run(models, offsets, samples, depths, pp, *, record_callback=None):
    ids = [sample for sample, _ in samples]
    with ExitStack() as hooks:
        obs, retained = oracle._observations(models, offsets, ids, hooks, depths=depths, cp=CP)
        compactions = moe.watch_compaction(hooks)
        hooks.enter_context(patch.object(ring, "_forward_step", _forward))
        hooks.enter_context(patch.object(ring, "SEQUENCE", SEQUENCE))
        loss, mtp = ring._step(
            models,
            [(i, row.cuda()) for i, row in samples],
            pp == 2,
            True,
            counts=COUNTS if pp == 2 else None,
            iteration=ITERATION,
            record_callback=record_callback,
        )
    routes = sum(k[0] == "routes" for k in obs)
    assert routes and len(compactions) >= routes and torch.stack(compactions).all()
    for key, value in retained.items():
        assert value.grad is not None, key
        grad = oracle._copy(value.grad)
        if key[0] == "source-grad":
            obs[key] = grad
        else:
            obs[key] = grad[:SEQUENCE]
            obs[("window-grad", *key[1:])] = grad[SEQUENCE:]
    # The other ep/cp shards hold different experts and query positions. Join only
    # matching context shards across the two DP replicas or two pipeline stages.
    group = ps.get_pipeline_model_parallel_group() if pp == 2 else ps.get_data_parallel_group()
    partial = [None] * group.size()
    dist.all_gather_object(partial, obs, group=group)
    records = {}
    for part in partial:
        assert records.keys().isdisjoint(part)
        records.update(part)
    grads = {
        ring._global_name(name, offset): oracle._copy(p.main_grad)
        for model, offset in zip(models, offsets, strict=True)
        for name, p in model.named_parameters()
    }
    return dict(loss), mtp, records, grads


def _check_records(actual, expected, depths):
    assert actual.keys() == expected.keys()
    source = {(layer, i) for layer in oracle.CORE for i in range(len(depths))}
    assert {
        (layer, sample) for kind, layer, lap, sample in actual if kind == "source-grad"
    } == source
    for lap in (2, 3):
        consumers = {(layer, i) for layer, i in source if depths[i] >= lap}
        for kind in ("read-grad", "window-grad", "source-topk"):
            assert {(n, i) for k, n, r, i in actual if k == kind and r == lap} == consumers
    assert {(n, i) for k, n, r, i in actual if k == "selector"} == {
        (n, i) for n, i in source if n in (4, 8)
    }
    for key, baseline in expected.items():
        kind, layer, lap, i = key
        value = actual[key]
        if kind == "routes":
            mask, probs = value
            old_mask, old_probs = baseline
            assert (mask.sum(-1) == moe.TOPK).all() and (old_mask.sum(-1) == moe.TOPK).all()
            same = mask & old_mask
            assert same.sum().item() / (moe.SEQUENCE * moe.TOPK) >= 0.995, key
            torch.testing.assert_close(
                probs[same], old_probs[same], rtol=0.01, atol=0.001, msg=str(key)
            )
        elif kind == "mtp-token-loss":
            assert value.numel() == moe.SEQUENCE
            torch.testing.assert_close(
                value.float(), baseline.float(), rtol=0.005, atol=0.01, msg=str(key)
            )
        elif kind == "source-topk":
            idx, sz = value
            old_idx, old_sz = baseline
            torch.testing.assert_close(sz, old_sz, rtol=0, atol=0, msg=str(key))
            valid = torch.arange(idx.shape[-1]) < sz[..., None]
            same = ((idx[..., None] == old_idx[..., None, :]) & valid[..., None, :]).any(-1)
            assert (same & valid).sum().item() / valid.sum().item() >= 0.995, key
            if lap == 3:
                for records in (actual, expected):
                    second = records[(kind, layer, 2, i)]
                    assert all(
                        torch.equal(x, y) for x, y in zip(records[key], second, strict=True)
                    ), key
        elif kind == "selector":
            assert lap == 1 and value is baseline is True
        else:
            assert baseline.norm() > 0 and value.norm() > 0, key
            oracle._assert_gradient(value, baseline, key)


def _compare_mixed_step(*, calibrate=None):
    rows = torch.randint(
        moe.VOCAB, (sum(COUNTS), SEQUENCE + 1), generator=torch.Generator().manual_seed(734)
    )
    samples = [(i, rows[i : i + 1]) for i in range(len(rows))]
    thresholds = None
    want_loss, want_records = {}, {}
    with moe.parallel(1, CP):
        # Initialize distributed and select this process's local GPU first.
        want_mtp = torch.zeros(2, device="cuda")
        full = oracle._pp1_model(cp=CP, rounds=3)
        state = {name: oracle._copy(v) for name, v in full.state_dict().items()}
        want_grad = {
            name: torch.zeros_like(p, device="cpu", dtype=torch.float32)
            for name, p in full.named_parameters()
        }
        reference_owner = (ps.get_context_parallel_rank(), ps.get_expert_model_parallel_rank())
        if calibrate is not None:
            thresholds, depths = calibrate(full, samples)
        else:
            cfg = moe.config(2, cp=CP, rounds=3, counts=COUNTS)
            depths = [stratified_depth(cfg, ITERATION, i) for i in range(len(rows))]
        assert [depths.count(d) for d in (1, 2, 3)] == COUNTS
        del full
        for d, n in enumerate(COUNTS, 1):
            model = moe.models(1, cp=CP, rounds=d)[0]
            model.load_state_dict({k: state[k] for k in model.state_dict()}, strict=True)
            assigned = [s for s in samples if depths[s[0]] == d][ps.get_data_parallel_rank() :: 2]
            loss, mtp, rec, grads = _run([model], [0], assigned, depths, 1)
            want_loss.update(loss)
            want_mtp += mtp
            want_records.update(rec)
            for name, grad in grads.items():
                want_grad[name].add_(grad, alpha=n / len(rows))
            del model
    with moe.parallel(2, CP):
        assert reference_owner == (
            ps.get_context_parallel_rank(),
            ps.get_expert_model_parallel_rank(),
        )
        models = moe.models(
            2, cp=CP, rounds=3, counts=COUNTS if thresholds is None else None, thresholds=thresholds
        )
        pp = ps.get_pipeline_model_parallel_rank()
        offsets = [sum(map(len, moe.CHUNKS[: 2 * vp + pp])) for vp in range(3)]
        for model, offset in zip(models, offsets, strict=True):
            model.load_state_dict(
                {k: state[ring._global_name(k, offset)] for k in model.state_dict()}, strict=True
            )
        records = []
        got_loss, got_mtp, got_records, got_grad = _run(
            models, offsets, samples, depths, 2, record_callback=records.append
        )
        failures = []
        try:
            if thresholds is not None and pp == 1:
                decisions = records[0].online_decisions
                assert [(i, r, stop) for i, r, stop, score in decisions] == [
                    (i, r, r == d) for i, d in enumerate(depths) for r in range(1, d + 1)
                ]
                for i, r, stop, score in decisions:
                    if r < 3:
                        assert stop == (score <= thresholds[r - 1])
                    else:
                        assert stop and score is None
            assert (
                got_loss.keys()
                == want_loss.keys()
                == {(i, cp) for i in range(8) for cp in range(CP)}
            )
            for i, value in got_loss.items():
                torch.testing.assert_close(
                    value.float(), want_loss[i].float(), rtol=0.005, atol=0.01, msg=str(i)
                )
            torch.testing.assert_close(got_mtp[:1], want_mtp[:1], rtol=0.002, atol=0)
            torch.testing.assert_close(got_mtp[1:], want_mtp[1:], rtol=0, atol=0)
            assert got_mtp.min() > 0
            _check_records(got_records, want_records, depths)
            for name, grad in got_grad.items():
                oracle._assert_gradient(grad, want_grad[name], name)
            for r in range(3):
                assert want_grad[f"decoder.loop_gates.{r}.proj.weight"].norm() > 0
        except AssertionError as error:
            failures.append(f"CP{reference_owner[0]} PP{pp}: {error}")
        failures = [x for group in ring._gather_objects(failures) for x in group]
        assert not failures, "\n".join(failures)


@pytest.mark.skipif(not ring.CUDA or ring.WORLD != 8, reason="requires eight SM100/103 CUDA ranks")
def test_stratified_mok_pp2_cp4_ep4_matches_fixed_models():
    _compare_mixed_step()
