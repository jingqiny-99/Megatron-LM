# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""A mixed-depth pipeline step against three independently run fixed-depth eager models."""

from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from megatron.core import parallel_state as ps
from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.forward_replay import in_forward_replay, statistics_round
from megatron.core.transformer.loop.cross_loop_kv import current_cross_loop_pass
from megatron.core.transformer.loop.depth_schedule import stratified_depth
from megatron.core.transformer.loop.glm_dsa import GLMLoopDSAttention
from tests.unit_tests.transformer.loop import test_glm_dsa_loop_ring_pp_gpu as ring
from tests.unit_tests.transformer.loop.test_loop_attn_res_dsv4_gpu import full_precision_gemms


def _run(models, offsets, samples, depths, *, counts=None, iteration):
    """Measure the full R1 latent adjoint and each isolated later read of that latent."""
    batches = [(index, row.cuda()) for index, row in samples]
    ids = [i for i, row in samples]
    visits, retained = Counter(), {}
    current = None
    attend = dsa._run_sparse_attention

    def on_attention(module, args):
        nonlocal current
        scope = current_cross_loop_pass()
        if scope is None or in_forward_replay():
            return
        lap = statistics_round()[0]
        visit = (module.layer_number, lap)
        sample = [i for i in ids if depths[i] >= lap][visits[visit]]
        visits[visit] += 1
        current = (sample, module.layer_number, lap)
        if scope.is_source:
            assert lap == 1
            args[1].retain_grad()
            retained[current] = args[1]

    def on_sparse(**kwargs):
        scope = current_cross_loop_pass()
        if scope is not None and not scope.is_source and not in_forward_replay():
            key = kwargs["key"]
            assert key.size(0) == 2 * ring.SEQUENCE
            key.retain_grad()
            retained[current] = key
        return attend(**kwargs)

    selector_calls = Counter()

    def on_selector(layer):
        def count(_module, _inputs):
            scope = current_cross_loop_pass()
            if scope is not None and not in_forward_replay():
                assert scope.is_source, "Later laps must read the first-pass top-k"
                selector_calls[layer] += 1

        return count

    with ExitStack() as hooks:
        hooks.enter_context(patch.object(dsa, "_run_sparse_attention", on_sparse))
        for model in models:
            for layer in model.modules():
                if isinstance(layer, GLMLoopDSAttention):
                    hooks.callback(layer.register_forward_pre_hook(on_attention).remove)
                    if layer.indexer is not None:
                        hooks.callback(
                            layer.indexer.linear_wq_b.register_forward_pre_hook(
                                on_selector(layer.layer_number)
                            ).remove
                        )
        losses, mtp = ring._step(
            models,
            batches,
            ring=counts is not None,
            collect_mtp=True,
            counts=counts,
            iteration=iteration,
        )
    expected_selectors = {
        layer.layer_number: len(ids)
        for model in models
        for layer in model.modules()
        if isinstance(layer, GLMLoopDSAttention)
        and layer.indexer is not None
        and (layer.layer_number, 1) in visits
    }
    assert selector_calls == expected_selectors
    sparse = {}
    for k, tensor in retained.items():
        assert tensor.grad is not None, k
        grad = tensor.grad.detach().cpu()
        if k[2] > 1:
            # The first half is the R1 latent read; the second half is this lap's
            # own independently projected sliding-window KV. Both receive gradients.
            first, current_window = grad.split(ring.SEQUENCE)
            assert first.norm() > 0 and current_window.norm() > 0, k
        sparse[k] = grad.clone()
    combined = {}
    for partial in ring._gather_objects(sparse):
        assert combined.keys().isdisjoint(partial)
        combined.update(partial)
    grads = {
        ring._global_name(name, offset): p.main_grad.detach().cpu().clone()
        for model, offset in zip(models, offsets)
        for name, p in model.named_parameters()
    }
    return dict(losses), mtp, combined, grads


def _check_depth_subbatches(counts):
    world, iteration = ring.WORLD, 7
    sections = ([2] + [4] * (world - 1), [4] * world, [4] * (world - 1) + [0])
    total = sum(counts)
    rows = torch.randint(
        ring.VOCAB, (total, ring.SEQUENCE + 1), generator=torch.Generator().manual_seed(802)
    )
    with ring._parallel(world):
        holder, _ = ring._build(world, True, sections, rounds=3, counts=counts)
        depths = [stratified_depth(holder[0].config, iteration, i) for i in range(total)]
        assert tuple(depths.count(d) for d in (1, 2, 3)) == counts
        del holder
    samples = [(i, rows[i : i + 1]) for i in range(total)]
    want_loss, want_sparse = {}, {}
    want_mtp = torch.zeros(2, device="cuda")
    with ring._parallel(1):
        full = ring._build(1, True, sections, rounds=3)[0][0]
        state = {k: v.clone() if torch.is_tensor(v) else v for k, v in full.state_dict().items()}
        want_grads = {n: torch.zeros_like(p, device="cpu") for n, p in full.named_parameters()}
        del full
        for depth, count in enumerate(counts, 1):
            if not count:
                continue
            models, offsets = ring._build(1, True, sections, rounds=depth)
            model = models[0]
            model.load_state_dict({k: state[k] for k in model.state_dict()}, strict=True)
            group = [row for row in samples if depths[row[0]] == depth]
            group = group[ps.get_data_parallel_rank() :: world]
            loss, mtp, sparse, grads = _run(models, offsets, group, depths, iteration=iteration)
            want_loss.update(loss)
            want_mtp += mtp
            want_sparse.update(sparse)
            for name, value in grads.items():
                want_grads[name].add_(value, alpha=count / total)
    with ring._parallel(world):
        models, offsets = ring._build(world, True, sections, rounds=3, counts=counts)
        for model, offset in zip(models, offsets):
            model.load_state_dict(
                {k: state[ring._global_name(k, offset)] for k in model.state_dict()}, strict=True
            )
        got_loss, got_mtp, got_sparse, got_grads = _run(
            models, offsets, samples, depths, counts=counts, iteration=iteration
        )
        first_core = sum(sections[0]) + 1
        oracle = {
            (i, layer, lap)
            for i in range(total)
            for layer in range(first_core, first_core + sum(sections[1]))
            for lap in range(1, depths[i] + 1)
        }
        assert got_sparse.keys() == want_sparse.keys() == oracle
        for k in sorted(oracle):
            assert want_sparse[k].norm() > 0, k
            torch.testing.assert_close(
                got_sparse[k], want_sparse[k], rtol=1e-4, atol=2e-6, msg=str(k)
            )
        assert got_loss.keys() == want_loss.keys() == set(range(total))
        for i in got_loss:
            torch.testing.assert_close(got_loss[i], want_loss[i], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(got_mtp, want_mtp, rtol=1e-5, atol=1e-5)
        names = Counter(n for rank in ring._gather_objects(list(got_grads)) for n in rank)
        assert names == Counter(
            {n: 2 if n == "embedding.word_embeddings.weight" else 1 for n in want_grads}
        )
        for r in range(3):
            assert bool(want_grads[f"decoder.loop_gates.{r}.proj.weight"].norm()) == any(counts[r:])
        for n, g in got_grads.items():
            torch.testing.assert_close(g, want_grads[n], rtol=1e-4, atol=2e-6, msg=n)


@pytest.mark.skipif(not ring.CUDA or ring.WORLD not in (2, 7), reason="requires PP2 or PP7")
def test_stratified_pp_matches_fixed_depth_subbatches(full_precision_gemms):
    _check_depth_subbatches((ring.WORLD, 2 * ring.WORLD, ring.WORLD))


@pytest.mark.skipif(not ring.CUDA or ring.WORLD not in (2, 7), reason="requires PP2 or PP7")
@pytest.mark.parametrize("depth", [1, 2], ids=["all1", "all2"])
def test_globally_empty_laps_match_pp1(depth, full_precision_gemms):
    counts = tuple(2 * ring.WORLD if d == depth else 0 for d in (1, 2, 3))
    _check_depth_subbatches(counts)
