# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real MoK backward on a null source, and online PP1 vs independent fixed-depth steps."""

from unittest.mock import patch

import pytest
import torch

from megatron.core import parallel_state as ps
from megatron.core.transformer.moe.megakernel.mok.backend import MoKMegakernel
from tests.unit_tests.transformer.loop import glm_ring_mok_utils as moe
from tests.unit_tests.transformer.loop import test_glm_dsa_loop_ring_pp_gpu as ring
from tests.unit_tests.transformer.loop import test_glm_ring_mok_pp_gpu as oracle
from tests.unit_tests.transformer.loop import test_glm_ring_online_mok_gpu as calibration
from tests.unit_tests.transformer.loop import test_glm_ring_stratified_mok_cp_gpu as mixed

pytestmark = pytest.mark.skipif(
    not ring.CUDA or ring.WORLD != 8, reason="requires 8 SM100/103 GPUs"
)


def test_null_source_keeps_remote_expert_weight_gradients():
    with patch.object(moe, "EP", 8), moe.parallel(1, cp=4):
        model = oracle._pp1_model(cp=4, rounds=3)
        module = next(m for m in model.modules() if isinstance(m, MoKMegakernel))
        for p in module.parameters():
            p.main_grad = torch.zeros_like(p, dtype=torch.float32)
        rank = ps.get_expert_model_parallel_rank()
        live = rank % 2 == 0
        torch.manual_seed(971 + rank)
        x = torch.randn(moe.SEQUENCE, 1, module.hidden_size, device="cuda", dtype=torch.bfloat16)
        upstream = torch.randn_like(x) if live else torch.zeros_like(x)
        probs = torch.full((moe.SEQUENCE, moe.EXPERTS), 1 / moe.TOPK, device=x.device)
        # Each token reaches one expert on every owner. Alternate its local expert
        # over rows so both local FC1/FC2 pairs on the stopped rank get peer gradients.
        rows = torch.arange(moe.SEQUENCE, device=x.device)
        experts = torch.arange(moe.EXPERTS, device=x.device)
        mask = rows[:, None] % 2 == experts[None] % 2
        if not live:
            x.zero_()
        x.requires_grad_()
        expected = module(x, probs, mask)
        expected.backward(upstream)
        expected_xgrad = x.grad.clone()
        expected_grad = [p.main_grad.clone() for p in module.autograd_routed_parameters]
        for p in module.parameters():
            p.grad = None
            p.main_grad.zero_()
        x.grad = None
        checks = []
        compact = module._apply_compact

        def checked(hidden, weights, ids):
            checks.append((hidden == 0).all() & (weights == 0).all() & (ids == -1).all())
            return compact(hidden, weights, ids)

        if live:
            actual = module(x, probs, mask)
        else:
            # The null source may hold arbitrary real activations; none may enter the MoK kernel.
            with torch.no_grad():
                x.fill_(float("nan"))
            with patch.object(module, "_apply_compact", checked):
                actual = module.forward_null(x)
        actual.backward(upstream)
        torch.testing.assert_close(x.grad, expected_xgrad, atol=0.002, rtol=0.01)
        if not live:
            assert checks[0].item() and x.grad.count_nonzero() == 0
            assert actual.count_nonzero() == 0
        for p, expected in zip(module.autograd_routed_parameters, expected_grad, strict=True):
            assert expected.norm() > 0
            oracle._assert_gradient(p.main_grad.cpu(), expected.cpu(), "remote-owner")


def test_online_dp_matches_fixed_depth_references():
    cp = mixed.CP
    rows = torch.randint(
        moe.VOCAB,
        (sum(mixed.COUNTS), mixed.SEQUENCE + 1),
        generator=torch.Generator().manual_seed(734),
    )
    samples = [(i, row[None]) for i, row in enumerate(rows)]
    with patch.object(moe, "EP", 8), moe.parallel(1, cp=cp):
        full = oracle._pp1_model(cp=cp, rounds=3)
        state = {k: oracle._copy(v) for k, v in full.state_dict().items()}
        thresholds, depths = calibration._thresholds_from_fixed_reference(full, samples)
        del full
        # Pair different depths so both homes exercise a null tail in a shared EP group.
        ordered = sorted(samples, key=lambda row: depths[row[0]])
        stream = [entry for pair in zip(ordered[:4], ordered[4:], strict=True) for entry in pair]
        dp = ps.get_data_parallel_rank()
        online = moe.models(1, cp=cp, rounds=3, thresholds=thresholds)[0]
        online.load_state_dict(state, strict=True)
        got_loss, got_mtp, got_records, got_grad = mixed._run(
            [online], [0], stream[dp::2], depths, 1
        )
        assert [d for d, _ in online.decoder.take_online_exit_records()] == [
            depths[i] for i, _ in stream[dp::2]
        ]
        del online
        want_loss, want_records, want_grad = {}, {}, {}
        want_mtp = torch.zeros_like(got_mtp)
        for depth, count in enumerate(mixed.COUNTS, 1):
            fixed = moe.models(1, cp=cp, rounds=depth)[0]
            fixed.load_state_dict({k: state[k] for k in fixed.state_dict()}, strict=True)
            assigned = [row for row in stream if depths[row[0]] == depth][dp::2]
            loss, mtp, rec, grads = mixed._run([fixed], [0], assigned, depths, 1)
            want_loss.update(loss)
            want_mtp += mtp
            want_records.update(rec)
            for name, grad in grads.items():
                want_grad.setdefault(name, torch.zeros_like(grad)).add_(
                    grad, alpha=count / len(rows)
                )
            del fixed
        assert got_loss.keys() == want_loss.keys()
        for sample in got_loss:
            torch.testing.assert_close(
                got_loss[sample].float(), want_loss[sample].float(), atol=0.01, rtol=0.005
            )
        torch.testing.assert_close(got_mtp, want_mtp, rtol=0.002, atol=0)
        mixed._check_records(got_records, want_records, depths)
        for name, grad in got_grad.items():
            oracle._assert_gradient(grad, want_grad[name], name)
