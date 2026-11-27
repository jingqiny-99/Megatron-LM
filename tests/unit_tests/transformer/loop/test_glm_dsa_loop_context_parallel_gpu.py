# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare the two-pass decoder across a CP boundary with an unsharded reference."""

import copy
import os

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA, model_parallel_once
from tests.unit_tests.transformer.loop.test_glm_dsa_loop_model_gpu import SEQUENCE, VOCAB, _model


@pytest.mark.skipif(
    not CUDA or int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="launch with torchrun on two GPUs",
)
def test_two_pass_checkpointed_decoder_cp2_matches_cp1(monkeypatch):
    # Compare different GEMM shapes in full FP32. The global CUDA override also
    # covers Transformer Engine GEMMs, which do not use torch.matmul's setting.
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    # Each rank computes the same full-sequence reference, then owns the front and
    # back zigzag slices of this example. No DP averaging is performed in either run.
    row = ((torch.arange(SEQUENCE + 1) * 17 + 5) % VOCAB).reshape(1, -1)
    positions = torch.arange(SEQUENCE).reshape(1, -1)
    with model_parallel_once(context_parallel_size=1):
        torch.manual_seed(42)
        reference = _model(True, False)
        initial = copy.deepcopy(reference.state_dict())
        expected = reference(row[:, :-1].cuda(), positions.cuda(), None, labels=row[:, 1:].cuda())
        expected.sum().backward()
        gradients = {
            n: p.grad.detach().clone() for n, p in reference.named_parameters() if p.grad is not None
        }
        expected = expected.detach()
        del reference

    with model_parallel_once(context_parallel_size=2):
        group = parallel_state.get_context_parallel_group()
        rank = dist.get_rank(group)
        chunks = positions.chunk(4, dim=1)
        local_ids = torch.cat([chunks[rank], chunks[-rank - 1]], dim=1).squeeze(0)
        model = _model(True, True, cp=2)
        model.load_state_dict(initial)
        observed = model(
            row[:, local_ids].cuda(),
            positions[:, local_ids].cuda(),
            None,
            labels=row[:, local_ids + 1].cuda(),
        )
        torch.testing.assert_close(observed, expected[:, local_ids], rtol=1e-5, atol=1e-5)
        observed.sum().backward()
        actual = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        assert actual.keys() == gradients.keys()
        failures = []
        for name, grad in actual.items():
            total = grad.contiguous()
            dist.all_reduce(total, group=group)
            expected_grad = gradients[name]
            if not torch.allclose(total, expected_grad, rtol=1e-3, atol=1e-4):
                delta = (total - expected_grad).norm()
                failures.append(
                    f"{name}: abs_max={(total - expected_grad).abs().max().item():.3g}, "
                    f"rel_L2={(delta / expected_grad.norm().clamp_min(1e-12)).item():.3g}, "
                    f"norm_ratio={(total.norm() / expected_grad.norm().clamp_min(1e-12)).item():.3g}"
                )
        assert not failures, "\n".join(failures)
