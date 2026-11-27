# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare native MoK and its MoonEP planner on a two-pass GLM pipeline with /E MTP."""

from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from megatron.core import parallel_state as ps
from megatron.core.models.hybrid import hybrid_model
from megatron.core.transformer.moe.moe_layer import MoELayer
from tests.unit_tests.transformer.loop import glm_ring_mok_utils as moe
from tests.unit_tests.transformer.loop import test_glm_dsa_loop_ring_pp_gpu as ring
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA


def _weights():
    # Native TE owns the same checkpoint in both arms. Distinguish EP shards and
    # use a nonconstant learned bias in the unmodified GLM sigmoid top-8 router.
    chunks = moe.models(2, backend=None)
    with torch.no_grad():
        for chunk in chunks:
            for layer in chunk.modules():
                if isinstance(layer, MoELayer):
                    bias = layer.router.expert_bias
                    bias.copy_(torch.linspace(-0.025, 0.025, moe.EXPERTS, device=bias.device))
                    for param in layer.experts.parameters():
                        param.mul_(0.75 + 0.125 * ps.get_expert_model_parallel_rank())
    return [
        {
            k: v.detach().to("cpu", copy=True) if torch.is_tensor(v) else v
            for k, v in chunk.state_dict().items()
        }
        for chunk in chunks
    ]


def _watch_moe(layer, hooks, moon):
    bridge = layer.megakernel_experts
    assert bridge is not None and layer.token_dispatcher is None
    for fc_name, routed in (
        ("linear_fc1", bridge.routed_fc1_parameters),
        ("linear_fc2", bridge.routed_fc2_parameters),
    ):
        native = getattr(layer.experts, fc_name)
        assert len(routed) == moe.EXPERTS // moe.EP
        assert all(p is native.get_parameter(f"weight{i}") for i, p in enumerate(routed))
    assert bridge.shared_fc1_weight is layer.shared_experts.linear_fc1.weight
    assert bridge.shared_fc2_weight is layer.shared_experts.linear_fc2.weight
    assert (bridge.virtual_layout is not None) == moon
    visits, pending = Counter(), set()

    def check_route(router, inputs, output):
        with torch.no_grad():
            probs, mask = output
            scores = router.gating(inputs[0]).reshape(-1, moe.EXPERTS).float().sigmoid()
            ids = (scores + router.expert_bias).topk(moe.TOPK, dim=-1).indices
            chosen = scores.gather(1, ids)
            weights = torch.zeros_like(scores).scatter_(
                1, ids, 2.5 * chosen / chosen.sum(-1, keepdim=True)
            )
            visits["router"] += 1
            if not torch.equal(mask, weights > 0) or not torch.allclose(
                probs, weights, rtol=0, atol=2e-6
            ):
                visits["bad_route"] += 1

    hooks.callback(layer.router.register_forward_hook(check_route).remove)
    if moon:
        transport = bridge.virtual_layout.transport
        make_plan, reduce_gradients = transport.plan, transport.reduce_gradients

        def plan(*args, **kwargs):
            result = make_plan(*args, **kwargs)
            pending.add(id(result))
            visits["plan"] += 1
            return result

        def reduce(plan, *args, **kwargs):
            result = reduce_gradients(plan, *args, **kwargs)
            if id(plan) not in pending:
                visits["unexpected_plan"] += 1
            pending.discard(id(plan))
            visits["reduce"] += 1
            return result

        hooks.enter_context(patch.object(transport, "plan", plan))
        hooks.enter_context(patch.object(transport, "reduce_gradients", reduce))
    return visits, pending


def _step(weights, batches, moon):
    chunks = moe.models(2, moon=moon)
    checks, mtp_tokens = [], []
    for chunk, state in zip(chunks, weights, strict=True):
        chunk.load_state_dict(state, strict=True)
    with ExitStack() as hooks:
        for vp, chunk in enumerate(chunks):
            for layer in chunk.modules():
                if isinstance(layer, MoELayer):
                    visits, pending = _watch_moe(layer, hooks, moon)
                    expected = moe.MICROBATCHES * (2 if vp == 1 else 1)
                    expected_counts = {"router": expected}
                    if moon:
                        expected_counts.update(plan=expected, reduce=expected)
                    checks.append((visits, pending, expected_counts))
        compaction = moe.watch_compaction(hooks)
        native_mtp_loss = hybrid_model.process_mtp_loss

        def capture_mtp(**kwargs):
            cross_entropy = kwargs["compute_language_model_loss"]

            def capture_tokens(*args, **loss_kwargs):
                output = cross_entropy(*args, **loss_kwargs)
                loss = output[0] if isinstance(output, tuple) else output
                mtp_tokens.append(loss.detach().float().cpu())
                return output

            return native_mtp_loss(**{**kwargs, "compute_language_model_loss": capture_tokens})

        hooks.enter_context(patch.object(hybrid_model, "process_mtp_loss", capture_mtp))
        hooks.enter_context(patch.object(ring, "SEQUENCE", moe.SEQUENCE))
        losses, mtp = ring._step(chunks, batches, ring=True, collect_mtp=True)
    problems = []
    if not checks or any(visits != expected or pending for visits, pending, expected in checks):
        problems.append("Router, Moon planner or matching backward reduce counts differ")
    if (
        not compaction
        or len(compaction) != sum(expected["router"] for _, _, expected in checks)
        or not torch.stack(compaction).all().item()
    ):
        problems.append("MoK compaction disagrees with the native router")
    expected_mtp = moe.MICROBATCHES if ps.get_pipeline_model_parallel_rank() == 1 else 0
    if len(mtp_tokens) != expected_mtp or any(v.numel() != moe.SEQUENCE for v in mtp_tokens):
        problems.append("Missing per-token MTP losses")
    if sorted(i for i, _ in losses) != list(range(moe.SAMPLES)) or any(
        loss.numel() != moe.SEQUENCE for _, loss in losses
    ):
        problems.append("Missing per-token LM losses")
    if mtp[1].item() != moe.SAMPLES * (moe.SEQUENCE - 1):
        problems.append("Wrong MTP token denominator")
    # Finish local checks on every rank before launching the second pipeline.
    failures = [issue for rank in ring._gather_objects(problems) for issue in rank]
    assert not failures, failures
    grads = {
        (vp, name): p.main_grad.detach().float().to("cpu", copy=True)
        for vp, chunk in enumerate(chunks)
        for name, p in chunk.named_parameters()
    }
    return dict(losses), mtp.cpu(), mtp_tokens, grads


@pytest.mark.skipif(not CUDA or ring.WORLD != 8, reason="requires eight CUDA ranks (PP2 x EP4)")
def test_glm_mok_moonep_matches_mok_pp2_ep4():
    """Hold the model, inputs, and fused SwiGLU fixed and change only MoonEP planning."""
    with moe.parallel(2):
        first = ps.get_data_parallel_rank() * moe.MICROBATCHES
        batches = moe.batches(moe.sample_rows(), range(first, first + moe.MICROBATCHES))
        weights = _weights()
        lm, mtp, mtp_tokens, grads = _step(weights, batches, moon=False)
        actual_lm, actual_mtp, actual_tokens, actual_grads = _step(weights, batches, moon=True)
        for i, loss in lm.items():
            torch.testing.assert_close(actual_lm[i], loss, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_mtp, mtp, rtol=1e-5, atol=1e-6)
        for new, ref in zip(actual_tokens, mtp_tokens, strict=True):
            torch.testing.assert_close(new, ref, rtol=1e-5, atol=1e-6)
        assert actual_grads.keys() == grads.keys()
        for key, ref in grads.items():
            delta = (actual_grads[key] - ref).norm().item()
            bound = 1e-5 * ref.norm().item() + 1e-12 * ref.numel() ** 0.5
            assert delta <= bound, f"{key}: main-grad delta {delta:g} exceeds {bound:g}"
