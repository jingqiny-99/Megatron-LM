# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare the two-pass GLM MoK pipeline under uniform and stage-specific checkpoint plans."""

from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from functools import wraps
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state as ps
from megatron.core.models.hybrid import hybrid_model
from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.forward_replay import in_forward_replay
from megatron.core.transformer.loop.cross_loop_kv import current_cross_loop_pass
from megatron.core.transformer.loop.glm_dsa import GLMLoopDSAttention
from megatron.core.transformer.moe.moe_layer import MoELayer
from tests.unit_tests.transformer.loop import glm_ring_mok_utils as moe
from tests.unit_tests.transformer.loop import test_glm_dsa_loop_ring_pp_gpu as ring
from tests.unit_tests.transformer.loop import test_glm_ring_mok_pp_gpu as layout
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA


def _models(plans):
    config = replace(
        moe.config(2),
        linear_cp_mode=None,
        linear_cp_layout=None,
        loop_recompute=True,
        loop_recompute_edges=True,
        loop_recompute_block_stride=4,
        loop_ring_recompute_plan="/".join(plans),
    )
    with patch.object(moe, "config", return_value=config):
        return moe.models(2)


def _run(models, offsets, batches, plan):
    ids = [i for i, _ in batches]
    records, retained, forward_calls = {}, {}, Counter()
    actual, expected = Counter(), Counter()
    read_key = None
    route_calls = 0

    def sample_key(kind, layer, lap):
        name = kind, layer, lap
        index = forward_calls[name]
        forward_calls[name] += 1
        return (*name, ids[index])

    def watch_attention(vp):
        def before(module, inputs):
            nonlocal read_key
            phase = current_cross_loop_pass()
            lap = (1 if phase.is_source else 2) if phase is not None else (0 if vp == 0 else 3)
            replay = in_forward_replay()
            actual[module.layer_number, lap, replay] += 1
            if replay or phase is None:
                return
            if phase.is_source:
                latent = inputs[1]
                assert latent.requires_grad
                latent.retain_grad()
                retained[sample_key("source-grad", module.layer_number, lap)] = latent
            else:
                read_key = sample_key("read-grad", module.layer_number, lap)

        return before

    attend = dsa._run_sparse_attention

    def watch_read(**kwargs):
        nonlocal read_key
        phase = current_cross_loop_pass()
        if not in_forward_replay() and phase is not None and not phase.is_source:
            joint = kwargs["key"]
            assert (
                read_key is not None and joint.shape[0] == 2 * moe.SEQUENCE and joint.requires_grad
            )
            joint.retain_grad()
            retained[read_key] = joint
            read_key = None
        return attend(**kwargs)

    def watch_router(name):
        def observe(module, inputs, output):
            nonlocal route_calls
            route_calls += 1
            if not in_forward_replay():
                phase = current_cross_loop_pass()
                lap = (1 if phase.is_source else 2) if phase is not None else 0
                probs, mask = output
                records[sample_key("routes", name, lap)] = (layout._copy(mask), layout._copy(probs))

        return observe

    process_mtp_loss = hybrid_model.process_mtp_loss

    def watch_mtp(**kwargs):
        ce = kwargs["compute_language_model_loss"]

        @wraps(ce)
        def loss(*args, **options):
            result = ce(*args, **options)
            values = result[0] if isinstance(result, tuple) else result
            records[sample_key("mtp-token-loss", "mtp", 0)] = layout._copy(values)
            return result

        return process_mtp_loss(**{**kwargs, "compute_language_model_loss": loss})

    with ExitStack() as hooks:
        for vp, (chunk, offset) in enumerate(zip(models, offsets, strict=True)):
            for name, layer in chunk.named_modules():
                if isinstance(layer, GLMLoopDSAttention):
                    for lap in ((0,), (1, 2), (3,))[vp]:
                        expected[layer.layer_number, lap, False] = len(ids)
                        if plan[lap] == "C":
                            expected[layer.layer_number, lap, True] = len(ids)
                    hooks.callback(layer.register_forward_pre_hook(watch_attention(vp)).remove)
                if isinstance(layer, MoELayer):
                    assert layer.megakernel_experts is not None and layer.token_dispatcher is None
                    hook = watch_router(ring._global_name(name, offset))
                    hooks.callback(layer.router.register_forward_hook(hook).remove)
        compact = moe.watch_compaction(hooks)
        hooks.enter_context(patch.object(dsa, "_run_sparse_attention", watch_read))
        hooks.enter_context(patch.object(hybrid_model, "process_mtp_loss", watch_mtp))
        hooks.enter_context(patch.object(ring, "SEQUENCE", moe.SEQUENCE))
        losses, mtp = ring._step(models, batches, ring=True, collect_mtp=True)
    failures = []
    if actual != expected:
        failures.append(f"{plan}: DSA visits missing {expected-actual}, extra {actual-expected}")
    if not compact or len(compact) != route_calls or not torch.stack(compact).all().item():
        failures.append(f"{plan}: MoK compaction failed for {route_calls} router calls")
    for key, node in retained.items():
        if node.grad is None:
            failures.append(f"{plan}: missing gradient for {key}")
        else:
            grad = node.grad[: moe.SEQUENCE] if key[0] == "read-grad" else node.grad
            records[key] = layout._copy(grad)
    errors = [msg for rank in ring._gather_objects(failures) for msg in rank]
    assert not errors, "\n".join(errors)
    observed = {}
    for part in ring._gather_objects(records):
        assert observed.keys().isdisjoint(part)
        observed.update(part)
    assert len(losses) == moe.SAMPLES and {i for i, _ in losses} == set(range(moe.SAMPLES))
    grads, biases = {}, {}
    for chunk, offset in zip(models, offsets, strict=True):
        for name, param in chunk.named_parameters():
            key = ring._global_name(name, offset)
            assert key not in grads
            grads[key] = layout._copy(param.main_grad)
        for name, module in chunk.named_modules():
            if isinstance(module, MoELayer):
                biases[ring._global_name(name, offset)] = layout._copy(module.router.expert_bias)
    return layout._Step(dict(losses), layout._copy(mtp), observed, grads, biases)


def _compare_plans(reference_plans, candidate_plans):
    with moe.parallel(2):
        ep = ps.get_expert_model_parallel_rank()
        stage = ps.get_pipeline_model_parallel_rank()
        offsets = [sum(map(len, moe.CHUNKS[: vp * 2 + stage])) for vp in range(3)]
        batches = layout._batches(moe.sample_rows(), 2)
        models = _models(reference_plans)
        # Distinguish logical expert owners and exercise bias-corrected top-k.
        with torch.no_grad():
            for chunk in models:
                for layer in chunk.modules():
                    if isinstance(layer, MoELayer):
                        bias = layer.router.expert_bias
                        bias.copy_(torch.linspace(-0.025, 0.025, moe.EXPERTS, device=bias.device))
                        local = moe.EXPERTS // moe.EP
                        for fc in (layer.experts.linear_fc1, layer.experts.linear_fc2):
                            for i in range(local):
                                fc.get_parameter(f"weight{i}").mul_(0.75 + 0.025 * (ep * local + i))
        weights = [{n: layout._copy(v) for n, v in m.state_dict().items()} for m in models]
        reference = _run(models, offsets, batches, reference_plans[stage])
        del models
        models = _models(candidate_plans)
        for model, state in zip(models, weights, strict=True):
            model.load_state_dict(state, strict=True)
        candidate = _run(models, offsets, batches, candidate_plans[stage])

        # The layout oracle expects the complete logical reference for this EP owner.
        # Gather only its two PP stages. The MTP tail duplicates the input embedding.
        shards = [None, None]
        dist.all_gather_object(
            shards,
            (reference.grads, reference.biases),
            group=ps.get_pipeline_model_parallel_group(),
        )
        all_grads = {name: value for grads, _ in shards for name, value in grads.items()}
        all_biases = {name: value for _, biases in shards for name, value in biases.items()}
        layout._compare(candidate, replace(reference, grads=all_grads, biases=all_biases), ep)


@pytest.mark.skipif(not CUDA or ring.WORLD != moe.SAMPLES, reason="requires eight CUDA ranks")
def test_mok_two_pass_ccnc_matches_cccc():
    _compare_plans(("CCCC", "CCCC"), ("CCnC", "CCnC"))


@pytest.mark.skipif(not CUDA or ring.WORLD != moe.SAMPLES, reason="requires eight CUDA ranks")
@pytest.mark.parametrize(
    "candidate",
    [
        ("CnnC", "CnnC"),
        ("Cnnn", "Cnnn"),
        ("nnnn", "Cnnn"),
        ("Cnnn", "nnnn"),
        ("nnnn", "nnnn"),
    ],
    ids=[
        "keep-both-core-passes",
        "also-keep-coda",
        "keep-first-prelude",
        "keep-last-prelude",
        "keep-adjacent-preludes",
    ],
)
def test_mok_selective_activation_plans_match_ccnc(candidate):
    _compare_plans(("CCnC", "CCnC"), candidate)
