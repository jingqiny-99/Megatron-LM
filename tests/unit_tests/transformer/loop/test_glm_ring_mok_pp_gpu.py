# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare an eager GLM decoder and its four-lap pipeline with identical BF16 MoK experts.

Eight ranks first run PP1 x DP8, then PP2 x DP4. The eight 512-token samples enter each
EP4 collective in the same order. Each sample runs both tied core passes and one /E MTP layer.
"""

from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from functools import wraps
from unittest.mock import patch

import pytest
import torch

from megatron.core import parallel_state as ps
from megatron.core.models.hybrid import hybrid_model
from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.forward_replay import in_forward_replay, statistics_round
from megatron.core.transformer.loop.cross_loop_kv import current_cross_loop_pass
from megatron.core.transformer.loop.glm_dsa import GLMLoopDSAttention
from megatron.core.transformer.moe.moe_layer import MoELayer
from tests.unit_tests.transformer.loop import glm_ring_mok_utils as moe
from tests.unit_tests.transformer.loop import test_glm_dsa_loop_ring_pp_gpu as ring
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA

# One-based DSA block numbers in the shared physical core.
CORE = range(4, 9)


def _pp1_model(**kwargs):
    model = moe.models(1, **kwargs)[0]
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, MoELayer):
                bias = layer.router.expert_bias
                bias.copy_(torch.linspace(-0.025, 0.025, moe.EXPERTS, device=bias.device))
                local = moe.EXPERTS // moe.EP
                for fc in (layer.experts.linear_fc1, layer.experts.linear_fc2):
                    for i in range(local):
                        logical_expert = ps.get_expert_model_parallel_rank() * local + i
                        fc.get_parameter(f"weight{i}").mul_(0.75 + 0.025 * logical_expert)
    return model


def _copy(value):
    return value.detach().to("cpu", copy=True) if isinstance(value, torch.Tensor) else value


def _batches(rows, pp):
    # Match the source-rank order of each expert collective as well as the global sample set.
    ep_rank = ps.get_expert_model_parallel_rank()
    if pp == 1:
        ids = [ep_rank + moe.EP * ps.get_expert_data_parallel_rank()]
    else:
        ids = [ep_rank, ep_rank + moe.EP]
    return moe.batches(rows, ids)


def _observations(models, offsets, ids, hooks, *, depths=None, cp=1):
    """Key each layer's FIFO of microbatches by global sample, separately in each core pass."""
    records, retained, visits = {}, {}, Counter()
    active_consumer = active_module = None

    def key_for(kind, layer, phase):
        visit = (kind, layer, phase)
        position = visits[visit]
        visits[visit] += 1
        eligible = ids if depths is None or not phase else [i for i in ids if depths[i] >= phase]
        return (kind, layer, phase, eligible[position])

    def watch_key(module, args):
        nonlocal active_consumer, active_module
        visit = current_cross_loop_pass()
        if visit is None or in_forward_replay():
            return
        phase = statistics_round()[0]
        active_module = module
        key = key_for("source-grad" if phase == 1 else "read-grad", module.layer_number, phase)
        if visit.is_source:
            latent = args[1]
            latent.retain_grad()
            retained[key] = latent
        else:
            active_consumer = key

    attend = dsa._run_sparse_attention

    def capture_read(**kwargs):
        nonlocal active_consumer
        visit = current_cross_loop_pass()
        if visit is not None and not visit.is_source and not in_forward_replay():
            assert active_consumer is not None
            joint = kwargs["key"]
            global_length = cp * moe.SEQUENCE
            if cp == 1:
                assert joint.shape[0] == 2 * global_length
            else:
                assert visit.read(active_module, "dsa_latent").shape[0] == global_length
                assert global_length < joint.shape[0] <= 2 * global_length
                records[("source-topk", *active_consumer[1:])] = (
                    _copy(visit.read(active_module, "dsa_topk")),
                    _copy(visit.read(active_module, "dsa_topk_length")),
                )
            joint.retain_grad()
            retained[active_consumer] = joint
            active_consumer = None
        return attend(**kwargs)

    def route_hook(name):
        def capture(_router, _inputs, output):
            # A core FFN can live on a stage without DSA. Its round is
            # defined by the schedule even when no sparse KV bank lives here.
            if in_forward_replay():
                return
            current = statistics_round()
            phase = current[0] if current is not None else 0
            probs, mask = output
            key = key_for("routes", name, phase)
            # Preserve the original expert IDs and the score at every selected ID.
            records[key] = (_copy(mask), _copy(probs))

        return capture

    def on_selector(layer):
        def count(_module, _inputs):
            scope = current_cross_loop_pass()
            if scope is not None and not in_forward_replay():
                assert scope.is_source
                records[key_for("selector", layer, 1)] = True

        return count

    for model, offset in zip(models, offsets, strict=True):
        for name, layer in model.named_modules():
            if isinstance(layer, GLMLoopDSAttention) and layer.layer_number in CORE:
                hooks.callback(layer.register_forward_pre_hook(watch_key).remove)
                if depths is not None and layer.indexer is not None:
                    hooks.callback(
                        layer.indexer.linear_wq_b.register_forward_pre_hook(
                            on_selector(layer.layer_number)
                        ).remove
                    )
            if isinstance(layer, MoELayer):
                assert layer.megakernel_experts is not None and layer.token_dispatcher is None
                router_hook = route_hook(ring._global_name(name, offset))
                hooks.callback(layer.router.register_forward_hook(router_hook).remove)
    hooks.enter_context(patch.object(dsa, "_run_sparse_attention", capture_read))

    mtp_loss = hybrid_model.process_mtp_loss

    @wraps(mtp_loss)
    def capture_mtp(**kwargs):
        loss_fn = kwargs["compute_language_model_loss"]

        @wraps(loss_fn)
        def capture_ce(*args, **loss_kwargs):
            result = loss_fn(*args, **loss_kwargs)
            loss = result[0] if isinstance(result, tuple) else result
            records[key_for("mtp-token-loss", "mtp", 0)] = _copy(loss)
            return result

        return mtp_loss(**{**kwargs, "compute_language_model_loss": capture_ce})

    hooks.enter_context(patch.object(hybrid_model, "process_mtp_loss", capture_mtp))
    return records, retained


@dataclass
class _Step:
    losses: dict
    mtp: torch.Tensor
    observations: dict
    grads: dict
    biases: dict


def _run(models, offsets, batches, pp):
    with ExitStack() as hooks:
        ids = [i for i, _ in batches]
        records, retained = _observations(models, offsets, ids, hooks)
        compact = moe.watch_compaction(hooks)
        hooks.enter_context(patch.object(ring, "SEQUENCE", moe.SEQUENCE))
        losses, mtp = ring._step(models, batches, ring=pp == 2, collect_mtp=True)
    # Each PP rank owns a recurrent MoE, including the stage with a head-only coda chunk.
    calls = sum(key[0] == "routes" for key in records)
    failures = []
    if not compact or len(compact) != calls or not torch.stack(compact).all().item():
        failures.append(f"MoK compaction disagreed with logical routing ({len(compact)}/{calls})")
    for key, tensor in retained.items():
        grad = tensor.grad
        if grad is None:
            failures.append(f"{key} has no gradient")
        else:
            # The front half is the isolated pass-2 adjoint of the pass-1 latent.
            if key[0] == "read-grad":
                grad = grad[: moe.SEQUENCE]
            records[key] = _copy(grad)
    combined = {}
    for part in ring._gather_objects(records):
        assert combined.keys().isdisjoint(part), "Two stages claimed the same sample observation"
        combined.update(part)
    by_sample = dict(losses)
    assert len(losses) == len(by_sample) == moe.SAMPLES and set(by_sample) == set(
        range(moe.SAMPLES)
    )
    grads, biases = {}, {}
    for model, offset in zip(models, offsets, strict=True):
        for name, param in model.named_parameters():
            key = ring._global_name(name, offset)
            assert key not in grads
            grads[key] = _copy(param.main_grad)
        for name, layer in model.named_modules():
            if isinstance(layer, MoELayer):
                biases[ring._global_name(name, offset)] = _copy(layer.router.expert_bias)
    failures = [msg for rank in ring._gather_objects(failures) for msg in rank]
    assert not failures, "\n".join(failures)
    return _Step(by_sample, _copy(mtp), combined, grads, biases)


def _assert_gradient(grad, target, key):
    grad, target = grad.float(), target.float()
    delta = (grad - target).norm().item()
    bound = 0.02 * target.norm().item() + 2e-8 * target.numel() ** 0.5
    assert torch.isfinite(grad).all() and torch.isfinite(target).all(), key
    assert delta <= bound, f"{key}: gradient L2 error {delta:g}, bound {bound:g}"


def _assert_observations(actual, expected):
    assert actual.keys() == expected.keys()
    source_pairs = {(layer, i) for layer in CORE for i in range(moe.SAMPLES)}
    for phase, kind in ((1, "source-grad"), (2, "read-grad")):
        assert {(n, i) for k, n, p, i in expected if k == kind and p == phase} == source_pairs
    expert_pairs = {(f"decoder.layers.{2 * n - 1}.mlp", i) for n, i in source_pairs}
    for phase in (1, 2):
        assert {(n, i) for k, n, p, i in expected if k == "routes" and p == phase} == expert_pairs
    assert {i for k, _, _, i in expected if k == "mtp-token-loss"} == set(range(moe.SAMPLES))

    for key, baseline in expected.items():
        current = actual[key]
        if key[0] == "routes":
            mask, probs = current
            old_mask, old_probs = baseline
            assert mask.shape == old_mask.shape == (moe.SEQUENCE, moe.EXPERTS), key
            assert (mask.sum(-1) == moe.TOPK).all() and (old_mask.sum(-1) == moe.TOPK).all(), key
            same = mask & old_mask
            agreement = same.count_nonzero().item() / (moe.SEQUENCE * moe.TOPK)
            assert agreement >= 0.995, f"{key}: selected expert overlap {agreement:.5%}"
            torch.testing.assert_close(
                probs[same], old_probs[same], rtol=0.01, atol=0.001, msg=str(key)
            )
        elif key[0] == "mtp-token-loss":
            assert current.numel() == moe.SEQUENCE, key
            torch.testing.assert_close(
                current.float(), baseline.float(), rtol=0.005, atol=0.01, msg=str(key)
            )
        else:
            assert baseline.norm() > 0 and current.norm() > 0, key
            _assert_gradient(current, baseline, key)


def _compare(actual, expected, ep_rank):
    # On PP2, an expert owner has one copy of each weight across its two stages. With /E
    # MTP, the tail also holds the synchronized copy of the input embedding.
    owners = ring._gather_objects((ep_rank, tuple(actual.grads), tuple(actual.biases)))
    counts = Counter(name for ep, names, _ in owners if ep == ep_rank for name in names)
    bias_counts = Counter(name for ep, _, names in owners if ep == ep_rank for name in names)
    wanted = Counter(
        {n: 2 if n == "embedding.word_embeddings.weight" else 1 for n in expected.grads}
    )
    # Complete all collectives even if only one rank's experts or gates disagree.
    failure = None
    try:
        assert counts == wanted, (wanted - counts, counts - wanted)
        assert bias_counts == Counter(expected.biases.keys())
        for sample, baseline in expected.losses.items():
            values = actual.losses[sample]
            assert values.numel() == moe.SEQUENCE
            torch.testing.assert_close(
                values.float(), baseline.float(), rtol=0.005, atol=0.01, msg=f"LM sample {sample}"
            )
        assert expected.mtp.min() > 0
        torch.testing.assert_close(actual.mtp[:1], expected.mtp[:1], rtol=0.002, atol=0)
        torch.testing.assert_close(actual.mtp[1:], expected.mtp[1:], rtol=0, atol=0)
        _assert_observations(actual.observations, expected.observations)
        for name, grad in actual.grads.items():
            _assert_gradient(grad, expected.grads[name], f"EP {ep_rank} {name}")
        for name, bias in actual.biases.items():
            torch.testing.assert_close(bias, expected.biases[name], rtol=0, atol=0, msg=name)
        for gate in (0, 1):
            for name in ("proj.weight", "norm.weight"):
                assert expected.grads[f"decoder.loop_gates.{gate}.{name}"].norm() > 0
    except AssertionError as error:
        failure = f"EP {ep_rank}, PP {ps.get_pipeline_model_parallel_rank()}: {error}"
    failures = [msg for msg in ring._gather_objects(failure) if msg is not None]
    assert not failures, "\n".join(dict.fromkeys(failures))


def _compare_pp1_and_pp2(segments):
    rows = moe.sample_rows()
    with moe.parallel(1):
        model = _pp1_model()
        state = {name: _copy(value) for name, value in model.state_dict().items()}
        ref_owner = ps.get_expert_model_parallel_rank()
        reference = _run([model], [0], _batches(rows, 1), pp=1)
        del model
    with moe.parallel(2):
        ep_rank = ps.get_expert_model_parallel_rank()
        # TP1, CP1, ETP1 and contiguous EP4 use the same local expert owner on both topologies.
        owners = ring._gather_objects((ref_owner, ep_rank))
        assert all(old == new for old, new in owners), owners
        models = moe.models(2, segments=segments)
        stage = ps.get_pipeline_model_parallel_rank()
        offsets = [sum(map(len, segments[: vp * 2 + stage])) for vp in range(3)]
        for model, offset in zip(models, offsets, strict=True):
            model.load_state_dict(
                {name: state[ring._global_name(name, offset)] for name in model.state_dict()},
                strict=True,
            )
        candidate = _run(models, offsets, _batches(rows, 2), pp=2)
        _compare(candidate, reference, ep_rank)


@pytest.mark.skipif(not CUDA or ring.WORLD != moe.SAMPLES, reason="requires eight CUDA ranks")
def test_glm_mok_pp1_ep4_matches_pp2_ep4():
    _compare_pp1_and_pp2(moe.CHUNKS)


@pytest.mark.skipif(not CUDA or ring.WORLD != moe.SAMPLES, reason="requires eight CUDA ranks")
def test_glm_mok_split_tail_ffn_matches_pp1_ep4():
    # Move only the last shared core FFN onto PP1. PP0 computes all five DSA
    # layers and passes ordinary hidden activations to PP1 on each visit.
    _compare_pp1_and_pp2(("D-D-", "D-", "DE" * 4 + "D", "E", "", "DE"))
