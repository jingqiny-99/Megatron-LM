# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Cross-round sparse banks on RSBF's detached sample sweeps and explicit backward items."""

from collections import Counter
from contextlib import ExitStack

import pytest
import torch

from megatron.core.transformer.loop.cross_loop_kv import current_cross_loop_pass
from megatron.core.transformer.loop.program import FixedDepthCriterion, LoopPhase
from megatron.core.transformer.loop.rsbf_executor import _RSBFAttentionPass
from tests.unit_tests.transformer.loop import test_loop_param_store_rsbf as store_test
from tests.unit_tests.transformer.loop import test_loop_rsbf_executor as base
from megatron.core.transformer.loop.rsbf_plan import MovePhase, known_depth_plan
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA, model_parallel_once
from tests.unit_tests.transformer.loop.test_loop_rsbf_plan import _home, _planner
from tests.unit_tests.transformer.loop.test_loop_attn_res_dsv4_gpu import (
    _hadamard_fallback,
    _model,
    full_precision_gemms,
)

hybrid = base.hybrid
deterministic_algorithms = base.deterministic_algorithms


@pytest.mark.parametrize("fill", ["drain", "sample_order"])
@pytest.mark.parametrize("eus,batch,per_eu", [(2, 1, 1), (4, 2, 4), (64, 1, 2)])
def test_fixed_round_bank_ownership(eus, batch, per_eu, fill):
    """Equal fixed depths keep each sample's original sweeps and backward chain on its EU."""
    homes = _home(eus, per_eu, batch)
    plan = known_depth_plan(_planner(eus, homes, fill=fill), [4] * len(homes))
    assert not any(isinstance(step, MovePhase) for step in plan)


def test_detached_bank_gradient_equals_connected_graph():
    """Two interleaved samples own their first bank and accumulate its three consumer adjoints."""
    source = torch.nn.Linear(2, 3, dtype=torch.float64)
    x = torch.tensor([[1.0, 3.0], [2.0, -1.0]], dtype=torch.float64, requires_grad=True)
    bank = [{}, {}]
    seeds = torch.arange(1, 13, dtype=torch.float64).reshape(4, 3)
    saved = [source(x[g]) for g in range(2)]
    expected = torch.autograd.grad(
        sum((value.tanh() * seeds[r]).sum() for r in range(4) for value in saved),
        (x, *source.parameters()),
    )
    for g in range(2):
        with _RSBFAttentionPass(1, bank[g], LoopPhase.SWEEP).activate():
            current_cross_loop_pass().publish(source, "compressed_kv", saved[g])
        assert bank[g][source, "compressed_kv"].grad_fn is None
    for r in range(4, 0, -1):
        for g in (1, 0):
            with _RSBFAttentionPass(r, bank[g], LoopPhase.RECOMPUTE).activate():
                scope = current_cross_loop_pass()
                if r == 1:
                    live = source(x[g])
                    scope.publish(source, "compressed_kv", live)
                else:
                    live = scope.read(source, "compressed_kv")
                (live.tanh() * seeds[r - 1]).sum().backward()
    for tensor, grad in zip((x, *source.parameters()), expected):
        torch.testing.assert_close(tensor.grad, grad)


@pytest.mark.skipif(not CUDA, reason="DSv4 CSA, HCA and MoE require CUDA")
@pytest.mark.parametrize(
    "form,use_store,dtype,reuse",
    [
        ("block", False, torch.bfloat16, False),
        ("block", False, torch.bfloat16, True),
        ("block", False, torch.float32, False),
        ("block", False, torch.float32, True),
        ("nested", False, torch.float32, True),
        ("block", True, torch.bfloat16, True),
    ],
)
def test_rsbf_dsv4_four_rounds_match_standard(
    hybrid, deterministic_algorithms, full_precision_gemms, form, use_store, dtype, reuse
):
    """The full DP graph is the reference for two interleaved fixed-depth samples.

    The FP32 test uses the same native-arithmetic rotation fallback in both arms;
    BF16 uses the compiled rotation. Deterministic gather backward and full-precision
    GEMMs are needed for a gradient comparison (also when reuse is off).
    """
    from megatron.core.transformer.experimental_attention_variant.csa import (
        CompressedSparseAttention,
    )

    kwargs = dict(
        loop_iterations=4,
        loop_reuse_sparse_kv=reuse,
        dsa_indexer_loss_coeff=1e-2,
        dsa_indexer_topk=8,
        gradient_accumulation_fusion=False,
        deterministic_mode=True,
    )
    with model_parallel_once(), ExitStack() as hooks:
        hooks.enter_context(_hadamard_fallback(dtype))
        standard = _model("injection", dtype, **kwargs)
        model = _model(
            "injection",
            dtype,
            **kwargs,
            loop_schedule="rsbf",
            loop_rsbf_item_form=form,
            loop_rsbf_block_stride=2,
        )
        model.load_state_dict(standard.state_dict())
        batches = base._batches(2, seq=256, dev=torch.device("cuda"))
        expected_losses, expected = base._eager(standard, hybrid, batches)
        decoder = model.decoder
        core = decoder.layers[decoder.loop_roles.core_start : decoder.loop_roles.core_end]
        calls = Counter()

        def count_selection(module, args):
            scope = current_cross_loop_pass()
            assert scope.is_source, "consumer sweep or replay reran the CSA selector"
            calls[scope.phase] += 1

        for attn in core.modules():
            if reuse and isinstance(attn, CompressedSparseAttention) and attn.indexer is not None:
                hooks.callback(
                    attn.indexer.linear_wq_b.register_forward_pre_hook(count_selection).remove
                )
        store = None
        if use_store:
            store = store_test._install(
                model,
                store_test._trivial_groups(),
                1,
                "core=lw,edges=lw,emb_head=rsg,core_regather=off",
            )
        executor = base._executor(model, criterion=FixedDepthCriterion(4), check_recompute=True)
        reports = executor.run_step(
            base._next_batch,
            iter(base._microbatches(model, hybrid, batches)),
            len(batches),
            model=model,
        )
        actual = {
            "0." + name: grad
            for name, grad in store_test._local_grads(model, store).items()
            if grad is not None
        }
        assert executor.stats.exit_counts == [0, 0, 0, 2]
        if reuse:
            assert calls[LoopPhase.SWEEP] == calls[LoopPhase.RECOMPUTE] == 2
        base._assert_losses_equal(base._lm_losses(reports), expected_losses)
        assert actual.keys() == expected.keys()
        errors = {}
        for name, ref in expected.items():
            reference = ref.float()
            difference = (actual[name].float() - reference).norm()
            errors[name] = (difference / reference.norm().clamp_min(1e-20)).item()
        tol = 2**-6 if dtype == torch.bfloat16 else 2e-5
        worst = max(errors, key=errors.get)
        assert errors[worst] <= tol, f"{worst}: relative L2 gradient error {errors[worst]:.3e}"
