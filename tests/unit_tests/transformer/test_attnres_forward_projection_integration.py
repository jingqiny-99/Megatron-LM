# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Forward-only metadata completion, transport identity and graph boundary contracts."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.attention_residual import (
    AttentionResidual,
    AttnResForwardProjectionStage,
)
from megatron.core.transformer.attention_residual_forward_state import (
    ForwardProjectionPlan,
    ForwardProjectionRow,
)
from megatron.core.transformer.transformer_block import TransformerBlock


def _plan(fraction=1.0):
    return ForwardProjectionPlan(16, 1024, 3, 2, 2, fraction, 1e-6)


def _module(consumer, queries, norms):
    return SimpleNamespace(
        forward_projection_consumer_id=consumer,
        pseudo_query=queries[consumer],
        key_norm_weight=norms[consumer],
    )


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA projection kernel")
def test_four_stage_full_prefix_and_end_of_chunk_completion(fraction):
    from fla.ops.attnres.fused import FusedAttnresFunction

    from megatron.core.transformer.attention_residual_forward_projection import (
        aggregate_with_forward_cache,
    )

    torch.manual_seed(81)
    plan = _plan(fraction)
    queries = (torch.randn(33, 1024, device="cuda") * 0.02).requires_grad_()
    norms = (torch.randn(33, 1024, device="cuda") * 0.1 + 1).requires_grad_()
    # Mirrors intentionally have different pointers from canonical parameters.
    query_bank, gamma_bank = queries.detach().clone(), norms.detach().clone()
    sources = tuple(
        torch.randn(16, 1, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(6)
    )
    partial = torch.randn_like(sources[0], requires_grad=True)
    payload = None
    for stage_id in range(4):
        first = plan.layer_offset(stage_id)
        last = plan.layer_offset(stage_id + 1)
        ids = tuple(range(2 * first, 2 * last + int(stage_id == 3)))
        state = AttnResForwardProjectionStage(
            plan,
            stage_id,
            query_bank.clone(),
            gamma_bank.clone(),
            None if payload is None else payload.clone(),
            16,
            1,
            ids,
        )
        if stage_id == 0:
            state.complete_source(0, sources[0])
        for layer in range(first + 1, last + 1):
            # The real graph performs this after the dense layer, before its next
            # source registration or a PP send. No fake value-gradient edge exists.
            if layer % 3 == 0 and layer < 16:
                state.after_layer(layer, sources[layer // 3])
        consumer = 2 * last - 1
        historical = (last - 1) // 3 + 1
        values = (*sources[:historical], partial)
        module = _module(consumer, queries, norms)
        caches = state.caches_for(module, values)
        output = aggregate_with_forward_cache(
            module.pseudo_query, module.key_norm_weight, values, caches
        )
        expected = FusedAttnresFunction.apply(
            module.pseudo_query, module.key_norm_weight, None, 1e-6, 1.0, False, 1, *values
        )[0]
        torch.testing.assert_close(output, expected, atol=0.016, rtol=0.01)
        relative = (output.double() - expected.double()).norm() / expected.double().norm()
        assert relative < 0.003
        assert all(row.requires_grad is False for row in state.rows.values())
        if stage_id < 3:
            payload = state.pack_exit()
            assert payload.shape == plan.payload_shape(16, 1) and not payload.requires_grad
        if stage_id == 2 and fraction:
            # L12's completed b4 is still the outgoing partial, not a registered
            # historical source. Its selected L13+ metadata must already cross PP.
            assert ForwardProjectionRow(4, None) in state.rows
            assert ForwardProjectionRow(4, None) in plan.entry_rows(3)
        if caches:
            for value, cache in zip(values, caches):
                if cache is not None:
                    assert cache.source is value
                    assert cache.query.data_ptr() == module.pseudo_query.data_ptr()
                    assert cache.query.data_ptr() != query_bank[consumer].data_ptr()


def test_pure_state_rejects_unknown_consumer_and_missing_selected_row():
    plan = _plan()
    banks = torch.zeros(33, 1024)
    state = AttnResForwardProjectionStage(plan, 0, banks, banks.clone(), None, 2, 1, (0,))
    values = (torch.zeros(2, 1, 1024, dtype=torch.bfloat16),)
    with pytest.raises(ValueError, match="authorize"):
        state.caches_for(_module(1, banks, banks), values)
    with pytest.raises(RuntimeError, match="missing"):
        state.caches_for(_module(0, banks, banks), values)


def test_host_graph_boundary_rejects_unprepared_and_foreign_snapshot():
    block = TransformerBlock.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.config = SimpleNamespace(attn_res_forward_projection=True)
    block.pre_process = True
    with pytest.raises(ValueError, match="prepared"):
        block._forward_projection_host_args(None, None)
    called = []

    def reject(epoch, plan):
        called.append((epoch, plan))
        raise RuntimeError("foreign snapshot")

    snapshot = SimpleNamespace(epoch=7, plan="plan", validate=reject)
    with pytest.raises(RuntimeError, match="foreign"):
        block._forward_projection_host_args(snapshot, None)
    assert called == [(7, "plan")]


def test_host_graph_boundary_authorizes_canonical_modules_and_explicit_banks():
    block = TransformerBlock.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.config = SimpleNamespace(attn_res_forward_projection=True)
    block.pre_process = False
    module = AttentionResidual.__new__(AttentionResidual)
    torch.nn.Module.__init__(module)
    module.forward_projection_consumer_id = 8
    block.aggregator = module
    calls = []
    snapshot = SimpleNamespace(
        epoch=3,
        plan="static",
        query_bank=torch.zeros(33, 1024),
        gamma_bank=torch.ones(33, 1024),
        validate=lambda epoch, plan: calls.append((epoch, plan)),
        validate_module=lambda cid, value: calls.append((cid, value)),
    )
    payload = torch.zeros(4, 1, 1)
    actual = block._forward_projection_host_args(snapshot, payload)
    assert calls == [(3, "static"), (8, module)]
    assert actual["forward_query_bank"] is snapshot.query_bank
    assert actual["forward_gamma_bank"] is snapshot.gamma_bank
    assert actual["forward_payload"] is payload
    with pytest.raises(ValueError, match="nondifferentiable"):
        block._forward_projection_host_args(snapshot, payload.requires_grad_())


def test_forward_projection_keeps_original_one_source_identity():
    module = AttentionResidual.__new__(AttentionResidual)
    torch.nn.Module.__init__(module)
    module.pseudo_query = torch.nn.Parameter(torch.zeros(1024))
    module.key_norm_weight = torch.nn.Parameter(torch.ones(1024))
    source = torch.randn(3, 1, 1024, dtype=torch.bfloat16, requires_grad=True)

    # The state must never be consulted for an identically one softmax.
    def forbidden(*args):
        raise AssertionError("one-source identity invoked projection machinery")

    state = SimpleNamespace(caches_for=forbidden)
    output = module((source,), forward_projection_state=state)
    gradient = torch.randn_like(output)
    output.backward(gradient)
    assert torch.equal(source.grad, gradient)
    assert torch.count_nonzero(module.pseudo_query.grad) == 0
    assert torch.count_nonzero(module.key_norm_weight.grad) == 0
