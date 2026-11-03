# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The routing tape at every loop activation-checkpoint site.

``loop_checkpoint_context_fn`` (``loop_transformer_block.py``) gives each loop checkpoint call a
``RoutingTape`` (``forward_replay.py``): the forward records every router top-k and the recompute
replays it, so a recompute that is not bitwise its forward still routes as the forward did (the
deep probe's ``CheckpointError: Recomputed values ... have different metadata`` under
``loop_recompute`` with learned routing). The sites: the GPT loop's Huginn round and MoR
recurrence (``LoopTransformerBlock``), the hybrid loop's Huginn round and MoR recurrence
(``loop/hybrid_stack.py``) and the attention-residual round (``loop/attn_res_stack.py``, k = 1
and 2).

Each stack has a test-only routed MoE MLP (``RoutedStubMoE``) in every 'E' sublayer (hybrid) or
layer (GPT): a softmax top-2 of 4 experts through ``moe_utils.topk_routing_with_score_function``
(the tape's hook), each expert on its own rows, so the saved tensors' shapes follow the routing.
Per site:

* exact recompute: ``loop_recompute`` is bitwise the run without it, and every router call of a
  round's recompute replayed the tape its forward recorded (one call each);
* perturbed recompute (the router input of a replay gets noise, a stand-in for the non-bitwise
  recompute of a learned-routing round): with the tape the backward runs and every gradient,
  the routers' included, is finite; the control, the site's checkpoint context without the tape
  (patched in the module that owns the site), raises the deep probe's ``CheckpointError``.

Local layers, fp32: CPU (gloo, one rank) or CUDA; model parallel is initialized once per module.
"""

from dataclasses import replace

import pytest
import torch
from torch.utils.checkpoint import CheckpointError

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import loop_transformer_block
from megatron.core.transformer.forward_replay import in_forward_replay, routing_tape_active
from megatron.core.transformer.loop import attn_res_stack, hybrid_stack
from megatron.core.transformer.loop.attn_res_stack import LoopAttnResHybridStack
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.loop_transformer_block import LoopTransformerBlock, recompute_context
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    BATCH,
    HIDDEN,
    SEQ,
    build_local_loop_stack,
    dense_config,
    device,
    local_gpt_layer_spec,
    local_hybrid_submodules,
    model_parallel_once,
    rope,
)

EXPERTS, TOPK = 4, 2
STUB_MOE = dict(num_moe_experts=EXPERTS, moe_ffn_hidden_size=128, moe_router_topk=TOPK)
ROUNDS = 3


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


class RoutedStubMoE(MegatronModule):
    """A test-only MoE MLP: a bias-free router, a softmax top-``TOPK`` of ``EXPERTS`` through
    ``topk_routing_with_score_function`` and one linear expert per expert, run on its rows
    (``index_add`` of the routed rows, as a dropless MoE layer's dispatch: the saved tensors'
    shapes follow the routing). With ``perturb_replay`` the router input of an activation
    replay gets seeded noise. ``calls`` logs ``(in_forward_replay(), routing_tape_active())``
    of every forward call. Built by ``TransformerLayer`` as its ``mlp`` submodule (the
    keywords ``TransformerLayer`` passes are accepted); returns ``(output, None)``."""

    perturb_replay = False
    calls: list = []

    def __init__(self, config, pg_collection=None, is_mtp_layer=False, name=None, **kwargs):
        del pg_collection, is_mtp_layer, name, kwargs
        super().__init__(config=config)
        hidden, dtype = config.hidden_size, config.params_dtype
        self.router = torch.nn.Linear(hidden, EXPERTS, bias=False, dtype=dtype)
        self.experts = torch.nn.ModuleList(
            torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype) for _ in range(EXPERTS)
        )

    def forward(self, hidden_states, padding_mask=None, **kwargs):
        """Route, run each expert on its rows, combine with the top-k probabilities."""
        del padding_mask, kwargs
        RoutedStubMoE.calls.append((in_forward_replay(), routing_tape_active()))
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_input = x
        if RoutedStubMoE.perturb_replay and in_forward_replay():
            noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(9))
            router_input = x + noise.to(x) * x.detach().std()
        probs, indices = topk_routing_with_score_function(
            self.router(router_input), TOPK, score_function="softmax", dense_output=True
        )
        out = torch.zeros_like(x)
        for expert, linear in enumerate(self.experts):
            rows, slot = (indices == expert).nonzero(as_tuple=True)
            out = out.index_add(0, rows, linear(x[rows]) * probs[rows, slot, None])
        return out.view_as(hidden_states), None


def _routed_moe_layer_spec():
    """The hybrid 'E' sublayer: pre-MLP RMSNorm + ``RoutedStubMoE`` + bda."""
    local = get_gpt_layer_local_submodules(normalization="RMSNorm")
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            pre_mlp_layernorm=local.pre_mlp_layernorm, mlp=RoutedStubMoE, mlp_bda=local.mlp_bda
        ),
    )


def _hybrid(module, recompute, **fields):
    """'*E*E*E*E' (prelude '*E', core '*E*E', coda '*E'), a routed stub in every 'E'."""
    submodules = local_hybrid_submodules()
    submodules.moe_layer = _routed_moe_layer_spec()
    return build_local_loop_stack(
        "*E*E*E*E",
        2,
        2,
        module=module,
        submodules=submodules,
        loop_iterations=ROUNDS,
        loop_recompute=recompute,
        **STUB_MOE,
        **fields,
    )


def _gpt(recompute, **fields):
    """Four local GPT layers with the routed stub as MLP (prelude 1, core 2, coda 1)."""
    spec = local_gpt_layer_spec()
    spec = replace(spec, submodules=replace(spec.submodules, mlp=RoutedStubMoE))
    loop = dict(loop_architecture="huginn", loop_iterations=ROUNDS, loop_recompute=recompute)
    loop.update(fields)
    config = dense_config(4, loop_prelude_layers=1, loop_core_layers=2, **loop)
    torch.manual_seed(1234)
    return LoopTransformerBlock(
        config=config,
        spec=spec,
        pre_process=True,
        post_process=True,
        post_layer_norm=False,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    ).to(device())


_MOR = dict(loop_architecture="mor", loop_token_execution="masked")
_ATTN_RES = dict(loop_carrier="attn_res")

# site: (the stack builder, given loop_recompute; the module whose loop_checkpoint_context_fn
# the site calls)
SITES = {
    "gpt_huginn": (lambda recompute: _gpt(recompute), loop_transformer_block),
    "gpt_mor": (lambda recompute: _gpt(recompute, **_MOR), loop_transformer_block),
    "hybrid_huginn": (lambda recompute: _hybrid(LoopHybridStack, recompute), hybrid_stack),
    "hybrid_mor": (lambda recompute: _hybrid(LoopHybridStack, recompute, **_MOR), hybrid_stack),
    "attn_res_k1": (
        lambda recompute: _hybrid(LoopAttnResHybridStack, recompute, **_ATTN_RES),
        attn_res_stack,
    ),
    "attn_res_k2": (
        lambda recompute: _hybrid(
            LoopAttnResHybridStack, recompute, loop_attn_res_blocks_per_round=2, **_ATTN_RES
        ),
        attn_res_stack,
    ),
}


def _untaped(prefix):
    """The sites' checkpoint context before the routing tape (the control)."""
    return lambda: (
        torch.profiler.record_function(f"{prefix}/phase=forward"),
        recompute_context(f"{prefix}/phase=recompute"),
    )


@pytest.fixture(autouse=True)
def _stub_state(monkeypatch):
    monkeypatch.setattr(RoutedStubMoE, "perturb_replay", False)
    monkeypatch.setattr(RoutedStubMoE, "calls", [])


def _run(stack):
    """Forward + backward of ``sum(y * upstream) + loop losses``: ``(y, dx, grads)``."""
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(SEQ, BATCH, HIDDEN, generator=generator).to(device()).requires_grad_(True)
    upstream = torch.randn(SEQ, BATCH, HIDDEN, generator=generator).to(device())
    y = stack(x, attention_mask=None, rotary_pos_emb=rope())
    ((y * upstream).sum() + sum(stack.loop_losses.values())).backward()
    grads = {name: p.grad for name, p in stack.named_parameters() if p.grad is not None}
    return y.detach(), x.grad, grads


@pytest.mark.parametrize("site", list(SITES))
def test_exact_recompute_is_bitwise_and_replays_the_tape(site):
    build, _ = SITES[site]
    reference = build(False)
    stack = build(True)
    stack.load_state_dict(reference.state_dict())
    y_ref, dx_ref, grads_ref = _run(reference)
    assert RoutedStubMoE.calls and not any(map(any, RoutedStubMoE.calls))
    RoutedStubMoE.calls = []
    y, dx, grads = _run(stack)
    assert torch.equal(y, y_ref) and torch.equal(dx, dx_ref)
    assert grads.keys() == grads_ref.keys() and any(
        name.endswith("router.weight") for name in grads
    )
    for name, grad in grads_ref.items():
        assert torch.equal(grads[name], grad), name
    calls = RoutedStubMoE.calls
    replays = [taped for replay, taped in calls if replay]
    recorded = calls.count((False, True))
    # Every core call runs once under the forward's tape and once in its replay (3 rounds).
    assert replays and all(replays), calls
    assert len(replays) == recorded and recorded % ROUNDS == 0, calls


@pytest.mark.parametrize("taped", [True, False], ids=["taped", "control"])
@pytest.mark.parametrize("site", list(SITES))
def test_perturbed_recompute_keeps_the_routing(monkeypatch, site, taped):
    build, owner = SITES[site]
    if not taped:
        monkeypatch.setattr(owner, "loop_checkpoint_context_fn", _untaped)
    stack = build(True)
    RoutedStubMoE.perturb_replay = True
    if not taped:
        with pytest.raises(CheckpointError):
            _run(stack)
        return
    _, dx, grads = _run(stack)
    assert any(replay for replay, _ in RoutedStubMoE.calls)
    assert torch.isfinite(dx).all()
    assert any(name.endswith("router.weight") for name in grads)
    assert all(torch.isfinite(grad).all() for grad in grads.values())
