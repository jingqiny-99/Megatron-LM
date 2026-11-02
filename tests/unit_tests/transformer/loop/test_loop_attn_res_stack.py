# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The Kimi-K3-style attention-residual carrier of the hybrid loop (``LoopAttnResHybridStack``,
``loop_carrier='attn_res'``) and its ``w = 0`` reference, the residual carrier
(``loop_carrier='residual'``).

Local dense layers (and the test-only 'E' MLP stub) on CPU (gloo, one rank) or CUDA, mostly in
float64 (the local layers run in float64; S9.0 probe), so the reference comparisons are tight:

* the layer-delta path (``return_layer_delta``) equals ``layer(h) - h``;
* the ``w = 0`` oracle: the carrier equals the residual carrier up to the RMSNorm eps
  (eps 1e-12; a control at eps 1e-2 differs by orders of magnitude more);
* random ``w``: outputs, every gradient and the per-call source counts equal a naive
  plain-autograd composition (``attn_res_reference.reference_k3_loop``), k = 1 and 2;
* per-row exit: a row equals itself run alone at its depth, its history stops growing (its
  later sources get weight exactly 0), its skipped rounds get exactly-zero gradient, and those
  rounds mark it as padding for the router and the DSv4 indexer-loss scope;
* loop recompute is bitwise and the diagnostics count each call once; guards; FLOPs;
  the residual carrier against a hand-written repeated core; diagnostics.
"""

import contextlib
import dataclasses
import re

import pytest
import torch

from megatron.core.models.hybrid.hybrid_block import HybridStackLayerContext
from megatron.core.tensor_observation import capture_tensor_observations
from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
    indexer_loss_padding_mask_in_scope,
)
from megatron.core.transformer.loop import guards
from megatron.core.transformer.loop.attn_res import AttnResStats
from megatron.core.transformer.loop.attn_res_stack import LoopAttnResHybridStack
from megatron.core.transformer.loop.hybrid_specs import (
    loop_attn_res_hybrid_dsv4_stack_spec,
    loop_attn_res_hybrid_stack_spec,
)
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.loop.reporting import loop_metric_report
from tests.unit_tests.transformer.loop import hybrid_loop_utils
from tests.unit_tests.transformer.loop.attn_res_reference import reference_k3_loop
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    BATCH,
    HIDDEN,
    SEQ,
    build_local_loop_stack,
    dense_config,
    device,
    hybrid_loop_config,
    local_hybrid_submodules,
    model_parallel_once,
    rope,
)
from tests.unit_tests.transformer.loop.multi_eu_utils import stub_moe_layer_spec

mhc_on_this_device = hybrid_loop_utils.mhc_on_this_device

DENSE = "*-*-*-*-"  # prelude '*-', core '*-*-' (k = 2 cuts it at 2), coda '*-'
F64 = dict(params_dtype=torch.float64)
STUB_MOE = dict(num_moe_experts=4, moe_ffn_hidden_size=128, moe_router_topk=2)


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


def _k3(k=1, pattern=DENSE, prelude=2, coda=2, **overrides):
    fields = dict(F64, loop_carrier="attn_res", loop_attn_res_blocks_per_round=k)
    fields.update(overrides)
    return build_local_loop_stack(pattern, prelude, coda, module=LoopAttnResHybridStack, **fields)


def _residual(pattern=DENSE, prelude=2, coda=2, **overrides):
    fields = dict(F64, loop_carrier="residual")
    fields.update(overrides)
    return build_local_loop_stack(pattern, prelude, coda, **fields)


def _inputs(batch=BATCH, seed=7, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(SEQ, batch, HIDDEN, generator=generator, dtype=torch.float64)
    upstream = torch.randn(SEQ, batch, HIDDEN, generator=generator, dtype=torch.float64)
    return x.to(dtype).to(device()), upstream.to(dtype).to(device())


def _rope(dtype=torch.float64):
    return rope().to(dtype)


def _forward(stack, x):
    return stack(x, attention_mask=None, rotary_pos_emb=_rope(x.dtype))


def _rms_norm(y, eps=1e-12):
    return y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)


def _forward_backward(stack, x, upstream, run=None, transform=None):
    """Forward (``run`` or the stack) + backward of ``sum(transform(y) * upstream)``."""
    x = x.detach().clone().requires_grad_(True)
    stack.zero_grad(set_to_none=True)
    y = (run or _forward)(stack, x)
    out = y if transform is None else transform(y)
    (out * upstream).sum().backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in stack.named_parameters()
        if param.grad is not None
    }
    return out.detach().clone(), x.grad.detach().clone(), grads


def _assert_bitwise(actual, expected):
    y, dx, grads = actual
    y_ref, dx_ref, grads_ref = expected
    assert torch.equal(y, y_ref)
    assert torch.equal(dx, dx_ref)
    assert grads.keys() == grads_ref.keys()
    for name, grad in grads_ref.items():
        assert torch.equal(grads[name], grad), name


def _assert_close(actual, expected, rtol, atol, names=None):
    y, dx, grads = actual
    y_ref, dx_ref, grads_ref = expected
    torch.testing.assert_close(y, y_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(dx, dx_ref, rtol=rtol, atol=atol)
    for name in names if names is not None else grads_ref:
        torch.testing.assert_close(grads[name], grads_ref[name], rtol=rtol, atol=atol, msg=name)


def _randomize_carrier(stack, seed=11):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in [*stack.loop_attn_res, stack.loop_attn_res_head]:
            module.pseudo_query.copy_(torch.randn(HIDDEN, generator=generator))
            module.key_norm_weight.copy_(1 + 0.2 * torch.randn(HIDDEN, generator=generator))


def _with_depth(stack, depth, fn):
    saved = stack.config.loop_iterations
    stack.config.loop_iterations = depth
    try:
        return fn()
    finally:
        stack.config.loop_iterations = saved


class RecordingSink:
    """A diagnostics sink keeping every call's key and masked weights (cloned)."""

    def __init__(self):
        self.calls = []

    def add(self, key, alpha, row_weight):
        self.calls.append((key, alpha.detach().clone()))


# ---------------------------------------------------------------------------
# Construction, keys, specs, guards
# ---------------------------------------------------------------------------


def test_carrier_parameters_are_children_and_fp32():
    stack = _k3(params_dtype=torch.float32)
    roles = stack.loop_roles
    assert len(stack.loop_attn_res) == roles.core + roles.coda
    names = {name for name, _ in stack.named_parameters()}
    assert {"loop_attn_res.0.pseudo_query", "loop_attn_res.5.key_norm_weight"} <= names
    assert {"loop_attn_res_head.pseudo_query", "loop_attn_res_head.key_norm_weight"} <= names
    assert not any(name.startswith(("loop_adapter", "loop_norm")) for name in names)
    for module in [*stack.loop_attn_res, stack.loop_attn_res_head]:
        assert module.pseudo_query.dtype == torch.float32
        assert module.pseudo_query.keep_in_fp32 and module.key_norm_weight.keep_in_fp32
    residual = _residual()
    assert not any(name.startswith("loop_") for name, _ in residual.named_parameters())


def test_specs_select_the_attn_res_stack(monkeypatch):
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.transformer.loop import hybrid_specs

    assert loop_attn_res_hybrid_stack_spec.module is LoopAttnResHybridStack
    assert loop_attn_res_hybrid_stack_spec.submodules is hybrid_stack_spec.submodules
    # The DSv4 factory needs an MLA config; only the module swap is under test here.
    monkeypatch.setattr(hybrid_specs, "hybrid_dsv4_stack_spec", lambda config: hybrid_stack_spec)
    spec = loop_attn_res_hybrid_dsv4_stack_spec(None)
    assert spec.module is LoopAttnResHybridStack
    assert spec.submodules is hybrid_stack_spec.submodules


def test_stack_class_must_match_the_carrier():
    """T-G1: CARRIER_STACK_MISMATCH both ways."""
    with pytest.raises(ValueError, match=re.escape(guards.CARRIER_STACK_MISMATCH)):
        build_local_loop_stack(DENSE, 2, 2, loop_carrier="attn_res")
    for carrier in ("injection", "residual"):
        with pytest.raises(ValueError, match=re.escape(guards.CARRIER_STACK_MISMATCH)):
            build_local_loop_stack(DENSE, 2, 2, module=LoopAttnResHybridStack, loop_carrier=carrier)


def test_blocks_must_cut_the_core_on_block_boundaries():
    """T-G1: ATTN_RES_BLOCKS (the core '*-*-' has 4 sublayers)."""
    for k in (3, 4):
        with pytest.raises(ValueError, match=re.escape(guards.ATTN_RES_BLOCKS)):
            _k3(k=k)
    assert _k3(k=2)._attn_res_offsets == (0, 2)


@pytest.mark.parametrize(
    "overrides,match",
    [
        (dict(loop_bptt_iterations=1), guards.ATTN_RES_TRUNCATED_BPTT),
        (dict(fp32_residual_connection=True), guards.ATTN_RES_FP32_RESIDUAL),
        (dict(mtp_num_layers=1), guards.ATTN_RES_WITH_MTP),
        (dict(enable_mhc_connections=True, mhc_num_residual_streams=4), guards.ATTN_RES_WITH_MHC),
        (dict(cuda_graph_impl="local"), guards.CUDA_GRAPHS_ATTN_RES),
        (dict(loop_architecture="mor"), "needs a hybrid Huginn loop"),
        (dict(loop_schedule="rsbf", calculate_per_token_loss=True), guards.RSBF_CARRIER),
    ],
)
def test_attn_res_guards(overrides, match):
    """T-G1: config and stack rules of the attention-residual carrier."""
    fields = dict(loop_carrier="attn_res")
    fields.update(overrides)
    with pytest.raises(ValueError, match=re.escape(match)):
        stack = build_local_loop_stack(DENSE, 2, 2, module=LoopAttnResHybridStack, **fields)
        del stack


def test_rsbf_refuses_the_residual_carrier():
    """T-G1: RSBF_CARRIER also names the residual carrier (not only 'attn_res')."""
    with pytest.raises(ValueError, match=re.escape(guards.RSBF_CARRIER)):
        build_local_loop_stack(
            DENSE,
            2,
            2,
            loop_carrier="residual",
            loop_schedule="rsbf",
            calculate_per_token_loss=True,
        )


def test_carrier_config_rules():
    """T-G1: the residual carrier refuses mHC; the carriers need a hybrid Huginn loop; k needs
    'attn_res'."""
    with pytest.raises(ValueError, match=re.escape(guards.RESIDUAL_WITH_MHC)):
        hybrid_loop_config(
            DENSE,
            2,
            2,
            loop_carrier="residual",
            enable_mhc_connections=True,
            mhc_num_residual_streams=4,
        )
    with pytest.raises(ValueError, match="require a hybrid loop"):
        dense_config(8, loop_carrier="attn_res")
    with pytest.raises(ValueError, match="needs a hybrid Huginn loop"):
        dense_config(8, loop_architecture="huginn", loop_carrier="residual")
    with pytest.raises(ValueError, match="loop_attn_res_blocks_per_round"):
        hybrid_loop_config(DENSE, 2, 2, loop_attn_res_blocks_per_round=2)
    with pytest.raises(ValueError, match="loop_attn_res_blocks_per_round"):
        hybrid_loop_config(DENSE, 2, 2, loop_carrier="attn_res", loop_attn_res_blocks_per_round=0)
    # The default stays the injection carrier; loop_sample_depth and recompute are accepted.
    config = hybrid_loop_config(DENSE, 2, 2)
    assert config.loop_carrier == "injection" and config.loop_attn_res_blocks_per_round == 1
    config = hybrid_loop_config(
        DENSE, 2, 2, loop_carrier="attn_res", loop_sample_depth=True, loop_recompute=True
    )
    guards.validate_loop_carrier(config)


def test_store_hook_is_refused():
    class _Hook:
        windows = ()

        def visit(self, *args):
            return contextlib.nullcontext()

    stack = _k3()
    stack._loop_store_hook = _Hook()
    x, _ = _inputs()
    with pytest.raises(RuntimeError, match="parameter store"):
        _forward(stack, x)


# ---------------------------------------------------------------------------
# T-D1: the layer-delta path
# ---------------------------------------------------------------------------


def _ctx(stack, x):
    _, ctx = stack._prepare_forward(
        x,
        None,
        None,
        _rope(x.dtype),
        packed_seq_params=None,
        padding_mask=None,
        input_ids=None,
        packed_seq_params_by_layout=None,
        cp_layout_plan=None,
    )
    return ctx


def test_layer_delta_is_the_layer_output_minus_its_input():
    """T-D1 for attention, MLP and the 'E' stub sublayers (float64)."""
    submodules = local_hybrid_submodules()
    submodules.moe_layer = stub_moe_layer_spec()
    pattern = "*E*-*E*-"
    stack = build_local_loop_stack(
        pattern,
        2,
        2,
        module=LoopAttnResHybridStack,
        submodules=submodules,
        loop_carrier="attn_res",
        **F64,
        **STUB_MOE,
    )
    x, _ = _inputs()
    ctx = _ctx(stack, x)
    delta_ctx = dataclasses.replace(ctx, return_layer_delta=True)
    for index in range(len(stack.layers)):
        h = x * (index + 1)
        full = stack._run_layer(index, h, ctx)
        delta = stack._run_layer(index, h, delta_ctx)
        torch.testing.assert_close(delta, full - h, rtol=1e-12, atol=1e-12, msg=str(index))


def test_layer_delta_rejects_other_layer_kinds():
    """T-D1: a non-TransformerLayer sublayer cannot return a delta."""
    stack = _k3()
    x, _ = _inputs()
    ctx = dataclasses.replace(_ctx(stack, x), return_layer_delta=True)
    stack.layers[0] = torch.nn.Identity()
    with pytest.raises(TypeError, match="plain TransformerLayer"):
        stack._run_layer(0, x, ctx)


def test_layer_delta_rejects_mhc_wrappers(mhc_on_this_device):
    stack = build_local_loop_stack(
        DENSE,
        2,
        2,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=5,
    )
    x = torch.randn(SEQ, BATCH, 4 * HIDDEN, device=device())
    ctx = dataclasses.replace(_ctx(stack, x), return_layer_delta=True)
    with pytest.raises(TypeError, match="HyperConnectionHybridLayer"):
        stack._run_layer(0, x, ctx)


class _FakeMoE(torch.nn.Module):
    """A MoE sublayer stub whose MLP returns one recorded update tensor per call."""

    is_moe_layer = True

    def __init__(self):
        super().__init__()
        self.updates = []

    def mlp(self, x):
        update = x + 2.0
        self.updates.append(update)
        return update, None


def test_empty_input_rule_in_delta_mode():
    """T-D1: on an empty input a MoE sublayer returns its update itself in delta mode and a new
    ``x + update`` otherwise; any other sublayer returns a new zero delta / its input. Shapes
    are empty, so the checks are on identity, call counts and the residual add."""
    stack = _k3()
    x, _ = _inputs()
    ctx = _ctx(stack, x)
    delta_ctx = dataclasses.replace(ctx, return_layer_delta=True)
    empty = torch.zeros(0, BATCH, HIDDEN, dtype=torch.float64, device=device())
    moe = _FakeMoE()
    stack.layers[1] = moe
    delta = stack._run_sublayer(1, empty, delta_ctx, "t")
    assert len(moe.updates) == 1 and delta is moe.updates[0]
    full = stack._run_sublayer(1, empty, ctx, "t")
    assert len(moe.updates) == 2
    assert full is not moe.updates[1] and full is not empty
    assert full.shape == empty.shape and full.dtype == empty.dtype
    zero = stack._run_sublayer(0, empty, delta_ctx, "t")
    assert zero is not empty and zero.shape == empty.shape and zero.dtype == empty.dtype
    assert stack._run_sublayer(0, empty, ctx, "t") is empty


# ---------------------------------------------------------------------------
# T-O1 / T-O2: the w = 0 oracle and the naive reference
# ---------------------------------------------------------------------------


def _carrier_free_state(stack):
    tensors = {**dict(stack.named_parameters()), **dict(stack.named_buffers())}
    return {name: t for name, t in tensors.items() if not name.startswith("loop_attn_res")}


@pytest.mark.parametrize("rounds", [1, 2, 3])
@pytest.mark.parametrize("k", [1, 2])
def test_zero_query_matches_the_residual_carrier(k, rounds):
    """T-O1: at w = 0 and eps 1e-12 the carrier equals the residual carrier after a final
    RMSNorm (output, input gradient and every layer's parameter gradients); at eps 1e-2 the
    difference is at least 100x larger (the eps dependence of the oracle)."""
    errors = {}
    for eps in (1e-12, 1e-2):
        k3 = _k3(k=k, loop_iterations=rounds, layernorm_epsilon=eps)
        residual = _residual(loop_iterations=rounds, layernorm_epsilon=eps)
        reference_state = _carrier_free_state(k3)
        residual_state = _carrier_free_state(residual)
        assert reference_state.keys() == residual_state.keys()
        for name, value in residual_state.items():
            assert torch.equal(value, reference_state[name]), name
        x, upstream = _inputs()
        actual = _forward_backward(k3, x, upstream, transform=_rms_norm)
        expected = _forward_backward(residual, x, upstream, transform=_rms_norm)
        assert set(actual[2]) - set(expected[2]) <= {
            name for name in actual[2] if name.startswith("loop_attn_res")
        }
        if eps == 1e-12:
            _assert_close(actual, expected, rtol=1e-8, atol=1e-8, names=list(expected[2]))
        errors[eps] = (actual[0] - expected[0]).abs().max().item()
    assert errors[1e-2] >= 100 * errors[1e-12]
    assert errors[1e-2] > 1e-6


@pytest.mark.parametrize("depths", [None, (1, 3), (2, 2)])
@pytest.mark.parametrize("k", [1, 2])
def test_random_queries_match_the_naive_reference(k, depths):
    """T-O2: random w and key-norm weights; output, input gradient, every parameter gradient
    (the carrier's included) and the per-call source counts equal the naive composition."""
    stack = _k3(k=k, loop_iterations=3)
    _randomize_carrier(stack)
    x, upstream = _inputs()
    sink = RecordingSink()
    stack.enable_carrier_diagnostics(True, sink)

    def run(module, hidden):
        if depths is not None:
            module.set_loop_depths(torch.tensor(depths, device=device()))
        return _forward(module, hidden)

    actual = _forward_backward(stack, x, upstream, run=run)
    counts = [alpha.shape[0] for _, alpha in sink.calls]
    stack.enable_carrier_diagnostics(False)
    reference_counts = []

    def reference(module, hidden):
        y, calls = reference_k3_loop(module, hidden, _rope(), depths)
        reference_counts.extend(calls)
        return y

    expected = _forward_backward(stack, x, upstream, run=reference)
    _assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert counts == reference_counts
    assert counts == _source_counts(stack, k, 3 if depths is None else max(depths))
    assert all(grad.abs().sum() > 0 for name, grad in actual[2].items() if "pseudo_query" in name)


def _source_counts(stack, k, rounds):
    """The per-call source counts of the design (2.2): round r, block beta:
    1 + k(r - 1) + (beta - 1) at the block start, one more after it; the coda 1 + kR (+1 after
    its first sublayer); the head 1 + kR + 1."""
    block = stack.loop_roles.core // k
    formula = []
    for r in range(1, rounds + 1):
        for offset in range(stack.loop_roles.core):
            beta = offset // block + 1
            formula.append(1 + k * (r - 1) + (beta - 1) + (offset % block > 0))
    for offset in range(stack.loop_roles.coda):
        formula.append(1 + k * rounds + (offset > 0))
    formula.append(1 + k * rounds + 1)
    return formula


@pytest.mark.parametrize("k", [1, 2])
def test_sampled_depth_runs_the_sampled_round_count(k):
    """``loop_sample_depth``: a training forward + backward runs the sampled R (which may differ
    from ``loop_iterations``) with the 2.2 source counts, and equals the uniform-depth run at
    that R bitwise."""
    stack = _k3(k=k, loop_iterations=2, loop_sample_depth=True)
    _randomize_carrier(stack)
    x, upstream = _inputs()
    seen_rounds = set()
    for seed in (0, 1, 2, 3, 4, 5):
        sink = RecordingSink()
        stack.enable_carrier_diagnostics(True, sink)
        torch.manual_seed(seed)
        sampled = _forward_backward(stack, x, upstream)
        stack.enable_carrier_diagnostics(False)
        rounds = sink.calls[-1][0][2]  # the head's key carries the forward's round count
        assert rounds in (1, 2, 4, 8)
        assert [alpha.shape[0] for _, alpha in sink.calls] == _source_counts(stack, k, rounds)
        assert torch.isfinite(sampled[0]).all() and torch.isfinite(sampled[1]).all()
        stack.config.loop_sample_depth = False
        try:
            uniform = _with_depth(stack, rounds, lambda: _forward_backward(stack, x, upstream))
        finally:
            stack.config.loop_sample_depth = True
        _assert_bitwise(sampled, uniform)
        seen_rounds.add(rounds)
    assert len(seen_rounds) > 1


# ---------------------------------------------------------------------------
# Per-row exit: T-E1, T-E2, T-E3, T-E4
# ---------------------------------------------------------------------------


def _rows(stack, x, upstream, depths, **kwargs):
    def run(module, hidden):
        module.set_loop_depths(torch.tensor(depths, device=device()))
        return _forward(module, hidden)

    return _forward_backward(stack, x, upstream, run=run, **kwargs)


@pytest.mark.parametrize("k,depths", [(1, (1, 3)), (2, (2, 4))])
def test_row_exit_matches_single_rows(k, depths):
    """T-E1: row b of a mixed batch equals that row alone at uniform depth d_b."""
    stack = _k3(k=k, loop_iterations=4)
    _randomize_carrier(stack)
    x, upstream = _inputs()
    y, dx, _ = _rows(stack, x, upstream, depths)
    for row, depth in enumerate(depths):
        y_row, dx_row, _ = _with_depth(
            stack,
            depth,
            lambda row=row: _forward_backward(
                stack, x[:, row : row + 1], upstream[:, row : row + 1]
            ),
        )
        torch.testing.assert_close(y[:, row : row + 1], y_row, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(dx[:, row : row + 1], dx_row, rtol=1e-10, atol=1e-10)


def test_equal_row_depths_are_uniform_depth_bitwise():
    """T-E1: equal depths add no mask and no padding: uniform depth bitwise."""
    stack = _k3(k=2, loop_iterations=4)
    _randomize_carrier(stack)
    x, upstream = _inputs()
    rows = _rows(stack, x, upstream, (3, 3))
    _assert_bitwise(rows, _with_depth(stack, 3, lambda: _forward_backward(stack, x, upstream)))


@pytest.mark.parametrize("k,depths", [(1, (1, 3)), (2, (2, 4)), (1, (1, 1))])
def test_exited_history_stops_growing(k, depths):
    """T-E2: in every later core call, the coda and the head, an exited row's weights on the
    history beyond its own ``1 + k d_b`` entries are exactly 0, and every row's weights sum to
    1 per token; the production sink's weights sum to 1 per call."""
    stack = _k3(k=k, loop_iterations=4)
    _randomize_carrier(stack)
    x, _ = _inputs()
    sink = RecordingSink()
    stack.enable_carrier_diagnostics(True, sink, per_sublayer=True)
    stack.set_loop_depths(torch.tensor(depths, device=device()))
    _forward(stack, x)
    rounds = max(depths)
    block = stack.loop_roles.core // k
    checked = 0
    for (segment, offset, round_index), alpha in sink.calls:
        torch.testing.assert_close(alpha.sum(0), torch.ones_like(alpha[0]))
        # The history entries of this call (the partial, if any, follows them).
        if segment == "core":
            n_hist = 1 + k * (round_index - 1) + offset // block
        else:
            n_hist = 1 + k * rounds
        assert alpha.shape[0] - n_hist in (0, 1)
        for row, depth in enumerate(depths):
            if segment == "core" and round_index <= depth:
                assert torch.count_nonzero(alpha[:, :, row] == 0) == 0
                continue
            beyond = alpha[1 + k * depth : n_hist, :, row]
            assert torch.count_nonzero(beyond) == 0, (segment, offset, round_index, row)
            checked += beyond.numel()
    assert checked > 0 or depths == (1, 1)
    totals = stack.take_carrier_diagnostics()["weights"]
    for key, (weights, tokens) in totals.items():
        torch.testing.assert_close(weights.sum(), tokens.to(weights.dtype), msg=str(key))


def _capture_blocks(stack, monkeypatch):
    """Record every round's block sums (the history entries the round appends)."""
    captured = []
    original = LoopAttnResHybridStack._attn_res_round

    def wrapped(self, sources, step, round_ctx, masks, sink, label):
        blocks = original(self, sources, step, round_ctx, masks, sink, label)
        for block in blocks:
            if block.requires_grad:
                captured.append((step + 1, block))
        return blocks

    monkeypatch.setattr(LoopAttnResHybridStack, "_attn_res_round", wrapped)
    return captured


@pytest.mark.parametrize("k,depths", [(1, (1, 3)), (2, (2, 4))])
def test_skipped_rounds_get_zero_gradient(monkeypatch, k, depths):
    """T-E3: an exited row's block sums of rounds ``> d_b`` get an exactly-zero gradient; with
    the loss on that row only, every parameter gradient equals the row's single-row run."""
    stack = _k3(k=k, loop_iterations=max(depths))
    _randomize_carrier(stack)
    x, upstream = _inputs()
    captured = _capture_blocks(stack, monkeypatch)
    grads = []
    stack.set_loop_depths(torch.tensor(depths, device=device()))
    x_leaf = x.clone().requires_grad_(True)
    y = _forward(stack, x_leaf)
    for round_index, block in captured:
        block.register_hook(lambda grad, r=round_index: grads.append((r, grad.detach().clone())))
    (y * upstream).sum().backward()
    assert len(grads) == k * max(depths)
    for round_index, grad in grads:
        for row, depth in enumerate(depths):
            if round_index > depth:
                assert torch.count_nonzero(grad[:, row]) == 0, (round_index, row)
            else:
                assert torch.count_nonzero(grad[:, row]) > 0, (round_index, row)
    monkeypatch.undo()
    row = 0
    mask = torch.zeros_like(upstream)
    mask[:, row] = 1
    _, _, batch_grads = _rows(stack, x, upstream * mask, depths)
    _, _, single = _with_depth(
        stack,
        depths[row],
        lambda: _forward_backward(stack, x[:, row : row + 1], upstream[:, row : row + 1]),
    )
    assert batch_grads.keys() == single.keys()
    for name, grad in single.items():
        torch.testing.assert_close(batch_grads[name], grad, rtol=1e-10, atol=1e-10, msg=name)


def test_exited_rows_are_padding_for_router_and_indexer_scope():
    """T-E4 (CPU part): in rounds ``> d_b`` the layer padding mask and the DSv4 indexer-loss
    scope mark the exited row; earlier rounds and the coda see no mask (no row-tail padding)."""
    _check_exit_padding_scopes(None)


def test_coda_indexer_scope_keeps_the_row_tail_padding():
    """T-E4 (CPU part) with row-tail padding on row 1: every core round, the coda and the head
    run inside the indexer-loss scope of the batch padding (rounds ``> d_b`` add the exited
    row), so the DSv4 indexer loss never trains on the padded tail in K3 runs."""
    _check_exit_padding_scopes((False, True))


def _check_exit_padding_scopes(tail):
    """Run depths (1, 3) with row-tail padding on the rows of ``tail`` (None: no padding) and
    check every sublayer's padding mask and indexer-loss scope (per-row ``any``)."""
    stack = _k3(loop_iterations=3)
    x, upstream = _inputs()
    padding = None
    if tail is not None:
        padding = torch.zeros(BATCH, SEQ, dtype=torch.bool, device=device())
        for row, padded in enumerate(tail):
            padding[row, -4:] = padded
    seen = []
    head_scopes = []
    head_forward = stack.loop_attn_res_head.forward

    def head(*args, **kwargs):
        scope = indexer_loss_padding_mask_in_scope()
        head_scopes.append(None if scope is None else tuple(scope.any(1).tolist()))
        return head_forward(*args, **kwargs)

    stack.loop_attn_res_head.forward = head

    def record(module, args, kwargs):
        scope = indexer_loss_padding_mask_in_scope()
        mask = kwargs.get("padding_mask")
        seen.append(
            (
                module.layer_number,
                None if mask is None else tuple(mask.any(1).tolist()),
                None if scope is None else tuple(scope.any(1).tolist()),
            )
        )

    handles = [layer.register_forward_pre_hook(record, with_kwargs=True) for layer in stack.layers]

    def run(module, hidden):
        module.set_loop_depths(torch.tensor((1, 3), device=device()))
        return module(
            hidden, attention_mask=None, rotary_pos_emb=_rope(hidden.dtype), padding_mask=padding
        )

    try:
        _forward_backward(stack, x, upstream, run=run)
    finally:
        for handle in handles:
            handle.remove()
    roles = stack.loop_roles
    core_numbers = range(roles.core_start + 1, roles.core_end + 1)
    core = [(mask, scope) for number, mask, scope in seen if number in core_numbers]
    coda_numbers = range(roles.core_end + 1, len(stack.layers) + 1)
    coda = [(mask, scope) for number, mask, scope in seen if number in coda_numbers]
    per_round = roles.core
    base = None if tail is None else tuple(tail)
    exited = (True, False) if tail is None else (True, tail[1])
    assert len(core) == 3 * per_round
    assert core[:per_round] == [(base, base)] * per_round
    assert core[per_round:] == [(exited, exited)] * (2 * per_round)
    assert len(coda) == roles.coda and coda == [(base, base)] * roles.coda
    assert head_scopes == [base]


# ---------------------------------------------------------------------------
# T-R1: recompute; T-F1: FLOPs; T-C1: residual carrier; diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depths", [None, (1, 3)])
@pytest.mark.parametrize("k", [1, 2])
def test_loop_recompute_is_bitwise_and_counts_once(k, depths):
    """T-R1."""
    reference = _k3(k=k, loop_iterations=3)
    recomputed = _k3(k=k, loop_iterations=3, loop_recompute=True)
    _randomize_carrier(reference)
    recomputed.load_state_dict(reference.state_dict())
    x, upstream = _inputs()
    sinks = []
    results = []
    for stack in (reference, recomputed):
        sink = RecordingSink()
        sinks.append(sink)
        stack.enable_carrier_diagnostics(True, sink)
        if depths is None:
            results.append(_forward_backward(stack, x, upstream))
        else:
            results.append(_rows(stack, x, upstream, depths))
    _assert_bitwise(results[1], results[0])
    assert [key for key, _ in sinks[0].calls] == [key for key, _ in sinks[1].calls]
    for (_, a), (_, b) in zip(sinks[0].calls, sinks[1].calls):
        assert torch.equal(a, b)


def test_flops_have_no_adapter_term():
    """T-F1: the carriers without an adapter report the effective sublayer rows only."""
    k3 = _k3(loop_iterations=3, params_dtype=torch.float32)
    residual = _residual(loop_iterations=3, params_dtype=torch.float32)
    injection = build_local_loop_stack(DENSE, 2, 2, loop_iterations=3)
    x, _ = _inputs(dtype=torch.float32)
    with torch.no_grad():
        for stack in (k3, residual, injection):
            _forward(stack, x)
    tokens = SEQ * BATCH
    rows = k3.range_forward_flops(0, 2, tokens, tokens * SEQ, SEQ)  # one '*-' block
    expected = (2 + 3 * 2) * rows  # prelude + 3 rounds of the 2-block core + coda
    metric = "loop forward flops per token"
    assert k3.loop_metrics[metric].item() == pytest.approx(expected / tokens)
    assert residual.loop_metrics[metric].item() == pytest.approx(expected / tokens)
    adapter = 3 * 4 * tokens * HIDDEN**2
    assert injection.loop_metrics[metric].item() == pytest.approx((expected + adapter) / tokens)
    assert k3.round_extra_forward_flops(tokens) == 0
    assert k3.loop_metrics["loop active token layers"] == tokens * 3 * 4


def _repeated_core(stack, x, depth):
    hidden, ctx = stack._prepare_forward(
        x,
        None,
        None,
        _rope(x.dtype),
        packed_seq_params=None,
        padding_mask=None,
        input_ids=None,
        packed_seq_params_by_layout=None,
        cp_layout_plan=None,
    )
    roles = stack.loop_roles
    for index in range(roles.core_start):
        hidden = stack._run_layer(index, hidden, ctx)
    for _ in range(depth):
        for index in range(roles.core_start, roles.core_end):
            hidden = stack._run_layer(index, hidden, ctx)
    for index in range(roles.core_end, len(stack.layers)):
        hidden = stack._run_layer(index, hidden, ctx)
    return stack._finalize_forward(hidden)


@pytest.mark.parametrize("recompute", [False, True])
def test_residual_carrier_is_the_repeated_core(recompute):
    """T-C1: the residual carrier equals a hand-written repeated core; row exit self-carries."""
    stack = _residual(loop_iterations=3, loop_recompute=recompute)
    x, upstream = _inputs()
    _assert_bitwise(
        _forward_backward(stack, x, upstream),
        _forward_backward(stack, x, upstream, run=lambda s, h: _repeated_core(s, h, 3)),
    )
    y, dx, _ = _rows(stack, x, upstream, (1, 3))
    for row, depth in enumerate((1, 3)):
        y_row, dx_row, _ = _with_depth(
            stack,
            depth,
            lambda row=row: _forward_backward(
                stack, x[:, row : row + 1], upstream[:, row : row + 1]
            ),
        )
        torch.testing.assert_close(y[:, row : row + 1], y_row, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(dx[:, row : row + 1], dx_row, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("k", [1, 2])
def test_summary_diagnostics(k):
    """The summary metrics: per segment the weights sum to 1; at w = 0 the round-r state RMS
    equals the residual carrier's (the same stream); populations follow the reporting rule;
    nothing is recorded in eval or with the diagnostics off."""
    rounds = 3
    k3 = _k3(k=k, loop_iterations=rounds, layernorm_epsilon=1e-12)
    residual = _residual(loop_iterations=rounds, layernorm_epsilon=1e-12)
    x, _ = _inputs()
    with torch.no_grad():
        _forward(k3, x)
    assert not any(name.startswith(("loop attnres", "loop state")) for name in k3.loop_metrics)
    k3.enable_carrier_diagnostics(True, per_sublayer=True)
    residual.enable_carrier_diagnostics(True, per_sublayer=True)
    padding = torch.zeros(BATCH, SEQ, dtype=torch.bool, device=device())
    padding[1, -4:] = True
    for stack in (k3, residual):
        stack(x, attention_mask=None, rotary_pos_emb=_rope(), padding_mask=padding)
    metrics, populations = k3.loop_metrics, k3.loop_metric_populations
    valid = BATCH * SEQ - 4
    for r in range(1, rounds + 1):
        tags = ["e", "current"] + [f"lag{d}" for d in range(1, r)]
        total = sum(metrics[f"loop attnres weight r{r} {tag}"] for tag in tags)
        assert total.item() == pytest.approx(1.0)
        assert int(populations[f"loop attnres weight r{r} e"]) == valid
        torch.testing.assert_close(
            metrics[f"loop state rms r{r}"],
            residual.loop_metrics[f"loop state rms r{r}"],
            rtol=1e-9,
            atol=1e-9,
        )
        assert f"loop attnres block rms r{r}" in metrics
    for segment in ("coda", "head"):
        tags = ["e", "current"] + [f"round{q}" for q in range(1, rounds + 1)]
        total = sum(metrics[f"loop attnres weight {segment} {tag}"] for tag in tags)
        assert total.item() == pytest.approx(1.0)
    # w = 0: the head's weight on e is 1 / (1 + kR + 1).
    assert metrics["loop attnres weight head e"].item() == pytest.approx(1 / (2 + k * rounds))
    assert metrics["loop state rms prelude"].item() == pytest.approx(
        residual.loop_metrics["loop state rms prelude"].item()
    )
    report = loop_metric_report(metrics, issued_tokens=BATCH * SEQ, populations=populations)
    assert report["loop attnres weight r1 e"][1].item() == valid
    sums = k3.take_carrier_diagnostics()
    assert sums["k"] == k
    assert ("head", 0, rounds) in sums["weights"] and ("core", 0, 1) in sums["weights"]
    assert k3.take_carrier_diagnostics()["weights"] == {}
    k3.eval()
    with torch.no_grad():
        _forward(k3, x)
    assert not any(name.startswith("loop attnres") for name in k3.loop_metrics)
    assert k3.take_carrier_diagnostics()["weights"] == {}
    # Summary only (no per-sublayer consumer): the metrics, but nothing accumulates.
    k3.train()
    k3.enable_carrier_diagnostics(True)
    _forward(k3, x)
    assert "loop attnres weight r1 e" in k3.loop_metrics
    assert k3.take_carrier_diagnostics()["weights"] == {}


@pytest.mark.parametrize("carrier", ["attn_res", "residual"])
def test_diagnostic_keys_do_not_depend_on_the_rounds_run(carrier):
    """The loss-dict reduction needs one key set on every microbatch and rank: a forward whose
    rows all exit early reports the rounds it did not run with population 0."""
    stack = _k3(loop_iterations=3) if carrier == "attn_res" else _residual(loop_iterations=3)
    stack.enable_carrier_diagnostics(True)
    x, upstream = _inputs()
    _forward_backward(stack, x, upstream)
    full = dict(stack.loop_metric_populations)
    _rows(stack, x, upstream, (1, 1))
    early = stack.loop_metric_populations
    assert set(early) == set(full)
    assert int(early["loop state rms r3"]) == 0 and int(full["loop state rms r3"]) == BATCH * SEQ
    if carrier == "attn_res":
        assert int(early["loop attnres weight r3 lag2"]) == 0
        assert stack.loop_metrics["loop attnres weight head round3"].item() == 0


@pytest.mark.parametrize("k", [1, 2])
def test_layer_residual_observations_follow_the_partials(k):
    """Tensor observation (``residual_accumulator`` / ``residual_contribution``): one pair per
    sublayer call; a K3 sublayer reports its block's partial (zero at a block start) and its
    delta, so the next accumulator in a block is the previous accumulator plus contribution."""
    rounds = 2
    stack = _k3(k=k, loop_iterations=rounds)
    _randomize_carrier(stack)
    x, _ = _inputs()
    observed = []

    def observer(owner, name, kind, tensor, *dims):
        observed.append((owner, kind, tensor.detach().clone()))

    kinds = frozenset({"residual_accumulator", "residual_contribution"})
    with capture_tensor_observations(observer, kinds):
        _forward(stack, x)
    roles = stack.loop_roles
    calls = roles.prelude + rounds * roles.core + roles.coda
    assert len(observed) == 2 * calls
    pairs = []
    for index in range(0, len(observed), 2):
        (owner_a, kind_a, accumulator), (owner_c, kind_c, contribution) = observed[
            index : index + 2
        ]
        assert (kind_a, kind_c) == ("residual_accumulator", "residual_contribution")
        assert owner_a is owner_c
        pairs.append((owner_a, accumulator, contribution))
    owners = [owner for owner, _, _ in pairs]
    layers = list(stack.layers)
    expected_owners = (
        layers[: roles.core_start]
        + layers[roles.core_start : roles.core_end] * rounds
        + layers[roles.core_end :]
    )
    assert all(a is b for a, b in zip(owners, expected_owners))
    block = roles.core // k
    starts = {roles.prelude + r * roles.core + b * block for r in range(rounds) for b in range(k)}
    starts.add(roles.prelude + rounds * roles.core)  # the coda block
    for call, (_, accumulator, contribution) in enumerate(pairs):
        if call in starts:
            assert torch.count_nonzero(accumulator) == 0, call
        elif call > 0:
            _, previous, previous_delta = pairs[call - 1]
            torch.testing.assert_close(
                accumulator, previous + previous_delta, rtol=1e-12, atol=1e-12, msg=str(call)
            )
        assert torch.count_nonzero(contribution) > 0, call


def test_production_sink_matches_the_recording_sink():
    stack = _k3(loop_iterations=2)
    _randomize_carrier(stack)
    sink = RecordingSink()
    stack.enable_carrier_diagnostics(True, sink, per_sublayer=True)
    x, _ = _inputs()
    _forward(stack, x)
    totals = stack.take_carrier_diagnostics()["weights"]
    expected = AttnResStats()
    for key, alpha in sink.calls:
        expected.add(key, alpha, None)
    for key, (weights, tokens) in expected.take().items():
        torch.testing.assert_close(totals[key][0], weights)
        assert totals[key][1].item() == tokens.item()


def test_base_stack_takes_no_sink():
    with pytest.raises(ValueError, match="sink"):
        _residual().enable_carrier_diagnostics(True, RecordingSink())
    assert _residual().take_carrier_diagnostics() is None


def test_context_default_is_not_a_delta():
    fields = {field.name: field.default for field in dataclasses.fields(HybridStackLayerContext)}
    assert fields["return_layer_delta"] is False
