# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for LoopHybridStack, the eager hybrid loop (megatron/core/transformer/loop/).

Dense and 4-stream mHC stacks of local layers run on CPU (gloo) and CUDA: construction and
guards, a round-by-round oracle, per-row exit, loop recompute, layer-call accounting
(observation, quantization context, hash token IDs), the FLOP metrics, HybridModel wiring
with MTP, and the training loss plumbing. The DSv4 / MoE / hash HybridModel case is CUDA-only.
"""

import contextlib
import re
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.backends import LocalSpecProvider
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_observation import capture_tensor_observations
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.loop import guards
from megatron.core.transformer.loop.hybrid_specs import (
    loop_hybrid_dsv4_stack_spec,
    loop_hybrid_stack_spec,
)
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack, LoopStreamRMSNorm
from megatron.core.transformer.loop_transformer_block import LoopRMSNorm
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionBlockSubmodules,
    get_mtp_layer_spec_for_backend,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from tests.unit_tests.transformer.loop import hybrid_loop_utils
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    BATCH,
    CUDA,
    HIDDEN,
    SEQ,
    build_local_loop_stack,
    device,
    hybrid_loop_config,
    local_hybrid_submodules,
    model_parallel_once,
    rope,
)

# The CPU mHC fixture, made visible to pytest in this module.
mhc_on_this_device = hybrid_loop_utils.mhc_on_this_device

DENSE = "*-*-*-*-"  # prelude '*-', core '*-*-', coda '*-'
MHC = dict(enable_mhc_connections=True, mhc_num_residual_streams=4, mhc_sinkhorn_iterations=5)

gpu = pytest.mark.skipif(not CUDA, reason="TE / MoE / DSv4 layers need CUDA")


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


@pytest.fixture
def deterministic_algorithms(monkeypatch):
    """Deterministic kernels for the bitwise GPU comparisons (the H.2 / H.3 harness setup).

    Set after the module's model-parallel initialization, which clears the NVTE variables.
    """
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _stack(pattern=DENSE, prelude=2, coda=2, **overrides):
    return build_local_loop_stack(pattern, prelude, coda, **overrides)


def _inputs(batch=BATCH, seed=7):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(SEQ, batch, HIDDEN, generator=generator).to(device())
    upstream = torch.randn(SEQ, batch, HIDDEN, generator=generator).to(device())
    return x, upstream


def _forward_backward(stack, x, upstream, run=None):
    """Forward (``run`` or the stack) + backward of ``sum(y * upstream)``; detached results."""
    x = x.detach().clone().requires_grad_(True)
    stack.zero_grad(set_to_none=True)
    y = (run or _forward)(stack, x)
    ((y * upstream).sum() + sum(stack.loop_losses.values())).backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in stack.named_parameters()
        if param.grad is not None
    }
    return y.detach().clone(), x.grad.detach().clone(), grads


def _forward(stack, x):
    return stack(x, attention_mask=None, rotary_pos_emb=rope())


def _assert_bitwise(actual, expected):
    y, dx, grads = actual
    y_ref, dx_ref, grads_ref = expected
    assert torch.equal(y, y_ref)
    assert torch.equal(dx, dx_ref)
    assert grads.keys() == grads_ref.keys()
    for name, grad in grads_ref.items():
        assert torch.equal(grads[name], grad), name


# ---------------------------------------------------------------------------
# Construction, guards, specs
# ---------------------------------------------------------------------------


def test_roles_and_loop_parameters_are_children():
    stack = _stack()
    assert stack.loop_roles == guards.LoopRoles(prelude=2, core=4, coda=2)
    assert stack._loop_symbols == tuple(DENSE)
    assert isinstance(stack.loop_norm, LoopRMSNorm)
    names = {name for name, _ in stack.named_parameters()}
    assert {"loop_adapter.weight", "loop_norm.weight"} <= names
    assert stack.loop_adapter.weight.shape == (HIDDEN, 2 * HIDDEN)

    mor = _stack(loop_architecture="mor", loop_token_execution="masked", loop_iterations=3)
    assert {"depth_router.0.weight", "depth_router.2.weight"} <= {
        name for name, _ in mor.named_parameters()
    }
    assert mor.depth_router[2].weight.shape == (3, HIDDEN)
    # Masked MoR: arbitrary masks on the core '*' sublayers only.
    from megatron.core.transformer.enums import AttnMaskType

    kinds = [
        getattr(getattr(layer, "self_attention", None), "attn_mask_type", None)
        for layer in mor.layers
    ]
    assert kinds[2] == kinds[4] == AttnMaskType.arbitrary
    assert kinds[0] != AttnMaskType.arbitrary and kinds[6] != AttnMaskType.arbitrary


def test_sharded_state_dict_holds_the_loop_parameters():
    keys = set(_stack().sharded_state_dict(prefix="decoder."))
    assert {"decoder.loop_adapter.weight", "decoder.loop_norm.weight"} <= keys


def test_mhc_streams_get_a_per_stream_norm(mhc_on_this_device):
    stack = _stack(**MHC)
    assert stack.loop_num_streams == 4
    assert isinstance(stack.loop_norm, LoopStreamRMSNorm)
    assert stack.loop_norm.weight.shape == (4, HIDDEN)


def test_stream_norm_is_the_loop_norm_per_stream():
    torch.manual_seed(3)
    norm = LoopStreamRMSNorm(HIDDEN, 3, 1e-6, torch.float32, "cpu")
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    x = torch.randn(5, 2, 3 * HIDDEN)
    expected = []
    for stream in range(3):
        single = LoopRMSNorm(HIDDEN, 1e-6, torch.float32, "cpu")
        with torch.no_grad():
            single.weight.copy_(norm.weight[stream])
        expected.append(single(x[..., stream * HIDDEN : (stream + 1) * HIDDEN]))
    torch.testing.assert_close(norm(x), torch.cat(expected, -1), rtol=0, atol=0)


def test_adapter_is_shared_by_every_stream(mhc_on_this_device):
    stack = _stack(**MHC)
    state = torch.randn(SEQ, BATCH, 4 * HIDDEN, device=device())
    injection = torch.randn(SEQ, BATCH, 4 * HIDDEN, device=device())
    expected = torch.cat(
        [
            stack.loop_adapter(
                torch.cat(
                    (
                        state[..., s * HIDDEN : (s + 1) * HIDDEN],
                        injection[..., s * HIDDEN : (s + 1) * HIDDEN],
                    ),
                    -1,
                )
            )
            for s in range(4)
        ],
        -1,
    )
    torch.testing.assert_close(stack._adapt(state, injection), expected, rtol=1e-6, atol=1e-6)


def test_stack_rejects_construction_without_a_loop():
    with pytest.raises(ValueError, match=guards.LOOP_STACK_WITHOUT_LOOP):
        _stack(loop_architecture="none", loop_prelude_sublayers=None, loop_coda_sublayers=None)


def test_stack_rejects_being_an_mtp_stack():
    config = hybrid_loop_config(DENSE, 2, 2)
    with pytest.raises(ValueError, match=guards.MTP_LOOP_STACK):
        LoopHybridStack(
            config=config,
            submodules=local_hybrid_submodules(),
            layer_config_list=validate_segment_layers(DENSE, config),
            post_layer_norm=False,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            is_mtp_layer=True,
        )


def test_stack_runs_the_config_and_role_guards():
    with pytest.raises(ValueError, match=guards.DROPOUT):
        _stack(hidden_dropout=0.1)
    with pytest.raises(ValueError, match=guards.CUT_NOT_ON_BLOCK):
        _stack(loop_prelude_sublayers=1, loop_coda_sublayers=3)


def test_stack_checks_its_inputs_every_forward():
    stack = _stack()
    x, _ = _inputs()
    three = torch.tensor([0, 4, 8, 16], dtype=torch.int32, device=device())
    packed = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=three, cu_seqlens_kv=three, max_seqlen_q=8, max_seqlen_kv=8
    )
    with pytest.raises(ValueError, match=guards.PACKED_MULTI_SEQUENCE):
        stack(x, attention_mask=None, rotary_pos_emb=rope(), packed_seq_params=packed)


def test_specs_select_the_loop_stack():
    assert loop_hybrid_stack_spec.module is LoopHybridStack
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec

    assert loop_hybrid_stack_spec.submodules is hybrid_stack_spec.submodules
    assert loop_hybrid_stack_spec.params == hybrid_stack_spec.params
    assert callable(loop_hybrid_dsv4_stack_spec)


def test_ds_attention_flop_row_takes_the_built_ratio():
    """A 'D' sublayer's FLOP row uses its built DSv4 attention's ratio (mHC-wrapped too); a
    'D' built as another attention (legacy DSAttention) is rejected."""

    def attention_layer(attention):
        return SimpleNamespace(self_attention=attention)

    stub = SimpleNamespace(
        layers=[
            attention_layer(SimpleNamespace(_dsv4_compress_ratio=128)),
            SimpleNamespace(inner_layer=attention_layer(SimpleNamespace(_dsv4_compress_ratio=4))),
            attention_layer(SimpleNamespace()),
        ]
    )
    assert LoopHybridStack._compress_ratio(stub, "D", 0) == 128
    assert LoopHybridStack._compress_ratio(stub, "D", 1) == 4
    assert LoopHybridStack._compress_ratio(stub, "*", 2) is None
    with pytest.raises(ValueError, match="DSv4 hybrid attention"):
        LoopHybridStack._compress_ratio(stub, "D", 2)


# ---------------------------------------------------------------------------
# Round-by-round oracle, loop recompute, row exit
# ---------------------------------------------------------------------------


def _oracle(stack, x, depth, *, rotary_pos_emb="local", input_ids=None):
    """Huginn composed by hand from the stack's phases and loop parameters."""
    hidden, ctx = stack._prepare_forward(
        x,
        None,
        None,
        rope() if rotary_pos_emb == "local" else rotary_pos_emb,
        packed_seq_params=None,
        padding_mask=None,
        input_ids=input_ids,
        packed_seq_params_by_layout=None,
        cp_layout_plan=None,
    )
    roles = stack.loop_roles
    for index in range(roles.core_start):
        hidden = stack._run_layer(index, hidden, ctx)
    embedding = hidden
    state = torch.zeros_like(embedding)
    streams = stack.loop_num_streams
    width = stack.config.hidden_size
    split = (x.shape[0], x.shape[1], streams, width)
    for _ in range(depth):
        adapted = stack.loop_adapter(
            torch.cat((state.view(split), embedding.view(split)), -1)
        ).view(embedding.shape)
        for index in range(roles.core_start, roles.core_end):
            adapted = stack._run_layer(index, adapted, ctx)
        per_stream = adapted.view(split)
        normed = per_stream.float() * torch.rsqrt(
            per_stream.float().square().mean(-1, keepdim=True) + stack.config.layernorm_epsilon
        )
        weight = stack.loop_norm.weight.view(1, 1, streams, width)
        state = (normed.to(adapted.dtype) * weight).view(adapted.shape)
    for index in range(roles.core_end, len(stack.layers)):
        state = stack._run_layer(index, state, ctx)
    return stack._finalize_forward(state)


@pytest.mark.parametrize("depth", [1, 3])
def test_round_oracle_dense(depth):
    stack = _stack(loop_iterations=depth)
    x, upstream = _inputs()
    _assert_bitwise(
        _forward_backward(stack, x, upstream),
        _forward_backward(stack, x, upstream, run=lambda s, h: _oracle(s, h, depth)),
    )


def test_round_oracle_mhc(mhc_on_this_device):
    stack = _stack(loop_iterations=2, **MHC)
    x, upstream = _inputs()
    _assert_bitwise(
        _forward_backward(stack, x, upstream),
        _forward_backward(stack, x, upstream, run=lambda s, h: _oracle(s, h, 2)),
    )


@pytest.mark.parametrize("mhc", [False, True], ids=["dense", "mhc"])
def test_loop_recompute_is_bitwise(request, mhc):
    """A non-reentrant checkpoint per round (over mHC wrappers too) changes nothing."""
    overrides = dict(loop_iterations=2)
    if mhc:
        request.getfixturevalue("mhc_on_this_device")
        overrides.update(MHC)
    reference = _stack(**overrides)
    recomputed = _stack(loop_recompute=True, **overrides)
    recomputed.load_state_dict(reference.state_dict())
    x, upstream = _inputs()
    _assert_bitwise(
        _forward_backward(recomputed, x, upstream), _forward_backward(reference, x, upstream)
    )


def test_truncated_bptt_detaches_early_rounds():
    stack = _stack(loop_iterations=3, loop_bptt_iterations=1)
    x, upstream = _inputs()
    y, dx, grads = _forward_backward(stack, x, upstream)
    full = _stack(loop_iterations=3)
    full.load_state_dict(stack.state_dict())
    y_full, dx_full, _ = _forward_backward(full, x, upstream)
    assert torch.equal(y, y_full)  # same forward
    assert not torch.equal(dx, dx_full)  # fewer differentiated rounds
    assert stack.loop_metrics["loop backward token layers"] == SEQ * BATCH * 1 * 4


def _uniform(stack, x, upstream, depth):
    """A uniform-depth run of ``stack`` at ``depth`` (the GPT tests mutate the depth so)."""
    saved = stack.config.loop_iterations
    stack.config.loop_iterations = depth
    try:
        return _forward_backward(stack, x, upstream)
    finally:
        stack.config.loop_iterations = saved


def _rows(stack, x, upstream, depths):
    stack.set_loop_depths(torch.tensor(depths, device=device()))
    return _forward_backward(stack, x, upstream)


def test_row_exit_matches_single_rows():
    """Row b of a ``depths=[1, 3]`` batch equals that row alone at uniform depth ``depths[b]``.

    (Not bitwise: a batch-size change may change GEMM blocking.)
    """
    stack = _stack(loop_iterations=4)
    x, upstream = _inputs()
    y, dx, _ = _rows(stack, x, upstream, [1, 3])
    assert stack.loop_metrics["loop depth"] == 2
    for row, depth in enumerate((1, 3)):
        y_row, dx_row, _ = _uniform(stack, x[:, row : row + 1], upstream[:, row : row + 1], depth)
        torch.testing.assert_close(y[:, row : row + 1], y_row, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(dx[:, row : row + 1], dx_row, rtol=1e-6, atol=1e-6)


def test_equal_row_depths_are_uniform_depth():
    stack = _stack(loop_iterations=4)
    x, upstream = _inputs()
    rows = _rows(stack, x, upstream, [3, 3])
    _assert_bitwise(rows, _uniform(stack, x, upstream, 3))
    # Depths apply to one forward only.
    _assert_bitwise(_forward_backward(stack, x, upstream), _uniform(stack, x, upstream, 4))


def test_row_mode_masks_only_rounds_with_exited_rows():
    """Rounds in which every row is alive run as uniform depth (no padding mask); later rounds
    mark the exited rows as padding for the MoE router."""
    stack = _stack(loop_iterations=3)
    x, upstream = _inputs()
    seen = []

    def record(module, args, kwargs):
        if "padding_mask" in kwargs:
            mask = kwargs["padding_mask"]
            summary = None if mask is None else tuple(mask.any(1).tolist())
            seen.append((module.layer_number, summary))

    handles = [layer.register_forward_pre_hook(record, with_kwargs=True) for layer in stack.layers]
    try:
        _rows(stack, x, upstream, [3, 3])
        assert seen and all(mask is None for _, mask in seen)
        seen.clear()
        _rows(stack, x, upstream, [1, 3])
    finally:
        for handle in handles:
            handle.remove()
    roles = stack.loop_roles
    core_numbers = range(roles.core_start + 1, roles.core_end + 1)
    core_masks = [mask for number, mask in seen if number in core_numbers]
    per_round = len(core_masks) // 3
    assert per_round > 0 and len(core_masks) == 3 * per_round
    assert core_masks[:per_round] == [None] * per_round  # round 1: both rows alive
    assert core_masks[per_round:] == [(True, False)] * (2 * per_round)  # row 0 exited
    assert all(mask is None for number, mask in seen if number not in core_numbers)


@gpu
def test_equal_row_depths_are_uniform_depth_with_a_moe_core(deterministic_algorithms):
    """With a MoE core (router padding path, aux loss on) equal row depths are still uniform
    depth bitwise: no padding mask is added while every row is alive."""
    from megatron.core.transformer.spec_utils import build_module
    from tests.unit_tests.transformer.loop.hybrid_parity_utils import (
        loop_fields,
        make_hybrid_config,
        make_inputs,
        te_base,
        te_rope,
    )

    pattern = "*-*E*E*-"
    base = te_base(
        torch.bfloat16,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=256,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_aux_loss_coeff=0.01,
        moe_z_loss_coeff=1e-3,
    )
    config = make_hybrid_config(base, loop_fields("huginn", 4, 0, False, "compact"))
    torch.manual_seed(1234)
    stack = build_module(
        loop_hybrid_stack_spec,
        config=config,
        pre_process=True,
        layer_config_list=validate_segment_layers(pattern, config),
        post_process=True,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    ).cuda()
    x, upstream = make_inputs(
        torch.bfloat16, hidden=config.hidden_size, scale=config.init_method_std, seed=3
    )
    rotary = te_rope(config.kv_channels)

    def run(module, hidden):
        return module(hidden, attention_mask=None, rotary_pos_emb=rotary)

    stack.set_loop_depths(torch.tensor([3, 3], device="cuda"))
    rows = _forward_backward(stack, x, upstream, run=run)
    saved = stack.config.loop_iterations
    stack.config.loop_iterations = 3
    try:
        uniform = _forward_backward(stack, x, upstream, run=run)
    finally:
        stack.config.loop_iterations = saved
    _assert_bitwise(rows, uniform)


def test_row_exit_mhc_self_carries_every_stream(mhc_on_this_device):
    stack = _stack(loop_iterations=3, **MHC)
    x, upstream = _inputs()
    y, dx, _ = _rows(stack, x, upstream, [1, 2])
    for row, depth in enumerate((1, 2)):
        y_row, dx_row, _ = _uniform(stack, x[:, row : row + 1], upstream[:, row : row + 1], depth)
        torch.testing.assert_close(y[:, row : row + 1], y_row, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(dx[:, row : row + 1], dx_row, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "overrides,depths,match",
    [
        (dict(loop_bptt_iterations=1), [1, 2], "full BPTT"),
        ({}, [0, 2], "Loop depths must be in"),
        ({}, [1, 5], "Loop depths must be in"),
        ({}, [1, 2, 3], "integer \\[B=2\\]"),
        (
            dict(loop_architecture="mor", loop_token_execution="masked"),
            [1, 2],
            "loop_architecture='huginn'",
        ),
    ],
)
def test_row_depths_are_validated(overrides, depths, match):
    stack = _stack(**{"loop_iterations": 4, **overrides})
    x, _ = _inputs()
    stack.set_loop_depths(torch.tensor(depths, device=device()))
    with pytest.raises(ValueError, match=match):
        _forward(stack, x)


# ---------------------------------------------------------------------------
# Layer calls: observation, quantization context, hash token IDs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", ["huginn", "mor"])
@pytest.mark.parametrize("recompute", [False, True])
def test_layer_residuals_observed_once_per_layer_call(architecture, recompute):
    stack = _stack(
        loop_architecture=architecture,
        loop_token_execution="masked",
        loop_iterations=2,
        loop_recompute=recompute,
    )
    x, upstream = _inputs()
    seen = []

    def observer(owner, name, kind, tensor, *dims):
        seen.append((owner.layer_number, kind))

    kinds = frozenset({"residual_accumulator", "residual_contribution"})
    with capture_tensor_observations(observer, kinds):
        _forward_backward(stack, x, upstream)
    # Prelude 1-2 and coda 7-8 once; core 3-6 per round; replays do not observe again.
    calls = [1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 8]
    assert seen == [(n, k) for n in calls for k in sorted(kinds)]


def test_quantization_context_numbers_repeat_per_round(monkeypatch):
    stack = _stack(loop_iterations=2)
    numbers = []
    prepare = stack._prepare_forward

    def recording_prepare(*args, **kwargs):
        hidden, ctx = prepare(*args, **kwargs)
        ctx.quant_context_factory = lambda config, number: (numbers.append(number), nullcontext())[
            1
        ]
        return hidden, ctx

    monkeypatch.setattr(stack, "_prepare_forward", recording_prepare)
    _forward(stack, _inputs()[0])
    assert numbers == [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7]


def test_input_ids_reach_only_hash_layers_once_per_call(monkeypatch):
    """Token IDs go to hash-routed sublayers only (here: sublayer 2, in the prelude)."""
    stack = _stack(loop_iterations=2)
    monkeypatch.setattr(
        LoopHybridStack, "_uses_hash_routing", staticmethod(lambda layer: layer.layer_number == 2)
    )
    received = []
    for layer in stack.layers:
        layer.register_forward_pre_hook(
            lambda module, args, kwargs: received.append(
                (module.layer_number, kwargs.get("input_ids") is not None)
            ),
            with_kwargs=True,
        )
    input_ids = torch.zeros(BATCH, SEQ, dtype=torch.long, device=device())
    stack(_inputs()[0], attention_mask=None, rotary_pos_emb=rope(), input_ids=input_ids)
    assert [number for number, has_ids in received if has_ids] == [2]
    assert [number for number, _ in received] == [1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 8]


# ---------------------------------------------------------------------------
# Metrics, FLOPs, losses
# ---------------------------------------------------------------------------


def _dense_pass_flops(stack, tokens, squared):
    config = stack.config
    q_width = config.num_attention_heads * config.kv_channels
    kv_width = config.num_query_groups * config.kv_channels
    attention = 4 * tokens * HIDDEN * (q_width + kv_width) + 2 * squared * q_width
    mlp = 6 * tokens * HIDDEN * config.ffn_hidden_size
    return attention, mlp


def test_flops_per_phase():
    depth = 2
    stack = _stack(loop_iterations=depth, loop_recompute=True)
    x, upstream = _inputs()
    _forward_backward(stack, x, upstream)
    tokens = SEQ * BATCH
    attention, mlp = _dense_pass_flops(stack, tokens, tokens * SEQ)
    block = attention + mlp
    adapter = 4 * tokens * HIDDEN**2
    rounds = depth * (2 * block + adapter)
    forward = 2 * block + rounds  # prelude + coda once, the core per round
    metrics = stack.loop_metrics
    assert metrics["loop forward flops per token"].item() == pytest.approx(forward / tokens)
    assert metrics["loop backward flops per token"].item() == pytest.approx(2 * forward / tokens)
    assert metrics["loop recompute flops per token"].item() == pytest.approx(rounds / tokens)
    assert metrics["loop depth"] == depth
    assert metrics["loop active token layers"] == tokens * depth * 4  # sublayers

    with torch.no_grad():
        stack.eval()
        _forward(stack, x)
    assert stack.loop_metrics["loop forward flops per token"].item() == pytest.approx(
        forward / tokens
    )
    assert stack.loop_metrics["loop backward flops per token"] == 0
    assert stack.loop_metrics["loop recompute flops per token"] == 0
    assert stack.loop_metrics["loop backward token layers"] == 0


def test_mhc_adapter_flops_count_every_stream(mhc_on_this_device):
    dense = _stack(loop_iterations=2)
    mhc = _stack(loop_iterations=2, **MHC)
    x, _ = _inputs()
    with torch.no_grad():
        _forward(dense, x)
        _forward(mhc, x)
    tokens = SEQ * BATCH
    # Same layer FLOPs (mHC mixing excluded); the shared adapter runs on 4 streams per round.
    extra = 2 * (4 - 1) * 4 * tokens * HIDDEN**2
    added = (
        mhc.loop_metrics["loop forward flops per token"]
        - dense.loop_metrics["loop forward flops per token"]
    ).item()
    assert added == pytest.approx(extra / tokens)


def test_head_flops_add_logits_and_mtp_depths():
    stack = _stack(loop_iterations=1)
    x, _ = _inputs()
    with torch.no_grad():
        _forward(stack, x)
    base = stack.loop_metrics["loop forward flops per token"].item()
    vocab = 128
    stack.set_loop_head_flops(vocab, "*-", 2)
    with torch.no_grad():
        _forward(stack, x)
    tokens = SEQ * BATCH
    attention, mlp = _dense_pass_flops(stack, tokens, tokens * SEQ)
    logits = 2 * tokens * HIDDEN * vocab
    mtp = 2 * (2 * (3 * HIDDEN + 2 * HIDDEN**2) * tokens + attention + mlp + logits)
    added = stack.loop_metrics["loop forward flops per token"].item() - base
    assert added == pytest.approx((logits + mtp) / tokens)


@pytest.mark.parametrize(
    "mtp_pattern,rule", [("*M", guards.UNSUPPORTED_SUBLAYERS), ("D-", guards.MTP_DS_ATTENTION)]
)
def test_head_flops_reject_mtp_sublayers_without_a_row(mtp_pattern, rule):
    with pytest.raises(ValueError, match=re.escape(rule)):
        _stack().set_loop_head_flops(128, mtp_pattern, 1)


def test_loss_and_metric_dicts_are_new_every_forward():
    stack = _stack(loop_architecture="mor", loop_token_execution="masked", loop_iterations=2)
    x, _ = _inputs()
    _forward(stack, x)
    losses, metrics = stack.loop_losses, stack.loop_metrics
    snapshot = {name: value.detach().clone() for name, value in metrics.items()}
    _forward(stack, x * 2)
    assert stack.loop_losses is not losses and stack.loop_metrics is not metrics
    assert set(losses) == {"depth balance loss", "depth z loss"}
    for name, value in snapshot.items():
        assert torch.equal(metrics[name], value), name


def test_pretrain_hybrid_loss_func_adds_loop_losses(monkeypatch):
    import pretrain_hybrid

    monkeypatch.setattr(
        pretrain_hybrid,
        "get_args",
        lambda: SimpleNamespace(
            logits_load_dir=None, check_for_nan_in_loss_and_grad=False, check_for_spiky_loss=False
        ),
    )
    monkeypatch.setattr(pretrain_hybrid, "has_nvidia_modelopt", False, raising=False)
    loss_mask = torch.ones(2, 4)
    output = torch.full((2, 4), 0.5)
    balance = torch.tensor(0.25, requires_grad=True)
    loss, num_tokens, report = pretrain_hybrid.loss_func(
        loss_mask,
        output,
        loop_losses={"depth balance loss": balance},
        loop_metrics={"loop depth": torch.tensor(3.0)},
    )
    assert num_tokens == 8
    assert loss.item() == pytest.approx(8 * 0.5 + 8 * 0.25)
    loss.backward()
    assert balance.grad.item() == 8
    assert report["depth balance loss"].tolist() == [2.0, 8]
    assert report["loop depth"].tolist() == [24.0, 8]
    _, _, plain_report = pretrain_hybrid.loss_func(loss_mask, output)
    assert set(plain_report) == {"lm loss"}


# ---------------------------------------------------------------------------
# HybridModel wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def cpu_mtp(monkeypatch):
    """Without TE the MTP layer's inference-linear isinstance check raises and the MTP loss
    logger allocates on CUDA (environment limitations, not the code under test)."""
    if not CUDA:
        from megatron.core.transformer import multi_token_prediction as mtp_module

        monkeypatch.setattr(mtp_module, "is_inference_column_parallel_linear", lambda module: False)
        monkeypatch.setattr(
            mtp_module.MTPLossLoggingHelper,
            "save_loss_to_tracker",
            staticmethod(lambda *args, **kwargs: None),
        )


def _local_model_spec(module):
    mtp_layer = get_mtp_layer_spec_for_backend(None, LocalSpecProvider(), rms_norm=True)
    submodules = local_hybrid_submodules()
    submodules.mtp_block_spec = ModuleSpec(
        module=MultiTokenPredictionBlock,
        submodules=MultiTokenPredictionBlockSubmodules(layer_specs=[mtp_layer]),
    )
    # No TE final norm on CPU.
    return ModuleSpec(module=module, params={"post_layer_norm": False}, submodules=submodules)


def _local_model(pattern, module, **overrides):
    main, *mtp = pattern.split("/")
    if mtp:
        overrides.update(mtp_num_layers=len(mtp), mtp_loss_scaling_factor=0.1)
    config = hybrid_loop_config(main, 2, 2, **overrides)
    torch.manual_seed(1234)
    model = HybridModel(
        config=config,
        hybrid_stack_spec=_local_model_spec(module),
        vocab_size=128,
        max_sequence_length=SEQ,
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
    )
    return model.to(device())


def test_hybrid_model_requires_the_loop_stack():
    with pytest.raises(ValueError, match=guards.NON_LOOP_DECODER):
        _local_model(DENSE, HybridStack)


def test_hybrid_model_trains_a_loop_with_mtp(cpu_mtp):
    model = _local_model(DENSE + "/*-", LoopHybridStack)
    assert type(model.decoder) is LoopHybridStack
    assert type(model.mtp.layers[0].mtp_model_layer) is HybridStack
    assert model.decoder._loop_head == (128, ("*", "-"), 1)
    generator = torch.Generator().manual_seed(5)
    tokens = torch.randint(0, 128, (BATCH, SEQ), generator=generator).to(device())
    labels = torch.randint(0, 128, (BATCH, SEQ), generator=generator).to(device())
    position_ids = torch.arange(SEQ, device=device()).repeat(BATCH, 1)
    loss = model(tokens, position_ids, None, labels=labels)
    assert loss.shape == (BATCH, SEQ) and torch.isfinite(loss).all()
    loss.sum().backward()
    for name, param in model.named_parameters():
        if name.startswith(("decoder.", "mtp.")):
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
    assert model.decoder.loop_metrics["loop forward flops per token"] > 0


@gpu
def test_dsv4_hybrid_model_loop_with_mhc_hash_and_mtp(deterministic_algorithms):
    """DSv4 W / C / H attention (unfused backend), MoE with one hash layer in the prelude, mHC 4,
    MTP 1, built through loop_hybrid_dsv4_stack_spec. SBHD: attention_mask=None (CSA owns
    causal masking). Deterministic kernels; a mismatch is diagnosed by a self-determinism
    rerun."""
    from unittest.mock import patch

    from megatron.core.transformer.transformer_config import MLATransformerConfig

    try:
        from fast_hadamard_transform import hadamard_transform  # noqa: F401

        hadamard_patch = contextlib.nullcontext()
    except ImportError:
        hadamard_patch = patch(
            "megatron.core.transformer.experimental_attention_variant.csa.rotate_activation",
            lambda x: x * (x.size(-1) ** -0.5),
        )
    pattern = "WECEHE/WE"  # prelude 'WE' (hash E = layer 2), core 'CE', coda 'HE'
    vocab = 128
    config = MLATransformerConfig(
        num_layers=6,
        hidden_size=512,
        num_attention_heads=8,
        multi_latent_attention=True,
        q_lora_rank=192,
        kv_lora_rank=64,
        qk_head_dim=16,
        qk_pos_emb_head_dim=8,
        v_head_dim=16,
        rope_type="rope",
        rotary_base=10000,
        rotary_percent=1.0,
        experimental_attention_variant="dsv4_hybrid",
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=32,
        dsa_indexer_loss_coeff=0.0,
        csa_window_size=32,
        # One ratio per main layer plus the MTP attention layer ('W' -> 0): the config needs
        # at least num_layers + mtp_num_layers entries.
        csa_compress_ratios=[0, 0, 4, 0, 128, 0, 0],
        num_moe_experts=8,
        moe_ffn_hidden_size=256,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=1e-4,
        moe_token_dispatcher_type="alltoall",
        moe_n_hash_layers=1,
        actual_vocab_size=vocab,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=5,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        loop_architecture="huginn",
        loop_iterations=2,
        loop_prelude_sublayers=2,
        loop_coda_sublayers=2,
        deterministic_mode=True,
        # Unfused, as the docstring says: it also agrees with the NVTE_FLASH_ATTN=0 /
        # NVTE_FUSED_ATTN=0 that tests/unit_tests/conftest.py sets for every test.
        attention_backend=AttnBackend.unfused,
    )
    seq = 256
    with hadamard_patch:
        model = HybridModel(
            config=config,
            hybrid_stack_spec=loop_hybrid_dsv4_stack_spec(config),
            vocab_size=vocab,
            max_sequence_length=1024,
            hybrid_layer_pattern=pattern,
            position_embedding_type="rope",
        ).cuda()
        decoder = model.decoder
        assert type(decoder) is LoopHybridStack
        assert decoder.loop_num_streams == 4 and decoder._has_dsv4_attention
        assert type(model.mtp.layers[0].mtp_model_layer) is HybridStack
        assert decoder.layers[1].inner_layer.mlp.router.is_hash_layer
        received = []
        for layer in decoder.layers:
            layer.register_forward_pre_hook(
                lambda module, args, kwargs: received.append(
                    (module.layer_number, kwargs.get("input_ids") is not None)
                ),
                with_kwargs=True,
            )
        generator = torch.Generator().manual_seed(5)
        tokens = torch.randint(0, vocab, (BATCH, seq), generator=generator).cuda()
        labels = torch.randint(0, vocab, (BATCH, seq), generator=generator).cuda()
        position_ids = torch.arange(seq, device="cuda").repeat(BATCH, 1)
        loss = model(tokens, position_ids, None, labels=labels)
        assert torch.isfinite(loss).all()
        loss.float().mean().backward()
        layer_calls = list(received)

        # Round-by-round oracle on the decoder: forward equals the phases run by hand
        # (the output and, with MTP, the pre-contraction streams).
        with torch.no_grad():
            hidden, _, hash_input_ids = model.embed(tokens, position_ids)
            actual = decoder(
                hidden_states=hidden,
                attention_mask=None,
                rotary_pos_emb=None,
                input_ids=hash_input_ids,
            )
            expected = _oracle(decoder, hidden, 2, rotary_pos_emb=None, input_ids=hash_input_ids)
            assert isinstance(actual, tuple) and len(actual) == len(expected) == 2
            if not all(torch.equal(got, want) for got, want in zip(actual, expected)):
                rerun = decoder(
                    hidden_states=hidden,
                    attention_mask=None,
                    rotary_pos_emb=None,
                    input_ids=hash_input_ids,
                )
                if not all(torch.equal(a, b) for a, b in zip(rerun, actual)):
                    pytest.fail("NONDETERMINISTIC: the decoder does not reproduce its forward")
                diff = max((got - want).abs().max().item() for got, want in zip(actual, expected))
                pytest.fail(f"oracle mismatch (max |diff| {diff})")
    # The hash layer (sublayer 2) runs once, in the prelude, and is the only one given ids.
    assert [number for number, has_ids in layer_calls if has_ids] == [2]
    assert [number for number, _ in layer_calls] == [1, 2, 3, 4, 3, 4, 5, 6]
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), name
    assert decoder.loop_adapter.weight.grad is not None
    assert decoder.loop_norm.weight.grad is not None
    assert decoder.loop_metrics["loop forward flops per token"] > 0
