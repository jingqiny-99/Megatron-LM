# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The K3 attention-residual carrier on DSv4 layers (GPU; stage 9, 12.2 of the design).

Runs under torchrun with CUDA at W = 8, 4 or 2 ranks (EP = W, model parallel initialized once
for the module), or on one GPU (EP = 1). The decoder is a DSV4-TINY-like ``HybridModel`` built
through ``loop_attn_res_hybrid_dsv4_stack_spec`` (DSv4 H / C / W attention with TE linears,
unfused attention, ``dsa_kernel_backend='none'``, 8 experts with the all-to-all dispatcher, no
mHC / MTP / hash layer): pattern ``'HEHECEWE'`` (prelude ``HE``, core ``HECE``, coda ``WE``).

* T-O1: at ``w = 0`` and RMSNorm eps 1e-12 the carrier equals ``loop_carrier='residual'`` with
  the same weights (the decoder output is after the final norm): fp32 within 1e-4 relative,
  bf16 with a cosine gate;
* T-D1: the layer delta of every sublayer (H / C / W attention, MoE) equals
  ``layer(h) - h`` (fp32);
* T-E1 / T-E3 (aux losses off): a row of a ``depths = [1, 3]`` batch equals that row alone at
  its depth; its block sums of skipped rounds get an exactly-zero gradient;
* T-E4 (router aux loss and DSv4 indexer loss on, main loss x 0): every sublayer input of a
  skipped round gets an exactly-zero gradient on the exited row; a control without the round
  padding mask sees a non-zero one (the test has teeth);
* the aggregation's memory at n = 10, h = 1536, S = 4096, B = 1 (bf16): saved state beyond the
  value references at most ``n S B x 12`` bytes (+ the mask), forward / backward peaks within
  the per-chunk bound, and the value gradients are separate allocations;
* the fp32 aggregation stays fp32 with TF32 matmuls on (the NGC default): output and every
  gradient match a float64 ``native_attn_res`` to about 1e-6 (TF32 would give ~5e-4).

The fp32 model comparisons (T-O1 fp32, T-D1, T-E1) run with full-precision GEMMs
(``NVIDIA_TF32_OVERRIDE=0``, ``allow_tf32 = False``), as the loop parity tests do.
"""

import contextlib
import dataclasses
import os

import pytest
import torch

from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    CUDA,
    VOCAB,
    determinism_env,
    model_parallel_once,
)

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(
    not CUDA or 8 % WORLD != 0, reason="DSv4 layers need CUDA (EP = W in 1, 2, 4, 8)"
)

PATTERN = "HEHECEWE"
_RATIOS = {"W": 0, "C": 4, "H": 128, "E": 0}
SEQ = 256
BATCH = 2


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    if not CUDA:
        yield
        return
    with determinism_env(), model_parallel_once(expert_model_parallel_size=WORLD):
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(False)


@contextlib.contextmanager
def _hadamard_fallback():
    from unittest.mock import patch

    try:
        from fast_hadamard_transform import hadamard_transform  # noqa: F401

        yield
    except ImportError:
        with patch(
            "megatron.core.transformer.experimental_attention_variant.csa.rotate_activation",
            lambda x: x * (x.size(-1) ** -0.5),
        ):
            yield


def _model(carrier, dtype, **overrides):
    """A DSV4-TINY-like loop HybridModel on the ``carrier`` (seeded identically)."""
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.loop.hybrid_specs import (
        loop_attn_res_hybrid_dsv4_stack_spec,
        loop_hybrid_dsv4_stack_spec,
    )
    from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    fields = dict(
        num_layers=len(PATTERN),
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
        csa_compress_ratios=[_RATIOS[symbol] for symbol in PATTERN],
        num_moe_experts=8,
        moe_ffn_hidden_size=256,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        actual_vocab_size=VOCAB,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
        use_cpu_initialization=True,
        loop_architecture="huginn",
        loop_iterations=3,
        loop_prelude_sublayers=2,
        loop_coda_sublayers=2,
        loop_carrier=carrier,
        calculate_per_token_loss=True,
        deterministic_mode=True,
        attention_backend=AttnBackend.unfused,
        expert_model_parallel_size=WORLD,
    )
    fields.update(overrides)
    config = MLATransformerConfig(**fields)
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    set_moe_metrics_tracker(MoEMetricsTracker())
    spec = (
        loop_attn_res_hybrid_dsv4_stack_spec(config)
        if carrier == "attn_res"
        else loop_hybrid_dsv4_stack_spec(config)
    )
    return HybridModel(
        config=config,
        hybrid_stack_spec=spec,
        vocab_size=VOCAB,
        max_sequence_length=1024,
        hybrid_layer_pattern=PATTERN,
        position_embedding_type="rope",
    ).cuda()


def _hidden(dtype, batch=BATCH, seed=3):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(SEQ, batch, 512, generator=generator) * 0.02
    upstream = torch.randn(SEQ, batch, 512, generator=generator)
    return x.to("cuda", dtype), upstream.to("cuda", dtype)


def _decoder_forward_backward(decoder, x, upstream, depths=None):
    x = x.detach().clone().requires_grad_(True)
    decoder.zero_grad(set_to_none=True)
    if depths is not None:
        decoder.set_loop_depths(torch.tensor(depths, device="cuda"))
    y = decoder(hidden_states=x, attention_mask=None, rotary_pos_emb=None)
    (y.float() * upstream.float()).sum().backward()
    return y.detach().float(), x.grad.detach().float()


def _copy_shared_weights(source, target):
    state = {
        name: value for name, value in source.state_dict().items() if ".loop_attn_res" not in name
    }
    target.load_state_dict(state, strict=True)


def _relative(actual, expected):
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()


@pytest.fixture
def full_precision_gemms(monkeypatch):
    """Full-precision fp32 GEMMs (no TF32) for the fp32 comparisons (hybrid_parity_utils)."""
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old_tf32


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_zero_query_matches_the_residual_carrier(dtype, k, full_precision_gemms):
    """T-O1 on DSv4 layers (the final norm of the decoder removes the 1 / n of the head)."""
    with _hadamard_fallback():
        k3 = _model("attn_res", dtype, layernorm_epsilon=1e-12, loop_attn_res_blocks_per_round=k)
        residual = _model("residual", dtype, layernorm_epsilon=1e-12)
        _copy_shared_weights(k3, residual)
        x, upstream = _hidden(dtype)
        y, dx = _decoder_forward_backward(k3.decoder, x, upstream)
        y_ref, dx_ref = _decoder_forward_backward(residual.decoder, x, upstream)
    assert torch.isfinite(y).all() and torch.isfinite(dx).all()
    if dtype == torch.float32:
        assert _relative(y, y_ref) < 1e-4
        assert _relative(dx, dx_ref) < 1e-3
    else:
        cosine = torch.nn.functional.cosine_similarity(y.flatten(), y_ref.flatten(), dim=0)
        assert cosine.item() > 0.999


def test_layer_deltas_of_dsv4_sublayers(full_precision_gemms):
    """T-D1: H / C / W attention and MoE sublayers (fp32)."""
    with _hadamard_fallback():
        model = _model("attn_res", torch.float32)
        decoder = model.decoder
        x, _ = _hidden(torch.float32)
        with torch.no_grad():
            _, ctx = decoder._prepare_forward(
                x,
                None,
                None,
                None,
                packed_seq_params=None,
                padding_mask=None,
                input_ids=None,
                packed_seq_params_by_layout=None,
                cp_layout_plan=None,
            )
            delta_ctx = dataclasses.replace(ctx, return_layer_delta=True)
            seen = set()
            for index, symbol in enumerate(decoder._loop_symbols):
                h = x * (1 + index)
                full = decoder._run_layer(index, h, ctx)
                delta = decoder._run_layer(index, h, delta_ctx)
                expected = full - h
                assert _relative(delta, expected) < 1e-4, (index, symbol)
                seen.add(symbol)
    assert seen == {"H", "C", "W", "E"}


def _capture_blocks(monkeypatch):
    from megatron.core.transformer.loop.attn_res_stack import LoopAttnResHybridStack

    captured = []
    original = LoopAttnResHybridStack._attn_res_round

    def wrapped(self, sources, step, round_ctx, masks, sink, label):
        blocks = original(self, sources, step, round_ctx, masks, sink, label)
        captured.extend((step + 1, block) for block in blocks if block.requires_grad)
        return blocks

    monkeypatch.setattr(LoopAttnResHybridStack, "_attn_res_round", wrapped)
    return captured


def test_row_exit_and_zero_gradient_for_skipped_rounds(monkeypatch, full_precision_gemms):
    """T-E1 and T-E3 with EP = W (aux losses off; fp32)."""
    depths = (1, 3)
    with _hadamard_fallback():
        model = _model("attn_res", torch.float32)
        decoder = model.decoder
        x, upstream = _hidden(torch.float32)
        captured = _capture_blocks(monkeypatch)
        grads = []
        x_leaf = x.clone().requires_grad_(True)
        decoder.set_loop_depths(torch.tensor(depths, device="cuda"))
        y = decoder(hidden_states=x_leaf, attention_mask=None, rotary_pos_emb=None)
        for round_index, block in captured:
            block.register_hook(lambda g, r=round_index: grads.append((r, g.detach().clone())))
        (y.float() * upstream.float()).sum().backward()
        monkeypatch.undo()
        assert len(grads) == max(depths)
        for round_index, grad in grads:
            for row, depth in enumerate(depths):
                nonzero = torch.count_nonzero(grad[:, row]).item()
                assert (nonzero == 0) == (round_index > depth), (round_index, row)
        y_rows, dx_rows = y.detach().float(), x_leaf.grad.float()
        for row, depth in enumerate(depths):
            saved = decoder.config.loop_iterations
            decoder.config.loop_iterations = depth
            try:
                y_row, dx_row = _decoder_forward_backward(
                    decoder, x[:, row : row + 1], upstream[:, row : row + 1]
                )
            finally:
                decoder.config.loop_iterations = saved
            assert _relative(y_rows[:, row : row + 1], y_row) < 1e-4, row
            assert _relative(dx_rows[:, row : row + 1], dx_row) < 1e-3, row


@pytest.mark.parametrize("control", [False, True], ids=["masked", "control"])
def test_aux_losses_do_not_reach_skipped_rounds(monkeypatch, control):
    """T-E4: router aux loss and DSv4 indexer loss on, the main loss x 0. Every core sublayer
    input of a round the row has left gets an exactly-zero gradient on that row; the control
    (round padding mask dropped by a test-only patch) gets a non-zero one."""
    from megatron.core.transformer.loop.attn_res_stack import LoopAttnResHybridStack

    depths = (1, 3)
    with _hadamard_fallback():
        model = _model(
            "attn_res", torch.float32, moe_aux_loss_coeff=1e-2, dsa_indexer_loss_coeff=1e-2
        )
        decoder = model.decoder
        if control:
            original = LoopAttnResHybridStack._attn_res_round

            def unmasked(self, sources, step, round_ctx, masks, sink, label):
                round_ctx = dataclasses.replace(round_ctx, padding_mask=None)
                return original(self, sources, step, round_ctx, masks, sink, label)

            monkeypatch.setattr(LoopAttnResHybridStack, "_attn_res_round", unmasked)
        roles = decoder.loop_roles
        inputs = []  # (round, sublayer input)
        calls = {}

        def record(module, args, output, index):
            calls[index] = calls.get(index, 0) + 1
            if output.requires_grad:
                inputs.append((calls[index], output))

        handles = [
            decoder.loop_attn_res[j].register_forward_hook(lambda m, a, o, j=j: record(m, a, o, j))
            for j in range(roles.core)
        ]
        try:
            x, upstream = _hidden(torch.float32)
            grads = []
            decoder.set_loop_depths(torch.tensor(depths, device="cuda"))
            y = decoder(
                hidden_states=x.requires_grad_(True), attention_mask=None, rotary_pos_emb=None
            )
            for round_index, h in inputs:
                h.register_hook(lambda g, r=round_index: grads.append((r, g.detach().clone())))
            ((y.float() * upstream.float()).sum() * 0.0).backward()
        finally:
            for handle in handles:
                handle.remove()
    skipped = [grad for round_index, grad in grads if round_index > depths[0]]
    assert skipped
    nonzero = sum(torch.count_nonzero(grad[:, 0]).item() for grad in skipped)
    if control:
        assert nonzero > 0
    else:
        assert nonzero == 0


def test_fp32_aggregation_ignores_tf32():
    """With TF32 matmuls on (as in NGC containers), the fp32 aggregation and its backward
    match a float64 ``native_attn_res`` to fp32 rounding; a control GEMM shows TF32 is on."""
    from megatron.core.transformer.loop import attn_res
    from tests.unit_tests.transformer.loop.attn_res_reference import native_attn_res

    n, sequence, batch, hidden = 7, 512, 2, 1536
    generator = torch.Generator(device="cuda").manual_seed(2)
    values = [
        torch.randn(sequence, batch, hidden, device="cuda", generator=generator) for _ in range(n)
    ]
    query = 0.05 * torch.randn(hidden, device="cuda", generator=generator)
    norm = 1 + 0.2 * torch.randn(hidden, device="cuda", generator=generator)
    upstream = torch.randn(sequence, batch, hidden, device="cuda", generator=generator)
    mask = torch.ones(n, batch, dtype=torch.bool, device="cuda")
    mask[-3:-1, 1] = False
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        a = torch.randn(256, hidden, device="cuda", generator=generator)
        control = _relative((a @ a.t()).double(), a.double() @ a.double().t())
        if control < 1e-5:
            pytest.skip(f"TF32 GEMMs are not active here (control error {control:.1e})")
        inputs = [value.clone().requires_grad_(True) for value in values]
        q = query.clone().requires_grad_(True)
        g = norm.clone().requires_grad_(True)
        out = attn_res.attn_res_aggregate(q, g, 1e-6, inputs, mask)
        grads = torch.autograd.grad(out, [q, g, *inputs], upstream)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    inputs64 = [value.double().requires_grad_(True) for value in values]
    q64 = query.double().requires_grad_(True)
    g64 = norm.double().requires_grad_(True)
    out64 = native_attn_res(inputs64, q64, g64, 1e-6, mask)
    grads64 = torch.autograd.grad(out64, [q64, g64, *inputs64], upstream.double())
    assert _relative(out.double(), out64) < 2e-6
    for index, (grad, grad64) in enumerate(zip(grads, grads64)):
        if grad64.abs().max() == 0:
            assert torch.count_nonzero(grad) == 0, index
        else:
            assert _relative(grad.double(), grad64) < 1e-5, index


def test_aggregation_memory():
    """Saved state, forward / backward peaks and gradient release of one aggregation
    (n = 10, h = 1536, S = 4096, B = 1, bf16)."""
    from megatron.core.transformer.loop import attn_res

    n, sequence, batch, hidden = 10, 4096, 1, 1536
    generator = torch.Generator(device="cuda").manual_seed(1)
    values = [
        torch.randn(
            sequence, batch, hidden, device="cuda", dtype=torch.bfloat16, generator=generator
        ).requires_grad_(True)
        for _ in range(n)
    ]
    query = torch.randn(hidden, device="cuda", generator=generator).requires_grad_(True)
    norm = torch.ones(hidden, device="cuda").requires_grad_(True)
    mask = torch.ones(n, batch, dtype=torch.bool, device="cuda")
    value_bytes = sequence * batch * hidden * 2
    chunk = attn_res.attn_res_chunk_tokens(n, batch, hidden, 4)
    chunk = min(chunk, sequence)
    # Stacked copies (bf16 + fp32) and a few fp32 [chunk, B, H] buffers (the source mix, the
    # upcast upstream gradient, one per-source product).
    transient = 6 * n * chunk * batch * hidden + 12 * chunk * batch * hidden
    slack = 32 * 2**20

    torch.cuda.synchronize()
    torch.empty(1, device="cuda")
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = attn_res.attn_res_aggregate(query, norm, 1e-6, values, mask)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    forward_peak = torch.cuda.max_memory_allocated() - base
    saved = after - base - value_bytes  # beyond the output
    assert saved <= n * sequence * batch * 12 + 2**20, saved
    assert forward_peak <= value_bytes + saved + transient + slack, forward_peak

    upstream = torch.randn_like(out)
    torch.cuda.synchronize()
    torch.empty(1, device="cuda")
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    grads = torch.autograd.grad(out, values, upstream)
    torch.cuda.synchronize()
    backward_peak = torch.cuda.max_memory_allocated() - base
    assert backward_peak <= n * value_bytes + transient + slack, backward_peak
    assert len({grad.untyped_storage().data_ptr() for grad in grads}) == n
    held = torch.cuda.memory_allocated()
    keep = grads[0]
    del grads
    torch.cuda.synchronize()
    released = held - torch.cuda.memory_allocated()
    assert released >= (n - 1) * value_bytes, released
    assert keep.shape == values[0].shape
