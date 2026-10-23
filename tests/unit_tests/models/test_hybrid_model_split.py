# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""HybridModel stages: ``forward`` equals ``embed`` -> ``rope_ctx`` -> decoder -> ``head_loss``.

The GPU tests build real models and compare ``model(...)`` with the stages run by hand,
bitwise on the loss (or logits), on every parameter gradient and, with precomputed
embeddings, on the input gradient, with and without MTP. The ``rope_ctx`` dispatch tests
run on CPU against a namespace stub.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.hybrid.hybrid_model import HybridModel

_CUDA = torch.cuda.is_available()
_SEQ, _BATCH, _VOCAB = 32, 2, 128

gpu = pytest.mark.skipif(not _CUDA, reason="HybridModel forward needs a GPU")


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    """Initialize model parallel once per module (CI process-group budget), GPU only."""
    if not _CUDA:
        yield
        return
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from tests.unit_tests.test_utilities import Utils

    deterministic_before = torch.are_deterministic_algorithms_enabled()
    with pytest.MonkeyPatch.context() as env:
        # TE attention reads both at construction; the models are built inside the tests.
        env.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        torch.use_deterministic_algorithms(True, warn_only=True)
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        try:
            yield
        finally:
            Utils.destroy_model_parallel()
            torch.use_deterministic_algorithms(deterministic_before)


# ---------------------------------------------------------------------------
# GPU: forward == stages
# ---------------------------------------------------------------------------

_MOE = dict(
    num_moe_experts=4,
    moe_ffn_hidden_size=128,
    moe_router_topk=2,
    moe_router_dtype="fp32",
    moe_router_load_balancing_type="aux_loss",
    moe_aux_loss_coeff=1e-2,
    moe_token_dispatcher_type="alltoall",
)
_MHC = dict(enable_mhc_connections=True, mhc_num_residual_streams=4, mhc_sinkhorn_iterations=5)


def _build_model(pattern, **overrides):
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.enums import AttnBackend

    main, *mtp = pattern.split("/")
    if mtp:
        overrides.setdefault("mtp_num_layers", len(mtp))
        overrides.setdefault("mtp_loss_scaling_factor", 0.1)
    config = TransformerConfig(
        num_layers=len(main),
        hidden_size=256,
        num_attention_heads=4,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        deterministic_mode=True,
        # fp32: unfused is TE's only fp32 backend, and it agrees with the NVTE_FLASH_ATTN=0 /
        # NVTE_FUSED_ATTN=0 that tests/unit_tests/conftest.py sets for every test.
        attention_backend=AttnBackend.unfused,
        **overrides,
    )
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=_VOCAB,
        max_sequence_length=_SEQ,
        hybrid_layer_pattern=pattern,
        position_embedding_type="rope",
        parallel_output=True,
    ).cuda()
    model.train()
    return model


def _batch():
    generator = torch.Generator().manual_seed(7)
    input_ids = torch.randint(0, _VOCAB, (_BATCH, _SEQ), generator=generator).cuda()
    labels = torch.randint(0, _VOCAB, (_BATCH, _SEQ), generator=generator).cuda()
    position_ids = torch.arange(_SEQ, dtype=torch.int64).repeat(_BATCH, 1).cuda()
    attention_mask = torch.triu(
        torch.ones(_BATCH, 1, _SEQ, _SEQ, dtype=torch.bool), diagonal=1
    ).cuda()
    loss_mask = torch.ones(_BATCH, _SEQ, dtype=torch.float32).cuda()
    return dict(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
        loss_mask=loss_mask,
    )


def _staged_forward(model, *, input_ids, position_ids, attention_mask, decoder_input=None, **head):
    """The stages of HybridModel.forward, run by hand (training, no CP / packing)."""
    hidden, decoder_padding_mask, hash_input_ids = model.embed(
        input_ids, position_ids, decoder_input=decoder_input
    )
    rotary_pos_emb = model.rope_ctx(hidden, position_ids)
    output = model.decoder(
        hidden_states=hidden,
        attention_mask=attention_mask,
        inference_context=None,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=None,
        padding_mask=decoder_padding_mask,
        packed_seq_params_by_layout=None,
        cp_layout_plan=None,
        input_ids=hash_input_ids,
    )
    hidden_states, multistream = output if isinstance(output, tuple) else (output, None)
    return model.head_loss(
        hidden_states,
        multistream,
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        mtp_decoder_input=decoder_input,
        rotary_pos_emb=rotary_pos_emb,
        **head,
    )


def _run(model, staged, batch, *, precomputed=None, **kwargs):
    model.zero_grad(set_to_none=True)
    decoder_input = None
    if precomputed is not None:
        decoder_input = precomputed.detach().clone().requires_grad_(True)
    torch.manual_seed(99)
    if staged:
        output = _staged_forward(model, decoder_input=decoder_input, **batch, **kwargs)
    else:
        output = model(decoder_input=decoder_input, **batch, **kwargs)
    cotangent = torch.randn(output.shape, generator=torch.Generator().manual_seed(5)).to(output)
    (output.float() * cotangent.float()).sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    input_grad = None if decoder_input is None else decoder_input.grad
    return output.detach().clone(), grads, input_grad


def _first_difference(a, b):
    (out_a, grads_a, dx_a), (out_b, grads_b, dx_b) = a, b
    if out_a.shape != out_b.shape or not torch.equal(out_a, out_b):
        return "output"
    if grads_a.keys() != grads_b.keys():
        return f"grad keys {sorted(grads_a.keys() ^ grads_b.keys())}"
    for name, grad in grads_a.items():
        if not torch.equal(grad, grads_b[name]):
            return f"grad {name}"
    if (dx_a is None) != (dx_b is None) or (dx_a is not None and not torch.equal(dx_a, dx_b)):
        return "input grad"
    return None


def _assert_same(model, batch, **kwargs):
    forward = _run(model, False, batch, **kwargs)
    difference = _first_difference(forward, _run(model, True, batch, **kwargs))
    if difference is not None:
        # Tell a broken split apart from a nondeterministic kernel.
        rerun = _first_difference(forward, _run(model, False, batch, **kwargs))
        assert rerun is None, f"model forward is not self-deterministic here ({rerun})"
    assert difference is None, f"forward and the stages differ in {difference}"
    output, grads, _ = forward
    return output, grads


@gpu
@pytest.mark.parametrize(
    "pattern,overrides",
    [
        pytest.param("*-*-", {}, id="dense"),
        pytest.param("*E*E/*E", _MOE, id="moe-mtp"),
        pytest.param("*-*-/*-", _MHC, id="mhc-mtp"),
    ],
)
def test_forward_equals_stages(pattern, overrides):
    model = _build_model(pattern, **overrides)
    batch = _batch()
    output, grads = _assert_same(model, batch)
    assert output.shape == (_BATCH, _SEQ)
    if "/" in pattern:
        assert any(name.startswith("mtp.") for name in grads), "MTP did not get gradients"


@gpu
def test_forward_equals_stages_with_precomputed_embeddings():
    model = _build_model("*E*E/*E", **_MOE)
    batch = _batch()
    precomputed = model.embedding(input_ids=batch["input_ids"], position_ids=batch["position_ids"])
    _, grads = _assert_same(model, batch, precomputed=precomputed)
    assert any(name.startswith("mtp.") for name in grads)


@gpu
def test_forward_equals_stages_logits_without_labels():
    model = _build_model("*E*E/*E", **_MOE)
    batch = _batch()
    batch["labels"] = None
    output, _ = _assert_same(model, batch)
    assert output.shape == (_BATCH, _SEQ, _VOCAB)


@gpu
def test_forward_equals_stages_without_mtp_loss():
    model = _build_model("*E*E/*E", **_MOE)
    batch = _batch()
    _, grads = _assert_same(model, batch, compute_mtp_loss=False)
    assert not any(name.startswith("mtp.") for name in grads)


@gpu
def test_rope_ctx_builds_model_rope():
    model = _build_model("*-")
    batch = _batch()
    hidden, _, _ = model.embed(batch["input_ids"], batch["position_ids"])
    rope = model.rope_ctx(hidden, batch["position_ids"])
    assert rope is not None
    assert torch.equal(rope, model.rotary_pos_emb(_SEQ, packed_seq=False, cp_group=None))


# ---------------------------------------------------------------------------
# CPU: rope_ctx dispatch on a namespace stub
# ---------------------------------------------------------------------------


class _RecordingRope:
    def __init__(self, returns):
        self.returns = returns
        self.calls = []

    def get_rotary_seq_len(self, inference_context, decoder, decoder_input, config, packed):
        return decoder_input.shape[0]

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.returns


def _rope_stub(position_embedding_type, multi_latent_attention, returns):
    return SimpleNamespace(
        position_embedding_type=position_embedding_type,
        config=SimpleNamespace(multi_latent_attention=multi_latent_attention),
        rotary_pos_emb=_RecordingRope(returns),
        decoder=None,
    )


def test_rope_ctx_is_none_when_attention_owns_rope():
    """MLA attention (DSv4 included) applies RoPE itself: no model-level embedding."""
    model = _rope_stub("rope", multi_latent_attention=True, returns=torch.ones(1))
    assert HybridModel.rope_ctx(model, torch.zeros(4, 1, 8), None) is None
    assert model.rotary_pos_emb.calls == []


def test_rope_ctx_is_none_without_rope():
    model = _rope_stub("none", multi_latent_attention=False, returns=torch.ones(1))
    assert HybridModel.rope_ctx(model, torch.zeros(4, 1, 8), None) is None
    assert model.rotary_pos_emb.calls == []


def test_rope_ctx_builds_rope_for_regular_attention():
    emb = torch.ones(4, 1, 1, 8)
    model = _rope_stub("rope", multi_latent_attention=False, returns=emb)
    assert HybridModel.rope_ctx(model, torch.zeros(4, 1, 8), None) is emb
    assert model.rotary_pos_emb.calls == [((4,), dict(packed_seq=False, cp_group=None))]


def test_rope_ctx_builds_yarn_even_under_mla():
    emb = torch.ones(4, 1, 1, 8)
    model = _rope_stub("yarn", multi_latent_attention=True, returns=(emb, 1.0))
    assert HybridModel.rope_ctx(model, torch.zeros(4, 1, 8), None) is emb
    assert len(model.rotary_pos_emb.calls) == 1
