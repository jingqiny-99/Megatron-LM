# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``--loop-carrier-diagnostics`` (``megatron/training/loop_diagnostics.py``): enabling on the
decoder, the summary metrics through the loop reporting rule, the ``full`` per-sublayer record
(collected on TensorBoard-interval iterations only, one all-reduce, the JSONL file, the
TensorBoard scalars with a fake writer), the argument rules, and the hidden-state RMS of the mHC
variants (T-M1: the plain-mhc boundary hooks of ``--loop-carrier-diagnostics-segments`` and the
loop-mhc round state, each against an independent hook). CPU (gloo, one rank) or CUDA; the mHC
cases need ``TORCHDYNAMO_DISABLE=1`` on CPU.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
    indexer_loss_padding_mask_in_scope,
)
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.loop.attn_res_stack import LoopAttnResHybridStack
from megatron.core.transformer.loop.reporting import loop_metric_report, stream_mean_rms
from megatron.training import arguments
from megatron.training.loop_diagnostics import (
    FORMAT,
    boundary_sublayers,
    enable_loop_carrier_diagnostics,
    log_loop_carrier_diagnostics,
    loop_carrier_diagnostics_scalars,
    record_loop_carrier_diagnostics,
)
from tests.unit_tests.transformer.loop import hybrid_loop_utils
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    BATCH,
    SEQ,
    VOCAB,
    build_local_loop_model,
    dense_config,
    device,
    local_model_spec,
    model_parallel_once,
)

mhc_on_this_device = hybrid_loop_utils.mhc_on_this_device

DENSE = "*-*-*-*-"


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


def _model(carrier="attn_res", **overrides):
    module = LoopAttnResHybridStack if carrier == "attn_res" else None
    kwargs = dict(loop_carrier=carrier, **overrides)
    if module is None:
        return build_local_loop_model(DENSE, **kwargs)
    return build_local_loop_model(DENSE, module, **kwargs)


def _step(model, seed=5, padding_mask=None):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, VOCAB, (BATCH, SEQ), generator=generator).to(device())
    labels = torch.randint(0, VOCAB, (BATCH, SEQ), generator=generator).to(device())
    position_ids = torch.arange(SEQ, device=device()).repeat(BATCH, 1)
    kwargs = {} if padding_mask is None else dict(padding_mask=padding_mask)
    loss = model(tokens, position_ids, None, labels=labels, **kwargs)
    loss.sum().backward()


class FakeWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, name, value, iteration):
        self.scalars[(name, iteration)] = value


def _args(mode, interval=2, path=None):
    return SimpleNamespace(
        loop_carrier_diagnostics=mode,
        tensorboard_log_interval=interval,
        loop_carrier_diagnostics_path=path,
    )


def test_off_leaves_the_decoder_alone():
    model = _model()
    enable_loop_carrier_diagnostics(_args("off"), model)
    _step(model)
    assert not any(name.startswith("loop attnres") for name in model.decoder.loop_metrics)


@pytest.mark.parametrize("carrier", ["attn_res", "residual"])
def test_summary_metrics_reach_the_loop_report(carrier):
    model = _model(carrier)
    enable_loop_carrier_diagnostics(_args("summary"), model)
    _step(model)
    decoder = model.decoder
    report = loop_metric_report(
        decoder.loop_metrics, issued_tokens=BATCH * SEQ, populations=decoder.loop_metric_populations
    )
    rounds = decoder.config.loop_iterations
    names = {"loop state rms prelude", *(f"loop state rms r{r}" for r in range(1, rounds + 1))}
    if carrier == "attn_res":
        names |= {f"loop attnres weight r{r} e" for r in range(1, rounds + 1)}
        names |= {"loop attnres weight head current", "loop attnres weight coda e"}
    assert names <= set(report)
    for name in names:
        assert report[name][1].item() == BATCH * SEQ
        assert torch.isfinite(report[name][0])
    # The summary takes no per-sublayer record and accumulates no per-sublayer sums.
    assert record_loop_carrier_diagnostics(_args("summary"), model, 2) is None
    if carrier == "attn_res":
        assert decoder.take_carrier_diagnostics()["weights"] == {}


def test_full_records_on_tensorboard_iterations_only(tmp_path, monkeypatch):
    path = tmp_path / "carrier.jsonl"
    args = _args("full", interval=2, path=str(path))
    model = _model()
    enable_loop_carrier_diagnostics(args, model)
    reduces = []
    original = torch.distributed.all_reduce
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, *a, **k: (reduces.append(tensor.numel()), original(tensor, *a, **k))[1],
    )
    _step(model)
    reduces.clear()  # the model's own collectives
    assert record_loop_carrier_diagnostics(args, model, 1) is None
    assert reduces == []
    _step(model, seed=6)
    reduces.clear()
    record = record_loop_carrier_diagnostics(args, model, 2)
    assert len(reduces) == 1
    assert record["format"] == FORMAT and record["iteration"] == 2 and record["k"] == 1
    decoder = model.decoder
    roles = decoder.loop_roles
    rounds = decoder.config.loop_iterations
    keys = {(segment, sublayer, r) for segment, sublayer, r, _ in record["entries"]}
    assert len(keys) == roles.core * rounds + roles.coda + 1
    for _, _, _, means in record["entries"]:
        assert sum(means) == pytest.approx(1.0)
    lines = path.read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0]) == json.loads(json.dumps(record))
    # Taken: the next record starts from zero.
    assert decoder.take_carrier_diagnostics()["weights"] == {}
    writer = FakeWriter()
    log_loop_carrier_diagnostics(args, 2, writer, None)
    scalars = loop_carrier_diagnostics_scalars(record)
    assert {name for name, _ in writer.scalars} == set(scalars)
    # The first core sublayer of round 1 reads e only; at w = 0 the head is uniform.
    assert writer.scalars[("loop attnres/core/sub000/r1/e", 2)] == pytest.approx(1.0)
    n_head = 1 + rounds + 1
    assert writer.scalars[(f"loop attnres/head/sub000/r{rounds}/e", 2)] == pytest.approx(1 / n_head)
    writer = FakeWriter()
    log_loop_carrier_diagnostics(args, 4, writer, None)  # no record for iteration 4
    assert writer.scalars == {}


def test_full_without_per_sublayer_statistics_records_nothing():
    model = _model("residual")
    args = _args("full")
    enable_loop_carrier_diagnostics(args, model)
    _step(model)
    assert record_loop_carrier_diagnostics(args, model, 2) is None


def _arg_namespace(**overrides):
    fields = dict(
        loop_carrier_diagnostics="summary",
        loop_carrier_diagnostics_path=None,
        loop_architecture="huginn",
        loop_prelude_sublayers=2,
        loop_coda_sublayers=2,
        loop_schedule="standard",
        loop_sample_depth=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    "overrides,match",
    [
        (dict(loop_carrier_diagnostics_path="x.jsonl"), "needs --loop-carrier-diagnostics full"),
        (dict(loop_schedule="rsbf"), "eager hybrid Huginn loop"),
        (dict(loop_sample_depth=True), "eager hybrid Huginn loop"),
        (dict(loop_architecture="mor"), "eager hybrid Huginn loop"),
        (dict(loop_prelude_sublayers=None, loop_coda_sublayers=None), "eager hybrid Huginn"),
    ],
)
def test_argument_rules(overrides, match):
    with pytest.raises(ValueError, match=match):
        arguments._validate_loop_carrier_diagnostics(_arg_namespace(**overrides))


def test_argument_rules_accept():
    arguments._validate_loop_carrier_diagnostics(_arg_namespace())
    arguments._validate_loop_carrier_diagnostics(
        _arg_namespace(loop_carrier_diagnostics="full", loop_carrier_diagnostics_path="x.jsonl")
    )
    arguments._validate_loop_carrier_diagnostics(_arg_namespace(loop_carrier_diagnostics="off"))


def test_enable_needs_a_loop_decoder():
    with pytest.raises(ValueError, match="hybrid loop decoder"):
        enable_loop_carrier_diagnostics(_args("summary"), SimpleNamespace(decoder=object()))


@pytest.mark.parametrize(
    "overrides,match",
    [
        (
            dict(loop_carrier_diagnostics="off", loop_carrier_diagnostics_segments=[2, 2, 2]),
            "segments needs --loop-carrier-diagnostics summary or full",
        ),
        (dict(loop_carrier_diagnostics_segments=[2, 2, 2]), "for a non-loop hybrid model"),
        (
            dict(loop_architecture="none", loop_prelude_sublayers=None, loop_coda_sublayers=None),
            "or a non-loop hybrid model with --loop-carrier-diagnostics-segments",
        ),
        (
            dict(
                loop_architecture="none",
                loop_carrier_diagnostics_segments=[2, 2, 2],
                hybrid_layer_pattern=None,
            ),
            "needs a hybrid model",
        ),
        (
            dict(
                loop_architecture="none",
                loop_carrier_diagnostics_segments=[2, 0, 2],
                hybrid_layer_pattern="*-",
            ),
            "ROUNDS >= 1",
        ),
    ],
)
def test_segment_argument_rules(overrides, match):
    with pytest.raises(ValueError, match=match):
        arguments._validate_loop_carrier_diagnostics(_arg_namespace(**overrides))


def test_segment_argument_rules_accept_a_non_loop_hybrid_model():
    arguments._validate_loop_carrier_diagnostics(
        _arg_namespace(
            loop_architecture="none",
            loop_carrier_diagnostics_segments=[2, 4, 2],
            hybrid_layer_pattern="*-*-",
        )
    )


def test_boundary_sublayers():
    assert boundary_sublayers(12, 2, 2, 2) == [1, 5, 9]
    assert boundary_sublayers(10, 0, 2, 2) == [-1, 3, 7]
    with pytest.raises(ValueError, match="not a positive multiple of 3"):
        boundary_sublayers(12, 2, 3, 2)


MHC = dict(enable_mhc_connections=True, mhc_num_residual_streams=4, mhc_sinkhorn_iterations=5)
_STREAMS = 4


def _independent_stream_mean_rms(x, valid=None):
    """RMS of the mean of the four h-wide streams over the ``valid`` ([S, B] bool) tokens (an
    independent formula for T-M1)."""
    streams = x.detach().double().reshape(*x.shape[:-1], _STREAMS, -1)
    mean = streams.mean(-2)
    if valid is not None:
        mean = mean[valid]
    return mean.square().mean().sqrt()


def _diverge_streams(model, seed=3):
    """Make the mHC streams differ (seeded random static biases, unit alphas): at init the
    streams of these small models are equal up to ~4e-5, so the stream mean, one stream and
    the full-vector RMS would agree to the test tolerance."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, HyperConnectionModule):
                module.bias.copy_(torch.randn(module.bias.shape, generator=generator))
                module.alpha_res.fill_(1.0)
                module.alpha_post.fill_(1.0)


def _assert_has_teeth(x, valid=None):
    """The stream mean's RMS differs from the full-vector RMS and from the first stream's RMS by
    far more than the 1e-5 comparison tolerance."""
    expected = _independent_stream_mean_rms(x, valid).item()
    flat = x.detach().double()
    first = flat[..., : flat.shape[-1] // _STREAMS]
    if valid is not None:
        flat, first = flat[valid], first[valid]
    for other in (flat.square().mean().sqrt().item(), first.square().mean().sqrt().item()):
        assert abs(other / expected - 1) > 1e-4, (other, expected)


def test_stream_mean_rms_definition():
    """``stream_mean_rms`` on a constructed [S, B, 4h] tensor with distinct streams and a mask:
    the RMS of the stream mean over the valid tokens, and their count."""
    generator = torch.Generator().manual_seed(0)
    h = 3
    x = torch.randn(5, 2, _STREAMS * h, generator=generator)
    x[..., h : 2 * h] += 2.0  # stream 1 shifted: the mean differs from every single stream
    valid = torch.ones(5, 2, dtype=torch.bool)
    valid[3:, 1] = False
    value, tokens = stream_mean_rms(x, _STREAMS, valid)
    assert tokens.item() == 8
    _assert_has_teeth(x, valid)
    assert value.double().item() == pytest.approx(_independent_stream_mean_rms(x, valid).item())
    value, tokens = stream_mean_rms(x, _STREAMS)
    assert tokens.item() == 10
    assert value.double().item() == pytest.approx(_independent_stream_mean_rms(x).item())
    # One stream: the plain RMS; no valid token: 0.
    assert stream_mean_rms(x, 1)[0].item() == pytest.approx(x.square().mean().sqrt().item())
    empty = stream_mean_rms(x, _STREAMS, torch.zeros(5, 2, dtype=torch.bool))
    assert empty[0].item() == 0.0 and empty[1].item() == 0


def _plain_mhc_model(pattern):
    """A non-loop ``HybridModel`` (plain HybridStack) of local layers with 4 mHC streams."""
    torch.manual_seed(1234)
    config = dense_config(len(pattern), is_hybrid_model=True, **MHC)
    model = HybridModel(
        config=config,
        hybrid_stack_spec=local_model_spec(HybridStack, hyper_connections=True),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
    )
    return model.to(device())


@pytest.mark.parametrize("padded", [False, True])
def test_plain_mhc_boundary_state_rms(mhc_on_this_device, padded):
    """T-M1 (plain-mhc): the boundary hooks' ``loop state rms`` values equal the RMS of the
    stream mean of the boundary sublayers' outputs over the non-padding tokens, captured by an
    independent hook. The streams are made distinct first (``_diverge_streams``), so the check
    tells the stream mean from the full-vector RMS and from one stream; with ``padded`` the
    last 3 tokens of row 1 are padding (SBHD document packing) and are left out."""
    pattern = "*-" + "*-*-" * 2 + "*-"
    model = _plain_mhc_model(pattern)
    _diverge_streams(model)
    padding_mask = None
    valid = None
    if padded:
        padding_mask = torch.zeros(BATCH, SEQ, dtype=torch.bool, device=device())
        padding_mask[1, -3:] = True
        valid = ~padding_mask.transpose(0, 1)
    args = SimpleNamespace(
        loop_carrier_diagnostics="summary",
        loop_carrier_diagnostics_segments=[2, 2, 2],
        tensorboard_log_interval=2,
        loop_carrier_diagnostics_path=None,
    )
    enable_loop_carrier_diagnostics(args, model)
    captured = {}
    for index in (1, 5, 9):
        model.decoder.layers[index].register_forward_hook(
            lambda module, inputs, output, index=index: captured.__setitem__(index, output[0])
        )
    model.train()
    _step(model, padding_mask=padding_mask)
    metrics = model.decoder.loop_metrics
    assert sorted(metrics) == ["loop state rms prelude", "loop state rms r1", "loop state rms r2"]
    population = SEQ * BATCH - (3 if padded else 0)
    for name, index in (
        ("loop state rms prelude", 1),
        ("loop state rms r1", 5),
        ("loop state rms r2", 9),
    ):
        assert captured[index].shape[-1] == _STREAMS * model.config.hidden_size
        _assert_has_teeth(captured[index], valid)
        if padded:  # the padded tokens change the value: the exclusion is visible
            unmasked = _independent_stream_mean_rms(captured[index]).item()
            masked = _independent_stream_mean_rms(captured[index], valid).item()
            assert abs(unmasked / masked - 1) > 1e-4, (unmasked, masked)
        expected = _independent_stream_mean_rms(captured[index], valid)
        assert metrics[name].double().item() == pytest.approx(expected.item(), rel=1e-5), name
        assert model.decoder.loop_metric_populations[name].item() == population
    if padded:
        return
    # The loss report treats them as loop metrics (a mean over the population).
    report = loop_metric_report(
        metrics, issued_tokens=SEQ * BATCH, populations=model.decoder.loop_metric_populations
    )
    assert report["loop state rms r1"][1].item() == SEQ * BATCH
    # A new dict every forward; evaluation records nothing.
    first = model.decoder.loop_metrics
    model.eval()
    with torch.no_grad():
        generator = torch.Generator().manual_seed(1)
        tokens = torch.randint(0, VOCAB, (BATCH, SEQ), generator=generator).to(device())
        position_ids = torch.arange(SEQ, device=device()).repeat(BATCH, 1)
        model(tokens, position_ids, None)
    assert model.decoder.loop_metrics == {} and model.decoder.loop_metrics is not first


def test_plain_mhc_boundary_state_rms_checks_the_segments(mhc_on_this_device):
    model = _plain_mhc_model("*-" + "*-*-" * 2 + "*-")
    args = SimpleNamespace(
        loop_carrier_diagnostics="summary", loop_carrier_diagnostics_segments=[2, 3, 2]
    )
    with pytest.raises(ValueError, match="not a positive multiple of 3"):
        enable_loop_carrier_diagnostics(args, model)


def test_loop_mhc_round_state_rms_is_the_round_output_stream_mean(mhc_on_this_device):
    """T-M1 (loop-mhc at R = 1): ``loop state rms r1`` equals the RMS of the stream mean of the
    round output before the loop norm (the loop norm's input, captured by an independent
    pre-hook)."""
    model = build_local_loop_model(DENSE, loop_iterations=1, **MHC)
    _diverge_streams(model)
    enable_loop_carrier_diagnostics(_args("summary"), model)
    captured = []
    model.decoder.loop_norm.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0])
    )
    model.train()
    _step(model)
    assert len(captured) == 1
    _assert_has_teeth(captured[0])
    expected = _independent_stream_mean_rms(captured[0])
    value = model.decoder.loop_metrics["loop state rms r1"]
    assert value.double().item() == pytest.approx(expected.item(), rel=1e-5)


@pytest.mark.parametrize("flag", [False, True])
def test_hybrid_stack_indexer_loss_row_mask(flag):
    """A loop-less ``HybridStack`` with ``indexer_loss_row_mask`` (``hybrid_builder`` sets it
    under SBHD document packing) runs its layers inside ``indexer_loss_padding_mask`` of the
    forward's padding mask, so DSv4 indexers do not train on the row-tail padding; without the
    flag, or without a padding mask, nothing is in scope (upstream behaviour)."""
    torch.manual_seed(1234)
    pattern = "*-*-"
    config = dense_config(len(pattern), is_hybrid_model=True)
    model = HybridModel(
        config=config,
        hybrid_stack_spec=local_model_spec(HybridStack),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
    ).to(device())
    assert not model.decoder.indexer_loss_row_mask  # off by default
    model.decoder.indexer_loss_row_mask = flag
    seen = []
    for layer in model.decoder.layers:
        layer.register_forward_pre_hook(
            lambda module, inputs: seen.append(indexer_loss_padding_mask_in_scope())
        )
    padding_mask = torch.zeros(BATCH, SEQ, dtype=torch.bool, device=device())
    padding_mask[0, -2:] = True
    model.train()
    _step(model, padding_mask=padding_mask)
    assert len(seen) == len(pattern)
    for mask in seen:
        if flag:
            assert mask is not None and torch.equal(mask, padding_mask)
        else:
            assert mask is None
    assert indexer_loss_padding_mask_in_scope() is None  # restored after the forward
    seen.clear()
    _step(model)  # no padding mask: nothing in scope
    assert seen == [None] * len(pattern)
