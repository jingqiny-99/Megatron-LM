# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the hybrid loop program and its cells (``loop/program.py``, ``loop/cells.py``).

The cell path (prelude, ``d`` core rounds, coda with the head) is checked against the eager
``HybridModel`` + ``LoopHybridStack`` forward at depth ``d`` per sample: per-token losses,
loss sums, token counts and reports bitwise; run as one autograd graph, the parameter
gradients bitwise; and run as the executor's backward items on fresh leaves (CODA, CORE d..1
in RECOMPUTE, PRELUDE), the parameter gradients at max-relative 1e-5. Dense, MTP, RoPE and
4-stream mHC (with and without MTP) models of local layers run on CPU (gloo) and CUDA; the
DSv4 cell oracle is CUDA-only. Further cases: port checks, the per-row self-carry (``alive``,
sweep and recompute), the exit-criterion contract (an online criterion's chain included), the
after-exit counter, the counter-based sampled depths, ``head_loss`` with the CP = 1 batch, the
purity of ``_prepare_forward``, the backward plans (the whole-round, block-checkpoint and nested
item forms, chosen by config) and the options the program rejects.
"""

import contextlib
import dataclasses
import math
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from megatron.core.context_parallel.utils import get_batches_on_this_cp_rank
from megatron.core.transformer.loop import guards
from megatron.core.transformer.loop.cells import RoundOutputs
from megatron.core.transformer.loop.guards import LoopRoles
from megatron.core.transformer.loop.program import (
    CORE_PORTS,
    PRELUDE_PORTS,
    CoreBackwardPlan,
    CoreBlock,
    FixedDepthCriterion,
    LoopPhase,
    LoopProgram,
    PrescribedDepthCriterion,
    SampledDepthCriterion,
    SampleInputs,
    SampleKey,
    block_boundaries_split_pairs,
    build_core_plan,
    build_exit_criterion,
    counter_seed,
    find_port,
    global_sample_index,
    validate_depth_probs,
)
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
from tests.unit_tests.transformer.loop import hybrid_loop_utils
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    CUDA,
    HIDDEN,
    SEQ,
    VOCAB,
    build_local_loop_model,
    device,
    model_parallel_once,
)
from tests.unit_tests.transformer.loop.rsbf_test_criteria import (
    ConstantExitCriterion,
    HaltingHead,
    OnlineHaltingCriterion,
)

# The CPU mHC, MTP and RoPE fixtures, made visible to pytest in this module.
mhc_on_this_device = hybrid_loop_utils.mhc_on_this_device
cpu_mtp = hybrid_loop_utils.cpu_mtp
cpu_rope = hybrid_loop_utils.cpu_rope

DENSE = "*-*-*-*-"  # prelude '*-', core '*-*-', coda '*-'
MHC = dict(enable_mhc_connections=True, mhc_num_residual_streams=4, mhc_sinkhorn_iterations=5)
RSBF = dict(loop_schedule="rsbf", calculate_per_token_loss=True)
R_MAX = 3
DEPTHS = (1, 3, 2)

gpu = pytest.mark.skipif(not CUDA, reason="TE / MoE / DSv4 layers need CUDA")


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


@pytest.fixture
def deterministic_algorithms(monkeypatch):
    """Deterministic kernels for the bitwise GPU comparisons (set after the module's
    model-parallel initialization, which clears the NVTE variables)."""
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingLoss:
    """``pretrain_hybrid.loss_func``'s default branch without validation; keeps the last
    per-token loss it saw."""

    def __init__(self):
        self.output = None

    def __call__(self, loss_mask, output):
        self.output = output.detach().clone()
        losses = output.view(-1).float()
        mask = loss_mask.view(-1).float()
        loss = torch.sum(losses * mask)
        num_tokens = mask.sum().clone().detach().to(torch.int)
        return (
            loss,
            num_tokens,
            {"lm loss": torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])},
        )


def _batch(rows, *, seq=SEQ, vocab=VOCAB, dev=None, seed=5, padded=False):
    """``rows`` samples of random tokens and labels; the loss mask drops a few tokens; with
    ``padded`` the last quarter of every odd row is padding."""
    dev = dev or device()
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, vocab, (rows, seq), generator=generator)
    labels = torch.randint(0, vocab, (rows, seq), generator=generator)
    loss_mask = (torch.rand(rows, seq, generator=generator) > 0.2).float()
    padding_mask = None
    if padded:
        padding_mask = torch.zeros(rows, seq, dtype=torch.bool)
        padding_mask[1::2, -(seq // 4) :] = True
        loss_mask = loss_mask * (~padding_mask).float()
    return dict(
        tokens=tokens.to(dev),
        labels=labels.to(dev),
        position_ids=torch.arange(seq).repeat(rows, 1).to(dev),
        loss_mask=loss_mask.to(dev),
        padding_mask=None if padding_mask is None else padding_mask.to(dev),
    )


def _rows(batch, start, stop):
    return {name: None if value is None else value[start:stop] for name, value in batch.items()}


def _sample(batch, start, stop=None, *, iteration=0):
    rows = _rows(batch, start, start + 1 if stop is None else stop)
    return SampleInputs(
        key=SampleKey(global_index=start, iteration=iteration),
        tokens=rows["tokens"],
        position_ids=rows["position_ids"],
        labels=rows["labels"],
        loss_mask=rows["loss_mask"],
        attention_mask=None,
        padding_mask=rows["padding_mask"],
        loss_func=RecordingLoss(),
    )


def _eager(model, batch, row, depth):
    """The eager reference: the model forward at per-row depth ``depth`` on one row."""
    rows = _rows(batch, row, row + 1)
    model.decoder.set_loop_depths(torch.tensor([depth], device=rows["tokens"].device))
    output = model(
        rows["tokens"],
        rows["position_ids"],
        None,
        labels=rows["labels"],
        loss_mask=rows["loss_mask"],
        padding_mask=rows["padding_mask"],
    )
    loss_func = RecordingLoss()
    loss, num_tokens, report = loss_func(rows["loss_mask"], output)
    return loss_func.output, loss, num_tokens, report


def _exited(rows, value=0):
    return torch.full((rows,), value, dtype=torch.int32, device=device())


def _cells(program, sample, depth):
    """Prelude, ``depth`` sweep rounds and the coda as one graph (no detach)."""
    prelude = program.prelude(sample, phase=LoopPhase.SWEEP)
    depth_port = torch.full((sample.rows,), depth, dtype=torch.int32, device=device())
    state, crit_state, rounds = None, prelude.crit_state, []
    for r in range(1, depth + 1):
        out = program.core(
            state,
            prelude.injection,
            sample,
            r=r,
            phase=LoopPhase.SWEEP,
            crit_state=crit_state,
            exited=_exited(sample.rows),
            depth=depth_port,
        )
        rounds.append(out)
        state, crit_state = out.state, out.crit_state
    return prelude, rounds, program.coda(state, sample)


def _grads(model):
    grads = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    model.zero_grad(set_to_none=True)
    return grads


def _assert_grads_bitwise(actual, expected):
    assert actual.keys() == expected.keys()
    for name, grad in expected.items():
        assert torch.equal(actual[name], grad), name


def _model(name, request, **overrides):
    """The model a case name asks for: ``mtp`` adds an MTP layer ('/*-'), ``mhc`` 4 mHC
    streams (with mHC projections in the MTP layer), ``rope`` model-level RoPE."""
    pattern = DENSE
    if "mtp" in name:
        request.getfixturevalue("cpu_mtp")
        pattern = DENSE + "/*-"
    if "mhc" in name:
        request.getfixturevalue("mhc_on_this_device")
        overrides.update(MHC)
    if "rope" in name:
        request.getfixturevalue("cpu_rope")
        overrides.update(position_embedding_type="rope")
    model = build_local_loop_model(pattern, loop_iterations=R_MAX, **overrides)
    assert bool(model.mtp_process) == ("mtp" in name)
    assert model.config.enable_mhc_connections == ("mhc" in name)
    if "mtp" in name and "mhc" in name:
        assert model.mtp.layers[0].e_proj is not None
    return model


def _program(model, depths=DEPTHS, **kwargs):
    criterion = PrescribedDepthCriterion(dict(enumerate(depths)))
    return LoopProgram(model, exit_criterion=criterion, **kwargs)


# ---------------------------------------------------------------------------
# C1: the cell path is the eager forward, per sample
# ---------------------------------------------------------------------------


CASES = ["dense", "dense_padded", "mtp", "mhc", "mhc_mtp", "mtp_rope"]


@pytest.mark.parametrize("name", CASES)
def test_cells_equal_the_eager_forward_per_sample(request, name):
    model = _model(name, request)
    program = _program(model, check_ports=True)
    batch = _batch(len(DEPTHS), padded=name.endswith("padded"))
    for row, depth in enumerate(DEPTHS):
        output, loss, num_tokens, report = _eager(model, batch, row, depth)
        sample = _sample(batch, row)
        _, rounds, coda = _cells(program, sample, depth)
        assert (sample.rotary_pos_emb is not None) == ("rope" in name)
        assert torch.equal(sample.loss_func.output, output), (name, row)
        assert torch.equal(coda.loss, loss) and torch.equal(coda.num_tokens, num_tokens)
        assert coda.report.keys() == report.keys()
        assert all(torch.equal(coda.report[key], report[key]) for key in report)
        assert [bool(out.exit.item()) for out in rounds] == [False] * (depth - 1) + [True]
    assert program.after_exit_violations.item() == 0


@pytest.mark.parametrize("name", ["dense", "mtp", "mhc_mtp", "mtp_rope"])
def test_cells_as_one_graph_give_the_eager_gradients(request, name, deterministic_algorithms):
    """Prelude + rounds + coda without detaching is the eager autograd graph, op for op
    (deterministic kernels: on CUDA the embedding backward accumulates)."""
    model = _model(name, request)
    program = _program(model)
    batch = _batch(len(DEPTHS))
    model.zero_grad(set_to_none=True)
    for row, depth in enumerate(DEPTHS):
        _eager(model, batch, row, depth)[1].backward()
    expected = _grads(model)
    for row, depth in enumerate(DEPTHS):
        _cells(program, _sample(batch, row), depth)[2].loss.backward()
    _assert_grads_bitwise(_grads(model), expected)
    assert (
        "decoder.loop_adapter.weight" in expected and "embedding.word_embeddings.weight" in expected
    )


def test_prelude_recompute_equals_the_sweep(request):
    """The PRELUDE item's recompute reuses the sweep's context and reproduces ``e``."""
    model = _model("mhc_mtp", request)
    program = _program(model)
    sample = _sample(_batch(1), 0)
    sweep = program.prelude(sample, phase=LoopPhase.SWEEP)
    ctx = sample.ctx
    recompute = program.prelude(sample, phase=LoopPhase.RECOMPUTE)
    assert sample.ctx is ctx  # the recompute's own context is discarded
    assert torch.equal(recompute.injection, sweep.injection)
    assert sweep.crit_state.shape == (1, 0) and recompute.crit_state is None
    with pytest.raises(ValueError, match="no rebuild"):
        program.prelude(sample, phase=LoopPhase.REBUILD)
    fresh = _sample(_batch(1), 0)
    with pytest.raises(ValueError, match="run the prelude sweep first"):
        program.prelude(fresh, phase=LoopPhase.RECOMPUTE)


def _max_relative_error(actual, expected):
    """The worst ``max|a - e| / max|e|`` over the parameters (the M1 gradient gate)."""
    assert actual.keys() == expected.keys(), actual.keys() ^ expected.keys()
    return max(
        ((actual[name] - grad).abs().max() / grad.abs().max().clamp_min(1e-30)).item()
        for name, grad in expected.items()
    )


def _sweep(program, sample, depth=None):
    """The executor's sweep: prelude and rounds with the checkpoints detached. Returns the
    injection, the round inputs ``s_0..s_d`` (``s_0`` None) and ``crit_0..crit_d``."""
    prelude = program.prelude(sample, phase=LoopPhase.SWEEP)
    injection = prelude.injection.detach()
    depth_port = (
        None if depth is None else torch.full((1,), depth, dtype=torch.int32, device=device())
    )
    states, crits = [None], [prelude.crit_state.detach()]
    for r in range(1, (depth or program.max_rounds) + 1):
        out = program.core(
            states[-1],
            injection,
            sample,
            r=r,
            phase=LoopPhase.SWEEP,
            crit_state=crits[-1],
            exited=_exited(1),
            depth=depth_port,
        )
        states.append(out.state.detach())
        crits.append(out.crit_state.detach())
    return injection, states, crits, depth_port


def _leaf(value):
    return None if value is None else value.clone().requires_grad_()


def _item_chain(program, sample, depth=None):
    """The backward items on fresh leaves, as the executor chains them: CODA, then CORE(r)
    for r = d..1 (each replays its round from the checkpoints in RECOMPUTE and back-propagates
    ``dL/ds_r``, ``dL/dcrit_r`` and its ``aux_loss``), then PRELUDE with ``sum_r dL/de`` and
    ``dL/dcrit_0``. With a differentiable criterion the objective is the loss + every round's
    ``aux_loss`` + ``sum(crit_d)`` (so ``dL/dcrit`` is nonzero). Returns the coda's loss."""
    injection, states, crits, depth_port = _sweep(program, sample, depth)
    final = _leaf(states[-1])
    coda = program.coda(final, sample)
    coda.loss.backward()
    differentiable = program.criterion.differentiable
    grad_state, grad_injection = final.grad, None
    grad_crit = torch.ones_like(crits[-1]) if differentiable else None
    for r in range(len(states) - 1, 0, -1):
        round_input, injection_leaf = _leaf(states[r - 1]), _leaf(injection)
        crit_leaf = _leaf(crits[r - 1]) if differentiable else crits[r - 1]
        out = program.core(
            round_input,
            injection_leaf,
            sample,
            r=r,
            phase=LoopPhase.RECOMPUTE,
            crit_state=crit_leaf,
            exited=_exited(1),
            depth=depth_port,
        )
        assert torch.equal(out.state, states[r]) and torch.equal(out.crit_state, crits[r])
        outputs, grads = [out.state], [grad_state]
        if differentiable:
            outputs += [out.crit_state, out.aux_loss]
            grads += [grad_crit, torch.ones_like(out.aux_loss)]
        torch.autograd.backward(outputs, grads)
        grad_state = None if round_input is None else round_input.grad
        grad_injection = (
            injection_leaf.grad if grad_injection is None else grad_injection + injection_leaf.grad
        )
        grad_crit = crit_leaf.grad if differentiable else None
    prelude = program.prelude(sample, phase=LoopPhase.RECOMPUTE)
    assert torch.equal(prelude.injection, injection)
    if differentiable:
        torch.autograd.backward(
            [prelude.injection, prelude.crit_state], [grad_injection, grad_crit]
        )
    else:
        assert prelude.crit_state is None
        prelude.injection.backward(grad_injection)
    return coda.loss.detach()


@pytest.mark.parametrize("name", CASES)
def test_item_chain_gives_the_eager_gradients(request, name):
    """The executor's item chain on fresh leaves (detached sweep checkpoints, CODA, CORE
    d..1 in RECOMPUTE, PRELUDE) gives the eager per-sample losses bitwise and the eager
    parameter gradients within the M1 gate (max-relative 1e-5)."""
    model = _model(name, request)
    program = _program(model, check_ports=True)
    batch = _batch(len(DEPTHS), padded=name.endswith("padded"))
    model.zero_grad(set_to_none=True)
    expected_losses = []
    for row, depth in enumerate(DEPTHS):
        loss = _eager(model, batch, row, depth)[1]
        loss.backward()
        expected_losses.append(loss.detach())
    expected = _grads(model)
    losses = [_item_chain(program, _sample(batch, row), depth) for row, depth in enumerate(DEPTHS)]
    actual = _grads(model)
    assert all(torch.equal(a, b) for a, b in zip(losses, expected_losses))
    assert _max_relative_error(actual, expected) < 1e-5
    assert program.after_exit_violations.item() == 0


def test_item_chain_with_an_online_criterion(request):
    """A differentiable criterion: the chain carries ``dL/dcrit`` and each round's
    ``aux_loss``, and the PRELUDE item back-propagates ``crit_0`` (rule 7). Equal to the same
    objective as one graph, halting-head parameters included."""
    model = _model("mhc", request)
    criterion = _halting(model, tau=10.0)
    program = LoopProgram(model, exit_criterion=criterion, check_ports=True)
    sample = _sample(_batch(1), 0)
    params = dict(model.named_parameters(), halting_w=criterion.head.w, halting_v=criterion.head.v)

    def grads():
        out = {name: p.grad.clone() for name, p in params.items() if p.grad is not None}
        for p in params.values():
            p.grad = None
        return out

    prelude = program.prelude(sample, phase=LoopPhase.SWEEP)
    state, crit, objective = None, prelude.crit_state, 0
    for r in range(1, R_MAX + 1):
        out = program.core(
            state,
            prelude.injection,
            sample,
            r=r,
            phase=LoopPhase.SWEEP,
            crit_state=crit,
            exited=_exited(1),
        )
        state, crit, objective = out.state, out.crit_state, objective + out.aux_loss
    (program.coda(state, sample).loss + objective + crit.sum()).backward()
    expected = grads()
    _item_chain(program, sample)
    actual = grads()
    assert {"halting_w", "halting_v"} <= expected.keys()
    assert _max_relative_error(actual, expected) < 1e-5


@pytest.mark.parametrize("name", ["dense", "dense_padded", "mhc"])
def test_alive_rows_in_a_recompute(request, name):
    """RECOMPUTE with one dead row (c = 2): the dead row passes its state gradient through
    unchanged and gets none for ``e``; the live row's input and parameter gradients equal its
    own one-row replay."""
    model = _model(name, request)
    program = LoopProgram(model)
    batch = _batch(2, padded=name.endswith("padded"))
    pair = _sample(batch, 0, 2)
    prelude = program.prelude(pair, phase=LoopPhase.SWEEP)
    injection = prelude.injection.detach()
    first = program.core(
        None,
        injection,
        pair,
        r=1,
        phase=LoopPhase.SWEEP,
        crit_state=prelude.crit_state,
        exited=_exited(2),
    )
    state = first.state.detach()
    model.zero_grad(set_to_none=True)
    round_input, injection_leaf = _leaf(state), _leaf(injection)
    out = program.core(
        round_input,
        injection_leaf,
        pair,
        r=2,
        phase=LoopPhase.RECOMPUTE,
        crit_state=first.crit_state,
        exited=_exited(2),
        alive=torch.tensor([True, False], device=device()),
    )
    grad = torch.randn(out.state.shape, generator=torch.Generator().manual_seed(7)).to(device())
    out.state.backward(grad)
    assert torch.equal(round_input.grad[:, 1], grad[:, 1])
    assert not injection_leaf.grad[:, 1].any()
    pair_grads = _grads(model)

    # Same-shape reference: both rows alive, no upstream gradient on the second row. The dead
    # row adds exact zeros to every parameter gradient, so the GEMMs see the same row-0 terms
    # in the same order: bitwise.
    both_input, both_injection = _leaf(state), _leaf(injection)
    live_only = grad.clone()
    live_only[:, 1] = 0
    program.core(
        both_input,
        both_injection,
        pair,
        r=2,
        phase=LoopPhase.RECOMPUTE,
        crit_state=first.crit_state,
        exited=_exited(2),
        alive=torch.tensor([True, True], device=device()),
    ).state.backward(live_only)
    _assert_grads_bitwise(pair_grads, _grads(model))
    assert torch.equal(round_input.grad[:, 0], both_input.grad[:, 0])
    assert torch.equal(injection_leaf.grad[:, 0], both_injection.grad[:, 0])

    single = _sample(batch, 0)
    single_prelude = program.prelude(single, phase=LoopPhase.SWEEP)
    single_input, single_injection = _leaf(state[:, :1]), _leaf(injection[:, :1])
    program.core(
        single_input,
        single_injection,
        single,
        r=2,
        phase=LoopPhase.RECOMPUTE,
        crit_state=single_prelude.crit_state,
        exited=_exited(1),
    ).state.backward(grad[:, :1])
    single_grads = _grads(model)
    assert pair_grads.keys() == single_grads.keys()
    if "mhc" in name:
        # The one-row replay runs GEMMs of another shape. The mHC alpha_* / bias gradients are
        # sums of many cancelling terms (|g| ~ 1e-7 measured on H100), so their fp32
        # re-association error exceeds any relative gate; the bitwise same-shape check above
        # is the parameter gate, here only the key set and the other parameters.
        single_grads = {k: g for k, g in single_grads.items() if "hyper_connection.alpha" not in k}
        single_grads = {k: g for k, g in single_grads.items() if "hyper_connection.bias" not in k}
        pair_grads = {k: pair_grads[k] for k in single_grads}
    assert _max_relative_error(pair_grads, single_grads) < 1e-5
    torch.testing.assert_close(round_input.grad[:, :1], single_input.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        injection_leaf.grad[:, :1], single_injection.grad, rtol=1e-5, atol=1e-6
    )


# ---------------------------------------------------------------------------
# C2: ports
# ---------------------------------------------------------------------------


def test_ports_are_checked(request):
    model = _model("dense", request)
    program = LoopProgram(model, check_ports=True)
    sample = _sample(_batch(1), 0)
    prelude = program.prelude(sample, phase=LoopPhase.SWEEP)

    def core(**changes):
        kwargs = dict(
            round_input=None,
            injection=prelude.injection,
            sample=sample,
            r=1,
            phase=LoopPhase.SWEEP,
            crit_state=prelude.crit_state,
            exited=_exited(1),
        )
        kwargs.update(changes)
        return program.core(**kwargs)

    assert isinstance(core(), RoundOutputs)
    with pytest.raises(ValueError, match="'injection': dtype"):
        core(injection=prelude.injection.double())
    with pytest.raises(ValueError, match="'round_input': shape"):
        core(round_input=prelude.injection[:, :, :8])
    with pytest.raises(ValueError, match="'crit_state': shape"):
        core(crit_state=torch.zeros(1, 3, device=device()))
    with pytest.raises(ValueError, match="'exited' is required"):
        core(exited=None)
    with pytest.raises(ValueError, match="'exited': dtype"):
        core(exited=_exited(1).bool())
    with pytest.raises(ValueError, match="round 4 outside"):
        core(r=4)
    with pytest.raises(ValueError, match="'exited': device"):
        core(exited=torch.zeros(1, dtype=torch.int32, device="meta"))
    rotary = find_port(PRELUDE_PORTS, "rotary_pos_emb", "out")
    with pytest.raises(ValueError, match="carries no gradient"):
        program.check(
            rotary,
            torch.zeros(SEQ, 1, 1, 8, device=device(), requires_grad=True),
            rows=1,
            sequence=SEQ,
        )
    program.check(rotary, None, rows=1, sequence=SEQ)  # optional
    bad = _sample(_batch(1), 0)
    bad.tokens = bad.tokens.int()
    with pytest.raises(ValueError, match="'tokens': dtype"):
        program.prelude(bad, phase=LoopPhase.SWEEP)
    assert {port.name for port in CORE_PORTS if port.direction == "out"} == {
        "state",
        "exit",
        "crit_state",
        "aux_loss",
    }


# ---------------------------------------------------------------------------
# C3: per-row self-carry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["dense", "dense_padded", "mhc"])
def test_alive_rows_self_carry(request, name):
    """A dead row keeps its state (every stream) and is padding for the layers; the live row
    equals its own one-row run."""
    model = _model(name, request)
    program = LoopProgram(model, check_ports=True)
    batch = _batch(2, padded=name.endswith("padded"))
    pair = _sample(batch, 0, 2)
    prelude = program.prelude(pair, phase=LoopPhase.SWEEP)
    first = program.core(
        None,
        prelude.injection,
        pair,
        r=1,
        phase=LoopPhase.SWEEP,
        crit_state=prelude.crit_state,
        exited=_exited(2),
    )
    roles = program.roles
    seen = []

    def record(module, args, kwargs):
        if roles.core_start < module.layer_number <= roles.core_end:
            seen.append(kwargs.get("padding_mask"))

    handles = [
        layer.register_forward_pre_hook(record, with_kwargs=True) for layer in model.decoder.layers
    ]
    try:
        alive = torch.tensor([False, True], device=device())
        second = program.core(
            first.state,
            prelude.injection,
            pair,
            r=2,
            phase=LoopPhase.SWEEP,
            crit_state=first.crit_state,
            exited=_exited(2),
            alive=alive,
        )
    finally:
        for handle in handles:
            handle.remove()
    assert torch.equal(second.state[:, 0], first.state[:, 0])
    assert second.exit.tolist() == [True, False]
    assert len(seen) == roles.core
    for mask in seen:
        assert mask[0].all()
        expected = batch["padding_mask"][1] if batch["padding_mask"] is not None else None
        assert torch.equal(mask[1], expected) if expected is not None else not mask[1].any()

    single = _sample(batch, 1)
    single_prelude = program.prelude(single, phase=LoopPhase.SWEEP)
    state, crit = None, single_prelude.crit_state
    for r in (1, 2):
        out = program.core(
            state,
            single_prelude.injection,
            single,
            r=r,
            phase=LoopPhase.SWEEP,
            crit_state=crit,
            exited=_exited(1),
        )
        state, crit = out.state, out.crit_state
    torch.testing.assert_close(second.state[:, 1:2], state, rtol=1e-6, atol=1e-6)
    with pytest.raises(NotImplementedError, match="whole rounds"):
        program.core(
            first.state,
            prelude.injection,
            pair,
            r=2,
            phase=LoopPhase.SWEEP,
            crit_state=first.crit_state,
            exited=_exited(2),
            alive=alive,
            block=CoreBlock(0, 2, "round_input", True, False),
        )


# ---------------------------------------------------------------------------
# C4: the exit-criterion contract
# ---------------------------------------------------------------------------


def _halting(model, **kwargs):
    stack = model.decoder
    head = HaltingHead(stack.loop_num_streams, stack.config.hidden_size, device=device())
    return OnlineHaltingCriterion(head, **kwargs)


def test_criterion_is_per_row(request):
    """Rule 1: a permuted chunk gives permuted outputs, bitwise."""
    model = _model("mhc", request)
    criterion = _halting(model)
    generator = torch.Generator().manual_seed(3)
    state = torch.randn(SEQ, 3, 4 * HIDDEN, generator=generator).to(device())
    crit = torch.randn(3, 1, generator=generator).to(device())
    valid = torch.rand(3, SEQ, generator=generator).to(device()) > 0.3
    r = torch.tensor(2, dtype=torch.int32, device=device())
    perm = torch.tensor([2, 0, 1], device=device())
    whole = criterion.step(state, crit, r, valid, None)
    permuted = criterion.step(state[:, perm], crit[perm], r, valid[perm], None)
    assert torch.equal(permuted.exit, whole.exit[perm])
    assert torch.equal(permuted.crit_state, whole.crit_state[perm])
    assert torch.equal(
        criterion.init_state(state[:, perm], valid[perm]), criterion.init_state(state, valid)[perm]
    )


class _HalfPrecisionCriterion(FixedDepthCriterion):
    state_width = 1

    def init_state(self, injection, valid):
        return torch.zeros(injection.shape[1], 1, dtype=torch.float32, device=injection.device)

    def step(self, state, crit_state, r, valid, depth):
        step = super().step(state, crit_state, r, valid, depth)
        return dataclasses.replace(step, crit_state=crit_state.bfloat16())


def test_criterion_state_is_fp32(request):
    """Rule 2, enforced by the port."""
    model = _model("dense", request)
    program = LoopProgram(model, check_ports=True, exit_criterion=_HalfPrecisionCriterion(2))
    sample = _sample(_batch(1), 0)
    prelude = program.prelude(sample, phase=LoopPhase.SWEEP)
    with pytest.raises(ValueError, match="'crit_state': dtype"):
        program.core(
            None,
            prelude.injection,
            sample,
            r=1,
            phase=LoopPhase.SWEEP,
            crit_state=prelude.crit_state,
            exited=_exited(1),
        )


def _exit_bits(program, sample, rounds):
    prelude = program.prelude(sample, phase=LoopPhase.SWEEP)
    state, crit, bits = None, prelude.crit_state, []
    for r in range(1, rounds + 1):
        out = program.core(
            state,
            prelude.injection,
            sample,
            r=r,
            phase=LoopPhase.SWEEP,
            crit_state=crit,
            exited=_exited(1),
        )
        state, crit = out.state, out.crit_state
        bits.append(bool(out.exit.item()))
    return bits


def test_exit_is_forced_at_r_max_and_held_before_r_min(request):
    """Rule 3: the program, not the criterion, forces the exit at R_max and holds it before
    r_min; host-side depths get the same forcing."""
    model = _model("dense", request)
    sample = _sample(_batch(1), 0)
    never = LoopProgram(model, exit_criterion=ConstantExitCriterion(False))
    assert _exit_bits(never, sample, R_MAX) == [False, False, True]
    always = LoopProgram(model, exit_criterion=ConstantExitCriterion(True))
    always.min_rounds = 2
    assert _exit_bits(always, _sample(_batch(1), 0), R_MAX) == [False, True, True]
    prescribed = LoopProgram(model, exit_criterion=PrescribedDepthCriterion({0: 1, 1: 3}))
    prescribed.min_rounds = 2
    keys = [SampleKey(0, 0), SampleKey(1, 0)]
    assert prescribed.known_depths(keys) == [2, 3]
    assert never.known_depths(keys) is None


def test_replay_starts_from_the_checkpointed_criterion_state(request):
    """Rule 4: round r replayed from (s_{r-1}, crit_{r-1}) reproduces the sweep; a later
    criterion state does not."""
    model = _model("mhc", request)
    program = LoopProgram(model, exit_criterion=_halting(model, tau=10.0))
    sample = _sample(_batch(1), 0)
    prelude = program.prelude(sample, phase=LoopPhase.SWEEP)
    # Sweep mode: grad enabled, inputs and outputs detached (the executor's checkpoints).
    states, crits = [None], [prelude.crit_state.detach()]
    for r in range(1, R_MAX + 1):
        out = program.core(
            states[-1],
            prelude.injection.detach(),
            sample,
            r=r,
            phase=LoopPhase.SWEEP,
            crit_state=crits[-1],
            exited=_exited(1),
        )
        states.append(out.state.detach())
        crits.append(out.crit_state.detach())

    def replay(crit_state):
        return program.core(
            states[1].clone().requires_grad_(),
            prelude.injection.detach().clone().requires_grad_(),
            sample,
            r=2,
            phase=LoopPhase.RECOMPUTE,
            crit_state=crit_state.clone().requires_grad_(),
            exited=_exited(1),
        )

    replayed = replay(crits[1])
    assert torch.equal(replayed.state, states[2])
    assert torch.equal(replayed.crit_state, crits[2])
    assert replayed.aux_loss is not None and replayed.aux_loss.requires_grad
    assert not torch.equal(replay(crits[2]).crit_state, crits[2])


def test_init_state_gradients_follow_the_criterion(request):
    """Rule 7: a differentiable criterion's crit_0 is recomputed with grad by the prelude
    item; a non-differentiable one must hand out a grad-free crit_0."""
    model = _model("mhc", request)
    criterion = _halting(model)
    program = LoopProgram(model, exit_criterion=criterion)
    sample = _sample(_batch(1), 0)
    program.prelude(sample, phase=LoopPhase.SWEEP)
    recomputed = program.prelude(sample, phase=LoopPhase.RECOMPUTE)
    assert recomputed.crit_state.requires_grad
    recomputed.crit_state.sum().backward()
    assert criterion.head.v.grad is not None and criterion.head.w.grad is None
    assert model.embedding.word_embeddings.weight.grad is not None

    class LeakyInit(FixedDepthCriterion):
        state_width = 1

        def init_state(self, injection, valid):
            return injection.float().mean((0, 2))[:, None]

    leaky = LoopProgram(model, exit_criterion=LeakyInit(2))
    with pytest.raises(ValueError, match="rule 7"):
        leaky.prelude(_sample(_batch(1), 0), phase=LoopPhase.SWEEP)
    pair = _sample(_batch(2), 0, 2)
    prelude = program.prelude(pair, phase=LoopPhase.SWEEP)
    with pytest.raises(NotImplementedError, match="one row per cell call"):
        program.core(
            None,
            prelude.injection,
            pair,
            r=1,
            phase=LoopPhase.SWEEP,
            crit_state=prelude.crit_state,
            exited=_exited(2),
            alive=torch.tensor([True, False], device=device()),
        )


# ---------------------------------------------------------------------------
# C5: the after-exit counter
# ---------------------------------------------------------------------------


def test_after_exit_counter_counts_sweeps_only(request):
    model = _model("dense", request)
    program = LoopProgram(model)
    sample = _sample(_batch(1), 0)
    prelude = program.prelude(sample, phase=LoopPhase.SWEEP)

    def core(phase, exited, target=sample, injection=prelude.injection, crit=prelude.crit_state):
        program.core(None, injection, target, r=1, phase=phase, crit_state=crit, exited=exited)
        return program.after_exit_violations.item()

    assert core(LoopPhase.SWEEP, _exited(1)) == 0
    assert core(LoopPhase.SWEEP, _exited(1, 1)) == 1
    assert core(LoopPhase.RECOMPUTE, _exited(1, 1)) == 1
    pair = _sample(_batch(2), 0, 2)
    pair_prelude = program.prelude(pair, phase=LoopPhase.SWEEP)
    assert (
        core(LoopPhase.SWEEP, _exited(2, 1), pair, pair_prelude.injection, pair_prelude.crit_state)
        == 3
    )
    # A carried dead row (alive False) is chunk padding, not work after exit; an exited row
    # that is advanced (alive True) counts.
    pair_kwargs = dict(phase=LoopPhase.SWEEP, crit_state=pair_prelude.crit_state)
    program.core(
        None,
        pair_prelude.injection,
        pair,
        r=1,
        exited=_exited(2, 1),
        alive=torch.tensor([False, True], device=device()),
        **pair_kwargs,
    )
    assert program.after_exit_violations.item() == 4
    program.core(
        None,
        pair_prelude.injection,
        pair,
        r=1,
        exited=torch.tensor([1, 0], dtype=torch.int32, device=device()),
        alive=torch.tensor([False, True], device=device()),
        **pair_kwargs,
    )
    assert program.after_exit_violations.item() == 4
    assert program.after_exit_violations.dtype == torch.int32
    program.reset_counters()
    assert program.after_exit_violations.item() == 0
    with pytest.raises(NotImplementedError, match="a rebuild runs through forward_detached"):
        program.core(
            None,
            prelude.injection,
            sample,
            r=1,
            phase=LoopPhase.REBUILD,
            crit_state=prelude.crit_state,
            exited=_exited(1),
        )
    # A forward_detached sweep counts like the sweep; a rebuild has no epilogue to count.
    program.reset_counters()
    for phase, end, count in ((LoopPhase.SWEEP, None, 1), (LoopPhase.REBUILD, 2, 1)):
        program.core.forward_detached(
            None,
            prelude.injection,
            sample,
            r=1,
            phase=phase,
            end=end,
            crit_state=prelude.crit_state,
            exited=_exited(1, 1),
        )
        assert program.after_exit_violations.item() == count


# ---------------------------------------------------------------------------
# C6: sampled depths and sample indices
# ---------------------------------------------------------------------------


def test_sampled_depths_are_counter_based():
    probs = [0.25, 0.25, 0.25, 0.25]
    keys = [SampleKey(index, 7) for index in range(64)]
    depths = SampledDepthCriterion(probs, seed=1234).known_depths(keys)
    assert depths == SampledDepthCriterion(probs, seed=1234).known_depths(keys)
    assert set(depths) <= {1, 2, 3, 4} and len(set(depths)) > 1
    # Keyed per sample: any subset or order of keys draws the same depths.
    assert SampledDepthCriterion(probs, seed=1234).known_depths(keys[::-3]) == depths[::-3]
    assert SampledDepthCriterion(probs, seed=99).known_depths(keys) != depths
    later = [SampleKey(index, 8) for index in range(64)]
    assert SampledDepthCriterion(probs, seed=1234).known_depths(later) != depths
    assert SampledDepthCriterion([0, 0, 1], seed=0).known_depths(keys) == [3] * 64
    assert SampledDepthCriterion([1, 0, 0], seed=0, min_rounds=2).known_depths(keys) == [2] * 64
    histogram = torch.bincount(
        torch.tensor(
            SampledDepthCriterion(probs, seed=5).known_depths(
                [SampleKey(index, 0) for index in range(4000)]
            )
        ),
        minlength=5,
    )[1:]
    assert ((histogram.float() / 4000 - 0.25).abs() < 0.04).all(), histogram
    assert counter_seed(1, 2, 3) != counter_seed(1, 3, 2)
    assert 0 <= counter_seed(-1, -1, -1) < 2**64


def test_global_sample_index_is_independent_of_placement():
    """DP 2 x MBS 1 and DP 1 x MBS 2 cover the same global positions; the depth of a
    position does not depend on which rank or microbatch holds it."""
    microbatches = 3
    dp2 = {
        (j, q): global_sample_index(j, 0, micro_batch_size=1, dp_rank=q, dp_size=2)
        for j in range(microbatches)
        for q in range(2)
    }
    mbs2 = {
        (j, b): global_sample_index(j, b, micro_batch_size=2, dp_rank=0, dp_size=1)
        for j in range(microbatches)
        for b in range(2)
    }
    assert sorted(dp2.values()) == sorted(mbs2.values()) == list(range(2 * microbatches))
    assert dp2[(1, 1)] == 3 and mbs2[(1, 1)] == 3
    assert global_sample_index(2, 1, micro_batch_size=2, dp_rank=3, dp_size=4) == 23
    with pytest.raises(ValueError, match="out of range"):
        global_sample_index(0, 2, micro_batch_size=2, dp_rank=0, dp_size=1)


def test_depth_probs_and_criterion_config():
    validate_depth_probs([0.5, 0.5], 2)
    for probs, rounds in (([0.5, 0.5], 3), ([1.5, -0.5], 2), ([0.3, 0.3], 2), (None, 2)):
        with pytest.raises(ValueError, match="loop depth probabilities"):
            validate_depth_probs(probs, rounds)
    fixed = build_exit_criterion(SimpleNamespace(loop_iterations=3))
    assert isinstance(fixed, FixedDepthCriterion) and fixed.rounds == 3
    sampled = build_exit_criterion(
        SimpleNamespace(
            loop_iterations=2,
            loop_exit_criterion="sampled",
            loop_depth_probs=[0.5, 0.5],
            loop_depth_seed=4,
            loop_min_rounds=2,
        )
    )
    assert isinstance(sampled, SampledDepthCriterion)
    assert sampled.seed == 4 and sampled.min_rounds == 2
    with pytest.raises(ValueError, match="need 3 entries"):  # probs must cover 1..R_max
        build_exit_criterion(
            SimpleNamespace(
                loop_iterations=3, loop_exit_criterion="sampled", loop_depth_probs=[0.5, 0.5]
            )
        )
    with pytest.raises(ValueError, match="'fixed'"):
        build_exit_criterion(SimpleNamespace(loop_iterations=2, loop_depth_probs=[0.5, 0.5]))
    with pytest.raises(ValueError, match="need 3 entries"):
        SampledDepthCriterion([0.5, 0.5], seed=0, max_rounds=3)
    with pytest.raises(ValueError, match="unknown loop_exit_criterion"):
        build_exit_criterion(SimpleNamespace(loop_iterations=2, loop_exit_criterion="learned"))
    with pytest.raises(ValueError, match=">= 1"):
        PrescribedDepthCriterion({0: 0})


# ---------------------------------------------------------------------------
# C7, C8: head_loss at CP = 1 and the purity of _prepare_forward
# ---------------------------------------------------------------------------


def test_head_loss_ignores_the_cp1_batch(cpu_mtp):
    model = build_local_loop_model(DENSE + "/*-", loop_iterations=2)
    rows = _rows(_batch(1), 0, 1)
    cp_batch = get_batches_on_this_cp_rank(
        {**rows, "attention_mask": None, "cu_seqlens": None},
        boundary_layout=model.config.linear_cp_layout,
        is_hybrid_cp=False,
        cp_group=model.pg_collection.cp,
        tokens_per_sample=SEQ,
    )
    hidden = torch.randn(SEQ, 1, HIDDEN, generator=torch.Generator().manual_seed(1)).to(device())

    def head(batch):
        return model.head_loss(
            hidden,
            None,
            input_ids=rows["tokens"],
            position_ids=rows["position_ids"],
            attention_mask=None,
            labels=rows["labels"],
            loss_mask=rows["loss_mask"],
            padding_mask=None,
            cp_batch=batch,
        )

    with torch.no_grad():
        assert torch.equal(head(cp_batch), head(None))


@pytest.mark.parametrize("name", ["dense", "mhc"])
def test_prepare_forward_is_pure(request, name):
    """A second ``_prepare_forward`` (the PRELUDE item's) expands bitwise alike and builds an
    equal context."""
    model = _model(name, request)
    rows = _rows(_batch(2, padded=True), 1, 2)  # row 1 is padded
    assert rows["padding_mask"].any()
    decoder_input, padding_mask, hash_ids = model.embed(
        rows["tokens"], rows["position_ids"], padding_mask=rows["padding_mask"]
    )

    def prepare():
        return model.decoder._prepare_forward(
            decoder_input,
            None,
            None,
            None,
            packed_seq_params=None,
            padding_mask=padding_mask,
            input_ids=hash_ids,
            packed_seq_params_by_layout=None,
            cp_layout_plan=None,
        )

    (x_a, ctx_a), (x_b, ctx_b) = prepare(), prepare()
    assert torch.equal(x_a, x_b) and x_a.shape[-1] == model.decoder.loop_num_streams * HIDDEN
    for field in dataclasses.fields(ctx_a):
        a, b = getattr(ctx_a, field.name), getattr(ctx_b, field.name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), field.name
        elif callable(a) or isinstance(a, contextlib.AbstractContextManager):
            assert type(a) is type(b), field.name
        else:
            assert a == b, field.name


@pytest.mark.parametrize("name", ["dense_padded", "mhc", "mtp_rope"])
def test_build_context_equals_the_sweep_context(request, name):
    """C10: a sample arriving on another EU rebuilds its activation-free metadata without
    the prelude (``PreludeCell.build_context``): the decoder padding mask, hash IDs, rotary
    embedding and every layer-context field equal the sweep's, and a core round and the coda
    run on the rebuilt context give the sweep context's outputs bitwise."""
    model = _model(name, request)
    program = _program(model, depths=(2, 2))
    batch = _batch(2, padded=name.endswith("padded"))
    swept, arrived = _sample(batch, 1), _sample(batch, 1)
    prelude = program.prelude(swept, phase=LoopPhase.SWEEP)
    program.prelude.build_context(arrived)
    for field_name in ("decoder_padding_mask", "hash_input_ids", "rotary_pos_emb"):
        a, b = getattr(swept, field_name), getattr(arrived, field_name)
        assert (a is None and b is None) or torch.equal(a, b), field_name
    if name == "mtp_rope":
        assert arrived.rotary_pos_emb is not None
    if name.endswith("padded"):
        assert arrived.decoder_padding_mask.any()
    for field in dataclasses.fields(swept.ctx):
        a, b = getattr(swept.ctx, field.name), getattr(arrived.ctx, field.name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), field.name
        elif callable(a) or isinstance(a, contextlib.AbstractContextManager):
            assert type(a) is type(b), field.name
        elif isinstance(a, list):
            assert len(a) == len(b) and [type(x) for x in a] == [type(x) for x in b], field.name
        else:
            assert a == b, field.name
    outputs = []
    for sample in (swept, arrived):
        out = program.core(
            None,
            prelude.injection.detach(),
            sample,
            r=1,
            phase=LoopPhase.SWEEP,
            crit_state=prelude.crit_state.detach(),
            exited=_exited(1),
            depth=torch.tensor([2], dtype=torch.int32, device=device()),
        )
        coda = program.coda(out.state.detach(), sample)
        outputs.append((out.state, coda.loss))
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])


# ---------------------------------------------------------------------------
# C9: backward plans
# ---------------------------------------------------------------------------


def test_core_plan_is_the_whole_round_and_blocks_compose(request):
    model = _model("mhc", request)
    program = _program(model)
    sample = _sample(_batch(1), 0)
    plan = program.core_plan(sample, 2)
    assert plan == CoreBackwardPlan.whole_round(program.roles.core)
    assert len(plan.blocks) == 1 and plan.stashed_block_inputs() == () and not plan.rebuild
    (block,) = plan.blocks
    assert (block.start, block.end) == (0, program.roles.core)
    assert block.with_prologue and block.with_epilogue and block.input_source == "round_input"

    prelude = program.prelude(sample, phase=LoopPhase.SWEEP)
    kwargs = dict(
        phase=LoopPhase.SWEEP,
        crit_state=prelude.crit_state,
        exited=_exited(1),
        depth=torch.tensor([2], dtype=torch.int32, device=device()),
    )
    first = program.core(None, prelude.injection, sample, r=1, **kwargs)
    whole = program.core(first.state, prelude.injection, sample, r=2, **kwargs)
    planned = program.core(first.state, prelude.injection, sample, r=2, block=block, **kwargs)
    assert torch.equal(planned.state, whole.state) and torch.equal(planned.exit, whole.exit)
    # Two blocks (the M8b shape) chain to the same round.
    split = CoreBackwardPlan(
        blocks=(
            CoreBlock(0, 2, "round_input", with_prologue=True, with_epilogue=False),
            CoreBlock(
                2, program.roles.core, "block_stash", with_prologue=False, with_epilogue=True
            ),
        )
    )
    assert split.stashed_block_inputs() == (2,)
    middle = program.core(
        first.state, prelude.injection, sample, r=2, block=split.blocks[0], **kwargs
    )
    assert isinstance(middle, torch.Tensor)
    last = program.core(middle, prelude.injection, sample, r=2, block=split.blocks[1], **kwargs)
    assert torch.equal(last.state, whole.state) and torch.equal(last.exit, whole.exit)
    with pytest.raises(ValueError, match="do not chain"):
        CoreBackwardPlan(
            blocks=(
                CoreBlock(0, 1, "round_input", with_prologue=True, with_epilogue=False),
                CoreBlock(2, 4, "block_stash", with_prologue=False, with_epilogue=True),
            )
        )
    with pytest.raises(ValueError, match="first block"):
        CoreBackwardPlan(blocks=(CoreBlock(0, 4, "block_stash", False, True),))
    with pytest.raises(ValueError, match="do not chain"):  # a later block reads a stash
        CoreBackwardPlan(
            blocks=(
                CoreBlock(0, 2, "round_input", with_prologue=True, with_epilogue=False),
                CoreBlock(2, 4, "round_input", with_prologue=False, with_epilogue=True),
            )
        )
    with pytest.raises(ValueError, match="start < end"):
        CoreBlock(2, 2, "block_stash", with_prologue=False, with_epilogue=False)
    with pytest.raises(ValueError, match="runs the prologue"):
        CoreBlock(2, 4, "round_input", with_prologue=True, with_epilogue=True)
    short = CoreBackwardPlan.whole_round(2)
    assert short.core_sublayers == 2 != program.roles.core
    with pytest.raises(ValueError, match="epilogue before the end"):
        program.core(first.state, prelude.injection, sample, r=2, block=short.blocks[0], **kwargs)
    with pytest.raises(ValueError, match="outside"):
        program.core.layers(prelude.injection, 0, program.roles.core + 1, sample.ctx, None, "x")


@pytest.mark.parametrize("core", [1, 4, 7, 8])
def test_strided_plans_cover_the_core_in_blocks_of_the_stride(core):
    """``strided(L, s)`` for every stride, dividing or not: blocks ``[j s, min((j + 1) s, L))``,
    block 0 from the round input with the prologue, later blocks from the sweep's stash, the
    last with the epilogue; ``ceil(L / s) - 1`` stashed inputs; ``s = L`` is the whole round."""
    for stride in range(1, core + 1):
        plan = CoreBackwardPlan.strided(core, stride)
        starts = list(range(0, core, stride))
        assert [(block.start, block.end) for block in plan.blocks] == [
            (start, min(start + stride, core)) for start in starts
        ]
        assert [block.input_source for block in plan.blocks] == ["round_input"] + [
            "block_stash"
        ] * (len(starts) - 1)
        assert [block.with_prologue for block in plan.blocks] == [True] + [False] * (
            len(starts) - 1
        )
        assert [block.with_epilogue for block in plan.blocks] == [False] * (len(starts) - 1) + [
            True
        ]
        assert plan.stashed_block_inputs() == tuple(starts[1:])
        assert len(plan.stashed_block_inputs()) == math.ceil(core / stride) - 1
        assert plan.rebuilt_layer_inputs() == () and not plan.rebuild and plan.rebuild_end == 0
        assert plan.core_sublayers == core and plan.stride == stride
        assert plan.form == ("whole_round" if stride == core else "block")
        assert plan.forward_units() == 4.0
    assert CoreBackwardPlan.strided(core, core) == CoreBackwardPlan.whole_round(core)
    assert CoreBackwardPlan.whole_round(core).form == "whole_round"
    for stride in (0, -1, core + 1):
        with pytest.raises(ValueError, match=f"a block stride must be in 1..{core}"):
            CoreBackwardPlan.strided(core, stride)


@pytest.mark.parametrize(
    "core, stride, rebuilt, rebuild_end",
    [
        (8, 1, (1, 2, 3, 4, 5, 6, 7), 7),
        (8, 2, (2, 4, 6), 6),
        (8, 3, (3, 6), 6),  # non-dividing: a short last block [6, 8)
        (8, 7, (7,), 7),
        (2, 1, (1,), 1),
    ],
)
def test_nested_plans_rebuild_up_to_the_last_block(core, stride, rebuilt, rebuild_end):
    """``nested(L, s)``: the strided blocks with every later block read from the rebuild's
    layer stash; the rebuild runs ``[0, (n_b - 1) s)`` and costs ``rebuild_end / L`` forward
    units on top of the 4 of every form."""
    plan = CoreBackwardPlan.nested(core, stride)
    strided = CoreBackwardPlan.strided(core, stride)
    assert plan.rebuild and plan.form == "nested" and plan.stride == stride
    assert [(b.start, b.end) for b in plan.blocks] == [(b.start, b.end) for b in strided.blocks]
    assert plan.blocks[0] == strided.blocks[0]
    assert all(block.input_source == "layer_stash" for block in plan.blocks[1:])
    assert plan.stashed_block_inputs() == ()
    assert plan.rebuilt_layer_inputs() == rebuilt
    assert plan.rebuild_end == rebuild_end == plan.blocks[-1].start
    assert plan.forward_units() == 4.0 + rebuild_end / core
    if stride == 1:
        assert CoreBackwardPlan.nested(core) == plan


def test_nested_and_strided_reject_bad_strides():
    with pytest.raises(ValueError, match="a nested stride must be in 1..7"):
        CoreBackwardPlan.nested(8, 8)
    with pytest.raises(ValueError, match="at least two blocks"):
        CoreBackwardPlan.nested(8, 0)
    with pytest.raises(ValueError, match="a nested stride must be in 1..0"):
        CoreBackwardPlan.nested(1)


def test_plans_do_not_mix_input_sources():
    """The new checks run after the chain checks (their messages unchanged, see
    test_core_plan_is_the_whole_round_and_blocks_compose) with messages of their own."""
    first = CoreBlock(0, 2, "round_input", with_prologue=True, with_epilogue=False)
    with pytest.raises(ValueError, match="mixed block input sources: .* non-rebuild plan"):
        CoreBackwardPlan(blocks=(first, CoreBlock(2, 4, "layer_stash", False, True)))
    with pytest.raises(ValueError, match="mixed block input sources: .* rebuild plan"):
        CoreBackwardPlan(blocks=(first, CoreBlock(2, 4, "block_stash", False, True)), rebuild=True)
    with pytest.raises(ValueError, match="mixed block input sources"):
        CoreBackwardPlan(
            blocks=(
                first,
                CoreBlock(2, 3, "block_stash", False, False),
                CoreBlock(3, 4, "layer_stash", False, True),
            )
        )
    with pytest.raises(ValueError, match="a rebuild plan needs at least two blocks"):
        CoreBackwardPlan(blocks=CoreBackwardPlan.whole_round(4).blocks, rebuild=True)
    # The chain checks still come first.
    with pytest.raises(ValueError, match="do not chain"):
        CoreBackwardPlan(blocks=(first, CoreBlock(3, 4, "layer_stash", False, True)), rebuild=True)


def _form_config(form=None, stride=None):
    """A duck-typed config with only the item-form fields that are not None."""
    fields = {}
    if form is not None:
        fields["loop_rsbf_item_form"] = form
    if stride is not None:
        fields["loop_rsbf_block_stride"] = stride
    return SimpleNamespace(**fields)


def test_build_core_plan_follows_the_config():
    roles = LoopRoles(prelude=2, core=8, coda=4)
    assert build_core_plan(_form_config(), roles) == CoreBackwardPlan.whole_round(8)
    assert build_core_plan(_form_config("whole_round"), roles) == CoreBackwardPlan.whole_round(8)
    assert build_core_plan(_form_config("block", 2), roles) == CoreBackwardPlan.strided(8, 2)
    assert build_core_plan(_form_config("block", 3), roles) == CoreBackwardPlan.strided(8, 3)
    assert build_core_plan(_form_config("block", 8), roles) == CoreBackwardPlan.whole_round(8)
    assert build_core_plan(_form_config("nested"), roles) == CoreBackwardPlan.nested(8, 1)
    assert build_core_plan(_form_config("nested", 2), roles) == CoreBackwardPlan.nested(8, 2)
    rejected = [
        (_form_config("block"), guards.RSBF_BLOCK_STRIDE_MISMATCH),
        (_form_config("whole_round", 2), guards.RSBF_BLOCK_STRIDE_MISMATCH),
        (_form_config("block", 0), guards.RSBF_BLOCK_STRIDE_MISMATCH),
        (_form_config("nested", -1), guards.RSBF_BLOCK_STRIDE_MISMATCH),
        (_form_config("block", 9), guards.RSBF_BLOCK_STRIDE_BEYOND_CORE),
        (_form_config("nested", 8), guards.RSBF_BLOCK_STRIDE_BEYOND_CORE),
        (_form_config("bogus"), guards.RSBF_ITEM_FORM),
    ]
    for config, rule in rejected:
        with pytest.raises(ValueError, match=f"LoopProgram does not support: {rule} "):
            build_core_plan(config, roles)
    with pytest.raises(ValueError, match=guards.RSBF_BLOCK_STRIDE_BEYOND_CORE):
        build_core_plan(_form_config("nested"), LoopRoles(prelude=2, core=1, coda=2))


@pytest.mark.parametrize(
    "form, stride, expected",
    [
        ("whole_round", None, CoreBackwardPlan.whole_round(4)),
        ("block", 1, CoreBackwardPlan.strided(4, 1)),
        ("block", 3, CoreBackwardPlan.strided(4, 3)),
        ("nested", None, CoreBackwardPlan.nested(4, 1)),
        ("nested", 2, CoreBackwardPlan.nested(4, 2)),
    ],
)
def test_core_plan_is_the_config_plan_for_every_round(request, form, stride, expected):
    """``core_plan`` returns the config's plan object (``uniform_core_plan``) for every
    (sample, round); ``item_form`` names its form."""
    model = _model(
        "dense", request, **RSBF, loop_rsbf_item_form=form, loop_rsbf_block_stride=stride
    )
    program = _program(model)
    assert program.uniform_core_plan == expected and program.item_form == expected.form
    batch = _batch(2)
    for row in range(2):
        sample = _sample(batch, row)
        for r in range(1, R_MAX + 1):
            assert program.core_plan(sample, r) is program.uniform_core_plan
    assert program.coda_recompute_modules == frozenset()


def test_program_checks_the_item_form_against_the_built_core(request):
    """The config-time rules see ``num_layers - prelude - coda``; the program re-checks the
    stride against the built pattern's core."""
    model = _model("dense", request, **RSBF, loop_rsbf_item_form="block", loop_rsbf_block_stride=4)
    assert _program(model).uniform_core_plan == CoreBackwardPlan.whole_round(4)
    model.config.loop_rsbf_block_stride = 5  # past __post_init__
    with pytest.raises(ValueError, match=guards.RSBF_BLOCK_STRIDE_BEYOND_CORE):
        _program(model)
    model.config.loop_rsbf_item_form, model.config.loop_rsbf_block_stride = "nested", 4
    with pytest.raises(ValueError, match=guards.RSBF_BLOCK_STRIDE_BEYOND_CORE):
        _program(model)
    model.config.loop_rsbf_block_stride = None
    assert _program(model).uniform_core_plan == CoreBackwardPlan.nested(4)


def test_coda_recompute_needs_a_dsv4_attention_in_the_coda(request):
    """D42: the flag on a model without a DSv4 attention in the coda or MTP layers has nothing
    to recompute (the config rule R21 needs dsv4_hybrid; set past __post_init__ here)."""
    model = _model("mtp", request)
    model.config.loop_rsbf_coda_recompute_mla_up_proj = True
    with pytest.raises(ValueError, match="nothing to recompute"):
        _program(model)


@pytest.mark.parametrize("name", ["dense", "mhc", "mtp"])
def test_core_parameters_belong_to_one_sublayer(request, name):
    model = _model(name, request)
    program = _program(model)
    assert program.core_parameter_overlap() == []
    # A parameter shared by two core sublayers (the two core MLP sublayers) is named.
    layers = model.decoder.layers
    first, second = program.roles.core_start + 1, program.roles.core_start + 3
    name, param = next(iter(layers[first].named_parameters()))
    path, _, attribute = name.rpartition(".")
    setattr(layers[second].get_submodule(path), attribute, param)
    assert program.core_parameter_overlap() == [f"layers.{first}.{name}"]


@pytest.mark.parametrize("name", ["dense", "dense_padded", "mtp", "mhc", "mhc_mtp"])
def test_forward_detached_is_the_round_with_the_graph_dropped(request, name):
    """Sweep mode: ``forward_detached`` gives the attached round's state, exit and criterion
    state bitwise, with every output detached; its trace equals the attached per-sublayer
    outputs (``__call__(trace=...)``, which equals the plain call); ``captured[l]`` is the
    trace entry ``l - 1`` (one tensor); a rebuild (``end``) runs a prefix without the
    epilogue; dropping the graph only at captured sublayers gives the same values."""
    model = _model(name, request)
    program = _program(model, check_ports=True)
    core = program.roles.core
    sample = _sample(_batch(2, padded=name.endswith("padded")), 1)
    prelude = program.prelude(sample, phase=LoopPhase.SWEEP)
    injection = prelude.injection.detach()
    kwargs = dict(
        crit_state=prelude.crit_state.detach(),
        exited=_exited(1),
        depth=torch.tensor([2], dtype=torch.int32, device=device()),
    )
    first = program.core(None, injection, sample, r=1, phase=LoopPhase.SWEEP, **kwargs)
    previous = first.state.detach()
    plain = program.core(previous, injection, sample, r=2, phase=LoopPhase.SWEEP, **kwargs)
    attached = []
    traced = program.core(
        previous, injection, sample, r=2, phase=LoopPhase.SWEEP, trace=attached, **kwargs
    )
    capture = range(1, core + 1)
    fwd = program.core.forward_detached(
        previous,
        injection,
        sample,
        r=2,
        phase=LoopPhase.SWEEP,
        capture=capture,
        trace=True,
        **kwargs,
    )
    for out in (traced, fwd.outputs):
        assert torch.equal(out.state, plain.state) and torch.equal(out.exit, plain.exit)
        assert torch.equal(out.crit_state, plain.crit_state)
    assert not fwd.outputs.state.requires_grad and not fwd.outputs.crit_state.requires_grad
    assert len(attached) == len(fwd.trace) == core
    assert attached[-1].requires_grad
    for l in range(core):
        assert torch.equal(fwd.trace[l], attached[l]), l
        assert not fwd.trace[l].requires_grad and fwd.trace[l].grad_fn is None
        assert fwd.captured[l + 1] is fwd.trace[l]
        assert not fwd.modified_after_capture(fwd.trace[l])
    fwd.trace[0].mul_(1.0)  # an in-place write after the capture (version bump)
    assert fwd.modified_after_capture(fwd.captured[1])
    coarse = program.core.forward_detached(
        previous,
        injection,
        sample,
        r=2,
        phase=LoopPhase.SWEEP,
        capture=(2,),
        trace=True,
        detach_every_sublayer=False,
        **kwargs,
    )
    assert torch.equal(coarse.outputs.state, plain.state)
    assert all(torch.equal(a, b) for a, b in zip(coarse.trace, fwd.trace))
    assert list(coarse.captured) == [2] and torch.equal(coarse.captured[2], fwd.trace[1])
    # A rebuild of the first sublayers: no epilogue, no criterion ports.
    rebuild = program.core.forward_detached(
        previous, injection, sample, r=2, phase=LoopPhase.REBUILD, capture=(1, 2), end=2
    )
    assert rebuild.outputs is None and rebuild.trace == ()
    assert torch.equal(rebuild.captured[1], fwd.trace[0])
    assert torch.equal(rebuild.captured[2], fwd.trace[1])


def test_forward_detached_rejects_bad_arguments(request):
    model = _model("dense", request)
    program = _program(model)
    sample = _sample(_batch(1), 0)
    prelude = program.prelude(sample, phase=LoopPhase.SWEEP)
    kwargs = dict(crit_state=prelude.crit_state, exited=_exited(1))
    detached = partial(program.core.forward_detached, None, prelude.injection, sample, r=1)
    with pytest.raises(ValueError, match="sweep mode"):
        detached(phase=LoopPhase.RECOMPUTE, **kwargs)
    with pytest.raises(ValueError, match="outside"):
        detached(phase=LoopPhase.SWEEP, capture=(0,), **kwargs)
    with pytest.raises(ValueError, match="outside"):
        detached(phase=LoopPhase.REBUILD, capture=(3,), end=2)
    with pytest.raises(ValueError, match="end 5 outside"):
        detached(phase=LoopPhase.REBUILD, end=program.roles.core + 1)
    with pytest.raises(ValueError, match="needs crit_state and exited"):
        detached(phase=LoopPhase.SWEEP)


def test_block_boundaries_that_split_an_attention_and_its_mlp():
    dense = list(DENSE)  # prelude '*-', core '*-*-', coda '*-'
    roles = LoopRoles(prelude=2, core=4, coda=2)
    assert block_boundaries_split_pairs(CoreBackwardPlan.strided(4, 1), dense, roles) == (1, 3)
    assert block_boundaries_split_pairs(CoreBackwardPlan.strided(4, 2), dense, roles) == ()
    assert block_boundaries_split_pairs(CoreBackwardPlan.strided(4, 3), dense, roles) == (3,)
    assert block_boundaries_split_pairs(CoreBackwardPlan.nested(4, 1), dense, roles) == (1, 3)
    assert block_boundaries_split_pairs(CoreBackwardPlan.whole_round(4), dense, roles) == ()
    dsv4 = list("HEHECEHECEHECE")  # prelude 'HE', core 'HECEHECE', coda 'HECE'
    roles = LoopRoles(prelude=2, core=8, coda=4)
    assert block_boundaries_split_pairs(CoreBackwardPlan.strided(8, 3), dsv4, roles) == (3,)
    assert block_boundaries_split_pairs(CoreBackwardPlan.strided(8, 2), dsv4, roles) == ()
    assert block_boundaries_split_pairs(CoreBackwardPlan.nested(8, 1), dsv4, roles) == (1, 3, 5, 7)


# ---------------------------------------------------------------------------
# Program construction
# ---------------------------------------------------------------------------


def test_program_unwraps_the_model_and_rejects_mor(request):
    model = _model("dense", request)

    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    program = LoopProgram(Wrapper(model))
    assert program.hybrid is model and program.stack is model.decoder
    assert isinstance(program.criterion, FixedDepthCriterion)
    assert program.criterion.rounds == R_MAX and program.min_rounds == 1
    assert program.hidden_width == HIDDEN and program.dtype == torch.float32
    assert int(program.phase_tensor(LoopPhase.RECOMPUTE)) == 1
    assert int(program.round_tensor(R_MAX)) == R_MAX
    assert not any(isinstance(value, LoopProgram) for value in vars(model).values())
    mor = build_local_loop_model(
        DENSE, loop_architecture="mor", loop_token_execution="masked", loop_iterations=2
    )
    with pytest.raises(ValueError, match="MoR stays on the eager path"):
        LoopProgram(mor)
    with pytest.raises(ValueError, match="2 depth probabilities, the loop R_max = 3"):
        LoopProgram(model, exit_criterion=SampledDepthCriterion([0.5, 0.5], seed=0))


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("loop_bptt_iterations", 1, "truncated BPTT"),
        ("loop_sample_depth", True, "loop_sample_depth"),
        ("loop_recompute", True, "loop_recompute"),
        ("freeze_base_model_for_mtp", True, "freeze_base_model_for_mtp"),
    ],
)
def test_program_rejects_eager_only_options(request, field, value, match):
    """Options the eager loop honours and the cells would silently ignore."""
    model = _model("dense", request)
    setattr(model.config, field, value)
    assert model.decoder.config is model.config
    with pytest.raises(ValueError, match=f"LoopProgram does not support: {match}"):
        LoopProgram(model)


# ---------------------------------------------------------------------------
# DSv4 cell oracle (CUDA)
# ---------------------------------------------------------------------------


def _dsv4_loop_config(vocab, **overrides):
    """The 'WECEHE/WE' DSv4 loop config of the GPU cases (prelude 'WE', core 'CE', coda 'HE')."""
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    fields = dict(
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
        dsa_indexer_loss_coeff=0.01,
        csa_window_size=32,
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
        attention_backend=AttnBackend.unfused,
    )
    fields.update(overrides)
    return MLATransformerConfig(**fields)


def _dsv4_loop_model(config, vocab):
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_dsv4_stack_spec

    torch.manual_seed(1234)
    return HybridModel(
        config=config,
        hybrid_stack_spec=loop_hybrid_dsv4_stack_spec(config),
        vocab_size=vocab,
        max_sequence_length=1024,
        hybrid_layer_pattern="WECEHE/WE",
        position_embedding_type="rope",
    ).cuda()


@gpu
def test_dsv4_cells_equal_the_eager_forward(deterministic_algorithms):
    """'WECEHE/WE' (DSv4 W / C / H attention, unfused; MoE with one hash layer in the prelude;
    mHC 4; MTP 1), bf16, deterministic kernels: per-sample losses of the cell path equal the
    eager model bitwise; as one graph the gradients do too wherever the eager reference
    reproduces itself, and elsewhere stay within the reference's own run-to-run spread."""
    from unittest.mock import patch

    try:
        from fast_hadamard_transform import hadamard_transform  # noqa: F401

        hadamard_patch = contextlib.nullcontext()
    except ImportError:
        hadamard_patch = patch(
            "megatron.core.transformer.experimental_attention_variant.csa.rotate_activation",
            lambda x: x * (x.size(-1) ** -0.5),
        )
    vocab, seq, depths = 128, 256, (2, 1)
    config = _dsv4_loop_config(vocab)
    with hadamard_patch:
        model = _dsv4_loop_model(config, vocab)
        program = _program(model, depths, check_ports=True)
        batch = _batch(len(depths), seq=seq, vocab=vocab, dev=torch.device("cuda"))

        def eager_pass():
            set_moe_metrics_tracker(MoEMetricsTracker())
            model.zero_grad(set_to_none=True)
            results = []
            for row, depth in enumerate(depths):
                output, loss, num_tokens, _ = _eager(model, batch, row, depth)
                loss.backward()
                results.append((output, loss, num_tokens))
            return results, _grads(model)

        expected, expected_grads = eager_pass()
        set_moe_metrics_tracker(MoEMetricsTracker())
        actual = []
        for row, depth in enumerate(depths):
            sample = _sample(batch, row)
            _, rounds, coda = _cells(program, sample, depth)
            coda.loss.backward()
            actual.append((sample.loss_func.output, coda.loss, coda.num_tokens))
            assert [bool(out.exit.item()) for out in rounds] == [False] * (depth - 1) + [True]
        actual_grads = _grads(model)
        for row, (got, want) in enumerate(zip(actual, expected)):
            if not all(torch.equal(a, b) for a, b in zip(got, want)):
                rerun, _ = eager_pass()
                if not all(torch.equal(a, b) for a, b in zip(rerun[row], want)):
                    pytest.fail("NONDETERMINISTIC: the eager model does not reproduce its loss")
                pytest.fail(f"sample {row}: cell loss differs from the eager forward")
        # Bitwise where the eager reference reproduces its own gradient; elsewhere the cell
        # gradient may differ from the reference by at most twice the reference's own
        # run-to-run difference (one rerun samples that noise; no fixed tolerance).
        rerun_grads = eager_pass()[1]
        assert actual_grads.keys() == expected_grads.keys() == rerun_grads.keys()
        noisy = []
        for name, grad in expected_grads.items():
            spread = (rerun_grads[name].float() - grad.float()).abs().max()
            error = (actual_grads[name].float() - grad.float()).abs().max()
            assert error <= 2 * spread, f"{name}: |cell - eager| {error} > 2 x spread {spread}"
            if spread > 0:
                noisy.append(name)
        print(f"DSv4 cell oracle: {len(noisy)} non-reproducible eager gradients: {noisy}")
    assert program.after_exit_violations.item() == 0


@gpu
def test_dsv4_coda_recompute_request_and_item_forms():
    """D42: with a DSv4 attention in the coda ('HE') and the MTP layer ('WE') the program asks
    the CODA item for the ``mla_up_proj`` recompute; the item form comes from the config."""
    vocab = 128
    config = _dsv4_loop_config(
        vocab, **RSBF, loop_rsbf_coda_recompute_mla_up_proj=True, loop_rsbf_item_form="nested"
    )
    # Held here: a LoopProgram references its model weakly.
    model = _dsv4_loop_model(config, vocab)
    program = _program(model, (2, 1))
    assert program.coda_recompute_modules == frozenset({"mla_up_proj"})
    # The coda 'H' attention and the MTP depth's 'W' attention, in run order.
    attentions = program.coda_recompute_attentions()
    assert [module._dsv4_compress_ratio for module in attentions] == [128, 0]
    assert program.uniform_core_plan == CoreBackwardPlan.nested(program.roles.core)
    assert program.core_parameter_overlap() == []
