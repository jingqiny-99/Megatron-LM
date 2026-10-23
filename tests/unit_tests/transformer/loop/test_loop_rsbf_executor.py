# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the one-EU RSBF executor (``loop/rsbf_executor.py``).

The reference is the eager ``HybridModel`` + ``LoopHybridStack`` under the standard schedule's
microbatch path (``forward_step_calc_loss`` and ``backward_step``, one microbatch per sample,
row depth set with ``set_loop_depths``), with ``pretrain_hybrid.loss_func`` in both arms and
the same initial weights. Per-sample losses are compared bitwise; parameter gradients in fp32
at rtol 1e-5 with an absolute floor of 1e-6 x max |g| per tensor (the executor runs one
backward per round item, so a parameter's per-round gradients are summed in a different
association than the eager graph's).

CPU (gloo) cases use local layers: dense, padded, MTP, RoPE and 4-stream mHC models. Further
cases: an online (learned) exit criterion against an unrolled autograd oracle, MBS / waves /
fill variants, replay and statistics-weight side effects, zero core work after exit with
buggy host plans, the host-sync count, the token normalizer and aux-loss scales, store
hygiene with poisoning and recompute checks, loss validation, FLOP records and the schedule
entry. GPU cases: the DSv4-like pattern and the sync-debug gate.
"""

import contextlib
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from megatron.core.pipeline_parallel import schedules
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.forward_replay import in_forward_replay, statistics_weight
from megatron.core.transformer.loop import rsbf_executor
from megatron.core.transformer.loop.program import (
    LoopPhase,
    LoopProgram,
    PrescribedDepthCriterion,
    SampledDepthCriterion,
    SampleInputs,
    SampleKey,
    global_sample_index,
)
from megatron.core.transformer.loop.rsbf_executor import (
    PLAN_MISMATCH,
    RSBF_BATCH_FUNC_ATTR,
    SWEEP_AFTER_EXIT,
    LoopMicrobatch,
    LoopRSBFExecutor,
    LoopRSBFSchedule,
    forward_backward_loop_rsbf,
    get_loop_rsbf_schedule,
)
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler
from tests.unit_tests.transformer.loop import hybrid_loop_utils
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    CUDA,
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
    retire_early,
    retire_late,
)

# The CPU mHC, MTP and RoPE fixtures, made visible to pytest in this module.
mhc_on_this_device = hybrid_loop_utils.mhc_on_this_device
cpu_mtp = hybrid_loop_utils.cpu_mtp
cpu_rope = hybrid_loop_utils.cpu_rope

DENSE = "*-*-*-*-"  # prelude '*-', core '*-*-', coda '*-'
MHC = dict(enable_mhc_connections=True, mhc_num_residual_streams=4, mhc_sinkhorn_iterations=5)
RSBF = dict(loop_schedule="rsbf", calculate_per_token_loss=True)
R_MAX = 4
DEPTHS = (1, 3, 2, 4)

gpu = pytest.mark.skipif(not CUDA, reason="TE / MoE / DSv4 layers and sync-debug need CUDA")


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        try:
            yield
        finally:
            get_loop_rsbf_schedule().reset()  # no stale groups for later modules


@pytest.fixture
def hybrid(monkeypatch):
    """``pretrain_hybrid`` with stub training args (its real ``loss_func`` in both arms)."""
    import pretrain_hybrid

    args = SimpleNamespace(
        logits_load_dir=None, check_for_nan_in_loss_and_grad=True, check_for_spiky_loss=False
    )
    monkeypatch.setattr(pretrain_hybrid, "get_args", lambda: args)
    monkeypatch.setattr(pretrain_hybrid, "has_nvidia_modelopt", False, raising=False)
    monkeypatch.setattr(pretrain_hybrid, "test_args", args, raising=False)
    return pretrain_hybrid


@pytest.fixture
def deterministic_algorithms(monkeypatch):
    """Deterministic kernels for the bitwise comparisons on GPU (the embedding backward
    accumulates with atomics otherwise); a no-op for the CPU kernels used here."""
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


def _model(name, request, **overrides):
    """The model a case name asks for (as in test_loop_program): ``mtp`` adds an MTP layer,
    ``mhc`` 4 mHC streams, ``rope`` model-level RoPE; always an RSBF config."""
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
    overrides.setdefault("loop_iterations", R_MAX)
    return build_local_loop_model(pattern, **RSBF, **overrides)


def _batches(samples, mbs=1, *, padded=False, seed=5, seq=SEQ, vocab=VOCAB, dev=None):
    """``samples`` rows of random tokens in microbatches of ``mbs`` rows; the loss mask drops
    a few tokens; with ``padded`` the last quarter of every odd row is padding."""
    dev = dev or device()
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, vocab, (samples, seq), generator=generator)
    labels = torch.randint(0, vocab, (samples, seq), generator=generator)
    loss_mask = (torch.rand(samples, seq, generator=generator) > 0.2).float()
    padding_mask = None
    if padded:
        padding_mask = torch.zeros(samples, seq, dtype=torch.bool)
        padding_mask[1::2, -(seq // 4) :] = True
        loss_mask = loss_mask * (~padding_mask).float()
    rows = dict(
        tokens=tokens,
        labels=labels,
        position_ids=torch.arange(seq).repeat(samples, 1),
        loss_mask=loss_mask,
        padding_mask=padding_mask,
    )
    return [
        {
            name: None if value is None else value[start : start + mbs].to(dev)
            for name, value in rows.items()
        }
        for start in range(0, samples, mbs)
    ]


def _rows(batches):
    """The batches as one-row microbatches (the eager reference runs MBS = 1)."""
    rows = []
    for batch in batches:
        for row in range(batch["tokens"].shape[0]):
            rows.append(
                {
                    name: None if value is None else value[row : row + 1]
                    for name, value in batch.items()
                }
            )
    return rows


def _grads(*modules):
    grads = {}
    for index, module in enumerate(modules):
        for name, param in module.named_parameters():
            if param.grad is not None:
                grads[f"{index}.{name}"] = param.grad.detach().clone()
        module.zero_grad(set_to_none=True)
    return grads


def _assert_grads_close(actual, expected, abs_sum=None, rtol=1e-5):
    """The M1 gradient gate: rtol 1e-5 with an absolute floor per tensor of 1e-6 x max |g|,
    or, given ``abs_sum`` (the per-sample gradients' absolute sum, ``_abs_sample_grads``),
    1e-6 x max sum_i |g_i|: the fp32 error of re-associating a sum is bounded by its addends,
    not by its result, and some parameters' per-sample gradients cancel (mHC biases)."""
    assert actual.keys() == expected.keys(), actual.keys() ^ expected.keys()
    for name, grad in expected.items():
        scale = grad if abs_sum is None else abs_sum[name]
        floor = 1e-6 * scale.abs().max().item()
        torch.testing.assert_close(
            actual[name], grad, rtol=rtol, atol=floor, msg=lambda text, name=name: f"{name}: {text}"
        )


def _assert_grads_equal(actual, expected):
    assert actual.keys() == expected.keys(), actual.keys() ^ expected.keys()
    for name, grad in expected.items():
        assert torch.equal(actual[name], grad), name


def _eager(model, hybrid, batches, depths):
    """The eager reference under the standard schedule's microbatch path (MBS = 1, one
    microbatch per sample): ``forward_step_calc_loss`` + ``backward_step``. Returns the
    per-sample 'lm loss' reports and the gradients."""
    config = model.config
    model.zero_grad(set_to_none=True)
    rows = _rows(batches)
    reports = []
    for row, depth in zip(rows, depths):
        model.decoder.set_loop_depths(torch.tensor([depth], device=row["tokens"].device))
        output = model(
            row["tokens"],
            row["position_ids"],
            None,
            labels=row["labels"],
            loss_mask=row["loss_mask"],
            padding_mask=row["padding_mask"],
        )
        loss, _ = schedules.forward_step_calc_loss(
            model,
            output,
            partial(hybrid.loss_func, row["loss_mask"], model=model),
            config,
            None,
            False,
            len(rows),
            reports,
            cp_group_size=1,
            is_last_stage=True,
        )
        schedules.backward_step(None, loss, None, config)
    return [report["lm loss"] for report in reports], _grads(model)


def _abs_sample_grads(model, hybrid, batches, depths):
    """``sum_i |g_i|`` over the samples' separate eager gradients (the scale of the fp32
    re-association error of their sum)."""
    total = {}
    for row, depth in zip(_rows(batches), depths):
        for name, grad in _eager(model, hybrid, [row], [depth])[1].items():
            total[name] = total[name] + grad.abs() if name in total else grad.abs()
    return total


def _microbatches(model, hybrid, batches, loss_funcs=None):
    loss_funcs = loss_funcs or [None] * len(batches)
    return [
        LoopMicrobatch(
            tokens=batch["tokens"],
            position_ids=batch["position_ids"],
            labels=batch["labels"],
            loss_mask=batch["loss_mask"],
            attention_mask=None,
            padding_mask=batch["padding_mask"],
            loss_func=loss_func or partial(hybrid.loss_func, model=model, validate=False),
        )
        for batch, loss_func in zip(batches, loss_funcs)
    ]


def _next_batch(iterator, model):
    del model
    return next(iterator)


def _executor(model, depths=None, *, criterion=None, check_ports=True, **kwargs):
    criterion = criterion or PrescribedDepthCriterion(dict(enumerate(depths)))
    program = LoopProgram(model, exit_criterion=criterion, check_ports=check_ports)
    return LoopRSBFExecutor(program, pg_collection=SimpleNamespace(dp=None), **kwargs)


def _rsbf(
    model,
    hybrid,
    batches,
    depths=None,
    *,
    criterion=None,
    extra_modules=(),
    loss_funcs=None,
    iteration=0,
    validate=None,
    **kwargs,
):
    """One RSBF step; returns the per-sample 'lm loss' reports, the executor and the grads."""
    executor = _executor(model, depths, criterion=criterion, **kwargs)
    model.zero_grad(set_to_none=True)
    reports = executor.run_step(
        _next_batch,
        iter(_microbatches(model, hybrid, batches, loss_funcs)),
        len(batches),
        model=model,
        iteration=iteration,
        validate_loss=validate,
    )
    return reports, executor, _grads(model, *extra_modules)


def _lm_losses(reports):
    return [report["lm loss"] for report in reports]


def _assert_losses_equal(actual, expected):
    assert len(actual) == len(expected)
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert torch.equal(got, want), (index, got, want)


# ---------------------------------------------------------------------------
# E1 / E2: equal to the eager reference under the standard schedule
# ---------------------------------------------------------------------------

CASES = ["dense", "dense_padded", "mtp", "mhc", "mhc_mtp", "mtp_rope"]


@pytest.mark.parametrize("fill", ["drain", "sample_order"])
@pytest.mark.parametrize("name", CASES)
def test_rsbf_equals_the_eager_reference(request, hybrid, name, fill, deterministic_algorithms):
    """Prescribed depths [1, 3, 2, 4] (R_max 4): per-sample losses bitwise, parameter
    gradients (embedding, prelude, core, adapter, loop norm, coda, head, MTP) within the gate;
    every sample's chain ran once and nothing was swept after its exit."""
    model = _model(name, request)
    batches = _batches(len(DEPTHS), padded=name.endswith("padded"))
    expected_losses, expected = _eager(model, hybrid, batches, DEPTHS)
    reports, executor, actual = _rsbf(model, hybrid, batches, DEPTHS, fill=fill)
    _assert_losses_equal(_lm_losses(reports), expected_losses)
    _assert_grads_close(actual, expected, _abs_sample_grads(model, hybrid, batches, DEPTHS))
    assert "0.decoder.loop_adapter.weight" in actual
    assert "0.embedding.word_embeddings.weight" in actual
    stats = executor.stats
    assert [stats.depths[SampleKey(i, 0)] for i in range(len(DEPTHS))] == list(DEPTHS)
    kinds = [kind for kind, _, _ in stats.item_order]
    assert kinds.count("coda") == kinds.count("prelude") == len(DEPTHS)
    assert kinds.count("core") == sum(DEPTHS)
    assert executor.program.after_exit_violations.item() == 0
    if fill == "drain":
        # Chains in (exit round, sample index) order.
        order = [index for kind, index, _ in stats.item_order if kind == "coda"]
        assert order == sorted(range(len(DEPTHS)), key=lambda i: (DEPTHS[i], i))
    else:
        order = [index for kind, index, _ in stats.item_order if kind == "coda"]
        assert order == list(range(len(DEPTHS)))


# ---------------------------------------------------------------------------
# E3: an online (learned) exit criterion
# ---------------------------------------------------------------------------


def _halting(model, **kwargs):
    stack = model.decoder
    head = HaltingHead(stack.loop_num_streams, model.config.hidden_size, device=device(), seed=3)
    kwargs.setdefault("tau", 1.4)
    kwargs.setdefault("carry", 0.05)
    return OnlineHaltingCriterion(head, **kwargs)


def _online_oracle(model, hybrid, batches, criterion):
    """One autograd graph per sample over the cells (= the eager forward op for op): rounds
    until the cell's exit bit, objective ``loss + num_tokens * sum_r aux_r``. Returns the
    depths, the 'lm loss' reports and the gradients (model and halting head)."""
    program = LoopProgram(model, exit_criterion=criterion)
    model.zero_grad(set_to_none=True)
    depths, losses = [], []
    for index, row in enumerate(_rows(batches)):
        sample = SampleInputs(
            key=SampleKey(index, 0),
            tokens=row["tokens"],
            position_ids=row["position_ids"],
            labels=row["labels"],
            loss_mask=row["loss_mask"],
            attention_mask=None,
            padding_mask=row["padding_mask"],
            loss_func=partial(hybrid.loss_func, model=model, validate=False),
        )
        prelude = program.prelude(sample, phase=LoopPhase.SWEEP)
        state, crit, aux = None, prelude.crit_state, []
        for r in range(1, program.max_rounds + 1):
            out = program.core(
                state,
                prelude.injection,
                sample,
                r=r,
                phase=LoopPhase.SWEEP,
                crit_state=crit,
                exited=torch.zeros(1, dtype=torch.int32, device=device()),
            )
            state, crit = out.state, out.crit_state
            aux.append(out.aux_loss)
            if out.exit.item():
                depths.append(r)
                break
        coda = program.coda(state, sample)
        objective = coda.loss
        for value in aux:
            objective = objective + value * coda.num_tokens
        objective.backward()
        losses.append(coda.report["lm loss"])
    return depths, losses, _grads(model, criterion.head)


@pytest.mark.parametrize("fill", ["drain", "sample_order"])
def test_online_criterion_matches_an_unrolled_oracle(
    request, hybrid, fill, deterministic_algorithms
):
    """4-stream mHC with a learned halting criterion (online: exit bits read every round):
    realized depths, per-sample losses and the gradients of the model, the halting head's
    round weights (aux loss) and its init_state weights (dL/dcrit_0 through the chain) equal
    an unrolled autograd oracle; one host sync per forward round plus one per step."""
    model = _model("mhc", request)
    criterion = _halting(model)
    batches = _batches(6, seed=11)
    depths, expected_losses, expected = _online_oracle(model, hybrid, batches, criterion)
    assert len(set(depths)) > 1, f"the halting head should give mixed depths, got {depths}"
    reports, executor, actual = _rsbf(
        model, hybrid, batches, criterion=criterion, extra_modules=(criterion.head,), fill=fill
    )
    stats = executor.stats
    assert [stats.depths[SampleKey(i, 0)] for i in range(len(depths))] == depths
    _assert_losses_equal(_lm_losses(reports), expected_losses)
    assert {"1.w", "1.v"} <= expected.keys()
    _assert_grads_close(actual, expected)
    assert stats.host_syncs == max(depths) + 1
    assert stats.host_syncs_by_wave == [max(depths)] == stats.rounds_by_wave


# ---------------------------------------------------------------------------
# E4: MBS, waves and fill variants (the waves gate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["dense", "mhc_mtp"])
def test_mbs_waves_and_fill_variants_agree(request, hybrid, name, deterministic_algorithms):
    """MBS 1 / 2, waves 1 / 2 and drain / sample_order (the M1 waves gate): per-sample losses
    bitwise (c = 1: every cell call sees one row); gradients bitwise between arms that run
    the same item sequence and within the gate otherwise (the per-item addends are equal,
    only their accumulation order differs)."""
    model = _model(name, request)
    depths = DEPTHS + (2, 1, 3, 4)
    arms = {
        "mbs1-w1-drain": dict(mbs=1, waves=1, fill="drain"),
        "mbs2-w1-drain": dict(mbs=2, waves=1, fill="drain"),
        "mbs1-w2-drain": dict(mbs=1, waves=2, fill="drain"),
        "mbs2-w2-order": dict(mbs=2, waves=2, fill="sample_order"),
        "mbs1-w1-order": dict(mbs=1, waves=1, fill="sample_order"),
    }
    results = {}
    for arm, options in arms.items():
        batches = _batches(len(depths), options["mbs"])
        reports, executor, grads = _rsbf(
            model, hybrid, batches, depths, waves=options["waves"], fill=options["fill"]
        )
        assert len(executor.stats.store_peaks) == options["waves"]
        results[arm] = (_lm_losses(reports), grads, executor.stats.item_order)
    abs_sum = _abs_sample_grads(model, hybrid, _batches(len(depths)), depths)
    base_losses, base_grads, base_order = results["mbs1-w1-drain"]
    assert results["mbs2-w1-drain"][2] == base_order  # MBS does not change the schedule
    assert results["mbs2-w2-order"][2] == results["mbs1-w1-order"][2] != base_order
    assert results["mbs1-w2-drain"][2] != base_order
    for arm, (losses, grads, order) in results.items():
        _assert_losses_equal(losses, base_losses)
        if order == base_order:
            _assert_grads_equal(grads, base_grads)
        else:
            _assert_grads_close(grads, base_grads, abs_sum)
    _assert_grads_equal(results["mbs2-w2-order"][1], results["mbs1-w1-order"][1])


# ---------------------------------------------------------------------------
# E5 / E6: side effects, zero core work after exit
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _layer_spy(model):
    """Forward pre-hooks on every decoder sublayer recording (index, replay, grad mode,
    statistics weight) per call."""
    calls = []
    handles = [
        layer.register_forward_pre_hook(
            lambda module, args, index=index: calls.append(
                (index, in_forward_replay(), torch.is_grad_enabled(), statistics_weight())
            )
        )
        for index, layer in enumerate(model.decoder.layers)
    ]
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()


def test_sweeps_count_statistics_and_items_replay(request, hybrid):
    """Sweeps and the coda run outside the replay gate, CORE / PRELUDE items inside it; every
    call has grad enabled; the statistics weight is 1 / MBS during the step and 1 after. Each
    core sublayer runs once per sample-round in the sweep and once in the items."""
    model = _model("dense", request)
    roles = model.decoder.loop_roles
    with _layer_spy(model) as calls:
        _rsbf(model, hybrid, _batches(len(DEPTHS), 2), DEPTHS)
    assert statistics_weight() == 1.0 and not in_forward_replay()
    assert all(grad and weight == 0.5 for _, _, grad, weight in calls)
    n, total = len(DEPTHS), sum(DEPTHS)
    for index in range(len(model.decoder.layers)):
        replays = [replay for layer, replay, _, _ in calls if layer == index]
        if index < roles.core_start:  # prelude: sweep + PRELUDE item
            assert (replays.count(False), replays.count(True)) == (n, n), index
        elif index < roles.core_end:  # core: one sweep and one item per sample-round
            assert (replays.count(False), replays.count(True)) == (total, total), index
        else:  # coda: the CODA item is its only forward
            assert (replays.count(False), replays.count(True)) == (n, 0), index


def test_buggy_host_plans_are_caught(request, hybrid, monkeypatch):
    """A host plan that sweeps a sample after its exit fails with the after-exit counter; one
    that retires a sample before its device exit fails the plan check; both at the step's
    sync (known depths) or at the round's sync (online)."""
    model = _model("dense", request)
    batches = _batches(len(DEPTHS))
    original = LoopRSBFExecutor._newly_exited
    monkeypatch.setattr(LoopRSBFExecutor, "_newly_exited", retire_late(original, sample=0))
    with pytest.raises(RuntimeError, match=SWEEP_AFTER_EXIT):
        _rsbf(model, hybrid, batches, DEPTHS)
    # Online: caught at the next round's exit-bit read.
    monkeypatch.setattr(LoopRSBFExecutor, "_newly_exited", retire_late(original, sample=0))
    with pytest.raises(RuntimeError, match=SWEEP_AFTER_EXIT) as excinfo:
        _rsbf(model, hybrid, batches, criterion=ConstantExitCriterion(True))
    assert "1 sample-round(s)" in str(excinfo.value)
    monkeypatch.setattr(
        LoopRSBFExecutor, "_newly_exited", retire_early(original, sample=1, at_round=1)
    )
    with pytest.raises(RuntimeError, match=PLAN_MISMATCH):
        _rsbf(model, hybrid, batches, DEPTHS)
    monkeypatch.setattr(LoopRSBFExecutor, "_newly_exited", original)
    # The correct plan passes again on the same model (the store and counters recover).
    _, executor, _ = _rsbf(model, hybrid, batches, DEPTHS)
    assert executor.program.after_exit_violations.item() == 0


def test_core_plan_is_consulted_by_sweep_and_item(request, hybrid, monkeypatch):
    """C9: the sweep and the CORE item of every (sample, round) read the same
    ``program.core_plan`` (what to stash / what to replay), once each."""
    model = _model("dense", request)
    executor = _executor(model, DEPTHS)
    calls = []
    real = executor.program.core_plan
    monkeypatch.setattr(
        executor.program,
        "core_plan",
        lambda sample, r: (calls.append((sample.key.global_index, r)), real(sample, r))[1],
    )
    executor.run_step(
        _next_batch,
        iter(_microbatches(model, hybrid, _batches(len(DEPTHS)))),
        len(DEPTHS),
        model=model,
    )
    expected = sorted((i, r) for i, depth in enumerate(DEPTHS) for r in range(1, depth + 1))
    assert sorted(calls) == sorted(expected * 2)


# ---------------------------------------------------------------------------
# E7: host syncs
# ---------------------------------------------------------------------------


def test_host_syncs_per_step(request, hybrid):
    """Known depths: one host sync per step, for any number of waves. Online: one per
    forward round of each wave plus one per step."""
    model = _model("dense", request)
    batches = _batches(8)
    depths = DEPTHS + (4, 2, 1, 1)
    for waves in (1, 2):
        _, executor, _ = _rsbf(model, hybrid, batches, depths, waves=waves)
        assert executor.stats.host_syncs == 1
        assert executor.stats.host_syncs_by_wave == [0] * waves
    criterion = _halting(model, tau=1.3)
    for waves in (1, 2):
        _, executor, _ = _rsbf(
            model,
            hybrid,
            batches,
            criterion=criterion,
            extra_modules=(criterion.head,),
            waves=waves,
        )
        stats = executor.stats
        per_wave = len(batches) // waves
        realized = [stats.depths[SampleKey(i, 0)] for i in range(len(batches))]
        assert stats.rounds_by_wave == [
            max(realized[w * per_wave : (w + 1) * per_wave]) for w in range(waves)
        ]
        assert stats.host_syncs_by_wave == stats.rounds_by_wave
        assert stats.host_syncs == sum(stats.rounds_by_wave) + 1


# ---------------------------------------------------------------------------
# E8: normalizer, records, loss scales
# ---------------------------------------------------------------------------


def test_normalizer_records_and_loss_scales(request, hybrid, monkeypatch, deterministic_algorithms):
    """finalize receives the local token count (the CODA items' num_tokens sum, int32, not
    reduced) once per step; reports are per sample in sample order with one key set; the
    aux-loss scales are set once per step from grad_scale_func; a scaled loss gives exactly
    scaled gradients."""
    model = _model("mtp", request)
    config = model.config
    batches = _batches(len(DEPTHS), 2)
    _, _, unscaled = _rsbf(model, hybrid, batches, DEPTHS)

    finalized, scale_calls = [], []
    monkeypatch.setattr(
        config,
        "finalize_model_grads_func",
        lambda models, num_tokens, pg_collection=None, force_all_reduce=False: finalized.append(
            (models, num_tokens.clone(), pg_collection, force_all_reduce)
        ),
    )
    monkeypatch.setattr(config, "grad_scale_func", lambda loss: loss * 2.0)
    real_scales = rsbf_executor.set_auxiliary_loss_scales
    monkeypatch.setattr(
        rsbf_executor,
        "set_auxiliary_loss_scales",
        lambda *args, **kwargs: (scale_calls.append(kwargs), real_scales(*args, **kwargs)),
    )
    monkeypatch.setattr(MTPLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))
    reports, executor, scaled = _rsbf(model, hybrid, batches, DEPTHS)

    assert len(finalized) == 1 and len(scale_calls) == 1
    models, num_tokens, pg_collection, force_all_reduce = finalized[0]
    assert models == [model] and pg_collection is executor.pg_collection and not force_all_reduce
    assert num_tokens.dtype == torch.int32 and num_tokens.dim() == 0
    assert num_tokens.item() == sum(int(report["lm loss"][1]) for report in reports)
    assert float(MTPLossAutoScaler.main_loss_backward_scale) == 2.0
    keys = set(reports[0])
    assert all(set(report) == keys for report in reports)
    assert {
        "lm loss",
        "loop forward flops per token",
        "loop backward flops per token",
        "loop recompute flops per token",
        "loop depth",
        "loop active token layers",
        "loop backward token layers",
        "loop sample moves",
    } == keys
    records = executor.stats.loss_records
    assert [record.key.global_index for record in records] == list(range(len(DEPTHS)))
    assert [record.depth for record in records] == list(DEPTHS)
    for report, record, depth in zip(reports, records, DEPTHS):
        assert torch.equal(report["lm loss"][0], record.loss)
        # Per issued token (loop/reporting.py): [depth x tokens, tokens].
        assert report["loop depth"].tolist() == [float(depth * SEQ), float(SEQ)]
        assert report["loop sample moves"].tolist() == [0.0, float(depth)]  # one EU: no moves
    _assert_grads_equal(scaled, {name: 2 * grad for name, grad in unscaled.items()})


# ---------------------------------------------------------------------------
# E9: store hygiene, recompute checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["dense_padded", "mhc_mtp"])
def test_poisoned_store_and_recompute_checks_change_nothing(
    request, hybrid, name, deterministic_algorithms
):
    """NaN-poisoned frees and check_recompute (every replayed round state and injection equals
    the sweep's bitwise; the RNG states are unchanged across each wave) give bitwise the plain
    run's losses and gradients; every wave ends with an empty store."""
    model = _model(name, request)
    batches = _batches(len(DEPTHS), padded=name.endswith("padded"))
    plain_reports, plain, plain_grads = _rsbf(model, hybrid, batches, DEPTHS, waves=2)
    reports, checked, grads = _rsbf(
        model, hybrid, batches, DEPTHS, waves=2, check_recompute=True, poison_on_free=True
    )
    _assert_losses_equal(_lm_losses(reports), _lm_losses(plain_reports))
    _assert_grads_equal(grads, plain_grads)
    assert checked.stats.debug_syncs == sum(DEPTHS) - len(DEPTHS) + 2 * len(DEPTHS)
    for stats in checked.stats.store_peaks + plain.stats.store_peaks:
        assert stats.live_entries == 0 and stats.live_bytes == 0 and stats.peak_bytes > 0
    assert checked.store.wave is None


# ---------------------------------------------------------------------------
# E10: loss validation once per step
# ---------------------------------------------------------------------------


def test_loss_is_validated_once_per_step_on_a_host_tensor(request, hybrid, monkeypatch):
    """A NaN in one sample's loss raises the fatal rerun-state-machine error at the end of
    the step, before finalize; validation runs once per step on a CPU tensor (NaN, Inf and
    spiky checks), never per sample."""
    from megatron.core.rerun_state_machine import get_rerun_state_machine

    model = _model("dense", request)
    batches = _batches(len(DEPTHS))
    if not CUDA:  # the fatal message names the CUDA device
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    machine = get_rerun_state_machine()
    checked = []
    real_validate = machine.validate_result
    monkeypatch.setattr(
        machine,
        "validate_result",
        lambda result, **kwargs: (
            checked.append((result.device.type, kwargs["message"])),
            real_validate(result=result, **kwargs),
        ),
    )
    finalized = []
    monkeypatch.setattr(
        model.config, "finalize_model_grads_func", lambda *args, **kwargs: finalized.append(1)
    )
    hybrid.test_args.check_for_spiky_loss = True
    _rsbf(model, hybrid, batches, DEPTHS, validate=hybrid.validate_loss)
    assert [message for _, message in checked] == [
        "found NaN in local forward loss calculation",
        "found Inf in local forward loss calculation",
        "Spiky loss",
    ]
    assert {kind for kind, _ in checked} == {"cpu"} and finalized == [1]

    base = partial(hybrid.loss_func, model=model, validate=False)

    def nan_loss(loss_mask, output):
        return base(loss_mask, output + float("nan"))

    checked.clear()
    finalized.clear()
    with pytest.raises(RuntimeError, match="found NaN in local forward loss calculation"):
        _rsbf(
            model,
            hybrid,
            batches,
            DEPTHS,
            validate=hybrid.validate_loss,
            loss_funcs=[None, None, nan_loss, None],
        )
    assert len(checked) == 1 and finalized == []


# ---------------------------------------------------------------------------
# F1: FLOP records
# ---------------------------------------------------------------------------


def test_flop_records_follow_the_per_symbol_table(request, hybrid):
    """Per sample: forward = F_P + d F_C + F_K + head, recompute = F_P + d F_C, backward =
    2 (F_P + d F_C + F_K) + head backward; forward and backward equal the eager stack's
    loop_metrics at the same depth; the report entries carry them per issued token."""
    model = _model("mtp", request)
    stack, roles = model.decoder, model.decoder.loop_roles
    batches = _batches(len(DEPTHS))
    reports, executor, _ = _rsbf(model, hybrid, batches, DEPTHS)
    tokens, squared = SEQ, SEQ * SEQ
    f_p = stack.range_forward_flops(0, roles.core_start, tokens, squared, SEQ)
    f_c = stack.range_forward_flops(
        roles.core_start,
        roles.core_end,
        tokens,
        squared,
        SEQ,
        extra=stack.round_extra_forward_flops(tokens),
    )
    f_k = stack.range_forward_flops(roles.core_end, len(stack.layers), tokens, squared, SEQ)
    head = stack.head_work_terms(tokens, squared, SEQ)
    assert len(head) == 3  # logits, MTP depth, MTP logits
    head_forward = sum(work for work, _ in head)
    head_backward = sum(factor * work for work, factor in head)
    rows = _rows(batches)
    for index, (depth, report) in enumerate(zip(DEPTHS, reports)):
        records = [record for record in executor.stats.work if record.sample == index]
        forward = sum(record.forward for record in records)
        backward = sum(record.backward for record in records)
        recompute = sum(record.recompute for record in records)
        assert forward == pytest.approx(f_p + depth * f_c + f_k + head_forward, rel=1e-12)
        assert recompute == pytest.approx(f_p + depth * f_c, rel=1e-12)
        assert backward == pytest.approx(2 * (f_p + depth * f_c + f_k) + head_backward, rel=1e-12)
        assert report["loop forward flops per token"].tolist() == pytest.approx(
            [forward, SEQ], rel=1e-6
        )
        assert report["loop recompute flops per token"][0].item() == pytest.approx(
            recompute, rel=1e-6
        )
        assert report["loop active token layers"].tolist() == [SEQ * depth * roles.core, SEQ]
        assert report["loop backward token layers"].tolist() == [SEQ * depth * roles.core, SEQ]
        # The eager stack's FLOP sums at the same depth (no loop_recompute: recompute 0).
        stack.set_loop_depths(torch.tensor([depth], device=device()))
        row = rows[index]
        model(row["tokens"], row["position_ids"], None, labels=row["labels"])
        metrics = stack.loop_metrics
        assert metrics["loop forward flops per token"].item() * SEQ == pytest.approx(
            forward, rel=1e-6
        )
        assert metrics["loop backward flops per token"].item() * SEQ == pytest.approx(
            backward, rel=1e-6
        )
        assert metrics["loop recompute flops per token"].item() == 0.0
    kinds = [record.kind for record in executor.stats.work]
    assert kinds.count("core_sweep") == kinds.count("core_item") == sum(DEPTHS)


# ---------------------------------------------------------------------------
# The schedule entry
# ---------------------------------------------------------------------------


def _forward_step_func(model, hybrid, batches):
    microbatches = _microbatches(model, hybrid, batches)

    def forward_step(data_iterator, model):
        raise AssertionError("the RSBF schedule must not call forward_step")

    forward_step.loop_rsbf_batch_func = _next_batch
    return forward_step, iter(microbatches)


def _schedule_kwargs(model, forward_step, iterator, num_microbatches, **overrides):
    """Exactly the keywords train_step passes to forward_backward_func."""
    kwargs = dict(
        forward_step_func=forward_step,
        data_iterator=[iterator],
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=SEQ,
        micro_batch_size=1,
        decoder_seq_length=SEQ,
        forward_only=False,
        adjust_tensor_shapes_fn=None,
        p2p_communicator=None,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        force_all_reduce=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_schedule_entry_runs_sampled_depths_per_iteration(request, hybrid):
    """forward_backward_loop_rsbf takes the training keywords plus ``iteration``; the
    config's sampled criterion draws per (seed, iteration, global index); the module schedule
    caches one executor per model."""
    model = _model(
        "dense",
        request,
        loop_exit_criterion="sampled",
        loop_depth_probs=[0.25, 0.25, 0.25, 0.25],
        loop_depth_seed=3,
    )
    batches = _batches(8)
    schedule = get_loop_rsbf_schedule()
    executors = []
    for iteration in (7, 8):
        forward_step, iterator = _forward_step_func(model, hybrid, batches)
        reports = forward_backward_loop_rsbf(
            iteration=iteration, **_schedule_kwargs(model, forward_step, iterator, len(batches))
        )
        assert schedule.iteration == iteration
        executor = schedule.cached_executor(model)
        executors.append(executor)
        # Global sample positions: the module's DP group spans every rank on GPU (DP 8).
        dp_group = ProcessGroupCollection.use_mpu_process_groups().dp
        dp = dict(
            dp_rank=torch.distributed.get_rank(dp_group),
            dp_size=torch.distributed.get_world_size(dp_group),
        )
        keys = [
            SampleKey(global_sample_index(index, 0, micro_batch_size=1, **dp), iteration)
            for index in range(len(batches))
        ]
        expected = SampledDepthCriterion([0.25] * 4, seed=3).known_depths(keys)
        assert len(set(expected)) > 1
        assert [executor.stats.depths[key] for key in keys] == expected
        assert [
            report["loop depth"][0].item() / report["loop depth"][1].item() for report in reports
        ] == expected
    assert executors[0] is executors[1]


def test_schedule_cache_does_not_keep_the_model_alive(hybrid):
    import gc
    import weakref

    schedule = LoopRSBFSchedule()
    model = build_local_loop_model(DENSE, loop_iterations=R_MAX, **RSBF)
    forward_step, iterator = _forward_step_func(model, hybrid, _batches(2))
    schedule(**_schedule_kwargs(model, forward_step, iterator, 2))
    assert schedule.cached_executor(model) is not None
    alive = weakref.ref(model)
    del model, forward_step, iterator
    gc.collect()
    assert alive() is None and len(schedule._cache) == 0


def test_schedule_rejects_what_it_cannot_run(request, hybrid):
    model = _model("dense", request)
    batches = _batches(2)
    forward_step, iterator = _forward_step_func(model, hybrid, batches)
    schedule = LoopRSBFSchedule()
    with pytest.raises(ValueError, match="training steps only"):
        schedule(**_schedule_kwargs(model, forward_step, iterator, 2, forward_only=True))
    del forward_step.loop_rsbf_batch_func
    with pytest.raises(ValueError, match=RSBF_BATCH_FUNC_ATTR):
        schedule(**_schedule_kwargs(model, forward_step, iterator, 2))
    forward_step.loop_rsbf_batch_func = _next_batch
    with pytest.raises(ValueError, match="must divide"):
        executor = _executor(model, DEPTHS, waves=2)
        executor.run_step(_next_batch, iterator, 3, model=model)
    with pytest.raises(ValueError, match="at least one microbatch"):
        _executor(model, DEPTHS).run_step(_next_batch, iterator, 0, model=model)
    standard = build_local_loop_model(DENSE, loop_iterations=R_MAX)
    with pytest.raises(ValueError, match="Loop RSBF does not support: calculate_per_token_loss"):
        _executor(standard, DEPTHS)


# ---------------------------------------------------------------------------
# GPU: the DSv4-like pattern and the sync-debug gate
# ---------------------------------------------------------------------------


def _dsv4_model(**overrides):
    """'WECEHE/WE' (DSv4 W / C / H attention, unfused; MoE alltoall with one hash layer in the
    prelude and expert bias; mHC 4; MTP 1; indexer loss on), bf16, RSBF config."""
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_dsv4_stack_spec
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
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=1e-4,
        moe_token_dispatcher_type="alltoall",
        moe_n_hash_layers=1,
        actual_vocab_size=VOCAB,
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
        **RSBF,
    )
    fields.update(overrides)
    config = MLATransformerConfig(**fields)
    torch.manual_seed(1234)
    return HybridModel(
        config=config,
        hybrid_stack_spec=loop_hybrid_dsv4_stack_spec(config),
        vocab_size=VOCAB,
        max_sequence_length=1024,
        hybrid_layer_pattern="WECEHE/WE",
        position_embedding_type="rope",
    ).cuda()


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


def _expert_bias_counts(model):
    return {
        name: module.local_tokens_per_expert.clone()
        for name, module in model.named_modules()
        if getattr(module, "local_tokens_per_expert", None) is not None
    }


@gpu
@pytest.mark.parametrize("fill", ["drain", "sample_order"])
def test_dsv4_rsbf_equals_the_eager_reference(hybrid, deterministic_algorithms, fill):
    """S1 / S2: DSv4-like 'WECEHE/WE' in bf16, deterministic env, prescribed depths
    [2, 1, 2, 2] at R_max 2: per-sample losses bitwise; replayed rounds equal the sweep
    (check_recompute); expert-bias token counts equal the eager reference's (one count per
    sample-round); gradients by the bf16 rule against an fp32-accumulated reference: within
    two bf16 ulps of the eager gradients normwise."""
    depths = (2, 1, 2, 2)
    with _hadamard_fallback():
        model = _dsv4_model()
        batches = _batches(len(depths), seq=256, dev=torch.device("cuda"))
        set_moe_metrics_tracker(MoEMetricsTracker())
        for counts in _expert_bias_counts(model).values():
            assert counts.abs().sum().item() == 0
        expected_losses, expected = _eager(model, hybrid, batches, depths)
        eager_counts = _expert_bias_counts(model)
        for name, module in model.named_modules():
            if getattr(module, "local_tokens_per_expert", None) is not None:
                module.local_tokens_per_expert.zero_()
        set_moe_metrics_tracker(MoEMetricsTracker())
        reports, executor, actual = _rsbf(
            model, hybrid, batches, depths, fill=fill, check_recompute=True
        )
        rsbf_counts = _expert_bias_counts(model)
    if not all(torch.equal(a, b) for a, b in zip(_lm_losses(reports), expected_losses)):
        rerun, _ = _eager(model, hybrid, batches, depths)
        if not all(torch.equal(a, b) for a, b in zip(rerun, expected_losses)):
            pytest.fail("NONDETERMINISTIC: the eager model does not reproduce its losses")
        pytest.fail("RSBF per-sample losses differ from the eager reference")
    assert eager_counts and eager_counts.keys() == rsbf_counts.keys()
    for name, counts in eager_counts.items():
        assert torch.equal(rsbf_counts[name], counts), name
    assert executor.stats.debug_syncs > 0
    assert actual.keys() == expected.keys()
    for name, grad in expected.items():
        reference = grad.float()
        error = (actual[name].float() - reference).norm() / reference.norm().clamp_min(1e-30)
        assert error <= 2.0**-7, f"{name}: normwise {error.item():.3e} > 2^-7"


@gpu
def test_dsv4_sampled_depths_equal_the_eager_reference(hybrid, deterministic_algorithms):
    """S3: R_max 4 with the 'sampled' R4-uniform criterion: per-sample losses bitwise against
    eager at the realized depths."""
    with _hadamard_fallback():
        model = _dsv4_model(
            loop_iterations=4,
            loop_exit_criterion="sampled",
            loop_depth_probs=[0.25, 0.25, 0.25, 0.25],
            loop_depth_seed=1234,
        )
        batches = _batches(4, seq=256, dev=torch.device("cuda"))
        program = LoopProgram(model)
        executor = LoopRSBFExecutor(program, pg_collection=SimpleNamespace(dp=None))
        model.zero_grad(set_to_none=True)
        set_moe_metrics_tracker(MoEMetricsTracker())
        reports = executor.run_step(
            _next_batch,
            iter(_microbatches(model, hybrid, batches)),
            len(batches),
            model=model,
            iteration=3,
        )
        depths = [executor.stats.depths[SampleKey(i, 3)] for i in range(len(batches))]
        set_moe_metrics_tracker(MoEMetricsTracker())
        expected_losses, _ = _eager(model, hybrid, batches, depths)
    _assert_losses_equal(_lm_losses(reports), expected_losses)


@gpu
def test_dense_step_has_no_hidden_host_syncs(hybrid):
    """E7 (GPU): a dense step with the real loss function and validation (NaN / Inf / spiky
    checks on) runs under ``torch.cuda.set_sync_debug_mode('error')``, which only the
    executor's own end-of-step sync lowers; one host sync per step."""
    model = build_local_loop_model(DENSE, loop_iterations=R_MAX, **RSBF)
    hybrid.test_args.check_for_spiky_loss = True
    batches = _batches(len(DEPTHS), 2)
    executor = _executor(model, DEPTHS, check_ports=False)
    microbatches = iter(_microbatches(model, hybrid, batches))
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        executor.run_step(
            _next_batch, microbatches, len(batches), model=model, validate_loss=hybrid.validate_loss
        )
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert executor.stats.host_syncs == 1
