# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training integration of the loop RSBF schedule (``loop_schedule='rsbf'``).

- T1: ``pretrain_hybrid.forward_step`` carries the RSBF batch function and the loss validator;
  the batch function returns a ``LoopMicrobatch`` whose loss function does not validate, and
  rejects what the schedule cannot run (packed THD input, cached-logit and ModelOpt losses).
- T2: the schedule accepts exactly the keywords ``train_step`` passes to
  ``forward_backward_func`` (read from ``training.py``) and those of
  ``forward_backward_no_pipelining``; ``forward_backward_loop_rsbf(iteration=...)`` sets the
  iteration.
- T3: the ``training.py`` routing helper picks the module-level schedule for
  ``loop_schedule='rsbf'`` and rejects the wrappers, sharded data parallelism and the
  debug-save hooks; the executor rejects a sharded DP model itself; ``validate_args`` checks
  ``loop_rsbf_waves`` against the microbatch count; ``train()`` sets the iteration before
  every ``train_step``.
- T4: one step through ``pretrain_hybrid``'s batch function and the schedule: every report
  has the same keys; ``train_step``'s reduction (a copy) and the loop FLOP accounting give the
  executor's work-record totals, which ``num_floating_point_operations`` returns; the FLOP
  keys ``train()`` reads (from its source) are the executor's; the validator sees the mean
  per-microbatch loss sum.
- T5: ``loss_func(..., validate=True)`` is bitwise and call-for-call the function before the
  ``validate`` keyword; ``validate=False`` runs no check.
"""

import ast
import inspect
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.data_parallel_base import _BaseDataParallel
from megatron.core.pipeline_parallel import schedules
from megatron.core.transformer.loop import rsbf_executor
from megatron.core.transformer.loop.guards import validate_rsbf_data_parallel
from megatron.core.transformer.loop.program import PrescribedDepthCriterion
from megatron.core.transformer.loop.rsbf_executor import (
    RSBF_BATCH_FUNC_ATTR,
    RSBF_LOSS_VALIDATOR_ATTR,
    LoopMicrobatch,
    LoopRSBFSchedule,
    forward_backward_loop_rsbf,
    get_loop_rsbf_schedule,
)
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    CUDA,
    SEQ,
    VOCAB,
    build_local_loop_model,
    device,
    model_parallel_once,
)

DENSE = "*-*-*-*-"  # prelude '*-', core '*-*-', coda '*-'
RSBF = dict(loop_schedule="rsbf", calculate_per_token_loss=True, loop_iterations=4)
DEPTHS = (1, 3, 2, 4)
LOOP_FLOP_KEYS = tuple(
    f"loop {phase} flops per token" for phase in ("forward", "backward", "recompute")
)


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


class _Timer:
    def __init__(self, log):
        self.log = log

    def start(self, barrier=False):
        self.log.append("start")

    def stop(self):
        self.log.append("stop")


class _SpyRerunStateMachine:
    """Records every ``validate_result`` call (result, rejection, message, tolerance, fatal)."""

    def __init__(self):
        self.calls = []

    def validate_result(self, result, rejection_func, message, tolerance, fatal):
        self.calls.append((result, rejection_func, message, tolerance, fatal))

    @staticmethod
    def is_unexpectedly_large(result, threshold, context):
        return False


class _CPBatch:
    """What ``pretrain_hybrid.get_batch`` returns (a ``ContextParallelBatch`` at CP = 1)."""

    def __init__(self, batch, packed_seq_params=None):
        self.batch = batch
        self.packed_seq_params = packed_seq_params

    def get_batch(self):
        return self.batch

    def get_packed_seq_params(self):
        return self.packed_seq_params


@pytest.fixture
def hybrid(monkeypatch):
    """``pretrain_hybrid`` with stub training args, timers, a spy rerun state machine and a
    ``get_batch`` that returns the next dict of the data iterator as a CP = 1 batch."""
    import pretrain_hybrid

    args = SimpleNamespace(
        logits_load_dir=None,
        check_for_nan_in_loss_and_grad=True,
        check_for_spiky_loss=True,
        modelopt_enabled=False,
    )
    timer_log = []
    machine = _SpyRerunStateMachine()
    monkeypatch.setattr(pretrain_hybrid, "get_args", lambda: args)
    monkeypatch.setattr(pretrain_hybrid, "get_timers", lambda: lambda *a, **k: _Timer(timer_log))
    monkeypatch.setattr(pretrain_hybrid, "get_rerun_state_machine", lambda: machine)
    monkeypatch.setattr(pretrain_hybrid, "has_nvidia_modelopt", False, raising=False)
    monkeypatch.setattr(
        pretrain_hybrid, "get_batch", lambda data_iterator, vp_stage=None: next(data_iterator)
    )
    monkeypatch.setattr(pretrain_hybrid, "test_args", args, raising=False)
    monkeypatch.setattr(pretrain_hybrid, "test_machine", machine, raising=False)
    monkeypatch.setattr(pretrain_hybrid, "test_timer_log", timer_log, raising=False)
    return pretrain_hybrid


def _rows(samples, mbs, *, seed=11):
    """CP = 1 batches of ``mbs`` rows (no attention mask, as the SBHD loop configs)."""
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, VOCAB, (samples, SEQ), generator=generator)
    labels = torch.randint(0, VOCAB, (samples, SEQ), generator=generator)
    loss_mask = (torch.rand(samples, SEQ, generator=generator) > 0.2).float()
    batches = []
    for start in range(0, samples, mbs):
        rows = slice(start, start + mbs)
        batches.append(
            _CPBatch(
                {
                    "tokens": tokens[rows].to(device()),
                    "labels": labels[rows].to(device()),
                    "loss_mask": loss_mask[rows].to(device()),
                    "position_ids": torch.arange(SEQ).repeat(mbs, 1).to(device()),
                    "attention_mask": None,
                    "padding_mask": None,
                    "cu_seqlens": None,
                }
            )
        )
    return batches


# ---------------------------------------------------------------------------
# T1: the batch function
# ---------------------------------------------------------------------------


def test_forward_step_carries_the_rsbf_hooks(hybrid):
    assert getattr(hybrid.forward_step, RSBF_BATCH_FUNC_ATTR) is hybrid.loop_rsbf_batch
    assert getattr(hybrid.forward_step, RSBF_LOSS_VALIDATOR_ATTR) is hybrid.validate_loss


def test_batch_function_returns_a_non_validating_microbatch(hybrid):
    model = build_local_loop_model(DENSE, **RSBF)
    (cp_batch,) = _rows(2, 2)
    microbatch = hybrid.loop_rsbf_batch(iter([cp_batch]), model)
    assert isinstance(microbatch, LoopMicrobatch)
    for name in ("tokens", "position_ids", "labels", "loss_mask"):
        assert getattr(microbatch, name) is cp_batch.batch[name]
    assert microbatch.attention_mask is None and microbatch.padding_mask is None
    assert hybrid.test_timer_log == ["start", "stop"]
    # The loss function: loss_func(loss_mask, per-token loss, model=model, validate=False).
    assert microbatch.loss_func.func is hybrid.loss_func
    assert microbatch.loss_func.keywords == dict(model=model, validate=False)
    per_token = torch.rand(1, SEQ, device=device())
    loss, num_tokens, report = microbatch.loss_func(microbatch.loss_mask[:1], per_token)
    assert hybrid.test_machine.calls == []
    expected = hybrid.loss_func(microbatch.loss_mask[:1], per_token, model=model)
    assert torch.equal(loss, expected[0]) and torch.equal(num_tokens, expected[1])
    assert report.keys() == expected[2].keys() == {"lm loss"}
    assert len(hybrid.test_machine.calls) == 3  # the validating call checks NaN, Inf, spiky


@pytest.mark.parametrize("what", ["packed", "cu_seqlens", "cached_logits", "modelopt"])
def test_batch_function_rejects_what_rsbf_cannot_run(hybrid, monkeypatch, what):
    model = build_local_loop_model(DENSE, **RSBF)
    (cp_batch,) = _rows(1, 1)
    message = "packed"
    if what == "packed":
        cp_batch.packed_seq_params = object()
    elif what == "cu_seqlens":
        cp_batch.batch["cu_seqlens"] = torch.tensor([0, SEQ])
    elif what == "cached_logits":
        hybrid.test_args.logits_load_dir = "/logits"
        message = "logits-load-dir"
    else:
        monkeypatch.setattr(hybrid, "has_nvidia_modelopt", True, raising=False)
        hybrid.test_args.modelopt_enabled = True
        message = "ModelOpt"
    with pytest.raises(ValueError, match=message):
        hybrid.loop_rsbf_batch(iter([cp_batch]), model)


# ---------------------------------------------------------------------------
# T2: the schedule's keywords
# ---------------------------------------------------------------------------


def _train_step_schedule_keywords():
    """The keywords ``train_step`` passes to ``forward_backward_func`` (from its source)."""
    from megatron.training import training

    tree = ast.parse(inspect.getsource(training))
    (train_step,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train_step"
    ]
    calls = [
        node
        for node in ast.walk(train_step)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "forward_backward_func"
    ]
    assert len(calls) == 1
    (call,) = calls
    assert not call.args and all(keyword.arg for keyword in call.keywords)
    return {keyword.arg for keyword in call.keywords}


def test_schedule_accepts_the_train_step_keywords(hybrid, monkeypatch):
    model = build_local_loop_model(DENSE, **RSBF)
    calls = []

    class _StubExecutor:
        def run_step(self, batch_func, data_iterator, num_microbatches, **kwargs):
            calls.append((batch_func, data_iterator, num_microbatches, kwargs))
            return ["report"]

    stub = _StubExecutor()
    monkeypatch.setattr(LoopRSBFSchedule, "executor_for", lambda self, *args, **kwargs: stub)
    iterator = iter([])
    kwargs = dict(
        forward_step_func=hybrid.forward_step,
        data_iterator=[iterator],
        model=[model],
        num_microbatches=4,
        seq_length=SEQ,
        micro_batch_size=2,
        decoder_seq_length=None,
        forward_only=False,
        adjust_tensor_shapes_fn=None,
        force_all_reduce=False,
        p2p_communicator=None,
        pg_collection=SimpleNamespace(dp=None),
    )
    assert kwargs.keys() == _train_step_schedule_keywords()
    accepted = inspect.signature(LoopRSBFSchedule.__call__).parameters
    standard = inspect.signature(schedules.forward_backward_no_pipelining).parameters
    assert set(standard) <= set(accepted), set(standard) - set(accepted)

    assert forward_backward_loop_rsbf(iteration=7, **kwargs) == ["report"]
    assert get_loop_rsbf_schedule().iteration == 7
    ((batch_func, data_iterator, num_microbatches, run_kwargs),) = calls
    assert batch_func is hybrid.loop_rsbf_batch and data_iterator is iterator
    assert num_microbatches == 4
    assert run_kwargs == dict(
        model=model,
        iteration=7,
        micro_batch_size=2,
        force_all_reduce=False,
        validate_loss=hybrid.validate_loss,
    )
    # Without the keyword the entry keeps the iteration last set (no TypeError, no reset).
    get_loop_rsbf_schedule().set_iteration(9)
    forward_backward_loop_rsbf(**kwargs)
    assert get_loop_rsbf_schedule().iteration == 9
    assert calls[-1][3]["iteration"] == 9
    get_loop_rsbf_schedule().set_iteration(0)


# ---------------------------------------------------------------------------
# T3: training.py routing
# ---------------------------------------------------------------------------


def _args(**overrides):
    fields = dict(
        cuda_graph_impl="none",
        moe_expert_rank_capacity_factor=None,
        use_megatron_fsdp=False,
        use_torch_fsdp2=False,
        save_activations_interval=None,
        save_dgrads_interval=None,
        save_tokens_per_expert_interval=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_training_routes_rsbf_configs_to_the_module_schedule():
    from megatron.training.training import get_loop_rsbf_forward_backward_func

    rsbf = SimpleNamespace(loop_schedule="rsbf")
    assert get_loop_rsbf_forward_backward_func(rsbf, _args()) is get_loop_rsbf_schedule()
    assert get_loop_rsbf_forward_backward_func(rsbf, _args()) is rsbf_executor._DEFAULT_SCHEDULE
    assert (
        get_loop_rsbf_forward_backward_func(SimpleNamespace(loop_schedule="standard"), _args())
        is None
    )
    assert get_loop_rsbf_forward_backward_func(SimpleNamespace(), _args()) is None
    with pytest.raises(ValueError, match="cuda_graph_impl=full_iteration"):
        get_loop_rsbf_forward_backward_func(rsbf, _args(cuda_graph_impl="full_iteration"))
    with pytest.raises(ValueError, match="moe_expert_rank_capacity_factor"):
        get_loop_rsbf_forward_backward_func(rsbf, _args(moe_expert_rank_capacity_factor=1.0))
    # Sharded data parallelism: per-module collectives, a different count per DP rank.
    for flag in ("use_megatron_fsdp", "use_torch_fsdp2"):
        with pytest.raises(ValueError, match=flag):
            get_loop_rsbf_forward_backward_func(rsbf, _args(**{flag: True}))
    # Debug-save hooks would also record the executor's replays.
    for interval in (
        "save_activations_interval",
        "save_dgrads_interval",
        "save_tokens_per_expert_interval",
    ):
        with pytest.raises(ValueError, match=interval):
            get_loop_rsbf_forward_backward_func(rsbf, _args(**{interval: 10}))
    # Every rejection is reported at once; the standard schedule is not affected.
    with pytest.raises(ValueError, match="use_megatron_fsdp, save_dgrads_interval"):
        get_loop_rsbf_forward_backward_func(
            rsbf, _args(use_megatron_fsdp=True, save_dgrads_interval=1)
        )
    standard = SimpleNamespace(loop_schedule="standard")
    assert get_loop_rsbf_forward_backward_func(standard, _args(use_megatron_fsdp=True)) is None


class _FakeShardedDP(_BaseDataParallel):
    """A non-DDP data-parallel wrapper (Megatron FSDP and torch FSDP2 subclass the base)."""


class _DDPWithMegatronFSDP(DistributedDataParallel):
    """``DistributedDataParallel`` whose config asks for Megatron FSDP (no process groups)."""

    def __init__(self, config, module):  # pylint: disable=super-init-not-called
        _BaseDataParallel.__init__(self, config, module)
        self.ddp_config = SimpleNamespace(use_megatron_fsdp=True)


class _PlainDDP(_DDPWithMegatronFSDP):
    def __init__(self, config, module):
        super().__init__(config, module)
        self.ddp_config = SimpleNamespace(use_megatron_fsdp=False)


def test_executor_rejects_sharded_data_parallel_models(hybrid):
    """The model-side check (callers other than train()): a sharded DP wrapper anywhere in the
    ``.module`` chain is rejected before the step reads data; DDP and a bare model pass."""
    model = build_local_loop_model(DENSE, **RSBF)
    validate_rsbf_data_parallel(model)
    validate_rsbf_data_parallel(_PlainDDP(model.config, model))
    for wrapper in (_FakeShardedDP, _DDPWithMegatronFSDP):
        with pytest.raises(ValueError, match="sharded data parallelism"):
            validate_rsbf_data_parallel(wrapper(model.config, model))
    # Nested under another wrapper (``.module`` chains, e.g. Float16Module).
    outer = _PlainDDP(model.config, _FakeShardedDP(model.config, model))
    with pytest.raises(ValueError, match="sharded data parallelism"):
        validate_rsbf_data_parallel(outer)

    # Through the schedule: run_step raises before the batch function is called.
    def no_batch(*_):
        raise AssertionError("the step must not read data")

    forward_step = SimpleNamespace(**{RSBF_BATCH_FUNC_ATTR: no_batch})
    with pytest.raises(ValueError, match="sharded data parallelism"):
        LoopRSBFSchedule()(
            forward_step_func=forward_step,
            data_iterator=iter([]),
            model=[_FakeShardedDP(model.config, model)],
            num_microbatches=1,
            seq_length=SEQ,
            micro_batch_size=1,
            pg_collection=SimpleNamespace(dp=None),
        )


def _waves_args(**overrides):
    fields = dict(
        loop_schedule="rsbf",
        loop_rsbf_waves=2,
        step_batch_size_schedule=None,
        decrease_batch_size_if_needed=False,
        micro_batch_size=1,
        data_parallel_size=2,
        gtp_weight_remat_size=1,
        global_batch_size=16,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_validate_args_checks_the_waves_against_the_microbatch_count():
    from megatron.training.arguments import _validate_loop_rsbf_waves

    _validate_loop_rsbf_waves(_waves_args())  # 16 / (1 x 2) = 8 microbatches, 2 waves
    _validate_loop_rsbf_waves(_waves_args(loop_rsbf_waves=8))
    _validate_loop_rsbf_waves(_waves_args(loop_schedule="standard", loop_rsbf_waves=3))
    _validate_loop_rsbf_waves(_waves_args(loop_rsbf_waves=1, step_batch_size_schedule="0:8"))
    with pytest.raises(ValueError, match=r"loop-rsbf-waves \(3\) must divide .* \(8 = "):
        _validate_loop_rsbf_waves(_waves_args(loop_rsbf_waves=3))
    with pytest.raises(ValueError, match=r"\(4 = "):  # the GTP-remat axis counts
        _validate_loop_rsbf_waves(_waves_args(loop_rsbf_waves=8, gtp_weight_remat_size=2))
    for schedule in (
        dict(step_batch_size_schedule="0:8 1K:16"),
        dict(decrease_batch_size_if_needed=True),
    ):
        with pytest.raises(ValueError, match="step-batch-size-schedule"):
            _validate_loop_rsbf_waves(_waves_args(**schedule))


def test_train_uses_the_rsbf_schedule_and_sets_the_iteration():
    """In ``train()``: the helper's schedule replaces ``forward_backward_func`` before the
    wrappers, and its iteration is set right before ``train_step`` with the loop's iteration."""
    from megatron.training import training

    tree = ast.parse(inspect.getsource(training.train))
    lines = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            text = ast.unparse(node.func)
            lines.setdefault(text, []).append(node.lineno)
    (helper,) = lines["get_loop_rsbf_forward_backward_func"]
    (wrapper,) = lines["FullCudaGraphWrapper"]
    (set_iteration,) = lines["loop_rsbf_schedule.set_iteration"]
    (step,) = lines["train_step"]
    assert helper < wrapper and set_iteration < step
    set_iteration_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "loop_rsbf_schedule.set_iteration"
    ]
    assert [ast.unparse(arg) for arg in set_iteration_calls[0].args] == ["iteration"]
    step_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "train_step"
    ]
    assert {keyword.arg: ast.unparse(keyword.value) for keyword in step_calls[0].keywords}[
        "iteration"
    ] == "iteration"
    # evaluate() keeps the standard schedule.
    assert "get_loop_rsbf_forward_backward_func" not in inspect.getsource(training.evaluate)


# ---------------------------------------------------------------------------
# T4: reports, train_step's reduction and the loop FLOP accounting
# ---------------------------------------------------------------------------


def _train_step_reduction(losses_reduced):
    """Copy of ``train_step``'s loss reduction (non-SFT branch), DP group = the world."""
    loss_reduced = {}
    for key in losses_reduced[0].keys():
        val = [x[key].view(-1) for x in losses_reduced]
        if val[0].numel() == 2:
            val = torch.vstack(val).sum(dim=0)
            dist.all_reduce(val)
            loss_reduced[key] = val[0] / val[1]
        elif val[0].numel() == 1:
            loss_reduced[key] = torch.cat(val).mean()
        else:
            raise ValueError(f"Invalid value shape: {val[0].shape} for key {key}")
    return loss_reduced


def _training_loop_flop_keys():
    """The ``loss_dict`` keys ``train()``'s loop FLOP accounting reads (from its source): the
    f-string of ``loss_dict.get(f"...{phase}...")`` over the phase tuple of its generator."""
    from megatron.training import training

    tree = ast.parse(inspect.getsource(training.train))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.GeneratorExp) or len(node.generators) != 1:
            continue
        gets = [
            call
            for call in ast.walk(node.elt)
            if isinstance(call, ast.Call)
            and ast.unparse(call.func) == "loss_dict.get"
            and isinstance(call.args[0], ast.JoinedStr)
        ]
        if not gets:
            continue
        (get,) = gets
        generator = node.generators[0]
        phases = ast.literal_eval(generator.iter)
        variable = generator.target.id
        parts = []
        for value in get.args[0].values:
            if isinstance(value, ast.Constant):
                parts.append(value.value)
            else:
                assert ast.unparse(value.value) == variable
                parts.append(None)
        found.append(
            {"".join(phase if part is None else part for part in parts) for phase in phases}
        )
    assert len(found) == 1, found
    return found[0]


def test_reports_reduce_to_the_work_record_totals(hybrid):
    """MBS 2, two microbatches, prescribed depths [1, 3, 2, 4]: the per-sample reports reduce
    (as train_step does) to the mean depth, the per-token loss and per-token FLOPs whose
    product with batch_size * seq_length (the loop FLOP accounting of train()) is the sum of
    the executor's work records; ``num_floating_point_operations`` returns that total."""
    from megatron.training.training import num_floating_point_operations

    model = build_local_loop_model(DENSE, **RSBF)
    mbs, samples = 2, len(DEPTHS)
    batches = _rows(samples, mbs)
    schedule = LoopRSBFSchedule()
    reports = schedule(
        forward_step_func=hybrid.forward_step,
        data_iterator=[iter(batches)],
        model=[model],
        num_microbatches=samples // mbs,
        seq_length=SEQ,
        micro_batch_size=mbs,
        pg_collection=SimpleNamespace(dp=None),
        exit_criterion=PrescribedDepthCriterion(dict(enumerate(DEPTHS))),
    )
    executor = schedule.cached_executor(model)
    assert len(reports) == samples
    keys = reports[0].keys()
    assert all(report.keys() == keys for report in reports)
    assert {"lm loss", "loop depth", "loop active token layers", *LOOP_FLOP_KEYS} <= set(keys)
    # Validation ran once for the step (NaN, Inf, spiky), on a host tensor.
    messages = [call[2] for call in hybrid.test_machine.calls]
    assert messages == [
        "found NaN in local forward loss calculation",
        "found Inf in local forward loss calculation",
        "Spiky loss",
    ]
    assert all(call[0].device.type == "cpu" for call in hybrid.test_machine.calls)

    reduced = _train_step_reduction(reports)
    assert reduced["loop depth"].item() == pytest.approx(sum(DEPTHS) / samples)
    records = executor.stats.loss_records
    total_loss = sum(record.loss.item() for record in records)
    total_tokens = sum(record.num_tokens.item() for record in records)
    # The validated value is the mean per-microbatch loss sum (the spiky-loss history stays
    # comparable across microbatch counts).
    num_microbatches = samples // mbs
    for result, *_ in hybrid.test_machine.calls:
        assert result.item() == pytest.approx(total_loss / num_microbatches, rel=1e-6)
    # training.py reads exactly the executor's FLOP report keys.
    assert _training_loop_flop_keys() == {key for key in keys if "flops" in key}
    assert reduced["lm loss"].item() == pytest.approx(total_loss / total_tokens, rel=1e-6)

    # train()'s loop FLOP accounting; batch_size = the global samples per step (every rank of
    # a multi-GPU run executes this same step, so the global work is world x the local work).
    world = dist.get_world_size()
    args = SimpleNamespace(loop_architecture="huginn", seq_length=SEQ)
    loss_dict = dict(reduced)
    batch_size = samples * world
    args._loop_flops_in_batch = (
        batch_size
        * args.seq_length
        * sum(
            float(loss_dict.get(f"loop {phase} flops per token", 0.0))
            for phase in ("forward", "backward", "recompute")
        )
    )
    work = executor.stats.work
    expected = world * sum(record.forward + record.backward + record.recompute for record in work)
    assert expected > 0
    assert args._loop_flops_in_batch == pytest.approx(expected, rel=1e-6)
    assert num_floating_point_operations(args, batch_size) == args._loop_flops_in_batch
    for phase in ("forward", "backward", "recompute"):
        per_phase = world * sum(getattr(record, phase) for record in work)
        logged = float(loss_dict[f"loop {phase} flops per token"]) * batch_size * SEQ
        assert logged == pytest.approx(per_phase, rel=1e-6)


# ---------------------------------------------------------------------------
# T5: loss_func(validate=True) is the function before the keyword
# ---------------------------------------------------------------------------


def _loss_func_before_validate_keyword(
    module, loss_mask, output_tensor, model=None, loop_losses=None, loop_metrics=None
):
    """``pretrain_hybrid.loss_func`` before the ``validate`` keyword (the default branch:
    no cached logits, no ModelOpt), with the module's ``get_args`` and rerun state machine."""
    args = module.get_args()
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    for name, auxiliary in (loop_losses or {}).items():
        loss = loss + num_tokens * auxiliary
        report[name] = torch.stack((auxiliary.detach() * num_tokens, num_tokens))
    for name, metric in (loop_metrics or {}).items():
        report[name] = torch.stack((metric.detach() * num_tokens, num_tokens))

    rerun_state_machine = module.get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=module.SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,
            fatal=False,
        )
    return loss, num_tokens, report


def _describe(calls):
    """Comparable form of the recorded validate_result calls."""
    described = []
    for result, rejection, message, tolerance, fatal in calls:
        if isinstance(rejection, partial):
            rejection = (rejection.func, tuple(sorted(rejection.keywords.items())))
        described.append((result.detach().clone(), rejection, message, tolerance, fatal))
    return described


@pytest.mark.parametrize("nan_check", [True, False])
@pytest.mark.parametrize("spiky_check", [True, False])
@pytest.mark.parametrize("loop_extras", [False, True])
def test_loss_func_validate_true_is_the_old_function(hybrid, nan_check, spiky_check, loop_extras):
    hybrid.test_args.check_for_nan_in_loss_and_grad = nan_check
    hybrid.test_args.check_for_spiky_loss = spiky_check
    generator = torch.Generator().manual_seed(3)
    loss_mask = (torch.rand(2, SEQ, generator=generator) > 0.3).float()
    output = torch.rand(2, SEQ, generator=generator)
    extras = {}
    if loop_extras:
        extras = dict(
            loop_losses={"loop router loss": torch.tensor(0.25)},
            loop_metrics={"loop depth": torch.tensor(2.5)},
        )
    machine = hybrid.test_machine
    new = hybrid.loss_func(loss_mask, output, **extras)
    new_calls = _describe(machine.calls)
    machine.calls.clear()
    old = _loss_func_before_validate_keyword(hybrid, loss_mask, output, **extras)
    old_calls = _describe(machine.calls)
    machine.calls.clear()
    assert torch.equal(new[0], old[0]) and torch.equal(new[1], old[1])
    assert new[2].keys() == old[2].keys()
    assert all(torch.equal(new[2][key], old[2][key]) for key in old[2])
    assert len(new_calls) == len(old_calls) == 2 * nan_check + spiky_check
    for got, want in zip(new_calls, old_calls):
        assert torch.equal(got[0], want[0]) and got[1:] == want[1:]
    # validate=False: the same values, no check.
    quiet = hybrid.loss_func(loss_mask, output, validate=False, **extras)
    assert machine.calls == []
    assert torch.equal(quiet[0], old[0]) and torch.equal(quiet[1], old[1])
