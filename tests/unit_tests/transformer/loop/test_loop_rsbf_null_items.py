# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lockstep MoE across the EUs of an RSBF domain (``loop/null_items.py``).

CPU cases (no process group beyond the module's one-rank world): the null runner (call order,
leaves, rows per form, backward only for the backward types, quantization contexts), the
plan's real-call checks (signatures, the per-item call log), the static-budget flags, the
adapter registry, discovery on a stub 'E' model (``build_lockstep_plan``), the ``MoELayer``
adapter's null forms and config signatures (dispatcher stubs), and the executor's refusal of a
null without a layer context (unequal home counts). The spawned multi-EU stub cases are M8 in
``test_loop_rsbf_multi_eu.py``.

GPU cases (``WORLD_SIZE >= 2``, EP = WORLD_SIZE = W, model parallel initialized once per
module):

* N1: one TE MoE layer (all-to-all): rank 0 real and the other ranks null, the reverse, and
  every rank real; no hang, rank 0's output equals the all-real one, and the expert gradients
  of the two partial arms add up to the all-real ones (the null ranks host their experts).
* G2 (with N2 / N4): the DSv4-like ``'WECEHE/WE'`` model (hash E in the prelude, MTP E) at
  EP = W with the all-to-all dispatcher, prescribed depths with idle EUs: per-sample losses
  against a one-EU EP = 1 reference with the gathered expert weights (bitwise, or at the bf16
  loss rule when the expert GEMMs are not row-invariant; self-determinism bitwise), gradients
  by the bf16 normwise rule (every arm adds each backward call's gradient into fp32, as DDP's
  ``main_grad`` does: ``_fp32_grad_accumulation``), expert-bias counts summed over the ranks
  exactly the reference's;
  ``check_lockstep`` (every real call's signature equals the null's) and ``check_plan`` on; the
  MoE collectives (``all_to_all_single`` / ``all_gather_into_tensor`` on the EP group, forward
  and backward) logged per rank are identical on every rank. With ``moe_router_dtype`` fp32 and
  None (bf16 probabilities).
* N4 (FP8 current scaling, H100): a null enters the same inner quantization contexts (layer
  config, layer number) as the real items of its type, in order, and the MTP depth's wrap
  (``MultiTokenPredictionLayer.get_inner_quantization_context``) around its MTP E.
* G3 / N3 (HybridEP static budget; needs HybridEP, the TE op fuser (SwiGLU experts) and a
  Blackwell GPU, else skipped -- run on lyris GB200): the DSv4-like model under ``moe_expert_rank_capacity_factor`` never sets
  ``over_budget`` (raise mode) and matches the all-to-all model with the same weights (losses
  at the loss rule, gradients by the normwise rule); every HybridEP dispatch is logged: a null
  sends no route, a real call sends ``topk`` routes per non-padded row, every rank's experts
  receive exactly the routes the EP group sent them (up to the GEMM alignment padding), and
  every call has the same static budget. N3 repeats the step with a quarter of every odd
  sample's tail as padding: the padded rows send nothing, the budgets equal the unpadded
  run's, ``over_budget`` stays unset. A 0.05 factor overflows and every rank raises with the
  same rank list.
* G4: waves 2 and 4 (n = 1 per wave and EU) against one pass on the G2 model (losses at the
  loss rule, gradients by the normwise rule; each arm self-deterministic).
* G6: the move gate on the G2 workload (sampled R4-uniform depths, 8 per EU, 8 steps): the
  step-aggregate moves <= 5% of the active sample-rounds; bulk / tail maxima printed.
"""

import contextlib
import copy
import os
import re
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.loop import null_items
from megatron.core.transformer.loop.null_items import (
    FORM_EMPTY,
    FORM_NOMINAL,
    LOCKSTEP,
    LockstepAdapter,
    LockstepEntry,
    LockstepPlan,
    LockstepSignature,
    build_lockstep_plan,
    register_lockstep_module,
    run_null,
)
from megatron.core.transformer.loop.program import (
    LoopProgram,
    PrescribedDepthCriterion,
    SampledDepthCriterion,
)
from megatron.core.transformer.loop.rsbf_executor import (
    OVER_BUDGET,
    LoopMicrobatch,
    LoopRSBFExecutor,
    get_loop_rsbf_schedule,
)
from megatron.core.transformer.loop.rsbf_plan import SubRoundType
from tests.unit_tests.transformer.loop import hybrid_loop_utils
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    CUDA,
    build_local_loop_model,
    model_parallel_once,
)
from tests.unit_tests.transformer.loop.multi_eu_utils import (
    CollectiveOpLogger,
    RecordingLockstepMLP,
    stub_moe_layer_spec,
)

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
EP_GPU = CUDA and WORLD >= 2
gpu_ep = pytest.mark.skipif(not EP_GPU, reason="lockstep MoE needs CUDA and at least 2 ranks")
cpu_only = pytest.mark.skipif(CUDA, reason="CPU unit cases (the GPU cases cover CUDA hosts)")

cpu_mtp = hybrid_loop_utils.cpu_mtp

RSBF = dict(loop_schedule="rsbf", calculate_per_token_loss=True)


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    """Once per module: EP = WORLD_SIZE on GPU (the lockstep cases), a one-rank world on CPU."""
    with model_parallel_once(expert_model_parallel_size=WORLD if EP_GPU else 1):
        try:
            yield
        finally:
            get_loop_rsbf_schedule().reset()


# ---------------------------------------------------------------------------------------------
# CPU: the null runner
# ---------------------------------------------------------------------------------------------


class _Recorder(torch.nn.Module):
    """A lockstep-like module whose null call scales by 2 and logs the calls and contexts."""

    def __init__(self, name, log):
        super().__init__()
        self.name = name
        self.log = log
        self.flag = torch.zeros(1, dtype=torch.bool)

    def forward(self, hidden):
        """A real call (the same op as the null)."""
        return _Scale.apply(hidden, self)


class _Scale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, module):  # pylint: disable=arguments-differ
        ctx.module = module
        module.log.append(("fwd", module.name, tuple(hidden.shape), hidden.dtype))
        return hidden * 2

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=arguments-differ
        ctx.module.log.append(("bwd", ctx.module.name))
        return grad * 2, None


class _RecorderAdapter(LockstepAdapter):
    def __init__(self, form=FORM_EMPTY, spans=True):
        self._form = form
        self.spans = spans

    def groups(self, module):
        return [module] if self.spans else []

    def form(self, module):
        return self._form

    def config_signature(self, module):
        return LockstepSignature(hidden_dtype=torch.float64, hidden_requires_grad=True)

    def observe(self, module, on_call, on_signature):
        def before(mod, args, kwargs):
            on_call(mod)
            on_signature(mod, LockstepSignature(args[0].dtype, args[0].requires_grad))

        return [module.register_forward_pre_hook(before, with_kwargs=True)]

    def null_forward(self, entry, hidden, signature):
        assert hidden.dtype == signature.hidden_dtype
        return _Scale.apply(hidden, entry.module)

    def over_budget(self, module):
        return module.flag

    def reset_over_budget(self, module):
        module.flag.fill_(False)


def _plan(names, form=FORM_EMPTY, log=None, wrap=None):
    """A hand-made plan: the same modules for every sub-round type."""
    log = [] if log is None else log
    adapter = _RecorderAdapter(form)
    modules = [_Recorder(name, log) for name in names]
    entries = tuple(
        LockstepEntry(
            module=module,
            adapter=adapter,
            layer_config=SimpleNamespace(hidden_size=8, name=f"config-{module.name}"),
            layer_number=index,
            form=form,
            name=module.name,
            wrap=wrap if index == len(modules) - 1 else None,
        )
        for index, module in enumerate(modules)
    )
    plan = LockstepPlan(
        by_type={t: entries for t in SubRoundType},
        budget=tuple((module.name, module, adapter) for module in modules),
    )
    for entry in entries:
        plan.signature[entry.module] = adapter.config_signature(entry.module)
        plan._names[id(entry.module)] = entry.name  # pylint: disable=protected-access
    return plan, modules, log


class _Contexts:
    """Records the quantization contexts a null enters."""

    def __init__(self):
        self.log = []

    @contextlib.contextmanager
    def _context(self, label):
        self.log.append(("enter", label))
        yield
        self.log.append(("exit", label))

    def outer(self):
        return self._context("outer")

    def inner(self, layer_config, layer_number):
        return self._context(("inner", layer_config.name, layer_number))

    def wrap(self):
        return self._context("mtp")


@cpu_only
@pytest.mark.parametrize("kind_type", list(SubRoundType))
def test_run_null_makes_the_type_calls_chained(kind_type):
    """Every MoE call of the type, in order, on a chained input (first a zero leaf of the
    signature's dtype with grad, ``rows`` 0 for the empty form); backward types back-propagate
    through the chain in reverse, forward types do not; each call inside the outer context and
    its inner (layer config, layer number) context, the MTP entry also inside its wrap."""
    contexts = _Contexts()
    plan, _, log = _plan(["a", "b", "mtp"], wrap=contexts.wrap)
    calls = run_null(
        plan,
        kind_type,
        outer_context=contexts.outer(),
        inner_context=contexts.inner,
        sequence=16,
        device=torch.device("cpu"),
    )
    assert calls == 3
    forward = [("fwd", name, (0, 1, 8), torch.float64) for name in ("a", "b", "mtp")]
    if kind_type in (SubRoundType.CORE_B, SubRoundType.CODA, SubRoundType.PRELUDE_B):
        assert log == forward + [("bwd", "mtp"), ("bwd", "b"), ("bwd", "a")]
    else:
        assert log == forward
    assert contexts.log == [
        ("enter", "outer"),
        ("enter", ("inner", "config-a", 0)),
        ("exit", ("inner", "config-a", 0)),
        ("enter", ("inner", "config-b", 1)),
        ("exit", ("inner", "config-b", 1)),
        ("enter", "mtp"),
        ("enter", ("inner", "config-mtp", 2)),
        ("exit", ("inner", "config-mtp", 2)),
        ("exit", "mtp"),
        ("exit", "outer"),
    ]


@cpu_only
def test_run_null_nominal_rows_and_empty_types():
    plan, _, log = _plan(["a"], form=FORM_NOMINAL)
    run_null(
        plan,
        SubRoundType.CORE_F,
        outer_context=contextlib.nullcontext(),
        inner_context=lambda config, number: contextlib.nullcontext(),
        sequence=16,
        device=torch.device("cpu"),
    )
    assert log == [("fwd", "a", (16, 1, 8), torch.float64)]
    empty = LockstepPlan(by_type={t: () for t in SubRoundType})
    assert not empty.active
    assert (
        run_null(
            empty,
            SubRoundType.CODA,
            outer_context=contextlib.nullcontext(),
            inner_context=lambda config, number: contextlib.nullcontext(),
            sequence=16,
            device=torch.device("cpu"),
        )
        == 0
    )


# ---------------------------------------------------------------------------------------------
# CPU: real-call checks, budget flags, registry
# ---------------------------------------------------------------------------------------------


@cpu_only
def test_real_calls_confirm_the_signature_and_the_call_list():
    plan, (a, b, mtp), _ = _plan(["a", "b", "mtp"])
    x = torch.zeros(2, 1, 8, dtype=torch.float64, requires_grad=True)
    with plan.observing(record_calls=True):
        plan.begin_item()
        for module in (a, b, mtp):
            module(x)
        plan.end_item(SubRoundType.CODA, "coda(0)")
        # One call missing: the plan lists another sequence than the real item made.
        plan.begin_item()
        a(x)
        with pytest.raises(RuntimeError, match=f"{LOCKSTEP}: the real coda item coda.1. called"):
            plan.end_item(SubRoundType.CODA, "coda(1)")
        # A real call with another signature than the null's (checked on every call here).
        with pytest.raises(RuntimeError, match=f"{LOCKSTEP}: a real call of a has the signature"):
            a(x.detach().float())
    assert all(plan.signature[m].observed for m in (a, b, mtp))
    # The hooks are removed after the step.
    a(x.detach().float())


@cpu_only
def test_first_real_call_confirms_once_without_record_calls():
    plan, (a, _, _), _ = _plan(["a", "b", "c"])
    x = torch.zeros(2, 1, 8, dtype=torch.float64, requires_grad=True)
    with plan.observing():
        a(x)
        a(x.detach().float())  # confirmed already: not compared again
    plan, (a, _, _), _ = _plan(["a", "b", "c"])
    with plan.observing(), pytest.raises(RuntimeError, match=LOCKSTEP):
        a(x.detach())  # requires_grad differs from the null's leaf


@cpu_only
def test_over_budget_flags_and_reset():
    plan, (a, b), _ = _plan(["a", "b"])
    names, flags = plan.over_budget_flags()
    assert names == ["a", "b"] and flags.tolist() == [0, 0] and flags.dtype == torch.int32
    b.flag.fill_(True)
    assert plan.over_budget_flags()[1].tolist() == [0, 1]
    plan.reset_over_budget()
    assert plan.over_budget_flags()[1].tolist() == [0, 0]
    del a
    assert LockstepPlan(by_type={}).over_budget_flags() == ([], None)


@cpu_only
def test_registry_prefers_the_latest_registration():
    first, second = _RecorderAdapter(), _RecorderAdapter()
    module = _Recorder("m", [])
    remove_first = register_lockstep_module(_Recorder, first)
    try:
        assert null_items._adapter_for(module) is first  # pylint: disable=protected-access
        remove_second = register_lockstep_module(_Recorder, second)
        assert null_items._adapter_for(module) is second  # pylint: disable=protected-access
        remove_second()
        assert null_items._adapter_for(module) is first  # pylint: disable=protected-access
    finally:
        remove_first()
    assert null_items._adapter_for(module) is None  # pylint: disable=protected-access
    assert null_items._adapter_for(torch.nn.Linear(2, 2)) is None  # pylint: disable=W0212


# ---------------------------------------------------------------------------------------------
# CPU: the MoELayer adapter (null forms, config signatures) on dispatcher stubs
# ---------------------------------------------------------------------------------------------


def _dispatchers():
    """Uninitialized dispatcher / manager instances (the adapter only checks their types)."""
    from megatron.core.transformer.moe import token_dispatcher as td

    def flex(manager):
        dispatcher = object.__new__(td.MoEFlexTokenDispatcher)
        dispatcher._comm_manager = object.__new__(manager)  # pylint: disable=protected-access
        return dispatcher

    return {
        "alltoall": object.__new__(td.MoEAlltoAllTokenDispatcher),
        "allgather": object.__new__(td.MoEAllGatherTokenDispatcher),
        "hybridep": flex(td._HybridEPManager),  # pylint: disable=protected-access
        "deepep": flex(td._DeepepManager),  # pylint: disable=protected-access
    }


def _moe_stub(dispatcher="alltoall", router=None, **config):
    fields = dict(
        moe_expert_capacity_factor=None,
        moe_token_dispatcher_type="flex" if dispatcher in ("hybridep", "deepep") else dispatcher,
        moe_flex_dispatcher_backend=dispatcher if dispatcher in ("hybridep", "deepep") else None,
        fp32_residual_connection=False,
        params_dtype=torch.bfloat16,
        moe_router_dtype=None,
    )
    fields.update(config)
    return SimpleNamespace(
        config=SimpleNamespace(**fields),
        token_dispatcher=_dispatchers()[dispatcher],
        router=SimpleNamespace(is_hash_layer=False) if router is None else router,
    )


@cpu_only
def test_moe_adapter_null_forms():
    """Dropless all-to-all: zero rows; flex + HybridEP: nominal rows; every other dispatcher,
    and capacity mode on either, has no null form."""
    adapter = null_items.MoELockstepAdapter()
    assert adapter.form(_moe_stub("alltoall")) == FORM_EMPTY
    assert adapter.form(_moe_stub("hybridep")) == FORM_NOMINAL
    for stub, name in (
        (_moe_stub("deepep"), "MoEFlexTokenDispatcher"),
        (_moe_stub("allgather"), "MoEAllGatherTokenDispatcher"),
        (_moe_stub("alltoall", moe_expert_capacity_factor=1.0), "MoEAlltoAllTokenDispatcher"),
        (_moe_stub("hybridep", moe_expert_capacity_factor=1.0), "MoEFlexTokenDispatcher"),
    ):
        with pytest.raises(ValueError, match=f"lockstep null items for the MoE dispatcher {name}"):
            adapter.form(stub)


def _router(hash_layer=False, dense=None, has_dense=True):
    router = SimpleNamespace(is_hash_layer=hash_layer)
    if has_dense:
        router._dense_route_indices_dtype = lambda: dense  # pylint: disable=protected-access
    return router


@cpu_only
@pytest.mark.parametrize(
    "config,router,expected",
    [
        # hidden: params dtype; probs follow the logits (the hidden dtype) without a router dtype.
        ({}, _router(), (torch.bfloat16, torch.bfloat16, torch.bool)),
        (dict(moe_router_dtype="fp32"), _router(), (torch.bfloat16, torch.float32, torch.bool)),
        (dict(moe_router_dtype="fp64"), _router(), (torch.bfloat16, torch.float64, torch.bool)),
        # fp32 residual stream: fp32 MoE input, and fp32 logits without a router dtype.
        (dict(fp32_residual_connection=True), _router(), (torch.float32,) * 2 + (torch.bool,)),
        (
            dict(fp32_residual_connection=True, moe_router_dtype="fp64"),
            _router(),
            (torch.float32, torch.float64, torch.bool),
        ),
        # Dense route indices (flex): the router's dtype; a hash layer keeps the bool map.
        ({}, _router(dense=torch.int16), (torch.bfloat16, torch.bfloat16, torch.int16)),
        ({}, _router(hash_layer=True, dense=torch.int16), (torch.bfloat16,) * 2 + (torch.bool,)),
        ({}, _router(has_dense=False), (torch.bfloat16, torch.bfloat16, torch.bool)),
    ],
    ids=[
        "default",
        "router_fp32",
        "router_fp64",
        "fp32_residual",
        "fp32_residual_router_fp64",
        "dense_int16",
        "hash_keeps_bool",
        "router_without_dense_helper",
    ],
)
def test_moe_adapter_config_signature(config, router, expected):
    signature = null_items.MoELockstepAdapter().config_signature(
        _moe_stub("hybridep", router=router, **config)
    )
    assert (signature.hidden_dtype, signature.probs_dtype, signature.routing_dtype) == expected
    # Training: the MoE input (a norm output) and the probabilities require grad.
    assert signature.hidden_requires_grad and signature.probs_requires_grad
    assert not signature.observed


@cpu_only
def test_executor_null_needs_a_layer_context():
    """A null on an EU none of whose samples has a layer context (only reachable with unequal
    home counts, which ``run_step`` does not produce) raises a ValueError naming the
    equal-microbatch contract; types without lockstep calls do nothing."""
    executor = object.__new__(LoopRSBFExecutor)
    plan, _, log = _plan(["a"])
    executor.lockstep = plan
    executor._samples = {0: SimpleNamespace(ctx=None)}  # pylint: disable=protected-access
    executor.dispatcher = SimpleNamespace(eu=1)
    with pytest.raises(ValueError, match="core_f null item on EU 1 .*same number of microbatches"):
        executor._run_null(SubRoundType.CORE_F)  # pylint: disable=protected-access
    assert log == []
    executor.lockstep = LockstepPlan(by_type={t: () for t in SubRoundType})
    assert executor._run_null(SubRoundType.CODA) is None  # pylint: disable=protected-access


# ---------------------------------------------------------------------------------------------
# CPU: discovery on the stub 'E' model
# ---------------------------------------------------------------------------------------------

STUB = "*E*-*E*E/*E"  # prelude '*E', core '*-*E', coda '*E', MTP '*E'
STUB_MOE = dict(num_moe_experts=4, moe_ffn_hidden_size=128, moe_router_topk=2)


class _SpanningStubAdapter(LockstepAdapter):
    """Treats every stub as spanning ranks (discovery only; never run)."""

    def __init__(self, form=FORM_EMPTY):
        self._form = form

    def groups(self, module):
        return ["group"]

    def form(self, module):
        return self._form(module) if callable(self._form) else self._form

    def config_signature(self, module):
        return LockstepSignature(module.config.params_dtype, True)


@pytest.fixture
def stub_model(cpu_mtp):
    del cpu_mtp
    RecordingLockstepMLP.instances = []
    model = build_local_loop_model(STUB, moe_layer=stub_moe_layer_spec(), **RSBF, **STUB_MOE)
    return model, list(RecordingLockstepMLP.instances)


@cpu_only
def test_discovery_lists_each_type_calls(stub_model, monkeypatch):
    model, stubs = stub_model
    program = LoopProgram(model)
    # One-rank groups: no lockstep calls, but every stub tracks a budget flag.
    remove = register_lockstep_module(RecordingLockstepMLP, _SpanningStubAdapter())
    try:
        monkeypatch.setattr(null_items, "_spans_ranks", lambda groups: False)
        plan = build_lockstep_plan(program)
        assert not plan.active
        assert [name for name, _, _ in plan.budget] == [
            "decoder.layers.1.mlp",
            "decoder.layers.5.mlp",
            "decoder.layers.7.mlp",
            "mtp.layers.0.mtp_model_layer.layers.1.mlp",
        ]
        monkeypatch.setattr(null_items, "_spans_ranks", lambda groups: True)
        plan = build_lockstep_plan(program)
    finally:
        remove()
    by_type = {t.value: [stubs.index(e.module) for e in plan.by_type[t]] for t in SubRoundType}
    assert by_type == {
        "prelude_f": [0],
        "prelude_b": [0],
        "core_f": [1],
        "core_b": [1],
        "coda": [2, 3],
    }
    stack = model.decoder
    (prelude,) = plan.by_type[SubRoundType.PRELUDE_F]
    assert prelude.layer_config is stack._execution_layer_config_list[1]
    assert prelude.layer_number == stack.layers[1].layer_number - 1 and prelude.wrap is None
    coda, mtp_entry = plan.by_type[SubRoundType.CODA]
    mtp_layer = model.mtp.layers[0]
    assert coda.wrap is None
    assert mtp_entry.wrap == mtp_layer.get_inner_quantization_context
    assert mtp_entry.layer_config is mtp_layer.mtp_model_layer._execution_layer_config_list[1]
    assert mtp_entry.layer_number == mtp_layer.mtp_model_layer.layers[1].layer_number - 1
    assert [entry.module for entry in plan.entries()] == stubs
    assert all(plan.signature[stub].hidden_dtype == torch.float32 for stub in stubs)


@cpu_only
def test_discovery_rejects_what_nulls_cannot_replay(stub_model, monkeypatch):
    model, stubs = stub_model
    program = LoopProgram(model)
    monkeypatch.setattr(null_items, "_spans_ranks", lambda groups: True)
    # Null forms that differ between modules.
    forms = _SpanningStubAdapter(lambda m: FORM_NOMINAL if m is stubs[0] else FORM_EMPTY)
    remove = register_lockstep_module(RecordingLockstepMLP, forms)
    try:
        with pytest.raises(ValueError, match="different null forms"):
            build_lockstep_plan(program)
    finally:
        remove()
    # A lockstep module outside the prelude / core / coda / MTP layers (here: the stubs' fc1
    # linear layers registered too, two per sublayer).
    remove = register_lockstep_module(torch.nn.Linear, _SpanningStubAdapter())
    try:
        with pytest.raises(ValueError, match="more than one MoE module in sublayer"):
            build_lockstep_plan(program)
    finally:
        remove()
    # No null form for the module's configuration.

    def no_form(module):
        raise ValueError("Loop RSBF does not support: lockstep null items for X")

    remove = register_lockstep_module(RecordingLockstepMLP, _SpanningStubAdapter(no_form))
    try:
        with pytest.raises(ValueError, match="lockstep null items for X"):
            build_lockstep_plan(program)
    finally:
        remove()
    # A module the scan never reaches (the output layer, registered as lockstep).
    remove_stub = register_lockstep_module(RecordingLockstepMLP, _SpanningStubAdapter())
    remove = register_lockstep_module(type(model.output_layer), _SpanningStubAdapter())
    try:
        with pytest.raises(ValueError, match=r"outside the loop's .*'output_layer'"):
            build_lockstep_plan(program)
    finally:
        remove()
        remove_stub()


# ---------------------------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------------------------


def test_expert_parameter_names_map_to_their_global_index():
    """The reference builder's name map (TE grouped and sequential experts; shared experts and
    dense parameters are not experts)."""
    assert _expert_key("decoder.layers.1.mlp.experts.linear_fc1.weight3") == (
        "decoder.layers.1.mlp.experts.linear_fc1.weight",
        3,
        "",
    )
    assert _expert_key("decoder.layers.1.mlp.experts.local_experts.12.linear_fc2.weight") == (
        "decoder.layers.1.mlp.experts.local_experts.",
        12,
        ".linear_fc2.weight",
    )
    assert _expert_key("decoder.layers.1.mlp.shared_experts.linear_fc1.weight") is None
    assert _expert_key("decoder.layers.1.mlp.router.weight") is None
    assert _expert_key("mtp.layers.0.mtp_model_layer.layers.1.mlp.experts.linear_fc2.weight0")


VOCAB = 128
SEQ_GPU = 256
PER_RANK = 4


@pytest.fixture
def deterministic(monkeypatch):
    """Deterministic kernels and full-precision GEMMs for the comparisons."""
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cuda.matmul.allow_tf32 = tf32


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


def _one_rank_collection():
    """This rank's process groups with every data / expert group of one rank (the EP = 1,
    one-EU reference); TP / CP / PP are one rank already."""
    from megatron.core.process_groups_config import ProcessGroupCollection

    if not hasattr(_one_rank_collection, "groups"):
        # Created once per module on every rank (the CI process-group budget).
        _one_rank_collection.groups = [dist.new_group([r]) for r in range(dist.get_world_size())]
    own = _one_rank_collection.groups[dist.get_rank()]
    pg_collection = copy.copy(ProcessGroupCollection.use_mpu_process_groups())
    # TP = PP = CP = 1: every group over more than one rank is a data / expert group.
    for name, value in list(vars(pg_collection).items()):
        if isinstance(value, dist.ProcessGroup) and dist.get_world_size(value) > 1:
            setattr(pg_collection, name, own)
    return pg_collection


def _dsv4_fields(**overrides):
    """'WECEHE/WE' (DSv4 W / C / H attention, unfused; MoE with one hash layer in the prelude
    and expert bias; mHC 4; MTP 1), bf16, RSBF, R_max 4, expert parallelism over the world."""
    from megatron.core.transformer.enums import AttnBackend

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
        num_moe_experts=2 * WORLD,
        moe_ffn_hidden_size=256,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=1e-4,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
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
        loop_iterations=4,
        loop_prelude_sublayers=2,
        loop_coda_sublayers=2,
        deterministic_mode=True,
        attention_backend=AttnBackend.unfused,
        expert_model_parallel_size=WORLD,
        **RSBF,
    )
    fields.update(overrides)
    return fields


def _dsv4_model(pg_collection=None, **overrides):
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_dsv4_stack_spec
    from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    config = MLATransformerConfig(**_dsv4_fields(**overrides))
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    set_moe_metrics_tracker(MoEMetricsTracker())
    return HybridModel(
        config=config,
        hybrid_stack_spec=loop_hybrid_dsv4_stack_spec(config),
        vocab_size=VOCAB,
        max_sequence_length=1024,
        hybrid_layer_pattern="WECEHE/WE",
        position_embedding_type="rope",
        pg_collection=pg_collection,
    ).cuda()


# TE grouped experts: ``...experts.linear_fc1.weight3``; sequential experts:
# ``...experts.local_experts.3.linear_fc1.weight``.
_EXPERT = re.compile(r"^(?P<head>.*\.experts\..*\.(?:weight|bias))(?P<index>\d+)(?P<tail>)$")
_LOCAL_EXPERT = re.compile(r"^(?P<head>.*\.local_experts\.)(?P<index>\d+)(?P<tail>\..*)$")


def _expert_key(name):
    """``(head, local index, tail)`` of an expert parameter name, or None."""
    for pattern in (_LOCAL_EXPERT, _EXPERT):
        match = pattern.match(name)
        if match is not None and ".experts." in name:
            return match["head"], int(match["index"]), match["tail"]
    return None


def _sync_dense(model):
    """Every rank's non-expert parameters and buffers equal rank 0's (one DP replica)."""
    with torch.no_grad():
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            if _expert_key(name) is None and tensor.is_floating_point():
                dist.broadcast(tensor.data, src=0)


def _reference(model, **overrides):
    """The EP = 1 one-EU model with the EP = W model's weights (experts gathered)."""
    reference = _dsv4_model(_one_rank_collection(), expert_model_parallel_size=1, **overrides)
    local = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if _expert_key(name) is not None
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    per_rank = model.config.num_moe_experts // dist.get_world_size()
    sources = dict(model.named_parameters())
    sources.update(dict(model.named_buffers()))
    with torch.no_grad():
        for name, tensor in list(reference.named_parameters()) + list(reference.named_buffers()):
            key = _expert_key(name)
            if key is None:
                tensor.copy_(sources[name])
                continue
            head, index, tail = key
            rank, local_index = divmod(index, per_rank)
            tensor.copy_(gathered[rank][f"{head}{local_index}{tail}"])
    return reference


def _loss_func(loss_mask, output):
    losses = output.view(-1).float()
    mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * mask)
    num_tokens = mask.sum().clone().detach().to(torch.int)
    return (
        loss,
        num_tokens,
        {"lm loss": torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])},
    )


def _global_batch(total, seed=5, padded=False):
    """``total`` one-row microbatches; with ``padded`` the last quarter of every odd sample is
    padding (``padding_mask`` True, loss mask 0)."""
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, VOCAB, (total, SEQ_GPU), generator=generator)
    labels = torch.randint(0, VOCAB, (total, SEQ_GPU), generator=generator)
    loss_mask = (torch.rand(total, SEQ_GPU, generator=generator) > 0.2).float()
    padding_mask = None
    if padded:
        padding_mask = torch.zeros(total, SEQ_GPU, dtype=torch.bool)
        padding_mask[1::2, -(SEQ_GPU // 4) :] = True
        loss_mask = loss_mask * (~padding_mask).float()
    return [
        LoopMicrobatch(
            tokens=tokens[i : i + 1].cuda(),
            position_ids=torch.arange(SEQ_GPU)[None].cuda(),
            labels=labels[i : i + 1].cuda(),
            loss_mask=loss_mask[i : i + 1].cuda(),
            attention_mask=None,
            padding_mask=None if padding_mask is None else padding_mask[i : i + 1].cuda(),
            loss_func=_loss_func,
        )
        for i in range(total)
    ]


def _local(batch):
    world, rank = dist.get_world_size(), dist.get_rank()
    return [batch[j * world + rank] for j in range(len(batch) // world)]


def _depths(total):
    """R4 depths with a sparse tail (sample 1 deep, every EU idle in some rounds)."""
    return {g: (4 if g == 1 else 1 + (7 * g + 3) % 3) for g in range(total)}


@contextlib.contextmanager
def _fp32_grad_accumulation(model):
    """Adds every backward call's parameter gradient into an fp32 buffer and clears ``.grad``
    (what DDP's ``main_grad`` hook does). Yields ``{name: fp32 gradient}`` of the parameters
    that received one. bf16 ``.grad`` would round after each of the backward calls, and the
    arms compared here make different numbers of them per parameter (the one-EU reference 4W
    per-sample calls, each of W EUs about 4): that accumulation-order difference alone
    exceeds the normwise gate at W >= 4, so every GPU gradient comparison accumulates in fp32
    (stage-2 comparison rules)."""
    grads, handles = {}, []

    def hook(param, name):
        if name in grads:
            grads[name].add_(param.grad.detach())
        else:
            grads[name] = param.grad.detach().float().clone()
        param.grad = None

    for name, param in model.named_parameters():
        if param.requires_grad:
            handles.append(
                param.register_post_accumulate_grad_hook(lambda param, name=name: hook(param, name))
            )
    try:
        yield grads
    finally:
        for handle in handles:
            handle.remove()


def _run(model, microbatches, criterion, pg_collection, **kwargs):
    from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker

    set_moe_metrics_tracker(MoEMetricsTracker())
    program = LoopProgram(model, exit_criterion=criterion)
    executor = LoopRSBFExecutor(program, pg_collection=pg_collection, **kwargs)
    model.zero_grad(set_to_none=True)
    with _fp32_grad_accumulation(model) as grads:
        reports = executor.run_step(
            lambda iterator, m: next(iterator), iter(microbatches), len(microbatches), model=model
        )
    losses = {
        record.key.global_index: report["lm loss"][0].clone()
        for record, report in zip(executor.stats.loss_records, reports)
    }
    # named_parameters order (not the hooks' firing order): callers all-reduce by iteration.
    grads = {name: grads[name] for name, _ in model.named_parameters() if name in grads}
    return losses, grads, executor


def _expert_bias_counts(model):
    return {
        name: module.local_tokens_per_expert.detach().clone()
        for name, module in model.named_modules()
        if getattr(module, "local_tokens_per_expert", None) is not None
    }


def _reset_expert_bias_counts(model):
    for module in model.modules():
        if getattr(module, "local_tokens_per_expert", None) is not None:
            module.local_tokens_per_expert.zero_()


def _assert_loss_rule(actual, expected, label):
    """Bitwise where every kernel has the same shape; otherwise within 2^-7 |L| (bf16 rule,
    MoE co-scheduling changes the expert GEMM row counts). Prints the measured maximum."""
    assert actual.keys() == expected.keys()
    worst = 0.0
    for key, loss in actual.items():
        difference = (loss.float() - expected[key].float()).abs().item()
        worst = max(worst, difference / max(expected[key].abs().item(), 1e-30))
        assert difference <= 2.0**-7 * expected[key].abs().item(), (label, key, loss, expected[key])
    print(f"{label}: max relative loss difference {worst:.3e}")


def _assert_normwise(actual, expected, label):
    for name, grad in expected.items():
        error = (actual[name] - grad).norm() / grad.norm().clamp_min(1e-30)
        assert error <= 2.0**-7, f"{label} {name}: normwise {error.item():.3e} > 2^-7"


def _dense_summed(model, grads):
    """The G2 comparison form of one EP = W run's gradients: dense gradients summed over the
    ranks (zeros where a rank has none, so every rank reduces the same names in the same
    order), expert gradients as they are (each expert lives on one rank)."""
    out = {}
    for name, param in model.named_parameters():
        grad = grads.get(name)
        if _expert_key(name) is None:
            total = torch.zeros_like(param, dtype=torch.float32) if grad is None else grad.clone()
            dist.all_reduce(total)
            out[name] = total
        elif grad is not None:
            out[name] = grad
    return out


def _ep_ranks(model):
    for module in model.modules():
        dispatcher = getattr(module, "token_dispatcher", None)
        if dispatcher is not None and getattr(dispatcher, "ep_group", None) is not None:
            return tuple(sorted(dist.get_process_group_ranks(dispatcher.ep_group)))
    raise AssertionError("no expert-parallel MoE layer")


# ---------------------------------------------------------------------------------------------
# N1: one MoE layer, null ranks host their experts
# ---------------------------------------------------------------------------------------------


def _moe_layer(grouped):
    import torch.nn.functional as F

    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=2 * WORLD,
        moe_ffn_hidden_size=128,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=grouped,
        expert_model_parallel_size=WORLD,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
    )
    submodules = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(
            num_experts=config.num_moe_experts, moe_grouped_gemm=grouped
        ).mlp
    )
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    layer = MoELayer(config, submodules).cuda()
    layer.set_layer_number(1)
    return layer


@gpu_ep
@pytest.mark.parametrize("grouped", [False, True], ids=["sequential", "te_grouped"])
def test_null_ranks_host_their_experts(deterministic, grouped):
    """N1: rank 0 real, the others null (and the reverse), against every rank real: no hang;
    the real ranks' outputs equal the all-real outputs; the experts' gradients of the two
    partial arms add up to the all-real ones (each token's expert gradient goes to the
    expert's owner whatever runs elsewhere); the null ranks' experts received rank 0's tokens
    (non-zero gradients); a null never runs the router."""
    del deterministic
    layer = _moe_layer(grouped)
    adapter = null_items.MoELockstepAdapter()
    entry = LockstepEntry(
        module=layer,
        adapter=adapter,
        layer_config=layer.config,
        layer_number=0,
        form=adapter.form(layer),
        name="moe",
    )
    assert entry.form == FORM_EMPTY
    plan = LockstepPlan(by_type={t: (entry,) for t in SubRoundType})
    plan.signature[layer] = adapter.config_signature(layer)
    rank = dist.get_rank()
    generator = torch.Generator().manual_seed(17 + rank)
    x = torch.randn(32, 1, 64, generator=generator).cuda()
    upstream = torch.randn(32, 1, 64, generator=generator).cuda()

    def arm(real):
        layer.zero_grad(set_to_none=True)
        output = None
        with plan.observing(record_calls=True):
            if real:
                plan.begin_item()
                leaf = x.clone().requires_grad_()
                output, _ = layer(leaf)
                torch.autograd.backward(output, upstream)
                plan.end_item(SubRoundType.CORE_B, "real")
            else:
                assert (
                    run_null(
                        plan,
                        SubRoundType.CORE_B,
                        outer_context=contextlib.nullcontext(),
                        inner_context=lambda config, number: contextlib.nullcontext(),
                        sequence=32,
                        device=x.device,
                    )
                    == 1
                )
        experts = {
            name: param.grad.detach().clone()
            for name, param in layer.experts.named_parameters()
            if param.grad is not None
        }
        router = [param.grad for param in layer.router.parameters()]
        return (None if output is None else output.detach().clone()), experts, router

    all_real = arm(True)
    only_zero = arm(rank == 0)
    others = arm(rank != 0)
    assert plan.signature[layer].observed
    if rank == 0:
        torch.testing.assert_close(only_zero[0], all_real[0], rtol=1e-5, atol=1e-6)
        assert all(grad is None for grad in others[2])
    else:
        torch.testing.assert_close(others[0], all_real[0], rtol=1e-5, atol=1e-6)
        assert all(grad is None for grad in only_zero[2])
        # Rank 0's tokens reached this rank's experts while it ran a null.
        assert any(grad.abs().sum() > 0 for grad in only_zero[1].values())
    assert all_real[1].keys() == only_zero[1].keys() == others[1].keys()
    for name, grad in all_real[1].items():
        torch.testing.assert_close(
            only_zero[1][name] + others[1][name],
            grad,
            rtol=1e-5,
            atol=1e-6 * grad.abs().max().item(),
            msg=lambda text, name=name: f"{name}: {text}",
        )


# ---------------------------------------------------------------------------------------------
# G2 (+ N2 / N4): the DSv4-like model at EP = W against one EU
# ---------------------------------------------------------------------------------------------


@gpu_ep
@pytest.mark.parametrize("router_dtype", ["fp32", None], ids=["router_fp32", "router_bf16"])
def test_dsv4_lockstep_w_eus_match_one_eu(deterministic, router_dtype):
    """G2 / N2 / N4 (see the module docstring)."""
    del deterministic
    world = dist.get_world_size()
    total = PER_RANK * world
    batch = _global_batch(total)
    depths = _depths(total)
    criterion = PrescribedDepthCriterion(depths)
    with _hadamard_fallback():
        model = _dsv4_model(moe_router_dtype=router_dtype)
        _sync_dense(model)
        reference = _reference(model, moe_router_dtype=router_dtype)
        ref_losses, ref_grads, ref_executor = _run(
            reference, batch, criterion, SimpleNamespace(dp=None)
        )
        assert ref_executor.num_eus == 1 and not ref_executor.lockstep.active
        ref_counts = _expert_bias_counts(reference)
        from megatron.core.process_groups_config import ProcessGroupCollection

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        ep = _ep_ranks(model)
        assert ep == tuple(range(world))
        runs = []
        for _ in range(2):
            _reset_expert_bias_counts(model)
            with CollectiveOpLogger(groups=[ep]) as log:
                losses, grads, executor = _run(
                    model,
                    _local(batch),
                    criterion,
                    pg_collection,
                    check_lockstep=True,
                    check_plan=False,
                    poison_on_free=True,
                )
            ops = [op for op in log.ops if op[0] in ("all_to_all_single", "all_gather_into_tensor")]
            runs.append((losses, grads, executor, ops, _expert_bias_counts(model)))
    losses, grads, executor, ops, counts = runs[0]
    stats = executor.stats
    plan = executor.lockstep
    # Discovery: prelude hash E, core E, coda E + MTP E.
    assert [len(plan.by_type[t]) for t in SubRoundType] == [1, 1, 1, 2, 1]
    assert all(plan.signature[entry.module].observed for entry in plan.entries())
    for kind in ("core_f", "core_b", "coda", "prelude_b"):
        assert stats.nulls[kind] > 0, (kind, stats.nulls)
    assert stats.flagged_ranks["over_budget"] == []
    # Self-determinism of the W-EU arm (bitwise), then the one-EU reference (loss rule).
    for key, loss in runs[1][0].items():
        assert torch.equal(loss, losses[key]), key
    assert runs[1][3] == ops
    every_loss = [None] * world
    dist.all_gather_object(every_loss, {k: v.cpu() for k, v in losses.items()})
    merged = {k: v for part in every_loss for k, v in part.items()}
    _assert_loss_rule(merged, {k: v.cpu() for k, v in ref_losses.items()}, "W EUs vs one EU")
    # The MoE collectives: the same sequence on every rank, forward and backward.
    assert any(phase == "bwd" for _, _, phase in ops) and any(p == "fwd" for _, _, p in ops)
    every_ops = [None] * world
    dist.all_gather_object(every_ops, ops)
    assert all(other == every_ops[0] for other in every_ops), "MoE collective order differs"
    # Expert-bias token counts: placement-free, summed over the ranks.
    assert counts.keys() == ref_counts.keys() and counts
    for name, count in counts.items():
        total_count = count.clone()
        dist.all_reduce(total_count)
        assert torch.equal(total_count, ref_counts[name]), name
    # Gradients: dense summed over the ranks; each expert lives on one rank.
    per_rank = model.config.num_moe_experts // world
    dense, experts, ref_dense, ref_experts = {}, {}, {}, {}
    for name, grad in grads.items():
        key = _expert_key(name)
        if key is None:
            dist.all_reduce(grad)
            dense[name] = grad
        else:
            head, index, tail = key
            experts[f"{head}{dist.get_rank() * per_rank + index}{tail}"] = grad
    for name, grad in ref_grads.items():
        (ref_dense if _expert_key(name) is None else ref_experts)[name] = grad
    assert dense.keys() == ref_dense.keys()
    _assert_normwise(dense, ref_dense, "dense")
    _assert_normwise(experts, {name: ref_experts[name] for name in experts}, "experts")


@gpu_ep
@pytest.mark.parametrize("waves", [2, 4])
def test_waves_match_one_pass_with_lockstep_moe(deterministic, waves):
    """G4 on the G2 model: waves 2 and 4 (n = 1 sample per wave and EU at 4) against one pass:
    losses at the loss rule, gradients (dense summed over the ranks, experts per owner) by the
    normwise rule; each arm self-deterministic (losses bitwise); the lockstep checks on."""
    del deterministic
    from megatron.core.process_groups_config import ProcessGroupCollection

    world = dist.get_world_size()
    total = PER_RANK * world
    batch = _global_batch(total)
    criterion = PrescribedDepthCriterion(_depths(total))
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    results = {}
    with _hadamard_fallback():
        model = _dsv4_model()
        _sync_dense(model)
        for arm in (1, waves, 1, waves):
            losses, grads, executor = _run(
                model, _local(batch), criterion, pg_collection, waves=arm, check_lockstep=True
            )
            assert len(executor.stats.store_peaks) == arm
            assert executor.stats.flagged_ranks["over_budget"] == []
            if arm in results:
                for key, loss in losses.items():
                    assert torch.equal(loss, results[arm][0][key]), (arm, key)
            results[arm] = (losses, _dense_summed(model, grads))
    _assert_loss_rule(results[waves][0], results[1][0], f"waves {waves} vs one pass")
    _assert_normwise(results[waves][1], results[1][1], f"waves {waves} vs one pass")


@gpu_ep
def test_move_gate_on_the_lockstep_workload():
    """G6: sampled R4-uniform depths, 8 samples per EU, 8 steps: the moves are <= 5% of the
    active sample-rounds over the steps (the aggregate gate); bulk / tail maxima printed."""
    from megatron.core.process_groups_config import ProcessGroupCollection

    world = dist.get_world_size()
    per_rank = 8
    batch = _global_batch(per_rank * world, seed=9)
    criterion = SampledDepthCriterion([0.25] * 4, seed=1234)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    moves = actives = 0
    bulk = tail = 0.0
    with _hadamard_fallback():
        model = _dsv4_model()
        _sync_dense(model)
        for iteration in range(8):
            program = LoopProgram(model, exit_criterion=criterion)
            executor = LoopRSBFExecutor(program, pg_collection=pg_collection)
            model.zero_grad(set_to_none=True)
            executor.run_step(
                lambda iterator, m: next(iterator),
                iter(_local(batch)),
                per_rank,
                model=model,
                iteration=iteration,
            )
            stats = executor.stats
            for wave_moves, wave_active in zip(stats.moves_by_round, stats.active_by_round):
                for m, a in zip(wave_moves, wave_active):
                    moves += m
                    actives += a
                    if a >= 2 * world:
                        bulk = max(bulk, m / a)
                    elif a:
                        tail = max(tail, m / a)
    aggregate = moves / actives
    print(
        f"W={world}: moves {aggregate:.2%} of active sample-rounds (bulk max {bulk:.1%}, "
        f"tail max {tail:.1%})"
    )
    assert aggregate <= 0.05


# ---------------------------------------------------------------------------------------------
# N4: quantization contexts of nulls (FP8 current scaling)
# ---------------------------------------------------------------------------------------------


def _fp8_available():
    try:
        from transformer_engine.pytorch.fp8 import check_fp8_support

        return bool(check_fp8_support()[0])
    except Exception:  # pylint: disable=broad-except
        return False


@gpu_ep
@pytest.mark.skipif(not EP_GPU or not _fp8_available(), reason="needs FP8 (Hopper or later)")
def test_null_quantization_contexts_match_the_real_items(monkeypatch):
    """N4: under FP8 current scaling every MoE call of a null enters the inner quantization
    context of its hosting sublayer (layer config, layer number; the MTP E its own) in the
    order of the real items of its type, and the MTP E also the MTP depth's wrap
    (``get_inner_quantization_context``: ``multi_token_prediction.get_fp8_context`` of the MTP
    layer's config) around it, as the real MTP call does; every real call's signature equals
    the null's (check_lockstep)."""
    from megatron.core.models.hybrid import hybrid_block
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer import multi_token_prediction
    from megatron.core.transformer.loop import rsbf_executor
    from megatron.core.transformer.loop.rsbf_plan import SubRound

    records = []
    state = {"marker": None, "serial": 0}

    def recording(module, kind):
        real_fp8_context = module.get_fp8_context

        def recording_fp8_context(config, layer_no=-1, is_init=False):
            records.append((state["marker"], (kind, id(config), layer_no)))
            return real_fp8_context(config, layer_no, is_init)

        monkeypatch.setattr(module, "get_fp8_context", recording_fp8_context)

    recording(hybrid_block, "stack")
    recording(multi_token_prediction, "mtp")
    original_execute = rsbf_executor.LoopRSBFExecutor._execute

    def execute(executor, steps):
        for step in steps:
            if isinstance(step, SubRound):
                null = step.items[executor.dispatcher.eu] is None
                state["marker"] = (state["serial"], step.type, null)
                state["serial"] += 1
            else:
                state["marker"] = None
            original_execute(executor, [step])
        state["marker"] = None

    monkeypatch.setattr(rsbf_executor.LoopRSBFExecutor, "_execute", execute)
    world = dist.get_world_size()
    total = PER_RANK * world
    batch = _global_batch(total)
    with _hadamard_fallback():
        model = _dsv4_model(
            fp8="hybrid", fp8_recipe="tensorwise", disable_parameter_transpose_cache=True
        )
        _sync_dense(model)
        _, _, executor = _run(
            model,
            _local(batch),
            PrescribedDepthCriterion(_depths(total)),
            ProcessGroupCollection.use_mpu_process_groups(),
            check_lockstep=True,
        )
    plan = executor.lockstep
    moe_contexts = {
        kind_type: [("stack", id(e.layer_config), e.layer_number) for e in plan.by_type[kind_type]]
        for kind_type in SubRoundType
    }
    # A null's exact sequence: each MoE call's inner context, the MTP E's preceded by its wrap.
    null_contexts = {}
    for kind_type in SubRoundType:
        sequence = []
        for entry in plan.by_type[kind_type]:
            if entry.wrap is not None:
                sequence.append(("mtp", id(entry.wrap.__self__.config), -1))
            sequence.append(("stack", id(entry.layer_config), entry.layer_number))
        null_contexts[kind_type] = sequence
    mtp_entries = [e for e in plan.by_type[SubRoundType.CODA] if e.wrap is not None]
    assert mtp_entries, "the CODA plan has no MTP E"
    keys = {key for value in moe_contexts.values() for key in value}
    seen_null = set()
    # The contexts entered per executed sub-round (serial).
    groups = {}
    for marker, key in records:
        if marker is not None:
            groups.setdefault(marker, []).append(key)
    for (_, kind_type, null), contexts in groups.items():
        moe = [key for key in contexts if key in keys]
        assert moe == moe_contexts[kind_type], (kind_type, null, moe)
        if null:
            assert contexts == null_contexts[kind_type], (kind_type, contexts)
            seen_null.add(kind_type)
        for entry in mtp_entries if kind_type == SubRoundType.CODA else ():
            # Real or null: the MTP layer's quantization context is entered before its E.
            key = ("stack", id(entry.layer_config), entry.layer_number)
            position = len(contexts) - 1 - contexts[::-1].index(key)  # the MTP E is last
            wrap = ("mtp", id(entry.wrap.__self__.config), -1)
            assert wrap in contexts[:position], (null, contexts)
    assert seen_null, "no null ran"
    assert SubRoundType.CODA in seen_null, ("no CODA null ran", seen_null)


# ---------------------------------------------------------------------------------------------
# G3 / N3: HybridEP static budget
# ---------------------------------------------------------------------------------------------


def _hybridep_available():
    """HybridEP importable and a Blackwell GPU: the static-budget HybridEP + TE op-fuser
    grouped MLP is exercised upstream on compute capability >= 10 only (the paged-stashing
    tests); on H100 the grouped linear rejects the budget-sized token buffer."""
    if not CUDA or torch.cuda.get_device_capability()[0] < 10:
        return False
    try:
        from megatron.core.transformer.moe import token_dispatcher

        return token_dispatcher.hybrid_ep_dispatch is not None
    except Exception:  # pylint: disable=broad-except
        return False


def _swiglu():
    """SwiGLU experts: the TE op-fuser grouped MLP fuses SwiGLU, not the default gelu."""
    import torch.nn.functional as F

    return dict(gated_linear_unit=True, activation_func=F.silu)


HYBRIDEP = dict(
    moe_token_dispatcher_type="flex",
    moe_flex_dispatcher_backend="hybridep",
    use_transformer_engine_op_fuser=True,
    moe_router_dtype="fp32",
    **_swiglu(),
)


@contextlib.contextmanager
def _hybridep_dispatch_log(model):
    """Logs every HybridEP dispatch of ``model``'s MoE layers on this rank, in call order:
    whether a null made it, the rows and padded rows passed, the routes sent per global expert
    (after the manager's padding exclusion), the static budget (``num_permuted_tokens``) and
    the tokens received per local expert. Yields the list of per-call dicts (host values)."""
    from megatron.core.transformer.moe.moe_layer import MoELayer

    calls, state, restore = [], {"null": False}, []

    def wrap(manager):
        setup, dispatch = manager.setup_metadata, manager.dispatch

        def setup_metadata(routing_map, probs, padding_mask=None):
            output = setup(routing_map, probs, padding_mask=padding_mask)
            if manager.routing_map is not None:
                sent = manager.routing_map.sum(0)
            else:
                index = manager.topk_idx.long()
                sent = torch.bincount(index[index >= 0], minlength=manager.num_experts)
            calls.append(
                dict(
                    null=state["null"],
                    rows=routing_map.shape[0],
                    padded=None if padding_mask is None else padding_mask.sum(),
                    sent=sent.detach().clone(),
                    budget=manager.num_permuted_tokens,
                )
            )
            return output

        def dispatch_(hidden_states, *args, **kwargs):
            output = dispatch(hidden_states, *args, **kwargs)
            calls[-1]["received"] = manager.tokens_per_expert.detach().clone()
            return output

        manager.setup_metadata, manager.dispatch = setup_metadata, dispatch_
        restore.append(manager)

    for module in model.modules():
        if isinstance(module, MoELayer):
            wrap(module.token_dispatcher._comm_manager)  # pylint: disable=protected-access
    real_null_forward = null_items.MoELockstepAdapter.null_forward

    def null_forward(adapter, entry, hidden, signature):
        state["null"] = True
        try:
            return real_null_forward(adapter, entry, hidden, signature)
        finally:
            state["null"] = False

    null_items.MoELockstepAdapter.null_forward = null_forward
    try:
        yield calls
    finally:
        null_items.MoELockstepAdapter.null_forward = real_null_forward
        for manager in restore:
            del manager.setup_metadata, manager.dispatch  # back to the class methods
        for call in calls:
            call["padded"] = 0 if call["padded"] is None else int(call["padded"])
            call["sent"] = call["sent"].cpu().tolist()
            call["received"] = call["received"].cpu().tolist()


def _check_dispatch_log(calls, config, label):
    """Every rank's dispatch log (gathered): the same call count; a null sends no route (and has
    no padding), a real call ``topk`` routes per non-padded row; per call, every local expert
    receives exactly the routes the EP group sent it (up to the GEMM alignment of each
    expert's segment), so padded and null rows take no budget; one static budget for every
    call. Returns the budgets (per rank, per call) and the padded real calls of this rank."""
    from megatron.core.transformer.moe.moe_utils import get_align_size_for_quantization

    world = dist.get_world_size()
    every = [None] * world
    dist.all_gather_object(every, calls)
    assert len({len(log) for log in every}) == 1, (label, [len(log) for log in every])
    topk, per_rank = config.moe_router_topk, config.num_moe_experts // world
    align = max(get_align_size_for_quantization(config), 1)
    for rank, log in enumerate(every):
        for index, call in enumerate(log):
            routes = sum(call["sent"])
            if call["null"]:
                assert routes == 0 and call["padded"] == 0, (label, rank, index, call)
            else:
                assert routes == topk * (call["rows"] - call["padded"]), (label, rank, index, call)
            for local, received in enumerate(call["received"]):
                expert = rank * per_rank + local
                sent = sum(other[index]["sent"][expert] for other in every)
                assert sent <= received < sent + align, (label, rank, index, expert, sent, received)
    budgets = {call["budget"] for log in every for call in log}
    assert len(budgets) == 1 and None not in budgets, (label, budgets)
    print(
        f"{label}: {len(calls)} dispatches per rank, {sum(c['null'] for c in calls)} nulls on "
        f"rank {dist.get_rank()}, budget {budgets}"
    )
    return [[call["budget"] for call in log] for log in every], [
        call for call in calls if call["padded"]
    ]


@gpu_ep
@pytest.mark.skipif(
    not EP_GPU or not _hybridep_available(),
    reason="needs HybridEP on Blackwell (run on lyris GB200)",
)
def test_hybridep_static_budget_nulls(deterministic):
    """G3 / N3: under the static per-rank budget (HybridEP, rank capacity 2.0) the nominal-row
    nulls next to real items never overflow (raise mode: the step completes) and take no
    budget (the dispatch log, ``_check_dispatch_log``); the losses equal the all-to-all model
    with the same weights at the loss rule and the gradients by the normwise rule (dense summed
    over the ranks, experts per owner); each arm is self-deterministic. N3: with a quarter of
    every odd sample's tail as padding, the real items hand their padding mask to HybridEP,
    the padded rows send nothing, the budgets equal the unpadded run's and ``over_budget``
    stays unset. With a 0.05 budget the dispatch overflows and every rank raises the same
    over-budget error naming the same EUs."""
    del deterministic
    from megatron.core.process_groups_config import ProcessGroupCollection

    world = dist.get_world_size()
    total = PER_RANK * world
    batch = _global_batch(total)
    criterion = PrescribedDepthCriterion(_depths(total))
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    with _hadamard_fallback():
        static = _dsv4_model(**HYBRIDEP, moe_expert_rank_capacity_factor=2.0)
        _sync_dense(static)
        plan = build_lockstep_plan(LoopProgram(static))
        assert {entry.form for entry in plan.entries()} == {FORM_NOMINAL}
        with _hybridep_dispatch_log(static) as calls:
            first, first_grads, executor = _run(static, _local(batch), criterion, pg_collection)
        budgets, _ = _check_dispatch_log(calls, static.config, "G3 static budget")
        second, _, _ = _run(static, _local(batch), criterion, pg_collection)
        assert executor.stats.flagged_ranks["over_budget"] == []
        assert executor.stats.null_moe_calls > 0
        for key, loss in second.items():
            assert torch.equal(loss, first[key]), key
        # N3: padded samples (every odd sample's last quarter).
        padded_batch = _global_batch(total, padded=True)
        with _hybridep_dispatch_log(static) as padded_calls:
            _, _, padded_executor = _run(static, _local(padded_batch), criterion, pg_collection)
        padded_budgets, padded_real = _check_dispatch_log(
            padded_calls, static.config, "N3 padded samples"
        )
        assert padded_executor.stats.flagged_ranks["over_budget"] == []
        assert padded_budgets == budgets
        # A quarter of the sample (the MTP depth's rolled mask may differ by one row).
        assert all(abs(call["padded"] - SEQ_GPU // 4) <= 1 for call in padded_real), padded_real
        any_padded = torch.tensor([len(padded_real)], device="cuda")
        dist.all_reduce(any_padded)
        assert any_padded.item() > 0, "no real call received a padding mask"
        first_grads = _dense_summed(static, first_grads)
        dynamic = _dsv4_model(
            use_transformer_engine_op_fuser=True,
            moe_router_dtype="fp32",
            moe_token_dispatcher_type="alltoall",
            **_swiglu(),
        )
        sources = dict(static.named_parameters())
        static_names = set(sources)
        dynamic_names = {name for name, _ in dynamic.named_parameters()}
        assert static_names == dynamic_names, sorted(static_names ^ dynamic_names)
        with torch.no_grad():
            for name, param in dynamic.named_parameters():
                param.copy_(sources[name])
        expected, expected_grads, _ = _run(dynamic, _local(batch), criterion, pg_collection)
        _assert_loss_rule(first, expected, "HybridEP static budget vs all-to-all")
        _assert_normwise(
            first_grads,
            _dense_summed(dynamic, expected_grads),
            "HybridEP static budget vs all-to-all",
        )
        tiny = _dsv4_model(**HYBRIDEP, moe_expert_rank_capacity_factor=0.05)
        _sync_dense(tiny)
        message = None
        try:
            _run(tiny, _local(batch), criterion, pg_collection)
        except RuntimeError as error:
            message = str(error)
    messages = [None] * world
    dist.all_gather_object(messages, message)
    assert all(text is not None and OVER_BUDGET in text for text in messages), messages
    flagged = {text.split(" on domain EUs ")[1].split(" (")[0] for text in messages}
    assert len(flagged) == 1, messages
