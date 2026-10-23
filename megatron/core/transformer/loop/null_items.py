# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Lockstep MoE across the EUs of an RSBF domain: the MoE calls of each sub-round type and the
null items that replay their collectives.

With expert parallelism the ranks of an EP group must issue the same sequence of MoE dispatch
/ combine collectives, forward and backward. The RSBF plan runs typed sub-rounds (PRELUDE-F,
CORE-F, CORE-B, CODA, PRELUDE-B) in which every EU of the domain runs one item of the
sub-round's type or a null of that type. A real item calls the MoE layers of its cell (the
prelude's, the core's, or the coda's followed by the MTP depths'), in forward order and, for the
backward types, back-propagates through them in reverse. A null makes the same calls without
real tokens:

* it bypasses the router and the shared experts (no router statistics, no aux / z / seq-aux
  loss, no expert-bias counts, no hash lookup; they issue no expert-parallel collective at
  TP = 1) and feeds the dispatcher an empty routing: zero rows for the all-to-all dispatcher
  (form "empty"; dropless only), or the nominal ``S x c`` all-excluded rows for HybridEP (form
  "nominal": every rank passes HybridEP the same row count, and under a static per-rank budget
  the null rank keeps the receive budget of a real item while sending nothing);
* its experts still compute, forward and backward, the tokens other ranks route to them (the
  null rank hosts its experts), inside the hosting sublayer's quantization contexts;
* its leaves take the real calls' dtypes and ``requires_grad`` pattern (``LockstepSignature``),
  so autograd builds the same functions and runs the combine, token and probs all-to-all
  backwards in the same order; the modules are chained (each null output feeds the next), so
  the backward visits them in reverse, as a real item's does.

``build_lockstep_plan`` discovers the MoE calls per sub-round type (``LockstepPlan.by_type``)
from the model; modules whose collective groups span one rank are not lockstep (dense models and
EP = 1 get empty tuples, and their nulls do nothing). ``register_lockstep_module`` lets tests
drive a stub module class through the same machinery.
"""

import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Optional, Sequence

import torch
import torch.distributed as dist
from torch import Tensor, nn

from megatron.core.transformer.loop.rsbf_plan import SubRoundType

__all__ = [
    "FORM_EMPTY",
    "FORM_NOMINAL",
    "LockstepAdapter",
    "LockstepEntry",
    "LockstepPlan",
    "LockstepSignature",
    "MoELockstepAdapter",
    "build_lockstep_plan",
    "register_lockstep_module",
    "run_null",
]

# Null forms (the design's form Z and form P).
FORM_EMPTY = "empty"  # zero rows (dropless all-to-all)
FORM_NOMINAL = "nominal"  # S x c rows, every route excluded (HybridEP)

UNSUPPORTED = "Loop RSBF does not support: "
LOCKSTEP = "lockstep"  # message prefix of the lockstep checks

# Sub-round types whose items run a backward (nulls of these types back-propagate too).
_BACKWARD_TYPES = frozenset((SubRoundType.CORE_B, SubRoundType.CODA, SubRoundType.PRELUDE_B))


@dataclass
class LockstepSignature:
    """What a null's leaves must look like for one lockstep module: the dtypes and
    ``requires_grad`` of the MoE input and the router probabilities, and the routing map's
    dtype (``torch.bool`` for a ``[rows, E]`` map, an integer dtype for dense ``[rows, topk]``
    route indices). Starts from the config rule and is confirmed by the first real call."""

    hidden_dtype: torch.dtype
    hidden_requires_grad: bool
    probs_dtype: Optional[torch.dtype] = None
    probs_requires_grad: bool = False
    routing_dtype: Optional[torch.dtype] = None
    observed: bool = False

    def key(self) -> tuple:
        """The compared fields."""
        return (
            self.hidden_dtype,
            self.hidden_requires_grad,
            self.probs_dtype,
            self.probs_requires_grad,
            self.routing_dtype,
        )


class LockstepAdapter:
    """How the executor drives one class of lockstep modules (``MoELayer`` by default; tests
    register stubs with ``register_lockstep_module``)."""

    def groups(self, module: nn.Module) -> list:
        """The process groups whose ranks issue the module's collectives (lockstep when any
        spans more than one rank)."""
        raise NotImplementedError

    def form(self, module: nn.Module) -> str:
        """``FORM_EMPTY`` or ``FORM_NOMINAL``; raises ``ValueError`` when no null form is
        defined for the module's configuration."""
        raise NotImplementedError

    def config_signature(self, module: nn.Module) -> LockstepSignature:
        """The null leaves' signature from the configuration (before any real call)."""
        raise NotImplementedError

    def observe(
        self,
        module: nn.Module,
        on_call: Callable[[nn.Module], None],
        on_signature: Callable[[nn.Module, LockstepSignature], None],
    ) -> list:
        """Install hooks reporting each real call (``on_call``, at its start) and its
        signature (``on_signature``); returns the hook handles."""
        raise NotImplementedError

    def null_forward(
        self, entry: "LockstepEntry", hidden: Tensor, signature: LockstepSignature
    ) -> Tensor:
        """One null call of ``entry.module`` on ``hidden`` (``[rows, c, H]``); returns its
        output (the same shape)."""
        raise NotImplementedError

    def over_budget(self, module: nn.Module) -> Optional[Tensor]:
        """The module's static-budget overflow flag (a device tensor), or None."""
        del module
        return None

    def reset_over_budget(self, module: nn.Module) -> None:
        """Clear the module's static-budget overflow flag."""
        del module


class MoELockstepAdapter(LockstepAdapter):
    """``MoELayer``: the all-to-all dispatcher (dropless, form "empty") and the flex dispatcher
    with the HybridEP backend (form "nominal"); everything else has no null form."""

    def groups(self, module):
        dispatcher = module.token_dispatcher
        groups = []
        for name in ("ep_group", "tp_ep_group"):
            group = getattr(dispatcher, name, None)
            if group is not None and all(group is not other for other in groups):
                groups.append(group)
        return groups

    def form(self, module):
        from megatron.core.transformer.moe.token_dispatcher import (
            MoEAlltoAllTokenDispatcher,
            MoEFlexTokenDispatcher,
            _HybridEPManager,
        )

        config = module.config
        dispatcher = module.token_dispatcher
        if config.moe_expert_capacity_factor is None:
            if isinstance(dispatcher, MoEAlltoAllTokenDispatcher):
                return FORM_EMPTY
            if isinstance(dispatcher, MoEFlexTokenDispatcher) and isinstance(
                getattr(dispatcher, "_comm_manager", None), _HybridEPManager
            ):
                return FORM_NOMINAL
        raise ValueError(
            UNSUPPORTED + f"lockstep null items for the MoE dispatcher {type(dispatcher).__name__} "
            f"(moe_token_dispatcher_type={config.moe_token_dispatcher_type!r}, "
            f"moe_flex_dispatcher_backend={config.moe_flex_dispatcher_backend!r}, "
            f"moe_expert_capacity_factor={config.moe_expert_capacity_factor!r}); expert "
            "parallelism under RSBF needs the dropless all-to-all dispatcher or flex + hybridep"
        )

    def config_signature(self, module):
        config = module.config
        hidden = torch.float32 if config.fp32_residual_connection else config.params_dtype
        probs = {"fp32": torch.float32, "fp64": torch.float64}.get(config.moe_router_dtype, hidden)
        router = module.router
        routing = torch.bool
        if not getattr(router, "is_hash_layer", False):
            dense = getattr(router, "_dense_route_indices_dtype", lambda: None)()
            if dense is not None:
                routing = dense
        # The MoE input is a norm output (trainable weights) and the probabilities a function
        # of it: both require grad in training.
        return LockstepSignature(
            hidden_dtype=hidden,
            hidden_requires_grad=True,
            probs_dtype=probs,
            probs_requires_grad=True,
            routing_dtype=routing,
        )

    def observe(self, module, on_call, on_signature):
        pending = {}

        def before(mod, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            pending["hidden"] = (hidden.dtype, hidden.requires_grad)
            on_call(mod)

        def after_router(router, args, output):
            del router, args
            probs, routing_map = output
            hidden_dtype, hidden_requires_grad = pending.pop("hidden", (None, False))
            on_signature(
                module,
                LockstepSignature(
                    hidden_dtype=hidden_dtype,
                    hidden_requires_grad=hidden_requires_grad,
                    probs_dtype=probs.dtype,
                    probs_requires_grad=probs.requires_grad,
                    routing_dtype=routing_map.dtype,
                ),
            )

        return [
            module.register_forward_pre_hook(before, with_kwargs=True),
            module.router.register_forward_hook(after_router),
        ]

    def null_forward(self, entry, hidden, signature):
        moe = entry.module
        # The dispatcher stages in MoELayer.forward's order, without route() and the shared
        # experts (no expert-parallel collectives at TP = 1).
        moe.select_token_dispatcher()
        config = moe.config
        rows = hidden.shape[0] * hidden.shape[1]
        probs = torch.zeros(
            (rows, config.num_moe_experts),
            dtype=signature.probs_dtype,
            device=hidden.device,
            requires_grad=signature.probs_requires_grad,
        )
        if signature.routing_dtype in (None, torch.bool):
            routing_map = torch.zeros(
                (rows, config.num_moe_experts), dtype=torch.bool, device=hidden.device
            )
        else:
            # Dense route indices: -1 marks no route (as padding rows get).
            routing_map = torch.full(
                (rows, config.moe_router_topk),
                -1,
                dtype=signature.routing_dtype,
                device=hidden.device,
            )
        hidden, probs = moe.preprocess(hidden, probs, routing_map, padding_mask=None)
        dispatched, probs = moe.dispatch(hidden, probs)
        output, _ = moe.routed_experts_compute(dispatched, probs)
        output = moe.combine(output)
        return moe.postprocess(output, shared_expert_output=None)

    def over_budget(self, module):
        check = getattr(module.token_dispatcher, "check_over_budget", None)
        return None if check is None else check()

    def reset_over_budget(self, module):
        reset = getattr(module.token_dispatcher, "reset_over_budget", None)
        if reset is not None:
            reset()


# (class, adapter), most recent registration first; MoELayer is appended lazily.
_REGISTRY: list = []
_MOE_REGISTERED = False


def register_lockstep_module(cls: type, adapter: LockstepAdapter) -> Callable[[], None]:
    """Treat instances of ``cls`` as lockstep modules driven by ``adapter`` (before the built-in
    ``MoELayer`` adapter); returns a function that removes the registration (tests)."""
    entry = (cls, adapter)
    _REGISTRY.insert(0, entry)

    def unregister():
        if entry in _REGISTRY:
            _REGISTRY.remove(entry)

    return unregister


def _adapter_for(module: nn.Module) -> Optional[LockstepAdapter]:
    global _MOE_REGISTERED  # pylint: disable=global-statement
    if not _MOE_REGISTERED:
        _MOE_REGISTERED = True
        try:
            from megatron.core.transformer.moe.moe_layer import MoELayer
        except ImportError:  # pragma: no cover - MoE dependencies missing
            pass
        else:
            _REGISTRY.append((MoELayer, MoELockstepAdapter()))
    for cls, adapter in _REGISTRY:
        if isinstance(module, cls):
            return adapter
    return None


def _spans_ranks(groups: Sequence) -> bool:
    if not dist.is_initialized():
        return False
    return any(group is not None and dist.get_world_size(group) > 1 for group in groups)


@dataclass(frozen=True, eq=False)
class LockstepEntry:
    """One MoE call of a sub-round type: the module, the hosting sublayer's config and
    ``layer_number - 1`` (the arguments of its inner quantization context), the null form,
    a name for messages, and ``wrap``, the extra quantization context a real call of an MTP
    depth runs in (``MultiTokenPredictionLayer.get_inner_quantization_context``)."""

    module: nn.Module
    adapter: LockstepAdapter
    layer_config: Any
    layer_number: int
    form: str
    name: str
    wrap: Optional[Callable[[], ContextManager]] = None


@dataclass
class LockstepPlan:
    """The lockstep MoE calls of each sub-round type, in forward call order, and the null
    signatures (see the module docstring). ``budget`` lists every registered module of the
    model (any group size) with a name, for the static-budget overflow flags."""

    by_type: dict
    signature: dict = field(default_factory=dict)
    budget: tuple = ()
    _names: dict = field(default_factory=dict)
    _calls: Optional[list] = None
    _check_every: bool = False

    @property
    def active(self) -> bool:
        """True when some sub-round type has lockstep MoE calls."""
        return any(self.by_type.values())

    def entries(self) -> list:
        """Every lockstep module's first entry, in plan order (prelude, core, coda, MTP)."""
        seen, unique = set(), []
        for kind_type in (SubRoundType.PRELUDE_F, SubRoundType.CORE_F, SubRoundType.CODA):
            for entry in self.by_type[kind_type]:
                if id(entry.module) not in seen:
                    seen.add(id(entry.module))
                    unique.append(entry)
        return unique

    # -- static budget -------------------------------------------------------------------

    def reset_over_budget(self) -> None:
        """Clear every module's overflow flag (the start of a step)."""
        for _, module, adapter in self.budget:
            adapter.reset_over_budget(module)

    def over_budget_flags(self) -> tuple:
        """``(names, int32 flags)`` of the modules that track a static-budget overflow (device
        tensor, no sync), or ``([], None)``."""
        names, flags = [], []
        for name, module, adapter in self.budget:
            flag = adapter.over_budget(module)
            if flag is not None:
                names.append(name)
                flags.append(flag.reshape(-1)[:1].to(torch.int32))
        if not flags:
            return [], None
        return names, torch.cat(flags)

    # -- real-call observation -----------------------------------------------------------

    @contextlib.contextmanager
    def observing(self, record_calls: bool = False):
        """Hooks on the lockstep modules for one step: the first real call of each module
        confirms its null signature (a mismatch raises: a null would issue mismatched
        collectives); with ``record_calls`` (``check_lockstep``) every real call is compared
        and logged for ``begin_item`` / ``end_item``.

        A mismatch raises on the rank of the real call, in the router hook, before that call's
        dispatch. It is a misconfiguration (the config rule of ``config_signature`` does not
        describe the real calls), and it is not raised on every rank at once: when the
        module's first real call on one EU meets a null of the same sub-round on another EU
        (typically a CODA sub-round: an EU's first CODA can be a null whenever its samples exit
        later than other EUs'), the null rank is already inside the dispatch collective and
        waits there until the launcher stops the job after the raising rank exits (torchrun
        stops every worker when one fails) or the collective times out. Refusing such nulls
        instead would reject valid runs (most sampled-depth steps have an EU whose first CODA is
        a null), so the error is reported by the real rank only."""
        handles = []
        self._check_every = record_calls
        try:
            for entry in self.entries():
                handles += entry.adapter.observe(
                    entry.module, self._on_call if record_calls else _ignore, self._on_signature
                )
            yield self
        finally:
            for handle in handles:
                handle.remove()
            self._calls = None
            self._check_every = False

    def _on_call(self, module: nn.Module) -> None:
        if self._calls is not None:
            self._calls.append(module)

    def _on_signature(self, module: nn.Module, observed: LockstepSignature) -> None:
        signature = self.signature[module]
        if signature.observed and not self._check_every:
            return
        if observed.key() != signature.key():
            raise RuntimeError(
                f"{LOCKSTEP}: a real call of {self._names[id(module)]} has the "
                f"signature {observed.key()} but its null items use {signature.key()} (hidden "
                "dtype, hidden requires_grad, probs dtype, probs requires_grad, routing dtype); "
                "the nulls would issue mismatched expert-parallel collectives"
            )
        signature.observed = True

    def begin_item(self) -> None:
        """Start logging the real MoE calls of one item (``check_lockstep``)."""
        self._calls = []

    def end_item(self, kind_type: SubRoundType, label: str) -> None:
        """Compare the logged real calls with the plan's calls of ``kind_type``."""
        calls, self._calls = self._calls or [], None
        expected = [entry.module for entry in self.by_type[kind_type]]
        if len(calls) != len(expected) or any(a is not b for a, b in zip(calls, expected)):

            def names(modules):
                return [self._names.get(id(m), type(m).__name__) for m in modules]

            raise RuntimeError(
                f"{LOCKSTEP}: the real {kind_type.value} item {label} called the MoE modules "
                f"{names(calls)} but the lockstep plan lists {names(expected)} (its nulls would "
                "issue other collectives)"
            )


def _ignore(module: nn.Module) -> None:
    del module


def _scan(stack, start: int, end: int, prefix: str, wrap, budget: list, seen: set) -> list:
    """The lockstep entries of ``stack.layers[start:end]`` in order (one module per sublayer
    at most); every registered module is also added to ``budget`` once."""
    entries = []
    for index in range(start, end):
        layer = stack.layers[index]
        hits = []
        for name, module in layer.named_modules():
            adapter = _adapter_for(module)
            if adapter is not None:
                hits.append((name, module, adapter))
        if not hits:
            continue
        if len(hits) > 1:
            raise ValueError(
                UNSUPPORTED + f"more than one MoE module in sublayer {prefix}.layers.{index}: "
                f"{[name for name, _, _ in hits]}"
            )
        name, module, adapter = hits[0]
        full_name = f"{prefix}.layers.{index}" + (f".{name}" if name else "")
        if id(module) not in seen:
            seen.add(id(module))
            budget.append((full_name, module, adapter))
        if not _spans_ranks(adapter.groups(module)):
            continue
        entries.append(
            LockstepEntry(
                module=module,
                adapter=adapter,
                layer_config=stack._execution_layer_config_list[index],
                layer_number=layer.layer_number - 1,
                form=adapter.form(module),
                name=full_name,
                wrap=wrap,
            )
        )
    return entries


def build_lockstep_plan(program) -> LockstepPlan:
    """The lockstep MoE calls of each sub-round type of ``program``'s model.

    PRELUDE-F / PRELUDE-B: the MoE modules of the stack sublayers ``[0, P)`` (hash-routed
    layers included); CORE-F / CORE-B: ``[P, P + K)``; CODA: ``[P + K, L)``, then for each MTP
    depth the modules of that depth's stack (a repeated MTP layer once per depth). Raises
    ``ValueError`` when a sublayer holds more than one MoE module, when a lockstep module has
    no null form, when the forms differ between modules, or when a lockstep module of the model
    is outside these layers (its calls would not be replayed by any null).
    """
    hybrid, stack, roles = program.hybrid, program.stack, program.roles
    budget: list = []
    seen: set = set()
    prelude = _scan(stack, 0, roles.core_start, "decoder", None, budget, seen)
    core = _scan(stack, roles.core_start, roles.core_end, "decoder", None, budget, seen)
    coda = _scan(stack, roles.core_end, len(stack.layers), "decoder", None, budget, seen)
    mtp = getattr(hybrid, "mtp", None)
    depths = (getattr(program.config, "mtp_num_layers", None) or 0) if mtp is not None else 0
    for depth in range(depths):
        index = 0 if getattr(mtp, "mtp_use_repeated_layer", False) else depth
        layer = mtp.layers[index]
        inner = getattr(layer, "mtp_model_layer", None)
        if inner is None or not hasattr(inner, "_execution_layer_config_list"):
            raise ValueError(
                UNSUPPORTED + f"lockstep discovery of MTP layer {index} without a hybrid stack"
            )
        wrap = getattr(layer, "get_inner_quantization_context", None)
        coda = coda + _scan(
            inner, 0, len(inner.layers), f"mtp.layers.{index}.mtp_model_layer", wrap, budget, seen
        )
    by_type = {
        SubRoundType.PRELUDE_F: tuple(prelude),
        SubRoundType.PRELUDE_B: tuple(prelude),
        SubRoundType.CORE_F: tuple(core),
        SubRoundType.CORE_B: tuple(core),
        SubRoundType.CODA: tuple(coda),
    }
    plan = LockstepPlan(by_type=by_type, budget=tuple(budget))
    entries = plan.entries()
    covered = {id(entry.module) for entry in entries}
    missing = []
    for name, module in hybrid.named_modules():
        adapter = _adapter_for(module)
        if adapter is not None and id(module) not in covered:
            if _spans_ranks(adapter.groups(module)):
                missing.append(name)
    if missing:
        raise ValueError(
            UNSUPPORTED + f"expert-parallel MoE modules outside the loop's prelude / core / coda "
            f"and MTP layers: {missing}"
        )
    forms = {entry.form for entry in entries}
    if len(forms) > 1:
        raise ValueError(UNSUPPORTED + f"lockstep MoE modules with different null forms {forms}")
    for entry in entries:
        plan.signature[entry.module] = entry.adapter.config_signature(entry.module)
        plan._names[id(entry.module)] = entry.name  # pylint: disable=protected-access
    for name, module, _ in budget:
        plan._names.setdefault(id(module), name)  # pylint: disable=protected-access
    return plan


def run_null(
    plan: LockstepPlan,
    kind_type: SubRoundType,
    *,
    outer_context: ContextManager,
    inner_context: Callable[[Any, int], ContextManager],
    sequence: int,
    device: torch.device,
) -> int:
    """Run the null item of ``kind_type``: every lockstep MoE call of the type on an empty
    input (rows per the entry's form, ``sequence`` x 1 for "nominal"), chained, then for the
    backward types a backward of zeros through the chain. ``outer_context`` and
    ``inner_context(layer_config, layer_number)`` are the stack's quantization contexts (as
    the cells and ``HybridStack._run_layer`` enter them). Returns the number of MoE calls."""
    entries = plan.by_type[kind_type]
    if not entries:
        return 0
    output = None
    with torch.enable_grad(), outer_context:
        for entry in entries:
            signature = plan.signature[entry.module]
            if output is None:
                rows = 0 if entry.form == FORM_EMPTY else sequence
                hidden = torch.zeros(
                    (rows, 1, entry.layer_config.hidden_size),
                    dtype=signature.hidden_dtype,
                    device=device,
                    requires_grad=signature.hidden_requires_grad,
                )
            else:
                # Chained: the backward visits the modules in reverse, as a real item's does.
                hidden = output.to(signature.hidden_dtype)
            wrap = entry.wrap() if entry.wrap is not None else contextlib.nullcontext()
            with wrap, inner_context(entry.layer_config, entry.layer_number):
                output = entry.adapter.null_forward(entry, hidden, signature)
        if kind_type in _BACKWARD_TYPES and output.requires_grad:
            torch.autograd.backward(output, torch.zeros_like(output))
    return len(entries)
