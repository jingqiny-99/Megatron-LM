# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Weight binding of the loop parameter store: how a stored weight reaches the linears.

The loop parameter store (``loop/param_store.py``) keeps the canonical shards of the dense
parameters on the shard axis and gives each layer full weights only around its use (Layer window)
or keeps one full compute copy (Resident, Resident + sharded grads). This module is the layer
between those buffers and the modules that consume the weights. It owns no buffer and runs no
collective: the store gathers a unit's weights into a slot (or a compute copy), then *binds* every
tensor of the unit to its region, and binds an fp32 *gradient target* (a scratch entry, a
transient, or the Resident accumulator) before the unit's backward.

Two bindings exist:

``"data"`` (the D31 fallback, the default; works on every TE version)
    The compute ``Parameter`` keeps its identity and attributes but leaves module registration
    (``hide_parameter``: popped from ``_parameters`` into the instance ``__dict__``, so
    ``module.weight`` and TE's ``getattr(self, name)`` still return it while DDP, the optimizer
    and ``state_dict`` never see it). ``bind_weight`` rebinds its ``.data`` onto the region;
    ``unbind_weight`` rebinds it onto a stride-0 NaN *placeholder* of the logical shape (shape
    readers stay correct, a stray read yields NaN, an in-place write raises).

``"distributed_weight"`` (TE >= 2.19, behind ``te_distributed_weight_available``)
    The module's weight becomes an ``RSBFDistributedWeight``: a shard-shaped tensor subclass
    implementing TE's ``DistributedWeight`` protocol. TE asks it for the full weight in the
    forward and the backward (``materialize_group_for_*``: the bound region; no collective, the
    store has already gathered) and hands it the wgrad (``finalize_group_grads``: written into or
    added to the gradient target). The stand-in itself holds no data (a shard-shaped placeholder).

Gradients reach the target by one of three capture classes (``GradCapture``):

``"lazy"``     TE linear-family weights with fused wgrad: TE reads ``weight.main_grad`` *in the
               backward*, so the target may be bound after the forward (and after a recompute).
``"forward"``  Megatron's ``LinearWithGradAccumulationAndAsyncCommunication`` with
               ``gradient_accumulation_fusion``: it captures ``weight.main_grad`` *in the forward*
               and always accumulates, so the target is bound at the forward visit and zeroed.
``"hook"``     everything else (``nn.Linear``, raw parameters, embedding lookups, CPU): autograd
               produces ``p.grad`` and a post-accumulate hook adds it into the target and drops it.
               The hook is DDP's arithmetic: ``p.grad`` is added unless a fused consumer flagged
               ``grad_added_to_main_grad``, and then still added when the weight has
               ``zero_out_wgrad`` (a tied weight whose fused consumer returns a zero dummy and
               whose lookup's gradient autograd sums with it).

The class is decided from the owner module (``classify_grad_capture``); a TE-owned weight read
raw by another consumer (absorbed MLA's ``linear_kv_up_proj.weight.view(...)``) must be
reclassified by the store's discovery as ``hook`` / multi-use. ``overwrite_main_grad`` (TE writes
the wgrad GEMM without accumulating) is allowed only for ``lazy`` tensors used once per segment;
every other target is zeroed when bound. Overwrite targets are checked after the segment's
backward (``check_segment_grads``): consumed more than once, or never through the owner's
forward, raises; a hook-path gradient raises at once; a target no gradient reached is zeroed
(what a Resident accumulator holds then). A target's pad tail (the rows that pad ``R`` to a
multiple of the shard size) is always zeroed when bound, so a reduce-scatter of the padded region
ships zeros into the shard's pad rows.

Saved tensors and re-gathers. ``p.data = x`` re-points the same ``TensorImpl``, so an op that
saved the leaf parameter reads whatever ``p`` is bound to in the backward. ``F.linear`` saves
``weight.t()`` and einsum on ``weight.view(...)`` saves the view: both alias the *forward-time*
storage. A unit re-gathered for its backward must therefore land in the same slot at the same
offsets (the store's slot rule); the distributed-weight binding checks it
(``materialize_group_for_backward`` raises on a moved region).

Environment facts (stage-6 design, Section 0): the EOS CI container ``mcore-ci-dev`` runs TE
2.16 without ``transformer_engine.pytorch.distributed_weight``, so ``"data"`` is the binding that
runs on GPU today; ``probe_store_environment`` records the facts at runtime.
"""

import importlib.util
import math
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple

import torch
from torch import nn

GradCapture = Literal["lazy", "forward", "hook"]
Binding = Literal["data", "distributed_weight"]

BINDINGS: Tuple[str, ...] = ("data", "distributed_weight")
GRAD_CAPTURES: Tuple[str, ...] = ("lazy", "forward", "hook")

TE_DISTRIBUTED_WEIGHT_MIN_VERSION = "2.19.0.dev0"
_TE_DISTRIBUTED_WEIGHT_MODULE = "transformer_engine.pytorch.distributed_weight"

# Instance attribute holding a module's hidden compute parameters ({attr: Parameter}).
HIDDEN_PARAMS_ATTR = "_loop_hidden_params"
# Set (True) on the store's mirror modules; ``named_logical_parameters`` skips their subtrees.
MIRROR_FLAG_ATTR = "_loop_store_mirror"


# ---------------------------------------------------------------------------------------------
# Environment probe
# ---------------------------------------------------------------------------------------------


_TE_IMPORT_ERROR: Dict[str, str] = {}


def te_version() -> Optional[str]:
    """The installed Transformer Engine version, or None when TE cannot be imported.

    Any import failure counts (a CUDA / library mismatch raises ``RuntimeError``, not
    ``ImportError``); its message is kept for ``distributed_weight_unavailable_reason``.
    """
    try:
        import transformer_engine  # pylint: disable=import-outside-toplevel,unused-import

        from megatron.core.utils import get_te_version  # pylint: disable=import-outside-toplevel

        version = str(get_te_version())
    except ModuleNotFoundError as exc:
        if exc.name != "transformer_engine":
            _TE_IMPORT_ERROR["error"] = f"{type(exc).__name__}: {exc}"
        else:
            _TE_IMPORT_ERROR.pop("error", None)
        return None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _TE_IMPORT_ERROR["error"] = f"{type(exc).__name__}: {exc}"
        return None
    _TE_IMPORT_ERROR.pop("error", None)
    return version


def te_distributed_weight_available() -> bool:
    """True when TE (>= 2.19) ships the ``DistributedWeight`` protocol module."""
    version = te_version()
    if version is None:
        return False
    from packaging.version import Version  # pylint: disable=import-outside-toplevel

    if Version(version) < Version(TE_DISTRIBUTED_WEIGHT_MIN_VERSION):
        return False
    try:
        return importlib.util.find_spec(_TE_DISTRIBUTED_WEIGHT_MODULE) is not None
    except (ImportError, ValueError):
        return False


def distributed_weight_unavailable_reason() -> Optional[str]:
    """Why ``binding="distributed_weight"`` cannot run here, or None when it can."""
    if te_distributed_weight_available():
        return None
    version = te_version()
    needs = "needs TE >= 2.19, e.g. the uv.lock pin 2.20.2"
    if version is None:
        error = _TE_IMPORT_ERROR.get("error")
        state = "is not installed" if error is None else f"cannot be imported ({error})"
        return (
            f"Transformer Engine {state}: no DistributedWeight protocol ({needs}); "
            "use binding='data'"
        )
    return f"TE {version} has no DistributedWeight protocol ({needs}); use binding='data'"


def _te_support(candidates: Sequence[Tuple[str, str]]) -> Tuple[bool, str]:
    """First importable TE support check of ``candidates`` ((module, function) pairs)."""
    for module_name, fn_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        fn = getattr(module, fn_name, None)
        if fn is None:
            continue
        try:
            result = fn(return_reason=True) if fn_name.startswith("is_") else fn()
        except TypeError:
            continue  # another signature in this TE version: try the next candidate
        except (RuntimeError, AssertionError) as exc:
            return False, f"{module_name}.{fn_name} raised: {exc}"
        if isinstance(result, tuple):
            return bool(result[0]), str(result[1] or "")
        return bool(result), ""
    return False, "no TE support check importable (TE missing or too old)"


@dataclass(frozen=True)
class StoreEnvironment:
    """The runtime facts the store's binding depends on (recorded in the stage report)."""

    torch_version: str
    te_version: Optional[str]
    distributed_weight: bool
    distributed_weight_reason: Optional[str]
    cuda: bool
    device_capability: Optional[Tuple[int, int]]
    fp8_block_scaling: bool
    fp8_block_scaling_reason: str
    mxfp8: bool
    mxfp8_reason: str
    local_wgrad_fusion: bool  # Megatron's fused_weight_gradient_mlp_cuda is importable

    @property
    def default_binding(self) -> str:
        """``"data"`` everywhere: the protocol binding is opt-in (design decision 1)."""
        return "data"

    def summary(self) -> str:
        """One line per fact, for logs and the stage report."""
        dw = "yes" if self.distributed_weight else f"no ({self.distributed_weight_reason})"
        cap = "-" if self.device_capability is None else "sm_%d%d" % self.device_capability
        fp8 = "yes" if self.fp8_block_scaling else f"no ({self.fp8_block_scaling_reason})"
        mx = "yes" if self.mxfp8 else f"no ({self.mxfp8_reason})"
        return "\n".join(
            [
                f"torch {self.torch_version}",
                f"transformer_engine {self.te_version or 'not installed'}",
                f"DistributedWeight protocol: {dw}",
                f"CUDA: {self.cuda} ({cap})",
                f"FP8 block scaling: {fp8}",
                f"MXFP8: {mx}",
                f"Megatron local wgrad fusion kernel: {self.local_wgrad_fusion}",
                f"store binding default: {self.default_binding}",
            ]
        )


def probe_store_environment() -> StoreEnvironment:
    """Probe TE, the protocol module, the quantization recipes and the local fusion kernel."""
    cuda = torch.cuda.is_available()
    capability = torch.cuda.get_device_capability() if cuda else None
    if cuda:
        fp8_ok, fp8_reason = _te_support(
            [
                ("transformer_engine.pytorch", "is_fp8_block_scaling_available"),
                ("transformer_engine.pytorch.quantization", "check_fp8_block_scaling_support"),
                ("transformer_engine.pytorch.fp8", "check_fp8_block_scaling_support"),
            ]
        )
        mx_ok, mx_reason = _te_support(
            [
                ("transformer_engine.pytorch", "is_mxfp8_available"),
                ("transformer_engine.pytorch.quantization", "check_mxfp8_support"),
                ("transformer_engine.pytorch.fp8", "check_mxfp8_support"),
            ]
        )
    else:
        fp8_ok, fp8_reason = False, "no CUDA device"
        mx_ok, mx_reason = False, "no CUDA device"
    from megatron.core.tensor_parallel import layers  # pylint: disable=import-outside-toplevel

    return StoreEnvironment(
        torch_version=str(torch.__version__),
        te_version=te_version(),
        distributed_weight=te_distributed_weight_available(),
        distributed_weight_reason=distributed_weight_unavailable_reason(),
        cuda=cuda,
        device_capability=tuple(capability) if capability is not None else None,
        fp8_block_scaling=fp8_ok,
        fp8_block_scaling_reason=fp8_reason,
        mxfp8=mx_ok,
        mxfp8_reason=mx_reason,
        local_wgrad_fusion=bool(getattr(layers, "_grad_accum_fusion_available", False)),
    )


# ---------------------------------------------------------------------------------------------
# Placeholders, hiding, logical parameter names
# ---------------------------------------------------------------------------------------------

_PLACEHOLDER_BASES: Dict[Tuple[torch.dtype, torch.device], torch.Tensor] = {}


def _device_key(device) -> torch.device:
    """``device`` with an explicit index for CUDA (``"cuda"`` and ``"cuda:<current>"`` agree)."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def logical_placeholder(
    shape: Sequence[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """A stride-0 view of one NaN element with ``shape``: the ``.data`` of an unbound weight.

    One element per (dtype, device) is allocated for the whole process. Reads see NaN, an in-place
    write raises ("more than one element of the written-to tensor refers to a single memory
    location"), and ``shape`` / ``numel`` stay logical.
    """
    if not dtype.is_floating_point:
        raise ValueError(f"a store placeholder needs a floating dtype, got {dtype}")
    key = (dtype, _device_key(device))
    base = _PLACEHOLDER_BASES.get(key)
    if base is None:
        base = torch.full((1,), float("nan"), dtype=dtype, device=key[1])
        _PLACEHOLDER_BASES[key] = base
    return base.as_strided(tuple(shape), (0,) * len(shape))


def is_placeholder(tensor: torch.Tensor) -> bool:
    """True when ``tensor`` is (a view of) a store placeholder."""
    base = _PLACEHOLDER_BASES.get((tensor.dtype, _device_key(tensor.device)))
    return base is not None and tensor.numel() > 0 and tensor.data_ptr() == base.data_ptr()


def _hidden(module: nn.Module) -> Dict[str, torch.Tensor]:
    hidden = module.__dict__.get(HIDDEN_PARAMS_ATTR)
    if hidden is None:
        hidden = {}
        object.__setattr__(module, HIDDEN_PARAMS_ATTR, hidden)
    return hidden


def hidden_parameters(module: nn.Module) -> Dict[str, torch.Tensor]:
    """The hidden compute parameters of ``module`` itself (not its children), by attribute."""
    return dict(module.__dict__.get(HIDDEN_PARAMS_ATTR) or {})


def hide_parameter(module: nn.Module, attr: str) -> nn.Parameter:
    """Move ``module.<attr>`` out of registration, keeping ``module.<attr>`` readable.

    The ``Parameter`` object (identity, ``requires_grad``, Megatron attributes) is kept in the
    instance ``__dict__``, which attribute lookup reads before ``nn.Module.__getattr__``; it is no
    longer in ``_parameters``, so ``parameters()``, ``state_dict()``, DDP and the optimizer skip
    it. ``named_logical_parameters`` still lists it under its original name.
    """
    param = module._parameters.get(attr)  # pylint: disable=protected-access
    if param is None:
        raise ValueError(f"{type(module).__name__} has no registered parameter {attr!r} to hide")
    del module._parameters[attr]  # pylint: disable=protected-access
    object.__setattr__(module, attr, param)
    _hidden(module)[attr] = param
    return param


def set_hidden_weight(module: nn.Module, attr: str, weight: torch.Tensor) -> None:
    """Make ``module.<attr>`` return ``weight`` (a hidden attribute's consumer-facing object)."""
    if attr not in _hidden(module):
        raise ValueError(f"{type(module).__name__}.{attr} is not hidden")
    object.__setattr__(module, attr, weight)


def restore_parameter(module: nn.Module, attr: str) -> nn.Parameter:
    """Undo ``hide_parameter``: re-register the hidden ``Parameter`` (tests, uninstall)."""
    hidden = _hidden(module)
    if attr not in hidden:
        raise ValueError(f"{type(module).__name__}.{attr} is not hidden")
    param = hidden.pop(attr)
    module.__dict__.pop(attr, None)
    module._parameters[attr] = param  # pylint: disable=protected-access
    return param


def register_logical(tensor: "StoreTensor") -> None:
    """Put ``tensor``'s compute ``Parameter`` back into its owner's ``_parameters`` (the
    checkpoint's *logical view*, ``param_store_checkpoint.logical_view``): the owner's
    ``sharded_state_dict`` / ``_save_to_state_dict`` then emit its entry under its original name.
    The instance ``__dict__`` entry stays, so consumers still read ``tensor.weight``. Raises when
    the slot holds another object."""
    params = tensor.module._parameters  # pylint: disable=protected-access
    current = params.get(tensor.attr)
    if current is not None and current is not tensor.compute:
        raise RuntimeError(
            f"{tensor.name}: {type(tensor.module).__name__}.{tensor.attr} is registered as "
            "another object; the logical view cannot register the stored weight"
        )
    params[tensor.attr] = tensor.compute


def unregister_logical(tensor: "StoreTensor") -> None:
    """Undo ``register_logical`` (only when the slot still holds ``tensor.compute``)."""
    params = tensor.module._parameters  # pylint: disable=protected-access
    if params.get(tensor.attr) is tensor.compute:
        del params[tensor.attr]


def named_logical_parameters(
    module: nn.Module, prefix: str = "", recurse: bool = True
) -> Iterator[Tuple[str, torch.Tensor]]:
    """``module.named_parameters()`` plus the hidden compute parameters, by their original names.

    The store's mirror modules (canonical shards) are skipped. Each module yields its registered
    parameters first, then its hidden ones; a parameter or module reachable twice is listed once.
    """
    seen_params = set()
    seen_modules = set()

    def visit(mod: nn.Module, mod_prefix: str) -> Iterator[Tuple[str, torch.Tensor]]:
        if id(mod) in seen_modules or getattr(mod, MIRROR_FLAG_ATTR, False):
            return
        seen_modules.add(id(mod))
        own = list(mod._parameters.items())  # pylint: disable=protected-access
        own += list((mod.__dict__.get(HIDDEN_PARAMS_ATTR) or {}).items())
        for name, param in own:
            if param is None or id(param) in seen_params:
                continue
            seen_params.add(id(param))
            yield (f"{mod_prefix}.{name}" if mod_prefix else name), param
        if not recurse:
            return
        for child_name, child in mod._modules.items():  # pylint: disable=protected-access
            if child is None:
                continue
            yield from visit(child, f"{mod_prefix}.{child_name}" if mod_prefix else child_name)

    yield from visit(module, prefix)


# ---------------------------------------------------------------------------------------------
# Consumers: TE linears, Megatron local fused linears
# ---------------------------------------------------------------------------------------------


def _te_base_module_class():
    # pylint: disable=import-outside-toplevel
    try:
        from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    return TransformerEngineBaseModule


def is_te_linear(module: nn.Module) -> bool:
    """A TE linear-family module (``Linear``, ``LayerNormLinear``, ... and Megatron's wrappers)."""
    base = _te_base_module_class()
    return (
        base is not None and isinstance(module, base) and hasattr(module, "fuse_wgrad_accumulation")
    )


def _local_fused_linear_classes() -> Tuple[type, ...]:
    from megatron.core.tensor_parallel.layers import (  # pylint: disable=import-outside-toplevel
        ColumnParallelLinear,
        RowParallelLinear,
    )

    return (ColumnParallelLinear, RowParallelLinear)


def is_local_fused_linear(module: nn.Module, attr: str = "weight") -> bool:
    """A Megatron local linear whose fused wgrad captures ``main_grad`` at forward (E15)."""
    return (
        attr == "weight"
        and isinstance(module, _local_fused_linear_classes())
        and bool(getattr(module, "gradient_accumulation_fusion", False))
    )


def classify_grad_capture(
    module: nn.Module, attr: str, te_linear: Optional[bool] = None
) -> GradCapture:
    """Where the wgrad of ``module.<attr>`` lands when the module consumes it directly.

    Decided from the owner only. A weight that another consumer also reads raw (absorbed MLA
    reads ``linear_kv_up_proj.weight.view(...)`` / ``linear_q_up_proj.weight``) gets its gradient
    through autograd, not TE: the store's discovery must pass ``grad_capture="hook"`` (and
    ``single_use=False``) for it. Left ``lazy`` under ``overwrite_main_grad``, the hook raises at
    the raw consumer's gradient and ``check_segment_grads`` raises for a raw-only use.
    ``te_linear`` overrides the TE class check (a test double of a TE linear).
    """
    if is_te_linear(module) if te_linear is None else te_linear:
        return "lazy" if getattr(module, "fuse_wgrad_accumulation", False) else "hook"
    if is_local_fused_linear(module, attr):
        return "forward"
    return "hook"


def disable_local_wgrad_fusion(module: nn.Module) -> bool:
    """Switch a Megatron local fused linear to the hook path; True when it was fused.

    Its forward-captured target (E15) would otherwise be pinned from the forward to the backward
    of every unit kept in between. Only test-time local specs are affected (DSv4 uses TE).
    """
    if is_local_fused_linear(module):
        module.gradient_accumulation_fusion = False
        return True
    return False


# ---------------------------------------------------------------------------------------------
# Layout helpers (rows padded to the shard size, aligned regions)
# ---------------------------------------------------------------------------------------------


def padded_rows(rows: int, shard_size: int) -> int:
    """``rows`` rounded up to a multiple of ``shard_size`` (GTP's dim-0 convention)."""
    if shard_size < 1:
        raise ValueError(f"shard_size must be >= 1, got {shard_size}")
    return -(-rows // shard_size) * shard_size


def shard_slice(full: torch.Tensor, shard_size: int, rank: int) -> torch.Tensor:
    """Rank ``rank``'s dim-0 shard of ``full`` padded with zero rows (a copy)."""
    rows = full.shape[0]
    rows_pad = padded_rows(rows, shard_size)
    if rows_pad != rows:
        pad = full.new_zeros((rows_pad - rows, *full.shape[1:]))
        full = torch.cat((full, pad), dim=0)
    per = rows_pad // shard_size
    return full[rank * per : (rank + 1) * per].clone()


def align_elements(offset: int, dtype: torch.dtype, align_bytes: int) -> int:
    """``offset`` (elements of ``dtype``) rounded up to an ``align_bytes``-aligned start."""
    elem = torch.empty((), dtype=dtype).element_size()
    if align_bytes % elem:
        raise ValueError(f"align_bytes={align_bytes} is not a multiple of {dtype}'s size {elem}")
    step = max(1, align_bytes // elem)
    return -(-offset // step) * step


# ---------------------------------------------------------------------------------------------
# Store tensors
# ---------------------------------------------------------------------------------------------


@dataclass(eq=False)
class StoreTensor:
    """One stored weight: the compute parameter, its canonical shard and its binding state.

    Static fields describe the weight; the runtime fields (``bound``, ``grad_target``,
    ``grad_written``, ``overwrite``, ``tracked``, ``uses``, ``forward_ptr``) are owned by the
    binding functions below.
    """

    name: str  # logical qualified name, e.g. "decoder.layers.3.self_attention.linear_qkv.weight"
    module: nn.Module  # owner; ``module.<attr>`` returns ``weight``
    attr: str
    compute: nn.Parameter  # the original Parameter object (logical shape)
    logical_shape: Tuple[int, ...]  # (R, *rest)
    rows_padded: int  # ceil(R / z) * z
    shard_size: int = 1
    grad_capture: GradCapture = "hook"
    single_use: bool = True
    te_linear: bool = False
    fused_wgrad: bool = False
    binding: Binding = "data"
    offset: int = 0  # element offset of this tensor's region in its unit's slot / scratch layout
    shard: Optional[nn.Parameter] = None  # canonical shard [R_pad / z, *rest] (set by the store)
    weight: Optional[torch.Tensor] = None  # what ``module.<attr>`` returns
    placeholder: Optional[torch.Tensor] = None
    # Runtime state.
    bound: Optional[torch.Tensor] = None
    grad_target: Optional[torch.Tensor] = None
    grad_written: bool = False  # a gradient reached grad_target since it was bound
    overwrite: bool = False
    tracked: bool = False  # track_uses installed the use counter
    uses: int = 0
    forward_ptr: Optional[int] = None
    handles: List[Any] = field(default_factory=list)

    @property
    def rows(self) -> int:
        """R, the logical dim-0 size."""
        return self.logical_shape[0]

    @property
    def row_elems(self) -> int:
        """Elements per dim-0 row."""
        return math.prod(self.logical_shape[1:])

    @property
    def numel(self) -> int:
        """Logical element count (R * row_elems)."""
        return self.rows * self.row_elems

    @property
    def numel_padded(self) -> int:
        """Element count with the pad rows (R_pad * row_elems)."""
        return self.rows_padded * self.row_elems

    @property
    def pad_length(self) -> int:
        """Pad rows (GTP's ``pad_length``)."""
        return self.rows_padded - self.rows

    @property
    def shard_shape(self) -> Tuple[int, ...]:
        """Shape of the canonical shard."""
        return (self.rows_padded // self.shard_size, *self.logical_shape[1:])

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the weight (and of every slot / compute copy holding it)."""
        return self.compute.dtype

    @property
    def is_bound(self) -> bool:
        """True while a weight region is bound."""
        return self.bound is not None


def make_store_tensor(
    name: str,
    module: nn.Module,
    attr: str,
    *,
    shard_size: int,
    binding: Binding = "data",
    grad_capture: Optional[GradCapture] = None,
    single_use: bool = True,
    te_linear: Optional[bool] = None,
    shard: Optional[nn.Parameter] = None,
) -> StoreTensor:
    """Take ``module.<attr>`` under the store: hide it, install the binding and the grad hook.

    ``binding="data"``: the compute ``Parameter`` stays the consumer-facing object; its ``.data``
    is left as is until the first ``unbind_weight`` (the store slices the shards first).
    ``binding="distributed_weight"``: needs TE's protocol (``distributed_weight_unavailable_reason``
    is None) and a TE linear-family owner; the consumer-facing object is an
    ``RSBFDistributedWeight`` of the shard shape backed by a placeholder (no storage of its own,
    so the DDP wrap that later moves the canonical ``shard`` into its buffer is not coupled to it),
    and the compute parameter's ``.data`` is released onto the logical placeholder. ``shard``, when
    given, is checked against the shard shape.
    """
    if binding not in BINDINGS:
        raise ValueError(f"unknown store binding {binding!r} (one of {BINDINGS})")
    param = module._parameters.get(attr)  # pylint: disable=protected-access
    if param is None:
        raise ValueError(f"{name}: {type(module).__name__}.{attr} is not a registered parameter")
    if param.dim() < 1:
        raise ValueError(f"{name}: a stored weight needs at least one dimension")
    te_owned = is_te_linear(module) if te_linear is None else bool(te_linear)
    capture = (
        classify_grad_capture(module, attr, te_owned) if grad_capture is None else grad_capture
    )
    if capture not in GRAD_CAPTURES:
        raise ValueError(f"{name}: unknown grad capture {capture!r} (one of {GRAD_CAPTURES})")
    logical = tuple(param.shape)
    tensor = StoreTensor(
        name=name,
        module=module,
        attr=attr,
        compute=param,
        logical_shape=logical,
        rows_padded=padded_rows(logical[0], shard_size),
        shard_size=shard_size,
        grad_capture=capture,
        single_use=single_use,
        te_linear=te_owned,
        fused_wgrad=capture in ("lazy", "forward"),
        binding=binding,
        shard=shard,
    )
    if binding == "distributed_weight":
        reason = distributed_weight_unavailable_reason()
        if reason is not None:
            raise ValueError(reason)
        if not te_owned:
            raise ValueError(
                f"{name}: only TE Linear-family weights take binding='distributed_weight' "
                f"(owner {type(module).__name__}); use binding='data'"
            )
    if shard is not None and tuple(shard.shape) != tensor.shard_shape:
        raise ValueError(
            f"{name}: the canonical shard must be {tensor.shard_shape}, got {tuple(shard.shape)}"
        )
    tensor.placeholder = logical_placeholder(logical, param.dtype, param.device)
    hide_parameter(module, attr)
    param.grad_added_to_main_grad = False
    param.loop_store_tensor = tensor
    if binding == "distributed_weight":
        weight = RSBFDistributedWeight.create(tensor)
        set_hidden_weight(module, attr, weight)
        param.data = tensor.placeholder
    else:
        weight = param
    tensor.weight = weight
    tensor.handles.append(weight.register_post_accumulate_grad_hook(_grad_hook(tensor)))
    return tensor


def release_store_tensor(tensor: StoreTensor, data: Optional[torch.Tensor] = None) -> nn.Parameter:
    """Undo ``make_store_tensor``: remove the hooks, unbind, re-register the ``Parameter``.

    ``data`` (logical shape) becomes the parameter's ``.data``; by default the bound region (a
    view, the caller keeps its buffer alive). An unbound tensor needs ``data``: re-registering the
    NaN placeholder would hand DDP / the optimizer a parameter without values.
    """
    if data is None and tensor.bound is None:
        raise ValueError(
            f"{tensor.name}: releasing an unbound store tensor needs the full weight as ``data``"
        )
    if data is not None and tuple(data.shape) != tensor.logical_shape:
        raise ValueError(
            f"{tensor.name}: release data {tuple(data.shape)} != {tensor.logical_shape}"
        )
    for handle in tensor.handles:
        handle.remove()
    tensor.handles.clear()
    unbind_grad(tensor)
    set_overwrite(tensor, False)
    param = tensor.compute
    if data is None:
        data = tensor.bound
    tensor.bound = None
    tensor.forward_ptr = None
    restore_parameter(tensor.module, tensor.attr)
    param.data = data
    for attr in ("loop_store_tensor", "grad_added_to_main_grad", "overwrite_main_grad"):
        param.__dict__.pop(attr, None)
    tensor.weight = None
    return param


# ---------------------------------------------------------------------------------------------
# Binding weights and gradient targets
# ---------------------------------------------------------------------------------------------


def tensor_region(flat: torch.Tensor, tensor: StoreTensor, *, padded: bool = False) -> torch.Tensor:
    """``tensor``'s region of a flat unit buffer (slot, compute copy, scratch, accumulator)."""
    length = tensor.numel_padded if padded else tensor.numel
    if flat.dim() != 1 or flat.numel() < tensor.offset + length:
        raise ValueError(
            f"{tensor.name}: flat buffer of {flat.numel()} elements cannot hold "
            f"[{tensor.offset}, {tensor.offset + length})"
        )
    return flat.narrow(0, tensor.offset, length)


def tensor_view(flat: torch.Tensor, tensor: StoreTensor) -> torch.Tensor:
    """``tensor``'s logical-shape view of a flat unit buffer (rows ``[0, R)``)."""
    return tensor_region(flat, tensor).view(tensor.logical_shape)


def layout_tensors(tensors: Sequence[StoreTensor], align_bytes: int = 256) -> int:
    """Assign aligned, disjoint regions (padded rows included) in order; returns the unit numel."""
    offset = 0
    for tensor in tensors:
        offset = align_elements(offset, tensor.dtype, align_bytes)
        tensor.offset = offset
        offset += tensor.numel_padded
    if tensors:
        offset = align_elements(offset, tensors[-1].dtype, align_bytes)
    return offset


def bind_weight(tensor: StoreTensor, view: torch.Tensor) -> None:
    """Bind ``tensor`` to ``view`` (its logical-shape region of a slot or compute copy)."""
    if tuple(view.shape) != tensor.logical_shape:
        raise ValueError(f"{tensor.name}: bind view {tuple(view.shape)} != {tensor.logical_shape}")
    if view.dtype != tensor.dtype or view.device != tensor.compute.device:
        raise ValueError(
            f"{tensor.name}: bind view {view.dtype}/{view.device} != "
            f"{tensor.dtype}/{tensor.compute.device}"
        )
    tensor.bound = view
    if tensor.binding == "data":
        tensor.compute.data = view


def unbind_weight(tensor: StoreTensor) -> None:
    """Unbind ``tensor``: the data binding's ``.data`` returns to the placeholder.

    The forward's region address (``forward_ptr``, distributed-weight binding) is kept until the
    backward has finalized the gradient, so a re-gather into another region is caught.
    """
    tensor.bound = None
    if tensor.binding == "data":
        tensor.compute.data = tensor.placeholder


def set_overwrite(tensor: StoreTensor, overwrite: bool) -> None:
    """Set ``overwrite_main_grad`` for ``tensor`` (fixed per step by the residency mode).

    TE reads the flag *at forward* (E4), so it is set before the unit's first forward of the
    step, independently of when the target is bound. Allowed only for ``lazy`` single-use
    tensors: TE honours it with ``accumulate = fuse_wgrad and not overwrite``; Megatron's fused
    linear and the hook path always accumulate, and a second use in a segment would overwrite the
    first one's wgrad.
    """
    if overwrite and not (tensor.grad_capture == "lazy" and tensor.single_use):
        raise ValueError(
            f"{tensor.name}: overwrite_main_grad needs a lazy (TE fused-wgrad) single-use tensor "
            f"(grad capture {tensor.grad_capture!r}, single_use={tensor.single_use})"
        )
    tensor.overwrite = bool(overwrite)
    if tensor.binding == "data":
        tensor.compute.overwrite_main_grad = tensor.overwrite


def bind_grad(
    tensor: StoreTensor, region: torch.Tensor, *, zero: Optional[bool] = None
) -> torch.Tensor:
    """Bind ``tensor``'s gradient target to ``region`` (its padded flat region); returns the target.

    ``zero`` defaults to ``not tensor.overwrite``: a target that is accumulated into (``hook`` /
    ``forward`` tensors, multi-use tensors, a fresh Resident accumulator) is zeroed whole; an
    overwrite target gets only its pad tail zeroed. ``zero=False`` on an accumulated target is
    the Resident accumulator rebound mid-step (its sum is kept; the pad tail is re-zeroed).
    """
    if region.dim() != 1 or region.numel() != tensor.numel_padded:
        raise ValueError(
            f"{tensor.name}: gradient region of shape {tuple(region.shape)} != "
            f"[{tensor.numel_padded}] (padded rows included)"
        )
    if not region.dtype.is_floating_point:
        raise ValueError(f"{tensor.name}: gradient region dtype {region.dtype} is not floating")
    if zero is None:
        zero = not tensor.overwrite
    if zero:
        region.zero_()
    elif tensor.pad_length:
        region.narrow(0, tensor.numel, tensor.numel_padded - tensor.numel).zero_()
    target = region.narrow(0, 0, tensor.numel).view(tensor.logical_shape)
    tensor.grad_target = target
    tensor.grad_written = False
    if tensor.binding == "data":
        tensor.compute.main_grad = target
    return target


def unbind_grad(tensor: StoreTensor) -> None:
    """Unbind the gradient target; the data binding's ``main_grad`` is deleted (not None).

    A stray fused call then fails (Megatron tests ``hasattr(weight, "main_grad")``) instead of
    writing into a stale buffer. The overwrite flag is a per-step mode setting and is kept.
    """
    tensor.grad_target = None
    if tensor.binding == "data":
        tensor.compute.__dict__.pop("main_grad", None)


def _grad_hook(tensor: StoreTensor):
    """Post-accumulate hook: DDP's arithmetic into the bound target, then drop ``p.grad``.

    ``p.grad`` is added unless a fused consumer (TE, Megatron's fused linear, or
    ``finalize_group_grads``) flagged ``grad_added_to_main_grad``; a data-bound weight with
    ``zero_out_wgrad`` is still added (its fused consumer returned a zero dummy that autograd
    summed with the other consumers' gradients, e.g. a tied embedding's lookup), as DDP does
    (``distributed_data_parallel.py`` hook). A hook-path gradient into an overwrite target raises:
    the target was not zeroed, and the weight has a consumer outside TE.
    """
    ref = weakref.ref(tensor)
    name = tensor.name

    def hook(param: torch.Tensor) -> None:
        store_tensor = ref()
        fused = bool(getattr(param, "grad_added_to_main_grad", False))
        if fused:
            param.grad_added_to_main_grad = False
        grad = param.grad
        add = grad is not None and (
            not fused
            or (
                getattr(param, "zero_out_wgrad", False)
                and store_tensor is not None
                and store_tensor.binding == "data"
            )
        )
        if add:
            target = None if store_tensor is None else store_tensor.grad_target
            if target is None:
                raise RuntimeError(
                    f"loop param store: a gradient of {name} arrived while no gradient target "
                    "is bound"
                )
            if store_tensor.overwrite and not fused:
                raise RuntimeError(
                    f"loop param store: hook-path gradient for an overwrite target: {name} is "
                    "consumed outside TE; classify it as hook / multi-use"
                )
            if grad.shape != target.shape:
                raise RuntimeError(
                    f"loop param store: gradient of {name} has shape {tuple(grad.shape)}, "
                    f"its target {tuple(target.shape)}"
                )
            target.add_(grad)
        if store_tensor is not None:
            store_tensor.grad_written = True
        param.grad = None

    return hook


# ---------------------------------------------------------------------------------------------
# Use counting and the per-segment check of overwrite targets
# ---------------------------------------------------------------------------------------------


def track_uses(tensor: StoreTensor) -> None:
    """Count the owning module's graph-building forward calls (a pre-hook; host integer only).

    Only calls under ``torch.is_grad_enabled()`` count: the no-grad first pass of a reentrant
    checkpoint (Megatron's ``CheckpointFunction`` / ``CheckpointWithoutOutput``, TE's checkpoint,
    D42's ``mla_up_proj`` recompute) is skipped and its recompute (inside the backward, under
    ``enable_grad``) counts, so a checkpointed single use counts once by the end of the backward.
    torch's non-reentrant ``checkpoint`` builds a graph in both passes and counts twice; no
    Megatron recompute path uses it.
    """
    ref = weakref.ref(tensor)

    def pre_hook(_module, _args):
        store_tensor = ref()
        if store_tensor is not None and torch.is_grad_enabled():
            store_tensor.uses += 1

    tensor.tracked = True
    tensor.handles.append(tensor.module.register_forward_pre_hook(pre_hook))


def reset_uses(tensors: Sequence[StoreTensor]) -> None:
    """Start a new segment's use count."""
    for tensor in tensors:
        tensor.uses = 0


def check_segment_grads(tensors: Sequence[StoreTensor], segment: Any) -> List[str]:
    """Check the overwrite targets of ``segment`` after its backward; returns the names zeroed.

    Raises when a tracked overwrite tensor was consumed more than once (the second wgrad would
    overwrite the first). An overwrite target no gradient reached still holds whatever the pool
    entry held; it is zeroed here, before the reduce-scatter, which is what Resident yields. That
    covers a detached or unused branch and a weight never consumed under grad in the segment
    (``uses == 0``: DSA's indexer runs ``forward_before_topk`` under ``no_grad`` when its loss is
    off). A raw read of a TE weight under grad (its gradient bypasses TE, see
    ``classify_grad_capture``) is not silently zeroed: its gradient reaches the hook, which raises
    for an overwrite target, or, arriving after this segment's reduce, finds no target bound.
    Accumulated targets were zeroed when bound and need nothing.
    """
    zeroed = []
    for tensor in tensors:
        if not tensor.overwrite:
            continue
        if tensor.tracked and tensor.uses > 1:
            raise RuntimeError(
                f"store: {tensor.name} consumed {tensor.uses} times in segment {segment} under "
                "overwrite_main_grad"
            )
        if tensor.grad_target is not None and not tensor.grad_written:
            tensor.grad_target.zero_()
            tensor.grad_written = True
            zeroed.append(tensor.name)
    return zeroed


# ---------------------------------------------------------------------------------------------
# The DistributedWeight binding (TE >= 2.19)
# ---------------------------------------------------------------------------------------------

_DUMMY_WGRADS: Dict[Tuple[Tuple[int, ...], torch.dtype, torch.device], torch.Tensor] = {}

# Parameter attributes not carried onto the stand-in (binding state of the compute parameter).
_NOT_COPIED = frozenset(
    {"main_grad", "overwrite_main_grad", "grad_added_to_main_grad", "loop_store_tensor"}
)


def _dummy_wgrad(
    shape: Sequence[int], dtype: torch.dtype, device: torch.device, zero: bool = False
) -> torch.Tensor:
    """A persistent dummy gradient per (shape, dtype, device), zero-filled on request.

    TE's ``get_dummy_wgrad`` contract: the hook drops an ``empty`` dummy and adds a zero one (a
    ``zero_out_wgrad`` weight), so ``zero`` refills it on every call.
    """
    key = (tuple(shape), dtype, _device_key(device))
    dummy = _DUMMY_WGRADS.get(key)
    if dummy is None:
        dummy = torch.empty(tuple(shape), dtype=dtype, device=key[2])
        _DUMMY_WGRADS[key] = dummy
    if zero:
        dummy.zero_()
    return dummy.detach()


class RSBFDistributedWeight(torch.Tensor):
    """Stand-in for one TE linear weight under ``binding="distributed_weight"``.

    A shard-shaped tensor subclass requiring grad and implementing TE's ``DistributedWeight``
    protocol. It holds no data: it is backed by a shard-shaped placeholder (the store never reads
    it, and TE 2.20 rebinds the weight to the materialized view before any data read, reading only
    shape / dtype / device of the stand-in), so the DDP wrap that moves the canonical shard into
    its param buffer does not touch it. The protocol methods never run a collective (the store's
    executor-ordered schedule has already gathered and bound the unit), so they are graph-safe.
    ``_unsharded_shape`` is the logical shape (dev GTP's convention for logical-size readers).
    """

    is_distributed_weight = True

    @staticmethod
    def __new__(cls, data: torch.Tensor, requires_grad: bool = True):
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    @classmethod
    def create(cls, tensor: StoreTensor) -> "RSBFDistributedWeight":
        """The stand-in of ``tensor``; copies the compute parameter's attributes."""
        data = logical_placeholder(tensor.shard_shape, tensor.dtype, tensor.compute.device)
        weight = cls(data, True)
        for key, value in tensor.compute.__dict__.items():
            if key not in _NOT_COPIED:
                setattr(weight, key, value)
        weight._unsharded_shape = torch.Size(tensor.logical_shape)
        weight.loop_store_tensor = tensor
        weight.grad_added_to_main_grad = False
        return weight

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Keep the subclass (and its attributes) through ``detach``; decay for every other op."""
        del types
        kwargs = kwargs or {}
        with torch._C.DisableTorchFunctionSubclass():
            result = func(*args, **kwargs)
        if func is torch.Tensor.detach and args and isinstance(args[0], cls):
            out = result.as_subclass(cls)
            out.__dict__.update(args[0].__dict__)
            return out
        return result

    def _store_tensor(self) -> StoreTensor:
        tensor = self.__dict__.get("loop_store_tensor")
        if tensor is None:
            raise RuntimeError("RSBFDistributedWeight without its store tensor")
        return tensor

    def materialize_group_for_forward(self) -> torch.Tensor:
        """Protocol: the bound full weight for the forward GEMM (records its address)."""
        tensor = self._store_tensor()
        if tensor.bound is None:
            raise RuntimeError(f"loop param store: {tensor.name} is used unbound (forward)")
        tensor.forward_ptr = tensor.bound.data_ptr()
        return tensor.bound

    def materialize_group_for_backward(self, nvtx_label: Optional[str] = None) -> torch.Tensor:
        """Protocol: the bound full weight for the backward GEMMs; must be the forward's region."""
        del nvtx_label
        tensor = self._store_tensor()
        if tensor.bound is None:
            raise RuntimeError(f"loop param store: {tensor.name} is used unbound (backward)")
        if tensor.forward_ptr is not None and tensor.bound.data_ptr() != tensor.forward_ptr:
            raise RuntimeError(
                f"loop param store: {tensor.name} was re-bound to a different region between its "
                "forward and its backward (a re-gather must reuse the forward's slot)"
            )
        return tensor.bound

    def grad_buffer(self) -> torch.Tensor:
        """Protocol: the buffer TE's fused wgrad GEMM writes (``accumulate=False``).

        An overwrite target is that buffer itself (``finalize_group_grads`` then has nothing to
        copy); an accumulated target gets a fresh fp32 buffer that finalize adds.
        """
        tensor = self._store_tensor()
        target = tensor.grad_target
        if target is None:
            raise RuntimeError(f"loop param store: no gradient target bound for {tensor.name}")
        if tensor.overwrite:
            return target
        return torch.empty(tensor.logical_shape, dtype=target.dtype, device=target.device)

    def finalize_group_grads(self, wgrads: Any, nvtx_label: Optional[str] = None) -> torch.Tensor:
        """Protocol: write (overwrite target) or add the wgrad into the target; return a dummy.

        TE's dispatcher passes the bare wgrad for a ``Linear`` (a list for a group). The dummy is
        shard-shaped, and zero for a ``zero_out_wgrad`` weight (TE replaces it on its fused path).
        """
        del nvtx_label
        wgrad = wgrads[0] if isinstance(wgrads, (list, tuple)) else wgrads
        tensor = self._store_tensor()
        target = tensor.grad_target
        if target is None:
            raise RuntimeError(f"loop param store: no gradient target bound for {tensor.name}")
        if tuple(wgrad.shape) != tuple(target.shape):
            raise RuntimeError(
                f"loop param store: wgrad of {tensor.name} has shape {tuple(wgrad.shape)}, its "
                f"target {tuple(target.shape)}"
            )
        in_place = wgrad.data_ptr() == target.data_ptr() and wgrad.stride() == target.stride()
        if tensor.overwrite:
            if not in_place:
                target.copy_(wgrad)
        elif in_place:
            raise RuntimeError(
                f"loop param store: wgrad of {tensor.name} was written into its accumulated "
                "target (grad_buffer hands a fresh buffer for it)"
            )
        else:
            target.add_(wgrad)
        tensor.grad_written = True
        tensor.forward_ptr = None
        self.grad_added_to_main_grad = True
        zero = bool(getattr(self, "zero_out_wgrad", False))
        return _dummy_wgrad(self.shape, self.dtype, self.device, zero=zero)
