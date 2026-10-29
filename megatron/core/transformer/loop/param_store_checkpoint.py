# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Checkpoints of the hybrid loop (M3c): coverage of the sharded state dict, and the loop
parameter store's row shards.

**Coverage.** The loop's own parameters (``LoopHybridStack.loop_adapter`` / ``loop_norm`` /
``depth_router``, and whatever a later stage registers on the stack: a direct ``Parameter``, a
child module, an ``nn.ParameterList``, a persistent buffer) need no loop-specific key code:
``HybridStack``'s ``sharded_state_dict`` walks every child but ``layers`` through
``sharded_state_dict_default`` and adds the stack's own tensors through ``_save_to_state_dict``,
so each gets a replicated entry under its module path (``decoder.<name>``; key table in
``hybrid_stack.py``). What can go wrong is a module whose custom ``sharded_state_dict`` drops one
of its tensors: the checkpoint then silently lacks it, and a resumed run re-initializes it.
``uncovered_state_names`` is the backstop against that (the training side raises on it at setup,
at the first save and on the load template, ``megatron/training/loop_checkpoint.py``).

Coverage rule. The state tensors of ``root`` are its registered parameters and persistent
buffers (what ``state_dict(keep_vars=True)`` holds, read from the modules directly so no state
dict hook or ``get_extra_state`` runs; names ending in ``_extra_state`` are skipped) plus the
loop parameter store's hidden compute parameters (``named_logical_parameters``' names); the
store's mirror modules (canonical shards) are not state tensors of their own. A tensor (counted
once however many names it has, e.g. tied embeddings) is covered when

* an entry's (top-level) dict key equals one of its names (dict keys are module paths), or
* the entries' data (a factory's ``data``) share its storage and their byte ranges together
  contain its bytes (identity is the special case; an owner emitting ``param.view(...)`` or
  disjoint row slices stays covered; an entry that is not a dense view of its storage, e.g. a
  column slice, contributes no range), or
* an entry's data is its canonical store shard.

A tensor without elements is covered trivially.

**Store shards (logical, z-independent layout).** A stored tensor is checkpointed under the key
its owner gives it without a store (the z = 1 key), as dim-0 row shards of the *logical* tensor
``[R, *rest]``: shard rank ``r`` of ``z`` writes rows ``[r * per, r * per + local_rows)`` with
``per = ceil(R / z)``; pad rows are never written. ``HybridModel.sharded_state_dict`` (and
``LoopHybridStack.sharded_state_dict``) run the owners' own ``sharded_state_dict`` inside the
*logical view* (``logical_view``: every hidden compute ``Parameter`` re-registered in its owner,
the store's mirror entries suppressed) and then ``convert_store_entries`` replaces each entry
whose data is a stored compute parameter by a ``ShardedTensorFactory`` (``row_shard_entry``) with
the same key, the same replica PP slot and the same ``allow_shape_mismatch`` flag, whose data is
the canonical shard ``Parameter`` and whose build emits one irregular-grid ``ShardedTensor``
(``axis_fragmentations=None``, ``global_offset = (start, 0, ...)``, replica ``(pp, 0,
replicate rank)``; an owner entry marked non-main between its PP slot and its last coordinate
keeps a non-main ``(pp, 1, replicate rank)``). A rank whose rows are all padding emits a 0-row
``ShardedTensor`` at offset ``R`` with a non-main replica id ``(pp, 0, 1 + shard rank)`` (an
empty build ``{}`` breaks the factory merge on load). Because the global tensor is logical, a
checkpoint saved at any z (or without a store) loads at any z (or without a store), in any
residency mode; the distributed
optimizer's model-space format (``fully_reshardable``) and the generic optimizer path derive
their entries from these factories (``replace(factory, key=..., data=state)``), so fp32 mains
and Adam moments reshard the same way.

Load. A ``load_state_dict`` pre-hook on the ``HybridModel`` (``install_load_hooks``) pops each
stored tensor's logical entry (the dict keys recorded by the conversion, or its name), turns it
into the canonical shard (``to_shard``: a factory merge output already is one; a full logical
tensor, e.g. from a legacy z = 1 state dict, is sliced) and inserts it under the mirror key; a
mirror-keyed state dict (a same-z ``state_dict()`` round trip) passes through. The post-hook
raises when a stored tensor got no entry (whatever ``strict`` is: ``load_checkpoint``'s
``strict=False`` retry would otherwise hide it) and calls ``LoopParamStore.on_checkpoint_load``
(compute copies unbound until the next ``begin_step`` / ``refresh``).
"""

import contextlib
import hashlib
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import nn

from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
from megatron.core.transformer.loop.param_store_binding import (
    HIDDEN_PARAMS_ATTR,
    MIRROR_FLAG_ATTR,
    StoreTensor,
    bind_weight,
    is_placeholder,
    padded_rows,
    register_logical,
    shard_slice,
    unbind_weight,
    unregister_logical,
)

if TYPE_CHECKING:  # pragma: no cover
    from megatron.core.transformer.loop.param_store import LoopParamStore

__all__ = [
    "EXTRA_STATE_SUFFIX",
    "MIRROR_PREFIX",
    "RowShard",
    "convert_store_entries",
    "install_load_hooks",
    "logical_layout_problem",
    "logical_sharded_state_dict",
    "logical_view",
    "named_state_tensors",
    "row_shard_entry",
    "store_checkpoint_layout",
    "to_shard",
    "uncovered_state_names",
]

EXTRA_STATE_SUFFIX = "_extra_state"
# The store's instance attribute on the hybrid model and its decoder (``LoopParamStore.install``).
_STORE_HOOK_ATTR = "_loop_store_hook"

_Range = Tuple[int, int]  # [start, end) in bytes of one storage


def named_state_tensors(root: nn.Module) -> List[Tuple[torch.Tensor, List[str]]]:
    """Every state tensor of ``root`` with all of its names, in module order.

    Registered parameters, persistent buffers and hidden compute parameters of every module under
    ``root`` (the store's mirror subtrees skipped); a tensor reachable under several names (tied
    weights, a shared module) appears once with every name. Names ending in ``_extra_state`` are
    skipped.
    """
    found: Dict[int, Tuple[torch.Tensor, List[str]]] = {}

    def add(name: str, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None or name.endswith(EXTRA_STATE_SUFFIX):
            return
        entry = found.setdefault(id(tensor), (tensor, []))
        if name not in entry[1]:
            entry[1].append(name)

    def visit(module: nn.Module, prefix: str, path: Tuple[int, ...]) -> None:
        if getattr(module, MIRROR_FLAG_ATTR, False) or id(module) in path:
            return
        path = path + (id(module),)
        # pylint: disable=protected-access
        for name, param in module._parameters.items():
            add(prefix + name, param)
        for name, param in (module.__dict__.get(HIDDEN_PARAMS_ATTR) or {}).items():
            add(prefix + name, param)
        for name, buffer in module._buffers.items():
            if name not in module._non_persistent_buffers_set:
                add(prefix + name, buffer)
        for name, child in module._modules.items():
            if child is not None:
                visit(child, f"{prefix}{name}.", path)

    visit(root, "", ())
    return list(found.values())


def _entry_data(value: Any) -> Optional[torch.Tensor]:
    """The tensor an entry carries: a plain tensor (legacy state dict), a ``ShardedTensor``'s or a
    ``ShardedTensorFactory``'s ``data``; None for objects."""
    if isinstance(value, torch.Tensor):
        return value
    data = getattr(value, "data", None)
    return data if isinstance(data, torch.Tensor) else None


def _dense_numel(tensor: torch.Tensor) -> Optional[int]:
    """Elements of the storage range ``tensor`` spans when it is a dense view (contiguous up to a
    permutation of its dims), else None."""
    if tensor.is_contiguous():
        return tensor.numel()
    expected = 1
    for stride, size in sorted((s, n) for s, n in zip(tensor.stride(), tensor.shape) if n != 1):
        if stride != expected:
            return None
        expected *= size
    return expected


def _byte_range(tensor: torch.Tensor) -> Optional[Tuple[Tuple[str, int], int, int]]:
    """``((device, storage pointer), start, end)`` of a dense view's bytes, or None (meta tensors,
    tensor subclasses without a plain storage, non-dense views)."""
    try:
        pointer = tensor.untyped_storage().data_ptr()
    except (RuntimeError, NotImplementedError, TypeError):
        return None
    numel = _dense_numel(tensor)
    if pointer == 0 or numel is None:
        return None
    size = tensor.element_size()
    start = tensor.storage_offset() * size
    return (str(tensor.device), pointer), start, start + numel * size


def _merge(ranges: Sequence[_Range]) -> List[_Range]:
    merged: List[_Range] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _store_shards(root: nn.Module) -> Dict[int, torch.Tensor]:
    """``id(compute parameter) -> canonical shard`` of every store installed under ``root``."""
    shards: Dict[int, torch.Tensor] = {}
    for module in root.modules():
        store = module.__dict__.get(_STORE_HOOK_ATTR)
        tensors = getattr(store, "tensors", None)
        if tensors is None:
            continue
        for tensor in tensors():
            if tensor.shard is not None:
                shards[id(tensor.compute)] = tensor.shard
    return shards


def _entries(sharded_state_dict: Any) -> Iterator[Any]:
    if isinstance(sharded_state_dict, (dict, list)):
        yield from nested_values(sharded_state_dict)


def uncovered_state_names(root: nn.Module, sharded_state_dict: Any) -> List[str]:
    """The state tensors of ``root`` that no entry of ``sharded_state_dict`` covers (module
    docstring), each by its first name.

    ``sharded_state_dict`` is what ``root.sharded_state_dict()`` returned (or a legacy
    ``state_dict``): dict keys relative to ``root``.
    """
    keys = set(sharded_state_dict) if isinstance(sharded_state_dict, dict) else set()
    data_ids = set()
    ranges: Dict[Tuple[str, int], List[_Range]] = defaultdict(list)
    for value in _entries(sharded_state_dict):
        data = _entry_data(value)
        if data is None:
            continue
        data_ids.add(id(data))
        span = _byte_range(data)
        if span is not None:
            ranges[span[0]].append((span[1], span[2]))
    merged = {storage: _merge(spans) for storage, spans in ranges.items()}
    shards = _store_shards(root)

    missing = []
    for tensor, names in named_state_tensors(root):
        if tensor.numel() == 0 or id(tensor) in data_ids or any(n in keys for n in names):
            continue
        shard = shards.get(id(tensor))
        if shard is not None and id(shard) in data_ids:
            continue
        span = _byte_range(tensor)
        if span is not None and any(
            start <= span[1] and span[2] <= end for start, end in merged.get(span[0], ())
        ):
            continue
        missing.append(names[0])
    return missing


# ---------------------------------------------------------------------------------------------
# Store shards: the row layout, the logical view, the conversion, the load hooks
# ---------------------------------------------------------------------------------------------

# Where the store registers its canonical shards, relative to the hybrid model.
MIRROR_PREFIX = "loop_param_store.mirror."
# The logical weights whose rows follow the (padded) vocabulary.
_VOCAB_ROWS = ("embedding.word_embeddings.weight", "output_layer.weight")


@dataclass(frozen=True)
class RowShard:
    """Shard rank ``shard_rank`` of ``shard_size``'s rows of a logical ``[rows, *rest]`` tensor:
    ``per = ceil(rows / z)`` padded rows, the real ones ``[start, start + local_rows)``."""

    rows: int
    rest: Tuple[int, ...]
    shard_size: int
    shard_rank: int

    @property
    def per(self) -> int:
        """Rows of a shard (pad rows included)."""
        return padded_rows(self.rows, self.shard_size) // self.shard_size

    @property
    def start(self) -> int:
        """The first logical row of this shard (``rows`` when all of its rows are padding)."""
        return min(self.shard_rank * self.per, self.rows)

    @property
    def local_rows(self) -> int:
        """The real (non-pad) rows of this shard."""
        return max(0, min(self.per, self.rows - self.shard_rank * self.per))

    @property
    def shard_shape(self) -> Tuple[int, ...]:
        """The canonical shard's shape."""
        return (self.per, *self.rest)

    @classmethod
    def of(cls, tensor: StoreTensor, shard_rank: int) -> "RowShard":
        """The row shard of store tensor ``tensor`` on shard rank ``shard_rank``."""
        return cls(
            rows=tensor.logical_shape[0],
            rest=tuple(tensor.logical_shape[1:]),
            shard_size=tensor.shard_size,
            shard_rank=shard_rank,
        )


def _pieces(built: Any) -> List[Any]:
    if isinstance(built, (dict, list)):
        return list(nested_values(built))
    return [built]


def _factory_layout_problem(entry: ShardedTensorFactory, tensor: StoreTensor) -> Optional[str]:
    logical = tuple(tensor.logical_shape)
    probe = torch.empty(logical, dtype=tensor.dtype, device="meta")
    try:
        built = entry.build_fn(entry.key, probe, entry.replica_id, entry.flattened_range)
    except Exception as error:  # pylint: disable=broad-exception-caught
        return f"its factory does not build on the logical shape ({error})"
    rows = []
    for piece in _pieces(built):
        if not isinstance(piece, ShardedTensor):
            return f"its factory builds a {type(piece).__name__}"
        if piece.key != entry.key:
            return f"its factory builds key {piece.key}, not {entry.key}"
        if piece.prepend_axis_num or tuple(piece.global_shape) != logical:
            return (
                f"its factory builds global shape {tuple(piece.global_shape)} with "
                f"{piece.prepend_axis_num} prepended axes, logical {logical}"
            )
        data = piece.data
        offset = tuple(piece.global_offset)
        if (
            data is None
            or not data.is_contiguous()
            or any(offset[1:])
            or tuple(data.shape[1:]) != logical[1:]
            or data.storage_offset() != offset[0] * tensor.row_elems
        ):
            return f"its factory builds a piece at offset {offset} that is not a row slice"
        rows.append((offset[0], offset[0] + data.shape[0]))
    covered = 0
    for start, end in sorted(rows):
        if start != covered:
            break
        covered = end
    if covered != logical[0]:
        return f"its factory's row slices {sorted(rows)} do not tile [0, {logical[0]})"
    return None


def logical_layout_problem(entry: Any, tensor: StoreTensor) -> Optional[str]:
    """Why ``entry`` (the owner's entry for stored ``tensor``) is not the logical tensor in row
    order, or None when it is: a ``ShardedTensor`` with global == local == logical shape and zero
    offsets, or a ``ShardedTensorFactory`` whose pieces (built on a ``meta`` probe) are
    contiguous dim-0 slices tiling the logical tensor at their global row offsets, under the
    entry's key (SwiGLU's fc1 factory at TP = 1 without singleton local shards)."""
    logical = tuple(tensor.logical_shape)
    if isinstance(entry, ShardedTensor):
        if entry.flattened_range is not None:
            return "a flattened range"
        if entry.prepend_axis_num:
            return f"{entry.prepend_axis_num} prepended axes"
        if tuple(entry.global_shape) != logical or tuple(entry.local_shape) != logical:
            return (
                f"global shape {tuple(entry.global_shape)}, local shape "
                f"{tuple(entry.local_shape)}, logical {logical}"
            )
        if any(entry.global_offset):
            return f"global offset {tuple(entry.global_offset)}"
        return None
    if isinstance(entry, ShardedTensorFactory):
        return _factory_layout_problem(entry, tensor)
    return f"a {type(entry).__name__} entry"


def row_shard_entry(
    entry: Any,
    tensor: StoreTensor,
    shard: RowShard,
    replicate_rank: Optional[int],
    *,
    on_merge: Optional[Callable[[torch.Tensor], None]] = None,
) -> ShardedTensorFactory:
    """The row-shard factory replacing ``entry`` for stored ``tensor`` (module docstring).

    ``data`` is the canonical shard ``Parameter`` (so the distributed optimizer's identity map
    and every ``replace(factory, data=state)`` carry optimizer states through the same rows).
    The build checks the shape it is given (``shard.shard_shape``) and emits the real rows (a
    0-row, non-main entry for an all-padding shard); the merge returns a tensor of the shard
    shape with zero pad rows: in place when the loaded rows are the built view of the tensor the
    build was given (no allocation), else a new tensor. ``on_merge`` sees every merge output
    (the store marks them as factory sources for its load pre-hook). ``replicate_rank``: the
    writer election over the shard's replicas (None keeps the entry's replica id)."""
    flag = bool(getattr(entry, "allow_shape_mismatch", False))
    rows, rest, local = shard.rows, shard.rest, shard.local_rows
    shape = shard.shard_shape
    zeros_rest = (0,) * len(rest)
    replica = entry.replica_id
    pp_slot = replica[0] if isinstance(replica, tuple) else 0
    dtype, device = tensor.shard.dtype, tensor.shard.device
    targets: Dict[int, Any] = {}

    def build(key, t, replica_id, flattened_range):
        if flattened_range is not None:
            raise ValueError(f"{key}: row-shard entries take no flattened range")
        if tuple(t.shape) != shape:
            raise ValueError(
                f"{key}: a row-shard entry needs data of shape {shape}, got {tuple(t.shape)}"
            )
        slot = replica_id[0] if isinstance(replica_id, tuple) else 0
        if local == 0:
            return ShardedTensor(
                key,
                t[:0],
                t.dtype,
                (0, *rest),
                (rows, *rest),
                (rows, *zeros_rest),
                None,
                replica_id=(slot, 0, 1 + shard.shard_rank),
                allow_shape_mismatch=flag,
            )
        view = t[:local]
        if view.data_ptr():
            targets[view.data_ptr()] = weakref.ref(t)
        return ShardedTensor(
            key,
            view,
            t.dtype,
            (local, *rest),
            (rows, *rest),
            (shard.start, *zeros_rest),
            None,
            replica_id=replica_id,
            allow_shape_mismatch=flag,
        )

    @torch.no_grad()
    def merge(loaded):
        out = None
        if isinstance(loaded, torch.Tensor) and local and loaded.numel():
            ref = targets.get(loaded.data_ptr())
            target = ref() if ref is not None else None
            if (
                target is not None
                and tuple(target.shape) == shape
                and target.dtype == loaded.dtype
                and target.device == loaded.device
            ):
                if shard.per > local:
                    target.narrow(0, local, shard.per - local).zero_()
                out = target
        if out is None:
            like = loaded if isinstance(loaded, torch.Tensor) else None
            out = torch.zeros(
                shape,
                dtype=like.dtype if like is not None else dtype,
                device=like.device if like is not None else device,
            )
            if local:
                out.narrow(0, 0, local).copy_(loaded)
        if on_merge is not None:
            on_merge(out)
        return out

    if replicate_rank is None:
        replica_id = replica
    else:
        # An owner entry marked non-main on a coordinate between the PP slot and the data-parallel
        # one (an alias / tied entry) stays non-main: it must not become a second main replica.
        marked = isinstance(replica, tuple) and any(replica[1:-1])
        replica_id = (pp_slot, int(marked), replicate_rank)
    return ShardedTensorFactory(entry.key, tensor.shard, build, merge, replica_id=replica_id)


def to_shard(
    tensor: StoreTensor, value: torch.Tensor, shard: RowShard, *, source: str
) -> torch.Tensor:
    """The canonical shard of ``tensor`` from a loaded ``value``, decided by its source:
    ``"factory"`` (a row-shard merge output: already the shard) or ``"full"`` (the logical
    tensor, e.g. a legacy z = 1 state dict: sliced, pad rows zero). A shape that does not fit
    raises (for the embedding / output weight naming a padded-vocab change)."""
    shape = tuple(value.shape)
    if source == "factory":
        if shape != shard.shard_shape:
            raise ValueError(
                f"loop parameter store: {tensor.name}: the loaded row shard is {shape}, "
                f"expected {shard.shard_shape}"
            )
        return value
    if source != "full":
        raise ValueError(f"unknown source {source!r}")
    logical = tuple(tensor.logical_shape)
    if shape != logical:
        if tensor.name in _VOCAB_ROWS and shape[1:] == logical[1:]:
            raise ValueError(
                f"loop parameter store: {tensor.name}: saved {shape[0]} rows, this run "
                f"{logical[0]} rows: a padded-vocab change (--make-vocab-size-divisible-by / "
                "tokenizer) would load a misaligned row range"
            )
        raise ValueError(
            f"loop parameter store: {tensor.name}: the loaded tensor is {shape}, expected the "
            f"logical {logical}"
        )
    return shard_slice(value.detach(), shard.shard_size, shard.shard_rank)


@contextlib.contextmanager
def logical_view(store: "LoopParamStore", root: nn.Module) -> Iterator[None]:
    """Re-register every stored compute ``Parameter`` in its owner while the owners build their
    checkpoint entries (re-entrant; the outermost entry records ``root``). The store's own
    ``sharded_state_dict`` returns ``{}`` meanwhile. Raises inside a training step.

    A bound compute copy (Resident / RSG after a step or ``refresh()``) is unbound for the view
    (the data binding's ``.data`` on the NaN placeholder), so an owner entry that carries a view
    of a stored weight rather than the parameter itself is a placeholder view, which
    ``convert_store_entries`` refuses, whatever the residency state; it can never save a compute
    copy's values. The owners' ``_parameters`` and the bindings are restored on exit, also when
    the body raises."""
    # pylint: disable=protected-access
    if store.in_step or store._cursor is not None:
        raise RuntimeError("store: checkpoint view inside a training step")
    outer = store._ckpt_view_depth == 0
    registered: List[StoreTensor] = []
    unbound: List[Tuple[StoreTensor, torch.Tensor]] = []
    if outer:
        object.__setattr__(store, "_ckpt_root", weakref.ref(root))
    store._ckpt_view_depth += 1
    try:
        if outer:
            for tensor in store.tensors():
                if tensor.bound is not None:
                    unbound.append((tensor, tensor.bound))
                    unbind_weight(tensor)
                register_logical(tensor)
                registered.append(tensor)
        yield
    finally:
        for tensor in registered:
            unregister_logical(tensor)
        for tensor, view in unbound:
            bind_weight(tensor, view)
        store._ckpt_view_depth -= 1
        if outer:
            object.__setattr__(store, "_ckpt_root", None)


def logical_sharded_state_dict(
    store: "LoopParamStore", root: nn.Module, produce: Callable[[], Any], *, prefix: str = ""
) -> Any:
    """``produce()`` (the owners' sharded state dict of ``root``) inside the logical view; the
    outermost call converts the stored tensors' entries (``convert_store_entries``)."""
    outer = store._ckpt_view_depth == 0  # pylint: disable=protected-access
    with logical_view(store, root):
        sharded_state_dict = produce()
        if outer:
            convert_store_entries(store, sharded_state_dict, root=root, prefix=prefix)
    return sharded_state_dict


def _walk(container: Any, path: Tuple[Any, ...] = ()) -> Iterator[Tuple[Any, Any, Tuple]]:
    """``(container, key, path)`` of every leaf of a nested dict / list."""
    items = container.items() if isinstance(container, dict) else enumerate(container)
    for key, value in list(items):
        if isinstance(value, (dict, list)):
            yield from _walk(value, path + (key,))
        else:
            yield container, key, path + (key,)


_NOT_LOGICAL = (
    "the store saves row shards of the logical tensor only (TP = 1, no singleton local shards)"
)


def convert_store_entries(
    store: "LoopParamStore", sharded_state_dict: Any, *, root: nn.Module, prefix: str = ""
) -> Dict[str, Tuple[str, ...]]:
    """Replace every entry whose data is a stored compute parameter by its row-shard factory
    (``row_shard_entry``); returns the dict keys per stored tensor (relative to ``prefix``),
    recorded for the load pre-hook when ``root`` is the hybrid model. Raises (message 8) for an
    entry that is not the logical tensor in row order (a view of a stored weight included), and
    for a stored tensor
    whose owner lies under ``root`` but emitted no entry; stored tensors outside ``root`` (the
    embedding for a decoder-only call) are not expected."""
    # pylint: disable=protected-access
    by_id = {id(tensor.compute): tensor for tensor in store.tensors()}
    keys: Dict[str, List[str]] = defaultdict(list)
    for container, key, path in list(_walk(sharded_state_dict)):
        value = container[key]
        data = getattr(value, "data", None)
        if not isinstance(data, torch.Tensor):
            continue
        label = ".".join(str(part) for part in path)
        tensor = by_id.get(id(data))
        if tensor is None:
            if data.device.type != "meta" and is_placeholder(data):
                base = by_id.get(id(getattr(data, "_base", None)))
                name = base.name if base is not None else "a stored weight"
                raise ValueError(
                    f"loop parameter store: {name}'s checkpoint entry {label} is not the logical "
                    f"tensor in row order (a view of the stored weight); {_NOT_LOGICAL}"
                )
            continue
        problem = logical_layout_problem(value, tensor)
        if problem is not None:
            raise ValueError(
                f"loop parameter store: {tensor.name}'s checkpoint entry {label} is not the "
                f"logical tensor in row order ({problem}); {_NOT_LOGICAL}"
            )
        container[key] = row_shard_entry(
            value,
            tensor,
            RowShard.of(tensor, store.shard_rank),
            store.replicate_rank,
            on_merge=store._mark_merged,
        )
        relative = label[len(prefix) :] if prefix and label.startswith(prefix) else label
        keys[tensor.name].append(relative)
    owners = {id(module) for module in root.modules()}
    for tensor in store.tensors():
        if tensor.name not in keys and id(tensor.module) in owners:
            raise ValueError(
                f"loop parameter store: {type(tensor.module).__name__} emitted no entry for "
                f"{tensor.name} in the logical view; {_NOT_LOGICAL}"
            )
    recorded = {name: tuple(found) for name, found in keys.items()}
    if root is store.hybrid:  # the load pre-hook runs on the hybrid model: its relative keys
        store._ckpt_entry_keys.update(recorded)
    return recorded


def install_load_hooks(store: "LoopParamStore", hybrid: nn.Module) -> List[Any]:
    """Register the load pre- and post-hooks on ``hybrid`` (module docstring); the handles are
    kept on the store and returned."""
    # pylint: disable=protected-access

    def pre_hook(_module, state_dict, prefix, *_args):
        consumed: Dict[str, str] = {}
        missing: Dict[str, List[str]] = {}
        for tensor in store.tensors():
            mirror_key = prefix + MIRROR_PREFIX + tensor.name
            candidates = [prefix + key for key in store._ckpt_entry_keys.get(tensor.name, ())]
            candidates = list(dict.fromkeys(candidates + [prefix + tensor.name]))
            present = [key for key in candidates if key in state_dict]
            if not present:
                if mirror_key in state_dict:
                    consumed[tensor.name] = "mirror"
                else:
                    missing[tensor.name] = candidates + [mirror_key]
                continue
            values = [state_dict.pop(key) for key in present]
            value = values[0]
            for other, key in zip(values[1:], present[1:]):
                same = other is value or (
                    isinstance(other, torch.Tensor)
                    and other.shape == value.shape
                    and torch.equal(other, value)
                )
                if not same:
                    raise ValueError(
                        f"loop parameter store: {tensor.name}: the loaded entries {present[0]} "
                        f"and {key} differ"
                    )
            if mirror_key in state_dict:
                raise ValueError(
                    f"loop parameter store: {tensor.name}: the loaded state has both its logical "
                    f"entry {present[0]} and its mirror shard {mirror_key}"
                )
            source = "factory" if store._is_merged(value) else "full"
            state_dict[mirror_key] = to_shard(
                tensor, value, RowShard.of(tensor, store.shard_rank), source=source
            )
            consumed[tensor.name] = source
        store._ckpt_consumed = consumed
        store._ckpt_missing = missing

    def post_hook(_module, _incompatible_keys):
        missing = store._ckpt_missing
        if missing:
            looked = "; ".join(f"{name}: {', '.join(keys)}" for name, keys in missing.items())
            raise RuntimeError(
                "loop parameter store: the loaded model state has no entry for "
                f"{', '.join(missing)} (keys looked up: {looked})"
            )
        store.on_checkpoint_load()

    handles = [
        hybrid._register_load_state_dict_pre_hook(pre_hook, with_module=True),
        hybrid.register_load_state_dict_post_hook(post_hook),
    ]
    store._ckpt_handles.extend(handles)
    return handles


def store_checkpoint_layout(store: "LoopParamStore") -> Dict[str, Any]:
    """The store's checkpoint-relevant layout (plain types): z, ``min_numel``, the binding, the
    stored dtype, the stored-tensor count, a digest of the stored set (names and logical shapes
    in unit order) and the unit count."""
    text = "\n".join(
        f"{tensor.name}:{','.join(str(n) for n in tensor.logical_shape)}"
        for tensor in store.tensors()
    )
    return {
        "z": int(store.shard_size),
        "min_numel": int(store.config.min_numel),
        "binding": str(store.config.binding),
        "dtype": str(store.dtype),
        "stored": sum(1 for _ in store.tensors()),
        "stored_sha": hashlib.sha256(text.encode()).hexdigest(),
        "units": len(store.units),
    }
