# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Scoped requests for selective activation recompute.

``recompute_granularity='selective'`` with ``recompute_modules`` is a config-wide switch: every
layer built from the config checkpoints the named modules. A schedule that wants a module
recomputed only in some of its forwards (the loop RSBF executor's CODA item, decision D42:
the up projections of the coda and MTP attentions, not those of the core rounds) asks for it
per forward instead:

    with request_selective_recompute({"mla_up_proj"}):
        output = coda(...)

Modules that support a request read ``selective_recompute_requested(name)`` when they run and
checkpoint the named part as the config switch would (the forward runs without saving the
part's activations; its backward re-runs it). Outside a request every module behaves as its
config says.

Supported names (``SELECTIVE_RECOMPUTE_MODULES``):

* ``"mla_up_proj"``: the query / key-value up projections and RoPE of ``DSv4HybridAttention``
  (``get_query_key_value_tensors``).
"""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

__all__ = [
    "SELECTIVE_RECOMPUTE_MODULES",
    "request_selective_recompute",
    "selective_recompute_requested",
]

# The module names a request may carry.
SELECTIVE_RECOMPUTE_MODULES = frozenset({"mla_up_proj"})

_REQUESTED: ContextVar[frozenset] = ContextVar("selective_recompute_requested", default=frozenset())


@contextmanager
def request_selective_recompute(modules: Iterable[str]) -> Iterator[None]:
    """Ask modules that support selective recompute to checkpoint ``modules`` in the forwards
    run inside the scope (the union with an enclosing request). An empty request changes
    nothing; unknown names raise ``ValueError``. The previous request is restored on exit,
    also when the body raises."""
    modules = frozenset(modules)
    unknown = modules - SELECTIVE_RECOMPUTE_MODULES
    if unknown:
        raise ValueError(
            f"unknown selective recompute modules {sorted(unknown)}; supported: "
            f"{sorted(SELECTIVE_RECOMPUTE_MODULES)}"
        )
    token = _REQUESTED.set(_REQUESTED.get() | modules)
    try:
        yield
    finally:
        _REQUESTED.reset(token)


def selective_recompute_requested(module: str) -> bool:
    """True inside a ``request_selective_recompute`` scope that names ``module``."""
    return module in _REQUESTED.get()
