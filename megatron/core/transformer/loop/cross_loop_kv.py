# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""First-pass attention tensors shared by tied core layers within one model forward."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from torch import Tensor, nn

_CURRENT_PASS: ContextVar[CrossLoopPass | None] = ContextVar("cross_loop_pass", default=None)


def current_cross_loop_pass() -> CrossLoopPass | None:
    """Return the active recurrent pass, or None outside a sharing-enabled core."""
    return _CURRENT_PASS.get()


class CrossLoopPass:
    """One round of a microbatch. All its rounds share a store keyed by the physical module."""

    def __init__(self, step: int, saved: dict[tuple[nn.Module, str], Tensor]) -> None:
        self.is_source = step == 0
        self._source_support = None
        self._saved: dict[tuple[nn.Module, str], Tensor] | None = saved

    def compact_source(self, signature, owners, build):
        """Share one immutable support within the open source visit; keep its inputs alive."""
        if not self.is_source or self._saved is None:
            return build()
        if self._source_support is None or self._source_support[0] != signature:
            self._source_support = signature, owners, build()
        return self._source_support[2]

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _CURRENT_PASS.set(self)
        try:
            yield
        finally:
            _CURRENT_PASS.reset(token)

    def publish(self, module: nn.Module, name: str, value: Tensor) -> None:
        """Publish the original, graph-connected tensor once; replay still runs the producer."""
        if not self.is_source:
            raise RuntimeError("Only the first loop pass can publish attention tensors")
        if self._saved is not None:
            self._saved.setdefault((module, name), value)

    def finish_source(self) -> None:
        """A producer checkpoint may replay, but must not retain its own autograd graph."""
        if self.is_source:
            self._source_support = None
            self._saved = None

    def read(self, module: nn.Module, name: str) -> Tensor:
        """Read the first-pass tensor of this same module for a later round."""
        if self.is_source or self._saved is None:
            raise RuntimeError("Reading shared attention tensors requires a later loop pass")
        try:
            return self._saved[module, name]
        except KeyError as exc:
            raise RuntimeError(
                f"First loop pass did not publish {name!r} for {type(module).__name__}"
            ) from exc
