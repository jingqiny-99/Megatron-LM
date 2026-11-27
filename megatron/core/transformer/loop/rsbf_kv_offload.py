# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pinned storage for the first-round native DSA bank in two-round RSBF training."""

from dataclasses import dataclass

import torch


@dataclass
class _Slot:
    host: torch.Tensor
    has_adjoint: bool = False


class SparseKVOffload:
    """One copy stream and one pinned page per active sample slot and physical layer.

    The page holds the source KV through the last consumer forward, then its gradient
    through the source backward. All host-page accesses are enqueued on the same stream,
    including reuse by a later wave. No GPU tensor lives in the slot.
    """

    def __init__(self, device, *, measure_time=False):
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.slots = {}
        self.pinned_bytes = 0
        self.stats = {"peak_pinned_bytes": 0}
        self.timings = [] if measure_time else None
        self.wait_tails = {}

    def _start(self, stream):
        if self.timings is not None:
            return stream.record_event(torch.cuda.Event(enable_timing=True))

    def _end(self, kind, start, stream):
        event = stream.record_event(torch.cuda.Event(enable_timing=start is not None))
        if start is not None:
            self.timings.append((kind, start, event))
        return event

    def _count(self, kind, host):
        key = kind + "_bytes"
        self.stats[key] = self.stats.get(key, 0) + host.nbytes

    def _spill(self, slot, value, kind):
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        start = self._start(self.stream)
        with torch.cuda.stream(self.stream):
            slot.host.copy_(value.detach(), non_blocking=True)
        value.record_stream(self.stream)
        if start is not None:
            self._end(kind, start, self.stream)
        self._count(kind, slot.host)

    def _load(self, slot, kind):
        consumer = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self.stream):
            start = self._start(self.stream)
            restored = slot.host.to(self.device, non_blocking=True)
            ready = self._end(kind, start, self.stream)
        wait_start = self._start(consumer)
        consumer.wait_event(ready)
        if wait_start is not None:
            self.wait_tails[consumer] = self._end("h2d_consumer_wait", wait_start, consumer)
        restored.record_stream(consumer)
        self._count(kind, slot.host)
        return restored

    def publish(self, key, value):
        slot = self.slots.get(key)
        if slot is None or slot.host.shape != value.shape or slot.host.dtype != value.dtype:
            if slot is not None:
                # A variable-length microbatch in a later wave can change the page size.
                # Drain its previous use before returning the old page to the pinned allocator.
                self.stream.synchronize()
                self.pinned_bytes -= slot.host.nbytes
            slot = self.slots[key] = _Slot(
                torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=True)
            )
            self.pinned_bytes += slot.host.nbytes
            self.stats["peak_pinned_bytes"] = max(self.stats["peak_pinned_bytes"], self.pinned_bytes)
        slot.has_adjoint = False
        self._spill(slot, value, "kv_d2h")

    def read(self, key, *, backward):
        slot = self.slots[key]
        if slot.has_adjoint:
            raise RuntimeError("A two-pass RSBF sparse KV consumer read after its backward")
        restored = self._load(slot, "kv_h2d")
        if backward:
            def spill_adjoint(grad):
                # Autograd combines all contributions to this one consumer leaf. Later
                # rounds would require a separate lifetime and gradient accumulation.
                if slot.has_adjoint:
                    raise RuntimeError("Two-pass RSBF sparse KV received a second consumer adjoint")
                self._spill(slot, grad, "grad_d2h")
                slot.has_adjoint = True

            restored.requires_grad_().register_hook(spill_adjoint)
        return restored

    def connect_source(self, key, value):
        slot = self.slots[key]
        if value.requires_grad and slot.has_adjoint:
            value.register_hook(lambda grad: grad + self._load(slot, "grad_h2d"))

    def join(self):
        """Join the copy stream before the existing end-of-step loss/exit host read."""
        compute = torch.cuda.current_stream(self.device)
        compute.wait_stream(self.stream)
        for end in self.wait_tails.values():
            compute.wait_event(end)

    def drain(self):
        """An interrupted step may have outstanding copies and skip its normal host read."""
        self.join()
        torch.cuda.current_stream(self.device).synchronize()

    def close(self):
        if self.timings:
            for kind, start, end in self.timings:
                key = kind + "_ms"
                self.stats[key] = self.stats.get(key, 0.0) + start.elapsed_time(end)
        self.slots.clear()
        return self.stats
