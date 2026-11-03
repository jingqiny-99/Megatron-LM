# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""NVTX ranges of the hybrid loop (``hybrid_stack.loop_range``).

The loop's profiler ranges (``torch.profiler.record_function``) are also NVTX start / end ranges
while Megatron's NVTX ranges are on (``configure_nvtx_profiling(True)``: ``--nvtx-ranges`` inside
the nsys-profiled steps), so an nsys report shows each microbatch, round, block and sublayer:

* Off (the default): the loop makes no NVTX call at all.
* On: one ``loop_hybrid/mb={m}`` range per forward, the ``/prelude`` and ``/coda`` edges,
  ``/loop={r}`` for every round, ``/block={k}`` for every block checkpoint and ``/layer={i}`` for
  every sublayer call. A checkpointed range (a block, or the round in round mode) opens again in
  its recompute during the backward; without recompute every range opens once. Every range that
  starts ends, also when the checkpoint's recompute stops early (``torch.utils.checkpoint``
  early stop raises through the checkpointed function), which then may not reach a block's last
  sublayers; with early stop off the recompute reopens every sublayer of the block.
* The ranges change no number: outputs, input gradients and parameter gradients with the ranges
  on equal those with them off bitwise, in every recompute mode.

Local layers in fp32 on CPU (gloo, one rank) and CUDA; ``torch.cuda.nvtx`` is replaced by a
recorder (a CPU build has no NVTX). Model parallel is initialized once per module.
"""

import collections
import itertools
import re

import pytest
import torch
from torch.utils.checkpoint import set_checkpoint_early_stop

from megatron.core import utils as core_utils
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    BATCH,
    HIDDEN,
    SEQ,
    build_local_loop_stack,
    device,
    model_parallel_once,
    rope,
)

PATTERN = "*-" * 6  # prelude '*-*-' (4 sublayers), core '*-*-' (4), coda '*-*-' (4)
P = C = D = 4
ROUNDS = 2
MODES = {
    "none": {},
    "round": dict(loop_recompute=True),
    "s2-edges": dict(loop_recompute=True, loop_recompute_block_stride=2, loop_recompute_edges=True),
    "s3": dict(loop_recompute=True, loop_recompute_block_stride=3),
}


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


class NvtxRecorder:
    """Stands in for ``torch.cuda.nvtx``: records start / end ranges and push / pop calls."""

    def __init__(self):
        self.ids = itertools.count(1)
        self.started = {}  # id -> name
        self.ended = []  # ids
        self.pushes = 0
        self.pops = 0

    def range_start(self, name):
        handle = next(self.ids)
        self.started[handle] = name
        return handle

    def range_end(self, handle):
        assert handle in self.started and handle not in self.ended, handle
        self.ended.append(handle)

    def range_push(self, _name):
        self.pushes += 1

    def range_pop(self):
        self.pops += 1

    def loop_names(self):
        """Started names of the loop's ranges, in start order."""
        return [self.started[h] for h in sorted(self.started)]


@pytest.fixture
def nvtx(monkeypatch):
    recorder = NvtxRecorder()
    for name in ("range_start", "range_end", "range_push", "range_pop"):
        monkeypatch.setattr(torch.cuda.nvtx, name, getattr(recorder, name))
    return recorder


@pytest.fixture
def nvtx_on():
    """Megatron's NVTX ranges on for one test (off again afterwards).

    Megatron's push / pop ranges inside the layers (``nvtx_range_push``) keep stale stack
    entries when a recompute stops early; the stack is cleared after the test."""
    core_utils.configure_nvtx_profiling(True)
    try:
        yield
    finally:
        core_utils.configure_nvtx_profiling(False)
        core_utils._nvtx_range_messages.clear()


def _stack(mode):
    return build_local_loop_stack(PATTERN, P, D, loop_iterations=ROUNDS, **MODES[mode])


def _run(stack, seed=7):
    """Forward + backward of ``sum(y * upstream)``; detached ``(y, dx, grads)``."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(SEQ, BATCH, HIDDEN, generator=generator).to(device()).requires_grad_(True)
    upstream = torch.randn(SEQ, BATCH, HIDDEN, generator=generator).to(device())
    stack.zero_grad(set_to_none=True)
    y = stack(x, attention_mask=None, rotary_pos_emb=rope())
    ((y * upstream).sum() + sum(stack.loop_losses.values())).backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in stack.named_parameters()
        if param.grad is not None
    }
    return y.detach().clone(), x.grad.detach().clone(), grads


def test_no_nvtx_call_when_ranges_are_off(nvtx):
    _run(_stack("s2-edges"))
    assert not nvtx.started and nvtx.pushes == 0


@pytest.mark.parametrize("early_stop", [True, False], ids=["early-stop", "full-recompute"])
@pytest.mark.parametrize("mode", list(MODES))
def test_ranges_name_every_microbatch_round_block_and_sublayer(mode, early_stop, nvtx, nvtx_on):
    stack = _stack(mode)
    with set_checkpoint_early_stop(early_stop):
        _run(stack)
    label = f"loop_hybrid/mb={stack._microbatch_id - 1}"
    names = nvtx.loop_names()
    counts = collections.Counter(names)
    assert sorted(nvtx.ended) == sorted(nvtx.started), "a loop range did not end"
    assert all(name.startswith(label) for name in names)
    if not early_stop:  # the layers' push / pop ranges balance when every recompute completes
        assert nvtx.pushes == nvtx.pops and not core_utils._nvtx_range_messages

    config = MODES[mode]
    recompute = config.get("loop_recompute", False)
    stride = config.get("loop_recompute_block_stride")
    edges = config.get("loop_recompute_edges", False)

    def sublayer_opens(name, replayed):
        """1 without a replay; 2 with a full recompute; 1 or 2 when it may stop early."""
        if not replayed:
            return counts[name] == 1
        return counts[name] == 2 if not early_stop else counts[name] in (1, 2)

    assert counts[label] == 1
    assert counts[f"{label}/prelude"] == 1 and counts[f"{label}/coda"] == 1
    # Round ranges: the block branch opens the round once (its blocks replay); round mode opens
    # it inside the checkpointed round, so the recompute opens it again.
    round_opens = 2 if recompute and stride is None else 1
    for r in range(1, ROUNDS + 1):
        assert counts[f"{label}/loop={r}"] == round_opens, (mode, r)
        for i in range(P, P + C):
            assert sublayer_opens(f"{label}/loop={r}/layer={i}", recompute), (mode, r, i)
        if recompute:  # every block's (or the round's) first sublayer runs in its recompute
            firsts = range(P, P + C, stride) if stride is not None else [P]
            assert all(counts[f"{label}/loop={r}/layer={i}"] == 2 for i in firsts), (mode, r)
        blocks = [n for n in counts if re.fullmatch(rf"{re.escape(label)}/loop={r}/block=\d+", n)]
        if stride is None:
            assert not blocks
        else:
            assert len(blocks) == -(-C // stride), (mode, r, blocks)
            assert all(counts[b] == 2 for b in blocks), (mode, r)
    for edge, first, length in (("prelude", 0, P), ("coda", P + C, D)):
        for i in range(first, first + length):
            assert sublayer_opens(f"{label}/{edge}/layer={i}", edges), (mode, edge, i)
        blocks = [n for n in counts if re.fullmatch(rf"{re.escape(label)}/{edge}/block=\d+", n)]
        if edges:
            assert len(blocks) == -(-length // stride), (mode, edge)
            assert all(counts[b] == 2 for b in blocks), (mode, edge)
        else:
            assert not blocks
    # The forward opens the microbatch range first and runs the rounds in order.
    assert names[0] == label
    first_round = [names.index(f"{label}/loop={r}") for r in range(1, ROUNDS + 1)]
    assert first_round == sorted(first_round)


@pytest.mark.parametrize("mode", list(MODES))
def test_ranges_change_no_number(mode, nvtx, nvtx_on):
    reference_stack = _stack(mode)
    core_utils.configure_nvtx_profiling(False)
    reference = _run(reference_stack)
    assert not nvtx.started
    stack = _stack(mode)
    stack.load_state_dict(reference_stack.state_dict())
    core_utils.configure_nvtx_profiling(True)
    result = _run(stack)
    assert nvtx.started, "the ranges did not fire"
    for name, value, expected in zip(("output", "input grad"), result[:2], reference[:2]):
        assert torch.equal(value, expected), (mode, name)
    assert result[2].keys() == reference[2].keys()
    for name, grad in reference[2].items():
        assert torch.equal(result[2][name], grad), (mode, name)
