# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The ring's injection (``ring_pp``): ``_RingInjectionSource`` / ``_RingInjectionUse`` and the
store (``RingInjection``).

Huginn feeds the prelude output ``e`` to the adapter of every round. At PP = 1 autograd sums
the four rounds' gradients in ``e``'s producer, in backward order: ``((g4 + g3) + g2) + g1``.
On the ring, round 1 runs on ``Source(e)`` and rounds 2..4 on ``Use(leaf)`` of a detached leaf;
the round calls are separate graphs joined by received buffers, run backward as round 4, 3, 2,
1. The tests check, in fp32 and bf16, that ``e``'s producer gets one gradient equal bit for bit
to the one-graph sum (the sink holds ``(g4 + g3) + g2`` when round 1's backward adds ``g1``),
and the store's lifetimes: the entry lives from round 1's forward to round 1's backward in
training, and the stack frees it after round R without autograd (forward-only); the real
stack's lifetimes in Megatron's order are checked through the ring emulation.
"""

import pytest
import torch

from megatron.core.transformer.loop.ring_pp import (
    RingInjection,
    ring_injection_source,
    ring_injection_use,
)
from tests.unit_tests.transformer.loop import hybrid_loop_utils as hlu
from tests.unit_tests.transformer.loop import ring_emulation as emu

ROUNDS = 4
SHAPE = (6, 2, 8)  # [S, B, h]

mhc_on_this_device = hlu.mhc_on_this_device
cpu_mtp = hlu.cpu_mtp


def _rounds(dtype, seed=0):
    """The prelude input, its weights and the per-round adapter weights (leaves)."""
    generator = torch.Generator().manual_seed(seed)
    hidden = SHAPE[-1]

    def leaf(*shape, scale=1.0):
        value = torch.randn(*shape, generator=generator) * scale
        return value.to(dtype).requires_grad_()

    return (
        leaf(*SHAPE),
        leaf(hidden, hidden, scale=hidden**-0.5),
        [leaf(hidden, 2 * hidden, scale=(2 * hidden) ** -0.5) for _ in range(ROUNDS)],
    )


def _prelude(x, weight):
    return torch.tanh(x @ weight)


def _round(state, injection, adapter):
    return torch.tanh(torch.cat((state, injection), -1) @ adapter.t())


def _one_graph(dtype):
    """PP = 1: one graph, ``e`` used by all four rounds."""
    x, weight, adapters = _rounds(dtype)
    e = _prelude(x, weight)
    seen = []
    e.register_hook(lambda gradient: seen.append(gradient.clone()))
    state = torch.zeros_like(e)
    for adapter in adapters:
        state = _round(state, e, adapter)
    state.float().sum().backward()
    assert len(seen) == 1
    return x.grad, [adapter.grad for adapter in adapters], seen[0]


def _ring(dtype, checkpoints=None):
    """The ring: four separate round graphs joined by received buffers, the injection through
    the store; backward round 4, 3, 2, 1. Returns the gradients, ``e``'s gradient and the sink
    states observed after each later round's backward."""
    x, weight, adapters = _rounds(dtype)
    e = _prelude(x, weight)
    seen = []
    e.register_hook(lambda gradient: seen.append(gradient.clone()))
    store = {}
    entry = RingInjection(store, 7, e)
    assert entry.grad_mode and store == {7: entry}
    assert entry.tensor.is_leaf and entry.tensor.requires_grad
    assert entry.tensor.data_ptr() == e.data_ptr()  # shared storage, no copy
    round_grads = {}
    outputs, received = [], []
    state = torch.zeros_like(e)
    for index, adapter in enumerate(adapters):
        if index == 0:
            injection = ring_injection_source(e, entry)
        else:
            injection = ring_injection_use(entry)
            state = outputs[-1].detach().requires_grad_()  # the received buffer
            received.append(state)
        injection.register_hook(lambda g, index=index: round_grads.__setitem__(index, g.clone()))
        outputs.append(_round(state, injection, adapter))
    sinks = []
    outputs[-1].float().sum().backward()  # round 4 (with the loss)
    sinks.append(entry.total.clone())
    for index in range(ROUNDS - 2, -1, -1):  # rounds 3, 2, 1
        outputs[index].backward(received[index].grad)
        if index > 0:
            sinks.append(entry.total.clone())
            assert store == {7: entry}  # held until round 1's backward
    assert store == {} and entry.tensor is None and entry.total is None
    assert len(seen) == 1, "e's producer gets exactly one gradient"
    assert received[0].grad is not None
    return x.grad, [adapter.grad for adapter in adapters], seen[0], round_grads, sinks


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_injection_sum_is_bitwise_the_one_graph_sum(dtype):
    x_grad, adapter_grads, e_grad = _one_graph(dtype)
    ring_x, ring_adapters, ring_e, round_grads, sinks = _ring(dtype)
    assert torch.equal(ring_e, e_grad)
    assert torch.equal(ring_x, x_grad)
    for ring_grad, grad in zip(ring_adapters, adapter_grads):
        assert torch.equal(ring_grad, grad)
    # The sink, in backward order: g4, g4 + g3, (g4 + g3) + g2; round 1 returns g1 + sink.
    g = round_grads
    assert torch.equal(sinks[0], g[3])
    assert torch.equal(sinks[1], g[3] + g[2])
    assert torch.equal(sinks[2], (g[3] + g[2]) + g[1])
    assert torch.equal(ring_e, g[0] + sinks[2])
    assert all(sink.dtype == dtype for sink in sinks)
    # The rounds matter: dropping the sink would change e's gradient.
    assert not torch.equal(ring_e, g[0])


def test_one_round_needs_no_sink():
    """R = 1: round 1's backward returns its own gradient and frees the entry."""
    x = torch.randn(*SHAPE, requires_grad=True)
    e = torch.tanh(x)
    store = {}
    entry = RingInjection(store, 0, e)
    (ring_injection_source(e, entry) * 3.0).sum().backward()
    assert store == {} and entry.total is None
    torch.testing.assert_close(x.grad, 3.0 * (1 - e.detach() ** 2), rtol=1e-6, atol=0)


def test_forward_only_entry_holds_the_output_itself():
    """Without autograd the entry stores the prelude output (no leaf, no sink); the later
    rounds read it; whoever runs round R releases it."""
    x = torch.randn(*SHAPE, requires_grad=True)
    with torch.no_grad():
        e = torch.tanh(x)
        store = {}
        entry = RingInjection(store, 3, e)
        assert not entry.grad_mode and entry.tensor is e
        assert ring_injection_source(e, entry) is e
        assert ring_injection_use(entry) is e
        entry.release()
    assert store == {} and entry.tensor is None


def test_store_refuses_a_second_entry_and_a_grad_mode_change():
    e = torch.randn(*SHAPE, requires_grad=True) * 1.0
    store = {}
    entry = RingInjection(store, 0, e)
    with pytest.raises(RuntimeError, match="already holds microbatch 0"):
        RingInjection(store, 0, e)
    with torch.no_grad(), pytest.raises(RuntimeError, match="autograd was on in round 1"):
        ring_injection_use(entry)
    entry.release()
    assert store == {}


def test_prelude_output_without_grad_is_not_tracked():
    """A prelude output that needs no gradient (e.g. frozen weights) is stored as is."""
    e = torch.randn(*SHAPE)
    store = {}
    entry = RingInjection(store, 1, e)
    assert not entry.grad_mode and ring_injection_source(e, entry) is e
    entry.release()


# ---------------------------------------------------------------------------------------------
# The real stack's store in Megatron's interleaved order (training)
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _model_parallel():
    with hlu.model_parallel_once():
        yield


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU emulation of the ring")
@pytest.mark.parametrize("chunks", [1, 2])
def test_training_entry_lives_from_round_one_forward_to_round_one_backward(
    _model_parallel, mhc_on_this_device, cpu_mtp, monkeypatch, chunks
):
    """On the round-start chunk, microbatch m's entry exists exactly between the forward and
    the backward of its round-1 view (view 0 of stage 0), and its sink fills during the later
    rounds' backwards."""
    del mhc_on_this_device, cpu_mtp
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    microbatches = 4
    batches = emu.make_batches(microbatches)
    reference, state, _ = emu.run_reference("moe", batches)
    ring = emu.build_ring("moe", 2, chunks, "mixed", state)
    first = ring.chunk_model(0, 0).decoder
    events = []

    def observe(stage, kind, view, microbatch):
        if stage != 0 or view % chunks != 0:
            return
        entry = first._ring_injections.get(microbatch)
        events.append((kind, view // chunks, microbatch, entry is not None))
        if entry is not None and kind == "B":
            assert entry.total is not None  # a later round's backward filled the sink

    result = emu.run_ring(ring, batches, observe=observe, capture=True)
    assert not emu.check_ring(reference, result, ring, microbatches)
    for kind, round_index, microbatch, held in events:
        expected = not (kind == "B" and round_index == 0)
        assert held == expected, (kind, round_index, microbatch)
    assert first.ring_injection_keys() == ()
