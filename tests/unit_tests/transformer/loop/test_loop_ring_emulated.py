# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The loop ring (``loop_schedule='ring'``) reproduces the PP = 1 eager loop: CPU emulation.

P stages of k chunks run in one process in Megatron's interleaved op order (``ring_emulation``:
Megatron's ``forward_step`` / ``backward_step`` per op, its schedule table and warmup counts for
V = R x k round views, P2P as received buffers and pseudo-freed outputs), against the PP = 1
eager loop in fp32. Models: local layers, dense or MoE (EP 1, natural sigmoid routing with
expert bias and the sequence aux loss, the prelude's MoE layers hash-routed), mHC 4, one MTP
depth, R = 4. Grid: P in {2, 4}, k in {1, 2}, M in {1, 3, 2P}, recompute plans all-C / all-n /
mixed / mixed-inv (``ring_emulation.check_ring`` lists the assertions):

* M = 1: per-token losses and every gradient bitwise;
* M > 1: losses bitwise; every gradient equals, bitwise, the ring's own per-call contributions
  summed in PP = 1's order (the runs differ only in the order of the same additions) and is
  within 1e-6 of PP = 1 relative to the contributions' magnitude;
* every parameter passes the G2 unit gate: cosine >= 0.9999 and ||dg|| / ||g|| <= 1e-3;
* the PP sum of the chunks' FLOPs per token equals PP = 1's exactly, as do every head call's
  loop metrics; each chunk records M x its per-microbatch FLOPs;
* the G1 counters are R / 1 / 1 (core runs per chunk per sample; prelude, coda and MTP once);
* injection stores empty, call roles cleared, built flags unchanged, MoE token counts and
  aux-loss trackers equal.

Forward-only runs (``torch.no_grad``, Megatron's forward-only order) free every store entry
after round R. The mutation tests break one rule each and must fail: no sink, the store keyed
by m - 1, the loop norm skipped in rounds 1-3, the embedding left on for later rounds, a
dropped round backward, round r's state fed to round r + 2, and a round closure that reads
live call state while other calls run before its recompute.

CPU only (the GPU ring parity is a separate test): mHC needs ``TORCHDYNAMO_DISABLE=1``.
"""

import dataclasses

import pytest
import torch

from megatron.core.transformer.loop import guards, ring_pp
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.loop.ring_pp import LoopRingRoundView
from tests.unit_tests.transformer.loop import hybrid_loop_utils as hlu
from tests.unit_tests.transformer.loop import ring_emulation as emu

pytestmark = pytest.mark.skipif(
    torch.cuda.is_available(), reason="CPU emulation of the ring (fp32 bitwise on CPU kernels)"
)

mhc_on_this_device = hlu.mhc_on_this_device
cpu_mtp = hlu.cpu_mtp


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with hlu.model_parallel_once():
        yield


@pytest.fixture
def cpu_ring(mhc_on_this_device, cpu_mtp, monkeypatch):
    """CPU environment of the emulation: the router's buffers on the CPU, one intra-op thread
    (both runs of a comparison use the same kernels)."""
    del mhc_on_this_device, cpu_mtp
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


_REFERENCES = {}


def _reference(kind, microbatches, forward_only=False):
    """The PP = 1 run of ``kind`` on ``microbatches`` batches (cached per module)."""
    key = (kind, microbatches, forward_only)
    if key not in _REFERENCES:
        batches = emu.make_batches(microbatches)
        reference, state, _ = emu.run_reference(kind, batches, forward_only=forward_only)
        _REFERENCES[key] = (batches, reference, state)
    return _REFERENCES[key]


def _run(kind, stages, chunks, microbatches, plan, **kwargs):
    """``(problems, ring, result)`` of one ring run against PP = 1."""
    batches, reference, state = _reference(kind, microbatches)
    ring = emu.build_ring(kind, stages, chunks, plan, state)
    result = emu.run_ring(ring, batches, capture=microbatches > 1, **kwargs)
    return emu.check_ring(reference, result, ring, microbatches), ring, result


GRID = [
    (kind, stages, chunks, microbatches, plan)
    for kind in emu.KINDS
    for stages in (2, 4)
    for chunks in (1, 2)
    for microbatches in (1, 3, 2 * stages)
    for plan in ("all-C", "all-n", "mixed", "mixed-inv")
]


@pytest.mark.parametrize("kind,stages,chunks,microbatches,plan", GRID)
def test_ring_reproduces_pp1(cpu_ring, kind, stages, chunks, microbatches, plan):
    problems, ring, result = _run(kind, stages, chunks, microbatches, plan)
    assert not problems, "\n".join(problems)
    # Every rank ran its 2 x M x V ops; the stores filled and drained.
    assert len(result.extra["executed"]) == 2 * stages * microbatches * ring.chunks * emu.ROUNDS


@pytest.mark.parametrize("kind", emu.KINDS)
@pytest.mark.parametrize("stages,chunks", [(2, 1), (4, 2)])
def test_forward_only_frees_the_injection_store(cpu_ring, kind, stages, chunks):
    """Evaluation (Megatron's forward-only order, no autograd): the losses and loop metrics are
    PP = 1's, and a microbatch's store entry lives from its round-1 call to its round-R call."""
    microbatches = 2 * stages
    batches, reference, state = _reference(kind, microbatches, forward_only=True)
    ring = emu.build_ring(kind, stages, chunks, "all-C", state)
    first = ring.chunk_model(0, 0).decoder
    seen = []

    def observe(stage, kind_, view, microbatch):
        del kind_
        if stage == 0 and view % chunks == 0:  # a call of the round-start chunk
            round_index = view // chunks
            held = microbatch in first.ring_injection_keys()
            seen.append(round_index)
            assert held == (round_index < emu.ROUNDS - 1), (round_index, microbatch)

    result = emu.run_ring(ring, batches, forward_only=True, observe=observe)
    problems = emu.check_ring(reference, result, ring, microbatches, forward_only=True)
    assert not problems, "\n".join(problems)
    assert sorted(seen) == sorted(list(range(emu.ROUNDS)) * microbatches)
    assert first.ring_injection_keys() == ()


def test_recompute_runs_after_other_calls(cpu_ring):
    """In Megatron's order a checkpointed round's recompute (in its backward) runs after other
    calls of its chunk: forwards of later microbatches and other rounds."""
    _, ring, result = _run("moe", 2, 1, 4, "all-C")
    executed = result.extra["executed"]
    between = []
    for index, (stage, kind, view, microbatch) in enumerate(executed):
        if kind != "F" or stage != 0:
            continue
        end = executed.index((stage, "B", view, microbatch))
        between.append(sum(op[0] == stage for op in executed[index + 1 : end]))
    assert min(between) > 0 and max(between) > ring.chunks * emu.ROUNDS


# ---------------------------------------------------------------------------------------------
# Mutations: each breaks one rule and must fail.
# ---------------------------------------------------------------------------------------------

MUTATED = ("moe", 2, 1, 4, "all-C")


def _fails(**kwargs):
    """True if the mutated run raises or does not reproduce PP = 1."""
    try:
        problems, _, _ = _run(*MUTATED, **kwargs)
    except (RuntimeError, ValueError, AssertionError):
        return True
    return bool(problems)


def test_baseline_of_the_mutations_passes(cpu_ring):
    problems, _, _ = _run(*MUTATED)
    assert not problems, "\n".join(problems)


def test_mutation_no_sink(cpu_ring, monkeypatch):
    """Rounds 2..R's injection gradients dropped (the sink never filled)."""
    monkeypatch.setattr(
        ring_pp._RingInjectionUse, "backward", staticmethod(lambda ctx, gradient: (None, None))
    )
    assert _fails()


def test_mutation_store_keyed_by_previous_microbatch(cpu_ring, monkeypatch):
    """Later rounds look the injection up under m - 1."""
    monkeypatch.setattr(
        LoopHybridStack,
        "_ring_injection_key",
        lambda self, call: call.microbatch - 1 if call.round > 0 else call.microbatch,
    )
    assert _fails()


def test_mutation_loop_norm_only_in_the_last_round(cpu_ring, monkeypatch):
    original = LoopHybridStack._ring_round_function

    def last_round_norm(self, step, ctx, label, *, adapt, norm):
        last = step == self.config.loop_iterations - 1
        return original(self, step, ctx, label, adapt=adapt, norm=norm and last)

    monkeypatch.setattr(LoopHybridStack, "_ring_round_function", last_round_norm)
    assert _fails()


def test_mutation_embedding_left_on_for_later_rounds(cpu_ring, monkeypatch):
    original = LoopRingRoundView.roles_for

    def embed_every_round(self, microbatch):
        call = original(self, microbatch)
        return dataclasses.replace(call, pre_process=bool(self._model.pre_process))

    monkeypatch.setattr(LoopRingRoundView, "roles_for", embed_every_round)
    with pytest.raises(ValueError, match=guards.RING_CALL_ROLES.split(" (")[0]):
        _run(*MUTATED)


def test_mutation_dropped_round_backward(cpu_ring):
    """Round 2's backward on the last stage never runs (its input gets a zero gradient)."""
    mutation = emu.RingMutation(
        drop_backward=lambda stage, view, microbatch: (stage, view, microbatch) == (1, 1, 0)
    )
    assert _fails(mutation=mutation)


def test_mutation_round_fed_to_round_plus_two(cpu_ring):
    """Round 3 on the round-start chunk gets round 1's state (the input round 2 received)."""
    chunks = MUTATED[2]

    def feed(stage, view, microbatch, received, inputs):
        if stage == 0 and view == 2 * chunks:
            return inputs[(0, chunks, microbatch)]
        return received

    assert _fails(mutation=emu.RingMutation(feed=feed))


def test_mutation_round_closure_reads_live_call_state(cpu_ring, monkeypatch):
    """A round body that looks the injection up through the chunk's latest call instead of
    taking it as a bound tensor: right in the forward, wrong in a recompute that runs after
    other calls (``test_recompute_runs_after_other_calls``)."""
    original_loop = LoopHybridStack._ring_layer_loop
    original_function = LoopHybridStack._ring_round_function

    def recording_loop(self, hidden_states, ctx):
        self.__dict__["_latest_call"] = self.call_roles  # stays after the call: live state
        return original_loop(self, hidden_states, ctx)

    def live_function(self, step, ctx, label, *, adapt, norm):
        body = original_function(self, step, ctx, label, adapt=adapt, norm=norm)
        own = self.call_roles.microbatch

        def round_body(previous, injection):
            if adapt:
                live = self._ring_injections.get(self.__dict__["_latest_call"].microbatch)
                mine = self._ring_injections.get(own)
                if live is not None and mine is not None and live is not mine:
                    injection = injection + (live.tensor - mine.tensor).detach()
            return body(previous, injection)

        return round_body

    monkeypatch.setattr(LoopHybridStack, "_ring_layer_loop", recording_loop)
    monkeypatch.setattr(LoopHybridStack, "_ring_round_function", live_function)
    assert _fails()
