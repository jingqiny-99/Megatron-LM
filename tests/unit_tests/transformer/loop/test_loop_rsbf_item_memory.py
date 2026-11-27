# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The arena rule of the RSBF CORE item forms, measured with the CUDA caching allocator (M8 /
M8b memory gates; GPU only).

``LoopRSBFExecutor(measure_memory=True)`` records every cell call's allocator peak and
retained bytes above its entry baseline. The bounds are calibrated in the same process from
the allocator, per block ``j`` of each plan and per core sublayer ``l`` (round 1 of one
sample, inputs from a sweep-mode forward, every input allocated before its baseline):

* ``A_j``: bytes a block's grad-mode recompute leaves alive (its saved activations and
  output), ``P_j``: its peak over that forward and its backward;
* ``A^1_l``: the same for the one-sublayer block ``{l}``; ``S_l``: the peak of sublayer
  ``l``'s sweep-mode forward (grad enabled, input detached).

Asserted, with ``delta = 1 MiB`` per comparison (allocator rounding) and ``state`` the bytes of
one round state:

* the arena rule ``A_j <= sum_{l in j} A^1_l`` for every block of every stride;
* a block / whole-round item peaks between ``max_j A_j`` and ``max_j P_j + 2 state`` (the
  incoming block gradient and the outgoing leaf gradient); a nested item at most at
  ``(n_b - 1) state + max(max_l S_l + state, max_j P_j + 2 state)``;
* a sweep peaks at most at ``max_l S_l + (n_b - 1) state + state + crit`` (one sublayer, the
  block-input stash captured so far, the live sublayer input) and retains ``(n_b - 1) state +
  state + crit`` (the stash, ``s_r``, ``crit_r``); the store's peak "block" bytes are exactly
  ``(n_b - 1) state`` (one sample-round in flight: one sample per wave, depth 1);
* strides that refine each other peak monotonically, and stride 1 strictly below the whole
  round (fewer sublayers live: the point of the gate).

Setup, two models at S 4096, both with a core of 8 sublayers: a dense TE ``'*-' x 6`` model
(hidden 1024, bf16, unfused attention) and the DSV4-TINY-like model of
``test_loop_rsbf_item_forms`` (``HEHECEHECEHECE/WE``: mHC 4 streams, so one round state is
the ``[S, 1, 4 H]`` stream state; MTP 1, hash layer, expert bias, the alltoall dispatcher at
EP 1, which releases its routing probabilities in ``combine_postprocess``: the allgather
dispatcher keeps them attached until its next call and would hold one sublayer's graph past the
sweep's detach). Parameter gradients are preallocated by a warm-up step
(``zero_grad(set_to_none=False)``), so a backward allocates only transients;
``check_recompute`` and ``verify_stash`` off (their trace entries and clones would be retained
bytes).

D42: in the CODA item of the DSV4-TINY-like model at S 8192, the coda up-projection recompute
lowers the bytes the forward leaves alive for the backward (measured when the backward starts)
by at least the released up-projection outputs of all coda / MTP attentions, each calibrated as
the bytes alive after the attention's forward without the request minus with it; the item's
peak does not rise. Both byte counts are requested bytes (``requested_bytes.all.current``): the
allocator counts a large block it does not split in full, up to 1 MiB more than requested,
which is the size of an up projection here. The peak itself drops by less: it falls inside the backward of the
CSA attention's indexer loss, when the attentions after it are already back-propagated and its
own up projection is re-materialized, so only the released outputs of the attentions before it
are still saved there.
"""

import gc
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.loop.program import (
    CoreBackwardPlan,
    LoopPhase,
    LoopProgram,
    PrescribedDepthCriterion,
    SampleInputs,
    SampleKey,
)
from megatron.core.transformer.loop.rsbf_executor import LoopMicrobatch, LoopRSBFExecutor
from megatron.core.transformer.loop_transformer_block import recompute_context
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
from megatron.core.transformer.selective_recompute import request_selective_recompute
from tests.unit_tests.transformer.loop import test_loop_rsbf_executor as base
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    VOCAB,
    hybrid_loop_config,
    model_parallel_once,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the allocator measurements need CUDA"
)

# ``pretrain_hybrid``'s loss function with stub training args (the DSv4 case).
hybrid = base.hybrid

PATTERN = "*-" * 6  # prelude '*-', core '*-*-*-*-' (L = 8), coda '*-'
CORE = 8
SEQ = 4096
HIDDEN = 1024
DEPTHS = (1, 1)  # one sample per wave, depth 1: one sample-round in flight
MIB = 1 << 20
DELTA = MIB
# Forms: name -> (item form, stride); "whole" is the whole round (stride L).
FORMS = {
    "whole": ("whole_round", None),
    "block1": ("block", 1),
    "block2": ("block", 2),
    "block4": ("block", 4),
    "nested1": ("nested", 1),
    "nested2": ("nested", 2),
}


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        try:
            yield
        finally:
            base.get_loop_rsbf_schedule().reset()


def _loss_func(loss_mask, output):
    """``pretrain_hybrid.loss_func``'s default branch (per-token sum, token count, report)."""
    losses = output.view(-1).float()
    mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * mask)
    num_tokens = mask.sum().clone().detach().to(torch.int)
    return (
        loss,
        num_tokens,
        {"lm loss": torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])},
    )


def _dense_model():
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_stack_spec

    config = hybrid_loop_config(
        PATTERN,
        2,
        2,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        num_query_groups=8,
        kv_channels=HIDDEN // 8,
        ffn_hidden_size=4 * HIDDEN,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.unfused,
        loop_iterations=2,
        loop_schedule="rsbf",
        calculate_per_token_loss=True,
    )
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    return HybridModel(
        config=config,
        hybrid_stack_spec=loop_hybrid_stack_spec,
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        hybrid_layer_pattern=PATTERN,
        position_embedding_type="none",
    ).cuda()


def _set_form(model, name):
    form, stride = FORMS[name]
    model.config.loop_rsbf_item_form = form
    model.config.loop_rsbf_block_stride = stride


def _microbatches(batches):
    return [
        LoopMicrobatch(
            tokens=batch["tokens"],
            position_ids=batch["position_ids"],
            labels=batch["labels"],
            loss_mask=batch["loss_mask"],
            attention_mask=None,
            padding_mask=None,
            loss_func=_loss_func,
        )
        for batch in batches
    ]


def _measured_step(model, make_microbatches, depths):
    """A warm-up step, then the measured step (fresh records), of a new executor."""
    program = LoopProgram(model, exit_criterion=PrescribedDepthCriterion(dict(enumerate(depths))))
    executor = LoopRSBFExecutor(
        program, pg_collection=SimpleNamespace(dp=None), waves=len(depths), measure_memory=True
    )
    for _ in range(2):
        model.zero_grad(set_to_none=False)
        set_moe_metrics_tracker(MoEMetricsTracker())
        executor.run_step(
            lambda it, m: next(it), iter(make_microbatches()), len(depths), model=model
        )
    return executor


def _sample(batch):
    return SampleInputs(
        key=SampleKey(0, 0),
        tokens=batch["tokens"],
        position_ids=batch["position_ids"],
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
        attention_mask=None,
        padding_mask=None,
        loss_func=_loss_func,
    )


def _requested_bytes():
    """The bytes the live tensors requested, without the allocator's block rounding (a large
    block that is not split counts in full in ``memory_allocated``)."""
    return torch.cuda.memory_stats()["requested_bytes.all.current"]


def _baseline():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


class _Calibration:
    """Allocator measurements of round 1 of one sample (block live / peak bytes, sublayer
    sweep peaks)."""

    def __init__(self, program, batch):
        self.program = program
        self.sample = _sample(batch)
        self.exited = torch.zeros(1, dtype=torch.int32, device="cuda")
        # The prescribed-depth criterion reads the depth port (depth 1: round 1 exits).
        self.depth = torch.ones(1, dtype=torch.int32, device="cuda")
        with torch.enable_grad():
            prelude = program.prelude(self.sample, phase=LoopPhase.SWEEP)
            self.injection = prelude.injection.detach()
            self.crit = prelude.crit_state.detach()
            fwd = program.core.forward_detached(
                None,
                self.injection,
                self.sample,
                r=1,
                phase=LoopPhase.SWEEP,
                capture=range(1, CORE),
                crit_state=self.crit,
                exited=self.exited,
                depth=self.depth,
            )
        with torch.no_grad():
            first = program.core.prologue(None, self.injection)
        # The input of every core sublayer (sublayer 0: the adapter output).
        self.inputs = {0: first.detach(), **fwd.captured}
        self.state = self.injection.numel() * self.injection.element_size()
        self.crit_bytes = max(self.crit.numel() * self.crit.element_size(), 512)
        self._blocks = {}

    def block(self, block):
        """``(A_j, P_j)`` of ``block`` (cached)."""
        key = (block.start, block.end)
        if key not in self._blocks:
            program = self.program
            x = None if block.start == 0 else self.inputs[block.start]
            seed = torch.randn_like(self.injection)
            leaf = None if x is None else x.detach().requires_grad_()
            injection = self.injection.detach().requires_grad_()
            base_bytes = _baseline()
            with torch.enable_grad(), recompute_context("calibration/phase=recompute"):
                out = program.core(
                    leaf,
                    injection,
                    self.sample,
                    r=1,
                    phase=LoopPhase.RECOMPUTE,
                    crit_state=self.crit,
                    exited=self.exited,
                    depth=self.depth,
                    block=block,
                )
                torch.cuda.synchronize()
                live = torch.cuda.memory_allocated() - base_bytes
                target = out.state if block.with_epilogue else out
                torch.autograd.backward([target], [seed])
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - base_bytes
            del out, target, leaf, injection, seed
            self._blocks[key] = (live, peak)
        return self._blocks[key]

    def sweep_sublayer(self, l):
        """``S_l``: the peak of sublayer ``l``'s sweep-mode forward (input allocated)."""
        program = self.program
        index = program.roles.core_start + l
        x = self.inputs[l].detach()
        ctx = self.sample.ctx
        base_bytes = _baseline()
        with torch.enable_grad(), ctx.outer_quant_context:
            y = program.stack._run_range(x, index, index + 1, ctx, "calibration").detach()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base_bytes
        del y
        return peak


@pytest.mark.parametrize("case", ["dense", "dsv4_tiny"])
def test_item_forms_follow_the_arena_rule(case, hybrid):
    """Every CORE item form and every sweep within the calibrated arena bounds (see the module
    docstring); stride 1 strictly below the whole round. Dense TE and DSV4-TINY-like."""
    batches = base._batches(len(DEPTHS), seq=SEQ, dev=torch.device("cuda"))
    with base._hadamard_fallback():
        if case == "dense":
            model = _dense_model()
            make_microbatches = lambda: _microbatches(batches)  # noqa: E731
        else:
            from tests.unit_tests.transformer.loop import test_loop_rsbf_item_forms as forms

            model = forms._dsv4_tiny(torch.bfloat16)
            assert model.config.moe_token_dispatcher_type == "alltoall"
            make_microbatches = lambda: base._microbatches(model, hybrid, batches)  # noqa: E731
        _check_arena(model, make_microbatches, batches)
    del model, make_microbatches
    gc.collect()
    torch.cuda.empty_cache()


def _check_arena(model, make_microbatches, batches):
    records = {}
    for name in FORMS:
        _set_form(model, name)
        executor = _measured_step(model, make_microbatches, DEPTHS)
        assert executor.program.item_form == FORMS[name][0]
        assert executor.program.roles.core == CORE
        records[name] = (executor.stats.memory, executor.stats.store_peaks)
        del executor
    _set_form(model, "whole")
    program = LoopProgram(model, exit_criterion=PrescribedDepthCriterion({0: 1}))
    calibration = _Calibration(program, batches[0])
    state, crit = calibration.state, calibration.crit_bytes
    sublayer_peaks = [calibration.sweep_sublayer(l) for l in range(CORE)]
    single = [calibration.block(b) for b in CoreBackwardPlan.strided(CORE, 1).blocks]
    max_sweep = max(sublayer_peaks)
    item_peaks = {}
    for name, (memory, store_peaks) in records.items():
        form, stride = FORMS[name]
        plan = CoreBackwardPlan.strided(CORE, stride or CORE)
        blocks = [calibration.block(b) for b in plan.blocks]
        stashed = len(plan.stashed_block_inputs()) if form == "block" else 0
        # The arena rule: a block holds at most its sublayers' live sets.
        for b, (live, _) in zip(plan.blocks, blocks):
            assert live <= sum(single[l][0] for l in range(b.start, b.end)) + DELTA, (name, b)
        max_live = max(live for live, _ in blocks)
        max_peak = max(peak for _, peak in blocks)
        items = [r for r in memory if r.kind == "core"]
        sweeps = [r for r in memory if r.kind == "core_sweep"]
        assert len(items) == len(sweeps) == sum(DEPTHS)
        for record in items:
            assert record.peak >= max_live - DELTA, (name, record, max_live)
            if form == "nested":
                bound = (len(plan.blocks) - 1) * state + max(
                    max_sweep + state, max_peak + 2 * state
                )
            else:
                bound = max_peak + 2 * state
            assert record.peak <= bound + DELTA, (name, record, bound)
        for record in sweeps:
            assert record.peak <= max_sweep + stashed * state + state + crit + DELTA, (name, record)
            assert abs(record.retained - (stashed * state + state + crit)) <= DELTA, (name, record)
        for peak in store_peaks:
            assert peak.peak_bytes_by_kind.get("block", 0) == stashed * state, (name, peak)
        item_peaks[name] = max(record.peak for record in items)
    print({name: peak / MIB for name, peak in item_peaks.items()})
    assert item_peaks["block1"] <= item_peaks["block2"] + DELTA
    assert item_peaks["block2"] <= item_peaks["block4"] + DELTA
    assert item_peaks["block4"] <= item_peaks["whole"] + DELTA
    assert item_peaks["block1"] < item_peaks["whole"]
    assert item_peaks["nested1"] < item_peaks["whole"]


def test_coda_up_proj_recompute_lowers_the_coda_item_memory(hybrid):
    """D42 on the DSV4-TINY-like model at S 8192: the CODA item's forward leaves at least the
    released up-projection outputs of all coda / MTP attentions fewer bytes alive for its
    backward, and its peak does not rise (see the module docstring)."""
    from tests.unit_tests.transformer.loop import test_loop_rsbf_item_forms as forms

    seq, depths = 8192, (1, 1)
    batches = base._batches(len(depths), seq=seq, dev=torch.device("cuda"))
    peaks, saved = {}, {}
    with base._hadamard_fallback():
        for request in (False, True):
            model = forms._dsv4_tiny(torch.bfloat16, loop_rsbf_coda_recompute_mla_up_proj=request)
            program = LoopProgram(
                model, exit_criterion=PrescribedDepthCriterion(dict(enumerate(depths)))
            )
            executor = LoopRSBFExecutor(
                program, pg_collection=SimpleNamespace(dp=None), waves=2, measure_memory=True
            )
            # Requested bytes alive when the CODA item seeds its backward (the loss is complete:
            # every activation the backward needs is alive) above those at the item's start,
            # one entry per CODA item.
            at_backward, item_start = [], [None]
            coda_item, grad_scale = executor._coda_item, executor._grad_scale

            def measured_coda_item(g, depth, coda_item=coda_item, item_start=item_start):
                item_start[0] = _requested_bytes()
                try:
                    coda_item(g, depth)
                finally:
                    item_start[0] = None

            def measured_grad_scale(
                loss, grad_scale=grad_scale, item_start=item_start, at_backward=at_backward
            ):
                if item_start[0] is not None:
                    at_backward.append(_requested_bytes() - item_start[0])
                return grad_scale(loss)

            executor._coda_item = measured_coda_item
            executor._grad_scale = measured_grad_scale
            microbatches = base._microbatches(model, hybrid, batches)
            for _ in range(2):
                at_backward.clear()
                model.zero_grad(set_to_none=False)
                set_moe_metrics_tracker(MoEMetricsTracker())
                executor.run_step(
                    lambda it, m: next(it), iter(microbatches), len(batches), model=model
                )
                microbatches = base._microbatches(model, hybrid, batches)
            records = [r for r in executor.stats.memory if r.kind == "coda"]
            assert len(records) == len(at_backward) == len(depths)
            peaks[request] = max(r.peak for r in records)
            saved[request] = list(at_backward)
            if request:
                released = []
                for attention in program.coda_recompute_attentions():
                    hidden = torch.randn(
                        seq, 1, model.config.hidden_size, device="cuda", dtype=torch.bfloat16
                    ).requires_grad_()
                    live = {}
                    for scoped in (False, True):
                        _baseline()
                        base_bytes = _requested_bytes()
                        with (
                            torch.enable_grad(),
                            request_selective_recompute({"mla_up_proj"} if scoped else ()),
                        ):
                            out, _ = attention(hidden_states=hidden, attention_mask=None)
                        torch.cuda.synchronize()
                        live[scoped] = _requested_bytes() - base_bytes
                        del out
                    released.append(live[False] - live[True])
                assert len(released) == 3 and min(released) > 0, released
            del model, program, executor
    savings = [plain - recompute for plain, recompute in zip(saved[False], saved[True])]
    print(
        f"D42 coda item peak {peaks[False] / MIB:.1f} -> {peaks[True] / MIB:.1f} MiB, "
        f"saved for the backward {[s / MIB for s in saved[False]]} -> "
        f"{[s / MIB for s in saved[True]]} MiB, "
        f"released per attention {[r / MIB for r in released]} MiB"
    )
    assert peaks[True] <= peaks[False]
    for saving in savings:
        assert saving >= sum(released) - DELTA, (savings, released)
