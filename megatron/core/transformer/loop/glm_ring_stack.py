# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Local execution of the prelude, shared-core passes and coda of a GLM ring."""

from contextlib import nullcontext

import torch

from megatron.core.transformer.forward_replay import forward_statistics_round
from megatron.core.transformer.loop.guards import validate_hybrid_loop_inputs
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.loop.ring_pp import (
    LoopRingCall,
    RingCrossLoopPass,
    RingInjection,
    ring_injection_source,
    ring_injection_use,
)
from megatron.core.transformer.loop_transformer_block import loop_round_checkpoint
from megatron.core.utils import make_viewless_tensor


def _publish_gate_input(value, bank, key):
    """Retain the full pass input on the last PP rank while a distinct tensor is sent."""
    entry = RingInjection(bank, key, value)
    outgoing = ring_injection_source(value, entry)
    # In forward-only mode the entry holds value itself. The scheduler may replace
    # outgoing.data after its P2P send; the saved object must keep its shape and storage.
    return make_viewless_tensor(outgoing.view_as(outgoing), value.requires_grad, keep_graph=True)


class GLMRingStack(LoopHybridStack):
    """Each physical GLM stack has one role. All core laps call the same physical stack."""

    _owns_full_recompute = True

    def _lap_uses_checkpoint(self, checkpoint, core):
        return getattr(self.config, "recompute_granularity", None) == "full" or (
            checkpoint and (core or self.config.loop_recompute_edges)
        )

    def _ring_lap_work(self, lap, checkpoint, sequence, batch):
        """Yield (forward FLOPs, backward factor, recomputed) for this physical call."""
        tokens = sequence * batch
        core = 1 <= lap <= self.config.loop_iterations
        sparse_pass = (
            RingCrossLoopPass(lap - 1, {}) if core and self._cross_loop_attention_layers else None
        )
        recompute = self._lap_uses_checkpoint(checkpoint, core)
        yield (
            self.range_forward_flops(
                0,
                len(self.layers),
                tokens,
                tokens * sequence,
                sequence,
                cross_loop_pass=sparse_pass,
            ),
            2,
            recompute,
        )
        if core and self.loop_ring.round_end:
            yield self.round_extra_forward_flops(tokens), 2, False
        if lap == self.config.loop_iterations + 1 and self.loop_ring.round_end:
            for value, factor in self.head_work_terms(tokens, tokens * sequence, sequence):
                yield value, factor, False

    def ring_microbatch_work(self, sequence, batch, codes, *, depth=None):
        if not getattr(self.config, "loop_ring_separate_edges", False):
            return super().ring_microbatch_work(sequence, batch, codes)
        rounds = self.config.loop_iterations
        depth = rounds if depth is None else depth
        laps = {"prelude": (0,), "core": range(1, rounds + 1), "coda": (rounds + 1,)}[
            self.loop_ring.role
        ]
        totals = [0.0, 0.0, 0.0]
        for lap, code in zip(laps, codes, strict=True):
            if code not in ("C", "n"):
                raise ValueError(f"Unknown GLM recompute code {code!r}")
            if self.loop_ring.role == "core" and lap > depth:
                continue
            for value, factor, recompute in self._ring_lap_work(lap, code == "C", sequence, batch):
                totals[0] += value
                totals[1] += value * factor
                if recompute:
                    totals[2] += value
        return tuple(totals)

    def _ring_layer_loop(self, hidden_states, ctx):
        if not getattr(self.config, "loop_ring_separate_edges", False):
            return super()._ring_layer_loop(hidden_states, ctx)
        call = self.call_roles
        rounds = self.config.loop_iterations
        if not isinstance(call, LoopRingCall) or call.lap not in range(rounds + 2):
            raise ValueError("GLM ring execution requires a call identifying its logical lap")
        lap, chunk = call.lap, self.loop_ring
        role = "prelude" if lap == 0 else "coda" if lap == rounds + 1 else "core"
        if chunk.role != role:
            raise ValueError(f"GLM lap {lap} cannot run the physical {chunk.role} chunk")
        if (
            call.pre_process != (lap == 0 and chunk.round_start)
            or call.post_process != (lap == rounds + 1 and chunk.round_end)
            or (call.mtp_process and not call.post_process)
        ):
            raise ValueError("GLM ring embedding and output roles do not match this call")
        if self._loop_depths is not None:
            raise ValueError("GLM ring depths must be supplied by the pipeline schedule")
        validate_hybrid_loop_inputs(
            packed_seq_params=ctx.packed_seq_params,
            inference_context=ctx.inference_context,
            attention_mask=ctx.attention_mask,
            allow_packed_documents=self.config.loop_exit_criterion == "fixed",
        )
        if ctx.packed_seq_params is not None and self.config.dynamic_context_parallel:
            # DSA resolves the active CP subgroup from the packed metadata. Each
            # physical PP link carries the same local budget, including CP1 packs.
            expected = self.config.max_seqlen_per_dp_cp_rank
            if hidden_states.shape[:2] != (expected, 1):
                raise ValueError(
                    f"Packed ring input has shape {tuple(hidden_states.shape[:2])}, expected ({expected}, 1)"
                )
        self.loop_losses, self.loop_metrics, self.loop_metric_populations = {}, {}, {}
        ctx.grouped_tail_to_skip = None
        sequence, batch = hidden_states.shape[:2]
        depth = rounds if call.depth is None else call.depth
        counts = self._ring_counts.setdefault(call.microbatch, [0, 0, 0, 0])
        if role == "core" and lap > depth:
            if call.online_exit is not None and lap == depth + 1 and call.sparse_banks is not None:
                # The previous lap could not know it was the last consumer before
                # its remote gate ran. Backward closures retain the bank they need.
                del call.sparse_banks[call.microbatch]
            # Keep the interleaved P2P stream in lockstep. Its send may deallocate the
            # result; the received leaf must survive to pass its adjoint straight through.
            return make_viewless_tensor(
                hidden_states.view_as(hidden_states), hidden_states.requires_grad, keep_graph=True
            )
        for value, factor, recompute in self._ring_lap_work(lap, call.checkpoint, sequence, batch):
            self._ring_add_work(value, recompute=recompute, backward_factor=factor)
        label = f"loop_hybrid/mb={call.microbatch}/{chunk.role}/lap={lap}"
        core = chunk.role == "core"
        scope = self._ring_sparse_pass(call) if core else None
        if core and call.round != lap - 1:
            raise ValueError("The GLM core lap and physical pass index disagree")
        with forward_statistics_round(lap, rounds) if core else nullcontext():
            recompute = self._lap_uses_checkpoint(call.checkpoint, core)
            state = self._ring_local_layers(hidden_states, ctx, label, scope, recompute)
            if scope is not None and scope.is_source:
                scope.finish_source()
            if core:
                counts[0] += 1
                if chunk.round_end:
                    key = (call.microbatch, lap - 1)
                    if call.gate_banks is None or key not in call.gate_banks:
                        raise RuntimeError(f"Missing GLM pass input for {key}")
                    entry = call.gate_banks[key]
                    state = self.loop_gates[lap - 1](ring_injection_use(entry), state)
                    if call.online_exit is not None and call.online_exit.gate(
                        call.microbatch, lap, state, padding_mask=ctx.padding_mask
                    ):
                        depth = lap
                    if not entry.grad_mode:
                        entry.release()
        if chunk.round_end and lap < depth:
            if call.gate_banks is None:
                raise RuntimeError("GLM ring requires a step-local gate input bank")
            state = _publish_gate_input(state, call.gate_banks, (call.microbatch, lap))
        if lap == 0 and chunk.round_start:
            counts[1] += 1
        if call.post_process:
            counts[2] += 1
            if call.model_work is not None:
                self.set_ring_model_work(sequence, batch, call.model_work)
            self._ring_head_metrics(state.device, sequence, batch, depth=depth)
        return state

    def _ring_local_layers(self, state, ctx, label, scope, recompute):
        """Bind the same per-pass KV and selection context inside original and replayed layers."""
        if not len(self.layers):
            # The schedule pseudo-deallocates the outgoing activation after its
            # send. Keep backward's received leaf intact on a pass-through stage.
            return make_viewless_tensor(state.view_as(state), state.requires_grad, keep_graph=True)
        full = getattr(self.config, "recompute_granularity", None) == "full"
        if recompute and torch.is_grad_enabled() and (full or self.config.loop_recompute_block_stride):
            return self._checkpointed_blocks(
                state, None, 0, len(self.layers), ctx, label, cross_loop_pass=scope,
                block_stride=(
                    self.config.recompute_num_layers
                    if full
                    else self.config.loop_recompute_block_stride
                ),
            )

        def run(x):
            with scope.activate() if scope is not None else nullcontext():
                return self._run_range(x, 0, len(self.layers), ctx, label)

        if recompute and torch.is_grad_enabled() and len(self.layers):
            return loop_round_checkpoint(run, state, label=label, round_ends_in_layer=True)
        return run(state)
