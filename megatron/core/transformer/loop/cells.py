# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The cells of the hybrid loop program: prelude, one core round, coda with the head.

Each cell runs the model's own code, in the order of the eager forward
(``HybridModel.forward`` with a ``LoopHybridStack`` decoder):

* ``PreludeCell``: ``HybridModel.embed`` -> ``rope_ctx`` (sweep only) ->
  ``HybridStack._prepare_forward`` (the mHC / wide-residual expansion and the per-forward layer
  context) -> prelude sublayers ``[0, P)``. Its output is the injection ``e``.
* ``CoreRound``: the Huginn round ``loop_norm(core(loop_adapter([s_{r-1}, e])))`` (the stack's
  cat-form adapter, shared by every stream), then the exit criterion. Split into prologue
  (adapter), layers (any core-relative range) and epilogue (loop norm, criterion, forced exits)
  so later block / nested backward items can run ranges of a round.
* ``CodaHeadCell``: coda sublayers -> ``HybridStack._finalize_forward`` (mHC contraction, final
  norm, the MTP multistream) -> ``HybridModel.head_loss`` (MTP and the LM loss) -> the sample's
  loss function.

For one row, prelude + ``d`` core rounds + coda is the eager forward at depth ``d`` op for op,
so per-sample losses equal the eager reference bitwise where the same kernels run. The cells
neither detach nor set the grad mode: the executor does (sweep: grad enabled, inputs detached,
graph dropped; items: fresh leaves, graph kept). The one exception is
``CoreRound.forward_detached``, whose contract is sweep mode: it drops the graph at every
sublayer boundary (same ops, same order as the attached round), so at most one sublayer's
saved activations are alive, and it captures sublayer inputs for the block-checkpoint stash.
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from megatron.core.models.hybrid.hybrid_block import HybridStackLayerContext
from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
    indexer_loss_padding_mask,
)
from megatron.core.transformer.loop.guards import validate_hybrid_loop_inputs
from megatron.core.transformer.loop.program import (
    CODA_PORTS,
    CORE_PORTS,
    PRELUDE_PORTS,
    CoreBackwardPlan,
    CoreBlock,
    LoopPhase,
    LoopProgram,
    Port,
    SampleInputs,
    find_port,
)
from megatron.core.transformer.selective_recompute import request_selective_recompute

__all__ = [
    "CodaHeadCell",
    "CodaOutputs",
    "CoreBackwardPlan",
    "CoreBlock",
    "CoreRound",
    "DetachedForward",
    "PreludeCell",
    "PreludeOutputs",
    "RoundOutputs",
]


def _label(sample: SampleInputs) -> str:
    return f"loop_rsbf/sample={sample.key.global_index}"


def _check_batch(program: LoopProgram, ports: tuple[Port, ...], sample: SampleInputs) -> None:
    for name in ("tokens", "position_ids", "labels", "loss_mask", "attention_mask", "padding_mask"):
        program.check(
            find_port(ports, name, "in"),
            getattr(sample, name),
            rows=sample.rows,
            sequence=sample.sequence,
        )


def _context(sample: SampleInputs) -> HybridStackLayerContext:
    """The sample's layer context (built by the prelude sweep), ready for a layer range."""
    ctx = sample.ctx
    if ctx is None:
        raise ValueError("the sample has no layer context: run the prelude sweep first")
    # The only field a layer call writes; graphs are rejected, so it stays None.
    assert ctx.grouped_tail_to_skip is None, "a grouped CUDA-graph tail is pending"
    return ctx


@dataclass
class PreludeOutputs:
    """The prelude's outputs for one cell call."""

    injection: Tensor  # e: [S, c, nH], act
    crit_state: Optional[Tensor]  # crit_0: [c, k] fp32; None in a recompute of a
    # non-differentiable criterion (nothing to back-propagate through it)


class PreludeCell:
    """Embedding, stack expansion and prelude sublayers: the injection ``e`` of the loop."""

    PORTS = PRELUDE_PORTS

    def __init__(self, program: LoopProgram):
        self.program = program

    def __call__(self, sample: SampleInputs, *, phase: LoopPhase) -> PreludeOutputs:
        """Run the prelude for ``sample``.

        ``SWEEP`` builds and stores the sample's activation-free metadata (decoder padding
        mask, hash token IDs, rotary embedding, layer context) and ``crit_0``. ``RECOMPUTE``
        (the PRELUDE backward item) re-embeds, re-expands (discarding the new context: the
        sweep's is reused, as an eager checkpoint replay reuses the forward's) and recomputes
        ``crit_0`` only for a differentiable criterion.
        """
        program = self.program
        hybrid, stack = program.hybrid, program.stack
        phase = LoopPhase(phase)
        if phase == LoopPhase.REBUILD:
            raise ValueError("the prelude has no rebuild phase")
        if program.check_ports:
            _check_batch(program, self.PORTS, sample)
        label = _label(sample)
        with torch.profiler.record_function(f"{label}/prelude/phase={phase.name.lower()}"):
            decoder_input, decoder_padding_mask, hash_input_ids = hybrid.embed(
                sample.tokens, sample.position_ids, padding_mask=sample.padding_mask
            )
            if phase == LoopPhase.SWEEP:
                rotary_pos_emb = hybrid.rope_ctx(decoder_input, sample.position_ids)
            else:
                _context(sample)
                rotary_pos_emb = sample.rotary_pos_emb
            hidden_states, ctx = stack._prepare_forward(
                decoder_input,
                sample.attention_mask,
                None,
                rotary_pos_emb,
                packed_seq_params=None,
                padding_mask=decoder_padding_mask,
                input_ids=hash_input_ids,
                packed_seq_params_by_layout=None,
                cp_layout_plan=None,
            )
            if phase == LoopPhase.SWEEP:
                validate_hybrid_loop_inputs(
                    packed_seq_params=ctx.packed_seq_params,
                    inference_context=ctx.inference_context,
                    attention_mask=ctx.attention_mask,
                    has_dsv4_attention=stack._has_dsv4_attention,
                )
                sample.decoder_padding_mask = decoder_padding_mask
                sample.hash_input_ids = hash_input_ids
                sample.rotary_pos_emb = rotary_pos_emb
                sample.ctx = ctx
            ctx = _context(sample)
            with ctx.outer_quant_context:
                injection = stack._run_range(
                    hidden_states, 0, program.roles.core_start, ctx, label + "/prelude"
                )
            criterion = program.criterion
            crit_state = None
            if phase == LoopPhase.SWEEP or criterion.differentiable:
                crit_state = criterion.init_state(injection, sample.valid())
                if not criterion.differentiable and crit_state.requires_grad:
                    raise ValueError(
                        f"{type(criterion).__name__} is not differentiable but its init_state "
                        "requires grad (exit-criterion rule 7)"
                    )
        if program.check_ports:
            for name, value in (
                ("rotary_pos_emb", sample.rotary_pos_emb),
                ("injection", injection),
                ("crit_state", crit_state),
            ):
                program.check(
                    find_port(self.PORTS, name, "out"),
                    value,
                    rows=sample.rows,
                    sequence=sample.sequence,
                )
        return PreludeOutputs(injection=injection, crit_state=crit_state)

    @torch.no_grad()
    def build_context(self, sample: SampleInputs) -> None:
        """Fill ``sample``'s activation-free metadata (decoder padding mask, hash token IDs,
        rotary embedding, layer context) without running the prelude: for a sample that
        arrives on an EU other than its home by a multi-EU move.

        ``embed`` returns a given decoder input untouched, and ``rope_ctx`` and
        ``_prepare_forward`` read only its shape (the expanded stub is discarded), so a zero
        stub of the embedding's shape gives the sweep's context: every field is a function of
        the tokens, position IDs, masks and the config.
        """
        program = self.program
        hybrid, stack = program.hybrid, program.stack
        if program.check_ports:
            _check_batch(program, self.PORTS, sample)
        # Zeros (never read for values): the discarded mHC expansion sees no NaN / Inf.
        stub = torch.zeros(
            sample.sequence,
            sample.rows,
            program.config.hidden_size,
            dtype=program.dtype,  # the embedding's (residual-stream) dtype, as in the sweep
            device=program.device,
        )
        _, decoder_padding_mask, hash_input_ids = hybrid.embed(
            sample.tokens, sample.position_ids, decoder_input=stub, padding_mask=sample.padding_mask
        )
        rotary_pos_emb = hybrid.rope_ctx(stub, sample.position_ids)
        _, ctx = stack._prepare_forward(
            stub,
            sample.attention_mask,
            None,
            rotary_pos_emb,
            packed_seq_params=None,
            padding_mask=decoder_padding_mask,
            input_ids=hash_input_ids,
            packed_seq_params_by_layout=None,
            cp_layout_plan=None,
        )
        validate_hybrid_loop_inputs(
            packed_seq_params=ctx.packed_seq_params,
            inference_context=ctx.inference_context,
            attention_mask=ctx.attention_mask,
            has_dsv4_attention=stack._has_dsv4_attention,
        )
        sample.decoder_padding_mask = decoder_padding_mask
        sample.hash_input_ids = hash_input_ids
        sample.rotary_pos_emb = rotary_pos_emb
        sample.ctx = ctx


@dataclass
class RoundOutputs:
    """One core round's outputs for one cell call."""

    state: Tensor  # s_r: [S, c, nH], act
    exit: Tensor  # [c] bool, after forcing (R_max) and r_min
    crit_state: Tensor  # crit_r: [c, k] fp32
    aux_loss: Optional[Tensor]  # [] fp32, differentiable criteria only


@dataclass
class DetachedForward:
    """The result of ``CoreRound.forward_detached`` (sweep mode, graph dropped)."""

    outputs: Optional[RoundOutputs]  # the epilogue's outputs; None when the call ends early
    captured: dict  # core-relative sublayer l -> its input x_l (detached), l in ``capture``
    trace: tuple  # ``trace=True``: the output of every sublayer run (detached), in order
    # id(tensor) -> its in-place version counter when it was produced, for every captured and
    # traced tensor: a later sublayer of the same forward writing into it in place (allowed by
    # autograd on a detached input) shows as a newer version (``modified_after_capture``).
    versions: dict = dataclasses.field(default_factory=dict)

    def modified_after_capture(self, tensor: Tensor) -> bool:
        """Whether ``tensor`` (captured or traced by this forward) was modified in place after
        it was produced."""
        return tensor._version != self.versions[id(tensor)]  # pylint: disable=protected-access


class CoreRound:
    """One Huginn core round over the layer runner.

    Split into ``prologue`` (adapter), ``layers`` (core-relative ``[start, end)``) and
    ``epilogue`` (loop norm, criterion, forced exits, the after-exit counter) so that block
    items (M8b) and nested items (M8) can run any range with the prologue / epilogue attached
    to the first / last block. ``alive`` (``[c]`` bool, None = every row alive) carries exited
    rows unchanged: per layer (``where(alive, new, old)`` over every stream) and per round
    (state and criterion state); the dead rows are padding for the layers (MoE routing
    statistics).
    """

    PORTS = CORE_PORTS

    def __init__(self, program: LoopProgram):
        self.program = program

    def prologue(self, round_input: Optional[Tensor], injection: Tensor) -> Tensor:
        """``loop_adapter([s_{r-1}, e])`` stream by stream; ``s_0 = 0`` when ``round_input``
        is None (r = 1)."""
        state = torch.zeros_like(injection) if round_input is None else round_input
        return self.program.stack._adapt(state, injection)

    def layers(
        self,
        x: Tensor,
        start: int,
        end: int,
        ctx: HybridStackLayerContext,
        alive: Optional[Tensor],
        label: str,
        trace: Optional[list] = None,
    ) -> Tensor:
        """Core-relative sublayers ``[start, end)`` (stack sublayers ``P + start .. P + end``).

        Without ``alive`` this is the eager round's layer pass. With it, each sublayer's
        output is kept only on the alive rows, and the dead rows are marked as padding. With
        a ``trace`` list, each sublayer runs in its own ``_run_range`` call (the same ops in
        the same order) and its output is appended (attached to the graph).
        """
        roles = self.program.roles
        if not 0 <= start <= end <= roles.core:
            raise ValueError(f"core range [{start}, {end}) outside [0, {roles.core})")
        stack = self.program.stack
        first, last = roles.core_start + start, roles.core_start + end
        if alive is None:
            if trace is None:
                return stack._run_range(x, first, last, ctx, label)
            for index in range(first, last):
                x = stack._run_range(x, index, index + 1, ctx, label)
                trace.append(x)
            return x
        rows = x.shape[1]
        dead = (~alive)[:, None]
        ctx = dataclasses.replace(
            ctx,
            padding_mask=(
                dead.repeat(1, x.shape[0]) if ctx.padding_mask is None else ctx.padding_mask | dead
            ),
        )
        keep = alive.reshape(1, rows, 1)
        for index in range(first, last):
            x = torch.where(keep, stack._run_range(x, index, index + 1, ctx, label), x)
            if trace is not None:
                trace.append(x)
        return x

    def epilogue(
        self,
        x: Tensor,
        round_input: Optional[Tensor],
        *,
        crit_state: Tensor,
        r: Tensor,
        phase: Tensor,
        alive: Optional[Tensor],
        exited: Tensor,
        depth: Optional[Tensor],
        valid: Optional[Tensor],
    ) -> RoundOutputs:
        """Loop norm, round self-carry, criterion step, forced exits and the after-exit count.

        ``exit = (criterion exit | r >= R_max) & r >= r_min`` (dead rows stay exited). A
        sweep call (``phase == 0``) that advances a row the executor marked ``exited`` adds
        that row to ``program.after_exit_violations`` on device; recomputes are legitimate and
        not counted. A dead row (``alive`` False) is only carried: its state and criterion
        state come out unchanged and its layer work is chunk padding, so it is not counted.
        Without ``alive`` (M1: one row per call) every row advances and every ``exited`` row
        counts.
        """
        program = self.program
        state = program.stack.loop_norm(x)
        if alive is not None:
            previous = torch.zeros_like(state) if round_input is None else round_input
            state = torch.where(alive.reshape(1, -1, 1), state, previous)
        step = program.criterion.step(state, crit_state, r, valid, depth)
        exit_bits = (step.exit | torch.ge(r, program.max_rounds)) & torch.ge(r, program.min_rounds)
        new_crit = step.crit_state
        if alive is not None:
            exit_bits = exit_bits | ~alive
            new_crit = torch.where(alive[:, None], new_crit, crit_state)
        advanced = exited.to(torch.int32)
        if alive is not None:
            advanced = advanced * alive.to(torch.int32)
        program.after_exit_violations.add_(
            advanced.sum(dtype=torch.int32) * torch.eq(phase, int(LoopPhase.SWEEP))
        )
        return RoundOutputs(
            state=state, exit=exit_bits, crit_state=new_crit, aux_loss=step.aux_loss
        )

    def __call__(
        self,
        round_input: Optional[Tensor],
        injection: Tensor,
        sample: SampleInputs,
        *,
        r: int,
        phase: LoopPhase,
        crit_state: Tensor,
        exited: Tensor,
        depth: Optional[Tensor] = None,
        alive: Optional[Tensor] = None,
        block: Optional[CoreBlock] = None,
        trace: Optional[list] = None,
    ) -> Union[RoundOutputs, Tensor]:
        """Run round ``r`` (or one block of it) for ``sample``.

        Args:
            round_input: ``s_{r-1}`` (None at r = 1), or the block input for a block without
                the prologue.
            injection: ``e``, the prelude output.
            r: the round, 1..R_max; passed to the criterion as a device scalar.
            phase: ``SWEEP`` or ``RECOMPUTE`` (a ``REBUILD`` runs through
                ``forward_detached``).
            crit_state: ``crit_{r-1}`` ``[c, k]`` fp32.
            exited: ``[c]`` int32 device flags of rows the executor saw exit.
            depth: ``[c]`` int32 depths for known-depth criteria.
            alive: ``[c]`` bool, None when every row is alive (always in M1).
            block: a block of ``program.core_plan``; None runs the whole round.
            trace: a list that receives the output of every sublayer of the block (attached;
                ``check_recompute``), see ``layers``.

        Returns:
            ``RoundOutputs`` when the call runs the epilogue, else the block output.
        """
        program = self.program
        phase = LoopPhase(phase)
        if phase == LoopPhase.REBUILD:
            raise NotImplementedError("a rebuild runs through forward_detached (sweep mode)")
        r_tensor = program.round_tensor(r)
        phase_tensor = program.phase_tensor(phase)
        if block is None:
            block = CoreBackwardPlan.whole_round(program.roles.core).blocks[0]
        if block.with_epilogue and block.end != program.roles.core:
            raise ValueError(
                f"block {block} runs the epilogue before the end of the core "
                f"({program.roles.core} sublayers)"
            )
        if alive is not None and not (block.with_prologue and block.with_epilogue):
            raise NotImplementedError("row self-carry runs whole rounds only")
        if alive is not None and program.criterion.differentiable:
            raise NotImplementedError("a differentiable criterion runs one row per cell call")
        ctx = _context(sample)
        if program.check_ports:
            self._check_inputs(sample, round_input, injection, crit_state, exited, depth, alive)
        label = f"{_label(sample)}/loop={r}/phase={phase.name.lower()}"
        with torch.profiler.record_function(label), ctx.outer_quant_context:
            x = self.prologue(round_input, injection) if block.with_prologue else round_input
            x = self.layers(x, block.start, block.end, ctx, alive, label, trace)
            if not block.with_epilogue:
                return x
            outputs = self.epilogue(
                x,
                round_input if block.with_prologue else None,
                crit_state=crit_state,
                r=r_tensor,
                phase=phase_tensor,
                alive=alive,
                exited=exited,
                depth=depth,
                valid=sample.valid(),
            )
        if program.check_ports:
            for name in ("state", "exit", "crit_state", "aux_loss"):
                program.check(
                    find_port(self.PORTS, name, "out"),
                    getattr(outputs, name),
                    rows=sample.rows,
                    sequence=sample.sequence,
                )
        return outputs

    def forward_detached(
        self,
        round_input: Optional[Tensor],
        injection: Tensor,
        sample: SampleInputs,
        *,
        r: int,
        phase: LoopPhase,
        capture: Sequence[int] = (),
        end: Optional[int] = None,
        crit_state: Optional[Tensor] = None,
        exited: Optional[Tensor] = None,
        depth: Optional[Tensor] = None,
        trace: bool = False,
        detach_every_sublayer: bool = True,
    ) -> DetachedForward:
        """Sweep mode (phase ``SWEEP`` or ``REBUILD``): the training-path forward of round
        ``r`` with the autograd graph dropped at every sublayer boundary, so at most one
        sublayer's saved activations are alive at a time.

        The values are bitwise those of ``__call__`` (the same sublayer calls in the same
        order; only ``requires_grad`` of the sublayer inputs differs). The caller holds
        ``torch.enable_grad()`` (the training kernel path, D35).

        Args:
            round_input: ``s_{r-1}`` (None at r = 1).
            injection: ``e``.
            r: the round, 1..R_max.
            phase: ``SWEEP`` (the CORE-F sweep) or ``REBUILD`` (the nested item's rebuild).
            capture: core-relative sublayers ``l`` in ``[1, end]`` whose input ``x_l`` (the
                output of sublayer ``l - 1``) is returned in ``captured``.
            end: run core sublayers ``[0, end)`` only, without the epilogue (a rebuild); None
                runs the whole core and the epilogue (needs ``crit_state`` and ``exited``).
            crit_state, exited, depth: the epilogue's inputs (as ``__call__``).
            trace: also return the output of every sublayer run (``check_recompute``).
            detach_every_sublayer: False drops the graph only at the captured sublayers (and
                at the end): a debug setting for bisecting a per-sublayer detach.

        Returns:
            ``DetachedForward``: the epilogue's outputs (None with ``end``), the captured
            inputs and the trace; every tensor detached.
        """
        program = self.program
        phase = LoopPhase(phase)
        if phase not in (LoopPhase.SWEEP, LoopPhase.REBUILD):
            raise ValueError(f"forward_detached runs sweep mode (SWEEP / REBUILD), got {phase!r}")
        core = program.roles.core
        stop = core if end is None else end
        if not 0 <= stop <= core:
            raise ValueError(f"end {end} outside [0, {core}]")
        capture = tuple(sorted(set(capture)))
        if any(not 1 <= l <= stop for l in capture):
            raise ValueError(f"captured sublayers {capture} outside [1, {stop}]")
        if end is None and (crit_state is None or exited is None):
            raise ValueError("a whole-round forward_detached needs crit_state and exited")
        ctx = _context(sample)
        if program.check_ports:
            if end is None:
                self._check_inputs(sample, round_input, injection, crit_state, exited, depth, None)
            else:
                self._check_inputs(sample, round_input, injection, rebuild=True)
        stack, first = program.stack, program.roles.core_start
        label = f"{_label(sample)}/loop={r}/phase={phase.name.lower()}"
        captured, traced, outputs, versions = {}, [], None, {}
        with torch.profiler.record_function(label), ctx.outer_quant_context:
            x = self.prologue(round_input, injection)
            if detach_every_sublayer:
                x = x.detach()
            for l in range(stop):
                x = stack._run_range(x, first + l, first + l + 1, ctx, label)
                keep = l + 1 in capture
                if detach_every_sublayer or keep:
                    x = x.detach()
                if trace:
                    # The captured tensor itself when it is one (a stash shares it).
                    traced.append(x if x.grad_fn is None else x.detach())
                    versions[id(traced[-1])] = traced[-1]._version  # pylint: disable=W0212
                if keep:
                    captured[l + 1] = x
                    versions[id(x)] = x._version  # pylint: disable=protected-access
            if end is None:
                outputs = self.epilogue(
                    x,
                    round_input,
                    crit_state=crit_state,
                    r=program.round_tensor(r),
                    phase=program.phase_tensor(phase),
                    alive=None,
                    exited=exited,
                    depth=depth,
                    valid=sample.valid(),
                )
        if outputs is not None:
            outputs = RoundOutputs(
                state=outputs.state.detach(),
                exit=outputs.exit,
                crit_state=outputs.crit_state.detach(),
                aux_loss=None if outputs.aux_loss is None else outputs.aux_loss.detach(),
            )
            if program.check_ports:
                for name in ("state", "exit", "crit_state", "aux_loss"):
                    program.check(
                        find_port(self.PORTS, name, "out"),
                        getattr(outputs, name),
                        rows=sample.rows,
                        sequence=sample.sequence,
                    )
        return DetachedForward(
            outputs=outputs, captured=captured, trace=tuple(traced), versions=versions
        )

    def _check_inputs(
        self,
        sample,
        round_input,
        injection,
        crit_state=None,
        exited=None,
        depth=None,
        alive=None,
        *,
        rebuild: bool = False,
    ):
        """Validate the input ports; a rebuild (``rebuild=True``) passes only ``round_input``
        and ``injection`` (it runs no epilogue)."""
        program = self.program
        values = [("round_input", round_input), ("injection", injection)]
        if not rebuild:
            values += [
                ("crit_state", crit_state),
                ("exited", exited),
                ("depth", depth),
                ("alive", alive),
            ]
        for name, value in values:
            program.check(
                find_port(self.PORTS, name, "in"), value, rows=sample.rows, sequence=sample.sequence
            )


@dataclass
class CodaOutputs:
    """The coda's outputs for one cell call (the per-sample loss record's source)."""

    loss: Tensor  # [] fp32: the loss function's sum over the rows' tokens (backward seed)
    num_tokens: Tensor  # []: the loss function's token count (the normalizer's addend)
    report: dict  # the loss function's report ('lm loss': [sum, num_tokens])


class CodaHeadCell:
    """Coda sublayers, the stack readout, the model head (MTP and LM loss) and the loss."""

    PORTS = CODA_PORTS

    def __init__(self, program: LoopProgram):
        self.program = program

    def __call__(self, state: Tensor, sample: SampleInputs) -> CodaOutputs:
        """Run the coda and the head on ``s_d`` and apply ``sample.loss_func``.

        ``cp_batch`` is None: at CP = 1 ``head_loss`` hands it only to the MTP layout
        conversion, which needs none. The coda sublayers and the head run inside
        ``request_selective_recompute(program.coda_recompute_modules)`` (D42: the coda and MTP
        attentions checkpoint their up projections, which the item's backward replays); the
        head also inside ``indexer_loss_padding_mask`` of the sample's padding mask, as the
        stack's layers (padded rows do not train the MTP indexers).
        """
        program = self.program
        hybrid, stack = program.hybrid, program.stack
        ctx = _context(sample)
        if program.check_ports:
            _check_batch(program, self.PORTS, sample)
            program.check(
                find_port(self.PORTS, "state", "in"),
                state,
                rows=sample.rows,
                sequence=sample.sequence,
            )
        label = f"{_label(sample)}/coda"
        with (
            torch.profiler.record_function(label),
            request_selective_recompute(program.coda_recompute_modules),
        ):
            with ctx.outer_quant_context:
                hidden_states = stack._run_range(
                    state, program.roles.core_end, len(stack.layers), ctx, label
                )
            decoder_output = stack._finalize_forward(hidden_states)
            if isinstance(decoder_output, tuple):
                hidden_states, mhc_multistream = decoder_output
            else:
                hidden_states, mhc_multistream = decoder_output, None
            with indexer_loss_padding_mask(sample.padding_mask):
                output = hybrid.head_loss(
                    hidden_states,
                    mhc_multistream,
                    input_ids=sample.tokens,
                    position_ids=sample.position_ids,
                    attention_mask=sample.attention_mask,
                    labels=sample.labels,
                    loss_mask=sample.loss_mask,
                    padding_mask=sample.padding_mask,
                    cp_batch=None,
                    rotary_pos_emb=sample.rotary_pos_emb,
                )
            loss, num_tokens, report = sample.loss_func(sample.loss_mask, output)
        if program.check_ports:
            for name, value in (("loss", loss), ("num_tokens", num_tokens)):
                program.check(
                    find_port(self.PORTS, name, "out"),
                    value,
                    rows=sample.rows,
                    sequence=sample.sequence,
                )
        return CodaOutputs(loss=loss, num_tokens=num_tokens, report=report)
