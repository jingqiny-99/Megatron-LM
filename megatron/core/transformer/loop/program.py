# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The hybrid loop as a program of cells, for the round-synchronous breadth-first executor.

A ``HybridModel`` whose decoder is a Huginn ``LoopHybridStack`` is run by RSBF as three cells
over the model's own modules (``cells.py``): the prelude (embedding, the stack expansion and the
prelude sublayers, producing the injection ``e``), one core round (the adapter on
``[state, e]``, the core sublayers and the loop norm) and the coda with the head (coda
sublayers, the stack readout, ``HybridModel.head_loss`` with MTP, and the loss function). This
module holds what the cells and the executor share:

* the port declarations (``Port``): shape, dtype, gradient and owner of every value that
  crosses a cell boundary;
* the per-sample inputs (``SampleKey``, ``SampleInputs``) and the phase of a cell call
  (``LoopPhase``: sweep, recompute with grad, rebuild);
* the backward plan of one core round (``CoreBlock``, ``CoreBackwardPlan``): the whole-round,
  block-checkpoint (stride ``s``) and nested item forms, chosen by config (``build_core_plan``);
* the exit-criterion contract (``ExitCriterion``) and the built-in criteria;
* ``LoopProgram``: the cells, the criterion, the device scalars (phase, round, the after-exit
  violation counter) and the per-(sample, round) backward plan.

``LoopProgram`` owns no parameters and is never stored on the model.
"""

import abc
import dataclasses
import enum
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, Mapping, Optional, Sequence

import torch
from torch import Tensor

from megatron.core.transformer.loop.guards import (
    LoopRoles,
    is_block_boundary,
    rsbf_item_form_violations,
)
from megatron.core.transformer.loop.param_store_binding import named_logical_parameters
from megatron.core.utils import get_attr_wrapped_model

if TYPE_CHECKING:
    from megatron.core.models.hybrid.hybrid_block import HybridStackLayerContext

# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------

PortDtype = Literal["act", "fp32", "float", "int64", "int32", "int", "bool"]
PortOwner = Literal["executor", "state_store", "chain_record", "cell", "program"]


@dataclass(frozen=True)
class Port:
    """One value crossing a cell boundary.

    ``shape`` is symbolic: ``"S"`` sequence, ``"c"`` rows per cell call (chunk), ``"nH"`` the
    residual width (mHC streams x hidden), ``"k"`` the exit-criterion state width, ``"1"`` a
    unit dimension and ``"*"`` any size. ``dtype`` is ``"act"`` (the residual-stream dtype:
    fp32 under ``fp32_residual_connection``, else ``params_dtype``), ``"fp32"``, ``"float"``
    (any floating dtype), ``"int64"``, ``"int32"``, ``"int"`` (any integer dtype) or
    ``"bool"``. ``grad`` says whether a gradient crosses the boundary; a port without one must
    never carry ``requires_grad``. ``optional`` ports may be None. Values must be on the
    program's device.
    """

    name: str
    direction: Literal["in", "out"]
    shape: tuple[str, ...]
    dtype: PortDtype
    grad: bool
    owner: PortOwner
    optional: bool = False


_BATCH_IN = (
    Port("tokens", "in", ("c", "S"), "int64", False, "executor"),
    Port("position_ids", "in", ("c", "S"), "int64", False, "executor"),
    Port("labels", "in", ("c", "S"), "int64", False, "executor"),
    Port("loss_mask", "in", ("c", "S"), "float", False, "executor"),
    Port("attention_mask", "in", ("c", "1", "S", "S"), "bool", False, "executor", optional=True),
    Port("padding_mask", "in", ("c", "S"), "bool", False, "executor", optional=True),
)

PRELUDE_PORTS: tuple[Port, ...] = _BATCH_IN + (
    Port("rotary_pos_emb", "out", ("S", "1", "1", "*"), "float", False, "executor", optional=True),
    Port("injection", "out", ("S", "c", "nH"), "act", True, "state_store"),
    # grad only for a differentiable criterion (contract rule 7).
    Port("crit_state", "out", ("c", "k"), "fp32", True, "state_store", optional=True),
)

CORE_PORTS: tuple[Port, ...] = (
    # None at r = 1: the zero state is built inside the cell.
    Port("round_input", "in", ("S", "c", "nH"), "act", True, "state_store", optional=True),
    Port("injection", "in", ("S", "c", "nH"), "act", True, "state_store"),
    Port("crit_state", "in", ("c", "k"), "fp32", True, "state_store"),
    Port("r", "in", (), "int32", False, "program"),
    Port("phase", "in", (), "int32", False, "program"),
    Port("alive", "in", ("c",), "bool", False, "executor", optional=True),
    Port("exited", "in", ("c",), "int32", False, "executor"),
    Port("depth", "in", ("c",), "int32", False, "executor", optional=True),
    Port("state", "out", ("S", "c", "nH"), "act", True, "state_store"),
    Port("exit", "out", ("c",), "bool", False, "executor"),
    Port("crit_state", "out", ("c", "k"), "fp32", True, "state_store"),
    Port("aux_loss", "out", (), "fp32", True, "cell", optional=True),
)

CODA_PORTS: tuple[Port, ...] = (
    Port("state", "in", ("S", "c", "nH"), "act", True, "state_store"),
    *_BATCH_IN,
    Port("loss", "out", (), "fp32", True, "executor"),
    Port("num_tokens", "out", (), "int", False, "executor"),
)

# Carried between backward items by the executor's chain records.
CHAIN_PORTS: tuple[Port, ...] = (
    Port("grad_state", "in", ("S", "c", "nH"), "act", False, "chain_record", optional=True),
    Port("grad_injection", "in", ("S", "c", "nH"), "act", False, "chain_record", optional=True),
    Port("grad_crit", "in", ("c", "k"), "fp32", False, "chain_record", optional=True),
    Port("expected_state", "in", ("S", "c", "nH"), "act", False, "chain_record", optional=True),
    # Multi-EU: the sample's loss record packed for a chain transfer, int32
    # [2 + 2 K_rep] = (loss fp32 bits, num_tokens, the report values' fp32 bits); it rides the
    # chain until PRELUDE runs at the sample's home EU.
    Port("loss_record", "in", ("*",), "int32", False, "chain_record", optional=True),
)

# Multi-EU sample and item moves (``rsbf_plan`` slot names): the checkpoint ports above
# (``round_input``, ``crit_state``, ``injection``; state store -> state store), the batch ports
# on a sample's first arrival on an EU (executor -> executor), and the device exit flags.
MOVE_PORTS: tuple[Port, ...] = (
    Port("round_input", "in", ("S", "c", "nH"), "act", False, "state_store"),
    Port("crit_state", "in", ("c", "k"), "fp32", False, "state_store"),
    Port("injection", "in", ("S", "c", "nH"), "act", False, "state_store", optional=True),
    # (exited, exit_round) of the sample, int32: a copy of the executor's device flags.
    Port("exit_flags", "in", ("*",), "int32", False, "executor"),
    *_BATCH_IN,
    # A moved CORE item's block-input stash (slots ``block_input:<l>``) and, under
    # check_recompute, the sweep's per-sublayer outputs (``trace:<l>``); the port is the slot
    # name without the ``:<l>`` suffix.
    Port("block_input", "in", ("S", "c", "nH"), "act", False, "state_store", optional=True),
    Port("trace", "in", ("S", "c", "nH"), "act", False, "state_store", optional=True),
)


def find_port(ports: Sequence[Port], name: str, direction: str) -> Port:
    """The port named ``name`` with ``direction`` in ``ports``."""
    for port in ports:
        if port.name == name and port.direction == direction:
            return port
    raise KeyError(f"no {direction} port {name!r}")


# ---------------------------------------------------------------------------
# Samples and phases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleKey:
    """A sample's identity: its position in the global batch of one iteration.

    Keys the counter-based depth draw, so depths do not depend on placement, DP size,
    microbatch size or waves.
    """

    global_index: int
    iteration: int


def global_sample_index(
    microbatch: int, row: int, *, micro_batch_size: int, dp_rank: int, dp_size: int
) -> int:
    """The global batch position of ``row`` of ``microbatch`` on DP rank ``dp_rank``.

    Megatron's pretraining samplers hand DP rank ``q`` contiguous ``micro_batch_size`` slices,
    so microbatch ``j``, row ``b`` of rank ``q`` is sample ``j * MBS * DP + q * MBS + b``.
    """
    if not (0 <= row < micro_batch_size and 0 <= dp_rank < dp_size and microbatch >= 0):
        raise ValueError(
            f"sample index out of range: microbatch {microbatch}, row {row}, "
            f"MBS {micro_batch_size}, DP rank {dp_rank} of {dp_size}"
        )
    return microbatch * micro_batch_size * dp_size + dp_rank * micro_batch_size + row


@dataclass
class SampleInputs:
    """One cell call's batch rows (``c`` rows; ``c = 1`` in the M1 executor). Owned by the
    executor.

    The last four fields are activation-free metadata filled by the prelude sweep and reused
    by every later cell call of the sample (the eager forward builds them once per forward).
    """

    key: SampleKey
    tokens: Tensor
    position_ids: Tensor
    labels: Tensor
    loss_mask: Tensor
    attention_mask: Optional[Tensor]
    padding_mask: Optional[Tensor]
    # (loss_mask, per-token loss) -> (loss sum, num_tokens, report); no per-sample validation.
    loss_func: Callable[[Tensor, Tensor], tuple]
    decoder_padding_mask: Optional[Tensor] = None
    hash_input_ids: Optional[Tensor] = None
    rotary_pos_emb: Optional[Tensor] = None
    ctx: Optional["HybridStackLayerContext"] = None

    @property
    def rows(self) -> int:
        """``c``, the rows of this cell call."""
        return self.tokens.shape[0]

    @property
    def sequence(self) -> int:
        """``S``, the sequence length."""
        return self.tokens.shape[1]

    def valid(self) -> Optional[Tensor]:
        """``[c, S]`` bool, True on non-padding tokens; None without a padding mask."""
        return None if self.padding_mask is None else ~self.padding_mask


class LoopPhase(enum.IntEnum):
    """What a cell call is for; passed to the cells as a device scalar."""

    # Forward round in sweep mode: grad enabled (the training kernel path), graph dropped by
    # the executor; forward statistics count here.
    SWEEP = 0
    # Grad-mode replay of a round inside a backward item; graph kept; statistics gated.
    RECOMPUTE = 1
    # Nested item (M8): sweep mode inside the backward item, filling the item's layer-input
    # stash; statistics gated.
    REBUILD = 2


# ---------------------------------------------------------------------------
# Backward plans of one core round
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoreBlock:
    """Core-relative sublayers ``[start, end)`` run as one unit of a backward item.

    ``input_source`` names where the block input comes from: the round input ``s_{r-1}``, a
    block-input stash (M8b) or a layer-input stash (M8). The prologue (adapter) runs before
    the first block of a round and the epilogue (loop norm, criterion) after the last.
    """

    start: int
    end: int
    input_source: Literal["round_input", "block_stash", "layer_stash"]
    with_prologue: bool
    with_epilogue: bool

    def __post_init__(self):
        if not 0 <= self.start < self.end:
            raise ValueError(f"a core block needs 0 <= start < end, got [{self.start}, {self.end})")
        if self.with_prologue and not (self.start == 0 and self.input_source == "round_input"):
            raise ValueError("only a block from sublayer 0 on the round input runs the prologue")


ItemForm = Literal["whole_round", "block", "nested"]


@dataclass(frozen=True)
class CoreBackwardPlan:
    """How the backward item of one (sample, round) replays the round.

    The sweep of that round reads ``stashed_block_inputs()`` (what to stash) and the item
    runs ``blocks`` in reverse, both from ``LoopProgram.core_plan``, so they cannot disagree.

    Three forms (``form``), built by the constructors below:

    * whole round (``whole_round``): one block over the core from the round input ``s_{r-1}``;
    * block-checkpoint with stride ``s`` (``strided``): blocks of ``s`` core sublayers (the
      last may be shorter); the sweep stashes the input of every later block, and the item
      recomputes and back-propagates the blocks in reverse, each from its stashed input;
    * nested (``nested``, ``rebuild``): the item first rebuilds ``[0, rebuild_end)`` without
      keeping a graph, filling an item-local stash with the inputs of every later block, then
      runs the blocks in reverse as the block item does.
    """

    blocks: tuple[CoreBlock, ...]
    rebuild: bool = False

    def __post_init__(self):
        if not self.blocks:
            raise ValueError("a core backward plan needs at least one block")
        first, last = self.blocks[0], self.blocks[-1]
        if not first.with_prologue:
            raise ValueError("the first block must start the round from its round input")
        if not last.with_epilogue:
            raise ValueError("the last block must end the round (epilogue)")
        for before, after in zip(self.blocks, self.blocks[1:]):
            if (
                before.end != after.start
                or before.with_epilogue
                or after.with_prologue
                or after.input_source == "round_input"
            ):
                raise ValueError(f"blocks {before} and {after} do not chain")
        # The sweep stashes block inputs of a plan without rebuild; the item's rebuild fills
        # the layer stash of a plan with one. A plan never mixes the two.
        expected = "layer_stash" if self.rebuild else "block_stash"
        sources = sorted({block.input_source for block in self.blocks[1:]} - {expected})
        if sources:
            raise ValueError(
                "mixed block input sources: every block after the first of a "
                f"{'rebuild' if self.rebuild else 'non-rebuild'} plan reads {expected!r}, "
                f"got {sources}"
            )
        if self.rebuild and len(self.blocks) < 2:
            raise ValueError("a rebuild plan needs at least two blocks")

    @property
    def core_sublayers(self) -> int:
        """The core sublayers the plan covers: ``[0, blocks[-1].end)``."""
        return self.blocks[-1].end

    @property
    def form(self) -> ItemForm:
        """``"nested"`` with a rebuild, else ``"whole_round"`` for one block, else
        ``"block"``."""
        if self.rebuild:
            return "nested"
        return "whole_round" if len(self.blocks) == 1 else "block"

    @property
    def stride(self) -> int:
        """The block length in core sublayers: the first block's (the last may be shorter)."""
        return self.blocks[0].end

    def stashed_block_inputs(self) -> tuple[int, ...]:
        """Core-relative sublayers whose input the sweep must stash (the block form's block
        starts after 0); none for the whole-round and nested forms."""
        return tuple(block.start for block in self.blocks if block.input_source == "block_stash")

    def rebuilt_layer_inputs(self) -> tuple[int, ...]:
        """Core-relative sublayers whose input the nested item's rebuild stores (its block
        starts after 0); none without a rebuild."""
        return tuple(block.start for block in self.blocks if block.input_source == "layer_stash")

    @property
    def rebuild_end(self) -> int:
        """The rebuild runs the prologue and core sublayers ``[0, rebuild_end)``: up to the
        last block's start, the last input it must store (the last block's output feeds
        nothing the item needs). 0 without a rebuild."""
        return self.blocks[-1].start if self.rebuild else 0

    def forward_units(self) -> float:
        """Core forward passes per sample-round, in units of one round's forward: the sweep
        (1), the item's recompute (1) and its backward (2), plus the nested rebuild's
        ``rebuild_end / L``."""
        return 4.0 + self.rebuild_end / self.core_sublayers

    @staticmethod
    def whole_round(core_sublayers: int) -> "CoreBackwardPlan":
        """One block over the whole core from the round input: the block item with stride
        ``s = L_core``."""
        return CoreBackwardPlan(
            blocks=(
                CoreBlock(
                    start=0,
                    end=core_sublayers,
                    input_source="round_input",
                    with_prologue=True,
                    with_epilogue=True,
                ),
            )
        )

    @staticmethod
    def strided(core_sublayers: int, stride: int) -> "CoreBackwardPlan":
        """The block-checkpoint plan: blocks ``[j s, min((j + 1) s, L))`` for ``j = 0 ..
        ceil(L / s) - 1``; block 0 from the round input with the prologue, later blocks from
        the sweep's block-input stash, the last with the epilogue. ``1 <= stride <= L``;
        ``strided(L, L) == whole_round(L)``."""
        if not 1 <= stride <= core_sublayers:
            raise ValueError(
                f"a block stride must be in 1..{core_sublayers} core sublayers, got {stride}"
            )
        blocks = []
        for start in range(0, core_sublayers, stride):
            end = min(start + stride, core_sublayers)
            blocks.append(
                CoreBlock(
                    start=start,
                    end=end,
                    input_source="round_input" if start == 0 else "block_stash",
                    with_prologue=start == 0,
                    with_epilogue=end == core_sublayers,
                )
            )
        return CoreBackwardPlan(blocks=tuple(blocks))

    @staticmethod
    def nested(core_sublayers: int, stride: int = 1) -> "CoreBackwardPlan":
        """The nested plan: ``strided(L, stride)`` with every later block read from the
        rebuild's layer stash. ``1 <= stride < L`` (a nested plan has at least two blocks);
        ``stride = 1`` is per sublayer, ``stride = 2`` per attention + MLP layer."""
        if not 1 <= stride < core_sublayers:
            raise ValueError(
                f"a nested stride must be in 1..{core_sublayers - 1} core sublayers (at least "
                f"two blocks), got {stride}"
            )
        strided = CoreBackwardPlan.strided(core_sublayers, stride)
        blocks = (strided.blocks[0],) + tuple(
            dataclasses.replace(block, input_source="layer_stash") for block in strided.blocks[1:]
        )
        return CoreBackwardPlan(blocks=blocks, rebuild=True)


def build_core_plan(config, roles: LoopRoles) -> CoreBackwardPlan:
    """The backward plan a config asks for on a core of ``roles.core`` sublayers:
    ``loop_rsbf_item_form`` ``whole_round`` / ``block`` (stride ``loop_rsbf_block_stride``)
    / ``nested`` (stride default 1). The fields are read with their ``TransformerConfig``
    defaults when a duck-typed config does not define them. Raises ``ValueError`` naming the
    broken item-form rules (``guards.rsbf_item_form_violations``) against the built core."""
    form = getattr(config, "loop_rsbf_item_form", "whole_round")
    stride = getattr(config, "loop_rsbf_block_stride", None)
    violations = rsbf_item_form_violations(form, stride, roles.core)
    if violations:
        raise ValueError(
            f"LoopProgram does not support: {', '.join(violations)} (loop_rsbf_item_form="
            f"{form!r}, loop_rsbf_block_stride={stride}, {roles.core} core sublayers)"
        )
    if form == "block":
        return CoreBackwardPlan.strided(roles.core, stride)
    if form == "nested":
        return CoreBackwardPlan.nested(roles.core, 1 if stride is None else stride)
    return CoreBackwardPlan.whole_round(roles.core)


def block_boundaries_split_pairs(
    plan: CoreBackwardPlan, layer_symbols: Sequence[str], roles: LoopRoles
) -> tuple[int, ...]:
    """Core-relative block starts of ``plan`` that are not on a block boundary of the pattern
    (``guards.is_block_boundary``), i.e. that separate an attention sublayer from its MLP /
    MoE sublayer. Harmless in eager execution (every sublayer call is a function of the
    residual state and the per-forward context only); reported, since a per-pair graph
    needs pair-aligned block starts."""
    symbols = list(layer_symbols)
    return tuple(
        block.start
        for block in plan.blocks[1:]
        if not is_block_boundary(symbols, roles.core_start + block.start)
    )


# ---------------------------------------------------------------------------
# Exit criteria
# ---------------------------------------------------------------------------


def valid_token_mean(x: Tensor, valid: Optional[Tensor]) -> Tensor:
    """Per-row mean over the valid tokens of ``x`` ``[S, c, ...]``, in fp32: ``[c, ...]``.

    ``valid`` is ``[c, S]`` bool (True on non-padding tokens) or None (a plain mean over S).
    Padded positions are selected away with ``torch.where``, so their values, non-finite ones
    included, never reach the result (``ExitCriterion`` rule 8); a row without a valid token
    gives 0.
    """
    x = x.float()
    if valid is None:
        return x.mean(0)
    if valid.dtype != torch.bool or valid.shape != (x.shape[1], x.shape[0]):
        raise ValueError(
            f"valid must be [c, S] = {[x.shape[1], x.shape[0]]} bool, got "
            f"{list(valid.shape)} {valid.dtype}"
        )
    mask = valid.t().reshape(*valid.t().shape, *([1] * (x.dim() - 2)))
    count = mask.sum(0).to(torch.float32).clamp_min(1.0)
    return torch.where(mask, x, torch.zeros((), dtype=x.dtype, device=x.device)).sum(0) / count


@dataclass(frozen=True)
class CriterionStep:
    """One round of an exit criterion, per row of the cell call."""

    exit: Tensor  # [c] bool, device, before the program's forcing (R_max, r_min)
    crit_state: Tensor  # [c, k] fp32
    aux_loss: Optional[Tensor] = None  # [] fp32, differentiable criteria only


class ExitCriterion(abc.ABC):
    """Per-row exit decision of the Huginn loop (the contract the executor relies on).

    1. Per row: ``step`` may reduce over S and the hidden width, never across rows.
    2. fp32, deterministic, static shapes: ``crit_state`` is fp32 ``[c, k]``; no RNG.
    3. The program forces the exit at ``R_max`` and holds it before ``r_min``; criteria do
       not.
    4. ``crit_state`` is checkpointed every round with the round's state; a replay of round
       ``r`` starts from ``crit_{r-1}``.
    5. A differentiable criterion (a halting head) trains through ``aux_loss`` per round.
    6. ``step`` sees the phase only through the core cell.
    7. ``init_state`` may be differentiable (in ``e`` and the criterion's parameters) only when
       ``differentiable``; otherwise it must not require grad.
    8. Reductions over S use only the ``valid`` tokens of the row (``valid_token_mean``),
       selecting with ``torch.where`` rather than multiplying by the mask, so padded positions
       with any value, non-finite ones included, cannot change ``exit``, ``crit_state`` or
       ``aux_loss``; ``valid`` None means every token is valid.
    """

    state_width: int = 0
    differentiable: bool = False

    def known_depths(self, keys: Sequence[SampleKey]) -> Optional[list[int]]:
        """Host-side depths, when known at the prelude (no per-round sync), else None."""
        return None

    def init_state(self, injection: Tensor, valid: Optional[Tensor]) -> Tensor:
        """``crit_0``: ``[c, k]`` fp32 from the injection ``[S, c, nH]``."""
        return torch.zeros(
            injection.shape[1], self.state_width, dtype=torch.float32, device=injection.device
        )

    @abc.abstractmethod
    def step(
        self,
        state: Tensor,
        crit_state: Tensor,
        r: Tensor,
        valid: Optional[Tensor],
        depth: Optional[Tensor],
    ) -> CriterionStep:
        """Exit bits and the new criterion state after round ``r`` (a device scalar)."""


class FixedDepthCriterion(ExitCriterion):
    """Every sample runs ``rounds`` rounds (``loop_exit_criterion='fixed'``)."""

    def __init__(self, rounds: int):
        if rounds < 1:
            raise ValueError(f"fixed depth must be >= 1, got {rounds}")
        self.rounds = rounds

    def known_depths(self, keys: Sequence[SampleKey]) -> list[int]:
        """``rounds`` for every sample."""
        return [self.rounds] * len(keys)

    def step(self, state, crit_state, r, valid, depth):
        """Exit once ``r >= rounds``."""
        rows = state.shape[1]
        return CriterionStep(
            exit=torch.ge(r, self.rounds).reshape(1).expand(rows), crit_state=crit_state
        )


class _KnownDepthCriterion(ExitCriterion):
    """Exit at a per-sample depth known on the host; the device sees it through ``depth``."""

    def step(self, state, crit_state, r, valid, depth):
        """Exit once ``r >= depth``."""
        if depth is None:
            raise ValueError(f"{type(self).__name__} needs the depth port")
        return CriterionStep(exit=torch.ge(r, depth), crit_state=crit_state)


class PrescribedDepthCriterion(_KnownDepthCriterion):
    """Given depths per global sample index (tests and oracles; Python API only)."""

    def __init__(self, depths: Mapping[int, int]):
        if any(depth < 1 for depth in depths.values()):
            raise ValueError(f"prescribed depths must be >= 1, got {dict(depths)}")
        self.depths = dict(depths)

    def known_depths(self, keys: Sequence[SampleKey]) -> list[int]:
        """The prescribed depth of each key's global index."""
        return [self.depths[key.global_index] for key in keys]


_MASK64 = (1 << 64) - 1


def splitmix64(value: int) -> int:
    """One SplitMix64 step (Steele et al.): a bijective 64-bit mix."""
    z = (value + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def counter_seed(seed: int, iteration: int, global_index: int) -> int:
    """A 64-bit generator seed for one (seed, iteration, sample): chained SplitMix64."""
    state = splitmix64(seed & _MASK64)
    state = splitmix64(state ^ (iteration & _MASK64))
    return splitmix64(state ^ (global_index & _MASK64))


class SampledDepthCriterion(_KnownDepthCriterion):
    """Per-sample depth ``d ~ probs`` (``P(d = 1..R_max)``), drawn on the host from a
    counter-based generator keyed by (seed, iteration, global index): no device sync, and
    independent of placement, DP size, microbatch size and waves. Clamped to ``>= min_rounds``.
    """

    def __init__(
        self,
        probs: Sequence[float],
        seed: int,
        min_rounds: int = 1,
        max_rounds: Optional[int] = None,
    ):
        # ``max_rounds`` (R_max) None: the probabilities define it.
        validate_depth_probs(probs, len(probs) if max_rounds is None else max_rounds)
        self.probs = torch.tensor(list(probs), dtype=torch.float64)
        self.seed = seed
        self.min_rounds = min_rounds

    def known_depths(self, keys: Sequence[SampleKey]) -> list[int]:
        """One draw per key."""
        depths = []
        for key in keys:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(counter_seed(self.seed, key.iteration, key.global_index))
            depth = int(torch.multinomial(self.probs, 1, generator=generator)) + 1
            depths.append(max(depth, self.min_rounds))
        return depths


def validate_depth_probs(probs: Optional[Sequence[float]], max_rounds: int) -> None:
    """``probs`` must be ``max_rounds`` nonnegative values summing to 1 (within 1e-6)."""
    if probs is None or len(probs) != max_rounds:
        raise ValueError(f"loop depth probabilities need {max_rounds} entries, got {probs}")
    if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-6:
        raise ValueError(f"loop depth probabilities must be nonnegative and sum to 1: {probs}")


def build_exit_criterion(config) -> ExitCriterion:
    """The criterion a config asks for: ``fixed`` (``loop_iterations`` rounds) or ``sampled``.

    The RSBF config fields (``loop_exit_criterion``, ``loop_depth_probs``, ``loop_depth_seed``,
    ``loop_min_rounds``) are read with their ``TransformerConfig`` defaults when a duck-typed
    config does not define them.
    """
    kind = getattr(config, "loop_exit_criterion", "fixed")
    probs = getattr(config, "loop_depth_probs", None)
    if kind == "fixed":
        if probs is not None:
            raise ValueError("loop_depth_probs is set but loop_exit_criterion is 'fixed'")
        return FixedDepthCriterion(config.loop_iterations)
    if kind == "sampled":
        return SampledDepthCriterion(
            probs,
            getattr(config, "loop_depth_seed", 0),
            min_rounds=getattr(config, "loop_min_rounds", 1),
            max_rounds=config.loop_iterations,
        )
    raise ValueError(f"unknown loop_exit_criterion {kind!r}")


# ---------------------------------------------------------------------------
# The program
# ---------------------------------------------------------------------------

_TORCH_DTYPES = {
    "fp32": (torch.float32,),
    "int64": (torch.int64,),
    "int32": (torch.int32,),
    "bool": (torch.bool,),
}


def _eager_only_options(config) -> list[str]:
    """Config options the eager loop honours and the cells would silently ignore (the design's
    RSBF guards R4, R5, R6 and R9; ``guards.validate_rsbf_config`` takes them over in S2.3)."""
    options = []
    if config.loop_bptt_iterations:
        options.append("truncated BPTT under RSBF (loop_bptt_iterations > 0)")
    if config.loop_sample_depth:
        options.append(
            "loop_sample_depth under RSBF (per-step depth; use loop_exit_criterion='sampled')"
        )
    if config.loop_recompute:
        options.append("loop_recompute under RSBF (RSBF always checkpoints round inputs)")
    if config.freeze_base_model_for_mtp:
        options.append("freeze_base_model_for_mtp under RSBF")
    return options


def _coda_dsv4_attentions(hybrid, stack) -> list:
    """The ``DSv4HybridAttention`` calls a D42 request reaches, in run order: the attentions of
    the coda sublayers, then those of each MTP depth's layer (a repeated MTP layer once per
    depth)."""
    # Lazy: the attention module imports the CSA kernels.
    from megatron.core.transformer.experimental_attention_variant import (
        deepseek_v4_hybrid_attention,
    )

    attention = deepseek_v4_hybrid_attention.DSv4HybridAttention
    parts = list(stack.layers[stack.loop_roles.core_end :])
    mtp = getattr(hybrid, "mtp", None)
    depths = (getattr(stack.config, "mtp_num_layers", None) or 0) if mtp is not None else 0
    for depth in range(depths):
        parts.append(mtp.layers[0 if getattr(mtp, "mtp_use_repeated_layer", False) else depth])
    return [module for part in parts for module in part.modules() if isinstance(module, attention)]


def _coda_recompute_request(config, hybrid, stack) -> frozenset:
    """D42: ``{"mla_up_proj"}`` when the config asks for the coda up-projection recompute; a
    ``ValueError`` when no DSv4 attention in the coda sublayers or the MTP layers could honour
    it."""
    if not getattr(config, "loop_rsbf_coda_recompute_mla_up_proj", False):
        return frozenset()
    if not _coda_dsv4_attentions(hybrid, stack):
        raise ValueError(
            "LoopProgram does not support: loop_rsbf_coda_recompute_mla_up_proj without a DSv4 "
            "attention in the coda sublayers or the MTP layers (nothing to recompute)"
        )
    return frozenset({"mla_up_proj"})


class LoopProgram:
    """The loop model as cells over one ``HybridModel`` with a Huginn ``LoopHybridStack``.

    Owns the cells, the exit criterion, the backward plan per (sample, round) and the device
    scalars: the phase and round tensors handed to the cells and the after-exit violation
    counter (a phase-0 core call on a row the executor marked exited). It has no parameters:
    every module it runs belongs to the model, so DDP and checkpoints see nothing new. Not
    stored on the model (deepcopy / EMA / modelopt clones of the model must not carry it), and
    it holds the model weakly, so a cached program does not keep a freed model alive.

    Args:
        model: the model, possibly wrapped (DDP, ``Float16Module``); the cells call the
            unwrapped ``HybridModel``'s stage methods.
        check_ports: validate shape, dtype and ``requires_grad`` of every port on every cell
            call (tests; no host cost when off).
        exit_criterion: overrides the criterion the config asks for.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        check_ports: bool = False,
        exit_criterion: Optional[ExitCriterion] = None,
    ):
        # Lazy imports: the stack imports the model package, the cells import this module.
        from megatron.core.transformer.loop.cells import CodaHeadCell, CoreRound, PreludeCell
        from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack

        hybrid = get_attr_wrapped_model(model, "head_loss", return_model_obj=True)
        stack = getattr(hybrid, "decoder", None)
        if not isinstance(stack, LoopHybridStack):
            raise ValueError(
                "LoopProgram needs a HybridModel whose decoder is a LoopHybridStack, got "
                f"{type(stack).__name__}"
            )
        config = stack.config
        if config.loop_architecture != "huginn":
            raise ValueError(
                "LoopProgram runs Huginn loops only (MoR stays on the eager path), got "
                f"loop_architecture={config.loop_architecture!r}"
            )
        if not (hybrid.pre_process and hybrid.post_process):
            raise ValueError("LoopProgram needs the whole model on one stage (PP = 1)")
        unsupported = _eager_only_options(config)
        if unsupported:
            raise ValueError(f"LoopProgram does not support: {', '.join(unsupported)}")
        # Weak: an executor caching this program must not keep the model alive.
        self._hybrid = weakref.ref(hybrid)
        self.roles = stack.loop_roles
        self.config = config
        self.num_streams = stack.loop_num_streams
        self.hidden_width = self.num_streams * config.hidden_size
        self.max_rounds = config.loop_iterations
        self.min_rounds = config.loop_min_rounds
        if not 1 <= self.min_rounds <= self.max_rounds:
            raise ValueError(
                f"loop_min_rounds must be in 1..{self.max_rounds}, got {self.min_rounds}"
            )
        self.device = next(stack.parameters()).device
        # The residual-stream ("act" port) dtype.
        self.dtype = torch.float32 if config.fp32_residual_connection else config.params_dtype
        self.check_ports = check_ports
        self.criterion = (
            exit_criterion if exit_criterion is not None else build_exit_criterion(config)
        )
        if isinstance(self.criterion, SampledDepthCriterion) and (
            len(self.criterion.probs) != self.max_rounds
        ):
            raise ValueError(
                f"the sampled-depth criterion has {len(self.criterion.probs)} depth "
                f"probabilities, the loop R_max = {self.max_rounds}"
            )
        self._phases = {
            phase: torch.tensor(int(phase), dtype=torch.int32, device=self.device)
            for phase in LoopPhase
        }
        self._rounds = {
            r: torch.tensor(r, dtype=torch.int32, device=self.device)
            for r in range(1, self.max_rounds + 1)
        }
        self.after_exit_violations = torch.zeros((), dtype=torch.int32, device=self.device)
        # The CORE item form: uniform over (sample, round) until a planner chooses per round.
        self._core_plan = build_core_plan(config, self.roles)
        self._coda_recompute_modules = _coda_recompute_request(config, hybrid, stack)
        self.prelude = PreludeCell(self)
        self.core = CoreRound(self)
        self.coda = CodaHeadCell(self)

    @property
    def hybrid(self):
        """The unwrapped ``HybridModel`` (held weakly)."""
        hybrid = self._hybrid()
        if hybrid is None:
            raise RuntimeError("the model of this LoopProgram has been freed")
        return hybrid

    @property
    def stack(self):
        """The model's ``LoopHybridStack`` decoder."""
        return self.hybrid.decoder

    def phase_tensor(self, phase: LoopPhase) -> Tensor:
        """The cached 0-dim int32 device tensor of ``phase``."""
        return self._phases[LoopPhase(phase)]

    def round_tensor(self, r: int) -> Tensor:
        """The cached 0-dim int32 device tensor of round ``r`` (1..R_max)."""
        if r not in self._rounds:
            raise ValueError(f"round {r} outside 1..{self.max_rounds}")
        return self._rounds[r]

    def reset_counters(self) -> None:
        """Zero the after-exit violation counter (once per step)."""
        self.after_exit_violations.zero_()

    def known_depths(self, keys: Sequence[SampleKey]) -> Optional[list[int]]:
        """The criterion's host-side depths after the program's forcing (``r_min``, ``R_max``),
        i.e. the first round whose exit bit the core cell sets; None for online criteria."""
        depths = self.criterion.known_depths(keys)
        if depths is None:
            return None
        if any(depth < 1 for depth in depths):
            raise ValueError(f"exit criterion returned a depth < 1: {depths}")
        return [min(max(depth, self.min_rounds), self.max_rounds) for depth in depths]

    def core_plan(self, sample: SampleInputs, r: int) -> CoreBackwardPlan:
        """The backward plan of ``sample``'s round ``r``: read by the sweep (what to stash) and
        by the CORE item (what to replay). The config's plan (``uniform_core_plan``) for every
        (sample, round)."""
        plan = self._core_plan
        if plan.core_sublayers != self.roles.core:
            raise ValueError(
                f"the core plan covers {plan.core_sublayers} of {self.roles.core} core sublayers"
            )
        return plan

    @property
    def uniform_core_plan(self) -> CoreBackwardPlan:
        """The config's backward plan, the object ``core_plan`` returns for every (sample,
        round): CORE-B lockstep nulls follow it, so the executor checks that the plan it read
        is this one."""
        return self._core_plan

    @property
    def item_form(self) -> ItemForm:
        """The CORE item form of the config's plan."""
        return self._core_plan.form

    @property
    def coda_recompute_modules(self) -> frozenset:
        """Modules the CODA item asks to recompute in its backward (D42: ``{"mla_up_proj"}``
        with ``loop_rsbf_coda_recompute_mla_up_proj``, else empty)."""
        return self._coda_recompute_modules

    def coda_recompute_attentions(self) -> list:
        """The attention calls the CODA item's ``coda_recompute_modules`` request reaches (the
        DSv4 attentions of the coda sublayers and of each MTP depth, in run order); empty
        without a request."""
        if not self._coda_recompute_modules:
            return []
        return _coda_dsv4_attentions(self.hybrid, self.stack)

    @property
    def param_store(self):
        """The loop parameter store installed on the model (``loop/param_store.py``), or None."""
        # Lazy: the store imports this module.
        from megatron.core.transformer.loop.param_store import find_param_store

        return find_param_store(self.hybrid)

    def core_parameter_overlap(self) -> list[str]:
        """Names of parameters used by more than one core sublayer (none for the patterns the
        loop builds). The block and nested items rely on this: each core parameter then gets
        one gradient addition per item in every form. A parameter store's hidden compute
        parameters are included (``named_logical_parameters``)."""
        stack, roles = self.stack, self.roles
        owners: dict = {}
        names: dict = {}
        for index in range(roles.core_start, roles.core_end):
            for name, param in named_logical_parameters(
                stack.layers[index], prefix=f"layers.{index}"
            ):
                owners.setdefault(id(param), set()).add(index)
                names.setdefault(id(param), name)
        return sorted(names[key] for key, layers in owners.items() if len(layers) > 1)

    def check(self, port: Port, value: Optional[Tensor], *, rows: int, sequence: int) -> None:
        """Validate ``value`` against ``port`` for a call with ``rows`` rows of ``sequence``
        tokens. Raises ``ValueError`` naming the port."""
        where = f"port {port.direction} {port.name!r}"
        if value is None:
            if port.optional:
                return
            raise ValueError(f"{where} is required")
        if not isinstance(value, Tensor):
            raise ValueError(f"{where} must be a tensor, got {type(value).__name__}")
        if value.device != self.device:
            raise ValueError(f"{where}: device {value.device} is not {self.device}")
        sizes = {
            "S": sequence,
            "c": rows,
            "nH": self.hidden_width,
            "k": self.criterion.state_width,
            "1": 1,
        }
        expected = tuple(sizes.get(symbol) for symbol in port.shape)
        if len(expected) != value.dim() or any(
            size is not None and size != actual for size, actual in zip(expected, value.shape)
        ):
            raise ValueError(
                f"{where}: shape {tuple(value.shape)} is not {port.shape} = {expected}"
            )
        dtype = value.dtype
        if port.dtype == "act":
            ok = dtype == self.dtype
        elif port.dtype == "float":
            ok = dtype.is_floating_point
        elif port.dtype == "int":
            ok = not dtype.is_floating_point and not dtype.is_complex and dtype != torch.bool
        else:
            ok = dtype in _TORCH_DTYPES[port.dtype]
        if not ok:
            raise ValueError(f"{where}: dtype {dtype} is not {port.dtype}")
        if not port.grad and value.requires_grad:
            raise ValueError(f"{where} carries no gradient but requires grad")
