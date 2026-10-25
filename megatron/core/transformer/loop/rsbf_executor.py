# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The round-synchronous breadth-first (RSBF) executor of the hybrid Huginn loop.

One training step of a ``HybridModel`` whose decoder is a Huginn ``LoopHybridStack``, run as
the cells of ``LoopProgram`` with per-sample exit (``loop_schedule='rsbf'``). Every rank of the
data-parallel group is one execution unit (EU); the group is the step's domain of ``W`` EUs
(``W = 1`` without data parallelism):

1. **Waves.** The step's microbatches are split into ``loop_rsbf_waves`` waves; each wave's rows
   are samples (one row per cell call, ``c = 1``), keyed by their global batch position; the
   wave-global id ``g`` of a sample is its position minus the wave's base, and its home is the
   EU that loaded it.
2. **The plan.** Every rank runs the same host plan (``rsbf_plan.WavePlanner``) on identical
   inputs, so the EUs agree on every step without communicating it: PRELUDE-F sub-rounds at
   home, per forward round the min-move rebalancing (only the surplus above
   ``ceil(active / W)`` moves, ``loop_rsbf_rebalance``) and the CORE-F sub-rounds, and per slot
   a backward phase of typed sub-rounds (CODA, CORE-B, PRELUDE-B) with one item or null per EU
   (bulk-only fill; below ``2 W`` remaining items, ready items move with their checkpoint to
   idle EUs, ``loop_rsbf_tail_item_moves``). An item runs where its checkpoint lives: CODA and
   CORE(g, r) on the EU that swept the round (or its item-move destination), PRELUDE at home.
3. **Forward.** Every sweep runs with grad enabled (the training kernel path) and drops its
   graph at every core sublayer boundary (``CoreRound.forward_detached``: at most one
   sublayer's saved activations live); the round inputs ``s_r`` and ``crit_r`` are
   checkpointed in this EU's ``LoopStateStore``, and so are the block inputs the round's CORE
   item reads (block-checkpoint form). A sample move carries ``s_{r-1}``, ``crit_{r-1}``, the
   injection ``e`` when the destination holds no copy, the device exit flags and, on the
   sample's first arrival on an EU, its batch rows (the arriving EU rebuilds the
   activation-free context with ``PreludeCell.build_context``).
4. **Exits.** Known depths (``fixed``, ``sampled``, prescribed) are host data for every sample of
   the domain's wave: the whole wave is planned without a per-round sync. An online criterion's
   exit bits are copied to the host once per round after the slot's backward work has been
   enqueued, and reduced over the domain with one status exchange (``all_reduce(MAX)`` of a
   ``uint8[G + 1]`` bitmap on the Gloo status group).
5. **Backward chains.** An exited sample's chain is ``CODA`` (coda + head forward and backward,
   the loss record), ``CORE`` for ``r = d..1`` (replay round ``r`` from its checkpoint with
   grad, then backward) and ``PRELUDE`` (replay the prelude, backward into the embedding).
   Every item starts from fresh detached leaves; gradients cross items only through a
   ``ChainRecord``, which travels (one hop, after the successor's placement is decided) to the
   EU of the next item together with the sample's loss record; the record reaches home with
   the PRELUDE item, where the sample's report and token count are kept. ``drain`` runs a
   sample's chain in the slot after its exit; ``sample_order`` runs every chain after the last
   forward round, in sample order.
6. **Finalize.** Once per step, with the local token count ``n_tok_local`` (the sum of the
   home samples' ``num_tokens``; ``finalize_model_grads`` all-reduces it and scales the
   gradients by its inverse).

Hard exit enforcement: the host never sweeps a retired sample; on the device, every sweep of
a row whose exit bit was already set increments ``LoopProgram.after_exit_violations`` (exit
flags travel with every sample move), and each sample's device exit round is compared with the
host plan on the EU that ran its exit sweep. Both counters (and the report-schema errors) are
reduced over the domain (``all_reduce(MAX)`` of a ``[W, 4]`` table) and read at the single
end-of-step host sync, which also carries the step's loss sum for the rerun-state-machine
checks (validated on a host tensor, no further sync, as the mean of the per-microbatch sums so
the spiky-loss history does not scale with the microbatch count) and the EU's padded-token
count; an online round's after-exit flag rides byte ``G`` of the status exchange. Every rank
therefore raises together.

Exit histogram: ``ExecutorStats.exit_counts`` is the step's realized exit-depth histogram
(``counts[d - 1]`` samples of the domain exited after round ``d``), a host reduction of the
plan's depths (identical on every EU, no sync); ``LoopRSBFSchedule.exit_counts`` and
``.padding`` hand the last step's histogram and the EU's padded / issued tokens to the trainer.

Collectives inside a step (``W > 1``): the move phases on the domain group (the
data-parallel group; ``batch_isend_irecv`` on the non-empty pairs or one ``all_to_all_single``
per phase), the MoE dispatch / combine collectives of expert parallelism (EP groups inside the
domain) in every real item and every null, one status exchange per online round on the Gloo
status group, one ``[W, 4]`` flags all-reduce on the domain after the last wave, and finalize's
DP / expert-DP reductions. Every rank executes the same plan, so every rank issues them in the
same order whatever its item counts.

Lockstep MoE (``null_items``): an EU without an item in a typed sub-round runs that type's null:
the sub-round type's MoE calls (prelude, core, or coda + MTP; forward, and for the backward
types a backward of zeros) on an empty routing, so every EP rank issues the same dispatch /
combine sequence and hosts its experts for the other ranks' tokens; nulls bypass the router
(no router statistics). Dense models and EP = 1 issue no collective inside a sub-round, and
their nulls do nothing. Under a static per-rank budget (HybridEP rank capacity) the dispatchers'
overflow flags are reset at the start of a step and reduced over the domain with the end-of-step
flags (``loop_rsbf_over_budget``: every rank raises, or the first rank warns).

The debug plan-digest checks (``check_plan``) gather over the status group.

MoE aux and DSA indexer losses: the router runs once per sample (batch 1) with the loss scale
of a one-microbatch step, so their gradient equals the standard schedule's at MBS 1 at any
MBS (batch invariance); at MBS ``b`` the standard schedule averages them over the ``b`` rows of
a microbatch, a ``1 / b`` smaller coefficient. Their logged values are weighted ``1 / b``
(``forward_statistics_weight``) to match the standard schedule's logs at the same MBS.

CORE item forms (``loop_rsbf_item_form``): the sweep and the CORE item of a round read the same
``LoopProgram.core_plan``, which is the config's plan for every (sample, round)
(``LoopProgram.uniform_core_plan``; CORE-B lockstep nulls follow it,
``null_items.core_b_segments``). The CORE item runs the plan's blocks in reverse: each block is
recomputed with grad from a fresh leaf on its input (block 0: ``s_{r-1}``; later blocks: the
input the sweep stashed, ``LoopStateStore.put_block_input``) and back-propagated at once with
the gradient the next block handed to its input (the last block: ``dL/ds_r``, ``dL/dcrit_r``
and the criterion's ``aux_loss``), so only one block's activations are alive. The whole round is
the one-block plan (stage 3, op for op). Stash entries alias the sweep's activations (no copy;
a view or a non-owning layout is cloned, ``ExecutorStats.stash_copies``); they stay on the EU
that swept the round, which is where its CORE item runs, travel with the item when a tail item
move carries it (``ExecutorStats.stash_moves``) and are freed right after their block's
backward. The nested item stashes nothing in the sweep: it first rebuilds the round's prefix
``[0, rebuild_end)`` in sweep mode inside the item (phase ``REBUILD``: grad enabled, the graph
dropped at every sublayer, statistics gated like a recompute), keeping the inputs of its later
blocks in an item-local layer stash, then runs the blocks in reverse from that stash as the
block item does; each layer-stash entry is dropped right after its block's backward. Graphs are
a later step.

Coda up-projection recompute (D42, ``loop_rsbf_coda_recompute_mla_up_proj``): the CODA item's
coda sublayers and head run inside ``request_selective_recompute({"mla_up_proj"})``
(``CodaHeadCell``), so the DSv4 attentions of the coda and the MTP layers checkpoint their
query / key-value up projections and RoPE and the item's backward replays them (the
``loop recompute flops per token`` report adds them); the core rounds are unaffected.

Statistics of loop rounds: each sweep runs inside ``forward_statistics_round(r, R_max)``, so the
MoE aux / z-loss and indexer-loss trackers also bin what the round records per (round, layer).

The sweep's one-sublayer bound assumes no module keeps a graph-attached tensor past its call:
the allgather MoE dispatcher keeps ``local_probs`` until its next call (one sublayer's graph
stays alive past the detach); the alltoall dispatcher clears it. Values are unaffected.
``measure_memory`` (debug, CUDA) records each cell call's allocator peak and retained bytes
above its entry baseline (``ExecutorStats.memory``), the quantities of the arena rule (one
sublayer live in a sweep, one block in a block item, one block plus the layer stash in a
nested item).
"""

import contextlib
import enum
import functools
import hashlib
import logging
import math
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Literal, Optional, Sequence, Union

import torch
import torch.distributed as dist
from torch import Tensor, nn

from megatron.core.pipeline_parallel.schedules import (
    _build_default_pg_collection,
    set_auxiliary_loss_scales,
)
from megatron.core.transformer.forward_replay import (
    forward_statistics_round,
    forward_statistics_weight,
)
from megatron.core.transformer.loop import flops
from megatron.core.transformer.loop.dispatcher import LoopSampleDispatcher, RSBFGroups
from megatron.core.transformer.loop.guards import validate_rsbf_config, validate_rsbf_data_parallel
from megatron.core.transformer.loop.null_items import LockstepPlan, build_lockstep_plan, run_null
from megatron.core.transformer.loop.program import (
    CHAIN_PORTS,
    MOVE_PORTS,
    CoreBackwardPlan,
    ExitCriterion,
    LoopPhase,
    LoopProgram,
    SampleInputs,
    SampleKey,
    block_boundaries_split_pairs,
    find_port,
    global_sample_index,
)
from megatron.core.transformer.loop.rsbf_plan import (
    BATCH_SLOTS,
    BLOCK_ENTRY,
    BLOCK_INPUT,
    CRIT_STATE,
    EXIT_FLAGS,
    EXPECTED_STATE,
    GRAD_CRIT,
    GRAD_INJECTION,
    GRAD_STATE,
    INJECTION,
    LOSS_RECORD,
    ROUND_INPUT,
    TRACE,
    TRACE_ENTRY,
    FreeInjection,
    MovePhase,
    SubRound,
    SubRoundType,
    Transfer,
    WavePlanner,
    block_slot,
    parse_sublayer_slot,
    trace_slot,
)
from megatron.core.transformer.loop.state_store import LoopStateStore, StoreKey
from megatron.core.transformer.loop_transformer_block import recompute_context
from megatron.core.utils import get_attr_wrapped_model, get_model_config

__all__ = [
    "BackwardItem",
    "ChainRecord",
    "ExecutorStats",
    "ItemKind",
    "LoopMicrobatch",
    "MemoryRecord",
    "LoopRSBFExecutor",
    "LoopRSBFSchedule",
    "RSBF_BATCH_FUNC_ATTR",
    "RSBF_LOSS_VALIDATOR_ATTR",
    "SampleLossRecord",
    "WorkRecord",
    "exit_depth_counts",
    "forward_backward_loop_rsbf",
    "get_loop_rsbf_schedule",
]

logger = logging.getLogger(__name__)

# Attribute of the training script's forward_step_func naming its RSBF batch function
# (``LoopRSBFBatchFunc``): the microbatch of forward_step without the model call.
RSBF_BATCH_FUNC_ATTR = "loop_rsbf_batch_func"
# Optional attribute of the forward_step_func: ``validate(loss)`` run once per step on the
# step's summed loss divided by the microbatch count (a host fp32 tensor: the mean
# per-microbatch loss sum, the value the standard schedule validates per microbatch), e.g. the
# rerun-state-machine NaN / Inf / spiky checks the per-sample loss functions skip.
RSBF_LOSS_VALIDATOR_ATTR = "loop_rsbf_validate_loss"

# Executor messages (tests match on them).
SWEEP_AFTER_EXIT = "core sweep after exit"
PLAN_MISMATCH = "host exit plan != device exits"
REPORT_SCHEMA = "loss function report keys differ from the microbatch's report_keys"
OVER_BUDGET = "static MoE budget exceeded"
UNSUPPORTED = "Loop RSBF does not support: "
# The plan of a (sample, round) is not the program's uniform plan.
NON_UNIFORM_PLAN = "a CORE plan other than the program's uniform plan"
# verify_stash: a stash entry no longer holds the value the sweep captured.
STASH_MODIFIED = "stash entry modified after capture"
# check_recompute: a replayed value differs from the sweep's.
RECOMPUTE_DIFFERS = "check_recompute: the replayed {what} differs from the sweep"

# Columns of the end-of-step flag table (one row per EU, reduced MAX over the domain).
_FLAG_AFTER_EXIT, _FLAG_PLAN_MISMATCH, _FLAG_OVER_BUDGET, _FLAG_REPORT_SCHEMA = range(4)
_FLAG_NAMES = ("after_exit", "plan_mismatch", "over_budget", "report_schema")
# Report entries the executor adds per sample, ``[value summed over the population, population
# size]`` as in loop/reporting.py: the population of every loop metric is the sample's issued
# tokens (FLOPs per token, depth per row, token-layer counts), except ``loop sample moves``
# (moves per sample-round: ``[moves, depth]``).
_LOOP_REPORT_KEYS = (
    "loop forward flops per token",
    "loop backward flops per token",
    "loop recompute flops per token",
    "loop depth",
    "loop active token layers",
    "loop backward token layers",
    "loop sample moves",
)


@dataclass
class LoopMicrobatch:
    """One microbatch for the RSBF schedule (``[b, S]`` rows), without the model call.

    ``loss_func(loss_mask, per_token_loss) -> (loss sum, num_tokens, report)`` must not
    validate (no per-sample host sync); the executor validates the step's loss once.
    ``report_keys`` are the keys of the report the loss function returns (each value a
    2-element fp32 tensor): a sample's loss record is packed by them when it crosses EUs.
    With data parallelism every microbatch of a step must carry an equivalent ``loss_func``
    (the same object, or ``functools.partial`` objects of the same function with identical
    arguments): a CODA item on an EU other than the sample's home calls that EU's own.

    The executor checks this contract (equivalent loss functions, equal ``report_keys``, one
    batch schema) over the rank's own microbatches only: a rank that violates it raises alone,
    while the other ranks wait in their next collective (a hang, not an error on every rank).
    Microbatches that agree within each rank but differ between ranks are caught on every rank
    only with ``check_plan`` (the loss-function descriptor, report keys and batch schema are
    part of the input digest gathered over the status group at wave start).
    """

    tokens: Tensor
    position_ids: Tensor
    labels: Tensor
    loss_mask: Tensor
    attention_mask: Optional[Tensor]
    padding_mask: Optional[Tensor]
    loss_func: Callable[[Tensor, Tensor], tuple]
    report_keys: tuple = ("lm loss",)


LoopRSBFBatchFunc = Callable[[Iterator, nn.Module], LoopMicrobatch]


class ItemKind(enum.Enum):
    """The typed backward items of one sample's chain."""

    CODA = "coda"
    CORE = "core"
    PRELUDE = "prelude"


@dataclass(frozen=True)
class BackwardItem:
    """One backward item: ``sample`` is the wave-global id ``g``; ``round`` is ``d`` for CODA,
    ``r`` for CORE and 0 for PRELUDE. A CORE item carries the plan of its round
    (``LoopProgram.core_plan``), the same plan the round's sweep consulted."""

    kind: ItemKind
    sample: int
    round: int
    plan: Optional[CoreBackwardPlan] = None


@dataclass
class SampleLossRecord:
    """One sample's loss: the per-sample sum and token count of its loss function."""

    key: SampleKey
    depth: int
    loss: Tensor  # detached fp32 per-sample sum
    num_tokens: Tensor  # the loss function's num_tokens
    report: dict


@dataclass
class ChainRecord:
    """What one item hands the next item of the same sample's chain (possibly on another
    EU: its tensors are the chain transfer's slots)."""

    sample: int
    # dL/ds_r for the next CORE item; None after CORE(1).
    grad_state: Optional[Tensor]
    # Running dL/de (activation dtype), summed CORE(d) .. CORE(1) as autograd would.
    grad_injection: Optional[Tensor]
    # dL/dcrit_r (differentiable criteria); after CORE(1) dL/dcrit_0 for the PRELUDE item.
    grad_crit: Optional[Tensor]
    # check_recompute only: the sweep's s_r, taken from the store, compared by CORE(r).
    expected_state: Optional[Tensor] = None
    # The sample's loss record (built by CODA, unpacked into the report at home by PRELUDE).
    loss_record: Optional[SampleLossRecord] = None


@dataclass(frozen=True)
class WorkRecord:
    """FLOPs of one cell call (per-symbol table, forward of one pass; backward = 2 x forward
    for layers, the head's own factors for the output layer / MTP)."""

    kind: Literal["prelude_sweep", "core_sweep", "coda", "prelude_item", "core_item"]
    sample: int  # global sample index
    round: int
    tokens: int
    forward: float
    recompute: float
    backward: float


@dataclass(frozen=True)
class MemoryRecord:
    """``measure_memory``: the allocator bytes of one cell call of this EU, relative to the
    bytes allocated when it started (``base``): its peak (``torch.cuda.max_memory_allocated``)
    and what it left allocated (``retained``; negative when it freed more than it kept)."""

    kind: str  # the plan item kind: prelude_sweep, core_sweep, coda, core, prelude
    sample: int  # wave-global id g
    round: int
    base: int
    peak: int
    retained: int


@dataclass
class ExecutorStats:
    """Host-side record of one step (tests and logging). Plan-derived fields (moves,
    sub-rounds, depths) are identical on every EU; item and work records are this EU's."""

    host_syncs: int = 0  # every executor host sync of the step
    host_syncs_by_wave: list = field(default_factory=list)  # per-round exit reads per wave
    debug_syncs: int = 0  # check_recompute comparisons and plan-digest checks (test-only)
    rounds_by_wave: list = field(default_factory=list)
    depths: dict = field(default_factory=dict)  # SampleKey -> realized depth, every EU's samples
    item_order: list = field(default_factory=list)  # (kind, global index, round), this EU
    store_peaks: list = field(default_factory=list)  # StoreStats per wave
    work: list = field(default_factory=list)  # WorkRecord per cell call of this EU
    loss_records: list = field(default_factory=list)  # SampleLossRecord, home samples in order
    status_exchanges: int = 0  # Gloo exit-bitmap exchanges (online rounds)
    status_exchanges_by_wave: list = field(default_factory=list)
    moves: dict = field(default_factory=lambda: {"sample": 0, "chain": 0, "item": 0})
    moves_by_round: list = field(default_factory=list)  # per wave: sample moves per round
    active_by_round: list = field(default_factory=list)  # per wave: active samples per round
    subrounds: dict = field(default_factory=lambda: {t.value: 0 for t in SubRoundType})
    nulls: dict = field(default_factory=lambda: {t.value: 0 for t in SubRoundType})
    plan_digests: list = field(default_factory=list)  # per wave, the final plan digest
    flagged_ranks: dict = field(default_factory=dict)  # flag name -> EUs that raised it
    null_moe_calls: int = 0  # MoE calls made by this EU's null items
    over_budget_layers: list = field(default_factory=list)  # this EU's overflowed MoE modules
    # Block-input stash entries carried by moved CORE items (plan-derived: every EU).
    stash_moves: int = 0
    # Stash entries the sweep had to copy (a view or a non-owning layout; expected 0).
    stash_copies: int = 0
    # Core-relative block starts of the CORE plan that split an attention + MLP / MoE pair.
    split_pairs: tuple = ()
    # check_recompute="report": max |replayed - sweep| per check name over the step.
    recompute_diffs: dict = field(default_factory=dict)
    # verify_stash: the debug syncs of its comparisons (not in debug_syncs).
    stash_verify_syncs: int = 0
    # Nested items: rebuilds run by this EU, and the most layer-stash entries one item held.
    rebuilds: int = 0
    layer_stash_peak: int = 0
    # measure_memory: one MemoryRecord per cell call of this EU, in execution order.
    memory: list = field(default_factory=list)
    # The realized exit-depth histogram of the step: ``exit_counts[d - 1]`` samples of the
    # domain exited after round ``d`` (d = 1..R_max), over every wave (plan-derived, identical
    # on every EU; sums to the step's samples, ``len(depths)``).
    exit_counts: list = field(default_factory=list)
    # This EU's home samples: padded tokens (``padding_mask``; 0 without one) and issued tokens
    # (the rows' token count) of the step, read with the end-of-step flags.
    padded_tokens: int = 0
    issued_tokens: int = 0


def exit_depth_counts(depths: Iterable[int], max_rounds: int) -> list:
    """The exit-depth histogram ``counts[d - 1]`` (d = 1..``max_rounds``) of the realized
    ``depths`` (host data). A depth outside ``1..max_rounds`` raises (a plan bug)."""
    counts = [0] * max_rounds
    for depth in depths:
        if not 1 <= depth <= max_rounds:
            raise ValueError(f"exit depth {depth} outside 1..R_max = {max_rounds}")
        counts[depth - 1] += 1
    return counts


@contextlib.contextmanager
def _sync_allowed(device: torch.device):
    """Lower ``torch.cuda`` sync-debug mode around an intended host sync."""
    if device.type != "cuda":
        yield
        return
    mode = torch.cuda.get_sync_debug_mode()
    if mode:
        torch.cuda.set_sync_debug_mode(0)
    try:
        yield
    finally:
        if mode:
            torch.cuda.set_sync_debug_mode(mode)


class _DeviceToHost:
    """A device -> host int32 read: on CUDA a non-blocking copy into a pinned buffer on a side
    stream (which first waits for the compute stream) plus an event; ``wait`` synchronizes on
    the event only. On CPU a plain copy."""

    def __init__(self, device: torch.device):
        self.device = device
        self.cuda = device.type == "cuda"
        self._host: Optional[Tensor] = None
        self._size = 0
        self._stream = torch.cuda.Stream(device=device) if self.cuda else None
        self._event = torch.cuda.Event() if self.cuda else None

    def start(self, source: Tensor) -> None:
        """Enqueue the copy of ``source`` (1-D int32 on the device)."""
        size = source.numel()
        if self._host is None or self._host.numel() < size:
            self._host = torch.empty(size, dtype=torch.int32, pin_memory=self.cuda)
        self._size = size
        if self.cuda:
            self._stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._stream):
                self._host[:size].copy_(source, non_blocking=True)
                source.record_stream(self._stream)
                self._event.record(self._stream)
        else:
            self._host[:size].copy_(source)

    def wait(self) -> list:
        """Block until the copy landed; the values as a list (call inside ``_host_sync``)."""
        if self.cuda:
            self._event.synchronize()
        return self._host[: self._size].tolist()


def _loss_funcs_equivalent(first: Callable, other: Callable) -> bool:
    """The same callable, or ``functools.partial`` objects of the same function whose
    positional and keyword arguments are identical objects (the loss-function contract).

    Arguments are compared by identity, not by value: partials built per microbatch around
    equal but distinct objects (e.g. a float recomputed for each microbatch) are rejected. This
    is conservative; ``pretrain_hybrid``'s ``partial(loss_func, model=model, validate=False)``
    passes the same ``model`` and ``False`` objects every time."""
    if first is other:
        return True
    if not (isinstance(first, functools.partial) and isinstance(other, functools.partial)):
        return False
    return (
        first.func is other.func
        and len(first.args) == len(other.args)
        and all(a is b for a, b in zip(first.args, other.args))
        and first.keywords.keys() == other.keywords.keys()
        and all(first.keywords[name] is other.keywords[name] for name in first.keywords)
    )


def _loss_func_descriptor(loss_func: Callable) -> str:
    """The loss function's identity for the plan digest (module, name, keyword names)."""
    func = loss_func.func if isinstance(loss_func, functools.partial) else loss_func
    keywords = sorted(loss_func.keywords) if isinstance(loss_func, functools.partial) else []
    name = getattr(func, "__qualname__", type(func).__qualname__)
    return f"{getattr(func, '__module__', '?')}.{name}({','.join(keywords)})"


class LoopRSBFExecutor:
    """One EU's RSBF training step over a ``LoopProgram`` (see the module docstring).

    Args:
        program: the cells over the model.
        pg_collection: the process groups; ``dp`` is the domain (the sample positions, the
            moves and the end-of-step flags) and finalize receives the collection.
        groups: the domain and the Gloo status group (``RSBFGroups``); None means
            ``RSBFGroups(domain=pg_collection.dp)`` without a status group, which suffices for
            one EU and for known-depth criteria without ``check_plan``.
        waves: RSBF passes per step (default ``config.loop_rsbf_waves``).
        fill: ``"drain"`` or ``"sample_order"`` (default ``config.loop_rsbf_fill``).
        rebalance: min-move rebalancing between rounds (default ``config.loop_rsbf_rebalance``).
        move_transport: ``"p2p"`` or ``"all_to_all"`` (default
            ``config.loop_rsbf_move_transport``).
        tail_item_moves: chain-parallel item moves (default
            ``config.loop_rsbf_tail_item_moves``).
        over_budget: ``"raise"`` or ``"warn"`` when a static MoE budget overflowed anywhere in
            the domain during a step (default ``config.loop_rsbf_over_budget``).
        check_recompute: ``True`` / ``"raise"``: assert every replayed round state and
            injection, the output of every recomputed core sublayer and every input a nested
            item's rebuild stores equals the sweep's (bitwise; test-only), and that the RNG
            states are unchanged across a wave. The state and injection checks cost one host
            sync each; the per-sublayer (and rebuild) checks of a CORE item are read in one
            host sync per item. ``"report"`` runs the same
            comparisons and records ``max |replayed - sweep|`` per check in
            ``ExecutorStats.recompute_diffs`` instead of raising (bitwise equality needs
            row-invariant expert GEMMs at EP > 1). False / None: off.
        poison_on_free: fill freed checkpoints with NaN (tests).
        check_plan: compare the plan digests of every EU at each wave's start (inputs) and end
            (every step) over the status group (debug; raises on every rank on a mismatch).
        check_lockstep: assert that every real item calls exactly the lockstep MoE modules of
            its sub-round type, in order, and that every real call has its module's null
            signature (debug; raises on the rank that ran the item).
        verify_stash: keep a clone of every stash entry at capture and compare it with the
            entry when the item reads it (catches writes the version counter cannot see, e.g.
            kernels writing through raw pointers; debug, one sync per entry, counted in
            ``ExecutorStats.stash_verify_syncs``).
        sweep_detach: ``"sublayer"`` (default) drops the graph of the sweep-mode forwards (the
            sweep and the nested rebuild) at every core sublayer; ``"block"`` only at the
            inputs they store (debug, for bisecting).
        measure_memory: record every cell call's allocator peak and retained bytes above its
            entry baseline in ``ExecutorStats.memory`` (debug, CUDA only: a device
            synchronize and a peak reset around every call).
    """

    def __init__(
        self,
        program: LoopProgram,
        *,
        pg_collection,
        groups: Optional[RSBFGroups] = None,
        waves: Optional[int] = None,
        fill: Optional[str] = None,
        rebalance: Optional[bool] = None,
        move_transport: Optional[str] = None,
        tail_item_moves: Optional[bool] = None,
        over_budget: Optional[str] = None,
        check_recompute: Union[bool, str, None] = False,
        poison_on_free: bool = False,
        check_plan: bool = False,
        check_lockstep: bool = False,
        verify_stash: bool = False,
        sweep_detach: str = "sublayer",
        measure_memory: bool = False,
    ):
        if pg_collection is None:
            raise ValueError("LoopRSBFExecutor needs a process group collection")
        config = program.config
        validate_rsbf_config(config)
        if measure_memory and program.device.type != "cuda":
            raise ValueError("measure_memory reads the CUDA caching allocator")
        self.program = program
        self.config = config
        self.pg_collection = pg_collection
        self.groups = groups
        self.waves = config.loop_rsbf_waves if waves is None else waves
        self.fill = config.loop_rsbf_fill if fill is None else fill
        self.rebalance = config.loop_rsbf_rebalance if rebalance is None else bool(rebalance)
        self.move_transport = (
            config.loop_rsbf_move_transport if move_transport is None else move_transport
        )
        self.tail_item_moves = (
            config.loop_rsbf_tail_item_moves if tail_item_moves is None else bool(tail_item_moves)
        )
        if self.waves < 1:
            raise ValueError(f"waves must be >= 1, got {self.waves}")
        if self.fill not in ("drain", "sample_order"):
            raise ValueError(f"unknown fill {self.fill!r}")
        if self.move_transport not in ("p2p", "all_to_all"):
            raise ValueError(f"unknown move transport {self.move_transport!r}")
        self.over_budget = config.loop_rsbf_over_budget if over_budget is None else over_budget
        if self.over_budget not in ("raise", "warn"):
            raise ValueError(f"unknown over_budget mode {self.over_budget!r}")
        if check_recompute is True:
            check_recompute = "raise"
        if check_recompute not in (False, None, "raise", "report"):
            raise ValueError(f"unknown check_recompute mode {check_recompute!r}")
        if sweep_detach not in ("sublayer", "block"):
            raise ValueError(f"unknown sweep_detach {sweep_detach!r}")
        # The comparison mode ("raise" / "report") and whether any check runs.
        self.recompute_mode = check_recompute or None
        self.check_recompute = self.recompute_mode is not None
        self.check_plan = check_plan
        self.check_lockstep = check_lockstep
        self.verify_stash = verify_stash
        self.sweep_detach = sweep_detach
        self.measure_memory = measure_memory
        self.device = program.device
        self.store = LoopStateStore(
            poison_on_free=poison_on_free,
            act_dtype=program.dtype,
            device=program.device,
            core_sublayers=program.roles.core,
        )
        self._split_pairs = block_boundaries_split_pairs(
            program.uniform_core_plan, program.stack.layer_type_list, program.roles
        )
        # verify_stash: (kind, sample, round, sublayer) -> the clone taken at capture.
        self._stash_clones: dict = {}
        self.stats = ExecutorStats()
        self.dispatcher: Optional[LoopSampleDispatcher] = None
        self._dispatcher_key = None
        self.lockstep: Optional[LockstepPlan] = None
        self._exit_reader = _DeviceToHost(self.device)
        self._step_reader = _DeviceToHost(self.device)
        self._pinned: dict = {}
        self._flops: dict = {}  # sequence length -> per-cell forward FLOPs
        self._samples: Optional[dict] = None
        self._chain: Optional[dict] = None
        self._records: Optional[dict] = None
        self._loss_func: Optional[Callable] = None

    @property
    def num_eus(self) -> int:
        """W, the EUs of the domain (1 before the first step)."""
        return 1 if self.dispatcher is None else self.dispatcher.num_eus

    # -----------------------------------------------------------------------------------
    # The domain
    # -----------------------------------------------------------------------------------

    def _setup_domain(self) -> LoopSampleDispatcher:
        """The dispatcher of this step's domain (``pg_collection.dp``), rebuilt when the groups
        change; one all-rank warm-up exchange per new multi-EU dispatcher."""
        dp = getattr(self.pg_collection, "dp", None)
        groups = self.groups if self.groups is not None else RSBFGroups(domain=dp)
        if groups.domain is not dp:
            raise ValueError(UNSUPPORTED + "an RSBF domain group other than pg_collection.dp")
        key = (groups.domain, groups.status, self.move_transport)
        if self.dispatcher is None or self._dispatcher_key != key:
            dispatcher = LoopSampleDispatcher(
                groups, transport=self.move_transport, device=self.device
            )
            self._setup_lockstep(dispatcher)
            if dispatcher.num_eus > 1:
                dispatcher.warm_up()
            self.dispatcher, self._dispatcher_key = dispatcher, key
        return self.dispatcher

    def _setup_lockstep(self, dispatcher: LoopSampleDispatcher) -> None:
        """The lockstep MoE plan of the model (built once), checked against the domain: every
        expert-parallel group of a lockstep module must lie inside the domain (its ranks run
        the same plan, so they issue the same MoE collectives)."""
        if self.lockstep is None:
            self.lockstep = build_lockstep_plan(self.program)
        domain = set(dispatcher.ranks)
        for entry in self.lockstep.entries():
            for group in entry.adapter.groups(entry.module):
                ranks = dist.get_process_group_ranks(group)
                if len(ranks) > 1 and not set(ranks) <= domain:
                    raise ValueError(
                        UNSUPPORTED + f"the expert-parallel group {sorted(ranks)} of "
                        f"{entry.name} reaching outside the RSBF domain {sorted(domain)} (expert "
                        "parallelism must stay inside the data-parallel group)"
                    )

    # -----------------------------------------------------------------------------------
    # The step
    # -----------------------------------------------------------------------------------

    def run_step(
        self,
        batch_func: LoopRSBFBatchFunc,
        data_iterator,
        num_microbatches: int,
        *,
        model: nn.Module,
        iteration: int = 0,
        micro_batch_size: Optional[int] = None,
        force_all_reduce: bool = False,
        validate_loss: Optional[Callable[[Tensor], None]] = None,
    ) -> list:
        """Run one training step over ``num_microbatches`` microbatches from
        ``batch_func(data_iterator, model)``; returns one report dict per sample this rank
        loaded, in sample order (every dict with the same keys, ``[value * weight, weight]``
        entries)."""
        program, config = self.program, self.config
        validate_rsbf_data_parallel(model)
        if num_microbatches < 1:
            raise ValueError(f"an RSBF step needs at least one microbatch, got {num_microbatches}")
        if num_microbatches % self.waves:
            raise ValueError(
                f"loop_rsbf_waves ({self.waves}) must divide the microbatches per step "
                f"({num_microbatches})"
            )
        if self.store.wave is not None:  # a previous step raised mid-wave
            self.store.reset()
        self.stats = ExecutorStats(split_pairs=self._split_pairs)
        program.reset_counters()
        device = self.device
        dispatcher = self._setup_domain()
        num_eus, eu = dispatcher.num_eus, dispatcher.eu
        status_before = dispatcher.stats.status_exchanges
        lockstep = self.lockstep
        lockstep.reset_over_budget()
        # Once per step (the standard schedule sets the same values after every microbatch).
        set_auxiliary_loss_scales(config, lambda: device, num_microbatches=1, cp_group_size=1)
        self._aux_seed = self._grad_scale(torch.ones((), dtype=torch.float32, device=device))
        self._n_tok_local = torch.zeros((), dtype=torch.int32, device=device)
        self._loss_sum_local = torch.zeros((), dtype=torch.float32, device=device)
        self._plan_mismatches = torch.zeros((), dtype=torch.int32, device=device)
        self._padded_local = torch.zeros((), dtype=torch.int32, device=device)
        self._schema_errors = 0
        per_wave = num_microbatches // self.waves
        reports = []
        microbatch = 0
        batch_rows = micro_batch_size
        try:
            with lockstep.observing(record_calls=self.check_lockstep):
                for wave in range(self.waves):
                    samples, batches = [], []
                    for _ in range(per_wave):
                        batch = batch_func(data_iterator, model)
                        rows = batch.tokens.shape[0]
                        if batch_rows is None:
                            batch_rows = rows
                        if rows != batch_rows:
                            raise ValueError(
                                f"microbatch {microbatch} has {rows} rows, expected {batch_rows}"
                            )
                        batches.append(batch)
                        samples += self._split(
                            batch, microbatch, iteration, micro_batch_size=rows, dp=(eu, num_eus)
                        )
                        microbatch += 1
                    with forward_statistics_weight(1.0 / batch_rows):
                        reports += self._run_wave(
                            wave, samples, batches, iteration=iteration, rows=batch_rows
                        )
        finally:
            # The samples hold the batch and the loss function (which may hold the model).
            self._samples = self._chain = self._records = None
            self._loss_func = None
        self.stats.status_exchanges = dispatcher.stats.status_exchanges - status_before
        self.stats.exit_counts = exit_depth_counts(self.stats.depths.values(), program.max_rounds)
        step_loss = self._end_of_step(dispatcher)
        if validate_loss is not None:
            # The mean per-microbatch sum: NaN / Inf survive the division, and the spiky-loss
            # history compares like values whatever the microbatch count (batch-size schedules).
            validate_loss(step_loss / num_microbatches)
        if config.finalize_model_grads_func is not None:
            config.finalize_model_grads_func(
                [model],
                self._n_tok_local,
                pg_collection=self.pg_collection,
                force_all_reduce=force_all_reduce,
            )
        return reports

    def _end_of_step(self, dispatcher: LoopSampleDispatcher) -> Tensor:
        """The one end-of-step read: every EU's after-exit count, plan mismatches and report
        schema errors (reduced MAX over the domain, so every rank decides alike), this EU's
        loss sum and its padded-token count (``ExecutorStats.padded_tokens``). Raises on every
        rank when any EU flagged a violation; returns the loss sum (host fp32)."""
        program, device = self.program, self.device
        num_eus, eu = dispatcher.num_eus, dispatcher.eu
        flags = torch.zeros((num_eus, len(_FLAG_NAMES)), dtype=torch.int32, device=device)
        flags[eu, _FLAG_AFTER_EXIT].copy_(program.after_exit_violations)
        flags[eu, _FLAG_PLAN_MISMATCH].copy_(self._plan_mismatches)
        if self._schema_errors:
            flags[eu, _FLAG_REPORT_SCHEMA] = self._schema_errors
        budget_names, budget_flags = self.lockstep.over_budget_flags()
        if budget_flags is not None:
            flags[eu, _FLAG_OVER_BUDGET].copy_(budget_flags.to(device).max())
        if num_eus > 1:
            dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=dispatcher.groups.domain)
        parts = [
            flags.reshape(-1),
            self._loss_sum_local.reshape(1).view(torch.int32),
            self._padded_local.reshape(1),
        ]
        if budget_flags is not None:
            # This rank's per-module flags ride the same read (they name the local layers).
            parts.append(budget_flags.to(device))
        self._step_reader.start(torch.cat(parts))
        values = self._host_sync(self._step_reader.wait)
        width = len(_FLAG_NAMES)
        table = [values[e * width : (e + 1) * width] for e in range(num_eus)]
        loss_bits = values[num_eus * width]
        self.stats.padded_tokens = values[num_eus * width + 1]
        local_budget = values[num_eus * width + 2 :]
        self.stats.over_budget_layers = [
            name for name, flag in zip(budget_names, local_budget) if flag
        ]
        self.stats.flagged_ranks = {
            name: [e for e in range(num_eus) if table[e][column]]
            for column, name in enumerate(_FLAG_NAMES)
        }

        def where(column):
            if num_eus == 1:
                return ""
            ranks = self.stats.flagged_ranks[_FLAG_NAMES[column]]
            return f" on domain EUs {ranks} (global ranks {[dispatcher.ranks[e] for e in ranks]})"

        violations = sum(row[_FLAG_AFTER_EXIT] for row in table)
        if violations:
            raise RuntimeError(
                f"{SWEEP_AFTER_EXIT}: {violations} sample-round(s) swept after exit"
                + where(_FLAG_AFTER_EXIT)
            )
        mismatches = sum(row[_FLAG_PLAN_MISMATCH] for row in table)
        if mismatches:
            raise RuntimeError(
                f"{PLAN_MISMATCH}: {mismatches} sample(s)" + where(_FLAG_PLAN_MISMATCH)
            )
        schema = sum(row[_FLAG_REPORT_SCHEMA] for row in table)
        if schema:
            raise RuntimeError(f"{REPORT_SCHEMA}: {schema} sample(s)" + where(_FLAG_REPORT_SCHEMA))
        if any(row[_FLAG_OVER_BUDGET] for row in table):
            local = self.stats.over_budget_layers
            text = (
                f"{OVER_BUDGET} (tokens dropped by a static expert-parallel budget)"
                + where(_FLAG_OVER_BUDGET)
                + (f"; on this rank in {local}" if local else "")
            )
            if self.over_budget == "raise":
                raise RuntimeError(text)
            if eu == 0:
                logger.warning(text)
        return torch.tensor([loss_bits], dtype=torch.int32).view(torch.float32)[0]

    def _split(self, batch, microbatch, iteration, *, micro_batch_size, dp):
        """The rows of one microbatch as samples (c = 1); counts their issued and padded tokens
        (the padded count on the device, read at the end of the step)."""
        dp_rank, dp_size = dp
        self.stats.issued_tokens += batch.tokens.numel()
        if batch.padding_mask is not None:
            self._padded_local.add_(batch.padding_mask.sum(dtype=torch.int32))
        samples = []
        for row in range(micro_batch_size):
            rows = slice(row, row + 1)
            mask = batch.attention_mask
            if mask is not None and mask.shape[0] == micro_batch_size:
                mask = mask[rows]
            samples.append(
                SampleInputs(
                    key=SampleKey(
                        global_index=global_sample_index(
                            microbatch,
                            row,
                            micro_batch_size=micro_batch_size,
                            dp_rank=dp_rank,
                            dp_size=dp_size,
                        ),
                        iteration=iteration,
                    ),
                    tokens=batch.tokens[rows],
                    position_ids=batch.position_ids[rows],
                    labels=batch.labels[rows],
                    loss_mask=batch.loss_mask[rows],
                    attention_mask=mask,
                    padding_mask=None if batch.padding_mask is None else batch.padding_mask[rows],
                    loss_func=batch.loss_func,
                )
            )
        return samples

    @staticmethod
    def _batch_schema(sample: SampleInputs) -> dict:
        """``{slot: (shape, dtype)}`` of the batch fields a sample has (``BATCH_SLOTS`` order)."""
        schema = {}
        for slot in BATCH_SLOTS:
            value = getattr(sample, slot)
            if value is not None:
                schema[slot] = (tuple(value.shape), value.dtype)
        return schema

    def _check_microbatches(self, batches: list, samples: list, first: int) -> None:
        """Multi-EU contract of a wave's microbatches (host, every rank): equivalent loss
        functions and equal report keys (a moved sample's CODA calls its EU's own loss
        function and its record is packed by the keys), and one batch schema."""
        reference = batches[0]
        differing = [
            first + j
            for j, batch in enumerate(batches)
            if not _loss_funcs_equivalent(reference.loss_func, batch.loss_func)
        ]
        if differing:
            raise ValueError(
                f"microbatches {differing} carry a loss function that is not equivalent to "
                f"microbatch {first}'s; with data parallelism every microbatch of an RSBF step "
                "must carry the same loss function (or partials of one function with identical "
                "arguments)"
            )
        keys = [
            first + j
            for j, batch in enumerate(batches)
            if batch.report_keys != reference.report_keys
        ]
        if keys:
            raise ValueError(
                f"microbatches {keys} declare report_keys other than {reference.report_keys}"
            )
        schema = self._batch_schema(samples[0])
        odd = [s.key.global_index for s in samples if self._batch_schema(s) != schema]
        if odd:
            raise ValueError(f"samples {odd} have another batch schema than {schema}")

    # -----------------------------------------------------------------------------------
    # One wave: the plan-driven slot loop
    # -----------------------------------------------------------------------------------

    def _known_depths(self, keys: list) -> Optional[list]:
        try:
            return self.program.known_depths(keys)
        except (KeyError, IndexError) as error:
            missing = []
            for key in keys:
                try:
                    self.program.criterion.known_depths([key])
                except (KeyError, IndexError):
                    missing.append(key.global_index)
            raise ValueError(
                "known-depth criteria must cover every sample of the domain's wave: missing "
                f"global indices {missing}"
            ) from error

    def _run_wave(
        self, wave: int, home_samples: Sequence[SampleInputs], batches: list, *, iteration, rows
    ) -> list:
        program, store, device = self.program, self.store, self.device
        dispatcher = self.dispatcher
        num_eus, eu = dispatcher.num_eus, dispatcher.eu
        n = len(home_samples)
        num = n * num_eus  # G: the domain's samples of this wave
        base = wave * num
        keys = [SampleKey(base + g, iteration) for g in range(num)]
        home = [(g // rows) % num_eus for g in range(num)]
        samples = {}
        for sample in home_samples:
            g = sample.key.global_index - base
            if not 0 <= g < num or home[g] != eu or g in samples:  # pragma: no cover
                raise AssertionError(f"sample {sample.key} is not a home sample of EU {eu}")
            samples[g] = sample
        self._loss_func = batches[0].loss_func
        self._report_keys = tuple(batches[0].report_keys)
        self._schema = self._batch_schema(home_samples[0])
        self._sequence = home_samples[0].sequence
        if num_eus > 1:
            self._check_microbatches(batches, list(home_samples), wave * len(batches))
        depths = self._known_depths(keys)
        online = depths is None
        if num_eus > 1 and dispatcher.groups.status is None and (online or self.check_plan):
            raise ValueError(
                UNSUPPORTED
                + ("an online exit criterion" if online else "check_plan")
                + f" over {num_eus} EUs without an RSBF status group (RSBFGroups.status)"
            )
        planner = WavePlanner(
            num_eus=num_eus,
            home=home,
            max_rounds=program.max_rounds,
            fill=self.fill,
            rebalance=self.rebalance,
            tail_item_moves=self.tail_item_moves,
            criterion_differentiable=program.criterion.differentiable,
            check_recompute=self.check_recompute,
            batch_slots=tuple(self._schema),
            report_keys=len(self._report_keys),
            core_stash=program.uniform_core_plan.stashed_block_inputs(),
            core_trace=program.roles.core if self.check_recompute else 0,
        )
        self._planner = planner
        context = "|".join(
            (
                f"depths={depths}",
                f"loss={_loss_func_descriptor(self._loss_func)}",
                f"reports={self._report_keys}",
                f"schema={sorted((k, v[0], str(v[1])) for k, v in self._schema.items())}",
                f"S={self._sequence}",
            )
        )
        if self.check_plan:
            self._check_digest(planner, context, "plan inputs")

        store.begin_wave(wave)
        self._stash_clones = {}
        rng = self._rng_state() if self.check_recompute else None
        home_samples_g = sorted(samples)
        self._samples = samples  # gains the samples that arrive on this EU
        self._chain = {}
        self._records = {}
        self._keys = keys
        self._moves_of = [0] * num
        self._wave_syncs = 0
        status_before = dispatcher.stats.status_exchanges
        self._exited_dev = torch.zeros(num, dtype=torch.int32, device=device)
        self._exit_round_dev = torch.zeros(num, dtype=torch.int32, device=device)
        self._depth_dev = None if online else self._upload(("depth", wave), depths)
        self._exit_buf = torch.zeros(num + 1, dtype=torch.int32, device=device) if online else None
        self._host_exit_round = [0] * num

        self._execute(planner.prelude_steps())
        r = 1
        rounds = 0
        while planner.active() or planner.pending():
            swept = False
            if planner.active():
                active = planner.active()
                if r > program.max_rounds:
                    raise RuntimeError(
                        f"samples {active} still active after R_max = {program.max_rounds}"
                    )
                rounds = r
                # [F] sample moves, then the forward sub-rounds over the active samples.
                self._execute(planner.forward_steps(r))
                swept = True
                if online:
                    self._exit_buf[num : num + 1].copy_(program.after_exit_violations.reshape(1))
                    self._exit_reader.start(self._exit_buf)
                else:
                    # Known depths: the exits are host data, their chains run this slot.
                    newly = self._newly_exited(active, r, depths=depths, bits=None)
                    planner.record_exits(r, self._retire(newly, r))
            # [B] the slot's backward work, enqueued before the (online) wait.
            self._execute(planner.backward_steps(final=False))
            if swept and online:
                bits = self._host_sync(self._exit_reader.wait)
                self._wave_syncs += 1
                local_violations = bits[num]
                bitmap = torch.tensor(
                    bits[:num] + [1 if local_violations else 0], dtype=torch.uint8
                )
                bitmap = dispatcher.exchange_status(bitmap)
                if bitmap[num]:
                    if num_eus == 1 or local_violations:
                        where = "" if num_eus == 1 else f" on EU {eu}"
                        raise RuntimeError(
                            f"{SWEEP_AFTER_EXIT}: {local_violations} sample-round(s) swept "
                            f"after exit{where}"
                        )
                    raise RuntimeError(f"{SWEEP_AFTER_EXIT}: on another EU of the domain")
                newly = self._newly_exited(active, r, depths=None, bits=bitmap[:num].tolist())
                planner.record_exits(r, self._retire(newly, r))
            r += 1
        self._execute(planner.backward_steps(final=True))
        # The executed host plan against the device's first exit rounds, on the EU that ran
        # each sample's exit sweep (read at step end).
        plan = [planner.exit_round(g) for g in range(num)]
        here = [int(planner.eu_of_round(g, plan[g]) == eu) for g in range(num)]
        plan_dev = self._upload(("plan", wave), plan)
        here_dev = self._upload(("here", wave), here)
        mismatched = torch.ne(self._exit_round_dev, plan_dev).to(torch.int32) * here_dev
        self._plan_mismatches.add_(mismatched.sum(dtype=torch.int32))
        if self.check_plan:
            self._check_digest(planner, context, "plan")
        self.stats.store_peaks.append(store.end_wave())
        self.stats.host_syncs_by_wave.append(self._wave_syncs)
        self.stats.status_exchanges_by_wave.append(
            dispatcher.stats.status_exchanges - status_before
        )
        self.stats.rounds_by_wave.append(rounds)
        self._record_plan_stats(planner, keys)
        if rng is not None and not self._rng_unchanged(rng):
            raise RuntimeError("the RNG state changed during an RSBF wave (exit-criterion rule 2)")
        return self._reports(wave, home_samples_g, planner)

    def _check_digest(self, planner: WavePlanner, context: str, what: str) -> None:
        digest = hashlib.sha256(planner.digest() + context.encode()).digest()
        self.stats.debug_syncs += 1
        self.dispatcher.check_identical(digest, what)

    def _record_plan_stats(self, planner: WavePlanner, keys: list) -> None:
        stats = self.stats
        plan = planner.stats
        stats.moves_by_round.append(list(plan.moves_by_round))
        stats.active_by_round.append(list(plan.active_by_round))
        for kind_type in SubRoundType:
            stats.subrounds[kind_type.value] += plan.subrounds[kind_type]
            stats.nulls[kind_type.value] += plan.nulls[kind_type]
        stats.plan_digests.append(planner.digest())
        for g, key in enumerate(keys):
            stats.depths[key] = planner.exit_round(g)

    def _newly_exited(self, active, r, *, depths, bits) -> list:
        """The active samples the host retires after round ``r``: from the known depths, or
        from the exit bits of the round's status exchange (online). The device checks this
        plan."""
        if bits is None:
            return [g for g in active if depths[g] == r]
        return [g for g in active if bits[g]]

    def _retire(self, newly: list, r: int) -> list:
        """Record round ``r`` as the exit round (the depth) of the ``newly`` exited samples."""
        for g in newly:
            if self._host_exit_round[g]:
                raise RuntimeError(f"sample {g} retired twice")
            self._host_exit_round[g] = r
        return newly

    # -----------------------------------------------------------------------------------
    # Plan steps
    # -----------------------------------------------------------------------------------

    def _execute(self, steps: Sequence) -> None:
        eu = self.dispatcher.eu
        for step in steps:
            if isinstance(step, SubRound):
                item = step.items[eu]
                if item is None:
                    self._run_null(step.type)
                    continue
                if self.check_lockstep:
                    self.lockstep.begin_item()
                measured = self._memory_begin() if self.measure_memory else None
                if item.kind == "prelude_sweep":
                    self._prelude_sweep(item.sample)
                elif item.kind == "core_sweep":
                    self._core_sweep(item.sample, item.round)
                elif item.kind == "coda":
                    self._coda_item(item.sample, item.round)
                elif item.kind == "core":
                    self._core_item(item.sample, item.round)
                else:
                    self._prelude_item(item.sample)
                if measured is not None:
                    self._memory_end(measured, item.kind, item.sample, item.round)
                if self.check_lockstep:
                    self.lockstep.end_item(
                        step.type,
                        f"{item.kind}(sample {item.sample}, round {item.round})",
                        core_plan=self.program.uniform_core_plan,
                        core_start=self.program.roles.core_start,
                    )
            elif isinstance(step, MovePhase):
                self._run_move_phase(step)
            elif isinstance(step, FreeInjection):
                if step.eu == eu:
                    self.store.free_injection(step.sample)
            else:  # pragma: no cover - the planner emits nothing else
                raise TypeError(f"unknown plan step {step!r}")

    def _memory_begin(self) -> int:
        """``measure_memory``: synchronize, reset the allocator peak; the baseline bytes."""
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        return torch.cuda.memory_allocated(self.device)

    def _memory_end(self, base: int, kind: str, g: int, r: int) -> None:
        """``measure_memory``: the call's peak and retained bytes above ``base``."""
        torch.cuda.synchronize(self.device)
        self.stats.memory.append(
            MemoryRecord(
                kind=kind,
                sample=g,
                round=r,
                base=base,
                peak=torch.cuda.max_memory_allocated(self.device) - base,
                retained=torch.cuda.memory_allocated(self.device) - base,
            )
        )

    def _run_null(self, kind_type: SubRoundType) -> None:
        """An EU without an item in a lockstep sub-round: the type's MoE calls on an empty
        routing (``null_items.run_null``), inside the stack's quantization contexts (those of
        any sample on this EU: they depend on the config only). Nothing for dense models and
        EP = 1.

        ``run_step`` loads the same number of microbatches on every EU (the wave's home table
        assumes it), so every EU sweeps its home samples' preludes in the wave's first
        PRELUDE-F sub-rounds before any null: the planner's PRELUDE-F nulls (unequal home
        counts) are not reachable through the executor, and a null without a layer context
        means that contract was broken."""
        if not self.lockstep.by_type[kind_type]:
            return
        ctx = next(
            (sample.ctx for sample in self._samples.values() if sample.ctx is not None), None
        )
        if ctx is None:
            raise ValueError(
                f"a {kind_type.value} null item on EU {self.dispatcher.eu} before any of its "
                "samples has a layer context: every EU of the RSBF domain must load the same "
                "number of microbatches per step (equal home counts); unequal home counts "
                "(PRELUDE-F nulls) are not supported by the executor"
            )
        self.stats.null_moe_calls += run_null(
            self.lockstep,
            kind_type,
            outer_context=ctx.outer_quant_context,
            inner_context=ctx.quant_context_factory,
            sequence=self._sequence,
            device=self.device,
            # CORE-B nulls follow the items' (uniform) backward plan.
            core_plan=self.program.uniform_core_plan,
            core_start=self.program.roles.core_start,
        )

    def _run_move_phase(self, phase: MovePhase) -> None:
        """Exchange one move phase; then the sources release what left (``send_*``, chain
        records) and the destinations install what arrived."""
        eu = self.dispatcher.eu
        for t in phase.transfers:
            self.stats.moves[t.kind] += 1
            if t.kind == "sample":
                self._moves_of[t.sample] += 1
            self.stats.stash_moves += sum(
                parse_sublayer_slot(kind)[0] == BLOCK_ENTRY for kind, _ in t.frees
            )
        received = self.dispatcher.exchange(phase, send=self._payload, recv=self._receive_buffers)
        for t in phase.transfers:
            if t.src == eu:
                self._release_sent(t)
        for t in phase.transfers:
            if t.dst == eu:
                self._install(t, received[t])

    @staticmethod
    def _entry_round(t: Transfer, kind: str) -> int:
        for entry, r in t.frees:
            if entry == kind:
                return r
        raise AssertionError(f"transfer {t} carries no {kind} entry")  # pragma: no cover

    def _slot_spec(self, slot: str) -> tuple:
        """``(shape, dtype)`` of a payload slot at c = 1 (the port table)."""
        program = self.program
        if slot in (ROUND_INPUT, INJECTION, GRAD_STATE, GRAD_INJECTION, EXPECTED_STATE):
            return (self._sequence, 1, program.hidden_width), program.dtype
        if parse_sublayer_slot(slot)[0] in (BLOCK_INPUT, TRACE):
            return (self._sequence, 1, program.hidden_width), program.dtype
        if slot in (CRIT_STATE, GRAD_CRIT):
            return (1, program.criterion.state_width), torch.float32
        if slot == EXIT_FLAGS:
            return (2,), torch.int32
        if slot == LOSS_RECORD:
            return (2 + 2 * len(self._report_keys),), torch.int32
        return self._schema[slot]

    def _payload(self, t: Transfer) -> list:
        """This EU's tensors of transfer ``t``, one per slot."""
        g, store = t.sample, self.store
        if t.kind == "chain":
            record = self._chain[g]
            tensors = []
            for slot in t.slots:
                if slot == LOSS_RECORD:
                    tensors.append(self._pack_loss(record.loss_record))
                    continue
                value = {
                    GRAD_STATE: record.grad_state,
                    GRAD_INJECTION: record.grad_injection,
                    GRAD_CRIT: record.grad_crit,
                    EXPECTED_STATE: record.expected_state,
                }[slot]
                if value is None:  # pragma: no cover - the producer always sets it
                    shape, dtype = self._slot_spec(slot)
                    value = torch.zeros(shape, dtype=dtype, device=self.device)
                tensors.append(value)
            return tensors
        tensors = []
        for slot in t.slots:
            if slot == ROUND_INPUT:
                tensors.append(store.get_state(g, self._entry_round(t, "state")))
            elif slot == CRIT_STATE:
                tensors.append(store.get_crit(g, self._entry_round(t, "crit")))
            elif slot == INJECTION:
                tensors.append(store.get_injection(g))
            elif slot == EXIT_FLAGS:
                tensors.append(
                    torch.cat((self._exited_dev[g : g + 1], self._exit_round_dev[g : g + 1]))
                )
            elif parse_sublayer_slot(slot)[0] in (BLOCK_INPUT, TRACE):
                # Contiguous and owning its storage (``_stash_tensor``): sent as is.
                base, sublayer = parse_sublayer_slot(slot)
                kind = BLOCK_ENTRY if base == BLOCK_INPUT else TRACE_ENTRY
                tensors.append(store.get_block_input(g, t.round, sublayer, kind=kind))
            else:
                tensors.append(getattr(self._samples[g], slot))
        return tensors

    def _receive_buffers(self, t: Transfer) -> list:
        buffers = []
        for slot in t.slots:
            shape, dtype = self._slot_spec(slot)
            buffers.append(torch.empty(shape, dtype=dtype, device=self.device))
        return buffers

    def _release_sent(self, t: Transfer) -> None:
        """What left this EU with ``t`` (after the phase completed)."""
        g = t.sample
        if t.kind == "chain":
            del self._chain[g]
            return
        for kind, r in t.frees:
            base, sublayer = parse_sublayer_slot(kind)
            if kind == "state":
                self.store.send_state(g, r)
            elif kind == "crit":
                self.store.send_crit(g, r)
            elif base in (BLOCK_ENTRY, TRACE_ENTRY):
                self.store.send_block_input(g, r, sublayer, kind=base)
                self._stash_clones.pop((base, g, r, sublayer), None)
            else:  # pragma: no cover - the planner frees nothing else
                raise AssertionError(f"transfer {t} releases an unknown entry {kind!r}")

    def _install(self, t: Transfer, tensors: list) -> None:
        """What arrived on this EU with ``t``."""
        g, store = t.sample, self.store
        slots = dict(zip(t.slots, tensors))
        if self.program.check_ports:
            ports = CHAIN_PORTS if t.kind == "chain" else MOVE_PORTS
            for slot, value in slots.items():
                # A per-sublayer slot ("block_input:<l>") is checked against its base port.
                self.program.check(
                    find_port(ports, parse_sublayer_slot(slot)[0], "in"),
                    value,
                    rows=1,
                    sequence=self._sequence,
                )
        if t.kind == "chain":
            if g in self._chain:  # pragma: no cover - one record per chain
                raise AssertionError(f"sample {g} already has a chain record on this EU")
            self._chain[g] = ChainRecord(
                sample=g,
                grad_state=slots.get(GRAD_STATE),
                grad_injection=slots.get(GRAD_INJECTION),
                grad_crit=slots.get(GRAD_CRIT),
                expected_state=slots.get(EXPECTED_STATE),
                loss_record=self._unpack_loss(g, slots[LOSS_RECORD]),
            )
            return
        if "tokens" in slots:
            self._arrive(g, slots)
        for kind, r in t.frees:
            base, sublayer = parse_sublayer_slot(kind)
            if kind == "state":
                store.put_received_state(g, r, slots[ROUND_INPUT])
            elif kind == "crit":
                store.put_received_crit(g, r, slots[CRIT_STATE])
            elif base in (BLOCK_ENTRY, TRACE_ENTRY):
                slot = block_slot(sublayer) if base == BLOCK_ENTRY else trace_slot(sublayer)
                store.put_received_block_input(g, r, sublayer, slots[slot], kind=base)
                self._keep_stash_clone(base, g, r, sublayer, slots[slot])
            else:  # pragma: no cover - the planner frees nothing else
                raise AssertionError(f"transfer {t} installs an unknown entry {kind!r}")
        if INJECTION in slots:
            store.put_received_injection(g, slots[INJECTION])
        # Exit flags only grow (0 -> exited, 0 -> the exit round), so the newest knowledge is
        # the maximum: an item move from an EU that swept an earlier round of ``g`` must not
        # erase the exit round recorded here by the exit sweep.
        flags = slots[EXIT_FLAGS]
        for flag, value in ((self._exited_dev, flags[0:1]), (self._exit_round_dev, flags[1:2])):
            flag[g : g + 1] = torch.maximum(flag[g : g + 1], value)

    def _arrive(self, g: int, slots: dict) -> None:
        """A sample's first arrival on this EU: its batch rows and the activation-free context
        (rebuilt, not sent); the CODA of a moved sample calls this EU's loss function."""
        if g in self._samples:  # pragma: no cover - the plan sends the batch once per EU
            raise AssertionError(f"sample {g} arrived twice on this EU")
        sample = SampleInputs(
            key=self._keys[g],
            tokens=slots["tokens"],
            position_ids=slots["position_ids"],
            labels=slots["labels"],
            loss_mask=slots["loss_mask"],
            attention_mask=slots.get("attention_mask"),
            padding_mask=slots.get("padding_mask"),
            loss_func=self._loss_func,
        )
        self.program.prelude.build_context(sample)
        self._samples[g] = sample

    def _pack_loss(self, record: SampleLossRecord) -> Tensor:
        """``[loss fp32 bits, num_tokens, report values' fp32 bits]`` int32 (the slot)."""
        parts = [
            record.loss.detach().reshape(1).to(torch.float32).view(torch.int32),
            record.num_tokens.detach().reshape(1).to(torch.int32),
        ]
        for key in self._report_keys:
            parts.append(record.report[key].detach().reshape(2).view(torch.int32))
        return torch.cat(parts)

    def _unpack_loss(self, g: int, packed: Tensor) -> SampleLossRecord:
        report = {}
        for j, key in enumerate(self._report_keys):
            report[key] = packed[2 + 2 * j : 4 + 2 * j].view(torch.float32)
        return SampleLossRecord(
            key=self._keys[g],
            depth=self._planner.exit_round(g),
            loss=packed[0:1].view(torch.float32).reshape(()),
            num_tokens=packed[1].clone(),
            report=report,
        )

    def _checked_report(self, report: dict) -> dict:
        """Multi-EU: the report must have exactly ``report_keys`` with 2-element fp32 values
        (the loss record's schema). Otherwise the error is counted (raised by every rank at
        the end of the step) and a zero report keeps the chain well-formed."""
        valid = set(report) == set(self._report_keys) and all(
            isinstance(report[key], Tensor)
            and report[key].numel() == 2
            and report[key].dtype == torch.float32
            for key in self._report_keys
        )
        if valid:
            return report
        self._schema_errors += 1
        return {
            key: torch.zeros(2, dtype=torch.float32, device=self.device)
            for key in self._report_keys
        }

    # -----------------------------------------------------------------------------------
    # Forward cell calls
    # -----------------------------------------------------------------------------------

    def _prelude_sweep(self, g: int) -> None:
        """PRELUDE-F of home sample ``g``: the injection and ``crit_0``."""
        program, store = self.program, self.store
        sample = self._samples[g]
        with torch.enable_grad():
            prelude = program.prelude(sample, phase=LoopPhase.SWEEP)
        store.put_injection(g, prelude.injection)
        store.put_crit(g, 0, prelude.crit_state)
        self._work("prelude_sweep", g, 0)
        del prelude

    def _core_sweep(self, g: int, r: int) -> None:
        """CORE-F of sample ``g``, round ``r`` (sweep mode), plus the device exit bookkeeping."""
        program, store = self.program, self.store
        if self._host_exit_round[g]:
            raise RuntimeError(f"{SWEEP_AFTER_EXIT}: the host plan swept retired sample {g}")
        sample = self._samples[g]
        # What the sweep stashes for the round's backward item (nothing for the whole round and
        # the nested item); the item consults the same plan.
        plan = self._checked_core_plan(sample, r)
        sl = slice(g, g + 1)
        stash = plan.stashed_block_inputs()
        # The counting forward of round r: statistics also binned per (round, layer).
        with torch.enable_grad(), forward_statistics_round(r, program.max_rounds):
            fwd = program.core.forward_detached(
                store.get_state(g, r - 1),
                store.get_injection(g),
                sample,
                r=r,
                phase=LoopPhase.SWEEP,
                capture=stash,
                crit_state=store.get_crit(g, r - 1),
                exited=self._exited_dev[sl],
                depth=None if self._depth_dev is None else self._depth_dev[sl],
                trace=self.check_recompute,
                detach_every_sublayer=self.sweep_detach == "sublayer",
            )
        out = fwd.outputs
        store.put_state(g, r, out.state)
        store.put_crit(g, r, out.crit_state)
        # Block inputs (and check_recompute traces): aliases of the sweep's activations, one
        # stored tensor per captured tensor (trace l - 1 is the block input l).
        stored = {}

        def stashed(tensor):
            if fwd.modified_after_capture(tensor):
                # A later sublayer wrote into its (detached) input in place.
                raise RuntimeError(
                    f"{STASH_MODIFIED}: sample {g}, round {r}: a core sublayer modified a "
                    "captured activation in place during the sweep"
                )
            if id(tensor) not in stored:
                stored[id(tensor)] = self._stash_tensor(tensor)
            return stored[id(tensor)]

        for sublayer in stash:
            self._stash_put(BLOCK_ENTRY, g, r, sublayer, stashed(fwd.captured[sublayer]))
        for sublayer, tensor in enumerate(fwd.trace):
            self._stash_put(TRACE_ENTRY, g, r, sublayer, stashed(tensor))
        del stored, fwd
        self._work("core_sweep", g, r)
        bit = out.exit.to(torch.int32)
        del out
        exit_round = self._exit_round_dev
        first = torch.gt(bit, 0) & torch.eq(exit_round[sl], 0)
        exit_round[sl] = torch.where(first, program.round_tensor(r), exit_round[sl])
        self._exited_dev[sl] |= bit
        if self._exit_buf is not None:
            self._exit_buf[sl].copy_(bit)

    # -----------------------------------------------------------------------------------
    # Backward items
    # -----------------------------------------------------------------------------------

    def _log_item(self, item: BackwardItem) -> str:
        sample = self._samples[item.sample]
        self.stats.item_order.append((item.kind.value, sample.key.global_index, item.round))
        return f"loop_rsbf/sample={sample.key.global_index}/item={item.kind.value}/r={item.round}"

    def _coda_item(self, g: int, depth: int) -> None:
        """Coda + head forward and backward on a fresh leaf of ``s_d``; the loss record."""
        store, sample = self.store, self._samples[g]
        label = self._log_item(BackwardItem(ItemKind.CODA, g, depth))
        state = store.take_state(g, depth) if self.check_recompute else store.get_state(g, depth)
        leaf = state.detach().requires_grad_()
        with torch.profiler.record_function(label), torch.enable_grad():
            out = self.program.coda(leaf, sample)
            torch.autograd.backward(self._grad_scale(out.loss))
        report = out.report
        if self.num_eus > 1:
            report = self._checked_report(report)
        record = SampleLossRecord(
            key=sample.key,
            depth=depth,
            loss=out.loss.detach(),
            num_tokens=out.num_tokens.detach(),
            report=report,
        )
        if not self.check_recompute:
            store.free_state(g, depth)
        store.free_crit(g, depth)
        self._work("coda", g, depth)
        self._chain[g] = ChainRecord(
            sample=g,
            grad_state=leaf.grad,
            grad_injection=None,
            grad_crit=None,
            expected_state=state if self.check_recompute else None,
            loss_record=record,
        )

    def _checked_core_plan(self, sample: SampleInputs, r: int) -> CoreBackwardPlan:
        """``program.core_plan(sample, r)`` (read once by the sweep and once by the item of
        every (sample, round)), which must be the program's uniform plan: the CORE-B lockstep
        nulls of every EU follow that plan."""
        plan = self.program.core_plan(sample, r)
        if plan is not self.program.uniform_core_plan:
            raise RuntimeError(
                f"{NON_UNIFORM_PLAN} for sample {sample.key.global_index}, round {r} (CORE-B "
                "sub-rounds would need to be typed by plan)"
            )
        return plan

    def _core_item(self, g: int, r: int) -> None:
        """Replay round ``r`` from its checkpoints with grad and back-propagate ``dL/ds_r``,
        ``dL/dcrit_r`` and the criterion's ``aux_loss``: the plan's blocks in reverse, each
        recomputed from a fresh leaf on its input (``s_{r-1}`` for block 0; for later blocks
        the sweep's stash, or for a nested item the layer stash its rebuild filled first) and
        back-propagated at once; the leaf's gradient seeds the previous block. The whole round
        is one block (stage 3, op for op)."""
        program, store = self.program, self.store
        sample = self._samples[g]
        item = BackwardItem(ItemKind.CORE, g, r, plan=self._checked_core_plan(sample, r))
        record = self._chain.pop(g)
        plan = item.plan
        label = self._log_item(item)
        check = self.check_recompute
        previous = None
        if r > 1:
            previous = store.take_state(g, r - 1) if check else store.get_state(g, r - 1)
        round_input = None if previous is None else previous.detach().requires_grad_()
        injection = store.get_injection(g).detach().requires_grad_()
        differentiable = program.criterion.differentiable
        crit = store.get_crit(g, r - 1).detach().requires_grad_(differentiable)
        sl = slice(g, g + 1)
        exited = self._exited_dev[sl]
        depth = None if self._depth_dev is None else self._depth_dev[sl]
        whole = len(plan.blocks) == 1
        # check_recompute: (name, device difference) of the per-sublayer comparisons (and the
        # rebuild's), read in one host sync after the block loop.
        pending = []
        # Nested: the item-local layer stash (core-relative sublayer -> its rebuilt input).
        layer_stash = self._rebuild(g, r, plan, previous, label, pending) if plan.rebuild else {}
        # dL/d(output of the block being replayed): dL/ds_r for the last block.
        grad = record.grad_state
        with torch.profiler.record_function(label), torch.enable_grad():
            for block in reversed(plan.blocks):
                if block.input_source == "round_input":
                    leaf = round_input
                elif block.input_source == "layer_stash":
                    leaf = layer_stash[block.start].detach().requires_grad_()
                else:
                    leaf = self._stash_get(BLOCK_ENTRY, g, r, block.start, take=check)
                    leaf = leaf.detach().requires_grad_()
                trace = [] if check else None
                name = label + (
                    "/phase=recompute" if whole else f"/block={block.start}/phase=recompute"
                )
                with recompute_context(name):
                    out = program.core(
                        leaf,
                        injection,
                        sample,
                        r=r,
                        phase=LoopPhase.RECOMPUTE,
                        crit_state=crit,
                        exited=exited,
                        depth=depth,
                        block=block,
                        trace=trace,
                    )
                if check:
                    for offset, actual in enumerate(trace):
                        sublayer = block.start + offset
                        expected = self._stash_get(TRACE_ENTRY, g, r, sublayer, take=True)
                        what = f"trace[sample {g}, round {r}, sublayer {sublayer}]"
                        pending.append((what, self._device_difference(actual, expected)))
                    del trace
                outputs, grads = [], []
                if block.with_epilogue:
                    if check:
                        self._check_equal(out.state, record.expected_state, f"s[{g}, round {r}]")
                    if grad is not None:
                        outputs.append(out.state)
                        grads.append(grad)
                    if out.aux_loss is not None:
                        # The criterion's per-round loss, token-weighted like loop losses (ph:
                        # num_tokens * auxiliary) and scaled like the main loss.
                        outputs.append(out.aux_loss * record.loss_record.num_tokens)
                        grads.append(self._aux_seed.reshape(out.aux_loss.shape))
                    if record.grad_crit is not None:
                        outputs.append(out.crit_state)
                        grads.append(record.grad_crit)
                else:
                    # A block output always feeds the next block in a residual stack; zeros
                    # keep the block's attached auxiliary losses if it ever does not.
                    outputs.append(out)
                    grads.append(torch.zeros_like(out) if grad is None else grad)
                torch.autograd.backward(outputs, grads)
                del out, outputs, grads
                # dL/d(block input), complete: the block input feeds this block only.
                grad = None if leaf is None else leaf.grad
                if block.input_source == "block_stash" and not check:
                    store.free_block_input(g, r, block.start)
                elif block.input_source == "layer_stash":
                    self._drop_layer_input(layer_stash, block.start)
                del leaf
        if layer_stash:  # pragma: no cover - every later block reads its own entry
            raise AssertionError(f"layer-stash entries {sorted(layer_stash)} were not read")
        if pending:
            self._read_checks(pending)
        record.expected_state = None
        grad_injection = (
            injection.grad
            if record.grad_injection is None
            else record.grad_injection + injection.grad
        )
        if r > 1 and not check:
            store.free_state(g, r - 1)
        store.free_crit(g, r - 1)
        self._work("core_item", g, r)
        self._chain[g] = ChainRecord(
            sample=g,
            grad_state=grad,
            grad_injection=grad_injection,
            grad_crit=crit.grad if differentiable else None,
            expected_state=previous if check else None,
            loss_record=record.loss_record,
        )

    def _rebuild(
        self,
        g: int,
        r: int,
        plan: CoreBackwardPlan,
        previous: Optional[Tensor],
        label: str,
        pending: list,
    ) -> dict:
        """Phase 2 of a nested CORE item: the prologue and core sublayers ``[0,
        plan.rebuild_end)`` of round ``r`` in sweep mode (grad enabled, the training path as
        in the sweep and the recompute, D35; the graph dropped at every sublayer; statistics
        gated and tensor observation suspended like a recompute; no epilogue). Returns the
        item-local layer stash: the inputs of the plan's later blocks. With check_recompute
        each stored input is queued against the sweep's output of the sublayer before it (the
        trace entry, read here and taken by the block that recomputes that sublayer)."""
        program, store = self.program, self.store
        with torch.enable_grad(), recompute_context(label + "/phase=rebuild"):
            fwd = program.core.forward_detached(
                previous,
                store.get_injection(g),
                self._samples[g],
                r=r,
                phase=LoopPhase.REBUILD,
                capture=plan.rebuilt_layer_inputs(),
                end=plan.rebuild_end,
                detach_every_sublayer=self.sweep_detach == "sublayer",
            )
        layer_stash = {}
        for sublayer, tensor in fwd.captured.items():
            if fwd.modified_after_capture(tensor):
                raise RuntimeError(
                    f"{STASH_MODIFIED}: sample {g}, round {r}: a core sublayer modified a "
                    "captured activation in place during the rebuild"
                )
            layer_stash[sublayer] = tensor
            if self.check_recompute:
                expected = store.get_block_input(g, r, sublayer - 1, kind=TRACE_ENTRY)
                what = f"rebuild[sample {g}, round {r}, input of sublayer {sublayer}]"
                pending.append((what, self._device_difference(tensor, expected)))
        del fwd
        self.stats.rebuilds += 1
        self.stats.layer_stash_peak = max(self.stats.layer_stash_peak, len(layer_stash))
        return layer_stash

    def _drop_layer_input(self, layer_stash: dict, sublayer: int) -> None:
        """Drop a nested item's layer-stash entry after its block's backward; with
        ``poison_on_free`` fill it with NaN first unless another entry shares its storage."""
        tensor = layer_stash.pop(sublayer)
        if not (self.store.poison_on_free and tensor.is_floating_point()):
            return
        storage = tensor.untyped_storage().data_ptr()
        if all(other.untyped_storage().data_ptr() != storage for other in layer_stash.values()):
            with torch.no_grad():
                tensor.fill_(float("nan"))

    # -----------------------------------------------------------------------------------
    # The block-input stash
    # -----------------------------------------------------------------------------------

    def _stash_tensor(self, tensor: Tensor) -> Tensor:
        """``tensor`` itself when it is contiguous and owns its whole storage (the store then
        counts exactly its bytes and the dispatcher can send it without a copy), else a
        contiguous copy (``ExecutorStats.stash_copies``)."""
        if (
            tensor.is_contiguous()
            and tensor.storage_offset() == 0
            and tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
        ):
            return tensor
        self.stats.stash_copies += 1
        return tensor.clone(memory_format=torch.contiguous_format)

    def _stash_put(self, kind: str, g: int, r: int, sublayer: int, tensor: Tensor) -> None:
        """Store a block input (``kind`` ``"block"``) or a sweep trace entry (``"trace"``)."""
        self.store.put_block_input(g, r, sublayer, tensor, kind=kind)
        self._keep_stash_clone(kind, g, r, sublayer, tensor)

    def _keep_stash_clone(self, kind: str, g: int, r: int, sublayer: int, tensor: Tensor):
        if self.verify_stash:
            self._stash_clones[(kind, g, r, sublayer)] = tensor.detach().clone()

    def _stash_get(self, kind: str, g: int, r: int, sublayer: int, *, take: bool) -> Tensor:
        """Read (``take``: remove) a stash entry; with ``verify_stash`` compare it with its
        capture-time clone first."""
        store = self.store
        if take:
            tensor = store.take_block_input(g, r, sublayer, kind=kind)
        else:
            tensor = store.get_block_input(g, r, sublayer, kind=kind)
        clone = self._stash_clones.pop((kind, g, r, sublayer), None)
        if clone is not None:
            self.stats.stash_verify_syncs += 1
            with _sync_allowed(self.device):
                equal = torch.equal(clone, tensor)
            if not equal:
                raise RuntimeError(
                    f"{STASH_MODIFIED}: {StoreKey(kind, g, r, sublayer)} (a kernel wrote into "
                    "a stashed activation; the stash must copy it for this configuration)"
                )
        return tensor

    def _prelude_item(self, g: int) -> None:
        """Replay the prelude with grad and back-propagate ``dL/de`` (and ``dL/dcrit_0`` into
        a differentiable criterion's ``init_state``); at home, so the sample's loss record
        arrives here."""
        sample = self._samples[g]
        record = self._chain.pop(g)
        label = self._log_item(BackwardItem(ItemKind.PRELUDE, g, 0))
        with torch.profiler.record_function(label), torch.enable_grad():
            with recompute_context(label + "/phase=recompute"):
                prelude = self.program.prelude(sample, phase=LoopPhase.RECOMPUTE)
            if self.check_recompute:
                self._check_equal(prelude.injection, self.store.get_injection(g), f"e[{g}]")
            outputs, grads = [prelude.injection], [record.grad_injection]
            if record.grad_crit is not None and prelude.crit_state is not None:
                outputs.append(prelude.crit_state)
                grads.append(record.grad_crit)
            torch.autograd.backward(outputs, grads)
        del prelude
        self.store.free_injection(g)
        self._work("prelude_item", g, 0)
        loss = record.loss_record
        self._n_tok_local.add_(loss.num_tokens)
        self._loss_sum_local.add_(loss.loss)
        self._records[g] = loss

    # -----------------------------------------------------------------------------------
    # Reports, FLOPs, helpers
    # -----------------------------------------------------------------------------------

    def _cell_flops(self, sequence: int) -> dict:
        """Forward FLOPs of one call of each cell on one SBHD row of ``sequence`` tokens (the
        per-symbol table), the coda's backward (2 x layers + the head's own factors), a nested
        CORE item's rebuild (the adapter + core sublayers ``[0, rebuild_end)``; 0 for the
        other forms) and the CODA item's recompute (D42: the up projections of the attentions
        its request reaches; 0 without one)."""
        cached = self._flops.get(sequence)
        if cached is not None:
            return cached
        stack, roles = self.program.stack, self.program.roles
        tokens, squared = sequence, sequence * sequence
        prelude = stack.range_forward_flops(0, roles.core_start, tokens, squared, sequence)
        core = stack.range_forward_flops(
            roles.core_start,
            roles.core_end,
            tokens,
            squared,
            sequence,
            extra=stack.round_extra_forward_flops(tokens),
        )
        coda = stack.range_forward_flops(
            roles.core_end, len(stack.layers), tokens, squared, sequence
        )
        plan = self.program.uniform_core_plan
        core_rebuild = 0.0
        if plan.rebuild:
            core_rebuild = stack.range_forward_flops(
                roles.core_start,
                roles.core_start + plan.rebuild_end,
                tokens,
                squared,
                sequence,
                extra=stack.round_extra_forward_flops(tokens),
            )
        coda_backward = 2 * coda
        for work, factor in stack.head_work_terms(tokens, squared, sequence):
            coda = coda + work
            coda_backward = coda_backward + factor * work
        coda_recompute = 0.0
        for attention in self.program.coda_recompute_attentions():
            coda_recompute = coda_recompute + flops.dsv4_up_proj_forward_flops(
                attention.config, tokens
            )
        table = {
            "prelude": float(prelude),
            "core": float(core),
            "coda": float(coda),
            "coda_backward": float(coda_backward),
            "core_rebuild": float(core_rebuild),
            "coda_recompute": float(coda_recompute),
        }
        # A CORE item's recompute: the round once, plus the nested rebuild.
        table["core_item_recompute"] = (
            table["core"] + table["core_rebuild"] if plan.rebuild else table["core"]
        )
        self._flops[sequence] = table
        return table

    def _sample_flops(self, sequence: int, depth: int) -> tuple:
        """(forward, backward, recompute) FLOPs of one sample's cell calls at ``depth``,
        summed in the order of its work records (prelude sweep, sweeps, CODA, CORE items,
        PRELUDE item): placement-free, so the home EU computes it for every sample."""
        table = self._cell_flops(sequence)
        forward = table["prelude"]
        for _ in range(depth):
            forward += table["core"]
        forward += table["coda"]
        backward = table["coda_backward"]
        recompute = table["coda_recompute"]  # 0.0 without a coda recompute request
        for _ in range(depth):
            backward += 2 * table["core"]
            recompute += table["core_item_recompute"]
        backward += 2 * table["prelude"]
        recompute += table["prelude"]
        return forward, backward, recompute

    def _work(self, kind: str, g: int, r: int) -> None:
        """Append the work record of one cell call of sample ``g`` (sweeps are forward work,
        items recompute + backward, the coda forward + backward)."""
        sample = self._samples[g]
        table = self._cell_flops(sample.sequence)
        forward, recompute, backward = {
            "prelude_sweep": (table["prelude"], 0.0, 0.0),
            "core_sweep": (table["core"], 0.0, 0.0),
            "coda": (table["coda"], table["coda_recompute"], table["coda_backward"]),
            "core_item": (0.0, table["core_item_recompute"], 2 * table["core"]),
            "prelude_item": (0.0, table["prelude"], 2 * table["prelude"]),
        }[kind]
        self.stats.work.append(
            WorkRecord(
                kind,
                sample.key.global_index,
                r,
                sample.sequence * sample.rows,
                forward=forward,
                recompute=recompute,
                backward=backward,
            )
        )

    def _reports(self, wave: int, home: list, planner: WavePlanner) -> list:
        """One report per home sample (``g`` order = sample order): the loss function's report
        plus the loop entries, all ``[value * weight, weight]``."""
        core = self.program.roles.core
        rows = []
        for g in home:
            sample = self._samples[g]
            tokens = float(sample.sequence * sample.rows)
            depth = planner.exit_round(g)
            forward, backward, recompute = self._sample_flops(sample.sequence, depth)
            rows.append(
                [
                    forward,
                    tokens,
                    backward,
                    tokens,
                    recompute,
                    tokens,
                    tokens * depth,
                    tokens,
                    tokens * depth * core,
                    tokens,
                    # Every executed round is differentiated (no truncated BPTT under RSBF).
                    tokens * depth * core,
                    tokens,
                    # Forward sample moves per sample-round (the M2 move gate's training form).
                    float(self._moves_of[g]),
                    float(depth),
                ]
            )
        width = 2 * len(_LOOP_REPORT_KEYS)
        values = self._upload(("report", wave), [v for row in rows for v in row], torch.float32)
        values = values.view(len(home), width)
        reports = []
        for i, g in enumerate(home):
            record = self._records[g]
            self.stats.loss_records.append(record)
            report = dict(record.report)
            for j, key in enumerate(_LOOP_REPORT_KEYS):
                report[key] = values[i, 2 * j : 2 * j + 2]
            reports.append(report)
        return reports

    def _grad_scale(self, loss: Tensor) -> Tensor:
        scale = self.config.grad_scale_func
        return loss if scale is None else scale(loss)

    def _upload(self, key, values, dtype=torch.int32) -> Tensor:
        """Host values -> device without a sync: written into a pinned buffer kept per key
        (rewritten only in a later step, after the end-of-step sync) and copied
        non-blocking."""
        size = len(values)
        cuda = self.device.type == "cuda"
        buffer = self._pinned.get(key)
        if buffer is None or buffer.numel() < size or buffer.dtype != dtype:
            buffer = torch.empty(max(size, 1), dtype=dtype, pin_memory=cuda)
            self._pinned[key] = buffer
        host = buffer[:size]
        host.copy_(torch.tensor(values, dtype=dtype))
        if cuda:
            return host.to(self.device, non_blocking=True)
        return host.clone()

    def _host_sync(self, read: Callable[[], Any]) -> Any:
        """Every executor host sync goes through here (counted)."""
        self.stats.host_syncs += 1
        with _sync_allowed(self.device):
            return read()

    def _check_equal(self, actual: Tensor, expected: Optional[Tensor], what: str) -> None:
        """One check_recompute comparison, read at once (one debug sync)."""
        if expected is None:
            return
        if self.recompute_mode == "report":
            self._read_checks([(what, self._device_difference(actual, expected))])
            return
        self.stats.debug_syncs += 1
        with _sync_allowed(self.device):
            equal = torch.equal(actual, expected)
        if not equal:
            raise RuntimeError(RECOMPUTE_DIFFERS.format(what=what))

    def _device_difference(self, actual: Tensor, expected: Tensor) -> Tensor:
        """A 0-dim fp32 device value, no sync: ``"raise"`` 1.0 where ``actual`` and
        ``expected`` differ anywhere (else 0.0), ``"report"`` ``max |actual - expected|``; inf
        for a shape or dtype mismatch."""
        with torch.no_grad():
            actual = actual.detach()
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                return torch.full((), float("inf"), dtype=torch.float32, device=actual.device)
            if self.recompute_mode == "report":
                return (actual.float() - expected.float()).abs().max()
            return torch.ne(actual, expected).any().to(torch.float32)

    def _read_checks(self, pending: list) -> None:
        """Read queued check_recompute differences in one debug sync; ``"raise"`` names the
        first failing check, ``"report"`` records the maxima per name."""
        names = [name for name, _ in pending]
        self.stats.debug_syncs += 1
        with _sync_allowed(self.device):
            values = torch.stack([value for _, value in pending]).tolist()
        for name, value in zip(names, values):
            if self.recompute_mode == "report":
                known = self.stats.recompute_diffs.get(name)
                self.stats.recompute_diffs[name] = (
                    value if known is None or math.isnan(value) else max(known, value)
                )
            elif value != 0.0:  # a NaN is a difference too
                raise RuntimeError(RECOMPUTE_DIFFERS.format(what=name))

    def _rng_state(self):
        cuda = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        return torch.get_rng_state(), cuda

    def _rng_unchanged(self, before) -> bool:
        cpu, cuda = self._rng_state()
        if not torch.equal(cpu, before[0]):
            return False
        return cuda is None or torch.equal(cuda, before[1])


class LoopRSBFSchedule:
    """A ``forward_backward_func``-compatible schedule running the RSBF executor.

    Accepts the keywords ``train_step`` passes to the standard schedules. Programs and
    executors are cached per unwrapped ``HybridModel`` in a ``WeakKeyDictionary`` (never on the
    model: deepcopy / EMA / modelopt clones must not carry them). The domain of the step is
    ``pg_collection.dp``; ``configure(status_group=...)`` gives the executor the Gloo status
    group of that domain (online exit criteria and plan checks with data parallelism).
    """

    def __init__(self):
        self._cache: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
        self.iteration = 0
        self._status_group = None

    def configure(self, *, status_group=None) -> None:
        """The Gloo group over the data-parallel ranks used for the status exchange (None:
        none; enough for known-depth criteria)."""
        self._status_group = status_group

    def reset(self) -> None:
        """Forget the status group, the iteration and every cached executor (tests)."""
        self._cache = weakref.WeakKeyDictionary()
        self.iteration = 0
        self._status_group = None

    def set_iteration(self, iteration: int) -> None:
        """The training iteration of the next step (keys the sampled depths)."""
        self.iteration = int(iteration)

    def cached_executor(self, model: nn.Module) -> Optional[LoopRSBFExecutor]:
        """The executor cached for ``model``, or None."""
        cached = self._cache.get(get_attr_wrapped_model(model, "head_loss", return_model_obj=True))
        return None if cached is None else cached[1]

    def invalidate(self, model: nn.Module) -> None:
        """Drop the executor (and program) cached for ``model`` only, so the next step rebuilds
        them from the model's config (a phase switch); the status group, the iteration and
        other models' executors are kept (unlike ``reset``)."""
        self._cache.pop(get_attr_wrapped_model(model, "head_loss", return_model_obj=True), None)

    def _last_stats(self, model: nn.Module) -> Optional[ExecutorStats]:
        """The statistics of the last step ``model``'s cached executor ran, or None (no
        executor, or none of its steps reached the end-of-step read)."""
        executor = self.cached_executor(model)
        if executor is None or not executor.stats.exit_counts:
            return None
        return executor.stats

    def exit_counts(self, model: nn.Module) -> Optional[list]:
        """The last step's exit-depth histogram of ``model`` (``ExecutorStats.exit_counts``:
        ``counts[d - 1]`` domain samples exited after round ``d``), or None before its first
        step."""
        stats = self._last_stats(model)
        return None if stats is None else list(stats.exit_counts)

    def padding(self, model: nn.Module) -> Optional[tuple]:
        """``(padded tokens, issued tokens)`` of this EU's home samples in the last step of
        ``model`` (padded 0 without a padding mask), or None before its first step."""
        stats = self._last_stats(model)
        return None if stats is None else (stats.padded_tokens, stats.issued_tokens)

    def executor_for(
        self,
        model: nn.Module,
        pg_collection,
        exit_criterion: Optional[ExitCriterion] = None,
        groups: Optional[RSBFGroups] = None,
    ) -> LoopRSBFExecutor:
        """The cached executor of ``model`` (built on first use, rebuilt for another exit
        criterion); it uses ``pg_collection`` and ``groups`` from this call on."""
        hybrid = get_attr_wrapped_model(model, "head_loss", return_model_obj=True)
        cached = self._cache.get(hybrid)
        if cached is None or cached[0] is not exit_criterion:
            program = LoopProgram(model, exit_criterion=exit_criterion)
            cached = (exit_criterion, LoopRSBFExecutor(program, pg_collection=pg_collection))
            self._cache[hybrid] = cached
        executor = cached[1]
        executor.pg_collection = pg_collection
        executor.groups = groups
        return executor

    def __call__(
        self,
        *,
        forward_step_func,
        data_iterator,
        model,
        num_microbatches: int,
        seq_length: int,  # unused
        micro_batch_size: int,
        decoder_seq_length: Optional[int] = None,  # unused
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
        first_val_step: Optional[bool] = None,  # unused
        adjust_tensor_shapes_fn: Optional[Callable] = None,
        p2p_communicator=None,  # unused
        pg_collection=None,
        force_all_reduce: Optional[bool] = False,
        exit_criterion: Optional[ExitCriterion] = None,
    ) -> list:
        del seq_length, decoder_seq_length, first_val_step, p2p_communicator
        if forward_only or collect_non_loss_data:
            raise ValueError(
                "the loop RSBF schedule runs training steps only; evaluation uses the standard "
                "schedule"
            )
        if adjust_tensor_shapes_fn is not None:
            raise ValueError("adjust_tensor_shapes_fn is not supported by the loop RSBF schedule")
        if isinstance(model, list):
            if len(model) != 1:
                raise ValueError("the loop RSBF schedule does not support model chunking")
            model = model[0]
        if isinstance(data_iterator, list):
            if len(data_iterator) != 1:
                raise ValueError("the loop RSBF schedule does not support model chunking")
            data_iterator = data_iterator[0]
        if pg_collection is None:
            # Migration fallback, as in the standard schedules: train() passes None today.
            pg_collection = _build_default_pg_collection()
        config = get_model_config(model)
        validate_rsbf_config(config)
        batch_func = getattr(forward_step_func, RSBF_BATCH_FUNC_ATTR, None)
        if batch_func is None:
            raise ValueError(
                f"loop_schedule='rsbf' needs forward_step_func.{RSBF_BATCH_FUNC_ATTR} (the "
                "microbatch of forward_step without the model call)"
            )
        groups = RSBFGroups(domain=getattr(pg_collection, "dp", None), status=self._status_group)
        executor = self.executor_for(model, pg_collection, exit_criterion, groups)
        if config.timers is not None:
            config.timers('forward-backward', log_level=1).start(
                barrier=config.barrier_with_L1_time
            )
        try:
            return executor.run_step(
                batch_func,
                data_iterator,
                num_microbatches,
                model=model,
                iteration=self.iteration,
                micro_batch_size=micro_batch_size,
                force_all_reduce=bool(force_all_reduce),
                validate_loss=getattr(forward_step_func, RSBF_LOSS_VALIDATOR_ATTR, None),
            )
        finally:
            if config.timers is not None:
                config.timers('forward-backward').stop()


_DEFAULT_SCHEDULE = LoopRSBFSchedule()


def get_loop_rsbf_schedule() -> LoopRSBFSchedule:
    """The module-level schedule (one cache of programs and executors per process)."""
    return _DEFAULT_SCHEDULE


def forward_backward_loop_rsbf(**kwargs) -> list:
    """Schedule entry: runs one RSBF step of the module schedule with the standard schedules'
    keywords. A keyword ``iteration`` sets the schedule's iteration first; without it the
    iteration last set (``set_iteration``) is kept."""
    if "iteration" in kwargs:
        _DEFAULT_SCHEDULE.set_iteration(kwargs.pop("iteration"))
    return _DEFAULT_SCHEDULE(**kwargs)
