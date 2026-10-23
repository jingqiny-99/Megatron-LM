# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The round-synchronous breadth-first (RSBF) executor of the hybrid Huginn loop, one EU.

One training step of a ``HybridModel`` whose decoder is a Huginn ``LoopHybridStack``, run as
the cells of ``LoopProgram`` with per-sample exit (``loop_schedule='rsbf'``):

1. **Waves.** The step's microbatches are split into ``loop_rsbf_waves`` waves; each wave's rows
   are samples (one row per cell call, ``c = 1``), keyed by their global batch position.
2. **Forward.** A prelude sub-round (every sample's injection ``e`` and ``crit_0``), then
   forward rounds over the samples still active. Every sweep runs with grad enabled (the
   training kernel path) and drops its graph; the round inputs ``s_r`` and ``crit_r`` are
   checkpointed in the ``LoopStateStore``.
3. **Exits.** Known depths (``fixed``, ``sampled``, prescribed) are host data: no per-round
   sync. An online criterion's exit bits are copied to the host once per round, after the
   slot's backward work has been enqueued.
4. **Backward chains.** An exited sample's chain is ``CODA`` (coda + head forward and backward,
   the loss record), ``CORE`` for ``r = d..1`` (replay round ``r`` from its checkpoint with
   grad, then backward) and ``PRELUDE`` (replay the prelude, backward into the embedding).
   Every item starts from fresh detached leaves; gradients cross items only through a
   ``ChainRecord``. ``drain`` runs a sample's chain in the slot after its exit;
   ``sample_order`` runs every chain after the last forward round, in sample order.
5. **Finalize.** Once per step, with the local token count ``n_tok_local`` (the sum of the
   CODA items' ``num_tokens``; ``finalize_model_grads`` all-reduces it and scales the
   gradients by its inverse).

Hard exit enforcement: the host never sweeps a retired sample; on the device, every sweep of
a row whose exit bit was already set increments ``LoopProgram.after_exit_violations``, and the
device exit rounds are compared with the host plan once per wave. Both counters are read at
the single end-of-step host sync, which also carries the step's loss sum for the
rerun-state-machine checks (validated on a host tensor, no further sync, as the mean of the
per-microbatch sums so the spiky-loss history does not scale with the microbatch count).

The executor never issues a collective over more than one rank inside a step (each DP rank is
one EU; ``validate_rsbf_config`` rejects TP / EP, ``validate_rsbf_data_parallel`` sharded
data parallelism), so ranks with different item counts cannot deadlock; finalize's
reductions are the step's only cross-rank collectives.

MoE aux and DSA indexer losses: the router runs once per sample (batch 1) with the loss scale
of a one-microbatch step, so their gradient equals the standard schedule's at MBS 1 at any
MBS (batch invariance); at MBS ``b`` the standard schedule averages them over the ``b`` rows
of a microbatch, a ``1 / b`` smaller coefficient. Their logged values are weighted ``1 / b``
(``forward_statistics_weight``) to match the standard schedule's logs at the same MBS. Multi-EU
dispatch, rebalancing, graphs and block / nested items are later milestones; the item
interface (``BackwardItem.plan``, ``CoreBackwardPlan``) is where block and nested items plug in.
"""

import contextlib
import enum
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal, Optional, Sequence

import torch
import torch.distributed as dist
from torch import Tensor, nn

from megatron.core.pipeline_parallel.schedules import (
    _build_default_pg_collection,
    set_auxiliary_loss_scales,
)
from megatron.core.transformer.forward_replay import forward_statistics_weight
from megatron.core.transformer.loop.guards import validate_rsbf_config, validate_rsbf_data_parallel
from megatron.core.transformer.loop.program import (
    CoreBackwardPlan,
    ExitCriterion,
    LoopPhase,
    LoopProgram,
    SampleInputs,
    SampleKey,
    global_sample_index,
)
from megatron.core.transformer.loop.state_store import LoopStateStore
from megatron.core.transformer.loop_transformer_block import recompute_context
from megatron.core.utils import get_attr_wrapped_model, get_model_config

__all__ = [
    "BackwardItem",
    "ChainRecord",
    "ExecutorStats",
    "ItemKind",
    "LoopMicrobatch",
    "LoopRSBFExecutor",
    "LoopRSBFSchedule",
    "RSBF_BATCH_FUNC_ATTR",
    "RSBF_LOSS_VALIDATOR_ATTR",
    "SampleLossRecord",
    "WorkRecord",
    "forward_backward_loop_rsbf",
    "get_loop_rsbf_schedule",
]

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


@dataclass
class LoopMicrobatch:
    """One microbatch for the RSBF schedule (``[b, S]`` rows), without the model call.

    ``loss_func(loss_mask, per_token_loss) -> (loss sum, num_tokens, report)`` must not
    validate (no per-sample host sync); the executor validates the step's loss once.
    """

    tokens: Tensor
    position_ids: Tensor
    labels: Tensor
    loss_mask: Tensor
    attention_mask: Optional[Tensor]
    padding_mask: Optional[Tensor]
    loss_func: Callable[[Tensor, Tensor], tuple]


LoopRSBFBatchFunc = Callable[[Iterator, nn.Module], LoopMicrobatch]


class ItemKind(enum.Enum):
    """The typed backward items of one sample's chain."""

    CODA = "coda"
    CORE = "core"
    PRELUDE = "prelude"


@dataclass(frozen=True)
class BackwardItem:
    """One backward item: ``sample`` is wave-local; ``round`` is ``d`` for CODA, ``r`` for
    CORE and 0 for PRELUDE. A CORE item carries the plan of its round
    (``LoopProgram.core_plan``), the same plan the round's sweep consulted."""

    kind: ItemKind
    sample: int
    round: int
    plan: Optional[CoreBackwardPlan] = None


@dataclass
class ChainRecord:
    """What one item hands the next item of the same sample's chain."""

    sample: int
    # dL/ds_r for the next CORE item; None after CORE(1).
    grad_state: Optional[Tensor]
    # Running dL/de (activation dtype), summed CORE(d) .. CORE(1) as autograd would.
    grad_injection: Optional[Tensor]
    # dL/dcrit_r (differentiable criteria); after CORE(1) dL/dcrit_0 for the PRELUDE item.
    grad_crit: Optional[Tensor]
    # check_recompute only: the sweep's s_r, taken from the store, compared by CORE(r).
    expected_state: Optional[Tensor] = None


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


@dataclass
class SampleLossRecord:
    """One sample's loss: the per-sample sum and token count of its loss function."""

    key: SampleKey
    depth: int
    loss: Tensor  # detached fp32 per-sample sum
    num_tokens: Tensor  # the loss function's num_tokens
    report: dict


@dataclass
class ExecutorStats:
    """Host-side record of one step (tests and logging)."""

    host_syncs: int = 0  # every executor host sync of the step
    host_syncs_by_wave: list = field(default_factory=list)  # per-round exit reads per wave
    debug_syncs: int = 0  # check_recompute comparisons (test-only)
    rounds_by_wave: list = field(default_factory=list)
    depths: dict = field(default_factory=dict)  # SampleKey -> realized depth
    item_order: list = field(default_factory=list)  # (kind, global index, round)
    store_peaks: list = field(default_factory=list)  # StoreStats per wave
    work: list = field(default_factory=list)  # WorkRecord per cell call
    loss_records: list = field(default_factory=list)  # SampleLossRecord, sample order


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


class LoopRSBFExecutor:
    """One EU's RSBF training step over a ``LoopProgram`` (see the module docstring).

    Args:
        program: the cells over the model.
        pg_collection: the process groups; ``dp`` gives the sample positions and finalize
            receives the collection.
        waves: RSBF passes per step (default ``config.loop_rsbf_waves``).
        fill: ``"drain"`` or ``"sample_order"`` (default ``config.loop_rsbf_fill``).
        check_recompute: assert every replayed round state and injection equals the sweep's
            (bitwise; test-only, one host sync per comparison) and that the RNG states are
            unchanged across a wave.
        poison_on_free: fill freed checkpoints with NaN (tests).
    """

    def __init__(
        self,
        program: LoopProgram,
        *,
        pg_collection,
        waves: Optional[int] = None,
        fill: Optional[str] = None,
        check_recompute: bool = False,
        poison_on_free: bool = False,
    ):
        if pg_collection is None:
            raise ValueError("LoopRSBFExecutor needs a process group collection")
        config = program.config
        validate_rsbf_config(config)
        self.program = program
        self.config = config
        self.pg_collection = pg_collection
        self.waves = config.loop_rsbf_waves if waves is None else waves
        self.fill = config.loop_rsbf_fill if fill is None else fill
        if self.waves < 1:
            raise ValueError(f"waves must be >= 1, got {self.waves}")
        if self.fill not in ("drain", "sample_order"):
            raise ValueError(f"unknown fill {self.fill!r}")
        self.check_recompute = check_recompute
        self.device = program.device
        self.store = LoopStateStore(
            poison_on_free=poison_on_free, act_dtype=program.dtype, device=program.device
        )
        self.stats = ExecutorStats()
        self._exit_reader = _DeviceToHost(self.device)
        self._step_reader = _DeviceToHost(self.device)
        self._pinned: dict = {}
        self._flops: dict = {}  # sequence length -> per-cell forward FLOPs
        self._wave_work: list = []

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
        ``batch_func(data_iterator, model)``; returns one report dict per sample, in sample
        order (every dict with the same keys, ``[value * weight, weight]`` entries)."""
        program, config = self.program, self.config
        validate_rsbf_data_parallel(model)
        if num_microbatches % self.waves:
            raise ValueError(
                f"loop_rsbf_waves ({self.waves}) must divide the microbatches per step "
                f"({num_microbatches})"
            )
        if self.store.wave is not None:  # a previous step raised mid-wave
            self.store.reset()
        self.stats = ExecutorStats()
        program.reset_counters()
        device = self.device
        dp_group = getattr(self.pg_collection, "dp", None)
        dp_rank = dist.get_rank(dp_group) if dp_group is not None else 0
        dp_size = dist.get_world_size(dp_group) if dp_group is not None else 1
        # Once per step (the standard schedule sets the same values after every microbatch).
        set_auxiliary_loss_scales(config, lambda: device, num_microbatches=1, cp_group_size=1)
        self._aux_seed = self._grad_scale(torch.ones((), dtype=torch.float32, device=device))
        self._n_tok_local = torch.zeros((), dtype=torch.int32, device=device)
        self._loss_sum_local = torch.zeros((), dtype=torch.float32, device=device)
        self._plan_mismatches = torch.zeros((), dtype=torch.int32, device=device)
        per_wave = num_microbatches // self.waves
        reports = []
        microbatch = 0
        batch_rows = micro_batch_size
        try:
            for wave in range(self.waves):
                samples = []
                for _ in range(per_wave):
                    batch = batch_func(data_iterator, model)
                    rows = batch.tokens.shape[0]
                    if batch_rows is None:
                        batch_rows = rows
                    if rows != batch_rows:
                        raise ValueError(
                            f"microbatch {microbatch} has {rows} rows, expected {batch_rows}"
                        )
                    samples += self._split(
                        batch, microbatch, iteration, micro_batch_size=rows, dp=(dp_rank, dp_size)
                    )
                    microbatch += 1
                with forward_statistics_weight(1.0 / batch_rows):
                    reports += self._run_wave(wave, samples)
        finally:
            # The samples hold the batch and the loss function (which may hold the model).
            self._samples = self._records = self._n_tok = None
        # The one end-of-step read: the after-exit counter, the plan mismatches and the loss sum.
        final = torch.cat(
            (
                program.after_exit_violations.reshape(1),
                self._plan_mismatches.reshape(1),
                self._loss_sum_local.reshape(1).view(torch.int32),
            )
        )
        self._step_reader.start(final)
        violations, mismatches, loss_bits = self._host_sync(self._step_reader.wait)
        if violations:
            raise RuntimeError(f"{SWEEP_AFTER_EXIT}: {violations} sample-round(s) swept after exit")
        if mismatches:
            raise RuntimeError(f"{PLAN_MISMATCH}: {mismatches} sample(s)")
        step_loss = torch.tensor([loss_bits], dtype=torch.int32).view(torch.float32)[0]
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

    def _split(self, batch, microbatch, iteration, *, micro_batch_size, dp):
        """The rows of one microbatch as samples (c = 1)."""
        dp_rank, dp_size = dp
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

    # -----------------------------------------------------------------------------------
    # One wave: the slot loop
    # -----------------------------------------------------------------------------------

    def _run_wave(self, wave: int, samples: Sequence[SampleInputs]) -> list:
        program, store, device = self.program, self.store, self.device
        n = len(samples)
        store.begin_wave(wave)
        rng = self._rng_state() if self.check_recompute else None
        self._samples = samples
        self._wave_syncs = 0
        self._exited_dev = torch.zeros(n, dtype=torch.int32, device=device)
        exit_round_dev = torch.zeros(n, dtype=torch.int32, device=device)
        keys = [sample.key for sample in samples]
        depths = program.known_depths(keys)
        online = depths is None
        self._depth_dev = None if online else self._upload(("depth", wave), depths)
        exit_buf = torch.zeros(n + 1, dtype=torch.int32, device=device) if online else None
        self._host_exit_round = [0] * n
        self._n_tok = [None] * n
        self._records: list = [None] * n
        self._wave_work = []

        # PRELUDE sub-round (sweep), sample-major.
        for i, sample in enumerate(samples):
            with torch.enable_grad():
                prelude = program.prelude(sample, phase=LoopPhase.SWEEP)
            store.put_injection(i, prelude.injection)
            store.put_crit(i, 0, prelude.crit_state)
            self._work("prelude_sweep", i, 0)
            del prelude

        active = list(range(n))
        pending: deque = deque()
        r = 1
        rounds = 0
        while active or pending:
            newly: list = []
            if active:
                if r > program.max_rounds:
                    raise RuntimeError(
                        f"samples {active} still active after R_max = {program.max_rounds}"
                    )
                rounds = r
                # [F] the forward sub-round over the active samples (sweep mode).
                for i in active:
                    self._core_sweep(i, r, exit_round_dev, exit_buf)
                if online:
                    exit_buf[n : n + 1].copy_(program.after_exit_violations.reshape(1))
                    self._exit_reader.start(exit_buf)
                else:
                    # Known depths: the exits are host data, their chains run this slot.
                    newly = self._retire(self._newly_exited(active, r, depths=depths, bits=None), r)
                    if self.fill == "drain":
                        pending.extend(newly)
            # [B] the slot's backward work, enqueued before the (online) wait.
            if self.fill == "drain":
                while pending:
                    self._run_chain(pending.popleft())
            if active and online:
                bits = self._host_sync(self._exit_reader.wait)
                self._wave_syncs += 1
                if bits[n]:
                    raise RuntimeError(
                        f"{SWEEP_AFTER_EXIT}: {bits[n]} sample-round(s) swept after exit"
                    )
                newly = self._retire(self._newly_exited(active, r, depths=None, bits=bits[:n]), r)
                if self.fill == "drain":  # their chains run in the next slot
                    pending.extend(newly)
            if active:
                retired = set(newly)
                active = [i for i in active if i not in retired]
            r += 1
        if self.fill == "sample_order":
            for i in range(n):
                self._run_chain(i)
        # The executed host plan against the device's first exit rounds (read at step end).
        plan_dev = self._upload(("plan", wave), self._host_exit_round)
        self._plan_mismatches.add_(torch.ne(exit_round_dev, plan_dev).sum(dtype=torch.int32))
        self.stats.store_peaks.append(store.end_wave())
        self.stats.host_syncs_by_wave.append(self._wave_syncs)
        self.stats.rounds_by_wave.append(rounds)
        if rng is not None and not self._rng_unchanged(rng):
            raise RuntimeError("the RNG state changed during an RSBF wave (exit-criterion rule 2)")
        return self._reports(wave, samples)

    def _newly_exited(self, active, r, *, depths, bits) -> list:
        """The active samples the host retires after round ``r``: from the known depths, or
        from the exit bits read this round (online). The device checks this plan."""
        if bits is None:
            return [i for i in active if depths[i] == r]
        return [i for i in active if bits[i]]

    def _retire(self, newly: list, r: int) -> list:
        """Record round ``r`` as the exit round (the depth) of the ``newly`` exited samples."""
        for i in newly:
            if self._host_exit_round[i]:
                raise RuntimeError(f"sample {i} retired twice")
            self._host_exit_round[i] = r
        return newly

    def _core_sweep(self, i: int, r: int, exit_round_dev: Tensor, exit_buf: Optional[Tensor]):
        """CORE-F of sample ``i``, round ``r`` (sweep mode), plus the device exit bookkeeping."""
        program, store = self.program, self.store
        if self._host_exit_round[i]:
            raise RuntimeError(f"{SWEEP_AFTER_EXIT}: the host plan swept retired sample {i}")
        sample = self._samples[i]
        # What the sweep would stash for the round's backward item (nothing for M1's
        # whole-round plan); the item consults the same plan.
        plan = program.core_plan(sample, r)
        if plan.stashed_block_inputs() or plan.rebuild:
            raise NotImplementedError("block / nested CORE items arrive with M8b / M8")
        sl = slice(i, i + 1)
        with torch.enable_grad():
            out = program.core(
                store.get_state(i, r - 1),
                store.get_injection(i),
                sample,
                r=r,
                phase=LoopPhase.SWEEP,
                crit_state=store.get_crit(i, r - 1),
                exited=self._exited_dev[sl],
                depth=None if self._depth_dev is None else self._depth_dev[sl],
            )
        store.put_state(i, r, out.state)
        store.put_crit(i, r, out.crit_state)
        self._work("core_sweep", i, r)
        bit = out.exit.to(torch.int32)
        del out
        first = torch.gt(bit, 0) & torch.eq(exit_round_dev[sl], 0)
        exit_round_dev[sl] = torch.where(first, program.round_tensor(r), exit_round_dev[sl])
        self._exited_dev[sl] |= bit
        if exit_buf is not None:
            exit_buf[sl].copy_(bit)

    # -----------------------------------------------------------------------------------
    # Backward items
    # -----------------------------------------------------------------------------------

    def _run_chain(self, i: int) -> None:
        """CODA -> CORE(d .. 1) -> PRELUDE of sample ``i``."""
        depth = self._host_exit_round[i]
        if depth < 1:
            raise RuntimeError(f"sample {i} has no exit round; its chain cannot run")
        sample = self._samples[i]
        self.stats.depths[sample.key] = depth
        record = self._coda_item(BackwardItem(ItemKind.CODA, i, depth))
        for r in range(depth, 0, -1):
            item = BackwardItem(ItemKind.CORE, i, r, plan=self.program.core_plan(sample, r))
            record = self._core_item(item, record)
        self._prelude_item(BackwardItem(ItemKind.PRELUDE, i, 0), record)

    def _log_item(self, item: BackwardItem) -> str:
        sample = self._samples[item.sample]
        self.stats.item_order.append((item.kind.value, sample.key.global_index, item.round))
        return f"loop_rsbf/sample={sample.key.global_index}/item={item.kind.value}/r={item.round}"

    def _coda_item(self, item: BackwardItem) -> ChainRecord:
        """Coda + head forward and backward on a fresh leaf of ``s_d``; the loss record."""
        i, depth = item.sample, item.round
        store, sample = self.store, self._samples[item.sample]
        label = self._log_item(item)
        state = store.take_state(i, depth) if self.check_recompute else store.get_state(i, depth)
        leaf = state.detach().requires_grad_()
        with torch.profiler.record_function(label), torch.enable_grad():
            out = self.program.coda(leaf, sample)
            torch.autograd.backward(self._grad_scale(out.loss))
        loss = out.loss.detach()
        self._n_tok_local.add_(out.num_tokens)
        self._loss_sum_local.add_(loss)
        self._n_tok[i] = out.num_tokens.detach()
        self._records[i] = SampleLossRecord(
            key=sample.key, depth=depth, loss=loss, num_tokens=self._n_tok[i], report=out.report
        )
        if not self.check_recompute:
            store.free_state(i, depth)
        store.free_crit(i, depth)
        self._work("coda", i, depth)
        return ChainRecord(
            sample=i,
            grad_state=leaf.grad,
            grad_injection=None,
            grad_crit=None,
            expected_state=state if self.check_recompute else None,
        )

    def _core_item(self, item: BackwardItem, record: ChainRecord) -> ChainRecord:
        """Replay round ``r`` from its checkpoints with grad (the plan's blocks in reverse; one
        whole-round block in M1), then back-propagate ``dL/ds_r``, ``dL/dcrit_r`` and the
        criterion's ``aux_loss``."""
        program, store = self.program, self.store
        i, r, plan = item.sample, item.round, item.plan
        sample = self._samples[i]
        label = self._log_item(item)
        if len(plan.blocks) != 1 or plan.rebuild:
            raise NotImplementedError("block / nested CORE items arrive with M8b / M8")
        block = plan.blocks[0]
        previous = None
        if r > 1:
            previous = (
                store.take_state(i, r - 1) if self.check_recompute else store.get_state(i, r - 1)
            )
        round_input = None if previous is None else previous.detach().requires_grad_()
        injection = store.get_injection(i).detach().requires_grad_()
        differentiable = program.criterion.differentiable
        crit = store.get_crit(i, r - 1).detach().requires_grad_(differentiable)
        sl = slice(i, i + 1)
        with torch.profiler.record_function(label), torch.enable_grad():
            with recompute_context(label + "/phase=recompute"):
                out = program.core(
                    round_input,
                    injection,
                    sample,
                    r=r,
                    phase=LoopPhase.RECOMPUTE,
                    crit_state=crit,
                    exited=self._exited_dev[sl],
                    depth=None if self._depth_dev is None else self._depth_dev[sl],
                    block=block,
                )
            if self.check_recompute:
                self._check_equal(out.state, record.expected_state, f"s[{i}, round {r}]")
            outputs, grads = [], []
            if record.grad_state is not None:
                outputs.append(out.state)
                grads.append(record.grad_state)
            if out.aux_loss is not None:
                # The criterion's per-round loss, token-weighted like loop losses (ph: num_tokens
                # * auxiliary) and scaled like the main loss.
                outputs.append(out.aux_loss * self._n_tok[i])
                grads.append(self._aux_seed.reshape(out.aux_loss.shape))
            if record.grad_crit is not None:
                outputs.append(out.crit_state)
                grads.append(record.grad_crit)
            torch.autograd.backward(outputs, grads)
        del out
        record.expected_state = None
        grad_injection = (
            injection.grad
            if record.grad_injection is None
            else record.grad_injection + injection.grad
        )
        if r > 1 and not self.check_recompute:
            store.free_state(i, r - 1)
        store.free_crit(i, r - 1)
        self._work("core_item", i, r)
        return ChainRecord(
            sample=i,
            grad_state=None if round_input is None else round_input.grad,
            grad_injection=grad_injection,
            grad_crit=crit.grad if differentiable else None,
            expected_state=previous if self.check_recompute else None,
        )

    def _prelude_item(self, item: BackwardItem, record: ChainRecord) -> None:
        """Replay the prelude with grad and back-propagate ``dL/de`` (and ``dL/dcrit_0`` into
        a differentiable criterion's ``init_state``)."""
        i = item.sample
        sample = self._samples[i]
        label = self._log_item(item)
        with torch.profiler.record_function(label), torch.enable_grad():
            with recompute_context(label + "/phase=recompute"):
                prelude = self.program.prelude(sample, phase=LoopPhase.RECOMPUTE)
            if self.check_recompute:
                self._check_equal(prelude.injection, self.store.get_injection(i), f"e[{i}]")
            outputs, grads = [prelude.injection], [record.grad_injection]
            if record.grad_crit is not None and prelude.crit_state is not None:
                outputs.append(prelude.crit_state)
                grads.append(record.grad_crit)
            torch.autograd.backward(outputs, grads)
        del prelude
        self.store.free_injection(i)
        self._work("prelude_item", i, 0)

    # -----------------------------------------------------------------------------------
    # Reports, FLOPs, helpers
    # -----------------------------------------------------------------------------------

    def _cell_flops(self, sequence: int) -> dict:
        """Forward FLOPs of one call of each cell on one SBHD row of ``sequence`` tokens (the
        per-symbol table), and the coda's backward (2 x layers + the head's own factors)."""
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
        coda_backward = 2 * coda
        for work, factor in stack.head_work_terms(tokens, squared, sequence):
            coda = coda + work
            coda_backward = coda_backward + factor * work
        table = {
            "prelude": float(prelude),
            "core": float(core),
            "coda": float(coda),
            "coda_backward": float(coda_backward),
        }
        self._flops[sequence] = table
        return table

    def _work(self, kind: str, i: int, r: int) -> None:
        """Append the work record of one cell call of sample ``i`` (Section 10 of the design:
        sweeps are forward work, items recompute + backward, the coda forward + backward)."""
        sample = self._samples[i]
        table = self._cell_flops(sample.sequence)
        forward, recompute, backward = {
            "prelude_sweep": (table["prelude"], 0.0, 0.0),
            "core_sweep": (table["core"], 0.0, 0.0),
            "coda": (table["coda"], 0.0, table["coda_backward"]),
            "core_item": (0.0, table["core"], 2 * table["core"]),
            "prelude_item": (0.0, table["prelude"], 2 * table["prelude"]),
        }[kind]
        record = WorkRecord(
            kind,
            sample.key.global_index,
            r,
            sample.sequence * sample.rows,
            forward=forward,
            recompute=recompute,
            backward=backward,
        )
        self._wave_work.append((i, record))
        self.stats.work.append(record)

    def _reports(self, wave: int, samples: Sequence[SampleInputs]) -> list:
        """One report per sample (sample order): the loss function's report plus the loop
        entries, all ``[value * weight, weight]``."""
        n = len(samples)
        sums = [[0.0, 0.0, 0.0] for _ in range(n)]
        for i, record in self._wave_work:
            totals = sums[i]
            totals[0] += record.forward
            totals[1] += record.backward
            totals[2] += record.recompute
        core = self.program.roles.core
        rows = []
        for i, sample in enumerate(samples):
            tokens = float(sample.sequence * sample.rows)
            depth = self._host_exit_round[i]
            forward, backward, recompute = sums[i]
            rows.append(
                [
                    forward,
                    tokens,
                    backward,
                    tokens,
                    recompute,
                    tokens,
                    float(depth),
                    1.0,
                    tokens * depth * core,
                    tokens,
                    # Every executed round is differentiated (no truncated BPTT under RSBF).
                    tokens * depth * core,
                    tokens,
                ]
            )
        values = self._upload(("report", wave), [v for row in rows for v in row], torch.float32)
        values = values.view(n, 12)
        reports = []
        for i in range(n):
            record = self._records[i]
            self.stats.loss_records.append(record)
            report = dict(record.report)
            report["loop forward flops per token"] = values[i, 0:2]
            report["loop backward flops per token"] = values[i, 2:4]
            report["loop recompute flops per token"] = values[i, 4:6]
            report["loop depth"] = values[i, 6:8]
            report["loop active token layers"] = values[i, 8:10]
            report["loop backward token layers"] = values[i, 10:12]
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
        if expected is None:
            return
        self.stats.debug_syncs += 1
        with _sync_allowed(self.device):
            equal = torch.equal(actual, expected)
        if not equal:
            raise RuntimeError(f"check_recompute: the replayed {what} differs from the sweep")

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
    model: deepcopy / EMA / modelopt clones must not carry them).
    """

    def __init__(self):
        self._cache: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
        self.iteration = 0

    def set_iteration(self, iteration: int) -> None:
        """The training iteration of the next step (keys the sampled depths)."""
        self.iteration = int(iteration)

    def cached_executor(self, model: nn.Module) -> Optional[LoopRSBFExecutor]:
        """The executor cached for ``model``, or None."""
        cached = self._cache.get(get_attr_wrapped_model(model, "head_loss", return_model_obj=True))
        return None if cached is None else cached[1]

    def executor_for(
        self, model: nn.Module, pg_collection, exit_criterion: Optional[ExitCriterion] = None
    ) -> LoopRSBFExecutor:
        """The cached executor of ``model`` (built on first use, rebuilt for another exit
        criterion); it uses ``pg_collection`` from this call on."""
        hybrid = get_attr_wrapped_model(model, "head_loss", return_model_obj=True)
        cached = self._cache.get(hybrid)
        if cached is None or cached[0] is not exit_criterion:
            program = LoopProgram(model, exit_criterion=exit_criterion)
            cached = (exit_criterion, LoopRSBFExecutor(program, pg_collection=pg_collection))
            self._cache[hybrid] = cached
        executor = cached[1]
        executor.pg_collection = pg_collection
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
        executor = self.executor_for(model, pg_collection, exit_criterion)
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
