# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Helpers of the loop parameter store tests (not collected).

``StoreDriver`` runs RSBF-shaped training steps over a local-layer hybrid loop model with the
store's API directly (no executor): every EU walks the same lockstep list of sub-rounds (PRELUDE-F,
CORE-F per round, CODA, CORE-B per round in reverse, PRELUDE-B), each item visiting its segments
in the order of the store's schedule (``segment`` around each forward, ``backward_segment``
around each segment's backward; a split per-segment backward, as the executor runs with a store;
the adapter and head visits are the model's own, ``LoopHybridStack._adapt`` and
``HybridModel.head_loss``, while the driver calls the embedding and ``_run_layer`` directly and
opens those visits itself),
and an EU without an item walks ``null_visitor``. With ``store=None`` the same driver is the
reference: the same items and the same segment-split backward on a model without a store.

``make_store_groups`` builds the shard / replicate groups of a world in the GTP grid's layout
(``tp-cp-gtp_remat-ep-dp-pp`` with TP = CP = EP = PP = 1: shard groups are consecutive ranks,
replicate groups strided), every rank creating every group in the same order.
"""

import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from megatron.core.transformer.loop.param_store import (
    LoopParamStore,
    LoopParamStoreConfig,
    StoreGroups,
)
from megatron.core.transformer.loop.param_store_binding import named_logical_parameters
from megatron.core.transformer.loop.param_store_schedule import SegmentKey
from megatron.core.transformer.loop.program import CoreBackwardPlan
from megatron.core.transformer.loop.rsbf_plan import SubRoundType

PRELUDE_F = SubRoundType.PRELUDE_F
CORE_F = SubRoundType.CORE_F
CORE_B = SubRoundType.CORE_B
CODA = SubRoundType.CODA
PRELUDE_B = SubRoundType.PRELUDE_B


def make_store_groups(world: int, z: int, *, shard_rs: bool = False) -> StoreGroups:
    """Shard groups ``[k z, (k + 1) z)``, replicate groups ``{r, r + z, ...}``, domain = the
    world (every rank creates every group, in the same order)."""
    rank = dist.get_rank()
    shard = replicate = shard_second = None
    for k in range(world // z):
        ranks = list(range(k * z, (k + 1) * z))
        group = dist.new_group(ranks)
        if rank in ranks:
            shard = group
    if shard_rs:
        for k in range(world // z):
            ranks = list(range(k * z, (k + 1) * z))
            group = dist.new_group(ranks)
            if rank in ranks:
                shard_second = group
    for r in range(z):
        ranks = list(range(r, world, z))
        group = dist.new_group(ranks)
        if rank in ranks:
            replicate = group
    return StoreGroups(
        shard=shard, domain=dist.group.WORLD, replicate=replicate, shard_rs=shard_second
    )


_TRIVIAL_GROUPS: List[Tuple[object, StoreGroups]] = []


def trivial_store_groups() -> StoreGroups:
    """The trivial store's (``z = 1``) shard / domain / replicate groups: this rank alone.

    In a one-rank world the world itself. Under torchrun at W ranks (the CI bucket runs every
    file at 8) a one-rank group per rank: every rank creates all W of them in rank order, once
    per default process group (cached; collective only on the first call)."""
    world_size = dist.get_world_size()
    if world_size == 1:
        world = dist.group.WORLD
        return StoreGroups(shard=world, domain=world, replicate=world)
    # pylint: disable=protected-access
    default = dist.distributed_c10d._get_default_group()
    live = dist.distributed_c10d._world.pg_map
    if _TRIVIAL_GROUPS and _TRIVIAL_GROUPS[0][0] is default:
        groups = _TRIVIAL_GROUPS[0][1]
        if groups.shard in live:
            return groups
    rank = dist.get_rank()
    own = None
    for r in range(world_size):
        group = dist.new_group([r])
        if r == rank:
            own = group
    groups = StoreGroups(shard=own, domain=own, replicate=own)
    _TRIVIAL_GROUPS[:] = [(default, groups)]
    return groups


def core_plan_of(form: str, core: int, stride: int = 2) -> CoreBackwardPlan:
    """``whole_round`` / ``block`` (stride) / ``nested`` (stride) core plans."""
    if form == "whole_round":
        return CoreBackwardPlan.whole_round(core)
    if form == "block":
        return CoreBackwardPlan.strided(core, stride)
    if form == "nested":
        return CoreBackwardPlan.nested(core, stride)
    raise ValueError(form)


@dataclass
class Sample:
    """One sample of one EU: tokens / labels ``[1, S]`` and its per-step state."""

    tokens: torch.Tensor
    labels: torch.Tensor
    position_ids: torch.Tensor
    ctx: object = None
    injection: Optional[torch.Tensor] = None
    states: Dict[int, torch.Tensor] = field(default_factory=dict)  # round -> s_r
    stash: Dict[Tuple[int, int], torch.Tensor] = field(default_factory=dict)  # (r, j) -> input
    grad_state: Optional[torch.Tensor] = None
    grad_injection: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


def make_samples(eu: int, count: int, *, vocab: int, seq: int, seed: int = 7) -> List[Sample]:
    """``count`` deterministic samples of EU ``eu``."""
    samples = []
    for i in range(count):
        gen = torch.Generator().manual_seed(seed * 1000 + eu * 100 + i)
        tokens = torch.randint(0, vocab, (1, seq), generator=gen)
        labels = torch.randint(0, vocab, (1, seq), generator=gen)
        samples.append(
            Sample(tokens=tokens, labels=labels, position_ids=torch.arange(seq).unsqueeze(0))
        )
    return samples


class StoreDriver:
    """RSBF-shaped steps over ``hybrid`` (a local loop ``HybridModel``) with ``store`` (or None).

    ``depth``: Huginn rounds per sample (uniform). ``core_plan``: the CORE-B item plan.
    ``lookahead``: pass the next sub-round's type to ``store.subround``.
    """

    def __init__(
        self,
        hybrid,
        store: Optional[LoopParamStore],
        *,
        depth: int,
        core_plan,
        lookahead: bool = True,
    ):
        self.hybrid = hybrid
        self.stack = hybrid.decoder
        self.store = store
        self.depth = depth
        self.core_plan = core_plan
        self.lookahead = lookahead
        roles = self.stack.loop_roles
        self.prelude, self.core_start, self.core_end = (
            roles.prelude,
            roles.core_start,
            roles.core_end,
        )
        self.layers = len(self.stack.layers)
        self.device = next(iter(hybrid.parameters())).device

    # -- contexts -------------------------------------------------------------------------

    def _fwd(self, seg: SegmentKey):
        return self.store.segment(seg) if self.store is not None else contextlib.nullcontext()

    def _bwd(self, seg: SegmentKey):
        if self.store is not None:
            return self.store.backward_segment(seg)
        return contextlib.nullcontext()

    # -- model pieces ---------------------------------------------------------------------

    def _embed(self, sample: Sample) -> torch.Tensor:
        with self._fwd(SegmentKey("embed")):
            x = self.hybrid.embedding(
                input_ids=sample.tokens.to(self.device),
                position_ids=sample.position_ids.to(self.device),
            )
        return x

    def _prepare(self, x: torch.Tensor, sample: Sample) -> torch.Tensor:
        hidden, ctx = self.stack._prepare_forward(
            x,
            None,
            None,
            None,
            packed_seq_params=None,
            padding_mask=None,
            input_ids=None,
            packed_seq_params_by_layout=None,
            cp_layout_plan=None,
        )
        if sample.ctx is None:
            sample.ctx = ctx
        return hidden

    def _layer(self, index: int, x: torch.Tensor, sample: Sample) -> torch.Tensor:
        with self._fwd(SegmentKey("sub", index)):
            return self.stack._run_layer(index, x, sample.ctx)

    def _adapter(self, state: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        # ``LoopHybridStack._adapt`` opens the store's adapter visit itself.
        return self.stack._adapt(state, injection)

    def _head(self, hidden: torch.Tensor, sample: Sample) -> torch.Tensor:
        # ``HybridModel.head_loss`` opens the store's head visit itself (``_finalize_forward``
        # reads replicated parameters only).
        out = self.stack._finalize_forward(hidden)
        hidden, multistream = out if isinstance(out, tuple) else (out, None)
        per_token = self.hybrid.head_loss(
            hidden,
            multistream,
            input_ids=sample.tokens.to(self.device),
            position_ids=sample.position_ids.to(self.device),
            attention_mask=None,
            labels=sample.labels.to(self.device),
            loss_mask=torch.ones_like(sample.labels, dtype=torch.float32).to(self.device),
            padding_mask=None,
            cp_batch=None,
            rotary_pos_emb=None,
        )
        return per_token.float().sum()

    # -- items ----------------------------------------------------------------------------

    def prelude_sweep(self, sample: Sample) -> None:
        """PRELUDE-F: the injection ``e`` (no graph)."""
        with torch.no_grad():
            x = self._prepare(self._embed(sample), sample)
            for index in range(self.prelude):
                x = self._layer(index, x, sample)
        sample.injection = x

    def core_sweep(self, sample: Sample, r: int) -> None:
        """CORE-F of round ``r``: ``s_r`` and the per-sublayer inputs (no graph)."""
        with torch.no_grad():
            state = sample.states.get(r - 1)
            if state is None:
                state = torch.zeros_like(sample.injection)
            x = self._adapter(state, sample.injection)
            for j, index in enumerate(range(self.core_start, self.core_end)):
                sample.stash[(r, j)] = x
                x = self._layer(index, x, sample)
            sample.states[r] = self.stack.loop_norm(x)

    def coda_item(self, sample: Sample) -> None:
        """CODA: coda sublayers and head on a leaf of ``s_d``, segment-split backward."""
        leaf0 = sample.states[self.depth].detach().requires_grad_()
        segments = []
        x = leaf0
        for index in range(self.core_end, self.layers):
            leaf = x if not segments else x.detach().requires_grad_()
            out = self._layer(index, leaf, sample)
            segments.append((SegmentKey("sub", index), leaf, out))
            x = out
        leaf = x.detach().requires_grad_()
        loss = self._head(leaf, sample)
        sample.loss = loss.detach()
        with self._bwd(SegmentKey("head")):
            torch.autograd.backward(loss)
        grad = leaf.grad
        for seg, seg_leaf, out in reversed(segments):
            with self._bwd(seg):
                torch.autograd.backward(out, grad)
            grad = seg_leaf.grad
        sample.grad_state = grad

    def core_item(self, sample: Sample, r: int) -> None:
        """CORE-B of round ``r`` following ``core_plan`` (rebuild, then the blocks in reverse,
        one backward per segment)."""
        plan = self.core_plan
        previous = sample.states.get(r - 1)
        round_input = (
            (torch.zeros_like(sample.injection) if previous is None else previous)
            .detach()
            .requires_grad_()
        )
        injection = sample.injection.detach().requires_grad_()
        layer_stash = {}
        if plan.rebuild:
            with torch.no_grad():
                x = self._adapter(round_input, injection)
                for j in range(plan.rebuild_end):
                    x = self._layer(self.core_start + j, x, sample)
                    layer_stash[j + 1] = x
        grad = sample.grad_state
        for block in reversed(plan.blocks):
            segments = []
            if block.with_prologue:
                adapted = self._adapter(round_input, injection)
                segments.append((SegmentKey("adapter"), None, adapted))
                x = adapted
            else:
                source = layer_stash if plan.rebuild else None
                x = source[block.start] if source is not None else sample.stash[(r, block.start)]
            for j in range(block.start, block.end):
                leaf = x.detach().requires_grad_()
                out = self._layer(self.core_start + j, leaf, sample)
                if block.with_epilogue and j == block.end - 1:
                    out = self.stack.loop_norm(out)
                segments.append((SegmentKey("sub", self.core_start + j), leaf, out))
                x = out
            for seg, leaf, out in reversed(segments):
                with self._bwd(seg):
                    torch.autograd.backward(out, grad)
                if leaf is not None:
                    grad = leaf.grad
        sample.grad_state = round_input.grad
        sample.grad_injection = (
            injection.grad
            if sample.grad_injection is None
            else sample.grad_injection + injection.grad
        )

    def prelude_item(self, sample: Sample) -> None:
        """PRELUDE-B: embedding and prelude recomputed, segment-split backward of ``dL/de``."""
        embedded = self._embed(sample)
        hidden = self._prepare(embedded, sample)
        segments = []
        x = hidden
        for index in range(self.prelude):
            leaf = x.detach().requires_grad_()
            out = self._layer(index, leaf, sample)
            segments.append((SegmentKey("sub", index), leaf, out))
            x = out
        grad = sample.grad_injection
        for seg, leaf, out in reversed(segments):
            with self._bwd(seg):
                torch.autograd.backward(out, grad)
            grad = leaf.grad
        with self._bwd(SegmentKey("embed")):
            torch.autograd.backward(hidden, grad)

    # -- the step -------------------------------------------------------------------------

    def subrounds(self, count: int) -> List[Tuple[SubRoundType, int, int]]:
        """The lockstep list ``(type, sample index, round)`` of one pass over ``count``
        samples per EU."""
        steps = [(PRELUDE_F, i, 0) for i in range(count)]
        for r in range(1, self.depth + 1):
            steps += [(CORE_F, i, r) for i in range(count)]
        steps += [(CODA, i, self.depth) for i in range(count)]
        for r in range(self.depth, 0, -1):
            steps += [(CORE_B, i, r) for i in range(count)]
        steps += [(PRELUDE_B, i, 0) for i in range(count)]
        return steps

    def run_pass(self, samples: Sequence[Sample], count: int) -> None:
        """One pass (a wave): ``count`` sub-rounds per type and round; ``samples`` may be
        shorter (this EU runs nulls for the rest)."""
        steps = self.subrounds(count)
        store = self.store
        for position, (kind, i, r) in enumerate(steps):
            real = i < len(samples)
            nxt = steps[position + 1][0] if position + 1 < len(steps) else None
            context = (
                store.subround(kind, real=real, lookahead=nxt if self.lookahead else None)
                if store is not None
                else contextlib.nullcontext()
            )
            with context:
                if not real:
                    if store is not None:
                        store.null_visitor(kind).visit_all()
                    continue
                sample = samples[i]
                if kind is PRELUDE_F:
                    self.prelude_sweep(sample)
                elif kind is CORE_F:
                    self.core_sweep(sample, r)
                elif kind is CODA:
                    self.coda_item(sample)
                elif kind is CORE_B:
                    self.core_item(sample, r)
                else:
                    self.prelude_item(sample)

    def run_step(self, waves: Sequence[Sequence[Sample]], count: int) -> None:
        """One step: ``begin_step``, one pass per wave, ``end_step``."""
        if self.store is not None:
            self.store.begin_step()
        try:
            for samples in waves:
                for sample in samples:
                    sample.states, sample.stash = {}, {}
                    sample.grad_injection = sample.grad_state = None
                    sample.ctx = None
                self.run_pass(samples, count)
        except BaseException:
            if self.store is not None:
                self.store.abort_step()
            raise
        if self.store is not None:
            self.store.end_step()


# ---------------------------------------------------------------------------------------------
# Gradients of a store run and of the reference
# ---------------------------------------------------------------------------------------------


def attach_shard_grads(store: LoopParamStore) -> None:
    """Zeroed fp32 ``main_grad`` buffers on every canonical shard (what DDP provides)."""
    for tensor in store.tensors():
        tensor.shard.main_grad = torch.zeros(
            tensor.shard.shape, dtype=torch.float32, device=tensor.shard.device
        )


def full_store_grads(store: LoopParamStore, groups: StoreGroups) -> Dict[str, torch.Tensor]:
    """Every store tensor's full gradient: the shard ``main_grad`` summed over the replicate
    group (DDP's role), all-gathered over the shard axis, pad rows dropped."""
    out = {}
    z = store.shard_size
    for tensor in store.tensors():
        grad = tensor.shard.main_grad.detach().clone()
        if dist.get_world_size(groups.replicate) > 1:
            dist.all_reduce(grad, group=groups.replicate)
        if z > 1:
            parts = [torch.empty_like(grad) for _ in range(z)]
            dist.all_gather(parts, grad, group=groups.shard)
            grad = torch.cat(parts, dim=0)
        out[tensor.name] = grad[: tensor.rows]
    return out


def shard_pad_rows(store: LoopParamStore) -> Dict[str, torch.Tensor]:
    """The pad rows of this rank's shard gradients (must be exactly zero)."""
    out = {}
    for tensor in store.tensors():
        per = tensor.rows_padded // store.shard_size
        start = store.shard_rank * per
        real = max(0, min(per, tensor.rows - start))
        if real < per:
            out[tensor.name] = tensor.shard.main_grad[real:]
    return out


def domain_sum(grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """All-reduce (SUM) every gradient over the world."""
    out = {}
    for name, grad in grads.items():
        grad = grad.detach().clone()
        dist.all_reduce(grad)
        out[name] = grad
    return out


def replicated_grads(hybrid) -> Dict[str, torch.Tensor]:
    """``.grad`` of the registered (non-store) parameters, by logical name."""
    out = {}
    for name, param in hybrid.named_parameters():
        if "loop_param_store." in name or param.grad is None:
            continue
        out[name] = param.grad
    return out


def logical_grads(hybrid) -> Dict[str, torch.Tensor]:
    """``.grad`` of every parameter of a model without a store, by name."""
    return {
        name: param.grad
        for name, param in named_logical_parameters(hybrid)
        if param.grad is not None
    }


def compare(actual: torch.Tensor, expected: torch.Tensor, *, rtol: float = 1e-5) -> float:
    """``max |a - e| / (rtol |e| + 1e-6 max |e|)`` (<= 1 passes; the stage-2 gradient rule)."""
    expected = expected.float()
    actual = actual.float()
    scale = expected.abs().max().clamp_min(1e-30)
    bound = rtol * expected.abs() + 1e-6 * scale
    return float(((actual - expected).abs() / bound).max())


def install_store(hybrid, groups: StoreGroups, z: int, **config) -> LoopParamStore:
    """Install a store with ``min_numel = 1`` (everything 2-D is stored) and shard grads."""
    config.setdefault("min_numel", 1)
    store = LoopParamStore.install(hybrid, LoopParamStoreConfig(shard_size=z, **config), groups)
    attach_shard_grads(store)
    return store
