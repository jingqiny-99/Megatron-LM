# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Module GTP weight remat on a fixed-depth DSv4 loop (GPU; the lifted guard
``GTP_WEIGHT_REMAT``).

Two training steps with an SGD step between them (no momentum: the step-2 weights are a plain
function of step 1's gradients, so a gather issued before the update and read after it fails
the step-2 gate; the second step is also the first with DDP's overlapped reduction armed) of
the model of ``strategy_grad_utils`` (a core of 8 sublayers run exactly 4 times, block + edge
``loop_recompute`` with stride 4, three microbatches of one row per rank, per-token loss,
``finalize_model_grads``; the persistent buffers restored before every step), from the same
full weights and rows in every arm. Step 2 is teacher-forced (``strategy_grad_utils``): after
its step-1 gradient is read (and no GTP reduce-scatter is pending), every arm's ``main_grad``
takes its reference's step-1 gradient (a GTP shard: its rows), and the arm runs its own
optimizer step, parameter gather and step 2:

* ``ref``: no GTP (``gtp_remat_size = 1``), DDP over the whole data-parallel group, with the
  precise embedding reference (``strategy_grad_utils.install_per_use_fp32_embedding``): each of
  the embedding's two lookups per microbatch (main and MTP input) adds its BF16 gradient into the
  FP32 ``main_grad`` on its own. GTP reduce-scatters each use's gradient on its own in FP32; the
  deterministic-mode lookup would instead sum the two uses in BF16 first, about 1.7e-3 off;
* ``gtp``: dense and expert weight remat of degree ``G`` (``LOOP_GTP_DEGREE``, default the world
  size: 2 or 4), DDP with the distributed optimizer, prefetch and asynchronous reduce-scatter
  on (the defaults), ``GTP_CONFIG.verify_prefetched_gathers`` on (set by the loop stack);
* ``gtp_overlap``: ``gtp`` with DDP's ``overlap_grad_reduce`` and small buckets (several bucket
  groups, so a group can start its reduction while another group's GTP shards still have uses
  to come; a GTP shard becomes grad-ready once per use, so a core shard's recorded count is 4);
* ``gtp_ep`` (against its own ``ref_ep`` at the same EP): the full recipe's layout, dense GTP of
  degree ``G``, experts expert-parallel (``LOOP_GTP_EP``, default 2) with no expert GTP, so the
  expert gradients take DDP's expert data-parallel reduction;
* ``gtp_simple``: ``gtp`` with prefetch and asynchronous reduce-scatter off (every use gathers
  and reduce-scatters synchronously: GTP's simplest path);
* ``noverify`` (diagnostic, not gated; forced): ``gtp`` with the prefetch verification turned
  off after construction, i.e. the static chains trusted.

Every use of a GTP weight gathers it and reduce-scatters its weight gradient into the shard's
``main_grad`` (a weight's next reduce-scatter first finalizes, i.e. adds, the previous one); a
core weight is used 4 times per microbatch, in the forward and again in the block recompute.

* Every parameter's reduced fp32 gradient (GTP shards all-gathered over their group, padding
  stripped) of both steps matches ``ref``'s: cosine >= 0.9999 and relative difference <= 1e-3 (a
  gradient zero to fp32 precision: absolute, at most 1e-6 of the whole gradient's norm); the
  losses of both steps and every row's LM loss equal ``ref``'s within 1e-6. Before step 2 is
  gated, the precondition: no GTP reduce-scatter was pending at the forcing, the forced gradient
  reads back on every rank, and every weight at the end of step 2 (the step-1 update the
  optimizer installed; a GTP shard is gathered per use from it) equals ``ref``'s exactly (a GTP
  shard: its rows of the zero-padded weight, so the padding rows stay zero).
* Given the same step-1 gradient the un-forced ``gtp`` path agrees at step 2: ``gtp_unforced``
  against ``ref_from_gtp`` (``ref`` stepping on its step-1 gradient), from exactly equal
  weights. The un-forced step 2 against ``ref`` is printed (diagnostic).
* Reduce-scatters per GTP weight (a dense weight, or a grouped expert projection's leading
  weight) and microbatch, one per backward use: 4 for a core weight, 1 for a prelude, coda or
  MTP weight, 1 + ``MTP_LAYERS`` for the embedding and the output layer (main and MTP heads); a
  GTP weight with no expected count fails. Per step, the output layer's accumulated wgrad (its
  uses add into one buffer, flushed once per step) is reduce-scattered once more by its flush;
  no other weight flushes. Under ``overlap_grad_reduce`` the recorded grad-ready counts: 4 for a
  core weight, 1 for a prelude, coda or MTP weight, 2 for the embedding (one per use), 1 for the
  output layer (one accumulated reduce-scatter).
* The embedding's weight gradient reaches its reduce-scatter in FP32 (``main_grad``'s dtype) in
  every arm and step; the reference counts ``ROWS x (1 + MTP_LAYERS)`` lookups per step. The
  reason for the precise reference, measured: ``ref_plain`` (the plain lookup, step 1) fails
  the gate against ``ref`` on the embedding, and on no other parameter.
* The prefetch verification's decisions are printed: how many consumes found a prefetch handle
  whose buffer did not hold this consume's gather (they gathered on demand).
* Teeth, must fail (step 1): ``drop_rs``, the second backward use's weight gradient of every
  core GTP weight and of the embedding zeroed before its reduce-scatter (one round's
  reduce-scatter dropped, and one embedding use): at least 90% of the core GTP weights and the
  embedding fail the gate. ``emb_bf16``, the embedding's gradient reduce-scattered in BF16 (not
  widened: the path before the FP32 fix): the embedding fails the gate.

Needs TransformerEngine >= 2.19 (GTP). Launch once per degree: ``torchrun --nproc-per-node 2 -m
pytest -q tests/unit_tests/transformer/loop/test_loop_gtp_gpu.py`` and ``--nproc-per-node 4``
(``LOOP_GTP_DEGREE=2`` on 4 GPUs gives replicate DP 2 x GTP 2). ``LOOP_GTP_EP`` sets the
``gtp_ep`` arm's expert-parallel size (1 skips the arm). ``MBS_REPORT_DIR`` dumps every
per-parameter report.
"""

import collections
import os

import pytest
import torch
import torch.distributed as dist

from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA, determinism_env

try:
    from megatron.core.tensor_parallel.gtp_api import HAVE_GTP
except ImportError:  # pragma: no cover
    HAVE_GTP = False

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
DEGREE = int(os.environ.get("LOOP_GTP_DEGREE", str(WORLD)))
pytestmark = pytest.mark.skipif(
    not CUDA or not HAVE_GTP or WORLD < 2 or DEGREE < 2 or WORLD % DEGREE != 0,
    reason="GTP needs CUDA, TransformerEngine >= 2.19 and a GTP degree >= 2 dividing W",
)

STEPS = 2
MIN_FAILING_CORE_FRACTION = 0.9
BUCKET_SIZE = 200_000  # elements: several bucket groups for the tiny model (overlap arm)
GTP_EP = int(os.environ.get("LOOP_GTP_EP", "2"))
RUN_GTP_EP = GTP_EP > 1 and WORLD % GTP_EP == 0 and sgu.NUM_EXPERTS % GTP_EP == 0
ARMS = ["gtp", "gtp_overlap", "gtp_simple"] + (["gtp_ep"] if RUN_GTP_EP else [])
REFERENCE = {arm: "ref_ep" if arm == "gtp_ep" else "ref" for arm in ARMS}


@pytest.fixture(scope="module", autouse=True)
def _environment():
    if not CUDA or not HAVE_GTP:
        yield
        return
    from tests.unit_tests.test_utilities import Utils

    with determinism_env():
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(False)
            Utils.destroy_model_parallel()


def _initialize(degree, expert_degree=None, ep=1):
    """Model parallelism with GTP weight remat of ``degree`` (1: none), expert GTP of
    ``expert_degree`` (default ``degree``) and expert parallelism ``ep``, and a clean GTP
    runtime (chains, buffer cache, ledger, the global parameter list)."""
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        _GTP_PARAMS,
        get_global_GTP_cache,
        reset_gtp_state,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from tests.unit_tests.test_utilities import Utils

    reset_gtp_state()
    _GTP_PARAMS.clear()
    get_global_GTP_cache().clear()
    Utils.initialize_model_parallel(
        1,
        1,
        gtp_remat_size=degree,
        expert_gtp_remat_size=degree if expert_degree is None else expert_degree,
        expert_model_parallel_size=ep,
    )
    model_parallel_cuda_manual_seed(123)


def _is_gtp(param):
    from megatron.core.tensor_parallel.gtp_api import is_gtp_param

    return is_gtp_param(param)


def _gtp_layout(param):
    """``(rows, group size, group rank)`` of a GTP shard (``sgu.gtp_rows``'s arguments)."""
    return param.shape[0], param.group.size(), param.group.rank()


def _shard_of(param, full):
    """The part of ``full`` (a full tensor of ``param``'s) that ``param`` holds: a GTP shard's
    rows of the zero-padded tensor, or all of it."""
    return sgu.gtp_rows(full, *_gtp_layout(param)) if _is_gtp(param) else full


def _load_full_state(model, state):
    """Load full (unsharded) weights into a model whose GTP weights are shards: shard ``r`` of
    a weight holds rows ``[r S, (r + 1) S)`` of the weight padded to ``S x G`` rows."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data.copy_(_shard_of(param, state[name].to("cuda", param.dtype)))
        _restore_buffers(model, state)


def _restore_buffers(model, state):
    """Reset the persistent buffers (the router expert bias that ``finalize_model_grads``
    updates) so every step of every arm routes from the same state."""
    with torch.no_grad():
        for name, buf in model.named_buffers():
            if name in state and torch.is_tensor(state[name]):
                buf.copy_(state[name])


def _full_grads(ddp, model):
    """Every parameter's reduced fp32 gradient: the distributed optimizer's shards gathered
    (``ddp_grads``), then every GTP shard's gradient all-gathered over its GTP group, padding
    stripped (collective, in parameter order on every rank)."""
    grads = sgu.ddp_grads(ddp, model)
    for name, param in model.named_parameters():
        if not param.requires_grad or not _is_gtp(param):
            continue
        shard = grads[name].contiguous()
        group = param.group
        full = torch.empty(
            (shard.shape[0] * group.size(), *shard.shape[1:]), dtype=shard.dtype, device="cuda"
        )
        dist.all_gather_into_tensor(full, shard, group=group)
        grads[name] = full[: param._unsharded_shape[0]].clone()
    return grads


def _ddp(model, *, overlap=False, distributed_optimizer=True):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.process_groups_config import ProcessGroupCollection

    return DistributedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            overlap_grad_reduce=overlap,
            use_distributed_optimizer=distributed_optimizer,
            grad_reduce_in_fp32=True,
            bucket_size=BUCKET_SIZE if overlap else None,
        ),
        model,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )


class _Instrument:
    """Counts, per GTP weight, its reduce-scatters per backward use and microbatch
    (``GTPShardedParam.wgrad_reduce_scatter``; a call made inside ``flush_accumulated_wgrad`` is
    the flush of an accumulated wgrad, counted per step in ``flush_rs`` and not as a use), the
    dtypes the uses hand it (``dtypes``), the local norm of every use of the weights in
    ``norm_ids`` (``norms``), and the prefetch verification's decisions (``_GatherLedger.take``).

    Mutations: ``drops`` ({id(weight): use}) zeroes that backward use's (1-based, per microbatch)
    weight gradient before its reduce-scatter; the weights in ``bf16_ids`` are not widened to
    ``main_grad``'s dtype before their reduce-scatter (``widen_wgrad_for_reduce_scatter`` is the
    identity for them: the BF16 reduce-scatter of the path before the FP32 fix)."""

    def __init__(self, drops=None, bf16_ids=(), norm_ids=()):
        from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp

        self.gtp = gtp
        self.original_rs = gtp.GTPShardedParam.wgrad_reduce_scatter
        self.original_flush = gtp.GTPShardedParam.flush_accumulated_wgrad
        self.original_widen = gtp.GTPShardedParam.widen_wgrad_for_reduce_scatter
        self.original_take = gtp._GATHER_LEDGER.take
        self.microbatch = -1
        self.rs = collections.Counter()  # (microbatch, id(weight)) -> reduce-scatters (uses)
        self.flush_rs = collections.Counter()  # id(weight) -> flush reduce-scatters
        self.dtypes = collections.defaultdict(set)  # id(weight) -> dtypes of its uses' wgrads
        self.norms = collections.defaultdict(list)  # id(weight) -> local norm of every use
        self.takes = collections.Counter()  # True / False
        self.drops = dict(drops or {})
        self.bf16_ids = set(bf16_ids)
        self.norm_ids = set(norm_ids)
        self.flush_depth = 0
        instrument = self

        def counting_rs(weight, wgrad, *args, **kwargs):
            if instrument.flush_depth:
                instrument.flush_rs[id(weight)] += 1
                return instrument.original_rs(weight, wgrad, *args, **kwargs)
            key = (instrument.microbatch, id(weight))
            instrument.rs[key] += 1
            tensors = list(wgrad) if isinstance(wgrad, (list, tuple)) else [wgrad]
            instrument.dtypes[id(weight)].update(tensor.dtype for tensor in tensors)
            if id(weight) in instrument.norm_ids:
                instrument.norms[id(weight)].append(float(tensors[0].float().norm()))
            if instrument.drops.get(id(weight)) == instrument.rs[key]:
                for tensor in tensors:
                    tensor.zero_()
            return instrument.original_rs(weight, wgrad, *args, **kwargs)

        def counting_flush(weight, *args, **kwargs):
            instrument.flush_depth += 1
            try:
                return instrument.original_flush(weight, *args, **kwargs)
            finally:
                instrument.flush_depth -= 1

        def widen(weight, wgrad):
            if id(weight) in instrument.bf16_ids:
                return wgrad
            return instrument.original_widen(weight, wgrad)

        def counting_take(weight, slot, buffer):
            current = instrument.original_take(weight, slot, buffer)
            instrument.takes[current] += 1
            return current

        self.counting_rs = counting_rs
        self.counting_flush = counting_flush
        self.widen = widen
        self.counting_take = counting_take

    def on_microbatch(self, index):
        self.microbatch = index

    def step_record(self):
        """This step's records; the use, flush, dtype and norm records are then cleared."""
        record = dict(
            rs=dict(self.rs),
            flush_rs=dict(self.flush_rs),
            dtypes={pid: set(dtypes) for pid, dtypes in self.dtypes.items()},
            norms={pid: list(norms) for pid, norms in self.norms.items()},
            takes=dict(self.takes),
        )
        for counter in (self.rs, self.flush_rs, self.dtypes, self.norms):
            counter.clear()
        return record

    def __enter__(self):
        cls = self.gtp.GTPShardedParam
        cls.wgrad_reduce_scatter = self.counting_rs
        cls.flush_accumulated_wgrad = self.counting_flush
        cls.widen_wgrad_for_reduce_scatter = self.widen
        self.gtp._GATHER_LEDGER.take = self.counting_take
        return self

    def __exit__(self, *exc):
        cls = self.gtp.GTPShardedParam
        cls.wgrad_reduce_scatter = self.original_rs
        cls.flush_accumulated_wgrad = self.original_flush
        cls.widen_wgrad_for_reduce_scatter = self.original_widen
        del self.gtp._GATHER_LEDGER.take  # back to the class method


def _run_reference(state, ep=1, forced=None, precise=True, steps=STEPS):
    """``steps`` steps without GTP, with the precise embedding reference
    (``sgu.install_per_use_fp32_embedding``; ``precise=False``: the plain deterministic lookup,
    whose autograd sums the two uses in BF16); returns ``[(step result, grads)]`` and a record:
    the full weights at the end of the last step (``weights``), the embedding lookups per step
    that went through the precise reference (``lookups``) and, with ``forced`` (step-1 gradients
    to step on: the teacher forcing), the read-back count (``readback``)."""
    _initialize(1, ep=ep)
    model = sgu.build_model(state, expert_model_parallel_size=ep)
    counter = sgu.install_per_use_fp32_embedding(model) if precise else None
    ddp = _ddp(model, distributed_optimizer=False)
    opt = sgu.optimizer(ddp)
    out = []
    info = dict(readback=None, pending=0, lookups=[])
    for step in range(steps):
        _restore_buffers(model, state)
        opt.zero_grad()
        result = sgu.run_step(ddp, model, sgu.make_rows(step))
        if counter is not None:
            info["lookups"].append(counter.lookups)
            counter.lookups = 0
        out.append((result, sgu.ddp_grads(ddp, model)))
        if step + 1 < steps:
            if forced is not None:
                sgu.force_main_grads(model, forced)
                info["readback"] = sgu.agreed(
                    len(sgu.mismatches(sgu.ddp_grads(ddp, model), forced))
                )
            sgu.step_optimizer(opt)
    info["weights"] = {name: param.detach().clone() for name, param in model.named_parameters()}
    del opt, ddp, model
    torch.cuda.empty_cache()
    return out, info


def _run_gtp(
    state,
    *,
    overlap=False,
    simple=False,
    verify=True,
    drop_use=None,
    drop_embedding_use=None,
    bf16_embedding=False,
    expert_degree=None,
    ep=1,
    steps=STEPS,
    forced=None,
):
    """``steps`` steps of one GTP arm; returns ``[(step result, grads, record)]`` and the info:
    the GTP names and consumers, the embedding's id (``emb_id``), the overlap's recorded counts,
    the weights at the end of the last step (``weights``; GTP shards) with each GTP shard's
    layout (``layout``) and, with
    ``forced`` (full step-1 gradients to step on: the teacher forcing), the GTP reduce-scatters
    still pending at the forcing (``pending``) and the read-back count (``readback``), both
    agreed over the ranks."""
    from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTP_CONFIG
    from megatron.core.tensor_parallel.gtp_api import (
        classify_gtp_remat_chains,
        configure_gtp_remat_from_recipe,
    )

    saved = (GTP_CONFIG.weight_prefetch, GTP_CONFIG.async_reduction)
    expert_degree = DEGREE if expert_degree is None else expert_degree
    _initialize(DEGREE, expert_degree=expert_degree, ep=ep)
    configure_gtp_remat_from_recipe(calculate_per_token_loss=True)
    GTP_CONFIG.weight_prefetch = not simple
    GTP_CONFIG.async_reduction = not simple
    try:
        model = sgu.build_model(
            None,
            gtp_weight_remat_size=DEGREE,
            expert_gtp_weight_remat_size=expert_degree,
            expert_model_parallel_size=ep,
        )
        assert GTP_CONFIG.verify_prefetched_gathers, "the loop stack turns the verification on"
        GTP_CONFIG.verify_prefetched_gathers = verify
        _load_full_state(model, state)
        ddp = _ddp(model, overlap=overlap)
        classify_gtp_remat_chains([ddp])
        opt = sgu.optimizer(ddp)
        gtp_names = {id(p): name for name, p in model.named_parameters() if _is_gtp(p)}
        assert gtp_names, "no GTP-sharded weight in the model"
        # The weights GTP reduce-scatters through: a dense weight, or the leading weight of a
        # grouped expert projection (its weight_list covers every local expert).
        consumers = {
            id(p)
            for p in model.parameters()
            if _is_gtp(p) and (not p.is_routed_expert or p.weight_list is not None)
        }
        core_ids = {pid for pid, name in gtp_names.items() if sgu.role(name)[0] == "core"}
        emb = model.embedding.word_embeddings.weight
        assert _is_gtp(emb) and id(emb) in consumers, "the embedding is not a GTP consumer"
        drops = {pid: drop_use for pid in core_ids & consumers} if drop_use else {}
        if drop_embedding_use:
            drops[id(emb)] = drop_embedding_use
        # The embedding's use norms are printed by the mutation tests only; reading one syncs
        # the host mid-backward, so the gated arms keep their timing.
        mutated = bool(drop_embedding_use or bf16_embedding)
        instrument = _Instrument(
            drops=drops,
            bf16_ids={id(emb)} if bf16_embedding else (),
            norm_ids={id(emb)} if mutated else (),
        )
        out = []
        forcing = dict(pending=None, readback=None)
        with instrument:
            for step in range(steps):
                _restore_buffers(model, state)
                opt.zero_grad()
                result = sgu.run_step(
                    ddp, model, sgu.make_rows(step), on_microbatch=instrument.on_microbatch
                )
                record = instrument.step_record()
                out.append((result, _full_grads(ddp, model), record))
                instrument.takes.clear()
                if step + 1 < steps:
                    if forced is not None:
                        # Nothing may land in main_grad after the forcing.
                        pending = len(gtp._GTP_PENDING_WGRAD_ACCUM) + sum(
                            getattr(p, "_wgrad_rs_handle", None) is not None
                            for p in model.parameters()
                            if _is_gtp(p)
                        )
                        forcing["pending"] = sgu.agreed(pending)
                        sgu.force_main_grads(
                            model, forced, shard=lambda name, param, full: _shard_of(param, full)
                        )
                        forcing["readback"] = sgu.agreed(
                            len(sgu.mismatches(_full_grads(ddp, model), forced))
                        )
                    sgu.step_optimizer(opt)
        golden = {}
        groups = ddp.bucket_groups + ddp.expert_parallel_bucket_groups
        if overlap:
            for group in groups:
                for param, count in group.golden_per_param_grad_ready_counts.items():
                    if id(param) in gtp_names:
                        golden[gtp_names[id(param)]] = count
        info = dict(
            gtp_names=gtp_names,
            consumers=consumers,
            emb_id=id(emb),
            core_names=sorted(gtp_names[i] for i in core_ids),
            # Bucket groups holding a GTP shard: a group of its own per shard set would never
            # start before another group's last use.
            gtp_groups=sum(
                any(id(param) in gtp_names for param in group.params) for group in groups
            ),
        )
        info["golden"] = golden
        info.update(forcing)
        info["weights"] = {name: p.detach().clone() for name, p in model.named_parameters()}
        info["layout"] = {
            name: _gtp_layout(p) for name, p in model.named_parameters() if _is_gtp(p)
        }
        del opt, ddp, model
        torch.cuda.empty_cache()
        return out, info
    finally:
        GTP_CONFIG.weight_prefetch, GTP_CONFIG.async_reduction = saved
        GTP_CONFIG.verify_prefetched_gathers = False


@pytest.fixture(scope="module")
def results():
    """Every arm, run once from the same full weights and rows; the arms take their reference's
    step-1 gradient before their optimizer step (teacher forcing). The preconditions are agreed
    over the ranks at the end (in a fixed order) and asserted by the tests."""
    _initialize(1)
    state = sgu.clone_state(sgu.build_model())
    out = {}
    out["ref"] = _run_reference(state)
    # Diagnostic: the plain lookup's reference (the two uses summed in BF16), one step.
    out["ref_plain"] = _run_reference(state, precise=False, steps=1)
    forced = out["ref"][0][0][1]
    out["gtp"] = _run_gtp(state, forced=forced)
    out["gtp_overlap"] = _run_gtp(state, overlap=True, forced=forced)
    out["gtp_simple"] = _run_gtp(state, simple=True, forced=forced)
    out["noverify"] = _run_gtp(state, verify=False, forced=forced)
    # The second backward use of every core GTP weight and of the embedding (the main lookup if,
    # as expected, the MTP lookup's backward runs first; either use's loss is far over the gate).
    # An embedding gradient feeds no dgrad, so the core weights' mutation is unaffected.
    out["drop_rs"] = _run_gtp(state, drop_use=2, drop_embedding_use=2, steps=1)
    out["emb_bf16"] = _run_gtp(state, bf16_embedding=True, steps=1)
    # Un-forced: gtp on its own step-1 gradient, and ref stepping on that same gradient.
    out["gtp_unforced"] = _run_gtp(state)
    out["ref_from_gtp"] = _run_reference(state, forced=out["gtp_unforced"][0][0][1])
    if RUN_GTP_EP:
        # Expert weights depend on the EP layout: the arm and its reference share a state
        # built at that EP (per rank: its local experts; the dense weights are replicated).
        _initialize(1, ep=GTP_EP)
        state_ep = sgu.clone_state(sgu.build_model(expert_model_parallel_size=GTP_EP))
        out["ref_ep"] = _run_reference(state_ep, ep=GTP_EP)
        out["gtp_ep"] = _run_gtp(
            state_ep, expert_degree=1, ep=GTP_EP, forced=out["ref_ep"][0][0][1]
        )
    for arm in ARMS + ["noverify"]:
        info = out[arm][1]
        start = _start_mismatches(info, out[REFERENCE.get(arm, "ref")][1]["weights"])
        info["forcing"] = sgu.forcing_record(info["readback"], start)
    info = out["ref_from_gtp"][1]
    start = _start_mismatches(out["gtp_unforced"][1], info["weights"])
    info["forcing"] = sgu.forcing_record(info["readback"], start)
    return out


def _start_mismatches(info, reference):
    """Where the weights at the end of a GTP arm's step 2 (``info["weights"]``, GTP shards)
    differ from ``reference``'s full weights, each GTP shard against its rows of the
    zero-padded weight (local)."""
    expected = {}
    for name, value in info["weights"].items():
        full = reference[name].to(value.device)
        layout = info["layout"].get(name)
        expected[name] = sgu.gtp_rows(full, *layout) if layout else full
    return sgu.mismatches(info["weights"], expected)


def _name(text):
    return sgu.tag(f"GTP G{DEGREE}", text)


def _assert_forced(results, arm):
    """The teacher-forcing precondition of a GTP arm: no GTP reduce-scatter pending at the
    forcing, the forced gradient read back, and the weights at the end of step 2 equal to its
    reference's (a GTP shard: its rows) exactly."""
    info = results[arm][1]
    assert (
        info["pending"] == 0
    ), f"{arm}: {info['pending']} GTP reduce-scatters pending at the forcing"
    sgu.assert_teacher_forced(info["forcing"], arm, f"{REFERENCE.get(arm, 'ref')}'s")


@pytest.mark.parametrize("arm", ARMS)
def test_step_two_starts_from_the_reference_state(results, arm):
    _assert_forced(results, arm)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("step", range(STEPS))
def test_gradients_match_the_reference(results, arm, step):
    if step > 0:
        _assert_forced(results, arm)
    _, reference = results[REFERENCE[arm]][0][step]
    _, grads, record = results[arm][0][step]
    print(f"[{_name(f'{arm} step {step + 1}')}] verification takes {record['takes']}", flush=True)
    report, failing = sgu.report_and_fail(_name(f"{arm} step {step + 1} vs ref"), grads, reference)
    assert not failing, sgu.mbs.worst(report, failing)


@pytest.mark.parametrize("arm", ARMS)
def test_losses_match_the_reference(results, arm):
    for step in range(STEPS):
        if step > 0:
            _assert_forced(results, arm)
        reference, _ = results[REFERENCE[arm]][0][step]
        result, _, _ = results[arm][0][step]
        sgu.assert_losses_match(result, reference, _name(f"{arm} step {step + 1}"))


def test_step_two_agrees_given_the_same_step_one_gradient(results):
    """The un-forced path: ``gtp_unforced`` steps on its own step-1 gradient and
    ``ref_from_gtp`` (``ref``) on that same gradient. From exactly equal weights, their step-2
    gradients pass the gate and their losses are equal."""
    sgu.assert_teacher_forced(
        results["ref_from_gtp"][1]["forcing"], "ref_from_gtp", "gtp_unforced's"
    )
    reference_result, reference = results["ref_from_gtp"][0][1]
    result, grads, _ = results["gtp_unforced"][0][1]
    report, failing = sgu.report_and_fail(
        _name("gtp_unforced step 2 vs ref_from_gtp"), grads, reference
    )
    assert not failing, sgu.mbs.worst(report, failing)
    sgu.assert_losses_match(result, reference_result, _name("gtp_unforced step 2"))


def test_report_unforced_step_two(results):
    """Diagnostic only (prints, asserts nothing): step 2 without teacher forcing against
    ``ref``: ``gtp_unforced`` (its own step-1 gradient), and ``ref_from_gtp`` (``ref`` on
    ``gtp_unforced``'s step-1 gradient, which differs from its own by fp32 reassociation)."""
    reference_result, reference = results["ref"][0][1]
    for arm in ("gtp_unforced", "ref_from_gtp"):
        result, grads = results[arm][0][1][:2]
        report, failing = sgu.report_and_fail(
            _name(f"{arm} step 2 vs ref (diagnostic)"), grads, reference
        )
        print(f"[{_name(arm)}] step 2 failing {len(failing)} of {len(report)}", flush=True)
        rel = sgu.mbs.relative_loss_differences(result.losses, reference_result.losses)
        print(f"[{_name(f'{arm} step 2')}] losses rel {rel} (diagnostic)", flush=True)


EMBEDDING = "embedding.word_embeddings.weight"
OUTPUT_LAYER = "output_layer.weight"


def _uses_per_microbatch(name):
    """A GTP weight's backward uses per microbatch, from the model's structure: a core weight
    once per round, a prelude, coda or MTP weight once, the embedding and the output layer once
    for the main input / head and once per MTP layer. Raises for a weight it does not know."""
    where = sgu.role(name)[0]
    if where == "core":
        return sgu.ROUNDS
    if where in ("prelude", "coda", "mtp"):
        return 1
    if name in (EMBEDDING, OUTPUT_LAYER):
        return 1 + sgu.MTP_LAYERS
    raise AssertionError(f"no expected use count for the GTP weight {name}")


def _grad_ready_per_microbatch(name):
    """A GTP weight's recorded grad-ready count under overlap_grad_reduce: one per reduce-scatter
    of a use, except the output layer, whose uses add into one buffer reduce-scattered once (the
    accumulation of a weight with several backwards). Raises for a weight it does not know."""
    if name == OUTPUT_LAYER:
        return 1
    return _uses_per_microbatch(name)


@pytest.mark.parametrize("arm", ARMS)
def test_one_reduce_scatter_per_use(results, arm):
    """Every GTP weight is reduce-scattered once per backward use and microbatch
    (``_uses_per_microbatch``); per step, the output layer's accumulated wgrad is flushed by one
    more reduce-scatter and no other weight flushes."""
    steps, info = results[arm]
    names = {pid: name for pid, name in info["gtp_names"].items() if pid in info["consumers"]}
    assert EMBEDDING in names.values() and OUTPUT_LAYER in names.values(), sorted(names.values())
    for step, (_, _, record) in enumerate(steps):
        for microbatch in range(sgu.ROWS):
            for pid, name in names.items():
                expected = _uses_per_microbatch(name)
                got = record["rs"].get((microbatch, pid), 0)
                assert got == expected, (arm, step, microbatch, name, got)
        for pid, name in names.items():
            expected = 1 if name == OUTPUT_LAYER else 0
            got = record["flush_rs"].get(pid, 0)
            assert got == expected, (arm, step, "flush", name, got)


def test_overlap_counts_one_grad_ready_per_use(results):
    """Under overlap_grad_reduce a GTP shard is grad-ready once per reduce-scatter
    (``_grad_ready_per_microbatch``): 4 for a core weight, 1 for a prelude, coda or MTP weight, 2
    for the embedding, 1 for the output layer."""
    _, info = results["gtp_overlap"]
    golden = info["golden"]
    print(
        f"[{_name('gtp_overlap golden counts')}] {golden}; bucket groups with GTP shards "
        f"{info['gtp_groups']}",
        flush=True,
    )
    assert golden
    # Several groups: one can start its reduction while another's shards still have uses.
    assert info["gtp_groups"] > 1, info["gtp_groups"]
    assert EMBEDDING in golden and OUTPUT_LAYER in golden, sorted(golden)[:8]
    for name, count in golden.items():
        assert count == _grad_ready_per_microbatch(name), (name, count)


def test_gtp_ep_arm_keeps_experts_out_of_gtp(results):
    """The full recipe's layout: no expert weight is a GTP shard, every dense GTP consumer is."""
    if not RUN_GTP_EP:
        pytest.skip("LOOP_GTP_EP does not give an expert-parallel layout here")
    model_info = results["gtp_ep"][1]
    names = model_info["gtp_names"].values()
    assert names and not any(".experts." in name for name in names), sorted(names)[:8]


def test_report_without_prefetch_verification(results):
    """Diagnostic only: the static chains trusted (the verification off; teacher-forced like
    ``gtp``). Prints the precondition and the gate's result; whether it fails depends on buffer
    sharing and timing, which this model may not hit."""
    forcing = results["noverify"][1]["forcing"]
    print(f"[{_name('noverify')}] forcing {forcing}", flush=True)
    _, reference = results["ref"][0][1]
    _, grads, record = results["noverify"][0][1]
    report, failing = sgu.report_and_fail(_name("noverify step 2 vs ref"), grads, reference)
    print(f"[{_name('noverify')}] failing {len(failing)} of {len(report)}", flush=True)


def test_dropped_reduce_scatter_fails_the_gate(results):
    """The mutation: one of the 4 backward uses' reduce-scatter carries a zero gradient for
    every core GTP weight; the gate must catch (nearly) all of them."""
    _, reference = results["ref"][0][0]
    _, grads, _ = results["drop_rs"][0][0]
    _, info = results["drop_rs"]
    report, failing = sgu.report_and_fail(_name("drop_rs (mutation) vs ref"), grads, reference)
    core = info["core_names"]
    assert core
    caught = [name for name in core if not report[name]["ok"]]
    print(f"[{_name('drop_rs')}] caught {len(caught)} of {len(core)} core GTP weights", flush=True)
    assert len(caught) >= MIN_FAILING_CORE_FRACTION * len(core), sgu.mbs.worst(
        report, [name for name in core if report[name]["ok"]]
    )


def test_plain_reference_is_off_by_more_than_the_gate(results):
    """Why ``ref`` takes the precise embedding reference, measured on this model: the plain
    deterministic lookup (autograd sums the main and MTP uses' BF16 gradients in BF16 before
    the FP32 ``main_grad``) puts the embedding's step-1 gradient outside the gate against
    ``ref``, while every other gradient passes it (only the lookup's backward differs)."""
    _, precise = results["ref"][0][0]
    _, plain = results["ref_plain"][0][0]
    report, failing = sgu.report_and_fail(_name("ref_plain (diagnostic) vs ref"), plain, precise)
    print(f"[{_name('ref_plain')}] embedding {report[EMBEDDING]}", flush=True)
    assert failing == [EMBEDDING], sgu.mbs.worst(report, failing or [EMBEDDING])


def test_reference_sums_embedding_uses_in_fp32(results):
    """Every reference run took every embedding lookup through the precise per-use FP32
    reference: ``ROWS x (1 + MTP_LAYERS)`` per step (main and MTP input of every microbatch)."""
    expected = sgu.ROWS * (1 + sgu.MTP_LAYERS)
    for arm in ["ref", "ref_from_gtp"] + (["ref_ep"] if RUN_GTP_EP else []):
        assert results[arm][1]["lookups"] == [expected] * STEPS, (arm, results[arm][1]["lookups"])


@pytest.mark.parametrize("arm", ARMS)
def test_embedding_gradient_reduce_scatters_in_fp32(results, arm):
    """The embedding's BF16 lookup gradient is widened to ``main_grad``'s FP32 before every one
    of its reduce-scatters, in every step."""
    steps, info = results[arm]
    for step, (_, _, record) in enumerate(steps):
        got = record["dtypes"].get(info["emb_id"])
        assert got == {torch.float32}, (arm, step, got)


def _embedding_entry(arm, results):
    """The gate's entry for the embedding of mutation ``arm`` (step 1) against ``ref``; prints the
    embedding's local use norms."""
    _, reference = results["ref"][0][0]
    _, grads, record = results[arm][0][0]
    norms = record["norms"].get(results[arm][1]["emb_id"])
    print(f"[{_name(arm)}] embedding use norms (local, backward order) {norms}", flush=True)
    report, _ = sgu.report_and_fail(
        _name(f"{arm} (mutation) embedding vs ref"),
        {EMBEDDING: grads[EMBEDDING]},
        {EMBEDDING: reference[EMBEDDING]},
    )
    return report[EMBEDDING]


def test_dropped_embedding_use_fails_the_gate(results):
    """The mutation ``drop_rs`` also zeroes the embedding's second backward use before its
    reduce-scatter (the printed use norms show which): the embedding must fail the gate."""
    entry = _embedding_entry("drop_rs", results)
    assert not entry["ok"], entry


def test_bf16_embedding_reduce_scatter_fails_the_gate(results):
    """The mutation ``emb_bf16``: the embedding's gradient reduce-scattered in BF16 (the path
    before the FP32 fix). It must reach the embedding (BF16 recorded, a gradient that differs
    from ``gtp``'s) and fail the gate against the precise reference."""
    steps, info = results["emb_bf16"]
    _, mutant, record = steps[0]
    got = record["dtypes"].get(info["emb_id"])
    assert got == {torch.bfloat16}, got
    _, fixed, _ = results["gtp"][0][0]
    reached = sgu.mismatches({EMBEDDING: mutant[EMBEDDING]}, {EMBEDDING: fixed[EMBEDDING]})
    assert reached, "the BF16 mutation did not change the embedding gradient"
    entry = _embedding_entry("emb_bf16", results)
    assert not entry["ok"], entry
