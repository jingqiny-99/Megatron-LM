# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Module GTP weight remat on a fixed-depth DSv4 loop (GPU; the lifted guard
``GTP_WEIGHT_REMAT``).

Two training steps without an optimizer step (the same weights; the second step is the first
with DDP's overlapped reduction armed) of the model of ``strategy_grad_utils`` (a core of 8
sublayers run exactly 4 times, block + edge ``loop_recompute`` with stride 4, three microbatches
of one row per rank, per-token loss, ``finalize_model_grads``), from the same full weights and
rows in every arm:

* ``ref``: no GTP (``gtp_remat_size = 1``), DDP over the whole data-parallel group;
* ``gtp``: dense and expert weight remat of degree ``G`` (``LOOP_GTP_DEGREE``, default the world
  size: 2 or 4), DDP with the distributed optimizer, prefetch and asynchronous reduce-scatter
  on (the defaults), ``GTP_CONFIG.verify_prefetched_gathers`` on (set by the loop stack);
* ``gtp_overlap``: ``gtp`` with DDP's ``overlap_grad_reduce`` (a GTP shard becomes grad-ready
  once per use, so a core shard's recorded count is 4);
* ``gtp_simple``: ``gtp`` with prefetch and asynchronous reduce-scatter off (every use gathers
  and reduce-scatters synchronously: GTP's simplest path);
* ``noverify`` (diagnostic, not gated): ``gtp`` with the prefetch verification turned off after
  construction, i.e. the static chains trusted.

Every use of a GTP weight gathers it and reduce-scatters its weight gradient into the shard's
``main_grad`` (a weight's next reduce-scatter first finalizes, i.e. adds, the previous one); a
core weight is used 4 times per microbatch, in the forward and again in the block recompute.

* Every parameter's reduced fp32 gradient (GTP shards all-gathered over their group, padding
  stripped) of both steps matches ``ref``'s: cosine >= 0.9999 and relative difference <= 1e-3 (a
  gradient zero to fp32 precision: absolute, at most 1e-6 of the whole gradient's norm); the
  losses and every row's LM loss equal ``ref``'s.
* Reduce-scatters per GTP weight (a dense weight, or a grouped expert projection's leading
  weight) and microbatch: 4 for a core weight, 1 for a prelude or coda weight (one per use).
* The prefetch verification's decisions are printed: how many consumes found a prefetch handle
  whose buffer did not hold this consume's gather (they gathered on demand).
* Teeth, must fail: ``drop_rs``, the second backward use's weight gradient of every core GTP
  weight zeroed before its reduce-scatter (one round's reduce-scatter dropped): at least 90% of
  the core GTP weights fail the gate.

Needs TransformerEngine >= 2.19 (GTP). Launch once per degree: ``torchrun --nproc-per-node 2 -m
pytest -q tests/unit_tests/transformer/loop/test_loop_gtp_gpu.py`` and ``--nproc-per-node 4``
(``LOOP_GTP_DEGREE=2`` on 4 GPUs gives replicate DP 2 x GTP 2). ``MBS_REPORT_DIR`` dumps every
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


def _initialize(degree):
    """Model parallelism with (expert) GTP weight remat of ``degree`` (1: none), and a clean
    GTP runtime (chains, buffer cache, ledger, the global parameter list)."""
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
    Utils.initialize_model_parallel(1, 1, gtp_remat_size=degree, expert_gtp_remat_size=degree)
    model_parallel_cuda_manual_seed(123)


def _is_gtp(param):
    from megatron.core.tensor_parallel.gtp_api import is_gtp_param

    return is_gtp_param(param)


def _load_full_state(model, state):
    """Load full (unsharded) weights into a model whose GTP weights are shards: shard ``r`` of
    a weight holds rows ``[r S, (r + 1) S)`` of the weight padded to ``S x G`` rows."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            full = state[name].to(param.dtype)
            if _is_gtp(param):
                rows, size = param.shape[0], param.group.size()
                padded = torch.zeros(
                    (rows * size, *full.shape[1:]), dtype=full.dtype, device="cuda"
                )
                padded[: full.shape[0]].copy_(full)
                rank = param.group.rank()
                param.data.copy_(padded[rank * rows : (rank + 1) * rows])
            else:
                param.data.copy_(full)
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
        ),
        model,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )


class _Instrument:
    """Counts the reduce-scatters of every GTP weight per microbatch (``GTPShardedParam.
    wgrad_reduce_scatter``) and the prefetch verification's decisions (``_GatherLedger.take``);
    ``drop_use`` zeroes that backward use's (1-based, per microbatch) weight gradient of every
    weight in ``drop_ids`` before its reduce-scatter (the mutation)."""

    def __init__(self, drop_ids=(), drop_use=None):
        from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp

        self.gtp = gtp
        self.original_rs = gtp.GTPShardedParam.wgrad_reduce_scatter
        self.original_take = gtp._GATHER_LEDGER.take
        self.microbatch = -1
        self.rs = collections.Counter()  # (microbatch, id(weight)) -> reduce-scatters
        self.takes = collections.Counter()  # True / False
        self.drop_ids = set(drop_ids)
        self.drop_use = drop_use
        instrument = self

        def counting_rs(weight, wgrad, *args, **kwargs):
            key = (instrument.microbatch, id(weight))
            instrument.rs[key] += 1
            if id(weight) in instrument.drop_ids and instrument.rs[key] == instrument.drop_use:
                for tensor in wgrad if isinstance(wgrad, (list, tuple)) else [wgrad]:
                    tensor.zero_()
            return instrument.original_rs(weight, wgrad, *args, **kwargs)

        def counting_take(weight, slot, buffer):
            current = instrument.original_take(weight, slot, buffer)
            instrument.takes[current] += 1
            return current

        self.counting_rs = counting_rs
        self.counting_take = counting_take

    def on_microbatch(self, index):
        self.microbatch = index

    def __enter__(self):
        self.gtp.GTPShardedParam.wgrad_reduce_scatter = self.counting_rs
        self.gtp._GATHER_LEDGER.take = self.counting_take
        return self

    def __exit__(self, *exc):
        self.gtp.GTPShardedParam.wgrad_reduce_scatter = self.original_rs
        del self.gtp._GATHER_LEDGER.take  # back to the class method


def _run_reference(state):
    _initialize(1)
    model = sgu.build_model(state)
    ddp = _ddp(model, distributed_optimizer=False)
    out = []
    for step in range(STEPS):
        _restore_buffers(model, state)
        result = sgu.run_step(ddp, model, sgu.make_rows(step))
        out.append((result, sgu.ddp_grads(ddp, model)))
    del ddp, model
    torch.cuda.empty_cache()
    return out, None


def _run_gtp(state, *, overlap=False, simple=False, verify=True, drop_use=None):
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTP_CONFIG
    from megatron.core.tensor_parallel.gtp_api import (
        classify_gtp_remat_chains,
        configure_gtp_remat_from_recipe,
    )

    saved = (GTP_CONFIG.weight_prefetch, GTP_CONFIG.async_reduction)
    _initialize(DEGREE)
    configure_gtp_remat_from_recipe(calculate_per_token_loss=True)
    GTP_CONFIG.weight_prefetch = not simple
    GTP_CONFIG.async_reduction = not simple
    try:
        model = sgu.build_model(
            None, gtp_weight_remat_size=DEGREE, expert_gtp_weight_remat_size=DEGREE
        )
        assert GTP_CONFIG.verify_prefetched_gathers, "the loop stack turns the verification on"
        GTP_CONFIG.verify_prefetched_gathers = verify
        _load_full_state(model, state)
        ddp = _ddp(model, overlap=overlap)
        classify_gtp_remat_chains([ddp])
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
        instrument = _Instrument(drop_ids=core_ids & consumers, drop_use=drop_use)
        out = []
        with instrument:
            for step in range(STEPS):
                _restore_buffers(model, state)
                instrument.rs.clear()
                result = sgu.run_step(
                    ddp, model, sgu.make_rows(step), on_microbatch=instrument.on_microbatch
                )
                out.append(
                    (
                        result,
                        _full_grads(ddp, model),
                        dict(rs=dict(instrument.rs), takes=dict(instrument.takes)),
                    )
                )
                instrument.takes.clear()
        golden = {}
        if overlap:
            for group in ddp.bucket_groups + ddp.expert_parallel_bucket_groups:
                for param, count in group.golden_per_param_grad_ready_counts.items():
                    if id(param) in gtp_names:
                        golden[gtp_names[id(param)]] = count
        info = dict(
            gtp_names=gtp_names,
            consumers=consumers,
            core_names=sorted(gtp_names[i] for i in core_ids),
        )
        info["golden"] = golden
        del ddp, model
        torch.cuda.empty_cache()
        return out, info
    finally:
        GTP_CONFIG.weight_prefetch, GTP_CONFIG.async_reduction = saved
        GTP_CONFIG.verify_prefetched_gathers = False


@pytest.fixture(scope="module")
def results():
    """Every arm, run once from the same full weights and rows."""
    _initialize(1)
    state = sgu.clone_state(sgu.build_model())
    out = {}
    out["ref"] = _run_reference(state)
    out["gtp"] = _run_gtp(state)
    out["gtp_overlap"] = _run_gtp(state, overlap=True)
    out["gtp_simple"] = _run_gtp(state, simple=True)
    out["noverify"] = _run_gtp(state, verify=False)
    out["drop_rs"] = _run_gtp(state, drop_use=2)
    return out


def _name(text):
    return sgu.tag(f"GTP G{DEGREE}", text)


@pytest.mark.parametrize("arm", ["gtp", "gtp_overlap", "gtp_simple"])
@pytest.mark.parametrize("step", range(STEPS))
def test_gradients_match_the_reference(results, arm, step):
    _, reference = results["ref"][0][step]
    _, grads, record = results[arm][0][step]
    print(f"[{_name(f'{arm} step {step + 1}')}] verification takes {record['takes']}", flush=True)
    report, failing = sgu.report_and_fail(_name(f"{arm} step {step + 1} vs ref"), grads, reference)
    assert not failing, sgu.mbs.worst(report, failing)


@pytest.mark.parametrize("arm", ["gtp", "gtp_overlap", "gtp_simple"])
def test_losses_match_the_reference(results, arm):
    for step in range(STEPS):
        reference, _ = results["ref"][0][step]
        result, _, _ = results[arm][0][step]
        sgu.assert_losses_match(result, reference, _name(f"{arm} step {step + 1}"))


@pytest.mark.parametrize("arm", ["gtp", "gtp_overlap", "gtp_simple"])
def test_one_reduce_scatter_per_use(results, arm):
    """Every core GTP weight is reduce-scattered 4 times per microbatch, every prelude and coda
    GTP weight once."""
    steps, info = results[arm]
    names = {pid: name for pid, name in info["gtp_names"].items() if pid in info["consumers"]}
    for step, (_, _, record) in enumerate(steps):
        for microbatch in range(sgu.ROWS):
            for pid, name in names.items():
                where = sgu.role(name)[0]
                if where not in ("core", "prelude", "coda"):
                    continue
                expected = sgu.ROUNDS if where == "core" else 1
                got = record["rs"].get((microbatch, pid), 0)
                assert got == expected, (arm, step, microbatch, name, got)


def test_overlap_counts_one_grad_ready_per_use(results):
    """Under overlap_grad_reduce a GTP shard is grad-ready once per use: the recorded count of
    every core GTP weight is 4, of every prelude and coda GTP weight 1."""
    _, info = results["gtp_overlap"]
    golden = info["golden"]
    print(f"[{_name('gtp_overlap golden counts')}] {golden}", flush=True)
    assert golden
    for name, count in golden.items():
        where = sgu.role(name)[0]
        if where == "core":
            assert count == sgu.ROUNDS, (name, count)
        elif where in ("prelude", "coda"):
            assert count == 1, (name, count)


def test_report_without_prefetch_verification(results):
    """Diagnostic only: the static chains trusted (the verification off). Prints the gate's
    result; whether it fails depends on buffer sharing and timing, which this model may not
    hit."""
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
