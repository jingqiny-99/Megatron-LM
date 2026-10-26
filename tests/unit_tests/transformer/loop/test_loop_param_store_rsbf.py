# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The loop parameter store under the RSBF executor (``rsbf_executor`` + ``param_store``).

One EU (in-process gloo world of one rank, CPU or CUDA): a trivial store (``z = 1``,
``allow_trivial_shard``: gathers copy, reduces add) on a local-layer hybrid loop model against the
same executor without a store. The executor then runs every item's backward per segment inside
the store's ``backward_segment`` and every sub-round inside ``store.subround``, and the model's
visit hooks (``LoopHybridStack._run_range`` / ``_adapt``, ``HybridModel.embed`` / ``head_loss``)
open the store's segment visits, so these tests check the wiring op for op:

* every residency mode (Resident, Resident + sharded grads, Layer window kept and re-gathered, a
  mixed set) on every item form (whole round, block stride 2 and 1, nested stride 2) and with two
  waves: per-sample losses, depths and every gradient **bitwise** equal to the store-free step
  (one EU, z = 1: every gradient is added in the same order; the per-segment split of the
  backward is bitwise with the joint one), the store's reduce counts exactly once per (item,
  unit) and once per step for Resident units, its sub-round counts equal to the plan's;
* dense, MTP and mHC + MTP models; the windowed loop adapter and MTP units (move E);
* eager evaluation (no grad) through the model's own forward after a step equals the RSBF
  forward's per-sample losses (the eager visits gather the windowed units); a grad-enabled eager
  forward raises in every mode;
* the executor's store checks (modes configured, the same core plan); a step that raises
  mid-wave leaves the store clean (``abort_step``) and the next step is unchanged.

Multi-EU (spawned Gloo worlds, ``z = W``): the store at W = 2 and 4 EUs against the same W-EU
executor without a store (prescribed depths with sample moves and null items, every item form,
waves): per-sample losses bitwise; Layer window (kept, re-gathered) == Resident + sharded grads
gradients bitwise; every mode within the stage-2 gradient rule of the store-free run; the literal
case (one sample per EU, depth 1, ``no_overwrite``): every mode == Resident bitwise; on a stub
lockstep 'E' model (every MoE call an all-gather over the domain) the combined host log of the
MoE exchanges and the store's collectives is identical on every rank (nulls walk the store's
schedule around their MoE calls) and every exchange agrees.

The DSV4-TINY-like GPU gates (TE specs, EP = W) are in ``test_loop_param_store_parity.py``.
"""

import re
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.loop.param_store import (
    TRAIN_OUTSIDE_RSBF,
    LoopParamStore,
    LoopParamStoreConfig,
    ResidencyMode,
    StoreModes,
)
from megatron.core.transformer.loop.program import LoopProgram, PrescribedDepthCriterion
from megatron.core.transformer.loop.rsbf_executor import LoopMicrobatch, LoopRSBFExecutor
from tests.unit_tests.transformer.loop import hybrid_loop_utils
from tests.unit_tests.transformer.loop import param_store_utils as psu
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    SEQ,
    VOCAB,
    build_local_loop_model,
    device,
    model_parallel_once,
)
from tests.unit_tests.transformer.loop.multi_eu_utils import (
    SPAWN_SKIP_REASON,
    run_spawned,
    spawn_tests_enabled,
)

mhc_on_this_device = hybrid_loop_utils.mhc_on_this_device
cpu_mtp = hybrid_loop_utils.cpu_mtp

DENSE = "*-*-*-*-"  # prelude '*-', core '*-*-', coda '*-'
R_MAX = 4
RSBF = dict(loop_schedule="rsbf", calculate_per_token_loss=True, loop_iterations=R_MAX)
MHC = dict(enable_mhc_connections=True, mhc_num_residual_streams=4, mhc_sinkhorn_iterations=5)
DEPTHS = (1, 3, 2, 4)

MODES = {
    "resident": "all=r",
    "rsg": "all=rsg",
    "lw": "core=lw,edges=lw,emb_head=rsg",
    "lw_regather": "core=lw,edges=lw,emb_head=rsg,core_regather=on,edge_regather=on",
    "mixed": "core=lw,prelude=rsg,coda=lw,mtp=rsg,emb_head=r,core_regather=on",
}
FORMS = {
    "whole_round": {},
    "block2": dict(loop_rsbf_item_form="block", loop_rsbf_block_stride=2),
    "block1": dict(loop_rsbf_item_form="block", loop_rsbf_block_stride=1),
    "nested2": dict(loop_rsbf_item_form="nested", loop_rsbf_block_stride=2),
}


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


# ---------------------------------------------------------------------------------------------
# Helpers (also used by the spawned workers)
# ---------------------------------------------------------------------------------------------


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


def _model(name, request=None, **overrides):
    """``dense`` / ``mtp`` / ``mhc_mtp`` local loop models (an RSBF config); ``request`` turns
    on the CPU fixtures in-process (spawned workers patch the environment themselves)."""
    pattern = DENSE + ("/*-" if "mtp" in name else "")
    if request is not None and "mtp" in name:
        request.getfixturevalue("cpu_mtp")
    if "mhc" in name:
        if request is not None:
            request.getfixturevalue("mhc_on_this_device")
        overrides.update(MHC)
    return build_local_loop_model(pattern, **RSBF, **overrides)


def _rows(total, *, seed=5, dev=None):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, VOCAB, (total, SEQ), generator=generator)
    labels = torch.randint(0, VOCAB, (total, SEQ), generator=generator)
    loss_mask = (torch.rand(total, SEQ, generator=generator) > 0.2).float()
    dev = dev or device()
    return [
        dict(
            tokens=tokens[i : i + 1].to(dev),
            labels=labels[i : i + 1].to(dev),
            position_ids=torch.arange(SEQ)[None].to(dev),
            loss_mask=loss_mask[i : i + 1].to(dev),
        )
        for i in range(total)
    ]


def _microbatches(rows, *, world=1, rank=0, loss_func=_loss_func):
    """Rank ``rank``'s one-row microbatches (global microbatch ``j W + q``)."""
    return [
        LoopMicrobatch(
            tokens=row["tokens"],
            position_ids=row["position_ids"],
            labels=row["labels"],
            loss_mask=row["loss_mask"],
            attention_mask=None,
            padding_mask=None,
            loss_func=loss_func,
        )
        for k, row in enumerate(rows)
        if k % world == rank
    ]


def _install(model, groups, z, modes_text, **config):
    """A store with ``min_numel = 1`` (every 2-D dense weight stored, the adapter included),
    zeroed fp32 shard gradients, modes configured for the model's own core plan."""
    config.setdefault("min_numel", 1)
    if z == 1:
        config.setdefault("allow_trivial_shard", True)
    store = LoopParamStore.install(model, LoopParamStoreConfig(shard_size=z, **config), groups)
    psu.attach_shard_grads(store)
    if modes_text is not None:
        store.configure_modes(StoreModes.parse(modes_text))
    return store


def _executor(model, depths, *, domain=None, status=None, **kwargs):
    from megatron.core.transformer.loop.dispatcher import RSBFGroups

    program = LoopProgram(
        model, exit_criterion=PrescribedDepthCriterion(dict(enumerate(depths))), check_ports=True
    )
    groups = None if domain is None else RSBFGroups(domain=domain, status=status)
    return LoopRSBFExecutor(
        program, pg_collection=SimpleNamespace(dp=domain), groups=groups, **kwargs
    )


def _step(model, rows, depths, *, world=1, rank=0, executor=None, loss_func=_loss_func, **kwargs):
    """One RSBF step of this rank's microbatches; returns ({global index: lm loss}, executor)."""
    executor = executor or _executor(model, depths, **kwargs)
    model.zero_grad(set_to_none=True)
    microbatches = _microbatches(rows, world=world, rank=rank, loss_func=loss_func)
    reports = executor.run_step(
        lambda iterator, _model: next(iterator), iter(microbatches), len(microbatches), model=model
    )
    keys = [record.key.global_index for record in executor.stats.loss_records]
    return {key: report["lm loss"] for key, report in zip(keys, reports)}, executor


def _local_grads(model, store=None):
    """This rank's gradients by logical name: ``.grad`` of the registered parameters (no store:
    every parameter), and with a (trivial) store each store tensor's shard ``main_grad``."""
    grads = {}
    for name, param in model.named_parameters():
        if "loop_param_store." in name:
            continue
        grads[name] = None if param.grad is None else param.grad.detach().clone()
    if store is not None:
        for tensor in store.tensors():
            assert tensor.compute.grad is None, tensor.name  # the hook dropped every .grad
            grads[tensor.name] = tensor.shard.main_grad.detach()[: tensor.rows].clone()
    return grads


def _expected_reduces(store, samples, depth_sum, mtp):
    """Per non-Resident unit, its reduces per sub-round type at one EU (one per item visiting
    it); Resident units: none per item."""
    expected = {}
    for unit in store.units:
        if store.mode_of(unit) is ResidencyMode.RESIDENT:
            continue
        if unit.group == "core":
            counts = {"core_b": depth_sum}
        elif unit.group == "prelude":
            counts = {"prelude_b": samples}
        elif unit.key == "emb_head:embedding":
            counts = {"prelude_b": samples, **({"coda": samples} if mtp else {})}
        else:  # coda, mtp, output
            counts = {"coda": samples}
        expected[unit.key] = counts
    return expected


def _exactly_once(store, mtp):
    """Every EU (real item or null) reduces each non-Resident unit once per sub-round of every
    type whose backward visits it, and each Resident unit once per step (``end_step``)."""
    subrounds = {}
    for (kind, _), count in store.stats.subrounds.items():
        subrounds[kind] = subrounds.get(kind, 0) + count
    got = _reduce_counts(store)
    for unit in store.units:
        if store.mode_of(unit) is ResidencyMode.RESIDENT:
            if unit.key in got or store.stats.resident_reduces[unit.key] != 1:
                return False
            continue
        kinds = _expected_reduces(store, 1, 1, mtp)[unit.key]
        if got.get(unit.key) != {kind: subrounds.get(kind, 0) for kind in kinds}:
            return False
    return True


def _reduce_counts(store):
    got = {}
    for (kind, key), count in store.stats.reduces.items():
        got.setdefault(key, {})[kind] = count
    return got


def _trivial_groups():
    return psu.trivial_store_groups()


# ---------------------------------------------------------------------------------------------
# One EU: the store-wired executor equals the store-free one bitwise
# ---------------------------------------------------------------------------------------------


def _assert_same(actual, expected):
    assert actual.keys() == expected.keys(), actual.keys() ^ expected.keys()
    for name, value in expected.items():
        got = actual[name]
        if value is None:
            assert got is None or not got.any(), name
            continue
        assert got is not None and torch.equal(got, value), name


def _store_vs_storeless(request, name, form, modes, *, waves=1, depths=DEPTHS):
    rows = _rows(len(depths))
    reference = _model(name, request, **FORMS[form])
    ref_losses, ref_executor = _step(reference, rows, depths, waves=waves)
    expected = _local_grads(reference)
    model = _model(name, request, **FORMS[form])
    store = _install(model, _trivial_groups(), 1, MODES[modes], poison_released=True)
    losses, executor = _step(model, rows, depths, waves=waves)
    assert executor.param_store is store and executor.program.param_store is store
    _assert_same(losses, ref_losses)
    _assert_same(_local_grads(model, store), expected)
    assert executor.stats.depths == ref_executor.stats.depths
    # Exactly once: one reduce per (item, unit), one Resident reduce per unit and step.
    assert _reduce_counts(store) == _expected_reduces(
        store, len(depths), sum(depths), "mtp" in name
    )
    for unit in store.units:
        resident = store.mode_of(unit) is ResidencyMode.RESIDENT
        assert store.stats.resident_reduces[unit.key] == (1 if resident else 0), unit.key
    # Every sub-round ran inside the store (one EU: all real).
    plan_subrounds = executor.stats.subrounds
    store_subrounds = {kind: n for (kind, real), n in store.stats.subrounds.items()}
    assert store_subrounds == {kind: n for kind, n in plan_subrounds.items() if n}
    assert not store.in_step
    # The segment-split backward left no window unit bound.
    for unit in store.units:
        if store.mode_of(unit) is ResidencyMode.LAYER_WINDOW:
            assert not unit.bound and all(t.bound is None for t in unit.tensors), unit.key
    return store, executor


@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("modes", list(MODES))
def test_every_mode_and_item_form_equals_the_storeless_step(request, modes, form):
    store, _ = _store_vs_storeless(request, "mtp", form, modes)
    if modes == "lw":
        # The adapter and both MTP units are windowed (move E); the head is RSG.
        windowed = {u.key for u in store.units if store.mode_of(u) is ResidencyMode.LAYER_WINDOW}
        assert {"core:adapter", "mtp:0:eh_proj", "mtp:0:L"} <= windowed
        assert store.stats.gathers["core:adapter"] > 0


@pytest.mark.parametrize("modes", ["resident", "lw", "lw_regather"])
@pytest.mark.parametrize("name", ["dense", "mhc_mtp"])
def test_models_and_waves(request, name, modes):
    _store_vs_storeless(request, name, "block2", modes, waves=2)


def test_lookahead_off_and_debug_checks_change_nothing(request):
    """Without the cross-sub-round lookahead (no prefetch, no carry) and with every debug check
    (``check_visits``, ``verify_gathers``) the step is unchanged."""
    rows = _rows(len(DEPTHS))
    reference = _model("mtp", request)
    ref_losses, _ = _step(reference, rows, DEPTHS)
    expected = _local_grads(reference)
    for config in (dict(lookahead=False), dict(check_visits=True, verify_gathers=True)):
        model = _model("mtp", request)
        store = _install(model, _trivial_groups(), 1, MODES["lw"], **config)
        losses, _ = _step(model, rows, DEPTHS)
        _assert_same(losses, ref_losses)
        _assert_same(_local_grads(model, store), expected)
        if config.get("lookahead") is False:
            assert store.stats.prefetches == 0 and store.stats.carries == 0
        else:
            assert store.stats.prefetches > 0 and store.stats.carries > 0


# ---------------------------------------------------------------------------------------------
# Eager evaluation and the training guard
# ---------------------------------------------------------------------------------------------


def _eager_losses(model, rows, depths):
    """Per-sample losses of the model's own eager forward (no grad) at the given depths."""
    losses = {}
    with torch.no_grad():
        for index, (row, depth) in enumerate(zip(rows, depths)):
            model.decoder.set_loop_depths(torch.tensor([depth], device=row["tokens"].device))
            output = model(
                row["tokens"],
                row["position_ids"],
                None,
                labels=row["labels"],
                loss_mask=row["loss_mask"],
            )
            losses[index] = _loss_func(row["loss_mask"], output)[2]["lm loss"]
    return losses


@pytest.mark.parametrize("modes", ["resident", "lw", "mixed"])
def test_eager_evaluation_visits_equal_the_rsbf_forward(request, modes):
    """After a step (no optimizer: the weights are those of the step), ``refresh()`` and the
    model's eager forward under no_grad visit the embedding, the adapter, every sublayer and the
    head (MTP included) through the same hooks: per-sample losses equal the RSBF step's; the
    windowed units were gathered by eager visits."""
    rows = _rows(len(DEPTHS))
    model = _model("mtp", request)
    store = _install(model, _trivial_groups(), 1, MODES[modes])
    losses, _ = _step(model, rows, DEPTHS)
    with pytest.raises(RuntimeError, match="stale"):
        _eager_losses(model, rows, DEPTHS)
    store.refresh()
    _assert_same(_eager_losses(model, rows, DEPTHS), losses)
    if store.windows:
        assert store.stats.eager_gathers > 0
    # Eager visits leave nothing bound.
    for unit in store.units:
        if store.mode_of(unit) is ResidencyMode.LAYER_WINDOW:
            assert all(t.bound is None for t in unit.tensors), unit.key


@pytest.mark.parametrize("modes", list(MODES))
def test_a_grad_enabled_eager_forward_raises_in_every_mode(request, modes):
    model = _model("dense", request)
    store = _install(model, _trivial_groups(), 1, MODES[modes])
    store.refresh()
    row = _rows(1)[0]
    with pytest.raises(RuntimeError, match=re.escape(TRAIN_OUTSIDE_RSBF)):
        model(row["tokens"], row["position_ids"], None, labels=row["labels"])


def test_the_stage_hooks_are_inert_without_a_store(request):
    """Without a store the model and stack carry no hook (the stage-5 path)."""
    model = _model("mtp", request)
    assert getattr(model, "_loop_store_hook", None) is None
    assert model.decoder._loop_store_hook is None  # pylint: disable=protected-access
    assert LoopProgram(model).param_store is None
    store = _install(model, _trivial_groups(), 1, MODES["lw"])
    assert model._loop_store_hook is store  # pylint: disable=protected-access
    assert model.decoder._loop_store_hook is store  # pylint: disable=protected-access
    # The core overlap check still sees the hidden compute parameters.
    assert LoopProgram(model).core_parameter_overlap() == []


# ---------------------------------------------------------------------------------------------
# Executor checks and failure hygiene
# ---------------------------------------------------------------------------------------------


def test_the_executor_needs_configured_modes_for_its_core_plan(request):
    rows = _rows(2)
    model = _model("dense", request, **FORMS["block2"])
    store = _install(model, _trivial_groups(), 1, None)
    with pytest.raises(ValueError, match="configure the parameter store's modes"):
        _step(model, rows, (1, 2))
    store.configure_modes(StoreModes.parse(MODES["lw"]), core_plan=psu.core_plan_of("nested", 4))
    with pytest.raises(ValueError, match="configured for the core plan"):
        _step(model, rows, (1, 2))
    store.configure_modes(StoreModes.parse(MODES["lw"]))
    _step(model, rows, (1, 2))
    # The DDP rules of the guard are applied with the model's DDP config.
    model.ddp_config = SimpleNamespace(grad_reduce_in_fp32=False)
    with pytest.raises(ValueError, match="grad_reduce_in_fp32=False"):
        _step(model, rows, (1, 2))


def test_a_raising_step_leaves_the_store_clean(request):
    """A loss function raising in the second CODA item: the step raises (``abort_step``), the
    store is out of its step with nothing bound, and the next step equals a fresh store-free
    step."""
    rows = _rows(len(DEPTHS))
    reference = _model("mtp", request)
    ref_losses, _ = _step(reference, rows, DEPTHS)
    expected = _local_grads(reference)
    model = _model("mtp", request)
    store = _install(model, _trivial_groups(), 1, MODES["lw"], poison_released=True)
    calls = []

    def failing(loss_mask, output):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("loss function failed")
        return _loss_func(loss_mask, output)

    executor = _executor(model, DEPTHS)
    with pytest.raises(RuntimeError, match="loss function failed"):
        _step(model, rows, DEPTHS, executor=executor, loss_func=failing)
    assert not store.in_step
    for unit in store.units:
        assert unit.slot is None and unit.grad_flat is None, unit.key
        for tensor in unit.tensors:
            assert tensor.grad_target is None or store.mode_of(unit) is ResidencyMode.RESIDENT
            if store.mode_of(unit) is ResidencyMode.LAYER_WINDOW:
                assert tensor.bound is None, tensor.name
    psu.attach_shard_grads(store)
    losses, _ = _step(model, rows, DEPTHS, executor=executor)
    _assert_same(losses, ref_losses)
    _assert_same(_local_grads(model, store), expected)


# ---------------------------------------------------------------------------------------------
# Multi-EU (spawned Gloo worlds, z = W)
# ---------------------------------------------------------------------------------------------

spawned = pytest.mark.skipif(
    not spawn_tests_enabled() or torch.cuda.is_available(),
    reason=SPAWN_SKIP_REASON + "; CPU hosts only",
)


def _skew_depths(total, world):
    """EU 0's samples run R_max rounds, the others 1 or 2 (sample moves, idle EUs, nulls)."""
    depths = {}
    for g in range(total):
        rank, j = g % world, g // world
        depths[g] = R_MAX if rank == 0 else 1 + j % 2
    return [depths[g] for g in range(total)]


def _domain_grads(model, store, groups):
    """Full gradients by logical name: replicated ``.grad`` summed over the world (finalize's
    shard-axis SUM plus DDP's replicate all-reduce) and the store tensors' full gradients (the
    shard ``main_grad`` summed over the replicate group, all-gathered over the shard axis); no
    store: every ``.grad`` summed over the world."""
    import torch.distributed as dist

    out = {}
    for name, param in sorted(model.named_parameters(), key=lambda item: item[0]):
        if "loop_param_store." in name:
            continue
        grad = torch.zeros_like(param) if param.grad is None else param.grad.detach().clone()
        dist.all_reduce(grad)
        out[name] = grad
    if store is not None:
        out.update(psu.full_store_grads(store, groups))
    return out


def _executor_modes_worker(rank, world, spec):
    import torch.distributed as dist

    from megatron.core import parallel_state
    from tests.unit_tests.transformer.loop.multi_eu_utils import (
        cpu_worker_setup,
        cpu_worker_teardown,
    )

    cpu_worker_setup()
    try:
        domain = parallel_state.get_data_parallel_group()
        status = parallel_state.get_data_parallel_group_gloo()
        groups = psu.make_store_groups(world, world)
        per_rank = spec["per_rank"]
        total = per_rank * world
        depths = spec.get("depths") or _skew_depths(total, world)
        rows = _rows(total)
        overrides = dict(FORMS[spec["form"]], ffn_hidden_size=91)
        options = dict(domain=domain, status=status, check_plan=True, waves=spec.get("waves", 1))

        def run(modes_text):
            model = _model(spec["model"], **overrides)
            store = None
            if modes_text is not None:
                store = _install(
                    model,
                    groups,
                    world,
                    modes_text,
                    poison_released=True,
                    verify_gathers=True,
                    record_calls=True,
                    **spec.get("store", {}),
                )
            losses, executor = _step(model, rows, depths, world=world, rank=rank, **options)
            return losses, _domain_grads(model, store, groups), store, executor

        ref_losses, ref_grads, _, ref_executor = run(None)
        runs = {name: run(MODES[name]) for name in spec["modes"]}
        checks = {}
        for name, (losses, grads, store, executor) in runs.items():
            checks[f"{name}: losses == no store"] = losses.keys() == ref_losses.keys() and all(
                torch.equal(losses[k], ref_losses[k]) for k in ref_losses
            )
            checks[f"{name}: grads ~ no store"] = (
                max(psu.compare(grads[n], ref_grads[n]) for n in ref_grads) <= 1.0
            )
            checks[f"{name}: pad rows zero"] = all(
                bool((pad == 0).all()) for pad in psu.shard_pad_rows(store).values()
            )
            checks[f"{name}: same moves and nulls"] = (
                executor.stats.moves == ref_executor.stats.moves
                and executor.stats.nulls == ref_executor.stats.nulls
            )
            logs = [None] * world
            dist.all_gather_object(logs, [(op, key) for op, key, _ in store.comm.log])
            checks[f"{name}: identical store call log on every rank"] = all(
                log == logs[0] for log in logs
            )
            # Every EU runs every sub-round of the plan (an item or a null) inside the store.
            per_type = {}
            for (kind, _), count in store.stats.subrounds.items():
                per_type[kind] = per_type.get(kind, 0) + count
            checks[f"{name}: every sub-round inside the store"] = per_type == {
                kind: n for kind, n in executor.stats.subrounds.items() if n
            }
            checks[f"{name}: reduces exactly once"] = _exactly_once(store, "mtp" in spec["model"])
        for left, right in spec.get("bitwise", ()):
            a, b = runs[left][1], runs[right][1]
            checks[f"{left} == {right} grads"] = all(torch.equal(a[n], b[n]) for n in a)
        store_nulls = sum(
            n
            for (_, real), n in runs[spec["modes"][0]][2].stats.subrounds.items()
            if real == "null"
        )
        return dict(
            checks=checks,
            nulls=sum(ref_executor.stats.nulls.values()),
            store_nulls=store_nulls,
            moves=dict(ref_executor.stats.moves),
        )
    finally:
        cpu_worker_teardown()


def _assert_checks(results):
    failed = [
        f"rank {rank}: {check}"
        for rank, result in enumerate(results)
        for check, ok in result["checks"].items()
        if not ok
    ]
    assert not failed, "\n".join(failed)


@spawned
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("form", ["whole_round", "block2", "nested2"])
def test_store_at_w_eus_equals_the_storeless_executor(world, form):
    spec = dict(
        model="mtp",
        form=form,
        per_rank=2,
        modes=["resident", "rsg", "lw", "lw_regather"],
        bitwise=[("lw", "rsg"), ("lw_regather", "rsg")],
    )
    results = run_spawned(_executor_modes_worker, world, spec, timeout=900)
    _assert_checks(results)
    assert results[0]["nulls"] > 0 and results[0]["moves"]["sample"] > 0
    # Some EU walked the store's schedule in a null sub-round.
    assert sum(result["store_nulls"] for result in results) > 0


@spawned
def test_store_with_waves_at_two_eus():
    spec = dict(
        model="dense", form="block2", per_rank=2, waves=2, modes=["resident", "lw"], bitwise=[]
    )
    _assert_checks(run_spawned(_executor_modes_worker, 2, spec, timeout=900))


@spawned
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("form", ["whole_round", "block2", "nested2"])
def test_literal_case_every_mode_equals_resident_bitwise(world, form):
    """z = W, one sample per EU, depth 1, ``no_overwrite``, no MTP: every unit gets one item
    per step on every EU, so Layer window and Resident + sharded grads reduce the same fp32
    values once, as Resident does at the end of the step (D-S6-9)."""
    spec = dict(
        model="dense",
        form=form,
        per_rank=1,
        depths=[1] * world,
        store=dict(no_overwrite=True),
        modes=["resident", "rsg", "lw", "lw_regather"],
        bitwise=[("lw", "resident"), ("lw_regather", "resident"), ("rsg", "resident")],
    )
    _assert_checks(run_spawned(_executor_modes_worker, world, spec, timeout=900))


STUB_BLOCKS = "*E*E*E*E/*E"  # prelude '*E', core '*E*E', coda '*E', MTP '*E'
STUB_MOE = dict(num_moe_experts=4, moe_ffn_hidden_size=128, moe_router_topk=2)


def _lockstep_store_worker(rank, world, spec):
    import torch.distributed as dist

    from megatron.core import parallel_state
    from megatron.core.transformer.loop.null_items import register_lockstep_module
    from megatron.core.transformer.loop.param_store_comm import StoreComm
    from tests.unit_tests.transformer.loop.multi_eu_utils import (
        RecordingLockstepMLP,
        StubLockstepAdapter,
        StubLog,
        cpu_worker_setup,
        cpu_worker_teardown,
        stub_moe_layer_spec,
    )

    cpu_worker_setup()
    unregister = register_lockstep_module(RecordingLockstepMLP, StubLockstepAdapter())
    original_record = StoreComm._record  # pylint: disable=protected-access

    def record(comm, op, unit):
        StubLog.entries.append(("store", op, unit.key))
        return original_record(comm, op, unit)

    StoreComm._record = record  # pylint: disable=protected-access
    try:
        domain = parallel_state.get_data_parallel_group()
        status = parallel_state.get_data_parallel_group_gloo()
        ep = dist.new_group(list(range(world)), backend="gloo")
        groups = psu.make_store_groups(world, world)
        total = 2 * world
        depths = _skew_depths(total, world)
        rows = _rows(total)
        overrides = dict(FORMS[spec["form"]], **STUB_MOE, **RSBF)

        def run(modes_text):
            RecordingLockstepMLP.instances = []
            model = build_local_loop_model(
                STUB_BLOCKS, moe_layer=stub_moe_layer_spec(), **overrides
            )
            stubs = list(RecordingLockstepMLP.instances)
            for stub in stubs:
                stub.group = ep
            store = None
            if modes_text is not None:
                store = _install(model, groups, world, modes_text, poison_released=True)
            StubLog.reset()
            losses, executor = _step(
                model,
                rows,
                depths,
                world=world,
                rank=rank,
                domain=domain,
                status=status,
                check_lockstep=True,
                check_plan=True,
            )
            log = list(StubLog.entries)
            return losses, _domain_grads(model, store, groups), stubs, executor, log

        ref_losses, ref_grads, _, ref_executor, _ = run(None)
        losses, grads, stubs, executor, log = run(MODES[spec["modes"]])
        logs = [None] * world
        dist.all_gather_object(logs, log)
        store_ops = sum(1 for entry in log if entry[0] == "store")
        return dict(
            losses=losses.keys() == ref_losses.keys()
            and all(torch.equal(losses[k], ref_losses[k]) for k in ref_losses),
            grads=max(psu.compare(grads[n], ref_grads[n]) for n in ref_grads) <= 1.0,
            mismatches=[stub.mismatches for stub in stubs],
            same_log=all(other == logs[0] for other in logs),
            store_ops=store_ops,
            null_moe_calls=executor.stats.null_moe_calls,
            nulls=dict(executor.stats.nulls),
            ref_nulls=dict(ref_executor.stats.nulls),
        )
    finally:
        StoreComm._record = original_record  # pylint: disable=protected-access
        unregister()
        cpu_worker_teardown()


@spawned
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("form", ["whole_round", "block2", "nested2"])
def test_nulls_interleave_the_store_schedule_with_lockstep_moe(world, form):
    """The stub 'E' model (every MoE forward and backward an all-gather of a tag over the
    domain) with the core, edges and MTP windowed: every rank, real or null, issues the same
    sequence of MoE exchanges and store collectives (one combined host log), every exchange
    agrees, nulls ran (with MoE calls), and losses / gradients match the store-free run."""
    spec = dict(form=form, modes="lw")
    results = run_spawned(_lockstep_store_worker, world, spec, timeout=900)
    for rank, result in enumerate(results):
        assert result["losses"] and result["grads"], rank
        assert all(m == 0 for m in result["mismatches"]), (rank, result["mismatches"])
        assert result["same_log"] and result["store_ops"] > 0, rank
        assert result["nulls"] == result["ref_nulls"], rank
    assert sum(r["null_moe_calls"] for r in results) > 0
    assert any(sum(r["nulls"].values()) > 0 for r in results)
