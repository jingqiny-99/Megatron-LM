# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoints of the loop parameter store on spawned CPU Gloo worlds (M3c,
``loop/param_store_checkpoint.py``).

A local-layer dense Huginn loop model (``*-*-*-*-``, prelude 2, core 4, coda 2) with a
vocabulary of 130 rows and fc1 of 182 rows (rows not divisible by z = 4), plus three raw
parameters on sublayer 2 whose rows leave whole shards empty: ``[3, h]`` (one all-padding rank at
z = 4), ``[1, h]`` (three at z = 4, one at z = 2) and ``[9, h]`` (three at z = 8). Every 2-D
parameter is stored (``min_numel = 1``). Shard groups are consecutive ranks, replicate groups
strided (``param_store_utils.make_store_groups``). Cases:

Before every load the load side's shard pad rows (and the Adam state's) are poisoned with 7.0,
so the zero-pad checks fail if the load does not zero them itself.

* model-only torch_dist save / load across {no store, z = 2, z = 4} in every direction (W = 4;
  "no store" is the regular z = 1 layout), with the Resident, Resident + sharded grads and Layer
  window modes on the load side: every logical weight bitwise (shards gathered over the shard
  axis), every shard pad row exactly 0; after the load the compute copies are unbound (NaN
  placeholder), a Resident eager visit raises until ``refresh()``, then the bound weights equal the
  loaded ones (Layer window: an eager visit gathers them);
* the load hooks at z = 4: the merge writes in place (the loaded entry is the canonical shard
  itself, no allocation; all-padding ranks get zeros); a legacy full-tensor ``state_dict`` of a
  store-free model loads through the pre-hook's slicing (the ``[1, h]`` row lands on shard rank 0
  only); a mirror-keyed ``state_dict()`` round trip at the same z; a padded-vocab shape change
  raises its message; a missing logical entry raises in the post-hook also with
  ``strict=False``; a load or a sharded state dict inside a training step raises;
* ``torch.optim.Adam`` state through ``optim_state_to_sharding_state`` (the generic optimizer
  path) saved at z = 2 and loaded at z = 4 and without a store: ``exp_avg`` / ``exp_avg_sq``
  bitwise against the logical state, pad rows 0;
* a world-size (data-parallel) change: model and Adam state saved at W = 4 (z = 2), loaded at
  W = 8 with z = 4 (replicate 2), z = 8 and without a store;
* W = 8, z = 8 with three all-padding ranks of ``[9, h]``: plain and fully parallel save
  (``FullyParallelSaveStrategyWrapper``), loaded at z = 8 and z = 2 (the fully parallel *load*
  exchanges on CUDA: its arm is in ``test_loop_checkpoint_store_gpu.py``).

The distributed optimizer (CUDA, E20 of the stage-6 design) is covered in
``test_loop_checkpoint_store_gpu.py``.
"""

import os
import tempfile

import pytest
import torch
from torch import nn

from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from tests.unit_tests.transformer.loop.multi_eu_utils import (
    SPAWN_SKIP_REASON,
    run_spawned,
    spawn_tests_enabled,
)

pytestmark = pytest.mark.skipif(not spawn_tests_enabled(), reason=SPAWN_SKIP_REASON)

PATTERN = "*-*-*-*-"
CORE = 4
VOCAB_ROWS = 130
MODEL = dict(ffn_hidden_size=91, vocab_size=VOCAB_ROWS)
MODES = ("r", "rsg", "lw")
MODE_TEXT = {"r": "all=r", "rsg": "all=rsg", "lw": "core=lw,edges=lw,emb_head=rsg"}


class CkptStack(LoopHybridStack):
    """The loop stack with three short raw parameters on sublayer 2 (all-padding shards)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = self.config.hidden_size
        layer = self.layers[2]
        layer.rows3 = nn.Parameter(torch.randn(3, hidden))
        layer.rows1 = nn.Parameter(torch.randn(1, hidden))
        layer.rows9 = nn.Parameter(torch.randn(9, hidden))


# ---------------------------------------------------------------------------------------------
# Worker helpers (imported inside the spawned processes)
# ---------------------------------------------------------------------------------------------


def _build():
    from tests.unit_tests.transformer.loop.hybrid_loop_utils import build_local_loop_model

    return build_local_loop_model(PATTERN, CkptStack, prelude=2, coda=2, **MODEL)


def _perturbed(seed=77):
    """A model whose every parameter is a seeded value (identical on every rank), and those
    values by logical name."""
    from megatron.core.transformer.loop.param_store_binding import named_logical_parameters

    model = _build()
    full = {}
    with torch.no_grad():
        for i, (name, param) in enumerate(sorted(named_logical_parameters(model))):
            gen = torch.Generator().manual_seed(seed * 1000 + i)
            param.copy_(torch.randn(param.shape, generator=gen, dtype=torch.float32))
            full[name] = param.detach().clone()
    return model, full


def _store(model, world, z, modes=None):
    from megatron.core.transformer.loop.param_store import StoreModes
    from tests.unit_tests.transformer.loop import param_store_utils as psu

    if z is None:
        return None
    store = psu.install_store(model, psu.make_store_groups(world, z), z)
    if modes is not None:
        store.configure_modes(
            StoreModes.parse(MODE_TEXT[modes]), core_plan=psu.core_plan_of("whole_round", CORE)
        )
    return store


def _logical(model, store):
    """Every logical weight (stored ones all-gathered over the shard axis, pad rows cut) and
    every stored tensor's shard pad rows."""
    import torch.distributed as dist

    from megatron.core.transformer.loop.param_store_checkpoint import RowShard

    out, pads = {}, {}
    stored = set()
    if store is not None:
        for tensor in store.tensors():
            stored.add(tensor.name)
            shard = tensor.shard.detach().clone()
            parts = [torch.empty_like(shard) for _ in range(store.shard_size)]
            dist.all_gather(parts, shard, group=store.groups.shard)
            out[tensor.name] = torch.cat(parts)[: tensor.rows]
            pads[tensor.name] = shard[RowShard.of(tensor, store.shard_rank).local_rows :]
    for name, param in model.named_parameters():
        if not name.startswith("loop_param_store.") and name not in stored:
            out[name] = param.detach().clone()
    return out, pads


POISON = 7.0


def _poison_pads(store, tensors=None):
    """Fill every canonical shard's pad rows (or those of ``tensors``, per store tensor: a
    ``{name: tensor of the shard shape}`` map such as optimizer states) with ``POISON``, so a
    load that leaves pad rows untouched fails the zero-pad checks."""
    from megatron.core.transformer.loop.param_store_checkpoint import RowShard

    if store is None:
        return
    with torch.no_grad():
        for tensor in store.tensors():
            local = RowShard.of(tensor, store.shard_rank).local_rows
            target = tensor.shard if tensors is None else tensors.get(tensor.name)
            if target is not None and target.shape[0] > local:
                target[local:].fill_(POISON)


def _mismatches(got, full):
    return sorted(
        name for name in full if name not in got or not torch.equal(got[name], full[name])
    ) + sorted(set(got) - set(full))


def _shared_dir(root, name):
    import torch.distributed as dist

    path = os.path.join(root, name)
    if dist.get_rank() == 0:
        os.makedirs(path, exist_ok=True)
    dist.barrier()
    return path


def _save(model, path, **kwargs):
    import torch.distributed as dist

    from megatron.core import dist_checkpointing

    dist_checkpointing.save(model.sharded_state_dict(), path, **kwargs)
    dist.barrier()


def _load(model, path, **kwargs):
    from megatron.core import dist_checkpointing

    loaded = dist_checkpointing.load(model.sharded_state_dict(), path, **kwargs)
    model.load_state_dict(loaded)
    return loaded


# ---------------------------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------------------------


def _matrix_worker(rank, world, root):
    """Save at {none, 2, 4}, load at {none, 2, 4} with the load-side modes cycling."""
    from megatron.core.transformer.loop.param_store_binding import is_placeholder
    from megatron.core.transformer.loop.param_store_schedule import SegmentKey
    from tests.unit_tests.transformer.loop.multi_eu_utils import cpu_worker_setup

    cpu_worker_setup()
    sizes = (None, 2, 4)
    full = None
    for z in sizes:
        model, full = _perturbed()
        _store(model, world, z)
        _save(model, _shared_dir(root, f"save_{z}"))
    results = []
    for i, (save_z, load_z) in enumerate((a, b) for a in sizes for b in sizes):
        model = _build()
        modes = MODES[i % len(MODES)]
        store = _store(model, world, load_z, modes)
        if store is not None:
            store.refresh()  # stale bound copies before the load
        _poison_pads(store)  # the load must zero the pad rows itself
        _load(model, os.path.join(root, f"save_{save_z}"))
        got, pads = _logical(model, store)
        record = {
            "case": (save_z, load_z, modes if store else None),
            "mismatch": _mismatches(got, full),
            "pads_nonzero": sorted(n for n, p in pads.items() if p.numel() and p.abs().sum() > 0),
        }
        if store is not None:
            record["unbound"] = all(
                t.bound is None and is_placeholder(t.compute.data) for t in store.tensors()
            )
            record["stale"] = not store._fresh  # pylint: disable=protected-access
            layer2 = [t for t in store.tensors() if t.name.startswith("decoder.layers.2.")]
            if modes != "lw":
                with torch.no_grad():
                    try:
                        with store.visit("sub", 2):
                            pass
                        record["eager_raises"] = False
                    except RuntimeError as error:
                        record["eager_raises"] = "stale" in str(error)
                store.refresh()
                record["refreshed"] = all(
                    torch.equal(t.compute.detach(), full[t.name]) for t in store.tensors()
                )
            else:
                with torch.no_grad(), store.segment(SegmentKey("sub", 2)):
                    record["refreshed"] = all(
                        torch.equal(t.compute.detach(), full[t.name]) for t in layer2
                    )
                record["eager_raises"] = True
        results.append(record)
    return results


def _hooks_worker(rank, world, root):
    """The load hooks at z = 4."""
    import torch.distributed as dist

    from megatron.core import dist_checkpointing
    from megatron.core.transformer.loop.param_store_checkpoint import MIRROR_PREFIX, RowShard
    from tests.unit_tests.transformer.loop.multi_eu_utils import cpu_worker_setup

    cpu_worker_setup()
    z = 4
    out = {}
    model, full = _perturbed()
    store = _store(model, world, z, "lw")
    path = _shared_dir(root, "z4")
    _save(model, path)

    # In-place merge: the loaded entry is the canonical shard itself (no allocation).
    other = _build()
    other_store = _store(other, world, z, "r")
    _poison_pads(other_store)
    template = other.sharded_state_dict()
    loaded = dist_checkpointing.load(template, path)
    same, zeros = [], []
    for tensor in other_store.tensors():
        value = loaded[tensor.name]
        if RowShard.of(tensor, other_store.shard_rank).local_rows:
            same.append(value is tensor.shard)
        else:
            zeros.append(bool((value == 0).all()) and value is not tensor.shard)
    out["in_place"] = all(same) and len(same) > 0
    out["empty_ranks_zero"] = all(zeros)
    out["empty_count"] = len(zeros)
    other.load_state_dict(loaded)
    out["consumed"] = sorted(set(other_store._ckpt_consumed.values()))  # pylint: disable=W0212
    other_logical, other_pads = _logical(other, other_store)
    out["dist_mismatch"] = _mismatches(other_logical, full)
    out["dist_pads_zero"] = all(not p.numel() or p.abs().sum() == 0 for p in other_pads.values())

    # A legacy full-tensor state dict of a store-free model (pre-hook slicing).
    reference, _ = _perturbed(seed=91)
    expected = {name: p.detach().clone() for name, p in reference.named_parameters()}
    legacy = _build()
    legacy_store = _store(legacy, world, z, "rsg")
    _poison_pads(legacy_store)
    legacy.load_state_dict(reference.state_dict())
    got, pads = _logical(legacy, legacy_store)
    out["legacy_mismatch"] = _mismatches(got, expected)
    out["legacy_consumed"] = sorted(set(legacy_store._ckpt_consumed.values()))  # noqa
    out["legacy_pads_zero"] = all(not p.numel() or p.abs().sum() == 0 for p in pads.values())
    rows1 = next(t for t in legacy_store.tensors() if t.name.endswith("rows1"))
    shard_rank = legacy_store.shard_rank
    out["rows1"] = (shard_rank, tuple(rows1.shard.shape), bool((rows1.shard != 0).any()))

    # A mirror-keyed state_dict() round trip at the same z.
    mirror = _build()
    mirror_store = _store(mirror, world, z, "lw")
    plain = model.state_dict()
    assert any(key.startswith(MIRROR_PREFIX) for key in plain)
    mirror.load_state_dict(plain)
    out["mirror_consumed"] = sorted(set(mirror_store._ckpt_consumed.values()))  # noqa
    out["mirror_mismatch"] = _mismatches(_logical(mirror, mirror_store)[0], full)

    # A padded-vocab change in a full state dict.
    bad = reference.state_dict()
    weight = bad["embedding.word_embeddings.weight"]
    bad["embedding.word_embeddings.weight"] = torch.cat((weight, weight[:2]))
    try:
        legacy.load_state_dict(bad)
        out["vocab"] = None
    except ValueError as error:
        out["vocab"] = str(error)

    # A missing logical entry: the post-hook raises also under strict=False.
    partial = reference.state_dict()
    del partial["decoder.layers.2.rows3"]
    try:
        legacy.load_state_dict(partial, strict=False)
        out["missing"] = None
    except RuntimeError as error:
        out["missing"] = str(error)

    # Inside a training step: no sharded state dict, no load.
    store.begin_step()
    try:
        try:
            model.sharded_state_dict()
            out["view_in_step"] = None
        except RuntimeError as error:
            out["view_in_step"] = str(error)
        try:
            model.load_state_dict(model.state_dict())
            out["load_in_step"] = None
        except RuntimeError as error:
            out["load_in_step"] = str(error)
    finally:
        store.abort_step()
    dist.barrier()
    return out


def _adam_grads(full):
    """Seeded logical gradients, one per logical weight."""
    grads = {}
    for i, name in enumerate(sorted(full)):
        gen = torch.Generator().manual_seed(500 + i)
        grads[name] = torch.randn(full[name].shape, generator=gen)
    return grads


def _adam_params(model, store):
    """(logical name, parameter, the store tensor or None) in model.parameters() order."""
    by_shard = {id(t.shard): t for t in store.tensors()} if store else {}
    out = []
    for name, param in model.named_parameters():
        tensor = by_shard.get(id(param))
        out.append((tensor.name if tensor else name, param, tensor))
    return out


def _adam_run(model, store, grads, steps):
    """``torch.optim.Adam`` over ``model``'s parameters, ``steps`` steps on the logical
    ``grads`` (each rank's shard rows for stored tensors)."""
    from megatron.core.transformer.loop.param_store_binding import shard_slice

    params = _adam_params(model, store)
    optim = torch.optim.Adam([p for _, p, _ in params], lr=1e-2)
    for _ in range(steps):
        for name, param, tensor in params:
            grad = grads[name]
            if tensor is not None:
                grad = shard_slice(grad, tensor.shard_size, store.shard_rank)
            param.grad = grad.clone()
        optim.step()
    return optim, params


def _adam_sharded(model, optim):
    from megatron.core.dist_checkpointing.optimizer import (
        get_param_id_to_sharded_param_map,
        optim_state_to_sharding_state,
    )

    model_sd = model.sharded_state_dict()
    param_map = get_param_id_to_sharded_param_map(model_sd, optim.param_groups[0]["params"])
    optim_sd = optim.state_dict()
    optim_state_to_sharding_state(optim_sd, param_map, exclude_keys=("step",))
    return {"model": model_sd, "optimizer": optim_sd}


def _adam_logical(optim, params, store):
    """``{"<state>.<logical name>": logical tensor}`` (stored tensors all-gathered over the shard
    axis, pad rows cut) and the stored tensors' local pad rows."""
    import torch.distributed as dist

    from megatron.core.transformer.loop.param_store_checkpoint import RowShard

    out, pads = {}, {}
    for index, (name, _param, tensor) in enumerate(params):
        state = optim.state_dict()["state"][index]
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            if tensor is not None:
                parts = [torch.empty_like(value) for _ in range(store.shard_size)]
                dist.all_gather(parts, value.contiguous(), group=store.groups.shard)
                local = RowShard.of(tensor, store.shard_rank).local_rows
                pads[f"{key}.{name}"] = value[local:]
                value = torch.cat(parts)[: tensor.rows]
            out[f"{key}.{name}"] = value.clone()
    return out, pads


def _helper_check(optim, model, store, states):
    """``store_ckpt_gpu_utils.logical_optimizer_states`` (the GPU tests' whole-run view) on the
    generic path against ``_adam_logical``'s ``states``: the names that differ and its
    problems (uncovered / conflict / pad)."""
    from tests.unit_tests.transformer.loop import store_ckpt_gpu_utils as cku

    helper, problems = cku.logical_optimizer_states(optim, model, store)
    renamed = {
        f"{name}.{key}": value
        for key_name, value in states.items()
        for key, name in [key_name.split(".", 1)]
    }
    params = {f"{name}.param": value for name, value in _logical(model, store)[0].items()}
    return _mismatches(helper, {**renamed, **params}) + problems


def _adam_load(world, path, load_z, grads, steps_saved):
    """A fresh model (store at ``load_z``) and Adam whose weight and state pad rows are poisoned,
    loaded from ``path``: (logical states, their pad rows, logical weights, weight pad rows)."""
    from megatron.core import dist_checkpointing

    other = _build()
    other_store = _store(other, world, load_z)
    # One step on zero gradients initializes the state (overwritten by the load).
    zero = {name: torch.zeros_like(g) for name, g in grads.items()}
    other_optim, other_params = _adam_run(other, other_store, zero, steps=1)
    for key in ("exp_avg", "exp_avg_sq"):
        _poison_pads(
            other_store,
            {t.name: other_optim.state[p][key] for _, p, t in other_params if t is not None},
        )
    _poison_pads(other_store)
    loaded = dist_checkpointing.load(_adam_sharded(other, other_optim), path)
    other.load_state_dict(loaded["model"])
    # ``step`` is excluded from the sharded state (a scalar; Megatron's optimizers keep it in
    # their param groups): restore the saved count.
    for state in loaded["optimizer"]["state"].values():
        state["step"] = torch.tensor(float(steps_saved))
    other_optim.load_state_dict(loaded["optimizer"])
    states, pads = _adam_logical(other_optim, other_params, other_store)
    weights, weight_pads = _logical(other, other_store)
    assert _helper_check(other_optim, other, other_store, states) == []
    return states, pads, weights, weight_pads


def _zero(pads):
    return all(not p.numel() or p.abs().sum() == 0 for p in pads.values())


def _adam_worker(rank, world, root):
    """``torch.optim.Adam`` state through the generic optimizer path, z = 2 -> z = 4 / none."""
    import torch.distributed as dist

    from megatron.core import dist_checkpointing
    from tests.unit_tests.transformer.loop.multi_eu_utils import cpu_worker_setup

    cpu_worker_setup()
    model, full = _perturbed()
    grads = _adam_grads(full)
    store = _store(model, world, 2)
    assert grads.keys() == full.keys()
    optim, params = _adam_run(model, store, grads, steps=2)
    saved, saved_pads = _adam_logical(optim, params, store)
    saved_weights = _logical(model, store)[0]  # after the two Adam steps
    helper = _helper_check(optim, model, store, saved)
    path = _shared_dir(root, "adam_z2")
    dist_checkpointing.save(_adam_sharded(model, optim), path)
    dist.barrier()
    out = {"saved_pads_zero": _zero(saved_pads), "helper": helper}
    for load_z in (4, None):
        states, pads, weights, weight_pads = _adam_load(world, path, load_z, grads, 2)
        out[f"z{load_z}"] = {
            "mismatch": _mismatches(states, saved),
            "pads_zero": _zero(pads) and _zero(weight_pads),
            "weights": _mismatches(weights, saved_weights),
        }
    return out


def _world_save_worker(rank, world, root):
    """W = 4, z = 2: the model after two Adam steps and its Adam state saved; rank 0 writes the
    logical weights and states the W = 8 load compares against."""
    import torch.distributed as dist

    from megatron.core import dist_checkpointing
    from tests.unit_tests.transformer.loop.multi_eu_utils import cpu_worker_setup

    cpu_worker_setup()
    model, full = _perturbed()
    grads = _adam_grads(full)
    store = _store(model, world, 2)
    optim, params = _adam_run(model, store, grads, steps=2)
    states, _ = _adam_logical(optim, params, store)
    weights = _logical(model, store)[0]
    dist_checkpointing.save(_adam_sharded(model, optim), _shared_dir(root, "world4"))
    dist.barrier()
    if rank == 0:
        torch.save({"states": states, "weights": weights}, os.path.join(root, "world4.pt"))
    dist.barrier()
    return True


def _world_load_worker(rank, world, root):
    """W = 8: the W = 4 checkpoint loaded at z = 4 (replicate 2), z = 8 and without a store."""
    from tests.unit_tests.transformer.loop.multi_eu_utils import cpu_worker_setup

    cpu_worker_setup()
    expected = torch.load(os.path.join(root, "world4.pt"))
    _, full = _perturbed()
    grads = _adam_grads(full)
    out = {}
    for load_z in (4, 8, None):
        states, pads, weights, weight_pads = _adam_load(
            world, os.path.join(root, "world4"), load_z, grads, 2
        )
        out[f"z{load_z}"] = {
            "states": _mismatches(states, expected["states"]),
            "weights": _mismatches(weights, expected["weights"]),
            "pads_zero": _zero(pads) and _zero(weight_pads),
        }
    return out


def _empty_worker(rank, world, root):
    """W = 8, z = 8: plain and fully parallel saves with three all-padding ranks."""
    from megatron.core.dist_checkpointing.strategies.fully_parallel import (
        FullyParallelSaveStrategyWrapper,
    )
    from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
    from megatron.core.transformer.loop.param_store_checkpoint import RowShard
    from tests.unit_tests.transformer.loop.multi_eu_utils import cpu_worker_setup

    cpu_worker_setup()
    out = {}
    for parallel in (False, True):
        model, full = _perturbed()
        store = _store(model, world, 8)
        rows9 = next(t for t in store.tensors() if t.name.endswith("rows9"))
        out["empty9"] = RowShard.of(rows9, store.shard_rank).local_rows == 0
        path = _shared_dir(root, f"z8_{int(parallel)}")
        kwargs = {}
        if parallel:
            kwargs["sharded_strategy"] = FullyParallelSaveStrategyWrapper(
                TorchDistSaveShardedStrategy()
            )
        _save(model, path, **kwargs)
        for load_z in (8, 2):
            other = _build()
            other_store = _store(other, world, load_z, "rsg")
            _poison_pads(other_store)
            _load(other, path)
            got, pads = _logical(other, other_store)
            out[(parallel, load_z)] = {
                "mismatch": _mismatches(got, full),
                "pads_zero": all(not p.numel() or p.abs().sum() == 0 for p in pads.values()),
            }
    return out


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def ckpt_root():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as root:
        yield root


def test_cross_z_and_cross_mode_loads_are_bitwise(ckpt_root):
    results = run_spawned(_matrix_worker, 4, ckpt_root, timeout=900)
    for rank, records in enumerate(results):
        assert len(records) == 9
        for record in records:
            case = record["case"]
            assert record["mismatch"] == [], (rank, case)
            assert record["pads_nonzero"] == [], (rank, case)
            if case[1] is not None:
                assert record["unbound"] and record["stale"], (rank, case)
                assert record["eager_raises"] and record["refreshed"], (rank, case)


def test_load_hooks(ckpt_root):
    results = run_spawned(_hooks_worker, 4, ckpt_root, timeout=600)
    for rank, out in enumerate(results):
        assert out["in_place"] and out["empty_ranks_zero"], rank
        assert out["consumed"] == ["factory"], rank
        assert out["dist_mismatch"] == [] and out["dist_pads_zero"], rank
        assert out["legacy_mismatch"] == [] and out["legacy_pads_zero"], rank
        assert out["legacy_consumed"] == ["full"], rank
        assert out["mirror_consumed"] == ["mirror"] and out["mirror_mismatch"] == [], rank
        assert "a padded-vocab change" in out["vocab"], rank
        assert "saved 132 rows, this run 130 rows" in out["vocab"], rank
        assert out["missing"].startswith(
            "loop parameter store: the loaded model state has no entry for "
            "decoder.layers.2.rows3"
        ), rank
        assert out["view_in_step"] == "store: checkpoint view inside a training step", rank
        assert out["load_in_step"] == "store: checkpoint load inside a training step", rank
    # The [1, h] parameter at z = 4: shard rank 0 holds the row, ranks 1-3 only padding.
    empties = {rank: out["empty_count"] for rank, out in enumerate(results)}
    # rows1 (per 1): ranks 1-3 empty; rows3 (per 1): rank 3; rows9 (per 3): rank 3.
    assert empties == {0: 0, 1: 1, 2: 1, 3: 3}
    for rank, out in enumerate(results):
        shard_rank, shape, nonzero = out["rows1"]  # plain types: no tensor crosses the queue
        assert shape[0] == 1
        assert nonzero == (shard_rank == 0), rank


def test_adam_state_reshards_through_the_generic_optimizer_path(ckpt_root):
    results = run_spawned(_adam_worker, 4, ckpt_root, timeout=600)
    for rank, out in enumerate(results):
        assert out["saved_pads_zero"], rank
        assert out["helper"] == [], rank
        for key in ("z4", "zNone"):
            assert out[key]["mismatch"] == [], (rank, key)
            assert out[key]["pads_zero"], (rank, key)
            assert out[key]["weights"] == [], (rank, key)


def test_world_size_change_w4_to_w8(ckpt_root):
    """A data-parallel (world-size) change: saved at W = 4 (z = 2), loaded at W = 8."""
    assert run_spawned(_world_save_worker, 4, ckpt_root, timeout=600) == [True] * 4
    results = run_spawned(_world_load_worker, 8, ckpt_root, timeout=900)
    for rank, out in enumerate(results):
        for key in ("z4", "z8", "zNone"):
            assert out[key]["states"] == [] and out[key]["weights"] == [], (rank, key)
            assert out[key]["pads_zero"], (rank, key)


def test_all_padding_shards_plain_and_fully_parallel_save(ckpt_root):
    results = run_spawned(_empty_worker, 8, ckpt_root, timeout=900)
    assert [out["empty9"] for out in results] == [False] * 5 + [True] * 3
    for rank, out in enumerate(results):
        for parallel in (False, True):
            for load_z in (8, 2):
                record = out[(parallel, load_z)]
                assert record["mismatch"] == [] and record["pads_zero"], (rank, parallel, load_z)
