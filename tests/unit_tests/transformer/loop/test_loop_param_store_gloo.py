# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Multi-rank tests of the loop parameter store on spawned CPU Gloo worlds (W = 2 and 4).

Each test spawns a fresh Gloo world (``multi_eu_utils.run_spawned``) whose ranks are the EUs of
one RSBF domain; shard groups are consecutive ranks and replicate groups strided (the GTP grid's
layout, ``param_store_utils.make_store_groups``), with z = W (replicate 1) and z = 2 at W = 4
(replicate 2, whose sum the tests do as DDP would). Cases:

* the collective engine on real store tensors with rows not divisible by z: gathered regions equal
  the full weights and keep zero pad rows; reduce-scatters add each rank's rows of the summed
  gradient into ``shard.main_grad`` and accumulate across calls; a null EU's zero reduce adds the
  other EUs' gradients; a second reduce-scatter communicator gives the same values;
* the peer-identity check at install (one weight differing in one ulp on one rank raises);
* RSBF-shaped steps through the store's API (``StoreDriver``) on a local-layer dense loop model,
  every residency mode against Resident and against the same steps without a store: per-sample
  losses bitwise; Layer window (kept and re-gathered) vs Resident + sharded grads gradients
  bitwise; every mode vs Resident and vs no store within the stage-2 gradient rule; with block,
  nested and whole-round items, two waves, null items (an EU with fewer samples), two steps with
  an SGD update of the shards; shard pad rows exactly zero; the host call log of the store's
  collectives identical on every rank;
* finalize's shard-axis SUM of the replicated parameters (``finalize_model_grads``'s helper)
  plus the replicate all-reduce equals the domain sum, and leaves the canonical shards alone.

The DDP + distributed optimizer arm allocates on CUDA (E20) and is in ``..._ddp.py``.
"""

import pytest

from tests.unit_tests.transformer.loop.multi_eu_utils import (
    SPAWN_SKIP_REASON,
    run_spawned,
    spawn_tests_enabled,
)

pytestmark = pytest.mark.skipif(not spawn_tests_enabled(), reason=SPAWN_SKIP_REASON)

PATTERN = "*-*-*-*-"
CORE = 4
SEQ = 16
# fc1 has 2 x 91 = 182 rows: padded to a multiple of z = 4 (two pad rows on the last rank).
MODEL = dict(ffn_hidden_size=91)


# ---------------------------------------------------------------------------------------------
# Workers (module level: spawned processes import them)
# ---------------------------------------------------------------------------------------------


def _comm_worker(rank, world, z, separate):
    import torch
    from torch import nn

    from megatron.core.transformer.loop.param_store_binding import (
        layout_tensors,
        make_store_tensor,
        shard_slice,
        tensor_view,
    )
    from megatron.core.transformer.loop.param_store_comm import StoreComm
    from tests.unit_tests.transformer.loop import param_store_utils as psu

    torch.set_num_threads(1)
    groups = psu.make_store_groups(world, z, shard_rs=separate)
    shard_rank = torch.distributed.get_rank(group=groups.shard)
    torch.manual_seed(5)
    owner = nn.Module()
    owner.a = nn.Parameter(torch.randn(7, 3))  # 7 rows: padded to 8 at z = 2 / 4
    owner.b = nn.Parameter(torch.randn(4, 5))
    full = {"a": owner.a.detach().clone(), "b": owner.b.detach().clone()}
    tensors = []
    for attr in ("a", "b"):
        tensor = make_store_tensor(attr, owner, attr, shard_size=z)
        tensor.shard = nn.Parameter(shard_slice(full[attr], z, shard_rank))
        tensor.shard.main_grad = torch.zeros(tensor.shard_shape)
        tensors.append(tensor)
    numel = layout_tensors(tensors, 16)

    class Unit:
        key = "u"

    unit = Unit()
    unit.tensors = tuple(tensors)
    comm = StoreComm(groups.shard, torch.device("cpu"), rs_group=groups.shard_rs, record_calls=True)
    out = {}
    flat = torch.full((numel,), float("nan"))
    comm.gather(unit, flat)
    out["gather"] = all(torch.equal(tensor_view(flat, t), full[t.attr]) for t in tensors)
    pads = []
    for t in tensors:
        region = flat.narrow(0, t.offset + t.numel, t.numel_padded - t.numel)
        pads.append(bool((region == 0).all()))
    out["pad_rows_gathered_zero"] = all(pads)
    # Reduce: every rank contributes rank-dependent gradients, twice (accumulation).
    grads = {t.attr: torch.full(t.logical_shape, float(rank + 1)) + t.offset for t in tensors}
    src = torch.zeros(numel)
    for t in tensors:
        src.narrow(0, t.offset, t.numel).copy_(grads[t.attr].reshape(-1))
    comm.reduce(unit, src)
    comm.reduce(unit, src.clone())
    expected_ok = []
    group_ranks = torch.distributed.get_process_group_ranks(groups.shard)
    for t in tensors:
        total = sum(torch.full(t.logical_shape, float(r + 1)) + t.offset for r in group_ranks)
        expected = 2 * shard_slice(total, z, shard_rank)
        expected_ok.append(torch.equal(t.shard.main_grad, expected))
    out["reduce"] = all(expected_ok)
    # A null EU (the last shard rank) reduces zeros: the others' gradients still arrive.
    for t in tensors:
        t.shard.main_grad.zero_()
    if shard_rank == z - 1:
        comm.zero_and_reduce(unit, None, numel=numel)
    else:
        comm.reduce(unit, src.clone())
    null_ok = []
    for t in tensors:
        total = sum(
            torch.full(t.logical_shape, float(r + 1)) + t.offset for r in group_ranks[: z - 1]
        )
        null_ok.append(torch.equal(t.shard.main_grad, shard_slice(total, z, shard_rank)))
    out["null_reduce"] = all(null_ok)
    out["log"] = [op for op, _, _ in comm.log]
    return out


def _peer_check_worker(rank, world):
    import torch

    from tests.unit_tests.transformer.loop import param_store_utils as psu
    from tests.unit_tests.transformer.loop.hybrid_loop_utils import build_local_loop_model
    from tests.unit_tests.transformer.loop.multi_eu_utils import cpu_worker_setup

    cpu_worker_setup()
    groups = psu.make_store_groups(world, world)
    model = build_local_loop_model(PATTERN, prelude=2, coda=2)
    if rank == 1:
        weight = model.decoder.layers[4].self_attention.linear_qkv.weight
        with torch.no_grad():
            bits = weight.data.view(torch.int32)
            bits[3, 2] += 1  # one ulp
    try:
        psu.install_store(model, groups, world)
    except ValueError as exc:
        return str(exc)
    return None


def _modes_worker(rank, world, z, configs):
    """Run every config (a dict) and return per-config comparison results."""
    import torch
    import torch.distributed as dist

    from megatron.core.distributed.finalize_model_grads import (
        _allreduce_replicated_grads_over_gtp_remat_group,
    )
    from megatron.core.transformer.loop.param_store import StoreModes
    from tests.unit_tests.transformer.loop import param_store_utils as psu
    from tests.unit_tests.transformer.loop.hybrid_loop_utils import VOCAB, build_local_loop_model
    from tests.unit_tests.transformer.loop.multi_eu_utils import cpu_worker_setup

    cpu_worker_setup()
    groups = psu.make_store_groups(world, z)

    def build():
        return build_local_loop_model(PATTERN, prelude=2, coda=2, **MODEL)

    def samples_for(config, wave, step):
        count = config["count"]
        mine = config.get("counts", {}).get(rank, count)
        return psu.make_samples(rank, mine, vocab=VOCAB, seq=SEQ, seed=31 + 7 * step + wave)

    def zero_missing(grads, named):
        # A parameter no local item reached has no .grad: zeros in its place (same key order on
        # every rank for the all-reduces).
        out = {}
        for name, param in sorted(named, key=lambda item: item[0]):
            if "loop_param_store." in name:
                continue
            grad = grads.get(name)
            out[name] = torch.zeros_like(param, dtype=torch.float32) if grad is None else grad
        return out

    def sgd(params_and_grads, lr=1e-2):
        with torch.no_grad():
            for param, grad in params_and_grads:
                param.sub_(lr * grad.to(param.dtype))

    def run(config, modes_text):
        """One run (store or reference); returns (losses, full grads per step, model, store)."""
        plan = psu.core_plan_of(config["form"], CORE)
        model = build()
        store = None
        if modes_text is not None:
            store = psu.install_store(
                model,
                groups,
                z,
                poison_released=True,
                verify_gathers=True,
                check_visits=True,
                record_calls=True,
                **config.get("store", {}),
            )
            store.configure_modes(StoreModes.parse(modes_text), core_plan=plan)
        driver = psu.StoreDriver(model, store, depth=config["depth"], core_plan=plan)
        losses, grads_per_step, finalize_ok = [], [], True
        for step in range(config.get("steps", 1)):
            passes = [samples_for(config, w, step) for w in range(config.get("waves", 1))]
            driver.run_step(passes, config["count"])
            losses.append([s.loss for p in passes for s in p])
            if store is None:
                grads = psu.domain_sum(
                    zero_missing(psu.logical_grads(model), psu.named_logical_parameters(model))
                )
                grads_per_step.append(grads)
                sgd([(p, grads[n]) for n, p in psu.named_logical_parameters(model)])
                for param in model.parameters():
                    param.grad = None
                continue
            # Replicated parameters: finalize's shard-axis SUM, then the replicate all-reduce.
            params = dict(model.named_parameters())
            replicated = zero_missing(psu.replicated_grads(model), params.items())
            for name, grad in replicated.items():
                params[name].main_grad = grad.clone()
            shard_before = [t.shard.main_grad.clone() for t in store.tensors()]
            _allreduce_replicated_grads_over_gtp_remat_group(
                [model], groups.shard, None, calculate_per_token_loss=True
            )
            finalize_ok &= all(
                torch.equal(t.shard.main_grad, before)
                for t, before in zip(store.tensors(), shard_before)
            )
            finished = {}
            for name, grad in replicated.items():
                summed = params[name].main_grad.clone()
                if dist.get_world_size(groups.replicate) > 1:
                    dist.all_reduce(summed, group=groups.replicate)
                finished[name] = summed
            store_grads = psu.full_store_grads(store, groups)
            pads = psu.shard_pad_rows(store)
            grads = dict(finished, **store_grads)
            grads_per_step.append(grads)
            pad_zero = all(bool((p == 0).all()) for p in pads.values())
            # SGD on the shards (their replicate-summed grads) and the replicated parameters.
            updates = []
            for tensor in store.tensors():
                grad = tensor.shard.main_grad.clone()
                if dist.get_world_size(groups.replicate) > 1:
                    dist.all_reduce(grad, group=groups.replicate)
                updates.append((tensor.shard, grad))
            updates += [(params[n], g) for n, g in finished.items()]
            sgd(updates)
            for tensor in store.tensors():
                tensor.shard.main_grad.zero_()
            for name in replicated:
                params[name].grad = None
                del params[name].main_grad
            if not pad_zero:
                finalize_ok = False
        return losses, grads_per_step, model, store, finalize_ok

    results = []
    for config in configs:
        ref_losses, ref_grads, _, _, _ = run(config, None)
        runs = {}
        for name, text in config["modes"].items():
            losses, grads, _, store, ok = run(config, text)
            logs = [None] * world
            dist.all_gather_object(logs, [(op, key) for op, key, _ in store.comm.log])
            runs[name] = dict(
                losses=losses,
                grads=grads,
                ok=ok,
                same_log=all(log == logs[0] for log in logs),
                pad_tensors=len(psu.shard_pad_rows(store)),
                prefetches=store.stats.prefetches,
                carries=store.stats.carries,
            )
        result = {"config": config["name"], "checks": {}}
        checks = result["checks"]
        for name, run_result in runs.items():
            step0 = run_result["grads"][0]
            checks[f"{name}: losses == reference (step 1)"] = all(
                torch.equal(a, b) for a, b in zip(run_result["losses"][0], ref_losses[0])
            )
            checks[f"{name}: grads ~ reference"] = (
                max(psu.compare(step0[n], ref_grads[0][n]) for n in ref_grads[0]) <= 1.0
            )
            checks[f"{name}: finalize SUM, shards untouched, pad rows 0"] = run_result["ok"]
            checks[f"{name}: identical call log on every rank"] = run_result["same_log"]
            if z == 4 and dist.get_rank(group=groups.shard) == z - 1:
                # fc1's 182 rows: the pad rows are on the last shard rank (checked zero above).
                checks[f"{name}: some tensors padded"] = run_result["pad_tensors"] > 0
            if "resident" in runs and name != "resident":
                res = runs["resident"]["grads"][0]
                checks[f"{name}: grads ~ resident"] = (
                    max(psu.compare(step0[n], res[n]) for n in res) <= 1.0
                )
                checks[f"{name}: losses == resident (step 1)"] = all(
                    torch.equal(a, b)
                    for a, b in zip(run_result["losses"][0], runs["resident"]["losses"][0])
                )
        for left, right in config.get("bitwise", ()):
            a, b = runs[left], runs[right]
            for step in range(len(a["grads"])):
                checks[f"{left} == {right} grads (step {step + 1})"] = all(
                    torch.equal(a["grads"][step][n], b["grads"][step][n]) for n in a["grads"][step]
                )
                checks[f"{left} == {right} losses (step {step + 1})"] = all(
                    torch.equal(x, y) for x, y in zip(a["losses"][step], b["losses"][step])
                )
        result["lookahead"] = {n: (r["prefetches"], r["carries"]) for n, r in runs.items()}
        results.append(result)
    return results


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("world,z,separate", [(2, 2, False), (4, 4, False), (4, 2, True)])
def test_collective_engine(world, z, separate):
    results = run_spawned(_comm_worker, world, z, separate, timeout=240)
    for rank, out in enumerate(results):
        assert out["gather"], rank
        assert out["pad_rows_gathered_zero"], rank
        assert out["reduce"], rank
        assert out["null_reduce"], rank
        assert out["log"] == ["gather", "reduce", "reduce", "reduce"], rank


def test_peer_weights_must_be_identical():
    messages = run_spawned(_peer_check_worker, 2, timeout=240)
    for message in messages:
        assert message is not None and "decoder.layers.4.self_attention.linear_qkv.weight" in (
            message
        ), message


LW = "core=lw,edges=lw,emb_head=rsg"
LW_REGATHER = LW + ",core_regather=on,edge_regather=on"
RSG = "all=rsg"
RESIDENT = "all=r"
ALL_MODES = {"resident": RESIDENT, "rsg": RSG, "lw": LW, "lw_regather": LW_REGATHER}


def _assert_checks(results):
    failed = []
    for rank, per_rank in enumerate(results):
        for result in per_rank:
            for check, ok in result["checks"].items():
                if not ok:
                    failed.append(f"rank {rank} {result['config']}: {check}")
    assert not failed, "\n".join(failed)


def test_modes_at_w2():
    configs = [
        dict(
            name="whole round, 2 samples, depth 2, 2 steps",
            form="whole_round",
            count=2,
            depth=2,
            steps=2,
            modes=ALL_MODES,
            bitwise=[("lw", "rsg"), ("lw_regather", "rsg")],
        ),
        dict(
            name="block, waves 2, null items",
            form="block",
            count=2,
            depth=2,
            waves=2,
            counts={1: 1},
            modes={"resident": RESIDENT, "rsg": RSG, "lw": LW},
            bitwise=[("lw", "rsg")],
        ),
    ]
    results = run_spawned(_modes_worker, 2, 2, configs, timeout=600)
    _assert_checks(results)
    lookahead = results[0][0]["lookahead"]
    assert lookahead["lw"][0] > 0 and lookahead["lw"][1] > 0


def test_modes_at_w4_z4():
    configs = [
        dict(
            name="nested, null items",
            form="nested",
            count=2,
            depth=2,
            counts={2: 1},
            modes=ALL_MODES,
            bitwise=[("lw", "rsg"), ("lw_regather", "rsg")],
        )
    ]
    _assert_checks(run_spawned(_modes_worker, 4, 4, configs, timeout=600))


def test_modes_at_w4_z2():
    configs = [
        dict(
            name="block, replicate 2",
            form="block",
            count=1,
            depth=2,
            steps=2,
            modes=ALL_MODES,
            bitwise=[("lw", "rsg"), ("lw_regather", "rsg")],
        )
    ]
    _assert_checks(run_spawned(_modes_worker, 4, 2, configs, timeout=600))


@pytest.mark.parametrize("world", [2, 4])
def test_literal_case_every_mode_equals_resident_bitwise(world):
    """The M3 gate's literal case (z = W, one sample per EU, depth 1, ``no_overwrite``): every
    unit gets one item per step, so Layer window and Resident + sharded grads add the same fp32
    values in the same order as Resident: gradients and losses bitwise (D-S6-9)."""
    configs = [
        dict(
            name=f"literal, {form}",
            form=form,
            count=1,
            depth=1,
            store=dict(no_overwrite=True),
            modes=ALL_MODES,
            bitwise=[("lw", "resident"), ("lw_regather", "resident"), ("rsg", "resident")],
        )
        for form in ("whole_round", "block")
    ]
    _assert_checks(run_spawned(_modes_worker, world, world, configs, timeout=600))
