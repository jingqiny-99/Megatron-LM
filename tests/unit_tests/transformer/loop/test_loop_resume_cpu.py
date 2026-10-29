# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resume continuity of the hybrid loop on spawned CPU Gloo worlds (M3c, the CPU half of the
stage-8 resume gates; the GPU half is ``test_loop_resume_continuity{,_store}.py``).

Each case trains a local-layer dense Huginn loop model (``*-*-*-*-``, prelude 2, core 4, coda 2;
vocabulary 130 and fc1 182 rows, so stored tensors have pad rows at z = 4) **uninterrupted** for
four steps (iterations 1..4), and **resumed**: two steps, a torch_dist save of the model and the
``torch.optim.Adam`` state (the generic optimizer path, ``optim_state_to_sharding_state``),
then fresh objects (a model whose weights are perturbed first, so a missed load shows; a new
executor and optimizer) that load the checkpoint and train iterations 3..4. The data of each
iteration and the sampled depths (``SampledDepthCriterion``: counter-based on (seed,
iteration, global sample index)) are functions of the iteration only, as in training.

* RSBF (the executor, sampled R4-uniform depths, two samples per EU, rebalancing moves), at
  W = 2 and 4: without a store and with the store in Resident, Resident + sharded grads and
  Layer window modes (z = W and z = 2 with a replicate group), for the whole-round, block
  (stride 2) and nested (stride 2) items, and two waves: every resumed step's per-sample
  losses, depths and every logical gradient (replicated: summed over the domain; stored: the
  shard gradients summed over the replicate group and gathered over the shard axis) **bitwise**
  equal to the uninterrupted run's;
* cross mode: saved in the Layer window, resumed in Resident + sharded grads (bitwise-equal
  modes) against the uninterrupted Layer-window run; saved Resident, resumed in the Layer
  window against an uninterrupted run that switches the modes in process between steps 2 and 3
  (``configure_modes``): bitwise;
* cross z: saved at z = 2, resumed at z = 4 (W = 4): the first resumed step's losses and depths
  bitwise (its forward reads the same weights; later steps reduce gradients in another
  association, report only);
* the eager ``LoopHybridStack`` (the standard forward / backward, ``--loop-sample-depth`` on,
  whose depth draws use the CPU default generator) at W = 1 and 2: with the RNG state saved and
  restored as Megatron's checkpoint does (Python, NumPy, torch CPU) the resumed losses are bitwise
  equal; without restoring it (the control) they differ;
* the mHC + MTP model (Layer window, whole round; needs ``TORCHDYNAMO_DISABLE=1`` on CPU).

Workers return plain types only (the sandbox blocks the shared-memory tensor transfer).
"""

import os
import random

import pytest
import torch

from tests.unit_tests.transformer.loop.multi_eu_utils import (
    SPAWN_SKIP_REASON,
    run_spawned,
    spawn_tests_enabled,
)

pytestmark = pytest.mark.skipif(not spawn_tests_enabled(), reason=SPAWN_SKIP_REASON)

DENSE = "*-*-*-*-"
R_MAX = 4
SEQ = 16
VOCAB_ROWS = 130
PER_RANK = 2
STEPS = 4  # uninterrupted iterations 1..4
SAVE_AT = 2  # resumed: iterations 1..2, save, fresh objects, load, iterations 3..4
RSBF = dict(
    loop_schedule="rsbf",
    calculate_per_token_loss=True,
    loop_iterations=R_MAX,
    loop_exit_criterion="sampled",
    loop_depth_probs=[0.25, 0.25, 0.25, 0.25],
    loop_depth_seed=1234,
)
MHC = dict(enable_mhc_connections=True, mhc_num_residual_streams=4, mhc_sinkhorn_iterations=5)
MODES = {"r": "all=r", "rsg": "all=rsg", "lw": "core=lw,edges=lw,emb_head=rsg"}
FORMS = {
    "whole_round": {},
    "block2": dict(loop_rsbf_item_form="block", loop_rsbf_block_stride=2),
    "nested2": dict(loop_rsbf_item_form="nested", loop_rsbf_block_stride=2),
}


# ---------------------------------------------------------------------------------------------
# Worker helpers (run inside the spawned processes)
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


def _rows(iteration, world):
    """The global batch of ``iteration`` (a function of the iteration only: the data order a
    resume restores through the consumed samples)."""
    generator = torch.Generator().manual_seed(1000 + iteration)
    total = PER_RANK * world
    tokens = torch.randint(0, VOCAB_ROWS, (total, SEQ), generator=generator)
    labels = torch.randint(0, VOCAB_ROWS, (total, SEQ), generator=generator)
    loss_mask = (torch.rand(total, SEQ, generator=generator) > 0.2).float()
    return [
        dict(
            tokens=tokens[i : i + 1],
            labels=labels[i : i + 1],
            position_ids=torch.arange(SEQ)[None],
            loss_mask=loss_mask[i : i + 1],
        )
        for i in range(total)
    ]


def _build(model_name, form, waves, **overrides):
    from tests.unit_tests.transformer.loop.hybrid_loop_utils import build_local_loop_model

    pattern = DENSE + ("/*-" if "mtp" in model_name else "")
    fields = dict(RSBF, **FORMS[form], loop_rsbf_waves=waves, ffn_hidden_size=91)
    if "mhc" in model_name:
        fields.update(MHC)
    fields.update(overrides)
    return build_local_loop_model(pattern, vocab_size=VOCAB_ROWS, **fields)


def _perturb(model, seed):
    """Overwrite every parameter with seeded values (the same on every rank): a fresh process's
    different initialization, so a weight the load misses is visible."""
    with torch.no_grad():
        for index, (_, param) in enumerate(sorted(model.named_parameters())):
            generator = torch.Generator().manual_seed(seed * 1000 + index)
            param.copy_(0.02 * torch.randn(param.shape, generator=generator))


class _Groups:
    """Store groups per z, created once per world (every rank in the same order)."""

    def __init__(self, world):
        self.world = world
        self.by_z = {}

    def __call__(self, z):
        from tests.unit_tests.transformer.loop import param_store_utils as psu

        if z not in self.by_z:
            self.by_z[z] = psu.make_store_groups(self.world, z)
        return self.by_z[z]


class _RsbfRun:
    """One training process of an RSBF case: model (+ store), executor, ``torch.optim.Adam``."""

    def __init__(self, world, groups, spec, *, z, modes, perturb=None):
        import torch.distributed as dist

        from megatron.core import parallel_state
        from megatron.core.transformer.loop.dispatcher import RSBFGroups
        from megatron.core.transformer.loop.param_store import (
            LoopParamStore,
            LoopParamStoreConfig,
            StoreModes,
        )
        from megatron.core.transformer.loop.program import LoopProgram
        from megatron.core.transformer.loop.rsbf_executor import LoopRSBFExecutor
        from tests.unit_tests.transformer.loop import param_store_utils as psu

        self.world, self.rank = world, dist.get_rank()
        self.model = _build(spec["model"], spec["form"], spec.get("waves", 1))
        if perturb is not None:
            _perturb(self.model, perturb)
        self.store = None
        if z is not None:
            config = LoopParamStoreConfig(shard_size=z, min_numel=1)
            self.store = LoopParamStore.install(self.model, config, groups(z))
            psu.attach_shard_grads(self.store)
            self.store.configure_modes(StoreModes.parse(MODES[modes]))
        self.by_shard = {id(t.shard): t for t in self.store.tensors()} if self.store else {}
        self.params = [param for _, param in self.model.named_parameters()]
        self.optim = torch.optim.Adam(self.params, lr=1e-3)
        domain = parallel_state.get_data_parallel_group()
        status = parallel_state.get_data_parallel_group_gloo()
        self.executor = LoopRSBFExecutor(
            LoopProgram(self.model),
            pg_collection=_Namespace(dp=domain),
            groups=RSBFGroups(domain=domain, status=status),
            check_plan=True,
        )

    def configure_modes(self, modes):
        from megatron.core.transformer.loop.param_store import StoreModes

        self.store.configure_modes(StoreModes.parse(MODES[modes]))

    def step(self, iteration):
        """One RSBF step of ``iteration`` and the optimizer step; returns the step's record
        (per-sample losses, depths, logical gradients) in plain types."""
        import torch.distributed as dist

        from megatron.core.transformer.loop.rsbf_executor import LoopMicrobatch

        microbatches = [
            LoopMicrobatch(
                tokens=row["tokens"],
                position_ids=row["position_ids"],
                labels=row["labels"],
                loss_mask=row["loss_mask"],
                attention_mask=None,
                padding_mask=None,
                loss_func=_loss_func,
            )
            for k, row in enumerate(_rows(iteration, self.world))
            if k % self.world == self.rank
        ]
        reports = self.executor.run_step(
            lambda it, _model: next(it),
            iter(microbatches),
            len(microbatches),
            model=self.model,
            iteration=iteration,
        )
        stats = self.executor.stats
        keys = [record.key.global_index for record in stats.loss_records]
        losses = {int(k): report["lm loss"][0].item() for k, report in zip(keys, reports)}
        depths = {int(getattr(k, "global_index", k)): int(d) for k, d in stats.depths.items()}
        grads = {}
        for name, param in self.model.named_parameters():
            tensor = self.by_shard.get(id(param))
            if tensor is not None:
                grad = param.main_grad.detach().clone()
                replicate = self.store.groups.replicate
                if dist.get_world_size(replicate) > 1:
                    dist.all_reduce(grad, group=replicate)
                param.grad = grad
                parts = [torch.empty_like(grad) for _ in range(self.store.shard_size)]
                dist.all_gather(parts, grad, group=self.store.groups.shard)
                grads[tensor.name] = torch.cat(parts)[: tensor.rows]
            else:
                grad = torch.zeros_like(param) if param.grad is None else param.grad.detach()
                grad = grad.clone()
                dist.all_reduce(grad)
                param.grad = grad
                grads[name] = grad
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)
        for tensor in self.by_shard.values():
            tensor.shard.main_grad.zero_()
        return dict(losses=losses, depths=depths, grads=_bits(grads))

    # -- checkpoint -------------------------------------------------------------------------

    def _sharded(self):
        from megatron.core.dist_checkpointing.optimizer import (
            get_param_id_to_sharded_param_map,
            optim_state_to_sharding_state,
        )

        model_sd = self.model.sharded_state_dict()
        param_map = get_param_id_to_sharded_param_map(model_sd, self.params)
        optim_sd = self.optim.state_dict()
        optim_state_to_sharding_state(optim_sd, param_map, exclude_keys=("step",))
        return {"model": model_sd, "optimizer": optim_sd}

    def save(self, path):
        import torch.distributed as dist

        from megatron.core import dist_checkpointing

        dist_checkpointing.save(self._sharded(), path)
        dist.barrier()

    def load(self, path, steps_saved):
        from megatron.core import dist_checkpointing

        # The optimizer's state template: one step on zero gradients (Adam leaves the weights).
        for param in self.params:
            param.grad = torch.zeros_like(param)
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)
        loaded = dist_checkpointing.load(self._sharded(), path)
        self.model.load_state_dict(loaded["model"])
        # ``step`` is not sharded (Megatron keeps it in the optimizer's param groups).
        for state in loaded["optimizer"]["state"].values():
            state["step"] = torch.tensor(float(steps_saved))
        self.optim.load_state_dict(loaded["optimizer"])


class _Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _bits(tensors):
    """Byte views of ``tensors`` (compared bitwise: ``-0.0`` vs ``0.0`` and NaN payloads count)."""
    return {
        name: tensor.detach().contiguous().clone().view(torch.uint8)
        for name, tensor in tensors.items()
    }


def _bits_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[name], b[name]) for name in a)


def _shared_dir(root, name):
    import torch.distributed as dist

    path = os.path.join(root, name)
    if dist.get_rank() == 0:
        os.makedirs(path, exist_ok=True)
    dist.barrier()
    return path


def _same(a, b, keys=("losses", "depths", "grads")):
    return all(_bits_equal(a[k], b[k]) if k == "grads" else a[k] == b[k] for k in keys)


def _rsbf_case(world, groups, root, case):
    """``{check: bool}`` of one RSBF case (``case``: name, model, form, waves, save / load
    (z, modes), ``switch`` (the uninterrupted run switches to the load modes after step
    ``SAVE_AT``), ``first_only`` (only the first resumed step is gated))."""
    save_z, save_modes = case["save"]
    load_z, load_modes = case.get("load", case["save"])
    reference = _RsbfRun(world, groups, case, z=save_z, modes=save_modes)
    expected = []
    for iteration in range(1, STEPS + 1):
        if case.get("switch") and iteration == SAVE_AT + 1:
            reference.configure_modes(load_modes)
        expected.append(reference.step(iteration))
    first = _RsbfRun(world, groups, case, z=save_z, modes=save_modes)
    early = [first.step(iteration) for iteration in range(1, SAVE_AT + 1)]
    path = _shared_dir(root, case["name"])
    first.save(path)
    resumed = _RsbfRun(world, groups, case, z=load_z, modes=load_modes, perturb=99)
    resumed.load(path, SAVE_AT)
    late = [resumed.step(iteration) for iteration in range(SAVE_AT + 1, STEPS + 1)]
    checks = {
        "before the save == uninterrupted": all(
            _same(a, b) for a, b in zip(early, expected[:SAVE_AT])
        ),
        "the depths vary": len({d for rec in expected for d in rec["depths"].values()}) > 1,
    }
    if case.get("first_only"):
        checks["first resumed step: losses and depths bitwise"] = _same(
            late[0], expected[SAVE_AT], keys=("losses", "depths")
        )
        checks["later steps differ only in association (report)"] = True
    else:
        for offset, (got, want) in enumerate(zip(late, expected[SAVE_AT:])):
            iteration = SAVE_AT + 1 + offset
            checks[f"iteration {iteration}: losses bitwise"] = got["losses"] == want["losses"]
            checks[f"iteration {iteration}: depths bitwise"] = got["depths"] == want["depths"]
            checks[f"iteration {iteration}: gradients bitwise"] = _bits_equal(
                got["grads"], want["grads"]
            )
    return checks


def _rsbf_worker(rank, world, cases, root):
    from tests.unit_tests.transformer.loop.multi_eu_utils import (
        cpu_worker_setup,
        cpu_worker_teardown,
    )

    del rank
    cpu_worker_setup()
    try:
        groups = _Groups(world)
        return {case["name"]: _rsbf_case(world, groups, root, case) for case in cases}
    finally:
        cpu_worker_teardown()


def _assert_cases(results):
    failed = [
        f"rank {rank} {case}: {check}"
        for rank, result in enumerate(results)
        for case, checks in result.items()
        for check, ok in checks.items()
        if not ok
    ]
    assert not failed, "\n".join(failed)


# ---------------------------------------------------------------------------------------------
# RSBF cases
# ---------------------------------------------------------------------------------------------


def _cases(world, z):
    cases = []
    for form in FORMS:
        cases.append(dict(name=f"none_{form}", model="dense", form=form, save=(None, None)))
        for modes in MODES:
            cases.append(
                dict(name=f"z{z}_{modes}_{form}", model="dense", form=form, save=(z, modes))
            )
    return cases


@pytest.mark.parametrize("world", [2])
def test_rsbf_resume_every_mode_and_item_form(tmp_path, world):
    results = run_spawned(_rsbf_worker, world, _cases(world, world), str(tmp_path), timeout=900)
    _assert_cases(results)
    assert len(results[0]) == 12


def test_rsbf_resume_four_eus_with_a_replicate_group(tmp_path):
    """W = 4: z = 4 (= W) and z = 2 (replicate 2), every mode on the whole round, the block and
    nested items in the Layer window at z = 2; the store-free nested item."""
    cases = [dict(name="none_nested2", model="dense", form="nested2", save=(None, None))]
    for modes in MODES:
        cases.append(dict(name=f"z4_{modes}", model="dense", form="whole_round", save=(4, modes)))
        cases.append(dict(name=f"z2_{modes}", model="dense", form="whole_round", save=(2, modes)))
    for form in ("block2", "nested2"):
        cases.append(dict(name=f"z2_lw_{form}", model="dense", form=form, save=(2, "lw")))
    _assert_cases(run_spawned(_rsbf_worker, 4, cases, str(tmp_path), timeout=900))


def test_rsbf_resume_waves_and_cross_mode(tmp_path):
    """Two waves (Layer window, whole round and block); saved in the Layer window, resumed in
    Resident + sharded grads (bitwise-equal modes); saved Resident, resumed in the Layer window
    against the uninterrupted run that switches modes in process."""
    cases = [
        dict(name="waves2_lw", model="dense", form="whole_round", waves=2, save=(2, "lw")),
        dict(name="waves2_lw_block2", model="dense", form="block2", waves=2, save=(2, "lw")),
        dict(name="lw_to_rsg", model="dense", form="whole_round", save=(2, "lw"), load=(2, "rsg")),
        dict(
            name="r_to_lw",
            model="dense",
            form="nested2",
            save=(2, "r"),
            load=(2, "lw"),
            switch=True,
        ),
    ]
    _assert_cases(run_spawned(_rsbf_worker, 2, cases, str(tmp_path), timeout=900))


def test_rsbf_resume_across_z(tmp_path):
    """Saved at z = 2, resumed at z = 4 (W = 4), and the reverse; also into a store-free run:
    the first resumed step's losses and depths bitwise."""
    cases = [
        dict(
            name="z2_to_z4",
            model="dense",
            form="whole_round",
            save=(2, "lw"),
            load=(4, "lw"),
            first_only=True,
        ),
        dict(
            name="z4_to_z2",
            model="dense",
            form="block2",
            save=(4, "rsg"),
            load=(2, "r"),
            first_only=True,
        ),
        dict(
            name="z4_to_none",
            model="dense",
            form="whole_round",
            save=(4, "lw"),
            load=(None, None),
            first_only=True,
        ),
    ]
    _assert_cases(run_spawned(_rsbf_worker, 4, cases, str(tmp_path), timeout=900))


def test_rsbf_resume_mhc_mtp(tmp_path):
    if os.environ.get("TORCHDYNAMO_DISABLE") != "1":
        pytest.skip("mHC on CPU needs TORCHDYNAMO_DISABLE=1 (its fused ops use torch.compile)")
    cases = [
        dict(name="mhc_mtp_lw", model="mhc_mtp", form="whole_round", save=(2, "lw")),
        dict(name="mhc_mtp_none_nested", model="mhc_mtp", form="nested2", save=(None, None)),
    ]
    _assert_cases(run_spawned(_rsbf_worker, 2, cases, str(tmp_path), timeout=900))


# ---------------------------------------------------------------------------------------------
# The eager LoopHybridStack with sampled depths (the CPU default generator)
# ---------------------------------------------------------------------------------------------


def _rng_state():
    """What Megatron's checkpoint keeps of the CPU RNG streams (``get_rng_state``: Python,
    NumPy, torch CPU; the CUDA states and the tracker are CUDA-only)."""
    import numpy as np

    return dict(random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())


def _set_rng_state(state):
    import numpy as np

    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


class _EagerRun:
    """The standard schedule over the eager stack: one forward / backward per microbatch, the
    gradients summed over the world, Adam."""

    def __init__(self, world, *, perturb=None):
        import torch.distributed as dist

        from tests.unit_tests.transformer.loop.hybrid_loop_utils import build_local_loop_model

        self.world, self.rank = world, dist.get_rank()
        self.model = build_local_loop_model(
            DENSE,
            vocab_size=VOCAB_ROWS,
            ffn_hidden_size=91,
            loop_iterations=R_MAX,
            loop_sample_depth=True,
        )
        if perturb is not None:
            _perturb(self.model, perturb)
        self.model.train()
        self.params = [param for _, param in self.model.named_parameters()]
        self.optim = torch.optim.Adam(self.params, lr=1e-3)

    def step(self, iteration):
        import torch.distributed as dist

        losses = []
        for k, row in enumerate(_rows(iteration, self.world)):
            if k % self.world != self.rank:
                continue
            out = self.model(row["tokens"], row["position_ids"], None, labels=row["labels"])
            loss = (out.float() * row["loss_mask"]).sum()
            loss.backward()
            losses.append(loss.item())
        grads = {}
        for name, param in self.model.named_parameters():
            grad = torch.zeros_like(param) if param.grad is None else param.grad.detach().clone()
            dist.all_reduce(grad)
            param.grad = grad
            grads[name] = grad
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)
        return dict(losses=losses, grads=_bits(grads))

    _sharded = _RsbfRun._sharded
    save = _RsbfRun.save
    load = _RsbfRun.load


def _eager_worker(rank, world, root):
    from tests.unit_tests.transformer.loop.multi_eu_utils import (
        cpu_worker_setup,
        cpu_worker_teardown,
    )

    del rank
    cpu_worker_setup()
    try:
        torch.manual_seed(5)
        reference = _EagerRun(world)
        expected = [reference.step(iteration) for iteration in range(1, STEPS + 1)]

        torch.manual_seed(5)
        first = _EagerRun(world)
        early = [first.step(iteration) for iteration in range(1, SAVE_AT + 1)]
        path = _shared_dir(root, f"eager_w{world}")
        first.save(path)
        saved_rng = _rng_state()  # saved with the checkpoint (per rank)

        torch.manual_seed(777)  # a fresh process seeds differently
        resumed = _EagerRun(world, perturb=99)
        resumed.load(path, SAVE_AT)
        rng_after_load = _rng_state()
        _set_rng_state(saved_rng)  # load_checkpoint restores the RNG state
        late = [resumed.step(iteration) for iteration in range(SAVE_AT + 1, STEPS + 1)]

        control = _EagerRun(world, perturb=99)
        control.load(path, SAVE_AT)
        _set_rng_state(rng_after_load)  # the control does not restore it
        stale = [control.step(iteration) for iteration in range(SAVE_AT + 1, STEPS + 1)]
        return {
            "before the save == uninterrupted": all(
                _same(a, b, ("losses", "grads")) for a, b in zip(early, expected[:SAVE_AT])
            ),
            "resumed losses bitwise": [r["losses"] for r in late]
            == [r["losses"] for r in expected[SAVE_AT:]],
            "resumed gradients bitwise": all(
                _bits_equal(got["grads"], want["grads"])
                for got, want in zip(late, expected[SAVE_AT:])
            ),
            "control without the RNG state differs": [r["losses"] for r in stale]
            != [r["losses"] for r in expected[SAVE_AT:]],
        }
    finally:
        cpu_worker_teardown()


@pytest.mark.parametrize("world", [1, 2])
def test_eager_sampled_depth_resume(tmp_path, world):
    """The eager stack's ``loop_sample_depth`` draws continue bitwise when the CPU RNG streams
    are restored, and differ without them (the control).

    The RNG state is saved and restored by hand (``_rng_state`` / ``_set_rng_state``: the
    Python / NumPy / torch CPU streams that Megatron's ``get_rng_state`` keeps), not through
    ``save_checkpoint`` / ``load_checkpoint``: this test pins the property (the depth draws use
    the CPU default generator, so restoring it is sufficient and necessary) on CPU. The real
    checkpoint path (per-rank ``rng_state`` saved by ``save_checkpoint``, restored at the end of
    ``load_checkpoint`` before the data iterators are built) is covered on GPU by
    ``test_loop_resume_continuity.py`` (the ``eager_plain_sampled_depth`` arm and its
    ``--no-load-rng`` control ``test_sampled_depth_resume_needs_the_rng_state``) and by the
    toolkit chain C1s."""
    results = run_spawned(_eager_worker, world, str(tmp_path), timeout=600)
    failed = [
        f"rank {rank}: {check}"
        for rank, checks in enumerate(results)
        for check, ok in checks.items()
        if not ok
    ]
    assert not failed, "\n".join(failed)
