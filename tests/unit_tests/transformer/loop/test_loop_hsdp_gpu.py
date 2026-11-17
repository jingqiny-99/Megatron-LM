# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-FSDP HSDP on a fixed-depth DSv4 loop (GPU; the lifted guard ``FSDP_HSDP`` of
``validate_loop_fsdp_config``).

Two training steps with an SGD step between them (no momentum: the step-2 weights are a plain
function of step 1's gradients) of the model of ``strategy_grad_utils`` (a core of 8 sublayers
run exactly 4 times, block + edge ``loop_recompute`` with stride 4, three microbatches of one
row per rank, per-token loss, ``finalize_model_grads``) on 4 GPUs (or 8) as two
distributed-optimizer instances (``num_distributed_optimizer_instances = 2``: DP-Shard 2 x
DP-Outer 2), from the same weights and rows in every arm:

* ``ref``: Megatron DDP over the whole data-parallel group (fp32 gradient all-reduce);
* ``hsdp``: Megatron-FSDP v1 ZeRO-3 (``optim_grads_params``) inside each instance, replicated
  across them (``outer_dp_sharding_strategy='no_shard'``), accumulate mode, fp32 main and
  communication gradients;
* ``hfsdp``: the same with the optimizer state sharded across the instances too
  (``outer_dp_sharding_strategy='optim'``);
* ``hfsdp_recipe``: the Pro FSDP recipes' layout: dense ZeRO-3, experts ZeRO-2
  (``expert_data_parallel_sharding_strategy='optim_grads'``), outer ``optim``. At the default
  EP (``W / 2``) the expert DP-Shard group is one rank, as at EP 128 on 256 GPUs with two
  instances: the experts' buckets are reduced once per step, over DP-Outer;
* ``hsdp_expert_optim``: outer ``no_shard`` with experts ZeRO-1 (``optim``: an unsharded
  gradient buffer of which only the rank's shard is reduced).

Megatron-FSDP reduces a parameter's gradient from its ``register_post_accumulate_grad_hook``,
which fires once per backward, after the last of the 4 uses (autograd's ``AccumulateGrad``):
every microbatch reduce-scatters a bucket once over DP-Shard into the accumulating shard, and
the last microbatch's reduction also reduces that shard over DP-Outer (``outer_fsdp_group_grad_
reduce``): once per step, after the last use. Step 2 also runs the parameter all-gathers of the
updated weights (under HFSDP: the DP-Outer and DP-Shard gathers).

* Every parameter's reduced fp32 gradient of both steps matches ``ref``'s: cosine >= 0.9999 and
  relative difference <= 1e-3 (a gradient zero to fp32 precision: absolute, at most 1e-6 of the
  whole gradient's norm); step 1's losses and every row's LM loss equal ``ref``'s, step 2's
  losses are within 1e-5.
* The recorded reductions, every step: every gradient bucket is reduced over DP-Shard exactly
  once per microbatch (not once per use; a bucket whose DP-Shard group is one rank once per
  step), and over DP-Outer exactly once per step, in the last microbatch.
* ``cpu_adam`` (the recipes' optimizer): ``hfsdp_recipe`` with Adam offloaded to the host
  (``optimizer_cpu_offload``, fraction 1.0, precision aware) builds and steps. Every leaf
  optimizer is a ``HybridDeviceOptimizer`` whose parameters are the Megatron-FSDP DTensors,
  each offloaded to a plain host tensor of its local shard; at the step, every parameter's
  updated host master matches a reference CPU AdamW (the CPU optimizer's own param group) on
  the host master and the gradient's local shard (relative 1e-6), the parameter's local shard
  (the FSDP main weight) equals the host master, and the state is presented as DTensors over
  host tensors. Step 1's gradients pass the gate against a DDP run, and its step-2 losses are
  within 1e-3 of that run stepped with the GPU Adam (a different update kernel) and closer to
  it than to the same run with a learning rate of zero (the update happened).
* ``cpu_adam`` checkpoint: the same optimizer on ``hfsdp_recipe`` (in ``Float16Module``, as in
  training), two steps on random gradients, an ``fsdp_dtensor`` save and a strict load into a
  fresh model and optimizer as save_checkpoint and load_checkpoint build them
  (``preprocess_fsdp_dtensor_state_dict``: the SwiGLU split of every ``linear_fc1``, the
  expert renames, the uneven-DTensor metadata). Every parameter's weight, host master,
  moments and step are restored bitwise, and the moments stay on the host.
* Teeth, each must fail (step 1): ``overwrite`` (accumulate mode off, the loop guard bypassed:
  each use overwrites the bucket, so the core's fused-wgrad weights keep one round);
  ``outer_each`` (the DP-Outer reduction on every microbatch: earlier microbatches' sums are
  reduced again); ``outer_never`` (no DP-Outer reduction: each instance keeps its own half).

Launch: ``torchrun --nproc-per-node 4 -m pytest -q tests/unit_tests/transformer/loop/
test_loop_hsdp_gpu.py``. ``LOOP_HSDP_EP`` sets the expert-parallel size (default ``W / 2``, the
recipes' expert layout; run it also with ``LOOP_HSDP_EP=1``, where the experts are sharded over
DP-Shard as well). ``MBS_REPORT_DIR`` dumps every per-parameter report.
"""

import collections
import os
import traceback

import pytest
import torch

from tests.unit_tests.transformer.loop import strategy_grad_utils as sgu
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    CUDA,
    determinism_env,
    model_parallel_once,
)

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
INSTANCES = 2
EP = int(os.environ.get("LOOP_HSDP_EP", str(max(WORLD // INSTANCES, 1))))
STEPS = 2
ADAM_LR = 1e-3
CPU_ADAM_LOSS_RELATIVE = 1e-3
HOST_ADAM_RTOL = 1e-6
HOST_ADAM_ATOL = 1e-9
pytestmark = pytest.mark.skipif(
    not CUDA
    or WORLD < 4
    or WORLD % (2 * INSTANCES) != 0
    or WORLD % EP != 0
    or (WORLD // EP) % INSTANCES != 0
    or sgu.NUM_EXPERTS % EP != 0,
    reason="HSDP needs CUDA, DP-Shard >= 2 in each of the 2 instances (W = 4 or 8) and an "
    "expert data-parallel size divisible by the instances",
)


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    if not CUDA:
        yield
        return
    with (
        determinism_env(),
        model_parallel_once(
            expert_model_parallel_size=EP, num_distributed_optimizer_instances=INSTANCES
        ),
    ):
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(False)


def _ddp(model):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.process_groups_config import ProcessGroupCollection

    return DistributedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            overlap_grad_reduce=False, use_distributed_optimizer=False, grad_reduce_in_fp32=True
        ),
        model,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )


def _hsdp(model, outer, accumulate=True, expert=None):
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel

    return FullyShardedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            use_distributed_optimizer=True,
            data_parallel_sharding_strategy="optim_grads_params",
            expert_data_parallel_sharding_strategy=expert,
            num_distributed_optimizer_instances=INSTANCES,
            outer_dp_sharding_strategy=outer,
            megatron_fsdp_accumulate_main_grad=accumulate,
            megatron_fsdp_main_grads_dtype=torch.float32,
            megatron_fsdp_grad_comm_dtype=torch.float32,
        ),
        model,
    )


def _adam(lr=ADAM_LR, cpu_offload=False):
    """Adam as the recipes set it; ``cpu_offload``: on the host (HybridDeviceOptimizer, the
    recipes' --optimizer-cpu-offload with fraction 1.0 and the precision-aware optimizer)."""
    fields = dict(optimizer="adam", lr=lr, adam_beta1=0.9, adam_beta2=0.95)
    if cpu_offload:
        fields.update(
            optimizer_cpu_offload=True,
            optimizer_offload_fraction=1.0,
            use_precision_aware_optimizer=True,
        )
    return fields


class _ReductionRecorder:
    """Records every bucket-group reduction of Megatron-FSDP (``GradReducePipeline.
    _bucket_group_gradient_reduce``) with the step, the microbatch and whether it reduced over
    DP-Outer; ``outer`` (``"each"`` / ``"never"``) overrides that flag for the mutations."""

    def __init__(self, outer=None):
        from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
            GradReducePipeline,
        )

        self.cls = GradReducePipeline
        self.original = GradReducePipeline._bucket_group_gradient_reduce
        self.step = 0
        self.microbatch = -1
        self.calls = []  # (step, microbatch, bucket ids, outer)
        recorder = self

        def recording(
            pipeline, bucket_group, async_op=False, outer_fsdp_group_grad_reduce=False, **kw
        ):
            if outer == "each":
                outer_fsdp_group_grad_reduce = True
            elif outer == "never":
                outer_fsdp_group_grad_reduce = False
            recorder.calls.append(
                (
                    recorder.step,
                    recorder.microbatch,
                    tuple(bucket_group),
                    bool(outer_fsdp_group_grad_reduce),
                )
            )
            return recorder.original(
                pipeline,
                bucket_group,
                async_op=async_op,
                outer_fsdp_group_grad_reduce=outer_fsdp_group_grad_reduce,
                **kw,
            )

        self.recording = recording

    def on_microbatch(self, index):
        self.microbatch = index

    def __enter__(self):
        self.cls._bucket_group_gradient_reduce = self.recording
        return self

    def __exit__(self, *exc):
        self.cls._bucket_group_gradient_reduce = self.original


def _leaf_optimizers(optimizer):
    """Every leaf Megatron optimizer (a ``ChainedOptimizer`` of dense and expert groups is
    walked through ``chained_optimizers``), in a fixed order."""
    pending, found = [optimizer], []
    while pending:
        current = pending.pop()
        chained = getattr(current, "chained_optimizers", None)
        if chained is not None:
            pending.extend(chained)
        else:
            found.append(current)
    return found


def _inner_optimizers(optimizer):
    """The torch-level optimizer of every leaf Megatron optimizer."""
    return [leaf.optimizer for leaf in _leaf_optimizers(optimizer)]


def _host(tensor):
    """A CPU fp32 copy of ``tensor`` (a DTensor: its local shard)."""
    from torch.distributed.tensor import DTensor

    with torch.no_grad():
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
        return tensor.detach().float().cpu().clone()


class _HostAdamCheck:
    """Checks the CPU-offloaded Adam (``HybridDeviceOptimizer``) on the Megatron-FSDP v1
    parameters. At build: every leaf optimizer is one, every parameter is a DTensor offloaded
    to a plain host tensor of its local shard (no device optimizer). At every step, every
    parameter with a gradient: the updated host master equals, bitwise, a reference CPU AdamW
    (the CPU optimizer's own param group) stepping a copy of the host master before the step
    with the gradient's local shard, and the parameter's local shard (the FSDP main weight)
    equals it; the state is presented as DTensors over host tensors."""

    def __init__(self, opt):
        from torch.distributed.tensor import DTensor

        from megatron.core.optimizer import HybridDeviceOptimizer

        self.problems = []
        self.checked = 0
        self.inexact = 0
        self.steps = 0
        self.hdos = _inner_optimizers(opt)
        for hdo in self.hdos:
            if not isinstance(hdo, HybridDeviceOptimizer):
                self.problems.append(f"leaf optimizer {type(hdo).__name__}")
                continue
            if hdo.gpu_optimizer is not None:
                self.problems.append("a device optimizer at offload fraction 1.0")
            for group in hdo.param_groups:
                for param in group["params"]:
                    host = hdo.gpu_params_map_cpu_copy.get(param)
                    if not isinstance(param, DTensor):
                        self.problems.append(f"a non-DTensor parameter {tuple(param.shape)}")
                    elif param not in hdo.offloaded_dtensor_params or host is None:
                        self.problems.append(f"a DTensor not offloaded {tuple(param.shape)}")
                    elif type(host) is not torch.Tensor or host.device.type != "cpu":
                        self.problems.append(f"host copy {type(host).__name__} {host.device}")
            hdo.register_step_pre_hook(self._before)
            hdo.register_step_post_hook(self._after)
        self.pending = {}

    @staticmethod
    def _cpu_group(hdo, host):
        for cpu_optimizer in hdo.cpu_optimizers:
            for group in cpu_optimizer.param_groups:
                if any(p is host for p in group["params"]):
                    return cpu_optimizer, group
        raise AssertionError("host copy not in a CPU optimizer")

    def _before(self, hdo, args, kwargs):
        records = []
        for group in hdo.param_groups:
            for param in group["params"]:
                grad = getattr(param, "decoupled_grad", param.grad)
                host = hdo.gpu_params_map_cpu_copy.get(param)
                if grad is None or host is None or host.numel() == 0:
                    continue
                cpu_optimizer, cpu_group = self._cpu_group(hdo, host)
                state = {
                    key: value.detach().clone()
                    for key, value in cpu_optimizer.state.get(host, {}).items()
                    if isinstance(value, torch.Tensor)
                }
                records.append((param, host.detach().clone(), _host(grad), state))
        self.pending[id(hdo)] = records

    def _after(self, hdo, args, kwargs):
        from torch.distributed.tensor import DTensor

        self.steps += 1
        for param, master_before, grad, state in self.pending.pop(id(hdo)):
            host = hdo.gpu_params_map_cpu_copy[param]
            _, cpu_group = self._cpu_group(hdo, host)
            master = torch.nn.Parameter(master_before)
            meta = {key: value for key, value in cpu_group.items() if key != "params"}
            reference = torch.optim.AdamW([{**meta, "params": [master]}])
            if state:
                reference.state[master] = {key: value.clone() for key, value in state.items()}
            master.grad = grad
            reference.step()
            shape = tuple(param.shape)
            expected = master.detach()
            # The same CPU kernel on the same inputs: equal up to a vector-tail rounding
            # difference; a wrong or missing update is ~lr per element.
            if not torch.allclose(host, expected, rtol=HOST_ADAM_RTOL, atol=HOST_ADAM_ATOL):
                diff = (host - expected).abs().max().item()
                self.problems.append(f"host master {shape} differs from AdamW by {diff:.3g}")
            elif not torch.equal(host, expected):
                self.inexact += 1
            if not torch.equal(_host(param), host.to(param.dtype).float()):
                self.problems.append(f"parameter shard {shape} is not the host master")
            presented = hdo.state.get(param, {})
            for key in ("exp_avg", "exp_avg_sq"):
                value = presented.get(key)
                if not isinstance(value, DTensor) or value.to_local().device.type != "cpu":
                    self.problems.append(f"state {key} of {shape}: {type(value).__name__}")
            self.checked += 1

    def summary(self):
        return dict(
            problems=self.problems, checked=self.checked, inexact=self.inexact, steps=self.steps
        )


def _run_ddp(state, steps=STEPS, **optimizer_fields):
    """``steps`` steps of the DDP reference; returns ``[(step result, grads)]`` and the core's
    fused-wgrad weight names."""
    model = sgu.build_model(state, expert_model_parallel_size=EP)
    ddp = _ddp(model)
    opt = sgu.optimizer(ddp, **optimizer_fields)
    out = []
    for step in range(steps):
        opt.zero_grad()
        result = sgu.run_step(ddp, model, sgu.make_rows(step))
        out.append((result, sgu.ddp_grads(ddp, model)))
        if step + 1 < steps:
            sgu.step_optimizer(opt)
    fused = sgu.fused_core_weights(model)
    del opt, ddp, model
    torch.cuda.empty_cache()
    return out, fused


def _run_fsdp(
    state,
    outer,
    *,
    accumulate=True,
    expert=None,
    outer_override=None,
    steps=STEPS,
    optimizer_fields=None,
    optimizer_check=None,
):
    """``steps`` steps of one Megatron-FSDP arm; returns ``[(step result, grads)]`` and the
    reduction record (with ``optimizer_check``, a class built on the optimizer: its
    ``summary()`` under ``"optimizer_check"``)."""
    from megatron.core.transformer.loop import guards

    original_guard = guards.validate_loop_fsdp_config
    if not accumulate:
        # The mutation: the loop guard would refuse it (FSDP_OVERWRITE_MAIN_GRAD).
        guards.validate_loop_fsdp_config = lambda config, ddp_config: None
    try:
        model = sgu.build_model(state, expert_model_parallel_size=EP)
        fsdp = _hsdp(model, outer, accumulate=accumulate, expert=expert)
    finally:
        guards.validate_loop_fsdp_config = original_guard
    opt = sgu.optimizer(fsdp, **(optimizer_fields or {}))
    check = optimizer_check(opt) if optimizer_check is not None else None
    out = []
    with _ReductionRecorder(outer=outer_override) as recorder:
        for step in range(steps):
            recorder.step = step
            opt.zero_grad()
            result = sgu.run_step(
                fsdp, model, sgu.make_rows(step), on_microbatch=recorder.on_microbatch
            )
            out.append((result, sgu.fsdp_grads(fsdp)))
            if step + 1 < steps:
                sgu.step_optimizer(opt)
    # {bucket id: DP-Shard size}: a bucket whose DP-Shard group is one rank shards nothing and
    # is reduced once per step (MegatronFSDP._reduces_grad_every_backward).
    buckets = {
        bucket_id: fsdp.module.grad_reduce_pipeline.get_fsdp_buffer(
            bucket_id
        ).data_parallel_group.size()
        for bucket_id, group in enumerate(fsdp.param_and_grad_buffer.parameter_groups)
        if group.main_grad_buffer is not None and group.requires_grad
    }
    fsdp.stop_communication()
    del opt, fsdp, model
    torch.cuda.empty_cache()
    record = dict(calls=recorder.calls, buckets=buckets)
    if check is not None:
        record["optimizer_check"] = check.summary()
    return out, record


def _fsdp_dtensor_state(fsdp, opt, is_loading):
    """The state of an ``fsdp_dtensor`` checkpoint as save_checkpoint and load_checkpoint build
    it: the model and every leaf optimizer's ``sharded_param_state_fsdp_dtensor`` (which, when
    loading, first allocates the state with a zero-gradient step), preprocessed by
    ``preprocess_fsdp_dtensor_state_dict`` with the model's SwiGLU and expert settings (the
    SwiGLU split of every ``linear_fc1``, the expert renames, the uneven-DTensor metadata).
    Also returns the leaf optimizers and their unsplit state dicts (a load hands those to the
    optimizers' ``load_state_dict``)."""
    from types import SimpleNamespace

    from megatron.training.checkpointing import preprocess_fsdp_dtensor_state_dict

    leaves = _leaf_optimizers(opt)
    raws = [leaf.sharded_param_state_fsdp_dtensor(is_loading=is_loading) for leaf in leaves]
    optimizer_state = {}
    for raw in raws:
        optimizer_state.update(raw["state"])
    args = SimpleNamespace(
        swiglu=bool(fsdp.config.gated_linear_unit), num_experts=fsdp.config.num_moe_experts
    )
    state = preprocess_fsdp_dtensor_state_dict(
        args,
        {"model": fsdp.state_dict_for_save_checkpoint(), "optimizer": {"state": optimizer_state}},
        fsdp,
    )
    return state, leaves, raws


def _random_step(hdo):
    """One step of ``hdo`` on random gradients of every parameter's local shard (as DTensors
    like the parameter; ``randn_like`` of an unevenly sharded DTensor would not match the
    local shard's shape)."""
    from torch.distributed.tensor import DTensor

    for group in hdo.param_groups:
        for param in group["params"]:
            local = param.to_local()
            if local.numel() == 0:
                param.grad = None
                continue
            param.grad = DTensor.from_local(
                torch.randn_like(local),
                param.device_mesh,
                param.placements,
                run_check=False,
                shape=param.shape,
                stride=param.stride(),
            )
    hdo.step()
    hdo.zero_grad()


def _cpu_adam_checkpoint_round_trip(state):
    """``fsdp_dtensor`` save of ``hfsdp_recipe`` with the CPU-offloaded Adam after two steps
    on random gradients (``HybridDeviceOptimizer.dummy_step``: moments and weights move), then
    a load into a fresh model and optimizer (strict: every key it asks for must be in the
    checkpoint), as load_checkpoint does it. The model is wrapped in ``Float16Module`` as in
    training, which gives the state dict keys the ``module.`` levels the checkpoint handlers
    expect. Returns the parameters whose weight, host master, moments or step differ (bitwise)
    from the saved run's, or whose moments are not on the host, the number of parameters
    checked and how many of them are SwiGLU ``linear_fc1`` weights (whose state the checkpoint
    splits into W and V)."""
    import shutil
    import tempfile

    import torch.distributed.checkpoint as dcp

    from megatron.core.transformer.module import Float16Module

    rank = torch.distributed.get_rank()
    directory = [tempfile.mkdtemp(prefix="loop_hsdp_ckpt_") if rank == 0 else None]
    torch.distributed.broadcast_object_list(directory, src=0)
    directory = directory[0]
    arms = []
    try:
        for train in (True, False):
            model = sgu.build_model(state, expert_model_parallel_size=EP)
            fsdp = _hsdp(
                Float16Module(model.config, model),
                ARMS["hfsdp_recipe"]["outer"],
                expert=ARMS["hfsdp_recipe"]["expert"],
            )
            opt = sgu.optimizer(fsdp, **_adam(cpu_offload=True))
            arms.append((model, fsdp, opt))
            if train:
                for _ in range(2):
                    for hdo in _inner_optimizers(opt):
                        _random_step(hdo)
        (_, fsdp_a, opt_a), (_, fsdp_b, opt_b) = arms
        saved, leaves_a, _ = _fsdp_dtensor_state(fsdp_a, opt_a, is_loading=False)
        dcp.save(saved, checkpoint_id=directory)
        loading, leaves_b, raws_b = _fsdp_dtensor_state(fsdp_b, opt_b, is_loading=True)
        with torch.no_grad():
            dcp.load(loading, checkpoint_id=directory)
        for leaf, raw in zip(leaves_b, raws_b):
            leaf.load_state_dict(raw)

        problems, checked, fc1 = [], 0, 0
        for leaf_a, leaf_b in zip(leaves_a, leaves_b):
            hdo_a, hdo_b = leaf_a.optimizer, leaf_b.optimizer
            params_b = {
                leaf_b._param_name(param): param
                for group in hdo_b.param_groups
                for param in group["params"]
            }
            for group in hdo_a.param_groups:
                for param_a in group["params"]:
                    name = leaf_a._param_name(param_a)
                    param_b = params_b[name]
                    if not torch.equal(_host(param_a), _host(param_b)):
                        problems.append(f"{name}: weight")
                    master_a = hdo_a.param_to_inner_param[param_a]
                    master_b = hdo_b.param_to_inner_param[param_b]
                    if not torch.equal(master_a, master_b):
                        problems.append(f"{name}: host master")
                    state_a, state_b = hdo_a.state.get(param_a, {}), hdo_b.state.get(param_b, {})
                    if set(state_a) != set(state_b):
                        problems.append(f"{name}: state keys {set(state_a)} {set(state_b)}")
                        continue
                    for key in ("step", "exp_avg", "exp_avg_sq"):
                        if key not in state_a:
                            continue
                        if not torch.equal(_host(state_a[key]), _host(state_b[key])):
                            problems.append(f"{name}: {key}")
                        if key != "step" and state_b[key].to_local().device.type != "cpu":
                            problems.append(f"{name}: {key} on {state_b[key].to_local().device}")
                    checked += 1
                    fc1 += int("linear_fc1" in name)
        return dict(problems=problems, checked=checked, fc1=fc1)
    finally:
        for _, fsdp, _ in arms:
            fsdp.stop_communication()
        del arms
        torch.cuda.empty_cache()
        torch.distributed.barrier()
        if rank == 0:
            shutil.rmtree(directory, ignore_errors=True)


ARMS = {
    "hsdp": dict(outer="no_shard"),
    "hfsdp": dict(outer="optim"),
    "hfsdp_recipe": dict(outer="optim", expert="optim_grads"),
    "hsdp_expert_optim": dict(outer="no_shard", expert="optim"),
}


@pytest.fixture(scope="module")
def results():
    """Every arm, run once from the same weights and rows. The ``cpu_adam`` smoke arm runs
    last, and an error of it (raised identically on every rank, e.g. while building the
    optimizer) is kept for its own test to raise: it does not take the gated arms and the
    mutations down with it."""
    state = sgu.clone_state(sgu.build_model(expert_model_parallel_size=EP))
    out = {}
    out["ref"], out["fused_core_weights"] = _run_ddp(state)
    for arm, kwargs in ARMS.items():
        out[arm] = _run_fsdp(state, **kwargs)
    out["overwrite"] = _run_fsdp(state, "no_shard", accumulate=False, steps=1)
    out["outer_each"] = _run_fsdp(state, "no_shard", outer_override="each", steps=1)
    out["outer_never"] = _run_fsdp(state, "no_shard", outer_override="never", steps=1)
    out["ref_adam"], _ = _run_ddp(state, **_adam())
    out["ref_frozen"], _ = _run_ddp(state, **_adam(lr=0.0))
    try:
        out["cpu_adam"] = _run_fsdp(
            state,
            **ARMS["hfsdp_recipe"],
            optimizer_fields=_adam(cpu_offload=True),
            optimizer_check=_HostAdamCheck,
        )
    except Exception:  # pylint: disable=broad-except
        out["cpu_adam_error"] = traceback.format_exc()
    try:
        out["cpu_adam_checkpoint"] = _cpu_adam_checkpoint_round_trip(state)
    except Exception:  # pylint: disable=broad-except
        out["cpu_adam_checkpoint_error"] = traceback.format_exc()
    return out


def _name(text):
    return sgu.tag(f"HSDP EP{EP}", text)


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("step", range(STEPS))
def test_gradients_match_the_reference(results, arm, step):
    _, reference = results["ref"][step]
    _, grads = results[arm][0][step]
    report, failing = sgu.report_and_fail(_name(f"{arm} step {step + 1} vs ref"), grads, reference)
    assert not failing, sgu.mbs.worst(report, failing)


@pytest.mark.parametrize("arm", list(ARMS))
def test_losses_match_the_reference(results, arm):
    for step in range(STEPS):
        reference, _ = results["ref"][step]
        result, _ = results[arm][0][step]
        sgu.assert_step_losses_match(step, result, reference, _name(f"{arm} step {step + 1}"))


@pytest.mark.parametrize("arm", list(ARMS))
def test_reductions_once_per_microbatch_and_outer_once_per_step(results, arm):
    """Every gradient bucket, every step: one DP-Shard reduction per microbatch (after its
    parameters' last use, not per use), and one DP-Outer reduction per step, in the last
    microbatch."""
    _, record = results[arm]
    assert record["buckets"], record
    for step in range(STEPS):
        per_microbatch = collections.Counter()
        outer = collections.Counter()
        for call_step, microbatch, bucket_ids, with_outer in record["calls"]:
            if call_step != step:
                continue
            for bucket_id in bucket_ids:
                per_microbatch[(microbatch, bucket_id)] += 1
                if with_outer:
                    outer[(microbatch, bucket_id)] += 1
        print(
            f"[{_name(f'{arm} step {step + 1}')}] reductions {dict(per_microbatch)} outer "
            f"{dict(outer)}",
            flush=True,
        )
        for bucket_id, shard_size in record["buckets"].items():
            expected = [1] * sgu.ROWS if shard_size > 1 else [0] * (sgu.ROWS - 1) + [1]
            got = [per_microbatch[(microbatch, bucket_id)] for microbatch in range(sgu.ROWS)]
            assert got == expected, (arm, step, bucket_id, shard_size, got)
            assert outer[(sgu.ROWS - 1, bucket_id)] == 1, (arm, step, bucket_id)
            earlier = sum(outer[(m, bucket_id)] for m in range(sgu.ROWS - 1))
            assert earlier == 0, (arm, step, bucket_id, earlier)


def test_recipe_arm_has_one_rank_expert_shards_at_the_default_ep(results):
    """At EP = W / 2 the recipes' expert layout holds: some buckets (the experts') have a
    one-rank DP-Shard group, so the gate checked the once-per-step expert path."""
    _, record = results["hfsdp_recipe"]
    sizes = sorted(set(record["buckets"].values()))
    print(f"[{_name('hfsdp_recipe')}] DP-Shard sizes {sizes}", flush=True)
    if EP != WORLD // INSTANCES:
        pytest.skip(f"EP {EP} is not the recipes' expert layout (W / {INSTANCES})")
    assert 1 in sizes and max(sizes) > 1, sizes


def test_cpu_offloaded_adam_steps_on_the_recipe_layout(results):
    """Smoke test of the recipes' optimizer on the recipes' layout (see the module docstring)."""
    if "cpu_adam_error" in results:
        pytest.fail(f"the cpu_adam arm raised:\n{results['cpu_adam_error']}", pytrace=False)
    check = results["cpu_adam"][1]["optimizer_check"]
    print(f"[{_name('cpu_adam host Adam')}] {check}", flush=True)
    assert not check["problems"], check["problems"][:20]
    # One optimizer step (between the two training steps) of every leaf optimizer, and every
    # parameter with a local shard and a gradient checked.
    assert check["steps"] >= 1 and check["checked"] > 0, check
    _, reference = results["ref_adam"][0]
    _, grads = results["cpu_adam"][0][0]
    report, failing = sgu.report_and_fail(_name("cpu_adam step 1 vs ref"), grads, reference)
    assert not failing, sgu.mbs.worst(report, failing)
    sgu.assert_losses_match(results["cpu_adam"][0][0][0], results["ref_adam"][0][0], _name("cpu"))
    step2 = results["cpu_adam"][0][1][0]
    to_adam = sgu.mbs.relative_loss_differences(step2.losses, results["ref_adam"][1][0].losses)
    to_frozen = sgu.mbs.relative_loss_differences(step2.losses, results["ref_frozen"][1][0].losses)
    print(f"[{_name('cpu_adam step 2')}] rel to GPU Adam {to_adam}, to lr 0 {to_frozen}")
    assert max(to_adam.values()) <= CPU_ADAM_LOSS_RELATIVE, to_adam
    assert max(to_adam.values()) < max(to_frozen.values()), (to_adam, to_frozen)


def test_cpu_offloaded_adam_fsdp_dtensor_checkpoint_round_trip(results):
    """The recipes' optimizer through an ``fsdp_dtensor`` save and load on the recipes' layout
    (see ``_cpu_adam_checkpoint_round_trip``): weights, host masters, moments and steps are
    restored bitwise and the moments stay on the host, SwiGLU ``linear_fc1`` included."""
    if "cpu_adam_checkpoint_error" in results:
        pytest.fail(
            f"the cpu_adam checkpoint round trip raised:\n{results['cpu_adam_checkpoint_error']}",
            pytrace=False,
        )
    record = results["cpu_adam_checkpoint"]
    print(
        f"[{_name('cpu_adam checkpoint')}] checked {record['checked']} (linear_fc1 "
        f"{record['fc1']}), problems {len(record['problems'])}",
        flush=True,
    )
    assert not record["problems"], record["problems"][:20]
    assert record["checked"] > 0 and record["fc1"] > 0, record


def test_overwrite_mutation_fails_on_the_core_fused_weights(results):
    """Without accumulate mode each use overwrites the gradient bucket: every core weight whose
    weight gradient TE accumulates in place keeps one round's gradient."""
    _, reference = results["ref"][0]
    _, grads = results["overwrite"][0][0]
    report, failing = sgu.report_and_fail(_name("overwrite (mutation) vs ref"), grads, reference)
    fused = results["fused_core_weights"]
    assert fused
    passing = [name for name in fused if report[name]["ok"]]
    assert not passing, sgu.mbs.worst(report, passing)


@pytest.mark.parametrize("mutation", ["outer_each", "outer_never"])
def test_outer_reduction_mutations_fail_the_gate(results, mutation):
    _, reference = results["ref"][0]
    _, grads = results[mutation][0][0]
    report, failing = sgu.report_and_fail(_name(f"{mutation} (mutation) vs ref"), grads, reference)
    assert failing, f"the {mutation} mutation passed the gate: the test has no teeth"
    print(f"[{_name(mutation)}] failing {len(failing)} of {len(report)}", flush=True)
