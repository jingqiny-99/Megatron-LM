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
  gradient buffer that accumulates every microbatch locally and is reduced once per step, only
  its shard).

Megatron-FSDP reduces a parameter's gradient from its ``register_post_accumulate_grad_hook``,
which fires once per backward, after the last of the 4 uses (autograd's ``AccumulateGrad``).
A bucket under a gradient-sharding strategy (ZeRO-2 / ZeRO-3) over a DP-Shard group wider than
one rank is reduce-scattered over DP-Shard once per microbatch into the accumulating shard; a
``no_shard`` / ZeRO-1 bucket, or one whose DP-Shard group is one rank (the experts at the
default EP), accumulates in its persistent buffer and is reduced once per step, in the last
microbatch. The last microbatch's reduction also reduces over DP-Outer
(``outer_fsdp_group_grad_reduce``): once per step, after the last use. Step 2 also runs the
parameter all-gathers of the updated weights (under HFSDP: the DP-Outer and DP-Shard gathers).

Step 2 is teacher-forced (``strategy_grad_utils``): after its step-1 gradient is read, every
arm's reduced gradient storage takes ``ref``'s step-1 gradient, and the arm runs its own
optimizer step, weight install, gradient-buffer reset, gathers and step 2.

* Every parameter's reduced fp32 gradient of both steps matches ``ref``'s: cosine >= 0.9999 and
  relative difference <= 1e-3 (a gradient zero to fp32 precision: absolute, at most 1e-6 of the
  whole gradient's norm); the losses of both steps and every row's LM loss equal ``ref``'s
  within 1e-6. Before step 2 is gated, the precondition: the forced gradient reads back on every
  rank, and the fp32 main weights the arm's optimizer produced (this rank's slice of each, by
  the test's own index arithmetic) and the persistent buffers equal ``ref``'s masters and
  buffers exactly.
* The compute weights each arm's step-1 optimizer step installed (this rank's shard of every
  model weight buffer, by the same arithmetic) equal ``ref``'s bf16 weights after that step
  exactly. Its teeth: ``no_install`` (``hsdp`` forced, the install skipped) keeps the
  precondition but fails this check and the step-2 gate.
* Given the same step-1 gradient the un-forced ``hsdp`` path agrees at step 2: ``hsdp_unforced``
  (stepping on its own step-1 gradient) against ``ref_from_hsdp`` (``ref`` stepping on that
  gradient), from exactly equal main weights. The un-forced step 2 against ``ref`` is printed
  (diagnostic, not gated).
* The recorded reductions, every step: every bucket's DP-Shard size is the layout's (``W /
  2`` dense, ``W / (2 EP)`` expert); a bucket is reduced over DP-Shard once per microbatch (not
  once per use) when its strategy shards gradients over more than one rank, else once per step
  in the last microbatch (from the configured strategies, not from a production predicate); and
  over DP-Outer exactly once per step, in the last microbatch. The checker
  (``sgu.reduction_problems``) rejects the ``outer_each`` and ``outer_never`` records here (its
  DP-Outer branch); its DP-Shard branch rejects synthetic records in
  ``test_loop_strategy_bookkeeping.py`` (a ZeRO-1 bucket reduced every microbatch, a ZeRO-2/3
  bucket over two ranks reduced once per step).
* ``cpu_adam`` (the recipes' optimizer): ``hfsdp_recipe`` with Adam offloaded to the host
  (``optimizer_cpu_offload``, fraction 1.0, precision aware) builds and steps. Every leaf
  optimizer is a ``HybridDeviceOptimizer`` whose parameters are the Megatron-FSDP DTensors,
  each offloaded to a plain host tensor of its local shard; at the step, every parameter's
  updated host master matches a reference CPU AdamW (the CPU optimizer's own param group) on
  the host master and the gradient's local shard (relative 1e-6), the parameter's local shard
  (the FSDP main weight) equals the host master, and the state is presented as DTensors over
  host tensors. Step 1's gradients pass the gate against a DDP run (``ref_adam``); the arm is
  then forced with ``ref_adam``'s step-1 gradient, which every optimizer parameter must hold
  exactly before the host step (the gradient the host Adam consumed); its step-2 losses are
  within 1e-3 of ``ref_adam``'s, stepped with the GPU Adam (a different update kernel, so no
  exact start: the host masters' difference is printed), and closer to it than to the same
  run with a learning rate of zero (the update happened).
* ``cpu_adam`` checkpoint: the same optimizer on ``hfsdp_recipe`` (in ``Float16Module``, as in
  training), two steps on random gradients, an ``fsdp_dtensor`` save and a strict load into a
  fresh model and optimizer as save_checkpoint and load_checkpoint build them
  (``preprocess_fsdp_dtensor_state_dict``: the SwiGLU split of every ``linear_fc1``, the
  expert renames, the uneven-DTensor metadata). Every parameter's weight, host master,
  moments and step are restored exactly, and the moments stay on the host.
* Teeth, each must fail (step 1): ``overwrite`` (accumulate mode off, the loop guard bypassed:
  each use overwrites the bucket, so the core's fused-wgrad weights keep one round);
  ``outer_each`` (the DP-Outer reduction on every microbatch: earlier microbatches' sums are
  reduced again); ``outer_never`` (no DP-Outer reduction: each instance keeps its own half).
  ``overwrite`` and ``outer_each`` act on per-backward reductions: they must fail every fused
  core weight whose bucket is reduced every microbatch, and leave those reduced once per step
  (the experts at the default EP; the set is derived from the layout and must equal the one
  observed on the ``hsdp`` arm) exactly equal to the ``hsdp`` arm's. ``outer_never`` must fail
  every fused core weight, the once-per-step ones included (there must be some at ``EP = W /
  2`` and none at ``EP = 1``).

Launch: ``torchrun --nproc-per-node 4 -m pytest -q tests/unit_tests/transformer/loop/
test_loop_hsdp_gpu.py``. ``LOOP_HSDP_EP`` sets the expert-parallel size (default ``W / 2``, the
recipes' expert layout; run it also with ``LOOP_HSDP_EP=1``, where the experts are sharded over
DP-Shard as well). ``MBS_REPORT_DIR`` dumps every per-parameter report.
"""

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
DENSE = "optim_grads_params"  # the dense sharding strategy of every arm
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
            data_parallel_sharding_strategy=DENSE,
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
    parameter with a gradient: the updated host master equals, exactly, a reference CPU AdamW
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


def _run_ddp(state, steps=STEPS, forced=None, **optimizer_fields):
    """``steps`` steps of the DDP reference; returns ``[(step result, grads)]`` and a record:
    the core's fused-wgrad weight names (``fused``), the fp32 masters and the model weights
    (bf16: the masters rounded) after the step-1 update (``masters``, ``weights``), the
    persistent buffers step 2 starts with (``buffers``) and, with ``forced`` (step-1 gradients
    to step on: the teacher forcing), the read-back count (``readback``)."""
    model = sgu.build_model(state, expert_model_parallel_size=EP)
    ddp = _ddp(model)
    opt = sgu.optimizer(ddp, **optimizer_fields)
    out = []
    info = dict(masters=None, weights=None, buffers=None, readback=None)
    for step in range(steps):
        opt.zero_grad()
        result = sgu.run_step(ddp, model, sgu.make_rows(step))
        out.append((result, sgu.ddp_grads(ddp, model)))
        if step + 1 < steps:
            if forced is not None:
                sgu.force_main_grads(model, forced)
                info["readback"] = sgu.agreed(
                    len(sgu.mismatches(sgu.ddp_grads(ddp, model), forced))
                )
            sgu.step_optimizer(opt)
            info["masters"] = sgu.float16_optimizer_masters(opt, model)
            info["weights"] = {name: p.detach().clone() for name, p in model.named_parameters()}
            info["buffers"] = sgu.persistent_buffers(model)
    info["fused"] = sgu.fused_core_weights(model)
    del opt, ddp, model
    torch.cuda.empty_cache()
    return out, info


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
    forced=None,
    skip_install=False,
):
    """``steps`` steps of one Megatron-FSDP arm; returns ``[(step result, grads)]`` and the
    record: the reductions (``calls``), the bucket layout (``sgu.fsdp_bucket_layout``: DP-Shard
    sizes under ``buckets``, names, expert flags), after the step-1 update this rank's fp32 main
    weight slices (``main``), its installed compute weight slices (``installed``, or the error
    reading them under ``installed_error``) and the persistent buffers (``buffers``), and with
    ``forced`` (step-1 gradients to step on: the teacher forcing) the read-back count
    (``readback``) and the optimizer parameters' gradients against ``forced``
    (``optimizer_grads``, local). With ``optimizer_check``, a class built on the optimizer: its
    ``summary()`` under ``"optimizer_check"``. ``skip_install``: the mutation, the step-1
    optimizer step does not install the updated main weights into the compute weights
    (``copy_main_weights_to_model_weights`` a no-op)."""
    from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
        ParamAndGradBuffer,
    )
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
    record = dict(main=None, installed=None, buffers=None, readback=None, optimizer_grads=None)
    original_install = ParamAndGradBuffer.copy_main_weights_to_model_weights
    with _ReductionRecorder(outer=outer_override) as recorder:
        for step in range(steps):
            recorder.step = step
            opt.zero_grad()
            result = sgu.run_step(
                fsdp, model, sgu.make_rows(step), on_microbatch=recorder.on_microbatch
            )
            out.append((result, sgu.fsdp_grads(fsdp)))
            if step + 1 < steps:
                if forced is not None:
                    sgu.force_fsdp_grads(fsdp, forced)
                    record["readback"] = sgu.agreed(
                        len(sgu.mismatches(sgu.fsdp_grads(fsdp), forced))
                    )
                    record["optimizer_grads"] = sgu.fsdp_optimizer_grad_mismatches(fsdp, forced)
                if skip_install:
                    ParamAndGradBuffer.copy_main_weights_to_model_weights = lambda buffer: None
                try:
                    sgu.step_optimizer(opt)
                finally:
                    ParamAndGradBuffer.copy_main_weights_to_model_weights = original_install
                record["main"] = sgu.fsdp_main_weight_slices(fsdp)
                try:  # local reads only: an error is kept for the install test to report
                    record["installed"] = sgu.fsdp_model_weight_slices(fsdp)
                except Exception:  # pylint: disable=broad-except
                    record["installed_error"] = traceback.format_exc()
                record["buffers"] = sgu.persistent_buffers(model)
    layout = sgu.fsdp_bucket_layout(fsdp)
    fsdp.stop_communication()
    del opt, fsdp, model
    torch.cuda.empty_cache()
    record.update(
        calls=recorder.calls,
        buckets=layout["sizes"],
        names=layout["names"],
        expert_params=layout["expert_params"],
    )
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

    from megatron.core.utils import unwrap_model
    from megatron.training.checkpointing import preprocess_fsdp_dtensor_state_dict

    leaves = _leaf_optimizers(opt)
    raws = [leaf.sharded_param_state_fsdp_dtensor(is_loading=is_loading) for leaf in leaves]
    optimizer_state = {}
    for raw in raws:
        optimizer_state.update(raw["state"])
    args = SimpleNamespace(
        swiglu=bool(fsdp.config.gated_linear_unit), num_experts=fsdp.config.num_moe_experts
    )
    # save_checkpoint and load_checkpoint unwrap the adapter before building the state
    # dict and resolving its SwiGLU parameters through the inner MegatronFSDP.
    model = unwrap_model(fsdp)
    state = preprocess_fsdp_dtensor_state_dict(
        args,
        {"model": model.state_dict_for_save_checkpoint(), "optimizer": {"state": optimizer_state}},
        model,
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
    expect. Returns the parameters whose weight, host master, moments or step differ at all
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
    """Every arm, run once from the same weights and rows; the arms take ``ref``'s step-1
    gradient before their optimizer step (teacher forcing). The ``cpu_adam`` smoke arm runs
    last, and an error of it (raised identically on every rank, e.g. while building the
    optimizer) is kept for its own test to raise: it does not take the gated arms and the
    mutations down with it. The preconditions are agreed over the ranks at the end (in a fixed
    order) and asserted by the tests."""
    state = sgu.clone_state(sgu.build_model(expert_model_parallel_size=EP))
    out = {}
    out["ref"], out["ref_info"] = _run_ddp(state)
    out["fused_core_weights"] = out["ref_info"]["fused"]
    forced = out["ref"][0][1]
    for arm, kwargs in ARMS.items():
        out[arm] = _run_fsdp(state, **kwargs, forced=forced)
    out["overwrite"] = _run_fsdp(state, "no_shard", accumulate=False, steps=1)
    out["outer_each"] = _run_fsdp(state, "no_shard", outer_override="each", steps=1)
    out["outer_never"] = _run_fsdp(state, "no_shard", outer_override="never", steps=1)
    # Forced like hsdp, but the step-1 optimizer step installs nothing into the compute weights.
    out["no_install"] = _run_fsdp(state, **ARMS["hsdp"], forced=forced, skip_install=True)
    # Un-forced: hsdp on its own step-1 gradient, and ref stepping on that same gradient.
    out["hsdp_unforced"] = _run_fsdp(state, **ARMS["hsdp"])
    out["ref_from_hsdp"], out["ref_from_hsdp_info"] = _run_ddp(
        state, forced=out["hsdp_unforced"][0][0][1]
    )
    out["ref_adam"], out["ref_adam_info"] = _run_ddp(state, **_adam())
    out["ref_frozen"], _ = _run_ddp(state, **_adam(lr=0.0))
    try:
        out["cpu_adam"] = _run_fsdp(
            state,
            **ARMS["hfsdp_recipe"],
            optimizer_fields=_adam(cpu_offload=True),
            optimizer_check=_HostAdamCheck,
            forced=out["ref_adam"][0][1],
        )
    except Exception:  # pylint: disable=broad-except
        out["cpu_adam_error"] = traceback.format_exc()
    try:
        out["cpu_adam_checkpoint"] = _cpu_adam_checkpoint_round_trip(state)
    except Exception:  # pylint: disable=broad-except
        out["cpu_adam_checkpoint_error"] = traceback.format_exc()
    for arm in list(ARMS) + ["no_install"]:
        record = out[arm][1]
        start = sgu.fsdp_start_mismatches(record, out["ref_info"])
        record["forcing"] = sgu.forcing_record(record["readback"], start)
        record["install"] = sgu.install_record(record, out["ref_info"]["weights"])
    record, info = out["ref_from_hsdp_info"], out["hsdp_unforced"][1]
    record["forcing"] = sgu.forcing_record(
        record["readback"], sgu.fsdp_start_mismatches(info, record)
    )
    if "cpu_adam" in out:
        # The host Adam and the GPU Adam round differently: the precondition is the gradient
        # every optimizer parameter held before the host step.
        record = out["cpu_adam"][1]
        record["forcing"] = sgu.forcing_record(record["readback"], record["optimizer_grads"])
    return out


def _name(text):
    return sgu.tag(f"HSDP EP{EP}", text)


@pytest.mark.parametrize("arm", list(ARMS))
def test_step_two_starts_from_the_reference_state(results, arm):
    """The teacher-forcing precondition: the forced gradient read back on every rank, and the
    arm's own optimizer step on it produced ``ref``'s fp32 masters (this rank's slices) and
    buffers exactly. On failure, ``optimizer_grads`` names gradients the forcing missed."""
    record = results[arm][1]
    print(f"[{_name(arm)}] optimizer grads off the forced ones {record['optimizer_grads']}")
    sgu.assert_teacher_forced(record["forcing"], arm)


@pytest.mark.parametrize("arm", list(ARMS))
def test_step_two_runs_on_the_installed_reference_weights(results, arm):
    """The compute weights the arm's step-1 optimizer step installed (this rank's shard of
    every model weight buffer, read right after the step: ``sgu.fsdp_model_weight_slices``)
    equal ``ref``'s model weights after the same step (the fp32 masters rounded to bf16)
    exactly, on every rank. Kept out of the step-2 precondition so that a defect of this
    reader does not hide the gates; the gathers of these shards are left to the step-2 gate."""
    install = results[arm][1]["install"]
    assert install["agreed"] == 0, (
        f"{arm}: the installed compute weights differ from ref's in {len(install['local'])} "
        f"tensors on this rank ({install['agreed']} on the worst rank)",
        sorted(install["local"].items())[:8],
    )


def test_skipped_install_fails_the_install_check_and_the_step_two_gate(results):
    """Teeth of the install check and of the step-2 gate on Megatron-FSDP's step-2-only path:
    ``no_install`` (``hsdp`` forced, its step-1 optimizer step installing nothing into the
    compute weights) starts from ``ref``'s main weights and buffers (the precondition holds),
    the install check reports its stale compute weights on every rank, and its step-2
    gradients fail the gate."""
    record = results["no_install"][1]
    sgu.assert_teacher_forced(record["forcing"], "no_install")
    local = record["install"]["local"]
    print(f"[{_name('no_install')}] install check: {len(local)} tensors differ", flush=True)
    assert "error" not in local, local["error"]
    assert local and record["install"]["agreed"] > 0, record["install"]
    _, reference = results["ref"][1]
    _, grads = results["no_install"][0][1]
    report, failing = sgu.report_and_fail(
        _name("no_install (mutation) step 2 vs ref"), grads, reference
    )
    assert failing, "the no_install mutation passed the step-2 gate: the test has no teeth"
    print(f"[{_name('no_install')}] failing {len(failing)} of {len(report)}", flush=True)


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("step", range(STEPS))
def test_gradients_match_the_reference(results, arm, step):
    if step > 0:
        sgu.assert_teacher_forced(results[arm][1]["forcing"], arm)
    _, reference = results["ref"][step]
    _, grads = results[arm][0][step]
    report, failing = sgu.report_and_fail(_name(f"{arm} step {step + 1} vs ref"), grads, reference)
    assert not failing, sgu.mbs.worst(report, failing)


@pytest.mark.parametrize("arm", list(ARMS))
def test_losses_match_the_reference(results, arm):
    for step in range(STEPS):
        if step > 0:
            sgu.assert_teacher_forced(results[arm][1]["forcing"], arm)
        reference, _ = results["ref"][step]
        result, _ = results[arm][0][step]
        sgu.assert_losses_match(result, reference, _name(f"{arm} step {step + 1}"))


def test_step_two_agrees_given_the_same_step_one_gradient(results):
    """The un-forced path: ``hsdp_unforced`` steps on its own step-1 gradient and
    ``ref_from_hsdp`` (``ref``) on that same gradient. From exactly equal fp32 main weights and
    buffers, their step-2 gradients pass the gate and their losses are equal."""
    sgu.assert_teacher_forced(
        results["ref_from_hsdp_info"]["forcing"], "ref_from_hsdp", "hsdp_unforced's"
    )
    reference_result, reference = results["ref_from_hsdp"][1]
    result, grads = results["hsdp_unforced"][0][1]
    report, failing = sgu.report_and_fail(
        _name("hsdp_unforced step 2 vs ref_from_hsdp"), grads, reference
    )
    assert not failing, sgu.mbs.worst(report, failing)
    sgu.assert_losses_match(result, reference_result, _name("hsdp_unforced step 2"))


def test_report_unforced_step_two(results):
    """Diagnostic only (prints, asserts nothing): step 2 without teacher forcing against
    ``ref``: ``hsdp_unforced`` (its own step-1 gradient), and ``ref_from_hsdp`` (``ref`` on
    ``hsdp_unforced``'s step-1 gradient, which differs from its own by fp32 reassociation)."""
    _, reference = results["ref"][1]
    for arm in ("hsdp_unforced", "ref_from_hsdp"):
        steps = results[arm] if arm == "ref_from_hsdp" else results[arm][0]
        report, failing = sgu.report_and_fail(
            _name(f"{arm} step 2 vs ref (diagnostic)"), steps[1][1], reference
        )
        print(f"[{_name(arm)}] step 2 failing {len(failing)} of {len(report)}", flush=True)
        rel = sgu.mbs.relative_loss_differences(steps[1][0].losses, results["ref"][1][0].losses)
        print(f"[{_name(f'{arm} step 2')}] losses rel {rel} (diagnostic)", flush=True)
    start = sgu.slice_mismatches(
        results["hsdp_unforced"][1]["main"], results["ref_info"]["masters"]
    )
    print(
        f"[{_name('hsdp_unforced')}] step-2 main weights differ from ref's masters in "
        f"{len(start)} params: {sorted(start.items(), key=lambda item: -item[1][0])}",
        flush=True,
    )


def _reduction_problems(record, expert, steps):
    """``sgu.reduction_problems`` of an arm of this file (``expert``: its expert strategy,
    ``None`` for the dense one)."""
    return sgu.reduction_problems(
        record, dense=DENSE, expert=expert, world=WORLD, ep=EP, instances=INSTANCES, steps=steps
    )


@pytest.mark.parametrize("arm", list(ARMS))
def test_reductions_once_per_microbatch_and_outer_once_per_step(results, arm):
    """Every gradient bucket, every step (see ``_reduction_problems``): a ZeRO-2/3 bucket over
    more than one rank reduced over DP-Shard once per microbatch (after its parameters' last
    use, not per use), a ZeRO-1 or one-rank bucket once per step, and every bucket over DP-Outer
    once per step, in the last microbatch."""
    _, record = results[arm]
    assert record["buckets"], record
    expert = ARMS[arm].get("expert")
    for step in range(STEPS):
        hooks, _, outer = sgu.reduction_counts(record["calls"], step)
        print(
            f"[{_name(f'{arm} step {step + 1}')}] reductions {dict(hooks)} outer {dict(outer)} "
            f"strategies dense {DENSE} expert {expert or DENSE}",
            flush=True,
        )
    kinds = {sgu.bucket_is_expert(record, bucket_id) for bucket_id in record["buckets"]}
    assert kinds == {False, True}, kinds  # both classes are covered
    problems = _reduction_problems(record, expert, STEPS)
    assert not problems, problems[:8]


@pytest.mark.parametrize("mutation", ["outer_each", "outer_never"])
def test_reduction_checker_rejects_the_outer_mutations(results, mutation):
    """The reductions checker's DP-Outer branch has teeth: no DP-Outer reduction is a problem in
    every bucket, a DP-Outer reduction on every microbatch in every bucket reduced every
    microbatch (a bucket reduced once per step is reduced, over DP-Outer too, only in the last
    microbatch). The DP-Shard branch's teeth are CPU tests on synthetic records
    (``test_reduction_problems_check_the_dp_shard_schedule``)."""
    _, record = results[mutation]
    problems = _reduction_problems(record, None, 1)
    print(f"[{_name(mutation)}] reduction problems {problems[:4]}", flush=True)
    flagged = {problem[2] for problem in problems if problem[0] == "DP-Outer"}
    expected = set(record["buckets"])
    if mutation == "outer_each":
        expected = {
            bucket_id
            for bucket_id in record["buckets"]
            if sgu.reduced_every_microbatch(
                DENSE,
                sgu.expected_dp_shard(
                    WORLD, EP, INSTANCES, sgu.bucket_is_expert(record, bucket_id)
                ),
            )
        }
    assert expected and flagged == expected, (sorted(flagged ^ expected)[:8], len(expected))


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
    # Forced with ref_adam's step-1 gradient: every optimizer parameter held it exactly.
    sgu.assert_teacher_forced(
        results["cpu_adam"][1]["forcing"], "cpu_adam", "ref_adam's step-1 gradient (consumed)"
    )
    masters = sgu.slice_relative_difference(
        results["cpu_adam"][1]["main"], results["ref_adam_info"]["masters"]
    )
    print(f"[{_name('cpu_adam')}] host masters vs GPU Adam masters: max rel {masters:.3g}")
    step2 = results["cpu_adam"][0][1][0]
    to_adam = sgu.mbs.relative_loss_differences(step2.losses, results["ref_adam"][1][0].losses)
    to_frozen = sgu.mbs.relative_loss_differences(step2.losses, results["ref_frozen"][1][0].losses)
    print(f"[{_name('cpu_adam step 2')}] rel to GPU Adam {to_adam}, to lr 0 {to_frozen}")
    assert max(to_adam.values()) <= CPU_ADAM_LOSS_RELATIVE, to_adam
    assert max(to_adam.values()) < max(to_frozen.values()), (to_adam, to_frozen)


def test_cpu_offloaded_adam_fsdp_dtensor_checkpoint_round_trip(results):
    """The recipes' optimizer through an ``fsdp_dtensor`` save and load on the recipes' layout
    (see ``_cpu_adam_checkpoint_round_trip``): weights, host masters, moments and steps are
    restored exactly and the moments stay on the host, SwiGLU ``linear_fc1`` included."""
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


def _once_per_step(results, mutation):
    """The parameters whose buckets are reduced once per step in the mutants' configuration
    (outer ``no_shard``, experts as dense): derived from the layout, and equal to those observed
    on the ``hsdp`` arm (the same configuration, both steps) and on ``mutation``'s own step."""
    names = list(results["ref"][0][1])
    once = sgu.expected_once_per_step(
        names, dense=DENSE, expert=None, world=WORLD, ep=EP, instances=INSTANCES
    )
    observed = sgu.observed_once_per_step(results["hsdp"][1], STEPS)
    assert observed == once, sorted(observed ^ once)[:8]
    if mutation != "outer_never":  # it drops the DP-Outer reductions the observation needs
        mutant = sgu.observed_once_per_step(results[mutation][1], 1)
        assert mutant == once, sorted(mutant ^ once)[:8]
    print(f"[{_name(mutation)}] {len(once)} parameters reduced once per step", flush=True)
    return once


def test_overwrite_mutation_fails_on_the_core_fused_weights(results):
    """Without accumulate mode each use overwrites the gradient bucket: every core weight whose
    weight gradient TE accumulates in place and whose bucket is reduced every microbatch keeps
    one round's gradient. A bucket reduced once per step accumulates in its persistent buffer
    whatever the mode, so its weights must equal the ``hsdp`` arm's exactly."""
    _, reference = results["ref"][0]
    _, grads = results["overwrite"][0][0]
    report, _ = sgu.report_and_fail(_name("overwrite (mutation) vs ref"), grads, reference)
    fused = results["fused_core_weights"]
    assert fused
    excluded = sgu.assert_per_backward_mutation(
        report, grads, results["hsdp"][0][0][1], fused, _once_per_step(results, "overwrite")
    )
    print(f"[{_name('overwrite')}] once-per-step fused weights {len(excluded)}", flush=True)


@pytest.mark.parametrize("mutation", ["outer_each", "outer_never"])
def test_outer_reduction_mutations_fail_the_gate(results, mutation):
    _, reference = results["ref"][0]
    _, grads = results[mutation][0][0]
    report, failing = sgu.report_and_fail(_name(f"{mutation} (mutation) vs ref"), grads, reference)
    assert failing, f"the {mutation} mutation passed the gate: the test has no teeth"
    print(f"[{_name(mutation)}] failing {len(failing)} of {len(report)}", flush=True)
    fused = results["fused_core_weights"]
    once = _once_per_step(results, mutation)
    if mutation == "outer_each":
        # Earlier microbatches reduced again over DP-Outer: every per-backward fused weight.
        sgu.assert_per_backward_mutation(report, grads, results["hsdp"][0][0][1], fused, once)
        return
    # No DP-Outer reduction: every fused core weight fails, the once-per-step ones included,
    # which exist exactly when the experts' DP-Shard group is one rank.
    excluded = [name for name in fused if name in once]
    if sgu.expected_dp_shard(WORLD, EP, INSTANCES, True) == 1:
        assert excluded, "no once-per-step fused weight at a one-rank expert DP-Shard"
    else:
        assert not excluded, excluded[:8]
    passing = [name for name in fused if report[name]["ok"]]
    assert not passing, sgu.mbs.worst(report, passing)
