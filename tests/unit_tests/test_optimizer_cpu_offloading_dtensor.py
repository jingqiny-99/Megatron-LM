# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""HybridDeviceOptimizer (CPU-offloaded optimizer) on DTensor parameters (CPU tests).

Megatron-FSDP v1 hands the optimizer DTensor parameters whose local shard is a view of the
rank's shard of the main weight buffer, with DTensor gradients over the main gradient buffer.
The offloaded optimizer keeps a plain host copy of each parameter's local shard, copies the
gradient's local shard device to host, steps the CPU optimizer, and copies the updated host
shard back into the parameter's local shard (the buffer).

These tests run two gloo processes on a CPU device mesh. There is no accelerator, so each
worker makes CPU tensors offload candidates (``_is_device_tensor``), skips pinning
(``_pin_memory``) and replaces the CUDA streams and events with no-ops; everything else is the
production code. Every parameter's local shard is a view of one flat per-rank buffer (as under
Megatron-FSDP), with uneven shards, an empty shard on rank 1 (no gradient there, as
Megatron-FSDP sets it) and a replicated parameter.

* Each step of the offloaded optimizer on DTensor shards matches, bitwise, a reference CPU
  optimizer of the same class stepping a plain copy of the local shard (fp32 parameters, and
  bf16 parameters with an fp32 master); the buffer holds the update, a DTensor is never pinned
  or cloned, and the same holds for plain parameters (unchanged path).
* Checkpoint, as Megatron-FSDP's ``fsdp_dtensor`` format does it (``DistributedOptimizer.
  sharded_param_state_fsdp_dtensor`` and its ``load_state_dict``): ``state`` presents the
  per-element state and the master as DTensors sharded like the parameter over the CPU
  optimizer's own host tensors; a ``torch.distributed.checkpoint`` save of the model and that
  state, then a load into a second optimizer from other weights (after the dummy step that
  allocates its state) and its ``load_state_dict``, restores the weights, masters and state
  bitwise, in place on the host, and the next step of both matches bitwise. Also with the
  uneven-DTensor preprocessing of ``fsdp_dtensor`` checkpoints.
* ``update_fp32_param_by_new_param`` (``reload_model_params``) refreshes the host copies from
  the parameters' local shards.
"""

import contextlib
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 2
STEPS = 3
STORE_TIMEOUT = timedelta(seconds=120)
# (global shape, placement): uneven Shard(0) chunks, an empty chunk on rank 1, a replicated
# vector.
LAYOUT = (((5, 3), "shard"), ((1, 4), "shard"), ((7,), "shard"), ((4,), "replicate"))
# Without the empty chunk: the uneven-DTensor checkpoint preprocessing runs one collective per
# state key, so every rank needs the same keys (an empty shard never has optimizer state).
LAYOUT_NO_EMPTY = tuple(entry for entry in LAYOUT if entry[0] != (1, 4))
ADAMW = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _NoOpStream:
    """Stands in for ``torch.cuda.Stream`` and its events on a CPU-only run."""

    def wait_stream(self, other):
        pass

    def record_event(self):
        return self

    def synchronize(self):
        pass

    def wait(self, stream=None):
        pass


def _emulate_offload_device(pinned):
    """Make CPU tensors offload candidates in this worker; ``pinned`` records every tensor
    passed to ``_pin_memory``."""
    from megatron.core.optimizer.cpu_offloading import hybrid_optimizer

    def pin(tensor):
        pinned.append(tensor)
        return tensor

    hybrid_optimizer._is_device_tensor = lambda tensor: True
    hybrid_optimizer._pin_memory = pin
    torch.cuda.current_stream = lambda *args, **kwargs: _NoOpStream()
    torch.cuda.Stream = _NoOpStream
    torch.cuda.stream = lambda stream: contextlib.nullcontext()


def _init(rank, port):
    store = dist.TCPStore("127.0.0.1", port, WORLD, rank == 0, timeout=STORE_TIMEOUT)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD, store=store, timeout=STORE_TIMEOUT)
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh("cpu", (WORLD,))


def _local_chunk(full, placement, rank):
    if placement == "replicate":
        return full
    chunks = full.chunk(WORLD, dim=0)
    return chunks[rank] if rank < len(chunks) else full[:0]


def _fulls(dtype, seed, layout):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=generator).to(dtype) for shape, _ in layout]


def _make_params(mesh, rank, dtype, dtensor, seed=0, layout=LAYOUT):
    """``(params, buffer)``: every parameter's local shard is a view of ``buffer``."""
    from torch.distributed.tensor import DTensor, Replicate, Shard

    fulls = _fulls(dtype, seed, layout)
    locals_ = [_local_chunk(full, placement, rank) for full, (_, placement) in zip(fulls, layout)]
    buffer = torch.empty(sum(local.numel() for local in locals_), dtype=dtype)
    params, offset = [], 0
    for full, local, (_, placement) in zip(fulls, locals_, layout):
        view = buffer[offset : offset + local.numel()].view(local.shape)
        offset += local.numel()
        view.copy_(local)
        if dtensor:
            tensor = DTensor.from_local(
                view,
                mesh,
                [Replicate() if placement == "replicate" else Shard(0)],
                run_check=False,
                shape=full.shape,
                stride=full.stride(),
            )
            param = torch.nn.Parameter(tensor)
            assert param.to_local().data_ptr() == view.data_ptr()
        else:
            param = torch.nn.Parameter(view)
        params.append(param)
    return params, buffer


def _grads(step, rank, dtype, layout=LAYOUT):
    """Per-parameter local gradient shards of ``step`` (None for an empty shard)."""
    generator = torch.Generator().manual_seed(1000 + step)
    grads = []
    for shape, placement in layout:
        full = torch.randn(shape, generator=generator).to(dtype)
        local = _local_chunk(full, placement, rank)
        grads.append(local if local.numel() > 0 else None)
    return grads


def _set_grads(params, grads, grad_attr="grad"):
    from torch.distributed.tensor import DTensor

    for param, grad in zip(params, grads):
        if grad is not None and isinstance(param, DTensor):
            grad = DTensor.from_local(
                grad.clone(),
                param.device_mesh,
                param.placements,
                run_check=False,
                shape=param.shape,
                stride=param.stride(),
            )
        elif grad is not None:
            grad = grad.clone()
        if grad_attr == "decoupled_grad":
            param.decoupled_grad = grad
        else:
            param.grad = grad


def _local(tensor):
    from torch.distributed.tensor import DTensor

    with torch.no_grad():
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor.detach()


def _hdo(params, fused, overlap=True):
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

    return HybridDeviceOptimizer(
        params,
        offload_fraction=1.0,
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=torch.optim.AdamW,
        param_update_in_fp32=True,
        pin_cpu_grads=False,
        pin_cpu_params=False,
        overlap_cpu_optimizer_d2h_h2d=overlap,
        fused=fused,
        **ADAMW,
    )


def _reference(rank, dtype, fused, seed=0, layout=LAYOUT):
    """Plain fp32 masters of the local shards stepped by the same CPU optimizer class."""
    masters = [
        torch.nn.Parameter(_local_chunk(full, placement, rank).float().clone())
        for full, (_, placement) in zip(_fulls(dtype, seed, layout), layout)
    ]
    return masters, torch.optim.AdamW(masters, fused=fused, **ADAMW)


def _step(hdo, params, step, rank, dtype, layout=LAYOUT):
    hdo.zero_grad()
    _set_grads(params, _grads(step, rank, dtype, layout))
    hdo.step()


def _step_matches_reference_worker(rank, port, dtype, dtensor, fused, grad_attr, overlap):
    from torch.distributed.tensor import DTensor

    mesh = _init(rank, port)
    pinned = []
    _emulate_offload_device(pinned)
    try:
        params, buffer = _make_params(mesh, rank, dtype, dtensor)
        hdo = _hdo(params, fused, overlap)
        assert hdo.gpu_optimizer is None
        assert set(hdo.gpu_params_map_cpu_copy) == set(params)
        for param in params:
            host = hdo.gpu_params_map_cpu_copy[param]
            assert type(host) is torch.Tensor and not isinstance(host, DTensor)
            assert host.dtype == torch.float32 and host.shape == _local(param).shape
        # Only plain host tensors are pinned (fp32: the host copy; bf16: before the cast).
        assert pinned and not any(isinstance(tensor, DTensor) for tensor in pinned)

        masters, reference = _reference(rank, dtype, fused)
        for step in range(STEPS):
            grads = _grads(step, rank, dtype)
            hdo.zero_grad()
            _set_grads(params, grads, grad_attr)
            for master, grad in zip(masters, grads):
                master.grad = None if grad is None else grad.float()
            hdo.step()
            reference.step()
            for index, (param, master) in enumerate(zip(params, masters)):
                local = _local(param)
                assert type(local) is torch.Tensor
                expected = master.detach().to(dtype)
                assert torch.equal(local, expected), (rank, step, index, local, expected)
                host = hdo.gpu_params_map_cpu_copy[param]
                assert torch.equal(host, master.detach()), (rank, step, index)
            # The update landed in the flat buffer the shards view.
            assert torch.equal(
                buffer, torch.cat([master.detach().to(dtype).reshape(-1) for master in masters])
            )
        # Rank 1's empty shard never had a gradient: no state; every other one has stepped.
        for param, master in zip(params, masters):
            assert (param in hdo.state) == (master in reference.state), (rank, param.shape)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("grad_attr", ["grad", "decoupled_grad"])
@pytest.mark.parametrize("fused", [True, False], ids=["fused", "foreach"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("dtensor", [True, False], ids=["dtensor", "plain"])
def test_cpu_adam_step_matches_reference_on_the_local_shard(dtensor, dtype, fused, grad_attr):
    mp.spawn(
        _step_matches_reference_worker,
        args=(_free_port(), dtype, dtensor, fused, grad_attr, True),
        nprocs=WORLD,
        join=True,
    )


def test_cpu_adam_step_without_overlap_matches_reference():
    """``overlap_cpu_optimizer_d2h_h2d=False``: one CPU optimizer for every parameter."""
    mp.spawn(
        _step_matches_reference_worker,
        args=(_free_port(), torch.float32, True, True, "grad", False),
        nprocs=WORLD,
        join=True,
    )


PER_ELEMENT_STATE = ("exp_avg", "exp_avg_sq", "master_param")


def _assert_state_is_presented_as_dtensors(hdo, params):
    """Every stepped parameter's per-element state in ``hdo.state`` is a DTensor like the
    parameter over the CPU optimizer's own host tensor (no copy); ``step`` stays plain."""
    from torch.distributed.tensor import DTensor

    for param in params:
        if param not in hdo.state:
            assert _local(param).numel() == 0
            continue
        host_param = hdo.param_to_inner_param[param]
        (cpu_optimizer,) = [
            opt
            for opt in hdo.cpu_optimizers
            if any(p is host_param for group in opt.param_groups for p in group["params"])
        ]
        host_state = cpu_optimizer.state[host_param]
        assert "master_param" not in host_state
        state = hdo.state[param]
        assert set(state) == {"step", *PER_ELEMENT_STATE}, set(state)
        assert not isinstance(state["step"], DTensor)
        for key in PER_ELEMENT_STATE:
            value = state[key]
            assert isinstance(value, DTensor), key
            assert value.placements == param.placements and value.device_mesh == param.device_mesh
            assert value.shape == param.shape and value.dtype == torch.float32
            local = value._local_tensor
            source = host_param if key == "master_param" else host_state[key]
            assert local is source, key
            assert type(local) is torch.Tensor and local.device.type == "cpu"


def _model_and_optimizer_state(hdo, params):
    model = {f"param{index}": param for index, param in enumerate(params)}
    state = {
        f"param{index}": hdo.state[param]
        for index, param in enumerate(params)
        if param in hdo.state
    }
    return {"model": model, "optimizer": {"state": state}}


def _optimizer_state_dict_by_name(hdo, params, state):
    """What ``DistributedOptimizer.load_state_dict`` hands the optimizer under Megatron-FSDP:
    the state by parameter name and param groups whose ``params`` are names."""
    names = {param: f"param{index}" for index, param in enumerate(params)}
    groups = []
    for group in hdo.param_groups:
        meta = {k: v for k, v in group.items() if k != "params"}
        groups.append({**meta, "params": [names[param] for param in group["params"]]})
    return {"state": state, "param_groups": groups}


def _checkpoint_round_trip_worker(rank, port, checkpoint_dir, dtype, uneven):
    import torch.distributed.checkpoint as dcp

    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        preprocess_state_dict_for_uneven_dtensor,
    )

    mesh = _init(rank, port)
    _emulate_offload_device([])
    layout = LAYOUT_NO_EMPTY if uneven else LAYOUT
    try:
        params_a, _ = _make_params(mesh, rank, dtype, True, seed=0, layout=layout)
        hdo_a = _hdo(params_a, fused=True)
        for step in range(2):
            _step(hdo_a, params_a, step, rank, dtype, layout)
        _assert_state_is_presented_as_dtensors(hdo_a, params_a)
        saved = _model_and_optimizer_state(hdo_a, params_a)
        if uneven:
            preprocess_state_dict_for_uneven_dtensor(saved)
        dcp.save(saved, checkpoint_id=checkpoint_dir)

        # A second optimizer over other weights; its state is allocated by one step with zero
        # gradients (DistributedOptimizer._init_optimizer_states_with_dummy_values).
        params_b, _ = _make_params(mesh, rank, dtype, True, seed=1, layout=layout)
        hdo_b = _hdo(params_b, fused=True)
        for param in params_b:
            if _local(param).numel() > 0:
                param.grad = torch.zeros_like(param)
        hdo_b.step()
        hdo_b.zero_grad()
        _assert_state_is_presented_as_dtensors(hdo_b, params_b)
        loading = _model_and_optimizer_state(hdo_b, params_b)
        if uneven:
            preprocess_state_dict_for_uneven_dtensor(loading)
        with torch.no_grad():
            dcp.load(loading, checkpoint_id=checkpoint_dir)
        # The load landed in place: the CPU optimizer's host tensors hold the saved state.
        loaded_ptrs = {}
        for name, state in loading["optimizer"]["state"].items():
            for key in PER_ELEMENT_STATE:
                loaded_ptrs[(name, key)] = state[key]._local_tensor.data_ptr()
        # torch's load casts every state tensor to its parameter's device and dtype: it must
        # see the host copy and plain host tensors (on a GPU a DTensor parameter would move the
        # whole state to the device).
        casts = []
        policy = torch.optim.Optimizer._process_value_according_to_param_policy

        def recording_policy(param, value, *args, **kwargs):
            casts.append((type(param), type(value)))
            return policy(param, value, *args, **kwargs)

        torch.optim.Optimizer._process_value_according_to_param_policy = staticmethod(
            recording_policy
        )
        try:
            hdo_b.load_state_dict(
                _optimizer_state_dict_by_name(hdo_b, params_b, loading["optimizer"]["state"])
            )
        finally:
            torch.optim.Optimizer._process_value_according_to_param_policy = staticmethod(policy)
        assert casts and all(types == (torch.Tensor, torch.Tensor) for types in casts), casts

        _assert_state_is_presented_as_dtensors(hdo_b, params_b)
        for index, (param_a, param_b) in enumerate(zip(params_a, params_b)):
            assert torch.equal(_local(param_a), _local(param_b)), index
            # The fp32 master (bf16 parameters: not the bf16-rounded weight).
            assert torch.equal(
                hdo_a.param_to_inner_param[param_a], hdo_b.param_to_inner_param[param_b]
            ), index
            assert (param_a in hdo_a.state) == (param_b in hdo_b.state), index
            if param_a not in hdo_a.state:
                continue
            for key in ("step", *PER_ELEMENT_STATE):
                value_a, value_b = hdo_a.state[param_a][key], hdo_b.state[param_b][key]
                assert torch.equal(_local(value_a), _local(value_b)), (index, key)
            for key in ("exp_avg", "exp_avg_sq"):
                # load_state_dict kept the loaded host tensors: no copy, no device move.
                ptr = hdo_b.state[param_b][key]._local_tensor.data_ptr()
                assert ptr == loaded_ptrs[(f"param{index}", key)], (index, key)

        # The next step of both optimizers matches bitwise.
        _step(hdo_a, params_a, 2, rank, dtype, layout)
        _step(hdo_b, params_b, 2, rank, dtype, layout)
        for index, (param_a, param_b) in enumerate(zip(params_a, params_b)):
            assert torch.equal(_local(param_a), _local(param_b)), index
            if param_a in hdo_a.state:
                for key in PER_ELEMENT_STATE:
                    assert torch.equal(
                        _local(hdo_a.state[param_a][key]), _local(hdo_b.state[param_b][key])
                    ), (index, key)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("uneven", [False, True], ids=["dcp", "uneven-preprocess"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_dtensor_checkpoint_round_trip(tmp_path, dtype, uneven):
    mp.spawn(
        _checkpoint_round_trip_worker,
        args=(_free_port(), str(tmp_path / "checkpoint"), dtype, uneven),
        nprocs=WORLD,
        join=True,
    )


def _reload_model_params_worker(rank, port, dtype):
    mesh = _init(rank, port)
    _emulate_offload_device([])
    try:
        params, buffer = _make_params(mesh, rank, dtype, True)
        hdo = _hdo(params, fused=True)
        _step(hdo, params, 0, rank, dtype)
        # A model checkpoint load writes the parameters' storage (the FSDP buffer) in place.
        buffer.copy_(torch.arange(buffer.numel(), dtype=dtype) / 7)
        hdo.update_fp32_param_by_new_param()
        for param in params:
            host = hdo.param_to_inner_param[param]
            assert torch.equal(host, _local(param).float())
        masters = [torch.nn.Parameter(_local(param).float().clone()) for param in params]
        reference = torch.optim.AdamW(masters, fused=True, **ADAMW)
        reference.load_state_dict(_reference_state(hdo, params, masters, reference))
        _step(hdo, params, 1, rank, dtype)
        for master, grad in zip(masters, _grads(1, rank, dtype)):
            master.grad = None if grad is None else grad.float()
        reference.step()
        for param, master in zip(params, masters):
            assert torch.equal(_local(param), master.detach().to(dtype))
    finally:
        dist.destroy_process_group()


def _reference_state(hdo, params, masters, reference):
    """``reference``'s state dict holding ``hdo``'s current moments and steps."""
    state_dict = reference.state_dict()
    for index, param in enumerate(params):
        if param in hdo.state:
            state_dict["state"][index] = {
                key: _local(value).clone() for key, value in hdo.state[param].items()
            }
            del state_dict["state"][index]["master_param"]
    return state_dict


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_reload_model_params_refreshes_the_host_copies(dtype):
    mp.spawn(_reload_model_params_worker, args=(_free_port(), dtype), nprocs=WORLD, join=True)
