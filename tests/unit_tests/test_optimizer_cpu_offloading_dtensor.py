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
    return full.chunk(WORLD, dim=0)[rank] if rank < len(full.chunk(WORLD, dim=0)) else full[:0]


def _fulls(dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=generator).to(dtype) for shape, _ in LAYOUT]


def _make_params(mesh, rank, dtype, dtensor, seed=0):
    """``(params, buffer)``: every parameter's local shard is a view of ``buffer``."""
    from torch.distributed.tensor import DTensor, Replicate, Shard

    fulls = _fulls(dtype, seed)
    locals_ = [_local_chunk(full, placement, rank) for full, (_, placement) in zip(fulls, LAYOUT)]
    buffer = torch.empty(sum(local.numel() for local in locals_), dtype=dtype)
    params, offset = [], 0
    for full, local, (_, placement) in zip(fulls, locals_, LAYOUT):
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


def _grads(step, rank, dtype):
    """Per-parameter local gradient shards of ``step`` (None for an empty shard)."""
    generator = torch.Generator().manual_seed(1000 + step)
    grads = []
    for shape, placement in LAYOUT:
        full = torch.randn(shape, generator=generator).to(dtype)
        local = _local_chunk(full, placement, rank)
        grads.append(local if local.numel() > 0 else None)
    return grads


def _set_grads(params, grads, grad_attr):
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


def _reference(rank, dtype, fused, seed=0):
    """Plain fp32 masters of the local shards stepped by the same CPU optimizer class."""
    masters = [
        torch.nn.Parameter(_local_chunk(full, placement, rank).float().clone())
        for full, (_, placement) in zip(_fulls(dtype, seed), LAYOUT)
    ]
    return masters, torch.optim.AdamW(masters, fused=fused, **ADAMW)


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
