# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Helpers of the multi-EU RSBF tests (not collected).

``run_spawned`` runs a module-level function in ``world_size`` freshly spawned processes that
form one Gloo world on the loopback interface, and returns each rank's result. It is the CPU
harness of the multi-EU suites: they run where no ``torchrun`` world exists (local CPU
gates) and are skipped under ``torchrun`` (``spawn_tests_enabled``), where their GPU
counterparts run instead.

``RecordingLockstepMLP`` (with ``StubLockstepAdapter`` and ``stub_moe_layer_spec``) is a
test-only 'E' MLP that lockstep MoE treats like an expert-parallel MoE layer: every forward and
every backward through it exchanges a tag (module index, phase) over its group, so a lockstep
break is a recorded mismatch (or a hang inside the spawn timeout), and every call is logged.
``CollectiveOpLogger`` records the ``torch.distributed`` collectives a block issues (op, group
ranks, autograd phase), for the GPU lockstep tests.
"""

import contextlib
import os
import queue as queue_module
import socket
import time
import traceback
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from megatron.core.transformer.loop.null_items import FORM_EMPTY, LockstepAdapter, LockstepSignature
from megatron.core.transformer.module import MegatronModule


def spawn_tests_enabled() -> bool:
    """Spawned Gloo worlds run outside a torchrun world, where Gloo is available."""
    return int(os.environ.get("WORLD_SIZE", "1")) <= 1 and dist.is_gloo_available()


SPAWN_SKIP_REASON = "spawned Gloo tests run outside torchrun (their GPU counterparts run under it)"


def free_port() -> int:
    """An unused loopback TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank: int, world_size: int, port: int, fn: Callable, args: tuple, results) -> None:
    try:
        dist.init_process_group(
            "gloo", rank=rank, world_size=world_size, init_method=f"tcp://127.0.0.1:{port}"
        )
        value = fn(rank, world_size, *args)
        results.put((rank, True, value))
    except BaseException:  # pylint: disable=broad-except
        results.put((rank, False, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_spawned(fn: Callable, world_size: int, *args, timeout: float = 300.0) -> list:
    """Run ``fn(rank, world_size, *args)`` on every rank of a spawned Gloo world; returns the
    results by rank. Raises with every failing rank's traceback, or on a timeout (a hang:
    the ranks that did not report are named). ``fn`` and ``args`` must be picklable (``fn`` at
    module level)."""
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    port = free_port()
    procs = [
        ctx.Process(target=_worker, args=(rank, world_size, port, fn, args, results), daemon=True)
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    done: dict[int, tuple[bool, Any]] = {}
    deadline = time.monotonic() + timeout
    failed_at = None
    try:
        while len(done) < world_size:
            now = time.monotonic()
            # After a failure the other ranks may block in a collective: short grace.
            limit = deadline if failed_at is None else min(deadline, failed_at + 15.0)
            if now >= limit:
                break
            try:
                rank, ok, value = results.get(timeout=min(1.0, limit - now))
            except queue_module.Empty:
                continue
            done[rank] = (ok, value)
            if not ok and failed_at is None:
                failed_at = time.monotonic()
    finally:
        for proc in procs:
            proc.join(timeout=5.0 if len(done) == world_size else 0.1)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)
    failures = {rank: value for rank, (ok, value) in done.items() if not ok}
    if failures:
        text = "\n".join(f"--- rank {rank} ---\n{tb}" for rank, tb in sorted(failures.items()))
        raise RuntimeError(f"spawned ranks failed:\n{text}")
    missing = sorted(set(range(world_size)) - set(done))
    if missing:
        raise TimeoutError(f"spawned ranks {missing} did not finish within {timeout} s (hang?)")
    return [done[rank][1] for rank in range(world_size)]


def cpu_worker_setup(*, threads: int = 1) -> None:
    """Prepare a spawned CPU worker for loop models: one intra-op thread (W processes share
    the host), model parallelism over the Gloo world (TP = PP = 1, so the data-parallel group
    is the world: the RSBF domain), and the CPU environment patches the in-process fixtures
    apply (``cpu_mtp``, ``cpu_rope``, ``mhc_on_this_device``). The process exits after the
    test, so nothing is restored. Pair with ``cpu_worker_teardown``."""
    import contextlib

    import torch

    from megatron.core import parallel_state
    from megatron.core.models.common.embeddings import rotary_pos_embedding
    from megatron.core.transformer import multi_token_prediction as mtp_module

    torch.set_num_threads(threads)
    parallel_state.initialize_model_parallel(1, 1)
    mtp_module.is_inference_column_parallel_linear = lambda module: False
    mtp_module.MTPLossLoggingHelper.save_loss_to_tracker = staticmethod(
        lambda *args, **kwargs: None
    )
    rotary_pos_embedding.torch.cuda.current_device = lambda: "cpu"
    torch.cuda.nvtx.range = lambda *a, **k: contextlib.nullcontext()
    torch.cuda.nvtx.range_push = lambda *a, **k: None
    torch.cuda.nvtx.range_pop = lambda *a, **k: None


def cpu_worker_teardown() -> None:
    """Undo ``cpu_worker_setup``'s model parallelism (before the harness destroys the world)."""
    from megatron.core import parallel_state

    parallel_state.destroy_model_parallel()


# ---------------------------------------------------------------------------
# Lockstep stub ('E' sublayers whose calls are collectives over a group)
# ---------------------------------------------------------------------------

_PHASES = {"fwd": 0, "bwd": 1}


class StubLog:
    """Per-process log of the stub's exchanges: ``(marker, module index, phase)``; ``marker``
    is whatever the test last put in ``StubLog.marker`` (e.g. the current sub-round)."""

    entries: list = []
    marker: Any = None

    @classmethod
    def reset(cls) -> None:
        """Forget the entries and the marker."""
        cls.entries = []
        cls.marker = None


class _LockstepTag(torch.autograd.Function):
    """Identity whose forward and backward each exchange the module's tag."""

    @staticmethod
    def forward(ctx, hidden, module):  # pylint: disable=arguments-differ
        ctx.module = module
        module.exchange("fwd")
        return hidden.clone()

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=arguments-differ
        ctx.module.exchange("bwd")
        return grad, None


class RecordingLockstepMLP(MegatronModule):
    """A test-only 'E' MLP (``silu(x W1) W2``, batch-local like a dropless MoE layer's math)
    whose every forward and backward call all-gathers ``(module index, phase)`` over
    ``self.group`` (set by the test; None: no exchange). A tag that differs between the ranks
    counts in ``mismatches``; a missing exchange hangs. ``trip_over_budget`` (a set of call
    counts) sets ``over_budget`` on those forward calls. Built by ``TransformerLayer`` as its
    ``mlp`` submodule (the keywords ``TransformerLayer`` passes are accepted)."""

    instances: list = []

    def __init__(self, config, pg_collection=None, is_mtp_layer=False, name=None, **kwargs):
        del pg_collection, is_mtp_layer, name, kwargs
        super().__init__(config=config)
        self.index = len(RecordingLockstepMLP.instances)
        RecordingLockstepMLP.instances.append(self)
        hidden, ffn = config.hidden_size, config.ffn_hidden_size
        self.fc1 = torch.nn.Linear(hidden, ffn, bias=False, dtype=config.params_dtype)
        self.fc2 = torch.nn.Linear(ffn, hidden, bias=False, dtype=config.params_dtype)
        self.group = None
        self.mismatches = 0
        self.forward_calls = 0
        self.trip_over_budget: set = set()
        self.over_budget = torch.zeros(1, dtype=torch.bool)

    def exchange(self, phase: str) -> None:
        """Log the call and all-gather its tag over the group."""
        StubLog.entries.append((StubLog.marker, self.index, phase))
        group = self.group
        if group is None or dist.get_world_size(group) == 1:
            return
        tag = torch.tensor([self.index, _PHASES[phase]], dtype=torch.int64)
        gathered = [torch.empty_like(tag) for _ in range(dist.get_world_size(group))]
        dist.all_gather(gathered, tag, group=group)
        if any(not torch.equal(other, tag) for other in gathered):
            self.mismatches += 1

    def forward(self, hidden_states, padding_mask=None, **kwargs):
        """The real call: ``(output, None)`` like an MLP / MoE layer."""
        del padding_mask, kwargs
        self.forward_calls += 1
        if self.forward_calls in self.trip_over_budget:
            self.over_budget.fill_(True)
        x = _LockstepTag.apply(hidden_states, self)
        return self.fc2(F.silu(self.fc1(x))), None

    def null(self, hidden: torch.Tensor) -> torch.Tensor:
        """A null call: the same exchanges, no tokens."""
        return _LockstepTag.apply(hidden, self)


class StubLockstepAdapter(LockstepAdapter):
    """Drives ``RecordingLockstepMLP`` through the lockstep plan (``register_lockstep_module``):
    its group is ``module.group``, its null form "empty", its signature the params dtype with
    a grad-requiring input (no router)."""

    def groups(self, module):
        return [] if module.group is None else [module.group]

    def form(self, module):
        return FORM_EMPTY

    def config_signature(self, module):
        return LockstepSignature(hidden_dtype=module.config.params_dtype, hidden_requires_grad=True)

    def observe(self, module, on_call, on_signature):
        def before(mod, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            on_call(mod)
            on_signature(mod, LockstepSignature(hidden.dtype, hidden.requires_grad))

        return [module.register_forward_pre_hook(before, with_kwargs=True)]

    def null_forward(self, entry, hidden, signature):
        del signature
        return entry.module.null(hidden)

    def over_budget(self, module):
        return module.over_budget

    def reset_over_budget(self, module):
        module.over_budget.fill_(False)


def stub_moe_layer_spec():
    """The 'E' sublayer spec of the stub: pre-MLP RMSNorm + ``RecordingLockstepMLP`` + bda."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_layer import (
        TransformerLayer,
        TransformerLayerSubmodules,
    )

    local = get_gpt_layer_local_submodules(normalization="RMSNorm")
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            pre_mlp_layernorm=local.pre_mlp_layernorm,
            mlp=RecordingLockstepMLP,
            mlp_bda=local.mlp_bda,
        ),
    )


# ---------------------------------------------------------------------------
# Collective op logger
# ---------------------------------------------------------------------------


class CollectiveOpLogger(contextlib.AbstractContextManager):
    """Within the block, log every ``torch.distributed`` collective issued through
    ``all_to_all_single``, ``all_gather_into_tensor`` (also Megatron's captured
    ``mappings.dist_all_gather_func``), ``all_gather``, ``all_reduce`` and
    ``batch_isend_irecv`` as ``(op, group ranks, phase)`` with phase ``"bwd"`` inside an
    autograd backward, else ``"fwd"``. ``groups`` (sets of global ranks) keeps only ops on those
    groups. Tensor sizes are not logged (a null sends less than a real item)."""

    _NAMES = (
        "all_to_all_single",
        "all_gather_into_tensor",
        "all_gather",
        "all_reduce",
        "batch_isend_irecv",
    )

    def __init__(self, groups=None):
        self.ops: list = []
        self.groups = None if groups is None else {tuple(sorted(g)) for g in groups}
        self._saved: list = []

    def _record(self, name, group):
        ranks = tuple(
            sorted(
                dist.get_process_group_ranks(group)
                if group is not None
                else range(dist.get_world_size())
            )
        )
        if self.groups is not None and ranks not in self.groups:
            return
        phase = "bwd" if torch._C._current_autograd_node() is not None else "fwd"
        self.ops.append((name, ranks, phase))

    def _wrap(self, name, function):
        logger = self

        def wrapped(*args, **kwargs):
            if name == "batch_isend_irecv":
                ops = args[0] if args else kwargs["p2p_op_list"]
                group = ops[0].group if ops else None
            else:
                group = kwargs.get("group")
            logger._record(name, group)
            return function(*args, **kwargs)

        return wrapped

    def __enter__(self):
        from megatron.core.tensor_parallel import mappings

        for name in self._NAMES:
            original = getattr(dist, name)
            self._saved.append((dist, name, original))
            setattr(dist, name, self._wrap(name, original))
        original = mappings.dist_all_gather_func
        self._saved.append((mappings, "dist_all_gather_func", original))
        mappings.dist_all_gather_func = self._wrap("all_gather_into_tensor", original)
        return self

    def __exit__(self, *exc):
        for owner, name, original in reversed(self._saved):
            setattr(owner, name, original)
        self._saved = []
        return False
