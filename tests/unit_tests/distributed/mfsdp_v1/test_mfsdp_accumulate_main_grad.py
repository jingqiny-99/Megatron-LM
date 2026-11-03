# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-FSDP accumulate mode (``megatron_fsdp_accumulate_main_grad``), on CPU.

A parameter whose gradients are sharded is reduce-scattered once per microbatch from an
unsharded bucket allocated for that microbatch. By default Megatron-FSDP marks the parameter
``overwrite_main_grad`` at every unshard, so a fused weight-gradient kernel (Transformer
Engine's ``fuse_wgrad_accumulation``) writes each backward use over the bucket: a parameter used
several times per microbatch (the core of a loop model) keeps the gradient of its last-computed
use only. In accumulate mode the bucket is zeroed on its first fetch after each allocation and
every use adds into it.

The tests drive the real code without process groups or a device:

* the overwrite policy (``MegatronFSDP._overwrites_main_grad``) and its application at every
  unshard (``MegatronFSDP.all_gather_and_wait_parameters_ready``);
* fresh-bucket zeroing (``DataParallelBuffer.fetch_bucket`` / ``free_bucket_storage``) with an
  allocator that fills every new allocation with NaN, so a missing zero shows;
* the gradient hook's accumulation (``MegatronFSDP._accumulate_grad``);
* Megatron's own fused-wgrad linear (``tensor_parallel.layers``), which follows
  ``overwrite_main_grad`` for a Megatron-FSDP weight as TE does;
* a 4-round loop over one weight with a TE-style fused weight-gradient function (the contract of
  TE ``linear.py``: ``overwrite_main_grad`` read at forward time, ``get_main_grad()`` in the
  backward, copy or add, ``grad_added_to_main_grad`` set and a dummy gradient handed to
  autograd), with and without a per-round activation checkpoint, against plain fp32 autograd:
  accumulate mode gives the 4-round sum (rel <= 1e-6), the legacy policy keeps round 1 only (the
  defect, which the sum check catches), plain autograd gradients are right in both modes, and a
  single use gets the same gradient in both modes, bitwise.

The fake data-parallel group has 2 ranks (this process is rank 0), so the gradient buffer is a
true shard of a 2-shard bucket; the reduce-scatter is emulated by reading the bucket.
"""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed.fsdp.src.megatron_fsdp.distributed_data_parallel_config import (
    DistributedDataParallelConfig as FSDPConfig,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import MegatronFSDP
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    Bucket,
    DataParallelBuffer,
    TemporaryBucketAllocator,
)

ROUNDS = 4
HIDDEN = 16
TOKENS = 8
FAKE_DP_WORLD = 2
MAX_RELATIVE = 1e-6


@pytest.fixture
def fake_dp_world(monkeypatch):
    """DataParallelBuffer reads only the world size of its group (``dp_rank`` is given)."""
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: FAKE_DP_WORLD)


class _PoisoningAllocator(TemporaryBucketAllocator):
    """The default temporary allocator (one bucket per ID until freed), filling every new
    allocation with NaN and counting allocations."""

    def __init__(self):
        super().__init__()
        self.allocations = 0

    def allocate(
        self, bucket_id, size, dtype, device, mem_alloc_context=None, strict_assignments=True
    ):
        if bucket_id not in self.buckets:
            self.allocations += 1
            data = torch.full((size,), float("nan"), dtype=dtype, device=device)
            self.buckets[bucket_id] = Bucket(data=data)
        return self.buckets[bucket_id]


def _grad_buffer(params, *, zero_fresh_bucket, is_data_distributed=True, allocator=None):
    """A gradient buffer of ``params`` as ParamAndGradBuffer builds it (fp32, optim_grads)."""
    return DataParallelBuffer(
        FSDPConfig(data_parallel_sharding_strategy="optim_grads"),
        params,
        is_data_distributed=is_data_distributed,
        bucket_id=0,
        dtype=torch.float32,
        device=torch.device("cpu"),
        data_parallel_group=object(),
        dp_rank=0,
        temporary_bucket_allocator=_PoisoningAllocator() if allocator is None else allocator,
        zero_fresh_bucket=zero_fresh_bucket,
    )


def _attach_main_grad(params, gbuf):
    """Megatron-FSDP's main-gradient getter (``ParamAndGradBuffer``): the parameter's view of
    the fetched communication bucket (fp32, the gradient communication dtype here)."""
    for item_id, param in enumerate(params):

        def get_main_grad(param=param, item_id=item_id):
            bucket = gbuf.fetch_bucket(dtype=torch.float32)
            return gbuf.get_item_from_bucket(bucket, item_id).view(param.shape)

        param.get_main_grad = get_main_grad
        param.grad_added_to_main_grad = False


class _FakeFsdp:
    """The MegatronFSDP state the accumulate-mode methods read: one parameter group with the
    gradient buffer ``gbuf``, sharded gradients, no process groups and no device. The methods
    are MegatronFSDP's own."""

    _overwrites_main_grad = MegatronFSDP._overwrites_main_grad
    _accumulate_grad = MegatronFSDP._accumulate_grad
    all_gather_and_wait_parameters_ready = MegatronFSDP.all_gather_and_wait_parameters_ready

    def __init__(self, params, gbuf, *, accumulate, reduces_every_backward=True):
        self.accumulate_main_grad = accumulate
        self.report_nan_in_param_grad = False
        self._reduces = reduces_every_backward
        group = SimpleNamespace(requires_grad=True, hfsdp_helper_gbuf=None, main_grad_buffer=gbuf)
        self.param_and_grad_buffer = SimpleNamespace(
            param_to_param_group={param: 0 for param in params}, parameter_groups=[group]
        )
        self.ddp_config = FSDPConfig(data_parallel_sharding_strategy="optim_grads")
        self.all_gather_pipeline = SimpleNamespace(all_gather_params=lambda **kwargs: None)
        self.dist_index = SimpleNamespace(use_hybrid_fsdp=False)
        self.microbatch_count = 0
        self.model_auto_sync = False
        self.suggested_AG_prefetch_size = None

    def _reduces_grad_every_backward(self, param):
        del param
        return self._reduces

    def unshard(self, params):
        """The pre-forward / pre-backward unshard of an FSDP unit (no gather to wait for)."""
        self.all_gather_and_wait_parameters_ready(params, wait_bucket_ready=False)


class _FusedWgradLinear(torch.autograd.Function):
    """``y = x W^T`` with Transformer Engine's fused weight-gradient contract for a
    Megatron-FSDP parameter (TE ``linear.py``)."""

    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x)
        ctx.weight = weight
        # Read at forward time and baked into the wgrad GEMM's accumulate flag.
        ctx.overwrite = bool(getattr(weight, "overwrite_main_grad", False))
        ctx.main_grad_func = weight.get_main_grad
        return x @ weight.t()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        weight = ctx.weight
        main_grad = ctx.main_grad_func()
        wgrad = grad_output.reshape(-1, grad_output.shape[-1]).t() @ x.reshape(-1, x.shape[-1])
        if ctx.overwrite:  # accumulate=False
            main_grad.copy_(wgrad)
        else:  # accumulate=True
            main_grad.add_(wgrad)
        weight.grad_added_to_main_grad = True
        # TE hands autograd a cached dummy gradient (get_dummy_wgrad) whose values are unused.
        return grad_output @ weight, torch.zeros_like(weight)


class _Reducer:
    """The post-accumulate-grad hook of the weight: ``_accumulate_grad``, then the bucket's
    reduce-scatter (emulated: rank 0 reads the bucket) and its release."""

    def __init__(self, fake, gbuf, weight):
        self.fake = fake
        self.gbuf = gbuf
        self.fires = 0
        self.reduced = None
        weight.register_post_accumulate_grad_hook(self.hook)

    def hook(self, param):
        self.fires += 1
        self.fake._accumulate_grad(param)
        bucket = self.gbuf.fetch_bucket(dtype=torch.float32)
        self.reduced = self.gbuf.get_item_from_bucket(bucket, 0).view(param.shape).clone()
        self.gbuf.free_bucket_storage()
        self.gbuf.reset_param_main_grad()


def _loop_loss(weight, injection, *, rounds, fused, use_checkpoint, unshard=None):
    """``rounds`` uses of ``weight`` (one per round), optionally one checkpoint per round."""

    def use(x):
        if unshard is not None:
            unshard()  # the FSDP unit's pre-forward unshard (also in a recompute)
        out = _FusedWgradLinear.apply(x, weight) if fused else x @ weight.t()
        if unshard is not None and out.requires_grad:
            # The pre-backward unshard of this use.
            out.register_hook(lambda grad: (unshard(), grad)[1])
        return out

    def body(state, inj):
        return torch.tanh(use(state + inj))

    state = torch.zeros_like(injection)
    for _ in range(rounds):
        if use_checkpoint:
            state = checkpoint(body, state, injection, use_reentrant=False)
        else:
            state = body(state, injection)
    return (state * torch.linspace(0.5, 1.5, state.shape[-1])).sum()


def _inputs(seed):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(HIDDEN, HIDDEN, generator=generator) / HIDDEN**0.5
    injection = torch.randn(TOKENS, HIDDEN, generator=generator)
    return weight, injection


def _autograd_grad(weight0, injection, rounds):
    weight = torch.nn.Parameter(weight0.clone())
    _loop_loss(weight, injection, rounds=rounds, fused=False, use_checkpoint=False).backward()
    return weight.grad


def _round_one_grad(weight0, injection, rounds):
    """The gradient of round 1's use alone (what survives when every use overwrites)."""
    weights = [torch.nn.Parameter(weight0.clone()) for _ in range(rounds)]
    state = torch.zeros_like(injection)
    for weight in weights:
        state = torch.tanh((state + injection) @ weight.t())
    (state * torch.linspace(0.5, 1.5, state.shape[-1])).sum().backward()
    return weights[0].grad


def _fsdp_grads(weight0, injections, *, rounds, fused, use_checkpoint, accumulate):
    """One weight under the emulated Megatron-FSDP path; one backward per injection
    (microbatch). Returns the reduced gradient of every microbatch and the hook fire count."""
    weight = torch.nn.Parameter(weight0.clone())
    gbuf = _grad_buffer([weight], zero_fresh_bucket=accumulate)
    _attach_main_grad([weight], gbuf)
    fake = _FakeFsdp([weight], gbuf, accumulate=accumulate)
    reducer = _Reducer(fake, gbuf, weight)
    reduced = []
    for injection in injections:
        loss = _loop_loss(
            weight,
            injection,
            rounds=rounds,
            fused=fused,
            use_checkpoint=use_checkpoint,
            unshard=lambda: fake.unshard([weight]),
        )
        loss.backward()
        reduced.append(reducer.reduced)
    return reduced, reducer.fires


def _relative(actual, expected):
    return ((actual.double() - expected.double()).norm() / expected.double().norm()).item()


# ---------------------------------------------------------------------------------------------
# The overwrite policy
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reduces_every_backward,accumulate,overwrite",
    [(True, False, True), (True, True, False), (False, False, False), (False, True, False)],
)
def test_overwrite_policy_is_applied_at_every_unshard(
    fake_dp_world, reduces_every_backward, accumulate, overwrite
):
    """Legacy: a parameter reduced on every backward is overwritten. Accumulate mode: never.
    A parameter reduced once per optimization cycle (persistent buffer) always accumulates."""
    param = torch.nn.Parameter(torch.ones(4))
    gbuf = _grad_buffer([param], zero_fresh_bucket=accumulate)
    fake = _FakeFsdp(
        [param], gbuf, accumulate=accumulate, reduces_every_backward=reduces_every_backward
    )
    assert fake._overwrites_main_grad(param) is overwrite
    param.overwrite_main_grad = None
    param.grad_added_to_main_grad = True
    fake.unshard([param])
    assert param.overwrite_main_grad is overwrite
    assert param.grad_added_to_main_grad is False


# ---------------------------------------------------------------------------------------------
# Fresh-bucket zeroing
# ---------------------------------------------------------------------------------------------


def test_fresh_bucket_is_zeroed_once_per_allocation(fake_dp_world):
    params = [torch.nn.Parameter(torch.ones(3, 5)), torch.nn.Parameter(torch.ones(7))]
    allocator = _PoisoningAllocator()
    gbuf = _grad_buffer(params, zero_fresh_bucket=True, allocator=allocator)
    assert gbuf.data_size * FAKE_DP_WORLD == gbuf.bucket_index.size  # a shard of the bucket

    bucket = gbuf.fetch_bucket()
    assert allocator.allocations == 1
    assert bucket.data.numel() == gbuf.bucket_index.size
    assert torch.equal(bucket.data, torch.zeros_like(bucket.data))  # fresh: zeroed

    bucket.data.add_(2.0)  # the gradient of a first use
    live = gbuf.fetch_bucket()
    assert allocator.allocations == 1
    assert live.data.data_ptr() == bucket.data.data_ptr()
    assert torch.all(live.data == 2.0)  # live: handed back unchanged

    gbuf.free_bucket_storage()
    again = gbuf.fetch_bucket()
    assert allocator.allocations == 2
    assert torch.equal(again.data, torch.zeros_like(again.data))  # reallocated: zeroed again


def test_only_sharded_accumulate_buckets_are_zeroed(fake_dp_world):
    params = [torch.nn.Parameter(torch.ones(3, 5))]
    # The legacy buffer (and every weight buffer) hands out the allocation as it is.
    legacy = _grad_buffer(params, zero_fresh_bucket=False)
    assert torch.isnan(legacy.fetch_bucket().data).all()
    # An unsharded buffer returns its persistent data, which accumulates across microbatches.
    unsharded = _grad_buffer(params, zero_fresh_bucket=True, is_data_distributed=False)
    unsharded.init_data(torch.full((unsharded.data_size,), 3.0))
    assert torch.all(unsharded.fetch_bucket().data == 3.0)


def test_gradient_hook_adds_in_accumulate_mode_and_copies_by_default(fake_dp_world):
    for accumulate, with_grad, expected in (
        (True, True, 5.0),
        (False, True, 2.0),
        (True, False, 3.0),
        (False, False, 0.0),
    ):
        param = torch.nn.Parameter(torch.ones(6))
        gbuf = _grad_buffer([param], zero_fresh_bucket=accumulate)
        _attach_main_grad([param], gbuf)
        fake = _FakeFsdp([param], gbuf, accumulate=accumulate)
        param.get_main_grad().fill_(3.0)  # the live bucket already holds earlier uses
        if with_grad:
            param.grad = torch.full((6,), 2.0)
        fake._accumulate_grad(param)
        assert torch.all(param.main_grad == expected), (accumulate, with_grad)
        assert param.grad is None


# ---------------------------------------------------------------------------------------------
# A weight used once per round: the TE fused weight-gradient emulation
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("use_checkpoint", [False, True], ids=["eager", "checkpoint"])
@pytest.mark.parametrize("fused", [True, False], ids=["fused", "autograd"])
def test_accumulate_mode_sums_every_round(fake_dp_world, fused, use_checkpoint):
    weight0, injection = _inputs(1234)
    _, injection2 = _inputs(4321)
    reference = [_autograd_grad(weight0, inj, ROUNDS) for inj in (injection, injection2)]
    reduced, fires = _fsdp_grads(
        weight0,
        [injection, injection2],
        rounds=ROUNDS,
        fused=fused,
        use_checkpoint=use_checkpoint,
        accumulate=True,
    )
    # One reduce per backward; the second microbatch starts from a re-zeroed bucket.
    assert fires == 2
    for got, expected in zip(reduced, reference):
        assert torch.isfinite(got).all()
        assert _relative(got, expected) <= MAX_RELATIVE


@pytest.mark.parametrize("use_checkpoint", [False, True], ids=["eager", "checkpoint"])
def test_legacy_policy_keeps_round_one_only(fake_dp_world, use_checkpoint):
    """The defect: with fused weight gradients every use overwrites, and the last-computed use
    (round 1) survives. The 4-round sum check fails."""
    weight0, injection = _inputs(1234)
    reference = _autograd_grad(weight0, injection, ROUNDS)
    round_one = _round_one_grad(weight0, injection, ROUNDS)
    (got,), _ = _fsdp_grads(
        weight0,
        [injection],
        rounds=ROUNDS,
        fused=True,
        use_checkpoint=use_checkpoint,
        accumulate=False,
    )
    assert _relative(got, round_one) <= MAX_RELATIVE
    assert _relative(got, reference) > 0.1


def test_legacy_policy_is_right_without_fused_weight_gradients(fake_dp_world):
    """Autograd sums the rounds before the hook copies the sum: only fused wgrad is affected."""
    weight0, injection = _inputs(1234)
    reference = _autograd_grad(weight0, injection, ROUNDS)
    (got,), _ = _fsdp_grads(
        weight0, [injection], rounds=ROUNDS, fused=False, use_checkpoint=False, accumulate=False
    )
    assert _relative(got, reference) <= MAX_RELATIVE


def test_mhc_hybrid_wrapper_is_a_default_unit():
    """The mHC hybrid wrapper calls its inner TransformerLayer's sublayer methods directly, so
    the wrapper (the module the stack calls) is the FSDP unit; the EP-overlap list is kept."""
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import _get_default_fsdp_unit_modules
    from megatron.core.models.hybrid.layers.hybrid_hyper_connection import (
        HyperConnectionHybridLayer,
    )

    assert HyperConnectionHybridLayer in _get_default_fsdp_unit_modules(False)
    assert HyperConnectionHybridLayer not in _get_default_fsdp_unit_modules(True)


@pytest.mark.parametrize("overwrite", [True, False, None], ids=["overwrite", "add", "unset"])
def test_local_linear_fused_wgrad_follows_overwrite_main_grad(overwrite):
    """Megatron's own fused-wgrad linear (``tensor_parallel.layers``) treats a Megatron-FSDP
    weight as TE does: it writes over the bucket only under ``overwrite_main_grad`` (unset:
    the legacy overwrite). The weight has no ``grad_added_to_main_grad`` so that the backward
    returns no dummy gradient (that branch allocates on the current CUDA device)."""
    from megatron.core.tensor_parallel.layers import (
        linear_with_grad_accumulation_and_async_allreduce,
    )

    generator = torch.Generator().manual_seed(7)
    weight = torch.nn.Parameter(torch.randn(4, 3, generator=generator))
    inputs = torch.randn(5, 2, 3, generator=generator)
    bucket = torch.full((4, 3), 5.0)
    weight.__fsdp_param__ = True
    weight.get_main_grad = lambda: bucket
    if overwrite is not None:
        weight.overwrite_main_grad = overwrite
    out = linear_with_grad_accumulation_and_async_allreduce(
        inputs,
        weight,
        None,
        gradient_accumulation_fusion=True,
        allreduce_dgrad=False,
        sequence_parallel=False,
        tp_group=object(),
    )
    out.sum().backward()
    wgrad = torch.ones(5 * 2, 4).t() @ inputs.reshape(-1, 3)
    expected = wgrad if overwrite in (True, None) else 5.0 + wgrad
    assert torch.allclose(bucket, expected, rtol=1e-6, atol=1e-6)
    assert weight.grad is None


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "autograd"])
def test_single_use_gradients_do_not_depend_on_the_mode(fake_dp_world, fused):
    weight0, injection = _inputs(1234)
    grads = {}
    for accumulate in (False, True):
        (grads[accumulate],), _ = _fsdp_grads(
            weight0, [injection], rounds=1, fused=fused, use_checkpoint=False, accumulate=accumulate
        )
    assert torch.equal(grads[False].view(torch.int32), grads[True].view(torch.int32))
    assert _relative(grads[True], _autograd_grad(weight0, injection, 1)) <= MAX_RELATIVE
