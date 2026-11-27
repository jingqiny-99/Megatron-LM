# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-FSDP accumulate mode with a weight used several times per forward (GPU).

A Transformer Engine Linear (the FSDP unit ``_Block``) is used k in {1, 4} times per forward,
one round of ``tanh(linear(state + injection)) * scale`` each, with and without one activation
checkpoint per use. Megatron-FSDP v1 ({optim_grads, optim_grads_params} x TE gradient
accumulation fusion {on, off} x ``megatron_fsdp_accumulate_main_grad`` {on, off}, fp32 main
and communication gradients) runs one training step against Megatron DDP (fp32 gradient
reduction), every rank on its own rows, at the torchrun world size (data parallel = W; run it
at W = 2 and W = 4):

* k = 4 with accumulate mode: every reduced fp32 gradient (the shards gathered) matches DDP's,
  relative difference <= 1e-6;
* k = 1: accumulate mode on and off give the same gradients, bitwise;
* k = 4 without accumulate mode, fusion on: the TE weight's gradient differs from DDP's (the
  defect: every use writes over the bucket and round 1's use survives). Its single-use weight
  and the non-fused scale still match;
* k = 4 without accumulate mode, fusion off: matches DDP (autograd sums the uses first).
"""

import itertools
import os

import pytest
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    import transformer_engine.pytorch as te

    HAVE_TE = True
except ImportError:  # pragma: no cover
    HAVE_TE = False

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_TE or WORLD < 2,
    reason="needs CUDA, Transformer Engine and torchrun with at least 2 ranks",
)

HIDDEN = 64
TOKENS = 32
MAX_RELATIVE = 1e-6
STRATEGIES = ("optim_grads", "optim_grads_params")


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


class _Block(torch.nn.Module):
    """The FSDP unit: a TE Linear (fused weight gradient per ``fusion``) and a plain scale."""

    def __init__(self, fusion):
        super().__init__()
        self.linear = te.Linear(
            HIDDEN, HIDDEN, bias=False, params_dtype=torch.bfloat16, fuse_wgrad_accumulation=fusion
        )
        self.scale = torch.nn.Parameter(torch.ones(HIDDEN, dtype=torch.bfloat16))

    def forward(self, x):
        return torch.tanh(self.linear(x)) * self.scale


class _Model(torch.nn.Module):
    """A single-use TE injection, then ``uses`` rounds of the block."""

    def __init__(self, fusion, uses, checkpoint_uses):
        super().__init__()
        self.inject = te.Linear(
            HIDDEN, HIDDEN, bias=False, params_dtype=torch.bfloat16, fuse_wgrad_accumulation=fusion
        )
        self.block = _Block(fusion)
        self.uses = uses
        self.checkpoint_uses = checkpoint_uses

    def forward(self, x):
        injection = self.inject(x)
        state = torch.zeros_like(injection)
        for _ in range(self.uses):
            if self.checkpoint_uses:
                state = checkpoint(self.block, state + injection, use_reentrant=False)
            else:
                state = self.block(state + injection)
        return state


def _model(fusion, uses, checkpoint_uses, state=None):
    torch.manual_seed(1234)
    model = _Model(fusion, uses, checkpoint_uses).cuda()
    with torch.no_grad():
        # Non-trivial scales (ones would hide a wrong scale gradient's sign or size).
        model.block.scale.copy_(torch.linspace(0.5, 1.5, HIDDEN, dtype=torch.bfloat16))
    if state is not None:
        model.load_state_dict(state)
    return model


def _inputs():
    """Every rank's own rows; the loss weights are shared."""
    generator = torch.Generator(device="cuda").manual_seed(4242 + dist.get_rank())
    x = torch.randn(TOKENS, HIDDEN, device="cuda", generator=generator).to(torch.bfloat16)
    weights = torch.linspace(-1.0, 1.0, TOKENS * HIDDEN, device="cuda").view(TOKENS, HIDDEN)
    return x, weights


def _config(fusion):
    return TransformerConfig(
        num_layers=1, hidden_size=HIDDEN, num_attention_heads=1, gradient_accumulation_fusion=fusion
    )


def _ddp_grads(fusion, uses, checkpoint_uses, state):
    model = _model(fusion, uses, checkpoint_uses, state)
    ddp = DistributedDataParallel(
        _config(fusion),
        DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, overlap_grad_reduce=False, use_distributed_optimizer=False
        ),
        model,
    )
    ddp.zero_grad_buffer()
    x, weights = _inputs()
    (ddp(x).float() * weights).sum().backward()
    ddp.finish_grad_sync()
    torch.cuda.synchronize()
    return {name: param.main_grad.float().clone() for name, param in model.named_parameters()}


def _fsdp_full_grads(fsdp):
    """Every parameter's reduced gradient: each group's gradient shards all-gathered over its
    data-parallel group (collective: every rank walks the groups in the same order)."""
    from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import Bucket

    buffer = fsdp.param_and_grad_buffer
    grads = {}
    for group in buffer.parameter_groups:
        gbuf = group.main_grad_buffer
        if gbuf is None:
            continue
        if gbuf.is_data_distributed:
            full = torch.empty(gbuf.bucket_index.size, dtype=gbuf.dtype, device="cuda")
            dist.all_gather_into_tensor(full, gbuf.data, group=gbuf.data_parallel_group)
        else:
            start = gbuf.bucket_index.global_data_index
            full = gbuf.data[start : start + gbuf.bucket_index.size]
        for param in group.params:
            item = gbuf.get_item_from_bucket(Bucket(data=full), gbuf.param_idx[param])
            grads[buffer.param_to_name[param]] = item.view(param.shape).float().clone()
    return grads


def _fsdp_grads(strategy, fusion, accumulate, uses, checkpoint_uses, state):
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel

    model = _model(fusion, uses, checkpoint_uses, state)
    fsdp = FullyShardedDataParallel(
        config=_config(fusion),
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            use_distributed_optimizer=True,
            data_parallel_sharding_strategy=strategy,
            megatron_fsdp_accumulate_main_grad=accumulate,
            megatron_fsdp_main_grads_dtype=torch.float32,
            megatron_fsdp_grad_comm_dtype=torch.float32,
        ),
        module=model,
        fsdp_unit_modules=[_Block],
    )
    fsdp.zero_grad_buffer()
    x, weights = _inputs()
    (fsdp(x).float() * weights).sum().backward()
    fsdp.finish_grad_sync()
    torch.cuda.synchronize()
    grads = _fsdp_full_grads(fsdp)
    fsdp.stop_communication()
    return grads


def _relative(actual, expected):
    return ((actual.double() - expected.double()).norm() / expected.double().norm()).item()


@pytest.fixture(scope="module")
def results():
    """Every arm, run once: DDP per (fusion, uses, checkpoint) and Megatron-FSDP per (strategy,
    fusion, accumulate, uses, checkpoint), from the same weights."""
    state = {
        name: value.clone() if torch.is_tensor(value) else value
        for name, value in _model(True, 1, False).state_dict().items()
    }
    ddp, fsdp = {}, {}
    for fusion, uses, checkpoint_uses in itertools.product((True, False), (1, 4), (False, True)):
        ddp[(fusion, uses, checkpoint_uses)] = _ddp_grads(fusion, uses, checkpoint_uses, state)
        for strategy, accumulate in itertools.product(STRATEGIES, (True, False)):
            key = (strategy, fusion, accumulate, uses, checkpoint_uses)
            fsdp[key] = _fsdp_grads(*key, state)
    return ddp, fsdp


def _tag(key):
    strategy, fusion, accumulate, uses, checkpoint_uses = key
    return (
        f"T4 W{WORLD} rank{dist.get_rank()} {strategy} fusion={fusion} "
        f"accumulate={accumulate} k={uses} ckpt={checkpoint_uses}"
    )


def test_accumulate_mode_matches_ddp_for_four_uses(results):
    ddp, fsdp = results
    for (strategy, fusion, accumulate, uses, checkpoint_uses), grads in fsdp.items():
        if not accumulate or uses != 4:
            continue
        reference = ddp[(fusion, uses, checkpoint_uses)]
        assert grads.keys() == reference.keys()
        rel = {name: _relative(grads[name], reference[name]) for name in reference}
        print(f"[{_tag((strategy, fusion, accumulate, uses, checkpoint_uses))}] rel {rel}")
        assert all(value <= MAX_RELATIVE for value in rel.values()), rel


def test_single_use_gradients_do_not_depend_on_accumulate_mode(results):
    ddp, fsdp = results
    for strategy, fusion, checkpoint_uses in itertools.product(
        STRATEGIES, (True, False), (False, True)
    ):
        on = fsdp[(strategy, fusion, True, 1, checkpoint_uses)]
        off = fsdp[(strategy, fusion, False, 1, checkpoint_uses)]
        reference = ddp[(fusion, 1, checkpoint_uses)]
        for name in reference:
            assert torch.equal(on[name].view(torch.int32), off[name].view(torch.int32)), name
            assert _relative(on[name], reference[name]) <= MAX_RELATIVE, name


def test_legacy_overwrite_loses_uses_with_fused_weight_gradients(results):
    """The documented defect: without accumulate mode a fused-wgrad weight used 4 times keeps
    one use's gradient; the single-use weight and the non-fused scale are unaffected."""
    ddp, fsdp = results
    for strategy, checkpoint_uses in itertools.product(STRATEGIES, (False, True)):
        key = (strategy, True, False, 4, checkpoint_uses)
        grads, reference = fsdp[key], ddp[(True, 4, checkpoint_uses)]
        rel = {name: _relative(grads[name], reference[name]) for name in reference}
        print(f"[{_tag(key)}] rel {rel}")
        assert rel["block.linear.weight"] > 0.1, rel
        assert rel["inject.weight"] <= MAX_RELATIVE, rel
        assert rel["block.scale"] <= MAX_RELATIVE, rel


def test_legacy_policy_is_right_without_fusion(results):
    ddp, fsdp = results
    for strategy, checkpoint_uses in itertools.product(STRATEGIES, (False, True)):
        grads = fsdp[(strategy, False, False, 4, checkpoint_uses)]
        reference = ddp[(False, 4, checkpoint_uses)]
        for name in reference:
            assert _relative(grads[name], reference[name]) <= MAX_RELATIVE, name
