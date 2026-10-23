# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for ``schedules.set_auxiliary_loss_scales`` (the aux-loss scale block of
``forward_step_calc_loss``, moved verbatim so the loop RSBF executor can set the scales once
per step): the MoE aux-loss, MTP and attention-variant (DSA indexer) scales for every
combination of per-token loss and scale functions, with the values the pre-refactor block
set written out; ``device_fn`` is evaluated only by a branch that builds a scale."""

import itertools
from types import SimpleNamespace

import pytest
import torch

from megatron.core.pipeline_parallel import schedules
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

NUM_MICROBATCHES = 4
CP = 2


@pytest.fixture(autouse=True)
def _fresh_scalers(monkeypatch):
    for scaler in (MoEAuxLossAutoScaler, MTPLossAutoScaler, DSAIndexerLossAutoScaler):
        monkeypatch.setattr(scaler, "main_loss_backward_scale", None)


def _config(*, moe, mtp, indexer, per_token, grad_scale, moe_func, mtp_func):
    return SimpleNamespace(
        num_moe_experts=8 if moe else None,
        mtp_num_layers=1 if mtp else None,
        experimental_attention_variant="dsv4_hybrid" if indexer else None,
        calculate_per_token_loss=per_token,
        grad_scale_func=(lambda x: x * 8.0) if grad_scale else None,
        moe_grad_scale_func=(lambda: torch.tensor(3.0)) if moe_func else None,
        mtp_grad_scale_func=(lambda: torch.tensor(5.0)) if mtp_func else None,
    )


def _expected(config):
    """The pre-refactor block's values (sched:355-399 at 8015a0f4b), written out."""
    base = 8.0 if config.grad_scale_func is not None else 1.0
    divisor = 1.0 if config.calculate_per_token_loss else NUM_MICROBATCHES
    cp = 1.0 if config.calculate_per_token_loss else CP
    moe = 3.0 if config.moe_grad_scale_func is not None else base
    mtp = 5.0 if config.mtp_grad_scale_func is not None else base
    return {
        MoEAuxLossAutoScaler: moe * cp / divisor if config.num_moe_experts else None,
        MTPLossAutoScaler: mtp / divisor if config.mtp_num_layers else None,
        DSAIndexerLossAutoScaler: (
            base * cp / divisor if config.experimental_attention_variant else None
        ),
    }


CASES = list(itertools.product([False, True], repeat=7))


@pytest.mark.parametrize(
    "moe,mtp,indexer,per_token,grad_scale,moe_func,mtp_func",
    CASES,
    ids=["-".join(str(int(flag)) for flag in case) for case in CASES],
)
def test_scales_match_the_pre_refactor_block(
    moe, mtp, indexer, per_token, grad_scale, moe_func, mtp_func
):
    config = _config(
        moe=moe,
        mtp=mtp,
        indexer=indexer,
        per_token=per_token,
        grad_scale=grad_scale,
        moe_func=moe_func,
        mtp_func=mtp_func,
    )
    calls = []

    def device_fn():
        calls.append(1)
        return torch.device("cpu")

    schedules.set_auxiliary_loss_scales(
        config, device_fn, num_microbatches=NUM_MICROBATCHES, cp_group_size=CP
    )
    for scaler, expected in _expected(config).items():
        if expected is None:
            assert scaler.main_loss_backward_scale is None, scaler.__name__
        else:
            value = scaler.main_loss_backward_scale
            assert value.numel() == 1 and value.item() == pytest.approx(expected), scaler.__name__
    assert len(calls) == int(moe) + int(mtp) + int(indexer)


def test_device_fn_is_not_evaluated_without_auxiliary_losses():
    config = _config(
        moe=False,
        mtp=False,
        indexer=False,
        per_token=True,
        grad_scale=True,
        moe_func=False,
        mtp_func=False,
    )

    def device_fn():
        raise AssertionError("device_fn must not be called")

    schedules.set_auxiliary_loss_scales(config, device_fn, num_microbatches=1, cp_group_size=None)


def test_forward_step_calc_loss_sets_the_same_scales():
    """The schedule's call site resolves the device from the loss lazily, as before."""
    config = _config(
        moe=True,
        mtp=True,
        indexer=True,
        per_token=False,
        grad_scale=True,
        moe_func=False,
        mtp_func=True,
    )
    config.timers = None
    store = []
    loss = torch.tensor(6.0)
    output, num_tokens = schedules.forward_step_calc_loss(
        SimpleNamespace(),
        torch.zeros(3),
        lambda output: (loss.clone(), torch.tensor(3, dtype=torch.int), {"lm loss": loss}),
        config,
        None,
        False,
        NUM_MICROBATCHES,
        store,
        cp_group_size=CP,
        is_last_stage=True,
    )
    assert output.item() == pytest.approx(6.0 / 3 / NUM_MICROBATCHES) and num_tokens.item() == 3
    for scaler, expected in _expected(config).items():
        assert scaler.main_loss_backward_scale.item() == pytest.approx(expected), scaler.__name__
