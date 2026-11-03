# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The training wiring of a loop model with FSDP, on CPU:

* ``--megatron-fsdp-accumulate-main-grad`` parses, reaches the DDP config through
  ``get_megatron_ddp_config`` and needs ``--use-megatron-fsdp``;
* argument validation (``_validate_loop_fsdp``, run by ``validate_args`` after its
  Megatron-FSDP normalization) applies ``guards.validate_loop_fsdp_config`` to the argument
  namespace and refuses torch FSDP2;
* ``core_transformer_config_from_args`` accepts the hybrid loop with Megatron-FSDP v1 and still
  refuses torch FSDP2, Megatron-FSDP v2 and the GPT loop.

The rules themselves are tested in ``transformer/loop/test_loop_guards.py``.
"""

import inspect
import sys
from dataclasses import dataclass

import pytest
import torch

from megatron.core.transformer.loop import guards
from megatron.training.argument_utils import core_transformer_config_from_args
from megatron.training.arguments import _validate_loop_fsdp, parse_args, validate_args
from megatron.training.training import get_megatron_ddp_config

HYBRID_LOOP = dict(loop_architecture="huginn", loop_prelude_sublayers=2, loop_coda_sublayers=2)
GPT_LOOP = dict(loop_architecture="huginn", loop_prelude_layers=1, loop_core_layers=2)
FSDP_LOOP = dict(
    use_megatron_fsdp=True,
    megatron_fsdp_version=1,
    megatron_fsdp_accumulate_main_grad=True,
    megatron_fsdp_main_grads_dtype=torch.float32,
    megatron_fsdp_grad_comm_dtype=torch.float32,
)


def _parsed(monkeypatch, *flags, **overrides):
    monkeypatch.setattr(sys, "argv", ["test", *flags])
    args = parse_args()
    for name, value in overrides.items():
        assert hasattr(args, name), name
        setattr(args, name, value)
    return args


def _reported(excinfo):
    message = str(excinfo.value)
    assert message.startswith(guards.FSDP_PREFIX), message
    return message[len(guards.FSDP_PREFIX) :].split(", ")


@dataclass(init=False)
class _CapturingConfig:
    """Records the kwargs core_transformer_config_from_args builds (no config validation)."""

    loop_architecture: str = "none"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_accumulate_flag_parses_and_reaches_the_ddp_config(monkeypatch):
    assert _parsed(monkeypatch).megatron_fsdp_accumulate_main_grad is False
    args = _parsed(monkeypatch, "--use-megatron-fsdp", "--megatron-fsdp-accumulate-main-grad")
    assert args.megatron_fsdp_accumulate_main_grad is True
    ddp_config = get_megatron_ddp_config(args, use_torch_fsdp2=False)
    assert ddp_config.megatron_fsdp_accumulate_main_grad is True


def test_accumulate_flag_requires_megatron_fsdp(monkeypatch):
    args = _parsed(
        monkeypatch,
        "--megatron-fsdp-accumulate-main-grad",
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        max_position_embeddings=1024,
        seq_length=1024,
        micro_batch_size=1,
        train_iters=1,
        lr=1e-4,
        tokenizer_type="NullTokenizer",
        vocab_size=1024,
    )
    with pytest.raises(AssertionError, match="--megatron-fsdp-accumulate-main-grad requires"):
        validate_args(args)


def test_argument_validation_runs_the_loop_fsdp_rules(monkeypatch):
    # Without a loop nothing is checked.
    _validate_loop_fsdp(_parsed(monkeypatch, use_megatron_fsdp=True))
    _validate_loop_fsdp(_parsed(monkeypatch, use_torch_fsdp2=True))
    # The hybrid loop with Megatron-FSDP v1 in accumulate mode passes, with the dtypes mapped
    # (torch dtypes) or not yet (the CLI strings).
    _validate_loop_fsdp(_parsed(monkeypatch, **HYBRID_LOOP, **FSDP_LOOP))
    _validate_loop_fsdp(
        _parsed(
            monkeypatch,
            **HYBRID_LOOP,
            **dict(
                FSDP_LOOP,
                megatron_fsdp_main_grads_dtype="fp32",
                megatron_fsdp_grad_comm_dtype="fp32",
            ),
        )
    )
    for overrides, rules in (
        (dict(megatron_fsdp_accumulate_main_grad=False), [guards.FSDP_OVERWRITE_MAIN_GRAD]),
        (dict(megatron_fsdp_grad_comm_dtype="auto"), [guards.FSDP_GRAD_DTYPE]),
        (dict(megatron_fsdp_version=2), [guards.FSDP_VERSION]),
        (dict(nccl_ub=True), [guards.FSDP_PERSISTENT_BUFFERS]),
        (dict(save="/checkpoints"), [guards.FSDP_CHECKPOINTING]),
        (dict(load="/checkpoints"), [guards.FSDP_CHECKPOINTING]),
        (dict(loop_sample_depth=True), [guards.FSDP_SAMPLED_DEPTHS]),
        (dict(num_distributed_optimizer_instances=2), [guards.FSDP_HSDP]),
    ):
        args = _parsed(monkeypatch, **HYBRID_LOOP, **dict(FSDP_LOOP, **overrides))
        with pytest.raises(ValueError) as excinfo:
            _validate_loop_fsdp(args)
        assert _reported(excinfo) == rules, overrides
    with pytest.raises(ValueError) as excinfo:
        _validate_loop_fsdp(_parsed(monkeypatch, **GPT_LOOP, **FSDP_LOOP))
    assert _reported(excinfo) == [guards.FSDP_GPT_LOOP]
    with pytest.raises(ValueError, match="does not support torch FSDP2"):
        _validate_loop_fsdp(_parsed(monkeypatch, **HYBRID_LOOP, use_torch_fsdp2=True))


def test_validate_args_checks_the_loop_after_the_fsdp_normalization():
    """The rules read the mapped dtypes and the double buffering nccl_ub / the MaxPool
    allocator imply, which validate_args sets in its Megatron-FSDP block."""
    source = inspect.getsource(validate_args)
    call = source.index("_validate_loop_fsdp(args)")
    assert source.count("_validate_loop_fsdp(args)") == 1
    assert source.index("args.megatron_fsdp_grad_comm_dtype = map_dtype(") < call
    assert source.rindex("args.fsdp_double_buffer = True") < call


@pytest.mark.parametrize(
    "overrides,refused",
    [
        (dict(HYBRID_LOOP, use_megatron_fsdp=True), False),
        (dict(use_megatron_fsdp=True), False),
        (dict(use_torch_fsdp2=True), False),
        (dict(HYBRID_LOOP, use_torch_fsdp2=True), True),
        (dict(HYBRID_LOOP, use_megatron_fsdp=True, megatron_fsdp_version=2), True),
        (dict(GPT_LOOP, use_megatron_fsdp=True), True),
        (dict(GPT_LOOP, use_torch_fsdp2=True), True),
    ],
    ids=[
        "hybrid-mfsdp1",
        "no-loop-mfsdp",
        "no-loop-fsdp2",
        "hybrid-fsdp2",
        "hybrid-mfsdp2",
        "gpt-mfsdp1",
        "gpt-fsdp2",
    ],
)
def test_config_from_args_accepts_the_hybrid_loop_with_megatron_fsdp_v1(
    monkeypatch, overrides, refused
):
    args = _parsed(monkeypatch, **overrides)
    args.params_dtype = torch.float32  # validate_args derives it; there is no flag
    if refused:
        with pytest.raises(ValueError, match="not torch FSDP2, Megatron-FSDP v2 or the GPT loop"):
            core_transformer_config_from_args(args, config_class=_CapturingConfig)
    else:
        config = core_transformer_config_from_args(args, config_class=_CapturingConfig)
        assert config.loop_architecture == overrides.get("loop_architecture", "none")
