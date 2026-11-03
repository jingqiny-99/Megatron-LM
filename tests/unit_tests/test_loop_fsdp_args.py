# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The training wiring of a loop model with FSDP, on CPU:

* ``--megatron-fsdp-accumulate-main-grad`` parses, reaches the DDP config through
  ``get_megatron_ddp_config`` and needs ``--use-megatron-fsdp``;
* argument validation (``_validate_loop_fsdp``, run by ``validate_args`` after its
  Megatron-FSDP normalization) applies ``guards.validate_loop_fsdp_config`` to the argument
  namespace and refuses torch FSDP2; a model-only ``--load`` (``--finetune --no-load-optim``)
  passes;
* ``core_transformer_config_from_args`` accepts the hybrid loop with Megatron-FSDP v1 and still
  refuses torch FSDP2, Megatron-FSDP v2 and the GPT loop;
* the model-only load of a loop model on Megatron-FSDP runs before the wrap
  (``setup_model_and_optimizer``): ``_load_loop_model_before_megatron_fsdp_wrap`` calls
  ``load_checkpoint`` without an optimizer and with ``ckpt_format`` torch_dist for the call
  (restored afterwards, also on an error) and refuses a checkpoint that is not torch_dist;
* ``loop_optimizer_param_group_lines`` (the optimizer's parameter groups in the log of a loop
  run) prints the optimizer's own grouping with the wrapper prefixes stripped.

The rules themselves are tested in ``transformer/loop/test_loop_guards.py``; the load on GPU
(the weights reach Megatron-FSDP's buffers bitwise) in
``transformer/loop/test_loop_grad_gate_gpu.py``.
"""

import inspect
import os
import socket
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.loop import guards
from megatron.training import checkpointing, training
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


def test_argument_validation_admits_a_model_only_load(monkeypatch):
    """The experiment's G2 / G4 load (``--load <torch_dist> --finetune --no-load-optim
    --no-load-rng``) passes argument validation with Megatron-FSDP; any other load does not."""
    base = dict(HYBRID_LOOP, **FSDP_LOOP, load="/checkpoints")
    for overrides in (
        dict(finetune=True, no_load_optim=True),
        dict(finetune=True, no_load_optim=True, no_load_rng=True),
    ):
        _validate_loop_fsdp(_parsed(monkeypatch, **base, **overrides))
    for overrides in (
        dict(),
        dict(finetune=True),
        dict(no_load_optim=True),
        dict(finetune=True, no_load_optim=True, init_model_with_meta_device=True),
        dict(finetune=True, no_load_optim=True, save="/checkpoints/out"),
    ):
        with pytest.raises(ValueError) as excinfo:
            _validate_loop_fsdp(_parsed(monkeypatch, **base, **overrides))
        assert _reported(excinfo) == [guards.FSDP_CHECKPOINTING], overrides


def test_the_pre_wrap_load_applies_to_loop_models_on_megatron_fsdp_with_load():
    applies = training._loads_loop_model_before_megatron_fsdp_wrap
    base = dict(
        loop_architecture="huginn",
        use_megatron_fsdp=True,
        load="/checkpoints",
        moe_use_upcycling=False,
    )
    assert applies(SimpleNamespace(**base))
    for change in (
        dict(loop_architecture="none"),
        dict(use_megatron_fsdp=False),
        dict(load=None),
        dict(moe_use_upcycling=True),
    ):
        assert not applies(SimpleNamespace(**dict(base, **change))), change


def test_model_only_load_runs_load_checkpoint_in_torch_dist_before_the_wrap(monkeypatch):
    """The pre-wrap load reads the checkpoint type first, then calls ``load_checkpoint`` (no
    optimizer, no scheduler, training state restored) with ``ckpt_format`` torch_dist; the run's
    format is restored afterwards, also when the checkpoint is refused."""
    calls = []
    args = SimpleNamespace(ckpt_format="fsdp_dtensor", load="/checkpoints")

    def base(load_dir, run_args, rank0=False):
        calls.append(("base", load_dir, run_args.ckpt_format, rank0))
        return {}, "/checkpoints/iter_0000001", False, ckpt_type

    def load(model, optimizer, scheduler, **kwargs):
        calls.append(("load", args.ckpt_format, optimizer, scheduler, kwargs))
        return 0, 0.0

    monkeypatch.setattr(checkpointing, "_load_base_checkpoint", base)
    monkeypatch.setattr(training, "load_checkpoint", load)
    model = [torch.nn.Linear(2, 2)]
    for ckpt_type in (checkpointing.CheckpointType.GLOBAL, None):
        calls.clear()
        assert training._load_loop_model_before_megatron_fsdp_wrap(model, args) == (0, 0.0)
        assert args.ckpt_format == "fsdp_dtensor"
        assert calls[0] == ("base", "/checkpoints", "torch_dist", True)
        assert calls[1][:4] == ("load", "torch_dist", None, None)
        assert calls[1][4]["restore_training_state"] is True
    for ckpt_type in (
        checkpointing.CheckpointType.FSDP_DTENSOR,
        checkpointing.CheckpointType.TORCH_DCP,
        checkpointing.CheckpointType.LEGACY,
    ):
        calls.clear()
        with pytest.raises(RuntimeError, match="from a torch_dist checkpoint only"):
            training._load_loop_model_before_megatron_fsdp_wrap(model, args)
        assert args.ckpt_format == "fsdp_dtensor"
        assert [call[0] for call in calls] == ["base"]


def test_setup_loads_before_the_wrap_and_skips_the_second_load():
    """``setup_model_and_optimizer``: the config-container build takes the pre-wrap load path
    for a loop model on Megatron-FSDP with ``--load`` (after the loop parameter store's path,
    before the plain build), the legacy build refuses it, and the post-wrap load is skipped when
    the weights were loaded before the wrap."""
    source = inspect.getsource(training.setup_model_and_optimizer)
    store = source.index("return _build_loop_param_store_models(")
    pre_wrap = source.index("_build_loop_megatron_fsdp_models(")
    plain = source.index("return builder.build_distributed_models(")
    assert store < pre_wrap < plain
    assert source.count("_loads_loop_model_before_megatron_fsdp_wrap(args)") == 2
    skip = source.index("if loaded_before_wrap:")
    post_wrap = source.index("load_checkpoint(\n")
    assert skip < post_wrap
    assert "elif (\n        args.load is not None or args.pretrained_checkpoint is not None" in (
        source[skip:post_wrap]
    )


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def gloo():
    """A one-rank gloo default group (the parameter grouping gathers its keys over it)."""
    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
        dist.init_process_group("gloo", rank=0, world_size=1)
        created = True
    yield
    if created:
        dist.destroy_process_group()


class _Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 4)  # weight [4, 3] decays; bias [4] does not
        self.norm = torch.nn.LayerNorm(3, elementwise_affine=True, bias=False)  # [3]: no decay


class _Wrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


def test_optimizer_param_group_lines_strip_wrappers(gloo):
    """The lines name parameters without wrapper prefixes, so a DDP run (``module.module.``)
    and a Megatron-FSDP run (``module.module.module.``) of one model print identical lines; the
    grouping is the optimizer's (no weight decay on biases and 1-D parameters)."""
    from megatron.core.optimizer import OptimizerConfig

    config = OptimizerConfig(optimizer="adam", lr=1e-3, weight_decay=0.1)
    inner = _Inner()
    ddp_like = training.loop_optimizer_param_group_lines([_Wrapper(_Wrapper(inner))], config, None)
    fsdp_like = training.loop_optimizer_param_group_lines(
        [_Wrapper(_Wrapper(_Wrapper(inner)))], config, None
    )
    assert ddp_like == fsdp_like
    assert ddp_like[0].startswith(
        " > optimizer parameter group 0: wd_mult 1.0, lr_mult 1.0, is_expert_parallel False, "
        "parameters 1, elements 12, names sha256 "
    )
    assert ddp_like[1] == " > optimizer parameter group 0 names: proj.weight"
    assert ddp_like[2].startswith(
        " > optimizer parameter group 1: wd_mult 0.0, lr_mult 1.0, is_expert_parallel False, "
        "parameters 2, elements 7, names sha256 "
    )
    assert ddp_like[3] == " > optimizer parameter group 1 names: norm.weight proj.bias"
    assert len(ddp_like) == 4
