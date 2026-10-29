# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU checkpoints of the loop parameter store with DDP and the distributed optimizer (M3c).

Runs under torchrun with CUDA at W = 8, 4 or 2 ranks on the store's GTP-remat grid (z = 4 at
W >= 8, else 2; replicate W / z), a local dense Huginn loop model (``*-*-*-*-``, vocabulary 130
and fc1 182 rows: rows not divisible by z; plus ``CkptStack``'s raw ``[1, h]`` / ``[3, h]``
parameters on sublayer 2, which leave all-padding shard ranks at z = 2 and 4), every 2-D dense
weight stored (``min_numel = 1``),
trained through the module RSBF schedule (prescribed depths, block items of stride 2) with
``finalize_model_grads`` and Adam, saved and loaded through ``save_checkpoint`` /
``load_checkpoint`` with the training's groups (``store_ckpt_gpu_utils``).

* Mode pairs (save -> load) over Resident, Resident + sharded grads, Layer window and Layer
  window with the copy engines, in both distributed-optimizer formats (``dp_reshardable``,
  ``fully_reshardable``): after the load into fresh objects the logical weights, fp32 mains and
  Adam moments equal the saved run's bitwise (per rank, and gathered by logical name), shard
  pad rows (poisoned before the load) are 0, the compute copies are
  unbound until ``refresh()`` / the next step; one more step of the loaded and of the
  uninterrupted run gives bitwise equal losses, and bitwise equal weights when both modes reduce
  gradients the same way (LW / RSG with the NCCL engine), else close.
* ``fully_reshardable`` across a layout change (a store-free z = 1 checkpoint into a store run,
  the reverse, another stored set by ``min_numel``): weights, fp32 mains and Adam moments
  bitwise by logical name over the whole world, the next step's losses bitwise, the weights
  after it close.
* The generic (non-distributed) optimizer; ``--async-save``; ``--ckpt-fully-parallel-load``.
* ``--optimizer-cpu-offload`` (``HybridDeviceOptimizer``, fraction 1.0, precision-aware), both
  formats: the stage-8 S8.0 (b) question (supported, or refused with message 9 on evidence).
* ``dp_reshardable`` into a run with a different stored set (``min_numel``): message 3 on every
  rank; the weights still load with ``--no-load-optim``.

The CPU counterparts (cross-z, all-padding shards, the load hooks, the generic optimizer path) are
``test_loop_checkpoint_store_gloo.py``.
"""

import os
from argparse import Namespace

import pytest
import torch
import torch.distributed as dist
from torch import nn

from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.loop.param_store_binding import is_placeholder
from megatron.core.transformer.loop.program import PrescribedDepthCriterion
from megatron.core.transformer.loop.rsbf_executor import (
    RSBF_BATCH_FUNC_ATTR,
    LoopMicrobatch,
    get_loop_rsbf_schedule,
)
from megatron.training import loop_param_store_setup as setup
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.loop import param_store_utils as psu
from tests.unit_tests.transformer.loop import store_ckpt_gpu_utils as cku
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    determinism_env,
    hybrid_loop_config,
    local_model_spec,
)

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
Z = 4 if WORLD >= 8 else 2
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or WORLD < 2 or WORLD % Z,
    reason="the store checkpoint GPU test needs CUDA and W in {2, 4, 8}",
)

DENSE = "*-*-*-*-"
VOCAB_ROWS = 130
SEQ = 16
COUNT = 2  # samples per EU per step
DEPTHS = (2, 1)
MODES = {
    "r": ("all=r", "nccl"),
    "rsg": ("all=rsg", "nccl"),
    "lw": ("core=lw,edges=lw,emb_head=rsg", "nccl"),
    "lw_copy": ("core=lw,edges=lw,emb_head=rsg", "copy"),
}
# Modes whose gradients are bitwise equal (the same reduce-scatter association).
GRAD_CLASS = {"r": "resident", "rsg": "sharded-nccl", "lw": "sharded-nccl", "lw_copy": "copy"}
PAIRS = [("lw", "rsg"), ("rsg", "r"), ("r", "lw"), ("lw_copy", "lw"), ("lw", "lw_copy")]


class CkptStack(LoopHybridStack):
    """The loop stack with two short raw parameters on core sublayer 2, ``rows1 [1, h]`` and
    ``rows3 [3, h]``, whose rows leave whole shards empty (all-padding ranks: ``rows1`` on shard
    ranks 1..z-1, ``rows3`` on shard rank 3 at z = 4), so the 0-row row-shard entries go through
    every save / load path here, the fully parallel load's exchange included. They train: the
    sublayer's output gets ``1e-3 * (rows1.sum() + rows3.sum())``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = self.config.hidden_size
        device = self.loop_adapter.weight.device
        layer = self.layers[2]
        layer.rows1 = nn.Parameter(torch.randn(1, hidden, device=device))
        layer.rows3 = nn.Parameter(torch.randn(3, hidden, device=device))

    def _run_layer(self, l_no, hidden_states, ctx):
        out = super()._run_layer(l_no, hidden_states, ctx)
        if l_no == 2:
            layer = self.layers[2]
            out = out + 1e-3 * (layer.rows1.sum() + layer.rows3.sum())
        return out


@pytest.fixture(scope="module", autouse=True)
def _grid():
    with determinism_env():
        yield from _grid_body()


def _grid_body():
    Utils.initialize_model_parallel(1, 1, gtp_remat_size=Z)
    model_parallel_cuda_manual_seed(123)
    torch.use_deterministic_algorithms(True, warn_only=True)
    get_loop_rsbf_schedule().reset()
    setup.reset_loop_param_store_groups()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(False)
        get_loop_rsbf_schedule().reset()
        setup.reset_loop_param_store_groups()
        Utils.destroy_model_parallel()


def _args(mode="lw", **overrides):
    modes, engine = MODES[mode]
    values = dict(
        loop_param_store_shard_size=Z,
        loop_param_store_modes=modes,
        loop_param_store_min_numel=1,
        loop_param_store_binding="data",
        loop_param_store_separate_rs_group=False,
        loop_param_store_debug="off",
        loop_param_store_gather_engine=engine,
        loop_param_store_reduce_engine=engine,
        loop_planner="off",
        use_gloo_process_groups=True,
    )
    values.update(overrides)
    return Namespace(**values)


def _build(pgc):
    config = hybrid_loop_config(
        DENSE,
        2,
        2,
        ffn_hidden_size=91,
        loop_iterations=max(DEPTHS),
        loop_schedule="rsbf",
        calculate_per_token_loss=True,
        loop_rsbf_item_form="block",
        loop_rsbf_block_stride=2,
    )
    torch.manual_seed(1234)
    model = HybridModel(
        config=config,
        hybrid_stack_spec=local_model_spec(CkptStack),
        vocab_size=VOCAB_ROWS,
        max_sequence_length=SEQ,
        hybrid_layer_pattern=DENSE,
        position_embedding_type="none",
        pg_collection=pgc,
    ).cuda()
    config.finalize_model_grads_func = finalize_model_grads
    return model


def _init_states(optimizer):
    """Seed the (generic) optimizer's state as ``tests/unit_tests/dist_checkpointing`` does."""
    for child in getattr(optimizer, "chained_optimizers", None) or [optimizer]:
        init = getattr(child, "init_state_fn", None)
        if init is not None:
            init(child.optimizer, child.config)


def _setup(mode="lw", *, store=True, dist_opt=True, hdo=False, **arg_overrides):
    """A run's model (+ store), DDP and optimizer on the grid, nothing trained."""
    args = _args(mode, **arg_overrides)
    full = ProcessGroupCollection.use_mpu_process_groups()
    hybrid = _build(setup.loop_model_pg_collection(full))
    the_store = setup.install_loop_param_store([hybrid], args, full) if store else None
    ddp = DistributedDataParallel(
        hybrid.config,
        DistributedDataParallelConfig(
            overlap_grad_reduce=False,
            use_distributed_optimizer=dist_opt,
            grad_reduce_in_fp32=True,
            average_in_collective=False,
        ),
        hybrid,
        pg_collection=full,
    )
    if store:
        setup.configure_loop_param_store(args, [ddp])
    extra = {}
    if hdo:
        extra = dict(
            optimizer_cpu_offload=True,
            optimizer_offload_fraction=1.0,
            use_precision_aware_optimizer=True,
        )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam", lr=1e-3, use_distributed_optimizer=dist_opt, clip_grad=1.0, **extra
        ),
        [ddp],
        use_gloo_process_groups=False,
        pg_collection=full,
    )
    if not dist_opt:
        _init_states(optimizer)  # a load template needs the generic optimizer's state
    status = setup.loop_rsbf_status_gloo_group(args) if store else None
    return dict(
        hybrid=hybrid, store=the_store, ddp=ddp, optimizer=optimizer, full=full, status=status
    )


def _rows(rank, step):
    samples = psu.make_samples(rank, COUNT, vocab=VOCAB_ROWS, seq=SEQ, seed=71 + step)
    return [
        dict(
            tokens=s.tokens.cuda(),
            labels=s.labels.cuda(),
            position_ids=s.position_ids.cuda(),
            loss_mask=torch.ones_like(s.labels, dtype=torch.float).cuda(),
        )
        for s in samples
    ]


def _loss_func(loss_mask, output):
    losses = output.view(-1).float()
    mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * mask)
    num_tokens = mask.sum().clone().detach().to(torch.int)
    return (
        loss,
        num_tokens,
        {"lm loss": torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])},
    )


def _forward_step():
    def forward_step(*_args, **_kwargs):  # pragma: no cover - the RSBF schedule calls the batch
        raise AssertionError("the RSBF schedule calls the batch function")

    setattr(forward_step, RSBF_BATCH_FUNC_ATTR, lambda iterator, _model: next(iterator))
    return forward_step


def _train(run, steps):
    """Optimizer steps ``steps`` (indices: the data of each step) through the module RSBF
    schedule; returns the per-step per-sample losses."""
    schedule = get_loop_rsbf_schedule()
    schedule.reset()
    schedule.configure(status_group=run["status"])
    rank = dist.get_rank()
    losses = []
    for step in steps:
        run["ddp"].zero_grad_buffer()
        run["optimizer"].zero_grad()
        depths = {g: DEPTHS[g % COUNT] for g in range(COUNT * WORLD)}
        microbatches = [
            LoopMicrobatch(
                tokens=row["tokens"],
                position_ids=row["position_ids"],
                labels=row["labels"],
                loss_mask=row["loss_mask"],
                attention_mask=None,
                padding_mask=None,
                loss_func=_loss_func,
            )
            for row in _rows(rank, step)
        ]
        reports = schedule(
            forward_step_func=_forward_step(),
            data_iterator=iter(microbatches),
            model=[run["ddp"]],
            num_microbatches=COUNT,
            seq_length=SEQ,
            micro_batch_size=1,
            exit_criterion=PrescribedDepthCriterion(depths),
        )
        losses.append(torch.stack([report["lm loss"][0] for report in reports]))
        success, _, _ = run["optimizer"].step()
        assert success
    return losses


def _logical_states(run):
    """The run's optimizer state by logical name; asserts no uncovered / conflicting element
    and zero pad rows."""
    states, problems = cku.logical_optimizer_states(run["optimizer"], run["ddp"], run["store"])
    assert problems == []
    return states


def _snapshot(run):
    return dict(
        weights=cku.logical_weights(run["store"], run["ddp"]),
        states=cku.optimizer_states(run["optimizer"], run["ddp"]),
        logical=_logical_states(run),
    )


def _close(actual, expected):
    return sorted(
        name
        for name, tensor in expected.items()
        if not torch.allclose(actual[name], tensor, rtol=1e-3, atol=1e-5)
    )


def _round_trip(tmp_path, name, save_mode, load_mode, *, fully_reshardable, **kwargs):
    """Train a run 2 steps in ``save_mode``, save, load into fresh ``load_mode`` objects; returns
    (saved run, loaded run, the saved snapshot)."""
    saved = _setup(save_mode, dist_opt=kwargs.get("dist_opt", True), hdo=kwargs.get("hdo", False))
    _train(saved, (0, 1))
    snapshot = _snapshot(saved)
    with cku.checkpoint_dir(tmp_path / name) as ckpt_dir:
        with cku.mock_checkpoint_args(
            ckpt_dir,
            fully_reshardable=fully_reshardable,
            async_save=kwargs.get("async_save", False),
            fully_parallel=kwargs.get("fully_parallel", False),
            dist_opt=kwargs.get("dist_opt", True),
            hybrid_layer_pattern=DENSE,
        ):
            cku.save(saved["ddp"], saved["optimizer"], saved["full"], iteration=2)
            loaded = _setup(
                load_mode, dist_opt=kwargs.get("dist_opt", True), hdo=kwargs.get("hdo", False)
            )
            # Bound (pre-load) Resident / RSG compute copies: the load must unbind them.
            setup.refresh_loop_param_store([loaded["ddp"]])
            # Poisoned pad rows (also the fp32 mains' that alias them): the load must zero them.
            cku.poison_pad_rows(loaded["store"])
            assert cku.load(loaded["ddp"], loaded["optimizer"], loaded["full"]) == 2
    return saved, loaded, snapshot


def _check_loaded(saved, loaded, snapshot, save_mode, load_mode):
    store = loaded["store"]
    assert cku.mismatches(cku.logical_weights(store, loaded["ddp"]), snapshot["weights"]) == []
    assert (
        cku.mismatches(cku.optimizer_states(loaded["optimizer"], loaded["ddp"]), snapshot["states"])
        == []
    )
    assert cku.mismatches(_logical_states(loaded), snapshot["logical"]) == []
    assert cku.pad_rows_zero(store)
    # No stale compute copy: unbound until the next step (or refresh()).
    assert all(t.bound is None and is_placeholder(t.compute.data) for t in store.tensors())
    # One more step of each: the forward sees the same weights, the losses are bitwise equal.
    expected = _train(saved, (2,))[0]
    actual = _train(loaded, (2,))[0]
    assert torch.equal(actual, expected)
    after_saved = cku.logical_weights(saved["store"], saved["ddp"])
    after_loaded = cku.logical_weights(store, loaded["ddp"])
    if GRAD_CLASS[save_mode] == GRAD_CLASS[load_mode]:
        assert cku.mismatches(after_loaded, after_saved) == []
    else:
        assert _close(after_loaded, after_saved) == []


def test_logical_keys_equal_the_store_free_keys():
    """The store saves under the store-free model's keys and global shapes (z-independent);
    each rank's row shard sits at ``shard_rank * ceil(R / z)``."""
    from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

    plain = _setup(store=False)["hybrid"].sharded_state_dict()
    run = _setup("lw")
    sharded = run["hybrid"].sharded_state_dict()
    assert sorted(sharded) == sorted(plain)
    stored = {id(t.shard): t for t in run["store"].tensors()}
    assert len(stored) > 0
    converted = 0
    for key, entry in sharded.items():
        tensor = stored.get(id(getattr(entry, "data", None)))
        if tensor is None:
            continue
        converted += 1
        assert isinstance(entry, ShardedTensorFactory)
        piece = entry.build_fn(entry.key, entry.data, entry.replica_id, None)
        assert isinstance(piece, ShardedTensor) and piece.key == plain[key].key
        assert tuple(piece.global_shape) == tensor.logical_shape
        per = tensor.rows_padded // Z
        assert piece.global_offset[0] == min(run["store"].shard_rank * per, tensor.rows)
    assert converted == len(stored)


@pytest.mark.parametrize("fully_reshardable", [False, True], ids=["dp_reshardable", "fully"])
@pytest.mark.parametrize("save_mode,load_mode", PAIRS)
def test_save_load_across_modes(tmp_path_dist_ckpt, save_mode, load_mode, fully_reshardable):
    name = f"store_modes_{save_mode}_{load_mode}_{int(fully_reshardable)}"
    saved, loaded, snapshot = _round_trip(
        tmp_path_dist_ckpt, name, save_mode, load_mode, fully_reshardable=fully_reshardable
    )
    _check_loaded(saved, loaded, snapshot, save_mode, load_mode)


@pytest.mark.parametrize(
    "arm",
    [
        dict(async_save=True),
        dict(fully_parallel=True),
        dict(fully_parallel=True, fully_reshardable=True),
    ],
    ids=["async_save", "fully_parallel_load", "fully_parallel_load_fully"],
)
def test_save_load_arms(tmp_path_dist_ckpt, arm):
    arm = dict(arm)
    fully = arm.pop("fully_reshardable", False)
    name = "store_arm_" + "_".join(sorted(arm)) + f"_{int(fully)}"
    saved, loaded, snapshot = _round_trip(
        tmp_path_dist_ckpt, name, "lw", "lw", fully_reshardable=fully, **arm
    )
    _check_loaded(saved, loaded, snapshot, "lw", "lw")


def test_generic_optimizer(tmp_path_dist_ckpt):
    saved, loaded, snapshot = _round_trip(
        tmp_path_dist_ckpt,
        "store_generic_optimizer",
        "rsg",
        "lw",
        fully_reshardable=False,
        dist_opt=False,
    )
    _check_loaded(saved, loaded, snapshot, "rsg", "lw")


@pytest.mark.parametrize("fully_reshardable", [False, True], ids=["dp_reshardable", "fully"])
def test_optimizer_cpu_offload(tmp_path_dist_ckpt, fully_reshardable):
    """The S8.0 (b) arm: HybridDeviceOptimizer with the store, saved and loaded."""
    pytest.importorskip("megatron.core.optimizer.cpu_offloading.hybrid_optimizer")
    saved, loaded, snapshot = _round_trip(
        tmp_path_dist_ckpt,
        f"store_hdo_{int(fully_reshardable)}",
        "lw",
        "lw",
        fully_reshardable=fully_reshardable,
        hdo=True,
    )
    _check_loaded(saved, loaded, snapshot, "lw", "lw")


LAYOUTS = {
    # (save run, load run): _setup keyword arguments.
    "plain_to_store": (dict(store=False), dict()),
    "store_to_plain": (dict(), dict(store=False)),
    "min_numel": (dict(), dict(loop_param_store_min_numel=8192)),
}


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_fully_reshardable_across_layouts(tmp_path_dist_ckpt, layout):
    """``fully_reshardable`` across a layout change: a store-free (z = 1) checkpoint into a store
    run, the reverse, and another stored set at the same z (``min_numel``). The logical weights,
    fp32 mains and Adam moments of the loaded run equal the saved run's bitwise (gathered over
    the whole world by logical name), pad rows 0; one more step gives bitwise equal losses
    (they depend on the weights only) and close weights (the two layouts reduce gradients in
    another association), which also exercises the loaded moments and step."""
    save_kwargs, load_kwargs = LAYOUTS[layout]
    saved = _setup("lw", **save_kwargs)
    _train(saved, (0, 1))
    weights = cku.logical_weights(saved["store"], saved["ddp"])
    states = _logical_states(saved)
    with cku.checkpoint_dir(tmp_path_dist_ckpt / f"store_layout_{layout}") as ckpt_dir:
        with cku.mock_checkpoint_args(ckpt_dir, fully_reshardable=True, hybrid_layer_pattern=DENSE):
            cku.save(saved["ddp"], saved["optimizer"], saved["full"], iteration=2)
            loaded = _setup("lw", **load_kwargs)
            if loaded["store"] is not None:
                assert sum(1 for _ in loaded["store"].tensors()) > 0
                cku.poison_pad_rows(loaded["store"])
            assert cku.load(loaded["ddp"], loaded["optimizer"], loaded["full"]) == 2
    assert cku.mismatches(cku.logical_weights(loaded["store"], loaded["ddp"]), weights) == []
    assert cku.mismatches(_logical_states(loaded), states) == []
    if loaded["store"] is not None:
        assert cku.pad_rows_zero(loaded["store"])
    assert torch.equal(_train(loaded, (2,))[0], _train(saved, (2,))[0])
    after_saved = cku.logical_weights(saved["store"], saved["ddp"])
    after_loaded = cku.logical_weights(loaded["store"], loaded["ddp"])
    assert _close(after_loaded, after_saved) == []


def test_dp_reshardable_with_another_stored_set_raises(tmp_path_dist_ckpt):
    saved = _setup("lw")
    _train(saved, (0,))
    weights = cku.logical_weights(saved["store"], saved["ddp"])
    with cku.checkpoint_dir(tmp_path_dist_ckpt / "store_message3") as ckpt_dir:
        with cku.mock_checkpoint_args(ckpt_dir, hybrid_layer_pattern=DENSE):
            cku.save(saved["ddp"], saved["optimizer"], saved["full"], iteration=1)
            other = _setup("lw", loop_param_store_min_numel=8192)
            assert sum(1 for _ in other["store"].tensors()) < sum(
                1 for _ in saved["store"].tensors()
            )
            message = r"dp_reshardable \(bucket\) format, whose layout depends on the loop"
            with pytest.raises(RuntimeError, match=message):
                cku.load(other["ddp"], other["optimizer"], other["full"])
        with cku.mock_checkpoint_args(ckpt_dir, no_load_optim=True, hybrid_layer_pattern=DENSE):
            weights_only = _setup("lw", loop_param_store_min_numel=8192)
            cku.load(weights_only["ddp"], weights_only["optimizer"], weights_only["full"])
    got = cku.logical_weights(weights_only["store"], weights_only["ddp"])
    assert cku.mismatches(got, weights) == []
