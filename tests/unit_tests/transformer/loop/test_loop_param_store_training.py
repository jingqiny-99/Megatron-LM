# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU test of the loop parameter store's training wiring (``megatron/training/
loop_param_store_setup.py``) on the GTP-remat grid.

Runs under torchrun with CUDA and an even world size W >= 2: the grid with ``gtp_remat_size =
z`` (z = W / 2 for W >= 4, else W; replicate W / z), a local dense loop model built with the
collection ``loop_model_pg_collection`` returns (gtp axes nulled: no ``GTPShardedParam`` on TE
2.16), ``install_loop_param_store`` before the DDP wrap (``--loop-param-store-*`` arguments),
DDP + the distributed optimizer over the full collection, ``configure_loop_param_store`` (explicit
modes), the module RSBF schedule ``train()`` runs (``get_loop_rsbf_schedule()``, the default
collection, the loop-owned Gloo status group from ``loop_rsbf_status_gloo_group``) with the
config's ``finalize_model_grads``, two optimizer steps.

Gates: no parameter is a module-GTP shard before the install; the store's groups are the grid's
(shard = ``gtp_remat``, domain = the full data-parallel group, replicate = ``dp``); the status
group is a Gloo group over the full data-parallel ranks; step-1 per-sample losses bitwise equal
to the same steps without a store, step-2 losses close; the eager evaluation path
(``refresh_loop_param_store``, no-grad forward with windowed units) equals the store-free model's
eager losses bitwise at the initial weights; a torch_dist ``save_checkpoint`` of the trained
store run (DDP + distributed optimizer, both optimizer formats) reloads through
``load_checkpoint`` into fresh objects bitwise (canonical shards, fp32 mains, Adam moments), and
a legacy-format save raises message 1 (M3c; the full matrix is
``test_loop_checkpoint_store_gpu.py``).
"""

import os
from argparse import Namespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection, resolve_gtp_remat_group
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.loop.param_store import find_param_store
from megatron.core.transformer.loop.program import PrescribedDepthCriterion
from megatron.core.transformer.loop.rsbf_executor import (
    RSBF_BATCH_FUNC_ATTR,
    LoopMicrobatch,
    get_loop_rsbf_schedule,
)
from megatron.training import loop_param_store_setup as setup
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.loop import param_store_utils as psu
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    VOCAB,
    hybrid_loop_config,
    local_model_spec,
)

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
Z = WORLD // 2 if WORLD >= 4 else WORLD
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or WORLD < 2 or WORLD % 2,
    reason="the store training test needs CUDA and an even world size >= 2",
)

DENSE = "*-*-*-*-"
SEQ = 16
COUNT = 2  # samples per EU per step
DEPTHS = (2, 1)
MODES = "core=lw,edges=lw,emb_head=rsg"


@pytest.fixture(scope="module", autouse=True)
def _grid():
    Utils.initialize_model_parallel(1, 1, gtp_remat_size=Z)
    model_parallel_cuda_manual_seed(123)
    get_loop_rsbf_schedule().reset()
    setup.reset_loop_param_store_groups()
    try:
        yield
    finally:
        get_loop_rsbf_schedule().reset()
        setup.reset_loop_param_store_groups()
        Utils.destroy_model_parallel()


def _args(**overrides):
    values = dict(
        loop_param_store_shard_size=Z,
        loop_param_store_modes=MODES,
        loop_param_store_min_numel=1,
        loop_param_store_binding="data",
        loop_param_store_separate_rs_group=False,
        loop_param_store_debug="check,poison",
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
        loop_iterations=max(DEPTHS),
        loop_schedule="rsbf",
        calculate_per_token_loss=True,
        loop_rsbf_item_form="block",
        loop_rsbf_block_stride=2,
    )
    torch.manual_seed(1234)
    model = HybridModel(
        config=config,
        hybrid_stack_spec=local_model_spec(LoopHybridStack),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        hybrid_layer_pattern=DENSE,
        position_embedding_type="none",
        pg_collection=pgc,
    ).cuda()
    config.finalize_model_grads_func = finalize_model_grads
    return model


def _rows(rank, step):
    samples = psu.make_samples(rank, COUNT, vocab=VOCAB, seq=SEQ, seed=61 + step)
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


def _eager_losses(hybrid, rows):
    out = []
    with torch.no_grad():
        for row, depth in zip(rows, DEPTHS):
            hybrid.decoder.set_loop_depths(torch.tensor([depth], device="cuda"))
            output = hybrid(row["tokens"], row["position_ids"], None, labels=row["labels"])
            out.append(_loss_func(row["loss_mask"], output)[0])
    return torch.stack(out)


def _train(with_store):
    """Two optimizer steps through the module RSBF schedule; returns per-step losses, the eager
    losses at the initial weights and the store."""
    args = _args()
    full = ProcessGroupCollection.use_mpu_process_groups()
    # Both runs build on the grid with the gtp axes nulled (a module resolving the grid's
    # gtp_remat group would attach GTPShardedParam, which needs TE >= 2.19).
    hybrid = _build(setup.loop_model_pg_collection(full))
    assert not any(getattr(p, "is_gtp_weight_remat", False) for p in hybrid.parameters())
    store = setup.install_loop_param_store([hybrid], args, full) if with_store else None
    ddp = DistributedDataParallel(
        hybrid.config,
        DistributedDataParallelConfig(
            overlap_grad_reduce=False,
            use_distributed_optimizer=True,
            grad_reduce_in_fp32=True,
            average_in_collective=False,
        ),
        hybrid,
        pg_collection=full,
    )
    rank = dist.get_rank()
    if with_store:
        assert find_param_store(ddp) is store
        setup.configure_loop_param_store(args, [ddp])
        assert store.modes.describe().startswith("core=lw,prelude=lw")
        assert setup.refresh_loop_param_store([ddp])  # the embedding / head copies
    eager = _eager_losses(hybrid, _rows(rank, 0))
    if with_store:
        assert store.stats.eager_gathers > 0
    optimizer = get_megatron_optimizer(
        OptimizerConfig(optimizer="adam", lr=1e-3, use_distributed_optimizer=True, clip_grad=1.0),
        [ddp],
        use_gloo_process_groups=False,
        pg_collection=full,
    )
    schedule = get_loop_rsbf_schedule()
    schedule.reset()
    status = setup.loop_rsbf_status_gloo_group(args) if with_store else None
    schedule.configure(status_group=status)
    losses = []
    for step in range(2):
        ddp.zero_grad_buffer()
        optimizer.zero_grad()
        rows = _rows(rank, step)
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
            for row in rows
        ]
        reports = schedule(
            forward_step_func=_forward_step(),
            data_iterator=iter(microbatches),
            model=[ddp],
            num_microbatches=COUNT,
            seq_length=SEQ,
            micro_batch_size=1,
            exit_criterion=PrescribedDepthCriterion(depths),
        )
        losses.append(torch.stack([report["lm loss"][0] for report in reports]))
        success, _, _ = optimizer.step()
        assert success
    return dict(
        losses=losses,
        eager=eager,
        store=store,
        ddp=ddp,
        status=status,
        full=full,
        optimizer=optimizer,
    )


@pytest.fixture(scope="module")
def runs():
    return {"none": _train(False), "store": _train(True)}


def test_store_groups_and_status_group_are_the_grids(runs):
    run = runs["store"]
    full, store, status = run["full"], run["store"], run["status"]
    assert resolve_gtp_remat_group(setup.loop_model_pg_collection(full), is_expert=False) is None
    ranks = dist.get_process_group_ranks
    assert ranks(store.groups.shard) == ranks(full.gtp_remat) and len(ranks(full.gtp_remat)) == Z
    assert ranks(store.groups.domain) == ranks(full.dp_gtp_remat) == list(range(WORLD))
    assert ranks(store.groups.replicate) == ranks(full.dp)
    # Metrics (the indexer loss) average over every distinct-data rank, not the replicate dp.
    from megatron.training.training import metric_data_parallel_group

    model_pgc = setup.loop_model_pg_collection(full)
    assert ranks(metric_data_parallel_group(model_pgc)) == list(range(WORLD))
    assert status is not None and ranks(status) == list(range(WORLD))
    assert dist.get_backend(status) == "gloo"
    # The second call returns the cached group (no new group creation on later steps).
    assert setup.loop_rsbf_status_gloo_group(_args()) is status


def test_losses_match_the_storeless_run(runs):
    assert torch.equal(runs["store"]["losses"][0], runs["none"]["losses"][0])
    assert torch.allclose(runs["store"]["losses"][1], runs["none"]["losses"][1], rtol=1e-3)
    assert torch.equal(runs["store"]["eager"], runs["none"]["eager"])


@pytest.mark.parametrize("fully_reshardable", [False, True])
def test_checkpoint_save_load(runs, tmp_path_dist_ckpt, fully_reshardable):
    from megatron.training.loop_checkpoint import check_loop_save_format
    from tests.unit_tests.transformer.loop import store_ckpt_gpu_utils as cku

    run = runs["store"]
    with pytest.raises(RuntimeError, match=r"--ckpt-format torch with the loop parameter store"):
        check_loop_save_format([run["ddp"]], ckpt_format="torch")
    check_loop_save_format([runs["none"]["ddp"]], ckpt_format="torch")  # no store: silent
    expected_weights = cku.logical_weights(run["store"], run["ddp"])
    expected_states = cku.optimizer_states(run["optimizer"], run["ddp"])
    name = f"store_training_ckpt_{int(fully_reshardable)}"
    with cku.checkpoint_dir(tmp_path_dist_ckpt / name) as ckpt_dir:
        with cku.mock_checkpoint_args(
            ckpt_dir, fully_reshardable=fully_reshardable, hybrid_layer_pattern=DENSE
        ):
            cku.save(run["ddp"], run["optimizer"], run["full"], iteration=2)
            fresh = _fresh_store_objects()
            assert cku.load(fresh["ddp"], fresh["optimizer"], run["full"]) == 2
    assert cku.mismatches(cku.logical_weights(fresh["store"], fresh["ddp"]), expected_weights) == []
    states = cku.optimizer_states(fresh["optimizer"], fresh["ddp"])
    assert cku.mismatches(states, expected_states) == []
    assert cku.pad_rows_zero(fresh["store"])


def _fresh_store_objects():
    """A new store run's model, DDP and distributed optimizer (no step taken)."""
    args = _args()
    full = ProcessGroupCollection.use_mpu_process_groups()
    hybrid = _build(setup.loop_model_pg_collection(full))
    store = setup.install_loop_param_store([hybrid], args, full)
    ddp = DistributedDataParallel(
        hybrid.config,
        DistributedDataParallelConfig(
            overlap_grad_reduce=False,
            use_distributed_optimizer=True,
            grad_reduce_in_fp32=True,
            average_in_collective=False,
        ),
        hybrid,
        pg_collection=full,
    )
    setup.configure_loop_param_store(args, [ddp])
    optimizer = get_megatron_optimizer(
        OptimizerConfig(optimizer="adam", lr=1e-3, use_distributed_optimizer=True, clip_grad=1.0),
        [ddp],
        use_gloo_process_groups=False,
        pg_collection=full,
    )
    return dict(store=store, ddp=ddp, optimizer=optimizer, full=full)
