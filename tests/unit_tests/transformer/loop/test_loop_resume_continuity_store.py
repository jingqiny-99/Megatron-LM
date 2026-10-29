# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""In-process resume continuity of RSBF runs with the loop parameter store (GPU, M3c).

Runs under torchrun with CUDA at W = 8, 4 or 2 ranks on the store's GTP-remat grid (z = min(4,
W), replicate W / z), the local dense Huginn loop model of the store checkpoint tests
(``*-*-*-*-``, vocabulary 130 and fc1 182 rows: pad rows at z = 4; every 2-D dense weight stored,
``min_numel = 1``), trained through the module RSBF schedule with sampled R4-uniform depths
(counter-based on (seed, iteration, sample)), ``finalize_model_grads``, DDP (fp32 gradient
reduction) and the distributed optimizer, saved and loaded through ``save_checkpoint`` /
``load_checkpoint`` with the training's groups (``store_ckpt_gpu_utils``), with
``torch.use_deterministic_algorithms`` set once for the module.

Each arm trains an uninterrupted run for ``2 K`` steps and a resumed one (``K`` steps, save,
new model / store / DDP / optimizer objects with perturbed weights, scrambled RNG, the schedule
reset, load, steps ``K + 1 .. 2 K``). Gate: every resumed step's per-sample loss reports and the
grad norm **bitwise** equal to the uninterrupted run's:

* every store mode (Resident, Resident + sharded grads, Layer window, Layer window with the copy
  engines) x the whole-round, block (stride 2) and nested (stride 2) items, both
  distributed-optimizer formats on the Layer window;
* cross mode: saved in the Layer window, resumed in Resident + sharded grads (bitwise-equal
  gradient association) against the uninterrupted Layer-window run; saved Resident, resumed in
  the Layer window against the uninterrupted run that switches to the Layer window in process
  between steps ``K`` and ``K + 1``.

The DSV4-TINY store runs (EP = 8, TE specs) are the toolkit chains C5 - C9 of the stage-8
design (``scripts/loop_ckpt_chain.sh``); the cross-z resumes are C10 / C10r there and the CPU
cases in ``test_loop_resume_cpu.py``.
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
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.loop.rsbf_executor import (
    RSBF_BATCH_FUNC_ATTR,
    LoopMicrobatch,
    get_loop_rsbf_schedule,
)
from megatron.training import loop_param_store_setup as setup
from megatron.training import loop_rsbf_planning as planning
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.loop import param_store_utils as psu
from tests.unit_tests.transformer.loop import store_ckpt_gpu_utils as cku
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    determinism_env,
    hybrid_loop_config,
    local_model_spec,
)

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
Z = min(4, WORLD)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or WORLD < 2 or WORLD % Z,
    reason="the store resume GPU test needs CUDA and W in {2, 4, 8}",
)

DENSE = "*-*-*-*-"
VOCAB_ROWS = 130
SEQ = 16
COUNT = 2  # samples per EU per step
K = 3
MODES = {
    "r": ("all=r", "nccl"),
    "rsg": ("all=rsg", "nccl"),
    "lw": ("core=lw,edges=lw,emb_head=rsg", "nccl"),
    "lw_copy": ("core=lw,edges=lw,emb_head=rsg", "copy"),
}
FORMS = {
    "whole_round": {},
    "block2": dict(loop_rsbf_item_form="block", loop_rsbf_block_stride=2),
    "nested2": dict(loop_rsbf_item_form="nested", loop_rsbf_block_stride=2),
}


@pytest.fixture(scope="module", autouse=True)
def _grid():
    with determinism_env():
        yield from _grid_body()


def _grid_body():
    Utils.initialize_model_parallel(1, 1, gtp_remat_size=Z)
    model_parallel_cuda_manual_seed(123)
    torch.use_deterministic_algorithms(True, warn_only=True)
    _reset()
    setup.reset_loop_param_store_groups()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(False)
        _reset()
        setup.reset_loop_param_store_groups()
        Utils.destroy_model_parallel()


def _reset():
    get_loop_rsbf_schedule().reset()
    planning.reset_loop_planner_session()
    planning.reset_loop_exit_histogram_log()


def _args(mode):
    modes, engine = MODES[mode]
    return Namespace(
        loop_param_store_shard_size=Z,
        loop_param_store_modes=modes,
        loop_param_store_min_numel=1,
        loop_param_store_binding="data",
        loop_param_store_separate_rs_group=False,
        loop_param_store_debug="off",
        loop_param_store_gather_engine=engine,
        loop_param_store_reduce_engine=engine,
        loop_param_store_copy_engine_timeout_ms=0,
        loop_planner="off",
        use_gloo_process_groups=True,
    )


def _store_args(mode):
    """The ``loop_param_store_*`` arguments of ``mode`` (for the mock checkpoint arguments)."""
    return {k: v for k, v in vars(_args(mode)).items() if k.startswith("loop_param_store_")}


def _build(pgc, form, *, perturb=None):
    config = hybrid_loop_config(
        DENSE,
        2,
        2,
        ffn_hidden_size=91,
        loop_iterations=4,
        loop_schedule="rsbf",
        calculate_per_token_loss=True,
        loop_exit_criterion="sampled",
        loop_depth_probs=[0.25, 0.25, 0.25, 0.25],
        loop_depth_seed=1234,
        **FORMS[form],
    )
    torch.manual_seed(1234)
    model = HybridModel(
        config=config,
        hybrid_stack_spec=local_model_spec(LoopHybridStack),
        vocab_size=VOCAB_ROWS,
        max_sequence_length=SEQ,
        hybrid_layer_pattern=DENSE,
        position_embedding_type="none",
        pg_collection=pgc,
    ).cuda()
    if perturb is not None:
        with torch.no_grad():
            for index, (_, param) in enumerate(sorted(model.named_parameters())):
                generator = torch.Generator().manual_seed(perturb * 1000 + index)
                param.copy_(0.02 * torch.randn(param.shape, generator=generator).cuda())
    config.finalize_model_grads_func = finalize_model_grads
    return model


def _setup(mode, form, *, perturb=None):
    """A run's model with its store (installed before the DDP wrap), DDP, the distributed
    optimizer and the loop's Gloo status group."""
    args = _args(mode)
    full = ProcessGroupCollection.use_mpu_process_groups()
    hybrid = _build(setup.loop_model_pg_collection(full), form, perturb=perturb)
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
    status = setup.loop_rsbf_status_gloo_group(args)
    return dict(
        hybrid=hybrid,
        store=store,
        ddp=ddp,
        optimizer=optimizer,
        full=full,
        status=status,
        args=args,
    )


def _rows(iteration):
    rank = dist.get_rank()
    samples = psu.make_samples(rank, COUNT, vocab=VOCAB_ROWS, seq=SEQ, seed=300 + iteration)
    return [
        LoopMicrobatch(
            tokens=s.tokens.cuda(),
            position_ids=s.position_ids.cuda(),
            labels=s.labels.cuda(),
            loss_mask=torch.ones_like(s.labels, dtype=torch.float).cuda(),
            attention_mask=None,
            padding_mask=None,
            loss_func=_loss_func,
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


def _train(run, iterations, *, switch_at=None, switch_to=None):
    """One optimizer step per iteration; ``switch_at``: configure the ``switch_to`` modes in
    process before that iteration. Returns per step the loss reports and the grad norm."""
    schedule = get_loop_rsbf_schedule()
    schedule.configure(status_group=run["status"])
    records = []
    for iteration in iterations:
        if switch_at is not None and iteration == switch_at:
            run["args"].loop_param_store_modes = MODES[switch_to][0]
            setup.configure_loop_param_store(run["args"], [run["ddp"]])
        run["ddp"].zero_grad_buffer()
        run["optimizer"].zero_grad()
        batches = _rows(iteration)
        schedule.set_iteration(iteration)
        reports = schedule(
            forward_step_func=_forward_step(),
            data_iterator=iter(batches),
            model=[run["ddp"]],
            num_microbatches=COUNT,
            seq_length=SEQ,
            micro_batch_size=1,
        )
        success, grad_norm, _ = run["optimizer"].step()
        assert success
        records.append(
            dict(
                losses=[report["lm loss"].float().tolist() for report in reports],
                grad_norm=float(grad_norm),
            )
        )
    return records


def _resume(tmp_path, name, form, save_mode, load_mode, *, switch=False, fully_reshardable=False):
    _reset()
    reference = _setup(save_mode, form)
    expected = _train(
        reference, range(1, 2 * K + 1), switch_at=K + 1 if switch else None, switch_to=load_mode
    )
    del reference
    _reset()
    first = _setup(save_mode, form)
    early = _train(first, range(1, K + 1))
    with cku.checkpoint_dir(tmp_path / name) as ckpt_dir:
        with cku.mock_checkpoint_args(
            ckpt_dir,
            fully_reshardable=fully_reshardable,
            loop_config=first["hybrid"].config,
            hybrid_layer_pattern=DENSE,
            **_store_args(save_mode),
        ) as args:
            cku.save(first["ddp"], first["optimizer"], first["full"], iteration=K)
            del first
            _reset()
            resumed = _setup(load_mode, form, perturb=7)
            # The load runs with this run's store arguments (a mode change: a resume note).
            for key, value in _store_args(load_mode).items():
                setattr(args, key, value)
            torch.manual_seed(97531)
            model_parallel_cuda_manual_seed(97531)
            cku.poison_pad_rows(resumed["store"])
            assert cku.load(resumed["ddp"], resumed["optimizer"], resumed["full"]) == K
    assert cku.pad_rows_zero(resumed["store"])
    late = _train(resumed, range(K + 1, 2 * K + 1))
    assert early == expected[:K], "the run before the save is not deterministic in process"
    for offset, (got, want) in enumerate(zip(late, expected[K:])):
        iteration = K + 1 + offset
        assert got["losses"] == want["losses"], f"iteration {iteration}: losses differ"
        assert got["grad_norm"] == want["grad_norm"], f"iteration {iteration}: grad norm differs"


@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("mode", sorted(MODES))
def test_store_resume_continues_bitwise(tmp_path_dist_ckpt, mode, form):
    _resume(tmp_path_dist_ckpt, f"store_resume_{mode}_{form}", form, mode, mode)


def test_store_resume_fully_reshardable(tmp_path_dist_ckpt):
    _resume(
        tmp_path_dist_ckpt,
        "store_resume_lw_fully",
        "whole_round",
        "lw",
        "lw",
        fully_reshardable=True,
    )


def test_store_resume_layer_window_into_sharded_grads(tmp_path_dist_ckpt):
    """Layer window and Resident + sharded grads reduce gradients in the same association: the
    resumed RSG run continues the uninterrupted Layer-window run bitwise."""
    _resume(tmp_path_dist_ckpt, "store_resume_lw_to_rsg", "block2", "lw", "rsg")


def test_store_resume_resident_into_layer_window(tmp_path_dist_ckpt):
    """Saved Resident, resumed in the Layer window, against the uninterrupted run that switches
    to the Layer window in process before step ``K + 1``."""
    _resume(tmp_path_dist_ckpt, "store_resume_r_to_lw", "nested2", "r", "lw", switch=True)
