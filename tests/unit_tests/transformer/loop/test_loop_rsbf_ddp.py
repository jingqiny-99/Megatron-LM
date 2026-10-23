# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DDP tests of the multi-EU RSBF executor (GPU, every DP rank one EU of the domain = the DP
group; run with WORLD_SIZE = 2, 4 and 8: the M2 "W EUs against one EU" GPU gates).

Each DP rank loads its slice of one fixed global batch and runs it through RSBF under
``DistributedDataParallel`` (``overlap_grad_reduce=False``, fp32 ``main_grad``), with
prescribed depths for every sample of the global batch (known-depth criteria must cover the
domain's wave), chosen so the ranks' work differs: rebalancing moves samples between ranks
(NCCL ``batch_isend_irecv``), and without it the ranks run different numbers of items.
Per-sample losses equal the single-rank reference's bitwise. After finalize, ``main_grad``
must equal

* a single-rank RSBF reference: the same model under DDP over a one-rank DP group running the
  whole global batch (the gradient scale does not depend on the DP size: the token normalizer
  receives each rank's local token count and finalize all-reduces it once), and
* the standard schedule (``forward_backward_no_pipelining``, eager ``LoopHybridStack`` at the
  same per-sample depths, MBS 1) under the same DDP wrap,

at rtol 1e-5 with an absolute floor of 1e-6 x max |g| (fp32; the executor sums per-round
gradients in a different association than the eager graph). The distributed optimizer's
reduce-scatter is checked by gathering the reduced shards back.

* D1 + D2 drive ``LoopRSBFExecutor.run_step`` directly (local layers), with rebalancing (moves
  happen) and without (item counts differ between ranks).
* D4: waves 2 and 4 against one pass (per-sample losses bitwise, gradients within the gate).
* D5: the sync budget at W > 1: known depths under ``set_sync_debug_mode('error')`` (one host
  sync, no status exchange); an online criterion with the Gloo status group (one status
  exchange per forward round, one host sync per round plus one).
* D6: a buggy host plan (the same on every rank) is raised by every rank naming the EU that
  ran the bad sweep; the next step runs (no poisoned communicator).
* D3 drives the path ``train()`` uses: the module schedule (``get_loop_rsbf_schedule()``)
  with ``pg_collection=None`` (the default collection), a ``forward_step_func`` carrying the
  batch-function and loss-validator hooks, the executor cache over two steps and finalize
  through the schedule; with local layers and with the TE spec and
  ``gradient_accumulation_fusion=True`` (TE adds wgrads straight into ``main_grad`` and marks
  ``grad_added_to_main_grad``; RSBF runs many backward calls per parameter per step).
"""

import copy
import os
from functools import partial

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.distributed.param_and_grad_buffer import shard_buffer
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import layers as tensor_parallel_layers
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.loop.dispatcher import RSBFGroups
from megatron.core.transformer.loop.program import LoopProgram, PrescribedDepthCriterion
from megatron.core.transformer.loop.rsbf_executor import (
    RSBF_BATCH_FUNC_ATTR,
    RSBF_LOSS_VALIDATOR_ATTR,
    SWEEP_AFTER_EXIT,
    LoopMicrobatch,
    LoopRSBFExecutor,
    get_loop_rsbf_schedule,
)
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.loop import hybrid_parity_utils
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    SEQ,
    VOCAB,
    build_local_loop_model,
    hybrid_loop_config,
)
from tests.unit_tests.transformer.loop.rsbf_test_criteria import (
    HaltingHead,
    OnlineHaltingCriterion,
    retire_late,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="DDP RSBF tests need CUDA and at least two ranks",
)

full_precision_gemms = hybrid_parity_utils.full_precision_gemms

DENSE = "*-*-*-*-"
R_MAX = 4
PER_RANK = 4  # microbatches (samples, MBS 1) per rank
RSBF = dict(loop_iterations=R_MAX, loop_schedule="rsbf", calculate_per_token_loss=True)


def _te_fused_wgrad_available():
    """TE layers and the output layer's fused wgrad kernel (fused_weight_gradient_mlp_cuda)."""
    try:
        return is_te_min_version("1.7.0") and tensor_parallel_layers._grad_accum_fusion_available
    except Exception:  # pylint: disable=broad-except
        return False


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    Utils.initialize_model_parallel(1, 1)
    try:
        yield
    finally:
        get_loop_rsbf_schedule().reset()
        Utils.destroy_model_parallel()


@pytest.fixture(scope="module")
def single_rank_groups():
    """A one-rank process-group collection per rank (DDP and finalize over this rank only)."""
    world = dist.get_world_size()
    groups = [dist.new_group([rank]) for rank in range(world)]
    own = groups[dist.get_rank()]
    pg_collection = copy.copy(ProcessGroupCollection.use_mpu_process_groups())
    for name in ("dp", "dp_cp", "dp_cp_gtp_remat", "expt_dp", "tp_dp_cp"):
        setattr(pg_collection, name, own)
    return pg_collection


def _depth(global_index):
    # Rank q of W holds samples j * W + q: their depth sums differ between ranks.
    return 1 + global_index % R_MAX


def _global_batch(world):
    """The fixed global batch: PER_RANK * world samples, as one-row dicts by global index."""
    total = PER_RANK * world
    generator = torch.Generator().manual_seed(17)
    tokens = torch.randint(0, VOCAB, (total, SEQ), generator=generator)
    labels = torch.randint(0, VOCAB, (total, SEQ), generator=generator)
    loss_mask = (torch.rand(total, SEQ, generator=generator) > 0.2).float()
    return [
        dict(
            tokens=tokens[i : i + 1].cuda(),
            labels=labels[i : i + 1].cuda(),
            position_ids=torch.arange(SEQ)[None].cuda(),
            loss_mask=loss_mask[i : i + 1].cuda(),
            depth=_depth(i),
        )
        for i in range(total)
    ]


def _local_slice(batch):
    """This rank's samples as (global index, sample): the pretraining sampler's slices."""
    world, rank = dist.get_world_size(), dist.get_rank()
    return [(j * world + rank, batch[j * world + rank]) for j in range(PER_RANK)]


def _loss_func(loss_mask, output):
    """``pretrain_hybrid.loss_func``'s default branch (per-token sum, token count, report)."""
    losses = output.view(-1).float()
    mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * mask)
    num_tokens = mask.sum().clone().detach().to(torch.int)
    return (
        loss,
        num_tokens,
        {"lm loss": torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])},
    )


def _build_model(te):
    """The dense loop model: local layers, or the TE spec with fused wgrad accumulation."""
    if not te:
        return build_local_loop_model(DENSE, **RSBF)
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_stack_spec

    config = hybrid_loop_config(
        DENSE,
        2,
        2,
        gradient_accumulation_fusion=True,
        attention_backend=AttnBackend.unfused,
        **RSBF,
    )
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    return HybridModel(
        config=config,
        hybrid_stack_spec=loop_hybrid_stack_spec,
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        hybrid_layer_pattern=DENSE,
        position_embedding_type="none",
    ).cuda()


def _ddp_model(pg_collection=None, *, distributed_optimizer=False, te=False):
    model = _build_model(te)
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=False,
        use_distributed_optimizer=distributed_optimizer,
        grad_reduce_in_fp32=True,
    )
    ddp = DistributedDataParallel(model.config, ddp_config, model, pg_collection=pg_collection)
    model.config.finalize_model_grads_func = finalize_model_grads
    ddp.zero_grad_buffer()
    return ddp


def _main_grads(ddp):
    return {
        name: param.main_grad.detach().clone()
        for name, param in ddp.module.named_parameters()
        if param.requires_grad
    }


def _gather_reduced_shards(ddp):
    """After a distributed-optimizer reduce-scatter only the local shard of each bucket is
    reduced; gather the shards back so every main_grad holds the reduced gradient."""
    for buffer in ddp.buffers:
        group = buffer.data_parallel_group
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        for bucket in buffer.buckets:
            shards = shard_buffer(bucket.grad_data, size)
            dist.all_gather_into_tensor(bucket.grad_data, shards[rank], group=group)


def _microbatch(sample):
    return LoopMicrobatch(
        tokens=sample["tokens"],
        position_ids=sample["position_ids"],
        labels=sample["labels"],
        loss_mask=sample["loss_mask"],
        attention_mask=None,
        padding_mask=None,
        loss_func=_loss_func,
    )


def _spy_finalize(config, record):
    """Wrap ``config.finalize_model_grads_func`` to record the token count it receives;
    returns the real function (restore it afterwards)."""
    real = config.finalize_model_grads_func

    def spy(models, num_tokens, **kwargs):
        record.append(num_tokens.clone())
        return real(models, num_tokens, **kwargs)

    config.finalize_model_grads_func = spy
    return real


def _criterion(batch):
    """Prescribed depths of every sample of the global batch (the domain's wave)."""
    return PrescribedDepthCriterion({index: sample["depth"] for index, sample in enumerate(batch)})


def _rsbf_step(
    ddp, samples, pg_collection, batch, record=None, criterion=None, executor=None, **kwargs
):
    """One RSBF step of ``samples``; a new program and executor unless ``executor`` is given
    (reused, as ``train()`` reuses its cached executor across steps)."""
    if executor is None:
        program = LoopProgram(ddp, exit_criterion=criterion or _criterion(batch))
        executor = LoopRSBFExecutor(program, pg_collection=pg_collection, **kwargs)
    microbatches = iter(_microbatch(sample) for _, sample in samples)
    config = ddp.module.config
    real = _spy_finalize(config, record) if record is not None else None
    try:
        reports = executor.run_step(
            lambda iterator, model: next(iterator), microbatches, len(samples), model=ddp
        )
    finally:
        if real is not None:
            config.finalize_model_grads_func = real
    return reports, executor


def _losses(reports, executor):
    """{global index: the sample's 'lm loss' report} of this rank's (home) samples."""
    return {
        record.key.global_index: report["lm loss"]
        for record, report in zip(executor.stats.loss_records, reports)
    }


def _single_rank_reference(batch, single_rank_groups, *, te=False, with_losses=False):
    """RSBF over the whole global batch under DDP over a one-rank group (one EU)."""
    reference = _ddp_model(single_rank_groups, te=te)
    tokens = []
    reports, executor = _rsbf_step(
        reference, list(enumerate(batch)), single_rank_groups, batch, record=tokens
    )
    assert executor.num_eus == 1
    assert tokens[0].item() == sum(int(sample["loss_mask"].sum()) for sample in batch)
    if with_losses:
        return _main_grads(reference), _losses(reports, executor)
    return _main_grads(reference)


def _assert_losses_equal(losses, reference):
    assert losses
    for index, loss in losses.items():
        assert torch.equal(loss, reference[index]), (index, loss, reference[index])


def _eager_reference(local, *, distributed_optimizer=False, te=False):
    """The standard schedule (eager loop stack at the same per-sample depths, MBS 1) on this
    rank's slice, under the same DDP wrap."""
    eager = _ddp_model(distributed_optimizer=distributed_optimizer, te=te)
    hybrid = eager.module
    rows = iter(sample for _, sample in local)

    def forward_step(data_iterator, model):
        sample = next(data_iterator)
        hybrid.decoder.set_loop_depths(torch.tensor([sample["depth"]], device="cuda"))
        output = model(
            sample["tokens"],
            sample["position_ids"],
            None,
            labels=sample["labels"],
            loss_mask=sample["loss_mask"],
        )
        return output, partial(_loss_func, sample["loss_mask"])

    forward_backward_no_pipelining(
        forward_step_func=forward_step,
        data_iterator=rows,
        model=[eager],
        num_microbatches=PER_RANK,
        seq_length=SEQ,
        micro_batch_size=1,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    if distributed_optimizer:
        _gather_reduced_shards(eager)
    return _main_grads(eager)


def _assert_close(actual, expected):
    assert actual.keys() == expected.keys()
    for name, grad in expected.items():
        torch.testing.assert_close(
            actual[name],
            grad,
            rtol=1e-5,
            atol=1e-6 * grad.abs().max().item(),
            msg=lambda text, name=name: f"{name}: {text}",
        )


def _assert_item_counts_differ(executor):
    world = dist.get_world_size()
    items = torch.tensor([len(executor.stats.item_order)], device="cuda")
    all_items = [torch.zeros_like(items) for _ in range(world)]
    dist.all_gather(all_items, items)
    assert len({int(count) for count in all_items}) > 1, "ranks should run different item counts"


@pytest.mark.parametrize("move_transport", ["p2p", "all_to_all"])
@pytest.mark.parametrize("rebalance", [True, False], ids=["rebalance", "no_rebalance"])
@pytest.mark.parametrize("distributed_optimizer", [False, True])
def test_dp_ranks_with_different_item_counts_match_the_references(
    single_rank_groups, distributed_optimizer, rebalance, move_transport
):
    """D1 + D2 (G1): W EUs over the DP group; with rebalancing samples move between ranks,
    without it (and without tail item moves) the ranks run different item counts. Per-sample
    losses equal the single-rank RSBF reference bitwise; after finalize main_grad equals the
    single-rank reference over the whole global batch and the standard schedule under the same
    DDP wrap; finalize receives the local (home) token count; one host sync per step. Both
    move transports: NCCL ``batch_isend_irecv`` on the non-empty pairs, and
    ``all_to_all_single`` on every rank in every move phase (zero splits for the pairs and
    ranks without a transfer)."""
    batch = _global_batch(dist.get_world_size())
    local = _local_slice(batch)

    # The DP arm: each rank one EU of the domain, loading its slice.
    ddp = _ddp_model(distributed_optimizer=distributed_optimizer)
    seen_tokens = []
    reports, executor = _rsbf_step(
        ddp,
        local,
        ProcessGroupCollection.use_mpu_process_groups(),
        batch,
        record=seen_tokens,
        rebalance=rebalance,
        tail_item_moves=rebalance,
        move_transport=move_transport,
    )
    if distributed_optimizer:
        _gather_reduced_shards(ddp)
    dp_grads = _main_grads(ddp)
    stats = executor.stats
    assert executor.num_eus == dist.get_world_size()
    assert stats.host_syncs == 1 and stats.status_exchanges == 0
    if rebalance:
        assert stats.moves["sample"] > 0
        # all_to_all runs on every rank in every move phase; p2p only posts on the pairs.
        transport = executor.dispatcher.stats
        if move_transport == "all_to_all":
            assert transport.all_to_alls > 0 and transport.p2p_ops == 0
        else:
            assert transport.all_to_alls == 0
    else:
        assert stats.moves == {"sample": 0, "chain": 0, "item": 0}
        _assert_item_counts_differ(executor)
    local_tokens = sum(int(sample["loss_mask"].sum()) for _, sample in local)
    assert len(seen_tokens) == 1 and seen_tokens[0].item() == local_tokens
    assert sum(int(report["lm loss"][1]) for report in reports) == local_tokens
    assert sorted(_losses(reports, executor)) == [index for index, _ in local]

    reference_grads, reference_losses = _single_rank_reference(
        batch, single_rank_groups, with_losses=True
    )
    _assert_losses_equal(_losses(reports, executor), reference_losses)
    _assert_close(dp_grads, reference_grads)
    _assert_close(dp_grads, _eager_reference(local, distributed_optimizer=distributed_optimizer))


@pytest.mark.parametrize(
    "te",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not _te_fused_wgrad_available(),
                reason="needs TE and the fused wgrad accumulation kernel",
            ),
        ),
    ],
    ids=["local", "te_fused_wgrad"],
)
def test_train_path_schedule_at_dp(single_rank_groups, full_precision_gemms, te):
    """D3: the module schedule as ``train()`` calls it (``pg_collection=None``, the
    forward_step hooks, the cached executor over two steps): every step's main_grad equals the
    single-rank RSBF reference and the standard schedule; finalize receives the local token
    count once per step and the validator the mean per-microbatch loss sum."""
    del full_precision_gemms
    batch = _global_batch(dist.get_world_size())
    local = _local_slice(batch)
    local_tokens = sum(int(sample["loss_mask"].sum()) for _, sample in local)
    single_rank = _single_rank_reference(batch, single_rank_groups, te=te)
    eager = _eager_reference(local, te=te)

    ddp = _ddp_model(te=te)
    criterion = _criterion(batch)
    validated = []

    def forward_step(data_iterator, model):  # pragma: no cover - the schedule uses the hooks
        raise AssertionError("the RSBF schedule must not call forward_step itself")

    setattr(forward_step, RSBF_BATCH_FUNC_ATTR, lambda iterator, model: _microbatch(next(iterator)))
    setattr(forward_step, RSBF_LOSS_VALIDATOR_ATTR, validated.append)

    schedule = get_loop_rsbf_schedule()
    seen_tokens = []
    real = _spy_finalize(ddp.module.config, seen_tokens)
    executors = []
    try:
        for step in range(2):
            ddp.zero_grad_buffer()
            schedule.set_iteration(step)
            reports = schedule(
                forward_step_func=forward_step,
                data_iterator=[iter(sample for _, sample in local)],
                model=[ddp],
                num_microbatches=PER_RANK,
                seq_length=SEQ,
                micro_batch_size=1,
                decoder_seq_length=SEQ,
                forward_only=False,
                adjust_tensor_shapes_fn=None,
                force_all_reduce=False,
                p2p_communicator=None,
                pg_collection=None,
                exit_criterion=criterion,
            )
            executors.append(schedule.cached_executor(ddp))
            assert len(seen_tokens) == step + 1 and seen_tokens[-1].item() == local_tokens
            assert sum(int(report["lm loss"][1]) for report in reports) == local_tokens
            step_loss = sum(float(report["lm loss"][0]) for report in reports)
            assert len(validated) == step + 1
            assert validated[-1].device.type == "cpu"
            assert validated[-1].item() == pytest.approx(step_loss / PER_RANK, rel=1e-6)
            # The default config rebalances: samples move between the ranks.
            assert executors[-1].stats.moves["sample"] > 0
            grads = _main_grads(ddp)
            _assert_close(grads, single_rank)
            _assert_close(grads, eager)
    finally:
        ddp.module.config.finalize_model_grads_func = real
        schedule.set_iteration(0)
    assert executors[0] is executors[1]  # cached; the default collection is refreshed per call
    assert executors[1].pg_collection.dp is not None


def test_waves_match_one_pass():
    """D4 (G4): waves 2 and 4 against one pass over the same W EUs: per-sample losses bitwise,
    main_grad after finalize within the gate."""
    batch = _global_batch(dist.get_world_size())
    local = _local_slice(batch)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    results = {}
    for waves in (1, 2, 4):
        ddp = _ddp_model()
        reports, executor = _rsbf_step(ddp, local, pg_collection, batch, waves=waves)
        assert len(executor.stats.store_peaks) == waves and executor.stats.host_syncs == 1
        results[waves] = (_losses(reports, executor), _main_grads(ddp))
    base_losses, base_grads = results[1]
    for losses, grads in results.values():
        _assert_losses_equal(losses, base_losses)
        _assert_close(grads, base_grads)


def test_sync_budget_at_w_eus(single_rank_groups):
    """D5 (G5): known depths, W EUs, under ``set_sync_debug_mode('error')`` (which only the
    executor's own end-of-step read lowers): one host sync and no status exchange per step.
    The program and executor are built once, before error mode, and reused for the warm-up
    and the checked step (as ``train()`` reuses its cached executor; the program constructor
    copies host constants to the device). An online criterion over the Gloo status group: one
    status exchange per forward round, one host sync per round plus one per step, and the
    one-EU run's depths and losses."""
    batch = _global_batch(dist.get_world_size())
    local = _local_slice(batch)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    ddp = _ddp_model()
    executor = LoopRSBFExecutor(
        LoopProgram(ddp, exit_criterion=_criterion(batch)), pg_collection=pg_collection
    )
    # One step first: NCCL creates the pair communicators of the moves lazily; finalize
    # (outside the executor) is not part of the budget.
    _rsbf_step(ddp, local, pg_collection, batch, executor=executor)
    ddp.zero_grad_buffer()
    config = ddp.module.config
    real, config.finalize_model_grads_func = config.finalize_model_grads_func, None
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _, checked = _rsbf_step(ddp, local, pg_collection, batch, executor=executor)
    finally:
        torch.cuda.set_sync_debug_mode(0)
        config.finalize_model_grads_func = real
    assert checked is executor
    assert executor.stats.host_syncs == 1 and executor.stats.status_exchanges == 0
    assert executor.stats.moves["sample"] > 0

    head = HaltingHead(1, ddp.module.config.hidden_size, device=torch.device("cuda"), seed=3)
    criterion = OnlineHaltingCriterion(head, tau=1.4)
    reference = _ddp_model(single_rank_groups)
    ref_reports, ref_executor = _rsbf_step(
        reference, list(enumerate(batch)), single_rank_groups, batch, criterion=criterion
    )
    ddp = _ddp_model()
    groups = RSBFGroups(
        domain=pg_collection.dp, status=parallel_state.get_data_parallel_group_gloo()
    )
    reports, executor = _rsbf_step(
        ddp, local, pg_collection, batch, criterion=criterion, groups=groups, check_plan=True
    )
    stats = executor.stats
    rounds = sum(stats.rounds_by_wave)
    assert stats.status_exchanges == rounds and stats.host_syncs == rounds + 1
    assert stats.depths == ref_executor.stats.depths
    _assert_losses_equal(_losses(reports, executor), _losses(ref_reports, ref_executor))


@pytest.mark.parametrize("case", ["home", "moved"])
def test_buggy_plan_raises_on_every_rank(monkeypatch, case):
    """D6 (G7): a host plan that sweeps sample 0 one round after its exit (the same plan on
    every rank) is caught by the after-exit counter of the EU that swept it; the end-of-step
    flags all-reduce makes every rank raise the same error naming that EU, and the next step
    on the same communicators runs. ``moved``: every sample exits at round 1 except sample W
    (3 rounds), so the rebalancer moves the late-retired sample 0 from EU 0 to EU 1 before
    its extra sweep, and only the exit flags of the move tell EU 1 that it already exited."""
    world = dist.get_world_size()
    batch = _global_batch(world)
    local = _local_slice(batch)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    ddp = _ddp_model()
    criterion = None
    if case == "moved":
        criterion = PrescribedDepthCriterion(
            {index: 3 if index == world else 1 for index in range(len(batch))}
        )
    original = LoopRSBFExecutor._newly_exited
    monkeypatch.setattr(LoopRSBFExecutor, "_newly_exited", retire_late(original, sample=0))
    program = LoopProgram(ddp, exit_criterion=criterion or _criterion(batch))
    executor = LoopRSBFExecutor(program, pg_collection=pg_collection)
    message = None
    try:
        executor.run_step(
            lambda iterator, model: next(iterator),
            iter(_microbatch(sample) for _, sample in local),
            len(local),
            model=ddp,
        )
    except RuntimeError as error:
        message = str(error)
    monkeypatch.setattr(LoopRSBFExecutor, "_newly_exited", original)
    messages = [None] * world
    dist.all_gather_object(messages, message)
    assert all(text is not None and SWEEP_AFTER_EXIT in text for text in messages), messages
    assert len(set(messages)) == 1 and "on domain EUs" in messages[0]
    if case == "moved":
        assert executor._moves_of[0] > 0, executor.stats.moves
        assert "on domain EUs [1]" in messages[0], messages
    ddp.zero_grad_buffer()
    _, executor = _rsbf_step(ddp, local, pg_collection, batch)
    assert executor.program.after_exit_violations.item() == 0
