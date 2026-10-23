# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DDP tests of the one-EU RSBF executor (GPU, every rank one EU; run with WORLD_SIZE = 2, the
M1 "DP = 2" gate, and with 8).

Each DP rank runs its slice of one fixed global batch through RSBF under
``DistributedDataParallel`` (``overlap_grad_reduce=False``, fp32 ``main_grad``), with
prescribed depths chosen so the ranks run different numbers of items; finalize is the step's
only cross-rank communication. After finalize, ``main_grad`` must equal

* a single-rank RSBF reference: the same model under DDP over a one-rank DP group running the
  whole global batch (the gradient scale does not depend on the DP size: the token normalizer
  receives each rank's local token count and finalize all-reduces it once), and
* the standard schedule (``forward_backward_no_pipelining``, eager ``LoopHybridStack`` at the
  same per-sample depths, MBS 1) under the same DDP wrap,

at rtol 1e-5 with an absolute floor of 1e-6 x max |g| (fp32; the executor sums per-round
gradients in a different association than the eager graph). The distributed optimizer's
reduce-scatter is checked by gathering the reduced shards back.

* D1 + D2 drive ``LoopRSBFExecutor.run_step`` directly (local layers).
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
from megatron.core.transformer.loop.program import LoopProgram, PrescribedDepthCriterion
from megatron.core.transformer.loop.rsbf_executor import (
    RSBF_BATCH_FUNC_ATTR,
    RSBF_LOSS_VALIDATOR_ATTR,
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


def _rsbf_step(ddp, samples, pg_collection, record=None):
    depths = {index: sample["depth"] for index, sample in samples}
    program = LoopProgram(ddp, exit_criterion=PrescribedDepthCriterion(depths))
    executor = LoopRSBFExecutor(program, pg_collection=pg_collection)
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


def _single_rank_reference(batch, single_rank_groups, *, te=False):
    """RSBF over the whole global batch under DDP over a one-rank group."""
    reference = _ddp_model(single_rank_groups, te=te)
    tokens = []
    _rsbf_step(reference, list(enumerate(batch)), single_rank_groups, record=tokens)
    assert tokens[0].item() == sum(int(sample["loss_mask"].sum()) for sample in batch)
    return _main_grads(reference)


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


@pytest.mark.parametrize("distributed_optimizer", [False, True])
def test_dp_ranks_with_different_item_counts_match_the_references(
    single_rank_groups, distributed_optimizer
):
    """D1 + D2: ranks run different item counts and the step completes; after finalize
    main_grad equals the single-rank RSBF reference over the whole global batch and the
    standard schedule under the same DDP wrap; finalize receives the local token count."""
    batch = _global_batch(dist.get_world_size())
    local = _local_slice(batch)

    # The DP arm: each rank one EU on its slice.
    ddp = _ddp_model(distributed_optimizer=distributed_optimizer)
    seen_tokens = []
    reports, executor = _rsbf_step(
        ddp, local, ProcessGroupCollection.use_mpu_process_groups(), record=seen_tokens
    )
    if distributed_optimizer:
        _gather_reduced_shards(ddp)
    dp_grads = _main_grads(ddp)
    _assert_item_counts_differ(executor)
    local_tokens = sum(int(sample["loss_mask"].sum()) for _, sample in local)
    assert len(seen_tokens) == 1 and seen_tokens[0].item() == local_tokens
    assert sum(int(report["lm loss"][1]) for report in reports) == local_tokens

    _assert_close(dp_grads, _single_rank_reference(batch, single_rank_groups))
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
    criterion = PrescribedDepthCriterion({index: sample["depth"] for index, sample in local})
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
            _assert_item_counts_differ(executors[-1])
            grads = _main_grads(ddp)
            _assert_close(grads, single_rank)
            _assert_close(grads, eager)
    finally:
        ddp.module.config.finalize_model_grads_func = real
        schedule.set_iteration(0)
    assert executors[0] is executors[1]  # cached; the default collection is refreshed per call
    assert executors[1].pg_collection.dp is not None
