# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real-model, distributed source-projection training qualification.

Run the initial PP2/VPP2 case on four GPUs from the repository root::

    torchrun --standalone --nproc-per-node=4 -m pytest -x -s \
        'tests/unit_tests/transformer/test_attention_residual_source_model.py::test_source_model_training[gpt_pp2_vpp2]'

Select one exact parametrized node when avoiding similarly named cases. The full
file exercises PP1/2/4, VPP, partial blocks, GPT/hybrid, detached/repeated MTP,
offloading, selective recompute, TP2+SP, CP2 and EP2 MoE. The parallel extensions
each fit four GPUs (PP2 times TP2/CP2/EP2); CP uses the native zigzag batch
partitioner and TE fused attention. The uneven hybrid case has chunk lengths
2/6/4/4, not an even-split approximation. ATTNRES_MODEL_STEPS defaults to 10 (minimum 10).
ATTNRES_MODEL_RESUME_STEPS=3 additionally saves model and optimizer after step 7,
rebuilds the source model, and replays the last three updates. The reference
implementation defaults to FLA; ATTNRES_MODEL_REFERENCE=eager is also supported.
ATTNRES_MODEL_FRACTION optionally overrides every case's source fraction.

These are normal model imports and production schedules with overlapped MCore
DDP and BF16 Adam/FP32 master parameters. The optimizer is deliberately the
nonsharded Megatron optimizer: comparisons inspect every reduced main_grad;
distributed-optimizer checkpoint resharding needs separate qualification.
No CPU surrogate, mocked communication, or kernel fallback is used here.
"""

import gc
import io
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention_residual import AttentionResidual
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_batch_on_this_cp_rank
from tests.unit_tests.test_utilities import Utils


@dataclass(frozen=True)
class _Case:
    name: str
    pp: int
    vp: int = 1
    hybrid: bool = False
    mtp: int = 0
    detach: bool = False
    offload: bool = False
    recompute: bool = False
    fraction: float = 1.0
    overlap_p2p: bool = False
    tp: int = 1
    cp: int = 1
    ep: int = 1
    sequence_parallel: bool = False
    repeated_mtp: bool = False
    hybrid_segments: tuple[int, ...] | None = None


_CASES = [
    _Case("gpt_pp1", 1),
    _Case("gpt_pp2", 2),
    _Case("gpt_pp4", 4),
    _Case("gpt_pp2_vpp2", 2, 2),
    _Case("gpt_pp4_vpp2", 4, 2, overlap_p2p=True),
    _Case("gpt_pp2_vpp4", 2, 4, fraction=0.5, overlap_p2p=True),
    _Case("gpt_pp2_vpp2_zero", 2, 2, fraction=0.0),
    _Case("hybrid_pp2_vpp2", 2, 2, hybrid=True, fraction=0.5),
    _Case("gpt_pp2_vpp2_mtp", 2, 2, mtp=2),
    _Case("gpt_pp2_vpp2_detach_offload", 2, 2, mtp=2, detach=True, offload=True),
    _Case(
        "hybrid_pp2_vpp2_detach_offload",
        2,
        2,
        hybrid=True,
        mtp=2,
        detach=True,
        offload=True,
        recompute=True,
    ),
    _Case("gpt_pp2_vpp2_recompute", 2, 2, recompute=True, fraction=0.5),
    _Case("gpt_tp2_sp_pp2_vpp2", 2, 2, tp=2, sequence_parallel=True, fraction=0.5),
    _Case("gpt_cp2_pp2_vpp2", 2, 2, cp=2, fraction=0.5),
    _Case("gpt_ep2_moe_pp2_vpp2", 2, 2, ep=2, fraction=0.5),
    _Case("gpt_pp2_vpp2_mtp_repeated", 2, 2, mtp=2, repeated_mtp=True),
    _Case(
        "hybrid_pp2_vpp2_mtp_repeated_detach",
        2,
        2,
        hybrid=True,
        mtp=2,
        detach=True,
        repeated_mtp=True,
        fraction=0.5,
    ),
    _Case("hybrid_pp2_vpp2_uneven", 2, 2, hybrid=True, hybrid_segments=(2, 6, 4, 4), fraction=0.5),
]

_SEQ_LENGTH = 128
_HIDDEN_SIZE = 1024
_VOCAB_SIZE = 4096
_NUM_LAYERS = 16


def _config(case, impl):
    chunks = case.pp * case.vp
    # Each hybrid entry is one sublayer. Explicit segments also exercise the
    # hybrid layout parser used to derive pipeline value/score payload shapes.
    if case.hybrid_segments is not None:
        assert case.hybrid
        assert len(case.hybrid_segments) == chunks
        assert sum(case.hybrid_segments) == _NUM_LAYERS
        assert all(length > 0 and length % 2 == 0 for length in case.hybrid_segments)
        pattern = "|".join("*-" * (length // 2) for length in case.hybrid_segments)
    else:
        pattern = "|".join(["*-" * (_NUM_LAYERS // chunks // 2)] * chunks)
    if case.mtp:
        pattern += "/*-" * case.mtp
    moe = {}
    if case.ep > 1:
        # Dense first two layers, then four small routed experts. The ordinary
        # all-to-all dispatcher and sequential expert MLP need no DeepEP or
        # grouped-GEMM extension; the model still exercises actual EP collectives.
        moe = dict(
            num_moe_experts=4,
            moe_ffn_hidden_size=512,
            moe_layer_freq=[0, 0] + [1] * (_NUM_LAYERS - 2),
            moe_router_topk=2,
            moe_router_dtype="fp32",
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0.01,
            moe_token_dispatcher_type="alltoall",
            moe_grouped_gemm=False,
        )
    return TransformerConfig(
        num_layers=_NUM_LAYERS,
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=8,
        ffn_hidden_size=2048,
        pipeline_model_parallel_size=case.pp,
        virtual_pipeline_model_parallel_size=case.vp if case.vp > 1 else None,
        tensor_model_parallel_size=case.tp,
        context_parallel_size=case.cp,
        expert_model_parallel_size=case.ep,
        expert_tensor_parallel_size=case.tp,
        sequence_parallel=case.sequence_parallel,
        is_hybrid_model=case.hybrid,
        hybrid_layer_pattern=pattern if case.hybrid else None,
        enable_attention_residuals=True,
        # Three does not divide the 2/4/8-layer stage widths: sources complete
        # both inside chunks and after crossing real pipeline boundaries.
        attn_res_block_layers=3,
        attn_res_impl=impl,
        attn_res_source_projection_fraction=float(
            os.environ.get("ATTNRES_MODEL_FRACTION", case.fraction)
        ),
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        # CP is implemented by TE's fused/flash backends, not unfused attention.
        attention_backend=AttnBackend.fused if case.cp > 1 else AttnBackend.unfused,
        cp_comm_type="p2p" if case.cp > 1 else None,
        deallocate_pipeline_outputs=True,
        overlap_p2p_comm=case.overlap_p2p,
        batch_p2p_comm=not case.overlap_p2p,
        mtp_num_layers=case.mtp or None,
        mtp_detach_heads=case.detach,
        mtp_use_repeated_layer=case.repeated_mtp,
        fine_grained_activation_offloading=case.offload,
        offload_modules=["attn_norm", "mlp_norm"] if case.offload else [],
        min_offloaded_tensor_size=1,
        recompute_granularity="selective" if case.recompute else None,
        recompute_modules=["core_attn"] if case.recompute else [],
        **moe,
    )


def _build(case, impl, pg_collection, initial_state=None):
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    config = _config(case, impl)
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    chunks = []
    for chunk_id in range(case.vp):
        vp_stage = chunk_id if case.vp > 1 else None
        kwargs = dict(
            config=config,
            vocab_size=_VOCAB_SIZE,
            max_sequence_length=_SEQ_LENGTH,
            pre_process=pp_rank == 0 and chunk_id == 0,
            post_process=pp_rank == case.pp - 1 and chunk_id == case.vp - 1,
            position_embedding_type="rope",
            share_embeddings_and_output_weights=False,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
        if case.hybrid:
            module = HybridModel(
                hybrid_stack_spec=hybrid_stack_spec,
                hybrid_layer_pattern=config.hybrid_layer_pattern,
                **kwargs,
            )
        else:
            spec = get_gpt_decoder_block_spec(
                config, use_transformer_engine=True, vp_stage=vp_stage, pp_rank=pp_rank
            )
            mtp_spec = (
                get_gpt_mtp_block_spec(config, spec, True, vp_stage=vp_stage, pp_rank=pp_rank)
                if case.mtp
                else None
            )
            module = GPTModel(transformer_layer_spec=spec, mtp_block_spec=mtp_spec, **kwargs)
        module = module.cuda().train()
        convert_module_to_dtype_except_fp32_marked(module, torch.bfloat16)
        with torch.no_grad():
            for residual in module.modules():
                if isinstance(residual, AttentionResidual):
                    # Zero queries would mask producer score and norm gradients.
                    residual.pseudo_query.normal_(std=0.005)
                    residual.key_norm_weight.uniform_(0.8, 1.2)
        if initial_state is not None:
            module.load_state_dict(initial_state[chunk_id], strict=True)
        chunks.append(
            DistributedDataParallel(
                config,
                DistributedDataParallelConfig(
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=True,
                    use_distributed_optimizer=False,
                    bucket_size=1_000_000,
                ),
                module,
                pg_collection=pg_collection,
            )
        )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=1e-4,
            weight_decay=0.01,
            clip_grad=1.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=False,
        ),
        chunks,
        pg_collection=pg_collection,
        use_gloo_process_groups=False,
    )
    config.grad_scale_func = optimizer.scale_loss
    config.finalize_model_grads_func = finalize_model_grads
    config.no_sync_func = [chunk.no_sync for chunk in chunks]
    config.grad_sync_func = [chunk.start_grad_sync for chunk in chunks]
    if case.vp == 1:
        config.no_sync_func = config.no_sync_func[0]
        config.grad_sync_func = config.grad_sync_func[0]
    return chunks, optimizer


def _batch_iterators(case, step):
    generator = torch.Generator().manual_seed(
        9000 + 100 * step + parallel_state.get_data_parallel_rank()
    )
    batches = []
    for _ in range(max(4, 2 * case.pp)):
        tokens = torch.randint(_VOCAB_SIZE, (1, _SEQ_LENGTH + 1), generator=generator).cuda()
        batches.append((tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()))
    return [iter(batches) for _ in range(case.vp)]


def _forward_step(data_iterator, model):
    tokens, labels = next(data_iterator)
    positions = torch.arange(_SEQ_LENGTH, device="cuda").unsqueeze(0)
    mask = torch.ones((1, 1, _SEQ_LENGTH, _SEQ_LENGTH), device="cuda", dtype=torch.bool)
    mask = torch.triu(mask, diagonal=1)
    batch = dict(
        tokens=tokens,
        labels=labels,
        position_ids=positions,
        attention_mask=mask,
        loss_mask=torch.ones_like(labels),
    )
    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size > 1:
        batch = get_batch_on_this_cp_rank(
            batch, cp_group=parallel_state.get_context_parallel_group()
        )
        assert batch['tokens'].shape[1] == _SEQ_LENGTH // cp_size
    output = model(
        batch['tokens'],
        batch['position_ids'],
        batch['attention_mask'],
        labels=batch['labels'],
        loss_mask=batch['loss_mask'],
    )

    def loss_func(losses):
        loss = losses.float().mean()
        if cp_size > 1:
            # Use the native sum/token-count protocol. The schedule handles
            # microbatch normalization and CP/DP gradient finalization; the
            # legacy two-item protocol applies an additional CP-size factor.
            total = (losses.float() * batch['loss_mask']).sum()
            count = batch['loss_mask'].sum().detach().to(torch.int)
            return total, count, {"loss": loss.detach().clone()}
        return loss, {"loss": loss.detach().clone()}

    return output, loss_func


def _backward(case, chunks, optimizer, step, pg_collection):
    for chunk in chunks:
        chunk.zero_grad_buffer()
    optimizer.zero_grad()
    communicator = (
        P2PCommunicator(pp_group=pg_collection.pp, config=chunks[0].config) if case.pp > 1 else None
    )
    return get_forward_backward_func()(
        forward_step_func=_forward_step,
        data_iterator=_batch_iterators(case, step),
        model=chunks,
        num_microbatches=max(4, 2 * case.pp),
        seq_length=_SEQ_LENGTH,
        micro_batch_size=1,
        forward_only=False,
        pg_collection=pg_collection,
        p2p_communicator=communicator,
    )


def _parameters(chunks):
    return {
        f"chunk{index}.{name}": parameter
        for index, chunk in enumerate(chunks)
        for name, parameter in chunk.module.named_parameters()
    }


def _install_operator_capture(chunks, label):
    """Optionally save matched-input AttnRes VJPs without changing the graph.

    This opt-in diagnostic performs CPU copies and file writes, so it is never
    enabled by the performance driver. The saved BF16 inputs, FP32 parameters
    and exact incoming dY allow independent replay without a pipeline schedule.
    """
    directory = os.environ.get("ATTNRES_MODEL_CAPTURE_DIR")
    if not directory:
        return
    selector = re.compile(os.environ.get("ATTNRES_MODEL_CAPTURE_REGEX", "attn_res"))
    rank = torch.distributed.get_rank()
    directory = Path(directory) / f"rank{rank}" / label
    directory.mkdir(parents=True, exist_ok=True)

    def make_hook(name):
        count = 0

        def hook(module, inputs, output):
            nonlocal count
            index = count
            count += 1
            record = dict(
                name=name,
                call=index,
                rank=rank,
                implementation=module.impl,
                consumer_id=module.projection_consumer_id,
                values=[value.detach().cpu().clone() for value in inputs[0]],
                pseudo_query=module.pseudo_query.detach().cpu().clone(),
                key_norm_weight=module.key_norm_weight.detach().cpu().clone(),
                eps=module.eps,
                output=output.detach().cpu().clone(),
            )
            if module.impl == "source":
                from megatron.core.transformer.attention_residual_projection_runtime import (
                    get_attn_res_projection_runtime,
                )
                from megatron.core.transformer.attention_residual_source_state import (
                    source_logits_for_consumer,
                )

                runtime = get_attn_res_projection_runtime(module)
                column = runtime.column(module)
                logits = [source_logits_for_consumer(value, column) for value in inputs[0]]
                record["precomputed_logits"] = [
                    logit.detach().cpu().clone() if logit is not None else None for logit in logits
                ]
                record["effective_query"] = runtime.effective_weight(module).detach().cpu().clone()

            def save_backward(gradient):
                record["grad_output"] = gradient.detach().cpu().clone()
                torch.save(record, directory / f"{name}.call{index}.pt")
                return gradient

            output.register_hook(save_backward)

        return hook

    for chunk_index, chunk in enumerate(chunks):
        for name, module in chunk.module.named_modules():
            qualified_name = f"chunk{chunk_index}.{name}"
            if isinstance(module, AttentionResidual) and selector.search(qualified_name):
                module.register_forward_hook(make_hook(qualified_name))


def _assert_globally(errors, context):
    # Make every rank fail at the same boundary instead of leaving healthy
    # ranks waiting in the next schedule's NCCL receive.
    failed = torch.tensor(bool(errors), dtype=torch.int, device="cuda")
    torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    if failed.item():
        messages = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(messages, errors[:8])
        pytest.fail(f"{context}: {messages}")


def _compare_tensors(actual, expected, *, context, similarity_epsilon=1e-3):
    errors = []
    mismatches = []
    directory = os.environ.get("ATTNRES_MODEL_CAPTURE_DIR")
    if directory and "main_grad" in context:
        rank = torch.distributed.get_rank()
        directory = Path(directory) / f"rank{rank}"
        directory.mkdir(parents=True, exist_ok=True)
        query_names = sorted(
            name
            for name in actual.keys() & expected.keys()
            if "pseudo_query" in name or "key_norm_weight" in name
        )
        torch.save(
            {
                "context": context,
                "dp_ranks": torch.distributed.get_process_group_ranks(
                    parallel_state.get_data_parallel_group(with_context_parallel=True)
                ),
                "actual": {
                    name: actual[name].detach().cpu().clone()
                    for name in query_names
                    if actual[name] is not None
                },
                "expected": {
                    name: expected[name].detach().cpu().clone()
                    for name in query_names
                    if expected[name] is not None
                },
            },
            directory / f"{context.replace(' ', '_')}.pt",
        )
    if actual.keys() != expected.keys():
        errors.append("parameter/tensor names differ")
    for name in sorted(actual.keys() & expected.keys()):
        left, right = actual[name], expected[name]
        if left is None or right is None:
            errors.append(f"{name}: missing tensor/gradient")
            continue
        left, right = left.detach().float().flatten(), right.detach().float().flatten()
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            errors.append(f"{name}: nonfinite value")
            continue
        denominator = (left.square() + right.square()).sum()
        if denominator == 0:
            continue
        # Same tensor/cosine thresholds as the real K3 AttnRes/MTP test. These
        # model-level metrics do not replace strict primitive dq/dgamma tests.
        similarity = (2 * (left * right).sum() / denominator).item()
        cosine = torch.nn.functional.cosine_similarity(left[None], right[None], eps=1e-30).item()
        if min(similarity, cosine) < 1 - similarity_epsilon:
            difference = left - right
            actual_norm = left.norm().item()
            expected_norm = right.norm().item()
            mismatches.append(
                (
                    min(similarity, cosine),
                    f"{name}: cosine={cosine:.8g}, similarity={similarity:.8g}, "
                    f"actual_norm={actual_norm:.8g}, expected_norm={expected_norm:.8g}, "
                    f"difference_norm={difference.norm().item():.8g}, "
                    f"maxabs={difference.abs().max().item():.8g}",
                )
            )
    # Report the strongest mismatch first, rather than whichever key happened
    # to occur first in the hash-dependent set intersection.
    errors.extend(message for _, message in sorted(mismatches))
    _assert_globally(errors, context)


def _compare_losses(actual, expected, step):
    errors = []
    if len(actual) != len(expected):
        errors.append("number of microbatch losses differs")
    for index, (left, right) in enumerate(zip(actual, expected)):
        if not torch.allclose(left["loss"], right["loss"], rtol=1e-3, atol=1e-3):
            errors.append(f"microbatch {index}: {left['loss'].item()} != {right['loss'].item()}")
    _assert_globally(errors, f"step {step} loss")


def _assert_live_queries(chunks, case):
    counts = torch.zeros(2, dtype=torch.int, device="cuda")
    errors = []
    for name, parameter in _parameters(chunks).items():
        if "pseudo_query" not in name and "key_norm_weight" not in name:
            continue
        if parameter.dtype != torch.float32:
            errors.append(f"{name}: canonical query/norm is not FP32")
        if parameter.main_grad is not None and torch.count_nonzero(parameter.main_grad):
            counts[0] += 1
            if "mtp" in name:
                counts[1] += 1
    torch.distributed.all_reduce(counts)
    if counts[0] == 0 or (case.mtp and counts[1] == 0):
        errors.append(f"expected nonzero trunk/MTP query gradients, got {counts.tolist()}")
    _assert_globally(errors, "source query gradients")


def _step(optimizer, context):
    success, grad_norm, _ = optimizer.step()
    errors = (
        []
        if success and grad_norm is not None and torch.isfinite(torch.as_tensor(grad_norm))
        else [f"optimizer failed or gradient norm is nonfinite: {success=}, {grad_norm=}"]
    )
    _assert_globally(errors, context)


def _query_snapshot(chunks):
    return {
        name: parameter.detach().clone()
        for name, parameter in _parameters(chunks).items()
        if "pseudo_query" in name or "key_norm_weight" in name
    }


def _assert_query_updates(chunks, before):
    errors = []
    for name, parameter in _parameters(chunks).items():
        if name in before and torch.count_nonzero(parameter.main_grad):
            if torch.equal(parameter.detach(), before[name]):
                errors.append(f"{name}: nonzero gradient did not update the canonical parameter")
    _assert_globally(errors, "source query optimizer updates")


def _assert_case_structure(chunks, case):
    """Ensure each new matrix entry instantiated the requested real path."""
    errors = []
    config = chunks[0].module.config
    if config.sequence_parallel != case.sequence_parallel:
        errors.append("sequence-parallel configuration differs from the case")
    for name, actual, expected in (
        ("TP", parallel_state.get_tensor_model_parallel_world_size(), case.tp),
        ("CP", parallel_state.get_context_parallel_world_size(), case.cp),
        ("EP", parallel_state.get_expert_model_parallel_world_size(), case.ep),
    ):
        if actual != expected:
            errors.append(f"{name} group has {actual} ranks, expected {expected}")
    counts = torch.zeros(2, dtype=torch.int, device="cuda")
    for chunk_id, chunk in enumerate(chunks):
        module = chunk.module
        counts[0] += sum(isinstance(layer, MoELayer) for layer in module.modules())
        if case.hybrid_segments is not None:
            stage = chunk_id * case.pp + parallel_state.get_pipeline_model_parallel_rank()
            if len(module.decoder.layers) != case.hybrid_segments[stage]:
                errors.append(f"chunk {chunk_id}: uneven hybrid segment length differs")
        mtp = getattr(module, "mtp", None)
        if case.repeated_mtp and mtp is not None:
            counts[1] += 1
            if len(mtp.layers) != 1 or not mtp.mtp_use_repeated_layer:
                errors.append("repeated MTP must construct exactly one shared layer")
    torch.distributed.all_reduce(counts)
    if case.ep > 1 and counts[0] == 0:
        errors.append("EP case did not instantiate a MoE layer")
    if case.repeated_mtp and counts[1] == 0:
        errors.append("repeated-MTP case did not instantiate an MTP block")
    _assert_globally(errors, "parallel/model qualification geometry")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA models and NCCL")
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_source_model_training(case, strict_fp32_matmul):
    """Compare losses, all gradients and all updates through real PP schedules."""
    # EP subdivides the dense data-parallel population; it is not multiplied
    # into the dense TP*CP*PP world-size divisor a second time.
    required_world_multiple = math.lcm(case.tp * case.cp * case.pp, case.tp * case.ep * case.pp)
    if Utils.world_size % required_world_multiple:
        pytest.skip(
            f"TP{case.tp}/CP{case.cp}/EP{case.ep}/PP{case.pp} requires world size "
            f"divisible by {required_world_multiple}"
        )
    steps = int(os.environ.get("ATTNRES_MODEL_STEPS", "10"))
    resume_steps = int(os.environ.get("ATTNRES_MODEL_RESUME_STEPS", "0"))
    assert steps >= 10, "qualification must include at least ten optimizer updates"
    assert resume_steps == 0 or 3 <= resume_steps < steps
    reference_impl = os.environ.get("ATTNRES_MODEL_REFERENCE", "fla")
    assert reference_impl in ("eager", "fla")
    Utils.initialize_model_parallel(
        case.tp,
        case.pp,
        virtual_pipeline_model_parallel_size=case.vp if case.vp > 1 else None,
        context_parallel_size=case.cp,
        expert_model_parallel_size=case.ep,
        expert_tensor_parallel_size=case.tp,
    )
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        reference, reference_optimizer = _build(case, reference_impl, pg_collection)
        initial = [chunk.module.state_dict() for chunk in reference]
        source, source_optimizer = _build(case, "source", pg_collection, initial)
        del initial
        _install_operator_capture(reference, reference_impl)
        _install_operator_capture(source, "source")
        _assert_case_structure(source, case)
        checkpoint = None
        replay_losses = {}
        for step in range(steps):
            reference_losses = _backward(case, reference, reference_optimizer, step, pg_collection)
            source_losses = _backward(case, source, source_optimizer, step, pg_collection)
            _compare_losses(source_losses, reference_losses, step)
            _compare_tensors(
                {name: p.main_grad for name, p in _parameters(source).items()},
                {name: p.main_grad for name, p in _parameters(reference).items()},
                context=f"step {step} main_grad",
            )
            _assert_live_queries(source, case)
            query_before = _query_snapshot(source)
            _step(reference_optimizer, f"reference step {step}")
            _step(source_optimizer, f"source step {step}")
            _assert_query_updates(source, query_before)
            _compare_tensors(
                _parameters(source), _parameters(reference), context=f"step {step} weights"
            )
            if resume_steps and step >= steps - resume_steps:
                replay_losses[step] = source_losses
            if resume_steps and step + 1 == steps - resume_steps:
                checkpoint = io.BytesIO()
                torch.save(
                    {
                        "model": [chunk.module.state_dict() for chunk in source],
                        "optimizer": source_optimizer.state_dict(),
                    },
                    checkpoint,
                )
            if torch.distributed.get_rank() == 0:
                print(f"AttnRes model parity: {case.name} step={step + 1}/{steps}", flush=True)
        if resume_steps:
            checkpoint.seek(0)
            state = torch.load(checkpoint, weights_only=False)
            resumed, resumed_optimizer = _build(case, "source", pg_collection, state["model"])
            resumed_optimizer.load_state_dict(state["optimizer"])
            del state
            for step in range(steps - resume_steps, steps):
                losses = _backward(case, resumed, resumed_optimizer, step, pg_collection)
                _compare_losses(losses, replay_losses[step], f"resumed {step}")
                _step(resumed_optimizer, f"resumed step {step}")
            _compare_tensors(
                _parameters(resumed),
                _parameters(source),
                context=f"{resume_steps} resumed source updates",
                similarity_epsilon=1e-6,
            )
            del resumed, resumed_optimizer
        del reference, reference_optimizer, source, source_optimizer
        gc.collect()
    finally:
        Utils.destroy_model_parallel()


@pytest.fixture
def strict_fp32_matmul():
    """Apply the projection precision contract to both reference and candidate."""
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn
