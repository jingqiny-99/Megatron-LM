# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The RSBF planner's plan applied to DSV4-TINY and run through the executor (GPU; run with
WORLD_SIZE = 8, also 2 and 4: EP = W, one EU per rank, model parallel initialized once).

DSV4-TINY: ``'HEHECEHECEHECE/WE'`` (prelude 'HE', core 'HECEHECE' = 4 layers, coda 'HECE', MTP
'WE'), 8 experts with EP = W (alltoall), a hash layer, mHC 4 streams, unfused DSv4 attention
(``dsa_kernel_backend='none'``), bf16, R_max 4 with sampled R4-uniform depths; S 256, 4 samples
per EU (MBS 1).

* ``plan_loop_rsbf(apply=True)`` with the h100 preset under each form policy (``d28``: the whole
  round on this 4-layer core; ``planner``: the raw pick; ``nested``): the executor the schedule
  builds from the model config runs the plan's item form, stride, waves and D42; two steps give
  finite losses and an exit histogram that sums to the step's samples on every rank. Report
  only (risk 3; the 10 % gate applies at R2 / R3): the planner's HBM against the max over ranks
  of ``max_memory_reserved`` (``max_memory_allocated`` next to it), and the planner's state store
  and block-input stash against the executor's ``StoreStats`` per EU (max and mean over EUs).
* The forced Adam-to-Grace arm (``--loop-planner-optimizer-offload always`` with a host tier,
  ``--loop-planner-grace-gb 400`` and a PCIe ``--loop-planner-c2c-gbps``, the precision-aware
  optimizer, AdamW): the trainer's optimizer config (``get_megatron_optimizer_config``) builds a
  CPU-offloaded optimizer (``HybridDeviceOptimizer``); two steps through the executor under DDP
  with the distributed optimizer give the first step's losses bitwise and the second's within
  1e-3 of the same plan without offload. Report only: the HBM delta and the host RSS delta
  against the planner's optimizer bytes (``12 (P_d + P_e) / N`` saved in HBM, ``16 (P_d + P_e) /
  N`` used on the host).
"""

import contextlib
import os
import resource
from argparse import Namespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.loop.planner_costs import GB
from megatron.core.transformer.loop.rsbf_executor import LoopMicrobatch, get_loop_rsbf_schedule
from megatron.training import loop_rsbf_planning as planning
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA, VOCAB, model_parallel_once

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
GPU_EP = CUDA and WORLD >= 2 and 8 % WORLD == 0
pytestmark = pytest.mark.skipif(
    not GPU_EP, reason="DSV4-TINY at EP = W needs CUDA and 2, 4 or 8 ranks"
)

PATTERN = "HEHECEHECEHECE/WE"
_RATIOS = {"W": 0, "C": 4, "H": 128, "E": 0}
SEQ = 256
PER_RANK = 4
STEPS = 2


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    if not GPU_EP:
        yield
        return
    with model_parallel_once(expert_model_parallel_size=WORLD):
        try:
            yield
        finally:
            get_loop_rsbf_schedule().reset()
            planning.reset_loop_planner_session()


@contextlib.contextmanager
def _hadamard_fallback():
    from unittest.mock import patch

    try:
        from fast_hadamard_transform import hadamard_transform  # noqa: F401

        yield
    except ImportError:
        with patch(
            "megatron.core.transformer.experimental_attention_variant.csa.rotate_activation",
            lambda x: x * (x.size(-1) ** -0.5),
        ):
            yield


def _tiny():
    """DSV4-TINY at EP = W (the fields of test_loop_rsbf_item_forms._dsv4_tiny), RSBF with
    sampled R4-uniform depths; the RSBF fields the planner owns at their defaults."""
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_dsv4_stack_spec
    from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    main, mtp = PATTERN.split("/")
    config = MLATransformerConfig(
        num_layers=len(main),
        hidden_size=512,
        num_attention_heads=8,
        multi_latent_attention=True,
        q_lora_rank=192,
        kv_lora_rank=64,
        qk_head_dim=16,
        qk_pos_emb_head_dim=8,
        v_head_dim=16,
        rope_type="rope",
        rotary_base=10000,
        rotary_percent=1.0,
        experimental_attention_variant="dsv4_hybrid",
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=32,
        dsa_indexer_loss_coeff=0.01,
        csa_window_size=32,
        csa_compress_ratios=[_RATIOS[symbol] for symbol in main + mtp],
        num_moe_experts=8,
        moe_ffn_hidden_size=256,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=1e-4,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_n_hash_layers=1,
        actual_vocab_size=VOCAB,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=5,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        loop_architecture="huginn",
        loop_iterations=4,
        loop_prelude_sublayers=2,
        loop_coda_sublayers=4,
        loop_schedule="rsbf",
        calculate_per_token_loss=True,
        loop_exit_criterion="sampled",
        loop_depth_probs=[0.25, 0.25, 0.25, 0.25],
        loop_depth_seed=1234,
        deterministic_mode=True,
        attention_backend=AttnBackend.unfused,
        expert_model_parallel_size=WORLD,
    )
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    set_moe_metrics_tracker(MoEMetricsTracker())
    return HybridModel(
        config=config,
        hybrid_stack_spec=loop_hybrid_dsv4_stack_spec(config),
        vocab_size=VOCAB,
        max_sequence_length=1024,
        hybrid_layer_pattern=PATTERN,
        position_embedding_type="rope",
    ).cuda()


def _args(**overrides):
    """The trainer's arguments of this run (TP = PP = CP = 1, DP = EP = W, MBS 1)."""
    values = dict(
        seq_length=SEQ,
        world_size=WORLD,
        data_parallel_size=WORLD,
        context_parallel_size=1,
        expert_model_parallel_size=WORLD,
        micro_batch_size=1,
        global_batch_size=PER_RANK * WORLD,
        loop_schedule="rsbf",
        loop_architecture="huginn",
        loop_prelude_sublayers=2,
        loop_coda_sublayers=4,
        loop_iterations=4,
        loop_rsbf_item_form="whole_round",
        loop_rsbf_block_stride=None,
        loop_rsbf_waves=1,
        loop_rsbf_coda_recompute_mla_up_proj=False,
        loop_planner="apply",
        loop_planner_hardware="h100",
        optimizer="adam",
        optimizer_cpu_offload=False,
        optimizer_offload_fraction=1.0,
        use_precision_aware_optimizer=False,
        use_distributed_optimizer=True,
        accumulate_allreduce_grads_in_fp32=True,
    )
    values.update(overrides)
    return Namespace(**values)


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


def _local_microbatches(step):
    """This rank's slice of the step's global batch (one row per microbatch)."""
    total = PER_RANK * WORLD
    generator = torch.Generator().manual_seed(100 + step)
    tokens = torch.randint(0, VOCAB, (total, SEQ), generator=generator)
    labels = torch.randint(0, VOCAB, (total, SEQ), generator=generator)
    rank = dist.get_rank()
    return [
        LoopMicrobatch(
            tokens=tokens[i : i + 1].cuda(),
            position_ids=torch.arange(SEQ)[None].cuda(),
            labels=labels[i : i + 1].cuda(),
            loss_mask=torch.ones(1, SEQ, device="cuda"),
            attention_mask=None,
            padding_mask=None,
            loss_func=_loss_func,
        )
        for i in range(rank, total, WORLD)
    ]


def _steps(model, executor_model=None, optimizer=None):
    """``STEPS`` training steps through the module schedule's executor of ``model`` (built from
    its config on first use); with ``optimizer`` the DDP grad buffer is zeroed before and the
    optimizer stepped after each. Returns (per-step {global index: loss}, executor)."""
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.loop.dispatcher import RSBFGroups

    executor_model = model if executor_model is None else executor_model
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    schedule = get_loop_rsbf_schedule()
    executor = schedule.executor_for(
        executor_model, pg_collection, None, RSBFGroups(domain=pg_collection.dp, status=None)
    )
    per_step = []
    for step in range(STEPS):
        if optimizer is not None:
            executor_model.zero_grad_buffer()
            optimizer.zero_grad()
        else:
            model.zero_grad(set_to_none=True)
        microbatches = _local_microbatches(step)
        reports = executor.run_step(
            lambda iterator, m: next(iterator),
            iter(microbatches),
            len(microbatches),
            model=executor_model,
            iteration=step,
        )
        losses = {
            record.key.global_index: report["lm loss"][0].float().item()
            for record, report in zip(executor.stats.loss_records, reports)
        }
        per_step.append(losses)
        counts = executor.stats.exit_counts
        assert sum(counts) == PER_RANK * WORLD == len(executor.stats.depths), counts
        every = [None] * WORLD
        dist.all_gather_object(every, list(counts))
        assert all(c == list(counts) for c in every), every
        if optimizer is not None:
            update_ok, _, _ = optimizer.step()
            assert update_ok
    return per_step, executor


def _merged(losses):
    every = [None] * WORLD
    dist.all_gather_object(every, losses)
    return {k: v for part in every for k, v in part.items()}


def _report_memory(label, plan, executor):
    """Report only: planner HBM vs measured reserved / allocated (max over ranks), planner store
    and stash vs the executor's per-EU store peaks (max and mean over EUs)."""
    measured = torch.tensor(
        [torch.cuda.max_memory_reserved(), torch.cuda.max_memory_allocated()],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(measured, op=dist.ReduceOp.MAX)
    peaks = {}
    for stats in executor.stats.store_peaks:
        for kind, value in stats.peak_bytes_by_kind.items():
            peaks[kind] = max(peaks.get(kind, 0), value)
    every = [None] * WORLD
    dist.all_gather_object(every, peaks)
    if dist.get_rank() == 0:
        applied = plan.applied
        print(
            f"[{label}] planner HBM {applied.hbm_bytes / GB:.3f} GB vs max over ranks "
            f"max_memory_reserved {measured[0].item() / GB:.3f} GB / max_memory_allocated "
            f"{measured[1].item() / GB:.3f} GB (report only; R2/R3 gate: within 10 %)"
        )
        kinds = sorted({kind for part in every for kind in part})
        for kind in kinds:
            values = [part.get(kind, 0) for part in every]
            print(
                f"[{label}] store peak '{kind}': max {max(values) / GB:.4f} GB, mean "
                f"{sum(values) / len(values) / GB:.4f} GB over {WORLD} EUs"
            )
        print(
            f"[{label}] planner state store {applied.store_bytes / GB:.4f} GB, block-input "
            f"stash need {applied.stash_need_bytes / GB:.4f} GB (averaged +1 sample-round rule)"
        )


@pytest.mark.parametrize("forms", ["d28", "planner", "nested"])
def test_the_applied_plan_runs_dsv4_tiny_at_w_eus(forms):
    from megatron.core.transformer.loop.program import build_core_plan

    planning.reset_loop_planner_session()
    get_loop_rsbf_schedule().reset()
    with _hadamard_fallback():
        model = _tiny()
        args = _args(loop_planner_forms=forms)
        plan = planning.plan_loop_rsbf(args, [model], apply=True)
        assert plan.applied.fits and plan.applied.applicable
        if forms == "d28":
            assert (plan.item_form, plan.block_stride) == ("whole_round", None)
        elif forms == "nested":
            assert (plan.item_form, plan.block_stride) == ("nested", 2)
        assert plan.coda_recompute == plan.applied.option.coda_recompute
        assert not plan.optimizer_offload and args.optimizer_cpu_offload is False
        for name, value in plan.config_overrides().items():
            assert getattr(model.config, name) == value == getattr(args, name), name
        torch.cuda.reset_peak_memory_stats()
        per_step, executor = _steps(model)
        program = executor.program
        roles = model.decoder.loop_roles
        assert executor.waves == plan.waves
        assert program.item_form == plan.item_form
        assert program.uniform_core_plan == build_core_plan(model.config, roles)
        assert bool(program.coda_recompute_modules) == plan.coda_recompute
        for losses in per_step:
            merged = _merged(losses)
            assert len(merged) == PER_RANK * WORLD
            assert all(torch.isfinite(torch.tensor(v)) for v in merged.values()), merged
        _report_memory(
            f"forms={forms} {plan.item_form}/{plan.block_stride} w={plan.waves}", plan, executor
        )
    get_loop_rsbf_schedule().reset()
    del model, executor


def _optimizer_arm(args, overrides=None):
    """A DDP DSV4-TINY with the distributed optimizer built by the trainer's optimizer config
    from ``args``; ``overrides`` (a plan's config overrides) are set instead of planning."""
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
        finalize_model_grads,
    )
    from megatron.core.optimizer import get_megatron_optimizer
    from megatron.training.training import get_megatron_optimizer_config

    planning.reset_loop_planner_session()
    get_loop_rsbf_schedule().reset()
    model = _tiny()
    ddp = DistributedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            overlap_grad_reduce=False, use_distributed_optimizer=True, grad_reduce_in_fp32=True
        ),
        model,
    )
    model.config.finalize_model_grads_func = finalize_model_grads
    plan = None
    if overrides is None:
        plan = planning.plan_loop_rsbf(args, [ddp], apply=True)
    else:
        for name, value in overrides.items():
            setattr(model.config, name, value)
    config, _ = get_megatron_optimizer_config(args)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    hbm_before = torch.cuda.memory_allocated()
    optimizer = get_megatron_optimizer(config, [ddp])
    return model, ddp, optimizer, plan, (rss_before, hbm_before)


def _inner_optimizers(optimizer):
    """The torch-level optimizer of every leaf Megatron optimizer. A ``ChainedOptimizer`` (dense
    and expert groups) is walked through ``chained_optimizers``; its ``optimizer`` property
    asserts when it has more than one child, so it is read on the leaves only."""
    pending, found = [optimizer], []
    while pending:
        current = pending.pop()
        chained = getattr(current, "chained_optimizers", None)
        if chained is not None:
            pending.extend(chained)
        else:
            found.append(current.optimizer)
    return found


def _is_hybrid_device(inner):
    from megatron.core.optimizer import HybridDeviceOptimizer

    return isinstance(inner, HybridDeviceOptimizer)


def test_forced_adam_to_grace_builds_the_offloaded_optimizer():
    common = dict(
        lr=1e-4,
        min_lr=1e-5,
        weight_decay=0.01,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_precision_aware_optimizer=True,
        clip_grad=0.0,
        loop_planner_waves=[1],
    )
    args = _args(
        loop_planner_optimizer_offload="always",
        loop_planner_grace_gb=400.0,
        loop_planner_c2c_gbps=25.0,
        **common,
    )
    with _hadamard_fallback():
        torch.cuda.reset_peak_memory_stats()
        model, ddp, optimizer, plan, (rss0, hbm0) = _optimizer_arm(args)
        assert plan.optimizer_offload and plan.applied.option.moves == "A"
        assert args.optimizer_cpu_offload is True and args.optimizer_offload_fraction == 1.0
        inners = _inner_optimizers(optimizer)
        assert inners and all(map(_is_hybrid_device, inners)), (
            "optimizer CPU offload not built on every parameter group: "
            f"{[type(inner).__name__ for inner in inners]}"
        )
        offloaded, _ = _steps(model, ddp, optimizer)
        # Measured after the steps: the optimizer state is allocated at the first step.
        offload_hbm = torch.cuda.memory_allocated() - hbm0
        offload_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0
        overrides = plan.config_overrides()
        adam_grace = plan.applied.adam_grace_bytes
        del model, ddp, optimizer
        get_loop_rsbf_schedule().reset()
        torch.cuda.empty_cache()

        reference_args = _args(**common)
        model, ddp, optimizer, _, (_, hbm0) = _optimizer_arm(reference_args, overrides)
        assert not reference_args.optimizer_cpu_offload
        inners = _inner_optimizers(optimizer)
        assert inners and not any(map(_is_hybrid_device, inners))
        reference, _ = _steps(model, ddp, optimizer)
        reference_hbm = torch.cuda.memory_allocated() - hbm0
        del model, ddp, optimizer
    get_loop_rsbf_schedule().reset()
    first, second = _merged(offloaded[0]), _merged(offloaded[1])
    ref_first, ref_second = _merged(reference[0]), _merged(reference[1])
    assert first == ref_first  # no optimizer step yet: the same computation
    worst = max(abs(second[k] - ref_second[k]) / max(abs(ref_second[k]), 1e-30) for k in second)
    assert worst <= 1e-3, worst
    if dist.get_rank() == 0:
        print(
            f"[forced A] step-2 max relative loss difference {worst:.3e}; optimizer HBM "
            f"{offload_hbm / GB:.4f} GB offloaded vs {reference_hbm / GB:.4f} GB (saved "
            f"{(reference_hbm - offload_hbm) / GB:.4f} GB, planner 12 (P_d + P_e) / N = "
            f"{adam_grace * 12 / 16 / GB:.4f} GB); host RSS +{offload_rss / 1e6:.4f} GB "
            f"(ru_maxrss KB; planner 16 (P_d + P_e) / N = {adam_grace / GB:.4f} GB)"
        )
