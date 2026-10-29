# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""In-process resume continuity of the hybrid loop on DSV4-TINY-like models (GPU, M3c).

Runs under torchrun with CUDA at W = 8, 4 or 2 ranks (EP = W, one EU per rank, model parallel
initialized once for the module). The model is the DSV4-TINY loop rung at test size
(``'HEHECEHECEHECE/WE'``: DSv4 H / C / W attention with TE linears, unfused attention,
``dsa_kernel_backend='none'``, 8 experts with the all-to-all dispatcher, a hash layer, expert
bias, mHC 4 streams, MTP 1, bf16; S 256, ``PER_RANK`` samples per rank) and its variants: mHC
without MTP, and the carrier ablation's ``loop-entry`` shape (``TINY-plain``: no mHC, no MTP,
no hash layer) with and without ``loop_sample_depth``. Training goes through DDP (fp32 gradient
reduction), the distributed optimizer and ``finalize_model_grads`` (which updates the router's
expert bias), with ``torch.use_deterministic_algorithms`` set once for the module.

Each arm trains an **uninterrupted** run for ``2 K`` steps and a **resumed** one: ``K`` steps,
``save_checkpoint`` (torch_dist, mock training arguments carrying the model config's ``loop_*``
values and pattern, ``store_ckpt_gpu_utils``), new model / DDP / optimizer objects whose weights
and router expert biases are perturbed, the RNG (CPU, CUDA, tracker) scrambled, the
RSBF schedule, planner session and exit-histogram log reset ("a fresh process"),
``load_checkpoint`` (training resume: the weights, the router expert bias, the optimizer, the
per-rank RNG state), then steps ``K + 1 .. 2 K``. Gate: every resumed step's per-microbatch
loss reports and the grad norm equal the uninterrupted run's **bitwise**; the steps before the
save equal it too (the run itself is deterministic in process).

Arms: the eager ``LoopHybridStack`` through the standard schedule (TINY, mHC without MTP,
TINY-plain, TINY-plain with ``loop_sample_depth``, whose depth draws use the CPU default
generator restored by the checkpoint's RNG state); the RSBF executor (sampled R4-uniform depths,
counter-based on (seed, iteration, sample)) with the whole-round, block (stride 2) and nested
(stride 2) items. The store arms are ``test_loop_resume_continuity_store.py``; fresh-process
chains through ``pretrain_hybrid.py`` are the toolkit's ``scripts/loop_ckpt_chain.sh``.
"""

import contextlib
import functools
import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.loop.rsbf_executor import (
    RSBF_BATCH_FUNC_ATTR,
    LoopMicrobatch,
    get_loop_rsbf_schedule,
)
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    CUDA,
    VOCAB,
    determinism_env,
    model_parallel_once,
)

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
GPU_EP = CUDA and WORLD >= 2 and 8 % WORLD == 0
pytestmark = pytest.mark.skipif(
    not GPU_EP, reason="DSV4-TINY at EP = W needs CUDA and 2, 4 or 8 ranks"
)

PATTERN = "HEHECEHECEHECE/WE"
_RATIOS = {"W": 0, "C": 4, "H": 128, "E": 0}
SEQ = 256
PER_RANK = 4
K = 3  # steps before the save; the uninterrupted run trains 2 K

VARIANTS = {
    "tiny": dict(),
    "tiny_nomtp": dict(mtp=False),
    "tiny_plain": dict(mtp=False, mhc=False, hash_layers=0),
}
ARMS = {
    # name: (variant, schedule, config overrides)
    "eager_tiny": ("tiny", "standard", {}),
    "eager_tiny_nomtp": ("tiny_nomtp", "standard", {}),
    "eager_plain": ("tiny_plain", "standard", {}),
    "eager_plain_sampled_depth": ("tiny_plain", "standard", dict(loop_sample_depth=True)),
    "rsbf_whole_round": ("tiny", "rsbf", {}),
    "rsbf_block2": ("tiny", "rsbf", dict(loop_rsbf_item_form="block", loop_rsbf_block_stride=2)),
    "rsbf_nested2": ("tiny", "rsbf", dict(loop_rsbf_item_form="nested", loop_rsbf_block_stride=2)),
}


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    if not GPU_EP:
        yield
        return
    with determinism_env(), model_parallel_once(expert_model_parallel_size=WORLD):
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(False)
            _reset_process_state()


def _reset_process_state():
    """What a new training process starts without: the RSBF schedule's cached executors and
    iteration, the planner session, the exit-histogram log."""
    from megatron.training import loop_rsbf_planning as planning

    get_loop_rsbf_schedule().reset()
    planning.reset_loop_planner_session()
    planning.reset_loop_exit_histogram_log()


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


def tiny_model(variant, schedule, overrides, *, world=WORLD):
    """A DSV4-TINY-like loop ``HybridModel`` (``variant``: ``VARIANTS``) on the ``schedule``
    (``standard``: the eager stack, fixed depth unless ``loop_sample_depth``; ``rsbf``: sampled
    R4-uniform depths), EP = ``world``."""
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_dsv4_stack_spec
    from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    shape = VARIANTS[variant]
    mtp, mhc = shape.get("mtp", True), shape.get("mhc", True)
    main = PATTERN.split("/")[0]
    pattern = PATTERN if mtp else main
    symbols = main + ("WE" if mtp else "")
    rsbf = schedule == "rsbf"
    fields = dict(
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
        csa_compress_ratios=[_RATIOS[symbol] for symbol in symbols],
        num_moe_experts=8,
        moe_ffn_hidden_size=256,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=1e-3,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=1e-4,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_n_hash_layers=shape.get("hash_layers", 1),
        actual_vocab_size=VOCAB,
        enable_mhc_connections=mhc,
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
        loop_schedule=schedule,
        calculate_per_token_loss=True,
        deterministic_mode=True,
        attention_backend=AttnBackend.unfused,
        expert_model_parallel_size=world,
    )
    if mhc:
        fields.update(mhc_num_residual_streams=4, mhc_sinkhorn_iterations=5)
    if mtp:
        fields.update(mtp_num_layers=1, mtp_loss_scaling_factor=0.1)
    if rsbf:
        fields.update(
            loop_exit_criterion="sampled",
            loop_depth_probs=[0.25, 0.25, 0.25, 0.25],
            loop_depth_seed=1234,
        )
    fields.update(overrides)
    config = MLATransformerConfig(**fields)
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    set_moe_metrics_tracker(MoEMetricsTracker())
    return HybridModel(
        config=config,
        hybrid_stack_spec=loop_hybrid_dsv4_stack_spec(config),
        vocab_size=VOCAB,
        max_sequence_length=1024,
        hybrid_layer_pattern=pattern,
        position_embedding_type="rope",
    ).cuda()


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


def microbatches(iteration, *, per_rank=PER_RANK):
    """This rank's slice of the global batch of ``iteration`` (one row per microbatch; a
    function of the iteration only: the data order a resume restores)."""
    total = per_rank * WORLD
    generator = torch.Generator().manual_seed(100 + iteration)
    tokens = torch.randint(0, VOCAB, (total, SEQ), generator=generator)
    labels = torch.randint(0, VOCAB, (total, SEQ), generator=generator)
    loss_mask = (torch.rand(total, SEQ, generator=generator) > 0.1).float()
    rank = dist.get_rank()
    return [
        LoopMicrobatch(
            tokens=tokens[i : i + 1].cuda(),
            position_ids=torch.arange(SEQ)[None].cuda(),
            labels=labels[i : i + 1].cuda(),
            loss_mask=loss_mask[i : i + 1].cuda(),
            attention_mask=None,
            padding_mask=None,
            loss_func=_loss_func,
        )
        for i in range(rank, total, WORLD)
    ]


def _eager_forward_step(data_iterator, model):
    batch = next(data_iterator)
    output = model(
        batch.tokens, batch.position_ids, None, labels=batch.labels, loss_mask=batch.loss_mask
    )
    return output, functools.partial(_loss_func, batch.loss_mask)


def _rsbf_forward_step():
    def forward_step(*_args, **_kwargs):  # pragma: no cover - the RSBF schedule calls the batch
        raise AssertionError("the RSBF schedule calls the batch function")

    setattr(forward_step, RSBF_BATCH_FUNC_ATTR, lambda iterator, _model: next(iterator))
    return forward_step


def build_run(model, *, perturb=None):
    """DDP (fp32 gradient reduction) and the distributed optimizer around ``model``;
    ``perturb``: overwrite every parameter and every router ``expert_bias`` buffer with seeded
    values first (a fresh process's other initialization, so a weight or an expert bias the load
    misses is visible: an unperturbed bias missed by the load would only show if a top-k
    selection flipped). The other persistent buffers (the hash layer's fixed token-to-expert
    table) are left as built."""
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
        finalize_model_grads,
    )
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.process_groups_config import ProcessGroupCollection

    if perturb is not None:
        with torch.no_grad():
            for index, (_, param) in enumerate(sorted(model.named_parameters())):
                generator = torch.Generator().manual_seed(perturb * 1000 + index)
                noise = 0.02 * torch.randn(param.shape, generator=generator)
                param.copy_(noise.to(param.dtype))
            biases = [(n, b) for n, b in model.named_buffers() if n.endswith("expert_bias")]
            assert biases or not model.config.moe_router_enable_expert_bias
            for index, (_, bias) in enumerate(sorted(biases, key=lambda item: item[0])):
                generator = torch.Generator().manual_seed(perturb * 1000 + 500 + index)
                bias.copy_(0.05 * torch.randn(bias.shape, generator=generator).to(bias.dtype))
    full = ProcessGroupCollection.use_mpu_process_groups()
    ddp = DistributedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            overlap_grad_reduce=False, use_distributed_optimizer=True, grad_reduce_in_fp32=True
        ),
        model,
        pg_collection=full,
    )
    model.config.finalize_model_grads_func = finalize_model_grads
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=1e-4,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
            clip_grad=1.0,
        ),
        [ddp],
        use_gloo_process_groups=False,
        pg_collection=full,
    )
    return dict(model=model, ddp=ddp, optimizer=optimizer, full=full)


def train_steps(run, iterations, *, per_rank=PER_RANK):
    """One optimizer step per iteration of ``iterations``; returns per step the per-microbatch
    loss reports (this rank's, as lists) and the grad norm."""
    from megatron.core.pipeline_parallel import get_forward_backward_func

    config = run["model"].config
    rsbf = config.loop_schedule == "rsbf"
    schedule = get_loop_rsbf_schedule()
    records = []
    for iteration in iterations:
        run["ddp"].zero_grad_buffer()
        run["optimizer"].zero_grad()
        batches = microbatches(iteration, per_rank=per_rank)
        if rsbf:
            schedule.set_iteration(iteration)
            reports = schedule(
                forward_step_func=_rsbf_forward_step(),
                data_iterator=iter(batches),
                model=[run["ddp"]],
                num_microbatches=len(batches),
                seq_length=SEQ,
                micro_batch_size=1,
            )
        else:
            reports = get_forward_backward_func()(
                forward_step_func=_eager_forward_step,
                data_iterator=iter(batches),
                model=[run["ddp"]],
                num_microbatches=len(batches),
                seq_length=SEQ,
                micro_batch_size=1,
                forward_only=False,
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


def scramble_rng():
    """A fresh process's RNG (CPU, CUDA, the model-parallel tracker): differently seeded."""
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    torch.manual_seed(97531)
    model_parallel_cuda_manual_seed(97531)


def resume_arm(tmp_path, name, make_model, *, save_kwargs=None, load_kwargs=None):
    """``(expected, early, late)``: the uninterrupted run's ``2 K`` step records, the resumed
    run's first ``K`` and, after save / fresh objects / load, its last ``K``. ``save_kwargs``:
    ``mock_checkpoint_args`` keywords; ``load_kwargs``: argument overrides for the load only.
    The mock arguments carry the model config's ``loop_*`` values and its layer pattern, so the
    load's resume check (message 5) runs on a loop run's real arguments."""
    from tests.unit_tests.transformer.loop import store_ckpt_gpu_utils as cku

    _reset_process_state()
    with _hadamard_fallback():
        reference = build_run(make_model())
        expected = train_steps(reference, range(1, 2 * K + 1))
        del reference
        _reset_process_state()
        first = build_run(make_model())
        early = train_steps(first, range(1, K + 1))
        with cku.checkpoint_dir(tmp_path / name) as ckpt_dir:
            with cku.mock_checkpoint_args(
                ckpt_dir,
                loop_config=first["model"].config,
                hybrid_layer_pattern=first["model"].hybrid_layer_pattern,
                **(save_kwargs or {}),
            ) as args:
                cku.save(first["ddp"], first["optimizer"], first["full"], iteration=K)
                del first
                _reset_process_state()
                resumed = build_run(make_model(), perturb=7)
                scramble_rng()
                for key, value in (load_kwargs or {}).items():
                    setattr(args, key, value)
                assert cku.load(resumed["ddp"], resumed["optimizer"], resumed["full"]) == K
        late = train_steps(resumed, range(K + 1, 2 * K + 1))
    torch.cuda.empty_cache()
    return expected, early, late


def assert_continuity(expected, early, late):
    assert early == expected[:K], "the run before the save is not deterministic in process"
    for offset, (got, want) in enumerate(zip(late, expected[K:])):
        iteration = K + 1 + offset
        assert got["losses"] == want["losses"], f"iteration {iteration}: losses differ"
        assert got["grad_norm"] == want["grad_norm"], f"iteration {iteration}: grad norm differs"


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_resume_continues_bitwise(tmp_path_dist_ckpt, arm):
    variant, schedule, overrides = ARMS[arm]
    expected, early, late = resume_arm(
        tmp_path_dist_ckpt, f"resume_{arm}", lambda: tiny_model(variant, schedule, overrides)
    )
    assert_continuity(expected, early, late)


def test_sampled_depth_resume_needs_the_rng_state(tmp_path_dist_ckpt):
    """The control of the sampled-depth arm: with ``--no-load-rng`` the resumed eager run draws
    other depths (the CPU default generator is the scrambled one), so its losses differ from the
    uninterrupted run's; with the RNG state they are equal (``test_resume_continues_bitwise``)."""
    expected, _, late = resume_arm(
        tmp_path_dist_ckpt,
        "resume_sampled_depth_no_rng",
        lambda: tiny_model("tiny_plain", "standard", dict(loop_sample_depth=True)),
        load_kwargs=dict(no_load_rng=True),
    )
    assert [step["losses"] for step in late] != [step["losses"] for step in expected[K:]]
