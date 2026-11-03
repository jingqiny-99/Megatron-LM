# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Micro-batch-size invariance of the eager hybrid loop step (GPU; FSDP-arm design test T3).

The DSv4 Flash loop experiment may raise the micro-batch size; that is allowed only if the step
is the same function of the global batch at every micro-batch size. This test runs one training
step of a DSv4 loop ``HybridModel`` at test width over the same 4 rows per rank as MBS 1 x 4,
MBS 2 x 2 and MBS 4 x 1 through the real no-pipelining schedule, DDP (fp32 gradient reduction)
and ``finalize_model_grads`` under ``calculate_per_token_loss`` (``mbs_invariance_utils``), with
every loss of the Flash recipe on: the LM loss, MTP (one depth), the router's sequence aux loss
and the CSA indexer's sparse KL loss.

Model: pattern ``'WEWECEHECEHECE/WE'`` (prelude ``WEWECE`` with three hash-routed MoE layers,
core ``HECE`` run exactly 4 times, coda ``HECE``, MTP ``WE``), DSv4 W / C / H attention
(unfused, ``dsa_kernel_backend='none'``, the parity reference's path), 8 experts top-2 with
sqrtsoftplus scores, top-k scaling 1.5, expert bias and one shared expert, mHC with 4 streams,
bf16, deterministic kernels. Runs under torchrun at W in {1, 2, 4, 8} (EP = W; each rank its
own 4 rows).

* ``moe_aux_loss_coeff = 1.0``: the learned routers' weight gradients are dominated by the
  sequence aux loss. The indexer weights get gradients only from the indexer loss (the indexer
  reads detached activations, ``csa.py`` unfused path).
* Gate (bf16, deterministic): every attached loss sum within 1e-3 relative of MBS 1; every
  parameter gradient cosine >= 0.9999 and relative norm difference <= 1e-3.
* Teeth: ``per_microbatch_mean_aux_scaling`` (aux and indexer gradients x 1 / micro-batch size,
  the per-microbatch mean the ``loop_schedule`` docstring describes) must fail the gate on every
  learned router weight and every indexer weight.
"""

import contextlib
import os

import pytest
import torch

from tests.unit_tests.transformer.loop import mbs_invariance_utils as mbs
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    CUDA,
    determinism_env,
    model_parallel_once,
)

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(
    not CUDA or 8 % WORLD != 0, reason="DSv4 / MoE layers need CUDA (EP = W in 1, 2, 4, 8)"
)

PATTERN = "WEWECEHECEHECE/WE"
_RATIOS = {"W": 0, "C": 4, "H": 128, "E": 0}
SEQ = 256
VOCAB = 512
ROWS = 4  # per rank
MIN_COSINE = 0.9999
MAX_RELATIVE = 1e-3


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    if not CUDA:
        yield
        return
    with determinism_env(), model_parallel_once(expert_model_parallel_size=WORLD):
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(False)


@contextlib.contextmanager
def _hadamard():
    """``rotate_activation`` (``dsa.py``) needs ``fast_hadamard_transform``; without it every
    arm uses the same scaled identity."""
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


def _model():
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
        qk_layernorm=True,
        o_groups=8,
        o_lora_rank=128,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        rope_type="rope",
        rotary_base=10000,
        rotary_percent=1.0,
        experimental_attention_variant="dsv4_hybrid",
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=32,
        dsa_indexer_loss_coeff=1e-2,
        dsa_indexer_use_sparse_loss=True,
        csa_window_size=32,
        csa_compress_ratios=[_RATIOS[symbol] for symbol in main + mtp],
        num_moe_experts=8,
        moe_ffn_hidden_size=256,
        moe_shared_expert_intermediate_size=256,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_router_score_function="sqrtsoftplus",
        moe_router_topk_scaling_factor=1.5,
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=1e-3,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=1.0,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_n_hash_layers=3,
        actual_vocab_size=VOCAB,
        activation_func_clamp_value=10.0,
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
        gradient_accumulation_fusion=True,
        loop_architecture="huginn",
        loop_carrier="injection",
        loop_iterations=4,
        loop_exit_criterion="fixed",
        loop_prelude_sublayers=6,
        loop_coda_sublayers=4,
        calculate_per_token_loss=True,
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


def learned_router_and_indexer_names(model):
    """The weights of the learned routers (the hash layers' routers route by a fixed table and
    have no aux loss) and every parameter of every DSA / CSA indexer."""
    from megatron.core.transformer.experimental_attention_variant.csa import CSAIndexer

    routers, indexers = [], []
    for module_name, module in model.named_modules():
        if hasattr(module, "is_hash_layer") and not module.is_hash_layer:
            routers.append(f"{module_name}.weight")
        if isinstance(module, CSAIndexer):
            indexers.extend(f"{module_name}.{name}" for name, _ in module.named_parameters())
    return routers, indexers


@pytest.fixture(scope="module")
def runner():
    with _hadamard():
        model = _model()
    return mbs.StepRunner(model)


def _arms(runner, mutant=False):
    rows = mbs.make_rows(ROWS, SEQ, VOCAB, seed=4242 + torch.distributed.get_rank())
    results = {}
    for micro_batch_size in (1, 2, 4):
        mutation = (
            mbs.per_microbatch_mean_aux_scaling(micro_batch_size)
            if mutant
            else contextlib.nullcontext()
        )
        with _hadamard(), mutation:
            results[micro_batch_size] = runner.run(rows, micro_batch_size, seq=SEQ)
    return results


def _print_report(tag, micro_batch_size, loss_rel, report):
    print(
        f"[T3 rank {torch.distributed.get_rank()} W {WORLD}] {tag} MBS {micro_batch_size} vs 1: "
        f"loss rel {loss_rel}; params {len(report)}, "
        f"failing {sum(not r['ok'] for r in report.values())}, "
        f"min cosine {min(r['cosine'] for r in report.values()):.8f}, "
        f"max rel {max(r['relative'] for r in report.values()):.3e}",
        flush=True,
    )


def test_step_is_invariant_to_micro_batch_size(runner):
    results = _arms(runner)
    ref_losses, ref_grads = results[1]
    routers, indexers = learned_router_and_indexer_names(runner.model)
    # The gate must not be vacuous: every loss is on and the checked weights train.
    assert all(ref_losses[key] != 0.0 for key in mbs.LOSS_KEYS), ref_losses
    assert routers and indexers
    for name in routers + indexers:
        assert ref_grads[name].norm() > 0, name
    for micro_batch_size in (2, 4):
        losses, grads = results[micro_batch_size]
        loss_rel = mbs.relative_loss_differences(losses, ref_losses)
        report = mbs.compare_gradients(
            grads, ref_grads, min_cosine=MIN_COSINE, max_relative=MAX_RELATIVE
        )
        _print_report("invariance", micro_batch_size, loss_rel, report)
        assert all(value <= 1e-3 for value in loss_rel.values()), loss_rel
        failing = [name for name, r in report.items() if not r["ok"]]
        assert not failing, mbs.worst(report, failing)


def test_per_microbatch_mean_aux_scaling_is_caught(runner):
    """The 1 / b mutation fails on every learned router weight and every indexer weight."""
    results = _arms(runner, mutant=True)
    _, ref_grads = results[1]
    routers, indexers = learned_router_and_indexer_names(runner.model)
    for micro_batch_size in (2, 4):
        _, grads = results[micro_batch_size]
        report = mbs.compare_gradients(
            grads, ref_grads, min_cosine=MIN_COSINE, max_relative=MAX_RELATIVE
        )
        _print_report("mutant", micro_batch_size, {}, report)
        passing = [name for name in routers + indexers if report[name]["ok"]]
        assert not passing, mbs.worst(report, passing)
