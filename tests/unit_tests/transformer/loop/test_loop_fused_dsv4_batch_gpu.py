# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The fused DSv4 kernels at micro-batch size b > 1, Flash widths (GPU; FSDP-arm design T7).

The DSv4 Flash loop runs the fused kernels (``dsa_kernel_backend='cudnn'``: the fused CSA
indexer + sparse attention with its indexer loss, the fused sparse attention of the H / W
layers, the fused mHC kernels) at the micro-batch size the arms choose. This test checks that a
b-row microbatch gives the step of b one-row microbatches over the same rows: one training step
of a Flash-width loop slice through the real no-pipelining schedule, DDP (fp32 gradient
reduction) and ``finalize_model_grads`` under ``calculate_per_token_loss``
(``mbs_invariance_utils``), b = 4 (4 x 1 vs 1 x 4) and b = 8 (8 x 1 vs 1 x 8; 32,768 query rows
per fused indexer call, the fused indexer's verified-safe row limit).

Slice (one GPU, EP 1; ``loop_hybrid_dsv4_stack_spec``): DSv4 Flash widths (hidden 4096, 64
heads, q LoRA 1024, v head 512, rope head 64, output groups 8 x LoRA 1024, qk layernorm; CSA
window 128; indexer 64 x 128, top-512, sparse loss 1e-2), pattern ``'WEWECEHECEHECE'``: prelude
``WEWECE`` (three hash-routed MoE layers), core ``HECE`` run exactly 4 times, coda ``HECE``; no
MTP. MoE 16 experts (FFN 2048) + one shared expert, top-6 sqrtsoftplus, top-k scaling 1.5,
expert bias, sequence aux loss 1e-4, clamp 10; mHC 4 streams (Sinkhorn 20, fused); bf16;
``loop_recompute`` (round checkpoints) so b = 8 fits one GB200; sequence 4096; vocabulary 4096
(the heads are not under test).

Gate (bf16): every attached loss sum (LM, aux, indexer) within 1e-3 relative; every parameter
gradient cosine >= 0.9999 and relative norm difference <= 1e-3, including the indexer
(``linear_wq_b``, ``linear_weights_proj``, its compressor) and the attention compressors
(``linear_wkv``, ``linear_wgate``, ``ape``, ``norm``). Both with deterministic kernels and with
training's non-deterministic ones (``deterministic_mode`` False, as the recipes run); the latter
also reports the noise floor (the 1-row arm against itself).

Recorded (printed, not gated): the batch sizes the fused CSA indexer + sparse attention ran at,
that the unfused CSA path never ran, and whether the DSA cuDNN sparse indexer loss path
(``dsa_cudnn_kernels._compute_sparse_indexer_loss_and_grads``, whose b > 1 top-k offset site
is the known 'D'-path issue) was reached; Flash builds only W / C / H, so it should not be.
"""

import collections
import os
import warnings

import pytest
import torch

from tests.unit_tests.transformer.loop import mbs_invariance_utils as mbs
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA, model_parallel_once

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(
    not CUDA or WORLD != 1, reason="one-GPU Flash-width slice (needs CUDA and W = 1)"
)

PATTERN = "WEWECEHECEHECE"
_RATIOS = {"W": 0, "C": 4, "H": 128, "E": 0}
SEQ = 4096
VOCAB = 4096
MIN_COSINE = 0.9999
MAX_RELATIVE = 1e-3


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    if not CUDA or WORLD != 1:
        yield
        return
    with model_parallel_once(expert_model_parallel_size=1):
        yield


def _model():
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_dsv4_stack_spec
    from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    config = MLATransformerConfig(
        num_layers=len(PATTERN),
        hidden_size=4096,
        num_attention_heads=64,
        kv_channels=512,
        multi_latent_attention=True,
        q_lora_rank=1024,
        kv_lora_rank=448,
        qk_head_dim=448,
        qk_pos_emb_head_dim=64,
        v_head_dim=512,
        qk_layernorm=True,
        o_groups=8,
        o_lora_rank=1024,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        rotary_base=10000,
        rotary_percent=1.0,
        rotary_scaling_factor=4,
        mscale=1.0,
        mscale_all_dim=1.0,
        original_max_position_embeddings=65536,
        experimental_attention_variant="dsv4_hybrid",
        dsa_kernel_backend="cudnn",
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=512,
        dsa_indexer_loss_coeff=1e-2,
        dsa_indexer_use_sparse_loss=True,
        csa_window_size=128,
        csa_compress_ratios=[_RATIOS[symbol] for symbol in PATTERN],
        csa_compress_rotary_base=40000,
        num_moe_experts=16,
        moe_ffn_hidden_size=2048,
        moe_shared_expert_intermediate_size=2048,
        moe_router_topk=6,
        moe_router_dtype="fp32",
        moe_router_score_function="sqrtsoftplus",
        moe_router_topk_scaling_factor=1.5,
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=1e-3,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=1e-4,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_n_hash_layers=3,
        actual_vocab_size=VOCAB,
        activation_func_clamp_value=10.0,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=20,
        use_fused_mhc=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        init_method_std=0.02,
        bf16=True,
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=True,
        loop_architecture="huginn",
        loop_carrier="injection",
        loop_iterations=4,
        loop_exit_criterion="fixed",
        loop_prelude_sublayers=6,
        loop_coda_sublayers=4,
        loop_recompute=True,
        calculate_per_token_loss=True,
        deterministic_mode=False,
        attention_backend=AttnBackend.auto,
    )
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    set_moe_metrics_tracker(MoEMetricsTracker())
    return HybridModel(
        config=config,
        hybrid_stack_spec=loop_hybrid_dsv4_stack_spec(config),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        hybrid_layer_pattern=PATTERN,
        position_embedding_type="rope",
    ).cuda()


@pytest.fixture(scope="module")
def runner():
    # Per-arm gradients go to host memory: three fp32 copies of a 3.8 B-parameter slice do not
    # fit next to the b = 8 activations.
    return mbs.StepRunner(_model(), grads_device="cpu")


class _KernelLog:
    """Counts of the CSA / DSA paths the step ran, by batch size."""

    def __init__(self, monkeypatch):
        from megatron.core.transformer.experimental_attention_variant import csa, dsa_cudnn_kernels

        self.fused_indexer_batches = collections.Counter()
        self.unfused_csa = 0
        self.dsa_sparse_indexer_loss = 0
        fused = csa.fused_csa_indexer_sparse_attn
        unfused = csa.CompressedSparseAttention._forward_unfused_csa
        dsa_loss = dsa_cudnn_kernels._compute_sparse_indexer_loss_and_grads

        def fused_wrapper(query, *args, **kwargs):
            self.fused_indexer_batches[int(query.shape[1])] += 1
            return fused(query, *args, **kwargs)

        def unfused_wrapper(attention, *args, **kwargs):
            self.unfused_csa += 1
            return unfused(attention, *args, **kwargs)

        def dsa_loss_wrapper(*args, **kwargs):
            self.dsa_sparse_indexer_loss += 1
            return dsa_loss(*args, **kwargs)

        monkeypatch.setattr(csa, "fused_csa_indexer_sparse_attn", fused_wrapper)
        monkeypatch.setattr(csa.CompressedSparseAttention, "_forward_unfused_csa", unfused_wrapper)
        monkeypatch.setattr(
            dsa_cudnn_kernels, "_compute_sparse_indexer_loss_and_grads", dsa_loss_wrapper
        )


def _checked_weight_names(model):
    """The indexer and compressor weights the gate must cover (design T7)."""
    names = [name for name, _ in model.named_parameters()]
    wanted = (
        ".indexer.linear_wq_b.",
        ".indexer.linear_weights_proj.",
        "compressor.linear_wkv.",
        "compressor.linear_wgate.",
        "compressor.ape",
        "compressor.norm.",
    )
    return {key: [name for name in names if key in name] for key in wanted}


@pytest.mark.parametrize("deterministic", [True, False], ids=["deterministic", "training"])
@pytest.mark.parametrize("batch", [4, 8])
def test_fused_kernels_batch_equals_single_rows(runner, monkeypatch, batch, deterministic):
    from megatron.training.determinism import DETERMINISM_ENV_VAR_DEFAULTS

    if deterministic:
        for name, value in DETERMINISM_ENV_VAR_DEFAULTS.items():
            monkeypatch.setenv(name, value)
    else:
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
    monkeypatch.setattr(runner.model.config, "deterministic_mode", deterministic)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    log = _KernelLog(monkeypatch)
    rows = mbs.make_rows(batch, SEQ, VOCAB, seed=777)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            singles = runner.run(rows, 1, seq=SEQ)
            batched = runner.run(rows, batch, seq=SEQ)
            floor = None if deterministic else runner.run(rows, 1, seq=SEQ)
    finally:
        torch.use_deterministic_algorithms(previous)
    ref_losses, ref_grads = singles
    losses, grads = batched
    loss_rel = mbs.relative_loss_differences(losses, ref_losses)
    report = mbs.compare_gradients(
        grads, ref_grads, min_cosine=MIN_COSINE, max_relative=MAX_RELATIVE
    )
    fallbacks = [str(w.message) for w in caught if "falling back" in str(w.message)]
    tag = f"b={batch} {'deterministic' if deterministic else 'training'}"
    print(
        f"[T7] {tag}: fused CSA indexer calls by batch {dict(log.fused_indexer_batches)}; "
        f"unfused CSA calls {log.unfused_csa}; DSA cuDNN sparse indexer-loss calls "
        f"(the 'D' path with the b > 1 offset site) {log.dsa_sparse_indexer_loss}; "
        f"fallback warnings {fallbacks}",
        flush=True,
    )
    print(
        f"[T7] {tag}: loss rel {loss_rel}; params {len(report)}, failing "
        f"{sum(not r['ok'] for r in report.values())}, min cosine "
        f"{min(r['cosine'] for r in report.values()):.8f}, max rel "
        f"{max(r['relative'] for r in report.values()):.3e}; worst: {mbs.worst(report)}",
        flush=True,
    )
    if floor is not None:
        floor_report = mbs.compare_gradients(
            floor[1], ref_grads, min_cosine=MIN_COSINE, max_relative=MAX_RELATIVE
        )
        print(
            f"[T7] {tag}: noise floor (1-row arm twice): loss rel "
            f"{mbs.relative_loss_differences(floor[0], ref_losses)}; min cosine "
            f"{min(r['cosine'] for r in floor_report.values()):.8f}, max rel "
            f"{max(r['relative'] for r in floor_report.values()):.3e}; worst: "
            f"{mbs.worst(floor_report)}",
            flush=True,
        )
    # The fused path ran at both batch sizes and the unfused CSA path never did.
    assert log.fused_indexer_batches[1] > 0 and log.fused_indexer_batches[batch] > 0
    assert log.unfused_csa == 0
    # The gate covers the indexer and compressor weights, and they train.
    for key, names in _checked_weight_names(runner.model).items():
        assert names, key
        for name in names:
            assert ref_grads[name].norm() > 0, name
    assert ref_losses["lm"] != 0.0 and ref_losses["aux"] != 0.0 and ref_losses["indexer"] != 0.0
    assert all(value <= 1e-3 for value in loss_rel.values()), loss_rel
    failing = [name for name, r in report.items() if not r["ok"]]
    assert not failing, mbs.worst(report, failing)
