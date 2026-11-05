# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The fused DSv4 kernels at micro-batch size b > 1, Flash widths (GPU; FSDP-arm design T7).

The DSv4 Flash loop runs the fused kernels (``dsa_kernel_backend='cudnn'``: the fused CSA
indexer + sparse attention with its indexer loss, the fused sparse attention of the H / W
layers, the fused mHC kernels) at the micro-batch size the arms choose. This test checks that a
b-row microbatch gives the step of b one-row microbatches over the same rows: one training step
of a Flash-width loop slice through the real no-pipelining schedule, DDP (fp32 gradient
reduction) and ``finalize_model_grads`` under ``calculate_per_token_loss``
(``mbs_invariance_utils``), b = 4 (1 x 4 vs 4 x 1) and b = 8 (1 x 8 vs 8 x 1; 32,768 query rows
per fused indexer call, the fused indexer's verified-safe row limit).

Slice (one GPU, EP 1; ``loop_hybrid_dsv4_stack_spec``): DSv4 Flash widths (hidden 4096, 64
heads, q LoRA 1024, v head 512, rope head 64, output groups 8 x LoRA 1024, qk layernorm; CSA
window 128; indexer 64 x 128, top-512, sparse loss 1e-2), pattern ``'WEWECEHECEHECE'``: prelude
``WEWECE`` (three hash-routed MoE layers), core ``HECE`` run exactly 4 times, coda ``HECE``; no
MTP. MoE 16 experts (FFN 2048) + one shared expert, top-6 sqrtsoftplus, top-k scaling 1.5,
expert bias, sequence aux loss 1e-4, SwiGLU with the clamp; mHC 4 streams (Sinkhorn 20, fused);
bf16; ``loop_recompute`` (round checkpoints) so b = 8 fits one GB200; sequence 4096; vocabulary
4096 (the heads are not under test).

The fused kernels are bf16-only, so the comparison is bf16 with a measured floor per parameter
(FSDP-arm deviations D6): the floor is the fused path's own run-to-run difference (the 1-row arm
twice with training's non-deterministic kernels). Gates, with deterministic kernels and with
training's: every attached loss sum (LM, aux, indexer) and every row's LM loss within 1e-3
relative; every parameter gradient (the indexer's ``linear_wq_b``, ``linear_weights_proj`` and
compressor, the attention compressors' ``linear_wkv``, ``linear_wgate``, ``ape`` and ``norm``
included) within max(1e-3, 3 x floor) of the 1-row arm.

The deterministic arms need cuDNN's deterministic DSA sparse-attention backward, which only some
cuDNN Frontend releases provide (``_deterministic_backward_missing``). Where it is missing, the
vendor raises in any deterministic fused backward, so the deterministic arms are not run, the
reason is printed, and the gates apply to training's kernels only: the kernels the fused path
trains with. CONTRACT D2's fused image (cuDNN Frontend 1.28.0) is such an environment, and D2
rules out ``--deterministic-mode`` with the fused path (FSDP-arm deviation D51).

Recorded (printed, not gated): the batch sizes the fused CSA indexer + sparse attention ran at,
that the unfused CSA path never ran, and whether the DSA cuDNN sparse indexer-loss path
(``dsa_cudnn_kernels._compute_sparse_indexer_loss_and_grads``, whose b > 1 top-k offset site
is the known 'D'-path issue) was reached; Flash builds only W / C / H, so it should not be.

The fused path needs a cuDNN Frontend whose compact DSA indexer wrapper takes ``deterministic``
(``TransformerConfig`` refuses ``dsa_kernel_backend='cudnn'`` otherwise); without one the module
is skipped with that error as the reason.
"""

import collections
import inspect
import math
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
MAX_RELATIVE = 1e-3
FLOOR_FACTOR = 3.0


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    if not CUDA or WORLD != 1:
        yield
        return
    with model_parallel_once(expert_model_parallel_size=1):
        yield


def _config():
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    return MLATransformerConfig(
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


def _model(config):
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_dsv4_stack_spec
    from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker

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
    try:
        config = _config()
    except ValueError as exc:
        if "cuDNN Frontend" not in str(exc):
            raise
        import cudnn

        pytest.skip(
            f"fused DSv4 path unavailable in this environment (cudnn-frontend "
            f"{getattr(cudnn, '__version__', '?')}): {exc}"
        )
    # Per-arm gradients go to host memory: several fp32 copies of a 3.8 B-parameter slice do
    # not fit next to the b = 8 activations.
    return mbs.StepRunner(_model(config), grads_device="cpu")


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


def _deterministic_backward_missing():
    """Why the fused DSA backward cannot run deterministically here, or ``None`` if it can.

    The same API check as the vendor's guard, which runs on every deterministic fused backward
    (``fused_sparse_attention._deterministic_sparse_bwd_kwargs``): the cuDNN Frontend's
    ``sparse_attention_backward_wrapper`` must take ``deterministic`` and ``workspace``, and the
    frontend must provide ``SparseAttentionBackward``. Without them the guard raises
    "deterministic mode: the installed cudnn-frontend has no deterministic DSA sparse-attention
    backward", whatever the batch size.
    """
    import cudnn

    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        fused_sparse_attention,
    )

    fused_sparse_attention._ensure_dsa_namespace()
    dsa = fused_sparse_attention._DSA
    wrapper = getattr(dsa, "sparse_attention_backward_wrapper", None)
    try:
        parameters = inspect.signature(wrapper).parameters if wrapper is not None else {}
    except (TypeError, ValueError):
        parameters = {}
    missing = [name for name in ("deterministic", "workspace") if name not in parameters]
    if getattr(dsa, "SparseAttentionBackward", None) is None:
        missing.append("SparseAttentionBackward")
    if not missing:
        return None
    return (
        f"cudnn-frontend {getattr(cudnn, '__version__', '?')} has no deterministic DSA "
        f"sparse-attention backward (missing: {', '.join(missing)})"
    )


def _set_kernels(monkeypatch, runner, deterministic):
    from megatron.training.determinism import DETERMINISM_ENV_VAR_DEFAULTS

    if deterministic:
        for name, value in DETERMINISM_ENV_VAR_DEFAULTS.items():
            monkeypatch.setenv(name, value)
    else:
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
    monkeypatch.setattr(runner.model.config, "deterministic_mode", deterministic)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


@pytest.mark.parametrize("batch", [4, 8])
def test_fused_kernels_batch_equals_single_rows(runner, monkeypatch, batch):
    log = _KernelLog(monkeypatch)
    rows = mbs.make_rows(batch, SEQ, VOCAB, seed=777)
    missing = _deterministic_backward_missing()
    # Training's kernels first, then (where available) the deterministic ones.
    flags = (False,) if missing else (False, True)
    modes = ("training",) if missing else ("deterministic", "training")
    print(
        f"[T7 b={batch}] deterministic arms: "
        + (f"not run: {missing}" if missing else "run")
        + f"; gated modes {modes}",
        flush=True,
    )
    previous = torch.are_deterministic_algorithms_enabled()
    arms = {}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for deterministic in flags:
                _set_kernels(monkeypatch, runner, deterministic)
                mode = "deterministic" if deterministic else "training"
                arms[(mode, 1)] = runner.run(rows, 1, seq=SEQ)
                arms[(mode, batch)] = runner.run(rows, batch, seq=SEQ)
                if not deterministic:
                    arms[(mode, "1 again")] = runner.run(rows, 1, seq=SEQ)
    finally:
        torch.use_deterministic_algorithms(previous)
    fallbacks = sorted({str(w.message)[:160] for w in caught if "falling back" in str(w.message)})
    print(
        f"[T7 b={batch}] fused CSA indexer calls by batch {dict(log.fused_indexer_batches)}; "
        f"unfused CSA calls {log.unfused_csa}; DSA cuDNN sparse indexer-loss calls (the 'D' "
        f"path with the b > 1 offset site) {log.dsa_sparse_indexer_loss}; fallback warnings "
        f"{fallbacks}",
        flush=True,
    )
    floor = mbs.compare_gradients(
        arms[("training", "1 again")][2],
        arms[("training", 1)][2],
        min_cosine=0.0,
        max_relative=math.inf,
    )
    checked = [name for names in _checked_weight_names(runner.model).values() for name in names]
    mbs.print_report(f"T7 b={batch} floor (training, 1-row arm twice)", floor, checked[:6])
    # The fused path ran at both batch sizes and the unfused CSA path never did.
    assert log.fused_indexer_batches[1] > 0 and log.fused_indexer_batches[batch] > 0
    assert log.unfused_csa == 0
    # The gate covers the indexer and compressor weights, and they train.
    ref_losses, _, ref_grads = arms[("training", 1)]
    for key, names in _checked_weight_names(runner.model).items():
        assert names, key
        for name in names:
            assert ref_grads[name].norm() > 0, name
    assert all(ref_losses[key] != 0.0 for key in ("lm", "aux", "indexer")), ref_losses
    failures = []
    for mode in modes:
        ref_losses, ref_rows, ref_grads = arms[(mode, 1)]
        losses, rows_b, grads = arms[(mode, batch)]
        loss_rel = mbs.relative_loss_differences(losses, ref_losses)
        row_rel = mbs.max_row_difference(rows_b, ref_rows)
        report = mbs.resolution_gate(
            mbs.compare_gradients(grads, ref_grads, min_cosine=0.0, max_relative=math.inf),
            floor,
            global_norm=mbs.global_comparison(ref_grads, ref_grads)["norm"],
            strict=False,
            min_relative=MAX_RELATIVE,
            factor=FLOOR_FACTOR,
        )
        tag = f"T7 b={batch} {mode} {batch} x 1 vs 1 x {batch}"
        print(f"[{tag}] loss rel {loss_rel}; max row rel {row_rel:.2e}", flush=True)
        mbs.print_report(tag, report, checked[:6])
        mbs.dump_report(tag, dict(per_param=report, total=mbs.global_comparison(grads, ref_grads)))
        if any(value > 1e-3 for value in loss_rel.values()) or row_rel > 1e-3:
            failures.append(f"{mode}: loss rel {loss_rel}, max row rel {row_rel:.2e}")
        failing = [name for name, r in report.items() if not r["ok"]]
        if failing:
            failures.append(f"{mode}: {len(failing)} params: {mbs.worst(report, failing)}")
    assert not failures, failures
