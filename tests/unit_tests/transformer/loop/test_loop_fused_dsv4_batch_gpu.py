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

Protocol, per b, with training's kernels: R = 5 alternating runs A1 B1 ... A5 B5 of the 1-row arm
A (b microbatches of one row) and the b-row arm B (one microbatch of b rows) in one process,
then one b-row run under the indexer mutant of gate 4. Gates:

1. Paths: the fused CSA indexer + sparse attention ran at batch 1 and at b; the unfused CSA path
   never ran.
2. Losses and rows: every attached loss sum (LM, aux, indexer) and every row's LM loss of every B
   run within 1e-3 (relative) of every A run.
3. Every parameter gradient, on the means over the runs (``mbs_invariance_utils.repeat_gate``):
   x = |mean B - mean A| / |mean A| against the noise of that difference measured from the same
   runs, e = sqrt((s_A^2 + s_B^2) / R) / |mean A| (s^2: the per-run variance within one arm).
   * Resolved gradients (per-run noise sqrt((s_A^2 + s_B^2) / 2) at most 0.5 |mean A|): pass iff
     x <= max(2^-7, 5 e).
   * Unresolved gradients pass iff |mean B - mean A| <= 1e-4 x the whole-gradient norm (mean A).
     The mHC stream-mixing scalars ``alpha_res`` / ``alpha_pre`` are always gated this way: they
     are zero in exact arithmetic at initialization, where the 4 residual streams are copies
     (T3's fp32 arms measured them at 1e-18 to 1e-10), so their bf16 values are rounding
     residuals of the GEMM and reduction shapes that no relative bound can hold (FSDP-arm
     deviation D54).
   * Every indexer and compressor weight (the indexer's ``linear_wq_b``, ``linear_weights_proj``
     and compressor; the attention compressors' ``linear_wkv``, ``linear_wgate``, ``ape`` and
     ``norm``) must be resolved, so that the weights this test is about keep the relative gate.
   * The TE RMSNorm weights are compared on an fp32 side channel instead of ``main_grad`` (below),
     after checking in every run that it equals ``main_grad`` within 2^-7.
4. Teeth: in one more b-row run, every fused CSA call of the step (the recompute replays
   included) halves the gradients of the last sample's indexer inputs (``q_indexer``,
   ``k_indexer``, ``weights``). Gated like mean B in gate 3, against mean A with one run's B noise,
   it must fail on every indexer weight of the C layers that run once per sample (the prelude's
   and the coda's: layers 4 and 12, 12 weights; the core's C layer is recorded only); the test
   fails if it passes on any of them. The losses, the rows and the whole gradient stay within
   their clean spread under this error, so only a per-parameter gate can catch it.

Why repeated runs, not a one-pair floor. With training's kernels the fused DSA backward is not
deterministic (cuDNN Frontend 1.28.0, CONTRACT D2's image), so every gradient carries run-to-run
noise and the b-row difference must be judged against it. The previous gate used one draw of
that noise (the 1-row arm twice) as the floor: relative <= max(1e-3, 3 x floor). For the small
mHC tensors (the 1-element ``alpha_post``, the biases) the run-to-run differences span one to two
orders of magnitude, so that floor was a lottery: it failed 2 to 10 parameters in every one of 23
(A, B, A) triples of correct runs, a different set each time. Means over R = 5 runs and the noise
of their difference measured from the same runs passed all 7 fused run sets of the lead's check
(2 for calibration, 5 on held-out rows; at most 0.67 of the bound), while the mutant of gate 4
failed every one of its 12 weights by 1.38x or more (lead check "T7 verdict", 2026-11-05).

Why an fp32 side channel for the TE RMSNorm weights. TE's RMSNorm backward reduces the weight
gradient over the rows in fp32 inside the kernel and rounds it to the bf16 weight dtype once per
call; autograd sums a microbatch's calls in bf16, and DDP adds that sum to the fp32 ``main_grad``
once per microbatch. Arm A therefore holds sum_i bf16(G_i) over its one-row microbatches and arm
B holds bf16(sum_i G_i): two roundings of a large row reduction into a small tensor, at different
granularity, which the runs' noise does not show (layer 4's indexer compressor norm, 128
elements: 1.6e-3 at a per-run noise of 1e-4, and the same on the unfused path). Accumulated in
fp32 the difference is at the noise (7e-6), and replaying the bf16 steps reproduces the measured
``main_grad`` bit for bit. ``mbs_invariance_utils.RMSNormWeightSideChannel`` accumulates
sum_rows (x * rstd) * dy in fp32 (fp64 sums) in every call of the op's backward, from its saved
tensors, before TE's own backward runs: a test-only patch that changes no gradient. A bf16
allowance in the bound would loosen every norm weight instead.

The constants: 2^-7 is one bf16 unit in the last place; it must exceed the one legitimate shape
effect measured, layer 12's ``attn_sink`` (3.2e-3 to 4.5e-3 between b-row and 1-row steps, on the
unfused reference path as well). With the factor 5 the clean runs stay at most at 0.67 of their
bound and the mutant at 1.38x or more of it.

Only training's kernels run. The gate measures their run-to-run noise; cuDNN Frontend 1.28.0 has
no deterministic DSA sparse-attention backward, and CONTRACT D2 rules out ``--deterministic-mode``
with the fused path. A deterministic fused backward would leave no noise to measure: it would
need a bound against a reference difference instead.

Recorded (printed, not gated): every run's losses, time and peak memory; the batch sizes the
fused CSA indexer + sparse attention ran at; whether the DSA cuDNN sparse indexer-loss path
(``dsa_cudnn_kernels._compute_sparse_indexer_loss_and_grads``, whose b > 1 top-k offset site is
the known 'D'-path issue) was reached (Flash builds only W / C / H, so it should not be); the
compressor norms' gate on ``main_grad`` next to their side-channel gate; the mutant's effect on
every indexer weight. Cost per b: 11 steps and 11 host copies of the fp32 gradient of this
3.9 B-element slice (about 170 GB of host memory).

The fused path needs a cuDNN Frontend whose compact DSA indexer wrapper takes ``deterministic``
(``TransformerConfig`` refuses ``dsa_kernel_backend='cudnn'`` otherwise); without one the module
is skipped with that error as the reason.
"""

import collections
import functools
import math
import os
import time
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
PRELUDE_SUBLAYERS = 6
CODA_SUBLAYERS = 4
_RATIOS = {"W": 0, "C": 4, "H": 128, "E": 0}
SEQ = 4096
VOCAB = 4096
ROW_SEED = 777

# The gate (module docstring).
REPEATS = 5  # R: runs of each arm, alternating A1 B1 ... A5 B5
GATE_FACTOR = 5.0  # a resolved gradient passes iff x <= max(MIN_RELATIVE, GATE_FACTOR x e)
MIN_RELATIVE = 2.0**-7  # one bf16 unit in the last place
UNRESOLVED_NOISE = 0.5  # per-run noise above this fraction of |mean A|: unresolved
ABSOLUTE_FRACTION = 1e-4  # unresolved: |mean B - mean A| <= this x the whole-gradient norm
LOSS_ROW_MAX_RELATIVE = 1e-3  # every loss sum and row LM loss, every B run vs every A run
SIDE_CHANNEL_MAX_RELATIVE = 2.0**-7  # |main_grad - side channel| / |side channel|, every run
MUTANT_INDEXER_SCALE = 0.5  # the teeth: the last sample's indexer-input gradients halved
# The mHC stream-mixing scalars, zero in exact arithmetic at initialization (gate 3).
SYMMETRIC_ZERO_SUFFIXES = (".hyper_connection.alpha_res", ".hyper_connection.alpha_pre")
# The indexer weights of one C layer (the teeth, gate 4).
INDEXER_WEIGHTS = (
    ".indexer.linear_wq_b.",
    ".indexer.linear_weights_proj.",
    ".indexer.compressor.linear_wkv.",
    ".indexer.compressor.linear_wgate.",
    ".indexer.compressor.ape",
    ".indexer.compressor.norm.",
)


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
        loop_prelude_sublayers=PRELUDE_SUBLAYERS,
        loop_coda_sublayers=CODA_SUBLAYERS,
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
    """The slice under DDP (``mbs_invariance_utils.StepRunner``), built once per module."""
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
    # Per-run gradients go to host memory: 11 fp32 copies of a 3.9 B-parameter slice do not fit
    # next to the b = 8 activations.
    return mbs.StepRunner(_model(config), grads_device="cpu")


class _KernelLog:
    """Counts of the CSA / DSA paths the step ran, by batch size, and the indexer mutant.

    While ``indexer_scale`` is set, every fused CSA indexer + sparse attention call with b > 1
    rows (SBHD: ``query`` is (s, b, h, d)), the loop's recompute replays included, scales the
    gradients of the last sample's indexer inputs (``q_indexer``, ``k_indexer``, ``weights``) by
    it in the backward (``mbs_invariance_utils.ScaleSampleGrad``); ``mutated`` counts those calls.
    """

    def __init__(self, monkeypatch):
        from megatron.core.transformer.experimental_attention_variant import csa, dsa_cudnn_kernels

        self.fused_indexer_batches = collections.Counter()
        self.unfused_csa = 0
        self.dsa_sparse_indexer_loss = 0
        self.indexer_scale = None
        self.mutated = 0
        fused = csa.fused_csa_indexer_sparse_attn
        unfused = csa.CompressedSparseAttention._forward_unfused_csa
        dsa_loss = dsa_cudnn_kernels._compute_sparse_indexer_loss_and_grads

        def fused_wrapper(
            query, kv_full, attn_sink, window_idxs, q_indexer, k_indexer, weights, *args, **kwargs
        ):
            batch = int(query.shape[1])
            self.fused_indexer_batches[batch] += 1
            if self.indexer_scale is not None and batch > 1:
                sample, scale = batch - 1, self.indexer_scale
                q_indexer = mbs.ScaleSampleGrad.apply(q_indexer, sample, scale)
                k_indexer = mbs.ScaleSampleGrad.apply(k_indexer, sample, scale)
                weights = mbs.ScaleSampleGrad.apply(weights, sample, scale)
                self.mutated += 1
            return fused(
                query,
                kv_full,
                attn_sink,
                window_idxs,
                q_indexer,
                k_indexer,
                weights,
                *args,
                **kwargs,
            )

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


def _indexer_weight_names(model):
    """``{C layer: its indexer weights}``, and the C layers that run once per sample.

    The latter are the prelude's and the coda's C layers, whose indexer weights the mutant must
    fail on (gate 4); the core's C layer runs every round and is reported only."""
    names = [name for name, _ in model.named_parameters()]
    layers = [index for index, symbol in enumerate(PATTERN) if symbol == "C"]
    by_layer = {
        layer: [name for name in names if f".layers.{layer}." in name and ".indexer." in name]
        for layer in layers
    }
    coda_start = len(PATTERN) - CODA_SUBLAYERS
    once = [layer for layer in layers if layer < PRELUDE_SUBLAYERS or layer >= coda_start]
    return by_layer, once


def _set_training_kernels(monkeypatch, runner):
    """Training's kernels: the fused DSA backward's non-deterministic algorithms."""
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
    monkeypatch.setattr(runner.model.config, "deterministic_mode", False)
    torch.use_deterministic_algorithms(False, warn_only=True)


def _relative(value, reference):
    """``|value - reference| / |reference|`` in fp64 (0 when both are zero)."""
    value, reference = value.double().reshape(-1), reference.double().reshape(-1)
    diff, norm = (value - reference).norm().item(), reference.norm().item()
    if norm == 0.0:
        return 0.0 if diff == 0.0 else math.inf
    return diff / norm


def _short(name):
    return (
        name.replace("decoder.layers.", "L")
        .replace(".inner_layer.self_attention.core_attention", "")
        .replace(".hyper_connection", ".hc")
    )


def _verdict(entry, prefix=""):
    """One gate entry as text: ``rel`` (resolved) or ``abs`` value, its bound and their ratio;
    ``prefix='single_'`` for the mutant run."""
    x, bound = entry[f"{prefix}x"], entry[f"{prefix}bound"]
    kind = "rel" if entry["resolved"] else "abs"
    return f"{kind} {x:.2e} / bound {bound:.2e} ({x / bound:.2f})"


def _run_arm(runner, log, side, rows, label, micro_batch_size, indexer_scale=None):
    """One step over ``rows`` in microbatches of ``micro_batch_size`` (with ``indexer_scale``:
    under the indexer mutant): its losses, rows, final gradients (on the host), the TE RMSNorm
    side channel in ``main_grad``'s units and its check against ``main_grad``, and its paths."""
    fused_before, mutated_before = collections.Counter(log.fused_indexer_batches), log.mutated
    log.indexer_scale = indexer_scale
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        losses, row_losses, grads = runner.run(rows, micro_batch_size, seq=SEQ)
    finally:
        log.indexer_scale = None
    seconds = time.perf_counter() - start
    sums, calls = side.take()
    # main_grad is the step's gradient sum over its loss tokens divided by their number
    # (calculate_per_token_loss); the side channel is the sum.
    tokens = float(rows["loss_mask"].sum().item())
    norms = {name: value / tokens for name, value in sums.items()}
    run = dict(
        label=label,
        losses=losses,
        rows=row_losses,
        grads=grads,
        side=norms,
        side_calls=calls,
        side_check={name: _relative(grads[name], value) for name, value in norms.items()},
        fused=dict(log.fused_indexer_batches - fused_before),
        mutated=log.mutated - mutated_before,
    )
    print(
        f"[T7 b={rows['tokens'].shape[0]}] run {label}: micro-batch size {micro_batch_size}, "
        f"{seconds:.1f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; losses "
        + ", ".join(f"{key} {value:.6e}" for key, value in losses.items())
        + f"; fused CSA calls by batch {run['fused']}; mutated calls {run['mutated']}; side "
        f"channel |main_grad - side| / |side| <= "
        f"{max(run['side_check'].values(), default=math.nan):.2e}",
        flush=True,
    )
    return run


def _run_protocol(runner, log, side, batch):
    """The runs and statistics of one b (module docstring); returns what the gates need.

    The per-run gradients (15.5 GB each on the host) are released on return."""
    rows = mbs.make_rows(batch, SEQ, VOCAB, seed=ROW_SEED)
    runs_a, runs_b = [], []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for repeat in range(1, REPEATS + 1):
            runs_a.append(_run_arm(runner, log, side, rows, f"A{repeat}", 1))
            runs_b.append(_run_arm(runner, log, side, rows, f"B{repeat}", batch))
        mutant = _run_arm(runner, log, side, rows, "M", batch, indexer_scale=MUTANT_INDEXER_SCALE)
    fallbacks = sorted({str(w.message)[:160] for w in caught if "falling back" in str(w.message)})
    checked = [name for names in _checked_weight_names(runner.model).values() for name in names]

    device = torch.cuda.current_device()
    main = mbs.repeat_statistics(
        [run["grads"] for run in runs_a],
        [run["grads"] for run in runs_b],
        single_b=mutant["grads"],
        device=device,
    )
    norms = mbs.repeat_statistics(
        [run["side"] for run in runs_a],
        [run["side"] for run in runs_b],
        single_b=mutant["side"],
        device=device,
    )
    whole_norm = math.sqrt(sum(entry["norm"] ** 2 for entry in main.values()))
    symmetric = [name for name in main if name.endswith(SYMMETRIC_ZERO_SUFFIXES)]
    gate = functools.partial(
        mbs.repeat_gate,
        whole_norm=whole_norm,
        factor=GATE_FACTOR,
        min_relative=MIN_RELATIVE,
        unresolved_noise=UNRESOLVED_NOISE,
        absolute_fraction=ABSOLUTE_FRACTION,
        absolute_names=symmetric,
    )
    pairs = [(run_b, run_a) for run_b in runs_b for run_a in runs_a]
    clean = runs_a + runs_b
    return dict(
        report=gate({**main, **norms}),  # the TE RMSNorm weights on the side channel
        norms_on_main_grad=gate({name: main[name] for name in norms}),  # recorded only
        whole_norm=whole_norm,
        symmetric=symmetric,
        checked=checked,
        not_training=[name for name in checked if runs_a[0]["grads"][name].norm() == 0],
        reference_losses=runs_a[0]["losses"],
        loss_rel={
            key: max(mbs.relative_loss_differences(b["losses"], a["losses"])[key] for b, a in pairs)
            for key in mbs.LOSS_KEYS
        },
        row_rel=max(mbs.max_row_difference(b["rows"], a["rows"]) for b, a in pairs),
        loss_rel_within_a=max(
            max(mbs.relative_loss_differences(a2["losses"], a1["losses"]).values())
            for a1 in runs_a
            for a2 in runs_a
            if a1 is not a2
        ),
        norm_names=sorted(norms),
        side_worst=max(
            (
                (value, name, run["label"])
                for run in clean + [mutant]
                for name, value in run["side_check"].items()
            ),
            default=(math.inf, "no TE RMSNorm weight", "-"),
        ),
        side_uncalled=[
            (run["label"], name)
            for run in clean + [mutant]
            for name in norms
            if not run["side_calls"].get(name)
        ],
        mutant_fused=mutant["fused"].get(batch, 0),
        mutant_mutated=mutant["mutated"],
        fallbacks=fallbacks,
    )


def _print_result(tag, batch, result, log, by_layer, once):
    """The diagnostics of one b (printed before the gates assert)."""
    report, norms_main = result["report"], result["norms_on_main_grad"]
    print(
        f"[{tag}] paths: fused CSA indexer calls by batch {dict(log.fused_indexer_batches)}; "
        f"unfused CSA calls {log.unfused_csa}; DSA cuDNN sparse indexer-loss calls (the 'D' "
        f"path with the b > 1 offset site) {log.dsa_sparse_indexer_loss}; fallback warnings "
        f"{result['fallbacks']}",
        flush=True,
    )
    print(
        f"[{tag}] losses, worst B vs A over {REPEATS * REPEATS} pairs: "
        + ", ".join(f"{key} {value:.2e}" for key, value in result["loss_rel"].items())
        + f"; worst row {result['row_rel']:.2e} (bound {LOSS_ROW_MAX_RELATIVE:.0e}); worst A vs A "
        f"{result['loss_rel_within_a']:.2e}",
        flush=True,
    )
    value, name, label = result["side_worst"]
    print(
        f"[{tag}] side channel: {len(result['norm_names'])} TE RMSNorm weights; "
        f"|main_grad - side| / |side| <= {value:.2e} over all runs (bound "
        f"{SIDE_CHANNEL_MAX_RELATIVE:.2e}; worst {_short(name)} in run {label}); weights without "
        f"a backward call in some run: {result['side_uncalled']}",
        flush=True,
    )
    ratios = sorted(((r["x"] / r["bound"], name) for name, r in report.items()), reverse=True)
    unresolved = [name for name, r in report.items() if not r["resolved"]]
    print(
        f"[{tag}] gate (R = {REPEATS}, x <= max(2^-7, {GATE_FACTOR:g} e) if resolved, else "
        f"|dg| <= {ABSOLUTE_FRACTION:.0e} x the whole-gradient norm {result['whole_norm']:.4e} = "
        f"{ABSOLUTE_FRACTION * result['whole_norm']:.2e}): {len(report)} params, "
        f"{len(report) - len(unresolved)} resolved, {len(unresolved)} unresolved "
        f"({len(result['symmetric'])} mHC stream-symmetry scalars); failing "
        f"{sum(not r['ok'] for r in report.values())}; worst x / bound {ratios[0][0]:.2f} "
        f"({_short(ratios[0][1])})",
        flush=True,
    )
    print(
        f"[{tag}] worst: "
        + "; ".join(
            f"{_short(name)} {_verdict(report[name])} e {report[name]['e']:.2e} "
            f"|g| {report[name]['norm']:.2e}"
            for _, name in ratios[:8]
        ),
        flush=True,
    )
    for name in result["norm_names"]:
        if "compressor.norm" in name:
            print(
                f"[{tag}]   {_short(name)}: side channel {_verdict(report[name])}; main_grad "
                f"(recorded only) {_verdict(norms_main[name])}",
                flush=True,
            )
    for name in report:
        if name.endswith(".attn_sink"):
            print(f"[{tag}]   {_short(name)}: {_verdict(report[name])}", flush=True)
    print(
        f"[{tag}] teeth idx_{MUTANT_INDEXER_SCALE:g}: the last sample's indexer-input gradients "
        f"scaled in {result['mutant_mutated']} of {result['mutant_fused']} fused CSA calls at "
        f"batch {batch}; gated against mean A with one run's B noise",
        flush=True,
    )
    for layer, names in by_layer.items():
        role = "runs once per sample: must fail" if layer in once else "core: recorded only"
        caught = sum(not report[name]["single_ok"] for name in names)
        smallest = min(
            (report[name]["single_x"] / report[name]["single_bound"] for name in names),
            default=math.nan,
        )
        print(
            f"[{tag}]   layer {layer} ({role}): {caught} of {len(names)} fail, smallest "
            f"x / bound {smallest:.2f}",
            flush=True,
        )
        for name in names:
            entry = report[name]
            print(
                f"[{tag}]     {_short(name)}: clean {_verdict(entry)}; mutant "
                f"{_verdict(entry, 'single_')} {'PASSES' if entry['single_ok'] else 'fails'}",
                flush=True,
            )


@pytest.mark.parametrize("batch", [4, 8])
def test_fused_kernels_batch_equals_single_rows(runner, monkeypatch, batch):
    """Gates 1-4 of the module docstring at micro-batch size ``batch``."""
    tag = f"T7 b={batch}"
    log = _KernelLog(monkeypatch)
    side = mbs.RMSNormWeightSideChannel(monkeypatch, runner.model)
    by_layer, once = _indexer_weight_names(runner.model)
    # What the gates cover, checked before the runs: the indexer and compressor weights; every
    # TE RMSNorm weight, a parameter of the model, on the side channel; all six indexer weights
    # of every C layer that runs once per sample in the teeth.
    for key, names in _checked_weight_names(runner.model).items():
        assert names, key
    parameters = {name for name, _ in runner.model.named_parameters()}
    assert side.names and set(side.names.values()) <= parameters, sorted(side.names.values())
    assert once, PATTERN
    for layer in once:
        for kind in INDEXER_WEIGHTS:
            assert any(kind in name for name in by_layer[layer]), (layer, kind)
    print(
        f"[{tag}] {REPEATS} alternating runs of {batch} x 1 (A) and 1 x {batch} (B), training's "
        f"kernels, then one 1 x {batch} run under the indexer mutant; side channel on "
        f"{len(side.names)} TE RMSNorm weights; teeth layers {once}",
        flush=True,
    )
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        _set_training_kernels(monkeypatch, runner)
        result = _run_protocol(runner, log, side, batch)
    finally:
        torch.use_deterministic_algorithms(previous)
    report = result["report"]
    _print_result(tag, batch, result, log, by_layer, once)
    mbs.dump_report(tag, result)

    # 1. The fused path ran at both batch sizes and the unfused CSA path never did.
    assert log.fused_indexer_batches[1] > 0 and log.fused_indexer_batches[batch] > 0
    assert log.unfused_csa == 0
    # The checked weights train, every loss is attached, every TE RMSNorm weight was recorded.
    assert not result["not_training"], result["not_training"]
    assert all(result["reference_losses"][key] != 0.0 for key in ("lm", "aux", "indexer"))
    assert sorted(side.names.values()) == result["norm_names"]

    failures = []
    # 2. Losses and rows: every B run vs every A run.
    if (
        any(value > LOSS_ROW_MAX_RELATIVE for value in result["loss_rel"].values())
        or result["row_rel"] > LOSS_ROW_MAX_RELATIVE
    ):
        failures.append(f"losses {result['loss_rel']}, max row rel {result['row_rel']:.2e}")
    # 3. The side channel is main_grad without the bf16 roundings, in every run ...
    if result["side_worst"][0] > SIDE_CHANNEL_MAX_RELATIVE or result["side_uncalled"]:
        failures.append(
            f"side channel vs main_grad: worst {result['side_worst']}; without a call "
            f"{result['side_uncalled']}"
        )
    # ... every parameter passes on the means, and the weights this test is about are resolved.
    unresolved_checked = [name for name in result["checked"] if not report[name]["resolved"]]
    if unresolved_checked:
        failures.append(f"checked weights not resolved: {unresolved_checked}")
    failing = sorted(
        (name for name, r in report.items() if not r["ok"]),
        key=lambda name: -report[name]["x"] / report[name]["bound"],
    )
    if failing:
        failures.append(
            f"{len(failing)} params: "
            + "; ".join(
                f"{name} x {report[name]['x']:.3e} bound {report[name]['bound']:.3e} "
                f"({'resolved' if report[name]['resolved'] else 'unresolved'})"
                for name in failing[:8]
            )
        )
    # 4. Teeth: the mutant ran on every fused call at b and fails on every once-per-sample
    # indexer weight.
    if not 0 < result["mutant_mutated"] == result["mutant_fused"]:
        failures.append(
            f"teeth: mutated {result['mutant_mutated']} of {result['mutant_fused']} fused CSA "
            f"calls at batch {batch}"
        )
    passed = [
        name
        for layer in once
        for name in by_layer[layer]
        if not report[name]["resolved"] or report[name]["single_ok"]
    ]
    if passed:
        failures.append(
            f"teeth: the idx_{MUTANT_INDEXER_SCALE:g} mutant passes the gate on {passed}"
        )
    assert not failures, failures
