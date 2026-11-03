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
sqrtsoftplus scores, top-k scaling 1.5, expert bias and one shared expert, SwiGLU with the
activation clamp, RMSNorm, mHC with 4 streams, deterministic kernels, full-precision GEMMs
(no TF32). Runs under torchrun at W in {1, 2, 4, 8} (EP = W; each rank its own 4 rows). The
DSv4 Hadamard rotation (``csa.rotate_activation``, bf16-only) is the same scaled identity in
every arm, so the fp32 and bf16 models compute the same function.

* ``moe_aux_loss_coeff = 1.0``: the learned routers' weight gradients are dominated by the
  sequence aux loss. The indexer weights get gradients only from the indexer loss (the indexer
  reads detached activations, ``csa.py`` unfused path).
* fp32 (the math): every attached loss sum and every row's LM loss within 1e-3 relative of
  MBS 1; the whole gradient cosine >= 0.9999 and relative difference <= 1e-3; every resolved
  parameter gradient the same (below).
* bf16 (the precision the arms train in): different micro-batch sizes run different GEMM shapes,
  whose rounding the network amplifies (FSDP-arm deviations D6). Each parameter's noise floor
  is bf16's own error on it: bf16 MBS 1 against the fp32 model with the same weights.
* Per parameter (``mbs_invariance_utils.resolution_gate``): a gradient the floor resolves (floor
  <= 0.5) must match in fp32 (cosine >= 0.9999, relative <= 1e-3) and in bf16 (relative <=
  max(1e-3, 3 x floor)). An unresolved gradient is zero up to rounding (at initialization the
  4 mHC streams are copies, so the gradients of the stream-mixing scalars vanish); it is gated
  absolutely, |g - g_ref| <= 1e-4 x the whole-gradient norm. The learned routers and every indexer
  weight must be resolved.
* Teeth: ``per_microbatch_mean_aux_scaling`` (aux and indexer gradients x 1 / micro-batch size,
  the per-microbatch mean the ``loop_schedule`` docstring describes) must fail the fp32 gate and
  the bf16 gate on every learned router weight and every indexer weight.
"""

import contextlib
import math
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
MBS = (1, 2, 4)
MIN_COSINE = 0.9999
MAX_RELATIVE = 1e-3
FLOOR_FACTOR = 3.0


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    """Deterministic kernels and full-precision fp32 GEMMs for the module: set before model
    parallelism and the first GEMM (NCCL, cuBLAS and TE read these once per process)."""
    if not CUDA:
        yield
        return
    previous = os.environ.get("NVIDIA_TF32_OVERRIDE")
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with determinism_env(), model_parallel_once(expert_model_parallel_size=WORLD):
            torch.use_deterministic_algorithms(True, warn_only=True)
            try:
                yield
            finally:
                torch.use_deterministic_algorithms(False)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        if previous is None:
            os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
        else:
            os.environ["NVIDIA_TF32_OVERRIDE"] = previous


@contextlib.contextmanager
def _scaled_identity_rotation():
    """``csa.rotate_activation`` -> ``x / sqrt(d)`` (the Hadamard transform is bf16-only)."""
    from unittest.mock import patch

    with patch(
        "megatron.core.transformer.experimental_attention_variant.csa.rotate_activation",
        lambda x: x * (x.size(-1) ** -0.5),
    ):
        yield


def _model(dtype):
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
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
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
def results():
    """Every arm of the module, run once: {(precision, mutant): {mbs: (losses, rows, grads)}}
    and the model names. The fp32 model starts from the bf16 model's weights."""
    bf16_model = _model(torch.bfloat16)
    fp32_model = _model(torch.float32)
    state = {
        name: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
        for name, value in bf16_model.state_dict().items()
    }
    fp32_model.load_state_dict(state, strict=True)
    runners = {"bf16": mbs.StepRunner(bf16_model), "fp32": mbs.StepRunner(fp32_model)}
    rows = mbs.make_rows(ROWS, SEQ, VOCAB, seed=4242 + torch.distributed.get_rank())
    out = {}
    with _scaled_identity_rotation():
        for precision in ("fp32", "bf16"):
            for mutant in (False, True):
                arms = {}
                for micro_batch_size in MBS:
                    if mutant and micro_batch_size == 1:
                        arms[1] = out[(precision, False)][1]  # x 1 / 1: the same step
                        continue
                    mutation = (
                        mbs.per_microbatch_mean_aux_scaling(micro_batch_size)
                        if mutant
                        else contextlib.nullcontext()
                    )
                    with mutation:
                        arms[micro_batch_size] = runners[precision].run(
                            rows, micro_batch_size, seq=SEQ
                        )
                out[(precision, mutant)] = arms
    routers, indexers = learned_router_and_indexer_names(bf16_model)
    return out, routers, indexers


def _precision_floor(out):
    """Per parameter and globally: bf16 MBS 1 against fp32 MBS 1 (same weights), bf16's own
    error on that gradient."""
    bf16, fp32 = out[("bf16", False)][1][2], out[("fp32", False)][1][2]
    floor = mbs.compare_gradients(bf16, fp32, min_cosine=0.0, max_relative=math.inf)
    return floor, mbs.global_comparison(bf16, fp32)


def _tag(text):
    return f"T3 W{WORLD} rank{torch.distributed.get_rank()} {text}"


def _gate(grads, ref_grads, floor, strict):
    """Per-parameter resolution gate (fp32: strict; bf16: floor-relative) and the global gate."""
    report = mbs.resolution_gate(
        mbs.compare_gradients(grads, ref_grads, min_cosine=0.0, max_relative=math.inf),
        floor,
        global_norm=mbs.global_comparison(ref_grads, ref_grads)["norm"],
        strict=strict,
        min_cosine=MIN_COSINE,
        min_relative=MAX_RELATIVE,
        factor=FLOOR_FACTOR,
    )
    return report, mbs.global_comparison(grads, ref_grads)


def test_routers_and_indexers_are_resolved(results):
    """The checked weights train and bf16 resolves their gradients (the gates below are then
    strict on them, and the mutation test has teeth)."""
    out, routers, indexers = results
    floor, global_floor = _precision_floor(out)
    ref_losses, _, ref_grads = out[("fp32", False)][1]
    mbs.print_report(
        _tag("bf16 precision floor (bf16 MBS 1 vs fp32 MBS 1)"), floor, routers[:2] + indexers[:2]
    )
    mbs.dump_report(_tag("floor bf16 vs fp32 MBS 1"), dict(per_param=floor, total=global_floor))
    print(f"[{_tag('global floor')}] {global_floor}", flush=True)
    assert all(ref_losses[key] != 0.0 for key in mbs.LOSS_KEYS), ref_losses
    assert routers and indexers
    for name in routers + indexers:
        assert ref_grads[name].norm() > 0, name
        assert floor[name]["relative"] <= 0.5, (name, floor[name])


def test_fp32_step_is_invariant_to_micro_batch_size(results):
    out, routers, indexers = results
    floor, _ = _precision_floor(out)
    arms = out[("fp32", False)]
    ref_losses, ref_rows, ref_grads = arms[1]
    for micro_batch_size in MBS[1:]:
        losses, rows, grads = arms[micro_batch_size]
        loss_rel = mbs.relative_loss_differences(losses, ref_losses)
        row_rel = mbs.max_row_difference(rows, ref_rows)
        report, total = _gate(grads, ref_grads, floor, strict=True)
        tag = _tag(f"fp32 MBS {micro_batch_size} vs 1")
        print(
            f"[{tag}] loss rel {loss_rel}; max row rel {row_rel:.2e}; whole gradient {total}",
            flush=True,
        )
        mbs.print_report(tag, report, routers[:2] + indexers[:2])
        mbs.dump_report(tag, dict(per_param=report, total=total, loss_rel=loss_rel))
        assert all(value <= 1e-3 for value in loss_rel.values()), loss_rel
        assert row_rel <= 1e-3, row_rel
        assert total["cosine"] >= MIN_COSINE and total["relative"] <= MAX_RELATIVE, total
        failing = [name for name, r in report.items() if not r["ok"]]
        assert not failing, mbs.worst(report, failing)


def test_bf16_step_is_invariant_within_bf16_precision(results):
    out, routers, indexers = results
    floor, global_floor = _precision_floor(out)
    arms = out[("bf16", False)]
    ref_losses, ref_rows, ref_grads = arms[1]
    for micro_batch_size in MBS[1:]:
        losses, rows, grads = arms[micro_batch_size]
        loss_rel = mbs.relative_loss_differences(losses, ref_losses)
        row_rel = mbs.max_row_difference(rows, ref_rows)
        report, total = _gate(grads, ref_grads, floor, strict=False)
        tag = _tag(f"bf16 MBS {micro_batch_size} vs 1")
        print(
            f"[{tag}] loss rel {loss_rel}; max row rel {row_rel:.2e}; whole gradient {total}",
            flush=True,
        )
        mbs.print_report(tag, report, routers[:2] + indexers[:2])
        mbs.dump_report(tag, dict(per_param=report, total=total, loss_rel=loss_rel))
        assert all(value <= 1e-3 for value in loss_rel.values()), loss_rel
        assert row_rel <= 1e-3, row_rel
        assert total["relative"] <= max(MAX_RELATIVE, FLOOR_FACTOR * global_floor["relative"]), (
            total,
            global_floor,
        )
        failing = [name for name, r in report.items() if not r["ok"]]
        assert not failing, mbs.worst(report, failing)


def test_per_microbatch_mean_aux_scaling_is_caught(results):
    """The 1 / b mutation fails the fp32 gate and the bf16 gate on every learned router weight
    and every indexer weight."""
    out, routers, indexers = results
    floor, _ = _precision_floor(out)
    for precision in ("fp32", "bf16"):
        _, _, ref_grads = out[(precision, False)][1]
        for micro_batch_size in MBS[1:]:
            _, _, grads = out[(precision, True)][micro_batch_size]
            report, total = _gate(grads, ref_grads, floor, strict=precision == "fp32")
            tag = _tag(f"mutant {precision} MBS {micro_batch_size} vs 1")
            print(f"[{tag}] whole gradient {total}", flush=True)
            mbs.print_report(tag, report, routers[:2] + indexers[:2])
            mbs.dump_report(tag, dict(per_param=report, total=total))
            passing = [name for name in routers + indexers if report[name]["ok"]]
            assert not passing, mbs.worst(report, passing)
