# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Spike (j): whole-layer CUDA-graph capture of DSv4 attention + MoE pairs on HybridStack.

The probe builds a plain dev ``HybridModel`` (no loop) with the DSv4 hybrid stack spec and
two attention + E pairs (``CECE``; ``HEHE`` / ``WEWE`` with ``--kind HCA`` / ``W``), static-budget
HybridEP dispatch (``moe_expert_rank_capacity_factor``), the TE op-fuser grouped experts, mHC
and no paged stash, and captures it two ways:

* arm A (``helper``): dev's ``TECudaGraphHelper`` with ``cuda_graph_modules = []``. Every
  ``HyperConnectionHybridLayer`` is one whole-layer callable, so a pair is two graphs; the
  helper's ``create_cudagraphs`` runs the static-budget branch of
  ``validate_moe_cuda_graph_support`` (``moe_cuda_graph_static_budget_without_paged_stash``).
* arm B (``pair``): one graph per attention + E pair, built with TE's ``make_graphed_callables``
  on an explicit pool (the executor-owned unit). Captures ``block`` (``_order = [1, -1]``),
  ``nested`` (``[2, -2, 1, -1]``) and ``nested-noreuse`` (``nested`` without TE's input /
  output buffer reuse).

Each capture is replayed in the item orders (``block``, ``nested``, ``nested-asc``, ``sweep``,
``interleaved``, ``twice``) against a fresh block-order replay right after capture (bitwise),
with the graph pool's free blocks NaN-poisoned between items, and against an eager twin. The
record also carries the static budget sizing, saved-activation bytes by component (a separate
eager process), HBM peak and pool bytes (a separate ``--memory-only`` process), static slots,
host and GPU time per replay, and a precise failure record (step, frames, CUDA error, the
host synchronizations seen before capture) when a capture fails.

Run from the vendor root on Blackwell with HybridEP and the TE op fuser, e.g.::

    python -m torch.distributed.run --nproc-per-node 4 \
        -m tests.unit_tests.transformer.loop.pair_capture --width pro-proxy --precision bf16 \
        --arms helper,pair --out spike_j.jsonl

The unit test ``test_loop_pair_capture.py`` runs the ``small`` width through the same helpers.
This module is test code: it reaches into private attributes of the graph helper, the HybridEP
manager and TE on purpose and is not collected by pytest.
"""

import argparse
import contextlib
import dataclasses
import faulthandler
import functools
import gc
import itertools
import json
import math
import os
import statistics
import time
import traceback
import warnings
from collections import defaultdict
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import MLATransformerConfig

# ---------------------------------------------------------------------------------------------
# Shapes and configuration
# ---------------------------------------------------------------------------------------------

WIDTHS = {
    # DSV4-PRO-PROXY widths (configs/benchmarking/models/deepseek_v4_pro_proxy.yaml); experts
    # per rank as DSV4-PRO-PROXY at EP 64 (384 / 64), so tokens per expert match the real run.
    "pro-proxy": dict(
        hidden_size=7168,
        num_attention_heads=128,
        v_head_dim=512,
        q_lora_rank=1536,
        output_projection_groups=16,
        output_projection_lora_rank=1024,
        qk_pos_emb_head_dim=64,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=1024,
        csa_window_size=128,
        experts_per_rank=6,
        moe_ffn_hidden_size=3072,
        moe_shared_expert_intermediate_size=2048,
        moe_router_topk=6,
        router="sqrtsoftplus",
        mhc_sinkhorn_iterations=20,
        vocab_size=1024,
    ),
    # The DSV4-MINI rung of the plan's early fallback (EP 8, 64 experts).
    "mini": dict(
        hidden_size=4096,
        num_attention_heads=64,
        v_head_dim=512,
        q_lora_rank=1024,
        output_projection_groups=8,
        output_projection_lora_rank=1024,
        qk_pos_emb_head_dim=64,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=512,
        csa_window_size=128,
        experts_per_rank=8,
        moe_ffn_hidden_size=1024,
        moe_shared_expert_intermediate_size=1024,
        moe_router_topk=6,
        router="sqrtsoftplus",
        mhc_sinkhorn_iterations=20,
        vocab_size=1024,
    ),
    # The unit-test width (every dimension a multiple of 32, so MXFP8 runs too).
    "small": dict(
        hidden_size=1024,
        num_attention_heads=16,
        v_head_dim=128,
        q_lora_rank=256,
        output_projection_groups=4,
        output_projection_lora_rank=256,
        qk_pos_emb_head_dim=32,
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=128,
        csa_window_size=64,
        experts_per_rank=2,
        moe_ffn_hidden_size=256,
        moe_shared_expert_intermediate_size=256,
        moe_router_topk=2,
        router="sigmoid",
        mhc_sinkhorn_iterations=5,
        vocab_size=128,
    ),
}

# Attention kind -> (hybrid layer symbol, CSA compress ratio). ``W`` (window-only attention: no
# compressor, no indexer) isolates the MoE half when the reference CSA core does not capture.
KINDS = {"CSA": ("C", 4), "HCA": ("H", 128), "W": ("W", 0)}

POISON_INT32 = 0x7FC07FC0  # a NaN as fp32 and as each bf16 half
MODEL_SAVED_MB_PER_TOKEN = {"pro-proxy": 0.97}  # DP Section 14 item 4 (dsv4.py:208)
NUM_STREAMS = 4
CAPTURE_ORDERS = {
    # capture -> (TE ``_order``, ``_num_layers_per_chunk``, chunk names)
    "block": ([1, -1], [2], [["P0", "P1"]]),
    "nested": ([2, -2, 1, -1], [1, 1], [["P0"], ["P1"]]),
    "nested-noreuse": ([2, -2, 1, -1], [1, 1], [["P0"], ["P1"]]),
}
# order -> forward replays per pair (router counts / DSA tracker multiply by it)
ORDER_FORWARDS = {
    "block": 1,
    "nested": 2,
    "nested-asc": 1,
    "sweep": 2,
    "interleaved": 1,
    "twice": 2,
}
# orders whose outputs, input gradients and main_grads must equal the oracle bitwise
GATED_ORDERS = ("block", "nested", "sweep", "interleaved")
# (capture, order) combinations that are open questions (DV-sp-2): reported, not gated
OPEN_ORDERS = {("nested", "block")}
# no-grad replay outputs of the ``sweep`` / ``nested`` orders -> the oracle output they must equal
NO_GRAD_OUTPUTS = {"sweep_y0": "y0", "sweep_y1": "y1", "rebuild_y0": "y0", "rebuild_y1": "y1"}
# capture -> the replay order it was captured in: the oracle is a fresh replay in this order
# (a block-order replay of a nested capture is itself one of the questions, not a reference)
NATIVE_ORDER = {"helper": "block", "block": "block", "nested": "nested", "nested-noreuse": "nested"}
ORDERS_PER_CAPTURE = {
    "helper": ("block", "nested", "sweep", "interleaved"),
    "block": ("block", "nested", "nested-asc", "sweep", "interleaved", "twice"),
    "nested": ("nested", "block"),
    "nested-noreuse": ("nested",),
}


# captures whose graph-pool poisoning runs report-only (design 5.6: arm A's pool is the helper's)
REPORT_ONLY_POISONED = ("helper",)
# P5: the replayed aux component may differ from eager's by at most this relative error
P5_REL_TOL = 0.05
# seconds a rank waits in a step agreement for the others (a one-sided failure: see agree_records)
AGREE_TIMEOUT_S = 300.0


def is_gated(capture: str, order: str, poisoned: bool = False) -> bool:
    """Whether replay ``order`` of capture ``capture`` (with the graph pool poisoned between
    items when ``poisoned``) must equal the oracle bitwise."""
    gated = order in GATED_ORDERS and (capture, order) not in OPEN_ORDERS
    return gated and not (poisoned and capture in REPORT_ONLY_POISONED)


def _te_ops():
    try:
        import transformer_engine.pytorch.ops as te_ops  # pylint: disable=import-outside-toplevel

        return te_ops
    except Exception:  # pylint: disable=broad-except
        return None


def _te_min(version: str) -> bool:
    try:
        from megatron.core.utils import is_te_min_version  # pylint: disable=import-outside-toplevel

        return bool(is_te_min_version(version))
    except Exception:  # pylint: disable=broad-except
        return False


def clamp_supported() -> bool:
    """Whether the op-fuser grouped MLP accepts clamped SwiGLU (TE >= 2.17.0.dev0 with
    ``ScaledClampedQGeGLU``; experts.py rejects it at construction otherwise)."""
    te_ops = _te_ops()
    return te_ops is not None and _te_min("2.17.0.dev0") and hasattr(te_ops, "ScaledClampedQGeGLU")


def op_fuser_grouped_mlp_available() -> bool:
    """TE >= 2.14 with the op-fuser ``GroupedLinear`` and ``ScaledSwiGLU`` ops."""
    te_ops = _te_ops()
    return (
        te_ops is not None
        and _te_min("2.14.0")
        and hasattr(te_ops, "GroupedLinear")
        and hasattr(te_ops, "ScaledSwiGLU")
    )


def hybridep_available(device: Optional[int] = None) -> bool:
    """HybridEP importable on a Blackwell GPU (the static-budget whole-MoE path). ``device``
    is the GPU to ask (the current device by default)."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(device)[0] < 10:
        return False
    try:
        from megatron.core.transformer.moe import (  # pylint: disable=import-outside-toplevel
            token_dispatcher,
        )

        return token_dispatcher.hybrid_ep_dispatch is not None
    except Exception:  # pylint: disable=broad-except
        return False


def mxfp8_available() -> bool:
    """TE reports MXFP8 support on this device."""
    if not torch.cuda.is_available():
        return False
    for module in ("transformer_engine.pytorch.fp8", "transformer_engine.pytorch.quantization"):
        try:
            check = getattr(
                __import__(module, fromlist=["check_mxfp8_support"]), "check_mxfp8_support"
            )
            return bool(check()[0])
        except Exception:  # pylint: disable=broad-except
            continue
    return False


def layer_pattern(kind: str = "CSA", pairs: int = 2, attention_only: bool = False) -> str:
    """``CECE`` for two CSA pairs (``C`` alone per pair when ``attention_only``)."""
    letter = KINDS[kind][0]
    return (letter if attention_only else letter + "E") * pairs


def compress_ratios(kind: str = "CSA", pairs: int = 2, attention_only: bool = False) -> List[int]:
    """``csa_compress_ratios`` matching :func:`layer_pattern` (0 at MoE positions)."""
    ratio = KINDS[kind][1]
    return ([ratio] if attention_only else [ratio, 0]) * pairs


def pattern_of(config) -> str:
    """The hybrid layer pattern a spike config was built for."""
    letters = {ratio: letter for letter, ratio in KINDS.values()}
    moe = bool(config.num_moe_experts)
    return "".join(
        "E" if moe and index % 2 == 1 else letters[ratio]
        for index, ratio in enumerate(config.csa_compress_ratios[: config.num_layers])
    )


def spike_config(
    width: str,
    *,
    ep: int,
    experts_per_rank: Optional[int] = None,
    kind: str = "CSA",
    precision: str = "bf16",
    factor: Optional[float] = 2.0,
    fused_mhc: bool = True,
    clamp: Optional[bool] = None,
    permute_fusion: bool = True,
    graphs: bool = True,
    seq: int = 4096,
    indexer_topk: Optional[int] = None,
    pairs: int = 2,
    dsa_backend: str = "none",
    attention_only: bool = False,
) -> MLATransformerConfig:
    """The spike's ``MLATransformerConfig`` (stage-7 design 5.2).

    ``clamp=None`` turns the SwiGLU clamp on only when the op-fuser grouped MLP supports it
    (:func:`clamp_supported`); router padding is set only for MXFP8 (it raises without fp8).
    ``graphs=False`` builds the model without graph settings (``cuda_graph_impl = "none"``: the
    saved-bytes process). The eager twin is NOT built from it: it uses the graph config and is
    never captured (an uncaptured layer runs eager), because CSA takes different code with
    ``cuda_graph_impl != "none"`` (compact-indexer workspaces, the kernels' ``cuda_graph=``).
    ``seq`` is not a config field; it is accepted so every caller states the sequence length the
    config was sized for. ``attention_only`` builds attention layers without MoE (the fused
    attention core bytes, ``fused_attn_bytes.py``).
    """
    del seq
    if precision not in ("bf16", "mxfp8"):
        raise ValueError(f"precision must be bf16 or mxfp8, got {precision!r}")
    shape = WIDTHS[width]
    if clamp is None:
        clamp = clamp_supported()
    head_dim = shape["v_head_dim"]
    fields: Dict[str, Any] = dict(
        num_layers=pairs if attention_only else 2 * pairs,
        hidden_size=shape["hidden_size"],
        num_attention_heads=shape["num_attention_heads"],
        kv_channels=head_dim,
        v_head_dim=head_dim,
        q_lora_rank=shape["q_lora_rank"],
        output_projection_groups=shape["output_projection_groups"],
        output_projection_lora_rank=shape["output_projection_lora_rank"],
        qk_pos_emb_head_dim=shape["qk_pos_emb_head_dim"],
        multi_latent_attention=True,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        qk_layernorm=True,
        rope_type="rope",
        rotary_base=10000,
        experimental_attention_variant="dsv4_hybrid",
        dsa_kernel_backend=dsa_backend,
        dsa_indexer_n_heads=shape["dsa_indexer_n_heads"],
        dsa_indexer_head_dim=shape["dsa_indexer_head_dim"],
        dsa_indexer_topk=indexer_topk or shape["dsa_indexer_topk"],
        dsa_indexer_loss_coeff=1e-2,
        dsa_indexer_use_sparse_loss=True,
        csa_window_size=shape["csa_window_size"],
        csa_compress_ratios=compress_ratios(kind, pairs, attention_only),
        csa_compress_rotary_base=40000.0,
        enable_hyper_connections=True,
        num_residual_streams=NUM_STREAMS,
        mhc_sinkhorn_iterations=shape["mhc_sinkhorn_iterations"],
        use_fused_mhc=fused_mhc,
        mhc_fused_backend="auto",
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        deterministic_mode=True,
        attention_backend=AttnBackend.unfused if dsa_backend == "none" else AttnBackend.auto,
        create_attention_mask_in_dataloader=False,
        use_te_rng_tracker=True,
        expert_model_parallel_size=1 if attention_only else ep,
    )
    if not attention_only:
        sqrtsoftplus = shape["router"] == "sqrtsoftplus"
        fields.update(
            num_moe_experts=(experts_per_rank or shape["experts_per_rank"]) * ep,
            moe_ffn_hidden_size=shape["moe_ffn_hidden_size"],
            moe_shared_expert_intermediate_size=shape["moe_shared_expert_intermediate_size"],
            moe_router_topk=shape["moe_router_topk"],
            moe_router_score_function=shape["router"],
            moe_router_topk_scaling_factor=2.5 if sqrtsoftplus else None,
            moe_router_enable_expert_bias=True,
            moe_router_load_balancing_type="seq_aux_loss" if sqrtsoftplus else "aux_loss",
            moe_aux_loss_coeff=1e-4,
            moe_router_dtype="fp32",
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            moe_expert_rank_capacity_factor=factor,
            moe_permute_fusion=permute_fusion,
            moe_grouped_gemm=True,
            use_transformer_engine_op_fuser=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            moe_mlp_glu_interleave_size=32,
            activation_func_clamp_value=10.0 if clamp else None,
            moe_router_padding_for_quantization=precision == "mxfp8",
        )
    if graphs:
        fields.update(
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=[],
            moe_cuda_graph_static_budget_without_paged_stash=not attention_only,
            moe_paged_stash=False,
            cuda_graph_warmup_steps=1,
        )
    if precision == "mxfp8":
        fields.update(fp8="e4m3", fp8_recipe="mxfp8", disable_parameter_transpose_cache=True)
    return MLATransformerConfig(**fields)


def static_budget_factor(
    max_recv_ratio: float, *, floor: float = 1.2, margin: float = 1.1, step: float = 0.05
) -> float:
    """``f = max(1.2, ceil_0.05(1.1 x max_recv_ratio))`` (stage-7 design 5.4)."""
    scaled = math.ceil(round(margin * max_recv_ratio / step, 9)) * step
    return round(max(floor, scaled), 6)


def capture_labels(order: Sequence[int], chunk_names: Sequence[Sequence[str]]) -> List[str]:
    """The step name of every ``capture_begin`` TE issues for ``_order`` (one microbatch):
    forward chunks capture their layers in order, backward chunks in reverse."""
    labels = []
    for chunk in order:
        names = chunk_names[abs(chunk) - 1]
        if chunk > 0:
            labels.extend(f"fwd capture {name}" for name in names)
        else:
            labels.extend(f"bwd capture {name}" for name in reversed(names))
    return labels


def poison_coverage(
    inactive_blocks: Sequence[Tuple[int, int]], poison_ranges: Sequence[Tuple[int, int]]
) -> Tuple[int, int]:
    """``(covered, inactive)`` bytes: how much of the pool's inactive ``(address, size)`` blocks
    the poison tensors' ``(address, size)`` ranges overlap."""
    covered = 0
    for start, size in inactive_blocks:
        end = start + size
        spans = sorted(
            (max(start, p_start), min(end, p_start + p_size))
            for p_start, p_size in poison_ranges
            if p_start < end and p_start + p_size > start
        )
        position = start
        for span_start, span_end in spans:
            span_start = max(span_start, position)
            if span_end > span_start:
                covered += span_end - span_start
                position = span_end
    return covered, sum(size for _, size in inactive_blocks)


def fused_core_saved_bytes(config, seq: int, micro_batch: int = 1) -> Dict[str, int]:
    """Analytic saved bytes of the fused DSv4 attention core of one attention sublayer
    (the cuDNN DSA path): q and output bf16 ``[S, np, hn]``, ``kv_full`` bf16 (window keys plus
    compressed keys), lse fp32 ``[S, np]``, top-k indices int32 ``[S, K]`` and, for ratio-4
    layers, the indexer's q / k / weights. Used when the measured J5 core is missing."""
    ratio = config.csa_compress_ratios[0]
    tokens = seq * micro_batch
    heads, head_dim = config.num_attention_heads, config.v_head_dim
    n_comp = seq // ratio if ratio > 0 else 0
    window = config.csa_window_size
    if ratio == 4:
        keys = window + min(config.dsa_indexer_topk, max(n_comp, 1))
    else:
        keys = window + n_comp
    terms = {
        "q": 2 * tokens * heads * head_dim,
        "output": 2 * tokens * heads * head_dim,
        "kv_full": 2 * (seq + n_comp) * micro_batch * head_dim,
        "lse": 4 * tokens * heads,
        "topk_indices": 4 * tokens * keys,
    }
    if ratio == 4:
        idx_heads, idx_dim = config.dsa_indexer_n_heads, config.dsa_indexer_head_dim
        terms.update(
            indexer_q=2 * tokens * idx_heads * idx_dim,
            indexer_k=2 * n_comp * micro_batch * idx_dim,
            indexer_weights=4 * tokens * idx_heads,
        )
    terms["total"] = sum(terms.values())
    return terms


def parse_list(text: str, cast: Callable = str) -> List:
    """Comma-separated values (empty entries dropped)."""
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


# ---------------------------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------------------------


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def agree_all(flag: bool, group=None) -> bool:
    """True on every rank iff ``flag`` is True on every rank of ``group``."""
    if not dist.is_initialized():
        return bool(flag)
    value = torch.tensor([1 if flag else 0], device=_device(), dtype=torch.int32)
    dist.all_reduce(value, op=dist.ReduceOp.MIN, group=group)
    return bool(value.item())


def max_over_ranks(value: float, group=None) -> float:
    """The maximum of ``value`` over ``group`` (NaN-safe: NaN wins)."""
    if not dist.is_initialized():
        return float(value)
    is_nan = math.isnan(value)
    tensor = torch.tensor(
        [1.0 if is_nan else 0.0, -math.inf if is_nan else float(value)], device=_device()
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    any_nan, result = tensor.tolist()
    return float("nan") if any_nan else result


def gather_objects(obj: Any, group=None) -> List[Any]:
    """Every rank's ``obj`` of ``group`` in rank order (``[obj]`` without a process group)."""
    if not dist.is_initialized():
        return [obj]
    gathered: List[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, obj, group=group)
    return gathered


def nan_max(*values: float) -> float:
    """``max`` that propagates NaN (a NaN difference means a NaN-poisoned result)."""
    if any(math.isnan(value) for value in values):
        return float("nan")
    return max(values)


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------


def build_pair_models(
    config,
    *,
    pg_collection,
    seed: int = 1234,
    twin: bool = True,
    twin_config=None,
    vocab_size: int = 1024,
    max_sequence_length: int = 8192,
    ddp_stream: str = "side",
):
    """The DDP-wrapped graph model and (``twin``) its eager twin, built from the same seeds.
    The twin is built from ``twin_config`` (the callers pass a second graph config, so both
    models take the same code paths) and is never captured (:func:`assert_never_captured`).

    Both share one fresh ``MoEMetricsTracker`` (the tracker is process-global and the graphs
    bake its storage). DDP: ``grad_reduce_in_fp32``, no overlap, no distributed optimizer, so
    ``use_forward_hook`` is False (F14).

    ``ddp_stream``: DDP retains every parameter's AccumulateGrad node (``grad_accs``) and torch
    binds an AccumulateGrad node to the stream that is current when the node is created. Built
    on the legacy default stream (``"current"``, the stage-7 round-1 behaviour, kept as the
    negative control), those nodes make every backward that TE warms up and captures on its
    own stream synchronize with the legacy stream: torch's "AccumulateGrad node's stream does
    not match" warning in the warm-up, then ``cudaErrorStreamCaptureImplicit`` in the backward
    capture. ``"side"`` (default) builds DDP on a dedicated stream, as training does for every
    graph implementation other than ``none`` / ``full_iteration``
    (``megatron/training/training.py``, ``get_model``); a non-default stream joins the capture
    through the engine's event wait and rejoins at the end of the backward.
    """
    if ddp_stream not in ("side", "current"):
        raise ValueError(f"ddp_stream must be 'side' or 'current', not {ddp_stream!r}")
    # pylint: disable=import-outside-toplevel
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker

    set_moe_metrics_tracker(MoEMetricsTracker())
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True, overlap_grad_reduce=False, use_distributed_optimizer=False
    )

    def build(cfg):
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(
            seed, te_rng_tracker=cfg.use_te_rng_tracker, force_reset_rng=True
        )
        model = HybridModel(
            config=cfg,
            hybrid_stack_spec=hybrid_dsv4_stack_spec(cfg),
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            hybrid_layer_pattern=pattern_of(cfg),
            position_embedding_type="rope",
            pg_collection=pg_collection,
        ).cuda()
        model.train()
        current = torch.cuda.current_stream()
        stream = torch.cuda.Stream() if ddp_stream == "side" else current
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            wrapped = DistributedDataParallel(cfg, ddp_config, model, pg_collection=pg_collection)
        current.wait_stream(stream)
        return wrapped

    graph_model = build(config)
    eager_model = None
    if twin:
        if twin_config is None:
            raise ValueError("build_pair_models(twin=True) needs twin_config")
        eager_model = build(twin_config)
        copy_model_state(graph_model, eager_model)
    return graph_model, eager_model


def accumulate_grad_streams(model, run: Callable[[], Any]) -> Dict[str, Any]:
    """The stream each AccumulateGrad node DDP retains (``model.grad_accs``) runs on during
    ``run()`` (one eager forward + backward; the callers run it on a fresh side stream, as TE's
    warm-up does). A post hook on each node reads the current stream: autograd runs a node and
    its hooks under the node's own stream. ``on_legacy_stream`` names the parameters whose node
    is bound to the legacy default stream (each one breaks a backward capture)."""
    legacy = torch.cuda.default_stream().cuda_stream
    names = {id(param): name for name, param in model.named_parameters()}
    seen: Dict[str, int] = {}
    handles = []
    for node in getattr(model, "grad_accs", []):
        name = names.get(id(getattr(node, "variable", None)), f"<node {len(handles)}>")

        def hook(_grad_inputs, _grad_outputs, name=name):
            seen[name] = torch.cuda.current_stream().cuda_stream

        handles.append(node.register_hook(hook))
    try:
        run()
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    on_legacy = sorted(name for name, stream in seen.items() if stream == legacy)
    return dict(
        retained=len(handles),
        fired=len(seen),
        streams=len(set(seen.values())),
        on_legacy_stream=len(on_legacy),
        legacy_names=on_legacy[:48],
    )


def census_on_side_stream(model, side: "PairSide", inputs) -> Dict[str, Any]:
    """:func:`accumulate_grad_streams` of one eager block order of ``side`` run on a fresh
    side stream (the stream situation of TE's warm-up and capture)."""
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)

    def run():
        with torch.cuda.stream(stream):
            execute_order("block", side, inputs)

    try:
        return accumulate_grad_streams(model, run)
    finally:
        current.wait_stream(stream)


@contextlib.contextmanager
def deterministic_algorithms(env: Optional[Dict[str, str]] = None):
    """Training's ``--deterministic-mode`` torch state (``megatron/training/determinism.py``):
    the canonical environment (``apply_determinism_env``: ``NCCL_ALGO``,
    ``CUBLAS_WORKSPACE_CONFIG``, ``NVTE_ALLOW_NONDETERMINISTIC_ALGO``; effective only before
    NCCL / cuBLAS / TE first use) and ``torch.use_deterministic_algorithms(True)`` without
    the uninitialized-memory fill; the previous torch state is restored on exit.

    The spike configs set ``deterministic_mode=True``, which selects Megatron's deterministic
    paths, but torch's own ops follow only the global flag: without it the attention (C)
    layer's backward is not bitwise run to run (the unfused CSA core's index accumulation;
    determinism probe J7e), so neither are graph replays of it. ``warn_only``: an op without a
    deterministic implementation warns instead of failing a capture."""
    # pylint: disable=import-outside-toplevel
    import torch.utils.deterministic

    from megatron.training.determinism import apply_determinism_env

    apply_determinism_env(os.environ if env is None else env)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill


class _FirstDivergence:
    """Forward hooks on every submodule of one layer: run 0 keeps a clone of each submodule's
    first floating tensor output (and, through a tensor hook, of the gradient that output
    receives); later runs compare in place, in call order. ``forward`` / ``backward`` list
    ``(order, module name, max abs diff)`` of every output / output gradient that differed."""

    def __init__(self, layer: torch.nn.Module):
        self.layer = layer
        self.run = 0
        self.reference: Dict[Tuple[str, str, int], torch.Tensor] = {}
        self.forward: List[Tuple[int, str, float]] = []
        self.backward: List[Tuple[int, str, float]] = []
        self._seen: Dict[str, int] = {}
        self._handles = [
            module.register_forward_hook(functools.partial(self._hook, name))
            for name, module in layer.named_modules()
            if name
        ]

    @staticmethod
    def _first_tensor(output):
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item) and item.is_floating_point():
                    return item
        return None

    def _check(self, key, value: torch.Tensor, sink, order: int) -> None:
        if self.run == 0:
            self.reference[key] = value.detach().clone()
            return
        expected = self.reference.get(key)
        if expected is None or expected.shape != value.shape:
            sink.append((order, key[1], float("nan")))
            return
        equal, diff = _diff(value.detach(), expected)
        if not equal:
            sink.append((order, key[1], diff))

    def _hook(self, name, _module, _inputs, output) -> None:
        tensor = self._first_tensor(output)
        if tensor is None or not tensor.is_floating_point():
            return
        call = self._seen.get(name, 0)
        self._seen[name] = call + 1
        order = len(self._seen) * 1000 + call
        self._check(("fwd", name, call), tensor, self.forward, order)
        if tensor.requires_grad:
            tensor.register_hook(
                lambda grad, key=("bwd", name, call), order=order: self._check(
                    key, grad, self.backward, order
                )
            )

    def next_run(self) -> None:
        """Start the next run (call counts restart)."""
        self.run += 1
        self._seen = {}

    def remove(self) -> None:
        """Remove every hook and drop the references."""
        for handle in self._handles:
            handle.remove()
        self.reference = {}


def determinism_probe(model, config, kwargs_per_layer, inputs, *, repeats: int = 2) -> List[Dict]:
    """Per layer, ``repeats`` eager forward + backward calls from the same input (each layer's
    input computed once from ``x0``) and the same upstream gradient ``g``: whether the output,
    the input gradient and each parameter's ``main_grad`` are bitwise equal across the calls,
    and the first submodules whose output / output gradient differ (:class:`_FirstDivergence`).
    Localizes the eager run-to-run spread (T0) to a layer and component."""
    layers = decoder_layers(model)
    layer_inputs = []
    with torch.no_grad():
        hidden = inputs["x0"]
        for layer, kwargs in zip(layers, kwargs_per_layer):
            layer_inputs.append(hidden.clone())
            hidden = call_layer(layer, hidden, kwargs)
    records = []
    for index, (layer, kwargs) in enumerate(zip(layers, kwargs_per_layer)):
        tracer = _FirstDivergence(layer)
        runs = []
        try:
            for _ in range(repeats):
                reset_after_capture([model], config)
                leaf = layer_inputs[index].clone().requires_grad_(True)
                output = call_layer(layer, leaf, kwargs)
                torch.autograd.backward(output, inputs["g"])
                torch.cuda.synchronize()
                grads = {
                    name: param.main_grad.detach().clone()
                    for name, param in layer.named_parameters()
                    if getattr(param, "main_grad", None) is not None
                }
                runs.append((output.detach().clone(), leaf.grad.detach().clone(), grads))
                del output, leaf
                tracer.next_run()
        finally:
            tracer.remove()
            reset_after_capture([model], config)
        first_y, first_dx, first_grads = runs[0]
        record: Dict[str, Any] = dict(layer=index, kind=type(layer).__name__)
        bad_grads = []
        for y, dx, grads in runs[1:]:
            record["bitwise_y"], record["max_abs_diff_y"] = _diff(y, first_y)
            record["bitwise_dx"], record["max_abs_diff_dx"] = _diff(dx, first_dx)
            for name, grad in grads.items():
                equal, diff = _diff(grad, first_grads[name])
                if not equal:
                    bad_grads.append((name, diff))
        record["nondeterministic_grads"] = len(bad_grads)
        record["nondeterministic_grad_names"] = [
            f"{name} ({diff:.3g})" for name, diff in sorted(bad_grads, key=lambda x: -x[1])[:24]
        ]
        record["first_forward_divergence"] = [
            f"{name} ({diff:.3g})" for _o, name, diff in sorted(tracer.forward)[:12]
        ]
        record["forward_divergences"] = len(tracer.forward)
        # gradients flow last-to-first: the first divergence in backward order is the largest
        # forward order among the differing output gradients
        record["first_backward_divergence"] = [
            f"{name} ({diff:.3g})" for _o, name, diff in sorted(tracer.backward, reverse=True)[:12]
        ]
        record["backward_divergences"] = len(tracer.backward)
        records.append(record)
        del runs
    return records


def assert_never_captured(model) -> None:
    """The eager twin runs eager: no layer holds TE graphs (its layers' ``__call__`` replays
    graphs once any are attached, since it shares the graph config)."""
    captured = [
        i for i, layer in enumerate(decoder_layers(model)) if getattr(layer, "cuda_graphs", None)
    ]
    assert not captured, f"the eager twin holds CUDA graphs on layers {captured}"


def decoder_layers(model) -> List[torch.nn.Module]:
    """The HybridStack layers of a (DDP-wrapped) HybridModel."""
    module = getattr(model, "module", model)
    return list(module.decoder.layers)


def moe_layers(model) -> List[torch.nn.Module]:
    """Every MoELayer of the model."""
    from megatron.core.transformer.moe.moe_layer import (  # pylint: disable=import-outside-toplevel
        MoELayer,
    )

    return [module for module in model.modules() if isinstance(module, MoELayer)]


def copy_model_state(source, target) -> None:
    """Copy every floating parameter and buffer of ``source`` into ``target`` by name (in place:
    ``target``'s addresses stay, so a captured target keeps its graphs)."""
    src = getattr(source, "module", source)
    dst = getattr(target, "module", target)
    tensors = dict(src.named_parameters())
    tensors.update(dict(src.named_buffers()))
    with torch.no_grad():
        for name, tensor in list(dst.named_parameters()) + list(dst.named_buffers()):
            if tensor.is_floating_point() or name.endswith("local_tokens_per_expert"):
                tensor.copy_(tensors[name])


def static_kwargs(model, seq: int, micro_batch: int = 1) -> List[Dict[str, torch.Tensor]]:
    """Per layer, the non-hidden static inputs the TE helper would capture (identical keys and
    values), so eager calls see the same keyword surface as replays."""
    result = []
    for layer in decoder_layers(model):
        inputs = layer.get_layer_static_inputs(seq, micro_batch)
        inputs.pop("hidden_states")
        result.append(inputs)
    return result


def make_inputs(config, seq: int, *, seed: int, micro_batch: int = 1) -> Dict[str, torch.Tensor]:
    """Fixed-seed n-stream inputs: ``x0`` (pair-0 input), ``g`` (upstream gradient of pair 1),
    ``xb`` (the second input of order ``twice``), ``g0`` (the injected gradient of
    ``nested-asc``) and ``x_other`` / ``g_other`` (the eager "other work")."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (seq, micro_batch, NUM_STREAMS * config.hidden_size)

    def draw(scale=1.0):
        return (
            torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32) * scale
        ).to(torch.bfloat16)

    return dict(
        x0=draw(), g=draw(1e-2), xb=draw(), g0=draw(1e-2), x_other=draw(), g_other=draw(1e-2)
    )


def apply_factor(models: Sequence[Any], factor: float) -> None:
    """Set the HybridEP rank-capacity factor on both models (config and every manager), so the
    budget ``setup_metadata`` derives is the same on the graph model and the eager twin."""
    for model in models:
        if model is None:
            continue
        module = getattr(model, "module", model)
        module.config.moe_expert_rank_capacity_factor = factor
        for layer in moe_layers(model):
            layer.config.moe_expert_rank_capacity_factor = factor
            manager = layer.token_dispatcher._comm_manager  # pylint: disable=protected-access
            manager.moe_expert_rank_capacity_factor = factor
            manager.config.moe_expert_rank_capacity_factor = factor


def num_permuted_tokens(model) -> List[Optional[int]]:
    """The static budget of every HybridEP manager (``None`` before the first dispatch)."""
    budgets = []
    for layer in moe_layers(model):
        value = layer.token_dispatcher._comm_manager.num_permuted_tokens  # pylint: disable=W0212
        budgets.append(None if value is None else int(value))
    return budgets


def budget_bytes(config, budgets: Sequence[Optional[int]]) -> Dict[str, int]:
    """Bytes the static budget implies per MoE layer (the largest budget): the dispatched hidden
    states and the expert intermediates (FC1 output, GLU doubled; FC2 input), bf16."""
    rows = max((budget for budget in budgets if budget is not None), default=0)
    ffn = config.moe_ffn_hidden_size or 0
    return dict(
        budget_rows=rows,
        dispatched_hidden_bytes=2 * rows * config.hidden_size,
        fc1_output_bytes=2 * rows * ffn * (2 if config.gated_linear_unit else 1),
        fc2_input_bytes=2 * rows * ffn,
    )


def _local_over_budget(model) -> bool:
    flags = [layer.token_dispatcher.check_over_budget() for layer in moe_layers(model)]
    return any(bool(flag.any().item()) for flag in flags if flag is not None)


def over_budget_any(model, group=None, *, reset: bool = True) -> bool:
    """Whether any MoE dispatcher's sticky ``over_budget`` flag is set on any rank of ``group``
    (all-reduced MAX). HybridEP ORs every dispatch into the flag, so one read covers every
    replay since the last reset."""
    flags = []
    for layer in moe_layers(model):
        flag = layer.token_dispatcher.check_over_budget()
        if flag is not None:
            flags.append(flag.reshape(-1)[:1].to(torch.int32))
    value = torch.cat(flags).max().reshape(1) if flags else torch.zeros(1, dtype=torch.int32)
    value = value.to("cuda")
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=group)
    if reset:
        for layer in moe_layers(model):
            layer.token_dispatcher.reset_over_budget()
    return bool(value.item())


def reset_after_capture(models: Sequence[Any], config) -> None:
    """Replicate ``TECudaGraphHelper._reset_after_capture`` (TE's warm-up backward ran with
    ``torch.empty_like`` gradients) plus the router counts and the dispatchers' over-budget flags.
    """
    # pylint: disable=import-outside-toplevel
    from megatron.core.distributed.finalize_model_grads import reset_model_temporary_tensors
    from megatron.core.transformer.experimental_attention_variant.dsa_logging import (
        DSAIndexerLossLoggingHelper,
    )
    from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker

    models = [model for model in models if model is not None]
    for model in models:
        model.zero_grad_buffer()
    get_moe_metrics_tracker().clear()
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker(preserve_groups=True)
    reset_model_temporary_tensors(config, models)
    for model in models:
        for layer in moe_layers(model):
            counts = getattr(layer.router, "local_tokens_per_expert", None)
            if counts is not None:
                counts.zero_()
            layer.token_dispatcher.reset_over_budget()
        for param in getattr(model, "module", model).parameters():
            param.grad = None


# ---------------------------------------------------------------------------------------------
# Graph sides (callables per pair) and replay orders
# ---------------------------------------------------------------------------------------------


# --skew-rank0-ms: rank 0 waits this long on the host and on its stream before every pair call
# (gpu-fix-r1: is the replay hang of a rank under nsys a rank-skew hazard of the captured
# HybridEP layers, or nsys itself?)
RANK0_SKEW_MS = [0.0]


@dataclasses.dataclass
class PairSide:
    """A model seen as two pairs: ``parts[i]`` are the callables (Tensor -> Tensor) pair ``i``
    applies in sequence (two layer calls for arm A and the eager twin, one graphed callable for
    arm B). ``pool`` is the graph pool (None for eager)."""

    name: str
    model: Any
    parts: List[List[Callable[[torch.Tensor], torch.Tensor]]]
    pool: Optional[Tuple[int, int]] = None
    names: Optional[List[List[str]]] = None

    def pair(self, index: int, hidden: torch.Tensor) -> torch.Tensor:
        """Apply pair ``index`` (after the rank-0 skew of ``--skew-rank0-ms``, if any)."""
        if RANK0_SKEW_MS[0] > 0 and _rank() == 0:
            # rank 0 late on the host and on its stream (emulates a slow / profiled rank)
            time.sleep(RANK0_SKEW_MS[0] / 1000)
            torch.cuda._sleep(int(RANK0_SKEW_MS[0] * 1e6))  # pylint: disable=protected-access
        for part in self.parts[index]:
            hidden = part(hidden)
        return hidden


def layer_quant_context(config, layer_number: int):
    """The quantization context HybridStack enters around the call of layer ``layer_number``
    (1-indexed; ``hybrid_block.py`` ``get_inner_quant_context``): the per-layer FP8 context
    for non-delayed recipes (MXFP8), the whole-pass FP8 context for the delayed recipe, the
    FP4 context, or a no-op for BF16. Layers do not enter it themselves, so every direct layer
    or graphed-callable call in this probe goes through it."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.enums import Fp8Recipe
    from megatron.core.fp4_utils import get_fp4_context
    from megatron.core.fp8_utils import get_fp8_context

    if config.fp8:
        if config.fp8_recipe == Fp8Recipe.delayed:
            return get_fp8_context(config)
        return get_fp8_context(config, layer_number - 1)
    if getattr(config, "fp4", None) is not None:
        return get_fp4_context(config, layer_number - 1)
    return contextlib.nullcontext()


def call_layer(layer, hidden: torch.Tensor, kwargs: Dict[str, Any]) -> torch.Tensor:
    """``layer(hidden_states=hidden, **kwargs)[0]`` inside the layer's quantization context."""
    with layer_quant_context(layer.config, layer.layer_number):
        return layer(hidden_states=hidden, **kwargs)[0]


def layer_side(name: str, model, kwargs_per_layer: Sequence[Dict[str, Any]], pool=None) -> PairSide:
    """Pairs through the layers' ``__call__`` (eager on the twin; TE replays on a model whose
    layers hold ``cuda_graphs``, i.e. arm A), each call inside the layer's quantization context
    as HybridStack enters it."""
    layers = decoder_layers(model)

    def call(layer, kwargs):
        return lambda hidden: call_layer(layer, hidden, kwargs)

    parts = [
        [
            call(layers[2 * i], kwargs_per_layer[2 * i]),
            call(layers[2 * i + 1], kwargs_per_layer[2 * i + 1]),
        ]
        for i in range(len(layers) // 2)
    ]
    names = [[f"L{2 * i}", f"L{2 * i + 1}"] for i in range(len(layers) // 2)]
    return PairSide(name=name, model=model, parts=parts, pool=pool, names=names)


def graphed_side(name: str, model, graphed: Sequence[Callable], kwargs_per_pair, pool) -> PairSide:
    """Pairs through arm-B graphed callables (one per pair), each replay inside the pair's
    attention layer's quantization context (one context per pair callable)."""
    layers = decoder_layers(model)

    def call(fn, kwargs, layer):
        def replay(hidden):
            with layer_quant_context(layer.config, layer.layer_number):
                return fn(hidden, **kwargs)[0]

        return replay

    parts = [
        [call(fn, kwargs, layers[2 * index])]
        for index, (fn, kwargs) in enumerate(zip(graphed, kwargs_per_pair))
    ]
    return PairSide(name=name, model=model, parts=parts, pool=pool, names=[["P0"], ["P1"]])


class PoisonGraph:
    """A graph captured on an arm's own pool that NaN-fills every block the pool holds free
    (stage-7 design 5.6, D-S7-11). ``replay()`` is a no-op when the pool had no free block."""

    def __init__(self, graph, coverage: Tuple[int, int], blocks: int, pool_delta: int):
        self.graph = graph
        self.covered, self.inactive = coverage
        self.blocks = blocks
        self.pool_delta = pool_delta

    def replay(self) -> None:
        """NaN-fill the pool's free blocks."""
        if self.graph is not None:
            self.graph.replay()

    def reset(self) -> None:
        """Release the poison graph."""
        if self.graph is not None:
            self.graph.reset()
            self.graph = None

    def record(self) -> Dict[str, Any]:
        """JSON fields of the poison graph."""
        return dict(
            poison_coverage=(self.covered / self.inactive) if self.inactive else 1.0,
            poison_inactive_bytes=self.inactive,
            poison_blocks=self.blocks,
            poison_pool_delta=self.pool_delta,
        )


def pool_segments(pool) -> List[Dict[str, Any]]:
    """Allocator segments of graph pool ``pool``."""
    key = tuple(pool)
    return [
        segment
        for segment in torch.cuda.memory_snapshot()
        if tuple(segment.get("segment_pool_id", (0, 0))) == key
    ]


def pool_bytes(pool) -> Dict[str, int]:
    """Reserved / active bytes and segment count of graph pool ``pool``."""
    segments = pool_segments(pool)
    active = sum(
        block["size"]
        for segment in segments
        for block in segment["blocks"]
        if block["state"].startswith("active")
    )
    return dict(
        pool_bytes=sum(segment["total_size"] for segment in segments),
        pool_active_bytes=active,
        pool_segments=len(segments),
    )


def pool_ids() -> set:
    """Every graph pool id the allocator holds segments for."""
    return {
        tuple(segment["segment_pool_id"])
        for segment in torch.cuda.memory_snapshot()
        if tuple(segment.get("segment_pool_id", (0, 0))) != (0, 0)
    }


def poison_graph_pool(pool) -> PoisonGraph:
    """Capture a graph on ``pool`` that allocates one uint8 tensor per inactive block (largest
    first) and fills it with :data:`POISON_INT32`; the tensors are released after capture, so
    replaying the graph writes NaN into memory the pool considers free."""
    torch.cuda.synchronize()
    blocks = sorted(
        (
            (block["address"], block["size"])
            for segment in pool_segments(pool)
            for block in segment["blocks"]
            if block["state"] == "inactive"
        ),
        key=lambda block: -block[1],
    )
    before = pool_bytes(pool)["pool_bytes"]
    if not blocks:
        return PoisonGraph(None, (0, 0), 0, 0)
    graph = torch.cuda.CUDAGraph()
    tensors = []
    with torch.cuda.graph(graph, pool=pool):
        for _, size in blocks:
            tensor = torch.empty(size, dtype=torch.uint8, device="cuda")
            tensor[: size - size % 4].view(torch.int32).fill_(POISON_INT32)
            tensors.append(tensor)
    ranges = [(tensor.data_ptr(), tensor.numel()) for tensor in tensors]
    del tensors
    torch.cuda.synchronize()
    return PoisonGraph(
        graph, poison_coverage(blocks, ranges), len(blocks), pool_bytes(pool)["pool_bytes"] - before
    )


def poison_default_allocator(*, keep_free_gb: float = 2.0, chunk_gb: float = 1.0) -> int:
    """Fill the free device memory minus ``keep_free_gb`` with :data:`POISON_INT32` in
    ``chunk_gb`` blocks of the default allocator and free them (they stay cached, NaN-filled).
    Returns the poisoned bytes; the caller runs its probe and then ``empty_cache()``."""
    torch.cuda.empty_cache()
    chunk = int(chunk_gb * 2**30)
    keep = int(keep_free_gb * 2**30)
    chunks, total = [], 0
    while torch.cuda.mem_get_info()[0] - chunk > keep:
        try:
            tensor = torch.empty(chunk, dtype=torch.uint8, device="cuda")
        except torch.OutOfMemoryError:
            break
        tensor.view(torch.int32).fill_(POISON_INT32)
        chunks.append(tensor)
        total += chunk
    torch.cuda.synchronize()
    del chunks
    return total


def _leaf(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().requires_grad_(True)


def _out(tensor: torch.Tensor, copy_out: bool) -> torch.Tensor:
    return tensor.detach().clone() if copy_out else tensor.detach()


def _grad_of(leaf: torch.Tensor, copy_out: bool) -> torch.Tensor:
    grad = leaf.grad
    leaf.grad = None
    return grad.clone() if copy_out else grad


def execute_order(
    order: str,
    side: PairSide,
    inputs: Dict[str, torch.Tensor],
    *,
    poison: Optional[PoisonGraph] = None,
    poison_where: str = "items",
    copy_out: bool = True,
    other_work: Optional[Callable[[], None]] = None,
    upstream_dx1: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Run ``order`` (stage-7 design 5.6) on ``side``; returns the pair outputs and input
    gradients (``y0``, ``y1``, ``dx1``, ``dx0`` and order-specific extras). With ``copy_out``
    every output and input gradient is cloned out of the pool before the next replay;
    without it, the returned tensors are the (possibly overwritten) pool tensors.

    ``poison_where = "items"`` replays the poison graph before and after every item;
    ``"fwd-bwd"`` replays it between pair 0's forward and its backward (probe P6)."""
    x0, g = inputs["x0"], inputs["g"]

    def boundary():
        if poison is not None and poison_where == "items":
            poison.replay()

    def between_fwd_bwd():
        if poison is not None and poison_where == "fwd-bwd":
            poison.replay()

    out: Dict[str, torch.Tensor] = {}
    if order in ("block", "sweep", "interleaved"):
        if order == "sweep":
            boundary()
            with torch.no_grad():
                swept0 = side.pair(0, x0)
                stash1 = swept0.clone()
                out["sweep_y0"] = stash1
                out["sweep_y1"] = _out(side.pair(1, stash1), True)
        boundary()
        x_a = _leaf(x0)
        y0 = side.pair(0, x_a)
        out["y0"] = _out(y0, copy_out)
        between_fwd_bwd()
        if order == "interleaved" and other_work is not None:
            other_work()
        x1 = _leaf(out["y0"])
        y1 = side.pair(1, x1)
        out["y1"] = _out(y1, copy_out)
        torch.autograd.backward(y1, g)
        out["dx1"] = _grad_of(x1, copy_out)
        if order == "interleaved" and other_work is not None:
            other_work()
        torch.autograd.backward(y0, out["dx1"])
        out["dx0"] = _grad_of(x_a, copy_out)
        boundary()
    elif order == "nested":
        boundary()
        with torch.no_grad():
            stash1 = side.pair(0, x0).clone()
            out["rebuild_y1"] = _out(side.pair(1, stash1), True)
        out["rebuild_y0"] = stash1
        boundary()
        x1 = _leaf(stash1)
        y1 = side.pair(1, x1)
        out["y1"] = _out(y1, copy_out)
        torch.autograd.backward(y1, g)
        out["dx1"] = _grad_of(x1, copy_out)
        boundary()
        x_a = _leaf(x0)
        y0 = side.pair(0, x_a)
        out["y0"] = _out(y0, copy_out)
        between_fwd_bwd()
        torch.autograd.backward(y0, out["dx1"])
        out["dx0"] = _grad_of(x_a, copy_out)
        boundary()
    elif order == "nested-asc":
        boundary()
        x_a = _leaf(x0)
        y0 = side.pair(0, x_a)
        out["y0"] = _out(y0, copy_out)
        between_fwd_bwd()
        torch.autograd.backward(y0, inputs["g0"])
        out["dx0_injected"] = _grad_of(x_a, copy_out)
        boundary()
        x1 = _leaf(out["y0"])
        y1 = side.pair(1, x1)
        out["y1"] = _out(y1, copy_out)
        torch.autograd.backward(y1, g)
        out["dx1"] = _grad_of(x1, copy_out)
        boundary()
    elif order == "twice":
        if upstream_dx1 is None:
            raise ValueError("order 'twice' needs the oracle's dx1 as pair 0's upstream gradient")
        boundary()
        x_a = _leaf(x0)
        y_a = side.pair(0, x_a)
        x_b = _leaf(inputs["xb"])
        side.pair(0, x_b)
        out["y0"] = _out(y_a, True)
        torch.autograd.backward(y_a, upstream_dx1)
        out["dx0"] = _grad_of(x_a, True)
        x_b.grad = None
        boundary()
    else:
        raise ValueError(f"unknown order {order!r}")
    if not copy_out:
        out = {key: value.clone() for key, value in out.items()}
    return out


def _named_main_grads(model) -> List[Tuple[str, torch.Tensor]]:
    module = getattr(model, "module", model)
    return [
        (name, param.main_grad)
        for name, param in module.named_parameters()
        if getattr(param, "main_grad", None) is not None
    ]


def _router_counts(model) -> List[torch.Tensor]:
    return [
        layer.router.local_tokens_per_expert
        for layer in moe_layers(model)
        if getattr(layer.router, "local_tokens_per_expert", None) is not None
    ]


def _dsa_values() -> Optional[torch.Tensor]:
    from megatron.core.transformer.experimental_attention_variant.dsa_logging import (  # pylint: disable=C0415
        DSAIndexerLossLoggingHelper,
    )

    return DSAIndexerLossLoggingHelper.tracker.get("values")


def _diff(actual: torch.Tensor, expected: torch.Tensor) -> Tuple[bool, float]:
    expected = expected.to(actual.device)
    if actual.shape != expected.shape:
        return False, float("nan")
    equal = torch.equal(actual, expected)
    if equal:
        return True, 0.0
    return False, float((actual.float() - expected.float()).abs().max().item())


class Snapshot:
    """Host copy of one order's result: pair outputs / input gradients, every ``main_grad``,
    the routers' token counts and the DSA metric tracker. ``compare`` is bitwise (NaN never
    equals)."""

    def __init__(self, result: Dict[str, torch.Tensor], model, forwards: int = 1):
        self.forwards = forwards
        self.tensors = {key: value.detach().cpu() for key, value in result.items()}
        self.grads = {name: grad.detach().cpu() for name, grad in _named_main_grads(model)}
        self.counts = [count.detach().cpu() for count in _router_counts(model)]
        values = _dsa_values()
        self.dsa = None if values is None else values.detach().cpu()

    def compare(
        self,
        result: Dict[str, torch.Tensor],
        model,
        *,
        keys: Optional[Iterable[str]] = None,
        grads: bool = True,
        forwards: int = 1,
    ) -> Dict[str, Any]:
        """Bitwise comparison of ``result`` / ``model`` with this snapshot. ``forwards`` is the
        number of forward replays per pair of ``result`` (router counts and DSA values scale by
        it relative to the snapshot's own ``forwards``)."""
        record: Dict[str, Any] = {}
        mismatches: List[str] = []
        worst = 0.0
        for key in keys if keys is not None else ("y0", "y1", "dx1", "dx0"):
            if key not in result or key not in self.tensors:
                continue
            equal, diff = _diff(result[key], self.tensors[key])
            record[f"bitwise_{key}"] = equal
            record[f"max_abs_diff_{key}"] = diff
            worst = nan_max(worst, diff)
            if not equal:
                mismatches.append(key)
        # The no-grad sweep / rebuild replays (``sweep`` and ``nested`` orders) must reproduce
        # the oracle's pair outputs too: they are part of the bitwise verdict (D35).
        for key, reference in NO_GRAD_OUTPUTS.items():
            if key not in result or reference not in self.tensors:
                continue
            equal, diff = _diff(result[key], self.tensors[reference])
            record[f"bitwise_{key}"] = equal
            record[f"max_abs_diff_{key}"] = diff
            worst = nan_max(worst, diff)
            if not equal:
                mismatches.append(key)
        if grads:
            grads_equal, grad_diff, bad = True, 0.0, []
            for name, grad in _named_main_grads(model):
                equal, diff = _diff(grad, self.grads[name])
                if not equal:
                    grads_equal = False
                    bad.append(name)
                    grad_diff = nan_max(grad_diff, diff)
            record.update(
                bitwise_main_grad=grads_equal,
                max_abs_diff_main_grad=grad_diff,
                main_grad_mismatches=bad[:8],
                main_grad_mismatch_count=len(bad),
            )
            worst = nan_max(worst, grad_diff)
            if not grads_equal:
                mismatches.append("main_grad")
        counts = _router_counts(model)
        record["router_counts_match"] = len(counts) == len(self.counts) and all(
            torch.equal(count.cpu().double() * self.forwards, expected.double() * forwards)
            for count, expected in zip(counts, self.counts)
        )
        values = _dsa_values()
        if self.dsa is not None and values is not None:
            record["dsa_values_match"] = torch.equal(
                values.cpu().double() * self.forwards, self.dsa.double() * forwards
            )
        record["mismatches"] = mismatches
        record["max_abs_diff"] = worst
        record["bitwise"] = not mismatches
        return record


# ---------------------------------------------------------------------------------------------
# Captures
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass
class CaptureResult:
    """One capture: its graphs and static surfaces, or the precise failure."""

    arm: str
    capture: str
    ok: bool
    capture_s: float = 0.0
    error: Optional[Dict[str, Any]] = None
    helper: Any = None
    graphed: Optional[List[Callable]] = None
    pool: Optional[Tuple[int, int]] = None
    sample_args: Optional[List[Tuple[torch.Tensor, ...]]] = None
    sample_kwargs: Optional[List[Dict[str, Any]]] = None
    kwargs_per_layer: Optional[List[Dict[str, Any]]] = None
    pool_record: Dict[str, Any] = dataclasses.field(default_factory=dict)
    over_budget: Optional[bool] = None  # the flag after TE's warm-up + capture (read, then reset)

    def delete(self) -> None:
        """Release the graphs (TE >= 2.10 ``reset``) and the static surfaces. The caller drops
        its own references (graphed sides, sample inputs) first; DDP / TE objects form
        reference cycles, so collect before returning the pool's memory."""
        if self.helper is not None and self.helper.graphs_created():
            self.helper.delete_cuda_graphs()
        for graphed in self.graphed or []:
            reset = getattr(graphed, "reset", None)
            if reset is not None:
                reset()
        self.graphed = None
        self.sample_args = None
        self.sample_kwargs = None
        self.kwargs_per_layer = None
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


@contextlib.contextmanager
def capture_step_tracker(labels: Sequence[str]):
    """Name the capture step an exception happens in: ``warm-up`` before TE's first
    ``capture_begin``, then the label of the current capture (:func:`capture_labels`)."""
    state = {"step": "warm-up", "count": 0}
    original = torch.cuda.CUDAGraph.capture_begin

    def capture_begin(graph, *args, **kwargs):
        index = state["count"]
        state["count"] += 1
        state["step"] = labels[index] if index < len(labels) else f"extra capture {index}"
        return original(graph, *args, **kwargs)

    torch.cuda.CUDAGraph.capture_begin = capture_begin
    try:
        yield state
    finally:
        torch.cuda.CUDAGraph.capture_begin = original


def exception_chain(error: BaseException, limit: int = 5) -> List[BaseException]:
    """``error`` and the exceptions it was raised from or during (``__cause__``, else
    ``__context__``), outermost first, at most ``limit``. ``torch.cuda.graph.__exit__`` always
    calls ``capture_end()``: an op that fails inside a capture surfaces as a capture-invalidated
    error whose ``__context__`` is the failing call's exception."""
    chain: List[BaseException] = []
    current: Optional[BaseException] = error
    while current is not None and len(chain) < limit and all(current is not e for e in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _exception_entry(error: BaseException) -> Dict[str, Any]:
    frames = traceback.extract_tb(error.__traceback__)[-30:]
    return dict(
        type=type(error).__name__,
        message=str(error)[:4000],
        frames=[f"{frame.filename}:{frame.lineno} {frame.name}" for frame in frames],
    )


def failure_record(error: BaseException, step: str, sync_report=None) -> Dict[str, Any]:
    """The cause (``type``, ``message``, last 30 ``frames`` of the innermost exception of the
    chain), the exception that reached the probe (``raised``), every level of the chain
    (``chain``, outermost first), the sticky CUDA error (if any), the step it failed in and the
    host synchronizations seen before capture."""
    chain = [_exception_entry(level) for level in exception_chain(error)]
    sticky = None
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception as cuda_error:  # pylint: disable=broad-except
            sticky = str(cuda_error)
    return dict(
        chain[-1],
        raised=dict(type=chain[0]["type"], message=chain[0]["message"][:1000]),
        chain=chain,
        cuda_error=sticky,
        step=step,
        sync_debug=list(sync_report or []),
    )


def _end_capture_flags() -> None:
    """Clear the Megatron and TE capture flags after a failed capture (neither helper uses
    ``finally``)."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer import cuda_graphs

    cuda_graphs._set_capture_end()  # pylint: disable=protected-access
    try:
        from transformer_engine.pytorch.graph import set_capture_end

        set_capture_end()
    except Exception:  # pylint: disable=broad-except
        pass
    # TECudaGraphHelper._start_capturing freezes the GC (CUDA_GRAPH_CAPTURE_FREEZE_GC).
    gc.unfreeze()


@contextlib.contextmanager
def _restore_class_calls(classes: Iterable[type]):
    """Restore the callables' class ``__call__`` afterwards: TE's ``make_graphed_callables``
    wraps it in an FP8 autocast and restores it only when the capture returns normally."""
    saved = {cls: cls.__dict__.get("__call__") for cls in set(classes)}
    try:
        yield
    finally:
        for cls, call in saved.items():
            if call is not None:
                cls.__call__ = call
            elif "__call__" in cls.__dict__:
                delattr(cls, "__call__")


@contextlib.contextmanager
def _record_make_graphed_callables(sink: Dict[str, Any]):
    """Record the (TE-rebound, after return) sample arguments the dev helper passes to TE."""
    from megatron.core.transformer import cuda_graphs  # pylint: disable=import-outside-toplevel

    original = cuda_graphs.make_graphed_callables

    def recording(callables, sample_args, **kwargs):
        sink["sample_args"] = sample_args
        sink["kwargs"] = kwargs
        return original(callables, sample_args, **kwargs)

    cuda_graphs.make_graphed_callables = recording
    try:
        yield sink
    finally:
        cuda_graphs.make_graphed_callables = original


def capture_with_helper(
    model, config, *, seq: int, pg_collection, sync_report=None
) -> CaptureResult:
    """Arm A: dev's ``TECudaGraphHelper`` (``cuda_graph_modules = []``: each mHC wrapper whole;
    one microbatch at PP 1, ``_order = [1, -1]``). Manual hooks only if DDP uses forward
    pre-hooks (F14); otherwise every captured layer's ``cuda_graph_manual_hooks`` must be
    empty."""
    from megatron.core.transformer.cuda_graphs import (  # pylint: disable=import-outside-toplevel
        TECudaGraphHelper,
    )

    layers = decoder_layers(model)
    labels = capture_labels([1, -1], [[f"L{i}" for i in range(len(layers))]])
    before = pool_ids()
    sink: Dict[str, Any] = {}
    result = CaptureResult(arm="helper", capture="helper", ok=False)
    with capture_step_tracker(labels) as state, _restore_class_calls(map(type, layers)):
        try:
            helper = TECudaGraphHelper(
                model=[model],
                config=config,
                seq_length=seq,
                micro_batch_size=1,
                optimizers=[],
                pg_collection=pg_collection,
            )
            result.helper = helper
            start = time.perf_counter()
            with _record_make_graphed_callables(sink):
                helper.create_cudagraphs()
            torch.cuda.synchronize()
            result.capture_s = time.perf_counter() - start
            state["step"] = "post-capture"
        except Exception as error:  # pylint: disable=broad-except
            _end_capture_flags()
            result.error = failure_record(error, state["step"], sync_report)
            return result
    try:
        if getattr(model, "use_forward_hook", False):
            helper.cuda_graph_set_manual_hooks()
        else:
            hooked = [i for i, layer in enumerate(layers) if layer.cuda_graph_manual_hooks]
            assert not hooked, f"manual hooks registered without forward pre-hooks: {hooked}"
    except Exception as error:  # pylint: disable=broad-except
        result.error = failure_record(error, "post-capture hooks", sync_report)
        return result
    new_pools = sorted(pool_ids() - before)
    result.pool = new_pools[0] if len(new_pools) == 1 else None
    result.pool_record = dict(pool_ids_new=[list(p) for p in new_pools])
    if result.pool is not None:
        result.pool_record.update(pool_bytes(result.pool))
    result.sample_args = sink.get("sample_args")
    result.sample_kwargs = sink.get("kwargs", {}).get("sample_kwargs")
    result.kwargs_per_layer = list(result.sample_kwargs or [{} for _ in layers])
    result.ok = True
    return result


class PairCallable(torch.nn.Module):
    """One attention + E pair as a single TE callable: the attention wrapper's and then the MoE
    wrapper's ``_te_cuda_graph_capture_impl`` (the same bodies the dev helper captures).

    ``parameters()`` are both layers'. MoE-layer keyword inputs travel with a ``moe__`` prefix
    so the two layers' static keyword surfaces stay separate."""

    MOE_PREFIX = "moe__"

    def __init__(self, attention_layer: torch.nn.Module, moe_layer: torch.nn.Module):
        super().__init__()
        self.attention_layer = attention_layer
        self.moe_layer = moe_layer
        for layer in (attention_layer, moe_layer):
            layer._mhc_recompute_manager = None  # pylint: disable=protected-access

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        """Attention wrapper, then MoE wrapper; returns ``(hidden_states,)``."""
        # pylint: disable=protected-access
        prefix = self.MOE_PREFIX
        attention_kwargs = {
            key: value for key, value in kwargs.items() if not key.startswith(prefix)
        }
        moe_kwargs = {
            key[len(prefix) :]: value for key, value in kwargs.items() if key.startswith(prefix)
        }
        hidden = self.attention_layer._te_cuda_graph_capture_impl(
            hidden_states, **attention_kwargs
        )[0]
        return self.moe_layer._te_cuda_graph_capture_impl(hidden, **moe_kwargs)


def pair_sample_inputs(model, config, *, seq: int, pg_collection):
    """Arm-B static inputs: exactly the dev helper's sample arguments of each pair's attention
    callable, the merged (``moe__``-prefixed) keyword inputs per pair, the helper's warm-up
    count and amax-reduction group."""
    from megatron.core.transformer.cuda_graphs import (  # pylint: disable=import-outside-toplevel
        TECudaGraphHelper,
    )

    helper = TECudaGraphHelper(
        model=[model],
        config=config,
        seq_length=seq,
        micro_batch_size=1,
        optimizers=[],
        pg_collection=pg_collection,
    )
    sample_args, make_kwargs = helper._get_cuda_graph_input_data()  # pylint: disable=W0212
    sample_kwargs = make_kwargs.get("sample_kwargs") or [{} for _ in sample_args]
    pair_args, pair_kwargs = [], []
    for index in range(0, len(sample_args), 2):
        pair_args.append(tuple(sample_args[index]))
        merged = dict(sample_kwargs[index])
        for key, value in sample_kwargs[index + 1].items():
            merged[PairCallable.MOE_PREFIX + key] = value
        pair_kwargs.append(merged)
    fp8_group = None
    if config.fp8:
        fp8_group = helper._get_amax_reduction_group(  # pylint: disable=protected-access
            with_context_parallel=True, tp_only_amax_red=config.tp_only_amax_red
        )
    return pair_args, pair_kwargs, make_kwargs["num_warmup_iters"], fp8_group


def capture_pairs(
    model,
    config,
    sample_args,
    sample_kwargs,
    *,
    precision: str,
    order: str = "block",
    reuse_buffers: bool = True,
    num_warmup_iters: int = 3,
    fp8_group=None,
    sync_report=None,
) -> CaptureResult:
    """Arm B: one TE graph per attention + E pair on an explicit pool (``block``:
    ``_order = [1, -1]`` over one chunk of both pairs; ``nested``: ``[2, -2, 1, -1]`` with one
    pair per chunk). ``fp8_weight_caching=False`` (the loop's transpose-cache-off world)."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.fp8_utils import get_fp8_recipe
    from megatron.core.transformer import cuda_graphs
    from megatron.core.transformer.moe.paged_stash import paged_stash_te_graph_capture

    te_order, per_chunk, chunk_names = CAPTURE_ORDERS[order]
    layers = decoder_layers(model)
    pairs = tuple(PairCallable(layers[2 * i], layers[2 * i + 1]) for i in range(len(layers) // 2))
    for pair in pairs:
        pair.train()
    pool = torch.cuda.graph_pool_handle()
    args_list = [tuple(args) for args in sample_args]
    kwargs = dict(
        sample_kwargs=tuple(sample_kwargs),
        num_warmup_iters=num_warmup_iters,
        allow_unused_input=True,
        retain_graph_in_backward=False,
        _order=list(te_order),
        _num_layers_per_chunk=list(per_chunk),
        _reuse_graph_input_output_buffers=reuse_buffers,
        pool=pool,
    )
    if _te_min("2.19.0"):
        kwargs["clone_param_grads_on_return"] = False
    if precision == "mxfp8":
        kwargs.update(
            fp8_enabled=(True,) * len(pairs),
            fp8_recipe=get_fp8_recipe(config),
            fp8_weight_caching=False,
            fp8_group=fp8_group,
        )
    else:
        kwargs["fp8_enabled"] = False
    result = CaptureResult(arm="pair", capture=order, ok=False, pool=pool)
    labels = capture_labels(te_order, chunk_names)
    with capture_step_tracker(labels) as state, _restore_class_calls([PairCallable]):
        try:
            torch.cuda.synchronize()
            cuda_graphs._set_capture_start()  # pylint: disable=protected-access
            start = time.perf_counter()
            try:
                with paged_stash_te_graph_capture(False, order=list(te_order), config=config):
                    graphed = cuda_graphs.make_graphed_callables(pairs, args_list, **kwargs)
            finally:
                cuda_graphs._set_capture_end()  # pylint: disable=protected-access
            torch.cuda.synchronize()
            result.capture_s = time.perf_counter() - start
            state["step"] = "post-capture"
        except Exception as error:  # pylint: disable=broad-except
            _end_capture_flags()
            result.error = failure_record(error, state["step"], sync_report)
            return result
    result.graphed = list(graphed)
    result.sample_args = args_list
    result.sample_kwargs = list(sample_kwargs)
    result.pool_record = pool_bytes(pool)
    result.ok = True
    return result


def static_io_record(capture: CaptureResult, side: PairSide, inputs) -> Dict[str, Any]:
    """Sizes and addresses of each callable's static input, its replay output and (pairs) its
    input-gradient slot; which of them alias (TE's buffer reuse)."""
    slots: Dict[str, Tuple[int, int]] = {}
    for index, args in enumerate(capture.sample_args or []):
        for position, tensor in enumerate(args):
            if torch.is_tensor(tensor):
                slots[f"in{index}.{position}"] = (
                    tensor.data_ptr(),
                    tensor.numel() * tensor.element_size(),
                )
    x_a = _leaf(inputs["x0"])
    y0 = side.pair(0, x_a)
    x1 = _leaf(y0.detach().clone())
    y1 = side.pair(1, x1)
    slots["out.P0"] = (y0.data_ptr(), y0.numel() * y0.element_size())
    slots["out.P1"] = (y1.data_ptr(), y1.numel() * y1.element_size())
    torch.autograd.backward(y1, inputs["g"])
    slots["grad_in.P1"] = (x1.grad.data_ptr(), x1.grad.numel() * x1.grad.element_size())
    dx1 = x1.grad.clone()
    torch.autograd.backward(y0, dx1)
    slots["grad_in.P0"] = (x_a.grad.data_ptr(), x_a.grad.numel() * x_a.grad.element_size())
    by_ptr: Dict[int, List[str]] = defaultdict(list)
    for name, (ptr, _) in slots.items():
        by_ptr[ptr].append(name)
    return dict(
        static_io_bytes={name: size for name, (_, size) in slots.items()},
        slot_ptrs={name: ptr for name, (ptr, _) in slots.items()},
        slot_aliases=[names for names in by_ptr.values() if len(names) > 1],
    )


# ---------------------------------------------------------------------------------------------
# Static budget, diagnostics, saved bytes, timing
# ---------------------------------------------------------------------------------------------


def size_static_budget(
    eager_model,
    inputs: Dict[str, torch.Tensor],
    *,
    kwargs_per_layer,
    steps: int = 10,
    seed: int = 7,
) -> Dict[str, Any]:
    """Eager sizing pass (design 5.4): ``steps`` forward + backward passes on fixed-seed random
    n-stream inputs; per MoE call and rank ``ratio = tokens received / (rows x topk)``."""
    side = layer_side("sizing", eager_model, kwargs_per_layer)
    calls: List[Tuple[int, torch.Tensor, int]] = []
    restore = []
    for layer in moe_layers(eager_model):
        manager = layer.token_dispatcher._comm_manager  # pylint: disable=protected-access
        topk = manager.router_topk

        def dispatch(hidden_states, *args, _manager=manager, _topk=topk, **kw):
            output = type(_manager).dispatch(_manager, hidden_states, *args, **kw)
            rows = _manager._padded_num_tokens or hidden_states.shape[0]  # pylint: disable=W0212
            counts = _manager.tokens_per_expert.detach().clone()
            calls.append((rows, counts, _topk, int(getattr(_manager, "pad_multiple", 0) or 0)))
            return output

        manager.dispatch = dispatch
        restore.append(manager)
    generator = torch.Generator(device="cuda").manual_seed(seed + 1000 * _rank())
    try:
        for _ in range(steps):
            x = torch.randn(inputs["x0"].shape, generator=generator, device="cuda").to(
                torch.bfloat16
            )
            g = torch.randn(inputs["x0"].shape, generator=generator, device="cuda").to(
                torch.bfloat16
            )
            execute_order("block", side, dict(x0=x, g=g * 1e-2))
    finally:
        for manager in restore:
            del manager.dispatch
    # The budget is in permuted rows, which carry every expert's alignment padding: size from the
    # padded counts (a no-op if HybridEP already returns padded counts; both are recorded).
    received = [padded_rows(counts, pad) for _, counts, _, pad in calls]
    ratios = [value / (rows * topk) for value, (rows, _, topk, _) in zip(received, calls)]
    raw_ratios = [float(counts.sum().item()) / (rows * topk) for rows, counts, topk, _ in calls]
    already_padded = all(
        pad <= 1 or bool((counts % pad == 0).all().item()) for _, counts, _, pad in calls
    )
    over = over_budget_any(eager_model)
    gathered: List[Any] = [None] * _world()
    if dist.is_initialized():
        dist.all_gather_object(gathered, ratios)
    else:
        gathered = [ratios]
    every = [ratio for per_rank in gathered for ratio in per_rank]
    eager_model.zero_grad_buffer()
    return dict(
        max_recv_ratio=max(every) if every else 0.0,
        mean_recv_ratio=statistics.fmean(every) if every else 0.0,
        p50_recv_ratio=statistics.median(every) if every else 0.0,
        recv_ratio_per_rank_max=[max(per_rank) if per_rank else 0.0 for per_rank in gathered],
        max_recv_ratio_unpadded=max_over_ranks(max(raw_ratios, default=0.0)),
        counts_already_padded=agree_all(already_padded),
        pad_multiple=max((pad for *_, pad in calls), default=0),
        sizing_calls=len(every),
        sizing_steps=steps,
        sizing_over_budget=over,
    )


def padded_rows(counts: torch.Tensor, pad: int) -> float:
    """Permuted rows of one dispatch: each expert's count rounded up to ``pad`` (``pad <= 1``:
    the plain sum)."""
    if pad <= 1:
        return float(counts.sum().item())
    return float(((counts + pad - 1) // pad * pad).sum().item())


def sync_debug_report(model, inputs, *, kwargs_per_layer) -> List[str]:
    """Host synchronizations ``torch.cuda.set_sync_debug_mode("warn")`` reports during one
    eager pair forward + backward of ``model`` (with their Python stacks). Necessary, not
    sufficient: synchronizations inside C++ extensions are invisible to it (design 5.5)."""
    side = layer_side("sync-debug", model, kwargs_per_layer)
    found: List[str] = []
    previous_mode = torch.cuda.get_sync_debug_mode()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        previous_show = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):
            del file, line
            stack = "".join(traceback.format_stack(limit=14)[:-1])
            found.append(f"{category.__name__}: {message} ({filename}:{lineno})\n{stack}")

        warnings.showwarning = show
        torch.cuda.set_sync_debug_mode("warn")
        try:
            x_a = _leaf(inputs["x0"])
            y0 = side.pair(0, x_a)
            torch.autograd.backward(y0, inputs["g"])
        finally:
            torch.cuda.set_sync_debug_mode(previous_mode)
            warnings.showwarning = previous_show
    torch.cuda.synchronize()  # outside the window: an explicit synchronize would be reported
    model.zero_grad_buffer()
    return [text for text in found if "synchroniz" in text.lower()]


class _TagStack:
    """Component tags for saved-tensor attribution: module pre/post hooks and function
    wrappers push and pop; the innermost tag wins."""

    def __init__(self):
        self.stack = ["other"]

    def top(self) -> str:
        """The current component."""
        return self.stack[-1]

    @contextlib.contextmanager
    def push(self, tag: str):
        """Tag the enclosed region."""
        self.stack.append(tag)
        try:
            yield
        finally:
            self.stack.pop()


def _module_tags(model) -> List[Tuple[torch.nn.Module, str]]:
    # pylint: disable=import-outside-toplevel
    from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
    from megatron.core.transformer.experimental_attention_variant.csa import (
        CompressedSparseAttention,
        Compressor,
        CSAIndexer,
    )
    from megatron.core.transformer.hyper_connection import HyperConnectionModule
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.transformer_layer import TransformerLayer

    # Hooks fire only on modules entered through ``__call__``: the mHC wrapper runs its inner
    # TransformerLayer through methods (the no-local-BDA fast path), so the inner layer's
    # submodules carry the attention / MoE tags and the wrapper's own BDA counts as mHC.
    tags = []
    for module in model.modules():
        if isinstance(module, (HyperConnectionModule, HyperConnectionHybridLayer)):
            tags.append((module, "mhc"))
        elif isinstance(module, CompressedSparseAttention):
            tags.append((module, "attention_core"))
        elif isinstance(module, CSAIndexer):
            tags.append((module, "indexer"))
        elif isinstance(module, Compressor):
            tags.append((module, "attention_rest"))
        elif isinstance(module, MoELayer):
            tags.append((module, "moe_rest"))
            tags.append((module.router, "router"))
            tags.append((module.experts, "experts"))
            if getattr(module, "shared_experts", None) is not None:
                tags.append((module.shared_experts, "shared_expert"))
        elif isinstance(module, TransformerLayer):
            for name, tag in (
                ("input_layernorm", "attention_rest"),
                ("self_attention", "attention_rest"),
                ("pre_mlp_layernorm", "moe_rest"),
            ):
                submodule = getattr(module, name, None)
                if isinstance(submodule, torch.nn.Module) and not isinstance(submodule, IdentityOp):
                    tags.append((submodule, tag))
    return tags


MOE_METHOD_TAGS = {
    "route": "router",
    "preprocess": "dispatch",
    "dispatch": "dispatch",
    "routed_experts_compute": "experts",
    "combine": "combine",
    "postprocess": "combine",
    "shared_experts_compute": "shared_expert",
}


@contextlib.contextmanager
def component_tags(model, tags: _TagStack, core_events: Optional[List] = None):
    """Install the tagging hooks (test code only): module hooks per :func:`_module_tags`, MoE
    method wrappers per :data:`MOE_METHOD_TAGS`, and wrappers around the reference CSA core
    (``unfused_compressed_sparse_attn``: attention core; the fused ``csa_sparse_attn`` and
    ``FusedCSAIndexerSparseAttnFromTopkFunc`` are tagged attention core too, for J5) and the
    indexer-loss reference
    (``FusedDSAIndexerLoss.apply``, ``_compute_unfused_csa_non_compressed_lse``). With
    ``core_events`` the core wrapper records CUDA events around each forward core call."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer.experimental_attention_variant import csa

    handles = []
    for module, tag in _module_tags(model):

        def pre(_module, _args, _tag=tag):
            tags.stack.append(_tag)

        def post(_module, _args, _output):
            if len(tags.stack) > 1:
                tags.stack.pop()

        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))
    patched_methods = []
    for layer in moe_layers(model):
        for method, tag in MOE_METHOD_TAGS.items():
            original = getattr(layer, method, None)
            if original is None:
                continue

            def wrapper(*args, _original=original, _tag=tag, **kwargs):
                with tags.push(_tag):
                    return _original(*args, **kwargs)

            setattr(layer, method, wrapper)
            patched_methods.append((layer, method))
    original_core = csa.unfused_compressed_sparse_attn
    original_lse = csa._compute_unfused_csa_non_compressed_lse  # pylint: disable=protected-access
    original_loss = csa.FusedDSAIndexerLoss
    original_fused_core = csa.csa_sparse_attn
    original_fused_core_loss = csa.FusedCSAIndexerSparseAttnFromTopkFunc

    def fused_core(*args, **kwargs):
        with tags.push("attention_core"):
            return original_fused_core(*args, **kwargs)

    class _FusedCoreWithLoss:  # pylint: disable=too-few-public-methods
        @staticmethod
        def apply(*args, **kwargs):
            """Tagged fused core + indexer loss (J5's fused path; not separable)."""
            with tags.push("attention_core"):
                return original_fused_core_loss.apply(*args, **kwargs)

    def core(*args, **kwargs):
        with tags.push("attention_core"):
            if core_events is None or not torch.is_grad_enabled():
                return original_core(*args, **kwargs)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            output = original_core(*args, **kwargs)
            end.record()
            core_events.append((start, end))
            return output

    def lse(*args, **kwargs):
        with tags.push("indexer_loss_reference"):
            return original_lse(*args, **kwargs)

    class _IndexerLoss:  # pylint: disable=too-few-public-methods
        @staticmethod
        def apply(*args, **kwargs):
            """Tagged ``FusedDSAIndexerLoss.apply``."""
            with tags.push("indexer_loss_reference"):
                return original_loss.apply(*args, **kwargs)

    csa.unfused_compressed_sparse_attn = core
    csa._compute_unfused_csa_non_compressed_lse = lse  # pylint: disable=protected-access
    csa.FusedDSAIndexerLoss = _IndexerLoss
    csa.csa_sparse_attn = fused_core
    csa.FusedCSAIndexerSparseAttnFromTopkFunc = _FusedCoreWithLoss
    try:
        yield tags
    finally:
        csa.csa_sparse_attn = original_fused_core
        csa.FusedCSAIndexerSparseAttnFromTopkFunc = original_fused_core_loss
        csa.unfused_compressed_sparse_attn = original_core
        csa._compute_unfused_csa_non_compressed_lse = original_lse  # pylint: disable=W0212
        csa.FusedDSAIndexerLoss = original_loss
        for layer, method in patched_methods:
            delattr(layer, method)
        for handle in handles:
            handle.remove()


def _storages(tensor: torch.Tensor) -> List[Tuple[int, int]]:
    """``(data_ptr, nbytes)`` of the storages behind ``tensor`` (TE quantized tensors: their
    data / scale components)."""
    found = []
    try:
        storage = tensor.untyped_storage()
        if storage.nbytes() > 0:
            found.append((storage.data_ptr(), storage.nbytes()))
    except Exception:  # pylint: disable=broad-except
        pass
    if found:
        return found
    for attr in (
        "_data",
        "_transpose",
        "_rowwise_data",
        "_columnwise_data",
        "_scale_inv",
        "_rowwise_scale_inv",
        "_columnwise_scale_inv",
    ):
        part = getattr(tensor, attr, None)
        if torch.is_tensor(part):
            try:
                storage = part.untyped_storage()
                found.append((storage.data_ptr(), storage.nbytes()))
            except Exception:  # pylint: disable=broad-except
                continue
    return found


def saved_tensor_bytes(
    model, hidden: torch.Tensor, *, kwargs_per_layer, core_events: Optional[List] = None
) -> Dict[str, int]:
    """Bytes autograd saves during one forward of every layer of ``model`` on ``hidden``, by
    component: unique storages only, parameters and the input excluded."""
    module = getattr(model, "module", model)
    excluded = set()
    for param in module.parameters():
        for ptr, _ in _storages(param):
            excluded.add(ptr)
    for ptr, _ in _storages(hidden):
        excluded.add(ptr)
    seen: Dict[int, Tuple[int, str]] = {}
    tags = _TagStack()

    def pack(tensor):
        for ptr, nbytes in _storages(tensor):
            if ptr not in excluded and ptr not in seen:
                seen[ptr] = (nbytes, tags.top())
        return tensor

    with component_tags(model, tags, core_events):
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = _leaf(hidden)
            for layer, kwargs in zip(decoder_layers(model), kwargs_per_layer):
                output = call_layer(layer, output, kwargs)
    del output
    by_component: Dict[str, int] = defaultdict(int)
    for nbytes, tag in seen.values():
        by_component[tag] += nbytes
    by_component["total"] = sum(nbytes for nbytes, _ in seen.values())
    return dict(by_component)


def representative_saved_bytes(
    raw: Dict[str, int], fused_core: Dict[str, int], pairs: int
) -> Dict[str, int]:
    """Raw saved bytes with the reference attention core and indexer-loss reference replaced by
    the fused core of one attention sublayer x ``pairs``: a J5 measurement (its
    ``attention_core`` + ``indexer_loss_reference`` components) or the analytic record of
    :func:`fused_core_saved_bytes` (its ``total``). A J5 record without ``attention_core`` is
    rejected: its ``total`` also holds the projections, mHC and the indexer (double count)."""
    replaced = raw.get("attention_core", 0) + raw.get("indexer_loss_reference", 0)
    if "attention_core" in fused_core:
        per_layer = fused_core["attention_core"] + fused_core.get("indexer_loss_reference", 0)
    elif "kv_full" in fused_core:  # the analytic core
        per_layer = fused_core["total"]
    else:
        raise ValueError("fused core record has neither attention_core nor the analytic terms")
    fused = per_layer * pairs
    return dict(total=raw["total"] - replaced + fused, replaced=replaced, fused_core=fused)


def host_and_gpu_times(
    side: PairSide, inputs: Dict[str, torch.Tensor], *, repeat: int = 20, sleep_ms: float = 400.0
) -> Dict[str, Any]:
    """Host time per replay call (``perf_counter``, the GPU kept busy by ``torch.cuda._sleep``
    so no call waits on the device) per callable (forward) and per pair (forward, backward),
    and GPU time per pair from CUDA events; median and p90 over ``repeat``.

    The window is valid only if the GPU is still busy when the host loop ends (a hidden host
    synchronization would drain it and inflate the host time): ``host_window_gpu_busy_P<i>``
    records it; the window is retried with a 4x longer sleep (at most 3 times) when the GPU
    drained, and ``host_window_sleep_ms_P<i>`` records the sleep used."""
    record: Dict[str, Any] = {}
    x0, g = inputs["x0"], inputs["g"]
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    clock_hz = getattr(device_props, "clock_rate", 1_800_000) * 1000
    for index in range(len(side.parts)):
        window_ms = sleep_ms
        for _attempt in range(3):
            host_parts: List[List[float]] = [[] for _ in side.parts[index]]
            host_fwd, host_bwd = [], []
            torch.cuda.synchronize()
            torch.cuda._sleep(int(clock_hz * window_ms / 1000))  # pylint: disable=W0212
            for _ in range(repeat):
                x = _leaf(x0)
                hidden = x
                start = time.perf_counter()
                for part_index, part in enumerate(side.parts[index]):
                    t0 = time.perf_counter()
                    hidden = part(hidden)
                    host_parts[part_index].append(time.perf_counter() - t0)
                middle = time.perf_counter()
                torch.autograd.backward(hidden, g)
                host_bwd.append(time.perf_counter() - middle)
                host_fwd.append(middle - start)
                x.grad = None
            gpu_busy = not torch.cuda.current_stream().query()
            torch.cuda.synchronize()
            if gpu_busy:
                break
            window_ms *= 4
        record[f"host_window_gpu_busy_P{index}"] = gpu_busy
        record[f"host_window_sleep_ms_P{index}"] = window_ms
        gpu_fwd, gpu_bwd = [], []
        for _ in range(repeat):
            x = _leaf(x0)
            events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            events[0].record()
            y = side.pair(index, x)
            events[1].record()
            torch.autograd.backward(y, g)
            events[2].record()
            torch.cuda.synchronize()
            gpu_fwd.append(events[0].elapsed_time(events[1]))
            gpu_bwd.append(events[1].elapsed_time(events[2]))
            x.grad = None
        names = (side.names or [[f"part{j}" for j in range(len(side.parts[index]))]])[index]
        for name, samples in zip(names, host_parts):
            record[f"host_ms_fwd_callable_{name}"] = _stats(samples, 1e3)
        record[f"host_ms_fwd_P{index}"] = _stats(host_fwd, 1e3)
        record[f"host_ms_bwd_P{index}"] = _stats(host_bwd, 1e3)
        record[f"gpu_ms_fwd_P{index}"] = _stats(gpu_fwd, 1.0)
        record[f"gpu_ms_bwd_P{index}"] = _stats(gpu_bwd, 1.0)
    side.model.zero_grad_buffer()
    return record


def _stats(samples: Sequence[float], scale: float) -> Dict[str, float]:
    values = sorted(value * scale for value in samples)
    if not values:
        return dict(median=float("nan"), p90=float("nan"), n=0)
    p90 = values[min(len(values) - 1, int(math.ceil(0.9 * len(values))) - 1)]
    return dict(median=statistics.median(values), p90=p90, n=len(values))


def copy_out_cost(
    side: PairSide, inputs, *, repeat: int = 20, static_input: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """GPU time and bytes of the copy-out clone of a pair output, and the bytes TE copies into
    the static input per replay (TE copies when the input is not the static input itself)."""
    copied = 0
    x0 = inputs["x0"]
    if static_input is not None and static_input.data_ptr() != x0.data_ptr():
        copied = x0.numel() * x0.element_size()
    with torch.no_grad():
        y0 = side.pair(0, inputs["x0"])
        times = []
        for _ in range(repeat):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            y0.clone()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    return dict(
        copy_out_ms=_stats(times, 1.0),
        copy_out_bytes=y0.numel() * y0.element_size(),
        input_copy_bytes_per_replay=copied if static_input is not None else None,
    )


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


class Writer:
    """Rank-0 JSONL writer; every record also carries its kind and the run's identity.
    Failures are also printed and written by the rank they happen on (:meth:`local`), before
    any collective, so a rank that never reaches the next agreement still leaves its record."""

    def __init__(self, path: Optional[str], identity: Dict[str, Any]):
        self.path = path
        self.identity = identity
        self.records: List[Dict[str, Any]] = []

    def write(self, record_type: str, **fields) -> Dict[str, Any]:
        """Append one record (rank 0 writes the file). A field named like an identity key (the
        saved-bytes records' per-record ``seq``) overrides the run's value."""
        record = dict(record=record_type, **{**self.identity, **fields})
        self.records.append(record)
        if self.path and _rank() == 0:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=_json_default) + "\n")
        return record

    def local_path(self) -> Optional[str]:
        """This rank's own JSONL (``<out>.r<rank>.jsonl``)."""
        if not self.path:
            return None
        return f"{os.path.splitext(self.path)[0]}.r{_rank()}.jsonl"

    def local(self, record_type: str, **fields) -> Dict[str, Any]:
        """Print one record on this rank and append it to this rank's own JSONL."""
        record = dict(record=record_type, rank=_rank(), **{**self.identity, **fields})
        text = json.dumps(record, default=_json_default)
        _say(text)
        path = self.local_path()
        if path:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(text + "\n")
        return record

    def agree(self, step: str, error: Optional[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """Record this rank's failure of ``step`` locally (if any), then exchange every rank's
        record (:func:`agree_records`): all ranks get the same list and branch alike."""
        if error is not None:
            self.local("failure", step=step, error=error)
        return agree_records(step, error)


def errors_by_rank(errors: Sequence[Optional[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """The failed ranks' records of an agreement, keyed by rank."""
    return {str(rank): error for rank, error in enumerate(errors) if error is not None}


def first_error(errors: Sequence[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """The lowest failed rank's record (None when every rank succeeded)."""
    return next((error for error in errors if error is not None), None)


def stop_on_cuda_error(errors: Sequence[Optional[Dict[str, Any]]]) -> None:
    """Every rank stops (the same decision everywhere) when any rank holds a sticky CUDA
    error: the device is unusable and later collectives would hang."""
    failed = [rank for rank, error in enumerate(errors) if error and error.get("cuda_error")]
    if failed:
        raise SystemExit(f"sticky CUDA error on ranks {failed}; see the per-rank records")


_AGREE_STEPS = itertools.count()


def agree_records(
    step: str, error: Optional[Dict[str, Any]], *, timeout_s: Optional[float] = None
) -> List[Optional[Dict[str, Any]]]:
    """Every rank's failure record of ``step`` (None = succeeded), in rank order, the same list
    on every rank. Exchanged through the c10d store, not NCCL: a rank that failed while the
    others are still inside a collective of the step does not join a mismatched collective; it
    waits here at most ``timeout_s`` (:data:`AGREE_TIMEOUT_S`) and then stops the job
    (``SystemExit``: torchrun tears the other ranks down) instead of hanging. Ranks that fail
    at the same point (the usual, deterministic case) all arrive and continue together."""
    if not dist.is_initialized() or _world() == 1:
        return [error]
    store = dist.distributed_c10d._get_default_store()  # pylint: disable=protected-access
    prefix = f"pair_capture/agree/{next(_AGREE_STEPS)}"
    payload = dict(step=step, error=error)
    store.set(f"{prefix}/{_rank()}", json.dumps(payload, default=_json_default))
    keys = [f"{prefix}/{rank}" for rank in range(_world())]
    try:
        store.wait(keys, timedelta(seconds=timeout_s or AGREE_TIMEOUT_S))
    except Exception as wait_error:  # pylint: disable=broad-except
        raise SystemExit(
            f"rank {_rank()}: not every rank reached {step!r} within "
            f"{timeout_s or AGREE_TIMEOUT_S:.0f} s (a failure on some rank left the others in "
            "a collective); see the per-rank records"
        ) from wait_error
    entries = [json.loads(store.get(key)) for key in keys]
    steps = sorted({entry["step"] for entry in entries})
    if len(steps) > 1:
        raise SystemExit(f"ranks disagree on the step being agreed: {steps}")
    return [entry["error"] for entry in entries]


def _json_default(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _say(message: str) -> None:
    print(f"[spike r{_rank()}] {message}", flush=True)


def environment(config, args) -> Dict[str, Any]:
    """Versions, device, MXFP8 support, mHC backend, clamp and the relevant environment."""
    record: Dict[str, Any] = dict(torch=torch.__version__)
    try:
        import transformer_engine  # pylint: disable=import-outside-toplevel

        record["te"] = transformer_engine.__version__
    except Exception:  # pylint: disable=broad-except
        record["te"] = None
    try:
        import deep_ep  # pylint: disable=import-outside-toplevel

        record["hybridep"] = getattr(deep_ep, "__version__", "importable")
    except Exception:  # pylint: disable=broad-except
        record["hybridep"] = None
    if torch.cuda.is_available():
        record["nccl"] = ".".join(str(v) for v in torch.cuda.nccl.version())
        record["device"] = torch.cuda.get_device_name()
        record["cc"] = list(torch.cuda.get_device_capability())
    record.update(
        mxfp8_supported=mxfp8_available(),
        use_fused_mhc=config.use_fused_mhc,
        mhc_fused_backend=config.mhc_fused_backend,
        clamp=config.activation_func_clamp_value,
        clamp_supported=clamp_supported(),
        env={
            key: value
            for key, value in os.environ.items()
            if key.startswith("NVTE_")
            or key
            in (
                "CUDA_GRAPHS_USE_NODE_PRIORITY",
                "CUDA_DEVICE_MAX_CONNECTIONS",
                "PYTORCH_CUDA_ALLOC_CONF",
                "NCCL_GRAPH_REGISTER",
                "USE_MNNVL",
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
                "NVLINK_DOMAIN_SIZE",
                "NCCL_ALGO",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        argv=vars(args),
    )
    return record


def parse_args(argv=None) -> argparse.Namespace:
    """The spike CLI (stage-7 design 5.9)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--width", choices=sorted(WIDTHS), default="pro-proxy")
    parser.add_argument("--kind", choices=sorted(KINDS), default="CSA")
    parser.add_argument("--experts-per-rank", type=int, default=None)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--indexer-topk", type=int, default=None)
    parser.add_argument("--precision", choices=("bf16", "mxfp8"), default="bf16")
    parser.add_argument("--arms", default="helper,pair")
    parser.add_argument("--captures", default="block,nested,nested-noreuse")
    parser.add_argument("--orders", default="block,nested,nested-asc,sweep,interleaved,twice")
    parser.add_argument("--poison", choices=("graph", "default", "both", "none"), default="both")
    parser.add_argument("--factor", default="auto", help="auto (sized) or a float")
    parser.add_argument("--factors", default="auto", help="arm-A memory sweep, e.g. auto,1.5,2.0")
    parser.add_argument("--sizing-factor", type=float, default=4.0)
    parser.add_argument("--no-fused-mhc", action="store_true")
    parser.add_argument("--no-permute-fusion", action="store_true")
    parser.add_argument(
        "--ddp-stream",
        choices=("side", "current"),
        default="side",
        help="the stream DDP is built on (build_pair_models): 'current' (the legacy default "
        "stream) is the negative control of the backward-capture fix",
    )
    parser.add_argument("--memory-only", action="store_true")
    parser.add_argument(
        "--determinism-only",
        action="store_true",
        help="localize the eager run-to-run spread: per layer, repeated forward + backward "
        "calls compared bitwise, with the first differing submodule (determinism_probe)",
    )
    parser.add_argument("--repeats-probe", type=int, default=2)
    parser.add_argument(
        "--skew-rank0-ms",
        type=float,
        default=0.0,
        help="delay rank 0 (host sleep + device sleep) before every pair call (RANK0_SKEW_MS)",
    )
    parser.add_argument(
        "--probe-p6",
        action="store_true",
        help="run probe P6 (graph-pool poison between pair 0's forward and its backward); a "
        "backward that reads a freed pool block can fault the device, so give it its own "
        "process (e.g. --arms helper --captures block --orders block --poison graph)",
    )
    parser.add_argument(
        "--no-deterministic-algorithms",
        action="store_true",
        help="skip deterministic_algorithms() (torch.use_deterministic_algorithms and the "
        "--deterministic-mode environment): the stage-7 round-1 setting",
    )
    parser.add_argument("--saved-bytes-only", action="store_true")
    parser.add_argument("--seqs", default="4096,8192")
    parser.add_argument("--fused-core-json", default=None, help="J5 output (fused_attn_bytes.py)")
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument(
        "--nsys-ranks",
        default="0,5,10,15",
        help="the global ranks the launcher runs under nsys (comma or space separated; as "
        "nsys_sampled.sh's NSYS_RANKS): with --nsys their host times are kept out of "
        "median_max_unprofiled",
    )
    parser.add_argument(
        "--hang-dump-s",
        type=float,
        default=0,
        help="dump every thread's stack every N s for the whole process (0: off)",
    )
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--headroom", type=float, default=1.25)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.factor != "auto":
        args.factor = float(args.factor)
    return args


def _configs(args, factor, *, indexer_topk=None, graphs=True, pairs=2):
    return spike_config(
        args.width,
        ep=_world(),
        experts_per_rank=args.experts_per_rank,
        kind=args.kind,
        precision=args.precision,
        factor=factor,
        fused_mhc=not args.no_fused_mhc,
        permute_fusion=not args.no_permute_fusion,
        graphs=graphs,
        seq=args.seq,
        indexer_topk=indexer_topk if indexer_topk is not None else args.indexer_topk,
        pairs=pairs,
    )


def _init_distributed():
    # pylint: disable=import-outside-toplevel
    from megatron.core import parallel_state
    from megatron.core.config import set_experimental_flag
    from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
    from megatron.core.process_groups_config import ProcessGroupCollection

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    free_before = torch.cuda.mem_get_info()[0]
    dist.init_process_group("nccl")
    probe = torch.ones(1, device="cuda")
    dist.all_reduce(probe)
    torch.cuda.synchronize()
    nccl_bytes = free_before - torch.cuda.mem_get_info()[0]
    world = dist.get_world_size()
    parallel_state.initialize_model_parallel(1, 1, expert_model_parallel_size=world)
    set_experimental_flag(True)
    init_num_microbatches_calculator(dist.get_rank(), None, world, 1, world)
    return ProcessGroupCollection.use_mpu_process_groups(), nccl_bytes


@contextlib.contextmanager
def _hybridep_buffer_probe(sink: Dict[str, Any]):
    """Record the free-memory delta of the (process-global) HybridEP buffer allocation."""
    from megatron.core.transformer.moe import fused_a2a  # pylint: disable=import-outside-toplevel

    original = fused_a2a.init_hybrid_ep_buffer

    def init(*args, **kwargs):
        torch.cuda.synchronize()
        before = torch.cuda.mem_get_info()[0]
        output = original(*args, **kwargs)
        torch.cuda.synchronize()
        sink["hybridep_buffer_bytes"] = before - torch.cuda.mem_get_info()[0]
        sink["hybridep_buffer_inits"] = sink.get("hybridep_buffer_inits", 0) + 1
        return output

    fused_a2a.init_hybrid_ep_buffer = init
    try:
        yield sink
    finally:
        fused_a2a.init_hybrid_ep_buffer = original


def _capacity_fields(obj, prefix: str = "", depth: int = 2) -> Dict[str, Any]:
    """Scalar attributes (int / float / bool) and tensor sizes (``nbytes``) of ``obj`` and of
    its plain-object attributes down to ``depth``: the HybridEP buffer's capacity signal
    (``max_num_of_tokens_per_rank`` and the like, and the sizes of the buffers it holds)."""
    fields: Dict[str, Any] = {}
    for name, value in sorted(getattr(obj, "__dict__", {}).items()):
        key = f"{prefix}{name}"
        if isinstance(value, (bool, int, float)):
            fields[key] = value
        elif torch.is_tensor(value):
            fields[key + ".nbytes"] = value.untyped_storage().nbytes()
        elif isinstance(value, (list, tuple)) and value and all(torch.is_tensor(v) for v in value):
            fields[key + ".nbytes"] = [v.untyped_storage().nbytes() for v in value]
        elif (
            depth > 1
            and hasattr(value, "__dict__")
            and not isinstance(value, (torch.nn.Module, type))
            and not type(value).__module__.startswith(("torch", "builtins"))
            and not callable(value)
        ):
            fields.update(_capacity_fields(value, key + ".", depth - 1))
    return fields


def hybridep_buffer_state() -> Optional[Dict[str, Any]]:
    """The process-global HybridEP buffer: its identity (``reset_hybrid_ep_buffer`` /
    re-initialization) and its capacity fingerprint (runtime regrowth inside the same
    object). None before the first dispatch."""
    from megatron.core.transformer.moe import fused_a2a  # pylint: disable=import-outside-toplevel

    buffer = getattr(fused_a2a, "_hybrid_ep_buffer", None)
    if buffer is None:
        return None
    return dict(id=id(buffer), fields=_capacity_fields(buffer))


def hybridep_buffer_changes(
    before: Optional[Dict[str, Any]], after: Optional[Dict[str, Any]]
) -> List[str]:
    """What changed between two :func:`hybridep_buffer_state` readings (empty when nothing)."""
    if before is None or after is None:
        return [] if before is after else ["created" if before is None else "released"]
    changes = ["id"] if before["id"] != after["id"] else []
    keys = set(before["fields"]) | set(after["fields"])
    changes += sorted(
        f"fields.{key}" for key in keys if before["fields"].get(key) != after["fields"].get(key)
    )
    return changes


@contextlib.contextmanager
def hybridep_dispatch_watch(sink: Dict[str, Any]):
    """Device memory the HybridEP buffer allocates outside PyTorch's allocator during an eager
    ``dispatch_with_permute`` (its runtime regrowth): the free-memory drop minus PyTorch's
    reserved growth, max over dispatches. Calls inside a stream capture are not measured."""
    from megatron.core.transformer.moe import fused_a2a  # pylint: disable=import-outside-toplevel

    cls = getattr(fused_a2a, "HybridEPBuffer", None)
    original = getattr(cls, "dispatch_with_permute", None)
    sink.setdefault("hybridep_dispatch_regrowth_bytes", 0)
    sink.setdefault("hybridep_dispatches_measured", 0)
    if original is None:
        yield sink
        return

    def dispatch_with_permute(self, *args, **kwargs):
        if sink.get("paused") or torch.cuda.is_current_stream_capturing():
            return original(self, *args, **kwargs)
        free_before = torch.cuda.mem_get_info()[0]
        reserved_before = torch.cuda.memory_reserved()
        output = original(self, *args, **kwargs)
        grown = (free_before - torch.cuda.mem_get_info()[0]) - (
            torch.cuda.memory_reserved() - reserved_before
        )
        sink["hybridep_dispatch_regrowth_bytes"] = max(
            sink["hybridep_dispatch_regrowth_bytes"], int(grown)
        )
        sink["hybridep_dispatches_measured"] += 1
        return output

    cls.dispatch_with_permute = dispatch_with_permute
    try:
        yield sink
    finally:
        cls.dispatch_with_permute = original


@contextlib.contextmanager
def watch_paused(sink: Dict[str, Any]):
    """Suspend :func:`hybridep_dispatch_watch` (its ``mem_get_info`` per dispatch would inflate
    host times)."""
    previous = sink.get("paused", False)
    sink["paused"] = True
    try:
        yield
    finally:
        sink["paused"] = previous


def _prepare(models, pg_collection) -> None:
    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer.experimental_attention_variant.dsa_logging import (
        initialize_dsa_metric_tracker,
    )

    for model in models:
        if model is not None:
            initialize_dsa_metric_tracker([model], pg_collection)
    # Both device-data loss scales, as ``set_auxiliary_loss_scales`` sets them in training
    # (shape [1]); before any capture, so the graphs bake these tensors. The DSA scaler would
    # otherwise be created lazily in the first backward.
    for scaler in loss_scalers().values():
        current = scaler.main_loss_backward_scale
        if current is None or tuple(current.shape) != (1,) or current.device.type != "cuda":
            scaler.main_loss_backward_scale = None
        scaler.set_loss_scale(torch.ones(1, device="cuda"))


def loss_scalers() -> Dict[str, Any]:
    """The process-global device-data loss scalers the captured backward reads:
    ``moe_aux`` (router aux loss) and ``dsa_indexer`` (indexer KL loss)."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexerLossAutoScaler,
    )
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    return dict(moe_aux=MoEAuxLossAutoScaler, dsa_indexer=DSAIndexerLossAutoScaler)


# parameters whose main_grad carries each scaler's loss component (P5)
SCALER_PARAMS = {
    "moe_aux": lambda name: name.endswith("router.weight"),
    "dsa_indexer": lambda name: ".indexer." in name,
}


def aux_scale_verdict(
    graph: Dict[float, Dict[str, torch.Tensor]], eager: Dict[float, Dict[str, torch.Tensor]]
) -> Dict[str, Any]:
    """P5 on one rank: ``graph[s][name]`` / ``eager[s][name]`` are the main_grads at loss scale
    ``s`` in {0, 1, 2} (changed in place) of the graph replays and of the eager twin.

    ``aux_component_max`` = max |g(1) - g(0)| (0 when the replay ignores the scale: then the
    linearity error is 0 too, so linearity alone proves nothing); ``aux_component_rel_err`` =
    max |g(2) - g(0) - 2 (g(1) - g(0))| / ``aux_component_max``; ``aux_vs_eager_rel_err`` =
    max |aux(graph) - aux(eager)| / max |aux(eager)| (NaN when eager has no aux component);
    ``bitwise_vs_eager`` = every scale's main_grads equal eager's."""
    names = sorted(graph[1.0])
    aux_max = eager_max = linear = vs_eager = 0.0
    for name in names:
        aux1 = graph[1.0][name] - graph[0.0][name]
        aux2 = graph[2.0][name] - graph[0.0][name]
        eager1 = eager[1.0][name] - eager[0.0][name]
        aux_max = max(aux_max, aux1.abs().max().item())
        eager_max = max(eager_max, eager1.abs().max().item())
        linear = max(linear, (aux2 - 2 * aux1).abs().max().item())
        vs_eager = max(vs_eager, (aux1 - eager1).abs().max().item())
    bitwise = all(
        torch.equal(graph[scale][name], eager[scale][name]) for scale in graph for name in names
    )
    return dict(
        aux_component_max=aux_max,
        eager_aux_component_max=eager_max,
        aux_component_rel_err=(linear / aux_max) if aux_max > 0 else float("nan"),
        aux_vs_eager_rel_err=(vs_eager / eager_max) if eager_max > 0 else float("nan"),
        bitwise_vs_eager=bitwise,
        params=len(names),
    )


def aux_scale_ok(verdict: Dict[str, Any]) -> Optional[bool]:
    """P5 pass: the scale reaches the replay (a non-zero aux component), the replayed component
    equals eager's (bitwise, or within :data:`P5_REL_TOL`) and is linear in the scale. None
    when eager itself shows no aux component (the probe cannot decide)."""
    if not verdict["eager_aux_component_max"] > 0:
        return None
    if not verdict["aux_component_max"] > 0:
        return False
    close = verdict["bitwise_vs_eager"] or verdict["aux_vs_eager_rel_err"] <= P5_REL_TOL
    return bool(close and verdict["aux_component_rel_err"] <= P5_REL_TOL)


def run_saved_bytes(args, writer: Writer, pg_collection) -> None:
    """The saved-bytes process: one eager pair (``CE`` / ``HE`` / ``WE``) at each ``--seqs``."""
    factor = args.factor if args.factor != "auto" else 2.0
    config = _configs(args, factor, graphs=False, pairs=1)
    model, _ = build_pair_models(
        config,
        pg_collection=pg_collection,
        seed=args.seed,
        twin=False,
        vocab_size=WIDTHS[args.width]["vocab_size"],
        max_sequence_length=max(parse_list(args.seqs, int)),
    )
    _prepare([model], pg_collection)
    fused_core = None
    if args.fused_core_json:
        # fused_attn_bytes.py: {kind: {seq: {component: bytes, ...}}}
        with open(args.fused_core_json, encoding="utf-8") as handle:
            measured = json.load(handle).get(args.kind, {})
        fused_core = {int(seq): rec for seq, rec in measured.items() if "error" not in rec}
    per_seq: Dict[int, Dict[str, int]] = {}
    for seq in parse_list(args.seqs, int):
        hidden = make_inputs(config, seq, seed=args.seed + _rank())["x0"]
        kwargs = static_kwargs(model, seq)
        torch.cuda.synchronize()
        events: List = []
        raw = saved_tensor_bytes(model, hidden, kwargs_per_layer=kwargs, core_events=events)
        torch.cuda.synchronize()
        core_ms = sum(start.elapsed_time(end) for start, end in events)
        if fused_core is not None and "attention_core" in fused_core.get(seq, {}):
            fused, source = fused_core[seq], "J5 measured"
        elif fused_core is not None and seq in fused_core:
            fused, source = fused_core_saved_bytes(config, seq), "analytic (J5 has no core)"
        else:
            fused, source = fused_core_saved_bytes(config, seq), "analytic"
        repr_bytes = representative_saved_bytes(raw, fused, pairs=1)
        per_seq[seq] = raw
        model_mb = MODEL_SAVED_MB_PER_TOKEN.get(args.width)
        per_token_mb = repr_bytes["total"] / seq / 2**20
        writer.write(
            "saved_bytes",
            seq=seq,
            saved_bytes_by_component=raw,
            saved_bytes_per_pair_raw=raw["total"],
            saved_bytes_per_pair_repr=repr_bytes["total"],
            repr_detail=repr_bytes,
            fused_core_source=source,
            per_token_mb_raw=raw["total"] / seq / 2**20,
            per_token_mb_repr=per_token_mb,
            model_mb_per_token=model_mb,
            model_ratio=(per_token_mb / model_mb) if model_mb else None,
            model_flag=(abs(per_token_mb / model_mb - 1) > 0.2) if model_mb else None,
            gpu_ms_core_eager_fwd=core_ms,
            validity="raw: reference CSA core (not representative); repr: fused core replaced",
        )
        model.zero_grad_buffer()
        torch.cuda.empty_cache()
    seqs = sorted(per_seq)
    if len(seqs) >= 2:
        low, high = per_seq[seqs[0]], per_seq[seqs[-1]]
        writer.write(
            "saved_bytes_scaling",
            seqs=[seqs[0], seqs[-1]],
            ratio_by_component={
                key: (high.get(key, 0) / low[key]) if low.get(key) else None for key in high
            },
            model_ratio=seqs[-1] / seqs[0],
        )


def read_over_budget(model, where: str, seen: Optional[List[str]] = None) -> bool:
    """:func:`over_budget_any` (all-reduced, then reset) of everything since the last reset;
    ``where`` is appended to ``seen`` when the flag was set (design 5.4 step 4: never set
    during warm-up, capture and every replay at the sized factor)."""
    flag = over_budget_any(model)
    if flag and seen is not None:
        seen.append(where)
    return flag


def run_determinism(args, writer: Writer, pg_collection) -> None:
    """The determinism process: the graph model only, one eager warm-up step, then
    :func:`determinism_probe` (one record per layer, per rank: the rank-0 JSONL keeps rank 0's,
    every rank's own file keeps its own)."""
    factor = args.factor if args.factor != "auto" else 2.0
    config = _configs(args, factor)
    model, _ = build_pair_models(
        config,
        pg_collection=pg_collection,
        seed=args.seed,
        twin=False,
        vocab_size=WIDTHS[args.width]["vocab_size"],
        max_sequence_length=args.seq,
        ddp_stream=args.ddp_stream,
    )
    _prepare([model], pg_collection)
    inputs = make_inputs(config, args.seq, seed=args.seed + _rank())
    kwargs = static_kwargs(model, args.seq)
    execute_order("block", layer_side("warm-up", model, kwargs), inputs)
    reset_after_capture([model], config)
    writer.write("env", process_kind="determinism", factor=factor, **environment(config, args))
    for record in determinism_probe(model, config, kwargs, inputs, repeats=args.repeats_probe):
        writer.local("determinism_layer", **record)
        any_bad = not (record["bitwise_y"] and record["bitwise_dx"])
        any_bad = any_bad or record["nondeterministic_grads"] > 0
        writer.write(
            "determinism_layer_agreed",
            layer=record["layer"],
            deterministic_on_every_rank=agree_all(not any_bad),
            max_abs_diff_y=max_over_ranks(record["max_abs_diff_y"]),
            max_abs_diff_dx=max_over_ranks(record["max_abs_diff_dx"]),
            rank0=record,
        )
    model = None
    gc.collect()
    torch.cuda.empty_cache()


def run_memory_only(args, writer: Writer, pg_collection, nccl_bytes: int) -> None:
    """The memory-only process (D-S7-10): build the graph model only (no twin, no poisoning),
    reset the peaks, capture one arm, run one block-order sequence; peaks max over ranks, the
    pool from the snapshot, the build bytes, and a memory-snapshot pickle on rank 0."""
    if args.factor == "auto":
        raise SystemExit("--memory-only needs the sized --factor from the main run")
    config = _configs(args, args.factor)
    sink: Dict[str, Any] = {}
    seen: List[str] = []
    side = None
    with _hybridep_buffer_probe(sink):
        torch.cuda.synchronize()
        free_before = torch.cuda.mem_get_info()[0]
        model, _ = build_pair_models(
            config,
            pg_collection=pg_collection,
            seed=args.seed,
            twin=False,
            vocab_size=WIDTHS[args.width]["vocab_size"],
            max_sequence_length=args.seq,
            ddp_stream=args.ddp_stream,
        )
        _prepare([model], pg_collection)
        torch.cuda.synchronize()
        build_bytes = free_before - torch.cuda.mem_get_info()[0]
        inputs = make_inputs(config, args.seq, seed=args.seed + _rank())
        kwargs = static_kwargs(model, args.seq)
        # cuda_graph_warmup_steps = 1: one eager step (also allocates the HybridEP buffer)
        execute_order("block", layer_side("warm-up", model, kwargs), inputs)
        read_over_budget(model, "eager warm-up step", seen)
        reset_after_capture([model], config)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_allocated = torch.cuda.memory_allocated()
        snapshot_path = None
        if _rank() == 0 and args.out:
            torch.cuda.memory._record_memory_history(max_entries=500000)  # pylint: disable=W0212
        arm = parse_list(args.arms)[0]
        if arm == "helper":
            capture = capture_with_helper(model, config, seq=args.seq, pg_collection=pg_collection)
        else:
            pair_args, pair_kwargs, warmup, fp8_group = pair_sample_inputs(
                model, config, seq=args.seq, pg_collection=pg_collection
            )
            capture = capture_pairs(
                model,
                config,
                pair_args,
                pair_kwargs,
                precision=args.precision,
                order="block",
                num_warmup_iters=warmup,
                fp8_group=fp8_group,
            )
        errors = writer.agree(f"memory-only capture {arm}", capture.error)
        captured = not any(errors)
        if captured:
            capture.over_budget = read_over_budget(model, f"warm-up + capture {arm}", seen)
            if arm == "helper":
                side = layer_side("helper", model, capture.kwargs_per_layer or kwargs)
            else:
                side = graphed_side("pair", model, capture.graphed, pair_kwargs, capture.pool)
            reset_after_capture([model], config)
            execute_order("block", side, inputs)
            torch.cuda.synchronize()
            read_over_budget(model, f"block replay {arm}", seen)
        if _rank() == 0 and args.out:
            snapshot_path = os.path.splitext(args.out)[0] + f".mem_{arm}_{args.precision}_r0.pickle"
            torch.cuda.memory._dump_snapshot(snapshot_path)  # pylint: disable=protected-access
            torch.cuda.memory._record_memory_history(enabled=None)  # pylint: disable=W0212
    pool = capture.pool_record if captured else {}
    if captured and capture.pool is not None:
        pool = pool_bytes(capture.pool)
    writer.write(
        "memory",
        arm=arm,
        capture_ok=captured,
        error=first_error(errors),
        errors_by_rank=errors_by_rank(errors),
        build_bytes=max_over_ranks(build_bytes),
        hbm_peak_allocated_gb=max_over_ranks(torch.cuda.max_memory_allocated() / 2**30),
        hbm_peak_reserved_gb=max_over_ranks(torch.cuda.max_memory_reserved() / 2**30),
        peak_over_base_gb=max_over_ranks(
            (torch.cuda.max_memory_allocated() - base_allocated) / 2**30
        ),
        hybridep_buffer_bytes=sink.get("hybridep_buffer_bytes"),
        nccl_bytes=max_over_ranks(nccl_bytes),
        static_budget=num_permuted_tokens(model),
        over_budget_during_capture=capture.over_budget,
        over_budget_seen=seen,
        memory_snapshot=snapshot_path,
        validity="structure valid; totals contain the reference CSA core (D-S7-7)",
        **pool,
    )
    del side
    capture.delete()
    stop_on_cuda_error(errors)


@dataclasses.dataclass
class _Main:
    """State of the main spike process."""

    args: argparse.Namespace
    writer: Writer
    pg_collection: Any
    config: Any = None
    model: Any = None
    twin: Any = None
    inputs: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    kwargs: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    factor: float = 2.0
    sync_report: List[str] = dataclasses.field(default_factory=list)
    eager_spread: float = 0.0
    buffer_state: Optional[Dict[str, Any]] = None
    dispatch_watch: Dict[str, Any] = dataclasses.field(default_factory=dict)
    pool_estimate: int = 0
    # where the sticky over-budget flag was found set (warm-up, captures, replays, probes)
    over_budget_seen: List[str] = dataclasses.field(default_factory=list)


def _other_work(main: _Main) -> Callable[[], None]:
    side = layer_side("other", main.twin, main.kwargs)
    other = dict(x0=main.inputs["x_other"], g=main.inputs["g_other"])
    return lambda: execute_order("block", side, other)


def _eager_reference(main: _Main) -> Tuple[Dict[str, torch.Tensor], "Snapshot"]:
    """The eager twin's block order on the graph model's current weights."""
    copy_model_state(main.model, main.twin)
    assert_never_captured(main.twin)
    reset_after_capture([main.twin], main.config)
    result = execute_order("block", layer_side("eager", main.twin, main.kwargs), main.inputs)
    snapshot = Snapshot(result, main.twin)
    read_over_budget(main.twin, "eager twin block", main.over_budget_seen)
    return result, snapshot


def _check_eager(main: _Main, oracle: Snapshot, label: str) -> Dict[str, Any]:
    """Oracle (ii): the eager twin against a graph capture's block oracle."""
    result, _ = _eager_reference(main)
    record = oracle.compare(result, main.twin)
    record["within_eager_spread"] = record["max_abs_diff"] <= main.eager_spread or record["bitwise"]
    record["bitwise"] = agree_all(record["bitwise"])
    record["max_abs_diff"] = max_over_ranks(record["max_abs_diff"])
    main.writer.write("eager_vs_graph", capture=label, **_jsonable(record))
    reset_after_capture([main.twin], main.config)
    return record


def _jsonable(record: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in record.items() if not torch.is_tensor(value)}


def _run_orders(main: _Main, side: PairSide, label: str, orders: Sequence[str], poison) -> Snapshot:
    """Oracle, then every order (graph-pool poison between items when ``poison``). A replay
    that raises on some rank is agreed on (every rank skips the order together) and recorded
    with every failed rank's record."""
    model, config, seen = main.model, main.config, main.over_budget_seen
    reset_after_capture([model], config)
    native = NATIVE_ORDER[label]
    oracle_result = execute_order(native, side, main.inputs)
    oracle = Snapshot(oracle_result, model, forwards=ORDER_FORWARDS[native])
    over = read_over_budget(model, f"{label} oracle {native}", seen)
    main.writer.write("oracle", capture=label, oracle_order=native, over_budget_any=over)
    for order in orders:
        for poisoned in (False, True) if poison is not None else (False,):
            step = f"{label} {order} poisoned={poisoned}"
            gated = is_gated(label, order, poisoned)
            reset_after_capture([model], config)
            error, result = None, {}
            try:
                result = execute_order(
                    order,
                    side,
                    main.inputs,
                    poison=poison if poisoned else None,
                    other_work=_other_work(main) if order == "interleaved" else None,
                    upstream_dx1=oracle_result["dx1"] if order == "twice" else None,
                )
            except Exception as exc:  # pylint: disable=broad-except
                error = failure_record(exc, f"replay {order}")
            errors = main.writer.agree(step, error)
            if any(errors):
                main.writer.write(
                    "order",
                    capture=label,
                    order=order,
                    poisoned=poisoned,
                    gated=gated,
                    error=first_error(errors),
                    errors_by_rank=errors_by_rank(errors),
                )
                stop_on_cuda_error(errors)
                continue
            keys = {"nested-asc": ("y0", "y1", "dx1"), "twice": ("y0", "dx0")}.get(order)
            # ``compare`` folds the no-grad sweep / rebuild outputs into ``bitwise``; main_grads
            # are compared for every order that is gated without poisoning
            compared = oracle.compare(
                result,
                model,
                keys=keys,
                grads=is_gated(label, order),
                forwards=ORDER_FORWARDS[order],
            )
            if order == "interleaved":
                # the eager twin's other work writes into the shared DSA tracker (DV-sp-5)
                compared.pop("dsa_values_match", None)
            _say(
                f"{step}: local bitwise={compared['bitwise']} "
                f"max_abs_diff={compared['max_abs_diff']} over_budget(local, before reset)="
                f"{_local_over_budget(model)} allocated="
                f"{torch.cuda.memory_allocated() / 2**30:.2f} GB"
            )
            main.writer.write(
                "order",
                capture=label,
                order=order,
                poisoned=poisoned,
                gated=gated,
                bitwise=agree_all(compared["bitwise"]),
                max_abs_diff=max_over_ranks(compared["max_abs_diff"]),
                over_budget_any=read_over_budget(model, step, seen),
                detail_rank0=_jsonable(compared),
            )
    reset_after_capture([model], config)
    return oracle


def _probe_p5(main: _Main, side: PairSide, label: str, native: str) -> None:
    """P5 (DV-sp-10, review B1 / N4): each device-data loss scale (MoE aux, DSA indexer) set to
    0, 1, 2 in place; the graph replays' and the eager twin's main_grads of the parameters the
    scaled loss reaches. The verdict needs a non-zero replayed aux component (a replay that
    ignores the scale gives g0 = g1 = g2, which a linearity check alone passes) that equals
    eager's and is linear."""
    model, config, inputs, writer = main.model, main.config, main.inputs, main.writer
    eager_side = layer_side("eager", main.twin, main.kwargs)
    copy_model_state(model, main.twin)
    assert_never_captured(main.twin)
    for scaler_name, scaler in loss_scalers().items():
        names = [name for name, _ in _named_main_grads(model) if SCALER_PARAMS[scaler_name](name)]
        if not agree_all(bool(names)):
            writer.write("probe", capture=label, probe=f"P5 {scaler_name}", skipped="no params")
            continue
        graph: Dict[float, Dict[str, torch.Tensor]] = {}
        eager: Dict[float, Dict[str, torch.Tensor]] = {}
        for scale in (0.0, 1.0, 2.0):
            scaler.set_loss_scale(torch.full((1,), scale, device="cuda"))
            reset_after_capture([model, main.twin], config)
            execute_order(native, side, inputs)
            grads = dict(_named_main_grads(model))
            graph[scale] = {name: grads[name].detach().clone() for name in names}
            read_over_budget(
                model, f"probe P5 {scaler_name} {scale} {label}", main.over_budget_seen
            )
            execute_order(native, eager_side, inputs)
            grads = dict(_named_main_grads(main.twin))
            eager[scale] = {name: grads[name].detach().clone() for name in names}
            read_over_budget(
                main.twin, f"probe P5 eager {scaler_name} {scale}", main.over_budget_seen
            )
        scaler.set_loss_scale(torch.ones(1, device="cuda"))
        local = aux_scale_verdict(graph, eager)
        agreed = dict(
            local,
            aux_component_max=max_over_ranks(local["aux_component_max"]),
            eager_aux_component_max=max_over_ranks(local["eager_aux_component_max"]),
            aux_component_rel_err=max_over_ranks(local["aux_component_rel_err"]),
            aux_vs_eager_rel_err=max_over_ranks(local["aux_vs_eager_rel_err"]),
            bitwise_vs_eager=agree_all(local["bitwise_vs_eager"]),
        )
        writer.write(
            "probe",
            capture=label,
            probe=f"P5 {scaler_name} loss scale changed in place",
            scale_reaches_replay=agreed["aux_component_max"] > 0,
            p5_ok=aux_scale_ok(agreed),
            tolerance=P5_REL_TOL,
            **agreed,
        )
    reset_after_capture([model, main.twin], config)


def _probes(main: _Main, side: PairSide, capture: CaptureResult, oracle: Snapshot, poison) -> None:
    """Report-only probes on a capture: P6 (poison between a forward and its backward), P3
    (static input overwrite), P4 (output aliasing), P5 (device-data loss scales), the copy-out
    and no-copy variants, and the default-allocator poisoning. Each record carries the
    over-budget flag of its own replays (read before the next reset)."""
    model, config, inputs, writer = main.model, main.config, main.inputs, main.writer
    label = capture.capture
    native = NATIVE_ORDER[label]
    forwards = ORDER_FORWARDS[native]

    def over(probe: str) -> bool:
        return read_over_budget(model, f"probe {probe} {label}", main.over_budget_seen)

    # P6 only in its own process (--probe-p6): at pro-proxy width the poison lands in a block
    # the backward reads as indices and the device asserts (gpu-fix-r1, J7c / J7d), which ends
    # the process and every record after it.
    if poison is not None and main.args.probe_p6:
        reset_after_capture([model], config)
        result = execute_order(native, side, inputs, poison=poison, poison_where="fwd-bwd")
        writer.write(
            "probe",
            capture=label,
            probe="P6 poison between fwd P0 and bwd P0",
            over_budget_any=over("P6"),
            **_agreed(oracle.compare(result, model, forwards=forwards)),
        )
    reset_after_capture([model], config)
    result = execute_order(native, side, inputs, copy_out=False)
    writer.write(
        "probe",
        capture=label,
        probe=f"{native} without copy-out",
        over_budget_any=over("no-copy"),
        **_agreed(oracle.compare(result, model, forwards=forwards)),
    )
    if capture.sample_args and native == "block":
        reset_after_capture([model], config)
        static_input = capture.sample_args[0][0]
        x_a = _leaf(inputs["x0"])
        y0 = side.pair(0, x_a)
        x1 = _leaf(y0.detach().clone())
        y1 = side.pair(1, x1)
        torch.autograd.backward(y1, inputs["g"])
        dx1 = x1.grad.clone()
        with torch.no_grad():
            static_input.fill_(float("nan"))
        torch.autograd.backward(y0, dx1)
        compared = oracle.compare(dict(dx0=x_a.grad.clone()), model, keys=("dx0",))
        writer.write(
            "probe",
            capture=label,
            probe="P3 static input overwritten before bwd P0",
            backward_reads_input_slot=not agree_all(compared["bitwise"]),
            over_budget_any=over("P3"),
            **_agreed(compared),
        )
    reset_after_capture([model], config)
    with torch.no_grad():
        first = side.pair(0, inputs["x0"])
        kept = first.clone()
        side.pair(0, inputs["xb"])
        changed_by_same = not torch.equal(first, kept)
        first = side.pair(0, inputs["x0"])
        kept = first.clone()
        side.pair(1, kept)
        changed_by_other = not torch.equal(first, kept)
    writer.write(
        "probe",
        capture=label,
        probe="P4 output aliasing",
        p4a_output_changed_by_second_replay=agree_all(changed_by_same),
        p4b_p0_output_changed_by_p1_replay_any_rank=not agree_all(not changed_by_other),
        over_budget_any=over("P4"),
    )
    reset_after_capture([model], config)
    _probe_p5(main, side, label, native)
    static_input = capture.sample_args[0][0] if capture.sample_args else None
    writer.write(
        "probe",
        capture=label,
        probe="copy-out",
        **copy_out_cost(side, inputs, static_input=static_input),
        over_budget_any=over("copy-out"),
    )
    if main.args.poison in ("default", "both"):
        poisoned = poison_default_allocator()
        reset_after_capture([model], config)
        result = execute_order(native, side, inputs)
        torch.cuda.empty_cache()
        writer.write(
            "probe",
            capture=label,
            probe="default-allocator poisoning",
            poison_bytes=poisoned,
            over_budget_any=over("default-allocator poisoning"),
            **_agreed(oracle.compare(result, model, forwards=forwards)),
        )
    reset_after_capture([model], config)


def _agreed(record: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        bitwise=agree_all(record["bitwise"]),
        max_abs_diff=max_over_ranks(record["max_abs_diff"]),
        mismatches_rank0=record["mismatches"],
    )


def _finish_capture(
    main: _Main, capture: CaptureResult, side: PairSide, orders, timing: bool
) -> None:
    """Common work after a successful capture: budget and HybridEP checks, slots, orders with
    and without graph-pool poisoning, the eager oracle, probes, timing. The capture's own
    over-budget flag (TE warm-up + capture) was read by :func:`_capture_agreed`."""
    model, config, writer = main.model, main.config, main.writer
    seen = main.over_budget_seen
    label = capture.capture
    reset_after_capture([model], config)
    poison = None
    if main.args.poison in ("graph", "both") and capture.pool is not None:
        poison = poison_graph_pool(capture.pool)
        writer.write("poison_graph", capture=label, **poison.record())
    reset_after_capture([model], config)
    slots = static_io_record(capture, side, main.inputs)
    over_static = read_over_budget(model, f"static-io replay {label}", seen)
    budgets = num_permuted_tokens(model)
    after_capture = hybridep_buffer_state()
    in_capture = hybridep_buffer_changes(main.buffer_state, after_capture)
    writer.write(
        "capture",
        arm=capture.arm,
        capture=label,
        capture_ok=True,
        capture_s=max_over_ranks(capture.capture_s),
        static_budget=budgets,
        budget_bytes=budget_bytes(config, budgets),
        over_budget_during_capture=capture.over_budget,
        over_budget_static_io=over_static,
        hybridep_buffer_unchanged_by_capture=agree_all(not in_capture),
        hybridep_buffer_changes_rank0=in_capture,
        **capture.pool_record,
        **slots,
    )
    oracle = _run_orders(
        main, side, label, [o for o in orders if o in parse_list(main.args.orders)], poison
    )
    _check_eager(main, oracle, label)
    _probes(main, side, capture, oracle, poison)
    if timing:
        reset_after_capture([model], config)
        with watch_paused(main.dispatch_watch):
            times = host_and_gpu_times(side, main.inputs, repeat=main.args.repeat)
        writer.write(
            "timing",
            capture=label,
            side="graph",
            over_budget_any=read_over_budget(model, f"timing {label}", seen),
            **_max_timing(times, profiled=_profiled(main.args)),
        )
        if main.args.nsys:
            _nsys_window(main, side)
    if poison is not None:
        poison.reset()
    after_replays = hybridep_buffer_changes(after_capture, hybridep_buffer_state())
    writer.write(
        "budget",
        capture=label,
        over_budget_any=read_over_budget(model, f"after the replays of {label}", seen),
        over_budget_seen=list(seen),
        hybridep_buffer_unchanged_by_replays=agree_all(not after_replays),
        hybridep_buffer_changes_rank0=after_replays,
        hybridep_dispatch_regrowth_bytes=max_over_ranks(
            main.dispatch_watch.get("hybridep_dispatch_regrowth_bytes", 0)
        ),
    )
    reset_after_capture([model], config)


def _finish_capture_safely(main: _Main, capture: CaptureResult, side: PairSide, orders, timing):
    """:func:`_finish_capture`, recording (not raising) a failure after a successful capture so
    the next capture still runs. Every rank's outcome is agreed through the store (review B4):
    each failed rank's record reaches rank 0; a sticky CUDA error stops every rank."""
    error = None
    try:
        _finish_capture(main, capture, side, orders, timing)
    except Exception as exc:  # pylint: disable=broad-except
        error = failure_record(exc, f"after capture {capture.capture}")
    errors = main.writer.agree(f"after capture {capture.capture}", error)
    if any(errors):
        main.writer.write(
            "capture_followup_error",
            capture=capture.capture,
            error=first_error(errors),
            errors_by_rank=errors_by_rank(errors),
        )
        stop_on_cuda_error(errors)


def _max_timing(record: Dict[str, Any], profiled: bool = False) -> Dict[str, Any]:
    """Statistics get their median's max over ranks, every rank's median (``median_by_rank``)
    and the max over the ranks NOT under nsys (``median_max_unprofiled``; None when every rank
    is profiled): ``profiled`` = this rank runs under nsys, whose injection stays loaded outside
    the capture range and can bias host times (review N2). Flags are agreed (all ranks True);
    other numbers are maxed over ranks. Every rank holds the same keys in the same order."""
    medians = {k: v["median"] for k, v in record.items() if isinstance(v, dict) and "median" in v}
    per_rank = gather_objects((bool(profiled), medians))
    result: Dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict) and "median" in value:
            unprofiled = [ranked[key] for flag, ranked in per_rank if not flag]
            result[key] = dict(
                value,
                median_max_over_ranks=max_over_ranks(value["median"]),
                median_by_rank=[ranked[key] for _flag, ranked in per_rank],
                profiled_ranks=[rank for rank, (flag, _m) in enumerate(per_rank) if flag],
                median_max_unprofiled=nan_max(*unprofiled) if unprofiled else None,
            )
        elif isinstance(value, bool):
            result[key] = agree_all(value)
        else:
            result[key] = max_over_ranks(float(value))
    return result


def _profiled(args: argparse.Namespace) -> bool:
    """True when this rank runs under nsys (``--nsys`` and its rank in ``--nsys-ranks``)."""
    ranks = {int(r) for r in args.nsys_ranks.replace(",", " ").split()}
    return bool(args.nsys) and _rank() in ranks


def _nsys_window(main: _Main, side: PairSide, iterations: int = 5, model=None) -> None:
    """Five block-order iterations between cudaProfilerStart / Stop (nsys
    ``--capture-range=cudaProfilerApi``), each in an NVTX range. ``model`` is the model the
    side runs (default: the graph model; the eager window passes the twin)."""
    model = main.model if model is None else model
    _say(f"[nsys] profiler range: {side.name} block x{iterations}")
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for iteration in range(iterations):
        torch.cuda.nvtx.range_push(f"spike {side.name} block {iteration}")
        execute_order("block", side, main.inputs)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    read_over_budget(model, f"nsys window {side.name}", main.over_budget_seen)
    reset_after_capture([model], main.config)


def _headroom_ok(main: _Main, label: str) -> bool:
    """Design 5.2: before every capture, free memory >= ``--headroom`` x the pool estimate on
    every rank (recorded)."""
    gc.collect()
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info()[0]
    ok = agree_all(free >= main.args.headroom * main.pool_estimate)
    main.writer.write(
        "headroom",
        capture=label,
        free_gb_min_over_ranks=-max_over_ranks(-free / 2**30),
        pool_estimate_gb=max_over_ranks(main.pool_estimate / 2**30),
        headroom=main.args.headroom,
        headroom_ok=ok,
    )
    return ok


def _capture_agreed(main: _Main, capture: CaptureResult, step: str, **fields) -> bool:
    """Agree on a capture's success on every rank (review B4). On failure (any rank): one
    record with every failed rank's error, this rank's graphs released, a sticky CUDA error
    stops every rank. On success: the over-budget flag of TE's warm-up + the capture is read
    before anything resets it (review B2)."""
    error = None if capture.ok else (capture.error or dict(type="unknown", step=step))
    errors = main.writer.agree(step, error)
    if any(errors):
        main.writer.write(
            "capture",
            arm=capture.arm,
            capture=capture.capture,
            capture_ok=False,
            error=first_error(errors),
            errors_by_rank=errors_by_rank(errors),
            **fields,
        )
        capture.delete()
        stop_on_cuda_error(errors)
        return False
    capture.over_budget = read_over_budget(
        main.model, f"warm-up + capture {capture.capture}", main.over_budget_seen
    )
    return True


def run_main(args, writer: Writer, pg_collection, nccl_bytes: int, indexer_topk=None) -> bool:
    """The main spike process. Returns False when the headroom check asks for a retry with
    ``--indexer-topk 256``."""
    sizing = args.factor == "auto"
    config = _configs(
        args, args.sizing_factor if sizing else args.factor, indexer_topk=indexer_topk
    )
    # the eager twin: the same (graph) config, never captured (review N1)
    twin_config = _configs(
        args, args.sizing_factor if sizing else args.factor, indexer_topk=indexer_topk
    )
    main = _Main(args=args, writer=writer, pg_collection=pg_collection, config=config)
    sink: Dict[str, Any] = {}
    with _hybridep_buffer_probe(sink):
        main.model, main.twin = build_pair_models(
            config,
            pg_collection=pg_collection,
            seed=args.seed,
            twin_config=twin_config,
            vocab_size=WIDTHS[args.width]["vocab_size"],
            max_sequence_length=args.seq,
            ddp_stream=args.ddp_stream,
        )
        _prepare([main.model, main.twin], pg_collection)
        main.inputs = make_inputs(config, args.seq, seed=args.seed + _rank())
        main.kwargs = static_kwargs(main.model, args.seq)
        writer.write(
            "env",
            nccl_bytes=nccl_bytes,
            indexer_topk=config.dsa_indexer_topk,
            pattern=pattern_of(config),
            twin="graph config, never captured (eager)",
            **environment(config, args),
        )
        if sizing:
            budget = size_static_budget(main.twin, main.inputs, kwargs_per_layer=main.kwargs)
            main.factor = static_budget_factor(budget["max_recv_ratio"])
        else:
            budget, main.factor = {}, args.factor
        apply_factor([main.model, main.twin], main.factor)
        writer.write(
            "sizing", factor=main.factor, sized=sizing, sizing_factor=args.sizing_factor, **budget
        )
        # T0: the eager twin twice on the same inputs.
        first, first_snapshot = _eager_reference(main)
        second, _ = _eager_reference(main)
        t0 = first_snapshot.compare(second, main.twin)
        main.eager_spread = max_over_ranks(t0["max_abs_diff"])
        writer.write(
            "t0_eager_self_determinism",
            bitwise=agree_all(t0["bitwise"]),
            max_abs_diff=main.eager_spread,
        )
        del first, second
        reset_after_capture([main.twin], config)
        # cuda_graph_warmup_steps = 1 on the graph model, measured as the pool estimate; the
        # host synchronizations seen in one eager pair forward + backward.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        execute_order("block", layer_side("warm-up", main.model, main.kwargs), main.inputs)
        torch.cuda.synchronize()
        main.pool_estimate = torch.cuda.max_memory_allocated() - base
        over_warmup = read_over_budget(main.model, "eager warm-up step", main.over_budget_seen)
        reset_after_capture([main.model], config)
        main.sync_report = sync_debug_report(main.model, main.inputs, kwargs_per_layer=main.kwargs)
        read_over_budget(main.model, "sync-debug pass", main.over_budget_seen)
        reset_after_capture([main.model], config)
        # which retained AccumulateGrad nodes sit on the legacy stream (the backward-capture
        # blocker of round 1; 0 with --ddp-stream side)
        census = census_on_side_stream(
            main.model, layer_side("census", main.model, main.kwargs), main.inputs
        )
        read_over_budget(main.model, "accumulate-grad census", main.over_budget_seen)
        reset_after_capture([main.model], config)
        writer.write(
            "accumulate_grad_streams",
            ddp_stream=args.ddp_stream,
            on_legacy_stream_max=int(max_over_ranks(census["on_legacy_stream"])),
            **census,
        )
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info()[0]
    headroom_ok = agree_all(free >= args.headroom * main.pool_estimate)
    writer.write(
        "diagnostics",
        sync_debug=main.sync_report,
        sync_debug_count=len(main.sync_report),
        hybridep_buffer_bytes=sink.get("hybridep_buffer_bytes"),
        hybridep_buffer_inits=sink.get("hybridep_buffer_inits"),
        pool_estimate_gb=max_over_ranks(main.pool_estimate / 2**30),
        free_gb=free / 2**30,
        headroom_ok=headroom_ok,
        over_budget_warmup=over_warmup,
        twin_param_bytes=sum(
            param.numel() * (param.element_size() + 4) for param in main.twin.module.parameters()
        ),
        twin_bytes_note="twin parameters + fp32 main_grad; its transient peak equals pool_estimate",
    )
    if not headroom_ok and indexer_topk is None and args.kind == "CSA":
        writer.write("retry", reason="headroom", next_indexer_topk=256)
        _release(main)
        return False
    arms = parse_list(args.arms)
    with hybridep_dispatch_watch(main.dispatch_watch):
        if "helper" in arms:
            _arm_helper(main)
        if "pair" in arms:
            _arm_pairs(main)
    writer.write(
        "summary",
        factor=main.factor,
        records=len(writer.records),
        over_budget_seen=main.over_budget_seen,
        over_budget_never_set=not main.over_budget_seen,
        **{key: value for key, value in main.dispatch_watch.items() if key != "paused"},
    )
    _release(main)
    return True


def _release(main: _Main) -> None:
    """Drop the run's models and inputs (DDP hook closures form reference cycles, so collect)
    and return their memory, so a retry does not hold two copies."""
    main.model = main.twin = None
    main.inputs, main.kwargs = {}, []
    gc.collect()
    torch.cuda.empty_cache()


def _arm_helper(main: _Main) -> None:
    args, model, config = main.args, main.model, main.config
    factors = [main.factor if f == "auto" else float(f) for f in parse_list(args.factors)]
    for index, factor in enumerate(factors):
        apply_factor([model, main.twin], factor)
        if not _headroom_ok(main, f"helper f={factor}"):
            main.writer.write(
                "capture", arm="helper", capture="helper", factor=factor, skipped="headroom"
            )
            continue
        main.buffer_state = hybridep_buffer_state()
        capture = capture_with_helper(
            model,
            config,
            seq=args.seq,
            pg_collection=main.pg_collection,
            sync_report=main.sync_report,
        )
        if not _capture_agreed(main, capture, f"capture helper f={factor}", factor=factor):
            continue
        if index == 0:
            side = layer_side("helper", model, capture.kwargs_per_layer)
            _finish_capture_safely(main, capture, side, ORDERS_PER_CAPTURE["helper"], timing=True)
            main.writer.write(
                "mxfp8_weight_cache",
                fp8_weight_caching=bool(config.fp8),
                disable_parameter_transpose_cache=config.disable_parameter_transpose_cache,
                observed="unverified (read the nsys of J3b)",
            )
            del side
        else:
            main.writer.write(
                "memory_sweep",
                factor=factor,
                static_budget=num_permuted_tokens(model),
                over_budget_during_capture=capture.over_budget,
                **capture.pool_record,
            )
        capture.delete()
    apply_factor([model, main.twin], main.factor)
    # eager GPU time per pair on the twin (graphed times are in the capture records)
    copy_model_state(model, main.twin)
    assert_never_captured(main.twin)
    reset_after_capture([main.twin], config)
    eager = layer_side("eager", main.twin, main.kwargs)
    with watch_paused(main.dispatch_watch):
        times = host_and_gpu_times(eager, main.inputs, repeat=main.args.repeat)
    main.writer.write(
        "timing", capture="eager", side="eager", **_max_timing(times, profiled=_profiled(main.args))
    )
    if main.args.nsys:  # the eager timing's timeline (evidence for the eager GPU times)
        with watch_paused(main.dispatch_watch):
            _nsys_window(main, eager, model=main.twin)
    events: List = []
    with component_tags(main.twin, _TagStack(), events):
        torch.cuda.synchronize()
        execute_order("block", eager, main.inputs)
        torch.cuda.synchronize()
    main.writer.write(
        "timing_core",
        gpu_ms_core_eager_fwd=max_over_ranks(sum(s.elapsed_time(e) for s, e in events)),
        core_calls=len(events),
    )
    read_over_budget(main.twin, "eager timing", main.over_budget_seen)
    reset_after_capture([main.twin], config)


def _one_pair_capture(main: _Main, name: str) -> None:
    """One arm-B capture, its orders, probes and (block) timing; every reference to its graphs
    dies before :meth:`CaptureResult.delete` returns the pool (one pool alive at a time)."""
    args, model, config = main.args, main.model, main.config
    if not _headroom_ok(main, f"pair {name}"):
        main.writer.write("capture", arm="pair", capture=name, skipped="headroom")
        return
    pair_args, pair_kwargs, warmup, fp8_group = pair_sample_inputs(
        model, config, seq=args.seq, pg_collection=main.pg_collection
    )
    main.buffer_state = hybridep_buffer_state()
    capture = capture_pairs(
        model,
        config,
        pair_args,
        pair_kwargs,
        precision=args.precision,
        order=name,
        reuse_buffers=name != "nested-noreuse",
        num_warmup_iters=warmup,
        fp8_group=fp8_group,
        sync_report=main.sync_report,
    )
    del pair_args
    if not _capture_agreed(main, capture, f"capture pair {name}"):
        return
    reset_after_capture([model, main.twin], config)
    side = graphed_side(f"pair-{name}", model, capture.graphed, pair_kwargs, capture.pool)
    _finish_capture_safely(main, capture, side, ORDERS_PER_CAPTURE[name], timing=name == "block")
    del side, pair_kwargs
    capture.delete()


def _arm_pairs(main: _Main) -> None:
    for name in parse_list(main.args.captures):
        _one_pair_capture(main, name)


def main(argv=None) -> None:
    """CLI entry point (one JSONL record per arm / capture / order, plus a summary)."""
    os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    args = parse_args(argv)
    RANK0_SKEW_MS[0] = args.skew_rank0_ms
    if args.hang_dump_s > 0:  # a hang leaves Python stacks before the launcher's timeout
        faulthandler.dump_traceback_later(args.hang_dump_s, repeat=True)
    # before NCCL / cuBLAS first use; held (a dropped generator context manager would be
    # closed by the garbage collector, restoring the nondeterministic state at once)
    determinism = contextlib.ExitStack()
    if not args.no_deterministic_algorithms:
        determinism.enter_context(deterministic_algorithms())
    pg_collection, nccl_bytes = _init_distributed()
    identity = dict(
        spike="j",
        width=args.width,
        kind=args.kind,
        precision=args.precision,
        ep=_world(),
        seq=args.seq,
        process=(
            "saved-bytes"
            if args.saved_bytes_only
            else (
                "determinism"
                if args.determinism_only
                else "memory-only" if args.memory_only else "main"
            )
        ),
    )
    writer = Writer(args.out, identity)
    finished = False
    try:
        if args.saved_bytes_only:
            run_saved_bytes(args, writer, pg_collection)
        elif args.determinism_only:
            run_determinism(args, writer, pg_collection)
        elif args.memory_only:
            run_memory_only(args, writer, pg_collection, nccl_bytes)
        elif not run_main(args, writer, pg_collection, nccl_bytes):
            gc.collect()
            torch.cuda.empty_cache()
            run_main(args, writer, pg_collection, nccl_bytes, indexer_topk=256)
        _say(
            f"done; free {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB, "
            f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
        )
        finished = True
    except (Exception, SystemExit) as error:  # pylint: disable=broad-except
        # every rank leaves its own record (the rank-0 JSONL may never see it)
        writer.local("fatal", error=failure_record(error, "main"))
        raise
    finally:
        # pylint: disable=import-outside-toplevel
        from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator

        destroy_num_microbatches_calculator()
        if finished:  # after a failure the other ranks may never reach a barrier
            dist.barrier()
            dist.destroy_process_group()
        if args.hang_dump_s > 0:
            faulthandler.cancel_dump_traceback_later()
        determinism.close()


if __name__ == "__main__":
    main()
