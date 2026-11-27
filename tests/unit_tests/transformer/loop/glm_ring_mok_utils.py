# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Small two-pass GLM model and input shared by the eight-rank MoK parity tests."""

import traceback
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.nn.functional as F

from megatron.core import parallel_state as ps
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.loop.glm_dsa import loop_hybrid_glm_dsa_stack_spec
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.loop.hybrid_loop_utils import VOCAB

CHUNKS = ("D-D-", "D-", "DE" * 4, "DE", "DE", "")
SEQUENCE = 512  # Minimum local token count in a MoK launch.
EP, EXPERTS, TOPK = 4, 16, 8
MICROBATCHES, SAMPLES = 2, 8


@contextmanager
def parallel(pp, cp=1):
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=3 if pp == 2 else None,
        context_parallel_size=cp,
        expert_model_parallel_size=EP,
    )
    torch.manual_seed(54)
    model_parallel_cuda_manual_seed(54)
    try:
        yield
    except Exception:
        # A local assertion can send this rank into the teardown barrier while peers
        # are still training. Preserve the originating stack in the distributed log.
        traceback.print_exc()
        raise
    finally:
        Utils.destroy_model_parallel()


def sample_rows():
    return torch.randint(
        VOCAB, (SAMPLES, SEQUENCE + 1), generator=torch.Generator().manual_seed(734)
    )


def batches(rows, ids):
    return [(i, rows[i : i + 1].cuda()) for i in ids]


def config(pp, *, moon=False, backend="mok", cp=1, rounds=2, counts=None, thresholds=None):
    """Build either PP layout directly so all inferred layout flags agree."""
    assert pp in (1, 2)
    assert backend == "mok" or (backend is None and not moon)
    checkpoint = counts is not None or thresholds is not None
    return MLATransformerConfig(
        num_attention_heads=4,
        normalization="RMSNorm",
        hidden_dropout=0,
        attention_dropout=0,
        add_bias_linear=False,
        use_cpu_initialization=True,
        multi_latent_attention=True,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_head_dim=16,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        qk_layernorm=True,
        rope_type="rope",
        apply_rope_fusion=False,
        attention_backend=AttnBackend.unfused,
        experimental_attention_variant="dsa",
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=32,
        dsa_indexer_topk=8,
        dsa_indexer_topk_freq=4,
        dsa_indexer_skip_topk_offset=3,
        dsa_indexer_loss_coeff=0.01,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_rotate_activation=False,
        calculate_per_token_loss=True,
        loop_architecture="huginn",
        loop_carrier="gated_residual",
        loop_iterations=rounds,
        context_parallel_size=cp,
        cp_comm_type="allgather" if cp > 1 else None,
        loop_recompute=checkpoint,
        loop_recompute_block_stride=4 if checkpoint else None,
        loop_recompute_edges=checkpoint,
        loop_ring_recompute_plan="/".join(["CnnCn"] * pp) if pp == 2 and checkpoint else None,
        loop_depth_counts=counts,
        loop_exit_thresholds=thresholds,
        loop_depth_group_size=1,
        loop_depth_seed=1234 if counts is not None else 0,
        loop_reuse_sparse_kv=True,
        loop_sparse_kv_window_size=4,
        num_layers=sum(map(len, CHUNKS)),
        loop_prelude_sublayers=len(CHUNKS[0] + CHUNKS[1]),
        loop_coda_sublayers=len(CHUNKS[4] + CHUNKS[5]),
        is_hybrid_model=True,
        loop_exit_criterion=(
            "online" if thresholds is not None else "stratified" if counts is not None else "fixed"
        ),
        loop_schedule="ring" if pp == 2 else "standard",
        loop_ring_separate_edges=pp == 2,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=3 if pp == 2 else None,
        microbatch_group_size_per_vp_stage=2 if pp == 2 else None,
        pipeline_dtype=torch.bfloat16,
        overlap_p2p_comm=pp == 2,
        batch_p2p_comm=False,
        deallocate_pipeline_outputs=True,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        hidden_size=256,
        ffn_hidden_size=512,
        params_dtype=torch.bfloat16,
        bf16=True,
        gradient_accumulation_fusion=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        expert_model_parallel_size=EP,
        expert_tensor_parallel_size=1,
        num_moe_experts=EXPERTS,
        moe_ffn_hidden_size=256,
        moe_shared_expert_intermediate_size=256,
        moe_grouped_gemm=True,
        moe_router_topk=TOPK,
        moe_router_pre_softmax=True,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=0.001,
        moe_router_topk_scaling_factor=2.5,
        moe_router_load_balancing_type="none",
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="alltoall",
        moe_megakernel_backend=backend,
        moe_virtual_expert_load_balance=moon,
    )


def models(
    pp, *, moon=False, backend="mok", segments=CHUNKS, cp=1, rounds=2, counts=None, thresholds=None
):
    cfg = config(
        pp, moon=moon, backend=backend, cp=cp, rounds=rounds, counts=counts, thresholds=thresholds
    )
    pattern = ("|" if pp == 2 else "").join(segments) + "/E"
    chunks = []
    for vp in range(3) if pp == 2 else [None]:
        chunk = HybridModel(
            config=cfg,
            hybrid_stack_spec=loop_hybrid_glm_dsa_stack_spec(cfg),
            hybrid_layer_pattern=pattern,
            vocab_size=VOCAB,
            max_sequence_length=SEQUENCE * cp,
            position_embedding_type="rope",
            share_embeddings_and_output_weights=False,
            pre_process=ps.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp),
            post_process=ps.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp),
            vp_stage=vp,
        ).cuda()
        chunks.append(convert_module_to_dtype_except_fp32_marked(chunk, torch.bfloat16))
    return chunks


def watch_compaction(hooks):
    """Collect equality checks on the indices and weights actually consumed by MoK."""
    from megatron.core.transformer.moe.megakernel.mok import backend as mok

    checks = []
    compact = mok.routing_map_to_mok_inputs

    def checked(probs, mask, topk):
        weights, indices = compact(probs, mask, topk)
        # Verify the actual kernel input, including ordering of its router weights.
        with torch.no_grad():
            ids = indices.long().clamp(0, EXPERTS - 1)
            restored_mask = torch.zeros_like(mask, dtype=torch.bool).scatter_(1, ids, True)
            checks.append(
                (indices >= 0).all()
                & (indices < EXPERTS).all()
                & (restored_mask == mask.bool()).all()
                & (weights == probs.gather(1, ids).float()).all()
            )
        return weights, indices

    hooks.enter_context(patch.object(mok, "routing_map_to_mok_inputs", checked))
    return checks
