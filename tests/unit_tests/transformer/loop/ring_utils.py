# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Test helpers for the ring schedule (``loop_schedule='ring'``: the hybrid loop over pipeline
stages): the DSv4 Flash and parity-mini loop patterns, their ``'|'`` stage layouts, ring config
fields, and a stub pipeline group that lets one process build any stage.

Layout (the PP arm's design): P stages, k core slices per stage, P x k segments; segment
``s`` is the physical chunk ``(p, j) = (s % P, s // P)``. The core layers are split over the
stages as evenly as possible (extras to the middle stages first: Flash at P 4 is
``[9, 10, 10, 9]``), each stage's share over its k slices differing by at most one layer; the
first segment holds the prelude before its slice, the last one the coda after its slice.
"""

import contextlib

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.transformer.transformer_config import MLATransformerConfig

# DSv4 Flash (43 layers: W W C + (H C) x 20) and the parity mini (13 layers): prelude 3 layers
# (the hash-routed W W C), coda 2 layers (H C), MTP 'WE'.
FLASH_MAIN = "WEWECE" + "HECE" * 20
MINI_MAIN = "WEWECE" + "HECE" * 5
MTP = "WE"
PRELUDE = 6
CODA = 4
HASH_LAYERS = 3
_BLOCK_START = set("*DCHW+MGK")
_RATIOS = {"W": 0, "C": 4, "H": 128}


def blocks(symbols):
    """Split sublayer symbols into attention + MLP blocks (a block starts at a mixer)."""
    out = []
    for symbol in symbols:
        if symbol in _BLOCK_START or not out:
            out.append(symbol)
        else:
            out[-1] += symbol
    return out


def balanced(total, parts):
    """``parts`` sizes summing to ``total``: equal, extras to the middle stages first, then the
    last stage, then the first."""
    base, rem = divmod(total, parts)
    sizes = [base] * parts
    for index in (list(range(1, parts - 1)) + [parts - 1, 0])[:rem]:
        sizes[index] += 1
    return sizes


def ring_layout(main, prelude, coda, stages, chunks):
    """``(piped main pattern, segments)``; each segment a dict with its stage, chunk, pattern,
    global offset and core layer count."""
    core_blocks = blocks(main[prelude : len(main) - coda])
    split = balanced(len(core_blocks), stages)
    sub = [
        [share // chunks + (1 if j < share % chunks else 0) for j in range(chunks)]
        for share in split
    ]
    segments, cursor, offset = [], 0, 0
    for index in range(stages * chunks):
        stage, chunk = index % stages, index // stages
        count = sub[stage][chunk]
        body = "".join(core_blocks[cursor : cursor + count])
        cursor += count
        if index == 0:
            body = main[:prelude] + body
        if index == stages * chunks - 1:
            body += main[len(main) - coda :]
        segments.append(
            dict(index=index, stage=stage, chunk=chunk, pattern=body, offset=offset, core=count)
        )
        offset += len(body)
    return "|".join(segment["pattern"] for segment in segments), segments


def ring_fields(stages, chunks, **overrides):
    """The config fields of a ring at P = ``stages``, k = ``chunks`` (P2P overlap on)."""
    fields = dict(
        loop_schedule="ring",
        pipeline_model_parallel_size=stages,
        virtual_pipeline_model_parallel_size=chunks if chunks > 1 else None,
        pipeline_dtype=torch.float32,
        overlap_p2p_comm=True,
        batch_p2p_comm=False,
    )
    fields.update(overrides)
    return fields


def dsv4_loop_config(main, mtp=MTP, **overrides):
    """A small-width DSv4 loop config of ``main`` (W / C / H + 'E', MTP ``mtp``): MLA under
    ``dsv4_hybrid``, 8 experts with the prelude's MoE layers hash-routed, mHC 4, R = 4; the
    roles of the Flash pattern. Config only (its layers need Transformer Engine)."""
    ratios = [_RATIOS.get(symbol, 0) for symbol in main + mtp]
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
        experimental_attention_variant="dsv4_hybrid",
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=32,
        csa_window_size=32,
        csa_compress_ratios=ratios,
        num_moe_experts=8,
        moe_ffn_hidden_size=256,
        moe_router_topk=2,
        moe_token_dispatcher_type="alltoall",
        moe_num_hash_layers=HASH_LAYERS,
        hash_moe_vocab_size=128,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mtp_num_layers=1 if mtp else None,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        is_hybrid_model=True,
        loop_architecture="huginn",
        loop_iterations=4,
        loop_prelude_sublayers=PRELUDE,
        loop_coda_sublayers=CODA,
    )
    fields.update(overrides)
    return MLATransformerConfig(**fields)


class StageGroup:
    """A P-rank pipeline group seen from rank ``rank`` (a stub: ``rank()`` / ``size()``)."""

    def __init__(self, rank, size):
        self._rank, self._size = rank, size

    def rank(self):
        """This rank in the stub group."""
        return self._rank

    def size(self):
        """The stub group's size (P)."""
        return self._size


@contextlib.contextmanager
def stage_group_ranks():
    """``torch.distributed.get_rank`` / ``get_world_size`` answer a ``StageGroup`` from the
    stub (``select_pipeline_segment`` reads the pipeline rank that way); other groups as
    before."""
    real_rank, real_size = dist.get_rank, dist.get_world_size

    def get_rank(group=None):
        return group.rank() if isinstance(group, StageGroup) else real_rank(group)

    def get_world_size(group=None):
        return group.size() if isinstance(group, StageGroup) else real_size(group)

    dist.get_rank, dist.get_world_size = get_rank, get_world_size
    try:
        yield
    finally:
        dist.get_rank, dist.get_world_size = real_rank, real_size
