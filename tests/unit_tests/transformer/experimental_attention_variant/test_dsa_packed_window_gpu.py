# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Two-bank packed DSA on CP2 with a short document, per-document padding and a dummy tail."""

from functools import partial
from unittest.mock import patch

import torch

from megatron.core.context_parallel_layout import prebuild_thd_cp_partition_routes
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.loop.ring_pp import RingCrossLoopPass
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant import (
    test_dsa_checkpoint_topk_reuse as kernels,
    test_dsa_cudnn_skip_score_fill_layer as layer,
)

WINDOW = 128
PHYSICAL = (0, 1536, 1600, 2048)
REAL_LENGTHS = (1491, 55, 0)


def _geometry(rank):
    # Generate query positions directly from the two equal zigzag pieces assigned to
    # each CP rank. The second document has only 16 tokens per piece, below WINDOW.
    positions, starts, valid = [], [], []
    for first, last, real in zip(PHYSICAL, PHYSICAL[1:], REAL_LENGTHS):
        piece = (last - first) // 4
        for owner_piece in (rank, 3 - rank):
            rows = torch.arange(owner_piece * piece, (owner_piece + 1) * piece, device="cuda")
            positions.append(first + rows)
            starts.append(torch.full_like(rows, first))
            valid.append(rows < real)
    return tuple(torch.cat(x) for x in (positions, starts, valid))


def _step(model, cfg, layout, group, *, compact):
    cfg.loop_compact_packed_window = compact
    model.zero_grad(set_to_none=True)
    dsa.DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    rank = group.rank()
    physical = torch.tensor(PHYSICAL, device="cuda", dtype=torch.int32)
    real = torch.tensor((0, *REAL_LENGTHS), device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=real,
        cu_seqlens_kv=real,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=PHYSICAL[-1],
        max_seqlen_kv=PHYSICAL[-1],
        cp_group=group,
        cp_partition_mode="zigzag",
    )
    prebuild_thd_cp_partition_routes(params, group)
    assert tuple(params.thd_cp_host_cu_seqlens_q) == PHYSICAL
    pos, start, valid = _geometry(rank)
    assert valid.any() and (~valid).any()
    bank, outputs, inputs, grads, result = {}, {}, {}, {}, {}
    bytes_by_layer, selector_count = [], []
    sparse = dsa._run_sparse_attention
    score = model[0].indexer.forward_before_topk

    def select(*args, **kwargs):
        selector_count.append(dsa.current_cross_loop_pass().is_source)
        return score(*args, **kwargs)

    def inspect(**kwargs):
        if not dsa.current_cross_loop_pass().is_source:
            indices = kwargs["topk_indices"].squeeze(0)
            count = kwargs["topk_length"].reshape(-1)
            physical_positions = kwargs["key_positions"]
            joint = kwargs["key"]
            assert joint.size(0) == physical_positions.numel()
            assert (count[~valid] == 0).all()
            # Canonicalize the compressed addresses, preserving bank identity.
            present = torch.arange(indices.size(-1), device="cuda") < count[:, None]
            is_current = (indices >= PHYSICAL[-1]) & present
            canonical = physical_positions[indices.long().clamp_min(0)]
            canon_with_bank = (canonical + is_current.long() * PHYSICAL[-1]).masked_fill(
                ~present, -1
            )
            j = len(bytes_by_layer)
            nsource = bank[model[j], "dsa_topk_length"].reshape(-1)
            for column_count in (is_current.sum(-1), count - nsource):
                assert torch.equal(
                    column_count, (pos - start + 1).clamp_max(WINDOW).masked_fill(~valid, 0)
                )
            selected_rows, selected_slots = is_current.nonzero(as_tuple=True)
            expected_first = torch.maximum(start, pos - WINDOW + 1)
            assert torch.equal(
                canonical[selected_rows, selected_slots],
                expected_first[selected_rows] + selected_slots - nsource[selected_rows],
            )
            # Every valid document-start query must include its current token in both banks.
            at_start = (pos == start) & valid
            assert ((canon_with_bank == pos[:, None]).any(-1)[at_start]).all()
            assert ((canon_with_bank == pos[:, None] + PHYSICAL[-1]).any(-1)[at_start]).all()
            result["support", j] = canon_with_bank
            bytes_by_layer.append((joint.size(0) - PHYSICAL[-1], joint.untyped_storage().nbytes()))
        return sparse(**kwargs)

    with (
        patch.object(dsa, "_run_sparse_attention", inspect),
        patch.object(model[0].indexer, "forward_before_topk", select),
        patch.object(dsa, "gather_packed_window_kv", wraps=dsa.gather_packed_window_kv) as gather,
    ):
        for lap in range(2):
            visit = RingCrossLoopPass(lap, bank)
            with visit.activate():
                for j, mod in enumerate(model):
                    data = layer._make_inputs(cfg, layout, rank + 101 * lap + 7 * j)
                    for name in ("key", "query"):
                        inputs[lap, j, name] = data[name].requires_grad_()
                    if lap == 0:
                        inputs["up", j] = data["up_v_weight"].requires_grad_()
                    data["up_v_weight"] = inputs["up", j]
                    y = mod(
                        **data,
                        value=None,
                        attention_mask=None,
                        attn_mask_type=AttnMaskType.causal,
                        packed_seq_params=params,
                    )
                    outputs[lap, j] = y
                    result["output", lap, j] = y.detach()
                    gen = torch.Generator(device="cuda").manual_seed(1007 * lap + 13 * j + rank)
                    grads[lap, j] = torch.randn(
                        y.shape, dtype=y.dtype, device="cuda", generator=gen
                    )
            if lap == 0:
                visit.finish_source()
        assert gather.call_count == (len(model) if compact else 0)
    assert selector_count == [True]
    for j, mod in enumerate(model):
        result["indices", j] = bank[mod, "dsa_topk"]
    # Only rank zero backpropagates this boundary query. Its preceding W-1 values
    # belong to rank one, so a positive gradient there exercises remote halo backward.
    boundary = (pos == 1152).view(-1, 1, 1)
    remote = torch.autograd.grad(
        outputs[1, 0], inputs[1, 0, "key"], grads[1, 0] * boundary, retain_graph=True
    )[0]
    if rank == 1:
        halo_rows = (pos >= 1152 - (WINDOW - 1)) & (pos < 1152)
        assert remote[halo_rows].float().norm() > 0 and torch.count_nonzero(remote[~halo_rows]) == 0
    result["remote"] = remote
    for lap in (1, 0):
        torch.autograd.backward(
            [outputs[lap, j] for j in range(len(model))], [grads[lap, j] for j in range(len(model))]
        )
        if lap == 1:
            for j, mod in enumerate(model):
                result["consumer_to_source", j] = bank[mod, "dsa_latent"].grad.detach().clone()
    for key, x in inputs.items():
        result["input_grad", key] = x.grad
    for name, p in model.named_parameters():
        result["param_grad", name] = p.grad
    result["loss"] = dsa.DSAIndexerLossLoggingHelper.tracker["values"].detach().clone()
    return result, bytes_by_layer


def test_packed_cp2_halo_matches_global_bank_outputs_support_and_gradients():
    kernels._skip_if_cudnn_dsa_unavailable(2)
    layout = layer._Layout(2, 2, False, 1024, REAL_LENGTHS[:2], 4, 128)
    layer._initialize_model_parallel(layout)
    try:
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        dsa.DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        config = layer._make_config(layout, deterministic=False)
        config.dsa_indexer_skip_topk_offset = 1
        config.loop_reuse_sparse_kv = True
        config.loop_sparse_kv_window_size = WINDOW
        skip = partial(
            dsa.DSAttention,
            config=config,
            submodules=dsa.DSAttentionSubmodules(),
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            cp_comm_type="allgather",
            pg_collection=groups,
        )
        model = (
            torch.nn.ModuleList(
                [layer._make_attention(config, groups), *(skip(layer_number=i) for i in (2, 3, 4))]
            )
            .cuda()
            .train()
        )
        baseline, old = _step(model, config, layout, groups.cp, compact=False)
        actual, new = _step(model, config, layout, groups.cp, compact=True)
        assert len(old) == len(new) == 4
        assert all(r == PHYSICAL[-1] and x < r and y < b for (r, b), (x, y) in zip(old, new))
        assert baseline.keys() == actual.keys()
        for key, ref in baseline.items():
            val = actual[key]
            if key == "loss" or key[0] in ("support", "indices", "output"):
                assert torch.equal(ref, val), key
            else:
                assert (
                    torch.isfinite(ref).all() and torch.isfinite(val).all() and ref.norm() > 0
                ), key
                error = (val.float() - ref.float()).abs()
                assert error.norm() <= layer._PRODUCTION_REL_L2 * ref.float().norm(), key
                assert error.max() <= layer._PRODUCTION_PEAK * ref.abs().max(), key
        print(
            f"CP {groups.cp.rank()} R2 current rows / actual joint KV storage bytes "
            f"{old[0]} -> {new[0]}, {len(actual)} checked tensors"
        )
    finally:
        dsa.DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()
