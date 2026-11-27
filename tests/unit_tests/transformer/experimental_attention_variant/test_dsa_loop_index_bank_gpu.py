# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real CP2 four-layer DSA selector sharing across two packed recurrent visits."""

from contextlib import nullcontext
from functools import partial
from unittest.mock import patch

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa, dsa_loop_kv
from megatron.core.transformer.loop.ring_pp import RingCrossLoopPass
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant import (
    test_dsa_checkpoint_topk_reuse as kernels,
    test_dsa_cudnn_skip_score_fill_layer as layer,
)


def _legacy(_visit, indices, lengths, *, packed_layout=None, **mask):
    return dsa_loop_kv.compact_source_topk(indices, lengths, **mask)


def _resident(tensors):
    storages = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in tensors}
    return len(storages), sum(storages.values())


def _run(model, config, layouts, group, *, legacy):
    model.zero_grad(set_to_none=True)
    dsa.DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    rank = group.rank()
    params = [layer._packed_seq_params(layout, group, rank) for layout in layouts]
    # Use actual local per-query padding rows. Different examples also have different
    # document boundaries and a different validity bitmap at exactly the same shape.
    for mb, metadata in enumerate(params):
        rows = torch.arange(layouts[mb].local_rows, device="cuda")
        metadata.real_token_mask_q = ((rows + mb) % 17 != 0).unsqueeze(0)

    banks = [{}, {}]
    leaves, outputs, upstream, result = {}, {}, {}, {}
    selections = []
    scorer = model[0].indexer.forward_before_topk

    def actual_select(*args, **kwargs):
        selections.append(dsa.current_cross_loop_pass().is_source)
        return scorer(*args, **kwargs)

    method = patch.object(dsa, "compact_loop_source_topk", _legacy) if legacy else nullcontext()
    with method, patch.object(model[0].indexer, "forward_before_topk", actual_select):
        for lap in range(2):
            # Run both source microbatches before either consumer.
            for mb, layout in enumerate(layouts):
                visit = RingCrossLoopPass(lap, banks[mb])
                with visit.activate():
                    for j, attn in enumerate(model):
                        slot = mb, lap, j
                        data = layer._make_inputs(
                            config, layout, rank + 11 * j + 101 * mb + 1009 * lap
                        )
                        for name in ("query", "key"):
                            leaves[slot, name] = data[name].requires_grad_()
                        if ("up_v_weight", j) not in leaves:
                            leaves["up_v_weight", j] = data["up_v_weight"].requires_grad_()
                        data["up_v_weight"] = leaves["up_v_weight", j]
                        y = attn(
                            **data,
                            value=None,
                            attention_mask=None,
                            attn_mask_type=AttnMaskType.causal,
                            packed_seq_params=params[mb],
                        )
                        outputs[slot] = y
                        generator = torch.Generator(device="cuda").manual_seed(
                            3001 + rank + 10 * j + 100 * mb + lap
                        )
                        upstream[slot] = torch.randn(
                            y.shape, dtype=y.dtype, device=y.device, generator=generator
                        )
                        result["out", slot] = y.detach()
                if lap == 0:
                    visit.finish_source()
    assert selections == [True, True], selections
    storage = {}
    for mb, bank in enumerate(banks):
        for name in ("dsa_topk", "dsa_topk_length", "dsa_latent"):
            values = [bank[attn, name] for attn in model]
            storage[mb, name] = _resident(values)
            count = 4 if legacy or name == "dsa_latent" else 1
            assert storage[mb, name][0] == count, (mb, name, storage[mb, name])
            for j, value in enumerate(values):
                result[mb, j, name] = value.detach()
        idx = bank[model[0], "dsa_topk"]
        lens = bank[model[0], "dsa_topk_length"]
        valid = params[mb].real_token_mask_q
        assert idx.dtype == lens.dtype == torch.int32
        assert (idx[~valid] == -1).all() and (lens[~valid] == 0).all()
        assert (lens[valid] > 0).any()
    for name in ("dsa_topk", "dsa_topk_length", "dsa_latent"):
        assert _resident([bank[model[0], name] for bank in banks])[0] == len(banks), name
    # The physical ring runs the consumer backward first and merges each bank leaf's
    # contribution when the corresponding source stage subsequently runs backward.
    for lap in (1, 0):
        keys = [slot for slot in outputs if slot[1] == lap]
        torch.autograd.backward([outputs[k] for k in keys], [upstream[k] for k in keys])
        if lap == 1:
            for mb, bank in enumerate(banks):
                for j, attn in enumerate(model):
                    result["consumer-to-source", mb, j] = (
                        bank[attn, "dsa_latent"].grad.detach().clone()
                    )
    for key, leaf in leaves.items():
        result["grad", key] = leaf.grad
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        result["param", name] = param.grad.detach()
    result["indexer-loss"] = dsa.DSAIndexerLossLoggingHelper.tracker["values"].detach().clone()
    return storage, result


def test_cp2_four_layers_share_one_masked_index_bank_with_exact_support():
    kernels._skip_if_cudnn_dsa_unavailable(2)
    layouts = (
        layer._Layout(2, 2, False, 512, (608, 272), 4, 128),
        layer._Layout(2, 2, False, 512, (416, 480), 4, 128),
    )
    layer._initialize_model_parallel(layouts[0])
    try:
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        dsa.DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        config = layer._make_config(layouts[0], deterministic=False)
        config.dsa_indexer_skip_topk_offset = 1
        config.loop_reuse_sparse_kv = True
        config.loop_sparse_kv_window_size = 128
        # Skip layers have no trainable indexer and share the resulting selector tensor.
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
                [layer._make_attention(config, groups), *(skip(layer_number=j) for j in (2, 3, 4))]
            )
            .cuda()
            .train()
        )
        assert [m.skip_topk for m in model] == [False, True, True, True]
        before, reference = _run(model, config, layouts, groups.cp, legacy=True)
        after, actual = _run(model, config, layouts, groups.cp, legacy=False)
        for mb in range(2):
            for field in ("dsa_topk", "dsa_topk_length"):
                assert before[mb, field][1] == 4 * after[mb, field][1]
        assert actual.keys() == reference.keys()
        for key, ref in reference.items():
            value = actual[key]
            if key == "indexer-loss" or key[0] == "out" or isinstance(key[0], int):
                assert torch.equal(value, ref), key
            else:
                assert ref.isfinite().all() and value.isfinite().all() and ref.norm() > 0, key
                error = (value.float() - ref.float()).abs()
                assert error.norm() <= layer._PRODUCTION_REL_L2 * ref.float().norm(), key
                assert error.max() <= layer._PRODUCTION_PEAK * ref.abs().max(), key
        print(
            f"CP {groups.cp.rank()} index/length bytes per pack "
            f"{before[0,'dsa_topk'][1]}/{before[0,'dsa_topk_length'][1]} -> "
            f"{after[0,'dsa_topk'][1]}/{after[0,'dsa_topk_length'][1]}, "
            f"{len(actual)} tensors checked"
        )
    finally:
        dsa.DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()


def test_same_source_mask_and_bound_changes_do_not_alias():
    kernels._skip_if_cudnn_dsa_unavailable(2)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
    try:
        source = torch.tensor(
            [[[0, 2, -1], [0, 1, 2], [0, 1, 2], [1, 2, 3]]], device="cuda", dtype=torch.int32
        )
        ends = torch.tensor([1, 2, 3, 4], device="cuda")
        mask = torch.tensor([[True, True, True, False]], device="cuda")
        visit = RingCrossLoopPass(0, {})

        lengths = torch.tensor([[1, 2, 3, 3]], dtype=torch.int32, device="cuda")

        def compact(bounds, validity):
            return dsa.compact_loop_source_topk(
                visit, source, lengths, causal_ends=bounds, valid_rows=validity
            )

        old, old_len = compact(ends, mask)
        again, _ = compact(ends, mask)
        assert old.data_ptr() == again.data_ptr()
        changed = mask.clone()
        changed[:, 0] = False
        new, new_len = compact(ends, changed)
        assert old.data_ptr() != new.data_ptr() and new_len[0, 0] == 0 and old_len[0, 0] == 1
        assert (new[0, 0] == -1).all()
        shortened, _ = compact(ends.clamp_max(1), mask)
        # Native lengths already encode support. New bounds must still invalidate the entry.
        assert shortened.data_ptr() != old.data_ptr()
        mask[:, 2] = False
        mutated, mutated_length = compact(ends, mask)
        assert mutated.data_ptr() != old.data_ptr() and mutated_length[0, 2] == 0
        visit.finish_source()
    finally:
        Utils.destroy_model_parallel()
