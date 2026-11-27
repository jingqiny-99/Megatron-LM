# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bound the physical DeepGEMM logits slab on CP2 and compare a recurrent DSA step."""

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import (
    dsa_deepgemm_scorer,
    dsa_deepselect_topk,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.loop.cross_loop_kv import CrossLoopPass
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant import (
    test_dsa_cudnn_skip_score_fill_layer as layer,
)
from tests.unit_tests.transformer.experimental_attention_variant import (
    test_dsa_deepgemm_layer_gpu as deepgemm,
)


def _two_passes(attention, operands, params, launches):
    attention.zero_grad(set_to_none=True)
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    generator = torch.Generator(device="cuda").manual_seed(159 + torch.distributed.get_rank())
    bank = {}
    up = operands[0]["up_v_weight"].detach().requires_grad_(True)
    query, key, output, upstream = [], [], [], []
    counts = []
    for index, data in enumerate(operands):
        visit = CrossLoopPass(index, bank)
        query.append(data["query"].detach().requires_grad_(True))
        key.append(data["key"].detach().requires_grad_(True))
        with visit.activate():
            y = attention(
                query=query[-1],
                key=key[-1],
                value=None,
                attention_mask=None,
                x=data["x"],
                qr=data["qr"],
                attn_mask_type=AttnMaskType.causal,
                packed_seq_params=params,
                up_v_weight=up,
            )
        if index == 0:
            visit.finish_source()
        output.append(y)
        upstream.append(torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=generator))
        counts.append(len(launches))
    assert counts[0] > 0 and counts[0] == counts[1], "the consumer must reuse the source top-k"
    # Backpropagate the consumer alone to check both its own sliding window and the source bank.
    consumer_grads = torch.autograd.grad(output[1], key, upstream[1], retain_graph=True)
    torch.autograd.backward(output, upstream)
    result = {
        "indices": bank[attention, "dsa_topk"].detach(),
        "lengths": bank[attention, "dsa_topk_length"].detach(),
        "out1": output[0].detach(),
        "out2": output[1].detach(),
        "loss": DSAIndexerLossLoggingHelper.tracker["values"][0].detach().clone(),
        "consumer_dsource": consumer_grads[0],
        "consumer_dwindow": consumer_grads[1],
        "dup": up.grad,
        "dq1": query[0].grad,
        "dq2": query[1].grad,
        "dk1": key[0].grad,
        "dk2": key[1].grad,
    }
    for name, parameter in attention.named_parameters():
        assert parameter.grad is not None, name
        result["grad." + name] = parameter.grad.detach()
    return result


def test_cp2_packed_slab_bound_preserves_two_pass_selection_and_gradients(monkeypatch):
    deepgemm._require_deepgemm_layer(2, monkeypatch)
    if dsa_deepselect_topk._load_provider() is None:
        pytest.skip("this real-kernel comparison requires DeepSelect")
    # Both ranks have >16K visible keys, the threshold for native DeepSelect BF16 top-k.
    layout = layer._Layout(2, 2, False, 16384, (24576, 4096), 4, 128)
    layer._initialize_model_parallel(layout)
    try:
        deepgemm._warm_up_this_rank(indexer_heads=32)
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        config = layer._make_config(layout, deterministic=False)
        deepgemm._use_deepgemm(config)
        config.dsa_indexer_topk_backend = "deepselect"
        config.loop_reuse_sparse_kv = True
        config.loop_sparse_kv_window_size = 128
        attention = layer._make_attention(config, groups)
        params = layer._packed_seq_params(layout, groups.cp, groups.cp.rank())
        inputs = [layer._make_inputs(config, layout, groups.cp.rank() + seed) for seed in (0, 37)]

        launches, selected, ties = [], [], []
        score = dsa_deepgemm_scorer.mqa_logits
        select = dsa_deepselect_topk.deepselect_topk

        def record_score(*args, **kwargs):
            logits = score(*args, **kwargs)
            launches.append((logits.shape[0], logits.untyped_storage().nbytes()))
            return logits

        def record_select(scores, lengths, *args, **kwargs):
            selected.append(scores.shape[0])
            # Count cutoff ties on valid long rows so strict index equality covers tie handling.
            sampled = scores[::256]
            bounds = lengths.reshape(-1)[::256]
            valid = torch.arange(scores.shape[1], device=scores.device) < bounds[:, None]
            ranked = sampled.masked_fill(~valid, -torch.inf).topk(layout.topk + 1).values
            ties.append(((ranked[:, -1] == ranked[:, -2]) & (bounds > layout.topk)).sum())
            return select(scores, lengths, *args, **kwargs)

        monkeypatch.setattr(dsa_deepgemm_scorer, "mqa_logits", record_score)
        monkeypatch.setattr(dsa_deepselect_topk, "deepselect_topk", record_select)
        config.dsa_cudnn_topk_scratch_gib = 8
        expected = _two_passes(attention, inputs, params, launches)
        assert len(launches) == 1 and sum(selected) == layout.local_rows
        assert sum(ties).item() > 0
        large_bytes = launches[0][1]
        launches.clear()
        selected.clear()
        ties.clear()
        budget = 64 * 1024**2
        config.dsa_cudnn_topk_scratch_gib = budget / 1024**3
        actual = _two_passes(attention, inputs, params, launches)
        assert sum(n for n, _ in launches) == sum(selected) == layout.local_rows
        assert len(launches) > 1 and max(size for _, size in launches) <= budget < large_bytes
        assert sum(ties).item() > 0

        for name, reference in expected.items():
            value = actual[name]
            if name in ("indices", "lengths", "out1", "out2", "loss"):
                assert torch.equal(value, reference), name
            else:
                # Sparse dKV uses atomic sums on BF16 operands. Use the production layer's
                # per-tensor bounds; these also cover BF16 reduction of the CP partials.
                error = (value.float() - reference.float()).abs()
                assert reference.isfinite().all() and value.isfinite().all(), name
                assert reference.abs().max() > 0, name
                assert error.norm() <= layer._PRODUCTION_REL_L2 * reference.float().norm(), name
                assert error.max() <= layer._PRODUCTION_PEAK * reference.abs().max(), name
        print(
            f"CP rank {groups.cp.rank()}: {large_bytes} bytes -> {len(launches)} slabs, "
            f"largest {max(size for _,size in launches)} bytes, cutoff ties {sum(ties).item()}"
        )
    finally:
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()
