# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Seven projected TV heads over a shared /DE MTP and certified packed CP2 halos."""

from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from megatron.core.context_parallel_layout import prebuild_thd_cp_partition_routes
from megatron.core.fusions import fused_mtp_tv
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import multi_token_prediction as mtp
from megatron.core.transformer.mtp_sequence_roll import (
    ZigzagPackedCPRollContext,
    prepare_mtp_sequence_roll_context,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_mtp_dsa_cross_depth_sharing import (
    BOTH_SHARED_COMPONENTS,
    _make_config,
)
from tests.unit_tests.transformer.test_mtp_tv_loss import _OutputLayer

DEPTHS, VOCAB = 7, 2048
PADDED, REAL = (0, 64, 112, 144), (57, 41, 0)


def _pack(cp):
    physical = torch.tensor(PADDED, device="cuda", dtype=torch.int32)
    real = torch.tensor((0, 57, 98, 98), device="cuda", dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=real,
        cu_seqlens_kv=real,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=64,
        max_seqlen_kv=64,
        cp_group=cp,
        local_cp_size=2,
        zigzag_cp_min_chunk_size=8,
        cp_partition_mode="zigzag",
    )
    prebuild_thd_cp_partition_routes(params, cp)
    positions, valid = [], []
    for start, end, length in zip(PADDED, PADDED[1:], REAL):
        chunk = (end - start) // 4
        for owner in (cp.rank(), 3 - cp.rank()):
            relative = torch.arange(owner * chunk, (owner + 1) * chunk, device="cuda")
            positions.append(relative)
            valid.append(relative < length)
    positions = torch.cat(positions).unsqueeze(0)
    valid = torch.cat(valid).unsqueeze(0)
    ids = (positions * 7 + 3).remainder(VOCAB)
    mask = valid & (positions != 17)
    return params, positions, ids, valid, mask


@pytest.mark.launch_on_gb200
def test_seven_e2e_tv_heads_remat_packed_cp2_shared_model_gradients_and_storage():
    if not torch.cuda.is_available() or Utils.world_size != 2:
        pytest.skip("requires two CUDA ranks")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(8311)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        params, positions, ids, valid, mask = _pack(pg.cp)
        context = prepare_mtp_sequence_roll_context(ids, pg.cp, params)
        assert isinstance(context, ZigzagPackedCPRollContext)
        config = _make_config(
            num_layers=156,
            mtp_num_layers=DEPTHS,
            mtp_detach_heads=True,
            mtp_loss_type="e2e_tv",
            mtp_loss_scaling_factor=0.67,
            calculate_per_token_loss=True,
            mtp_repeated_layer_shared_components=BOTH_SHARED_COMPONENTS,
            context_parallel_size=2,
            cp_comm_type="all_gather",
            num_moe_experts=4,
            moe_router_topk=2,
            moe_ffn_hidden_size=128,
            ffn_hidden_size=128,
            moe_grouped_gemm=True,
        )
        spec = hybrid_stack_spec.submodules
        model = (
            mtp.MultiTokenPredictionBlock(
                config=config,
                spec=spec.mtp_block_spec,
                pg_collection=pg,
                mtp_layer_pattern="DE",
                mtp_num_depths=DEPTHS,
                hybrid_submodules=spec,
            )
            .bfloat16()
            .cuda()
        )
        assert len(model.layers) == 1 and len(model.layers[0].mtp_model_layer.layers) == 2
        torch.manual_seed(2718)
        embed = torch.nn.Embedding(VOCAB, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        output_layer = _OutputLayer(
            torch.randn(VOCAB, config.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.10
        )
        base = torch.randn(ids.numel(), 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        # The main loss enters with a nonuniform cotangent. The depth-wise auxiliary
        # cotangents arise natively through the true seven-depth prefix product.
        main_grad = torch.randn_like(base)

        def run(enabled, chunk_size=0):
            config.mtp_recompute_tv_heads = enabled
            config.mtp_tv_head_chunk_size = chunk_size
            model.zero_grad(set_to_none=True)
            trunk = base.detach().requires_grad_()
            result = model(
                input_ids=ids,
                position_ids=positions,
                hidden_states=trunk,
                attention_mask=None,
                packed_seq_params=params,
                padding_mask=~valid,
                sequence_roll_context=context,
                sequence_roll_padding_mask=~valid,
                embedding=lambda input_ids, position_ids: embed(input_ids)
                .transpose(0, 1)
                .contiguous(),
            )
            result.retain_grad()
            calls = Counter()
            targets = set()
            halos = set()
            saved_logits, saved_bits = {}, {}
            draft_storage = set()
            prefix_grad = {}
            draft_row_counts = []
            normalized = []
            prefix_losses = []
            original_tv = mtp.vocab_parallel_tv_distance
            original_apply = mtp.MTPLossAutoScaler.apply
            original_fused = fused_mtp_tv._fused_vocab_parallel_tv_distance
            teacher_ptr = None
            in_backward = False

            def projection(module, inputs, out):
                nonlocal teacher_ptr
                kind = "teacher" if not torch.is_grad_enabled() else "draft"
                calls[kind, in_backward] += 1
                if kind == "teacher":
                    teacher_ptr = out[0].untyped_storage().data_ptr()
                else:
                    draft_storage.add(out[0].untyped_storage().data_ptr())

            def tv(draft, target, **kwargs):
                assert (
                    kwargs["target_row_indices"] is not None
                    and kwargs["target_halo_logits"] is not None
                )
                targets.add(target.untyped_storage().data_ptr())
                halos.add(kwargs["target_halo_logits"].untyped_storage().data_ptr())
                distance = original_tv(draft, target, **kwargs)
                if not in_backward:
                    index = len(draft_row_counts)
                    draft_row_counts.append(draft.shape[0])
                    distance.register_hook(
                        lambda g: prefix_grad.update({index: g.detach().clone()})
                    )
                return distance

            def fused(*args, **kwargs):
                calls["fused", in_backward] += 1
                return original_fused(*args, **kwargs)

            def pack(tensor):
                storage = tensor.untyped_storage()
                if (
                    tensor.dtype == torch.bfloat16
                    and tensor.ndim == 3
                    and tensor.shape[-1] == VOCAB
                    and storage.data_ptr() in draft_storage
                ):
                    saved_logits[storage.data_ptr()] = storage.nbytes()
                if (
                    tensor.dtype == torch.uint8
                    and tensor.ndim == 2
                    and tensor.shape[-1] == VOCAB // 8
                ):
                    saved_bits[storage.data_ptr()] = storage.nbytes()
                return tensor.detach()

            def scale(output, loss):
                normalized.append(loss.detach().clone())
                return original_apply(output, loss)

            def track(value, n, depth, num_depths, **kwargs):
                prefix_losses.append(value.detach().clone())

            mtp.MTPLossAutoScaler.set_loss_scale(torch.tensor(0.73, device="cuda"))
            with ExitStack() as h:
                h.callback(output_layer.register_forward_hook(projection).remove)
                h.enter_context(patch.object(mtp, "vocab_parallel_tv_distance", tv))
                h.enter_context(
                    patch.object(fused_mtp_tv, "_fused_vocab_parallel_tv_distance", fused)
                )
                h.enter_context(patch.object(mtp.MTPLossAutoScaler, "apply", scale))
                h.enter_context(
                    patch.object(mtp.MTPLossLoggingHelper, "save_loss_to_tracker", track)
                )
                # Outside the rematerializer so PyTorch's native inner checkpoint hooks still own its saves.
                with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                    output = mtp.process_mtp_loss(
                        hidden_states=result,
                        labels=ids,
                        loss_mask=mask.float(),
                        output_layer=output_layer,
                        output_weight=None,
                        runtime_gather_output=False,
                        is_training=True,
                        compute_language_model_loss=None,
                        config=config,
                        cp_group=pg.cp,
                        tp_group=pg.tp,
                        dp_cp_group=pg.cp,
                        packed_seq_params=params,
                        sequence_roll_context=context,
                        main_hidden_states=trunk,
                    )
                in_backward = True
                output.backward(main_grad)
            assert output_layer.weight.grad is None and embed.weight.grad is None
            assert calls["teacher", False] == 1 and calls["teacher", True] == 0
            row_limit = chunk_size or ids.numel()
            chunk_rows = [min(row_limit, ids.numel() - i) for i in range(0, ids.numel(), row_limit)]
            parts = len(chunk_rows)
            assert draft_row_counts == chunk_rows * DEPTHS
            assert calls["draft", False] == calls["fused", False] == DEPTHS * parts
            assert (
                calls["draft", True] == calls["fused", True] == (DEPTHS * parts if enabled else 0)
            )
            assert len(halos) == len(targets) == 1 and targets == {teacher_ptr}
            assert len(prefix_losses) == DEPTHS and len(prefix_grad) == DEPTHS * parts
            # TV distances use [rows, batch]; reconstruct their true incoming gradients
            # in depth and sequence order, independent of the checkpoint replay order.
            prefix_by_depth = [
                torch.cat([prefix_grad[d * parts + i] for i in range(parts)], dim=0)
                for d in range(DEPTHS)
            ]
            for grad in prefix_by_depth:
                assert grad.min() != grad.max() and grad.norm() > 0
            assert len(saved_logits) == len(saved_bits) == (0 if enabled else DEPTHS)
            obs = {
                "output": output.detach(),
                "native_loss": normalized[0],
                "main_grad": trunk.grad,
                "prefix_losses": torch.stack(prefix_losses),
                "prefix_cotangents": torch.stack(prefix_by_depth),
                "hidden_grad": result.grad,
            }
            for depth, grad in enumerate(result.grad.chunk(DEPTHS + 1)):
                assert torch.isfinite(grad).all() and (depth == 0 or grad.norm() > 0)
            grads = {
                n: p.grad.detach().clone() if p.grad is not None else None
                for n, p in model.named_parameters()
            }
            return obs, grads, (sum(saved_logits.values()), sum(saved_bits.values()))

        a, ag, old = run(False)
        b, bg, new = run(True)
        for key in a:
            torch.testing.assert_close(b[key], a[key], rtol=0, atol=0, msg=key)
        assert ag.keys() == bg.keys()
        for name, grad in ag.items():
            assert (grad is None) == (bg[name] is None), name
            if grad is not None:
                torch.testing.assert_close(bg[name], grad, rtol=0, atol=0, msg=name)
        # 72 CP-local rows, with a shorter final block across the all-dummy tail.
        c, cg, chunk_saved = run(True, chunk_size=16)
        assert chunk_saved == (0, 0)
        for key in a:
            torch.testing.assert_close(c[key], a[key], rtol=0, atol=0, msg=key)
        for name, expected in ag.items():
            assert (expected is None) == (cg[name] is None), name
            if expected is not None:
                torch.testing.assert_close(cg[name], expected, rtol=0, atol=0, msg=name)
        assert any(
            "linear_kv_down_proj" in k and v is not None and v.norm() > 0 for k, v in ag.items()
        )
        assert old == (
            DEPTHS * ids.numel() * VOCAB * 2,
            DEPTHS * ids.numel() * VOCAB // 8,
        ) and new == (0, 0)
        print(
            f"packed CP{pg.cp.size()} rank {pg.cp.rank()}: seven fused TV heads, "
            f"saved draft/sign bytes {old} -> {new}; {len(ag)} shared model parameters checked"
        )
    finally:
        mtp.MTPLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("chunk_size", [0, 16])
def test_seven_checkpointed_tv_heads_cuda_graph_replay_with_dynamic_inputs(chunk_size):
    """Capture addressed CP2 teacher halos and exact F+B, then change the graph inputs."""
    if not torch.cuda.is_available() or Utils.world_size != 2:
        pytest.skip("requires two CUDA ranks")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(5194)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        params, _, ids, _, mask = _pack(pg.cp)
        ctx = prepare_mtp_sequence_roll_context(ids, pg.cp, params)
        assert isinstance(ctx, ZigzagPackedCPRollContext)
        config = _make_config(
            mtp_num_layers=DEPTHS,
            mtp_detach_heads=True,
            mtp_loss_type="e2e_tv",
            mtp_loss_scaling_factor=0.67,
            calculate_per_token_loss=True,
        )
        torch.manual_seed(356)
        project = _OutputLayer(
            torch.randn(VOCAB, config.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.11
        )
        hidden = torch.randn(
            (DEPTHS + 1) * ids.numel(),
            1,
            config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        hidden.grad = torch.empty_like(hidden)
        main_grad = torch.randn_like(hidden[: ids.numel()])
        mask = mask.float()
        mtp.MTPLossAutoScaler.set_loss_scale(torch.tensor(0.73, device="cuda"))

        def step(value):
            value.grad.zero_()
            output = mtp.process_mtp_loss(
                hidden_states=value,
                labels=ids,
                loss_mask=mask,
                output_layer=project,
                output_weight=None,
                runtime_gather_output=False,
                is_training=True,
                compute_language_model_loss=None,
                config=config,
                cp_group=pg.cp,
                tp_group=pg.tp,
                dp_cp_group=pg.cp,
                packed_seq_params=params,
                sequence_roll_context=ctx,
            )
            output.backward(main_grad)
            return output

        # Logging happens once in native forward; its sums are validated by the model parity case.
        with patch.object(mtp.MTPLossLoggingHelper, "save_loss_to_tracker", lambda *a, **kw: None):
            config.mtp_recompute_tv_heads = True
            config.mtp_tv_head_chunk_size = chunk_size
            warm = torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for _ in range(3):
                    step(hidden)
            torch.cuda.current_stream().wait_stream(warm)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = step(hidden)
            for shift in (0.19, -0.88):
                with torch.no_grad():
                    hidden.add_(shift)
                expected = hidden.detach().clone().requires_grad_()
                expected.grad = torch.empty_like(expected)
                config.mtp_recompute_tv_heads = False
                config.mtp_tv_head_chunk_size = 0
                output = step(expected)
                graph.replay()
                torch.testing.assert_close(captured, output, rtol=0, atol=0)
                torch.testing.assert_close(hidden.grad, expected.grad, rtol=0, atol=0)
                assert all(g.norm() > 0 for g in hidden.grad.chunk(1 + DEPTHS))
            print(
                "CP2 graph replay matched two distinct updates, including all seven TV draft gradients"
            )
    finally:
        mtp.MTPLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()
