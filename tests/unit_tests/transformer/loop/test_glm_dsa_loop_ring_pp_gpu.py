# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare the GLM separate-edge NCCL pipeline to the standard, tied, two-pass PP1 model.

Use torchrun with two, seven or eight ranks. PP7 exercises 26 + 28 + 24 physical DSA/FFN
blocks at the small widths of the native GLM parity test. Its fourth logical view ends
on the head-only last rank. The PP2 cases exercise four distinct microbatches.
"""

import os
from collections import Counter
from contextlib import ExitStack, contextmanager

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state as ps
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.loop.cross_loop_kv import current_cross_loop_pass
from megatron.core.transformer.loop.glm_dsa import loop_hybrid_glm_dsa_stack_spec
from megatron.core.transformer.loop.ring_pp import LoopRingSchedule
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.utils import get_attr_wrapped_model

from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA, VOCAB
from tests.unit_tests.transformer.loop.test_glm_dsa_loop_model_gpu import SEQUENCE, _config
from tests.unit_tests.transformer.loop.test_loop_attn_res_dsv4_gpu import full_precision_gemms

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(
    not CUDA or WORLD not in (2, 7, 8), reason="requires 2, 7 or 8 CUDA ranks"
)


@contextmanager
def _parallel(stages):
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=stages,
        virtual_pipeline_model_parallel_size=3 if stages > 1 else None,
    )
    torch.manual_seed(54)
    model_parallel_cuda_manual_seed(54)
    try:
        yield
    finally:
        Utils.destroy_model_parallel()


def _gather_objects(local):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, local)
    return values


def _global_name(name, offset):
    if name.startswith("decoder.layers."):
        _, _, local, suffix = name.split(".", 3)
        return f"decoder.layers.{int(local) + offset}.{suffix}"
    return name


def _build(stages, mtp, sections, *, rounds=2, counts=None):
    # Complete selector groups remain together in each section. The leading two
    # prelude blocks have independent selectors; all other groups contain four.
    physical = [
        "D-" * count if isinstance(count, int) else count for group in sections for count in group
    ]
    main = ("|" if stages > 1 else "").join(physical)
    pattern = main + ("/-" if mtp else "")
    extra = {}
    if stages > 1:
        extra.update(
            loop_schedule="ring",
            loop_ring_separate_edges=True,
            pipeline_model_parallel_size=stages,
            virtual_pipeline_model_parallel_size=3,
            pipeline_dtype=torch.float32,
            overlap_p2p_comm=True,
            batch_p2p_comm=False,
            microbatch_group_size_per_vp_stage=stages,
        )
    if counts is not None:
        extra.update(
            loop_depth_counts=list(counts),
            loop_depth_group_size=1,
            loop_depth_seed=1234,
            loop_ring_recompute_plan="/".join(["CnnCn"] * stages),
        )
    config = _config(
        True,
        counts is not None,
        num_layers=sum(map(len, physical)),
        prelude=sum(map(len, physical[: len(sections[0])])),
        coda=sum(map(len, physical[-len(sections[2]) :])),
        is_hybrid_model=True,
        deallocate_pipeline_outputs=True,
        loop_exit_criterion="stratified" if counts is not None else "fixed",
        loop_iterations=rounds,
        mtp_num_layers=1 if mtp else None,
        mtp_loss_scaling_factor=0.1,
        **extra,
    )
    models, offsets = [], []
    stage = ps.get_pipeline_model_parallel_rank()
    for v in range(3 if stages > 1 else 1):
        vp = v if stages > 1 else None
        model = HybridModel(
            config=config,
            hybrid_stack_spec=loop_hybrid_glm_dsa_stack_spec(config),
            hybrid_layer_pattern=pattern,
            vocab_size=VOCAB,
            max_sequence_length=SEQUENCE,
            position_embedding_type="rope",
            pre_process=ps.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp),
            post_process=ps.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp),
            vp_stage=vp,
        ).cuda()
        first = v * stages + stage if stages > 1 else 0
        offset = sum(len(chunk) for chunk in physical[:first])
        expected_layers = len(physical[first]) if stages > 1 else sum(map(len, physical))
        assert len(model.decoder.layers) == expected_layers
        offsets.append(offset)
        models.append(model)
    return models, offsets


def _forward_step(data_iterator, model):
    hybrid = get_attr_wrapped_model(model, "decoder", return_model_obj=True)
    batch = next(data_iterator) if hybrid.pre_process or hybrid.post_process else None
    index, rows = batch if batch is not None else (None, None)
    tokens = rows[:, :-1].contiguous() if rows is not None else None
    positions = torch.arange(SEQUENCE, device="cuda")[None] if rows is not None else None
    labels = rows[:, 1:].contiguous() if hybrid.post_process else None
    mask = torch.ones_like(labels, dtype=torch.float32) if labels is not None else None
    loss = model(tokens, positions, None, labels=labels, loss_mask=mask)

    def reduce_loss(output):
        return output.sum(), mask.sum().int(), (index, output.detach().cpu())

    return loss, reduce_loss


def _step(models, batches, ring, collect_mtp, *, counts=None, iteration=0, record_callback=None):
    config = models[0].config
    config.finalize_model_grads_func = finalize_model_grads
    ddp = DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False)
    wrapped = [
        DistributedDataParallel(config, ddp, model, disable_bucketing=i > 0)
        for i, model in enumerate(models)
    ]
    for model in wrapped:
        model.zero_grad_buffer()
    source = [iter(batches) if m.pre_process or m.post_process else None for m in models]
    schedule = LoopRingSchedule() if ring else get_forward_backward_func()
    if ring:
        schedule.set_iteration(iteration)
    with pytest.MonkeyPatch.context() as patch:
        # Isolate the native MTP logger across PP1 and PP runs, including group references.
        patch.setattr(MTPLossLoggingHelper, "tracker", {})
        reports = schedule(
            forward_step_func=_forward_step,
            data_iterator=source if ring else source[0],
            model=wrapped,
            num_microbatches=len(batches),
            seq_length=SEQUENCE,
            micro_batch_size=1,
            forward_only=False,
        )
        mtp = torch.zeros(2, device="cuda")
        if collect_mtp:
            tracker = MTPLossLoggingHelper.tracker
            if "loss_sums" in tracker:
                mtp.copy_(torch.cat((tracker["loss_sums"], tracker["num_tokens"])))
            dist.all_reduce(mtp)
    if ring:
        record = schedule.last_step
        if record_callback is not None:
            record_callback(record)
        scheduled_depth = (
            config.loop_iterations
            if counts is None or config.loop_exit_criterion == "online"
            else max(d for d, count in enumerate(counts, 1) if count)
        )
        assert record.views == scheduled_depth + 2 and record.microbatches == len(batches)
        if counts is None:
            assert record.rounds == (config.loop_iterations, config.loop_iterations)
        else:
            assert record.exit_counts == tuple(counts)
            assert record.rounds == (
                min(i for i, n in enumerate(counts, 1) if n),
                max(i for i, n in enumerate(counts, 1) if n),
            )
        assert record.prelude == record.coda == (1, 1)
        assert record.mtp == (int(collect_mtp), int(collect_mtp))
    return [pair for rank in _gather_objects(reports or []) for pair in rank], mtp


def _check_fixed_two_pass(mtp, layout="reference"):
    sections = ([2] + [4] * (WORLD - 1), [4] * WORLD, [4] * (WORLD - 1) + [0])
    if WORLD == 8:
        sections = (
            [1, 1, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, "D-D-D-D", "-"],
            [4, 4, 0, 0, 4, 4, 0, 8],
        )
    if layout == "balanced":
        sections = (
            [1, 1, 4, 4, 4, 4, 4, 4],
            [4, 4, "D-D-D-D", "-D-D-D-D-", "D-D-D-D", "-D-D-D-D-", "D-D-D-D", "-"],
            [4, 4, 4, 0, 4, 0, 4, 4],
        )
    if layout == "critical_path":
        sections = (
            [1, 1, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, "D-D-D-D", "-D-D-D-D", "-"],
            [4, 4, 4, 4, 4, 4, 0, 0],
        )
    batch_count = max(4, WORLD)
    rows = torch.randint(
        VOCAB, (batch_count, SEQUENCE + 1), generator=torch.Generator().manual_seed(734)
    )
    with _parallel(1):
        reference = _build(1, mtp, sections)[0][0]
        state = {
            name: value.clone() if torch.is_tensor(value) else value
            for name, value in reference.state_dict().items()
        }
        ids = range(
            ps.get_data_parallel_rank() * batch_count // WORLD,
            (ps.get_data_parallel_rank() + 1) * batch_count // WORLD,
        )
        expected_losses, expected_mtp = _step(
            [reference], [(i, rows[i : i + 1].cuda()) for i in ids], False, mtp
        )
        expected = {name: p.main_grad.detach().clone() for name, p in reference.named_parameters()}
    assert sorted(i for i, _ in expected_losses) == list(range(batch_count))
    expected_losses = dict(expected_losses)
    for r in range(2):
        for parameter in ("norm", "proj"):
            assert expected[f"decoder.loop_gates.{r}.{parameter}.weight"].norm() > 0
    if mtp:
        assert expected_mtp.min() > 0
        assert any(n.startswith("mtp.") and g.norm() > 0 for n, g in expected.items())
    del reference

    with _parallel(WORLD), ExitStack() as hooks:
        models, offsets = _build(WORLD, mtp, sections)
        for model, offset in zip(models, offsets):
            local_state = {name: state[_global_name(name, offset)] for name in model.state_dict()}
            model.load_state_dict(local_state, strict=True)
        # Both passes must visit every local core layer for each distinct microbatch.
        # The same first-pass KV bank is consumed by the second pass without rerunning top-k.
        visits = Counter()

        def observe(attention, _inputs):
            phase = current_cross_loop_pass()
            visits[attention.layer_number, phase.is_source if phase is not None else None] += 1

        selector_visits = Counter()

        def observe_selector(_module, _inputs):
            phase = current_cross_loop_pass()
            selector_visits[phase.is_source if phase is not None else None] += 1

        local_dsa = [
            layer for i, layer in enumerate(models[1].decoder.layers, offsets[1]) if i % 2 == 0
        ]
        tail_ffn_visits = []
        if WORLD == 8 and ps.get_pipeline_model_parallel_rank() == 7:
            assert offsets[1] == 107 and len(models[1].decoder.layers) == 1
            hooks.callback(
                models[1]
                .decoder.layers[0]
                .register_forward_pre_hook(lambda *_: tail_ffn_visits.append(True))
                .remove
            )

        for layer in local_dsa:
            attention = layer.self_attention.core_attention
            if attention.indexer is not None:
                hooks.callback(
                    attention.indexer.linear_wq_b.register_forward_pre_hook(observe_selector).remove
                )
            hooks.callback(
                layer.self_attention.core_attention.register_forward_pre_hook(observe).remove
            )
        observed, observed_mtp = _step(
            models, [(i, rows[i : i + 1].cuda()) for i in range(batch_count)], True, mtp
        )
        assert sorted(i for i, _ in observed) == list(range(batch_count))
        for i, values in observed:
            torch.testing.assert_close(values, expected_losses[i], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(observed_mtp, expected_mtp, rtol=1e-5, atol=1e-5)
        expected_visits = {
            (layer.self_attention.core_attention.layer_number, source): batch_count
            for layer in local_dsa
            for source in (False, True)
        }
        # Gather all names before checking rank-local values so a numerical mismatch
        # on one stage cannot strand another in a still-pending collective.
        names_and_grads = [
            (_global_name(name, offset), p.main_grad)
            for model, offset in zip(models, offsets)
            for name, p in model.named_parameters()
        ]
        holders = Counter(
            name for rank in _gather_objects([n for n, _ in names_and_grads]) for name in rank
        )
        assert holders == Counter(
            {n: 2 if mtp and n == "embedding.word_embeddings.weight" else 1 for n in expected}
        )
        failure = None
        try:
            assert visits == expected_visits
            if WORLD == 8 and ps.get_pipeline_model_parallel_rank() == 7:
                assert len(tail_ffn_visits) == 2 * batch_count
            # R2 consumes the R1 selection. The FFN-only PP8 tail owns no selector.
            assert selector_visits == ({True: batch_count} if local_dsa else {})
            gate_holders = [
                v
                for v, model in enumerate(models)
                if any(n.startswith("decoder.loop_gates.") for n, _ in model.named_parameters())
            ]
            assert gate_holders == (
                [1] if ps.get_pipeline_model_parallel_rank() == WORLD - 1 else []
            )
            assert [v for v, model in enumerate(models) if model.mtp_process] == (
                [2] if mtp and ps.get_pipeline_model_parallel_rank() == WORLD - 1 else []
            )
            for name, grad in names_and_grads:
                torch.testing.assert_close(
                    grad,
                    expected[name],
                    rtol=1e-4,
                    atol=2e-6,
                    msg=lambda reason, name=name: f"{name}: {reason}",
                )
        except AssertionError as exc:
            failure = f"PP rank {ps.get_pipeline_model_parallel_rank()}: {exc}"
        failures = [message for message in _gather_objects(failure) if message is not None]
        assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    "mtp", (False, True) if WORLD == 2 else (True,), ids=lambda x: "mtp" if x else "lm"
)
def test_fixed_two_pass_separate_edges_match_pp1(mtp, full_precision_gemms):
    _check_fixed_two_pass(mtp)


@pytest.mark.skipif(WORLD != 8, reason="requires the 78-block, eight-stage layout")
def test_balanced_pp8_three_shared_ffn_boundaries_match_pp1(full_precision_gemms):
    _check_fixed_two_pass(mtp=True, layout="balanced")


@pytest.mark.skipif(WORLD != 8, reason="requires the 78-block, eight-stage layout")
def test_critical_path_pp8_tail_ffns_match_pp1(full_precision_gemms):
    _check_fixed_two_pass(mtp=True, layout="critical_path")
