# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare GLM RSBF training to the standard schedule on the same context shards."""

import os
from collections import Counter
from contextlib import ExitStack

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import _build_default_pg_collection
from megatron.core.transformer.loop.cross_loop_kv import current_cross_loop_pass
from megatron.core.transformer.loop.program import FixedDepthCriterion, LoopPhase, LoopProgram
from megatron.core.transformer.loop.rsbf_executor import LoopRSBFExecutor
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from tests.unit_tests.transformer.loop import test_loop_rsbf_executor as base
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA, model_parallel_once
from tests.unit_tests.transformer.loop.test_glm_dsa_loop_model_gpu import CORE, SEQUENCE, _model
from tests.unit_tests.transformer.loop.test_loop_attn_res_dsv4_gpu import full_precision_gemms

hybrid = base.hybrid
deterministic_algorithms = base.deterministic_algorithms


@pytest.mark.skipif(
    not CUDA or int(os.environ.get("WORLD_SIZE", "1")) not in (1, 2, 4, 16),
    reason="launch on one GPU, CP2, CP2 x DP2, or CP16",
)
@pytest.mark.parametrize(
    "edge_checkpoint_stride,offload,mtp",
    [(None, False, False), (4, False, False), (4, True, False), pytest.param(4, False, True, id="mtp")],
)
def test_rsbf_gated_two_pass_matches_standard(
    hybrid, deterministic_algorithms, full_precision_gemms, edge_checkpoint_stride, offload, mtp, monkeypatch
):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    cp = 16 if world == 16 else min(world, 2)
    # Use complete front/back chunk pairs on every native zigzag CP rank.
    sequence = ((SEQUENCE + 2 * cp - 1) // (2 * cp)) * (2 * cp)
    with model_parallel_once(context_parallel_size=cp), ExitStack() as hooks:
        dp_rank = parallel_state.get_data_parallel_rank()
        cp_rank = parallel_state.get_context_parallel_rank()
        torch.manual_seed(54)
        head = dict(
            mtp_pattern="E",
            mtp_num_layers=1,
            mtp_loss_scaling_factor=0.1,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_ffn_hidden_size=64,
            moe_router_load_balancing_type="none",
            is_hybrid_model=True,
        ) if mtp else {}
        standard = _model(True, False, cp=cp, sequence_length=sequence, **head)
        rsbf = _model(
            True,
            False,
            cp=cp,
            sequence_length=sequence,
            edge_checkpoint_stride=edge_checkpoint_stride,
            loop_schedule="rsbf",
            loop_rsbf_item_form="block",
            loop_rsbf_block_stride=4,
            loop_rsbf_offload_sparse_kv=offload,
            loop_rsbf_waves=2 if offload else 1,
            **head,
        )
        rsbf.load_state_dict(standard.state_dict())
        batches = base._batches(2, seq=sequence, seed=23 + dp_rank, dev=torch.device("cuda"))
        if cp > 1:
            chunks = torch.arange(sequence, device="cuda").chunk(2 * cp)
            local = torch.cat([chunks[cp_rank], chunks[-1 - cp_rank]])
            batches = [
                {name: None if v is None else v.index_select(1, local) for name, v in batch.items()}
                for batch in batches
            ]
        if mtp:
            monkeypatch.setattr(MTPLossLoggingHelper, "tracker", {})
        expected_losses, expected_grads = base._eager(standard, hybrid, batches)
        if mtp:
            expected_mtp = {
                key: MTPLossLoggingHelper.tracker[key].detach().clone()
                for key in ("loss_sums", "num_tokens")
            }
            assert all(value.sum() > 0 for value in expected_mtp.values())
            assert any("mtp." in n and g.norm() > 0 for n, g in expected_grads.items())
            monkeypatch.setattr(MTPLossLoggingHelper, "tracker", {})
        calls = Counter()

        def count_selector(_module, _args):
            scope = current_cross_loop_pass()
            assert scope.is_source, "pass two invoked a selector"
            calls[scope.phase] += 1

        owners = 0
        for layer in rsbf.decoder.layers[::2]:
            attention = layer.self_attention.core_attention
            if attention.layer_number in CORE and attention.indexer is not None:
                owners += 1
                hooks.callback(
                    attention.indexer.linear_wq_b.register_forward_pre_hook(count_selector).remove
                )
        program = LoopProgram(rsbf, exit_criterion=FixedDepthCriterion(2))
        executor = LoopRSBFExecutor(program, pg_collection=_build_default_pg_collection())
        reports = executor.run_step(
            base._next_batch,
            iter(base._microbatches(rsbf, hybrid, batches)),
            len(batches),
            model=rsbf,
        )
        metrics = executor.stats.sparse_kv_offload
        if offload:
            bank_bytes = len(CORE) * sequence * (32 + 16) * 4
            assert metrics["peak_pinned_bytes"] == bank_bytes  # one sample slot reused by both waves
            assert metrics["kv_d2h_bytes"] == metrics["grad_d2h_bytes"] == 2 * bank_bytes
            assert metrics["grad_h2d_bytes"] == 2 * bank_bytes
            assert metrics["kv_h2d_bytes"] == 4 * bank_bytes  # sweep and consumer replay
        else:
            assert metrics == {}
        assert owners == 2
        assert calls[LoopPhase.SWEEP] == calls[LoopPhase.RECOMPUTE] == owners * len(batches)
        torch.testing.assert_close(base._lm_losses(reports), expected_losses, rtol=1e-5, atol=1e-5)
        if mtp:
            for key, expected in expected_mtp.items():
                torch.testing.assert_close(MTPLossLoggingHelper.tracker[key], expected)
        actual_grads = base._grads(rsbf)
        assert actual_grads.keys() == expected_grads.keys()
        for i in range(2):
            for param in ("norm", "proj"):
                assert expected_grads[f"0.decoder.loop_gates.{i}.{param}.weight"].norm() > 0
        for name, expected in expected_grads.items():
            torch.testing.assert_close(
                actual_grads[name],
                expected,
                rtol=1e-4,
                atol=2e-6,
                msg=lambda reason, name=name: f"{name}: {reason}",
            )


@pytest.mark.skipif(not CUDA or int(os.environ.get("WORLD_SIZE", "1")) != 8, reason="CP2 DP4 EP4")
@pytest.mark.parametrize("waves", [3, 1])
def test_stratified_depth_matches_individual_fixed_depth_models(
    hybrid, deterministic_algorithms, full_precision_gemms, monkeypatch, waves
):
    """Differentiate ordinary K=1,2,3 models independently, including the two KV consumers."""
    from megatron.core.transformer.loop.depth_schedule import stratified_depth
    from tests.unit_tests.transformer.loop import test_glm_dsa_loop_model_gpu as glm

    with model_parallel_once(context_parallel_size=2, expert_model_parallel_size=4), ExitStack() as hooks:
        cp_rank = parallel_state.get_context_parallel_rank()
        dp_rank = parallel_state.get_data_parallel_rank()
        assert parallel_state.get_data_parallel_world_size() == 4
        torch.manual_seed(54)
        # Exercise communication through selected routed experts on all three visits and in MTP.
        # FP32 all-to-all gives the small oracle an independent, deterministic native reference.
        monkeypatch.setattr(glm, "PATTERN", "DE" * 13)
        common = dict(
            cp=2, mtp_pattern="E", mtp_num_layers=1, mtp_loss_scaling_factor=0.1,
            num_moe_experts=8, expert_model_parallel_size=4, moe_router_topk=2,
            moe_ffn_hidden_size=64, moe_router_load_balancing_type="none",
            moe_token_dispatcher_type="alltoall", is_hybrid_model=True,
        )
        rsbf = _model(
            True, False, edge_checkpoint_stride=4, loop_iterations=3, loop_schedule="rsbf",
            loop_exit_criterion="stratified", loop_depth_counts=[4, 4, 4],
            loop_depth_group_size=4, loop_depth_seed=1234, loop_rsbf_waves=waves,
            loop_rsbf_item_form="block", loop_rsbf_block_stride=4, **common,
        )
        state = rsbf.state_dict()
        batches = base._batches(3, seq=SEQUENCE, seed=77 + dp_rank, dev=torch.device("cuda"))
        parts = torch.arange(SEQUENCE, device="cuda").chunk(4)
        local = torch.cat([parts[cp_rank], parts[3 - cp_rank]])
        batches = [{k: None if v is None else v.index_select(1, local) for k, v in b.items()} for b in batches]
        iteration = 2
        ids = [4 * j + dp_rank for j in range(3)]
        depths = [stratified_depth(rsbf.config, iteration, g) for g in ids]
        assert sorted(depths) == [1, 2, 3]
        expected_grads = {}
        expected_losses = []
        monkeypatch.setattr(MTPLossLoggingHelper, "tracker", {})
        for depth, batch in zip(depths, batches):
            standard = _model(True, False, loop_iterations=depth, **common)
            standard.load_state_dict({k: state[k] for k in standard.state_dict()})
            losses, grads = base._eager(standard, hybrid, [batch])
            expected_losses.extend(losses)
            for key, value in grads.items():
                if key in expected_grads:
                    expected_grads[key].add_(value)
                else:
                    expected_grads[key] = value
            del standard
        expected_mtp = {k: MTPLossLoggingHelper.tracker[k].clone() for k in ("loss_sums", "num_tokens")}
        monkeypatch.setattr(MTPLossLoggingHelper, "tracker", {})
        selectors = Counter()
        owners = 0

        def actual_selector(module, inputs):
            scope = current_cross_loop_pass()
            assert scope.is_source, "a later pass reran the selector"
            selectors[scope.phase] += 1

        for layer in rsbf.decoder.layers[::2]:
            attention = layer.self_attention.core_attention
            if attention.layer_number in CORE and attention.indexer is not None:
                owners += 1
                hooks.callback(attention.indexer.linear_wq_b.register_forward_pre_hook(actual_selector).remove)
        program = LoopProgram(rsbf)
        executor = LoopRSBFExecutor(program, pg_collection=_build_default_pg_collection())
        reports = executor.run_step(base._next_batch, iter(base._microbatches(rsbf, hybrid, batches)),
                                    3, model=rsbf, iteration=iteration)
        assert selectors == {LoopPhase.SWEEP: owners * 3, LoopPhase.RECOMPUTE: owners * 3}
        expected_rounds = {(g, r) for g, k in zip(ids, depths) for r in range(1, k + 1)}
        for kind in ("core_sweep", "core_item"):
            visits = [(rec.sample, rec.round) for rec in executor.stats.work if rec.kind == kind]
            assert len(visits) == len(expected_rounds) and set(visits) == expected_rounds
        assert all(not any(moves) for moves in executor.stats.moves_by_round)
        assert all(value == 0 for value in executor.stats.nulls.values())
        assert {key.global_index: k for key, k in executor.stats.depths.items()} == {
            i: stratified_depth(rsbf.config, iteration, i) for i in range(12)
        }
        torch.testing.assert_close(base._lm_losses(reports), expected_losses, rtol=1e-5, atol=1e-5)
        for key, wanted in expected_mtp.items():
            torch.testing.assert_close(MTPLossLoggingHelper.tracker[key], wanted)
        actual = base._grads(rsbf)
        assert actual.keys() == expected_grads.keys()
        for i in range(3):
            assert expected_grads[f"0.decoder.loop_gates.{i}.proj.weight"].norm() > 0
        for name, expected in expected_grads.items():
            torch.testing.assert_close(actual[name], expected, rtol=1e-4, atol=2e-6,
                                       msg=lambda reason, name=name: f"{name}: {reason}")


@pytest.mark.skipif(not CUDA or int(os.environ.get("WORLD_SIZE", "1")) != 8,
                    reason="launch CP2 x DP4 with EP4")
@pytest.mark.parametrize("check_recompute", [False, "report"], ids=["training", "debug-replay"])
def test_online_refill_matches_fixed_models(
    hybrid, deterministic_algorithms, full_precision_gemms, monkeypatch, check_recompute
):
    """Same real inputs, weights and thresholds. The reference differentiates a fixed model per depth."""
    from megatron.core.transformer.loop.dispatcher import RSBFGroups
    from megatron.core.transformer.loop.online_exit import round_exit_score
    from megatron.core.transformer.loop.rsbf_plan import SubRoundType
    from tests.unit_tests.transformer.loop import test_glm_dsa_loop_model_gpu as glm

    with model_parallel_once(context_parallel_size=2, expert_model_parallel_size=4):
        monkeypatch.setattr(glm, "PATTERN", "DE" * 13)
        cp = parallel_state.get_context_parallel_group()
        dp_rank = parallel_state.get_data_parallel_rank()
        torch.manual_seed(54)
        common = dict(cp=2, mtp_pattern="E", mtp_num_layers=1, mtp_loss_scaling_factor=0.1,
                      num_moe_experts=8, expert_model_parallel_size=4, moe_router_topk=2,
                      moe_ffn_hidden_size=64, moe_router_load_balancing_type="none",
                      moe_token_dispatcher_type="alltoall", is_hybrid_model=True)
        rsbf = _model(True, False, edge_checkpoint_stride=4, loop_iterations=3, loop_schedule="rsbf",
                      loop_exit_criterion="online", loop_exit_thresholds=[0., 0.], loop_rsbf_waves=1,
                      loop_rsbf_fill="refill", loop_rsbf_refill_window=2,
                      loop_rsbf_item_form="block", loop_rsbf_block_stride=4, **common)
        state = rsbf.state_dict()
        parts = torch.arange(SEQUENCE, device="cuda").chunk(4)
        local = torch.cat([parts[cp.rank()], parts[3 - cp.rank()]])
        corpus = [{k: None if v is None else v.index_select(1, local) for k, v in row.items()}
                  for row in base._batches(3, seq=SEQUENCE, seed=975, dev=torch.device("cuda"))]

        # Probe real gate outputs once to select two thresholds between distinct scores.
        # Then permute the same three input sequences across homes to force online refills.
        calibration = _model(True, False, loop_iterations=3, **common)
        calibration.load_state_dict(state)
        scores = [[], []]
        with ExitStack() as hooks, torch.no_grad():
            for i, gate in enumerate(calibration.decoder.loop_gates[:2]):
                def observe(module, args, output, i=i):
                    scores[i].append(round_exit_score(output, cp_group=cp).item())
                hooks.callback(gate.register_forward_hook(observe).remove)
            for row in corpus:
                calibration(row["tokens"], row["position_ids"], None, labels=row["labels"],
                            loss_mask=row["loss_mask"])
        del calibration
        shortest, *rest = sorted(range(3), key=scores[0].__getitem__)
        medium, longest = sorted(rest, key=scores[1].__getitem__)
        assert scores[0][shortest] < min(scores[0][i] for i in rest)
        assert scores[1][medium] < scores[1][longest]
        thresholds = torch.tensor([(scores[0][shortest] + min(scores[0][i] for i in rest)) / 2,
                                   (scores[1][medium] + scores[1][longest]) / 2], device="cuda")
        torch.distributed.broadcast(thresholds, src=0)
        rsbf.config.loop_exit_thresholds = thresholds.tolist()
        by_depth = [corpus[i] for i in (shortest, medium, longest)]
        order = [1, 2, 3] if dp_rank % 2 == 0 else [3, 2, 1]
        batches = [by_depth[d - 1] for d in order]
        expected_grads, expected_losses = {}, []
        monkeypatch.setattr(MTPLossLoggingHelper, "tracker", {})
        for d, row in enumerate(by_depth, 1):
            standard = _model(True, False, loop_iterations=d, **common)
            standard.load_state_dict({k: state[k] for k in standard.state_dict()})
            losses, grads = base._eager(standard, hybrid, [row])
            expected_losses.extend(losses)
            for name, grad in grads.items():
                if name in expected_grads:
                    expected_grads[name].add_(grad)
                else:
                    expected_grads[name] = grad
            del standard
        mtp = {k: MTPLossLoggingHelper.tracker[k].clone() for k in ("loss_sums", "num_tokens")}
        monkeypatch.setattr(MTPLossLoggingHelper, "tracker", {})
        pg = _build_default_pg_collection()
        groups = RSBFGroups(domain=pg.dp, status=parallel_state.get_data_parallel_group_gloo())
        program = LoopProgram(rsbf)
        # The training path consumes the saved exit bit. Debug mode independently
        # evaluates the replay's state and compares its decision to the original one.
        probes = Counter()
        step = program.criterion.step
        def probe(state, crit, r, valid, depth):
            probes[r] += 1
            return step(state, crit, r, valid, depth)
        monkeypatch.setattr(program.criterion, "step", probe)
        executor = LoopRSBFExecutor(program, pg_collection=pg, groups=groups, check_recompute=check_recompute)
        selectors = Counter()
        def observe_selector(module, args):
            visit = current_cross_loop_pass()
            assert visit.is_source
            selectors[visit.phase] += 1
        with ExitStack() as hooks:
            owners = 0
            for layer in rsbf.decoder.layers[::2]:
                attn = layer.self_attention.core_attention
                if attn.layer_number in CORE and attn.indexer is not None:
                    owners += 1
                    hooks.callback(attn.indexer.linear_wq_b.register_forward_pre_hook(observe_selector).remove)
            reads = []
            def input_rows():
                for batch in base._microbatches(rsbf, hybrid, batches):
                    reads.append(sum(rec.kind == "core_sweep" for rec in executor.stats.work))
                    yield batch
            reports = executor.run_step(base._next_batch, input_rows(), 3, model=rsbf)
            assert reads[0] == 0 and reads[1] == 1 and reads[2] > reads[1]
        assert selectors == {LoopPhase.SWEEP: owners * 3, LoopPhase.RECOMPUTE: owners * 3}
        assert executor.stats.exit_counts == [4, 4, 4]
        assert executor.stats.refill_peak_live == 2
        n = 2 if check_recompute else 1
        assert probes == {1: 3 * n, 2: 2 * n, 3: n}
        if check_recompute:
            compared = [v for k, v in executor.stats.recompute_diffs.items() if k.startswith("exit[")]
            assert len(compared) == 6 and all(v == 0 for v in compared)
        assert len(executor.stats.refill_scores) == 5
        assert {key.global_index: d for key, d in executor.stats.depths.items()} == {
            4 * j + e: ([1, 2, 3] if e % 2 == 0 else [3, 2, 1])[j] for e in range(4) for j in range(3)
        }
        # Home 0 starts a previously unexecuted sample as home 1 advances its original live one.
        assert any(t.type == SubRoundType.CORE_F and t.items[0] is not None and t.items[1] is not None
                   and (t.items[0].sample, t.items[0].round) == (4, 1)
                   and (t.items[1].sample, t.items[1].round) == (1, 2) for t in executor.stats.refill_ticks)
        torch.testing.assert_close(base._lm_losses(reports), [expected_losses[d - 1] for d in order],
                                   rtol=1e-5, atol=1e-5)
        for k, expected in mtp.items():
            torch.testing.assert_close(MTPLossLoggingHelper.tracker[k], expected)
        actual = base._grads(rsbf)
        assert actual.keys() == expected_grads.keys()
        for name, expected in expected_grads.items():
            torch.testing.assert_close(actual[name], expected, rtol=1e-4, atol=2e-6,
                                       msg=lambda failure, name=name: f"{name}: {failure}")
