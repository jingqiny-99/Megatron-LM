# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Block and edge ``loop_recompute`` with an expert-parallel MoE (EP = 4, alltoall).

A TE MoE hybrid loop stack ('*E' x 6: prelude, core and coda of two attention + MoE blocks each)
on 4 ranks, each rank with its own tokens, so the ranks route differently and every MoE
sublayer's dispatch / combine is a collective of the EP group. Every block and edge mode (strides
1, 2, 3 and C, edges off and on) gives the output, the input gradient and every parameter
gradient of no recompute bitwise on deterministic kernels; a mismatch is diagnosed by a
self-rerun (only a non-deterministic kernel falls back to cosine >= 0.99999, rel <= 1e-5).
Every checkpoint's routing tape is recorded and fully replayed, and the per-block tape counts
are the same on every rank (each block replays as many MoE collectives on every EP rank).

Its own module so that model parallel (with ``expert_model_parallel_size=4``) is initialized
once per module (the CI process-group budget).
"""

import os
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
from torch.utils.checkpoint import set_checkpoint_early_stop

from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import forward_replay, loop_transformer_block
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.loop import hybrid_stack
from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_stack_spec
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA, model_parallel_once
from tests.unit_tests.transformer.loop.hybrid_parity_utils import make_inputs, te_base, te_rope

pytestmark = pytest.mark.skipif(
    not CUDA or int(os.environ.get("WORLD_SIZE", "1")) < 4,
    reason="EP = 4 needs CUDA and at least 4 ranks",
)

EP = 4
PATTERN = "*E" * 6  # prelude '*E*E', core '*E*E', coda '*E*E'
P = C = D = 4
ROUNDS = 2
MODES = ["round"] + [(stride, edges) for stride in (1, 2, 3, C) for edges in (False, True)]


@pytest.fixture(scope="module", autouse=True)
def _expert_parallel():
    with model_parallel_once(expert_model_parallel_size=EP):
        yield


@pytest.fixture
def deterministic_algorithms(monkeypatch):
    """Deterministic kernels (set after the module's model-parallel initialization)."""
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _mode_fields(mode):
    if mode == "none":
        return {}
    if mode == "round":
        return dict(loop_recompute=True)
    stride, edges = mode
    return dict(loop_recompute=True, loop_recompute_block_stride=stride, loop_recompute_edges=edges)


def _mode_id(mode):
    return mode if isinstance(mode, str) else f"s{mode[0]}{'-edges' if mode[1] else ''}"


def _stack(mode):
    base = te_base(
        torch.bfloat16,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_ffn_hidden_size=256,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_aux_loss_coeff=0.01,
        moe_z_loss_coeff=1e-3,
        expert_model_parallel_size=EP,
        attention_backend=AttnBackend.unfused,
    )
    config = replace(
        base,
        num_layers=len(PATTERN),
        is_hybrid_model=True,
        loop_architecture="huginn",
        loop_iterations=ROUNDS,
        loop_prelude_sublayers=P,
        loop_coda_sublayers=D,
        **_mode_fields(mode),
    )
    torch.manual_seed(1234)
    stack = build_module(
        loop_hybrid_stack_spec,
        config=config,
        pre_process=True,
        layer_config_list=validate_segment_layers(PATTERN, config),
        post_process=True,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    ).cuda()
    assert type(stack) is LoopHybridStack
    return stack


def _run(stack, x, upstream, rotary):
    set_moe_metrics_tracker(MoEMetricsTracker())
    x = x.detach().clone().requires_grad_(True)
    stack.zero_grad(set_to_none=True)
    # Without early stop every replay runs to its block's end (replay_routing then also checks
    # that each tape was consumed).
    with set_checkpoint_early_stop(False):
        y = stack(x, attention_mask=None, rotary_pos_emb=rotary)
    ((y * upstream).float().sum() + sum(stack.loop_losses.values())).backward()
    grads = {
        name: param.grad.detach().float().clone()
        for name, param in stack.named_parameters()
        if param.grad is not None
    }
    return [y.detach().float().clone(), x.grad.detach().float().clone()], grads


def _rel_cos(actual, expected):
    a, b = actual.flatten().double(), expected.flatten().double()
    if not b.norm() and not a.norm():
        return 0.0, 1.0
    rel = ((a - b).norm() / b.norm()).item() if b.norm() else float("inf")
    return rel, torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _close(actual, expected):
    """The non-bitwise fallback gate: cosine >= 0.99999 and relative norm <= 1e-5."""
    rel, cosine = _rel_cos(actual, expected)
    return cosine >= 0.99999 and rel <= 1e-5


def _everywhere(flag):
    """True if ``flag`` holds on every rank (every rank must call this: a collective)."""
    value = torch.tensor([int(bool(flag))], device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.MIN)
    return bool(value.item())


def _moe_per_block(stride):
    """MoE sublayers per checkpoint of one forward, in checkpoint order (edges on)."""
    counts = []
    for start, length, repeats in ((0, P, 1), (P, C, ROUNDS), (P + C, D, 1)):
        segment = PATTERN[start : start + length]
        counts += [segment[cut : cut + stride].count("E") for cut in range(0, length, stride)] * (
            repeats
        )
    return counts


def test_ep_blocks_are_bitwise_and_replay_alike_on_every_rank(
    monkeypatch, deterministic_algorithms
):
    labels, tapes = [], []
    context_fn = hybrid_stack.loop_checkpoint_context_fn

    def recording_context_fn(prefix):
        inner = context_fn(prefix)

        def wrapped():
            labels.append(prefix)
            return inner()

        return wrapped

    class RecordingTape(forward_replay.RoutingTape):
        """A routing tape that registers itself."""

        def __init__(self):
            super().__init__()
            tapes.append(self)

    monkeypatch.setattr(hybrid_stack, "loop_checkpoint_context_fn", recording_context_fn)
    monkeypatch.setattr(loop_transformer_block, "RoutingTape", RecordingTape)

    reference_stack = _stack("none")
    config = reference_stack.config
    rank = dist.get_rank()
    x, upstream = make_inputs(
        torch.bfloat16, hidden=config.hidden_size, scale=config.init_method_std, seed=3 + rank
    )
    rotary = te_rope(config.kv_channels)
    tensors_ref, grads_ref = _run(reference_stack, x, upstream, rotary)
    assert not tapes
    # Every decision that changes the collective sequence (a rerun) is agreed by all ranks, and
    # problems are collected and raised on every rank at the end, so no rank leaves its EP peers
    # waiting in a collective.
    problems = []
    deterministic, rerun_grads = None, None
    for mode in MODES:
        labels.clear()
        tapes.clear()
        stack = _stack(mode)
        stack.load_state_dict(reference_stack.state_dict())
        tensors, grads = _run(stack, x, upstream, rotary)
        name = _mode_id(mode)
        keys_ok = grads.keys() == grads_ref.keys()
        if not keys_ok:
            problems.append(f"{name}: gradient keys differ")

        # The tapes: one per checkpoint call, every one fully replayed, the same per-block
        # counts on every EP rank.
        stats = [(label, len(tape.indices), tape.cursor) for label, tape in zip(labels, tapes)]
        if not len(tapes) == len(labels) == len(set(labels)):
            problems.append(f"{name}: {len(tapes)} tapes, {len(labels)} checkpoint labels")
        if not all(count == cursor for _, count, cursor in stats):
            problems.append(f"{name}: a tape not fully replayed {stats}")
        if mode != "round":
            stride, edges = mode
            expected = _moe_per_block(stride)
            if not edges:
                expected = expected[
                    len(range(0, P, stride)) : len(expected) - len(range(0, D, stride))
                ]
            if [count for _, count, _ in stats] != expected:
                problems.append(f"{name}: tape counts {stats}, expected {expected}")
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, stats)
        if not all(other == stats for other in gathered):
            problems.append(f"{name}: tape counts differ between ranks {gathered}")

        bitwise = (
            keys_ok
            and all(torch.equal(a, b) for a, b in zip(tensors, tensors_ref))
            and all(torch.equal(grads[key], grads_ref[key]) for key in grads_ref)
        )
        print(f"[T2] rank {rank} {name}: {'bitwise' if bitwise else 'not bitwise'}")
        if _everywhere(bitwise):
            continue
        if deterministic is None:
            labels.clear()
            tapes.clear()
            rerun_tensors, rerun_grads = _run(reference_stack, x, upstream, rotary)
            deterministic = _everywhere(
                all(torch.equal(a, b) for a, b in zip(rerun_tensors, tensors_ref))
                and all(torch.equal(rerun_grads[key], grads_ref[key]) for key in grads_ref)
            )
            print(f"[T2] rank {rank}: the reference repeats itself bitwise: {deterministic}")
        # Rank-local from here on (no collective below).
        if bitwise or not keys_ok:
            continue
        if deterministic:
            worst = max(((grads[k] - grads_ref[k]).abs().max().item(), k) for k in grads_ref)
            problems.append(f"{name}: not bitwise on deterministic kernels ({worst})")
            continue
        failing = [k for k in grads_ref if not _close(grads[k], grads_ref[k])]
        if failing or not all(_close(a, b) for a, b in zip(tensors, tensors_ref)):
            rows = []
            for key in failing[:10]:
                rel, cosine = _rel_cos(grads[key], grads_ref[key])
                floor, _ = _rel_cos(rerun_grads[key], grads_ref[key])
                rows.append(f"{key}: rel {rel:.2e} cos {cosine:.7f} (rerun rel {floor:.2e})")
            problems.append(f"{name}: {len(failing)} parameters off the fallback gate: {rows}")

    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, problems)
    failed = {r: p for r, p in enumerate(reports) if p}
    assert not failed, f"rank {rank} sees problems on ranks {sorted(failed)}: {failed}"
