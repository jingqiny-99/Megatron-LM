"""Bounded integration of one live DSv4 core pair in the normal RSBF training schedule."""

import dataclasses
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from megatron.core.transformer.experimental_attention_variant.dsa_logging import (
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.loop_transformer_block import recompute_context
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, get_moe_metrics_tracker
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from tests.unit_tests.transformer.loop import test_loop_rsbf_executor as spec
from tests.unit_tests.transformer.loop.test_loop_rsbf_executor import (
    _model_parallel,
    deterministic_algorithms,
    hybrid,
)


def _model():
    return spec._dsv4_model(
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_expert_rank_capacity_factor=4.0,
        moe_permute_fusion=True,
        moe_grouped_gemm=True,
        use_transformer_engine_op_fuser=True,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        moe_mlp_glu_interleave_size=32,
    )


def _train_step(model, hybrid, batches, criterion, iteration):
    spec.set_moe_metrics_tracker(spec.MoEMetricsTracker())
    fwd, data = spec._forward_step_func(model, hybrid, batches)
    rows = spec.forward_backward_loop_rsbf(
        iteration=iteration,
        exit_criterion=criterion,
        **spec._schedule_kwargs(model, fwd, data, len(batches), seq_length=256),
    )
    stats = spec.get_loop_rsbf_schedule().cached_executor(model).stats
    assert [stats.depths[spec.SampleKey(i, iteration)] for i in range(2)] == [1, 2]
    assert sum(k == "core" for k, _, _ in stats.item_order) == 3
    return spec._lm_losses(rows)


def _context_values(ctx):
    return tree_flatten({f.name: getattr(ctx, f.name) for f in dataclasses.fields(ctx)})


class _CorePair(torch.nn.Module):
    """Capture the normal stack call with every tensor-valued context input supplied on replay."""

    def __init__(self, stack, ctx, start, end):
        super().__init__()
        self.pair = torch.nn.ModuleList(list(stack.layers[start:end]))
        self.invoke = stack._run_range
        self.start, self.end, self.ctx = start, end, ctx
        self.leaves, self.layout = _context_values(ctx)
        self.positions = [i for i, x in enumerate(self.leaves) if torch.is_tensor(x)]

    def args(self, hidden, ctx):
        flat, layout = _context_values(ctx)
        assert layout == self.layout
        return (hidden, *(flat[i] for i in self.positions))

    def forward(self, hidden, *args):
        flat = self.leaves.copy()
        for i, t in zip(self.positions, args):
            flat[i] = t
        ctx = dataclasses.replace(self.ctx, **tree_unflatten(flat, self.layout))
        # The executor normally supplies this scope around a recomputation. TE calls this
        # module outside the executor during capture, so the graph must establish it itself.
        with recompute_context("rsbf/core-pair/phase=recompute"):
            return self.invoke(hidden, self.start, self.end, ctx, "rsbf/core-pair")


def test_pair_preserves_auxiliary_derivative_without_counting_recomputation():
    """TE invokes the pair outside the executor during capture; recomputation semantics apply."""

    @dataclasses.dataclass
    class Context:
        scale: float

    tracker = MoEMetricsTracker()

    def run(hidden, first, last, ctx, label):
        auxiliary_loss = ctx.scale * hidden.square().sum()
        tracker.record("aux", auxiliary_loss, layer_number=1, num_layers=1)
        return MoEAuxLossAutoScaler.apply(hidden, auxiliary_loss)

    ctx = Context(3.0)
    stack = SimpleNamespace(layers=[torch.nn.Identity()], _run_range=run)
    x = torch.tensor([2.0], requires_grad=True)
    pair = _CorePair(stack, ctx, 0, 1)
    with patch.object(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0)):
        pair(x).sum().backward()
    assert not tracker._metrics
    assert torch.equal(x.grad, torch.tensor([13.0]))


def _statistics_buffers(model):
    """Strongly hold the tracked storage to check that capture and replay leave it untouched."""
    buffers = {
        f"moe/{name}/{field}": buf
        for name, entry in get_moe_metrics_tracker()._metrics.items()
        for field, buf in (("values", entry.values), ("round_values", entry.round_values))
        if buf is not None
    }
    buffers.update(
        {
            f"dsa/{name}": value
            for name, value in DSAIndexerLossLoggingHelper.tracker.items()
            if torch.is_tensor(value)
        }
    )
    buffers.update(
        {
            f"expert_counts/{name}": module.local_tokens_per_expert
            for name, module in model.named_modules()
            if getattr(module, "local_tokens_per_expert", None) is not None
        }
    )
    return buffers


def _capture_core_pair(actual, x, ctx, request):
    from transformer_engine.pytorch import make_graphed_callables

    from megatron.core.transformer import cuda_graphs
    from megatron.core.transformer.moe.paged_stash import paged_stash_te_graph_capture

    start, end = actual.decoder.loop_roles.core_start, actual.decoder.loop_roles.core_end
    pair = _CorePair(actual.decoder, ctx, start, end)
    args = pair.args(x, ctx)
    static = tuple(v.detach().clone().requires_grad_(v.requires_grad) for v in args)
    print("RSBF_PAIR: capture", [(tuple(v.shape), str(v.dtype)) for v in static], flush=True)
    stats_buffers = _statistics_buffers(actual)
    assert any(name.startswith("moe/") for name in stats_buffers)
    before_capture = {name: value.clone() for name, value in stats_buffers.items()}
    cuda_graphs._set_capture_start()
    try:
        with paged_stash_te_graph_capture(False, order=[1, -1], config=actual.config):
            replay = make_graphed_callables(
                pair,
                static,
                allow_unused_input=True,
                retain_graph_in_backward=False,
                pool=torch.cuda.graph_pool_handle(),
                clone_param_grads_on_return=True,
                fp8_enabled=False,
            )
    finally:
        cuda_graphs._set_capture_end()

    def release_graph():
        # NCCL communicators used by captured collectives must outlive the CUDA graph.
        # TE retains the graph in its callable, including through Python reference cycles.
        torch.cuda.synchronize()
        replay.reset()

    request.addfinalizer(release_graph)
    assert all(torch.equal(value, before_capture[name]) for name, value in stats_buffers.items())
    print("RSBF_PAIR: capture preserved", len(stats_buffers), "statistic buffers", flush=True)
    print("RSBF_PAIR: CUDA capture complete", flush=True)
    return pair, replay, stats_buffers, before_capture


@pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 1,
    reason="one CUDA execution unit",
)
def test_single_core_pair_matches_two_training_updates(hybrid, deterministic_algorithms, request):
    """Compare the real core pair's recomputations, losses, accumulated gradients and updates."""
    from tests.unit_tests.transformer.loop import pair_capture as pc

    if not (
        pc.hybridep_available(0) and pc.op_fuser_grouped_mlp_available() and pc._te_min("2.19.0")
    ):
        pytest.skip("needs Blackwell HybridEP and TE grouped experts with graph gradient copy-out")

    criterion = spec.PrescribedDepthCriterion({0: 1, 1: 2})
    seq = 256
    with spec._hadamard_fallback():
        eager, actual = _model(), _model()
        actual.load_state_dict(eager.state_dict())
        start, end = actual.decoder.loop_roles.core_start, actual.decoder.loop_roles.core_end
        assert end - start == 2
        print("RSBF_PAIR: models built", flush=True)
        sample = spec._batches(2, seq=seq, seed=11, dev=torch.device("cuda"))
        captured_input = []
        original = actual.decoder._run_range

        def observe(hidden, first, last, ctx, label):
            if not captured_input and start == first and last == end and hidden.requires_grad:
                captured_input.append((hidden.detach().clone().requires_grad_(), ctx))
            return original(hidden, first, last, ctx, label)

        with patch.object(actual.decoder, "_run_range", side_effect=observe):
            _train_step(actual, hybrid, sample, criterion, 0)
        assert len(captured_input) == 1
        print("RSBF_PAIR: eager schedule and observed pair complete", flush=True)
        x, ctx = captured_input[0]
        pair, replay, stats_buffers, before_capture = _capture_core_pair(actual, x, ctx, request)
        pc.copy_model_state(eager, actual)
        actual.zero_grad(set_to_none=True)
        overlapping_grad_checks = [0]
        for name, parameter in pair.pair.named_parameters():

            def check_owned(grad, p=parameter, name=name):
                if p.grad is not None:
                    assert (
                        p.grad.untyped_storage().data_ptr() != grad.untyped_storage().data_ptr()
                    ), name
                    overlapping_grad_checks[0] += 1

            parameter.register_hook(check_owned)
        count = [0]

        def apply_pair(hidden, first, last, ctx, label):
            if first == start and last == end and hidden.requires_grad:
                count[0] += 1
                return replay(*pair.args(hidden, ctx)).clone()
            return original(hidden, first, last, ctx, label)

        opts = [torch.optim.SGD(m.parameters(), lr=0.05) for m in (eager, actual)]
        with patch.object(actual.decoder, "_run_range", side_effect=apply_pair):
            for it in (1, 2):
                batches = spec._batches(2, seq=seq, seed=37 + it, dev=torch.device("cuda"))
                losses = []
                for m, opt in zip((eager, actual), opts):
                    opt.zero_grad(set_to_none=True)
                    losses.append(_train_step(m, hybrid, batches, criterion, it))
                spec._assert_losses_equal(*losses)
                # Each schedule entry installs its own tracker. The capture-time MoE tracker
                # stays live as a sentinel and must receive no writes from a replay.
                for name, buf in stats_buffers.items():
                    if name.startswith("moe/"):
                        assert torch.equal(buf, before_capture[name]), name
                compared = 0
                for (name, p), (other, q) in zip(
                    eager.named_parameters(), actual.named_parameters()
                ):
                    assert name == other and (p.grad is None) == (q.grad is None), name
                    if p.grad is not None:
                        assert torch.equal(p.grad, q.grad), name
                        compared += 1
                assert count[0] == 3 * it and compared > 0 and overlapping_grad_checks[0] > 0
                for opt in opts:
                    opt.step()
                assert all(
                    torch.equal(p, q) for p, q in zip(eager.parameters(), actual.parameters())
                )
                print(
                    "RSBF_PAIR: step",
                    it,
                    "pair forwards",
                    count[0],
                    "gradients",
                    compared,
                    "ownership comparisons",
                    overlapping_grad_checks[0],
                    flush=True,
                )
