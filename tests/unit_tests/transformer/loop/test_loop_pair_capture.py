# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Spike (j): whole-layer capture of DSv4 attention + E pairs on HybridStack (``pair_capture``).

CPU: the spike configs construct and take the static-budget validator branch; the replay orders
issue their forwards, backwards and poison replays in the designed sequence; the pure helpers.
GPU (Blackwell, HybridEP, the TE op fuser; ``small`` width, S = 512, EP = WORLD_SIZE): arm A
(dev's TE helper) against the eager twin, arm-B pair graphs in every item order with the graph
pool poisoned between items, output aliasing, and the over-budget flag after a replay.
"""

import contextlib
import functools
import gc
import itertools
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.unit_tests.transformer.loop import pair_capture as pc
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA, model_parallel_once

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
SEQ = 512
_EXISTING_ASSERT = (
    "moe cuda graph is only supported with drop-padding MoE or transformer_engine "
    "sync-free HybridEP with rank capacity and paged stash."
)


GPU_REASON = "needs HybridEP + the TE op fuser on Blackwell (run on lyris GB200)"


@functools.lru_cache(maxsize=1)
def _gpu_ready() -> bool:
    """Evaluated lazily (in the module fixture, not at collection) and on this rank's own
    device (``LOCAL_RANK``), so collecting the module never touches device 0's context."""
    if not CUDA or WORLD < 2:
        return False
    device = int(os.environ.get("LOCAL_RANK", "0"))
    return pc.hybridep_available(device) and pc.op_fuser_grouped_mlp_available()


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    """Once per module: EP = WORLD_SIZE where the GPU cases run, else a one-rank world; the
    GPU cases run with deterministic TE algorithms and torch's deterministic algorithms (the
    spike's numerics, ``pc.deterministic_algorithms``; the environment is only validated here,
    on a copy: NCCL and cuBLAS of this process are already initialized)."""
    GPU = _gpu_ready()  # pylint: disable=invalid-name
    saved_env = os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO")
    determinism = contextlib.ExitStack()
    if GPU:
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
        determinism.enter_context(pc.deterministic_algorithms(env=dict(os.environ)))
    with determinism, model_parallel_once(expert_model_parallel_size=WORLD if GPU else 1):
        if GPU:
            # pylint: disable=import-outside-toplevel
            from megatron.core.num_microbatches_calculator import (
                destroy_num_microbatches_calculator,
                init_num_microbatches_calculator,
            )

            # One microbatch per rank (the TE helper reads it); a calculator another module left
            # behind would make the init assert.
            destroy_num_microbatches_calculator()
            init_num_microbatches_calculator(dist.get_rank(), None, WORLD, 1, WORLD)
        try:
            yield
        finally:
            if GPU:
                destroy_num_microbatches_calculator()
                if saved_env is None:
                    os.environ.pop("NVTE_ALLOW_NONDETERMINISTIC_ALGO", None)
                else:
                    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = saved_env
            if CUDA:
                torch.cuda.empty_cache()


@pytest.fixture
def te_gates(monkeypatch):
    """Without Transformer Engine (the CPU host), let the config-time TE version gates pass:
    these tests construct configs only."""
    try:
        import transformer_engine  # pylint: disable=import-outside-toplevel,unused-import

        return
    except ImportError:
        pass
    from packaging.version import Version  # pylint: disable=import-outside-toplevel

    from megatron.core.transformer import transformer_config  # pylint: disable=C0415
    from megatron.core.transformer.moe import moe_utils  # pylint: disable=C0415

    monkeypatch.setattr(transformer_config, "is_te_min_version", lambda *a, **k: True)
    monkeypatch.setattr(transformer_config, "get_te_version", lambda: Version("2.16.0"))
    for name in (
        "fused_permute",
        "fused_permute_with_probs",
        "fused_sort_chunks_by_index",
        "fused_sort_chunks_by_index_with_probs",
        "fused_unpermute",
    ):
        if getattr(moe_utils, name, None) is None:
            monkeypatch.setattr(moe_utils, name, object(), raising=False)


# ---------------------------------------------------------------------------------------------
# CPU: configs and pure helpers
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "width, precision",
    [
        (width, precision)
        for width in ("small", "mini", "pro-proxy")
        for precision in ("bf16", "mxfp8")
    ],
)
def test_spike_configs_construct_and_pass_the_static_budget_branch(te_gates, width, precision):
    """T1: the spike configs construct and take the M0.4 branch of the validator; without the
    opt-in the existing assert fires; with the paged stash the config raises."""
    # pylint: disable=import-outside-toplevel
    import dataclasses

    from megatron.core.transformer.cuda_graph_config import validate_moe_cuda_graph_support

    del te_gates
    config = pc.spike_config(width, ep=4, precision=precision, seq=4096)
    assert config.num_layers == 4 and pc.pattern_of(config) == "CECE"
    assert config.num_moe_experts == pc.WIDTHS[width]["experts_per_rank"] * 4
    assert config.moe_router_padding_for_quantization == (precision == "mxfp8")
    assert (config.activation_func_clamp_value is not None) == pc.clamp_supported()
    assert config.cuda_graph_impl == "transformer_engine" and config.cuda_graph_modules == []
    assert config.moe_cuda_graph_static_budget_without_paged_stash and not config.moe_paged_stash
    assert config.use_transformer_engine_op_fuser and config.moe_use_grouped_tensor
    assert (
        config.dsa_kernel_backend == "none"
        and config.hidden_size == pc.WIDTHS[width]["hidden_size"]
    )
    validate_moe_cuda_graph_support(config)
    twin = pc.spike_config(width, ep=4, precision=precision, seq=4096, graphs=False)
    assert twin.cuda_graph_impl == "none"
    config.moe_cuda_graph_static_budget_without_paged_stash = False
    with pytest.raises(AssertionError) as error:
        validate_moe_cuda_graph_support(config)
    assert str(error.value) == _EXISTING_ASSERT
    config.moe_cuda_graph_static_budget_without_paged_stash = True
    with pytest.raises(ValueError, match="moe_cuda_graph_static_budget_without_paged_stash"):
        dataclasses.replace(config, moe_paged_stash=True)


@pytest.mark.parametrize("kind, pattern", [("CSA", "CECE"), ("HCA", "HEHE"), ("W", "WEWE")])
def test_spike_config_kinds_and_widths(te_gates, kind, pattern):
    """Every attention kind maps to its pattern and ratios; the reduced-expert densities."""
    del te_gates
    config = pc.spike_config("mini", ep=8, kind=kind, experts_per_rank=8, factor=1.5)
    assert pc.pattern_of(config) == pattern == pc.layer_pattern(kind)
    assert config.csa_compress_ratios == pc.compress_ratios(kind)
    assert config.num_moe_experts == 64 and config.moe_expert_rank_capacity_factor == 1.5
    dense = pc.spike_config("pro-proxy", ep=4, experts_per_rank=3, indexer_topk=256)
    assert dense.num_moe_experts == 12 and dense.dsa_indexer_topk == 256
    attention = pc.spike_config(
        "pro-proxy", ep=1, kind=kind, pairs=1, graphs=False, attention_only=True
    )
    assert attention.num_moe_experts is None and pc.pattern_of(attention) == pattern[0]


def test_pure_helpers():
    """Budget rule, capture labels, poison coverage, analytic fused-core bytes, budget bytes."""
    assert pc.static_budget_factor(0.3) == 1.2
    assert pc.static_budget_factor(1.5) == 1.65
    assert pc.static_budget_factor(1.0909) == 1.2  # 1.19999 rounds up to the 1.2 floor
    assert pc.static_budget_factor(1.2) == 1.35  # 1.32 -> next 0.05 step
    assert pc.capture_labels([1, -1], [["P0", "P1"]]) == [
        "fwd capture P0",
        "fwd capture P1",
        "bwd capture P1",
        "bwd capture P0",
    ]
    assert pc.capture_labels([2, -2, 1, -1], [["P0"], ["P1"]]) == [
        "fwd capture P1",
        "bwd capture P1",
        "fwd capture P0",
        "bwd capture P0",
    ]
    # overlapping poison ranges count once; a range outside every block counts nothing
    assert pc.poison_coverage([(0, 100), (200, 50)], [(0, 60), (40, 30), (210, 10), (500, 5)]) == (
        80,
        150,
    )
    assert pc.poison_coverage([], [(0, 10)]) == (0, 0)

    class _Config:  # pylint: disable=too-few-public-methods
        csa_compress_ratios = [4, 0]
        num_attention_heads, v_head_dim, csa_window_size = 2, 8, 4
        dsa_indexer_topk, dsa_indexer_n_heads, dsa_indexer_head_dim = 3, 2, 4
        hidden_size, moe_ffn_hidden_size, gated_linear_unit = 16, 8, True

    core = pc.fused_core_saved_bytes(_Config, 16)
    # q / output bf16 [16, 2, 8]; kv_full bf16 [16 + 4, 8]; lse fp32 [16, 2]; top-k int32
    # [16, 4 + 3]; indexer q bf16 [16, 2, 4], k bf16 [4, 4], weights fp32 [16, 2]
    assert core == dict(
        q=512,
        output=512,
        kv_full=320,
        lse=128,
        topk_indices=448,
        indexer_q=256,
        indexer_k=32,
        indexer_weights=128,
        total=2336,
    )
    _Config.csa_compress_ratios = [128, 0]
    assert pc.fused_core_saved_bytes(_Config, 256)["topk_indices"] == 4 * 256 * (4 + 2)
    assert pc.budget_bytes(_Config, [None, 512, 256]) == dict(
        budget_rows=512,
        dispatched_hidden_bytes=2 * 512 * 16,
        fc1_output_bytes=2 * 512 * 16,
        fc2_input_bytes=2 * 512 * 8,
    )
    raw = dict(total=1000, attention_core=300, indexer_loss_reference=100, mhc=50)
    assert pc.representative_saved_bytes(raw, dict(kv_full=10, total=40), pairs=2) == dict(
        total=680, replaced=400, fused_core=80
    )
    assert pc.representative_saved_bytes(
        raw, dict(attention_core=30, indexer_loss_reference=5, total=99), pairs=1
    ) == dict(total=635, replaced=400, fused_core=35)
    # a J5 record without its core: its total holds projections, mHC and the indexer too
    with pytest.raises(ValueError, match="attention_core"):
        pc.representative_saved_bytes(raw, dict(mhc=7, total=99), pairs=1)
    # sizing counts permuted rows with every expert's alignment padding
    assert pc.padded_rows(torch.tensor([1, 256, 257, 0]), 256) == 256 + 256 + 512
    assert pc.padded_rows(torch.tensor([1, 256, 257]), 0) == 514
    args = pc.parse_args(["--width", "small", "--factor", "1.5", "--arms", "pair"])
    assert args.factor == 1.5 and pc.parse_list(args.arms) == ["pair"]
    assert pc.parse_args([]).factor == "auto"


def test_deterministic_algorithms_sets_and_restores_torch_state():
    """The spike's determinism context: torch's deterministic algorithms on (warn-only) without
    the uninitialized-memory fill, the --deterministic-mode environment filled into the given
    mapping only, and the previous torch state restored on exit."""
    import torch.utils.deterministic  # pylint: disable=import-outside-toplevel

    before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.utils.deterministic.fill_uninitialized_memory,
    )
    env = {}
    with pc.deterministic_algorithms(env=env):
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert not torch.utils.deterministic.fill_uninitialized_memory
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and env["NCCL_ALGO"] == "Ring"
    assert env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"
    after = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.utils.deterministic.fill_uninitialized_memory,
    )
    assert after == before
    with pytest.raises(AssertionError):  # a nondeterministic setting is refused, not overridden
        with pc.deterministic_algorithms(env={"NVTE_ALLOW_NONDETERMINISTIC_ALGO": "1"}):
            pass


class _Noisy(torch.nn.Module):
    """Scales by fresh noise on every call (a nondeterministic component)."""

    def forward(self, x):  # pylint: disable=missing-function-docstring
        return x * (1 + torch.randn_like(x).detach() * 1e-3)


def test_first_divergence_names_the_nondeterministic_submodule():
    """The determinism probe's tracer: the first linear agrees across runs, the noisy
    submodule and the linear after it differ in forward. With a fixed linear loss the output
    gradients of the last two submodules agree and only the first linear's output gradient
    (which flows back through the noise) differs."""
    torch.manual_seed(0)
    layer = torch.nn.Sequential(torch.nn.Linear(4, 4), _Noisy(), torch.nn.Linear(4, 4))
    tracer = pc._FirstDivergence(layer)  # pylint: disable=protected-access
    x, weight = torch.randn(3, 4), torch.randn(3, 4)
    try:
        for _ in range(2):
            leaf = x.clone().requires_grad_(True)
            (layer(leaf) * weight).sum().backward()
            tracer.next_run()
    finally:
        tracer.remove()
    assert [name for _order, name, _diff in sorted(tracer.forward)] == ["1", "2"], tracer.forward
    assert [name for _order, name, _diff in tracer.backward] == ["0"], tracer.backward


def test_capture_step_tracker_names_each_capture(monkeypatch):
    """The step a capture failure is attributed to: warm-up, then TE's capture sequence."""
    monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_begin", lambda graph, *a, **k: None)
    stub = torch.cuda.CUDAGraph.capture_begin
    labels = pc.capture_labels([1, -1], [["P0", "P1"]])
    with pc.capture_step_tracker(labels) as state:
        assert state["step"] == "warm-up"
        seen = []
        for _ in range(5):
            torch.cuda.CUDAGraph.capture_begin(object())
            seen.append(state["step"])
    assert seen == labels + ["extra capture 4"]
    assert torch.cuda.CUDAGraph.capture_begin is stub


def test_class_calls_are_restored_after_a_failed_capture():
    """TE wraps the callables' class ``__call__`` during capture and restores it only on
    success; the spike restores both an own and an inherited ``__call__``."""

    class _Own(torch.nn.Module):
        def __call__(self, *args, **kwargs):
            return "own"

    class _Inherited(torch.nn.Module):
        pass

    own = _Own.__dict__["__call__"]
    with pytest.raises(RuntimeError):
        with pc._restore_class_calls([_Own, _Inherited, _Own]):  # pylint: disable=W0212
            _Own.__call__ = lambda self: "wrapped"
            _Inherited.__call__ = lambda self: "wrapped"
            raise RuntimeError("capture failed")
    assert _Own.__dict__["__call__"] is own and "__call__" not in _Inherited.__dict__


def test_saved_tensor_bytes_attributes_unique_storages(monkeypatch):
    """The saved-bytes attribution: unique storages only, parameters and the input excluded,
    the innermost component tag wins (here two tagged inner modules and untagged rest)."""

    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.linspace(0.5, 1.5, 8))

        def forward(self, hidden):
            # mul saves its input (the stack input: excluded; later the previous layer's output:
            # already seen) and the weight (a parameter: excluded); sigmoid saves its output.
            return torch.sigmoid(hidden * self.weight)

    class _Layer(torch.nn.Module):
        def __init__(self, layer_number):
            super().__init__()
            self.inner = _Inner()
            # a BF16 layer: ``call_layer`` enters a no-op quantization context
            self.config = SimpleNamespace(fp8=None, fp4=None)
            self.layer_number = layer_number

        def forward(self, hidden_states):
            return torch.exp(self.inner(hidden_states)), None  # exp saves its output

    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([_Layer(1), _Layer(2)])
    layers = list(model.decoder.layers)
    monkeypatch.setattr(
        pc,
        "_module_tags",
        lambda _model: [(layers[0].inner, "inner0"), (layers[1].inner, "inner1")],
    )
    saved = pc.saved_tensor_bytes(model, torch.randn(4, 1, 8), kwargs_per_layer=[{}, {}])
    assert saved == dict(inner0=128, inner1=128, other=256, total=512)


class _Tap(torch.autograd.Function):
    """Identity that logs ``bwd <name>`` when the gradient reaches it."""

    @staticmethod
    def forward(ctx, tensor, log, name):  # pylint: disable=arguments-differ
        ctx.log, ctx.name = log, name
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=arguments-differ
        ctx.log.append(f"bwd {ctx.name}")
        return grad, None, None


def _logging_side(log):
    """Two deterministic CPU 'pairs' that log their forwards (and grad mode) and backwards."""
    weights = [torch.linspace(0.5, 1.5, 8), torch.linspace(-1.0, 1.0, 8)]

    def pair(index):
        def run(hidden):
            log.append(f"fwd P{index}" + ("" if torch.is_grad_enabled() else " (no grad)"))
            output = torch.tanh(hidden * weights[index]) + hidden
            return _Tap.apply(output, log, f"P{index}")

        return run

    return pc.PairSide(name="cpu", model=torch.nn.Module(), parts=[[pair(0)], [pair(1)]])


class _LoggingPoison:  # pylint: disable=too-few-public-methods
    def __init__(self, log):
        self.log = log

    def replay(self):
        """Log the poison replay."""
        self.log.append("poison")


EXPECTED_SEQUENCE = {
    "block": ["poison", "fwd P0", "fwd P1", "bwd P1", "bwd P0", "poison"],
    "sweep": [
        "poison",
        "fwd P0 (no grad)",
        "fwd P1 (no grad)",
        "poison",
        "fwd P0",
        "fwd P1",
        "bwd P1",
        "bwd P0",
        "poison",
    ],
    "interleaved": ["poison", "fwd P0", "other", "fwd P1", "bwd P1", "other", "bwd P0", "poison"],
    "nested": [
        "poison",
        "fwd P0 (no grad)",
        "fwd P1 (no grad)",
        "poison",
        "fwd P1",
        "bwd P1",
        "poison",
        "fwd P0",
        "bwd P0",
        "poison",
    ],
    "nested-asc": ["poison", "fwd P0", "bwd P0", "poison", "fwd P1", "bwd P1", "poison"],
    "twice": ["poison", "fwd P0", "fwd P0", "bwd P0", "poison"],
}


@pytest.mark.parametrize("copy_out", [True, False], ids=["copy-out", "no-copy"])
@pytest.mark.parametrize("order", sorted(EXPECTED_SEQUENCE))
def test_replay_orders_follow_the_design_sequence(order, copy_out):
    """Each order issues its forwards, backwards, 'other work' and poison replays (between
    items) in the design-table sequence, and the gated orders reproduce the block result."""
    log = []
    side = _logging_side(log)
    generator = torch.Generator().manual_seed(3)
    inputs = {name: torch.randn(4, 1, 8, generator=generator) for name in ("x0", "g", "xb", "g0")}
    reference = pc.execute_order("block", _logging_side([]), inputs)
    log.clear()
    result = pc.execute_order(
        order,
        side,
        inputs,
        poison=_LoggingPoison(log),
        copy_out=copy_out,
        other_work=lambda: log.append("other"),
        upstream_dx1=reference["dx1"],
    )
    assert log == EXPECTED_SEQUENCE[order]
    if order in pc.GATED_ORDERS:
        snapshot = pc.Snapshot(reference, side.model)
        compared = snapshot.compare(result, side.model, grads=False)
        assert compared["bitwise"], compared
    if order == "nested-asc":
        assert torch.equal(result["y1"], reference["y1"])
    if order == "twice":  # the second forward never changes these CPU 'pairs' (no static slots)
        assert torch.equal(result["dx0"], reference["dx0"])
    if order in ("sweep", "nested"):
        prefix = "sweep" if order == "sweep" else "rebuild"
        assert torch.equal(result[f"{prefix}_y1"], reference["y1"])


def _no_grad_broken_side():
    """CPU 'pairs' whose no-grad pair-1 replay returns NaN (a broken sweep / rebuild)."""
    weights = [torch.linspace(0.5, 1.5, 8), torch.linspace(-1.0, 1.0, 8)]

    def pair(index):
        def run(hidden):
            if index == 1 and not torch.is_grad_enabled():
                return torch.full_like(hidden, float("nan"))
            return torch.tanh(hidden * weights[index]) + hidden

        return run

    return pc.PairSide(name="cpu", model=torch.nn.Module(), parts=[[pair(0)], [pair(1)]])


@pytest.mark.parametrize("order", ["sweep", "nested"])
def test_no_grad_replays_are_part_of_the_bitwise_verdict(order):
    """A wrong no-grad sweep / rebuild replay (D35) fails ``Snapshot.compare`` even when the
    grad-mode outputs and input gradients are exact."""
    generator = torch.Generator().manual_seed(3)
    inputs = {name: torch.randn(4, 1, 8, generator=generator) for name in ("x0", "g")}
    reference = pc.execute_order("block", _logging_side([]), inputs)
    snapshot = pc.Snapshot(reference, torch.nn.Module())
    result = pc.execute_order(order, _no_grad_broken_side(), inputs)
    compared = snapshot.compare(result, torch.nn.Module(), grads=False)
    prefix = "sweep" if order == "sweep" else "rebuild"
    assert all(compared[f"bitwise_{key}"] for key in ("y0", "y1", "dx1", "dx0"))
    assert compared[f"bitwise_{prefix}_y0"]
    assert not compared[f"bitwise_{prefix}_y1"]
    assert not compared["bitwise"] and compared["mismatches"] == [f"{prefix}_y1"]


def test_block_replay_of_a_nested_capture_is_not_gated():
    """DV-sp-2: a block-order replay of the nested capture is an open question (reported);
    design 5.6: arm A's graph-pool poisoning is report-only, arm B's is gated."""
    assert not pc.is_gated("nested", "block")
    assert pc.is_gated("block", "block") and pc.is_gated("nested", "nested")
    assert pc.is_gated("helper", "sweep") and not pc.is_gated("block", "twice")
    assert not pc.is_gated("helper", "block", poisoned=True)
    assert pc.is_gated("block", "block", poisoned=True)


def _grads(values):
    return {name: torch.tensor(value) for name, value in values.items()}


def test_p5_fails_when_the_replay_ignores_the_loss_scale():
    """P5 (review B1): a replay that bakes the scale gives g0 = g1 = g2 (linearity error 0);
    the verdict fails it because the aux component is zero while eager's is not."""
    eager = {scale: _grads({"router.weight": [1.0 + 0.5 * scale, -2.0]}) for scale in (0, 1, 2)}
    eager = {float(scale): grads for scale, grads in eager.items()}
    baked = {scale: _grads({"router.weight": [1.0, -2.0]}) for scale in eager}
    verdict = pc.aux_scale_verdict(baked, eager)
    assert verdict["aux_component_max"] == 0.0 and verdict["eager_aux_component_max"] == 0.5
    assert pc.aux_scale_ok(verdict) is False
    assert pc.aux_scale_ok(pc.aux_scale_verdict(eager, eager)) is True
    # a component that reaches the replay but at the wrong size (e.g. a stale scale tensor)
    wrong = {scale: _grads({"router.weight": [1.0 + 0.25 * scale, -2.0]}) for scale in eager}
    verdict = pc.aux_scale_verdict(wrong, eager)
    assert verdict["aux_component_rel_err"] == 0.0 and verdict["aux_vs_eager_rel_err"] == 0.5
    assert pc.aux_scale_ok(verdict) is False
    # eager itself without an aux component: the probe cannot decide
    flat = {scale: _grads({"router.weight": [1.0]}) for scale in eager}
    assert pc.aux_scale_ok(pc.aux_scale_verdict(flat, flat)) is None


def test_failure_record_reports_the_innermost_cause():
    """Review B3: ``torch.cuda.graph.__exit__`` raises from ``capture_end`` while the failing
    op's exception is only its ``__context__``; the record names the inner one as the cause."""

    def failing_op():
        raise RuntimeError("operation not permitted when stream is capturing")

    try:
        try:
            failing_op()
        finally:
            raise RuntimeError(
                "CUDA error: operation failed due to a previous error during capture"
            )
    except RuntimeError as error:
        record = pc.failure_record(error, "fwd capture P0")
    assert record["message"].startswith("operation not permitted")
    assert any("failing_op" in frame for frame in record["frames"])
    assert record["raised"]["message"].startswith("CUDA error: operation failed")
    assert [level["message"][:9] for level in record["chain"]] == ["CUDA erro", "operation"]
    assert record["step"] == "fwd capture P0"


@pytest.fixture
def one_rank(monkeypatch):
    """The helpers' view of a process without a process group (rank 0 of 1), whatever world
    the test file runs in: under torchrun the module fixture initializes model parallel over
    WORLD_SIZE ranks, and the rank-aggregating helpers would then gather every rank's values."""
    monkeypatch.setattr(pc.dist, "is_initialized", lambda: False)


def test_failures_are_recorded_by_their_own_rank(tmp_path, capsys, one_rank):
    """Review B4: a failure is printed and written to the failing rank's own JSONL before any
    collective; without a process group the agreement is this rank's record alone."""
    out = tmp_path / "spike.jsonl"
    writer = pc.Writer(str(out), dict(spike="j"))
    error = dict(type="RuntimeError", message="boom", cuda_error=None)
    assert writer.agree("capture pair block", error) == [error]
    local = [json.loads(line) for line in (tmp_path / "spike.r0.jsonl").read_text().splitlines()]
    assert local[0]["record"] == "failure" and local[0]["error"]["message"] == "boom"
    assert '"step": "capture pair block"' in capsys.readouterr().out
    assert pc.errors_by_rank([None, error]) == {"1": error}
    assert pc.first_error([None, error, None]) is error
    pc.stop_on_cuda_error([None, error])
    with pytest.raises(SystemExit, match=r"ranks \[1\]"):
        pc.stop_on_cuda_error([None, dict(error, cuda_error="illegal address")])


def test_agreement_goes_through_the_store_and_times_out(monkeypatch):
    """Review B4: the step agreement exchanges every rank's record through the c10d store (no
    NCCL collective a failed rank could mismatch) and stops the job when a rank never comes."""
    store = torch.distributed.HashStore()
    monkeypatch.setattr(pc.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(pc, "_world", lambda: 2)
    monkeypatch.setattr(pc, "_rank", lambda: 0)
    monkeypatch.setattr(pc.dist.distributed_c10d, "_get_default_store", lambda: store)
    monkeypatch.setattr(pc, "_AGREE_STEPS", itertools.count(7))
    other = dict(type="AssertionError", message="rank 1 failed", cuda_error=None)
    store.set("pair_capture/agree/7/1", json.dumps(dict(step="replay block", error=other)))
    assert pc.agree_records("replay block", None) == [None, other]
    store.set("pair_capture/agree/8/1", json.dumps(dict(step="replay nested", error=None)))
    with pytest.raises(SystemExit, match="disagree"):
        pc.agree_records("replay block", None)
    with pytest.raises(SystemExit, match="not every rank reached"):
        pc.agree_records("replay sweep", None, timeout_s=0.2)


def test_timing_aggregation_keeps_flags_and_numbers(one_rank):
    """Host-time validity flags survive the over-ranks aggregation (review N3)."""
    record = pc._max_timing(  # pylint: disable=protected-access
        dict(
            host_ms_fwd_P0=dict(median=1.0, p90=2.0, n=3),
            host_window_gpu_busy_P0=True,
            host_window_sleep_ms_P0=400.0,
        )
    )
    assert record["host_ms_fwd_P0"]["median_max_over_ranks"] == 1.0
    assert record["host_window_gpu_busy_P0"] is True
    assert record["host_window_sleep_ms_P0"] == 400.0
    assert record["host_ms_fwd_P0"]["median_by_rank"] == [1.0]
    assert record["host_ms_fwd_P0"]["median_max_unprofiled"] == 1.0
    assert record["host_ms_fwd_P0"]["profiled_ranks"] == []


def test_timing_aggregation_excludes_the_nsys_rank(one_rank):
    """The host-time gate number comes from ranks not under nsys (review N2): with every rank
    profiled there is no unprofiled number; ``--nsys-ranks`` decides which ranks count."""
    record = pc._max_timing(  # pylint: disable=protected-access
        dict(host_ms_fwd_P0=dict(median=1.0, p90=2.0, n=3)), profiled=True
    )
    assert record["host_ms_fwd_P0"]["median_max_over_ranks"] == 1.0
    assert record["host_ms_fwd_P0"]["median_max_unprofiled"] is None
    assert record["host_ms_fwd_P0"]["profiled_ranks"] == [0]
    profiled = pc._profiled  # pylint: disable=protected-access
    assert profiled(pc.parse_args(["--nsys"]))
    assert not profiled(pc.parse_args([]))
    assert not profiled(pc.parse_args(["--nsys", "--nsys-ranks", "1 2 3"]))


class _FakeHybridEPConfig:  # pylint: disable=too-few-public-methods
    def __init__(self, tokens):
        self.max_num_of_tokens_per_rank = tokens
        self.hidden_dim = 64


class _FakeHybridEPBuffer:  # pylint: disable=too-few-public-methods
    def __init__(self, tokens):
        self.config = _FakeHybridEPConfig(tokens)
        self.rdma_buffer = torch.empty(tokens * 4, dtype=torch.uint8)
        self.group = torch.nn.Module()  # a module attribute is not descended into


def test_hybridep_buffer_state_sees_regrowth_in_place(monkeypatch):
    """The buffer check sees a capacity change inside the same object (runtime regrowth), not
    only a replaced object."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer.moe import fused_a2a

    buffer = _FakeHybridEPBuffer(128)
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", buffer, raising=False)
    before = pc.hybridep_buffer_state()
    assert before["fields"]["config.max_num_of_tokens_per_rank"] == 128
    assert pc.hybridep_buffer_changes(before, pc.hybridep_buffer_state()) == []
    buffer.config.max_num_of_tokens_per_rank = 256
    buffer.rdma_buffer = torch.empty(256 * 4, dtype=torch.uint8)
    assert pc.hybridep_buffer_changes(before, pc.hybridep_buffer_state()) == [
        "fields.config.max_num_of_tokens_per_rank",
        "fields.rdma_buffer.nbytes",
    ]
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", _FakeHybridEPBuffer(128), raising=False)
    assert pc.hybridep_buffer_changes(before, pc.hybridep_buffer_state()) == ["id"]
    assert pc.hybridep_buffer_changes(None, None) == []
    assert pc.hybridep_buffer_changes(None, before) == ["created"]


def test_layer_quant_context_follows_hybrid_stack(monkeypatch):
    """BF16 layers run without a quantization context (a no-op); FP8 / FP4 configs take the
    same per-layer contexts HybridStack enters."""
    # pylint: disable=import-outside-toplevel
    from megatron.core import fp4_utils, fp8_utils
    from megatron.core.enums import Fp8Recipe

    assert isinstance(
        pc.layer_quant_context(SimpleNamespace(fp8=None, fp4=None), 1), contextlib.nullcontext
    )
    calls = []
    monkeypatch.setattr(
        fp8_utils, "get_fp8_context", lambda config, layer_no=-1: calls.append(("fp8", layer_no))
    )
    monkeypatch.setattr(
        fp4_utils, "get_fp4_context", lambda config, layer_no=-1: calls.append(("fp4", layer_no))
    )
    pc.layer_quant_context(SimpleNamespace(fp8="e4m3", fp8_recipe=Fp8Recipe.mxfp8, fp4=None), 3)
    pc.layer_quant_context(SimpleNamespace(fp8="e4m3", fp8_recipe=Fp8Recipe.delayed, fp4=None), 3)
    pc.layer_quant_context(SimpleNamespace(fp8=None, fp4="nvfp4"), 2)
    assert calls == [("fp8", 2), ("fp8", -1), ("fp4", 1)]


def test_poison_between_forward_and_backward_is_placed_after_pair0_forward():
    """Probe P6 replays the poison graph between pair 0's forward and its backward."""
    log = []
    generator = torch.Generator().manual_seed(4)
    inputs = {name: torch.randn(4, 1, 8, generator=generator) for name in ("x0", "g")}
    pc.execute_order(
        "block", _logging_side(log), inputs, poison=_LoggingPoison(log), poison_where="fwd-bwd"
    )
    assert log == ["fwd P0", "poison", "fwd P1", "bwd P1", "bwd P0"]


# ---------------------------------------------------------------------------------------------
# GPU: Blackwell + HybridEP + the TE op fuser
# ---------------------------------------------------------------------------------------------


class _Spike:
    """The small-width graph model, its eager twin, inputs and static keyword inputs."""

    def __init__(self, precision: str, factor: float = 2.0):
        # pylint: disable=import-outside-toplevel
        from megatron.core.process_groups_config import ProcessGroupCollection

        self.config = pc.spike_config(
            "small", ep=WORLD, precision=precision, factor=factor, seq=SEQ
        )
        # the eager twin: the graph config, never captured (review N1)
        twin_config = pc.spike_config(
            "small", ep=WORLD, precision=precision, factor=factor, seq=SEQ
        )
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.model, self.twin = pc.build_pair_models(
            self.config,
            pg_collection=self.pg_collection,
            twin_config=twin_config,
            vocab_size=pc.WIDTHS["small"]["vocab_size"],
            max_sequence_length=SEQ,
        )
        pc._prepare([self.model, self.twin], self.pg_collection)  # pylint: disable=W0212
        self.inputs = pc.make_inputs(self.config, SEQ, seed=1234 + dist.get_rank())
        self.kwargs = pc.static_kwargs(self.model, SEQ)
        # cuda_graph_warmup_steps = 1: one eager step on the graph model before any capture
        pc.execute_order("block", pc.layer_side("warm-up", self.model, self.kwargs), self.inputs)
        # reported, not asserted (design 5.4 step 4; the sized factor is the CLI's job)
        print(f"over_budget after the eager warm-up step: {pc.over_budget_any(self.model)}")
        pc.reset_after_capture([self.model, self.twin], self.config)
        self._spread = None

    def eager(self):
        """The eager twin's block order on the graph model's weights (left in its grads)."""
        pc.copy_model_state(self.model, self.twin)
        pc.assert_never_captured(self.twin)
        pc.reset_after_capture([self.twin], self.config)
        return pc.execute_order(
            "block", pc.layer_side("eager", self.twin, self.kwargs), self.inputs
        )

    def eager_spread(self) -> float:
        """T0: the twin twice on the same inputs (0.0 when bitwise self-deterministic)."""
        if self._spread is None:
            first = pc.Snapshot(self.eager(), self.twin)
            compared = first.compare(self.eager(), self.twin)
            self._spread = (
                0.0
                if pc.agree_all(compared["bitwise"])
                else pc.max_over_ranks(compared["max_abs_diff"])
            )
            print(f"T0 eager self-determinism: spread {self._spread}")
        return self._spread

    def assert_matches_eager(self, oracle: "pc.Snapshot", label: str) -> None:
        """Oracle (ii): graph == eager bitwise, or within the eager run-to-run spread."""
        compared = oracle.compare(self.eager(), self.twin)
        spread = self.eager_spread()
        worst = pc.max_over_ranks(compared["max_abs_diff"])
        if spread == 0.0:
            assert pc.agree_all(compared["bitwise"]), (label, compared)
        else:
            assert worst <= spread, (label, worst, spread, compared)
        pc.reset_after_capture([self.twin], self.config)

    def pair_capture(self, order: str):
        """An arm-B capture (asserted successful) and its graphed side."""
        pair_args, pair_kwargs, warmup, fp8_group = pc.pair_sample_inputs(
            self.model, self.config, seq=SEQ, pg_collection=self.pg_collection
        )
        capture = pc.capture_pairs(
            self.model,
            self.config,
            pair_args,
            pair_kwargs,
            precision="mxfp8" if self.config.fp8 else "bf16",
            order=order,
            reuse_buffers=order != "nested-noreuse",
            num_warmup_iters=warmup,
            fp8_group=fp8_group,
        )
        assert capture.ok, capture.error
        # TE's warm-up runs on the helper's constant static inputs (every token to the same
        # experts), so it may overflow: printed, not asserted (an M4a warm-up-input question)
        capture.over_budget = pc.over_budget_any(self.model)
        print(f"pair {order}: over_budget during TE warm-up + capture: {capture.over_budget}")
        pc.reset_after_capture([self.model, self.twin], self.config)
        side = pc.graphed_side(
            f"pair-{order}", self.model, capture.graphed, pair_kwargs, capture.pool
        )
        return capture, side


@pytest.fixture(scope="module")
def bf16_spike():
    """One BF16 small-width spike shared by the GPU cases (captures are per test)."""
    if not _gpu_ready():
        pytest.skip(GPU_REASON)
    spike = _Spike("bf16")
    yield spike
    del spike
    torch.cuda.empty_cache()


def test_ddp_grad_accumulators_are_off_the_legacy_stream(bf16_spike):
    """T1b: no AccumulateGrad node DDP retains runs on the legacy default stream (those nodes
    broke every backward capture in stage-7 round 1: cudaErrorStreamCaptureImplicit). The
    negative control (DDP built on the current, legacy stream) must show such nodes, which
    also proves the census sees a node's own stream."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer.moe.moe_logging import (
        get_moe_metrics_tracker,
        set_moe_metrics_tracker,
    )

    spike = bf16_spike
    census = pc.census_on_side_stream(
        spike.model, pc.layer_side("census", spike.model, spike.kwargs), spike.inputs
    )
    pc.reset_after_capture([spike.model], spike.config)
    print(f"T1b side-stream DDP: {census}")
    assert census["fired"] > 0, census
    assert pc.agree_all(census["on_legacy_stream"] == 0), census
    tracker = get_moe_metrics_tracker()
    try:
        legacy, _ = pc.build_pair_models(
            spike.config,
            pg_collection=spike.pg_collection,
            twin=False,
            vocab_size=pc.WIDTHS["small"]["vocab_size"],
            max_sequence_length=SEQ,
            ddp_stream="current",
        )
        pc._prepare([legacy], spike.pg_collection)  # pylint: disable=W0212
        control = pc.census_on_side_stream(
            legacy, pc.layer_side("control", legacy, pc.static_kwargs(legacy, SEQ)), spike.inputs
        )
        print(f"T1b legacy-stream DDP (negative control): {control}")
        assert pc.agree_all(control["on_legacy_stream"] > 0), control
    finally:
        set_moe_metrics_tracker(tracker)
        legacy = None
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_whole_layer_capture_through_the_te_helper(bf16_spike, precision):
    """T2: arm A captures every mHC wrapper whole through dev's helper (static-budget branch),
    registers no manual hooks (no forward pre-hooks), equals the eager twin, never sets
    ``over_budget`` and keeps the process-global HybridEP buffer."""
    if precision == "mxfp8" and not pc.mxfp8_available():
        pytest.skip("MXFP8 is not supported on this device / TE build")
    spike = bf16_spike if precision == "bf16" else _Spike("mxfp8")
    buffer_state = pc.hybridep_buffer_state()
    assert buffer_state is not None  # the warm-up step allocated it
    watch = {}
    with pc.hybridep_dispatch_watch(watch):
        capture = pc.capture_with_helper(
            spike.model, spike.config, seq=SEQ, pg_collection=spike.pg_collection
        )
    assert capture.ok, capture.error
    try:
        layers = pc.decoder_layers(spike.model)
        assert all(len(layer.cuda_graphs) == 1 for layer in layers)
        assert not any(layer.cuda_graph_manual_hooks for layer in layers)
        side = pc.layer_side("helper", spike.model, capture.kwargs_per_layer)
        # constant static inputs in TE's warm-up: printed, not asserted (see _Spike.pair_capture)
        print(
            f"T2 {precision}: over_budget during TE warm-up + capture: "
            f"{pc.over_budget_any(spike.model)}"
        )
        pc.reset_after_capture([spike.model], spike.config)
        with pc.hybridep_dispatch_watch(watch):
            oracle = pc.Snapshot(pc.execute_order("block", side, spike.inputs), spike.model)
        assert not pc.over_budget_any(spike.model)
        spike.assert_matches_eager(oracle, f"helper {precision}")
        changes = pc.hybridep_buffer_changes(buffer_state, pc.hybridep_buffer_state())
        assert pc.agree_all(not changes), changes
        # reported, not asserted: other lazy non-PyTorch allocations land in the same delta
        print(f"T2 {precision}: HybridEP dispatch watch {watch}")
    finally:
        capture.delete()
        pc.reset_after_capture([spike.model, spike.twin], spike.config)


@pytest.mark.parametrize("order", ["block", "nested", "sweep", "interleaved"])
def test_pair_graph_replay_orders(bf16_spike, order):
    """T3: one graph per pair (B-block; also B-nested for ``nested``): every order equals a fresh
    replay of the same capture in its capture order (``NATIVE_ORDER``) bitwise, no-grad sweep /
    rebuild outputs included, without and with the graph pool NaN-poisoned between items (the
    poison covering the pool's free blocks); router counts scale with the forward replays; the
    B-block and B-nested oracles equal the eager twin."""
    spike = bf16_spike
    other = pc.layer_side("other", spike.twin, spike.kwargs)
    other_inputs = dict(x0=spike.inputs["x_other"], g=spike.inputs["g_other"])
    for capture_name in ("block", "nested") if order == "nested" else ("block",):
        capture, side = spike.pair_capture(capture_name)
        poison = pc.poison_graph_pool(capture.pool)
        try:
            print(f"T3 {capture_name}: {poison.record()} {capture.pool_record}")
            # the poisoned leg must poison something: the pool's free blocks, nearly all of them
            assert pc.agree_all(poison.inactive > 0), poison.record()
            assert pc.agree_all(poison.covered >= 0.9 * poison.inactive), poison.record()
            native = pc.NATIVE_ORDER[capture_name]
            oracle = pc.Snapshot(
                pc.execute_order(native, side, spike.inputs),
                spike.model,
                forwards=pc.ORDER_FORWARDS[native],
            )
            for poisoned in (None, poison):
                pc.reset_after_capture([spike.model], spike.config)
                result = pc.execute_order(
                    order,
                    side,
                    spike.inputs,
                    poison=poisoned,
                    other_work=lambda: pc.execute_order("block", other, other_inputs),
                )
                compared = oracle.compare(result, spike.model, forwards=pc.ORDER_FORWARDS[order])
                label = (capture_name, order, poisoned is not None)
                # ``bitwise`` includes the no-grad sweep / rebuild outputs (D35)
                if order in ("sweep", "nested"):
                    prefix = "sweep" if order == "sweep" else "rebuild"
                    assert f"bitwise_{prefix}_y1" in compared, (label, compared)
                assert pc.agree_all(compared["bitwise"]), (label, compared)
                assert pc.agree_all(compared["router_counts_match"]), (label, compared)
            assert not pc.over_budget_any(spike.model)
            # every capture's oracle (the nested one too) equals the eager twin: a wrong but
            # repeatable nested capture (TE's input / kwargs reuse) fails here
            if order in ("block", "nested"):
                spike.assert_matches_eager(oracle, f"pair {capture_name}")
        finally:
            poison.reset()
            capture.delete()
            pc.reset_after_capture([spike.model, spike.twin], spike.config)


def test_pair_graph_outputs_alias_the_pool(bf16_spike):
    """T4 (P4): a second forward replay of pair 0 overwrites the tensor the first replay
    returned (TE returns its static output); a clone taken before is the first result. P4 (b)
    (pair 0's output after a pair-1 replay) is printed, not asserted."""
    spike = bf16_spike
    capture, side = spike.pair_capture("block")
    try:
        with torch.no_grad():
            first = side.pair(0, spike.inputs["x0"])
            kept = first.clone()
            side.pair(0, spike.inputs["xb"])
            assert pc.agree_all(not torch.equal(first, kept))
            again = side.pair(0, spike.inputs["x0"]).clone()
            assert torch.equal(again, kept)
            first = side.pair(0, spike.inputs["x0"])
            kept = first.clone()
            side.pair(1, kept)
            print(
                f"T4 P4(b): pair-0 output changed by a pair-1 replay: {not torch.equal(first, kept)}"
            )
    finally:
        capture.delete()
        pc.reset_after_capture([spike.model, spike.twin], spike.config)


def test_over_budget_is_reported_after_replay(bf16_spike):
    """T5: with a 0.05 budget a replayed dispatch overflows: ``over_budget_any`` is True on
    every rank (all-reduced) and the next replay still runs (no hang)."""
    spike = bf16_spike
    pc.apply_factor([spike.model, spike.twin], 0.05)
    try:
        capture, side = spike.pair_capture("block")
        try:
            pc.execute_order("block", side, spike.inputs)
            assert pc.over_budget_any(spike.model)
            pc.execute_order("block", side, spike.inputs)
            torch.cuda.synchronize()
            assert pc.over_budget_any(spike.model)
        finally:
            capture.delete()
    finally:
        pc.apply_factor([spike.model, spike.twin], 2.0)
        pc.reset_after_capture([spike.model, spike.twin], spike.config)
