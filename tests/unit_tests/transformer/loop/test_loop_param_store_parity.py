# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU parity gates of the loop parameter store under the RSBF executor (M3, scaled to H100).

Runs under torchrun with CUDA at WORLD_SIZE = W in {2, 4, 8} (one EU per rank, the domain = the
world): the DSV4-TINY-like loop model (``'HEHECEHECEHECE/WE'``: DSv4 H / C / W attention with
TE linears, unfused attention, ``dsa_kernel_backend='none'``, 8 experts at EP = W with the
dropless all-to-all dispatcher, one hash layer, expert bias, mHC 4, MTP 1, bf16) with a store on
the shard axis z = W (shard group = the world, replicate groups of one rank; the store does not
need the GTP-remat grid, whose wiring is the ``_ddp`` / ``_training`` tests'). ``min_numel`` is
low enough that the loop adapter, ``linear_o_group_proj`` and the MTP projections are windowed.
Each rank loads its slice of a fixed global batch with prescribed per-sample depths (sample moves,
idle EUs and null items happen) and runs one RSBF step; gradients are read as DDP would hold them
(a stand-in hook accumulates every registered parameter's gradient into an fp32 ``main_grad``,
DDP's arithmetic; the store's tensors reduce into their shards' ``main_grad``), then summed over
the world (store tensors: all-gathered over the shard axis).

Arms per item form (whole round, block stride 2 and 4 = one and two layers, nested stride 2) and
waves 1 / 2: no store (the stage-5 path), store Resident, Resident + sharded grads, Layer window
kept and re-gathered (core, edges and MTP windowed: move E; embedding and head sharded grads:
move M), with ``check_recompute="raise"``, ``poison_released`` and ``check_visits``.

Gates:

* per-sample losses bitwise in every arm (step 1);
* Layer window (kept, re-gathered) == Resident + sharded grads: every gradient bitwise;
* every store arm against the store-free run and against the store Resident: within the fp32
  association bound (normwise relative error <= 1e-5 per tensor; the per-item reduce-scatters
  re-associate the per-step sum), expert gradients and expert-bias token counts bitwise;
* the store reduces each non-Resident unit exactly once per sub-round of every type visiting
  it (real or null) and each Resident unit once per step;
* the fused-wgrad arm (``gradient_accumulation_fusion=True``, no MTP, untied embedding, move M:
  TE writes the wgrad into the bound target with ``overwrite_main_grad`` for single-use units, the
  output layer's Megatron fused linear captures its target at the head forward): the same gates;
* the literal case (D-S6-9): one sample per EU, depth 1, no MTP, ``no_overwrite``: Layer window
  (kept, re-gathered) and Resident + sharded grads == store Resident, every gradient and the
  losses **bitwise**, with fusion off and on;
* the ``DistributedWeight`` binding: Layer window with ``binding="distributed_weight"`` ==
  ``binding="data"`` bitwise; skipped on TE < 2.19 (the CI image has TE 2.16, E1 / E2).
"""

import contextlib
import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.loop.param_store import (
    LoopParamStore,
    LoopParamStoreConfig,
    ResidencyMode,
    StoreModes,
)
from megatron.core.transformer.loop.param_store_binding import (
    distributed_weight_unavailable_reason,
    te_distributed_weight_available,
)
from megatron.core.transformer.loop.program import LoopProgram, PrescribedDepthCriterion
from megatron.core.transformer.loop.rsbf_executor import LoopMicrobatch, LoopRSBFExecutor
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.loop import param_store_utils as psu
from tests.unit_tests.transformer.loop.hybrid_loop_utils import VOCAB

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
EXPERTS = 8
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or WORLD < 2 or EXPERTS % WORLD,
    reason="the store parity tests need CUDA and 2, 4 or 8 ranks (EP = W over 8 experts)",
)

PATTERN = "HEHECEHECEHECE"  # prelude 'HE', core 'HECEHECE' (L = 8), coda 'HECE'
MTP_PATTERN = "WE"
_RATIOS = {"W": 0, "C": 4, "H": 128, "E": 0}
SEQ = 256
R_MAX = 2
PER_RANK = 2
MIN_NUMEL = 4096

MODES = {
    "resident": "all=r",
    "rsg": "all=rsg",
    "lw": "core=lw,edges=lw,emb_head=rsg",
    "lw_regather": "core=lw,edges=lw,emb_head=rsg,core_regather=on,edge_regather=on",
}
FORMS = {
    "whole_round": {},
    "block2": dict(loop_rsbf_item_form="block", loop_rsbf_block_stride=2),
    "block4": dict(loop_rsbf_item_form="block", loop_rsbf_block_stride=4),
    "nested2": dict(loop_rsbf_item_form="nested", loop_rsbf_block_stride=2),
}


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=WORLD)
    model_parallel_cuda_manual_seed(123)
    try:
        yield
    finally:
        Utils.destroy_model_parallel()


@pytest.fixture(scope="module")
def store_groups():
    """Shard axis = the world (z = W), replicate groups of one rank; created once."""
    return psu.make_store_groups(WORLD, WORLD)


@pytest.fixture
def deterministic(monkeypatch):
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@contextlib.contextmanager
def _hadamard_fallback():
    from unittest.mock import patch

    try:
        from fast_hadamard_transform import hadamard_transform  # noqa: F401

        yield
    except ImportError:
        with patch(
            "megatron.core.transformer.experimental_attention_variant.csa.rotate_activation",
            lambda x: x * (x.size(-1) ** -0.5),
        ):
            yield


def _fused_wgrad_available():
    from megatron.core.tensor_parallel import layers as tensor_parallel_layers

    try:
        return bool(tensor_parallel_layers._grad_accum_fusion_available)
    except Exception:  # pylint: disable=broad-except
        return False


# ---------------------------------------------------------------------------------------------
# Model, batch, step
# ---------------------------------------------------------------------------------------------


def _dsv4_tiny(*, mtp=True, **overrides):
    """DSV4-TINY-like at EP = W (``test_loop_rsbf_item_forms._dsv4_tiny``'s fields)."""
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_dsv4_stack_spec
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    symbols = PATTERN + (MTP_PATTERN if mtp else "")
    fields = dict(
        num_layers=len(PATTERN),
        hidden_size=512,
        num_attention_heads=8,
        multi_latent_attention=True,
        q_lora_rank=192,
        kv_lora_rank=64,
        qk_head_dim=16,
        qk_pos_emb_head_dim=8,
        v_head_dim=16,
        rope_type="rope",
        rotary_base=10000,
        rotary_percent=1.0,
        experimental_attention_variant="dsv4_hybrid",
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=32,
        dsa_indexer_loss_coeff=0.01,
        csa_window_size=32,
        csa_compress_ratios=[_RATIOS[symbol] for symbol in symbols],
        num_moe_experts=EXPERTS,
        moe_ffn_hidden_size=256,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=1e-4,
        moe_token_dispatcher_type="alltoall",
        expert_model_parallel_size=WORLD,
        moe_n_hash_layers=1,
        actual_vocab_size=VOCAB,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=5,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        loop_architecture="huginn",
        loop_iterations=R_MAX,
        loop_prelude_sublayers=2,
        loop_coda_sublayers=4,
        deterministic_mode=True,
        attention_backend=AttnBackend.unfused,
        loop_schedule="rsbf",
        calculate_per_token_loss=True,
    )
    if mtp:
        fields.update(mtp_num_layers=1, mtp_loss_scaling_factor=0.1)
    fields.update(overrides)
    config = MLATransformerConfig(**fields)
    torch.manual_seed(1234)
    return HybridModel(
        config=config,
        hybrid_stack_spec=loop_hybrid_dsv4_stack_spec(config),
        vocab_size=VOCAB,
        max_sequence_length=1024,
        hybrid_layer_pattern=PATTERN + (f"/{MTP_PATTERN}" if mtp else ""),
        position_embedding_type="rope",
    ).cuda()


def _loss_func(loss_mask, output):
    """``pretrain_hybrid.loss_func``'s default branch (per-token sum, token count, report)."""
    losses = output.view(-1).float()
    mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * mask)
    num_tokens = mask.sum().clone().detach().to(torch.int)
    return (
        loss,
        num_tokens,
        {"lm loss": torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])},
    )


def _global_batch(total):
    generator = torch.Generator().manual_seed(17)
    tokens = torch.randint(0, VOCAB, (total, SEQ), generator=generator)
    labels = torch.randint(0, VOCAB, (total, SEQ), generator=generator)
    loss_mask = (torch.rand(total, SEQ, generator=generator) > 0.2).float()
    return [
        dict(
            tokens=tokens[i : i + 1].cuda(),
            labels=labels[i : i + 1].cuda(),
            position_ids=torch.arange(SEQ)[None].cuda(),
            loss_mask=loss_mask[i : i + 1].cuda(),
        )
        for i in range(total)
    ]


def _skew_depths(total):
    """EU 0's samples run R_max rounds, the others 1 (moves at round 2, idle EUs, nulls)."""
    return [R_MAX if g % WORLD == 0 else 1 for g in range(total)]


def _attach_main_grads(model):
    """DDP's gradient arithmetic without DDP: every registered parameter that requires grad
    (the store's canonical shards excluded: the store reduces into them) gets a zeroed fp32
    ``main_grad`` and a post-accumulate hook adding ``.grad`` unless a fused consumer already
    wrote it (``grad_added_to_main_grad``), as DDP's hook does."""

    def hook(param):
        fused = getattr(param, "grad_added_to_main_grad", False)
        if param.grad is not None and (not fused or getattr(param, "zero_out_wgrad", False)):
            param.main_grad.add_(param.grad)
        param.grad_added_to_main_grad = False
        param.grad = None

    for name, param in model.named_parameters():
        if "loop_param_store." in name or not param.requires_grad:
            continue
        param.main_grad = torch.zeros(param.shape, dtype=torch.float32, device=param.device)
        param.grad_added_to_main_grad = False
        param.register_post_accumulate_grad_hook(hook)


def _install(model, groups, modes_text, **config):
    config.setdefault("min_numel", MIN_NUMEL)
    store = LoopParamStore.install(model, LoopParamStoreConfig(shard_size=WORLD, **config), groups)
    psu.attach_shard_grads(store)
    store.configure_modes(StoreModes.parse(modes_text))
    return store


def _expert_bias_counts(model):
    return {
        name: module.local_tokens_per_expert.clone()
        for name, module in model.named_modules()
        if getattr(module, "local_tokens_per_expert", None) is not None
    }


def _run(
    groups,
    modes_text,
    *,
    form="whole_round",
    waves=1,
    mtp=True,
    fusion=False,
    per_rank=PER_RANK,
    depths=None,
    store_config=None,
    steps=1,
):
    """One RSBF step of this rank's slice; returns a dict with the per-sample losses, the
    world-summed gradients, the expert-bias counts and the store (None without one).
    ``steps > 1`` repeats the step (gradients accumulate), scaling every canonical shard by
    ``1 + 2^-7`` between steps (an optimizer update the next step's gathers must see)."""
    from megatron.core.process_groups_config import ProcessGroupCollection

    rank = dist.get_rank()
    total = per_rank * WORLD
    batch = _global_batch(total)
    depths = depths or _skew_depths(total)
    overrides = dict(FORMS[form], gradient_accumulation_fusion=fusion)
    with _hadamard_fallback():
        model = _dsv4_tiny(mtp=mtp, **overrides)
        store = None
        if modes_text is not None:
            store_config = dict(store_config or {})
            store_config.setdefault("poison_released", True)
            store_config.setdefault("check_visits", True)
            store = _install(model, groups, modes_text, **store_config)
        _attach_main_grads(model)
        set_moe_metrics_tracker(MoEMetricsTracker())
        program = LoopProgram(
            model, exit_criterion=PrescribedDepthCriterion(dict(enumerate(depths)))
        )
        executor = LoopRSBFExecutor(
            program,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            waves=waves,
            check_recompute="raise",
        )
        microbatches = [
            LoopMicrobatch(
                tokens=sample["tokens"],
                position_ids=sample["position_ids"],
                labels=sample["labels"],
                loss_mask=sample["loss_mask"],
                attention_mask=None,
                padding_mask=None,
                loss_func=_loss_func,
            )
            for g, sample in enumerate(batch)
            if g % WORLD == rank
        ]
        for step in range(steps):
            if step and store is not None:
                with torch.no_grad():
                    for tensor in store.tensors():
                        tensor.shard.mul_(1 + 2**-7)
            reports = executor.run_step(
                lambda iterator, _model: next(iterator),
                iter(microbatches),
                len(microbatches),
                model=model,
            )
        torch.cuda.synchronize()
    keys = [record.key.global_index for record in executor.stats.loss_records]
    losses = {key: report["lm loss"] for key, report in zip(keys, reports)}
    grads = {}
    for name, param in sorted(model.named_parameters(), key=lambda item: item[0]):
        if "loop_param_store." in name or not param.requires_grad:
            continue
        grad = param.main_grad.detach().clone()
        if getattr(param, "allreduce", True):
            dist.all_reduce(grad)  # dense: summed over the domain (experts stay local)
        grads[name] = grad
    if store is not None:
        grads.update(psu.full_store_grads(store, groups))
    return dict(
        losses=losses,
        grads=grads,
        counts=_expert_bias_counts(model),
        store=store,
        nulls=sum(executor.stats.nulls.values()),
        moves=dict(executor.stats.moves),
        subrounds=dict(executor.stats.subrounds),
    )


# ---------------------------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------------------------


def _assert_losses_bitwise(actual, expected, what):
    assert actual["losses"].keys() == expected["losses"].keys(), what
    for key, loss in expected["losses"].items():
        assert torch.equal(actual["losses"][key], loss), (what, key, actual["losses"][key], loss)


def _assert_grads_bitwise(actual, expected, what):
    assert actual["grads"].keys() == expected["grads"].keys(), (
        what,
        actual["grads"].keys() ^ expected["grads"].keys(),
    )
    for name, grad in expected["grads"].items():
        assert torch.equal(actual["grads"][name], grad), (what, name)


def _assert_grads_close(actual, expected, what, bound=1e-5):
    """Per tensor, the normwise relative error of the fp32 gradient (the re-association bound
    of a sum of per-item addends); exactly zero tensors must stay zero."""
    assert actual["grads"].keys() == expected["grads"].keys(), what
    worst = 0.0
    for name, grad in expected["grads"].items():
        reference = grad.float()
        norm = reference.norm()
        difference = (actual["grads"][name].float() - reference).norm()
        if norm == 0:
            assert difference == 0, (what, name)
            continue
        error = (difference / norm).item()
        worst = max(worst, error)
        assert error <= bound, f"{what}: {name} normwise {error:.3e} > {bound:.0e}"
    return worst


def _assert_counts_equal(actual, expected, what):
    assert actual["counts"].keys() == expected["counts"].keys(), what
    for name, value in expected["counts"].items():
        assert torch.equal(actual["counts"][name], value), (what, name)


def _assert_exactly_once(store, mtp):
    """Each non-Resident unit: one reduce per sub-round (real or null) of every type whose
    backward visits it; each Resident unit: one reduce per step."""
    subrounds = {}
    for (kind, _), count in store.stats.subrounds.items():
        subrounds[kind] = subrounds.get(kind, 0) + count
    got = {}
    for (kind, key), count in store.stats.reduces.items():
        got.setdefault(key, {})[kind] = count
    for unit in store.units:
        if store.mode_of(unit) is ResidencyMode.RESIDENT:
            assert unit.key not in got and store.stats.resident_reduces[unit.key] == 1, unit.key
            continue
        if unit.group == "core":
            kinds = ("core_b",)
        elif unit.group == "prelude":
            kinds = ("prelude_b",)
        elif unit.key == "emb_head:embedding":
            kinds = ("prelude_b", "coda") if mtp else ("prelude_b",)
        else:
            kinds = ("coda",)
        assert got.get(unit.key) == {kind: subrounds[kind] for kind in kinds}, (
            unit.key,
            got.get(unit.key),
        )


def _windowed(store):
    return {u.key for u in store.units if store.mode_of(u) is ResidencyMode.LAYER_WINDOW}


# ---------------------------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("waves", [1, 2])
@pytest.mark.parametrize("form", list(FORMS))
def test_layer_window_equals_sharded_grads_and_resident(store_groups, deterministic, form, waves):
    # PER_RANK samples per EU in every wave (waves are contiguous slices of the EU's
    # microbatches): EU 0 then has PER_RANK active samples in round 2 while every other EU is
    # idle, so the rebalancing moves one in every wave (with one sample per wave nothing moves).
    per_rank = PER_RANK * waves
    runs = {"none": _run(store_groups, None, form=form, waves=waves, per_rank=per_rank)}
    for name, text in MODES.items():
        runs[name] = _run(store_groups, text, form=form, waves=waves, per_rank=per_rank)
    reference = runs["none"]
    assert reference["nulls"] > 0 and reference["moves"]["sample"] > 0
    lw = runs["lw"]["store"]
    assert {"core:adapter", "mtp:0:eh_proj", "mtp:0:L"} <= _windowed(lw)
    assert any(
        "linear_o_group_proj" in tensor.name
        for unit in lw.units
        if unit.key in _windowed(lw)
        for tensor in unit.tensors
    )
    for name, run in runs.items():
        _assert_losses_bitwise(run, reference, name)
        _assert_counts_equal(run, reference, name)
        if name != "none":
            _assert_exactly_once(run["store"], mtp=True)
            assert all(
                bool((pad == 0).all()) for pad in psu.shard_pad_rows(run["store"]).values()
            ), name
    _assert_grads_bitwise(runs["lw"], runs["rsg"], "lw == rsg")
    _assert_grads_bitwise(runs["lw_regather"], runs["rsg"], "lw_regather == rsg")
    for name in MODES:
        _assert_grads_close(runs[name], reference, f"{name} ~ no store")
        if name != "resident":
            _assert_grads_close(runs[name], runs["resident"], f"{name} ~ resident")
    assert runs["lw"]["store"].stats.prefetches > 0


@pytest.mark.parametrize("form", ["whole_round", "block2", "nested2"])
def test_fused_wgrad_arm(store_groups, deterministic, form):
    """``gradient_accumulation_fusion=True``, no MTP, untied embedding, move M: TE's fused
    wgrad into the bound targets (overwrite for single-use units), the output layer's
    forward-captured target."""
    if not _fused_wgrad_available():
        pytest.skip("the fused wgrad kernel (fused_weight_gradient_mlp_cuda) is not available")
    runs = {"none": _run(store_groups, None, form=form, mtp=False, fusion=True)}
    for name in ("resident", "rsg", "lw", "lw_regather"):
        runs[name] = _run(store_groups, MODES[name], form=form, mtp=False, fusion=True)
    lw = runs["lw"]["store"]
    output = [t for t in lw.tensors() if t.name.startswith("output_layer.")]
    assert output and all(t.grad_capture == "forward" for t in output)
    assert any(t.grad_capture == "lazy" and t.overwrite for t in lw.tensors())
    for name, run in runs.items():
        _assert_losses_bitwise(run, runs["none"], name)
        if name != "none":
            _assert_exactly_once(run["store"], mtp=False)
    _assert_grads_bitwise(runs["lw"], runs["rsg"], "lw == rsg (fused)")
    _assert_grads_bitwise(runs["lw_regather"], runs["rsg"], "lw_regather == rsg (fused)")
    for name in ("resident", "rsg", "lw"):
        _assert_grads_close(runs[name], runs["none"], f"{name} ~ no store (fused)")


@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("form", ["whole_round", "block2", "nested2"])
def test_literal_case_every_mode_equals_resident_bitwise(store_groups, deterministic, form, fusion):
    """D-S6-9: z = W, one sample per EU, depth 1, no MTP, ``no_overwrite``: every unit gets one
    item per step on every EU and every target is ``0 + g`` with the same GEMM call, so Layer
    window and Resident + sharded grads reduce exactly what Resident reduces at the step's end:
    every gradient and the losses bitwise."""
    if fusion and not _fused_wgrad_available():
        pytest.skip("the fused wgrad kernel (fused_weight_gradient_mlp_cuda) is not available")
    common = dict(
        form=form,
        mtp=False,
        fusion=fusion,
        per_rank=1,
        depths=[1] * WORLD,
        store_config=dict(no_overwrite=True),
    )
    runs = {name: _run(store_groups, MODES[name], **common) for name in MODES}
    reference = runs["resident"]
    for name in ("rsg", "lw", "lw_regather"):
        _assert_losses_bitwise(runs[name], reference, name)
        _assert_grads_bitwise(runs[name], reference, f"literal {name} == resident")
        _assert_exactly_once(runs[name]["store"], mtp=False)


def _copy_engine_unavailable_reason():
    try:
        import torch.distributed._symmetric_memory  # pylint: disable=unused-import
    except ImportError as exc:  # pragma: no cover - depends on the torch build
        return f"torch symmetric memory is unavailable: {exc}"
    return None


@pytest.mark.parametrize("form", ["whole_round", "block2", "nested2"])
def test_copy_engine_gathers_equal_nccl(store_groups, deterministic, form):
    """``gather_engine='copy'``: every in-step gather (window slots, Resident / RSG compute
    copies) pulls the peers' shards from the symmetric mirror with copy-engine copies. Each
    window gather is checked against an NCCL reference gathered at step begin
    (``verify_gathers``), and the losses and every gradient equal the NCCL engine's bitwise,
    over two waves and two steps (the shards change between the steps, so a stale mirror
    fails the check)."""
    reason = _copy_engine_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    common = dict(form=form, waves=2, per_rank=2 * PER_RANK, steps=2)
    for name in ("lw", "lw_regather", "rsg"):
        nccl = _run(store_groups, MODES[name], **common)
        copy = _run(
            store_groups,
            MODES[name],
            store_config=dict(gather_engine="copy", verify_gathers=True),
            **common,
        )
        _assert_losses_bitwise(copy, nccl, f"copy engine {name}")
        _assert_grads_bitwise(copy, nccl, f"copy engine {name} == nccl")
        _assert_exactly_once(copy["store"], mtp=True)
        store = copy["store"]
        assert store.comm.copy_engine is not None, name
        assert store.memory_report().copy_engine_bytes == store.comm.copy_engine_bytes > 0
        assert nccl["store"].memory_report().copy_engine_bytes == 0


@pytest.mark.parametrize("form", ["whole_round", "block2", "nested2"])
def test_copy_engine_reduce_scatters(store_groups, deterministic, form):
    """``reduce_engine='copy'`` (with the copy-engine gathers): every reduce-scatter pushes the
    rows into the owners' symmetric receive slots and sums them locally. The sum is associated
    differently from NCCL's ring, so the gates are: losses bitwise against the NCCL engines;
    Layer window (kept, re-gathered) == Resident + sharded grads bitwise (the copy reduce is
    deterministic and mode-independent); every gradient within the fp32 association bound of
    the NCCL engines; exactly-once reduce counts; the receive slots in the memory report."""
    reason = _copy_engine_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    common = dict(form=form, waves=2, per_rank=2 * PER_RANK, steps=2)
    copy_config = dict(gather_engine="copy", reduce_engine="copy", verify_gathers=True)
    runs = {}
    for name in ("rsg", "lw", "lw_regather"):
        nccl = _run(store_groups, MODES[name], **common)
        runs[name] = _run(store_groups, MODES[name], store_config=copy_config, **common)
        _assert_losses_bitwise(runs[name], nccl, f"copy reduce {name}")
        _assert_grads_close(runs[name], nccl, f"copy reduce {name} ~ nccl")
        _assert_exactly_once(runs[name]["store"], mtp=True)
        store = runs[name]["store"]
        assert store.comm.copy_reduce is not None and store.comm.copy_engine is not None, name
        assert store.memory_report().copy_engine_bytes == store.comm.copy_engine_bytes
        assert store.comm.copy_engine_bytes > store.comm.copy_engine.nbytes > 0
    _assert_grads_bitwise(runs["lw"], runs["rsg"], "copy reduce lw == rsg")
    _assert_grads_bitwise(runs["lw_regather"], runs["rsg"], "copy reduce lw_regather == rsg")


@pytest.fixture(scope="module")
def half_store_groups():
    """Shard groups of W / 2 ranks (two disjoint shard groups, replicate groups of two ranks);
    created once, on every rank, only when a test asks for them (W >= 4)."""
    if WORLD < 4:
        return None
    return psu.make_store_groups(WORLD, WORLD // 2)


def _fake_units(z, scale, seed):
    """Two store-shaped units (bf16 shards with an fp32 ``main_grad``, regions 256-B aligned)
    with rank-dependent shards; ``scale`` grows the first tensor of each unit."""
    from types import SimpleNamespace

    from megatron.core.transformer.loop.param_store_binding import align_elements

    generator = torch.Generator().manual_seed(1000 * seed + dist.get_rank())
    units = []
    for uid, sizes in enumerate(((1000 * scale, 77), (4096 * scale, 3))):
        tensors, offset = [], 0
        for index, shard_numel in enumerate(sizes):
            offset = align_elements(offset, torch.bfloat16, 256)
            shard = torch.randn(shard_numel, generator=generator).to(torch.bfloat16).cuda()
            shard.main_grad = torch.zeros(shard_numel, dtype=torch.float32, device="cuda")
            tensors.append(
                SimpleNamespace(
                    name=f"unit{uid}.tensor{index}",
                    dtype=torch.bfloat16,
                    numel_padded=z * shard_numel,
                    offset=offset,
                    shard=shard,
                )
            )
            offset += z * shard_numel
        units.append(
            SimpleNamespace(
                uid=uid,
                key=f"unit{uid}",
                tensors=tuple(tensors),
                numel=align_elements(offset, torch.bfloat16, 256),
            )
        )
    return units


def _comm_step(copy, nccl, units, seed):
    """One step of both engines on the same units: every gather bitwise equal to NCCL's; every
    copy-engine reduce deterministic (two runs bitwise) and within fp32 rounding of NCCL's."""
    from megatron.core.transformer.loop.param_store_binding import tensor_region

    copy.step_begin()
    nccl.step_begin()
    generator = torch.Generator().manual_seed(7919 * seed + dist.get_rank())
    for unit in units:
        got = torch.full((unit.numel,), float("nan"), dtype=torch.bfloat16, device="cuda")
        want = torch.full_like(got, float("nan"))
        copy.wait(copy.gather(unit, got))
        nccl.wait(nccl.gather(unit, want))
        for tensor in unit.tensors:
            assert torch.equal(
                tensor_region(got, tensor, padded=True), tensor_region(want, tensor, padded=True)
            ), (seed, tensor.name)
        src = torch.randn(unit.numel, generator=generator).cuda()
        results = []
        for comm in (copy, copy, nccl):
            for tensor in unit.tensors:
                tensor.shard.main_grad.zero_()
            comm.wait(comm.reduce(unit, src))
            results.append([tensor.shard.main_grad.clone() for tensor in unit.tensors])
        for first, again, reference, tensor in zip(*results, unit.tensors):
            assert torch.equal(first, again), (seed, tensor.name)
            torch.testing.assert_close(first, reference, rtol=1.3e-6, atol=1e-5)
    copy.join()
    nccl.join()
    torch.cuda.synchronize()


def test_copy_engine_stores_rebuilt_on_sub_world_shard_groups(half_store_groups):
    """R3-A: copy-engine stores built one after another in one process on shard groups smaller
    than the world (z = W / 2, two disjoint groups), the previous store freed before the next is
    built, two steps each (shards changed between them): no deadlock, every gather bitwise equal
    to NCCL's, every copy-engine reduce deterministic and within fp32 rounding of NCCL's. The
    second store reuses the first's symmetric buffers; the third, larger one gets new ones.
    Two stores stepping interleaved on one shard group raise instead of reading each other's
    mirror."""
    import gc

    from megatron.core.transformer.loop.param_store_comm import StoreComm

    reason = _copy_engine_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    if half_store_groups is None:
        pytest.skip("needs W >= 4 (z = W / 2 > 1)")
    shard = half_store_groups.shard
    z = WORLD // 2
    device = torch.device("cuda", torch.cuda.current_device())
    nccl = StoreComm(shard, device)
    shared = []
    for seed, scale in enumerate((1, 1, 2)):
        copy = StoreComm(shard, device, gather_engine="copy", reduce_engine="copy")
        units = _fake_units(z, scale, seed)
        copy.setup_copy_engine(units, torch.bfloat16)
        shared.append((copy.copy_engine.shared, copy.copy_reduce.shared))
        for step in range(2):
            if step:
                with torch.no_grad():
                    for unit in units:
                        for tensor in unit.tensors:
                            tensor.shard.mul_(1 + 2**-7)
            _comm_step(copy, nccl, units, 10 * seed + step)
        del copy, units
        gc.collect()
        torch.cuda.empty_cache()
    assert shared[1][0] is shared[0][0] and shared[1][1] is shared[0][1]
    assert shared[2][0] is not shared[0][0] and shared[2][1] is not shared[0][1]
    first = StoreComm(shard, device, gather_engine="copy")
    second = StoreComm(shard, device, gather_engine="copy")
    units = _fake_units(z, 1, 99)
    first.setup_copy_engine(units, torch.bfloat16)
    second.setup_copy_engine(units, torch.bfloat16)
    first.step_begin()
    second.step_begin()
    dest = torch.empty(units[0].numel, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="holds another store's shards"):
        first.gather(units[0], dest)
    second.join()
    first.join()
    torch.cuda.synchronize()


def test_distributed_weight_binding_equals_the_data_binding(store_groups, deterministic):
    reason = distributed_weight_unavailable_reason()
    if not te_distributed_weight_available():
        pytest.skip(reason)
    data = _run(store_groups, MODES["lw"], form="block2")
    weight = _run(
        store_groups, MODES["lw"], form="block2", store_config=dict(binding="distributed_weight")
    )
    assert any(t.binding == "distributed_weight" for t in weight["store"].tensors())
    _assert_losses_bitwise(weight, data, "distributed_weight")
    _assert_grads_bitwise(weight, data, "distributed_weight == data")
