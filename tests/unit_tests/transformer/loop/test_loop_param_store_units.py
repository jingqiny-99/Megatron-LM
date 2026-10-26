# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the loop parameter store on one EU (``loop/param_store.py``).

A trivial shard axis (``z = 1``, ``allow_trivial_shard``: the gather copies, the reduce adds, no
collective) on a local-layer hybrid loop model (``*-*-*-*-``, prelude 2, core 4, coda 2; CPU and
CUDA): unit discovery (eligibility, store groups, layer-pair units and the per-sublayer
fallback), the canonical shards and their GTP tags, the mirror names, hidden compute parameters
(absent from ``named_parameters`` / ``state_dict``, present in ``named_logical_parameters`` with
logical shapes), install guards, gradient-capture classes and the local-fusion switch; then
whole RSBF-shaped steps through the store's API (``param_store_utils.StoreDriver``): every
residency mode and re-gather policy on every item form, with waves, against the same steps
without a store, **bitwise** (one EU and z = 1: every gradient is added in the same order);
exactly-once reduce counts, null sub-rounds, eager (no-grad) visits, the visit-order check,
step guards and ``abort_step``, and the memory report. Multi-rank shards (z > 1, padding,
nulls, finalize's shard-axis SUM) are in ``test_loop_param_store_gloo.py``.
"""

import itertools

import pytest
import torch
from torch import nn

from megatron.core.transformer.loop import param_store as ps
from megatron.core.transformer.loop.hybrid_stack import LoopStreamRMSNorm
from megatron.core.transformer.loop.param_store import (
    LoopParamStore,
    LoopParamStoreConfig,
    ResidencyMode,
    StoreModes,
    find_param_store,
)
from megatron.core.transformer.loop.param_store_binding import (
    is_placeholder,
    named_logical_parameters,
)
from megatron.core.transformer.loop.param_store_comm import StoreComm
from megatron.core.transformer.loop.param_store_schedule import SegmentKey
from megatron.core.transformer.loop.rsbf_plan import SubRoundType
from tests.unit_tests.transformer.loop import param_store_utils as psu
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    VOCAB,
    build_local_loop_model,
    model_parallel_once,
)

PATTERN = "*-*-*-*-"
SEQ = 16
CORE = 4

MODES = {
    "resident": "all=r",
    "rsg": "all=rsg,emb_head=rsg",
    "lw": "core=lw,edges=lw,emb_head=rsg",
    "lw_regather": "core=lw,edges=lw,emb_head=rsg,core_regather=on,edge_regather=on",
    "mixed": "core=lw,prelude=rsg,coda=lw,emb_head=r,core_regather=on",
}
FORMS = ("whole_round", "block", "nested")


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


def _groups():
    return psu.trivial_store_groups()


def _model(**overrides):
    return build_local_loop_model(PATTERN, prelude=2, coda=2, **overrides)


def _store(model=None, **config):
    model = _model() if model is None else model
    config.setdefault("allow_trivial_shard", True)
    return model, psu.install_store(model, _groups(), 1, **config)


# ---------------------------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------------------------


def test_units_groups_and_layer_pairs():
    model, store = _store()
    assert [(u.key, u.group, u.sublayers) for u in store.units] == [
        ("emb_head:embedding", "emb_head", ()),
        ("prelude:L0", "prelude", (0, 1)),
        ("core:L2", "core", (2, 3)),
        ("core:L4", "core", (4, 5)),
        ("coda:L6", "coda", (6, 7)),
        ("core:adapter", "core", ()),
        ("emb_head:output", "emb_head", ()),
    ]
    core = store.units[2]
    # Module registration order (Attention registers linear_proj before linear_qkv).
    assert [t.name for t in core.tensors] == [
        "decoder.layers.2.self_attention.linear_proj.weight",
        "decoder.layers.2.self_attention.linear_qkv.weight",
        "decoder.layers.3.mlp.linear_fc1.weight",
        "decoder.layers.3.mlp.linear_fc2.weight",
    ]
    stored = {t.name for t in store.tensors()}
    replicated = {name for name, _ in model.named_parameters() if "loop_param_store." not in name}
    # Norms, the loop norm: replicated (1-D or under a norm module).
    assert all("norm" in name for name in replicated), replicated
    assert not stored & replicated
    # Segments: each pair's two sublayers, the adapter, the embedding (untied, no MTP: embed only).
    assert store.segment_units(SegmentKey("sub", 3)) == (core,)
    assert [u.key for u in store.segment_units(SegmentKey("head"))] == ["emb_head:output"]
    assert [u.key for u in store.segment_units(SegmentKey("embed"))] == ["emb_head:embedding"]
    assert [u.key for u in store.segment_units(SegmentKey("adapter"))] == ["core:adapter"]


def test_eligibility_rules():
    model = _model()
    layer = model.decoder.layers[2].self_attention
    layer.linear_proj.weight.allreduce = False  # an expert parameter: EP-resident
    layer.linear_qkv.weight.keep_in_fp32 = True
    model.decoder.layers[3].mlp.linear_fc2.weight.data = model.decoder.layers[
        3
    ].mlp.linear_fc2.weight.data.double()
    _, store = _store(model)
    names = {t.name for t in store.tensors()}
    assert "decoder.layers.2.self_attention.linear_proj.weight" not in names
    assert "decoder.layers.2.self_attention.linear_qkv.weight" not in names
    assert "decoder.layers.3.mlp.linear_fc2.weight" not in names
    assert "decoder.layers.3.mlp.linear_fc1.weight" in names
    # min_numel: only the large tensors (fc1 [256, 64] and the 128-row ones) are stored.
    _, big = _store(min_numel=16384)
    assert {t.name.rsplit(".", 2)[-2] for t in big.tensors()} == {"linear_fc1"}


def test_excluded_owner_classes():
    from megatron.core.transformer.hyper_connection import HyperConnectionModule
    from megatron.core.transformer.moe.router import TopKRouter

    router = TopKRouter.__new__(TopKRouter)
    mhc = HyperConnectionModule.__new__(HyperConnectionModule)
    norm = LoopStreamRMSNorm(8, 2, 1e-6, torch.float32, "cpu")
    assert ps._excluded_owner([nn.Module(), router]) == "router"
    assert ps._excluded_owner([mhc, nn.Linear(2, 2)]) == "mHC"
    assert ps._excluded_owner([norm]) == "norm"  # a 2-D [streams, h] norm weight
    assert ps._excluded_owner([nn.Linear(2, 2)]) is None
    # Only modules whose class name ends in "norm" are norms: TE's fused norm + linear modules
    # (attention linear_qkv, MLP linear_fc1, MLA / GDN / Mamba in-projections in the TE hybrid
    # specs) own the largest dense weights and are stored.
    for name in (
        "TELayerNormColumnParallelLinear",
        "TERMSNormDuplicatedLinear",
        "InferenceLayerNormColumnParallelLinear",
    ):
        fused = type(name, (nn.Linear,), {})(2, 2)
        assert ps._excluded_owner([nn.Module(), fused]) is None, name
    for name in ("LayerNorm", "RMSNorm", "TENorm", "FusedLayerNorm", "L2Norm", "WrappedTorchNorm"):
        assert ps._excluded_owner([type(name, (nn.Module,), {})()]) == "norm", name


def test_per_sublayer_units_under_a_pair_splitting_plan():
    rsbf = dict(
        loop_schedule="rsbf",
        calculate_per_token_loss=True,
        loop_rsbf_item_form="block",
        loop_rsbf_block_stride=1,
    )
    _, store = _store(_model(**rsbf))
    assert [u.key for u in store.units if u.group == "core"] == [
        "core:L2",
        "core:L3",
        "core:L4",
        "core:L5",
        "core:adapter",
    ]
    # Prelude / coda items run whole: their units stay pairs.
    assert [u.sublayers for u in store.units if u.group in ("prelude", "coda")] == [(0, 1), (6, 7)]
    store.configure_modes(StoreModes(core=ResidencyMode.LAYER_WINDOW), core_plan=None)
    # Pair units refuse a plan whose blocks split a pair.
    _, paired = _store()
    with pytest.raises(ValueError, match="inside the store's layer units"):
        paired.configure_modes(
            StoreModes(core=ResidencyMode.LAYER_WINDOW), core_plan=psu.core_plan_of("block", 4, 1)
        )


def test_shards_tags_mirror_and_hidden_parameters():
    reference = _model()
    full = {name: p.detach().clone() for name, p in reference.named_parameters()}
    model, store = _store()
    named = dict(model.named_parameters())
    for tensor in store.tensors():
        mirror_name = f"loop_param_store.mirror.{tensor.name}"
        assert named[mirror_name] is tensor.shard
        assert tensor.name not in named
        shard = tensor.shard
        assert shard.is_gtp_weight_remat and shard.pad_length == 0
        assert (
            shard.group is store.groups.shard and shard.gtp_replica_group is store.groups.replicate
        )
        assert shard.loop_store_name == tensor.name and shard.requires_grad
        assert torch.equal(shard.detach(), full[tensor.name])
        # Megatron attributes are carried (the optimizer and the grad norm read them).
        for attr in ("tensor_model_parallel", "partition_dim", "partition_stride"):
            assert getattr(shard, attr, None) == getattr(tensor.compute, attr, None), attr
        assert shard.allreduce is getattr(tensor.compute, "allreduce", True)
        # The hidden compute parameter: logical shape, NaN placeholder, write-protected.
        assert tuple(tensor.compute.shape) == tensor.logical_shape
        assert is_placeholder(tensor.compute.data)
        with pytest.raises(RuntimeError):
            tensor.compute.data.add_(1.0)
    logical = dict(named_logical_parameters(model))
    assert set(logical) == set(full)
    assert all(tuple(logical[name].shape) == tuple(full[name].shape) for name in full)
    assert "decoder.layers.2.self_attention.linear_qkv.weight" not in model.state_dict()
    assert "loop_param_store.mirror.decoder.layers.2.self_attention.linear_qkv.weight" in (
        model.state_dict()
    )
    with pytest.raises(NotImplementedError, match="M3c"):
        store.sharded_state_dict()


def test_install_guards_and_find_param_store():
    model, store = _store()
    with pytest.raises(ValueError, match="already installed"):
        LoopParamStore.install(model, LoopParamStoreConfig(1, allow_trivial_shard=True), _groups())

    class Wrapper(nn.Module):
        def __init__(self, inner, ddp=False):
            super().__init__()
            self.module = inner
            if ddp:
                self.ddp_config = object()

    assert find_param_store(Wrapper(Wrapper(model))) is store
    assert find_param_store(_model()) is None
    with pytest.raises(ValueError, match="before the DDP wrap"):
        LoopParamStore.install(
            Wrapper(_model(), ddp=True),
            LoopParamStoreConfig(1, allow_trivial_shard=True),
            _groups(),
        )
    with pytest.raises(ValueError, match="shard_size 1 means no store"):
        LoopParamStore.install(_model(), LoopParamStoreConfig(1), _groups())
    with pytest.raises(ValueError, match="DistributedWeight"):
        LoopParamStore.install(
            _model(),
            LoopParamStoreConfig(1, allow_trivial_shard=True, binding="distributed_weight"),
            _groups(),
        )
    for config, match in (
        (dict(gather_engine="nvls"), "unknown gather_engine"),
        (dict(gather_engine="copy"), "needs a shard axis z > 1"),
        (dict(copy_streams=0), "copy_streams must be >= 1"),
        (dict(reduce_engine="nvls"), "unknown reduce_engine"),
        (dict(reduce_engine="copy"), "reduce_engine='copy' needs a shard axis z > 1"),
    ):
        with pytest.raises(ValueError, match=match):
            LoopParamStore.install(
                _model(), LoopParamStoreConfig(1, allow_trivial_shard=True, **config), _groups()
            )
    with pytest.raises(ValueError, match="gather needs CUDA and a non-trivial shard axis"):
        StoreComm(None, torch.device("cpu"), trivial=True, gather_engine="copy")
    with pytest.raises(ValueError, match="reduce-scatter needs CUDA and a non-trivial shard"):
        StoreComm(None, torch.device("cpu"), trivial=True, reduce_engine="copy")
    with pytest.raises(ValueError, match="unknown reduce engine"):
        StoreComm(None, torch.device("cpu"), trivial=True, reduce_engine="nvls")
    qkv = model.decoder.layers[2].self_attention.linear_qkv
    other = _model()
    other.decoder.layers[2].self_attention.linear_qkv.weight.is_gtp_weight_remat = True
    with pytest.raises(ValueError, match="module-GTP shard"):
        LoopParamStore.install(other, LoopParamStoreConfig(1, allow_trivial_shard=True), _groups())
    # The owner still reads its (hidden) compute parameter by attribute.
    by_name = {t.name: t for t in store.tensors()}
    assert qkv.weight is by_name["decoder.layers.2.self_attention.linear_qkv.weight"].compute


def test_copy_engine_buffers_are_shared_and_never_freed(monkeypatch):
    """R3-A: the copy engines' symmetric buffers are allocated once per (group, purpose, dtype,
    device) and reused by every later store that fits; a larger request retires the old buffer
    (kept allocated, never handed back) and allocates a new one; a new group object with a
    reused name gets its own buffer. Host logic only (the allocation is stubbed)."""
    from megatron.core.transformer.loop import param_store_comm as comm

    made = []

    class FakeBuffer:
        def __init__(self, group, device, dtype, numel):
            self.group, self.numel, self.dtype = group, numel, dtype
            made.append(self)

    class FakeGroup:
        def __init__(self, name):
            self.group_name = name

    monkeypatch.setattr(comm, "_SymmetricBuffer", FakeBuffer)
    monkeypatch.setattr(comm, "_SYMMETRIC_BUFFERS", {})
    monkeypatch.setattr(comm, "_RETIRED_SYMMETRIC_BUFFERS", [])
    device = torch.device("cuda", 0)
    group = FakeGroup("7")
    first = comm._shared_symmetric_buffer(group, device, torch.bfloat16, 100, "gather")
    assert comm._shared_symmetric_buffer(group, device, torch.bfloat16, 100, "gather") is first
    assert comm._shared_symmetric_buffer(group, device, torch.bfloat16, 60, "gather") is first
    reduce = comm._shared_symmetric_buffer(group, device, torch.float32, 100, "reduce")
    assert reduce is not first and len(made) == 2
    grown = comm._shared_symmetric_buffer(group, device, torch.bfloat16, 101, "gather")
    assert grown is not first and grown.numel == 101
    assert comm._RETIRED_SYMMETRIC_BUFFERS == [first]
    assert comm._shared_symmetric_buffer(group, device, torch.bfloat16, 50, "gather") is grown
    renamed = FakeGroup("7")  # a later group object reusing the name of a destroyed one
    fresh = comm._shared_symmetric_buffer(renamed, device, torch.bfloat16, 50, "gather")
    assert fresh is not grown and fresh.group is renamed
    assert comm._RETIRED_SYMMETRIC_BUFFERS == [first, grown]
    other_device = comm._shared_symmetric_buffer(
        renamed, torch.device("cuda", 1), torch.bfloat16, 50, "gather"
    )
    assert other_device is not fresh and len(made) == 5


def test_grad_capture_classes_and_local_fusion_switch():
    model = _model()
    model.decoder.layers[2].self_attention.linear_qkv.gradient_accumulation_fusion = True
    model.output_layer.gradient_accumulation_fusion = True
    _, store = _store(model)
    by_name = {t.name: t for t in store.tensors()}
    qkv = by_name["decoder.layers.2.self_attention.linear_qkv.weight"]
    out = by_name["output_layer.weight"]
    # A sublayer unit's local fused linear is switched to the hook path (D-S6-16).
    assert not model.decoder.layers[2].self_attention.linear_qkv.gradient_accumulation_fusion
    assert qkv.grad_capture == "hook"
    assert store.fusion_disabled == ("decoder.layers.2.self_attention.linear_qkv",)
    # The output layer keeps its fusion: its target is bound at the head forward (E15).
    assert out.grad_capture == "forward" and out.fused_wgrad
    assert by_name["embedding.word_embeddings.weight"].grad_capture == "hook"
    assert by_name["decoder.loop_adapter.weight"].grad_capture == "hook"
    single = {t.name.split(".")[0] + ":" + str(t.single_use) for t in store.tensors()}
    assert "embedding:False" in single and "output_layer:False" in single
    assert not by_name["decoder.loop_adapter.weight"].single_use
    assert qkv.single_use
    store.configure_modes(StoreModes.parse("all=rsg,emb_head=rsg"), core_plan=None)
    coda = store.schedule(SubRoundType.CODA)
    head_fwd = next(
        i for i, op in enumerate(coda.ops) if op.direction == "fwd" and op.segment.kind == "head"
    )
    out_uid = next(u.uid for u in store.units if u.key == "emb_head:output")
    assert any(e.action == "bind_grad" and e.unit == out_uid for e in coda.at(head_fwd, "before"))


def _cpu_fused_wgrad(monkeypatch):
    """Megatron's fused wgrad accumulation on CPU: the CUDA extension's fp32 GEMM-accumulate and
    TE's dummy wgrad, so the forward-capture path of a local fused linear runs in a CPU test."""
    from types import SimpleNamespace

    from megatron.core.tensor_parallel import layers

    def accum(total_input, grad_output, main_grad):
        main_grad.add_(grad_output.t().matmul(total_input))

    monkeypatch.setattr(
        layers,
        "fused_weight_gradient_mlp_cuda",
        SimpleNamespace(wgrad_gemm_accum_fp32=accum),
        raising=False,
    )
    monkeypatch.setattr(layers, "HAVE_TE", True)
    monkeypatch.setattr(
        layers,
        "get_dummy_wgrad",
        lambda shape, dtype, zero=False: torch.zeros(shape, dtype=dtype),
        raising=False,
    )


@pytest.mark.parametrize("fused", (False, True))
def test_tied_embedding_with_a_fused_output_layer(monkeypatch, fused):
    """A tied embedding under Resident + sharded grads: the fused output layer captures
    ``main_grad`` at the head forward, so the embedding's target is bound there (``forward`` class
    in the head segment); the lookup's gradient still takes the hook in the embed segment."""
    if fused:
        if torch.cuda.is_available():
            pytest.skip("the CPU stand-in of the fused wgrad kernel")
        _cpu_fused_wgrad(monkeypatch)
    plan = psu.core_plan_of("whole_round", CORE)
    reference = _model(share_embeddings_and_output_weights=True)
    passes = [psu.make_samples(0, 2, vocab=VOCAB, seq=SEQ, seed=11)]
    psu.StoreDriver(reference, None, depth=2, core_plan=plan).run_step(passes, 2)
    ref_losses = [s.loss for s in passes[0]]
    ref_grads = psu.logical_grads(reference)

    model = _model(share_embeddings_and_output_weights=True)
    model.output_layer.gradient_accumulation_fusion = fused
    _, store = _store(model)
    emb = next(t for t in store.tensors() if t.name == "embedding.word_embeddings.weight")
    assert "output_layer.weight" not in {t.name for t in store.tensors()}
    assert emb.grad_capture == ("forward" if fused else "hook")
    store.configure_modes(StoreModes.parse("all=r,emb_head=rsg"), core_plan=plan)
    coda = store.schedule(SubRoundType.CODA)
    head_fwd = next(
        i for i, op in enumerate(coda.ops) if op.direction == "fwd" and op.segment.kind == "head"
    )
    emb_uid = next(u.uid for u in store.units if u.key == "emb_head:embedding")
    at_forward = any(
        e.action == "bind_grad" and e.unit == emb_uid for e in coda.at(head_fwd, "before")
    )
    assert at_forward is fused
    samples = psu.make_samples(0, 2, vocab=VOCAB, seq=SEQ, seed=11)
    psu.StoreDriver(model, store, depth=2, core_plan=plan).run_step([samples], 2)
    assert all(torch.equal(s.loss, ref) for s, ref in zip(samples, ref_losses))
    grads = psu.full_store_grads(store, store.groups)
    grads.update(psu.replicated_grads(model))
    assert set(grads) == set(ref_grads)
    # Head wgrads, then lookup wgrads, in item order: the reference's accumulation order.
    for name, grad in grads.items():
        assert torch.equal(grad, ref_grads[name]), name


def test_modes_parse_and_validate():
    modes = StoreModes.parse("core=lw,edges=rsg,emb_head=rsg,core_regather=on")
    assert (
        modes.core is ResidencyMode.LAYER_WINDOW
        and modes.mtp is ResidencyMode.RESIDENT_SHARDED_GRADS
    )
    assert modes.core_regather and not modes.edge_regather
    assert StoreModes.parse(modes.describe()) == modes
    assert StoreModes.parse("") == StoreModes()
    for bad, match in (
        ("emb_head=lw", "not supported"),
        ("core=fast", "unknown mode"),
        ("trunk=lw", "unknown group"),
        ("core", "key=value"),
        ("core_regather=maybe", "on / off"),
        ("auto", "resolved by the trainer"),
    ):
        with pytest.raises(ValueError, match=match):
            StoreModes.parse(bad)


# ---------------------------------------------------------------------------------------------
# Steps through the store's API against the same steps without a store (bitwise at z = 1)
# ---------------------------------------------------------------------------------------------


def _reference(form, waves, count, depth=2, **overrides):
    model = _model(**overrides)
    plan = psu.core_plan_of(form, CORE)
    passes = [psu.make_samples(0, count, vocab=VOCAB, seq=SEQ, seed=11 + w) for w in range(waves)]
    psu.StoreDriver(model, None, depth=depth, core_plan=plan).run_step(passes, count)
    return [s.loss for samples in passes for s in samples], psu.logical_grads(model)


def _store_run(form, modes, waves, count, depth=2, lookahead=True, steps=1, **config):
    model, store = _store(**config)
    plan = psu.core_plan_of(form, CORE)
    store.configure_modes(StoreModes.parse(modes), core_plan=plan)
    driver = psu.StoreDriver(model, store, depth=depth, core_plan=plan, lookahead=lookahead)
    for _ in range(steps):
        passes = [
            psu.make_samples(0, count, vocab=VOCAB, seq=SEQ, seed=11 + w) for w in range(waves)
        ]
        driver.run_step(passes, count)
    losses = [s.loss for samples in passes for s in samples]
    grads = psu.full_store_grads(store, store.groups)
    grads.update(psu.replicated_grads(model))
    return model, store, losses, grads


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("mode", sorted(MODES))
def test_store_steps_equal_the_storeless_steps_bitwise(form, mode):
    waves = 2 if mode in ("lw", "resident") else 1
    ref_losses, ref_grads = _reference(form, waves, 2)
    _, store, losses, grads = _store_run(
        form, MODES[mode], waves, 2, poison_released=True, verify_gathers=True, check_visits=True
    )
    assert all(torch.equal(a, b) for a, b in zip(losses, ref_losses))
    assert set(grads) == set(ref_grads)
    for name, grad in grads.items():
        assert torch.equal(grad, ref_grads[name]), name
    assert not store.stats.zeroed_targets


def test_lookahead_prefetches_and_carries_without_changing_results():
    _, store, losses, grads = _store_run("whole_round", MODES["lw"], 1, 2)
    _, plain, plain_losses, plain_grads = _store_run(
        "whole_round", MODES["lw"], 1, 2, lookahead=False
    )
    assert store.stats.prefetches > 0 and store.stats.carries > 0
    assert plain.stats.prefetches == plain.stats.carries == 0
    # Lookahead saves gathers (a carried unit is not re-gathered).
    assert sum(store.stats.gathers.values()) < sum(plain.stats.gathers.values())
    assert all(torch.equal(a, b) for a, b in zip(losses, plain_losses))
    assert all(torch.equal(grads[n], plain_grads[n]) for n in grads)


def test_exactly_once_reduce_counts():
    count, depth = 2, 2
    _, store, _, _ = _store_run("block", "core=lw,prelude=rsg,coda=lw,emb_head=rsg", 1, count)
    reduces = store.stats.reduces
    for unit in store.units:
        mode = store.mode_of(unit)
        if mode is ResidencyMode.RESIDENT:
            assert store.stats.resident_reduces[unit.key] == 1
            assert not any(key == unit.key for _, key in reduces)
            continue
        expected = {
            "core": {"core_b": count * depth},
            "prelude": {"prelude_b": count},
            "coda": {"coda": count},
        }.get(unit.group)
        if unit.key == "emb_head:embedding":
            expected = {"prelude_b": count}
        elif unit.key == "emb_head:output":
            expected = {"coda": count}
        got = {kind: n for (kind, key), n in reduces.items() if key == unit.key}
        assert got == expected, unit.key


def test_null_subrounds_reduce_zeros():
    model, store = _store()
    plan = psu.core_plan_of("whole_round", CORE)
    store.configure_modes(StoreModes.parse(MODES["lw"]), core_plan=plan)
    driver = psu.StoreDriver(model, store, depth=2, core_plan=plan)
    samples = psu.make_samples(0, 1, vocab=VOCAB, seq=SEQ, seed=11)
    driver.run_step([samples], 2)  # one real item per type, then a null of each
    nulls = {kind for (kind, real), _ in store.stats.subrounds.items() if real == "null"}
    assert nulls == {k.value for k in SubRoundType}
    ref_losses, ref_grads = _reference("whole_round", 1, 1)
    assert torch.equal(samples[0].loss, ref_losses[0])
    grads = psu.full_store_grads(store, store.groups)
    for name, grad in grads.items():
        assert torch.equal(grad, ref_grads[name]), name
    # A real sub-round carried its last unit into the next (null) one: the null release unbinds
    # it, so after the step no compute parameter points into a window slot.
    assert store.stats.carries > 0
    for unit in store.units:
        if store.mode_of(unit) is ResidencyMode.LAYER_WINDOW:
            assert not unit.bound and unit.slot is None, unit.key
            for tensor in unit.tensors:
                assert tensor.bound is None and is_placeholder(tensor.compute.data), tensor.name


def test_null_subrounds_issue_the_real_items_collectives():
    model, store = _store(record_calls=True)
    plan = psu.core_plan_of("nested", CORE)
    store.configure_modes(StoreModes.parse(MODES["lw_regather"]), core_plan=plan)
    driver = psu.StoreDriver(model, store, depth=2, core_plan=plan, lookahead=False)
    driver.run_step([psu.make_samples(0, 2, vocab=VOCAB, seq=SEQ)], 2)
    real_log, store.comm.log = store.comm.log, []
    store.begin_step()
    for kind, _, _ in driver.subrounds(2):
        with store.subround(kind, real=False):
            store.null_visitor(kind).visit_all()
    store.end_step()
    assert real_log and store.comm.log == real_log


def test_null_visitor_order_and_type_checks():
    model, store = _store()
    store.configure_modes(StoreModes.parse(MODES["lw"]), core_plan=None)
    store.begin_step()
    with pytest.raises(RuntimeError, match="null_visitor"):
        store.null_visitor(SubRoundType.CODA)
    with store.subround(SubRoundType.CODA, real=False):
        visitor = store.null_visitor(SubRoundType.CODA)
        with pytest.raises(RuntimeError, match="null visit order"):
            visitor.after(0)
        visitor.before(0)
        with pytest.raises(RuntimeError, match="null visit order"):
            visitor.before(1)
        visitor.after(0)
        visitor.visit_all()
    store.end_step()


def test_visit_order_is_checked():
    model, store = _store()
    plan = psu.core_plan_of("whole_round", CORE)
    store.configure_modes(StoreModes.parse(MODES["lw"]), core_plan=plan)
    store.begin_step()
    with pytest.raises(RuntimeError, match="store visit order: expected fwd sub 6, got fwd sub 7"):
        with store.subround(SubRoundType.CODA):
            with store.segment(SegmentKey("sub", 7)):
                pass
    store.abort_step()
    store.begin_step()
    with pytest.raises(RuntimeError, match="ended after 0 of 6"):
        with store.subround(SubRoundType.CODA):
            pass
    store.abort_step()
    # The aborted steps left nothing bound: a full step still matches the reference.
    driver = psu.StoreDriver(model, store, depth=2, core_plan=plan)
    samples = psu.make_samples(0, 2, vocab=VOCAB, seq=SEQ, seed=11)
    psu.attach_shard_grads(store)
    driver.run_step([samples], 2)
    ref_losses, ref_grads = _reference("whole_round", 1, 2)
    assert all(torch.equal(s.loss, ref) for s, ref in zip(samples, ref_losses))
    grads = psu.full_store_grads(store, store.groups)
    assert all(torch.equal(grads[n], ref_grads[n]) for n in grads)


def test_a_raising_item_leaves_the_store_clean():
    model, store = _store()
    plan = psu.core_plan_of("block", CORE)
    store.configure_modes(StoreModes.parse(MODES["lw_regather"]), core_plan=plan)
    driver = psu.StoreDriver(model, store, depth=2, core_plan=plan)

    def boom(*args, **kwargs):
        raise RuntimeError("item failure")

    original = driver.core_item
    driver.core_item = boom
    with pytest.raises(RuntimeError, match="item failure"):
        driver.run_step([psu.make_samples(0, 2, vocab=VOCAB, seq=SEQ, seed=11)], 2)
    for unit in store.units:
        if store.mode_of(unit) is ResidencyMode.LAYER_WINDOW:
            assert unit.slot is None and all(t.bound is None for t in unit.tensors)
        assert unit.grad_flat is None
    driver.core_item = original
    psu.attach_shard_grads(store)
    samples = psu.make_samples(0, 2, vocab=VOCAB, seq=SEQ, seed=11)
    driver.run_step([samples], 2)
    ref_losses, ref_grads = _reference("block", 1, 2)
    grads = psu.full_store_grads(store, store.groups)
    assert all(torch.equal(s.loss, ref) for s, ref in zip(samples, ref_losses))
    assert all(torch.equal(grads[n], ref_grads[n]) for n in grads)


def test_step_guards():
    model, store = _store()
    with pytest.raises(ValueError, match="configure_modes"):
        store.begin_step()
    store.configure_modes(StoreModes(), core_plan=None)
    with pytest.raises(RuntimeError, match="outside begin_step"):
        with store.subround(SubRoundType.CODA):
            pass
    with pytest.raises(RuntimeError, match="without begin_step"):
        store.end_step()
    store.begin_step()
    with pytest.raises(RuntimeError, match="inside a step"):
        store.begin_step()
    with pytest.raises(RuntimeError, match="between steps"):
        store.configure_modes(StoreModes(), core_plan=None)
    with store.subround(SubRoundType.PRELUDE_F, lookahead=SubRoundType.CODA):
        with pytest.raises(RuntimeError, match="inside another"):
            with store.subround(SubRoundType.CODA):
                pass
        with pytest.raises(RuntimeError, match="end_step with a sub-round open"):
            store.end_step()
        with torch.no_grad():
            with store.segment(SegmentKey("embed")):
                pass
            for index in range(2):
                with store.segment(SegmentKey("sub", index)):
                    pass
    store.end_step()


def test_eager_visits_need_no_grad_and_fresh_copies():
    model, store = _store()
    plan = psu.core_plan_of("whole_round", CORE)
    store.configure_modes(StoreModes.parse("core=lw,edges=lw,emb_head=rsg"), core_plan=plan)
    reference = _model()
    sample = psu.make_samples(0, 1, vocab=VOCAB, seq=SEQ)[0]
    ref_sample = psu.make_samples(0, 1, vocab=VOCAB, seq=SEQ)[0]
    psu.StoreDriver(reference, None, depth=2, core_plan=plan).prelude_sweep(ref_sample)
    driver = psu.StoreDriver(model, store, depth=2, core_plan=plan)
    with pytest.raises(RuntimeError, match="stale"):
        driver.prelude_sweep(sample)  # the RSG embedding copy was never gathered
    store.refresh()
    driver.prelude_sweep(sample)  # no sub-round: eager visits (synchronous gathers)
    assert torch.equal(sample.injection, ref_sample.injection)
    # The prelude pair, once per sublayer visit (the RSG embedding reads its fresh copy).
    assert store.stats.eager_gathers == 2
    assert all(t.bound is None for u in store.units if u.group == "prelude" for t in u.tensors)
    with pytest.raises(RuntimeError, match="trains only through loop_schedule='rsbf'"):
        with store.segment(SegmentKey("sub", 0)):
            pass


def test_memory_report_counts_every_buffer():
    model, store = _store()
    plan = psu.core_plan_of("whole_round", CORE)
    store.configure_modes(
        StoreModes.parse("core=lw,prelude=rsg,coda=r,emb_head=rsg"), core_plan=plan
    )
    report = store.memory_report()
    elem = 4
    by_group = {}
    for unit in store.units:
        by_group.setdefault(unit.group, []).append(unit)
    assert report.shard_bytes == {
        group: sum(u.shard_numel for u in units) * (elem + 4) for group, units in by_group.items()
    }
    assert report.resident_copy_bytes == {
        group: sum(u.numel_padded for u in units) * elem
        for group, units in by_group.items()
        if group != "core"
    }
    assert report.accumulator_bytes == {"coda": by_group["coda"][0].numel_padded * 4}
    lw = [u for u in by_group["core"]]
    assert report.slots == store.slot_count == 3  # whole round kept: adapter + two pairs
    assert report.slot_bytes == 3 * max(u.numel_padded for u in lw) * elem
    assert report.scratch_entries == 2
    assert report.scratch_bytes == 2 * max(u.numel_padded for u in lw) * 4
    assert report.padding_bytes == 0 and report.alignment_bytes == 0
    assert report.copy_engine_bytes == 0  # gather_engine="nccl"
    assert report.total_persistent() > 0
    terms = store.memory_terms()
    assert terms.shard_axis == 1 and terms.adapter_unit and terms.scratch_entries == 2
    assert terms.slots[SubRoundType.CORE_B.value] == 3 and terms.cast_units_kept == 3
    # Logical shapes only: the same terms after a step (compute parameters unbound).
    driver = psu.StoreDriver(model, store, depth=2, core_plan=plan)
    driver.run_step([psu.make_samples(0, 1, vocab=VOCAB, seq=SEQ)], 1)
    assert store.memory_terms() == terms
    # Measured buffers equal the report (CPU: the tensors the store holds).
    measured = sum(t.numel() * t.element_size() for t in store._slots) + sum(
        t.numel() * t.element_size() for t in store._scratch
    )
    assert measured == report.slot_bytes + report.scratch_bytes
    copies = sum(u.copy.numel() * u.copy.element_size() for u in store.units if u.copy is not None)
    assert copies == sum(report.resident_copy_bytes.values())


def test_coda_transient_terms_are_disjoint():
    """An output layer larger than the window pool's entries reduces from a transient in the
    coda; the planner adds ``head_scratch_coda`` and ``rsg_transient_coda``, so each counts its
    own units (head-segment units / the others) and they sum to no less than the coda's peak."""
    _, store = _store(_model(vocab_size=8 * VOCAB))
    plan = psu.core_plan_of("whole_round", CORE)
    output = next(u for u in store.units if u.key == "emb_head:output")
    for text, rsg_peak in (
        ("core=lw,prelude=rsg,coda=r,emb_head=rsg", 0),
        ("core=lw,edges=rsg,emb_head=rsg", 0),  # coda units fit the pool: no transient
    ):
        store.configure_modes(StoreModes.parse(text), core_plan=plan)
        assert output.numel_padded > store.scratch_numel
        terms = store.memory_terms()
        coda_peak = store.memory_report().transient_scratch_peak[SubRoundType.CODA.value]
        assert coda_peak == output.numel_padded * 4
        assert terms.head_scratch_coda == coda_peak and terms.rsg_transient_coda == rsg_peak


def test_every_mode_combination_builds_checked_schedules():
    _, store = _store()
    names = ("r", "rsg", "lw")
    for core, prelude, coda in itertools.product(names, repeat=3):
        for form in FORMS:
            text = f"core={core},prelude={prelude},coda={coda},emb_head=rsg"
            store.configure_modes(StoreModes.parse(text), core_plan=psu.core_plan_of(form, CORE))
            assert store.slot_count >= (2 if "lw" in (core,) else 0)
