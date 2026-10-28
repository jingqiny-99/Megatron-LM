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


class _FakeGroup:
    def __init__(self, name):
        self.group_name = name


def test_copy_engine_buffers_are_shared_and_never_freed(monkeypatch):
    """R3-A: the copy engines' symmetric buffers are allocated once per (group, purpose, dtype,
    device) and reused by every later store that fits; a larger request retires the old buffer
    (kept allocated, never handed back) and allocates a new one; a new group object with a
    reused name gets its own buffer. Every request first checks the group's agreement on
    (reuse, requested elements, generation, handshake sequence) and then handshakes the buffer
    at its generation and the request's sequence number (one more per request of the key, so a
    reuse handshake writes new words) with every rank's process nonce. Host logic only
    (allocation and collectives stubbed)."""
    from megatron.core.transformer.loop import param_store_comm as comm

    made, handshakes, agreements = [], [], []

    class FakeBuffer:
        def __init__(self, group, device, dtype, numel, *, purpose, timeout_ms, generation):
            self.group, self.numel, self.dtype = group, numel, dtype
            self.purpose, self.timeout_ms, self.generation = purpose, timeout_ms, generation
            self.nbytes = numel * torch.empty((), dtype=dtype).element_size()
            made.append(self)

        def _handshake(self, generation, nonces, sequence):
            handshakes.append((self, generation, tuple(nonces), sequence))

    def one_rank_gather(group, device, values):
        agreements.append(tuple(values))
        return [list(values)]

    monkeypatch.setattr(comm, "_SymmetricBuffer", FakeBuffer)
    monkeypatch.setattr(comm, "_SYMMETRIC_BUFFERS", {})
    monkeypatch.setattr(comm, "_RETIRED_SYMMETRIC_BUFFERS", [])
    monkeypatch.setattr(comm, "_SYMMETRIC_GENERATIONS", {})
    monkeypatch.setattr(comm, "_HANDSHAKE_SEQUENCES", {})
    monkeypatch.setattr(comm, "_all_gather_ints", one_rank_gather)
    nonce = comm._process_nonce()
    assert nonce == comm._process_nonce()  # once per process
    device = torch.device("cuda", 0)
    group = _FakeGroup("7")

    def shared(group, numel, purpose="gather", dtype=torch.bfloat16, device=device):
        return comm._shared_symmetric_buffer(group, device, dtype, numel, purpose, timeout_ms=5)

    first = shared(group, 100)
    assert (first.purpose, first.timeout_ms, first.generation) == ("gather", 5, 1)
    assert shared(group, 100) is first
    assert shared(group, 60) is first
    reduce = shared(group, 100, "reduce", torch.float32)
    assert reduce is not first and len(made) == 2
    grown = shared(group, 101)
    assert grown is not first and grown.numel == 101 and grown.generation == 2
    assert comm._RETIRED_SYMMETRIC_BUFFERS == [first]
    assert shared(group, 50) is grown
    renamed = _FakeGroup("7")  # a later group object reusing the name of a destroyed one
    fresh = shared(renamed, 50)
    assert fresh is not grown and fresh.group is renamed and fresh.generation == 3
    assert comm._RETIRED_SYMMETRIC_BUFFERS == [first, grown]
    other_device = shared(renamed, 50, device=torch.device("cuda", 1))
    assert other_device is not fresh and len(made) == 5 and other_device.generation == 1
    # (reuse, requested elements, generation before the request, handshake sequence of the
    # key) + the private nonce, per call.
    assert agreements == [
        (0, 100, 0, 1, nonce),
        (1, 100, 1, 2, nonce),
        (1, 60, 1, 3, nonce),
        (0, 100, 0, 1, nonce),
        (0, 101, 1, 4, nonce),
        (1, 50, 2, 5, nonce),
        (0, 50, 2, 6, nonce),
        (0, 50, 0, 1, nonce),
    ]
    # Every request (reuse included) handshakes the buffer it returns, at its generation and
    # the request's sequence number.
    assert [(buffer, generation, seq) for buffer, generation, _, seq in handshakes] == [
        (first, 1, 1),
        (first, 1, 2),
        (first, 1, 3),
        (reduce, 1, 1),
        (grown, 2, 4),
        (grown, 2, 5),
        (fresh, 3, 6),
        (other_device, 1, 1),
    ]
    assert all(nonces == (nonce,) for _, _, nonces, _ in handshakes)
    held = comm.copy_engine_buffer_bytes()
    assert held["retired"] == (100 + 101) * 2
    assert held["live"] == 100 * 4 + 50 * 2 + 50 * 2


def test_copy_engine_buffer_cache_is_unchanged_by_a_failed_request(monkeypatch):
    """A request that fails leaves no half-updated cache: a construction (rendezvous) failure
    keeps the cached buffer cached, not retired, at its generation; a handshake failure
    evicts the buffer it was checking to the retired list (counted once), so the next request
    allocates afresh instead of reusing it. The handshake sequence advances on every request."""
    from megatron.core.transformer.loop import param_store_comm as comm

    made, handshakes = [], []
    fail = {"build": False, "handshake": False}

    class FakeBuffer:
        def __init__(self, group, device, dtype, numel, *, purpose, timeout_ms, generation):
            if fail["build"]:
                raise RuntimeError("rendezvous failed")
            self.group, self.numel, self.generation = group, numel, generation
            self.nbytes = numel * torch.empty((), dtype=dtype).element_size()
            made.append(self)

        def _handshake(self, generation, nonces, sequence):
            handshakes.append((self, sequence))
            if fail["handshake"]:
                raise RuntimeError("handshake failed")

    monkeypatch.setattr(comm, "_SymmetricBuffer", FakeBuffer)
    monkeypatch.setattr(comm, "_SYMMETRIC_BUFFERS", {})
    monkeypatch.setattr(comm, "_RETIRED_SYMMETRIC_BUFFERS", [])
    monkeypatch.setattr(comm, "_SYMMETRIC_GENERATIONS", {})
    monkeypatch.setattr(comm, "_HANDSHAKE_SEQUENCES", {})
    monkeypatch.setattr(comm, "_all_gather_ints", lambda group, device, values: [list(values)])
    group, device = _FakeGroup("9"), torch.device("cuda", 0)
    key = ("9", "gather", torch.bfloat16, 0)

    def shared(numel):
        return comm._shared_symmetric_buffer(
            group, device, torch.bfloat16, numel, "gather", timeout_ms=0
        )

    first = shared(100)
    fail["build"] = True
    with pytest.raises(RuntimeError, match="rendezvous failed"):
        shared(200)  # grow: the new buffer's construction fails
    assert comm._SYMMETRIC_BUFFERS[key] is first and not comm._RETIRED_SYMMETRIC_BUFFERS
    assert comm._SYMMETRIC_GENERATIONS[key] == 1
    assert comm.copy_engine_buffer_bytes() == {"live": 200, "retired": 0}
    fail["build"] = False
    fail["handshake"] = True
    with pytest.raises(RuntimeError, match="handshake failed"):
        shared(100)  # a reuse whose handshake fails
    assert key not in comm._SYMMETRIC_BUFFERS and comm._RETIRED_SYMMETRIC_BUFFERS == [first]
    with pytest.raises(RuntimeError, match="handshake failed"):
        shared(100)  # a new buffer whose handshake fails
    second = made[-1]
    assert second is not first and second.generation == 2
    assert key not in comm._SYMMETRIC_BUFFERS
    assert comm._RETIRED_SYMMETRIC_BUFFERS == [first, second]
    assert comm.copy_engine_buffer_bytes() == {"live": 0, "retired": 400}
    fail["handshake"] = False
    third = shared(100)
    assert third is not second and third.generation == 3
    assert comm._SYMMETRIC_BUFFERS[key] is third
    assert comm.copy_engine_buffer_bytes() == {"live": 200, "retired": 400}
    # One sequence number per request, failed ones included (request 2 failed before it).
    assert [seq for _, seq in handshakes] == [1, 3, 4, 5]


def test_copy_engine_buffer_kept_when_construction_fails_after_allocation(monkeypatch):
    """R3-A "never freed": a ``_SymmetricBuffer`` whose constructor raises after the allocation
    (rendezvous error, a handle of another world size, a rank that does not map onto the group)
    keeps the allocation on the retired list (counted in ``copy_engine_buffer_bytes``) instead
    of letting it be garbage-collected. Allocation and rendezvous stubbed on CPU."""
    import torch.distributed._symmetric_memory as symm_mem

    from megatron.core.transformer.loop import param_store_comm as comm

    class Group(_FakeGroup):
        def size(self):
            return 2

    class Handle:
        def __init__(self, world_size, rank):
            self.world_size, self.rank = world_size, rank

    allocated = []

    def fake_empty(nbytes, dtype, device, *, group_name, purpose):
        allocated.append(torch.empty(nbytes, dtype=dtype, device=device))
        return allocated[-1]

    monkeypatch.setattr(comm, "_RETIRED_SYMMETRIC_BUFFERS", [])
    monkeypatch.setattr(comm, "_symmetric_empty", fake_empty)
    monkeypatch.setattr(comm.dist, "get_process_group_ranks", lambda group: [4, 5])
    monkeypatch.setattr(comm.dist, "get_rank", lambda group=None: 0)

    def rendezvous_error(raw, group_name):
        raise RuntimeError("no NVLink peer")

    for rendezvous, match in (
        (rendezvous_error, "rendezvous of the gather buffer .* failed: no NVLink peer"),
        (lambda raw, group_name: Handle(4, 0), "rendezvous of the gather buffer .* failed"),
        (lambda raw, group_name: Handle(2, 1), "symmetric-memory rank 1 != group rank 0"),
    ):
        monkeypatch.setattr(symm_mem, "rendezvous", rendezvous)
        with pytest.raises(RuntimeError, match=match):
            comm._SymmetricBuffer(
                Group("5"), torch.device("cpu"), torch.bfloat16, 100, purpose="gather"
            )
    retired = comm._RETIRED_SYMMETRIC_BUFFERS
    assert len(retired) == 3 and [entry.raw for entry in retired] == allocated
    assert all(entry.nbytes == 200 for entry in retired)
    assert comm.copy_engine_buffer_bytes()["retired"] == 600


def test_copy_engine_agreement_mismatch_names_disagreeing_ranks(monkeypatch):
    """The setup-time agreement check: equal records pass; differing ones raise on every rank
    (before any allocation) with every rank's record."""
    from megatron.core.transformer.loop import param_store_comm as comm

    assert comm._agreement_mismatch([(1, 100, 2), (1, 100, 2)]) is None
    assert comm._agreement_mismatch([(1, 100, 2)]) is None
    message = comm._agreement_mismatch([(1, 100, 2), (0, 101, 2), (1, 100, 2)])
    assert message == "rank 0: (1, 100, 2); rank 1: (0, 101, 2); rank 2: (1, 100, 2)"

    made = []
    monkeypatch.setattr(comm, "_SymmetricBuffer", lambda *a, **k: made.append(a))
    monkeypatch.setattr(comm, "_SYMMETRIC_BUFFERS", {})
    monkeypatch.setattr(comm, "_RETIRED_SYMMETRIC_BUFFERS", [])
    monkeypatch.setattr(comm, "_SYMMETRIC_GENERATIONS", {})
    monkeypatch.setattr(comm, "_HANDSHAKE_SEQUENCES", {})
    monkeypatch.setattr(comm.dist, "get_process_group_ranks", lambda group: [4, 5])
    # Group rank 1 asks a larger buffer than rank 0 (rank 0 would reuse, rank 1 allocate).
    monkeypatch.setattr(
        comm, "_all_gather_ints", lambda group, device, values: [list(values), [0, 101, 0, 1, 9]]
    )
    with pytest.raises(
        RuntimeError, match=r"\[4, 5\] disagree on the gather buffer .*rank 1: \(0, 101, 0, 1\)"
    ):
        comm._shared_symmetric_buffer(
            _FakeGroup("3"), torch.device("cuda", 0), torch.bfloat16, 100, "gather", timeout_ms=0
        )
    assert not made and not comm._SYMMETRIC_BUFFERS and not comm._SYMMETRIC_GENERATIONS
    # The private words (the nonces) are returned, not compared.
    monkeypatch.setattr(
        comm, "_all_gather_ints", lambda group, device, values: [[1, 2, 11], [1, 2, 12]]
    )
    assert comm._check_group_agreement(None, None, (1, 2), "x", private=(11,)) == [(11,), (12,)]


def test_copy_engine_allocation_refuses_the_pool_fallback(monkeypatch):
    """R3-A: a torch whose dedicated symmetric allocator takes another signature is refused;
    the implicit symmetric MemPool is never used. A dedicated allocation records its path."""
    from megatron.core.transformer.loop import param_store_comm as comm

    class OtherSignature:
        @staticmethod
        def empty_strided_p2p(*args, **kwargs):
            raise TypeError("empty_strided_p2p(): incompatible function arguments")

    class Dedicated:
        @staticmethod
        def empty_strided_p2p(size, stride, dtype, device, group_name=None, alloc_id=None):
            return torch.empty(size, dtype=dtype, device=device)

    monkeypatch.setattr(comm, "_ALLOCATION_PATHS", {})
    monkeypatch.setattr(comm, "_symmetric_memory_class", lambda: OtherSignature)
    with pytest.raises(RuntimeError, match=r"implicit-pool fallback is refused \(R3-A\)"):
        comm._symmetric_empty(8, torch.uint8, torch.device("cpu"), group_name="7", purpose="gather")
    assert comm.copy_engine_allocation_paths() == {}
    monkeypatch.setattr(comm, "_symmetric_memory_class", lambda: Dedicated)
    tensor = comm._symmetric_empty(
        8, torch.uint8, torch.device("cpu"), group_name="7", purpose="reduce"
    )
    assert tensor.shape == (8,) and tensor.dtype == torch.uint8
    assert comm.copy_engine_allocation_paths() == {("7", "reduce"): "empty_strided_p2p"}


def test_copy_engine_api_check_reports_missing_pieces(monkeypatch):
    """The API check names every missing piece of torch's symmetric memory, and the copy-engine
    setup refuses with that reason before any allocation ("unavailable", not an R3-A hang)."""
    from types import SimpleNamespace

    from megatron.core.transformer.loop import param_store_comm as comm

    real = comm.copy_engine_api_check()
    assert real["torch"] == torch.__version__
    assert set(real) == {"symmetric_memory", "empty_strided_p2p", "barrier_timeout_ms", "torch"}
    if not real["symmetric_memory"]:
        pytest.skip("torch.distributed._symmetric_memory does not import in this build")

    class Missing:
        def barrier(self, channel=0):  # no timeout_ms
            pass

    class Complete:
        @staticmethod
        def empty_strided_p2p(size, stride, dtype, device, group_name=None, alloc_id=None):
            pass

        def barrier(self, channel=0, timeout_ms=0):
            pass

    monkeypatch.setattr(comm, "_symmetric_memory_class", lambda: Missing)
    check = comm.copy_engine_api_check()
    assert check["symmetric_memory"] and not check["empty_strided_p2p"]
    assert not check["barrier_timeout_ms"]
    reason = comm.copy_engine_unavailable_reason()
    assert "copy engines unavailable" in reason and "empty_strided_p2p" in reason
    assert "barrier(timeout_ms=)" in reason
    for gather, reduce in (("copy", "nccl"), ("nccl", "copy")):
        stand_in = SimpleNamespace(
            gather_engine=gather, reduce_engine=reduce, copy_engine=None, copy_reduce=None
        )
        with pytest.raises(RuntimeError, match="copy engines unavailable.*empty_strided_p2p"):
            StoreComm.setup_copy_engine(stand_in, (), torch.bfloat16)
    # The NCCL engines never check.
    StoreComm.setup_copy_engine(
        SimpleNamespace(gather_engine="nccl", reduce_engine="nccl", copy_engine=None),
        (),
        torch.bfloat16,
    )
    monkeypatch.setattr(comm, "_symmetric_memory_class", lambda: Complete)
    check = comm.copy_engine_api_check()
    assert check["empty_strided_p2p"] and check["barrier_timeout_ms"]
    assert comm.copy_engine_unavailable_reason() is None
    # The pybind form: no signature, the docstring's first line names the keywords.
    pybind = SimpleNamespace(
        __doc__="barrier(self: _SymmetricMemory, channel: int = 0, timeout_ms: int = 0) -> None"
    )
    assert comm._accepts_keyword(pybind, "timeout_ms")
    assert not comm._accepts_keyword(pybind, "group_name")


def test_copy_engine_timeout_config_validated():
    assert LoopParamStoreConfig(1).copy_engine_timeout_ms == 0
    with pytest.raises(ValueError, match="copy_engine_timeout_ms must be >= 0, got -1"):
        LoopParamStore.install(
            _model(),
            LoopParamStoreConfig(1, allow_trivial_shard=True, copy_engine_timeout_ms=-1),
            _groups(),
        )
    _, store = _store(copy_engine_timeout_ms=60_000)
    assert store.comm.copy_engine_timeout_ms == 60_000
    _, store = _store()
    assert store.comm.copy_engine_timeout_ms == 0
    with pytest.raises(ValueError, match="copy_engine_timeout_ms must be >= 0"):
        StoreComm(None, torch.device("cpu"), trivial=True, copy_engine_timeout_ms=-1)


def test_copy_engine_experimental_warning():
    """Selecting a copy engine warns (the store calls the helper at construction; a CPU store
    cannot use copy engines, so the helper is checked directly)."""
    import warnings

    from megatron.core.transformer.loop.param_store_comm import COPY_ENGINE_EXPERIMENTAL_NOTE

    assert "experimental" in COPY_ENGINE_EXPERIMENTAL_NOTE
    for engines in (dict(gather_engine="copy"), dict(reduce_engine="copy")):
        with pytest.warns(UserWarning, match="copy engines .* are experimental"):
            assert ps.warn_copy_engines_experimental(LoopParamStoreConfig(8, **engines))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert not ps.warn_copy_engines_experimental(LoopParamStoreConfig(8))


def test_store_construction_calls_the_experimental_warning(monkeypatch):
    """The store's constructor passes its config to ``warn_copy_engines_experimental`` (the
    helper above), so removing the call fails here."""
    calls = []
    monkeypatch.setattr(ps, "warn_copy_engines_experimental", calls.append)
    _, store = _store()
    assert len(calls) == 1 and calls[0] is store.config


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
