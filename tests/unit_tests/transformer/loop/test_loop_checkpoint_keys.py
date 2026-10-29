# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint keys of the hybrid loop (M3c, ``loop/param_store_checkpoint.py``,
``megatron/training/loop_checkpoint.py``).

Local-layer loop models (CPU on gloo with one rank, or CUDA): the Huginn dense model
(``*-*-*-*-``, prelude 2, core 4, coda 2), Huginn with 4 mHC streams and an MTP layer
(``*-*-*-*-/*-``; on CPU this case needs ``TORCHDYNAMO_DISABLE=1``) and masked MoR.

* Golden keys: every entry of the z = 1 sharded state dict (dict key, ``ShardedTensor.key``,
  global shape and offset, dtype, replica id) equals ``loop_checkpoint_keys_golden.json``,
  recorded at ``cfb14081d`` (stage 7) before any M3c change; replica ids are compared on a
  one-rank world only (they carry the data-parallel rank). After a deliberate key change, re-record
  with ``LOOP_CHECKPOINT_KEYS_RECORD=1`` (the cases then skip).
* Coverage: no state tensor without an entry (sharded and legacy state dicts, tied embeddings).
* Later loop parameters: a ``LoopHybridStack`` subclass registering a direct ``Parameter``, a
  child ``nn.Linear``, an ``nn.ParameterList``, a persistent and a non-persistent buffer and a
  parameter inside a layer gets an entry for each (the non-persistent buffer none) under its
  module path, and a torch_dist round trip restores each bitwise, with no new key code.
* The backstop: a child whose custom ``sharded_state_dict`` drops a parameter is named by
  ``uncovered_state_names`` and ``check_loop_sharded_state_dict`` raises; views and disjoint row
  slices of a parameter cover it (storage rule), a copy or a partial slice does not; an
  ``_extra_state`` returned as a uint8 tensor is skipped; the store's hidden compute parameters
  count and its mirror shards do not; non-loop models are a no-op.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import (
    ShardedObject,
    ShardedTensor,
    ShardedTensorFactory,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.loop.param_store_checkpoint import (
    RowShard,
    named_state_tensors,
    row_shard_entry,
    uncovered_state_names,
)
from megatron.training.loop_checkpoint import (
    COVERAGE_CHECKED_KEY,
    check_loop_sharded_state_dict,
    is_loop_model,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.transformer.loop import hybrid_loop_utils
from tests.unit_tests.transformer.loop import param_store_utils as psu
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    HIDDEN,
    SEQ,
    VOCAB,
    build_local_loop_model,
    dense_config,
    local_model_spec,
    model_parallel_once,
)

mhc_on_this_device = hybrid_loop_utils.mhc_on_this_device
cpu_mtp = hybrid_loop_utils.cpu_mtp

DENSE = "*-*-*-*-"
MHC = dict(enable_mhc_connections=True, mhc_num_residual_streams=4, mhc_sinkhorn_iterations=5)
GOLDEN = Path(__file__).with_name("loop_checkpoint_keys_golden.json")
RECORD = os.environ.get("LOOP_CHECKPOINT_KEYS_RECORD") == "1"


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


def _model(name, request, module=LoopHybridStack, **overrides):
    """The model a case name asks for (``dense``, ``mhc_mtp``, ``mor``, ``tied``)."""
    pattern = DENSE
    if name == "mhc_mtp":
        request.getfixturevalue("mhc_on_this_device")
        request.getfixturevalue("cpu_mtp")
        pattern = DENSE + "/*-"
        overrides.update(MHC)
    elif name == "mor":
        overrides.update(loop_architecture="mor", loop_token_execution="masked")
    elif name == "tied":
        overrides.update(share_embeddings_and_output_weights=True)
    return build_local_loop_model(pattern, module, **overrides)


def _replica(replica_id):
    return list(replica_id) if isinstance(replica_id, tuple) else replica_id


def _describe(sharded_state_dict):
    """``{dict key: [kind, key, shape, offset, dtype, replica id]}`` (JSON-ready)."""
    out = {}
    for name, value in sorted(sharded_state_dict.items()):
        if isinstance(value, ShardedTensor):
            out[name] = [
                "ShardedTensor",
                value.key,
                list(value.global_shape),
                list(value.global_offset),
                str(value.dtype),
                _replica(value.replica_id),
            ]
        elif isinstance(value, ShardedTensorFactory):
            out[name] = [
                "ShardedTensorFactory",
                value.key,
                list(value.data.shape),
                None,
                str(value.data.dtype),
                _replica(value.replica_id),
            ]
        elif isinstance(value, ShardedObject):
            out[name] = [
                "ShardedObject",
                value.key,
                list(value.global_shape),
                list(value.global_offset),
                None,
                _replica(value.replica_id),
            ]
        else:
            out[name] = [type(value).__name__]
    return out


def _dumps(golden):
    """The golden file's text: one line per entry."""
    cases = []
    for case in sorted(golden):
        rows = [
            f"  {json.dumps(key)}: {json.dumps(entry)}"
            for key, entry in sorted(golden[case].items())
        ]
        cases.append(f" {json.dumps(case)}: {{\n" + ",\n".join(rows) + "\n }")
    return "{\n" + ",\n".join(cases) + "\n}\n"


def _state(module):
    """Every state tensor of ``module`` by its first name (cloned)."""
    return {names[0]: t.detach().clone() for t, names in named_state_tensors(module)}


# ---------------------------------------------------------------------------------------------
# Golden keys and coverage of today's loop models
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["dense", "mhc_mtp", "mor"])
def test_golden_keys(request, name):
    observed = _describe(_model(name, request).sharded_state_dict())
    if RECORD:
        golden = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        golden[name] = observed
        if not dist.is_initialized() or dist.get_rank() == 0:
            GOLDEN.write_text(_dumps(golden))
        pytest.skip(f"recorded {name} into {GOLDEN.name}")
    expected = json.loads(GOLDEN.read_text())[name]
    assert sorted(observed) == sorted(expected)
    one_rank = not dist.is_initialized() or dist.get_world_size() == 1
    for key, entry in expected.items():
        actual = observed[key]
        if one_rank:
            assert actual == entry, key
        else:
            assert actual[:-1] == entry[:-1], key


@pytest.mark.parametrize("name", ["dense", "mhc_mtp", "mor", "tied"])
def test_every_state_tensor_has_an_entry(request, name):
    model = _model(name, request)
    sharded = model.sharded_state_dict()
    # The walk reads the modules directly: the same tensors as state_dict(keep_vars=True).
    walked = {name: t for t, names in named_state_tensors(model) for name in names}
    plain = {
        name: t
        for name, t in model.state_dict(keep_vars=True).items()
        if not name.endswith("_extra_state")
    }
    assert walked.keys() == plain.keys()
    assert all(walked[name] is plain[name] for name in plain)
    assert uncovered_state_names(model, sharded) == []
    assert uncovered_state_names(model, model.state_dict()) == []
    check_loop_sharded_state_dict(model, when="setup")
    # Dropping any loop parameter's entry is caught, by name.
    loop_keys = [key for key in sharded if key.startswith(("decoder.loop_", "decoder.depth_"))]
    assert len(loop_keys) == 2
    for key in loop_keys:
        partial = {k: v for k, v in sharded.items() if k != key}
        assert uncovered_state_names(model, partial) == [key]
    hidden = model.config.hidden_size
    if name == "mor":
        assert tuple(sharded["decoder.depth_router.0.weight"].global_shape) == (hidden, hidden)
        assert tuple(sharded["decoder.depth_router.2.weight"].global_shape) == (
            model.config.loop_iterations,
            hidden,
        )
    else:
        assert tuple(sharded["decoder.loop_adapter.weight"].global_shape) == (hidden, 2 * hidden)
        streams = (4,) if name == "mhc_mtp" else ()
        assert tuple(sharded["decoder.loop_norm.weight"].global_shape) == streams + (hidden,)
    if name == "mhc_mtp":
        # dev's mHC key nesting: the wrapped layer under inner_layer., the mHC beside it.
        layers = model.decoder.layers
        for index in range(len(layers)):
            prefix = f"decoder.layers.{index}."
            assert any(key.startswith(prefix + "inner_layer.") for key in sharded), index
            assert any(key.startswith(prefix + "hyper_connection.") for key in sharded), index
    if name == "tied":
        assert model.share_embeddings_and_output_weights


# ---------------------------------------------------------------------------------------------
# Loop parameters added later get entries without new code
# ---------------------------------------------------------------------------------------------


class _CarrierStack(LoopHybridStack):
    """A stack registering parameters the way a later loop variant would (e.g. a carrier)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = self.config.hidden_size
        device = self.loop_adapter.weight.device
        self.carrier_gate = nn.Parameter(torch.randn(3, hidden, device=device))
        self.carrier_proj = nn.Linear(hidden, hidden, bias=False, device=device)
        self.carrier_scales = nn.ParameterList(
            [nn.Parameter(torch.randn(hidden, device=device)) for _ in range(2)]
        )
        self.register_buffer("carrier_stats", torch.randn(4, device=device))
        self.register_buffer("carrier_scratch", torch.randn(4, device=device), persistent=False)
        self.layers[2].carrier_scale = nn.Parameter(torch.randn(hidden, device=device))


CARRIER_KEYS = {
    "decoder.carrier_gate": (3, HIDDEN),
    "decoder.carrier_proj.weight": (HIDDEN, HIDDEN),
    "decoder.carrier_scales.0": (HIDDEN,),
    "decoder.carrier_scales.1": (HIDDEN,),
    "decoder.carrier_stats": (4,),
    "decoder.layers.2.carrier_scale": (HIDDEN,),
}


def test_later_loop_parameters_get_entries_and_round_trip(request, tmp_path_dist_ckpt):
    model = _model("dense", request, _CarrierStack)
    sharded = model.sharded_state_dict()
    for key, shape in CARRIER_KEYS.items():
        entry = sharded[key]
        assert isinstance(entry, ShardedTensor), key
        assert entry.key == key and tuple(entry.global_shape) == shape, key
        assert not any(entry.global_offset), key
    assert not any("carrier_scratch" in key for key in sharded)
    assert uncovered_state_names(model, sharded) == []

    saved = _state(model)
    with TempNamedDir(tmp_path_dist_ckpt / "loop_carrier_round_trip") as ckpt_dir:
        dist_checkpointing.save(sharded, ckpt_dir)
        other = _model("dense", request, _CarrierStack)
        generator = torch.Generator().manual_seed(99)
        with torch.no_grad():
            for tensor, _ in named_state_tensors(other):
                noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
                tensor.copy_(noise.to(tensor.dtype))
        scratch = other.decoder.carrier_scratch.clone()
        loaded = dist_checkpointing.load(other.sharded_state_dict(), ckpt_dir)
        other.load_state_dict(loaded, strict=True)
    restored = _state(other)
    assert restored.keys() == saved.keys()
    for name, tensor in saved.items():
        assert torch.equal(restored[name], tensor), name
    assert torch.equal(other.decoder.carrier_scratch, scratch)  # not checkpointed


# ---------------------------------------------------------------------------------------------
# The coverage backstop
# ---------------------------------------------------------------------------------------------


class _Emitter(nn.Module):
    """A child with a ``[4, h]`` weight whose ``sharded_state_dict`` emits ``mode``'s entries."""

    def __init__(self, mode, hidden, device):
        super().__init__()
        self.mode = mode
        self.weight = nn.Parameter(torch.randn(4, hidden, device=device))
        self.kept = nn.Parameter(torch.randn(hidden, device=device))

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Entries per ``mode`` (``kept`` always by itself)."""
        entries = {f"{prefix}kept": self.kept}
        weight = self.weight
        if self.mode == "view":
            entries[f"{prefix}flat"] = weight.view(-1)
        elif self.mode == "both":  # the parameter itself and an extra view of it
            entries[f"{prefix}weight"] = weight
            entries[f"{prefix}flat"] = weight.view(-1)
        elif self.mode == "row_slices":
            entries[f"{prefix}top"], entries[f"{prefix}bottom"] = weight[:1], weight[1:]
        elif self.mode == "transpose":
            entries[f"{prefix}transposed"] = weight.t()
        elif self.mode == "copy":
            entries[f"{prefix}copied"] = weight.detach().clone()
        elif self.mode == "partial":
            entries[f"{prefix}top"] = weight[:3]
        return {key: ShardedTensor.from_rank_offsets(key, value) for key, value in entries.items()}


class _ExtraStateChild(nn.Module):
    """A child whose extra state is a uint8 tensor (as TE's may be)."""

    def __init__(self, hidden, device):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden, device=device))

    def get_extra_state(self):
        """A uint8 tensor, not a ``BytesIO``."""
        return torch.zeros(3, dtype=torch.uint8)

    def set_extra_state(self, state):
        """Nothing to restore."""


def _stack_with(child_factory):
    class _Stack(LoopHybridStack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.carrier_cell = child_factory(
                self.config.hidden_size, self.loop_adapter.weight.device
            )

    return _Stack


@pytest.mark.parametrize(
    "mode,covered",
    [
        ("drop", False),
        ("view", True),
        ("row_slices", True),
        ("transpose", True),
        ("copy", False),
        ("partial", False),
    ],
)
def test_storage_rule_and_the_backstop(request, mode, covered):
    stack = _stack_with(lambda hidden, device: _Emitter(mode, hidden, device))
    model = _model("dense", request, stack)
    sharded = model.sharded_state_dict()
    missing = uncovered_state_names(model, sharded)
    if covered:
        assert missing == []
        check_loop_sharded_state_dict(model, when="save", sharded_state_dict=sharded)
        return
    assert missing == ["decoder.carrier_cell.weight"]
    message = (
        r"loop checkpoint \(save\): the sharded state dict has no entry for "
        r"decoder\.carrier_cell\.weight \(a parameter or persistent buffer"
    )
    with pytest.raises(RuntimeError, match=message):
        check_loop_sharded_state_dict(model, when="save", sharded_state_dict=sharded)
    with pytest.raises(RuntimeError, match=r"loop checkpoint \(setup\)"):
        check_loop_sharded_state_dict(model, when="setup")  # builds the dict itself


def test_the_save_check_runs_once_per_context(request):
    context = {}
    check_loop_sharded_state_dict(_model("dense", request), when="save", context=context)
    assert context == {COVERAGE_CHECKED_KEY: True}
    broken = _model("dense", request, _stack_with(lambda h, d: _Emitter("drop", h, d)))
    check_loop_sharded_state_dict(broken, when="save", context=context)  # cached: no rebuild
    fresh = {}
    with pytest.raises(RuntimeError, match="carrier_cell.weight"):
        check_loop_sharded_state_dict(broken, when="save", context=fresh)
    assert fresh == {}


def test_uint8_extra_state_is_skipped(request):
    model = _model("dense", request, _stack_with(_ExtraStateChild))
    sharded = model.sharded_state_dict()
    assert isinstance(sharded["decoder.carrier_cell._extra_state"], ShardedObject)
    assert "decoder.carrier_cell._extra_state" in model.state_dict(keep_vars=True)
    names = [name for _, names in named_state_tensors(model) for name in names]
    assert not any(name.endswith("_extra_state") for name in names)
    assert uncovered_state_names(model, sharded) == []


def test_store_hidden_parameters_count_and_mirror_shards_do_not(request):
    model = _model("dense", request)
    store = psu.install_store(model, psu.trivial_store_groups(), 1, allow_trivial_shard=True)
    stored = list(store.tensors())
    assert stored
    names = {name for _, names in named_state_tensors(model) for name in names}
    assert {tensor.name for tensor in stored} <= names
    assert not any("loop_param_store" in name for name in names)
    # Every non-stored tensor by its key; every stored one by an entry on its canonical shard.
    stored_names = {tensor.name for tensor in stored}
    by_key = {name: None for name in names - stored_names}
    by_shard = {f"shard.{i}": SimpleNamespace(data=t.shard) for i, t in enumerate(stored)}
    assert uncovered_state_names(model, {**by_key, **by_shard}) == []
    del by_shard["shard.0"]
    assert uncovered_state_names(model, {**by_key, **by_shard}) == [stored[0].name]


def test_non_loop_models_are_a_no_op():
    torch.manual_seed(1234)
    config = dense_config(len(DENSE), is_hybrid_model=True)
    plain = HybridModel(
        config=config,
        hybrid_stack_spec=local_model_spec(HybridStack),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        hybrid_layer_pattern=DENSE,
        position_embedding_type="none",
    ).to(hybrid_loop_utils.device())
    plain.decoder.carrier_cell = _Emitter("drop", config.hidden_size, hybrid_loop_utils.device())
    assert not is_loop_model(plain)
    assert uncovered_state_names(plain, plain.sharded_state_dict()) == [
        "decoder.carrier_cell.weight"
    ]
    for model in (plain, [plain], nn.Linear(2, 2), [plain, plain]):
        check_loop_sharded_state_dict(model, when="save")
        check_loop_sharded_state_dict(model, when="load", sharded_state_dict={})


def test_loop_models_are_found_through_wrappers(request):
    model = _model("dense", request)
    assert is_loop_model(model) and is_loop_model([model])
    assert not is_loop_model([model, model])


# ---------------------------------------------------------------------------------------------
# Stored tensors: the logical view (trivial store, z = 1)
# ---------------------------------------------------------------------------------------------


def _layer_stack_with(child_factory):
    """A stack whose layer 2 holds ``child_factory``'s module (a stored unit's owner)."""

    class _Stack(LoopHybridStack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.layers[2].carrier_cell = child_factory(
                self.config.hidden_size, self.loop_adapter.weight.device
            )

    return _Stack


def _trivial_store(model):
    return psu.install_store(model, psu.trivial_store_groups(), 1, allow_trivial_shard=True)


def _owners_parameters(store):
    """``{id(owner): {attr: parameter}}`` of every stored tensor's owner (a snapshot)."""
    # pylint: disable=protected-access
    return {id(t.module): dict(t.module._parameters) for t in store.tensors()}


def _built(entry):
    """The ``ShardedTensor`` a row-shard factory builds on its own data."""
    return entry.build_fn(entry.key, entry.data, entry.replica_id, None)


@pytest.mark.parametrize("name", ["dense", "mhc_mtp", "tied"])
def test_trivial_store_logical_view_keeps_the_store_free_keys(request, name):
    reference = _describe(_model(name, request).sharded_state_dict())
    model = _model(name, request)
    store = _trivial_store(model)
    stored = list(store.tensors())
    assert stored
    shards = {id(t.shard): t for t in stored}
    before = _owners_parameters(store)
    sharded = model.sharded_state_dict()
    assert _owners_parameters(store) == before  # the view leaves _parameters as it was
    assert sorted(sharded) == sorted(reference)
    assert not any(key.startswith("loop_param_store") for key in sharded)
    converted = []
    for key, entry in sharded.items():
        tensor = shards.get(id(getattr(entry, "data", None)))
        if tensor is None:
            assert _describe({key: entry})[key] == reference[key], key
            continue
        converted.append(tensor.name)
        assert isinstance(entry, ShardedTensorFactory), key
        assert entry.key == reference[key][1], key
        piece = _built(entry)
        assert isinstance(piece, ShardedTensor), key
        assert piece.key == entry.key, key
        assert list(piece.global_shape) == list(tensor.logical_shape), key
        assert list(piece.global_shape)[1:] == reference[key][2][1:], key
        assert not any(piece.global_offset), key
        assert str(piece.dtype) == reference[key][4] or reference[key][4] is None, key
        # The trivial store keeps the owners' replica ids (no replicate group of its own).
        assert _replica(entry.replica_id) == reference[key][-1], key
    assert sorted(converted) == sorted(t.name for t in stored)  # each converted once
    assert uncovered_state_names(model, sharded) == []
    assert set(store._ckpt_entry_keys) == {t.name for t in stored}  # pylint: disable=W0212


def test_decoder_view_converts_only_the_decoders_stored_tensors(request):
    model = _model("dense", request)
    store = _trivial_store(model)
    sharded = model.decoder.sharded_state_dict(prefix="decoder.")
    shards = {id(t.shard): t.name for t in store.tensors()}
    converted = sorted(
        shards[id(entry.data)]
        for entry in sharded.values()
        if isinstance(entry, ShardedTensorFactory) and id(entry.data) in shards
    )
    expected = sorted(t.name for t in store.tensors() if t.name.startswith("decoder."))
    outside = [t.name for t in store.tensors() if not t.name.startswith("decoder.")]
    assert converted == expected and outside  # the embedding / output: no error, no entry
    assert not store.in_checkpoint_view


def test_store_entry_outside_the_view_raises_and_the_view_restores_on_error(request):
    model = _model("dense", request)
    store = _trivial_store(model)
    with pytest.raises(RuntimeError, match="saved through HybridModel.sharded_state_dict"):
        store.sharded_state_dict()
    before = _owners_parameters(store)

    def broken():
        assert store.in_checkpoint_view
        raise KeyError("boom")

    with pytest.raises(KeyError, match="boom"):
        store.logical_sharded_state_dict(model, broken)
    assert _owners_parameters(store) == before and not store.in_checkpoint_view


@pytest.mark.parametrize("mode", ["view", "row_slices"])
def test_a_stored_weight_emitted_as_a_view_raises_message_8(request, mode):
    stack = _layer_stack_with(lambda hidden, device: _Emitter(mode, hidden, device))
    model = _model("dense", request, stack)
    store = _trivial_store(model)
    name = "decoder.layers.2.carrier_cell.weight"
    assert any(t.name == name for t in store.tensors())
    message = (
        r"loop parameter store: decoder\.layers\.2\.carrier_cell\.weight's checkpoint entry "
        r"decoder\.layers\.2\.carrier_cell\.\w+ is not the logical tensor in row order"
    )
    with pytest.raises(ValueError, match=message):
        model.sharded_state_dict()
    assert not any(t.attr in t.module._parameters for t in store.tensors())  # pylint: disable=W0212


def test_an_extra_view_of_a_bound_resident_weight_raises_message_8(request):
    """The parameter itself plus a view of it, with the Resident compute copies bound (the
    state after a step or ``refresh()``): the view must not save the copy's values under the
    extra key. The view unbinds the copies for its duration and restores them afterwards."""
    from megatron.core.transformer.loop.param_store import StoreModes
    from megatron.core.transformer.loop.param_store_binding import is_placeholder

    stack = _layer_stack_with(lambda hidden, device: _Emitter("both", hidden, device))
    model = _model("dense", request, stack)
    store = _trivial_store(model)
    store.configure_modes(StoreModes.parse("all=r"), core_plan=psu.core_plan_of("whole_round", 4))
    store.refresh()
    bound = {t.name: t.bound for t in store.tensors()}
    assert all(view is not None for view in bound.values())
    message = (
        r"loop parameter store: decoder\.layers\.2\.carrier_cell\.weight's checkpoint entry "
        r"decoder\.layers\.2\.carrier_cell\.flat is not the logical tensor in row order"
    )
    with pytest.raises(ValueError, match=message):
        model.sharded_state_dict()
    for tensor in store.tensors():  # bindings restored after the (failed) view
        assert tensor.bound is bound[tensor.name], tensor.name
        assert tensor.compute.data.data_ptr() == bound[tensor.name].data_ptr(), tensor.name
        assert not is_placeholder(tensor.compute.data), tensor.name


def test_row_shard_replica_ids_keep_alias_entries_non_main(request):
    """The writer election replaces the owner's data-parallel coordinate by the replicate rank;
    an owner entry marked non-main on its PP slot or on a middle coordinate (a tied / alias
    entry) stays non-main."""
    model = _model("dense", request)
    store = _trivial_store(model)
    tensor = next(iter(store.tensors()))
    shard = RowShard.of(tensor, 0)
    cases = {
        (0, 0, 5): (0, 0, 2),  # a main entry of DP rank 5: main iff replicate rank 0
        (0, 3, 0): (0, 1, 2),  # marked on a middle coordinate: kept non-main
        (1, 0, 0): (1, 0, 2),  # the PP slot is kept
    }
    for owner, expected in cases.items():
        entry = SimpleNamespace(key=tensor.name, replica_id=owner)
        assert row_shard_entry(entry, tensor, shard, 2).replica_id == expected, owner
        if owner == (0, 3, 0):
            assert row_shard_entry(entry, tensor, shard, 0).replica_id == (0, 1, 0)
    # The trivial store (no replicate group) keeps the owner's replica id.
    entry = SimpleNamespace(key=tensor.name, replica_id=(0, 3, 1))
    assert row_shard_entry(entry, tensor, shard, None).replica_id == (0, 3, 1)


def test_a_stored_weight_without_an_entry_raises_message_8(request):
    stack = _layer_stack_with(lambda hidden, device: _Emitter("drop", hidden, device))
    model = _model("dense", request, stack)
    _trivial_store(model)
    message = r"_Emitter emitted no entry for decoder\.layers\.2\.carrier_cell\.weight"
    with pytest.raises(ValueError, match=message):
        model.sharded_state_dict()


def test_swiglu_factory_converts_and_singleton_local_shards_raise(request):
    model = _model("dense", request)
    store = _trivial_store(model)
    fc1 = next(t for t in store.tensors() if t.name.endswith("mlp.linear_fc1.weight"))
    sharded = model.sharded_state_dict(metadata={"singleton_local_shards": False})
    entry = sharded[fc1.name]
    assert isinstance(entry, ShardedTensorFactory) and entry.data is fc1.shard
    assert tuple(_built(entry).global_shape) == fc1.logical_shape
    message = r"linear_fc1\.weight's checkpoint entry .* builds key .*_w"
    with pytest.raises(ValueError, match=message):
        model.sharded_state_dict(metadata={"singleton_local_shards": True})
