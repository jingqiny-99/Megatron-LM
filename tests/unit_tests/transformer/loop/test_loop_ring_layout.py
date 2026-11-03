# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Stage layout and roles of the ring schedule (``loop_schedule='ring'``: the hybrid loop over
pipeline stages).

* DSv4 Flash at P in {2, 4, 8}, k in {1, 2, 3, 4} and the parity mini at P in {2, 4}, k in
  {1, 2} (configs only; their DSv4 layers need Transformer Engine): the vendor's segment
  selection for every (pipeline rank, virtual-pipeline stage) through a P-rank stub group, the
  chunk roles of ``validate_ring_layout`` and of each chunk's ``resolve_hybrid_loop_ring_roles``
  (equal), the model roles (equal to PP = 1), the round-start / round-end chunks, the per-stage
  core split, the hash threshold, and the per-layer compress ratios: piped equal to pipe-free
  (the layer configs at the same global index, and
  ``normalize_dsv4_hybrid_csa_compress_ratios``).
* One chunk's rules (``resolve_hybrid_loop_ring_roles``), each alone.
* Built ring chunks (``LoopHybridStack`` of local layers, CPU): global layer numbers; the loop
  adapter only on the round-start chunk and the loop norm only on the round-end chunk (with and
  without mHC); ``is_repeated_layer`` exactly on the core modules that track
  ``is_first_microbatch``; every parameter of the PP = 1 stack once over the chunks; the
  tripwires (an eager call, per-row depths, carrier diagnostics); the PP = 1 stack unchanged.

Model parallelism is initialized once for the module (gloo on CPU, NCCL on CUDA).
"""

import argparse
import dataclasses
import functools

import pytest
import torch
import torch.distributed as dist

from megatron.core.models.hybrid.hybrid_layer_allocation import (
    get_layer_type_list_from_layer_config_list,
    select_pipeline_segment,
    validate_segment_layers,
)
from megatron.core.models.hybrid.hybrid_model import _get_hash_moe_layer_threshold
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.loop import guards
from megatron.core.transformer.loop.guards import LoopRoles
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.training.models.deepseek_v4 import normalize_dsv4_hybrid_csa_compress_ratios
from tests.unit_tests.transformer.loop import hybrid_loop_utils
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    HIDDEN,
    device,
    hybrid_loop_config,
    local_hybrid_submodules,
    model_parallel_once,
)
from tests.unit_tests.transformer.loop.ring_utils import (
    CODA,
    FLASH_MAIN,
    HASH_LAYERS,
    MINI_MAIN,
    MTP,
    PRELUDE,
    StageGroup,
    blocks,
    dsv4_loop_config,
    ring_fields,
    ring_layout,
    stage_group_ranks,
)

mhc_on_this_device = hybrid_loop_utils.mhc_on_this_device

_RING_PREFIX = "Loop ring does not support: "
_HYBRID_PREFIX = "Hybrid loop does not support: "
MHC = dict(enable_mhc_connections=True, mhc_num_residual_streams=4, mhc_sinkhorn_iterations=5)


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


def _reported(excinfo, prefix):
    message = str(excinfo.value)
    assert message.startswith(prefix), message
    return message[len(prefix) :].split(", ")


# ---------------------------------------------------------------------------------------------
# DSv4 Flash and the parity mini: layout, roles, compress ratios
# ---------------------------------------------------------------------------------------------

_GRID = [("flash", p, k) for p in (2, 4, 8) for k in (1, 2, 3, 4)] + [
    ("mini", p, k) for p in (2, 4) for k in (1, 2)
]


@pytest.mark.parametrize("name,stages,chunks", _GRID, ids=[f"{n}-P{p}k{k}" for n, p, k in _GRID])
def test_layout_and_roles(name, stages, chunks):
    main = FLASH_MAIN if name == "flash" else MINI_MAIN
    config = dsv4_loop_config(main, **ring_fields(stages, chunks))
    piped, segments = ring_layout(main, PRELUDE, CODA, stages, chunks)
    assert piped.replace("|", "") == main and piped.count("|") == stages * chunks - 1
    threshold = _get_hash_moe_layer_threshold(piped, HASH_LAYERS)
    assert threshold == _get_hash_moe_layer_threshold(main, HASH_LAYERS) == PRELUDE
    model_roles = guards.resolve_hybrid_loop_roles(config, list(main), threshold)
    assert model_roles == LoopRoles(PRELUDE, len(main) - PRELUDE - CODA, CODA)

    chunks_found = guards.validate_ring_layout(config, piped, threshold)
    assert len(chunks_found) == stages * chunks
    world = dist.group.WORLD
    plain_configs = validate_segment_layers(main, config)
    plain_ratios = [getattr(lc, "compress_ratio", None) for lc in plain_configs]
    per_stage = [0] * stages
    with stage_group_ranks():
        for segment, chunk in zip(segments, chunks_found):
            # The vendor's own selection for pipeline rank p, virtual-pipeline stage j.
            layer_configs, offset = select_pipeline_segment(
                piped,
                config,
                StageGroup(segment["stage"], stages),
                segment["chunk"] if chunks > 1 else None,
                tp_group=world,
                dp_cp_group=world,
            )
            symbols = get_layer_type_list_from_layer_config_list(layer_configs)
            assert "".join(symbols) == segment["pattern"] and offset == segment["offset"]
            # Per-layer compress ratios: piped == pipe-free at the same global index.
            ratios = [getattr(lc, "compress_ratio", None) for lc in layer_configs]
            assert ratios == plain_ratios[offset : offset + len(symbols)]
            # The stack's own resolution (from the chunk's symbols and offset) agrees.
            roles, own = guards.resolve_hybrid_loop_ring_roles(config, symbols, offset, threshold)
            assert roles == model_roles and own == chunk
            assert chunk.offset == offset
            assert chunk.round_start == (segment["index"] == 0)
            assert chunk.round_end == (segment["index"] == stages * chunks - 1)
            per_stage[segment["stage"]] += len(
                blocks(symbols[chunk.local.core_start : chunk.local.core_end])
            )
    # The chunks cover the model: one prelude, one coda, the core once.
    assert sum(c.local.prelude for c in chunks_found) == PRELUDE
    assert sum(c.local.coda for c in chunks_found) == CODA
    assert sum(c.local.core for c in chunks_found) == model_roles.core
    assert chunks_found[0].local.prelude == PRELUDE and chunks_found[-1].local.coda == CODA
    # The design's core balance for Flash (38 core layers); the mini's 8 split evenly.
    flash_split = {2: [19, 19], 4: [9, 10, 10, 9], 8: [4, 5, 5, 5, 5, 5, 5, 4]}
    assert sum(per_stage) == model_roles.core // 2
    if name == "flash":
        assert per_stage == flash_split[stages]
    assert max(per_stage) - min(per_stage) <= 1
    # normalize_dsv4_hybrid_csa_compress_ratios: the same per-sublayer list piped and pipe-free.
    lists = []
    for pattern in (piped + "/" + MTP, main + "/" + MTP):
        kwargs = {"experimental_attention_variant": "dsv4_hybrid"}
        args = argparse.Namespace(
            experimental_attention_variant="dsv4_hybrid",
            csa_compress_ratios=[r for r, s in zip(config.csa_compress_ratios, main + MTP)
                                 if s != "E"],
        )  # fmt: skip
        normalize_dsv4_hybrid_csa_compress_ratios(args, kwargs, pattern)
        lists.append(kwargs["csa_compress_ratios"])
    assert lists[0] == lists[1] == list(config.csa_compress_ratios)


def test_flash_layout_of_the_design():
    """P 4, k 1: the stage cut of VERIFY.md (9 / 10 / 10 / 9 core layers)."""
    piped, _ = ring_layout(FLASH_MAIN, PRELUDE, CODA, 4, 1)
    assert piped == (
        "WEWECEHECEHECEHECEHECEHE|CEHECEHECEHECEHECEHE|CEHECEHECEHECEHECEHE|"
        "CEHECEHECEHECEHECEHECE"
    )


# ---------------------------------------------------------------------------------------------
# One chunk's rules (resolve_hybrid_loop_ring_roles), each alone
# ---------------------------------------------------------------------------------------------

# Dense: '*-' x 6 (prelude 2, core 8, coda 2) at P 2, k 1.
_DENSE = "*-" * 6


def _dense_ring_config(**overrides):
    return hybrid_loop_config(_DENSE, 2, 2, loop_iterations=4, **ring_fields(2, 1), **overrides)


@pytest.mark.parametrize(
    "prefix,rule,symbols,offset,threshold,overrides",
    [
        (_RING_PREFIX, guards.RING_CHUNK_RANGE, "*-*-", 10, None, {}),
        (_RING_PREFIX, guards.RING_CHUNK_RANGE, "*-", -2, None, {}),
        (_RING_PREFIX, guards.RING_SEGMENT_WITHOUT_CORE, "", 4, None, {}),
        (_RING_PREFIX, guards.RING_SEGMENT_WITHOUT_CORE, "*-", 0, None, {}),  # prelude only
        (_RING_PREFIX, guards.RING_SEGMENT_WITHOUT_CORE, "*-", 10, None, {}),  # coda only
        (_RING_PREFIX, guards.RING_CUT_NOT_ON_BLOCK, "-*-*-", 5, None, {}),  # starts mid-block
        (_RING_PREFIX, guards.RING_CUT_NOT_ON_BLOCK, "*-*-*", 0, None, {}),  # ends mid-block
        (_RING_PREFIX, guards.RING_PRELUDE_SPLIT, "*-*-*-", 2, None, dict(prelude=4)),
        (_RING_PREFIX, guards.RING_CODA_SPLIT, "*-*-*-", 4, None, dict(coda=4)),
        (_HYBRID_PREFIX, guards.UNSUPPORTED_SUBLAYERS, "M-*-", 4, None, {}),
        (_HYBRID_PREFIX, guards.HASH_BEYOND_PRELUDE, "*-*-*-", 0, 4, {}),
        (_HYBRID_PREFIX, guards.CUT_NOT_ON_BLOCK, "*-*-*-", 0, None, dict(prelude=3)),
        (_HYBRID_PREFIX, guards.ROLES_DO_NOT_MATCH, "*-*-", 0, None, dict(prelude=6, coda=6)),
    ],
)
def test_chunk_rule_is_reported_alone(prefix, rule, symbols, offset, threshold, overrides):
    config = _dense_ring_config()
    # Role fields set directly (no __post_init__): the rule is the chunk resolver's.
    config.loop_prelude_sublayers = overrides.get("prelude", 2)
    config.loop_coda_sublayers = overrides.get("coda", 2)
    with pytest.raises(ValueError) as excinfo:
        guards.resolve_hybrid_loop_ring_roles(config, list(symbols), offset, threshold)
    assert _reported(excinfo, prefix) == [rule]


# ---------------------------------------------------------------------------------------------
# Built ring chunks (local layers, CPU)
# ---------------------------------------------------------------------------------------------


class _TrackedColumnLinear(ColumnParallelLinear):
    """A local column-parallel linear that tracks ``is_first_microbatch`` as TE linears do."""

    is_first_microbatch = True


def _tracked_submodules():
    """``local_hybrid_submodules`` with the MLP's fc1 a ``_TrackedColumnLinear`` (the local MLP
    is a ``functools.partial`` of ``MLP.as_mlp_submodule`` over its ``MLPSubmodules``)."""
    submodules = local_hybrid_submodules()
    mlp_layer = submodules.mlp_layer
    mlp = mlp_layer.submodules.mlp
    keywords = dict(mlp.keywords)
    keywords["submodules"] = dataclasses.replace(
        keywords["submodules"], linear_fc1=_TrackedColumnLinear
    )
    tracked = functools.partial(mlp.func, *mlp.args, **keywords)
    submodules.mlp_layer = dataclasses.replace(
        mlp_layer, submodules=dataclasses.replace(mlp_layer.submodules, mlp=tracked)
    )
    return submodules


# The parity mini's role structure in local dense layers: prelude 3 blocks, core 8, coda 2.
_MINI_DENSE = "*-" * 13


def _stack(config, pattern, offset, first, last, seed=1234):
    torch.manual_seed(seed)
    stack = LoopHybridStack(
        config=config,
        submodules=_tracked_submodules(),
        layer_config_list=validate_segment_layers(pattern, config),
        pp_layer_offset=offset,
        pre_process=first,
        post_process=last,
        post_layer_norm=False,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    return stack.to(device())


def _ring_chunks(stages, chunks, **overrides):
    """The ring config and the built chunk stacks (in segment order) of ``_MINI_DENSE``."""
    config = hybrid_loop_config(
        _MINI_DENSE, PRELUDE, CODA, loop_iterations=4, **ring_fields(stages, chunks), **overrides
    )
    _, segments = ring_layout(_MINI_DENSE, PRELUDE, CODA, stages, chunks)
    last = len(segments) - 1
    stacks = [
        _stack(config, s["pattern"], s["offset"], s["index"] == 0, s["index"] == last)
        for s in segments
    ]
    return config, segments, stacks


def _plain_stack(**overrides):
    config = hybrid_loop_config(_MINI_DENSE, PRELUDE, CODA, loop_iterations=4, **overrides)
    return _stack(config, _MINI_DENSE, 0, True, True)


def _tracked_modules(layer):
    return [m for m in layer.modules() if hasattr(m, "is_first_microbatch")]


@pytest.mark.parametrize("mhc", [False, True], ids=["plain", "mhc"])
@pytest.mark.parametrize("stages,chunks", [(2, 1), (2, 2), (4, 1), (4, 2)])
def test_built_chunks(request, stages, chunks, mhc):
    extra = MHC if mhc else {}
    if mhc:
        request.getfixturevalue("mhc_on_this_device")
    config, segments, stacks = _ring_chunks(stages, chunks, **extra)
    plain = _plain_stack(**extra)
    layout = guards.validate_ring_layout(config, "|".join(s["pattern"] for s in segments), None)
    streams = (4,) if mhc else ()
    for segment, stack, chunk in zip(segments, stacks, layout):
        assert stack.loop_ring == chunk
        assert stack.loop_roles == plain.loop_roles == LoopRoles(PRELUDE, 16, CODA)
        offset = segment["offset"]
        assert [layer.layer_number for layer in stack.layers] == list(
            range(offset + 1, offset + len(stack.layers) + 1)
        )
        # The adapter on the round-start chunk only, the norm on the round-end chunk only.
        if chunk.round_start:
            assert tuple(stack.loop_adapter.weight.shape) == (HIDDEN, 2 * HIDDEN)
        else:
            assert stack.loop_adapter is None
        if chunk.round_end:
            assert tuple(stack.loop_norm.weight.shape) == streams + (HIDDEN,)
        else:
            assert stack.loop_norm is None
        # Repeated layers: exactly the tracked modules of the core sublayers.
        local = chunk.local
        for index, layer in enumerate(stack.layers):
            repeated = local.core_start <= index < local.core_end
            for module in _tracked_modules(layer):
                assert getattr(module, "is_repeated_layer", False) == repeated, (index, module)
    assert sum(bool(_tracked_modules(layer)) for s in stacks for layer in s.layers) == 13
    # PP = 1: nothing marked; adapter and norm on the one stack.
    assert plain.loop_ring is None
    assert not any(
        getattr(m, "is_repeated_layer", False) for layer in plain.layers for m in layer.modules()
    )
    assert plain.loop_adapter is not None and plain.loop_norm is not None
    # Every PP = 1 parameter exactly once over the chunks, by global name and shape.
    plain_params = {name: tuple(p.shape) for name, p in plain.named_parameters()}
    ring_params = {}
    for segment, stack in zip(segments, stacks):
        for name, param in stack.named_parameters():
            if name.startswith("layers."):
                _, index, rest = name.split(".", 2)
                name = f"layers.{int(index) + segment['offset']}.{rest}"
            assert name not in ring_params, name
            ring_params[name] = tuple(param.shape)
    assert ring_params == plain_params


def test_ring_chunk_tripwires():
    _, _, stacks = _ring_chunks(2, 1)
    first = stacks[0]
    hidden = torch.randn(hybrid_loop_utils.SEQ, hybrid_loop_utils.BATCH, HIDDEN).to(device())
    with pytest.raises(ValueError) as excinfo:
        first(hidden, attention_mask=None)
    assert _reported(excinfo, _RING_PREFIX) == [guards.RING_EAGER_CALL]
    with pytest.raises(ValueError) as excinfo:
        first.set_loop_depths(torch.tensor([1, 2], device=device()))
    assert _reported(excinfo, _RING_PREFIX) == [guards.RING_ROW_DEPTHS]
    first.set_loop_depths(None)
    with pytest.raises(ValueError) as excinfo:
        first.enable_carrier_diagnostics(True)
    assert _reported(excinfo, _RING_PREFIX) == [guards.RING_CARRIER_DIAGNOSTICS]
    first.enable_carrier_diagnostics(False)


def test_ring_stack_runs_the_ring_config_guard():
    """A ring config that skipped __post_init__'s guard (e.g. edited after construction) is
    refused by the stack itself."""
    config = hybrid_loop_config(_MINI_DENSE, PRELUDE, CODA, **ring_fields(2, 1))
    config.loop_sample_depth = True
    with pytest.raises(ValueError) as excinfo:
        _stack(config, "*-*-*-*-*-*-*-", 0, True, False)
    assert _reported(excinfo, _RING_PREFIX) == [guards.RING_SAMPLE_DEPTH]
