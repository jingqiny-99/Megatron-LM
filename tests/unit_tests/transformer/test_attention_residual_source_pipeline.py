# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU integration checks of the actual source/value pipeline lifecycle.

A fixed layer layout replaces only distributed layer ownership. Production source
state, value packing, cache taps and kernels execute unchanged. P2P boundaries
are distinct leaves; reverse chunk backward runs without retaining graphs. The
oracle is an unsplit, ordinary-autograd model with independent RMS/softmax math.
The synthetic package avoids unrelated CUDA-only MCore import dependencies.
"""

import ast
import importlib.util
import sys
import types
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[3]
_TRANSFORMER = _ROOT / 'megatron/core/transformer'


@pytest.fixture
def lifecycle(monkeypatch):
    package_name = '_attnres_source_pipeline_test'
    package = types.ModuleType(package_name)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package_name, package)

    def load(name):
        qualified = package_name + '.' + name
        spec = importlib.util.spec_from_file_location(qualified, _TRANSFORMER / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified, module)
        spec.loader.exec_module(module)
        return module

    kernels = load('attention_residual_projection_kernels')
    projection_calls = []
    original_project = kernels.project_source

    def project_source(*args, **kwargs):
        projection_calls.append(args[1].shape[0])
        return original_project(*args, **kwargs)

    monkeypatch.setattr(kernels, 'project_source', project_source)
    runtime = types.ModuleType(package_name + '.attention_residual_projection_runtime')
    runtime.get_attn_res_projection_runtime = lambda config: config._runtime
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    source = load('attention_residual_source_state')

    path = _TRANSFORMER / 'attention_residual.py'
    module = types.ModuleType(package_name + '.attention_residual')
    module.__package__ = package_name
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(
        torch=torch,
        Tensor=torch.Tensor,
        TransformerConfig=object,
        List=List,
        Optional=Optional,
        Sequence=Sequence,
        Tuple=Tuple,
        nvtx_range_push=lambda **_: None,
        nvtx_range_pop=lambda **_: None,
        _stage_layers_before=lambda config, stage: sum(config._layout[:stage]),
    )
    names = {
        'is_attn_res_block_start',
        'attn_res_num_sources',
        'attn_res_final_num_sources',
        'attn_res_num_payload_slices',
        '_sources_formed_through',
        'attn_res_boundary_delta_slices',
        'attn_res_uniform_payload_slices',
        'pack_attn_res_payload',
        'unpack_attn_res_payload',
        '_AttnResSourceCache',
        'get_attn_res_source_cache',
        'attn_res_source_cache_reset',
        '_AttnResGradTap',
        'attn_res_tap_source',
        'AttnResStageSources',
    }
    assignments = {'_UNIFORM_SLICES_MEMO', '_SOURCE_CACHE'}
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in assignments for target in node.targets
        ):
            selected.append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in assignments
        ):
            selected.append(node)
    assert {node.name for node in selected if hasattr(node, 'name')} == names
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return types.SimpleNamespace(
        value=module, source=source, kernels=kernels, projection_calls=projection_calls
    )


def _make_config(pp, vp, layout, block, fraction, bank, mtp):
    layers = sum(layout)
    metadata = [
        {
            'column': 2 * (layer - 1) + sublayer,
            'id': (0, 0, layer, sublayer),
            'stop_source_grad': False,
        }
        for layer in range(1, layers + 1)
        for sublayer in range(2)
    ]
    metadata.append(
        {'column': len(metadata), 'id': (0, 0, layers + 1, 0), 'stop_source_grad': False}
    )
    if mtp:
        metadata.append(
            {'column': len(metadata), 'id': (1, 1, layers + 1, 0), 'stop_source_grad': True}
        )
    return types.SimpleNamespace(
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vp,
        num_layers=layers,
        attn_res_block_layers=block,
        attn_res_source_projection_fraction=fraction,
        attn_res_impl='source',
        layernorm_epsilon=1e-6,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
        _layout=layout,
        _runtime=types.SimpleNamespace(bank=bank, consumer_metadata=metadata),
    )


@pytest.mark.parametrize("contiguous", [True, False])
def test_source_query_selection_preserves_rows_and_gradients(lifecycle, monkeypatch, contiguous):
    """Contiguous rows use a protected bank view; external ID layouts can still gather."""
    torch.manual_seed(618)
    bank = torch.randn(6, 16, requires_grad=True)
    value = torch.randn(3, 2, 16, requires_grad=True)
    config = _make_config(1, None, [4], 2, 1.0, bank, False)
    if contiguous:
        ids = [(0, 0, layer, slot) for layer in (1, 4) for slot in range(3)]
        columns = (3, 4, 5)
    else:
        # Sorted external semantic IDs can place past and future consumers in
        # alternating rows when their second ID component differs.
        ids = [(0, group, layer, 0) for group in range(3) for layer in (1, 4)]
        columns = (1, 3, 5)
    config._runtime.consumer_metadata = [
        dict(column=index, id=identifier, stop_source_grad=False)
        for index, identifier in enumerate(ids)
    ]
    state = lifecycle.source.ProjectionSourceState(
        types.SimpleNamespace(config=config, interleaved=False), layers_before=2
    )
    captured = {}
    original_project = lifecycle.kernels.project_source
    original_tensor = torch.tensor
    index_constructions = []

    def observe_project(source, queries, *args, **kwargs):
        captured["queries"] = queries
        return original_project(source, queries, *args, **kwargs)

    def observe_tensor(data, *args, **kwargs):
        if isinstance(data, tuple) and data == columns and kwargs.get("dtype") == torch.long:
            index_constructions.append(data)
            assert not contiguous, "Contiguous query rows must not allocate an index tensor"
        return original_tensor(data, *args, **kwargs)

    monkeypatch.setattr(lifecycle.kernels, "project_source", observe_project)
    monkeypatch.setattr(torch, "tensor", observe_tensor)
    projected = state.complete(value, source_id=1)
    record = projected._attn_res_source_projection
    assert record.columns == columns
    queries = captured["queries"]
    torch.testing.assert_close(queries, bank[list(columns)], atol=0, rtol=0)
    assert queries._do_not_offload
    assert index_constructions == ([] if contiguous else [columns])
    if contiguous:
        assert queries.untyped_storage().data_ptr() == bank.untyped_storage().data_ptr()

    upstream = torch.randn_like(record.scores)
    record.scores.backward(upstream)
    # An independent FP64 projection verifies both selected-row values and the
    # mapping of their VJPs back into the canonical bank, including untouched rows.
    reference_bank = bank.detach().double().requires_grad_()
    wide = value.detach().double()
    normalized = wide * (wide.square().mean(-1, keepdim=True) + config.layernorm_epsilon).rsqrt()
    reference = (normalized.unsqueeze(-2) * reference_bank[list(columns)]).sum(-1)
    reference.backward(upstream.double())
    torch.testing.assert_close(record.scores.double(), reference, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(bank.grad.double(), reference_bank.grad, atol=2e-5, rtol=2e-4)
    unselected = [index for index in range(bank.shape[0]) if index not in columns]
    assert torch.count_nonzero(bank.grad[unselected]) == 0
    assert value.grad is None  # Producer scores retain only the query-gradient path.


def _independent_mix(values, query, eps):
    """Independent ordinary-autograd oracle; no production forward/backward calls."""
    stacked = torch.stack(values).float()
    rstd = (stacked.square().mean(dim=-1) + eps).rsqrt()
    scores = (stacked * query).sum(dim=-1) * rstd
    probabilities = scores.softmax(dim=0)
    mixed = (probabilities[..., None] * stacked).sum(dim=0)
    return mixed.to(values[0].dtype)


def _sublayer(mixed, update, layer, sublayer):
    # A small nonlinear stand-in gives every source and layer a gradient without
    # sharing a production model implementation with the independent oracle.
    return (
        torch.tanh(mixed.float() * (0.15 + 0.01 * sublayer) + update) * (0.7 + 0.01 * layer)
    ).to(mixed.dtype)


def _run_unsplit(config, embedding, updates, direction, mtp):
    bank = config._runtime.bank
    partial = embedding
    values = []
    for layer in range(1, config.num_layers + 1):
        starts = (layer - 1) % config.attn_res_block_layers == 0
        if starts:
            values.append(partial)
        for sublayer in range(2):
            inputs = values if starts and sublayer == 0 else values + [partial]
            column = 2 * (layer - 1) + sublayer
            mixed = _independent_mix(inputs, bank[column], config.layernorm_epsilon)
            update = _sublayer(mixed, updates[layer - 1, sublayer], layer, sublayer)
            partial = update if starts and sublayer == 0 else partial + update
    values = values + [partial]
    output = _independent_mix(values, bank[2 * config.num_layers], config.layernorm_epsilon)
    loss = (output.float() * direction).sum()
    outputs = [output]
    if mtp:
        # Trunk activations stop here while the MTP pseudo-query remains trainable.
        mtp_output = _independent_mix(
            [value.detach() for value in values], bank[-1], config.layernorm_epsilon
        )
        loss = loss + 0.37 * (mtp_output.float() * direction.flip(-1)).sum()
        outputs.append(mtp_output)
    loss.backward()
    return outputs


def _run_pipeline(api, config, embedding, updates, direction, mtp):
    bank = config._runtime.bank
    value_api, source_api = api.value, api.source
    pp = config.pipeline_model_parallel_size
    vp = config.virtual_pipeline_model_parallel_size
    rank_caches = [value_api._AttnResSourceCache() for _ in range(pp)]
    stage_inputs, stage_outputs, cache_leaves = [], [], []
    boundary_completed = 0
    local_scores = precomputed_scores = 0
    all_values = []

    def aggregate(values, column):
        nonlocal local_scores, precomputed_scores
        logits = [source_api.source_logits_for_consumer(value, column) for value in values]
        local_scores += sum(score is None for score in logits)
        precomputed_scores += sum(score is not None for score in logits)
        return api.kernels.aggregate_preprojected(
            values, bank[column], logits, eps=config.layernorm_epsilon, precomputed_value_grad=True
        )

    carried = None
    before = 0
    for stage, layer_count in enumerate(config._layout):
        rank, chunk = stage % pp, stage // pp
        value_api._SOURCE_CACHE = rank_caches[rank]
        if stage == 0:
            incoming = embedding
            stage_inputs.append(None)
        else:
            incoming = [tensor.detach().requires_grad_(True) for tensor in carried]
            stage_inputs.append(incoming)
        state, partial = value_api.AttnResStageSources.enter(
            config,
            incoming,
            layers_before=before,
            pp_rank=rank,
            vp_stage=chunk if vp else None,
            microbatch_id=7 if vp else None,
            pre_process=stage == 0,
        )
        assert isinstance(state.projection, source_api.ProjectionSourceState)
        if stage and before % config.attn_res_block_layers == 0:
            record = partial._attn_res_source_projection
            assert record.source_id == before // config.attn_res_block_layers
            boundary_completed += 1
        for layer in range(before + 1, before + layer_count + 1):
            starts = value_api.is_attn_res_block_start(layer, config.attn_res_block_layers)
            if starts:
                state.append_block_start(partial)
            for sublayer in range(2):
                values = (
                    state.graph_sources
                    if starts and sublayer == 0
                    else [*state.graph_sources, partial]
                )
                mixed = aggregate(values, 2 * (layer - 1) + sublayer)
                update = _sublayer(mixed, updates[layer - 1, sublayer], layer, sublayer)
                partial = update if starts and sublayer == 0 else partial + update
            partial = state.after_layer(partial, layer)
        before += layer_count
        all_values.extend(state.graph_sources)
        cache_leaves.extend(state._cache_leaves)
        cache_leaves.extend(record.scores for record in state.projection.cache_records.values())
        if stage == len(config._layout) - 1:
            values = state.exit_aggregate_values(partial)
            output = aggregate(values, 2 * config.num_layers)
            outputs = [output]
            loss = (output.float() * direction).sum()
            if mtp:
                detached = [source_api.detach_attn_res_source(value) for value in values]
                for old, new in zip(values, detached):
                    assert new._attn_res_source_projection is old._attn_res_source_projection
                    assert not new.requires_grad
                    assert new._do_not_offload
                mtp_output = aggregate(detached, len(bank) - 1)
                loss = loss + 0.37 * (mtp_output.float() * direction.flip(-1)).sum()
                outputs.append(mtp_output)
        else:
            carried = state.exit_pack(partial)
            assert isinstance(carried, list) and len(carried) == 2
            assert carried[0].dtype == embedding.dtype
            assert carried[1].dtype == torch.float32
            assert all(tensor._base is None for tensor in carried)
            next_rank, next_chunk = (stage + 1) % pp, (stage + 1) // pp
            layout = source_api._boundary_layout(config, next_rank, next_chunk)
            rows = carried[1].reshape(-1, *embedding.shape[:2])
            assert torch.count_nonzero(rows[len(layout) :]) == 0
            if config.attn_res_source_projection_fraction == 0:
                assert not layout
                assert carried[1].requires_grad
                assert torch.count_nonzero(carried[1]) == 0
            stage_outputs.append(carried)
    assert all(not cache.sources and not cache.projections for cache in rank_caches)
    assert all(getattr(value, '_do_not_offload', False) for value in all_values)
    loss.backward()
    for stage in range(len(stage_outputs) - 1, -1, -1):
        incoming = stage_inputs[stage + 1]
        # The padded zero score channel can be unused locally. Sending an
        # explicit zero is the pipeline's correct mathematical VJP in that case.
        grads = [
            tensor.grad if tensor.grad is not None else torch.zeros_like(tensor)
            for tensor in incoming
        ]
        torch.autograd.backward(stage_outputs[stage], grads)
    undrained = [leaf for leaf in cache_leaves if leaf.grad is not None]
    assert local_scores > 0  # Running partials still execute local scoring.
    if config.attn_res_source_projection_fraction:
        assert precomputed_scores > 0
    else:
        assert precomputed_scores == 0
    return outputs, boundary_completed, undrained


_LAYOUTS = [
    pytest.param(2, None, (4, 4), 2, id='pp2-aligned'),
    pytest.param(4, None, (3, 3, 3, 3), 2, id='pp4-crossing'),
    pytest.param(2, 2, (2,) * 4, 2, id='pp2-vpp2-aligned'),
    pytest.param(4, 2, (2,) * 8, 3, id='pp4-vpp2-crossing'),
    pytest.param(2, 4, (2,) * 8, 3, id='pp2-vpp4-crossing'),
    pytest.param(4, 4, (2,) * 16, 2, id='pp4-vpp4-aligned'),
]


@pytest.mark.parametrize('pp,vp,layout,block', _LAYOUTS)
@pytest.mark.parametrize('fraction', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('mtp', [False, True])
def test_source_pipeline_matches_unsplit(lifecycle, pp, vp, layout, block, fraction, mtp):
    _assert_pipeline(lifecycle, pp, vp, layout, block, fraction, mtp, torch.float32)


@pytest.mark.parametrize('fraction', [0.0, 0.5, 1.0])
def test_bf16_source_pipeline_matches_unsplit(lifecycle, fraction):
    _assert_pipeline(lifecycle, 2, 2, (2,) * 4, 3, fraction, True, torch.bfloat16)


def _assert_pipeline(api, pp, vp, layout, block, fraction, mtp, dtype):
    generator = torch.Generator().manual_seed(20260923)
    layers, hidden = sum(layout), 32
    embedding = torch.randn(3, 2, hidden, generator=generator, dtype=torch.float32).to(dtype)
    updates = torch.randn(layers, 2, hidden, generator=generator) * 0.2
    bank = torch.randn(2 * layers + 1 + int(mtp), hidden, generator=generator) * 0.03
    # A token-mean scalar objective matches training loss scaling; standalone
    # kernel tests cover arbitrary, unnormalized incoming gradients separately.
    direction = torch.randn(3, 2, hidden, generator=generator) / (3 * 2)
    actual = [tensor.detach().clone().requires_grad_(True) for tensor in (embedding, updates, bank)]
    expected = [
        tensor.detach().clone().requires_grad_(True) for tensor in (embedding, updates, bank)
    ]
    config = _make_config(pp, vp, layout, block, fraction, actual[2], mtp)
    ref_config = _make_config(pp, vp, layout, block, fraction, expected[2], mtp)
    reference = _run_unsplit(ref_config, expected[0], expected[1], direction, mtp)
    result, completed_boundaries, undrained = _run_pipeline(
        api, config, actual[0], actual[1], direction, mtp
    )
    if any(sum(layout[:index]) % block == 0 for index in range(1, len(layout))):
        assert completed_boundaries > 0
    for got, wanted in zip(result, reference):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(
            got,
            wanted,
            atol=0.016 if dtype == torch.bfloat16 else 2e-5,
            rtol=0.01 if dtype == torch.bfloat16 else 2e-4,
        )
    for name, got, wanted in zip(('embedding', 'updates', 'query_bank'), actual, expected):
        assert got.grad is not None and torch.isfinite(got.grad).all(), name
        torch.testing.assert_close(
            got.grad,
            wanted.grad,
            atol=0.016 if dtype == torch.bfloat16 else 2e-5,
            rtol=0.01 if dtype == torch.bfloat16 else 2e-4,
            msg=lambda message: name + ': ' + message,
        )
    assert not undrained, [(tuple(leaf.shape), leaf.grad.norm().item()) for leaf in undrained]
    # Embedding plus each completed block is projected exactly once, on its
    # producing stage, including the final incomplete block and outgoing partial.
    expected_projections = 1 + (layers + block - 1) // block if fraction else 0
    assert len(api.projection_calls) == expected_projections
    # The first attention consumer has exactly one source: query derivative is zero.
    assert torch.count_nonzero(actual[2].grad[0]) == 0
    if mtp:
        assert torch.count_nonzero(actual[2].grad[-1]) > 0


def test_source_cache_reset_clears_both_channels(lifecycle):
    cache = lifecycle.value.get_attn_res_source_cache()
    cache.sources[3] = [torch.zeros(1)]
    cache.projections[3] = {0: object()}
    lifecycle.value.attn_res_source_cache_reset()
    assert not cache.sources and not cache.projections
