# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Host regressions for model entrypoints accepting the typed AttnRes payload.

Execute exact production method bodies, including the real viewless-tensor and
pipeline-layout helpers, without constructing GPU layers. Only unrelated module
imports and distributed rank discovery are supplied by the fixture. These tests
complement the source-state tests: an otherwise correct payload must also pass
through the outer model's viewless-output and positional-embedding entrypoints.
"""

import __future__

import ast
import builtins
import copy
import functools
import types
import warnings
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[3]
_CORE = _ROOT / 'megatron/core'


def _tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def _method(path, class_name, method_name):
    cls = next(
        node
        for node in _tree(path).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return copy.deepcopy(
        next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
    )


def _execute(nodes, namespace, path):
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(tree, str(path), 'exec', flags=__future__.annotations.compiler_flag), namespace)


def _load_functions(path, names, namespace):
    nodes = [
        copy.deepcopy(node)
        for node in _tree(path).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    _execute(nodes, namespace, path)


@pytest.fixture
def model_contracts():
    rank = [1]
    namespace = dict(torch=torch, functools=functools, warnings=warnings, _UNIFORM_SLICES_MEMO={})
    parallel_state = types.SimpleNamespace(get_pipeline_model_parallel_rank=lambda: rank[0])

    def resolve_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'megatron.core':
            return types.SimpleNamespace(parallel_state=parallel_state)
        if name in (
            'megatron.core.transformer.attention_residual',
            'megatron.core.transformer.transformer_layer',
            'megatron.core.pipeline_parallel.utils',
        ):
            return types.SimpleNamespace(**namespace)
        return builtins.__import__(name, globals, locals, fromlist, level)

    namespace['__builtins__'] = dict(vars(builtins), __import__=resolve_import)
    _load_functions(
        _CORE / 'utils.py',
        (
            '_kernel_make_viewless_tensor',
            'MakeViewlessTensor',
            'make_viewless_tensor',
            'deprecate_inference_params',
        ),
        namespace,
    )
    _load_functions(_CORE / 'pipeline_parallel/utils.py', ('is_vp_first_stage',), namespace)
    _load_functions(
        _CORE / 'transformer/transformer_layer.py', ('get_transformer_layer_offset',), namespace
    )
    _load_functions(
        _CORE / 'transformer/attention_residual.py',
        (
            'attn_res_num_sources',
            'attn_res_num_payload_slices',
            '_hybrid_entries_before_segment',
            '_hybrid_layers_before_pp_rank',
            'attn_res_payload_slices_for_pp_rank',
            '_sources_formed_through',
            '_stage_layers_before',
            'attn_res_boundary_delta_slices',
            'attn_res_uniform_payload_slices',
        ),
        namespace,
    )
    rotary_path = _CORE / 'models/common/embeddings/rotary_pos_embedding.py'
    _execute(
        [_method(rotary_path, 'RotaryEmbedding', 'get_rotary_seq_len')], namespace, rotary_path
    )

    hybrid_path = _CORE / 'models/hybrid/hybrid_block.py'
    forward = _method(hybrid_path, 'HybridStack', 'forward')
    # Extract the real output tail beginning with its viewless conversion,
    # retaining the MTP/mHC auxiliary-output branches and the final return.
    start = next(
        index
        for index, node in enumerate(forward.body)
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == 'make_viewless_tensor'
            for child in ast.walk(node)
        )
    )
    tail = ast.parse(
        'def hybrid_tail(self, hidden_states, mhc_multistream=None, mtp_attn_res_sources=None): pass'
    ).body[0]
    tail.body = forward.body[start:]
    _execute([tail], namespace, hybrid_path)
    return types.SimpleNamespace(**namespace, rank=rank)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_hybrid_tail_preserves_typed_viewless_channels_and_gradients(model_contracts, dtype):
    values = torch.randn(24, dtype=dtype, requires_grad=True)
    scores = torch.randn(12, dtype=torch.float32, requires_grad=True)
    payload = [values.view(3, 2, 4), scores.view(6, 2, 1)]
    model = types.SimpleNamespace(config=types.SimpleNamespace(attn_res_impl='source'))
    output = model_contracts.hybrid_tail(model, payload)
    assert isinstance(output, list) and len(output) == 2
    assert all(tensor._base is None for tensor in output)
    assert [tensor.dtype for tensor in output] == [dtype, torch.float32]
    for before, after in zip(payload, output):
        torch.testing.assert_close(after, before)
        assert after.data_ptr() == before.data_ptr()
    (output[0].float().sum() + 3 * output[1].sum()).backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))
    torch.testing.assert_close(scores.grad, torch.full_like(scores, 3))


@pytest.mark.parametrize('backend', ['eager', 'source'])
def test_hybrid_tail_keeps_tensor_and_mtp_source_contract(model_contracts, backend):
    value = torch.randn(24, requires_grad=True)
    sources = (torch.randn(3, 2, 4), torch.randn(3, 2, 4))
    model = types.SimpleNamespace(config=types.SimpleNamespace(attn_res_impl=backend))
    output, returned_sources = model_contracts.hybrid_tail(
        model, value.view(3, 2, 4), mtp_attn_res_sources=sources
    )
    assert isinstance(output, torch.Tensor) and output._base is None
    assert returned_sources is sources
    output.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))


_ROPE_LAYOUTS = [
    pytest.param(False, None, 16, None, 5, id='gpt-pp2'),
    pytest.param(False, 2, 16, None, 3, id='gpt-pp2-vpp2'),
    pytest.param(True, None, 8, 'AAA|AAAAA', 3, id='hybrid-uneven-pp2'),
    pytest.param(True, 2, 8, 'AA|A|AAA|AA', 2, id='hybrid-uneven-pp2-vpp2'),
]


def _rope_config(hybrid, vp, layers, pattern, sp, cp, backend='source'):
    return types.SimpleNamespace(
        attn_res_impl=backend,
        enable_attention_residuals=True,
        num_layers=layers,
        attn_res_block_layers=2,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=vp,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
        pipeline_model_parallel_layout=None,
        num_layers_in_first_pipeline_stage=None,
        num_layers_in_last_pipeline_stage=None,
        is_hybrid_model=hybrid,
        hybrid_layer_pattern=pattern,
        sequence_parallel=sp,
        tensor_model_parallel_size=2,
        context_parallel_size=cp,
    )


@pytest.mark.parametrize('hybrid,vp,layers,pattern,slices', _ROPE_LAYOUTS)
@pytest.mark.parametrize('sp,cp', [(False, 1), (True, 1), (False, 2), (True, 2)])
@pytest.mark.parametrize('backend', ['eager', 'source'])
def test_rope_uses_value_channel_and_recovers_true_sequence_length(
    model_contracts, hybrid, vp, layers, pattern, slices, sp, cp, backend
):
    tokens = 7
    config = _rope_config(hybrid, vp, layers, pattern, sp, cp, backend)
    values = torch.empty(slices * tokens, 2, 4, dtype=torch.bfloat16)
    # Deliberately unrelated scalar-channel width: using it must not appear to work.
    scores = torch.empty(37 * tokens, 2, 1, dtype=torch.float32)
    decoder = types.SimpleNamespace(
        input_tensor=[values, scores] if backend == 'source' else values, pre_process=False
    )
    length = model_contracts.get_rotary_seq_len(None, None, decoder, None, config)
    assert length == tokens * (2 if sp else 1) * cp


def test_rope_first_stage_and_packed_lengths_remain_unchanged(model_contracts):
    config = _rope_config(False, None, 16, None, True, 2)
    decoder = types.SimpleNamespace(input_tensor=None, pre_process=True)
    length = model_contracts.get_rotary_seq_len(None, None, decoder, torch.empty(7, 2, 4), config)
    assert length == 28
    packed = types.SimpleNamespace(max_seqlen_q=13, max_seqlen_kv=19)
    length = model_contracts.get_rotary_seq_len(None, None, decoder, None, config, packed)
    assert length == 19


def test_rope_rejects_malformed_value_payload(model_contracts):
    config = _rope_config(False, None, 16, None, False, 1)
    decoder = types.SimpleNamespace(
        input_tensor=[torch.empty(36, 2, 4), torch.empty(1, 2, 1)], pre_process=False
    )
    with pytest.raises(AssertionError, match='divisible'):
        model_contracts.get_rotary_seq_len(None, None, decoder, None, config)


@pytest.fixture
def validate_source_config():
    path = _CORE / 'transformer/transformer_config.py'
    method = _method(path, 'TransformerConfig', '_validate_attention_residuals')
    names = {
        node.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == 'self'
    }
    defaults = {}
    for config_path, class_name in (
        (_CORE / 'model_parallel_config.py', 'ModelParallelConfig'),
        (path, 'TransformerConfig'),
    ):
        cls = next(
            node
            for node in _tree(config_path).body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        for node in cls.body:
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id in names
            ):
                defaults[node.target.id] = ast.literal_eval(node.value)
    assert (
        defaults.keys() == names
    ), 'Extend the host fixture for newly added validation dependencies'
    namespace = {}
    _execute([method], namespace, path)
    return namespace['_validate_attention_residuals'], defaults


@pytest.mark.parametrize('fraction', [0.0, 0.25, 0.5, 0.75, 1.0])
def test_source_fraction_accepts_full_integration_flags(validate_source_config, fraction):
    validate, defaults = validate_source_config
    config = types.SimpleNamespace(**defaults)
    config.enable_attention_residuals = True
    config.attn_res_impl = 'source'
    config.attn_res_block_layers = 2
    config.attn_res_source_projection_fraction = fraction
    config.pipeline_model_parallel_size = 2
    config.virtual_pipeline_model_parallel_size = 2
    config.is_hybrid_model = True
    config.hybrid_layer_pattern = 'AA|A|AAA|AA'
    config.mtp_num_layers = 2
    config.recompute_granularity = 'selective'
    config.fine_grained_activation_offloading = True
    validate(config)


@pytest.mark.parametrize('fraction', [-0.01, 1.01, float('nan'), float('inf')])
def test_source_fraction_rejects_out_of_range_values(validate_source_config, fraction):
    validate, defaults = validate_source_config
    config = types.SimpleNamespace(**defaults)
    config.attn_res_source_projection_fraction = fraction
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        validate(config)


def test_source_requires_attention_residuals_and_preserves_default(validate_source_config):
    validate, defaults = validate_source_config
    config = types.SimpleNamespace(**defaults)
    assert not config.enable_attention_residuals
    assert config.attn_res_impl == 'fla'
    validate(config)
    config.attn_res_impl = 'source'
    with pytest.raises(ValueError, match='requires enable_attention_residuals'):
        validate(config)


@pytest.mark.parametrize(
    'field,value,match',
    [
        ('attn_res_impl', 'unknown', 'attn_res_impl'),
        ('cuda_graph_impl', 'local', 'cuda_graph_impl'),
        ('recompute_granularity', 'full', 'recompute_granularity'),
        ('variable_seq_lengths', True, 'variable_seq_lengths'),
    ],
)
def test_source_retains_existing_unsupported_guards(validate_source_config, field, value, match):
    validate, defaults = validate_source_config
    config = types.SimpleNamespace(**defaults)
    config.enable_attention_residuals = True
    config.attn_res_impl = 'source'
    config.attn_res_block_layers = 2
    config.pipeline_model_parallel_size = 2
    setattr(config, field, value)
    with pytest.raises(ValueError, match=match):
        validate(config)


@pytest.mark.parametrize("before,after", [("fla", "source"), ("source", "eager")])
@pytest.mark.parametrize("sharding", ["dp_reshardable", "dp_zero_gather_scatter", None])
def test_optimizer_backend_change_requires_named_checkpoint(before, after, sharding):
    path = _ROOT / "megatron/training/checkpointing.py"
    namespace = dict(
        DistributedOptimizer=types.SimpleNamespace(
            checkpoint_fully_reshardable_formats={"fully_reshardable", "fully_sharded_model_space"}
        )
    )
    _load_functions(path, ("_validate_attn_res_optimizer_checkpoint",), namespace)
    metadata = None if sharding is None else {"distrib_optim_sharding_type": sharding}
    with pytest.raises(RuntimeError, match="original attn_res_impl"):
        namespace["_validate_attn_res_optimizer_checkpoint"](
            types.SimpleNamespace(attn_res_impl=after),
            types.SimpleNamespace(attn_res_impl=before),
            metadata,
        )


@pytest.mark.parametrize(
    "before,after,sharding",
    [
        ("fla", "source", "fully_reshardable"),
        ("source", "eager", "fully_sharded_model_space"),
        ("source", "source", "dp_reshardable"),
        ("fla", "eager", "dp_reshardable"),
    ],
)
def test_optimizer_checkpoint_preserves_supported_restore(before, after, sharding):
    path = _ROOT / "megatron/training/checkpointing.py"
    namespace = dict(
        DistributedOptimizer=types.SimpleNamespace(
            checkpoint_fully_reshardable_formats={"fully_reshardable", "fully_sharded_model_space"}
        )
    )
    _load_functions(path, ("_validate_attn_res_optimizer_checkpoint",), namespace)
    namespace["_validate_attn_res_optimizer_checkpoint"](
        types.SimpleNamespace(attn_res_impl=after),
        types.SimpleNamespace(attn_res_impl=before),
        {"distrib_optim_sharding_type": sharding},
    )
