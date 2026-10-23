# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Static-budget whole-MoE CUDA graphs without the paged stash (opt-in validator branch).

Config-only tests: no process group, no CUDA, no Transformer Engine import.
"""

import pytest

from megatron.core.transformer.cuda_graph_config import validate_moe_cuda_graph_support
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.transformer_config import TransformerConfig

# The existing validator message; a missing prerequisite must still report it unchanged.
_EXISTING_ASSERT = (
    "moe cuda graph is only supported with drop-padding MoE or transformer_engine "
    "sync-free HybridEP with rank capacity and paged stash."
)
_OPT_IN_FIELD = "moe_cuda_graph_static_budget_without_paged_stash"

_MODULE_SCOPES = pytest.mark.parametrize(
    "cuda_graph_modules", [[CudaGraphModule.moe], []], ids=["explicit-moe", "full-layer"]
)


def _static_budget_config(**overrides) -> TransformerConfig:
    kwargs = {
        "num_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "cuda_graph_impl": "transformer_engine",
        "cuda_graph_modules": [CudaGraphModule.moe],
        "num_moe_experts": 4,
        "moe_token_dispatcher_type": "flex",
        "moe_flex_dispatcher_backend": "hybridep",
        "moe_expert_rank_capacity_factor": 1.2,
        "use_transformer_engine_op_fuser": True,
        _OPT_IN_FIELD: True,
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_opt_in_field_defaults_off():
    config = TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4)
    assert getattr(config, _OPT_IN_FIELD) is False


@_MODULE_SCOPES
def test_opt_in_with_all_prerequisites_validates(cuda_graph_modules):
    # __post_init__ runs the validator for [moe]; the full-layer scope ([]) reaches it only
    # from TECudaGraphHelper.create_cudagraphs, so call it directly for both.
    config = _static_budget_config(cuda_graph_modules=cuda_graph_modules)
    validate_moe_cuda_graph_support(config)
    assert config.cuda_graph_modules == cuda_graph_modules
    assert not config.moe_paged_stash


def test_opt_in_does_not_apply_paged_stash_only_checks():
    # The paged-stash path needs TE >= 2.19, >= 2 warm-up steps and a fixed microbatch
    # schedule; none of those are keyed on the opt-in, so this config constructs.
    config = _static_budget_config(cuda_graph_warmup_steps=0, cuda_graph_dynamic_microbatches=True)
    validate_moe_cuda_graph_support(config)


def test_opt_in_with_paged_stash_is_rejected():
    with pytest.raises(ValueError, match=f"{_OPT_IN_FIELD} is the static budget without"):
        _static_budget_config(moe_paged_stash=True)


@_MODULE_SCOPES
def test_opt_in_without_rank_capacity_keeps_existing_assert(cuda_graph_modules):
    # alltoall without rank capacity passes __post_init__ for the full-layer scope only.
    config = _static_budget_config(
        cuda_graph_modules=[],
        moe_token_dispatcher_type="alltoall",
        moe_flex_dispatcher_backend="deepep",
        moe_expert_rank_capacity_factor=None,
        use_transformer_engine_op_fuser=False,
    )
    config.cuda_graph_modules = cuda_graph_modules
    with pytest.raises(AssertionError) as error:
        validate_moe_cuda_graph_support(config)
    assert str(error.value) == _EXISTING_ASSERT


@_MODULE_SCOPES
def test_opt_in_with_grouped_tensor_instead_of_op_fuser_keeps_existing_assert(cuda_graph_modules):
    # TransformerConfig's rank-capacity rule accepts moe_use_grouped_tensor in place of the
    # op fuser; the validator (and its static-budget branch) still requires the op fuser.
    config = _static_budget_config(
        cuda_graph_modules=[],
        use_transformer_engine_op_fuser=False,
        moe_grouped_gemm=True,
        moe_use_grouped_tensor=True,
    )
    config.cuda_graph_modules = cuda_graph_modules
    with pytest.raises(AssertionError) as error:
        validate_moe_cuda_graph_support(config)
    assert str(error.value) == _EXISTING_ASSERT


def test_opt_in_with_grouped_tensor_raises_from_post_init_for_explicit_moe():
    # __post_init__ runs the validator only for an explicit moe scope.
    with pytest.raises(AssertionError, match="sync-free HybridEP with rank capacity"):
        _static_budget_config(
            use_transformer_engine_op_fuser=False,
            moe_grouped_gemm=True,
            moe_use_grouped_tensor=True,
        )


@_MODULE_SCOPES
def test_without_opt_in_static_budget_is_still_rejected(cuda_graph_modules):
    # The case pinned by test_cuda_graphs.py: no paged stash and no opt-in.
    with pytest.raises(AssertionError, match="sync-free HybridEP with rank capacity and paged"):
        config = _static_budget_config(
            cuda_graph_modules=cuda_graph_modules, **{_OPT_IN_FIELD: False}
        )
        validate_moe_cuda_graph_support(config)


def test_opt_in_leaves_moe_router_scope_unaffected():
    # A router-only graph does not capture the whole MoE; the validator returns early.
    config = _static_budget_config(
        cuda_graph_modules=[CudaGraphModule.moe_router],
        moe_token_dispatcher_type="alltoall",
        moe_flex_dispatcher_backend="deepep",
        moe_expert_rank_capacity_factor=None,
        use_transformer_engine_op_fuser=False,
    )
    validate_moe_cuda_graph_support(config)
    assert config.cuda_graph_modules == [CudaGraphModule.moe_router]
