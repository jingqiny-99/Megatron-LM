# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Arguments of SBHD document packing (``--sbhd-document-packing``), CPU only.

``_validate_sbhd_document_packing`` on bare argument namespaces (every option it rejects, with
its message; the compatible options pass), the DSA indexer-loss rows (the default DSv4 recipe
resolves to the fused cudnn kernels and is rejected; a plain DSA indexer loss; loop-less DSv4),
``resolve_dsa_kernel_backend`` against ``TransformerConfig.__post_init__`` on every (variant,
backend, fusion switch) combination, the CLI flags, ``pretrain_hybrid``'s padding-mask switch
and dataset config, the entry points that are not wired (``pretrain_gpt``, the flex entry) and
the offline cache builder's config.
"""

import itertools
import sys
from argparse import Namespace

import pytest

from megatron.core.transformer import transformer_config
from megatron.core.transformer.transformer_config import (
    MLATransformerConfig,
    TransformerConfig,
    resolve_dsa_kernel_backend,
)
from megatron.training.arguments import _validate_sbhd_document_packing, parse_args


def _args(**overrides):
    return Namespace(sbhd_document_packing=True, **overrides)


# ---------------------------------------------------------------------------
# The validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(sft=True), "--sft"),
        (dict(use_varlen_dataset=True), "--use-varlen-dataset"),
        (dict(use_varlen_dataset=True, varlen_sbhd_validation=True), "--use-varlen-dataset"),
        (dict(sequence_packing_scheduler="dp_balanced"), "--sequence-packing-scheduler"),
        (dict(dataloader_inter_document_masking=True), "--dataloader-inter-document-masking"),
        (dict(context_parallel_size=2), "--context-parallel-size"),
        (dict(dynamic_context_parallel=True), "--dynamic-context-parallel"),
        (dict(dataloader_defer_npy_index_mmap=True), "--dataloader-defer-npy-index-mmap"),
        (dict(reset_attention_mask=True), "--reset-attention-mask"),
        (dict(moe_expert_capacity_factor=1.0), "--moe-expert-capacity-factor"),
        (dict(moe_router_load_balancing_type="sinkhorn"), "sinkhorn"),
        (dict(moe_router_load_balancing_type=["aux_loss", "sinkhorn"]), "sinkhorn"),
        (dict(moe_router_load_balancing_type="quantile_balancing"), "quantile_balancing"),
        (
            dict(
                moe_router_load_balancing_type="quantile_balancing",
                moe_router_quantile_balancing_estimation_scope="micro_batch",
            ),
            "quantile_balancing",
        ),
        (dict(sbhd_packing_open_rows=0), "--sbhd-packing-open-rows"),
    ],
)
def test_rejected_options(overrides, match):
    with pytest.raises(ValueError, match=match):
        _validate_sbhd_document_packing(_args(**overrides))
    # Packing off: the validator checks nothing.
    _validate_sbhd_document_packing(Namespace(sbhd_document_packing=False, **overrides))
    _validate_sbhd_document_packing(Namespace(**overrides))


def test_compatible_options_pass():
    _validate_sbhd_document_packing(_args())
    _validate_sbhd_document_packing(
        _args(
            sbhd_packing_open_rows=1,
            reset_position_ids=True,
            eod_mask_loss=True,
            create_attention_mask_in_dataloader=True,
            mock_data=True,
            dataloader_fast_cache_load=True,
            mid_level_dataset_surplus=0.005,
            context_parallel_size=1,
            sequence_packing_scheduler=None,
            moe_expert_capacity_factor=None,
            moe_router_load_balancing_type=["seq_aux_loss", "global_aux_loss"],
            dsa_indexer_loss_coeff=0.0,
            experimental_attention_variant="dsv4_hybrid",
        )
    )


DSV4 = dict(experimental_attention_variant="dsv4_hybrid", dsa_indexer_loss_coeff=0.01)


@pytest.mark.parametrize(
    "overrides, match",
    [
        # The default recipe: backend unset resolves to cudnn.
        (dict(loop_architecture="huginn"), "resolves to 'cudnn'"),
        (dict(loop_architecture="huginn", apply_dsa_kernel_fusion=True), "resolves to 'cudnn'"),
        (dict(loop_architecture="huginn", dsa_kernel_backend="cudnn"), "resolves to 'cudnn'"),
        (dict(loop_architecture="huginn", dsa_kernel_backend="tilelang"), "resolves to 'tilelang'"),
        (dict(dsa_kernel_backend="none"), "without --loop-architecture"),
        (dict(loop_architecture="none", dsa_kernel_backend="none"), "without --loop-architecture"),
    ],
)
def test_dsv4_indexer_loss_rows(overrides, match):
    with pytest.raises(ValueError, match=match):
        _validate_sbhd_document_packing(_args(**DSV4, **overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        dict(loop_architecture="huginn", dsa_kernel_backend="none"),
        dict(loop_architecture="huginn", apply_dsa_kernel_fusion=False),
        dict(loop_architecture="huginn", dsa_kernel_backend="none", apply_dsa_kernel_fusion=False),
    ],
)
def test_dsv4_indexer_loss_on_the_unfused_loop_path_passes(overrides):
    _validate_sbhd_document_packing(_args(**DSV4, **overrides))


def test_plain_dsa_indexer_loss_is_rejected():
    with pytest.raises(ValueError, match="experimental_attention_variant='dsa'"):
        _validate_sbhd_document_packing(
            _args(experimental_attention_variant="dsa", dsa_indexer_loss_coeff=0.01)
        )
    # Without an indexer loss nothing reads the padding there.
    _validate_sbhd_document_packing(
        _args(experimental_attention_variant="dsa", dsa_indexer_loss_coeff=0.0)
    )


# ---------------------------------------------------------------------------
# resolve_dsa_kernel_backend == TransformerConfig.__post_init__
# ---------------------------------------------------------------------------


class _Resolved(Exception):
    pass


def _post_init_backend(variant, backend, fusion):
    """The backend ``__post_init__`` resolves (read at its dependency check, which every DSA
    variant runs right after the resolution), or the ValueError it raises first."""
    fields = dict(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        dsa_kernel_backend=backend,
        apply_dsa_kernel_fusion=fusion,
    )
    if variant is None:
        return TransformerConfig(**fields).dsa_kernel_backend
    fields.update(
        multi_latent_attention=True,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_head_dim=16,
        qk_pos_emb_head_dim=8,
        v_head_dim=16,
        add_bias_linear=False,
        experimental_attention_variant=variant,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=32,
        dsa_indexer_topk=16,
        dsa_indexer_loss_coeff=0.01,
    )
    if variant == "dsv4_hybrid":
        fields.update(csa_window_size=32, csa_compress_ratios=[0, 4])
    try:
        MLATransformerConfig(**fields)
    except _Resolved as resolved:
        return resolved.args[0]
    except ValueError as error:
        if "does not support dsa_kernel_backend='tilelang'" in str(error):
            return "tilelang"  # rejected for DSv4 after the resolution
        raise
    raise AssertionError("the dependency check was not reached")


@pytest.mark.parametrize(
    "variant, backend, fusion",
    list(
        itertools.product(
            [None, "dsa", "dsv4_hybrid"], [None, "none", "tilelang", "cudnn"], [None, True, False]
        )
    ),
)
def test_resolve_dsa_kernel_backend_equals_post_init(monkeypatch, variant, backend, fusion):
    def stop(resolved):
        raise _Resolved(resolved)

    monkeypatch.setattr(transformer_config, "_validate_dsa_kernel_backend_dependencies", stop)
    try:
        expected = _post_init_backend(variant, backend, fusion)
    except ValueError as error:
        assert "Conflicting DSA kernel controls" in str(error)
        with pytest.raises(ValueError, match="Conflicting DSA kernel controls"):
            resolve_dsa_kernel_backend(variant, backend, fusion)
        return
    assert resolve_dsa_kernel_backend(variant, backend, fusion) == expected
    if variant == "dsv4_hybrid" and backend is None:
        assert expected == ("cudnn" if fusion in (None, True) else "none")


# ---------------------------------------------------------------------------
# CLI, pretrain_hybrid, entry points, offline cache
# ---------------------------------------------------------------------------


def test_cli_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test"])
    args = parse_args()
    assert args.sbhd_document_packing is False and args.sbhd_packing_open_rows == 64
    monkeypatch.setattr(
        sys, "argv", ["test", "--sbhd-document-packing", "--sbhd-packing-open-rows", "8"]
    )
    args = parse_args()
    assert args.sbhd_document_packing is True and args.sbhd_packing_open_rows == 8


def test_pretrain_hybrid_padding_mask_switch_and_dataset_config(monkeypatch):
    import pretrain_hybrid

    assert not pretrain_hybrid._needs_padding_mask(Namespace())
    assert pretrain_hybrid._needs_padding_mask(Namespace(sbhd_document_packing=True))
    assert pretrain_hybrid._needs_padding_mask(
        Namespace(use_varlen_dataset=True, varlen_sbhd_validation=True)
    )
    assert not pretrain_hybrid._needs_padding_mask(Namespace(use_varlen_dataset=True))

    monkeypatch.setattr(pretrain_hybrid, "get_run_config", lambda: Namespace(rng=Namespace(seed=7)))
    from megatron.core.tokenizers import MegatronTokenizer

    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-text"}, vocab_size=1024
    )
    monkeypatch.setattr(pretrain_hybrid, "build_tokenizer", lambda args: tokenizer)
    monkeypatch.setattr(sys, "argv", ["test", "--sbhd-document-packing", "--mock-data"])
    args = parse_args()
    args.seq_length = 128
    args.data_parallel_size = 1
    config = pretrain_hybrid.core_gpt_dataset_config_from_args(args)
    assert config.sbhd_document_packing and config.sbhd_packing_open_rows == 64
    args.sbhd_document_packing = False
    assert not pretrain_hybrid.core_gpt_dataset_config_from_args(args).sbhd_document_packing


@pytest.mark.parametrize(
    "variant, loop_architecture, backend, match",
    [
        ("dsv4_hybrid", "huginn", "cudnn", "resolves to 'cudnn'"),
        ("dsv4_hybrid", "none", "none", "without --loop-architecture"),
        ("dsa", "huginn", "none", "experimental_attention_variant='dsa'"),
        ("dsa", "none", "none", "experimental_attention_variant='dsa'"),
        ("dsv4_hybrid", "huginn", "none", None),
        (None, "none", "none", None),
    ],
)
def test_model_config_backstop(monkeypatch, variant, loop_architecture, backend, match):
    """The dataset provider re-checks every indexer-loss row on the built config (its resolved
    variant, loop architecture and kernel backend)."""
    import pretrain_hybrid

    built = Namespace(
        experimental_attention_variant=variant,
        dsa_indexer_loss_coeff=0.01,
        loop_architecture=loop_architecture,
        dsa_kernel_backend=backend,
    )
    monkeypatch.setattr(pretrain_hybrid, "core_transformer_config_from_args", lambda args: built)
    pretrain_hybrid._check_sbhd_packing_model_config(Namespace(sbhd_document_packing=False))
    if match is None:
        pretrain_hybrid._check_sbhd_packing_model_config(Namespace(sbhd_document_packing=True))
    else:
        with pytest.raises(ValueError, match=match):
            pretrain_hybrid._check_sbhd_packing_model_config(Namespace(sbhd_document_packing=True))
    built.dsa_indexer_loss_coeff = 0.0
    pretrain_hybrid._check_sbhd_packing_model_config(Namespace(sbhd_document_packing=True))


@pytest.mark.parametrize(
    "pattern, overrides, match",
    [
        # DSv4 C/H/W symbols imply dsv4_hybrid; an unset backend resolves to cudnn.
        ("WCHE", dict(loop_architecture="none"), "without --loop-architecture"),
        ("WCHE", dict(loop_architecture="huginn"), "resolves to 'cudnn'"),
        ("WE", dict(loop_architecture="huginn", dsa_kernel_backend="cudnn"), "resolves to 'cudnn'"),
        ("CE", dict(dsa_kernel_backend="none"), "without --loop-architecture"),
        # A D symbol implies plain dsa.
        ("DE", dict(), "experimental_attention_variant='dsa'"),
        ("DE", dict(loop_architecture="huginn"), "experimental_attention_variant='dsa'"),
        ("MD*", dict(dsa_kernel_backend="none"), "experimental_attention_variant='dsa'"),
    ],
)
def test_pattern_inferred_indexer_loss_rows(pattern, overrides, match):
    """``--hybrid-layer-pattern`` implies the attention variant without
    ``--experimental-attention-variant``; the validator sees the variant the model is built
    with."""
    with pytest.raises(ValueError, match=match):
        _validate_sbhd_document_packing(
            _args(hybrid_layer_pattern=pattern, dsa_indexer_loss_coeff=0.01, **overrides)
        )
    # Without an indexer loss nothing reads the padding there.
    _validate_sbhd_document_packing(
        _args(hybrid_layer_pattern=pattern, dsa_indexer_loss_coeff=0.0, **overrides)
    )


def test_pattern_inferred_unfused_loop_dsv4_passes():
    _validate_sbhd_document_packing(
        _args(
            hybrid_layer_pattern="WCHE",
            dsa_indexer_loss_coeff=0.01,
            loop_architecture="huginn",
            dsa_kernel_backend="none",
        )
    )
    # A pattern without DSA symbols implies no indexer.
    _validate_sbhd_document_packing(_args(hybrid_layer_pattern="M*E", dsa_indexer_loss_coeff=0.01))


def test_pattern_inference_matches_the_built_config():
    """The validator's inference is the helper ``core_transformer_config_from_args`` builds
    the config with."""
    from megatron.training.argument_utils import hybrid_pattern_attention_variant

    for pattern, variant in {"WCHE": "dsv4_hybrid", "DE": "dsa", "KE": "kda", "M*E": None}.items():
        args = Namespace(hybrid_layer_pattern=pattern, experimental_attention_variant=None)
        assert hybrid_pattern_attention_variant(args) == variant, pattern
    # An explicit variant wins over D / K (nothing inferred), never over C/H/W.
    assert (
        hybrid_pattern_attention_variant(
            Namespace(hybrid_layer_pattern="DE", experimental_attention_variant="dsv4_hybrid")
        )
        is None
    )
    with pytest.raises(ValueError, match="requires experimental_attention_variant='dsv4_hybrid'"):
        hybrid_pattern_attention_variant(
            Namespace(hybrid_layer_pattern="WE", experimental_attention_variant="dsa")
        )
    assert hybrid_pattern_attention_variant(Namespace()) is None


def test_seq_aux_loss_mbs_warning(monkeypatch):
    """seq_aux_loss at MBS > 1 on the standard schedule shares one normalisation across rows
    with unequal padding: the validator warns (it does not reject)."""
    from megatron.training import arguments

    warnings = []
    monkeypatch.setattr(arguments, "warn_rank_0", warnings.append)
    seq_aux = dict(moe_router_load_balancing_type="seq_aux_loss")
    _validate_sbhd_document_packing(_args(micro_batch_size=1, **seq_aux))
    _validate_sbhd_document_packing(_args(micro_batch_size=2, loop_schedule="rsbf", **seq_aux))
    _validate_sbhd_document_packing(_args(micro_batch_size=2))
    assert warnings == []
    _validate_sbhd_document_packing(_args(micro_batch_size=2, **seq_aux))
    assert len(warnings) == 1 and "seq_aux_loss" in warnings[0]


def test_unwired_entry_points_raise(monkeypatch):
    import pretrain_gpt

    monkeypatch.setattr(pretrain_gpt, "get_args", lambda: Namespace(sbhd_document_packing=True))
    with pytest.raises(ValueError, match="pretrain_hybrid.py only"):
        pretrain_gpt.train_valid_test_datasets_provider([1, 1, 1])
    try:
        from megatron.elastification import pretrain_hybrid_flex
    except Exception as error:  # optional dependencies of the flex entry
        pytest.skip(f"the flex entry does not import here: {error}")
    with pytest.raises(ValueError, match="pretrain_hybrid.py only"):
        pretrain_hybrid_flex.core_gpt_dataset_config_from_args(
            Namespace(sbhd_document_packing=True)
        )


def test_prepare_cache_forwards_the_packing_fields(tmp_path):
    from tests.unit_tests.data.test_prepare_cache import _build_prepare_cache_args
    from tools.prepare_cache import core_gpt_dataset_config_from_args

    args = _build_prepare_cache_args(
        [str(tmp_path / "file_0")],
        tmp_path / "cache",
        dataloader_defer_npy_index_mmap=False,
        dataloader_fast_cache_load=False,
        sbhd_document_packing=True,
        sbhd_packing_open_rows=8,
    )
    config = core_gpt_dataset_config_from_args(args)
    assert config.sbhd_document_packing and config.sbhd_packing_open_rows == 8
    del args.sbhd_document_packing, args.sbhd_packing_open_rows
    config = core_gpt_dataset_config_from_args(args)
    assert not config.sbhd_document_packing and config.sbhd_packing_open_rows == 64
