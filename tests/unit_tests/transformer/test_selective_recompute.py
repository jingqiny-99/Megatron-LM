# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the scoped selective-recompute requests (``transformer/selective_recompute.py``,
decision D42 of the loop plan). CPU only: the scope itself; the DSv4 attention honouring it is
tested on GPU in ``experimental_attention_variant/test_dsv4_selective_up_proj_recompute.py``."""

import pytest

from megatron.core.transformer.selective_recompute import (
    SELECTIVE_RECOMPUTE_MODULES,
    request_selective_recompute,
    selective_recompute_requested,
)


def test_requests_nest_as_unions_and_reset_on_exit():
    assert SELECTIVE_RECOMPUTE_MODULES == frozenset({"mla_up_proj"})
    assert not selective_recompute_requested("mla_up_proj")
    with request_selective_recompute(()):
        assert not selective_recompute_requested("mla_up_proj")
        with request_selective_recompute({"mla_up_proj"}):
            assert selective_recompute_requested("mla_up_proj")
            # An inner empty request keeps the outer one (union).
            with request_selective_recompute(frozenset()):
                assert selective_recompute_requested("mla_up_proj")
            assert selective_recompute_requested("mla_up_proj")
        assert not selective_recompute_requested("mla_up_proj")
    assert not selective_recompute_requested("core_attn")


def test_requests_reset_when_the_body_raises():
    with pytest.raises(RuntimeError, match="boom"):
        with request_selective_recompute(["mla_up_proj"]):
            raise RuntimeError("boom")
    assert not selective_recompute_requested("mla_up_proj")


def test_unknown_modules_are_rejected():
    with pytest.raises(ValueError, match="unknown selective recompute modules"):
        with request_selective_recompute({"mla_up_proj", "core_attn"}):
            pass
    assert not selective_recompute_requested("mla_up_proj")
