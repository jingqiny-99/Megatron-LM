# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hybrid stack specs whose decoder runs the hybrid loop (``LoopHybridStack``).

Each spec is its base hybrid stack spec with ``module=LoopHybridStack`` (or, for the
Kimi-K3-style attention-residual carrier ``loop_carrier='attn_res'``,
``module=LoopAttnResHybridStack``) and nothing else changed: the specs are static, one stack
class per spec. Within ``LoopHybridStack`` the loop's carrier and core layout are config
fields of the stack class, as its Huginn / MoR architecture is: ``loop_carrier``
('injection' / 'residual'), ``loop_untie_core`` (the plain-inject untied core) and the mHC
streams select the round glue at construction; the attention-residual carrier is the one
behaviour with its own stack class and spec (``CARRIER_STACK_MISMATCH`` refuses a carrier on
the other family's spec). Use them through ``HybridModel(hybrid_stack_spec=...)`` or on the
command line, e.g.
``--spec megatron.core.transformer.loop.hybrid_specs loop_hybrid_dsv4_stack_spec``
(``hybrid_builder`` calls a config-aware spec factory with the config).
"""

import dataclasses

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec, hybrid_stack_spec
from megatron.core.transformer.loop.attn_res_stack import LoopAttnResHybridStack
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.spec_utils import ModuleSpec

# ``hybrid_stack_spec`` (TE layers) with the loop decoder.
loop_hybrid_stack_spec: ModuleSpec = dataclasses.replace(hybrid_stack_spec, module=LoopHybridStack)

# ``hybrid_stack_spec`` (TE layers) with the attention-residual loop decoder.
loop_attn_res_hybrid_stack_spec: ModuleSpec = dataclasses.replace(
    hybrid_stack_spec, module=LoopAttnResHybridStack
)


def loop_hybrid_dsv4_stack_spec(config) -> ModuleSpec:
    """``hybrid_dsv4_stack_spec(config)`` (DSv4 W / C / H / D attention) with the loop decoder."""
    return dataclasses.replace(hybrid_dsv4_stack_spec(config), module=LoopHybridStack)


def loop_attn_res_hybrid_dsv4_stack_spec(config) -> ModuleSpec:
    """``hybrid_dsv4_stack_spec(config)`` with the attention-residual loop decoder
    (``loop_carrier='attn_res'``)."""
    return dataclasses.replace(hybrid_dsv4_stack_spec(config), module=LoopAttnResHybridStack)
