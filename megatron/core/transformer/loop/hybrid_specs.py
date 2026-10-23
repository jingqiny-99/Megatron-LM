# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hybrid stack specs whose decoder runs the hybrid loop (``LoopHybridStack``).

Each spec is its base hybrid stack spec with ``module=LoopHybridStack`` and nothing else
changed: one behaviour per spec, no runtime switch. Use them through
``HybridModel(hybrid_stack_spec=...)`` or on the command line, e.g.
``--spec megatron.core.transformer.loop.hybrid_specs loop_hybrid_dsv4_stack_spec``
(``hybrid_builder`` calls a config-aware spec factory with the config).
"""

import dataclasses

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec, hybrid_stack_spec
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.spec_utils import ModuleSpec

# ``hybrid_stack_spec`` (TE layers) with the loop decoder.
loop_hybrid_stack_spec: ModuleSpec = dataclasses.replace(hybrid_stack_spec, module=LoopHybridStack)


def loop_hybrid_dsv4_stack_spec(config) -> ModuleSpec:
    """``hybrid_dsv4_stack_spec(config)`` (DSv4 W / C / H / D attention) with the loop decoder."""
    return dataclasses.replace(hybrid_dsv4_stack_spec(config), module=LoopHybridStack)
