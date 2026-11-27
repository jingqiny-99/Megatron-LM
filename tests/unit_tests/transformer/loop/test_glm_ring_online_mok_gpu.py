# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare causal PP gate decisions to independently measured PP1 gate outputs and gradients."""

from contextlib import ExitStack
from unittest.mock import patch

import pytest

from megatron.core import parallel_state as ps
from megatron.core.transformer.loop.online_exit import round_exit_score
from tests.unit_tests.transformer.loop import test_glm_dsa_loop_ring_pp_gpu as ring
from tests.unit_tests.transformer.loop import test_glm_ring_stratified_mok_cp_gpu as mixed


def _thresholds_from_fixed_reference(model, samples):
    """Calibrate cutoffs on PP1 only. The pipeline gets two scalars and the original input order."""
    batches = samples[ps.get_data_parallel_rank() :: 2]
    local = {}

    def observe(lap):
        ids = iter(i for i, _ in batches)

        def collect(_gate, _args, output):
            sample = next(ids)
            score = round_exit_score(output, cp_group=ps.get_context_parallel_group())
            local[sample, lap] = float(score.item())

        return collect

    with ExitStack() as hooks:
        for lap in (1, 2):
            hooks.callback(
                model.decoder.loop_gates[lap - 1].register_forward_hook(observe(lap)).remove
            )
        hooks.enter_context(patch.object(ring, "_forward_step", mixed._forward))
        hooks.enter_context(patch.object(ring, "SEQUENCE", mixed.SEQUENCE))
        ring._step([model], [(i, row.cuda()) for i, row in batches], False, True)
    scores = {}
    for values in ring._gather_objects(local):
        for token, score in values.items():
            assert token not in scores or scores[token] == score
            scores[token] = score
    alive = list(range(len(samples)))
    depths = [3] * len(samples)
    thresholds = []
    for lap, count in enumerate(mixed.COUNTS[:2], 1):
        alive.sort(key=lambda i: scores[i, lap])
        lo, hi = scores[alive[count - 1], lap], scores[alive[count], lap]
        assert lo < hi, (lap, lo, hi)
        thresholds.append((lo + hi) / 2)
        for i in alive[:count]:
            depths[i] = lap
        alive = alive[count:]
    return thresholds, depths


@pytest.mark.skipif(not ring.CUDA or ring.WORLD != 8, reason="requires eight SM100/103 CUDA ranks")
def test_online_mok_pp2_cp4_ep4_matches_fixed_models():
    mixed._compare_mixed_step(calibrate=_thresholds_from_fixed_reference)
