# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The loop ring's recompute plan on real kernels (GPU): checkpointed round views ('C') equal kept
ones ('n') bitwise, with one routing tape per checkpoint call.

Runs under torchrun with CUDA at W = 2, 4 or 8 ranks on ``ring_gpu_utils``' tiny DSv4 loop (EP 1)
in deterministic mode. For each ring layout (P = 2 with k = 1 and k = 2, P = 4 with k = 1 when
W allows) one step runs from the same weights and batches twice: ``loop_recompute`` off (plan all
'n') and on (plan all 'C': every round body, and the prelude in round 1, is one non-reentrant
activation checkpoint whose recompute runs inside the backward, after other microbatches' and
rounds' calls and after the call roles were cleared). Gates:

* every parameter's ``main_grad`` after finalize, every sequence's loss report and every router
  expert bias after the update are **bitwise** equal between the two runs (the recompute repeats
  its forward exactly: the routing tape replays the MoE top-k, deterministic kernels the rest);
* one ``RoutingTape`` per checkpoint call: this rank's tapes = R x M round-body checkpoints per
  chunk plus M prelude checkpoints on the round-start chunk; every tape is replayed exactly once
  and to its end (``torch.utils.checkpoint`` early stop off for the 'C' run), the round-body tapes
  of a chunk all record the same number of router calls, a non-zero multiple of the chunk's
  learned-routing MoE sublayers, and the prelude's tapes record none (its MoE layers are
  hash-routed).
"""

from unittest.mock import patch

import pytest
import torch

from megatron.core.transformer import loop_transformer_block
from tests.unit_tests.transformer.loop.hybrid_loop_utils import determinism_env
from tests.unit_tests.transformer.loop.ring_gpu_utils import (
    CUDA,
    ROUNDS,
    WORLD,
    Layout,
    build_chunks,
    deterministic_algorithms,
    full_precision_matmuls,
    gather,
    load_state,
    parallel_layout,
    rank_batches,
    ring_layouts,
    run_step,
    run_with_precision,
    state_snapshot,
    tiny_config,
    wrap_ddp,
)

pytestmark = pytest.mark.skipif(
    not CUDA or WORLD < 2 or WORLD % 2, reason="the ring needs CUDA and an even number of ranks"
)

LAYOUTS = (
    [
        layout
        for layout in ring_layouts(recompute=(True,))
        if (layout.stages, layout.chunks) in ((2, 1), (2, 2), (4, 1))
    ]
    if CUDA and WORLD >= 2 and WORLD % 2 == 0
    else []
)
NAMES = [layout.name for layout in LAYOUTS]


class _Tapes:
    """Counts the routing tapes the loop's checkpoints make and their replays."""

    def __init__(self):
        self.tapes = []
        self.replays = {}
        real_tape = loop_transformer_block.RoutingTape
        real_replay = loop_transformer_block.replay_routing
        tapes, replays = self.tapes, self.replays

        class CountingTape(real_tape):
            """A routing tape that registers itself."""

            def __init__(self):
                super().__init__()
                tapes.append(self)

        def counting_replay(tape):
            replays[id(tape)] = replays.get(id(tape), 0) + 1
            return real_replay(tape)

        self.patches = (
            patch.object(loop_transformer_block, "RoutingTape", CountingTape),
            patch.object(loop_transformer_block, "replay_routing", counting_replay),
        )

    def __enter__(self):
        for item in self.patches:
            item.__enter__()
        return self

    def __exit__(self, *exc):
        for item in reversed(self.patches):
            item.__exit__(*exc)


def _moe_sublayers(model):
    """The learned-routing MoE sublayers ('E') of a chunk's core slice."""
    decoder = model.decoder
    local = decoder.loop_ring.local
    return sum(
        1
        for index in range(local.core_start, local.core_end)
        if decoder.layer_type_list[index] == "E"
    )


def _one(dtype, recompute_layout):
    """Plan all 'n' then all 'C' from the same weights; this rank's comparison and tape counts."""
    keep = Layout(stages=recompute_layout.stages, chunks=recompute_layout.chunks, recompute=False)
    with parallel_layout(recompute_layout):
        batches = rank_batches()
        config = tiny_config(dtype, keep)
        models = build_chunks(config, keep)
        state = state_snapshot(models)
        kept = run_step(wrap_ddp(models, config), keep, batches)
        del models

        config = tiny_config(dtype, recompute_layout)
        models = build_chunks(config, recompute_layout)
        load_state(models, state)
        ddp = wrap_ddp(models, config)
        with _Tapes() as tapes, torch.utils.checkpoint.set_checkpoint_early_stop(False):
            checkpointed = run_step(ddp, recompute_layout, batches)
        microbatches = len(batches)
        expected_tapes = sum(
            ROUNDS * microbatches + (microbatches if model.decoder.loop_ring.round_start else 0)
            for model in models
        )
        moe_per_chunk = [_moe_sublayers(model) for model in models]
        lengths = sorted({len(tape.indices) for tape in tapes.tapes})
        result = dict(
            grads_bitwise=all(
                torch.equal(checkpointed.grads[key], kept.grads[key]) for key in kept.grads
            ),
            grads_differ=sorted(
                key
                for key in kept.grads
                if not torch.equal(checkpointed.grads[key], kept.grads[key])
            )[:8],
            same_keys=set(kept.grads) == set(checkpointed.grads),
            reports_bitwise=kept.reports == checkpointed.reports,
            reports=len(kept.reports),
            bias_bitwise=all(
                torch.equal(checkpointed.expert_bias[key], kept.expert_bias[key])
                for key in kept.expert_bias
            ),
            tapes=len(tapes.tapes),
            expected_tapes=expected_tapes,
            replays=sorted({tapes.replays.get(id(tape), 0) for tape in tapes.tapes}),
            fully_replayed=all(tape.cursor == len(tape.indices) for tape in tapes.tapes),
            lengths=lengths,
            moe_per_chunk=moe_per_chunk,
            recorded_recompute=checkpointed.record.model_work[2] > 0,
        )
        del ddp, models
    return result


@pytest.fixture(scope="module")
def recompute():
    """Every layout's two runs (fp32 unless the fp32 run raises)."""

    def compute(dtype):
        return {layout.name: gather(_one(dtype, layout)) for layout in LAYOUTS}

    with determinism_env(), deterministic_algorithms(), full_precision_matmuls():
        dtype, probe, results = run_with_precision(compute)
    yield dict(dtype=dtype, probe=probe, results=results)


@pytest.mark.parametrize("name", NAMES)
def test_recompute_is_bitwise(recompute, name):
    """Plan 'C' equals plan 'n' bitwise: gradients, losses and expert biases on every rank."""
    ranks = recompute["results"][name]
    for rank, result in enumerate(ranks):
        assert result["same_keys"], rank
        assert result["grads_bitwise"], (rank, result["grads_differ"])
        assert result["reports_bitwise"], rank
        assert result["bias_bitwise"], rank
        assert result["recorded_recompute"], rank
    print(
        f"loop ring recompute {name} ({recompute['dtype']}): gradients, "
        f"{sum(result['reports'] for result in ranks)} sequence losses and expert biases "
        "bitwise equal to plan 'n' on every rank"
    )


@pytest.mark.parametrize("name", NAMES)
def test_one_routing_tape_per_checkpoint_call(recompute, name):
    """One tape per checkpoint call, each replayed once and to its end."""
    for rank, result in enumerate(recompute["results"][name]):
        assert result["tapes"] == result["expected_tapes"], (rank, result)
        assert result["replays"] == [1], (rank, result["replays"])
        assert result["fully_replayed"], rank
        lengths = [length for length in result["lengths"] if length]
        assert len(lengths) == 1, (rank, result["lengths"])
        moe = {count for count in result["moe_per_chunk"] if count}
        assert all(lengths[0] % count == 0 for count in moe), (rank, lengths, moe)
    first = recompute["results"][name][0]
    print(
        f"loop ring tapes {name}: rank 0 made {first['tapes']} tapes (expected "
        f"{first['expected_tapes']}), each replayed once and fully; router calls per tape "
        f"{first['lengths']}, core MoE sublayers per chunk {first['moe_per_chunk']}"
    )
