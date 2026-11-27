# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Terminal residual-gate rematerialization on a two-pass PP2/EP4 DSA/MoK/MTP ring."""

from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.forward_replay import in_forward_replay
from megatron.core.transformer.loop import glm_ring_stack, ring_pp
from megatron.core.transformer.loop.hybrid_stack import LoopResidualGate
from tests.unit_tests.transformer.loop import glm_ring_mok_utils as moe
from tests.unit_tests.transformer.loop import test_glm_ring_mok_recompute_gpu as oracle
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA


@pytest.mark.skipif(not CUDA or oracle.ring.WORLD != 8, reason="requires eight CUDA ranks")
def test_gate_replay_preserves_both_source_adjoints_and_saves_no_intermediates(monkeypatch):
    modes = iter((False, True))
    gates = {}
    saved = defaultdict(list)
    counters = defaultdict(Counter)
    gradients = defaultdict(dict)
    phase = False
    examples = {}
    gate_forward = LoopResidualGate.forward
    checkpoint = glm_ring_stack.checkpoint
    add_gradient = ring_pp.RingInjection.add

    def models(plans, *, layernorm=False):
        nonlocal phase
        phase = next(modes)
        config = replace(
            moe.config(2),
            num_attention_heads=64,
            num_query_groups=64,
            q_lora_rank=512,
            kv_lora_rank=512,
            qk_head_dim=128,
            qk_pos_emb_head_dim=64,
            v_head_dim=128,
            kv_channels=128,
            attention_backend=AttnBackend.auto,
            dsa_kernel_backend="cudnn",
            dsa_indexer_head_dim=128,
            dsa_indexer_n_heads=32,
            dsa_indexer_topk=64,
            dsa_indexer_reuse_checkpoint_topk=True,
            dsa_indexer_weights_proj_use_quantization=False,
            dsa_indexer_weights_proj_output_dtype="bf16",
            linear_cp_mode=None,
            linear_cp_layout=None,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=4,
            loop_recompute=True,
            loop_recompute_edges=True,
            loop_recompute_block_stride=4,
            loop_ring_recompute_plan="/".join(plans),
            loop_recompute_ring_gates=phase,
        )
        with patch.object(moe, "config", return_value=config):
            chunks = moe.models(2)
        for chunk in chunks:
            for lap, gate in enumerate(getattr(chunk.decoder, "loop_gates", ())):
                gates[id(gate)] = lap + 1
        return chunks

    @contextmanager
    def observe_saved(gate, previous, candidate):
        # Save no Python references to intermediate tensors in the observer. Exclude the
        # existing gate inputs and weights by storage identity, accounting for views once.
        base = {t.untyped_storage().data_ptr() for t in (previous, candidate, *gate.parameters())}
        extra = {}

        def pack(tensor):
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in base:
                extra[storage.data_ptr()] = storage.nbytes()
            return tensor.detach()

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            yield
        saved[phase, gates[id(gate)]].append(sum(extra.values()))

    def forward(gate, previous, candidate):
        replay = torch._C._current_graph_task_id() >= 0
        counters[phase]["gate", gates[id(gate)], replay] += 1
        if phase:
            return gate_forward(gate, previous, candidate)
        with observe_saved(gate, previous, candidate):
            output = gate_forward(gate, previous, candidate)
        lap = gates[id(gate)]
        if lap not in examples:
            # Keep a single actual input pair and its true LM/MTP cotangent for this lap.
            # It isolates the pure gate from low-bit BF16 changes upstream in the MoE run.
            sample = dict(
                module=deepcopy(gate), inputs=[t.detach().clone() for t in (previous, candidate)]
            )
            examples[lap] = sample

            def capture_cotangent(grad):
                sample["cotangent"] = grad.detach().clone()

            output.register_hook(capture_cotangent)
        return output

    def checkpoint_gate(gate, previous, candidate, **kwargs):
        # Install the observer *outside* the non-reentrant checkpoint. Installing a saved
        # tensor hook in gate.forward would replace PyTorch's own rematerialization hook.
        with observe_saved(gate, previous, candidate):
            return checkpoint(gate, previous, candidate, **kwargs)

    def observe_event(name, function):
        def wrapped(*args, **kwargs):
            key = args[0].key if name == "use" else args[2]
            counters[phase][name, tuple(key), torch._C._current_graph_task_id() >= 0] += 1
            return function(*args, **kwargs)

        return wrapped

    def capture_adjoint(entry, gradient):
        assert entry.key not in gradients[phase]
        gradients[phase][entry.key] = gradient.detach().float().cpu()
        return add_gradient(entry, gradient)

    monkeypatch.setattr(oracle, "_models", models)
    monkeypatch.setattr(
        oracle,
        "in_forward_replay",
        lambda: in_forward_replay() or torch._C._current_graph_task_id() >= 0,
    )
    monkeypatch.setattr(LoopResidualGate, "forward", forward)
    monkeypatch.setattr(glm_ring_stack, "checkpoint", checkpoint_gate)
    monkeypatch.setattr(
        glm_ring_stack,
        "ring_injection_use",
        observe_event("use", glm_ring_stack.ring_injection_use),
    )
    monkeypatch.setattr(
        glm_ring_stack,
        "_publish_gate_input",
        observe_event("publish", glm_ring_stack._publish_gate_input),
    )
    monkeypatch.setattr(ring_pp.RingInjection, "add", capture_adjoint)
    oracle._compare_plans(("CCCC", "CCCC"), ("CCCC", "CCCC"))

    native, remat = counters[False], counters[True]
    expected_replays = Counter(
        {(name, key, True): n for (name, key, _), n in native.items() if name == "gate"}
    )
    assert remat == native + expected_replays
    assert gradients[False].keys() == gradients[True].keys()
    for key, adjoint in gradients[False].items():
        assert adjoint.norm() > 0
        oracle.layout._assert_gradient(gradients[True][key], adjoint, f"terminal source {key}")
    relative = (
        max(
            ((gradients[True][key] - adjoint).norm() / adjoint.norm()).item()
            for key, adjoint in gradients[False].items()
        )
        if gradients[False]
        else 0
    )
    if gradients[False]:
        assert {lap for _, lap in gradients[False]} == {0, 1}
        for lap in (1, 2):
            sample = examples[lap]
            module = sample["module"]

            def actual_gate(p, c):
                return gate_forward(module, p, c)

            def isolated(remat):
                inputs = tuple(t.detach().requires_grad_() for t in sample["inputs"])
                y = (
                    checkpoint(actual_gate, *inputs, use_reentrant=False, preserve_rng_state=False)
                    if remat
                    else actual_gate(*inputs)
                )
                grad = torch.autograd.grad(y, (*inputs, *module.parameters()), sample["cotangent"])
                return (y, *grad)

            for candidate_tensor, baseline_tensor in zip(
                isolated(True), isolated(False), strict=True
            ):
                torch.testing.assert_close(candidate_tensor, baseline_tensor, rtol=0, atol=0)
            baseline, candidate = saved[False, lap], saved[True, lap]
            assert len(baseline) == len(candidate) == native["gate", lap, False]
            assert all(size > 0 for size in baseline) and all(size == 0 for size in candidate)
            print(
                f"gate lap {lap}: extra saved storage bytes {baseline} -> {candidate}; "
                f"{remat['gate', lap, True]} replays, {len(gradients[False])} matched source adjoints; maximum relative L2 {relative:.7g}, exact fixed-cotangent gate F+B"
            )
