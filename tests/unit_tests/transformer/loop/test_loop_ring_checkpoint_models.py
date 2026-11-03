# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training-side support of the loop ring's multi-chunk models (CPU).

* ``loop_hybrid_models`` returns every chunk of a k-chunk ring stage (through the wrappers);
  ``loop_hybrid_model`` stays the one-chunk accessor; a list mixing loop and other chunks raises.
* The checkpoint coverage check runs on every chunk (an uncovered tensor on chunk j is named
  with its chunk), with the per-chunk dicts of a save / load (``loop_model_state_dicts``).
* The ``loop_resume`` entry and the resume check read the roles of the first chunk (every ring
  chunk carries the model's roles); carrier diagnostics refuse a multi-chunk model.
* The unique-parameter census (``loop_ring_report``): the pipeline sum of the stages' non-expert
  parameters without the MTP stage's embedding copy plus EP x their routed experts equals the
  PP = 1 model's parameter count, and the raw stage sum exceeds it by the copy (V x h).
* The per-tensor checksums keyed by global name: the ring stages' lines equal the PP = 1
  model's (layer offsets, the embedding copy merged); routed experts get their global index; a
  disagreement prints ``MISMATCH``.
* ``get_loop_ring_forward_backward_func``: the ring schedule for ``loop_schedule='ring'`` and its
  refusals.
"""

import re
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.core.transformer.loop.ring_pp import get_loop_ring_schedule
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
from megatron.training import loop_checkpoint as lc
from megatron.training import loop_ring_report as report
from tests.unit_tests.transformer.loop import hybrid_loop_utils as hlu
from tests.unit_tests.transformer.loop import ring_emulation as emu
from tests.unit_tests.transformer.loop.ring_utils import StageGroup, stage_group_ranks

ROUNDS = emu.ROUNDS

pytestmark = pytest.mark.skipif(
    torch.cuda.is_available(), reason="CPU ring chunks (the GPU ring runs through job-launch)"
)

mhc_on_this_device = hlu.mhc_on_this_device
cpu_mtp = hlu.cpu_mtp


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with hlu.model_parallel_once():
        yield


@pytest.fixture
def cpu_ring(mhc_on_this_device, cpu_mtp, monkeypatch):
    """The emulation's CPU environment (router buffers on the CPU)."""
    del mhc_on_this_device, cpu_mtp
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")


class _Wrapped(Float16Module):
    """A ``Float16Module`` around a chunk without its dtype conversion (``unwrap_model`` walks
    it like the real one)."""

    def __init__(self, module):
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        self.module = module


def _reference_and_ring(kind, stages, chunks):
    main, mtp = emu.patterns(kind)
    set_moe_metrics_tracker(MoEMetricsTracker())
    reference = emu.build_model(kind, emu.model_config(kind), main + "/" + mtp)
    state = {
        name: value.detach().clone() if torch.is_tensor(value) else value
        for name, value in reference.state_dict().items()
    }
    return reference, emu.build_ring(kind, stages, chunks, "all-C", state)


def _stage(ring, stage):
    """Stage ``stage``'s chunks, wrapped, in chunk order."""
    return [_Wrapped(ring.chunk_model(stage, chunk)) for chunk in range(ring.chunks)]


# ---------------------------------------------------------------------------------------------
# loop_hybrid_models and the coverage check
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("stages,chunks", [(2, 1), (2, 2), (4, 2)])
def test_loop_hybrid_models_covers_every_chunk(cpu_ring, stages, chunks):
    del cpu_ring
    _, ring = _reference_and_ring("dense", stages, chunks)
    for stage in range(stages):
        wrapped = _stage(ring, stage)
        models = [ring.chunk_model(stage, chunk) for chunk in range(chunks)]
        assert lc.loop_hybrid_models(wrapped) == models
        assert lc.loop_hybrid_models(wrapped[0]) == models[:1]
        assert lc.loop_hybrid_model(wrapped) is (models[0] if chunks == 1 else None)
    assert lc.loop_hybrid_models([nn.Linear(2, 2)]) == []
    with pytest.raises(ValueError, match="mixes hybrid loop chunks"):
        lc.loop_hybrid_models([_stage(ring, 0)[0], nn.Linear(2, 2)])


class _Uncovered(nn.Module):
    """A child whose weight has no checkpoint entry (its sharded state dict is empty)."""

    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, hidden))

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        del prefix, sharded_offsets, metadata
        return {}


@pytest.mark.parametrize("stages,chunks", [(2, 2), (4, 2)])
def test_coverage_check_runs_on_every_chunk(cpu_ring, stages, chunks):
    del cpu_ring
    _, ring = _reference_and_ring("dense", stages, chunks)
    wrapped = _stage(ring, stages - 1)
    with stage_group_ranks():
        lc.check_loop_sharded_state_dict(wrapped, when="setup")
        dicts = [model.sharded_state_dict() for model in lc.loop_hybrid_models(wrapped)]
        # The save / load form: generate_state_dict's 'model<i>' entries, one per chunk.
        state = {f"model{index}": entry for index, entry in enumerate(dicts)}
        assert lc.loop_model_state_dicts(state, wrapped) == dicts
        assert lc.loop_model_state_dicts({"model": dicts[0]}, wrapped[:1]) is dicts[0]
        context = {}
        lc.check_loop_sharded_state_dict(
            wrapped, when="save", sharded_state_dict=dicts, context=context
        )
        assert context[lc.COVERAGE_CHECKED_KEY]
        # An uncovered tensor on the last chunk only: named with its chunk.
        last = lc.loop_hybrid_models(wrapped)[-1]
        last.decoder.carrier_cell = _Uncovered(hlu.HIDDEN)
        with pytest.raises(RuntimeError, match=rf"\(setup\) chunk {chunks - 1}: .*carrier_cell"):
            lc.check_loop_sharded_state_dict(wrapped, when="setup")
        stale = dicts[:-1] + [last.sharded_state_dict()]
        with pytest.raises(RuntimeError, match=rf"\(load\) chunk {chunks - 1}"):
            lc.check_loop_sharded_state_dict(wrapped, when="load", sharded_state_dict=stale)
        with pytest.raises(ValueError, match="one sharded state dict for"):
            lc.check_loop_sharded_state_dict(wrapped, when="load", sharded_state_dict=dicts[0])
        # The first save already passed in this context: not checked again.
        lc.check_loop_sharded_state_dict(wrapped, when="save", context=context)


def test_resume_entry_and_check_read_the_first_chunk(cpu_ring):
    del cpu_ring
    reference, ring = _reference_and_ring("dense", 2, 2)
    wrapped = _stage(ring, 1)
    args = SimpleNamespace(
        hybrid_layer_pattern="P", expert_model_parallel_size=1, world_size=2, loop_planner="off"
    )
    entry = lc.loop_resume_state(args, wrapped, None)
    roles = reference.decoder.loop_roles
    assert entry["roles"] == {"prelude": roles.prelude, "core": roles.core, "coda": roles.coda}
    saved = {"args": SimpleNamespace(), "loop_resume": dict(entry)}
    assert lc.check_loop_resume_state(args, wrapped, saved, restore_training_state=True) == []
    changed = dict(entry, roles=dict(entry["roles"], core=roles.core - 2))
    with pytest.raises(RuntimeError, match="loop roles"):
        lc.check_loop_resume_state(
            args,
            wrapped,
            {"args": SimpleNamespace(), "loop_resume": changed},
            restore_training_state=True,
        )


def test_carrier_diagnostics_refuse_a_multi_chunk_model(cpu_ring):
    del cpu_ring
    from megatron.training.loop_diagnostics import record_loop_carrier_diagnostics

    _, ring = _reference_and_ring("dense", 2, 2)
    args = SimpleNamespace(loop_carrier_diagnostics="full", tensorboard_log_interval=1)
    with pytest.raises(ValueError, match="one-chunk loop model"):
        record_loop_carrier_diagnostics(args, _stage(ring, 0), 1)


# ---------------------------------------------------------------------------------------------
# Census and checksums
# ---------------------------------------------------------------------------------------------


def _census_args():
    return SimpleNamespace(
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        num_experts=emu.EXPERTS,
        loop_param_store_shard_size=1,
    )


@pytest.mark.parametrize("stages,chunks", [(2, 1), (2, 2), (4, 1)])
def test_census_counts_every_parameter_once(cpu_ring, monkeypatch, stages, chunks):
    del cpu_ring
    reference, ring = _reference_and_ring("moe", stages, chunks)
    args = _census_args()
    total = sum(param.numel() for param in reference.parameters())
    pp1 = report.loop_param_census([reference], args)
    assert pp1["unique"] == total
    assert pp1["per_rank"][1] == total
    # One process holds every stage: count each stage alone (no pipeline all-reduce) and add.
    monkeypatch.setattr(report, "_pipeline_group", lambda model: None)
    dense = routed = raw = 0
    for stage in range(stages):
        census = report.loop_param_census(_stage(ring, stage), args)
        dense += census["dense"]
        routed += census["routed"]
        raw += sum(
            param.numel()
            for chunk in range(chunks)
            for param in ring.chunk_model(stage, chunk).parameters()
        )
    assert dense == pp1["dense"] and routed == pp1["routed"]
    copy = hlu.VOCAB * hlu.HIDDEN  # the MTP stage's word-embedding copy (V x h)
    assert raw == total + copy
    assert routed > 0 and set(pp1["per_rank"]) == {1, 2, 4}
    assert pp1["per_rank"][4] == pp1["dense"] + pp1["routed"] // 4


def test_census_is_none_for_other_models():
    assert report.loop_param_census([nn.Linear(2, 2)], _census_args()) is None


@pytest.mark.parametrize("kind,stages,chunks", [("dense", 2, 2), ("moe", 2, 1), ("moe", 4, 2)])
def test_checksums_of_the_ring_stages_equal_pp1(cpu_ring, kind, stages, chunks):
    del cpu_ring
    reference, ring = _reference_and_ring(kind, stages, chunks)
    pp1 = report.loop_param_checksum_lines([report.loop_param_checksums([reference])])
    stages_lines = report.loop_param_checksum_lines(
        [report.loop_param_checksums(_stage(ring, stage)) for stage in range(stages)]
    )
    assert stages_lines == pp1
    assert not any("MISMATCH" in line for line in pp1)
    names = [re.match(r"loop param checksum (\S+):", line).group(1) for line in pp1]
    assert "decoder.loop_adapter.weight" in names and "embedding.word_embeddings.weight" in names
    assert (
        len(names)
        == len(set(names))
        == len([n for n, v in reference.state_dict().items() if torch.is_tensor(v)])
    )
    # A changed weight on one stage shows as two values of one name.
    model = ring.chunk_model(stages - 1, chunks - 1)
    with torch.no_grad():
        model.embedding.word_embeddings.weight[0, 0] += 1.0
    lines = report.loop_param_checksum_lines(
        [report.loop_param_checksums(_stage(ring, stage)) for stage in range(stages)]
    )
    mismatched = [line for line in lines if "MISMATCH" in line]
    assert len(mismatched) == 2
    assert all("embedding.word_embeddings.weight" in line for line in mismatched)


class _Experts(nn.Module):
    """A routed-experts module (``num_local_experts``, ``ep_group``) with TE-grouped or
    sequential parameter names."""

    def __init__(self, ep_rank, ep_size, local, sequential):
        super().__init__()
        self.num_local_experts = local
        self.ep_group = StageGroup(ep_rank, ep_size)
        if sequential:
            self.local_experts = nn.ModuleList(
                [nn.ModuleDict({"linear_fc1": nn.Linear(2, 2, bias=False)}) for _ in range(local)]
            )
        else:
            self.linear_fc1 = nn.Module()
            for index in range(local):
                self.linear_fc1.register_parameter(f"weight{index}", nn.Parameter(torch.ones(2)))


def test_global_names_of_routed_experts_and_layers():
    for sequential in (False, True):
        layer = nn.Module()
        layer.mlp = nn.Module()  # the MoE layer: router, shared expert, routed experts
        layer.mlp.num_local_experts, layer.mlp.ep_group = 4, StageGroup(1, 2)
        layer.mlp.router = nn.Linear(2, 2, bias=False)
        layer.mlp.shared_experts = nn.Linear(2, 2, bias=False)
        layer.mlp.experts = _Experts(1, 2, 4, sequential)
        root = nn.Module()
        root.decoder = nn.Module()
        root.decoder.layers = nn.ModuleList([layer])
        routed = report._routed_experts(root)
        assert routed == [("decoder.layers.0.mlp.experts", 4, 1)]
        names = {name: report._global_name(name, 10, routed) for name, _ in root.named_parameters()}
        assert names["decoder.layers.0.mlp.router.weight"] == "decoder.layers.10.mlp.router.weight"
        assert names["decoder.layers.0.mlp.shared_experts.weight"] == (
            "decoder.layers.10.mlp.shared_experts.weight"
        )
        if sequential:
            assert names["decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight"] == (
                "decoder.layers.10.mlp.experts.local_experts.5.linear_fc1.weight"
            )
        else:
            assert names["decoder.layers.0.mlp.experts.linear_fc1.weight3"] == (
                "decoder.layers.10.mlp.experts.linear_fc1.weight7"
            )


# ---------------------------------------------------------------------------------------------
# The training helper
# ---------------------------------------------------------------------------------------------


def _helper_args(**overrides):
    fields = dict(
        cuda_graph_impl="none",
        use_megatron_fsdp=False,
        use_torch_fsdp2=False,
        save_activations_interval=None,
        save_dgrads_interval=None,
        save_tokens_per_expert_interval=None,
        defer_embedding_wgrad_compute=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_training_selects_the_ring_schedule_and_refuses_what_it_cannot_run():
    from megatron.training.training import get_loop_ring_forward_backward_func

    ring = SimpleNamespace(loop_schedule="ring", defer_embedding_wgrad_compute=False)
    assert get_loop_ring_forward_backward_func(ring, _helper_args()) is get_loop_ring_schedule()
    standard = SimpleNamespace(loop_schedule="standard")
    assert get_loop_ring_forward_backward_func(standard, _helper_args()) is None
    for name, value, message in (
        ("cuda_graph_impl", "full_iteration", "cuda_graph_impl=full_iteration"),
        ("use_megatron_fsdp", True, "use_megatron_fsdp"),
        ("use_torch_fsdp2", True, "use_torch_fsdp2"),
        ("save_activations_interval", 1, "save_activations_interval"),
        ("save_dgrads_interval", 1, "save_dgrads_interval"),
        ("save_tokens_per_expert_interval", 1, "save_tokens_per_expert_interval"),
        ("defer_embedding_wgrad_compute", True, "defer_embedding_wgrad_compute"),
    ):
        with pytest.raises(ValueError, match=f"loop_schedule='ring' does not support {message}"):
            get_loop_ring_forward_backward_func(ring, _helper_args(**{name: value}))
    # The wgrad dump stays allowed (force_all_reduce passes through).
    assert get_loop_ring_forward_backward_func(ring, _helper_args(save_wgrads_interval=1))
