# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The checkpoint rules of the loop parameter store (``megatron/training/loop_checkpoint.py``,
M3c), on CPU with stub argument namespaces and a local loop model (one-rank Gloo):

* the argument rules (``loop_checkpoint_problems``, ``_validate_loop_checkpoint``): saving
  other than torch_dist (message 1), a local non-persistent checkpoint, the mem-efficient fully
  reshardable optimizer state (message 4), the chunked optimizer-state offload (message 10), the
  ``distributed_weight`` binding with ``--save`` / ``--load`` (message 12); and what passes (no
  store, torch_dist, ``--load`` alone with a legacy format, the default binding);
* the save backstop (``check_loop_save_format``, message 1 and the local message) and the load
  format rules (``check_loop_load_format``: a legacy checkpoint's optimizer state, message 2;
  ``torch_dcp`` / ``fsdp_dtensor``; the GPT-to-hybrid maps, message 11), silent without a store;
* the bucket-layout check (``check_loop_optimizer_resharding``, message 3): equal layouts pass,
  a different ``per_bucket_numel_unpadded`` on one data-parallel group index raises on every rank
  (the stubbed MAX reduction), equal bucket totals with another z or stored set raise, a
  checkpoint without a bucket record (saved without a store) raises, the model-space formats and store-free runs pass;
* the ``loop_resume`` entry (``loop_resume_state``): its fields, plain types that survive
  ``torch.load(weights_only=True)``, None for non-loop models; every rule is a no-op for a
  non-loop model.
* the resume check (``check_loop_resume_state``, message 5) against the checkpoint's saved
  ``args`` and ``loop_resume`` entry: equal runs pass silently; a numerics-relevant ``loop_*``
  change, a role or pattern change refuse (every change named) unless ``--loop-resume-changes
  allow``; the layout-neutral arguments (store layout: a note; planner inputs, paths, reports,
  debug: silent), the whole round <-> nested swap (a note; the block stride ignored), world / EP
  changes and store layout changes are notes; block / stride / waves changes refuse; an argument
  the checkpoint predates compares against its parser default; a checkpoint without the entry
  is a note; non-loop models, finetuning and ``restore_training_state=False`` return [] and log
  nothing; the CLI flag; a golden resume class (refused / note / silent) for every ``loop_*``
  parser argument, so a new argument needs an explicit decision;
* the RSBF plan on resume (``loop_rsbf_planning``): ``loop_plan_record`` (plain types, in
  ``loop_resume`` when a plan is applied), ``check_resume_plan`` (a choice change refused, a
  digest-only change accepted, whole round <-> nested a note, the depth distribution compared
  only when both runs plan from the exit criterion), ``pin_saved_plan`` (in the menu and feasible:
  pinned; not in the menu / infeasible: the reason), ``read_saved_plan`` on a real torch_dist
  common state (None under ``--non-persistent-ckpt-type local``), and the session pin under ``--loop-planner apply`` (pinned choice applied to the
  config; a failed pin planned afresh and refused by the resume check, ``allow`` passes);
* the E18 pin: building the pretraining data loader and creating its iterator leaves the CPU
  default generator (the eager stack's ``--loop-sample-depth`` draws, restored by the
  checkpoint's RNG state) unchanged.
"""

import contextlib
import io
import os
import re
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.core import parallel_state
from megatron.core.transformer.loop import planner as P
from megatron.core.transformer.loop.param_store import StoreModes
from megatron.training import loop_checkpoint as lc
from megatron.training import loop_rsbf_planning as planning
from megatron.training.arguments import _validate_loop_checkpoint
from tests.unit_tests.transformer.loop import param_store_utils as psu
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    build_local_loop_model,
    model_parallel_once,
)

DENSE = "*-*-*-*-"

# The torch_dist save / read tests write to the per-rank ``tmp_path`` and assume a one-rank
# world (a collective save over W ranks with W different directories hangs); under torchrun the
# multi-rank path is the GPU continuity tests' and the toolkit chain C2a's.
ONE_RANK_ONLY = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) > 1,
    reason="one-rank torch_dist save into the per-rank tmp_path (multi-rank: chain C2a)",
)


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


@pytest.fixture
def session():
    """A fresh planner session (the process's applied plan feeds ``loop_resume_state``)."""
    planning.reset_loop_planner_session()
    try:
        yield
    finally:
        planning.reset_loop_planner_session()


def _args(**overrides):
    values = dict(
        loop_param_store_shard_size=8,
        loop_param_store_binding="data",
        save="/ckpt",
        load=None,
        pretrained_checkpoint=None,
        ckpt_format="torch_dist",
        non_persistent_ckpt_type=None,
        non_persistent_save_interval=None,
        use_distributed_optimizer=True,
        dist_ckpt_optim_fully_reshardable=False,
        distrib_optim_fully_reshardable_mem_efficient=False,
        chunked_optimizer_state_offload=False,
        offload_optimizer_states=False,
        optimizer_state_offload_fraction=1.0,
        optimizer_cpu_offload=False,
        expert_model_parallel_size=1,
        world_size=8,
        hybrid_layer_pattern=DENSE,
    )
    values.update(overrides)
    return Namespace(**values)


def _store_model(**config):
    model = build_local_loop_model(DENSE, prelude=2, coda=2)
    store = psu.install_store(
        model, psu.trivial_store_groups(), 1, allow_trivial_shard=True, **config
    )
    return model, store


def _plain_model():
    return build_local_loop_model(DENSE, prelude=2, coda=2)


# ---------------------------------------------------------------------------------------------
# Argument rules
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides,match",
    [
        (dict(ckpt_format="torch"), r"--ckpt-format torch with the loop parameter store \(z = 8\)"),
        (dict(ckpt_format="torch_dcp"), "--ckpt-format torch_dcp with .* z - 1 of the z"),
        (dict(ckpt_format="fsdp_dtensor"), "save with --ckpt-format torch_dist"),
        (
            dict(non_persistent_ckpt_type="local", non_persistent_save_interval=10),
            r"local non-persistent checkpoints \(--non-persistent-ckpt-type local\)",
        ),
        (
            dict(
                dist_ckpt_optim_fully_reshardable=True,
                distrib_optim_fully_reshardable_mem_efficient=True,
            ),
            "--distrib-optim-fully-reshardable-mem-efficient with the loop parameter store",
        ),
        (
            dict(chunked_optimizer_state_offload=True, save=None),
            r"--chunked-optimizer-state-offload .* is untested \(S8\.0 chunked-offload arm",
        ),
        (dict(offload_optimizer_states=True), "chunked-optimizer-state-offload"),
        (
            dict(loop_param_store_binding="distributed_weight", save=None, load="/ckpt"),
            "checkpointing with --loop-param-store-binding distributed_weight is untested",
        ),
    ],
)
def test_argument_rules(overrides, match):
    problems = lc.loop_checkpoint_problems(_args(**overrides))
    assert len(problems) == 1 and re.search(match, problems[0]), problems
    with pytest.raises(ValueError, match="loop parameter store checkpoints: "):
        _validate_loop_checkpoint(_args(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        dict(),  # torch_dist save
        dict(save=None, load="/ckpt", ckpt_format="torch"),  # a legacy load (weights, msg 2)
        dict(loop_param_store_shard_size=1, ckpt_format="torch"),  # no store
        dict(loop_param_store_shard_size=None, ckpt_format="torch"),
        dict(dist_ckpt_optim_fully_reshardable=True),
        dict(distrib_optim_fully_reshardable_mem_efficient=True, save=None),  # no checkpoint
        dict(chunked_optimizer_state_offload=True, optimizer_state_offload_fraction=0.0),
        dict(optimizer_cpu_offload=True),  # HDO: decided by the GPU phase (not refused)
        dict(loop_param_store_binding="distributed_weight", save=None),
        dict(non_persistent_ckpt_type="global", non_persistent_save_interval=10),
    ],
)
def test_what_the_rules_accept(overrides):
    assert lc.loop_checkpoint_problems(_args(**overrides)) == []
    _validate_loop_checkpoint(_args(**overrides))


# ---------------------------------------------------------------------------------------------
# Save and load formats
# ---------------------------------------------------------------------------------------------


def test_save_and_load_format_backstops():
    model, _ = _store_model()
    plain = _plain_model()
    lc.check_loop_save_format([model], ckpt_format="torch_dist")
    with pytest.raises(RuntimeError, match=r"--ckpt-format torch with the loop parameter store"):
        lc.check_loop_save_format([model], ckpt_format="torch")
    with pytest.raises(RuntimeError, match="local non-persistent checkpoints"):
        lc.check_loop_save_format([model], ckpt_format="torch_dist", local=True)
    for fmt in ("torch", "torch_dcp", "fsdp_dtensor"):
        lc.check_loop_save_format([plain], ckpt_format=fmt, local=True)  # no store: silent
        lc.check_loop_load_format([plain], ckpt_format=fmt, load_optim=True, gpt_compat=True)

    lc.check_loop_load_format([model], ckpt_format="torch_dist", load_optim=True, gpt_compat=False)
    lc.check_loop_load_format([model], ckpt_format="torch", load_optim=False, gpt_compat=False)
    lc.check_loop_load_format([model], ckpt_format=None, load_optim=True, gpt_compat=False)
    message = (
        r"loading the optimizer state of a legacy \(torch\) checkpoint into a loop parameter "
        r"store run \(z = 1\) is not supported: .* --no-load-optim"
    )
    with pytest.raises(RuntimeError, match=message):
        lc.check_loop_load_format([model], ckpt_format="torch", load_optim=True, gpt_compat=False)
    for fmt in ("torch_dcp", "fsdp_dtensor"):
        with pytest.raises(RuntimeError, match=f"loading a {fmt} checkpoint into a loop"):
            lc.check_loop_load_format([model], ckpt_format=fmt, load_optim=False, gpt_compat=False)
    with pytest.raises(RuntimeError, match="GPT-to-hybrid checkpoint maps with the loop"):
        lc.check_loop_load_format(
            [model], ckpt_format="torch_dist", load_optim=False, gpt_compat=True
        )


# ---------------------------------------------------------------------------------------------
# The dp_reshardable bucket layout (message 3)
# ---------------------------------------------------------------------------------------------


class _DistOpt:
    """What the check reads of a distributed-optimizer instance."""

    def __init__(self, group_idx, numels):
        self.data_parallel_group_idx = group_idx
        self.per_bucket_numel_unpadded = numels


def _chained(*instances):
    return SimpleNamespace(chained_optimizers=list(instances))


def test_bucket_layout_check():
    model, store = _store_model()
    plain = _plain_model()
    dense = _DistOpt(0, [{"torch.float32": [1024, 512]}])
    experts = _DistOpt(3, [{"torch.float32": [256]}])
    optimizer = _chained(dense, experts)
    saved = lc.loop_resume_state(_args(), [model], optimizer)
    assert saved["buckets"] == {
        "0": {"0": [{"torch.float32": [1024, 512]}]},
        "1": {"3": [{"torch.float32": [256]}]},
    }
    calls = []

    def reduce_max(flag):
        calls.append(flag)
        return flag

    lc.check_loop_optimizer_resharding(
        _args(), [model], optimizer, saved, "dp_reshardable", reduce_max=reduce_max
    )
    assert calls == [0]
    # Another data-parallel group index of the same instance in the record: still this rank's.
    other = dict(saved, buckets={"0": {"0": saved["buckets"]["0"]["0"], "1": [[7]]}, "1": {}})
    other["buckets"]["1"] = saved["buckets"]["1"]
    lc.check_loop_optimizer_resharding(
        _args(), [model], optimizer, other, "dp_reshardable", reduce_max=reduce_max
    )
    # A different layout on one rank only: the stubbed MAX makes every rank raise.
    changed = _chained(_DistOpt(0, [{"torch.float32": [1024, 500]}]), experts)
    message = (
        r"the checkpoint's distributed-optimizer state is in the dp_reshardable \(bucket\) "
        r"format, whose layout depends on the loop parameter store \(saved: z = 1, \d+ stored "
        r"tensors \(set [0-9a-f]{12}\), min_numel 1, EP 1; this run: z = 1, \d+ stored tensors "
        r"\(set [0-9a-f]{12}\), min_numel 1, EP 1\); "
        r"it reshards over the data-parallel \(replicate\) degree only. Save with "
        r"--dist-ckpt-optim-fully-reshardable"
    )
    with pytest.raises(RuntimeError, match=message):
        lc.check_loop_optimizer_resharding(_args(), [model], changed, saved, "dp_reshardable")
    with pytest.raises(RuntimeError, match=message):  # this rank equal, another rank not
        lc.check_loop_optimizer_resharding(
            _args(), [model], optimizer, saved, "dp_reshardable", reduce_max=lambda flag: 1
        )
    # Equal per-bucket totals, but another z or stored set: refused (misordered state).
    for key, value in (("z", 2), ("stored_sha", "0" * 64)):
        moved = dict(saved, store=dict(saved["store"], **{key: value}))
        with pytest.raises(RuntimeError, match=r"dp_reshardable \(bucket\) format, whose"):
            lc.check_loop_optimizer_resharding(
                _args(), [model], optimizer, moved, "dp_reshardable", reduce_max=reduce_max
            )
    # A checkpoint saved without a store (no record, or no loop_resume at all).
    for record in (dict(saved, buckets=None), None):
        with pytest.raises(RuntimeError, match=r"saved without a loop parameter store"):
            lc.check_loop_optimizer_resharding(
                _args(), [model], optimizer, record, "dp_reshardable", reduce_max=reduce_max
            )
    # Model-space formats reshard; store-free runs keep dev's behaviour.
    for sharding in ("fully_reshardable", "fully_sharded_model_space", None):
        lc.check_loop_optimizer_resharding(
            _args(), [model], changed, saved, sharding, reduce_max=reduce_max
        )
    lc.check_loop_optimizer_resharding(_args(), [plain], changed, None, "dp_reshardable")
    assert calls == [0, 0, 1, 1, 1, 1]


# ---------------------------------------------------------------------------------------------
# The loop_resume entry
# ---------------------------------------------------------------------------------------------


def test_loop_resume_state_is_plain_and_complete(session):
    model, store = _store_model(min_numel=4096)
    store.configure_modes(
        StoreModes.parse("core=lw,edges=lw,emb_head=rsg"),
        core_plan=psu.core_plan_of("whole_round", 4),
    )
    optimizer = _chained(_DistOpt(0, [{"torch.float32": [10]}]))
    state = lc.loop_resume_state(_args(expert_model_parallel_size=2), [model], optimizer)
    assert state["plan"] is None  # no applied plan in this process
    assert set(state) == {
        "version",
        "roles",
        "pattern",
        "store",
        "store_modes",
        "buckets",
        "plan",
        "ep",
        "world",
    }
    assert state["version"] == lc.LOOP_RESUME_VERSION == 1
    assert state["roles"] == {"prelude": 2, "core": 4, "coda": 2}
    assert state["pattern"] == DENSE and state["ep"] == 2 and state["world"] == 8
    assert state["store"]["z"] == 1 and state["store"]["min_numel"] == 4096
    assert state["store"]["stored"] == sum(1 for _ in store.tensors())
    assert len(state["store"]["stored_sha"]) == 64
    assert state["store_modes"].startswith("core=lw,prelude=lw,coda=lw,mtp=lw,emb_head=rsg")
    buffer = io.BytesIO()
    torch.save({lc.LOOP_RESUME_KEY: state}, buffer)
    buffer.seek(0)
    assert torch.load(buffer, weights_only=True)[lc.LOOP_RESUME_KEY] == state
    # Without a store: no layout, no bucket record (and no collective).
    plain = lc.loop_resume_state(_args(), [_plain_model()], optimizer)
    assert plain["store"] is None and plain["store_modes"] is None and plain["buckets"] is None
    assert plain["roles"] == {"prelude": 2, "core": 4, "coda": 2}


def test_non_loop_models_are_a_no_op():
    linear = nn.Linear(2, 2)
    optimizer = _chained(_DistOpt(0, [[4]]))
    assert lc.loop_resume_state(_args(), [linear], optimizer) is None
    lc.check_loop_save_format([linear], ckpt_format="torch", local=True)
    lc.check_loop_load_format([linear], ckpt_format="torch", load_optim=True, gpt_compat=True)
    lc.check_loop_optimizer_resharding(
        _args(), [linear], optimizer, None, "dp_reshardable", reduce_max=None
    )
    # The rules read every argument with getattr (the checkpointing tests' mock namespaces).
    assert lc.loop_checkpoint_problems(Namespace()) == []


# ---------------------------------------------------------------------------------------------
# The resume check (message 5)
# ---------------------------------------------------------------------------------------------

MESSAGE_5 = "loop resume: this run differs from the checkpoint in "


def _run_args(**overrides):
    """A loop run's arguments: every ``loop_*`` parser default, a Huginn RSBF loop and the
    sizes the resume check reads."""
    values = dict(lc._parser_defaults())
    values.update(
        loop_architecture="huginn",
        loop_prelude_sublayers=2,
        loop_coda_sublayers=2,
        loop_schedule="rsbf",
        loop_exit_criterion="sampled",
        loop_depth_probs=[0.25, 0.25, 0.25, 0.25],
        loop_depth_seed=1234,
        hybrid_layer_pattern=DENSE,
        world_size=8,
        expert_model_parallel_size=1,
        finetune=False,
    )
    values.update(overrides)
    return Namespace(**values)


def _saved(model, args, **resume_overrides):
    """A checkpoint's common state: the saved ``args`` and the ``loop_resume`` entry."""
    entry = lc.loop_resume_state(args, [model], None)
    entry.update(resume_overrides)
    return {"args": Namespace(**vars(args)), lc.LOOP_RESUME_KEY: entry}


def _check(args, model, saved, **kwargs):
    kwargs.setdefault("restore_training_state", True)
    return lc.check_loop_resume_state(args, [model], saved, **kwargs)


def test_resume_check_equal_runs_pass_silently(session, capsys):
    model = _plain_model()
    args = _run_args()
    saved = _saved(model, args)
    capsys.readouterr()  # building the parser (for its defaults) may print
    assert _check(args, model, saved) == []
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "name,value",
    [
        ("loop_iterations", 2),
        ("loop_depth_seed", 7),
        ("loop_depth_probs", [0.5, 0.5, 0.0, 0.0]),
        ("loop_exit_criterion", "fixed"),
        ("loop_min_rounds", 2),
        ("loop_rsbf_waves", 2),
        ("loop_sample_depth", True),
        ("loop_rsbf_rebalance", False),
    ],
)
def test_resume_check_refuses_numerics_relevant_changes(session, name, value):
    model = _plain_model()
    saved = _saved(model, _run_args())
    args = _run_args(**{name: value})
    old = getattr(saved["args"], name)
    message = (
        re.escape(f"{MESSAGE_5}{name}: {old!r} -> {value!r}") + r".*--loop-resume-changes allow"
    )
    with pytest.raises(RuntimeError, match=message):
        _check(args, model, saved)
    # --loop-resume-changes allow: resumed, the change logged.
    notes = _check(_run_args(**{name: value}, loop_resume_changes="allow"), model, saved)
    assert any(f"{name}: {old!r} -> {value!r}" in note and "allowed" in note for note in notes)


def test_resume_check_item_forms(session):
    model = _plain_model()
    whole = _run_args()
    saved = _saved(model, whole)
    # Whole round <-> nested: bitwise interchangeable (the block stride does not apply).
    nested = _run_args(loop_rsbf_item_form="nested", loop_rsbf_block_stride=2)
    notes = _check(nested, model, saved)
    assert notes == [
        "loop resume: loop_rsbf_item_form 'whole_round' -> 'nested' (bitwise interchangeable)"
    ]
    notes = _check(whole, model, _saved(model, nested))
    assert len(notes) == 1 and "'nested' -> 'whole_round'" in notes[0]
    # Block items, their stride, waves: numerics-relevant.
    block = _run_args(loop_rsbf_item_form="block", loop_rsbf_block_stride=2)
    with pytest.raises(RuntimeError, match="loop_rsbf_item_form: 'whole_round' -> 'block'"):
        _check(block, model, saved)
    block4 = _run_args(loop_rsbf_item_form="block", loop_rsbf_block_stride=4)
    with pytest.raises(RuntimeError, match=re.escape("loop_rsbf_block_stride: 2 -> 4")):
        _check(block4, model, _saved(model, block))
    nested4 = _run_args(loop_rsbf_item_form="nested", loop_rsbf_block_stride=4)
    with pytest.raises(RuntimeError, match="loop_rsbf_item_form: 'block' -> 'nested'"):
        _check(nested4, model, _saved(model, block))


def test_resume_check_layout_neutral_arguments(session):
    model = _plain_model()
    saved = _saved(model, _run_args())
    # The store layout: a note each.
    store = dict(
        loop_param_store_shard_size=4,
        loop_param_store_modes="all=rsg",
        loop_param_store_min_numel=65536,
        loop_param_store_gather_engine="copy",
    )
    notes = _check(_run_args(**store), model, saved)
    assert len(notes) == 4 and all("layout-neutral" in note for note in notes)
    assert any("loop_param_store_shard_size 1 -> 4" in note for note in notes)
    # Silent: planner inputs, the planner mode, paths / reports / debug settings, this switch.
    silent = dict(
        loop_planner="report",
        loop_planner_hbm_cap_gb=60.0,
        loop_planner_waves=[1, 2],
        loop_planner_depths="histogram",
        loop_planner_histogram_path="h.jsonl",
        loop_exit_histogram_path="x.jsonl",
        loop_planner_report_path="plan.json",
        loop_param_store_debug="checks",
        loop_param_store_copy_engine_timeout_ms=10,
        loop_resume_changes="allow",
    )
    assert _check(_run_args(**silent), model, saved) == []
    for name in silent:
        assert lc.is_layout_neutral(name), name
    for name in ("loop_iterations", "loop_rsbf_item_form", "loop_depth_seed", "loop_untie_core"):
        assert not lc.is_layout_neutral(name), name
    # No suffix rule: a future numerics-relevant ``loop_*_interval`` is refused by default.
    assert not lc.is_layout_neutral("loop_carrier_schedule_interval")


# How a change of each ``loop_*`` argument is treated on resume: "refused" (message 5 unless
# --loop-resume-changes allow; the item-form swap whole_round <-> nested is a note), "note" (a
# store-layout note, the run continues) or "silent". A new argument fails this test until it
# is added here, so every loop argument gets an explicit resume decision.
LOOP_ARGUMENT_RESUME_CLASSES = {
    "loop_architecture": "refused",
    "loop_bptt_iterations": "refused",
    "loop_coda_sublayers": "refused",
    "loop_core_layers": "refused",
    "loop_depth_probs": "refused",
    "loop_depth_seed": "refused",
    "loop_exit_criterion": "refused",
    "loop_exit_histogram_path": "silent",
    "loop_iterations": "refused",
    "loop_min_rounds": "refused",
    "loop_param_store_binding": "note",
    "loop_param_store_copy_engine_timeout_ms": "silent",
    "loop_param_store_debug": "silent",
    "loop_param_store_gather_engine": "note",
    "loop_param_store_min_numel": "note",
    "loop_param_store_modes": "note",
    "loop_param_store_reduce_engine": "note",
    "loop_param_store_separate_rs_group": "note",
    "loop_param_store_shard_size": "note",
    "loop_planner": "silent",
    "loop_planner_act_bytes_per_token": "silent",
    "loop_planner_c2c_gbps": "silent",
    "loop_planner_c2c_margin": "silent",
    "loop_planner_depths": "silent",
    "loop_planner_ep_reserve_gb": "silent",
    "loop_planner_forms": "silent",
    "loop_planner_grace_gb": "silent",
    "loop_planner_grace_margin_gb": "silent",
    "loop_planner_hardware": "silent",
    "loop_planner_hbm_cap_gb": "silent",
    "loop_planner_histogram_path": "silent",
    "loop_planner_optimizer_offload": "silent",
    "loop_planner_report_path": "silent",
    "loop_planner_small_core_layers": "silent",
    "loop_planner_strides": "silent",
    "loop_planner_waves": "silent",
    "loop_prelude_layers": "refused",
    "loop_prelude_sublayers": "refused",
    "loop_recompute": "refused",
    "loop_resume_changes": "silent",
    "loop_router_balance_coeff": "refused",
    "loop_router_z_loss_coeff": "refused",
    "loop_rsbf_block_stride": "refused",
    "loop_rsbf_coda_recompute_mla_up_proj": "refused",
    "loop_rsbf_fill": "refused",
    "loop_rsbf_item_form": "refused",
    "loop_rsbf_move_transport": "refused",
    "loop_rsbf_over_budget": "refused",
    "loop_rsbf_rebalance": "refused",
    "loop_rsbf_tail_item_moves": "refused",
    "loop_rsbf_waves": "refused",
    "loop_sample_depth": "refused",
    "loop_schedule": "refused",
    "loop_token_execution": "refused",
    "loop_untie_core": "refused",
}


def _resume_class(name):
    """The resume treatment ``_loop_argument_changes`` gives a changed argument."""
    if name.startswith(lc._STORE_LAYOUT_PREFIX):
        return "silent" if name in lc._SILENT_STORE_ARGS else "note"
    return "silent" if lc.is_layout_neutral(name) else "refused"


def test_every_loop_argument_has_a_resume_decision():
    names = set(lc._parser_defaults())
    missing = sorted(names - set(LOOP_ARGUMENT_RESUME_CLASSES))
    stale = sorted(set(LOOP_ARGUMENT_RESUME_CLASSES) - names)
    assert not missing, (
        f"new loop arguments {missing}: decide whether a change on resume is refused (the "
        "default), a store-layout note, or silent (LAYOUT_NEUTRAL_LOOP_ARGS in "
        "megatron/training/loop_checkpoint.py), then add them to LOOP_ARGUMENT_RESUME_CLASSES"
    )
    assert not stale, f"removed loop arguments still listed: {stale}"
    actual = {name: _resume_class(name) for name in sorted(names)}
    assert actual == LOOP_ARGUMENT_RESUME_CLASSES


def test_resume_check_arguments_the_checkpoint_predates_or_this_run_lacks(session):
    model = _plain_model()
    saved = _saved(model, _run_args())
    del saved["args"].loop_min_rounds  # added after the checkpoint was written
    assert _check(_run_args(), model, saved) == []  # this run at the default: silent
    with pytest.raises(RuntimeError, match=re.escape("loop_min_rounds: 1 -> 3")):
        _check(_run_args(loop_min_rounds=3), model, saved)
    saved = _saved(model, _run_args())
    saved["args"].loop_retired_flag = 5  # an argument this run no longer has
    notes = _check(_run_args(), model, saved)
    assert notes == [
        "loop resume: the checkpoint's argument loop_retired_flag=5 is not one of this run's "
        "(ignored)"
    ]


def test_resume_check_roles_pattern_world_and_entry(session):
    model = _plain_model()
    args = _run_args()
    with pytest.raises(RuntimeError, match=re.escape("loop roles: {'prelude': 2, 'core': 2,")):
        _check(args, model, _saved(model, args, roles={"prelude": 2, "core": 2, "coda": 4}))
    with pytest.raises(RuntimeError, match=re.escape("hybrid_layer_pattern: '*-*-' -> '*-*-*-*-'")):
        _check(args, model, _saved(model, args, pattern="*-*-"))
    notes = _check(args, model, _saved(model, args, world=4, ep=2))
    assert notes == [
        "loop resume: world 4 -> 8, EP 2 -> 1 (the data-parallel reduction changes: not bitwise)"
    ]
    # A checkpoint without the entry: a note; its saved loop arguments are still compared.
    saved = _saved(model, args)
    del saved[lc.LOOP_RESUME_KEY]
    notes = _check(args, model, saved)
    assert len(notes) == 1 and "has no loop_resume entry" in notes[0]
    with pytest.raises(RuntimeError, match="loop_iterations: 4 -> 2"):
        _check(_run_args(loop_iterations=2), model, saved)
    # No saved args and no entry (an old or foreign checkpoint): the note only.
    assert len(_check(args, model, {})) == 1


def test_resume_check_store_layout_notes(session):
    model, store = _store_model(min_numel=4096)
    args = _run_args()
    saved = _saved(model, args)
    assert _check(args, model, saved) == []
    plain = _plain_model()
    notes = _check(args, plain, saved)
    assert (
        len(notes) == 1 and "parameter store z 1, stored" in notes[0] and "-> no store" in notes[0]
    )
    # A z change seen both in the arguments and in the layout: one note (the layout's), not two.
    layout = dict(store.checkpoint_layout(), z=2)
    resaved = _saved(model, _run_args(loop_param_store_shard_size=2), store=layout)
    notes = _check(args, model, resaved)
    assert len(notes) == 1 and notes[0].startswith("loop resume: parameter store z 2, stored")
    # Without a layout change the argument note stays (the checkpoint's z equals this run's).
    notes = _check(args, model, _saved(model, _run_args(loop_param_store_shard_size=2)))
    assert len(notes) == 1 and notes[0].startswith("loop resume: loop_param_store_shard_size 2")
    store.configure_modes(StoreModes.parse("all=rsg"), core_plan=psu.core_plan_of("whole_round", 4))
    moved = _saved(model, args, store_modes="core=r,prelude=r,coda=r,mtp=r,emb_head=r")
    notes = _check(args, model, moved)
    assert len(notes) == 1 and notes[0].startswith("loop resume: store modes core=r")


def test_resume_check_is_silent_where_it_does_not_apply(session, capsys):
    model = _plain_model()
    changed = _run_args(loop_iterations=2)
    saved = _saved(model, _run_args())
    capsys.readouterr()
    assert _check(changed, model, saved, restore_training_state=False) == []
    assert _check(_run_args(loop_iterations=2, finetune=True), model, saved) == []
    linear = nn.Linear(2, 2)
    assert lc.check_loop_resume_state(changed, [linear], saved, restore_training_state=True) == []
    # Mock namespaces (getattr everywhere): a non-loop model reads nothing.
    assert (
        lc.check_loop_resume_state(Namespace(), [linear], None, restore_training_state=True) == []
    )
    assert capsys.readouterr().out == ""


def test_resume_changes_flag(monkeypatch):
    from megatron.training.arguments import parse_args

    monkeypatch.setattr(sys, "argv", ["test"])
    assert parse_args().loop_resume_changes == "refuse"
    monkeypatch.setattr(sys, "argv", ["test", "--loop-resume-changes", "allow"])
    assert parse_args().loop_resume_changes == "allow"


# ---------------------------------------------------------------------------------------------
# The RSBF plan on resume
# ---------------------------------------------------------------------------------------------

PLAN_PATTERN = "*-*-*-*-*-*-"  # core 4 layers
PLAN_RSBF = dict(
    loop_schedule="rsbf",
    calculate_per_token_loss=True,
    loop_iterations=4,
    loop_exit_criterion="sampled",
    loop_depth_probs=[0.25, 0.25, 0.25, 0.25],
)


def _plan_model():
    return build_local_loop_model(PLAN_PATTERN, prelude=2, coda=2, **PLAN_RSBF)


def _plan_args(**overrides):
    """An 8-GPU RSBF run of the planner model (8 samples per EU, so waves 1 / 2 / 4 / 8), the
    h100 preset (no Grace tier: the Adam-to-Grace rows are not applicable), form policy d28 (the
    whole round on this 4-layer core)."""
    values = dict(lc._parser_defaults())
    values.update(
        loop_architecture="huginn",
        loop_prelude_sublayers=2,
        loop_coda_sublayers=2,
        loop_planner="apply",
        loop_planner_hardware="h100",
        hybrid_layer_pattern=PLAN_PATTERN,
        seq_length=4096,
        world_size=8,
        data_parallel_size=8,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        micro_batch_size=1,
        global_batch_size=64,
        optimizer_cpu_offload=False,
        optimizer_offload_fraction=1.0,
        use_precision_aware_optimizer=False,
        optimizer="adam",
        use_distributed_optimizer=True,
        accumulate_allreduce_grads_in_fp32=True,
        load=None,
        finetune=False,
        **PLAN_RSBF,
    )
    values.update(overrides)
    return Namespace(**values)


def _feasible_rows(plan):
    return [r for r in plan.table if r.fits and r.applicable and not r.refusals]


def _choice_of(row, plan):
    return planning._option_choice(row.option, plan.core_layers)


def _probe_plan():
    """A fresh plan (report mode) and another feasible choice of its menu."""
    plan = planning.LoopPlannerSession().on_phase_start(
        _plan_args(loop_planner="report"), [_plan_model()], apply=False
    )
    fresh = planning.loop_plan_choice(plan)
    other = next(r for r in _feasible_rows(plan) if _choice_of(r, plan) != fresh)
    return plan, _choice_of(other, plan)


def test_plan_record_is_plain_and_saved_when_applied(session):
    model = _plan_model()
    args = _plan_args()
    plan = planning.plan_loop_rsbf(args, [model], apply=True)
    record = planning.loop_plan_record(plan)
    assert record["choice"] == planning.loop_plan_choice(plan)
    assert record["choice"]["item_form"] == plan.item_form and record["digest"] == plan.digest
    assert record["depth"] == {
        "kind": "pmf",
        "params": [0.25, 0.25, 0.25, 0.25],
        "max_rounds": 4,
        "name": plan.depth.name,
        "source": "config",
    }
    assert record["hardware"] == "h100" and record["phase"]["gpus"] == 8
    buffer = io.BytesIO()
    torch.save(record, buffer)
    buffer.seek(0)
    assert torch.load(buffer, weights_only=True) == record
    # In the loop_resume entry once a plan is applied (report mode records none).
    assert lc.loop_resume_state(args, [model], None)["plan"] == record
    planning.reset_loop_planner_session()
    planning.plan_loop_rsbf(_plan_args(loop_planner="report"), [_plan_model()], apply=False)
    assert lc.loop_resume_state(args, [model], None)["plan"] is None


def test_check_resume_plan(session):
    plan, other = _probe_plan()
    record = planning.loop_plan_record(plan)
    args = _plan_args()
    assert planning.check_resume_plan(args, record, plan) == (None, [])
    assert planning.check_resume_plan(args, None, plan) == (None, [])
    assert planning.check_resume_plan(args, record, None) == (None, [])
    # A digest-only change (a memory-model input that changes no choice): accepted.
    assert planning.check_resume_plan(args, dict(record, digest="0" * 64), plan) == (None, [])
    # Another choice: the changed fields named, with the pin reason.
    moved = dict(record, choice=other)
    problem, notes = planning.check_resume_plan(args, moved, plan, pin_reason="infeasible: x")
    assert problem.startswith("the RSBF plan: ") and notes == []
    assert problem.endswith("; the saved plan could not be pinned: infeasible: x")
    changed = [key for key in other if other[key] != record["choice"][key]]
    assert changed and all(f"{key}: " in problem for key in changed)
    # Whole round <-> nested: a note.
    assert plan.item_form == "whole_round"
    nested = dict(record["choice"], item_form="nested", block_stride=2)
    problem, notes = planning.check_resume_plan(args, dict(record, choice=nested), plan)
    assert problem is None and notes == [
        "loop resume: RSBF item form nested -> whole_round (bitwise interchangeable)"
    ]
    # The depth distribution: compared when both runs plan from the exit criterion.
    depth = dict(record["depth"], params=[0.4, 0.2, 0.2, 0.2])
    problem, _ = planning.check_resume_plan(args, dict(record, depth=depth), plan)
    assert problem.startswith("the RSBF plan: depth distribution: ")
    histogram = _plan_args(loop_planner_depths="histogram")
    assert planning.check_resume_plan(histogram, dict(record, depth=depth), plan) == (None, [])
    from_histogram = dict(record, depth=dict(depth, source="histogram"))
    assert planning.check_resume_plan(args, from_histogram, plan) == (None, [])


def test_pin_saved_plan(session):
    plan, other = _probe_plan()
    same, reason = planning.pin_saved_plan(plan, planning.loop_plan_record(plan))
    assert same is plan and reason is None
    pinned, reason = planning.pin_saved_plan(plan, {"choice": other})
    assert reason is None and planning.loop_plan_choice(pinned) == other
    assert (pinned.item_form, pinned.block_stride, pinned.waves) == (
        other["item_form"],
        other["block_stride"],
        other["waves"],
    )
    assert pinned.digest != plan.digest and pinned.table == plan.table
    assert pinned.notes[-1].startswith("pinned to the checkpoint's choice")
    # Deterministic on every rank: the same inputs pin to the same digest.
    assert planning.pin_saved_plan(plan, {"choice": other})[0].digest == pinned.digest
    _, reason = planning.pin_saved_plan(plan, {"choice": dict(other, waves=3)})
    assert reason == "not in this run's menu"
    blocked = next(r for r in plan.table if "A" in r.option.moves)  # no Grace tier on h100
    _, reason = planning.pin_saved_plan(plan, {"choice": _choice_of(blocked, plan)})
    assert reason.startswith("infeasible: ") and "Adam to Grace" in reason
    _, reason = planning.pin_saved_plan(plan, {})
    assert reason == "the checkpoint records no choice"


@ONE_RANK_ONLY
def test_read_saved_plan_from_a_torch_dist_common_state(session, tmp_path):
    from megatron.core import dist_checkpointing

    plan, other = _probe_plan()
    record = dict(planning.loop_plan_record(plan), choice=other)
    ckpt = tmp_path / "ckpt"
    iteration_dir = ckpt / "iter_0000003"
    iteration_dir.mkdir(parents=True)
    dist_checkpointing.save({lc.LOOP_RESUME_KEY: {"plan": record}}, str(iteration_dir))
    args = _plan_args(
        load=str(ckpt),
        ckpt_step=3,
        non_persistent_global_ckpt_dir=None,
        non_persistent_ckpt_type=None,
        exit_on_missing_checkpoint=False,
    )
    assert planning.read_saved_plan(args) == record
    for overrides in (dict(loop_planner="report"), dict(finetune=True), dict(load=None)):
        assert planning.read_saved_plan(_plan_args(**dict(vars(args), **overrides))) is None
    # A checkpoint without a plan record.
    bare = tmp_path / "bare" / "iter_0000003"
    bare.mkdir(parents=True)
    dist_checkpointing.save({lc.LOOP_RESUME_KEY: {"plan": None}}, str(bare))
    assert planning.read_saved_plan(_plan_args(**dict(vars(args), load=str(bare.parent)))) is None
    # Local non-persistent checkpoints have no rank-0 read (no LocalCheckpointManager yet):
    # no pin, no crash on the missing checkpointing context.
    local = _plan_args(**dict(vars(args), non_persistent_ckpt_type="local"))
    assert planning.read_saved_plan(local) is None


def _save_checkpoint_args(ckpt_dir):
    """``save_checkpoint`` arguments on CPU (one-rank Gloo): torch_dist, no optimizer, no RNG
    (the CUDA streams are CUDA-only); the rest as ``init_checkpointing_mock_args``."""
    from megatron.training.arguments import parse_args

    args = parse_args(ignore_unknown_args=True)
    values = dict(
        save=str(ckpt_dir),
        load=str(ckpt_dir),
        non_persistent_global_ckpt_dir=None,
        non_persistent_ckpt_type=None,
        pretrained_checkpoint=None,
        ckpt_format="torch_dist",
        ckpt_fully_parallel_save=False,
        ckpt_fully_parallel_load=False,
        ckpt_assume_constant_structure=False,
        async_save=False,
        use_dist_ckpt=True,
        no_save_optim=True,
        no_save_rng=True,
        use_distributed_optimizer=False,
        save_tokenizer_assets=False,
        log_progress=False,
        consumed_train_samples=0,
        skipped_train_samples=0,
        consumed_valid_samples=0,
        ckpt_step=None,
        world_size=1,
        hybrid_layer_pattern=PLAN_PATTERN,
    )
    for name, value in values.items():
        setattr(args, name, value)
    return args


@ONE_RANK_ONLY
def test_applied_plan_survives_save_checkpoint_and_is_pinned(
    session, monkeypatch, tmp_path, run_config
):
    """End to end on CPU: a run whose applied plan is not the planner's fresh choice saves
    through ``save_checkpoint`` (the ``loop_resume`` entry of ``common.pt``); a fresh planner
    session with ``--load`` reads it back with the real ``read_saved_plan`` and pins it. The
    load / continuity half on DSV4-TINY is the toolkit chain C2a (``--loop-planner apply``)."""
    from unittest import mock

    from megatron.training.checkpointing import save_checkpoint
    from megatron.training.training import preprocess_common_state_dict

    plan, other = _probe_plan()
    assert planning.loop_plan_choice(plan) != other
    # The run being saved: its applied plan pinned to ``other`` (an earlier resume's choice).
    with monkeypatch.context() as patch:
        patch.setattr(planning, "read_saved_plan", lambda args: {"choice": other})
        model = _plan_model()
        applied = planning.plan_loop_rsbf(_plan_args(load="/earlier"), [model], apply=True)
    assert planning.loop_plan_choice(applied) == other
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    save_args = _save_checkpoint_args(ckpt)
    # get_rng_state reads the CUDA streams; --no-save-rng drops its result anyway.
    with (
        mock.patch("megatron.training.checkpointing.get_args", new=lambda: save_args),
        mock.patch("megatron.training.checkpointing.get_rng_state", return_value=None),
    ):
        save_checkpoint(
            5, [model], None, None, 0, preprocess_common_state_dict_fn=preprocess_common_state_dict
        )
    tracker = ckpt / "latest_checkpointed_iteration.txt"
    assert tracker.read_text().strip() == "5"
    # A fresh process: a new session reads the checkpoint at --ckpt-step 5 (on CPU without the
    # tracker file: read_metadata all-reduces the tracker's iteration on a CUDA tensor).
    tracker.unlink()
    planning.reset_loop_planner_session()
    resume_args = _plan_args(
        load=str(ckpt),
        ckpt_step=5,
        non_persistent_global_ckpt_dir=None,
        non_persistent_ckpt_type=None,
        exit_on_missing_checkpoint=False,
        ckpt_format="torch_dist",
    )
    record = planning.read_saved_plan(resume_args)
    assert record == planning.loop_plan_record(applied, depth_source="config")
    resumed_model = _plan_model()
    pinned = planning.plan_loop_rsbf(resume_args, [resumed_model], apply=True)
    assert planning.loop_plan_choice(pinned) == other and pinned.digest == applied.digest
    assert resumed_model.config.loop_rsbf_item_form == other["item_form"]
    session_ = planning.get_loop_planner_session()
    assert session_.pin_reason is None and session_.saved_plan == record


def test_session_pins_the_saved_choice_on_resume(session, monkeypatch, capsys):
    from megatron.training.utils import is_rank0

    plan, other = _probe_plan()
    record = dict(planning.loop_plan_record(plan), choice=other)
    monkeypatch.setattr(planning, "read_saved_plan", lambda args: record)
    model = _plan_model()
    args = _plan_args(load="/ckpt")
    pinned = planning.plan_loop_rsbf(args, [model], apply=True)
    assert planning.loop_plan_choice(pinned) == other
    assert model.config.loop_rsbf_item_form == other["item_form"]
    assert args.loop_rsbf_waves == other["waves"] == model.config.loop_rsbf_waves
    session_ = planning.get_loop_planner_session()
    assert session_.pin_reason is None and session_.saved_plan == record
    if is_rank0():
        assert "loop planner: resumed with the checkpoint's plan" in capsys.readouterr().out
    # The resume check then passes (saved args = this run's after the pin).
    saved = {"args": Namespace(**vars(args)), lc.LOOP_RESUME_KEY: {"plan": record}}
    assert lc.check_loop_resume_state(args, [model], saved, restore_training_state=True) == []


def test_session_plans_afresh_when_the_pin_fails(session, monkeypatch):
    plan, _ = _probe_plan()
    record = planning.loop_plan_record(plan)
    stale = dict(record, choice=dict(record["choice"], waves=3))
    monkeypatch.setattr(planning, "read_saved_plan", lambda args: stale)
    model = _plan_model()
    args = _plan_args(load="/ckpt")
    from megatron.training.utils import is_rank0

    # warn_rank_0: only rank 0 warns under torchrun.
    expect_warning = (
        pytest.warns(UserWarning, match=r"could not be pinned \(not in this run's menu\)")
        if is_rank0()
        else contextlib.nullcontext()
    )
    with expect_warning:
        fresh = planning.plan_loop_rsbf(args, [model], apply=True)
    assert planning.loop_plan_choice(fresh) == record["choice"]
    assert planning.get_loop_planner_session().pin_reason == "not in this run's menu"
    saved_args = Namespace(**dict(vars(args), loop_rsbf_waves=3))
    saved = {"args": saved_args, lc.LOOP_RESUME_KEY: {"plan": stale}}
    waves = record["choice"]["waves"]
    message = (
        re.escape(
            f"{MESSAGE_5}loop_rsbf_waves: 3 -> {waves}; the RSBF plan: waves: 3 -> {waves}; the "
            "saved plan could not be pinned: not in this run's menu"
        )
        + ".*--loop-planner report with the saved choice set by hand"
    )
    with pytest.raises(RuntimeError, match=message):
        lc.check_loop_resume_state(args, [model], saved, restore_training_state=True)
    args.loop_resume_changes = "allow"
    notes = lc.check_loop_resume_state(args, [model], saved, restore_training_state=True)
    assert len(notes) == 2 and all("allowed by --loop-resume-changes allow" in n for n in notes)


# ---------------------------------------------------------------------------------------------
# E18: the data loader does not draw from the CPU default generator
# ---------------------------------------------------------------------------------------------


class _Rows(torch.utils.data.Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return {"tokens": torch.full((4,), index)}


def test_data_loader_leaves_the_default_generator_unchanged(monkeypatch, run_config):
    """The eager stack's ``--loop-sample-depth`` draws use the CPU default generator, which
    ``load_checkpoint`` restores before the data iterators are built; the pretraining loader
    owns its generator, so creating its iterator draws nothing from the default one."""
    from megatron.training.datasets import data_samplers

    # The loader reads the batch sizes and the signal handler from the run config.
    run_config.train.micro_batch_size = 1
    run_config.train.global_batch_size = 8
    run_config.train.exit_signal_handler = False
    run_config.validation.full_validation = False
    loader_args = Namespace(
        dataloader_type="single",
        micro_batch_size=1,
        full_validation=False,
        num_workers=0,
        exit_signal_handler=False,
        use_varlen_dataset=False,
        varlen_sbhd_validation=False,
        dynamic_context_parallel=False,
        sequence_packing_scheduler=None,
        global_batch_size=8,
    )
    monkeypatch.setattr(data_samplers, "get_args", lambda: loader_args)
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    loader = data_samplers.build_pretraining_data_loader(_Rows(), consumed_samples=3)
    batch = next(iter(loader))
    # The sampler resumed at the consumed samples (this data-parallel rank's micro batch).
    assert int(batch["tokens"][0, 0]) == 3 + parallel_state.get_data_parallel_rank()
    assert torch.equal(torch.get_rng_state(), before)
    # Control: a loader without its own generator draws its base seed from the default one.
    next(iter(torch.utils.data.DataLoader(_Rows(), batch_size=1)))
    assert not torch.equal(torch.get_rng_state(), before)
