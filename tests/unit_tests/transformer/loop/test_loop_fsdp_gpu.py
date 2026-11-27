# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-FSDP training of a DSv4 hybrid loop (GPU; FSDP-arm design test T5).

One training step of a DSv4 loop ``HybridModel`` at test width, through the real path: the
wrapper, the no-pipelining schedule (``get_forward_backward_func``, two microbatches of one row
per rank) and ``finalize_model_grads`` under ``calculate_per_token_loss``. Model: pattern
``'WEWECEHECEHECE/WE'`` (prelude ``WEWECE`` with three hash-routed MoE layers, core ``HECE``
run exactly 4 times, coda ``HECE``, MTP ``WE``), DSv4 W / C / H attention (unfused,
``dsa_kernel_backend='none'``), 8 experts top-2 with the Flash router (sqrtsoftplus, top-k
scaling 1.5, expert bias, sequence aux loss, one shared expert), the sparse indexer loss, mHC
with 4 streams, bf16 with TE fused weight gradients, deterministic kernels, and block + edge
``loop_recompute`` (stride 2: one attention + MoE pair per checkpoint block). Expert
parallelism ``EP = W`` (expert data parallelism 1), or ``LOOP_FSDP_EP=2`` (experts sharded over
expert data parallelism 2, so they use accumulate mode too).

Arms, all from the same weights: Megatron DDP (fp32 gradient reduction; the reference) and
Megatron-FSDP v1 in accumulate mode with fp32 main and communication gradients, ``optim_grads``
(Z2) and ``optim_grads_params`` (Z3); a Z3 arm with prefetch off counts all-gathers; the
mutation is Z3 without accumulate mode (the loop guard bypassed).

* Every parameter's reduced fp32 gradient (Megatron-FSDP's shards all-gathered) matches DDP's:
  cosine >= 0.9999 and relative difference <= 1e-3, or, for a gradient that is zero to fp32
  precision (at most 1e-6 of the whole gradient's norm), an absolute difference at most that.
  Both wrappers run the same bf16 kernels on the same weights; they differ in the reduction
  collective and its order.
* The attached losses (LM, MTP, sequence aux, indexer), every row's LM loss, the last
  microbatch's loop metrics and the MoE metric bins (per layer and per round) equal DDP's.
* The FSDP units are the mHC hybrid wrappers (the modules the stack calls), one per sublayer,
  and every core parameter belongs to one.
* Z3 all-gathers per unit and microbatch, prefetch off: exactly one per forward use, and one or
  two per backward use (the recompute and / or the backward; design Section 2.6: three per
  use, two in round R). The exact counts per unit are printed and dumped.
* Teeth: without accumulate mode every fused-wgrad GEMM weight of the core fails the gradient
  gate (and the shared output layer, used by the main and the MTP head).
"""

import collections
import functools
import json
import os
import re
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.unit_tests.transformer.loop import mbs_invariance_utils as mbs
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    CUDA,
    determinism_env,
    model_parallel_once,
)

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
EP = int(os.environ.get("LOOP_FSDP_EP", str(WORLD)))
pytestmark = pytest.mark.skipif(
    not CUDA or WORLD < 2 or WORLD % EP != 0 or 8 % EP != 0,
    reason="DSv4 / MoE layers need CUDA; Megatron-FSDP needs data parallelism (W >= 2)",
)

PATTERN = "WEWECEHECEHECE/WE"
_RATIOS = {"W": 0, "C": 4, "H": 128, "E": 0}
PRELUDE, CODA = 6, 4
ROUNDS = 4
STRIDE = 2
SEQ = 256
VOCAB = 512
ROWS = 2  # per rank: two microbatches of one row
MIN_COSINE = 0.9999
MAX_RELATIVE = 1e-3
ZERO_FRACTION = 1e-6
LOSS_RELATIVE = 1e-6


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    if not CUDA:
        yield
        return
    with determinism_env(), model_parallel_once(expert_model_parallel_size=EP):
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(False)


def _model(state=None):
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_dsv4_stack_spec
    from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    main, mtp = PATTERN.split("/")
    config = MLATransformerConfig(
        num_layers=len(main),
        hidden_size=512,
        num_attention_heads=8,
        multi_latent_attention=True,
        q_lora_rank=192,
        kv_lora_rank=64,
        qk_head_dim=16,
        qk_pos_emb_head_dim=8,
        v_head_dim=16,
        qk_layernorm=True,
        o_groups=8,
        o_lora_rank=128,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        rope_type="rope",
        rotary_base=10000,
        rotary_percent=1.0,
        experimental_attention_variant="dsv4_hybrid",
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=32,
        dsa_indexer_loss_coeff=1e-2,
        dsa_indexer_use_sparse_loss=True,
        csa_window_size=32,
        csa_compress_ratios=[_RATIOS[symbol] for symbol in main + mtp],
        num_moe_experts=8,
        moe_ffn_hidden_size=256,
        moe_shared_expert_intermediate_size=256,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_router_score_function="sqrtsoftplus",
        moe_router_topk_scaling_factor=1.5,
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=1e-3,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=1e-4,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_n_hash_layers=3,
        actual_vocab_size=VOCAB,
        activation_func_clamp_value=10.0,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=5,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=True,
        loop_architecture="huginn",
        loop_carrier="injection",
        loop_iterations=ROUNDS,
        loop_exit_criterion="fixed",
        loop_prelude_sublayers=PRELUDE,
        loop_coda_sublayers=CODA,
        loop_recompute=True,
        loop_recompute_block_stride=STRIDE,
        loop_recompute_edges=True,
        calculate_per_token_loss=True,
        deterministic_mode=True,
        attention_backend=AttnBackend.unfused,
        expert_model_parallel_size=EP,
    )
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    set_moe_metrics_tracker(MoEMetricsTracker())
    model = HybridModel(
        config=config,
        hybrid_stack_spec=loop_hybrid_dsv4_stack_spec(config),
        vocab_size=VOCAB,
        max_sequence_length=1024,
        hybrid_layer_pattern=PATTERN,
        position_embedding_type="rope",
    ).cuda()
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def _ddp(model):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.process_groups_config import ProcessGroupCollection

    return DistributedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            overlap_grad_reduce=False, use_distributed_optimizer=False, grad_reduce_in_fp32=True
        ),
        model,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )


def _fsdp(model, strategy, accumulate=True):
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel

    return FullyShardedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            use_distributed_optimizer=True,
            data_parallel_sharding_strategy=strategy,
            megatron_fsdp_accumulate_main_grad=accumulate,
            megatron_fsdp_main_grads_dtype=torch.float32,
            megatron_fsdp_grad_comm_dtype=torch.float32,
        ),
        model,
    )


class _GatherCounter:
    """Counts the all-gathers Megatron-FSDP launches, per (microbatch, phase, bucket): a bucket
    gather is launched when the bucket is neither in flight nor ready."""

    def __init__(self, fsdp):
        from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
            BucketStatus,
        )

        pipeline = fsdp.module.all_gather_pipeline
        original = pipeline.async_bucket_gather
        self.counts = collections.Counter()
        self.microbatch = -1
        self.phase = "fwd"

        def counting(bucket_id, bwd):
            status = pipeline.bucket_status[pipeline.get_bucket_key(bucket_id, bwd)]
            if status not in (BucketStatus.COMMUNICATING, BucketStatus.READY_TO_USE):
                self.counts[(self.microbatch, self.phase, bucket_id)] += 1
            return original(bucket_id, bwd)

        pipeline.async_bucket_gather = counting

    def start_microbatch(self):
        self.microbatch += 1
        self.phase = "fwd"

    def start_backward(self):
        self.phase = "bwd"


def _step(wrapped, model, rows, counter=None):
    """One training step over ``rows`` (microbatches of one row); returns the attached loss
    sums, per-row LM losses, the MoE metric bins and the last microbatch's loop metrics."""
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.transformer.moe.moe_logging import (
        MoEMetricsTracker,
        get_moe_metrics_tracker,
        set_moe_metrics_tracker,
    )

    model.config.finalize_model_grads_func = finalize_model_grads
    set_moe_metrics_tracker(MoEMetricsTracker())
    wrapped.zero_grad_buffer()
    recorder = mbs._Recorder()

    def forward_step(data_iterator, model_chunk):
        if counter is not None:
            counter.start_microbatch()
        output, loss_func = mbs._forward_step(recorder, data_iterator, model_chunk)

        def loss_then_backward(*args):
            if counter is not None:
                counter.start_backward()
            return loss_func(*args)

        return output, loss_then_backward

    batches = mbs._microbatches(rows, 1)
    with mbs._recording_attached_losses(recorder):
        get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=iter(batches),
            model=[wrapped],
            num_microbatches=len(batches),
            seq_length=SEQ,
            micro_batch_size=1,
            forward_only=False,
        )
    torch.cuda.synchronize()
    bins = {
        name: (
            entry.values.detach().float().clone(),
            None if entry.round_values is None else entry.round_values.detach().float().clone(),
        )
        for name, entry in get_moe_metrics_tracker().metrics.items()
    }
    loop_metrics = {
        name: value.detach().float().clone() if torch.is_tensor(value) else value
        for name, value in model.decoder.loop_metrics.items()
    }
    return SimpleNamespace(
        losses=dict(recorder.sums), rows=list(recorder.rows), bins=bins, loop_metrics=loop_metrics
    )


def _ddp_grads(model):
    return {
        name: param.main_grad.detach().float().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _fsdp_grads(fsdp):
    """Every parameter's reduced gradient: each group's gradient shards all-gathered over its
    data-parallel group (collective: every rank walks the groups in the same order)."""
    from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import Bucket

    buffer = fsdp.param_and_grad_buffer
    grads = {}
    for group in buffer.parameter_groups:
        gbuf = group.main_grad_buffer
        if gbuf is None or not group.requires_grad:
            continue
        if gbuf.is_data_distributed:
            full = torch.empty(gbuf.bucket_index.size, dtype=gbuf.dtype, device="cuda")
            dist.all_gather_into_tensor(full, gbuf.data, group=gbuf.data_parallel_group)
        else:
            start = gbuf.bucket_index.global_data_index
            full = gbuf.data[start : start + gbuf.bucket_index.size]
        for param in group.params:
            item = gbuf.get_item_from_bucket(Bucket(data=full), gbuf.param_idx[param])
            grads[buffer.param_to_name[param]] = item.view(param.shape).float().clone()
    return grads


_LAYER = re.compile(r"^decoder\.layers\.(\d+)\.")


def _role(name):
    """``(role, sublayer index)`` of a parameter name: prelude / core / coda by the sublayer of
    the main pattern, mtp, or other (embedding, heads, loop adapter and norm, final norm)."""
    match = _LAYER.match(name)
    if match:
        index = int(match.group(1))
        core = len(PATTERN.split("/")[0]) - PRELUDE - CODA
        if index < PRELUDE:
            return "prelude", index
        if index < PRELUDE + core:
            return "core", index
        return "coda", index
    if name.startswith("mtp."):
        return "mtp", None
    return "other", None


def _unit_buckets(fsdp):
    """``{bucket id: (role, sublayer, sharded)}`` of the buckets that belong to FSDP units;
    ``sharded``: the bucket's weights are sharded (re-gathered for every use). A bucket with
    unsharded weights (experts at expert data parallelism 1) is gathered once and kept."""
    buffer = fsdp.param_and_grad_buffer
    units = {}
    for bucket_id, group in enumerate(buffer.parameter_groups):
        if group.fsdp_unit_id is None:
            continue
        roles = {_role(buffer.param_to_name[param]) for param in group.params}
        assert len(roles) == 1, roles  # a unit is one sublayer (or one MTP sublayer)
        units[bucket_id] = (*roles.pop(), group.has_sharded_model_weights())
    return units


def _fused_core_weights(model, expert_dp):
    """The core's GEMM weights whose weight gradient TE accumulates in place (fused wgrad),
    without the experts when expert data parallelism is 1 (their gradients are not sharded)."""
    names = []
    for module_name, module in model.named_modules():
        if not getattr(module, "fuse_wgrad_accumulation", False):
            continue
        for param_name, param in module.named_parameters(recurse=False):
            name = f"{module_name}.{param_name}"
            if _role(name)[0] != "core" or not re.fullmatch(r"weight\d*", param_name):
                continue
            if expert_dp == 1 and not getattr(param, "allreduce", True):
                continue
            names.append(name)
    return names


def _gate(grads, reference):
    report = mbs.compare_gradients(
        grads, reference, min_cosine=MIN_COSINE, max_relative=MAX_RELATIVE
    )
    total = mbs.global_comparison(grads, reference)
    bound = ZERO_FRACTION * total["norm"]
    for entry in report.values():
        entry["zero"] = entry["norm"] <= bound
        if entry["zero"]:
            entry["ok"] = entry["diff"] <= bound
    return report, total


def _rank():
    return dist.get_rank()


def _tag(text):
    return f"T5 W{WORLD} EP{EP} rank{_rank()} {text}"


def _dump(tag, payload):
    directory = os.environ.get("MBS_REPORT_DIR")
    if not directory:
        return
    os.makedirs(directory, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
    with open(os.path.join(directory, f"{name}.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, default=float)


@pytest.fixture(scope="module")
def results():
    """Every arm of the module, run once from the same weights and rows."""
    from megatron.core.transformer.loop import guards

    reference_state = {
        name: value.detach().clone() if torch.is_tensor(value) else value
        for name, value in _model().state_dict().items()
    }
    rows = mbs.make_rows(ROWS, SEQ, VOCAB, seed=4242 + _rank())
    out = {}

    model = _model(reference_state)
    ddp = _ddp(model)
    out["ddp"] = (_step(ddp, model, rows), _ddp_grads(model))
    fused = _fused_core_weights(model, expert_dp=WORLD // EP)
    del ddp, model

    for arm, strategy in (("z2", "optim_grads"), ("z3", "optim_grads_params")):
        model = _model(reference_state)
        fsdp = _fsdp(model, strategy)
        counter = _GatherCounter(fsdp) if arm == "z3" else None
        step = _step(fsdp, model, rows, counter)
        out[arm] = (step, _fsdp_grads(fsdp))
        if arm == "z3":
            out["z3_units"] = _unit_buckets(fsdp)
            out["z3_unit_modules"] = sorted(
                key[len("release module ") : -len(" parameters")]
                for key in fsdp.module.forward_hooks
                if key.startswith("release module ")
            )
            out["z3_prefetch_counts"] = dict(counter.counts)
        fsdp.stop_communication()
        del fsdp, model

    # Gather counts with prefetch off: the units' own gathers only.
    model = _model(reference_state)
    fsdp = _fsdp(model, "optim_grads_params")
    fsdp.module.suggested_AG_prefetch_size = 0
    counter = _GatherCounter(fsdp)
    step = _step(fsdp, model, rows, counter)
    out["z3_no_prefetch"] = (step, _fsdp_grads(fsdp))
    out["z3_counts"] = dict(counter.counts)
    fsdp.stop_communication()
    del fsdp, model

    # The mutation: Z3 without accumulate mode (the loop guard would refuse it).
    original_guard = guards.validate_loop_fsdp_config
    guards.validate_loop_fsdp_config = lambda config, ddp_config: None
    try:
        model = _model(reference_state)
        fsdp = _fsdp(model, "optim_grads_params", accumulate=False)
        step = _step(fsdp, model, rows)
        out["z3_legacy"] = (step, _fsdp_grads(fsdp))
        fsdp.stop_communication()
        del fsdp, model
    finally:
        guards.validate_loop_fsdp_config = original_guard
    out["fused_core_weights"] = fused
    return out


@pytest.mark.parametrize("arm", ["z2", "z3", "z3_no_prefetch"])
def test_fsdp_gradients_match_ddp(results, arm):
    _, reference = results["ddp"]
    _, grads = results[arm]
    assert grads.keys() == reference.keys()
    report, total = _gate(grads, reference)
    tag = _tag(f"{arm} vs DDP")
    print(f"[{tag}] whole gradient {total}", flush=True)
    mbs.print_report(tag, report)
    _dump(tag, dict(per_param=report, total=total))
    failing = [name for name, entry in report.items() if not entry["ok"]]
    assert not failing, mbs.worst(report, failing)


@pytest.mark.parametrize("arm", ["z2", "z3", "z3_no_prefetch", "z3_legacy"])
def test_losses_metrics_and_bins_match_ddp(results, arm):
    """The forward is the same function of the same weights under every wrapper."""
    reference, _ = results["ddp"]
    step, _ = results[arm]
    loss_rel = mbs.relative_loss_differences(step.losses, reference.losses)
    print(f"[{_tag(arm)}] losses {step.losses} rel {loss_rel}", flush=True)
    assert all(reference.losses[key] != 0.0 for key in mbs.LOSS_KEYS), reference.losses
    assert all(value <= LOSS_RELATIVE for value in loss_rel.values()), loss_rel
    assert mbs.max_row_difference(step.rows, reference.rows) <= LOSS_RELATIVE
    assert step.loop_metrics.keys() == reference.loop_metrics.keys()
    for name, value in reference.loop_metrics.items():
        torch.testing.assert_close(step.loop_metrics[name], value, rtol=1e-6, atol=0, msg=name)
    assert step.bins.keys() == reference.bins.keys()
    for name, (values, round_values) in reference.bins.items():
        torch.testing.assert_close(step.bins[name][0], values, rtol=1e-6, atol=1e-12, msg=name)
        if round_values is None:
            assert step.bins[name][1] is None, name
        else:
            torch.testing.assert_close(
                step.bins[name][1], round_values, rtol=1e-6, atol=1e-12, msg=name
            )


def test_every_round_records_moe_statistics(results):
    reference, _ = results["ddp"]
    round_values = reference.bins["seq_load_balancing_loss"][1]
    assert round_values is not None and round_values.shape[0] == ROUNDS
    assert all(round_values[r].abs().sum() > 0 for r in range(ROUNDS)), round_values


def test_units_are_the_mhc_wrappers_and_cover_the_core(results):
    _, grads = results["z3"]
    units = results["z3_units"]
    modules = results["z3_unit_modules"]
    main = PATTERN.split("/")[0]
    # One unit per sublayer of the main pattern (the mHC wrapper, not its inner layer) and the
    # MTP sublayers.
    decoder_units = [name for name in modules if name.startswith("decoder.")]
    assert decoder_units == sorted(f"decoder.layers.{i}" for i in range(len(main))), modules
    assert not any(name.endswith("inner_layer") for name in modules), modules
    assert any(role == "mtp" for role, _, _ in units.values()), units
    covered = {(role, index) for role, index, _ in units.values()}
    for name in grads:
        role, index = _role(name)
        if role in ("prelude", "core", "coda"):
            assert (role, index) in covered, name


def _expected_uses(role):
    return ROUNDS if role == "core" else 1


def test_z3_gathers_per_unit_and_microbatch(results):
    """Prefetch off: one gather per forward use; per backward use one (a gather the recompute
    and the backward share) or two (the recompute's gather released before the backward); the
    MTP sublayers are not checkpointed and gather once in the backward. Design Section 2.6
    predicts three per use with two in round R; the measured counts are printed and dumped."""
    units = results["z3_units"]
    counts = results["z3_counts"]
    table = collections.defaultdict(dict)
    assert any(sharded for _, _, sharded in units.values())
    for microbatch in range(ROWS):
        for bucket_id, (role, index, sharded) in units.items():
            if not sharded:
                continue
            fwd = counts.get((microbatch, "fwd", bucket_id), 0)
            bwd = counts.get((microbatch, "bwd", bucket_id), 0)
            table[f"{role} {index} b{bucket_id}"][microbatch] = (fwd, bwd)
            uses = _expected_uses(role)
            assert fwd == uses, (role, index, microbatch, fwd, bwd)
            if role == "mtp":
                assert bwd == 1, (role, index, microbatch, fwd, bwd)
            else:
                assert uses <= bwd <= 2 * uses, (role, index, microbatch, fwd, bwd)
    print(f"[{_tag('z3 gathers per microbatch (fwd, bwd), prefetch off')}] {dict(table)}")
    prefetch = collections.defaultdict(dict)
    for (microbatch, phase, bucket_id), count in results["z3_prefetch_counts"].items():
        if bucket_id in units:
            role, index, _ = units[bucket_id]
            prefetch[f"{role} {index} b{bucket_id}"][f"{microbatch} {phase}"] = count
    print(f"[{_tag('z3 gathers, prefetch on')}] {dict(prefetch)}")
    _dump(_tag("z3 gathers"), dict(no_prefetch=dict(table), prefetch=dict(prefetch)))


def test_without_accumulate_mode_core_weights_lose_rounds(results):
    """The mutation: every use overwrites the gradient bucket, so the fused-wgrad weights of
    the core keep one round's gradient, and the output layer one head's."""
    _, reference = results["ddp"]
    _, grads = results["z3_legacy"]
    report, total = _gate(grads, reference)
    tag = _tag("z3 without accumulate mode vs DDP")
    print(f"[{tag}] whole gradient {total}", flush=True)
    mbs.print_report(tag, report)
    _dump(tag, dict(per_param=report, total=total))
    fused = results["fused_core_weights"]
    assert fused
    passing = [name for name in fused if report[name]["ok"]]
    assert not passing, mbs.worst(report, passing)
    assert not report["output_layer.weight"]["ok"], report["output_layer.weight"]
