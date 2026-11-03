# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The per-parameter gradient gate of the DSv4 Flash fixed-4-round loop (GPU; FSDP-arm design
test T6, the experiment's equivalence gate G2 "Gradients").

One training step of the parity proxy's model (``dsv4_flash_loop_r4_mini``: pattern
``'WEWECEHECEHECEHECEHECEHECE/WE'``, the prelude's 6 sublayers with the three hash-routed MoE
layers, a 16-sublayer core run exactly 4 times, a 4-sublayer coda, MTP ``WE``; hidden 512 and the
proxy's DSv4 W / C / H attention, unfused (``dsa_kernel_backend='none'``); 16 experts top-6 with
the Flash router and one shared expert; the DSA indexer 64 x 128, top-512, with its sparse loss;
mHC with 4 streams; sequence 4096; vocabulary 129,280) at the proxy's layout (EP 4 on 4 GPUs,
4 microbatches of one row per rank: the proxy's GBS 16), with the same weights and rows in every
arm, through each layout's real wrapper, the no-pipelining schedule (``get_forward_backward_func``)
and ``finalize_model_grads`` under per-token loss:

* ``ref``: the parity reference's layout: DDP with the distributed optimizer (fp32 gradient
  reduce-scatter), no ``loop_recompute``;
* ``f1``: DDP with the distributed optimizer and block + edge ``loop_recompute`` (stride 4);
* ``z2`` / ``z3``: Megatron-FSDP v1, ``optim_grads`` / ``optim_grads_params``, in accumulate mode
  with fp32 main and communication gradients, block + edge ``loop_recompute`` (stride 4).

Precision: bf16 compute and fp32 main gradients (the training precision), deterministic kernels
(``deterministic_mode``, the determinism environment). Both sides of every comparison run the same
bf16 kernels on the same weights and rows; they differ in the fp32 association of the
reductions: DDP reduce-scatters the step's sum once, Megatron-FSDP every microbatch's gradient;
the rounds and microbatches add into fp32 buffers in the same order. Expected difference: about
1e-7 to 1e-6 relative per parameter (the Megatron-FSDP training test measured at most 1.7e-6);
the noise-floor arm (``ref`` with its microbatches in reverse order) measures the fp32
reassociation alone. Dropping one of the 4 rounds removes that round's share of every core
weight's gradient, orders of magnitude above the gate.

* Gate (every parameter, core ones listed apart): cosine >= 0.9999 and relative norm difference
  <= 1e-3 against ``ref``; a gradient that is zero to fp32 precision (at most 1e-6 of the whole
  gradient's norm: the mHC stream-symmetry scalars at initialization) is gated absolutely at that
  bound. Every parameter's comparison is printed (rank 0: distribution and worst) and dumped.
* The noise floor (``ref`` vs ``ref`` in reverse microbatch order) is reported and passes the gate.
* The attached losses (LM, MTP, sequence aux, indexer) and every row's LM loss equal ``ref``'s
  within 1e-6 relative (the forward is the same function of the same weights in every arm), and
  every round 1..4 records MoE statistics.
* Teeth, both must fail: (a) ``loop_bptt_iterations = 3`` (round 1 under ``no_grad``) on ``f1``
  and ``z3``: every resolved core parameter fails; (b) the legacy overwrite (Megatron-FSDP
  without accumulate mode, the loop guard bypassed) on ``z2`` and ``z3``: every core weight that
  TE's fused weight gradient writes fails (the others are summed over the uses by autograd before
  Megatron-FSDP sees them, so the overwrite cannot touch them; their count is reported).
* The model-only load of the Megatron-FSDP parity runs (training's
  ``_load_loop_model_before_megatron_fsdp_wrap``): a torch_dist checkpoint saved from ``ref``'s
  model loads, under ``--dist-ckpt-strictness raise_unexpected``, into a model with other weights
  before the Megatron-FSDP wrap, every parameter and persistent buffer bitwise; the wrap then
  holds those weights (bf16 model weights) and their fp32 copies (main weights), bitwise.

Run with ``LOOP_GATE_EP`` (default W) to change the expert-parallel size.
"""

import collections
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
EP = int(os.environ.get("LOOP_GATE_EP", str(WORLD)))
pytestmark = pytest.mark.skipif(
    not CUDA or WORLD < 2 or WORLD % EP != 0 or 16 % EP != 0,
    reason="DSv4 / MoE layers need CUDA; the layouts need data parallelism (W >= 2)",
)

PATTERN = "WEWECEHECEHECEHECEHECEHECE/WE"
_RATIOS = {"W": 0, "C": 4, "H": 128, "E": 0}
PRELUDE, CODA = 6, 4
ROUNDS = 4
STRIDE = 4
SEQ = int(os.environ.get("LOOP_GATE_SEQ", "4096"))
VOCAB = 129280
ROWS = 4  # per rank: 4 microbatches of one row (GBS 16 at W = 4)
MIN_COSINE = 0.9999
MAX_RELATIVE = 1e-3
ZERO_FRACTION = 1e-6
LOSS_RELATIVE = 1e-6

_BLOCKS = dict(loop_recompute=True, loop_recompute_block_stride=STRIDE, loop_recompute_edges=True)


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


def _model(state=None, seed=1234, **overrides):
    """The parity proxy's model (``dsv4_flash_loop_r4_mini``) at test settings: deterministic
    kernels, unfused attention, CPU initialization (the same weights at every EP rank count)."""
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_dsv4_stack_spec
    from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    main, mtp = PATTERN.split("/")
    fields = dict(
        num_layers=len(main),
        hidden_size=512,
        num_attention_heads=8,
        multi_latent_attention=True,
        q_lora_rank=192,
        kv_lora_rank=8,
        qk_head_dim=8,
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
        dsa_indexer_precision="bf16",
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=512,
        dsa_indexer_loss_coeff=1e-2,
        dsa_indexer_use_sparse_loss=True,
        csa_window_size=128,
        csa_compress_rotary_base=40000,
        csa_compress_ratios=[_RATIOS[symbol] for symbol in main + mtp],
        num_moe_experts=16,
        moe_ffn_hidden_size=256,
        moe_shared_expert_intermediate_size=256,
        moe_router_topk=6,
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
        mhc_sinkhorn_iterations=20,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        init_method_std=0.02,
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
        calculate_per_token_loss=True,
        deterministic_mode=True,
        attention_backend=AttnBackend.unfused,
        expert_model_parallel_size=EP,
    )
    fields.update(overrides)
    config = MLATransformerConfig(**fields)
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    set_moe_metrics_tracker(MoEMetricsTracker())
    model = HybridModel(
        config=config,
        hybrid_stack_spec=loop_hybrid_dsv4_stack_spec(config),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        hybrid_layer_pattern=PATTERN,
        position_embedding_type="rope",
    ).cuda()
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def _ddp(model):
    """The reference's wrapper: Megatron DDP with the distributed optimizer's buffers (fp32
    gradient reduce-scatter; the loop refuses overlapped reduction)."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.process_groups_config import ProcessGroupCollection

    return DistributedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            overlap_grad_reduce=False, use_distributed_optimizer=True, grad_reduce_in_fp32=True
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


def _step(wrapped, model, rows):
    """One training step over ``rows`` (microbatches of one row, in row order); returns the
    attached loss sums, the per-row LM losses and the MoE metric bins."""
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
    batches = mbs._microbatches(rows, 1)
    with mbs._recording_attached_losses(recorder):
        get_forward_backward_func()(
            forward_step_func=lambda iterator, chunk: mbs._forward_step(recorder, iterator, chunk),
            data_iterator=iter(batches),
            model=[wrapped],
            num_microbatches=len(batches),
            seq_length=SEQ,
            micro_batch_size=1,
            forward_only=False,
        )
    torch.cuda.synchronize()
    bins = {
        name: None if entry.round_values is None else entry.round_values.detach().float().clone()
        for name, entry in get_moe_metrics_tracker().metrics.items()
    }
    return SimpleNamespace(losses=dict(recorder.sums), rows=list(recorder.rows), bins=bins)


def _ddp_grads(ddp, model):
    """Every parameter's reduced fp32 gradient under DDP with the distributed optimizer: each
    rank holds the reduce-scattered shard of every bucket, so every bucket's shards are first
    all-gathered over the bucket group's data-parallel group (collective)."""
    from megatron.core.distributed.param_and_grad_buffer import shard_buffer

    for group in ddp.bucket_groups + ddp.expert_parallel_bucket_groups:
        size = group.intra_distributed_optimizer_instance_size
        if size == 1:
            continue
        rank = group.intra_distributed_optimizer_instance_rank
        for bucket in group.buckets:
            shard = shard_buffer(bucket.grad_data, size)[rank].clone()
            dist.all_gather_into_tensor(
                bucket.grad_data, shard, group=group.intra_distributed_optimizer_instance_group
            )
    torch.cuda.synchronize()
    return {
        name: param.main_grad.detach().float().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _fsdp_items(fsdp, buffer_name):
    """``{parameter name: values}`` of one Megatron-FSDP buffer kind (``main_grad_buffer``,
    ``model_weight_buffer``, ``main_weight_buffer``): each group's shards all-gathered over its
    data-parallel group (collective: every rank walks the groups in the same order)."""
    from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import Bucket

    buffer = fsdp.param_and_grad_buffer
    out = {}
    for group in buffer.parameter_groups:
        gbuf = getattr(group, buffer_name)
        if gbuf is None or (buffer_name == "main_grad_buffer" and not group.requires_grad):
            continue
        if gbuf.is_data_distributed:
            full = torch.empty(gbuf.bucket_index.size, dtype=gbuf.dtype, device="cuda")
            dist.all_gather_into_tensor(full, gbuf.data, group=gbuf.data_parallel_group)
        else:
            start = gbuf.bucket_index.global_data_index
            full = gbuf.data[start : start + gbuf.bucket_index.size]
        for param in group.params:
            item = gbuf.get_item_from_bucket(Bucket(data=full), gbuf.param_idx[param])
            out[buffer.param_to_name[param]] = item.view(param.shape).clone()
    return out


def _fsdp_grads(fsdp):
    return {name: grad.float() for name, grad in _fsdp_items(fsdp, "main_grad_buffer").items()}


_LAYER = re.compile(r"^decoder\.layers\.(\d+)\.")


def _role(name):
    """prelude / core / coda by the sublayer of the main pattern, ``loop`` for the Huginn adapter
    and loop norm (shared by the rounds), mtp, or other (embedding, heads, final norm)."""
    match = _LAYER.match(name)
    if match:
        index = int(match.group(1))
        core = len(PATTERN.split("/")[0]) - PRELUDE - CODA
        if index < PRELUDE:
            return "prelude"
        if index < PRELUDE + core:
            return "core"
        return "coda"
    if name.startswith("decoder.loop_"):
        return "loop"
    if name.startswith("mtp."):
        return "mtp"
    return "other"


def _is_core(name):
    """The parameters every round uses: the core sublayers' and the Huginn adapter / loop norm."""
    return _role(name) in ("core", "loop")


def _fused_core_weights(model, expert_dp):
    """The core GEMM weights whose weight gradient TE writes into the main gradient (fused
    wgrad), without the experts at expert data parallelism 1 (their gradients are unsharded)."""
    names = []
    for module_name, module in model.named_modules():
        if not getattr(module, "fuse_wgrad_accumulation", False):
            continue
        for param_name, param in module.named_parameters(recurse=False):
            name = f"{module_name}.{param_name}"
            if not _is_core(name) or not re.fullmatch(r"weight\d*", param_name):
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
    for name, entry in report.items():
        entry["zero"] = entry["norm"] <= bound
        entry["core"] = _is_core(name)
        if entry["zero"]:
            entry["ok"] = entry["diff"] <= bound
    return report, total


def _tag(text):
    return f"T6 W{WORLD} EP{EP} rank{dist.get_rank()} {text}"


def _dump(tag, payload):
    directory = os.environ.get("MBS_REPORT_DIR")
    if not directory:
        return
    os.makedirs(directory, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
    with open(os.path.join(directory, f"{name}.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, default=float)


def _summary(tag, report, total):
    """Print (rank 0) the whole-gradient comparison, the distribution and worst parameters, and
    the core / other split; dump every parameter's comparison (every rank)."""
    split = collections.Counter(
        ("core" if entry["core"] else "other", "ok" if entry["ok"] else "FAIL")
        for entry in report.values()
    )
    if dist.get_rank() == 0:
        print(f"[{tag}] whole gradient {total}; (role, verdict) counts {dict(split)}", flush=True)
    mbs.print_report(tag, report)
    _dump(tag, dict(per_param=report, total=total))


def _release(*objects):
    for item in objects:
        stop = getattr(item, "stop_communication", None)
        if stop is not None:
            stop()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _reversed_rows(rows):
    return {name: value.flip(0) for name, value in rows.items()}


@pytest.fixture(scope="module")
def results():
    """Every arm of the module, run once from the same weights and rows."""
    from megatron.core.transformer.loop import guards

    reference_state = {
        name: value.detach().clone() if torch.is_tensor(value) else value
        for name, value in _model().state_dict().items()
    }
    rows = mbs.make_rows(ROWS, SEQ, VOCAB, seed=4242 + dist.get_rank())
    out = {}

    def ddp_arm(name, step_rows=rows, **overrides):
        model = _model(reference_state, **overrides)
        ddp = _ddp(model)
        step = _step(ddp, model, step_rows)
        out[name] = (step, _ddp_grads(ddp, model))
        if name == "ref":
            out["fused_core_weights"] = _fused_core_weights(model, expert_dp=WORLD // EP)
            out["names"] = sorted(out[name][1])
        _release(ddp)

    def fsdp_arm(name, strategy, accumulate=True, **overrides):
        model = _model(reference_state, **dict(_BLOCKS, **overrides))
        fsdp = _fsdp(model, strategy, accumulate=accumulate)
        step = _step(fsdp, model, rows)
        out[name] = (step, _fsdp_grads(fsdp))
        _release(fsdp)

    ddp_arm("ref")
    ddp_arm("ref_reversed", step_rows=_reversed_rows(rows))
    ddp_arm("f1", **_BLOCKS)
    fsdp_arm("z2", "optim_grads")
    fsdp_arm("z3", "optim_grads_params")
    # Teeth (a): round 1 runs under no_grad, so its contribution leaves every core gradient.
    ddp_arm("f1_bptt3", loop_bptt_iterations=3, **_BLOCKS)
    fsdp_arm("z3_bptt3", "optim_grads_params", loop_bptt_iterations=3)
    # Teeth (b): the legacy overwrite policy (the loop guard would refuse it).
    original_guard = guards.validate_loop_fsdp_config
    guards.validate_loop_fsdp_config = lambda config, ddp_config: None
    try:
        fsdp_arm("z2_legacy", "optim_grads", accumulate=False)
        fsdp_arm("z3_legacy", "optim_grads_params", accumulate=False)
    finally:
        guards.validate_loop_fsdp_config = original_guard
    out["reference_state"] = reference_state
    return out


@pytest.mark.parametrize("arm", ["f1", "z2", "z3"])
def test_layout_gradients_match_the_reference(results, arm):
    """G2 "Gradients": every parameter's fp32 gradient of the layout equals the reference
    layout's: cosine >= 0.9999 and relative difference <= 1e-3 (zero gradients absolutely)."""
    _, reference = results["ref"]
    _, grads = results[arm]
    assert grads.keys() == reference.keys()
    report, total = _gate(grads, reference)
    _summary(_tag(f"{arm} vs ref"), report, total)
    assert any(entry["core"] and not entry["zero"] for entry in report.values())
    failing = [name for name, entry in report.items() if not entry["ok"]]
    assert not failing, mbs.worst(report, failing)


def test_noise_floor_reverse_microbatch_order(results):
    """The same layout with its microbatches in reverse order: the fp32 reassociation alone. It
    is reported, and it passes the gate (else the gate could not tell layouts from noise)."""
    _, reference = results["ref"]
    _, grads = results["ref_reversed"]
    report, total = _gate(grads, reference)
    _summary(_tag("noise floor: ref reversed vs ref"), report, total)
    failing = [name for name, entry in report.items() if not entry["ok"]]
    assert not failing, mbs.worst(report, failing)


@pytest.mark.parametrize(
    "arm", ["ref_reversed", "f1", "z2", "z3", "f1_bptt3", "z3_bptt3", "z2_legacy", "z3_legacy"]
)
def test_losses_and_rows_match_the_reference(results, arm):
    """Every arm computes the same forward: the attached loss sums and every row's LM loss."""
    reference, _ = results["ref"]
    step, _ = results[arm]
    rows = step.rows[::-1] if arm == "ref_reversed" else step.rows
    loss_rel = mbs.relative_loss_differences(step.losses, reference.losses)
    if dist.get_rank() == 0:
        print(f"[{_tag(arm)}] losses {step.losses} rel {loss_rel}", flush=True)
    assert all(reference.losses[key] != 0.0 for key in mbs.LOSS_KEYS), reference.losses
    assert all(value <= LOSS_RELATIVE for value in loss_rel.values()), loss_rel
    assert mbs.max_row_difference(rows, reference.rows) <= LOSS_RELATIVE


def test_every_round_records_moe_statistics(results):
    reference, _ = results["ref"]
    round_values = reference.bins["seq_load_balancing_loss"]
    assert round_values is not None and round_values.shape[0] == ROUNDS
    assert all(round_values[r].abs().sum() > 0 for r in range(ROUNDS)), round_values


@pytest.mark.parametrize("arm", ["f1_bptt3", "z3_bptt3"])
def test_dropping_round_one_fails_every_core_parameter(results, arm):
    """Teeth (a): without round 1's backward every resolved core parameter fails the gate."""
    _, reference = results["ref"]
    _, grads = results[arm]
    report, total = _gate(grads, reference)
    _summary(_tag(f"teeth (a) {arm} vs ref"), report, total)
    core = [name for name, entry in report.items() if entry["core"] and not entry["zero"]]
    assert core
    passing = [name for name in core if report[name]["ok"]]
    assert not passing, mbs.worst(report, passing)


@pytest.mark.parametrize("arm", ["z2_legacy", "z3_legacy"])
def test_legacy_overwrite_fails_every_fused_core_weight(results, arm):
    """Teeth (b): without accumulate mode each use of a core weight overwrites the gradient of
    its earlier uses: every core weight TE's fused weight gradient writes fails the gate (and the
    output layer, shared by the main and the MTP head). The core parameters it cannot touch
    (autograd sums their uses before Megatron-FSDP's hook) are counted."""
    _, reference = results["ref"]
    _, grads = results[arm]
    report, total = _gate(grads, reference)
    _summary(_tag(f"teeth (b) {arm} vs ref"), report, total)
    fused = results["fused_core_weights"]
    assert fused
    passing = [name for name in fused if report[name]["ok"]]
    assert not passing, mbs.worst(report, passing)
    assert not report["output_layer.weight"]["ok"], report["output_layer.weight"]
    untouched = [
        name
        for name, entry in report.items()
        if entry["core"] and entry["ok"] and name not in fused and not entry["zero"]
    ]
    if dist.get_rank() == 0:
        print(
            f"[{_tag(f'teeth (b) {arm}')}] fused core weights {len(fused)} (all fail); "
            f"other resolved core parameters that pass: {len(untouched)}",
            flush=True,
        )


def _state_tensors(state):
    """The parameters and persistent buffers of a state dict (TE's ``_extra_state`` left out:
    serialized metadata, not weights)."""
    return {
        name: value
        for name, value in state.items()
        if torch.is_tensor(value) and not name.endswith("_extra_state")
    }


def _same(actual, expected):
    """Names whose tensors differ (dtype, shape or any bit)."""
    assert actual.keys() == expected.keys(), set(actual) ^ set(expected)
    return [
        name
        for name, value in expected.items()
        if actual[name].dtype != value.dtype or not torch.equal(actual[name], value)
    ]


def test_model_only_load_before_the_megatron_fsdp_wrap(results, tmp_path_dist_ckpt):
    """The Megatron-FSDP parity runs' load (training's ``_load_loop_model_before_megatron_fsdp_
    wrap``): a model-only torch_dist checkpoint of ``ref``'s weights (``save_checkpoint`` with
    ``--no-save-optim --no-save-rng``, as the shared initial checkpoint was made) loads under
    ``raise_unexpected`` into a model with other weights, with the run's ``--ckpt-format
    fsdp_dtensor`` restored afterwards; then Megatron-FSDP (optim_grads_params) holds exactly
    those weights in bf16 and their fp32 copies as main weights."""
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.training.training import _load_loop_model_before_megatron_fsdp_wrap
    from tests.unit_tests.transformer.loop import store_ckpt_gpu_utils as cku

    tensors = _state_tensors(results["reference_state"])
    full = ProcessGroupCollection.use_mpu_process_groups()
    with cku.checkpoint_dir(tmp_path_dist_ckpt / "loop_fsdp_model_only") as ckpt_dir:
        saved = _model(results["reference_state"])
        saved_ddp = _ddp(saved)
        with cku.mock_checkpoint_args(
            ckpt_dir, loop_config=saved.config, no_save_optim=True, no_save_rng=True
        ):
            cku.save(saved_ddp, None, full, iteration=1)
        _release(saved_ddp)
        del saved_ddp, saved

        model = _model(seed=97)  # other weights
        assert _same(_state_tensors(model.state_dict()), tensors)  # the load has work to do
        with cku.mock_checkpoint_args(
            ckpt_dir,
            loop_config=model.config,
            finetune=True,
            no_load_optim=True,
            no_load_rng=True,
            exit_on_missing_checkpoint=True,
            dist_ckpt_strictness="raise_unexpected",
            ckpt_format="fsdp_dtensor",
            use_megatron_fsdp=True,
        ) as args:
            iteration, _ = _load_loop_model_before_megatron_fsdp_wrap([model], args)
            assert args.ckpt_format == "fsdp_dtensor"
    assert iteration == 0  # --finetune
    assert _same(_state_tensors(model.state_dict()), tensors) == []

    weights = tensors
    params = dict(model.named_parameters())
    fsdp = _fsdp(model, "optim_grads_params")
    model_weights = _fsdp_items(fsdp, "model_weight_buffer")
    main_weights = _fsdp_items(fsdp, "main_weight_buffer")
    _release(fsdp)
    assert set(model_weights) == set(params) and set(main_weights) == set(params)
    assert _same(model_weights, {name: weights[name] for name in params}) == []
    assert _same(main_weights, {name: weights[name].float() for name in params}) == []
