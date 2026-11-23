# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared pieces of the per-parameter gradient tests of the weight-distribution strategies on a
fixed-depth DSv4 loop (GPU): DDP communication overlap (``test_loop_ddp_overlap_gpu.py``),
Megatron-FSDP HSDP (``test_loop_hsdp_gpu.py``) and module GTP (``test_loop_gtp_gpu.py``).

The model is the Pro experiment's layer roles at test width: pattern
``'WEWECEHECEHECEHECE/WE'`` (a 6-sublayer prelude with the three hash-routed MoE layers, an
8-sublayer core run exactly 4 times, a 4-sublayer coda ``HECE``, MTP ``WE``), DSv4 W / C / H
attention (unfused, ``dsa_kernel_backend='none'``), 8 experts top-2 with the Flash router (sqrt
softplus, top-k scaling 1.5, expert bias, sequence aux loss, one shared expert), the sparse
indexer loss, mHC with 4 streams, MTP, bf16 with TE fused weight gradients, deterministic kernels,
per-token loss, and block + edge ``loop_recompute`` with stride 4 (two blocks per round, the
adapter in the first and the loop norm in the second; the prelude in blocks of 4 + 2 and the coda
in one block). A core weight is therefore used 4 times per microbatch in the forward and again in
the recompute.

Teacher forcing of step 2. Step 1 of every arm is gated against its reference as it is. Then the
arm's reduced step-1 gradient storage (what its optimizer reads) is overwritten in place with the
reference's reduced step-1 gradient, and the arm runs its own optimizer step, gradient-buffer
reset, parameter gathers and step-2 forward / backward. Step 2 is gated only after an asserted
precondition: the arm starts step 2 from a state exactly equal to the reference's (weights, or
for Megatron-FSDP the fp32 main weights the optimizer produced, and the persistent buffers; the
Megatron-FSDP compute weights the optimizer step installed are checked in a test of their own).
"Exactly equal" is ``mismatches``: same dtype and shape and ``torch.equal`` element values, so
-0.0 equals +0.0 (a read-back that sums a rank's part with the other ranks' zeros turns -0.0
into +0.0) and NaN equals nothing. Without the forcing the step-2 gate would ask the model to be
insensitive to step-1 gradients that differ by fp32 reassociation: the zero-initialised fp32
parameters (``hyper_connection.bias``, ``attn_sink``, ``hc_head_base``) take those last bits
straight into their weights, and a tipped bf16 activation can flip a top-k routing or indexer
decision. The un-forced step 2 is kept as an
ungated diagnostic. Forcing hides a production write into the gradient storage between the end
of ``finalize_model_grads`` and the optimizer step; a late asynchronous write would still break
the precondition.

Precise embedding reference (``install_per_use_fp32_embedding``, installed by
``test_loop_gtp_gpu.py`` only). The model looks the embedding up twice per microbatch (the main
input and the MTP input). In ``deterministic_mode`` the lookup is ``weight[ids]``, so autograd sums
the two uses' BF16 gradients in BF16 before DDP adds the sum into the FP32 ``main_grad``: about
1.7e-3 relative error on the embedding gradient (a CPU rounding model and real CPU autograd ops).
GTP reduce-scatters each use's BF16 gradient on its own in FP32, which is exact up to FP32
reassociation, so against that reference even an exact GTP embedding fails the 1e-3 gate. The
installer makes the reference add each use's BF16 gradient (the same index backward the GTP arm
runs) into the FP32 ``main_grad`` on its own, what non-deterministic DP's fused embedding wgrad
does too. The gate is unchanged.

Every helper that communicates is collective: all ranks call it in the same order.
"""

import collections
import json
import os
import re
from types import SimpleNamespace

import torch
import torch.distributed as dist

from tests.unit_tests.transformer.loop import mbs_invariance_utils as mbs

PATTERN = "WEWECEHECEHECEHECE/WE"
_RATIOS = {"W": 0, "C": 4, "H": 128, "E": 0}
PRELUDE, CODA = 6, 4
CORE = len(PATTERN.split("/")[0]) - PRELUDE - CODA
ROUNDS = 4
STRIDE = 4
SEQ = 256
VOCAB = 512
NUM_EXPERTS = 8
ROWS = 3  # per rank: three microbatches of one row (gradient accumulation across microbatches)
MTP_LAYERS = 1  # the embedding and output_layer are each used 1 + MTP_LAYERS times per microbatch
MIN_COSINE = 0.9999
MAX_RELATIVE = 1e-3
ZERO_FRACTION = 1e-6
LOSS_RELATIVE = 1e-6


def build_model(state=None, *, seed=1234, **overrides):
    """The test model (see the module docstring) on the current device. ``state``: a state
    dict to load (strict). ``overrides``: config fields (e.g. ``expert_model_parallel_size``,
    ``gtp_weight_remat_size``)."""
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
        num_moe_experts=NUM_EXPERTS,
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
        mtp_num_layers=MTP_LAYERS,
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
        max_sequence_length=1024,
        hybrid_layer_pattern=PATTERN,
        position_embedding_type="rope",
    ).cuda()
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def clone_state(model):
    """A detached copy of ``model``'s state dict (the shared starting point of every arm)."""
    return {
        name: value.detach().clone() if torch.is_tensor(value) else value
        for name, value in model.state_dict().items()
    }


def run_step(wrapped, model, rows, *, on_microbatch=None):
    """One training step over ``rows`` (microbatches of one row) through the no-pipelining
    schedule and ``finalize_model_grads``; returns the attached loss sums and per-row LM losses.
    ``on_microbatch(index)`` runs before each microbatch's forward.

    The schedule's ``no_sync_func`` is set as ``megatron.training.training.train`` sets it
    (``reduces_grads_during_backward``): the wrapper's ``no_sync`` when it reduces gradients from
    its backward hooks (DDP or Megatron-FSDP with ``overlap_grad_reduce``, which Megatron-FSDP
    turns on itself for the gradient-sharding strategies). Without it Megatron-FSDP never sees
    the last microbatch, and its HSDP / HFSDP DP-Outer reduction never runs."""
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
    from megatron.training.training import reduces_grads_during_backward

    model.config.finalize_model_grads_func = finalize_model_grads
    model.config.no_sync_func = wrapped.no_sync if reduces_grads_during_backward(wrapped) else None
    set_moe_metrics_tracker(MoEMetricsTracker())
    wrapped.zero_grad_buffer()
    recorder = mbs._Recorder()
    counter = iter(range(1 << 30))

    def forward_step(data_iterator, model_chunk):
        if on_microbatch is not None:
            on_microbatch(next(counter))
        return mbs._forward_step(recorder, data_iterator, model_chunk)

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
    return SimpleNamespace(losses=dict(recorder.sums), rows=list(recorder.rows))


def ddp_grads(ddp, model):
    """Every parameter's reduced fp32 gradient under Megatron DDP. With the distributed
    optimizer each rank holds the reduce-scattered shard of every bucket, so the shards are
    first all-gathered over the bucket group's data-parallel group (in place: the optimizer
    reads only its own shard, which is unchanged)."""
    from megatron.core.distributed.param_and_grad_buffer import shard_buffer

    if ddp.ddp_config.use_distributed_optimizer:
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


def fsdp_bucket_part(gbuf, strategy):
    """This rank's reduced part of a Megatron-FSDP gradient bucket (``gbuf``: a group's main
    gradient buffer; ``strategy``: the group's sharding strategy) in a bucket-sized tensor, and
    whether the other ranks' parts have to be summed in over ``gbuf.data_parallel_group``.

    * ``no_shard``: the whole bucket is all-reduced on every rank (returned as is).
    * Otherwise only the rank's shard is reduced. A sharded buffer stores just that shard; an
      ``optim`` (ZeRO-1) buffer is unsharded, and outside the shard it keeps the rank's own
      unreduced sums. The shard is placed at its offset in the bucket and the rest is zero.
      The offset comes from the buffer's own shard index: under DP-Outer sharding (HFSDP)
      shards are indexed by the logical hybrid rank (DP-Shard major), not by the rank's
      position in the group, so the parts are summed rather than all-gathered in group order.
    """
    from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
        DataParallelBuffer,
    )

    size = gbuf.bucket_index.size
    if strategy == "no_shard":
        start = gbuf.bucket_index.global_data_index
        return gbuf.data[start : start + size], False
    index = gbuf.shard_bucket_index
    part = torch.zeros(size, dtype=gbuf.dtype, device=gbuf.data.device)
    shard = DataParallelBuffer.get_shard_from_local_buffer(gbuf)
    part[index.bucket_data_index : index.bucket_data_index + index.size].copy_(shard)
    return part, True


def fsdp_grads(fsdp):
    """Every parameter's reduced gradient under Megatron-FSDP: each group's reduced shards
    summed over the group's data-parallel group (``fsdp_bucket_part``)."""
    from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import Bucket

    buffer = fsdp.param_and_grad_buffer
    grads = {}
    for group in buffer.parameter_groups:
        gbuf = group.main_grad_buffer
        if gbuf is None or not group.requires_grad:
            continue
        full, partial = fsdp_bucket_part(gbuf, group.sharding_strategy)
        if partial:
            dist.all_reduce(full, group=gbuf.data_parallel_group)
        for param in group.params:
            item = gbuf.get_item_from_bucket(Bucket(data=full), gbuf.param_idx[param])
            grads[buffer.param_to_name[param]] = item.view(param.shape).float().clone()
    return grads


_LAYER = re.compile(r"^(?:module\.)*decoder\.layers\.(\d+)\.")


def role(name):
    """``(role, sublayer index)`` of a parameter name: prelude / core / coda by the sublayer of
    the main pattern, mtp, or other (embedding, heads, loop adapter and norm, final norm)."""
    match = _LAYER.match(name)
    if match:
        index = int(match.group(1))
        if index < PRELUDE:
            return "prelude", index
        if index < PRELUDE + CORE:
            return "core", index
        return "coda", index
    if re.match(r"^(?:module\.)*mtp\.", name):
        return "mtp", None
    return "other", None


def gate(grads, reference):
    """Per-parameter comparison against ``reference``: cosine >= ``MIN_COSINE`` and relative
    norm difference <= ``MAX_RELATIVE``; a gradient that is zero to fp32 precision (at most
    ``ZERO_FRACTION`` of the whole gradient's norm) is gated absolutely at that bound."""
    assert grads.keys() == reference.keys(), sorted(set(grads) ^ set(reference))[:8]
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


def tag(prefix, text):
    return f"{prefix} W{dist.get_world_size()} rank{dist.get_rank()} {text}"


def dump(name, payload):
    """Write ``payload`` to ``$MBS_REPORT_DIR/<name>.json`` when the variable is set."""
    directory = os.environ.get("MBS_REPORT_DIR")
    if not directory:
        return
    os.makedirs(directory, exist_ok=True)
    file_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    with open(os.path.join(directory, f"{file_name}.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, default=float)


def report_and_fail(name, grads, reference):
    """Gate ``grads`` against ``reference``, print and dump the report; returns
    ``(report, failing names)``."""
    report, total = gate(grads, reference)
    print(f"[{name}] whole gradient {total}", flush=True)
    mbs.print_report(name, report)
    dump(name, dict(per_param=report, total=total))
    return report, [param for param, entry in report.items() if not entry["ok"]]


def assert_losses_match(step, reference, name):
    """The forward is the same function of the same weights: attached losses and per-row LM
    losses equal the reference's within ``LOSS_RELATIVE``."""
    loss_rel = mbs.relative_loss_differences(step.losses, reference.losses)
    print(f"[{name}] losses {step.losses} rel {loss_rel}", flush=True)
    assert all(reference.losses[key] != 0.0 for key in mbs.LOSS_KEYS), reference.losses
    assert all(value <= LOSS_RELATIVE for value in loss_rel.values()), loss_rel
    assert mbs.max_row_difference(step.rows, reference.rows) <= LOSS_RELATIVE


SGD_LR = 0.1
# The distributed optimizer refuses torch/TE SGD (DistributedOptimizer.__init__ takes Adam or an
# optimizer with an init_state_fn, for checkpointing), so SGD without momentum runs as Adam with
# beta1 = beta2 = 0 (m = g, v = g * g, both bias corrections 1) and an eps far above every
# gradient entry: the update lr * g / (|g| + eps) is SGD_LR * g / (1 + |g| / eps), i.e.
# SGD_LR * g to fp32 rounding while |g| < 4 (half an fp32 ulp of 1e8). 1e8 and SGD_LR * 1e8 are
# exact in fp32. Every arm, with or without the distributed optimizer, uses the same optimizer.
SGD_ADAM_EPS = 1e8
SGD_AS_ADAM = dict(
    optimizer="adam",
    lr=SGD_LR * SGD_ADAM_EPS,
    adam_beta1=0.0,
    adam_beta2=0.0,
    adam_eps=SGD_ADAM_EPS,
)


def optimizer(wrapped, **overrides):
    """A Megatron optimizer over ``wrapped`` (DDP or Megatron-FSDP): SGD without momentum by
    default (``SGD_AS_ADAM``), so the step-2 weights are a plain function of step 1's gradients;
    ``overrides``: further ``OptimizerConfig`` fields (e.g. the CPU-offloaded Adam). An override
    that names ``optimizer`` replaces the SGD emulation as a whole."""
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.process_groups_config import ProcessGroupCollection

    fields = dict(
        weight_decay=0.0,
        clip_grad=0.0,
        use_distributed_optimizer=wrapped.ddp_config.use_distributed_optimizer,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    if "optimizer" not in overrides:
        fields.update(SGD_AS_ADAM)
    fields.update(overrides)
    return get_megatron_optimizer(
        OptimizerConfig(**fields),
        [wrapped],
        use_gloo_process_groups=False,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )


def step_optimizer(opt):
    update_successful, _, _ = opt.step()
    assert update_successful


def fused_core_weights(model):
    """The core's GEMM weights whose weight gradient TE accumulates in place (fused wgrad)."""
    names = []
    for module_name, module in model.named_modules():
        if not getattr(module, "fuse_wgrad_accumulation", False):
            continue
        for param_name, _ in module.named_parameters(recurse=False):
            name = f"{module_name}.{param_name}"
            if role(name)[0] == "core" and re.fullmatch(r"weight\d*", param_name):
                names.append(name)
    return names


def make_rows(step=0):
    """``ROWS`` rows of random tokens for this rank and ``step``: the same in every arm,
    different on every rank (every rank is a data-parallel rank in these tests)."""
    return mbs.make_rows(ROWS, SEQ, VOCAB, seed=4242 + 97 * step + dist.get_rank())


# ---------------------------------------------------------------------------------------------
# Teacher forcing of step 2 (see the module docstring)
# ---------------------------------------------------------------------------------------------


def _local(tensor):
    """``tensor``'s local data (a DTensor: its local shard)."""
    from torch.distributed.tensor import DTensor

    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def force_main_grads(model, reference, shard=None):
    """Teacher forcing under Megatron DDP: copy ``reference`` (full reduced gradients by name)
    in place into every parameter's ``main_grad``, the storage the optimizer reads at its step
    (the distributed optimizer its range of it, the float16 optimizer all of it). ``shard(name,
    param, full)``: the part of ``full`` this rank's ``main_grad`` holds (a GTP shard's rows);
    default the whole tensor."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            value = reference[name]
            if shard is not None:
                value = shard(name, param, value)
            assert value.shape == param.main_grad.shape, (name, value.shape, param.main_grad.shape)
            param.main_grad.copy_(value)


def fsdp_force_bucket_part(gbuf, strategy, full):
    """The inverse of ``fsdp_bucket_part``: write this rank's reduced part of ``full`` (a whole
    bucket of reduced gradients) into the storage ``fsdp_bucket_part`` reads, and the optimizer
    too (its gradients are views of it). ``no_shard``: the whole bucket; otherwise the rank's
    shard, at the buffer's own shard index (DP-Outer sharding's logical order included). An
    ``optim`` (ZeRO-1) buffer keeps its unreduced local sums outside the shard."""
    from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
        DataParallelBuffer,
    )

    size = gbuf.bucket_index.size
    assert full.numel() == size, (full.numel(), size)
    if strategy == "no_shard":
        start = gbuf.bucket_index.global_data_index
        gbuf.data[start : start + size].copy_(full)
        return
    index = gbuf.shard_bucket_index
    shard = DataParallelBuffer.get_shard_from_local_buffer(gbuf)
    shard.copy_(full[index.bucket_data_index : index.bucket_data_index + index.size])


def force_fsdp_grads(fsdp, reference):
    """Teacher forcing under Megatron-FSDP: every group's reduced gradient storage set to
    ``reference``'s values (the mirror of ``fsdp_grads``; local, no communication)."""
    from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import Bucket

    buffer = fsdp.param_and_grad_buffer
    with torch.no_grad():
        for group in buffer.parameter_groups:
            gbuf = group.main_grad_buffer
            if gbuf is None or not group.requires_grad:
                continue
            full = torch.zeros(gbuf.bucket_index.size, dtype=gbuf.dtype, device=gbuf.data.device)
            for param in group.params:
                item = gbuf.get_item_from_bucket(Bucket(data=full), gbuf.param_idx[param])
                item.copy_(reference[buffer.param_to_name[param]].reshape(-1))
            fsdp_force_bucket_part(gbuf, group.sharding_strategy, full)


def item_window(item_start, item_size, shard_start, shard_size, sharded):
    """``[start, end)`` within an item (``item_size`` elements at ``item_start`` in its bucket)
    of what a rank's optimizer holds: the whole item when the optimizer state is not sharded,
    else its intersection with the rank's shard window ``[shard_start, shard_start +
    shard_size)`` of the bucket (empty: ``start == end``)."""
    if not sharded:
        return 0, item_size
    start = min(max(shard_start - item_start, 0), item_size)
    end = max(min(shard_start + shard_size - item_start, item_size), start)
    return start, end


def _fsdp_optimizer_windows(fsdp, buffer_of):
    """``(name, optimizer parameter, (start, end))`` of every trainable Megatron-FSDP parameter:
    the window of the flattened item this rank's optimizer parameter holds, computed by the test
    (``item_window``) from the indices of the group's ``buffer_of(group)``."""
    buffer = fsdp.param_and_grad_buffer
    for name, param in buffer.optimizer_named_parameters:
        group = buffer.parameter_groups[buffer.param_to_param_group[param.orig_param]]
        if not group.requires_grad or group.main_grad_buffer is None:
            continue
        buf = buffer_of(group)
        item = buf.item_index_map[buf.param_idx[param.orig_param]]
        window = item_window(
            item.global_data_index - buf.bucket_index.global_data_index,
            item.size,
            buf.shard_bucket_index.bucket_data_index,
            buf.shard_bucket_index.size,
            group.sharding_strategy != "no_shard",
        )
        yield name, param, window


def fsdp_main_weight_slices(fsdp):
    """{name: (window, fp32 values)} of this rank's Megatron-FSDP main weights (the optimizer's
    parameters, i.e. its output after a step); the window within the flattened item is the
    test's own arithmetic on the main weight buffer's indices (local)."""
    out = {}
    for name, param, window in _fsdp_optimizer_windows(
        fsdp, lambda group: group.main_weight_buffer or group.model_weight_buffer
    ):
        out[name] = (window, _local(param).detach().reshape(-1).clone())
    return out


def buffer_item_shard(buf, item_id):
    """``((start, end), values)``: the part of item ``item_id`` in this rank's shard of a
    sharded Megatron-FSDP buffer (or the virtual shard of an unsharded one), i.e. where this
    rank's optimizer step writes it; the window within the flattened item is ``item_window`` on
    the buffer's own indices, the values are read from ``get_shard_from_local_buffer`` (local)."""
    from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
        DataParallelBuffer,
    )

    item = buf.item_index_map[item_id]
    item_start = item.global_data_index - buf.bucket_index.global_data_index
    index = buf.shard_bucket_index
    start, end = item_window(item_start, item.size, index.bucket_data_index, index.size, True)
    shard = DataParallelBuffer.get_shard_from_local_buffer(buf)
    offset = item_start - index.bucket_data_index
    return (start, end), shard[offset + start : offset + end].detach().clone()


def fsdp_model_weight_slices(fsdp):
    """{name: (window, values)} of the Megatron-FSDP compute weights this rank's optimizer step
    installs (``copy_main_weights_to_model_weights``: main weight shard -> model weight buffer),
    read right after the step: this rank's shard of each trainable parameter's model weight
    buffer (``buffer_item_shard``), or the whole module parameter of a ``no_shard`` group,
    which has no model weight buffer (local)."""
    buffer = fsdp.param_and_grad_buffer
    out = {}
    for group in buffer.parameter_groups:
        if not group.requires_grad or group.main_grad_buffer is None:
            continue
        wbuf = group.model_weight_buffer
        for param in group.params:
            name = buffer.param_to_name[param]
            if wbuf is None:
                values = _local(param).detach().reshape(-1).clone()
                out[name] = ((0, values.numel()), values)
            else:
                out[name] = buffer_item_shard(wbuf, wbuf.param_idx[param])
    return out


def install_mismatches(installed, weights, main):
    """Where the compute weights a Megatron-FSDP arm installed (``installed``:
    ``fsdp_model_weight_slices``) differ from a DDP run's weights after the same step
    (``weights``: full tensors by name, bf16 = the fp32 masters rounded) in the same windows
    (local, ``mismatches``' format). Every name of the main weight slices (``main``) must be
    installed in the window its optimizer parameter holds. Never raises on a mismatch: the
    result is a test's to assert."""
    found = {name: ("not installed",) for name in sorted(set(main) - set(installed))}
    got, expected = {}, {}
    for name, (window, values) in installed.items():
        if name not in weights:
            found[name] = ("no reference weight",)
        elif name in main and main[name][0] != window:
            found[name] = ("window", window, main[name][0])
        else:
            got[name] = values
            expected[name] = weights[name].reshape(-1)[window[0] : window[1]].to(values.device)
    found.update(mismatches(got, expected))
    return found


def install_record(record, weights):
    """The install check of a Megatron-FSDP arm (collective: call it for every arm in the same
    order on every rank): this rank's ``install_mismatches`` of ``record["installed"]`` against
    ``weights`` (or the error reading them) and the largest count over the ranks."""
    if record.get("installed") is None:
        local = {"error": record.get("installed_error", "the installed weights were not read")}
    else:
        local = install_mismatches(record["installed"], weights, record["main"])
    return dict(local=local, agreed=agreed(len(local)))


def slice_mismatches(slices, full):
    """Exact comparison (local, ``mismatches``' format) of ``slices``
    (``fsdp_main_weight_slices``) with the same windows of ``full`` (full tensors by name)."""
    assert slices.keys() == full.keys(), sorted(set(slices) ^ set(full))[:8]
    got, expected = {}, {}
    for name, ((start, end), values) in slices.items():
        got[name] = values
        # No cast: a slice of another dtype is a mismatch, not a comparison of rounded values.
        expected[name] = full[name].reshape(-1)[start:end].to(values.device)
    return mismatches(got, expected)


def slice_relative_difference(slices, full):
    """The largest per-tensor relative norm difference of ``slices`` against the same windows of
    ``full`` (local; a diagnostic where no exact equality is expected)."""
    worst = 0.0
    for name, ((start, end), values) in slices.items():
        want = full[name].reshape(-1)[start:end].to(values.device, torch.float64)
        diff = (values.double() - want).norm().item()
        worst = max(worst, diff / max(want.norm().item(), 1e-30) if diff else 0.0)
    return worst


def fsdp_start_mismatches(record, reference):
    """Where the state a Megatron-FSDP arm's step 2 starts from differs from a DDP
    run's (local): the arm's fp32 main weight slices (``record["main"]``) against the run's fp32
    masters, and the persistent buffers (``record`` / ``reference["buffers"]``)."""
    found = slice_mismatches(record["main"], reference["masters"])
    for name, value in mismatches(record["buffers"], reference["buffers"]).items():
        found[f"buffer {name}"] = value
    return found


def fsdp_optimizer_grad_mismatches(fsdp, reference):
    """Local diagnostic of a forced Megatron-FSDP arm (``mismatches``' format): every optimizer
    parameter's gradient, looked up as ``HybridDeviceOptimizer`` and the precision-aware
    optimizer consume it (``decoupled_grad`` when the attribute exists, even when it is None,
    else ``grad``; its local shard), against the same window of ``reference``, on the main
    gradient buffer's indices. A missing gradient is a mismatch. A mismatch names a gradient
    the forcing did not reach (a copy of the buffer, not a view)."""
    got, expected = {}, {}
    for name, param, (start, end) in _fsdp_optimizer_windows(
        fsdp, lambda group: group.main_grad_buffer
    ):
        # hybrid_optimizer.py: getattr(param, "decoupled_grad", param.grad).
        grad = param.decoupled_grad if hasattr(param, "decoupled_grad") else param.grad
        want = reference[name].reshape(-1)[start:end].float()
        if grad is None:
            got[name] = torch.full_like(want, float("nan"))
        else:
            got[name] = _local(grad).detach().reshape(-1).float()
        expected[name] = want.to(got[name].device)
    return mismatches(got, expected)


def float16_optimizer_masters(opt, model):
    """{name: fp32 master} of a Megatron float16 optimizer (no distributed optimizer; a
    ``ChainedOptimizer`` is walked): the fp32 copies of the bf16 parameters and the fp32
    parameters themselves."""
    names = {param: name for name, param in model.named_parameters()}
    masters = {}
    pending = [opt]
    while pending:
        current = pending.pop()
        chained = getattr(current, "chained_optimizers", None)
        if chained is not None:
            pending.extend(chained)
            continue
        for model_group, main_group in zip(
            current.float16_groups, current.fp32_from_float16_groups
        ):
            for model_param, main_param in zip(model_group, main_group):
                masters[names[model_param]] = main_param.detach().float().clone()
        for group in current.fp32_from_fp32_groups:
            for param in group:
                masters[names[param]] = param.detach().float().clone()
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert masters.keys() == trainable, sorted(set(masters) ^ trainable)[:8]
    return masters


def persistent_buffers(model):
    """A copy of ``model``'s persistent buffers (the router expert bias that
    ``finalize_model_grads`` updates)."""
    out = {}
    for module_name, module in model.named_modules():
        for name, buf in module.named_buffers(recurse=False):
            if buf is None or name in module._non_persistent_buffers_set:
                continue
            out[f"{module_name}.{name}" if module_name else name] = buf.detach().clone()
    return out


def mismatches(got, expected):
    """Exact comparison (local): ``{name: (differing elements, dtype, max abs difference)}``
    of every tensor of ``expected`` that ``got`` does not equal exactly: a different dtype or
    shape, or an element whose value differs (``torch.equal``: -0.0 equals +0.0, NaN equals
    nothing). Not a comparison of bit patterns: the sign of a zero is not compared."""
    assert got.keys() == expected.keys(), sorted(set(got) ^ set(expected))[:8]
    found = {}
    for name, want in expected.items():
        have = got[name]
        if have.dtype != want.dtype:
            found[name] = ("dtype", str(have.dtype), str(want.dtype))
        elif have.shape != want.shape:
            found[name] = ("shape", tuple(have.shape), tuple(want.shape))
        elif not torch.equal(have, want):
            diff = (have.double() - want.double()).abs()
            found[name] = (
                int((have != want).sum().item()),
                str(want.dtype).replace("torch.", ""),
                float(diff.max().item()),
            )
    return found


def agreed(count):
    """The largest ``count`` over all ranks (collective)."""
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    value = torch.tensor([int(count)], dtype=torch.int64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return int(value.item())


def forcing_record(readback, start):
    """The precondition record of a forced arm (collective: call it for every forced arm in the
    same order on every rank). ``readback``: the number of tensors whose reduced gradient, read
    back after the forcing, differs from the forced one (already agreed over the ranks);
    ``start``: this rank's ``mismatches`` of the state step 2 starts from."""
    return dict(readback=readback, start=start, agreed=agreed(len(start)))


def assert_teacher_forced(forcing, arm, what="the reference's"):
    """The precondition of a step-2 gate: the arm read the forced gradient back on every rank
    and started step 2 from a state exactly equal to ``what``."""
    assert forcing is not None, f"{arm} was not teacher-forced"
    assert forcing["readback"] == 0, (
        f"teacher forcing: {arm}'s reduced gradient read back after the forcing differs from "
        f"the forced one in {forcing['readback']} tensors (worst rank)"
    )
    local = forcing["start"]
    assert forcing["agreed"] == 0, (
        f"teacher-forcing precondition: {arm} starts step 2 from a state that differs "
        f"from {what} in {len(local)} tensors on this rank ({forcing['agreed']} on the worst "
        "rank): its optimizer step on the forced step-1 gradient (or its parameter gather) did "
        "not reproduce the update",
        sorted(local.items())[:8],
    )


def gtp_rows(full, rows, size, rank):
    """GTP shard ``rank`` of ``full``: rows ``[rank rows, (rank + 1) rows)`` of ``full`` padded
    with zero rows to ``rows x size`` (the layout of a GTP weight and its ``main_grad``)."""
    padded = torch.zeros((rows * size, *full.shape[1:]), dtype=full.dtype, device=full.device)
    padded[: full.shape[0]].copy_(full)
    return padded[rank * rows : (rank + 1) * rows].clone()


# ---------------------------------------------------------------------------------------------
# Precise embedding reference (test_loop_gtp_gpu.py only; see the module docstring)
# ---------------------------------------------------------------------------------------------


class _PerUseFp32WeightGrad(torch.autograd.Function):
    """Identity on a weight; its backward adds THIS use's gradient (in the weight's dtype) into
    the FP32 ``main_grad`` on its own (an exact widening, one FP32 rounding), flags
    ``grad_added_to_main_grad``, and returns a zero placeholder. With PP=1 and MTP, DDP
    adds the placeholder because the input embedding has ``zero_out_wgrad`` set."""

    @staticmethod
    def forward(ctx, weight):
        ctx.weight = weight
        return weight.view_as(weight)

    @staticmethod
    def backward(ctx, grad):
        weight = ctx.weight
        assert weight.main_grad.dtype == torch.float32, weight.main_grad.dtype
        assert grad.dtype == weight.dtype, (grad.dtype, weight.dtype)
        weight.main_grad.add_(grad)
        weight.grad_added_to_main_grad = True
        return torch._efficientzerotensor(weight.shape, dtype=weight.dtype, device=weight.device)


def _assert_plain_lookup(module):
    """The conditions under which ``VocabParallelEmbedding.forward`` is ``weight[ids]``: no
    tensor-parallel split, no GTP, ``deterministic_mode`` and no reduce-scatter of the output."""
    layout = dict(
        tp=module.tp_group.size(),
        gtp=getattr(module, "gtp_remat_size", 1),
        deterministic=module.deterministic_mode,
        reduce_scatter=module.reduce_scatter_embeddings,
    )
    assert layout == dict(tp=1, gtp=1, deterministic=True, reduce_scatter=False), (
        "the per-use FP32 embedding reference replaces the deterministic TP=1 lookup only",
        layout,
    )


def install_per_use_fp32_embedding(model):
    """Make ``model``'s embedding lookups add each use's gradient into the FP32 ``main_grad`` on
    its own (``_PerUseFp32WeightGrad``); returns a counter whose ``lookups`` counts the lookups
    (both uses go through the same module: the main input and the MTP input).

    Replaces ``model.embedding.word_embeddings.forward`` on that instance by ``weight[ids]``
    through the per-use Function, which is the original forward's deterministic TP=1 branch, so
    each use's gradient comes from the same index backward. Asserts the layout at install and on
    every call, an untied weight (a tied one also gets the output layer's gradient through
    ``.grad``, which the flag would make DDP skip), and on the first call that the output equals
    the original forward's exactly."""
    module = model.embedding.word_embeddings
    _assert_plain_lookup(module)
    assert not getattr(model, "share_embeddings_and_output_weights", False), "tied embedding"
    original = module.forward
    counter = SimpleNamespace(lookups=0, checked=False)

    def forward(input_):
        _assert_plain_lookup(module)
        weight = module.weight
        if not counter.checked:
            with torch.no_grad():
                expected = original(input_)
                assert torch.equal(weight[input_], expected), "the lookup drifted from the original"
            counter.checked = True
        counter.lookups += 1
        return _PerUseFp32WeightGrad.apply(weight)[input_]

    module.forward = forward
    return counter


# ---------------------------------------------------------------------------------------------
# Megatron-FSDP reduction schedule: expectations from the test's own layout arithmetic and the
# configured strategies (no production predicate is called)
# ---------------------------------------------------------------------------------------------

EXPERT = re.compile(r"\.mlp\.experts\.")  # routed experts, not ``shared_experts``
GRAD_SHARDING = ("optim_grads", "optim_grads_params")


def is_expert(name):
    return bool(EXPERT.search(name))


def expected_dp_shard(world, ep, instances, expert):
    """The DP-Shard size of a dense (``W / instances``) or expert (``W / (EP instances)``)
    gradient bucket."""
    return world // (ep * instances) if expert else world // instances


def reduced_every_microbatch(strategy, dp_shard):
    """Whether a bucket is reduced over DP-Shard on every microbatch: only under a
    gradient-sharding strategy (ZeRO-2 / ZeRO-3) over a DP-Shard group wider than one rank. A
    ``no_shard`` or ``optim`` (ZeRO-1) bucket keeps a persistent unsharded gradient buffer that
    accumulates every microbatch locally and is reduced once per step, in the last microbatch
    (an earlier reduction would count the in-place reduced shard again); so does a sharding
    strategy over a one-rank group, whose buffer is not distributed either."""
    return strategy in GRAD_SHARDING and dp_shard > 1


def expected_once_per_step(names, *, dense, expert, world, ep, instances):
    """The parameter names whose buckets are reduced once per step, from the layout and the
    configured strategies (``expert``: the expert strategy, ``None`` for ``dense``'s)."""
    assert world // instances > 1, "the dense DP-Shard group always has more than one rank"
    expert = expert or dense
    return {
        name
        for name in names
        if not reduced_every_microbatch(
            expert if is_expert(name) else dense,
            expected_dp_shard(world, ep, instances, is_expert(name)),
        )
    }


def fsdp_bucket_layout(fsdp):
    """Per gradient bucket of a ``FullyShardedDataParallel``: the DP-Shard size of its reducing
    buffer (``sizes``), its parameter names (``names``) and its group's ``is_expert_param``
    (``expert_params``, to check the name rule against)."""
    megatron_fsdp = fsdp.module
    buffer = megatron_fsdp.param_and_grad_buffer
    layout = dict(sizes={}, names={}, expert_params={})
    for bucket_id, group in enumerate(buffer.parameter_groups):
        if group.main_grad_buffer is None or not group.requires_grad:
            continue
        reducing = megatron_fsdp.grad_reduce_pipeline.get_fsdp_buffer(bucket_id)
        layout["sizes"][bucket_id] = reducing.data_parallel_group.size()
        layout["names"][bucket_id] = [buffer.param_to_name[param] for param in group.params]
        layout["expert_params"][bucket_id] = bool(group.is_expert_param)
    return layout


def bucket_is_expert(layout, bucket_id):
    """Whether a bucket holds expert parameters, by the name rule; asserts the bucket is
    homogeneous and agrees with its group's ``is_expert_param``."""
    kinds = {is_expert(name) for name in layout["names"][bucket_id]}
    assert kinds == {layout["expert_params"][bucket_id]}, (
        bucket_id,
        kinds,
        layout["names"][bucket_id][:4],
    )
    return kinds.pop()


def reduction_counts(calls, step):
    """``(hooks, uses, outer)`` Counters over ``(microbatch, bucket id)`` of one step's recorded
    Megatron-FSDP reductions ``(step, microbatch, bucket ids, outer[, per_use])``: the gradient
    hooks' reductions, the per-use ones, and those that also reduced over DP-Outer."""
    hooks, uses, outer = (collections.Counter() for _ in range(3))
    for call in calls:
        if call[0] != step:
            continue
        per_use = len(call) > 4 and call[4]
        for bucket_id in call[2]:
            (uses if per_use else hooks)[(call[1], bucket_id)] += 1
            if call[3]:
                outer[(call[1], bucket_id)] += 1
    return hooks, uses, outer


def reduction_problems(record, *, dense, expert, world, ep, instances, steps, rows=ROWS):
    """The recorded Megatron-FSDP reductions of an accumulate-mode arm (``record``: ``calls``,
    ``buckets`` = bucket id -> DP-Shard size, ``names``, ``expert_params``; ``dense`` /
    ``expert``: its strategies, ``expert`` ``None`` for the dense one) against the expectation
    from the layout and the configured strategies, every step: every bucket's DP-Shard size is
    the layout's (``expected_dp_shard``); its gradient hooks reduce it ``[1, 1, 1]`` over the
    microbatches when its strategy shards gradients over more than one rank
    (``reduced_every_microbatch``), else ``[0, 0, 1]``; over DP-Outer ``[0, 0, 1]``; no per-use
    reduction. Returns the problems, ``(kind, step, bucket id, ...)`` with kind ``"DP-Shard
    size"``, ``"DP-Shard"``, ``"DP-Outer"`` or ``"per use"``."""
    problems = []
    last = [0] * (rows - 1) + [1]
    for step in range(steps):
        hooks, uses, outer = reduction_counts(record["calls"], step)
        for bucket_id, shard_size in record["buckets"].items():
            expert_bucket = bucket_is_expert(record, bucket_id)
            strategy = (expert or dense) if expert_bucket else dense
            layout_size = expected_dp_shard(world, ep, instances, expert_bucket)
            if shard_size != layout_size:
                problems.append(("DP-Shard size", step, bucket_id, shard_size, layout_size))
            every = reduced_every_microbatch(strategy, layout_size)
            expected = [1] * rows if every else last
            got = [hooks[(m, bucket_id)] for m in range(rows)]
            if got != expected:
                problems.append(("DP-Shard", step, bucket_id, strategy, shard_size, got, expected))
            got = [outer[(m, bucket_id)] for m in range(rows)]
            if got != last:
                problems.append(("DP-Outer", step, bucket_id, got))
            got = [uses[(m, bucket_id)] for m in range(rows)]
            if any(got):
                problems.append(("per use", step, bucket_id, got))
    return problems


def observed_once_per_step(record, steps, rows=ROWS):
    """The parameter names (``record["names"]``: bucket id -> names) of every bucket that, in
    each of ``steps`` steps, was reduced by its gradient hook only in the last microbatch, never
    per use, and over DP-Outer only in the last microbatch."""
    last = [0] * (rows - 1) + [1]
    counts = [reduction_counts(record["calls"], step) for step in range(steps)]
    found = set()
    for bucket_id, names in record["names"].items():
        if all(
            [hooks[(m, bucket_id)] for m in range(rows)] == last
            and not any(uses[(m, bucket_id)] for m in range(rows))
            and [outer[(m, bucket_id)] for m in range(rows)] == last
            for hooks, uses, outer in counts
        ):
            found.update(names)
    return found


def assert_per_backward_mutation(report, mutant_grads, arm_grads, fused, once):
    """A mutation of the per-backward reduction (accumulate mode off, a dropped per-use
    reduction, DP-Outer on every microbatch) must fail the gate on every fused core weight whose
    bucket is reduced every microbatch. It cannot reach one reduced once per step (``once``),
    whose gradient must equal the same-config arm's exactly. Returns those excluded names."""
    per_backward = [name for name in fused if name not in once]
    excluded = [name for name in fused if name in once]
    assert per_backward, (fused[:8], sorted(once)[:8])
    passing = [name for name in per_backward if report[name]["ok"]]
    assert not passing, mbs.worst(report, passing)
    reached = [name for name in excluded if not torch.equal(mutant_grads[name], arm_grads[name])]
    assert not reached, ("the mutation changed weights reduced once per step", reached[:8])
    return excluded
