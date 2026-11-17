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

Every helper that communicates is collective: all ranks call it in the same order.
"""

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
    ``on_microbatch(index)`` runs before each microbatch's forward."""
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker

    model.config.finalize_model_grads_func = finalize_model_grads
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
# Step 2 runs on weights updated from fp32-reassociated step-1 gradients.
STEP2_LOSS_RELATIVE = 1e-5


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


def assert_step_losses_match(step_index, step, reference, name, step2_relative=None):
    """Step 1: ``assert_losses_match``. Later steps run on weights updated from gradients that
    differ by fp32 reassociation: relative difference <= ``step2_relative`` (default
    ``STEP2_LOSS_RELATIVE``)."""
    if step_index == 0:
        assert_losses_match(step, reference, name)
        return
    bound = STEP2_LOSS_RELATIVE if step2_relative is None else step2_relative
    rel = mbs.relative_loss_differences(step.losses, reference.losses)
    print(f"[{name}] losses {step.losses} rel {rel}", flush=True)
    assert all(value <= bound for value in rel.values()), rel


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
