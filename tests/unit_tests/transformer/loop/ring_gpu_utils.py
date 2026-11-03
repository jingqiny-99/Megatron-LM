# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU helpers of the loop ring tests (``loop_schedule='ring'``): a tiny DSv4 Huginn loop with the
parity mini's role structure, built as the PP = 1 eager reference and as ring stages from its
weights, wrapped in DDP, and stepped through Megatron's schedules over NCCL.

Model (``tiny_config``): DSV4-TINY widths (hidden 512, 8 heads, MLA under the DSv4 hybrid W / C /
H attention on the unfused path ``dsa_kernel_backend='none'``, the DSA indexer and its sparse loss
on every C layer), TE linears, 8 routed experts top-2 plus a shared expert (sqrtsoftplus router
with expert bias and the sequence aux loss; the prelude's three MoE layers hash-routed), mHC with
4 streams, one MTP depth, Huginn with R = 4 fixed rounds. Pattern: the parity mini's
``WEWECE + HECE x 5`` and MTP ``WE`` (prelude 6, core 16, coda 4 sublayers). Every layout runs at
**EP 1**, so every microbatch's MoE math is the same in every layout.

Layouts (``Layout``): PP 1 x DP W (the eager reference, ``loop_schedule='standard'``) and the
ring at P stages x k core slices per stage (DP = W / P), with ``loop_recompute`` (recompute plan
all 'C') or without (all 'n'). A ring layout's chunks are built as ``get_model`` builds them
(``is_pipeline_first_stage`` / ``is_pipeline_last_stage`` with the chunk's ``vp_stage``); their
weights are loaded from a reference state by global name (``decoder.layers.<i>`` shifted by the
chunk's offset), and each chunk gets its own DDP (fp32 gradient reduction, no distributed
optimizer) with Megatron's ``finalize_model_grads``.

One step (``run_step``) runs this data rank's share of a global batch of ``GLOBAL_BATCH``
sequences of ``SEQ`` tokens (MBS 1; data rank d takes sequences ``d * M .. (d + 1) * M - 1``): PP 1
through ``get_forward_backward_func()``, the ring through ``ring_pp.LoopRingSchedule``. Its
``StepResult`` holds every local parameter's ``main_grad`` after the step (by global name), the
per-sequence loss reports, the head call's loop metrics, the router expert biases after the
step's update and the ring's ``LoopRingStepRecord``.
"""

import contextlib
import dataclasses
import functools
import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.loop.hybrid_loop_utils import VOCAB
from tests.unit_tests.transformer.loop.ring_utils import (
    CODA,
    MINI_MAIN,
    MTP,
    PRELUDE,
    ring_fields,
    ring_layout,
)

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
CUDA = torch.cuda.is_available()
MAIN = MINI_MAIN
SEQ = 256
GLOBAL_BATCH = 16
ROUNDS = 4
SEED = 1234
_RATIOS = {"W": 0, "C": 4, "H": 128}


@dataclasses.dataclass(frozen=True)
class Layout:
    """One parallel layout: ``stages`` = P (1: the eager PP = 1 reference), ``chunks`` = k core
    slices per stage, ``recompute`` (``loop_recompute``: plan all 'C'), ``fusion``
    (``gradient_accumulation_fusion``: TE adds the weight gradients into ``main_grad``)."""

    stages: int = 1
    chunks: int = 1
    recompute: bool = False
    fusion: bool = True

    @property
    def ring(self) -> bool:
        """True for the loop ring (P > 1)."""
        return self.stages > 1

    @property
    def name(self) -> str:
        """``pp1`` or ``pp<P>k<k>_<C|n>`` (``_nofusion`` without gradient accumulation fusion)."""
        if not self.ring:
            base = "pp1"
        else:
            base = f"pp{self.stages}k{self.chunks}_{'C' if self.recompute else 'n'}"
        return base if self.fusion else base + "_nofusion"

    @property
    def data_parallel(self) -> int:
        """The data-parallel size at ``WORLD`` ranks (EP 1, TP 1)."""
        return WORLD // self.stages

    def microbatches(self, global_batch: int = GLOBAL_BATCH) -> int:
        """Microbatches per pipeline per step (MBS 1)."""
        return global_batch // self.data_parallel

    def pattern(self) -> str:
        """The hybrid layer pattern: the mini's main pattern (with P x k ``'|'`` segments on the
        ring) and the MTP part."""
        main = (
            MAIN if not self.ring else ring_layout(MAIN, PRELUDE, CODA, self.stages, self.chunks)[0]
        )
        return f"{main}/{MTP}"


def ring_layouts(stages=(2, 4), chunks=(1, 2), recompute=(False, True)) -> List[Layout]:
    """The ring layouts that fit ``WORLD`` ranks (P divides W) and the global batch (M >= P,
    M a multiple of P)."""
    layouts = []
    for p in stages:
        for k in chunks:
            for r in recompute:
                layout = Layout(stages=p, chunks=k, recompute=r)
                if WORLD % p or layout.microbatches() < p or layout.microbatches() % p:
                    continue
                layouts.append(layout)
    return layouts


def tiny_config(dtype: torch.dtype, layout: Layout, **overrides):
    """The tiny DSv4 loop config of ``layout`` in ``dtype`` (module docstring)."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    symbols = MAIN + MTP
    fields = dict(
        num_layers=len(MAIN),
        hidden_size=512,
        num_attention_heads=8,
        multi_latent_attention=True,
        q_lora_rank=192,
        kv_lora_rank=64,
        qk_head_dim=16,
        qk_pos_emb_head_dim=8,
        v_head_dim=16,
        rope_type="rope",
        rotary_base=10000,
        rotary_percent=1.0,
        normalization="RMSNorm",
        experimental_attention_variant="dsv4_hybrid",
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=32,
        dsa_indexer_loss_coeff=0.01,
        dsa_indexer_use_sparse_loss=True,
        csa_window_size=32,
        csa_compress_ratios=[_RATIOS.get(symbol, 0) for symbol in symbols],
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
        moe_num_hash_layers=3,
        hash_moe_vocab_size=VOCAB,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=5,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=layout.fusion,
        deallocate_pipeline_outputs=True,
        is_hybrid_model=True,
        loop_architecture="huginn",
        loop_iterations=ROUNDS,
        loop_exit_criterion="fixed",
        loop_prelude_sublayers=PRELUDE,
        loop_coda_sublayers=CODA,
        loop_recompute=layout.recompute,
        calculate_per_token_loss=True,
        deterministic_mode=True,
        attention_backend=AttnBackend.unfused,
        expert_model_parallel_size=1,
    )
    if layout.ring:
        fields.update(ring_fields(layout.stages, layout.chunks, pipeline_dtype=dtype))
    fields.update(overrides)
    return MLATransformerConfig(**fields)


# ---------------------------------------------------------------------------------------------
# Process state
# ---------------------------------------------------------------------------------------------


@contextlib.contextmanager
def parallel_layout(layout: Layout):
    """Model parallelism of ``layout`` (TP 1, EP 1, PP P, VPP k at k > 1) for the scope; the
    model-parallel seed is reset."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=layout.stages,
        virtual_pipeline_model_parallel_size=layout.chunks if layout.chunks > 1 else None,
        expert_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(SEED)
    try:
        yield
    finally:
        Utils.destroy_model_parallel()


@contextlib.contextmanager
def deterministic_algorithms():
    """``torch.use_deterministic_algorithms(True, warn_only=True)`` for the scope."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(False)


@contextlib.contextmanager
def full_precision_matmuls():
    """fp32 GEMMs without TF32 (``NVIDIA_TF32_OVERRIDE=0``, read by cuBLAS when its first handle
    is made, and ``allow_tf32 = False``), the previous settings restored on exit."""
    previous_env = os.environ.get("NVIDIA_TF32_OVERRIDE")
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
        if previous_env is None:
            os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
        else:
            os.environ["NVIDIA_TF32_OVERRIDE"] = previous_env


@contextlib.contextmanager
def hadamard_fallback(dtype: torch.dtype):
    """The DSv4 indexer's ``rotate_activation`` runs the bf16-only ``fast_hadamard_transform``
    kernel. An fp32 model, or a container without the package, gets a scaled identity instead
    (the same in every layout of a comparison)."""
    # pylint: disable=import-outside-toplevel
    from unittest.mock import patch

    try:
        from fast_hadamard_transform import hadamard_transform  # noqa: F401

        native = dtype == torch.bfloat16
    except ImportError:
        native = False
    if native:
        yield
        return
    with patch(
        "megatron.core.transformer.experimental_attention_variant.csa.rotate_activation",
        lambda x: x * (x.size(-1) ** -0.5),
    ):
        yield


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------


def global_name(name: str, offset: int) -> str:
    """The PP = 1 name of a chunk's state entry (EP 1: only the sublayer index moves)."""
    if offset and name.startswith("decoder.layers."):
        _, _, index, rest = name.split(".", 3)
        return f"decoder.layers.{int(index) + offset}.{rest}"
    return name


def chunk_offset(model) -> int:
    """The global index of a model's first decoder sublayer (0 at PP = 1)."""
    ring = getattr(model.decoder, "loop_ring", None)
    return ring.offset if ring is not None else 0


def build_chunks(config, layout: Layout, *, seed: int = SEED) -> list:
    """This rank's model chunks of ``layout`` (one at k = 1), seeded, on the GPU."""
    # pylint: disable=import-outside-toplevel
    from megatron.core import parallel_state as ps
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.loop.hybrid_specs import loop_hybrid_dsv4_stack_spec
    from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker

    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    set_moe_metrics_tracker(MoEMetricsTracker())
    chunks = []
    for index in range(layout.chunks):
        vp_stage = index if layout.chunks > 1 else None
        chunks.append(
            HybridModel(
                config=config,
                hybrid_stack_spec=loop_hybrid_dsv4_stack_spec(config),
                vocab_size=VOCAB,
                max_sequence_length=1024,
                hybrid_layer_pattern=layout.pattern(),
                position_embedding_type="rope",
                pre_process=ps.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage),
                post_process=ps.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage),
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
                vp_stage=vp_stage,
            ).cuda()
        )
    return chunks


def state_snapshot(chunks: Sequence) -> Dict[str, object]:
    """The chunks' state by global name (tensors cloned): weights and persistent buffers."""
    state = {}
    for model in chunks:
        offset = chunk_offset(model)
        for name, value in model.state_dict().items():
            key = global_name(name, offset)
            if key not in state:
                state[key] = value.detach().clone() if torch.is_tensor(value) else value
    return state


def load_state(chunks: Sequence, state: Dict[str, object]) -> None:
    """Load ``state`` (by global name, ``state_snapshot``) into every chunk, strictly."""
    for model in chunks:
        offset = chunk_offset(model)
        mapped = {name: state[global_name(name, offset)] for name in model.state_dict()}
        model.load_state_dict(mapped, strict=True)


def wrap_ddp(chunks: Sequence, config, *, distributed_optimizer: bool = False) -> list:
    """Each chunk in its own DDP (fp32 gradient reduction; bucketing on the first chunk only, as
    ``get_model``), and Megatron's ``finalize_model_grads`` on the model config."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
        finalize_model_grads,
    )

    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=False,
        use_distributed_optimizer=distributed_optimizer,
        grad_reduce_in_fp32=True,
    )
    wrapped = [
        DistributedDataParallel(config, ddp_config, model, disable_bucketing=index > 0)
        for index, model in enumerate(chunks)
    ]
    config.finalize_model_grads_func = finalize_model_grads
    return wrapped


def core_te_modules(model) -> List[tuple]:
    """``(local name, module)`` of every TE module (it tracks ``is_first_microbatch``) in a
    chunk's core slice."""
    decoder = model.decoder
    ring = getattr(decoder, "loop_ring", None)
    roles = ring.local if ring is not None else decoder.loop_roles
    found = []
    for index in range(roles.core_start, roles.core_end):
        for name, module in decoder.layers[index].named_modules():
            if hasattr(module, "is_first_microbatch"):
                found.append(
                    (
                        f"decoder.layers.{index}.{name}" if name else f"decoder.layers.{index}",
                        module,
                    )
                )
    return found


def edge_te_modules(model) -> List[tuple]:
    """``(local name, module)`` of every TE module of a chunk's prelude and coda sublayers."""
    decoder = model.decoder
    ring = getattr(decoder, "loop_ring", None)
    roles = ring.local if ring is not None else decoder.loop_roles
    found = []
    for index, layer in enumerate(decoder.layers):
        if roles.core_start <= index < roles.core_end:
            continue
        for name, module in layer.named_modules():
            if hasattr(module, "is_first_microbatch"):
                found.append((f"decoder.layers.{index}.{name}", module))
    return found


# ---------------------------------------------------------------------------------------------
# Data and one step
# ---------------------------------------------------------------------------------------------


def global_batch(seed: int = 11, size: int = GLOBAL_BATCH):
    """``(tokens, labels, loss_mask)`` of a global batch of ``size`` sequences (CPU, seeded; the
    loss mask has about 10 % zeros)."""
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, VOCAB, (size, SEQ), generator=generator)
    labels = torch.randint(0, VOCAB, (size, SEQ), generator=generator)
    loss_mask = (torch.rand(size, SEQ, generator=generator) > 0.1).float()
    return tokens, labels, loss_mask


def rank_batches(seed: int = 11, size: int = GLOBAL_BATCH) -> List[dict]:
    """This data rank's microbatches (MBS 1) of the global batch, in order; each carries its
    global sequence index."""
    # pylint: disable=import-outside-toplevel
    from megatron.core import parallel_state as ps

    tokens, labels, loss_mask = global_batch(seed, size)
    rank, ranks = ps.get_data_parallel_rank(), ps.get_data_parallel_world_size()
    per_rank = size // ranks
    device = torch.device("cuda", torch.cuda.current_device())
    position_ids = torch.arange(SEQ, device=device)[None]
    return [
        dict(
            index=index,
            tokens=tokens[index : index + 1].to(device),
            position_ids=position_ids,
            labels=labels[index : index + 1].to(device),
            loss_mask=loss_mask[index : index + 1].to(device),
        )
        for index in range(rank * per_rank, (rank + 1) * per_rank)
    ]


def _loss_func(loss_mask, metrics, index, output_tensor):
    """``pretrain_hybrid.loss_func``'s per-token reduction; the report carries the sequence
    index and the head call's loop metrics."""
    losses = output_tensor.view(-1).float()
    mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * mask)
    num_tokens = mask.sum().clone().detach().to(torch.int)
    report = {
        "index": index,
        "lm loss": torch.stack((loss.clone().detach(), num_tokens.float())),
        "metrics": {name: float(value) for name, value in metrics.items()},
    }
    return loss, num_tokens, report


def forward_step(data_iterator, model):
    """``pretrain_hybrid.forward_step`` for these batches: a batch only on the data-loading chunks
    (tokens on the first, everything on the last, nothing in between); the loss closure captures
    the call's loop metrics."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.utils import get_attr_wrapped_model

    hybrid = get_attr_wrapped_model(model, "decoder", return_model_obj=True)
    batch = next(data_iterator) if (hybrid.pre_process or hybrid.post_process) else {}
    if hybrid.pre_process and not hybrid.post_process:
        batch = {"tokens": batch["tokens"], "position_ids": batch["position_ids"]}
    output = model(
        batch.get("tokens"),
        batch.get("position_ids"),
        None,
        labels=batch.get("labels"),
        loss_mask=batch.get("loss_mask"),
    )
    return output, functools.partial(
        _loss_func, batch.get("loss_mask"), hybrid.decoder.loop_metrics, batch.get("index")
    )


@dataclasses.dataclass
class StepResult:
    """What one step left on this rank (``run_step``)."""

    grads: Dict[str, torch.Tensor]  # global name -> main_grad after the step ('#copy': 2nd holder)
    dtypes: Dict[str, torch.dtype]  # global name -> parameter dtype
    fused: set  # global names whose weight gradient TE added into main_grad
    reports: Dict[int, tuple]  # sequence index -> (loss sum, tokens), on the loss stage
    metrics: Dict[int, dict]  # sequence index -> the head call's loop metrics
    expert_bias: Dict[str, torch.Tensor]  # global name -> router expert bias after the step
    record: object = None  # the ring's LoopRingStepRecord


def run_step(
    ddp_chunks: Sequence, layout: Layout, batches: List[dict], schedule=None, collect: bool = True
) -> StepResult:
    """One training step of ``batches`` (forward, backward, finalize; no optimizer step): PP 1
    through ``get_forward_backward_func()``, the ring through ``schedule`` (a fresh
    ``LoopRingSchedule`` by default). ``collect=False`` skips copying the gradients and biases."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.transformer.loop.ring_pp import LoopRingSchedule

    for chunk in ddp_chunks:
        chunk.zero_grad_buffer()
    models = [chunk.module for chunk in ddp_chunks]
    if layout.ring:
        schedule = schedule or LoopRingSchedule()
        iterators = [
            iter(batches) if (model.pre_process or model.post_process) else None for model in models
        ]
        outputs = schedule(
            forward_step_func=forward_step,
            data_iterator=iterators,
            model=list(ddp_chunks),
            num_microbatches=len(batches),
            seq_length=SEQ,
            micro_batch_size=1,
        )
        record = schedule.last_step
    else:
        outputs = get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=iter(batches),
            model=list(ddp_chunks),
            num_microbatches=len(batches),
            seq_length=SEQ,
            micro_batch_size=1,
            forward_only=False,
        )
        record = None
    torch.cuda.synchronize()
    grads, dtypes, fused, biases = {}, {}, set(), {}
    for model in models if collect else ():
        offset = chunk_offset(model)
        for name, param in model.named_parameters():
            key = global_name(name, offset)
            if key in grads:
                key += "#copy"
            grads[key] = param.main_grad.detach().float().clone()
            dtypes[key] = param.dtype
            if getattr(param, "grad_added_to_main_grad", False):
                fused.add(key)
        for name, buffer in model.named_buffers():
            if name.endswith("expert_bias"):
                biases[global_name(name, offset)] = buffer.detach().float().clone()
    reports, metrics = {}, {}
    for report in outputs or []:
        reports[report["index"]] = tuple(float(value) for value in report["lm loss"].tolist())
        metrics[report["index"]] = dict(report["metrics"])
    return StepResult(grads, dtypes, fused, reports, metrics, biases, record)


def run_with_precision(compute):
    """``(dtype, fp32 probe error or None, compute(dtype))``: ``compute`` (a function of the
    parameter dtype, run inside ``hadamard_fallback``) in fp32, or in bf16 when the fp32 run raises
    on any rank. ``LOOP_RING_GPU_PRECISION`` = ``fp32`` / ``bf16`` forces one (no fallback)."""
    forced = os.environ.get("LOOP_RING_GPU_PRECISION", "auto")
    if forced == "bf16":
        with hadamard_fallback(torch.bfloat16):
            return torch.bfloat16, None, compute(torch.bfloat16)
    error, result = None, None
    try:
        with hadamard_fallback(torch.float32):
            result = compute(torch.float32)
    except Exception as exc:  # pylint: disable=broad-except
        if forced == "fp32":
            raise
        error = f"{type(exc).__name__}: {exc}"
        print(f"loop ring gpu: the fp32 run raised on rank {dist.get_rank()}: {error}")
    if all(gather(error is None)):
        return torch.float32, None, result
    with hadamard_fallback(torch.bfloat16):
        return (
            torch.bfloat16,
            error or "the fp32 run raised on another rank",
            compute(torch.bfloat16),
        )


# ---------------------------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------------------------


def gather(value) -> list:
    """``all_gather_object`` over the default group (every rank's ``value``)."""
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def merged(dicts: Sequence[dict]) -> dict:
    """The union of several ranks' dicts (later ranks do not overwrite earlier ones)."""
    out = {}
    for entry in dicts:
        for key, value in entry.items():
            out.setdefault(key, value)
    return out


def grad_stats(
    grads: Dict[str, torch.Tensor], reference: Dict[str, torch.Tensor]
) -> Dict[str, dict]:
    """Per local parameter (global name, '#copy' kept): the cosine with the reference gradient,
    the relative norm difference ``|g - r| / |r|``, the reference norm and bitwise equality."""
    stats = {}
    for key, grad in grads.items():
        ref = reference[key.split("#", 1)[0]].to(grad.device)
        g, r = grad.double().flatten(), ref.double().flatten()
        ref_norm, diff_norm, grad_norm = r.norm().item(), (g - r).norm().item(), g.norm().item()
        if ref_norm > 0 and grad_norm > 0:
            cosine = float(torch.dot(g, r).item() / (grad_norm * ref_norm))
        else:
            cosine = 1.0 if diff_norm == 0 else 0.0
        relative = (
            diff_norm / ref_norm if ref_norm > 0 else (0.0 if diff_norm == 0 else float("inf"))
        )
        stats[key] = dict(
            cosine=cosine,
            relative=relative,
            ref_norm=ref_norm,
            bitwise=bool(torch.equal(grad, ref)),
        )
    return stats
