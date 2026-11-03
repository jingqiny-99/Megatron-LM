# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU emulation of the loop ring (``loop_schedule='ring'``) against the PP = 1 eager loop.

Local-layer ``HybridModel`` loops (SDPA attention; dense MLPs, or real ``MoELayer`` s at EP 1 with
the allgather dispatcher, natural sigmoid routing with expert bias and the sequence aux loss, the
prelude's three MoE layers hash-routed), mHC 4, one MTP depth, R = 4, on the parity mini's role
structure (prelude 3 blocks, core 8 blocks, coda 2 blocks).

The PP = 1 reference runs ``M`` microbatches of forward + backward through Megatron's own
``schedules.forward_step`` / ``backward_step`` (the training loss function's reduction, the
auxiliary loss scales). The ring builds the P x k stage chunks of the same model in one process
(a P-rank stub pipeline group per stage, ``vp_stage`` j at k > 1, weights copied from the
reference by global layer index), wraps each in R round views (``ring_pp.LoopRingRoundView``)
and runs every pipeline rank's op list in Megatron's interleaved order: the order
``schedules.get_schedule_table`` and ``get_pp_rank_microbatches`` give for V = R x k model chunks
(``convert_schedule_table_to_order``: warmup forwards, 1F1B pairs, cooldown backwards). Each op
is Megatron's ``forward_step`` / ``backward_step`` on the view; P2P is emulated: a stage's output
reaches the next virtual stage as a detached clone requiring grad (a received buffer) and is then
pseudo-deallocated (``deallocate_output_tensor``; the backward runs ``custom_backward``), and the
received tensor's gradient goes back. A rank runs its next op as soon as its inputs exist, so the
global interleaving is one Megatron could produce; the results depend only on each rank's own
order (every parameter, store and counter belongs to one rank).

``M < N`` (fewer microbatches than the microbatch group size, which defaults to P) is below the
real interleaved schedule's minimum; the emulation still runs Megatron's table and warmup count
for it (all forwards, then all backwards).

``run_ring(..., mutation=...)`` takes test hooks that break the emulation's P2P (feed a view another
view's input, drop a backward) for the mutation tests.
"""

import contextlib
import copy
import functools
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple

import torch

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel import schedules
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.loop.guards import resolve_loop_ring_recompute_plan
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.loop.ring_pp import LoopRingRoundView, ring_sample_counts
from megatron.core.transformer.moe.moe_logging import (
    MoEMetricsTracker,
    get_moe_metrics_tracker,
    set_moe_metrics_tracker,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    MoETransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.utils import get_attr_wrapped_model
from tests.unit_tests.transformer.loop import hybrid_loop_utils as hlu
from tests.unit_tests.transformer.loop.ring_utils import (
    StageGroup,
    ring_fields,
    ring_layout,
    stage_group_ranks,
)

ROUNDS = 4
BLOCKS = 13  # prelude 3 + core 8 + coda 2 blocks: P x k up to 8 core slices
PRELUDE, CODA = 6, 4
EXPERTS = 4
MHC = dict(enable_mhc_connections=True, mhc_num_residual_streams=4, mhc_sinkhorn_iterations=5)
MOE = dict(
    num_moe_experts=EXPERTS,
    moe_ffn_hidden_size=96,
    moe_router_topk=2,
    moe_token_dispatcher_type="allgather",
    moe_router_load_balancing_type="seq_aux_loss",
    moe_aux_loss_coeff=1e-4,
    moe_router_enable_expert_bias=True,
    moe_router_score_function="sigmoid",
    moe_router_dtype="fp32",
    moe_num_hash_layers=3,
    hash_moe_vocab_size=hlu.VOCAB,
)
KINDS = ("dense", "moe")


def patterns(kind: str) -> Tuple[str, str]:
    """``(main pattern, MTP pattern)`` of a model kind."""
    block = "*-" if kind == "dense" else "*E"
    return block * BLOCKS, block


def plan_codes(plan: str, stages: int, views: int) -> Tuple[str, ...]:
    """The per-stage recompute codes of a named plan: ``all-C``, ``all-n``, ``mixed`` ('C' where
    stage + view is even: the prelude checkpointed) or ``mixed-inv`` (the complement)."""
    if plan == "all-C":
        return tuple("C" * views for _ in range(stages))
    if plan == "all-n":
        return tuple("n" * views for _ in range(stages))
    if plan in ("mixed", "mixed-inv"):
        parity = 0 if plan == "mixed" else 1
        return tuple(
            "".join("C" if (stage + view) % 2 == parity else "n" for view in range(views))
            for stage in range(stages)
        )
    raise ValueError(plan)


def _moe_layer_spec():
    local = get_gpt_layer_local_submodules(num_experts=EXPERTS, normalization="RMSNorm")
    return ModuleSpec(
        module=MoETransformerLayer,
        submodules=TransformerLayerSubmodules(
            pre_mlp_layernorm=local.pre_mlp_layernorm, mlp=local.mlp, mlp_bda=get_bias_dropout_add
        ),
    )


def model_config(kind: str, **overrides):
    """The loop config of a model kind (``overrides``: e.g. the ring fields)."""
    main, _ = patterns(kind)
    fields = dict(loop_iterations=ROUNDS, mtp_num_layers=1, mtp_loss_scaling_factor=0.1, **MHC)
    if kind == "moe":
        fields.update(MOE)
    fields.update(overrides)
    return hlu.hybrid_loop_config(main, PRELUDE, CODA, **fields)


def build_model(
    kind, config, pattern, *, pre_process=True, post_process=True, pg_collection=None, vp_stage=None
):
    """A local-layer loop ``HybridModel`` (seeded; the ring copies its weights anyway)."""
    torch.manual_seed(1234)
    spec = hlu.local_model_spec(
        LoopHybridStack,
        hyper_connections=True,
        moe_layer=_moe_layer_spec() if kind == "moe" else None,
    )
    model = HybridModel(
        config=config,
        hybrid_stack_spec=spec,
        vocab_size=hlu.VOCAB,
        max_sequence_length=hlu.SEQ,
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
        pre_process=pre_process,
        post_process=post_process,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )
    return model.to(hlu.device())


def make_batches(microbatches: int, seed: int = 7) -> List[dict]:
    """``microbatches`` batches of random tokens / labels, a loss mask with some zeros."""
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(microbatches):
        tokens = torch.randint(0, hlu.VOCAB, (hlu.BATCH, hlu.SEQ), generator=generator)
        labels = torch.randint(0, hlu.VOCAB, (hlu.BATCH, hlu.SEQ), generator=generator)
        loss_mask = torch.ones(hlu.BATCH, hlu.SEQ)
        loss_mask[-1, -3:] = 0.0
        position_ids = torch.arange(hlu.SEQ).repeat(hlu.BATCH, 1)
        batches.append(
            {
                name: value.to(hlu.device())
                for name, value in dict(
                    tokens=tokens, labels=labels, loss_mask=loss_mask, position_ids=position_ids
                ).items()
            }
        )
    return batches


def _loss_func(loss_mask, metrics, output_tensor):
    """``pretrain_hybrid.loss_func``'s reduction (sum of masked per-token losses, token count)."""
    losses = output_tensor.view(-1).float()
    mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * mask)
    num_tokens = mask.sum().clone().detach().to(torch.int)
    report = {"per_token": output_tensor.detach().clone(), "metrics": metrics}
    return loss, num_tokens, report


def _forward_step_func(batch, model):
    """``pretrain_hybrid.forward_step``: the model on the batch fields this chunk loads (None on
    the other chunks), the loop metrics captured with the loss closure."""
    batch = batch or {}
    output = model(
        batch.get("tokens"),
        batch.get("position_ids"),
        None,
        labels=batch.get("labels"),
        loss_mask=batch.get("loss_mask"),
    )
    decoder = get_attr_wrapped_model(model, "decoder")
    return output, functools.partial(_loss_func, batch.get("loss_mask"), decoder.loop_metrics)


def _step_config(config, virtual_size=None):
    """The schedule's config (a shallow copy: the ring's V, pseudo-deallocated outputs)."""
    step = copy.copy(config)
    step.virtual_pipeline_model_parallel_size = virtual_size
    step.deallocate_pipeline_outputs = True
    return step


@dataclass
class RunResult:
    """What one run produced, keyed by global names."""

    losses: List[torch.Tensor]  # per microbatch: the per-token loss
    grads: Dict[str, Optional[torch.Tensor]]
    head_metrics: List[dict]  # per microbatch: the loss closure's loop metrics
    token_counts: Dict[str, torch.Tensor]  # router expert-bias token counts
    moe_tracker: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]
    extra: dict = field(default_factory=dict)


def _moe_snapshot():
    return {
        name: (
            entry.values.detach().clone(),
            None if entry.round_values is None else entry.round_values.detach().clone(),
        )
        for name, entry in get_moe_metrics_tracker().metrics.items()
    }


def _global_name(name: str, offset: int) -> str:
    """A chunk's state name with its decoder sublayer index made global (``+ offset``)."""
    if name.startswith("decoder.layers."):
        _, _, index, rest = name.split(".", 3)
        return f"decoder.layers.{int(index) + offset}.{rest}"
    return name


def _token_counts(model, offset=0):
    counts = {}
    for name, buffer in model.named_buffers():
        if name.endswith("local_tokens_per_expert"):
            counts[_global_name(name, offset)] = buffer.detach().clone()
    return counts


def run_reference(
    kind: str, batches, *, forward_only: bool = False
) -> Tuple[RunResult, dict, HybridModel]:
    """The PP = 1 eager loop on ``batches``: per-microbatch forward + backward through
    Megatron's ``forward_step`` / ``backward_step`` (``forward_only``: forwards under
    ``torch.no_grad``, as evaluation runs). Returns the result, the model's state dict (the
    ring's weights) and the model."""
    main, mtp = patterns(kind)
    config = model_config(kind)
    set_moe_metrics_tracker(MoEMetricsTracker())
    model = build_model(kind, config, main + "/" + mtp)
    # Tensors cloned; other entries (the local linears' ``_extra_state`` None) as they are.
    state = {
        name: value.detach().clone() if torch.is_tensor(value) else value
        for name, value in model.state_dict().items()
    }
    step_config = _step_config(config)
    reports = []
    for microbatch, batch in enumerate(batches):
        with torch.no_grad() if forward_only else contextlib.nullcontext():
            output, _ = schedules.forward_step(
                _forward_step_func,
                batch,
                model,
                len(batches),
                None,
                reports,
                step_config,
                1,
                current_microbatch=microbatch,
                is_last_stage=True,
            )
        if not forward_only:
            schedules.backward_step(None, output, None, step_config)
    result = RunResult(
        losses=[report["per_token"] for report in reports],
        grads={
            name: None if param.grad is None else param.grad.detach().clone()
            for name, param in model.named_parameters()
        },
        head_metrics=[report["metrics"] for report in reports],
        token_counts=_token_counts(model),
        moe_tracker=_moe_snapshot(),
    )
    return result, state, model


@dataclass
class RingMutation:
    """Test hooks that break the emulated P2P (the mutation tests).

    ``feed(stage, view, microbatch, received, inputs)`` returns the tensor a forward uses as its
    input (``inputs``: every input received so far, by ``(stage, view, microbatch)``);
    ``drop_backward(stage, view, microbatch)`` skips that backward (its input gets a zero
    gradient)."""

    feed: Optional[Callable] = None
    drop_backward: Optional[Callable] = None


def megatron_rank_ops(stages, stage, views, microbatches, group_size, forward_only=False):
    """Pipeline rank ``stage``'s op list ``[(kind, view, microbatch)]`` (kind 'F' / 'B') in
    Megatron's interleaved order (``convert_schedule_table_to_order`` with microbatch ids;
    ``forward_only``: the forwards in table order)."""
    table = schedules.get_schedule_table(microbatches, views, group_size)
    communicator = SimpleNamespace(
        pp_group=StageGroup(stage, stages), virtual_pipeline_model_parallel_size=views
    )
    _, _, warmup, remaining = schedules.get_pp_rank_microbatches(
        microbatches, views, group_size, forward_only=forward_only, p2p_communicator=communicator
    )
    forward = [("F", view, microbatch) for microbatch, view in table]
    if forward_only:
        assert warmup == len(forward) and remaining == 0
        return forward
    backward = [("B", views - 1 - view, microbatch) for microbatch, view in table]
    order = schedules.convert_schedule_table_to_order(warmup, views, table)
    ops = forward[:warmup]
    for index in range(remaining):
        ops += [forward[warmup + index], backward[index]]
    ops += backward[remaining:]
    # Cross-check against Megatron's own order (1-based chunk ids, negative for backward).
    assert [view + 1 if kind == "F" else -(view + 1) for kind, view, _ in ops] == order
    return ops


def group_size_for(stages, microbatches):
    """The microbatch group size N: P when the real schedule accepts it, else M when that is
    valid (N in [P, M], the last group 0 or >= P), else P (M < P: emulation only)."""
    if microbatches >= stages and microbatches % stages == 0:
        return stages
    if microbatches >= stages:
        return microbatches
    return stages


@dataclass
class Ring:
    """The ring's chunks of one layout, built from the reference weights."""

    stages: int
    chunks: int
    codes: Tuple[str, ...]
    segments: list
    models: list  # by segment index
    views: list  # [stage][view]
    config: object
    step_config: object
    built: list = field(default_factory=list)  # built_flags of each model, at build time

    def chunk_model(self, stage, chunk):
        """The model of physical chunk (stage, chunk)."""
        return self.models[chunk * self.stages + stage]

    def chunk_codes(self, stage, chunk):
        """The recompute codes of chunk (stage, chunk), one per round."""
        rounds = ROUNDS
        return tuple(self.codes[stage][r * self.chunks + chunk] for r in range(rounds))


def _stage_pg_collection(stage, stages):
    collection = copy.copy(ProcessGroupCollection.use_mpu_process_groups())
    collection.pp = StageGroup(stage, stages)
    collection.embd = None
    return collection


def build_ring(kind, stages, chunks, plan, state) -> Ring:
    """The ring chunks of ``kind`` at P = ``stages``, k = ``chunks`` with recompute ``plan``,
    weights from the reference ``state`` (by global name; the MTP embedding copy gets the
    first stage's), and their round views."""
    main, mtp = patterns(kind)
    views = ROUNDS * chunks
    codes = plan_codes(plan, stages, views)
    recompute = any("C" in group for group in codes)
    config = model_config(
        kind,
        loop_recompute=recompute,
        loop_ring_recompute_plan="/".join(codes),
        **ring_fields(stages, chunks),
    )
    assert resolve_loop_ring_recompute_plan(config) == codes
    piped, segments = ring_layout(main, PRELUDE, CODA, stages, chunks)
    last = len(segments) - 1
    models = []
    with stage_group_ranks():
        for segment in segments:
            model = build_model(
                kind,
                config,
                piped + "/" + mtp,
                pre_process=segment["index"] == 0,
                post_process=segment["index"] == last,
                pg_collection=_stage_pg_collection(segment["stage"], stages),
                vp_stage=segment["chunk"] if chunks > 1 else None,
            )
            mapped = {
                name: state[_global_name(name, segment["offset"])] for name in model.state_dict()
            }
            model.load_state_dict(mapped, strict=True)
            models.append(model)
    step_config = _step_config(config, virtual_size=views)
    ring = Ring(stages, chunks, codes, segments, models, [], config, step_config)
    ring.built = [built_flags(model) for model in models]
    for stage in range(stages):
        ring.views.append(
            [
                LoopRingRoundView(
                    ring.chunk_model(stage, view % chunks),
                    step_config,
                    round_index=view // chunks,
                    chunk_index=view % chunks,
                    chunks=chunks,
                    checkpoint=codes[stage][view] == "C",
                )
                for view in range(views)
            ]
        )
    return ring


def ring_model_work(ring: Ring):
    """The model's per-microbatch FLOPs: the sum of every chunk's ``ring_microbatch_work`` (the
    ring schedule's all-reduce over the pipeline, in one process)."""
    total = [0.0, 0.0, 0.0]
    for stage in range(ring.stages):
        for chunk in range(ring.chunks):
            decoder = ring.chunk_model(stage, chunk).decoder
            work = decoder.ring_microbatch_work(hlu.SEQ, hlu.BATCH, ring.chunk_codes(stage, chunk))
            total = [a + b for a, b in zip(total, work)]
    return tuple(total)


def _batch_for(segment_index, last, batch):
    """The batch fields a chunk loads (``get_batch``): tokens on the first chunk, labels, the
    loss mask and the MTP inputs on the last one, nothing in between."""
    if segment_index == 0:
        return {"tokens": batch["tokens"], "position_ids": batch["position_ids"]}
    if segment_index == last:
        return dict(batch)
    return {}


def run_ring(
    ring: Ring,
    batches,
    *,
    mutation: Optional[RingMutation] = None,
    capture: bool = False,
    forward_only: bool = False,
    observe: Optional[Callable] = None,
) -> RunResult:
    """One step of ``batches`` on the ring in Megatron's interleaved order (module docstring).

    ``capture``: also record every backward call's gradient contribution to every parameter
    (``extra['contributions']``: ``(global name, microbatch, round) -> [grad]``), the inputs of
    ``reference_order_sum``. ``forward_only``: Megatron's forward-only order under
    ``torch.no_grad`` (evaluation). ``observe(stage, kind, view, microbatch)`` runs after every
    op."""
    mutation = mutation or RingMutation()
    stages, chunks = ring.stages, ring.chunks
    views = ROUNDS * chunks
    microbatches = len(batches)
    group_size = group_size_for(stages, microbatches)
    last_segment = stages * chunks - 1
    set_moe_metrics_tracker(MoEMetricsTracker())
    work = ring_model_work(ring)
    for model in ring.models:
        if model.decoder.loop_ring.round_end:
            model.decoder.set_ring_model_work(hlu.SEQ, hlu.BATCH, work)
    for stage_views in ring.views:
        for view in stage_views:
            view.reset_calls()
    ops = [
        megatron_rank_ops(stages, stage, views, microbatches, group_size, forward_only)
        for stage in range(stages)
    ]
    received: Dict[tuple, Optional[torch.Tensor]] = {}  # (stage, view, mb) -> received input
    used_inputs: Dict[tuple, Optional[torch.Tensor]] = {}
    outputs: Dict[tuple, torch.Tensor] = {}
    output_grads: Dict[tuple, torch.Tensor] = {}
    reports = []
    executed = []
    current = {}
    contributions: Dict[tuple, List[torch.Tensor]] = {}
    handles = []
    if capture:
        for segment, model in zip(ring.segments, ring.models):
            for name, param in model.named_parameters():

                def hook(grad, key=_global_name(name, segment["offset"])):
                    _, view, microbatch = current["op"]
                    contributions.setdefault((key, microbatch, view // chunks), []).append(
                        grad.detach().clone()
                    )

                handles.append(param.register_hook(hook))

    def successor(stage, view):
        if stage < stages - 1:
            return stage + 1, view
        return (0, view + 1) if view < views - 1 else None

    def predecessor(stage, view):
        if stage > 0:
            return stage - 1, view
        return (stages - 1, view - 1) if view > 0 else None

    def ready(op, stage):
        kind, view, microbatch = op
        if kind == "F":
            return predecessor(stage, view) is None or (stage, view, microbatch) in received
        if (stage, view, microbatch) not in outputs:
            return False
        return successor(stage, view) is None or (stage, view, microbatch) in output_grads

    def forward(stage, view, microbatch):
        segment = ring.views[stage][view].chunk_index * stages + stage
        batch = _batch_for(segment, last_segment, batches[microbatch])
        is_last = stage == stages - 1 and view == views - 1
        input_tensor = received.get((stage, view, microbatch))
        if mutation.feed is not None:
            input_tensor = mutation.feed(stage, view, microbatch, input_tensor, used_inputs)
        with torch.no_grad() if forward_only else contextlib.nullcontext():
            output, _ = schedules.forward_step(
                _forward_step_func,
                batch,
                ring.views[stage][view],
                microbatches,
                input_tensor,
                reports,
                ring.step_config,
                1,
                is_first_microbatch=microbatch == 0,
                current_microbatch=microbatch,
                vp_stage=view,
                is_last_stage=is_last,
            )
        target = successor(stage, view)
        if target is not None:
            # P2P: the receiver gets a fresh buffer; the sender pseudo-frees its output.
            buffer = output.detach().clone()
            received[target + (microbatch,)] = buffer if forward_only else buffer.requires_grad_()
            schedules.deallocate_output_tensor(output, True)
        if forward_only:  # released after the forward, as Megatron's forward-only schedule does
            received.pop((stage, view, microbatch), None)
        else:
            used_inputs[(stage, view, microbatch)] = input_tensor
            outputs[(stage, view, microbatch)] = output

    def backward(stage, view, microbatch):
        output = outputs.pop((stage, view, microbatch))
        grad = output_grads.pop((stage, view, microbatch), None)
        input_tensor = used_inputs.pop((stage, view, microbatch))
        received.pop((stage, view, microbatch), None)
        if mutation.drop_backward is not None and mutation.drop_backward(stage, view, microbatch):
            input_grad = None if input_tensor is None else torch.zeros_like(input_tensor)
        else:
            current["op"] = (stage, view, microbatch)
            input_grad = schedules.backward_step(input_tensor, output, grad, ring.step_config)
        source = predecessor(stage, view)
        if source is not None:
            if input_grad is None:
                input_grad = torch.zeros_like(input_tensor)
            output_grads[source + (microbatch,)] = input_grad

    pointers = [0] * stages
    with stage_group_ranks():
        while any(pointer < len(rank_ops) for pointer, rank_ops in zip(pointers, ops)):
            progressed = False
            for stage in range(stages):
                while pointers[stage] < len(ops[stage]) and ready(
                    ops[stage][pointers[stage]], stage
                ):
                    kind, view, microbatch = ops[stage][pointers[stage]]
                    (forward if kind == "F" else backward)(stage, view, microbatch)
                    executed.append((stage, kind, view, microbatch))
                    if observe is not None:
                        observe(stage, kind, view, microbatch)
                    pointers[stage] += 1
                    progressed = True
            if not progressed:
                raise RuntimeError(f"emulated ring deadlocked at {pointers}")
    for handle in handles:
        handle.remove()
    assert not outputs and not output_grads, "every forward had its backward"

    grads: Dict[str, Optional[torch.Tensor]] = {}
    token_counts: Dict[str, torch.Tensor] = {}
    for segment, model in zip(ring.segments, ring.models):
        for name, param in model.named_parameters():
            key = _global_name(name, segment["offset"])
            grad = None if param.grad is None else param.grad.detach().clone()
            if key in grads:  # the word embedding: the first chunk's and the MTP copy's
                previous = grads[key]
                grads[key] = (
                    grad if previous is None else (previous if grad is None else previous + grad)
                )
            else:
                grads[key] = grad
        token_counts.update(_token_counts(model, segment["offset"]))
    chunk_counts = [model.decoder.take_ring_counts() for model in ring.models]
    extra = dict(
        model_work=work,
        recorded_work=[model.decoder.take_ring_work() for model in ring.models],
        analytic_work=[
            ring.chunk_model(s % stages, s // stages).decoder.ring_microbatch_work(
                hlu.SEQ, hlu.BATCH, ring.chunk_codes(s % stages, s // stages)
            )
            for s in range(len(ring.models))
        ],
        sample_counts=ring_sample_counts(chunk_counts),
        chunk_counts=chunk_counts,
        injection_keys=[model.decoder.ring_injection_keys() for model in ring.models],
        call_roles=[model.decoder.call_roles for model in ring.models],
        calls=[[view.calls for view in stage_views] for stage_views in ring.views],
        executed=executed,
        contributions=contributions,
    )
    return RunResult(
        losses=[report["per_token"] for report in reports],
        grads=grads,
        head_metrics=[report["metrics"] for report in reports],
        token_counts=token_counts,
        moe_tracker=_moe_snapshot(),
        extra=extra,
    )


def compare_grads(
    reference: Dict[str, Optional[torch.Tensor]], ring: Dict[str, Optional[torch.Tensor]]
):
    """Per parameter: ``(bitwise equal, cosine, relative norm difference)`` (None grads on both
    sides count as equal; zero gradients as cosine 1)."""
    assert set(reference) == set(ring), sorted(set(reference) ^ set(ring))
    rows = {}
    for name, expected in reference.items():
        actual = ring[name]
        if expected is None or actual is None:
            assert expected is None and actual is None, name
            rows[name] = (True, 1.0, 0.0)
            continue
        expected64, actual64 = expected.double().flatten(), actual.double().flatten()
        norm, actual_norm = expected64.norm().item(), actual64.norm().item()
        diff = (actual64 - expected64).norm().item()
        if norm == 0.0 or actual_norm == 0.0:
            cosine = 1.0 if norm == actual_norm == 0.0 else 0.0
            relative = 0.0 if diff == 0.0 else float("inf")
        else:
            # Exact in float64 (no eps clamp: some gradients here are tiny).
            cosine = torch.dot(expected64, actual64).item() / (norm * actual_norm)
            relative = diff / norm
        rows[name] = (torch.equal(expected, actual), cosine, relative)
    return rows


# The contract's G2 unit gate (fp32): every parameter's gradient against PP = 1.
G2_COSINE, G2_RELATIVE = 0.9999, 1e-3
# M > 1: the gradient difference relative to the magnitude of the summed contributions.
ORDER_RELATIVE = 1e-6


def reference_order_sum(contributions, microbatches: int) -> Dict[str, Optional[torch.Tensor]]:
    """Every parameter's gradient summed from the ring's per-call contributions in the PP = 1
    eager loop's order: within a microbatch the rounds in backward order (round R first; the
    word embedding's MTP copy, used in round R, before its first-stage copy, used in round 1, as
    PP = 1's one embedding receives them), then the microbatches in order."""
    sums = {}
    for name in sorted({key[0] for key in contributions}):
        total = None
        for microbatch in range(microbatches):
            partial = None
            for round_index in reversed(range(ROUNDS)):
                for grad in contributions.get((name, microbatch, round_index), []):
                    partial = grad if partial is None else partial + grad
            if partial is not None:
                total = partial if total is None else total + partial
        sums[name] = total
    return sums


def contribution_magnitude(contributions) -> Dict[str, torch.Tensor]:
    """Per parameter: ``sum |g|`` over its per-call contributions (float64), the scale of the
    rounding error of any order of summing them."""
    magnitude = {}
    for (name, _, _), grads in contributions.items():
        for grad in grads:
            value = grad.double().abs()
            magnitude[name] = value if name not in magnitude else magnitude[name] + value
    return magnitude


def check_ring(
    reference: RunResult,
    result: RunResult,
    ring: Ring,
    microbatches: int,
    *,
    forward_only: bool = False,
) -> List[str]:
    """Every way the ring run differs from the PP = 1 reference (empty: it reproduces it).

    ``forward_only`` (both runs forward-only, under ``torch.no_grad``): no gradient checks;
    backward and recompute FLOPs are 0.

    * Per-token losses bitwise at every M (the forward does not depend on the accumulation).
    * Gradients: the G2 gate for every parameter (cosine >= 0.9999, ||dg|| / ||g|| <= 1e-3);
      M = 1 bitwise; M > 1: the ring's per-call contributions summed in PP = 1's order equal
      PP = 1's gradient bitwise (the two differ only in the order of the same additions) and
      ||dg|| <= 1e-6 * ||sum |contributions| ||.
    * Loop metrics of every head call equal PP = 1's (depth, token layers, forward and backward
      FLOPs per token: the PP sum of the chunks' FLOPs); recompute FLOPs per token = the plan's;
      each chunk's recorded FLOPs = M x its per-microbatch FLOPs.
    * G1: every microbatch ran the core R times on every chunk, the prelude, coda and MTP once.
    * Every injection store empty, every call role cleared, every view called M times, the built
      flags unchanged; MoE expert-bias token counts equal, the aux-loss tracker's per-layer and
      per-(round, layer) sums equal (M = 1) or within 1e-6 (M > 1).
    """
    problems = []
    tokens = hlu.SEQ * hlu.BATCH
    if len(result.losses) != microbatches:
        problems.append(f"{len(result.losses)} losses for {microbatches} microbatches")
    for microbatch, (want, got) in enumerate(zip(reference.losses, result.losses)):
        if not torch.equal(want, got):
            problems.append(f"loss of microbatch {microbatch} differs")
    rows = {} if forward_only else compare_grads(reference.grads, result.grads)
    if forward_only and any(grad is not None for grad in result.grads.values()):
        problems.append("a forward-only run produced gradients")
    for name, (equal, cosine, relative) in rows.items():
        if not (cosine >= G2_COSINE and relative <= G2_RELATIVE):
            problems.append(f"G2 gate: {name} cosine {cosine:.9f} relative {relative:.3e}")
        if microbatches == 1 and not equal:
            problems.append(f"{name}: not bitwise at M = 1 (relative {relative:.3e})")
    if microbatches > 1 and not forward_only:
        contributions = result.extra.get("contributions") or {}
        if not contributions:
            problems.append("M > 1 needs the contributions (run_ring(capture=True))")
        rebuilt = reference_order_sum(contributions, microbatches)
        magnitude = contribution_magnitude(contributions)
        for name, expected in reference.grads.items():
            got = rebuilt.get(name)
            if (expected is None) != (got is None) or (
                expected is not None and not torch.equal(expected, got)
            ):
                problems.append(f"{name}: the ring's contributions in PP = 1 order differ")
            if expected is not None and result.grads[name] is not None:
                diff = (result.grads[name].double() - expected.double()).norm().item()
                scale = magnitude[name].norm().item() if name in magnitude else 0.0
                if diff > ORDER_RELATIVE * scale:
                    problems.append(f"{name}: |dg| {diff:.3e} above 1e-6 x {scale:.3e}")
    forward, backward, recompute = result.extra["model_work"]
    for microbatch, (want, got) in enumerate(zip(reference.head_metrics, result.head_metrics)):
        if list(want) != list(got):
            problems.append(f"loop metric keys {list(got)} != {list(want)}")
            continue
        for key, value in want.items():
            if key == "loop recompute flops per token" and not forward_only:
                value = torch.as_tensor(recompute, dtype=torch.float32) / tokens
            if not torch.equal(got[key], value.to(got[key].device)):
                problems.append(f"microbatch {microbatch} {key}: {got[key]} != {value}")
    for segment, recorded, analytic in zip(
        ring.segments, result.extra["recorded_work"], result.extra["analytic_work"]
    ):
        if forward_only:
            analytic = (analytic[0], 0.0, 0.0)
        if tuple(recorded) != tuple(microbatches * value for value in analytic):
            problems.append(f"segment {segment['index']} recorded {recorded} != M x {analytic}")
    expected_counts = {m: (ROUNDS, ROUNDS, 1, 1, 1) for m in range(microbatches)}
    if result.extra["sample_counts"] != expected_counts:
        problems.append(f"G1 counters {result.extra['sample_counts']} != {expected_counts}")
    if any(result.extra["injection_keys"]):
        problems.append(f"injection stores not empty: {result.extra['injection_keys']}")
    if any(roles is not None for roles in result.extra["call_roles"]):
        problems.append("call roles left set")
    if any(calls != microbatches for stage in result.extra["calls"] for calls in stage):
        problems.append(f"view calls {result.extra['calls']}")
    if [built_flags(model) for model in ring.models] != ring.built:
        problems.append("built flags changed")
    if set(reference.token_counts) != set(result.token_counts) or any(
        not torch.equal(value, result.token_counts[name])
        for name, value in reference.token_counts.items()
    ):
        problems.append("expert-bias token counts differ")
    if set(reference.moe_tracker) != set(result.moe_tracker):
        problems.append("MoE tracker metrics differ")
    for name, (values, rounds) in reference.moe_tracker.items():
        got_values, got_rounds = result.moe_tracker.get(name, (None, None))
        for want, got in ((values, got_values), (rounds, got_rounds)):
            if want is None or got is None:
                if want is not got:
                    problems.append(f"MoE tracker {name}: round bins on one side only")
                continue
            close = (
                torch.equal(want, got)
                if microbatches == 1
                else torch.allclose(want, got, rtol=1e-6, atol=0.0)
            )
            if not close:
                problems.append(f"MoE tracker {name} differs")
    return problems


def built_flags(model) -> tuple:
    """A chunk model's built pipeline flags (they must never change)."""
    decoder = model.decoder
    return (
        model.pre_process,
        model.post_process,
        model.mtp_process,
        decoder.pre_process,
        decoder.post_process,
    )
