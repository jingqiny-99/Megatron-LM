# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Micro-batch-size invariance harness for the hybrid loop (GPU).

One training step of a loop ``HybridModel`` over the same rows, cut into microbatches of
different sizes, through the real path: DDP (fp32 gradient reduction), the no-pipelining
schedule (``get_forward_backward_func``: per-microbatch loss and auxiliary-loss scales,
``set_auxiliary_loss_scales``) and ``finalize_model_grads`` (division by the global loss-token
count under ``calculate_per_token_loss``). Per arm it returns:

* the sum of every loss the step attached: the LM loss (the loss function's masked token sum),
  and the auxiliary losses as attached to the graph by ``MoEAuxLossAutoScaler`` (sequence aux),
  ``MTPLossAutoScaler`` (MTP) and ``DSAIndexerLossAutoScaler`` (DSA / CSA indexer); activation
  replays (``forward_replay``) are not counted, they rebuild what the forward attached;
* every row's LM loss (masked token sum), in row order;
* every parameter's final fp32 gradient (``main_grad`` after ``finalize_model_grads``).

Arms run on one model object: its buffers (router expert biases and token counts, which
``finalize_model_grads`` updates) are restored before every arm, so every arm starts from the
same state. ``per_microbatch_mean_aux_scaling`` is the mutation the invariance tests must catch:
auxiliary and indexer loss gradients scaled by ``1 / micro_batch_size``, i.e. averaged over the
rows of a microbatch instead of summed like the per-token LM loss.

Comparisons. ``compare_gradients`` gives per parameter the cosine, the relative norm
difference ``|g - g_ref| / |g_ref|`` and the absolute one; ``global_comparison`` the same for the
whole gradient. ``resolution_gate`` gates every parameter against a measured noise floor of the
same parameter: a resolved gradient by the strict (fp32) or floor-relative (bf16) relative gate,
an unresolved one (zero up to rounding) absolutely, against the global gradient norm.
"""

import contextlib
import functools
import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence

import torch

LOSS_KEYS = ("lm", "aux", "mtp", "indexer")


def make_rows(num_rows: int, seq: int, vocab: int, *, seed: int):
    """``num_rows`` rows of random tokens and labels (CPU-seeded), all tokens in the loss."""
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, vocab, (num_rows, seq), generator=generator)
    labels = torch.randint(0, vocab, (num_rows, seq), generator=generator)
    return dict(
        tokens=tokens.cuda(),
        labels=labels.cuda(),
        loss_mask=torch.ones(num_rows, seq, dtype=torch.float32).cuda(),
        position_ids=torch.arange(seq).repeat(num_rows, 1).cuda(),
    )


def _microbatches(rows, micro_batch_size: int) -> List[dict]:
    num_rows = rows["tokens"].shape[0]
    assert num_rows % micro_batch_size == 0, (num_rows, micro_batch_size)
    return [
        {name: value[start : start + micro_batch_size] for name, value in rows.items()}
        for start in range(0, num_rows, micro_batch_size)
    ]


class _Recorder:
    def __init__(self):
        self.sums = {key: 0.0 for key in LOSS_KEYS}
        self.rows: List[float] = []

    def add(self, key, value):
        self.sums[key] += float(value.detach().double().sum())


def _loss_func(recorder, loss_mask, output):
    masked = output.float() * loss_mask.float()
    recorder.rows.extend(masked.detach().double().sum(dim=1).tolist())
    losses = output.view(-1).float()
    mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * mask)
    recorder.add("lm", loss)
    num_tokens = mask.sum().clone().detach().to(torch.int)
    return (
        loss,
        num_tokens,
        {"lm loss": torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])},
    )


def _forward_step(recorder, data_iterator, model):
    batch = next(data_iterator)
    output = model(
        batch["tokens"],
        batch["position_ids"],
        None,
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
    )
    return output, functools.partial(_loss_func, recorder, batch["loss_mask"])


@contextlib.contextmanager
def _recording_attached_losses(recorder):
    """Record every auxiliary loss the forward attaches (not the activation replays)."""
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexerLossAutoScaler,
    )
    from megatron.core.transformer.forward_replay import in_forward_replay
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    patched = []
    for key, scaler in (
        ("aux", MoEAuxLossAutoScaler),
        ("mtp", MTPLossAutoScaler),
        ("indexer", DSAIndexerLossAutoScaler),
    ):
        original = scaler.apply  # torch.autograd.Function.apply, bound to the scaler

        def recording_apply(output, loss, key=key, original=original):
            if not in_forward_replay():
                recorder.add(key, loss)
            return original(output, loss)

        patched.append((scaler, scaler.__dict__.get("apply")))
        scaler.apply = recording_apply
    try:
        yield
    finally:
        for scaler, own_apply in patched:
            if own_apply is None:
                del scaler.apply  # back to the inherited Function.apply
            else:
                scaler.apply = own_apply


@contextlib.contextmanager
def per_microbatch_mean_aux_scaling(micro_batch_size: int):
    """The mutation: after the schedule sets the auxiliary-loss scales of a microbatch, scale
    the MoE aux-loss and the DSA / CSA indexer-loss gradients by ``1 / micro_batch_size``."""
    from megatron.core.pipeline_parallel import schedules
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexerLossAutoScaler,
    )
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    original = schedules.set_auxiliary_loss_scales

    def mutant(config, device_fn, *, num_microbatches, cp_group_size):
        original(config, device_fn, num_microbatches=num_microbatches, cp_group_size=cp_group_size)
        for scaler in (MoEAuxLossAutoScaler, DSAIndexerLossAutoScaler):
            if scaler.main_loss_backward_scale is not None:
                scaler.main_loss_backward_scale = scaler.main_loss_backward_scale / float(
                    micro_batch_size
                )

    schedules.set_auxiliary_loss_scales = mutant
    try:
        yield
    finally:
        schedules.set_auxiliary_loss_scales = original


class StepRunner:
    """DDP around ``model`` (fp32 gradient reduction, ``finalize_model_grads``); ``run`` one
    step over ``rows`` cut into microbatches of ``micro_batch_size``. ``grads_device``: where
    ``run`` keeps its copy of the final gradients."""

    def __init__(
        self, model, *, use_distributed_optimizer: bool = False, grads_device: str = "cuda"
    ):
        from megatron.core.distributed import (
            DistributedDataParallel,
            DistributedDataParallelConfig,
            finalize_model_grads,
        )
        from megatron.core.process_groups_config import ProcessGroupCollection

        assert model.config.calculate_per_token_loss
        self.model = model
        self.ddp = DistributedDataParallel(
            model.config,
            DistributedDataParallelConfig(
                overlap_grad_reduce=False,
                use_distributed_optimizer=use_distributed_optimizer,
                grad_reduce_in_fp32=True,
            ),
            model,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        model.config.finalize_model_grads_func = finalize_model_grads
        self.grads_device = grads_device
        self._buffers = {name: buf.detach().clone() for name, buf in model.named_buffers()}

    def run(self, rows, micro_batch_size: int, *, seq: int):
        """``(attached loss sums, per-row LM losses, {name: final fp32 gradient})``."""
        from megatron.core.pipeline_parallel import get_forward_backward_func

        with torch.no_grad():
            for name, buf in self.model.named_buffers():
                buf.copy_(self._buffers[name])
        self.ddp.zero_grad_buffer()
        recorder = _Recorder()
        batches = _microbatches(rows, micro_batch_size)
        with _recording_attached_losses(recorder):
            get_forward_backward_func()(
                forward_step_func=functools.partial(_forward_step, recorder),
                data_iterator=iter(batches),
                model=[self.ddp],
                num_microbatches=len(batches),
                seq_length=seq,
                micro_batch_size=micro_batch_size,
                forward_only=False,
            )
        torch.cuda.synchronize()
        grads = {
            name: param.main_grad.detach().float().to(self.grads_device, copy=True)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        return recorder.sums, recorder.rows, grads


def compare_gradients(
    grads: Dict[str, torch.Tensor],
    reference: Dict[str, torch.Tensor],
    *,
    min_cosine: float,
    max_relative: float,
) -> Dict[str, dict]:
    """Per parameter: cosine, relative norm difference ``|g - g_ref| / |g_ref|`` and absolute
    difference ``|g - g_ref|`` against the reference, and whether the cosine and relative gates
    hold (two all-zero gradients pass)."""
    assert grads.keys() == reference.keys()
    report = {}
    for name, expected in reference.items():
        actual = grads[name].double().flatten()
        expected = expected.double().flatten()
        expected_norm = expected.norm().item()
        actual_norm = actual.norm().item()
        diff = (actual - expected).norm().item()
        if expected_norm == 0.0 or actual_norm == 0.0:
            both_zero = expected_norm == 0.0 and actual_norm == 0.0
            report[name] = dict(
                cosine=1.0 if both_zero else 0.0,
                relative=0.0 if both_zero else math.inf,
                diff=diff,
                norm=expected_norm,
                numel=expected.numel(),
                ok=both_zero,
            )
            continue
        cosine = torch.dot(actual, expected).item() / (actual_norm * expected_norm)
        relative = diff / expected_norm
        report[name] = dict(
            cosine=cosine,
            relative=relative,
            diff=diff,
            norm=expected_norm,
            numel=expected.numel(),
            ok=cosine >= min_cosine and relative <= max_relative,
        )
    return report


def global_comparison(grads: Dict[str, torch.Tensor], reference: Dict[str, torch.Tensor]):
    """Cosine and relative difference of the whole gradient (every parameter concatenated),
    and the reference's global norm."""
    dot = diff2 = norm2 = ref2 = 0.0
    for name, expected in reference.items():
        actual = grads[name].double().flatten()
        expected = expected.double().flatten()
        dot += torch.dot(actual, expected).item()
        diff2 += (actual - expected).pow(2).sum().item()
        norm2 += actual.pow(2).sum().item()
        ref2 += expected.pow(2).sum().item()
    return dict(
        cosine=dot / math.sqrt(norm2 * ref2), relative=math.sqrt(diff2 / ref2), norm=ref2**0.5
    )


def resolution_gate(
    report: Dict[str, dict],
    floor: Dict[str, dict],
    *,
    global_norm: float,
    strict: bool,
    min_cosine: float = 0.9999,
    min_relative: float = 1e-3,
    factor: float = 3.0,
    unresolved_floor: float = 0.5,
    absolute_fraction: float = 1e-4,
    strict_floor_fraction: float = 0.0,
) -> Dict[str, dict]:
    """``report`` re-gated per parameter by whether the floor resolves its gradient.

    ``floor`` gives each parameter's measured noise (the relative difference of two computations
    of the same gradient). A parameter is *resolved* when that noise is at most
    ``unresolved_floor`` of its gradient; then ``strict`` gates cosine >= ``min_cosine`` and
    relative <= max(``min_relative``, ``strict_floor_fraction`` x floor) (a difference well
    below the floor), and the non-strict gate is relative <= max(``min_relative``, ``factor`` x
    floor). An *unresolved* gradient (zero up to rounding, such as the mHC
    stream-symmetry parameters at initialization) has no meaningful relative difference; it is
    gated absolutely: ``|g - g_ref| <= absolute_fraction x global_norm``."""
    out = {}
    for name, entry in report.items():
        noise = floor[name]["relative"]
        resolved = noise <= unresolved_floor
        if not resolved:
            bound = absolute_fraction * global_norm
            ok = entry["diff"] <= bound
        elif strict:
            bound = max(min_relative, strict_floor_fraction * noise)
            ok = entry["cosine"] >= min_cosine and entry["relative"] <= bound
        else:
            bound = max(min_relative, factor * noise)
            ok = entry["relative"] <= bound
        out[name] = dict(entry, floor=noise, bound=bound, resolved=resolved, ok=ok)
    return out


def dump_report(tag: str, report: dict) -> None:
    """With ``MBS_REPORT_DIR`` set: the comparison as JSON, one file per tag and rank."""
    directory = os.environ.get("MBS_REPORT_DIR")
    if not directory:
        return
    os.makedirs(directory, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
    path = os.path.join(directory, f"{name}.rank{torch.distributed.get_rank()}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, default=float)


def relative_loss_differences(losses: dict, reference: dict) -> Dict[str, float]:
    """``|loss - loss_ref| / |loss_ref|`` per attached loss (0 when both are 0)."""
    out = {}
    for key in LOSS_KEYS:
        expected, actual = reference[key], losses[key]
        if expected == 0.0:
            out[key] = 0.0 if actual == 0.0 else math.inf
        else:
            out[key] = abs(actual - expected) / abs(expected)
    return out


def max_row_difference(rows: Sequence[float], reference: Sequence[float]) -> float:
    """The largest relative difference between per-row LM losses (same row order)."""
    assert len(rows) == len(reference), (len(rows), len(reference))
    return max(abs(a - b) / abs(b) for a, b in zip(rows, reference))


def worst(report: Dict[str, dict], names: Optional[Sequence[str]] = None, count: int = 5) -> str:
    """A one-line summary of the worst parameters of ``report`` (for assertion messages)."""
    items = [(name, report[name]) for name in (names or report)]
    items.sort(key=lambda item: (item[1]["ok"], -item[1]["relative"]))
    return "; ".join(
        f"{name}: cos {r['cosine']:.6f} rel {r['relative']:.2e} |g| {r['norm']:.2e}"
        + (f" floor {r['floor']:.2e}" if "floor" in r else "")
        + ("" if r.get("resolved", True) else f" unresolved |dg| {r['diff']:.2e}")
        for name, r in items[:count]
    )


def print_report(tag: str, report: Dict[str, dict], focus: Sequence[str] = ()) -> None:
    """Rank 0: the distribution of ``relative`` over parameters, the 8 worst parameters and
    every ``focus`` parameter (diagnostics for the per-rank logs)."""
    if torch.distributed.get_rank() != 0:
        return
    rels = sorted(r["relative"] for r in report.values())

    def quantile(q):
        return rels[min(len(rels) - 1, int(q * (len(rels) - 1)))]

    print(
        f"[{tag}] params {len(report)}, failing {sum(not r['ok'] for r in report.values())}; "
        f"relative p50 {quantile(0.5):.2e} p90 {quantile(0.9):.2e} max {rels[-1]:.2e}; "
        f"min cosine {min(r['cosine'] for r in report.values()):.8f}",
        flush=True,
    )
    print(f"[{tag}] worst: {worst(report, count=8)}", flush=True)
    unresolved = [name for name, r in report.items() if not r.get("resolved", True)]
    if unresolved:
        print(f"[{tag}] unresolved (gated absolutely): {len(unresolved)}", flush=True)
    for name in focus:
        r = report[name]
        extra = f" floor {r['floor']:.2e} bound {r['bound']:.2e}" if "floor" in r else ""
        if not r.get("resolved", True):
            extra += f" unresolved |dg| {r['diff']:.2e}"
        print(
            f"[{tag}]   {name}: cos {r['cosine']:.8f} rel {r['relative']:.2e} "
            f"|g| {r['norm']:.2e}{extra} {'ok' if r['ok'] else 'FAIL'}",
            flush=True,
        )
