# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Carrier diagnostics of the eager hybrid loop (``--loop-carrier-diagnostics``).

* ``enable_loop_carrier_diagnostics`` (``hybrid_builder``, once the model is built): turns the
  decoder's diagnostics on. With ``summary`` or ``full`` every training forward adds the
  hidden-state RMS per round and, for ``--loop-carrier attn_res``, the summary weights to the
  decoder's ``loop_metrics``; the loss function reports them with the other loop metrics
  (``loop/reporting.py``), so they reach the console, TensorBoard and W&B every log interval.
* ``full`` only, at the TensorBoard interval: ``record_loop_carrier_diagnostics`` (after
  ``train_step``, on every rank) takes the per-sublayer weight sums the decoder accumulated
  since the last record (``LoopAttnResHybridStack.take_carrier_diagnostics``), sums them over
  the data-parallel group (one ``all_gather_object`` of the key list and one fp32
  ``all_reduce``), stores the record on ``args`` and appends it to
  ``--loop-carrier-diagnostics-path`` (last rank; one JSON object per line, format
  ``megatron-loop-carrier-diagnostics/1``: ``iteration``, ``k``, ``entries`` =
  ``[segment, sublayer, round, [mean weight per source]]``); ``log_loop_carrier_diagnostics``
  (``training_log``) writes ``loop attnres/<segment>/sub<j>/r<r>/<source tag>`` scalars
  (``attn_res_stack.attn_res_source_tag``; a core key's round is the round, a coda / head
  key's round is the forward's round count).
* A non-loop hybrid model (the carrier ablation's ``plain-mhc``) with
  ``--loop-carrier-diagnostics-segments PRELUDE ROUNDS CODA``: ``enable_loop_carrier_diagnostics``
  installs forward hooks (``install_boundary_state_rms``) that log the same ``loop state rms
  prelude`` / ``loop state rms r{r}`` metrics at the sublayers where a loop's prelude and rounds
  would end (the RMS of the mHC stream mean, ``loop/reporting.stream_mean_rms``); the decoder's
  forward is unchanged.
"""

import json
from typing import Dict, List, Optional, Sequence

import torch

from megatron.core import parallel_state
from megatron.core.transformer.forward_replay import in_forward_replay
from megatron.core.transformer.loop.reporting import stream_mean_rms
from megatron.training.loop_checkpoint import loop_hybrid_model, loop_hybrid_models

FORMAT = "megatron-loop-carrier-diagnostics/1"


def enable_loop_carrier_diagnostics(args, model) -> None:
    """Turn the decoder's carrier diagnostics on for ``--loop-carrier-diagnostics summary|full``
    (``model``: the ``HybridModel``); only ``full`` accumulates the per-sublayer weight sums
    that ``record_loop_carrier_diagnostics`` takes."""
    mode = getattr(args, "loop_carrier_diagnostics", "off")
    if mode == "off":
        return
    decoder = getattr(model, "decoder", None)
    segments = getattr(args, "loop_carrier_diagnostics_segments", None)
    if segments is not None and decoder is not None:
        install_boundary_state_rms(decoder, *segments)
        return
    enable = getattr(decoder, "enable_carrier_diagnostics", None)
    if enable is None:
        raise ValueError("--loop-carrier-diagnostics needs a hybrid loop decoder (LoopHybridStack)")
    enable(True, per_sublayer=mode == "full")


def boundary_sublayers(num_sublayers: int, prelude: int, rounds: int, coda: int) -> List[int]:
    """The sublayer indices after which a loop's prelude and each of its ``rounds`` rounds
    would end, for a pattern of ``num_sublayers`` read as ``prelude`` sublayers, ``rounds``
    equal segments and ``coda`` sublayers (the prelude entry is -1 when ``prelude`` is 0)."""
    core = num_sublayers - prelude - coda
    if core < rounds or core % rounds:
        raise ValueError(
            f"--loop-carrier-diagnostics-segments {prelude} {rounds} {coda}: the "
            f"{num_sublayers}-sublayer pattern has {core} core sublayers, not a positive "
            f"multiple of {rounds} segments"
        )
    segment = core // rounds
    return [prelude - 1] + [prelude + (step + 1) * segment - 1 for step in range(rounds)]


def install_boundary_state_rms(decoder, prelude: int, rounds: int, coda: int) -> Sequence:
    """Hidden-state RMS per segment of a non-loop ``HybridStack`` (``plain-mhc``).

    A forward pre-hook on ``decoder`` rebinds ``decoder.loop_metrics`` /
    ``loop_metric_populations`` to new dicts every forward (the loss closure captures them, as
    for a loop decoder); forward hooks on the boundary sublayers (``boundary_sublayers``) record
    ``loop state rms prelude`` and ``loop state rms r{r}`` = ``stream_mean_rms`` of the
    sublayer's output over its non-padding tokens, in training forwards only (never an
    activation replay). Returns the hook handles.
    """
    layers = decoder.layers
    config = decoder.config
    streams = config.mhc_num_residual_streams if config.enable_mhc_connections else 1
    ends = boundary_sublayers(len(layers), prelude, rounds, coda)
    names = ["loop state rms prelude"] + [f"loop state rms r{step + 1}" for step in range(rounds)]

    def reset(module, args, kwargs):
        module.loop_metrics = {}
        module.loop_metric_populations = {}

    def record(name):
        def hook(module, args, kwargs, output):
            if not decoder.training or in_forward_replay():
                return
            hidden = output[0] if isinstance(output, tuple) else output
            padding = kwargs.get("padding_mask")
            valid = None if padding is None else ~padding.transpose(0, 1)
            with torch.no_grad():
                value, population = stream_mean_rms(hidden, streams, valid)
            decoder.loop_metrics[name] = value
            decoder.loop_metric_populations[name] = population

        return hook

    handles = [decoder.register_forward_pre_hook(reset, with_kwargs=True)]
    for index, name in zip(ends, names):
        if index >= 0:
            handles.append(layers[index].register_forward_hook(record(name), with_kwargs=True))
    return handles


def _is_record_iteration(args, iteration: int) -> bool:
    interval = getattr(args, "tensorboard_log_interval", None) or 0
    return (
        getattr(args, "loop_carrier_diagnostics", "off") == "full"
        and interval > 0
        and iteration % interval == 0
    )


def record_loop_carrier_diagnostics(args, model, iteration: int, group=None) -> Optional[dict]:
    """The post-``train_step`` hook (every rank, every step; acts at the TensorBoard interval of
    ``--loop-carrier-diagnostics full``): see the module docstring. Returns the record, or None
    (not a record iteration, or a carrier without per-sublayer statistics)."""
    if not _is_record_iteration(args, iteration):
        return None
    if len(loop_hybrid_models(model)) > 1:
        raise ValueError(
            "--loop-carrier-diagnostics full needs a one-chunk loop model (the loop ring's "
            "virtual-pipeline chunks each hold one segment of the rounds)"
        )
    hybrid = loop_hybrid_model(model)
    if hybrid is None:
        return None
    taken = hybrid.decoder.take_carrier_diagnostics()
    if taken is None:
        return None
    weights = taken["weights"]
    device = next(hybrid.decoder.parameters()).device
    if torch.distributed.is_initialized():
        group = group if group is not None else parallel_state.get_data_parallel_group()
        local = sorted((key, int(value[0].shape[0])) for key, value in weights.items())
        gathered = [None] * torch.distributed.get_world_size(group)
        torch.distributed.all_gather_object(gathered, local, group=group)
        layout = sorted({item for part in gathered for item in part})
    else:
        group = None
        layout = sorted((key, int(value[0].shape[0])) for key, value in weights.items())
    pieces = []
    for key, n in layout:
        if key in weights:
            sums, tokens = weights[key]
            pieces.append(sums.detach().float().to(device))
            pieces.append(tokens.detach().float().to(device).reshape(1))
        else:
            pieces.append(torch.zeros(n + 1, dtype=torch.float32, device=device))
    flat = torch.cat(pieces) if pieces else torch.zeros(0, device=device)
    if group is not None:
        torch.distributed.all_reduce(flat, group=group)
    values = flat.tolist()
    entries = []
    offset = 0
    for (segment, sublayer, round_index), n in layout:
        sums, tokens = values[offset : offset + n], values[offset + n]
        offset += n + 1
        entries.append([segment, sublayer, round_index, [s / max(tokens, 1.0) for s in sums]])
    record = {"format": FORMAT, "iteration": iteration, "k": taken["k"], "entries": entries}
    args._loop_carrier_diagnostics = record
    path = getattr(args, "loop_carrier_diagnostics_path", None)
    is_last = (
        not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == torch.distributed.get_world_size() - 1
    )
    if path and is_last:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    return record


def loop_carrier_diagnostics_scalars(record: dict) -> Dict[str, float]:
    """The TensorBoard / W&B values of one record: per sublayer call the mean weight of each
    source tag (``attn_res_source_tag``)."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer.loop.attn_res_stack import attn_res_source_tag

    scalars: Dict[str, float] = {}
    k = record["k"]
    for segment, sublayer, round_index, means in record["entries"]:
        for source, value in enumerate(means):
            tag = attn_res_source_tag(segment, round_index, source, k)
            name = f"loop attnres/{segment}/sub{sublayer:03d}/r{round_index}/{tag}"
            scalars[name] = scalars.get(name, 0.0) + value
    return scalars


def log_loop_carrier_diagnostics(args, iteration: int, writer, wandb_writer) -> None:
    """Write the record of ``iteration`` (``args._loop_carrier_diagnostics``) to the TensorBoard
    and W&B writers that exist (``training_log`` calls it at the TensorBoard interval)."""
    record = getattr(args, "_loop_carrier_diagnostics", None)
    if record is None or record["iteration"] != iteration:
        return
    scalars = loop_carrier_diagnostics_scalars(record)
    if writer:
        for name, value in scalars.items():
            writer.add_scalar(name, value, iteration)
    if wandb_writer:
        wandb_writer.log(scalars, iteration)
