# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Spike (j), J5: saved bytes and eager GPU time of the fused DSv4 attention core.

One attention-only HyperConnectionHybridLayer (``C`` or ``H``) at the spike's width with
``dsa_kernel_backend="cudnn"`` (FlashMLA + cuDNN DSA, the ``dsv4-gb200`` container), one GPU,
per sequence length: the bytes autograd saves by component (the same attribution as
``pair_capture.saved_tensor_bytes``) and GPU times from CUDA events (synchronized):
``core_ms_fwd`` around the sparse-attention core call alone (``csa_sparse_attn``, or the
reference ``unfused_compressed_sparse_attn`` when the fused path is not taken), the same
boundary as the spike's ``gpu_ms_core_eager`` and the one a composite "raw - reference core +
J5 core" may use; ``core_with_indexer_loss_ms_fwd`` around the fused core + indexer-loss
Function (not separable; not comparable with the spike's core); ``csa_module_ms_fwd`` around
the whole ``CompressedSparseAttention`` forward (indexer and compressor included). ``pair_capture --saved-bytes-only --fused-core-json``
replaces the reference core with these numbers. A build or run failure is recorded per entry
(the analytic ``fused_core_saved_bytes`` is used instead).

    python -m tests.unit_tests.transformer.loop.fused_attn_bytes --width pro-proxy \
        --kinds CSA,HCA --seqs 4096,8192 --out fused_core.json
"""

import argparse
import contextlib
import json
import os
from typing import Any, Dict, List

import torch

from tests.unit_tests.transformer.loop import pair_capture as pc


def _timed(function, events: List):
    """``function`` with CUDA events recorded around each call into ``events``."""

    def call(*args, **kwargs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        output = function(*args, **kwargs)
        end.record()
        events.append((start, end))
        return output

    return call


@contextlib.contextmanager
def _fused_core_timing(core_events: List, core_loss_events: List):
    """CUDA events around the fused sparse-attention core calls of ``csa``: ``csa_sparse_attn``
    (core only) and ``FusedCSAIndexerSparseAttnFromTopkFunc.apply`` (core + indexer loss)."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer.experimental_attention_variant import csa

    original_core = csa.csa_sparse_attn
    original_loss = csa.FusedCSAIndexerSparseAttnFromTopkFunc

    class _TimedCoreWithLoss:  # pylint: disable=too-few-public-methods
        apply = staticmethod(_timed(original_loss.apply, core_loss_events))

    csa.csa_sparse_attn = _timed(original_core, core_events)
    csa.FusedCSAIndexerSparseAttnFromTopkFunc = _TimedCoreWithLoss
    try:
        yield
    finally:
        csa.csa_sparse_attn = original_core
        csa.FusedCSAIndexerSparseAttnFromTopkFunc = original_loss


def _elapsed_ms(events: List) -> float:
    return sum(start.elapsed_time(end) for start, end in events)


def _core_timing_hooks(model, events: List):
    """CUDA events around every ``CompressedSparseAttention`` forward (the whole module)."""
    # pylint: disable=import-outside-toplevel
    from megatron.core.transformer.experimental_attention_variant.csa import (
        CompressedSparseAttention,
    )

    handles, pending = [], []

    def pre(_module, _args):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        pending.append(start)

    def post(_module, _args, _output):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        events.append((pending.pop(), end))

    for module in model.modules():
        if isinstance(module, CompressedSparseAttention):
            handles.append(module.register_forward_pre_hook(pre))
            handles.append(module.register_forward_hook(post))
    return handles


def measure(
    width: str, kind: str, seq: int, backend: str, seed: int, pg_collection
) -> Dict[str, Any]:
    """Saved bytes by component and the core's forward GPU time for one attention sublayer."""
    config = pc.spike_config(
        width,
        ep=1,
        kind=kind,
        graphs=False,
        seq=seq,
        pairs=1,
        dsa_backend=backend,
        attention_only=True,
    )
    model, _ = pc.build_pair_models(
        config,
        pg_collection=pg_collection,
        seed=seed,
        twin=False,
        vocab_size=pc.WIDTHS[width]["vocab_size"],
        max_sequence_length=seq,
    )
    pc._prepare([model], pg_collection)  # pylint: disable=protected-access
    hidden = pc.make_inputs(config, seq, seed=seed)["x0"]
    kwargs = pc.static_kwargs(model, seq)
    timing: List = []
    reference_calls: List = []
    fused_calls: List = []
    fused_loss_calls: List = []
    handles = _core_timing_hooks(model, timing)
    try:
        with _fused_core_timing(fused_calls, fused_loss_calls):
            # warm-up (kernel selection, workspaces), then the measured forward
            pc.saved_tensor_bytes(model, hidden, kwargs_per_layer=kwargs)
            for events in (timing, fused_calls, fused_loss_calls):
                events.clear()
            torch.cuda.synchronize()
            record: Dict[str, Any] = dict(
                pc.saved_tensor_bytes(
                    model, hidden, kwargs_per_layer=kwargs, core_events=reference_calls
                )
            )
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    record.update(
        core_ms_fwd=_elapsed_ms(fused_calls) + _elapsed_ms(reference_calls),
        core_calls=len(fused_calls) + len(reference_calls),
        core_with_indexer_loss_ms_fwd=_elapsed_ms(fused_loss_calls),
        core_with_indexer_loss_calls=len(fused_loss_calls),
        csa_module_ms_fwd=_elapsed_ms(timing),
        csa_module_calls=len(timing),
        reference_core_calls=len(reference_calls),
        fused_path=len(reference_calls) == 0,
        backend=backend,
        analytic=pc.fused_core_saved_bytes(config, seq),
    )
    del model
    torch.cuda.empty_cache()
    return record


def main(argv=None) -> None:
    """CLI: one JSON object ``{kind: {seq: record}}``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--width", choices=sorted(pc.WIDTHS), default="pro-proxy")
    parser.add_argument("--kinds", default="CSA,HCA")
    parser.add_argument("--seqs", default="4096,8192")
    parser.add_argument("--backend", default="cudnn")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    for key, value in (
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29533"),
    ):
        os.environ.setdefault(key, value)
    pg_collection, _ = pc._init_distributed()  # pylint: disable=protected-access
    result: Dict[str, Dict[str, Any]] = {}
    try:
        for kind in pc.parse_list(args.kinds):
            result[kind] = {}
            for seq in pc.parse_list(args.seqs, int):
                try:
                    result[kind][str(seq)] = measure(
                        args.width, kind, seq, args.backend, args.seed, pg_collection
                    )
                except Exception as error:  # pylint: disable=broad-except
                    result[kind][str(seq)] = dict(error=pc.failure_record(error, "build/run"))
                print(f"[fused_attn_bytes] {kind} {seq}: {result[kind][str(seq)]}", flush=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1, default=str)
    finally:
        # pylint: disable=import-outside-toplevel
        from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator

        destroy_num_microbatches_calculator()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
