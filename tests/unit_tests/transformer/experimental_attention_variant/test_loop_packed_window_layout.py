# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Global row and causal-window equivalence for the packed recurrent DSA bank."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_packed_allgather_cp_local_positions_from_host,
)
from megatron.core.transformer.experimental_attention_variant.dsa_loop_kv import (
    packed_window_positions,
    two_bank_topk,
)


@pytest.mark.parametrize("cp_size,window", [(2, 1), (2, 7), (4, 17), (4, 65), (4, 129)])
def test_packed_current_window_never_crosses_document_or_loses_remote_rows(cp_size, window):
    offsets = (0, 512, 512, 544, 672)
    size = offsets[-1]
    for rank in range(cp_size):
        q = build_packed_allgather_cp_local_positions_from_host(
            offsets, cp_size, rank, torch.device("cpu")
        )
        key_positions, compact_ends = packed_window_positions(
            offsets, cp_size, rank, window, torch.device("cpu")
        )
        starts = torch.tensor(offsets)[torch.bucketize(q, torch.tensor(offsets[1:]), right=True)]
        source = q.to(torch.int32).view(1, -1, 1)
        source_lengths = torch.ones_like(source[..., 0])
        valid = (q != 510).view(1, -1)
        ends = q + 1
        compact_starts = compact_ends - (ends - starts).clamp(max=window)
        idx, lengths = two_bank_topk(
            source, source_lengths, window, causal_ends=compact_ends,
            causal_starts=compact_starts, valid_rows=valid, bank_size=size,
        )
        assert torch.equal(key_positions[:size], torch.arange(size))
        for row, pos in enumerate(q.tolist()):
            n = lengths[0, row].item()
            if not valid[0, row]:
                assert n == 0
                continue
            first = max(pos - window + 1, starts[row].item())
            assert key_positions[idx[0, row, :n].long()].tolist() == [pos, *range(first, pos + 1)]
            assert idx[0, row, 0] < size
            assert (idx[0, row, 1:n] >= size).all()
