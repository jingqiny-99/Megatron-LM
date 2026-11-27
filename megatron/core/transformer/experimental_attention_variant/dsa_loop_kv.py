# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Top-k metadata for a first-pass sparse MLA bank and a second-pass local bank."""

import torch
from torch import Tensor

from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region


def compact_source_topk(
    indices: Tensor, lengths: Tensor | None, *, causal_ends: Tensor, causal_starts: Tensor | None = None,
    valid_rows: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Preserve the first pass's support, including its backend's padding convention."""
    if lengths is not None:
        # cuDNN already returns sorted, compacted rows with an explicit valid prefix.
        if valid_rows is not None:
            lengths = lengths.masked_fill(~valid_rows, 0)
            indices = indices.masked_fill(~valid_rows.unsqueeze(-1), -1)
        return indices, lengths
    # The reference and TileLang paths use -1 sentinels, potentially between valid entries.
    valid = (indices >= 0) & (indices < causal_ends.view(1, -1, 1))
    if causal_starts is not None:
        valid &= indices >= causal_starts.view(1, -1, 1)
    if valid_rows is not None:
        valid &= valid_rows.unsqueeze(-1)
    sentinel = torch.iinfo(torch.int32).max
    compacted = indices.int().masked_fill(~valid, sentinel).sort(dim=-1).values
    return compacted.masked_fill_(compacted == sentinel, -1), valid.sum(-1, dtype=torch.int32)


def two_bank_topk(
    indices: Tensor, lengths: Tensor, window_size: int, *, causal_ends: Tensor, bank_size: int,
    causal_starts: Tensor | None = None, valid_rows: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Join two sorted, disjoint index ranges in one compact prefix per batch/query."""
    batch, rows, k = indices.shape
    width = min(window_size, bank_size)
    if width < 1:
        raise ValueError("Two-bank DSA requires a nonempty sequence and a positive window")
    device = indices.device
    lengths = lengths.to(torch.int32)
    ends = causal_ends.to(torch.int32)
    offsets = torch.arange(width, dtype=torch.int32, device=device)
    window_starts = (ends - width).clamp_min(0)
    if causal_starts is not None:
        window_starts = torch.maximum(window_starts, causal_starts.to(torch.int32))
    local_ids = window_starts[:, None] + offsets
    # Offset the second bank by S: both versions of a token may appear in the softmax.
    local_ids = (local_ids + bank_size).masked_fill(local_ids >= ends[:, None], -1)

    joint = torch.full((batch, rows, k + width), -1, device=device, dtype=torch.int32)
    source_columns = torch.arange(k, dtype=torch.int32, device=device)
    joint[..., :k] = indices.masked_fill(source_columns >= lengths.unsqueeze(-1), -1)
    dest = (lengths.unsqueeze(-1) + offsets).long()
    joint.scatter_(-1, dest, local_ids.expand(batch, -1, -1))
    joint_lengths = lengths + (ends - window_starts).clamp_min(0)
    if valid_rows is not None:
        joint_lengths = joint_lengths.masked_fill(~valid_rows, 0)
        joint = joint.masked_fill(~valid_rows.unsqueeze(-1), -1)
    return joint, joint_lengths


def gather_zigzag_window_kv(key: Tensor, window: int, cp_group) -> Tensor:
    """Pack the two local chunks and their causal halos; backward sums remote tail gradients."""
    chunk = key.size(0) // 2
    halo = window - 1
    if halo == 0:
        return key
    local = key.unflatten(0, (2, chunk))
    tails = local[:, -halo:].reshape(-1, *key.shape[1:]).contiguous()
    size, rank = cp_group.size(), cp_group.rank()
    gathered = gather_from_sequence_parallel_region(tails, group=cp_group)
    gathered = gathered.unflatten(0, (size, 2, halo))
    parts = []
    for side, ordinal in enumerate((rank, 2 * size - 1 - rank)):
        if ordinal > 0:
            prev = ordinal - 1
            owner = min(prev, 2 * size - 1 - prev)
            parts.append(gathered[owner, int(prev >= size)])
        parts.append(local[side])
    return torch.cat(parts, dim=0)


def zigzag_window_positions(
    local_rows: int, cp_size: int, cp_rank: int, window: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Map compact second-bank offsets to global token positions for masking and selection."""
    chunk = local_rows // 2
    global_rows = local_rows * cp_size
    positions = [torch.arange(global_rows, device=device, dtype=torch.int64)]
    ends = []
    offset = 0
    for ordinal in (cp_rank, 2 * cp_size - 1 - cp_rank):
        start = ordinal * chunk
        halo = min(start, window - 1)
        positions.append(
            torch.arange(start - halo, start + chunk, device=device, dtype=torch.int64)
        )
        first_end = offset + halo + 1
        ends.append(torch.arange(first_end, first_end + chunk, device=device, dtype=torch.int32))
        offset += halo + chunk
    return torch.cat(positions), torch.cat(ends)
