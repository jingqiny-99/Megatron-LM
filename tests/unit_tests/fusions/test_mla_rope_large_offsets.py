# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch


@pytest.mark.internal
@pytest.mark.experimental
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_concat_long_packed_mla_rows_and_adjoints():
    """Check every row across the signed 32-bit element boundary in the absorbed query."""
    from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_concat

    device = torch.device("cuda", torch.cuda.current_device())
    free, _ = torch.cuda.mem_get_info(device)
    if free < 32 * 2**30:
        pytest.skip("Full 64K-row BF16 forward and backward need 32 GiB of free GPU memory")

    cp_size = 16
    cp_rank = (0, 4, 11, 15)[device.index % 4]
    lengths = [1_015_840, 32_736]
    rows, heads, content, rotary = 65_536, 64, 512, 64
    cu = torch.tensor([0, lengths[0], sum(lengths)], device=device, dtype=torch.int32)
    assert rows * heads * content == 2**31
    first_wrapped_row = (2**31 + heads * (content + rotary) - 1) // (heads * (content + rotary))
    assert first_wrapped_row == 58_255

    # Independent token positions inside each packed document, including its CP-mirrored half.
    pieces = []
    for length in lengths:
        chunks = torch.arange(length, device=device).view(2 * cp_size, -1)
        pieces.extend([chunks[cp_rank], chunks[2 * cp_size - 1 - cp_rank]])
    position = torch.cat(pieces)
    assert position.numel() == rows

    # Quarter turns vary by token and frequency. BF16 can represent all oracle arithmetic exactly.
    phase = torch.arange(max(lengths), device=device, dtype=torch.int32)[:, None]
    phase = (phase + torch.arange(rotary // 2, device=device, dtype=torch.int32)) % 4
    cos = ((phase == 0).to(torch.int8) - (phase == 2).to(torch.int8)).to(torch.bfloat16)
    sin = ((phase == 1).to(torch.int8) - (phase == 3).to(torch.int8)).to(torch.bfloat16)
    cos = torch.cat((cos, cos), dim=1)
    sin = torch.cat((sin, sin), dim=1)
    del phase

    gen = torch.Generator(device=device).manual_seed(2026)

    def values(shape):
        tensor = torch.empty(shape, dtype=torch.bfloat16, device=device)
        return tensor.random_(-16, 17, generator=gen)

    nope = values((rows, heads, content)).requires_grad_()
    # Production obtains this non-contiguous slice by splitting a 256-wide Q projection.
    rope = values((rows, heads, 256))[..., -rotary:].detach().requires_grad_()
    out = fused_mla_rope_concat(nope, rope, cos, sin, cu, cp_rank, cp_size)
    dy = values(out.shape)
    dnope, drope = torch.autograd.grad(out, (nope, rope), dy)

    # Comparing in row blocks keeps the independent reference far below the model's HBM footprint.
    # Split exactly around the first wrapped address to also identify a boundary failure directly.
    boundaries = sorted(
        {
            *range(0, rows, 1024),
            first_wrapped_row - 1,
            first_wrapped_row,
            first_wrapped_row + 1,
            rows,
        }
    )
    for start, end in zip(boundaries, boundaries[1:]):
        span = slice(start, end)
        cos_rows = cos[position[span], None, : rotary // 2]
        sin_rows = sin[position[span], None, : rotary // 2]
        even, odd = rope[span, :, 0::2].detach(), rope[span, :, 1::2].detach()
        left, right = dy[span, :, content:].chunk(2, dim=-1)
        assert torch.equal(out[span, :, :content], nope[span]), (start, end, "content")
        assert torch.equal(
            out[span, :, content : content + rotary // 2], even * cos_rows - odd * sin_rows
        ), (start, end, "rotation left")
        assert torch.equal(
            out[span, :, content + rotary // 2 :], odd * cos_rows + even * sin_rows
        ), (start, end, "rotation right")
        assert torch.equal(dnope[span], dy[span, :, :content]), (start, end, "dcontent")
        assert torch.equal(drope[span, :, 0::2], left * cos_rows + right * sin_rows), (
            start,
            end,
            "drotation even",
        )
        assert torch.equal(drope[span, :, 1::2], right * cos_rows - left * sin_rows), (
            start,
            end,
            "drotation odd",
        )
