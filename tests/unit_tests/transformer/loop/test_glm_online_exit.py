# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""An online stop depends on the current output and agrees across zigzag CP shards."""
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.transformer.loop.online_exit import online_exit_decision, round_exit_score


def test_round_output_changes_the_exit_of_the_same_row():
    cfg = SimpleNamespace(loop_iterations=3, loop_exit_thresholds=[0.0, -2.0])
    state = torch.tensor([[[4., 0., 0., 1.], [1., 0., 0., 4.]]])
    stop, score = online_exit_decision(cfg, state, 1, return_score=True)
    assert stop.tolist() == [False, True]
    assert (score[0] > 0) and (score[1] < 0)
    assert online_exit_decision(cfg, -state, 1).tolist() == [True, False]
    assert online_exit_decision(cfg, state, 2).tolist() == [False, False]
    assert online_exit_decision(cfg, state, 3).all()


def _check_zigzag(rank, filename):
    dist.init_process_group('gloo', init_method='file://' + filename, rank=rank, world_size=2)
    try:
        full = torch.randn(16, 4, 72, generator=torch.Generator().manual_seed(1234))
        # Four different terminal owners/positions, including an empty sequence.
        last = torch.tensor([15, 10, 2, -1])
        valid = torch.arange(16)[None, :] <= last[:, None]
        for values, mask in ((full, None), (full.masked_fill(~valid.T[..., None], float('nan')), valid)):
            chunks = torch.arange(16).chunk(4)
            idx = torch.cat((chunks[rank], chunks[3-rank]))
            actual = round_exit_score(values[idx], None if mask is None else mask[:, idx], dist.group.WORLD)
            expected = round_exit_score(values, mask)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            if mask is not None:
                assert not actual[-1].item() and actual.isfinite().all()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason='needs a CPU process group')
def test_masked_shard_reads_only_last_valid_global_token(tmp_path):
    mp.spawn(_check_zigzag, args=(str(tmp_path / 'rendezvous'),), nprocs=2)
