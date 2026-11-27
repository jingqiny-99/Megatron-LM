# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The logical parameter count (dense parameters once, routed experts times EP) and its rank-0
log line (``megatron.core.utils.get_logical_num_parameters``,
``megatron.training.models.dist_utils.logical_num_params_message``).

CPU only: fake model chunks of plain parameters, the routed-expert ones marked as Megatron marks
them (``allreduce = False``), and a fake process-group collection. No process group is created.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.utils import get_logical_num_parameters
from megatron.training.models import dist_utils


def _param(*shape, expert=False, gtp=False, device="cpu"):
    param = torch.nn.Parameter(torch.empty(*shape, device=device))
    if expert:
        param.allreduce = False
    if gtp:
        param.is_gtp_weight_remat = True
    return param


class _Chunk(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        for name, param in params.items():
            self.register_parameter(name, param)


def _chunks(device="cpu"):
    """Two chunks: 26 dense elements (one parameter shared by both chunks) and 57 local
    routed-expert elements (two local experts of one layer and one of another)."""
    shared = _param(3, 5, device=device)
    first = _Chunk(
        dict(
            proj=shared,
            norm=_param(7, device=device),
            expert0=_param(4, 6, expert=True, device=device),
            expert1=_param(4, 6, expert=True, device=device),
        )
    )
    second = _Chunk(
        dict(
            proj=shared,
            out=_param(2, 2, device=device),
            expert0=_param(3, 3, expert=True, device=device),
        )
    )
    return [first, second]


DENSE = 15 + 7 + 4
LOCAL_EXPERTS = 24 + 24 + 9


@pytest.mark.parametrize("ep", [1, 2, 4, 64])
def test_dense_once_and_routed_experts_times_ep(ep):
    expected = DENSE + ep * LOCAL_EXPERTS
    assert get_logical_num_parameters(_chunks(), expert_model_parallel_size=ep) == expected
    if ep > 1:
        # Teeth: counting the local experts once, or the shared parameter twice, is wrong.
        assert expected != DENSE + LOCAL_EXPERTS
        assert expected != DENSE + 15 + ep * LOCAL_EXPERTS


def test_meta_device_chunks_and_a_single_module():
    chunks = _chunks(device="meta")
    assert get_logical_num_parameters(chunks, expert_model_parallel_size=4) == (
        DENSE + 4 * LOCAL_EXPERTS
    )
    only = chunks[1]
    assert get_logical_num_parameters(only, expert_model_parallel_size=4) == 15 + 4 + 4 * 9


def test_a_parameter_reachable_twice_counts_once():
    chunk = _Chunk(dict(weight=_param(5, 5)))
    chunk.child = _Chunk(dict(weight=chunk.weight))  # the same parameter through a submodule
    # ... and the same chunk listed twice (a parameter shared between chunks).
    assert get_logical_num_parameters([chunk, chunk], expert_model_parallel_size=2) == 25


@pytest.mark.parametrize(
    "sizes",
    [
        dict(tensor_model_parallel_size=2),
        dict(pipeline_model_parallel_size=2),
        dict(expert_tensor_parallel_size=2),
    ],
    ids=["tp2", "pp2", "etp2"],
)
def test_undefined_when_another_axis_splits_parameters(sizes):
    assert get_logical_num_parameters(_chunks(), expert_model_parallel_size=4, **sizes) is None


def test_undefined_with_gtp_sharded_parameters_or_unknown_ep():
    gtp = _Chunk(dict(weight=_param(4, 4, gtp=True)))
    assert get_logical_num_parameters([gtp], expert_model_parallel_size=1) is None
    assert get_logical_num_parameters(_chunks(), expert_model_parallel_size=None) is None
    dense_only = _Chunk(dict(weight=_param(4, 4), bias=_param(4)))
    assert get_logical_num_parameters([dense_only], expert_model_parallel_size=None) == 20


class _Group:
    def __init__(self, size, rank=0):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _collection(*, tp=1, pp=1, ep=4, etp=1, dp_rank=0, with_ep=True):
    groups = dict(
        tp=_Group(tp),
        pp=_Group(pp),
        dp=_Group(4, dp_rank),
        cp=_Group(1),
        expt_tp=_Group(etp),
        gtp_remat=None,  # GTP remat off (get_pg_rank(None) is 0)
    )
    if with_ep:
        groups["ep"] = _Group(ep)
    return SimpleNamespace(**groups)


@pytest.fixture
def distributed_initialized(monkeypatch):
    """``get_pg_size`` / ``get_pg_rank`` read the fake groups only when torch.distributed says
    it is initialized."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)


def test_rank0_log_line(capsys, distributed_initialized):
    chunks = _chunks()
    dist_utils._print_num_params(chunks, pg_collection=_collection(ep=4))
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        " > number of parameters on (tensor, gtp_remat, pipeline) model parallel rank "
        f"(0, 0, 0): {DENSE + 15 + LOCAL_EXPERTS}",
        " > logical number of parameters (dense once, routed experts x EP 4): "
        f"{DENSE + 4 * LOCAL_EXPERTS}",
    ]


@pytest.mark.parametrize(
    "collection",
    [_collection(tp=2), _collection(pp=2), _collection(etp=2), _collection(with_ep=False)],
    ids=["tp2", "pp2", "etp2", "no_ep_group"],
)
def test_no_logical_line_where_undefined(capsys, distributed_initialized, collection):
    dist_utils._print_num_params(_chunks(), pg_collection=collection)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and "logical" not in lines[0]


def test_only_data_parallel_rank0_prints(capsys, distributed_initialized):
    dist_utils._print_num_params(_chunks(), pg_collection=_collection(dp_rank=1))
    assert capsys.readouterr().out == ""
