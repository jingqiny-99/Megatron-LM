# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Typed forward metadata, structural backward slots and real peer identity."""

import os
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.pipeline_parallel import p2p_communication as p2p


def _config(batched=False, dynamic=False):
    return SimpleNamespace(
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=2,
        variable_seq_lengths=dynamic,
        mtp_standalone=False,
        use_ring_exchange_p2p=False,
        batch_p2p_comm=batched,
        batch_p2p_sync=False,
        timers=None,
    )


def _specs():
    return [
        p2p.PipelineTensorSpec((3, 2, 8), torch.bfloat16),
        p2p.PipelineTensorSpec((5, 3, 2), torch.float32, requires_grad=False),
    ]


def _payload(backward=False):
    return [
        None if backward and not spec.requires_grad else torch.ones(spec.shape, dtype=spec.dtype)
        for spec in _specs()
    ]


def _communicator(monkeypatch, batched=False):
    communicator = p2p.P2PCommunicator.__new__(p2p.P2PCommunicator)
    communicator.pp_group = SimpleNamespace(rank=lambda: 0, size=lambda: 4)
    communicator.prev_rank, communicator.next_rank = 3, 1
    communicator.config = _config(batched)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: torch.device('cpu'))
    return communicator


def _record(monkeypatch, expected):
    posted, waited = [], []

    class Work:
        def __init__(self, index):
            self.index = index

        def wait(self, *args, **kwargs):
            assert len(posted) == expected
            waited.append(self.index)
            return True

    def post(direction, tensor, peer):
        posted.append((direction, tensor, peer))
        return Work(len(posted) - 1)

    def send(tensor, dst, group):
        return post('send', tensor, dst)

    def recv(tensor, src, group):
        return post('recv', tensor, src)

    monkeypatch.setattr(dist, 'isend', send)
    monkeypatch.setattr(dist, 'irecv', recv)
    monkeypatch.setattr(
        dist,
        'P2POp',
        lambda op, tensor, peer, group: SimpleNamespace(op=op, tensor=tensor, peer=peer),
    )
    monkeypatch.setattr(
        dist,
        'batch_isend_irecv',
        lambda ops: [post('send' if op.op is send else 'recv', op.tensor, op.peer) for op in ops],
    )
    return posted, waited


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('backward', [False, True])
def test_combined_posts_all_channels_and_skips_metadata_backward(monkeypatch, batched, backward):
    communicator = _communicator(monkeypatch, batched)
    posted, waited = _record(monkeypatch, 3)
    if backward:
        received = communicator.send_backward_recv_forward(_payload(True), _specs(), False)
        assert [tensor.requires_grad for tensor in received] == [True, False]
        assert [tensor.dtype for tensor in received] == [torch.bfloat16, torch.float32]
    else:
        received = communicator.send_forward_recv_backward(_payload(), _specs(), False)
        assert received[0].requires_grad
        assert received[1] is None
    assert len(posted) == 3 and sorted(waited) == [0, 1, 2]


@pytest.mark.parametrize('backward', [False, True])
def test_overlap_keeps_directional_wait_ownership(monkeypatch, backward):
    communicator = _communicator(monkeypatch)
    count = 2 if backward else 4
    posted, waited = _record(monkeypatch, count)
    if backward:
        received, requests = communicator.send_backward_recv_backward(
            _payload(True), True, _specs(), overlap_p2p_comm=True
        )
        assert received[1] is None
        assert set(requests) == {'send_prev', 'recv_next'}
    else:
        received, requests = communicator.send_forward_recv_forward(
            _payload(), True, _specs(), overlap_p2p_comm=True
        )
        assert received[1].requires_grad is False
        assert set(requests) == {'send_next', 'recv_prev'}
        assert all(isinstance(work, p2p._P2PWorkGroup) for work in requests.values())
    assert len(posted) == count and waited == []
    for request in requests.values():
        request.wait()
    assert sorted(waited) == list(range(count))


def test_forward_only_middle_slot_does_not_shift_later_gradients(monkeypatch):
    communicator = _communicator(monkeypatch)
    communicator.config.pipeline_dtype = None
    specs = [*_specs(), p2p.PipelineTensorSpec((2, 4), torch.float32)]
    payload = [*_payload(), torch.ones(2, 4)]
    posted, waited = _record(monkeypatch, 5)
    grads = communicator.send_forward_recv_backward(payload, specs, False)
    assert len(grads) == 3 and grads[1] is None
    assert grads[0].shape == specs[0].shape and grads[0].dtype == torch.bfloat16
    assert grads[2].shape == specs[2].shape and grads[2].dtype == torch.float32
    assert len(posted) == len(waited) == 5


def test_send_backward_without_schema_preserves_explicit_none_slot(monkeypatch):
    communicator = _communicator(monkeypatch)
    posted, waited = _record(monkeypatch, 1)
    communicator.send_backward(_payload(True), False)
    assert len(posted) == len(waited) == 1
    assert posted[0][0] == 'send' and posted[0][1].dtype == torch.bfloat16


@pytest.mark.parametrize('batched', [False, True])
def test_forward_only_backward_has_no_allocations_work_or_sync(monkeypatch, batched):
    communicator = _communicator(monkeypatch, batched)
    communicator.config.batch_p2p_sync = True
    spec = p2p.PipelineTensorSpec((2, 3), torch.float32, False)

    def unexpected(*args, **kwargs):
        raise AssertionError('Forward-only backward must be a structural no-op')

    monkeypatch.setattr(torch, 'empty', unexpected)
    monkeypatch.setattr(torch.cuda, 'synchronize', unexpected)
    monkeypatch.setattr(p2p, '_p2p_ops', unexpected)
    monkeypatch.setattr(p2p, '_batched_p2p_ops', unexpected)
    forward, backward, requests = communicator._communicate(
        tensor_send_next=None,
        tensor_send_prev=[None],
        recv_prev=False,
        recv_next=True,
        tensor_shape=[spec],
    )
    assert forward is None and backward == [None]
    assert requests == ([] if batched else {})
    assert communicator.recv_backward(spec, False) is None
    assert communicator.recv_backward([], False) == []


@pytest.mark.parametrize(
    'problem', ['dtype', 'shape', 'live_metadata', 'missing_value', 'count', 'aux_grad']
)
def test_typed_validation_precedes_all_posts(monkeypatch, problem):
    communicator = _communicator(monkeypatch)
    posted, waited = _record(monkeypatch, 0)
    payload = _payload()
    backward = problem == 'aux_grad'
    if problem == 'dtype':
        payload[1] = payload[1].to(torch.bfloat16)
    elif problem == 'shape':
        payload[1] = payload[1][1:]
    elif problem == 'live_metadata':
        payload[1].requires_grad_()
    elif problem == 'missing_value':
        payload[0] = None
    elif problem == 'count':
        payload.pop()
    with pytest.raises(ValueError):
        if backward:
            communicator.send_backward(payload, False, tensor_shapes=_specs())
        else:
            communicator.send_forward(payload, False, tensor_shapes=_specs())
    assert posted == waited == []


@pytest.mark.parametrize(
    'shape,dtype,requires_grad,error',
    [
        ((-1, 2), torch.float32, True, ValueError),
        ((True, 2), torch.float32, True, ValueError),
        ((1.5, 2), torch.float32, True, ValueError),
        ((2,), 'float32', True, TypeError),
        ((2,), torch.float32, 1, TypeError),
        ((2,), torch.int64, True, ValueError),
    ],
)
def test_invalid_channel_spec(shape, dtype, requires_grad, error):
    with pytest.raises(error):
        p2p.PipelineTensorSpec(shape, dtype, requires_grad)


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('multiple', [False, True])
def test_legacy_shapes_keep_all_gradient_channels(monkeypatch, batched, multiple):
    communicator = _communicator(monkeypatch, batched)
    shape = (3, 2, 8)
    payload = torch.ones(shape, dtype=torch.bfloat16)
    shapes = [shape, shape] if multiple else shape
    sends = [payload, payload.clone()] if multiple else payload
    posted, waited = _record(monkeypatch, 4 if multiple else 2)
    received = communicator.send_forward_recv_backward(sends, shapes, False)
    values = received if multiple else [received]
    assert all(value.requires_grad and value.dtype == torch.bfloat16 for value in values)
    assert len(posted) == len(waited) == (4 if multiple else 2)


def _exchange(communicator, device, *, dynamic=False, overlap=False, typed=True):
    specs = _specs() if typed else [_specs()[0].shape]
    rank = dist.get_rank()

    def marker(sender, backward, step, channel):
        return sender * 32 + int(backward) * 16 + step * 4 + channel + 1

    def payload(backward, step):
        values = []
        for channel, spec in enumerate(_specs() if typed else _specs()[:1]):
            if backward and not spec.requires_grad:
                values.append(None)
                continue
            shape = list(spec.shape)
            if dynamic:
                shape[0] += step
            values.append(
                torch.full(
                    shape, marker(rank, backward, step, channel), dtype=spec.dtype, device=device
                )
            )
        return values

    for step in range(3):
        forward, backward, requests = communicator._communicate(
            tensor_send_next=payload(False, step),
            tensor_send_prev=payload(True, step),
            recv_prev=True,
            recv_next=True,
            tensor_shape=specs,
            wait_on_reqs=not overlap,
        )
        if overlap:
            assert set(requests) == {'send_prev', 'recv_prev', 'send_next', 'recv_next'}
            for work in requests.values():
                work.wait()
        else:
            assert requests is None
        for reverse, tensors, peer in (
            (False, forward, communicator.prev_rank),
            (True, backward, communicator.next_rank),
        ):
            for channel, tensor in enumerate(tensors):
                if reverse and channel == 1:
                    assert tensor is None
                    continue
                expected = marker(peer, reverse, step, channel)
                assert bool(torch.all(tensor == expected))
                assert tensor.requires_grad == (channel == 0)
                assert tensor.shape[0] == _specs()[channel].shape[0] + (step if dynamic else 0)


def _gloo_worker(rank, world_size, rendezvous):
    dist.init_process_group(
        'gloo',
        init_method='file://' + rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=45),
    )
    try:
        groups = [dist.group.WORLD]
        if world_size == 4:
            # PP-group parity differs from world parity in [1, 2].
            for ranks in ([0, 3], [1, 2]):
                group = dist.new_group(ranks=ranks, backend='gloo')
                if rank in ranks:
                    groups.append(group)
        with (
            mock.patch.object(torch.cuda, 'current_device', lambda: torch.device('cpu')),
            mock.patch.object(torch.cuda, 'synchronize', lambda: None),
        ):
            for group in groups:
                for batched in (False, True):
                    for dynamic in (False, True):
                        communicator = p2p.P2PCommunicator(group, _config(batched, dynamic))
                        _exchange(communicator, 'cpu', dynamic=dynamic)
                        if not batched:
                            _exchange(communicator, 'cpu', dynamic=dynamic, overlap=True)
                    _exchange(p2p.P2PCommunicator(group, _config(batched)), 'cpu', typed=False)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    int(os.environ.get('WORLD_SIZE', '1')) != 1, reason='Spawn Gloo from one parent'
)
@pytest.mark.parametrize('world_size', [2, 4])
def test_gloo_forward_metadata_real_transport(tmp_path, world_size):
    mp.spawn(
        _gloo_worker, args=(world_size, str(tmp_path / 'gloo-init')), nprocs=world_size, join=True
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get('WORLD_SIZE', '1')) != 2,
    reason='Run with torchrun --nproc-per-node 2 on CUDA',
)
@pytest.mark.parametrize('batched', [False, True])
def test_nccl_forward_metadata_real_transport(batched):
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_distributed()
    communicator = p2p.P2PCommunicator(dist.group.WORLD, _config(batched))
    _exchange(communicator, 'cuda')
    if not batched:
        _exchange(communicator, 'cuda', overlap=True)
    dist.barrier()
