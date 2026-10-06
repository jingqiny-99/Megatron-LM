# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Mixed-precision pipeline channels and shared-graph backward contracts."""

import os
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.pipeline_parallel import p2p_communication as p2p
from megatron.core.pipeline_parallel import schedules


class _Group:
    def rank(self):
        return 0

    def size(self):
        return 4


def _communicator(monkeypatch, *, batched=False):
    communicator = p2p.P2PCommunicator.__new__(p2p.P2PCommunicator)
    communicator.pp_group = _Group()
    communicator.prev_rank, communicator.next_rank = 3, 1
    communicator.config = SimpleNamespace(
        pipeline_dtype=torch.bfloat16,
        variable_seq_lengths=False,
        mtp_standalone=False,
        use_ring_exchange_p2p=False,
        batch_p2p_comm=batched,
        batch_p2p_sync=False,
        timers=None,
    )
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: torch.device('cpu'))
    return communicator


def _specs():
    return [
        p2p.PipelineTensorSpec((3, 2, 16), torch.bfloat16),
        p2p.PipelineTensorSpec((5, 3, 2), torch.float32),
    ]


def _payload():
    return [torch.ones(spec.shape, dtype=spec.dtype) for spec in _specs()]


def _record_p2p(monkeypatch, *, expected_before_wait):
    posted, waited = [], []

    class Work:
        def __init__(self, index):
            self.index = index

        def wait(self):
            # In particular, forward send + backward recv must send BOTH
            # channels before waiting: the peer needs both to run backward.
            assert len(posted) == expected_before_wait
            waited.append(self.index)

    def post(operation, tensor, peer):
        posted.append((operation, tensor, peer))
        return Work(len(posted) - 1)

    def isend(tensor, dst, group):
        return post('send', tensor, dst)

    def irecv(tensor, src, group):
        return post('recv', tensor, src)

    monkeypatch.setattr(torch.distributed, 'isend', isend)
    monkeypatch.setattr(torch.distributed, 'irecv', irecv)
    monkeypatch.setattr(
        torch.distributed,
        'P2POp',
        lambda op, tensor, peer, group: SimpleNamespace(op=op, tensor=tensor, peer=peer),
    )

    def batched(ops):
        return [post('send' if op.op is isend else 'recv', op.tensor, op.peer) for op in ops]

    monkeypatch.setattr(torch.distributed, 'batch_isend_irecv', batched)
    return posted, waited


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('method', ['send_forward_recv_backward', 'send_backward_recv_forward'])
def test_combined_exchange_posts_all_channels_before_wait(monkeypatch, batched, method):
    communicator = _communicator(monkeypatch, batched=batched)
    posted, waited = _record_p2p(monkeypatch, expected_before_wait=4)
    received = getattr(communicator, method)(_payload(), _specs(), False)
    assert len(received) == 2
    assert [tensor.dtype for tensor in received] == [torch.bfloat16, torch.float32]
    assert [tuple(tensor.shape) for tensor in received] == [spec.shape for spec in _specs()]
    assert all(tensor.requires_grad for tensor in received)
    assert len(posted) == 4
    assert sorted(waited) == list(range(4))


@pytest.mark.parametrize('backward', [False, True])
def test_overlapped_exchange_keeps_directional_wait_handles(monkeypatch, backward):
    communicator = _communicator(monkeypatch)
    posted, waited = _record_p2p(monkeypatch, expected_before_wait=4)
    if backward:
        received, handles = communicator.send_backward_recv_backward(
            _payload(), True, _specs(), overlap_p2p_comm=True
        )
        expected_directions = {'send_prev', 'recv_next'}
    else:
        received, handles = communicator.send_forward_recv_forward(
            _payload(), True, _specs(), overlap_p2p_comm=True
        )
        expected_directions = {'send_next', 'recv_prev'}
    assert set(handles) == expected_directions
    assert [tensor.dtype for tensor in received] == [torch.bfloat16, torch.float32]
    assert len(posted) == 4 and waited == []
    for handle in handles.values():
        handle.wait()
    assert sorted(waited) == list(range(4))


@pytest.mark.parametrize('batched', [False, True])
def test_bidirectional_payload_exchange(monkeypatch, batched):
    communicator = _communicator(monkeypatch, batched=batched)
    posted, waited = _record_p2p(monkeypatch, expected_before_wait=8)
    forward, backward = communicator.send_forward_backward_recv_forward_backward(
        _payload(), _payload(), True, True, _specs()
    )
    assert len(posted) == len(waited) == 8
    assert [tensor.dtype for tensor in forward] == [torch.bfloat16, torch.float32]
    assert [tensor.dtype for tensor in backward] == [torch.bfloat16, torch.float32]


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('same_peer', [False, True])
def test_batched_exchange_matches_opposite_directions_for_the_same_peer(
    monkeypatch, rank, same_peer
):
    """Direction order must pair forward with forward, even for equal wire shapes."""
    group = SimpleNamespace(rank=lambda: rank)
    tensors = {name: _payload() for name in ('send_prev', 'recv_prev', 'send_next', 'recv_next')}
    posted, _ = _record_p2p(monkeypatch, expected_before_wait=8)
    p2p._batched_p2p_ops(
        **{'tensor_' + name: payload for name, payload in tensors.items()},
        group=group,
        prev_pipeline_rank=7,
        next_pipeline_rank=7 if same_peer else 3,
    )
    directions = (
        ('send_next', 'recv_next', 'send_prev', 'recv_prev')
        if same_peer and rank == 0
        else ('send_prev', 'recv_prev', 'send_next', 'recv_next')
    )
    expected = [tensor for name in directions for tensor in tensors[name]]
    assert [id(tensor) for _, tensor, _ in posted] == [id(tensor) for tensor in expected]


def _gloo_bidirectional_worker(rank, world_size, rendezvous, typed):
    """Check message identity on real CPU transports without an accelerator."""
    dist = torch.distributed
    dist.init_process_group(
        'gloo',
        init_method='file://' + rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        groups = [dist.group.WORLD]
        if world_size == 4:
            # Global rank 1 is even in its PP group. Direction selection must
            # use the pipeline-group rank, not the global rank.
            for ranks in ([0, 3], [1, 2]):
                group = dist.new_group(ranks=ranks, backend='gloo')
                if rank in ranks:
                    groups.append(group)
        for group in groups:
            config = SimpleNamespace(
                pipeline_dtype=torch.bfloat16,
                virtual_pipeline_model_parallel_size=2,
                variable_seq_lengths=False,
                mtp_standalone=False,
                use_ring_exchange_p2p=False,
                batch_p2p_comm=True,
                batch_p2p_sync=False,
                timers=None,
            )
            communicator = p2p.P2PCommunicator(group, config)
            shapes = _specs() if typed else _specs()[0].shape

            def payload(sender, backward, step):
                tensors = []
                for channel, spec in enumerate(_specs() if typed else _specs()[:1]):
                    shape = list(spec.shape)
                    if config.variable_seq_lengths:
                        shape[0] += sender + (4 if backward else 0)
                    value = 20 * sender + 8 * step + 2 * channel + int(backward) + 1
                    tensors.append(torch.full(shape, value, dtype=spec.dtype))
                return tensors if typed else tensors[0]

            with (
                mock.patch.object(torch.cuda, 'current_device', return_value=torch.device('cpu')),
                mock.patch.object(torch.cuda, 'synchronize'),
            ):
                for dynamic_shapes in (False, True):
                    config.variable_seq_lengths = dynamic_shapes
                    for step in range(2):
                        forward, backward = (
                            communicator.send_forward_backward_recv_forward_backward(
                                payload(rank, False, step),
                                payload(rank, True, step),
                                True,
                                True,
                                shapes,
                            )
                        )
                        torch.testing.assert_close(
                            forward, payload(communicator.prev_rank, False, step), rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            backward, payload(communicator.next_rank, True, step), rtol=0, atol=0
                        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('world_size', [2, 4])
@pytest.mark.parametrize('typed', [False, True])
def test_real_gloo_bidirectional_message_identity(tmp_path, world_size, typed):
    """PP2/VPP must preserve direction identity as well as payload shape/dtype."""
    if not torch.distributed.is_gloo_available():
        pytest.skip('Requires the Gloo CPU transport')
    if int(os.environ.get('RANK', '0')) != 0:
        pytest.skip('Only one runner rank needs to launch the isolated Gloo regression')
    torch.multiprocessing.spawn(
        _gloo_bidirectional_worker,
        args=(world_size, str(tmp_path / 'p2p-store'), typed),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize('typed', [False, True])
def test_real_nccl_pp2_bidirectional_message_identity(typed):
    """Use the normal torchrun ranks to detect same-peer forward/backward swaps."""
    if not torch.cuda.is_available() or not torch.distributed.is_nccl_available():
        pytest.skip('Requires the distributed NCCL GPU runner')
    if int(os.environ.get('WORLD_SIZE', '1')) < 2:
        pytest.skip('Requires at least two torchrun ranks')

    from megatron.core import parallel_state
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2
    )
    try:
        group = parallel_state.get_pipeline_model_parallel_group()
        config = SimpleNamespace(
            pipeline_dtype=torch.bfloat16,
            virtual_pipeline_model_parallel_size=2,
            variable_seq_lengths=False,
            mtp_standalone=False,
            use_ring_exchange_p2p=False,
            batch_p2p_comm=True,
            batch_p2p_sync=False,
            timers=None,
        )
        communicator = p2p.P2PCommunicator(group, config)
        assert communicator.prev_rank == communicator.next_rank
        shapes = _specs() if typed else _specs()[0].shape

        def payload(sender, backward, step):
            tensors = [
                torch.full(
                    spec.shape,
                    20 * sender + 8 * step + 2 * channel + int(backward) + 1,
                    dtype=spec.dtype,
                    device=torch.cuda.current_device(),
                )
                for channel, spec in enumerate(_specs() if typed else _specs()[:1])
            ]
            return tensors if typed else tensors[0]

        rank = torch.distributed.get_rank()
        for step in range(3):
            forward, backward = communicator.send_forward_backward_recv_forward_backward(
                payload(rank, False, step), payload(rank, True, step), True, True, shapes
            )
            torch.testing.assert_close(
                forward, payload(communicator.prev_rank, False, step), rtol=0, atol=0
            )
            torch.testing.assert_close(
                backward, payload(communicator.next_rank, True, step), rtol=0, atol=0
            )
    finally:
        Utils.destroy_model_parallel()


def test_typed_channel_does_not_require_default_pipeline_dtype(monkeypatch):
    communicator = _communicator(monkeypatch)
    communicator.config.pipeline_dtype = None
    _record_p2p(monkeypatch, expected_before_wait=1)
    tensor = communicator.recv_forward(_specs()[1], False)
    assert isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32


@pytest.mark.parametrize('shape', [(3, 2, 16), [3, 2, 16], torch.Size([3, 2, 16])])
def test_legacy_shape_and_default_dtype_preserved(monkeypatch, shape):
    communicator = _communicator(monkeypatch)
    _record_p2p(monkeypatch, expected_before_wait=1)
    tensor = communicator.recv_forward(shape, False)
    assert isinstance(tensor, torch.Tensor) and tensor.dtype == torch.bfloat16
    assert tuple(tensor.shape) == tuple(shape)


@pytest.mark.parametrize('backend', ['unbatched', 'batched', 'ring'])
@pytest.mark.parametrize('dynamic_shapes', [False, True])
def test_empty_channel_list_is_a_communication_noop(monkeypatch, backend, dynamic_shapes):
    communicator = _communicator(monkeypatch, batched=backend == 'batched')
    communicator.config.use_ring_exchange_p2p = backend == 'ring'
    communicator.config.variable_seq_lengths = dynamic_shapes
    communicator.config.batch_p2p_sync = True

    def unexpected(*args, **kwargs):
        pytest.fail('A zero-channel exchange must not allocate or communicate')

    monkeypatch.setattr(torch, 'empty', unexpected)
    monkeypatch.setattr(torch.cuda, 'synchronize', unexpected)
    monkeypatch.setattr(communicator, '_communicate_shapes', unexpected)
    monkeypatch.setattr(p2p, '_batched_p2p_ops', unexpected)
    monkeypatch.setattr(p2p, '_p2p_ops', unexpected)
    monkeypatch.setattr(torch.distributed, 'ring_exchange', unexpected, raising=False)

    assert communicator.recv_forward([], False) == []
    assert communicator.recv_backward([], False) == []
    assert communicator.send_forward([], False) is None
    assert communicator.send_backward([], False) is None
    for payload in ([], None):
        assert communicator.send_forward_recv_backward(payload, [], False) == []
        assert communicator.send_backward_recv_forward(payload, [], False) == []
        assert communicator.send_forward_backward_recv_forward_backward(
            payload, payload, True, True, []
        ) == ([], [])

    empty_requests = {} if backend == 'unbatched' else []
    for recv_prev, recv_next in ((True, True), (True, False), (False, True), (False, False)):
        for wait_on_reqs in (False, True):
            forward, backward, requests = communicator._communicate(
                tensor_send_next=[],
                tensor_send_prev=[],
                recv_prev=recv_prev,
                recv_next=recv_next,
                tensor_shape=[],
                wait_on_reqs=wait_on_reqs,
            )
            assert forward == ([] if recv_prev else None)
            assert backward == ([] if recv_next else None)
            assert requests == empty_requests

    for method in ('send_forward_recv_forward', 'send_backward_recv_backward'):
        assert getattr(communicator, method)([], True, [], overlap_p2p_comm=True) == (
            [],
            empty_requests,
        )


@pytest.mark.parametrize('shape', [torch.Size([]), p2p.PipelineTensorSpec((), torch.float32)])
def test_explicit_scalar_shape_is_not_an_empty_channel_list(monkeypatch, shape):
    communicator = _communicator(monkeypatch)
    posted, waited = _record_p2p(monkeypatch, expected_before_wait=1)
    result = communicator.recv_forward(shape, False)
    assert result.shape == torch.Size([])
    assert len(posted) == len(waited) == 1


@pytest.mark.parametrize('method', ['send_forward_recv_backward', 'send_backward_recv_forward'])
@pytest.mark.parametrize('list_payload', [False, True])
@pytest.mark.parametrize('list_shapes', [False, True])
def test_combined_legacy_container_contract(monkeypatch, method, list_payload, list_shapes):
    communicator = _communicator(monkeypatch)
    _record_p2p(monkeypatch, expected_before_wait=2)
    payload = _payload()[0]
    shape = (3, 2, 16)
    received = getattr(communicator, method)(
        [payload] if list_payload else payload, [shape] if list_shapes else shape, False
    )
    assert isinstance(received, list) == list_payload
    tensor = received[0] if list_payload else received
    assert tensor.shape == shape and tensor.dtype == torch.bfloat16


def test_mismatched_channel_count_rejected_before_posting(monkeypatch):
    communicator = _communicator(monkeypatch)
    posted, _ = _record_p2p(monkeypatch, expected_before_wait=0)
    with pytest.raises(ValueError, match='equal lengths'):
        communicator.send_forward_recv_backward(_payload()[:1], _specs(), False)
    assert posted == []


def test_wrong_channel_dtype_rejected_before_posting(monkeypatch):
    communicator = _communicator(monkeypatch)
    posted, _ = _record_p2p(monkeypatch, expected_before_wait=0)
    payload = _payload()
    payload[1] = payload[1].bfloat16()
    with pytest.raises(ValueError, match='expected torch.float32'):
        communicator.send_forward_recv_backward(payload, _specs(), False)
    assert posted == []


@pytest.mark.parametrize('deallocate', [False, True])
def test_shared_graph_backward_combines_both_channel_gradients(deallocate):
    value = torch.randn(3, 2, 16, requires_grad=True)
    weight = torch.randn(16, 5, requires_grad=True)
    shared = value.square()
    outputs = [shared.clone(), shared @ weight]
    gradients = [torch.randn_like(output) for output in outputs]
    reference = torch.autograd.grad(outputs, (value, weight), gradients, retain_graph=True)
    if deallocate:
        schedules.deallocate_output_tensor(outputs, True)
        assert all(output.numel() == 1 for output in outputs)
    config = SimpleNamespace(
        timers=None, grad_scale_func=None, deallocate_pipeline_outputs=deallocate
    )
    input_grads = schedules.backward_step([value], outputs, gradients, config)
    torch.testing.assert_close(input_grads[0], reference[0])
    torch.testing.assert_close(weight.grad, reference[1])


def test_unused_source_score_channel_sends_zero_gradient():
    value = torch.randn(3, 2, 16, requires_grad=True)
    scores = torch.randn(5, 3, 2, requires_grad=True)
    config = SimpleNamespace(
        timers=None, grad_scale_func=None, deallocate_pipeline_outputs=False, attn_res_impl='source'
    )
    gradients = schedules.backward_step([value, scores], value.sum(), None, config)
    torch.testing.assert_close(gradients[0], torch.ones_like(value))
    torch.testing.assert_close(gradients[1], torch.zeros_like(scores))


def test_auxiliary_loss_scale_uses_value_channel_device():
    assert schedules.get_tensor_device(_payload()) == torch.device('cpu')


def test_forward_step_preserves_flat_payload(monkeypatch):
    captured = []
    model = SimpleNamespace(set_input_tensor=captured.append)
    payload = _payload()
    config = SimpleNamespace(timers=None, enable_autocast=False)
    monkeypatch.setattr(
        schedules, 'get_attr_wrapped_model', lambda module, name: getattr(module, name)
    )
    monkeypatch.setattr(
        schedules, 'forward_step_calc_loss', lambda model, output, *args: (output, torch.tensor(0))
    )
    output, _ = schedules.forward_step(
        lambda data, module: (payload, None),
        None,
        model,
        1,
        [None, None],
        [],
        config,
        1,
        is_last_stage=False,
    )
    assert output is payload
    assert captured == [[None, None]]


@pytest.mark.parametrize(
    'pipeline_size,virtual_size,rank,overlap',
    [(1, None, 0, False), (2, None, 0, False), (2, None, 1, False)]
    + [(2, 2, rank, overlap) for rank in [0, 1] for overlap in [False, True]],
)
@pytest.mark.parametrize('forward_only', [False, True])
def test_projection_schedule_lifecycle(
    monkeypatch, pipeline_size, virtual_size, rank, overlap, forward_only
):
    """Every schedule prepares before receive and finalizes after communication/no_sync.

    Communication completes through recorded work handles; real tensor graphs
    exercise the scalar terminal loss and both intermediate payload channels.
    """
    from contextlib import contextmanager

    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer import attention_residual

    events, pending = [], set()
    sync_depth = 0

    @contextmanager
    def no_sync():
        nonlocal sync_depth
        sync_depth += 1
        try:
            yield
        finally:
            sync_depth -= 1

    config = SimpleNamespace(
        attn_res_impl='source',
        enable_attention_residuals=True,
        moe_paged_stash=False,
        timers=None,
        no_sync_func=no_sync,
        grad_sync_func=None,
        param_sync_func=None,
        overlap_moe_expert_parallel_comm=False,
        overlap_p2p_comm=overlap,
        overlap_p2p_comm_warmup_flush=overlap,
        batch_p2p_comm=not overlap,
        virtual_pipeline_model_parallel_size=virtual_size,
        microbatch_group_size_per_vp_stage=2,
        pipeline_dtype=torch.bfloat16,
        hidden_size=16,
        variable_seq_lengths=False,
        sequence_parallel=False,
        num_microbatches_with_partial_activation_checkpoints=None,
        enable_autocast=False,
        deallocate_pipeline_outputs=True,
        grad_scale_func=None,
        calculate_per_token_loss=False,
        defer_embedding_wgrad_compute=False,
        fine_grained_activation_offloading=False,
        cuda_graph_impl='none',
    )

    def prepare(model, config, group, only_forward):
        assert not events
        assert only_forward == forward_only
        events.append('prepare')

    def finalize_projection(model, config, group, only_forward):
        if only_forward:
            return
        assert sync_depth == 0
        assert not pending
        events.append('projection-finalize')

    def finalize_model(*args, **kwargs):
        assert events[-1] == 'projection-finalize'
        events.append('model-finalize')

    config.finalize_model_grads_func = finalize_model
    monkeypatch.setattr(schedules, '_prepare_attn_res_projection', prepare)
    monkeypatch.setattr(schedules, '_finalize_attn_res_projection', finalize_projection)
    monkeypatch.setattr(
        attention_residual, 'attn_res_source_cache_reset', lambda: events.append('cache-reset')
    )
    monkeypatch.setattr(attention_residual, 'attn_res_uniform_payload_slices', lambda config: 1)
    monkeypatch.setattr(
        attention_residual,
        'attn_res_projection_payload_shape',
        lambda config, seq, batch, pp_rank, vp_stage=None: (5, seq, batch),
    )
    monkeypatch.setattr(
        schedules,
        'get_tensor_shapes',
        lambda **kwargs: (
            _specs()[:1]
            if (kwargs['is_recv'] and rank == 0)
            or (not kwargs['is_recv'] and rank == pipeline_size - 1)
            else _specs()
        ),
    )
    monkeypatch.setattr(schedules, 'get_model_config', lambda model: model.config)
    monkeypatch.setattr(schedules, 'get_model_type', lambda model: None)
    monkeypatch.setattr(
        schedules, 'get_attr_wrapped_model', lambda model, name: getattr(model, name)
    )
    monkeypatch.setattr(schedules, 'set_current_microbatch', lambda *args: None)
    original_zeros = torch.zeros

    def zeros(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, 'zeros', zeros)

    class Group:
        def __init__(self, group_rank, size):
            self.group_rank, self.group_size = group_rank, size

        def rank(self):
            return self.group_rank

        def size(self):
            return self.group_size

    # These tests supply explicit group doubles without initializing global
    # distributed state; stage decisions must use those supplied groups.
    for module in (p2p, schedules):
        monkeypatch.setattr(module, 'is_pp_first_stage', lambda group: group.rank() == 0)
        monkeypatch.setattr(
            module, 'is_pp_last_stage', lambda group: group.rank() == group.size() - 1
        )
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = Group(0, 1)
    groups.pp = Group(rank, pipeline_size)
    communicator = p2p.P2PCommunicator.__new__(p2p.P2PCommunicator)
    communicator.config = config
    communicator.pp_group = groups.pp
    communicator.virtual_pipeline_model_parallel_size = virtual_size

    class Work:
        def __init__(self, event):
            self.event = event
            pending.add(self)

        def wait(self):
            pending.discard(self)
            events.append('wait-' + self.event)

    def communicate(
        *, tensor_send_next, tensor_send_prev, recv_prev, recv_next, tensor_shape, wait_on_reqs=True
    ):
        assert events[:2] == ['prepare', 'cache-reset']
        shapes = [tensor_shape] if p2p.is_single_shape(tensor_shape) else tensor_shape
        handles = {}
        for name, present in [
            ('send_next', tensor_send_next is not None),
            ('send_prev', tensor_send_prev is not None),
            ('recv_prev', recv_prev),
            ('recv_next', recv_next),
        ]:
            if present:
                events.append(name)
                handles[name] = Work(name)

        def receive(enabled):
            if not enabled:
                return None
            tensors = [
                torch.ones(spec.shape, dtype=spec.dtype, requires_grad=True) for spec in shapes
            ]
            return tensors[0] if p2p.is_single_shape(tensor_shape) else tensors

        if wait_on_reqs:
            for handle in handles.values():
                handle.wait()
            handles = None
        return receive(recv_prev), receive(recv_next), handles

    communicator._communicate = communicate

    class Model(torch.nn.Module):
        def __init__(self, chunk):
            super().__init__()
            self.config = config
            self.vp_stage = chunk if virtual_size is not None else None
            self.weight = torch.nn.Parameter(torch.tensor(0.25))
            self.input_tensor = None

        def set_input_tensor(self, tensors):
            self.input_tensor = tensors

    models = [Model(chunk) for chunk in range(virtual_size or 1)]

    def forward(data, model):
        assert events[:2] == ['prepare', 'cache-reset']
        source = model.input_tensor[0]
        if source is None:
            source = torch.ones(3, 2, 16, dtype=torch.bfloat16)
        value = (source * model.weight).clone()
        scores = value.float().sum(-1).unsqueeze(0).expand(5, 3, 2).clone()
        terminal = rank == pipeline_size - 1 and model is models[-1]
        output = value.float().sum() + scores.sum() if terminal else [value, scores]
        return output, lambda loss: (loss, {'loss': loss.detach()})

    function = (
        schedules.forward_backward_no_pipelining
        if pipeline_size == 1
        else (
            schedules.forward_backward_pipelining_with_interleaving
            if virtual_size is not None
            else schedules.forward_backward_pipelining_without_interleaving
        )
    )
    function(
        forward_step_func=forward,
        data_iterator=[None] * len(models),
        model=models,
        num_microbatches=2,
        seq_length=3,
        micro_batch_size=2,
        forward_only=forward_only,
        p2p_communicator=communicator,
        pg_collection=groups,
    )
    assert events[:2] == ['prepare', 'cache-reset']
    assert not pending
    if forward_only:
        assert 'projection-finalize' not in events
    else:
        assert events[-2:] == ['projection-finalize', 'model-finalize']
        assert all(model.weight.grad is not None for model in models)
