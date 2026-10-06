# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.utils import nvtx_decorator

# Types
Shape = Union[List[int], Tuple[int, ...], torch.Size]


@dataclass(frozen=True)
class PipelineTensorSpec:
    """Shape and wire dtype of one pipeline channel.

    Plain shapes continue to use ``config.pipeline_dtype``. Explicit specs let
    auxiliary channels retain their own precision in forward and backward.
    """

    shape: Shape
    dtype: torch.dtype


PipelineTensor = Optional[torch.Tensor]
PipelinePayload = Union[PipelineTensor, List[PipelineTensor]]
PipelineSendPayload = Union[PipelinePayload, Tuple[PipelineTensor, ...]]
PipelineShape = Union[Shape, PipelineTensorSpec]
PipelineShapeSequence = Union[List[PipelineShape], Tuple[PipelineShape, ...]]
PipelineShapes = Union[PipelineShape, PipelineShapeSequence]


class _P2PWorkGroup:
    """One directional wait handle covering every tensor in a payload."""

    def __init__(self, requests):
        self.requests = tuple(requests)

    def wait(self):
        """Join all channel requests before the schedule reuses the payload."""
        for request in self.requests:
            request.wait()


P2PRequests = Union[List[dist.Work], dict[str, Union[dist.Work, _P2PWorkGroup]]]


def _tensor_list(tensor):
    if tensor is None:
        return []
    return tensor if isinstance(tensor, (list, tuple)) else [tensor]


def _append_p2p_ops(ops, op, tensors, peer, group):
    for tensor in _tensor_list(tensors):
        if tensor is not None:
            ops.append(torch.distributed.P2POp(op, tensor, peer, group))


def _batched_p2p_ops(
    *,
    tensor_send_prev: PipelineSendPayload,
    tensor_recv_prev: PipelinePayload,
    tensor_send_next: PipelineSendPayload,
    tensor_recv_next: PipelinePayload,
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
) -> List[dist.Work]:
    ops: List[dist.P2POp] = []
    if prev_pipeline_rank == next_pipeline_rank and group.rank() % 2 == 0:
        # With PP2, VPP forward and backward use the same peer and communicator.
        # Sends match receives in posting order, so the two ranks must visit
        # opposite directions first. Otherwise a forward tensor can be received
        # as a backward gradient (silently when their shapes and dtypes match).
        _append_p2p_ops(ops, torch.distributed.isend, tensor_send_next, next_pipeline_rank, group)
        _append_p2p_ops(ops, torch.distributed.irecv, tensor_recv_next, next_pipeline_rank, group)
        _append_p2p_ops(ops, torch.distributed.isend, tensor_send_prev, prev_pipeline_rank, group)
        _append_p2p_ops(ops, torch.distributed.irecv, tensor_recv_prev, prev_pipeline_rank, group)
    else:
        _append_p2p_ops(ops, torch.distributed.isend, tensor_send_prev, prev_pipeline_rank, group)
        _append_p2p_ops(ops, torch.distributed.irecv, tensor_recv_prev, prev_pipeline_rank, group)
        _append_p2p_ops(ops, torch.distributed.isend, tensor_send_next, next_pipeline_rank, group)
        _append_p2p_ops(ops, torch.distributed.irecv, tensor_recv_next, next_pipeline_rank, group)
    return torch.distributed.batch_isend_irecv(ops) if ops else []


def _p2p_ops(
    *,
    tensor_send_prev: PipelineSendPayload,
    tensor_recv_prev: PipelinePayload,
    tensor_send_next: PipelineSendPayload,
    tensor_recv_next: PipelinePayload,
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
) -> dict[str, Union[dist.Work, _P2PWorkGroup]]:
    reqs: dict[str, Union[dist.Work, _P2PWorkGroup]] = {}
    even_send_odd_recv_group = group
    if group.size() == 2 and torch.distributed.get_backend(group) != 'ucc':
        # Use the global process group for one of the two p2p communications
        # to allow the overlap of the independent communications.
        # Using the global process group is compatible because the pipeline-parallel
        # communications set the source and destination by global rank.
        # The only exception occurs when using the ‘ucc’ backend.
        # Because the global communicator always uses the ‘nccl’ backend,
        # we must ensure the else path is followed for the ‘ucc’ backend.
        even_recv_odd_send_group = torch.distributed.group.WORLD
    else:
        even_recv_odd_send_group = group

    def post(direction, tensors, peer, communication_group, *, send):
        requests: List[dist.Work] = []
        for tensor in _tensor_list(tensors):
            if tensor is None:
                continue
            if send:
                request = torch.distributed.isend(tensor, dst=peer, group=communication_group)
            else:
                request = torch.distributed.irecv(tensor, src=peer, group=communication_group)
            requests.append(request)
        if requests:
            reqs[direction] = requests[0] if len(requests) == 1 else _P2PWorkGroup(requests)

    if group.rank() % 2 == 0:
        post("send_next", tensor_send_next, next_pipeline_rank, even_send_odd_recv_group, send=True)
        post(
            "recv_prev", tensor_recv_prev, prev_pipeline_rank, even_recv_odd_send_group, send=False
        )
        post("send_prev", tensor_send_prev, prev_pipeline_rank, even_send_odd_recv_group, send=True)
        post(
            "recv_next", tensor_recv_next, next_pipeline_rank, even_recv_odd_send_group, send=False
        )
    else:
        post(
            "recv_prev", tensor_recv_prev, prev_pipeline_rank, even_send_odd_recv_group, send=False
        )
        post("send_next", tensor_send_next, next_pipeline_rank, even_recv_odd_send_group, send=True)
        post(
            "recv_next", tensor_recv_next, next_pipeline_rank, even_send_odd_recv_group, send=False
        )
        post("send_prev", tensor_send_prev, prev_pipeline_rank, even_recv_odd_send_group, send=True)
    return reqs


def is_single_shape(x) -> bool:
    """Check if the input is a single shape."""
    if isinstance(x, (torch.Size, PipelineTensorSpec)):
        return True
    if isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(d, int) for d in x):
        return True
    return False


class P2PCommunicator:
    """P2P (Point-to-Point) Communicator for pipeline parallelism.

    This class handles communication between pipeline stages by managing
    tensor exchanges between consecutive stages in the pipeline.
    """

    def __init__(self, pp_group: dist.ProcessGroup, config: ModelParallelConfig):
        # Basic attrs
        self.pp_group = pp_group
        self.config = config

        world_size = self.pp_group.size()
        curr_rank_in_pg = self.pp_group.rank()

        next_rank_pg = (curr_rank_in_pg + 1) % world_size
        prev_rank_pg = (curr_rank_in_pg - 1) % world_size

        self.next_rank: int = dist.get_global_rank(self.pp_group, next_rank_pg)
        self.prev_rank: int = dist.get_global_rank(self.pp_group, prev_rank_pg)
        self.virtual_pipeline_model_parallel_size = (
            config.virtual_pipeline_model_parallel_size
            if config.virtual_pipeline_model_parallel_size is not None
            else None
        )

    @property
    def is_pp_first_stage(self) -> bool:
        """Return True if pp first stage."""
        return is_pp_first_stage(self.pp_group)

    @property
    def is_pp_last_stage(self) -> bool:
        """Return True if pp last stage."""
        return is_pp_last_stage(self.pp_group)

    @property
    def total_stages(self) -> int:
        """Return total number of pipeline stages."""
        return self.pp_group.size()

    @property
    def current_stage(self) -> int:
        """Return current pipeline stage index (0-indexed)."""
        return self.pp_group.rank()

    def _communicate_shapes(self, tensor_send_next, tensor_send_prev, recv_prev, recv_next):
        """Communicate tensor shapes between stages. Used to communicate
        tensor shapes before the actual tensor communication happens.
        This is required when the sequence lengths across micro batches
        are not uniform.

        Args:
            tensor_send_next: tensor to send to next rank (no tensor sent if
                            set to None).
            tensor_send_prev: tensor to send to prev rank (no tensor sent if
                            set to None).
            recv_prev: boolean for whether tensor should be received from
                    previous rank.
            recv_next: boolean for whether tensor should be received from
                    next rank.
        Returns:
            (recv_prev_shape, recv_next_shape)
        """
        config = self.config
        recv_prev_shape_tensor = None
        recv_next_shape_tensor = None
        send_prev_shape_tensor = None
        send_next_shape_tensor = None
        if recv_prev:
            recv_prev_shape_tensor = torch.empty(
                (3,), device=torch.cuda.current_device(), dtype=torch.int64
            )
        if recv_next:
            recv_next_shape_tensor = torch.empty(
                (3,), device=torch.cuda.current_device(), dtype=torch.int64
            )
        if tensor_send_prev is not None:
            send_prev_shape_tensor = torch.tensor(
                tensor_send_prev.size(), device=torch.cuda.current_device(), dtype=torch.int64
            )
        if tensor_send_next is not None:
            send_next_shape_tensor = torch.tensor(
                tensor_send_next.size(), device=torch.cuda.current_device(), dtype=torch.int64
            )

        if config.use_ring_exchange_p2p:
            torch.distributed.ring_exchange(
                tensor_send_prev=send_prev_shape_tensor,
                tensor_recv_prev=recv_prev_shape_tensor,
                tensor_send_next=send_next_shape_tensor,
                tensor_recv_next=recv_next_shape_tensor,
                group=self.pp_group,
            )
        else:
            reqs = _batched_p2p_ops(
                tensor_send_prev=send_prev_shape_tensor,
                tensor_recv_prev=recv_prev_shape_tensor,
                tensor_send_next=send_next_shape_tensor,
                tensor_recv_next=recv_next_shape_tensor,
                group=self.pp_group,
                prev_pipeline_rank=self.prev_rank,
                next_pipeline_rank=self.next_rank,
            )
            for req in reqs:
                req.wait()

            # To protect against race condition when using batch_isend_irecv().
            # should take this out once the bug with batch_isend_irecv is resolved.
            torch.cuda.synchronize()

        recv_prev_shape = [0, 0, 0]
        if recv_prev_shape_tensor is not None:
            recv_prev_shape = recv_prev_shape_tensor.tolist()

        recv_next_shape = [0, 0, 0]
        if recv_next_shape_tensor is not None:
            recv_next_shape = recv_next_shape_tensor.tolist()

        return recv_prev_shape, recv_next_shape

    def _communicate(
        self,
        *,
        tensor_send_next: PipelineSendPayload,
        tensor_send_prev: PipelineSendPayload,
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Optional[PipelineShapes],
        wait_on_reqs: bool = True,
    ) -> Tuple[PipelinePayload, PipelinePayload, Optional[P2PRequests]]:
        """Communicate tensors between stages. Used as helper method in other
        communication methods that are used in megatron/schedules.py.

        Args:
            tensor_send_next (torch.Tensor, optional):
                Tensor to send to next rank (no tensor sent if None)

            tensor_send_prev (torch.Tensor, optional):
                Tensor to send to prev rank (no tensor sent if None)

            recv_prev (boolean, required):
                whether tensor should be received from previous rank.

            recv_next (boolean, required):
                whether tensor should be received from next rank.

            tensor_shape (shape, PipelineTensorSpec, or list, required):
                Receive shape and optional wire dtype, or one spec per channel.
                Corresponding forward/backward channels use the same spec.
                All channels are posted before any request is waited on.

            wait_on_reqs (boolean, optional, default=False):
                For non-batched p2p communication, wait on each request
                before returning.

        Returns:
            tuple containing

            - tensor_recv_prev: tensor or channel list if recv_prev is True, None otherwise.
            - tensor_recv_next: tensor or channel list if recv_next is True, None otherwise.
            - requests: outstanding work handles, or None after waiting for nonempty work.

        """

        config = self.config
        multiple = tensor_shape is not None and not is_single_shape(tensor_shape)
        specs: List[Optional[PipelineShape]] = (
            list(cast(PipelineShapeSequence, tensor_shape))
            if multiple
            else [cast(Optional[PipelineShape], tensor_shape)]
        )
        if tensor_shape is None:
            channels = max(len(_tensor_list(tensor_send_next)), len(_tensor_list(tensor_send_prev)))
            multiple = channels > 1
            if multiple:
                specs = [None] * channels
        if multiple:
            for payload in (tensor_send_next, tensor_send_prev):
                if payload is not None and len(_tensor_list(payload)) != len(specs):
                    raise ValueError("Pipeline payload and tensor specs must have equal lengths")

        if not specs or (
            not recv_prev
            and not recv_next
            and not any(
                tensor is not None
                for tensor in _tensor_list(tensor_send_next) + _tensor_list(tensor_send_prev)
            )
        ):
            # An empty outer shape list has zero channels, not one scalar
            # channel. Preserve this legacy no-op without allocating, posting
            # shape exchanges, or synchronizing CUDA. Empty asynchronous
            # requests retain their backend-specific container type.
            requests: P2PRequests = (
                [] if config.use_ring_exchange_p2p or config.batch_p2p_comm else {}
            )
            return ([] if recv_prev else None), ([] if recv_next else None), requests

        if not multiple:
            for payload in (tensor_send_next, tensor_send_prev):
                if len(_tensor_list(payload)) > 1:
                    raise ValueError("Multiple pipeline tensors require a spec for each channel")
            tensor_send_next = next(iter(_tensor_list(tensor_send_next)), None)
            tensor_send_prev = next(iter(_tensor_list(tensor_send_prev)), None)
        send_next = _tensor_list(tensor_send_next)
        send_prev = _tensor_list(tensor_send_prev)
        recv_prev_tensors: List[PipelineTensor] = []
        recv_next_tensors: List[PipelineTensor] = []
        for channel, spec in enumerate(specs):
            dtype = spec.dtype if isinstance(spec, PipelineTensorSpec) else config.pipeline_dtype
            shape = spec.shape if isinstance(spec, PipelineTensorSpec) else spec
            if isinstance(spec, PipelineTensorSpec):
                for payload in (send_next, send_prev):
                    if payload and payload[channel] is not None and payload[channel].dtype != dtype:
                        raise ValueError(
                            f"Pipeline channel {channel} has dtype {payload[channel].dtype}, "
                            f"expected {dtype}"
                        )
            if config.variable_seq_lengths or config.mtp_standalone:
                recv_prev_shape, recv_next_shape = self._communicate_shapes(
                    send_next[channel] if send_next else None,
                    send_prev[channel] if send_prev else None,
                    recv_prev,
                    recv_next,
                )
            else:
                recv_prev_shape = recv_next_shape = shape
            if (recv_prev or recv_next) and dtype is None:
                raise RuntimeError(
                    "pipeline_dtype or an explicit tensor dtype is required to receive"
                )
            if (recv_prev or recv_next) and shape is None:
                raise RuntimeError(
                    "tensor_shape must be specified when receiving a pipeline tensor"
                )

            def allocate(receive, receive_shape):
                return (
                    torch.empty(
                        receive_shape,
                        requires_grad=True,
                        device=torch.cuda.current_device(),
                        dtype=dtype,
                    )
                    if receive
                    else None
                )

            recv_prev_tensors.append(allocate(recv_prev, recv_prev_shape))
            recv_next_tensors.append(allocate(recv_next, recv_next_shape))
        tensor_recv_prev: PipelinePayload
        tensor_recv_next: PipelinePayload
        if not multiple:
            tensor_recv_prev = recv_prev_tensors[0]
            tensor_recv_next = recv_next_tensors[0]
        else:
            tensor_recv_prev = recv_prev_tensors if recv_prev else None
            tensor_recv_next = recv_next_tensors if recv_next else None

        # Send tensors in both the forward and backward directions as appropriate.
        p2p_func: Callable[..., P2PRequests]
        if config.use_ring_exchange_p2p:

            def _ring_exchange_wrapper(**kwargs) -> List[dist.Work]:
                # ring_exchange has no rank arguments and exchanges one channel at a time.
                kwargs.pop("prev_pipeline_rank")
                kwargs.pop("next_pipeline_rank")
                if multiple:
                    for channel in range(len(specs)):
                        channel_kwargs = {
                            key: (value[channel] if value is not None else None)
                            for key, value in kwargs.items()
                            if key != "group"
                        }
                        torch.distributed.ring_exchange(**channel_kwargs, group=kwargs["group"])
                else:
                    torch.distributed.ring_exchange(**kwargs)
                return []

            p2p_func = _ring_exchange_wrapper
        elif config.batch_p2p_comm:
            assert wait_on_reqs
            p2p_func = _batched_p2p_ops
        else:
            p2p_func = _p2p_ops

        pp_group = self.pp_group
        next_rank = self.next_rank
        prev_rank = self.prev_rank

        reqs: Optional[P2PRequests]
        if config.use_ring_exchange_p2p or config.batch_p2p_comm:
            reqs = []
        else:
            reqs = {}

        p2p_reqs = p2p_func(
            tensor_send_prev=tensor_send_prev,
            tensor_recv_prev=tensor_recv_prev,
            tensor_send_next=tensor_send_next,
            tensor_recv_next=tensor_recv_next,
            group=pp_group,
            prev_pipeline_rank=prev_rank,
            next_pipeline_rank=next_rank,
        )
        if isinstance(p2p_reqs, list):
            cast(List[dist.Work], reqs).extend(p2p_reqs)
        else:
            cast(dict[str, Union[dist.Work, _P2PWorkGroup]], reqs).update(p2p_reqs)

        if wait_on_reqs and len(reqs) > 0:
            for req in reqs if isinstance(reqs, list) else reqs.values():
                req.wait()
            reqs = None

        if (
            config.batch_p2p_comm
            and config.batch_p2p_sync
            and not torch.cuda.is_current_stream_capturing()
        ):
            # To protect against race condition when using batch_isend_irecv().
            # User should assert that we have a modern enough PyTorch to not need this.
            #
            # Skipped while a stream is capturing, which cuda_graph_impl=
            # "full_iteration" makes reachable by recording the whole pipeline
            # schedule. torch.cuda.synchronize() is a host-side device sync and is
            # illegal under capture -- measured, not assumed: it raises
            # "operation failed due to a previous error during capture", while the
            # batch_isend_irecv and the req.wait() above it both capture cleanly.
            # So no captured pipeline can carry this workaround in any form; the
            # only choice is to skip it or to reject capture outright.
            #
            # Skipping is the right choice because the workaround targets a bug in
            # older PyTorch, and capturing NCCL point-to-point work needs a far
            # newer stack than that -- any build that can reach this line under
            # capture is one the comment above says does not need the sync. This is
            # a version argument, not a stream-ordering one: the race is cross-rank,
            # so do not carry this skip over to a non-captured path by reasoning
            # that local stream order stands in for it.
            torch.cuda.synchronize()

        return tensor_recv_prev, tensor_recv_next, reqs

    @nvtx_decorator()
    def recv_forward(self, tensor_shapes, is_first_stage: bool) -> PipelinePayload:
        """Receive all forward channels before waiting for completion."""
        single = is_single_shape(tensor_shapes)
        if is_first_stage:
            return None if single else [None] * len(tensor_shapes)
        if self.config.timers is not None:
            self.config.timers('forward-recv', log_level=2).start()
        input_tensor, _, _ = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shapes,
        )
        if self.config.timers is not None:
            self.config.timers('forward-recv').stop()
        return input_tensor

    @nvtx_decorator()
    def recv_backward(self, tensor_shapes, is_last_stage: bool) -> PipelinePayload:
        """Receive the gradient of every output channel."""
        single = is_single_shape(tensor_shapes)
        if is_last_stage:
            return None if single else [None] * len(tensor_shapes)
        if self.config.timers is not None:
            self.config.timers('backward-recv', log_level=2).start()
        _, output_tensor_grad, _ = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shapes,
        )
        if self.config.timers is not None:
            self.config.timers('backward-recv').stop()
        return output_tensor_grad

    @nvtx_decorator()
    def send_forward(self, output_tensors, is_last_stage: bool) -> None:
        """Send every output channel to the next stage."""
        if is_last_stage:
            return
        if self.config.timers is not None:
            self.config.timers('forward-send', log_level=2).start()
        self._communicate(
            tensor_send_next=output_tensors,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
        )
        if self.config.timers is not None:
            self.config.timers('forward-send').stop()

    @nvtx_decorator()
    def send_backward(self, input_tensor_grads, is_first_stage: bool) -> None:
        """Send every input-channel gradient to the previous stage."""
        if is_first_stage:
            return
        if self.config.timers is not None:
            self.config.timers('backward-send', log_level=2).start()
        self._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grads,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
        )
        if self.config.timers is not None:
            self.config.timers('backward-send').stop()

    @nvtx_decorator()
    def send_forward_recv_backward(
        self, output_tensors, tensor_shapes, is_last_stage: bool
    ) -> PipelinePayload:
        """Send outputs and receive their gradients as one multi-channel exchange.

        Waiting on one channel before sending the rest would deadlock: the peer
        cannot run backward until it has received the complete forward payload.
        """
        single = not isinstance(output_tensors, (list, tuple))
        if is_last_stage:
            return None if single else [None] * len(output_tensors)
        if self.config.timers is not None:
            self.config.timers('forward-send-backward-recv', log_level=2).start()
        _, output_tensor_grad, _ = self._communicate(
            tensor_send_next=output_tensors,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shapes,
        )
        if self.config.timers is not None:
            self.config.timers('forward-send-backward-recv').stop()
        if single and isinstance(output_tensor_grad, list):
            return output_tensor_grad[0] if output_tensor_grad else []
        if not single and not isinstance(output_tensor_grad, list):
            return [output_tensor_grad]
        return output_tensor_grad

    @nvtx_decorator()
    def send_backward_recv_forward(
        self, input_tensor_grads, tensor_shapes, is_first_stage: bool
    ) -> PipelinePayload:
        """Send input gradients and receive the next complete forward payload."""
        single = not isinstance(input_tensor_grads, (list, tuple))
        if is_first_stage:
            return None if single else [None] * len(input_tensor_grads)
        if self.config.timers is not None:
            self.config.timers('backward-send-forward-recv', log_level=2).start()
        input_tensor, _, _ = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grads,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shapes,
        )
        if self.config.timers is not None:
            self.config.timers('backward-send-forward-recv').stop()
        if single and isinstance(input_tensor, list):
            return input_tensor[0] if input_tensor else []
        if not single and not isinstance(input_tensor, list):
            return [input_tensor]
        return input_tensor

    @nvtx_decorator()
    def send_forward_recv_forward(
        self,
        output_tensor: PipelineSendPayload,
        recv_prev: bool,
        tensor_shape: PipelineShapes,
        overlap_p2p_comm: bool = False,
    ) -> Union[PipelinePayload, Tuple[PipelinePayload, Optional[P2PRequests]]]:
        """Batched recv from previous rank and send to next rank in pipeline."""
        config = self.config
        if config.timers is not None:
            config.timers('forward-send-forward-recv', log_level=2).start()
        input_tensor, _, wait_handles = self._communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=False,
            tensor_shape=tensor_shape,
            wait_on_reqs=(not overlap_p2p_comm),
        )
        if config.timers is not None:
            config.timers('forward-send-forward-recv').stop()
        if overlap_p2p_comm:
            return input_tensor, wait_handles
        return input_tensor

    @nvtx_decorator()
    def send_backward_recv_backward(
        self,
        input_tensor_grad: PipelineSendPayload,
        recv_next: bool,
        tensor_shape: PipelineShapes,
        overlap_p2p_comm: bool = False,
    ) -> Union[PipelinePayload, Tuple[PipelinePayload, Optional[P2PRequests]]]:
        """Batched recv from next rank and send to previous rank in pipeline."""
        config = self.config
        if config.timers is not None:
            config.timers('backward-send-backward-recv', log_level=2).start()
        _, output_tensor_grad, wait_handles = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            wait_on_reqs=(not overlap_p2p_comm),
        )
        if config.timers is not None:
            config.timers('backward-send-backward-recv').stop()
        if overlap_p2p_comm:
            return output_tensor_grad, wait_handles
        return output_tensor_grad

    @nvtx_decorator()
    def send_forward_backward_recv_forward_backward(
        self,
        output_tensor: PipelineSendPayload,
        input_tensor_grad: PipelineSendPayload,
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: PipelineShapes,
    ) -> Tuple[PipelinePayload, PipelinePayload]:
        """Batched send and recv with previous and next ranks in pipeline."""
        config = self.config
        if config.timers is not None:
            config.timers('forward-backward-send-forward-backward-recv', log_level=2).start()
        input_tensor, output_tensor_grad, _ = self._communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=input_tensor_grad,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
        )
        if config.timers is not None:
            config.timers('forward-backward-send-forward-backward-recv').stop()
        return input_tensor, output_tensor_grad
