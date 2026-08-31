# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Full iteration CUDA graph for training."""

import gc
import logging

import torch

from megatron.core.tensor_parallel.random import get_all_rng_states

logger = logging.getLogger(__name__)

# Process-wide handle so full-iter and optimizer graph captures share one pool and one
# non-default stream (per-stream alloc segments can inflate memory_reserved; see
# tools/debug_cuda_graph_pool_memory*.py).
_shared_graph_pool = None
_shared_capture_stream = None


def get_shared_capture_stream():
    """Return one `torch.cuda.Stream` for all full-iter and optimizer graph captures.

    Call after the target CUDA device is selected.
    """
    global _shared_capture_stream
    if _shared_capture_stream is None:
        _shared_capture_stream = torch.cuda.Stream()
    return _shared_capture_stream


def get_shared_graph_pool():
    """Return a process-wide handle so all call sites share one graph memory pool.

    `torch.cuda.graph_pool_handle()` returns a new pool each time; this lazy singleton
    ensures e.g. full-iteration and optimizer captures reuse the same pool.
    """
    global _shared_graph_pool
    if _shared_graph_pool is None:
        _shared_graph_pool = torch.cuda.graph_pool_handle()
    return _shared_graph_pool


def get_graph_pool(use_single_mempool):
    """Return graph pool handle for full-iter/optimizer graph capture.

    When `use_single_mempool` is True, train/eval and optimizer captures reuse one
    process-wide pool. Otherwise, each capture call gets a new pool handle.
    """
    if use_single_mempool:
        return get_shared_graph_pool()
    return torch.cuda.graph_pool_handle()


# The below functions traverse through nested data structures (tuples, lists, dicts)
# present in src and creates a deep copy where all PyTorch tensors are cloned,
# detached from the computation graph, and moved to CUDA device. Non-tensor objects
# are returned as-is.


def copy_tensors_in_struct(src):
    """Copy src to new tensors."""
    if isinstance(src, tuple):
        return tuple(copy_tensors_in_struct(i) for i in src)
    elif isinstance(src, list):
        return list(copy_tensors_in_struct(i) for i in src)
    elif isinstance(src, dict):
        return {k: copy_tensors_in_struct(src[k]) for k in src}
    elif isinstance(src, torch.Tensor):
        return src.clone().detach().cuda()
    else:
        return src


def clone_tensors_in_struct(tgt, src, path="inputs"):
    """Copy ``src`` into graph-static ``tgt`` tensors.

    Once a CUDA Graph input buffer exists, changing its structure, tensor
    shape, or dtype would either invalidate the captured addresses or replay
    stale data. Reject those changes explicitly instead of reallocating a
    replacement that the captured graph cannot observe.
    """
    if isinstance(src, tuple):
        if not isinstance(tgt, tuple) or len(tgt) != len(src):
            raise RuntimeError(
                f"Full-iteration CUDA Graph static input structure changed at {path}: "
                f"expected tuple length {len(tgt) if isinstance(tgt, tuple) else 'n/a'}, "
                f"got {len(src)}."
            )
        result = []
        for i, (target_value, source_value) in enumerate(zip(tgt, src)):
            if isinstance(source_value, (tuple, list, dict, torch.Tensor)):
                result.append(clone_tensors_in_struct(target_value, source_value, f"{path}[{i}]"))
            elif type(target_value) is not type(source_value) or target_value != source_value:
                raise RuntimeError(
                    f"Full-iteration CUDA Graph non-tensor input changed at {path}[{i}]: "
                    f"expected {target_value!r}, got {source_value!r}."
                )
            else:
                result.append(target_value)
        return tuple(result)
    elif isinstance(src, list):
        if not isinstance(tgt, list) or len(tgt) != len(src):
            raise RuntimeError(
                f"Full-iteration CUDA Graph static input structure changed at {path}: "
                f"expected list length {len(tgt) if isinstance(tgt, list) else 'n/a'}, "
                f"got {len(src)}."
            )
        for i in range(len(src)):
            if isinstance(src[i], (tuple, list, dict, torch.Tensor)):
                tgt[i] = clone_tensors_in_struct(tgt[i], src[i], f"{path}[{i}]")
            elif type(tgt[i]) is not type(src[i]) or tgt[i] != src[i]:
                raise RuntimeError(
                    f"Full-iteration CUDA Graph non-tensor input changed at {path}[{i}]: "
                    f"expected {tgt[i]!r}, got {src[i]!r}."
                )
        return tgt
    elif isinstance(src, dict):
        if not isinstance(tgt, dict):
            raise RuntimeError(
                f"Full-iteration CUDA Graph static input structure changed at {path}: "
                f"expected {type(tgt).__name__}, got dict."
            )
        if tgt.keys() != src.keys():
            raise RuntimeError(
                f"Full-iteration CUDA Graph static input keys changed at {path}: "
                f"expected {sorted(tgt)}, got {sorted(src)}."
            )
        for k in src:
            if isinstance(src[k], (tuple, list, dict, torch.Tensor)):
                tgt[k] = clone_tensors_in_struct(tgt[k], src[k], f"{path}.{k}")
            elif type(tgt[k]) is not type(src[k]) or tgt[k] != src[k]:
                raise RuntimeError(
                    f"Full-iteration CUDA Graph non-tensor input changed at {path}.{k}: "
                    f"expected {tgt[k]!r}, got {src[k]!r}."
                )
        return tgt
    elif isinstance(src, torch.Tensor):
        if not isinstance(tgt, torch.Tensor):
            raise RuntimeError(
                f"Full-iteration CUDA Graph static input type changed at {path}: "
                f"expected {type(tgt).__name__}, got Tensor."
            )
        if tgt.shape != src.shape or tgt.dtype != src.dtype or tgt.layout != src.layout:
            raise RuntimeError(
                f"Full-iteration CUDA Graph static tensor metadata changed at {path}: "
                f"expected shape={tuple(tgt.shape)}, dtype={tgt.dtype}, layout={tgt.layout}; "
                f"got shape={tuple(src.shape)}, dtype={src.dtype}, layout={src.layout}."
            )
        tgt.copy_(src, non_blocking=True)
        return tgt
    else:
        raise TypeError(f"Expected a tensor container at {path}, got {type(src)}")


# Class to copy dataloader output to static CUDA tensors for CUDA graph input. This
# maintains separate static buffers for training and validation CUDA graphs.
class StaticBufferLoader:
    """Load data to static buffers."""

    static_buffers: dict = {'training': {}, 'validation': {}}

    def __init__(self):
        self.stream = torch.cuda.Stream()

    def __call__(self, inputs, stage, microbatch, model_chunk=0):
        assert stage in ['training', 'validation']
        assert isinstance(microbatch, int) and microbatch >= 0
        assert isinstance(model_chunk, int) and model_chunk >= 0
        if isinstance(inputs, tuple) and inputs and isinstance(inputs[0], dict):
            inputs = inputs[0]

        assert isinstance(inputs, dict)
        buffer_key = (model_chunk, microbatch)
        stage_buffers = StaticBufferLoader.static_buffers[stage]
        if buffer_key not in stage_buffers:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                stage_buffers[buffer_key] = copy_tensors_in_struct(inputs)
        else:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                clone_tensors_in_struct(
                    stage_buffers[buffer_key],
                    inputs,
                    path=f"{stage}[model_chunk={model_chunk},microbatch={microbatch}]",
                )
        torch.cuda.current_stream().wait_stream(self.stream)
        # Batch consumers may add metadata keys or replace tensor values with
        # local views (for example, static THD context-parallel slicing).  Keep
        # the graph-owned container immutable while preserving the identities
        # of the static tensors that CUDA Graph replay observes.
        return dict(stage_buffers[buffer_key])


class FullCudaGraphWrapper:
    """Wrapper class to enable FullIterationCUDAgraph."""

    curr_iteration = {'training': 0, 'validation': 0}
    cuda_graph = {'training': None, 'validation': None}
    result = {'training': None, 'validation': None}
    captured_num_microbatches = {'training': None, 'validation': None}

    def __init__(self, forward_backward_func, cuda_graph_warmup_steps=1, use_single_mempool=False):
        self.forward_backward_func = forward_backward_func
        self.static_loader = StaticBufferLoader()
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        self.use_single_mempool = use_single_mempool

    def data_read(self, data_iterator, model, training, num_microbatches):
        """Read all microbatch inputs from Dataloader and copy to static buffers."""
        if not isinstance(model, list) or len(model) == 1:
            assert not isinstance(data_iterator, list) or len(data_iterator) == 1
            iterator0 = data_iterator if not isinstance(data_iterator, list) else data_iterator[0]
            data_list = []
            if iterator0 is not None:
                for b in range(num_microbatches):
                    data_list.append(
                        self.static_loader(
                            next(iterator0),
                            'training' if training else 'validation',
                            b,
                            model_chunk=0,
                        )
                    )
                data_list = [iter(data_list)]
            else:
                data_list.append(None)
        else:
            assert isinstance(data_iterator, list) and len(data_iterator) == len(model)
            data_list = []
            for i in range(len(model)):
                if data_iterator[i] is not None:
                    data_list_i = []
                    for b in range(num_microbatches):
                        data_list_i.append(
                            self.static_loader(
                                next(data_iterator[i]),
                                'training' if training else 'validation',
                                b,
                                model_chunk=i,
                            )
                        )
                    data_list.append(iter(data_list_i))
                else:
                    data_list.append(None)
        return data_list

    def __call__(self, *args, **kwargs):
        assert len(args) == 0, 'forward_backward_func does not accept positional args'
        assert all(
            [
                kwarg in kwargs
                for kwarg in [
                    'model',
                    'data_iterator',
                    'num_microbatches',
                    'seq_length',
                    'forward_only',
                ]
            ]
        )
        model = kwargs['model']
        num_microbatches = kwargs['num_microbatches']

        training = not kwargs['forward_only']
        training_str = 'training' if training else 'validation'
        captured_num_microbatches = FullCudaGraphWrapper.captured_num_microbatches[training_str]
        if (
            FullCudaGraphWrapper.cuda_graph[training_str] is not None
            and captured_num_microbatches != num_microbatches
        ):
            raise RuntimeError(
                f"Full-iteration CUDA Graph captured {captured_num_microbatches} "
                f"{training_str} microbatches, but the current iteration produced "
                f"{num_microbatches}. CUDA Graph replay requires a fixed iteration schedule."
            )
        data_iterator = kwargs['data_iterator']
        data_list = self.data_read(data_iterator, model, training, num_microbatches)
        kwargs['data_iterator'] = data_list

        curr_iteration = self.curr_iter(training_str)
        if curr_iteration == self.cuda_graph_warmup_steps:
            logger.info(f'Capture CUDA graph for {training_str}!!!')
            if hasattr(torch.autograd.graph, 'set_override_stale_capture_stream'):
                torch.autograd.graph.set_override_stale_capture_stream(True)
            else:
                logger.warning(
                    'torch.autograd.graph.set_override_stale_capture_stream is not '
                    'available in this PyTorch version; CUDA graph capture may fail '
                    'if autograd nodes hold stale references to non-capturing streams. '
                    'Upgrade to a PyTorch build that includes pytorch/pytorch#180090.'
                )
            torch.distributed.barrier()
            # Release cached blocks reserved during the eager warmup iterations
            # before the capture allocates its private pool: the two pools
            # coexist for the lifetime of the graph, and warmup fragmentation
            # (reserved-but-unallocated blocks) otherwise counts against the
            # capture's headroom.
            gc.collect()
            torch.cuda.empty_cache()
            assert FullCudaGraphWrapper.cuda_graph[training_str] is None
            FullCudaGraphWrapper.cuda_graph[training_str] = torch.cuda.CUDAGraph()
            for _, state in get_all_rng_states().items():
                FullCudaGraphWrapper.cuda_graph[training_str].register_generator_state(state)
            torch.cuda.synchronize()
            capture_stream = get_shared_capture_stream()
            with torch.cuda.graph(
                FullCudaGraphWrapper.cuda_graph[training_str],
                stream=capture_stream,
                pool=get_graph_pool(self.use_single_mempool),
                capture_error_mode="thread_local",
            ):
                FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(
                    *args, **kwargs
                )
            torch.cuda.synchronize()
            torch.distributed.barrier()
            FullCudaGraphWrapper.captured_num_microbatches[training_str] = num_microbatches
            logger.info(f'CUDA graph capture done for {training_str}!!!')
        if FullCudaGraphWrapper.cuda_graph[training_str] is None:
            FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(*args, **kwargs)
        else:
            FullCudaGraphWrapper.cuda_graph[training_str].replay()
        self.next_iter(training_str)
        return FullCudaGraphWrapper.result[training_str]

    def curr_iter(self, stage):
        """Return current training/validation iteration."""
        return FullCudaGraphWrapper.curr_iteration[stage]

    def next_iter(self, stage):
        """Increment current training/validation iteration."""
        FullCudaGraphWrapper.curr_iteration[stage] += 1

    def reset_cuda_graph(self, stage=None):
        """Reset CUDA graph."""
        if stage is None or stage == 'training':
            if FullCudaGraphWrapper.cuda_graph['training'] is not None:
                del FullCudaGraphWrapper.cuda_graph['training']
                FullCudaGraphWrapper.cuda_graph['training'] = None
            FullCudaGraphWrapper.result['training'] = None
            FullCudaGraphWrapper.curr_iteration['training'] = 0
            FullCudaGraphWrapper.captured_num_microbatches['training'] = None
        if stage is None or stage == 'validation':
            if FullCudaGraphWrapper.cuda_graph['validation'] is not None:
                del FullCudaGraphWrapper.cuda_graph['validation']
                FullCudaGraphWrapper.cuda_graph['validation'] = None
            FullCudaGraphWrapper.result['validation'] = None
            FullCudaGraphWrapper.curr_iteration['validation'] = 0
            FullCudaGraphWrapper.captured_num_microbatches['validation'] = None
        gc.collect()
