"""Executor-owned CUDA graphs for the shared RSBF injection adapter."""

import torch
from torch import nn
from torch.nn import functional as F


class _OwnedGraphGradient(torch.autograd.Function):
    """Give each consumer its own gradient before the graph reuses its return slot."""

    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, grad):
        return grad.clone()


class RSBFAdapterLinear(nn.Linear):
    """Keep the normal module call and its DDP hooks when executing a captured adapter."""

    def forward(self, value, *, rsbf_graph=None):
        if rsbf_graph is None or not value.requires_grad or not torch.is_grad_enabled():
            return super().forward(value)
        return rsbf_graph(value, self.weight)


class AdapterGraphs:
    """One private forward/backward graph per input geometry, owned by the loop program."""

    def __init__(self):
        self.graphs = {}
        self.step_forwards = 0

    def __call__(self, value, weight):
        if not value.is_cuda:
            raise ValueError("loop_rsbf_graph_adapter requires CUDA")
        autocast = torch.is_autocast_enabled("cuda")
        dtype = torch.get_autocast_dtype("cuda") if autocast else None
        key = (value.device, value.dtype, value.shape, weight.dtype, weight.shape, dtype)
        graph = self.graphs.get(key)
        if graph is None:
            # Private samples avoid retaining a sample or a live parameter-store compute view.
            args = tuple(t.detach().clone().requires_grad_() for t in (value, weight))
            with torch.autocast("cuda", enabled=autocast, dtype=dtype, cache_enabled=False):
                graph = torch.cuda.make_graphed_callables(F.linear, args)
            self.graphs[key] = graph
        self.step_forwards += 1
        return graph(_OwnedGraphGradient.apply(value), _OwnedGraphGradient.apply(weight)).clone()
