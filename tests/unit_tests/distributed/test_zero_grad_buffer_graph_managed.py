# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""zero_grad_buffer keeps grad_added_to_main_grad for graph-managed parameters.

A CUDA graph whose backward accumulates into ``main_grad`` cannot set
``grad_added_to_main_grad`` again on replay. ``zero_grad_buffer`` already keeps the flag for
every parameter under ``cuda_graph_impl='transformer_engine'``; a parameter marked
``graph_managed_main_grad`` gets the same treatment under any impl.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _param(graph_managed: bool) -> torch.nn.Parameter:
    param = torch.nn.Parameter(torch.zeros(2))
    param.grad_added_to_main_grad = True
    if graph_managed:
        param.graph_managed_main_grad = True
    return param


def _ddp_stub(cuda_graph_impl: str, params):
    return SimpleNamespace(
        config=SimpleNamespace(cuda_graph_impl=cuda_graph_impl),
        params_with_grad=params,
        buffers=[],
        expert_parallel_buffers=[],
        bucket_groups=[],
        expert_parallel_bucket_groups=[],
    )


@pytest.mark.parametrize("cuda_graph_impl", ["none", "local", "full_iteration"])
def test_zero_grad_buffer_keeps_flag_for_graph_managed_params(cuda_graph_impl):
    managed, plain = _param(graph_managed=True), _param(graph_managed=False)
    DistributedDataParallel.zero_grad_buffer(_ddp_stub(cuda_graph_impl, [managed, plain]))
    assert managed.grad_added_to_main_grad is True
    assert plain.grad_added_to_main_grad is False


def test_zero_grad_buffer_false_graph_managed_flag_resets():
    param = _param(graph_managed=False)
    param.graph_managed_main_grad = False
    DistributedDataParallel.zero_grad_buffer(_ddp_stub("none", [param]))
    assert param.grad_added_to_main_grad is False


def test_zero_grad_buffer_te_impl_resets_no_param():
    # The existing global rule for TE graphs is unchanged.
    managed, plain = _param(graph_managed=True), _param(graph_managed=False)
    DistributedDataParallel.zero_grad_buffer(_ddp_stub("transformer_engine", [managed, plain]))
    assert managed.grad_added_to_main_grad is True
    assert plain.grad_added_to_main_grad is True


def test_zero_grad_buffer_without_cuda_graph_impl_attribute_resets():
    # Configs without the attribute (getattr default 'none') reset unflagged params.
    managed, plain = _param(graph_managed=True), _param(graph_managed=False)
    stub = _ddp_stub("none", [managed, plain])
    stub.config = SimpleNamespace()
    DistributedDataParallel.zero_grad_buffer(stub)
    assert managed.grad_added_to_main_grad is True
    assert plain.grad_added_to_main_grad is False


class _TwoLinear(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim)
        self.linear2 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


@pytest.fixture(scope="module")
def _model_parallel():
    # Initialized once per module (CI process-group budget); only the GPU test needs it.
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DDP grad buffers need CUDA")
def test_ddp_graph_managed_param_keeps_flag_and_skips_hook_add(_model_parallel):
    dim = 8
    torch.manual_seed(0)
    module = _TwoLinear(dim).cuda()
    reference = _TwoLinear(dim).cuda()
    reference.load_state_dict(module.state_dict())

    config = TransformerConfig(num_attention_heads=1, num_layers=1)
    ddp = DistributedDataParallel(config, ddp_config=DistributedDataParallelConfig(), module=module)
    managed = module.linear1.weight
    managed.graph_managed_main_grad = True
    plain = [p for p in module.parameters() if p is not managed]

    # A previous step left every flag set (a graph-managed param would have been accumulated
    # into main_grad by its graph; a plain param by TE's fused wgrad).
    for param in module.parameters():
        param.grad_added_to_main_grad = True
    ddp.zero_grad_buffer()
    assert managed.grad_added_to_main_grad is True
    assert all(param.grad_added_to_main_grad is False for param in plain)
    for param in module.parameters():
        assert torch.count_nonzero(param.main_grad) == 0

    # Backward produces an ordinary .grad for every parameter. The DDP hook must not add the
    # graph-managed one (its graph owns main_grad) and must add every other one.
    x = torch.randn(4, dim, device="cuda")
    ddp(x).sum().backward()
    reference(x).sum().backward()

    assert torch.count_nonzero(managed.main_grad) == 0
    assert managed.grad is None
    reference_grads = dict(reference.named_parameters())
    for name, param in module.named_parameters():
        if param is managed:
            continue
        torch.testing.assert_close(
            param.main_grad, reference_grads[name].grad.to(param.main_grad.dtype)
        )
