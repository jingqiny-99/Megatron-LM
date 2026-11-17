# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The DSA indexer's FP32 weights projection (``_DSAWeightsProjection``) with fused WGrad
accumulation, on CPU (torch.mm path; no TE):

* one call carrying ``is_first_microbatch=True`` writes over ``main_grad`` (a fresh microbatch);
* a module called again before the first call's backward (a loop core run several times per
  microbatch, a repeated MTP layer) adds every call's gradient: the first call's backward runs
  last, and its overwrite would drop the later calls' gradients;
* a layer marked ``is_repeated_layer`` never overwrites;
* the next microbatch accumulates.
"""

from types import SimpleNamespace

import torch

from megatron.core.transformer.experimental_attention_variant.dsa import _DSAWeightsProjection

IN, OUT, TOKENS = 16, 4, 6


class _Linear(torch.nn.Module):
    """Stand-in for the indexer's ``linear_weights_proj`` with fused WGrad accumulation."""

    def __init__(self, repeated=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(OUT, IN, dtype=torch.bfloat16))
        self.weight.main_grad = torch.zeros(OUT, IN, dtype=torch.float32)
        self.weight.grad_added_to_main_grad = False
        self.fuse_wgrad_accumulation = True
        self.is_first_microbatch = True
        if repeated:
            self.is_repeated_layer = True


def _indexer():
    return SimpleNamespace(_weights_proj_te_gemm_supported=False)


def _inputs(count, seed):
    generator = torch.Generator().manual_seed(seed)
    xs = [torch.randn(TOKENS, IN, generator=generator).bfloat16() for _ in range(count)]
    grads = [torch.randn(TOKENS, OUT, generator=generator) for _ in range(count)]
    return xs, grads


def _wgrad(xs, grads):
    return sum(g.t().float() @ x.float() for x, g in zip(xs, grads))


def _microbatch(linear, xs, grads):
    """Every call's forward first, then one backward through all of them (a microbatch whose
    module is called ``len(xs)`` times)."""
    indexer = _indexer()
    outputs = [_DSAWeightsProjection.apply(x, linear.weight, linear, indexer) for x in xs]
    torch.autograd.backward(outputs, grads)


def test_single_call_with_first_microbatch_overwrites_main_grad():
    linear = _Linear()
    linear.weight.main_grad.fill_(7.0)
    xs, grads = _inputs(1, seed=0)
    _microbatch(linear, xs, grads)
    torch.testing.assert_close(linear.weight.main_grad, _wgrad(xs, grads))
    assert linear.weight.grad_added_to_main_grad
    assert getattr(linear, "_dsa_pending_first_call", None) is None


def test_repeated_calls_add_every_call_and_the_next_microbatch_accumulates():
    linear = _Linear()
    xs, grads = _inputs(4, seed=1)
    _microbatch(linear, xs, grads)
    expected = _wgrad(xs, grads)
    # The previous code overwrote with the first call's gradient alone.
    torch.testing.assert_close(linear.weight.main_grad, expected)
    xs2, grads2 = _inputs(4, seed=2)
    _microbatch(linear, xs2, grads2)
    torch.testing.assert_close(linear.weight.main_grad, expected + _wgrad(xs2, grads2))


def test_a_repeated_layer_never_overwrites():
    linear = _Linear(repeated=True)
    linear.weight.main_grad.fill_(7.0)
    xs, grads = _inputs(1, seed=3)
    _microbatch(linear, xs, grads)
    torch.testing.assert_close(linear.weight.main_grad, 7.0 + _wgrad(xs, grads))
