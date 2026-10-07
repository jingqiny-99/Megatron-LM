# Native CUDA graph outputs with different gradient requirements

A training graph may return a differentiable activation together with floating
point forward-only metadata. Capture already excludes nondifferentiable outputs
from `autograd.grad` and records their `static_grad_outputs` slots as `None`.
However, a custom autograd Function makes floating point outputs differentiable
by default whenever an input requires gradients. Without an explicit replay
contract, detached metadata consequently acquires a backward edge during replay.

`_CudaGraphRunner.create_fwd_graph` saves a tuple of gradient requirements from
the captured tensor outputs, before weak-reference conversion. The replay
Function uses that tuple to call `ctx.mark_non_differentiable` on the actual
returned tensors. This happens after the existing last-layer output clones, so
both cloned and directly returned graph surfaces preserve capture's contract.
The weak-reference implementation, output ownership and graph allocation remain
unchanged. The flags must come from capture: output copies created inside the
custom Function's forward run without gradient recording.

Input/output tensor ordering and the native `None` gradient slots stay intact.
In particular, a floating point input with `requires_grad=False` consumes an
input slot but no entry in `autograd.grad`'s differentiable input list. This does
not shift gradients of later inputs or connected parameters. Unused
differentiable outputs retain PyTorch's materialized-zero behavior; replay does
not disable gradient materialization. Parameter gradients continue accumulating
into `main_grad` inside the native backward graph, with the original completion
event published to the connected parameters.

The focused regression file covers both output-copy modes, mixed input/output
slots, an unused differentiable output, updated parameters and metadata, and
three consecutive gradient accumulations. Its CUDA cases execute native eager
recording, warmup, actual F/B capture and replay in FP32 and BF16. They preserve
the installed weak-reference path. A separate CPU test exercises the real
replay autograd Function with simple graph-call substitutes; it verifies the
boundary semantics but is not CUDA capture or storage-lifetime evidence.

From the Megatron checkout in a supported GPU unit-test environment:

```bash
python -m torch.distributed.run --nproc-per-node 1 -m pytest -q \
  tests/unit_tests/transformer/test_cuda_graph_mixed_outputs.py
```

The change does not extend recording to arbitrary disconnected output-only
backward orders: the native record node remains attached to the first output.
The tests use the existing contract in which that first activation participates
in backward. All-nondifferentiable training outputs, changed gradient
requirements between record/capture/replay, PP metadata transport, and a model's
custom VJP are separate contracts and are not validated by this regression.
