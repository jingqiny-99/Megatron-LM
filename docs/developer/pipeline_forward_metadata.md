# Pipeline channels with forward-only metadata

`PipelineTensorSpec(shape, dtype, requires_grad=True)` describes one pipeline
wire channel. Plain shapes keep using `config.pipeline_dtype` and the existing
differentiable receive behavior. An explicit `requires_grad=False` channel is
sent in forward, received as a nondifferentiable tensor, and absent from the
backward wire. This API has no dependency on an Attention Residual mode flag.
It does not enable a model path by itself.

For example:

```python
specs = [
    PipelineTensorSpec(value_shape, torch.bfloat16),
    PipelineTensorSpec(metadata_shape, torch.float32, requires_grad=False),
]
communicator.send_forward([value, metadata], False, tensor_shapes=specs)
received = communicator.recv_forward(specs, False)  # [value_leaf, metadata]
communicator.send_backward([value_grad, None], False, tensor_shapes=specs)
grads = communicator.recv_backward(specs, False)  # [value_grad, None]
```

The two peers must agree on the ordered schema. Backward lists preserve every
slot: they must not compact away the metadata entry. All backward receive and
combined exchange helpers skip allocating or posting for that entry. A typed
send rejects a non-`None` gradient in a forward-only slot. Forward-only payloads
must themselves have `requires_grad=False`; the transport cannot repair a
model's incorrect autograd dependency. Differentiable forward values may be
detached on the sending side, as in the existing pipeline protocol.

Send-only helpers accept an optional `tensor_shapes` argument so callers can
validate the entire payload before posting. Their existing two-argument forms
remain valid; without a schema they transmit non-`None` tensors and cannot
infer whether a non-`None` gradient was meant to be forward-only. New typed
callers should always supply their schema. Combined exchange helpers retain
their existing positional signatures and accept the schema as `tensor_shapes`
or `tensor_shape`, as appropriate.

Explicit specs validate immutable nonnegative integer shapes, torch dtypes,
boolean gradient roles, payload arity, static send shapes, and channel dtypes
before any communication. Dynamic shape exchange retains the existing three
dimensional protocol and validates that typed shapes and sends have three
dimensions; it omits backward shape messages for forward-only slots too.
Explicit specs support batched and nonbatched P2P. The legacy ring-exchange
path remains available for plain shapes; explicit specs reject it.

An exchange posts all required tensor channels before waiting. Nonbatched
asynchronous calls keep the existing direction keys (`send_prev`, `recv_prev`,
`send_next`, `recv_next`); a direction with several tensors has one grouped
wait handle. A direction with no tensors has no handle. Waiting for one group
waits all of its original Works. A backward exchange containing only metadata
returns aligned `None` slots and an empty request container without allocation,
P2P operations, or optional CUDA synchronization. Empty outer shape lists remain
zero-channel operations; scalar and ordinary list payloads retain their return
structure and gradient channels.

For PP2, both logical edges can address the same peer. Batched tensor and shape
exchanges therefore visit the next edge first on even **PP-group** ranks and
the previous edge first on odd ranks. Using world-rank parity, or visiting the
same edge first on both peers, can match forward values to backward gradients.
Nonbatched calls preserve the native PP2 communicator selection and direction
order. The fix applies to plain channels as well as typed channels.

The focused tests include pre-post validation failures, all-channel posting,
receive autograd flags, structural missing backward slots, empty-work behavior,
legacy channels, and asynchronous directional ownership. Spawned Gloo workers
exercise PP2/PP4 and noncontiguous PP2 groups with distinguishable forward and
backward values, batched/nonbatched calls and static/dynamic shapes. A separate
two-rank NCCL test uses actual BF16/FP32 device tensors and the native asynchronous
path. The Gloo tests only redirect allocation to CPU and remove the CUDA-only
shape-exchange synchronization; their distributed operations are real.

From the checkout in an environment with PyTorch and Megatron dependencies:

```bash
python -m pytest -q tests/unit_tests/pipeline_parallel/test_p2p_forward_metadata.py -k 'not nccl'
python -m torch.distributed.run --nproc-per-node 2 -m pytest -q \
  tests/unit_tests/pipeline_parallel/test_p2p_forward_metadata.py -k nccl
```

These are transport prerequisites. They do not establish source-projection
numerical parity, end-to-end schedule integration, graph metadata ownership,
or a training performance improvement.
