# Forward-only Attention Residual metadata in PP schedules

The bounded PP2/VP1-or-2 forward projection path prepares one model-owned query
snapshot at schedule entry, after explicit process groups are available and
before the first blocking receive. `pg_collection.pp` supplies publication and
`pg_collection.dp_cp.size()` verifies the DP1/CP1 scope. The helper resets the
ordinary source cache at entry and calls each wrapped GPT model's
`set_attn_res_forward_snapshot` with the same prepared snapshot. It does not
publish query gradients or add a finalization collective.

With `attn_res_forward_projection=False`, this entry helper returns immediately.
The ordinary shape and output-wrapping branches remain in place. Existing
gradient-bearing inputs retain the same backward path; `retain_grad` now skips
nondifferentiable inputs so metadata is a valid structural input slot.

Noninterleaved `get_tensor_shapes` and the interleaved uniform-shape construction
add two explicit `PipelineTensorSpec` channels when enabled:

1. The original BF16 value payload, including its unchanged depth-source slices.
2. The static full-prefix FP32 auxiliary payload with `requires_grad=False`.

The auxiliary shape uses the local token sequence length **before** expansion
by value-source slices. The full-prefix layout and uniform padding are defined
by `ForwardProjectionPlan`. Variable sequence lengths, multimodule pipelines
and custom shape adjustment are rejected for this bounded experiment.

Nonfinal GPT outputs are the flat list `[value, auxiliary]`. `forward_step`
validates and returns this list directly, including on a first stage whose input
started as `None`; nesting it in an additional one-element list would corrupt
P2P channel matching. Final-stage output and loss normalization keep their
ordinary path. The final stage receives both input slots and returns only the
normal model output or loss.

`backward_step` runs the original value-output backward, preserves the auxiliary
input's `None` gradient slot, and rejects any auxiliary output gradient before
performing backward. Typed P2P omits this slot from both sends and receives.
Noninterleaved send-only call sites pass their exact schemas; combined exchanges
use the same schemas through their existing arguments. All-channel posting and
wait ownership are implemented by the P2P communicator, without schedule
reordering.

The original recursive `deallocate_output_tensor` remains unchanged. Nonfinal
value and auxiliary outputs own their storage; the graph adapter clones exported
auxiliary tensors outside the native pool, and the receiving graph owns an
input clone before unpacking statistics. Existing send completion waits cover
both channels before their output storage is pseudo-deallocated. No metadata
gradient or autograd dependency is introduced to extend that lifetime.

Targeted CPU tests exercise matching peer schemas, local-token shape accounting,
first/intermediate/final forward return structures, unchanged final loss
normalization, absent metadata gradients with and without pseudo-deallocation,
rejection of a spurious auxiliary gradient, and once-per-step snapshot dispatch.
They are prerequisite checks, not distributed numerical qualification:

```bash
python -m pytest -q tests/unit_tests/pipeline_parallel/test_schedules_forward_metadata.py
```

Actual PP2/VP1 and PP2/VP2 training still require the full independent-update
numerical, graph replay, value-cache lifetime and optimizer-state gates.
