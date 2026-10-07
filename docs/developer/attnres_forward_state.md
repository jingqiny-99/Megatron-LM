# Attention Residual forward projection state

`attention_residual_forward_state.py` defines the static statistic layout and
the model-owned step snapshot for the forward-only projection experiment. It
does not introduce producer-owned gradients. Canonical Attention Residual query
and RMS-weight Parameters, their optimizer state and their complete consumer
backward path retain their ordinary ownership.

The supported experiment is dense, uniform PP2 with one or two virtual chunks,
TP=CP=EP=DP=1, no MTP, hybrid stack, offloading or custom layer partition. The
configuration fields are `attn_res_forward_projection` (default disabled) and
`attn_res_forward_projection_fraction` (default 1). Disabled preparation returns
`None` without registering consumers, touching parameter readiness, or
communicating. Model integration and typed P2P transport are separate modules.

## Consumer and source identity

For global dense layer `l`, the attention consumer ID is `2*(l-1)` and its MLP
consumer ID is `2*(l-1)+1`. The final aggregation has ID `2*L`, giving `2*L+1`
rows. Its logical consumer layer is `L+1`. Canonical module instances carry the
explicit attribute `forward_projection_consumer_id`.

Source zero is the completed embedding, available before layer 1. Historical
source `i > 0` completes after layer `i * block_layers`, before its value is
registered by the next block-start layer. Completion strictly precedes `L`:
the trailing partial used by final aggregation always remains locally scored,
including when the final layer exactly completes a depth block.

For each historical source, eligible consumers have logical layer greater than
its completion layer. The plan selects the latest
`ceil(fraction * eligible_count)` consumers. Selected IDs form a contiguous
suffix, allowing direct slices of raw query/norm banks. Fraction zero selects
no statistics; it remains an integration control with explicitly disclosed
snapshot and transport overhead.

`ForwardProjectionPlan` is frozen and contains only ordinary immutable Python
metadata. `from_config(config, data_parallel_size=...)` verifies the bounded
layout. It provides `consumer_layer`, `consumer_owner`,
`source_completion_layer`, `completed_sources`, `selected_consumers`,
`layer_offset`, `entry_rows`, `exit_rows`, `padded_rows` and `payload_shape`.
Stages are chunk-major: `global_stage = vp_stage * 2 + pp_rank`.

## Full-prefix auxiliary payload

Each active source contributes one inverse-RMS row, then one unnormalized-dot
row for every selected consumer still downstream of the boundary. A row key is
the immutable `ForwardProjectionRow(source_id, consumer_id)`, with `None` as
the consumer ID for inverse RMS. Rows are ordered by source, then consumer.

The payload carries the complete needed historical prefix at every boundary,
including a source that has completed but is still the outgoing value partial.
No auxiliary VPP cache or backward drain is required. For L16/block3/PP2/VP2 and
fraction one, the boundaries after layers 4, 8 and 12 carry 52, 54 and 50 logical
rows. The last boundary includes source 4, which completes at layer 12 and is
not registered as a historical value until layer 13.

All boundaries use uniform shape `[max_rows*S, B, 1]`, with `max_rows >= 1`.
This preserves the static VPP channel shape; unused rows are zero padded.
Fraction zero sends a real zero-filled row. No auxiliary gradient slot is
allocated or transmitted by typed P2P.

```python
plan = ForwardProjectionPlan.from_config(config, data_parallel_size=1)
row_map = unpack_forward_rows(plan, stage, received_aux, seq_length, batch_size)
# Add newly completed source statistics as contiguous nondifferentiable [S, B] rows.
outgoing_aux = pack_forward_rows(
    plan, stage + 1, row_map, seq_length, batch_size, hidden_states.device
)
```

Packing selects exactly the next boundary's rows, rejects missing or invalid
statistics and creates owning FP32 storage. The map may contain additional
valid local statistics no longer needed downstream. Unpacking returns explicit
views into the auxiliary payload. The graph adapter must own any backing
storage that is needed after another runner reuses its inputs. These helpers
do not attach Python attributes to tensors, infer provenance from tensor values,
or create a score-gradient edge.

## Model-owned preparation and graph boundary

The schedule calls:

```python
snapshot = prepare_attn_res_forward_projection(
    model_chunks, explicit_pp_group, data_parallel_size=1
)
```

The runtime belongs to the actual tuple of model chunks. It is never attached
to a shared config, so two models built from the same config register separately.
`get_attn_res_forward_runtime` accepts the same prepared chunk tuple; it is not
a lookup through an inner decoder. The schedule hands the returned snapshot
explicitly to each owning GPT model's host setter.

The first call performs one `all_gather_object` over the explicit PP group. The
registry verifies that all `2*L+1` IDs occur exactly once, physical owners match
the uniform partition, plans agree, and canonical FP32 query/norm Parameters
have a common epsilon and the original gradient ownership. Distinct consumer
modules retain distinct rows even if their Parameters alias. The registry does
not mark Parameters externally managed or replace any canonical buffer.

Every call makes local canonical Parameters ready through `ensure_params_ready`
before reading them, creates fresh raw FP32 `[2*L+1, H]` query and gamma banks,
and publishes each bank by one PP SUM. Every row has exactly one owner. The banks
are nondifferentiable and are neither Parameters nor persistent buffers. There
is no query-gradient SUM, finalization callback, or score backward protocol.

Epochs increment automatically per model. Tests and explicit callers may supply
a positive, strictly increasing epoch. All PP participants must call preparation
in the same schedule order. Preparation must finish before a blocking receive
or graph dispatch and must never occur inside CUDA graph capture.

The immutable snapshot exposes `epoch`, `plan`, `query_bank` and `gamma_bank`.
Before graph dispatch, the host calls
`snapshot.validate(snapshot.epoch, snapshot.plan)` and
`snapshot.validate_module(consumer_id, module)`. These reject a stale snapshot,
foreign module, replaced canonical Parameter, changed pointer/version/layout,
mutated plan, or modified bank. Host validation authorizes canonical consumers;
it is not a graph operation. The graph function receives only explicit tensor
banks, auxiliary tensors and static plan metadata. Do not pass the changing
epoch integer to native graph input metadata, or look up runtime state inside
the captured body.

## Validation status

The pure plan was checked locally against an independent row oracle for 224
combinations of depth, block size, VP count and placement fraction. Focused unit
tests additionally cover completion versus registration, zero/partial/full
placement, full-prefix padding/ownership, invalid row inputs, unique registry
ownership, Parameter aliases, readiness ordering, independent models sharing a
config, fresh raw snapshots and stale host surfaces. Registry/publication tests
use explicit mocked collectives and CPU tensors; they do not qualify real PP
training or CUDA graph behavior. Distributed numerical qualification remains a
separate required integration gate.

```bash
python -m pytest -q tests/unit_tests/transformer/test_attention_residual_forward_state.py
```
