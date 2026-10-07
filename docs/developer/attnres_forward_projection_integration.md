# Forward projection inside decoder pipeline graphs

This opt-in prototype extends the qualified decoder graph boundaries with
forward-only FP32 statistics. It preserves original BF16 source publication,
source taps, cache leaves, exactly-once cross-chunk gradient drains and canonical
query/norm parameter ownership. It does not move backward computation.

The standalone primitive is described in
[the primitive document](attnres_forward_projection_primitive.md). Its first
GPU revision passed 19 dedicated tests, including original FLA and independent
FP64 checks, eight/33-consumer fanout, stale bindings and fresh graph replay.
Those operator results do not qualify the model integration documented here.
Executed PP2/VP2 training passes for the disabled and fraction-zero controls;
fractions 0.5 and 1 fail the unchanged second-update gradient gate and were not
benchmarked. Fraction zero adds framework overhead. See the
[results and qualification scope](attnres_forward_projection_results.md).

## Host snapshot authorization

The schedule prepares a model-scoped snapshot after canonical parameters become
ready. GPT receives that snapshot through its explicit host setter and passes
`forward_projection_snapshot` plus its incoming `forward_projection_payload`
to the decoder. No shared configuration field stores a mutable runtime.

Before each decoder graph call, the host validates the snapshot's epoch,
current runtime identity, parameter identities/versions, bank versions and each
local canonical consumer module. Queries use IDs `2 * (layer - 1)`, MLP queries
use the next ID, and the final aggregation uses `2 * num_layers`.

Only the raw query bank, raw norm bank and incoming FP32 payload enter the
native manager as flat tensor inputs. Snapshot objects and epochs are not graph
arguments. Captured bodies clone all three tensors into owning graph storage.
The banks and metadata have `requires_grad=False`; canonical local parameters
remain differentiable parameters of the original decoder module.

## Completion, registration and source order

The pure `AttnResForwardProjectionStage` is recreated for each recorded decoder
invocation from the immutable plan and explicit graph surfaces. It unpacks a
full prefix of still-needed source statistics. There is no new mutable VPP
statistics cache and no statistics gradient cache.

Projection happens when a value becomes immutable:

* The embedding is source zero, projected immediately before layer one.
* A block completed by layer `k`, `2k`, etc. is projected immediately after that
  layer. The trailing final block remains local to the final consumer.
* Original BF16 source registration still occurs at the next block-start layer.
  Projection does not move the original tap or its gradient addition.

For the 16-layer, block-three, PP2/VP2 plan, layer 12 completes source four at
the end of the third logical stage. Its BF16 value is still the outgoing
partial. Its future statistics must nevertheless be packed before PP transfer,
so layer 13 receives them before registering that partial as source four.

The current plan selects a contiguous suffix of future consumers per source.
A source is projected once with the corresponding raw bank slices. Consumers
receive the same ordered BF16 source sequence as before, plus an explicit
per-call projection state. They bind selected `(source_id, consumer_id)` rows
to their canonical query/norm parameters and current owning value tensors.
This trusted rebinding is authorized by the host snapshot and immutable plan;
it does not disable the primitive's pointer/version/shape checks. Missing rows
or unauthorized consumers raise errors. The current partial is always local.

The exact one-source identity fast path stays unchanged. For multiple sources,
a consumer with no selected rows calls original FLA; a selected consumer calls
the paired primitive with complete original FLA backward. No gradient from the
raw bank or transported cache is added to the original parameter gradients.

## Mixed graph outputs and pipeline lifetime

Earlier VPP graph outputs are flat and ordered as
`(value_payload, forward_metadata, *original_value_exports)`.
The host takes an owning copy of metadata, removes it from the list passed to
the original BF16 output bridge, and preserves that bridge's source identity
and exactly-once drain behavior. Final virtual chunks have no value exports;
nonfinal physical stages still return `(value_payload, forward_metadata)`.
The final model stage returns the usual hidden output.

The native graph manager must preserve recorded nondifferentiable outputs.
FP32 metadata therefore has no backward graph output-gradient buffer. Typed
PP communication marks the metadata channel forward-only on both peers, so
neither peer posts a metadata backward message. The BF16 value channel retains
its original differentiable transport.

An auxiliary payload's host Python lifetime is not sufficient ownership for
native graph-pool storage. The graph body owns copies of incoming metadata and
banks, while the host owns a copy of outgoing metadata until its downstream
consumer has received it. Original value-input gradient ownership remains
unchanged. All copies and full-prefix transport costs belong in end-to-end
measurements.

## Required qualification

The new integration tests exercise four-stage prefix transport, layer-12
completion before registration, rebinding to distinct canonical query storage,
missing-row/unauthorized-consumer failures, explicit host snapshot checks and
the unchanged one-source gradient identity. The original stage/VPP tests must
continue to pass.

The executed PP2/VP2 gate uses ten independent Adam updates plus three after
each arm reloads its own checkpoint into the same model/graph storage. Disabled
and fraction-zero paths pass; their separate graph trajectories are bitwise
equal across losses, gradients, weights, masters and Adam state. Positive
placement fractions 0.5 and 1 fail before the second optimizer update, so their
training and pending-microbatch behavior is not accepted. The guarded PP2/VP1
path has no independent training trajectory in the current evidence.

Operator or same-input diagnostics are not substitutes for these gates. Timing
is restricted to qualified graph arms and includes query publication, FP32
packing and transport, owning copies, full backward and unchanged source drains.
The existing fraction-zero comparison measures combined framework overhead;
it does not establish a projection-placement benefit or PP-bubble reduction.
Any numerical correction needs fresh independent qualification before timing.
