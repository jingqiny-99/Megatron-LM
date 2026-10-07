# GPT boundary for forward-only Attention Residual metadata

The opt-in forward-projection path gives GPT two explicit host-owned inputs:
the current prepared query snapshot and the incoming FP32 projection payload.
They are separate from the usual BF16 decoder tensor. This boundary is only
supported for the bounded dense BF16 PP2/VP1-or-2 graph experiment described in
[the integration document](attnres_forward_projection_integration.md).

The schedule calls `set_attn_res_forward_snapshot(snapshot)` after preparing
current canonical parameters and before the first pipeline receive. The setter
validates the snapshot's epoch, runtime ownership and current tensor bindings.
Decoder dispatch validates it again, including the identity of every local
canonical Attention Residual consumer. A snapshot on another model or from an
earlier publication cannot authorize the current decoder.

`set_input_tensor` handles the two forward channels together:

* The first model stage accepts only empty pipeline slots and clears previously
  stored metadata.
* Other stages require a two-element list containing BF16 values and FP32
  metadata on the same device. Metadata must not require gradients.
* The decoder's ordinary input setter receives only the BF16 value tensor. GPT
  retains the explicit metadata reference for the immediately following forward.
* Every enabled input update first clears old metadata. Invalid replacement
  input raises before a new decoder binding is installed.

During forward, GPT passes `forward_projection_snapshot` and
`forward_projection_payload` as explicit decoder keywords. Those names belong
to the schedule; caller `extra_block_kwargs` cannot override them. GPT copies
that caller dictionary before adding the projection inputs. A missing snapshot
raises before decoder execution.

A nonfinal decoder returns `(values, metadata)` and GPT returns the flat
list `[values, metadata]` expected by typed pipeline communication. The final
model stage retains normal hidden-output/loss postprocessing. With the feature
disabled, the original single-tensor GPT boundary is unchanged.

The config must enable Attention Residuals with original FLA backward, BF16
parameters and pipeline values, fixed shapes, zero dropout, and either ordinary
stage graphs or full VPP graphs. Final-chunk-only graphs, eager projection,
unsupported parallel dimensions, offloading, recomputation and fused gradient
accumulation are rejected. Fraction zero is an enabled transport/snapshot
control; it does not disable the protocol's costs.

The dedicated boundary tests use the real GPT methods with a recording decoder
to verify input validation, metadata replacement, stale/missing snapshots,
reserved keyword ownership, disabled behavior and final-stage output behavior.
Separate config tests cover legal VP1/VP2 fractions and unsupported scopes.
These focused tests do not replace real graph capture or independent optimizer
trajectory qualification.
