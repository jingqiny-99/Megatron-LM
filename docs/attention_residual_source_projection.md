# Source-owned Attention Residual projections

## Contract

This opt-in follow-up is based on the fine-grained offloading branch
`e54c5cfc5`. `attn_res_impl=source` computes selected immutable depth-source
scores at their producer. Existing implementations and their defaults remain
unchanged. `attn_res_source_projection_fraction` is in [0, 1], defaults to 1,
and selects the last ceil(fraction * eligible_consumers) consumers for each
source. Partial residuals are scored at their consumer.

For BF16 or FP32 source V, effective query W=q*gamma, and unweighted RMS
normalization U, projection is Z=U W^T. The model projects a detached,
contiguous source snapshot at its producer and keeps the original value graph
in the value payload. Producer backward owns dW=G^T U; it does not return a
gradient to that detached source. Consumer backward owns the complete value
derivative, including both the direct mixing term and the score derivative.

Specifically, for source i, r_i=rsqrt(mean(V_i^2)+eps),
u_i=<dY,V_i>, delta=sum(alpha_i*u_i), and
dZ_i=alpha_i*(u_i-delta). The consumer combines
dV_i=alpha_i*dY+dZ_i*r_i*W-V_i*(dZ_i*r_i^2*Z_i/H) in FP32 before one
source-dtype cast. Contributions from separate consumers then accumulate
through the original value graph. This restores the consumer's BF16 rounding
boundary while retaining producer-owned score projection and query gradients.
Locally scored partials compute both derivatives at their consumer.

For detached MTP heads, the consumer's history values are detached while their
score tensors retain the producer query graph. The complete value derivative
is therefore discarded at the detached value inputs, and query gradients
remain live. Query parameters, statistics, reductions and accumulation remain
FP32; no TF32 or BF16 query conversion is allowed.

## Ownership and runtime API

`attention_residual_projection_runtime.py` provides
`mark_attn_res_projection_parameters(module)`,
`prepare_attn_res_projection(model, pp_group)`,
`finalize_attn_res_projection(model, pp_group)`, and
`get_attn_res_projection_runtime(module_or_config)`.

Each AttentionResidual has a globally ordered `projection_consumer_id` tuple
`(domain, depth, layer_number, slot)` and `projection_stop_source_grad`.
Domain 0 is trunk (depth 0); domain 1 is MTP. GPT attention/MLP slots are 0/1,
hybrid entries use slot 0, and MTP final aggregation uses slot 2. Trunk final
aggregation has layer_number=num_layers+1. Repeated MTP modules reuse the
same bank record; separate modules with aliased parameters retain separate
records and merge only at canonical-parameter gradient scatter.

The runtime exposes a differentiable non-Parameter FP32 `bank[Q,H]`,
`column(module)`, and ordered `consumer_metadata` records with `id`,
`stop_source_grad`, `owner`, and `column`. It attaches to both module and
configuration objects. Metadata/banks/caches are nonpersistent. Every rank
prepares its bank before its first pipeline receive, using ensure_params_ready
before reading canonical query parameters. Gradients are PP-SUM-reduced after
backward P2P completes, then mapped into original q/gamma main_grad before
ordinary DP/TP finalization. No PP averaging is applied.
At every schedule entry, shared configuration objects are rebound to the
current model's registry. Alternating independently constructed models with a
shared config must not reuse the other model's bank or gradient destination.

Managed parameters have separate buffer/group identity, shared by DDP and
optimizer layout construction. Both ordinary readiness hooks and explicit
early synchronization respect deferred readiness. Finalization marks each
canonical parameter ready once, including zero gradients and aliases. Other
parameter groups retain communication overlap. The bank remains differentiable
even when all source inputs are detached.

Built-in consumer IDs make selected query rows a contiguous suffix. Source
completion uses a protected bank view for this case, avoiding host-to-device
index copies and a gather. External metadata can yield noncontiguous rows;
those retain the general index-select path. This does not enable full-model
CUDA graphs, which remain rejected by the existing Attention Residual config
validation.

## Kernel interfaces

`attention_residual_projection_kernels.py` provides:

* `project_source(value, effective_queries, live_source_mask=None, *, eps=1e-6,
  backend=None) -> (value_proxy, logits)`, preserving source token dimensions
  and appending the query dimension to logits.
* `aggregate_preprojected(values, local_query, logits=None, *, eps=1e-6,
  backend=None, precomputed_value_grad=False) -> output`, where logits is a
  same-length list of FP32 token-shaped tensors or None for locally scored
  sources.

The public primitive defaults are unchanged. With
`precomputed_value_grad=False`, provided logits are independent score inputs:
the consumer returns their dZ and only the direct value contribution, while
`project_source` can compute source and query derivatives when its value input
requires gradients. Its optional live-source mask continues to gate only the
source derivative.

The model explicitly selects `precomputed_value_grad=True`. This is a paired
contract: each provided score must come from a detached copy of the same value,
the same effective query and the same normalization epsilon. The consumer
adds the score's value derivative, but sends its query derivative through the
provided score tensor only. Using a live source in the producer projection
with this flag would double-count its score derivative. For precomputed
sources, the consumer recomputes r_i in backward; locally scored sources reuse
their saved statistic. One-source identity and exact zero query/score
derivatives are preserved in both modes.

CUDA uses custom Triton kernels; an explicit Torch implementation and independent
ordinary-autograd references support correctness testing. No numeric-failure
fallback is permitted. CUDA source backward skips the producer value-gradient
kernel when its input is detached and executes only the query-gradient kernels.
The Torch debugging path currently computes and discards that unused producer
value derivative. Complete dV executes at each consumer's backward; producer
dW executes when the score graph becomes ready. Only cross-PP parameter-gradient
communication is delayed to step end. No cross-microbatch batching delays the
source gradient.
This placement moves forward score projection and query-gradient work to
producers; score-induced value-gradient work remains at consumers. Its effect
on backward rank balance and total iteration time must be measured separately.

## Pipeline and source lifecycle

The model transports `[value_payload, score_payload]`. Both are fresh non-view
tensors. PipelineTensorSpec(shape,dtype) keeps activation dtype and FP32 scores
separate in both directions. Each exchange posts every channel before waiting;
waiting for a backward value before sending forward scores would deadlock.
Directional work groups preserve asynchronous VPP waits. Both outputs participate in a single autograd
engine invocation. Final stages retain the existing scalar-loss interface.

PP2 virtual stages use the same physical peer for both logical edges. Batched
exchanges post the next edge first on even PP-group ranks and the previous edge
first on odd ranks, for both shape metadata and payloads. Posting the previous
edge first on both ranks silently swaps forward activations and backward
gradients during VPP steady state when their shapes agree. Global rank parity
is not sufficient because PP groups can contain noncontiguous world ranks.

`attn_res_projection_payload_shape(config, seq_length, micro_batch_size,
pp_rank, vp_stage=None)` describes scores entering the specified receiving
stage. The sequence length is already local to CP/SP. Interleaved schedules
use a common padded score shape. Layout is deterministic from the registry
and depth schedule; padding has no gradient.

Source IDs are embedding 0 followed by completed depth blocks. A source is
projected as soon as its producing block completes, including a completed
outgoing partial at a PP boundary, before the next block registers it as a
historical source. Score deltas therefore use completed-source counts rather
than blindly copying the existing registered-value prefix count. Source and
score VPP leaves retain their immutable storage and separate chunk graphs.
Value leaves drain complete consumer dV into the original source graph; score
leaves drain dZ into the detached producer projection's query graph, exactly
once. Cache entries are keyed by
microbatch/source and evicted on the last local virtual chunk; backward retains
captured references. Partial projection requires an explicit zero-VJP anchor
from the outgoing score payload to each current score tap: a chunk can consume
none of a source's selected columns while a later local chunk still accumulates
into its cached leaf. The anchor reads no score values and prevents orphaned
backward taps. Both caches reset independently of paged stash at schedule entry.
Tensor metadata must be preserved at explicit detach,
view, tap and pipeline pack/unpack boundaries.

Shared source values, query banks and score caches outlive temporary offload
groups. They must not be force-released; existing do-not-offload mechanisms
protect these references. Temporary norm/MLP offload scopes preserve the base
branch's distinction between aggregate inputs and live partials. This change
does not introduce shared-source CPU offloading. MTP retains a fixed trunk
source tuple and a separate partial at every prediction depth. Detaching MTP
history preserves score metadata, so it stops the trunk value graph without
stopping the corresponding producer-owned query gradients.

## Validation and delivery

Required comparisons are untouched offload baseline, feature disabled on this
branch, independent source reference, and custom kernels. Test BF16/FP32,
H1024/H7168, source counts 1/3/9, changing partials, aliases, zero queries,
one-source identity, identical sources, and every source/query/norm gradient.
Retain existing numerical thresholds: BF16 output/dV atol .016, rtol .01,
relative L2 <.003; FP32 statistics atol 2e-5, rtol 2e-4, relative L2 <2e-5;
dq atol 1e-3, dgamma atol 5e-5, rtol 5e-4, relative L2 <1e-4. Exact zeros must
remain zero. Report interface rounding separately; never loosen thresholds.

Real distributed validation covers PP1/2/4, PP2VPP2/PP4VPP2/PP2VPP4, uneven
hybrid layouts, MTP1/2 with repeated/detached paths, TP2+SP, CP2, supported EP,
overlap modes, selective recompute and the existing offload matrix. Run at
least ten optimizer steps and three resumed steps; verify feature-off to
feature-on distributed optimizer checkpoint restore and resharding.

Measure end-to-end producer/consumer F+B and full pipeline iterations, including
query-bank synchronization, per-rank compute/exposed communication, memory and
wire bytes. Compare fractions 0/.25/.5/.75/1 against FLA with matched seeds,
hardware and containers, three repetitions of ten warmup and fifty measured
iterations. No performance claim is established before executed evidence.

## Implementation and executed evidence

The implementation is opt-in and preserves canonical model state-dict keys.
Standard MCore DDP and optimizer layouts distinguish externally managed query
parameters from ordinary parameters. Torch/Megatron FSDP wrappers are rejected
because their external-gradient publication lifecycle is not implemented.
Existing Attention Residual limitations remain in force.

The following CPU results were obtained before the consumer-local complete-dV
integration. They cover the earlier producer-value-gradient contract and must
not be presented as qualification of the revised model path:

* 27 kernel forward/backward cases and 40 PP/VPP source lifecycle cases,
  including the fraction-.5 orphaned-score-tap regression. These load real
  production operator/state code; the pipeline reference is independent
  unsplit ordinary-autograd math, not a simulated production schedule.
* 57 typed transport and schedule lifecycle cases, rerun on 2026-10-06.
  Spawned Gloo processes cover PP2/PP4, scalar BF16 and typed BF16/FP32 payloads,
  static/dynamic shapes, repeated bidirectional exchanges and noncontiguous PP
  groups. An old-order negative control fails with all forward values receiving
  the backward marker. These host checks complement earlier PP2 gradient and
  PP4 batched/unbatched/asynchronous exchange checks; they do not qualify NCCL
  or full-model training by themselves.
* 16 runtime/DDP/layout cases plus a real two-process Gloo query publication
  and gradient-SUM run over two optimizer steps.

Before the complete-dV integration on October 6, normal CUDA pytest execution
passed all 148 operator, source-state, and model-contract cases. Four-rank
torchrun passed all 18 runtime cases,
including NCCL publication, and both directional NCCL transport regressions.
These results complement the host checks; full-model numerical acceptance and
end-to-end speedup remain unqualified. The standalone operator benchmark records strict correctness
before timing and labels its Torch consumer-owned baseline separately from FLA.
Full-model tests use normal GPT/hybrid imports, production schedules, DDP and
Megatron Adam, with ten matched updates and optional three-step resume replay.


### Checkpoint layout compatibility

The canonical model parameter names are unchanged. Source mode separates query
parameters into externally managed optimizer buffers, so toggling it changes the
flattened optimizer layout. The supported `fully_reshardable` format addresses
optimizer state by model parameters and permits this transition. The deprecated
`fully_sharded_model_space` writer still requests `ShardedTensor.flattened_range`,
which current checkpoint mapping rejects; a GPU save with the unchanged eager
backend failed before migration, and the same incompatibility exists in base
`e54c5cfc5`. Its stale optimizer-API default is not evidence of format support. The
normal training default, `dp_reshardable`, addresses buffers by index and cannot
be loaded across that change. The loader rejects the transition early with
conversion guidance: load/save once using the checkpoint's original backend and
`--dist-ckpt-optim-fully-reshardable`, then enable source mode. Setting the flag
only while loading does not convert an existing checkpoint. The same rule
applies when disabling source mode. Optimizer moments must never be discarded
to make a migration appear successful.

Before the complete-dV integration, the supported-format migration test passed
on four GB200 GPUs for both FP32
and BF16. It trained the eager backend for ten updates with real MCore DDP and
DistributedOptimizer, restored the source backend while changing PP1/DP4 to
PP2/DP2, and verified exact model tensors, master parameters, Adam moments and
step counters. Three resumed source updates matched an independent checkpoint
replay bitwise, and the resulting state restored exactly back to PP1/DP4. The
small model is replicated across PP to isolate optimizer DP resharding; this
evidence does not qualify full-model PP/VPP gradient parity. The test is
`tests/unit_tests/distributed/test_attn_res_projection_checkpoint.py`.

### Earlier producer-value-gradient operator evidence

Six GB200 BF16 cases passed strict forward/all-gradient checks at 256 tokens,
H1024/H7168, source counts1/3/9, Q8 and fractions0/.5/1. The container used
Torch2.12.0a0 (26.04), CUDA13.2 and Triton3.6; TF32 was disabled. Both exact-zero
and finite-value gates were enforced. Initial operator timings include Python
launch gaps and compare against ordinary Torch math, not FLA or full training.
At H1024/S3/Q8, fraction0 measured2.02ms and fraction1 measured3.51ms; at
H7168/S9/Q8 they measured2.60ms and7.10ms. These do not demonstrate a speedup from
projection placement; full-pipeline balance and step time remain the deciding
measurements.

A separate90-case FP32 CPU stress matrix passed all7200 strict checks with worst
relativeL2 4.82e-7. Some Q32/64 BF16 graph comparisons exceed the unchanged
pointwise dV threshold because repeated BF16 edge accumulation amplifies small
per-consumer differences. Matched-value FP32 graphs, individual consumer VJPs
and producer VJPs with identical incoming gradients pass; failed BF16 stress
cases remain explicitly failed and are not timed. This evidence does not waive
end-to-end model validation.

### October 6 diagnostics before the complete-dV integration

The PP2 same-peer ordering regression passed both scalar BF16 and typed
BF16/FP32 payload cases under real four-rank NCCL on GB200. The repaired
PP2/VPP2 full-model comparison passes its first two optimizer updates, then
fails the unchanged gradient gate on update three (worst reported query
similarity 0.99867284 against the required 0.999). The ten-update comparison
and its resume path remain unqualified.

Ten-update FLA-to-FLA and source-to-source PP1 controls pass. Eager-to-FLA
fails on update two, and source fraction zero versus one fails on update
three. These independently evolving BF16 trajectories establish a numerical
diagnostic context; they do not waive the source-backend acceptance gate.

Sixteen exact operator captures from the failing PP2/VPP2 update were replayed
on GB200 with identical values, canonical queries, norm weights, and incoming
output gradients. FLA, local-source, and preprojected-source paths each pass
the original operator thresholds against independent FP64 autograd equations
preserving their public dtype interfaces. Saved source scores and query-bank
snapshots equal fresh projections bitwise. This validates the captured local
operators, not multi-consumer accumulation or a complete training trajectory.

The earlier source placement changed a BF16 gradient rounding boundary: direct
value gradients were rounded at each consumer before the producer combined
score gradients. Matched CPU captures show identical forward outputs and query/norm
gradient differences around 1e-7 relative, while projected value gradients
differ by approximately 0.003 relative. This motivated the current consumer-local
complete value VJP with producer-owned query VJP described above.

The current diagnostic container uses Torch 2.12.0a0 (26.04), CUDA 13.2,
Triton 3.6, fla-core 0.5.1, and nvidia-resiliency-ext 0.6.0, with TF32
explicitly disabled. The last two packages were installed into the held
container; these results do not describe the unmodified image.

Ten source-to-source controls additionally passed ten optimizer updates each:
PP4/VPP2, PP2/VPP4, MTP2, detached MTP with activation offloading, selective
recompute, TP2 with SP, CP2, EP2, repeated MTP, and uneven hybrid layouts. All
40 per-rank reports pass. Self-controls establish reproducible execution and
updates, not parity against an independent backend.

The earlier producer-value-gradient path also fails a same-parameter-state FLA
comparison on update five (query similarity 0.99889159). Its outstanding
numerical issue therefore cannot be attributed solely to divergent optimizer
trajectories.

### Current consumer-local complete-dV integration

The consumer-local complete-dV candidate passed 16 matched GPU operator captures
against its independent reference. The revised production implementation, with
no experimental module override, then passed the original loss and gradient
gates at all ten successive FLA reference parameter states in a four-GB200
BF16 GPT PP2/VPP2 diagnostic. This reproduces the earlier candidate's ten-step
same-state result. Before each pair, the diagnostic strictly copies
all reference parameters into independently stored candidate parameters; only
the FLA optimizer advances. This establishes same-state forward/backward parity
for that case. It does not qualify the candidate optimizer, independently
evolving trajectories, resumed training or other parallel configurations.

The independently evolving candidate placement comparison still fails on update
five, and its FLA comparison fails on update four. Those failures and the
original acceptance thresholds are retained. The positive same-state diagnostic
does not replace the required ten-update and three-resumed-update comparisons.

The current production integration explicitly pairs detached producer projection
with `precomputed_value_grad=True` in AttentionResidual and in its lifecycle
tests. New primitive tests cover FP32/BF16, fractions 0/.5/1, mixed local and
precomputed sources, changing partials, complete value/query/norm gradients,
detached MTP history, and skipping the CUDA producer dV kernel. Normal CUDA
pytest execution of the revised production operator, source-state and
model-contract tests passed all 177 cases, including the 29 new cases. The
four-rank runtime regressions also passed all 18 cases per rank, including
real NCCL gradient publication. The independently evolving production FLA/source
comparison fails on update four: query similarity 0.99881339 and norm-weight
similarity 0.99872333 against the unchanged 0.999 threshold. Its resume path
is therefore not reached. Earlier parallel-model self-control and
checkpoint results above retain their original contract and scope. No
end-to-end speedup or independent-trajectory qualification is claimed.

The revised BF16 operator fanout check (H1024, three sources, 64 tokens) passes
Q8 at fractions 0/.5/1 and Q32 at fractions .5/1. Q32 at fraction 0 and Q64 at
all three fractions fail the original pointwise source-gradient gate, with
maximum absolute error .03125 and relative L2 below 9e-5. All failed cases
remain untimed. A matched-capture probe finds source and local saved RMS/scores
bitwise identical, while softmax probabilities differ. The local consumer now
uses an explicit FP32 round-to-nearest multiplication for dot-times-rstd. This
matches the stored producer-score boundary and prevents contraction into the
following score-minus-maximum operation, while other compiler fusion remains
enabled. All 16 matched captures then have identical probabilities and outputs.
Six deterministic CUDA regression cases cover FP32/BF16 and one, two or three
precomputed sources. A separate pinned-old-kernel negative control reproduces
the probability mismatch. The final production operator/state/contract suite
passes 183 cases, and its PP2/VPP2 same-state diagnostic again passes ten steps.
Independent source-fraction and FLA trajectories still fail on update four;
the rounding fix does not establish trajectory acceptance.

After adding contiguous query-bank views, all 42 source-lifecycle cases pass,
including two new contiguous/noncontiguous gradient-mapping regressions. A
four-GB200 PP2/VPP2, MTP2, detached-history and activation-offload source
self-control also passes ten independent updates. This checks the shared view's
lifetime; it remains a same-backend control rather than independent parity.

### Current operator performance

The final kernel was measured on GB200 with BF16 values, 256 tokens, eight
consumers, and a detached last-consumer history. All six exact workloads pass
the unchanged independent operator gates before timing. Each measurement uses
ten warmup iterations and three repetitions of fifty complete forward/backward
iterations, including producer projection and query-gradient work.

| Execution | Hidden / sources | Fraction 0 | Fraction .5 | Fraction 1 |
| --- | --- | ---: | ---: | ---: |
| Eager | 1024 / 3 | 1.838 ms | 2.665 ms | 3.294 ms |
| Eager | 7168 / 9 | 2.560 ms | 5.258 ms | 6.832 ms |
| Operator CUDA Graph replay | 1024 / 3 | 0.165 ms | 0.226 ms | 0.269 ms |
| Operator CUDA Graph replay | 7168 / 9 | 0.632 ms | 0.892 ms | 1.131 ms |

The benchmark uses a contiguous-tail bank slice and clone, preserving its fresh
bank allocation without a Python-list index transfer. Earlier measurements
using CPU index copies are superseded. Graph capture covers the complete
producer/consumer forward and backward; two replays also pass the unchanged
output, score and gradient gates against eager execution before timing.

All entries are median CUDA-event timings. Eager execution includes Python
launch gaps. Graph replay removes recurring Python construction/dispatch, but
full source placement still increases measured GPU execution by about 63% and
79% at the two shapes. These data show added operator cost, not a net speedup;
they do not identify the individual kernels responsible. The separately
measured Torch autograd reference is not FLA. No PP transport, bank
synchronization, DDP, optimizer or transformer layers are timed here.
Full-model Attention Residual CUDA graphs remain unsupported, and distributed
critical-path benefit remains unproven.
