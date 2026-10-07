# Forward projection experiment: qualification and results

As of October 7, 2026, this opt-in implementation has **no demonstrated training
speedup or PP-bubble reduction**. The disabled path and enabled fraction-zero
control pass the bounded training gate. Fractions 0.5 and 1 fail the unchanged
second-update gradient gate and were not benchmarked. Fraction zero adds about
3.16% update latency, establishing integration overhead before moving any
projection work. Keep the feature disabled by default.

Implementation details are in [the primitive](attnres_forward_projection_primitive.md),
[graph integration](attnres_forward_projection_integration.md),
[model-owned snapshot](attnres_forward_state.md), and
[typed transport](pipeline_forward_metadata.md) documents.

## Experiment and numerical contract

The model is a dense BF16 proxy: L16, H1024, FFN2048, eight attention heads,
sequence length 4096, vocabulary 4096, and AttnRes block size 3. Parallelism is
PP2/VP2 with TP=CP=EP=DP1, microbatch size 1, eight microbatches and VP group size 2.
Both decoder chunks use native local CUDA graphs. P2P overlap and output
deallocation are enabled; warmup/flush overlap is disabled. There is no activation
offload, MoE, MTP or CP. This is not a full Kimi K3 architecture benchmark.

The run used two NVIDIA GB200 devices on one held node, Python 3.12.3,
Torch `2.12.0a0+0291f960b6.nv26.04.48445190`, CUDA 13.2, NCCL 2.29.7 and original
FLA with FP32 canonical query/norm parameters. TF32 was disabled. Queries were
initialized nonzero; this was not a zero-query-only model check.

Each candidate was compared with an independent original-FLA eager model from
one shared initial parameter state. Real Adam trajectories then evolved
independently. The gate checks losses, every pre-optimizer main gradient, updated
parameters, FP32 masters and Adam state, query updates, source-cache publication
and exactly-once drains, native graph replay/readiness, and snapshot epochs.
Loss tolerances remain `atol=rtol=0.001`; tensor cosine and similarity must both
be at least 0.999. No failing tolerance was relaxed.

After ten updates, each arm saved and reloaded its **own** checkpoint into its
existing model/graph storage, then ran three more updates. This tests a
storage-preserving checkpoint roundtrip; it does not establish fresh-process
graph reconstruction. One graph candidate is captured per process. Direct
graph-to-graph comparison uses saved canonical tensors from separate processes,
without copying them back into either trajectory.

| Mode | Independent training result | Direct graph comparison | End-to-end timing |
| --- | --- | --- | --- |
| Feature disabled, original FLA | 10 + 3 updates passed | Reference | Measured |
| Enabled, fraction 0 | 10 + 3 updates passed | Bitwise equal to disabled across all 13 updates | Measured |
| Enabled, fraction 0.5 | First update passed; second pre-optimizer gradient gate failed | Not accepted | Not timed |
| Enabled, fraction 1 | First update passed; second pre-optimizer gradient gate failed | Not accepted | Not timed |

The direct disabled/fraction-zero comparison covers 104 snapshot artifacts,
26 rank-steps and 9,971 reported tensor comparisons: losses, main gradients,
parameters, FP32 masters and both Adam moments. All are bitwise equal; scalar
optimizer metadata also matches. Neither positive placement fraction has this
acceptance. PP2/VP1 is implemented as a guarded graph path but has no executed
independent training trajectory in this experiment.

Both failed fractions identify global layer 12's MLP AttnRes query/norm gradients.
The minimum reported similarity is 0.99867582 for fraction 0.5 and 0.9981795 for
fraction 1, below the 0.999 gate. Host snapshot epoch and canonical-bank checks
had passed before these failures. That observation does not by itself prove
all later graph-input copies correct.

## Qualified framework overhead

Both timing arms use the same integration branch and frozen production tree.
"Disabled" therefore means this branch with the flag off, not an untouched
checkout of the parent graph PR. Fraction zero moves no projections but retains
snapshot publication, extra graph inputs, owning copies, a minimum zero metadata
payload and typed forward transport.

Four fresh-process pairs alternate OFF/F0, F0/OFF, OFF/F0 and F0/OFF. Each process
runs one initialization update followed by four subblocks of ten warmups and
50 measured complete updates. Each sample is the maximum of the two rank times.
The clock includes zero-grad, F/B, bank publication, P2P, gradient finalization,
Adam and the original ending CUDA synchronization. There are no profiler/event
observers in these timing runs. The eager reference remains resident alongside
the graph candidate.

| Statistic | Disabled | Enabled, fraction 0 |
| --- | ---: | ---: |
| Measured updates | 800 | 800 |
| Mean update latency (ms) | 73.185050 | 75.499634 |
| Median update latency (ms) | 73.189268 | 75.452399 |

The added mean latency is 2.314584 ms, or 3.162645%; throughput falls 3.065689%.
Every process pair is slower: 3.079774%, 3.400972%, 3.053523% and 3.116267%.
All samples are retained, including one 110.210480 ms fraction-zero outlier.
All four median comparisons have the same slowdown direction. The four process
pairs are the comparison units; adjacent update samples are repeated observations.

Allocated peaks increase by 321,536/369,664 bytes on ranks 0/1. These peaks include
both resident models. Reserved-memory variation is not a memory-saving result.
The benchmark measures combined framework cost; it does not isolate bank
collectives, host validation, copies or communication individually. It cannot
predict the latency of the numerically failed fractions.

## Pipeline dependency and preparation evidence

A separate qualified fraction-zero diagnostic ran 540 updates with six balanced
permutations of no observation, phase/graph events, and phase/graph/wait events.
It used preallocated CUDA events and read them after the original ending
synchronization, without adding a synchronization, query or profiler. Its
uninstrumented control averaged 75.448324 ms. Phase observation added 0.5432%;
phase-plus-wait observation added 0.8276%. Readout was outside the update timer
but could still affect subsequent rank arrival.

Rank-local event totals in phase-plus-wait mode, milliseconds per update:

| Rank | Decoder graphs | Scheduled phases | Phase minus graph | Existing waits |
| --- | ---: | ---: | ---: | ---: |
| 0 | 61.289466 | 66.061992 | 4.772526 | 7.072087 |
| 1 | 61.486054 | 67.261118 | 5.775064 | 3.225575 |

Decoder graph service is nearly balanced across physical ranks. Phase minus
graph includes staging, owning copies, endpoint work, CPU enqueue gaps and
observation effects; it is not a pure CPU or idle category. These rank-local
intervals do not establish a shared clock or a cross-rank critical-path budget.

There are 48 directional waits per rank/update. Two-channel forward work groups
and one-channel backward work produce 72 endpoint channel waits; these are not
72 unique wire messages. The nondifferentiable auxiliary slot is absent from
backward traffic. Rank 0 spends 5.893151 ms in backward receive intervals,
including 3.511458 ms during cooldown. Its largest cooldown waits remain before
VP1 microbatches 6/7, at 1.478632/1.977650 ms. Fraction zero retains this backward
dependency pattern and offers no evidence of reducing the pipeline tail.
Neither wait counts nor wait intervals alone measure transport latency or
intrinsic fill/drain bubbles.

Preparation is a concrete next framework target. Static inspection of the warm
runtime finds 19 full-model validations per rank/update: one during preparation,
two GPT snapshot setters and sixteen graph forwards. These repeatedly validate
both local chunks' immutable plans and canonical parameter bindings. Even
fraction zero publishes two raw FP32 banks with two PP SUM collectives before
the first receive. Current events begin after preparation, so they do not time
bank publication or validation separately. Any reduction of repeated checks
must retain rejection of foreign/stale snapshots, changed configuration,
module replacement and changed parameter ownership before consumption.

Earlier original-source event runs provide nonpaired context only: source,
observer adapter and execution time differ. Their subtraction cannot decompose
the matched 2.314584 ms OFF/F0 slowdown. CPU observations suggest more forward
host work, but CPU and queued GPU execution overlap. A controlled preparation
and validation measurement is needed before assigning savings or changing the
contract; no such optimization is qualified here.

## Operator and same-input evidence

At T4096/H1024/Q8, all six primitive benchmark cells passed their original
numerical gates, but total producer-plus-consumer F/B did not improve:

| Sources | Placement fraction | Original FLA F/B (ms) | Candidate F/B (ms) |
| --- | ---: | ---: | ---: |
| 3 | 0 | 0.618066 | 0.647025 |
| 3 | 0.5 | 0.616059 | 0.640082 |
| 3 | 1 | 0.616407 | 0.625930 |
| 7 | 0 | 0.854241 | 0.890635 |
| 7 | 0.5 | 0.854774 | 0.935014 |
| 7 | 1 | 0.854671 | 0.972212 |

These operator numbers exclude pipeline transport, snapshot publication, Adam
and surrounding layers. The backward remains the original complete consumer
FLA implementation. Passing this test does not replace the failed model gate.

A diagnostic ran the original capture/setup schedule without an optimizer
update. For each non-captured cached forward it also evaluated original FLA on
the **same** values, canonical query and norm, returning the original candidate
outputs unchanged to the model. In the 136 real-input record calls, actual owning
banks equal the published snapshot and canonical consumer rows bitwise. Inverse
RMS is also exact. BL8, four warps and three stages match for both kernels.

The first cached consumer, global layer 9 attention, already has 30 differing BF16
output elements out of 4,194,304. Its cached-source logits differ by at most
8.9406967e-8. Across real-input calls the maximum logit difference is 1.1920929e-7;
maximum BF16 output difference is 0.0009765625. Rank0's logit mismatch counts are
entirely accounted for by cached-source dot-times-RMS differences. Its failing
layer 12 MLP consumer does not use the single-local-source fast path. Later rank1
consumers show an additional local-source reduction discrepancy.

Native graph warmups use synthetic zero-valued pool buffers. Their bank/RMS
mismatches are excluded from these real-input observations; they are not evidence
of stale training snapshots. This diagnostic does not inspect second-update
replay copies or establish the complete gradient-error propagation chain.

Four saved real-input cases reproduce original/candidate outputs and statistics
exactly in an isolated replay. Freshly generated producer statistics are bitwise
equal to carried statistics. The isolated primitive checks and original
tensor-comparison thresholds pass for these cases under an explicitly synthetic
BF16 upstream gradient, including original FLA and independent FP64 comparisons.
This narrows the arithmetic question but does not reproduce the failing model's
actual upstream gradient or trajectory.

A separate diagnostic producer using a logical `[BL,H]` query-row tile also
passes its strict isolated gates, but does **not** make projected logits bitwise
equal in any of the four saved cases. Matching logical tile shape alone is not
a correction. Compiled original and producer kernels have the same hidden
thread/warp topology in the observed case, but their dot-product contraction
sequences differ: the original uses a scalar FMA chain where the producer
materializes additional rounded products and additions. This is stronger than
an inference from logical tile dimensions.

A diagnostic-only H1024/BL8/four-warp producer explicitly reproduced the observed
original FMA contraction order. Its cached-source logits and inverse RMS are
bitwise equal to original FLA for all four saved cases. For consumers 16, 22 and
24, the complete forward output, logits and log-sum-exp also become bitwise equal.
Consumer 31 retains 2,541 differing local-source logits and 41 BF16 output
elements because its separate single-local-source fast path is unchanged. This
isolates a concrete producer arithmetic discrepancy and an additional consumer
discrepancy in these inputs; it does not qualify a corrected training trajectory.

The first contraction diagnostic did not compile because the installed Triton
lacks `tl.mul_rn`. Its preserved second revision uses explicit `mul.rn.f32`
inline assembly and completes all four cases with unchanged forward comparison
gates. Original FLA, the production producer/consumer, model gradients and
training tolerances were not modified. This specialized diagnostic has no timing
result and does not establish portability to other hidden sizes or compiler
configurations. A production correction still needs new operator, real backward
and independent training validation.

## Validation and reproducibility

Executed focused checks include 149 integration test passes, 80 graph/primitive
regression passes, and two NCCL transport cases passing on both ranks. These are
separate test selections with overlap, not 229 unique cases. The integration
selection includes real two/four-rank Gloo transport. The final qualification
source changes only a copyright header, docstring and equivalent string wrapping
from that unit-tested revision; its executable AST is unchanged.

The production source is based on parent graph commit
`d28a16486f1d3e054b2c75e2053b42d7879605b1`. Exact qualified identities are:

| Artifact | SHA-256 |
| --- | --- |
| Complete 587-file production Python tree | `0f6b3e4ae47aa25e364569a10434908780bd85c1f0a31dabdbeceb73317db850` |
| Source archive | `0ec07d7b005bf4e704009f71d0c85c6ac7727a7161ca70b6a9cedd89dc0ff287` |
| Integration qualifier | `b78c7b1e936b69b291b3da3ddbf530a9186ab3dcb297cc0ba26a7ac2c4541258` |
| Direct OFF/F0 comparison report | `ca898d827328fb5c43e40eda0a2bf2d4dd677fa129588fbfac1e3e9562de4370` |
| Original installed FLA module | `8ccfc512fded550b36314f459c69e0ed7af3f7bf35b21989b60da22b01f22371` |

Evidence is retained in the companion development workspace under
`runtime/attnres-source-20261007/framework-experiments/source-balance-graph/`:
`qual-{off,f0,f05,f1}-r3/`, `direct-off-f0-r3.json`,
`control-benchmark-analysis.{json,md}`, `primitive-r2-bench-t4096.json`,
`events-f0-r1/`, `f0-events-analysis.{json,md}`,
`math-f05-r1/`, `math-f05-analysis.md`, `math-replay-f05-r1/` and
`math-layout-f05-r1/`, `math-fma-f05-r2/` and `compiled-math-analysis.md`.
These generated artifacts are not installed with Megatron.

Any arithmetic correction must pass the unchanged operator and independent
training gates again, including the direct graph comparison, before timing.
A future performance claim must beat the qualified disabled graph baseline
including integration costs and explain the corresponding PP dependency change.
No K3-wide, backward-work or intrinsic fill/drain reduction follows from the
current evidence.
