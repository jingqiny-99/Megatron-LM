# Attention Residual + pure PP bubble-reduction plan

## Corrected objective and hard scope

Re-analyze the timeline of ORIGINAL Attention Residual with pipeline parallelism,
then implement and evaluate optimizations in evidence-based priority order.
The user's corrections explicitly supersede the earlier source/DP-oriented plan.

- Baseline code: offload follow-up `e54c5cfc5def622cdac1e2732cb009ed3e5b6a07`,
  before the source-projection / compute-balance implementation.
- Branch: `jingqiny/attention-residuals-wait-reduction`, now based directly on e54.
- TP=CP=EP=DP=1. WORLD_SIZE must equal PP. Study ordinary PP and interleaved VPP.
- No source projection, query-bank publication, score payload, or compute-balance
  algorithm. All imported Megatron code must resolve inside the pinned checkout.
- Initially disable offloading, MTP, MoE and recomputation. They are not targets
  of this goal. The offload follow-up is a source-code baseline, not an enabled
  offload experiment.
- DDP remains only the ordinary DP1 model/optimizer wrapper. No bucket/alignment
  tuning or cross-DP wait analysis is part of this plan.
- Prior 9ab FLA/DP2 traces are historical context only. That branch changed common
  pipeline code, so disabling its source flag is not an original baseline.

The withdrawn plan is preserved in the toolkit artifact directory as
`wait-experiments/withdrawn-plan-source-and-dp.md`; it is not an execution input.

## Original-code constraints already rechecked

The original CLI prohibits PP2 with VPP when P2P overlap is disabled. Its batched
same-peer posting order swaps directions even for the original single-tensor
path. Do not call the core fixture with that forbidden configuration. Legal
initial cells are PP2/VP1 without overlap and PP2/VP2 with overlap=True,
batch=False. PP4/VP1 and PP4/VP2 are independent topology controls. Full-model
CUDA graphs and arbitrary dense uneven stage layouts are unsupported and may
not be enabled by deleting guards.

## Fixed model and measurement contract

Initial proxy: GPT16L, H1024, FFN2048, 8 heads, sequence4096, vocabulary4096,
BF16, FLA Attention Residual, block3, zero dropout, TF32 disabled, nonsharded
Adam with FP32 master weights and global norm clipping1.0. Initial MBS1/M8 gives
GBS8 for every topology because DP1. Keep all these fields, seeds, data and
optimizer settings fixed across scheduling comparisons.

Use four GB200s on one held node; torchrun starts exactly PP processes. Record
actual hardware, image digest, package versions, NCCL/NVTE overrides, source
and driver hashes, imported module paths, per-rank placement and actual payload
shapes. No unit-test-only NCCL restrictions are added to benchmark runs.

Correctness before comparative performance:

1. Each architecture gets its own independent self-repeatability control.
2. Scheduling/implementation candidates use exactly matched initial state once,
   separate model/config/optimizer storage, and at least 10 independent updates.
3. Compare loss, every reduced gradient before clipping, and weights after both
   optimizers step, using unchanged existing numerical gates. No per-step state
   recopy, fallback kernels or threshold changes.
4. AttnRes ON/OFF is an architectural diagnostic, not a numerical-equivalence
   claim. OFF is constructed with enable_attention_residuals=False and
   attn_res_block_layers=None. Qualify each independently; do not compare their
   different functions using a fake parity gate.
5. A failed candidate is untimed until diagnosed and fixed. An asymmetric
   distributed failure terminates that fresh torchrun; no cleanup barrier is
   entered after a rank fails.

Unprofiled measurements include zeroing, complete PP schedule, required norm
synchronization and optimizer update. Use 10 warmups and three repetitions of 50
updates, gather rank data outside timing, and report max-rank latency, tokens/s,
dispersion and memory. No per-iteration world barrier is added. Repeat the
anchor around a promising candidate. Profile only short diagnostic windows and
measure overhead separately; instrumented idle durations are not speed claims.

## Ordered priorities and concrete deliverables

| Priority | Work | Required evidence |
|---|---|---|
| P0 | Fresh original-code baseline and legal pure-PP topology matrix | Module/hash identity, independent self-controls, raw unprofiled samples, all-rank timeline |
| P1 | Re-analyze AttnRes-specific bubbles against standard-residual controls | Per-microbatch readiness, compute/communication/host gaps, fill/drain vs steady state, no DP confounder |
| P2 | Tune legal PP/VPP scheduling and overlap | Same-work10-step parity plus matched timing for each scheduling delta |
| P3 | Fix measured send-lifetime or synchronization serialization | Original-path ownership/order tests and multi-rank correctness, then whole-update timing |
| P4 | Reduce original consumer-side AttnRes backward/materialization overhead | Independent operator reference, unchanged gradients, model tests and critical-path benefit |
| P5 | Reduce static PP payload padding and packing costs | Exact bytes/shapes in both directions, cache lifetime coverage, end-to-end benefit |
| P6 | Validate selected combination and publish follow-up | Repeat matched baseline, representative PP/VPP matrix, documentation and signed reviewable commits |

Every entry in the live ledger records its parent cell, single changed factor,
qualification, timing, explanation and decision. Complete a priority with
accepted or measured-rejected candidates; do not substitute an easier goal.
If new evidence changes ordering, record the reason before proceeding.

### P0: fresh baseline, starting now

A. Pin e54 on the new worktree and remote source mirror. Verify imported Megatron
paths and source hashes; external helper files may supply existing test gates,
but never replace production kernels, communication or validation.
B. Run AttnRes self-controls for PP2/VP1 and legal PP2/VP2, DP1 in both.
C. Capture fresh timelines for the same cases and gather unprofiled latency.
D. Add standard-residual OFF controls at identical geometry and their own self
qualification. Add PP1 and PP4 controls where needed to distinguish local
AttnRes cost from pipeline amplification. Do not conflate topology changes with
a single overlap switch.

Fresh profile/capture is diagnostic and explicitly separate from the qualified
3x50 performance results. The first correctness/run failure is investigated on
the original baseline; it is not worked around by switching to the source branch.

### P1: re-analyze the new data

For each microbatch and virtual chunk, identify forward input/output readiness,
backward input/output readiness, send/receive submission/completion, layer and
AttnRes GPU work, and final norm/optimizer completion. Join sender and receiver
operations by direction/group/count, keeping host and GPU clocks distinct.

Partition observed gaps into mandatory fill/drain, unavailable peer data,
serialization despite independent ready work, GPU compute, and unexplained host
submission gaps. Attribute source-count-dependent forward and dV costs separately.
Compare AttnRes ON/OFF and PP1/PP2/PP4 without calling different architectures
numerically equivalent. Waiting intervals across ranks are not additive savings.
Produce a fresh bottleneck ranking before any implementation decision.

### P2: legal scheduling controls

For PP2/VPP2, steady P2P overlap is already required by the original baseline.
Test warmup/flush overlap and group2 versus group4 independently, with M8 fixed. PP2/VP1 has no
supported overlap flag. PP4/VPP2 can compare supported steady overlap modes with
its own fixed topology. Preserve send/deallocation guards initially.

Microbatch experiments must preserve tokens/update when used as latency
comparisons: for example, the GBS8 anchor can compare MBS2/M4 with MBS1/M8.
Compare both at the same tokens/update, not to older GBS4 latency.
Report the memory and compute-granularity tradeoff. More microbatches or VPP
chunks are not assumed to improve performance.

### P3: implementation only where the fresh timeline proves exposure

Investigate the original async forward-send wait before backward and the
batched device synchronization policy. A safe overlap change retains output
storage and Work handles until a later completed-send reap point, then releases
it. Preserve original value payloads, PP2 message direction, source-cache
ownership and once-only gradient drains. Do not delete waits while retaining
premature freeing, and do not enable unsupported configurations by bypass.

### P4: original Attention Residual kernels

Keep projection at its original consumer. Target measured source stacking,
copying, normalization, dV and gradient accumulation costs on the late path.
Use direct source reads/fusion only with unchanged mathematics and independent
reference tests. Measure kernel and complete-iteration effects separately.
No producer-owned future projection or rebalanced work placement is introduced.
Host/input-helper optimizations, if selected by evidence, are separately labeled
so synthetic fixture overhead is not presented as a production kernel gain.

### P5: value payloads only

Original VPP already transports source deltas. Investigate deterministic
boundary-specific shapes to avoid uniform padding, preserving sender/receiver
agreement in forward and backward and cached gradient ownership. Report useful
and actual bytes. There is no new FP32 score channel in this work. A byte
reduction qualifies as a speed optimization only if whole-update timing improves.

### P6: finish and review

Re-run the selected combination against the pinned original baseline at matched
workload and hardware, including ordinary PP and supported VPP, all numerical
gates, peak memory and traces explaining the change. Reject regressions or
scope support explicitly. Update the original module design documentation;
commit signed and signed-off changes on this branch and create/update the
user-authorized personal-fork follow-up PR with actual evidence.

The goal remains active until the required priorities have concrete outcomes.
## Execution ledger (2026-10-06)

### Baseline and attribution

P0's original e54 mirror passed its complete 585-file source manifest. Fresh
PP2/VP1 and legal PP2/VP2 ON/OFF captures, plus PP1/VP1 and PP4/VP1 controls,
all use TP=CP=EP=DP=1 and no source projection. PP1 contains no NCCL events but
still has local AttnRes dispatch gaps; those are not classified as PP bubbles.
With PP4, ON-minus-OFF stage compute grows 3.144/4.435/5.776/7.689 ms from
rank0 to rank3. Original AttnRes adds source-count-dependent stage imbalance.

Default-environment original ON repeatability failed before update5, while OFF
passed. Same-state backward differed in 79/74 tensors across ranks; selecting
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 made backward gradients bitwise equal.
Identical-gradient optimizer/model/master states were bitwise equal in both
environments. Subsequent qualification and performance fix that environment
for both arms, retaining all numerical gates. No default-environment timings
are pooled with these results.

P1 finds peer readiness, rather than link bandwidth, behind the longest PP
receives. In a clock-aligned Nsight VP2 step, a 22.072 ms receive starts
22.008 ms before its matching peer send; completion follows that send start
by 63.919 us. The sender must finish the preceding forward/backward first.
This is an instrumented example: Nsight injection alone adds about 12%, and
active collection about 36% relative to a no-injection control. PyTorch CPU+CUDA
collection adds about 70%, CUDA-only about 31%. Absolute gaps are diagnostic;
unprofiled whole-update measurements determine speedups.

### Scheduling decision

P2's baseline and both candidates passed ten-update qualification. All five
3x50-update runs are complete. Median-of-block-median latency is 312.364 ms for
VP1, 292.931/293.546 ms for VP2 before/after anchors, 287.092 ms for warmup/flush
overlap, and 288.547 ms for group4. VP2 baseline block medians span 284.789--294.637
ms. Small schedule signals are not separated from observed variation; retain
original group2 / flush-disabled settings. Group4 adds 0.978 GiB peak allocated
memory. VP1/VP2 are independently qualified, not a cross-topology parity claim.

### Consumer implementation

P4 advanced before P3 because the original FLA wrapper creates source/output
views at every consumer. The aggregate already reads a list directly, so there
is no source stack/cat to remove. A diagnostic late-stage backward has 792
ViewBackward nodes and 16.23 ms associated host time. GPU association may include
gradient accumulation; views themselves are metadata operations.

The runtime shape-preserving adapter passed 32 bitwise operator cases and ten
independent full-model updates for both PP2/VP1 and PP2/VP2. Same-process
AB/BA/AB measurements (three 50-update blocks per arm) give mean latency
295.435 to 285.305 ms for VP1 and 275.881 to 265.174 ms for VP2. Every paired block
favors the candidate: throughput gains 3.13--3.51% and 3.97--4.95% respectively.
Recorded per-rank peak allocation is identical. These are whole-update proxy
results; candidate peer-readiness attribution is being analyzed separately.

Production integration isolates the pinned FLA 0.5.1 private Function behind a
module-level shape-preserving adapter. It leaves arithmetic, kernels, PP
payloads and scheduling intact. All 140 tests in test_attention_residual.py
pass on GB200, including independent native parity, wrapper bitwise equality,
noncontiguous and aliased sources, and optional arguments. Actual production
PP2/VP1, PP2/VP2 and PP4/VP2 each passed ten independent updates under the final
harness, with exact realized-environment and source checks. Runtime-adapter
qualifications were not substituted. See developer/attnres_shape_preserving_fla.md.

### Remaining decisions and integration

P3 is deferred based on the original/candidate Nsight audit of 36 steady-state
fresh-send to older-backward boundaries. Original fresh sends take a median
48.416 us, but the earliest backward kernel API submission follows send
completion by at least 220.407 us. Twelve boundaries still await the incoming
gradient after that send completes. Boundary APIs include cudaStreamWaitEvent,
with no recorded host synchronization call; send duration is not CPU blocking
time. The capture does not establish the exposure needed to justify modifying
storage lifetimes and scheduling. No P3 code change is selected for this proxy.

P5 is deferred. Boundary-specific shapes could remove 22.2% of directed VPP
value bytes, but the first transfer has no padding. The matched long receive
is dominated by peer readiness, followed by about 63 us of protocol/transport/
synchronization. The byte estimate does not establish useful update latency
savings. Neither deferred proposal is claimed as an implemented optimization.

The qualified runtime adapter makes the same late sender ready 8.468 ms earlier
within one profiled update, while the receive is posted earlier too and shrinks
only 0.779 ms. GPU compute and final inter-rank skew remain effectively unchanged.
These are diagnostic milestones, not throughput gains. The original pipeline
data dependencies and intrinsic fill/drain remain.

P6 validation is complete for this bounded proxy. The final production helper
passed all 140 unit tests, independent review and ten-update qualifications for
PP2/VP1, PP2/VP2 and PP4/VP2. Each topology completed three paired blocks of
fifty full updates per arm on a second GB200 node. Auditing 2,400 rank samples
reconstructs all 900 max-rank samples and verifies source/qualification/realized
environment bindings. No production Python changed after these measurements.

| Layout | Original mean ms | Production mean ms | Throughput gain | Block median gain range |
|---|---:|---:|---:|---:|
| PP2/VP1 | 298.030 | 286.319 | 4.09% | 3.20--5.22% |
| PP2/VP2 | 274.693 | 266.154 | 3.21% | 1.94--4.26% |
| PP4/VP2 | 186.991 | 182.382 | 2.53% | 1.85--3.08% |

All nine blocks favor the production helper, including the reversed BA blocks.
Per-rank allocated peaks match exactly. Memory includes both resident models;
reserved allocator peaks vary slightly in two VP2 blocks. The selected change
and its validation/design documentation are ready for stacked review against
jingqiny/attention-residuals-offload-followup. P3/P5 remain explicit measured
priority deferrals, with no untested implementation included. These results do
not eliminate intrinsic PP fill/drain or source-count-dependent GPU compute skew.

Detailed artifacts and interactive reports are in the toolkit directory
runtime/attnres-source-20261006/wait-experiments/. In particular, analysis/,
nsys/analysis/, schedule-results.md and viewfree/results.md preserve raw sample
links, source/environment hashes and methodology limitations.
