<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# Full virtual-pipeline Attention Residual stage graphs

Set `TransformerConfig.attn_res_vpp_cuda_graph=True` and
`cuda_graph_impl="local"` to capture both decoder chunks in the bounded
PP2/VP2 Attention Residual workload. This is mutually exclusive with ordinary
PP stage graphs and `attn_res_vpp_final_chunk_cuda_graph`. The implementation
retains the final-chunk path and adds explicit source outputs for earlier
chunks. The bounded dense proxy passes the numerical and lifetime gates below.

Supported scope remains dense BF16 GPT with the FLA Attention Residual backend,
TP=CP=EP=DP=1, fixed shapes, zero dropout, and no offloading, recompute, MoE,
MTP, hybrid attention or fused gradient accumulation. Each earlier chunk must
contain at least one local block start; construction rejects layouts without
a source to export. Larger VP/PP sizes and arbitrary layouts need separate
lifetime and numerical qualification.

## Source ownership and forward behavior

Every actual forward enters through the original host
`AttnResStageSources.enter`. Incoming delta-payload sources receive the
original host taps. The initial embedding remains a partial sum until layer 1
appends b0, preserving its original production position.

The native graph manager receives the partial, each depth source as a separate
`source_N` argument, mask/RoPE and the actual microbatch ID. A pure body clones
every input into owning storage and runs the original decoder using
`manage_cache=False`. These clones protect tensors retained by FLA backward
from the native runner's weakening and reuse of staging-input storage.

In an earlier chunk, `_AttnResGraphSourceTap` produces two differentiable
identity views at each original block start. One feeds local consumers and the
outgoing payload; the other becomes an explicit graph output. The output tuple
is `(payload, source_export_0, ...)`, in the source order derived from layer
block-start metadata. All exports are differentiable; no runner support for
nondifferentiable outputs is required.

The host clones each export into a fresh detached leaf with independent storage.
It appends those leaves to the original incoming cache leaves and publishes the
actual microbatch cache once. Graph-output storage can subsequently recycle
without changing values needed by the later local chunk. Final chunks preserve
the original final-visit eviction. Autograd references keep pending leaves alive
after their forward cache metadata disappears.

For L16 and block size 3, each chunk owns four layers:

| Physical rank | Earlier chunk | Host incoming taps | Graph source exports |
| --- | --- | --- | --- |
| 0 | Layers 1–4 | None | b0, b1 |
| 1 | Layers 5–8 | b0, b1 | b2 |

Final chunks receive their usual cached prefix and incoming delta sources.
They have no source exports, because no later local chunk consumes their new
sources.

## Backward and addition order

The eager `_AttnResGraphOutputBridge` consumes the payload and every source
export, returning only the payload to the schedule. Its autograd context owns
an immutable tuple of this invocation's leaves, source IDs, microbatch ID and
VP stage. It does not read mutable current-microbatch fields during backward.

The later local chunk completes backward first. Existing
`_AttnResGraphInputGradientOwner` adapters copy native graph-pool input
gradients into ordinary owning storage before accumulation into cached leaves.
At producer backward, the bridge requires every external gradient to be
present, drains each leaf exactly once, and returns those tensor gradients on
the corresponding differentiable graph outputs. It checks all leaves before
clearing any; missing or repeated drains are errors.

Native output-gradient staging copies the returned gradients into static
backward input buffers. The captured source tap then adds
`local_gradient + external_gradient` exactly once, in BF16, after the local
consumer and outgoing-payload fan-in has accumulated. This is the original
tap's addition point and operand order. The new tap rejects a missing local or
external gradient instead of substituting zero.

Original incoming-payload `_AttnResGradTap` nodes still execute eagerly and
drain their leaves at their existing points. Original final-chunk internal taps
remain captured with structurally absent later-local gradients. Original tap,
cache implementation, FLA kernels, communication, optimizer, DDP main-gradient
accumulation and replay-completion events are unchanged. All ownership and
payload preservation copies belong to the timed model path.

For eight microbatches, each rank has sixteen forward and sixteen backward
stage replays. Rank0 drains sixteen exported leaves and zero original host
cached leaves; rank1 drains eight exported leaves plus sixteen original host
cached leaves. Final-visit incoming taps have no later-local external
contribution and are separate from these true drains.

## Validation contract

Focused tests cover exact BF16 addition order, missing-gradient rejection,
once-only drains, source output order, cache eviction before backward,
captured input ownership, owning host exports, viewless pipeline outputs and
scope restrictions.

Distributed qualification must compare ten independently evolving updates
against the unchanged eager model, checking original loss, main-gradient,
model-weight, FP32-master and Adam gates. Runtime observers must prove every
real graph replay, publication, eviction and source drain. A separate
synchronized diagnostic checks owned captured sources and exported host leaves
while other microbatches remain pending; it is excluded from timing.

The native capture lifecycle closes globally after one complete recording
schedule. Full and final-only graph modes must be qualified in separate fresh
processes, each against eager. Do not reset native capture globals or pool
ownership to make a second graph model fit the same process. Any comparison
of full versus final-only timing must disclose this separate-process setup.
No performance claim is authorized by these focused tests alone.

## Qualified dense proxy

On two GB200 GPUs, L16/H1024/FFN2048/S4096/vocab4096, block size 3,
PP2/VP2, MBS1/M8/group2, both full and final-only modes pass ten independent
Adam updates against eager. Each comparison has 80 bitwise-equal loss scalars
and passes all original gradient, weight, FP32-master and Adam-state gates.
All 55 focused tests pass. Across ten updates, full mode proves 640 actual
forward/backward graph launches and the same 400 once-only cached-gradient
drains as eager. The separate lifetime diagnostic passes 568 captured-input
and 561 exported-source value checks, including pending-microbatch overlap.

Unprofiled complete-update timings include zero-grad, all copies, the original
pipeline schedule, gradient finalization, real Adam and ending synchronization.
Each mode uses four alternating AB/BA blocks with ten warmups and fifty measured
updates per arm/block; samples use the slower rank.

| Graph mode | Eager mean ms | Graph mean ms | Throughput ratio |
| --- | ---: | ---: | ---: |
| Full VPP | 279.103 | 66.291 | 4.210x |
| Final chunk only | 279.426 | 162.659 | 1.718x |

Full-mode paired block ratios range from 4.180x to 4.253x. Comparing graph
means gives 2.454x additional throughput over final-only mode. These modes run
in separate fresh processes on the same node and follow-up source; this is not
a direct same-process comparison or a measurement of the literal parent
revision. Their eager baselines differ by 0.116%.

Both models, optimizers and graph pools remain resident in each experiment.
Allocated peaks on rank0 are 6.098/3.916 GiB for eager/full, and rank1 peaks
are 5.734/4.416 GiB. Reserved peaks are 9.693/8.414 GiB by rank in either arm.
These shared-process values do not establish standalone model memory savings.
The qualified source preserves original Attention Residual placement; no
source-projection balancing, CP, offloading or native K3 result is claimed.

A separate two-update profiler capture shows host kernel calls falling from
6200 to 440 on rank0 and 6529 to 593 on rank1, with 32 graph launches per rank
and update. FLA forward/dV/dqdw counts remain 120 on rank0 and 136 on rank1.
The original 48 Work.wait calls per rank/update remain; device kernel counts
increase by 56/96 because the ownership copies execute real work. The gain
comes from reduced host dispatch and autograd overhead across both chunks.
Intrinsic pipeline fill/drain and stage compute imbalance remain. Profiled
durations do not determine the unprofiled speedup above.
