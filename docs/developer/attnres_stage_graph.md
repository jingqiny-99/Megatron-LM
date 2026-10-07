# Experimental Attention Residual decoder-stage CUDA graphs

## Motivation and supported scope

Attention Residual fanout adds host autograd and kernel-launch work inside each
pipeline stage. This opt-in captures the complete dense GPT decoder stage with
the existing local CUDA graph manager. Embedding, output projection, loss,
pipeline communication, gradient finalization and optimizer remain eager.
The original FLA arithmetic, depth-source placement and pipeline schedule are
unchanged; no source-side projection or compute-balancing algorithm is added.

Enable `TransformerConfig.attn_res_stage_cuda_graph=True` together with
`cuda_graph_impl="local"`. This is a direct MCore configuration option. The
initial implementation requires ordinary PP, dense GPT, BF16, FLA AttnRes,
fixed sequence shapes, zero dropout, TP=CP=EP=DP=1, and full decoder graph
scope. VPP, Hybrid/K3, MoE, MTP, activation offloading, recomputation, FP8/FP4,
gradient accumulation fusion and inference are rejected. Other AttnRes CUDA
graph combinations retain their existing rejection.

For TE attention, select the supported TE RNG tracker **before constructing
models** using `initialize_rng_tracker(use_te_rng_tracker=True)`, and seed with
`model_parallel_cuda_manual_seed(seed, te_rng_tracker=True)`. Changing the
tracker after TE modules or graphs have been created is invalid. The native
local graph manager records the first complete pipeline schedule and captures
its forward/backward runners through the existing scheduler integration.

## Ownership and backward behavior

The decoder owns the graph manager; nested transformer layers remain ordinary
modules. On nonfirst stages, the received PP tensor is an explicit graph input.
The capture adapter temporarily binds the decoder to that static input and
restores the live `input_tensor` in `finally`, avoiding replay of stale Python
state. The decoder is both the first and last local graph boundary, so native
input staging, output-gradient staging and output preservation apply. The
recording path also returns a viewless output for pipeline pseudo-deallocation.

First-stage AttnRes retains its initial source directly in FLA's autograd
context. The native runner weakens its static input tensor's storage after
forward capture. Retaining that same tensor object is insufficient: under
`F0, F1, B0`, the graph pool can reuse its storage before B0. The adapter makes
an owning source clone **inside the graph** before the original decoder body.
This source remains live until backward; the outer staging input can be
recycled. Nonfirst-stage payload unpacking already creates independently
owning views. The extra first-stage copy is included in measured execution.

Native graph backward accumulates parameter gradients into real DDP
`main_grad` buffers and records replay-completion events. Existing DDP hooks
consume those events and mark parameters ready. There are no substitute
parameter hooks or synthetic gradients. The prototype uses eight native
forward/backward runner pairs per rank for the measured M8 workload; it adds
no new slot-reuse policy.

## Validation and measurements

Focused tests cover unsupported configurations, actual PP input selection,
input restoration after exceptions, differentiable source ownership after
input storage replacement, and the viewless-output contract. All 22 pass in
the GPU training container.

Full-model qualification uses two independent models and real Adam optimizers,
one exact initial-state copy, then ten independently evolving updates with
changing input samples. Both backwards finish before either optimizer step.
The unchanged gates compare losses, every reduced gradient, weights, FP32
masters and Adam moments/scalars. Capture must preserve parameters, masters,
optimizer state and storage addresses; only execution buffers and RNG contents
are restored in place. Each update proves eight actual forward and eight
actual backward graph replays per rank, complete decoder parameter coverage
and ready-state ownership. All ten updates pass.

The initial measured workload is L16/H1024/FFN2048/S4096/vocab4096, AttnRes
block size3, BF16, PP2/VP1, MBS1/M8/GBS8 on two GB200 GPUs. It uses TE2.16,
FLA0.5.1, Torch2.12/NV26.04, TF32 disabled and
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`. The reference is the shape-preserving
AttnRes implementation at `0fa49b269953ae5eae8e28dd58d8c26eb3cf8eb8`.

Four alternating paired AB/BA blocks, each with ten warmup and fifty measured
complete Adam updates per arm, measured **285.044 → 72.593 ms/update** using
the maximum rank latency for each update: **3.9266× throughput**. Block ratios
were 3.9678, 3.8706, 3.9541 and 3.9140. Input/output/gradient staging and source
preservation copies are included. This is a dense proxy result, not a native
K3, VPP, offloading or large-model speedup claim.

Both models, optimizers and graph pools remain resident during paired timing.
Per-arm allocated peaks therefore describe transient work with shared resident
overhead; they do not measure standalone graph capacity or memory savings.
Pipeline fill/drain and communication dependencies remain. Trace diagnostics
and unprofiled throughput are reported separately.

Raw source manifests, qualification state, rank timing samples and traces are
in the development toolkit's
`runtime/attnres-source-20261007/framework-experiments/`, including
`qual-stage-graph-pp2-vp1-r3`, `bench-stage-graph-pp2-vp1-r3`, and
`profile-stage-graph-pp2-vp1-r3`. The subsequent formatting-only revision has
the same parsed Python AST and also passes ten updates under its own source hash
(`fc034a8dd9985073b47f091902c08345aeac0122b1530e20dd85471db5a998d7`).
A fresh-process four-block repeat on this final source measured
**287.688 → 72.571 ms/update, 3.9642× throughput**, with block ratios
3.9774, 3.9961, 3.9257 and 3.9577. Its artifacts use the corresponding `r4`
suffix. Black26.3, isort, pylint, Ruff and copyright checks pass.
