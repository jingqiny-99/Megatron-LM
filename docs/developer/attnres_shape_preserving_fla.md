# Shape-preserving FLA Attention Residual calls

## Scope and motivation

This follow-up starts from the original Attention Residual/offload branch at
`e54c5cfc5def622cdac1e2732cb009ed3e5b6a07`. It reduces autograd graph overhead at
each Attention Residual consumer. Projection, aggregation arithmetic, FLA
kernels, pipeline payloads, and pipeline scheduling are unchanged. It introduces
no source-side projection or compute-balance algorithm.

FLA 0.5.1's public `fused_attnres` wrapper reshapes every source to `[N, H]`,
calls `FusedAttnresFunction`, and views the result back to its original shape.
For contiguous inputs these are metadata operations, not GPU copies. However,
they add one `ViewBackward` node per source plus one for the output at every
consumer. Source reuse across layers makes that host autograd work frequent and
can delay when a pipeline peer receives its next input or gradient.

The internal FLA function already supports the original shape: it derives
`N = numel / H`, allocates the output in the first source's shape, allocates
statistics as `[sources, *token_shape]`, and allocates each source gradient with
`empty_like`. Its kernel indexing is linear in `N` and `H`.

## Implementation and dependency contract

`_get_fla_fused_attnres` lazily imports the private
`fla.ops.attnres.fused.FusedAttnresFunction` and caches a callable adapter. The
adapter passes contiguous sources in their original shapes directly to the
same autograd function. A module-level helper and `functools.partial` keep the
callable pickleable; there is no per-forward backend import or global FLA patch.

The existing `flash-linear-attention==0.5.1` dependency pins in `pyproject.toml`
are the compatibility basis. This private class is not a stable upstream API;
upgrading FLA requires rechecking its signature, shape allocation, indexing,
autograd behavior and the parity tests below. Production execution does not
depend on an installed-file hash and does not silently select another kernel
if the private class is unavailable. Missing FLA still produces the existing
optional-dependency error when the FLA backend is selected; eager/compile
backend selection is unchanged.

The returned callable retains the public wrapper's arguments, empty-source and
checkpoint-level validation, output dtype/shape, optional output RMSNorm,
optional nondifferentiable depth probabilities, scale and checkpoint forwarding.
Noncontiguous sources still receive contiguous copies with autograd connections
to their original storage. Sources are neither detached nor deduplicated; repeat
references and shared storage retain normal gradient accumulation. The existing
one-source shortcut in `AttentionResidual.forward` is unchanged.

## Validation and limits

`tests/unit_tests/transformer/test_attention_residual.py` retains the independent
native PyTorch reference for module outputs, source gradients and both parameter
gradients. The wrapper regression matrix separately compares against the
unchanged public FLA operator using independently cloned input storage:

- BF16 sources, nonzero FP32 pseudo-query and RMSNorm weight, H1024;
- 2D `[128, 1024]` and 3D `[64, 2, 1024]` layouts;
- one, two, three and six sources;
- contiguous inputs, hidden-strided inputs, repeated tensor objects, and distinct
  views sharing storage, including the unused storage's zero gradient;
- exact output, probability and all original-storage/parameter gradient equality;
- the expected reduction of `sources + 1` view-backward nodes for contiguous
  inputs, without treating views as device copies.

Additional bounded operator cases cover checkpoint levels 0/1 and optional
output RMSNorm, including its parameter gradient. They do not establish new
full-model support for those options. Invalid checkpoint levels and empty input
retain their errors. Run the CUDA cases with the pinned FLA installed; CPU-only
test collection intentionally skips that backend's numerical tests.

Before production integration, a runtime-only adapter of the same shape-preserving
call passed 32 operator cases bitwise and ten complete independent training
updates for original AttnRes with PP2/VP1 and legal PP2/VP2. The tested workload was
16 layers, H1024, sequence4096, MBS1/M8/GBS8, BF16, TP=CP=EP=DP=1, no offload,
MTP, MoE or recomputation. Both arms used
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` after a separate original-baseline
repeatability failure was diagnosed; numerical gates were unchanged.

Alternating paired measurements of that runtime adapter, three blocks of fifty
complete updates per arm, showed throughput gains of 3.13–3.51% for VP1 and
3.97–4.95% for VP2. These are bounded experimental results, not a guarantee for
other model sizes or parallel layouts. Nsight critical-path attribution and production integration results are
reported separately below.
Raw experiment artifacts reside in the development toolkit under
`runtime/attnres-source-20261006/wait-experiments/viewfree/`.

This change does not claim to eliminate intrinsic pipeline fill/drain or all
stage compute imbalance. It removes repeated host graph work that can contribute
to peer readiness delays. Offloading, CP, MTP, full-model CUDA graphs and other
unmeasured combinations remain outside this follow-up's performance evidence.

## Production verification

The complete Attention Residual unit-test file passed on GB200: **140 passed**,
including all CUDA FLA cases. The actual production getter, with no candidate
runtime replacement, also passed ten independent updates against a reference
using public FLA for each of PP2/VP1, PP2/VP2 and PP4/VP2. The reference and
candidate use separate model and optimizer storage with initial state copied
only once. Loss, every reduced main gradient, model parameters and live FP32
pseudo-query updates are checked using unchanged gates.

A separate complete 585-file source manifest identifies this production tree;
only attention_residual.py differs from the original source manifest. The
qualification harness records the model-realized TE backend environment and
checks it remains unchanged. Old runtime-adapter results are not reused as
production qualifications. Paired production benchmarks passed raw-sample audit:


| Pure PP layout | Public FLA mean ms | Production mean ms | Throughput gain | Paired block median gains |
|---|---:|---:|---:|---:|
| PP2 / VP1 | 298.030 | 286.319 | 4.09% | 3.20--5.22% |
| PP2 / VP2 | 274.693 | 266.154 | 3.21% | 1.94--4.26% |
| PP4 / VP2 | 186.991 | 182.382 | 2.53% | 1.85--3.08% |

Measurements use the geometry and deterministic TE setting above on one GB200
node, with three alternating AB/BA/AB blocks per topology. Each block has ten
warmups and fifty complete updates per arm. Timings include zeroing, the full
pipeline schedule, gradient finalization, clipping, Adam and ending device
synchronization; no profiler or per-iteration rank barrier is present. Latency
is the maximum rank for each update; pooled throughput is inversely proportional
to mean latency at identical tokens/update. All nine paired blocks favor the
production implementation, including reversed-order blocks. Block dispersion
is shown rather than treating every update as an independent experiment.

Per-rank allocated peaks match exactly in every paired block. The maximums are
4.457, 4.911 and 4.024 GiB respectively; both models and optimizers remain
resident, so these are not standalone-model memory measurements. Reserved
allocator peaks vary slightly in two VP2 blocks, while overall maximum reserved
memory is unchanged. This is not a memory-saving claim.

A separate same-node original/runtime-adapter Nsight comparison shows essentially
unchanged GPU compute and earlier producer readiness. One matched late gradient
send occurs 8.468 ms earlier in its profiled step; its receive is posted earlier
too and shrinks only 0.779 ms. The approximately 63 us completion tail and final
stage compute skew remain. Nsight adds material overhead, so these milestones
explain the mechanism but do not supply the throughput numbers above.

Black26.3, isort, pylint, Ruff and copyright checks pass. The repository's advisory
mypy check reports FLA's absent typing metadata on both original and changed
sources; no suppression was added. These results apply to the bounded dense,
BF16, pure-PP proxy. Larger models and enabled activation offloading require their
own measurements before extending the performance claim.
