# Experimental Attention Residual graphs for the final virtual chunk

## Scope and configuration

This opt-in extends the [ordinary-PP decoder-stage graph](attnres_stage_graph.md)
to dense GPT with PP2/VP2. It captures the final local virtual chunk on each
pipeline rank. Earlier chunks execute eagerly, preserving the original
Attention Residual source-cache publication and cross-chunk gradient drain.
Embedding, output projection, loss, pipeline communication, gradient
finalization and optimizer remain eager. The original FLA arithmetic, source
placement and pipeline schedule are preserved.

Pass these direct MCore fields when constructing `TransformerConfig`, along
with the existing BF16/FLA PP2/VP2 model settings:

```python
graph_options = dict(
    attn_res_vpp_final_chunk_cuda_graph=True,
    attn_res_stage_cuda_graph=False,
    cuda_graph_impl="local",
)
config = TransformerConfig(**model_options, **graph_options)
```

The two graph modes are mutually exclusive. The VPP mode requires PP2/VP2,
TP=CP=EP=DP=1, dense GPT, BF16, the FLA Attention Residual backend, fixed
shapes, zero dropout and full decoder graph scope. Hybrid/K3, MoE, MTP,
activation offloading, recomputation, FP8/FP4, gradient accumulation fusion,
variable sequence lengths and inference remain unsupported. Other VPP sizes
and pipeline layouts have not been qualified.

Select the supported TE RNG tracker before constructing either models or
graphs with `initialize_rng_tracker(use_te_rng_tracker=True)`, and seed with
`model_parallel_cuda_manual_seed(seed, te_rng_tracker=True)`. The native local
graph manager records the complete initial interleaved schedule and captures
its forward/backward runners through the existing scheduler integration.

## Forward boundary and source ownership

The original VPP protocol publishes detached source leaves into a rank-local
cache keyed by microbatch. Later local chunks consume these leaves; earlier
chunks' GradTaps eventually drain their gradients. Capturing Python cache
updates or the dynamic `leaf.grad` branch would freeze those effects at
capture time. The final local chunk provides a bounded graph boundary because
no later local chunk consumes sources that it creates.

For each actual final-chunk forward, the host adapter performs the original
payload unpacking and source-cache lookup, including incoming-source taps.
It passes the partial tensor, every source as a separate `source_N` tensor,
and the actual microbatch ID to the graph manager. These are flat arguments:
the native runner does not discover tensors nested inside an opaque tuple.

The captured body copies every source and the partial into owning storage,
then runs the original decoder body with an explicit
`AttnResStageSources(manage_cache=False)` state. This private state mode is
restricted to the final virtual chunk. It skips only the host entry work
already performed; local block construction, aggregation and payload packing
retain their original implementations. The host evicts the real microbatch's
cache metadata exactly once after the actual forward. Metadata eviction does
not end the autograd lifetime of the original cached leaves.

Owning copies inside the graph are required because native graph input
surfaces can lose storage ownership after capture. An autograd context that
retains the same input Tensor object does not prevent its storage from being
reused under overlapping forwards. The copies remain live through their
backward. Native output preservation and the recording path's viewless output
also protect pipeline pseudo-deallocation.

Only the final decoder owns a graph manager. Earlier chunks delete their
unused manager at construction; child layer managers are suppressed across
both chunks. The measured M8 workload uses eight native forward/backward
runner pairs per rank, with no additional slot-reuse policy.

## Backward ownership and original gradient drains

The graph computes the complete final-chunk gradient for each explicit source.
An eager identity autograd adapter, `_AttnResGraphInputGradientOwner`, clones
each returned gradient into ordinary owning storage before outer autograd
accumulates it into the original cached leaf. This copy is necessary: native
graph-pool gradient storage may be recycled before the earlier producing
chunk reaches its scheduled backward. The producer then executes the original
`_AttnResGradTap.backward`, adds the cached contribution once and clears the
leaf's `.grad`. Copying a BF16 tensor adds no reduction or rounding operation.

Three source cases have different lifetimes:

| Source | External contribution and drain |
| --- | --- |
| Previously cached source | The final graph returns an owning gradient; the earlier eager producer drains it. |
| Newly received entry source | Its tap executes on the host. There is no later local consumer, so its external leaf gradient stays absent. |
| Source formed inside the final graph | Its original tap is captured. Its leaf is never published to a later local consumer, so the no-external-gradient branch is structural. |

Remote contributions still follow the original outgoing payload edges. No
captured tap performs a dynamic cross-chunk drain. Native graph backward
continues to accumulate into real DDP `main_grad` buffers, record completion
events and invoke the existing outer readiness hooks. The implementation adds
no synthetic parameter gradients or replacement parameter hooks.

Graphing earlier virtual chunks would additionally require explicit source
exports and stable source-gradient inputs before their backward, plus a host
once-only drain contract. This implementation does not provide that wider
runtime interface.

## Validation status

All 36 focused GPU tests pass: the 22 ordinary-PP tests and 14 VPP tests.
Coverage includes mode/topology guards, host-only cache lifecycle, owned
source values after input storage replacement, returned-gradient ownership
after graph-buffer recycling, and the original producer's once-only drain.

The distributed qualification uses L16/H1024/FFN2048/S4096/vocab4096,
Attention Residual block size3, BF16, PP2/VP2, MBS1/M8/GBS8 and microbatch
group size2 on two GB200 GPUs. Two models and real Adam optimizers receive
one initial-state copy, then evolve independently for ten updates with
changing samples and live nonzero queries. Both backwards finish before
either optimizer step. All ten updates pass the unchanged loss, every
`main_grad`, model-weight, FP32-master and Adam-state numerical gates.

Capture checks preserve parameter/master/optimizer values and storage
addresses, restore execution buffers and RNG contents in place, and clear
real DDP readiness state. Each update proves eight actual forward and eight
actual backward graph replays per rank; only the final decoder owns a
manager. Host observers verify source publication, final-visit eviction and
complete original gradient drains, with cached gradients distinct from
recyclable graph-gradient surfaces.

The qualified core/training source hash is
`b4e58b98b9ef2cb3b1b3804c1500856fc5af0c4eb764743407008d0541da5501`.
The development toolkit stores its source manifests, qualification results and
scripts under `runtime/attnres-source-20261007/framework-experiments/`, including
`qual-vpp-final-chunk-r2` and `vpp-stage-graph/`.

A separate diagnostic observes the actual owning clone buffers using TE
raw-pointer wrappers that retain no original storage. It checks exact values
after each forward and until the corresponding backward. The original
schedule overlaps final-chunk forwards on rank0 (`F0 F1 B0 B1`); rank1 pairs
each final-chunk forward with its backward (`F0 B0 F1 B1`). The diagnostic
therefore requires cross-slot coverage on rank0 and reports rank1's lack of
overlap explicitly. Both ranks pass: rank0 has 96 exact lifetime checks,
including 32 cross-slot checks, and rank1 has 80 exact checks with no cross-slot
overlap. Its artifacts are `diag-vpp-final-chunk-r2`. CPU byte snapshots
synchronize this diagnostic; it is excluded from performance measurements.

## Performance and memory

On the same PP2/VP2 workload, four alternating paired AB/BA blocks, each with
ten warmup and fifty measured complete Adam updates per arm, measure
**268.650 → 156.394 ms/update**, using the maximum rank latency for each
update. This is **1.7178× throughput**, or **41.79% lower update latency**.
The four block ratios are 1.7147, 1.7250, 1.7136 and 1.7179. Each arm has
200 measured updates; paired blocks are the replication unit, rather than
treating every iteration as statistically independent.

The unprofiled host wall-clock interval includes zeroing gradients, the whole
forward/backward schedule, gradient finalization, real Adam and the ending
CUDA synchronization. Host entry/cache/tap work, input/output staging and all
forward/backward ownership copies are included. The reference uses the same
source tree with the graph opt-in disabled; its eager path is bound to the
unchanged shape-preserving Attention Residual implementation. The environment
is Torch2.12/NV26.04, CUDA13.2, TE2.16 and FLA0.5.1, with TF32 disabled and
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`.

Both models, optimizers and graph pools remain resident during paired timing.
The observed allocated peaks are therefore shared-process measurements:

| Rank | Eager allocated peak | Graph allocated peak | Reserved peak in either arm |
| --- | --- | --- | --- |
| 0 | 5.598 GiB | 5.171 GiB | 7.822 GiB |
| 1 | 5.109 GiB | 5.104 GiB | 6.846 GiB |

Capture setup adds approximately 788 MiB and 1028 MiB of allocated memory on
ranks 0 and 1, respectively, including its persistent graph/input surfaces.
These observations do not establish standalone graph capacity or a model
memory saving. The benchmark artifacts are `bench-vpp-final-chunk-r2`.

This result applies to the qualified dense PP2/VP2 proxy with only the final
local chunk graphed. Earlier chunks, pipeline fill/drain and communication
dependencies remain. It does not establish a native K3, offloading, VP4 or
large-model speedup, or quantify the individual bubble components without a
separate trace analysis.
