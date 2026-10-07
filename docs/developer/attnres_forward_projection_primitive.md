# Attention Residual forward projection primitive

This experimental CUDA primitive exposes a standalone API for producer-side
forward projection without changing gradient ownership. The opt-in
[decoder graph integration](attnres_forward_projection_integration.md) uses it
with model-owned snapshots and forward-only PP metadata. Positive projection
placement has not passed the bounded independent training gate; see
[executed results and limitations](attnres_forward_projection_results.md).
The feature remains disabled by default.

For source `V` and a consumer's canonical FP32 query `q` and norm weight `w`,
the producer computes FP32 `D = sum(V * (q * w))` and
`R = rsqrt(mean(V * V) + eps)`. The consumer uses `D * R` in its depth softmax
and mixes the original BF16 values. Caches are computational hints for this
paired function, not independent differentiable inputs. The consumer invokes
the installed FLA `fused_attnres_bwd` unchanged for the complete `dV`, `dq` and
`dw`. There is no producer gradient path and no missing gradient replaced by
an artificial zero.

## API and bindings

```python
caches = project_forward_source(value, queries, norm_weights, eps=1e-6)
output, rstd, logit, lse = aggregate_with_forward_cache(
    queries[row], norm_weights[row], values,
    caches=consumer_caches, return_stats=True,
)
```

`value` is contiguous CUDA BF16 `[..., H]`; query and norm banks are contiguous
CUDA FP32 `[Q, H]`. The returned tuple has one `ForwardSourceCache` per row.
Each exposes a contiguous FP32 `.dot` tensor with the source's token shape,
and all rows share the same `.rstd` tensor. Neither has an autograd edge.
An empty query bank returns an empty tuple.

The consumer takes contiguous CUDA FP32 `[H]` query/norm views and a nonempty
ordered sequence of equally shaped BF16 sources. Its cache sequence has the
same length: each entry is a correctly bound cache or `None` for a local
projection. Query and norm views must refer to the bank rows actually used by
the producer. Caches retain their source and bank-row tensors, and validate
pointer, version, shape, stride, dtype, device, epsilon and cache-statistic
bindings. Wrong rows, reordered sources and ordinary in-place mutation before
consumption fail explicitly. These local bindings are combined with explicit
source/consumer IDs, model-owned step epochs and host snapshot authorization
in the integrated PP path. That path certifies transported copies before
rebinding them to canonical local parameters.

Version checks run during eager execution and graph recording. CUDA graph
replay does not execute those Python checks. The standalone capture test records
producer and consumer together on owning static surfaces, changes sources and
banks between replays, and compares fresh FLA results. The integrated pipeline
uses separate graph boundaries and explicit owning metadata inputs/outputs;
its host validates the model snapshot before each graph call. Both protocols
must preserve the producer-to-consumer dependency and current tensor contents.
Out-of-band tensor `.data` replacement, raw-pointer writes, inference-mode
versionless tensors and replay without its paired producer are not supported.

`return_stats=True` returns `(output, rstd, logit, lse)`. All three statistics
are FP32, explicitly nondifferentiable, and have the same shapes as FLA.
Supported arithmetic is scale one, no output RMSNorm and FLA checkpoint level
one. The primitive uses the installed FLA internal API and intentionally fails
if its required launch interface changes.

## Forward arithmetic and launch policy

The online-softmax kernel is derived from FLA's original forward, preserving
source order, its selected `BL`, hidden reduction shape, exponential helper,
weighted-value sum and saved statistics. On the first eager call for a
shape/dtype/device bucket, the original FLA forward runs normally to resolve
its launch configuration. The candidate copies the selected `BL`, warps and
stages; it does not change the original tuner cache or independently choose
a different source tile. Warm both forward and backward before capture.
No autotuning, pointer-table H2D copies or tensor-value host reads occur in a
warmed captured execution.

There are three paths:

* All cached sources load dot/RMS statistics and skip both reductions.
* Exactly one local source plus cached sources reduces the local source once
  as a hidden vector and places its statistics into the original source tile.
  This targets historical sources plus the current changing partial.
* General mixed tiles retain the original two-dimensional reductions when
  they contain a local row. Fully cached tiles skip reductions. This path can
  duplicate cached-row work within a mixed tile and must not be presented as
  eliminating every cached projection.

`forward_cache_launch_info(...)` reports the original launch choice, local
source count, and whether the single-local reduction path applies. Producer
projection reuses one loaded source for all query rows and one RMS value per
token. The additional FP32 store/load boundaries and vector reduction layouts
can change rounding. Preserving source order alone does not prove parity.

## Validation and performance boundaries

The dedicated tests compare output, FP32 saved statistics and every gradient
against both unmodified FLA and independent FP64 Torch RMS/softmax/value mixing.
They cover no cache, mixed cache, one local source, all cached sources, source
counts five and nine, query fanout eight and 33, zero query, one source, stale
bindings and three graph replays with fresh source/query/norm data.

The existing numerical gates remain unchanged: BF16 output/dV absolute
`0.016`, relative `0.01`, relative L2 below `0.003`; FP32 statistics absolute
`2e-5`, relative `2e-4`, relative L2 below `2e-5`; query absolute `1e-3`, norm
absolute `5e-5`, parameter-gradient relative `5e-4` and relative L2 below
`1e-4`. Nonzero tensors also require cosine and tensor similarity above
`0.999`; mathematically zero references require exact zeros. Failed cases are
failures, not grounds for changing thresholds.

End-to-end benefit requires timing the complete producer plus all consumer
forward/backward work under graphs, including owning copies and query
publication/transport. A consumer-only speedup omits producer overhead.
This primitive moves no backward computation. The implemented PP2/VP2 path
has executed independent training gates: disabled and fraction-zero controls
pass, while fractions 0.5 and 1 fail the second-update gradient gate. The isolated
primitive results do not override those failures. PP2/VP1 training qualification
remains unexecuted; see the [results document](attnres_forward_projection_results.md).

## FLA attribution and license

The adapted online-softmax code comes from `fla/ops/attnres/fused.py` in
`installed-fla-python.tar.gz`. The source archive SHA256 is
`64d9a27ad1a11256f70e61f50315ea4dc1473559ca2b6f2c50b0afd4bb6eb79b`;
the contained module SHA256, also pinned by the executed tests, is
`8ccfc512fded550b36314f459c69e0ed7af3f7bf35b21989b60da22b01f22371`.
These identify the archive and its module respectively. The original module's
copyright is retained in the Python file. The original
[FLA MIT license](https://github.com/fla-org/flash-linear-attention/blob/main/LICENSE)
is reproduced below; these notices apply to the adapted portions.

```text
MIT License

Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
