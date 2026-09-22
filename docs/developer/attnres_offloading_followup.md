# AttnRes activation offloading follow-up (isolated WIP)

Base: `64fbfe3c22edca24c94146bad12a12b473738dc5` (PR #6840).
Branch: `jingqiny/attention-residuals-offload-followup`.
Original split base: `2fa542825268deaa77d353cc821a9d9d849e4f5e`.

This draft follow-up preserves the unfinished expansion from attention-only
offloading to all fine-grained offload modules. It is stacked on the K3 AttnRes
+ MTP branch and is not a completed support claim. The broader runtime matrix
must pass before merging. The base retains the already committed three
attention scopes.

## Design under investigation

Depth sources have longer lifetimes than per-sublayer activations. Norm offload
must release the aggregated norm input, not the live residual partial. Runtime
tests must distinguish checkpoint-owned inputs (intentionally not transferred)
from uncheckpointed inputs (which must actually transfer to host).

The September 22 rebase preserves the base's unified hybrid raw-delta path for
both offload and non-offload runs, including normal module and parameter
all-gather hooks. The obsolete offload-only raw-branch helper is removed.
GPT AttnRes separately passes the helper-owned norm input to the offload
manager while retaining the running partial for residual accumulation. The
rebased combination still requires GPU output/gradient and transfer validation.

## Reproduction and evidence

Toolkit artifacts are under `runtime/attnres_review/`. The entry point is
`run_attnres_all_offload_tests.sh`, run through `job-launch exec` on Lyris with
the `gb200-gb300-2604-te216-pr2932-mcore-gdn-opt-c3c69be-dsl452-causalconv1d-pr2831`
image, FLA / fla-core 0.5.1, and nvidia-resiliency-ext 0.6.0.
Recorded allocation: 2971338, node lyris0035, 2026-09-09.

- `attnres_all_offload_2971338.log`: 43 configuration tests passed.
- `attnres_hybrid_offload_2971338_v3.log`: dense no-recompute eager/compile/FLA
  passed (3 cases); BF16 hybrid MoE failed for all three backends, with maximum
  output difference 0.03125. Those results predate the unified hybrid raw-delta
  integration; they must be rerun before claiming that the mismatch is resolved.
- `attnres_gpt_offload_2971338.log`: all seven non-fused scopes transferred
  activations and output/gradient checks passed for eager/compile/FLA, but the
  memory-savings assertion failed: 27.3-29.3 MiB deviation from prediction,
  approximately 2.0-2.2%, above the existing 20 MiB absolute threshold.
- Fused-group, expanded MTP, and recompute combinations have not completed the
  runtime acceptance matrix. Earlier dependency/assertion failures are retained
  in the v1/v2 logs; they are not passing runtime evidence.

Do not weaken tolerances or remove validation to declare this work complete.
## Requested fine-grained validation matrix

The user supplied two explicit acceptance targets after the branch split. They
are recorded in
`tests/unit_tests/pipeline_parallel/attnres_offload_matrix.yaml`, independently
of the MTP implementation branch. Preparation is active; GPU execution awaits
Lyris connectivity. The wider support implementation remains unfinished.

The non-fused target recomputes `gdn`, `mla_up_proj`, `layernorm`,
`moe_latent_proj`, `shared_experts`, and `moe_act`; it offloads `mlp_norm`,
`expert_fc1`, `moe_act`, `attn_proj`, `qkv_linear`, and `core_attn` at fraction
1.0. Keep `moe_act` in both lists: its existing checkpoint/offload path must
reload the FC1 output before activation recomputation, and this ownership and
ordering is a central test target.

The fused target omits only activation recomputation and replaces
`expert_fc1`/`moe_act` offloading with `fused_group_mlp`. It requires TE op fuser,
expert TP 1, real fused GroupedLinear support, and `ScaledSiTUGLU` for K3's
activation. The current image's ordinary `SiTUGLU` operation is known to be
absent; the scaled fused operation must be checked separately before running.
`use_transformer_engine_op_fuser=True` alone is not proof: `_is_fused_impl_supported`
can return False and fall back. Tests must assert actual fused-stack execution
and retain evidence of the grouped MLP kernels.

`linear_qkv` in the request is normalized explicitly to the existing selector
`qkv_linear`. No requested module is dropped. Do not conflate MLA attention
offload scopes with KDA: the latter uses the `gdn` recompute path. Each trace
must report the actual owning module and whether a selected offload scope
transferred an activation, retained a checkpoint-owned input, or did not apply.
An inapplicable/retained scope is not evidence of a host transfer.

### Integrated prerequisite and remaining validation

The September 22 rebase includes the main-branch K3 fix `8635c04b6` through
base `64fbfe3c2`. Both offload states use the same raw branch semantics and
retain normal module hooks. The additional `mlp_norm_input` argument preserves
the distinct ownership of the aggregated norm input and the running partial.
The historical BF16 subtraction baseline is no longer used for this comparison.
Integration is complete at the source level; GPU validation remains pending.

### Execution and acceptance order

1. Finish the remaining no-offload K3 MTP validation on the main branch
   (PP2, selective recompute, compile training, checkpoint save/resume). Keep
   this separate from offload-specific failures.
2. Validate the exact merged TransformerConfig, dependencies, activation, and
   fused-stack selection. Missing backend support is a prerequisite failure,
   not a successful fallback run.
3. For each target, compare four controls with identical weights, inputs,
   seeds, routing, dtype, and AttnRes backend: neither feature, recompute only,
   offload only, and the requested combination. In the fused target also compare
   against the non-fused semantic reference.
4. Compare outputs, input/source gradients, and every mapped trainable parameter
   gradient. Retain the existing numerical thresholds; report cosine/tensor
   similarity, relative-L2 and max-absolute error separately. Never loosen a
   threshold after observing failure to manufacture a pass.
5. Run at least 10 complete optimizer steps for MTP1 and MTP2, then TP2+SP and
   PP2 with a residual block crossing the pipeline boundary. Check finite LM
   and every MTP loss, finite gradients, actual MTP weight updates, and zero
   skipped/NaN steps. Use real collectives, not fake process groups.
6. Record per-scope offload/reload bytes, ownership, and ordering. Add isolated
   no-recompute positive controls for checkpoint-owned or inapplicable scopes;
   do not count a no-op offload flag as demonstrated transfer support. Measure
   peak memory and explain retained residual sources and staging-buffer cost.
7. Save optimizer/RNG/model state and resume at least three steps. Collect an
   nsys report with NVTX scope markers, host/device copies, recomputation, and
   actual grouped-MLP kernels; copy reports and full logs locally. GPU kernel
   time excludes CPU/JIT warmup and is reported separately from step time.

Failures retain their exact manifest, source hash, image/dependency versions,
seed, full traceback, numerical mismatch, and trace where available. Isolate
the failing module pair, fix the cause on this branch, re-run the minimal repro,
and then re-run the unchanged complete target. Neither target has passed yet.
