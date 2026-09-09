# AttnRes activation offloading follow-up (paused WIP)

Base: `2fa542825268deaa77d353cc821a9d9d849e4f5e`.
Branch: `jingqiny/attention-residuals-offload-followup`.

This branch preserves the unfinished expansion from attention-only offloading
to all fine-grained offload modules. It is not a completed support claim and
must not be merged into the K3 AttnRes + MTP work until independently validated.
The original branch retains the already committed three attention scopes.

## Design under investigation

Depth sources have longer lifetimes than per-sublayer activations. Norm offload
must release the aggregated norm input, not the live residual partial. Runtime
tests must distinguish checkpoint-owned inputs (intentionally not transferred)
from uncheckpointed inputs (which must actually transfer to host).

The hybrid norm-only raw-branch path is unfinished. Using raw deltas only when
offloading is enabled differs numerically from the baseline's BF16 add/subtract
reconstruction. A future integration must use consistent residual semantics
and preserve normal module hooks, including parameter all-gather hooks.

## Reproduction and evidence

Toolkit artifacts are under `runtime/attnres_review/`. The entry point is
`run_attnres_all_offload_tests.sh`, run through `job-launch exec` on Lyris with
the `gb200-gb300-2604-te216-pr2932-mcore-gdn-opt-c3c69be-dsl452-causalconv1d-pr2831`
image, FLA / fla-core 0.5.1, and nvidia-resiliency-ext 0.6.0.
Recorded allocation: 2971338, node lyris0035, 2026-09-09.

- `attnres_all_offload_2971338.log`: 43 configuration tests passed.
- `attnres_hybrid_offload_2971338_v3.log`: dense no-recompute eager/compile/FLA
  passed (3 cases); BF16 hybrid MoE failed for all three backends, with maximum
  output difference 0.03125. The residual reconstruction difference remains
  unresolved on this branch.
- `attnres_gpt_offload_2971338.log`: all seven non-fused scopes transferred
  activations and output/gradient checks passed for eager/compile/FLA, but the
  memory-savings assertion failed: 27.3-29.3 MiB deviation from prediction,
  approximately 2.0-2.2%, above the existing 20 MiB absolute threshold.
- Fused-group, expanded MTP, and recompute combinations have not completed the
  runtime acceptance matrix. Earlier dependency/assertion failures are retained
  in the v1/v2 logs; they are not passing runtime evidence.

Do not weaken tolerances or remove validation to declare this work complete.
Resume the wider offloading matrix only after K3 AttnRes + MTP is delivered.
