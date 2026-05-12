#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

OUT=${1:-mhc-full-fwd-bwd-autotuned}

nsys profile --sample=none --cpuctxsw=none -t cuda-sw,nvtx \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=node -f true -x true \
  -o "${OUT}" \
  env MHC_RUN_MHC_PROFILE=1 MHC_NSYS_CAPTURE=1 \
      MHC_PROFILE_S="${MHC_PROFILE_S:-4096}" \
      MHC_PROFILE_B="${MHC_PROFILE_B:-1}" \
      MHC_PROFILE_N="${MHC_PROFILE_N:-4}" \
      MHC_PROFILE_C="${MHC_PROFILE_C:-7168}" \
      MHC_PROFILE_SINKHORN_ITERS="${MHC_PROFILE_SINKHORN_ITERS:-5}" \
      MHC_PROFILE_WARMUP="${MHC_PROFILE_WARMUP:-1}" \
      MHC_PROFILE_REPS="${MHC_PROFILE_REPS:-1}" \
  python3 -m pytest -q -s \
    tests/unit_tests/fusions/test_fused_mhc_kernels_profile.py::test_mhc_full_fwd_bwd_autotuned_profile
