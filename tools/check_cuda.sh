#!/usr/bin/env bash
# CUDA: as check_metal.sh, on the CUDA backend (needs nvcc on PATH, e.g. PATH=/opt/cuda/bin:$PATH).
#
#   tools/check_cuda.sh [CLIP.y4m ...]
set -euo pipefail
repo=$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)
pass=$(git -C "$repo" log --format=%H --grep='^kernels: readability pass' -n 1)
[ -n "$pass" ] || { echo "check_cuda.sh: readability commit not found"; exit 2; }
BACKEND=cuda exec "$repo/tools/compare_revisions.sh" "${BEFORE:-$pass~1}" "${AFTER:-HEAD}" "$@"
