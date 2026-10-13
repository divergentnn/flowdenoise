#!/usr/bin/env bash
# Metal (macOS): the readability pass of the kernels must not change a single output bit. Builds the revision before
# the pass and the one after it, runs both presets on every clip with the Metal backend and compares the output hashes.
#
#   tools/check_metal.sh [CLIP.y4m ...]
#
# Clips: 10-bit 4:2:0 y4m (ffmpeg ... -pix_fmt yuv420p10le -strict -1 -f yuv4mpegpipe); use a 1080p and a 4K one.
# Without clips, synthetic ones are generated. BEFORE / AFTER override the revisions (default: the parent of the
# readability commit, and HEAD).
set -euo pipefail
repo=$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)
[ "$(uname -s)" = Darwin ] || { echo "check_metal.sh: needs macOS (Metal)"; exit 2; }
pass=$(git -C "$repo" log --format=%H --grep='^kernels: readability pass' -n 1)
[ -n "$pass" ] || { echo "check_metal.sh: readability commit not found"; exit 2; }
BACKEND=metal exec "$repo/tools/compare_revisions.sh" "${BEFORE:-$pass~1}" "${AFTER:-HEAD}" "$@"
