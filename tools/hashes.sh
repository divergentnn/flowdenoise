#!/usr/bin/env bash
# Output hashes of a flowdenoise binary: one line "<clip> <preset> <md5>" per clip and preset.
#
#   tools/hashes.sh FLOWDENOISE_BINARY CLIP.y4m [CLIP.y4m ...]
#
# Environment: BACKEND (auto, cuda, hip, metal; default auto), PRESETS (default "production measure"),
# NICE (default 15).
set -euo pipefail
bin=$1
shift
backend=${BACKEND:-auto}
nice_level=${NICE:-15}
md5() { if command -v md5sum >/dev/null; then md5sum | cut -d' ' -f1; else md5 -q; fi; }
for clip in "$@"; do
    for preset in ${PRESETS:-production measure}; do
        h=$(nice -n "$nice_level" "$bin" -q --backend "$backend" --preset "$preset" "$clip" - | md5)
        echo "$(basename "$clip") $preset $h"
    done
done
