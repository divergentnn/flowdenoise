#!/usr/bin/env bash
# Build two revisions of flowdenoise and compare their outputs byte for byte (both presets, every clip).
#
#   tools/compare_revisions.sh BEFORE_REV AFTER_REV [CLIP.y4m ...]
#
# Without clips, synthetic 10-bit 4:2:0 clips are generated with ffmpeg (1920x1080 x 24 frames, 3840x2160 x 12
# frames: moving test pattern plus temporal noise); real footage is a better test, pass it when you have it.
# Environment: BACKEND (auto, cuda, hip, metal), CARGO_JOBS (default 4), KEEP=1 keeps the work directory.
# Exit status 0 when every output is identical.
set -euo pipefail
before=$1
after=$2
shift 2
repo=$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)
work=$(mktemp -d "${TMPDIR:-/tmp}/flowdenoise-compare.XXXXXX")
cleanup() {
    for d in before after; do git -C "$repo" worktree remove --force "$work/$d" 2>/dev/null || true; done
    if [ "${KEEP:-0}" = 1 ]; then echo "work directory kept: $work"; else rm -rf "$work"; fi
}
trap cleanup EXIT

clips=("$@")
if [ ${#clips[@]} -eq 0 ]; then
    echo "no clips given: generating synthetic ones in $work"
    ffmpeg -v error -nostdin -f lavfi -i "testsrc2=size=1920x1080:rate=24,noise=alls=12:allf=t" -frames:v 24 \
        -pix_fmt yuv420p10le -strict -1 -f yuv4mpegpipe "$work/synthetic_1080.y4m"
    ffmpeg -v error -nostdin -f lavfi -i "testsrc2=size=3840x2160:rate=24,noise=alls=12:allf=t" -frames:v 12 \
        -pix_fmt yuv420p10le -strict -1 -f yuv4mpegpipe "$work/synthetic_2160.y4m"
    clips=("$work/synthetic_1080.y4m" "$work/synthetic_2160.y4m")
fi

for side in before after; do
    rev=$before
    [ "$side" = after ] && rev=$after
    git -C "$repo" worktree add --detach "$work/$side" "$rev" >/dev/null
    echo "building $side ($(git -C "$work/$side" rev-parse --short HEAD))"
    (cd "$work/$side" && nice -n 15 cargo build --release -j "${CARGO_JOBS:-4}" --features binary --target-dir "$work/target-$side" -q)
    "$repo/tools/hashes.sh" "$work/target-$side/release/flowdenoise" "${clips[@]}" > "$work/$side.txt"
done
echo "before ($before)                          after ($after)"
paste -d' ' "$work/before.txt" "$work/after.txt" | awk '{ s = ($3 == $6) ? "identical" : "DIFFERENT"; print $1, $2, $3, $6, s }'
if cmp -s "$work/before.txt" "$work/after.txt"; then
    echo "all outputs identical"
else
    echo "OUTPUTS DIFFER"
    exit 1
fi
