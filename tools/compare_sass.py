#!/usr/bin/env python3
"""Compare the machine code (SASS) of the CUDA kernels between two git revisions, kernel by kernel, and the
embedded PTX (the JIT input for newer GPUs) up to the names a rename changes.

    tools/compare_sass.py BEFORE_REV [AFTER_REV] [--arch 75,80,86,89,90,120] [--ptx-arch 120]

AFTER_REV defaults to the working tree. Both revisions' kernels/*.cu are compiled to cubins with the build's flags
(nvcc -O3 -std=c++17, --use_fast_math for flowdenoise.cu) and disassembled with cuobjdump; kernels renamed between
the revisions are matched through RENAMED. Identical SASS for every kernel and architecture means the edit cannot
change any output on those GPUs, for any input (a stronger check than output hashes on test clips).

Needs nvcc and cuobjdump on PATH (or CUDA_PATH).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# old entry-point name -> new name (the readability pass)
RENAMED = {
    **{f"sep{t}_0": f"separable_filter_{t}_x" for t in (3, 5, 7, 8, 19, 37)},
    **{f"sep{t}_1": f"separable_filter_{t}_y" for t in (3, 5, 7, 8, 19, 37)},
    "sep": "separable_filter_any",
    "down": "downsample_2x2",
    "raster": "motion_search_sweep",
    "dense": "densify_flow",
    "warp_y": "warp_luma",
    "weight_only": "temporal_weight",
    "reduce_frames": "fuse_frames",
    "chroma": "warp_accumulate_chroma",
    "u16_to_f32": "u16_to_float",
    "ew_absdiff": "abs_diff",
    "ew_energy": "squared_diff",
    "ew_sub": "subtract",
    "ew_sqr": "square",
    "ew_abs": "absolute",
    "ew_div_bc": "divide_broadcast",
    "ew_q8": "quantize_8bit",
    "ew_rint": "round_to_integer",
    "ew_guide": "guided_coefficients",
    "ew_alpha": "fallback_alpha",
    "ew_blend": "guided_blend",
    "ew_finish": "detail_gate_luma",
    "ew_pack": "pack_10bit",
    "gather_stride": "gather_strided",
    "sel_init": "radix_select_init",
    "sel_hist": "radix_select_histogram",
    "sel_scan": "radix_select_scan",
    "sel_value": "radix_select_value",
    "noise_var": "noise_variance",
    "quantile_finish": "quantile_lerp",
}
# kernels the reference compiled but never launched; dropped by the readability pass
REMOVED = {"motion", "accum"}
UNITS = {"flowdenoise.cu": True, "flowdenoise_ops.cu": False}   # file: fast math
KDIR = "flowdenoise-core/kernels"


def tool(name):
    for root in (os.environ.get("CUDA_PATH"), "/opt/cuda", "/usr/local/cuda"):
        if root and (Path(root) / "bin" / name).exists():
            return str(Path(root) / "bin" / name)
    return shutil.which(name) or name


def checkout(repo, rev, dst):
    """kernels of `rev` (None: the working tree) into dst"""
    dst.mkdir(parents=True)
    for f in [*UNITS, "portable.h"]:
        if rev is None:
            shutil.copy(repo / KDIR / f, dst / f)
        else:
            data = subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{KDIR}/{f}"], check=True,
                                  capture_output=True).stdout
            (dst / f).write_bytes(data)


def sass(src_dir, unit, fast, arch, out):
    cubin = out / f"{unit}.{arch}.cubin"
    cmd = [tool("nvcc"), "-cubin", "-O3", "-std=c++17", "-I", str(src_dir), f"-arch=sm_{arch}", "-o", str(cubin),
           str(src_dir / unit), "-diag-suppress", "177"]
    if fast:
        cmd.insert(1, "--use_fast_math")
    subprocess.run(cmd, check=True)
    text = subprocess.run([tool("cuobjdump"), "-sass", str(cubin)], check=True, capture_output=True,
                          text=True).stdout
    funcs, name = {}, None
    for line in text.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            name = m.group(1)
            funcs[name] = []
        elif name and line.strip():
            funcs[name].append(line.strip())
    return funcs


def ptx(src_dir, unit, fast, arch, out):
    """{entry: PTX body} with the names that a rename changes normalised away (parameter and shared-variable names,
    basic-block label numbers)"""
    path = out / f"{unit}.{arch}.ptx"
    cmd = [tool("nvcc"), "-ptx", "-O3", "-std=c++17", "-I", str(src_dir), f"-arch=compute_{arch}", "-o", str(path),
           str(src_dir / unit), "-diag-suppress", "177"]
    if fast:
        cmd.insert(1, "--use_fast_math")
    subprocess.run(cmd, check=True)
    funcs = {}
    for m in re.finditer(r"\.visible \.entry (\w+)\((.*?)\n\}\n", path.read_text(), re.S):
        body = re.sub(r"\b" + m.group(1) + r"_param_", "P_param_", m.group(2))
        body = re.sub(r"_ZZ\w+", "SHARED", body)
        body = re.sub(r"\$L__BB\d+_", "$L__BB_", body)
        funcs[m.group(1)] = re.sub(r"//.*", "", body).split("\n")
    return funcs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after", nargs="?")
    ap.add_argument("--arch", default="75,80,86,89,90,120")
    ap.add_argument("--ptx-arch", default="120", help="also compare the embedded PTX for this compute capability")
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        checkout(repo, a.before, tmp / "before")
        checkout(repo, a.after, tmp / "after")
        jobs = [(f"sm_{arch}", sass, arch) for arch in a.arch.split(",") if arch]
        jobs += [(f"ptx_{arch}", ptx, arch) for arch in a.ptx_arch.split(",") if arch]
        for label, compile_, arch in jobs:
            for unit, fast in UNITS.items():
                before = compile_(tmp / "before", unit, fast, arch, tmp / "before")
                after = compile_(tmp / "after", unit, fast, arch, tmp / "after")
                seen = set()
                for old, code in sorted(before.items()):
                    if old in REMOVED:
                        print(f"{label:7} {unit:20} {old:24} removed")
                        continue
                    new = RENAMED.get(old, old)
                    seen.add(new)
                    if new not in after:
                        print(f"{label:7} {unit:20} {old:24} MISSING after ({new})")
                        bad += 1
                    elif after[new] != code:
                        print(f"{label:7} {unit:20} {old:24} -> {new:28} SASS DIFFERS")
                        bad += 1
                    else:
                        print(f"{label:7} {unit:20} {old:24} -> {new:28} identical ({len(code)} lines)")
                for new in sorted(set(after) - seen):
                    print(f"{label:7} {unit:20} {'':24} -> {new:28} NEW (no counterpart)")
                    bad += 1
    print("all kernels identical" if not bad else f"{bad} kernel(s) differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
