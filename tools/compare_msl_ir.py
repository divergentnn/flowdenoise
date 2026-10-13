#!/usr/bin/env python3
"""Compare the Metal kernels of two revisions without a Mac: each .metal file is compiled as C++ (clang, with
tools/msl_shim/metal_shim.h standing in for <metal_stdlib>) to LLVM IR, and every kernel's IR is compared with its
counterpart (renamed kernels matched through compare_sass.py's RENAMED plus METAL_RENAMED).

    tools/compare_msl_ir.py BEFORE_REV [AFTER_REV] [--files a.metal,b.metal] [--kdir DIR] [--repo DIR]
                            [--renames map.json]

AFTER_REV defaults to the working tree. The Metal built-ins are opaque calls and no fast math is applied, so the IR
records each kernel's own arithmetic in source order (FMA contraction points included): identical IR means the edit
changed names, comments and layout only. This is a pre-flight check; the definitive one is tools/check_metal.sh on a
Mac (the Metal compiler and its fast math are not modelled here). Needs clang++ and llvm-cxxfilt.
"""
import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("compare_sass", HERE / "compare_sass.py")
_cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cs)
METAL_RENAMED = {"copy_u32": "copy_words", "copy_u8": "copy_bytes", "fill_u8": "fill_bytes"}
RENAMED = {**_cs.RENAMED, **METAL_RENAMED}
REMOVED = set()


def source(repo, rev, path):
    if rev is None:
        return (repo / path).read_text()
    return subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{path}"], check=True, capture_output=True,
                          text=True).stdout


def host_names(pre):
    """{(template, args): kernel name} from the explicit instantiations of the preprocessed source"""
    out = {}
    pat = r'template\s*\[\[\s*host_name\(((?:\s*"[^"]*")+)\s*\)\s*\]\]\s*(?:kernel\s+)?void\s+(\w+)\s*<([^>]*)>'
    for m in re.finditer(pat, pre):
        name = "".join(re.findall(r'"([^"]*)"', m.group(1)))
        args = ",".join(a.strip().replace("AXIS_X", "0").replace("AXIS_Y", "1") for a in m.group(3).split(","))
        out[(m.group(2), args)] = name
    return out


def kernels_ir(text, work, tag):
    """{kernel name: normalised IR lines} of one MSL source"""
    src = work / f"{tag}.metal.cpp"
    body = text.replace("#include <metal_stdlib>", '#include "metal_shim.h"').replace("using namespace metal;", "")
    body = body.replace("static inline", "static_inline_always")
    pre = subprocess.run(["clang++", "-E", "-P", "-x", "c++", "-std=c++17", "-I", str(HERE / "msl_shim"), "-"],
                         input=body, capture_output=True, text=True, check=True).stdout
    names = host_names(pre)
    pre = re.sub(r"\[\[[^\[\]]*\]\]", "", pre)            # Metal attributes ([[buffer(0)]], [[host_name(...)]], ...)
    src.write_text(pre)
    ll = work / f"{tag}.ll"
    subprocess.run(["clang++", "-x", "c++", "-std=c++17", "-O2", "-S", "-emit-llvm", "-Xclang", "-discard-value-names",
                    "-Wno-everything", "-o", str(ll), str(src)], check=True)
    funcs, cur, lines = {}, None, []
    for line in ll.read_text().splitlines():
        m = re.match(r"define .*@(\S+?)\(", line)
        if m:
            cur, lines = m.group(1), []
            continue
        if cur is not None:
            if line == "}":
                funcs[cur] = lines
                cur = None
            else:
                lines.append(line)
    demangled = subprocess.run(["llvm-cxxfilt"], input="\n".join(funcs), capture_output=True, text=True,
                               check=True).stdout.splitlines()
    out = {}
    for mangled, dm in zip(funcs, demangled):
        m = re.match(r"(?:void )?(\w+)(?:<([^>]*)>)?\(", dm)
        if not m:
            continue
        base, targs = m.group(1), m.group(2)
        name = names.get((base, ",".join(a.strip() for a in targs.split(",")))) if targs else base
        if name is None:
            continue
        norm = []
        for line in funcs[mangled]:
            line = re.sub(r"%struct\.[\w.]+", "%struct.T", line)
            line = re.sub(r",? ![\w.]+ !\d+", "", line)
            line = re.sub(r" #\d+", "", line)
            line = line.replace("noundef ", "")             # return-value annotations of inlined helpers
            norm.append(line)
        out[name] = norm
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after", nargs="?")
    ap.add_argument("--kdir", default="flowdenoise-core/kernels/metal")
    ap.add_argument("--files", default="flowdenoise.metal,flowdenoise_ops.metal")
    ap.add_argument("--repo", default=str(HERE.parent))
    ap.add_argument("--renames", help='JSON {"renamed": {old: new}, "removed": [old, ...]} instead of the built-in map')
    a = ap.parse_args()
    renamed, removed = RENAMED, REMOVED
    if a.renames:
        m = json.loads(Path(a.renames).read_text())
        renamed, removed = m.get("renamed", {}), set(m.get("removed", []))
    if not shutil.which("clang++") or not shutil.which("llvm-cxxfilt"):
        sys.exit("needs clang++ and llvm-cxxfilt")
    repo = Path(a.repo)
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for f in a.files.split(","):
            path = f"{a.kdir}/{f}"
            before = kernels_ir(source(repo, a.before, path), work, "before")
            after = kernels_ir(source(repo, a.after, path), work, "after")
            seen = set()
            for old, code in sorted(before.items()):
                if old in removed:
                    continue
                new = renamed.get(old, old)
                seen.add(new)
                if new not in after:
                    print(f"{f:24} {old:24} MISSING after ({new})")
                    bad += 1
                elif after[new] != code:
                    print(f"{f:24} {old:24} -> {new:28} IR DIFFERS")
                    bad += 1
                else:
                    print(f"{f:24} {old:24} -> {new:28} identical ({len(code)} IR lines)")
            for new in sorted(set(after) - seen):
                print(f"{f:24} {'':24} -> {new:28} NEW (no counterpart)")
                bad += 1
    print("all kernels identical" if not bad else f"{bad} kernel(s) differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
