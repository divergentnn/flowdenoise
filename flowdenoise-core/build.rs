//! Builds the kernels (kernels/*.cu) into device images that the crate embeds:
//!   cuda feature: nvcc -fatbin (CUDA 12+; archs from FLOWDENOISE_CUDA_ARCHS, default 75;80;86;89;90;120 + PTX)
//!   hip feature:  hipcc --genco (ROCm; archs from FLOWDENOISE_HIP_ARCHS) -- compile-checked only
//! flowdenoise.cu is compiled with fast math (as the Python reference's CuPy RawModule), flowdenoise_ops.cu without
//! (as CuPy's elementwise kernels). The Metal kernels are compiled from source at run time.
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_tool(var: &str, sub: &str, fallbacks: &[&str], name: &str) -> PathBuf {
    if let Ok(p) = env::var(var) {
        return PathBuf::from(p);
    }
    for root in ["CUDA_PATH", "CUDA_HOME", "ROCM_PATH", "HIP_PATH"] {
        if let Ok(r) = env::var(root) {
            let p = Path::new(&r).join(sub);
            if p.exists() {
                return p;
            }
        }
    }
    for f in fallbacks {
        if Path::new(f).exists() {
            return PathBuf::from(f);
        }
    }
    PathBuf::from(name)
}

fn run(mut cmd: Command, what: &str) {
    eprintln!("flowdenoise-core build: {:?}", cmd);
    let st = cmd.status().unwrap_or_else(|e| panic!("{what}: cannot run {:?}: {e}", cmd.get_program()));
    if !st.success() {
        panic!("{what} failed ({st})");
    }
}

fn main() {
    let man = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kdir = man.join("kernels");
    for f in ["flowdenoise.cu", "flowdenoise_ops.cu", "portable.h"] {
        println!("cargo:rerun-if-changed={}", kdir.join(f).display());
    }
    for v in ["FLOWDENOISE_CUDA_ARCHS", "FLOWDENOISE_HIP_ARCHS", "NVCC", "HIPCC", "CUDA_PATH", "ROCM_PATH"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    // macOS has no CUDA / HIP: those features are inert there (the Metal backend is used)
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        return;
    }
    // (file, fast math)
    let units = [("flowdenoise", true), ("flowdenoise_ops", false)];
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        let nvcc = find_tool("NVCC", "bin/nvcc", &["/opt/cuda/bin/nvcc", "/usr/local/cuda/bin/nvcc"], "nvcc");
        let archs = env::var("FLOWDENOISE_CUDA_ARCHS").unwrap_or_else(|_| "75;80;86;89;90;120".into());
        let archs: Vec<&str> = archs.split(';').map(str::trim).filter(|s| !s.is_empty()).collect();
        for (unit, fast) in units {
            let mut c = Command::new(&nvcc);
            c.arg("-fatbin").arg("-O3").arg("-std=c++17").arg("-I").arg(&kdir);
            if fast {
                c.arg("--use_fast_math");
            }
            for a in &archs {
                c.arg(format!("-gencode=arch=compute_{a},code=sm_{a}"));
            }
            if let Some(last) = archs.last() {
                c.arg(format!("-gencode=arch=compute_{last},code=compute_{last}")); // PTX for newer GPUs
            }
            c.arg("-o").arg(out.join(format!("{unit}.fatbin"))).arg(kdir.join(format!("{unit}.cu")));
            run(c, "nvcc");
        }
    }
    if env::var("CARGO_FEATURE_HIP").is_ok() {
        let hipcc = find_tool("HIPCC", "bin/hipcc", &["/opt/rocm/bin/hipcc"], "hipcc");
        let archs = env::var("FLOWDENOISE_HIP_ARCHS")
            .unwrap_or_else(|_| "gfx906,gfx908,gfx90a,gfx1030,gfx1100,gfx1101,gfx1102,gfx1200,gfx1201".into());
        for (unit, fast) in units {
            let mut c = Command::new(&hipcc);
            c.arg("--genco").arg("-O3").arg("-std=c++17").arg("-x").arg("hip").arg("-I").arg(&kdir);
            if fast {
                c.arg("-ffast-math");
            }
            for a in archs.split(',') {
                c.arg(format!("--offload-arch={}", a.trim()));
            }
            c.arg("-o").arg(out.join(format!("{unit}.hsaco"))).arg(kdir.join(format!("{unit}.cu")));
            run(c, "hipcc");
        }
    }
}
