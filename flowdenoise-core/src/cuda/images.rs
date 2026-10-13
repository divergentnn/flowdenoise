//! Device images built by build.rs and the kernels each one provides.
use super::driver::Kind;

/// kernels/flowdenoise.cu (fast math)
pub const DENOISE_KERNELS: &[&str] = &[
    "sep3_0", "sep3_1", "sep5_0", "sep5_1", "sep7_0", "sep7_1", "sep8_0", "sep8_1", "sep19_0", "sep19_1",
    "sep37_0", "sep37_1", "sep", "down", "raster", "dense", "warp_y", "weight_only", "reduce_frames", "chroma",
];
/// kernels/flowdenoise_ops.cu (no fast math)
pub const OPS_KERNELS: &[&str] = &[
    "u16_to_f32", "ew_absdiff", "ew_energy", "ew_sub", "ew_sqr", "ew_abs", "ew_div_bc", "ew_q8", "ew_rint", "ew_guide",
    "ew_alpha", "ew_blend", "ew_finish", "ew_pack", "gather_stride", "sel_init", "sel_hist", "sel_scan", "sel_value",
    "noise_var", "quantile_finish",
];

#[cfg(feature = "cuda")]
static CUDA_DENOISE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/flowdenoise.fatbin"));
#[cfg(feature = "cuda")]
static CUDA_OPS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/flowdenoise_ops.fatbin"));
#[cfg(feature = "hip")]
static HIP_DENOISE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/flowdenoise.hsaco"));
#[cfg(feature = "hip")]
static HIP_OPS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/flowdenoise_ops.hsaco"));

pub fn available(b: Kind) -> bool {
    match b {
        Kind::Cuda => cfg!(feature = "cuda"),
        Kind::Hip => cfg!(feature = "hip"),
    }
}

/// (image, kernels) per module
pub fn modules(b: Kind) -> Vec<(&'static [u8], &'static [&'static str])> {
    match b {
        #[cfg(feature = "cuda")]
        Kind::Cuda => vec![(CUDA_DENOISE, DENOISE_KERNELS), (CUDA_OPS, OPS_KERNELS)],
        #[cfg(feature = "hip")]
        Kind::Hip => vec![(HIP_DENOISE, DENOISE_KERNELS), (HIP_OPS, OPS_KERNELS)],
        #[allow(unreachable_patterns)]
        _ => vec![],
    }
}
