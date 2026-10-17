//! Device images built by build.rs and the kernels each one provides.
use super::driver::Kind;

/// kernels/flowdenoise.cu (fast math)
pub const DENOISE_KERNELS: &[&str] = &[
    "separable_filter_3_x", "separable_filter_3_y", "separable_filter_5_x", "separable_filter_5_y",
    "separable_filter_7_x", "separable_filter_7_y", "separable_filter_8_x", "separable_filter_8_y",
    "separable_filter_19_x", "separable_filter_19_y", "separable_filter_37_x", "separable_filter_37_y",
    "separable_filter_any", "downsample_2x2", "motion_search_sweep", "densify_flow", "warp_luma", "temporal_weight",
    "fuse_frames", "warp_accumulate_chroma",
];
/// kernels/flowdenoise_ops.cu (no fast math)
pub const OPS_KERNELS: &[&str] = &[
    "u16_to_float", "abs_diff", "squared_diff", "subtract", "square", "absolute", "divide_broadcast", "quantize_8bit",
    "round_to_integer", "guided_coefficients", "fallback_alpha", "guided_blend", "detail_gate_luma", "pack_10bit",
    "gather_strided", "radix_select_init", "radix_select_histogram", "radix_select_scan", "radix_select_value",
    "noise_variance", "quantile_lerp", "stability_share", "scale_by_complement", "range_sample_gather",
    "range_sample_fallback", "range_thresholds", "range_check", "radix_select_init_median",
    "radix_select_histogram_counted", "multiply", "persist_correlation", "persist_gain", "persist_accumulate",
    "persist_apply",
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
