// flowdenoise_ops.cu: elementwise and order-statistic kernels of the denoiser (format conversions, the guided
// fallback, the detail gate, output packing, a radix select for medians and quantiles).
//
// Compiled WITHOUT fast math: these are the Python reference's CuPy elementwise operations, which CuPy compiles
// without it (IEEE division and square root). Do not move them into flowdenoise.cu. As there, every operation and
// its order is the reference's (README.md, "Numerical rules").
//
// All elementwise kernels: one thread per element, 1-D grid of 256-thread blocks, `count` elements (64-bit index).
// Planes are float32, laid out as in flowdenoise.cu.
#include "portable.h"

// the element index of this thread; threads past the end return
#define ELEMENT_INDEX                                           \
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= count) {                                           \
        return;                                                 \
    }

// Value clamped into [0, 1].
__device__ __forceinline__ float saturate(float value) {
    return fminf(1.f, fmaxf(0.f, value));
}

// ------------------------------------------------------------------------------------------------ conversions

// 10-bit codes (uint16) to float.
extern "C" __global__ void u16_to_float(const uint16_t* src, float* dst, int64_t count) {
    ELEMENT_INDEX
    dst[i] = (float)src[i];
}

// Motion pyramid base: 10-bit luma to rounded 8-bit codes, clip(rint(x / 4), 0, 255) (rint: ties to even).
extern "C" __global__ void quantize_8bit(const float* src, float* dst, int64_t count) {
    ELEMENT_INDEX
    float v = rintf(src[i] / 4.f);
    dst[i] = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
}

// Rounding of the motion pyramid levels to whole codes (rint: ties to even).
extern "C" __global__ void round_to_integer(const float* src, float* dst, int64_t count) {
    ELEMENT_INDEX
    dst[i] = rintf(src[i]);
}

// ------------------------------------------------------------------------------------------------ elementwise

// |a - b| (the SAD input).
extern "C" __global__ void abs_diff(const float* a, const float* b, float* out, int64_t count) {
    ELEMENT_INDEX
    out[i] = fabsf(a[i] - b[i]);
}

// (a - b)^2 (the detail energy).
extern "C" __global__ void squared_diff(const float* a, const float* b, float* out, int64_t count) {
    ELEMENT_INDEX
    float d = a[i] - b[i];
    out[i] = d * d;
}

// a - b.
extern "C" __global__ void subtract(const float* a, const float* b, float* out, int64_t count) {
    ELEMENT_INDEX
    out[i] = a[i] - b[i];
}

// a^2.
extern "C" __global__ void square(const float* a, float* out, int64_t count) {
    ELEMENT_INDEX
    out[i] = a[i] * a[i];
}

// |a|.
extern "C" __global__ void absolute(const float* a, float* out, int64_t count) {
    ELEMENT_INDEX
    out[i] = fabsf(a[i]);
}

// out = a / b[i % period]: normalises the temporal sums (luma: weighted_sum / weight_sum; chroma: the U and V sums
// over one shared weight plane, period = one chroma plane).
extern "C" __global__ void divide_broadcast(const float* a, const float* b, float* out, int64_t count, int64_t period) {
    ELEMENT_INDEX
    out[i] = a[i] / b[i % period];
}

// ------------------------------------------------------------------------------------------------ guided fallback
// Where few frames were fused, the result keeps more noise; there it is blended toward a self-guided filter of
// itself (He, Sun, Tang, "Guided Image Filtering", ECCV 2010), with the box means computed by the separable filters.

// fallback_alpha: blend factor from the effective number of fused frames n = weight_sum^2 / weight_sq_sum
// (1 = only the current frame): alpha = clamp((fallback - n) / (fallback - 1), 0, 1), 1 at n = 1, 0 from n = fallback.
extern "C" __global__ void fallback_alpha(const float* weight_sum, const float* weight_sq_sum, float fallback,
                                          float* alpha, int64_t count) {
    ELEMENT_INDEX
    alpha[i] = saturate((fallback - weight_sum[i] * weight_sum[i] / weight_sq_sum[i]) / (fallback - 1));
}

// guided_coefficients: per-pixel linear coefficients of the self-guided filter from the box means of x and x^2:
// variance = max(mean_sq - mean^2, 0); a = variance / (variance + eps); b = mean - a * mean. eps is one value per
// plane (eps[i / plane_size]): fallback_eps times the plane's noise variance.
extern "C" __global__ void guided_coefficients(const float* mean, const float* mean_sq, const float* eps, float* a,
                                               float* b, int64_t count, int64_t plane_size) {
    ELEMENT_INDEX
    float e = eps[i / plane_size];
    float m = mean[i];
    float variance = fmaxf(mean_sq[i] - m * m, 0.f);
    float coef_a = variance / (variance + e);
    a[i] = coef_a;
    b[i] = m - coef_a * m;
}

// guided_blend: out = x + alpha * (mean_a * x + mean_b - x), the guided filter output (mean_a, mean_b: the box means
// of the coefficients) weighted by alpha (alpha[i % alpha_period]: one luma-sized plane for both chroma planes).
extern "C" __global__ void guided_blend(const float* x, const float* alpha, const float* mean_a, const float* mean_b,
                                        float* out, int64_t count, int64_t alpha_period) {
    ELEMENT_INDEX
    float v = x[i];
    out[i] = v + alpha[i % alpha_period] * (mean_a[i] * v + mean_b[i] - v);
}

// ------------------------------------------------------------------------------------------------ detail gate, output

// detail_gate_luma: puts the current frame's own fine detail back where it is coherent texture, and writes the
// output luma.
//
// Inputs (luma planes): fused (after the fallback) and fused_low, its Gaussian sigma 1; current_high, the current
// frame minus its Gaussian sigma 1; high_sum_smooth = Gaussian sigma 1 of |high_sum| and high_abs_sum_smooth =
// Gaussian sigma 1 of high_abs_sum (fuse_frames); weight_sum, weight_sq_sum; energy, the Gaussian sigma 3 of
// (fused - fused_low)^2; energy_floor, one float in device memory: the 25th percentile of `energy` (subsampled);
// gate_threshold (detail_gate), boost (detail_boost).
//   noise_coherence = sqrt(weight_sq_sum) / weight_sum: the coherence |sum w h| / sum w |h| that independent noise
//                     would reach;
//   coherence = clamp((high_sum / high_abs_sum - noise_coherence) / max(1 - noise_coherence, 0.05), 0, 1):
//               0 for noise, 1 when the frames agree on the detail;
//   texture   = clamp((energy / energy_floor - 3) / 3, 0, 1): 0 in flat areas, 1 from 6 x the flat-area energy;
//   gate      = clamp((coherence - gate_threshold) / (1 - gate_threshold), 0, 1) * texture;
//   out       = fused_low + gate * (1 + (boost - 1) * texture) * current_high + (1 - gate) * (fused - fused_low),
// rounded to the nearest integer (ties to even) and clamped to 0 ... 1023.
// Output: out, int16 luma.
extern "C" __global__ void detail_gate_luma(const float* fused, const float* fused_low, const float* current_high,
                                            const float* high_sum_smooth, const float* high_abs_sum_smooth,
                                            const float* weight_sq_sum, const float* weight_sum, const float* energy,
                                            const float* energy_floor, float gate_threshold, float boost, int16_t* out,
                                            int64_t count) {
    ELEMENT_INDEX
    float y = fused[i];
    float low = fused_low[i];
    float high = current_high[i];
    float h_sum = high_sum_smooth[i];
    float h_abs_sum = high_abs_sum_smooth[i];
    float w_sq_sum = weight_sq_sum[i];
    float w_sum = weight_sum[i];
    float e = energy[i];
    float e_floor = *energy_floor;
    float noise_coherence = sqrtf(w_sq_sum) / w_sum;
    float coherence = saturate((h_sum / fmaxf(h_abs_sum, 1e-6f) - noise_coherence) / fmaxf(1 - noise_coherence, .05f));
    float texture = saturate((e / fmaxf(e_floor, 1e-6f) - 3) / 3);
    float gate = saturate((coherence - gate_threshold) / (1 - gate_threshold)) * texture;
    float z = low + gate * (1 + (boost - 1) * texture) * high + (1 - gate) * (y - low);
    out[i] = (short)__float2int_rn(fminf(1023.f, fmaxf(0.f, z)));
}

// pack_10bit: float to int16, rounded to the nearest integer (ties to even) and clamped to 0 ... 1023 (the chroma
// output).
extern "C" __global__ void pack_10bit(const float* src, int16_t* dst, int64_t count) {
    ELEMENT_INDEX
    dst[i] = (short)__float2int_rn(fminf(1023.f, fmaxf(0.f, src[i])));
}

// ------------------------------------------------------------------------------------------------ order statistics
// The k-th smallest of n floats with a 4-pass radix select (8 bits per pass, most significant first) on order-
// preserving integer keys. The state stays on the device: no host round trip between the passes.
//   state[0] = the key prefix found so far, state[1] = the rank still to find among the keys with that prefix;
//   histogram = 256 counters (zeroed by radix_select_init and after each scan).

// Order-preserving map of float bits to uint32: negative floats reversed below the positive ones.
__device__ __forceinline__ uint32_t float_to_ordered_key(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Inverse of float_to_ordered_key.
__device__ __forceinline__ float ordered_key_to_float(uint32_t key) {
    uint32_t u = (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
    return __uint_as_float(u);
}

// gather_strided: out[p * m + j] = |x[p * plane_size + j * stride]| (or the value itself with take_abs = 0), for
// planes p < planes and j < m: the reference's x[:, ::stride] subsample of each plane.
extern "C" __global__ void gather_strided(const float* x, float* out, int64_t plane_size, int64_t stride, int64_t m,
                                          int planes, int take_abs) {
    int64_t count = m * planes;
    ELEMENT_INDEX
    int64_t p = i / m;
    int64_t j = i % m;
    float v = x[p * plane_size + j * stride];
    out[i] = take_abs ? fabsf(v) : v;
}

// radix_select_init: state = (0, k), histogram zeroed. One block.
extern "C" __global__ void radix_select_init(uint32_t* state, uint32_t* histogram, uint32_t k) {
    int t = threadIdx.x;
    if (t == 0) {
        state[0] = 0;
        state[1] = k;
    }
    for (int b = t; b < 256; b += blockDim.x) {
        histogram[b] = 0;
    }
}

// radix_select_histogram: counts the keys that match the prefix above bit shift + 8 by their digit at `shift`
// (24, 16, 8, 0). Grid-stride loop; per-block counts in shared memory, then added to the global histogram.
extern "C" __global__ void radix_select_histogram(const float* values, int64_t n, const uint32_t* state,
                                                  uint32_t* histogram, int shift) {
    __shared__ uint32_t block_hist[256];
    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        block_hist[b] = 0;
    }
    __syncthreads();
    uint32_t prefix = state[0];
    uint32_t prefix_mask = shift >= 24 ? 0u : (0xffffffffu << (shift + 8));
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        uint32_t key = float_to_ordered_key(values[i]);
        if ((key & prefix_mask) == (prefix & prefix_mask)) {
            atomicAdd(&block_hist[(key >> shift) & 255u], 1u);
        }
    }
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        if (block_hist[b]) {
            atomicAdd(&histogram[b], block_hist[b]);
        }
    }
}

// radix_select_scan: finds the digit that holds the wanted rank, appends it to the prefix, reduces the rank by the
// counts below it and clears the histogram. One thread.
extern "C" __global__ void radix_select_scan(uint32_t* state, uint32_t* histogram, int shift) {
    if (threadIdx.x != 0) {
        return;
    }
    uint32_t rank = state[1];
    uint32_t below = 0;
    for (int b = 0; b < 256; b++) {
        uint32_t c = histogram[b];
        if (below + c > rank) {
            state[0] |= ((uint32_t)b << shift);
            state[1] = rank - below;
            break;
        }
        below += c;
    }
    for (int b = 0; b < 256; b++) {
        histogram[b] = 0;
    }
}

// radix_select_value: the selected value (after the four passes the prefix is the whole key).
extern "C" __global__ void radix_select_value(const uint32_t* state, float* out) {
    if (threadIdx.x == 0) {
        *out = ordered_key_to_float(state[0]);
    }
}

// noise_variance: the noise variance of a plane from the median absolute high-pass value (a robust estimate:
// sigma = 1.4826 * median |hp|, 1.4826 = 1 / Phi^-1(3/4)), floored at 0.001 and scaled by `scale` (fallback_eps):
// var = max((1.4826 median)^2, 0.001) * scale.
extern "C" __global__ void noise_variance(const float* median, float* variance, float scale) {
    if (threadIdx.x == 0) {
        float sigma = 1.4826f * *median;
        *variance = fmaxf(sigma * sigma, 0.001f) * scale;
    }
}

// quantile_lerp: the linear-interpolation quantile from its two neighbouring order statistics (lo = sorted value at
// floor(q (n - 1)), hi = the next one, w = the fraction), in NumPy / CuPy's form: lo + (hi - lo) w for w < 0.5,
// else hi - (hi - lo) (1 - w).
extern "C" __global__ void quantile_lerp(const float* lo, const float* hi, float w, float* out) {
    if (threadIdx.x != 0) {
        return;
    }
    float a = *lo;
    float b = *hi;
    float diff = b - a;
    *out = w < 0.5f ? a + diff * w : b - diff * (1 - w);
}
