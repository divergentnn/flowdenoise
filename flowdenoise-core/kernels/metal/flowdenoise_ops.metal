// flowdenoise_ops.metal: the elementwise and order-statistic kernels in Metal Shading Language (a port of
// flowdenoise_ops.cu; read the kernel descriptions there), plus the copy and fill kernels of the Metal driver.
//
// Compiled WITHOUT fast math, as the CUDA file. Each kernel takes its CUDA argument list as one constant struct
// (same order, C layout, device pointers inside it, packed by the host). Elementwise kernels: one thread per
// element, 1-D grid of 256-thread threadgroups; the index is 32 bit (every buffer here has fewer than 2^32 elements,
// and 64-bit division is slow on the GPU).
#include <metal_stdlib>
using namespace metal;

#define KERNEL_INDICES                                                                                   \
    uint3 blockIdx [[threadgroup_position_in_grid]], uint3 threadIdx [[thread_position_in_threadgroup]], \
        uint3 blockDim [[threads_per_threadgroup]], uint3 gridDim [[threadgroups_per_grid]],             \
        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
#define SYNC_THREADGROUP() threadgroup_barrier(mem_flags::mem_threadgroup)
// the element index of this thread; threads past the end return
#define ELEMENT_INDEX                               \
    uint i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= (ulong)count) {                        \
        return;                                     \
    }
typedef device const float* in_plane;
typedef device float* out_plane;
typedef device const uchar* in_bytes;
typedef device uchar* out_bytes;

static inline float saturate_unit(float value) {
    return fmin(1.f, fmax(0.f, value));
}

// ------------------------------------------------------------------------------------------------ copies (driver)

// dst[i] = src[i] over count 32-bit words (device-to-device and staged host-to-device copies)
struct CopyWordsArgs {
    device uint* dst;
    device const uint* src;
    long count;
};
kernel void copy_words(constant CopyWordsArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    args.dst[i] = args.src[i];
}
// dst[i] = src[i] over count bytes (unaligned copies)
struct CopyBytesArgs {
    out_bytes dst;
    in_bytes src;
    long count;
};
kernel void copy_bytes(constant CopyBytesArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    args.dst[i] = args.src[i];
}
// dst[i] = value over count bytes (memset)
struct FillBytesArgs {
    out_bytes dst;
    int value;
    long count;
};
kernel void fill_bytes(constant FillBytesArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    args.dst[i] = (uchar)args.value;
}

// ------------------------------------------------------------------------------------------------ conversions

// 10-bit codes (uint16) to float
struct U16ToFloatArgs {
    device const ushort* src;
    out_plane dst;
    long count;
};
kernel void u16_to_float(constant U16ToFloatArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    args.dst[i] = (float)args.src[i];
}

// unary elementwise kernels: out = f(a)
struct UnaryArgs {
    in_plane a;
    out_plane out;
    long count;
};
// binary elementwise kernels: out = f(a, b)
struct BinaryArgs {
    in_plane a;
    in_plane b;
    out_plane out;
    long count;
};

// motion pyramid base: clip(rint(x / 4), 0, 255)
kernel void quantize_8bit(constant UnaryArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    float v = rint(args.a[i] / 4.f);
    args.out[i] = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
}
// rint (ties to even)
kernel void round_to_integer(constant UnaryArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    args.out[i] = rint(args.a[i]);
}

// ------------------------------------------------------------------------------------------------ elementwise

// |a - b|
kernel void abs_diff(constant BinaryArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    args.out[i] = fabs(args.a[i] - args.b[i]);
}
// (a - b)^2
kernel void squared_diff(constant BinaryArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    float d = args.a[i] - args.b[i];
    args.out[i] = d * d;
}
// a - b
kernel void subtract(constant BinaryArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    args.out[i] = args.a[i] - args.b[i];
}
// a^2
kernel void square(constant UnaryArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    args.out[i] = args.a[i] * args.a[i];
}
// |a|
kernel void absolute(constant UnaryArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    args.out[i] = fabs(args.a[i]);
}
// out = a / b[i % period]
struct DivideBroadcastArgs {
    in_plane a;
    in_plane b;
    out_plane out;
    long count;
    long period;
};
kernel void divide_broadcast(constant DivideBroadcastArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    args.out[i] = args.a[i] / args.b[i % (uint)args.period];
}

// ------------------------------------------------------------------------------------------------ guided fallback

// guided filter coefficients: variance = max(mean_sq - mean^2, 0); a = variance / (variance + eps); b = mean - a mean
// (eps per plane: eps[i / plane_size])
struct GuidedCoefficientsArgs {
    in_plane mean;
    in_plane mean_sq;
    in_plane eps;
    out_plane a;
    out_plane b;
    long count;
    long plane_size;
};
kernel void guided_coefficients(constant GuidedCoefficientsArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    float e = args.eps[i / (uint)args.plane_size];
    float m = args.mean[i];
    float variance = fmax(args.mean_sq[i] - m * m, 0.f);
    float coef_a = variance / (variance + e);
    args.a[i] = coef_a;
    args.b[i] = m - coef_a * m;
}
// fallback blend factor: clamp((fallback - weight_sum^2 / weight_sq_sum) / (fallback - 1), 0, 1)
struct FallbackAlphaArgs {
    in_plane weight_sum;
    in_plane weight_sq_sum;
    float fallback;
    out_plane alpha;
    long count;
};
kernel void fallback_alpha(constant FallbackAlphaArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    const float fallback = args.fallback;
    ELEMENT_INDEX
    args.alpha[i] =
        saturate_unit((fallback - args.weight_sum[i] * args.weight_sum[i] / args.weight_sq_sum[i]) / (fallback - 1));
}
// out = x + alpha (mean_a x + mean_b - x), alpha broadcast (alpha[i % alpha_period])
struct GuidedBlendArgs {
    in_plane x;
    in_plane alpha;
    in_plane mean_a;
    in_plane mean_b;
    out_plane out;
    long count;
    long alpha_period;
};
kernel void guided_blend(constant GuidedBlendArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    float v = args.x[i];
    args.out[i] = v + args.alpha[i % (uint)args.alpha_period] * (args.mean_a[i] * v + args.mean_b[i] - v);
}

// ------------------------------------------------------------------------------------------------ stability gate

// stable-texture share s = clamp((energy / max(noise_var fb_q weight_sq_sum / weight_sum^2, 1e-6) - 1) / gate_span)
// (flowdenoise_ops.cu, stability_share)
struct StabilityShareArgs {
    in_plane energy;
    in_plane noise_var;
    float fb_q;
    in_plane weight_sq_sum;
    in_plane weight_sum;
    float gate_span;
    out_plane share;
    long count;
};
kernel void stability_share(constant StabilityShareArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    float scaled_var = *args.noise_var * args.fb_q;
    float w_sum = args.weight_sum[i];
    float expected = fmax(scaled_var * args.weight_sq_sum[i] / (w_sum * w_sum), 1e-6f);
    float ratio = args.energy[i] / expected;
    args.share[i] = saturate_unit((ratio - 1) / args.gate_span);
}
// out = a (1 - s), in place allowed
struct ScaleByComplementArgs {
    in_plane a;
    in_plane s;
    out_plane out;
    long count;
};
kernel void scale_by_complement(constant ScaleByComplementArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    args.out[i] = args.a[i] * (1 - args.s[i]);
}

// ------------------------------------------------------------------------------------------------ range check
// (flowdenoise_ops.cu has the description)

// samples[(*sample_count)++] = abs_diff[j * stride] where the current luma there is in (100, 900), j < m
struct RangeSampleGatherArgs {
    in_plane abs_diff;
    in_plane luma;
    out_plane samples;
    device atomic_uint* sample_count;
    long stride;
    long m;
};
kernel void range_sample_gather(constant RangeSampleGatherArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.m;
    ELEMENT_INDEX
    uint p = i * (uint)args.stride;
    float v = args.luma[p];
    if (v > 100.f && v < 900.f) {
        args.samples[atomic_fetch_add_explicit(args.sample_count, 1u, memory_order_relaxed)] = args.abs_diff[p];
    }
}
// fewer than `minimum` mid-luma samples: all of them, samples[j] = abs_diff[j * stride]
struct RangeSampleFallbackArgs {
    in_plane abs_diff;
    out_plane samples;
    device const uint* sample_count;
    uint minimum;
    long stride;
    long m;
};
kernel void range_sample_fallback(constant RangeSampleFallbackArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.m;
    ELEMENT_INDEX
    if (*args.sample_count < args.minimum) {
        args.samples[i] = args.abs_diff[i * (uint)args.stride];
    }
}
// sigma_d = max(1.4826 median, 0.5); thresholds = (sigma_d k, sigma_d k_soft)
struct RangeThresholdsArgs {
    in_plane median;
    float k, k_soft;
    out_plane thresholds;
};
kernel void range_thresholds(constant RangeThresholdsArgs& args [[buffer(0)]], KERNEL_INDICES) {
    if (threadIdx.x == 0) {
        float sigma = fmax(1.4826f * *args.median, .5f);
        args.thresholds[0] = sigma * args.k;
        args.thresholds[1] = sigma * args.k_soft;
    }
}
// weight *= clamp((t1 - abs_diff) / max(t1 - t0, 1e-6), 0, 1)
struct RangeCheckArgs {
    in_plane abs_diff;
    in_plane thresholds;
    out_plane weight;
    long count;
};
kernel void range_check(constant RangeCheckArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    float t0 = args.thresholds[0];
    float t1 = args.thresholds[1];
    args.weight[i] *= saturate_unit((t1 - args.abs_diff[i]) / fmax(t1 - t0, 1e-6f));
}

// ------------------------------------------------------------------------------------------------ detail gate, output

// detail gate and output luma (flowdenoise_ops.cu, detail_gate_luma)
struct DetailGateArgs {
    in_plane fused;
    in_plane fused_low;
    in_plane current_high;
    in_plane high_sum_smooth;
    in_plane high_abs_sum_smooth;
    in_plane weight_sq_sum;
    in_plane weight_sum;
    in_plane energy;
    in_plane energy_floor;
    float gate_threshold, boost;
    device short* out;
    long count;
};
kernel void detail_gate_luma(constant DetailGateArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    const float gate_threshold = args.gate_threshold;
    const float boost = args.boost;
    ELEMENT_INDEX
    float y = args.fused[i];
    float low = args.fused_low[i];
    float high = args.current_high[i];
    float h_sum = args.high_sum_smooth[i];
    float h_abs_sum = args.high_abs_sum_smooth[i];
    float w_sq_sum = args.weight_sq_sum[i];
    float w_sum = args.weight_sum[i];
    float e = args.energy[i];
    float e_floor = *args.energy_floor;
    float noise_coherence = sqrt(w_sq_sum) / w_sum;
    float coherence =
        saturate_unit((h_sum / fmax(h_abs_sum, 1e-6f) - noise_coherence) / fmax(1 - noise_coherence, .05f));
    float texture = saturate_unit((e / fmax(e_floor, 1e-6f) - 3) / 3);
    float gate = saturate_unit((coherence - gate_threshold) / (1 - gate_threshold)) * texture;
    float z = low + gate * (1 + (boost - 1) * texture) * high + (1 - gate) * (y - low);
    args.out[i] = (short)(int)rint(fmin(1023.f, fmax(0.f, z)));
}
// float to int16, rounded (ties to even) and clamped to 0 ... 1023
struct Pack10BitArgs {
    in_plane src;
    device short* dst;
    long count;
};
kernel void pack_10bit(constant Pack10BitArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.count;
    ELEMENT_INDEX
    args.dst[i] = (short)(int)rint(fmin(1023.f, fmax(0.f, args.src[i])));
}

// ------------------------------------------------------------------------------------------------ order statistics
// 4-pass radix select on order-preserving keys (flowdenoise_ops.cu). state[0] = key prefix, state[1] = rank left,
// state[2] = n (the device-counted median only).

static inline uint float_to_ordered_key(float f) {
    uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
static inline float ordered_key_to_float(uint key) {
    uint u = (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
    return as_type<float>(u);
}
// out[p * m + j] = |x[p * plane_size + j * stride]| (or the value itself with take_abs = 0)
struct GatherStridedArgs {
    in_plane x;
    out_plane out;
    long plane_size;
    long stride;
    long m;
    int planes, take_abs;
};
kernel void gather_strided(constant GatherStridedArgs& args [[buffer(0)]], KERNEL_INDICES) {
    long count = args.m * args.planes;
    ELEMENT_INDEX
    uint m = (uint)args.m;
    uint p = i / m;
    uint j = i % m;
    float v = args.x[p * args.plane_size + j * args.stride];
    args.out[i] = args.take_abs ? fabs(v) : v;
}
struct RadixSelectInitArgs {
    device uint* state;
    device uint* histogram;
    uint k;
};
kernel void radix_select_init(constant RadixSelectInitArgs& args [[buffer(0)]], KERNEL_INDICES) {
    uint t = threadIdx.x;
    if (t == 0) {
        args.state[0] = 0;
        args.state[1] = args.k;
    }
    for (uint b = t; b < 256; b += blockDim.x) {
        args.histogram[b] = 0;
    }
}
struct RadixSelectHistogramArgs {
    in_plane values;
    long n;
    device const uint* state;
    device atomic_uint* histogram;
    int shift;
};
kernel void radix_select_histogram(constant RadixSelectHistogramArgs& args [[buffer(0)]], KERNEL_INDICES) {
    threadgroup atomic_uint block_hist[256];
    for (uint b = threadIdx.x; b < 256; b += blockDim.x) {
        atomic_store_explicit(&block_hist[b], 0u, memory_order_relaxed);
    }
    SYNC_THREADGROUP();
    const int shift = args.shift;
    uint prefix = args.state[0];
    uint prefix_mask = shift >= 24 ? 0u : (0xffffffffu << (shift + 8));
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < args.n; i += (long)gridDim.x * blockDim.x) {
        uint key = float_to_ordered_key(args.values[i]);
        if ((key & prefix_mask) == (prefix & prefix_mask)) {
            atomic_fetch_add_explicit(&block_hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
        }
    }
    SYNC_THREADGROUP();
    for (uint b = threadIdx.x; b < 256; b += blockDim.x) {
        uint c = atomic_load_explicit(&block_hist[b], memory_order_relaxed);
        if (c) {
            atomic_fetch_add_explicit(&args.histogram[b], c, memory_order_relaxed);
        }
    }
}
struct RadixSelectScanArgs {
    device uint* state;
    device uint* histogram;
    int shift;
};
kernel void radix_select_scan(constant RadixSelectScanArgs& args [[buffer(0)]], KERNEL_INDICES) {
    if (threadIdx.x != 0) {
        return;
    }
    uint rank = args.state[1];
    uint below = 0;
    for (int b = 0; b < 256; b++) {
        uint c = args.histogram[b];
        if (below + c > rank) {
            args.state[0] |= ((uint)b << args.shift);
            args.state[1] = rank - below;
            break;
        }
        below += c;
    }
    for (int b = 0; b < 256; b++) {
        args.histogram[b] = 0;
    }
}
struct RadixSelectValueArgs {
    device const uint* state;
    out_plane out;
};
kernel void radix_select_value(constant RadixSelectValueArgs& args [[buffer(0)]], KERNEL_INDICES) {
    if (threadIdx.x == 0) {
        *args.out = ordered_key_to_float(args.state[0]);
    }
}
// the median of a device-counted set: n = *sample_count if >= minimum, else fallback_n; state = (0, (n - 1) / 2, n)
struct RadixSelectInitMedianArgs {
    device uint* state;
    device uint* histogram;
    device const uint* sample_count;
    uint minimum, fallback_n;
};
kernel void radix_select_init_median(constant RadixSelectInitMedianArgs& args [[buffer(0)]], KERNEL_INDICES) {
    uint t = threadIdx.x;
    if (t == 0) {
        uint c = *args.sample_count;
        uint n = c >= args.minimum ? c : args.fallback_n;
        args.state[0] = 0;
        args.state[1] = (n - 1) / 2;
        args.state[2] = n;
    }
    for (uint b = t; b < 256; b += blockDim.x) {
        args.histogram[b] = 0;
    }
}
// radix_select_histogram over the first min(state[2], capacity) values
struct RadixSelectHistogramCountedArgs {
    in_plane values;
    long capacity;
    device const uint* state;
    device atomic_uint* histogram;
    int shift;
};
kernel void radix_select_histogram_counted(constant RadixSelectHistogramCountedArgs& args [[buffer(0)]],
                                           KERNEL_INDICES) {
    threadgroup atomic_uint block_hist[256];
    for (uint b = threadIdx.x; b < 256; b += blockDim.x) {
        atomic_store_explicit(&block_hist[b], 0u, memory_order_relaxed);
    }
    SYNC_THREADGROUP();
    const int shift = args.shift;
    uint prefix = args.state[0];
    long n = (long)args.state[2] < args.capacity ? (long)args.state[2] : args.capacity;
    uint prefix_mask = shift >= 24 ? 0u : (0xffffffffu << (shift + 8));
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        uint key = float_to_ordered_key(args.values[i]);
        if ((key & prefix_mask) == (prefix & prefix_mask)) {
            atomic_fetch_add_explicit(&block_hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
        }
    }
    SYNC_THREADGROUP();
    for (uint b = threadIdx.x; b < 256; b += blockDim.x) {
        uint c = atomic_load_explicit(&block_hist[b], memory_order_relaxed);
        if (c) {
            atomic_fetch_add_explicit(&args.histogram[b], c, memory_order_relaxed);
        }
    }
}
// noise variance from the median absolute high-pass value: max((1.4826 median)^2, 0.001) * scale
struct NoiseVarianceArgs {
    in_plane median;
    out_plane variance;
    float scale;
};
kernel void noise_variance(constant NoiseVarianceArgs& args [[buffer(0)]], KERNEL_INDICES) {
    if (threadIdx.x == 0) {
        float sigma = 1.4826f * *args.median;
        *args.variance = fmax(sigma * sigma, 0.001f) * args.scale;
    }
}
// linear-interpolation quantile from its two neighbouring order statistics (NumPy / CuPy's form)
struct QuantileLerpArgs {
    in_plane lo;
    in_plane hi;
    float w;
    out_plane out;
};
kernel void quantile_lerp(constant QuantileLerpArgs& args [[buffer(0)]], KERNEL_INDICES) {
    if (threadIdx.x != 0) {
        return;
    }
    float a = *args.lo;
    float b = *args.hi;
    float diff = b - a;
    float w = args.w;
    *args.out = w < 0.5f ? a + diff * w : b - diff * (1 - w);
}
