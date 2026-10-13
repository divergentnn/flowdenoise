// flowdenoise.metal: the image kernels of the denoiser in Metal Shading Language, a port of flowdenoise.cu (read the
// kernel descriptions there; this file notes what differs on Metal).
//
// Each kernel takes its CUDA argument list as one constant struct (same order, C layout, device pointers inside it,
// packed by the host); the body reads the same names as the CUDA kernel. The CUDA thread / block indices keep their
// CUDA names (KERNEL_INDICES). Compiled with fast math, as the CUDA kernels; not bit-exact with CUDA (another compiler,
// other fast-math functions, another summation order in the motion search), but the same operations in the same
// order otherwise (README.md, "Numerical rules").
#include <metal_stdlib>
using namespace metal;

#define KERNEL_INDICES                                                                                   \
    uint3 blockIdx [[threadgroup_position_in_grid]], uint3 threadIdx [[thread_position_in_threadgroup]], \
        uint3 blockDim [[threads_per_threadgroup]]
#define SYNC_THREADGROUP() threadgroup_barrier(mem_flags::mem_threadgroup)
typedef device const float* in_plane;
typedef device float* out_plane;

static inline int max_int(int a, int b) {
    return a > b ? a : b;
}
static inline int min_int(int a, int b) {
    return a < b ? a : b;
}
// |(a, b)| (CUDA hypotf)
static inline float hypot2(float a, float b) {
    return sqrt(a * a + b * b);
}

// ------------------------------------------------------------------------------------------------ sampling helpers

static inline int clamp_index(int index, int size) {
    return max_int(0, min_int(index, size - 1));
}
static inline float saturate_unit(float value) {
    return fmin(1.f, fmax(0.f, value));
}
static inline float sample_clamped(in_plane plane, int x, int y, int width, int height) {
    return plane[clamp_index(y, height) * width + clamp_index(x, width)];
}
static inline float sample_bilinear(in_plane plane, float x, float y, int width, int height) {
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    float fx = x - x0;
    float fy = y - y0;
    return (1 - fy) * ((1 - fx) * sample_clamped(plane, x0, y0, width, height) +
                       fx * sample_clamped(plane, x0 + 1, y0, width, height)) +
           fy * ((1 - fx) * sample_clamped(plane, x0, y0 + 1, width, height) +
                 fx * sample_clamped(plane, x0 + 1, y0 + 1, width, height));
}

// ------------------------------------------------------------------------------------------------ separable filters

enum {
    AXIS_X = 0,
    AXIS_Y = 1
};

// separable_filter_<TAPS>_x / _y: one pass of a separable FIR filter with the tile in threadgroup memory
// (flowdenoise.cu, separable_filter_tile). Grid (ceil(width / 32), ceil(height / 8), planes), 32 x 8 threads.
struct SeparableFilterArgs {
    in_plane src;
    out_plane dst;
    int width, height;
    in_plane taps;
    int zero_pad, prefilter_edges;
};
template <int TAPS, int AXIS>
kernel void separable_filter_tile(constant SeparableFilterArgs& args [[buffer(0)]], KERNEL_INDICES) {
    constexpr int TILE_W = 32 + (AXIS == AXIS_X ? TAPS - 1 : 0);
    constexpr int TILE_H = 8 + (AXIS == AXIS_Y ? TAPS - 1 : 0);
    constexpr int tap_stride = AXIS == AXIS_X ? 1 : TILE_W;
    threadgroup float tile[TILE_W * TILE_H];
    in_plane src = args.src;
    out_plane dst = args.dst;
    const int width = args.width;
    const int height = args.height;
    in_plane taps = args.taps;
    const int zero_pad = args.zero_pad;
    const int prefilter_edges = args.prefilter_edges;

    int thread_index = threadIdx.y * 32 + threadIdx.x;
    int tile_x0 = blockIdx.x * 32;
    int tile_y0 = blockIdx.y * 8;
    src += blockIdx.z * width * height;
    dst += blockIdx.z * width * height;
    for (int j = thread_index; j < TILE_W * TILE_H; j += 256) {
        int x = tile_x0 + j % TILE_W - (AXIS == AXIS_X ? TAPS / 2 : 0);
        int y = tile_y0 + j / TILE_W - (AXIS == AXIS_Y ? TAPS / 2 : 0);
        tile[j] = (zero_pad && (x < 0 || x >= width || y < 0 || y >= height))
                      ? 0
                      : src[clamp_index(y, height) * width + clamp_index(x, width)];
    }
    SYNC_THREADGROUP();

    int x = tile_x0 + threadIdx.x;
    int y = tile_y0 + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    float sum = 0;
    if (TAPS % 2) {
        int centre =
            (threadIdx.y + (AXIS == AXIS_Y ? TAPS / 2 : 0)) * TILE_W + threadIdx.x + (AXIS == AXIS_X ? TAPS / 2 : 0);
        sum = taps[TAPS / 2] * tile[centre];
#pragma unroll
        for (int j = 1; j <= TAPS / 2; j++) {
            sum += taps[TAPS / 2 + j] * (tile[centre + j * tap_stride] + tile[centre - j * tap_stride]);
        }
    } else {
#pragma unroll
        for (int j = 0; j < TAPS; j++) {
            sum += taps[j] *
                   tile[(threadIdx.y + (AXIS == AXIS_Y ? j : 0)) * TILE_W + threadIdx.x + (AXIS == AXIS_X ? j : 0)];
        }
    }
    // two edge-clamped [1 2 1] / 4 passes == the binomial-5 pass except at the first and last sample
    if (TAPS == 5 && prefilter_edges) {
        int position = AXIS == AXIS_X ? x : y;
        int extent = AXIS == AXIS_X ? width : height;
        int centre = (threadIdx.y + (AXIS == AXIS_Y ? 2 : 0)) * TILE_W + threadIdx.x + (AXIS == AXIS_X ? 2 : 0);
        if (position == 0) {
            sum += (tile[centre + tap_stride] - tile[centre]) * .0625f;
        }
        if (position == extent - 1) {
            sum += (tile[centre - tap_stride] - tile[centre]) * .0625f;
        }
    }
    dst[y * width + x] = sum;
}
#define SEPARABLE_FILTER(TAPS)                                                                              \
    template [[host_name("separable_filter_" #TAPS "_x")]] kernel void separable_filter_tile<TAPS, AXIS_X>( \
        constant SeparableFilterArgs&, uint3, uint3, uint3);                                                \
    template [[host_name("separable_filter_" #TAPS "_y")]] kernel void separable_filter_tile<TAPS, AXIS_Y>( \
        constant SeparableFilterArgs&, uint3, uint3, uint3);
SEPARABLE_FILTER(3)
SEPARABLE_FILTER(5)
SEPARABLE_FILTER(7)
SEPARABLE_FILTER(8)
SEPARABLE_FILTER(19)
SEPARABLE_FILTER(37)

// separable_filter_any: one pass of a separable FIR filter with any number of taps, one thread per output sample
// (flowdenoise.cu).
struct SeparableFilterAnyArgs {
    in_plane src;
    out_plane dst;
    int width, height, count;
    in_plane taps;
    int tap_count, axis, zero_pad;
};
kernel void separable_filter_any(constant SeparableFilterAnyArgs& args [[buffer(0)]], KERNEL_INDICES) {
    in_plane src = args.src;
    out_plane dst = args.dst;
    const int width = args.width;
    const int height = args.height;
    const int count = args.count;
    const int tap_count = args.tap_count;
    const int axis = args.axis;
    const int zero_pad = args.zero_pad;
    in_plane taps = args.taps;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    int x = i % width;
    int y = (i / width) % height;
    int plane_start = i - y * width - x;
    float sum = 0;
    for (int j = 0; j < tap_count; j++) {
        int xx = x + (axis == AXIS_X ? j - tap_count / 2 : 0);
        int yy = y + (axis == AXIS_Y ? j - tap_count / 2 : 0);
        if (!zero_pad || (xx >= 0 && xx < width && yy >= 0 && yy < height)) {
            sum += taps[j] * src[plane_start + clamp_index(yy, height) * width + clamp_index(xx, width)];
        }
    }
    dst[i] = sum;
}

// ------------------------------------------------------------------------------------------------ motion pyramid

// downsample_2x2: halves a plane by averaging each 2x2 block (flowdenoise.cu). One thread per output sample.
struct DownsampleArgs {
    in_plane src;
    out_plane dst;
    int width, height;
};
kernel void downsample_2x2(constant DownsampleArgs& args [[buffer(0)]], KERNEL_INDICES) {
    in_plane src = args.src;
    out_plane dst = args.dst;
    const int width = args.width;
    const int height = args.height;
    int out_w = width / 2;
    int out_h = height / 2;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_w * out_h) {
        return;
    }
    int p = (i / out_w * 2) * width + (i % out_w * 2);
    dst[i] = (src[p] + src[p + 1] + src[p + width] + src[p + width + 1]) * .25f;
}

// ------------------------------------------------------------------------------------------------ motion search

// Sum over one 8x8 patch: each of the 32 lanes of the SIMD group holds two of its pixels (a, b), so the patch sum is
// one hardware SIMD reduction, without threadgroup barriers. (CUDA: 64 threads, a shuffle tree per warp, then the two
// warps: another summation order.)
static inline float patch_sum(float a, float b) {
    return simd_sum(a + b);
}

// motion_search_sweep: DIS-style patch motion search on one pyramid level, one sweep along each row of patches
// (flowdenoise.cu has the algorithm). Metal differences:
// - one SIMD group of 32 threads per patch row; lane t holds the patch pixels (t % 8, t / 8) and (t % 8, t / 8 + 4)
//   (suffixes 0 and 1 below), so every patch sum is patch_sum(pixel 0 term, pixel 1 term);
// - the neighbours' levels come as up to 6 pointers (neighbours[d]) instead of one stacked copy.
// Grid (patches_y, neighbours), 32 threads per threadgroup.
struct MotionSearchArgs {
    in_plane current;
    in_plane neighbours[6];
    int width, height;
    in_plane coarse_flow;
    int coarse_w, coarse_h;
    in_plane forward;
    out_plane vectors;
    int patches_x, patches_y, reverse, iterations;
};
kernel void motion_search_sweep(constant MotionSearchArgs& args [[buffer(0)]], KERNEL_INDICES) {
    in_plane current = args.current;
    in_plane coarse_flow = args.coarse_flow;
    in_plane forward = args.forward;
    out_plane vectors = args.vectors;
    const int width = args.width;
    const int height = args.height;
    const int coarse_w = args.coarse_w;
    const int coarse_h = args.coarse_h;
    const int patches_x = args.patches_x;
    const int patches_y = args.patches_y;
    const int reverse = args.reverse;
    const int iterations = args.iterations;
    int patch_y = blockIdx.x;
    int thread_index = threadIdx.x;
    int neighbour = blockIdx.y;
    in_plane other = args.neighbours[neighbour];
    vectors += neighbour * 2 * patches_x * patches_y;
    if (coarse_flow) {
        coarse_flow += neighbour * 2 * coarse_w * coarse_h;
    }
    if (forward) {
        forward += neighbour * 2 * patches_x * patches_y;
    }
    float last_u = 0;
    float last_v = 0;
    for (int col = 0; col < patches_x; col++) {
        int patch_x = reverse ? patches_x - 1 - col : col;
        int px = patch_x * 4 + thread_index % 8;
        int py0 = patch_y * 4 + thread_index / 8;
        int py1 = py0 + 4;

        // template pixels and their Sobel gradients
        float c0 = sample_clamped(current, px, py0, width, height);
        float c1 = sample_clamped(current, px, py1, width, height);
        float gx0 = 0;
        float gy0 = 0;
        float gx1 = 0;
        float gy1 = 0;
        for (int j = -1; j <= 1; j++) {
            float sobel_w = j == 0 ? 2.f : 1.f;
            gx0 += sobel_w * (sample_clamped(current, px + 1, py0 + j, width, height) -
                              sample_clamped(current, px - 1, py0 + j, width, height));
            gy0 += sobel_w * (sample_clamped(current, px + j, py0 + 1, width, height) -
                              sample_clamped(current, px + j, py0 - 1, width, height));
            gx1 += sobel_w * (sample_clamped(current, px + 1, py1 + j, width, height) -
                              sample_clamped(current, px - 1, py1 + j, width, height));
            gy1 += sobel_w * (sample_clamped(current, px + j, py1 + 1, width, height) -
                              sample_clamped(current, px + j, py1 - 1, width, height));
        }
        float gxx = patch_sum(gx0 * gx0, gx1 * gx1);
        float gyy = patch_sum(gy0 * gy0, gy1 * gy1);
        float gxy = patch_sum(gx0 * gy0, gx1 * gy1);
        float mean_gx = patch_sum(gx0, gx1) / 64;
        float mean_gy = patch_sum(gy0, gy1) / 64;

        // initial vector: forward sweep, else coarser level at the patch centre, else zero
        float u = 0;
        float v = 0;
        if (forward) {
            u = forward[patch_y * patches_x + patch_x];
            v = forward[patches_x * patches_y + patch_y * patches_x + patch_x];
        } else if (coarse_flow) {
            float cx = (patch_x * 4 + 4 + .5f) / 2 - .5f;
            float cy = (patch_y * 4 + 4 + .5f) / 2 - .5f;
            u = 2 * sample_bilinear(coarse_flow, cx, cy, coarse_w, coarse_h);
            v = 2 * sample_bilinear(coarse_flow + coarse_w * coarse_h, cx, cy, coarse_w, coarse_h);
        }

        // candidates: initial, previous patch of this sweep, (reverse sweep) the forward vector of the patch below
        float best_ssd = 1e30f;
        float best_u = u;
        float best_v = v;
        for (int cand = 0; cand < 3; cand++) {
            float cand_u = u;
            float cand_v = v;
            if (cand == 1) {
                if (col == 0) {
                    continue;
                }
                cand_u = last_u;
                cand_v = last_v;
            }
            if (cand == 2) {
                if (!forward || patch_y + 1 >= patches_y) {
                    continue;
                }
                cand_u = forward[(patch_y + 1) * patches_x + patch_x];
                cand_v = forward[patches_x * patches_y + (patch_y + 1) * patches_x + patch_x];
            }
            float diff0 = sample_bilinear(other, px + cand_u, py0 + cand_v, width, height) - c0;
            float diff1 = sample_bilinear(other, px + cand_u, py1 + cand_v, width, height) - c1;
            float diff_sum = patch_sum(diff0, diff1);
            float ssd = patch_sum(diff0 * diff0, diff1 * diff1) - diff_sum * diff_sum / 64;
            if (ssd < best_ssd) {
                best_ssd = ssd;
                best_u = cand_u;
                best_v = cand_v;
            }
        }

        // inverse-compositional Gauss-Newton refinement (update applied before the convergence test)
        u = best_u;
        v = best_v;
        float det = fmax(gxx * gyy - gxy * gxy, .001f);
        float prev_ssd = 1e30f;
        for (int it = 0; it < iterations; it++) {
            float diff0 = sample_bilinear(other, px + u, py0 + v, width, height) - c0;
            float diff1 = sample_bilinear(other, px + u, py1 + v, width, height) - c1;
            float diff_sum = patch_sum(diff0, diff1);
            float ssd = patch_sum(diff0 * diff0, diff1 * diff1) - diff_sum * diff_sum / 64;
            float rx = patch_sum(diff0 * gx0, diff1 * gx1) - diff_sum * mean_gx;
            float ry = patch_sum(diff0 * gy0, diff1 * gy1) - diff_sum * mean_gy;
            u -= (gyy * rx - gxy * ry) / det;
            v -= (gxx * ry - gxy * rx) / det;
            if (ssd >= prev_ssd) {
                break;
            }
            prev_ssd = ssd;
        }
        // reject a refinement that moved further than the patch size
        if (hypot2(u - best_u, v - best_v) > 8) {
            u = best_u;
            v = best_v;
        }
        last_u = u;
        last_v = v;
        if (thread_index == 0) {
            vectors[patch_y * patches_x + patch_x] = u;
            vectors[patches_x * patches_y + patch_y * patches_x + patch_x] = v;
        }
    }
}

// densify_flow: dense flow of one level from the overlapping patch vectors, each weighted by
// 1 / max(1, photometric error) (flowdenoise.cu). Grid (ceil(width * height / 256), neighbours); the neighbours'
// levels come as pointers.
struct DensifyArgs {
    in_plane current;
    in_plane neighbours[6];
    in_plane vectors;
    out_plane flow;
    int width, height, patches_x, patches_y;
};
kernel void densify_flow(constant DensifyArgs& args [[buffer(0)]], KERNEL_INDICES) {
    in_plane current = args.current;
    in_plane neighbour = args.neighbours[blockIdx.y];
    in_plane vectors = args.vectors;
    out_plane flow = args.flow;
    const int width = args.width;
    const int height = args.height;
    const int patches_x = args.patches_x;
    const int patches_y = args.patches_y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) {
        return;
    }
    int x = i % width;
    int y = i / width;
    vectors += blockIdx.y * 2 * patches_x * patches_y;
    flow += blockIdx.y * 2 * width * height;
    float sum_u = 0;
    float sum_v = 0;
    float sum_w = 0;
    // patches with a top-left corner in [x - 7, x] x [y - 7, y], clamped to the patch grid
    for (int py = min_int(patches_y - 1, max_int(0, (y - 7 + 3) / 4)); py <= min_int(patches_y - 1, y / 4); py++) {
        for (int px = min_int(patches_x - 1, max_int(0, (x - 7 + 3) / 4)); px <= min_int(patches_x - 1, x / 4); px++) {
            float u = vectors[py * patches_x + px];
            float v = vectors[patches_x * patches_y + py * patches_x + px];
            float w = 1.f / fmax(1.f, fabs(sample_bilinear(neighbour, x + u, y + v, width, height) - current[i]));
            sum_u += w * u;
            sum_v += w * v;
            sum_w += w;
        }
    }
    flow[i] = sum_u / fmax(sum_w, 1e-6f);
    flow[width * height + i] = sum_v / fmax(sum_w, 1e-6f);
}

// ------------------------------------------------------------------------------------------------ Lanczos-3 warp

// Lanczos-3 kernel sin(pi d) sin(pi d / 3) / (pi^2 d^2 / 3); sin(pi x) instead of CUDA's sinpif (fast math).
static inline float lanczos3(float offset) {
    offset = fabs(offset);
    if (offset < 1e-6f) {
        return 1;
    }
    if (offset >= 3) {
        return 0;
    }
    return sin(M_PI_F * offset) * sin(M_PI_F * offset / 3) / (3.289868133696453f * offset * offset);
}
// separable Lanczos-3 sample, 6 x 6 taps, normalised weights, edges clamped (flowdenoise.cu)
static inline float sample_lanczos3(in_plane plane, float x, float y, int width, int height) {
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    float wx[6];
    float wy[6];
    float sum_wx = 0;
    float sum_wy = 0;
#pragma unroll
    for (int k = 0; k < 6; k++) {
        wx[k] = lanczos3(x - x0 - k + 2);
        wy[k] = lanczos3(y - y0 - k + 2);
        sum_wx += wx[k];
        sum_wy += wy[k];
    }
    float sum = 0;
#pragma unroll
    for (int j = 0; j < 6; j++) {
        float row = 0;
#pragma unroll
        for (int k = 0; k < 6; k++) {
            row += wx[k] * sample_clamped(plane, x0 + k - 2, y0 + j - 2, width, height);
        }
        sum += wy[j] * row;
    }
    return sum / (sum_wx * sum_wy);
}
// sample_lanczos3 of the U and V planes at once (V at planes + width * height)
static inline float2 sample_lanczos3_pair(in_plane planes, float x, float y, int width, int height) {
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    float wx[6];
    float wy[6];
    float sum_wx = 0;
    float sum_wy = 0;
#pragma unroll
    for (int k = 0; k < 6; k++) {
        wx[k] = lanczos3(x - x0 - k + 2);
        wy[k] = lanczos3(y - y0 - k + 2);
        sum_wx += wx[k];
        sum_wy += wy[k];
    }
    float sum_u = 0;
    float sum_v = 0;
#pragma unroll
    for (int j = 0; j < 6; j++) {
        float row_u = 0;
        float row_v = 0;
#pragma unroll
        for (int k = 0; k < 6; k++) {
            int p = clamp_index(y0 + j - 2, height) * width + clamp_index(x0 + k - 2, width);
            row_u += wx[k] * planes[p];
            row_v += wx[k] * planes[width * height + p];
        }
        sum_u += wy[j] * row_u;
        sum_v += wy[j] * row_v;
    }
    return float2(sum_u / (sum_wx * sum_wy), sum_v / (sum_wx * sum_wy));
}

// warp_luma: one neighbour's luma aligned to the current frame, |warped - current| and the flow speed
// (flowdenoise.cu). One thread per luma pixel.
struct WarpLumaArgs {
    in_plane neighbour;
    in_plane current;
    in_plane flow;
    out_plane warped;
    out_plane abs_diff;
    out_plane speed;
    int width, height, flow_w, flow_h;
    float scale;
};
kernel void warp_luma(constant WarpLumaArgs& args [[buffer(0)]], KERNEL_INDICES) {
    in_plane neighbour = args.neighbour;
    in_plane current = args.current;
    in_plane flow = args.flow;
    out_plane warped = args.warped;
    out_plane abs_diff = args.abs_diff;
    out_plane speed = args.speed;
    const int width = args.width;
    const int height = args.height;
    const int flow_w = args.flow_w;
    const int flow_h = args.flow_h;
    const float scale = args.scale;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) {
        return;
    }
    int x = i % width;
    int y = i / width;
    float fx = (x + .5f) / scale - .5f;
    float fy = (y + .5f) / scale - .5f;
    float u = sample_bilinear(flow, fx, fy, flow_w, flow_h) * scale;
    float v = sample_bilinear(flow + flow_w * flow_h, fx, fy, flow_w, flow_h) * scale;
    float z = sample_lanczos3(neighbour, x + u, y + v, width, height);
    warped[i] = z;
    abs_diff[i] = fabs(z - current[i]);
    speed[i] = hypot2(u, v);
}

// ------------------------------------------------------------------------------------------------ weights, fusion

// temporal_weight: MVTools DegrainWeight on the block SAD with a motion-dependent threshold, times the spike
// rejection factor; also the neighbour's fine detail (flowdenoise.cu). One thread per luma pixel.
struct TemporalWeightArgs {
    in_plane warped;
    in_plane warped_low;
    in_plane sad;
    in_plane abs_diff;
    in_plane abs_diff_mean;
    in_plane speed;
    out_plane weight;
    out_plane warped_high;
    int width, height;
    float thsad, frame_distance, fast_th, spike;
};
kernel void temporal_weight(constant TemporalWeightArgs& args [[buffer(0)]], KERNEL_INDICES) {
    in_plane warped = args.warped;
    in_plane warped_low = args.warped_low;
    in_plane sad = args.sad;
    in_plane abs_diff = args.abs_diff;
    in_plane abs_diff_mean = args.abs_diff_mean;
    in_plane speed = args.speed;
    out_plane weight = args.weight;
    out_plane warped_high = args.warped_high;
    const int width = args.width;
    const int height = args.height;
    const float thsad = args.thsad;
    const float frame_distance = args.frame_distance;
    const float fast_th = args.fast_th;
    const float spike = args.spike;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) {
        return;
    }
    float threshold = thsad / (1 + speed[i] / frame_distance / fast_th);
    float block_sad = sad[i] * 16;
    float w = block_sad < threshold
                  ? (threshold * threshold - block_sad * block_sad) / (threshold * threshold + block_sad * block_sad)
                  : 0;
    int x = i % width;
    int y = i / width;
    float largest = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            largest = fmax(largest, sample_clamped(abs_diff, x + dx, y + dy, width, height));
        }
    }
    float t = spike * 1.25f * abs_diff_mean[i];
    w *= saturate_unit((1.5f * t - largest) / fmax(.5f * t, .001f));
    weight[i] = w;
    warped_high[i] = warped[i] - warped_low[i];
}

// fuse_frames: the temporal sums of the luma, the current frame with weight 1 and each neighbour in order
// (flowdenoise.cu). One thread per luma pixel.
struct FuseFramesArgs {
    in_plane current;
    in_plane current_high;
    in_plane warped;
    in_plane weights;
    in_plane warped_high;
    out_plane weighted_sum;
    out_plane weight_sum;
    out_plane weight_sq_sum;
    out_plane high_sum;
    out_plane high_abs_sum;
    int count, neighbour_count;
};
kernel void fuse_frames(constant FuseFramesArgs& args [[buffer(0)]], KERNEL_INDICES) {
    in_plane current = args.current;
    in_plane current_high = args.current_high;
    in_plane warped = args.warped;
    in_plane weights = args.weights;
    in_plane warped_high = args.warped_high;
    out_plane weighted_sum = args.weighted_sum;
    out_plane weight_sum = args.weight_sum;
    out_plane weight_sq_sum = args.weight_sq_sum;
    out_plane high_sum = args.high_sum;
    out_plane high_abs_sum = args.high_abs_sum;
    const int count = args.count;
    const int neighbour_count = args.neighbour_count;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    float acc = current[i];
    float w_sum = 1;
    float w_sq_sum = 1;
    float h_sum = current_high[i];
    float h_abs_sum = fabs(h_sum);
    for (int d = 0; d < neighbour_count; d++) {
        int j = d * count + i;
        float w = weights[j];
        float h = warped_high[j];
        acc += w * warped[j];
        w_sum += w;
        w_sq_sum += w * w;
        h_sum += w * h;
        h_abs_sum += w * fabs(h);
    }
    weighted_sum[i] = acc;
    weight_sum[i] = w_sum;
    weight_sq_sum[i] = w_sq_sum;
    high_sum[i] = h_sum;
    high_abs_sum[i] = h_abs_sum;
}

// warp_accumulate_chroma: one neighbour's chroma aligned with the 2x2-averaged luma flow and weights, added to the
// chroma sums (flowdenoise.cu). One thread per chroma pixel; width x height is the chroma size.
struct WarpChromaArgs {
    in_plane neighbour;
    in_plane flow;
    in_plane weight;
    out_plane sums;
    out_plane weight_sum;
    int width, height, flow_w, flow_h;
    float scale;
};
kernel void warp_accumulate_chroma(constant WarpChromaArgs& args [[buffer(0)]], KERNEL_INDICES) {
    in_plane neighbour = args.neighbour;
    in_plane flow = args.flow;
    in_plane weight = args.weight;
    out_plane sums = args.sums;
    out_plane weight_sum = args.weight_sum;
    const int width = args.width;
    const int height = args.height;
    const int flow_w = args.flow_w;
    const int flow_h = args.flow_h;
    const float scale = args.scale;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) {
        return;
    }
    int x = i % width;
    int y = i / width;
    float u = 0;
    float v = 0;
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            float fx = (2 * x + dx + .5f) / scale - .5f;
            float fy = (2 * y + dy + .5f) / scale - .5f;
            u += sample_bilinear(flow, fx, fy, flow_w, flow_h) * scale * .125f;
            v += sample_bilinear(flow + flow_w * flow_h, fx, fy, flow_w, flow_h) * scale * .125f;
        }
    }
    // luma index of (2x, 2y); the luma plane is 2 * width wide
    int p = 2 * y * 2 * width + 2 * x;
    float w = (weight[p] + weight[p + 1] + weight[p + 2 * width] + weight[p + 2 * width + 1]) * .25f;
    float2 z = sample_lanczos3_pair(neighbour, x + u, y + v, width, height);
    sums[i] += w * z.x;
    sums[width * height + i] += w * z.y;
    weight_sum[i] += w;
}
