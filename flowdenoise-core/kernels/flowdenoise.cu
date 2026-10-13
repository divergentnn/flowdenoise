// flowdenoise.cu: the image kernels of the denoiser (separable filters, motion pyramid, motion search, flow
// densification, Lanczos-3 warps, temporal weights, fusion). kernels/README.md shows how they fit together.
//
// Compiled WITH fast math (nvcc --use_fast_math), as the Python reference compiles them (CuPy RawModule,
// options=("--use_fast_math",)). The output is bit-identical to the reference only because every operation, and the
// order of every sum, is the reference's: do not reorder arithmetic, split or merge expressions, or change the
// unrolled loops (README.md, "Numerical rules"). flowdenoise_ops.cu holds the kernels compiled without fast math.
//
// Conventions for every kernel in this file:
// - An image plane is width x height float32 samples, row-major, no row padding. Several planes of the same size are
//   stored back to back (plane p starts at p * width * height).
// - A flow (motion vector) field is two planes: horizontal components u, then vertical components v, in pixels of
//   the plane it belongs to; a vector (u, v) at (x, y) points to (x + u, y + v) in the neighbour frame.
// - Reads outside a plane are clamped to the nearest edge sample (replicate padding) unless stated otherwise.
// - Luma holds 10-bit codes (0 ... 1023) as floats, except in the motion pyramid, which holds 8-bit codes.
// - "One thread per pixel" kernels are launched as 1-D grids of 256-thread blocks over the plane's pixels.
#include "portable.h"

// ------------------------------------------------------------------------------------------------ sampling helpers

// Index clamped into [0, size - 1].
__device__ __forceinline__ int clamp_index(int index, int size) {
    return max(0, min(index, size - 1));
}

// Value clamped into [0, 1].
__device__ __forceinline__ float saturate(float value) {
    return fminf(1.f, fmaxf(0.f, value));
}

// Sample (x, y) of a width x height plane, clamped to the edges.
__device__ float sample_clamped(const float* plane, int x, int y, int width, int height) {
    return plane[clamp_index(y, height) * width + clamp_index(x, width)];
}

// Bilinear sample at the fractional position (x, y), edges clamped: the 2x2 samples around (floor(x), floor(y)),
// blended along x, then along y.
__device__ float sample_bilinear(const float* plane, float x, float y, int width, int height) {
    int x0 = floorf(x);
    int y0 = floorf(y);
    float fx = x - x0;
    float fy = y - y0;
    return (1 - fy) * ((1 - fx) * sample_clamped(plane, x0, y0, width, height) +
                       fx * sample_clamped(plane, x0 + 1, y0, width, height)) +
           fy * ((1 - fx) * sample_clamped(plane, x0, y0 + 1, width, height) +
                 fx * sample_clamped(plane, x0 + 1, y0 + 1, width, height));
}

// ------------------------------------------------------------------------------------------------ separable filters

// Filter axis of the separable filter kernels.
enum {
    AXIS_X = 0,
    AXIS_Y = 1
};

// separable_filter_tile<TAPS, AXIS>: one pass of a separable FIR filter, with the input tile in shared memory.
//
// Computes dst(x, y) = sum_j taps[j] * src(x + j - TAPS / 2, y) for AXIS_X (or along y for AXIS_Y): a correlation
// with the taps centred on the output sample; with an even TAPS (the 8-tap box of the SAD) the window covers
// x - TAPS / 2 ... x + TAPS / 2 - 1.
//
// Inputs:  src, planes of width x height; taps, TAPS floats in device memory.
//          zero_pad: 1 = samples outside the plane are 0 (the SAD box), 0 = clamped to the edge.
//          prefilter_edges: with TAPS 5, turn the binomial [1 4 6 4 1] / 16 into the exact result of two successive
//          edge-clamped [1 2 1] / 4 passes (see below).
// Output:  dst, the same planes filtered along one axis.
// Mapping: grid (ceil(width / 32), ceil(height / 8), planes), blocks of 32 x 8 threads; each block filters a 32 x 8
//          output tile of plane blockIdx.z. The 256 threads first load the tile plus an apron of TAPS - 1 samples
//          along the filtered axis into shared memory, then each thread computes one output sample.
// Numerics: odd TAPS are assumed symmetric: the sum starts at the centre tap and adds the mirrored pairs from the
//          inside out, taps[c + j] * (left + right), which is how the reference sums them (not tap by tap).
template <int TAPS, int AXIS>
__device__ void separable_filter_tile(const float* src, float* dst, int width, int height, const float* taps,
                                      int zero_pad, int prefilter_edges) {
    // tile with its apron; consecutive samples along the filtered axis are `tap_stride` apart in shared memory
    constexpr int TILE_W = 32 + (AXIS == AXIS_X ? TAPS - 1 : 0);
    constexpr int TILE_H = 8 + (AXIS == AXIS_Y ? TAPS - 1 : 0);
    constexpr int tap_stride = AXIS == AXIS_X ? 1 : TILE_W;
    __shared__ float tile[TILE_W * TILE_H];

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
    __syncthreads();

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
    // The reference prefilters with [1 2 1] / 4 twice, each pass clamped at the edges. That equals one edge-clamped
    // binomial [1 4 6 4 1] / 16 pass except at the first and last sample, which differ by (next - this) / 16 and
    // (previous - this) / 16.
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

// Entry points separable_filter_<TAPS>_x / _y for the tap counts the denoiser uses: 3 (chroma box), 5 (prefilter),
// 7 (sigma-1 Gaussian, luma box, motion pre-blur), 8 (SAD box), 19 (sigma-3 Gaussian), 37 (sigma-6 Gaussian).
#define SEPARABLE_FILTER(TAPS)                                                                                     \
    extern "C" __global__ void separable_filter_##TAPS##_x(const float* src, float* dst, int width, int height,    \
                                                           const float* taps, int zero_pad, int prefilter_edges) { \
        separable_filter_tile<TAPS, AXIS_X>(src, dst, width, height, taps, zero_pad, prefilter_edges);             \
    }                                                                                                              \
    extern "C" __global__ void separable_filter_##TAPS##_y(const float* src, float* dst, int width, int height,    \
                                                           const float* taps, int zero_pad, int prefilter_edges) { \
        separable_filter_tile<TAPS, AXIS_Y>(src, dst, width, height, taps, zero_pad, prefilter_edges);             \
    }
SEPARABLE_FILTER(3)
SEPARABLE_FILTER(5)
SEPARABLE_FILTER(7)
SEPARABLE_FILTER(8)
SEPARABLE_FILTER(19)
SEPARABLE_FILTER(37)

// separable_filter_any: one pass of a separable FIR filter with any number of taps (the tiled kernels above cover
// the tap counts of the presets; this one serves other parameters, e.g. a guided-filter radius of 4).
//
// Computes dst(x, y) = sum_j taps[j] * src(x + j - tap_count / 2, y) along `axis` (AXIS_X or AXIS_Y), the taps
// summed in order j = 0 ... tap_count - 1.
// Inputs:  src, planes of width x height (count = width * height * planes samples); taps, tap_count floats.
//          zero_pad: 1 = samples outside the plane contribute nothing, 0 = clamped to the edge.
// Output:  dst, same layout.
// Mapping: one thread per output sample, 1-D grid over all planes.
extern "C" __global__ void separable_filter_any(const float* src, float* dst, int width, int height, int count,
                                                const float* taps, int tap_count, int axis, int zero_pad) {
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

// downsample_2x2: halves a plane by averaging each 2x2 block (an odd last row / column is dropped).
//
// Input:   src, width x height.
// Output:  dst, (width / 2) x (height / 2): (a + b + c + d) * 0.25.
// Mapping: one thread per output sample.
// Used for the motion pyramid, for the motion input at half resolution (4K) and for the chroma fallback weights.
extern "C" __global__ void downsample_2x2(const float* src, float* dst, int width, int height) {
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

// Sum of `value` over the 64 threads of a block (one 8x8 patch), returned to every thread: a shuffle-down tree
// within each 32-lane warp (offsets 16, 8, 4, 2, 1), then warp 0's sum plus warp 1's. `warp_sums` is two floats of
// shared memory. Every thread of the block must call it (it synchronizes).
__device__ float patch_sum(float value, float* warp_sums) {
    int thread_index = threadIdx.x;
    for (int offset = 16; offset; offset /= 2) {
        value += FLOWDENOISE_SHFL_DOWN(value, offset);
    }
    if (!(thread_index & 31)) {
        warp_sums[thread_index / 32] = value;
    }
    __syncthreads();
    float sum = warp_sums[0] + warp_sums[1];
    __syncthreads();
    return sum;
}

// motion_search_sweep: patch motion search on one pyramid level, one sweep along each row of patches.
//
// Algorithm: DIS-style inverse search (Kroeger et al., "Fast Optical Flow using Dense Inverse Search", ECCV 2016).
// The current frame is covered by overlapping 8x8 patches with a stride of 4 pixels. For each patch, the best of a
// few candidate vectors is refined by inverse-compositional Gauss-Newton steps on the mean-normalised SSD between the
// patch and the bilinearly warped neighbour. The host runs two sweeps per level: forward (left to right, propagating
// each patch's result to its right-hand neighbour) and reverse (right to left, also trying the forward sweep's
// vector of the patch below). Patches of one row are visited in order by one block, so the propagation has no races.
//
// Inputs:  current, the current frame's pyramid level (width x height, 8-bit luma codes);
//          neighbours, the same level of each neighbour frame (planes back to back, one per blockIdx.y);
//          coarse_flow, per neighbour the dense flow of the next coarser level (2 planes of coarse_w x coarse_h), or
//          null at the coarsest level; used by the forward sweep only;
//          forward, per neighbour the forward sweep's patch vectors (2 planes of patches_x x patches_y), or null in
//          the forward sweep itself;
//          reverse: 0 = forward sweep, 1 = reverse sweep; iterations: Gauss-Newton steps per patch in this sweep.
// Output:  vectors, per neighbour 2 planes of patches_x x patches_y (u, v of each patch, in pixels of this level).
// Mapping: grid (patches_y, neighbours), 64 threads per block: the block walks along its row of patches; thread t
//          holds pixel (t % 8, t / 8) of the current patch, whose top-left pixel is (4 patch_x, 4 patch_y).
// Numerics: the gradients are 3x3 Sobel sums (not divided by 8); the structure tensor (gxx, gxy, gyy) is not
//          mean-normalised but the residual terms are; the Gauss-Newton update is applied before the convergence
//          test, so the returned vector is one step past the last SSD evaluated (as in the reference). A refined
//          vector more than 8 pixels from its starting candidate is rejected (DIS's patch-size rule).
extern "C" __global__ void motion_search_sweep(const float* current, const float* neighbours, int width, int height,
                                               const float* coarse_flow, int coarse_w, int coarse_h,
                                               const float* forward, float* vectors, int patches_x, int patches_y,
                                               int reverse, int iterations) {
    int patch_y = blockIdx.x;
    int thread_index = threadIdx.x;
    int neighbour = blockIdx.y;
    __shared__ float warp_sums[2];
    neighbours += neighbour * width * height;  // this block's neighbour frame
    vectors += neighbour * 2 * patches_x * patches_y;
    if (coarse_flow) {
        coarse_flow += neighbour * 2 * coarse_w * coarse_h;
    }
    if (forward) {
        forward += neighbour * 2 * patches_x * patches_y;
    }
    // result of the previous patch of this sweep (spatial propagation along the row)
    float last_u = 0;
    float last_v = 0;
    for (int col = 0; col < patches_x; col++) {
        int patch_x = reverse ? patches_x - 1 - col : col;
        int px = patch_x * 4 + thread_index % 8;
        int py = patch_y * 4 + thread_index / 8;

        // template pixel and its Sobel gradient
        float c = sample_clamped(current, px, py, width, height);
        float gx = 0;
        float gy = 0;
        for (int j = -1; j <= 1; j++) {
            float sobel_w = j == 0 ? 2.f : 1.f;
            gx += sobel_w * (sample_clamped(current, px + 1, py + j, width, height) -
                             sample_clamped(current, px - 1, py + j, width, height));
            gy += sobel_w * (sample_clamped(current, px + j, py + 1, width, height) -
                             sample_clamped(current, px + j, py - 1, width, height));
        }
        // structure tensor of the patch and its mean gradient
        float gxx = patch_sum(gx * gx, warp_sums);
        float gyy = patch_sum(gy * gy, warp_sums);
        float gxy = patch_sum(gx * gy, warp_sums);
        float mean_gx = patch_sum(gx, warp_sums) / 64;
        float mean_gy = patch_sum(gy, warp_sums) / 64;

        // initial vector: the forward sweep's result for this patch, else the coarser level's dense flow at the
        // patch centre (4 patch_x + 4, 4 patch_y + 4) mapped to coarser pixel coordinates (x + 0.5) / 2 - 0.5 and
        // scaled by 2, else zero
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

        // candidates: 0 the initial vector, 1 the previous patch's result in this sweep, 2 (reverse sweep) the
        // forward sweep's vector of the patch below; keep the one with the lowest mean-normalised SSD
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
            float diff = sample_bilinear(neighbours, px + cand_u, py + cand_v, width, height) - c;
            float diff_sum = patch_sum(diff, warp_sums);
            float ssd = patch_sum(diff * diff, warp_sums) - diff_sum * diff_sum / 64;
            if (ssd < best_ssd) {
                best_ssd = ssd;
                best_u = cand_u;
                best_v = cand_v;
            }
        }

        // inverse-compositional Gauss-Newton refinement: solve [gxx gxy; gxy gyy] (du, dv) = (rx, ry) with the
        // mean-normalised residual-gradient sums (rx, ry); stop once the SSD no longer falls
        u = best_u;
        v = best_v;
        float det = fmaxf(gxx * gyy - gxy * gxy, .001f);
        float prev_ssd = 1e30f;
        for (int it = 0; it < iterations; it++) {
            float diff = sample_bilinear(neighbours, px + u, py + v, width, height) - c;
            float diff_sum = patch_sum(diff, warp_sums);
            float ssd = patch_sum(diff * diff, warp_sums) - diff_sum * diff_sum / 64;
            float rx = patch_sum(diff * gx, warp_sums) - diff_sum * mean_gx;
            float ry = patch_sum(diff * gy, warp_sums) - diff_sum * mean_gy;
            u -= (gyy * rx - gxy * ry) / det;
            v -= (gxx * ry - gxy * rx) / det;
            if (ssd >= prev_ssd) {
                break;
            }
            prev_ssd = ssd;
        }
        // reject a refinement that moved further than the patch size
        if (hypotf(u - best_u, v - best_v) > 8) {
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

// densify_flow: dense flow of one pyramid level from the overlapping patch vectors (DIS densification).
//
// Each pixel averages the vectors of all patches that cover it (top-left corner within 7 pixels above / left, on
// the stride-4 grid, clamped to the grid), each weighted by 1 / max(1, |neighbour(x + u, y + v) - current(x, y)|):
// vectors that explain this pixel's intensity count more.
// Inputs:  current, neighbours (as motion_search_sweep); vectors, the reverse sweep's patch vectors per neighbour.
// Output:  flow, per neighbour 2 planes of width x height.
// Mapping: grid (ceil(width * height / 256), neighbours), one thread per pixel.
extern "C" __global__ void densify_flow(const float* current, const float* neighbours, const float* vectors,
                                        float* flow, int width, int height, int patches_x, int patches_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) {
        return;
    }
    int x = i % width;
    int y = i / width;
    neighbours += blockIdx.y * width * height;
    vectors += blockIdx.y * 2 * patches_x * patches_y;
    flow += blockIdx.y * 2 * width * height;
    float sum_u = 0;
    float sum_v = 0;
    float sum_w = 0;
    // patches with a top-left corner in [x - 7, x] x [y - 7, y], clamped to the patch grid: (x - 7 + 3) / 4 rounds
    // (x - 7) / 4 up for x >= 7 and gives 0 or -1 (clamped to 0) below. (The bounds stay in the loop headers: hoisting
    // them changes the generated code.)
    for (int py = min(patches_y - 1, max(0, (y - 7 + 3) / 4)); py <= min(patches_y - 1, y / 4); py++) {
        for (int px = min(patches_x - 1, max(0, (x - 7 + 3) / 4)); px <= min(patches_x - 1, x / 4); px++) {
            float u = vectors[py * patches_x + px];
            float v = vectors[patches_x * patches_y + py * patches_x + px];
            float w = 1.f / fmaxf(1.f, fabsf(sample_bilinear(neighbours, x + u, y + v, width, height) - current[i]));
            sum_u += w * u;
            sum_v += w * v;
            sum_w += w;
        }
    }
    flow[i] = sum_u / fmaxf(sum_w, 1e-6f);
    flow[width * height + i] = sum_v / fmaxf(sum_w, 1e-6f);
}

// ------------------------------------------------------------------------------------------------ Lanczos-3 warp

// Lanczos-3 kernel: sinc(d) sinc(d / 3) = sin(pi d) sin(pi d / 3) / (pi^2 d^2 / 3); 1 at d = 0, 0 for |d| >= 3.
// 3.289868133696453 = pi^2 / 3.
__device__ float lanczos3(float offset) {
    offset = fabsf(offset);
    if (offset < 1e-6f) {
        return 1;
    }
    if (offset >= 3) {
        return 0;
    }
    return sinpif(offset) * sinpif(offset / 3) / (3.289868133696453f * offset * offset);
}

// Separable Lanczos-3 sample of a plane at the fractional position (x, y), edges clamped: 6 x 6 samples
// x0 - 2 ... x0 + 3 (x0 = floor(x)), each row filtered along x first, the rows then along y; the weights are
// normalised by their sums, so a constant plane stays constant.
__device__ float sample_lanczos3(const float* plane, float x, float y, int width, int height) {
    int x0 = floorf(x);
    int y0 = floorf(y);
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

// sample_lanczos3 of two planes at once (U at `planes`, V at planes + width * height), sharing the weights.
__device__ float2 sample_lanczos3_pair(const float* planes, float x, float y, int width, int height) {
    int x0 = floorf(x);
    int y0 = floorf(y);
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
    return make_float2(sum_u / (sum_wx * sum_wy), sum_v / (sum_wx * sum_wy));
}

// warp_luma: aligns one neighbour's luma to the current frame.
//
// The flow (computed at 1 / scale of the luma resolution) is sampled bilinearly at the luma pixel's centre,
// ((x + 0.5) / scale - 0.5, (y + 0.5) / scale - 0.5), and scaled to luma pixels; the neighbour is then sampled
// with Lanczos-3 at (x + u, y + v).
// Inputs:  neighbour, current: luma planes (width x height); flow: 2 planes of flow_w x flow_h.
// Outputs: warped, the aligned neighbour; abs_diff = |warped - current|; speed = |(u, v)| in luma pixels.
// Mapping: one thread per luma pixel.
extern "C" __global__ void warp_luma(const float* neighbour, const float* current, const float* flow, float* warped,
                                     float* abs_diff, float* speed, int width, int height, int flow_w, int flow_h,
                                     float scale) {
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
    abs_diff[i] = fabsf(z - current[i]);
    speed[i] = hypotf(u, v);
}

// ------------------------------------------------------------------------------------------------ weights, fusion

// temporal_weight: the per-pixel weight of one aligned neighbour.
//
// 1. MVTools' DegrainWeight on the block SAD s: w = (T^2 - s^2) / (T^2 + s^2) for s < T, else 0. s is the 8x8 box
//    mean of |prefiltered warped - prefiltered current| (10-bit codes) times 16, i.e. a SAD over an 8x8 block in
//    8-bit units (x 64 / 4), MVTools' scale for thSAD. T = thsad / (1 + speed / frame_distance / fast_th): the
//    threshold falls as the motion per frame of temporal distance grows.
// 2. Spike rejection: with t = spike * 1.25 * abs_diff_mean (the Gaussian sigma-6 mean of |warped - current|) and
//    m the largest |warped - current| in the 3x3 window, w *= clamp((1.5 t - m) / max(0.5 t, 0.001), 0, 1): full
//    weight up to m = t, none from m = 1.5 t.
// Inputs:  warped, warped_low (its Gaussian sigma 1), sad, abs_diff, abs_diff_mean, speed: planes of width x height;
//          thsad, frame_distance (|temporal offset|, frames), fast_th, spike: scalars.
// Outputs: weight; warped_high = warped - warped_low (the neighbour's fine detail, for the detail gate).
// Mapping: one thread per luma pixel. The weights of all neighbours are fused in one pass afterwards (fuse_frames).
extern "C" __global__ void temporal_weight(const float* warped, const float* warped_low, const float* sad,
                                           const float* abs_diff, const float* abs_diff_mean, const float* speed,
                                           float* weight, float* warped_high, int width, int height, float thsad,
                                           float frame_distance, float fast_th, float spike) {
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
            largest = fmaxf(largest, sample_clamped(abs_diff, x + dx, y + dy, width, height));
        }
    }
    float t = spike * 1.25f * abs_diff_mean[i];
    w *= saturate((1.5f * t - largest) / fmaxf(.5f * t, .001f));
    weight[i] = w;
    warped_high[i] = warped[i] - warped_low[i];
}

// fuse_frames: the temporal sums of the luma, the current frame with weight 1 and each neighbour in order.
//
// Inputs:  current, current_high (current - its Gaussian sigma 1): planes of `count` pixels;
//          warped, weights, warped_high: neighbour_count planes each (neighbour d at d * count).
// Outputs: weighted_sum = current + sum w warped        (the fused luma is weighted_sum / weight_sum)
//          weight_sum = 1 + sum w,  weight_sq_sum = 1 + sum w^2   (effective frame count weight_sum^2 / weight_sq_sum)
//          high_sum = current_high + sum w warped_high,  high_abs_sum = |current_high| + sum w |warped_high|
//          (their ratio measures how coherent the fine detail is across the frames: the detail gate).
// Mapping: one thread per luma pixel.
extern "C" __global__ void fuse_frames(const float* current, const float* current_high, const float* warped,
                                       const float* weights, const float* warped_high, float* weighted_sum,
                                       float* weight_sum, float* weight_sq_sum, float* high_sum, float* high_abs_sum,
                                       int count, int neighbour_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    float acc = current[i];
    float w_sum = 1;
    float w_sq_sum = 1;
    float h_sum = current_high[i];
    float h_abs_sum = fabsf(h_sum);
    for (int d = 0; d < neighbour_count; d++) {
        int j = d * count + i;
        float w = weights[j];
        float h = warped_high[j];
        acc += w * warped[j];
        w_sum += w;
        w_sq_sum += w * w;
        h_sum += w * h;
        h_abs_sum += w * fabsf(h);
    }
    weighted_sum[i] = acc;
    weight_sum[i] = w_sum;
    weight_sq_sum[i] = w_sq_sum;
    high_sum[i] = h_sum;
    high_abs_sum[i] = h_abs_sum;
}

// warp_accumulate_chroma: aligns one neighbour's chroma and adds it to the chroma sums.
//
// The chroma vector of a chroma pixel is the mean of the luma-resolution flow over its 2x2 luma footprint, halved:
// each of the four bilinear flow samples is scaled by scale * 0.125 (= scale / 4 / 2). (Sampling the flow once at
// the chroma pixel's centre would be identical except at the clamped outermost half pixel.) Its weight is the mean
// of the four luma weights. Both chroma planes are sampled with Lanczos-3 at (x + u, y + v).
// Inputs:  neighbour, the neighbour's U and V planes (width x height each: the CHROMA size); flow, 2 planes of
//          flow_w x flow_h; weight, the luma weights (2 width x 2 height).
// In/out:  sums (U then V) += w * neighbour chroma; weight_sum += w (started at the current frame and 1).
// Mapping: one thread per chroma pixel; neighbours are accumulated in launch order.
extern "C" __global__ void warp_accumulate_chroma(const float* neighbour, const float* flow, const float* weight,
                                                  float* sums, float* weight_sum, int width, int height, int flow_w,
                                                  int flow_h, float scale) {
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
