// flowdenoise elementwise and order-statistic kernels (compiled WITHOUT fast math): the CuPy elementwise operations of
// the Python reference and a radix select for its median / quantile. Plain CUDA C++; hipcc compiles it too.
//
// Moved verbatim from GRAIL (kernels/grail.cu).
#include "portable.h"

#define IDX                                                     \
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= n) return;

// ------------------------------------------------------------------------------------------------ conversions
extern "C" __global__ void u16_to_f32(const uint16_t* a, float* b, int64_t n) { IDX b[i] = (float)a[i]; }

// ------------------------------------------------------------------------------------------------ flowdegrain2 (CuPy)
extern "C" __global__ void ew_absdiff(const float* a, const float* b, float* c, int64_t n) { IDX c[i] = fabsf(a[i] - b[i]); }
extern "C" __global__ void ew_energy(const float* a, const float* b, float* c, int64_t n) {
    IDX float d = a[i] - b[i]; c[i] = d * d;
}
extern "C" __global__ void ew_sub(const float* a, const float* b, float* c, int64_t n) { IDX c[i] = a[i] - b[i]; }
extern "C" __global__ void ew_sqr(const float* a, float* c, int64_t n) { IDX c[i] = a[i] * a[i]; }
extern "C" __global__ void ew_abs(const float* a, float* c, int64_t n) { IDX c[i] = fabsf(a[i]); }
// c = a / b[i % m]  (oy = acc / ws; oc = ac / wc with wc broadcast over both chroma planes)
extern "C" __global__ void ew_div_bc(const float* a, const float* b, float* c, int64_t n, int64_t m) {
    IDX c[i] = a[i] / b[i % m];
}
// motion pyramid base: clip(rint(g / 4), 0, 255)
extern "C" __global__ void ew_q8(const float* g, float* o, int64_t n) {
    IDX float v = rintf(g[i] / 4.f); o[i] = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
}
extern "C" __global__ void ew_rint(const float* a, float* o, int64_t n) { IDX o[i] = rintf(a[i]); }
// GUIDE: v = max(m2 - m m, 0); a = v / (v + eps); b = m - a m   (eps per plane: eps[i / ps])
extern "C" __global__ void ew_guide(const float* m, const float* m2, const float* eps, float* a, float* b, int64_t n,
                                    int64_t ps) {
    IDX float e = eps[i / ps], mm = m[i];
    float v = fmaxf(m2[i] - mm * mm, 0.f);
    float aa = v / (v + e);
    a[i] = aa; b[i] = mm - aa * mm;
}
// ALPHA: a = sat((fallback - ws^2 / sq) / (fallback - 1))
extern "C" __global__ void ew_alpha(const float* ws, const float* sq, float fallback, float* a, int64_t n) {
    IDX a[i] = fminf(1.f, fmaxf(0.f, (fallback - ws[i] * ws[i] / sq[i]) / (fallback - 1)));
}
// BLEND: y = x + a (ga x + gb - x), alpha broadcast (a[i % am])
extern "C" __global__ void ew_blend(const float* x, const float* a, const float* ga, const float* gb, float* y,
                                    int64_t n, int64_t am) {
    IDX float xx = x[i]; y[i] = xx + a[i % am] * (ga[i] * xx + gb[i] - xx);
}
// FINISH (detail gate) -> 10-bit luma
extern "C" __global__ void ew_finish(const float* y_, const float* lo_, const float* hc_, const float* hs_,
                                     const float* ha_, const float* sq_, const float* ws_, const float* e_,
                                     const float* efp, float c0, float boost, int16_t* out, int64_t n) {
    IDX float y = y_[i], lo = lo_[i], hc = hc_[i], hs = hs_[i], ha = ha_[i], sq = sq_[i], ws = ws_[i], e = e_[i];
    float ef = *efp;
    float cn = sqrtf(sq) / ws;
    float co = fminf(1.f, fmaxf(0.f, (hs / fmaxf(ha, 1e-6f) - cn) / fmaxf(1 - cn, .05f)));
    float tm = fminf(1.f, fmaxf(0.f, (e / fmaxf(ef, 1e-6f) - 3) / 3));
    float gate = fminf(1.f, fmaxf(0.f, (co - c0) / (1 - c0))) * tm;
    float z = lo + gate * (1 + (boost - 1) * tm) * hc + (1 - gate) * (y - lo);
    out[i] = (short)__float2int_rn(fminf(1023.f, fmaxf(0.f, z)));
}
extern "C" __global__ void ew_pack(const float* a, int16_t* b, int64_t n) {
    IDX b[i] = (short)__float2int_rn(fminf(1023.f, fmaxf(0.f, a[i])));
}

// ------------------------------------------------------------------------------------------------ order statistics
// Radix select on order-preserving uint32 keys of floats: 4 passes of 8 bits, state on the device (no host sync).
__device__ __forceinline__ uint32_t fkey(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float fkey_inv(uint32_t k) {
    uint32_t u = (k & 0x80000000u) ? (k & 0x7fffffffu) : ~k;
    return __uint_as_float(u);
}
// out[p * m + j] = |x[p * ps + j * stride]|  (hp[:, ::stride] of the reference, abs) or the value itself (absv = 0)
extern "C" __global__ void gather_stride(const float* x, float* out, int64_t ps, int64_t stride, int64_t m, int planes,
                                         int absv) {
    int64_t n = m * planes;
    IDX int64_t p = i / m, j = i % m;
    float v = x[p * ps + j * stride];
    out[i] = absv ? fabsf(v) : v;
}
// state: st[0] = key prefix, st[1] = rank still to find inside the prefix; hist: 256 counters (zeroed by sel_init)
extern "C" __global__ void sel_init(uint32_t* st, uint32_t* hist, uint32_t k) {
    int t = threadIdx.x;
    if (t == 0) { st[0] = 0; st[1] = k; }
    for (int b = t; b < 256; b += blockDim.x) hist[b] = 0;
}
extern "C" __global__ void sel_hist(const float* v, int64_t n, const uint32_t* st, uint32_t* hist, int shift) {
    __shared__ uint32_t h[256];
    for (int b = threadIdx.x; b < 256; b += blockDim.x) h[b] = 0;
    __syncthreads();
    uint32_t prefix = st[0];
    uint32_t maskhi = shift >= 24 ? 0u : (0xffffffffu << (shift + 8));
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        uint32_t k = fkey(v[i]);
        if ((k & maskhi) == (prefix & maskhi)) atomicAdd(&h[(k >> shift) & 255u], 1u);
    }
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += blockDim.x)
        if (h[b]) atomicAdd(&hist[b], h[b]);
}
extern "C" __global__ void sel_scan(uint32_t* st, uint32_t* hist, int shift) {
    if (threadIdx.x != 0) return;
    uint32_t k = st[1], c = 0;
    for (int b = 0; b < 256; b++) {
        uint32_t hb = hist[b];
        if (c + hb > k) { st[0] |= ((uint32_t)b << shift); st[1] = k - c; break; }
        c += hb;
    }
    for (int b = 0; b < 256; b++) hist[b] = 0;
}
extern "C" __global__ void sel_value(const uint32_t* st, float* out) {
    if (threadIdx.x == 0) *out = fkey_inv(st[0]);
}
// flowdegrain2 noise(): var = max((1.4826 med)^2, 0.001)
extern "C" __global__ void noise_var(const float* med, float* var, float mult) {
    if (threadIdx.x == 0) { float t = 1.4826f * *med; *var = fmaxf(t * t, 0.001f) * mult; }
}
// cupy quantile, method 'linear', float32 (cupy_percentile_weightnening): lo, hi = sorted values at floor(idx), +1
extern "C" __global__ void quantile_finish(const float* lo, const float* hi, float w, float* out) {
    if (threadIdx.x != 0) return;
    float a = *lo, b = *hi, diff = b - a;
    *out = w < 0.5f ? a + diff * w : b - diff * (1 - w);
}

