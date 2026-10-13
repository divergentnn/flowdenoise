// flowdenoise elementwise, copy and order-statistic kernels in Metal Shading Language (compiled WITHOUT fast math):
// flowdenoise_ops.cu mapped to MSL, plus the copy / fill kernels of the Metal driver. Each kernel takes its CUDA
// argument list as one constant struct (same order, C layout; device pointers inside it). Moved verbatim from GRAIL
// (kernels/metal/grail.metal).
#include <metal_stdlib>
using namespace metal;

#define KARGS uint3 blockIdx [[threadgroup_position_in_grid]], uint3 threadIdx [[thread_position_in_threadgroup]], \
              uint3 blockDim [[threads_per_threadgroup]], uint3 gridDim [[threadgroups_per_grid]], \
              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
#define SYNC() threadgroup_barrier(mem_flags::mem_threadgroup)
// 32-bit element index (every buffer here has fewer than 2^32 elements; 64-bit division is slow on the GPU)
#define IDX                                                     \
    uint i = blockIdx.x * blockDim.x + threadIdx.x;             \
    if (i >= (ulong)n) return;
typedef device const float* cfp;
typedef device float* fp;
typedef device const uchar* cbp;
typedef device uchar* bp;

static inline float mul_rn(float a, float b) { return fma(a, b, 0.0f); }

// ------------------------------------------------------------------------------------------------ copies (driver)
struct CopyArgs { device uint* d; device const uint* s; long n; };
kernel void copy_u32(constant CopyArgs& P [[buffer(0)]], KARGS) { long n = P.n; IDX P.d[i] = P.s[i]; }
struct CopyBArgs { bp d; cbp s; long n; };
kernel void copy_u8(constant CopyBArgs& P [[buffer(0)]], KARGS) { long n = P.n; IDX P.d[i] = P.s[i]; }
struct FillArgs { bp d; int v; long n; };
kernel void fill_u8(constant FillArgs& P [[buffer(0)]], KARGS) { long n = P.n; IDX P.d[i] = (uchar)P.v; }

// ------------------------------------------------------------------------------------------------ conversions
struct ConvArgs { device const ushort* a; fp b; long n; };
kernel void u16_to_f32(constant ConvArgs& P [[buffer(0)]], KARGS) { long n = P.n; IDX P.b[i] = (float)P.a[i]; }

// ------------------------------------------------------------------------------------------------ flowdegrain2 (CuPy)
struct EW2 { cfp a; cfp b; fp c; long n; };
struct EW1 { cfp a; fp c; long n; };
kernel void ew_absdiff(constant EW2& P [[buffer(0)]], KARGS) { long n = P.n; IDX P.c[i] = fabs(P.a[i] - P.b[i]); }
kernel void ew_energy(constant EW2& P [[buffer(0)]], KARGS) {
    long n = P.n; IDX float d = P.a[i] - P.b[i]; P.c[i] = d * d;
}
kernel void ew_sub(constant EW2& P [[buffer(0)]], KARGS) { long n = P.n; IDX P.c[i] = P.a[i] - P.b[i]; }
kernel void ew_sqr(constant EW1& P [[buffer(0)]], KARGS) { long n = P.n; IDX P.c[i] = P.a[i] * P.a[i]; }
kernel void ew_abs(constant EW1& P [[buffer(0)]], KARGS) { long n = P.n; IDX P.c[i] = fabs(P.a[i]); }
// c = a / b[i % m]
struct DivArgs { cfp a; cfp b; fp c; long n; long m; };
kernel void ew_div_bc(constant DivArgs& P [[buffer(0)]], KARGS) { long n = P.n; IDX P.c[i] = P.a[i] / P.b[i % (uint)P.m]; }
// motion pyramid base: clip(rint(g / 4), 0, 255)
kernel void ew_q8(constant EW1& P [[buffer(0)]], KARGS) {
    long n = P.n; IDX float v = rint(P.a[i] / 4.f); P.c[i] = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
}
kernel void ew_rint(constant EW1& P [[buffer(0)]], KARGS) { long n = P.n; IDX P.c[i] = rint(P.a[i]); }
// GUIDE: v = max(m2 - m m, 0); a = v / (v + eps); b = m - a m   (eps per plane: eps[i / ps])
struct GuideArgs { cfp m; cfp m2; cfp eps; fp a; fp b; long n; long ps; };
kernel void ew_guide(constant GuideArgs& P [[buffer(0)]], KARGS) {
    long n = P.n; IDX float e = P.eps[i / (uint)P.ps], mm = P.m[i];
    float v = fmax(P.m2[i] - mm * mm, 0.f);
    float aa = v / (v + e);
    P.a[i] = aa; P.b[i] = mm - aa * mm;
}
// ALPHA: a = sat((fallback - ws^2 / sq) / (fallback - 1))
struct AlphaArgs { cfp ws; cfp sq; float fallback; fp a; long n; };
kernel void ew_alpha(constant AlphaArgs& P [[buffer(0)]], KARGS) {
    long n = P.n; const float fb = P.fallback;
    IDX P.a[i] = fmin(1.f, fmax(0.f, (fb - P.ws[i] * P.ws[i] / P.sq[i]) / (fb - 1)));
}
// BLEND: y = x + a (ga x + gb - x), alpha broadcast (a[i % am])
struct BlendArgs { cfp x; cfp a; cfp ga; cfp gb; fp y; long n; long am; };
kernel void ew_blend(constant BlendArgs& P [[buffer(0)]], KARGS) {
    long n = P.n; IDX float xx = P.x[i]; P.y[i] = xx + P.a[i % (uint)P.am] * (P.ga[i] * xx + P.gb[i] - xx);
}
// FINISH (detail gate) -> 10-bit luma
struct FinishArgs { cfp y_; cfp lo_; cfp hc_; cfp hs_; cfp ha_; cfp sq_; cfp ws_; cfp e_; cfp efp; float c0, boost;
                    device short* out; long n; };
kernel void ew_finish(constant FinishArgs& P [[buffer(0)]], KARGS) {
    long n = P.n; const float c0 = P.c0, boost = P.boost;
    IDX float y = P.y_[i], lo = P.lo_[i], hc = P.hc_[i], hs = P.hs_[i], ha = P.ha_[i], sq = P.sq_[i], ws = P.ws_[i], e = P.e_[i];
    float ef = *P.efp;
    float cn = sqrt(sq) / ws;
    float co = fmin(1.f, fmax(0.f, (hs / fmax(ha, 1e-6f) - cn) / fmax(1 - cn, .05f)));
    float tm = fmin(1.f, fmax(0.f, (e / fmax(ef, 1e-6f) - 3) / 3));
    float gate = fmin(1.f, fmax(0.f, (co - c0) / (1 - c0))) * tm;
    float z = lo + gate * (1 + (boost - 1) * tm) * hc + (1 - gate) * (y - lo);
    P.out[i] = (short)(int)rint(fmin(1023.f, fmax(0.f, z)));
}
struct PackArgs { cfp a; device short* b; long n; };
kernel void ew_pack(constant PackArgs& P [[buffer(0)]], KARGS) {
    long n = P.n; IDX P.b[i] = (short)(int)rint(fmin(1023.f, fmax(0.f, P.a[i])));
}

// ------------------------------------------------------------------------------------------------ order statistics
// Radix select on order-preserving uint32 keys of floats: 4 passes of 8 bits, state on the device (no host sync).
static inline uint fkey(float f) {
    uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
static inline float fkey_inv(uint k) {
    uint u = (k & 0x80000000u) ? (k & 0x7fffffffu) : ~k;
    return as_type<float>(u);
}
// out[p * m + j] = |x[p * ps + j * stride]|  (hp[:, ::stride] of the reference, abs) or the value itself (absv = 0)
struct GatherArgs { cfp x; fp out; long ps; long stride; long m; int planes, absv; };
kernel void gather_stride(constant GatherArgs& P [[buffer(0)]], KARGS) {
    long n = P.m * P.planes;
    IDX uint m = (uint)P.m, p = i / m, j = i % m;
    float v = P.x[p * P.ps + j * P.stride];
    P.out[i] = P.absv ? fabs(v) : v;
}
struct SelInitArgs { device uint* st; device uint* hist; uint k; };
kernel void sel_init(constant SelInitArgs& P [[buffer(0)]], KARGS) {
    uint t = threadIdx.x;
    if (t == 0) { P.st[0] = 0; P.st[1] = P.k; }
    for (uint b = t; b < 256; b += blockDim.x) P.hist[b] = 0;
}
struct SelHistArgs { cfp v; long n; device const uint* st; device atomic_uint* hist; int shift; };
kernel void sel_hist(constant SelHistArgs& P [[buffer(0)]], KARGS) {
    threadgroup atomic_uint h[256];
    for (uint b = threadIdx.x; b < 256; b += blockDim.x) atomic_store_explicit(&h[b], 0u, memory_order_relaxed);
    SYNC();
    const int shift = P.shift;
    uint prefix = P.st[0];
    uint maskhi = shift >= 24 ? 0u : (0xffffffffu << (shift + 8));
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < P.n; i += (long)gridDim.x * blockDim.x) {
        uint k = fkey(P.v[i]);
        if ((k & maskhi) == (prefix & maskhi)) atomic_fetch_add_explicit(&h[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    SYNC();
    for (uint b = threadIdx.x; b < 256; b += blockDim.x) {
        uint c = atomic_load_explicit(&h[b], memory_order_relaxed);
        if (c) atomic_fetch_add_explicit(&P.hist[b], c, memory_order_relaxed);
    }
}
struct SelScanArgs { device uint* st; device uint* hist; int shift; };
kernel void sel_scan(constant SelScanArgs& P [[buffer(0)]], KARGS) {
    if (threadIdx.x != 0) return;
    uint k = P.st[1], c = 0;
    for (int b = 0; b < 256; b++) {
        uint hb = P.hist[b];
        if (c + hb > k) { P.st[0] |= ((uint)b << P.shift); P.st[1] = k - c; break; }
        c += hb;
    }
    for (int b = 0; b < 256; b++) P.hist[b] = 0;
}
struct SelValueArgs { device const uint* st; fp out; };
kernel void sel_value(constant SelValueArgs& P [[buffer(0)]], KARGS) {
    if (threadIdx.x == 0) *P.out = fkey_inv(P.st[0]);
}
// flowdegrain2 noise(): var = max((1.4826 med)^2, 0.001)
struct NoiseArgs { cfp med; fp var; float mult; };
kernel void noise_var(constant NoiseArgs& P [[buffer(0)]], KARGS) {
    if (threadIdx.x == 0) { float t = 1.4826f * *P.med; *P.var = fmax(t * t, 0.001f) * P.mult; }
}
// cupy quantile, method 'linear', float32: lo, hi = sorted values at floor(idx), +1
struct QuantArgs { cfp lo; cfp hi; float w; fp out; };
kernel void quantile_finish(constant QuantArgs& P [[buffer(0)]], KARGS) {
    if (threadIdx.x != 0) return;
    float a = *P.lo, b = *P.hi, diff = b - a, w = P.w;
    *P.out = w < 0.5f ? a + diff * w : b - diff * (1 - w);
}

