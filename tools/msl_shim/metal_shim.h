// A C++ stand-in for <metal_stdlib>, for tools/compare_msl_ir.py only: it lets clang parse the kernels' MSL as C++ so
// that two versions of a kernel can be compared as LLVM IR. The Metal built-ins are opaque declarations: what is
// compared is the kernels' own arithmetic, in source order, not what the Metal compiler makes of it.
#pragma once
#include <stdint.h>

#define kernel
#define device
#define constant const
#define threadgroup
#define thread
#define static_inline_always static inline __attribute__((always_inline))

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef unsigned long ulong;

struct uint3 {
    uint x, y, z;
};
struct float2 {
    float x, y;
    float2(float a, float b) : x(a), y(b) {}
};

// opaque math (no constant folding or library-call lowering differences between the two versions)
extern "C" {
float msl_floor(float);
float msl_fabs(float);
float msl_fmin(float, float);
float msl_fmax(float, float);
float msl_sqrt(float);
float msl_sin(float);
float msl_rint(float);
float msl_fma(float, float, float);
float msl_simd_sum(float);
void msl_barrier(int);
}
#define floor msl_floor
#define fabs msl_fabs
#define fmin msl_fmin
#define fmax msl_fmax
#define sqrt msl_sqrt
#define sin msl_sin
#define rint msl_rint
#define fma msl_fma
#define simd_sum msl_simd_sum
#define M_PI_F 3.14159265358979323846f
namespace mem_flags {
const int mem_threadgroup = 1;
}
#define threadgroup_barrier msl_barrier

template <typename T, typename U>
T as_type(U);

// overloaded built-ins: opaque declarations (mangled per argument type)
int min(int, int);
int max(int, int);
uint min(uint, uint);
uint max(uint, uint);
bool simd_any(bool);
uint simd_min(uint);
uint simd_max(uint);
float simd_min(float);
float simd_max(float);
namespace precise {
float msl_sqrt(float);  // precise::sqrt (the sqrt macro renames it)
}
namespace fast {
float msl_sqrt(float);
}

struct atomic_uint {
    uint value;
};
#define memory_order_relaxed 0
extern "C" {
void atomic_store_explicit(atomic_uint*, uint, int);
uint atomic_fetch_add_explicit(atomic_uint*, uint, int);
uint atomic_load_explicit(atomic_uint*, int);
}
