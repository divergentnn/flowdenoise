// Portability shims for the flowdenoise kernels: plain CUDA C++ that hipify / hipcc accept unchanged.
// Only device code lives in these files; the host side (module load, launch) is in Rust.
#pragma once
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
// wave64 hardware: the 64-thread reductions below only read lanes 0 and 32, which a full-width
// shuffle-down by 16..1 leaves exact (see README, "HIP").
#define FLOWDENOISE_SHFL_DOWN(x, d) __shfl_down((x), (d))
#else
#define FLOWDENOISE_SHFL_DOWN(x, d) __shfl_down_sync(0xffffffffu, (x), (d))
#endif
#include <stdint.h>
