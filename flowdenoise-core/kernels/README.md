# flowdenoise kernels

| file | compiled | contents |
|---|---|---|
| `flowdenoise.cu` | nvcc **with** `--use_fast_math` | separable filters, motion pyramid, motion search, flow densification, Lanczos-3 warps, temporal weights, fusion |
| `flowdenoise_ops.cu` | nvcc **without** fast math | conversions, elementwise steps of the guided fallback and the detail gate, output packing, radix select |
| `portable.h` | | the HIP shim (warp shuffle) |
| `metal/flowdenoise.metal` | Metal, fast math | MSL port of `flowdenoise.cu` |
| `metal/flowdenoise_ops.metal` | Metal, no fast math | MSL port of `flowdenoise_ops.cu`, plus the Metal driver's copy / fill kernels |

The host side (`src/cuda/engine.rs`, `src/metal/engine.rs`) launches them in the order below. Each kernel has a header
comment with what it computes, its inputs and outputs (layout, units), its thread mapping and its numerical notes.

## How the stages fit together

Frames are 10-bit 4:2:0. Every input frame is prepared once (*ingest*); every output frame t is then made from frame t
and its neighbours t - r ... t + r (r = 3: six neighbours, fewer at the ends of the stream).

```
ingest (once per input frame)
  u16_to_float                      Y, U, V as float (10-bit codes)
  separable_filter_7 x, y           Gaussian pre-blur of Y (sigma ~1)        \
  downsample_2x2                    (only above 2000 px width: motion at half resolution)
  quantize_8bit                     8-bit codes: pyramid level 0            | the motion pyramid
  downsample_2x2 [+ round_to_integer from level `finest` on]  levels 1 .. 6 /
  separable_filter_5 x, y (edge fix) prefiltered Y, 2 x [1 2 1] / 4, for the SAD

frame t
  1. motion, coarse to fine (pyramid level 6 down to `finest` = 2), all neighbours in one grid per launch
       motion_search_sweep (forward)  patch vectors: candidates, Gauss-Newton refinement, left-to-right propagation
       motion_search_sweep (reverse)  right to left, also trying the forward vector of the patch below
       densify_flow                   dense flow of the level; the next finer level starts from it
  2. per neighbour d
       warp_luma                      Lanczos-3 warp of the neighbour's Y; |warped - Y|; flow speed
       separable_filter_5 (edge fix)  prefiltered warped Y
       abs_diff, separable_filter_8   |prefiltered warped - prefiltered Y|, 8x8 box mean (zero padded) = SAD
       separable_filter_37            Gaussian sigma 6 of |warped - Y| (spike reference)
       separable_filter_7             Gaussian sigma 1 of warped Y (its fine detail = warped - this)
       temporal_weight                MVTools DegrainWeight x spike rejection; warped fine detail
       warp_accumulate_chroma         chroma warped with the 2x2-averaged flow, weighted, summed
  3. fusion
       fuse_frames                    luma sums: weighted, weights, squared weights, detail (signed, absolute)
       divide_broadcast               fused Y = sum / weight sum; fused U, V likewise
  4. guided fallback (where few frames agree: effective frame count (sum w)^2 / sum w^2 below `fallback`)
       fallback_alpha                 blend factor
       noise variance per plane       gather_strided (every 53rd |high-pass| sample), radix select (median),
                                      noise_variance
       guided filter                  box mean of x and x^2 (separable_filter_7 / _3), guided_coefficients, box means
                                      of the coefficients, guided_blend; chroma at half size with downsample_2x2(alpha)
  5. detail gate and output
       separable_filter_7, squared_diff, separable_filter_19   local detail energy of the fused Y (sigma 1, sigma 3)
       gather_strided, radix select x 2, quantile_lerp         its 25th percentile (every 97th sample): flat level
       absolute, separable_filter_7 x 2                        smoothed detail coherence terms
       detail_gate_luma               current frame's own detail put back where it is coherent texture -> int16 Y
       pack_10bit                     -> int16 U, V
```

The tap counts of the separable filters: 3 (chroma guided-filter box, radius 1), 5 (prefilter), 7 (Gaussian sigma 1,
luma guided-filter box radius 3, motion pre-blur), 8 (SAD box), 19 (Gaussian sigma 3), 37 (Gaussian sigma 6). Other
parameters fall back to `separable_filter_any`.

Algorithms with a reference: the motion search follows DIS (Kroeger, Timofte, Dai, Van Gool, *Fast Optical Flow using
Dense Inverse Search*, ECCV 2016): inverse-compositional patch search on a pyramid, mean-normalised SSD, spatial
propagation, densification weighted by the photometric error. The temporal weight is MVTools' DegrainWeight
(MDegrain). The fallback is a self-guided filter (He, Sun, Tang, *Guided Image Filtering*, ECCV 2010). The noise
level is the median absolute deviation estimate (sigma = 1.4826 x median |high-pass|). The detail gate is this
project's own. No third-party code is included; see the repository README.

## Numerical rules

The CUDA output is bit-identical to the Python reference (CuPy). That holds only while these rules hold:

1. **Fast math is on deliberately** for `flowdenoise.cu`: the reference compiles those kernels as a CuPy RawModule with
   `--use_fast_math` (flush-to-zero, approximate division and square root, fast transcendentals). Turning it off, or
   moving a kernel between the two files, changes the output.
2. **Fast math is off** for `flowdenoise_ops.cu`: these are the reference's CuPy elementwise kernels, which CuPy
   compiles without it (IEEE division and square root).
3. **Summation order matters.** Float addition is not associative. Every sum runs in the reference's order: the
   symmetric filters add the centre tap, then the mirrored pairs from the inside out; the patch sums are a warp
   shuffle tree, then warp 0 plus warp 1; the temporal sums start from the current frame (weight 1) and add the
   neighbours in the order t - r, ..., t - 1, t + 1, ..., t + r (luma in `fuse_frames`, chroma in launch order).
4. **Expression structure matters.** nvcc contracts `a * b + c` into one FMA within an expression (`--fmad=true`,
   the default). Splitting an expression into several statements, merging statements, or hoisting a subexpression can
   change where FMAs form, and with them the rounding.
5. **Rounding**: `rintf` and `__float2int_rn` round half to even, as NumPy / CuPy's `rint`. The median is the lower
   middle element (k = (n - 1) / 2), the quantile is NumPy's linear interpolation in its `lerp` form.
6. **Determinism**: no floating-point atomics anywhere (the only atomics count integers in the radix select), so the
   output is the same on every run.
7. **Metal** is not bit-exact with CUDA (another compiler, other fast-math functions: `sin(pi x)` instead of `sinpif`;
   the motion search's patch sums use SIMD-group reductions in another order). It is deterministic.

Comments, names and layout can change freely. Anything else needs the checks below to stay identical.

## Checking an edit

| check | what it compares | where |
|---|---|---|
| `tools/compare_sass.py REV` | the SASS of every CUDA kernel (sm_75 ... sm_120) and the embedded PTX, kernel by kernel, against revision REV | needs nvcc |
| `tools/compare_msl_ir.py REV` | every MSL kernel compiled as C++ to LLVM IR (Metal built-ins opaque), against REV: a pre-flight check without a Mac | needs clang |
| `tools/check_cuda.sh [clips]` | output hashes of both presets, before and after the readability commit | CUDA GPU |
| `tools/check_metal.sh [clips]` | the same on Metal | Mac |

`.clang-format` at the repository root is the style: `clang-format -i flowdenoise-core/kernels/*.cu
flowdenoise-core/kernels/*.h flowdenoise-core/kernels/metal/*.metal`.

## Names

The readability pass renamed the kernels; the Python reference and older GRAIL code use the old names.

| old | new | | old | new |
|---|---|---|---|---|
| `sep{3,5,7,8,19,37}_0` / `_1` | `separable_filter_{taps}_x` / `_y` | | `u16_to_f32` | `u16_to_float` |
| `filter_tile<L, AX>` | `separable_filter_tile<TAPS, AXIS>` | | `ew_absdiff` | `abs_diff` |
| `sep` | `separable_filter_any` | | `ew_energy` | `squared_diff` |
| `down` | `downsample_2x2` | | `ew_sub`, `ew_sqr`, `ew_abs` | `subtract`, `square`, `absolute` |
| `reduce` | `patch_sum` | | `ew_div_bc` | `divide_broadcast` |
| `raster` | `motion_search_sweep` | | `ew_q8`, `ew_rint` | `quantize_8bit`, `round_to_integer` |
| `dense` | `densify_flow` | | `ew_guide`, `ew_alpha`, `ew_blend` | `guided_coefficients`, `fallback_alpha`, `guided_blend` |
| `lanc`, `warp`, `warp2` | `lanczos3`, `sample_lanczos3`, `sample_lanczos3_pair` | | `ew_finish` | `detail_gate_luma` |
| `warp_y` | `warp_luma` | | `ew_pack` | `pack_10bit` |
| `weight_only` | `temporal_weight` | | `gather_stride` | `gather_strided` |
| `reduce_frames` | `fuse_frames` | | `sel_init`, `sel_hist`, `sel_scan`, `sel_value` | `radix_select_init`, `_histogram`, `_scan`, `_value` |
| `chroma` | `warp_accumulate_chroma` | | `fkey`, `fkey_inv` | `float_to_ordered_key`, `ordered_key_to_float` |
| `cl`, `sat`, `at`, `bil` | `clamp_index`, `saturate`, `sample_clamped`, `sample_bilinear` | | `noise_var`, `quantile_finish` | `noise_variance`, `quantile_lerp` |
| Metal `copy_u32`, `copy_u8`, `fill_u8` | `copy_words`, `copy_bytes`, `fill_bytes` | | | |

Two kernels of the reference were dropped: `motion` (its experimental parallel patch matcher, `--motion patch`) and
`accum` (superseded by `temporal_weight` + `fuse_frames`); the production path never launched either.
