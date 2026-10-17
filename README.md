# flowdenoise

Motion-compensated temporal video denoising on the GPU, built to remove film grain before an encode without
smearing the picture under it.

flowdenoise aligns each frame's neighbours (t - 3 ... t + 3; t - 8 ... t + 8 for cel animation) with a GPU patch
motion search and a Lanczos-3 warp,
weights them per pixel the way MVTools' MDegrain does, and fuses them. Where too few frames agree it falls back toward
a guided filter, and it puts back the current frame's own fine detail wherever the aligned frames agree that the
detail is texture rather than noise.

It is the denoiser of an AV1 encode pipeline (grain removal before film-grain synthesis) and the measurement denoiser
of [GRAILS](https://github.com/divergentnn/grail), a grain-aware video quality metric.

## Features

- **GPU motion search** - a DIS-style inverse search on overlapping 8x8 patches over a 7-level pyramid, with
  row-sweep propagation and photometric densification. No CPU optical flow; the CPU stays free for the encoder.
- **Lanczos-3 warps** - neighbours are resampled with a normalised 6x6 Lanczos-3 kernel; bilinear or bicubic warps
  low-pass every sub-pixel-shifted neighbour and cost detail.
- **MVTools-style weighting** - DegrainWeight on an 8x8 SAD of prefiltered luma, with a threshold that falls with the
  motion speed and a spike rejection that drops neighbours where they disagree locally.
- **Guided fallback** - where the effective number of fused frames is low (occlusions, cuts), the result is blended
  toward a self-guided filter, so the noise level stays even.
- **Texture-only detail gate** - the current frame's fine detail is restored where it is temporally coherent and the
  area is textured, never in flat areas, where it would be grain.
- **Three presets** - `production` (the encode pipeline's), `measure` (GRAIL's; no detail boost) and `cel` (2D cel
  animation with grainy cels: 16 neighbours, the guided fallback everywhere, gated by stability).
- **Range check** (opt-in) - per pixel, a neighbour that disagrees with the current frame by more than a few noise
  sigmas loses its weight, so faint drawn strokes, rain and sprites survive a 1-pixel misalignment.
- **Library and binary** - a Rust library (frames in, denoised frames out, lookahead handled internally) and a
  y4m command-line tool behind the `binary` feature.
- **10-bit 4:2:0** - frames are processed as 10-bit planes; 8-bit sources convert losslessly.

## Hardware support

| backend | feature | status |
|---|---|---|
| NVIDIA GPUs (CUDA 12+, sm_75 and newer; PTX for newer GPUs) | `cuda` (default) | tested (RTX 3090) |
| Apple GPUs (Metal, macOS) | `metal` (default; inert elsewhere) | tested (M1 Max); tiny numeric differences from CUDA. The `cel` preset (radius 8), its stability gate and the range check are compile-checked only on Metal: not run yet |
| AMD GPUs (HIP / ROCm) | `hip` | the kernels compile with hipcc; not run yet |

There is no CPU backend.

## Building

```sh
cargo build --release --features binary          # target/release/flowdenoise
```

- CUDA builds need `nvcc` (CUDA 12 or newer) at build time: on `PATH`, or `NVCC`, `CUDA_PATH`. The kernels are
  compiled for `FLOWDENOISE_CUDA_ARCHS` (default `75;80;86;89;90;120`, plus PTX for newer GPUs) and embedded in the
  binary; at run time only the NVIDIA driver (`libcuda`) is needed.
- On macOS the Metal kernels are compiled from source when the GPU is opened; nothing else is needed.
- HIP: `--no-default-features --features hip,binary` with ROCm's `hipcc` (`FLOWDENOISE_HIP_ARCHS`).

## Command line

```sh
# denoise a clip (input and output are y4m; - is stdin / stdout)
ffmpeg -i input.mkv -pix_fmt yuv420p10le -strict -1 -f yuv4mpegpipe - | flowdenoise - denoised.y4m

# straight into an encoder
ffmpeg -i input.mkv -pix_fmt yuv420p10le -strict -1 -f yuv4mpegpipe - \
  | flowdenoise --preset production - - \
  | SvtAv1EncApp -i stdin --preset 6 --crf 30 --film-grain 8 --film-grain-denoise 0 -b out.ivf
```

```
flowdenoise [OPTIONS] <INPUT> <OUTPUT>

  -p, --preset <PRESET>     production (default), measure or cel  [env: FLOWDENOISE_PRESET]
  -b, --backend <BACKEND>   auto, cuda, hip or metal              [env: FLOWDENOISE_BACKEND]
      --gpu <N>             GPU index among the backend's devices [env: FLOWDENOISE_GPU]
  -q, --quiet               no progress line on stderr
      --print-params        print the effective parameters and exit
  parameters (default: the preset's): --radius --thsad --flow-scale --levels --finest --iterations
      --no-round-motion --detail-gate --detail-boost --spike --fast-th --fallback --fallback-eps --fallback-radius
      --fb-gate --fb-q --fb-sg --fb-win --range-k --range-soft --range-sigma
```

```sh
# 2D cel animation; the range check on, sigma estimated per frame and neighbour
flowdenoise --preset cel --range-k 2 in.y4m out.y4m
```

The input must be 10-bit 4:2:0 y4m (`C420p10`). The output header is the input's, byte for byte; frame parameters on
`FRAME` lines are not carried over. The progress line on stderr is rewritten in place on a terminal and printed every
10 s otherwise.

## Presets and parameters

| parameter | production | measure | cel | meaning |
|---|---|---|---|---|
| `radius` | 3 | 3 | **8** | frames t - r ... t + r are fused |
| `thsad` | 450 | 450 | 450 | SAD threshold of the weights (MVTools thSAD: 8x8 block, 8-bit units) |
| `flow-scale` | auto | auto | auto | motion at half resolution above 2000 px width |
| `levels`, `finest` | 7, 2 | 7, 2 | 7, 2 | motion pyramid depth; finest level searched (flow interpolated from there) |
| `iterations` | 12 | 12 | 12 | Gauss-Newton steps per patch, over both sweeps |
| `round-motion` | on | on | on | pyramid rounded to 8-bit codes |
| `spike` | 3 | 3 | 3 | spike rejection strength |
| `fast-th` | 8 | 8 | 8 | the SAD threshold falls as thsad / (1 + speed / fast_th) |
| `fallback`, `fallback-eps`, `fallback-radius` | 4, 2, 3 | 4, 2, 3 | **1000**, 2, **5** | guided fallback below `fallback` effective frames (1000: everywhere); eps in noise variances; box radius |
| `fb-gate` | 0 | 0 | **3** | stability gate on the fallback (0 = off): skipped where the fused frame's high-pass energy is `fb-gate` x the averaged grain's or more |
| `fb-q`, `fb-sg`, `fb-win` | 1, 2, 3 | 1, 2, 3 | 1, 2, 3 | stability gate: scale of the expected grain energy; high-pass sigma; energy window sigma |
| `range-k` | 0 | 0 | 0 | range check (0 = off): a neighbour pixel keeps its weight up to `range-k` sigma_d of difference from the current frame |
| `range-soft` | 1.5 | 1.5 | 1.5 | range check: the weight falls to 0 over this many more sigma_d |
| `range-sigma` | 0 | 0 | 0 | range check: the source's pixel noise in 10-bit codes (sigma_d = sqrt(2) x it); 0 = estimated per frame and neighbour |
| `detail-gate` | 0.6 | 0.6 | 0.6 | coherence above which the current frame's detail is restored |
| `detail-boost` | **1.3** | **1** | 1.3 | gain of the restored detail |

`production` is the Python reference's defaults, as the encode pipeline runs it; `measure` is what GRAILS pins for
its measurements (no sharpening of the reference it measures against); `cel` is the reference's `--preset cel`.

### Cel animation

For 2D cel animation with grainy cels, the `cel` preset fuses 17 frames and blends the result toward the guided
filter almost everywhere (`fallback 1000`), which takes the grain off the flat cels. Drawn texture (hatching, painted
backgrounds) would be smoothed by that filter, so the **stability gate** skips it where the fused frame keeps
structure well above what the averaged grain would leave: R, the local energy of its high-pass (x - G(x, 2),
window G(., 3)), over the energy the averaged grain would leave there (noise variance x sum w^2 / (sum w)^2), gives
s = clamp((R - 1) / (fb_gate - 1), 0, 1), and the fallback is blended in by (1 - s).

### Range check

`--range-k K` (off by default in every preset) protects thin detail that the motion search cannot align to the pixel:
faint drawn strokes, rain and small sprites. A neighbour 1 pixel off on such a stroke still passes the 8x8
SAD weight, and the average erases the stroke. With the range check, each neighbour pixel keeps its weight while
|warped - current| <= K sigma_d and loses it linearly up to (K + `range-soft`) sigma_d, where sigma_d is the expected
difference of two noisy samples: sqrt(2) x `--range-sigma` (the source's noise, e.g. from a scene pre-pass), or with
`--range-sigma 0` 1.4826 x the median |warped - current| over every 53rd pixel of mid luma (100 ... 900; all of them if
fewer than 1000), at least 0.5, per frame and neighbour. The estimate runs on the GPU (a counted gather and the radix
select); there is no host round trip. Flat areas dominate cel frames, so the median reads the noise, not the
misaligned lines.

## Library

```toml
[dependencies]
flowdenoise = "0.1"
```

```rust
use flowdenoise::{Backend, Denoiser, Frame, Params};

let mut den = Denoiser::new(1920, 1080, &Params::production(), Backend::Auto)?;
for frame in frames {                       // flowdenoise::Frame: 10-bit 4:2:0 planes
    den.send_frame(&frame)?;
    while let Some(out) = den.receive_frame()? {
        write(out);
    }
}
den.flush();                                // end of input: the last `radius` frames come out
while let Some(out) = den.receive_frame()? {
    write(out);
}
```

`Frame` converts from and to little-endian bytes (raw `yuv420p10le`, y4m payloads), strided 10-bit planes and 8-bit
planes; with the `v_frame` feature also `v_frame::frame::Frame<u16>`. The engines crate, `flowdenoise-core`, exposes
the backends directly for embedders that keep frames on the GPU (GRAILS shares the CUDA context and loads its own
kernels next to flowdenoise's).

## Exactness

The CUDA build reproduces the Python reference (CuPy) bit for bit: the kernels are the reference's, compiled with the
same fast-math settings, and the host code issues the same launches in the same order. Checked on a 1080p clip
(24 frames), a 4K HDR clip (16 frames) and an odd-sized 1000x562 clip (5 frames): `production` equals the reference
with its default arguments, `measure` equals it with `--detail-boost 1` and equals `grail denoise`. On two 2D animation
clips (1920x804 and 1920x1080, 24 frames each) `cel`, `cel --range-k 2` and `cel --range-k 2 --range-sigma 2` equal
the reference's `--preset cel` with the same options; so do `cel --range-k 2` on a dark clip (no mid-luma pixels: the
estimate's fallback to all samples) and `production --range-k 2`. `--range-k 0` is the same as no range check, byte
for byte.

`flowdenoise-core/kernels/README.md` explains the numerical rules that keep it that way and the tools that check a
kernel edit (identical SASS and PTX per kernel; output hashes before and after).

The readability pass of the kernels (renames, comments, layout) is proven a no-op on CUDA only: identical SASS on
sm_75 ... sm_120, identical PTX and identical outputs. For Metal, the equivalence is established by
`tools/check_metal.sh` on a Mac, which compares the output hashes before and after the pass; that run is still
pending. The LLVM IR comparison of the MSL compiled as C++ (`tools/compare_msl_ir.py`) is only a pre-flight check,
because it does not model the Metal compiler. HIP has no such check yet: its kernels compile with hipcc but have not
been run. No SASS comparison covers Metal or HIP.

## Speed

On an RTX 3090 shared with other jobs, the CLI runs about as fast as the Python reference: about 100 frames/s at
1080p and 30 frames/s at 4K (y4m from disk, output discarded; the reference: 105 and 29).

The library loop with the frames in memory (upload, GPU work and download, one frame at a time), per frame on the
same GPU: at 1920x1080 `production` 9.5 ... 10 ms, `cel` 19.5 ... 20 ms, `cel --range-k 2` 20.7 ms (the estimated
sigma: about 1.1 ms for 16 neighbours), with `--range-sigma` 19.8 ms; at 1920x804 7.6, 14.8, 16.0 and 15.1 ms. The GPU waits while a frame
is copied in and out; overlapping those copies with the next frame's work is on the roadmap.

## Algorithms and acknowledgements

flowdenoise contains no third-party code. It implements published methods:

- the motion search follows DIS: T. Kroeger, R. Timofte, D. Dai, L. Van Gool, *Fast Optical Flow using Dense Inverse
  Search*, ECCV 2016 (inverse-compositional patch search, mean-normalised SSD, spatial propagation, densification
  weighted by the photometric error). The CPU denoiser it replaced ran OpenCV's DIS (ultrafast preset); this GPU
  search keeps DIS's design (8-bit input pyramid, 8x8 patches, two propagation sweeps, rejection of updates larger
  than the patch) but not its exact output;
- the temporal weight is MVTools' DegrainWeight formula, (thSAD^2 - SAD^2) / (thSAD^2 + SAD^2) (MVTools: Manao,
  Fizick and contributors; MDegrain);
- the fallback is a self-guided filter: K. He, J. Sun, X. Tang, *Guided Image Filtering*, ECCV 2010;
- the noise level is the median absolute deviation estimate; the quantile is NumPy's linear interpolation, reproduced
  for exactness.

## Roadmap

- a VapourSynth plugin crate;
- publishing on crates.io;
- HIP: a run on AMD hardware (compile-checked only so far); a Vulkan backend;
- overlapping uploads, kernels and downloads across frames (pinned buffers, asynchronous copies) for more speed.

## Licence

GPL-3.0-only; see `LICENSE`.
