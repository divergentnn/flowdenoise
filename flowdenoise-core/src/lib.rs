//! Kernels and GPU backends of [flowdenoise](https://crates.io/crates/flowdenoise), a motion-compensated temporal
//! video denoiser.
//!
//! The kernels are plain CUDA C++ (`kernels/*.cu`, compiled by `build.rs` with nvcc, or hipcc for HIP) and Metal
//! Shading Language (`kernels/metal/*.metal`, compiled from source when the device is opened). The host side here
//! loads them, owns the device memory and runs the pipeline of `kernels/README.md` frame by frame:
//!
//! * [`cuda`]: the CUDA driver API (libcuda, loaded at run time), or the HIP module API with the same shape;
//! * [`metal`]: Metal on Apple GPUs (macOS only).
//!
//! Frames are 10-bit 4:2:0, planes Y, U, V back to back as `u16` (`width * height * 3 / 2` values). [`open`] gives a
//! [`Denoise`] on the platform's backend; embedders that keep frames on the device use the backend modules
//! directly (`cuda::Denoiser::step` writes into a device buffer).
//!
//! Features: `cuda` (inert on macOS), `hip` (compile-checked only), `metal` (inert outside macOS).
mod params;

pub use params::{Params, Preset};

use anyhow::{bail, Result};

#[cfg(all(any(feature = "cuda", feature = "hip"), not(target_os = "macos")))]
pub mod cuda;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

/// A temporal denoiser over a stream of frames (10-bit 4:2:0, planes Y U V back to back, `width * height * 3 / 2`
/// values). Frame t can be produced once frames up to t + radius have been pushed, or once the stream has ended.
pub trait Denoise {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    /// the next input frame
    fn push(&mut self, frame: &[u16]) -> Result<()>;
    /// end of the stream: the remaining frames become ready
    fn finish(&mut self);
    /// index of the next frame that can be produced now
    fn ready(&self) -> Option<usize>;
    /// advance past the next frame without producing it (its neighbours stay available)
    fn skip(&mut self) -> Result<usize>;
    /// produce the next frame into `out` (`width * height * 3 / 2` values); returns its index
    fn produce(&mut self, out: &mut [u16]) -> Result<usize>;
}

/// GPU backend choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Backend {
    /// Metal on macOS; CUDA, then HIP elsewhere
    #[default]
    Auto,
    Cuda,
    Hip,
    Metal,
}

impl std::str::FromStr for Backend {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Backend> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "auto" | "" => Backend::Auto,
            "cuda" => Backend::Cuda,
            "hip" | "rocm" => Backend::Hip,
            "metal" => Backend::Metal,
            x => bail!("unknown backend {x:?} (auto, cuda, hip or metal)"),
        })
    }
}

/// backends compiled into this build for this platform
#[allow(clippy::vec_init_then_push)]
pub fn compiled_backends() -> Vec<Backend> {
    #[allow(unused_mut)]
    let mut v = vec![];
    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    v.push(Backend::Cuda);
    #[cfg(all(feature = "hip", not(target_os = "macos")))]
    v.push(Backend::Hip);
    #[cfg(all(feature = "metal", target_os = "macos"))]
    v.push(Backend::Metal);
    v
}

/// A denoiser for `width` x `height` frames on the chosen backend and GPU (`device`: index among the backend's
/// devices; Metal uses the system default device). The GPU is opened once per process: later calls reuse it.
#[allow(unused_variables)]
pub fn open(width: usize, height: usize, params: &Params, backend: Backend, device: usize) -> Result<Box<dyn Denoise>> {
    params.validate()?;
    params.check_size(width, height)?;
    match backend {
        #[cfg(all(any(feature = "cuda", feature = "hip"), not(target_os = "macos")))]
        Backend::Auto | Backend::Cuda | Backend::Hip => {
            let kind = match backend {
                Backend::Cuda => Some(cuda::Kind::Cuda),
                Backend::Hip => Some(cuda::Kind::Hip),
                _ => None,
            };
            let g = cuda::init(kind, device as i32)?;
            g.bind_thread()?; // the context is current on the thread that opened it; make it current here too
            if let Some(k) = kind {
                if g.kind != k {
                    bail!("the GPU is already open with the {:?} backend", g.kind);
                }
            }
            Ok(Box::new(cuda::Denoiser::new(width, height, params.clone())?))
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Backend::Auto | Backend::Metal => {
            metal::try_init()?;
            Ok(Box::new(metal::Denoiser::new(width, height, params.clone())?))
        }
        b => bail!("backend {b:?} is not available in this build (compiled: {:?})", compiled_backends()),
    }
}

/// A kernel argument, passed by value (the kernels' C argument lists).
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    /// device pointer (CUDA / HIP device address, Metal GPU address)
    P(u64),
    I(i32),
    L(i64),
    U(u32),
    F(f32),
}

impl From<u64> for Arg {
    fn from(p: u64) -> Arg {
        Arg::P(p)
    }
}
impl From<i32> for Arg {
    fn from(x: i32) -> Arg {
        Arg::I(x)
    }
}
impl From<i64> for Arg {
    fn from(x: i64) -> Arg {
        Arg::L(x)
    }
}
impl From<u32> for Arg {
    fn from(x: u32) -> Arg {
        Arg::U(x)
    }
}
impl From<f32> for Arg {
    fn from(x: f32) -> Arg {
        Arg::F(x)
    }
}

/// `args![a, b, ...]`: a kernel argument list (`&[Arg]`) from device buffers, pointers and scalars
#[macro_export]
macro_rules! args {
    ($($a:expr),* $(,)?) => { &[$($crate::Arg::from($a)),*] };
}
