//! Metal backend (Apple GPUs, macOS): the kernels of kernels/metal/*.metal (the CUDA kernels in Metal Shading
//! Language), compiled from source when the device is opened.
mod buf;
pub mod driver;
mod engine;
mod ops;

pub use buf::{launch, launch_n, launch_reduce, reduce_blocks, DBuf};
pub use driver::{gpu, opened, pack, try_init, Metal, DENOISE_KERNELS, OPS_KERNELS};
pub use engine::{Denoiser, Engine, Ingested};
pub use ops::Select;
