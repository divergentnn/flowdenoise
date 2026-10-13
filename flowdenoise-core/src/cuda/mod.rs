//! CUDA backend (and HIP, compile-checked): the kernels of kernels/*.cu, compiled by build.rs with nvcc (or hipcc),
//! embedded in the binary and launched through the driver API loaded at run time.
mod buf;
mod driver;
mod engine;
mod images;
mod ops;

pub use buf::{launch, launch_n, launch_reduce, DBuf};
pub use driver::{gpu, init, opened, try_init, Gpu, Kind};
pub use engine::{Denoiser, Engine, Ingested};
pub use images::{DENOISE_KERNELS, OPS_KERNELS};
pub use ops::Select;
