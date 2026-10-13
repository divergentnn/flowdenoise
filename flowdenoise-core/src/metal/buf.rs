//! Device buffers and kernel launch helpers (the same interface as the CUDA backend's, on Metal).
use super::driver::{gpu, pack};
use crate::Arg;
use anyhow::Result;
use std::ffi::c_void;

/// device memory, freed on drop
pub struct DBuf {
    pub ptr: u64,
    pub bytes: usize,
}

impl DBuf {
    pub fn new(bytes: usize) -> Result<DBuf> {
        Ok(DBuf { ptr: gpu().alloc(bytes)?, bytes })
    }
    pub fn f32(n: usize) -> Result<DBuf> {
        DBuf::new(n * 4)
    }
    pub fn zeroed(bytes: usize) -> Result<DBuf> {
        let b = DBuf::new(bytes)?;
        b.zero()?;
        Ok(b)
    }
    pub fn zero(&self) -> Result<()> {
        gpu().memset(self.ptr, 0, self.bytes)
    }
    pub fn upload<T: Copy>(data: &[T]) -> Result<DBuf> {
        let b = DBuf::new(std::mem::size_of_val(data))?;
        b.write(data)?;
        Ok(b)
    }
    pub fn write<T: Copy>(&self, data: &[T]) -> Result<()> {
        let n = std::mem::size_of_val(data);
        assert!(n <= self.bytes, "write of {n} bytes into a {} byte buffer", self.bytes);
        gpu().htod(self.ptr, data.as_ptr() as *const c_void, n)
    }
    /// synchronous read of the first data.len() elements
    pub fn read<T: Copy>(&self, data: &mut [T]) -> Result<()> {
        let n = std::mem::size_of_val(data);
        assert!(n <= self.bytes);
        gpu().dtoh(data.as_mut_ptr() as *mut c_void, self.ptr, n)
    }
    pub fn to_vec<T: Copy + Default>(&self, n: usize) -> Result<Vec<T>> {
        let mut v = vec![T::default(); n];
        self.read(&mut v)?;
        Ok(v)
    }
    pub fn at(&self, elem_bytes: usize, off: usize) -> u64 {
        self.ptr + (elem_bytes * off) as u64
    }
    pub fn copy_from(&self, src: u64, bytes: usize) -> Result<()> {
        gpu().dtod(self.ptr, src, bytes)
    }
}

impl Drop for DBuf {
    fn drop(&mut self) {
        gpu().free(self.ptr);
    }
}

impl From<&DBuf> for Arg {
    fn from(b: &DBuf) -> Arg {
        Arg::P(b.ptr)
    }
}

pub fn launch(name: &str, grid: (u32, u32, u32), block: (u32, u32, u32), _shmem: u32, args: &[Arg]) -> Result<()> {
    let ptrs: Vec<u64> = args.iter().filter_map(|a| if let Arg::P(p) = a { Some(*p) } else { None }).collect();
    gpu().launch_raw(name, grid, block, &pack(args), &ptrs)
}

/// 1-D launch over n elements, 256 threads per block
pub fn launch_n(name: &str, n: usize, args: &[Arg]) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    launch(name, (n.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, args)
}

/// grid-stride reduction launch: enough blocks to fill the GPU, at most one per 256 * 8 elements
pub fn launch_reduce(name: &str, n: usize, args: &[Arg]) -> Result<()> {
    launch(name, (reduce_blocks(n), 1, 1), (256, 1, 1), 0, args)
}

pub fn reduce_blocks(n: usize) -> u32 {
    (n.div_ceil(256 * 8)).clamp(1, 1024) as u32
}
