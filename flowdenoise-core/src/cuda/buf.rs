//! Device buffers and kernel launch helpers.
use super::driver::gpu;
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
        gpu().dtoh(data.as_mut_ptr() as *mut c_void, self.ptr, n)?;
        gpu().sync()
    }
    pub fn to_vec<T: Copy + Default>(&self, n: usize) -> Result<Vec<T>> {
        let mut v = vec![T::default(); n];
        self.read(&mut v)?;
        Ok(v)
    }
    /// element offset view (f32 / u16 ... sized by `elem`)
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

pub fn launch(name: &str, grid: (u32, u32, u32), block: (u32, u32, u32), shmem: u32, args: &[Arg]) -> Result<()> {
    let mut vals: Vec<[u8; 8]> = args
        .iter()
        .map(|a| {
            let mut v = [0u8; 8];
            match a {
                Arg::P(p) => v.copy_from_slice(&p.to_ne_bytes()),
                Arg::I(x) => v[..4].copy_from_slice(&x.to_ne_bytes()),
                Arg::L(x) => v.copy_from_slice(&x.to_ne_bytes()),
                Arg::U(x) => v[..4].copy_from_slice(&x.to_ne_bytes()),
                Arg::F(x) => v[..4].copy_from_slice(&x.to_ne_bytes()),
            }
            v
        })
        .collect();
    let mut ptrs: Vec<*mut c_void> = vals.iter_mut().map(|v| v.as_mut_ptr() as *mut c_void).collect();
    gpu().launch_raw(name, grid, block, shmem, &mut ptrs)
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
    let blocks = (n.div_ceil(256 * 8)).clamp(1, 1024) as u32;
    launch(name, (blocks, 1, 1), (256, 1, 1), 0, args)
}
