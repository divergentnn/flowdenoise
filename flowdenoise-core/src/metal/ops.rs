//! Order statistics on the device (as the CUDA backend's): radix selection (no host round trip) and CuPy's linear
//! quantile.
use super::buf::{launch, launch_reduce, DBuf};
use crate::args;
use anyhow::Result;

/// k-th smallest (0-based) of n floats at `v` -> *out (device float); the input is not modified
pub struct Select {
    st: DBuf,
    hist: DBuf,
}
impl Select {
    pub fn new() -> Result<Select> {
        Ok(Select { st: DBuf::zeroed(8)?, hist: DBuf::zeroed(256 * 4)? })
    }
    pub fn kth(&self, v: u64, n: usize, k: u32, out: u64) -> Result<()> {
        launch("radix_select_init", (1, 1, 1), (256, 1, 1), 0, args![&self.st, &self.hist, k])?;
        for shift in [24i32, 16, 8, 0] {
            launch_reduce("radix_select_histogram", n, args![v, n as i64, &self.st, &self.hist, shift])?;
            launch("radix_select_scan", (1, 1, 1), (32, 1, 1), 0, args![&self.st, &self.hist, shift])?;
        }
        launch("radix_select_value", (1, 1, 1), (32, 1, 1), 0, args![&self.st, out])
    }
    /// cupy.quantile(x, q) (method linear, float32 result) of n floats -> *out; scratch: two floats at tmp
    pub fn quantile(&self, v: u64, n: usize, q: f64, out: u64, tmp: &DBuf) -> Result<()> {
        let idx = q * (n as f64 - 1.0);
        let lo = idx.floor();
        let hi = (lo as usize + 1).min(n - 1);
        let w = (idx - lo) as f32;
        self.kth(v, n, lo as u32, tmp.ptr)?;
        self.kth(v, n, hi as u32, tmp.ptr + 4)?;
        launch("quantile_lerp", (1, 1, 1), (32, 1, 1), 0, args![tmp.ptr, tmp.ptr + 4, w, out])
    }
}
