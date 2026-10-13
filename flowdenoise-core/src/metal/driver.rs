//! Metal driver: device memory, kernel launch and synchronization with the semantics the CUDA host code relies on
//! (one in-order stream, device pointers with arithmetic, asynchronous launches, a synchronous read).
//!
//! Device pointers are the buffers' GPU addresses (MTLBuffer.gpuAddress), so the orchestration code adds offsets to
//! them exactly as it does with CUDA device pointers. A kernel receives its arguments as one constant struct with the
//! CUDA argument list's C layout (pointers included, MSL 3 / argument buffers tier 2); every buffer a launch points
//! into is declared with useResource, which makes it resident and lets Metal track the hazards between launches.
//! Launches are encoded into one compute encoder (serial dispatch: in order) and committed in batches; a read
//! commits and waits. Host-to-device copies go through a staging buffer and a copy kernel, so they are ordered with
//! the launches like cuMemcpyHtoDAsync on the stream.
use anyhow::{anyhow, bail, Context, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion,
    MTLLibrary, MTLMathMode, MTLResource, MTLResourceOptions, MTLResourceUsage, MTLSize,
};
use crate::Arg;
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Mutex, MutexGuard, OnceLock, RwLock};

type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;

/// launches per command buffer before it is committed (the GPU starts on it while the host encodes the next)
const BATCH: usize = 256;

struct Stream {
    cb: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    enc: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    launches: usize,
    /// command buffers committed but not waited for
    inflight: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    /// staging buffers of pending copies (released after the next wait)
    staging: Vec<Buf>,
}

pub struct Metal {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// compute pipelines by kernel name, from every loaded library
    pipes: RwLock<HashMap<&'static str, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
    /// live buffers by GPU address
    bufs: Mutex<BTreeMap<u64, Buf>>,
    stream: Mutex<Stream>,
    pub device_name: String,
    /// FLOWDENOISE_METAL_PROFILE=1: every launch in its own command buffer, GPU time per kernel (name -> launches,
    /// seconds)
    profile: Option<Mutex<HashMap<String, (usize, f64)>>>,
    /// FLOWDENOISE_METAL_PROFILE=host: launches, encode time, waits, wait time (ns)
    host: Option<[std::sync::atomic::AtomicU64; 4]>,
}

// Metal devices, queues, pipelines and buffers are thread-safe; the encoder state is behind a mutex.
unsafe impl Send for Metal {}
unsafe impl Sync for Metal {}

static METAL: OnceLock<Result<Metal, String>> = OnceLock::new();

/// the process-wide Metal device (initialized on first use: device, queue, kernels compiled from source)
pub fn try_init() -> Result<&'static Metal> {
    METAL.get_or_init(|| Metal::new().map_err(|e| format!("{e:#}"))).as_ref().map_err(|e| anyhow!("Metal: {e}"))
}

/// the device if it was opened (no initialization)
pub fn opened() -> Option<&'static Metal> {
    METAL.get().and_then(|r| r.as_ref().ok())
}

pub fn gpu() -> &'static Metal {
    try_init().expect("Metal not initialized")
}

/// MSL sources: the denoiser kernels with fast math (as the reference's nvcc --use_fast_math), the elementwise and
/// order-statistic kernels without
pub const DENOISE_SRC: &str = include_str!("../../kernels/metal/flowdenoise.metal");
pub const OPS_SRC: &str = include_str!("../../kernels/metal/flowdenoise_ops.metal");

/// kernels of flowdenoise.metal
pub const DENOISE_KERNELS: &[&str] = &[
    "separable_filter_3_x", "separable_filter_3_y", "separable_filter_5_x", "separable_filter_5_y",
    "separable_filter_7_x", "separable_filter_7_y", "separable_filter_8_x", "separable_filter_8_y",
    "separable_filter_19_x", "separable_filter_19_y", "separable_filter_37_x", "separable_filter_37_y",
    "separable_filter_any", "downsample_2x2", "motion_search_sweep", "densify_flow", "warp_luma", "temporal_weight",
    "fuse_frames", "warp_accumulate_chroma",
];
/// kernels of flowdenoise_ops.metal
pub const OPS_KERNELS: &[&str] = &[
    "copy_words", "copy_bytes", "fill_bytes", "u16_to_float", "abs_diff", "squared_diff", "subtract", "square",
    "absolute", "divide_broadcast", "quantize_8bit", "round_to_integer", "guided_coefficients", "fallback_alpha",
    "guided_blend", "detail_gate_luma", "pack_10bit", "gather_strided", "radix_select_init", "radix_select_histogram",
    "radix_select_scan", "radix_select_value", "noise_variance", "quantile_lerp",
];

impl Metal {
    fn new() -> Result<Metal> {
        let device = MTLCreateSystemDefaultDevice().context("no Metal device")?;
        let queue = device.newCommandQueue().context("Metal command queue")?;
        let device_name = device.name().to_string();
        let m = Metal {
            device,
            queue,
            pipes: RwLock::new(HashMap::new()),
            bufs: Mutex::new(BTreeMap::new()),
            stream: Mutex::new(Stream { cb: None, enc: None, launches: 0, inflight: vec![], staging: vec![] }),
            device_name,
            profile: std::env::var("FLOWDENOISE_METAL_PROFILE").ok().filter(|v| v == "1").map(|_| Mutex::new(HashMap::new())),
            host: std::env::var("FLOWDENOISE_METAL_PROFILE").ok().filter(|v| v == "host").map(|_| Default::default()),
        };
        m.load_library(DENOISE_SRC, true, DENOISE_KERNELS)?;
        m.load_library(OPS_SRC, false, OPS_KERNELS)?;
        Ok(m)
    }

    /// compile MSL source `src` (fast math or not) and make its kernels `names` launchable by name. Kernel names are
    /// global: they must not clash with flowdenoise's own.
    pub fn load_library(&self, src: &str, fast: bool, names: &[&'static str]) -> Result<()> {
        let opts = MTLCompileOptions::new();
        opts.setLanguageVersion(MTLLanguageVersion::Version3_0);
        opts.setMathMode(if fast { MTLMathMode::Fast } else { MTLMathMode::Safe });
        let lib = self.device.newLibraryWithSource_options_error(&NSString::from_str(src), Some(&opts))
            .map_err(|e| anyhow!("compiling the Metal kernels: {}", e.localizedDescription()))?;
        let mut pipes = self.pipes.write().unwrap();
        for &n in names {
            if pipes.contains_key(n) {
                bail!("kernel {n} is already loaded from another library");
            }
            let f = lib.newFunctionWithName(&NSString::from_str(n)).with_context(|| format!("kernel {n} missing"))?;
            let p = self.device.newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| anyhow!("pipeline {n}: {}", e.localizedDescription()))?;
            pipes.insert(n, p);
        }
        Ok(())
    }

    fn new_buffer(&self, bytes: usize) -> Result<Buf> {
        self.device
            .newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared
                                         | MTLResourceOptions::HazardTrackingModeTracked)
            .with_context(|| format!("Metal: allocating {bytes} bytes"))
    }

    pub fn alloc(&self, bytes: usize) -> Result<u64> {
        let b = self.new_buffer(bytes)?;
        let a = b.gpuAddress();
        self.bufs.lock().unwrap().insert(a, b);
        Ok(a)
    }

    pub fn free(&self, p: u64) {
        // pending command buffers retain the buffers they use
        self.bufs.lock().unwrap().remove(&p);
    }

    /// the buffer holding address p and the offset of p in it
    fn find(&self, p: u64) -> Result<(Buf, usize)> {
        let m = self.bufs.lock().unwrap();
        let (base, b) = m.range(..=p).next_back().ok_or_else(|| anyhow!("Metal: {p:#x} is not a device address"))?;
        let off = (p - base) as usize;
        if off >= b.length() {
            bail!("Metal: {p:#x} is past the end of its buffer");
        }
        Ok((b.clone(), off))
    }

    /// host pointer of a device address (shared storage: the same memory)
    fn host_ptr(&self, p: u64, bytes: usize) -> Result<*mut u8> {
        let (b, off) = self.find(p)?;
        if off + bytes > b.length() {
            bail!("Metal: access of {bytes} bytes at {p:#x} overruns its buffer");
        }
        Ok(unsafe { (b.contents().as_ptr() as *mut u8).add(off) })
    }

    fn lock(&self) -> MutexGuard<'_, Stream> {
        self.stream.lock().unwrap()
    }

    fn encoder<'a>(&self, s: &'a mut Stream) -> Result<&'a ProtocolObject<dyn MTLComputeCommandEncoder>> {
        if s.enc.is_none() {
            let cb = self.queue.commandBuffer().context("Metal command buffer")?;
            let enc = cb.computeCommandEncoder().context("Metal compute encoder")?;
            s.cb = Some(cb);
            s.enc = Some(enc);
            s.launches = 0;
        }
        Ok(s.enc.as_ref().unwrap())
    }

    fn commit(&self, s: &mut Stream) {
        if let (Some(enc), Some(cb)) = (s.enc.take(), s.cb.take()) {
            enc.endEncoding();
            cb.commit();
            s.inflight.push(cb);
        }
    }

    fn wait(&self, s: &mut Stream) -> Result<()> {
        let t0 = std::time::Instant::now();
        let r = self.wait_inner(s);
        if let Some(h) = &self.host {
            use std::sync::atomic::Ordering::Relaxed;
            h[2].fetch_add(1, Relaxed);
            h[3].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        }
        r
    }

    fn wait_inner(&self, s: &mut Stream) -> Result<()> {
        self.commit(s);
        let mut err = None;
        for cb in s.inflight.drain(..) {
            cb.waitUntilCompleted();
            if cb.status() == MTLCommandBufferStatus::Error && err.is_none() {
                err = Some(cb.error().map(|e| e.localizedDescription().to_string()).unwrap_or_default());
            }
        }
        s.staging.clear();
        match err {
            Some(e) => bail!("Metal command buffer failed: {e}"),
            None => Ok(()),
        }
    }

    /// wait for all launched work
    pub fn sync(&self) -> Result<()> {
        let mut s = self.lock();
        self.wait(&mut s)
    }

    /// launch a kernel: grid of blocks, block of threads, arguments packed as the kernel's argument struct
    pub fn launch_raw(&self, name: &str, grid: (u32, u32, u32), block: (u32, u32, u32), args: &[u8], ptrs: &[u64])
                      -> Result<()> {
        let t0 = std::time::Instant::now();
        let pipe = self.pipes.read().unwrap().get(name).cloned().ok_or_else(|| anyhow!("Metal: no kernel {name}"))?;
        let mut used: Vec<Buf> = Vec::with_capacity(ptrs.len());
        for &p in ptrs {
            if p != 0 {
                let (b, _) = self.find(p)?;
                if !used.iter().any(|u| u.gpuAddress() == b.gpuAddress()) {
                    used.push(b);
                }
            }
        }
        let mut s = self.lock();
        let enc = self.encoder(&mut s)?;
        enc.setComputePipelineState(&pipe);
        for b in &used {
            let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**b);
            enc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
        }
        let data = if args.is_empty() { &[0u8; 8][..] } else { args };
        unsafe { enc.setBytes_length_atIndex(NonNull::new(data.as_ptr() as *mut c_void).unwrap(), data.len(), 0) };
        let sz = |t: (u32, u32, u32)| MTLSize { width: t.0 as usize, height: t.1 as usize, depth: t.2 as usize };
        enc.dispatchThreadgroups_threadsPerThreadgroup(sz(grid), sz(block));
        s.launches += 1;
        if let Some(prof) = &self.profile {
            let cb = s.cb.clone().unwrap();
            self.wait(&mut s)?;
            let t = cb.GPUEndTime() - cb.GPUStartTime();
            let mut m = prof.lock().unwrap();
            let e = m.entry(name.to_string()).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += t;
        } else if s.launches >= BATCH {
            self.commit(&mut s);
        }
        if let Some(h) = &self.host {
            use std::sync::atomic::Ordering::Relaxed;
            h[0].fetch_add(1, Relaxed);
            h[1].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        }
        Ok(())
    }

    /// FLOWDENOISE_METAL_PROFILE=1: GPU time per kernel so far, most expensive first
    pub fn profile_report(&self) -> Option<String> {
        if let Some(h) = &self.host {
            use std::sync::atomic::Ordering::Relaxed;
            let v: Vec<u64> = h.iter().map(|x| x.load(Relaxed)).collect();
            return Some(format!("Metal host: {} launches, encoding {:.3} s; {} waits, waiting {:.3} s", v[0],
                                v[1] as f64 * 1e-9, v[2], v[3] as f64 * 1e-9));
        }
        let m = self.profile.as_ref()?.lock().unwrap();
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
        let tot: f64 = v.iter().map(|x| x.1 .1).sum();
        let mut out = format!("Metal GPU time per kernel (total {tot:.3} s):\n");
        for (k, (n, t)) in v {
            out += &format!("  {k:16} {n:7} launches {t:8.3} s {:5.1} %\n", 100.0 * t / tot.max(1e-12));
        }
        Some(out)
    }

    /// stream-ordered host-to-device copy (staged)
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn htod(&self, dst: u64, src: *const c_void, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let st = unsafe {
            self.device.newBufferWithBytes_length_options(NonNull::new(src as *mut c_void).unwrap(), bytes,
                                                          MTLResourceOptions::StorageModeShared)
        }
        .context("Metal staging buffer")?;
        let a = st.gpuAddress();
        self.bufs.lock().unwrap().insert(a, st.clone());
        let r = self.copy(dst, a, bytes);
        self.bufs.lock().unwrap().remove(&a);
        self.lock().staging.push(st);
        r
    }

    /// synchronous device-to-host copy (waits for the launched work)
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn dtoh(&self, dst: *mut c_void, src: u64, bytes: usize) -> Result<()> {
        self.sync()?;
        let p = self.host_ptr(src, bytes)?;
        unsafe { std::ptr::copy_nonoverlapping(p, dst as *mut u8, bytes) };
        Ok(())
    }

    pub fn dtod(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        self.copy(dst, src, bytes)
    }

    fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        if bytes % 4 == 0 && dst % 4 == 0 && src % 4 == 0 {
            let n = (bytes / 4) as i64;
            let a = pack(&[Arg::P(dst), Arg::P(src), Arg::L(n)]);
            self.launch_raw("copy_words", (blocks(n as usize), 1, 1), (256, 1, 1), &a, &[dst, src])
        } else {
            let n = bytes as i64;
            let a = pack(&[Arg::P(dst), Arg::P(src), Arg::L(n)]);
            self.launch_raw("copy_bytes", (blocks(n as usize), 1, 1), (256, 1, 1), &a, &[dst, src])
        }
    }

    /// stream-ordered memset (byte value)
    pub fn memset(&self, dst: u64, v: u8, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let n = bytes as i64;
        let a = pack(&[Arg::P(dst), Arg::I(v as i32), Arg::L(n)]);
        self.launch_raw("fill_bytes", (blocks(n as usize), 1, 1), (256, 1, 1), &a, &[dst])
    }
}

fn blocks(n: usize) -> u32 {
    n.div_ceil(256).max(1) as u32
}

/// a kernel argument list packed with C struct layout (the kernels' argument structs)
pub fn pack(args: &[Arg]) -> Vec<u8> {
    let mut v: Vec<u8> = Vec::with_capacity(args.len() * 8);
    let mut put = |bytes: &[u8], align: usize| {
        while v.len() % align != 0 {
            v.push(0);
        }
        v.extend_from_slice(bytes);
    };
    for a in args {
        match *a {
            Arg::P(p) => put(&p.to_ne_bytes(), 8),
            Arg::L(x) => put(&x.to_ne_bytes(), 8),
            Arg::I(x) => put(&x.to_ne_bytes(), 4),
            Arg::U(x) => put(&x.to_ne_bytes(), 4),
            Arg::F(x) => put(&x.to_ne_bytes(), 4),
        }
    }
    while v.len() % 8 != 0 {
        v.push(0);
    }
    v
}
