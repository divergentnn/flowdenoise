//! Minimal GPU driver layer: the CUDA driver API (libcuda) or the HIP module API (libamdhip64), loaded at run time,
//! with the same shape for both (module load from an embedded image, kernel launch by name, device memory, one
//! stream). Device pointers are u64 on both sides.
//!
//! The device is opened once per process ([`init`] / [`try_init`]) with flowdenoise's own kernel images loaded;
//! an embedder adds its own kernels to the same context with [`Gpu::load_module`] and launches them by name like
//! flowdenoise's.
use anyhow::{anyhow, bail, Context, Result};
use libloading::Library;
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::{OnceLock, RwLock};

type Ptr = *mut c_void;

/// which driver API the device was opened with
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Cuda,
    Hip,
}

#[allow(clippy::type_complexity)]
struct Api {
    init: unsafe extern "C" fn(u32) -> i32,
    module_load: unsafe extern "C" fn(*mut Ptr, *const c_void) -> i32,
    get_function: unsafe extern "C" fn(*mut Ptr, Ptr, *const c_char) -> i32,
    launch: unsafe extern "C" fn(Ptr, u32, u32, u32, u32, u32, u32, u32, Ptr, *mut Ptr, *mut Ptr) -> i32,
    alloc: unsafe extern "C" fn(*mut u64, usize) -> i32,
    free: unsafe extern "C" fn(u64) -> i32,
    htod: unsafe extern "C" fn(u64, *const c_void, usize, Ptr) -> i32,
    dtoh: unsafe extern "C" fn(*mut c_void, u64, usize, Ptr) -> i32,
    dtod: unsafe extern "C" fn(u64, u64, usize, Ptr) -> i32,
    memset8: unsafe extern "C" fn(u64, u8, usize, Ptr) -> i32,
    stream_create: unsafe extern "C" fn(*mut Ptr, u32) -> i32,
    stream_sync: unsafe extern "C" fn(Ptr) -> i32,
    host_alloc: unsafe extern "C" fn(*mut Ptr, usize, u32) -> i32,
    host_free: unsafe extern "C" fn(Ptr) -> i32,
    // CUDA only
    device_get: Option<unsafe extern "C" fn(*mut i32, i32) -> i32>,
    ctx_retain: Option<unsafe extern "C" fn(*mut Ptr, i32) -> i32>,
    ctx_set: Option<unsafe extern "C" fn(Ptr) -> i32>,
    cu_err: Option<unsafe extern "C" fn(i32, *mut *const c_char) -> i32>,
    cu_name: Option<unsafe extern "C" fn(*mut c_char, i32, i32) -> i32>,
    // HIP only
    hip_set_device: Option<unsafe extern "C" fn(i32) -> i32>,
    hip_err: Option<unsafe extern "C" fn(i32) -> *const c_char>,
    hip_name: Option<unsafe extern "C" fn(*mut c_char, i32, i32) -> i32>,
}

pub struct Gpu {
    pub kind: Kind,
    pub device_name: String,
    /// device index (among the driver's devices)
    pub device: i32,
    api: Api,
    ctx: Ptr,
    stream: Ptr,
    /// kernels by name, from every loaded module
    funcs: RwLock<HashMap<&'static str, Ptr>>,
    _lib: Library,
}
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

static GPU: OnceLock<Gpu> = OnceLock::new();

/// the process-wide GPU (opened on first use with [`try_init`]'s defaults; exits if there is none)
pub fn gpu() -> &'static Gpu {
    if let Some(g) = GPU.get() {
        return g;
    }
    try_init().unwrap_or_else(|e| {
        eprintln!("flowdenoise: no usable GPU: {e:#}");
        std::process::exit(3)
    })
}

/// open the GPU (once per process): `kind` None tries CUDA, then HIP; `device` is the driver's device index. Later
/// calls return the device opened first, whatever they ask for.
pub fn init(kind: Option<Kind>, device: i32) -> Result<&'static Gpu> {
    if let Some(g) = GPU.get() {
        return Ok(g);
    }
    let g = Gpu::new(kind, device)?;
    Ok(GPU.get_or_init(|| g))
}

/// [`init`] with FLOWDENOISE_BACKEND (cuda | hip) and FLOWDENOISE_GPU (device index) from the environment
pub fn try_init() -> Result<&'static Gpu> {
    let kind = match std::env::var("FLOWDENOISE_BACKEND").unwrap_or_default().to_lowercase().as_str() {
        "cuda" => Some(Kind::Cuda),
        "hip" | "rocm" => Some(Kind::Hip),
        _ => None,
    };
    let device = std::env::var("FLOWDENOISE_GPU").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    init(kind, device)
}

/// the GPU if it was opened (no initialization)
pub fn opened() -> Option<&'static Gpu> {
    GPU.get()
}

macro_rules! sym {
    ($lib:expr, $name:expr) => {{
        let s: libloading::Symbol<*const c_void> =
            unsafe { $lib.get(concat!($name, "\0").as_bytes()) }.with_context(|| format!("symbol {}", $name))?;
        unsafe { std::mem::transmute::<*const c_void, _>(*s) }
    }};
}

impl Gpu {
    fn new(kind: Option<Kind>, device: i32) -> Result<Gpu> {
        let mut errs = vec![];
        let order: Vec<Kind> = match kind {
            Some(k) => vec![k],
            None => vec![Kind::Cuda, Kind::Hip],
        };
        for b in order {
            if !super::images::available(b) {
                errs.push(format!("{b:?}: not compiled in (cargo feature)"));
                continue;
            }
            match Self::open(b, device) {
                Ok(g) => return Ok(g),
                Err(e) => errs.push(format!("{b:?}: {e:#}")),
            }
        }
        bail!("{}", errs.join("; "))
    }

    // the driver's entry points are typed by the Api fields they are stored in
    #[allow(clippy::missing_transmute_annotations)]
    fn open(b: Kind, dev_id: i32) -> Result<Gpu> {
        let lib = match b {
            Kind::Cuda => unsafe { Library::new("libcuda.so.1") }.or_else(|_| unsafe { Library::new("libcuda.so") }),
            Kind::Hip => unsafe { Library::new("libamdhip64.so") }
                .or_else(|_| unsafe { Library::new("libamdhip64.so.6") })
                .or_else(|_| unsafe { Library::new("libamdhip64.so.7") }),
        }
        .context("loading the GPU driver library")?;
        let api = match b {
            Kind::Cuda => Api {
                init: sym!(lib, "cuInit"),
                module_load: sym!(lib, "cuModuleLoadData"),
                get_function: sym!(lib, "cuModuleGetFunction"),
                launch: sym!(lib, "cuLaunchKernel"),
                alloc: sym!(lib, "cuMemAlloc_v2"),
                free: sym!(lib, "cuMemFree_v2"),
                htod: sym!(lib, "cuMemcpyHtoDAsync_v2"),
                dtoh: sym!(lib, "cuMemcpyDtoHAsync_v2"),
                dtod: sym!(lib, "cuMemcpyDtoDAsync_v2"),
                memset8: sym!(lib, "cuMemsetD8Async"),
                stream_create: sym!(lib, "cuStreamCreate"),
                stream_sync: sym!(lib, "cuStreamSynchronize"),
                host_alloc: sym!(lib, "cuMemHostAlloc"),
                host_free: sym!(lib, "cuMemFreeHost"),
                device_get: Some(sym!(lib, "cuDeviceGet")),
                ctx_retain: Some(sym!(lib, "cuDevicePrimaryCtxRetain")),
                ctx_set: Some(sym!(lib, "cuCtxSetCurrent")),
                cu_err: Some(sym!(lib, "cuGetErrorString")),
                cu_name: Some(sym!(lib, "cuDeviceGetName")),
                hip_set_device: None,
                hip_err: None,
                hip_name: None,
            },
            Kind::Hip => Api {
                init: sym!(lib, "hipInit"),
                module_load: sym!(lib, "hipModuleLoadData"),
                get_function: sym!(lib, "hipModuleGetFunction"),
                launch: sym!(lib, "hipModuleLaunchKernel"),
                alloc: sym!(lib, "hipMalloc"),
                free: sym!(lib, "hipFree"),
                htod: sym!(lib, "hipMemcpyHtoDAsync"),
                dtoh: sym!(lib, "hipMemcpyDtoHAsync"),
                dtod: sym!(lib, "hipMemcpyDtoDAsync"),
                memset8: sym!(lib, "hipMemsetD8Async"),
                stream_create: sym!(lib, "hipStreamCreateWithFlags"),
                stream_sync: sym!(lib, "hipStreamSynchronize"),
                host_alloc: sym!(lib, "hipHostMalloc"),
                host_free: sym!(lib, "hipHostFree"),
                device_get: None,
                ctx_retain: None,
                ctx_set: None,
                cu_err: None,
                cu_name: None,
                hip_set_device: Some(sym!(lib, "hipSetDevice")),
                hip_err: Some(sym!(lib, "hipGetErrorString")),
                hip_name: Some(sym!(lib, "hipDeviceGetName")),
            },
        };
        let mut g = Gpu {
            kind: b,
            device_name: String::new(),
            device: dev_id,
            api,
            ctx: std::ptr::null_mut(),
            stream: std::ptr::null_mut(),
            funcs: RwLock::new(HashMap::new()),
            _lib: lib,
        };
        g.check(unsafe { (g.api.init)(0) }, "init")?;
        let mut name = [0 as c_char; 256];
        match b {
            Kind::Cuda => {
                let mut dev = 0i32;
                g.check(unsafe { (g.api.device_get.unwrap())(&mut dev, dev_id) }, "cuDeviceGet")?;
                let mut ctx: Ptr = std::ptr::null_mut();
                g.check(unsafe { (g.api.ctx_retain.unwrap())(&mut ctx, dev) }, "cuDevicePrimaryCtxRetain")?;
                g.check(unsafe { (g.api.ctx_set.unwrap())(ctx) }, "cuCtxSetCurrent")?;
                g.ctx = ctx;
                unsafe { (g.api.cu_name.unwrap())(name.as_mut_ptr(), 255, dev) };
            }
            Kind::Hip => {
                g.check(unsafe { (g.api.hip_set_device.unwrap())(dev_id) }, "hipSetDevice")?;
                unsafe { (g.api.hip_name.unwrap())(name.as_mut_ptr(), 255, dev_id) };
            }
        }
        g.device_name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
        let mut st: Ptr = std::ptr::null_mut();
        // non-blocking stream: no implicit synchronization with the legacy default stream libvship uses
        g.check(unsafe { (g.api.stream_create)(&mut st, 1) }, "stream create")?;
        g.stream = st;
        for (image, names) in super::images::modules(b) {
            g.load_module(image, names)?;
        }
        Ok(g)
    }

    /// load a device image (a cubin / fatbin for CUDA, a code object for HIP) into the context and make its kernels
    /// `names` launchable by name. Kernel names are global: they must not clash with flowdenoise's own.
    pub fn load_module(&self, image: &[u8], names: &[&'static str]) -> Result<()> {
        // module loaders want 8-byte aligned images; the caller's bytes (include_bytes!) have no alignment guarantee.
        // The aligned copy lives for the process, as the module does.
        let mut v: Vec<u64> = vec![0; image.len().div_ceil(8) + 1];
        let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, image.len()) };
        bytes.copy_from_slice(image);
        let aligned: &'static [u64] = Box::leak(v.into_boxed_slice());
        let mut m: Ptr = std::ptr::null_mut();
        self.check(unsafe { (self.api.module_load)(&mut m, aligned.as_ptr() as *const c_void) }, "module load")?;
        let mut funcs = self.funcs.write().unwrap();
        for &n in names {
            if funcs.contains_key(n) {
                bail!("kernel {n} is already loaded from another module");
            }
            let mut f: Ptr = std::ptr::null_mut();
            let cn = CString::new(n).unwrap();
            self.check(unsafe { (self.api.get_function)(&mut f, m, cn.as_ptr()) }, n)?;
            funcs.insert(n, f);
        }
        Ok(())
    }

    fn err_string(&self, r: i32) -> String {
        unsafe {
            if let Some(f) = self.api.cu_err {
                let mut p: *const c_char = std::ptr::null();
                f(r, &mut p);
                if !p.is_null() {
                    return CStr::from_ptr(p).to_string_lossy().into_owned();
                }
            }
            if let Some(f) = self.api.hip_err {
                let p = f(r);
                if !p.is_null() {
                    return CStr::from_ptr(p).to_string_lossy().into_owned();
                }
            }
        }
        format!("error {r}")
    }

    pub fn check(&self, r: i32, what: &str) -> Result<()> {
        if r != 0 {
            bail!("{what}: {} ({r})", self.err_string(r));
        }
        Ok(())
    }

    /// make the context current on this thread (CUDA; HIP keeps the device per thread too)
    pub fn bind_thread(&self) -> Result<()> {
        if let Some(f) = self.api.ctx_set {
            self.check(unsafe { f(self.ctx) }, "cuCtxSetCurrent")?;
        }
        if let Some(f) = self.api.hip_set_device {
            self.check(unsafe { f(self.device) }, "hipSetDevice")?;
        }
        Ok(())
    }

    pub fn alloc(&self, bytes: usize) -> Result<u64> {
        let mut p = 0u64;
        self.check(unsafe { (self.api.alloc)(&mut p, bytes.max(4)) }, "device alloc")
            .with_context(|| format!("{bytes} bytes"))?;
        Ok(p)
    }
    pub fn free(&self, p: u64) {
        unsafe { (self.api.free)(p) };
    }
    // `src` must be valid for `bytes` bytes (DBuf::write passes a slice)
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn htod(&self, dst: u64, src: *const c_void, bytes: usize) -> Result<()> {
        self.check(unsafe { (self.api.htod)(dst, src, bytes, self.stream) }, "copy to device")
    }
    // `dst` must be valid for `bytes` bytes (DBuf::read passes a slice)
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn dtoh(&self, dst: *mut c_void, src: u64, bytes: usize) -> Result<()> {
        self.check(unsafe { (self.api.dtoh)(dst, src, bytes, self.stream) }, "copy to host")
    }
    pub fn dtod(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        self.check(unsafe { (self.api.dtod)(dst, src, bytes, self.stream) }, "device copy")
    }
    pub fn memset(&self, dst: u64, v: u8, bytes: usize) -> Result<()> {
        self.check(unsafe { (self.api.memset8)(dst, v, bytes, self.stream) }, "memset")
    }
    pub fn sync(&self) -> Result<()> {
        self.check(unsafe { (self.api.stream_sync)(self.stream) }, "stream sync")
    }
    pub fn host_alloc(&self, bytes: usize) -> Result<*mut c_void> {
        let mut p: Ptr = std::ptr::null_mut();
        self.check(unsafe { (self.api.host_alloc)(&mut p, bytes.max(4), 0) }, "pinned alloc")?;
        Ok(p)
    }
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn host_free(&self, p: *mut c_void) {
        unsafe { (self.api.host_free)(p) };
    }

    /// launch kernel `name` with argument pointers (each points at the argument's value)
    pub fn launch_raw(&self, name: &str, grid: (u32, u32, u32), block: (u32, u32, u32), shmem: u32,
                      args: &mut [*mut c_void]) -> Result<()> {
        let f = *self.funcs.read().unwrap().get(name).ok_or_else(|| anyhow!("kernel {name} not loaded"))?;
        let r = unsafe {
            (self.api.launch)(f, grid.0, grid.1, grid.2, block.0, block.1, block.2, shmem, self.stream,
                              args.as_mut_ptr(), std::ptr::null_mut())
        };
        self.check(r, name)
    }
}
