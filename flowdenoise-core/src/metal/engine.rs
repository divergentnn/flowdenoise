//! The denoiser's host-side orchestration on Metal: the Python reference's Engine.ingest / Engine.process, step for
//! step, as the CUDA backend's (src/cuda/engine.rs); the kernels are kernels/metal/*.metal. Differences: the motion
//! search takes the neighbours' pyramid levels by pointer (no stacked copy) and runs one SIMD group per patch row.
use super::buf::{launch, launch_n, DBuf};
use super::ops::Select;
use crate::{args, Arg, Denoise, Params};
use anyhow::Result;
use std::collections::{HashMap, VecDeque};


/// one input frame on the GPU: luma, chroma (2 planes), prefiltered luma, motion pyramid
pub struct Ingested {
    yc: DBuf,
    pc: DBuf,
    levels: Vec<(DBuf, usize, usize)>,
    n: usize,
}
impl Ingested {
    fn y(&self) -> u64 {
        self.yc.ptr
    }
    fn c(&self) -> u64 {
        self.yc.ptr + 4 * self.n as u64
    }
}

/// neighbour pointers the motion kernels take (flowdenoise.metal MAX_NEIGHBOURS): radius 8
const MAX_NEIGHBOURS: usize = 16;

struct Taps {
    k: DBuf,
    len: usize,
}

pub struct Engine {
    pub w: usize,
    pub h: usize,
    sc: usize,
    p: Params,
    taps: HashMap<&'static str, Taps>,
    pool: HashMap<&'static str, DBuf>,
    sel: Select,
}

/// the range check's thresholds (t0, t1) for a fixed sigma: flowdegrain2 range_sigma() with --range-sigma S > 0,
/// sigma_d = float32(sqrt(2) S), then sigma_d * float32(k) and sigma_d * float32(k + soft) in float32
fn range_fixed_thresholds(p: &Params) -> [f32; 2] {
    let sd = (2f64.sqrt() * p.range_sigma) as f32;
    [sd * p.range_k as f32, sd * (p.range_k + p.range_soft) as f32]
}

fn gauss_taps(sigma: f64) -> Vec<f32> {
    // flowdegrain2 Engine.gauss: t = arange(-int(3 sigma), int(3 sigma) + 1); exp(-t t / (2 sigma sigma)); normalized
    let r = (3.0 * sigma) as i64;
    let d = 2.0 * sigma * sigma;
    let k: Vec<f64> = (-r..=r).map(|t| { let t = t as f64; (-t * t / d).exp() }).collect();
    let s: f64 = k.iter().sum();
    k.iter().map(|v| (v / s) as f32).collect()
}

impl Engine {
    pub fn new(w: usize, h: usize, p: Params) -> Result<Engine> {
        p.validate()?;
        p.check_size(w, h)?;
        let sc = p.motion_scale(w);
        let mut taps = HashMap::new();
        let mut add = |name: &'static str, k: Vec<f32>| -> Result<()> {
            taps.insert(name, Taps { len: k.len(), k: DBuf::upload(&k)? });
            Ok(())
        };
        let ing: Vec<f32> = [0.0044f64, 0.054, 0.242, 0.399, 0.242, 0.054, 0.0044].iter().map(|v| (v / 0.9998) as f32).collect();
        add("ingest", ing)?;
        add("pre", [1.0f64, 4.0, 6.0, 4.0, 1.0].iter().map(|v| (v / 16.0) as f32).collect())?;
        add("sad", vec![(1.0f64 / 8.0) as f32; 8])?;
        add("g1", gauss_taps(1.0))?;
        add("g3", gauss_taps(3.0))?;
        add("g6", gauss_taps(6.0))?;
        let r = p.fallback_radius;
        add("box_y", vec![(1.0 / (2 * r + 1) as f64) as f32; 2 * r + 1])?;
        let rc = (r / 2).max(1);
        add("box_c", vec![(1.0 / (2 * rc + 1) as f64) as f32; 2 * rc + 1])?;
        if p.fb_gate > 0.0 {
            add("g_fb_sg", gauss_taps(p.fb_sg))?;
            add("g_fb_win", gauss_taps(p.fb_win))?;
        }
        let mut pool = HashMap::new();
        if p.range_k > 0.0 && p.range_sigma > 0.0 {
            pool.insert("range_fixed", DBuf::upload(&range_fixed_thresholds(&p))?);
        }
        Ok(Engine { w, h, sc, p, taps, pool, sel: Select::new()? })
    }

    /// scratch buffer `name` of at least `bytes` (reused across frames)
    fn buf(&mut self, name: &'static str, bytes: usize) -> Result<u64> {
        if self.pool.get(name).map(|b| b.bytes < bytes).unwrap_or(true) {
            self.pool.insert(name, DBuf::new(bytes)?);
        }
        Ok(self.pool[name].ptr)
    }

    /// Engine.filt: separable taps along x (axis 0 kernel) then y, over `planes` planes of w x h
    #[allow(clippy::too_many_arguments)]
    fn filt(&mut self, src: u64, dst: u64, w: usize, h: usize, planes: usize, taps: &'static str, zero: bool,
            pre: bool) -> Result<()> {
        let tmp = self.buf("filt_tmp", w * h * planes * 4)?;
        let t = &self.taps[taps];
        let (kp, len) = (t.k.ptr, t.len);
        let grid = (w.div_ceil(32) as u32, h.div_ceil(8) as u32, planes as u32);
        let (z, pr) = (zero as i32, pre as i32);
        match len {
            3 | 5 | 7 | 8 | 19 | 37 => {
                launch(&format!("separable_filter_{len}_x"), grid, (32, 8, 1), 0, args![src, tmp, w as i32, h as i32, kp, z, pr])?;
                launch(&format!("separable_filter_{len}_y"), grid, (32, 8, 1), 0, args![tmp, dst, w as i32, h as i32, kp, z, pr])
            }
            _ => {
                let n = (w * h * planes) as i32;
                launch_n("separable_filter_any", n as usize, args![src, tmp, w as i32, h as i32, n, kp, len as i32, 0i32, z])?;
                launch_n("separable_filter_any", n as usize, args![tmp, dst, w as i32, h as i32, n, kp, len as i32, 1i32, z])
            }
        }
    }

    fn down(&self, src: u64, dst: u64, w: usize, h: usize) -> Result<()> {
        launch_n("downsample_2x2", (w / 2) * (h / 2), args![src, dst, w as i32, h as i32])
    }

    /// Engine.ingest of one 10-bit 4:2:0 frame (w x h, planes Y U V)
    pub fn ingest(&mut self, raw: &[u16]) -> Result<Ingested> {
        let (w, h) = (self.w, self.h);
        let n = w * h;
        let nc = (w / 2) * (h / 2);
        assert_eq!(raw.len(), n + 2 * nc);
        let up = self.buf("raw", raw.len() * 2)?;
        super::driver::gpu().htod(up, raw.as_ptr() as *const _, raw.len() * 2)?;
        let yc = DBuf::f32(n + 2 * nc)?;
        launch_n("u16_to_float", n + 2 * nc, args![up, &yc, (n + 2 * nc) as i64])?;
        let g0 = self.buf("ingest_g", n * 4)?;
        self.filt(yc.ptr, g0, w, h, 1, "ingest", false, false)?;
        let (mut lw, mut lh) = (w, h);
        let mut g = g0;
        if self.sc == 2 {
            let g2 = self.buf("ingest_g2", (w / 2) * (h / 2) * 4)?;
            self.down(g0, g2, w, h)?;
            g = g2;
            lw = w / 2;
            lh = h / 2;
        }
        let l0 = DBuf::f32(lw * lh)?;
        launch_n("quantize_8bit", lw * lh, args![g, &l0, (lw * lh) as i64])?;
        let mut levels = vec![(l0, lw, lh)];
        for level in 1..self.p.levels {
            let (pw, ph) = (levels[level - 1].1, levels[level - 1].2);
            let nxt = DBuf::f32((pw / 2) * (ph / 2))?;
            self.down(levels[level - 1].0.ptr, nxt.ptr, pw, ph)?;
            if self.p.round_motion && level >= self.p.finest {
                launch_n("round_to_integer", (pw / 2) * (ph / 2), args![&nxt, &nxt, ((pw / 2) * (ph / 2)) as i64])?;
            }
            levels.push((nxt, pw / 2, ph / 2));
        }
        let pc = DBuf::f32(n)?;
        self.filt(yc.ptr, pc.ptr, w, h, 1, "pre", false, true)?;
        Ok(Ingested { yc, pc, levels, n })
    }

    /// Engine.raster_batch: dense flow (D, 2, h_f, w_f) at the finest level for each neighbour
    fn raster_batch(&mut self, cur: &Ingested, nbrs: &[&Ingested]) -> Result<(u64, usize, usize)> {
        let d = nbrs.len();
        let mut pred: Option<(u64, usize, usize)> = None;
        let mut flip = false;
        for level in (self.p.finest..self.p.levels).rev() {
            let (cbuf, w, h) = (&cur.levels[level].0, cur.levels[level].1, cur.levels[level].2);
            let c = cbuf.ptr;
            // the neighbours' levels by pointer (the CUDA version copies them into one stack): MAX_NEIGHBOURS of them
            assert!(d <= MAX_NEIGHBOURS, "at most {MAX_NEIGHBOURS} neighbours");
            let ns: Vec<Arg> = (0..MAX_NEIGHBOURS).map(|j| Arg::P(nbrs.get(j).map(|nb| nb.levels[level].0.ptr).unwrap_or(0))).collect();
            let bw = 1.max(1 + (w as i64 - 8).div_euclid(4)) as usize;
            let bh = 1.max(1 + (h as i64 - 8).div_euclid(4)) as usize;
            let fwd = self.buf("r_fwd", d * 2 * bw * bh * 4)?;
            let back = self.buf("r_back", d * 2 * bw * bh * 4)?;
            let (pp, pw, ph) = pred.unwrap_or((0, 0, 0));
            let it = self.p.iterations / 2;
            let grid = (bh as u32, d as u32, 1);
            // one SIMD group (32 lanes, two pixels each) per patch row
            let with_nbrs = |tail: &[Arg]| -> Vec<Arg> { [&[Arg::P(c)][..], &ns[..], tail].concat() };
            launch("motion_search_sweep", grid, (32, 1, 1), 0,
                   &with_nbrs(args![w as i32, h as i32, pp, pw as i32, ph as i32, 0u64, fwd, bw as i32, bh as i32, 0i32, it]))?;
            launch("motion_search_sweep", grid, (32, 1, 1), 0,
                   &with_nbrs(args![w as i32, h as i32, pp, pw as i32, ph as i32, fwd, back, bw as i32, bh as i32, 1i32, it]))?;
            // ping-pong between two dense buffers: the next level reads `pred` while writing the other
            let name = if flip { "r_pred_a" } else { "r_pred_b" };
            flip = !flip;
            let np = self.buf(name, d * 2 * w * h * 4)?;
            launch("densify_flow", ((w * h).div_ceil(256) as u32, d as u32, 1), (256, 1, 1), 0,
                   &with_nbrs(args![back, np, w as i32, h as i32, bw as i32, bh as i32]))?;
            pred = Some((np, w, h));
        }
        Ok(pred.unwrap())
    }

    /// flowdegrain2 noise(): per plane, max((1.4826 median |hp[::53]|)^2, 0.001) * mult -> var[plane] (device);
    /// returns the device float that holds the last plane's median (valid until the next call)
    fn noise(&mut self, hp: u64, plane: usize, planes: usize, var: u64, mult: f32) -> Result<u64> {
        let m = plane.div_ceil(53);
        let g = self.buf("noise_g", m * planes * 4)?;
        launch_n("gather_strided", m * planes, args![hp, g, plane as i64, 53i64, m as i64, planes as i32, 1i32])?;
        let med = self.buf("noise_med", 4)?;
        for p in 0..planes {
            self.sel.kth(g + (p * m * 4) as u64, m, ((m - 1) / 2) as u32, med)?;
            launch("noise_variance", (1, 1, 1), (32, 1, 1), 0, args![med, var + (p * 4) as u64, mult])?;
        }
        Ok(med)
    }

    /// flowdegrain2 stability(): the stable-texture share s of the fused luma x -> share (luma plane):
    /// hp = x - G(x, fb_sg); e = G(hp^2, fb_win); s = clamp((e / max(nv fb_q sq / ws^2, 1e-6) - 1) / (fb_gate - 1))
    fn stability(&mut self, x: u64, nv: u64, sq: u64, ws: u64, share: u64) -> Result<()> {
        let (w, h) = (self.w, self.h);
        let n = w * h;
        let g = self.buf("st_g", n * 4)?;
        self.filt(x, g, w, h, 1, "g_fb_sg", false, false)?;
        let hp = self.buf("st_hp", n * 4)?;
        launch_n("subtract", n, args![x, g, hp, n as i64])?;
        let hp2 = self.buf("st_hp2", n * 4)?;
        launch_n("square", n, args![hp, hp2, n as i64])?;
        let e = self.buf("st_e", n * 4)?;
        self.filt(hp2, e, w, h, 1, "g_fb_win", false, false)?;
        let span = (self.p.fb_gate - 1.0).max(1e-3) as f32;
        launch_n("stability_share", n, args![e, nv, self.p.fb_q as f32, sq, ws, span, share, n as i64])
    }

    /// flowdegrain2 range_sigma() without --range-sigma, on the device: sigma_d from the median of dd[::53] over
    /// mid-luma pixels of the current frame y (all of dd[::53] if fewer than 1000), and the thresholds
    /// (sigma_d k, sigma_d (k + soft)) -> the returned device buffer (two floats)
    fn range_thresholds(&mut self, dd: u64, y: u64) -> Result<u64> {
        let n = self.w * self.h;
        let m = n.div_ceil(53);
        let samples = self.buf("rg_samples", m * 4)?;
        let count = self.buf("rg_count", 4)?;
        super::driver::gpu().memset(count, 0, 4)?;
        launch_n("range_sample_gather", m, args![dd, y, samples, count, 53i64, m as i64])?;
        launch_n("range_sample_fallback", m, args![dd, samples, count, 1000u32, 53i64, m as i64])?;
        let med = self.buf("rg_median", 4)?;
        self.sel.median_counted(samples, m, count, 1000, m as u32, med)?;
        let t = self.buf("rg_thresholds", 8)?;
        let (k, k_soft) = (self.p.range_k as f32, (self.p.range_k + self.p.range_soft) as f32);
        launch("range_thresholds", (1, 1, 1), (32, 1, 1), 0, args![med, k, k_soft, t])?;
        Ok(t)
    }

    /// flowdegrain2 fallback(): guided filter blend of x (planes x n) with alpha (broadcast modulo am)
    #[allow(clippy::too_many_arguments)]
    fn fallback(&mut self, x: u64, out: u64, alpha: u64, am: usize, eps: u64, w: usize, h: usize, planes: usize,
                taps: &'static str, pfx: &'static [&'static str; 6]) -> Result<()> {
        let n = w * h * planes;
        let m = self.buf(pfx[0], n * 4)?;
        let x2 = self.buf(pfx[1], n * 4)?;
        let m2 = self.buf(pfx[2], n * 4)?;
        let ga = self.buf(pfx[3], n * 4)?;
        let gb = self.buf(pfx[4], n * 4)?;
        let tmp = self.buf(pfx[5], n * 4)?;
        self.filt(x, m, w, h, planes, taps, false, false)?;
        launch_n("square", n, args![x, x2, n as i64])?;
        self.filt(x2, m2, w, h, planes, taps, false, false)?;
        launch_n("guided_coefficients", n, args![m, m2, eps, ga, gb, n as i64, (w * h) as i64])?;
        self.filt(ga, m, w, h, planes, taps, false, false)?;     // box(ga)
        self.filt(gb, tmp, w, h, planes, taps, false, false)?;   // box(gb)
        launch_n("guided_blend", n, args![x, alpha, m, tmp, out, n as i64, am as i64])
    }

    /// Engine.process: the denoised frame (int16 luma + chroma) into `out`
    pub fn process(&mut self, cur: &Ingested, nbrs: &[(i32, &Ingested)], out: &DBuf) -> Result<()> {
        let (w, h) = (self.w, self.h);
        let n = w * h;
        let (w2, h2) = (w / 2, h / 2);
        let nc = w2 * h2;
        let d = nbrs.len();
        let p = self.p.clone();
        let flows = if d > 0 {
            let refs: Vec<&Ingested> = nbrs.iter().map(|x| x.1).collect();
            Some(self.raster_batch(cur, &refs)?)
        } else {
            None
        };
        let y = cur.y();
        let g1 = self.buf("g1", n * 4)?;
        self.filt(y, g1, w, h, 1, "g1", false, false)?;
        let hc = self.buf("hc", n * 4)?;
        launch_n("subtract", n, args![y, g1, hc, n as i64])?;
        let acc = self.buf("acc", n * 4)?;
        let ws = self.buf("ws", n * 4)?;
        let sq = self.buf("sq", n * 4)?;
        let hs = self.buf("hs", n * 4)?;
        let ha = self.buf("ha", n * 4)?;
        let ac = self.buf("ac", 2 * nc * 4)?;
        super::driver::gpu().dtod(ac, cur.c(), 2 * nc * 4)?;
        let wc = self.buf("wc", nc * 4)?;
        if !self.pool.contains_key("ones_c") {
            self.pool.insert("ones_c", DBuf::upload(&vec![1.0f32; nc])?);
        }
        super::driver::gpu().dtod(wc, self.pool["ones_c"].ptr, nc * 4)?;
        let warped = self.buf("warped", d.max(1) * n * 4)?;
        let weights = self.buf("weights", d.max(1) * n * 4)?;
        let high = self.buf("high", d.max(1) * n * 4)?;
        if let Some((fs, fw, fh)) = flows {
            let scale = (self.sc * (1 << p.finest)) as f32;
            let dd = self.buf("dd", n * 4)?;
            let speed = self.buf("speed", n * 4)?;
            let t1 = self.buf("t1", n * 4)?;
            let t2 = self.buf("t2", n * 4)?;
            let sad = self.buf("sad", n * 4)?;
            let spread = self.buf("spread", n * 4)?;
            let low = self.buf("low", n * 4)?;
            for (j, (dist, other)) in nbrs.iter().enumerate() {
                let f = fs + (j * 2 * fw * fh * 4) as u64;
                let yw = warped + (j * n * 4) as u64;
                launch_n("warp_luma", n, args![other.y(), y, f, yw, dd, speed, w as i32, h as i32, fw as i32, fh as i32, scale])?;
                self.filt(yw, t1, w, h, 1, "pre", false, true)?;
                launch_n("abs_diff", n, args![t1, cur.pc.ptr, t2, n as i64])?;
                self.filt(t2, sad, w, h, 1, "sad", true, false)?;
                self.filt(dd, spread, w, h, 1, "g6", false, false)?;
                self.filt(yw, low, w, h, 1, "g1", false, false)?;
                let wt = weights + (j * n * 4) as u64;
                let hp = high + (j * n * 4) as u64;
                launch_n("temporal_weight", n, args![yw, low, sad, dd, spread, speed, wt, hp, w as i32, h as i32,
                                                 p.thsad, dist.abs() as f32, p.fast_th, p.spike])?;
                if p.range_k > 0.0 {
                    let t = match self.pool.get("range_fixed") {
                        Some(b) => b.ptr,
                        None => self.range_thresholds(dd, y)?,
                    };
                    launch_n("range_check", n, args![dd, t, wt, n as i64])?;
                }
                launch_n("warp_accumulate_chroma", nc, args![other.c(), f, wt, ac, wc, w2 as i32, h2 as i32, fw as i32, fh as i32, scale])?;
            }
        }
        launch_n("fuse_frames", n, args![y, hc, warped, weights, high, acc, ws, sq, hs, ha, n as i32, d as i32])?;
        let oy = self.buf("oy", n * 4)?;
        launch_n("divide_broadcast", n, args![acc, ws, oy, n as i64, n as i64])?;
        let oc = self.buf("oc", 2 * nc * 4)?;
        launch_n("divide_broadcast", 2 * nc, args![ac, wc, oc, (2 * nc) as i64, nc as i64])?;
        let (oy2, oc2) = if p.fallback > 1.0 {
            let alpha = self.buf("alpha", n * 4)?;
            launch_n("fallback_alpha", n, args![ws, sq, p.fallback, alpha, n as i64])?;
            let var_y = self.buf("var_y", 4)?;
            let med_y = self.noise(hc, n, 1, var_y, p.fallback_eps)?;
            let alpha_c = self.buf("alpha_c", nc * 4)?;
            self.down(alpha, alpha_c, w, h)?;
            if p.fb_gate > 0.0 {
                // stability gate (preset cel): alpha *= 1 - s; the chroma factor is down(alpha) (taken before the
                // gate) times 1 - down(s)
                let nv = self.buf("fb_nv", 4)?;
                launch("noise_variance", (1, 1, 1), (32, 1, 1), 0, args![med_y, nv, 1.0f32])?;
                let s = self.buf("fb_s", n * 4)?;
                self.stability(oy, nv, sq, ws, s)?;
                launch_n("scale_by_complement", n, args![alpha, s, alpha, n as i64])?;
                let s_c = self.buf("fb_s_c", nc * 4)?;
                self.down(s, s_c, w, h)?;
                launch_n("scale_by_complement", nc, args![alpha_c, s_c, alpha_c, nc as i64])?;
            }
            let oy2 = self.buf("oy2", n * 4)?;
            self.fallback(oy, oy2, alpha, n, var_y, w, h, 1, "box_y", &["fy_m", "fy_x2", "fy_m2", "fy_ga", "fy_gb", "fy_t"])?;
            // chroma: hp = c - gauss(c, 1) per plane
            let gc = self.buf("gc", 2 * nc * 4)?;
            self.filt(cur.c(), gc, w2, h2, 2, "g1", false, false)?;
            let hpc = self.buf("hpc", 2 * nc * 4)?;
            launch_n("subtract", 2 * nc, args![cur.c(), gc, hpc, (2 * nc) as i64])?;
            let var_c = self.buf("var_c", 8)?;
            self.noise(hpc, nc, 2, var_c, p.fallback_eps)?;
            let oc2 = self.buf("oc2", 2 * nc * 4)?;
            self.fallback(oc, oc2, alpha_c, nc, var_c, w2, h2, 2, "box_c", &["fc_m", "fc_x2", "fc_m2", "fc_ga", "fc_gb", "fc_t"])?;
            (oy2, oc2)
        } else {
            (oy, oc)
        };
        // detail gate
        let low = self.buf("f_low", n * 4)?;
        self.filt(oy2, low, w, h, 1, "g1", false, false)?;
        let en = self.buf("f_en", n * 4)?;
        launch_n("squared_diff", n, args![oy2, low, en, n as i64])?;
        let e = self.buf("f_e", n * 4)?;
        self.filt(en, e, w, h, 1, "g3", false, false)?;
        let m97 = n.div_ceil(97);
        let es = self.buf("f_es", m97 * 4)?;
        launch_n("gather_strided", m97, args![e, es, n as i64, 97i64, m97 as i64, 1i32, 0i32])?;
        let ef = self.buf("f_ef", 4)?;
        let qtmp = self.pool.remove("f_qtmp").map(Ok).unwrap_or_else(|| DBuf::new(8))?;
        self.sel.quantile(es, m97, 0.25, ef, &qtmp)?;
        self.pool.insert("f_qtmp", qtmp);
        let ahs = self.buf("f_ahs", n * 4)?;
        launch_n("absolute", n, args![hs, ahs, n as i64])?;
        let ghs = self.buf("f_ghs", n * 4)?;
        self.filt(ahs, ghs, w, h, 1, "g1", false, false)?;
        let gha = self.buf("f_gha", n * 4)?;
        self.filt(ha, gha, w, h, 1, "g1", false, false)?;
        launch_n("detail_gate_luma", n, args![oy2, low, hc, ghs, gha, sq, ws, e, ef, p.detail_gate, p.detail_boost, out.ptr, n as i64])?;
        launch_n("pack_10bit", 2 * nc, args![oc2, out.ptr + (n * 2) as u64, (2 * nc) as i64])
    }
}

/// streaming wrapper: the reference's main loop (frame t is processed once frames up to t + radius are ingested)
pub struct Denoiser {
    pub eng: Engine,
    frames: VecDeque<(usize, Ingested)>,
    count: usize,
    next: usize,
    eof: bool,
    radius: usize,
    /// output of `Denoise::produce` (int16, w * h * 3 / 2)
    out: Option<DBuf>,
}

impl Denoiser {
    pub fn new(w: usize, h: usize, p: Params) -> Result<Denoiser> {
        let radius = p.radius;
        Ok(Denoiser { eng: Engine::new(w, h, p)?, frames: VecDeque::new(), count: 0, next: 0, eof: false, radius, out: None })
    }
    pub fn push(&mut self, raw: &[u16]) -> Result<()> {
        let ing = self.eng.ingest(raw)?;
        self.frames.push_back((self.count, ing));
        self.count += 1;
        Ok(())
    }
    pub fn finish(&mut self) {
        self.eof = true;
    }
    /// index of the next frame that can be produced now
    pub fn ready(&self) -> Option<usize> {
        if self.next < self.count && (self.eof || self.count > self.next + self.radius) {
            Some(self.next)
        } else {
            None
        }
    }
    /// produce the next frame into `out` (int16, w*h*3/2) if `want`, else skip it (its neighbours stay available)
    pub fn step(&mut self, want: bool, out: &DBuf) -> Result<usize> {
        let t = self.next;
        if want {
            let r = self.radius as i64;
            let cur = self.frames.iter().position(|(k, _)| *k == t).expect("current frame");
            let mut nb: Vec<(i32, &Ingested)> = vec![];
            for dlt in -r..=r {
                if dlt == 0 {
                    continue;
                }
                let k = t as i64 + dlt;
                if let Some((_, f)) = self.frames.iter().find(|(i, _)| *i as i64 == k) {
                    nb.push((dlt as i32, f));
                }
            }
            // disjoint fields: the frames are borrowed while the engine is borrowed mutably
            self.eng.process(&self.frames[cur].1, &nb, out)?;
        }
        while let Some((k, _)) = self.frames.front() {
            if (*k as i64) < t as i64 - self.radius as i64 + 1 {
                self.frames.pop_front();
            } else {
                break;
            }
        }
        self.next += 1;
        Ok(t)
    }
}

impl Denoise for Denoiser {
    fn width(&self) -> usize {
        self.eng.w
    }
    fn height(&self) -> usize {
        self.eng.h
    }
    fn push(&mut self, frame: &[u16]) -> Result<()> {
        Denoiser::push(self, frame)
    }
    fn finish(&mut self) {
        Denoiser::finish(self)
    }
    fn ready(&self) -> Option<usize> {
        Denoiser::ready(self)
    }
    fn skip(&mut self) -> Result<usize> {
        let out = self.output()?;
        let t = self.step(false, &out);
        self.out = Some(out);
        t
    }
    fn produce(&mut self, frame: &mut [u16]) -> Result<usize> {
        let n = self.eng.w * self.eng.h * 3 / 2;
        assert_eq!(frame.len(), n, "output frame size");
        let out = self.output()?;
        let r = self.step(true, &out).and_then(|t| out.read(frame).map(|_| t));
        self.out = Some(out);
        r
    }
}

impl Denoiser {
    /// the produce() output buffer (taken out of self while step borrows self)
    fn output(&mut self) -> Result<DBuf> {
        match self.out.take() {
            Some(b) => Ok(b),
            None => DBuf::new(self.eng.w * self.eng.h * 3),
        }
    }
}
