//! flowdenoise: a motion-compensated temporal video denoiser on the GPU.
//!
//! Each frame is fused with its neighbours t - 3 ..= t + 3 after aligning them with a DIS-style patch motion search
//! and a Lanczos-3 warp. Neighbours are weighted per pixel by MVTools' DegrainWeight on an 8x8 SAD, with a
//! motion-dependent threshold and spike rejection; where too few frames agree, the result is blended toward a guided
//! filter, and coherent fine detail of the current frame is put back where the aligned neighbours agree that it is
//! texture, not noise.
//!
//! ```no_run
//! use flowdenoise::{Denoiser, Frame, Params};
//! # fn frames() -> Vec<Frame> { vec![] }
//! let mut den = Denoiser::new(1920, 1080, &Params::production(), Default::default())?;
//! let mut out = Vec::new();
//! for f in frames() {
//!     den.send_frame(&f)?;
//!     while let Some(d) = den.receive_frame()? {
//!         out.push(d);
//!     }
//! }
//! den.flush();
//! while let Some(d) = den.receive_frame()? {
//!     out.push(d);
//! }
//! # anyhow::Ok(())
//! ```
//!
//! Frames are 10-bit 4:2:0 ([`Frame`]); [`Frame`] converts from and to little-endian bytes (raw `yuv420p10le`, y4m
//! payloads), strided planes, 8-bit planes and, with the `v_frame` feature, `v_frame::frame::Frame<u16>`.
//!
//! The two presets ([`Preset`]): `production` (the encode pipeline's denoiser) and `measure` (GRAIL's
//! pinned measurement settings: the same without the detail boost).
use anyhow::{bail, ensure, Result};

pub use flowdenoise_core::{compiled_backends, Backend, Denoise, Params, Preset};

/// One 10-bit 4:2:0 frame: planes Y (width x height), U, V (width / 2 x height / 2) back to back, values 0 ..= 1023.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    width: usize,
    height: usize,
    data: Vec<u16>,
}

impl Frame {
    /// a black-level (all zero) frame; width and height must be even
    pub fn new(width: usize, height: usize) -> Result<Frame> {
        check_dims(width, height)?;
        Ok(Frame { width, height, data: vec![0; width * height * 3 / 2] })
    }

    /// from planar data (Y, U, V back to back, `width * height * 3 / 2` values)
    pub fn from_planar(width: usize, height: usize, data: Vec<u16>) -> Result<Frame> {
        check_dims(width, height)?;
        ensure!(data.len() == width * height * 3 / 2, "planar data: {} values for {width}x{height}", data.len());
        Ok(Frame { width, height, data })
    }

    /// from little-endian 16-bit samples (raw yuv420p10le, a y4m C420p10 frame payload)
    pub fn from_le_bytes(width: usize, height: usize, bytes: &[u8]) -> Result<Frame> {
        check_dims(width, height)?;
        ensure!(bytes.len() == width * height * 3, "yuv420p10le data: {} bytes for {width}x{height}", bytes.len());
        let data = bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        Ok(Frame { width, height, data })
    }

    /// from three strided 10-bit planes (strides in samples)
    pub fn from_planes(width: usize, height: usize, planes: [(&[u16], usize); 3]) -> Result<Frame> {
        let mut f = Frame::new(width, height)?;
        for (p, (src, stride)) in planes.into_iter().enumerate() {
            let (pw, ph) = f.plane_size(p);
            ensure!(stride >= pw && src.len() >= stride * (ph - 1) + pw, "plane {p}: too short for {pw}x{ph}, stride {stride}");
            let dst = f.plane_mut(p);
            for y in 0..ph {
                dst[y * pw..(y + 1) * pw].copy_from_slice(&src[y * stride..y * stride + pw]);
            }
        }
        Ok(f)
    }

    /// from three strided 8-bit planes, scaled to 10 bits (x 4, as a left shift by 2)
    pub fn from_8bit_planes(width: usize, height: usize, planes: [(&[u8], usize); 3]) -> Result<Frame> {
        let mut f = Frame::new(width, height)?;
        for (p, (src, stride)) in planes.into_iter().enumerate() {
            let (pw, ph) = f.plane_size(p);
            ensure!(stride >= pw && src.len() >= stride * (ph - 1) + pw, "plane {p}: too short for {pw}x{ph}, stride {stride}");
            let dst = f.plane_mut(p);
            for y in 0..ph {
                for x in 0..pw {
                    dst[y * pw + x] = (src[y * stride + x] as u16) << 2;
                }
            }
        }
        Ok(f)
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    /// all three planes, back to back
    pub fn data(&self) -> &[u16] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [u16] {
        &mut self.data
    }
    pub fn into_data(self) -> Vec<u16> {
        self.data
    }
    /// (width, height) of plane 0 (Y), 1 (U) or 2 (V)
    pub fn plane_size(&self, p: usize) -> (usize, usize) {
        if p == 0 {
            (self.width, self.height)
        } else {
            (self.width / 2, self.height / 2)
        }
    }
    fn plane_range(&self, p: usize) -> std::ops::Range<usize> {
        let ny = self.width * self.height;
        let nc = ny / 4;
        match p {
            0 => 0..ny,
            1 => ny..ny + nc,
            2 => ny + nc..ny + 2 * nc,
            _ => panic!("plane index {p}"),
        }
    }
    /// plane 0 (Y), 1 (U) or 2 (V), rows of plane_size(p).0 samples
    pub fn plane(&self, p: usize) -> &[u16] {
        &self.data[self.plane_range(p)]
    }
    pub fn plane_mut(&mut self, p: usize) -> &mut [u16] {
        let r = self.plane_range(p);
        &mut self.data[r]
    }
    pub fn y(&self) -> &[u16] {
        self.plane(0)
    }
    pub fn u(&self) -> &[u16] {
        self.plane(1)
    }
    pub fn v(&self) -> &[u16] {
        self.plane(2)
    }

    /// little-endian 16-bit samples (raw yuv420p10le, a y4m C420p10 frame payload), appended to `out`
    pub fn write_le_bytes(&self, out: &mut Vec<u8>) {
        out.reserve(self.data.len() * 2);
        for &v in &self.data {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut v = Vec::new();
        self.write_le_bytes(&mut v);
        v
    }
}

fn check_dims(width: usize, height: usize) -> Result<()> {
    if width == 0 || height == 0 || width % 2 == 1 || height % 2 == 1 {
        bail!("4:2:0 frames need even, non-zero dimensions (got {width}x{height})");
    }
    Ok(())
}

#[cfg(feature = "v_frame")]
mod vframe {
    use super::Frame;
    use anyhow::{ensure, Result};
    use std::num::{NonZeroU8, NonZeroUsize};
    use v_frame::chroma::ChromaSubsampling;
    use v_frame::frame::FrameBuilder;

    impl TryFrom<&v_frame::frame::Frame<u16>> for Frame {
        type Error = anyhow::Error;
        /// a 10-bit 4:2:0 v_frame frame (padding is dropped)
        fn try_from(f: &v_frame::frame::Frame<u16>) -> Result<Frame> {
            ensure!(f.bit_depth.get() == 10, "v_frame: bit depth {}, not 10", f.bit_depth);
            ensure!(f.subsampling == ChromaSubsampling::Yuv420, "v_frame: not 4:2:0");
            let (w, h) = (f.y_plane.width().get(), f.y_plane.height().get());
            let mut out = Frame::new(w, h)?;
            for p in 0..3 {
                let src = f.plane(p).ok_or_else(|| anyhow::anyhow!("v_frame: plane {p} missing"))?;
                let dst = out.plane_mut(p);
                for (d, s) in dst.iter_mut().zip(src.pixels()) {
                    *d = s;
                }
            }
            Ok(out)
        }
    }

    impl Frame {
        /// as a 10-bit 4:2:0 v_frame frame (no padding)
        pub fn to_v_frame(&self) -> Result<v_frame::frame::Frame<u16>> {
            let mut f = FrameBuilder::new(
                NonZeroUsize::new(self.width).unwrap(),
                NonZeroUsize::new(self.height).unwrap(),
                ChromaSubsampling::Yuv420,
                NonZeroU8::new(10).unwrap(),
            )
            .build::<u16>()?;
            for p in 0..3 {
                f.plane_mut(p).unwrap().copy_from_slice(self.plane(p))?;
            }
            Ok(f)
        }
    }
}

/// The denoiser: frames go in with [`send_frame`](Denoiser::send_frame), denoised frames come out of
/// [`receive_frame`](Denoiser::receive_frame) in order, `radius` frames later (the lookahead is internal); after
/// [`flush`](Denoiser::flush) the remaining frames come out.
pub struct Denoiser {
    inner: Box<dyn Denoise>,
    width: usize,
    height: usize,
    sent: usize,
    received: usize,
}

impl Denoiser {
    /// a denoiser for `width` x `height` frames with `params` on `backend` (the GPU is opened once per process)
    pub fn new(width: usize, height: usize, params: &Params, backend: Backend) -> Result<Denoiser> {
        Denoiser::with_device(width, height, params, backend, 0)
    }

    /// as [`new`](Denoiser::new) on GPU `device` (index among the backend's devices; Metal: the system default)
    pub fn with_device(width: usize, height: usize, params: &Params, backend: Backend, device: usize) -> Result<Denoiser> {
        check_dims(width, height)?;
        let inner = flowdenoise_core::open(width, height, params, backend, device)?;
        Ok(Denoiser { inner, width, height, sent: 0, received: 0 })
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    /// frames sent so far
    pub fn frames_in(&self) -> usize {
        self.sent
    }
    /// frames received so far
    pub fn frames_out(&self) -> usize {
        self.received
    }

    /// the next input frame
    pub fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        ensure!(
            frame.width == self.width && frame.height == self.height,
            "frame {}x{} sent to a {}x{} denoiser",
            frame.width,
            frame.height,
            self.width,
            self.height
        );
        self.inner.push(&frame.data)?;
        self.sent += 1;
        Ok(())
    }

    /// end of the input: the frames still held for lookahead become available
    pub fn flush(&mut self) {
        self.inner.finish();
    }

    /// the next denoised frame, if it can be produced now
    pub fn receive_frame(&mut self) -> Result<Option<Frame>> {
        let mut f = Frame::new(self.width, self.height)?;
        Ok(if self.receive_frame_into(&mut f)? { Some(f) } else { None })
    }

    /// the next denoised frame into `out` (no allocation); false if none can be produced now
    pub fn receive_frame_into(&mut self, out: &mut Frame) -> Result<bool> {
        ensure!(out.width == self.width && out.height == self.height, "output frame size");
        if self.inner.ready().is_none() {
            return Ok(false);
        }
        self.inner.produce(&mut out.data)?;
        self.received += 1;
        Ok(true)
    }
}
