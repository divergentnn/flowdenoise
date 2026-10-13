//! Denoiser parameters and the two presets.
use anyhow::{bail, Result};
use std::fmt;
use std::str::FromStr;

/// Every parameter of the denoiser. The two presets differ only in `detail_boost`.
///
/// Units: luma in 10-bit codes for the image arithmetic, but `thsad` is an MVTools-style SAD in 8-bit units over an
/// 8x8 block (MVTools' thSAD); distances in pixels; the motion pyramid works on 8-bit luma.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// temporal radius: frames t - radius ..= t + radius are fused (at most 8; the Metal backend takes at most 3)
    pub radius: usize,
    /// SAD threshold of the temporal weights (MVTools thSAD, 8x8 block, 8-bit units): weight 0 at and above it
    pub thsad: f32,
    /// motion is estimated at 1 / flow_scale of the input resolution; 0 = automatic (2 above 2000 px width, else 1)
    pub flow_scale: usize,
    /// motion pyramid depth (levels 0 ..= levels - 1, each half the size of the one before)
    pub levels: usize,
    /// finest pyramid level the motion search runs on (2 = a quarter of the motion resolution); the flow is
    /// interpolated from there
    pub finest: usize,
    /// gradient-descent iterations per patch, split over the two propagation sweeps
    pub iterations: i32,
    /// round the motion pyramid to whole 8-bit codes (as the 8-bit DIS input it reproduces)
    pub round_motion: bool,
    /// detail gate threshold: temporal coherence of the current frame's fine detail above which it is put back
    pub detail_gate: f32,
    /// gain of the restored detail where the gate is fully open (1 = put back as it was)
    pub detail_boost: f32,
    /// spike rejection: a neighbour's weight falls from 1 to 0 as its largest difference from the current frame in a
    /// 3x3 window goes from 1.25 x spike to 1.875 x spike times the local mean difference (Gaussian, sigma 6)
    pub spike: f32,
    /// motion-dependent threshold: thsad falls with the flow speed per frame of distance, thsad / (1 + speed / fast_th)
    pub fast_th: f32,
    /// guided fallback: below this effective frame count (sum w)^2 / sum w^2 the result is blended toward a guided
    /// filter of itself; at or below 1 the fallback is off
    pub fallback: f32,
    /// guided filter regularisation, in units of the frame's noise variance
    pub fallback_eps: f32,
    /// guided filter box radius on luma (chroma: half, at least 1)
    pub fallback_radius: usize,
}

/// Named parameter sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Preset {
    /// the encode pipeline's settings (the Python reference's defaults): detail restored with a 1.3 boost
    #[default]
    Production,
    /// GRAIL's pinned measurement settings: as production, but restored detail is not boosted (detail_boost 1)
    Measure,
}

impl Preset {
    pub const ALL: [Preset; 2] = [Preset::Production, Preset::Measure];

    pub fn name(self) -> &'static str {
        match self {
            Preset::Production => "production",
            Preset::Measure => "measure",
        }
    }
}

impl fmt::Display for Preset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Preset {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Preset> {
        match s {
            "production" | "prod" => Ok(Preset::Production),
            "measure" | "measurement" => Ok(Preset::Measure),
            x => bail!("unknown preset {x:?} (production or measure)"),
        }
    }
}

impl Default for Params {
    fn default() -> Params {
        Params::production()
    }
}

impl Params {
    /// the encode pipeline's denoiser: the Python reference (flowdegrain2.py) with its default arguments
    pub fn production() -> Params {
        Params {
            radius: 3,
            thsad: 450.0,
            flow_scale: 0,
            levels: 7,
            finest: 2,
            iterations: 12,
            round_motion: true,
            detail_gate: 0.6,
            detail_boost: 1.3,
            spike: 3.0,
            fast_th: 8.0,
            fallback: 4.0,
            fallback_eps: 2.0,
            fallback_radius: 3,
        }
    }

    /// GRAIL's measurement denoiser: production with detail_boost 1 (flowdegrain2.py --detail-boost 1)
    pub fn measure() -> Params {
        Params { detail_boost: 1.0, ..Params::production() }
    }

    pub fn preset(p: Preset) -> Params {
        match p {
            Preset::Production => Params::production(),
            Preset::Measure => Params::measure(),
        }
    }

    /// the preset these parameters equal, if any
    pub fn matches_preset(&self) -> Option<Preset> {
        Preset::ALL.into_iter().find(|&p| *self == Params::preset(p))
    }

    /// the ranges the Python reference accepts
    pub fn validate(&self) -> Result<()> {
        let ok = self.radius <= 8
            && self.finest < self.levels
            && self.levels <= 7
            && self.iterations >= 0
            && self.flow_scale <= 2
            && (0.0..1.0).contains(&self.detail_gate)
            && self.fast_th > 0.0
            && self.spike > 0.0
            && self.thsad > 0.0
            && self.fallback_eps > 0.0
            && self.fallback_radius >= 1;
        if !ok {
            bail!("invalid denoiser parameters: {self:?}");
        }
        Ok(())
    }

    /// 1 / the resolution of the motion search input for a frame `width` pixels wide
    pub fn motion_scale(&self, width: usize) -> usize {
        match self.flow_scale {
            0 => {
                if width > 2000 {
                    2
                } else {
                    1
                }
            }
            s => s,
        }
    }

    /// whether a `width` x `height` frame can be denoised with these parameters
    pub fn check_size(&self, width: usize, height: usize) -> Result<()> {
        if width % 2 == 1 || height % 2 == 1 || width.min(height) < 32 {
            bail!("denoiser: need even dimensions >= 32 (got {width}x{height})");
        }
        if width.min(height) / ((1 << (self.levels - 1)) * self.motion_scale(width)) < 2 {
            bail!("denoiser: image {width}x{height} too small for the motion pyramid");
        }
        Ok(())
    }
}
