//! Denoiser parameters and the presets.
use anyhow::{bail, Result};
use std::fmt;
use std::str::FromStr;

/// Every parameter of the denoiser. `production` and `measure` differ only in `detail_boost`; `cel` (2D cel animation)
/// fuses more frames and gates the guided fallback by stability.
///
/// Units: luma in 10-bit codes for the image arithmetic, but `thsad` is an MVTools-style SAD in 8-bit units over an
/// 8x8 block (MVTools' thSAD); distances in pixels; the motion pyramid works on 8-bit luma.
///
/// The floating-point fields after `fallback_radius` are `f64`: the reference computes with them in double precision (Gaussian taps,
/// sums of two parameters) before rounding to float, and parses them as doubles.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// temporal radius: frames t - radius ..= t + radius are fused (at most 8)
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
    /// stability gate on the guided fallback (0 = off): the fallback's blend factor is multiplied by 1 - s, with
    /// s = clamp((R - 1) / (fb_gate - 1), 0, 1) and R the local energy of the fused frame's high-pass over the energy
    /// the averaged grain would leave there. Stable texture (R well above 1) keeps the temporal result.
    pub fb_gate: f64,
    /// stability gate: scale of the expected residual grain energy, var * fb_q / effective frame count
    pub fb_q: f64,
    /// stability gate: sigma of the Gaussian whose residual is the high-pass, x - G(x, fb_sg)
    pub fb_sg: f64,
    /// stability gate: sigma of the Gaussian window of the high-pass energy
    pub fb_win: f64,
    /// range check (0 = off): a motion-compensated neighbour pixel keeps its weight while |warped - current| <=
    /// range_k * sigma_d and loses it linearly up to (range_k + range_soft) * sigma_d. Keeps faint drawn strokes, rain
    /// and sprites that a 1-pixel misalignment of the neighbour would otherwise average away.
    pub range_k: f64,
    /// range check: width of the falloff, in sigma_d
    pub range_soft: f64,
    /// range check: the source's pixel noise in 10-bit codes (sigma_d = sqrt(2) * range_sigma); 0 = sigma_d per frame
    /// and neighbour, 1.4826 x the median |warped - current| over every 53rd pixel of mid luma (100 ... 900; all of
    /// them if fewer than 1000), at least 0.5
    pub range_sigma: f64,
    /// persistence restore (0 = off): after the guided fallback, put back fine detail that persists at the same place
    /// in the UNWARPED neighbours at least persist_gap frames away (static painted texture, held drawings; grain
    /// decorrelates over that distance; with none that far, the farthest ones). The reference's --persist-gap G
    /// --persist-static; its motion-compensated variant is not ported.
    pub persist_gap: usize,
    /// persistence restore: local correlation (median over the far neighbours) where the restore starts
    pub persist_lo: f64,
    /// persistence restore: correlation of a full restore
    pub persist_hi: f64,
    /// persistence restore: sigma of the Gaussian correlation window
    pub persist_win: f64,
}

/// Named parameter sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Preset {
    /// the encode pipeline's settings (the Python reference's defaults): detail restored with a 1.3 boost
    #[default]
    Production,
    /// GRAIL's pinned measurement settings: as production, but restored detail is not boosted (detail_boost 1)
    Measure,
    /// 2D cel animation with grainy cels (the reference's --preset cel): radius 8, the guided fallback everywhere
    /// (fallback 1000, radius 5) gated by stability (fb_gate 3), so flat cels lose their grain and drawn texture stays
    Cel,
}

impl Preset {
    pub const ALL: [Preset; 3] = [Preset::Production, Preset::Measure, Preset::Cel];

    pub fn name(self) -> &'static str {
        match self {
            Preset::Production => "production",
            Preset::Measure => "measure",
            Preset::Cel => "cel",
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
            "cel" => Ok(Preset::Cel),
            x => bail!("unknown preset {x:?} (production, measure or cel)"),
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
            fb_gate: 0.0,
            fb_q: 1.0,
            fb_sg: 2.0,
            fb_win: 3.0,
            range_k: 0.0,
            range_soft: 1.5,
            range_sigma: 0.0,
            persist_gap: 0,
            persist_lo: 0.3,
            persist_hi: 0.6,
            persist_win: 2.0,
        }
    }

    /// GRAIL's measurement denoiser: production with detail_boost 1 (flowdegrain2.py --detail-boost 1)
    pub fn measure() -> Params {
        Params { detail_boost: 1.0, ..Params::production() }
    }

    /// 2D cel animation: flowdegrain2.py --preset cel (--radius 8 --fallback 1000 --fallback-radius 5 --fb-gate 3)
    pub fn cel() -> Params {
        Params { radius: 8, fallback: 1000.0, fallback_radius: 5, fb_gate: 3.0, ..Params::production() }
    }

    pub fn preset(p: Preset) -> Params {
        match p {
            Preset::Production => Params::production(),
            Preset::Measure => Params::measure(),
            Preset::Cel => Params::cel(),
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
            && self.fallback_radius >= 1
            && self.fb_gate >= 0.0
            && self.fb_q > 0.0
            && self.fb_sg > 0.0
            && self.fb_win > 0.0
            && self.range_k >= 0.0
            && self.range_soft > 0.0
            && self.range_sigma >= 0.0
            && self.persist_win > 0.0;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets() {
        for p in Preset::ALL {
            assert_eq!(p.name().parse::<Preset>().unwrap(), p);
            assert_eq!(Params::preset(p).matches_preset(), Some(p));
            Params::preset(p).validate().unwrap();
        }
        let c = Params::cel();
        assert_eq!((c.radius, c.fallback, c.fallback_radius, c.fb_gate), (8, 1000.0, 5, 3.0));
        assert_eq!(Params { fb_gate: 0.0, radius: 3, fallback: 4.0, fallback_radius: 3, ..c }, Params::production());
        // off by default in every preset
        assert!(Preset::ALL.iter().all(|&p| Params::preset(p).range_k == 0.0 && Params::preset(p).persist_gap == 0));
        assert!(Params { range_soft: 0.0, ..Params::cel() }.validate().is_err());
    }
}
