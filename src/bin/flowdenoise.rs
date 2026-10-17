//! flowdenoise: y4m in (10-bit 4:2:0), denoised y4m out. "-" reads stdin / writes stdout.
//!
//! The output header is the input's, byte for byte; frame parameters (tags on FRAME lines) are not propagated.
//! Reading, denoising and writing run on three threads with small bounded queues between them.
use anyhow::{bail, Context, Result};
use clap::Parser;
use flowdenoise::{Backend, Denoiser, Frame, Params, Preset};
use std::io::{BufReader, BufWriter, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "flowdenoise", version, about = "Motion-compensated temporal video denoiser on the GPU (10-bit 4:2:0 y4m in, y4m out)")]
struct Cli {
    /// input y4m (C420p10), or - for stdin
    input: PathBuf,
    /// output y4m, or - for stdout
    output: PathBuf,
    /// parameter set: production (the encode pipeline's), measure (GRAIL's; no detail boost) or cel (2D cel
    /// animation: radius 8, gated guided fallback)
    #[arg(short, long, default_value = "production", env = "FLOWDENOISE_PRESET")]
    preset: Preset,
    /// GPU backend: auto, cuda, hip or metal
    #[arg(short, long, default_value = "auto", env = "FLOWDENOISE_BACKEND")]
    backend: Backend,
    /// GPU index among the backend's devices
    #[arg(long, default_value_t = 0, env = "FLOWDENOISE_GPU")]
    gpu: usize,
    /// no progress line on stderr
    #[arg(short, long)]
    quiet: bool,
    /// print the effective parameters on stderr and exit
    #[arg(long)]
    print_params: bool,

    /// temporal radius (frames t - r ..= t + r)
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    radius: Option<usize>,
    /// SAD threshold of the temporal weights (MVTools thSAD: 8x8 block, 8-bit units)
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    thsad: Option<f32>,
    /// motion resolution divisor: 0 auto (2 above 2000 px width), 1 or 2
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    flow_scale: Option<usize>,
    /// motion pyramid depth
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    levels: Option<usize>,
    /// finest pyramid level of the motion search
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    finest: Option<usize>,
    /// gradient-descent iterations per patch (over both sweeps)
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    iterations: Option<i32>,
    /// keep the motion pyramid fractional (no rounding to 8-bit codes)
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    no_round_motion: bool,
    /// detail gate threshold (temporal coherence of fine detail), 0 ..< 1
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    detail_gate: Option<f32>,
    /// gain of restored detail (1 = as it was)
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    detail_boost: Option<f32>,
    /// spike rejection strength
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    spike: Option<f32>,
    /// motion-dependent SAD threshold: thsad / (1 + speed / fast_th)
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    fast_th: Option<f32>,
    /// guided fallback below this effective frame count (<= 1: off)
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    fallback: Option<f32>,
    /// guided filter regularisation, in noise variances
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    fallback_eps: Option<f32>,
    /// guided filter radius (luma; chroma half)
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    fallback_radius: Option<usize>,
    /// stability gate on the guided fallback: skipped where the fused frame keeps structure G times above the
    /// averaged grain's energy (0 = off; cel: 3)
    #[arg(long, value_name = "G", help_heading = "Parameters (default: the preset's)")]
    fb_gate: Option<f64>,
    /// stability gate: scale of the expected residual grain energy
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    fb_q: Option<f64>,
    /// stability gate: high-pass sigma
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    fb_sg: Option<f64>,
    /// stability gate: energy window sigma
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    fb_win: Option<f64>,
    /// range check: a neighbour pixel keeps its weight while |warped - current| <= K sigma_d and loses it linearly up
    /// to (K + range-soft) sigma_d; keeps faint strokes, rain and sprites (0 = off, the default of every preset)
    #[arg(long, value_name = "K", help_heading = "Parameters (default: the preset's)")]
    range_k: Option<f64>,
    /// range check: width of the falloff, in sigma_d
    #[arg(long, help_heading = "Parameters (default: the preset's)")]
    range_soft: Option<f64>,
    /// range check: the source's pixel noise in 10-bit codes (sigma_d = sqrt(2) S); 0 = per frame and neighbour from
    /// the median |warped - current| over mid-luma pixels
    #[arg(long, value_name = "S", help_heading = "Parameters (default: the preset's)")]
    range_sigma: Option<f64>,
}

impl Cli {
    fn params(&self) -> Params {
        let mut p = Params::preset(self.preset);
        macro_rules! set {
            ($($f:ident),*) => { $(if let Some(v) = self.$f { p.$f = v; })* };
        }
        set!(radius, thsad, flow_scale, levels, finest, iterations, detail_gate, detail_boost, spike, fast_th, fallback,
             fallback_eps, fallback_radius, fb_gate, fb_q, fb_sg, fb_win, range_k, range_soft, range_sigma);
        if self.no_round_motion {
            p.round_motion = false;
        }
        p
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("flowdenoise: {e:#}");
        std::process::exit(1);
    }
}

/// input frames from the reader thread; None at the end of the stream
type FrameMsg = Result<Option<Frame>>;

fn run() -> Result<()> {
    let cli = Cli::parse();
    let params = cli.params();
    params.validate()?;
    if cli.print_params {
        let what = params.matches_preset().map(|p| format!("preset {p}")).unwrap_or_else(|| "custom".into());
        eprintln!("flowdenoise: {what}: {params:#?}");
        return Ok(());
    }
    let rd: Box<dyn Read + Send> = if cli.input.as_os_str() == "-" {
        Box::new(std::io::stdin())
    } else {
        Box::new(std::fs::File::open(&cli.input).with_context(|| format!("{}", cli.input.display()))?)
    };
    let mut dec = y4m::Decoder::new(BufReader::with_capacity(1 << 22, rd)).map_err(|e| anyhow::anyhow!("input: {e}"))?;
    if !matches!(dec.get_colorspace(), y4m::Colorspace::C420p10) {
        bail!("input: only 10-bit 4:2:0 y4m (C420p10) is supported, not {:?} (ffmpeg: -pix_fmt yuv420p10le -strict -1)",
              dec.get_colorspace());
    }
    let (w, h) = (dec.get_width(), dec.get_height());
    let mut header = b"YUV4MPEG2 ".to_vec();
    header.extend_from_slice(dec.get_raw_params());
    header.push(b'\n');
    let mut den = Denoiser::with_device(w, h, &params, cli.backend, cli.gpu)?;

    let wr: Box<dyn Write + Send> = if cli.output.as_os_str() == "-" {
        Box::new(std::io::stdout())
    } else {
        Box::new(std::fs::File::create(&cli.output).with_context(|| format!("{}", cli.output.display()))?)
    };

    // reader -> GPU -> writer, two frames of slack on each side
    let (in_tx, in_rx): (SyncSender<FrameMsg>, Receiver<FrameMsg>) = sync_channel(2);
    let (out_tx, out_rx) = sync_channel::<Frame>(2);
    let (free_tx, free_rx) = sync_channel::<Frame>(4);
    let reader = std::thread::spawn(move || {
        loop {
            let msg = match dec.read_frame() {
                Ok(f) => {
                    // the payload is contiguous Y U V, but the decoder hands out three slices
                    let mut bytes = Vec::with_capacity(w * h * 3);
                    for p in [f.get_y_plane(), f.get_u_plane(), f.get_v_plane()] {
                        bytes.extend_from_slice(p);
                    }
                    Frame::from_le_bytes(w, h, &bytes).map(Some)
                }
                Err(y4m::Error::EOF) => Ok(None),
                Err(e) => Err(anyhow::anyhow!("input: {e}")),
            };
            let stop = !matches!(msg, Ok(Some(_)));
            if in_tx.send(msg).is_err() || stop {
                break;
            }
        }
    });
    let writer = std::thread::spawn(move || -> Result<()> {
        let mut wr = BufWriter::with_capacity(1 << 22, wr);
        wr.write_all(&header)?;
        let mut bytes = Vec::new();
        for frame in out_rx {
            wr.write_all(b"FRAME\n")?;
            if cfg!(target_endian = "little") {
                wr.write_all(frame.as_le_bytes())?;
            } else {
                bytes.clear();
                frame.write_le_bytes(&mut bytes);
                wr.write_all(&bytes)?;
            }
            let _ = free_tx.try_send(frame);
        }
        wr.flush()?;
        Ok(())
    });

    let mut progress = Progress::new(!cli.quiet);
    let mut eof = false;
    let result = (|| -> Result<()> {
        loop {
            // produce whatever is ready; otherwise feed the lookahead (the reference's order of work)
            if den_ready(&mut den, &out_tx, &free_rx, &mut progress)? {
                continue;
            }
            if eof {
                return Ok(());
            }
            match in_rx.recv().context("reader thread")?? {
                Some(f) => den.send_frame(&f)?,
                None => {
                    eof = true;
                    den.flush();
                }
            }
        }
    })();
    drop(out_tx);
    drop(in_rx);
    let wres = writer.join().map_err(|_| anyhow::anyhow!("writer thread panicked"))?;
    let _ = reader.join();
    result?;
    wres.context("output")?;
    progress.done(den.frames_out(), w, h, &params);
    Ok(())
}

/// produce one frame if the denoiser has one ready and hand it to the writer; false if none was ready
fn den_ready(den: &mut Denoiser, out_tx: &SyncSender<Frame>, free_rx: &Receiver<Frame>, progress: &mut Progress)
             -> Result<bool> {
    if den.ready().is_none() {
        return Ok(false);
    }
    // a frame buffer the writer has finished with, or a new one
    let mut out = match free_rx.try_recv() {
        Ok(f) => f,
        Err(_) => Frame::new(den.width(), den.height())?,
    };
    den.receive_frame_into(&mut out)?;
    out_tx.send(out).map_err(|_| anyhow::anyhow!("writer thread stopped"))?;
    progress.tick(den.frames_out());
    Ok(true)
}

/// the progress line on stderr: rewritten in place on a terminal, one line every 10 s otherwise
struct Progress {
    on: bool,
    tty: bool,
    t0: Instant,
    last: Instant,
}

impl Progress {
    fn new(on: bool) -> Progress {
        let now = Instant::now();
        Progress { on, tty: std::io::stderr().is_terminal(), t0: now, last: now }
    }
    fn tick(&mut self, frames: usize) {
        if !self.on {
            return;
        }
        let every = if self.tty { Duration::from_millis(500) } else { Duration::from_secs(10) };
        if self.last.elapsed() < every {
            return;
        }
        self.last = Instant::now();
        let s = self.t0.elapsed().as_secs_f64();
        let line = format!("flowdenoise: {frames} frames, {:.1} fps, {:.0} s", frames as f64 / s, s);
        if self.tty {
            eprint!("\r{line}   ");
        } else {
            eprintln!("{line}");
        }
    }
    fn done(&self, frames: usize, w: usize, h: usize, p: &Params) {
        if !self.on {
            return;
        }
        let s = self.t0.elapsed().as_secs_f64();
        let what = p.matches_preset().map(|p| format!("preset {p}")).unwrap_or_else(|| "custom parameters".into());
        if self.tty {
            eprint!("\r");
        }
        eprintln!("flowdenoise: {frames} frames {w}x{h} in {s:.1} s ({:.1} fps), {what}", frames as f64 / s.max(1e-9));
    }
}
