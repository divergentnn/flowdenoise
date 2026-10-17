//! End-to-end exactness against reference outputs (needs a GPU and test media; skipped without them).
//!
//!   FLOWDENOISE_TEST_CLIP=clip.y4m                 10-bit 4:2:0 input
//!   FLOWDENOISE_TEST_EXPECT_PRODUCTION=out.y4m     the Python reference's output (flowdegrain2.py defaults)
//!   FLOWDENOISE_TEST_EXPECT_MEASURE=out.y4m        its output with --detail-boost 1 (or `grail denoise`'s)
//!   FLOWDENOISE_TEST_EXPECT_CEL=out.y4m            its output with --preset cel
//!   FLOWDENOISE_TEST_EXPECT_CEL_RANGE=out.y4m      --preset cel --range-k 2 (sigma_d estimated per neighbour)
//!   FLOWDENOISE_TEST_EXPECT_CEL_RANGE_SIGMA=out.y4m  --preset cel --range-k 2 --range-sigma 2
//!   FLOWDENOISE_TEST_EXPECT_CEL_PERSIST=out.y4m    --preset cel --range-k 2 --persist-gap 6 --persist-static
//!
//! Each expected file is compared frame by frame, every sample, through the library API.
use flowdenoise::{Backend, Denoiser, Frame, Params, Preset};
use std::io::{BufRead, BufReader, Read};

/// (width, height, frames) of a C420p10 y4m file
fn read_y4m(path: &str) -> (usize, usize, Vec<Frame>) {
    let mut r = BufReader::new(std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}")));
    let mut hdr = String::new();
    r.read_line(&mut hdr).unwrap();
    assert!(hdr.starts_with("YUV4MPEG2 ") && hdr.contains(" C420p10"), "{path}: not a C420p10 y4m");
    let tag = |c: char| hdr.split_whitespace().find(|t| t.starts_with(c)).unwrap()[1..].parse::<usize>().unwrap();
    let (w, h) = (tag('W'), tag('H'));
    let mut frames = vec![];
    loop {
        let mut line = String::new();
        if r.read_line(&mut line).unwrap() == 0 {
            break;
        }
        assert!(line.starts_with("FRAME"));
        let mut buf = vec![0u8; w * h * 3];
        r.read_exact(&mut buf).unwrap();
        frames.push(Frame::from_le_bytes(w, h, &buf).unwrap());
    }
    (w, h, frames)
}

fn check(preset: Preset, var: &str) {
    check_params(&Params::preset(preset), preset.name(), var)
}

fn check_params(params: &Params, label: &str, var: &str) {
    let (Ok(clip), Ok(expect)) = (std::env::var("FLOWDENOISE_TEST_CLIP"), std::env::var(var)) else {
        eprintln!("skipped: set FLOWDENOISE_TEST_CLIP and {var}");
        return;
    };
    let (w, h, input) = read_y4m(&clip);
    let (_, _, expected) = read_y4m(&expect);
    assert_eq!(input.len(), expected.len());
    let mut den = Denoiser::new(w, h, params, Backend::Auto).unwrap();
    let mut out = vec![];
    for f in &input {
        den.send_frame(f).unwrap();
        while let Some(d) = den.receive_frame().unwrap() {
            out.push(d);
        }
    }
    den.flush();
    while let Some(d) = den.receive_frame().unwrap() {
        out.push(d);
    }
    assert_eq!(out.len(), expected.len());
    let (mut bad, mut worst, mut total, mut frames) = (0usize, 0i32, 0usize, vec![]);
    for (i, (a, b)) in out.iter().zip(&expected).enumerate() {
        let d = a.data().iter().zip(b.data()).filter(|(x, y)| x != y).count();
        worst = worst.max(a.data().iter().zip(b.data()).map(|(&x, &y)| (x as i32 - y as i32).abs()).max().unwrap_or(0));
        bad += d;
        total += a.data().len();
        if d > 0 {
            frames.push(i);
        }
    }
    assert_eq!(bad, 0, "{label}: {bad} of {total} samples differ ({:.3e}), max abs diff {worst}, frames {frames:?}",
               bad as f64 / total as f64);
}

fn cel_range(sigma: f64) -> Params {
    Params { range_k: 2.0, range_sigma: sigma, ..Params::cel() }
}

#[test]
fn production_matches_reference() {
    check(Preset::Production, "FLOWDENOISE_TEST_EXPECT_PRODUCTION");
}

#[test]
fn measure_matches_reference() {
    check(Preset::Measure, "FLOWDENOISE_TEST_EXPECT_MEASURE");
}

#[test]
fn cel_matches_reference() {
    check(Preset::Cel, "FLOWDENOISE_TEST_EXPECT_CEL");
}

#[test]
fn cel_range_check_matches_reference() {
    check_params(&cel_range(0.0), "cel --range-k 2", "FLOWDENOISE_TEST_EXPECT_CEL_RANGE");
}

#[test]
fn cel_range_check_fixed_sigma_matches_reference() {
    check_params(&cel_range(2.0), "cel --range-k 2 --range-sigma 2", "FLOWDENOISE_TEST_EXPECT_CEL_RANGE_SIGMA");
}

#[test]
fn cel_persistence_restore_matches_reference() {
    let p = Params { persist_gap: 6, ..cel_range(0.0) };
    check_params(&p, "cel --range-k 2 --persist-gap 6", "FLOWDENOISE_TEST_EXPECT_CEL_PERSIST");
}
