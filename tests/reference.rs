//! End-to-end exactness against reference outputs (needs a GPU and test media; skipped without them).
//!
//!   FLOWDENOISE_TEST_CLIP=clip.y4m                 10-bit 4:2:0 input
//!   FLOWDENOISE_TEST_EXPECT_PRODUCTION=out.y4m     the Python reference's output (flowdegrain2.py defaults)
//!   FLOWDENOISE_TEST_EXPECT_MEASURE=out.y4m        its output with --detail-boost 1 (or `grail denoise`'s)
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
    let (Ok(clip), Ok(expect)) = (std::env::var("FLOWDENOISE_TEST_CLIP"), std::env::var(var)) else {
        eprintln!("skipped: set FLOWDENOISE_TEST_CLIP and {var}");
        return;
    };
    let (w, h, input) = read_y4m(&clip);
    let (_, _, expected) = read_y4m(&expect);
    assert_eq!(input.len(), expected.len());
    let mut den = Denoiser::new(w, h, &Params::preset(preset), Backend::Auto).unwrap();
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
    for (i, (a, b)) in out.iter().zip(&expected).enumerate() {
        let bad = a.data().iter().zip(b.data()).filter(|(x, y)| x != y).count();
        assert_eq!(bad, 0, "{preset} frame {i}: {bad} samples differ");
    }
}

#[test]
fn production_matches_reference() {
    check(Preset::Production, "FLOWDENOISE_TEST_EXPECT_PRODUCTION");
}

#[test]
fn measure_matches_reference() {
    check(Preset::Measure, "FLOWDENOISE_TEST_EXPECT_MEASURE");
}
