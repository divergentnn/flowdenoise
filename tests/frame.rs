//! Frame conversions and presets (no GPU).
use flowdenoise::{Frame, Params, Preset};

#[test]
fn planes_and_bytes_round_trip() {
    let (w, h) = (6, 4);
    let data: Vec<u16> = (0..(w * h * 3 / 2) as u16).map(|v| v * 7 % 1024).collect();
    let f = Frame::from_planar(w, h, data.clone()).unwrap();
    assert_eq!(f.y().len(), 24);
    assert_eq!(f.u().len(), 6);
    assert_eq!(f.v(), &data[30..36]);
    let bytes = f.to_le_bytes();
    assert_eq!(bytes.len(), w * h * 3);
    assert_eq!(Frame::from_le_bytes(w, h, &bytes).unwrap(), f);
}

#[test]
fn strided_and_8bit_planes() {
    let (w, h) = (4, 2);
    let y: Vec<u16> = (0..16).collect(); // stride 8
    let u = [100u16, 101, 0, 0];
    let v = [200u16, 201, 0, 0];
    let f = Frame::from_planes(w, h, [(&y, 8), (&u, 4), (&v, 4)]).unwrap();
    assert_eq!(f.y(), &[0, 1, 2, 3, 8, 9, 10, 11]);
    assert_eq!(f.u(), &[100, 101]);
    let y8 = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let f8 = Frame::from_8bit_planes(w, h, [(&y8, 4), (&[9u8, 10], 2), (&[11u8, 12], 2)]).unwrap();
    assert_eq!(f8.y()[0], 4);
    assert_eq!(f8.v(), &[44, 48]);
}

#[test]
fn odd_sizes_rejected() {
    assert!(Frame::new(5, 4).is_err());
    assert!(Frame::from_planar(4, 4, vec![0; 10]).is_err());
}

#[test]
fn presets() {
    let p = Params::production();
    let m = Params::measure();
    assert_eq!(p.detail_boost, 1.3);
    assert_eq!(m.detail_boost, 1.0);
    assert_eq!(Params { detail_boost: 1.3, ..m.clone() }, p);
    assert_eq!(p.matches_preset(), Some(Preset::Production));
    assert_eq!(m.matches_preset(), Some(Preset::Measure));
    assert_eq!("measure".parse::<Preset>().unwrap(), Preset::Measure);
    p.validate().unwrap();
    assert_eq!(p.motion_scale(1920), 1);
    assert_eq!(p.motion_scale(3840), 2);
    assert!(p.check_size(1920, 1080).is_ok());
    assert!(p.check_size(64, 64).is_err());
}
