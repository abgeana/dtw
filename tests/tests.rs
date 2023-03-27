use dtw::{dtw_ex, fastdtw_ex, ConstrainedWindow, DistanceMode, FullWindow};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize)]
struct DtwTestCase {
    name: String,
    time_series_a: Array1<f64>,
    time_series_b: Array1<f64>,
    distance: f64,
    distance_mode: String,
    warp_path: Array1<(usize, usize)>,
}

#[derive(Serialize, Deserialize)]
struct ProjectionTestCase {
    name: String,
    low_res_path: Array1<(usize, usize)>,
    resolution_factor: usize,
    search_radius: usize,
    high_res_rows: usize,
    high_res_columns: usize,
    projected_window: Array1<(usize, usize)>,
}

#[test]
fn test_dtw() {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("tests/dtw.yaml");
    let f = std::fs::File::open(d).expect("could not open dtw.yaml");
    let test_cases: Vec<DtwTestCase> =
        serde_yaml::from_reader(f).expect("could not read test values from dtw.yaml");

    for tc in test_cases {
        let rows = tc.time_series_b.shape()[0];
        let columns = tc.time_series_a.shape()[0];
        let distance_mode = match tc.distance_mode.as_str() {
            "manhattan" => DistanceMode::Manhattan,
            "euclidean" => DistanceMode::Euclidean,
            _ => panic!("unknown distance mode specified"),
        };

        let (distance, path) = dtw_ex(
            &tc.time_series_a.view(),
            &tc.time_series_b.view(),
            FullWindow::new(rows, columns),
            distance_mode,
        );

        assert_eq!(distance, tc.distance);
        assert_eq!(path, tc.warp_path);
    }
}

#[test]
fn test_window_projection() {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("tests/projection.yaml");
    let f = std::fs::File::open(d).expect("could not open projection.yaml");
    let test_cases: Vec<ProjectionTestCase> =
        serde_yaml::from_reader(f).expect("could not read test values from projection.yaml");

    for tc in test_cases {
        let mut window = ConstrainedWindow::from_low_res_path(
            tc.low_res_path,
            tc.resolution_factor,
            tc.search_radius,
            tc.high_res_rows,
            tc.high_res_columns,
        );

        for i in 0..tc.projected_window.shape()[0] {
            assert_eq!(tc.projected_window[i], window.next().unwrap());
        }
    }
}

#[test]
fn test_fast_dtw() {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("tests/dtw.yaml");
    let f = std::fs::File::open(d).expect("could not open dtw.yaml");
    let test_cases: Vec<DtwTestCase> =
        serde_yaml::from_reader(f).expect("could not read test values from dtw.yaml");

    for tc in test_cases {
        let distance_mode = match tc.distance_mode.as_str() {
            "manhattan" => DistanceMode::Manhattan,
            "euclidean" => DistanceMode::Euclidean,
            _ => panic!("unknown distance mode specified"),
        };

        let (distance, path) = fastdtw_ex(
            &tc.time_series_a.view(),
            &tc.time_series_b.view(),
            2,
            10,
            distance_mode,
        );

        assert_eq!(distance, tc.distance);
        assert_eq!(path, tc.warp_path);
    }
}
