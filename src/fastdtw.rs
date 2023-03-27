use ndarray::{Array1, ArrayView1};

use crate::{dtw_ex, ConstrainedWindow, DistanceMode, FullWindow};

fn coarse_time_series<T>(ts: &ArrayView1<T>, resolution_factor: usize) -> Array1<f64>
where
    T: std::ops::Add + Default + Copy + std::convert::Into<f64>,
{
    assert!(resolution_factor > 0);

    let rounded_coarsed_size = (ts.shape()[0] as f64 / resolution_factor as f64).ceil() as usize;
    let mut result = Array1::<f64>::default(rounded_coarsed_size);

    for pos in (0..rounded_coarsed_size * resolution_factor).step_by(resolution_factor) {
        let mut sum: f64 = 0f64;
        let end = std::cmp::min(pos + resolution_factor, ts.shape()[0]);
        for i in pos..end {
            sum += ts[i].into();
        }
        let average = sum / (end - pos) as f64;
        result[pos / resolution_factor] = average;
    }
    result
}

pub fn fastdtw<T>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> (f64, Array1<(usize, usize)>)
where
    T: std::ops::Add + std::ops::Sub + std::convert::Into<f64> + Default + Copy,
{
    fastdtw_ex(x, y, 2, 1, DistanceMode::Euclidean)
}

pub fn fastdtw_ex<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    resolution_factor: usize,
    search_radius: usize,
    distance_mode: DistanceMode,
) -> (f64, Array1<(usize, usize)>)
where
    T: std::ops::Add + std::ops::Sub + std::convert::Into<f64> + Default + Copy,
{
    let min_ts_size: usize = search_radius + 2;
    let rows = y.shape()[0];
    let columns = x.shape()[0];

    if x.shape()[0] <= min_ts_size || y.shape()[0] <= min_ts_size {
        // base case: for a very small time series run the full dtw algorithm
        dtw_ex(x, y, FullWindow::new(rows, columns), distance_mode)
    } else {
        /* recursive case:
         * project the warp path from a coarser resolution onto the current resolution
         * run dtw only along the projected path (and also 'search_radius' cells from the projected path)
         */
        let coarse_x = coarse_time_series(x, resolution_factor);
        let coarse_y = coarse_time_series(y, resolution_factor);

        let (_, low_res_path) = fastdtw_ex(
            &coarse_x.view(),
            &coarse_y.view(),
            resolution_factor,
            search_radius,
            distance_mode,
        );

        let constrained_window = ConstrainedWindow::from_low_res_path(
            low_res_path,
            resolution_factor,
            search_radius,
            y.shape()[0], /* high_res_rows */
            x.shape()[0], /* high_res_columns */
        );

        dtw_ex(x, y, constrained_window, distance_mode)
    }
}
