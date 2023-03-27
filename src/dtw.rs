use ndarray::{s, Array1, ArrayView1};

use crate::cost;
use crate::enums::*;
use crate::window::*;

/// Find the minimum of the three adjacent cells at each step of the cost matrix calculation.
///
/// # Arguments
///
/// * `i` - the value of the cell above the current one (i stands for insertion)
///
/// * `d` - the value of the cell to the left of the current one (d stands for deletion)
///
/// * `m` - the value of the cell above and to the left of the current one (m stands for matching)
///
/// # Returns
///
/// The return value is a tuple with two elements. The first element is an f64 denoting the minimum
/// value of the three arguments. The second element is of type `Action` and denotes the specific
/// element that was chosen.
fn minimum(i: f64, d: f64, m: f64) -> (f64, Action) {
    if i < d {
        if i < m {
            return (i, Action::Inserted);
        }
    } else if d < m {
        return (d, Action::Deleted);
    }
    (m, Action::Matched)
}

pub fn dtw<T>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> (f64, Array1<(usize, usize)>)
where
    T: std::ops::Sub + std::marker::Copy + std::convert::Into<f64>,
{
    let rows = y.shape()[0];
    let columns = x.shape()[0];
    dtw_ex(
        x,
        y,
        FullWindow::new(rows, columns),
        DistanceMode::Euclidean,
    )
}

pub fn dtw_ex<T, W>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    window: W,
    distance_mode: DistanceMode,
) -> (f64, Array1<(usize, usize)>)
where
    T: std::ops::Sub + std::marker::Copy + std::convert::Into<f64>,
    W: Iterator<Item = (usize, usize)>,
{
    let x_size = x.shape()[0];
    let y_size = y.shape()[0];
    let mut cost_storage = cost::cost_storage(y_size, x_size);

    for (row, column) in window {
        let cost = match distance_mode {
            DistanceMode::Manhattan => f64::abs(x[column - 1].into() - y[row - 1].into()),
            DistanceMode::Euclidean => {
                let difference = x[column - 1].into() - y[row - 1].into();
                difference * difference
            }
        };

        let (value, action) = minimum(
            cost_storage.get_cost(row - 1, column), // insertion - the cell above
            cost_storage.get_cost(row, column - 1), // deletion - the cell to the left
            cost_storage.get_cost(row - 1, column - 1), // match - the cell above and to the left
        );

        cost_storage.set_cost(row, column, cost + value);
        cost_storage.set_action(row, column, action);
    }
    let distance = match distance_mode {
        DistanceMode::Manhattan => cost_storage.get_cost(y_size, x_size),
        DistanceMode::Euclidean => cost_storage.get_cost(y_size, x_size).sqrt(),
    };

    /* generate the warp path based on the cost matrix
     * the path is allocated as a x_size + y_size array for the worst case scenario
     * afterwards, the path is truncated based on the actual number of elements
     */
    let mut path = Array1::<(usize, usize)>::default(x_size + y_size);
    let mut path_len = 0;
    let mut row = y_size;
    let mut column = x_size;
    while row != 0 && column != 0 {
        /* the search window and cost matrix use the first row and first column themselves
         * the results are thus 1 based indices of the time series samples, and not 0 based
         * in order to convert back to 0 based indices, we subtract 1 below
         */
        path[path_len] = (row - 1, column - 1);
        (row, column) = match cost_storage.get_action(row, column) {
            Action::Inserted => (row - 1, column),
            Action::Deleted => (row, column - 1),
            Action::Matched => (row - 1, column - 1),
            Action::Unknown => {
                /* this should actually never happen if everything went well during the generation
                 * of the cost matrix
                 */
                panic!("unknown error during the generation of the warp path");
            }
        };
        path_len += 1;
    }
    let path = path.slice_move(s![..path_len;-1]);

    (distance, path)
}
