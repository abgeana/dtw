use ndarray::Array1;

/// The FullWindow iterator is used for the classical dynamic time warping algorithm to visit all cells in the cost
/// matrix. This window does not implement any optimizing constraints regarding the visited cells.
///
/// The layout of the window is based on the general depictions related to the dynamic time warping algorithm. The x
/// time series is horizontal and the y time series is vertical. Thus, the size of x denotes the number of columns and
/// the size of y denotes the number of rows.
pub struct FullWindow {
    /// The current column in the cost matrix.
    row: usize,
    /// The current row in the cost matrix.
    column: usize,
    /// The last column in the cost matrix (exclusive).
    end_row: usize,
    /// The last row in the cost matrix (exclusive).
    end_column: usize,
}

impl FullWindow {
    pub fn new(x_size: usize, y_size: usize) -> Self {
        Self {
            row: 1,
            column: 1,
            end_row: x_size + 1,
            end_column: y_size + 1,
        }
    }
}

impl Iterator for FullWindow {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        /* The iteration of the cost matrix is performed per row. This means that each column of a row is first
         * iterated, and when a row end is reached, the iterator goes to the next row.
         */
        let mut result = None;

        if self.column < self.end_column {
            // can advance to the next column on the current row
            result = Some((self.row, self.column));
            self.column += 1;
        } else if self.row < self.end_row - 1 {
            // can advance to the next row
            self.row += 1;
            self.column = 1;
            result = Some((self.row, self.column));
            self.column += 1;
        }

        result
    }
}

pub struct ConstrainedWindow {
    /// For each row, the minimum and maximum column values in the cost matrix are stored as a tuple. The number of
    /// elements in the constraints array thus denotes the number of rows in the cost matrix.
    constraints: Array1<(usize, usize)>,
    row: usize,
    column: usize,
}

impl ConstrainedWindow {
    pub fn from_low_res_path(
        low_res_path: Array1<(usize, usize)>,
        resolution_factor: usize,
        search_radius: usize,
        high_res_rows: usize,
        high_res_columns: usize,
    ) -> Self {
        let mut window = ConstrainedWindow {
            constraints: Array1::<(usize, usize)>::from_elem(high_res_rows + 1, (usize::MAX, 0)),
            row: 1,
            column: 1,
        };

        let mut prev_low_res_row: usize = usize::MAX;
        let mut prev_low_res_column: usize = usize::MAX;
        for (mut low_res_row, mut low_res_column) in low_res_path {
            // convert the 0 based indices to 1 based indices
            low_res_row += 1;
            low_res_column += 1;
            /* project the low resolution coordinates to the higher resolution
             * one cell in the lower resolution matrix is mapped to (at most)
             * $resolution_factor cells in the higher resolution matrix
             */
            for row in 0..resolution_factor {
                let high_res_row = (low_res_row - 1) * resolution_factor + 1 + row;
                for column in 0..resolution_factor {
                    let high_res_column = (low_res_column - 1) * resolution_factor + 1 + column;
                    if high_res_row < high_res_rows + 1 && high_res_column < high_res_columns + 1 {
                        window.visit(high_res_row, high_res_column);
                    }
                }
            }
            /* if a diagonal move was performed, add two cells to the edges of the two blocks
             * in the projected path to create a continuous path with even width
             * avoid a path of boxes connected only at their corners
             * example when the $resolution_factor is 2
             *
             *                        |_|_|x|x|     then mark      |_|_|x|x|
             *        projected path: |_|_|x|x|  --2 more cells->  |_|X|x|x|
             *                        |x|x|_|_|        (X's)       |x|x|X|_|
             *                        |x|x|_|_|                    |x|x|_|_|
             *
             * to generalize, the idea is to add two blocks of width = $resolution_factor / 2
             * on either side of the connected corners
             */
            if prev_low_res_row < low_res_row && prev_low_res_column < low_res_column {
                let corner_bottom_left_row: usize =
                    (prev_low_res_row - 1) * resolution_factor + resolution_factor;
                let corner_bottom_left_column: usize =
                    (prev_low_res_column - 1) * resolution_factor + resolution_factor;

                let corner_top_right_row: usize = corner_bottom_left_row + 1;
                let corner_top_right_column: usize = corner_bottom_left_column + 1;

                let half_resolution_factor: usize =
                    (resolution_factor as f64 / 2f64).ceil() as usize;

                for row in 0..half_resolution_factor {
                    for column in 0..half_resolution_factor {
                        // add first small block to the right of the bottom left block
                        if corner_top_right_column + column < high_res_columns + 1 {
                            window.visit(
                                corner_bottom_left_row - row,
                                corner_top_right_column + column,
                            );
                        }
                        // add second small black to the left of the top right block
                        if corner_top_right_row + row < high_res_rows + 1 {
                            window.visit(
                                corner_top_right_row + row,
                                corner_bottom_left_column - column,
                            );
                        }
                    }
                }
            }
            prev_low_res_row = low_res_row;
            prev_low_res_column = low_res_column;
        }
        /* the last step is to expand the high resolution warp path with the search radius
         * for each minimum value we expand in the left, top and top left directions
         * for each maximum value, we expand in the right, bottom and bottom right directions
         *
         * this is done in two steps:
         * 1. first iterate each row and expand on right, bottom and bottom right directions
         * 2. then iterate in reverse order and expand to the left, top and top left directions
         */
        for row in 1..window.constraints.shape()[0] {
            let (_, row_max) = window.constraints[row];
            for i in 0..search_radius + 1 {
                let expanded_row_max: usize = usize::min(row_max + search_radius, high_res_columns);
                if row > i && row - i >= 1 {
                    window.visit(row - i, expanded_row_max);
                }
            }
        }
        for row in (1..window.constraints.shape()[0]).rev() {
            let (row_min, _) = window.constraints[row];
            for i in 0..search_radius + 1 {
                let expanded_row_min = match row_min > search_radius {
                    true => row_min - search_radius,
                    false => 1,
                };
                if row + i <= high_res_rows {
                    window.visit(row + i, expanded_row_min);
                }
            }
        }

        window
    }

    fn visit(&mut self, row: usize, column: usize) {
        if self.constraints[row].0 > column {
            self.constraints[row].0 = column;
        }
        if self.constraints[row].1 < column {
            self.constraints[row].1 = column;
        }
    }
}

impl Iterator for ConstrainedWindow {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        /* The iteration of the cost matrix is performed per row. This means that each column of a row is first
         * iterated, and when a row end is reached, the iterator goes to the next row.
         */
        let mut result = None;

        if self.column <= self.constraints[self.row].1 {
            // can advance to the next column on the current row
            result = Some((self.row, self.column));
            self.column += 1;
        } else if self.row < self.constraints.shape()[0] - 1 {
            // can advance to the next row
            self.row += 1;
            self.column = self.constraints[self.row].0;
            result = Some((self.row, self.column));
            self.column += 1;
        }

        result
    }
}
