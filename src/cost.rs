use ndarray::{Array1, Array2};
use std::collections::HashMap;

use crate::enums::Action;

pub trait CostStorage {
    fn get_cost(&self, row: usize, column: usize) -> f64;
    fn set_cost(&mut self, row: usize, column: usize, cost: f64);
    fn get_action(&self, row: usize, column: usize) -> Action;
    fn set_action(&mut self, row: usize, column: usize, action: Action);
}

pub struct CostMatrix {
    cost_matrix: Array2<f64>,
    actions_matrix: Array2<Action>,
}

impl CostMatrix {
    pub fn new(rows: usize, columns: usize) -> Self {
        let mut cost_matrix = Self {
            cost_matrix: Array2::<f64>::from_elem((rows, columns), f64::INFINITY),
            actions_matrix: Array2::<Action>::default((rows, columns)),
        };
        cost_matrix.cost_matrix[[0, 0]] = 0f64;
        cost_matrix
    }
}

impl CostStorage for CostMatrix {
    fn get_cost(&self, row: usize, column: usize) -> f64 {
        if row == 0 && column == 0 {
            return 0f64;
        } else if row == 0 || column == 0 {
            return f64::INFINITY;
        }
        self.cost_matrix[[row - 1, column - 1]]
    }

    fn set_cost(&mut self, row: usize, column: usize, cost: f64) {
        assert_ne!(row, 0);
        assert_ne!(column, 0);
        self.cost_matrix[[row - 1, column - 1]] = cost;
    }

    fn get_action(&self, row: usize, column: usize) -> Action {
        assert_ne!(row, 0);
        assert_ne!(column, 0);
        self.actions_matrix[[row - 1, column - 1]]
    }

    fn set_action(&mut self, row: usize, column: usize, action: Action) {
        assert_ne!(row, 0);
        assert_ne!(column, 0);
        self.actions_matrix[[row - 1, column - 1]] = action;
    }
}

pub struct CostCache {
    cost_cache: Array1<HashMap<usize, f64>>,
    actions_cache: Array1<HashMap<usize, Action>>,
}

impl CostCache {
    pub fn new(rows: usize) -> Self {
        Self {
            cost_cache: Array1::<HashMap<usize, f64>>::default(rows),
            actions_cache: Array1::<HashMap<usize, Action>>::default(rows),
        }
    }
}

impl CostStorage for CostCache {
    fn get_cost(&self, row: usize, column: usize) -> f64 {
        if row == 0 && column == 0 {
            return 0f64;
        } else if row == 0 || column == 0 {
            return f64::INFINITY;
        }
        *self.cost_cache[row - 1]
            .get(&(column - 1))
            .unwrap_or(&f64::INFINITY)
    }

    fn set_cost(&mut self, row: usize, column: usize, cost: f64) {
        assert_ne!(row, 0);
        assert_ne!(column, 0);
        self.cost_cache[row - 1].insert(column - 1, cost);
    }

    fn get_action(&self, row: usize, column: usize) -> Action {
        assert_ne!(row, 0);
        assert_ne!(column, 0);
        self.actions_cache[row - 1][&(column - 1)]
    }

    fn set_action(&mut self, row: usize, column: usize, action: Action) {
        assert_ne!(row, 0);
        assert_ne!(column, 0);
        self.actions_cache[row - 1].insert(column - 1, action);
    }
}

static mut MAX_COST_STORAGE_MATRIX: usize = 32 * 1024 * 1024 * 1024;

pub fn config_max_cost_storage_matrix(max: usize) {
    unsafe {
        MAX_COST_STORAGE_MATRIX = max;
    }
}

pub(crate) fn cost_storage(rows: usize, columns: usize) -> Box<dyn CostStorage> {
    if unsafe { rows * columns * std::mem::size_of::<f64>() < MAX_COST_STORAGE_MATRIX } {
        return Box::new(CostMatrix::new(rows, columns));
    }
    Box::new(CostCache::new(rows))
}
