use crate::solver::{Certainty, Solver};
use std::collections::HashMap;

use super::{board::SolverCell, SolverAction, SolverBoard, SolverResult};
use crate::Position;

#[derive(Debug)]
struct LinearSystem {
    // Each row represents an equation where coefficients[i][j] * x_j + ... = constants[i]
    coefficients: Vec<Vec<f64>>,
    constants: Vec<f64>,
    // Maps board coordinates to matrix column indices and back
    variables: HashMap<Position, usize>,
    inv_variables: Vec<Position>,
}

impl LinearSystem {
    fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            constants: Vec::new(),
            variables: HashMap::new(),
            inv_variables: Vec::new(),
        }
    }

    fn add_equation(&mut self, coeffs: Vec<f64>, constant: f64) {
        assert_eq!(
            coeffs.len(),
            self.inv_variables.len(),
            "Coefficient vector length must match number of variables"
        );
        self.coefficients.push(coeffs);
        self.constants.push(constant);
    }

    fn solve(&self) -> Vec<(Position, bool)> {
        let mut results = Vec::new();
        if self.coefficients.is_empty() || self.inv_variables.is_empty() {
            return results;
        }

        let mut matrix = self.coefficients.clone();
        let mut constants = self.constants.clone();
        let n = matrix.len();
        let m = self.inv_variables.len();

        // Gaussian elimination
        for i in 0..n.min(m) {
            // Only go up to the smaller dimension
            // Find pivot in column i
            let mut max_val = 0.0;
            let mut max_row = i;
            for j in i..n {
                let abs_val = matrix[j][i].abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    max_row = j;
                }
            }

            // Skip if entire column is effectively zero
            if max_val < 1e-10 {
                continue;
            }

            // Swap rows if necessary
            if max_row != i {
                matrix.swap(i, max_row);
                constants.swap(i, max_row);
            }

            // Normalize pivot row
            let pivot = matrix[i][i];
            for j in i..m {
                matrix[i][j] /= pivot;
            }
            constants[i] /= pivot;

            // Eliminate column in other rows
            for j in 0..n {
                if i != j {
                    let factor = matrix[j][i];
                    for k in i..m {
                        matrix[j][k] -= factor * matrix[i][k];
                    }
                    constants[j] -= factor * constants[i];
                }
            }
        }

        // Special analysis for minesweeper constraints:
        // Look for rows that have been reduced to certainties
        for i in 0..n {
            let mut single_var = None;
            let mut var_count = 0;

            // Count non-zero coefficients in this row
            for j in 0..m {
                if matrix[i][j].abs() > 1e-10 {
                    var_count += 1;
                    single_var = Some(j);
                }
            }

            if var_count == 1 {
                if let Some(j) = single_var {
                    let value = constants[i].round() as i32;
                    // Only consider 0 or 1 as valid solutions (since cells must be mines or not mines)
                    if value == 0 || value == 1 {
                        let pos = self.inv_variables[j];
                        results.push((pos, value == 1));
                    }
                }
            }

            // Look for special cases like all coefficients = 1 and sum = 0 (all safe) or
            // all coefficients = 1 and sum = number of variables (all mines)
            if var_count > 1 {
                let mut all_ones = true;
                let mut vars = Vec::new();
                for j in 0..m {
                    if matrix[i][j].abs() > 1e-10 {
                        if (matrix[i][j] - 1.0).abs() > 1e-10 {
                            all_ones = false;
                            break;
                        }
                        vars.push(j);
                    }
                }
                if all_ones {
                    if constants[i].abs() < 1e-10 {
                        // All variables in this equation must be 0 (safe)
                        for j in vars {
                            let pos = self.inv_variables[j];
                            results.push((pos, false));
                        }
                    } else if (constants[i] - vars.len() as f64).abs() < 1e-10 {
                        // All variables in this equation must be 1 (mines)
                        for j in vars {
                            let pos = self.inv_variables[j];
                            results.push((pos, true));
                        }
                    }
                }
            }
        }

        results
    }
}

pub struct MatrixSolver;

impl MatrixSolver {
    fn build_system(&self, board: &SolverBoard) -> LinearSystem {
        let mut system = LinearSystem::new();
        let mut var_idx = 0;

        // First pass: identify and map all unknown variables
        for pos in board.iter_positions() {
            if let Some(SolverCell::Covered) = board.get(pos) {
                system.variables.insert(pos, var_idx);
                system.inv_variables.push(pos);
                var_idx += 1;
            }
        }

        // If no variables, return empty system
        if var_idx == 0 {
            return system;
        }

        // Second pass: build equations from revealed numbers
        for pos in board.iter_positions() {
            if let Some(SolverCell::Revealed(n)) = board.get(pos) {
                let mut coeffs = vec![0.0; var_idx];
                let mut constant = n as f64;
                let mut has_unknowns = false;

                // Check each neighbor
                for npos in board.neighbors(pos) {
                    match board.get(npos) {
                        Some(SolverCell::Covered) => {
                            let idx = system.variables[&npos];
                            coeffs[idx] = 1.0;
                            has_unknowns = true;
                        }
                        Some(SolverCell::Flagged) => {
                            constant -= 1.0;
                        }
                        _ => {}
                    }
                }

                // Only add equation if it involves unknown variables
                if has_unknowns {
                    system.add_equation(coeffs, constant);
                }
            }
        }

        // Add global mine count constraint if we have it
        let remaining_mines = board.total_mines() - board.mines_marked();
        if remaining_mines > 0 && var_idx > 0 {
            let coeffs = vec![1.0; var_idx];
            system.add_equation(coeffs, remaining_mines as f64);
        }

        system
    }
}

impl Solver for MatrixSolver {
    fn solve(&self, board: &SolverBoard) -> SolverResult {
        // Build and solve linear system
        let system = self.build_system(board);
        let solutions = system.solve();

        // Convert solutions to actions
        let actions = solutions
            .into_iter()
            .map(|(pos, is_mine)| {
                if is_mine {
                    SolverAction::Flag(pos)
                } else {
                    SolverAction::Reveal(pos)
                }
            })
            .collect();

        SolverResult {
            actions,
            certainty: Certainty::Deterministic,
        }
    }

    fn name(&self) -> &str {
        "Matrix Equation Solver"
    }
}
