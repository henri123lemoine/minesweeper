use std::collections::{HashMap, HashSet};
// Removed unused ndarray imports

use super::{board::SolverCell, Certainty, Solver, SolverAction, SolverBoard, SolverResult};
use crate::Position;

/// Represents a system of linear equations for solving Minesweeper constraints
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

    /// Solves the system using Gaussian elimination with partial pivoting
    /// Returns a vector of (Position, is_mine) pairs for squares that can be definitively determined
    fn solve(&self) -> Vec<(Position, bool)> {
        let mut results = Vec::new();
        if self.coefficients.is_empty() || self.inv_variables.is_empty() {
            return results;
        }

        let mut matrix = self.coefficients.clone();
        let mut constants = self.constants.clone();
        let n = matrix.len();
        let m = self.inv_variables.len();

        // Gaussian elimination with partial pivoting
        for i in 0..n.min(m) {
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

        // Analyze results for minesweeper-specific constraints
        for i in 0..n {
            // Count non-zero coefficients in this row
            let mut single_var = None;
            let mut var_count = 0;
            let mut all_ones = true;
            let mut vars = Vec::new();

            for j in 0..m {
                if matrix[i][j].abs() > 1e-10 {
                    var_count += 1;
                    single_var = Some(j);
                    if (matrix[i][j] - 1.0).abs() > 1e-10 {
                        all_ones = false;
                    }
                    vars.push(j);
                }
            }

            // Case 1: Single variable equations
            if var_count == 1 {
                if let Some(j) = single_var {
                    let value = constants[i].round() as i32;
                    if value == 0 || value == 1 {
                        let pos = self.inv_variables[j];
                        results.push((pos, value == 1));
                    }
                }
            }

            // Case 2: Special cases with all ones
            if var_count > 1 && all_ones {
                if constants[i].abs() < 1e-10 {
                    // All variables must be 0 (safe)
                    for j in vars {
                        let pos = self.inv_variables[j];
                        results.push((pos, false));
                    }
                } else if (constants[i] - vars.len() as f64).abs() < 1e-10 {
                    // All variables must be 1 (mines)
                    for j in vars {
                        let pos = self.inv_variables[j];
                        results.push((pos, true));
                    }
                }
            }
        }

        results
    }
}

/// Represents a connected component in the Minesweeper board
#[derive(Debug)]
struct Component {
    positions: HashSet<Position>,
    constraints: Vec<(Position, u8)>, // Position and value of constraining numbers
}

pub struct MatrixSolver;

impl MatrixSolver {
    /// Identifies connected components in the board
    fn find_components(&self, board: &SolverBoard) -> Vec<Component> {
        let mut components = Vec::new();
        let mut visited = HashSet::new();

        // Helper function for DFS component finding
        fn explore_component(
            pos: Position,
            board: &SolverBoard,
            component: &mut Component,
            visited: &mut HashSet<Position>,
        ) {
            if !visited.insert(pos) {
                return;
            }

            match board.get(pos) {
                Some(SolverCell::Covered) => {
                    component.positions.insert(pos);
                    // Explore neighbors to find connecting constraints
                    for npos in board.neighbors(pos) {
                        if !visited.contains(&npos) {
                            explore_component(npos, board, component, visited);
                        }
                    }
                }
                Some(SolverCell::Revealed(n)) => {
                    component.constraints.push((pos, n));
                    // Explore all neighbors
                    for npos in board.neighbors(pos) {
                        if !visited.contains(&npos) {
                            explore_component(npos, board, component, visited);
                        }
                    }
                }
                _ => {}
            }
        }

        // Find all components
        for pos in board.iter_positions() {
            if !visited.contains(&pos) {
                if let Some(SolverCell::Covered) = board.get(pos) {
                    let mut component = Component {
                        positions: HashSet::new(),
                        constraints: Vec::new(),
                    };
                    explore_component(pos, board, &mut component, &mut visited);
                    if !component.positions.is_empty() {
                        components.push(component);
                    }
                }
            }
        }

        // Merge components that share constraints
        let mut i = 0;
        while i < components.len() {
            let mut merged = false;
            let component_i_constraints: HashSet<_> = components[i]
                .constraints
                .iter()
                .map(|&(pos, _)| pos)
                .collect();

            let mut j = i + 1;
            while j < components.len() {
                // Check if components share any constraints
                let shares_constraint = components[j]
                    .constraints
                    .iter()
                    .any(|&(pos, _)| component_i_constraints.contains(&pos));

                if shares_constraint {
                    // Take ownership of component j
                    let component_j = components.remove(j);
                    // Merge into component i
                    components[i].positions.extend(component_j.positions);
                    components[i].constraints.extend(component_j.constraints);
                    merged = true;
                } else {
                    j += 1;
                }
            }

            if !merged {
                i += 1;
            }
        }

        components
    }

    /// Builds a linear system for a single component
    fn build_component_system(&self, board: &SolverBoard, component: &Component) -> LinearSystem {
        let mut system = LinearSystem::new();

        // Create variables for each unknown square
        for (idx, &pos) in component.positions.iter().enumerate() {
            system.variables.insert(pos, idx);
            system.inv_variables.push(pos);
        }

        // Add equations from constraints
        for &(pos, value) in &component.constraints {
            let mut coeffs = vec![0.0; system.inv_variables.len()];
            let mut constant = value as f64;

            // Count existing flags and build equation
            for npos in board.neighbors(pos) {
                match board.get(npos) {
                    Some(SolverCell::Covered) if component.positions.contains(&npos) => {
                        coeffs[system.variables[&npos]] = 1.0;
                    }
                    Some(SolverCell::Flagged) => {
                        constant -= 1.0;
                    }
                    _ => {}
                }
            }

            system.add_equation(coeffs, constant);
        }

        // Add global mine count constraint
        let total_mines = board.total_mines();
        let remaining_mines = total_mines - board.mines_marked();
        let coeffs = vec![1.0; system.inv_variables.len()];
        system.add_equation(coeffs, remaining_mines as f64);

        system
    }
}

impl Solver for MatrixSolver {
    fn solve(&self, board: &SolverBoard) -> SolverResult {
        let mut actions = Vec::new();

        // Find all components
        let components = self.find_components(board);

        // Solve each component
        for component in components {
            let system = self.build_component_system(board, &component);
            let solutions = system.solve();

            // Convert solutions to actions
            for (pos, is_mine) in solutions {
                let action = if is_mine {
                    SolverAction::Flag(pos)
                } else {
                    SolverAction::Reveal(pos)
                };
                actions.push(action);
            }
        }

        SolverResult {
            actions,
            certainty: Certainty::Deterministic,
        }
    }

    fn name(&self) -> &str {
        "Matrix Equation Solver"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Board, RevealResult};

    #[test]
    fn test_simple_component() {
        let mut board = Board::new(3, 3, 1).unwrap();
        // Set up a simple board with one known mine
        board.reveal(Position::new(1, 1)).unwrap();

        let solver_board = SolverBoard::new(&board);
        let solver = MatrixSolver;
        let components = solver.find_components(&solver_board);

        assert_eq!(components.len(), 1);
        let component = &components[0];
        assert!(component.positions.contains(&Position::new(0, 0)));
        assert!(component
            .constraints
            .iter()
            .any(|&(pos, _)| pos == Position::new(1, 1)));
    }

    #[test]
    fn test_linear_system_solution() {
        let mut system = LinearSystem::new();

        // Add test equations
        system.inv_variables.push(Position::new(0, 0));
        system.inv_variables.push(Position::new(0, 1));
        system.variables.insert(Position::new(0, 0), 0);
        system.variables.insert(Position::new(0, 1), 1);

        system.add_equation(vec![1.0, 1.0], 1.0); // x + y = 1
        system.add_equation(vec![1.0, 0.0], 1.0); // x = 1

        let solutions = system.solve();
        assert_eq!(solutions.len(), 2);
        assert!(solutions.contains(&(Position::new(0, 0), true)));
        assert!(solutions.contains(&(Position::new(0, 1), false)));
    }

    #[test]
    fn test_merge_components() {
        // Create a board with just 1 mine to minimize interference
        let mut board = Board::new(4, 4, 1).unwrap();

        // First reveal a "safe" position in the middle to establish a pattern
        match board.reveal(Position::new(1, 1)) {
            Ok(RevealResult::Mine) => {
                // If we hit a mine, try the adjacent position
                board.reveal(Position::new(2, 1)).unwrap();
            }
            _ => {
                // If first position was safe, reveal the adjacent one
                board.reveal(Position::new(2, 1)).unwrap();
            }
        }

        let solver_board = SolverBoard::new(&board);
        let solver = MatrixSolver;
        let components = solver.find_components(&solver_board);

        // Should be merged into a single component due to shared constraints
        assert_eq!(components.len(), 1, "Expected one merged component");

        // Verify that at least one of our revealed positions is in the constraints
        let constraint_positions: HashSet<Position> = components[0]
            .constraints
            .iter()
            .map(|(pos, _)| *pos)
            .collect();

        assert!(
            constraint_positions.contains(&Position::new(1, 1))
                || constraint_positions.contains(&Position::new(2, 1)),
            "Component should contain at least one of the revealed positions"
        );
    }
}
