use std::collections::{HashMap, HashSet};

use super::{board::SolverCell, Certainty, Solver, SolverAction, SolverBoard, SolverResult};
use crate::Position;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Ordering {
    LessThanEq,
    Equal,
    GreaterThanEq,
}

#[derive(Debug)]
struct LinearSystem {
    coefficients: Vec<Vec<f64>>,
    constants: Vec<f64>,
    operators: Vec<Ordering>,
    variables: HashMap<Position, usize>,
    inv_variables: Vec<Position>,
}

impl LinearSystem {
    fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            constants: Vec::new(),
            operators: Vec::new(),
            variables: HashMap::new(),
            inv_variables: Vec::new(),
        }
    }

    fn add_equation(&mut self, coeffs: Vec<f64>, constant: f64, operator: Ordering) {
        assert_eq!(
            coeffs.len(),
            self.inv_variables.len(),
            "Coefficient vector length must match number of variables"
        );
        self.coefficients.push(coeffs);
        self.constants.push(constant);
        self.operators.push(operator);
    }

    fn solve(&self) -> Vec<(Position, bool)> {
        let mut results = Vec::new();
        if self.coefficients.is_empty() || self.inv_variables.is_empty() {
            return results;
        }

        // First process pure equality constraints
        let mut eq_matrix = Vec::new();
        let mut eq_constants = Vec::new();
        let mut ineq_matrix = Vec::new();
        let mut ineq_constants = Vec::new();
        let mut ineq_operators = Vec::new();

        for (i, op) in self.operators.iter().enumerate() {
            match op {
                Ordering::Equal => {
                    eq_matrix.push(self.coefficients[i].clone());
                    eq_constants.push(self.constants[i]);
                }
                _ => {
                    ineq_matrix.push(self.coefficients[i].clone());
                    ineq_constants.push(self.constants[i]);
                    ineq_operators.push(*op);
                }
            }
        }

        // Solve equality system first
        let mut eq_results = self.solve_equalities(&eq_matrix, &eq_constants);

        // Use inequalities to further constrain results
        if !ineq_matrix.is_empty() {
            self.apply_inequality_constraints(
                &mut eq_results,
                &ineq_matrix,
                &ineq_constants,
                &ineq_operators,
            );
        }

        // Convert results to Position-based format
        for (idx, &value) in eq_results.iter().enumerate() {
            if let Some(v) = value {
                if (v - 0.0).abs() < 1e-10 || (v - 1.0).abs() < 1e-10 {
                    results.push((self.inv_variables[idx], (v - 1.0).abs() < 1e-10));
                }
            }
        }

        results
    }

    fn solve_equalities(&self, matrix: &[Vec<f64>], constants: &[f64]) -> Vec<Option<f64>> {
        let n = matrix.len();
        let m = self.inv_variables.len();
        let epsilon = 1e-10;

        let mut aug_matrix = matrix.to_vec();
        let mut aug_constants = constants.to_vec();
        let mut results = vec![None; m];

        // Gaussian elimination with partial pivoting
        for i in 0..n.min(m) {
            // Find pivot
            let mut max_val = 0.0;
            let mut max_row = i;
            for j in i..n {
                let abs_val = aug_matrix[j][i].abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    max_row = j;
                }
            }

            if max_val < epsilon {
                continue;
            }

            // Swap rows if necessary
            if max_row != i {
                aug_matrix.swap(i, max_row);
                aug_constants.swap(i, max_row);
            }

            // Normalize pivot row
            let pivot = aug_matrix[i][i];
            if pivot.abs() > epsilon {
                for j in i..m {
                    aug_matrix[i][j] /= pivot;
                }
                aug_constants[i] /= pivot;

                // Eliminate column
                for j in 0..n {
                    if i != j {
                        let factor = aug_matrix[j][i];
                        for k in i..m {
                            aug_matrix[j][k] -= factor * aug_matrix[i][k];
                            if aug_matrix[j][k].abs() < epsilon {
                                aug_matrix[j][k] = 0.0;
                            }
                        }
                        aug_constants[j] -= factor * aug_constants[i];
                    }
                }
            }
        }

        // Process integer constraints
        self.process_integer_constraints(&mut aug_matrix, &mut aug_constants);

        // Back substitution and result extraction
        for i in 0..n {
            let mut single_var = None;
            let mut var_count = 0;
            let mut all_ones = true;

            for j in 0..m {
                if aug_matrix[i][j].abs() > epsilon {
                    var_count += 1;
                    single_var = Some(j);
                    if (aug_matrix[i][j] - 1.0).abs() > epsilon {
                        all_ones = false;
                    }
                }
            }

            if var_count == 1 {
                if let Some(j) = single_var {
                    let value = aug_constants[i];
                    if value.abs() < epsilon || (value - 1.0).abs() < epsilon {
                        results[j] = Some(value);
                    }
                }
            } else if all_ones {
                let constant = aug_constants[i];
                if constant.abs() < epsilon {
                    // All variables must be 0
                    for j in 0..m {
                        if aug_matrix[i][j].abs() > epsilon {
                            results[j] = Some(0.0);
                        }
                    }
                } else if (constant - var_count as f64).abs() < epsilon {
                    // All variables must be 1
                    for j in 0..m {
                        if aug_matrix[i][j].abs() > epsilon {
                            results[j] = Some(1.0);
                        }
                    }
                }
            }
        }

        results
    }

    fn process_integer_constraints(&self, matrix: &mut Vec<Vec<f64>>, constants: &mut Vec<f64>) {
        let epsilon = 1e-10;
        for i in 0..matrix.len() {
            let row_sum: f64 = matrix[i].iter().sum();

            // Check if this row represents an integer constraint
            if (row_sum.round() - row_sum).abs() < epsilon {
                let constant = constants[i];
                if (constant.round() - constant).abs() >= epsilon {
                    // This equation is impossible - force variables to 0
                    for j in 0..matrix[i].len() {
                        if matrix[i][j].abs() > epsilon {
                            matrix[i] = vec![0.0; matrix[i].len()];
                            matrix[i][j] = 1.0;
                            constants[i] = 0.0;
                            break;
                        }
                    }
                }
            }
        }
    }

    fn apply_inequality_constraints(
        &self,
        results: &mut Vec<Option<f64>>,
        ineq_matrix: &[Vec<f64>],
        ineq_constants: &[f64],
        ineq_operators: &[Ordering],
    ) {
        let epsilon = 1e-10;

        for (i, coeffs) in ineq_matrix.iter().enumerate() {
            let constant = ineq_constants[i];
            let operator = ineq_operators[i];

            // Count known and unknown variables
            let mut sum_known = 0.0;
            let mut unknown_vars = Vec::new();
            let mut unknown_coeffs = Vec::new();

            for (j, coeff) in coeffs.iter().enumerate() {
                if coeff.abs() < epsilon {
                    continue;
                }

                if let Some(value) = results[j] {
                    sum_known += coeff * value;
                } else {
                    unknown_vars.push(j);
                    unknown_coeffs.push(*coeff);
                }
            }

            // Special case: single unknown variable
            if unknown_vars.len() == 1 {
                let idx = unknown_vars[0];
                let coeff = unknown_coeffs[0];
                let remaining = constant - sum_known;

                match operator {
                    Ordering::LessThanEq => {
                        if coeff > 0.0 && remaining / coeff <= epsilon {
                            results[idx] = Some(0.0);
                        }
                    }
                    Ordering::GreaterThanEq => {
                        if coeff > 0.0 && remaining / coeff >= 1.0 - epsilon {
                            results[idx] = Some(1.0);
                        }
                    }
                    Ordering::Equal => {
                        let value = remaining / coeff;
                        if value.abs() < epsilon || (value - 1.0).abs() < epsilon {
                            results[idx] = Some(value);
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
struct Component {
    positions: HashSet<Position>,
    constraints: Vec<(Position, u8)>,
}

pub struct MatrixSolver;

impl MatrixSolver {
    fn find_components(&self, board: &SolverBoard) -> Vec<Component> {
        let mut components = Vec::new();
        let mut visited = HashSet::new();

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
                    for npos in board.neighbors(pos) {
                        if !visited.contains(&npos) {
                            explore_component(npos, board, component, visited);
                        }
                    }
                }
                Some(SolverCell::Revealed(n)) => {
                    component.constraints.push((pos, n));
                    for npos in board.neighbors(pos) {
                        if !visited.contains(&npos) {
                            explore_component(npos, board, component, visited);
                        }
                    }
                }
                _ => {}
            }
        }

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

        self.merge_components(&mut components);
        components
    }

    fn merge_components(&self, components: &mut Vec<Component>) {
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
                if components[j]
                    .constraints
                    .iter()
                    .any(|&(pos, _)| component_i_constraints.contains(&pos))
                {
                    let component_j = components.remove(j);
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
    }

    fn build_component_system(&self, board: &SolverBoard, component: &Component) -> LinearSystem {
        let mut system = LinearSystem::new();

        // Create variables for each unknown square
        for (idx, &pos) in component.positions.iter().enumerate() {
            system.variables.insert(pos, idx);
            system.inv_variables.push(pos);
        }

        // Add equations and inequalities from constraints
        for &(pos, value) in &component.constraints {
            let mut coeffs = vec![0.0; system.inv_variables.len()];
            let mut covered_count = 0;
            let mut flagged_count = 0;

            for npos in board.neighbors(pos) {
                match board.get(npos) {
                    Some(SolverCell::Covered) if component.positions.contains(&npos) => {
                        coeffs[system.variables[&npos]] = 1.0;
                        covered_count += 1;
                    }
                    Some(SolverCell::Flagged) => {
                        flagged_count += 1;
                    }
                    _ => {}
                }
            }

            let constant = value as f64 - flagged_count as f64;

            // Add equality constraint
            system.add_equation(coeffs.clone(), constant, Ordering::Equal);

            // Add inequality constraints
            if covered_count > 0 {
                system.add_equation(coeffs.clone(), 0.0, Ordering::GreaterThanEq);
                system.add_equation(coeffs.clone(), covered_count as f64, Ordering::LessThanEq);
            }
        }

        // Add global constraints for mine count
        if !component.positions.is_empty() {
            let total_mines = board.total_mines();
            let remaining_mines = total_mines - board.mines_marked();
            let coeffs = vec![1.0; system.inv_variables.len()];

            // Total mines constraint as equality
            system.add_equation(coeffs.clone(), remaining_mines as f64, Ordering::Equal);

            // Basic bounds
            system.add_equation(coeffs.clone(), 0.0, Ordering::GreaterThanEq);
            system.add_equation(
                coeffs.clone(),
                component.positions.len() as f64,
                Ordering::LessThanEq,
            );
        }

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
    use crate::{Board, Cell};

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

        system.add_equation(vec![1.0, 1.0], 1.0, Ordering::Equal); // x + y = 1
        system.add_equation(vec![1.0, 0.0], 1.0, Ordering::Equal); // x = 1

        let solutions = system.solve();
        assert_eq!(solutions.len(), 2);
        assert!(solutions.contains(&(Position::new(0, 0), true)));
        assert!(solutions.contains(&(Position::new(0, 1), false)));
    }

    #[test]
    fn test_merge_components() {
        let mut board = Board::new(4, 4, 0).unwrap(); // Start with no mines

        // Manually insert numbers to create a guaranteed shared constraint
        board.cells.insert(Position::new(1, 1), Cell::Revealed(1));
        board.cells.insert(Position::new(2, 1), Cell::Revealed(1));

        let solver_board = SolverBoard::new(&board);
        let solver = MatrixSolver;
        let components = solver.find_components(&solver_board);

        assert_eq!(components.len(), 1, "Expected one merged component");
    }
}
