use std::collections::{HashMap, HashSet};
// Removed unused ndarray imports

use super::{board::SolverCell, Certainty, Solver, SolverAction, SolverBoard, SolverResult};
use crate::Position;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Ordering {
    LessThanEq,
    Equal,
    GreaterThanEq,
}

/// Represents a system of linear equations for solving Minesweeper constraints
#[derive(Debug)]
struct LinearSystem {
    // Each row represents coeffs[i][j] * x_j + ... {op} constants[i]
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

    /// Solves the system using Gaussian elimination with partial pivoting
    /// Returns a vector of (Position, is_mine) pairs for squares that can be definitively determined
    fn solve(&self) -> Vec<(Position, bool)> {
        let mut results = Vec::new();
        if self.coefficients.is_empty() || self.inv_variables.is_empty() {
            return results;
        }

        // First convert inequalities to equalities where possible by combining them
        let (matrix, constants, operators) = self.preprocess_inequalities();

        // Separate equations and inequalities for processing
        let mut eq_matrix = Vec::new();
        let mut eq_constants = Vec::new();
        let mut ineq_matrix = Vec::new();
        let mut ineq_constants = Vec::new();
        let mut ineq_operators = Vec::new();

        for (i, op) in operators.iter().enumerate() {
            if *op == Ordering::Equal {
                eq_matrix.push(matrix[i].clone());
                eq_constants.push(constants[i]);
            } else {
                ineq_matrix.push(matrix[i].clone());
                ineq_constants.push(constants[i]);
                ineq_operators.push(*op);
            }
        }

        // Process equalities using Gaussian elimination
        let mut eq_results = self.solve_equalities(&eq_matrix, &eq_constants);

        // Use inequalities to further constrain results
        self.apply_inequality_constraints(
            &mut eq_results,
            &ineq_matrix,
            &ineq_constants,
            &ineq_operators,
        );

        // Convert results to Position-based format
        for (idx, &value) in eq_results.iter().enumerate() {
            if let Some(v) = value {
                if v == 0.0 || v == 1.0 {
                    results.push((self.inv_variables[idx], v == 1.0));
                }
            }
        }

        results
    }

    fn preprocess_inequalities(&self) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Ordering>) {
        let mut matrix = self.coefficients.clone();
        let mut constants = self.constants.clone();
        let mut operators = self.operators.clone();

        // Combine pairs of inequalities that can form equalities
        let mut i = 0;
        while i < operators.len() {
            let mut j = i + 1;
            while j < operators.len() {
                if self.can_combine_inequalities(
                    &matrix[i],
                    &matrix[j],
                    operators[i],
                    operators[j],
                    constants[i],
                    constants[j],
                ) {
                    // Combine the inequalities
                    let (new_coeffs, new_constant) = self.combine_inequalities(
                        &matrix[i],
                        &matrix[j],
                        constants[i],
                        constants[j],
                    );
                    matrix[i] = new_coeffs;
                    constants[i] = new_constant;
                    operators[i] = Ordering::Equal;

                    // Remove the second inequality
                    matrix.remove(j);
                    constants.remove(j);
                    operators.remove(j);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }

        (matrix, constants, operators)
    }

    fn combine_inequalities(
        &self,
        coeffs1: &[f64],
        _coeffs2: &[f64], // Add underscore
        const1: f64,
        _const2: f64, // Add underscore
    ) -> (Vec<f64>, f64) {
        (coeffs1.to_vec(), const1)
    }

    fn solve_equalities(&self, matrix: &[Vec<f64>], constants: &[f64]) -> Vec<Option<f64>> {
        let n = matrix.len();
        let m = self.inv_variables.len();
        let epsilon = 1e-10;

        let mut aug_matrix = matrix.to_vec();
        let mut aug_constants = constants.to_vec();

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

            // Normalize pivot row - now with stability check
            let pivot = aug_matrix[i][i];
            if pivot.abs() > epsilon {
                for j in i..m {
                    aug_matrix[i][j] /= pivot;
                }
                aug_constants[i] /= pivot;

                // Eliminate column with robust checking
                for j in 0..n {
                    if i != j {
                        let factor = aug_matrix[j][i];
                        if factor.abs() > epsilon {
                            for k in i..m {
                                aug_matrix[j][k] -= factor * aug_matrix[i][k];
                                // Clean up near-zero values
                                if aug_matrix[j][k].abs() < epsilon {
                                    aug_matrix[j][k] = 0.0;
                                }
                            }
                            aug_constants[j] -= factor * aug_constants[i];
                        }
                    }
                }
            }
        }

        // Process integer constraints before back substitution
        self.apply_integer_constraints(&mut aug_matrix, &mut aug_constants);

        let mut results = vec![None; m];
        for i in 0..n {
            let mut single_var = None;
            let mut var_count = 0;

            for j in 0..m {
                if aug_matrix[i][j].abs() > epsilon {
                    var_count += 1;
                    single_var = Some(j);
                }
            }

            if var_count == 1 {
                if let Some(j) = single_var {
                    let value = aug_constants[i];
                    if (value - 0.0).abs() < epsilon || (value - 1.0).abs() < epsilon {
                        results[j] = Some(value);
                    }
                }
            }
        }

        results
    }

    fn apply_integer_constraints(&self, matrix: &mut Vec<Vec<f64>>, constants: &mut Vec<f64>) {
        let epsilon = 1e-10;
        let mut i = 0;
        while i < matrix.len() {
            // Look for equations that must have integer solutions
            let row_sum: f64 = matrix[i].iter().sum();

            // If coefficients sum to integer but constant isn't integer, equation impossible
            if (row_sum.fract().abs() < epsilon) && (constants[i].fract().abs() > epsilon) {
                // This equation is impossible - all variables must be 0
                for j in 0..matrix[i].len() {
                    if matrix[i][j].abs() > epsilon {
                        matrix[i] = vec![0.0; matrix[i].len()];
                        matrix[i][j] = 1.0;
                        constants[i] = 0.0;
                        break;
                    }
                }
            }
            i += 1;
        }
    }

    fn apply_inequality_constraints(
        &self,
        results: &mut Vec<Option<f64>>,
        ineq_matrix: &[Vec<f64>],
        ineq_constants: &[f64],
        ineq_operators: &[Ordering],
    ) {
        // For each inequality, check if it forces any variables
        for (i, coeffs) in ineq_matrix.iter().enumerate() {
            let constant = ineq_constants[i];
            let operator = ineq_operators[i];

            // Count known and unknown variables in this inequality
            let mut sum_known = 0.0;
            let mut unknown_vars = Vec::new();
            let mut unknown_coeffs = Vec::new();

            for (j, coeff) in coeffs.iter().enumerate() {
                if let Some(value) = results[j] {
                    sum_known += coeff * value;
                } else if coeff.abs() > 1e-10 {
                    unknown_vars.push(j);
                    unknown_coeffs.push(*coeff);
                }
            }

            // Check if this inequality forces any values
            if unknown_vars.len() == 1 {
                let idx = unknown_vars[0];
                let coeff = unknown_coeffs[0];

                match operator {
                    Ordering::LessThanEq => {
                        let bound = (constant - sum_known) / coeff;
                        if bound <= 0.0 {
                            results[idx] = Some(0.0);
                        } else if bound <= 1.0 && coeff > 0.0 {
                            results[idx] = Some(0.0);
                        }
                    }
                    Ordering::GreaterThanEq => {
                        let bound = (constant - sum_known) / coeff;
                        if bound >= 1.0 {
                            results[idx] = Some(1.0);
                        } else if bound >= 0.0 && coeff < 0.0 {
                            results[idx] = Some(1.0);
                        }
                    }
                    Ordering::Equal => {
                        let value = (constant - sum_known) / coeff;
                        if (value - 0.0).abs() < 1e-10 || (value - 1.0).abs() < 1e-10 {
                            results[idx] = Some(value);
                        }
                    }
                }
            }
        }
    }

    fn can_combine_inequalities(
        &self,
        coeffs1: &[f64],
        coeffs2: &[f64],
        op1: Ordering,
        op2: Ordering,
        const1: f64,
        const2: f64,
    ) -> bool {
        let epsilon = 1e-10;

        // Case 1: Opposite inequalities forming equality
        if (op1 == Ordering::LessThanEq && op2 == Ordering::GreaterThanEq)
            || (op1 == Ordering::GreaterThanEq && op2 == Ordering::LessThanEq)
        {
            // Check coefficients and constants match
            coeffs1
                .iter()
                .zip(coeffs2.iter())
                .all(|(c1, c2)| (c1 - c2).abs() < epsilon)
                && (const1 - const2).abs() < epsilon
        }
        // Case 2: Identical inequalities can be merged
        else if op1 == op2 {
            coeffs1
                .iter()
                .zip(coeffs2.iter())
                .all(|(c1, c2)| (c1 - c2).abs() < epsilon)
                && (const1 - const2).abs() < epsilon
        } else {
            false
        }
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

    fn build_component_system(&self, board: &SolverBoard, component: &Component) -> LinearSystem {
        let mut system = LinearSystem::new();

        // Create variables for each unknown square (as before)
        for (idx, &pos) in component.positions.iter().enumerate() {
            system.variables.insert(pos, idx);
            system.inv_variables.push(pos);
        }

        // Add equations and inequalities from each constraint
        for &(pos, value) in &component.constraints {
            let mut coeffs = vec![0.0; system.inv_variables.len()];
            let mut covered_count = 0;
            let mut flagged_count = 0;

            // Count existing flags and build equation
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

            // Add equality constraint: sum = number
            system.add_equation(coeffs.clone(), constant, Ordering::Equal);

            // Add inequality constraints
            if covered_count > 0 {
                // No negative mines: sum ≥ 0
                system.add_equation(coeffs.clone(), 0.0, Ordering::GreaterThanEq);

                // Can't exceed number of covered cells: sum ≤ covered_count
                system.add_equation(coeffs.clone(), covered_count as f64, Ordering::LessThanEq);
            }
        }

        // Add global constraints
        if !component.positions.is_empty() {
            // Global mine count constraint as equality
            let total_mines = board.total_mines();
            let remaining_mines = total_mines - board.mines_marked();
            let coeffs = vec![1.0; system.inv_variables.len()];
            system.add_equation(coeffs.clone(), remaining_mines as f64, Ordering::Equal);

            // Add inequality constraints for global mine count
            // Sum of all variables must be ≥ 0
            system.add_equation(coeffs.clone(), 0.0, Ordering::GreaterThanEq);

            // Sum of all variables must be ≤ number of positions
            system.add_equation(
                coeffs.clone(),
                component.positions.len() as f64,
                Ordering::LessThanEq,
            );
        }

        // Add constraints for overlapping regions
        self.add_overlapping_constraints(board, component, &mut system);

        system
    }

    fn add_overlapping_constraints(
        &self,
        board: &SolverBoard,
        component: &Component,
        system: &mut LinearSystem,
    ) {
        let mut processed = HashSet::new();

        // For each pair of constraints that share variables
        for &(pos1, value1) in &component.constraints {
            for &(pos2, value2) in &component.constraints {
                if pos1 >= pos2 || processed.contains(&(pos1, pos2)) {
                    continue;
                }

                let neighbors1: HashSet<Position> = board
                    .neighbors(pos1)
                    .into_iter()
                    .filter(|p| component.positions.contains(p))
                    .collect();

                let neighbors2: HashSet<Position> = board
                    .neighbors(pos2)
                    .into_iter()
                    .filter(|p| component.positions.contains(p))
                    .collect();

                // Find overlapping region
                let overlap: HashSet<_> = neighbors1.intersection(&neighbors2).cloned().collect();

                if !overlap.is_empty() {
                    // Create coefficient vectors for both constraints
                    let mut coeffs1 = vec![0.0; system.inv_variables.len()];
                    let mut coeffs2 = vec![0.0; system.inv_variables.len()];

                    for &pos in &neighbors1 {
                        if let Some(&idx) = system.variables.get(&pos) {
                            coeffs1[idx] = 1.0;
                        }
                    }

                    for &pos in &neighbors2 {
                        if let Some(&idx) = system.variables.get(&pos) {
                            coeffs2[idx] = 1.0;
                        }
                    }

                    // Add subtraction-based constraints
                    // If we have N1 mines in region1 and N2 mines in region2,
                    // then overlap region must have at least max(0, N1 + N2 - total_squares)
                    // and at most min(N1, N2)
                    let total_squares = (neighbors1.union(&neighbors2).count()) as f64;
                    let min_overlap = (value1 + value2) as f64 - total_squares;
                    if min_overlap > 0.0 {
                        let mut overlap_coeffs = vec![0.0; system.inv_variables.len()];
                        for &pos in &overlap {
                            if let Some(&idx) = system.variables.get(&pos) {
                                overlap_coeffs[idx] = 1.0;
                            }
                        }
                        system.add_equation(
                            overlap_coeffs.clone(),
                            min_overlap,
                            Ordering::GreaterThanEq,
                        );
                    }

                    // The overlap can't contain more mines than either constraint
                    let max_overlap = value1.min(value2) as f64;
                    let mut overlap_coeffs = vec![0.0; system.inv_variables.len()];
                    for &pos in &overlap {
                        if let Some(&idx) = system.variables.get(&pos) {
                            overlap_coeffs[idx] = 1.0;
                        }
                    }
                    system.add_equation(overlap_coeffs, max_overlap, Ordering::LessThanEq);

                    processed.insert((pos1, pos2));
                }
            }
        }
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
