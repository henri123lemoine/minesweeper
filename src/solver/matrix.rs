use super::board::{SolverBoard, SolverCell};
use super::solver_test_suite;
use super::traits::{DeterministicResult, DeterministicSolver, Solver};
use crate::Position;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq)]
enum Ordering {
    LessThanEq,
    Equal,
    GreaterThanEq,
}

/// Represents a system of linear equations in the context of Minesweeper
#[derive(Debug)]
struct LinearSystem {
    /// Matrix of coefficients where each row represents an equation
    coefficients: Vec<Vec<f64>>,
    /// Right-hand side constants for each equation
    constants: Vec<f64>,
    /// Types of constraints (equality or inequality)
    operators: Vec<Ordering>,
    /// Maps board positions to matrix column indices
    position_to_var: HashMap<Position, usize>,
    /// Maps matrix column indices back to board positions
    var_to_position: Vec<Position>,
}

impl LinearSystem {
    fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            constants: Vec::new(),
            operators: Vec::new(),
            position_to_var: HashMap::new(),
            var_to_position: Vec::new(),
        }
    }

    /// Adds a new variable (board position) to the system
    fn add_variable(&mut self, pos: Position) -> usize {
        if let Some(&idx) = self.position_to_var.get(&pos) {
            return idx;
        }
        let idx = self.var_to_position.len();
        self.position_to_var.insert(pos, idx);
        self.var_to_position.push(pos);
        idx
    }

    /// Adds a new equation to the system: coefficients * variables = constant
    fn add_equation(&mut self, coeffs: Vec<f64>, constant: f64, operator: Ordering) {
        assert_eq!(
            coeffs.len(),
            self.var_to_position.len(),
            "Coefficient vector length must match number of variables"
        );
        self.coefficients.push(coeffs);
        self.constants.push(constant);
        self.operators.push(operator);
    }

    /// Solves the system using Gaussian elimination with partial pivoting
    /// Returns positions that are definitely mines or definitely safe
    fn solve(&self) -> DeterministicResult {
        let mut result = DeterministicResult::default();
        if self.coefficients.is_empty() || self.var_to_position.is_empty() {
            return result;
        }

        // Separate equality and inequality constraints
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

        // Apply inequality constraints
        if !ineq_matrix.is_empty() {
            self.apply_inequality_constraints(
                &mut eq_results,
                &ineq_matrix,
                &ineq_constants,
                &ineq_operators,
            );
        }

        // Convert results to definite mines and safe positions
        for (idx, &value) in eq_results.iter().enumerate() {
            if let Some(v) = value {
                let pos = self.var_to_position[idx];
                if v.abs() < 1e-10 {
                    result.safe.insert(pos);
                } else if (v - 1.0).abs() < 1e-10 {
                    result.mines.insert(pos);
                }
            }
        }

        result
    }

    fn solve_equalities(&self, matrix: &[Vec<f64>], constants: &[f64]) -> Vec<Option<f64>> {
        let n = matrix.len();
        let m = self.var_to_position.len();
        let epsilon = 1e-10;

        let mut aug_matrix = matrix.to_vec();
        let mut aug_constants = constants.to_vec();
        let mut results = vec![None; m];

        // Gaussian elimination with partial pivoting
        for i in 0..n.min(m) {
            // Find pivot
            let mut max_val = 0.0;
            let mut max_row = i;

            for (j, row) in aug_matrix.iter().enumerate().take(n).skip(i) {
                let abs_val = row[i].abs();
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

                // Store the pivot row values we need
                let pivot_row: Vec<f64> = aug_matrix[i][i..m].to_vec();

                // Eliminate column
                for j in 0..n {
                    if i != j {
                        let factor = aug_matrix[j][i];
                        for k in i..m {
                            aug_matrix[j][k] -= factor * pivot_row[k - i];
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
        for (i, row) in aug_matrix.iter().enumerate().take(n) {
            let mut single_var = None;
            let mut var_count = 0;
            let mut all_ones = true;

            for (j, &coeff) in row.iter().enumerate() {
                if coeff.abs() > epsilon {
                    var_count += 1;
                    single_var = Some(j);
                    if (coeff - 1.0).abs() > epsilon {
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
                    for (j, &coeff) in row.iter().enumerate() {
                        if coeff.abs() > epsilon {
                            results[j] = Some(0.0);
                        }
                    }
                } else if (constant - var_count as f64).abs() < epsilon {
                    // All variables must be 1
                    for (j, &coeff) in row.iter().enumerate() {
                        if coeff.abs() > epsilon {
                            results[j] = Some(1.0);
                        }
                    }
                }
            }
        }

        results
    }

    fn process_integer_constraints(&self, matrix: &mut [Vec<f64>], constants: &mut [f64]) {
        let epsilon = 1e-10;
        for (i, row) in matrix.iter_mut().enumerate() {
            let row_sum: f64 = row.iter().sum();

            // Check if this row represents an integer constraint
            if (row_sum.round() - row_sum).abs() < epsilon {
                let constant = constants[i];
                if (constant.round() - constant).abs() >= epsilon {
                    // This equation is impossible - force variables to 0
                    for (j, val) in row.iter_mut().enumerate() {
                        if val.abs() > epsilon {
                            *row = vec![0.0; row.len()];
                            row[j] = 1.0;
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
        results: &mut [Option<f64>],
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

/// Represents a connected component of the Minesweeper board
#[derive(Debug)]
struct Component {
    /// Unknown positions in this component
    positions: HashSet<Position>,
    /// Known number constraints (position and value)
    constraints: Vec<(Position, u8)>,
}

#[derive(Debug, Default)]
pub struct MatrixSolver;

impl MatrixSolver {
    /// Finds connected components of unknown cells and their constraints
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
                    // Explore neighbors for connecting constraints
                    for npos in board.neighbors(pos) {
                        if let Some(SolverCell::Revealed(_)) = board.get(npos) {
                            explore_component(npos, board, component, visited);
                        }
                    }
                }
                Some(SolverCell::Revealed(n)) => {
                    component.constraints.push((pos, n));
                    // Continue exploration from neighbors
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

        self.merge_components(&mut components);
        components
    }

    /// Merges components that share constraints
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
                    // Merge component j into i
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

    /// Builds linear system for a component
    fn build_component_system(&self, board: &SolverBoard, component: &Component) -> LinearSystem {
        let mut system = LinearSystem::new();

        // Create variables for each unknown position
        for &pos in &component.positions {
            system.add_variable(pos);
        }

        // Add equations from number constraints
        for &(pos, value) in &component.constraints {
            let mut coeffs = vec![0.0; system.var_to_position.len()];
            let mut mine_count = 0;

            for npos in board.neighbors(pos) {
                match board.get(npos) {
                    Some(SolverCell::Covered) if component.positions.contains(&npos) => {
                        coeffs[system.position_to_var[&npos]] = 1.0;
                    }
                    Some(SolverCell::Flagged) => mine_count += 1,
                    _ => {}
                }
            }

            let constant = value as f64 - mine_count as f64;

            // Add equality constraint
            system.add_equation(coeffs.clone(), constant, Ordering::Equal);

            // Add inequality constraints
            if !coeffs.iter().all(|&x| x.abs() < 1e-10) {
                system.add_equation(coeffs.clone(), 0.0, Ordering::GreaterThanEq);
                let sum: f64 = coeffs.iter().sum();
                system.add_equation(coeffs.clone(), sum, Ordering::LessThanEq);
            }
        }

        // Add global mine count constraint if available, properly scoped to this component
        if let Some(total_remaining_mines) = board.remaining_mines() {
            // Count how many covered cells are not in this component
            let mut other_covered_count = 0;
            for pos in board.iter_positions() {
                if let Some(SolverCell::Covered) = board.get(pos) {
                    if !component.positions.contains(&pos) {
                        other_covered_count += 1;
                    }
                }
            }

            // We can now add constraints based on minimum and maximum possible mines in this component
            let coeffs = vec![1.0; system.var_to_position.len()];

            // At minimum, this component must contain max(0, total_remaining - other_covered)
            let min_mines = (total_remaining_mines as i32 - other_covered_count).max(0);
            system.add_equation(coeffs.clone(), min_mines as f64, Ordering::GreaterThanEq);

            // At maximum, this component can contain min(total_remaining, component_size)
            let max_mines = total_remaining_mines.min(component.positions.len() as u32);
            system.add_equation(coeffs.clone(), max_mines as f64, Ordering::LessThanEq);
        }

        system
    }
}

impl Solver for MatrixSolver {
    fn name(&self) -> &str {
        "Matrix Equation Solver"
    }
}

impl DeterministicSolver for MatrixSolver {
    fn solve(&self, board: &SolverBoard) -> DeterministicResult {
        let mut result = DeterministicResult::default();

        // Find all components
        let components = self.find_components(board);

        // Solve each component
        for component in components {
            let system = self.build_component_system(board, &component);
            let component_result = system.solve();
            result.mines.extend(component_result.mines);
            result.safe.extend(component_result.safe);
        }

        result
    }
}

solver_test_suite!(MatrixSolver, deterministic);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Board, Cell};

    /// Helper function to create a board with known mine positions
    fn create_test_board(width: u32, height: u32, mines: &[(i32, i32)]) -> Board {
        let mut board = Board::new(width, height, mines.len() as u32).unwrap();
        board.cells.clear();

        // Initialize all cells as non-mines
        for x in 0..width as i32 {
            for y in 0..height as i32 {
                board.cells.insert(Position::new(x, y), Cell::Hidden(false));
            }
        }

        // Place mines at specified positions
        for &(x, y) in mines {
            board.cells.insert(Position::new(x, y), Cell::Hidden(true));
        }

        board
    }

    #[test]
    fn test_linear_system_simple() {
        let mut system = LinearSystem::new();

        // Create a simple 2x2 system: x + y = 1, x = 1
        let pos1 = Position::new(0, 0);
        let pos2 = Position::new(0, 1);
        system.add_variable(pos1);
        system.add_variable(pos2);

        system.add_equation(vec![1.0, 1.0], 1.0, Ordering::Equal);
        system.add_equation(vec![1.0, 0.0], 1.0, Ordering::Equal);

        let result = system.solve();

        assert!(
            result.mines.contains(&pos1),
            "Position (0,0) should be a mine"
        );
        assert!(result.safe.contains(&pos2), "Position (0,1) should be safe");
    }

    #[test]
    fn test_component_isolation() {
        // Let's make a board with a true gap:
        // [1|?|_|1|?]  (_ is empty space/gap)
        // [?|?|_|?|?]
        let mut board = create_test_board(5, 2, &[(0, 0), (3, 0)]);
        // Create a gap in the middle
        board.cells.remove(&Position::new(2, 0));
        board.cells.remove(&Position::new(2, 1));
        board.reveal(Position::new(0, 0)).unwrap();
        board.reveal(Position::new(3, 0)).unwrap();

        let solver = MatrixSolver;
        let solver_board = SolverBoard::new(&board);
        let components = solver.find_components(&solver_board);

        assert_eq!(
            components.len(),
            2,
            "Should find exactly two separate components"
        );

        // Verify components don't overlap
        let component1_positions: HashSet<_> = components[0].positions.iter().cloned().collect();
        let component2_positions: HashSet<_> = components[1].positions.iter().cloned().collect();
        assert!(
            component1_positions.is_disjoint(&component2_positions),
            "Components should not share positions"
        );
    }

    #[test]
    fn test_component_merging() {
        // Create a board with components that should merge due to shared constraints
        // [1|?|1]
        // [?|?|?]
        let mut board = create_test_board(3, 2, &[(0, 0), (2, 0)]);
        board.reveal(Position::new(0, 0)).unwrap();
        board.reveal(Position::new(2, 0)).unwrap();

        let solver = MatrixSolver;
        let solver_board = SolverBoard::new(&board);
        let components = solver.find_components(&solver_board);

        assert_eq!(
            components.len(),
            1,
            "Components sharing constraints should be merged into one"
        );
        assert!(
            components[0].positions.len() > 3,
            "Merged component should contain all connecting positions"
        );
    }

    #[test]
    fn test_numerical_stability() {
        // Create a system with potential numerical stability issues
        let mut system = LinearSystem::new();

        // Add several variables
        for i in 0..5 {
            system.add_variable(Position::new(i, 0));
        }

        // Add equations that could cause numerical instability
        system.add_equation(vec![0.1, 0.1, 0.1, 0.1, 0.1], 0.5, Ordering::Equal);
        system.add_equation(vec![0.01, 0.01, 0.01, 0.01, 0.01], 0.05, Ordering::Equal);

        let result = system.solve();
        assert!(
            !result.mines.is_empty() || !result.safe.is_empty(),
            "Should handle small coefficients properly"
        );
    }

    #[test]
    fn test_overdetermined_system() {
        // Create an overdetermined system with consistent equations
        let mut system = LinearSystem::new();

        let pos = Position::new(0, 0);
        system.add_variable(pos);

        // Multiple constraints that agree: x = 1
        system.add_equation(vec![1.0], 1.0, Ordering::Equal);
        system.add_equation(vec![1.0], 1.0, Ordering::Equal);
        system.add_equation(vec![2.0], 2.0, Ordering::Equal);

        let result = system.solve();
        assert!(
            result.mines.contains(&pos),
            "Should correctly solve overdetermined system"
        );
    }

    #[test]
    fn test_inequality_constraints() {
        let mut system = LinearSystem::new();

        let pos = Position::new(0, 0);
        system.add_variable(pos);

        // Add constraints: x ≤ 1, x ≥ 1
        system.add_equation(vec![1.0], 1.0, Ordering::LessThanEq);
        system.add_equation(vec![1.0], 1.0, Ordering::GreaterThanEq);

        let result = system.solve();
        assert!(
            result.mines.contains(&pos),
            "Should deduce x=1 from inequalities"
        );
    }

    #[test]
    fn test_complex_board() {
        // Create a configuration that must have a deterministic solution:
        // [1|2|1]
        // [#|2|#]  (# indicates revealed mine count)
        // [?|S|?]  (S must be safe)
        let mut board = create_test_board(3, 3, &[(0, 1), (2, 1)]);

        // Reveal the top row
        board.reveal(Position::new(0, 0)).unwrap();
        board.reveal(Position::new(1, 0)).unwrap();
        board.reveal(Position::new(2, 0)).unwrap();

        // Reveal the middle number
        board.reveal(Position::new(1, 1)).unwrap();

        let solver = MatrixSolver;
        let solver_board = SolverBoard::new(&board);
        let result = solver.solve(&solver_board);

        // The middle cell of the bottom row must be safe because:
        // - We know where both mines are (from the 2)
        // - Therefore middle bottom must be safe
        assert!(
            result.safe.contains(&Position::new(1, 2)),
            "Middle bottom cell must be safe given mine positions"
        );
    }

    #[test]
    fn test_boundary_conditions() {
        // Test solver behavior at board boundaries
        let mut board = create_test_board(2, 2, &[(0, 0)]);
        board.reveal(Position::new(1, 1)).unwrap();

        let solver_board = SolverBoard::new(&board);
        let components = MatrixSolver.find_components(&solver_board);

        assert!(
            !components.is_empty(),
            "Should handle boundary cells properly"
        );
        assert!(
            !components[0].constraints.is_empty(),
            "Should include boundary cell constraints"
        );
    }

    // TODO: test_global_constraints
}
