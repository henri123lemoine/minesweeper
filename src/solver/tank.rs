use super::board::{SolverBoard, SolverCell};
use super::solver_test_suite;
use super::traits::{DeterministicResult, ProbabilisticResult, ProbabilisticSolver, Solver};
use crate::Position;
use std::collections::{HashMap, HashSet, VecDeque};

/// A constraint in Tank's algorithm, representing the sum of mines in a set of positions
#[derive(Debug, Clone)]
struct MineConstraint {
    positions: HashSet<Position>,
    mines_required: u8,
}

/// Tank Solver implementation based on Simon Tatham's algorithm
/// https://www.chiark.greenend.org.uk/~sgtatham/mines/#solve
#[derive(Debug)]
pub struct TankSolver {
    confidence_threshold: f64,
}

impl Default for TankSolver {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.95,
        }
    }
}

impl TankSolver {
    pub fn new(confidence_threshold: f64) -> Self {
        Self {
            confidence_threshold,
        }
    }

    /// Collects all constraints from the board state
    fn get_constraints(&self, board: &SolverBoard) -> Vec<MineConstraint> {
        let mut constraints = Vec::new();
        let mut seen_positions = HashSet::new();

        // Collect constraints from revealed numbers
        for pos in board.iter_positions() {
            if let Some(SolverCell::Revealed(num)) = board.get(pos) {
                let mut positions = HashSet::new();
                let mut flagged_count = 0;

                // Check all neighbors
                for npos in board.neighbors(pos) {
                    match board.get(npos) {
                        Some(SolverCell::Covered) => {
                            positions.insert(npos);
                            seen_positions.insert(npos);
                        }
                        Some(SolverCell::Flagged) => flagged_count += 1,
                        _ => {}
                    }
                }

                if !positions.is_empty() {
                    constraints.push(MineConstraint {
                        positions,
                        mines_required: num - flagged_count,
                    });
                }
            }
        }

        // Add global mine count constraint
        let remaining_positions: HashSet<_> = board
            .iter_positions()
            .filter(|&pos| {
                matches!(board.get(pos), Some(SolverCell::Covered))
                    && !seen_positions.contains(&pos)
            })
            .collect();

        if !remaining_positions.is_empty() {
            let remaining_mines = board.total_mines() - board.mines_marked();
            constraints.push(MineConstraint {
                positions: remaining_positions,
                mines_required: remaining_mines as u8,
            });
        }

        constraints
    }

    /// Performs probability flooding to calculate mine probabilities
    fn flood_probabilities(&self, constraints: &[MineConstraint]) -> HashMap<Position, f64> {
        let mut probabilities = HashMap::new();
        let mut queue = VecDeque::new();
        let mut processed = HashSet::new();

        // Initialize probabilities from single-position constraints
        for constraint in constraints {
            if constraint.positions.len() == 1 {
                let pos = *constraint.positions.iter().next().unwrap();
                let prob = constraint.mines_required as f64;
                probabilities.insert(pos, prob);
                queue.push_back(pos);
                processed.insert(pos);
            } else {
                // Initialize multi-position constraints with uniform probability
                let prob = constraint.mines_required as f64 / constraint.positions.len() as f64;
                for &pos in &constraint.positions {
                    if !processed.contains(&pos) {
                        probabilities.insert(pos, prob);
                        queue.push_back(pos);
                        processed.insert(pos);
                    }
                }
            }
        }

        // Process probability queue
        while let Some(pos) = queue.pop_front() {
            let prob = *probabilities.get(&pos).unwrap();

            // Update probabilities for all constraints containing this position
            for constraint in constraints {
                if !constraint.positions.contains(&pos) {
                    continue;
                }

                let other_positions: Vec<_> = constraint
                    .positions
                    .iter()
                    .filter(|&&p| p != pos && !processed.contains(&p))
                    .collect();

                if other_positions.is_empty() {
                    continue;
                }

                // Calculate new probabilities for other positions
                let remaining_mines = (constraint.mines_required as f64 - prob).max(0.0);
                let new_prob = remaining_mines / other_positions.len() as f64;

                // Update probabilities and queue
                for &new_pos in &other_positions {
                    match probabilities.get(new_pos) {
                        Some(&existing) => {
                            if (new_prob - existing).abs() > f64::EPSILON {
                                probabilities.insert(*new_pos, (new_prob + existing) / 2.0);
                                if !queue.contains(new_pos) {
                                    queue.push_back(*new_pos);
                                }
                            }
                        }
                        None => {
                            probabilities.insert(*new_pos, new_prob);
                            queue.push_back(*new_pos);
                        }
                    }
                    processed.insert(*new_pos);
                }
            }
        }

        probabilities
    }

    /// Separates definite results from probabilistic ones
    fn analyze_probabilities(&self, probabilities: HashMap<Position, f64>) -> ProbabilisticResult {
        let mut result = ProbabilisticResult::default();
        let epsilon = 1e-10;

        for (pos, prob) in probabilities {
            if (prob - 0.0).abs() < epsilon {
                result.deterministic.safe.insert(pos);
            } else if (prob - 1.0).abs() < epsilon {
                result.deterministic.mines.insert(pos);
            } else {
                result.probabilities.push((pos, prob));
            }
        }

        result.probabilities.sort_unstable_by(|a, b| {
            let a_certainty = if a.1 < 0.5 { 1.0 - a.1 } else { a.1 };
            let b_certainty = if b.1 < 0.5 { 1.0 - b.1 } else { b.1 };
            b_certainty.partial_cmp(&a_certainty).unwrap()
        });

        result
    }
}

impl Solver for TankSolver {
    fn name(&self) -> &str {
        "Tank Solver"
    }
}

impl ProbabilisticSolver for TankSolver {
    fn solve(&self, board: &SolverBoard) -> ProbabilisticResult {
        // Special case: opening move
        if board.is_start() {
            let (width, height) = board.dimensions();
            let center_pos = Position::new((width / 2) as i32, (height / 2) as i32);

            return ProbabilisticResult {
                deterministic: DeterministicResult::default(),
                probabilities: vec![(center_pos, board.mine_density())],
            };
        }

        // Get constraints and calculate probabilities
        let constraints = self.get_constraints(board);
        if constraints.is_empty() {
            return ProbabilisticResult::default();
        }

        let probabilities = self.flood_probabilities(&constraints);
        self.analyze_probabilities(probabilities)
    }

    fn confidence_threshold(&self) -> f64 {
        self.confidence_threshold
    }
}

solver_test_suite!(TankSolver, probabilistic);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Board, Cell};

    #[test]
    fn test_opening_move() {
        let board = Board::new(8, 8, 10).unwrap();
        let solver = TankSolver::default();
        let solver_board = SolverBoard::new(&board);

        let result = solver.solve(&solver_board);
        assert!(result.deterministic.mines.is_empty());
        assert!(result.deterministic.safe.is_empty());
        assert_eq!(result.probabilities.len(), 1);

        // Should suggest center as opening move
        let (pos, _) = result.probabilities[0];
        assert_eq!(pos, Position::new(4, 4));
    }

    #[test]
    fn test_constraint_generation() {
        let mut board = Board::new(3, 3, 1).unwrap();
        board.reveal(Position::new(1, 1)).unwrap();

        let solver = TankSolver::default();
        let solver_board = SolverBoard::new(&board);
        let constraints = solver.get_constraints(&solver_board);

        assert!(
            !constraints.is_empty(),
            "Should generate at least one constraint"
        );
    }

    #[test]
    fn test_definite_results() {
        let mut board = Board::new(3, 3, 1).unwrap();

        // Create a known board state where one position must be a mine
        board.reveal(Position::new(1, 1)).unwrap();
        if let Ok(Cell::Revealed(1)) = board.get_cell(Position::new(1, 1)) {
            let solver = TankSolver::default();
            let solver_board = SolverBoard::new(&board);
            let result = solver.solve(&solver_board);

            assert!(
                !result.deterministic.mines.is_empty() || !result.deterministic.safe.is_empty(),
                "Should find at least one definite result"
            );
        }
    }

    #[test]
    fn test_probability_flooding() {
        let mut board = Board::new(3, 3, 1).unwrap();
        board.reveal(Position::new(1, 1)).unwrap();

        let solver = TankSolver::default();
        let solver_board = SolverBoard::new(&board);
        let constraints = solver.get_constraints(&solver_board);
        let probabilities = solver.flood_probabilities(&constraints);

        // Verify probability bounds
        for &prob in probabilities.values() {
            assert!(
                (0.0..=1.0).contains(&prob),
                "Probabilities must be between 0 and 1"
            );
        }
    }
}
