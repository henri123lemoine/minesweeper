use super::board::SolverBoard;
use crate::Position;
use std::collections::HashSet;

/// Represents positions that a solver has determined to be mines or safe
#[derive(Debug, Clone, Default)]
pub struct DeterministicResult {
    /// Positions that are definitely mines
    pub mines: HashSet<Position>,
    /// Positions that are definitely safe
    pub safe: HashSet<Position>,
}

/// Represents probabilistic analysis of board positions
#[derive(Debug, Clone, Default)]
pub struct ProbabilisticResult {
    /// Positions that are definitely mines or safe (100% certain)
    pub deterministic: DeterministicResult,
    /// For uncertain positions, maps position to probability of being a mine
    pub probabilities: Vec<(Position, f64)>,
}

/// Base trait for all solvers
pub trait Solver {
    fn name(&self) -> &str;
}

/// Trait for solvers that make deterministic decisions
pub trait DeterministicSolver: Solver {
    fn solve(&self, board: &SolverBoard) -> DeterministicResult;
}

/// Trait for solvers that make probabilistic decisions
pub trait ProbabilisticSolver: Solver {
    fn solve(&self, board: &SolverBoard) -> ProbabilisticResult;

    /// Returns the solver's confidence threshold for making deterministic decisions
    fn confidence_threshold(&self) -> f64;
}

#[doc(hidden)]
#[macro_export]
macro_rules! solver_test_suite {
    ($solver:ty, deterministic) => {
        mod solver_tests {
            use super::*;
            use crate::{Board, Cell, Position};

            #[test]
            fn test_deterministic_consistency() {
                let solver = <$solver>::default();
                let board = Board::new(3, 3, 1).unwrap();
                let solver_board = SolverBoard::new(&board);
                let result = solver.solve(&solver_board);

                // No overlap between mines and safe positions
                assert!(result.mines.is_disjoint(&result.safe));
            }

            #[test]
            fn test_deterministic_correctness() {
                let mut board = Board::new(3, 3, 1).unwrap();
                let solver = <$solver>::default();

                // Place a mine in a known position
                board.cells.insert(Position::new(0, 0), Cell::Hidden(true));
                board.reveal(Position::new(1, 1)).unwrap();

                let solver_board = SolverBoard::new(&board);
                let result = solver.solve(&solver_board);

                // Solver should not mark safe positions as mines
                for pos in &result.safe {
                    if let Ok(Cell::Hidden(is_mine)) = board.get_cell(*pos) {
                        assert!(!is_mine, "Solver incorrectly marked mine as safe");
                    }
                }
            }
        }
    };

    ($solver:ty, probabilistic) => {
        mod solver_tests {
            use super::*;
            use crate::{Board, Cell, Position};
            use std::collections::HashMap;

            #[test]
            fn test_probabilistic_calibration() {
                let solver = <$solver>::default();
                let mut total_predictions = HashMap::new();
                let mut correct_predictions = HashMap::new();

                // Run multiple games to collect statistics
                for _ in 0..10000 {
                    let board = Board::new(8, 8, 10).unwrap();
                    let solver_board = SolverBoard::new(&board);
                    let result = solver.solve(&solver_board);

                    // Bin predictions by confidence level
                    for (pos, prob) in result.probabilities {
                        let bin = (prob * 10.0).floor() as i32;
                        if let Ok(Cell::Hidden(is_mine)) = board.get_cell(pos) {
                            *total_predictions.entry(bin).or_insert(0) += 1;
                            if *is_mine == (prob >= 0.5) {
                                *correct_predictions.entry(bin).or_insert(0) += 1;
                            }
                        }
                    }
                }

                // Check calibration
                for bin in 0..10 {
                    if let Some(&total) = total_predictions.get(&bin) {
                        if total > 0 {
                            let correct = *correct_predictions.get(&bin).unwrap_or(&0);
                            let accuracy = correct as f64 / total as f64;
                            let expected = (bin as f64 + 0.5) / 10.0;
                            assert!(
                                (accuracy - expected).abs() < 0.2,
                                "Poor calibration for confidence bin {}",
                                bin
                            );
                        }
                    }
                }
            }

            #[test]
            fn test_deterministic_subset() {
                let solver = <$solver>::default();
                let board = Board::new(3, 3, 1).unwrap();
                let solver_board = SolverBoard::new(&board);
                let result = solver.solve(&solver_board);

                assert!(
                    result
                        .deterministic
                        .mines
                        .is_disjoint(&result.deterministic.safe),
                    "Deterministic results contain contradictions"
                );
            }
        }
    };
}
