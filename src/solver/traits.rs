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
            use statrs::distribution::{ContinuousCDF, Normal};
            use std::collections::HashMap;

            #[test]
            fn test_probabilistic_calibration() {
                let solver = <$solver>::default();

                // Store (predicted_probability, actual_outcome) pairs
                let mut predictions: Vec<(f64, bool)> = Vec::new();

                // Collect predictions across multiple games
                for _ in 0..10000 {
                    let board = Board::new(8, 8, 10).unwrap();
                    let solver_board = SolverBoard::new(&board);
                    let result = solver.solve(&solver_board);

                    for (pos, predicted_prob) in result.probabilities {
                        if let Ok(Cell::Hidden(is_mine)) = board.get_cell(pos) {
                            predictions.push((predicted_prob, *is_mine));
                        }
                    }
                }

                let n = predictions.len() as f64;

                // Calculate calibration error using Brier score decomposition
                let reliability = predictions
                    .iter()
                    .map(|(pred, actual)| {
                        let actual = if *actual { 1.0 } else { 0.0 };
                        (pred - actual).powi(2)
                    })
                    .sum::<f64>()
                    / n;

                let (mean_predicted, mean_actual) = predictions.iter().fold(
                    (0.0, 0.0),
                    |(sum_pred, sum_actual), (pred, actual)| {
                        (
                            sum_pred + pred,
                            sum_actual + if *actual { 1.0 } else { 0.0 },
                        )
                    },
                );
                let (mean_predicted, mean_actual) = (mean_predicted / n, mean_actual / n);

                // Calculate standard error under null hypothesis
                let var_estimate = predictions
                    .iter()
                    .map(|(p, a)| {
                        let a = if *a { 1.0 } else { 0.0 };
                        let e = (p - a).powi(2) - reliability;
                        e.powi(2)
                    })
                    .sum::<f64>()
                    / (n.powi(2));

                let standard_error = var_estimate.sqrt();
                let z_score = reliability / standard_error;

                // Use statrs for p-value calculation
                let normal = Normal::new(0.0, 1.0).unwrap();
                let p_value = 2.0 * (1.0 - normal.cdf(z_score.abs()));

                const ALPHA: f64 = 0.001; // Significance level

                assert!(
                    p_value >= ALPHA,
                    "Calibration test failed: p-value = {:.6}, reliability = {:.6}, \
                     mean_predicted = {:.3}, mean_actual = {:.3}, n = {}",
                    p_value,
                    reliability,
                    mean_predicted,
                    mean_actual,
                    n as u32
                );
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
