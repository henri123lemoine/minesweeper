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

                // Calculate detailed calibration statistics
                let pred_ranges = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
                let mut range_stats = vec![];

                predictions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                for i in 0..pred_ranges.len() - 1 {
                    let range_preds: Vec<_> = predictions
                        .iter()
                        .filter(|(p, _)| *p >= pred_ranges[i] && *p < pred_ranges[i + 1])
                        .collect();

                    if !range_preds.is_empty() {
                        let n_range = range_preds.len();
                        let mean_pred =
                            range_preds.iter().map(|(p, _)| p).sum::<f64>() / n_range as f64;
                        let actual_freq =
                            range_preds.iter().filter(|(_, a)| *a).count() as f64 / n_range as f64;
                        range_stats.push((
                            pred_ranges[i]..pred_ranges[i + 1],
                            mean_pred,
                            actual_freq,
                            n_range,
                        ));
                    }
                }

                const MAX_RELIABILITY: f64 = 0.15;

                let error_msg = format!(
                    "\nCalibration Analysis:\n\
                     Overall Statistics:\n\
                     - Reliability Score: {:.6} (lower is better)\n\
                     - Mean Predicted Probability: {:.3}\n\
                     - Actual Mine Frequency: {:.3}\n\
                     - Number of Predictions: {}\n\n\
                     Detailed Breakdown by Prediction Range:\n",
                    reliability, mean_predicted, mean_actual, n as u32
                );

                let mut full_msg = String::from(error_msg);
                for (range, mean_pred, actual_freq, count) in range_stats {
                    full_msg.push_str(&format!(
                        "Range {:.1}-{:.1}:\n\
                         - Count: {}\n\
                         - Mean Predicted Prob: {:.3}\n\
                         - Actual Frequency: {:.3}\n\
                         - Calibration Error: {:.3}\n\n",
                        range.start,
                        range.end,
                        count,
                        mean_pred,
                        actual_freq,
                        (mean_pred - actual_freq).abs()
                    ));
                }

                full_msg.push_str(&format!(
                    "Visual Calibration Error:\n\
                     Mean Predicted ({:.3}) vs Actual ({:.3}): {}{}|\n\
                     Difference: {:.3} ({})",
                    mean_predicted,
                    mean_actual,
                    "=".repeat((mean_predicted * 50.0) as usize),
                    if mean_predicted <= mean_actual {
                        "<"
                    } else {
                        ">"
                    },
                    (mean_predicted - mean_actual).abs(),
                    if (mean_predicted - mean_actual).abs() < 0.05 {
                        "GOOD"
                    } else {
                        "CONCERNING"
                    }
                ));

                // Always print diagnostics
                println!("{}", full_msg);

                assert!(
                    reliability < MAX_RELIABILITY,
                    "Calibration failed: reliability {:.6} > threshold {}",
                    reliability,
                    MAX_RELIABILITY
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
