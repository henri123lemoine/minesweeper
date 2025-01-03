mod board;
mod counting;
mod matrix;
pub mod solver_validation;
mod tank;
pub mod test_utils;
mod traits;

pub use crate::Position;
pub use board::SolverBoard;
pub use counting::CountingSolver;
pub use matrix::MatrixSolver;
pub use tank::TankSolver;
pub use traits::{
    DeterministicResult, DeterministicSolver, ProbabilisticResult, ProbabilisticSolver,
    ProbabilityMap, Solver,
};

use std::collections::HashMap;

#[doc(hidden)]
pub use crate::solver_test_suite;

pub struct SolverChain {
    deterministic_solvers: Vec<Box<dyn DeterministicSolver>>,
    probabilistic_solvers: Vec<Box<dyn ProbabilisticSolver>>,
    confidence_threshold: f64,
}

impl SolverChain {
    pub fn new(confidence_threshold: f64) -> Self {
        Self {
            deterministic_solvers: Vec::new(),
            probabilistic_solvers: Vec::new(),
            confidence_threshold,
        }
    }

    pub fn add_deterministic<S: DeterministicSolver + 'static>(mut self, solver: S) -> Self {
        self.deterministic_solvers.push(Box::new(solver));
        self
    }

    pub fn add_probabilistic<S: ProbabilisticSolver + 'static>(mut self, solver: S) -> Self {
        self.probabilistic_solvers.push(Box::new(solver));
        self
    }

    pub fn solve(&self, board: &SolverBoard) -> ChainResult {
        // First try deterministic solvers
        for solver in &self.deterministic_solvers {
            let result = solver.solve(board);
            if !result.mines.is_empty() || !result.safe.is_empty() {
                return ChainResult::Deterministic(result);
            }
        }

        // Try probabilistic solvers
        let mut probability_maps = Vec::new();

        for solver in &self.probabilistic_solvers {
            match solver.assess(board) {
                // If any solver finds certainty, use it immediately
                ProbabilisticResult::Certain(result) => {
                    return ChainResult::Deterministic(result);
                }
                // Collect uncertain results for later combination
                ProbabilisticResult::Uncertain(probs) => {
                    probability_maps.push(probs);
                }
            }
        }

        // If we got here, we only have uncertain results
        if !probability_maps.is_empty() {
            let combined = self.combine_probability_maps(probability_maps);

            // Find the most confident move  TODO: Verify this is the best way to choose a move
            if let Some((pos, prob)) = combined.probabilities.iter().max_by(|(_, a), (_, b)| {
                let a_conf = (a - 0.5).abs();
                let b_conf = (b - 0.5).abs();
                a_conf.partial_cmp(&b_conf).unwrap()
            }) {
                let confidence = (prob - 0.5).abs() * 2.0; // Scale to 0-1
                if confidence >= self.confidence_threshold {
                    return ChainResult::Probabilistic {
                        position: *pos,
                        confidence,
                    };
                }
            }
        }

        ChainResult::NoMoves
    }

    fn combine_probability_maps(&self, maps: Vec<ProbabilityMap>) -> ProbabilityMap {
        let mut position_probs: HashMap<Position, Vec<f64>> = HashMap::new();

        // Collect all probabilities for each position
        for map in maps {
            for (pos, prob) in map.probabilities {
                position_probs.entry(pos).or_default().push(prob);
            }
        }

        // Average probabilities
        let probabilities = position_probs
            .into_iter()
            .map(|(pos, probs)| {
                let avg = probs.iter().sum::<f64>() / probs.len() as f64;
                (pos, avg)
            })
            .collect();

        ProbabilityMap { probabilities }
    }
}

#[derive(Debug, Clone)]
pub enum ChainResult {
    Deterministic(DeterministicResult),
    Probabilistic { position: Position, confidence: f64 },
    NoMoves,
}

/// Factory function to create a standard solver chain with recommended configuration
pub fn create_default_chain() -> SolverChain {
    SolverChain::new(0.95)
        .add_deterministic(CountingSolver)
        .add_deterministic(MatrixSolver)
        .add_probabilistic(TankSolver)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Board;

    #[test]
    fn test_solver_chain_ordering() {
        let chain = create_default_chain();
        let board = Board::new(8, 8, 10).unwrap();
        let solver_board = SolverBoard::new(&board);

        // Initial move should come from a probabilistic solver
        match chain.solve(&solver_board) {
            ChainResult::Probabilistic { .. } => {}
            _ => panic!("Expected probabilistic solution for initial move"),
        }
    }

    #[test]
    fn test_deterministic_priority() {
        let mut board = Board::new(3, 3, 1).unwrap();

        // Create a simple board state where deterministic solution exists
        board.reveal(Position::new(1, 1)).unwrap();

        let chain = create_default_chain();
        let solver_board = SolverBoard::new(&board);

        match chain.solve(&solver_board) {
            ChainResult::Deterministic(result) => {
                assert!(
                    !result.mines.is_empty() || !result.safe.is_empty(),
                    "Deterministic solver should find solution"
                );
            }
            _ => panic!("Expected deterministic solution"),
        }
    }

    #[test]
    fn test_chain_composition() {
        let chain = SolverChain::new(0.95)
            .add_deterministic(CountingSolver)
            .add_probabilistic(TankSolver)
            .add_deterministic(MatrixSolver);

        let board = Board::new(8, 8, 10).unwrap();
        let solver_board = SolverBoard::new(&board);

        // Chain should still work even with "incorrect" ordering
        chain.solve(&solver_board); // Just ensure it doesn't panic
    }
}
