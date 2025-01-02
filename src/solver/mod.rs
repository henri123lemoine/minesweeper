mod board;
mod counting;
mod matrix;
mod tank;
mod traits;

pub use crate::Position;
pub use board::SolverBoard;
pub use counting::CountingSolver;
pub use matrix::MatrixSolver;
pub use tank::TankSolver;
pub use traits::{
    DeterministicResult, DeterministicSolver, ProbabilisticResult, ProbabilisticSolver, Solver,
};

#[doc(hidden)]
pub use crate::solver_test_suite;

/// Represents a chain of solvers that can include both deterministic and probabilistic approaches
#[derive(Default)]
pub struct SolverChain {
    deterministic_solvers: Vec<Box<dyn DeterministicSolver>>,
    probabilistic_solvers: Vec<Box<dyn ProbabilisticSolver>>,
}

impl SolverChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_deterministic<S: DeterministicSolver + 'static>(mut self, solver: S) -> Self {
        self.deterministic_solvers.push(Box::new(solver));
        self
    }

    pub fn add_probabilistic<S: ProbabilisticSolver + 'static>(mut self, solver: S) -> Self {
        self.probabilistic_solvers.push(Box::new(solver));
        self
    }

    /// Attempts to solve the board using all available solvers
    pub fn solve(&self, board: &SolverBoard) -> ChainResult {
        // First try all deterministic solvers
        for solver in &self.deterministic_solvers {
            let result = solver.solve(board);
            if !result.mines.is_empty() || !result.safe.is_empty() {
                return ChainResult::Deterministic(result);
            }
        }

        // If no deterministic solution, try probabilistic solvers
        let mut best_probabilistic: Option<ProbabilisticResult> = None;
        let mut best_confidence = 0.0;

        for solver in &self.probabilistic_solvers {
            let result = solver.solve(board);

            // If solver found any deterministic results, return those immediately
            if !result.deterministic.mines.is_empty() || !result.deterministic.safe.is_empty() {
                return ChainResult::Deterministic(result.deterministic);
            }

            // Track the highest confidence probabilistic result
            if let Some((_, confidence)) = result
                .probabilities
                .iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            {
                if *confidence > best_confidence {
                    best_confidence = *confidence;
                    best_probabilistic = Some(result);
                }
            }
        }

        // Return the best probabilistic result if we found one
        best_probabilistic.map_or(ChainResult::NoSolution, ChainResult::Probabilistic)
    }
}

/// Result from attempting to solve a board using the solver chain
#[derive(Debug, Clone)]
pub enum ChainResult {
    /// A deterministic solution was found
    Deterministic(DeterministicResult),
    /// Only probabilistic solutions were found
    Probabilistic(ProbabilisticResult),
    /// No solution could be found
    NoSolution,
}

/// Factory function to create a standard solver chain with recommended configuration
pub fn create_default_chain() -> SolverChain {
    SolverChain::new()
        .add_deterministic(CountingSolver)
        .add_deterministic(MatrixSolver)
        .add_probabilistic(TankSolver::new(0.95))
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
            ChainResult::Probabilistic(_) => (),
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
        let chain = SolverChain::new()
            .add_deterministic(CountingSolver)
            .add_probabilistic(TankSolver::new(0.99))
            .add_deterministic(MatrixSolver);

        let board = Board::new(8, 8, 10).unwrap();
        let solver_board = SolverBoard::new(&board);

        // Chain should still work even with "incorrect" ordering
        chain.solve(&solver_board); // Just ensure it doesn't panic
    }
}
