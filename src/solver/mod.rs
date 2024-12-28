mod board;
mod counting;
mod matrix;
mod probabilistic;
mod traits;

pub use board::SolverBoard;
pub use counting::CountingSolver;
pub use matrix::MatrixSolver;
pub use probabilistic::ProbabilisticSolver;
pub use traits::{Certainty, Solver, SolverAction, SolverResult};

// Factory method for creating a full solver chain
pub fn create_full_solver() -> impl Solver {
    ChainSolver::new(vec![
        Box::new(CountingSolver),
        Box::new(MatrixSolver),
        Box::new(ProbabilisticSolver {
            min_confidence: 0.95,
        }),
    ])
}

// Common solver chain implementation
pub struct ChainSolver {
    solvers: Vec<Box<dyn Solver>>,
}

impl ChainSolver {
    pub fn new(solvers: Vec<Box<dyn Solver>>) -> Self {
        Self { solvers }
    }
}

impl Solver for ChainSolver {
    fn solve(&self, board: &SolverBoard) -> SolverResult {
        for solver in &self.solvers {
            let result = solver.solve(board);
            if !result.actions.is_empty() {
                return result;
            }
        }
        SolverResult {
            actions: vec![],
            certainty: Certainty::Deterministic,
        }
    }

    fn name(&self) -> &str {
        "Chain Solver"
    }
}
