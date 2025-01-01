mod board;
mod counting;
mod matrix;
mod probabilistic;
mod tank;
mod traits;

pub use board::SolverBoard;
pub use counting::CountingSolver;
pub use matrix::MatrixSolver;
pub use probabilistic::ProbabilisticSolver;
pub use tank::TankSolver;
pub use traits::{Certainty, Solver, SolverAction, SolverResult};

// Factory method for creating a full solver chain
pub fn create_full_solver() -> Result<impl Solver, &'static str> {
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
    pub fn new(solvers: Vec<Box<dyn Solver>>) -> Result<Self, &'static str> {
        // Validate solver ordering: deterministic solvers must come before probabilistic
        let mut seen_probabilistic = false;
        for solver in &solvers {
            if solver.is_deterministic() {
                if seen_probabilistic {
                    return Err("Invalid solver chain: deterministic solver found after probabilistic solver");
                }
            } else {
                seen_probabilistic = true;
            }
        }
        Ok(Self { solvers })
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

    fn is_deterministic(&self) -> bool {
        self.solvers.iter().all(|s| s.is_deterministic())
    }
}
