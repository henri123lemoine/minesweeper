use super::board::SolverBoard;
use crate::Position;

#[derive(Debug, Clone, PartialEq)]
pub enum SolverAction {
    Reveal(Position),
    Flag(Position),
}

#[derive(Debug, Clone)]
pub struct SolverResult {
    pub actions: Vec<SolverAction>,
    pub certainty: Certainty,
}

#[derive(Debug, Clone)]
pub enum Certainty {
    Deterministic,
    Probabilistic(f64), // Confidence level
}

pub trait Solver {
    fn solve(&self, board: &SolverBoard) -> SolverResult;
    fn name(&self) -> &str;
}
