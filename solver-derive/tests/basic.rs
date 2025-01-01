use minesweeper_solver_derive::SolverTest;

// Mock types to simulate the minesweeper crate
pub struct Board;
pub struct SolverBoard;
pub struct Position;
pub enum Cell {}
pub enum SolverAction {}
pub enum Certainty {}

pub struct SolverResult {
    pub actions: Vec<SolverAction>,
    pub certainty: Certainty,
}

pub trait Solver {
    fn solve(&self, board: &SolverBoard) -> SolverResult;
    fn name(&self) -> &str;
    fn is_deterministic(&self) -> bool;
}

// Test a basic solver implementation
#[derive(SolverTest)]
struct BasicSolver;

impl Solver for BasicSolver {
    fn solve(&self, _board: &SolverBoard) -> SolverResult {
        SolverResult {
            actions: Vec::new(),
            certainty: Certainty::Deterministic,
        }
    }

    fn name(&self) -> &str {
        "Basic Solver"
    }

    fn is_deterministic(&self) -> bool {
        true
    }
}

// This test ensures the derive macro compiles successfully
#[test]
fn test_derive_compiles() {
    let _solver = BasicSolver;
}
