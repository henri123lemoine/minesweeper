pub mod board;
pub mod error;
pub mod game;
pub mod position;
pub mod solver;

pub use board::{Board, BoardIterator, Cell, RevealResult};
pub use error::GameError;
pub use game::{Action, Game, GameState};
pub use position::Position;
pub use solver::{
    CountingSolver, DeterministicResult, DeterministicSolver, MatrixSolver, ProbabilisticResult,
    ProbabilisticSolver, Solver, SolverBoard, SolverChain, TankSolver,
};
