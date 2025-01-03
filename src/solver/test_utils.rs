use super::{
    board::SolverBoard,
    traits::{DeterministicSolver, ProbabilisticResult, ProbabilisticSolver},
    ChainResult, SolverChain,
};
use crate::{Board, Cell, Position};
use rand::prelude::*;
use std::collections::HashSet;

/// Configuration for test board generation
#[derive(Debug, Clone)]
pub struct TestBoardConfig {
    pub width: u32,
    pub height: u32,
    pub mine_density: f64,
    pub revealed_percentage: f64,
}

impl Default for TestBoardConfig {
    fn default() -> Self {
        Self {
            width: 8,
            height: 8,
            mine_density: 0.15,
            revealed_percentage: 0.3,
        }
    }
}

/// Generates test boards with known solutions
pub struct TestBoardGenerator {
    config: TestBoardConfig,
    rng: StdRng,
}

impl TestBoardGenerator {
    pub fn new(config: TestBoardConfig) -> Self {
        Self {
            config,
            rng: StdRng::from_entropy(),
        }
    }

    pub fn with_seed(config: TestBoardConfig, seed: u64) -> Self {
        Self {
            config,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Generates a board with some cells revealed
    pub fn generate(&mut self) -> (Board, HashSet<Position>) {
        let mines_count = (self.config.width as f64
            * self.config.height as f64
            * self.config.mine_density) as u32;
        let mut board = Board::new(self.config.width, self.config.height, mines_count).unwrap();

        // Track which positions contain mines
        let mine_positions: HashSet<Position> = board
            .cells
            .iter()
            .filter_map(|(&pos, cell)| match cell {
                Cell::Hidden(true) => Some(pos),
                _ => None,
            })
            .collect();

        // Reveal some safe cells
        let cells_to_reveal = (self.config.width as f64
            * self.config.height as f64
            * self.config.revealed_percentage) as usize;
        let mut revealed = HashSet::new();

        while revealed.len() < cells_to_reveal {
            let x = self.rng.gen_range(0..self.config.width) as i32;
            let y = self.rng.gen_range(0..self.config.height) as i32;
            let pos = Position::new(x, y);

            if !mine_positions.contains(&pos) && !revealed.contains(&pos) {
                board.reveal(pos).unwrap();
                revealed.insert(pos);
            }
        }

        (board, mine_positions)
    }

    /// Generates multiple test cases
    pub fn generate_batch(&mut self, count: usize) -> Vec<(Board, HashSet<Position>)> {
        (0..count).map(|_| self.generate()).collect()
    }
}

/// Validates deterministic solver correctness
pub fn validate_deterministic_solver<S: DeterministicSolver>(
    solver: &S,
    board: &Board,
    mine_positions: &HashSet<Position>,
) -> bool {
    let solver_board = SolverBoard::new(board);
    let result = solver.solve(&solver_board);

    // Verify that identified mines are actually mines
    for pos in &result.mines {
        if !mine_positions.contains(pos) {
            println!(
                "Solver {} incorrectly identified safe position {:?} as mine",
                solver.name(),
                pos
            );
            return false;
        }
    }

    // Verify that identified safe positions are actually safe
    for pos in &result.safe {
        if mine_positions.contains(pos) {
            println!(
                "Solver {} incorrectly identified mine {:?} as safe",
                solver.name(),
                pos
            );
            return false;
        }
    }

    true
}

/// Validates probabilistic solver calibration
pub fn validate_probabilistic_solver<S: ProbabilisticSolver>(
    solver: &S,
    board: &Board,
    mine_positions: &HashSet<Position>,
) -> bool {
    let solver_board = SolverBoard::new(board);
    match solver.assess(&solver_board) {
        ProbabilisticResult::Certain(det_result) => {
            // Verify deterministic results directly without using DeterministicSolver trait
            for pos in &det_result.mines {
                if !mine_positions.contains(pos) {
                    println!(
                        "Solver {} incorrectly identified safe position {:?} as mine",
                        solver.name(),
                        pos
                    );
                    return false;
                }
            }
            for pos in &det_result.safe {
                if mine_positions.contains(pos) {
                    println!(
                        "Solver {} incorrectly identified mine {:?} as safe",
                        solver.name(),
                        pos
                    );
                    return false;
                }
            }
            true
        }
        ProbabilisticResult::Uncertain(prob_map) => {
            // For uncertain results, verify probability bounds
            prob_map.probabilities.iter().all(|(_, prob)| {
                if !(0.0..=1.0).contains(prob) {
                    println!(
                        "Solver {} produced invalid probability: {}",
                        solver.name(),
                        prob
                    );
                    false
                } else {
                    true
                }
            })
        }
    }
}

/// Validates a solver chain
pub fn validate_solver_chain(
    chain: &SolverChain,
    board: &Board,
    mine_positions: &HashSet<Position>,
) -> bool {
    let solver_board = SolverBoard::new(board);
    match chain.solve(&solver_board) {
        ChainResult::Deterministic(result) => {
            // Verify deterministic results
            for pos in &result.mines {
                if !mine_positions.contains(pos) {
                    println!(
                        "Chain incorrectly identified safe position {:?} as mine",
                        pos
                    );
                    return false;
                }
            }
            for pos in &result.safe {
                if mine_positions.contains(pos) {
                    println!("Chain incorrectly identified mine {:?} as safe", pos);
                    return false;
                }
            }
            true
        }
        ChainResult::Probabilistic { position, .. } => {
            // For probabilistic results, just verify the position is valid
            board.is_within_bounds(position)
        }
        ChainResult::NoMoves => true,
    }
}
