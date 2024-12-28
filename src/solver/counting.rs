use super::board::{SolverBoard, SolverCell};
use super::traits::{Certainty, Solver, SolverAction};
use super::SolverResult;
use crate::Position;

/// Implements basic counting rules for minesweeper solving:
/// - If a numbered cell has exactly as many hidden neighbors as its number, they must all be mines
/// - If a numbered cell has exactly as many flagged neighbors as its number, all other neighbors must be safe
/// - Any neighbor of a 0 is safe
pub struct CountingSolver;

impl CountingSolver {
    fn analyze_cell(&self, board: &SolverBoard, pos: Position) -> Vec<SolverAction> {
        let mut actions = Vec::new();

        // Only analyze revealed cells with numbers
        let cell = match board.get(pos) {
            Some(SolverCell::Revealed(n)) => n,
            _ => return actions,
        };

        let neighbors = board.neighbors(pos);
        let mut covered_count = 0;
        let mut flagged_count = 0;
        let mut covered_positions = Vec::new();

        // Count flagged and covered neighbors
        for &pos in &neighbors {
            match board.get(pos) {
                Some(SolverCell::Covered) => {
                    covered_count += 1;
                    covered_positions.push(pos);
                }
                Some(SolverCell::Flagged) => {
                    flagged_count += 1;
                }
                _ => {}
            }
        }

        // If cell=0, all neighbors must be safe
        if cell == 0 {
            for pos in covered_positions {
                actions.push(SolverAction::Reveal(pos));
            }
            return actions;
        }

        // If number of flags equals cell number, all other covered neighbors must be safe
        if cell as usize == flagged_count && covered_count > 0 {
            for pos in covered_positions {
                actions.push(SolverAction::Reveal(pos));
            }
            return actions;
        }

        // If remaining mines equals number of covered cells, they must all be mines
        let remaining_mines = cell as usize - flagged_count;
        if remaining_mines > 0 && remaining_mines == covered_count {
            for pos in covered_positions {
                actions.push(SolverAction::Flag(pos));
            }
            return actions;
        }

        actions
    }
}

impl Solver for CountingSolver {
    fn solve(&self, board: &SolverBoard) -> SolverResult {
        let mut actions = Vec::new();

        for pos in board.iter_positions() {
            let mut cell_actions = self.analyze_cell(board, pos);
            actions.append(&mut cell_actions);
        }

        // All counting solver deductions are deterministic
        SolverResult {
            actions,
            certainty: Certainty::Deterministic,
        }
    }

    fn name(&self) -> &str {
        "Basic Counting Solver"
    }
}
