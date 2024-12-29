use crate::solver::{Certainty, Solver};
use crate::Position;
use std::collections::{HashMap, HashSet};

use super::{board::SolverCell, SolverAction, SolverBoard, SolverResult};

pub struct ProbabilisticSolver {
    pub min_confidence: f64,
}

#[derive(Debug)]
struct LocalConstraint {
    cells: HashSet<Position>,
    mine_count: u8,
}

impl ProbabilisticSolver {
    fn get_constraints(&self, board: &SolverBoard) -> Vec<LocalConstraint> {
        let mut constraints = Vec::new();

        for pos in board.iter_positions() {
            if let Some(SolverCell::Revealed(n)) = board.get(pos) {
                let mut cells = HashSet::new();
                let mut mine_count = n;

                // Collect covered neighbors and adjust mine count for flags
                for npos in board.neighbors(pos) {
                    match board.get(npos) {
                        Some(SolverCell::Covered) => {
                            cells.insert(npos);
                        }
                        Some(SolverCell::Flagged) => {
                            mine_count -= 1;
                        }
                        _ => {}
                    }
                }

                // Only add constraints that still have some unknown cells
                if !cells.is_empty() {
                    constraints.push(LocalConstraint { cells, mine_count });
                }
            }
        }

        constraints
    }

    fn calculate_probabilities(
        &self,
        board: &SolverBoard,
        constraints: &[LocalConstraint],
    ) -> HashMap<Position, f64> {
        let mut probabilities = HashMap::new();
        let mut covered_cells = HashSet::new();

        // First collect all covered cells
        for pos in board.iter_positions() {
            if let Some(SolverCell::Covered) = board.get(pos) {
                covered_cells.insert(pos);
            }
        }

        // Calculate local probabilities for cells in constraints
        for constraint in constraints {
            // Basic probability for this constraint
            let prob = constraint.mine_count as f64 / constraint.cells.len() as f64;

            // Update probabilities for cells in this constraint
            for &cell in &constraint.cells {
                let entry = probabilities.entry(cell).or_insert(0.0);
                // Average with existing probability if we've seen this cell before
                *entry = if *entry == 0.0 {
                    prob
                } else {
                    (*entry + prob) / 2.0
                };

                covered_cells.remove(&cell);
            }
        }

        // For remaining cells not in any constraint, use global probability
        if !covered_cells.is_empty() {
            let remaining_mines = board.total_mines() - board.mines_marked();
            // Subtract mines we've accounted for in local constraints
            let accounted_mines: f64 = probabilities.values().sum();
            let remaining_global_mines = (remaining_mines as f64 - accounted_mines).max(0.0);
            let global_prob = remaining_global_mines / covered_cells.len() as f64;

            for cell in covered_cells {
                probabilities.insert(cell, global_prob);
            }
        }

        probabilities
    }

    fn find_best_move(&self, probabilities: &HashMap<Position, f64>) -> Option<SolverAction> {
        // Find the cell with lowest probability of being a mine
        let mut best_move = None;
        let mut lowest_prob = 1.0;

        for (&pos, &prob) in probabilities {
            if prob < lowest_prob {
                lowest_prob = prob;
                best_move = Some((pos, prob));
            }
        }

        // If we found a move and it's safe enough...
        best_move.and_then(|(pos, prob)| {
            if prob < 1.0 - self.min_confidence {
                Some(SolverAction::Reveal(pos))
            } else if prob > self.min_confidence {
                Some(SolverAction::Flag(pos))
            } else {
                None
            }
        })
    }
}

impl Solver for ProbabilisticSolver {
    fn solve(&self, board: &SolverBoard) -> SolverResult {
        // Get local constraints from the board
        let constraints = self.get_constraints(board);

        // Calculate probabilities for all unknown cells
        let probabilities = self.calculate_probabilities(board, &constraints);

        // Find best move based on probabilities
        if let Some(action) = self.find_best_move(&probabilities) {
            // Return single best action with its certainty
            let certainty = match action {
                SolverAction::Reveal(pos) => 1.0 - probabilities[&pos],
                SolverAction::Flag(pos) => probabilities[&pos],
            };

            SolverResult {
                actions: vec![action],
                certainty: Certainty::Probabilistic(certainty),
            }
        } else {
            // No move meets our confidence threshold
            SolverResult {
                actions: vec![],
                certainty: Certainty::Probabilistic(0.0),
            }
        }
    }

    fn name(&self) -> &str {
        "Probabilistic Solver"
    }
}
