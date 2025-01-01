use super::{
    board::{SolverBoard, SolverCell},
    traits::{Certainty, Solver, SolverAction, SolverResult},
};
use crate::Position;
use minesweeper_solver_derive::SolverTest;
use std::collections::{HashMap, HashSet, VecDeque};

/// Tank Solver implementation based on Simon Tatham's algorithm
/// https://www.chiark.greenend.org.uk/~sgtatham/mines/#solve
#[derive(SolverTest)]
pub struct TankSolver {
    min_confidence: f64,
}

impl Default for TankSolver {
    fn default() -> Self {
        Self {
            min_confidence: 0.95,
        }
    }
}

#[derive(Debug, Clone)]
struct MineConstraint {
    positions: HashSet<Position>,
    mines_required: u8,
}

impl TankSolver {
    pub fn new(min_confidence: f64) -> Self {
        Self { min_confidence }
    }

    fn get_constraints(&self, board: &SolverBoard) -> Vec<MineConstraint> {
        let mut constraints = Vec::new();
        let mut seen_positions = HashSet::new();

        // Collect constraints from revealed numbers
        for pos in board.iter_positions() {
            if let Some(SolverCell::Revealed(num)) = board.get(pos) {
                let mut positions = HashSet::new();
                let mut flagged_count = 0;

                // Check all neighbors
                for npos in board.neighbors(pos) {
                    match board.get(npos) {
                        Some(SolverCell::Covered) => {
                            positions.insert(npos);
                            seen_positions.insert(npos);
                        }
                        Some(SolverCell::Flagged) => flagged_count += 1,
                        _ => {}
                    }
                }

                if !positions.is_empty() {
                    constraints.push(MineConstraint {
                        positions,
                        mines_required: num - flagged_count,
                    });
                }
            }
        }

        // Add global mine count constraint
        let remaining_positions: HashSet<_> = board
            .iter_positions()
            .filter(|&pos| {
                matches!(board.get(pos), Some(SolverCell::Covered))
                    && !seen_positions.contains(&pos)
            })
            .collect();

        if !remaining_positions.is_empty() {
            let remaining_mines = board.total_mines() - board.mines_marked();
            constraints.push(MineConstraint {
                positions: remaining_positions,
                mines_required: remaining_mines as u8,
            });
        }

        constraints
    }

    fn flood_probability(&self, constraints: &[MineConstraint]) -> HashMap<Position, f64> {
        let mut probabilities = HashMap::new();
        let mut queue = VecDeque::new();
        let mut processed = HashSet::new();

        // Initialize with single-position constraints
        for constraint in constraints {
            if constraint.positions.len() == 1 {
                let pos = *constraint.positions.iter().next().unwrap();
                let prob = constraint.mines_required as f64;
                probabilities.insert(pos, prob);
                queue.push_back(pos);
                processed.insert(pos);
            } else {
                // For multi-position constraints, initialize with uniform probability
                let prob = constraint.mines_required as f64 / constraint.positions.len() as f64;
                for &pos in &constraint.positions {
                    if !processed.contains(&pos) {
                        probabilities.insert(pos, prob);
                        queue.push_back(pos);
                        processed.insert(pos);
                    }
                }
            }
        }

        // If we have no probabilities yet but do have constraints, initialize them
        if probabilities.is_empty() && !constraints.is_empty() {
            let first_constraint = &constraints[0];
            let prob =
                first_constraint.mines_required as f64 / first_constraint.positions.len() as f64;
            for &pos in &first_constraint.positions {
                probabilities.insert(pos, prob);
                queue.push_back(pos);
                processed.insert(pos);
            }
        }

        while let Some(pos) = queue.pop_front() {
            let prob = *probabilities.get(&pos).unwrap();

            // Find all constraints containing this position
            for constraint in constraints {
                if !constraint.positions.contains(&pos) {
                    continue;
                }

                // Calculate new probabilities for other positions in constraint
                let other_positions: Vec<_> = constraint
                    .positions
                    .iter()
                    .filter(|&&p| p != pos && !processed.contains(&p))
                    .collect();

                if other_positions.is_empty() {
                    continue;
                }

                let remaining_mines = (constraint.mines_required as f64 - prob).max(0.0);
                let new_prob = remaining_mines / other_positions.len() as f64;

                // Update probabilities and queue
                for &new_pos in &other_positions {
                    match probabilities.get(&new_pos) {
                        Some(&existing) => {
                            if (new_prob - existing).abs() > f64::EPSILON {
                                probabilities.insert(*new_pos, (new_prob + existing) / 2.0);
                                if !queue.contains(&new_pos) {
                                    queue.push_back(*new_pos);
                                }
                            }
                        }
                        None => {
                            probabilities.insert(*new_pos, new_prob);
                            queue.push_back(*new_pos);
                        }
                    }
                    processed.insert(*new_pos);
                }
            }
        }

        probabilities
    }
}

impl Solver for TankSolver {
    fn solve(&self, board: &SolverBoard) -> SolverResult {
        // Handle opening move
        if board
            .iter_positions()
            .all(|pos| !matches!(board.get(pos), Some(SolverCell::Revealed(_))))
        {
            let (width, height) = board.dimensions();
            return SolverResult {
                actions: vec![SolverAction::Reveal(Position::new(
                    (width / 2) as i32,
                    (height / 2) as i32,
                ))],
                certainty: Certainty::Probabilistic(0.9),
            };
        }

        // Get constraints and calculate probabilities
        let constraints = self.get_constraints(board);

        // If we have no constraints but uncovered squares, pick one at random
        if constraints.is_empty() {
            if let Some(pos) = board
                .iter_positions()
                .find(|&pos| matches!(board.get(pos), Some(SolverCell::Covered)))
            {
                return SolverResult {
                    actions: vec![SolverAction::Reveal(pos)],
                    certainty: Certainty::Probabilistic(0.5),
                };
            }
        }

        let probabilities = self.flood_probability(&constraints);

        // Find best move
        let mut best_action = None;
        let mut best_certainty = 0.0;

        for (&pos, &prob) in &probabilities {
            let certainty = if prob < 0.5 { 1.0 - prob } else { prob };
            if certainty > best_certainty {
                best_certainty = certainty;
                best_action = Some(if prob < 0.5 {
                    SolverAction::Reveal(pos)
                } else {
                    SolverAction::Flag(pos)
                });
            }
        }

        // Only return action if we meet confidence threshold
        if let Some(action) = best_action {
            if best_certainty >= self.min_confidence {
                return SolverResult {
                    actions: vec![action],
                    certainty: Certainty::Probabilistic(best_certainty),
                };
            }
        }

        // If we get here and still have no action but have covered squares, pick one
        if let Some(pos) = board
            .iter_positions()
            .find(|&pos| matches!(board.get(pos), Some(SolverCell::Covered)))
        {
            SolverResult {
                actions: vec![SolverAction::Reveal(pos)],
                certainty: Certainty::Probabilistic(0.5),
            }
        } else {
            SolverResult {
                actions: vec![],
                certainty: Certainty::Probabilistic(best_certainty),
            }
        }
    }

    fn name(&self) -> &str {
        "Tank Solver"
    }

    fn is_deterministic(&self) -> bool {
        false
    }
}

// Additional tests specific to Tank solver
#[cfg(test)]
mod tank_specific_tests {
    use super::*;
    use crate::{Board, Cell};

    #[test]
    fn test_constraint_generation() {
        let mut board = Board::new(3, 3, 2).unwrap();
        board.cells.insert(Position::new(1, 1), Cell::Revealed(2));
        let solver = TankSolver::new(0.95);
        let solver_board = SolverBoard::new(&board);

        let constraints = solver.get_constraints(&solver_board);
        assert!(
            !constraints.is_empty(),
            "Should generate at least one constraint"
        );
    }

    #[test]
    fn test_probability_flooding() {
        let mut board = Board::new(3, 3, 1).unwrap();
        board.cells.insert(Position::new(1, 1), Cell::Revealed(1));
        let solver = TankSolver::new(0.95);
        let solver_board = SolverBoard::new(&board);

        let constraints = solver.get_constraints(&solver_board);
        let probabilities = solver.flood_probability(&constraints);

        assert!(
            !probabilities.is_empty(),
            "Should calculate probabilities for covered cells"
        );

        // All probabilities should be between 0 and 1
        for &prob in probabilities.values() {
            assert!(
                prob >= 0.0 && prob <= 1.0,
                "Probabilities must be between 0 and 1"
            );
        }
    }
}
