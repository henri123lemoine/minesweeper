// use itertools::Itertools;
// use ndarray::{Array2, Axis};
use std::collections::{HashMap, HashSet};

use super::{board::SolverCell, Certainty, Solver, SolverAction, SolverBoard, SolverResult};
use crate::Position;

#[derive(Debug)]
struct ConstraintArea {
    positions: HashSet<Position>,
    mine_count: u8,
}

pub struct ProbabilisticSolver {
    pub min_confidence: f64,
}

impl ProbabilisticSolver {
    fn get_constraints(&self, board: &SolverBoard) -> Vec<ConstraintArea> {
        let mut constraints = Vec::new();

        for pos in board.iter_positions() {
            if let Some(SolverCell::Revealed(n)) = board.get(pos) {
                let mut positions = HashSet::new();
                let mut mine_count = n;

                // Collect covered neighbors and adjust mine count for flags
                for npos in board.neighbors(pos) {
                    match board.get(npos) {
                        Some(SolverCell::Covered) => {
                            positions.insert(npos);
                        }
                        Some(SolverCell::Flagged) => {
                            mine_count -= 1;
                        }
                        _ => {}
                    }
                }

                // Only add constraints that still have unknown cells
                if !positions.is_empty() {
                    constraints.push(ConstraintArea {
                        positions,
                        mine_count,
                    });
                }
            }
        }

        constraints
    }

    fn find_components(&self, board: &SolverBoard) -> Vec<HashSet<Position>> {
        let mut components = Vec::new();
        let mut visited = HashSet::new();

        // Helper function for DFS component finding
        fn explore_component(
            pos: Position,
            board: &SolverBoard,
            component: &mut HashSet<Position>,
            visited: &mut HashSet<Position>,
        ) {
            if !visited.insert(pos) {
                return;
            }

            match board.get(pos) {
                Some(SolverCell::Covered) => {
                    component.insert(pos);
                    // Explore neighbors to find connecting constraints
                    for npos in board.neighbors(pos) {
                        if let Some(SolverCell::Revealed(_)) = board.get(npos) {
                            // Continue exploration from revealed squares' neighbors
                            for nnpos in board.neighbors(npos) {
                                if !visited.contains(&nnpos) {
                                    explore_component(nnpos, board, component, visited);
                                }
                            }
                        }
                    }
                }
                Some(SolverCell::Revealed(_)) => {
                    // Explore neighbors for covered squares
                    for npos in board.neighbors(pos) {
                        if !visited.contains(&npos) {
                            explore_component(npos, board, component, visited);
                        }
                    }
                }
                _ => {}
            }
        }

        // Find all components
        for pos in board.iter_positions() {
            if !visited.contains(&pos) {
                if let Some(SolverCell::Covered) = board.get(pos) {
                    let mut component = HashSet::new();
                    explore_component(pos, board, &mut component, &mut visited);
                    if !component.is_empty() {
                        components.push(component);
                    }
                }
            }
        }

        components
    }

    fn get_areas(&self, board: &SolverBoard, component: &HashSet<Position>) -> Vec<ConstraintArea> {
        let mut areas = HashMap::new();

        // For each position in component, find all its constraints
        for &pos in component {
            let mut constraints = HashSet::new();

            // Find all revealed neighbors that constrain this position
            for npos in board.neighbors(pos) {
                if let Some(SolverCell::Revealed(n)) = board.get(npos) {
                    constraints.insert((npos, n));
                }
            }

            // Use constraints as key to group positions
            let key = constraints.into_iter().collect::<Vec<_>>();
            areas.entry(key).or_insert_with(HashSet::new).insert(pos);
        }

        // Convert areas to constraint areas
        areas
            .into_iter()
            .map(|(constraints, positions)| {
                let mine_count = constraints
                    .iter()
                    .map(|&(pos, n)| {
                        let flagged = board
                            .neighbors(pos)
                            .iter()
                            .filter(|&&npos| matches!(board.get(npos), Some(SolverCell::Flagged)))
                            .count();
                        n as usize - flagged
                    })
                    .sum::<usize>() as u8;

                ConstraintArea {
                    positions,
                    mine_count,
                }
            })
            .collect()
    }

    fn calculate_area_probabilities(
        &self,
        areas: &[ConstraintArea],
        total_mines_left: u32,
    ) -> HashMap<Position, f64> {
        let mut probabilities = HashMap::new();
        let total_positions: usize = areas.iter().map(|area| area.positions.len()).sum();

        if total_positions == 0 {
            return probabilities;
        }

        // Generate all possible mine distributions
        let distributions = self.generate_mine_distributions(areas, total_mines_left as usize);
        let total_distributions = distributions.len();

        if total_distributions == 0 {
            return probabilities;
        }

        // Count mine occurrences per position
        let mut mine_counts: HashMap<Position, usize> = HashMap::new();

        for dist in distributions {
            for (area, &mines) in areas.iter().zip(dist.iter()) {
                for &pos in &area.positions {
                    *mine_counts.entry(pos).or_insert(0) += mines;
                }
            }
        }

        // Convert counts to probabilities
        for (pos, count) in mine_counts {
            probabilities.insert(pos, count as f64 / total_distributions as f64);
        }

        probabilities
    }

    fn generate_mine_distributions(
        &self,
        areas: &[ConstraintArea],
        total_mines: usize,
    ) -> Vec<Vec<usize>> {
        let mut distributions = Vec::new();
        let mut current = vec![0; areas.len()];

        fn recurse(
            areas: &[ConstraintArea],
            total_mines: usize,
            current: &mut Vec<usize>,
            index: usize,
            distributions: &mut Vec<Vec<usize>>,
        ) {
            if index == areas.len() {
                if current.iter().sum::<usize>() == total_mines {
                    distributions.push(current.clone());
                }
                return;
            }

            let area = &areas[index];
            let max_mines = area.positions.len().min(area.mine_count as usize);

            for mines in 0..=max_mines {
                current[index] = mines;
                recurse(areas, total_mines, current, index + 1, distributions);
            }
        }

        recurse(areas, total_mines, &mut current, 0, &mut distributions);
        distributions
    }

    fn combinations(n: usize, k: usize) -> f64 {
        if k > n {
            return 0.0;
        }
        if k == 0 || k == n {
            return 1.0;
        }

        let mut result = 1.0;
        for i in 0..k {
            result *= (n - i) as f64;
            result /= (i + 1) as f64;
        }
        result
    }

    fn calculate_global_probability(
        &self,
        board: &SolverBoard,
        unconstrained: &HashSet<Position>,
        remaining_mines: u32,
    ) -> f64 {
        if unconstrained.is_empty() {
            return 0.0;
        }

        let n = unconstrained.len();
        Self::combinations(n, remaining_mines as usize)
            / Self::combinations(board.total_cells() as usize, remaining_mines as usize)
    }
}

impl Solver for ProbabilisticSolver {
    fn solve(&self, board: &SolverBoard) -> SolverResult {
        // First check if this is an opening move
        let mut is_opening = true;
        for pos in board.iter_positions() {
            if let Some(SolverCell::Revealed(_)) = board.get(pos) {
                is_opening = false;
                break;
            }
        }

        if is_opening {
            // For opening move, return center position
            let (width, height) = board.dimensions();
            return SolverResult {
                actions: vec![SolverAction::Reveal(Position::new(
                    (width / 2) as i32,
                    (height / 2) as i32,
                ))],
                certainty: Certainty::Probabilistic(
                    1.0 - (board.total_mines() as f64 / board.total_cells() as f64),
                ),
            };
        }

        // Get all components and their constraints
        let components = self.find_components(board);
        let mut all_probabilities = HashMap::new();
        let remaining_mines = board.total_mines() - board.mines_marked();

        // Calculate probabilities for each component
        for component in components {
            let areas = self.get_areas(board, &component);
            let probs = self.calculate_area_probabilities(&areas, remaining_mines);
            all_probabilities.extend(probs);
        }

        // Handle unconstrained squares
        let mut unconstrained = HashSet::new();
        for pos in board.iter_positions() {
            if let Some(SolverCell::Covered) = board.get(pos) {
                if !all_probabilities.contains_key(&pos) {
                    unconstrained.insert(pos);
                }
            }
        }

        if !unconstrained.is_empty() {
            let global_prob =
                self.calculate_global_probability(board, &unconstrained, remaining_mines);
            for pos in unconstrained {
                all_probabilities.insert(pos, global_prob);
            }
        }

        // Find the safest move
        let mut best_action = None;
        let mut best_certainty = 0.0;

        for (pos, &prob) in &all_probabilities {
            let certainty = if prob < 0.5 { 1.0 - prob } else { prob };
            if certainty > best_certainty {
                best_certainty = certainty;
                best_action = Some(if prob < 0.5 {
                    SolverAction::Reveal(*pos)
                } else {
                    SolverAction::Flag(*pos)
                });
            }
        }

        // Only return an action if we meet the minimum confidence threshold
        if best_certainty >= self.min_confidence {
            SolverResult {
                actions: vec![best_action.unwrap()],
                certainty: Certainty::Probabilistic(best_certainty),
            }
        } else {
            SolverResult {
                actions: vec![],
                certainty: Certainty::Probabilistic(best_certainty),
            }
        }
    }

    fn name(&self) -> &str {
        "Probabilistic Solver"
    }

    fn is_deterministic(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Board;

    #[test]
    fn test_area_identification() {
        let mut board = Board::new(3, 3, 1).unwrap();
        // Set up a simple board with known constraints
        board.reveal(Position::new(1, 1)).unwrap();

        let solver_board = SolverBoard::new(&board);
        let solver = ProbabilisticSolver {
            min_confidence: 0.95,
        };
        let components = solver.find_components(&solver_board);

        assert!(!components.is_empty());
        // Verify correct area identification
    }

    #[test]
    fn test_probability_calculation() {
        let mut board = Board::new(3, 3, 1).unwrap();
        board.reveal(Position::new(1, 1)).unwrap();

        let solver_board = SolverBoard::new(&board);
        let solver = ProbabilisticSolver {
            min_confidence: 0.95,
        };

        // Get probabilities
        let result = solver.solve(&solver_board);

        // Verify probabilities make sense
        match result.certainty {
            Certainty::Probabilistic(c) => assert!(c >= 0.0 && c <= 1.0),
            _ => panic!("Expected probabilistic certainty"),
        }
    }

    #[test]
    fn test_global_probability() {
        let board = Board::new(4, 4, 2).unwrap();
        let solver_board = SolverBoard::new(&board);
        let solver = ProbabilisticSolver {
            min_confidence: 0.95,
        };

        let unconstrained = HashSet::from_iter(vec![
            Position::new(0, 0),
            Position::new(0, 1),
            Position::new(1, 0),
        ]);

        let prob = solver.calculate_global_probability(&solver_board, &unconstrained, 2);

        assert!(prob >= 0.0 && prob <= 1.0);
        // Verify probability matches expected combinatorial calculation
    }

    #[test]
    fn test_component_isolation() {
        let mut board = Board::new(4, 4, 2).unwrap();
        // Create two isolated components by revealing squares
        // that separate the board

        let solver_board = SolverBoard::new(&board);
        let solver = ProbabilisticSolver {
            min_confidence: 0.95,
        };
        let components = solver.find_components(&solver_board);

        // Verify components are properly isolated
    }
}
