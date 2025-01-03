use super::board::{SolverBoard, SolverCell};
use super::solver_test_suite;
use super::traits::{DeterministicResult, DeterministicSolver, Solver};
use crate::Position;

/// Implements basic counting rules for minesweeper solving:
/// - If a numbered cell has exactly as many hidden neighbors as its number, they must all be mines
/// - If a numbered cell has exactly as many flagged/known mines as its number, all other neighbors must be safe
/// - Any neighbor of a 0 is safe
#[derive(Debug, Default)]
pub struct CountingSolver;

impl CountingSolver {
    fn analyze_cell(&self, board: &SolverBoard, pos: Position) -> DeterministicResult {
        let mut result = DeterministicResult::default();

        // Only analyze revealed cells with numbers
        let cell_value = match board.get(pos) {
            Some(SolverCell::Revealed(n)) => n,
            _ => return result,
        };

        // Get all neighbors
        let neighbors = board.neighbors(pos);

        // Count and collect unknown and flagged neighbors
        let mut unknown_positions = Vec::new();
        let mut flagged_count = 0;

        for npos in neighbors {
            match board.get(npos) {
                Some(SolverCell::Covered) => unknown_positions.push(npos),
                Some(SolverCell::Flagged) => flagged_count += 1,
                _ => {}
            }
        }

        let unknown_count = unknown_positions.len();
        let remaining_mines = cell_value.saturating_sub(flagged_count);

        // Case 1: Zero cell - all neighbors are safe
        if cell_value == 0 {
            result.safe.extend(unknown_positions);
            return result;
        }

        // Case 2: If we've found all mines, remaining unknowns must be safe
        if remaining_mines == 0 {
            result.safe.extend(unknown_positions);
            return result;
        }

        // Case 3: If remaining mines equals number of unknown cells, they must all be mines
        if remaining_mines as usize == unknown_count && unknown_count > 0 {
            result.mines.extend(unknown_positions);
            return result;
        }

        result
    }
}

impl Solver for CountingSolver {
    fn name(&self) -> &str {
        "Basic Counting Solver"
    }
}

impl DeterministicSolver for CountingSolver {
    fn solve(&self, board: &SolverBoard) -> DeterministicResult {
        let mut result = DeterministicResult::default();

        // Analyze each cell on the board
        for pos in board.iter_positions() {
            let cell_result = self.analyze_cell(board, pos);
            result.mines.extend(cell_result.mines);
            result.safe.extend(cell_result.safe);
        }

        result
    }
}

solver_test_suite!(CountingSolver, deterministic);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Board, Cell};
    use std::collections::HashSet;

    /// Helper to create a test board with a specific configuration
    fn create_test_board(width: u32, height: u32) -> Board {
        Board::new(width, height, 0).unwrap()
    }

    /// Helper to set a cell's revealed value
    fn reveal_cell(board: &mut Board, pos: Position, value: u8) {
        board.cells.insert(pos, Cell::Revealed(value));
    }

    /// Helper to set a cell as flagged
    fn flag_cell(board: &mut Board, pos: Position) {
        board.cells.insert(pos, Cell::Flagged(true));
    }

    #[test]
    fn test_zero_cell() {
        let mut board = create_test_board(3, 3);
        let center = Position::new(1, 1);

        // Reveal center as 0
        reveal_cell(&mut board, center, 0);

        let solver = CountingSolver;
        let solver_board = SolverBoard::new(&board);
        let result = solver.solve(&solver_board);

        // All neighbors should be safe
        let expected_safe: HashSet<_> = center
            .neighbors()
            .filter(|&pos| board.is_within_bounds(pos))
            .collect();

        assert_eq!(
            result.safe, expected_safe,
            "All neighbors of a zero should be safe"
        );
        assert!(
            result.mines.is_empty(),
            "Zero cell should not identify any mines"
        );
    }

    #[test]
    fn test_simple_constraint() {
        let mut board = create_test_board(2, 2);

        // Reveal three cells, leaving only one unknown
        reveal_cell(&mut board, Position::new(0, 0), 1);
        reveal_cell(&mut board, Position::new(1, 0), 0);
        reveal_cell(&mut board, Position::new(1, 1), 0);
        // Position (0,1) is left unknown and must be a mine

        let solver = CountingSolver;
        let solver_board = SolverBoard::new(&board);
        let result = solver.solve(&solver_board);

        let expected_mines = HashSet::from([Position::new(0, 1)]);
        assert_eq!(
            result.mines, expected_mines,
            "Should identify single constrained mine"
        );
    }

    #[test]
    fn test_corner_constraint() {
        let mut board = create_test_board(3, 3);

        // Set up corner with a '1' and reveal adjacent cells as '0'
        reveal_cell(&mut board, Position::new(0, 0), 1);
        reveal_cell(&mut board, Position::new(1, 0), 0);
        reveal_cell(&mut board, Position::new(0, 1), 0);
        // Position (1,1) is left unknown and must be a mine

        let solver = CountingSolver;
        let solver_board = SolverBoard::new(&board);
        let result = solver.solve(&solver_board);

        let expected_mines = HashSet::from([Position::new(1, 1)]);
        assert_eq!(
            result.mines, expected_mines,
            "Should identify corner-adjacent mine"
        );
    }

    #[test]
    fn test_flag_interaction() {
        let mut board = create_test_board(3, 3);

        // Reveal center with 2 mines
        reveal_cell(&mut board, Position::new(1, 1), 2);
        // Flag one neighbor
        flag_cell(&mut board, Position::new(0, 0));
        // One more mine must be among the remaining neighbors

        let solver = CountingSolver;
        let solver_board = SolverBoard::new(&board);
        let result = solver.solve(&solver_board);

        // The solver can't determine which remaining cell has the mine
        assert!(
            result.mines.is_empty(),
            "Cannot determine exact mine location"
        );
        assert!(
            result.safe.is_empty(),
            "Cannot determine safe cells with remaining mine"
        );
    }

    #[test]
    fn test_exhaustion() {
        let mut board = create_test_board(3, 3);

        // Reveal a '2' and flag both its mines
        reveal_cell(&mut board, Position::new(1, 1), 2);
        flag_cell(&mut board, Position::new(0, 0));
        flag_cell(&mut board, Position::new(2, 2));

        let solver = CountingSolver;
        let solver_board = SolverBoard::new(&board);
        let result = solver.solve(&solver_board);

        // All unflagged neighbors must be safe
        let expected_safe: HashSet<_> = Position::new(1, 1)
            .neighbors()
            .filter(|&pos| {
                board.is_within_bounds(pos)
                    && pos != Position::new(0, 0)
                    && pos != Position::new(2, 2)
            })
            .collect();

        assert_eq!(
            result.safe, expected_safe,
            "Should identify all remaining cells as safe"
        );
        assert!(
            result.mines.is_empty(),
            "Should not identify additional mines"
        );
    }
}
