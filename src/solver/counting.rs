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
    /// Analyzes a single cell and its neighbors to find definite mines and safe squares
    fn analyze_cell(&self, board: &SolverBoard, pos: Position) -> DeterministicResult {
        let mut result = DeterministicResult::default();

        // Only analyze revealed cells with numbers
        let cell_value = match board.get(pos) {
            Some(SolverCell::Revealed(n)) => n,
            _ => return result,
        };

        let neighbors: Vec<_> = board.neighbors(pos);
        let mut unknown_neighbors = Vec::new();
        let mut mine_count = 0;

        // Count known mines and collect unknown positions
        for &npos in &neighbors {
            match board.get(npos) {
                Some(SolverCell::Covered) => unknown_neighbors.push(npos),
                Some(SolverCell::Flagged) => mine_count += 1,
                _ => {}
            }
        }

        // Case 1: If cell=0, all neighbors must be safe
        if cell_value == 0 {
            result.safe.extend(unknown_neighbors);
            return result;
        }

        let remaining_mines = cell_value as usize - mine_count;

        // Case 2: If remaining mines equals number of unknown neighbors, they must all be mines
        if remaining_mines == unknown_neighbors.len() {
            result.mines.extend(unknown_neighbors);
            return result;
        }

        // Case 3: If we've found all mines, remaining unknowns must be safe
        if remaining_mines == 0 {
            result.safe.extend(unknown_neighbors);
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
    use crate::Board;

    #[test]
    fn test_zero_cell() {
        let mut board = Board::new(3, 3, 1).unwrap();

        // Create a known board state with a zero cell
        board.cells.clear();
        for x in 0..3 {
            for y in 0..3 {
                board
                    .cells
                    .insert(Position::new(x, y), crate::Cell::Hidden(false));
            }
        }

        // Reveal center cell (should be 0 as no adjacent mines)
        board.reveal(Position::new(1, 1)).unwrap();

        let solver = CountingSolver;
        let solver_board = SolverBoard::new(&board);
        let result = solver.solve(&solver_board);

        // All neighbors of a zero should be marked safe
        assert_eq!(
            result.safe.len(),
            8,
            "All neighbors of a zero should be safe"
        );
        assert!(result.mines.is_empty(), "No mines should be identified");
    }

    #[test]
    fn test_fully_constrained_cell() {
        let mut board = Board::new(3, 3, 2).unwrap();

        // Create a known board state where center cell has exactly 2 mine neighbors
        board.cells.clear();
        board
            .cells
            .insert(Position::new(0, 0), crate::Cell::Hidden(true));
        board
            .cells
            .insert(Position::new(0, 1), crate::Cell::Hidden(true));
        for x in 0..3 {
            for y in 0..3 {
                if !matches!(
                    board.get_cell(Position::new(x, y)),
                    Ok(crate::Cell::Hidden(true))
                ) {
                    board
                        .cells
                        .insert(Position::new(x, y), crate::Cell::Hidden(false));
                }
            }
        }

        // Reveal center cell
        board.reveal(Position::new(1, 1)).unwrap();

        let solver = CountingSolver;
        let solver_board = SolverBoard::new(&board);
        let result = solver.solve(&solver_board);

        assert_eq!(result.mines.len(), 2, "Should identify exactly 2 mines");
        assert!(result.mines.contains(&Position::new(0, 0)));
        assert!(result.mines.contains(&Position::new(0, 1)));
    }

    #[test]
    fn test_partially_constrained_cell() {
        let mut board = Board::new(3, 3, 1).unwrap();

        // Create a board state with one known mine and one revealed number
        board.cells.clear();
        board
            .cells
            .insert(Position::new(0, 0), crate::Cell::Hidden(true));
        board
            .cells
            .insert(Position::new(1, 1), crate::Cell::Revealed(1));

        let solver = CountingSolver;
        let solver_board = SolverBoard::new(&board);
        let result = solver.solve(&solver_board);

        // Should identify remaining neighbors as safe since we know where the one mine is
        for pos in Position::new(1, 1).neighbors() {
            if pos != Position::new(0, 0) && board.is_within_bounds(pos) {
                assert!(
                    result.safe.contains(&pos),
                    "Position {:?} should be marked safe",
                    pos
                );
            }
        }
    }
}
