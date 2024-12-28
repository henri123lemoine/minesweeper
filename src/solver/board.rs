use crate::{Board, BoardIterator, Cell, Position};

/// A view of the game board that hides information the solver shouldn't have access to
#[derive(Debug)]
pub struct SolverBoard<'a> {
    board: &'a Board,
}

impl<'a> SolverBoard<'a> {
    pub fn new(board: &'a Board) -> Self {
        Self { board }
    }

    /// Gets cell state without revealing mine information
    pub fn get(&self, pos: Position) -> Option<SolverCell> {
        self.board.get_cell(pos).ok().map(|cell| match cell {
            Cell::Hidden(_) => SolverCell::Covered,
            Cell::Revealed(n) => SolverCell::Revealed(*n),
            Cell::Flagged(_) => SolverCell::Flagged,
        })
    }

    pub fn dimensions(&self) -> (u32, u32) {
        self.board.dimensions()
    }

    pub fn total_mines(&self) -> u32 {
        self.board.mines_count()
    }

    pub fn mines_marked(&self) -> u32 {
        self.board
            .cells
            .values()
            .filter(|cell| matches!(cell, Cell::Flagged(_)))
            .count() as u32
    }

    pub fn neighbors(&self, pos: Position) -> Vec<Position> {
        pos.neighbors()
            .filter(|p| self.board.is_within_bounds(*p))
            .collect()
    }

    pub fn iter_positions(&self) -> BoardIterator {
        self.board.iter_positions()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolverCell {
    Covered,      // Unknown and unflagged cell
    Revealed(u8), // Number of neighboring mines
    Flagged,      // Marked as a mine
}
