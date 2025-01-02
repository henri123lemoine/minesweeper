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

    /// Returns true if no cells have been revealed yet
    pub fn is_start(&self) -> bool {
        !self
            .board
            .cells
            .values()
            .any(|cell| matches!(cell, Cell::Revealed(_)))
    }

    /// Returns the probability of any cell being a mine
    pub fn mine_density(&self) -> f64 {
        self.board.mines_count() as f64 / (self.board.width * self.board.height) as f64
    }

    /// Returns the number of mines remaining to be found
    pub fn remaining_mines(&self) -> Option<u32> {
        let total = self.total_mines();
        let marked = self.mines_marked();

        if marked > total {
            return None;
        }

        Some(total - marked)
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

    pub fn total_cells(&self) -> u32 {
        self.board.total_cells()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolverCell {
    Covered,      // Unknown and unflagged cell
    Revealed(u8), // Number of neighboring mines
    Flagged,      // Marked as a mine
}
