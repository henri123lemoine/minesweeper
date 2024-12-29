use crate::{GameError, Position};
use rand::Rng;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    Hidden(bool),
    Revealed(u8),
    Flagged(bool),
}

#[derive(Debug, PartialEq)]
pub enum RevealResult {
    Safe,
    Mine,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Board {
    pub cells: HashMap<Position, Cell>,
    width: u32,
    height: u32,
    mines_count: u32,
    revealed_count: u32,
}

pub struct BoardIterator {
    width: i32,
    height: i32,
    current: Option<Position>,
}

impl Iterator for BoardIterator {
    type Item = Position;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current?;

        self.current = if current.x + 1 < self.width {
            Some(Position::new(current.x + 1, current.y))
        } else if current.y + 1 < self.height {
            Some(Position::new(0, current.y + 1))
        } else {
            None
        };

        Some(current)
    }
}

impl Board {
    pub fn new(width: u32, height: u32, mines_count: u32) -> Result<Self, GameError> {
        if mines_count >= width * height {
            return Err(GameError::TooManyMines {
                width,
                height,
                mines: mines_count,
            });
        }

        let mut board = Board {
            cells: HashMap::new(),
            width,
            height,
            mines_count,
            revealed_count: 0,
        };
        board.initialize_cells();
        board.place_mines();
        Ok(board)
    }

    fn initialize_cells(&mut self) {
        for pos in self.iter_positions() {
            self.cells.insert(pos, Cell::Hidden(false));
        }
    }

    fn place_mines(&mut self) {
        let mut rng = rand::thread_rng();
        let mut mines_placed = 0;

        while mines_placed < self.mines_count {
            let x = rng.gen_range(0..self.width) as i32;
            let y = rng.gen_range(0..self.height) as i32;
            let pos = Position::new(x, y);

            if let Some(Cell::Hidden(false)) = self.cells.get(&pos) {
                self.cells.insert(pos, Cell::Hidden(true));
                mines_placed += 1;
            }
        }
    }

    pub fn is_within_bounds(&self, pos: Position) -> bool {
        pos.x >= 0 && pos.x < self.width as i32 && pos.y >= 0 && pos.y < self.height as i32
    }

    pub fn get_cell(&self, pos: Position) -> Result<&Cell, GameError> {
        self.cells.get(&pos).ok_or(GameError::OutOfBounds(pos))
    }

    pub fn count_adjacent_mines(&self, pos: Position) -> u8 {
        pos.neighbors()
            .filter(|p| self.is_within_bounds(*p))
            .filter(|p| {
                matches!(
                    self.cells.get(p),
                    Some(Cell::Hidden(true) | Cell::Flagged(true))
                )
            })
            .count() as u8
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn mines_count(&self) -> u32 {
        self.mines_count
    }

    pub fn reveal(&mut self, pos: Position) -> Result<RevealResult, GameError> {
        if !self.is_within_bounds(pos) {
            return Err(GameError::OutOfBounds(pos));
        }

        match self.get_cell(pos)? {
            Cell::Revealed(_) => return Err(GameError::AlreadyRevealed(pos)),
            Cell::Flagged(_) => return Ok(RevealResult::Safe),
            Cell::Hidden(true) => {
                self.cells.insert(pos, Cell::Revealed(0));
                return Ok(RevealResult::Mine);
            }
            Cell::Hidden(false) => {
                let mut to_reveal = HashSet::new();
                to_reveal.insert(pos);

                while !to_reveal.is_empty() {
                    let mut next_batch = HashSet::new();

                    for &current_pos in &to_reveal {
                        if let Cell::Hidden(false) = self.get_cell(current_pos)? {
                            let adjacent_mines = self.count_adjacent_mines(current_pos);
                            self.cells
                                .insert(current_pos, Cell::Revealed(adjacent_mines));

                            if adjacent_mines == 0 {
                                for neighbor_pos in current_pos.neighbors() {
                                    if self.is_within_bounds(neighbor_pos) {
                                        if let Ok(Cell::Hidden(false)) = self.get_cell(neighbor_pos)
                                        {
                                            next_batch.insert(neighbor_pos);
                                        }
                                    }
                                }
                            }
                            self.revealed_count += 1;
                        }
                    }

                    to_reveal = next_batch;
                }
            }
        }

        Ok(RevealResult::Safe)
    }

    pub fn toggle_flag(&mut self, pos: Position) -> Result<(), GameError> {
        if !self.is_within_bounds(pos) {
            return Err(GameError::OutOfBounds(pos));
        }

        match self.get_cell(pos)? {
            Cell::Hidden(has_mine) => {
                self.cells.insert(pos, Cell::Flagged(*has_mine));
            }
            Cell::Flagged(has_mine) => {
                self.cells.insert(pos, Cell::Hidden(*has_mine));
            }
            Cell::Revealed(_) => return Err(GameError::AlreadyRevealed(pos)),
        }

        Ok(())
    }

    pub fn iter_positions(&self) -> BoardIterator {
        BoardIterator {
            width: self.width as i32,
            height: self.height as i32,
            current: Some(Position::new(0, 0)),
        }
    }

    pub fn revealed_count(&self) -> u32 {
        self.revealed_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_creation() {
        let board = Board::new(5, 5, 5).unwrap();
        assert_eq!(board.dimensions(), (5, 5));
        assert_eq!(board.mines_count(), 5);

        // Count mines
        let mine_count = board
            .cells
            .values()
            .filter(|cell| matches!(cell, Cell::Hidden(true)))
            .count();
        assert_eq!(mine_count, 5);
    }

    #[test]
    fn test_too_many_mines() {
        let result = Board::new(5, 5, 26);
        assert!(matches!(
            result,
            Err(GameError::TooManyMines {
                width: 5,
                height: 5,
                mines: 26
            })
        ));
    }

    #[test]
    fn test_within_bounds() {
        let board = Board::new(5, 5, 5).unwrap();

        assert!(board.is_within_bounds(Position::new(0, 0)));
        assert!(board.is_within_bounds(Position::new(4, 4)));
        assert!(!board.is_within_bounds(Position::new(5, 5)));
        assert!(!board.is_within_bounds(Position::new(-1, 0)));
    }

    #[test]
    fn test_count_adjacent_mines() {
        let mut board = Board::new(3, 3, 0).unwrap();

        // Place mines manually
        board.cells.insert(Position::new(0, 0), Cell::Hidden(true));
        board.cells.insert(Position::new(1, 0), Cell::Hidden(true));

        assert_eq!(board.count_adjacent_mines(Position::new(0, 1)), 2);
        assert_eq!(board.count_adjacent_mines(Position::new(2, 2)), 0);
    }
}
