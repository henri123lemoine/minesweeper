use crate::{Board, Cell, GameError, Position};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GameState {
    Playing,
    Won,
    Lost,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
    Reveal,
    Flag,
}

pub struct Game {
    board: Board,
    state: GameState,
    revealed_count: u32,
}

impl Game {
    pub fn new(width: u32, height: u32, mines_count: u32) -> Result<Self, GameError> {
        Ok(Self {
            board: Board::new(width, height, mines_count)?,
            state: GameState::Playing,
            revealed_count: 0,
        })
    }

    pub fn get_cell(&self, pos: Position) -> Result<&Cell, GameError> {
        self.board.get_cell(pos)
    }

    pub fn perform_action(&mut self, pos: Position, action: Action) -> Result<(), GameError> {
        if self.state != GameState::Playing {
            return Err(GameError::InvalidGameState);
        }

        match action {
            Action::Reveal => self.reveal(pos),
            Action::Flag => self.toggle_flag(pos),
        }
    }

    fn reveal(&mut self, pos: Position) -> Result<(), GameError> {
        if !self.board.is_within_bounds(pos) {
            return Err(GameError::OutOfBounds(pos));
        }

        match self.board.get_cell(pos)? {
            Cell::Revealed(_) => return Err(GameError::AlreadyRevealed(pos)),
            Cell::Flagged(_) => return Ok(()),
            Cell::Hidden(true) => {
                self.board.cells.insert(pos, Cell::Revealed(0));
                self.state = GameState::Lost;
                return Ok(());
            }
            Cell::Hidden(false) => {
                let mut to_reveal = HashSet::new();
                to_reveal.insert(pos);

                while !to_reveal.is_empty() {
                    let mut next_batch = HashSet::new();

                    for &current_pos in &to_reveal {
                        if let Cell::Hidden(false) = self.board.get_cell(current_pos)? {
                            let adjacent_mines = self.board.count_adjacent_mines(current_pos);
                            self.revealed_count += 1;
                            self.board
                                .cells
                                .insert(current_pos, Cell::Revealed(adjacent_mines));

                            if adjacent_mines == 0 {
                                for neighbor_pos in current_pos.neighbors() {
                                    if self.board.is_within_bounds(neighbor_pos) {
                                        if let Ok(Cell::Hidden(false)) =
                                            self.board.get_cell(neighbor_pos)
                                        {
                                            next_batch.insert(neighbor_pos);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    to_reveal = next_batch;
                }
            }
        }

        self.check_win_condition();
        Ok(())
    }

    fn toggle_flag(&mut self, pos: Position) -> Result<(), GameError> {
        if !self.board.is_within_bounds(pos) {
            return Err(GameError::OutOfBounds(pos));
        }

        match self.board.get_cell(pos)? {
            Cell::Hidden(has_mine) => {
                self.board.cells.insert(pos, Cell::Flagged(*has_mine));
            }
            Cell::Flagged(has_mine) => {
                self.board.cells.insert(pos, Cell::Hidden(*has_mine));
            }
            Cell::Revealed(_) => return Err(GameError::AlreadyRevealed(pos)),
        }

        Ok(())
    }

    fn check_win_condition(&mut self) {
        let (width, height) = self.board.dimensions();
        let total_non_mine_cells = (width * height) - self.board.mines_count();

        if self.revealed_count == total_non_mine_cells {
            self.state = GameState::Won;
        }
    }

    pub fn state(&self) -> GameState {
        self.state
    }

    pub fn dimensions(&self) -> (u32, u32) {
        self.board.dimensions()
    }
}
