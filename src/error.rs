use crate::Position;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GameError {
    #[error("Position {0:?} is out of bounds")]
    OutOfBounds(Position),
    #[error("Cannot reveal or flag cell in current game state")]
    InvalidGameState,
    #[error("Cell at {0:?} is already revealed")]
    AlreadyRevealed(Position),
    #[error("Too many mines ({mines}) for board size {width}x{height}")]
    TooManyMines { width: u32, height: u32, mines: u32 },
}
