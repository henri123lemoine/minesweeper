use crate::{Board, Cell, GameError, Position, RevealResult};

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
    pub board: Board,
    state: GameState,
}

impl Game {
    pub fn new(width: u32, height: u32, mines_count: u32) -> Result<Self, GameError> {
        Ok(Self {
            board: Board::new(width, height, mines_count)?,
            state: GameState::Playing,
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
            Action::Reveal => match self.board.reveal(pos)? {
                RevealResult::Mine => {
                    self.state = GameState::Lost;
                }
                RevealResult::Safe => {
                    self.check_win_condition();
                }
            },
            Action::Flag => self.board.toggle_flag(pos)?,
        }
        Ok(())
    }

    fn check_win_condition(&mut self) {
        let (width, height) = self.board.dimensions();
        let total_non_mine_cells = (width * height) - self.board.mines_count();

        if self.board.revealed_count() == total_non_mine_cells {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_game() -> Game {
        Game::new(3, 3, 2).unwrap()
    }

    #[test]
    fn test_game_creation() {
        let game = create_test_game();
        assert_eq!(game.state(), GameState::Playing);
        assert_eq!(game.dimensions(), (3, 3));
    }

    #[test]
    fn test_reveal_empty_cell() {
        let mut game = create_test_game();
        let pos = Position::new(0, 0);

        // Ensure the cell at (0,0) is not a mine by setting it explicitly
        game.board.cells.insert(pos, Cell::Hidden(false));

        game.board.reveal(pos).unwrap();
        match game.get_cell(pos).unwrap() {
            Cell::Revealed(_) => {}
            _ => panic!("Cell should be revealed"),
        }
        assert_eq!(game.state(), GameState::Playing);
    }

    #[test]
    fn test_reveal_mine() {
        let mut game = create_test_game();
        let pos = Position::new(0, 0);

        // Place a mine at (0,0)
        game.board.cells.insert(pos, Cell::Hidden(true));

        game.perform_action(pos, Action::Reveal).unwrap();
        assert_eq!(game.state(), GameState::Lost);
    }

    #[test]
    fn test_flag_cell() {
        let mut game = create_test_game();
        let pos = Position::new(0, 0);

        game.board.toggle_flag(pos).unwrap();
        assert!(matches!(game.get_cell(pos).unwrap(), Cell::Flagged(_)));

        // Toggle flag again should return to hidden
        game.board.toggle_flag(pos).unwrap();
        assert!(matches!(game.get_cell(pos).unwrap(), Cell::Hidden(_)));
    }

    #[test]
    fn test_out_of_bounds() {
        let mut game = create_test_game();
        let pos = Position::new(5, 5);

        assert!(matches!(
            game.board.reveal(pos),
            Err(GameError::OutOfBounds(_))
        ));
    }

    #[test]
    fn test_reveal_flagged_cell() {
        let mut game = create_test_game();
        let pos = Position::new(0, 0);

        game.board.toggle_flag(pos).unwrap();
        assert!(matches!(game.get_cell(pos).unwrap(), Cell::Flagged(_)));

        // Revealing a flagged cell should have no effect
        game.board.reveal(pos).unwrap();
        assert!(matches!(game.get_cell(pos).unwrap(), Cell::Flagged(_)));
    }

    #[test]
    fn test_win_condition() {
        let mut game = Game::new(2, 1, 1).unwrap();
        let mine_pos = Position::new(0, 0);
        let safe_pos = Position::new(1, 0);

        // Set up a simple board with one mine
        game.board.cells.insert(mine_pos, Cell::Hidden(true));
        game.board.cells.insert(safe_pos, Cell::Hidden(false));

        // Reveal the safe cell
        game.perform_action(safe_pos, Action::Reveal).unwrap();
        assert_eq!(game.state(), GameState::Won);
    }
}
