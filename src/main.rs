use minesweeper::{Action, Cell, Game, GameError, GameState, Position};
use std::io::{self, Write};

fn main() {
    match run_game() {
        Ok(_) => println!("Thanks for playing!"),
        Err(e) => eprintln!("Game error: {}", e),
    }
}

fn run_game() -> Result<(), GameError> {
    let mut game = Game::new(10, 10, 10)?;

    while game.state() == GameState::Playing {
        print_board(&game);

        if let Some((pos, action)) = get_user_input(&game) {
            if let Err(e) = game.perform_action(pos, action) {
                println!("Error: {}", e);
                continue;
            }
        }
    }

    print_board(&game);
    match game.state() {
        GameState::Won => println!("Congratulations! You won!"),
        GameState::Lost => println!("Game Over!"),
        GameState::Playing => unreachable!(),
    }

    Ok(())
}

fn print_board(game: &Game) {
    let (width, height) = game.dimensions();

    // Print column numbers
    print!("  ");
    for x in 0..width {
        print!("{} ", x);
    }
    println!();

    // Print rows
    for y in 0..height {
        print!("{} ", y);
        for x in 0..width {
            let pos = Position::new(x as i32, y as i32);
            match game.get_cell(pos).unwrap() {
                Cell::Hidden(_) => print!("□ "),
                Cell::Revealed(0) => print!("  "),
                Cell::Revealed(n) => print!("{} ", n),
                Cell::Flagged(_) => print!("⚑ "),
            }
        }
        println!();
    }
}

fn get_user_input(game: &Game) -> Option<(Position, Action)> {
    print!("Enter command (x y [r/f]): ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).ok()?;

    let mut parts = input.split_whitespace();

    let x = parts.next()?.parse().ok()?;
    let y = parts.next()?.parse().ok()?;
    let action = parts.next()?.chars().next()?;

    let pos = Position::new(x, y);

    if !game.get_cell(pos).is_ok() {
        println!("Position out of bounds");
        return None;
    }

    let action = match action {
        'r' => Some(Action::Reveal),
        'f' => Some(Action::Flag),
        _ => {
            println!("Invalid action. Use 'r' to reveal or 'f' to flag");
            None
        }
    }?;

    Some((pos, action))
}
