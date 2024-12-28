use minesweeper::{Board, Cell};
use std::io::{self, Write};

fn main() {
    let mut board = Board::new(10, 10, 10); // 10x10 board with 10 mines

    while !board.is_game_over() && !board.is_won() {
        print_board(&board);

        if let Some((x, y, action)) = get_user_input(&board) {
            match action {
                'r' => {
                    if !board.reveal(x, y) {
                        println!("Game Over!");
                        break;
                    }
                }
                'f' => {
                    board.toggle_flag(x, y);
                }
                _ => println!("Invalid action"),
            }
        }
    }

    print_board(&board);
    if board.is_won() {
        println!("Congratulations! You won!");
    }
}

fn print_board(board: &Board) {
    let (width, height) = board.dimensions();

    // Print column numbers
    print!("  ");
    for x in 0..width {
        print!("{} ", x);
    }
    println!();

    // Print rows
    for y in 0..height {
        print!("{} ", y); // Row numbers
        for x in 0..width {
            match board.get_cell(x, y).unwrap() {
                Cell::Hidden(_) => print!("□ "),
                Cell::Revealed(0) => print!("  "),
                Cell::Revealed(n) => print!("{} ", n),
                Cell::Flagged => print!("⚑ "),
            }
        }
        println!();
    }
}

fn get_user_input(board: &Board) -> Option<(usize, usize, char)> {
    print!("Enter command (x y [r/f]): ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).ok()?;

    let mut parts = input.split_whitespace();

    let x = parts.next()?.parse().ok()?;
    let y = parts.next()?.parse().ok()?;
    let action = parts.next()?.chars().next()?;

    let (width, height) = board.dimensions();
    if x >= width || y >= height {
        println!("Position out of bounds");
        return None;
    }

    if action != 'r' && action != 'f' {
        println!("Invalid action. Use 'r' to reveal or 'f' to flag");
        return None;
    }

    Some((x, y, action))
}
