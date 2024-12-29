use criterion::{criterion_group, criterion_main, Criterion};
use minesweeper::{
    solver::{
        CountingSolver, MatrixSolver, ProbabilisticSolver, Solver, SolverAction, SolverBoard,
    },
    Board, Cell, Position,
};

#[derive(Debug, Default)]
struct SolverStats {
    games_played: usize,
    games_won: usize,
    games_lost: usize,
    total_moves: usize,
    safe_moves: usize,
    mines_hit: usize,
    cells_remaining: usize,
}

impl SolverStats {
    fn success_rate(&self) -> f64 {
        if self.games_played == 0 {
            return 0.0;
        }
        self.games_won as f64 / self.games_played as f64 * 100.0
    }

    fn average_completion(&self, board_size: usize) -> f64 {
        if self.games_played == 0 {
            return 0.0;
        }
        (board_size - self.cells_remaining) as f64 / board_size as f64 * 100.0
    }
}

fn solve_single_game(board: &mut Board, solver: &dyn Solver) -> SolverStats {
    let mut stats = SolverStats::default();
    stats.games_played = 1;

    // Make initial move at (1,1)
    if let Ok(Cell::Hidden(is_mine)) = board.get_cell(Position::new(1, 1)) {
        if *is_mine {
            stats.games_lost = 1;
            stats.mines_hit = 1;
            return stats;
        }
    }
    let _ = board.reveal(Position::new(1, 1));

    // Keep solving until we can't make progress or hit a mine
    for _ in 0..100 {
        let solver_board = SolverBoard::new(board);
        let result = solver.solve(&solver_board);

        if result.actions.is_empty() {
            break;
        }

        stats.total_moves += 1;

        for action in result.actions {
            match action {
                SolverAction::Reveal(pos) => {
                    if let Ok(Cell::Hidden(is_mine)) = board.get_cell(pos) {
                        if *is_mine {
                            stats.mines_hit += 1;
                            stats.games_lost = 1;
                            return stats;
                        } else {
                            stats.safe_moves += 1;
                        }
                    }
                    let _ = board.reveal(pos);
                }
                SolverAction::Flag(_) => {}
            }
        }
    }

    // Count remaining hidden cells
    let total_cells = board.dimensions().0 * board.dimensions().1;
    let mines = board.mines_count();
    stats.cells_remaining = board
        .iter_positions()
        .filter(|&pos| matches!(board.get_cell(pos), Ok(Cell::Hidden(_))))
        .count();

    // If we cleared all non-mine cells, we won
    if stats.cells_remaining as u32 == mines {
        stats.games_won = 1;
    }

    stats
}

fn run_solver_benchmark(
    width: u32,
    height: u32,
    mines: u32,
    solver: &dyn Solver,
    iterations: usize,
) -> SolverStats {
    let mut total_stats = SolverStats::default();
    total_stats.games_played = iterations;

    for _ in 0..iterations {
        let mut board = Board::new(width, height, mines).unwrap();
        let game_stats = solve_single_game(&mut board, solver);

        total_stats.games_won += game_stats.games_won;
        total_stats.games_lost += game_stats.games_lost;
        total_stats.total_moves += game_stats.total_moves;
        total_stats.safe_moves += game_stats.safe_moves;
        total_stats.mines_hit += game_stats.mines_hit;
        total_stats.cells_remaining += game_stats.cells_remaining;
    }

    total_stats
}

fn benchmark_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("Solvers");

    let test_configs = vec![
        (8, 8, 10),   // Beginner
        (16, 16, 40), // Intermediate
        (30, 16, 99), // Expert
    ];

    let solvers: Vec<(&dyn Solver, &str)> = vec![
        (&CountingSolver, "Counting"),
        (&MatrixSolver, "Matrix"),
        (
            &ProbabilisticSolver {
                min_confidence: 0.95,
            },
            "Probabilistic",
        ),
    ];

    for (width, height, mines) in test_configs {
        for (solver, name) in &solvers {
            // Performance benchmark
            group.bench_function(format!("{} {}x{}", name, width, height), |b| {
                b.iter_with_setup(
                    || Board::new(width, height, mines).unwrap(),
                    |mut board| {
                        let stats = solve_single_game(&mut board, *solver);
                        criterion::black_box(stats)
                    },
                );
            });

            // Effectiveness stats (100 iterations)
            let stats = run_solver_benchmark(width, height, mines, *solver, 100);
            let board_size = (width * height) as usize;

            println!("\n{} on {}x{} board:", name, width, height);
            println!("Success rate: {:.1}%", stats.success_rate());
            println!(
                "Board completion: {:.1}%",
                stats.average_completion(board_size)
            );
            println!(
                "Average moves per game: {:.1}",
                stats.total_moves as f64 / 100.0
            );
            println!("Mine hits: {}", stats.mines_hit);
            println!("Safe moves: {}", stats.safe_moves);
        }
    }

    group.finish();
}

criterion_group!(benches, benchmark_solvers);
criterion_main!(benches);
