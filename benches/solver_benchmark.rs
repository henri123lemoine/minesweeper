use criterion::{criterion_group, criterion_main, Criterion};
use minesweeper::{
    solver::{
        ChainSolver, CountingSolver, MatrixSolver, ProbabilisticSolver, Solver, SolverAction,
        SolverBoard,
    },
    Board, Cell, Position,
};
use std::time::Duration;

#[derive(Debug, Default)]
struct GameStats {
    won: bool,
    lost: bool,
    moves_made: usize,
    safe_moves: usize,
    mines_hit: usize,
    cells_remaining: usize,
}

#[derive(Debug, Default)]
struct AggregateStats {
    games: Vec<GameStats>,
}

impl AggregateStats {
    fn games_played(&self) -> usize {
        self.games.len()
    }

    fn success_rate(&self) -> f64 {
        if self.games_played() == 0 {
            return 0.0;
        }
        self.games.iter().filter(|g| g.won).count() as f64 / self.games_played() as f64 * 100.0
    }

    fn average_completion(&self, board_size: usize) -> f64 {
        if self.games_played() == 0 {
            return 0.0;
        }
        let completions: Vec<f64> = self
            .games
            .iter()
            .map(|g| (board_size - g.cells_remaining) as f64 / board_size as f64 * 100.0)
            .collect();
        completions.iter().sum::<f64>() / self.games_played() as f64
    }

    fn average_moves(&self) -> f64 {
        if self.games_played() == 0 {
            return 0.0;
        }
        self.games.iter().map(|g| g.moves_made).sum::<usize>() as f64 / self.games_played() as f64
    }

    fn total_mines_hit(&self) -> usize {
        self.games.iter().map(|g| g.mines_hit).sum()
    }

    fn total_safe_moves(&self) -> usize {
        self.games.iter().map(|g| g.safe_moves).sum()
    }
}

fn solve_single_game(board: &mut Board, solver: &dyn Solver) -> GameStats {
    let mut stats = GameStats::default();
    let total_safe_cells = (board.dimensions().0 * board.dimensions().1) - board.mines_count();

    // Create a list of safe starting positions to try
    let starting_positions = vec![
        Position::new(0, 0),
        Position::new(1, 1),
        Position::new(2, 2),
        Position::new(
            board.dimensions().0 as i32 / 2,
            board.dimensions().1 as i32 / 2,
        ),
    ];

    // Try each starting position until we find a safe one
    let mut started = false;
    for pos in starting_positions {
        if let Ok(Cell::Hidden(is_mine)) = board.get_cell(pos) {
            if !is_mine {
                let _ = board.reveal(pos);
                started = true;
                stats.safe_moves += 1;
                break;
            }
        }
    }

    if !started {
        stats.lost = true;
        stats.mines_hit += 1;
        return stats;
    }

    // Keep solving until we can't make progress or hit a mine
    let mut previous_revealed = 0;
    let mut stall_count = 0;

    while stall_count < 3 {
        // Allow a few attempts to break through stuck positions
        let solver_board = SolverBoard::new(board);
        let result = solver.solve(&solver_board);

        if result.actions.is_empty() {
            stall_count += 1;
            continue;
        }

        stats.moves_made += 1;
        let mut made_progress = false;

        for action in result.actions {
            match action {
                SolverAction::Reveal(pos) => {
                    if let Ok(Cell::Hidden(is_mine)) = board.get_cell(pos) {
                        if *is_mine {
                            stats.mines_hit += 1;
                            stats.lost = true;
                            return stats;
                        } else {
                            stats.safe_moves += 1;
                            let _ = board.reveal(pos);
                            made_progress = true;
                        }
                    }
                }
                SolverAction::Flag(pos) => {
                    if let Ok(Cell::Hidden(is_mine)) = board.get_cell(pos) {
                        if *is_mine {
                            let _ = board.toggle_flag(pos);
                            made_progress = true;
                        } else {
                            // Incorrectly flagging a safe cell should count as a loss
                            stats.lost = true;
                            return stats;
                        }
                    }
                }
            }
        }

        let current_revealed = board.revealed_count();
        if current_revealed == previous_revealed && !made_progress {
            stall_count += 1;
        } else {
            stall_count = 0;
        }
        previous_revealed = current_revealed;

        // Check if we've won
        if board.revealed_count() == total_safe_cells {
            stats.won = true;
            break;
        }
    }

    // Count remaining hidden cells
    stats.cells_remaining = board
        .iter_positions()
        .filter(|&pos| matches!(board.get_cell(pos), Ok(Cell::Hidden(_))))
        .count();

    stats
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
        let board_size = (width * height) as usize;

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

            // Effectiveness stats (50 iterations)
            let mut aggregate = AggregateStats::default();
            for _ in 0..50 {
                let mut board = Board::new(width, height, mines).unwrap();
                let game_stats = solve_single_game(&mut board, *solver);
                aggregate.games.push(game_stats);
            }

            println!("\n{} on {}x{} board:", name, width, height);
            println!("Success rate: {:.1}%", aggregate.success_rate());
            println!(
                "Average board completion: {:.1}%",
                aggregate.average_completion(board_size)
            );
            println!("Average moves per game: {:.1}", aggregate.average_moves());
            println!("Total mine hits: {}", aggregate.total_mines_hit());
            println!("Total safe moves: {}", aggregate.total_safe_moves());
            println!("Games played: {}", aggregate.games_played());
        }
    }

    group.finish();
}

fn create_solver_chains() -> Vec<(Box<dyn Solver>, &'static str)> {
    vec![
        // SINGLE //
        // Counting
        (
            Box::new(ChainSolver::new(vec![Box::new(CountingSolver)])),
            "Counting Only",
        ),
        // Matrix
        (
            Box::new(ChainSolver::new(vec![Box::new(MatrixSolver)])),
            "Matrix Only",
        ),
        // Probabilistic
        // (
        //     Box::new(ChainSolver::new(vec![Box::new(ProbabilisticSolver {
        //         min_confidence: 0.95,
        //     })])),
        //     "Probabilistic Only",
        // ),
        // CHAINS //
        // Counting + Matrix
        (
            Box::new(ChainSolver::new(vec![
                Box::new(CountingSolver),
                Box::new(MatrixSolver),
            ])),
            "Counting + Matrix",
        ),
        // Matrix + Probabilistic
        (
            Box::new(ChainSolver::new(vec![
                Box::new(CountingSolver),
                Box::new(ProbabilisticSolver {
                    min_confidence: 0.95,
                }),
            ])),
            "Counting + Probabilistic",
        ),
        // FULL CHAIN //
        // (
        //     Box::new(ChainSolver::new(vec![
        //         Box::new(CountingSolver),
        //         Box::new(MatrixSolver),
        //         Box::new(ProbabilisticSolver {
        //             min_confidence: 0.95,
        //         }),
        //     ])),
        //     "Full Chain",
        // ),
    ]
}

fn benchmark_solver_chains(c: &mut Criterion) {
    let mut group = c.benchmark_group("Solver Chains");

    if std::env::var("QUICK_BENCH").is_ok() {
        group
            .warm_up_time(Duration::from_millis(100))
            .measurement_time(Duration::from_secs(1))
            .sample_size(50);
    }

    let test_configs = vec![
        (8, 8, 10),   // Beginner
        (16, 16, 40), // Intermediate
        (30, 16, 99), // Expert
    ];

    let solver_chains = create_solver_chains();

    for (width, height, mines) in test_configs {
        let board_size = (width * height) as usize;

        for (chain, name) in &solver_chains {
            let config_name = format!("{} {}x{}", name, width, height);

            // Performance benchmark
            group.bench_function(&config_name, |b| {
                b.iter_with_setup(
                    || Board::new(width, height, mines).unwrap(),
                    |mut board| {
                        let stats = solve_single_game(&mut board, chain.as_ref());
                        criterion::black_box(stats)
                    },
                );
            });

            // Effectiveness stats (100 iterations)
            let mut aggregate = AggregateStats::default();
            for _ in 0..100 {
                let mut board = Board::new(width, height, mines).unwrap();
                let game_stats = solve_single_game(&mut board, chain.as_ref());
                aggregate.games.push(game_stats);
            }

            println!("\n{} results:", config_name);
            println!("Success rate: {:.1}%", aggregate.success_rate());
            println!(
                "Average board completion: {:.1}%",
                aggregate.average_completion(board_size)
            );
            println!("Average moves per game: {:.1}", aggregate.average_moves());
            println!("Total safe moves: {}", aggregate.total_safe_moves());
            println!("Games played: {}", aggregate.games_played());

            // Note: Detailed timing analysis would need to be handled separately
            // as Criterion doesn't expose raw timing data in this way
        }
    }

    group.finish();
}

criterion_group!(benches, benchmark_solver_chains); //, benchmark_solvers);
criterion_main!(benches);
