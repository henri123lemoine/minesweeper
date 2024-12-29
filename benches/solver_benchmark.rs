use criterion::{criterion_group, criterion_main, Criterion};
use minesweeper::{
    solver::{
        CountingSolver, MatrixSolver, ProbabilisticSolver, Solver, SolverAction, SolverBoard,
    },
    Board, Cell, Position,
};

#[derive(Debug)]
struct SolverStats {
    moves_made: usize,
    mines_hit: usize,
    cells_cleared: usize,
}

fn solve_board(board: &mut Board, solver: &dyn Solver) -> SolverStats {
    let mut stats = SolverStats {
        moves_made: 0,
        mines_hit: 0,
        cells_cleared: 0,
    };

    // Just make a single move at (1,1) to start
    let _ = board.reveal(Position::new(1, 1));

    // Try at most 100 solver moves to prevent infinite loops
    for i in 0..100 {
        // println!("Solver iteration {}", i);
        let solver_board = SolverBoard::new(board);
        let result = solver.solve(&solver_board);
        // println!("Solver suggested {} moves", result.actions.len());

        if result.actions.is_empty() {
            break;
        }

        stats.moves_made += 1;

        // Only try the first suggested action
        if let Some(action) = result.actions.first() {
            match action {
                SolverAction::Reveal(pos) => {
                    if let Ok(Cell::Hidden(is_mine)) = board.get_cell(*pos) {
                        if *is_mine {
                            stats.mines_hit += 1;
                        } else {
                            stats.cells_cleared += 1;
                        }
                    }
                    let _ = board.reveal(*pos);
                }
                SolverAction::Flag(pos) => {
                    let _ = board.toggle_flag(*pos);
                }
            }
        }
    }

    stats
}

fn benchmark_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("Solvers");

    // Just test one small board size initially
    let width = 8;
    let height = 8;
    let mines = 10;

    let solvers: Vec<(&dyn Solver, &str)> = vec![
        (&CountingSolver, "Counting"),
        (
            &ProbabilisticSolver {
                min_confidence: 0.95,
            },
            "Probabilistic",
        ),
    ];

    for (solver, name) in &solvers {
        group.bench_function(format!("{} {}x{}", name, width, height), |b| {
            b.iter_with_setup(
                || Board::new(width, height, mines).unwrap(),
                |mut board| {
                    let stats = solve_board(&mut board, *solver);
                    criterion::black_box(stats)
                },
            );
        });

        // Run once and print stats
        let mut board = Board::new(width, height, mines).unwrap();
        let stats = solve_board(&mut board, *solver);
        // println!(
        //     "{} on {}x{} board: {} moves, {} mines hit, {} cells cleared",
        //     name, width, height, stats.moves_made, stats.mines_hit, stats.cells_cleared
        // );
    }

    group.finish();
}

criterion_group!(benches, benchmark_solvers);
criterion_main!(benches);
