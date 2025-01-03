#![cfg(feature = "test-utils")]

use minesweeper::{
    solver::test_utils::{
        validate_deterministic_solver, validate_probabilistic_solver, validate_solver_chain,
        TestBoardConfig, TestBoardGenerator,
    },
    CountingSolver, MatrixSolver, SolverChain, TankSolver,
};

#[test]
fn test_counting_solver_extensive() {
    let config = TestBoardConfig {
        width: 16,
        height: 16,
        mine_density: 0.15,
        revealed_percentage: 0.3,
    };
    let mut generator = TestBoardGenerator::with_seed(config, 12345);
    let solver = CountingSolver::default();

    // Generate and test 10,000 boards
    let test_cases = generator.generate_batch(10_000);
    let mut failures = 0;

    for (idx, (board, mine_positions)) in test_cases.iter().enumerate() {
        if !validate_deterministic_solver(&solver, board, mine_positions) {
            println!("Failure on test case {}", idx);
            failures += 1;
        }
    }

    assert_eq!(
        failures, 0,
        "Counting solver failed on {} out of 10,000 test cases",
        failures
    );
}

#[test]
fn test_matrix_solver_extensive() {
    let config = TestBoardConfig {
        width: 16,
        height: 16,
        mine_density: 0.15,
        revealed_percentage: 0.3,
    };
    let mut generator = TestBoardGenerator::with_seed(config, 12345);
    let solver = MatrixSolver::default();

    // Generate and test 10,000 boards
    let test_cases = generator.generate_batch(10_000);
    let mut failures = 0;

    for (idx, (board, mine_positions)) in test_cases.iter().enumerate() {
        if !validate_deterministic_solver(&solver, board, mine_positions) {
            println!("Failure on test case {}", idx);
            failures += 1;
        }
    }

    assert_eq!(
        failures, 0,
        "Matrix solver failed on {} out of 10,000 test cases",
        failures
    );
}

#[test]
fn test_tank_solver_calibration() {
    let config = TestBoardConfig {
        width: 16,
        height: 16,
        mine_density: 0.15,
        revealed_percentage: 0.3,
    };
    let mut generator = TestBoardGenerator::with_seed(config, 12345);
    let solver = TankSolver::default();

    // Generate and test 10,000 boards
    let test_cases = generator.generate_batch(10_000);
    let mut failures = 0;

    for (idx, (board, mine_positions)) in test_cases.iter().enumerate() {
        if !validate_probabilistic_solver(&solver, board, &mine_positions) {
            println!("Failure on test case {}", idx);
            failures += 1;
        }
    }

    assert_eq!(
        failures, 0,
        "Tank solver failed on {} out of 10,000 test cases",
        failures
    );
}

#[test]
fn test_solver_chain_extensive() {
    let config = TestBoardConfig {
        width: 16,
        height: 16,
        mine_density: 0.15,
        revealed_percentage: 0.3,
    };
    let mut generator = TestBoardGenerator::with_seed(config, 12345);
    let chain = SolverChain::new(0.95)
        .add_deterministic(CountingSolver::default())
        .add_deterministic(MatrixSolver::default())
        .add_probabilistic(TankSolver::default());

    // Generate and test 5,000 boards
    let test_cases = generator.generate_batch(5_000);
    let mut failures = 0;

    for (idx, (board, mine_positions)) in test_cases.iter().enumerate() {
        if !validate_solver_chain(&chain, board, &mine_positions) {
            println!("Failure on test case {}", idx);
            failures += 1;
        }
    }

    assert_eq!(
        failures, 0,
        "Solver chain failed on {} out of 5,000 test cases",
        failures
    );
}
