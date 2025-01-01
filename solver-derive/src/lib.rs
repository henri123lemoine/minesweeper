use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

/// Derives the standard test suite for a Minesweeper solver implementation.
///
/// This macro generates:
/// 1. Basic solver validation tests
/// 2. Benchmark integration
/// 3. Property-based tests for solver correctness
///
/// # Example
/// ```rust
/// use minesweeper_solver_derive::SolverTest;
///
/// #[derive(SolverTest)]
/// struct MySolver;
///
/// impl Solver for MySolver {
///     // ... solver implementation
/// }
/// ```
#[proc_macro_derive(SolverTest)]
pub fn derive_solver_test(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // Generate the test module name
    let test_mod_name = syn::Ident::new(
        &format!("{}_tests", name.to_string().to_lowercase()),
        proc_macro2::Span::call_site(),
    );

    // Generate implementation
    let expanded = quote! {
        #[cfg(test)]
        mod #test_mod_name {
            use super::*;
            use crate::{Board, Cell, Position, Solver, SolverBoard};
            use std::collections::HashSet;

            /// Test basic solver functionality
            #[test]
            fn test_basic_solve() {
                let mut board = Board::new(3, 3, 1).unwrap();
                let solver = #name::default();

                // Place a mine in a known position
                board.cells.insert(Position::new(0, 0), Cell::Hidden(true));

                // Reveal center square
                board.reveal(Position::new(1, 1)).unwrap();

                let solver_board = SolverBoard::new(&board);
                let result = solver.solve(&solver_board);

                assert!(!result.actions.is_empty(), "Solver should produce at least one action");
            }

            /// Test solver determinism properties
            #[test]
            fn test_determinism_consistency() {
                let solver = #name::default();
                assert_eq!(
                    solver.is_deterministic(),
                    !matches!(solver.solve(&SolverBoard::new(&Board::new(3, 3, 1).unwrap())).certainty,
                             crate::Certainty::Probabilistic(_)),
                    "Determinism flag must match solver behavior"
                );
            }

            /// Test solver name consistency
            #[test]
            fn test_solver_name() {
                let solver = #name::default();
                assert!(!solver.name().is_empty(), "Solver name must not be empty");
            }

            /// Test solver behavior on edge cases
            #[test]
            fn test_edge_cases() {
                let solver = #name::default();

                // Empty board
                let empty_board = Board::new(1, 1, 0).unwrap();
                let result = solver.solve(&SolverBoard::new(&empty_board));
                assert!(result.actions.len() <= 1, "Empty board should produce at most one action");

                // Full board
                let mut full_board = Board::new(2, 2, 0).unwrap();
                for x in 0..2 {
                    for y in 0..2 {
                        let _ = full_board.reveal(Position::new(x, y));
                    }
                }
                let result = solver.solve(&SolverBoard::new(&full_board));
                assert!(result.actions.is_empty(), "Fully revealed board should produce no actions");
            }

            /// Test that solver actions are valid
            #[test]
            fn test_action_validity() {
                let solver = #name::default();
                let board = Board::new(5, 5, 5).unwrap();
                let solver_board = SolverBoard::new(&board);
                let result = solver.solve(&solver_board);

                let (width, height) = board.dimensions();
                for action in result.actions {
                    match action {
                        crate::SolverAction::Reveal(pos) | crate::SolverAction::Flag(pos) => {
                            assert!(
                                pos.x >= 0 && pos.x < width as i32 && pos.y >= 0 && pos.y < height as i32,
                                "Solver action must be within board bounds"
                            );
                        }
                    }
                }
            }

            /// Test solver performance guarantees
            #[test]
            fn test_performance_constraints() {
                use std::time::Instant;

                let solver = #name::default();
                let sizes = [(8, 8, 10), (16, 16, 40)];

                for (width, height, mines) in sizes {
                    let board = Board::new(width, height, mines).unwrap();
                    let solver_board = SolverBoard::new(&board);

                    let start = Instant::now();
                    let _ = solver.solve(&solver_board);
                    let duration = start.elapsed();

                    // Basic performance check - should complete within reasonable time
                    assert!(duration.as_secs() < 1, "Solver took too long for {}x{} board", width, height);
                }
            }

            /// Integration with criterion benchmarks
            #[cfg(test)]
            pub fn generate_benchmark(c: &mut criterion::Criterion) {
                let solver = #name::default();
                let board = Board::new(16, 16, 40).unwrap();
                let solver_board = SolverBoard::new(&board);

                c.bench_function(
                    &format!("{}_solve_intermediate", solver.name()),
                    |b| b.iter(|| solver.solve(&solver_board))
                );
            }
        }
    };

    // Return the generated implementation
    TokenStream::from(expanded)
}
