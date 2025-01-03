# Rust Minesweeper Project

![GitHub](https://img.shields.io/github/license/henri123lemoine/minesweeper)

An implementation of Minesweeper in Rust.

## Usage

Ensure you have the latest version of Rust installed.
Clone the repository by running `git clone https://github.com/henri123lemoine/minesweeper.git`, followed by `cd minesweeper`.

Then:

- Run `cargo build` to build the project.
- Run `cargo test` to run the test suite.
- Run `cargo run` to start the game.

## TODOs

- [x] Basic game structure
  - [x] Board representation
  - [x] Cell struct with state handling
  - [x] Game state management
- [x] Core game mechanics
  - [x] Mine placement
  - [x] Cell revealing logic
  - [x] Flag placement
  - [x] Win/loss detection
- [x] Basic CLI interface
  - [x] Board display
  - [x] User input handling
  - [x] Game state visualization
- [x] Solver implementation
  - [x] Basic solver structure
  - [x] Solver chaining
  - Solvers
    - [x] Counting solver
    - [x] Matrix solver
    - [x] Probability solver
- [ ] Solvable board generation
  - [ ] Board validation system
  - [ ] Generation algorithm
  - [ ] Integration with core game
- [ ] Infinite board implementation
  - [ ] Infinite grid system
  - [ ] Procedural generation
  - [ ] Memory management
- [ ] Solvable infinite boards
- [ ] Bevy implementation
  - [ ] Basic Bevy setup
  - [ ] Game visualization
  - [ ] State management
- [ ] Multiplayer implementation
  - [ ] Network architecture
  - [ ] Multiplayer features
  - [ ] Web deployment

## Acknowledgments

- Rust community for resources and inspiration
- Bevy engine documentation and community
- New Sonnet 3.5
