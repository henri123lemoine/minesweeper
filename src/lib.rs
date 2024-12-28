use rand::Rng;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    Hidden(bool), // bool represents whether there's a mine (true = mine)
    Revealed(u8), // u8 represents number of adjacent mines
    Flagged,
}

#[derive(Debug)]
pub struct Board {
    width: usize,
    height: usize,
    cells: Vec<Cell>,
    mines_count: usize,
    game_over: bool,
    revealed_count: usize,
}

impl Board {
    pub fn new(width: usize, height: usize, mines_count: usize) -> Self {
        if mines_count >= width * height {
            panic!("Too many mines for the board size");
        }

        let cells = vec![Cell::Hidden(false); width * height];
        let mut board = Board {
            width,
            height,
            cells,
            mines_count,
            game_over: false,
            revealed_count: 0,
        };
        board.place_mines();
        board
    }

    fn place_mines(&mut self) {
        let mut rng = rand::thread_rng();
        let mut mines_placed = 0;
        let total_cells = self.width * self.height;

        while mines_placed < self.mines_count {
            let pos = rng.gen_range(0..total_cells);
            if let Cell::Hidden(false) = self.cells[pos] {
                self.cells[pos] = Cell::Hidden(true);
                mines_placed += 1;
            }
        }
    }

    pub fn get_cell(&self, x: usize, y: usize) -> Option<&Cell> {
        if x >= self.width || y >= self.height {
            None
        } else {
            Some(&self.cells[y * self.width + x])
        }
    }

    fn count_adjacent_mines(&self, x: usize, y: usize) -> u8 {
        let mut count = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }

                let new_x = x as i32 + dx;
                let new_y = y as i32 + dy;

                if new_x >= 0
                    && new_x < self.width as i32
                    && new_y >= 0
                    && new_y < self.height as i32
                {
                    if let Cell::Hidden(true) | Cell::Flagged =
                        self.cells[(new_y as usize) * self.width + new_x as usize]
                    {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    pub fn reveal(&mut self, x: usize, y: usize) -> bool {
        if self.game_over || x >= self.width || y >= self.height {
            return false;
        }

        let pos = y * self.width + x;
        match self.cells[pos] {
            Cell::Hidden(true) => {
                self.cells[pos] = Cell::Revealed(0);
                self.game_over = true;
                false
            }
            Cell::Hidden(false) => {
                let mines = self.count_adjacent_mines(x, y);
                self.cells[pos] = Cell::Revealed(mines);
                self.revealed_count += 1;

                // If no adjacent mines, reveal neighboring cells
                if mines == 0 {
                    self.reveal_neighbors(x, y);
                }
                true
            }
            _ => false,
        }
    }

    fn reveal_neighbors(&mut self, x: usize, y: usize) {
        let mut to_reveal = HashSet::new();
        to_reveal.insert((x, y));

        while !to_reveal.is_empty() {
            let mut next_batch = HashSet::new();

            for &(cx, cy) in &to_reveal {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let new_x = cx as i32 + dx;
                        let new_y = cy as i32 + dy;

                        if new_x >= 0
                            && new_x < self.width as i32
                            && new_y >= 0
                            && new_y < self.height as i32
                        {
                            let new_x = new_x as usize;
                            let new_y = new_y as usize;
                            let pos = new_y * self.width + new_x;

                            if let Cell::Hidden(false) = self.cells[pos] {
                                let mines = self.count_adjacent_mines(new_x, new_y);
                                self.cells[pos] = Cell::Revealed(mines);
                                self.revealed_count += 1;

                                if mines == 0 {
                                    next_batch.insert((new_x, new_y));
                                }
                            }
                        }
                    }
                }
            }

            to_reveal = next_batch;
        }
    }

    pub fn toggle_flag(&mut self, x: usize, y: usize) -> bool {
        if self.game_over || x >= self.width || y >= self.height {
            return false;
        }

        let pos = y * self.width + x;
        match self.cells[pos] {
            Cell::Hidden(_) => {
                self.cells[pos] = Cell::Flagged;
                true
            }
            Cell::Flagged => {
                self.cells[pos] = Cell::Hidden(false);
                true
            }
            _ => false,
        }
    }

    pub fn is_game_over(&self) -> bool {
        self.game_over
    }

    pub fn is_won(&self) -> bool {
        !self.game_over && self.revealed_count == self.width * self.height - self.mines_count
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}
