#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

impl Position {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    pub fn neighbors(&self) -> impl Iterator<Item = Position> + '_ {
        (-1..=1).flat_map(move |dy| {
            (-1..=1).filter_map(move |dx| {
                if dx == 0 && dy == 0 {
                    None
                } else {
                    Some(Position::new(self.x + dx, self.y + dy))
                }
            })
        })
    }
}
