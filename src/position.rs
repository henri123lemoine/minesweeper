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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_creation() {
        let pos = Position::new(5, 10);
        assert_eq!(pos.x, 5);
        assert_eq!(pos.y, 10);
    }

    #[test]
    fn test_neighbors() {
        let pos = Position::new(1, 1);
        let neighbors: Vec<Position> = pos.neighbors().collect();

        assert_eq!(neighbors.len(), 8);
        assert!(neighbors.contains(&Position::new(0, 0))); // Top-left
        assert!(neighbors.contains(&Position::new(1, 0))); // Top
        assert!(neighbors.contains(&Position::new(2, 0))); // Top-right
        assert!(neighbors.contains(&Position::new(0, 1))); // Left
        assert!(neighbors.contains(&Position::new(2, 1))); // Right
        assert!(neighbors.contains(&Position::new(0, 2))); // Bottom-left
        assert!(neighbors.contains(&Position::new(1, 2))); // Bottom
        assert!(neighbors.contains(&Position::new(2, 2))); // Bottom-right
    }
}
