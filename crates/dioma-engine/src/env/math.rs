//! Плоская векторная математика и идентификатор комнаты (порт a1-core).
//!
//! Нужны карте мира (env::map) — аналог Vector2D / RoomId из Creatures 2e.

use std::fmt;
use std::ops::{Add, AddAssign, Mul, Sub};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    pub fn new(x: f32, y: f32) -> Self {
        Vec2 { x, y }
    }

    pub fn len(self) -> f32 {
        self.x.hypot(self.y)
    }

    pub fn dist(self, other: Vec2) -> f32 {
        (self - other).len()
    }

    /// Единичный вектор в сторону `other`; нулевой вектор остаётся нулевым.
    pub fn dir_to(self, other: Vec2) -> Vec2 {
        let d = other - self;
        let l = d.len();
        if l <= f32::EPSILON {
            Vec2::ZERO
        } else {
            Vec2::new(d.x / l, d.y / l)
        }
    }
}

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x + o.x, self.y + o.y)
    }
}

impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x - o.x, self.y - o.y)
    }
}

impl Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, k: f32) -> Vec2 {
        Vec2::new(self.x * k, self.y * k)
    }
}

impl AddAssign for Vec2 {
    fn add_assign(&mut self, o: Vec2) {
        self.x += o.x;
        self.y += o.y;
    }
}

/// Идентификатор комнаты на карте.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RoomId(pub u32);

impl fmt::Display for RoomId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dist_and_dir() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(3.0, 4.0);
        assert_eq!(a.dist(b), 5.0);
        let d = a.dir_to(b);
        assert!((d.len() - 1.0).abs() < 1e-6);
        assert_eq!(Vec2::ZERO.dir_to(Vec2::ZERO), Vec2::ZERO);
    }
}
