//! Карта мира: комнаты-прямоугольники, двери и клеточные автоматы веществ
//! (порт a1-map; аналог RoomCA/CARates из C2e). Тепло, влажность и прочее
//! диффундируют через двери и постепенно распадаются.

use crate::env::math::{RoomId, Vec2};

/// Сколько веществ живёт в воздухе комнат.
pub const SUBSTANCES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Substance {
    Temperature = 0,
    Humidity = 1,
    Light = 2,
    Nutrients = 3,
}

impl Substance {
    pub fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub min: Vec2,
    pub max: Vec2,
}

impl Rect {
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Rect {
            min: Vec2::new(x0.min(x1), y0.min(y1)),
            max: Vec2::new(x0.max(x1), y0.max(y1)),
        }
    }

    pub fn contains(&self, p: Vec2) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    pub fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }
}

#[derive(Debug, Clone)]
pub struct Room {
    pub rect: Rect,
    /// Значения клеточного автомата: [SUBSTANCES].
    pub ca: [f32; SUBSTANCES],
}

/// Проход между двумя комнатами. Проницаемость 0..1 на каждое вещество.
#[derive(Debug, Clone)]
pub struct Door {
    pub a: RoomId,
    pub b: RoomId,
    pub perm: [f32; SUBSTANCES],
}

/// Глобальные коэффициенты переноса/распада.
#[derive(Debug, Clone, Copy)]
pub struct CaRates {
    pub flow: f32,
    /// Доля распада за такт по каждому веществу (0..1).
    pub decay: [f32; SUBSTANCES],
}

impl Default for CaRates {
    fn default() -> Self {
        CaRates {
            flow: 0.25,
            decay: [0.02; SUBSTANCES],
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct WorldMap {
    pub rooms: Vec<Room>,
    pub doors: Vec<Door>,
    pub rates: CaRates,
}

impl WorldMap {
    pub fn new(rates: CaRates) -> Self {
        WorldMap {
            rooms: Vec::new(),
            doors: Vec::new(),
            rates,
        }
    }

    pub fn add_room(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) -> RoomId {
        let id = RoomId(self.rooms.len() as u32);
        self.rooms.push(Room {
            rect: Rect::new(x0, y0, x1, y1),
            ca: [0.0; SUBSTANCES],
        });
        id
    }

    pub fn connect(&mut self, a: RoomId, b: RoomId, perm: [f32; SUBSTANCES]) {
        assert!(a.0 < self.rooms.len() as u32 && b.0 < self.rooms.len() as u32);
        self.doors.push(Door { a, b, perm });
    }

    /// Комната, содержащая точку (линейный поиск — комнат немного).
    pub fn room_at(&self, p: Vec2) -> Option<RoomId> {
        self.rooms
            .iter()
            .position(|r| r.rect.contains(p))
            .map(|i| RoomId(i as u32))
    }

    /// Один такт диффузии и распада.
    pub fn step(&mut self) {
        for d in &self.doors {
            let (a, b) = (d.a.0 as usize, d.b.0 as usize);
            for s in 0..SUBSTANCES {
                let diff = self.rooms[a].ca[s] - self.rooms[b].ca[s];
                let flux = self.rates.flow * d.perm[s] * diff * 0.5;
                self.rooms[a].ca[s] -= flux;
                self.rooms[b].ca[s] += flux;
            }
        }
        for r in &mut self.rooms {
            for s in 0..SUBSTANCES {
                let v = r.ca[s] * (1.0 - self.rates.decay[s]);
                r.ca[s] = if v < 1e-4 { 0.0 } else { v };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heat_equalizes_through_door() {
        // почти без распада, чтобы проверить именно выравнивание
        let mut m = WorldMap::new(CaRates {
            flow: 0.25,
            decay: [0.000_5; SUBSTANCES],
        });
        let a = m.add_room(0.0, 0.0, 10.0, 10.0);
        let b = m.add_room(10.0, 0.0, 20.0, 10.0);
        m.connect(a, b, [1.0; SUBSTANCES]);
        m.rooms[a.0 as usize].ca[Substance::Temperature.idx()] = 100.0;

        for _ in 0..200 {
            m.step();
        }
        let ta = m.rooms[a.0 as usize].ca[Substance::Temperature.idx()];
        let tb = m.rooms[b.0 as usize].ca[Substance::Temperature.idx()];
        // тепло растеклось поровну (и частично распалось)
        assert!((ta - tb).abs() < 0.5, "ta={ta} tb={tb}");
        assert!(ta > 20.0 && ta < 60.0);
    }

    #[test]
    fn isolated_room_decays_to_zero() {
        let mut m = WorldMap::default();
        let a = m.add_room(0.0, 0.0, 5.0, 5.0);
        m.rooms[a.0 as usize].ca[Substance::Nutrients.idx()] = 50.0;
        for _ in 0..2000 {
            m.step();
        }
        assert_eq!(m.rooms[a.0 as usize].ca[Substance::Nutrients.idx()], 0.0);
    }

    #[test]
    fn point_lookup() {
        let mut m = WorldMap::default();
        let a = m.add_room(0.0, 0.0, 10.0, 10.0);
        let b = m.add_room(10.0, 0.0, 20.0, 10.0);
        assert_eq!(m.room_at(Vec2::new(3.0, 4.0)), Some(a));
        assert_eq!(m.room_at(Vec2::new(15.0, 4.0)), Some(b));
        assert_eq!(m.room_at(Vec2::new(-1.0, 4.0)), None);
    }
}
