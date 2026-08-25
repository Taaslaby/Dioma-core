//! -------------------------------------------------------------------------
//! Filename:    src/world/side/sim.rs
//! Class:       SemantikaEngine
//! Purpose:     шаг бокового мира: сенсоры → мозг → тело → события
//! Description: SideWorld связывает комнаты/физику с органными мозгами.
//!              Плоды зреют на балконе-кроне и ПАДАЮТ вниз через портал:
//!              гравитация сама создаёт задачу «ждать под кроной или идти».
//! author: Atlasov anton
// -------------------------------------------------------------------------
use super::{scene, Body, Portal, Room, EAT_DIST};
use crate::body::life::OrganSim;
use crate::engine::instincts::Instinct;
use crate::world::ecosystem::Taste;
use crate::rng::XorShift;

/// Тик мира (мс): как в экосистеме, 25 мс.
const DT: f64 = 0.025;
/// Плодов зреет одновременно.
pub const N_FRUITS: usize = 6;
/// Тиков созревания плода на ветке до отрыва.
pub const RIPEN_TICKS: usize = 240;

/// Плод: зреет на ветке → падает → лежит → съеден.
#[derive(Clone, Copy, Debug)]
pub struct Fruit {
    pub body: Body,
    pub taste: Taste,
    /// None = лежит/летит; Some(t) = тик созревания на ветке.
    pub ripen_until: Option<usize>,
    pub alive: bool,
}

/// Существо бокового мира.
pub struct SideCreature {
    pub id: usize,
    pub sim: OrganSim,
    pub body: Body,
    pub meals: usize,
    pub bitter_eaten: usize,
    pub born_tick: usize,
    /// Тиков бодрствования подряд.
    pub awake_ticks: usize,
    /// Осталось тиков сна (>0 = спит).
    pub sleeping_ticks: usize,
}

/// Боковой мир.
pub struct SideWorld {
    pub rooms: Vec<Room>,
    pub portals: Vec<Portal>,
    pub creatures: Vec<SideCreature>,
    pub fruits: Vec<Fruit>,
    pub tick: usize,
    #[allow(dead_code)]
    next_id: usize,
    rng: XorShift,
}

impl SideWorld {
    /// Мир с `n` основателями; инстинкты + родильный сон как в экосистеме.
    pub fn new(n: usize, base_seed: u64) -> Self {
        let (rooms, portals) = scene();
        let mut creatures = Vec::new();
        for i in 0..n {
            let seed = base_seed.wrapping_add(i as u64 * 7919);
            let mut sim = OrganSim::new(false, seed).with_mind(seed ^ 0x5EED);
            sim.instincts = innate_instincts();
            sim.birth_sleep();
            // Основатели стоят по земле в ряд.
            let x = 260.0 + i as f64 * 40.0;
            creatures.push(SideCreature {
                id: i,
                sim,
                body: Body::new(0, x, 232.0),
                meals: 0,
                bitter_eaten: 0,
                born_tick: 0,
                awake_ticks: 0,
                sleeping_ticks: 0,
            });
        }
        let mut rng = XorShift::new(base_seed);
        let fruits = (0..N_FRUITS)
            .map(|i| Fruit {
                body: Body::new(1, 395.0 + i as f64 * 20.0, 42.0),
                taste: if i % 4 == 3 { Taste::Bitter } else { Taste::Sweet },
                ripen_until: Some((rng.unit() * RIPEN_TICKS as f64 * 2.0) as usize),
                alive: true,
            })
            .collect();
        Self {
            rooms,
            portals,
            creatures,
            fruits,
            tick: 0,
            next_id: n,
            rng,
        }
    }

    /// Игрок бросил плод (сразу спелый).
    pub fn drop_fruit(&mut self, x: f64, y: f64, bitter: bool) {
        let taste = if bitter { Taste::Bitter } else { Taste::Sweet };
        let room = self
            .rooms
            .iter()
            .find(|r| r.contains(x, y))
            .map(|r| r.id)
            .unwrap_or(0);
        let slot = match self.fruits.iter_mut().find(|f| !f.alive) {
            Some(f) => Some(f),
            None => self.fruits.first_mut(),
        };
        if let Some(f) = slot {
            *f = Fruit {
                body: Body::new(room, x.clamp(5.0, 515.0), y),
                taste,
                ripen_until: None,
                alive: true,
            };
            f.body.vy = -60.0; // лёгкий подброс от «руки»
        }
    }

    pub fn alive_count(&self) -> usize {
        self.creatures.len()
    }

    /// Один тик мира.
    pub fn step_tick(&mut self) {
        self.tick += 1;

        // ── Фрукты: зреют → падают → лежат. ──
        for f in &mut self.fruits {
            if !f.alive {
                continue;
            }
            match f.ripen_until {
                Some(t) if self.tick >= t => f.ripen_until = None, // отрыв
                Some(_) => {}
                None => f.body.step_fall(&self.rooms, &self.portals, DT),
            }
        }
        // Пересев съеденного: раз в N тиков новый плод на ветке.
        if self.tick % 300 == 0 {
            let slot = self.fruits.iter_mut().find(|f| !f.alive);
            if let Some(f) = slot {
                *f = Fruit {
                    body: Body::new(
                        1,
                        392.0 + self.rng.unit() * 80.0,
                        40.0,
                    ),
                    taste: if self.rng.unit() < 0.25 { Taste::Bitter } else { Taste::Sweet },
                    ripen_until: Some(self.tick + RIPEN_TICKS),
                    alive: true,
                };
            }
        }

        // ── Существа: сон → сенсоры → мозг → тело → еда. ──
        for c in &mut self.creatures {
            if c.sleeping_ticks > 0 {
                c.sleeping_ticks -= 1;
                let _ = c.sim.brain_step();
                if c.sleeping_ticks.is_multiple_of(4) {
                    c.sim.organs.cortex.memory_recall(0.16);
                }
                continue;
            }
            c.awake_ticks += 1;
            if c.awake_ticks >= 500 {
                c.awake_ticks = 0;
                c.sleeping_ticks = 40;
                c.sim.organs.cortex.memory_snapshot();
            }

            // Обоняние: ближайший сладкий/горький в комнате тела.
            let (sl, sr) = smell(&c.body, &self.fruits, c.body.room, Taste::Sweet);
            let (bl, br) = smell(&c.body, &self.fruits, c.body.room, Taste::Bitter);

            let obs = c.sim.brain_step_sensed_dual(sl, sr, bl, br);
            let ml = obs.cortex_regions[3] as f64;
            let mr = obs.cortex_regions[4] as f64;

            // Ориентирный рефлекс: ноздря с более сильным запахом задаёт
            // направление (горький привлекает слабее — интерес без жадности).
            let s_l = sl + bl * 0.25;
            let s_r = sr + br * 0.25;
            if s_l + s_r > 0.04 {
                c.body.facing = if s_r >= s_l { 1.0 } else { -1.0 };
            }
            // Моторы → скорость; синхронный залп обеих ног = прыжок.
            let drive = ((ml + mr) / 24.0).clamp(0.0, 1.0);
            let jump = ((ml.min(mr)) / 16.0).clamp(0.0, 1.0);
            c.body.step(&self.rooms, &self.portals, drive, jump, DT);

            // Еда при контакте (только упавшие/лежащие).
            for f in &mut self.fruits {
                if !f.alive || f.ripen_until.is_some() {
                    continue;
                }
                if f.body.room == c.body.room
                    && (f.body.x - c.body.x).abs() < EAT_DIST
                    && (f.body.y - c.body.y).abs() < EAT_DIST * 1.5
                {
                    f.alive = false;
                    c.meals += 1;
                    match f.taste {
                        Taste::Sweet => {
                            c.sim.blood.ingest_food(90.0);
                            c.sim.organs.cortex.schedule_reward(1.0);
                            if let Some(m) = &mut c.sim.mind {
                                m.reward(1.0);
                            }
                        }
                        Taste::Bitter => {
                            let bl2 = &mut c.sim.blood;
                            bl2.conc[crate::body::metabolism::chem::GLUCOSE] =
                                (bl2.conc[crate::body::metabolism::chem::GLUCOSE] - 25.0)
                                    .max(-20.0);
                            c.sim.organs.cortex.schedule_reward(-3.0);
                            if let Some(m) = &mut c.sim.mind {
                                m.reward(-2.0);
                            }
                            c.bitter_eaten += 1;
                        }
                    }
                }
            }
        }
    }
}

/// Запах направления к ближайшему плоду нужного вкуса (в той же комнате).
fn smell(body: &Body, fruits: &[Fruit], _room: usize, want: Taste) -> (f64, f64) {
    let mut best: Option<(f64, f64, f64)> = None; // (dist², x, y)
    for f in fruits {
        if !f.alive || f.taste != want || f.ripen_until.is_some() {
            continue;
        }
        let d = (f.body.x - body.x).powi(2) + (f.body.y - body.y).powi(2);
        if best.map_or(true, |(bd, _, _)| d < bd) {
            best = Some((d, f.body.x, f.body.y));
        }
    }
    let Some((_, fx, fy)) = best else {
        return (0.0, 0.0);
    };
    let dx = fx - body.x;
    let dy = fy - body.y;
    let dist = (dx * dx + dy * dy).sqrt().max(1e-3);
    let prox = 60.0 / (60.0 + dist);
    // В боковом мире запах направленный по горизонтали + вертикальная нота.
    let h = dx / dist; // −1 слева … +1 справа
    // (левая ноздря, правая ноздря)
    ((-h.min(0.0)).max(0.0) * prox, h.max(0.0) * prox)
}

/// Те же врождённые рефлексы, что в экосистеме (+ отвращение).
pub fn innate_instincts() -> Vec<Instinct> {
    crate::world::ecosystem::innate_instincts()
}
