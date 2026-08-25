//! -------------------------------------------------------------------------
//! Filename:    src/world/side/sim.rs
//! Class:       SemantikaEngine
//! Purpose:     шаг бокового мира: сенсоры → мозг → тело → события
//! Description: SideWorld связывает комнаты/физику с органными мозгами.
//!              Плоды зреют на балконе-кроне и ПАДАЮТ вниз через проём;
//!              мяч-игрушка пинается; одиночество давит; смерть оставляет
//!              надгробие и геном в graveyard/.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use super::{resolve_room, scene, Body, Portal, Room, EAT_DIST, GRAVITY};
use crate::body::life::OrganSim;
use crate::engine::instincts::Instinct;
use crate::rng::XorShift;
use crate::world::ecosystem::Taste;

/// Тик мира (с): 25 мс.
const DT: f64 = 0.025;
/// Плодов в мире одновременно.
pub const N_FRUITS: usize = 6;
/// Тиков созревания плода на ветке до отрыва.
pub const RIPEN_TICKS: usize = 240;

/// Плод: зреет на ветке → падает → лежит → съеден.
#[derive(Clone, Copy, Debug)]
pub struct Fruit {
    pub body: Body,
    pub taste: Taste,
    /// None = лежит/летит; Some(t) = тик отрыва.
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
    pub awake_ticks: usize,
    pub sleeping_ticks: usize,
    /// Одиночество 0..100: копится без соседей, лечится компанией.
    pub lonely: f32,
    /// Темперамент: множитель драйва (смелый >1, трус <1).
    pub tempo: f32,
    /// Скука за погоню за той же целью.
    pub boredom: f32,
    /// Тики страха (замер после шлепка).
    pub fear_ticks: usize,
    /// Тики разворота.
    pub turn_pause: usize,
    /// Случайная прогулка без причины.
    pub stroll: usize,
}

/// Игрушка-мяч: пружинит, пинается существами.
pub struct Ball {
    pub body: Body,
}

/// Надгробие: память о жизни.
pub struct Grave {
    pub x: f64,
    pub y: f64,
    pub id: usize,
    pub meals: usize,
}

pub struct SideWorld {
    pub ball: Ball,
    pub graves: Vec<Grave>,
    /// Кто пинал последним и сколько подряд разных существ.
    pub last_kicker: Option<usize>,
    pub pass_streak: usize,
    /// Сколько тиков мяч лежит на земле.
    ball_rest: usize,
    pub rooms: Vec<Room>,
    pub portals: Vec<Portal>,
    pub creatures: Vec<SideCreature>,
    pub fruits: Vec<Fruit>,
    pub tick: usize,
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
                lonely: 0.0,
                tempo: if i % 2 == 0 { 1.15 } else { 0.75 },
                boredom: 0.0,
                fear_ticks: 0,
                turn_pause: 0,
                stroll: 0,
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
            ball: Ball { body: Body::new(0, 200.0, 100.0) },
            graves: Vec::new(),
            last_kicker: None,
            pass_streak: 0,
            ball_rest: 0,
            rooms,
            portals,
            creatures,
            fruits,
            tick: 0,
            rng,
        }
    }

    pub fn alive_count(&self) -> usize {
        self.creatures.len()
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
            f.body.vy = -60.0;
        }
    }

    /// Погладить: прямое подкрепление игрока (+).
    pub fn pet(&mut self, id: usize) {
        if let Some(c) = self.creatures.iter_mut().find(|c| c.id == id) {
            c.sim.organs.cortex.reward_now(1.0);
        }
    }

    /// Шлёпнуть: прямое наказание игрока (−).
    /// Дамп всех мозгов в нативный .taas для ConnScope.
    pub fn dump_brains(&self) -> std::io::Result<Vec<std::path::PathBuf>> {
        let dir = std::path::Path::new("brains");
        std::fs::create_dir_all(dir)?;
        let mut out = Vec::new();
        for c in &self.creatures {
            let path = dir.join(format!("brain_{:03}_t{:06}.taas", c.id, self.tick));
            let f = std::fs::File::create(&path)?;
            let mut w = std::io::BufWriter::new(f);
            c.sim.organs.cortex.save_baked(&mut w)?;
            out.push(path);
        }
        Ok(out)
    }

    pub fn smack(&mut self, id: usize) {
        if let Some(c) = self.creatures.iter_mut().find(|c| c.id == id) {
            c.sim.organs.cortex.reward_now(-1.0);
        }
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
                Some(t) if self.tick >= t => f.ripen_until = None,
                Some(_) => {}
                None => f.body.step_fall(&self.rooms, &self.portals, DT),
            }
        }
        // Пересев: раз в 300 тиков новый плод на ветке.
        if self.tick % 300 == 0 {
            if let Some(f) = self.fruits.iter_mut().find(|f| !f.alive) {
                *f = Fruit {
                    body: Body::new(1, 392.0 + self.rng.unit() * 80.0, 40.0),
                    taste: if self.rng.unit() < 0.25 { Taste::Bitter } else { Taste::Sweet },
                    ripen_until: Some(self.tick + RIPEN_TICKS),
                    alive: true,
                };
            }
        }

        // ── Мяч: гравитация помягче + упругие отскоки. ──
        {
            let b = &mut self.ball.body;
            b.vy += GRAVITY * 0.6 * DT;
            b.x += b.vx * DT;
            b.y += b.vy * DT;
            let r = &self.rooms[b.room];
            if b.y > r.y_bot - 5.0 {
                b.y = r.y_bot - 5.0;
                b.vy = -b.vy * 0.75;
            }
            if b.y < r.y_top + 5.0 {
                b.y = r.y_top + 5.0;
                b.vy = -b.vy * 0.75;
            }
            if b.x < r.x0 + 5.0 || b.x > r.x1 - 5.0 {
                b.vx = -b.vx * 0.8;
                b.x = b.x.clamp(r.x0 + 5.0, r.x1 - 5.0);
            }
            b.room = resolve_room(&self.rooms, &self.portals, b.room, &mut b.x, &mut b.y, b.vy);
            let resting = b.vy.abs() < 1.0 && b.y >= r.y_bot - 6.0;
            if resting {
                self.ball_rest += 1;
                if self.ball_rest > 60 {
                    self.pass_streak = 0;
                    self.last_kicker = None;
                }
            } else {
                self.ball_rest = 0;
            }
        }

        // ── Существа: сон → сенсоры → мозг → тело → еда. ──
        let ball_pos = (self.ball.body.x, self.ball.body.y);
        let prev_kicker = self.last_kicker;
        let prev_streak = self.pass_streak;
        let others: Vec<(usize, f64)> =
            self.creatures.iter().map(|o| (o.id, o.body.x)).collect();
        let mut kicks: Vec<(usize, f64)> = Vec::new();
        let mut dying: Vec<(usize, usize, f64, f64)> = Vec::new(); // id, meals, x, y

        for ci in 0..self.creatures.len() {
            let c = &mut self.creatures[ci];

            // Сон: мозг работает, веса возвращаются к снимку.
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

            // Нос: сладкий и горький следы.
            let (sl, sr) = smell(&c.body, &self.fruits, Taste::Sweet);
            let (bl, br) = smell(&c.body, &self.fruits, Taste::Bitter);

            let obs = c.sim.brain_step_sensed_dual(sl, sr, bl, br);
            let ml = obs.cortex_regions[3] as f64;
            let mr = obs.cortex_regions[4] as f64;

            // Пинок мяча: подошёл — толкнул, это весело (+награда).
            {
                let dx = ball_pos.0 - c.body.x;
                let dy = ball_pos.1 - c.body.y;
                if dx.abs() < 14.0 && dy.abs() < 16.0 {
                    kicks.push((ci, c.body.facing * 120.0 + c.body.vx * 0.5));
                    c.sim.organs.cortex.schedule_reward(0.3);
                    // Совместная игра слаще: пас от другого существа — бонус.
                    if prev_kicker.map_or(false, |k| k != c.id) && prev_streak >= 1 {
                        c.sim.organs.cortex.schedule_reward(0.4);
                    }
                }
            }

            // Одиночество: компания лечит, пустота давит.
            let crowd = others
                .iter()
                .any(|&(oid, ox)| oid != c.id && (ox - c.body.x).abs() < 70.0);
            if crowd {
                c.lonely = (c.lonely - 2.0).max(0.0);
            } else {
                c.lonely = (c.lonely + 1.0).min(100.0);
            }
            if c.lonely > 80.0 && self.tick % 50 == 0 {
                c.sim.organs.cortex.schedule_reward(-0.2);
            }

            // Ориентирный рефлекс: сильнейшая ноздря поворачивает.
            let s_l = sl + bl * 0.25;
            let s_r = sr + br * 0.25;
            if s_l + s_r > 0.04 {
                c.body.facing = if s_r >= s_l { 1.0 } else { -1.0 };
            }

            // ── ВНИМАНИЕ И СКУКА: цель надоедает. ──
            let chasing = s_l + s_r > 0.04;
            if chasing {
                c.boredom = (c.boredom + 1.0).min(120.0);
                if c.boredom > 90.0 && self.rng.unit() < 0.01 {
                    c.boredom = 0.0;
                    c.stroll = 60; // «пойду посмотрю другое»
                    c.body.facing = -c.body.facing;
                }
            } else {
                c.boredom = (c.boredom - 0.5).max(0.0);
            }

            // ── ПОРЫВЫ БЕЗ ПРИЧИНЫ: жизнь без запаха не молчит. ──
            if !chasing && c.fear_ticks == 0 {
                if c.stroll > 0 {
                    c.stroll -= 1;
                } else {
                    let r = self.rng.unit();
                    if r < 0.004 {
                        c.stroll = 30 + (self.rng.unit() * 50.0) as usize; // прогуляться
                        c.body.facing = if self.rng.unit() < 0.5 { 1.0 } else { -1.0 };
                    } else if r < 0.005 && c.body.on_ground {
                        c.body.vy = -300.0; // подпрыгнул просто так
                    }
                }
            }

            // Страх: замер (апатия движения), потом отпускает.
            if c.fear_ticks > 0 {
                c.fear_ticks -= 1;
            }
            let afraid = c.fear_ticks > 25;

            // Моторы → скорость; синхронный залп ног = прыжок.
            let mut drive = ((ml + mr) / 24.0).clamp(0.0, 1.0);
            let jump = ((ml.min(mr)) / 16.0).clamp(0.0, 1.0);

            // Характер и состояние красят движение:
            // прогулка без запаха, апатия одиночества, страх, темперамент.
            if !chasing && c.stroll > 0 {
                drive = drive.max(0.45); // идёт смотреть мир
            }
            drive *= c.tempo as f64 * (1.0 - c.lonely as f64 / 250.0);
            if afraid {
                drive *= 0.15;
            }
            if c.turn_pause > 0 {
                c.turn_pause -= 1;
                drive *= 0.2; // разворот на месте
            }
            let prev_facing = c.body.facing;
            c.body.step(&self.rooms, &self.portals, drive.min(1.0), jump, DT);
            if (c.body.facing - prev_facing).abs() > 0.5 && c.turn_pause == 0 {
                c.turn_pause = 6; // развернулся — секунду соображает
            }

            // Еда при контакте.
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
                    if f.taste == Taste::Sweet && c.body.on_ground {
                        c.body.vy = -140.0; // радостный подскок
                        c.boredom = 0.0;
                    }
                    match f.taste {
                        Taste::Sweet => {
                            c.sim.blood.ingest_food(90.0);
                            c.sim.organs.cortex.schedule_reward(1.0);
                            if let Some(m) = &mut c.sim.mind {
                                m.reward(1.0);
                            }
                        }
                        Taste::Bitter => {
                            let g = &mut c.sim.blood.conc[crate::body::metabolism::chem::GLUCOSE];
                            *g = (*g - 25.0).max(-20.0);
                            c.sim.organs.cortex.schedule_reward(-3.0);
                            if let Some(m) = &mut c.sim.mind {
                                m.reward(-2.0);
                            }
                            c.bitter_eaten += 1;
                        }
                    }
                }
            }

            // Смерть от истощения: тело уходит, геном остаётся.
            let g = c.sim.blood.conc[crate::body::metabolism::chem::GLUCOSE];
            let dmg = c.sim.blood.organ.damage;
            if g <= -20.0 || dmg >= 1.0 {
                dying.push((c.id, c.meals, c.body.x, c.body.y));
            }
        }

        // Мёртвые: надгробие в мире + геном в graveyard/.
        if !dying.is_empty() {
            let dir = std::path::Path::new("graveyard");
            let _ = std::fs::create_dir_all(dir);
            let (kept, gone): (Vec<SideCreature>, Vec<SideCreature>) = self
                .creatures
                .drain(..)
                .partition(|c| !dying.iter().any(|(id, ..)| *id == c.id));
            self.creatures = kept;
            for c in gone {
                let (_, meals, x, y) =
                    dying.iter().find(|(id, ..)| *id == c.id).copied().unwrap();
                self.graves.push(Grave { x, y: y - 6.0, id: c.id, meals });
                let path = dir.join(format!("creature_{:04}_t{:06}.bin.taas", c.id, self.tick));
                let f = std::fs::File::create(&path).expect("grave create");
                let mut w = std::io::BufWriter::new(f);
                let _ = c.sim.organs.cortex.save_baked(&mut w);
            }
            dying.clear();
        }

        // Серия пасов: разные существа подряд — серия растёт.
        if let Some(&(ci, _)) = kicks.first() {
            if let Some(c) = self.creatures.get(ci) {
                if prev_kicker == Some(c.id) {
                    self.pass_streak = 0; // сам себе пас — не считается
                } else {
                    self.pass_streak = prev_streak + 1;
                }
                self.last_kicker = Some(c.id);
            }
        }

        // Пинки применяем после прохода (мяч один на всех).
        for &(ci, kick) in &kicks {
            if self.creatures.get(ci).is_some() {
                let b = &mut self.ball.body;
                b.vx = kick;
                b.vy = -140.0;
            }
        }
    }
}

/// Запах направления к ближайшему плоду нужного вкуса (сквозь комнаты).
fn smell(body: &Body, fruits: &[Fruit], want: Taste) -> (f64, f64) {
    let mut best: Option<(f64, f64, f64)> = None;
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
    let h = dx / dist; // −1 слева … +1 справа
    // (левая ноздря, правая ноздря)
    ((-h.min(0.0)).max(0.0) * prox, h.max(0.0) * prox)
}

/// Те же врождённые рефлексы, что в экосистеме (+ отвращение).
pub fn innate_instincts() -> Vec<Instinct> {
    crate::world::ecosystem::innate_instincts()
}
