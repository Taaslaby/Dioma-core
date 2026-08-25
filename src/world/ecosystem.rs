//! -------------------------------------------------------------------------
//! Filename:    src/ecosystem.rs
//! Class:       main
//! Purpose:     multi-creature ecosystem — THE GAME LOOP
//! Description: N существ с органными мозгами в общем мире с едой.
//!              Смерть от истощения · спаривание сытых рядом · ребёнок
//!              наследует мозги родителей ± мутация. Игрок кликает чтобы
//!              подбросить еду или принудительно спарить.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use crate::body::life::OrganSim;
use crate::engine::instincts::Instinct;
use crate::heritage::spawn_child_crossover;
use crate::rng::XorShift;
use rayon::prelude::*;

pub const ECO_TICKS: usize = 8000; // 200 с
pub const EAT_RADIUS: f64 = 14.0;
pub const BREED_RADIUS: f64 = 70.0;
/// «Сытый середняк»: основатели стартуют с глюкозой 40 — могут дать
/// первое поколение сразу при встрече.
pub const BREED_GLUCOSE_MIN: f32 = 35.0;
/// Голодная черта: ниже неё копится голодный счётчик.
pub const STARVE_LINE: f32 = 15.0;
/// Сколько суммарных голодных тиков переживается до смерти
/// (~75 с жизни впроголодь). Сытость гасит счётчик вдвое быстрее.
pub const STARVE_DEATH_TICKS: usize = 6000;
/// «В настроении»: столько тиков после еды существо готово к паре
/// (~22 с). Сигнал сытости вместо недостижимого абсолютного порога.
pub const BREED_AFTER_MEAL_TICKS: usize = 900;
/// Максимум существ в экосистеме — без капа размножение экспоненциальное.
pub const MAX_CREATURES: usize = 12;
pub const ARENA_HALF: f64 = 130.0;
/// Дефицит еды: источники восстанавливаются медленно — отбор включается.
pub const N_FOOD: usize = 10;
pub const FOOD_REGROW: usize = 450;
/// Цена движения в этой экосистеме: выше глобальной (0.0015), чтобы
/// эффективность поиска еды напрямую отображалась на выживание.
pub const MOVE_COST_PER_ACTIVITY: f32 = 0.0045;
/// Тиков бодрствования до отхода ко сну (~12.5 с жизни).
pub const SLEEP_EVERY: usize = 500;
/// Длительность сна в тиках (~1 с): консолидация без движения и еды.
pub const SLEEP_FOR: usize = 40;
/// Сила возврата к снимку памяти за тик сна.
const RECALL_ALPHA: f32 = 0.02;

/// Врождённый набор рефлексов основателей: «запах слева → мотор слева»
/// и зеркально. Индексы коры agent_layout: SL 0..20, SR 20..40,
/// ML 56..76, MR 76..96 (96 клеток).
pub fn innate_instincts() -> Vec<Instinct> {
    vec![
        Instinct {
            inputs: vec![(0, 5.0), (1, 5.0), (2, 5.0), (58, 10.0), (60, 10.0), (62, 10.0)],
            dopamine: 2.0,
        },
        Instinct {
            inputs: vec![(20, 5.0), (21, 5.0), (22, 5.0), (78, 10.0), (80, 10.0), (82, 10.0)],
            dopamine: 2.0,
        },
        // Отвращение: горький запах → тормозная зона I (гасит моторы).
        Instinct {
            inputs: vec![(12, 6.0), (15, 6.0), (32, 6.0), (35, 6.0), (44, 8.0), (48, 8.0)],
            dopamine: 1.5,
        },
    ]
}
/// Моторных спайков за тик для полной скорости.
#[allow(dead_code)]
const MOTOR_GATE_FULL: f32 = 24.0;

// ─── Еда ────────────────────────────────────────────────────────────────────
/// Вкус: сладкая лечит, горькая жжёт (отрицательный дофамин учит избегать).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Taste {
    Sweet,
    Bitter,
}

pub struct Food {
    pub x: f64,
    pub y: f64,
    pub alive: bool,
    regrow_at: usize,
    pub taste: Taste,
}

/// Препятствие: AABB-блок. Удар = урон + штраф дофамина.
#[derive(Clone, Copy, Debug)]
pub struct Wall {
    pub x: f64,
    pub y: f64,
    pub half_w: f64,
    pub half_h: f64,
}

// ─── Существо ───────────────────────────────────────────────────────────────
pub struct Creature {
    pub id: usize,
    pub sim: OrganSim,
    pub x: f64,
    pub y: f64,
    pub heading: f64,
    pub alive: bool,
    pub generation: usize,
    /// Накопленный путь за жизнь (для метрики эффективности).
    pub path_len: f64,
    pub meals: usize,
    /// Тик рождения (метрика продолжительности жизни).
    pub born_tick: usize,
    /// Тиков бодрствования подряд.
    awake_ticks: usize,
    /// Осталось тиков сна (0 = бодрствует).
    pub sleeping_for: usize,
    /// Накопленный голод (тик ниже черты +1, сытость −4).
    pub hungry_ticks: usize,
    /// Тик последней еды («в настроении» для пары).
    pub last_meal_tick: usize,
    /// Съедено горькой (видимая метрика обучения избеганию).
    pub bitter_eaten: usize,
}

impl Creature {
    fn new(id: usize, seed: u64, gen: usize, x: f64, y: f64, heading: f64, mind: bool) -> Self {
        let mut sim = OrganSim::new(false, seed);
        if mind {
            sim = sim.with_mind(seed ^ 0x5EED);
        }
        // Основатель: врождённые рефлексы + родильный сон + жёсткий
        // энергетический бюджет этой экосистемы.
        sim.instincts = innate_instincts();
        sim.birth_sleep();
        sim.blood.organ.cost_per_activity = MOVE_COST_PER_ACTIVITY;
        Self {
            id,
            sim,
            x,
            y,
            heading,
            alive: true,
            generation: gen,
            path_len: 0.0,
            meals: 0,
            born_tick: 0,
            awake_ticks: 0,
            sleeping_for: 0,
            hungry_ticks: 0,
            last_meal_tick: 0,
            bitter_eaten: 0,
        }
    }

    /// Дистанция до точки.
    #[allow(dead_code)]
    fn dist(&self, x: f64, y: f64) -> f64 {
        ((self.x - x).powi(2) + (self.y - y).powi(2)).sqrt()
    }
}

// ─── Экосистема ─────────────────────────────────────────────────────────────
pub struct Ecosystem {
    pub creatures: Vec<Creature>,
    pub foods: Vec<Food>,
    pub tick: usize,
    next_id: usize,
    #[allow(dead_code)]
    next_food_regrow: usize,
    rng: XorShift,
    /// История рождений: (поколение, id ребёнка, id родителей).
    pub births: Vec<(usize, usize, [usize; 2])>,
    /// История смертей: (id, поколение, тик смерти).
    pub deaths: Vec<(usize, usize, usize)>,
    /// Тики рождений: (id, тик) — для метрик продолжительности жизни.
    pub born_ticks: Vec<(usize, usize)>,
    /// Еды съедено за жизнь: (id, meals) — пишется при смерти.
    pub meals_log: Vec<(usize, usize)>,
    /// Всего проведённых снов всеми существами.
    pub sleeps_done: usize,
    /// Неподвижные препятствия (удар = урон и штраф).
    pub walls: Vec<Wall>,
}

impl Ecosystem {
    pub fn new(n_start: usize, base_seed: u64, with_mind: bool) -> Self {
        let mut rng = XorShift::new(base_seed);
        let mut creatures = Vec::with_capacity(n_start);
        let mut foods = Vec::new();
        for fi in 0..N_FOOD {
            foods.push(Food {
                x: (rng.unit() * 2.0 - 1.0) * ARENA_HALF,
                y: (rng.unit() * 2.0 - 1.0) * ARENA_HALF,
                alive: true,
                regrow_at: 0,
                taste: if fi % 4 == 3 { Taste::Bitter } else { Taste::Sweet },
            });
        }
        for i in 0..n_start {
            let x = (rng.unit() * 2.0 - 1.0) * ARENA_HALF * 0.3;
            let y = (rng.unit() * 2.0 - 1.0) * ARENA_HALF * 0.3;
            let heading = rng.unit() * std::f64::consts::PI * 2.0;
            let seed = base_seed.wrapping_add(i as u64 * 7919);
            creatures.push(Creature::new(i, seed, 0, x, y, heading, with_mind));
        }
        Self {
            creatures,
            foods,
            tick: 0,
            next_id: n_start,
            next_food_regrow: 0,
            rng,
            births: Vec::new(),
            deaths: Vec::new(),
            born_ticks: Vec::new(),
            meals_log: Vec::new(),
            sleeps_done: 0,
            walls: vec![
                Wall { x: -45.0, y: 40.0, half_w: 12.0, half_h: 55.0 },
                Wall { x: 45.0, y: -40.0, half_w: 12.0, half_h: 55.0 },
                Wall { x: 0.0, y: 0.0, half_w: 26.0, half_h: 10.0 },
            ],
        }
    }

    /// Один тик мира: все живые существа шагают параллельно, потом едят.
    pub fn step_tick(&mut self) -> usize {
        self.tick += 1;

        // Regrow еды.
        for f in self.foods.iter_mut() {
            if !f.alive && self.tick >= f.regrow_at {
                f.alive = true;
                f.x = (self.rng.unit() * 2.0 - 1.0) * ARENA_HALF;
                f.y = (self.rng.unit() * 2.0 - 1.0) * ARENA_HALF;
                f.taste = if self.rng.unit() < 0.25 { Taste::Bitter } else { Taste::Sweet };
            }
        }

        // Снимки мира для параллельной фазы (иммутабельные).
        let food_snapshot: Vec<(f64, f64, Taste)> = self
            .foods
            .iter()
            .filter(|f| f.alive)
            .map(|f| (f.x, f.y, f.taste))
            .collect();
        let walls_snapshot: Vec<Wall> = self.walls.clone();

        // ─── Фаза 1: Параллельный шаг мозга каждого существа ───
        // Каждое существо независимо — rayon раскидывает по ядрам.
        struct After {
            slept: bool,
            x: f64,
            y: f64,
        }
        let step_results: Vec<Option<After>> = self
            .creatures
            .par_iter_mut()
            .enumerate()
            .map(|(_ci, c)| {
                if !c.alive {
                    return None;
                }

                // ── СОН: мозг работает (кровь тратится), движения нет,
                // веса подтягиваются к снимку выученного. ──
                if c.sleeping_for > 0 {
                    c.sleeping_for -= 1;
                    let _ = c.sim.brain_step();
                    if c.sim.blood.conc[chem_glucose()] < STARVE_LINE {
                        c.hungry_ticks += 1;
                    } else {
                        c.hungry_ticks = c.hungry_ticks.saturating_sub(4);
                    }
                    if c.sleeping_for % 4 == 0 {
                        c.sim.organs.cortex.memory_recall(RECALL_ALPHA * 8.0);
                    }
                    return Some(After { slept: true, x: c.x, y: c.y });
                }

                let (x, y) = (c.x, c.y);

                // Ближайняя СЛАДКАЯ и ближайшая ГОРЬКАЯ — два канала носа.
                let nearest = |want: Taste| -> (f64, f64) {
                    food_snapshot
                        .iter()
                        .filter(|f| f.2 == want)
                        .map(|&(fx, fy, _)| ((fx - x).powi(2) + (fy - y).powi(2), fx, fy))
                        .min_by(|a, b| a.0.total_cmp(&b.0))
                        .map(|(_, fx, fy)| (fx, fy))
                        .unwrap_or((0.0, 0.0))
                };
                let (sweet, bitter) = (
                    nearest(Taste::Sweet),
                    nearest(Taste::Bitter),
                );
                let smell_of = |p: (f64, f64)| -> (f64, f64) {
                    let dx = p.0 - x;
                    let dy = p.1 - y;
                    let dist = (dx * dx + dy * dy).sqrt();
                    let prox = 60.0 / (60.0 + dist);
                    let mut rel = dy.atan2(dx) - c.heading;
                    while rel > std::f64::consts::PI { rel -= 2.0 * std::f64::consts::PI; }
                    while rel < -std::f64::consts::PI { rel += 2.0 * std::f64::consts::PI; }
                    (rel.sin().max(0.0) * prox, (-rel.sin()).max(0.0) * prox)
                };
                let (sl, sr) = smell_of(sweet);
                let (bl, br) = smell_of(bitter);

                // Шаг мозга: два вкуса в двух половинках сенсорных пулов.
                let step_obs = c.sim.brain_step_sensed_dual(sl, sr, bl, br);
                let ml = step_obs.cortex_regions[3] as f64;
                let mr = step_obs.cortex_regions[4] as f64;

                // Движение.
                let gate = ((ml + mr) as f32 / 24.0).min(1.0) as f64;
                let speed = 2.0 * gate;
                c.heading += 0.06 * (ml - mr);
                c.x += speed * c.heading.cos();
                c.y += speed * c.heading.sin();

                // ── Стены: вытолкнуть и развернуть, удар чувствителен. ──
                for w in &walls_snapshot {
                    if c.x > w.x - w.half_w - 6.0
                        && c.x < w.x + w.half_w + 6.0
                        && c.y > w.y - w.half_h - 6.0
                        && c.y < w.y + w.half_h + 6.0
                    {
                        // Выталкивание по меньшей оси проникновения.
                        let push_x = if c.x > w.x { w.x + w.half_w + 6.0 } else { w.x - w.half_w - 6.0 };
                        let push_y = if c.y > w.y { w.y + w.half_h + 6.0 } else { w.y - w.half_h - 6.0 };
                        if (c.x - push_x).abs() < (c.y - push_y).abs() {
                            c.x = push_x;
                            c.heading = std::f64::consts::PI - c.heading;
                        } else {
                            c.y = push_y;
                            c.heading = -c.heading;
                        }
                        c.sim.blood.bump_wall();
                        c.sim.organs.cortex.schedule_reward(-0.5);
                    }
                }

                if c.x.abs() > ARENA_HALF {
                    c.x = c.x.clamp(-ARENA_HALF, ARENA_HALF);
                    c.heading = std::f64::consts::PI - c.heading;
                }
                if c.y.abs() > ARENA_HALF {
                    c.y = c.y.clamp(-ARENA_HALF, ARENA_HALF);
                    c.heading = -c.heading;
                }

                // ── Голодный счётчик: ниже черты копится, сытость гасит. ──
                if c.sim.blood.conc[chem_glucose()] < STARVE_LINE {
                    c.hungry_ticks += 1;
                } else {
                    c.hungry_ticks = c.hungry_ticks.saturating_sub(4);
                }

                // ── Пора спать? Снимок памяти на входе в сон. ──
                c.awake_ticks += 1;
                if c.awake_ticks >= SLEEP_EVERY {
                    c.awake_ticks = 0;
                    c.sleeping_for = SLEEP_FOR;
                    c.sim.organs.cortex.memory_snapshot();
                }

                Some(After { slept: false, x: c.x, y: c.y })
            })
            .collect();
        self.sleeps_done += step_results
            .iter()
            .flatten()
            .filter(|a| a.slept)
            .count();

        // ─── Фаза 2: Последовательная еда (общий ресурс). ───
        let mut ate_flags = vec![false; self.creatures.len()];
        for (ci, result) in step_results.iter().enumerate() {
            let Some(sr) = result else { continue };
            if sr.slept { continue; } // спящий не ест
            if self.blood_wants_food_pub(ci) {
                for f in self.foods.iter_mut() {
                    if !f.alive { continue; }
                    let d = ((f.x - sr.x).powi(2) + (f.y - sr.y).powi(2)).sqrt();
                    if d < EAT_RADIUS {
                        f.alive = false;
                        f.regrow_at = self.tick + FOOD_REGROW;
                        match f.taste {
                            Taste::Sweet => {
                                self.creatures[ci].sim.blood.ingest_food(90.0);
                                self.creatures[ci].sim.organs.cortex.schedule_reward(1.0);
                                if let Some(m) = &mut self.creatures[ci].sim.mind {
                                    m.reward(1.0);
                                }
                            }
                            Taste::Bitter => {
                                // Горько: энергия минус, лёгкий урон, ОТРИЦАТЕЛЬНЫЙ
                                // дофамин — R-STDP ослабляет путь к этой ошибке.
                                let bl = &mut self.creatures[ci].sim.blood;
                                bl.conc[chem_glucose()] = (bl.conc[chem_glucose()] - 25.0).max(-20.0);
                                bl.organ.damage = (bl.organ.damage + 0.01).min(1.0);
                                self.creatures[ci].sim.organs.cortex.schedule_reward(-3.0);
                                if let Some(m) = &mut self.creatures[ci].sim.mind {
                                    m.reward(-2.0);
                                }
                                self.creatures[ci].bitter_eaten += 1;
                            }
                        }
                        self.creatures[ci].meals += 1;
                        self.creatures[ci].last_meal_tick = self.tick;
                        ate_flags[ci] = true;
                        break;
                    }
                }
            }
        }

        // Размножение: два недавно евших существа рядом → ребёнок.
        // Слоты жёстко: не больше, чем свободно до капа, за один тик.
        let slots = MAX_CREATURES.saturating_sub(self.alive_count());
        if slots > 0 {
            let mut born_this_tick = 0usize;
            for (a_idx, b_idx) in self.find_breed_pairs() {
                if born_this_tick >= slots { break; }
                let child_id = self.next_id;
                self.next_id += 1;
                let gen = self.creatures[a_idx].generation.max(self.creatures[b_idx].generation) + 1;
                let cx = (self.creatures[a_idx].x + self.creatures[b_idx].x) / 2.0;
                let cy = (self.creatures[a_idx].y + self.creatures[b_idx].y) / 2.0;

                // Ребёнок: тело и кора от A (± мутация), политика W2 —
                // кроссовер, гены правил и инстинкты чередуются,
                // родильный сон внутри.
                let mut child_sim = spawn_child_crossover(
                    &self.creatures[a_idx].sim,
                    &self.creatures[b_idx].sim,
                    child_seed(child_id),
                );
                // Бюджет экосистемы (кроссовер создаёт тело с дефолтом).
                child_sim.blood.organ.cost_per_activity = MOVE_COST_PER_ACTIVITY;
                let heading = self.rng.unit() * std::f64::consts::PI * 2.0;
                self.births
                    .push((gen, child_id, [self.creatures[a_idx].id, self.creatures[b_idx].id]));
                self.born_ticks.push((child_id, self.tick));
                self.creatures.push(Creature {
                    id: child_id,
                    sim: child_sim,
                    x: cx,
                    y: cy,
                    heading,
                    alive: true,
                    generation: gen,
                    path_len: 0.0,
                    meals: 0,
                    born_tick: self.tick,
                    awake_ticks: 0,
                    sleeping_for: SLEEP_FOR, // новорождённый досыпает
                    hungry_ticks: 0,
                    last_meal_tick: 0,
                    bitter_eaten: 0,
                });
                born_this_tick += 1;
            }
        } // конец капа популяции

        // Смерть и очистка.
        let mut died = Vec::new();
        for (ci, c) in self.creatures.iter_mut().enumerate() {
            if !c.alive { continue; }
            // Спящие уже шагнули в фазе 1 — не шагаем дважды.
            // (Новорождённые этой фазы ещё нет в step_results.)
            let already_stepped = step_results
                .get(ci)
                .and_then(|r| r.as_ref())
                .is_some_and(|a| a.slept);
            if !already_stepped {
                let _ = c.sim.brain_step(); // кровь/обмен без движения
            }
            if c.sim.blood.organ.damage >= 1.0
                || c.sim.blood.conc[chem_glucose()] <= -20.0
                || c.hungry_ticks >= STARVE_DEATH_TICKS
            {
                c.alive = false;
                died.push((c.id, c.generation, self.tick));
                self.meals_log.push((c.id, c.meals));
            } else if c.hungry_ticks == 0 && c.sim.blood.conc[chem_glucose()] > STARVE_LINE * 3.0 {
                // Сытая жизнь лечит: градиент давления вместо храповика.
                c.sim.blood.organ.damage = (c.sim.blood.organ.damage - 0.0005).max(0.0);
            }
        }
        for d in &died {
            self.deaths.push(*d);
        }
        self.creatures.retain(|c| c.alive);

        ate_flags.iter().filter(|&&a| a).count()
    }

    /// Живых существ.
    pub fn alive_count(&self) -> usize {
        self.creatures.iter().filter(|c| c.alive).count()
    }

    /// Игрок бросил еду в точку (оживляет ближайший мёртвый слот).
    pub fn drop_food(&mut self, x: f64, y: f64) -> bool {
        let x = x.clamp(-ARENA_HALF, ARENA_HALF);
        let y = y.clamp(-ARENA_HALF, ARENA_HALF);
        if let Some(f) = self.foods.iter_mut().find(|f| !f.alive) {
            f.x = x;
            f.y = y;
            f.alive = true;
            true
        } else {
            false
        }
    }

    fn find_breed_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        // «В настроении»: ел недавно и не в глубоком голоде.
        let ready: Vec<usize> = (0..self.creatures.len())
            .filter(|&i| {
                let c = &self.creatures[i];
                c.alive
                    && self.tick.saturating_sub(c.last_meal_tick) <= BREED_AFTER_MEAL_TICKS
                    && c.hungry_ticks < STARVE_DEATH_TICKS / 2
                    && self.creatures[i].sim.blood.conc[chem_glucose()] > STARVE_LINE
            })
            .collect();
        for i in 0..ready.len() {
            for j in (i + 1)..ready.len() {
                let (ai, bi) = (ready[i], ready[j]);
                let d = ((self.creatures[ai].x - self.creatures[bi].x).powi(2)
                    + (self.creatures[ai].y - self.creatures[bi].y).powi(2))
                    .sqrt();
                if d < BREED_RADIUS {
                    pairs.push((ai, bi));
                }
            }
        }
        pairs
    }

    fn blood_glucose(&self, ci: usize) -> f32 {
        self.creatures[ci].sim.blood.conc[chem_glucose()]
    }

    #[allow(dead_code)]
    fn blood_fatigue(&self, ci: usize) -> f32 {
        self.creatures[ci].sim.blood.conc[chem_fatigue()]
    }

    #[allow(dead_code)]
    fn blood_wants_food_pub(&self, ci: usize) -> bool {
        self.creatures[ci].sim.blood.wants_food()
    }

    #[allow(dead_code)]
    fn blood_wants_food(&self, ci: usize) -> bool {
        self.creatures[ci].sim.blood.wants_food()
    }
}

#[allow(dead_code)]
fn chem_hunger() -> usize { crate::body::metabolism::chem::HUNGER }
#[allow(dead_code)]
fn chem_glucose() -> usize { crate::body::metabolism::chem::GLUCOSE }
#[allow(dead_code)]
fn chem_fatigue() -> usize { crate::body::metabolism::chem::FATIGUE }

fn child_seed(id: usize) -> u64 {
    id as u64 | 0xC41D_0000_0000_0000
}
