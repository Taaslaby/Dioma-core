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
use crate::heritage::MUTATE_W2_SIGMA;
use crate::organ_life::OrganSim;
use crate::rng::XorShift;
use rayon::prelude::*;

pub const ECO_TICKS: usize = 8000; // 200 с
pub const EAT_RADIUS: f64 = 14.0;
pub const BREED_RADIUS: f64 = 30.0;
pub const BREED_GLUCOSE_MIN: f32 = 80.0;
/// Максимум существ в экосистеме — без капа размножение экспоненциальное.
pub const MAX_CREATURES: usize = 20;
pub const ARENA_HALF: f64 = 200.0;
pub const N_FOOD: usize = 24;
pub const FOOD_REGROW: usize = 300;
/// Моторных спайков за тик для полной скорости.
#[allow(dead_code)]
const MOTOR_GATE_FULL: f32 = 24.0;

// ─── Еда ────────────────────────────────────────────────────────────────────
pub struct Food {
    pub x: f64,
    pub y: f64,
    pub alive: bool,
    regrow_at: usize,
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
}

impl Creature {
    fn new(id: usize, seed: u64, gen: usize, x: f64, y: f64, heading: f64, mind: bool) -> Self {
        let mut sim = OrganSim::new(false, seed);
        if mind {
            sim = sim.with_mind(seed ^ 0x5EED);
        }
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
}

impl Ecosystem {
    pub fn new(n_start: usize, base_seed: u64, with_mind: bool) -> Self {
        let mut rng = XorShift::new(base_seed);
        let mut creatures = Vec::with_capacity(n_start);
        let mut foods = Vec::new();
        for _ in 0..N_FOOD {
            foods.push(Food {
                x: (rng.unit() * 2.0 - 1.0) * ARENA_HALF,
                y: (rng.unit() * 2.0 - 1.0) * ARENA_HALF,
                alive: true,
                regrow_at: 0,
            });
        }
        for i in 0..n_start {
            let x = (rng.unit() * 2.0 - 1.0) * ARENA_HALF * 0.8;
            let y = (rng.unit() * 2.0 - 1.0) * ARENA_HALF * 0.8;
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
            }
        }

        // Собираем позиции еды для сенсорики (иммутабельно).
        let food_snapshot: Vec<(f64, f64)> = self
            .foods
            .iter()
            .filter(|f| f.alive)
            .map(|f| (f.x, f.y))
            .collect();

        // ─── Фаза 1: Параллельный шаг мозга каждого существа ───
        // Каждое существо независимо — rayon раскидывает по ядрам.
        struct StepResult {
            #[allow(dead_code)]
            ml: i32,
            #[allow(dead_code)]
            mr: i32,
            new_x: f64,
            new_y: f64,
            #[allow(dead_code)]
            new_heading: f64,
        }
        let step_results: Vec<Option<StepResult>> = self
            .creatures
            .par_iter_mut()
            .enumerate()
            .map(|(_ci, c)| {
                if !c.alive {
                    return None;
                }
                let (x, y) = (c.x, c.y);

                // Ближайшая еда из снимка.
                let (fx, fy) = food_snapshot
                    .iter()
                    .map(|&(fx, fy)| {
                        let d = (fx - x).powi(2) + (fy - y).powi(2);
                        (d, fx, fy)
                    })
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .map(|(_, fx, fy)| (fx, fy))
                    .unwrap_or((0.0, 0.0));

                // Сенсорика.
                let dx = fx - x;
                let dy = fy - y;
                let dist = (dx * dx + dy * dy).sqrt();
                let prox = 60.0 / (60.0 + dist);
                let mut rel = dy.atan2(dx) - c.heading;
                while rel > std::f64::consts::PI { rel -= 2.0 * std::f64::consts::PI; }
                while rel < -std::f64::consts::PI { rel += 2.0 * std::f64::consts::PI; }
                let smell_l = rel.sin().max(0.0) * prox;
                let smell_r = (-rel.sin()).max(0.0) * prox;

                // Шаг мозга.
                let step_obs = c.sim.brain_step_sensed(smell_l, smell_r);
                let ml = step_obs.cortex_regions[3] as f64;
                let mr = step_obs.cortex_regions[4] as f64;

                // Движение.
                let gate = ((ml + mr) as f32 / 24.0).min(1.0) as f64;
                let speed = 2.0 * gate;
                c.heading += 0.06 * (ml - mr);
                c.x += speed * c.heading.cos();
                c.y += speed * c.heading.sin();
                if c.x.abs() > ARENA_HALF {
                    c.x = c.x.clamp(-ARENA_HALF, ARENA_HALF);
                    c.heading = std::f64::consts::PI - c.heading;
                }
                if c.y.abs() > ARENA_HALF {
                    c.y = c.y.clamp(-ARENA_HALF, ARENA_HALF);
                    c.heading = -c.heading;
                }

                Some(StepResult {
                    ml: step_obs.cortex_regions[3],
                    mr: step_obs.cortex_regions[4],
                    new_x: c.x,
                    new_y: c.y,
                    new_heading: c.heading,
                })
            })
            .collect();

        // ─── Фаза 2: Последовательная еда (общий ресурс). ───
        let mut ate_flags = vec![false; self.creatures.len()];
        for (ci, result) in step_results.iter().enumerate() {
            let Some(sr) = result else { continue };
            if self.blood_wants_food_pub(ci) {
                for f in self.foods.iter_mut() {
                    if !f.alive { continue; }
                    let d = ((f.x - sr.new_x).powi(2) + (f.y - sr.new_y).powi(2)).sqrt();
                    if d < EAT_RADIUS {
                        f.alive = false;
                        f.regrow_at = self.tick + FOOD_REGROW;
                        self.creatures[ci].sim.blood.ingest_food(90.0);
                        self.creatures[ci].sim.organs.cortex.schedule_reward(1.0);
                        if let Some(m) = &mut self.creatures[ci].sim.mind {
                            m.reward(1.0);
                        }
                        self.creatures[ci].meals += 1;
                        ate_flags[ci] = true;
                        break;
                    }
                }
            }
        }

        // Размножение: два сытых существа рядом → ребёнок.
        // Кап популяции: без него размножение экспоненциальное → зависание.
        if self.alive_count() < MAX_CREATURES {
            let breed_pairs = self.find_breed_pairs();
            for (a_idx, b_idx) in breed_pairs {
            let child_id = self.next_id;
            self.next_id += 1;
            let gen = self.creatures[a_idx].generation.max(self.creatures[b_idx].generation) + 1;
            let cx = (self.creatures[a_idx].x + self.creatures[b_idx].x) / 2.0;
            let cy = (self.creatures[a_idx].y + self.creatures[b_idx].y) / 2.0;

            // Ребёнок: тело нового сида, мозги = микс родителей ± мутация.
            let mut child_sim = OrganSim::new(false, child_seed(child_id));
            if let Some(mind_a) = &self.creatures[a_idx].sim.mind {
                if let Some(mind_b) = &self.creatures[b_idx].sim.mind {
                    let wa = mind_a.weight_heat();
                    let wb = mind_b.weight_heat();
                    let mixed: Vec<f32> = wa
                        .iter()
                        .zip(wb.iter())
                        .enumerate()
                        .map(|(j, (&a, &b))| {
                            let pick = if j % 2 == 0 { a } else { b };
                            let noise = gauss_noise(child_id as u64, j as u64) * MUTATE_W2_SIGMA;
                            pick + noise
                        })
                        .collect();
                    child_sim.mind = Some(crate::mind::CreatureMind::new(child_seed(child_id)));
                    if let Some(cm) = &mut child_sim.mind {
                        cm.set_weights(&mixed);
                    }
                }
            }

            let heading = self.rng.unit() * std::f64::consts::PI * 2.0;
            self.births.push((gen, child_id, [self.creatures[a_idx].id, self.creatures[b_idx].id]));
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
            });
        }
        } // конец капа популяции

        // Смерть и очистка.
        let mut died = Vec::new();
        for c in self.creatures.iter_mut() {
            if !c.alive { continue; }
            let obs = c.sim.brain_step(); // мозг + кровь (без движения — уже подвинуты)
            if obs.sitting { /* сидит */ }
            if c.sim.blood.organ.damage >= 1.0 || c.sim.blood.conc[chem_glucose()] <= -20.0 {
                c.alive = false;
                died.push((c.id, c.generation, self.tick));
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

    fn find_breed_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        let alive: Vec<usize> = (0..self.creatures.len())
            .filter(|&i| {
                self.creatures[i].alive && self.blood_glucose(i) > BREED_GLUCOSE_MIN
            })
            .collect();
        for i in 0..alive.len() {
            for j in (i + 1)..alive.len() {
                let (ai, bi) = (alive[i], alive[j]);
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
fn chem_hunger() -> usize { crate::metabolism::chem::HUNGER }
#[allow(dead_code)]
fn chem_glucose() -> usize { crate::metabolism::chem::GLUCOSE }
#[allow(dead_code)]
fn chem_fatigue() -> usize { crate::metabolism::chem::FATIGUE }

fn child_seed(id: usize) -> u64 {
    id as u64 | 0xC41D_0000_0000_0000
}

fn gauss_noise(seed: u64, i: u64) -> f32 {
    let x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(i.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    (((x >> 40) & 0xFFFF) as f32 / 65536.0) * 2.0 - 1.0
}
