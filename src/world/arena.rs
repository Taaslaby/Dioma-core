//! -------------------------------------------------------------------------
//! Filename:    src/arena.rs
//! Class:       DebugSuite
//! Purpose:     agent test demo headless suite
//! Description:
//!
//! regions: 0=SL 1=SR 2=I 3=ML 4=MR.
//! author: Atlasov anton
//! 2026.08.23
//! -------------------------------------------------------------------------

use crate::engine::neurons::{gains_from, Brain, PlasticityParams, BRAIN_STEPS_PER_TICK, DT_MS};
use crate::engine::grow::{agent_layout, grow, Steering};
use crate::rng::XorShift;
use crate::Grown;

pub const N_SL: usize = 20;
pub const N_SR: usize = 20;
pub const N_I: usize = 16;
pub const N_ML: usize = 20;
pub const N_MR: usize = 20;

pub const N_REGIONS: usize = 5;
pub const ARENA_HALF: f64 = 200.0;
const EAT_RADIUS: f64 = 14.0;
pub const EPISODE_TIMEOUT_TICKS: usize = 600;

pub const TONIC_DRIVE: f32 = 0.12;
const SENSOR_GAIN: f32 = 3.5;
const AGENT_SPEED: f64 = 2.0;
const TURN_GAIN: f64 = 0.06;

/// Гейны путей арены — все из генома (третье измерение эволюционного поиска:
/// steering уже даёт noise_weight/synapse_dist, это — сила проводки).
pub fn arena_gains(genome: &Genome) -> Vec<Vec<f32>> {
    gains_from(
        5,
        &[
            (0, 3, genome.w_sm),
            (0, 4, genome.w_sm),
            (1, 3, genome.w_sm),
            (1, 4, genome.w_sm),
            (0, 2, genome.w_si),
            (1, 2, genome.w_si),
            (0, 0, genome.w_ss),
            (1, 1, genome.w_ss),
            (2, 3, genome.w_i),
            (2, 4, genome.w_i),
        ],
    )
}

/// «Геном» агента арены: параметры роста + врождённый путь + тело.
#[derive(Clone, Debug)]
pub struct Genome {
    pub seed: u64,
    pub gradient_weight: f64,
    pub noise_weight: f64,
    pub max_len_s: f64,
    pub synapse_dist: f64,
    pub w_sm: f32,
    pub sensor_gain: f32,
    pub turn_gain: f64,
    /// Гейн S→I (рекрутирование торможения) — кодируется геномом.
    pub w_si: f32,
    /// Гейн S→S (рецидивное возбуждение) — кодируется геномом.
    pub w_ss: f32,
    /// Гейн I→M (торможение моторов) — кодируется геномом.
    pub w_i: f32,
    /// Гены экспрессии каналов (HAAP-физиология). Нокауты = 0.
    pub physio: crate::body::physiology::PhysioGenes,
    /// Модель нейронов: дикий тип — LIF (доказанный баланс); HAAP —
    /// экспериментальная ветка (VNC-физиология, би-стабильность лечится).
    pub brain_model: crate::engine::neurons::BrainModel,
}

impl Genome {
    /// Дикий тип — параметры, на которых отлажена арена.
    pub fn wildtype(seed: u64) -> Self {
        Self {
            seed,
            gradient_weight: 0.5,
            noise_weight: 0.15,
            max_len_s: 170.0,
            synapse_dist: 13.0,
            w_sm: 0.30,
            sensor_gain: SENSOR_GAIN,
            turn_gain: TURN_GAIN,
            w_si: 0.15,
            w_ss: 0.04,
            w_i: 0.30,
            physio: crate::body::physiology::PhysioGenes::default(),
            brain_model: crate::engine::neurons::BrainModel::LifAlpha,
        }
    }

    pub fn steering(&self) -> Steering {
        Steering {
            noise_weight: self.noise_weight,
            gradient_weight: self.gradient_weight,
            synapse_dist: self.synapse_dist,
            ..Steering::default()
        }
    }
}

pub fn grow_agent(genome: &Genome) -> Grown {
    let mut layout = agent_layout();
    // Длина сенсорных аксонов и радиус синапса — из генома.
    let s_ax = crate::engine::grow::axon("ML", genome.max_len_s);
    let s_ax2 = crate::engine::grow::axon("MR", genome.max_len_s);
    layout.regions[0].axons = vec![s_ax.clone(), s_ax];
    layout.regions[1].axons = vec![s_ax2.clone(), s_ax2];
    grow(&layout, &genome.steering(), genome.seed)
}

/// Базовые параметры арены + твик колбэком (для лаборатории мозга).
pub fn arena_brain_with(
    grown: &Grown,
    genome: &Genome,
    tweak: impl FnOnce(&mut PlasticityParams),
) -> Brain {
    // Рекрутирование торможения — врождённое: пластичность только на
    // возбуждающих путях (дофамин в S→I разгонял молчание моторов).
    // S→S рекуррентное возбуждение — тоже врождённое: пластичная S→S
    // при награде уходит в кап и даёт самоподдерживающуюся судорогу
    // сенсорного пула (сеть горит без входа).
    let mut params = PlasticityParams {
        frozen_pairs: vec![(0, 2), (1, 2), (0, 0), (1, 1)],
        w_max: 3.0, // потолок: выше — моторы симметрично сатурятся
        ..Default::default()
    };
    // HAAP: без рефрактерного периода и с τ_syn=12 мс пластичные пути
    // уходят в самоподдерживающуюся судорогу — гомеостаз каждый тик.
    if genome.brain_model == crate::engine::neurons::BrainModel::Haap {
        params.homeo_every_ticks = 1;
    }
    tweak(&mut params);
    match genome.brain_model {
        crate::engine::neurons::BrainModel::LifAlpha => Brain::new(grown, arena_gains(genome), params),
        crate::engine::neurons::BrainModel::Haap => {
            let columns = crate::body::physiology::brain_columns(
                &grown.region_of,
                crate::body::physiology::arena_archetype,
                &genome.physio,
            );
            Brain::new_haap(grown, arena_gains(genome), params, &columns)
        }
    }
}

pub fn arena_brain(grown: &Grown, genome: &Genome) -> Brain {
    arena_brain_with(grown, genome, |_| {})
}

// ─── Тело ───────────────────────────────────────────────────────────────────
pub struct Agent {
    pub x: f64,
    pub y: f64,
    pub heading: f64,
    pub food_x: f64,
    pub food_y: f64,
    pub spawn_dist_min: f64,
    pub prox_range: f64,
    rng: XorShift,
}

impl Agent {
    /// Умеренный мир: еда недалеко, сенсор дальнобойный.
    pub fn new(seed: u64) -> Self {
        Self::with_difficulty(seed, 80.0, 60.0)
    }

    /// Жёсткий мир (эволюция): еда далеко, сенсор короткий.
    pub fn new_hard(seed: u64) -> Self {
        Self::with_difficulty(seed, 150.0, 30.0)
    }

    pub fn with_difficulty(seed: u64, spawn_dist_min: f64, prox_range: f64) -> Self {
        let mut rng = XorShift::new(seed);
        let mut a = Agent {
            x: 0.0,
            y: 0.0,
            heading: rng.unit() * 2.0 * std::f64::consts::PI,
            food_x: 0.0,
            food_y: 0.0,
            spawn_dist_min,
            prox_range,
            rng,
        };
        a.respawn_food();
        a
    }

    pub fn respawn_food(&mut self) {
        for _ in 0..100 {
            let fx = (self.rng.unit() * 2.0 - 1.0) * ARENA_HALF;
            let fy = (self.rng.unit() * 2.0 - 1.0) * ARENA_HALF;
            let d = ((fx - self.x).powi(2) + (fy - self.y).powi(2)).sqrt();
            if d >= self.spawn_dist_min {
                self.food_x = fx;
                self.food_y = fy;
                return;
            }
        }
        self.food_x = -self.x.clamp(-ARENA_HALF, ARENA_HALF);
        self.food_y = -self.y.clamp(-ARENA_HALF, ARENA_HALF);
    }

    pub fn sensors(&self) -> (f64, f64) {
        let (dx, dy) = (self.food_x - self.x, self.food_y - self.y);
        let dist = (dx * dx + dy * dy).sqrt();
        let mut rel = dy.atan2(dx) - self.heading;
        while rel > std::f64::consts::PI {
            rel -= 2.0 * std::f64::consts::PI;
        }
        while rel < -std::f64::consts::PI {
            rel += 2.0 * std::f64::consts::PI;
        }
        let prox = self.prox_range / (self.prox_range + dist);
        (rel.sin().max(0.0) * prox, (-rel.sin()).max(0.0) * prox)
    }

    pub fn move_agent(&mut self, spikes_l: i32, spikes_r: i32, turn_gain: f64, speed: f64) {
        let d = (spikes_l - spikes_r) as f64;
        self.heading += turn_gain * d;
        self.x += speed * self.heading.cos();
        self.y += speed * self.heading.sin();
        if self.x.abs() > ARENA_HALF {
            self.x = self.x.clamp(-ARENA_HALF, ARENA_HALF);
            self.heading = std::f64::consts::PI - self.heading;
        }
        if self.y.abs() > ARENA_HALF {
            self.y = self.y.clamp(-ARENA_HALF, ARENA_HALF);
            self.heading = -self.heading;
        }
    }

    pub fn at_food(&self) -> bool {
        let (dx, dy) = (self.food_x - self.x, self.food_y - self.y);
        (dx * dx + dy * dy).sqrt() < EAT_RADIUS
    }
}

/// Один эпизод: агент ищет еду. Возвращает число тиков до еды (или timeout).
pub fn run_episode(
    brain: &mut Brain,
    agent: &mut Agent,
    genome: &Genome,
    dopamine_on: bool,
) -> usize {
    run_episode_debug(brain, agent, genome, dopamine_on).0
}

/// Канонический эпизод со статистикой стабильности: (тики до еды,
/// макс спайков за тик). После еды агент «жует»: reward_delay+1 тик
/// тишины — дофамин приходит в покой и гомеостаз его подрезает; без
/// паузы награда детонирует в старте следующего эпизода (сенсорный
/// драйв × кап весов = судорога всей сети).
pub fn run_episode_debug(
    brain: &mut Brain,
    agent: &mut Agent,
    genome: &Genome,
    dopamine_on: bool,
) -> (usize, i64) {
    let n = brain.n;
    let mut worst_tick = 0i64;
    for tick in 0..EPISODE_TIMEOUT_TICKS {
        let (left, right) = agent.sensors();
        let mut ext = vec![TONIC_DRIVE; n];
        for (i, cell) in ext.iter_mut().enumerate().take(N_SL) {
            *cell += genome.sensor_gain * left as f32 * (0.5 + 0.5 * crate::rng::hash01(i) as f32);
        }
        for (i, cell) in ext.iter_mut().enumerate().take(N_SL + N_SR).skip(N_SL) {
            *cell += genome.sensor_gain * right as f32 * (0.5 + 0.5 * crate::rng::hash01(i + 999) as f32);
        }

        let mut regions = [0i32; N_REGIONS];
        for _ in 0..brain.steps_per_tick() {
            let spikes = brain.step(&ext);
            for (r, sum) in regions.iter_mut().enumerate() {
                *sum += brain.region_spikes(&spikes, r);
            }
        }
        worst_tick =
            worst_tick.max(regions.iter().map(|&x| x as i64).sum::<i64>());
        agent.move_agent(regions[REG_ML], regions[REG_MR], genome.turn_gain, AGENT_SPEED);

        if agent.at_food() {
            if dopamine_on {
                brain.schedule_reward(1.0);
            }
            agent.respawn_food();
            // Жевание: тишина, пока не придёт отложенная награда.
            let zero = vec![TONIC_DRIVE; n];
            for _ in 0..=brain.reward_delay_ticks() {
                let mut chew = [0i32; N_REGIONS];
                for _ in 0..brain.steps_per_tick() {
                    let spikes = brain.step(&zero);
                    for (r, sum) in chew.iter_mut().enumerate() {
                        *sum += brain.region_spikes(&spikes, r);
                    }
                }
                worst_tick =
                    worst_tick.max(chew.iter().map(|&x| x as i64).sum::<i64>());
            }
            return (tick + 1, worst_tick);
        }
    }
    (EPISODE_TIMEOUT_TICKS, worst_tick)
}

/// Приспособленность генома (эволюция): жёсткий мир, среднее время до еды.
pub fn evaluate(genome: &Genome, dopamine_on: bool, n_episodes: usize) -> f64 {
    let grown = grow_agent(genome);
    let mut brain = arena_brain(&grown, genome);
    let mut agent = Agent::new_hard(genome.seed ^ 0xE0A1);
    let mut total = 0usize;
    for _ in 0..n_episodes {
        total += run_episode(&mut brain, &mut agent, genome, dopamine_on);
    }
    total as f64 / n_episodes as f64 * BRAIN_STEPS_PER_TICK as f64 * DT_MS as f64 / 1000.0
}

pub const REG_SL: usize = 0;
pub const REG_SR: usize = 1;
pub const REG_I: usize = 2;
pub const REG_ML: usize = 3;
pub const REG_MR: usize = 4;

