//! -------------------------------------------------------------------------
//! Filename:    src/organ_life.rs
//! Class:       DebugSuite
//! Purpose:     free life on the ORGAN brain (vertical slice)
//! Description: тот же мир/кровь, что free_life, но тело ведёт органный
//!              мозг: кора (LIF-арена) ищет еду и учится на дофамине,
//!              средний мозг (HAAP) переводит голод крови в мотивационный
//!              тонус коры, спина (HH) даёт болевой рефлекс от ударов о
//!              стену. Наблюдение — OrganTickObs для UI/протоколов.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use crate::arena::{arena_gains, grow_agent, Genome, N_REGIONS, N_SL, N_SR};
use crate::brain::PlasticityParams;
use crate::free_life::{WorldSim, FOOD_PORTION};
use crate::metabolism::{chem, locus, Blood};
use crate::organs::{standard_spine_conn, Axon, OrganBrain, OrganId};

pub use crate::free_life::{ARENA_HALF, SIM_TICKS};

/// Голод [0..255] → ток среднего мозга.
///
/// Шкала заточена под рот (wants_food = HUNGER>40): к порогу рта драйв уже
/// близок к максимуму — «слегка голоден» должно значить «активно ищи».
const HUNGER_DRIVE_MAX: f32 = 6.0;
fn hunger_drive(hunger: f32) -> f32 {
    (hunger * 0.12).min(HUNGER_DRIVE_MAX)
}
/// Вес аксона средний-мозг→кора: спайки за тик × вес = мотивационный ток.
///
/// Калибровка: лёгкий голод (15–40, основная полоса жизни) даёт ~33 сп/тик
/// → +0.6 тока на сенсорику (≈ половина полного сигнала); глубокий голод
/// (~300 сп/тик) → +5.4 — сенсорика гейтится на полную.
const MID_GAIN: f32 = 0.018;
/// Длительность болевого разряда в спине после удара о стену (тиков).
const PAIN_TICKS: usize = 8;
/// Боль в спину: ток на сенсорный пул (HH-единицы).
const PAIN_AMNT: f32 = 15.0;
/// Порог сытости органной жизни: выше него дефицита нет и голод гаснет
/// (в free_life порог 150 — существо никогда не наедается и не сидит).
const ORGAN_FED_THRESHOLD: f32 = 100.0;
/// Тормоз покоя на ML/MR: сытое существо сидит (голод снимает тормоз).
const REST_INHIBIT: f32 = 0.6;
/// Полная скорость при сумме моторных спайков за тик ≥ этого порога.
const MOTOR_GATE_FULL: f32 = 24.0;

/// Пошаговая свободная жизнь на органном мозге.
pub struct OrganSim {
    pub organs: OrganBrain,
    pub blood: Blood,
    pub world: WorldSim,
    pub tick: usize,
    pain_ticks: usize,
    /// Накопитель наград коры между дофаминовыми доставками.
    rewards_pending: usize,
}

impl OrganSim {
    pub fn new(robot: bool, seed: u64) -> Self {
        let genome = Genome::wildtype(seed);
        let grown = grow_agent(&genome);
        let mut organs = OrganBrain::new(
            standard_spine_conn(),
            grown,
            arena_gains(&genome),
            PlasticityParams {
                // Врождённые дуги арены: торможение и S→S не учатся.
                frozen_pairs: vec![(0, 2), (1, 2), (0, 0), (1, 1)],
                w_max: 3.0,
                ..Default::default()
            },
            24,
        );
        // Средний мозг → кора: мотивационный ГЕЙТ НА СЕНСОРИКУ.
        //
        // Аксон на сенсорные пулы (SL/SR), а не на весь мозг: голод
        // усиливает сенсорный сигнал («еда пахнет сильнее»), сохраняя
        // ML/MR дифференциал. Тонус на ВСЕ нейроны симметрично заливает
        // моторные пулы — дифференциал поворотов падает в 50 раз и
        // существо дёргается на месте (diag: |ML-MR| 7.5→0.14).
        organs.add_axon(Axon {
            from: OrganId::Midbrain,
            pre: (0..24).collect(),
            to: OrganId::Cortex,
            post: (0..N_SL + N_SR).collect(),
            weight_per_spike: MID_GAIN,
        });
        Self {
            organs,
            blood: {
                let mut b = Blood::creature(robot);
                if !robot {
                    // Химическое существо: короткий стартовый запас и меньшая
                    // порция — циклы «голод→поиск→еда» повторяются всю жизнь,
                    // мотивационный канал живёт постоянно.
                    b.conc[chem::GLUCOSE] = 40.0;
                }
                b
            },
            world: WorldSim::new(seed ^ 0xB100),
            tick: 0,
            pain_ticks: 0,
            rewards_pending: 0,
        }
    }

    /// Один тик мира (25 мс).
    pub fn step_tick(&mut self) -> OrganTickObs {
        let n_cortex = self.organs.cortex.n;
        self.world.regrow(self.tick);
        let (left, right) = self.world.sensors();

        // Кора: тоник + сенсорика (как в арене).
        let hunger = self.blood.conc[chem::HUNGER];
        let mut cortex_ext = vec![self.blood.motor_tonic(); n_cortex];
        let sg = self.blood.sensor_gain();
        for (i, cell) in cortex_ext.iter_mut().enumerate().take(N_SL) {
            *cell += sg * left as f32 * (0.5 + 0.5 * crate::rng::hash01(i) as f32);
        }
        for (i, cell) in cortex_ext.iter_mut().enumerate().take(N_SL + N_SR).skip(N_SL) {
            *cell += sg * right as f32 * (0.5 + 0.5 * crate::rng::hash01(i + 999) as f32);
        }
        // Тормоз покоя: сытому существу незачем двигаться — подавляем
        // моторные пулы (голод снимает тормоз линейно к порогу рта).
        let rest = REST_INHIBIT * (1.0 - (hunger / 40.0).min(1.0));
        if rest > 0.0 {
            let (ml0, ml1) = self.organs.cortex.region_bounds(3);
            let (mr0, mr1) = self.organs.cortex.region_bounds(4);
            for cell in cortex_ext[ml0..ml1].iter_mut() {
                *cell -= rest;
            }
            for cell in cortex_ext[mr0..mr1].iter_mut() {
                *cell -= rest;
            }
        }

        // Средний мозг: голод из крови → драйв.
        let mid_ext = vec![hunger_drive(self.blood.conc[chem::HUNGER]); 24];

        // Спина: боль после удара о стену.
        if self.world.last_wall_hit && self.pain_ticks == 0 {
            self.pain_ticks = PAIN_TICKS;
        }
        let spine_ext = if self.pain_ticks > 0 {
            let mut e = vec![0.0f32; crate::organs::SPINE_N];
            for cell in e.iter_mut().take(crate::organs::SPINE_N_S) {
                *cell = PAIN_AMNT;
            }
            self.pain_ticks -= 1;
            e
        } else {
            vec![0.0f32; crate::organs::SPINE_N]
        };

        let world_in = crate::organs::WorldInputs {
            spine: spine_ext,
            midbrain: mid_ext,
            cortex: cortex_ext,
        };
        let spikes = self.organs.tick(&world_in);

        let mut cortex_regions = [0i32; N_REGIONS];
        for (r, sum) in cortex_regions.iter_mut().enumerate() {
            *sum = self.organs.cortex.region_spikes(&spikes.cortex, r);
        }
        let midbrain_total: i64 = spikes.midbrain.iter().map(|&x| x as i64).sum();
        let spine_m: i64 = spikes.spine[crate::organs::SPINE_N_S + crate::organs::SPINE_N_I..]
            .iter()
            .map(|&x| x as i64)
            .sum();

        // Кровь: моторная активность коры и дефицит глюкозы.
        let (ml, mr) = (cortex_regions[3], cortex_regions[4]);
        self.blood.loci[locus::RATE_MOTOR] = (ml + mr) as f32;
        {
            let deficit = (ORGAN_FED_THRESHOLD - self.blood.conc[chem::GLUCOSE]).max(0.0);
            self.blood.loci[locus::GLUCOSE_DEFICIT] = deficit;
        }

        // Рот и награда.
        let ate = self.world.try_eat(self.tick, &self.blood);
        if ate {
            // Полная порция: после еды дефицит обнуляется и наступает
            // сытость (покой), пока запас не выгорит.
            self.blood.ingest_food(FOOD_PORTION);
            self.organs.cortex.schedule_reward(1.0);
            self.rewards_pending += 1;
        }
        if self.world.last_wall_hit {
            self.blood.bump_wall();
        }

        self.blood.update();
        // Гейт скорости: молчащая кора → существо сидит на месте.
        let gate = ((ml + mr) as f32 / MOTOR_GATE_FULL).min(1.0);
        let sitting = (ml + mr) < 6;
        self.world.move_body_gated(ml, mr, &self.blood, gate);
        self.tick += 1;

        OrganTickObs {
            tick: self.tick,
            chemicals: self.blood.conc.clone(),
            cortex_regions,
            midbrain_spikes: midbrain_total,
            spine_m_spikes: spine_m,
            x: self.world.agent_x,
            y: self.world.agent_y,
            heading: self.world.heading,
            ate,
            wall_hit: self.world.last_wall_hit,
            alive: self.blood.organ.damage < 1.0,
            sitting,
        }
    }

    /// Средний вес пути коры (интроспекция обучения).
    pub fn cortex_path_weight(&self, a: usize, b: usize) -> (f32, usize) {
        self.organs.cortex.mean_path_weight(a, b)
    }
}

/// Наблюдение за тиком органной жизни.
pub struct OrganTickObs {
    pub tick: usize,
    pub chemicals: Vec<f32>,
    /// Спайки коры по регионам арены [SL, SR, I, ML, MR].
    pub cortex_regions: [i32; N_REGIONS],
    /// Суммарные спайки среднего мозга за тик.
    pub midbrain_spikes: i64,
    /// Моторные спайки спины за тик (рефлекс).
    pub spine_m_spikes: i64,
    pub x: f64,
    pub y: f64,
    pub heading: f64,
    pub ate: bool,
    pub wall_hit: bool,
    pub alive: bool,
    /// Моторы молчат (существо сидит).
    pub sitting: bool,
}

/// Отчёт жизни органного существа.
pub struct OrganLifeReport {
    pub eaten: usize,
    pub wall_hits: usize,
    pub reflex_fires: usize,
    pub mean_midbrain_quiet: f64,
    pub mean_midbrain_hungry: f64,
    pub w_sm_start: f32,
    pub w_sm_end: f32,
}

/// Полный прогон жизни (headless).
pub fn live_organ_life(sim: &mut OrganSim) -> OrganLifeReport {
    let (w_sm_start, _) = sim.cortex_path_weight(0, 3);
    let mut eaten = 0usize;
    let mut wall_hits = 0usize;
    let mut reflex_fires = 0usize;
    let mut hungry_sum = 0f64;
    let mut quiet_sum = 0f64;
    let mut hungry_n = 0usize;
    let mut quiet_n = 0usize;
    for _ in 0..SIM_TICKS {
        let obs = sim.step_tick();
        if !obs.alive {
            break;
        }
        if obs.ate {
            eaten += 1;
        }
        if obs.wall_hit {
            wall_hits += 1;
        }
        if obs.spine_m_spikes > 0 {
            reflex_fires += 1;
        }
        if sim.blood.conc[chem::HUNGER] > 40.0 {
            hungry_sum += obs.midbrain_spikes as f64;
            hungry_n += 1;
        } else {
            quiet_sum += obs.midbrain_spikes as f64;
            quiet_n += 1;
        }
    }
    let (w_sm_end, _) = sim.cortex_path_weight(0, 3);
    OrganLifeReport {
        eaten,
        wall_hits,
        reflex_fires,
        mean_midbrain_quiet: quiet_sum / quiet_n.max(1) as f64,
        mean_midbrain_hungry: hungry_sum / hungry_n.max(1) as f64,
        w_sm_start,
        w_sm_end,
    }
}
