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
//!-------------------------------------------------------------------------

use crate::body::metabolism::{chem, locus, Blood};
use crate::body::organs::{standard_spine_conn, Axon, OrganBrain, OrganId};
use crate::engine::neurons::PlasticityParams;
use crate::world::arena::{arena_gains, grow_agent, Genome, N_REGIONS, N_SL, N_SR};
use crate::world::free_life::{WorldSim, FOOD_PORTION};

pub use crate::world::free_life::{ARENA_HALF, SIM_TICKS};

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
#[allow(dead_code)]
const PAIN_TICKS: usize = 8;
/// Боль в спину: ток на сенсорный пул (HH-единицы).
#[allow(dead_code)]
const PAIN_AMNT: f32 = 15.0;
/// Порог сытости органной жизни: выше него дефицита нет и голод гаснет
/// (в free_life порог 150 — существо никогда не наедается и не сидит).
const ORGAN_FED_THRESHOLD: f32 = 100.0;
/// Тормоз покоя на ML/MR: сытое существо сидит (голод снимает тормоз).
const REST_INHIBIT: f32 = 0.6;
/// Полная скорость при сумме моторных спайков за тик ≥ этого порога.
#[allow(dead_code)]
const MOTOR_GATE_FULL: f32 = 24.0;

// ─── Температура тела («кипящий суп» спины) ────────────────────────────────
/// Окружающая среда (°C), к ней остывает тело.
#[allow(dead_code)]
const AMBIENT_TEMP: f32 = 20.0;
/// Стартовая температура тела.
const START_TEMP: f32 = 24.0;
/// Скорость охлаждения: доля разницы с средой в секунду.
#[allow(dead_code)]
const TEMP_COOL_PER_S: f32 = 0.18;
/// Нагрев от моторной активности: при длительной активной погоне
/// (motor_norm ≈ 1) равновесие ≈ 35°C — спина «закипает».
#[allow(dead_code)]
const TEMP_HEAT_PER_S: f32 = 2.7;
/// Боль дополнительно греет (мышечный тонус).
#[allow(dead_code)]
const PAIN_HEAT: f32 = 0.6;
/// Перегрев: выше этой температуры спина входит в спонтанные разряды
/// («горит»: дёргается без боли), ток на градус перегрева.
pub const FEVER_TEMP: f32 = 28.0;
const FEVER_DRIVE_PER_DEG: f32 = 1.2;

/// Пошаговая свободная жизнь на органном мозге.
pub struct OrganSim {
    /// Сид генома/мира — рост детерминирован, сохранять не нужно.
    pub seed: u64,
    pub organs: OrganBrain,
    pub blood: Blood,
    pub world: WorldSim,
    pub mind: Option<crate::mind::CreatureMind>,
    pub tick: usize,
    pub body_temp: f32,
    #[allow(dead_code)]
    pain_ticks: usize,
    /// Затухающая свежая боль (для канала pain разума).
    pub pain_recent: f32,
    #[allow(dead_code)]
    /// Были в лихорадке на прошлом тике (штраф только за вход).
    was_fever: bool,
    /// Шум сенсорики [0..1] — добавка к запаху, экзамен T3.
    pub sensory_noise: f32,
    /// Накопитель наград коры между дофаминовыми доставками.
    rewards_pending: usize,
    /// Гены правил Пифии по регионам (наследуются с мутацией).
    pub rule_genes: Vec<crate::engine::rules::RuleGene>,
    /// Врождённые рефлексы (прожигаются родильным сном).
    pub instincts: Vec<crate::engine::instincts::Instinct>,
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
            seed,
            world: WorldSim::new(seed ^ 0xB100),
            tick: 0,
            body_temp: START_TEMP,
            pain_ticks: 0,
            pain_recent: 0.0,
            was_fever: false,
            sensory_noise: 0.0,
            rewards_pending: 0,
            mind: None,
            rule_genes: Vec::new(),
            instincts: Vec::new(),
        }
    }

    /// Включить когнитивный разум (karmac SpikingBrain).
    pub fn with_mind(mut self, seed: u64) -> Self {
        self.mind = Some(crate::mind::CreatureMind::new(seed));
        self
    }

    /// Заменить кору на внешний коннектом (например sarDNA трансформер).
    pub fn with_external_cortex(mut self, conn: pillars_connectome::ConnectomeData) -> Self {
        let _ = self.organs.cortex.set_csr_values(&conn.export_values());
        self
    }

    /// Задать гены правил Пифии (наследству/лаборатории).
    pub fn with_rule_genes(mut self, genes: Vec<crate::engine::rules::RuleGene>) -> Self {
        self.rule_genes = genes;
        self
    }

    /// Задать врождённые инстинкты (наследству/лаборатории).
    pub fn with_instincts(mut self, instincts: Vec<crate::engine::instincts::Instinct>) -> Self {
        self.instincts = instincts;
        self
    }

    /// Родильный сон: прожечь унаследованные инстинкты в кору.
    /// Вызывается один раз при рождении — после этого рефлексы постоянны
    /// (снимок памяти коры обновлён, гомеостаз принял новую норму).
    pub fn birth_sleep(&mut self) {
        for inst in &self.instincts.clone() {
            inst.burn_into(&mut self.organs.cortex, 0.12);
        }
    }

    /// Съедено за текущую жизнь (для сохранения существа).
    pub fn eaten_total(&self) -> usize {
        self.rewards_pending
    }

    /// Шаг мозга с внешней сенсорикой — для экосистемы где запах приходит
    /// из общего мира. `sensory_noise` игнорируется (шум на стороне мира).
    pub fn brain_step_sensed(&mut self, left: f64, right: f64) -> BrainStepObs {
        self.brain_step_sensed_dual(left, right, 0.0, 0.0)
    }

    /// Раздельное обоняние: сладкая еда → первые 10 клеток каждого
    /// сенсорного пула, горькая → вторые 10. Существо различает вкусы
    /// с рождения (проводка), учится ОТНОШЕНИЮ к ним (пластичность).
    pub fn brain_step_sensed_dual(
        &mut self,
        sweet_l: f64,
        sweet_r: f64,
        bitter_l: f64,
        bitter_r: f64,
    ) -> BrainStepObs {
        let n_cortex = self.organs.cortex.n;
        let hunger = self.blood.conc[chem::HUNGER];
        let glucose = self.blood.conc[chem::GLUCOSE];
        let fatigue = self.blood.conc[chem::FATIGUE];
        self.pain_recent *= 0.97;

        // Средний мозг: голод → драйв.
        let mid_ext = vec![hunger_drive(hunger); 24];
        // Спина: лихорадка.
        let fever_drive = (self.body_temp - FEVER_TEMP).max(0.0) * FEVER_DRIVE_PER_DEG;
        let spine_ext = vec![fever_drive; crate::body::organs::SPINE_N_S];

        // Кора: тоник + внешняя сенсорика × sensor_gain.
        // Сладкое — перед сенсорных пулов, горькое — зад; форма
        // рецепторов детерминированная (хэш), канал свой у каждого вкуса.
        let mut cortex_ext = vec![self.blood.motor_tonic(); n_cortex];
        let sg = self.blood.sensor_gain();
        let shape = |seed: usize| 0.5 + 0.5 * crate::rng::hash01(seed) as f32;
        for (i, cell) in cortex_ext.iter_mut().enumerate().take(10) {
            *cell += sg * sweet_l as f32 * shape(i);
        }
        for (i, cell) in cortex_ext.iter_mut().enumerate().take(20).skip(10) {
            *cell += sg * bitter_l as f32 * shape(i + 500);
        }
        for (i, cell) in cortex_ext.iter_mut().enumerate().take(30).skip(20) {
            *cell += sg * sweet_r as f32 * shape(i + 999);
        }
        for (i, cell) in cortex_ext.iter_mut().enumerate().take(40).skip(30) {
            *cell += sg * bitter_r as f32 * shape(i + 1499);
        }

        // Тормоз покоя + разум.
        let mut rest = REST_INHIBIT * (1.0 - (hunger / 40.0).min(1.0));
        if let Some(mind) = &mut self.mind {
            let sated01 = (glucose / ORGAN_FED_THRESHOLD).clamp(0.0, 1.0);
            let channels = crate::mind::CreatureMind::channels(
                hunger / 255.0,
                fatigue / 255.0,
                sweet_l as f32,
                sweet_r as f32,
                self.pain_recent,
                sated01,
                ((self.body_temp - FEVER_TEMP).max(0.0) / 5.0).clamp(0.0, 1.0),
                (glucose / ORGAN_FED_THRESHOLD).clamp(0.0, 1.0),
            );
            let action = mind.decide(&channels);
            if action == crate::mind::CreatureAction::Rest {
                rest = REST_INHIBIT;
            }
        }
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

        // Шаг органов.
        let world_in = crate::body::organs::WorldInputs {
            spine: spine_ext,
            midbrain: mid_ext,
            cortex: cortex_ext,
        };
        let spike_data = self.organs.tick(&world_in);
        self.organs.set_spine_temperature(self.body_temp);

        let cortex_regions: [i32; 5] =
            std::array::from_fn(|r| self.organs.cortex.region_spikes(&spike_data.cortex, r));
        let midbrain_total: i64 = spike_data.midbrain.iter().map(|&x| x as i64).sum();
        let spine_m: i64 = spike_data.spine
            [crate::body::organs::SPINE_N_S + crate::body::organs::SPINE_N_I..]
            .iter()
            .map(|&x| x as i64)
            .sum();

        // Кровь.
        let (ml, mr) = (cortex_regions[3], cortex_regions[4]);
        self.blood.loci[dioma_metabolism_locus_rate_motor()] = (ml + mr) as f32;
        self.blood.loci[dioma_metabolism_locus_deficit()] =
            (ORGAN_FED_THRESHOLD - glucose).max(0.0);
        self.blood.update();

        BrainStepObs {
            cortex_regions,
            midbrain_spikes: midbrain_total,
            spine_m_spikes: spine_m,
            action: self.mind.as_ref().map(|m| m.last_action),
            sitting: (ml + mr) < 6,
        }
    }

    /// Награда разуму (фазичный дофамин).
    pub fn mind_reward(&mut self, r: f32) {
        if let Some(mind) = &mut self.mind {
            mind.reward(r);
        }
    }

    /// Затухание боли + фиксация удара.
    pub fn register_pain(&mut self) {
        self.pain_recent = (self.pain_recent + 0.6).min(1.0);
    }

    /// Один тик мира (25 мс).
    pub fn step_tick(&mut self) -> OrganTickObs {
        let obs = self.brain_step();
        // Движение телом (используем собственный мир).
        let (ml, mr) = (obs.cortex_regions[3], obs.cortex_regions[4]);
        self.world.move_body(ml, mr, &self.blood);
        self.world.regrow(self.tick);

        // Рот и награда.
        let ate = self.world.try_eat(self.tick, &self.blood);
        if ate {
            self.blood.ingest_food(FOOD_PORTION);
            self.organs.cortex.schedule_reward(1.0);
            self.rewards_pending += 1;
        }
        if self.world.last_wall_hit {
            self.blood.bump_wall();
        }

        self.tick += 1;
        OrganTickObs {
            tick: self.tick,
            chemicals: self.blood.conc.clone(),
            cortex_regions: obs.cortex_regions,
            midbrain_spikes: obs.midbrain_spikes,
            spine_m_spikes: obs.spine_m_spikes,
            x: self.world.agent_x,
            y: self.world.agent_y,
            heading: self.world.heading,
            ate,
            wall_hit: self.world.last_wall_hit,
            alive: self.blood.organ.damage < 1.0,
            sitting: obs.sitting,
            body_temp: self.body_temp,
            action: obs.action,
        }
    }

    /// Шаг мозга без движения тела — для экосистемы где позицию контролирует
    /// внешний мир. Возвращает наблюдение с кортикальными регионами.
    pub fn brain_step(&mut self) -> BrainStepObs {
        let n_cortex = self.organs.cortex.n;
        let hunger = self.blood.conc[chem::HUNGER];
        let glucose = self.blood.conc[chem::GLUCOSE];
        let fatigue = self.blood.conc[chem::FATIGUE];
        self.pain_recent *= 0.97;

        let mut cortex_ext = vec![self.blood.motor_tonic(); n_cortex];
        let sg = self.blood.sensor_gain();

        // Средний мозг.
        let mid_ext = vec![hunger_drive(hunger); 24];

        // Спина.
        let fever = (self.body_temp - FEVER_TEMP).max(0.0) * FEVER_DRIVE_PER_DEG;
        let spine_ext = vec![fever; crate::body::organs::SPINE_N_S];

        // Кора: тоник.
        for cell in cortex_ext.iter_mut() {
            *cell += 0.12; // тоник
        }
        // Сенсорика из собственного мира.
        let (left, right) = self.world.sensors();
        for (i, cell) in cortex_ext.iter_mut().enumerate().take(N_SL) {
            *cell += sg * left as f32 * (0.5 + 0.5 * crate::rng::hash01(i) as f32);
        }
        for (i, cell) in cortex_ext
            .iter_mut()
            .enumerate()
            .take(N_SL + N_SR)
            .skip(N_SL)
        {
            *cell += sg * right as f32 * (0.5 + 0.5 * crate::rng::hash01(i + 999) as f32);
        }

        // Тормоз покоя.
        let mut rest = REST_INHIBIT * (1.0 - (hunger / 40.0).min(1.0));
        if let Some(mind) = &mut self.mind {
            let sated01 = (glucose / ORGAN_FED_THRESHOLD).clamp(0.0, 1.0);
            let channels = crate::mind::CreatureMind::channels(
                hunger / 255.0,
                fatigue / 255.0,
                left as f32,
                right as f32,
                self.pain_recent,
                sated01,
                ((self.body_temp - FEVER_TEMP).max(0.0) / 5.0).clamp(0.0, 1.0),
                (glucose / ORGAN_FED_THRESHOLD).clamp(0.0, 1.0),
            );
            let action = mind.decide(&channels);
            if action == crate::mind::CreatureAction::Rest {
                rest = REST_INHIBIT;
            }
        }
        // Без разума — голодный тормоз работает сам.
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

        // Шаг органов.
        let world_in = crate::body::organs::WorldInputs {
            spine: spine_ext,
            midbrain: mid_ext,
            cortex: cortex_ext.clone(),
        };
        let _spike_data = self.organs.tick(&world_in);
        self.organs.set_spine_temperature(self.body_temp);

        // Кровь.
        let cortex_regions: [i32; 5] =
            std::array::from_fn(|r| self.organs.cortex.region_spikes(&_spike_data.cortex, r));
        let (ml, mr) = (cortex_regions[3], cortex_regions[4]);
        self.blood.loci[locus::RATE_MOTOR] = (ml + mr) as f32;
        self.blood.loci[locus::GLUCOSE_DEFICIT] = (ORGAN_FED_THRESHOLD - glucose).max(0.0);
        self.blood.update();

        let midbrain_total: i64 = _spike_data.midbrain.iter().map(|&x| x as i64).sum();
        let spine_m: i64 = _spike_data.spine
            [crate::body::organs::SPINE_N_S + crate::body::organs::SPINE_N_I..]
            .iter()
            .map(|&x| x as i64)
            .sum();

        BrainStepObs {
            cortex_regions,
            midbrain_spikes: midbrain_total,
            spine_m_spikes: spine_m,
            action: self.mind.as_ref().map(|m| m.last_action),
            sitting: (ml + mr) < 6,
        }
    }

    /// Средний вес пути коры (интроспекция обучения).
    pub fn cortex_path_weight(&self, a: usize, b: usize) -> (f32, usize) {
        self.organs.cortex.mean_path_weight(a, b)
    }
}

fn dioma_metabolism_locus_rate_motor() -> usize {
    crate::body::metabolism::locus::RATE_MOTOR
}
fn dioma_metabolism_locus_deficit() -> usize {
    crate::body::metabolism::locus::GLUCOSE_DEFICIT
}

/// Наблюдение за шагом мозга (без движения тела) — для экосистемы.
pub struct BrainStepObs {
    /// Спайки коры по регионам арены [SL, SR, I, ML, MR].
    pub cortex_regions: [i32; 5],
    pub midbrain_spikes: i64,
    pub spine_m_spikes: i64,
    pub action: Option<crate::mind::CreatureAction>,
    pub sitting: bool,
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
    /// Температура тела (°C) — она же температура спины.
    pub body_temp: f32,
    /// Решение разума на этом тике (None — разум выключен).
    pub action: Option<crate::mind::CreatureAction>,
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
