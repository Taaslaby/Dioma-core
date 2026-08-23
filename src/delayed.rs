//! -------------------------------------------------------------------------
//! Filename:    src/delayed.rs
//! Class:       DebugSuite
//! Purpose:     delayed-response task (grown GNW/WM)
//! Description: beacon (agent frozen, looking) → sensory blackout → smell near food;
//!
//! only memory can hold the course. WS ignition gates WM write.
//! regions: 0=SL 1=SR 2=I 3=ML 4=MR 5=WSL 6=WSR 7=WML 8=WMR.
//! author: Atlasov anton
//! -------------------------------------------------------------------------

use crate::brain::{gains_from, Brain, PlasticityParams};
use crate::grow::{cognitive_layout, grow, Steering};
use crate::rng::XorShift;
use crate::Grown;

pub const SEED: u64 = 0x60_46_77; // "GNW"

pub const ARENA_HALF: f64 = 200.0;
const EAT_RADIUS: f64 = 14.0;
pub const EPISODE_TIMEOUT_TICKS: usize = 600;
const N_EPISODES: usize = 30;

const TONIC_DRIVE: f32 = 0.12;
const SENSOR_GAIN: f32 = 3.5;
const AGENT_SPEED: f64 = 2.0;
const TURN_GAIN: f64 = 0.06;

// Отложенный ответ.
const CUE_TICKS: usize = 30; // 0.75 с маяк (агент «смотрит», не двигается)
const CUE_PROX: f64 = 250.0;
const SMELL_RADIUS: f64 = 25.0;
const SMELL_PROX: f64 = 30.0;
const PATCH_A: (f64, f64) = (-55.0, 60.0);
const PATCH_B: (f64, f64) = (55.0, 60.0);
const MEMORY_RESET_STEPS: usize = 150;
/// Нисходящая проекция памяти: доворот по асимметрии WM_L/WM_R
/// (stand-in NMDA-интегратора: current-based LIF аттрактор недолговечен).
const WM_TURN_GAIN: f64 = 0.004;

pub const REG_WSL: usize = 5;
pub const REG_WSR: usize = 6;
pub const REG_WML: usize = 7;
pub const REG_WMR: usize = 8;

/// Гейны путей когнитивной схемы.
pub fn cognitive_gains() -> Vec<Vec<f32>> {
    gains_from(
        9,
        &[
            // Сенсоры → рефлекс / интернейроны / workspace (свою сторону).
            (0, 3, 0.20),
            (0, 4, 0.20),
            (1, 3, 0.20),
            (1, 4, 0.20),
            (0, 2, 0.25),
            (1, 2, 0.25),
            (0, 5, 0.25),
            (1, 6, 0.25),
            (0, 6, 0.10),
            (1, 5, 0.10),
            // Workspace → запись, broadcast, рекуррентность.
            (5, 7, 0.8),
            (6, 8, 0.8),
            (5, 8, 0.05),
            (6, 7, 0.05),
            (5, 3, 0.15),
            (5, 4, 0.15),
            (6, 3, 0.15),
            (6, 4, 0.15),
            (5, 5, 0.15),
            (6, 6, 0.15),
            (5, 6, 0.15),
            (6, 5, 0.15),
            // Память: аттрактор, чтение мотора, re-entry.
            // Самозамыкание ослаблено (0.32→0.24): полный аттрактор не
            // отпускает никогда — фон не возвращается в тишину, контраст
            // ignition пропадает. 0.24 держит плато секунды, но гаснет.
            (7, 7, 0.24),
            (8, 8, 0.24),
            (7, 8, 0.08),
            (8, 7, 0.08),
            (7, 3, 0.32),
            (7, 4, 0.32),
            (8, 3, 0.32),
            (8, 4, 0.32),
            (7, 5, 0.06),
            (7, 6, 0.06),
            (8, 5, 0.06),
            (8, 6, 0.06),
            // Интернейроны → все.
            (2, 3, 0.6),
            (2, 4, 0.6),
            (2, 5, 0.6),
            (2, 6, 0.6),
        ],
    )
}

pub fn grow_cognitive(seed: u64) -> Grown {
    grow(&cognitive_layout(), &Steering::default(), seed)
}

pub fn cognitive_brain(grown: &Grown) -> Brain {
    Brain::new(grown, cognitive_gains(), PlasticityParams::default())
}

/// Вырезанные WS/WM: контроль-абляция (те же веса, минус когниция).
pub fn ablated_edges(grown: &Grown) -> Vec<(usize, usize, f32)> {
    grown
        .edges
        .iter()
        .filter(|&&(pre, post, _)| grown.region_of[pre] < 5 && grown.region_of[post] < 5)
        .copied()
        .collect()
}

// ─── Тело ───────────────────────────────────────────────────────────────────
pub struct Agent {
    pub x: f64,
    pub y: f64,
    pub heading: f64,
    pub food_x: f64,
    pub food_y: f64,
    rng: XorShift,
}

impl Agent {
    pub fn new(seed: u64) -> Self {
        let mut rng = XorShift::new(seed);
        let mut a = Agent {
            x: 0.0,
            y: 0.0,
            heading: rng.unit() * 2.0 * std::f64::consts::PI,
            food_x: 0.0,
            food_y: 0.0,
            rng,
        };
        a.spawn_food();
        a
    }

    fn spawn_food(&mut self) {
        if self.rng.unit() < 0.5 {
            (self.food_x, self.food_y) = PATCH_A;
        } else {
            (self.food_x, self.food_y) = PATCH_B;
        }
    }

    fn sensors(&self, prox_range: f64) -> (f64, f64) {
        let (dx, dy) = (self.food_x - self.x, self.food_y - self.y);
        let dist = (dx * dx + dy * dy).sqrt();
        let mut rel = dy.atan2(dx) - self.heading;
        while rel > std::f64::consts::PI {
            rel -= 2.0 * std::f64::consts::PI;
        }
        while rel < -std::f64::consts::PI {
            rel += 2.0 * std::f64::consts::PI;
        }
        let prox = prox_range / (prox_range + dist);
        (rel.sin().max(0.0) * prox, (-rel.sin()).max(0.0) * prox)
    }

    fn move_turn(&mut self, spikes_l: i32, spikes_r: i32, extra_turn: f64) {
        let d = (spikes_l - spikes_r) as f64;
        self.heading += TURN_GAIN * d + extra_turn;
        self.x += AGENT_SPEED * self.heading.cos();
        self.y += AGENT_SPEED * self.heading.sin();
        if self.x.abs() > ARENA_HALF {
            self.x = self.x.clamp(-ARENA_HALF, ARENA_HALF);
            self.heading = std::f64::consts::PI - self.heading;
        }
        if self.y.abs() > ARENA_HALF {
            self.y = self.y.clamp(-ARENA_HALF, ARENA_HALF);
            self.heading = -self.heading;
        }
    }

    fn at_food(&self) -> bool {
        let (dx, dy) = (self.food_x - self.x, self.food_y - self.y);
        (dx * dx + dy * dy).sqrt() < EAT_RADIUS
    }
}

/// Статистика эпизода.
pub struct EpisodeStats {
    pub ticks: usize,
    /// Спайков WS за тик во время маяка (ignition).
    pub ws_rate_cue: f32,
    /// Спайков WM за тик после маяка (удержание памяти).
    pub wm_rate_delay: f32,
}

pub fn run_episode(brain: &mut Brain, agent: &mut Agent, dopamine_on: bool) -> EpisodeStats {
    let n = brain.n;
    // Диапазоны регионов из схемы (не магические числа).
    let (sl0, sl1) = brain.region_bounds(0);
    let (sr0, sr1) = brain.region_bounds(1);
    let (ml0, ml1) = brain.region_bounds(3);
    let (mr0, mr1) = brain.region_bounds(4);
    // Сброс памяти между пробами: гиперполяризация WS/WM.
    brain.hyperpolarize(
        &[REG_WSL, REG_WSR, REG_WML, REG_WMR],
        MEMORY_RESET_STEPS,
        -8.0,
    );

    let mut ws_cue = 0i32;
    let mut wm_delay = 0i32;
    let mut delay_ticks = 0usize;
    let mut eaten_tick = EPISODE_TIMEOUT_TICKS;

    for tick in 0..EPISODE_TIMEOUT_TICKS {
        let cue_on = tick < CUE_TICKS;
        // Маяк → полная слепота → «запах» вблизи.
        let dist = {
            let (dx, dy) = (agent.food_x - agent.x, agent.food_y - agent.y);
            (dx * dx + dy * dy).sqrt()
        };
        let prox = if cue_on {
            CUE_PROX
        } else if dist < SMELL_RADIUS {
            SMELL_PROX
        } else {
            1e-6
        };
        let (left, right) = agent.sensors(prox);
        let mut ext = vec![TONIC_DRIVE; n];
        for (i, cell) in ext.iter_mut().enumerate().take(sl1).skip(sl0) {
            *cell += SENSOR_GAIN * left as f32 * (0.5 + 0.5 * crate::rng::hash01(i) as f32);
        }
        for (i, cell) in ext.iter_mut().enumerate().take(sr1).skip(sr0) {
            *cell += SENSOR_GAIN * right as f32 * (0.5 + 0.5 * crate::rng::hash01(i + 999) as f32);
        }

        let mut ml = 0i32;
        let mut mr = 0i32;
        let mut wm_l = 0i32;
        let mut wm_r = 0i32;
        let mut ws = 0i32;
        for _ in 0..brain.steps_per_tick() {
            let spikes = brain.step(&ext);
            ml += spikes[ml0..ml1].iter().sum::<i32>();
            mr += spikes[mr0..mr1].iter().sum::<i32>();
            ws += brain.region_spikes(&spikes, REG_WSL) + brain.region_spikes(&spikes, REG_WSR);
            wm_l += brain.region_spikes(&spikes, REG_WML);
            wm_r += brain.region_spikes(&spikes, REG_WMR);
        }
        if cue_on {
            ws_cue += ws;
        } else {
            wm_delay += wm_l + wm_r;
            delay_ticks += 1;
        }

        // Во время маяка агент «смотрит» (движение заморожено). Поворот =
        // моторный рефлекс + нисходящая проекция памяти (WM).
        if !cue_on {
            let memory_turn = WM_TURN_GAIN * (wm_l - wm_r) as f64;
            agent.move_turn(ml, mr, memory_turn);
        }

        if !cue_on && agent.at_food() {
            if dopamine_on {
                brain.schedule_reward(1.0);
            }
            eaten_tick = tick + 1;
            break;
        }
    }

    EpisodeStats {
        ticks: eaten_tick,
        ws_rate_cue: ws_cue as f32 / CUE_TICKS.max(1) as f32,
        wm_rate_delay: wm_delay as f32 / delay_ticks.max(1) as f32,
    }
}

/// Протокол: полный мозг против абляции, одинаковые миры.
pub fn run_protocol_n(edges: &[(usize, usize, f32)], n_episodes: usize) -> Vec<EpisodeStats> {
    let grown_stub = Grown {
        edges: edges.to_vec(),
        region_of: cognitive_layout().region_of(),
        nt: {
            let layout = cognitive_layout();
            layout
                .regions
                .iter()
                .flat_map(|r| {
                    std::iter::repeat_n(if r.inhibitory { -1.0f32 } else { 1.0 }, r.count)
                })
                .collect()
        },
        growth_s: 0.0,
    };
    let mut brain = cognitive_brain(&grown_stub);
    let mut stats = Vec::with_capacity(n_episodes);
    for k in 0..n_episodes {
        let mut agent = Agent::new(SEED.wrapping_add(k as u64));
        stats.push(run_episode(&mut brain, &mut agent, false));
    }
    stats
}

pub fn run_full_protocol() -> Vec<EpisodeStats> {
    let grown = grow_cognitive(SEED);
    run_protocol_n(&grown.edges, N_EPISODES)
}

pub fn run_ablated_protocol() -> Vec<EpisodeStats> {
    let grown = grow_cognitive(SEED);
    run_protocol_n(&ablated_edges(&grown), N_EPISODES)
}

