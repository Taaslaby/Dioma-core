//! -------------------------------------------------------------------------
//! Filename:    src/exam.rs
//! Class:       DebugSuite
//! Purpose:     creature exam — настоящая сила, а не «пожрал»
//! Description: стандартизированная батарея: T1 поиск (медиана тиков до еды,
//!              эффективность пути), T2 экономика (еды на глюкозу),
//!              T3 помехи (шум сенсорики), T4 характер (профиль действий).
//! author: Atlasov anton
// -------------------------------------------------------------------------
use crate::metabolism::chem;
use crate::mind::CreatureAction;
use crate::organ_life::{OrganSim, SIM_TICKS};

pub const EPISODE_TICKS: usize = 600;

/// Результат одного эпизода поиска.
struct Episode {
    ticks_to_food: usize, // usize::MAX = не нашёл
    path_len: f64,
    net_dist: f64,
    glucose_burned: f32,
    eaten: usize,
}

/// Инструментированный эпизод: путь + глюкоза + еда.
fn instrumented_episode(sim: &mut OrganSim) -> Episode {
    let g0 = sim.blood.conc[chem::GLUCOSE];
    let (x0, y0) = (sim.world.agent_x, sim.world.agent_y);
    let mut path = 0.0f64;
    let mut px = x0;
    let mut py = y0;
    let mut ticks_to_food = usize::MAX;
    let mut eaten = 0usize;

    for t in 0..EPISODE_TICKS {
        let obs = sim.step_tick();
        path += ((obs.x - px).powi(2) + (obs.y - py).powi(2)).sqrt();
        px = obs.x;
        py = obs.y;
        if obs.ate {
            eaten += 1;
            if ticks_to_food == usize::MAX {
                ticks_to_food = t;
            }
        }
    }
    let glucose_burned =
        (g0 + eaten as f32 * 90.0 - sim.blood.conc[chem::GLUCOSE]).max(0.0);
    Episode {
        ticks_to_food,
        path_len: path,
        net_dist: ((px - x0).powi(2) + (py - y0).powi(2)).sqrt(),
        glucose_burned,
        eaten,
    }
}

/// Отчёт экзамена.
#[derive(Debug, Clone, Default)]
pub struct ExamReport {
    /// Медиана тиков до первой еды (usize::MAX если не находит).
    pub seek_median_ticks: usize,
    /// Съедено за все эпизоды.
    pub meals_total: usize,
    /// Прямой путь / пройденный путь [0..1], 1 = идеально прямо.
    pub path_efficiency: f64,
    /// Порций еды на 100 единиц сожжённой глюкозы.
    pub economy: f64,
    /// Поиск при шуме сенсорики / чистый поиск.
    pub noise_robustness: f64,
    /// Доля тиков в покое.
    pub sit_pct: f64,
    /// Доля тиков активного поиска.
    pub seek_pct: f64,
    /// Лихорадочных тиков.
    pub fever_ticks: usize,
}

impl ExamReport {
    pub fn print(&self, name: &str) {
        println!("── {name} ──");
        println!(
            "  поиск: медиана {} тиков · еды всего {}",
            self.seek_median_ticks, self.meals_total
        );
        println!(
            "  эффективность пути {:.2} · экономика {:.2} порций/100глк",
            self.path_efficiency, self.economy
        );
        println!(
            "  помехоустойчивость {:.2} · сидел {:.0}% · искал {:.0}% · жар {}",
            self.noise_robustness, self.sit_pct, self.seek_pct, self.fever_ticks
        );
    }

    /// Сводный балл силы [0..100]: поиск + эффективность + экономика +
    /// устойчивость, штраф за беспрерывный поиск без отдыха.
    pub fn score(&self) -> f64 {
        let seek = if self.seek_median_ticks == usize::MAX {
            0.0
        } else {
            (600.0 - self.seek_median_ticks as f64).max(0.0) / 600.0 * 40.0
        };
        let eff = self.path_efficiency.clamp(0.0, 1.0) * 20.0;
        let eco = (self.economy / 3.0).min(1.0) * 20.0;
        let robust = self.noise_robustness.clamp(0.0, 1.5).min(1.0) * 15.0;
        // Штраф за жизнь «без отдыха»: выгорание — не сила.
        let balance = 1.0 - (self.sit_pct / 100.0 - 0.25).abs().min(1.0);
        seek + eff + eco + robust + balance * 5.0
    }
}

/// Конфигурация существа для экзамена.
pub struct ExamConfig {
    pub name: &'static str,
    pub seed: u64,
    pub with_mind: bool,
    pub sensory_noise: f32,
    pub episodes: usize,
}

/// Прогнать экзамен.
pub fn run_exam(cfg: &ExamConfig) -> ExamReport {
    let mut sim = if cfg.with_mind {
        OrganSim::new(false, cfg.seed).with_mind(cfg.seed ^ 0x5EED)
    } else {
        OrganSim::new(false, cfg.seed)
    };
    sim.sensory_noise = cfg.sensory_noise;

    let mut first_seeks: Vec<usize> = Vec::new();
    let mut meals_total = 0usize;
    let mut path_sum = 0.0f64;
    let mut net_sum = 0.0f64;
    let mut burned_sum = 0.0f32;

    let sim_ref: &mut OrganSim = &mut sim;
    for _ in 0..cfg.episodes {
        let ep = instrumented_episode(sim_ref);
        if ep.ticks_to_food != usize::MAX {
            first_seeks.push(ep.ticks_to_food);
        }
        meals_total += ep.eaten;
        path_sum += ep.path_len;
        net_sum += ep.net_dist;
        burned_sum += ep.glucose_burned;
    }

    first_seeks.sort_unstable();
    let median = if first_seeks.is_empty() {
        usize::MAX
    } else {
        first_seeks[first_seeks.len() / 2]
    };

    let efficiency = if path_sum > 1e-9 { net_sum / path_sum } else { 0.0 };
    let economy = if burned_sum > 1e-6 {
        meals_total as f64 / (burned_sum as f64 / 100.0)
    } else {
        0.0
    };

    // T3: шумовые эпизоды — та же жизнь, сенсорика грязная.
    let mut noisy = if cfg.with_mind {
        OrganSim::new(false, cfg.seed).with_mind(cfg.seed ^ 0x5EED)
    } else {
        OrganSim::new(false, cfg.seed)
    };
    noisy.sensory_noise = cfg.sensory_noise.max(0.5);
    let mut noisy_ok = 0usize;
    for _ in 0..cfg.episodes {
        for _ in 0..EPISODE_TICKS {
            if noisy.step_tick().ate {
                noisy_ok += 1;
                break;
            }
        }
    }
    let noise_robustness = noisy_ok as f64 / cfg.episodes.max(1) as f64;

    // T4: характер — профиль действий за шумовой прогон (он уже прошёл).
    let mut probe = if cfg.with_mind {
        OrganSim::new(false, cfg.seed).with_mind(cfg.seed ^ 0x5EED)
    } else {
        OrganSim::new(false, cfg.seed)
    };
    let mut sit = 0usize;
    let mut seek = 0usize;
    let mut total = 0usize;
    let mut fever_ticks_report = 0usize;
    for _ in 0..SIM_TICKS.min(2000) {
        let obs = probe.step_tick();
        total += 1;
        match obs.action {
            Some(CreatureAction::Rest) => sit += 1,
            Some(CreatureAction::SeekFood) | Some(CreatureAction::Approach) => seek += 1,
            _ => {}
        }
        if obs.body_temp > crate::organ_life::FEVER_TEMP {
            fever_ticks_report += 1;
        }
    }

    ExamReport {
        seek_median_ticks: median,
        meals_total,
        path_efficiency: efficiency,
        economy,
        noise_robustness,
        sit_pct: sit as f64 / total.max(1) as f64 * 100.0,
        seek_pct: seek as f64 / total.max(1) as f64 * 100.0,
        fever_ticks: fever_ticks_report,
    }
}
