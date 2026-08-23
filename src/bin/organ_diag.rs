//! -------------------------------------------------------------------------
//! Filename:    src/bin/organ_diag.rs
//! Class:       DebugSuite
//! Purpose:     диагностика голодной фазы органного существа
//! Description: разбивка по полосам голода: скорость, мотор, дифференциал
//!              поворотов, дистанция до еды — понять, что душит поиск.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::metabolism::chem;
use dioma::organ_life::{OrganSim, SIM_TICKS};

struct Band {
    n: usize,
    speed: f64,
    motor: f64,
    diff: f64,
    dist: f64,
    midbrain: f64,
    temp: f64,
    spine_m: f64,
}

/// Снимок полосы за тик.
struct Sample {
    speed: f64,
    motor: f64,
    diff: f64,
    dist: f64,
    midbrain: f64,
    temp: f32,
    spine_m: f64,
}

impl Band {
    fn new() -> Self {
        Self { n: 0, speed: 0.0, motor: 0.0, diff: 0.0, dist: 0.0, midbrain: 0.0, temp: 0.0, spine_m: 0.0 }
    }
    fn push(&mut self, s: Sample) {
        self.n += 1;
        self.speed += s.speed;
        self.motor += s.motor;
        self.diff += s.diff;
        self.dist += s.dist;
        self.midbrain += s.midbrain;
        self.temp += s.temp as f64;
        self.spine_m += s.spine_m;
    }
    fn report(&self, name: &str) {
        if self.n == 0 {
            println!("{name:>12}: тиков нет");
            return;
        }
        let n = self.n as f64;
        println!(
            "{name:>12}: n={:<5} speed={:.2} motor={:6.1} |ML-MR|={:5.2} dist={:6.1} mid={:6.1} T={:5.1}°C рефлекс={:5.1}",
            self.n,
            self.speed / n,
            self.motor / n,
            self.diff / n,
            self.dist / n,
            self.midbrain / n,
            self.temp / n,
            self.spine_m / n,
        );
    }
}

fn main() {
    let mut sim = OrganSim::new(false, 0x0000_AE11);
    let mut sated = Band::new();
    let mut peckish = Band::new();
    let mut hungry = Band::new();
    let mut starving = Band::new();
    let mut sit_ticks = 0usize;
    let mut sit_runs = 0usize;
    let mut in_sit = false;
    let mut longest_sit = 0usize;
    let mut cur_sit = 0usize;
    let mut lookarounds_during_sit = 0i64;
    let mut fever_ticks = 0usize;
    let mut max_temp = 0f32;

    for _ in 0..SIM_TICKS {
        let obs = sim.step_tick();
        if obs.body_temp > dioma::organ_life::FEVER_TEMP {
            fever_ticks += 1;
        }
        max_temp = max_temp.max(obs.body_temp);
        let h = sim.blood.conc[chem::HUNGER];
        let speed = sim.blood.speed();
        let ml = obs.cortex_regions[3] as f64;
        let mr = obs.cortex_regions[4] as f64;
        let dist = sim.world.nearest_food_dist();
        let band = if h < 15.0 {
            &mut sated
        } else if h < 40.0 {
            &mut peckish
        } else if h < 150.0 {
            &mut hungry
        } else {
            &mut starving
        };
        band.push(Sample {
            speed,
            motor: ml + mr,
            diff: (ml - mr).abs(),
            dist,
            midbrain: obs.midbrain_spikes as f64,
            temp: obs.body_temp,
            spine_m: obs.spine_m_spikes as f64,
        });

        if obs.sitting {
            sit_ticks += 1;
            cur_sit += 1;
            if !in_sit {
                sit_runs += 1;
                in_sit = true;
            }
            lookarounds_during_sit += obs.midbrain_spikes;
        } else {
            in_sit = false;
            longest_sit = longest_sit.max(cur_sit);
            cur_sit = 0;
        }
    }
    longest_sit = longest_sit.max(cur_sit);

    println!("полосы голода (химия):");
    sated.report("сытый <15");
    peckish.report("лёгкий 15-40");
    hungry.report("голодный 40-150");
    starving.report("голодуха >150");
    println!(
        "покой: тиков сидения {}/{} ({}%), сессий {}, самая длинная {} тиков",
        sit_ticks,
        SIM_TICKS,
        (sit_ticks as f64 / SIM_TICKS as f64 * 100.0) as usize,
        sit_runs,
        longest_sit
    );
    println!(
        "«изучает» в покое: средний мозг во время сидения = {:.1} сп/тик (спонтанные осмотры)",
        lookarounds_during_sit as f64 / sit_ticks.max(1) as f64
    );
    println!(
        "температура: лихорадочных тиков {fever_ticks}/{} (порог {}°C), максимум {:.1}°C",
        SIM_TICKS,
        dioma::organ_life::FEVER_TEMP,
        max_temp
    );
}
