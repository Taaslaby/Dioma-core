//! -------------------------------------------------------------------------
//! Filename:    src/free_life.rs
//! Class:       DebugSuite
//! Purpose:     free-life world: blood over grown brain
//! Description: arena brain + Blood v2 + food regrow; meals/hunger/fatigue/organ
//! 
//! damage metrics; robot control = hunger pinned to 255.
//! author: Atlasov anton
//! -------------------------------------------------------------------------
use crate::arena::{
    arena_brain, grow_agent, Genome, N_REGIONS, N_SL, N_SR, REG_ML, REG_MR,
};
use crate::brain::Brain;
use crate::metabolism::{locus, Blood};
use crate::rng::XorShift;

pub const N_FOOD: usize = 16;
pub const FOOD_REGROW_TICKS: usize = 400; // 10 с
pub const SIM_TICKS: usize = 4000; // 100 с жизни
const EAT_RADIUS: f64 = 14.0;
pub const ARENA_HALF: f64 = 200.0;
const FOOD_PORTION: f32 = 90.0;
/// Порог сытости для локуса дефицита глюкозы.
pub const GLUCOSE_FED_THRESHOLD: f32 = 150.0;
/// Пауза между приёмами пищи, разделяющая «меалы» (тиков).
pub const MEAL_GAP_TICKS: isize = 50;
/// Голод выше этой химии считается «голодным тиком».
pub const HUNGRY_CHEM: f32 = 40.0;

struct FoodItem {
    x: f64,
    y: f64,
    alive: bool,
    regrow_at: usize,
}

pub struct WorldSim {
    foods: Vec<FoodItem>,
    rng: XorShift,
    pub agent_x: f64,
    pub agent_y: f64,
    pub heading: f64,
    last_wall_hit: bool,
}

impl WorldSim {
    pub fn new(seed: u64) -> Self {
        let mut rng = XorShift::new(seed);
        let mut foods = Vec::new();
        for _ in 0..N_FOOD {
            foods.push(FoodItem {
                x: (rng.unit() * 2.0 - 1.0) * ARENA_HALF,
                y: (rng.unit() * 2.0 - 1.0) * ARENA_HALF,
                alive: true,
                regrow_at: 0,
            });
        }
        let heading = rng.unit() * 2.0 * std::f64::consts::PI;
        Self {
            foods,
            rng,
            agent_x: 0.0,
            agent_y: 0.0,
            heading,
            last_wall_hit: false,
        }
    }

    fn nearest_food(&self) -> Option<(f64, f64)> {
        self.foods
            .iter()
            .filter(|f| f.alive)
            .map(|f| {
                let (dx, dy) = (f.x - self.agent_x, f.y - self.agent_y);
                dx * dx + dy * dy
            })
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| (self.foods[i].x, self.foods[i].y))
    }

    fn sensors(&self) -> (f64, f64) {
        let Some((fx, fy)) = self.nearest_food() else {
            return (0.0, 0.0);
        };
        let (dx, dy) = (fx - self.agent_x, fy - self.agent_y);
        let dist = (dx * dx + dy * dy).sqrt();
        let prox = 60.0 / (60.0 + dist);
        let mut rel = dy.atan2(dx) - self.heading;
        while rel > std::f64::consts::PI {
            rel -= 2.0 * std::f64::consts::PI;
        }
        while rel < -std::f64::consts::PI {
            rel += 2.0 * std::f64::consts::PI;
        }
        (rel.sin().max(0.0) * prox, (-rel.sin()).max(0.0) * prox)
    }

    fn move_body(&mut self, spikes_l: i32, spikes_r: i32, blood: &Blood) {
        let d = (spikes_l - spikes_r) as f64;
        self.heading += blood.turn_gain() * d;
        let sp = blood.speed();
        self.agent_x += sp * self.heading.cos();
        self.agent_y += sp * self.heading.sin();
        self.last_wall_hit = false;
        if self.agent_x.abs() > ARENA_HALF {
            self.agent_x = self.agent_x.clamp(-ARENA_HALF, ARENA_HALF);
            self.heading = std::f64::consts::PI - self.heading;
            self.last_wall_hit = true;
        }
        if self.agent_y.abs() > ARENA_HALF {
            self.agent_y = self.agent_y.clamp(-ARENA_HALF, ARENA_HALF);
            self.heading = -self.heading;
            self.last_wall_hit = true;
        }
    }

    /// Рецептор «рта»: есть только голодный.
    fn try_eat(&mut self, tick: usize, blood: &Blood) -> bool {
        if !blood.wants_food() {
            return false;
        }
        for f in self.foods.iter_mut() {
            if !f.alive {
                continue;
            }
            let (dx, dy) = (f.x - self.agent_x, f.y - self.agent_y);
            if (dx * dx + dy * dy).sqrt() < EAT_RADIUS {
                f.alive = false;
                f.regrow_at = tick + FOOD_REGROW_TICKS;
                return true;
            }
        }
        false
    }

    /// Снапшот еды для отрисовки: (x, y, жива).
    pub fn foods_snapshot(&self) -> Vec<(f64, f64, bool)> {
        self.foods.iter().map(|f| (f.x, f.y, f.alive)).collect()
    }

    fn regrow(&mut self, tick: usize) {
        for f in self.foods.iter_mut() {
            if !f.alive && tick >= f.regrow_at {
                f.alive = true;
                f.x = (self.rng.unit() * 2.0 - 1.0) * ARENA_HALF;
                f.y = (self.rng.unit() * 2.0 - 1.0) * ARENA_HALF;
            }
        }
    }
}

pub struct LifeReport {
    pub eaten: usize,
    /// Число приёмов пищи (серий с паузой ≥50 тиков).
    pub meals: usize,
    /// Тиков, когда голод был выше порога «рта» (мера гомеостаза).
    pub hungry_ticks: usize,
    pub min_glucose: f32,
    pub mean_hunger: f32,
    pub peak_fatigue: f32,
    pub brain_damage: f32,
    pub dead: bool,
}

/// Наблюдение за один тик жизни (для UI и протоколов).
pub struct TickObs {
    pub tick: usize,
    /// Концентрации всех химий (0..255).
    pub chemicals: Vec<f32>,
    /// Спайки за тик по регионам мозга.
    pub region_spikes: Vec<i32>,
    pub x: f64,
    pub y: f64,
    pub heading: f64,
    pub ate: bool,
    pub wall_hit: bool,
    pub alive: bool,
}

/// Пошаговая свободная жизнь: UI ведёт тик-за-тиком, протоколы — циклом.
pub struct Sim {
    pub brain: Brain,
    pub blood: Blood,
    pub world: WorldSim,
    pub tick: usize,
}

impl Sim {
    pub fn new(robot: bool, seed: u64) -> Self {
        let genome = Genome::wildtype(seed);
        let grown = grow_agent(&genome);
        Self {
            brain: arena_brain(&grown, &genome),
            blood: Blood::creature(robot),
            world: WorldSim::new(seed ^ 0xB100),
            tick: 0,
        }
    }

    /// Один тик мира (25 мс). Возвращает наблюдение для отрисовки/метрик.
    pub fn step_tick(&mut self) -> TickObs {
        let n = self.brain.n;
        self.world.regrow(self.tick);
        let (left, right) = self.world.sensors();
        let mut ext = vec![self.blood.motor_tonic(); n];
        let sg = self.blood.sensor_gain();
        for (i, cell) in ext.iter_mut().enumerate().take(N_SL) {
            *cell += sg * left as f32 * (0.5 + 0.5 * crate::rng::hash01(i) as f32);
        }
        for (i, cell) in ext.iter_mut().enumerate().take(N_SL + N_SR).skip(N_SL) {
            *cell += sg * right as f32 * (0.5 + 0.5 * crate::rng::hash01(i + 999) as f32);
        }

        let mut spikes_by_region = vec![0i32; N_REGIONS];
        for _ in 0..self.brain.steps_per_tick() {
            let spikes = self.brain.step(&ext);
            for (r, acc) in spikes_by_region.iter_mut().enumerate() {
                *acc += self.brain.region_spikes(&spikes, r);
            }
        }
        let (ml, mr) = (spikes_by_region[REG_ML], spikes_by_region[REG_MR]);

        // Активность моторов → локус: орган (энергия) и утомление читают её.
        self.blood.loci[locus::RATE_MOTOR] = (ml + mr) as f32;
        // Дефицит глюкозы → локус (орган «печень»): голод растёт из него.
        {
            let deficit =
                (GLUCOSE_FED_THRESHOLD - self.blood.conc[crate::metabolism::chem::GLUCOSE]).max(0.0);
            self.blood.loci[locus::GLUCOSE_DEFICIT] = deficit;
        }

        let ate = self.world.try_eat(self.tick, &self.blood);
        if ate {
            self.blood.ingest_food(FOOD_PORTION);
            self.brain.schedule_reward(1.0);
        }
        if self.world.last_wall_hit {
            self.blood.bump_wall();
        }

        self.blood.update();
        self.world.move_body(ml, mr, &self.blood);
        self.tick += 1;

        let alive = self.blood.organ.damage < 1.0;
        TickObs {
            tick: self.tick,
            chemicals: self.blood.conc.clone(),
            region_spikes: spikes_by_region,
            x: self.world.agent_x,
            y: self.world.agent_y,
            heading: self.world.heading,
            ate,
            wall_hit: self.world.last_wall_hit,
            alive,
        }
    }
}

/// Полный прогон жизни (протокол poc-chemo).
pub fn live_life(sim: &mut Sim) -> LifeReport {
    let mut eaten = 0usize;
    let mut meals = 0usize;
    let mut last_eat_tick: isize = -1000;
    let mut hungry_ticks = 0usize;
    let mut min_glucose = f32::INFINITY;
    let mut hunger_sum = 0.0f32;
    let mut peak_fatigue = 0.0f32;

    for _ in 0..SIM_TICKS {
        let obs = sim.step_tick();
        if !obs.alive {
            break;
        }
        if obs.ate {
            eaten += 1;
            if (obs.tick as isize - last_eat_tick) > MEAL_GAP_TICKS + 1 {
                meals += 1;
            }
            last_eat_tick = obs.tick as isize;
        }
        use crate::metabolism::chem;
        min_glucose = min_glucose.min(sim.blood.conc[chem::GLUCOSE]);
        hunger_sum += sim.blood.conc[chem::HUNGER];
        peak_fatigue = peak_fatigue.max(sim.blood.conc[chem::FATIGUE]);
        if sim.blood.conc[chem::HUNGER] > HUNGRY_CHEM {
            hungry_ticks += 1;
        }
    }

    LifeReport {
        eaten,
        meals,
        hungry_ticks,
        min_glucose,
        mean_hunger: hunger_sum / SIM_TICKS as f32,
        peak_fatigue,
        brain_damage: sim.blood.organ.damage,
        dead: sim.blood.organ.damage >= 1.0,
    }
}

/// Прогнать одну жизнь (рост + мозг + кровь + мир).
pub fn run_life(robot: bool, seed: u64) -> LifeReport {
    let mut sim = Sim::new(robot, seed);
    live_life(&mut sim)
}

