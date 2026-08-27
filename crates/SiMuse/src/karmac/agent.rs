//! -------------------------------------------------------------------------
//! Class:       agent
//! Purpose:     engine-to-engine bridge — «мир ↔ мозг»
//! Description: Тонкий слой общения движков. Арена (мир) и мозг (контроллер)
//!              не знают друг о друге: мир реализует `AgentWorld` (сенсоры +
//!              применение действий + конец эпизода), мозг реализует
//!              `AgentBrain` (шаг + награда + сброс + веса). Общий драйвер
//!              `run_episode` гоняет их вместе. Любой мозг (SpikingBrain
//!              сегодня, pillars `api::learn::brain` завтра) подключается к
//!              любой арене через ~20 строк адаптера — без правок арены.
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

use super::brain::{SpikingBrain, N_ACTIONS, N_CHANNELS};

/// Мозг как контроллер: всё, что нужно игре, чтобы говорить с мозгом.
pub trait AgentBrain {
    /// Сенсорный вектор в действие (индекс 0..n_actions).
    fn step(&mut self, sensors: &[f32]) -> usize;
    /// Награда за последнее действие (R-STDP / дофамин / негатив).
    fn reward(&mut self, reward: f32);
    /// Сброс динамики на границе эпизода (мембраны, следы, буферы).
    fn reset(&mut self);
    /// Плоские веса (наследственность, чекпоинты, heatmap).
    fn save(&self) -> Vec<f32>;
    /// Загрузить веса (длины должны совпадать).
    fn load(&mut self, weights: &[f32]);
    /// Текущие веса (диагностика, интроспекция).
    fn weights(&self) -> &[f32];
    fn n_sensors(&self) -> usize;
    fn n_actions(&self) -> usize;
}

/// Мир как окружение: всё, что нужно мозгу, чтобы говорить с миром.
pub trait AgentWorld {
    /// Сенсорный вектор текущего состояния (длина = n_sensors мозга).
    fn observe(&self) -> Vec<f32>;
    /// Применить действие, вернуть награду за этот шаг.
    fn act(&mut self, action: usize) -> f32;
    /// Эпизод окончен (смерть, финиш, таймаут).
    fn done(&self) -> bool;
}

/// Итог эпизода.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EpisodeReport {
    pub steps: usize,
    pub total_reward: f32,
}

/// Драйвер: мир и мозг вместе, один шаг = observe → step → act.
/// Награды копятся, мозг получает их после действия.
pub fn run_episode<W: AgentWorld, B: AgentBrain>(
    world: &mut W,
    brain: &mut B,
    max_steps: usize,
) -> EpisodeReport {
    let mut report = EpisodeReport::default();
    brain.reset();
    while !world.done() && report.steps < max_steps {
        let sensors = world.observe();
        let action = brain.step(&sensors);
        let reward = world.act(action);
        brain.reward(reward);
        report.total_reward += reward;
        report.steps += 1;
    }
    report
}

/// Адаптер: SpikingBrain (8 сенсорных каналов → 5 действий) как AgentBrain.
pub struct SpikingAgent(pub SpikingBrain);

impl AgentBrain for SpikingAgent {
    fn step(&mut self, sensors: &[f32]) -> usize {
        let mut ch = [0.0f32; N_CHANNELS];
        let n = sensors.len().min(N_CHANNELS);
        ch[..n].copy_from_slice(&sensors[..n]);
        self.0.step(&ch)
    }

    fn reward(&mut self, reward: f32) {
        self.0.apply_reward(reward);
    }

    fn reset(&mut self) {
        self.0.reset_dynamics();
    }

    fn save(&self) -> Vec<f32> {
        self.0.weights().to_vec()
    }

    fn load(&mut self, weights: &[f32]) {
        self.0.set_weights(weights);
    }

    fn weights(&self) -> &[f32] {
        self.0.weights()
    }

    fn n_sensors(&self) -> usize {
        N_CHANNELS
    }

    fn n_actions(&self) -> usize {
        N_ACTIONS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Детерминированный игрушечный мир: сенсоры фиксированы, действие 1
    /// даёт +1, остальные 0; эпизод длится 100 шагов.
    struct DummyWorld {
        t: usize,
    }
    impl AgentWorld for DummyWorld {
        fn observe(&self) -> Vec<f32> {
            vec![0.0; N_CHANNELS]
        }
        fn act(&mut self, action: usize) -> f32 {
            self.t += 1;
            if action == 1 {
                1.0
            } else {
                0.0
            }
        }
        fn done(&self) -> bool {
            self.t >= 100
        }
    }

    #[test]
    fn episode_drives_brain_and_world() {
        let mut world = DummyWorld { t: 0 };
        let mut brain = SpikingAgent(SpikingBrain::new());
        let before = brain.save();

        let report = run_episode(&mut world, &mut brain, 1000);
        assert_eq!(report.steps, 100);
        // Мозг что-то выбирает; веса после эпизода могли измениться,
        // но главное — интерфейс работает и награды дошли до мозга.
        assert!(report.total_reward >= 0.0);
        assert_eq!(brain.n_sensors(), N_CHANNELS);
        assert_eq!(brain.n_actions(), N_ACTIONS);

        // save/load — раундтрип.
        let saved = brain.save();
        brain.load(&saved);
        assert_eq!(brain.save(), saved);
        assert_eq!(brain.save().len(), before.len());
    }

    #[test]
    fn action_index_in_range() {
        let mut world = DummyWorld { t: 0 };
        let mut brain = SpikingAgent(SpikingBrain::new());
        let sensors = world.observe();
        for _ in 0..50 {
            let a = brain.step(&sensors);
            assert!(a < brain.n_actions(), "действие вне диапазона: {a}");
            brain.reward(world.act(a));
        }
    }
}
