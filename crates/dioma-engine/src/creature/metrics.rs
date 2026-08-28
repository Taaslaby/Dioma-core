//! Метрики и телеметрия для профилирования и отладки существ.

use crate::biology::metabolism::chem;
use crate::creature::Creature;

/// Метрики существа за один тик.
#[derive(Debug, Clone, Copy)]
pub struct CreatureMetrics {
    /// Количество спайков мозга за тик.
    pub brain_spikes: u32,
    /// Уровень глюкозы (0..255).
    pub glucose_level: f32,
    /// Уровень голода (0..255).
    pub hunger_level: f32,
    /// Уровень боли (0..255).
    pub pain_level: f32,
    /// Уровень награды/дофамина (0..255).
    pub reward_level: f32,
    /// Уровень усталости (0..255).
    pub fatigue_level: f32,
    /// Уровень серотонина (0..255).
    pub serotonin_level: f32,
    /// Уровень кортизола (стресс, 0..255).
    pub cortisol_level: f32,
    /// Повреждение мозга (0..1).
    pub brain_damage: f32,
    /// Текущий learning rate (с учётом бюджета).
    pub effective_learning_rate: f32,
    /// Бюджет пластичности (0..1).
    pub plasticity_budget: f32,
    /// Количество съеденной еды за жизнь.
    pub food_eaten: u32,
    /// Текущий тик существа.
    pub tick: usize,
}

impl Creature {
    /// Возвращает текущие метрики существа.
    ///
    /// # Примеры
    ///
    /// ```
    /// use dioma_engine::creature::{Creature, Sense};
    ///
    /// let mut creature = Creature::new(42);
    /// let sense = Sense::default();
    /// creature.tick(&sense);
    ///
    /// let metrics = creature.metrics();
    /// println!("Glucose: {:.1}", metrics.glucose_level);
    /// println!("Brain damage: {:.2}%", metrics.brain_damage * 100.0);
    /// ```
    pub fn metrics(&self) -> CreatureMetrics {
        CreatureMetrics {
            brain_spikes: 0, // Счётчик спайков требует модификации Brain::step
            glucose_level: self.blood.conc[chem::GLUCOSE],
            hunger_level: self.blood.conc[chem::HUNGER],
            pain_level: self.blood.conc[chem::PAIN],
            reward_level: self.blood.conc[chem::REWARD],
            fatigue_level: self.blood.conc[chem::FATIGUE],
            serotonin_level: self.blood.conc[chem::SEROTONIN],
            cortisol_level: self.blood.conc[chem::CORTISOL],
            brain_damage: self.blood.organ.damage,
            effective_learning_rate: self.brain.plasticity_params().lr
                * self.blood.plasticity_budget(),
            plasticity_budget: self.blood.plasticity_budget(),
            food_eaten: self.food_eaten,
            tick: self.blood.tick,
        }
    }

    /// Возвращает детальную информацию о состоянии крови.
    pub fn blood_state(&self) -> BloodState {
        BloodState {
            tick: self.blood.tick,
            concentrations: self.blood.conc.clone(),
            loci: self.blood.loci.clone(),
            damage: self.blood.organ.damage,
        }
    }

    /// Возвращает статистику мозга.
    pub fn brain_stats(&self) -> BrainStats {
        BrainStats {
            neuron_count: self.brain.n,
            region_count: self.brain.n_regions,
            plasticity_w_max: self.brain.plasticity_w_max(),
            step_count: self.brain.step_count(),
        }
    }
}

/// Детальное состояние крови.
#[derive(Debug, Clone)]
pub struct BloodState {
    pub tick: usize,
    pub concentrations: Vec<f32>,
    pub loci: Vec<f32>,
    pub damage: f32,
}

/// Статистика мозга.
#[derive(Debug, Clone, Copy)]
pub struct BrainStats {
    pub neuron_count: usize,
    pub region_count: usize,
    pub plasticity_w_max: f32,
    pub step_count: usize,
}

/// Событие для трейсинга.
#[derive(Debug, Clone)]
pub enum CreatureEvent {
    /// Существо съело еду.
    Ate { tick: usize, glucose_gained: f32 },
    /// Существо ударилось о стену.
    HitWall { tick: usize, pain: f32 },
    /// Мозг получил награду.
    Reward { tick: usize, amount: f32 },
    /// Существо спит (консолидация памяти).
    Sleep { tick: usize },
    /// Повреждение от голода.
    Starvation { tick: usize, damage: f32 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creature::{CreatureBuilder, Sense};

    #[test]
    fn metrics_basic() {
        let mut creature = CreatureBuilder::new().seed(42).build();
        let sense = Sense::default();
        
        creature.tick(&sense);
        let metrics = creature.metrics();
        
        assert!(metrics.glucose_level >= 0.0);
        assert!(metrics.brain_damage >= 0.0 && metrics.brain_damage <= 1.0);
        assert_eq!(metrics.food_eaten, 0);
    }

    #[test]
    fn blood_state() {
        let creature = CreatureBuilder::new().seed(123).build();
        let state = creature.blood_state();
        
        assert_eq!(state.concentrations.len(), chem::N);
        assert!(state.damage >= 0.0 && state.damage <= 1.0);
    }

    #[test]
    fn brain_stats() {
        let creature = CreatureBuilder::new().seed(456).build();
        let stats = creature.brain_stats();
        
        assert!(stats.neuron_count > 0);
        assert!(stats.region_count > 0);
    }
}
