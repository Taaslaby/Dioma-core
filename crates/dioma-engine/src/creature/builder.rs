//! Builder pattern для удобного создания существ с различными конфигурациями.

use crate::creature::{Creature, CreatureGenome};
use crate::engine::grow::{avoidance_layout, grow, grow_gpu};
use crate::engine::neurons::PlasticityParams;

/// Builder для создания существ с гибкой конфигурацией.
///
/// # Примеры
///
/// ```
/// use dioma_engine::creature::CreatureBuilder;
///
/// // Простое существо
/// let creature = CreatureBuilder::new()
///     .seed(42)
///     .build();
///
/// // Существо на GPU с кастомными параметрами
/// let creature = CreatureBuilder::new()
///     .seed(123)
///     .gpu(true)
///     .plasticity_lr(0.01)
///     .build();
/// ```
pub struct CreatureBuilder {
    seed: u64,
    genome: Option<CreatureGenome>,
    plasticity_params: Option<PlasticityParams>,
    use_gpu: bool,
    gpu_grown: bool,
}

impl Default for CreatureBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CreatureBuilder {
    /// Создаёт новый builder с дефолтными настройками.
    pub fn new() -> Self {
        Self {
            seed: 0,
            genome: None,
            plasticity_params: None,
            use_gpu: false,
            gpu_grown: false,
        }
    }

    /// Устанавливает сид для детерминированной генерации.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Устанавливает кастомный геном.
    pub fn genome(mut self, genome: CreatureGenome) -> Self {
        self.genome = Some(genome);
        self
    }

    /// Устанавливает параметры пластичности.
    pub fn plasticity_params(mut self, params: PlasticityParams) -> Self {
        self.plasticity_params = Some(params);
        self
    }

    /// Устанавливает learning rate для пластичности.
    pub fn plasticity_lr(mut self, lr: f32) -> Self {
        let mut params = self.plasticity_params.unwrap_or_default();
        params.lr = lr;
        self.plasticity_params = Some(params);
        self
    }

    /// Устанавливает максимальный вес синапса.
    pub fn plasticity_w_max(mut self, w_max: f32) -> Self {
        let mut params = self.plasticity_params.unwrap_or_default();
        params.w_max = w_max;
        self.plasticity_params = Some(params);
        self
    }

    /// Включает GPU-динамику для мозга.
    pub fn gpu(mut self, enabled: bool) -> Self {
        self.use_gpu = enabled;
        self
    }

    /// Включает GPU для роста коннектома (требует gpu = true).
    pub fn gpu_grown(mut self, enabled: bool) -> Self {
        self.gpu_grown = enabled;
        if enabled {
            self.use_gpu = true;
        }
        self
    }

    /// Собирает существо с заданными параметрами.
    pub fn build(self) -> Creature {
        let genome = self.genome.unwrap_or_default();

        if self.gpu_grown {
            // GPU рост + GPU динамика
            let grown = grow_gpu(&avoidance_layout(), &genome.steering, self.seed);
            Creature::from_genome_grown(genome, self.seed, grown)
        } else if self.use_gpu {
            // CPU рост + GPU динамика
            let grown = grow(&avoidance_layout(), &genome.steering, self.seed);
            Creature::assemble_gpu(grown, genome, self.seed)
        } else if let Some(params) = self.plasticity_params {
            // CPU с кастомными параметрами
            let grown = grow(&avoidance_layout(), &genome.steering, self.seed);
            Creature::assemble(grown, genome, self.seed, params)
        } else {
            // Дефолтное CPU существо
            Creature::from_genome(genome, self.seed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_default() {
        let creature = CreatureBuilder::new().seed(42).build();
        assert_eq!(creature.seed, 42);
    }

    #[test]
    fn builder_with_params() {
        let creature = CreatureBuilder::new()
            .seed(123)
            .plasticity_lr(0.01)
            .plasticity_w_max(32.0)
            .build();
        assert_eq!(creature.seed, 123);
    }

    #[test]
    fn builder_gpu() {
        let creature = CreatureBuilder::new()
            .seed(456)
            .gpu(true)
            .build();
        assert_eq!(creature.seed, 456);
    }
}
