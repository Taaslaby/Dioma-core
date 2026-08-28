//! Builder pattern для удобного создания существ с различными конфигурациями.

use crate::creature::{Creature, CreatureGenome};
use crate::brain::grow::{
    avoidance_layout, grow, grow_gpu, stage_b_layout, stage_c_lite_layout, Layout,
};
use crate::brain::neurons::PlasticityParams;

/// Раскладка мозга существа. По умолчанию — стадия C-lite (760 нейронов).
/// Стадия B (300) остаётся для эталонных бенчмарков/сравнений.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BrainLayout {
    /// Стадия C-lite: ядро + зрение 60×2 + кора 400/120 (760 нейронов).
    StageCLite,
    /// Стадия B: ядро + зрение 30×2 + ASSOC 90/30 (300 нейронов).
    StageB,
    /// Голое ядро avoidance (120 нейронов, регионы 0..6): крошечные
    /// существа для GPU-тестов паритета пула (быстрый рост).
    Avoidance,
}

impl Default for BrainLayout {
    fn default() -> Self {
        BrainLayout::StageB
    }
}

impl BrainLayout {
    fn layout(self) -> Layout {
        match self {
            BrainLayout::StageCLite => stage_c_lite_layout(),
            BrainLayout::StageB => stage_b_layout(),
            BrainLayout::Avoidance => avoidance_layout(),
        }
    }
}

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
    layout: BrainLayout,
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
            layout: BrainLayout::default(),
        }
    }

    /// Устанавливает сид для детерминированной генерации.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Устанавливает раскладку мозга (по умолчанию — стадия C-lite).
    pub fn layout(mut self, layout: BrainLayout) -> Self {
        self.layout = layout;
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
        let layout = self.layout.layout();

        if self.gpu_grown {
            // GPU рост + GPU динамика
            let grown = grow_gpu(&layout, &genome.steering, self.seed);
            Creature::from_genome_grown(genome, self.seed, grown)
        } else if self.use_gpu {
            // CPU рост + GPU динамика
            let grown = grow(&layout, &genome.steering, self.seed);
            Creature::assemble_gpu(grown, genome, self.seed)
        } else if let Some(params) = self.plasticity_params {
            // CPU с кастомными параметрами
            let grown = grow(&layout, &genome.steering, self.seed);
            Creature::assemble(grown, genome, self.seed, params)
        } else {
            // Дефолтное CPU существо.
            // ИСПРАВЛЕНИЕ perf-регрессии 42f30ed: здесь раньше стоял голый
            // `Creature::from_genome(genome, self.seed)` — выбранный layout
            // МОЛЧА ИГНОРИРОВАЛСЯ: from_genome всегда растит
            // stage_c_lite_layout (760 нейронов, 175k синапсов). До стадии
            // C это было бит-в-бит невидимо (from_genome растил stage_b —
            // единственную схему), после 42f30ed бенчмарочная строка
            // «CPU@300» мерила 760n-мозги → «одинаково с CPU@760». Растим
            // ВЫБРАННУЮ схему через from_genome_grown — для дефолтного
            // StageCLite путь бит-в-бит тот же (from_genome делает ровно
            // это), для явных layout'ов возвращается документированное
            // поведение («Стадия B — эталонные бенчмарки»).
            let grown = grow(&layout, &genome.steering, self.seed);
            Creature::from_genome_grown(genome, self.seed, grown)
        }
    }

    /// Собрать существо для батч-пула арены (`brain::gpu_pool::BrainPool`):
    /// мембрана — общий GPU-раннер, пластичность — CPU. Спавнить через
    /// `Arena::spawn_creature` на арене из `Arena::new_pooled`.
    pub fn build_pooled(self, pool: &mut crate::brain::gpu_pool::BrainPool) -> Creature {
        let genome = self.genome.unwrap_or_default();
        let layout = self.layout.layout();
        let grown = grow(&layout, &genome.steering, self.seed);
        Creature::assemble_pooled(grown, genome, self.seed, pool)
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
