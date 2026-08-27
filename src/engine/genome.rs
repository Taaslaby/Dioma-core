//! -------------------------------------------------------------------------
//! Filename:    src/engine/genome.rs
//! Class:       neuworks
//! Purpose:     универсальный результат роста — геном существа
//! Description: Grown = рёбра коннектома + регионы + типы нейронов +
//!              гены правил Пифии. Всё, что наследуется и мутирует.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use crate::engine::rules::RuleGene;

/// Универсальный результат роста.
#[derive(Clone, Debug)]
pub struct Grown {
    /// (pre, post, сырой вес) — направленные синапсы.
    pub edges: Vec<(usize, usize, f32)>,
    /// Регион каждого нейрона (индекс в layout).
    pub region_of: Vec<usize>,
    /// +1 возбуждающий, −1 тормозной.
    pub nt: Vec<f32>,
    pub growth_s: f64,
    /// Ген правила динамики на регион (Пифия). Пустой = нейтральный
    /// интегратор везде; наследуется с мутацией вместе с весами.
    pub rule_genes: Vec<RuleGene>,
    /// Врождённые рефлексы, прожигаемые во сне при рождении.
    pub instincts: Vec<crate::engine::instincts::Instinct>,
}

impl Grown {
    /// Число регионов (по максимальному индексу).
    pub fn n_regions(&self) -> usize {
        self.region_of.iter().copied().max().map_or(0, |m| m + 1)
    }

    /// Ген региона; пустой список = нейтральный интегратор.
    pub fn gene_for(&self, region: usize) -> RuleGene {
        self.rule_genes.get(region).copied().unwrap_or_default()
    }

    /// Границы нейронов региона [start, end). Регионы в region_of идут
    /// сплошными блоками по порядку добавления (так растит grow()).
    pub fn region_bounds(&self, region: usize) -> (usize, usize) {
        let start = self
            .region_of
            .iter()
            .position(|&r| r == region)
            .unwrap_or(0);
        let end = self
            .region_of
            .iter()
            .rposition(|&r| r == region)
            .map_or(start, |p| p + 1);
        // Инвариант контигуальности: между первым и последним нейроном региона
        // не должно быть чужих регионов, иначе [start, end) не настоящие границы.
        debug_assert!(
            self.region_of[start..end].iter().all(|&r| r == region),
            "region_of должен идти сплошными блоками: нарушено у региона {region}"
        );
        (start, end)
    }
}
