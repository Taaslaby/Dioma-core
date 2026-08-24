//! -------------------------------------------------------------------------
//! Filename:    src/lib.rs
//! Class:       main
//! Purpose:     dioma game core library
//! Description: thin layer over taas-checkmate (path deps, no engine forks):
//!
//! grown spiking agents — grow/brain/blood/worlds; five headless PoC
//! protocols in src/bin (poc-reflex|arena|evolve|gnw|chemo).
//! author: Atlasov anton
//! -------------------------------------------------------------------------
pub mod rng;

pub mod brain;
pub mod grow;
pub mod plasticity;
pub mod pythia;
pub mod ecosystem;
pub mod exam;
pub mod heritage;
pub mod metabolism;
pub mod mind;
pub mod organ_life;
pub mod organs;
pub mod physiology;
pub mod ui;

pub mod arena;
pub mod delayed;
pub mod free_life;

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
    pub rule_genes: Vec<crate::pythia::RuleGene>,
}

impl Grown {
    /// Число регионов (по максимальному индексу).
    pub fn n_regions(&self) -> usize {
        self.region_of.iter().copied().max().map_or(0, |m| m + 1)
    }

    /// Ген региона; пустой список = нейтральный интегратор.
    pub fn gene_for(&self, region: usize) -> crate::pythia::RuleGene {
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
        (start, end)
    }
}
