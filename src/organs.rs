//! -------------------------------------------------------------------------
//! Filename:    src/organs.rs
//! Class:       organs
//! Purpose:     organ brain — mixed neuron models per subsystem
//! Description: HH спина (быстрый точный рефлекс) + HAAP средний мозг
//!              (мотивация/настроение) + LIF-кора (пластичность, «словарь»).
//!              Органы — отдельные BrainRunner со своими dt; связь —
//!              межорганные аксоны с задержкой в 1 мировой тик (25 мс):
//!              спайки пре-группы за тик × вес приходят током в целевой
//!              орган на следующем тике.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};

use crate::brain::{Brain, PlasticityParams};
use crate::Grown;

/// Мировой тик (мс) — общий для всех органов.
pub const TICK_MS: f32 = 25.0;
/// Шаг интеграции HH и HAAP.
const DT_FINE: f32 = 0.05;
/// Подшагов тонкого органа на тик.
const SUBSTEPS_FINE: usize = 500;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrganId {
    Spine = 0,
    Midbrain = 1,
    Cortex = 2,
}

impl OrganId {
    pub fn all() -> [OrganId; 3] {
        [OrganId::Spine, OrganId::Midbrain, OrganId::Cortex]
    }

    fn substeps(self) -> usize {
        match self {
            OrganId::Cortex => crate::brain::BRAIN_STEPS_PER_TICK,
            _ => SUBSTEPS_FINE,
        }
    }
}

/// Межорганный аксон: группа пре-нейронов → ток в пост-нейроны.
///
/// За тик суммируются спайки `pre`; на СЛЕДУЮЩИЙ тик каждый пост-нейрон
/// получает `sum × weight_per_spike` во внешний ток. Один тик проводки —
/// честная аксонная задержка между органами.
#[derive(Clone)]
pub struct Axon {
    pub from: OrganId,
    pub pre: Vec<usize>,
    pub to: OrganId,
    pub post: Vec<usize>,
    pub weight_per_spike: f32,
}

/// Внешние входы мира на один тик (в единицах каждого органа).
#[derive(Default, Clone)]
pub struct WorldInputs {
    pub spine: Vec<f32>,
    pub midbrain: Vec<f32>,
    /// Кора: диома-Brain сам масштабирует (LIF-единицы).
    pub cortex: Vec<f32>,
}

/// Спайковая активность органов за тик (сумма по подшагам).
#[derive(Default, Clone)]
pub struct OrganSpikes {
    pub spine: Vec<i32>,
    pub midbrain: Vec<i32>,
    pub cortex: Vec<i32>,
}

impl OrganSpikes {
    pub fn of(&self, o: OrganId) -> &[i32] {
        match o {
            OrganId::Spine => &self.spine,
            OrganId::Midbrain => &self.midbrain,
            OrganId::Cortex => &self.cortex,
        }
    }

    fn total(&self, o: OrganId) -> i64 {
        self.of(o).iter().map(|&s| s as i64).sum()
    }
}

pub struct OrganBrain {
    pub spine: BrainRunner,
    pub midbrain: BrainRunner,
    /// Кора — полный диома-Brain: LIF-α + дофаминовая пластичность.
    pub cortex: Brain,
    axons: Vec<Axon>,
    /// Ток, доставляемый на текущем тике (от спайков прошлого).
    carry: [Vec<f32>; 3],
}

impl OrganBrain {
    /// Полная сборка.
    ///
    /// * `spine_conn` — CSR спинного мозга (HH, канонические параметры);
    /// * `cortex_grown` — схема коры (LIF+STDP), регионы/рёбра внутри;
    /// * `cortex_gains` — гейн-матрица коры (регионы × регионы);
    /// * `cortex_params` — параметры пластичности коры;
    /// * `midbrain_n` — размер среднего мозга (пейсмекерный микс 1/3).
    pub fn new(
        spine_conn: ConnectomeData,
        cortex_grown: Grown,
        cortex_gains: Vec<Vec<f32>>,
        cortex_params: PlasticityParams,
        midbrain_n: usize,
    ) -> Self {
        let spine =
            BrainRunner::new(spine_conn, BrainConfig::hh_synaptic(DT_FINE))
                .expect("spine runner");
        let mut midbrain = BrainRunner::new(
            ConnectomeData::new(
                vec![0; midbrain_n + 1],
                vec![],
                vec![],
                vec![1.0; midbrain_n],
                vec![],
            ),
            BrainConfig::haap_synaptic(DT_FINE, 12.0),
        )
        .expect("midbrain runner");
        let cols = crate::physiology::pacemaker_mix_columns(
            midbrain_n,
            &crate::physiology::PhysioGenes::default(),
        );
        crate::physiology::apply_haap_columns(&mut midbrain, &cols);

        let cortex = Brain::new(&cortex_grown, cortex_gains, cortex_params);
        Self {
            spine,
            midbrain,
            cortex,
            axons: Vec::new(),
            carry: [Vec::new(), Vec::new(), Vec::new()],
        }
    }

    pub fn add_axon(&mut self, axon: Axon) {
        self.axons.push(axon);
    }

    fn size_of(&self, o: OrganId) -> usize {
        match o {
            OrganId::Spine => self.spine.n_neurons(),
            OrganId::Midbrain => self.midbrain.n_neurons(),
            OrganId::Cortex => self.cortex.n,
        }
    }

    /// Один мировой тик: обмен токами → шаг органов своими подшагами.
    pub fn tick(&mut self, world: &WorldInputs) -> OrganSpikes {
        // 0. Внешние токи: мир + перенесённые аксонные тики прошлого.
        let mut ext: [Vec<f32>; 3] = [
            world.spine.clone(),
            world.midbrain.clone(),
            world.cortex.clone(),
        ];
        for o in OrganId::all() {
            let i = o as usize;
            let n = self.size_of(o);
            let v = &mut ext[i];
            if v.len() < n {
                v.resize(n, 0.0);
            }
            let carry = &self.carry[i];
            for (dst, &c) in v.iter_mut().zip(carry.iter()) {
                *dst += c;
            }
        }

        // 1. Шаг органов — у каждого свой dt/подшаги.
        let spine = run_fine(&mut self.spine, &ext[0], SUBSTEPS_FINE);
        let midbrain = run_fine(&mut self.midbrain, &ext[1], SUBSTEPS_FINE);
        let cortex = {
            let mut sums = vec![0i32; self.cortex.n];
            for _ in 0..OrganId::Cortex.substeps() {
                let s = self.cortex.step(&ext[2]);
                for (d, &x) in sums.iter_mut().zip(s.iter()) {
                    *d += x;
                }
            }
            sums
        };
        let spikes = OrganSpikes {
            spine,
            midbrain,
            cortex,
        };

        // 2. Аксонная доставка на следующий тик.
        for o in OrganId::all() {
            let n = self.size_of(o);
            self.carry[o as usize] = vec![0.0; n];
        }
        for axon in &self.axons {
            let total: i64 = axon
                .pre
                .iter()
                .map(|&i| spikes.of(axon.from)[i] as i64)
                .sum();
            let cur = total as f32 * axon.weight_per_spike;
            let carry = &mut self.carry[axon.to as usize];
            for &post in &axon.post {
                if post < carry.len() {
                    carry[post] += cur;
                }
            }
        }
        spikes
    }

    /// Суммарные спайки органа за последний тик.
    pub fn total(&self, s: &OrganSpikes, o: OrganId) -> i64 {
        s.total(o)
    }
}

/// Подшаги тонкого органа (HH/HAAP): run(1) × N, сумма спайков.
fn run_fine(runner: &mut BrainRunner, ext: &[f32], substeps: usize) -> Vec<i32> {
    let n = runner.n_neurons();
    let mut sums = vec![0i32; n];
    for _ in 0..substeps {
        let buf = runner.run(1, ext);
        for (d, &x) in sums.iter_mut().zip(buf.iter()) {
            *d += x;
        }
    }
    sums
}
