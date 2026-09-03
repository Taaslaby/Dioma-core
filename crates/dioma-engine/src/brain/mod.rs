//! # Brain — мозги
//!
//! Низкоуровневое ядро Dioma. Здесь нет карты, объектов и рендера: только
//! нейроны, коннектом, генетические параметры и вычислительные правила.
//! Биология клетки — в [`crate::body`], общая регистровая ВМ — в
//! [`crate::vm::vm`].
//!
//! Основной поток данных:
//!
//! ```text
//! layout + genome → grow → Grown → Brain → spikes/actions
//! ```
//!
//! Подпапки:
//! - [`dynamics`] — исполнение: `Brain`, GPU-раннеры, численное ядро;
//! - [`genes`] — гены: правила Пифии, поведение, инстинкты, регуляция;
//! - [`growth`] — проводка: геном, рост коннектома, живой граф;
//!
//! Исторически плоские пути (`brain::neurons`, `brain::grow`, …) сохранены
//! ре-экспортами — внешний код не менялся.
pub mod dynamics;
pub mod genes;
pub mod growth;
pub mod models;

// ── Стабильные пути модулей (плоская структура brain::X) ──────────────
pub use dynamics::{gpu, gpu_pool, neurons, perf};
pub use genes::{behavior, instincts, pythia_vm, regulatory, rules};
pub use growth::{genome, graph, grow};
// ── Ключевые типы ядра на один импорт ──────────────────────────────────
pub use genome::Grown;
pub use instincts::Instinct;
pub use neurons::{Brain, BrainModel, PlasticityParams};
pub use regulatory::{RegGene, RegGenome, Target};
pub use rules::RuleGene;

