//! # Brain — мозги
//!
//! Низкоуровневое ядро Dioma. Здесь нет карты, объектов и рендера: только
//! нейроны, коннектом, генетические параметры и вычислительные правила.
//! Биология клетки — в [`crate::biology`], общая регистровая ВМ — в
//! [`crate::logic::vm`].
//!
//! Основной поток данных:
//!
//! ```text
//! layout + genome → grow → Grown → Brain → spikes/actions
//! ```
//!
//! - [`grow`] строит или восстанавливает проводку;
//! - [`neurons`] исполняет динамику и пластичность;
//! - [`baked`] грузит запечённые веса DBW1 и исполняет MLP/GRU-инференс;
//! - [`graph`] меняет живую топологию по активности;
//! - [`gpu`] содержит GPU-специфичный путь, сохраняя CPU fallback;
//! - [`instincts`] и [`rules`] описывают наследуемые поведенческие правила.
pub mod behavior;
pub mod baked;
pub mod genome;
pub mod gpu;
pub mod gpu_pool;
pub mod graph;
pub mod grow;
pub mod instincts;
pub mod neurons;
pub mod perf;
pub mod pythia_vm;
pub mod regulatory;
pub mod rules;
pub mod viz;

pub use baked::{BakedGru, BakedMlp, BakedTensors, DBW_MAGIC};
/// Ключевые типы ядра на один импорт.
pub use genome::Grown;
pub use instincts::Instinct;
pub use neurons::{Brain, BrainModel, PlasticityParams};
pub use regulatory::{RegGene, RegGenome, Target};
pub use rules::RuleGene;
/// Визуальные данные выращенного мозга (внешние визуализаторы).
pub use viz::{BrainViz, LinkViz, NeuronViz, RegionViz, VizMeta, ZoneViz};
