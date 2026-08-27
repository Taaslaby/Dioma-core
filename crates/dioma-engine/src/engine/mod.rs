//! # Neural engine
//!
//! Низкоуровневое ядро Dioma. Здесь нет карты, объектов и рендера: только
//! нейроны, коннектом, генетические параметры и вычислительные правила.
//!
//! Основной поток данных:
//!
//! ```text
//! layout + genome → grow → Grown → Brain → spikes/actions
//! ```
//!
//! - [`grow`] строит или восстанавливает проводку;
//! - [`neurons`] исполняет динамику и пластичность;
//! - [`behavior`] и [`biovm`] исполняют программы клеточного поведения;
//! - [`graph`] меняет живую топологию по активности;
//! - [`gpu`] содержит GPU-специфичный путь, сохраняя CPU fallback;
//! - [`instincts`] и [`rules`] описывают наследуемые поведенческие правила.
pub mod behavior;
pub mod biocell;
pub mod biovm;
pub mod genome;
pub mod gpu;
pub mod graph;
pub mod grow;
pub mod instincts;
pub mod neurons;
pub mod perf;
pub mod pythia_vm;
pub mod regulatory;
pub mod rules;
pub mod viz;
/// Низкоуровневый SIMD-executor: приватный, используйте biovm/pythia_vm.
pub(crate) mod vm;

pub use biocell::BioCellRuntime;
pub use biovm::{BioEvent, BioEventKind, BioPopulation, BioProgram, SubstanceId};
/// Ключевые типы ядра на один импорт.
pub use genome::Grown;
pub use instincts::Instinct;
pub use neurons::{Brain, BrainModel, PlasticityParams};
pub use regulatory::{RegGene, RegGenome, Target};
pub use rules::RuleGene;
/// Визуальные данные выращенного мозга (внешние визуализаторы).
pub use viz::{BrainViz, LinkViz, NeuronViz, RegionViz, VizMeta, ZoneViz};
