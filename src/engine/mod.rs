//! -------------------------------------------------------------------------
//! Filename:    src/engine/mod.rs
//! Class:       neuworks
//! Purpose:     ЯДРО ДВИЖКА — геном и нейродинамика, без тела и мира
//! Description: слой знает только о нейронах и связях:
//!               genome — Grown (геном: связи+регионы+гены правил);
//!               grow   — рост коннектома из layout (мост в prometheus);
//!               neurons— Brain: спайковая динамика + пластичность;
//!               rules  — Пифия: ВМ правил динамики клеток;
//!               graph  — тропическая перестройка живого графа.
//! author: Atlasov anton
// -------------------------------------------------------------------------
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
