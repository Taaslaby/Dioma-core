//! Karmac — SPAUN-inspired cognitive architecture modules.
//!
//! # Modules
//! - `brain`: SpikingBrain — real spiking controller (pillars LIF) with
//!   spiking GNW workspace + reward-modulated STDP learning
//! - `gnw`: Global Neuronal Workspace (Baars/Dehaene) — attention broadcast
//! - `lod`: LOD — Carmack BSP-style neuron level-of-detail (HH→Izh→LIF→tanh)
//! - `basal_ganglia`: Action selection (winner-take-all routing)
//! - `working_memory`: Prefrontal cortex-like state retention
//! - `semantic_pointer`: SPA operations (bind/unbind/superpose via FFT)

#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]

pub mod agent;
pub mod basal_ganglia;
pub mod brain;
pub mod gnw;
pub mod lod;
pub mod min_brain;
pub mod semantic_pointer;
pub mod working_memory;

pub use agent::{run_episode, AgentBrain, AgentWorld, EpisodeReport, SpikingAgent};
pub use basal_ganglia::{BasalGanglia, BgConfig};
pub use brain::SpikingBrain;
pub use lod::{CarmackBsp, Coord, LodLevel};
pub use semantic_pointer::SemanticPointer;
pub use working_memory::WorkingMemory;
