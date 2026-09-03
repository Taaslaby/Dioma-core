//! # Biology — биология
//!
//! Физиология существа поверх мозгов ([`crate::brain`]): органы, кровь,
//! обмен веществ, химия клеток и программируемые клеточные популяции на
//! BioVM. Этот слой не знает о карте и рендере — он получает только
//! локальные сигналы и события от [`crate::creature`].

pub mod biocell;
pub mod biovm;
pub mod cell_chemistry;
pub mod metabolism;
pub mod organs;
pub mod physiology;
pub mod tissue;

pub use biocell::BioCellRuntime;
pub use biovm::{BioEvent, BioEventKind, BioPopulation, BioProgram, SubstanceId};
