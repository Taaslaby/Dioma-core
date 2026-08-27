//! # Body
//!
//! Физиология существа поверх нейронного ядра: органы, кровь, обмен веществ
//! и химия клеток. Этот слой не знает о карте и рендере — он получает только
//! локальные сигналы и события от [`crate::creature`].

pub mod cell_chemistry;
pub mod metabolism;
pub mod organs;
pub mod physiology;
pub mod tissue;
