//! -------------------------------------------------------------------------
//! Filename:    src/body/mod.rs
//! Class:       DebugSuite
//! Purpose:     ОРГАНИЗМ — тело поверх ядра: органы, физиология, обмен
//! Description: слой добавляет к мозгу кровь, температуру, голод и
//!               пошаговую физиологию (восприятие→мозг→действие).
//! author: Atlasov anton
// -------------------------------------------------------------------------
pub mod cell_chemistry;
pub mod metabolism;
pub mod organs;
pub mod physiology;
pub mod tissue;
