//! -------------------------------------------------------------------------
//! Filename:    src/world/mod.rs
//! Class:       world
//! Purpose:     МИР ОБЪЕКТОВ — агенты со слотами скриптов, сообщения, таймеры
//! Description: порт a1-agents поверх env + script. Существа в этом слое —
//!               болванки (позиция/скорость/класс); разум подключается позже
//!               контроллером (engine::Brain + body::Blood) в слоты агента.
//! author: Atlasov anton
//! -------------------------------------------------------------------------
pub mod agent;
pub mod sim;

pub use agent::{slot, Agent, Cls, Message, Script};
pub use sim::World;

pub use agent::slot::{
    ACTIVATE1 as SLOT_ACTIVATE1, ACTIVATE2 as SLOT_ACTIVATE2, DEACTIVATE as SLOT_DEACTIVATE,
    EAT as SLOT_EAT, HIT as SLOT_HIT, TIMER as SLOT_TIMER,
};