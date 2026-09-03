//! # Object world
//!
//! Игровые объекты и их событийная жизнь: агенты, классы, слоты скриптов,
//! сообщения и таймеры.
//!
//! `World` намеренно не содержит нейронный мозг существа. Он отвечает за
//! пассивные объекты вроде ягод и дверей; [`crate::creature::Creature`] живёт
//! рядом и получает сенсоры через [`crate::sim::Arena`].
pub mod agent;
pub mod sim;

pub use agent::{Agent, Cls, Message};
pub use sim::World;

// Legacy slot ids (VM удалена, константы оставлены для совместимости)
pub const SLOT_ACTIVATE1: u8 = 0;
pub const SLOT_EAT: u8 = 4;
pub const SLOT_TIMER: u8 = 5;
