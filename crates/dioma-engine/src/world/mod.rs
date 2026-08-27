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

pub use agent::{slot, Agent, Cls, Message, Script};
pub use sim::World;

pub use agent::slot::{
    ACTIVATE1 as SLOT_ACTIVATE1, ACTIVATE2 as SLOT_ACTIVATE2, DEACTIVATE as SLOT_DEACTIVATE,
    EAT as SLOT_EAT, HIT as SLOT_HIT, TIMER as SLOT_TIMER,
};
