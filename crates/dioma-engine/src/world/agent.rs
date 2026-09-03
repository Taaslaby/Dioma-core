//! -------------------------------------------------------------------------
//! Filename:    src/world/agent.rs
//! Purpose:     agent - world obj
//! Description:
//!
//! position, speed, event, scripts slots
//!
//! author: Atlasov anton
//! -------------------------------------------------------------------------

use crate::env::math::Vec2;
use crate::env::AgentId;

/// family, gen, spec
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cls {
    pub family: u8,
    pub genus: u8,
    pub species: u16,
}

impl Cls {
    pub fn new(family: u8, genus: u8, species: u16) -> Self {
        Cls {
            family,
            genus,
            species,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Message {
    pub from: AgentId,
    pub slot: u8,
    pub param: i64,
}

#[derive(Debug, Clone)]
pub struct Agent {
    pub id: AgentId,
    pub cls: Cls,
    pub pos: Vec2,
    pub vel: Vec2,
    pub alive: bool,
    pub sprite: String,
    pub vars: [f32; 16],
}

impl Agent {
    pub fn new(id: AgentId, cls: Cls, pos: Vec2, sprite: &str) -> Self {
        Agent {
            id,
            cls,
            pos,
            vel: Vec2::ZERO,
            alive: true,
            sprite: sprite.to_string(),
            vars: [0.0; 16],
        }
    }
}
