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
use crate::script::{Machine, Program};

/// scripts slots
pub mod slot {
    pub const ACTIVATE1: u8 = 0; // «трогают»
    pub const ACTIVATE2: u8 = 1; // второе действие
    pub const DEACTIVATE: u8 = 2;
    pub const HIT: u8 = 3; // удар/столкновение
    pub const EAT: u8 = 4; // съеден
    pub const TIMER: u8 = 5; // периодический тик
    pub const N: usize = 6;
}

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

/// mashine script exc
#[derive(Debug, Clone)]
pub struct Script {
    pub prog: Program,
    pub machine: Machine,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub from: AgentId,
    /// slot-worker on agents
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
    /// sprite name
    pub sprite: String,
    /// int vars ov0..ov15.
    pub vars: [f32; 16],
    pub scripts: [Option<Script>; slot::N],
    /// Период таймерного скрипта в тактах (0 = выключен).
    pub timer_period: u32,
    timer_acc: u32,
    inbox: Vec<Message>,
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
            scripts: Default::default(),
            timer_period: 0,
            timer_acc: 0,
            inbox: Vec::new(),
        }
    }

    pub fn receive(&mut self, msg: Message) {
        self.inbox.push(msg);
    }

    pub fn drain_inbox(&mut self) -> Vec<Message> {
        std::mem::take(&mut self.inbox)
    }

    /// Прошёл ли таймер до срабатывания.
    pub fn tick_timer(&mut self) -> bool {
        if self.timer_period == 0 {
            return false;
        }
        self.timer_acc += 1;
        if self.timer_acc >= self.timer_period {
            self.timer_acc = 0;
            true
        } else {
            false
        }
    }
}
