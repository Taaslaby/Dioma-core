//! Идентификаторы мира: агент и такт (порт a1-core). Вся симуляция дискретна.

use std::fmt;

/// Идентификатор агента. `0` зарезервирован как «никто».
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AgentId(pub u64);

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AgentId {
    pub const NONE: AgentId = AgentId(0);

    pub fn is_none(self) -> bool {
        self.0 == 0
    }
}

/// Номер такта мира.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Tick(pub u64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_agent() {
        assert!(AgentId::NONE.is_none());
        assert!(!AgentId(7).is_none());
    }
}
