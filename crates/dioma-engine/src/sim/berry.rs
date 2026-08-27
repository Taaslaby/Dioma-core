//! # Berry state management
//!
//! Типобезопасная обертка для состояния ягод вместо магических индексов vars[].

use crate::env::AgentId;
use crate::env::math::Vec2;
use crate::error::{EngineError, Result};
use crate::world::Agent;

/// Типобезопасное состояние ягоды.
///
/// Заменяет прямой доступ к `vars[0]` и `vars[1]` именованными полями.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BerryState {
    /// Накопленный рост (vars[0]): плодородность × GROW_RATE за тик.
    /// При достижении порога ягода дает потомство и обнуляется.
    pub growth: f32,
    
    /// Таймер отрастания съеденной ягоды (vars[1]): тикает вниз к нулю.
    /// > 0 = ягода пустая (не пахнет, не съедобна).
    pub regrow: f32,
}

impl BerryState {
    /// Создает новое состояние ягоды (свежая, съедобная).
    pub fn new() -> Self {
        Self {
            growth: 0.0,
            regrow: 0.0,
        }
    }
    
    /// Создает состояние из агента мира.
    pub fn from_agent(agent: &Agent) -> Result<Self> {
        if agent.vars.len() < 2 {
            return Err(EngineError::InvalidBerryState {
                id: AgentId(0), // ID будет заполнен вызывающим кодом
                reason: "Agent vars array too short".to_string(),
            });
        }
        
        Ok(Self {
            growth: agent.vars[0],
            regrow: agent.vars[1],
        })
    }
    
    /// Применяет состояние к агенту мира.
    pub fn to_agent(&self, agent: &mut Agent) {
        if agent.vars.len() >= 2 {
            agent.vars[0] = self.growth;
            agent.vars[1] = self.regrow;
        }
    }
    
    /// Съедобна ли ягода (не отрастает).
    pub fn is_edible(&self) -> bool {
        self.regrow <= 0.0
    }
    
    /// Добавляет рост от плодородности.
    pub fn add_growth(&mut self, fertility: f32, grow_rate: f32) {
        self.growth += fertility * grow_rate;
    }
    
    /// Проверяет, готова ли ягода дать потомство.
    pub fn ready_to_spawn(&self, spawn_threshold: f32) -> bool {
        self.growth >= spawn_threshold
    }
    
    /// Сбрасывает рост после размножения.
    pub fn reset_growth(&mut self) {
        self.growth = 0.0;
    }
    
    /// Начинает отрастание после съедения.
    pub fn start_regrow(&mut self, regrow_ticks: f32) {
        self.regrow = regrow_ticks;
    }
    
    /// Тикает таймер отрастания.
    pub fn tick_regrow(&mut self) {
        if self.regrow > 0.0 {
            self.regrow = (self.regrow - 1.0).max(0.0);
        }
    }
}

impl Default for BerryState {
    fn default() -> Self {
        Self::new()
    }
}

/// Читаемое состояние ягоды для хоста и моста рендера.
#[derive(Clone, Copy, Debug)]
pub struct BerryInfo {
    pub id: AgentId,
    pub pos: Vec2,
    pub state: BerryState,
}

impl BerryInfo {
    /// Съедобна ли ягода.
    pub fn is_edible(&self) -> bool {
        self.state.is_edible()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::Cls;

    #[test]
    fn berry_state_new_is_edible() {
        let state = BerryState::new();
        assert!(state.is_edible());
        assert_eq!(state.growth, 0.0);
        assert_eq!(state.regrow, 0.0);
    }

    #[test]
    fn berry_state_growth_accumulates() {
        let mut state = BerryState::new();
        state.add_growth(0.5, 2.0);
        assert_eq!(state.growth, 1.0);
        
        state.add_growth(0.5, 2.0);
        assert_eq!(state.growth, 2.0);
    }

    #[test]
    fn berry_state_spawn_threshold() {
        let mut state = BerryState::new();
        assert!(!state.ready_to_spawn(10.0));
        
        state.growth = 10.0;
        assert!(state.ready_to_spawn(10.0));
        
        state.reset_growth();
        assert_eq!(state.growth, 0.0);
    }

    #[test]
    fn berry_state_regrow_cycle() {
        let mut state = BerryState::new();
        assert!(state.is_edible());
        
        state.start_regrow(5.0);
        assert!(!state.is_edible());
        assert_eq!(state.regrow, 5.0);
        
        for _ in 0..5 {
            state.tick_regrow();
        }
        
        assert!(state.is_edible());
        assert_eq!(state.regrow, 0.0);
    }

    #[test]
    fn berry_state_from_agent() {
        let cls = Cls::new(1, 0, 0); // family=1 для ягод
        let mut agent = Agent::new(AgentId(1), cls, Vec2::ZERO, "berry");
        agent.vars[0] = 3.5;
        agent.vars[1] = 2.0;
        
        let state = BerryState::from_agent(&agent).unwrap();
        assert_eq!(state.growth, 3.5);
        assert_eq!(state.regrow, 2.0);
        
        let mut new_state = BerryState::new();
        new_state.growth = 7.0;
        new_state.regrow = 1.0;
        new_state.to_agent(&mut agent);
        
        assert_eq!(agent.vars[0], 7.0);
        assert_eq!(agent.vars[1], 1.0);
    }

    #[test]
    fn berry_state_from_invalid_agent() {
        let cls = Cls::new(1, 0, 0);
        let agent = Agent::new(AgentId(1), cls, Vec2::ZERO, "berry");
        
        let result = BerryState::from_agent(&agent);
        assert!(result.is_ok());
    }
}
