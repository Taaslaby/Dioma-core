//! # Error types
//!
//! Централизованная обработка ошибок для Dioma Engine.

use std::fmt;

/// Основной тип ошибок движка.
#[derive(Debug, Clone)]
pub enum EngineError {
    /// Агент не найден в мире.
    AgentNotFound(crate::env::AgentId),
    
    /// Неверное состояние ягоды.
    InvalidBerryState {
        id: crate::env::AgentId,
        reason: String,
    },
    
    /// Ошибка инициализации GPU.
    GpuError(String),
    
    /// Неверное значение сенсора.
    InvalidSenseValue {
        field: &'static str,
        value: f32,
    },
    
    /// Неверная конфигурация.
    InvalidConfig(String),
    
    /// Ошибка при работе с геномом.
    GenomeError(String),
    
    /// Ошибка при работе с мозгом.
    BrainError(String),
    
    /// Ошибка при работе с метаболизмом.
    MetabolismError(String),
    
    /// Общая ошибка ввода-вывода.
    IoError(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AgentNotFound(id) => write!(f, "Agent not found: {:?}", id),
            Self::InvalidBerryState { id, reason } => {
                write!(f, "Invalid berry state for {:?}: {}", id, reason)
            }
            Self::GpuError(msg) => write!(f, "GPU error: {}", msg),
            Self::InvalidSenseValue { field, value } => {
                write!(f, "Invalid sense value for {}: {}", field, value)
            }
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            Self::GenomeError(msg) => write!(f, "Genome error: {}", msg),
            Self::BrainError(msg) => write!(f, "Brain error: {}", msg),
            Self::MetabolismError(msg) => write!(f, "Metabolism error: {}", msg),
            Self::IoError(msg) => write!(f, "IO error: {}", msg),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<std::io::Error> for EngineError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err.to_string())
    }
}

/// Тип результата для операций движка.
pub type Result<T> = std::result::Result<T, EngineError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_works() {
        let err = EngineError::InvalidSenseValue {
            field: "smell_l",
            value: 1.5,
        };
        assert!(err.to_string().contains("smell_l"));
        assert!(err.to_string().contains("1.5"));
    }

    #[test]
    fn io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let engine_err: EngineError = io_err.into();
        assert!(matches!(engine_err, EngineError::IoError(_)));
    }
}
