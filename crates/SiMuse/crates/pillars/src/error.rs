//! Зонтичный тип ошибок pillars.
//!
//! API-слои (`builder`, `validate`) имеют собственные типы ошибок без
//! конверсий между собой, поэтому коду, комбинирующему их (собрать сеть →
//! провалидировать), приходилось матчить каждый тип отдельно. `Error`
//! объединяет их через `From`, чтобы `?` пробрасывал всё единообразно:
//!
//! ```
//! use pillars::prelude::*;
//!
//! fn load_and_check(n: usize) -> std::result::Result<(), Error> {
//!     // Типизированный билдер LIF возвращает Result — BuildError
//!     // прокидывается в единый `Error` через `?`.
//!     let _pop = LIFBuilder::new(n).try_build()?;
//!     Ok(())
//! }
//!
//! assert!(load_and_check(8).is_ok());
//! ```

use std::fmt;

use crate::api::builder::BuildError;
use crate::api::validate::ValidationError;

/// Ошибка любого API-слоя pillars.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Ошибка сборки модели из билдеров (`api::builder`).
    Build(BuildError),
    /// Ошибка валидации против эталона (`api::validate`).
    Validate(ValidationError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Build(err) => write!(f, "builder error: {err}"),
            Error::Validate(err) => write!(f, "validation error: {err}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Build(err) => Some(err),
            Error::Validate(err) => Some(err),
        }
    }
}

impl From<BuildError> for Error {
    fn from(err: BuildError) -> Self {
        Error::Build(err)
    }
}

impl From<ValidationError> for Error {
    fn from(err: ValidationError) -> Self {
        Error::Validate(err)
    }
}

/// `Result<T, pillars::Error>` для сквозного `?` через все API-слои.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_conversions_source_chained() {
        let validate = ValidationError::DimensionMismatch { expected: 2, actual: 3 };
        let err: Error = validate.into();
        assert!(err.to_string().contains("expected 2, got 3"));
    }
}
