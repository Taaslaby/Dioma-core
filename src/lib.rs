//! Headless backend for Unity (cdylib).
//! Вся симуляция — в `dioma-engine`; этот крейт — только тонкий cdylib +
//! экспортёры golden для `com.dioma.engine`.

pub use dioma_engine::*;

/// Стабильный C ABI для Unity: один тик существа за вызов, без GUI/рендера.
pub mod unity_bridge;
