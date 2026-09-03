//! # Environment — геометрия мира (карта + математика)
//! Климат и часы мира удалены — теперь это забота Unity.
pub mod id;
pub mod map;
pub mod math;

pub use id::{AgentId, Tick};
pub use map::{CaRates, Door, Rect, Room, Substance, WorldMap, SUBSTANCES};
pub use math::{RoomId, Vec2};
