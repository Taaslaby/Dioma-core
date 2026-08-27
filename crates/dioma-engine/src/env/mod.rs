//! # Environment
//!
//! Геометрия и химическая среда мира. Карта состоит из комнат и дверей;
//! каждая комната хранит CA-концентрации веществ, а [`Climate`] переносит
//! тепло, свет и питание по пространственной сетке.
//!
//! Этот слой не управляет существами, скриптами или рендером. Он только
//! отвечает на вопросы «где находится точка?» и «какие условия здесь?».
pub mod climate;
pub mod id;
pub mod map;
pub mod math;
pub mod world_clock;

pub use climate::Climate;
pub use id::{AgentId, Tick};
pub use map::{CaRates, Door, Rect, Room, Substance, WorldMap, SUBSTANCES};
pub use math::{RoomId, Vec2};
pub use world_clock::{
    Season, TimeOfDay, WorldClock, DAY_LENGTH_MINUTES, DAYS_PER_SEASON, PHASE_LIGHT, SEASONS,
    TICK_MS, TICKS_PER_DAY, YEAR_DAYS,
};
