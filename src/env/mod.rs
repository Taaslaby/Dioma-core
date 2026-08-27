//! -------------------------------------------------------------------------
//! Filename:    src/env/mod.rs
//! Class:       environment
//! Purpose:     СРЕДА МИРА — комнаты, двери, клеточные автоматы веществ и id
//! Description: порт a1-map/a1-core. Тепло, влажность, свет и питание
//!               диффундируют между комнатами через двери и распадаются.
//!               Ничего не знает об агентах и скриптах — только геометрия,
//!               химия комнат и общие идентификаторы (аналог C2e).
//! author: Atlasov anton
//! -------------------------------------------------------------------------
pub mod climate;
pub mod id;
pub mod map;
pub mod math;

pub use climate::Climate;
pub use id::{AgentId, Tick};
pub use map::{CaRates, Door, Rect, Room, Substance, WorldMap, SUBSTANCES};
pub use math::{RoomId, Vec2};
