//! # Dioma Engine — чистый headless runtime
//!
//! Headless: без окна/GUI. Unity грузит `dioma.dll` и зовёт `unity_bridge`.
//!
//! ## Структура (по доменам, а не по файлам)
//!
//! ```text
//! brain/     — нейроны, STDP, гены, рост коннектома
//! body/      — тело: кровь, органы, биоклетки (ex-biology)
//! creature/  — существо: мост Sense → мозг → Action + ДНК
//! world/     — мир: arena + ягоды + карта + климат
//!   env/       карта комнат, вещества, время
//!   world/     агенты-ягоды (скоро схлопнется в world)
//!   sim/       Arena — композиция мира + существ
//! vm/        — виртуалки: TNMT, BioVM, Пифия, a1-script (ex-logic)
//! tools/     — наблюдение: SpikeRaster, Trace (ex-sci)
//! support    — rng, error
//! ```

pub mod body;
pub mod brain;
pub mod creature;
pub mod env;
pub mod error;
pub mod vm;
pub mod rng;
pub mod tools;
pub mod sim;
pub mod world;

// Совместимость
#[deprecated(note = "use crate::body instead")]
pub use body as biology;
#[deprecated(note = "use crate::vm instead")]
pub use vm as logic;
#[deprecated(note = "use crate::tools instead")]
pub use tools as sci;

/// Удобный экспорт генетической проводки существа.
pub use brain::Grown;
