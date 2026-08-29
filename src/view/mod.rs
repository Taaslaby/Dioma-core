//! Мост «сим ↔ рендер»: проекция арены в данные для рисования и обратно —
//! ввод в команды симуляции.
//!
//! Поток данных за кадр:
//!
//! ```text
//!   арена (dioma-engine, чистая)
//!        │  SimView::capture(&arena)  — дешёвая read-only копия
//!        ▼
//!   snapshot (SimView)
//!        │  Presenter::build(&view)   — чистая функция: снимок → спрайты
//!        ▼
//!   DrawScene (tiles / shadows / bodies / faces / berries)
//!        │  SpriteBatch::push + upload → Renderer::render
//!        ▼
//!   экран
//!
//!   ввод: клики/клавиши → InputMapper → GameCommand → arena
//! ```
//!
//! Снимок — копия, а не ссылки: рендер требует `&mut Renderer` и не должен
//! держать `&Arena`; презентация остаётся чистой функцией (детерминизм,
//! replay-friendly). Тик фиксированный: [`TICK`] = 50 мс, accumulator в
//! игровом цикле.

/// Мировой тик, секунды. 50 мс — как в C3 (`App::GetWorldTickInterval()`
/// возвращает 50, engine/App.h:157). Транш 2: 25 мс → 50 мс.
pub const TICK: f32 = 0.05;

mod input;
mod present;
mod snapshot;

pub use dioma_engine::creature::Sex;
pub use dioma_engine::env::{Season, TimeOfDay};
pub use input::{GameCommand, InputMapper};
pub use present::{DrawScene, Presenter, LAYER_TINT};
pub use snapshot::{CritterView, SimView};
