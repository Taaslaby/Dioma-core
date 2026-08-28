//! dioma-render — 2D спрайтовый рендер-движок на wgpu.
//!
//! Чистый рендерер: никакой зависимости от dioma, никакого egui.
//! Принимает примитивы wgpu и рисует инстансированные текстурированные
//! квады (спрайты) с 2D-камерой, per-sprite эффектами (контур, свечение,
//! растворение, дрожь), произвольными 2D-мешами (текстура + per-vertex цвет;
//! вершины в мировых координатах — основа для Spine-скиннинга и деформаций),
//! радиальными источниками света, частицами и пост-обработкой
//! (виньетка/зерно, размытие). Текстуры собираются в атлас из процедурных
//! генераторов (круг, кольцо, ягода, полоса, мягкий круг) — ноль
//! ассет-файлов. Тайловые карты (Tilemap/TileBatch — сетка u16-тайлов,
//! рендерящаяся базовым спрайтовым батчем) и параллакс-слои фона
//! ([`Renderer::render_parallax`]) — без новых пайплайнов. Все шейдеры
//! встроены через `include_str!` и собираются в `shaders.rs`: общий пролог
//! (камера/хелперы) + тело эффекта в своём файле `shaders/**.wgsl`.
//!
//! Соглашение осей (сайд-вью): мировая Y растёт вверх, верх экрана = больший
//! Y мира (NDC Y вверх); X — вправо. [`Camera::screen_to_world`] согласован
//! с рендером — используйте его для пикинга.
//!
//! Группировка по папкам:
//! - `scene/` — спрайтовый конвейер и мировая геометрия (атлас, батч,
//!   спрайты, флипбук, тайлы, меши);
//! - `fx/` — эффекты поверх сцены (свет, частицы, пост-обработка);
//! - `debug/` — отладка (текстовый шрифт, профайлер);
//! - `renderer.rs` — оркестратор проходов, `camera.rs` — 2D-камера;
//! - `shaders.rs` + `shaders/**.wgsl` — WGSL: общий пролог + тела эффектов.
//!
//! Публичные типы реэкспортируются в корне; старые плоские пути модулей
//! (`dioma_render::atlas`, `dioma_render::light`, …) работают через алиасы.

pub mod camera;
pub mod debug;
pub mod fx;
pub mod renderer;
pub mod scene;
mod shaders;
pub mod spine;
pub mod ui;

// Обратная совместимость старых плоских путей модулей (dioma_render::atlas,
// dioma_render::light, …): сами модули теперь живут в scene/ и fx/.
pub use fx::{light, particles, postprocess};
pub use scene::{atlas, batch, flipbook, mesh, sprite, tilemap};
pub use debug::profiler;

// Типы — как раньше, плоско: dioma_render::{Renderer, Sprite, …}
pub use atlas::SpriteAtlas;
pub use batch::SpriteBatch;
pub use camera::Camera;
pub use debug::{DebugFont, RenderDebug};
pub use flipbook::Flipbook;
pub use fx::light::{Light, LightBatch};
pub use fx::particles::{Particle, ParticleSystem};
pub use fx::postprocess::{PostFx, PostProcess};
pub use mesh::{Mesh, MeshBatch, MeshBlend, MeshVertex};
pub use profiler::RenderStats;
pub use renderer::{Renderer, ScissorRect};
pub use scene::sprite::{Sprite, SpriteFx};
pub use tilemap::{TileBatch, Tilemap, TilesetUv};

// UI-каркас: основные типы плоско.
pub use ui::{GlassBackdrop, UiInput, UiSystem};
