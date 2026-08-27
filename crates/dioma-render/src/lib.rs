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
//! встроены через `include_str!`.
//!
//! Соглашение осей (сайд-вью): мировая Y растёт вверх, верх экрана = больший
//! Y мира (NDC Y вверх); X — вправо. [`Camera::screen_to_world`] согласован
//! с рендером — используйте его для пикинга.

pub mod atlas;
pub mod batch;
pub mod camera;
pub mod debug;
pub mod flipbook;
pub mod light;
pub mod mesh;
pub mod particles;
pub mod postprocess;
pub mod profiler;
pub mod renderer;
pub mod sprite;
pub mod tilemap;

pub use atlas::SpriteAtlas;
pub use batch::SpriteBatch;
pub use camera::Camera;
pub use debug::{DebugFont, RenderDebug};
pub use flipbook::Flipbook;
pub use light::{Light, LightBatch};
pub use mesh::{Mesh, MeshBatch, MeshBlend, MeshVertex};
pub use particles::{Particle, ParticleSystem};
pub use postprocess::{PostFx, PostProcess};
pub use profiler::RenderStats;
pub use renderer::{Renderer, ScissorRect};
pub use sprite::{Sprite, SpriteFx};
pub use tilemap::{TileBatch, Tilemap, TilesetUv};
