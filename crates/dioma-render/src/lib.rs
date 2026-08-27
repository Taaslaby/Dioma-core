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
//! ассет-файлов. Все шейдеры встроены через `include_str!`.

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
pub use renderer::Renderer;
pub use sprite::{Sprite, SpriteFx};
