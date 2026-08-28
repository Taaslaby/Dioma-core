//! Профилировщик рендера: статистика кадра (FPS, время кадра, счётчики
//! draw calls и инстансов по каждому пайплайну).
//!
//! Счётчики обнуляются в начале кадра ([`Renderer::begin_frame`]) и
//! накапливаются проходами рендера ([`Renderer::render`],
//! [`Renderer::render_meshes`], [`Renderer::render_lights`],
//! [`Renderer::render_particles`]); текущий снимок читается через
//! [`Renderer::stats`]. Оверлей текста (см. [`crate::debug`]) просто
//! печатает эти числа.

/// Снимок статистики одного кадра.
///
/// * `fps`/`frame_ms` — сглаженные EMA-значения (alpha ≈ 0.1);
/// * `draw_calls` — суммарно выданных draw calls во всех проходах;
/// * `sprites` — инстансы базового спрайтового пайплайна (fx = None);
/// * `fx_sprites` — инстансы fx-пайплайнов (outline/glow/dissolve/wobble);
/// * `meshes`/`mesh_vertices`/`mesh_indices` — меши и их вершин/индексов;
/// * `lights` — источники света;
/// * `particles` — частицы.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RenderStats {
    /// Сглаженный FPS (EMA, alpha ≈ 0.1).
    pub fps: f32,
    /// Сглаженное время кадра в миллисекундах (EMA, alpha ≈ 0.1).
    pub frame_ms: f32,
    /// Суммарное количество draw calls за кадр.
    pub draw_calls: u32,
    /// Спрайты базового пайплайна (без эффекта).
    pub sprites: u32,
    /// Спрайты fx-пайплайнов.
    pub fx_sprites: u32,
    /// Меши (по одному на push в батч).
    pub meshes: u32,
    /// Вершины мешей, ушедшие на GPU.
    pub mesh_vertices: u32,
    /// Индексы мешей, ушедшие на GPU.
    pub mesh_indices: u32,
    /// Источники света.
    pub lights: u32,
    /// Частицы.
    pub particles: u32,
}
