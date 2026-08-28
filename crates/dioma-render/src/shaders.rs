//! Единая точка загрузки WGSL-шейдеров dioma-render.
//!
//! Каждый шейдер живёт в своём файле (`shaders/**/*.wgsl`) и описывает только
//! свою специфику — тела `vs_main`/`fs_main` и свои вершинные структуры.
//! Общие части приклеиваются при компиляции модуля:
//!
//! - мировые шейдеры (спрайты, fx, меши, свет, частицы, debug) получают
//!   пролог [`COMMON_WGSL`] — uniform камеры и хелперы
//!   (`world_to_ndc`, `pulse_scale`, `quad_world`, `hash01`);
//! - пост-проходы получают `post/post_common.wgsl` — полноэкранный квад,
//!   uniform и сэмплер сцены.
//!
//! Порядок: пролог → тело файла. Функции возвращают готовый исходник
//! (`String`) для `wgpu::ShaderSource::Wgsl(..)`.

const COMMON_WGSL: &str = include_str!("shaders/common.wgsl");
const POST_COMMON_WGSL: &str = include_str!("shaders/post/post_common.wgsl");

/// Мировой шейдер: пролог камеры/хелперов + тело эффекта.
fn world(body: &'static str) -> String {
    format!("{}\n{}", COMMON_WGSL, body)
}

/// Пост-шейдер: пролог полноэкранного квада + тело эффекта.
fn post(body: &'static str) -> String {
    format!("{}\n{}", POST_COMMON_WGSL, body)
}

// ── Спрайтовое семейство (инстансные квады, группа 1 — атлас) ──────────────

/// Базовый спрайт: текстурированный квад с tint'ом.
pub fn sprite() -> String {
    world(include_str!("shaders/sprite.wgsl"))
}

/// Контур выделенного существа (`SpriteFx::Outline`).
pub fn fx_outline() -> String {
    world(include_str!("shaders/fx/outline.wgsl"))
}

/// Внешнее свечение (`SpriteFx::Glow`).
pub fn fx_glow() -> String {
    world(include_str!("shaders/fx/glow.wgsl"))
}

/// Растворение/смерть (`SpriteFx::Dissolve`).
pub fn fx_dissolve() -> String {
    world(include_str!("shaders/fx/dissolve.wgsl"))
}

/// Волнистость/дрожь (`SpriteFx::Wobble`).
pub fn fx_wobble() -> String {
    world(include_str!("shaders/fx/wobble.wgsl"))
}

/// Мягкая drop-тень (`SpriteFx::Shadow`).
pub fn fx_shadow() -> String {
    world(include_str!("shaders/fx/shadow.wgsl"))
}

/// Обесцвечивание (`SpriteFx::Grayscale`).
pub fn fx_grayscale() -> String {
    world(include_str!("shaders/fx/grayscale.wgsl"))
}

/// Стеклянная подложка панели (группа 2 — размытый бэкдроп).
pub fn fx_frost() -> String {
    world(include_str!("shaders/fx/frost.wgsl"))
}

// ── Прочие мировые проходы ──────────────────────────────────────────────────

/// Меши: произвольная триангулированная геометрия (CPU-скиннинг).
pub fn mesh() -> String {
    world(include_str!("shaders/mesh/mesh.wgsl"))
}

/// Радиальный аддитивный свет.
pub fn light_radial() -> String {
    world(include_str!("shaders/light/radial.wgsl"))
}

/// Частицы: мягкие аддитивные круги.
pub fn particle() -> String {
    world(include_str!("shaders/particle/particle.wgsl"))
}

/// Debug-шейпы (круги/прямоугольники/линии) — цвет без текстуры.
pub fn debug_shape() -> String {
    world(include_str!("shaders/debug/shape.wgsl"))
}

// ── Пост-проходы ────────────────────────────────────────────────────────────

/// Виньетка + плёночное зерно.
pub fn post_vignette_grain() -> String {
    post(include_str!("shaders/post/vignette_grain.wgsl"))
}

/// 9-точечное размытие.
pub fn post_blur() -> String {
    post(include_str!("shaders/post/blur.wgsl"))
}

/// Тьма/ночь: умножение сцены на tint.
pub fn post_darkness() -> String {
    post(include_str!("shaders/post/darkness.wgsl"))
}
