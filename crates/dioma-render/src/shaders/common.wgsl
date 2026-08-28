// dioma-render — ОБЩИЙ ПРОЛОГ всех «мировых» шейдеров.
//
// Приклеивается перед телом шейдера в `shaders.rs` (см. `world()`), поэтому
// здесь только то, что нужно каждому проходу в мировых координатах:
// uniform камеры, преобразование мир→NDC, пульс-дыхание и детерминированный
// хэш. Пост-проходы (экранные) используют свой пролог — post/post_common.wgsl.
//
// НДЦ: Y вверх = верх экрана (соглашение сайд-вью); мировая Y растёт вверх.
//
// Uniform камеры — 32 байта (2×vec4):
//   camera.xy    — центр камеры (мировые координаты)
//   camera.z     — зум
//   camera.w     — время (секунды; пульс, волнистость, зерно)
//   viewport.xy  — размер вьюпорта в пикселях
//   viewport.zw  — не используется (0, 0)

struct Camera {
    camera: vec4<f32>,
    viewport: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam: Camera;

// Мир → NDC: одна формула на все проходы (спрайты, меши, свет, частицы, debug).
fn world_to_ndc(world: vec2<f32>) -> vec2<f32> {
    return (world - cam.camera.xy) / (cam.viewport.xy * 0.5 / cam.camera.z);
}

// Пульс-дыхание спрайта: масштаб «дышит» между pulse.x и pulse.y по синусу.
//   pulse = (min_scale, max_scale, speed, phase) — instance_pulse из спрайта.
fn pulse_scale(pulse: vec4<f32>) -> f32 {
    return mix(pulse.x, pulse.y, 0.5 + 0.5 * sin(cam.camera.w * pulse.z + pulse.w));
}

// Квад спрайта в мировых координатах: corner_pos ∈ [-0.5, 0.5] → размер →
// масштаб (пульс) → смещение по X (волнение, до поворота) → поворот → позиция.
fn quad_world(corner_pos: vec2<f32>, size: vec2<f32>, scale: f32, x_off: f32, rot: f32, pos: vec2<f32>) -> vec2<f32> {
    let scaled = corner_pos * size * scale + vec2<f32>(x_off, 0.0);
    let c = cos(rot);
    let s = sin(rot);
    return vec2<f32>(
        scaled.x * c - scaled.y * s,
        scaled.x * s + scaled.y * c,
    ) + pos;
}

// Детерминированный целочисленный хэш 0..1 (одинаков на всех GPU —
// пиксельные тесты стабильны). Используется зерном и растворением.
fn hash01(cell: vec2<u32>) -> f32 {
    var h: u32 = cell.x * 1664525u + cell.y * 1013904223u + 374761393u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h ^= h >> 16u;
    return f32(h & 0xFFFFFFu) / 16777216.0;
}
