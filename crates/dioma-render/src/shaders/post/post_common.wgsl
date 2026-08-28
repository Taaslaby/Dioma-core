// dioma-render — ОБЩИЙ ПРОЛОГ пост-проходов (экранные эффекты).
//
// Приклеивается перед телом пост-шейдера в `shaders.rs` (см. `post()`).
// Полноэкранный квад (6 вершин в NDC), uniform-вектор и сэмплер сцены общие
// для всех пост-эффектов; тело шейдера определяет только `fs_main`.
//
// Uniform — 16 байт, значение зависит от эффекта (см. доку каждого файла).

struct PostUniform {
    p: vec4<f32>,
};

@group(0) @binding(0) var<uniform> post: PostUniform;
@group(0) @binding(1) var scene_tex: texture_2d<f32>;
@group(0) @binding(2) var scene_sampler: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VsOut {
    var out: VsOut;
    out.clip_pos = vec4<f32>(pos, 0.0, 1.0);
    // pos — NDC (ось Y вверх); UV с флипом по V, чтобы сцена копировалась 1:1.
    out.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    return out;
}
