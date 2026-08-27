// dioma-render: пост-проход «виньетка + плёночное зерно».
//
// Полноэкранный квад (6 вершин в NDC); сэмплирует текстуру сцены (группа 0),
// применяет виньетку (затемнение к углам) и анимированное по времени зерно.
//
// Uniform: vec4(u_time, vignette_strength, grain_strength, 0).
// Vignette 0.0 → проход вырождается в пассивное копирование сцены.

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

// Детерминированный псевдослучайный шум по ячейке (для зерна).
fn hash01(v: vec2<u32>) -> f32 {
    var h: u32 = v.x * 1664525u + v.y * 1013904223u + 374761393u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h ^= h >> 16u;
    return f32(h & 0xFFFFFFu) / 16777216.0;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(scene_tex, scene_sampler, in.uv);

    // Виньетка: плавное затемнение от центра (d=0) к углам (d≈0.707).
    let d = distance(in.uv, vec2<f32>(0.5));
    let vig = 1.0 - post.p.y * smoothstep(0.2, 0.72, d);

    // Зерно: шум по пикселю (встроенная позиция фрагмента), плывущий со временем.
    let seed = vec2<u32>(floor(in.clip_pos.xy + vec2<f32>(post.p.x * 13.0)));
    let g = (hash01(seed) - 0.5) * post.p.z;

    return vec4<f32>(c.rgb * vig + vec3<f32>(g), c.a);
}
