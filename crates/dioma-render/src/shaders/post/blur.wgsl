// dioma-render: пост-проход «размытие» (9-точечный box blur).
//
// Полноэкранный квад; сэмплирует текстуру сцены 9 точками вокруг фрагмента.
// Радиус (в пикселях) — из uniform; radius 0 вырождается в пассивное копирование.
//
// Uniform: vec4(u_time, vignette_strength, grain_strength, blur_radius) —
// используется только blur_radius.

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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let texel = vec2<f32>(1.0) / vec2<f32>(textureDimensions(scene_tex));
    let r = post.p.w * texel;
    var sum: vec4<f32> = vec4<f32>(0.0);
    var offs = array<vec2<f32>, 9>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0,  0.0), vec2<f32>(0.0,  0.0), vec2<f32>(1.0,  0.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>(0.0,  1.0), vec2<f32>(1.0,  1.0),
    );
    for (var i: u32 = 0u; i < 9u; i = i + 1u) {
        sum = sum + textureSample(scene_tex, scene_sampler, in.uv + offs[i] * r);
    }
    return sum / 9.0;
}
