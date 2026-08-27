// dioma-render: пост-проход «тьма/ночь» — умножение сцены на цвет.
//
// Полноэкранный квад (6 вершин в NDC) умножает отрисованную сцену на tint:
// ночное небо (тёмно-синий tint) затемняет мир, сохраняя оттенки. Вместе с
// аддитивными источниками света ([`crate::Renderer::render_lights`]) даёт
// цикл дня и ночи: свет рисуется ПОВЕРХ тьмы и «пробивает» её собой.
//
// Uniform: vec4(tint.r, tint.g, tint.b, strength).
//   tint.rgb    — множитель цвета сцены (1,1,1 = без изменений);
//   strength    — сила эффекта 0..1 (смешение с исходной сценой);
//                 0 → проход вырождается в пассивное копирование.

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
    let c = textureSample(scene_tex, scene_sampler, in.uv);
    let darkened = c.rgb * post.p.rgb;
    return vec4<f32>(mix(c.rgb, darkened, post.p.a), c.a);
}
