// dioma-render — радиальный свет (аддитивный слой подсветки).
//
// Квад света имеет размер 2×radius; falloff (1 − smoothstep(0,1,dist))
// доходит до нуля на краях квада. Текстура не нужна — чистый математический
// falloff; группа 1 не используется. Камера и `world_to_ndc` — в common.wgsl
// (приклеивается в shaders.rs, см. `shaders::light_radial()`).
//
// Вместе с пост-проходом «тьма» (post/darkness.wgsl) даёт цикл дня и ночи:
// свет рисуется ПОВЕРХ тьмы и «пробивает» её собой.

struct VsIn {
    // Per-vertex (квад):
    @location(0) corner_pos: vec2<f32>,
    @location(1) corner_uv: vec2<f32>,
    // Per-instance (источник света):
    @location(2) instance_pos: vec2<f32>,
    @location(3) instance_radius: f32,
    @location(4) instance_intensity: f32,
    @location(5) instance_color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) intensity: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let world = in.corner_pos * (2.0 * in.instance_radius) + in.instance_pos;

    var out: VsOut;
    out.clip_pos = vec4<f32>(world_to_ndc(world), 0.0, 1.0);
    out.uv = in.corner_uv;
    out.color = in.instance_color;
    out.intensity = in.instance_intensity;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dist = distance(in.uv, vec2<f32>(0.5)) * 2.0;
    let fall = 1.0 - smoothstep(0.0, 1.0, dist);
    let a = in.intensity * fall;
    return vec4<f32>(in.color.rgb * a, a);
}
