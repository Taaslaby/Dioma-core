// dioma-render — базовый спрайт: текстурированный квад с tint'ом.
//
// Единый инстансный квад для ВСЕГО мира: тайлы, существа, ягоды, тон суток.
// Камера и хелперы (`world_to_ndc`, `pulse_scale`, `quad_world`) — в
// common.wgsl (приклеивается в shaders.rs, см. `shaders::sprite()`).
//
// Атлас (группа 1): регион спрайта задаётся `instance_uv_rect` (xy = левый
// нижний, zw = правый верхний), интерполяция по corner_uv квада.
//
// Слои/сортировка — на CPU (layer, depth); шейдер только рисует.

// Атлас (группа 1): общая текстура всех спрайтовых проходов.
@group(1) @binding(0) var atlas_tex: texture_2d<f32>;
@group(1) @binding(1) var atlas_sampler: sampler;

struct VsIn {
    // Per-vertex (квад):
    @location(0) corner_pos: vec2<f32>,
    @location(1) corner_uv: vec2<f32>,
    // Per-instance (спрайт):
    @location(2) instance_pos: vec2<f32>,
    @location(3) instance_size: vec2<f32>,
    @location(4) instance_rot: f32,
    @location(5) instance_tint: vec4<f32>,
    @location(6) instance_uv_rect: vec4<f32>,
    @location(7) instance_pulse: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let world = quad_world(
        in.corner_pos,
        in.instance_size,
        pulse_scale(in.instance_pulse),
        0.0, // без смещения — оно нужно только волнению (fx/wobble.wgsl)
        in.instance_rot,
        in.instance_pos,
    );

    var out: VsOut;
    out.clip_pos = vec4<f32>(world_to_ndc(world), 0.0, 1.0);
    // V-флип: первая строка данных текстуры (v0) — ВЕРХ изображения, поэтому
    // верх квада (corner_uv.y = 1, мировая Y вверх) должен брать v0.
    out.uv = mix(
        in.instance_uv_rect.xy,
        in.instance_uv_rect.zw,
        vec2<f32>(in.corner_uv.x, 1.0 - in.corner_uv.y),
    );
    out.tint = in.instance_tint;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(atlas_tex, atlas_sampler, in.uv);
    return texel * in.tint;
}
