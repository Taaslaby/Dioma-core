// dioma-render — волнистость/болезненная дрожь (`SpriteFx::Wobble`).
//
// Спрайтовое семейство: атрибуты 0–10 те же, что в sprite.wgsl; камера и
// хелперы — в common.wgsl (приклеивается в shaders.rs, см. `shaders::fx_wobble()`).
// Параметры эффекта: fx_a = (amplitude, frequency, speed, 0).
//
// Единственный эффект со смещением: горизонтальное дрожание вершины
//   x_off = amplitude × sin(мировая_Y × frequency + время × speed)
// выполняется ДО поворота — передаётся в `quad_world` как x_off.

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
    // Параметры эффекта (общие для fx-семейства):
    @location(8) instance_fx_id: u32,
    @location(9) instance_fx_a: vec4<f32>,
    @location(10) instance_fx_b: vec4<f32>,
    // Вертикальный градиент tint: значение у НИЗА квада (верх — instance_tint).
    @location(11) instance_tint_end: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    // Мировая Y вершины до поворота (фаза волны привязана к точке в мире).
    let wob_y = in.instance_pos.y + in.corner_pos.y * in.instance_size.y;
    let x_off = in.instance_fx_a.x
        * sin(wob_y * in.instance_fx_a.y + cam.camera.w * in.instance_fx_a.z);

    let world = quad_world(
        in.corner_pos,
        in.instance_size,
        pulse_scale(in.instance_pulse),
        x_off,
        in.instance_rot,
        in.instance_pos,
    );

    var out: VsOut;
    out.clip_pos = vec4<f32>(world_to_ndc(world), 0.0, 1.0);
    out.uv = mix(in.instance_uv_rect.xy, in.instance_uv_rect.zw, in.corner_uv);
    out.tint = mix(in.instance_tint_end, in.instance_tint, in.corner_uv.y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(atlas_tex, atlas_sampler, in.uv);
    return texel * in.tint;
}
