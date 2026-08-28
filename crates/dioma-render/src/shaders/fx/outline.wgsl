// dioma-render — контур выделенного существа (`SpriteFx::Outline`).
//
// Спрайтовое семейство: атрибуты 0–10 те же, что в sprite.wgsl; камера и
// хелперы — в common.wgsl (приклеивается в shaders.rs, см. `shaders::fx_outline()`).
// Параметры эффекта: fx_a = (width, 0, 0, 0), fx_b = цвет контура
// (width — ширина полосы в текселях атласа).
//
// Фрагмент рисует спрайт как обычно, а вокруг края его альфы добавляет
// полосу контура: на расстоянии `width` (в текселях) по 8 направлениям
// ищутся опорные точки с высокой альфой; их максимум задаёт силу контура.

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
    @location(8) instance_fx_id: u32, // не читается — на каждый эффект свой пайплайн
    @location(9) instance_fx_a: vec4<f32>,
    @location(10) instance_fx_b: vec4<f32>,
    // Вертикальный градиент tint: значение у НИЗА квада (верх — instance_tint).
    @location(11) instance_tint_end: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
    @location(2) fx_a: vec4<f32>,
    @location(3) fx_b: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let world = quad_world(
        in.corner_pos,
        in.instance_size,
        pulse_scale(in.instance_pulse),
        0.0,
        in.instance_rot,
        in.instance_pos,
    );

    var out: VsOut;
    out.clip_pos = vec4<f32>(world_to_ndc(world), 0.0, 1.0);
    out.uv = mix(in.instance_uv_rect.xy, in.instance_uv_rect.zw, in.corner_uv);
    out.tint = mix(in.instance_tint_end, in.instance_tint, in.corner_uv.y);
    out.fx_a = in.instance_fx_a;
    out.fx_b = in.instance_fx_b;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(atlas_tex, atlas_sampler, in.uv);
    let base = texel * in.tint;

    // Внутри спрайта (альфа > 0.5) — обычный цвет без контура.
    if (base.a > 0.5) {
        return base;
    }

    // Полоса контура: выборка по 8 направлениям на расстоянии width (тексели).
    let texel_size = vec2<f32>(1.0) / vec2<f32>(textureDimensions(atlas_tex));
    let off = in.fx_a.x * texel_size;
    var cover: f32 = 0.0;
    var dirs = array<vec2<f32>, 8>(
        vec2<f32>( 1.0, 0.0),
        vec2<f32>( 0.70710678,  0.70710678),
        vec2<f32>( 0.0, 1.0),
        vec2<f32>(-0.70710678,  0.70710678),
        vec2<f32>(-1.0, 0.0),
        vec2<f32>(-0.70710678, -0.70710678),
        vec2<f32>( 0.0, -1.0),
        vec2<f32>( 0.70710678, -0.70710678),
    );
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let n = textureSample(atlas_tex, atlas_sampler, in.uv + dirs[i] * off);
        cover = max(cover, n.a);
    }
    if (cover <= 0.0) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(in.fx_b.rgb, cover * in.fx_b.a);
}
