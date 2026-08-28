// dioma-render — мягкая drop-тень силуэта (`SpriteFx::Shadow`).
//
// Спрайтовое семейство: атрибуты 0–10 те же, что в sprite.wgsl; камера и
// хелперы — в common.wgsl (приклеивается в shaders.rs, см. `shaders::fx_shadow()`).
// Параметры эффекта: fx_a = (offset_x, offset_y, radius, 0), fx_b = цвет тени.
// offset — в текселях атласа (uv-пространство: y = вверх мира); radius —
// радиус размытия края в текселях.
//
// Фрагмент: тело спрайта рисуется как обычно; вокруг него — тень: сэмплы
// альфы силуэта вокруг точки, СМЕЩЁННОЙ на offset (тень = силуэт, сдвинутый
// и размытый). Два кольца сэмплов (0.5r и 1.0r) дают мягкий край.

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

    // Тело спрайта рисуется непрозрачно, тень скрывается под ним.
    if (base.a > 0.5) {
        return base;
    }

    let texel_size = vec2<f32>(1.0) / vec2<f32>(textureDimensions(atlas_tex));
    // Тень = силуэт, сдвинутый на offset: альфу ищем в точке uv − offset.
    let center = in.uv - in.fx_a.xy * texel_size;
    let r = in.fx_a.z * texel_size;

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

    // Центр + два кольца (0.5r, 1.0r): 17 сэмплов — мягкий край.
    var acc: f32 = textureSampleLevel(atlas_tex, atlas_sampler, center, 0.0).a;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        acc = acc + textureSampleLevel(atlas_tex, atlas_sampler, center + dirs[i] * r * 0.5, 0.0).a;
        acc = acc + textureSampleLevel(atlas_tex, atlas_sampler, center + dirs[i] * r, 0.0).a;
    }
    let coverage = clamp(acc / 17.0, 0.0, 1.0);

    return vec4<f32>(in.fx_b.rgb, in.fx_b.a * coverage);
}
