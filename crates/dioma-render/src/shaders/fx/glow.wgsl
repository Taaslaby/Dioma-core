// dioma-render — внешнее свечение (`SpriteFx::Glow`).
//
// Счастливые существа, дофаминовое сияние. Спрайтовое семейство: атрибуты
// 0–10 те же, что в sprite.wgsl; камера и хелперы — в common.wgsl
// (приклеивается в shaders.rs, см. `shaders::fx_glow()`).
// Параметры эффекта: fx_a = (radius, intensity, 0, 0), fx_b = цвет свечения
// (radius/intensity — в мировых единицах).
//
// Спрайт рисуется как обычно, ПЛЮС внешнее свечение: радиальное кольцо у краёв
// квада (расстояние от края в UV). Альфа-блендинг; тело спрайта остаётся
// непрозрачным (множитель (1 − base.a) убирает свечение внутри него).

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
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
    @location(2) size: vec2<f32>,
    @location(3) fx_a: vec4<f32>,
    @location(4) fx_b: vec4<f32>,
    @location(5) corner_uv: vec2<f32>,
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
    out.tint = in.instance_tint;
    out.size = in.instance_size;
    out.fx_a = in.instance_fx_a;
    out.fx_b = in.instance_fx_b;
    out.corner_uv = in.corner_uv;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(atlas_tex, atlas_sampler, in.uv);
    let base = texel * in.tint;

    // Внутри спрайта (альфа > 0.5) — обычный цвет без свечения.
    if (base.a > 0.5) {
        return base;
    }

    // Расстояние от края квада в мировых единицах (по corner_uv — UV 0..1 по кваду).
    let radius = in.fx_a.x;
    let intensity = in.fx_a.y;
    let c = in.corner_uv;
    let edge_x = min(c.x, 1.0 - c.x) * in.size.x;
    let edge_y = min(c.y, 1.0 - c.y) * in.size.y;
    let edge_world = min(edge_x, edge_y);
    let fall = 1.0 - smoothstep(0.0, 1.0, edge_world / radius);
    let glow = intensity * fall * (1.0 - base.a);

    let rgb = base.rgb + in.fx_b.rgb * glow;
    let alpha = max(base.a, glow);
    return vec4<f32>(rgb, alpha);
}
