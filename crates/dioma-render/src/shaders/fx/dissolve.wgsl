// dioma-render — растворение/смерть существа (`SpriteFx::Dissolve`).
//
// Спрайтовое семейство: атрибуты 0–10 те же, что в sprite.wgsl; камера и
// хелперы — в common.wgsl (приклеивается в shaders.rs, см. `shaders::fx_dissolve()`).
// Параметры эффекта: fx_a = (progress, edge_width, 0, 0), fx_b = цвет границы.
//
// Пропорция progress (0..1) пикселей уничтожается (discard) по
// детерминированному хэшу ячейки UV (`hash01` из common.wgsl — одинаков на
// всех GPU); у границы растворения (хэш чуть больше progress) рисуется
// светлая полоса edge_width.

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

    let progress = in.fx_a.x;
    let edge_width = in.fx_a.y;

    if (progress <= 0.0) {
        return base; // не началось — спрайт цел
    }
    if (progress >= 1.0) {
        discard; // закончилось — спрайт полностью растворён
    }
    if (base.a <= 0.01) {
        discard; // пустота не рисуется
    }

    // Сетка 1024×1024 по UV спрайта — узор привязан к спрайту, не к экрану.
    let cell = vec2<u32>(floor(in.uv * vec2<f32>(1024.0)));
    let h = hash01(cell);

    if (h < progress) {
        discard; // этот пиксель уже растворён
    }

    // Светлая полоса у границы растворения (h чуть больше progress).
    let band = smoothstep(0.0, edge_width, h - progress);
    let rgb = mix(in.fx_b.rgb, base.rgb, band);
    let alpha = mix(in.fx_b.a, base.a, band);
    return vec4<f32>(rgb, alpha);
}
