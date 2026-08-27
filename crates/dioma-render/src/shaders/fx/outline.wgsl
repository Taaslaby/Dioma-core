// dioma-render: шейдер контура/подсветки выделенного существа.
//
// вершинная часть синхронизирована с shaders/sprite.wgsl
// (атрибуты 0–10 те же; параметры эффекта: fx_a = (width, 0, 0, 0), fx_b = цвет контура).
//
// Фрагмент рисует спрайт как обычно, а вокруг края его альфы добавляет
// полосу контура: на расстоянии `width` (в текселях) по 8 направлениям
// ищутся опорные точки с высокой альфой; их максимум задаёт силу контура.

struct Camera {
    camera: vec4<f32>,
    viewport: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam: Camera;

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
    @location(8) instance_fx_id: u32,
    @location(9) instance_fx_a: vec4<f32>,
    @location(10) instance_fx_b: vec4<f32>,
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
    let pulse_scale = mix(
        in.instance_pulse.x,
        in.instance_pulse.y,
        0.5 + 0.5 * sin(cam.camera.w * in.instance_pulse.z + in.instance_pulse.w),
    );
    let scaled = in.corner_pos * in.instance_size * pulse_scale;
    let c = cos(in.instance_rot);
    let s = sin(in.instance_rot);
    let rotated = vec2<f32>(
        scaled.x * c - scaled.y * s,
        scaled.x * s + scaled.y * c,
    );
    let world = rotated + in.instance_pos;

    let center = cam.camera.xy;
    let zoom = cam.camera.z;
    let vp = cam.viewport.xy;
    let ndc = vec2<f32>(
        (world.x - center.x) / (vp.x * 0.5 / zoom),
        -(world.y - center.y) / (vp.y * 0.5 / zoom),
    );

    var out: VsOut;
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = mix(in.instance_uv_rect.xy, in.instance_uv_rect.zw, in.corner_uv);
    out.tint = in.instance_tint;
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
