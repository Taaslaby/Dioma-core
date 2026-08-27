// dioma-render: шейдер волнистости/болезненной дрожи (больные существа, вода).
//
// вершинная часть синхронизирована с shaders/sprite.wgsl
// NDC: Y вверх = верх экрана (соглашение сайд-вью); мировая Y растёт вверх.
// (атрибуты 0–10 те же; параметры эффекта: fx_a = (amplitude, frequency, speed, 0)).
//
// Вершинное смещение по X до поворота:
//   x_off = amplitude × sin(мировая_Y × frequency + время × speed)

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
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let pulse_scale = mix(
        in.instance_pulse.x,
        in.instance_pulse.y,
        0.5 + 0.5 * sin(cam.camera.w * in.instance_pulse.z + in.instance_pulse.w),
    );

    // Мировая Y вершины до поворота (для фазы волны).
    let wob_y = in.instance_pos.y + in.corner_pos.y * in.instance_size.y;
    let x_off = in.instance_fx_a.x
        * sin(wob_y * in.instance_fx_a.y + cam.camera.w * in.instance_fx_a.z);

    let scaled = in.corner_pos * in.instance_size * pulse_scale + vec2<f32>(x_off, 0.0);
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
        (world.y - center.y) / (vp.y * 0.5 / zoom),
    );

    var out: VsOut;
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = mix(in.instance_uv_rect.xy, in.instance_uv_rect.zw, in.corner_uv);
    out.tint = in.instance_tint;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(atlas_tex, atlas_sampler, in.uv);
    return texel * in.tint;
}
