// dioma-render: шейдер растворения/смерти существа.
//
// вершинная часть синхронизирована с shaders/sprite.wgsl
// (атрибуты 0–10 те же; параметры эффекта:
// fx_a = (progress, edge_width, 0, 0), fx_b = цвет границы растворения).
//
// Пропорция progress (0..1) пикселей уничтожается (discard) по
// детерминированному целочисленному хэшу ячейки UV; у границы растворения
// (хэш чуть больше progress) рисуется светлая полоса edge_width.

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

// Детерминированный целочисленный хэш (одинаков на всех GPU — тесты стабильны).
fn hash_u(v: vec2<u32>) -> u32 {
    var h: u32 = v.x * 1664525u + v.y * 1013904223u + 374761393u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return h ^ (h >> 16u);
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
    let h = f32(hash_u(cell) & 0xFFFFFFu) / 16777216.0;

    if (h < progress) {
        discard; // этот пиксель уже растворён
    }

    // Светлая полоса у границы растворения (h чуть больше progress).
    let band = smoothstep(0.0, edge_width, h - progress);
    let rgb = mix(in.fx_b.rgb, base.rgb, band);
    let alpha = mix(in.fx_b.a, base.a, band);
    return vec4<f32>(rgb, alpha);
}
