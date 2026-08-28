// dioma-render — частицы (мягкие аддитивные круги: пыльца, искры).
//
// Аддитивный блендинг: dst += src.rgb (всегда), поэтому фрагмент выдаёт
// премультиплицированный цвет color.rgb × color.a × soft. Alpha цвета уже
// умножена на долю жизни на CPU. Текстуры нет — чистый falloff; камера и
// `world_to_ndc` — в common.wgsl (приклеивается в shaders.rs,
// см. `shaders::particle()`).

struct VsIn {
    // Per-vertex (квад):
    @location(0) corner_pos: vec2<f32>,
    @location(1) corner_uv: vec2<f32>,
    // Per-instance (частица):
    @location(2) instance_pos: vec2<f32>,
    @location(3) instance_size: f32,
    @location(4) instance_color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let world = in.corner_pos * in.instance_size + in.instance_pos;

    var out: VsOut;
    out.clip_pos = vec4<f32>(world_to_ndc(world), 0.0, 1.0);
    out.uv = in.corner_uv;
    out.color = in.instance_color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = distance(in.uv, vec2<f32>(0.5)) * 2.0;
    let soft = 1.0 - smoothstep(0.0, 1.0, d);
    return vec4<f32>(in.color.rgb * in.color.a * soft, in.color.a * soft);
}
