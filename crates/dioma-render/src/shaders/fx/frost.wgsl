// dioma-render — стеклянная (frosted glass) подложка панели UI.
//
// Спрайтовое семейство (квад с камерой, инстансные location 0..11), плюс
// группа 2 — РАЗМЫТАЯ сцена за панелью (см. `shaders::fx_frost()` и
// `Renderer::set_ui_backdrop`). Блюр делает `PostProcess::apply(Blur)`
// в текстуру [`crate::ui::GlassBackdrop`]; здесь только сэмплирование.
//
// Инстанс = прямоугольник панели: uv_rect — маска скругления (регион
// «ui/panel»), tint — цвет/прозрачность стекла (RGB множитель сцены,
// A — непрозрачность), tint_end — то же для нижнего края (градиент).
//
// Экранные UV берутся из FragCoord / viewport: v0 текстуры рендер-цели —
// верх экрана, поэтому флип не нужен.

@group(1) @binding(0) var atlas_tex: texture_2d<f32>;
@group(1) @binding(1) var atlas_sampler: sampler;

@group(2) @binding(0) var backdrop_tex: texture_2d<f32>;
@group(2) @binding(1) var backdrop_sampler: sampler;

struct VsIn {
    @location(0) corner_pos: vec2<f32>,
    @location(1) corner_uv: vec2<f32>,
    @location(2) instance_pos: vec2<f32>,
    @location(3) instance_size: vec2<f32>,
    @location(5) instance_tint: vec4<f32>,
    @location(6) instance_uv_rect: vec4<f32>,
    // Градиент стекла (верх/низ).
    @location(11) instance_tint_end: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,  // маска скругления в атласе
    @location(1) tint: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let world = quad_world(in.corner_pos, in.instance_size, 1.0, 0.0, 0.0, in.instance_pos);

    var out: VsOut;
    out.clip_pos = vec4<f32>(world_to_ndc(world), 0.0, 1.0);
    // V-флип как в sprite.wgsl: v0 — верх изображения-маски.
    out.uv = mix(
        in.instance_uv_rect.xy,
        in.instance_uv_rect.zw,
        vec2<f32>(in.corner_uv.x, 1.0 - in.corner_uv.y),
    );
    out.tint = mix(in.instance_tint_end, in.instance_tint, in.corner_uv.y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Экранные UV: FragCoord (px, y вниз) / viewport (v0 — верх рендер-цели).
    let vp = max(cam.viewport.xy, vec2<f32>(1.0));
    let base = in.clip_pos.xy / vp;
    // 5 сэмплов с микро-сдвигом — дешёвое досглаживание поверх блюра сцены.
    let o = vec2<f32>(1.5) / vp;
    var scene = textureSampleLevel(backdrop_tex, backdrop_sampler, base, 0.0);
    scene += textureSampleLevel(backdrop_tex, backdrop_sampler, base + vec2<f32>(o.x, 0.0), 0.0);
    scene += textureSampleLevel(backdrop_tex, backdrop_sampler, base - vec2<f32>(o.x, 0.0), 0.0);
    scene += textureSampleLevel(backdrop_tex, backdrop_sampler, base + vec2<f32>(0.0, o.y), 0.0);
    scene += textureSampleLevel(backdrop_tex, backdrop_sampler, base - vec2<f32>(0.0, o.y), 0.0);
    scene = scene / 5.0;

    // Маска скруглённого прямоугольника + цвет стекла.
    let mask = textureSample(atlas_tex, atlas_sampler, in.uv);
    let rgb = scene.rgb * in.tint.rgb;
    return vec4<f32>(rgb, mask.a * in.tint.a);
}
