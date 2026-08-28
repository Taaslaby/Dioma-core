// dioma-render — меши: произвольная триангулированная 2D-геометрия.
//
// Вершины уже в мировых координатах (CPU-скиннинг): позиция, UV в атласе и
// per-vertex цвет. Основа для деформаций и Spine-подобной анимации.
// Камера и `world_to_ndc` — в common.wgsl (приклеивается в shaders.rs,
// см. `shaders::mesh()`); раскладка вершины — `MeshBatch::vertex_layout()`.
// Тот же исходник собирает три пайплайна: обычный, аддитивный и wireframe.

@group(1) @binding(0) var atlas_tex: texture_2d<f32>;
@group(1) @binding(1) var atlas_sampler: sampler;

struct VsIn {
    @location(0) position: vec2<f32>, // мировые координаты
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = vec4<f32>(world_to_ndc(in.position), 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(atlas_tex, atlas_sampler, in.uv);
    return texel * in.color;
}
