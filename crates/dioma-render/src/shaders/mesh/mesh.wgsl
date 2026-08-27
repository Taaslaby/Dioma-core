// dioma-render: mesh-пайплайн: вершины уже в мировых координатах (CPU-скиннинг).
//
// Произвольная триангулированная 2D-геометрия: per-vertex позиция (мир),
// UV в атласе и цвет. Мир → NDC камерой — то же преобразование, что в
// спрайтовом шейдере (uniform камеры 2×vec4):
//   camera.xy    — центр камеры (мировые координаты)
//   camera.z     — зум
//   camera.w     — время (секунды)
//   viewport.xy  — размер вьюпорта в пикселях
//   viewport.zw  — не используется (0, 0)

struct Camera {
    camera: vec4<f32>,
    viewport: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam: Camera;

@group(1) @binding(0) var atlas_tex: texture_2d<f32>;
@group(1) @binding(1) var atlas_sampler: sampler;

struct VsIn {
    @location(0) position: vec2<f32>,
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
    // Мир → NDC (ось Y вверх), как в спрайтовом шейдере.
    let center = cam.camera.xy;
    let zoom = cam.camera.z;
    let vp = cam.viewport.xy;
    let ndc = vec2<f32>(
        (in.position.x - center.x) / (vp.x * 0.5 / zoom),
        -(in.position.y - center.y) / (vp.y * 0.5 / zoom),
    );

    var out: VsOut;
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(atlas_tex, atlas_sampler, in.uv);
    return texel * in.color;
}
