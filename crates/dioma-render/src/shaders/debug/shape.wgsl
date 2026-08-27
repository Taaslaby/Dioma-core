// dioma-render: пайплайн debug-шейпов (контурные круги/прямоугольники/линии).
//
// Та же раскладка вершины, что у мешей ([`crate::mesh::MeshVertex`]: позиция в
// мировых координатах + UV + цвет), но БЕЗ текстуры: фрагмент красит чистым
// per-vertex цветом. UV-атрибут присутствует и игнорируется — layout
// `MeshBatch::vertex_layout()` переиспользуется как есть.
//
// Uniform камеры 2×vec4 (как во всех проходах):
//   camera.xy    — центр камеры (мировые координаты)
//   camera.z     — зум
//   camera.w     — время (секунды, не используется)
//   viewport.xy  — размер вьюпорта в пикселях

struct Camera {
    camera: vec4<f32>,
    viewport: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam: Camera;

struct VsIn {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    // Мир → NDC: то же преобразование, что в спрайтовом и мешевом шейдерах
    // (мировая Y вверх = верх экрана).
    let center = cam.camera.xy;
    let zoom = cam.camera.z;
    let vp = cam.viewport.xy;
    let ndc = vec2<f32>(
        (in.position.x - center.x) / (vp.x * 0.5 / zoom),
        (in.position.y - center.y) / (vp.y * 0.5 / zoom),
    );

    var out: VsOut;
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
