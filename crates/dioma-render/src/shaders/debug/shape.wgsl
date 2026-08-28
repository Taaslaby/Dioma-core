// dioma-render — debug-шейпы (контурные круги/прямоугольники/линии).
//
// Та же раскладка вершины, что у мешей ([`crate::mesh::MeshVertex`]: позиция
// в мировых координатах + UV + цвет), но БЕЗ текстуры: фрагмент красит
// чистым per-vertex цветом. UV-атрибут присутствует и игнорируется — layout
// `MeshBatch::vertex_layout()` переиспользуется как есть. Камера и
// `world_to_ndc` — в common.wgsl (приклеивается в shaders.rs,
// см. `shaders::debug_shape()`).

struct VsIn {
    @location(0) position: vec2<f32>, // мировые координаты
    @location(1) uv: vec2<f32>,       // игнорируется (совместимость с мешами)
    @location(2) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = vec4<f32>(world_to_ndc(in.position), 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
