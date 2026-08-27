//! Радиальные источники света — аддитивный слой подсветки поверх спрайтов.

use crate::batch::{unit_quad_vertices, QuadVertex};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Радиальный источник света.
///
/// * `position` — центр света в мировых координатах;
/// * `radius` — радиус falloff (свет гаснет к краю квада 2×radius);
/// * `intensity` — сила 0..1+ (аддитивно добавляется к сцене);
/// * `color` — цвет света (RGB).
#[derive(Clone, Copy, Debug, Default)]
pub struct Light {
    pub position: [f32; 2],
    pub radius: f32,
    pub intensity: f32,
    pub color: [f32; 3],
}

/// Инстанс источника света для `shaders/light/radial.wgsl` (32 байта).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct LightInstance {
    position: [f32; 2],
    radius: f32,
    intensity: f32,
    color: [f32; 4],
}

unsafe impl Pod for LightInstance {}
unsafe impl Zeroable for LightInstance {}

impl From<&Light> for LightInstance {
    fn from(l: &Light) -> Self {
        Self {
            position: l.position,
            radius: l.radius,
            intensity: l.intensity,
            color: [l.color[0], l.color[1], l.color[2], 1.0],
        }
    }
}

/// Батч источников света: свой инстанс-буфер (вместимость фиксирована).
///
/// Вершинный буфер — общий единичный квад; инстансы масштабируют его до
/// 2×radius. Перед рендером вызывайте [`Self::upload`].
pub struct LightBatch {
    lights: Vec<Light>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    capacity: usize,
    dirty: bool,
}

impl LightBatch {
    /// Батч на `capacity` источников света.
    pub fn new(device: &wgpu::Device, capacity: usize) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dioma-render light vertices"),
            contents: bytemuck::cast_slice(&unit_quad_vertices()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dioma-render light indices"),
            contents: bytemuck::cast_slice(&[0u16, 1, 2, 1, 3, 2]),
            usage: wgpu::BufferUsages::INDEX,
        });
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dioma-render light instances"),
            size: (capacity * std::mem::size_of::<LightInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            lights: Vec::with_capacity(capacity),
            vertex_buffer,
            index_buffer,
            instance_buffer,
            capacity,
            dirty: false,
        }
    }

    /// Добавляет источник света. Паника при переполнении вместимости.
    pub fn push(&mut self, light: Light) {
        assert!(
            self.lights.len() < self.capacity,
            "LightBatch переполнен: вместимость {} инстансов",
            self.capacity
        );
        self.lights.push(light);
        self.dirty = true;
    }

    pub fn clear(&mut self) {
        if !self.lights.is_empty() {
            self.dirty = true;
        }
        self.lights.clear();
    }

    pub fn len(&self) -> usize {
        self.lights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lights.is_empty()
    }

    /// Заливает инстансы на GPU. Вызывайте после push/clear перед рендером.
    pub fn upload(&mut self, queue: &wgpu::Queue) {
        if !self.dirty || self.lights.is_empty() {
            return;
        }
        let data: Vec<LightInstance> = self.lights.iter().map(LightInstance::from).collect();
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&data));
        self.dirty = false;
    }

    /// Вершинные слоты для `light/radial.wgsl`: слот 0 — квад, слот 1 — инстанс
    /// (позиция, радиус, интенсивность, цвет).
    pub(crate) fn vertex_layout() -> [wgpu::VertexBufferLayout<'static>; 2] {
        [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<QuadVertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 8,
                        shader_location: 1,
                    },
                ],
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<LightInstance>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 2,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32,
                        offset: 8,
                        shader_location: 3,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32,
                        offset: 12,
                        shader_location: 4,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 16,
                        shader_location: 5,
                    },
                ],
            },
        ]
    }

    pub(crate) fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertex_buffer
    }

    pub(crate) fn index_buffer(&self) -> &wgpu::Buffer {
        &self.index_buffer
    }

    /// Инстанс-буфер источников света (заполнен после [`Self::upload`]).
    pub fn instance_buffer(&self) -> &wgpu::Buffer {
        &self.instance_buffer
    }

    /// Количество источников на GPU (после [`Self::upload`]).
    pub fn count(&self) -> usize {
        self.lights.len()
    }
}
