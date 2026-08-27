//! 2D-меши: произвольная триангулированная геометрия с текстурой атласа и
//! per-vertex цветом. Вершины — в МИРОВЫХ координатах (CPU-скиннинг Spine
//! и деформации происходят до push в батч): шейдер конвертирует мир → NDC
//! камерой. Основа для скелетной анимации и деформируемых форм.

use bytemuck::{Pod, Zeroable};

/// Вершина меша (32 байта): позиция в мировых координатах, UV в атласе, цвет.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MeshVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

// f32-массивы без паддингов: Pod/Zeroable вручную (без фичи derive у bytemuck).
unsafe impl Pod for MeshVertex {}
unsafe impl Zeroable for MeshVertex {}

/// Меш: произвольная триангулированная 2D-геометрия в мировых координатах.
///
/// Индексы ссылаются на вершины меша (по 0); ремаппинг смещением базовой
/// вершины делает [`MeshBatch::upload`] при интерливировании в один буфер.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    /// Очищает вершины и индексы (для переиспользования меша кадр к кадру).
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    /// Квад: центр, размер, UV-прямоугольник и цвет. Два треугольника
    /// (индексы `[0,1,2, 2,1,3]`); UV мапятся через `uv_rect`, как в
    /// спрайтовом шейдере: `mix(uv_rect.xy, uv_rect.zw, corner_uv)`.
    pub fn quad(center: [f32; 2], size: [f32; 2], uv_rect: [f32; 4], color: [f32; 4]) -> Mesh {
        // Углы единичного квада: corner_pos ∈ [-0.5, 0.5], corner_uv 0..1.
        let corners: [([f32; 2], [f32; 2]); 4] = [
            ([-0.5, -0.5], [0.0, 0.0]),
            ([0.5, -0.5], [1.0, 0.0]),
            ([-0.5, 0.5], [0.0, 1.0]),
            ([0.5, 0.5], [1.0, 1.0]),
        ];
        let vertices: Vec<MeshVertex> = corners
            .map(|(corner_pos, corner_uv)| MeshVertex {
                position: [
                    center[0] + corner_pos[0] * size[0],
                    center[1] + corner_pos[1] * size[1],
                ],
                uv: [
                    uv_rect[0] + (uv_rect[2] - uv_rect[0]) * corner_uv[0],
                    uv_rect[1] + (uv_rect[3] - uv_rect[1]) * corner_uv[1],
                ],
                color,
            })
            .to_vec();
        Mesh { vertices, indices: vec![0, 1, 2, 2, 1, 3] }
    }

    /// Веер из треугольника: центральная вершина + `segments` вершин на ободе.
    ///
    /// UV обода — центр `uv_rect` + направление × половинный размер региона:
    /// текстура смотрится «радиально» от центра. `segments` клампится к
    /// минимуму 3. Индексы в стиле `[0, i, i+1]` с замыканием на последний.
    pub fn circle_fan(
        center: [f32; 2],
        radius: f32,
        segments: u32,
        uv_rect: [f32; 4],
        color: [f32; 4],
    ) -> Mesh {
        let segments = segments.max(3);
        let uv_center = [(uv_rect[0] + uv_rect[2]) * 0.5, (uv_rect[1] + uv_rect[3]) * 0.5];
        let uv_half = [(uv_rect[2] - uv_rect[0]) * 0.5, (uv_rect[3] - uv_rect[1]) * 0.5];

        let mut vertices = Vec::with_capacity(segments as usize + 1);
        vertices.push(MeshVertex { position: center, uv: uv_center, color });
        for i in 0..segments {
            let ang = i as f32 / segments as f32 * std::f32::consts::TAU;
            let dir = [ang.cos(), ang.sin()];
            vertices.push(MeshVertex {
                position: [center[0] + dir[0] * radius, center[1] + dir[1] * radius],
                uv: [uv_center[0] + dir[0] * uv_half[0], uv_center[1] + dir[1] * uv_half[1]],
                color,
            });
        }

        let mut indices = Vec::with_capacity(segments as usize * 3);
        for i in 0..segments {
            indices.push(0);
            indices.push(i + 1);
            indices.push((i + 1) % segments + 1);
        }
        Mesh { vertices, indices }
    }
}

/// Режим смешивания меша с уже нарисованной сценой.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshBlend {
    /// Обычный альфа-блендинг (`SrcAlpha / OneMinusSrcAlpha`) — сплошные тела.
    Alpha,
    /// Аддитивный (`SrcAlpha / One`) — свечения, огонь, вода на свету.
    Additive,
}

/// Батч мешей: CPU-список + общие вершинный/индексный буферы на GPU.
///
/// Все пушнутые меши интерливируются в ОДИН вершинный и ОДИН индексный массив
/// (индексы ремапятся смещением базовой вершины). Вместимость фиксирована:
/// `max_vertices` вершин и `max_indices` индексов (u32). Перед рендером
/// вызывайте [`Self::upload`] — контракт как у [`crate::batch::SpriteBatch`].
pub struct MeshBatch {
    meshes: Vec<Mesh>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    max_vertices: usize,
    max_indices: usize,
    drawn_vertex_count: u32,
    drawn_index_count: u32,
}

impl MeshBatch {
    /// Батч с фиксированной вместимостью `max_vertices` вершин и `max_indices`
    /// индексов (по числу вершин/индексов ВСЕХ пушнутых мешей суммарно).
    pub fn new(device: &wgpu::Device, max_vertices: usize, max_indices: usize) -> Self {
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dioma-render mesh vertices"),
            size: (max_vertices * std::mem::size_of::<MeshVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dioma-render mesh indices"),
            size: (max_indices * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            meshes: Vec::new(),
            vertex_buffer,
            index_buffer,
            max_vertices,
            max_indices,
            drawn_vertex_count: 0,
            drawn_index_count: 0,
        }
    }

    /// Добавляет меш. Паника при переполнении вместимости (как
    /// [`crate::batch::SpriteBatch::push`]).
    pub fn push(&mut self, mesh: &Mesh) {
        let pushed_vertices: usize = self.meshes.iter().map(|m| m.vertices.len()).sum();
        let pushed_indices: usize = self.meshes.iter().map(|m| m.indices.len()).sum();
        assert!(
            pushed_vertices + mesh.vertices.len() <= self.max_vertices,
            "MeshBatch переполнен: вместимость {} вершин",
            self.max_vertices
        );
        assert!(
            pushed_indices + mesh.indices.len() <= self.max_indices,
            "MeshBatch переполнен: вместимость {} индексов",
            self.max_indices
        );
        self.meshes.push(mesh.clone());
    }

    pub fn clear(&mut self) {
        self.meshes.clear();
        self.drawn_vertex_count = 0;
        self.drawn_index_count = 0;
    }

    pub fn len(&self) -> usize {
        self.meshes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }

    /// Интерливирует все пушнутые меши в один вершинный и один индексный
    /// массив и заливает их на GPU (индексы ремапятся смещением базовой
    /// вершины). Пустой батч пропускается. Вызывайте после push/clear перед
    /// рендером — контракт как у [`crate::batch::SpriteBatch::upload`].
    pub fn upload(&mut self, queue: &wgpu::Queue) {
        if self.meshes.is_empty() {
            self.drawn_vertex_count = 0;
            self.drawn_index_count = 0;
            return;
        }
        let total_vertices: usize = self.meshes.iter().map(|m| m.vertices.len()).sum();
        let total_indices: usize = self.meshes.iter().map(|m| m.indices.len()).sum();
        let mut vertices: Vec<MeshVertex> = Vec::with_capacity(total_vertices);
        let mut indices: Vec<u32> = Vec::with_capacity(total_indices);
        for mesh in &self.meshes {
            let base = vertices.len() as u32;
            vertices.extend_from_slice(&mesh.vertices);
            indices.extend(mesh.indices.iter().map(|i| i + base));
        }
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
        self.drawn_vertex_count = total_vertices as u32;
        self.drawn_index_count = total_indices as u32;
    }

    /// Вершинный layout меша: один интерливированный буфер, строка 32 байта:
    /// location 0 — позиция (Float32x2, offset 0), 1 — UV (Float32x2, offset 8),
    /// 2 — цвет (Float32x4, offset 16).
    pub(crate) fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as wgpu::BufferAddress,
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
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 2,
                },
            ],
        }
    }

    pub(crate) fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertex_buffer
    }

    pub(crate) fn index_buffer(&self) -> &wgpu::Buffer {
        &self.index_buffer
    }

    /// Количество вершин на GPU (после [`Self::upload`]).
    ///
    /// Draw использует только индексный счётчик, но вершинный держим для
    /// симметрии и будущих проверок вместимости.
    #[allow(dead_code)]
    pub(crate) fn drawn_vertex_count(&self) -> u32 {
        self.drawn_vertex_count
    }

    /// Количество индексов на GPU (после [`Self::upload`]).
    pub(crate) fn drawn_index_count(&self) -> u32 {
        self.drawn_index_count
    }
}
