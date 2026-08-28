//! CPU-батч спрайтов + построение вершинных/инстансных буферов.

use crate::scene::sprite::Sprite;
use bytemuck::{Pod, Zeroable};
use std::ops::Range;
use wgpu::util::DeviceExt;

/// Вершина квада (per-vertex): позиция в единичном кваде + UV угла.
///
/// Используется и спрайтами, и источниками света, и частицами — общий
/// единичный квад с UV 0..1, который инстансы масштабируют и сдвигают.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct QuadVertex {
    pub corner_pos: [f32; 2],
    pub corner_uv: [f32; 2],
}

// f32-массивы без паддингов: Pod/Zeroable вручную (без фичи derive у bytemuck).
unsafe impl Pod for QuadVertex {}
unsafe impl Zeroable for QuadVertex {}

/// Инстансные атрибуты спрайта (per-instance), строка 120 байт.
///
/// Раскладка (смещения/шейдерные location см. в [`SpriteBatch::vertex_layouts`]):
/// позиция, размер, поворот, тинт, UV-прямоугольник, пульс, id эффекта и два
/// vec4 параметров эффекта (см. [`SpriteFx::params`]).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct InstanceData {
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub rotation: f32,
    pub tint: [f32; 4],
    pub uv_rect: [f32; 4],
    pub pulse: [f32; 4],
    pub fx_id: u32,
    pub fx_a: [f32; 4],
    pub fx_b: [f32; 4],
    pub tint_end: [f32; 4],
}

unsafe impl Pod for InstanceData {}
unsafe impl Zeroable for InstanceData {}

impl From<&Sprite> for InstanceData {
    fn from(s: &Sprite) -> Self {
        let (fx_a, fx_b) = s.fx.params();
        Self {
            position: s.position,
            size: s.size,
            rotation: s.rotation,
            tint: s.tint,
            uv_rect: s.uv_rect,
            pulse: s.pulse,
            fx_id: s.fx.id(),
            fx_a,
            fx_b,
            tint_end: s.tint_end,
        }
    }
}

/// Батч спрайтов: CPU-список инстансов + GPU-буферы.
///
/// Геометрия квада (4 вершины единичного квада с UV 0..1 + 6 индексов)
/// создаётся один раз при `new`; инстанс-буфер резервируется на
/// `capacity` спрайтов. Перед рендером вызывайте [`Self::upload`],
/// который сортирует спрайты по слою и заливает инстанс-данные на GPU.
pub struct SpriteBatch {
    sprites: Vec<Sprite>,
    /// Спрайты, реально ушедшие на GPU после последнего [`Self::upload`]
    /// с активным прямоугольником отсечения ([`Self::set_cull_rect`]).
    /// Пуст при отключённом отсечении — тогда рисуется весь `sprites`.
    visible: Vec<Sprite>,
    /// Мировой прямоугольник отсечения `[min_x, min_y, max_x, max_y]`
    /// (`None` — без отсечения).
    cull_rect: Option<[f32; 4]>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    capacity: usize,
    dirty: bool,
}

impl SpriteBatch {
    /// Батч с фиксированной вместимостью `capacity` спрайтов.
    pub fn new(device: &wgpu::Device, capacity: usize) -> Self {
        let quad_vertices = unit_quad_vertices();
        let quad_indices: [u16; 6] = [0, 1, 2, 1, 3, 2];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dioma-render quad vertices"),
            contents: bytemuck::cast_slice(&quad_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dioma-render quad indices"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dioma-render instances"),
            size: (capacity * std::mem::size_of::<InstanceData>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            sprites: Vec::with_capacity(capacity),
            visible: Vec::new(),
            cull_rect: None,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            capacity,
            dirty: false,
        }
    }

    /// Добавляет спрайт. Паника при переполнении вместимости (см. [`Self::new`]).
    pub fn push(&mut self, sprite: Sprite) {
        assert!(
            self.sprites.len() < self.capacity,
            "SpriteBatch переполнен: вместимость {} инстансов",
            self.capacity
        );
        self.sprites.push(sprite);
        self.dirty = true;
    }

    pub fn clear(&mut self) {
        if !self.sprites.is_empty() {
            self.dirty = true;
        }
        self.sprites.clear();
    }

    pub fn len(&self) -> usize {
        self.sprites.len()
    }

    /// Итератор по спрайтам батча (для тестов/отладки).
    pub fn iter(&self) -> impl Iterator<Item = &Sprite> {
        self.sprites.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.sprites.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Текущий CPU-список спрайтов (уже отсортирован после [`Self::upload`]).
    /// Например, для пикинга по позициям.
    pub fn sprites(&self) -> &[Sprite] {
        &self.sprites
    }

    /// Количество спрайтов, уходящих на GPU ([`Self::upload`] с активным
    /// прямоугольником отсечения может рисовать меньше, чем пушнуто).
    pub fn len_drawn(&self) -> usize {
        self.drawn().len()
    }

    /// Устанавливает мировой прямоугольник отсечения
    /// `[min_x, min_y, max_x, max_y]`: при следующем [`Self::upload`] на GPU
    /// уйдут только спрайты с центром внутри него (границы включены).
    ///
    /// Проходит без ценой пересборки батча: полный список сохраняется,
    /// фильтрация происходит при заливке инстансов. `None` — вернуть полное
    /// рисование (по умолчанию). Готовый прямоугольник экрана камеры даёт
    /// [`crate::Camera::visible_world_rect`].
    pub fn set_cull_rect(&mut self, rect: Option<[f32; 4]>) {
        if self.cull_rect != rect {
            self.cull_rect = rect;
            self.dirty = true;
        }
    }

    /// Список спрайтов, который рисует проход: отфильтрованный при активном
    /// отсечении или полный список иначе.
    fn drawn(&self) -> &[Sprite] {
        if self.cull_rect.is_some() {
            &self.visible
        } else {
            &self.sprites
        }
    }

    /// Группы подряд идущих спрайтов с одинаковым id эффекта:
    /// `(fx_id, диапазон индексов)` — по одному draw call на группу.
    ///
    /// Батч отсортирован по (слою, затем глубине Y) — стабильная сортировка,
    /// поэтому спрайты одного слоя/глубины остаются подряд, а группировка
    /// по fx_id даёт минимум переключений пайплайна.
    pub fn iter_fx_groups(&self) -> impl Iterator<Item = (u32, Range<usize>)> + '_ {
        let sprites = self.drawn();
        let mut i = 0;
        std::iter::from_fn(move || {
            if i >= sprites.len() {
                return None;
            }
            let id = sprites[i].fx.id();
            let start = i;
            while i < sprites.len() && sprites[i].fx.id() == id {
                i += 1;
            }
            Some((id, start..i))
        })
    }

    /// Сортирует спрайты для отрисовки (слой, затем глубина Y) и заливает
    /// инстансы на GPU. Вызывайте после push/clear перед рендером.
    pub fn upload(&mut self, queue: &wgpu::Queue) {
        if !self.dirty || self.sprites.is_empty() {
            return;
        }
        Sprite::sort_for_draw(&mut self.sprites);
        // Отсечение: раздельные заимствования полей — одна аллокация не нужна.
        let Self {
            sprites,
            visible,
            cull_rect,
            ..
        } = self;
        if let Some(rect) = *cull_rect {
            visible.clear();
            visible.extend(
                sprites
                    .iter()
                    .filter(|s| {
                        s.position[0] >= rect[0]
                            && s.position[0] <= rect[2]
                            && s.position[1] >= rect[1]
                            && s.position[1] <= rect[3]
                    })
                    .copied(),
            );
        }
        let data: Vec<InstanceData> = self.drawn().iter().map(InstanceData::from).collect();
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&data));
        self.dirty = false;
    }

    /// Описания вершинных буферов: слот 0 — вершины квада, слот 1 — инстансы.
    ///
    /// Инстанс (104 байта), location 2..=10:
    /// 2 — позиция, 3 — размер, 4 — поворот, 5 — тинт, 6 — UV-прямоугольник,
    /// 7 — пульс, 8 — id эффекта (u32), 9/10 — параметры эффекта (fx_a, fx_b).
    pub(crate) fn vertex_layouts() -> [wgpu::VertexBufferLayout<'static>; 2] {
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
                array_stride: std::mem::size_of::<InstanceData>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 2,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 8,
                        shader_location: 3,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32,
                        offset: 16,
                        shader_location: 4,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 20,
                        shader_location: 5,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 36,
                        shader_location: 6,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 52,
                        shader_location: 7,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Uint32,
                        offset: 68,
                        shader_location: 8,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 72,
                        shader_location: 9,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 88,
                        shader_location: 10,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 104,
                        shader_location: 11,
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

    pub(crate) fn instance_buffer(&self) -> &wgpu::Buffer {
        &self.instance_buffer
    }
}

/// 4 вершины единичного квада (corner_pos ∈ [-0.5, 0.5]) с UV 0..1.
pub(crate) fn unit_quad_vertices() -> [QuadVertex; 4] {
    [
        QuadVertex {
            corner_pos: [-0.5, -0.5],
            corner_uv: [0.0, 0.0],
        },
        QuadVertex {
            corner_pos: [0.5, -0.5],
            corner_uv: [1.0, 0.0],
        },
        QuadVertex {
            corner_pos: [-0.5, 0.5],
            corner_uv: [0.0, 1.0],
        },
        QuadVertex {
            corner_pos: [0.5, 0.5],
            corner_uv: [1.0, 1.0],
        },
    ]
}
