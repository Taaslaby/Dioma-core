//! Система частиц: инстансированные мягкие аддитивные круги (пыльца, искры).

use crate::scene::batch::{unit_quad_vertices, QuadVertex};
use crate::renderer::Renderer;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Частица: позиция, скорость, время жизни, размер, цвет, трение и гравитация.
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    /// Оставшееся время жизни, секунды.
    pub life: f32,
    /// Полное время жизни, секунды (доля `life / max_life` — прозрачность).
    pub max_life: f32,
    /// Размер квада частицы в мировых пикселях.
    pub size: f32,
    /// Цвет частицы (RGB; alpha перекрывается долей жизни при рендере).
    pub color: [f32; 4],
    /// Трение: скорость умножается на `(1 − drag·dt)` каждый кадр.
    pub drag: f32,
    /// Гравитация: `vel.y −= gravity·dt` каждый кадр.
    pub gravity: f32,
}

impl Particle {
    /// Частица без трения и гравитации с полным временем жизни.
    pub fn new(pos: [f32; 2], vel: [f32; 2], life: f32, size: f32, color: [f32; 4]) -> Self {
        Self {
            pos,
            vel,
            life,
            max_life: life,
            size,
            color,
            drag: 0.0,
            gravity: 0.0,
        }
    }
}

/// Инстанс частицы для `shaders/particle/particle.wgsl` (28 байт).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ParticleInstance {
    pos: [f32; 2],
    size: f32,
    color: [f32; 4],
}

unsafe impl Pod for ParticleInstance {}
unsafe impl Zeroable for ParticleInstance {}

/// Система частиц: CPU-список + свои вершинный/инстансный/индексный буферы.
///
/// Рендер аддитивный (мягкие круги) через частичный пайплайн рендерера:
/// [`ParticleSystem::render`] вызывает [`Renderer::render_particles`].
pub struct ParticleSystem {
    particles: Vec<Particle>,
    instance_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    capacity: usize,
}

impl ParticleSystem {
    /// Система на `capacity` частиц.
    pub fn new(device: &wgpu::Device, capacity: usize) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dioma-render particle vertices"),
            contents: bytemuck::cast_slice(&unit_quad_vertices()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dioma-render particle indices"),
            contents: bytemuck::cast_slice(&[0u16, 1, 2, 1, 3, 2]),
            usage: wgpu::BufferUsages::INDEX,
        });
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dioma-render particle instances"),
            size: (capacity * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            particles: Vec::with_capacity(capacity),
            instance_buffer,
            vertex_buffer,
            index_buffer,
            capacity,
        }
    }

    /// Добавляет частицу. Паника при переполнении вместимости.
    pub fn spawn(&mut self, particle: Particle) {
        assert!(
            self.particles.len() < self.capacity,
            "ParticleSystem переполнен: вместимость {} частиц",
            self.capacity
        );
        self.particles.push(particle);
    }

    /// Интеграция: `pos += vel·dt`; `vel *= (1 − drag·dt)`;
    /// `vel.y −= gravity·dt`; `life −= dt`; мёртвые удаляются.
    pub fn update(&mut self, dt: f32) {
        for p in &mut self.particles {
            p.pos[0] += p.vel[0] * dt;
            p.pos[1] += p.vel[1] * dt;
            let drag = (1.0 - p.drag * dt).max(0.0);
            p.vel[0] *= drag;
            p.vel[1] *= drag;
            p.vel[1] -= p.gravity * dt;
            p.life -= dt;
        }
        self.particles.retain(|p| p.life > 0.0);
    }

    /// Заливает живые частицы на GPU (alpha = доля жизни) и рисует их
    /// аддитивным частичным пайплайном рендерера.
    pub fn render(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        camera: &crate::camera::Camera,
        renderer: &mut Renderer,
    ) {
        if self.particles.is_empty() {
            return;
        }
        let data: Vec<ParticleInstance> = self
            .particles
            .iter()
            .map(|p| ParticleInstance {
                pos: p.pos,
                size: p.size,
                color: [
                    p.color[0],
                    p.color[1],
                    p.color[2],
                    p.color[3] * (p.life / p.max_life).clamp(0.0, 1.0),
                ],
            })
            .collect();
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&data));
        renderer.render_particles(queue, encoder, view, camera, self);
    }

    pub fn len(&self) -> usize {
        self.particles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    pub fn clear(&mut self) {
        self.particles.clear();
    }

    /// Вершинные слоты для `particle/particle.wgsl`: слот 0 — квад,
    /// слот 1 — инстанс (позиция, размер, цвет).
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
                array_stride: std::mem::size_of::<ParticleInstance>() as wgpu::BufferAddress,
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
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 12,
                        shader_location: 4,
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

    pub(crate) fn count(&self) -> usize {
        self.particles.len()
    }
}
