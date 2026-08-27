//! Рендерер: пайплайны спрайтов (базовый + fx), мешей, света и частиц
//! + uniform камеры + профилировщик и debug-оверлей.

use crate::atlas::{atlas_bind_group_layout, SpriteAtlas};
use crate::batch::SpriteBatch;
use crate::camera::Camera;
use crate::debug::{DebugFont, RenderDebug};
use crate::light::LightBatch;
use crate::mesh::{Mesh, MeshBatch, MeshBlend};
use crate::particles::ParticleSystem;
use crate::profiler::RenderStats;

/// Аддитивный блендинг (в wgpu 22 нет готового `BlendState::ADDITIVE`).
const ADDITIVE_BLENDING: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

/// Аддитивный блендинг мешей: цвет умножается на альфу (SrcAlpha/One), чтобы
/// полупрозрачные меши (вода, туман) добавляли света пропорционально своей
/// альфе, а не на всю яркость.
const MESH_ADDITIVE_BLENDING: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

/// Рендерер инстансированных спрайтов, 2D-мешей, радиальных источников света
/// и частиц.
///
/// Владеет всеми пайплайнами (базовый спрайт, по одному на вариант
/// [`SpriteFx`], три мешевых — обычный, аддитивный и wireframe-каркас, плюс
/// свет и частицы), uniform-буфером камеры, временем, статистикой кадра
/// ([`RenderStats`]) и debug-тогглами ([`RenderDebug`]). Атлас отдаёт свой
/// bind group (текстура + сэмплер) — его bind group layout описан идентично
/// лэйауту пайплайнов, и wgpu дедуплицирует одинаковые bind group layout'ы.
pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    fx_pipelines: [wgpu::RenderPipeline; 4],
    mesh_pipeline: wgpu::RenderPipeline,
    mesh_additive_pipeline: wgpu::RenderPipeline,
    /// Wireframe-каркас мешей (`PolygonMode::Line`); `None`, если
    /// бэкенд не поддерживает `POLYGON_MODE_LINE` — флаг игнорируется.
    mesh_wireframe_pipeline: Option<wgpu::RenderPipeline>,
    /// Внутренний батч для [`Self::render_debug_overlay`] (8192 вершин /
    /// 16384 индексов).
    debug_mesh_batch: MeshBatch,
    light_pipeline: wgpu::RenderPipeline,
    particle_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    clear_color: wgpu::Color,
    time: f32,
    /// Debug-тогглы проходов и оверлея (публичное поле — меняйте напрямую).
    pub debug: RenderDebug,
    /// Статистика текущего кадра (см. [`Self::begin_frame`] и [`Self::stats`]).
    stats: RenderStats,
    /// EMA-сглаженный FPS (alpha ≈ 0.1).
    smooth_fps: f32,
    /// EMA-сглаженное время кадра в миллисекундах.
    smooth_ms: f32,
    /// Длительность предыдущего кадра в секундах (0.0 — кадр ещё не был).
    _prev_frame_time: f32,
}

impl Renderer {
    /// Создаёт ВСЕ пайплайны под целевой `format` (поверхность или offscreen):
    /// базовый спрайт, четыре fx-пайплайна (outline/glow/dissolve/wobble),
    /// три мешевых (`shaders/mesh/mesh.wgsl`: обычный, аддитивный и
    /// wireframe-каркас — последний только если устройство поддерживает
    /// `wgpu::Features::POLYGON_MODE_LINE`), световой (`light/radial.wgsl`)
    /// и частичный (`particle/particle.wgsl`).
    ///
    /// В wgpu 22 создание пайплайна синхронно и не возвращает ошибку: ошибки
    /// валидации шейдера приходят асинхронно через device error handler.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Renderer {
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dioma-render camera bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(32),
                },
                count: None,
            }],
        });
        let atlas_layout = atlas_bind_group_layout(device);

        let sprite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dioma-render sprite pipeline layout"),
            bind_group_layouts: &[&camera_layout, &atlas_layout],
            push_constant_ranges: &[],
        });

        let pipeline = build_sprite_pipeline(
            device,
            &sprite_layout,
            format,
            include_str!("shaders/sprite.wgsl"),
            "dioma-render sprite pipeline",
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );

        // По одному пайплайну на вариант SpriteFx (индекс = id − 1).
        let fx_pipelines = [
            build_sprite_pipeline(
                device,
                &sprite_layout,
                format,
                include_str!("shaders/fx/outline.wgsl"),
                "dioma-render outline pipeline",
                Some(wgpu::BlendState::ALPHA_BLENDING),
            ),
            build_sprite_pipeline(
                device,
                &sprite_layout,
                format,
                include_str!("shaders/fx/glow.wgsl"),
                "dioma-render glow pipeline",
                Some(wgpu::BlendState::ALPHA_BLENDING),
            ),
            build_sprite_pipeline(
                device,
                &sprite_layout,
                format,
                include_str!("shaders/fx/dissolve.wgsl"),
                "dioma-render dissolve pipeline",
                Some(wgpu::BlendState::ALPHA_BLENDING),
            ),
            build_sprite_pipeline(
                device,
                &sprite_layout,
                format,
                include_str!("shaders/fx/wobble.wgsl"),
                "dioma-render wobble pipeline",
                Some(wgpu::BlendState::ALPHA_BLENDING),
            ),
        ];

        // Меши: layout как у спрайтов (группа 0 — камера, группа 1 — атлас),
        // один буфер интерливированных вершин, треугольники без инстансов.
        // Два пайплайна: обычный альфа-блендинг и аддитивный, плюс
        // wireframe-каркас (полигоны-линии), если бэкенд умеет.
        let mesh_pipeline = build_pipeline(
            device,
            &sprite_layout,
            format,
            include_str!("shaders/mesh/mesh.wgsl"),
            "dioma-render mesh pipeline",
            std::slice::from_ref(&MeshBatch::vertex_layout()),
            Some(wgpu::BlendState::ALPHA_BLENDING),
            wgpu::PolygonMode::Fill,
        );
        let mesh_additive_pipeline = build_pipeline(
            device,
            &sprite_layout,
            format,
            include_str!("shaders/mesh/mesh.wgsl"),
            "dioma-render mesh additive pipeline",
            std::slice::from_ref(&MeshBatch::vertex_layout()),
            Some(MESH_ADDITIVE_BLENDING),
            wgpu::PolygonMode::Fill,
        );
        let mesh_wireframe_pipeline = if device
            .features()
            .contains(wgpu::Features::POLYGON_MODE_LINE)
        {
            Some(build_pipeline(
                device,
                &sprite_layout,
                format,
                include_str!("shaders/mesh/mesh.wgsl"),
                "dioma-render mesh wireframe pipeline",
                std::slice::from_ref(&MeshBatch::vertex_layout()),
                Some(wgpu::BlendState::ALPHA_BLENDING),
                wgpu::PolygonMode::Line,
            ))
        } else {
            None
        };

        // Свет и частицы: только камера (группа 0), текстура атласа не нужна.
        let light_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dioma-render light pipeline layout"),
            bind_group_layouts: &[&camera_layout],
            push_constant_ranges: &[],
        });
        let light_pipeline = build_pipeline(
            device,
            &light_layout,
            format,
            include_str!("shaders/light/radial.wgsl"),
            "dioma-render light pipeline",
            &LightBatch::vertex_layout(),
            Some(ADDITIVE_BLENDING),
            wgpu::PolygonMode::Fill,
        );

        let particle_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dioma-render particle pipeline layout"),
            bind_group_layouts: &[&camera_layout],
            push_constant_ranges: &[],
        });
        let particle_pipeline = build_pipeline(
            device,
            &particle_layout,
            format,
            include_str!("shaders/particle/particle.wgsl"),
            "dioma-render particle pipeline",
            &ParticleSystem::vertex_layout(),
            Some(ADDITIVE_BLENDING),
            wgpu::PolygonMode::Fill,
        );

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dioma-render camera uniform"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dioma-render camera bind group"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let debug_mesh_batch = MeshBatch::new(device, 8192, 16384);

        Renderer {
            pipeline,
            fx_pipelines,
            mesh_pipeline,
            mesh_additive_pipeline,
            mesh_wireframe_pipeline,
            debug_mesh_batch,
            light_pipeline,
            particle_pipeline,
            uniform_buffer,
            camera_bind_group,
            clear_color: wgpu::Color::BLACK,
            time: 0.0,
            debug: RenderDebug::default(),
            stats: RenderStats::default(),
            smooth_fps: 0.0,
            smooth_ms: 0.0,
            _prev_frame_time: 0.0,
        }
    }

    /// Цвет заливки (Clear) при каждом вызове [`Self::render`].
    pub fn set_clear_color(&mut self, color: wgpu::Color) {
        self.clear_color = color;
    }

    /// Устанавливает время в секундах — питает пульс, волнистость и зерно.
    pub fn set_time(&mut self, t: f32) {
        self.time = t;
    }

    /// Начало кадра: обновляет EMA-сглаженные FPS и время кадра из `dt`
    /// (секунды) и обнуляет все счётчики [`RenderStats`].
    ///
    /// Вызывайте один раз в начале кадра, ДО проходов рендера. Первый кадр
    /// (`smooth_fps == 0`) инициализируется напрямую, без EMA-разогрева.
    pub fn begin_frame(&mut self, dt: f32) {
        let dt = dt.max(0.0);
        let fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
        let frame_ms = dt * 1000.0;
        if self.smooth_fps == 0.0 {
            self.smooth_fps = fps;
            self.smooth_ms = frame_ms;
        } else {
            self.smooth_fps = self.smooth_fps * 0.9 + fps * 0.1;
            self.smooth_ms = self.smooth_ms * 0.9 + frame_ms * 0.1;
        }
        self._prev_frame_time = dt;
        self.stats.fps = self.smooth_fps;
        self.stats.frame_ms = self.smooth_ms;
        self.stats.draw_calls = 0;
        self.stats.sprites = 0;
        self.stats.fx_sprites = 0;
        self.stats.meshes = 0;
        self.stats.mesh_vertices = 0;
        self.stats.mesh_indices = 0;
        self.stats.lights = 0;
        self.stats.particles = 0;
    }

    /// Текущий снимок статистики кадра (после [`Self::begin_frame`] и
    /// проходов рендера).
    pub fn stats(&self) -> &RenderStats {
        &self.stats
    }

    /// Рисует батч спрайтов из атласа в произвольный `view` (поверхность или
    /// offscreen-текстура — отдельный offscreen-путь не нужен).
    ///
    /// Пишет uniform камеры (включая время), открывает render pass на `view`
    /// с очисткой `clear_color`, ставит bind groups камеры и атласа и рисует
    /// батч группами: подряд идущие спрайты с одинаковым fx — одним draw call
    /// через соответствующий пайплайн (fx=None — базовый). Счётчики
    /// `sprites`/`fx_sprites`/`draw_calls` накапливаются в [`RenderStats`];
    /// debug-тогглы [`RenderDebug::draw_sprites`]/[`RenderDebug::draw_fx`]
    /// отключают соответствующие группы.
    pub fn render(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        atlas: &SpriteAtlas,
        batch: &SpriteBatch,
        camera: &Camera,
    ) {
        self.write_camera_uniform(queue, camera);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dioma-render sprite pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.clear_color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        pass.set_bind_group(1, atlas.bind_group(), &[]);
        pass.set_vertex_buffer(0, batch.vertex_buffer().slice(..));
        pass.set_vertex_buffer(1, batch.instance_buffer().slice(..));
        pass.set_index_buffer(batch.index_buffer().slice(..), wgpu::IndexFormat::Uint16);
        for (fx_id, range) in batch.iter_fx_groups() {
            // Debug-тогглы: пропускаем группы выключенных пайплайнов.
            let skip = match fx_id {
                0 => !self.debug.draw_sprites,
                _ => !self.debug.draw_fx,
            };
            if skip {
                continue;
            }
            let pipeline = match fx_id {
                0 => &self.pipeline,
                id @ 1..=4 => &self.fx_pipelines[(id - 1) as usize],
                _ => unreachable!("SpriteFx::id возвращает только 0..=4"),
            };
            pass.set_pipeline(pipeline);
            pass.draw_indexed(0..6, 0, range.start as u32..range.end as u32);
            self.stats.draw_calls += 1;
            let count = (range.end - range.start) as u32;
            match fx_id {
                0 => self.stats.sprites += count,
                _ => self.stats.fx_sprites += count,
            }
        }
        // pass падает здесь, encoder готов к finish()/дальнейшим командам.
    }

    /// Рисует батч 2D-мешей поверх уже нарисованной сцены (свои вершинный и
    /// индексный буферы внутри `MeshBatch`, свои пайплайны). Pass открывается
    /// с Load — вызывайте ПОСЛЕ [`Self::render`], чтобы не затирать сцену.
    ///
    /// Контракт: [`MeshBatch::upload`] вызывается вызывающим кодом ДО вызова
    /// этого метода (как у [`SpriteBatch`] с [`Self::render`]). При
    /// `debug.wireframe_meshes` рисует wireframe-пайплайном (если бэкенд
    /// поддерживает `POLYGON_MODE_LINE`); при `!debug.draw_meshes` — ранний
    /// выход. Счётчики `meshes`/`mesh_vertices`/`mesh_indices`/`draw_calls`
    /// накапливаются в [`RenderStats`].
    pub fn render_meshes(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        atlas: &SpriteAtlas,
        batch: &MeshBatch,
        camera: &Camera,
        blend: MeshBlend,
    ) {
        if batch.is_empty() || !self.debug.draw_meshes {
            return;
        }
        self.write_camera_uniform(queue, camera);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dioma-render mesh pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        let pipeline = if self.debug.wireframe_meshes {
            self.mesh_wireframe_pipeline
                .as_ref()
                .unwrap_or(&self.mesh_pipeline)
        } else {
            match blend {
                MeshBlend::Alpha => &self.mesh_pipeline,
                MeshBlend::Additive => &self.mesh_additive_pipeline,
            }
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        pass.set_bind_group(1, atlas.bind_group(), &[]);
        pass.set_vertex_buffer(0, batch.vertex_buffer().slice(..));
        pass.set_index_buffer(batch.index_buffer().slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..batch.drawn_index_count(), 0, 0..1);
        self.stats.draw_calls += 1;
        self.stats.meshes += batch.len() as u32;
        self.stats.mesh_vertices += batch.drawn_vertex_count();
        self.stats.mesh_indices += batch.drawn_index_count();
    }

    /// Аддитивный проход по источникам света (свой инстанс-буфер внутри
    /// `LightBatch`, свой пайплайн; атлас не нужен). Pass открывается с
    /// Load — вызывайте ПОСЛЕ [`Self::render`], чтобы не затирать сцену.
    /// При `!debug.draw_lights` — ранний выход; счётчики `lights`/`draw_calls`
    /// накапливаются в [`RenderStats`].
    pub fn render_lights(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        camera: &Camera,
        lights: &LightBatch,
    ) {
        if lights.is_empty() || !self.debug.draw_lights {
            return;
        }
        self.write_camera_uniform(queue, camera);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dioma-render light pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.light_pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        pass.set_vertex_buffer(0, lights.vertex_buffer().slice(..));
        pass.set_vertex_buffer(1, lights.instance_buffer().slice(..));
        pass.set_index_buffer(lights.index_buffer().slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..6, 0, 0..lights.count() as u32);
        self.stats.draw_calls += 1;
        self.stats.lights += lights.count() as u32;
    }

    /// Аддитивный проход по частицам (инстанс-буфер внутри `ParticleSystem`).
    /// Pass открывается с Load — вызывайте ПОСЛЕ [`Self::render`].
    /// При `!debug.draw_particles` — ранний выход; счётчики
    /// `particles`/`draw_calls` накапливаются в [`RenderStats`].
    pub fn render_particles(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        camera: &Camera,
        particles: &ParticleSystem,
    ) {
        if particles.is_empty() || !self.debug.draw_particles {
            return;
        }
        self.write_camera_uniform(queue, camera);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dioma-render particle pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.particle_pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        pass.set_vertex_buffer(0, particles.vertex_buffer().slice(..));
        pass.set_vertex_buffer(1, particles.instance_buffer().slice(..));
        pass.set_index_buffer(particles.index_buffer().slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..6, 0, 0..particles.count() as u32);
        self.stats.draw_calls += 1;
        self.stats.particles += particles.count() as u32;
    }

    /// Рисует текстовый оверлей (строки `lines`) поверх сцены шрифтом `font`.
    ///
    /// Строки печатаются с отступом 8 px от верхнего левого угла, высота
    /// строки 10 px, масштаб глифов 1.0 (ячейка 8×8 px), цвет белый.
    /// Передавайте «identity» камеру: [`Camera::new(viewport)`] (центр 0,
    /// зум 1) — текст позиционируется в экранных пикселях.
    ///
    /// Строки собираются в ОДИН меш и рисуются обычным (альфа) меш-пайплайном
    /// через внутренний батч рендерера (вместимость 8192 вершин / 16384
    /// индексов — следите, чтобы суммарный текст умещался).
    pub fn render_debug_overlay(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        font: &DebugFont,
        camera: &Camera,
        lines: &[String],
    ) {
        if lines.is_empty() {
            return;
        }
        const SCALE: f32 = 1.0;
        const LINE_HEIGHT: f32 = 10.0;
        const MARGIN: f32 = 8.0;
        let mut mesh = Mesh { vertices: Vec::new(), indices: Vec::new() };
        for (i, line) in lines.iter().enumerate() {
            let line_mesh = font.text_mesh(
                line,
                MARGIN,
                MARGIN + i as f32 * LINE_HEIGHT,
                SCALE,
                [1.0, 1.0, 1.0, 1.0],
                camera.viewport,
            );
            let base = mesh.vertices.len() as u32;
            mesh.vertices.extend_from_slice(&line_mesh.vertices);
            mesh.indices.extend(line_mesh.indices.iter().map(|i| i + base));
        }
        self.debug_mesh_batch.clear();
        self.debug_mesh_batch.push(&mesh);
        self.debug_mesh_batch.upload(queue);

        self.write_camera_uniform(queue, camera);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dioma-render debug overlay pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.mesh_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, font.atlas.bind_group(), &[]);
            pass.set_vertex_buffer(0, self.debug_mesh_batch.vertex_buffer().slice(..));
            pass.set_index_buffer(
                self.debug_mesh_batch.index_buffer().slice(..),
                wgpu::IndexFormat::Uint32,
            );
            pass.draw_indexed(0..self.debug_mesh_batch.drawn_index_count(), 0, 0..1);
        }
        self.stats.draw_calls += 1;
        self.debug_mesh_batch.clear();
    }

    /// Uniform камеры: `[cx, cy, zoom, time] + [vx, vy, 0, 0]` (32 байта).
    fn write_camera_uniform(&self, queue: &wgpu::Queue, camera: &Camera) {
        let u = camera.uniform_data();
        let packed: [f32; 8] = [u[0], u[1], u[2], self.time, u[3], u[4], 0.0, 0.0];
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&packed));
    }
}

/// Спрайтовый пайплайн: layout `[камера, атлас]`, вершины — [`SpriteBatch::vertex_layouts`].
fn build_sprite_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    shader_src: &str,
    label: &str,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    build_pipeline(
        device,
        layout,
        format,
        shader_src,
        label,
        &SpriteBatch::vertex_layouts(),
        blend,
        wgpu::PolygonMode::Fill,
    )
}

/// Общий конструктор пайплайна: шейдер из `shader_src`, свои вершинные слоты.
fn build_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    shader_src: &str,
    label: &str,
    vertex_layouts: &[wgpu::VertexBufferLayout<'_>],
    blend: Option<wgpu::BlendState>,
    polygon_mode: wgpu::PolygonMode,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: vertex_layouts,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    })
}
