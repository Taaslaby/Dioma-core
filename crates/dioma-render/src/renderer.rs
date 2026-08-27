//! Рендерер: пайплайны спрайтов (базовый + fx), мешей, света и частиц
//! + uniform камеры + параллакс-слои фона (см. [`Renderer::render_parallax`])
//! + профилировщик и debug-оверлей.

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

/// Размер кольца униформ-буферов камеры для параллакс-проходов.
///
/// `queue.write_buffer` применяется ко ВСЕМ pass'ам одного submit ДО их
/// исполнения (последняя запись в буфер побеждает), поэтому у каждого
/// прохода со СВОЕЙ камерой должен быть СВОЙ буфер. Кольцо из 4 слотов
/// покрывает типичные 2-3 параллакс-слоя; больше 4 проходов
/// [`Renderer::render_parallax`] с разными камерами за один кадр —
/// не поддерживается (слоты начнут перекрываться).
const PARALLAX_RING: usize = 4;

/// Какую пару (униформ-буфер камеры, bind group) использует спрайтовый проход.
///
/// У каждого прохода со СВОЕЙ камерой — свой буфер: `queue.write_buffer`
/// применяется ко всем pass'ам одного submit ДО их исполнения (последняя
/// запись в буфер побеждает). `Main` — мировой проход (render/meshes/light/
/// particles делят один буфер, камера у них одна), `Parallax(i)` — i-й слот
/// параллакс-кольца. У debug-оверлея (mesh-проход) тоже свой буфер — см.
/// `overlay_uniform_buffer`.
#[derive(Clone, Copy)]
enum CameraSlot {
    Main,
    Parallax(usize),
}

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
    /// Кольцо отдельных униформ-буферов + bind group'ов для параллакс-проходов
    /// (см. [`PARALLAX_RING`] — почему не один буфер).
    parallax_uniforms: Vec<wgpu::Buffer>,
    parallax_bind_groups: Vec<wgpu::BindGroup>,
    /// Индекс следующего свободного слота кольца (растёт без сброса — >4
    /// параллакс-проходов за кадр перекрываются).
    parallax_ring_index: usize,
    /// Свой униформ-буфер камеры у debug-оверлея: он рисует identity-камерой,
    /// и его запись не должна перетирать мировую камеру основного прохода.
    overlay_uniform_buffer: wgpu::Buffer,
    overlay_camera_bind_group: wgpu::BindGroup,
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

        // Параллакс-кольцо и отдельный буфер оверлея: каждый проход со своей
        // камерой — свой буфер (см. PARALLAX_RING).
        let make_camera_pair = |label: &str| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &camera_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
            (buffer, bind_group)
        };
        let (overlay_uniform_buffer, overlay_camera_bind_group) =
            make_camera_pair("dioma-render overlay camera uniform");
        let mut parallax_uniforms = Vec::with_capacity(PARALLAX_RING);
        let mut parallax_bind_groups = Vec::with_capacity(PARALLAX_RING);
        for i in 0..PARALLAX_RING {
            let (buffer, bg) = make_camera_pair(&format!("dioma-render parallax {i} camera uniform"));
            parallax_uniforms.push(buffer);
            parallax_bind_groups.push(bg);
        }

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
            parallax_uniforms,
            parallax_bind_groups,
            parallax_ring_index: 0,
            overlay_uniform_buffer,
            overlay_camera_bind_group,
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
    ///
    /// Это ЕДИНСТВЕННЫЙ проход, который очищает цель (`LoadOp::Clear`).
    /// Поверх уже нарисованного рисуйте [`Self::render_meshes`],
    /// [`Self::render_lights`], [`Self::render_particles`],
    /// [`Self::render_debug_overlay`] и параллакс-слои [`Self::render_parallax`]
    /// (они открывают pass с `Load`).
    pub fn render(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        atlas: &SpriteAtlas,
        batch: &SpriteBatch,
        camera: &Camera,
    ) {
        self.draw_sprite_pass(queue, encoder, view, atlas, batch, camera, CameraSlot::Main, true, true);
    }

    /// Рисует батч спрайтов со сдвинутой камерой — параллакс-слой фона.
    ///
    /// Камера слоя: центр = `camera.center · factor`, зум и вьюпорт — без
    /// изменений. `factor` — глубина слоя относительно мира:
    ///
    /// * `0.0` — фиксированный слой (HUD/фон, приколоченный к экрану);
    /// * `0.3..0.7` — дальний фон (двигается медленнее камеры);
    /// * `1.0` — обычный мир (центр не сдвигается).
    ///
    /// Значение вне `0.0..=1.0` (включая `NaN`) — ошибка с русским
    /// сообщением; ничего не рисуется.
    ///
    /// Pass открывается с `LoadOp::Load` и НЕ очищает цель, поэтому слои
    /// рисуются поверх уже очищенного кадра (в демо — ПОСЛЕ [`Self::render`],
    /// в свободной от сцены области; painter's algorithm без depth buffer).
    /// Используется только базовый спрайтовый пайплайн: спрайты с
    /// [`SpriteFx`] в `batch` пропускаются. Debug-тоггл
    /// [`RenderDebug::draw_sprites`] уважается. Draw calls и инстансы
    /// накапливаются в [`RenderStats`] как обычные спрайты (`sprites`,
    /// `draw_calls`) — тот же внутренний проход, что у [`Self::render`].
    ///
    /// Каждый вызов берёт свой слот кольца униформ-буферов (см.
    /// [`PARALLAX_RING`]) — не более 4 параллакс-проходов с разными камерами
    /// за один кадр.
    pub fn render_parallax(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        atlas: &SpriteAtlas,
        batch: &SpriteBatch,
        camera: &Camera,
        factor: f32,
    ) -> Result<(), String> {
        if !(0.0..=1.0).contains(&factor) {
            return Err(format!(
                "render_parallax: factor {factor} вне диапазона 0.0..=1.0 (клампить намеренно не стали — ошибка приходит сразу)"
            ));
        }
        let shifted = Camera {
            center: [camera.center[0] * factor, camera.center[1] * factor],
            ..*camera
        };
        let slot = self.parallax_ring_index % PARALLAX_RING;
        self.parallax_ring_index += 1;
        self.draw_sprite_pass(
            queue,
            encoder,
            view,
            atlas,
            batch,
            &shifted,
            CameraSlot::Parallax(slot),
            false,
            false,
        );
        Ok(())
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
        self.write_camera_uniform(queue, &self.uniform_buffer, camera);
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
        self.write_camera_uniform(queue, &self.uniform_buffer, camera);
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
        self.write_camera_uniform(queue, &self.uniform_buffer, camera);
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
    /// зум 1) — текст позиционируется в экранных пикселях. Оверлей пишет
    /// камеру в СВОЙ униформ-буфер (см. [`PARALLAX_RING`]), поэтому его
    /// identity-камера не перетирает камеру мировых проходов.
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

        self.write_camera_uniform(queue, &self.overlay_uniform_buffer, camera);
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
            pass.set_bind_group(0, &self.overlay_camera_bind_group, &[]);
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

    /// Пара (униформ-буфер камеры, bind group) для слота [`CameraSlot`].
    fn camera_pair(&self, slot: CameraSlot) -> (&wgpu::Buffer, &wgpu::BindGroup) {
        match slot {
            CameraSlot::Main => (&self.uniform_buffer, &self.camera_bind_group),
            CameraSlot::Parallax(i) => {
                (&self.parallax_uniforms[i], &self.parallax_bind_groups[i])
            }
        }
    }

    /// Uniform камеры: `[cx, cy, zoom, time] + [vx, vy, 0, 0]` (32 байта).
    ///
    /// Пишет в `buffer` — буфер камеры ИМЕННО этого прохода (см.
    /// [`PARALLAX_RING`]: write_buffer применяется до исполнения всех pass'ов
    /// submit'а, последняя запись в буфер побеждает).
    fn write_camera_uniform(
        &self,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        camera: &Camera,
    ) {
        let u = camera.uniform_data();
        let packed: [f32; 8] = [u[0], u[1], u[2], self.time, u[3], u[4], 0.0, 0.0];
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&packed));
    }

    /// Внутренний спрайтовый проход — общий для [`Self::render`] и
    /// [`Self::render_parallax`], чтобы статистика считалась одинаково.
    ///
    /// `camera_slot` — чью пару (униформ-буфер, bind group) использовать:
    /// у каждого прохода со своей камерой свой буфер (см. [`CameraSlot`] и
    /// [`PARALLAX_RING`]). `clear` — очищать ли цель (`true` — первый проход
    /// кадра, `false` — поверх уже нарисованного, `LoadOp::Load`).
    /// `allow_fx` — включать ли fx-пайплайны (`false` для параллакса: только
    /// базовый шейдер, fx-группы пропускаются).
    fn draw_sprite_pass(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        atlas: &SpriteAtlas,
        batch: &SpriteBatch,
        camera: &Camera,
        camera_slot: CameraSlot,
        clear: bool,
        allow_fx: bool,
    ) {
        let (uniform_buffer, camera_bind_group) = self.camera_pair(camera_slot);
        self.write_camera_uniform(queue, uniform_buffer, camera);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dioma-render sprite pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if clear {
                        wgpu::LoadOp::Clear(self.clear_color)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_bind_group(0, camera_bind_group, &[]);
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
            // Параллакс: только базовый пайплайн, fx-группы пропускаем.
            if !allow_fx && fx_id != 0 {
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
