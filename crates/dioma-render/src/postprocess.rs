//! Пост-обработка: полноэкранные проходы (виньетка + зерно, размытие,
//! тьма/ночь).
//!
//! Мир рендерится НЕ в поверхность, а в промежуточную текстуру
//! ([`PostProcess::scene_view`]); затем [`PostProcess::apply`] копирует её
//! в целевую поверхность одним полноэкранным квадом с эффектом.
//!
//! Порядок кадра при ночи ([`PostFx::Darkness`]): спрайты/меши/частицы →
//! `scene_view` → [`PostProcess::apply`] → и уже ПОСЛЕ него аддитивные
//! источники света ([`crate::Renderer::render_lights`]) на итоговую цель:
//! свет рисуется поверх тьмы и «пробивает» её собой.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Пост-эффект для [`PostProcess::apply`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PostFx {
    /// Без эффекта — пассивное копирование сцены.
    None,
    /// Виньетка (затемнение к углам) + анимированное плёночное зерно.
    VignetteGrain {
        /// Сила виньетки 0..1 (0 — выключено).
        vignette: f32,
        /// Сила зерна (амплитуда шума в линейном пространстве).
        grain: f32,
    },
    /// 9-точечный box blur. `radius` — в пикселях; 0 — выключено.
    Blur { radius: f32 },
    /// Тьма/ночь: умножение RGB сцены на цвет + сила эффекта.
    ///
    /// * `tint` — множитель цвета (например, ночь `[0.30, 0.34, 0.55]`,
    ///   закат `[0.9, 0.7, 0.55]`);
    /// * `strength` — смешение исходного и затемнённого цвета, 0..1
    ///   (клампится: 0 — день, 1 — полная тьма).
    ///
    /// Аддитивные источники света рисуйте ПОСЛЕ этого эффекта — см. доки
    /// модуля.
    Darkness { tint: [f32; 3], strength: f32 },
}

/// Uniform пост-проходов: `vec4(u_time, vignette, grain, blur_radius)`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct PostUniform {
    data: [f32; 4],
}

// f32-массив без паддингов: Pod/Zeroable вручную (без фичи derive у bytemuck).
unsafe impl Pod for PostUniform {}
unsafe impl Zeroable for PostUniform {}

/// Полноэкранная вершина (NDC-позиция; UV считается в шейдере).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct FullscreenVertex {
    pos: [f32; 2],
}

unsafe impl Pod for FullscreenVertex {}
unsafe impl Zeroable for FullscreenVertex {}

/// Полноэкранный пост-процессор.
///
/// Владеет промежуточной текстурой (формат = формат поверхности),
/// полноэкранным квадом (2 треугольника, 6 вершин в NDC), сценовым bind group
/// и двумя пайплайнами (`post/vignette_grain.wgsl`, `post/blur.wgsl`).
pub struct PostProcess {
    bind_group_layout: wgpu::BindGroupLayout,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    pipeline_vignette: wgpu::RenderPipeline,
    pipeline_blur: wgpu::RenderPipeline,
    /// Пайплайн тьмы/ночи (`post/darkness.wgsl`) — uniform vec4(tint, strength).
    pipeline_darkness: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
}

impl PostProcess {
    /// Создаёт пост-процессор под целевой `format` поверхности.
    ///
    /// Текстура сцены создаётся размера 1×1 — перед первым рендером вызовите
    /// [`Self::resize`] с реальным размером окна.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dioma-render post bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dioma-render post uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("dioma-render post sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Полноэкранный квад: два треугольника (6 вершин) в NDC.
        let vertices = [
            FullscreenVertex { pos: [-1.0, -1.0] },
            FullscreenVertex { pos: [1.0, -1.0] },
            FullscreenVertex { pos: [-1.0, 1.0] },
            FullscreenVertex { pos: [-1.0, 1.0] },
            FullscreenVertex { pos: [1.0, -1.0] },
            FullscreenVertex { pos: [1.0, 1.0] },
        ];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dioma-render post quad vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let pipeline_vignette = Self::build_pipeline(
            device,
            &layout,
            format,
            include_str!("shaders/post/vignette_grain.wgsl"),
            "dioma-render vignette_grain pipeline",
        );
        let pipeline_blur = Self::build_pipeline(
            device,
            &layout,
            format,
            include_str!("shaders/post/blur.wgsl"),
            "dioma-render blur pipeline",
        );
        let pipeline_darkness = Self::build_pipeline(
            device,
            &layout,
            format,
            include_str!("shaders/post/darkness.wgsl"),
            "dioma-render darkness pipeline",
        );

        // Текстура сцены 1×1; реальный размер — через resize().
        let (texture, view, bind_group) =
            Self::create_scene(&device, &layout, &uniform_buffer, &sampler, format, 1, 1);

        Self {
            bind_group_layout: layout,
            texture,
            view,
            sampler,
            bind_group,
            uniform_buffer,
            pipeline_vignette,
            pipeline_blur,
            pipeline_darkness,
            vertex_buffer,
            format,
            width: 1,
            height: 1,
        }
    }

    /// Пересоздаёт текстуру сцены под новый размер окна. Вызывайте при Resized.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }
        let (texture, view, bind_group) = Self::create_scene(
            device,
            &self.bind_group_layout,
            &self.uniform_buffer,
            &self.sampler,
            self.format,
            width.max(1),
            height.max(1),
        );
        self.texture = texture;
        self.view = view;
        self.bind_group = bind_group;
        self.width = width.max(1);
        self.height = height.max(1);
    }

    /// Вид текстуры сцены: рендерите мир ИМЕННО сюда, затем [`Self::apply`].
    pub fn scene_view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Полноэкранный пост-проход из текстуры сцены в `target_view`.
    ///
    /// Пишет uniform (time + параметры эффекта), ставит нужный пайплайн и
    /// рисует 6 вершин квада. С `PostFx::None` работает пайплайн виньетки
    /// с нулевой силой — он вырождается в пассивное копирование сцены
    /// (а `Blur{radius: 0.0}` — то же самое через blur-пайплайн).
    pub fn apply(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        fx: PostFx,
        time: f32,
    ) {
        // darkness.wgsl читает uniform иначе: vec4(tint.rgb, strength).
        let uniform_data: [f32; 4] = match fx {
            PostFx::None => [time, 0.0, 0.0, 0.0],
            PostFx::VignetteGrain { vignette, grain } => [time, vignette, grain, 0.0],
            PostFx::Blur { radius } => [time, 0.0, 0.0, radius],
            PostFx::Darkness { tint, strength } => {
                [tint[0], tint[1], tint[2], strength.clamp(0.0, 1.0)]
            }
        };
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&PostUniform { data: uniform_data }),
        );

        let pipeline = match fx {
            PostFx::Blur { .. } => &self.pipeline_blur,
            PostFx::Darkness { .. } => &self.pipeline_darkness,
            _ => &self.pipeline_vignette,
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dioma-render post pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Квад покрывает всю поверхность — заливка не видна.
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..6, 0..1);
    }

    /// Создаёт текстуру сцены + её bind group.
    #[allow(clippy::too_many_arguments)]
    fn create_scene(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform_buffer: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::BindGroup) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dioma-render post scene texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dioma-render post scene bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        (texture, view, bind_group)
    }

    fn build_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        format: wgpu::TextureFormat,
        shader_src: &str,
        label: &str,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[layout],
            push_constant_ranges: &[],
        });
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<FullscreenVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            }],
        };
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[vertex_layout],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
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
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        })
    }
}
