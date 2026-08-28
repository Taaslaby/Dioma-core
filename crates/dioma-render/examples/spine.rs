//! Пример Spine 2D: скелетная анимация через наш мешевый пайплайн.
//!
//! Загружает тестовый скелет (assets/spine/strawberry: strawberry из Spine
//! 3.8) и крутит анимацию `animation` по кругу. `--frames=N` — смоук.

use std::path::PathBuf;
use std::sync::Arc;

use dioma_render::{Camera, MeshBatch, MeshBlend, Renderer, SpriteAtlas};

const VIEWPORT: [f32; 2] = [1280.0, 720.0];
const ASSET_DIR: &str = "assets/spine/strawberry";

fn main() {
    let smoke = std::env::args()
        .find_map(|a| a.strip_prefix("--frames=").and_then(|s| s.parse::<u32>().ok()));
    if let Some(n) = smoke {
        println!("смоук-режим: отрисую {n} кадров и выйду");
    }
    let events = winit::event_loop::EventLoop::new().unwrap();
    events.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = SpineDemo { smoke, ..Default::default() };
    events.run_app(&mut app).unwrap();
}

#[derive(Default)]
struct SpineDemo {
    smoke: Option<u32>,
    instance: wgpu::Instance,
    window: Option<Arc<winit::window::Window>>,
    surface: Option<wgpu::Surface<'static>>,
    device: Option<Arc<wgpu::Device>>,
    queue: Option<wgpu::Queue>,
    config: Option<wgpu::SurfaceConfiguration>,
    renderer: Option<Renderer>,
    atlas: Option<SpriteAtlas>,
    mesh_batch: Option<MeshBatch>,
    spine: Option<dioma_render::spine::SpineInstance>,
    last_frame: Option<std::time::Instant>,
}

impl SpineDemo {
    fn frame(&mut self) {
        let dt = self
            .last_frame
            .replace(std::time::Instant::now())
            .map(|t| t.elapsed().as_secs_f32())
            .unwrap_or(1.0 / 60.0);
        let Some(device) = &self.device else { return };
        let Some(queue) = &self.queue else { return };
        let Some(surface) = &self.surface else { return };
        let Some(config) = &self.config else { return };
        let Some(atlas) = &self.atlas else { return };
        let Some(renderer) = &mut self.renderer else { return };
        let Some(batch) = &mut self.mesh_batch else { return };
        let Some(spine) = &mut self.spine else { return };

        spine.update(dt);
        batch.clear();
        batch.push(&spine.build_mesh());
        batch.upload(queue);

        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                surface.configure(device, config);
                return;
            }
            Err(e) => {
                eprintln!("[spine] surface: {e}");
                return;
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("spine") });
        renderer.begin_frame(dt);
        let camera = Camera {
            center: [0.0, 0.0],
            zoom: 1.2, // модель ~440 мировых px → влезает в 720
            viewport: VIEWPORT,
        };
        renderer.render_meshes(
            &mut encoder,
            queue,
            &view,
            atlas,
            batch,
            &camera,
            MeshBlend::Alpha,
        );
        queue.submit(Some(encoder.finish()));
        frame.present();
    }
}

impl winit::application::ApplicationHandler for SpineDemo {
    fn resumed(&mut self, loop_: &winit::event_loop::ActiveEventLoop) {
        let window = Arc::new(
            loop_
                .create_window(
                    winit::window::Window::default_attributes()
                        .with_title("dioma-render — Spine 2D")
                        .with_inner_size(winit::dpi::LogicalSize::new(
                            VIEWPORT[0],
                            VIEWPORT[1],
                        )),
                )
                .unwrap(),
        );
        let size = window.inner_size();
        let surface = self.instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(
            self.instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        )
        .expect("нет GPU");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("spine demo"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .unwrap();
        let device = Arc::new(device);

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // Spine: atlas + skeleton из ассетов.
        let dir = PathBuf::from(ASSET_DIR);
        let atlas_text =
            std::fs::read_to_string(dir.join("skeleton.atlas")).expect("skeleton.atlas");
        let json = std::fs::read_to_string(dir.join("skeleton.json")).expect("skeleton.json");
        let png = std::fs::read(dir.join("skeleton.png")).expect("skeleton.png");
        let data = dioma_render::spine::SpineData::load(&device, &queue, &atlas_text, &png, &json)
            .expect("Spine загружается");
        let spine = data.instance("animation");
        println!("Spine: костей {}, слотов {}, длительность {:.2}с",
            data.skeleton.bones.len(), data.skeleton.slots.len(), spine.duration());

        let mut renderer = Renderer::new(&device, format);
        renderer.set_clear_color(wgpu::Color { r: 0.05, g: 0.07, b: 0.10, a: 1.0 });

        self.window = Some(window);
        self.surface = Some(surface);
        self.device = Some(device.clone());
        self.queue = Some(queue);
        self.config = Some(config);
        self.renderer = Some(renderer);
        self.atlas = Some(data.atlas);
        self.mesh_batch = Some(MeshBatch::new(&device, 4096, 8192));
        self.spine = Some(spine);
        self.last_frame = None;
    }

    fn window_event(
        &mut self,
        loop_: &winit::event_loop::ActiveEventLoop,
        _id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::CloseRequested => loop_.exit(),
            winit::event::WindowEvent::Resized(size) => {
                if let (Some(surface), Some(config)) = (&self.surface, &mut self.config) {
                    config.width = size.width.max(1);
                    config.height = size.height.max(1);
                    surface.configure(self.device.as_ref().unwrap(), config);
                }
            }
            winit::event::WindowEvent::RedrawRequested => {
                self.frame();
                if let Some(n) = &mut self.smoke {
                    *n = n.saturating_sub(1);
                    if *n == 0 {
                        println!("смоук: все кадры отрисованы без паники, выход");
                        loop_.exit();
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _loop_: &winit::event_loop::ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}
