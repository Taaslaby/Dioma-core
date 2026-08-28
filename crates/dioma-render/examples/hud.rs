//! Пример UI-каркаса: сцена dioma-render + собственный immediate-mode UI
//! поверх неё (без egui): панель с метками, кнопкой паузы и прогресс-баром.
//!
//! Демонстрирует полный цикл кадра UI:
//!   `UiInput` из winit-событий → `UiSystem::begin` → виджеты в замыкании
//!   панели → `upload` → `Renderer::render_ui` поверх сцены.
//!
//! `--frames=N` — смоук-режим: отрисовать N кадров и выйти (для CI/проверки).

use std::sync::Arc;
use std::time::Instant;

use dioma_render::ui::{Anchor, Button, UiInput, UiSystem, UiTheme};
use dioma_render::{
    atlas, Camera, Light, LightBatch, Renderer, Sprite, SpriteAtlas, SpriteBatch, SpriteFx,
};

const VIEWPORT: [f32; 2] = [1280.0, 720.0];

fn main() {
    let smoke_frames = std::env::args()
        .find_map(|a| a.strip_prefix("--frames=").and_then(|s| s.parse::<u32>().ok()));
    if let Some(n) = smoke_frames {
        println!("смоук-режим: отрисую {n} кадров и выйду");
    }

    let events = winit::event_loop::EventLoop::new().unwrap();
    events.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = HudDemo::new(smoke_frames);
    events.run_app(&mut app).unwrap();
}

/// Настройки, редактируемые из UI (mutable-состояние «игры»).
#[derive(Default)]
struct HudState {
    paused: bool,
    clicks: u32,
    load: f32,
    light_theme: bool,
    sound: bool,
    volume: f32,
    action: Option<&'static str>,
}

struct HudDemo {
    smoke_frames: Option<u32>,
    instance: wgpu::Instance,
    window: Option<Arc<winit::window::Window>>,
    surface: Option<wgpu::Surface<'static>>,
    device: Option<Arc<wgpu::Device>>,
    queue: Option<wgpu::Queue>,
    config: Option<wgpu::SurfaceConfiguration>,
    renderer: Option<Renderer>,
    atlas: Option<SpriteAtlas>,
    batch: Option<SpriteBatch>,
    lights: Option<LightBatch>,
    ui: Option<UiSystem>,
    state: HudState,
    cursor: [f32; 2],
    mouse_down: bool,
    mouse_down_last_frame: bool,
    key: Option<dioma_render::ui::UiKey>,
    started: Instant,
    last_frame: Option<Instant>,
}

impl HudDemo {
    fn new(smoke_frames: Option<u32>) -> Self {
        Self {
            smoke_frames,
            instance: wgpu::Instance::new(&wgpu::InstanceDescriptor::default()),
            window: None,
            surface: None,
            device: None,
            queue: None,
            config: None,
            renderer: None,
            atlas: None,
            batch: None,
            lights: None,
            ui: None,
            state: HudState {
        load: 0.42,
        light_theme: false,
        sound: true,
        volume: 0.6,
        action: None,
        ..Default::default()
    },
            cursor: [0.0; 2],
            mouse_down: false,
            mouse_down_last_frame: false,
            key: None,
            started: Instant::now(),
            last_frame: None,
        }
    }

    fn frame(&mut self) {
        let now = Instant::now();
        let dt = self.last_frame.map(|t| (now - t).as_secs_f32()).unwrap_or(1.0 / 60.0);
        self.last_frame = Some(now);
        let t = self.started.elapsed().as_secs_f32();

        let Some(device) = &self.device else { return };
        let Some(queue) = &self.queue else { return };
        let Some(surface) = &self.surface else { return };
        let Some(config) = &self.config else { return };
        let Some(atlas) = self.atlas.as_mut() else { return };
        let Some(renderer) = self.renderer.as_mut() else { return };
        let Some(batch) = &mut self.batch else { return };
        let Some(lights) = &mut self.lights else { return };
        let Some(ui) = &mut self.ui else { return };

        // ── Сцена: два «существа» на орбитах (пауза останавливает время). ──
        let sim_t = if self.state.paused { 0.0 } else { t };
        renderer.begin_frame(dt);
        renderer.set_time(t);

        let body_uv = atlas.region("body").expect("регион body создаётся в resumed");
        batch.clear();
        // Градиентное небо (новая фича: with_gradient) — прямоугольник на весь
        // «мир» позади существ.
        {
            let sky_uv = [0.0, 0.0, 1.0, 1.0]; // единый квад без атласной области: используем ui/flat
            let flat_uv = atlas.region("ui/rect").expect("регион ui/rect создаётся в UiSystem::new");
            let _ = sky_uv;
            let mut sky = Sprite::new([320.0, 100.0], [640.0, 200.0], flat_uv)
                .with_gradient([0.35, 0.55, 0.95, 0.9], [0.10, 0.16, 0.40, 0.9]);
            sky.layer = -1.0;
            batch.push(sky);
        }
        for i in 0..2u32 {
            let ang = sim_t * 0.8 + i as f32 * std::f32::consts::PI;
            let pos = [320.0 + ang.cos() * 120.0, 80.0 + ang.sin() * 50.0];
            let mut s = Sprite::new(pos, [26.0, 26.0], body_uv);
            // Drop-тень: новый fx — силуэт, смещённый вниз и размытый.
            s.fx = SpriteFx::Shadow { offset: [0.0, -4.0], radius: 3.0, color: [0.0, 0.0, 0.0, 0.5] };
            batch.push(s);
        }
        batch.upload(queue);

        lights.clear();
        lights.push(Light {
            position: [320.0, 80.0],
            radius: 240.0,
            intensity: 0.5,
            color: [1.0, 0.72, 0.42],
        });
        lights.upload(queue);

        // ── UI: панель HUD в левом верхнем углу. ──
        let input = UiInput {
            cursor: self.cursor,
            down: self.mouse_down,
            just_pressed: self.mouse_down && !self.mouse_down_last_frame,
            just_released: !self.mouse_down && self.mouse_down_last_frame,
            key: self.key,
        };
        self.mouse_down_last_frame = self.mouse_down;
        self.key = None; // клавиша «на один кадр»

        {
            let mut frame = ui.begin(atlas, queue, input, VIEWPORT);
            let state = &mut self.state;
            state.load = 0.5 + 0.45 * (t * 0.7).sin(); // бар плавно двигается
            frame.panel(Anchor::TopLeft, [300.0, 190.0], |p| {
                p.label("Пример HUD: dioma-render UI");
                p.label(if state.paused { "состояние: пауза" } else { "состояние: работает" });
                if p.button(if state.paused { "▶ Пуск" } else { "⏸ Пауза" }).clicked {
                    state.paused = !state.paused;
                    state.clicks += 1;
                }
                p.progress_bar("Загрузка", state.load);
            });
            frame.upload(queue);
        } // frame отдан: дальше атлас нужен только на чтение

        // ── Кадр: сцена → свет → UI → present. ──
        let fg = match surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                surface.configure(device, config);
                return;
            }
            Err(e) => {
                eprintln!("[hud] surface error: {e}");
                return;
            }
        };
        let view = fg.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("hud") });
        let camera = Camera::new(VIEWPORT);
        renderer.render(queue, &mut encoder, &view, &*atlas, batch, &camera);
        renderer.render_lights(queue, &mut encoder, &view, &camera, lights);
        renderer.render_ui(queue, &mut encoder, &view, &*atlas, ui.batch(), VIEWPORT);
        queue.submit(Some(encoder.finish()));
        fg.present();
    }
}

impl winit::application::ApplicationHandler for HudDemo {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    winit::window::Window::default_attributes()
                        .with_title("dioma-render — UI каркас")
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
        .expect("нет GPU-адаптера");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("hud device"),
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

        // Атлас: процедурное «тело» для сцены + регионы UI + глифы TTF
        // (UiSystem::new растеризует шрифт — создать ДО других кэшей UV).
        let mut atlas = SpriteAtlas::new(&device, 1024);
        let (bw, bh, body_px) = atlas::soft_circle_rgba(14, [120, 220, 160, 255]);
        atlas.add_rgba(&queue, "body", bw, bh, &body_px).unwrap();
        let mut renderer = Renderer::new(&device, format);
        renderer.set_clear_color(wgpu::Color { r: 0.05, g: 0.07, b: 0.10, a: 1.0 });

        self.ui = Some(UiSystem::new(&device, &queue, &mut atlas));

        self.window = Some(window);
        self.surface = Some(surface);
        self.device = Some(device.clone());
        self.queue = Some(queue);
        self.config = Some(config);
        self.renderer = Some(renderer);
        self.atlas = Some(atlas);
        self.batch = Some(SpriteBatch::new(&device, 256));
        self.lights = Some(LightBatch::new(&device, 8));
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        use winit::event::{ElementState, MouseButton, WindowEvent};
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let (Some(surface), Some(config)) = (&self.surface, &mut self.config) {
                    config.width = size.width.max(1);
                    config.height = size.height.max(1);
                    surface.configure(self.device.as_ref().unwrap(), config);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = [position.x as f32, position.y as f32];
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.mouse_down = state == ElementState::Pressed;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::PhysicalKey;
                if event.state == ElementState::Pressed {
                    self.key = match event.physical_key {
                        PhysicalKey::Code(winit::keyboard::KeyCode::ArrowUp) => {
                            Some(dioma_render::ui::UiKey::Up)
                        }
                        PhysicalKey::Code(winit::keyboard::KeyCode::ArrowDown) => {
                            Some(dioma_render::ui::UiKey::Down)
                        }
                        PhysicalKey::Code(winit::keyboard::KeyCode::Enter) => {
                            Some(dioma_render::ui::UiKey::Enter)
                        }
                        _ => None,
                    };
                } else {
                    self.key = None;
                }
            }
            WindowEvent::RedrawRequested => {
                self.frame();
                if let Some(n) = &mut self.smoke_frames {
                    *n = n.saturating_sub(1);
                    if *n == 0 {
                        println!("смоук: все кадры отрисованы без паники, выход");
                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}
