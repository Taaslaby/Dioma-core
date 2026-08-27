//! Пример UI-оверлея: сцена dioma-render + egui поверх неё в той же поверхности.
//!
//! Показывает, как красиво оверлеить игру: HUD-панель со статистикой (FPS,
//! draw calls), меню-пауза по Esc с настройками (зум, сила света, скорость
//! времени, тьма/ночь) и справкой по клавишам. Русский текст — шрифт egui
//! (Ubuntu-Light) поддерживает кириллицу из коробки.
//!
//! egui подключён только в dev-dependencies: библиотека dioma-render остаётся
//! чистой (никакого egui в крейт-ядре). Порядок проходов кадра:
//!
//! ```text
//! render (спрайты) → render_meshes → render_lights → particles → post.apply
//! → egui (Load поверх поверхности) → present
//! ```
//!
//! Управление:
//!   Esc — меню-пауза;  клик по кнопкам — действия;
//!   X / закрытие окна — выход.
//!   `--frames=N` — смоук-режим: отрисовать N кадров и выйти (для CI/проверки).

use std::sync::Arc;
use std::time::Instant;

use dioma_render::{
    atlas, Camera, Light, LightBatch, Particle, ParticleSystem, PostFx, PostProcess, Renderer,
    Sprite, SpriteAtlas, SpriteBatch, TileBatch, Tilemap, TilesetUv,
};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let smoke_frames = std::env::args()
        .find_map(|a| a.strip_prefix("--frames=").and_then(|s| s.parse::<u32>().ok()));
    if let Some(n) = smoke_frames {
        println!("смоук-режим: отрисую {n} кадров и выйду");
    }
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = UiDemo::new(smoke_frames);
    event_loop.run_app(&mut app)?;
    Ok(())
}

/// Настройки, редактируемые из egui-замыкания (слайдеры, пауза, выход).
/// Вынесены в отдельную структуру, чтобы замыкание `run` захватывало только
/// их + зум камеры, а не весь [`UiDemo`] (иначе конфликт заимствований).
struct UiState {
    /// Множитель времени (меняется слайдером; пауза = 0).
    time_scale: f32,
    /// Сила света (меняется слайдером, пересоздаётся каждый кадр).
    light_intensity: f32,
    /// Сила тьмы (слайдер; 0 — день, 1 — ночь).
    night: f32,
    menu_open: bool,
    settings_open: bool,
    exit_requested: bool,
}

/// Простое состояние UI-демо: окно, wgpu, сцена dioma-render и egui-оверлей.
struct UiDemo {
    instance: wgpu::Instance,
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
    device: Option<Arc<wgpu::Device>>,
    queue: Option<wgpu::Queue>,
    format: wgpu::TextureFormat,
    camera: Camera,
    renderer: Option<Renderer>,
    atlas: Option<SpriteAtlas>,
    batch: Option<SpriteBatch>,
    lights: Option<LightBatch>,
    particles: Option<ParticleSystem>,
    post: Option<PostProcess>,
    /// egui-стек (0.29 — пара к wgpu 22 + winit 0.30).
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,
    last_frame: Option<Instant>,
    started: Instant,
    /// Настройки, редактируемые из egui (слайдеры, пауза, выход).
    ui: UiState,
    smoke_frames: Option<u32>,
}

impl UiDemo {
    fn new(smoke_frames: Option<u32>) -> Self {
        Self {
            instance: wgpu::Instance::new(wgpu::InstanceDescriptor::default()),
            window: None,
            surface: None,
            config: None,
            device: None,
            queue: None,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            camera: Camera::new([800.0, 600.0]),
            renderer: None,
            atlas: None,
            batch: None,
            lights: None,
            particles: None,
            post: None,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            egui_renderer: None,
            last_frame: None,
            started: Instant::now(),
            ui: UiState {
                time_scale: 1.0,
                light_intensity: 0.45,
                night: 0.0,
                menu_open: false,
                settings_open: false,
                exit_requested: false,
            },
            smoke_frames,
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
        let Some(atlas) = &self.atlas else { return };
        let Some(renderer) = &mut self.renderer else { return };
        let Some(batch) = &mut self.batch else { return };
        let Some(lights) = &mut self.lights else { return };
        let Some(particles) = &mut self.particles else { return };
        let Some(post) = &mut self.post else { return };
        let Some(window) = &self.window else { return };

        // ── Симуляция (пауза через меню замораживает время) ──
        let sim_dt = dt * if self.ui.menu_open { 0.0 } else { self.ui.time_scale };
        renderer.begin_frame(dt);
        renderer.set_time(t);
        particles.update(sim_dt);

        // Орбита существ + пыльца.
        batch.clear();
        let body_uv = atlas.region("body").unwrap();
        let ring_uv = atlas.region("ring").unwrap();
        for i in 0..2u32 {
            let ang = t * 0.8 + i as f32 * std::f32::consts::PI;
            let pos = [320.0 + ang.cos() * 90.0, 150.0 + ang.sin() * 40.0];
            let mut s = Sprite::new(pos, [26.0, 26.0], body_uv);
            s.rotation = t * 1.2 + i as f32;
            batch.push(s);
        }
        // Вращающееся кольцо вокруг орбит существ — живая деталь сцены.
        let mut ring = Sprite::new([320.0, 150.0], [180.0, 180.0], ring_uv);
        ring.rotation = t * 0.3;
        batch.push(ring);
        batch.upload(queue);

        // Свет: тёплый над сценой (интенсивность из слайдера).
        lights.clear();
        lights.push(Light {
            position: [320.0, 150.0],
            radius: 260.0,
            intensity: self.ui.light_intensity,
            color: [1.0, 0.72, 0.42],
        });
        lights.upload(queue);

        // Пыльца: редкие вспышки частиц (пауза их не спавнит).
        if !self.ui.menu_open {
            let p = Particle::new(
                [300.0 + (t * 13.0).sin() * 120.0, 80.0 + (t * 7.0).cos() * 30.0],
                [20.0, 30.0],
                2.5,
                6.0,
                [0.9, 0.9, 0.6, 0.9],
            );
            particles.spawn(p);
        }

        // ── Кадр поверхности ──
        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                surface.configure(device, config);
                return;
            }
            Err(e) => {
                eprintln!("[ui] surface error: {e}");
                return;
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ui encoder"),
        });

        // Мир → текстура сцены пост-процессора, затем пост → поверхность.
        renderer.render(queue, &mut encoder, post.scene_view(), atlas, batch, &self.camera);
        renderer.render_lights(queue, &mut encoder, post.scene_view(), &self.camera, lights);
        particles.render(queue, &mut encoder, post.scene_view(), &self.camera, renderer);
        // Тьма/ночь: до света её бы не пробили фонари, поэтому свет выше.
        let post_fx = if self.ui.night > 0.0 {
            PostFx::Darkness { tint: [0.30, 0.34, 0.55], strength: self.ui.night }
        } else {
            PostFx::VignetteGrain { vignette: 0.35, grain: 0.05 }
        };
        post.apply(queue, &mut encoder, &view, post_fx, t);

        // ── egui поверх поверхности (Load — сцена уже нарисована) ──
        let egui_state = self.egui_state.as_mut().expect("egui state создан в resumed");
        let egui_renderer = self.egui_renderer.as_mut().expect("egui renderer создан в resumed");

        let raw_input = egui_state.take_egui_input(window);
        let stats = *renderer.stats(); // RenderStats — Copy: снимок до замыкания
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            UiDemo::build_ui(ctx, &mut self.ui, &mut self.camera.zoom, &stats)
        });
        egui_state.handle_platform_output(window, full_output.platform_output);
        let clipped = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [config.width, config.height],
            pixels_per_point: window.scale_factor() as f32,
        };
        for (id, delta) in &full_output.textures_delta.set {
            egui_renderer.update_texture(device, queue, *id, delta);
        }
        egui_renderer.update_buffers(device, queue, &mut encoder, &clipped, &screen_descriptor);
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_scissor_rect(0, 0, config.width, config.height);
            // forget_lifetime: egui хочет RenderPass<'static>, у нас — с заимствованием
            // encoder'а; после drop pass'а encoder снова свободен.
            egui_renderer.render(&mut rpass.forget_lifetime(), &clipped, &screen_descriptor);
        }
        for id in &full_output.textures_delta.free {
            egui_renderer.free_texture(id);
        }

        queue.submit(Some(encoder.finish()));
        frame.present();
    }

/// Собирает egui-интерфейс: HUD, справка и модальное меню-пауза.
///
/// Статический метод без `self`: egui-замыкание в [`UiDemo::frame`] захватывает
/// только `UiState` и зум камеры — не весь `UiDemo`, иначе конфликт
/// заимствований с полями, занятыми на время кадра (device/renderer/...).
/// Параметр назван `state`, потому что внутри замыкания `.show(ui, |ui| ...)`
/// переменная `ui` — это `&mut egui::Ui`.
fn build_ui(ctx: &egui::Context, state: &mut UiState, camera_zoom: &mut f32, stats: &dioma_render::RenderStats) {
        // Тёмная тема с акцентом в цветах игры (teal-кольцо).
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgba_unmultiplied(18, 22, 34, 235);
        visuals.window_fill = egui::Color32::from_rgba_unmultiplied(14, 18, 28, 245);
        visuals.widgets.inactive.weak_bg_fill = egui::Color32::from_rgba_unmultiplied(30, 38, 58, 255);
        visuals.widgets.hovered.weak_bg_fill = egui::Color32::from_rgba_unmultiplied(40, 52, 80, 255);
        visuals.widgets.active.weak_bg_fill = egui::Color32::from_rgba_unmultiplied(26, 34, 52, 255);
        visuals.selection.bg_fill = egui::Color32::from_rgb(80, 200, 190);
        ctx.set_visuals(visuals);
        let mut style = (*ctx.style()).clone();
        style.spacing.item_spacing = egui::vec2(10.0, 8.0);
        style.visuals.widgets.noninteractive.rounding = egui::Rounding::same(8.0);
        style.visuals.widgets.inactive.rounding = egui::Rounding::same(6.0);
        ctx.set_style(style);

        // ── HUD: статистика кадра ──
        egui::Area::new(egui::Id::new("hud"))
            .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 10.0))
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.strong(format!("FPS {:.0} · {:.2} ms", stats.fps, stats.frame_ms));
                    ui.label(format!(
                        "draw calls {} · спрайтов {} · света {} · частиц {}",
                        stats.draw_calls, stats.sprites, stats.lights, stats.particles
                    ));
                });
            });

        // ── Справка по клавишам ──
        egui::Area::new(egui::Id::new("help"))
            .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-10.0, -10.0))
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.weak("Esc — меню · X — выход · V — смоук-режим");
                });
            });

        // ── Модальное меню-пауза ──
        if !state.menu_open {
            return;
        }
        egui::Area::new(egui::Id::new("menu-dim"))
            .order(egui::Order::Background)
            .show(ctx, |ui| {
                let rect = ui.max_rect();
                ui.painter().rect_filled(rect, 0.0, egui::Color32::from_black_alpha(160));
            });
        egui::Area::new(egui::Id::new("menu"))
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                egui::Frame::window(ui.style())
                    .inner_margin(egui::Margin::same(24.0))
                    .show(ui, |ui| {
                        ui.set_min_width(300.0);
                        ui.heading("Меню");
                        ui.add_space(8.0);
                        if ui.button("▶ Продолжить").clicked() {
                            state.menu_open = false;
                            state.settings_open = false;
                        }
                        if ui.button("⚙ Настройки").clicked() {
                            state.settings_open = !state.settings_open;
                        }
                        if state.settings_open {
                            ui.separator();
                            ui.label("Зум камеры");
                            ui.add(egui::Slider::new(camera_zoom, 0.5..=3.0).logarithmic(true));
                            ui.label("Сила света");
                            ui.add(egui::Slider::new(&mut state.light_intensity, 0.0..=1.5));
                            ui.label("Скорость времени");
                            ui.add(egui::Slider::new(&mut state.time_scale, 0.0..=3.0));
                            ui.label("Ночь");
                            ui.add(egui::Slider::new(&mut state.night, 0.0..=1.0));
                            ui.separator();
                        }
                        if ui.button("✕ Выход").clicked() {
                            state.exit_requested = true;
                        }
                    });
            });
    }
}

impl ApplicationHandler for UiDemo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("dioma-render + egui")
                        .with_inner_size(LogicalSize::new(800.0, 600.0)),
                )
                .expect("окно"),
        );
        let size = window.inner_size();

        let surface = self.instance.create_surface(window.clone()).expect("поверхность");
        let adapter = pollster::block_on(
            self.instance
                .request_adapter(&wgpu::RequestAdapterOptions::default()),
        )
        .expect("нет GPU-адаптера");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ui example device"),
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
        self.format = format;

        // ── Процедурный атлас (ноль ассетов, как во всём крейте) ──
        let mut atlas = SpriteAtlas::new(&device, 256);
        let (w, h, px) = atlas::circle_rgba(24, [255, 166, 77, 255]);
        atlas.add_rgba(&queue, "sun", w, h, &px).unwrap();
        let (w, h, px) = atlas::ring_rgba(56, 5.0, [80, 200, 190, 255]);
        atlas.add_rgba(&queue, "ring", w, h, &px).unwrap();
        let (w, h, px) = atlas::berry_rgba(0.0, [220, 70, 110, 255]);
        atlas.add_rgba(&queue, "berry", w, h, &px).unwrap();
        let (w, h, px) = atlas::circle_rgba(18, [255, 255, 255, 255]);
        atlas.add_rgba(&queue, "body", w, h, &px).unwrap();
        let (tw, th, tpx) = ui_tileset_rgba();
        atlas.add_rgba(&queue, "tileset", tw, th, &tpx).unwrap();

        // Тайловый пол: 12×2 тайла по 32 px (мировая Y 0..64).
        let rows: [&[u16]; 2] = [
            &[4, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 4],
            &[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        ];
        let room = Tilemap::from_rows(&rows, 32.0);
        let tileset = TilesetUv::new(atlas.region("tileset").unwrap(), 2, 2);
        let tile_batch = TileBatch::new(&device, &room, &tileset);
        let mut batch = SpriteBatch::new(&device, 128);
        for s in tile_batch.sprite_batch().sprites() {
            batch.push(*s);
        }
        let mut sun = Sprite::new([320.0, 200.0], [56.0, 56.0], atlas.region("sun").unwrap());
        sun.rotation = 0.0;
        batch.push(sun);
        batch.push(Sprite::new([140.0, 60.0], [40.0, 40.0], atlas.region("berry").unwrap()));
        batch.push(Sprite::new([500.0, 60.0], [40.0, 40.0], atlas.region("berry").unwrap()));

        let mut renderer = Renderer::new(&device, format);
        renderer.set_clear_color(wgpu::Color { r: 0.02, g: 0.03, b: 0.10, a: 1.0 });

        let lights = LightBatch::new(&device, 4);
        let particles = ParticleSystem::new(&device, 256);
        let mut post = PostProcess::new(&device, format);
        post.resize(&device, size.width, size.height);

        // ── egui-стек (0.29: wgpu 22 + winit 0.30) ──
        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);

        self.camera = Camera::new([size.width as f32, size.height as f32]);
        self.camera.center = [320.0, 80.0];
        self.window = Some(window);
        self.surface = Some(surface);
        self.config = Some(config);
        self.device = Some(device);
        self.queue = Some(queue);
        self.atlas = Some(atlas);
        self.batch = Some(batch);
        self.lights = Some(lights);
        self.particles = Some(particles);
        self.post = Some(post);
        self.renderer = Some(renderer);
        self.egui_state = Some(egui_state);
        self.egui_renderer = Some(egui_renderer);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Матчим ПО ССЫЛКЕ: иначе частичное перемещение `event` из паттерна
        // запретит передать его egui ниже.
        match &event {
            WindowEvent::CloseRequested => event_loop.exit(),
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
            WindowEvent::Resized(size) => {
                if let (Some(surface), Some(config)) = (&self.surface, &mut self.config) {
                    config.width = size.width.max(1);
                    config.height = size.height.max(1);
                    surface.configure(self.device.as_ref().unwrap(), config);
                }
                self.camera.viewport = [size.width as f32, size.height as f32];
                if let (Some(device), Some(post)) = (&self.device, &mut self.post) {
                    post.resize(device, size.width.max(1), size.height.max(1));
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        match code {
                            KeyCode::Escape => self.ui.menu_open = !self.ui.menu_open,
                            KeyCode::KeyX => event_loop.exit(),
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
        // Все события отдаём egui (наведение, клики, фокус).
        if let (Some(window), Some(state)) = (&self.window, &mut self.egui_state) {
            let _ = state.on_window_event(window, &event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.ui.exit_requested {
            event_loop.exit();
            return;
        }
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

}

/// Тайлсет 2×2 (32×32): трава, земля, вода, стена — процедурно, как в демо.
fn ui_tileset_rgba() -> (u32, u32, Vec<u8>) {
    const TILE: u32 = 16;
    let w = TILE * 2;
    let h = TILE * 2;
    let mut px = vec![0u8; (w * h * 4) as usize];
    let palette: [[u8; 4]; 4] = [
        [90, 150, 90, 255],   // 1 трава
        [150, 120, 70, 255],  // 2 земля
        [70, 110, 170, 255],  // 3 вода
        [110, 110, 120, 255], // 4 стена
    ];
    for ty in 0..2u32 {
        for tx in 0..2u32 {
            let idx = (ty * 2 + tx) as usize;
            let c = palette[idx];
            // Лёгкая «шумовая» модуляция по пикселям (детерминированная).
            for y in 0..TILE {
                for x in 0..TILE {
                    let shade = 0.85 + 0.15 * (((x * 7 + y * 13 + idx as u32 * 5) % 4) as f32 / 3.0);
                    let o = ((ty * TILE + y) * w + tx * TILE + x) as usize;
                    px[o * 4] = (c[0] as f32 * shade) as u8;
                    px[o * 4 + 1] = (c[1] as f32 * shade) as u8;
                    px[o * 4 + 2] = (c[2] as f32 * shade) as u8;
                    px[o * 4 + 3] = c[3];
                }
            }
        }
    }
    (w, h, px)
}
