//! Игровой бинарь «dioma» — мост «сим ↔ рендер» в действии (этап 1).
//!
//! Паттерн окна — как демо dioma-render (winit 0.30 ApplicationHandler,
//! wgpu 22), смоук-режим `--frames=N` (отрисовать N кадров и выйти).
//!
//! Цикл кадра: накопление времени → фиксированный тик [`TICK`] (50 мс, как
//! в C3 `App::GetWorldTickInterval`) с множителем скорости →
//! [`SimView::capture`] → [`Presenter::build`] → заливка батча → рендер →
//! оверлей (FPS, тик, пауза, скорость, время суток, выделенное).
//!
//! День/ночь и сезоны: оверлей показывает фазу суток/сезон (транслитом),
//! а мир тонируется полупрозрачным спрайтом по фазе (тёплый рассвет,
//! синяя ночь). Сезоны визуально не меняют мир — только текст (документировано).
//!
//! Управление: Space — пауза, 1/2/3 — скорость ×1/×2/×4, Esc — снять
//! выделение, Shift+клик — спавн существа, B — спавн ягоды у курсора.
//!
//! Запуск: `cargo run -p dioma --bin game` или смоук `-- --frames=150`.

use std::sync::Arc;
use std::time::Instant;

use dioma::env::map::WorldMap;
use dioma::env::{Substance, SUBSTANCES};
use dioma::sim::Arena;
use dioma::view::{
    GameCommand, InputMapper, Presenter, Season, Sex, SimView, TICK, TimeOfDay, LAYER_TINT,
};
use dioma_render::{Camera, DebugFont, Renderer, Sprite, SpriteAtlas, SpriteBatch, SpriteFx};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowId};

// ─── Константы сцены (этап 1) ──────────────────────────────────────────────

/// Зум камеры: тело существа (~1.2 мировых ед.) рендерится ~22 px.
const ZOOM: f32 = 18.0;
/// Центр камеры — середина двух комнат.
const CAMERA_CENTER: [f32; 2] = [40.0, 7.0];
/// Вместимость общего батча: 2 комнаты × 560 тайлов + существа/ягоды.
const BATCH_CAPACITY: usize = 4096;
/// Сид мира.
const WORLD_SEED: u64 = 0xC0FFEE;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --frames=N: смоук-режим — прогнать N кадров и выйти самому.
    let smoke_frames = std::env::args()
        .find_map(|a| a.strip_prefix("--frames=").and_then(|s| s.parse::<u32>().ok()));
    if let Some(n) = smoke_frames {
        println!("смоук-режим: отрисую {n} кадров и выйду");
    }
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = Game::new(smoke_frames);
    event_loop.run_app(&mut app)?;
    Ok(())
}

/// Стартовая арена: две комнаты (40×14 и 40×14), дверь на общей стене
/// (y ≈ 6..8 — визуально двери на этапе 1 нет, это этап 2), 3 существа
/// с разными сидами и 6 ягод.
fn default_arena() -> Arena {
    let mut map = WorldMap::default();
    let a = map.add_room(0.0, 0.0, 40.0, 14.0);
    let b = map.add_room(40.0, 0.0, 80.0, 14.0);
    // Дверь: полная проницаемость веществ (тепло/влага/свет/питание текут).
    map.connect(a, b, [1.0; SUBSTANCES]);
    // Плодородность: тепло+свет+питание ~200 (потолок) — трава растёт.
    for room in &mut map.rooms {
        room.ca[Substance::Temperature.idx()] = 200.0;
        room.ca[Substance::Light.idx()] = 200.0;
        room.ca[Substance::Nutrients.idx()] = 200.0;
        room.ca[Substance::Humidity.idx()] = 150.0;
    }

    let mut arena = Arena::new(WORLD_SEED, map);
    arena.spawn(dioma::env::Vec2::new(8.0, 7.0), 0.0, 1);
    arena.spawn(dioma::env::Vec2::new(22.0, 7.0), 1.0, 2);
    arena.spawn(dioma::env::Vec2::new(60.0, 7.0), 2.0, 3);
    for (x, y) in [(5.0, 5.0), (18.0, 10.0), (30.0, 4.0), (48.0, 5.0), (62.0, 10.0), (72.0, 4.0)] {
        arena.add_berry(dioma::env::Vec2::new(x, y));
    }
    arena
}

/// Игровое состояние: wgpu-окружение (Option — создаётся в `resumed`) +
/// симуляция + ввод/камера.
struct Game {
    instance: wgpu::Instance,
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    device: Option<Arc<wgpu::Device>>,
    queue: Option<wgpu::Queue>,
    config: Option<wgpu::SurfaceConfiguration>,
    renderer: Option<Renderer>,
    atlas: Option<SpriteAtlas>,
    presenter: Option<Presenter>,
    batch: Option<SpriteBatch>,
    font: Option<DebugFont>,
    // Симуляция.
    arena: Arena,
    // Состояние кадра.
    camera: Camera,
    input: InputMapper,
    selected: Option<dioma::env::AgentId>,
    paused: bool,
    /// Множитель тиков на кадр (1/2/4).
    speed: u32,
    accumulator: f32,
    /// Счётчик спавнов (сид новых существ — детерминированный).
    spawn_seed: u64,
    /// Зажат ли Shift (для Shift+клик = спавн).
    shift: bool,
    started: Option<Instant>,
    last_frame: Option<Instant>,
    smoke_frames: Option<u32>,
    cursor: [f32; 2],
}

impl Game {
    fn new(smoke_frames: Option<u32>) -> Self {
        let arena = default_arena();
        Self {
            instance: wgpu::Instance::new(wgpu::InstanceDescriptor::default()),
            window: None,
            surface: None,
            device: None,
            queue: None,
            config: None,
            renderer: None,
            atlas: None,
            presenter: None,
            batch: None,
            font: None,
            arena,
            camera: Camera::new([1.0, 1.0]),
            input: InputMapper::new(),
            selected: None,
            paused: false,
            speed: 1,
            accumulator: 0.0,
            spawn_seed: 100,
            shift: false,
            started: None,
            last_frame: None,
            smoke_frames,
            cursor: [0.0, 0.0],
        }
    }

    /// Применяет команду ввода к состоянию игры.
    fn apply(&mut self, cmd: GameCommand) {
        match cmd {
            GameCommand::TogglePause => self.paused = !self.paused,
            GameCommand::Speed(n) => self.speed = n,
            GameCommand::Select(id) => self.selected = id,
            GameCommand::SpawnCritter(pos) => {
                self.spawn_seed += 1;
                self.arena.spawn(self.clamp_into_room(pos), 0.0, self.spawn_seed);
            }
            GameCommand::SpawnBerry(pos) => {
                self.arena.add_berry(self.clamp_into_room(pos));
            }
        }
    }

    /// Точка вне комнат (клик по стене/пустоте) — ближайший центр комнаты.
    fn clamp_into_room(&self, pos: dioma::env::Vec2) -> dioma::env::Vec2 {
        if self.arena.world.map.room_at(pos).is_some() {
            return pos;
        }
        let mut best = pos;
        let mut bd = f32::MAX;
        for r in &self.arena.world.map.rooms {
            let c = r.rect.center();
            let d = c.dist(pos);
            if d < bd {
                bd = d;
                best = c;
            }
        }
        best
    }

    // ── Кадр ────────────────────────────────────────────────────────────────

    /// Один кадр: тики сима → снимок → презентация → батч → рендер → оверлей.
    fn frame(&mut self) {
        let now = Instant::now();
        let dt = self.last_frame.map(|t| (now - t).as_secs_f32()).unwrap_or(1.0 / 60.0);
        self.last_frame = Some(now);
        let t = self.started.map(|s| s.elapsed().as_secs_f32()).unwrap_or(0.0);

        // 1) Накопление времени; фиксированный тик (пауза останавливает сим).
        self.accumulator += dt;
        while self.accumulator >= TICK {
            if !self.paused {
                for _ in 0..self.speed {
                    self.arena.step();
                }
            }
            self.accumulator -= TICK;
        }

        // 2) Проекция сима → данные для рисования (чистая функция).
        let view = SimView::capture(&self.arena);
        let Some(presenter) = &self.presenter else { return };
        let mut scene = presenter.build(&view);

        // Выделение: контур вокруг тела выбранного существа.
        if let Some(sel) = self.selected {
            if let Some(c) = view.critters.iter().find(|c| c.id == sel) {
                for s in &mut scene.bodies {
                    if s.position == [c.pos.x, c.pos.y] {
                        s.fx = SpriteFx::Outline { width: 2.0, color: [0.4, 1.0, 0.6, 1.0] };
                        break;
                    }
                }
            }
        }

        // Тон суток: полупрозрачный спрайт на весь вьюпорт (тёплый рассвет,
        // синяя ночь). Позиция/размер — от камеры (экран = центр ± вьюпорт/зум),
        // слой выше ягод, под оверлеем. Сезоны визуально не меняют мир (транш 3).
        {
            let cam = &self.camera;
            let size = [
                cam.viewport[0] / cam.zoom,
                cam.viewport[1] / cam.zoom,
            ];
            let mut tint = Sprite::new(cam.center, size, presenter.tint_uv())
                .with_tint(phase_tint(view.time_of_day));
            tint.layer = LAYER_TINT; // поверх ягод, под оверлеем
            scene.tint.push(tint);
        }

        // Строки оверлея собираем до мутабельного заимствования рендерера.
        let lines = self.overlay_lines(&view);

        let Some(device) = &self.device else { return };
        let Some(queue) = &self.queue else { return };
        let Some(surface) = &self.surface else { return };
        let Some(config) = &self.config else { return };
        let Some(atlas) = &self.atlas else { return };
        let Some(renderer) = &mut self.renderer else { return };
        let Some(batch) = &mut self.batch else { return };
        let Some(font) = &self.font else { return };

        // 3) Слои в один батч (внутри батча — сортировка по layer/depth).
        batch.clear();
        for s in &scene.tiles {
            batch.push(*s);
        }
        for s in &scene.shadows {
            batch.push(*s);
        }
        for s in &scene.bodies {
            batch.push(*s);
        }
        for s in &scene.faces {
            batch.push(*s);
        }
        for s in &scene.berries {
            batch.push(*s);
        }
        for s in &scene.tint {
            batch.push(*s);
        }
        batch.upload(queue);

        // 4) Рендер: единственный проход с очисткой — вся сцена одним батчем.
        renderer.begin_frame(dt);
        renderer.set_time(t);
        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                surface.configure(device, config);
                return;
            }
            Err(e) => {
                eprintln!("[game] surface error: {e}");
                return;
            }
        };
        let view_tex = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("game encoder"),
        });
        renderer.render(queue, &mut encoder, &view_tex, atlas, batch, &self.camera);

        // 5) Оверлей (DebugFont — ASCII, поэтому строки транслитом).
        renderer.render_debug_overlay(
            &mut encoder,
            queue,
            &view_tex,
            font,
            &Camera::new(self.camera.viewport),
            &lines,
        );

        queue.submit(Some(encoder.finish()));
        frame.present();
    }

    /// Строки оверлея: FPS, тик, пауза, скорость, выделенное существо.
    ///
    /// DebugFont умеет только ASCII 32..=126 (кириллица → '?'), поэтому
    /// строки — транслит (см. доки dioma-render/debug.rs).
    fn overlay_lines(&self, view: &SimView) -> Vec<String> {
        let s = self.renderer.as_ref().map(|r| *r.stats()).unwrap_or_default();
        let mut lines = vec![format!(
            "FPS {:.1} | tick {} | pauza {} | skorost x{}",
            s.fps,
            self.arena.tick,
            onoff(self.paused),
            self.speed,
        )];
        lines.push(format!(
            "sushestv {} | yagod {} | komnat {}",
            view.critters.len(),
            view.berries.len(),
            view.rooms.len(),
        ));
        // Время суток и сезон: «Utro | Den 2 Vesny». Транслит (DebugFont —
        // ASCII); у сезона пока нет визуала, только текст (документировано).
        lines.push(format!(
            "{} | Den {} {}",
            tod_latin(view.time_of_day),
            view.day_in_season + 1,
            season_genitive(view.season),
        ));
        if let Some(sel) = self.selected {
            if let Some(c) = view.critters.iter().find(|c| c.id == sel) {
                lines.push(format!(
                    "vydeleno #{}: golod {} glukoza {} ustalost {}",
                    sel.0,
                    bar(c.hunger),
                    bar(c.glucose),
                    bar(c.fatigue),
                ));
                lines.push(format!(
                    "  syedeno {} | skorost {:.2} | ziv {} | pol {}",
                    c.food_eaten,
                    c.speed,
                    onoff(c.alive),
                    sex_glyph(c.sex),
                ));
                // Стадия B: зрение существа (левый/правый глаз-бин).
                lines.push(format!(
                    "  vizhu edu: {} (L {:.2} R {:.2})",
                    if c.vis_l > 0.0 || c.vis_r > 0.0 { "da" } else { "net" },
                    c.vis_l,
                    c.vis_r,
                ));
            } else {
                lines.push(format!("vydeleno #{}: net v snimke", sel.0));
            }
        }
        lines.push("Space pauza | 1/2/3 skorost | Esc snyat | Shift+klik spawn | B yagoda".to_string());
        lines
    }
}

/// Полоса состояния как текст: `[#####---]`.
fn bar(v: f32) -> String {
    const N: usize = 10;
    let filled = (v.clamp(0.0, 1.0) * N as f32).round() as usize;
    let mut s = String::with_capacity(N + 2);
    s.push('[');
    for i in 0..N {
        s.push(if i < filled { '#' } else { '-' });
    }
    s.push(']');
    s
}

/// «вкл»/«выкл» для строк оверлея.
fn onoff(b: bool) -> &'static str {
    if b { "vkl" } else { "vykl" }
}

/// Фаза суток транслитом (DebugFont — ASCII, кириллица → «?»).
fn tod_latin(t: TimeOfDay) -> &'static str {
    match t {
        TimeOfDay::Dawn => "Rassvet",
        TimeOfDay::Morning => "Utro",
        TimeOfDay::Afternoon => "Den",
        TimeOfDay::Evening => "Vecher",
        TimeOfDay::Night => "Noch",
    }
}

/// Сезон в родительном падеже: «Den 2 Vesny» / «Den 1 Zimy».
fn season_genitive(s: Season) -> &'static str {
    match s {
        Season::Spring => "Vesny",
        Season::Summer => "Leta",
        Season::Autumn => "Oseni",
        Season::Winter => "Zimy",
    }
}

/// Пол существа: «M»/«F» (DebugFont — ASCII, без ♂/♀).
fn sex_glyph(sex: Sex) -> &'static str {
    match sex {
        Sex::Male => "M",
        Sex::Female => "F",
    }
}

/// Цвет и прозрачность тона суток: тёплый рассвет, синяя ночь; днём тон
/// почти прозрачен. Альфа — «сила» затемнения мира (0.02..0.35).
fn phase_tint(t: TimeOfDay) -> [f32; 4] {
    match t {
        TimeOfDay::Dawn => [1.0, 0.55, 0.30, 0.08], // тёплый оранжевый
        TimeOfDay::Morning => [1.0, 0.80, 0.55, 0.04], // чуть тёплый
        TimeOfDay::Afternoon => [1.0, 1.0, 0.90, 0.02], // нейтральный
        TimeOfDay::Evening => [1.0, 0.45, 0.30, 0.12], // закатный
        TimeOfDay::Night => [0.08, 0.10, 0.32, 0.35], // синяя ночь
    }
}

// ── winit ApplicationHandler ────────────────────────────────────────────────

impl ApplicationHandler for Game {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("dioma")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0)),
                )
                .unwrap(),
        );
        let size = window.inner_size();

        let surface = self.instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(
            self.instance
                .request_adapter(&wgpu::RequestAdapterOptions::default()),
        )
        .expect("нет GPU-адаптера");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("dioma game device"),
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

        // Атлас: тело, тень, 4 лица, 4 фазы ягоды, тайлсет пола/стен.
        let mut atlas = SpriteAtlas::new(&device, 256);
        let presenter = Presenter::new(&queue, &mut atlas);

        let mut renderer = Renderer::new(&device, format);
        renderer.set_clear_color(wgpu::Color { r: 0.05, g: 0.07, b: 0.10, a: 1.0 });
        let font = DebugFont::new(&device, &queue);
        let batch = SpriteBatch::new(&device, BATCH_CAPACITY);

        let mut camera = Camera::new([size.width as f32, size.height as f32]);
        camera.center = CAMERA_CENTER;
        camera.zoom = ZOOM;

        self.window = Some(window);
        self.surface = Some(surface);
        self.device = Some(device);
        self.queue = Some(queue);
        self.config = Some(config);
        self.renderer = Some(renderer);
        self.atlas = Some(atlas);
        self.presenter = Some(presenter);
        self.batch = Some(batch);
        self.font = Some(font);
        self.camera = camera;
        self.started = Some(Instant::now());
        self.last_frame = None;
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let (Some(surface), Some(config)) = (&self.surface, &mut self.config) {
                    config.width = size.width.max(1);
                    config.height = size.height.max(1);
                    surface.configure(self.device.as_ref().unwrap(), config);
                }
                self.camera.viewport = [size.width as f32, size.height as f32];
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = [position.x as f32, position.y as f32];
                self.input.on_cursor(self.cursor, &self.camera);
            }
            WindowEvent::ModifiersChanged(m) => {
                self.shift = m.state().shift_key();
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if let Some(cmd) = self.input.on_click(self.shift, &self.arena) {
                    self.apply(cmd);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        if let Some(cmd) = self.input.on_key(code) {
                            self.apply(cmd);
                        }
                    }
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

    /// Непрерывная анимация: каждый тик цикла просим перерисовку.
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}
