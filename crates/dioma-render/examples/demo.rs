//! Демо dioma-render: существа на орбитах с per-sprite эффектами, радиальным
//! светом, частицами-пыльцой, пост-обработкой, флипбук-криттерами с Y-сортировкой,
//! тайловой комнатой (Tilemap/TileBatch — травяной пол, пруд, стены) и двумя
//! параллакс-слоями фона (солнце+кольцо 0.4, ягоды 0.7) и debug-оверлеем.
//! winit 0.30 + wgpu 22.
//!
//! Соглашение осей (сайд-вью): мировая Y растёт вверх, верх экрана = больший
//! Y мира. Сцена разложена под это: комната стоит в НИЖНЕЙ части кадра
//! (мировые Y ≈ 0..128 — «пол» комнаты), а небо с солнцем/кольцом/ягодами —
//! в верхней (Y ≈ 150..260). Криттеры и PNG-криттер шагают ПО ПОЛУ комнаты.
//!
//! Управление:
//!   клик — выбрать существо (контур Outline);
//!   стрелки — панорамирование камеры (виден параллакс: фон едет медленнее мира);
//!   D — растворить выбранное (Dissolve), через 2 c оно возродится;
//!   R — возродить всех;
//!   V — переключить пост-эффект (None → VignetteGrain → Blur);
//!   F1 — оверлей статистики (FPS, draw calls, счётчики проходов);
//!   F2 — каркас мешей (блоб);
//!   1..6 — тогглы проходов: спрайты / fx / меши / свет / частицы / пост;
//!   Esc / закрытие окна — выход.
//!
//! Запуск: `cargo run -p dioma-render --example demo`

use std::sync::Arc;
use std::time::Instant;

use dioma_render::{
    atlas, Camera, DebugFont, Flipbook, Light, LightBatch, Mesh, MeshBatch, MeshBlend, MeshVertex,
    Particle, ParticleSystem, PostFx, PostProcess, Renderer, Sprite, SpriteAtlas, SpriteBatch,
    SpriteFx, TileBatch, Tilemap, TilesetUv,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --frames=N: смоук-режим — прогнать N кадров и выйти самому
    // (для проверки, что демо живёт без ручного закрытия окна).
    let smoke_frames = std::env::args()
        .find_map(|a| a.strip_prefix("--frames=").and_then(|s| s.parse::<u32>().ok()));
    if let Some(n) = smoke_frames {
        println!("смоук-режим: отрисую {n} кадров и выйду");
    }
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = Demo::new(smoke_frames);
    event_loop.run_app(&mut app)?;
    Ok(())
}

/// Крошечный детерминированный LCG (xorshift-стиль) — без зависимости на rand.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next() as f32
    }
}

// ── Раскладка сцены (соглашение: мировая Y вверх = верх экрана) ────────────
//
// Комната: 20×4 тайла по 32 px → мировая область X 0..640, Y 0..128.
// Визуально (Y вверх) ряд данных 0 — это ПОЛ комнаты (низ экрана), последний
// ряд — потолок. Небо/параллакс живёт выше: Y ≈ 150..260.

/// Верхняя грань ряда-пола (world Y); криттеры стоят на ней ногами.
const FLOOR_Y: f32 = 32.0;

/// Центр панорамирования ходока-криттера по полу (комната по X 0..640).
const WALK_CENTER_X: f32 = 320.0;

/// Мировая точка, где на экране висит солнце параллакс-слоя при стартовой
/// камере (320, 80): солнце лежит в слое factor 0.4 на world (60, 190), что
/// с учётом сдвига камеры слоя даёт ту же экранную позицию, что и мир
/// (252, 238) в основном проходе. Существа кружат вокруг этой точки.
const ORBIT_CENTER: [f32; 2] = [252.0, 238.0];

/// Настроение существа — задаёт базовый per-sprite эффект.
#[derive(Clone, Copy, PartialEq)]
enum Mood {
    /// Счастливое: дофаминовое свечение + пульс размера.
    Happy,
    /// Больное: дрожь (Wobble).
    Sick,
    /// Спокойное: без базового эффекта (выделение — Outline).
    Calm,
}

/// Существо: орбита вокруг точки солнца на экране + состояние жизни.
struct Creature {
    orbit_angle: f32,
    orbit_speed: f32,
    orbit_radius: f32,
    size: f32,
    tint: [f32; 4],
    mood: Mood,
    selected: bool,
    dissolving: bool,
    dissolve_progress: f32,
    dead: bool,
    respawn_timer: f32,
}

impl Creature {
    fn pos(&self) -> [f32; 2] {
        [
            ORBIT_CENTER[0] + self.orbit_radius * self.orbit_angle.cos(),
            ORBIT_CENTER[1] + self.orbit_radius * self.orbit_angle.sin(),
        ]
    }

    /// Базовый эффект настроения (без учёта выделения/растворения).
    fn base_fx(&self) -> SpriteFx {
        match self.mood {
            Mood::Happy => {
                SpriteFx::Glow { radius: 9.0, intensity: 0.5, color: [1.0, 0.95, 0.4, 1.0] }
            }
            Mood::Sick => SpriteFx::Wobble { amplitude: 2.4, frequency: 0.9, speed: 6.0 },
            Mood::Calm => SpriteFx::None,
        }
    }

    /// Итоговый эффект кадра: растворение > выделение > базовый.
    fn frame_fx(&self) -> SpriteFx {
        if self.dissolving {
            SpriteFx::Dissolve {
                progress: self.dissolve_progress,
                edge_width: 0.06,
                edge_color: [1.0, 0.55, 0.15, 1.0],
            }
        } else if self.selected {
            SpriteFx::Outline { width: 3.0, color: [0.4, 1.0, 0.6, 1.0] }
        } else {
            self.base_fx()
        }
    }
}

/// Криттер: ходок «шагает» по полу комнаты (анимируется флипбуком) или стоит
/// на месте — с разной глубиной Y, чтобы была видна Y-сортировка спрайтов.
struct Critter {
    /// Позиция (для стоячих); у ходока пересчитывается из `angle`/`radius`.
    pos: [f32; 2],
    /// Текущий угол ходока (фаза шагания по полу).
    angle: f32,
    /// Амплитуда шагания ходока вокруг `WALK_CENTER_X`.
    radius: f32,
    /// Угловая скорость ходока.
    speed: f32,
    /// Время анимации флипбука (растёт каждый кадр).
    anim_time: f32,
    size: f32,
    tint: [f32; 4],
    /// Глубина Y-сортировки (у ходока = позиция по Y).
    depth: f32,
    walking: bool,
}

struct Demo {
    instance: wgpu::Instance,
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    device: Option<Arc<wgpu::Device>>,
    queue: Option<wgpu::Queue>,
    config: Option<wgpu::SurfaceConfiguration>,
    renderer: Option<Renderer>,
    atlas: Option<SpriteAtlas>,
    batch: Option<SpriteBatch>,
    mesh_batch: Option<MeshBatch>,
    lights: Option<LightBatch>,
    particles: Option<ParticleSystem>,
    post: Option<PostProcess>,
    font: Option<DebugFont>,
    flipbook: Option<Flipbook>,
    /// Тайловая комната: батч перестраивается при изменении карты (здесь —
    /// один раз), спрайты тайлов каждый кадр копируются в общий `batch`.
    tile_batch: Option<TileBatch>,
    /// Параллакс-слои фона: солнце+кольцо (factor 0.4) и ягоды (factor 0.7).
    parallax_far: Option<SpriteBatch>,
    parallax_mid: Option<SpriteBatch>,
    camera: Camera,
    /// Направление панорамирования стрелками (компоненты −1/0/1).
    pan_dir: [f32; 2],
    started: Option<Instant>,
    last_frame: Option<Instant>,
    /// Смоук-режим: после N кадров приложение само выходит (None = обычный режим).
    smoke_frames: Option<u32>,
    cursor: [f32; 2],
    creatures: Vec<Creature>,
    critters: Vec<Critter>,
    post_fx: PostFx,
    post_enabled: bool,
    pollen_timer: f32,
    rng: Rng,
}

impl Demo {
    fn new(smoke_frames: Option<u32>) -> Self {
        Self {
            instance: wgpu::Instance::new(wgpu::InstanceDescriptor::default()),
            window: None,
            surface: None,
            device: None,
            queue: None,
            config: None,
            renderer: None,
            atlas: None,
            batch: None,
            mesh_batch: None,
            lights: None,
            particles: None,
            post: None,
            font: None,
            flipbook: None,
            tile_batch: None,
            parallax_far: None,
            parallax_mid: None,
            camera: Camera::new([1.0, 1.0]),
            pan_dir: [0.0, 0.0],
            started: None,
            last_frame: None,
            smoke_frames,
            cursor: [0.0, 0.0],
            creatures: Vec::new(),
            critters: Vec::new(),
            post_fx: PostFx::None,
            post_enabled: true,
            pollen_timer: 0.0,
            rng: Rng(0x9E3779B97F4A7C15),
        }
    }

    // ── Кадр ────────────────────────────────────────────────────────────────

    /// Один кадр: обновления → мир → свет → частицы → (пост) → оверлей.
    fn frame(&mut self) {
        let now = Instant::now();
        let dt = self.last_frame.map(|t| (now - t).as_secs_f32()).unwrap_or(1.0 / 60.0);
        self.last_frame = Some(now);
        let t = self.started.map(|s| s.elapsed().as_secs_f32()).unwrap_or(0.0);

        self.update_creatures(dt);
        self.spawn_pollen(dt);

        let Some(device) = &self.device else { return };
        let Some(queue) = &self.queue else { return };
        let Some(surface) = &self.surface else { return };
        let Some(config) = &self.config else { return };
        let Some(atlas) = &self.atlas else { return };
        let Some(renderer) = &mut self.renderer else { return };
        let Some(batch) = &mut self.batch else { return };
        let Some(mesh_batch) = &mut self.mesh_batch else { return };
        let Some(lights) = &mut self.lights else { return };
        let Some(particles) = &mut self.particles else { return };
        let Some(post) = &mut self.post else { return };
        let Some(tile_batch) = &self.tile_batch else { return };
        let Some(parallax_far) = &mut self.parallax_far else { return };
        let Some(parallax_mid) = &mut self.parallax_mid else { return };

        renderer.begin_frame(dt);
        renderer.set_time(t);
        particles.update(dt);

        // Панорамирование стрелками; кламп держит и пол комнаты (Y=0), и небо
        // параллакса (до Y≈260) в кадре: центр X ∈ [100, 540] (комната 0..640
        // почти целиком), центр Y ∈ [−60, 220] — при желании поменяйте границы.
        let speed = 300.0 * dt;
        self.camera.pan([self.pan_dir[0] * speed, self.pan_dir[1] * speed]);
        self.camera.center[0] = self.camera.center[0].clamp(100.0, 540.0);
        self.camera.center[1] = self.camera.center[1].clamp(-60.0, 220.0);

        // Криттеры: у ходока крутится угол, у всех растёт время анимации.
        for c in &mut self.critters {
            c.anim_time += dt;
            if c.walking {
                c.angle += c.speed * dt;
            }
        }

        // ── Мир: тайлы комнаты (слой LAYER_TILE = 0.0) + существа поверх ──
        batch.clear();
        for s in tile_batch.sprite_batch().sprites() {
            batch.push(*s); // Sprite — Copy; тайлы перестраиваются только при rebuild
        }
        build_world(batch, atlas, &self.creatures);
        if let Some(fb) = &self.flipbook {
            build_critters(batch, atlas, &mut self.critters, fb);
        }
        build_png_critter(batch, atlas, t);
        batch.upload(queue);

        // ── Параллакс-фон: солнце+кольцо (дальний слой 0.4) и ягоды (0.7) ──
        // Мировые Y 150..260 — НЕБО НАД комнатой (Y вверх = верх экрана).
        parallax_far.clear();
        let mut sun = Sprite::new([60.0, 190.0], [56.0, 56.0], atlas.region("sun").unwrap());
        sun.rotation = t * 0.3;
        parallax_far.push(sun);
        let mut ring = Sprite::new([60.0, 190.0], [150.0, 150.0], atlas.region("ring").unwrap());
        ring.rotation = t * 0.12;
        parallax_far.push(ring);
        parallax_far.upload(queue);

        parallax_mid.clear();
        let berry_pos = [[-40.0, 215.0], [200.0, 250.0], [420.0, 175.0]];
        for (i, name) in ["berry-a", "berry-b", "berry-c"].iter().enumerate() {
            let bob = (t * 0.8 + i as f32).sin() * 6.0; // лёгкое покачивание
            let mut berry = Sprite::new(
                [berry_pos[i][0], berry_pos[i][1] + bob],
                [46.0, 46.0],
                atlas.region(name).unwrap(),
            );
            berry.rotation = t * 0.6 + i as f32;
            parallax_mid.push(berry);
        }
        parallax_mid.upload(queue);

        // ── «Блоб»: деформируемый 2D-меш (CPU-деформация каждый кадр) ──
        // Стоит справа от комнаты, на уровне её пола.
        mesh_batch.clear();
        mesh_batch.push(&blob_mesh([660.0, 60.0], 48.0, t, atlas.region("sun").unwrap()));
        mesh_batch.upload(queue);

        // ── Радиальные источники света ──
        lights.clear();
        // Тёплый — над солнцем и существами (центр камеры (320, 80)).
        lights.push(Light {
            position: [252.0, 238.0],
            radius: 240.0,
            intensity: 0.30,
            color: [1.0, 0.72, 0.42],
        });
        // Холодный — над комнатой (правая половина пола и стены).
        lights.push(Light {
            position: [480.0, 80.0],
            radius: 320.0,
            intensity: 0.22,
            color: [0.35, 0.55, 1.0],
        });
        lights.upload(queue);

        // ── Кадр поверхности ──
        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                surface.configure(device, config);
                return;
            }
            Err(e) => {
                eprintln!("[demo] surface error: {e}");
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("demo encoder"),
        });

        // Мир → текстура сцены (пост включён) или сразу поверхность (пост выкл).
        let scene_target = if self.post_enabled { post.scene_view() } else { &view };
        renderer.render(queue, &mut encoder, scene_target, atlas, batch, &self.camera);
        // Параллакс-слои: поверх очищенного кадра (Load), камера со сдвинутым
        // центром. Рисуются ПОСЛЕ render(): render() — единственный проход с
        // очисткой, поэтому слой позади сцены читался бы только в свободной
        // от комнаты области экрана; солнце/кольцо/ягоды стоят в мировых
        // Y ≈ 150..260 — на экране это небо НАД комнатой (Y вверх).
        renderer
            .render_parallax(&mut encoder, queue, scene_target, atlas, parallax_far, &self.camera, 0.4)
            .expect("factor 0.4 в диапазоне 0.0..=1.0");
        renderer
            .render_parallax(&mut encoder, queue, scene_target, atlas, parallax_mid, &self.camera, 0.7)
            .expect("factor 0.7 в диапазоне 0.0..=1.0");
        renderer.render_meshes(
            &mut encoder,
            queue,
            scene_target,
            atlas,
            mesh_batch,
            &self.camera,
            MeshBlend::Alpha,
        );
        renderer.render_lights(queue, &mut encoder, scene_target, &self.camera, lights);
        particles.render(queue, &mut encoder, scene_target, &self.camera, renderer);
        if self.post_enabled {
            post.apply(queue, &mut encoder, &view, self.post_fx, t);
        }

        // Оверлей статистики — поверх поверхности, после пост-прохода.
        if renderer.debug.show_overlay {
            if let Some(font) = &self.font {
                let s = *renderer.stats();
                let d = renderer.debug;
                let post_on = self.post_enabled;
                let mut lines = vec![
                    format!("FPS {:.1}  кадр {:.2} ms", s.fps, s.frame_ms),
                    format!("draw calls {}", s.draw_calls),
                    format!("спрайты {} (fx {})", s.sprites, s.fx_sprites),
                    format!("меши {} (верш {} / инд {})", s.meshes, s.mesh_vertices, s.mesh_indices),
                    format!("свет {}  частицы {}", s.lights, s.particles),
                    format!(
                        "проходы: спр{} fx{} меш{} св{} част{} пост{}",
                        onoff(d.draw_sprites),
                        onoff(d.draw_fx),
                        onoff(d.draw_meshes),
                        onoff(d.draw_lights),
                        onoff(d.draw_particles),
                        onoff(post_on),
                    ),
                ];
                // Строка о PNG-ассете (добавляется только при включённом оверлее).
                if let Some(info) = atlas.region_info("png-critter") {
                    lines.push(format!(
                        "PNG: загружен + обрезан (offset {}, {})",
                        info.offset[0] as u32,
                        info.offset[1] as u32,
                    ));
                }
                lines.push(format!("Tilemap: {} тайлов | Parallax: 2 слоя", tile_batch.len()));
                renderer.render_debug_overlay(
                    &mut encoder,
                    queue,
                    &view,
                    font,
                    &Camera::new(self.camera.viewport),
                    &lines,
                );
            }
        }

        queue.submit(Some(encoder.finish()));
        frame.present();
    }

    // ── Симуляция ───────────────────────────────────────────────────────────

    fn update_creatures(&mut self, dt: f32) {
        for c in &mut self.creatures {
            if c.dead {
                c.respawn_timer -= dt;
                if c.respawn_timer <= 0.0 {
                    c.dead = false;
                    c.dissolving = false;
                    c.dissolve_progress = 0.0;
                }
                continue;
            }
            c.orbit_angle += c.orbit_speed * dt;
            if c.dissolving {
                c.dissolve_progress += dt * 0.6;
                if c.dissolve_progress >= 1.0 {
                    c.dead = true;
                    c.respawn_timer = 2.0;
                }
            }
        }
    }

    /// Пыльца: каждые 0.04 c появляются две лёгкие частицы (до 200 живых).
    fn spawn_pollen(&mut self, dt: f32) {
        self.pollen_timer -= dt;
        if self.pollen_timer > 0.0 {
            return;
        }
        self.pollen_timer = 0.04;
        let Some(particles) = &mut self.particles else { return };
        if particles.len() >= 200 {
            return;
        }
        for _ in 0..2 {
            // Пыльца дрейфует над всей сценой: от пола комнаты до неба.
            let pos = [self.rng.range(80.0, 560.0), self.rng.range(40.0, 300.0)];
            let vel = [self.rng.range(-18.0, 18.0), self.rng.range(-9.0, 9.0)];
            let shade = self.rng.range(0.75, 1.0);
            let mut p = Particle::new(
                pos,
                vel,
                self.rng.range(4.0, 8.0),
                self.rng.range(3.0, 7.0),
                [shade, shade, self.rng.range(0.65, 0.95), 0.9],
            );
            p.drag = 0.35;
            p.gravity = 0.0;
            particles.spawn(p);
        }
    }

    /// Начинает растворение выбранного живого существа.
    fn start_dissolve(&mut self) {
        for c in &mut self.creatures {
            if c.selected && !c.dead && !c.dissolving {
                c.dissolving = true;
                c.dissolve_progress = 0.0;
            }
        }
    }

    fn respawn_all(&mut self) {
        for c in &mut self.creatures {
            c.dead = false;
            c.dissolving = false;
            c.dissolve_progress = 0.0;
        }
    }

    /// Цикл пост-эффектов: None → VignetteGrain → Blur → None.
    fn cycle_post_fx(&mut self) {
        self.post_fx = match self.post_fx {
            PostFx::None => PostFx::VignetteGrain { vignette: 0.35, grain: 0.06 },
            PostFx::VignetteGrain { .. } => PostFx::Blur { radius: 2.0 },
            PostFx::Blur { .. } => PostFx::None,
        };
        self.update_title();
    }

    /// F1: оверлей статистики.
    fn toggle_overlay(&mut self) {
        if let Some(r) = &mut self.renderer {
            r.debug.show_overlay = !r.debug.show_overlay;
        }
        self.update_title();
    }

    /// F2: каркас мешей (влияет на блоб).
    fn toggle_wireframe(&mut self) {
        if let Some(r) = &mut self.renderer {
            r.debug.wireframe_meshes = !r.debug.wireframe_meshes;
        }
        self.update_title();
    }

    /// 1..6: тогглы проходов — спрайты / fx / меши / свет / частицы / пост.
    fn toggle_pass(&mut self, idx: u8) {
        let Some(r) = &mut self.renderer else { return };
        match idx {
            0 => r.debug.draw_sprites = !r.debug.draw_sprites,
            1 => r.debug.draw_fx = !r.debug.draw_fx,
            2 => r.debug.draw_meshes = !r.debug.draw_meshes,
            3 => r.debug.draw_lights = !r.debug.draw_lights,
            4 => r.debug.draw_particles = !r.debug.draw_particles,
            5 => self.post_enabled = !self.post_enabled,
            _ => {}
        }
        self.update_title();
    }

    /// Заголовок окна: текущий пост-эффект + состояния тогглов.
    fn update_title(&self) {
        if let Some(window) = &self.window {
            let d = self.renderer.as_ref().map(|r| r.debug).unwrap_or_default();
            window.set_title(&format!(
                "dioma-render demo — пост {:?} | оверлей {} | каркас {} | \
                 спр {} fx {} меш {} св {} част {} пост {}",
                self.post_fx,
                onoff(d.show_overlay),
                onoff(d.wireframe_meshes),
                onoff(d.draw_sprites),
                onoff(d.draw_fx),
                onoff(d.draw_meshes),
                onoff(d.draw_lights),
                onoff(d.draw_particles),
                onoff(self.post_enabled),
            ));
        }
    }

    /// Пикинг: ближайшее живое существо в радиусе 0.8·size от точки `world`.
    fn pick_creature(&self, world: [f32; 2]) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (i, c) in self.creatures.iter().enumerate() {
            if c.dead {
                continue;
            }
            let d = dist2(world, c.pos());
            if d <= (c.size * 0.8).powi(2) && best.map_or(true, |(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        best.map(|(i, _)| i)
    }
}

// ── Построение мира (чистая функция: батч + атлас + существа) ──────────────

/// «Блоб»: веер из 20 сегментов с волнистым радиусом обода.
///
/// Каждый кадр радиус пересчитывается на CPU по закону
/// `base + sin(time·speed + angle·k)·amp` — это и есть применение
/// меш-пайплайна: вершины деформируются ДО push, шейдер просто рисует.
fn blob_mesh(center: [f32; 2], base_radius: f32, t: f32, uv_rect: [f32; 4]) -> Mesh {
    const SEGMENTS: u32 = 20;
    let color = [1.0, 0.6, 0.3, 1.0]; // тёплый оранжевый
    let uv_center = [(uv_rect[0] + uv_rect[2]) * 0.5, (uv_rect[1] + uv_rect[3]) * 0.5];
    let uv_half = [(uv_rect[2] - uv_rect[0]) * 0.5, (uv_rect[3] - uv_rect[1]) * 0.5];

    let mut mesh = Mesh {
        vertices: Vec::with_capacity(SEGMENTS as usize + 1),
        indices: Vec::with_capacity(SEGMENTS as usize * 3),
    };
    mesh.vertices.push(MeshVertex { position: center, uv: uv_center, color });
    for i in 0..SEGMENTS {
        let ang = i as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
        // Три «лепестка»: радиус пульсирует с углом и временем.
        let radius = base_radius + (t * 2.2 + ang * 3.0).sin() * 7.0;
        let dir = [ang.cos(), ang.sin()];
        mesh.vertices.push(MeshVertex {
            position: [center[0] + dir[0] * radius, center[1] + dir[1] * radius],
            uv: [uv_center[0] + dir[0] * uv_half[0], uv_center[1] + dir[1] * uv_half[1]],
            color,
        });
    }
    for i in 0..SEGMENTS {
        mesh.indices.push(0);
        mesh.indices.push(i + 1);
        mesh.indices.push((i + 1) % SEGMENTS + 1);
    }
    mesh
}

/// Один кадр криттера 36×36: тело-круг + две ноги-линии + глаз.
///
/// Фазы походки: 0 — нейтрально, 1 — левая вперёд / правая назад,
/// 2 — нейтрально, 3 — правая вперёд / левая назад.
fn critter_frame_rgba(phase: u32) -> (u32, u32, Vec<u8>) {
    const S: u32 = 36;
    let mut img = vec![0u8; (S * S * 4) as usize];
    let leg = [55, 60, 80, 255];
    let body = [205, 215, 235, 255];
    // Ноги: от низа тела (y=24) вниз до y=32; вперёд = наружу от корпуса.
    let (leg_l, leg_r): ((f32, f32, f32, f32), (f32, f32, f32, f32)) = match phase {
        0 => ((14.0, 24.0, 14.0, 32.0), (22.0, 24.0, 22.0, 32.0)),
        1 => ((14.0, 24.0, 7.0, 32.0), (22.0, 24.0, 29.0, 32.0)),
        2 => ((14.0, 24.0, 14.0, 32.0), (22.0, 24.0, 22.0, 32.0)),
        _ => ((14.0, 24.0, 21.0, 32.0), (22.0, 24.0, 15.0, 32.0)),
    };
    let (_, _, lp) = atlas::line_rgba(S, S, leg_l.0, leg_l.1, leg_l.2, leg_l.3, 3.0, leg);
    let (_, _, rp) = atlas::line_rgba(S, S, leg_r.0, leg_r.1, leg_r.2, leg_r.3, 3.0, leg);
    blit(&mut img, S, S, &lp, S, S, 0, 0);
    blit(&mut img, S, S, &rp, S, S, 0, 0);
    // Тело — мягкий круг поверх ног (перекрывает верх ног): центр (18, 16).
    let (bw, bh, body_px) = atlas::soft_circle_rgba(10, body);
    blit(&mut img, S, S, &body_px, bw, bh, 18 - bw as i32 / 2, 16 - bh as i32 / 2);
    // Глаз.
    let (ew, eh, eye_px) = atlas::circle_rgba(2, [40, 45, 65, 255]);
    blit(&mut img, S, S, &eye_px, ew, eh, 20 - ew as i32 / 2, 13 - eh as i32 / 2);
    (S, S, img)
}

/// Криттер 48×48 для PNG-экспорта: прозрачное поле ~6px по краям, тело —
/// мягкий круг, глаз со смещением, две ноги-линии и хохолок на макушке.
/// Тот же «сборный» подход, что у [`critter_frame_rgba`], но крупнее и с
/// полями — после обрезки при загрузке (add_png) останется ~36×36.
fn critter_png_rgba() -> (u32, u32, Vec<u8>) {
    const S: u32 = 48;
    let mut img = vec![0u8; (S * S * 4) as usize];
    let leg = [60, 65, 85, 255];
    let body = [235, 200, 130, 255];
    let eye = [45, 50, 70, 255];
    // Ноги: от низа тела к нижним углам кадра (до поля ~6px).
    let (_, _, lp) = atlas::line_rgba(S, S, 20.0, 31.0, 8.0, 39.5, 3.5, leg);
    let (_, _, rp) = atlas::line_rgba(S, S, 28.0, 31.0, 40.0, 39.5, 3.5, leg);
    blit(&mut img, S, S, &lp, S, S, 0, 0);
    blit(&mut img, S, S, &rp, S, S, 0, 0);
    // Тело — мягкий круг в центре кадра (24, 24), радиус 14.
    let (bw, bh, body_px) = atlas::soft_circle_rgba(14, body);
    blit(&mut img, S, S, &body_px, bw, bh, 24 - bw as i32 / 2, 24 - bh as i32 / 2);
    // Глаз — тёмный кружок со смещением вправо-вверх.
    let (ew, eh, eye_px) = atlas::circle_rgba(3, eye);
    blit(&mut img, S, S, &eye_px, ew, eh, 29 - ew as i32 / 2, 16 - eh as i32 / 2);
    // Хохолок: вертикальная линия до верхнего поля (касается y=6).
    let (_, _, tuft_px) = atlas::line_rgba(S, S, 24.0, 9.0, 24.0, 7.0, 2.0, leg);
    blit(&mut img, S, S, &tuft_px, S, S, 0, 0);
    (S, S, img)
}

/// Процедурный тайлсет 16×16: 4 тайла 8×8 в сетке 2×2 (индексы 1..=4).
/// 1 — трава (зелёная с «шумными» крапинками по хэшу), 2 — земля (коричневая),
/// 3 — вода (синяя со светлой линией волны), 4 — стена (тёмный камень со швами).
fn tileset_rgba() -> (u32, u32, Vec<u8>) {
    const S: u32 = 16;
    let mut img = vec![0u8; (S * S * 4) as usize];
    let hash = |x: u32, y: u32, seed: u32| -> u32 {
        let mut h = x
            .wrapping_mul(374_761_393)
            .wrapping_add(y.wrapping_mul(668_265_263))
            .wrapping_add(seed);
        h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
        (h ^ (h >> 16)) & 0xff
    };
    for y in 0..S {
        for x in 0..S {
            let (cx, cy) = (x / 8, y / 8);
            let (lx, ly) = (x % 8, y % 8);
            let i = ((y * S + x) * 4) as usize;
            let c: [u8; 4] = match (cx, cy) {
                (0, 0) => {
                    // Трава: зелёная база + крапинки-перепады оттенка по хэшу.
                    let n = hash(x, y, 11);
                    [34 + (n % 28) as u8, 128 + (n % 36) as u8, 42 + (n % 20) as u8, 255]
                }
                (1, 0) => {
                    // Земля: коричневая с тёмными вкраплениями.
                    let n = hash(x, y, 23);
                    [124 + (n % 32) as u8, 82 + (n % 24) as u8, 44 + (n % 14) as u8, 255]
                }
                (0, 1) => {
                    // Вода: синяя + светлая линия волны на ly == 5.
                    let wave = if ly == 5 && (lx == 1 || lx == 4) { 190 } else { 0 };
                    // (186 + wave) в u8 переполняется ДО .min() — считаем в u16.
                    [26 + wave / 4, 88 + wave / 4, (186 + u16::from(wave)).min(255) as u8, 255]
                }
                _ => {
                    // Стена: тёмный камень со швами по краям ячейки.
                    let edge = if lx == 7 || ly == 7 { 26 } else { 0 };
                    [62 + edge / 2, 58 + edge / 2, 70 + edge / 2, 255]
                }
            };
            img[i..i + 4].copy_from_slice(&c);
        }
    }
    (S, S, img)
}

/// PNG-криттер: шагает ПО ПОЛУ комнаты (как флипбук-ходок), кадр один —
/// загружен из PNG-файла и обрезан при загрузке. Мировая позиция
/// соответствует центру НЕОБРЕЗАННОГО изображения: после обрезки полей квад
/// сдвигается на обрезанное смещение, пересчитанное в мировые пиксели:
/// `world_pos − (offset / img_size) · sprite_size` — offset в пикселях
/// исходного изображения (48×48), поэтому до масштабирования делим на img_size.
fn build_png_critter(batch: &mut SpriteBatch, atlas: &SpriteAtlas, t: f32) {
    let Some(info) = atlas.region_info("png-critter") else { return };
    let img_size = 48.0; // исходный размер PNG до обрезки
    let sprite_size = 44.0;
    // Ходьба по полу: X колеблется по комнате, ноги на FLOOR_Y.
    let world_pos = [WALK_CENTER_X + 230.0 * (t * 0.35).sin(), FLOOR_Y + sprite_size * 0.5];
    let shift = [
        info.offset[0] / img_size * sprite_size,
        info.offset[1] / img_size * sprite_size,
    ];
    // world_pos − (offset / img_size) · sprite_size
    let mut s = Sprite::new(
        [world_pos[0] - shift[0], world_pos[1] - shift[1]],
        [sprite_size, sprite_size],
        info.uv,
    )
    .with_depth(world_pos[1]);
    s.rotation = (t * 8.0).sin() * 0.12; // лёгкое покачивание «на ходу»
    s.layer = 0.15;
    batch.push(s);
}

/// Копирует RGBA-буфер `src` (sw×sh) в `dst` (dw×dh) со смещением (dx, dy),
/// пропуская полностью прозрачные пиксели.
fn blit(dst: &mut [u8], dw: u32, dh: u32, src: &[u8], sw: u32, sh: u32, dx: i32, dy: i32) {
    for y in 0..sh {
        for x in 0..sw {
            let si = ((y * sw + x) * 4) as usize;
            if src[si + 3] == 0 {
                continue;
            }
            let px = dx + x as i32;
            let py = dy + y as i32;
            if px < 0 || py < 0 || px >= dw as i32 || py >= dh as i32 {
                continue;
            }
            let di = ((py as u32 * dw + px as u32) * 4) as usize;
            dst[di..di + 4].copy_from_slice(&src[si..si + 4]);
        }
    }
}

/// Криттеры: ходок «шагает» по полу комнаты (X колеблется, ноги на `FLOOR_Y`),
/// стоячие стоят на полу (глубина = Y позиции — при лёгком перекрытии видна
/// Y-сортировка: больший Y рисуется поверх).
fn build_critters(
    batch: &mut SpriteBatch,
    atlas: &SpriteAtlas,
    critters: &mut [Critter],
    flipbook: &Flipbook,
) {
    let uv0 = atlas.region("critter_0").unwrap_or([0.0, 0.0, 1.0, 1.0]);
    for c in critters.iter() {
        let pos = if c.walking {
            [
                WALK_CENTER_X + c.radius * c.angle.cos(),
                FLOOR_Y + c.size * 0.5, // центр спрайта: ноги стоят на полу
            ]
        } else {
            c.pos
        };
        let mut s = Sprite::new(pos, [c.size, c.size], uv0)
            .with_frame(flipbook, c.anim_time)
            .with_tint(c.tint)
            .with_depth(if c.walking { pos[1] } else { c.depth });
        s.layer = if c.walking { 0.15 } else { 0.3 };
        batch.push(s);
    }
}

fn build_world(batch: &mut SpriteBatch, atlas: &SpriteAtlas, creatures: &[Creature]) {
    // Фон (солнце, кольцо, ягоды) переехал в параллакс-слои — см. frame:
    // render_parallax рисует их со сдвинутой камерой. Здесь — только существа.

    // Существа: тело + глаз (два региона атласа) + полоса состояния.
    for (i, c) in creatures.iter().enumerate() {
        if c.dead {
            continue;
        }
        let pos = c.pos();
        let layer = 0.2 + i as f32 * 0.01;

        let mut body = Sprite::new(pos, [c.size, c.size], atlas.region("creature-body").unwrap())
            .with_tint(c.tint)
            .with_fx(c.frame_fx())
            .with_rotation(c.orbit_angle);
        body.layer = layer;
        if c.mood == Mood::Happy {
            body = body.with_pulse(0.92, 1.08, 2.5, c.orbit_angle * 2.0);
        }
        batch.push(body);

        // Глаз: смотрит наружу от солнца по орбите.
        let look = [c.orbit_angle.cos(), c.orbit_angle.sin()];
        let eye_pos = [
            pos[0] + look[0] * c.size * 0.28,
            pos[1] + look[1] * c.size * 0.28,
        ];
        let mut eye =
            Sprite::new(eye_pos, [c.size * 0.22, c.size * 0.22], atlas.region("creature-eye").unwrap())
                .with_rotation(c.orbit_angle);
        eye.layer = layer + 0.001;
        batch.push(eye);

        // Полоса состояния над выбранным существом (UV-клип показывает долю).
        if c.selected {
            let health = if c.dissolving { (1.0 - c.dissolve_progress).max(0.0) } else { 1.0 };
            let bar_y = pos[1] + c.size * 0.75;
            let mut bg = Sprite::new([pos[0], bar_y], [44.0, 7.0], atlas.region("bar-bg").unwrap());
            bg.layer = layer + 0.002;
            batch.push(bg);

            let fill_uv = atlas.region("bar-fill").unwrap();
            let mut fill = Sprite::new([pos[0], bar_y], [44.0, 7.0], fill_uv);
            fill.uv_rect = [
                fill_uv[0],
                fill_uv[1],
                fill_uv[0] + (fill_uv[2] - fill_uv[0]) * health,
                fill_uv[3],
            ];
            fill.layer = layer + 0.003;
            batch.push(fill);
        }
    }
}

fn dist2(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

/// «вкл»/«выкл» для заголовка окна и строк оверлея.
fn onoff(b: bool) -> &'static str {
    if b {
        "вкл"
    } else {
        "выкл"
    }
}

// ── winit ApplicationHandler ────────────────────────────────────────────────

impl ApplicationHandler for Demo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("dioma-render demo")
                        .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0)),
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
                label: Some("dioma-render demo device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .unwrap();
        // wgpu 22 Device не клонируется; атлас хранит клон устройства для
        // авто-роста, поэтому держим его в Arc.
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

        // ── Процедурный атлас: солнце, кольцо, 4 ягоды, тело+глаз существа,
        // полосы и 4 кадра криттера (ноги line_rgba + тело soft_circle) ──
        let mut atlas = SpriteAtlas::new(&device, 256);
        let (w, h, px) = atlas::circle_rgba(24, [255, 166, 77, 255]);
        atlas.add_rgba(&queue, "sun", w, h, &px).unwrap();
        let (w, h, px) = atlas::ring_rgba(56, 5.0, [80, 200, 190, 255]);
        atlas.add_rgba(&queue, "ring", w, h, &px).unwrap();
        for (i, name) in ["berry-a", "berry-b", "berry-c", "berry-d"].iter().enumerate() {
            let (w, h, px) = atlas::berry_rgba(i as f32 * 1.7, [200 + i as u8 * 12, 60, 90, 255]);
            atlas.add_rgba(&queue, name, w, h, &px).unwrap();
        }
        let (w, h, px) = atlas::circle_rgba(18, [255, 255, 255, 255]);
        atlas.add_rgba(&queue, "creature-body", w, h, &px).unwrap();
        let (w, h, px) = atlas::circle_rgba(3, [30, 30, 40, 255]);
        atlas.add_rgba(&queue, "creature-eye", w, h, &px).unwrap();
        let (w, h, px) = atlas::bar_rgba(44, 7, [70, 80, 110, 255], [70, 80, 110, 255], 0.0);
        atlas.add_rgba(&queue, "bar-bg", w, h, &px).unwrap();
        let (w, h, px) = atlas::bar_rgba(44, 7, [120, 230, 140, 255], [120, 230, 140, 255], 1.0);
        atlas.add_rgba(&queue, "bar-fill", w, h, &px).unwrap();
        for i in 0..4u32 {
            let (w, h, px) = critter_frame_rgba(i);
            atlas.add_rgba(&queue, &format!("critter_{}", i), w, h, &px).unwrap();
        }

        // ── Тайлсет 2×2 (16×16): трава, земля, вода, стена ──
        let (tw, th, tpx) = tileset_rgba();
        atlas.add_rgba(&queue, "tileset", tw, th, &tpx).unwrap();
        let tileset = TilesetUv::new(atlas.region("tileset").unwrap(), 2, 2);

        // ── Комната 20×4 тайла по 32 px (мировая Y 0..128): ряд данных 0 —
        // ПОЛ комнаты (стена-камень; при Y вверх это низ экрана), ряды 1..2 —
        // травяной пол с прудом воды (3) и пятнами земли (2), ряд 3 — потолок.
        // Криттеры шагают по полу: ноги на FLOOR_Y = 32. ──
        let room_rows: [&[u16]; 4] = [
            &[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            &[4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 4],
            &[4, 1, 1, 3, 3, 3, 3, 3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 4],
            &[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        ];
        let room = Tilemap::from_rows(&room_rows, 32.0);
        let mut tile_batch = TileBatch::new(&device, &room, &tileset);
        tile_batch.upload(&queue);
        // Параллакс-батчи наполняются каждый кадр (см. frame).
        let parallax_far = SpriteBatch::new(&device, 2);
        let parallax_mid = SpriteBatch::new(&device, 3);

        // ── PNG-ассет: криттер 48×48 с прозрачным полем 6px → temp-файл →
        // в атлас через add_png (поля обрезаются при загрузке). Пример
        // save_png + add_png; файл остаётся на диске (не удаляем). ──
        let (iw, ih, ipx) = critter_png_rgba();
        let png_path = std::env::temp_dir().join("dioma_render_critter.png");
        atlas::save_png(png_path.to_str().unwrap(), iw, ih, &ipx).expect("сохранить PNG-криттера");
        atlas
            .add_png(&queue, "png-critter", png_path.to_str().unwrap())
            .expect("загрузить PNG-криттера в атлас");

        // Флипбук походки: 4 кадра, 8 fps, зациклен.
        let critter_frames: Vec<[f32; 4]> =
            (0..4).map(|i| atlas.region(&format!("critter_{}", i)).unwrap()).collect();
        let critter_flipbook = Flipbook::new(critter_frames, 8.0, true);

        let mut renderer = Renderer::new(&device, format);
        renderer.set_clear_color(wgpu::Color { r: 0.015, g: 0.02, b: 0.06, a: 1.0 });

        let mut post = PostProcess::new(&device, format);
        post.resize(&device, size.width.max(1), size.height.max(1));

        let font = DebugFont::new(&device, &queue);

        let creatures = vec![
            Creature {
                orbit_angle: 0.0,
                orbit_speed: 0.5,
                orbit_radius: 120.0,
                size: 44.0,
                tint: [1.0, 0.85, 0.5, 1.0],
                mood: Mood::Happy,
                selected: false,
                dissolving: false,
                dissolve_progress: 0.0,
                dead: false,
                respawn_timer: 0.0,
            },
            Creature {
                orbit_angle: 2.1,
                orbit_speed: 0.35,
                orbit_radius: 135.0,
                size: 38.0,
                tint: [0.6, 0.9, 1.0, 1.0],
                mood: Mood::Sick,
                selected: false,
                dissolving: false,
                dissolve_progress: 0.0,
                dead: false,
                respawn_timer: 0.0,
            },
            Creature {
                orbit_angle: 4.2,
                orbit_speed: 0.45,
                orbit_radius: 105.0,
                size: 40.0,
                tint: [1.0, 0.7, 0.85, 1.0],
                mood: Mood::Calm,
                selected: false,
                dissolving: false,
                dissolve_progress: 0.0,
                dead: false,
                respawn_timer: 0.0,
            },
        ];

        // Криттеры: ходок шагает по полу (X 80..560, ноги на FLOOR_Y),
        // стоячие стоят на полу в ряд (двое слегка перекрываются с разной
        // глубиной Y — видна Y-сортировка).
        let critters = vec![
            Critter {
                pos: [0.0, 0.0],
                angle: 0.0,
                radius: 240.0,
                speed: 0.45,
                anim_time: 0.0,
                size: 44.0,
                tint: [1.0, 0.95, 0.6, 1.0],
                depth: 0.0,
                walking: true,
            },
            Critter {
                pos: [170.0, 60.0],
                angle: 0.0,
                radius: 0.0,
                speed: 0.0,
                anim_time: 3.0,
                size: 56.0,
                tint: [0.7, 0.85, 1.0, 1.0],
                depth: 60.0,
                walking: false,
            },
            Critter {
                pos: [184.0, 64.0],
                angle: 0.0,
                radius: 0.0,
                speed: 0.0,
                anim_time: 6.0,
                size: 56.0,
                tint: [1.0, 0.75, 0.8, 1.0],
                depth: 64.0,
                walking: false,
            },
            Critter {
                pos: [520.0, 56.0],
                angle: 0.0,
                radius: 0.0,
                speed: 0.0,
                anim_time: 1.0,
                size: 48.0,
                tint: [0.8, 1.0, 0.7, 1.0],
                depth: 56.0,
                walking: false,
            },
        ];

        self.window = Some(window);
        self.surface = Some(surface);
        self.device = Some(device);
        self.queue = Some(queue);
        self.config = Some(config);
        self.renderer = Some(renderer);
        self.atlas = Some(atlas);
        // 200 тайлов комнаты + существа/криттеры/PNG — с запасом.
        self.batch = Some(SpriteBatch::new(&self.device.as_ref().unwrap(), 256));
        self.mesh_batch = Some(MeshBatch::new(&self.device.as_ref().unwrap(), 512, 2048));
        self.lights = Some(LightBatch::new(&self.device.as_ref().unwrap(), 4));
        self.particles = Some(ParticleSystem::new(&self.device.as_ref().unwrap(), 240));
        self.post = Some(post);
        self.font = Some(font);
        self.flipbook = Some(critter_flipbook);
        self.tile_batch = Some(tile_batch);
        self.parallax_far = Some(parallax_far);
        self.parallax_mid = Some(parallax_mid);
        // Камера смотрит на комнату: центр (320, 80). Комната — мировая
        // Y 0..128, поэтому при Y вверх она занимает нижнюю часть кадра
        // (строки ~252..380), а небо параллакса (Y 150..260) — верхнюю.
        self.camera = Camera::new([size.width as f32, size.height as f32]);
        self.camera.center = [320.0, 80.0];
        self.creatures = creatures;
        self.critters = critters;
        self.post_enabled = true;
        self.started = Some(Instant::now());
        self.last_frame = None;
        self.update_title();
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
                if let (Some(device), Some(post)) = (&self.device, &mut self.post) {
                    post.resize(device, size.width, size.height);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = [position.x as f32, position.y as f32];
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                let world = self.camera.screen_to_world(self.cursor);
                if let Some(picked) = self.pick_creature(world) {
                    for c in &mut self.creatures {
                        c.selected = false;
                    }
                    self.creatures[picked].selected = true;
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    // Стрелки: зажатие задаёт направление панорамирования
                    // (см. frame), отпускание — сбрасывает компоненту.
                    let pressed = event.state == ElementState::Pressed;
                    match code {
                        KeyCode::ArrowLeft => self.pan_dir[0] = if pressed { -1.0 } else { 0.0 },
                        KeyCode::ArrowRight => self.pan_dir[0] = if pressed { 1.0 } else { 0.0 },
                        KeyCode::ArrowUp => self.pan_dir[1] = if pressed { 1.0 } else { 0.0 },
                        KeyCode::ArrowDown => self.pan_dir[1] = if pressed { -1.0 } else { 0.0 },
                        _ => {}
                    }
                    if pressed {
                        match code {
                            KeyCode::Escape => event_loop.exit(),
                            KeyCode::KeyD => self.start_dissolve(),
                            KeyCode::KeyR => self.respawn_all(),
                            KeyCode::KeyV => self.cycle_post_fx(),
                            KeyCode::F1 => self.toggle_overlay(),
                            KeyCode::F2 => self.toggle_wireframe(),
                            KeyCode::Digit1 => self.toggle_pass(0),
                            KeyCode::Digit2 => self.toggle_pass(1),
                            KeyCode::Digit3 => self.toggle_pass(2),
                            KeyCode::Digit4 => self.toggle_pass(3),
                            KeyCode::Digit5 => self.toggle_pass(4),
                            KeyCode::Digit6 => self.toggle_pass(5),
                            _ => {}
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
