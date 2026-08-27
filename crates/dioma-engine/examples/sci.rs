//! Научная станция: живой спайк-растр, мембранный потенциал и активность
//! мозга существа, нарисованные egui поверх окна.
//!
//! Симуляция — чистый CPU ([`Creature::new`]): мозг шагает детерминированно,
//! данные тянутся ЧЕРЕЗ API существа (`tick` → `spike_hist` для растра,
//! `brain.debug_cpu_v` для потенциала, `brain.region_bounds` для регионов)
//! и складываются в модуль `dioma_engine::sci`. Окно рисует только egui —
//! никакой сцены, чистая наука.
//!
//! Управление:
//!   поле сида + «Сброс» — пересоздать существо; «Пауза»/«Шаг» — стоп-кадр;
//!   слайдер «тиков/кадр» — скорость; Esc — пауза, X — выход.
//!   `--frames=N` — смоук-режим (для CI/проверки).
//!
//! Панели: слева — осциллограф мембранного потенциала выбранного нейрона и
//! активность по регионам; по центру — спайк-растр (строки = нейроны,
//! столбцы = подшаги, точка = спайк).

use std::sync::Arc;
use std::time::Instant;

use dioma_engine::creature::{Action, Creature, Sense, REGION_I, REGION_ML, REGION_MR, REGION_SL, REGION_SR, REGION_WL, REGION_WR};
use dioma_engine::sci::{Activity, SpikeRaster, Trace};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// Порог спайка LIF (мВ) — рисуется оранжевой линией на осциллографе.
const V_THRESH_MV: f32 = -45.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let smoke_frames = std::env::args()
        .find_map(|a| a.strip_prefix("--frames=").and_then(|s| s.parse::<u32>().ok()));
    if let Some(n) = smoke_frames {
        println!("смоук-режим: отрисую {n} кадров и выйду");
    }
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = SciApp::new(smoke_frames);
    event_loop.run_app(&mut app)?;
    Ok(())
}

/// Состояние, редактируемое из egui-замыкания (вынесено отдельно — замыкание
/// не должно захватывать весь [`SciApp`], см. паттерн `examples/ui.rs`).
#[derive(Clone, PartialEq)]
struct SciUi {
    seed_input: String,
    paused: bool,
    step_once: bool,
    ticks_per_frame: usize,
    selected: usize,
    show_raster: bool,
    show_potential: bool,
    show_activity: bool,
    /// Эксперимент «сердце»: тонический ток на нейроны региона I → ритм.
    pacemaker_on: bool,
    restart_requested: bool,
    exit_requested: bool,
}

/// Порядок регионов и имена — как в `avoidance_layout` (константы индексов
/// публичны в `creature`).
const REGION_IDS: [usize; 7] = [REGION_SL, REGION_SR, REGION_WL, REGION_WR, REGION_I, REGION_ML, REGION_MR];
const REGION_NAMES: [&str; 7] = ["SL", "SR", "WL", "WR", "I", "ML", "MR"];
/// Цвета регионов (палитра демо).
const REGION_COLORS: [egui::Color32; 7] = [
    egui::Color32::from_rgb(255, 166, 77),   // SL
    egui::Color32::from_rgb(255, 106, 94),   // SR
    egui::Color32::from_rgb(80, 200, 190),   // WL
    egui::Color32::from_rgb(122, 155, 216),  // WR
    egui::Color32::from_rgb(200, 106, 200),  // I (тормозной)
    egui::Color32::from_rgb(120, 230, 140),  // ML
    egui::Color32::from_rgb(230, 211, 122),  // MR
];

/// Научная станция: существо + записывающие тулзы + egui.
struct SciApp {
    instance: wgpu::Instance,
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
    device: Option<Arc<wgpu::Device>>,
    queue: Option<wgpu::Queue>,
    format: wgpu::TextureFormat,
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,
    ui: SciUi,
    creature: Option<Creature>,
    /// Диапазоны регионов (для разделителей растра и активности).
    regions: Vec<(String, (usize, usize))>,
    raster: Option<SpikeRaster>,
    trace: Option<Trace>,
    activity: Option<Activity>,
    /// ЭКГ: спайки пейсмейкера за тик (кольцо, как `trace`).
    heart: Vec<f32>,
    tick: u64,
    last_action: Action,
    last_frame: Option<Instant>,
    smoke_frames: Option<u32>,
}

impl SciApp {
    fn new(smoke_frames: Option<u32>) -> Self {
        // Существо строится без GPU (CPU-динамика LIF) — сразу в new().
        let mut app = Self {
            instance: wgpu::Instance::new(wgpu::InstanceDescriptor::default()),
            window: None,
            surface: None,
            config: None,
            device: None,
            queue: None,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            egui_renderer: None,
            ui: SciUi {
                seed_input: "7".to_string(),
                paused: false,
                step_once: false,
                ticks_per_frame: 4,
                selected: 0,
                show_raster: true,
                show_potential: true,
                show_activity: true,
                pacemaker_on: false,
                restart_requested: false,
                exit_requested: false,
            },
            creature: None,
            regions: Vec::new(),
            raster: None,
            trace: None,
            activity: None,
            heart: Vec::new(),
            tick: 0,
            last_action: Action::default(),
            last_frame: None,
            smoke_frames,
        };
        app.build_creature(7);
        app
    }

    /// Создаёт существо из сида и пересоздаёт записывающие тулзы под его мозг.
    fn build_creature(&mut self, seed: u64) {
        let creature = Creature::new(seed);
        let n = creature.brain.n;
        let regions: Vec<(String, (usize, usize))> = REGION_IDS
            .iter()
            .zip(REGION_NAMES.iter())
            .map(|(&id, name)| (name.to_string(), creature.brain.region_bounds(id)))
            .collect();
        self.raster = Some(SpikeRaster::new(n, 1200)); // ~12 тиков по 100 подшагов
        self.trace = Some(Trace::new(n, 400));
        self.activity = Some(Activity::new(n, 60, regions.clone()));
        self.heart = Vec::new();
        self.regions = regions;
        self.creature = Some(creature);
        self.tick = 0;
        self.last_action = Action::default();
    }

    /// Детерминированный стимул: плывущие запахи, периодические удары о стену
    /// и еда — растр получается живым, но воспроизводимым от сида.
    fn stimulus(&self, tick: u64) -> Sense {
        let t = tick as f32;
        Sense {
            smell_l: 0.5 + 0.5 * (t * 0.05).sin(),
            smell_r: 0.5 + 0.5 * (t * 0.043).cos(),
            wall_l: ((t * 0.11).sin() * 0.5 + 0.5).min(1.0),
            wall_r: ((t * 0.09).cos() * 0.5 + 0.5).min(1.0),
            food_near: tick % 90 == 0,
            wall: tick % 150 == 0,
            place: (tick % 12) as u32,
            heading: t * 0.1,
            see_eat: false,
        }
    }

    fn step_sim(&mut self, ticks: usize) {
        for _ in 0..ticks {
            if self.ui.paused {
                break;
            }
            // Заимствование существа ограничено блоком: остальные поля (растр,
            // трейс, tick) трогаем только после его конца.
            let tick = self.tick;
            let sense = self.stimulus(tick);
            let (action, hist, v, pm_spikes) = {
                let Some(creature) = &mut self.creature else { return };
                let action = creature.tick(&sense);
                let hist = creature.spike_hist().to_vec();
                let v = creature.brain.debug_cpu_v();
                // ЧСС: сумма спайков пейсмейкер-нейронов за тик (0, если выключен).
                let pm_spikes: f32 = match creature.pacemaker() {
                    Some((idx, _)) => hist
                        .iter()
                        .map(|col| idx.iter().map(|&i| col[i]).sum::<i32>())
                        .sum::<i32>() as f32,
                    None => 0.0,
                };
                (action, hist, v, pm_spikes)
            };
            self.last_action = action;
            // Тулзы тянут данные через API существа.
            if let (Some(raster), Some(trace), Some(activity)) =
                (&mut self.raster, &mut self.trace, &mut self.activity)
            {
                for col in &hist {
                    raster.push(col);
                }
                if v.len() == hist.first().map_or(0, |c| c.len()) {
                    trace.push(&v);
                }
                activity.push_tick(&hist);
            }
            // ЭКГ-кольцо: держим последние 400 тиков ЧСС.
            self.heart.push(pm_spikes);
            if self.heart.len() > 400 {
                self.heart.drain(..self.heart.len() - 400);
            }
            self.tick += 1;
        }
    }

    /// Синхронизирует тумблер UI с существом: включение даёт тонический ток
    /// чуть выше реобазы (ритм), выключение снимает ток полностью.
    fn sync_pacemaker(&mut self) {
        let on = self.ui.pacemaker_on;
        let pm = self.creature.as_ref().and_then(|c| c.pacemaker()).is_some();
        match (on, pm) {
            (true, false) => {
                if let Some(c) = &mut self.creature {
                    c.enable_pacemaker(3, 0.06);
                }
            }
            (false, true) => {
                if let Some(c) = &mut self.creature {
                    c.disable_pacemaker();
                }
            }
            _ => {}
        }
    }

    fn frame(&mut self) {
        let now = Instant::now();
        let _dt = self.last_frame.map(|t| (now - t).as_secs_f32()).unwrap_or(1.0 / 60.0);
        self.last_frame = Some(now);

        // Симуляция ДО заимствования GPU-полей (иначе конфликт с &self.device и др.).
        if self.ui.restart_requested {
            self.ui.restart_requested = false;
            let seed = self.ui.seed_input.trim().parse::<u64>().unwrap_or(0);
            self.build_creature(seed);
        }
        self.sync_pacemaker();
        if self.ui.step_once {
            self.ui.paused = true;
            self.ui.step_once = false;
            self.step_sim(1);
        } else {
            self.step_sim(self.ui.ticks_per_frame.max(1));
        }

        let Some(device) = &self.device else { return };
        let Some(queue) = &self.queue else { return };
        let Some(surface) = &self.surface else { return };
        let Some(config) = &self.config else { return };
        let Some(window) = &self.window else { return };

        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                surface.configure(device, config);
                return;
            }
            Err(e) => {
                eprintln!("[sci] surface error: {e}");
                return;
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sci encoder"),
        });
        // Очистка тёмным + egui поверх (никакой сцены).
        let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sci clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.015, g: 0.02, b: 0.045, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // ── egui ──
        let egui_state = self.egui_state.as_mut().expect("egui state создан в resumed");
        let egui_renderer = self.egui_renderer.as_mut().expect("egui renderer создан в resumed");
        let raw_input = egui_state.take_egui_input(window);
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            build_ui(
                ctx,
                &mut self.ui,
                self.creature.as_ref(),
                self.raster.as_ref(),
                self.trace.as_ref(),
                self.activity.as_ref(),
                &self.regions,
                &self.heart,
                &self.last_action,
                self.tick,
            )
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
            egui_renderer.render(&mut rpass.forget_lifetime(), &clipped, &screen_descriptor);
        }
        for id in &full_output.textures_delta.free {
            egui_renderer.free_texture(id);
        }
        queue.submit(Some(encoder.finish()));
        frame.present();
    }
}

/// Собирает интерфейс научной станции.
fn build_ui(
    ctx: &egui::Context,
    ui_state: &mut SciUi,
    creature: Option<&Creature>,
    raster: Option<&SpikeRaster>,
    trace: Option<&Trace>,
    activity: Option<&Activity>,
    regions: &[(String, (usize, usize))],
    heart: &[f32],
    last_action: &Action,
    tick: u64,
) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = egui::Color32::from_rgba_unmultiplied(14, 18, 30, 240);
    visuals.window_fill = egui::Color32::from_rgba_unmultiplied(10, 14, 24, 250);
    visuals.selection.bg_fill = egui::Color32::from_rgb(80, 200, 190);
    ctx.set_visuals(visuals);

    // Верхняя панель: управление и показания.
    egui::TopBottomPanel::top("controls").show(ctx, |ui| {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.heading("Dioma — научная станция");
            ui.separator();
            ui.label("Сид");
            let seed_edit = ui.add(egui::TextEdit::singleline(&mut ui_state.seed_input).desired_width(70.0));
            if seed_edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                ui_state.restart_requested = true;
            }
            if ui.button("Сброс").clicked() {
                ui_state.restart_requested = true;
            }
            ui.separator();
            if ui
                .button(if ui_state.paused { "▶ Продолжить" } else { "⏸ Пауза" })
                .clicked()
            {
                ui_state.paused = !ui_state.paused;
            }
            if ui.button("⏭ Шаг (1 тик)").clicked() {
                ui_state.step_once = true;
            }
            ui.label("тиков/кадр");
            ui.add(egui::Slider::new(&mut ui_state.ticks_per_frame, 1..=40));
            ui.separator();
            ui.toggle_value(&mut ui_state.show_raster, "Растр");
            ui.toggle_value(&mut ui_state.show_potential, "Потенциал");
            ui.toggle_value(&mut ui_state.show_activity, "Активность");
            ui.separator();
            ui.toggle_value(&mut ui_state.pacemaker_on, "🫀 Пейсмейкер");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("✕ Выход").clicked() {
                    ui_state.exit_requested = true;
                }
            });
        });
        ui.add_space(4.0);
    });

    // Левая панель: осциллограф потенциала + активность регионов.
    egui::SidePanel::left("science")
        .resizable(true)
        .default_width(320.0)
        .show(ctx, |ui| {
            ui.add_space(6.0);
            ui.strong("Показания");
            ui.label(format!(
                "тик {} · steer {:.2} · drive {:.2} · eat {}",
                tick,
                last_action.steer,
                last_action.drive,
                if last_action.eat { "да" } else { "нет" }
            ));
            if let Some(c) = creature {
                ui.label(format!("нейронов {} · регионов {}", c.brain.n, regions.len()));
                ui.label(format!("спайков всего {}", raster.map_or(0, |r| r.total_spikes())));
                ui.label(format!("еда: {} порций", c.food_eaten()));
            }

            // Эксперимент «сердце»: ЭКГ пейсмейкера и текущая ЧСС.
            if ui_state.pacemaker_on && !heart.is_empty() {
                ui.add_space(10.0);
                ui.strong("🫀 Сердце (пейсмейкер, регион I)");
                let bpm = heart.last().copied().unwrap_or(0.0);
                ui.label(format!("ЧСС: {bpm:.1} спайков/тик · {:.0} уд/мин", bpm * 1200.0));
                draw_ecg(ui, heart);
            }

            if ui_state.show_potential {
                ui.add_space(10.0);
                ui.strong("Мембранный потенциал");
                if let (Some(trace), Some(creature)) = (trace, creature) {
                    if ui_state.selected >= creature.brain.n {
                        ui_state.selected = 0;
                    }
                    let region_of = |neuron: usize| -> &str {
                        for (name, (s, e)) in regions {
                            if neuron >= *s && neuron < *e {
                                return name;
                            }
                        }
                        "?"
                    };
                    egui::ComboBox::from_id_salt("neuron")
                        .selected_text(format!("{} ({})", ui_state.selected, region_of(ui_state.selected)))
                        .show_ui(ui, |ui| {
                            for i in 0..creature.brain.n {
                                ui.selectable_value(&mut ui_state.selected, i, format!("{i} ({})", region_of(i)));
                            }
                        });
                    ui.label(format!("порог спайка: {V_THRESH_MV} мВ"));
                    draw_potential(ui, trace, ui_state.selected);
                }
            }

            if ui_state.show_activity {
                ui.add_space(10.0);
                ui.strong("Активность регионов (спайков/тик на нейрон)");
                if let Some(activity) = activity {
                    for i in 0..activity.n_regions() {
                        let rate = activity.region_rate(i);
                        let name = activity.region_name(i);
                        ui.horizontal(|ui| {
                            ui.label(format!("{name}"));
                            let (rect, _) = ui.allocate_exact_size(egui::vec2(140.0, 10.0), egui::Sense::hover());
                            ui.painter().rect_filled(
                                egui::Rect::from_min_size(rect.min, egui::vec2(rect.width() * (rate * 4.0).min(1.0), rect.height())),
                                2.0,
                                REGION_COLORS[i],
                            );
                            ui.monospace(format!("{rate:.2}"));
                        });
                    }
                }
            }
            ui.add_space(4.0);
        });

    // Центр: спайк-растр.
    egui::CentralPanel::default().show(ctx, |ui| {
        if ui_state.show_raster {
            if let (Some(raster), Some(creature)) = (raster, creature) {
                ui.strong(format!("Спайк-растр · {} нейронов × {} подшагов", raster.n_neurons(), raster.len()));
                let pm = creature.pacemaker().map(|(idx, _)| idx);
                draw_raster(ui, raster, regions, creature.brain.n, pm);
            } else {
                ui.label("Растр пуст — запустите симуляцию");
            }
        }
    });
}

/// Рисует спайк-растр: строки = нейроны (сверху SL → снизу MR), столбцы =
/// подшаги, точка = спайк. Разделители и подписи — по регионам. Нейроны
/// пейсмейкера (`pm`) подсвечиваются белым, а не цветом региона.
fn draw_raster(
    ui: &mut egui::Ui,
    raster: &SpikeRaster,
    regions: &[(String, (usize, usize))],
    n_neurons: usize,
    pm: Option<&[usize]>,
) {
    let avail = ui.available_size();
    let height = avail.y.max(120.0).min(900.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(avail.x, height), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 4.0, egui::Color32::from_rgba_unmultiplied(6, 9, 16, 255));

    let cols = raster.len();
    if cols == 0 {
        return;
    }
    let cell_w = rect.width() / cols as f32;
    let cell_h = rect.height() / n_neurons.max(1) as f32;
    let dot = cell_w.max(cell_h).min(3.0);

    // Спайки: только ненулевые клетки (растр разрежен).
    for (ci, col) in raster.iter().enumerate() {
        for (ri, &s) in col.iter().enumerate() {
            if s <= 0 {
                continue;
            }
            let x = rect.left() + ci as f32 * cell_w;
            let y = rect.top() + ri as f32 * cell_h;
            let is_pm = pm.map_or(false, |pm| pm.contains(&ri));
            let color = if is_pm {
                egui::Color32::WHITE
            } else {
                region_color(ri, regions)
            };
            painter.rect_filled(
                egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(dot.max(cell_w), dot.max(cell_h))),
                0.0,
                color,
            );
        }
    }

    // Разделители регионов + подписи.
    for (name, (start, _end)) in regions {
        let y = rect.top() + *start as f32 * cell_h;
        painter.line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            egui::Stroke::new(1.0_f32, egui::Color32::from_white_alpha(40)),
        );
        painter.text(
            egui::pos2(rect.left() + 4.0, y + 2.0),
            egui::Align2::LEFT_TOP,
            name.as_str(),
            egui::FontId::monospace(10.0),
            egui::Color32::from_white_alpha(160),
        );
    }
}

/// Цвет региона нейрона `ri` (по диапазонам).
fn region_color(neuron: usize, regions: &[(String, (usize, usize))]) -> egui::Color32 {
    for (i, (_, (s, e))) in regions.iter().enumerate() {
        if neuron >= *s && neuron < *e {
            return REGION_COLORS[i];
        }
    }
    egui::Color32::GRAY
}

/// ЭКГ: спайки пейсмейкера за тик (линия-зубец, как кардиограмма).
fn draw_ecg(ui: &mut egui::Ui, heart: &[f32]) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 90.0), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 4.0, egui::Color32::from_rgba_unmultiplied(6, 9, 16, 255));
    if heart.len() < 2 {
        painter.text(rect.center(), egui::Align2::CENTER_CENTER, "ждём спайки…", egui::FontId::proportional(12.0), egui::Color32::GRAY);
        return;
    }
    let hi = heart.iter().copied().fold(0.0f32, f32::max).max(1.0);
    let y_of = |v: f32| rect.bottom() - (v / hi) * rect.height();
    let x_of = |i: usize| rect.left() + i as f32 / (heart.len() - 1) as f32 * rect.width();
    let points: Vec<egui::Pos2> = heart.iter().enumerate().map(|(i, &v)| egui::pos2(x_of(i), y_of(v))).collect();
    painter.add(egui::Shape::line(points, egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(255, 90, 90))));
    painter.text(
        egui::pos2(rect.left() + 4.0, rect.top() + 2.0),
        egui::Align2::LEFT_TOP,
        format!("ЭКГ · последние {} тиков", heart.len()),
        egui::FontId::monospace(10.0),
        egui::Color32::from_white_alpha(160),
    );
}

/// Осциллограф мембранного потенциала выбранного нейрона.
fn draw_potential(ui: &mut egui::Ui, trace: &Trace, neuron: usize) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 180.0), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 4.0, egui::Color32::from_rgba_unmultiplied(6, 9, 16, 255));

    let series = trace.series(neuron);
    if series.len() < 2 || !trace.min().is_finite() {
        painter.text(rect.center(), egui::Align2::CENTER_CENTER, "ждём сэмплы…", egui::FontId::proportional(12.0), egui::Color32::GRAY);
        return;
    }
    let (lo, hi) = (trace.min(), trace.max());
    let span = (hi - lo).max(1e-3);
    let y_of = |v: f32| rect.bottom() - (v - lo) / span * rect.height();
    let x_of = |i: usize| rect.left() + i as f32 / (series.len() - 1) as f32 * rect.width();

    // Порог спайка.
    let ty = y_of(V_THRESH_MV);
    if ty >= rect.top() && ty <= rect.bottom() {
        painter.line_segment(
            [egui::pos2(rect.left(), ty), egui::pos2(rect.right(), ty)],
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(255, 140, 60)),
        );
    }

    let points: Vec<egui::Pos2> = series.iter().enumerate().map(|(i, &v)| egui::pos2(x_of(i), y_of(v))).collect();
    painter.add(egui::Shape::line(points, egui::Stroke::new(1.5_f32, REGION_COLORS[0])));
    painter.text(
        egui::pos2(rect.left() + 4.0, rect.top() + 2.0),
        egui::Align2::LEFT_TOP,
        format!("{lo:.1} … {hi:.1} мВ"),
        egui::FontId::monospace(10.0),
        egui::Color32::from_white_alpha(160),
    );
}

impl ApplicationHandler for SciApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Dioma — научная станция")
                        .with_inner_size(LogicalSize::new(1280.0, 720.0)),
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
                label: Some("sci station device"),
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
        // egui-стек создаём ДО перемещения device в self (Arc не Copy).
        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);
        self.window = Some(window);
        self.surface = Some(surface);
        self.config = Some(config);
        self.device = Some(device);
        self.queue = Some(queue);
        self.egui_state = Some(egui_state);
        self.egui_renderer = Some(egui_renderer);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Матчим по ссылке — иначе частичное перемещение запретит передать egui.
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
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if event.state == winit::event::ElementState::Pressed {
                        match code {
                            KeyCode::Escape => self.ui.paused = !self.ui.paused,
                            KeyCode::KeyX => event_loop.exit(),
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
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
