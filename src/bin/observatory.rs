//! -------------------------------------------------------------------------
//! Filename:    src/bin/observatory.rs
//! Class:       DebugSuite
//! Purpose:     poc-observatory: egui live view of a creature
//! Description: free life in real time — world canvas, brain graph with
//!              flash, chemistry plots, playback controls. Two brains:
//!              arena (LIF) и органный (HH спина + HAAP средний мозг +
//!              LIF кора) — переключаются в top bar.
//!              Every constant comes from ui::theme tokens.
//!
//! author: Atlasov anton
// -------------------------------------------------------------------------

use dioma::free_life::{Sim, ARENA_HALF};
use dioma::arena::{grow_agent, Genome};
use dioma::grow::agent_layout;
use dioma::metabolism::chem;
use dioma::organ_life::OrganSim;
use dioma::ui::theme::{region_color, ARENA_REGION_NAMES, CHEM_NAMES, Theme};
use dioma::ui::{dot, heading_arrow, label, mix, WorldTransform};

use eframe::egui;
use egui::{Pos2, Stroke};
use egui_plot::{Legend, Line, Plot, PlotPoints};
use std::collections::VecDeque;

/// Какой мозг живёт в обсерватории.
#[derive(Clone, Copy, PartialEq)]
enum BrainKind {
    Arena,
    Organs,
}

impl BrainKind {
    fn label(self) -> &'static str {
        match self {
            BrainKind::Arena => "arena (LIF)",
            BrainKind::Organs => "organs (HH+HAAP+LIF)",
        }
    }
}

/// Унифицированное наблюдение за тиком (для UI).
struct ObsView {
    chemicals: Vec<f32>,
    /// Спайки коры по регионам [SL, SR, I, ML, MR].
    region_spikes: [i32; dioma::arena::N_REGIONS],
    /// Суммарные спайки среднего мозга (арена: 0).
    midbrain_spikes: i64,
    /// Моторные спайки спины (арена: 0).
    spine_m_spikes: i64,
    /// Существо сидит (органный режим).
    sitting: bool,
    x: f64,
    y: f64,
    ate: bool,
}

/// Модель жизни: классическая арена или органный мозг.
#[allow(clippy::large_enum_variant)]
enum LifeModel {
    Arena(Sim),
    Organs(OrganSim),
}

impl LifeModel {
    fn new(kind: BrainKind, seed: u64) -> Self {
        match kind {
            BrainKind::Arena => LifeModel::Arena(Sim::new(false, seed)),
            BrainKind::Organs => {
                LifeModel::Organs(OrganSim::new(false, seed).with_mind(seed ^ 0x5EED))
            }
        }
    }

    fn kind(&self) -> BrainKind {
        match self {
            LifeModel::Arena(_) => BrainKind::Arena,
            LifeModel::Organs(_) => BrainKind::Organs,
        }
    }

    fn step(&mut self) -> ObsView {
        match self {
            LifeModel::Arena(sim) => {
                let o = sim.step_tick();
                ObsView {
                    region_spikes: [
                        o.region_spikes[0],
                        o.region_spikes[1],
                        o.region_spikes[2],
                        o.region_spikes[3],
                        o.region_spikes[4],
                    ],
                    midbrain_spikes: 0,
                    spine_m_spikes: 0,
                    sitting: false,
                    chemicals: o.chemicals,
                    x: o.x,
                    y: o.y,
                    ate: o.ate,
                }
            }
            LifeModel::Organs(sim) => {
                let o = sim.step_tick();
                ObsView {
                    region_spikes: o.cortex_regions,
                    midbrain_spikes: o.midbrain_spikes,
                    spine_m_spikes: o.spine_m_spikes,
                    sitting: o.sitting,
                    chemicals: o.chemicals,
                    x: o.x,
                    y: o.y,
                    ate: o.ate,
                }
            }
        }
    }

    fn blood(&self) -> &dioma::metabolism::Blood {
        match self {
            LifeModel::Arena(s) => &s.blood,
            LifeModel::Organs(s) => &s.blood,
        }
    }

    fn blood_mut(&mut self) -> &mut dioma::metabolism::Blood {
        match self {
            LifeModel::Arena(s) => &mut s.blood,
            LifeModel::Organs(s) => &mut s.blood,
        }
    }

    fn tick(&self) -> usize {
        match self {
            LifeModel::Arena(s) => s.tick,
            LifeModel::Organs(s) => s.tick,
        }
    }

    fn world(&self) -> &dioma::free_life::WorldSim {
        match self {
            LifeModel::Arena(s) => &s.world,
            LifeModel::Organs(s) => &s.world,
        }
    }

    fn alive(&self) -> bool {
        self.blood().organ.damage < 1.0
    }
}

/// Состояние обсерватории.
struct Observatory {
    theme: Theme,
    model: LifeModel,
    /// История химий: тик → концентрации.
    chem_history: VecDeque<[f32; chem::N]>,
    /// История моторной активности (спайков/тик).
    motor_history: VecDeque<f32>,
    /// История органов: (midbrain, spine_m) спайков/тик.
    organ_history: VecDeque<[f64; 2]>,
    /// Сглаженная «вспышка» регионов [0..1].
    flash: Vec<f32>,
    /// След агента.
    trail: VecDeque<[f64; 2]>,
    playing: bool,
    speed_idx: usize,
    seed: u64,
    eaten_total: usize,
    trace_on: bool,
    /// Существо сидит (последний тик) — визуальный режим «покой».
    sitting: bool,
    /// Агрегированный граф регионов: (pre, post, средний вес).
    region_edges: Vec<(usize, usize, f32)>,
    /// Позиции регионов (центры layout, x/y).
    region_pos: Vec<[f64; 2]>,
}

impl Observatory {
    fn new(kind: BrainKind, seed: u64) -> Self {
        let model = LifeModel::new(kind, seed);
        // Тот же сид — тот же коннектом: агрегируем рёбра по регионам
        // (кора органного мозга — та же арена-схема).
        let genome = Genome::wildtype(seed);
        let grown = grow_agent(&genome);
        let mut sums = [0.0f32; dioma::arena::N_REGIONS * dioma::arena::N_REGIONS];
        let mut cnts = vec![0usize; sums.len()];
        for &(pre, post, w) in &grown.edges {
            let (a, b) = (grown.region_of[pre], grown.region_of[post]);
            sums[a * dioma::arena::N_REGIONS + b] += w;
            cnts[a * dioma::arena::N_REGIONS + b] += 1;
        }
        let region_edges = sums
            .iter()
            .zip(&cnts)
            .enumerate()
            .filter(|(_, (_, &c))| c > 0)
            .map(|(i, (&s, &c))| {
                (i / dioma::arena::N_REGIONS, i % dioma::arena::N_REGIONS, s / c as f32)
            })
            .collect();
        let region_pos = agent_layout()
            .regions
            .iter()
            .map(|r| [r.center[0], r.center[1]])
            .collect();

        let n_regions = dioma::arena::N_REGIONS;
        Self {
            theme: Theme::default(),
            model,
            chem_history: VecDeque::new(),
            motor_history: VecDeque::new(),
            organ_history: VecDeque::new(),
            flash: vec![0.0; n_regions],
            trail: VecDeque::new(),
            playing: true,
            speed_idx: 1,
            seed,
            eaten_total: 0,
            trace_on: false,
            sitting: false,
            region_edges,
            region_pos,
        }
    }

    /// Продвинуть симуляцию на один тик и записать историю.
    fn advance(&mut self) {
        let obs = self.model.step();
        self.sitting = obs.sitting;
        self.eaten_total += obs.ate as usize;
        self.chem_history.push_back(obs.chemicals.try_into().unwrap());
        self.motor_history
            .push_back((obs.region_spikes[dioma::arena::REG_ML] + obs.region_spikes[dioma::arena::REG_MR]) as f32);
        self.organ_history
            .push_back([obs.midbrain_spikes as f64, obs.spine_m_spikes as f64]);
        self.trail.push_back([obs.x, obs.y]);
        let hist = self.theme.viz.history_points;
        if self.chem_history.len() > hist {
            self.chem_history.pop_front();
            self.motor_history.pop_front();
            self.organ_history.pop_front();
        }
        if self.trail.len() > self.theme.viz.trail_len {
            self.trail.pop_front();
        }
        // Вспышки регионов: нормируем на размер региона, сглаживаем.
        let sizes: Vec<usize> = agent_layout().regions.iter().map(|r| r.count).collect();
        for (r, f) in self.flash.iter_mut().enumerate() {
            let norm = obs.region_spikes[r] as f32 / sizes[r].max(1) as f32;
            *f = (*f + self.theme.viz.flash_rise * norm).min(1.0);
        }
    }


    fn reset(&mut self, kind: BrainKind, seed: u64) {
        *self = Observatory::new(kind, seed);
    }

    /// Верхняя панель: воспроизведение, скорость, сброс, сводка.
    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let txt = if self.playing { "⏸ pause" } else { "▶ play" };
            if ui.button(txt).clicked() {
                self.playing = !self.playing;
            }

            ui.add_space(self.theme.spacing.gap);
            ui.label("speed ×");
            egui::ComboBox::from_id_salt("speed")
                .selected_text(format!("{}", self.theme.viz.speed_steps[self.speed_idx]))
                .show_ui(ui, |ui| {
                    for (i, s) in self.theme.viz.speed_steps.iter().enumerate() {
                        ui.selectable_value(&mut self.speed_idx, i, format!("×{s}"));
                    }
                });

            ui.add_space(self.theme.spacing.gap);
            ui.label("brain");
            egui::ComboBox::from_id_salt("brain_kind")
                .selected_text(self.model.kind().label())
                .show_ui(ui, |ui| {
                    let current = self.model.kind();
                    for kind in [BrainKind::Arena, BrainKind::Organs] {
                        if ui
                            .selectable_label(current == kind, kind.label())
                            .clicked()
                            && current != kind
                        {
                            self.reset(kind, self.seed);
                        }
                    }
                });

            ui.add_space(self.theme.spacing.gap);
            if ui.checkbox(&mut self.trace_on, "trace blood").changed()
                && self.trace_on {
                    self.model.blood_mut().enable_trace(TRACE_CAP);
                }
            if ui.button("↺ reset").clicked() {
                let kind = self.model.kind();
                self.reset(kind, self.seed);
            }
            if ui.button("🎲 new seed").clicked() {
                self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let kind = self.model.kind();
                self.reset(kind, self.seed);
            }

            ui.add_space(self.theme.spacing.gap);
            let blood = self.model.blood();
            ui.colored_label(
                self.theme.colors.text_dim,
                format!(
                    "tick {} ({:.1} s) · eaten {} · glucose {:.0} · hunger {:.0} · fatigue {:.0} · damage {:.2}",
                    self.model.tick(),
                    self.model.tick() as f64 * 0.025,
                    self.eaten_total,
                    blood.conc[chem::GLUCOSE],
                    blood.conc[chem::HUNGER],
                    blood.conc[chem::FATIGUE],
                    blood.organ.damage,
                ),
            );
        });
    }

    /// Канвас мира: сетка, еда, след, агент.
    fn world_canvas(&self, ui: &mut egui::Ui) {
        let t = &self.theme;
        let painter = ui.painter_at(ui.available_rect_before_wrap());
        let rect = ui.available_rect_before_wrap();
        painter.rect_filled(rect, 0.0, t.colors.background);

        let tr = WorldTransform::fit(rect, ARENA_HALF, t.layout.canvas_margin);

        // Сетка мира.
        let mut g = 0.0f64;
        while g <= ARENA_HALF {
            for (x, _y) in [(g, 0.0), (-g, 0.0)] {
                let a = tr.to_screen(x, ARENA_HALF);
                let b = tr.to_screen(x, -ARENA_HALF);
                painter.line_segment([a, b], Stroke::new(GRID_LINE_WIDTH, t.colors.grid));
            }
            for (x, _y) in [(0.0, g), (0.0, -g)] {
                let a = tr.to_screen(ARENA_HALF, x);
                let b = tr.to_screen(-ARENA_HALF, x);
                painter.line_segment([a, b], Stroke::new(GRID_LINE_WIDTH, t.colors.grid));
            }
            g += t.viz.world_grid_step;
        }

        // Еда.
        for (x, y, alive) in self.model.world().foods_snapshot() {
            let c = if alive { t.colors.food } else { t.colors.trail };
            dot(painter.clone(), tr.to_screen(x, y), t.viz.food_radius, c);
        }

        // След.
        if self.trail.len() >= 2 {
            let pts: Vec<Pos2> = self
                .trail
                .iter()
                .map(|&[x, y]| tr.to_screen(x, y))
                .collect();
            painter.add(egui::epaint::PathShape {
                points: pts,
                closed: false,
                fill: egui::Color32::TRANSPARENT,
                stroke: egui::epaint::PathStroke::new(TRAIL_LINE_WIDTH, t.colors.trail),
            });
        }

        // Агент.
        let world = self.model.world();
        let at = tr.to_screen(world.agent_x, world.agent_y);
        let alive = self.model.alive();
        let color = if alive { t.colors.agent } else { t.colors.dead };
        dot(painter.clone(), at, t.viz.agent_radius, color);
        if self.sitting {
            // Покой: стрелки нет, над агентом «z z».
            label(
                painter.clone(),
                Pos2::new(at.x, at.y - t.viz.agent_radius - t.spacing.inner),
                "z z",
                t.colors.text_dim,
            );
        } else {
            heading_arrow(
                painter.clone(),
                at,
                world.heading,
                t.viz.agent_radius,
                color,
            );
        }
    }

    /// Граф мозга: узлы регионов с вспышками, рёбра по среднему весу.
    fn brain_graph(&self, ui: &mut egui::Ui) {
        let t = &self.theme;
        let (rect, _resp) = ui.allocate_exact_size(
            egui::Vec2::new(ui.available_width(), t.layout.plot_height),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, t.colors.panel);

        // Вписываем region_pos (мир роста) в rect.
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        for p in &self.region_pos {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let fit = |p: [f64; 2]| -> Pos2 {
            let norm = |v: f64, a: f64, b: f64| ((v - a) / (b - a).max(1e-9)) * 2.0 - 1.0;
            Pos2::new(
                rect.center().x + norm(p[0], lo[0], hi[0]) as f32
                    * (rect.width() / 2.0 - t.viz.node_radius - t.spacing.pad),
                rect.center().y - norm(p[1], lo[1], hi[1]) as f32
                    * (rect.height() / 2.0 - t.viz.node_radius - t.spacing.pad),
            )
        };

        // Рёбра: толщина/альфа по среднему весу.
        let w_max = self
            .region_edges
            .iter()
            .map(|&(_, _, w)| w)
            .fold(0.0f32, f32::max)
            .max(1e-6);
        for &(a, b, w) in &self.region_edges {
            let strength = w / w_max;
            let color = mix(
                t.colors.panel_stroke,
                region_color(&self.theme, a),
                strength * EDGE_ALPHA_MAX,
            );
            painter.line_segment(
                [fit(self.region_pos[a]), fit(self.region_pos[b])],
                Stroke::new(EDGE_LINE_MIN + strength * EDGE_LINE_SPAN, color),
            );
        }

        // Узлы: заливка = вспышка активности.
        for (r, &pos) in self.region_pos.iter().enumerate() {
            let at = fit(pos);
            let base = region_color(&self.theme, r);
            let fill = mix(t.colors.panel, base, self.flash[r]);
            dot(painter.clone(), at, t.viz.node_radius, fill);
            painter.circle_stroke(
                at,
                t.viz.node_radius,
                Stroke::new(NODE_STROKE_WIDTH, base),
            );
            label(
                painter.clone(),
                Pos2::new(at.x, at.y + t.viz.node_radius + t.spacing.inner),
                ARENA_REGION_NAMES[r],
                t.colors.text_dim,
            );
        }
    }

    /// Панель трейса крови: последние тики с событиями (отладка химии).
    fn trace_panel(&self, ui: &mut egui::Ui) {
        if !self.trace_on {
            return;
        }
        ui.add_space(self.theme.spacing.gap);
        ui.heading("blood trace");
        egui::ScrollArea::vertical()
            .max_height(self.theme.layout.plot_height * TRACE_PANEL_HEIGHT_FRAC)
            .show(ui, |ui| {
                let Some(trace) = self.model.blood().trace() else {
                    ui.label("trace выключен");
                    return;
                };
                for t in trace.iter().rev().take(TRACE_ROWS) {
                    let mark = if t.ate { "🍽" } else { " " };
                    let wall = if t.wall { "💥" } else { " " };
                    ui.monospace(format!(
                        "{mark}{wall} t{:>5} G={:>5.1} H={:>5.1} F={:>5.1} dmg={:.2} speed={:.2}",
                        t.tick,
                        t.chemicals[chem::GLUCOSE],
                        t.chemicals[chem::HUNGER],
                        t.chemicals[chem::FATIGUE],
                        t.damage,
                        t.loci[dioma::metabolism::locus::SPEED],
                    ));
                }
            });
    }

    /// Панель разума (karmac SpikingBrain): полосы действий, дофамин,
    /// теплокарта W2 [5 действий × 128 выходных нейронов].
    fn mind_panel(&self, ui: &mut egui::Ui) {
        let t = &self.theme;
        let mind = match &self.model {
            LifeModel::Organs(s) => s.mind.as_ref(),
            _ => None,
        };
        let Some(mind) = mind else {
            return;
        };

        // Полосы действий: частота пула + подсветка выбранного.
        let names = ["сидит", "ищет", "подходит", "бежит", "бродит"];
        let rates = mind.action_rates();
        let max = rates.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
        for (i, name) in names.iter().enumerate() {
            let selected = self.last_action_index() == Some(i);
            ui.horizontal(|ui| {
                let w = 64.0;
                ui.add_sized([w, 14.0], egui::Label::new(
                    egui::RichText::new(*name).color(if selected {
                        t.colors.accent
                    } else {
                        t.colors.text_dim
                    }),
                ));
                let rect = ui.allocate_exact_size(
                    egui::Vec2::new(ui.available_width(), 10.0),
                    egui::Sense::hover(),
                ).0;
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 2.0, t.colors.panel);
                let fill_w = rect.width() * (rates[i] / max).clamp(0.0, 1.0);
                if fill_w > 1.0 {
                    let color = if selected { t.colors.accent } else { t.colors.trail };
                    painter.rect_filled(
                        egui::Rect::from_min_size(rect.min, egui::Vec2::new(fill_w, rect.height())),
                        2.0,
                        color,
                    );
                }
            });
        }

        // Дофамин и базлайн.
        ui.horizontal(|ui| {
            let da = mind.dopamine();
            let col = if da >= 0.0 {
                egui::Color32::from_rgb(120, 220, 120)
            } else {
                egui::Color32::from_rgb(230, 90, 90)
            };
            ui.colored_label(col, format!("DA {da:+.2}"));
            ui.colored_label(t.colors.text_dim, format!("base {:.2}", mind.baseline()));
        });

        ui.add_space(t.spacing.gap);
        ui.label("W2 (5×128)");

        // Теплокарта W2: строки — действия, столбцы — выходные нейроны.
        let heat = mind.weight_heat();
        let cols = 128usize;
        let rows = 5usize;
        let cell_w = ui.available_width() / cols as f32;
        let cell_h = 4.0;
        let (rect, _) = ui.allocate_exact_size(
            egui::Vec2::new(ui.available_width(), cell_h * rows as f32),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, t.colors.panel);
        for r in 0..rows {
            for c in 0..cols {
                let v = heat[r * cols + c];
                // Отрицательные — красные, положительные — зелёные,
                // интенсивность по модулю.
                let a = (v.abs() / W2_HEAT_MAX).clamp(0.05, 1.0);
                let col32 = if v < 0.0 {
                    egui::Color32::from_rgba_unmultiplied(230, 90, 90, (a * 255.0) as u8)
                } else {
                    egui::Color32::from_rgba_unmultiplied(90, 220, 130, (a * 255.0) as u8)
                };
                let p = Pos2::new(rect.left() + c as f32 * cell_w, rect.top() + r as f32 * cell_h);
                painter.rect_filled(
                    egui::Rect::from_min_size(p, egui::Vec2::new(cell_w, cell_h)),
                    0.0,
                    col32,
                );
            }
        }
    }

    /// Индекс последнего действия разума (для подсветки полос).
    fn last_action_index(&self) -> Option<usize> {
        match &self.model {
            LifeModel::Organs(s) => s.mind.as_ref().map(|m| m.last_action as usize),
            _ => None,
        }
    }

    /// Графики химий, моторной активности и органов.
    fn plots(&self, ui: &mut egui::Ui) {
        let t = &self.theme;
        Plot::new("chemistry")
            .height(t.layout.plot_height)
            .legend(Legend::default())
            .include_y(0.0)
            .include_y(t.viz.chem_axis_max)
            .show_background(true)
            .show(ui, |plot_ui| {
                for c in 0..chem::N {
                    let pts: PlotPoints = self
                        .chem_history
                        .iter()
                        .enumerate()
                        .map(|(i, conc)| [i as f64, conc[c] as f64])
                        .collect();
                    plot_ui.line(
                        Line::new(pts)
                            .name(CHEM_NAMES[c])
                            .color(t.colors.chemicals[c])
                            .width(t.viz.line_width),
                    );
                }
            });

        ui.add_space(t.spacing.gap);

        Plot::new("motor")
            .height(t.layout.plot_height * MOTOR_PLOT_HEIGHT_FRAC)
            .include_y(0.0)
            .include_y(t.viz.motor_axis_max)
            .show(ui, |plot_ui| {
                let pts: PlotPoints = self
                    .motor_history
                    .iter()
                    .enumerate()
                    .map(|(i, v)| [i as f64, *v as f64])
                    .collect();
                plot_ui.line(
                    Line::new(pts)
                        .name("motor spikes/tick")
                        .color(t.colors.accent)
                        .width(t.viz.line_width),
                );
            });

        // Органный режим: активность среднего мозга и спинного рефлекса.
        if self.model.kind() == BrainKind::Organs {
            ui.add_space(t.spacing.gap);
            Plot::new("organs")
                .height(t.layout.plot_height * MOTOR_PLOT_HEIGHT_FRAC)
                .legend(Legend::default())
                .include_y(0.0)
                .show_background(true)
                .show(ui, |plot_ui| {
                    let mid: PlotPoints = self
                        .organ_history
                        .iter()
                        .enumerate()
                        .map(|(i, v)| [i as f64, v[0]])
                        .collect();
                    plot_ui.line(
                        Line::new(mid)
                            .name("midbrain (мотивация)")
                            .color(egui::Color32::from_rgb(255, 170, 60))
                            .width(t.viz.line_width),
                    );
                    let spine: PlotPoints = self
                        .organ_history
                        .iter()
                        .enumerate()
                        .map(|(i, v)| [i as f64, v[1] * 10.0]) // ×10 для видимости
                        .collect();
                    plot_ui.line(
                        Line::new(spine)
                            .name("спина ×10 (рефлекс)")
                            .color(egui::Color32::from_rgb(230, 90, 90))
                            .width(t.viz.line_width),
                    );
                });
        }
    }
}

// Доли/альфы — токены бинаря.
const EDGE_ALPHA_MAX: f32 = 0.7;
const GRID_LINE_WIDTH: f32 = 1.0;
const TRAIL_LINE_WIDTH: f32 = 1.0;
const NODE_STROKE_WIDTH: f32 = 1.0;
const EDGE_LINE_MIN: f32 = 1.0;
const EDGE_LINE_SPAN: f32 = 2.0;
const MOTOR_PLOT_HEIGHT_FRAC: f32 = 0.5;

impl eframe::App for Observatory {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.playing {
            for _ in 0..self.theme.viz.speed_steps[self.speed_idx] {
                self.advance();
                if !self.model.alive() {
                    self.playing = false;
                    break;
                }
            }
            // Затухание вспышек — за кадр.
            for f in self.flash.iter_mut() {
                *f *= self.theme.viz.flash_decay;
            }
            ctx.request_repaint();
        }

        egui::TopBottomPanel::top("top_bar")
            .exact_height(self.theme.layout.top_bar_height)
            .show(ctx, |ui| {
                ui.add_space(self.theme.spacing.pad);
                self.top_bar(ui);
            });

        egui::SidePanel::right("inspector")
            .exact_width(self.theme.layout.right_panel_width)
            .show(ctx, |ui| {
                ui.add_space(self.theme.spacing.pad);
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.vertical(|ui| {
                        ui.heading("brain");
                        self.brain_graph(ui);
                        if self.model.kind() == BrainKind::Organs {
                            ui.add_space(self.theme.spacing.gap);
                            ui.heading("mind");
                            self.mind_panel(ui);
                            ui.add_space(self.theme.spacing.gap);
                        }
                        ui.heading("blood");
                        self.plots(ui);
                        self.trace_panel(ui);
                    });
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.world_canvas(ui);
        });
    }
}

fn main() -> eframe::Result {
    let theme = Theme::default();
    let window_size = theme.layout.window;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(window_size),
        ..Default::default()
    };
    eframe::run_native(
        "dioma observatory",
        options,
        Box::new(|_cc| Ok(Box::new(Observatory::new(BrainKind::Organs, DEFAULT_SEED)))),
    )
}

const DEFAULT_SEED: u64 = 0x0000_AE11;
const TRACE_CAP: usize = 256;
const TRACE_ROWS: usize = 12;
const TRACE_PANEL_HEIGHT_FRAC: f32 = 0.5;
const W2_HEAT_MAX: f32 = 0.35;
