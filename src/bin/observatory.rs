//! -------------------------------------------------------------------------
//! Filename:    src/bin/observatory.rs
//! Class:       DebugSuite
//! Purpose:     poc-observatory: egui live view of a creature
//! Description: free-life Sim in real time — world canvas, brain graph
//!              with flash, chemistry plots, playback controls.
//!              Every constant comes from ui::theme tokens.
//!
//! author: Atlasov anton
//! -------------------------------------------------------------------------

use dioma::free_life::{Sim, TickObs, ARENA_HALF};
use dioma::arena::{grow_agent, Genome};
use dioma::grow::agent_layout;
use dioma::metabolism::chem;
use dioma::ui::theme::{region_color, ARENA_REGION_NAMES, CHEM_NAMES, Theme};
use dioma::ui::{dot, heading_arrow, label, mix, WorldTransform};

use eframe::egui;
use egui::{Pos2, Stroke};
use egui_plot::{Legend, Line, Plot, PlotPoints};
use std::collections::VecDeque;

/// Состояние обсерватории.
struct Observatory {
    theme: Theme,
    sim: Sim,
    /// История химий: тик → концентрации.
    chem_history: VecDeque<[f32; chem::N]>,
    /// История моторной активности (спайков/тик).
    motor_history: VecDeque<f32>,
    /// Сглаженная «вспышка» регионов [0..1].
    flash: Vec<f32>,
    /// След агента.
    trail: VecDeque<[f64; 2]>,
    playing: bool,
    speed_idx: usize,
    seed: u64,
    eaten_total: usize,
    trace_on: bool,
    /// Агрегированный граф регионов: (pre, post, средний вес).
    region_edges: Vec<(usize, usize, f32)>,
    /// Позиции регионов (центры layout, x/y).
    region_pos: Vec<[f64; 2]>,
}

impl Observatory {
    fn new(seed: u64) -> Self {
        let sim = Sim::new(false, seed);
        // Тот же сид — тот же коннектом: агрегируем рёбра по регионам.
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
            sim,
            chem_history: VecDeque::new(),
            motor_history: VecDeque::new(),
            flash: vec![0.0; n_regions],
            trail: VecDeque::new(),
            playing: true,
            speed_idx: 1,
            seed,
            eaten_total: 0,
            trace_on: false,
            region_edges,
            region_pos,
        }
    }

    /// Продвинуть симуляцию на один тик и записать историю.
    fn advance(&mut self) {
        let obs: TickObs = self.sim.step_tick();
        self.eaten_total += obs.ate as usize;
        self.chem_history.push_back(obs.chemicals.try_into().unwrap());
        self.motor_history
            .push_back((obs.region_spikes[dioma::arena::REG_ML] + obs.region_spikes[dioma::arena::REG_MR]) as f32);
        self.trail.push_back([obs.x, obs.y]);
        let hist = self.theme.viz.history_points;
        if self.chem_history.len() > hist {
            self.chem_history.pop_front();
            self.motor_history.pop_front();
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
        let _ = obs.alive;
    }


    fn reset(&mut self, seed: u64) {
        *self = Observatory::new(seed);
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
            if ui.checkbox(&mut self.trace_on, "trace blood").changed()
                && self.trace_on {
                    self.sim.blood.enable_trace(TRACE_CAP);
                }
            if ui.button("↺ reset").clicked() {
                self.reset(self.seed);
            }
            if ui.button("🎲 new seed").clicked() {
                self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                self.reset(self.seed);
            }

            ui.add_space(self.theme.spacing.gap);
            let blood = &self.sim.blood;
            ui.colored_label(
                self.theme.colors.text_dim,
                format!(
                    "tick {} ({:.1} s) · eaten {} · glucose {:.0} · hunger {:.0} · fatigue {:.0} · damage {:.2}",
                    self.sim.tick,
                    self.sim.tick as f64 * 0.025,
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
        for (x, y, alive) in self.sim.world.foods_snapshot() {
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
        let at = tr.to_screen(self.sim.world.agent_x, self.sim.world.agent_y);
        let alive = self.sim.blood.organ.damage < 1.0;
        let color = if alive { t.colors.agent } else { t.colors.dead };
        dot(painter.clone(), at, t.viz.agent_radius, color);
        heading_arrow(
            painter.clone(),
            at,
            self.sim.world.heading,
            t.viz.agent_radius,
            color,
        );
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
                let Some(trace) = self.sim.blood.trace() else {
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

    /// Графики химий и моторной активности.
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
                if self.sim.blood.organ.damage >= 1.0 {
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
                ui.vertical(|ui| {
                    ui.heading("brain");
                    self.brain_graph(ui);
                    ui.add_space(self.theme.spacing.gap);
                    ui.heading("blood");
                    self.plots(ui);
                    self.trace_panel(ui);
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
        Box::new(|_cc| Ok(Box::new(Observatory::new(DEFAULT_SEED)))),
    )
}

const DEFAULT_SEED: u64 = 0xA_E11;
const TRACE_CAP: usize = 256;
const TRACE_ROWS: usize = 12;
const TRACE_PANEL_HEIGHT_FRAC: f32 = 0.5;
