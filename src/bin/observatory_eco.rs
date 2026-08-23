//! -------------------------------------------------------------------------
//! Filename:    src/bin/observatory_eco.rs
//! Class:       DebugSuite
//! Purpose:     экосистема в реальном времени — несколько существ · еда ·
//!              поколения
//! Description: canvas с существами и едой + панель демографии.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::ecosystem::{Ecosystem, ARENA_HALF};
use dioma::metabolism::chem;
use eframe::egui;
use egui::{Color32, Pos2, Stroke};

struct EcoObs {
    eco: Ecosystem,
    playing: bool,
    speed: usize,
    tick: usize,
}

const SPEEDS: [usize; 5] = [1, 5, 25, 100, 400];

const CREATURE_COLORS: [Color32; 8] = [
    Color32::from_rgb(255, 128, 128),
    Color32::from_rgb(128, 255, 128),
    Color32::from_rgb(128, 128, 255),
    Color32::from_rgb(255, 255, 128),
    Color32::from_rgb(255, 128, 255),
    Color32::from_rgb(128, 255, 255),
    Color32::from_rgb(255, 200, 128),
    Color32::from_rgb(200, 128, 255),
];

impl EcoObs {
    fn new() -> Self {
        Self {
            eco: Ecosystem::new(5, 0x0000_AE11, true),
            playing: true,
            speed: 2,
            tick: 0,
        }
    }
}

impl eframe::App for EcoObs {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ── Симуляция ──
        if self.playing {
            for _ in 0..SPEEDS[self.speed.min(SPEEDS.len()-1)] {
                self.eco.step_tick();
                self.tick += 1;
            }
            ctx.request_repaint();
        }

        // ── Top bar ──
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let txt = if self.playing { "⏸" } else { "▶" };
                if ui.button(txt).clicked() { self.playing = !self.playing; }
                ui.label(format!("×{}", SPEEDS[self.speed.min(SPEEDS.len()-1)]));
                if ui.button("+").clicked() { self.speed = (self.speed+1).min(4); }
                if ui.button("−").clicked() { self.speed = self.speed.saturating_sub(1); }
                if ui.button("↺").clicked() { self.eco = Ecosystem::new(5, 0x0000_AE11, true); self.tick = 0; }

                ui.separator();
                let alive = self.eco.alive_count();
                let gen = self.eco.creatures.iter().map(|c| c.generation).max().unwrap_or(0);
                let born = self.eco.births.len();
                let died = self.eco.deaths.len();
                ui.label(format!("t={} живых={} поколение={} 🐣{} 💀{}", 
                    self.tick, alive, gen, born, died));

                if let Some(c) = self.eco.creatures.first() {
                    ui.separator();
                    ui.label(format!("G={:.0} H={:.0} F={:.0}",
                        c.sim.blood.conc[chem::GLUCOSE],
                        c.sim.blood.conc[chem::HUNGER],
                        c.sim.blood.conc[chem::FATIGUE]));
                }
            });
        });

        // ── Right panel: демография ──
        egui::SidePanel::right("demo").exact_width(220.0).show(ctx, |ui| {
            ui.heading("Существа");
            for (i, c) in self.eco.creatures.iter().enumerate() {
                if !c.alive { continue; }
                let col = CREATURE_COLORS[i % CREATURE_COLORS.len()];
                ui.colored_label(col, format!(
                    "#{id} g{gen} meals:{meals} x={:.0} y={:.0}",
                    c.x, c.y,
                    id = c.id, gen = c.generation, meals = c.meals,
                ));
            }
            
            ui.add_space(8.0);
            ui.heading("Рождения");
            egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                for &(gen, id, ref parents) in self.eco.births.iter().rev().take(10) {
                    ui.label(format!("g{gen}: #{id} ← {}&{}", parents[0], parents[1]));
                }
            });

            ui.add_space(8.0);
            ui.heading("Смерти");
            egui::ScrollArea::vertical().max_height(100.0).show(ui, |ui| {
                for &(id, gen, tick) in self.eco.deaths.iter().rev().take(10) {
                    ui.label(format!("#{id} g{gen} умер t{tick}"));
                }
            });
        });

        // ── Central: мир ──
        egui::CentralPanel::default().show(ctx, |ui| {
            let painter = ui.painter_at(ui.available_rect_before_wrap());
            let rect = ui.available_rect_before_wrap();
            painter.rect_filled(rect, 0.0, Color32::from_rgb(15, 15, 20));

            // Масштаб: мир → экран.
            let margin = 20.0f32;
            let size = rect.width().min(rect.height()) - 2.0 * margin;
            let scale = size / ((ARENA_HALF * 2.0) as f32);
            let cx = rect.center().x;
            let cy = rect.center().y;
            let to_screen = |wx: f64, wy: f64| -> Pos2 {
                Pos2::new(
                    cx + (wx as f32) * scale,
                    cy - (wy as f32) * scale,
                )
            };

            // Сетка.
            for g in [-200.0, -100.0, 0.0, 100.0, 200.0] {
                painter.line_segment(
                    [to_screen(g, -ARENA_HALF), to_screen(g, ARENA_HALF)],
                    Stroke::new(0.5, Color32::from_rgb(30, 30, 40)),
                );
                painter.line_segment(
                    [to_screen(-ARENA_HALF, g), to_screen(ARENA_HALF, g)],
                    Stroke::new(0.5, Color32::from_rgb(30, 30, 40)),
                );
            }

            // Еда.
            for f in &self.eco.foods {
                if f.alive {
                    painter.circle_filled(to_screen(f.x, f.y), 3.0, Color32::from_rgb(80, 200, 80));
                } else {
                    painter.circle_stroke(to_screen(f.x, f.y), 2.0, Stroke::new(0.5, Color32::from_rgb(50, 80, 50)));
                }
            }

            // Существа.
            for (i, c) in self.eco.creatures.iter().enumerate() {
                if !c.alive { continue; }
                let col = CREATURE_COLORS[i % CREATURE_COLORS.len()];
                let at = to_screen(c.x, c.y);
                
                // Тело.
                painter.circle_filled(at, 6.0, col);
                
                // Направление.
                let dx = c.heading.cos() * 12.0;
                let dy = -c.heading.sin() * 12.0;
                painter.line_segment([at, Pos2::new(at.x + dx as f32, at.y + dy as f32)],
                    Stroke::new(1.5, col));
                
                // ID.
                painter.text(at + egui::vec2(0.0, -14.0), egui::Align2::CENTER_CENTER,
                    format!("#{}", c.id), egui::FontId::proportional(9.0), col);
            }
        });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1024.0, 768.0]),
        ..Default::default()
    };
    eframe::run_native(
        "dioma ecosystem",
        options,
        Box::new(|_cc| Ok(Box::new(EcoObs::new()))),
    )
}
