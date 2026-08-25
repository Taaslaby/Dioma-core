//! -------------------------------------------------------------------------
//! Filename:    src/bin/observatory_side.rs
//! Class:       DebugSuite
//! Purpose:     боковой мир в реальном времени — гравитация, плоды,
//!              существа ходят, прыгают, едят
//! Description: вид сбоку как у старых симов жизни. Клик — бросить плод
//!              (сладкий), Shift+клик — горький.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::world::side::sim::{SideWorld, N_FRUITS};
use eframe::egui;
use egui::{Color32, Pos2, Stroke};

struct SideObs {
    world: SideWorld,
    playing: bool,
    speed: usize,
}

const SPEEDS: [usize; 5] = [1, 4, 16, 64, 256];
const SKY: Color32 = Color32::from_rgb(24, 28, 44);
const GROUND: Color32 = Color32::from_rgb(52, 44, 36);

impl Default for SideObs {
    fn default() -> Self {
        Self { world: SideWorld::new(3, 2026_0826), playing: true, speed: 1 }
    }
}

impl eframe::App for SideObs {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.playing {
            for _ in 0..SPEEDS[self.speed.min(SPEEDS.len() - 1)] {
                self.world.step_tick();
            }
            ctx.request_repaint();
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let txt = if self.playing { "⏸" } else { "▶" };
                if ui.button(txt).clicked() { self.playing = !self.playing; }
                ui.label(format!("×{}", SPEEDS[self.speed.min(SPEEDS.len() - 1)]));
                if ui.button("+").clicked() { self.speed = (self.speed + 1).min(4); }
                if ui.button("−").clicked() { self.speed = self.speed.saturating_sub(1); }
                if ui.button("↺").clicked() {
                    self.world = SideWorld::new(3, 2026_0826);
                }
                ui.separator();
                let meals: usize = self.world.creatures.iter().map(|c| c.meals).sum();
                let poison: usize = self.world.creatures.iter().map(|c| c.bitter_eaten).sum();
                let ripe = self.world.fruits.iter().filter(|f| f.alive && f.ripen_until.is_none()).count();
                ui.label(format!(
                    "t={} · 🧍{} · 🍴{meals} · ☠{poison} · 🍎 на земле: {ripe}/{N_FRUITS}",
                    self.world.tick, self.world.alive_count()
                ));
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_rect_before_wrap().size(), egui::Sense::click());
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 0.0, SKY);

            // Масштаб: мир 0..520 × 0..370 → экран.
            const WORLD_W: f64 = 520.0;
            const WORLD_H: f64 = 370.0;
            let scale = (rect.width() / WORLD_W as f32).min(rect.height() / WORLD_H as f32);
            let off = egui::vec2(
                (rect.width() - WORLD_W as f32 * scale) * 0.5,
                rect.bottom() - WORLD_H as f32 * scale - 8.0,
            );
            let to_screen = |wx: f64, wy: f64| -> Pos2 {
                Pos2::new(
                    rect.left() + off.x + wx as f32 * scale,
                    rect.top() + off.y + wy as f32 * scale,
                )
            };
            let w2s = |r: egui::Rect| -> egui::Rect {
                egui::Rect::from_two_pos(
                    to_screen(r.left() as f64, r.top() as f64),
                    to_screen(r.right() as f64, r.bottom() as f64),
                )
            };

            // Комнаты.
            for room in &self.world.rooms {
                let rr = w2s(egui::Rect::from_min_max(
                    Pos2::new(room.x0 as f32, room.y_top as f32),
                    Pos2::new(room.x1 as f32, room.y_bot as f32),
                ));
                let fill = match room.id {
                    1 => Color32::from_rgb(34, 46, 34),   // крона
                    2 => Color32::from_rgb(30, 26, 38),   // пещера
                    _ => GROUND,
                };
                painter.rect_filled(rr, 2.0, fill);
                // Пол жирнее.
                painter.line_segment(
                    [rr.left_bottom(), rr.right_bottom()],
                    Stroke::new(3.0, Color32::from_rgb(90, 78, 60)),
                );
            }

            // Плоды: зреющие тусклые на ветках, спелые яркие.
            for f in &self.world.fruits {
                if !f.alive { continue; }
                let at = to_screen(f.body.x, f.body.y);
                let col = match f.taste {
                    dioma::world::ecosystem::Taste::Sweet => Color32::from_rgb(110, 220, 100),
                    dioma::world::ecosystem::Taste::Bitter => Color32::from_rgb(235, 140, 40),
                };
                let col = if f.ripen_until.is_some() { col.gamma_multiply(0.35) } else { col };
                painter.circle_filled(at, 4.0 * scale.max(0.7), col);
            }

            // Существа: капля с глазом, нос по направлению.
            for c in &self.world.creatures {
                let at = to_screen(c.body.x, c.body.y);
                let body_col = Color32::from_rgb(210, 190, 255);
                let r = 7.0 * scale.max(0.7);
                painter.circle_filled(at, r, body_col);
                // Глаз смотрит по facing.
                let eye = at + egui::vec2(c.body.facing as f32 * r * 0.45, -r * 0.25);
                painter.circle_filled(eye, r * 0.22, Color32::BLACK);
                // Спит: z-z над головой.
                if c.sleeping_ticks > 0 {
                    painter.text(at + egui::vec2(r, -r - 4.0), egui::Align2::LEFT_CENTER,
                        "z", egui::FontId::proportional(12.0),
                        Color32::from_rgb(150, 170, 220));
                }
                // Метка id/еды.
                painter.text(at + egui::vec2(0.0, -r - 10.0), egui::Align2::CENTER_CENTER,
                    format!("#{}·{}", c.id, c.meals),
                    egui::FontId::proportional(9.0),
                    Color32::from_rgb(170, 170, 190));
            }

            // Подсказка.
            painter.text(rect.left_bottom() + egui::vec2(8.0, -8.0),
                egui::Align2::LEFT_BOTTOM,
                "клик = спелый сладкий плод · Shift+клик = горький · плоды зреют на балконе сами",
                egui::FontId::proportional(11.0), Color32::from_rgb(95, 100, 115));

            // Взаимодействие.
            if response.clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let wx = ((pos.x - rect.left() - off.x) / scale) as f64;
                    let wy = ((pos.y - rect.top() - off.y) / scale) as f64;
                    let bitter = ctx.input(|i| i.modifiers.shift);
                    self.world.drop_fruit(wx, wy, bitter);
                }
            }
        });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        "dioma side world",
        options,
        Box::new(|_cc| Ok(Box::new(SideObs::default()))),
    )
}
