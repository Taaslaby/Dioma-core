//! -------------------------------------------------------------------------
//! Filename:    src/ui/mod.rs
//! Class:       VizSuite
//! Purpose:     observatory UI helpers
//! Description: theme tokens + canvas painters (world view, brain graph)
//!              shared by the observatory bin and future game shell.
//!
//! author: Atlasov anton
//! -------------------------------------------------------------------------

pub mod theme;

use egui::{Color32, Painter, Pos2, Rect, Stroke, Vec2};

/// Преобразование координат мира (±half, y вверх) в пиксели канваса
/// (y вниз, вписано в rect с отступом margin).
pub struct WorldTransform {
    pub scale: f32,
    pub origin: Pos2,
}

impl WorldTransform {
    pub fn fit(rect: Rect, half: f64, margin: f32) -> Self {
        let size = (rect.width() - 2.0 * margin)
            .max(1.0)
            .min(rect.height() - 2.0 * margin);
        let scale = size as f64 / (2.0 * half);
        Self {
            scale: scale as f32,
            origin: rect.center(),
        }
    }

    pub fn to_screen(&self, x: f64, y: f64) -> Pos2 {
        Pos2::new(
            self.origin.x + x as f32 * self.scale,
            self.origin.y - y as f32 * self.scale,
        )
    }

    /// Длина мирового вектора в пикселях.
    pub fn len(&self, v: f64) -> f32 {
        v as f32 * self.scale
    }
}

/// Смешение цвета с фоном по доле [0..1] (для «вспышек» активности).
pub fn mix(base: Color32, target: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    Color32::from_rgb(
        (base.r() as f32 + (target.r() as f32 - base.r() as f32) * t) as u8,
        (base.g() as f32 + (target.g() as f32 - base.g() as f32) * t) as u8,
        (base.b() as f32 + (target.b() as f32 - base.b() as f32) * t) as u8,
    )
}

/// Точка с радиусом.
pub fn dot(p: Painter, at: Pos2, radius: f32, color: Color32) {
    p.circle_filled(at, radius, color);
}

/// Стрелка курса агента (треугольник по направлению heading, рад).
pub fn heading_arrow(p: Painter, at: Pos2, heading: f64, size: f32, color: Color32) {
    let dir = Vec2::new(heading.cos() as f32, -heading.sin() as f32);
    let tip = at + dir * size * HEADING_ARROW_TIP;
    let left =
        at + Vec2::new(dir.y, -dir.x) * size * HEADING_ARROW_WING - dir * size * HEADING_ARROW_BACK;
    let right =
        at + Vec2::new(-dir.y, dir.x) * size * HEADING_ARROW_WING - dir * size * HEADING_ARROW_BACK;
    p.add(egui::Shape::convex_polygon(
        vec![tip, left, right],
        color,
        Stroke::NONE,
    ));
}

// Пропорции стрелки (доли от size) — токены модуля.
const LABEL_FONT_SIZE: f32 = 10.0;
const HEADING_ARROW_TIP: f32 = 1.6;
const HEADING_ARROW_WING: f32 = 0.7;
const HEADING_ARROW_BACK: f32 = 0.6;

/// Подпись мелким текстом.
pub fn label(p: Painter, at: Pos2, text: &str, color: Color32) {
    p.text(
        at,
        egui::Align2::CENTER_CENTER,
        text,
        egui::FontId::proportional(LABEL_FONT_SIZE),
        color,
    );
}
