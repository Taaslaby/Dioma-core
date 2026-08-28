//! Nine-patch: растяжимая скруглённая панель из одного региона атласа.
//!
//! Процедурная текстура 48×48 (скруглённый прямоугольник с рамкой) делится
//! на 3×3 ячейки: углы сохраняют размер (`margin` px), кромки тянутся в одну
//! сторону, центр — в обе. Рисуется девятью [`Sprite`] с под-UV региона.
//!
//! Панель симметрична по вертикали, поэтому флип по Y (конвенция UV
//! спрайтового шейдера: v0 внизу квада) на картинку не влияет.

use crate::scene::sprite::Sprite;
use crate::ui::layout::Rect;

/// Сторона процедурной текстуры панели (px) — генерируется один раз.
pub const PATCH_SIZE: u32 = 48;
/// Радиус скругления в текстуре (px).
pub const PATCH_RADIUS: u32 = 10;
/// Размер угловой ячейки nine-patch (px) — >= радиуса + толщина рамки.
pub const MARGIN: f32 = 12.0;

/// Растеризует панель: скруглённый прямоугольник с рамкой и полупрозрачной
/// заливкой. Coverage — по расстоянию до скруглённого контура (1 сабсэмплинг
/// по 4 точкам), формат RGBA8.
pub fn panel_rgba(border: [u8; 4], fill: [u8; 4]) -> (u32, u32, Vec<u8>) {
    let s = PATCH_SIZE;
    let r = PATCH_RADIUS as f32;
    let b = 2.0; // толщина рамки (px)
    let mut px = vec![0u8; (s * s * 4) as usize];

    // SDF скруглённого прямоугольника: расстояние до границы контура
    // [b, s-1-b] с радиусом r. <=0 — внутри; 0..1 — край (сглаживание).
    let inner = |p: (f32, f32)| -> f32 {
        let (x, y) = p;
        let (hx, hy) = ((s as f32 - 1.0 - b) / 2.0, (s as f32 - 1.0 - b) / 2.0);
        let (cx, cy) = (x - (b + hx), y - (b + hy));
        let qx = (cx.abs() - (hx - r)).max(0.0);
        let qy = (cy.abs() - (hy - r)).max(0.0);
        hx - r - (qx.hypot(qy)) // >0 внутри
    };

    for y in 0..s {
        for x in 0..s {
            // 4 сабсэмплинга на пиксель — мягкий край без полноценного AA.
            let mut cover = 0.0;
            for (dx, dy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                let d = inner((x as f32 + dx, y as f32 + dy));
                cover += d.clamp(0.0, 1.0);
            }
            let cover = cover / 4.0;
            if cover <= 0.0 {
                continue;
            }
            // Рамка: полоса контура толщиной b внутри контура.
            let border_mix = if inner((x as f32 + 0.5, y as f32 + 0.5)) > b { 0.0 } else { 1.0 };
            let mix = |a: u8, c: u8| -> u8 { (a as f32 + (c as f32 - a as f32) * border_mix) as u8 };
            let i = ((y * s + x) * 4) as usize;
            px[i] = mix(fill[0], border[0]);
            px[i + 1] = mix(fill[1], border[1]);
            px[i + 2] = mix(fill[2], border[2]);
            px[i + 3] = ((fill[3] as f32 + (border[3] as f32 - fill[3] as f32) * border_mix) * cover) as u8;
        }
    }
    (s, s, px)
}

/// Чистый белый прямоугольник 8×8 (без скругления): для заливок, которые
/// растягиваются по обеим осям (прогресс-бары) — искажений не бывает.
pub fn rect_rgba() -> (u32, u32, Vec<u8>) {
    let s = 8u32;
    let px = vec![255u8; (s * s * 4) as usize];
    // Край 1px оставляем мягким, чтобы без рамки не было лесенки при растяжении.
    (s, s, px)
}

/// Регион панели в атласе + правило разрезания на 9 ячеек.
#[derive(Clone, Copy, Debug)]
pub struct NinePatch {
    /// UV-регион всей текстуры панели `[u0, v0, u1, v1]`.
    pub uv: [f32; 4],
    /// Сторона текстуры в px (для перевода margin в UV).
    pub texture_px: f32,
    /// Размер угловой ячейки (px текстуры).
    pub margin: f32,
}

impl NinePatch {
    pub fn new(uv: [f32; 4], texture_px: f32, margin: f32) -> Self {
        Self { uv, texture_px, margin }
    }

    /// Девять спрайтов панели в прямоугольнике `rect` (экранные px, Y вниз).
    ///
    /// Порядок: по строкам сверху вниз. Углы — `margin` px, кромки тянутся,
    /// центр занимает остаток.
    pub fn sprites(&self, rect: Rect, viewport: [f32; 2], tint: [f32; 4], layer: f32) -> [Sprite; 9] {
        let m = self.margin.min(rect.w * 0.5).min(rect.h * 0.5);
        let tex = self.texture_px;
        let [u0, v0, u1, v1] = self.uv;
        let du = (u1 - u0) / tex;
        let dv = (v1 - v0) / tex;

        // Горизонтальные полосы (x, w) и вертикальные (y, h) в px назначения.
        let cols = [
            (rect.x, m),
            (rect.x + m, rect.w - 2.0 * m),
            (rect.x + rect.w - m, m),
        ];
        let rows = [
            (rect.y, m),
            (rect.y + m, rect.h - 2.0 * m),
            (rect.y + rect.h - m, m),
        ];
        // Источник в texel-ах: углы m×m, кромки — interior текстуры.
        let src_cols = [
            (0.0, self.margin),
            (self.margin, tex - 2.0 * self.margin),
            (tex - self.margin, self.margin),
        ];
        let src_rows = src_cols;

        let mut out = [Sprite::new([0.0, 0.0], [1.0, 1.0], self.uv); 9];
        for (row, &(ry, rh)) in rows.iter().enumerate() {
            for (col, &(cx, cw)) in cols.iter().enumerate() {
                let dst = Rect::new(cx, ry, cw, rh);
                let (sx, sw) = src_cols[col];
                let (sy, sh) = src_rows[row];
                let uv = [
                    u0 + sx * du,
                    v0 + sy * dv,
                    u0 + (sx + sw) * du,
                    v0 + (sy + sh) * dv,
                ];
                let mut s = Sprite::new(dst.world_center(viewport), [dst.w, dst.h], uv);
                s.tint = tint;
                s.layer = layer;
                s.depth = -(rect.y + rh) as f32; // y-сортировка внутри панели
                out[row * 3 + col] = s;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patch() -> NinePatch {
        NinePatch::new([0.1, 0.2, 0.7, 0.8], PATCH_SIZE as f32, MARGIN)
    }

    #[test]
    fn nine_sprites_cover_rect_exactly() {
        let p = patch();
        let rect = Rect::new(100.0, 50.0, 300.0, 200.0);
        let vp = [640.0, 480.0];
        let sp = p.sprites(rect, vp, [1.0; 4], 10.0);

        assert_eq!(sp.len(), 9);
        // Сумма площадей ячеек = площадь прямоугольника (без перекрытий).
        let area: f32 = sp.iter().map(|s| s.size[0] * s.size[1]).sum();
        assert!((area - rect.w * rect.h).abs() < 1e-3, "{area} vs {}", rect.w * rect.h);

        // Углы — margin×margin, центры на месте.
        assert_eq!(sp[0].size, [MARGIN, MARGIN]);
        assert_eq!(sp[4].size, [rect.w - 2.0 * MARGIN, rect.h - 2.0 * MARGIN]);
        assert_eq!(sp[8].size, [MARGIN, MARGIN]);
    }

    #[test]
    fn uv_subrects_partition_region() {
        let p = patch();
        let sp = p.sprites(Rect::new(0.0, 0.0, 96.0, 96.0), [100.0, 100.0], [1.0; 4], 0.0);
        // Левая колонка u0, правая — u1.
        assert!((sp[0].uv_rect[0] - p.uv[0]).abs() < 1e-6);
        assert!((sp[2].uv_rect[2] - p.uv[2]).abs() < 1e-6);
        // Верхняя строка берёт v-полосу верхней кромки текстуры, нижняя — нижней.
        assert!(sp[0].uv_rect[1] < sp[6].uv_rect[1]);
        assert!((sp[8].uv_rect[3] - p.uv[3]).abs() < 1e-6);
        // Центральная ячейка — interior текстуры.
        let c = sp[4].uv_rect;
        assert!(c[2] - c[0] < p.uv[2] - p.uv[0]);
    }

    #[test]
    fn panel_rgba_center_is_fill_corner_is_transparent() {
        let (w, h, px) = panel_rgba([255, 0, 0, 255], [10, 20, 30, 200]);
        assert_eq!((w, h), (PATCH_SIZE, PATCH_SIZE));
        let at = |x: u32, y: u32| -> [u8; 4] {
            let i = ((y * w + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };
        assert_eq!(at(0, 0), [0, 0, 0, 0], "угол вне скругления прозрачен");
        assert_eq!(at(24, 24)[0], 10, "центр — цвет заливки");
        assert_eq!(at(1, 24)[0], 255, "левая кромка — цвет рамки");
    }
}
