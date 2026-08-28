//! UI-текст: TTF-шрифт (кириллица) → атлас глифов → спрайты.
//!
//! Шрифт вшит в крейт (`include_bytes!`, PT Sans, лицензия OFL — см.
//! `assets/fonts/OFL.txt`) и растеризуется [`fontdue`] в белые RGBA-глифы,
//! которые складываются в ОБЩИЙ атлас крейта (`SpriteAtlas::add_rgba`) —
//! поэтому UI рисуется одним спрайтовым батчем без второй текстуры.
//!
//! Базовый размер глифа — [`BASE_PX`] px; стили масштабируют его (дробные
//! масштабы дают лёгкое размягчение краёв — сглаживание растеризации).
//! Глифы базового набора (ASCII + кириллица А-я, Ёё) создаются при старте,
//! остальные — лениво при первом использовании.

use std::collections::HashMap;

use fontdue::Font;

use crate::scene::atlas::SpriteAtlas;
use crate::scene::sprite::Sprite;

/// Базовый размер растеризации глифа (px, высота эм-квадрата шрифта).
pub const BASE_PX: f32 = 32.0;
/// Шрифт UI: PT Sans Regular (ParaType, OFL) — кириллица «из коробки».
pub const FONT_BYTES: &[u8] = include_bytes!("../../assets/fonts/PTSans-Regular.ttf");

/// Слой спрайтов текста в UI-батче (см. `super::LAYER_*`).
pub const LAYER_TEXT: f32 = 1002.0;

/// Метрики глифа в px при [`BASE_PX`] (масштабируются стилем).
#[derive(Clone, Copy, Debug)]
pub struct GlyphInfo {
    /// UV-регион в атласе.
    pub uv: [f32; 4],
    /// Размер растра (px).
    pub w: f32,
    pub h: f32,
    /// Отступ левого края от пера (px, вправо).
    pub left: f32,
    /// Верх растра НАД базовой линией (px; у «p» отрицательный — хвост ниже).
    pub above: f32,
    /// Ширина шага пера (px).
    pub advance: f32,
}

/// Шрифт UI + кэш глифов в атласе.
pub struct UiFont {
    font: Font,
    glyphs: HashMap<char, GlyphInfo>,
    /// Подъём строки над базовой линией (px, >0).
    pub ascent: f32,
    /// Спуск строки (px, <0) — низ строки = базовая линия + descent.
    pub descent: f32,
}

impl UiFont {
    /// Создаёт шрифт и растеризует базовый набор глифов в общий атлас.
    pub fn new(_device: &wgpu::Device, queue: &wgpu::Queue, atlas: &mut SpriteAtlas) -> Self {
        let font = Font::from_bytes(FONT_BYTES, fontdue::FontSettings {
            collection_index: 0,
            ..Default::default()
        })
        .expect("вшитый шрифт PT Sans валиден");
        let line = font
            .horizontal_line_metrics(BASE_PX)
            .expect("PT Sans имеет горизонтальные метрики");

        let mut this = Self {
            font,
            glyphs: HashMap::new(),
            ascent: line.ascent,
            descent: line.descent,
        };
        // Базовый набор: печатный ASCII + кириллица А-я, Ёё, а также «№».
        let mut chars: Vec<char> = (32u32..127).map(|c| char::from_u32(c).unwrap()).collect();
        chars.extend((0x410u32..=0x44F).map(|c| char::from_u32(c).unwrap()));
        chars.push('Ё');
        chars.push('ё');
        chars.push('№');
        for c in chars {
            this.ensure_glyph(queue, atlas, c);
        }
        this
    }

    /// Глиф из кэша; при отсутствии — растеризует и добавляет в атлас.
    pub fn glyph(&mut self, queue: &wgpu::Queue, atlas: &mut SpriteAtlas, c: char) -> GlyphInfo {
        if !self.glyphs.contains_key(&c) {
            self.ensure_glyph(queue, atlas, c);
        }
        self.glyphs[&c]
    }

    fn ensure_glyph(&mut self, queue: &wgpu::Queue, atlas: &mut SpriteAtlas, c: char) {
        let (m, coverage) = self.font.rasterize(c, BASE_PX);
        // Пустой глиф (пробел): квада нет, шаг пера остаётся.
        if m.width == 0 || m.height == 0 {
            self.glyphs.insert(c, GlyphInfo {
                uv: [0.0; 4], w: 0.0, h: 0.0, left: 0.0, above: 0.0, advance: m.advance_width,
            });
            return;
        }
        let mut rgba = vec![0u8; (m.width * m.height * 4) as usize];
        for (i, &a) in coverage.iter().enumerate() {
            // Белый глиф: цвет задаёт tint спрайта, альфа — coverage.
            rgba[i * 4] = 255;
            rgba[i * 4 + 1] = 255;
            rgba[i * 4 + 2] = 255;
            rgba[i * 4 + 3] = a;
        }
        let name = format!("ui/glyph/{:04X}", c as u32);
        if let Err(e) = atlas.add_rgba(queue, &name, m.width.max(1) as u32, m.height.max(1) as u32, &rgba) {
            // Атлас не вырос (без устройства) — заменяем пустотой: текст без
            // этого глифа, но UI не падает. На практике не происходит:
            // атлас создан через SpriteAtlas::new и умеет расти.
            eprintln!("[ui] глиф {c:?} не добавлен: {e}");
            return;
        }
        let uv = atlas.region(&name).expect("регион только что добавлен");
        self.glyphs.insert(
            c,
            GlyphInfo {
                uv,
                w: m.width as f32,
                h: m.height as f32,
                left: m.xmin as f32,
                // Верх растра над базовой линией: fontdue — ось Y вверх,
                // ymin = низ растра относительно базовой линии.
                above: (m.ymin + m.height as i32) as f32,
                advance: m.advance_width,
            },
        );
    }

    /// Высота строки текста при масштабе `scale` (px).
    pub fn line_height(&self, scale: f32) -> f32 {
        (self.ascent - self.descent) * scale
    }

    /// Ширина однострочного текста (px) при масштабе `scale`.
    pub fn text_width(&mut self, queue: &wgpu::Queue, atlas: &mut SpriteAtlas, text: &str, scale: f32) -> f32 {
        text.chars()
            .map(|c| self.glyph(queue, atlas, c).advance * scale)
            .sum()
    }

    /// Пушит текст спрайтами глифов; `at` — левый ВЕРХНИЙ угол блока
    /// (экранные px, Y вниз). Многострочность — по `\n`, без переноса слов.
    #[allow(clippy::too_many_arguments)]
    pub fn push_text(
        &mut self,
        out: &mut Vec<Sprite>,
        queue: &wgpu::Queue,
        atlas: &mut SpriteAtlas,
        text: &str,
        at: [f32; 2],
        scale: f32,
        color: [f32; 4],
        viewport: [f32; 2],
    ) {
        let line_h = self.line_height(scale);
        for (li, line) in text.split('\n').enumerate() {
            let mut pen_x = at[0];
            let baseline = at[1] + self.ascent * scale + li as f32 * line_h;
            for c in line.chars() {
                let g = self.glyph(queue, atlas, c);
                pen_x += g.left * scale;
                if g.w > 0.0 && g.h > 0.0 {
                    // Верх растра на экране: базовая линия − «над базовой линией».
                    let top = baseline - g.above * scale;
                    let center_screen = [pen_x + g.w * scale * 0.5, top + g.h * scale * 0.5];
                    let world = [
                        center_screen[0] - viewport[0] * 0.5,
                        viewport[1] * 0.5 - center_screen[1],
                    ];
                    let mut s = Sprite::new(world, [g.w * scale, g.h * scale], g.uv);
                    s.tint = color;
                    s.layer = LAYER_TEXT;
                    out.push(s);
                }
                pen_x += g.advance * scale - g.left * scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// fontdue: ось Y вверх, ymin — низ растра относительно базовой линии.
    /// У «p» хвост ниже базовой линии (ymin < 0), у «A» низ на линии (≈0).
    #[test]
    fn fontdue_y_axis_points_up() {
        let font = Font::from_bytes(FONT_BYTES, Default::default()).unwrap();
        let (a, _) = font.rasterize('A', BASE_PX);
        let (p, _) = font.rasterize('p', BASE_PX);
        assert!(p.ymin < 0, "хвост «p» ниже базовой линии: ymin={}", p.ymin);
        assert!(a.ymin >= 0, "«A» стоит на базовой линии: ymin={}", a.ymin);
        assert!(p.ymin + i32::try_from(p.height).unwrap() < a.ymin + i32::try_from(a.height).unwrap(), "верх «p» ниже верха «A»");
    }

    #[test]
    fn line_metrics_sane() {
        let font = Font::from_bytes(FONT_BYTES, Default::default()).unwrap();
        let line = font.horizontal_line_metrics(BASE_PX).unwrap();
        assert!(line.ascent > 0.0 && line.descent < 0.0);
        assert!(line.new_line_size > line.ascent);
    }
}
