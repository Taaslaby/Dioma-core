//! UI-текст: TTF-шрифт (кириллица) → атлас глифов → спрайты.
//!
//! Шрифт вшит в крейт (`include_bytes!`, PT Sans, лицензия OFL — см.
//! `assets/fonts/OFL.txt`) и растеризуется [`fontdue`] в белые RGBA-глифы,
//! которые складываются в ОБЩИЙ атлас крейта (`SpriteAtlas::add_rgba`) —
//! поэтому UI рисуется одним спрайтовым батчем без второй текстуры.
//!
//! Глифы кэшируются ПО РАЗМЕРУ: ключ `(символ, px)`. При DPI-масштабе
//! (`UiSystem::set_scale`) текст запрашивается в фактическом размере —
//! глифы растеризуются заново и остаются резкими (без апскейла 32px).
//! Базовый набор (ASCII + кириллица А-я, Ёё) создаётся при старте для 32px;
//! [`UiFont::prepare`] прогревает набор под новый масштаб при смене DPI.

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

/// Метрики глифа в px при размере растеризации `px`.
#[derive(Clone, Debug)]
pub struct GlyphInfo {
    /// Имя региона в атласе (`ui/glyph/{px}/{U+XXXX}`); UV резолвится при
    /// пуше — атлас может расти и переупаковывать регионы.
    pub name: String,
    /// Размер растра (px).
    pub w: f32,
    pub h: f32,
    /// Отступ левого края от пера (px, вправо).
    pub left: f32,
    /// Верх растра НАД базовой линией (px; у «p» отрицательный — хвост ниже).
    pub above: f32,
    /// Ширина шага пера (px).
    pub advance: f32,
    /// Размер растеризации, для которого посчитаны метрики.
    pub px: u32,
}

/// Шрифт UI + кэш глифов (по символу И размеру).
pub struct UiFont {
    font: Font,
    glyphs: HashMap<(char, u32), GlyphInfo>,
    /// Подъём строки над базовой линией (px при [`BASE_PX`], >0).
    pub ascent: f32,
    /// Спуск строки (px при [`BASE_PX`], <0) — низ = базовая линия + descent.
    pub descent: f32,
}

/// Базовый набор глифов: печатный ASCII + кириллица А-я, Ёё + «№».
fn base_chars() -> Vec<char> {
    let mut chars: Vec<char> = (32u32..127).map(|c| char::from_u32(c).unwrap()).collect();
    chars.extend((0x410u32..=0x44F).map(|c| char::from_u32(c).unwrap()));
    chars.push('Ё');
    chars.push('ё');
    chars.push('№');
    chars
}

impl UiFont {
    /// Создаёт шрифт и растеризует базовый набор глифов (32px) в общий атлас.
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
        for c in base_chars() {
            this.ensure_glyph(queue, atlas, c, BASE_PX as u32);
        }
        this
    }

    /// Прогревает базовый набор под новый масштаб DPI (без хичча на кадре).
    pub fn prepare(&mut self, queue: &wgpu::Queue, atlas: &mut SpriteAtlas, scale: f32) {
        let px = (BASE_PX * scale).round().max(1.0) as u32;
        for c in base_chars() {
            self.ensure_glyph(queue, atlas, c, px);
        }
    }

    /// Глиф из кэша (по символу и размеру); при отсутствии — растеризует.
    pub fn glyph(&mut self, queue: &wgpu::Queue, atlas: &mut SpriteAtlas, c: char, px: u32) -> GlyphInfo {
        let key = (c, px);
        if !self.glyphs.contains_key(&key) {
            self.ensure_glyph(queue, atlas, c, px);
        }
        self.glyphs[&key].clone()
    }

    fn ensure_glyph(&mut self, queue: &wgpu::Queue, atlas: &mut SpriteAtlas, c: char, px: u32) {
        let (m, coverage) = self.font.rasterize(c, px as f32);
        // Пустой глиф (пробел): квада нет, шаг пера остаётся.
        if m.width == 0 || m.height == 0 {
            self.glyphs.insert((c, px), GlyphInfo {
                name: String::new(),
                w: 0.0,
                h: 0.0,
                left: 0.0,
                above: 0.0,
                advance: m.advance_width,
                px,
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
        let name = format!("ui/glyph/{px:03}/{:04X}", c as u32);
        if let Err(e) = atlas.add_rgba(queue, &name, m.width as u32, m.height as u32, &rgba) {
            // Атлас не вырос (без устройства) — текст без этого глифа, но UI
            // не падает. На практике не происходит: атлас умеет расти.
            eprintln!("[ui] глиф {c:?}@{px} не добавлен: {e}");
            return;
        }
        self.glyphs.insert(
            (c, px),
            GlyphInfo {
                name,
                w: m.width as f32,
                h: m.height as f32,
                left: m.xmin as f32,
                // Верх растра над базовой линией: fontdue — ось Y вверх,
                // ymin = низ растра относительно базовой линии.
                above: (m.ymin + m.height as i32) as f32,
                advance: m.advance_width,
                px,
            },
        );
    }

    /// Высота строки текста при масштабе `scale` (px).
    pub fn line_height(&self, scale: f32) -> f32 {
        (self.ascent - self.descent) * scale
    }

    /// Ширина текста по метрикам шрифта БЕЗ атласа (для лэйаута/тестов;
    /// не растеризует новые глифы в атлас).
    pub fn measure(&self, text: &str, scale: f32) -> f32 {
        let px = (BASE_PX * scale).round().max(1.0);
        text.chars()
            .map(|c| match self.glyphs.get(&(c, px as u32)) {
                Some(g) => g.advance,
                None => {
                    let (m, _) = self.font.rasterize(c, px);
                    m.advance_width
                }
            })
            .sum()
    }

    /// Перенос по словам: строки шириной ≤ `max_width` (px при `scale`).
    /// Слово длиннее строки остаётся целиком на своей строке (без разрыва).
    pub fn wrap(&self, text: &str, max_width: f32, scale: f32) -> Vec<String> {
        let space_w = self.measure(" ", scale);
        let mut lines: Vec<String> = Vec::new();
        let mut cur = String::new();
        let mut cur_w = 0.0f32;
        for para in text.split('\n') {
            for word in para.split(' ') {
                let w = self.measure(word, scale);
                let with_space = if cur.is_empty() { 0.0 } else { space_w };
                if !cur.is_empty() && cur_w + with_space + w > max_width {
                    lines.push(std::mem::take(&mut cur));
                    cur_w = 0.0;
                }
                if !cur.is_empty() {
                    cur.push(' ');
                    cur_w += with_space;
                }
                cur.push_str(word);
                cur_w += w;
            }
            // Абзац кончился (учитываем пустые строки из двух \n подряд).
            lines.push(std::mem::take(&mut cur));
            cur_w = 0.0;
        }
        lines
    }

    /// Ширина однострочного текста (px) при масштабе `scale` (с атласом).
    pub fn text_width(&mut self, queue: &wgpu::Queue, atlas: &mut SpriteAtlas, text: &str, scale: f32) -> f32 {
        let px = (BASE_PX * scale).round().max(1.0) as u32;
        text.chars()
            .map(|c| self.glyph(queue, atlas, c, px).advance)
            .sum()
    }

    /// Пушит текст спрайтами глифов; `at` — левый ВЕРХНИЙ угол блока
    /// (экранные px, Y вниз). Многострочность — по `\n`. Глифы растеризуются
    /// под фактический размер (`BASE_PX × scale`) — резкость на любом DPI.
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
        let px = (BASE_PX * scale).round().max(1.0) as u32;
        let line_h = self.line_height(scale);
        for (li, line) in text.split('\n').enumerate() {
            let mut pen_x = at[0];
            let baseline = at[1] + self.ascent * scale + li as f32 * line_h;
            for c in line.chars() {
                let g = self.glyph(queue, atlas, c, px);
                pen_x += g.left;
                if g.w > 0.0 && g.h > 0.0 {
                    // UV на момент отрисовки: атлас мог вырасти и переупаковать
                    // регионы, поэтому кэшировать UV глифов нельзя.
                    let uv = atlas.region(&g.name).unwrap_or([0.0; 4]);
                    // Верх растра на экране: базовая линия − «над базовой линией».
                    let top = baseline - g.above;
                    let center_screen = [pen_x + g.w * 0.5, top + g.h * 0.5];
                    let world = [
                        center_screen[0] - viewport[0] * 0.5,
                        viewport[1] * 0.5 - center_screen[1],
                    ];
                    let mut s = Sprite::new(world, [g.w, g.h], uv);
                    s.tint = color;
                    s.layer = LAYER_TEXT;
                    out.push(s);
                }
                pen_x += g.advance - g.left;
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
        assert!(
            p.ymin + i32::try_from(p.height).unwrap() < a.ymin + i32::try_from(a.height).unwrap(),
            "верх «p» ниже верха «A»"
        );
    }

    #[test]
    fn line_metrics_sane() {
        let font = Font::from_bytes(FONT_BYTES, Default::default()).unwrap();
        let line = font.horizontal_line_metrics(BASE_PX).unwrap();
        assert!(line.ascent > 0.0 && line.descent < 0.0);
        assert!(line.new_line_size > line.ascent);
    }

    fn test_font() -> UiFont {
        let font = Font::from_bytes(FONT_BYTES, Default::default()).unwrap();
        let line = font.horizontal_line_metrics(BASE_PX).unwrap();
        // Без атласа: measure/wrap работают по метрикам шрифта напрямую.
        UiFont {
            font,
            glyphs: HashMap::new(),
            ascent: line.ascent,
            descent: line.descent,
        }
    }

    #[test]
    fn wrap_breaks_by_words() {
        let f = test_font();
        let lines = f.wrap("Привет большой мир снова", 120.0, 0.5);
        assert!(lines.len() >= 2, "должно быть несколько строк: {lines:?}");
        for l in &lines {
            let w = f.measure(l, 0.5);
            assert!(w <= 120.0 + 0.5, "строка шире лимита: {l:?} = {w}");
        }
        // Слова не разорваны и не потеряны.
        let joined = lines.join(" ");
        assert_eq!(joined, "Привет большой мир снова", "{lines:?}");
    }

    #[test]
    fn wrap_respects_paragraph_breaks() {
        let f = test_font();
        let lines = f.wrap("один\n\nдва", 300.0, 0.5);
        assert_eq!(lines, vec!["один".to_string(), "".to_string(), "два".to_string()], "{lines:?}");
    }

    #[test]
    fn long_word_stays_whole() {
        let f = test_font();
        let lines = f.wrap("короткое ДлинноеСлово", 60.0, 0.5);
        assert!(lines.iter().any(|l| l == "ДлинноеСлово"), "длинное слово целиком: {lines:?}");
    }

    /// Метрики линейны по размеру: measure при 0.5 и 1.0 соотносятся как px.
    #[test]
    fn measure_scales_linearly() {
        let f = test_font();
        let w_half = f.measure("Тест", 0.5);
        let w_full = f.measure("Тест", 1.0);
        assert!(
            (w_full / w_half - 2.0).abs() < 0.02,
            "линейность: {w_half} vs {w_full}"
        );
    }
}
