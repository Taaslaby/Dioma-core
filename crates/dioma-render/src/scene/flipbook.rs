//! Флипбук-анимация: последовательность кадров (UV-прямоугольников в атласе),
//! проигрываемая по времени. Кадры — это НОРМАЛИЗОВАННЫЕ UV-прямоугольники
//! `[u0, v0, u1, v1]` в атласе; приложение продвигает время и каждый кадр
//! вызывает [`Sprite::with_frame`] перед push в батч.

/// Флипбук: список UV-кадров + скорость проигрывания.
///
/// * `frames` — UV-прямоугольники кадров в атласе (порядок проигрывания);
/// * `fps` — кадров в секунду;
/// * `looping` — зацикленность: `true` — кадры повторяются по кругу,
///   `false` — после последнего кадра анимация замирает на нём.
#[derive(Clone, Debug, PartialEq)]
pub struct Flipbook {
    /// UV-прямоугольники кадров `[u0, v0, u1, v1]` в атласе.
    pub frames: Vec<[f32; 4]>,
    /// Скорость проигрывания, кадров в секунду.
    pub fps: f32,
    /// Зацикленность анимации.
    pub looping: bool,
}

impl Flipbook {
    /// Флипбук из готового списка UV-прямоугольников.
    pub fn new(frames: Vec<[f32; 4]>, fps: f32, looping: bool) -> Flipbook {
        Flipbook { frames, fps, looping }
    }

    /// Разбивает прямоугольник `sheet_uv` в атласе на сетку `cols`×`rows`
    /// одинаковых ячеек (ряд-мажор: слева-направо, сверху-вниз) и берёт
    /// поддиапазон `start..start + count` как кадры.
    ///
    /// Например, лист 2×2 из региона `[0, 0, 0.5, 0.5]` с `start = 0`,
    /// `count = 4` даст кадры в порядке: верхний-левый, верхний-правый,
    /// нижний-левый, нижний-правый.
    ///
    /// `start`/`count` зажимаются к границам сетки (если `count` выходит за
    /// край, берётся сколько есть; при `start >= размер сетки` — пусто).
    pub fn from_grid(
        sheet_uv: [f32; 4],
        cols: u32,
        rows: u32,
        start: usize,
        count: usize,
        fps: f32,
        looping: bool,
    ) -> Flipbook {
        let cols = cols.max(1);
        let rows = rows.max(1);
        let total = (cols * rows) as usize;
        let start = start.min(total);
        let count = count.min(total - start);
        let cell_w = (sheet_uv[2] - sheet_uv[0]) / cols as f32;
        let cell_h = (sheet_uv[3] - sheet_uv[1]) / rows as f32;
        let mut frames = Vec::with_capacity(count);
        for i in start..start + count {
            let col = i % cols as usize;
            let row = i / cols as usize;
            frames.push([
                sheet_uv[0] + col as f32 * cell_w,
                sheet_uv[1] + row as f32 * cell_h,
                sheet_uv[0] + (col + 1) as f32 * cell_w,
                sheet_uv[1] + (row + 1) as f32 * cell_h,
            ]);
        }
        Flipbook { frames, fps, looping }
    }

    /// UV-прямоугольник кадра в момент `time` (секунды).
    ///
    /// Зацикленный: `(time·fps) mod len`; незацикленный: клампится к
    /// последнему кадру. Пустой флипбук (0 кадров) возвращает `[0, 0, 1, 1]` —
    /// весь атлас (спрайт просто нарисуется целиком).
    pub fn frame_at(&self, time: f32) -> [f32; 4] {
        if self.frames.is_empty() {
            return [0.0, 0.0, 1.0, 1.0];
        }
        let idx = (time * self.fps) as usize;
        if self.looping {
            self.frames[idx % self.frames.len()]
        } else {
            self.frames[idx.min(self.frames.len() - 1)]
        }
    }

    /// Количество кадров.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Пуст ли флипбук (0 кадров).
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}
