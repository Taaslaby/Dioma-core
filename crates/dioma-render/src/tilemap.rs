//! Тайлмап: сетка u16-тайлов + рендер через обычный [`SpriteBatch`]
//! (один квад на непустой тайл — никакого нового пайплайна).
//!
//! Конвенция индексов тайлов: **0 = пустая клетка** (не рисуется, экономит
//! инстансы), тайлы нумеруются с 1 по `cols·rows` и раскладываются по сетке
//! [`TilesetUv`] слева-направо, сверху-вниз — так же, как кадры у
//! [`crate::Flipbook::from_grid`].

use crate::batch::SpriteBatch;
use crate::sprite::Sprite;

/// Слой тайлмапа в батче. Игра рисует существ и объекты выше (layer > 0.0),
/// либо использует глубину: тайлы получают `depth = y` (ряд тайлов).
pub const LAYER_TILE: f32 = 0.0;

/// UV-сетка одинаковых тайлов внутри одного региона атласа.
///
/// Регион `[u0, v0, u1, v1]` делится на `cols`×`rows` равных ячеек (ряд-мажор:
/// слева-направо, сверху-вниз). **Конвенция индексов:** `0` — пустая клетка
/// (не рисуется; [`TilesetUv::cell_uv`] для неё возвращает весь атлас
/// `[0, 0, 1, 1]`), реальные тайлы — `1..=cols·rows`. Индекс вне сетки (кроме
/// 0) тоже возвращает `[0, 0, 1, 1]` — защитный fallback, приложение обязано
/// держать индексы в границах.
///
/// Индексы — `u16`, поэтому `cols·rows` не должен превышать 65535
/// ([`TilesetUv::new`] проверяет и паникует).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TilesetUv {
    /// UV-прямоугольник региона в атласе: `[u0, v0, u1, v1]`.
    pub region: [f32; 4],
    /// Количество столбцов сетки.
    pub cols: u32,
    /// Количество строк сетки.
    pub rows: u32,
}

impl TilesetUv {
    /// Сетка `cols`×`rows` внутри региона `region`.
    ///
    /// Паникует (русское сообщение), если `cols·rows > 65535` — индексы тайлов
    /// `u16` просто не влезут. `cols`/`rows` = 0 допустимы (пустой тайлсет:
    /// `cell_uv` вернёт весь атлас), но на практике не используются.
    pub fn new(region: [f32; 4], cols: u32, rows: u32) -> Self {
        assert!(
            (cols as u64) * (rows as u64) <= u16::MAX as u64,
            "TilesetUv::new: сетка {cols}×{rows} = {} ячеек больше 65535 — индексы тайлов u16 не влезут",
            (cols as u64) * (rows as u64)
        );
        Self {
            region,
            cols: cols.max(1),
            rows: rows.max(1),
        }
    }

    /// UV-прямоугольник ячейки `index` в атласе.
    ///
    /// `0` — пустая клетка (весь атлас `[0, 0, 1, 1]`, батч её пропускает);
    /// `1..=cell_count()` — ячейки сетки слева-направо, сверху-вниз
    /// (1 — верхний левый угол региона). Индекс вне сетки — fallback
    /// `[0, 0, 1, 1]` (см. доки [`TilesetUv`]).
    pub fn cell_uv(&self, index: u16) -> [f32; 4] {
        if index == 0 {
            return [0.0, 0.0, 1.0, 1.0];
        }
        let n = index as u32 - 1;
        let total = self.cols * self.rows;
        if n >= total {
            return [0.0, 0.0, 1.0, 1.0];
        }
        let col = n % self.cols;
        let row = n / self.cols;
        let cell_w = (self.region[2] - self.region[0]) / self.cols as f32;
        let cell_h = (self.region[3] - self.region[1]) / self.rows as f32;
        [
            self.region[0] + col as f32 * cell_w,
            self.region[1] + row as f32 * cell_h,
            self.region[0] + (col + 1) as f32 * cell_w,
            self.region[1] + (row + 1) as f32 * cell_h,
        ]
    }

    /// Количество ячеек в сетке (`cols·rows`).
    pub fn cell_count(&self) -> u32 {
        self.cols * self.rows
    }
}

/// Прямоугольная сетка u16-тайлов (row-major, `tiles.len() == cols·rows`).
///
/// `0` — пустая клетка (не рисуется); `1..` — индексы ячеек [`TilesetUv`].
/// Координаты — столбец/строка от верхнего ряда данных; мировая позиция
/// клетки — `(x·tile_size, y·tile_size)` (левый верхний угол ряда в данных).
/// Мир — в пикселях, камера держит ось Y вверх, поэтому ряд данных 0
/// (наименьший мировой Y) рендерится в НИЖНЕЙ части экрана, а последний ряд —
/// в верхней.
#[derive(Clone, Debug, PartialEq)]
pub struct Tilemap {
    /// Ширина сетки в тайлах.
    pub cols: u32,
    /// Высота сетки в тайлах.
    pub rows: u32,
    /// Размер стороны тайла в мировых пикселях.
    pub tile_size: f32,
    /// Row-major данные: `tiles[y·cols + x]`, 0 = пустая клетка.
    tiles: Vec<u16>,
}

impl Tilemap {
    /// Пустая карта `cols`×`rows` (все клетки = 0).
    pub fn new(cols: u32, rows: u32, tile_size: f32) -> Self {
        Self {
            cols,
            rows,
            tile_size,
            tiles: vec![0; (cols * rows) as usize],
        }
    }

    /// Карта из строк одинаковой длины; `cols` берётся из первой строки.
    ///
    /// Паникует (русское сообщение), если длина какой-то строки отличается
    /// от первой. Пустой список строк — пустая карта 0×0.
    pub fn from_rows(rows_data: &[&[u16]], tile_size: f32) -> Self {
        let rows = rows_data.len() as u32;
        let cols = rows_data.first().map_or(0, |r| r.len() as u32);
        for (i, r) in rows_data.iter().enumerate() {
            assert!(
                r.len() as u32 == cols,
                "Tilemap::from_rows: строка {} имеет длину {}, а первая — {} \
                 (все строки должны быть одинаковой длины)",
                i + 1,
                r.len(),
                cols
            );
        }
        let mut tiles = Vec::with_capacity((cols * rows) as usize);
        for r in rows_data {
            tiles.extend_from_slice(r);
        }
        Self {
            cols,
            rows,
            tile_size,
            tiles,
        }
    }

    /// Ставит тайл в клетку `(x, y)`. Паникует (русское сообщение) при выходе
    /// за границы карты.
    pub fn set(&mut self, x: u32, y: u32, tile: u16) {
        assert!(
            x < self.cols && y < self.rows,
            "Tilemap::set: клетка ({x}, {y}) за границами карты {}×{}",
            self.cols,
            self.rows
        );
        self.tiles[(y * self.cols + x) as usize] = tile;
    }

    /// Тайл в клетке `(x, y)`. Паникует (русское сообщение) при выходе за
    /// границы карты.
    pub fn get(&self, x: u32, y: u32) -> u16 {
        assert!(
            x < self.cols && y < self.rows,
            "Tilemap::get: клетка ({x}, {y}) за границами карты {}×{}",
            self.cols,
            self.rows
        );
        self.tiles[(y * self.cols + x) as usize]
    }

    /// Заливает прямоугольник `(x0, y0)..=(x1, y1)` тайлом `tile` — границы
    /// инклюзивны и зажимаются к размерам карты (пересечение не выходит за
    /// пределы). Пустая карта — no-op.
    pub fn fill_rect(&mut self, x0: u32, y0: u32, x1: u32, y1: u32, tile: u16) {
        if self.cols == 0 || self.rows == 0 {
            return;
        }
        let x0 = x0.min(self.cols - 1);
        let y0 = y0.min(self.rows - 1);
        let x1 = x1.min(self.cols - 1);
        let y1 = y1.min(self.rows - 1);
        for y in y0..=y1 {
            for x in x0..=x1 {
                self.tiles[(y * self.cols + x) as usize] = tile;
            }
        }
    }

    /// Ширина сетки в тайлах.
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// Высота сетки в тайлах.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Размер стороны тайла в мировых пикселях.
    pub fn tile_size(&self) -> f32 {
        self.tile_size
    }

    /// Размер карты в мировых пикселях: `[cols·tile_size, rows·tile_size]`.
    pub fn world_size(&self) -> [f32; 2] {
        [
            self.cols as f32 * self.tile_size,
            self.rows as f32 * self.tile_size,
        ]
    }
}

/// Готовый к рендеру тайлмап: непустые тайлы превращены в спрайты
/// (один квад на тайл) внутри обычного [`SpriteBatch`].
///
/// Контракт рендера: никакого нового пайплайна не нужно — тайлмап рисуется
/// базовым спрайтовым проходом, передавая внутренний батч напрямую:
///
/// ```no_run
/// # let mut renderer: dioma_render::Renderer = unimplemented!();
/// # let queue: &wgpu::Queue = unimplemented!();
/// # let mut encoder: wgpu::CommandEncoder = unimplemented!();
/// # let view: &wgpu::TextureView = unimplemented!();
/// # let atlas: &dioma_render::SpriteAtlas = unimplemented!();
/// # let camera: &dioma_render::Camera = unimplemented!();
/// # let tile_batch: &dioma_render::TileBatch = unimplemented!();
/// renderer.render(&queue, &mut encoder, view, atlas, tile_batch.sprite_batch(), camera);
/// ```
///
/// Квад тайла: центр в `(x·tile_size + tile_size/2, y·tile_size + tile_size/2)`,
/// размер `tile_size`×`tile_size`, UV = `tileset.cell_uv(tile)`, поворот 0,
/// белый тинт, слой [`LAYER_TILE`] (0.0), глубина `y` — ряды сортируются
/// естественно, а существа на слое 1.0+ рисуются поверх.
///
/// Вместимость батча = `cols·rows` (максимум непустых тайлов), поэтому
/// `push` в [`SpriteBatch`] не паникует. Для больших карт используйте
/// [`Self::upload_visible`] — на GPU уйдут только тайлы в кадре.
pub struct TileBatch {
    sprites: SpriteBatch,
    /// Размер стороны тайла в мировых пикселях (для расширения прямоугольника
    /// отсечения на полтайла в [`Self::upload_visible`]).
    tile_size: f32,
    /// Флаг перестройки: после [`Self::rebuild`] нужен новый upload.
    dirty: bool,
}

impl TileBatch {
    /// Строит батч из карты и тайлсета (все непустые тайлы → спрайты).
    /// Для изменения карты перестройте батч через [`Self::rebuild`].
    pub fn new(device: &wgpu::Device, tilemap: &Tilemap, tileset: &TilesetUv) -> Self {
        let mut b = Self {
            sprites: SpriteBatch::new(device, 1),
            tile_size: tilemap.tile_size,
            dirty: true,
        };
        b.rebuild(device, tilemap, tileset);
        b
    }

    /// Перестраивает батч по новой карте/тайлсету — вызывайте, когда карта
    /// изменилась (старые спрайты отбрасываются, инстанс-буфер
    /// пересоздаётся с вместимостью `cols·rows`).
    pub fn rebuild(&mut self, device: &wgpu::Device, tilemap: &Tilemap, tileset: &TilesetUv) {
        let capacity = ((tilemap.cols * tilemap.rows) as usize).max(1);
        self.tile_size = tilemap.tile_size;
        self.sprites = SpriteBatch::new(device, capacity);
        let size = [tilemap.tile_size, tilemap.tile_size];
        for y in 0..tilemap.rows {
            for x in 0..tilemap.cols {
                let tile = tilemap.get(x, y);
                if tile == 0 {
                    continue; // пустая клетка: без инстанса
                }
                let mut s = Sprite::new(
                    [
                        x as f32 * tilemap.tile_size + tilemap.tile_size * 0.5,
                        y as f32 * tilemap.tile_size + tilemap.tile_size * 0.5,
                    ],
                    size,
                    tileset.cell_uv(tile),
                );
                s.layer = LAYER_TILE;
                s.depth = y as f32;
                self.sprites.push(s);
            }
        }
        self.dirty = true;
    }

    /// Заливает инстансы на GPU ([`SpriteBatch::upload`]); после
    /// [`Self::rebuild`] это происходит ровно один раз — тайлы между
    /// перестройками не меняются.
    pub fn upload(&mut self, queue: &wgpu::Queue) {
        if self.dirty {
            self.sprites.upload(queue);
            self.dirty = false;
        }
    }

    /// Заливает на GPU только тайлы, пересекающие `view_rect`
    /// (`[min_x, min_y, max_x, max_y]` в мировых пикселях; границы включены).
    ///
    /// Прямоугольник расширяется на полтайла по каждой стороне, чтобы
    /// краевые тайлы, лишь касающиеся экрана, не пропали. Готовый прямоугольник
    /// экрана даёт [`crate::Camera::visible_world_rect`]. Вызывайте каждый кадр:
    /// смена прямоугольника сама помечает батч к перезаливке; если камера
    /// не двигалась — заливки не будет.
    pub fn upload_visible(&mut self, queue: &wgpu::Queue, view_rect: [f32; 4]) {
        let half = self.tile_size * 0.5;
        self.sprites.set_cull_rect(Some([
            view_rect[0] - half,
            view_rect[1] - half,
            view_rect[2] + half,
            view_rect[3] + half,
        ]));
        self.sprites.upload(queue);
    }

    /// Количество тайлов, уходящих на GPU после последнего
    /// [`Self::upload_visible`] (или всех непустых — после обычного
    /// [`Self::upload`]).
    pub fn len_visible(&self) -> usize {
        self.sprites.len_drawn()
    }

    /// Внутренний батч для [`crate::Renderer::render`]
    /// (`renderer.render(encoder, queue, view, atlas, tile_batch.sprite_batch(), camera)`).
    pub fn sprite_batch(&self) -> &SpriteBatch {
        &self.sprites
    }

    /// Количество видимых (непустых) тайлов.
    pub fn len(&self) -> usize {
        self.sprites.len()
    }
}
