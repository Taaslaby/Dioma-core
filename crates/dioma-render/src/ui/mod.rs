//! Immediate-mode UI на спрайтах dioma-render (свой каркас, без egui).
//!
//! Виджеты рисуются **обычными спрайтами** (панели — nine-patch из
//! [`nine_patch`], текст — глифы TTF из общего атласа, см. [`text`]) в
//! отдельный [`SpriteBatch`] и рисуются одним проходом
//! [`Renderer::render_ui`](crate::Renderer::render_ui) поверх сцены:
//! identity-камера, экранные пиксели, Y вниз (см. [`layout`]).
//!
//! API — immediate-mode, egui-подобный: каждый кадр игра описывает UI заново,
//! интерактив возвращается через [`Response`]. Порядок кадра:
//!
//! ```text
//! let mut ui = ui_system.begin(&mut atlas, &queue, input, viewport);
//! ui.panel(Anchor::TopLeft, [280.0, 140.0], |p| {
//!     p.label("Существо #7");
//!     if p.button("Пауза").clicked { paused = !paused; }
//!     p.progress_bar("Голод", 0.42);
//! });
//! ui.upload(&queue);                       // собрать SpriteBatch
//! renderer.render_ui(&queue, &mut encoder, &view, &atlas, ui.batch(), viewport);
//! ```
//!
//! Ограничения каркаса (сознательно): нет скролла/клиппинга контента,
//! слайдеров и drag&drop; id виджетов — hash подписи + порядковый номер,
//! поэтому стабильны при стабильном порядке вызовов; перенос текста —
//! только по `\n`.

pub mod layout;
pub mod nine_patch;
pub mod text;

use crate::scene::atlas::SpriteAtlas;
use crate::scene::batch::SpriteBatch;
use crate::scene::sprite::Sprite;
pub use layout::{anchored_rect, Anchor, Rect};
pub use nine_patch::{panel_rgba, rect_rgba, NinePatch, MARGIN as PATCH_MARGIN, PATCH_RADIUS, PATCH_SIZE};
pub use text::{UiFont, BASE_PX};

/// Вместимость UI-батча (спрайтов): панели + глифы с запасом.
const UI_BATCH_CAPACITY: usize = 8192;

/// Слои UI-батча (выше любых мировых слоёв).
pub const LAYER_PANEL: f32 = 1000.0;
pub const LAYER_WIDGET: f32 = 1001.0;
/// (текст — внутри [`text::LAYER_TEXT`]).

/// Отступ содержимого от края панели (px).
const CONTENT_MARGIN: f32 = 12.0;
/// Отступ панелей от края экрана (px) в [`UiFrame::panel`].
pub const SCREEN_MARGIN: f32 = 16.0;
/// Высота кнопки (px).
const BUTTON_H: f32 = 36.0;
/// Высота прогресс-бара (px).
const BAR_H: f32 = 14.0;
/// Шаг вертикального стека содержимого (px).
const ROW_GAP: f32 = 10.0;
/// Базовый масштаб текста (от [`BASE_PX`]): глиф 16 px.
const TEXT_SCALE: f32 = 0.5;

/// Цвета каркаса (tint поверх процедурных текстур).
mod colors {
    /// Кнопка обычная / под курсором / нажатая (текстура светлая, tint темнит).
    pub const BTN: [f32; 4] = [0.86, 0.88, 0.92, 1.0];
    pub const BTN_HOVER: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    pub const BTN_PRESSED: [f32; 4] = [0.70, 0.73, 0.80, 1.0];
    /// Текст: тёплый белый.
    pub const TEXT: [f32; 4] = [0.95, 0.96, 0.92, 1.0];
    /// Текст на светлых виджетах (кнопка): тёмный.
    pub const BTN_TEXT: [f32; 4] = [0.12, 0.14, 0.18, 1.0];
    /// Бар: тёмная подложка, зелёная заливка.
    pub const BAR_BG: [f32; 4] = [0.16, 0.17, 0.20, 1.0];
    pub const BAR_FILL: [f32; 4] = [0.45, 0.85, 0.45, 1.0];
}

/// Ввод UI за кадр (из winit-событий; экранная система координат, Y вниз).
#[derive(Clone, Copy, Debug, Default)]
pub struct UiInput {
    /// Позиция курсора в экранных пикселях.
    pub cursor: [f32; 2],
    /// Кнопка мыши зажата (с предыдущего кадра).
    pub down: bool,
    /// Нажата В ЭТОМ кадре.
    pub just_pressed: bool,
    /// Отпущена В ЭТОМ кадре.
    pub just_released: bool,
}

/// Результат взаимодействия с виджетом за кадр.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Response {
    /// Курсор над виджетом в этом кадре.
    pub hovered: bool,
    /// Клик завершён на виджете (нажатие и отпускание внутри).
    pub clicked: bool,
}

/// Долгоживущие ресурсы UI: шрифт, процедурные регионы, батч и состояние
/// взаимодействия (press между кадрами).
pub struct UiSystem {
    font: UiFont,
    /// Nine-patch панели (тёмная подложка); UV обновляется каждый кадр
    /// в `begin` — атлас может расти и переупаковывать регионы.
    patch: NinePatch,
    /// Nine-patch кнопки (светлая, с рамкой).
    button_patch: NinePatch,
    /// Имя/UV плоского белого прямоугольника (бары).
    rect_name: &'static str,
    rect_uv: [f32; 4],
    batch: SpriteBatch,
    /// Курсор текущего кадра (hit-test).
    cursor: [f32; 2],
    /// id виджета, на котором началось нажатие (клик = нажать и отпустить
    /// на одном виджете).
    pressed: Option<u64>,
}

impl UiSystem {
    /// Регистрирует процедурные регионы UI в атласе и растеризует шрифт
    /// (базовый набор: ASCII + кириллица). Создавать один раз, до кадров.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, atlas: &mut SpriteAtlas) -> Self {
        // Панель: скругление с рамкой; заливка темнее, рамка светлее.
        let (w, h, px) = panel_rgba([188, 196, 214, 255], [22, 26, 38, 216]);
        atlas
            .add_rgba(queue, "ui/panel", w, h, &px)
            .expect("атлас растёт под регион ui/panel");
        // Кнопка: светлый nine-patch (рамка темнее заливки) — тинт меняет
        // состояния (обычная/hover/нажатая), геометрия углов не искажается.
        let (w, h, px) = panel_rgba([96, 106, 130, 255], [232, 238, 248, 255]);
        atlas
            .add_rgba(queue, "ui/button", w, h, &px)
            .expect("атлас растёт под регион ui/button");
        // Плоский белый ПРЯМОУГОЛЬНИК (без скругления) для баров: его можно
        // растягивать как угодно — искажений не видно.
        let (w, h, px) = rect_rgba();
        atlas
            .add_rgba(queue, "ui/rect", w, h, &px)
            .expect("атлас растёт под регион ui/rect");
        let uv = atlas.region("ui/panel").expect("регион только что добавлен");
        let patch = NinePatch::new(uv, PATCH_SIZE as f32, PATCH_MARGIN);
        let buv = atlas.region("ui/button").expect("регион только что добавлен");
        let button_patch = NinePatch::new(buv, PATCH_SIZE as f32, PATCH_MARGIN);
        let rect_uv = atlas.region("ui/rect").expect("регион только что добавлен");

        Self {
            font: UiFont::new(device, queue, atlas),
            patch,
            button_patch,
            rect_name: "ui/rect",
            rect_uv,
            batch: SpriteBatch::new(device, UI_BATCH_CAPACITY),
            cursor: [0.0, 0.0],
            pressed: None,
        }
    }

    /// Батч последнего кадра (для [`crate::Renderer::render_ui`] вне
    /// [`UiFrame`], например в тестах).
    pub fn batch(&self) -> &SpriteBatch {
        &self.batch
    }

    /// Начинает кадр UI. `atlas` и `queue` нужны для ленивой растеризации
    /// глифов (первое появление нового символа в тексте).
    pub fn begin<'a>(
        &'a mut self,
        atlas: &'a mut SpriteAtlas,
        queue: &'a wgpu::Queue,
        input: UiInput,
        viewport: [f32; 2],
    ) -> UiFrame<'a> {
        self.cursor = input.cursor;
        // Клик «нажать-отпустить на виджете»: pressed живёт до кадра ПОСЛЕ
        // отпускания (в кадре отпускания just_released ещё обрабатывается).
        if !input.down && !input.just_released {
            self.pressed = None;
        }
        // Свежие UV: добавление глифов могло вырастить атлас и переупаковать
        // регионы — кэшировать UV между кадрами нельзя.
        self.patch.uv = atlas.region("ui/panel").expect("регион ui/panel удалён из атласа");
        self.button_patch.uv =
            atlas.region("ui/button").expect("регион ui/button удалён из атласа");
        self.rect_uv = atlas.region(self.rect_name).expect("регион ui/rect удалён из атласа");
        UiFrame {
            ui: self,
            atlas,
            queue,
            viewport,
            input,
            sprites: Vec::with_capacity(256),
            next_id: 0,
        }
    }
}

/// Кадр UI: собирает спрайты виджетов, потом заливается в батч.
pub struct UiFrame<'a> {
    ui: &'a mut UiSystem,
    atlas: &'a mut SpriteAtlas,
    queue: &'a wgpu::Queue,
    viewport: [f32; 2],
    input: UiInput,
    sprites: Vec<Sprite>,
    next_id: u64,
}

impl<'a> UiFrame<'a> {
    /// Панель, приколоченная якорем `anchor` к краю экрана; содержимое —
    /// вертикальный стек внутри замыкания ([`Panel`]).
    pub fn panel(&mut self, anchor: Anchor, size: [f32; 2], content: impl FnOnce(&mut Panel)) {
        let rect = anchored_rect(anchor, size, self.viewport, SCREEN_MARGIN);
        let panel_sprites = self.ui.patch.sprites(rect, self.viewport, [1.0; 4], LAYER_PANEL);
        self.sprites.extend(panel_sprites);

        let mut cursor_y = rect.y + CONTENT_MARGIN;
        let mut panel = Panel {
            ui: &mut *self.ui,
            atlas: &mut *self.atlas,
            queue: self.queue,
            sprites: &mut self.sprites,
            viewport: self.viewport,
            input: self.input,
            content: rect.inner(CONTENT_MARGIN),
            cursor_y: &mut cursor_y,
            next_id: &mut self.next_id,
        };
        content(&mut panel);
    }

    /// Заливает собранные спрайты в батч (`Renderer::render_ui` рисует его).
    pub fn upload(&mut self, queue: &wgpu::Queue) {
        self.ui.batch.clear();
        for s in self.sprites.drain(..) {
            self.ui.batch.push(s);
        }
        self.ui.batch.upload(queue);
    }

    /// Батч для [`crate::Renderer::render_ui`] (после [`Self::upload`]).
    pub fn batch(&self) -> &SpriteBatch {
        &self.ui.batch
    }

    /// Спрайты кадра (для отладки/тестов).
    pub fn sprites(&self) -> &[Sprite] {
        &self.sprites
    }

    /// Идёт ли сборка спрайтов (для отладки/тестов).
    pub fn sprite_count(&self) -> usize {
        self.sprites.len()
    }
}

/// Вертикальный стек содержимого панели. Виджеты добавляются сверху вниз;
/// `cursor_y` — верх следующей строки.
pub struct Panel<'a> {
    ui: &'a mut UiSystem,
    atlas: &'a mut SpriteAtlas,
    queue: &'a wgpu::Queue,
    sprites: &'a mut Vec<Sprite>,
    viewport: [f32; 2],
    input: UiInput,
    /// Область содержимого (панель минус поля).
    content: Rect,
    /// Верх следующей строки (экранные px).
    cursor_y: &'a mut f32,
    next_id: &'a mut u64,
}

impl<'a> Panel<'a> {
    /// Уникальный id виджета: hash подписи + порядковый номер кадра.
    fn next_id(&mut self, label: &str) -> u64 {
        *self.next_id += 1;
        // FNV-1a по подписи, чтобы id не «съезжал» при изменении числа
        // виджетов выше по стеку (в пределах одного текста подписи).
        let mut h: u64 = 0xcbf29ce484222325 ^ (*self.next_id << 32);
        for b in label.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Строка текста слева, во всю ширину содержимого.
    pub fn label(&mut self, text: &str) {
        let h = self.ui.font.line_height(TEXT_SCALE);
        self.ui.font.push_text(
            self.sprites,
            self.queue,
            self.atlas,
            text,
            [self.content.x, *self.cursor_y],
            TEXT_SCALE,
            colors::TEXT,
            self.viewport,
        );
        *self.cursor_y += h + ROW_GAP;
    }

    /// Кнопка во всю ширину содержимого с текстом по центру.
    pub fn button(&mut self, label: &str) -> Response {
        let id = self.next_id(label);
        let rect = Rect::new(self.content.x, *self.cursor_y, self.content.w, BUTTON_H);
        let hovered = rect.contains(self.ui.cursor);
        if self.input.just_pressed && hovered {
            self.ui.pressed = Some(id);
        }
        let clicked = hovered && self.input.just_released && self.ui.pressed == Some(id);

        // Кнопка — свой nine-patch: углы не растягиваются ни при какой
        // ширине/высоте. Состояния — тинтом (норма 0.86 → hover 1.0 →
        // нажатая 0.7): умножение может только темнить, поэтому база светлее.
        let tint = if hovered && self.input.down {
            colors::BTN_PRESSED
        } else if hovered {
            colors::BTN_HOVER
        } else {
            colors::BTN
        };
        let cells = self.ui.button_patch.sprites(rect, self.viewport, tint, LAYER_WIDGET);
        self.sprites.extend(cells);

        // Подпись по центру кнопки.
        let tw = self
            .ui
            .font
            .text_width(self.queue, self.atlas, label, TEXT_SCALE);
        let th = self.ui.font.line_height(TEXT_SCALE);
        self.ui.font.push_text(
            self.sprites,
            self.queue,
            self.atlas,
            label,
            [rect.x + (rect.w - tw) * 0.5, rect.y + (rect.h - th) * 0.5],
            TEXT_SCALE,
            colors::BTN_TEXT,
            self.viewport,
        );

        *self.cursor_y += BUTTON_H + ROW_GAP;
        Response { hovered, clicked }
    }

    /// Подпись + горизонтальный бар `0..=1` во всю ширину содержимого.
    pub fn progress_bar(&mut self, caption: &str, progress: f32) {
        let progress = progress.clamp(0.0, 1.0);
        // Подпись мелкая над баром.
        let cap_h = self.ui.font.line_height(0.375);
        self.ui.font.push_text(
            self.sprites,
            self.queue,
            self.atlas,
            caption,
            [self.content.x, *self.cursor_y],
            0.375,
            colors::TEXT,
            self.viewport,
        );
        *self.cursor_y += cap_h + 4.0;

        let rect = Rect::new(self.content.x, *self.cursor_y, self.content.w, BAR_H);
        let mut bg = Sprite::new(rect.world_center(self.viewport), [rect.w, rect.h], self.ui.rect_uv);
        bg.tint = colors::BAR_BG;
        bg.layer = LAYER_WIDGET;
        self.sprites.push(bg);

        if progress > 0.0 {
            let w = (rect.w - 2.0) * progress;
            let fill = Rect::new(rect.x + 1.0, rect.y + 1.0, w, rect.h - 2.0);
            let mut s = Sprite::new(fill.world_center(self.viewport), [fill.w, fill.h], self.ui.rect_uv);
            s.tint = colors::BAR_FILL;
            s.layer = LAYER_WIDGET;
            self.sprites.push(s);
        }

        *self.cursor_y += BAR_H + ROW_GAP;
    }

    /// Свободное место (px) в стеке.
    pub fn space(&mut self, px: f32) {
        *self.cursor_y += px;
    }

    /// Прямоугольник содержимого (для нестандартных виджетов поверх).
    pub fn content_rect(&self) -> Rect {
        self.content
    }
}
