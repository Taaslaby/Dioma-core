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
pub mod motion;
pub mod nine_patch;
pub mod text;
pub mod theme;

use std::time::Instant;

use crate::scene::atlas::SpriteAtlas;
use crate::scene::batch::SpriteBatch;
use crate::scene::sprite::Sprite;
pub use layout::{anchored_rect, Anchor, Rect};
pub use motion::Motion;
pub use nine_patch::{panel_rgba, rect_rgba, NinePatch, MARGIN as PATCH_MARGIN, PATCH_RADIUS, PATCH_SIZE};
pub use text::{UiFont, BASE_PX};
pub use theme::{Size, UiTheme};

/// Вместимость UI-батча (спрайтов): панели + глифы с запасом.
const UI_BATCH_CAPACITY: usize = 8192;

/// Слои UI-батча (выше любых мировых слоёв).
pub const LAYER_PANEL: f32 = 1000.0;
pub const LAYER_WIDGET: f32 = 1001.0;
/// (текст — внутри [`text::LAYER_TEXT`]).

/// Отступ панелей от края экрана (px) по умолчанию (тема может
/// переопределить: `theme.screen_margin`).
pub const SCREEN_MARGIN: f32 = 16.0;
/// Длительность переходов hover/значений по умолчанию (как в gpui-component).
pub const TRANSITION_MS: u64 = 160;

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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Response {
    /// Курсор над виджетом в этом кадре (у disabled — всегда false).
    pub hovered: bool,
    /// Клик завершён на виджете (нажатие и отпускание внутри).
    pub clicked: bool,
    /// Виджет недоступен: интерактив заглушён.
    pub disabled: bool,
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
    /// Тема (роли цветов + отступы); переключается `set_theme`.
    pub(crate) theme: UiTheme,
    /// Keyed-анимации (transition/spring) между кадрами.
    pub(crate) motion: Motion,
    /// Момент начала текущего кадра (для анимаций).
    pub(crate) now: Instant,
    /// Глобальное отключение анимаций (доступность/скриншоты).
    pub reduced_motion: bool,
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
            theme: UiTheme::dark(),
            motion: Motion::new(),
            now: Instant::now(),
            reduced_motion: false,
        }
    }

    /// Батч последнего кадра (для [`crate::Renderer::render_ui`] вне
    /// [`UiFrame`], например в тестах).
    pub fn batch(&self) -> &SpriteBatch {
        &self.batch
    }

    /// Сменить тему (роли цветов, отступы). Вступает в силу со следующего
    /// кадра; анимации не сбрасываются.
    pub fn set_theme(&mut self, theme: UiTheme) {
        self.theme = theme;
    }

    /// Доступ к текущей теме (например, чтобы рисовать своим виджетом в
    /// цветах темы).
    pub fn theme(&self) -> &UiTheme {
        &self.theme
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
        self.now = Instant::now();
        let theme = self.theme.clone();
        self.now = Instant::now();
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
            theme,
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
    /// Тема кадра (снимок ролей на момент begin).
    theme: UiTheme,
}

impl<'a> UiFrame<'a> {
    /// Панель, приколоченная якорем `anchor` к краю экрана; содержимое —
    /// вертикальный стек внутри замыкания ([`Panel`]).
    pub fn panel(&mut self, anchor: Anchor, size: [f32; 2], content: impl FnOnce(&mut Panel)) {
        let rect = anchored_rect(anchor, size, self.viewport, self.theme.screen_margin);
        let panel_sprites = self.ui.patch.sprites(rect, self.viewport, self.theme.panel, LAYER_PANEL);
        self.sprites.extend(panel_sprites);

        let mut cursor_y = rect.y + self.theme.content_margin;
        let mut panel = Panel {
            ui: &mut *self.ui,
            atlas: &mut *self.atlas,
            queue: self.queue,
            sprites: &mut self.sprites,
            viewport: self.viewport,
            input: self.input,
            theme: &self.theme,
            next_id: &mut self.next_id,
            content: rect.inner(self.theme.content_margin),
            cursor_y: &mut cursor_y,
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

/// Спецификация кнопки (builder): `p.button(Button::new("X").small().disabled())`
/// или просто `p.button("X")`.
pub struct Button<'l> {
    label: &'l str,
    size: Size,
    disabled: bool,
}

impl<'l> Button<'l> {
    pub fn new(label: &'l str) -> Self {
        Self { label, size: Size::Medium, disabled: false }
    }

    /// Компактная кнопка (см. [`Size`]).
    pub fn small(mut self) -> Self {
        self.size = Size::Small;
        self
    }

    /// Крупная кнопка (меню/диалоги).
    pub fn large(mut self) -> Self {
        self.size = Size::Large;
        self
    }

    /// Недоступная кнопка: приглушена, hover/клик игнорируются.
    pub fn disabled(mut self) -> Self {
        self.disabled = true;
        self
    }
}

impl<'l> From<&'l str> for Button<'l> {
    fn from(label: &'l str) -> Self {
        Button::new(label)
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
    theme: &'a UiTheme,
    next_id: &'a mut u64,
    /// Область содержимого (панель минус поля).
    content: Rect,
    /// Верх следующей строки (экранные px).
    cursor_y: &'a mut f32,
}

impl<'a> Panel<'a> {
    /// Уникальный id виджета: hash подписи + порядковый номер кадра.
    ///
    /// Контракт (заимствовано из gpui-component, «Keyed element state»):
    /// подпись — часть интерфейса. Переименование сбрасывает анимации
    /// виджета; один id никогда не используется для разных каналов.
    fn next_id(&mut self, label: &str) -> u64 {
        *self.next_id += 1;
        let mut h: u64 = 0xcbf29ce484222325 ^ (*self.next_id << 32);
        for b in label.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Плавно едет к `target` по каналу `channel` (см. [`motion`]).
    fn animate(&mut self, id: u64, channel: u8, target: f32) -> f32 {
        self.ui.motion.animate(
            id,
            channel,
            target,
            std::time::Duration::from_millis(TRANSITION_MS),
            self.ui.now,
            self.ui.reduced_motion,
        )
    }

    /// Строка текста слева, во всю ширину содержимого.
    pub fn label(&mut self, text: &str) {
        self.label_sized(text, self.theme.text, Size::Medium);
    }

    /// Приглушённая строка (второстепенное).
    pub fn label_muted(&mut self, text: &str) {
        self.label_sized(text, self.theme.text_muted, Size::Small);
    }

    /// Строка текста с явным цветом/размером (для нестандартных случаев).
    pub fn label_sized(&mut self, text: &str, color: [f32; 4], size: Size) {
        let scale = size.text_scale();
        let h = self.ui.font.line_height(scale);
        self.ui.font.push_text(
            self.sprites,
            self.queue,
            self.atlas,
            text,
            [self.content.x, *self.cursor_y],
            scale,
            color,
            self.viewport,
        );
        *self.cursor_y += h + self.theme.row_gap;
    }

    /// Кнопка во всю ширину содержимого. Принимает `&str` или [`Button`].
    pub fn button(&mut self, spec: impl Into<Button<'a>>) -> Response {
        let Button { label, size, disabled } = spec.into();
        let id = self.next_id(label);
        let h = size.button_h();
        // Если кнопка недоступна, она не участвует в hit-test вовсе
        // (заимствовано из gpui-component: «Disabled interaction appearance»).
        let rect = Rect::new(self.content.x, *self.cursor_y, self.content.w, h);
        let hovered = !disabled && rect.contains(self.ui.cursor);
        if self.input.just_pressed && hovered {
            self.ui.pressed = Some(id);
        }
        let clicked = hovered && self.input.just_released && self.ui.pressed == Some(id);

        // Подсветка hover плавно едет к цели (keyed transition); у disabled
        // она всегда 0 — disabled глушит hover/active визуально.
        let hover_target = if hovered { 1.0 } else { 0.0 };
        let hover = self.animate(id, motion::channels::HOVER, hover_target);
        let tint = lerp_tint(self.theme.button, self.theme.button_hover, hover);
        let tint = if disabled {
            self.theme.button_disabled
        } else if self.ui.pressed == Some(id) && self.input.down {
            self.theme.button_pressed
        } else {
            tint
        };

        let cells = self.ui.button_patch.sprites(rect, self.viewport, tint, LAYER_WIDGET);
        self.sprites.extend(cells);

        // Подпись по центру кнопки.
        let scale = size.text_scale();
        let tw = self.ui.font.text_width(self.queue, self.atlas, label, scale);
        let th = self.ui.font.line_height(scale);
        let text_color = if disabled {
            self.theme.button_text_faded()
        } else {
            self.theme.button_text
        };
        self.ui.font.push_text(
            self.sprites,
            self.queue,
            self.atlas,
            label,
            [rect.x + (rect.w - tw) * 0.5, rect.y + (rect.h - th) * 0.5],
            scale,
            text_color,
            self.viewport,
        );

        *self.cursor_y += h + self.theme.row_gap;
        Response { hovered, clicked, disabled }
    }

    /// Подпись + горизонтальный бар `0..=1` во всю ширину содержимого.
    ///
    /// Отображаемое значение плавно догоняет фактическое (keyed transition
    /// по каналу FILL) — резкие скачки данных не дёргают интерфейс.
    pub fn progress_bar(&mut self, caption: &str, progress: f32) {
        self.progress_bar_sized(caption, progress, Size::Medium);
    }

    /// Бар с явным размером.
    pub fn progress_bar_sized(&mut self, caption: &str, progress: f32, size: Size) {
        let progress = progress.clamp(0.0, 1.0);
        let id = self.next_id(caption);

        // Подпись мелкая над баром.
        let cap_scale = size.text_scale() * 0.75;
        let cap_h = self.ui.font.line_height(cap_scale);
        self.ui.font.push_text(
            self.sprites,
            self.queue,
            self.atlas,
            caption,
            [self.content.x, *self.cursor_y],
            cap_scale,
            self.theme.text_muted,
            self.viewport,
        );
        *self.cursor_y += cap_h + 4.0;

        let bar_h = size.bar_h();
        let rect = Rect::new(self.content.x, *self.cursor_y, self.content.w, bar_h);
        let mut bg = Sprite::new(rect.world_center(self.viewport), [rect.w, rect.h], self.ui.rect_uv);
        bg.tint = self.theme.bar_bg;
        bg.layer = LAYER_WIDGET;
        self.sprites.push(bg);

        let shown = self.animate(id, motion::channels::FILL, progress);
        if shown > 0.0 {
            let w = (rect.w - 2.0) * shown;
            let fill = Rect::new(rect.x + 1.0, rect.y + 1.0, w, rect.h - 2.0);
            let mut s = Sprite::new(fill.world_center(self.viewport), [fill.w, fill.h], self.ui.rect_uv);
            s.tint = self.theme.bar_fill;
            s.layer = LAYER_WIDGET;
            self.sprites.push(s);
        }

        *self.cursor_y += bar_h + self.theme.row_gap;
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

/// Линейная интерполяция tint'а (для плавной подсветки hover).
fn lerp_tint(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}
