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

pub mod glass;
pub mod layout;
pub mod motion;
pub mod scene;
pub mod nine_patch;
pub mod text;
pub mod theme;

use std::collections::HashMap;
use std::time::Instant;

use crate::scene::atlas::SpriteAtlas;
use crate::scene::batch::SpriteBatch;
use crate::scene::sprite::Sprite;
pub use layout::{anchored_rect, place_popup, Anchor, Edges, Rect, Side};
pub use motion::Motion;
pub use nine_patch::{panel_rgba, rect_rgba, NinePatch, MARGIN as PATCH_MARGIN, PATCH_RADIUS, PATCH_SIZE};
pub use glass::GlassBackdrop;
pub use text::{UiFont, BASE_PX};
pub use theme::{Size, UiTheme};

/// Вместимость UI-батча (спрайтов): панели + глифы с запасом.
const UI_BATCH_CAPACITY: usize = 8192;

/// Слои UI-батча (выше любых мировых слоёв).
pub const LAYER_PANEL: f32 = 1000.0;
pub const LAYER_WIDGET: f32 = 1001.0;
/// (текст — внутри [`text::LAYER_TEXT`]); tooltip выше всего.
pub const LAYER_TOOLTIP: f32 = 1003.0;

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
    /// Клавиша UI этого кадра (стрелки/Enter — навигация фокусом).
    pub key: Option<UiKey>,
    /// Дельта колеса мыши за кадр (положительная — «на себя», контент вниз).
    pub scroll: f32,
}

/// Клавиша, интересная UI (маппится из winit в бинаре).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiKey {
    Up,
    Down,
    Enter,
}

/// Результат взаимодействия с виджетом за кадр.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Response {
    /// Курсор над виджетом в этом кадре (у disabled — всегда false).
    pub hovered: bool,
    /// Клик завершён на виджете (нажатие и отпускание внутри).
    pub clicked: bool,
    /// Виджет недоступен: интерактив заглушён.
    pub disabled: bool,
    /// Значение controlled-виджета (checkbox/switch/slider) изменилось.
    pub changed: bool,
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
    /// Nine-patch рамки стеклянной панели (тонкая рамка + почти прозрачный fill).
    glass_patch: NinePatch,
    /// UV маски скругления для стекла (регион ui/panel).
    panel_uv: [f32; 4],
    /// Отдельный батч стеклянных подложек (рисуется своим frost-проходом).
    frost_batch: SpriteBatch,
    /// Спрайты стекла текущего кадра (до upload).
    frost_sprites: Vec<Sprite>,
    /// Имя/UV плоского белого прямоугольника (бары).
    rect_name: &'static str,
    rect_uv: [f32; 4],
    /// Имя/UV белого круга (бегунки switch/slider).
    thumb_name: &'static str,
    thumb_uv: [f32; 4],
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
    /// Масштаб DPI (pixels_per_point окна): все метрики UI умножаются.
    /// Ставится `set_scale(..)` из события `ScaleFactorChanged`.
    pub(crate) scale: f32,
    /// С какого кадра виджет в hover (для задержки tooltip'а).
    pub(crate) hover_since: HashMap<u64, Instant>,
    /// id виджета, который тащат мышью (slider).
    pub(crate) dragging: Option<u64>,
    /// Счётчик виджетов кадра (часть ключа id; сбрасывается в `begin`).
    pub(crate) frame_seq: u64,
    /// Клавиатурный фокус (id виджета); перемещается стрелками.
    focus: Option<u64>,
    /// Порядок фокусируемых виджетов текущего кадра (строится в begin..upload).
    focus_order: Vec<u64>,
    /// Порядок фокуса ЗАВЕРШЁННОГО кадра (по нему работают стрелки).
    focus_order_prev: Vec<u64>,
    /// Открытое popup-меню (id его кнопки); одновременно открыто одно.
    menu_open: Option<u64>,
    /// Смещения скролла по id панели (между кадрами).
    scroll_offsets: HashMap<u64, f32>,
    /// Скролл-слои завершённого кадра (для `Renderer::render_ui_clip`).
    pub(crate) scroll_layers: Vec<ScrollLayer>,
    /// Пул батчей скролл-слоёв (ключ — id панели; переиспользуются).
    scroll_pool: HashMap<u64, SpriteBatch>,
    /// Копия устройства: ленивое создание батчей пула.
    device: wgpu::Device,
}

/// Скролл-слой: контент панели, обрезаемый по её прямоугольнику.
/// Рисуется `Renderer::render_ui_clip(.., Some(layer.clip()))` ПОСЛЕ
/// основного батча (клип в экранных пикселях, origin — верхний левый угол).
pub struct ScrollLayer {
    /// Стабильный id панели (ключ батча в пуле UiSystem).
    pub id: u64,
    pub(crate) rect: Rect,
}

impl ScrollLayer {
    /// Прямоугольник отсечения для `render_ui_clip`.
    pub fn clip(&self) -> crate::renderer::ScissorRect {
        crate::renderer::ScissorRect {
            x: self.rect.x.round() as u32,
            y: self.rect.y.round() as u32,
            width: self.rect.w.round() as u32,
            height: self.rect.h.round() as u32,
        }
    }

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
        // Белый круг для бегунков switch/slider (цвет задаёт tint).
        let (w, h, px) = crate::scene::atlas::circle_rgba(12, [255, 255, 255, 255]);
        atlas
            .add_rgba(queue, "ui/thumb", w, h, &px)
            .expect("атлас растёт под регион ui/thumb");
        // Стеклянная рамка: та же геометрия, но fill почти прозрачный —
        // через него виден размытый бэкдроп (frost-проход под ней).
        let (w, h, px) = panel_rgba([188, 196, 214, 255], [255, 255, 255, 26]);
        atlas
            .add_rgba(queue, "ui/panel_glass", w, h, &px)
            .expect("атлас растёт под регион ui/panel_glass");
        let uv = atlas.region("ui/panel").expect("регион только что добавлен");
        let patch = NinePatch::new(uv, PATCH_SIZE as f32, PATCH_MARGIN);
        let guv = atlas.region("ui/panel_glass").expect("регион только что добавлен");
        let glass_patch = NinePatch::new(guv, PATCH_SIZE as f32, PATCH_MARGIN);
        let buv = atlas.region("ui/button").expect("регион только что добавлен");
        let button_patch = NinePatch::new(buv, PATCH_SIZE as f32, PATCH_MARGIN);
        let rect_uv = atlas.region("ui/rect").expect("регион только что добавлен");
        let thumb_uv = atlas.region("ui/thumb").expect("регион только что добавлен");

        Self {
            font: UiFont::new(device, queue, atlas),
            patch,
            glass_patch,
            panel_uv: uv,
            frost_batch: SpriteBatch::new(device, 256),
            frost_sprites: Vec::new(),
            button_patch,
            rect_name: "ui/rect",
            rect_uv,
            thumb_name: "ui/thumb",
            thumb_uv,
            batch: SpriteBatch::new(device, UI_BATCH_CAPACITY),
            cursor: [0.0, 0.0],
            pressed: None,
            theme: UiTheme::dark(),
            motion: Motion::new(),
            now: Instant::now(),
            reduced_motion: false,
            scale: 1.0,
            hover_since: HashMap::new(),
            dragging: None,
            frame_seq: 0,
            focus: None,
            focus_order: Vec::new(),
            focus_order_prev: Vec::new(),
            menu_open: None,
            scroll_offsets: HashMap::new(),
            scroll_layers: Vec::new(),
            scroll_pool: HashMap::new(),
            device: device.clone(),
        }
    }

    /// Батч последнего кадра (для [`crate::Renderer::render_ui`] вне
    /// [`UiFrame`], например в тестах).
    pub fn batch(&self) -> &SpriteBatch {
        &self.batch
    }

    /// Батч стекла последнего кадра (см. [`UiFrame::frost_batch`]).
    pub fn frost_batch(&self) -> &SpriteBatch {
        &self.frost_batch
    }

    /// Скролл-слои последнего кадра (см. [`UiFrame::scroll_panel`]):
    /// рисуются после `render_ui`, каждый со своим клипом:
    ///
    /// ```text
    /// for layer in ui.scroll_layers() {
    ///     let batch = ui.scroll_batch(layer.id).unwrap();
    ///     renderer.render_ui_clip(queue, enc, view, atlas, batch, vp, Some(layer.clip()));
    /// }
    /// ```
    pub fn scroll_layers(&self) -> &[ScrollLayer] {
        &self.scroll_layers
    }

    /// Батч скролл-панели по id (см. [`Self::scroll_layers`]).
    pub fn scroll_batch(&self, id: u64) -> Option<&SpriteBatch> {
        self.scroll_pool.get(&id)
    }

    /// Масштаб DPI: множитель всех метрик (высоты кнопок, шрифты, отступы).
    /// Передавайте `window.scale_factor()`; hit-test остаётся корректным.
    pub fn set_scale(&mut self, pixels_per_point: f32) {
        self.scale = pixels_per_point.max(0.25);
    }

    /// Текущий масштаб DPI.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Сменить тему (роли цветов, отступы). Текстуры панели/кнопки
    /// ПЕРЕГЕНЕРИРУЮТСЯ: tint поверх тёмной текстуры не может осветлить её,
    /// поэтому светлая тема — это другая текстура. Старые регионы остаются
    /// в атласе (место тратится только при частых сменах темы).
    pub fn set_theme(&mut self, atlas: &mut SpriteAtlas, queue: &wgpu::Queue, theme: UiTheme) {
        let (w, h, px) = panel_rgba(theme.panel_border, theme.panel_fill);
        atlas
            .replace_rgba(queue, "ui/panel", w, h, &px)
            .expect("текстура ui/panel перезаписывается (тот же размер)");
        let (w, h, px) = panel_rgba(theme.button_border, theme.button_fill);
        atlas
            .replace_rgba(queue, "ui/button", w, h, &px)
            .expect("текстура ui/button перезаписывается (тот же размер)");
        let (w, h, px) = panel_rgba(theme.panel_border, [255, 255, 255, 26]);
        atlas
            .replace_rgba(queue, "ui/panel_glass", w, h, &px)
            .expect("текстура ui/panel_glass перезаписывается (тот же размер)");
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
        self.frame_seq = 0;
        // Навигация фокусом: по порядку ЗАВЕРШЁННОГО кадра.
        self.focus_order.clear();
        match input.key {
            Some(UiKey::Down) => {
                let idx = self
                    .focus_order_prev
                    .iter()
                    .position(|&i| Some(i) == self.focus);
                let next = match idx {
                    None => 0,
                    Some(i) => (i + 1).min(self.focus_order_prev.len().saturating_sub(1)),
                };
                self.focus = self.focus_order_prev.get(next).copied().or(self.focus);
            }
            Some(UiKey::Up) => {
                let idx = self
                    .focus_order_prev
                    .iter()
                    .position(|&i| Some(i) == self.focus);
                let prev = match idx {
                    None => self.focus_order_prev.len().saturating_sub(1),
                    Some(0) => 0,
                    Some(i) => i - 1,
                };
                self.focus = self.focus_order_prev.get(prev).copied().or(self.focus);
            }
            _ => {}
        }
        self.cursor = input.cursor;
        // Клик «нажать-отпустить на виджете»: pressed живёт до кадра ПОСЛЕ
        // отпускания (в кадре отпускания just_released ещё обрабатывается).
        if !input.down && !input.just_released {
            self.pressed = None;
        }
        // Свежие UV: добавление глифов могло вырастить атлас и переупаковать
        // регионы — кэшировать UV между кадрами нельзя.
        self.patch.uv = atlas.region("ui/panel").expect("регион ui/panel удалён из атласа");
        self.panel_uv = self.patch.uv;
        self.glass_patch.uv =
            atlas.region("ui/panel_glass").expect("регион ui/panel_glass удалён из атласа");
        self.frost_sprites.clear();
        self.scroll_layers.clear();
        self.button_patch.uv =
            atlas.region("ui/button").expect("регион ui/button удалён из атласа");
        self.rect_uv = atlas.region(self.rect_name).expect("регион ui/rect удалён из атласа");
        self.thumb_uv = atlas.region(self.thumb_name).expect("регион ui/thumb удалён из атласа");
        UiFrame {
            ui: self,
            atlas,
            queue,
            viewport,
            input,
            sprites: Vec::with_capacity(256),
            frost: Vec::new(),
            scrolls: Vec::new(),
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
    /// Стеклянные подложки кадра (рисуются отдельным frost-проходом).
    frost: Vec<Sprite>,
    /// Скролл-панели кадра: (id, прямоугольник панели, спрайты контента).
    scrolls: Vec<(u64, Rect, Vec<Sprite>)>,
    /// Тема кадра (снимок ролей на момент begin).
    theme: UiTheme,
}

impl<'a> UiFrame<'a> {
    /// Панель, приколоченная якорем `anchor` к краю экрана; содержимое —
    /// вертикальный стек внутри замыкания ([`Panel`]).
    pub fn panel(&mut self, anchor: Anchor, size: [f32; 2], content: impl FnOnce(&mut Panel)) {
        let rect =
            anchored_rect(anchor, size, self.viewport, self.theme.screen_margin * self.ui.scale);
        let panel_sprites = self.ui.patch.sprites(rect, self.viewport, self.theme.panel, LAYER_PANEL);
        self.sprites.extend(panel_sprites);

        let s = self.ui.scale;
        let mut cursor_y = rect.y + self.theme.content_margin * s;
        let mut panel = Panel {
            ctx: Ctx {
                ui: &mut *self.ui,
                atlas: &mut *self.atlas,
                queue: self.queue,
                sprites: &mut self.sprites,
                viewport: self.viewport,
                input: self.input,
            },
            content: rect.inner(self.theme.content_margin * s),
            cursor_y: &mut cursor_y,
            row_gap: self.theme.row_gap * s,
        };
        content(&mut panel);
    }

    /// Стеклянная панель: размытая сцена за прямоугольником (frosted glass).
    ///
    /// Подложка уходит в отдельный батч для `Renderer::render_ui_frost`
    /// (вызывается ДО `render_ui`); сверху рисуется тонкая рамка темы и
    /// обычное содержимое. Требует `renderer.set_ui_backdrop(..)`.
    pub fn glass_panel(&mut self, anchor: Anchor, size: [f32; 2], content: impl FnOnce(&mut Panel)) {
        let rect =
            anchored_rect(anchor, size, self.viewport, self.theme.screen_margin * self.ui.scale);
        let mut frost = Sprite::new(rect.world_center(self.viewport), [rect.w, rect.h], self.ui.panel_uv);
        frost.tint = self.theme.glass;
        frost.tint_end = self.theme.glass;
        frost.layer = LAYER_PANEL;
        self.frost.push(frost);

        // Рамка поверх стекла (тонкая, полупрозрачный fill в текстуре).
        let border = self.ui.glass_patch.sprites(rect, self.viewport, [1.0; 4], LAYER_PANEL);
        self.sprites.extend(border);

        let mut cursor_y = rect.y + self.theme.content_margin;
        let mut panel = Panel {
            ctx: Ctx {
                ui: &mut *self.ui,
                atlas: &mut *self.atlas,
                queue: self.queue,
                sprites: &mut self.sprites,
                viewport: self.viewport,
                input: self.input,
            },
            content: rect.inner(self.theme.content_margin),
            cursor_y: &mut cursor_y,
            row_gap: self.theme.row_gap,
        };
        content(&mut panel);
    }

    /// Панель со скроллом: контент выше панели прокручивается колесом мыши
    /// (`UiInput.scroll`), обрезается по прямоугольнику панели (scissor),
    /// справа рисуется полоса прокрутки.
    ///
    /// Контент рисуется в отдельный батч — игра рендерит его через
    /// `Renderer::render_ui_clip(.., Some(layer.clip()))` для каждого слоя
    /// [`UiSystem::scroll_layers`] ПОСЛЕ основного `render_ui`.
    pub fn scroll_panel(&mut self, anchor: Anchor, size: [f32; 2], content: impl FnOnce(&mut Panel)) {
        let rect = anchored_rect(anchor, size, self.viewport, self.theme.screen_margin);
        let id = self.scroll_id();
        // Колесо применяем ДО сборки: hit-test этого же кадра видит свежий
        // скролл (иначе клики по прокрученному контенту отстают на кадр).
        let wheel = self.input.scroll * 48.0;
        let old_scroll = self
            .ui
            .scroll_offsets
            .get(&id)
            .copied()
            .unwrap_or(0.0)
            .max(0.0)
            + wheel;

        // Фон и рамка панели — в основном батче (не скроллятся).
        // Запоминаем индекс начала КОНТЕНА: после замыкания забираем всё,
        // что добавилось ПОСЛЕ рамки (в кадре могут быть и другие панели).
        let content_start = self.sprites.len();
        let panel_sprites = self.ui.patch.sprites(rect, self.viewport, self.theme.panel, LAYER_PANEL);
        self.sprites.extend(panel_sprites);

        // Смещаем hit-test на скролл: курсор в координатах контента.
        let saved_cursor = self.ui.cursor;
        self.ui.cursor = [saved_cursor[0], saved_cursor[1] + old_scroll];

        let content_rect = rect.inner(self.theme.content_margin * self.ui.scale);
        let mut cursor_y = content_rect.y - old_scroll;
        let mut panel = Panel {
            ctx: Ctx {
                ui: &mut *self.ui,
                atlas: &mut *self.atlas,
                queue: self.queue,
                sprites: &mut self.sprites, // временно пишем в общий список
                viewport: self.viewport,
                input: self.input,
            },
            content: content_rect,
            cursor_y: &mut cursor_y,
            row_gap: self.theme.row_gap,
        };
        content(&mut panel);
        let content_h = cursor_y + old_scroll - content_rect.y;
        self.ui.cursor = saved_cursor;

        // Кламп по факту контента (max зависит от высоты контента).
        let max_scroll = (content_h - rect.h).max(0.0);
        let scroll = old_scroll.clamp(0.0, max_scroll);
        self.ui.scroll_offsets.insert(id, scroll);

        // Забираем спрайты КОНТЕНА панели (добавленные замыканием) и
        // сдвигаем вверх на scroll (мировая Y вверх = экранная вверх).
        let taken: Vec<Sprite> = self.sprites.drain(content_start + panel_sprites.len()..).collect();
        let mut shifted = taken;
        for s in &mut shifted {
            s.position[1] += scroll;
        }
        self.scrolls.push((id, rect, shifted));

        // Полоса прокрутки (основной батч): справа внутри панели.
        if max_scroll > 0.0 {
            let track_w = 4.0;
            let track_x = rect.x + rect.w - track_w - 2.0;
            let visible = rect.h / content_h;
            let thumb_h = (rect.h * visible).max(16.0);
            let travel = rect.h - thumb_h - 4.0;
            let ty = rect.y + 2.0 + (scroll / max_scroll) * travel;
            let mut track = Sprite::new(
                Rect::new(track_x, rect.y + 2.0, track_w, rect.h - 4.0).world_center(self.viewport),
                [track_w, rect.h - 4.0],
                self.ui.rect_uv,
            );
            track.tint = self.theme.bar_bg;
            track.layer = LAYER_WIDGET + 1.0;
            self.sprites.push(track);
            let mut thumb = Sprite::new(
                Rect::new(track_x, ty, track_w, thumb_h).world_center(self.viewport),
                [track_w, thumb_h],
                self.ui.rect_uv,
            );
            thumb.tint = self.theme.accent;
            thumb.layer = LAYER_WIDGET + 1.0;
            self.sprites.push(thumb);
        }
    }

    /// Заливает собранные спрайты в батч (`Renderer::render_ui` рисует его).
    pub fn upload(&mut self, queue: &wgpu::Queue) {
        // Стекло — в свой батч.
        self.ui.frost_batch.clear();
        for s in self.frost.drain(..) {
            self.ui.frost_batch.push(s);
        }
        self.ui.frost_batch.upload(queue);
        // Скролл-слои — в пул батчей (по id панели).
        self.ui.scroll_layers.clear();
        for (id, rect, sprites) in self.scrolls.drain(..) {
            if !self.ui.scroll_pool.contains_key(&id) {
                self.ui
                    .scroll_pool
                    .insert(id, SpriteBatch::new(&self.ui.device, 4096));
            }
            let batch = self.ui.scroll_pool.get_mut(&id).expect("только что вставлен");
            batch.clear();
            for s in sprites {
                batch.push(s);
            }
            batch.upload(queue);
            self.ui.scroll_layers.push(ScrollLayer { id, rect });
        }
        // Кадр завершён: порядок фокуса становится «прошлым».
        self.ui.focus_order_prev = std::mem::take(&mut self.ui.focus_order);
        if let Some(f) = self.ui.focus {
            if !self.ui.focus_order_prev.contains(&f) {
                self.ui.focus = None; // виджет исчез — фокус сброшен
            }
        }
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

    /// Батч стеклянных подложек для [`crate::Renderer::render_ui_frost`].
    pub fn frost_batch(&self) -> &SpriteBatch {
        &self.ui.frost_batch
    }

    /// Скролл-слои кадра: каждый рисуется `render_ui_clip(.., Some(clip))`.
    pub fn scroll_layers(&self) -> &[(u64, Rect, Vec<Sprite>)] {
        &self.scrolls
    }

    /// Стабильный id скролл-панели кадра (порядок вызова).
    fn scroll_id(&mut self) -> u64 {
        0x5C10_0000 + self.scrolls.len() as u64
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
/// или просто `p.button("X")`. Для динамически собранного текста —
/// [`Button::owned`] (карта UI, подстановки).
pub struct Button<'l> {
    label: std::borrow::Cow<'l, str>,
    size: Size,
    disabled: bool,
    tooltip: Option<std::borrow::Cow<'l, str>>,
}

impl<'l> Button<'l> {
    pub fn new(label: &'l str) -> Self {
        Self {
            label: std::borrow::Cow::Borrowed(label),
            size: Size::Medium,
            disabled: false,
            tooltip: None,
        }
    }

    /// Кнопка с владеемым текстом (текст собран в рантайме).
    pub fn owned(label: impl Into<String>) -> Self {
        Self {
            label: std::borrow::Cow::Owned(label.into()),
            size: Size::Medium,
            disabled: false,
            tooltip: None,
        }
    }

    /// Текст кнопки.
    pub fn label(&self) -> &str {
        &self.label
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

    /// Подсказка, всплывающая после задержки hover (см. `TOOLTIP_DELAY_MS`).
    pub fn tooltip(mut self, text: impl Into<std::borrow::Cow<'l, str>>) -> Self {
        self.tooltip = Some(text.into());
        self
    }
}

impl<'l> From<&'l str> for Button<'l> {
    fn from(label: &'l str) -> Self {
        Button::new(label)
    }
}

/// Задержка появления tooltip'а после начала hover (заимствовано из
/// gpui-component: enter 150 мс).
pub const TOOLTIP_DELAY_MS: u64 = 150;

/// Общий контекст отрисовки виджетов — у [`Panel`] и [`Row`] один и тот же
/// набор операций; различается только способ вычисления прямоугольника.
pub(crate) struct Ctx<'a> {
    ui: &'a mut UiSystem,
    atlas: &'a mut SpriteAtlas,
    queue: &'a wgpu::Queue,
    sprites: &'a mut Vec<Sprite>,
    viewport: [f32; 2],
    input: UiInput,
}

impl<'a> Ctx<'a> {
    fn theme(&self) -> &UiTheme {
        &self.ui.theme
    }

    /// Масштаб DPI текущего кадра.
    fn s(&self) -> f32 {
        self.ui.scale
    }

    /// Уникальный id виджета: hash подписи + порядковый номер кадра.
    ///
    /// Контракт (заимствовано из gpui-component, «Keyed element state»):
    /// подпись — часть интерфейса. Переименование сбрасывает анимации
    /// виджета; один id никогда не используется для разных каналов.
    fn id(&mut self, label: &str) -> u64 {
        self.ui.frame_seq += 1;
        let mut h: u64 = 0xcbf29ce484222325 ^ (self.ui.frame_seq << 32);
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

    fn push_text(&mut self, text: &str, at: [f32; 2], scale: f32, color: [f32; 4]) {
        self.ui.font.push_text(self.sprites, self.queue, self.atlas, text, at, scale, color, self.viewport);
    }

    fn text_width(&mut self, text: &str, scale: f32) -> f32 {
        self.ui.font.text_width(self.queue, self.atlas, text, scale)
    }

    fn line_height(&self, scale: f32) -> f32 {
        self.ui.font.line_height(scale)
    }

    /// Регистрирует виджет в порядке фокуса; возвращает true, если он в фокусе.
    fn register_focus(&mut self, id: u64) -> bool {
        self.ui.focus_order.push(id);
        self.ui.focus == Some(id)
    }

    /// Акцентная рамка вокруг виджета в фокусе (токен `accent` темы).
    fn focus_ring(&mut self, rect: Rect) {
        let d = 2.0 * self.s();
        let ring = Rect::new(rect.x - d, rect.y - d, rect.w + 2.0 * d, rect.h + 2.0 * d);
        let mut s = Sprite::new(ring.world_center(self.viewport), [ring.w, ring.h], self.ui.rect_uv);
        s.tint = self.theme().accent;
        s.layer = LAYER_WIDGET; // пушится ДО тела виджета — под ним
        self.sprites.push(s);
    }

    /// Popup-меню (общая логика для Panel; id = hash подписи кнопки).
    fn menu_at(&mut self, label: &str, rect: Rect, items: &[&str]) -> Option<usize> {
        let id = self.id(label);
        let focused = self.register_focus(id);
        let open = self.ui.menu_open == Some(id);
        let hovered = rect.contains(self.ui.cursor);
        let enter = focused && self.input.key == Some(UiKey::Enter);

        // Открытие/закрытие: клик по кнопке или Enter.
        if (self.input.just_pressed && hovered) || enter {
            self.ui.menu_open = if open { None } else { Some(id) };
        }
        let open = self.ui.menu_open == Some(id);

        // Кнопка меню (подсвечена, когда открыто).
        let tint = if open {
            self.theme().button_pressed
        } else {
            self.theme().button
        };
        let mut bg = Sprite::new(rect.world_center(self.viewport), [rect.w, rect.h], self.ui.rect_uv);
        bg.tint = tint;
        bg.layer = LAYER_WIDGET;
        self.sprites.push(bg);
        let cells = self.ui.button_patch.sprites(rect, self.viewport, self.theme().panel, LAYER_WIDGET);
        self.sprites.extend(cells);
        let scale = Size::Medium.text_scale();
        let th = self.line_height(scale);
        let tw = self.text_width(label, scale);
        self.push_text(
            label,
            [rect.x + (rect.w - tw) * 0.5, rect.y + (rect.h - th) * 0.5],
            scale,
            self.theme().button_text,
        );

        if !open {
            return None;
        }

        // Popup под кнопкой (флип наверх, если снизу нет места).
        let item_h = 24.0 * self.s();
        let pad = 4.0 * self.s();
        let popup_size = [rect.w, items.len() as f32 * item_h + pad * 2.0];
        let popup = place_popup(rect, popup_size, self.viewport, layout::Side::Below, 4.0);
        // Клик вне кнопки и вне popup — закрыть (только закрытие, без выбора).
        if self.input.just_pressed && !hovered && !popup.contains(self.ui.cursor) {
            self.ui.menu_open = None;
            return None;
        }

        let cells = self.ui.patch.sprites(popup, self.viewport, self.theme().panel, LAYER_TOOLTIP);
        self.sprites.extend(cells);

        let mut selected = None;
        let mut y = popup.y + pad;
        for (i, item) in items.iter().enumerate() {
            let irect = Rect::new(popup.x + pad, y, popup.w - pad * 2.0, item_h);
            let hov = irect.contains(self.ui.cursor);
            if hov {
                let mut hl = Sprite::new(irect.world_center(self.viewport), [irect.w, irect.h], self.ui.rect_uv);
                hl.tint = self.theme().button_hover;
                hl.layer = LAYER_TOOLTIP;
                self.sprites.push(hl);
            }
            let it_th = self.line_height(Size::Small.text_scale() * self.s());
            self.push_text(
                item,
                [irect.x + 6.0 * self.s(), irect.y + (irect.h - it_th) * 0.5],
                Size::Small.text_scale() * self.s(),
                self.theme().text,
            );
            if hov && self.input.just_released {
                selected = Some(i);
                self.ui.menu_open = None;
            }
            y += item_h;
        }
        selected
    }

    /// Задержка hover для tooltip'а: true, если курсор над виджетом уже
    /// достаточно долго. Также ведёт учёт `hover_since`.
    fn tooltip_due(&mut self, id: u64, hovered: bool) -> bool {
        if hovered {
            let since = *self.ui.hover_since.entry(id).or_insert(self.ui.now);
            self.ui.now.duration_since(since).as_millis() as u64 >= TOOLTIP_DELAY_MS
        } else {
            self.ui.hover_since.remove(&id);
            false
        }
    }

    /// Кнопка в явном прямоугольнике (общая для [`Panel`] и [`Row`]).
    fn button_at(&mut self, rect: Rect, spec: Button<'_>) -> Response {
        let Button { label, size, disabled, tooltip } = spec;
        let label = label.as_ref();
        let tooltip = tooltip.as_ref().map(|t| t.as_ref());
        let id = self.id(label);
        let focused = self.register_focus(id);
        let hovered = !disabled && rect.contains(self.ui.cursor);
        if self.input.just_pressed && hovered {
            self.ui.pressed = Some(id);
        }
        let enter = focused && !disabled && self.input.key == Some(UiKey::Enter);
        let clicked = (hovered && self.input.just_released && self.ui.pressed == Some(id)) || enter;
        if focused && !disabled {
            self.focus_ring(rect);
        }

        // Подсветка hover плавно едет к цели (keyed transition); у disabled
        // она всегда 0 — disabled глушит hover/active визуально.
        let hover_target = if hovered { 1.0 } else { 0.0 };
        let hover = self.animate(id, motion::channels::HOVER, hover_target);
        let tint = lerp_tint(self.theme().button, self.theme().button_hover, hover);
        let tint = if disabled {
            self.theme().button_disabled
        } else if self.ui.pressed == Some(id) && self.input.down {
            self.theme().button_pressed
        } else {
            tint
        };

        let cells = self.ui.button_patch.sprites(rect, self.viewport, tint, LAYER_WIDGET);
        self.sprites.extend(cells);

        // Подпись по центру кнопки.
        let scale = size.text_scale() * self.s();
        let tw = self.text_width(label, scale);
        let th = self.line_height(scale);
        let text_color = if disabled {
            self.theme().button_text_faded()
        } else {
            self.theme().button_text
        };
        self.push_text(
            label,
            [rect.x + (rect.w - tw) * 0.5, rect.y + (rect.h - th) * 0.5],
            scale,
            text_color,
        );

        // Tooltip: только после задержки hover, поверх всего, с флипом у края.
        if !disabled && tooltip.is_some() && self.tooltip_due(id, hovered) {
            let text = tooltip.unwrap_or_default();
            let dpi = self.s();
            let scale = Size::Small.text_scale() * dpi;
            let tw = self.text_width(text, scale);
            let th = self.line_height(scale);
            let popup = [tw + 16.0 * dpi, th + 10.0 * dpi];
            let where_ = place_popup(rect, popup, self.viewport, layout::Side::Above, 6.0 * dpi);
            let cells = self.ui.patch.sprites(where_, self.viewport, self.theme().panel, LAYER_TOOLTIP);
            self.sprites.extend(cells);
            self.push_text(
                text,
                [where_.x + 8.0 * dpi, where_.y + 5.0 * dpi],
                scale,
                self.theme().text,
            );
        }

        Response { hovered, clicked, disabled, changed: false }
    }

    /// Checkbox в явном прямоугольнике: коробка + подпись, клик переключает.
    /// Управляемый виджет: значение передаётся снаружи (`&mut bool`).
    fn checkbox_at(&mut self, rect: Rect, label: &str, value: &mut bool) -> Response {
        let id = self.id(label);
        let focused = self.register_focus(id);
        let hovered = rect.contains(self.ui.cursor);
        if self.input.just_pressed && hovered {
            self.ui.pressed = Some(id);
        }
        let enter = focused && self.input.key == Some(UiKey::Enter);
        let clicked = (hovered && self.input.just_released && self.ui.pressed == Some(id)) || enter;
        let changed = clicked;
        if clicked {
            *value = !*value;
        }
        if focused {
            let w = 18.0 + 8.0 + self.text_width(label, Size::Small.text_scale());
            self.focus_ring(Rect::new(rect.x, rect.y, w, rect.h));
        }

        // Коробка 18×18 (× DPI): включена — залита акцентом, выключена — рамка.
        let scale = Size::Small.text_scale() * self.s();
        let box_h = 18.0 * self.s();
        let ty = rect.y + (rect.h - box_h) * 0.5;
        let box_rect = Rect::new(rect.x, ty, box_h, box_h);
        let tint = if *value { self.theme().accent } else { self.theme().bar_bg };
        let mut b = Sprite::new(box_rect.world_center(self.viewport), [box_h, box_h], self.ui.rect_uv);
        b.tint = tint;
        b.layer = LAYER_WIDGET;
        self.sprites.push(b);
        // Рамка поверх (независимо от состояния — читаемость на любом фоне).
        let cells = self
            .ui
            .button_patch
            .sprites(box_rect, self.viewport, self.theme().panel, LAYER_WIDGET);
        self.sprites.extend(cells);

        // Подпись справа от коробки.
        let th = self.line_height(scale);
        self.push_text(
            label,
            [rect.x + box_h + 8.0 * self.s(), ty + (box_h - th) * 0.5],
            scale,
            self.theme().text,
        );

        Response { hovered, clicked, disabled: false, changed }
    }

    /// Switch в явном прямоугольнике: «пилл» с бегунком (позиция анимируется).
    fn switch_at(&mut self, rect: Rect, label: &str, value: &mut bool) -> Response {
        let id = self.id(label);
        let focused = self.register_focus(id);
        let hovered = rect.contains(self.ui.cursor);
        if self.input.just_pressed && hovered {
            self.ui.pressed = Some(id);
        }
        let enter = focused && self.input.key == Some(UiKey::Enter);
        let clicked = (hovered && self.input.just_released && self.ui.pressed == Some(id)) || enter;
        let changed = clicked;
        if clicked {
            *value = !*value;
        }
        if focused {
            let w = 44.0 + 8.0 + self.text_width(label, Size::Small.text_scale());
            self.focus_ring(Rect::new(rect.x, rect.y, w, rect.h));
        }

        // Трек: 44×24 (× DPI), бегунок 20 px, ход 20 px (паддинг 2).
        let dpi = self.s();
        let track_w = 44.0 * dpi;
        let track_h = 24.0 * dpi;
        let thumb_d = 20.0 * dpi;
        let ty = rect.y + (rect.h - track_h) * 0.5;
        let track = Rect::new(rect.x, ty, track_w, track_h);
        let mut t = Sprite::new(track.world_center(self.viewport), [track_w, track_h], self.ui.rect_uv);
        t.tint = if *value { self.theme().accent } else { self.theme().bar_bg };
        t.layer = LAYER_WIDGET;
        self.sprites.push(t);

        // Бегунок плавно скользит (keyed transition по каналу THUMB).
        let travel = track_w - thumb_d - 4.0 * dpi;
        let pos = self.animate(id, motion::channels::THUMB, if *value { 1.0 } else { 0.0 });
        let cx = track.x + 2.0 * dpi + travel * pos + thumb_d * 0.5;
        let cy = track.y + 2.0 * dpi + thumb_d * 0.5;
        let thumb_uv = self.ui.thumb_uv;
        let mut th_sprite = Sprite::new(
            [cx - self.viewport[0] * 0.5, self.viewport[1] * 0.5 - cy],
            [thumb_d, thumb_d],
            thumb_uv,
        );
        th_sprite.tint = self.theme().button;
        th_sprite.layer = LAYER_WIDGET;
        self.sprites.push(th_sprite);

        // Подпись справа.
        let scale = Size::Small.text_scale() * dpi;
        let th = self.line_height(scale);
        self.push_text(label, [rect.x + track_w + 8.0 * dpi, ty + (track_h - th) * 0.5], scale, self.theme().text);

        Response { hovered, clicked, disabled: false, changed }
    }

    /// Slider в явном прямоугольнике: drag меняет `value` 0..=1.
    /// Hit-target — ВЕСЬ прямоугольник, дорожка внутри тоньше (заимствовано
    /// из gpui-component: «hit-target больше отрисовки»).
    fn slider_at(&mut self, rect: Rect, label: &str, value: &mut f32) -> Response {
        let id = self.id(label);
        let focused = self.register_focus(id);
        let hovered = rect.contains(self.ui.cursor);
        let dragging = self.ui.dragging == Some(id) && self.input.down;

        if self.input.just_pressed && hovered {
            self.ui.dragging = Some(id);
        }
        let mut changed = false;
        if self.ui.dragging == Some(id) && self.input.down {
            let t = ((self.ui.cursor[0] - rect.x) / rect.w).clamp(0.0, 1.0);
            if (t - *value).abs() > 1e-4 {
                *value = t;
                changed = true;
            }
        }
        // Клавиатура сфокусированного слайдера: Up/Down двигают значение
        // (доступность на пульте/геймпаде; горизонтальных клавиш в UiKey нет).
        if focused {
            let d = match self.input.key {
                Some(UiKey::Up) => 0.1,
                Some(UiKey::Down) => -0.1,
                _ => 0.0,
            };
            if d != 0.0 {
                let nv = (*value + d).clamp(0.0, 1.0);
                if (nv - *value).abs() > 1e-4 {
                    *value = nv;
                    changed = true;
                }
            }
        }
        if self.input.just_released && self.ui.dragging == Some(id) {
            self.ui.dragging = None;
        }

        // Дорожка по вертикальному центру + заполнение + бегунок (x DPI).
        let track_h = 6.0 * self.s();
        let cy = rect.y + rect.h * 0.5;
        let mut track = Sprite::new(
            [rect.x + rect.w * 0.5 - self.viewport[0] * 0.5, self.viewport[1] * 0.5 - cy],
            [rect.w, track_h],
            self.ui.rect_uv,
        );
        track.tint = self.theme().bar_bg;
        track.layer = LAYER_WIDGET;
        self.sprites.push(track);

        let shown = self.animate(id, motion::channels::FILL, *value);
        let fill_w = rect.w * shown;
        if fill_w > 0.0 {
            let fill = Rect::new(rect.x, cy - track_h * 0.5, fill_w, track_h);
            let mut f = Sprite::new(fill.world_center(self.viewport), [fill.w, fill.h], self.ui.rect_uv);
            f.tint = self.theme().accent;
            f.layer = LAYER_WIDGET;
            self.sprites.push(f);
        }

        let thumb_d = 18.0 * self.s();
        let thumb_x = rect.x + fill_w.max(thumb_d * 0.5) - thumb_d * 0.5;
        let thumb_uv = self.ui.thumb_uv;
        let mut thumb = Sprite::new(
            [thumb_x + thumb_d * 0.5 - self.viewport[0] * 0.5, self.viewport[1] * 0.5 - cy],
            [thumb_d, thumb_d],
            thumb_uv,
        );
        thumb.tint = if dragging { self.theme().accent } else { self.theme().button };
        thumb.layer = LAYER_WIDGET;
        self.sprites.push(thumb);

        // Подпись слева над дорожкой, значение — справа.
        let scale = Size::Small.text_scale() * self.s();
        let th = self.line_height(scale);
        self.push_text(label, [rect.x, rect.y], scale, self.theme().text_muted);
        let val_text = format!("{}", (*value * 100.0).round() as u32);
        let vw = self.text_width(&val_text, scale);
        self.push_text(&val_text, [rect.x + rect.w - vw, rect.y], scale, self.theme().text_muted);
        let _ = th;

        Response { hovered: hovered || dragging, clicked: false, disabled: false, changed }
    }
}

/// Вертикальный стек содержимого панели. Виджеты добавляются сверху вниз;
/// `cursor_y` — верх следующей строки.
pub struct Panel<'a> {
    ctx: Ctx<'a>,
    content: Rect,
    cursor_y: &'a mut f32,
    row_gap: f32,
}

impl<'a> Panel<'a> {
    /// Строка текста слева, во всю ширину содержимого.
    pub fn label(&mut self, text: &str) {
        self.label_sized(text, self.ctx.theme().text, Size::Medium);
    }

    /// Приглушённая строка (второстепенное).
    pub fn label_muted(&mut self, text: &str) {
        self.label_sized(text, self.ctx.theme().text_muted, Size::Small);
    }

    /// Строка с переносом по словам на ширину содержимого (для описаний).
    pub fn label_wrapped(&mut self, text: &str, color: [f32; 4], size: Size) {
        let scale = size.text_scale();
        let h = self.ctx.line_height(scale);
        for line in self.ctx.ui.font.wrap(text, self.content.w, scale) {
            self.ctx.push_text(&line, [self.content.x, *self.cursor_y], scale, color);
            *self.cursor_y += h;
        }
        *self.cursor_y += self.row_gap;
    }

    /// Строка текста с явным цветом/размером (для нестандартных случаев).
    pub fn label_sized(&mut self, text: &str, color: [f32; 4], size: Size) {
        let scale = size.text_scale() * self.ctx.s();
        let h = self.ctx.line_height(scale);
        self.ctx.push_text(text, [self.content.x, *self.cursor_y], scale, color);
        *self.cursor_y += h + self.row_gap;
    }

    /// Кнопка во всю ширину содержимого. Принимает `&str` или [`Button`].
    pub fn button<'s>(&mut self, spec: impl Into<Button<'s>>) -> Response {
        let b = spec.into();
        let h = b.size.button_h() * self.ctx.s();
        let rect = Rect::new(self.content.x, *self.cursor_y, self.content.w, h);
        let r = self.ctx.button_at(rect, b);
        *self.cursor_y += h + self.row_gap;
        r
    }

    /// Checkbox во всю ширину содержимого: клик переключает `value`.
    pub fn checkbox(&mut self, label: &str, value: &mut bool) -> Response {
        let h = 26.0 * self.ctx.s();
        let rect = Rect::new(self.content.x, *self.cursor_y, self.content.w, h);
        let r = self.ctx.checkbox_at(rect, label, value);
        *self.cursor_y += h + self.row_gap;
        r
    }

    /// Switch во всю ширину содержимого: клик переключает `value`.
    pub fn switch(&mut self, label: &str, value: &mut bool) -> Response {
        let h = 30.0 * self.ctx.s();
        let rect = Rect::new(self.content.x, *self.cursor_y, self.content.w, h);
        let r = self.ctx.switch_at(rect, label, value);
        *self.cursor_y += h + self.row_gap;
        r
    }

    /// Slider 0..=1 во всю ширину содержимого: drag меняет `value`.
    pub fn slider(&mut self, label: &str, value: &mut f32) -> Response {
        let h = 36.0 * self.ctx.s();
        let rect = Rect::new(self.content.x, *self.cursor_y, self.content.w, h);
        let r = self.ctx.slider_at(rect, label, value);
        *self.cursor_y += h + self.row_gap;
        r
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
        let id = self.ctx.id(caption);

        // Подпись мелкая над баром.
        let cap_scale = size.text_scale() * 0.75 * self.ctx.s();
        let cap_h = self.ctx.line_height(cap_scale);
        self.ctx.push_text(
            caption,
            [self.content.x, *self.cursor_y],
            cap_scale,
            self.ctx.theme().text_muted,
        );
        *self.cursor_y += cap_h + 4.0 * self.ctx.s();

        let bar_h = size.bar_h() * self.ctx.s();
        let rect = Rect::new(self.content.x, *self.cursor_y, self.content.w, bar_h);
        let mut bg = Sprite::new(rect.world_center(self.ctx.viewport), [rect.w, rect.h], self.ctx.ui.rect_uv);
        bg.tint = self.ctx.theme().bar_bg;
        bg.layer = LAYER_WIDGET;
        self.ctx.sprites.push(bg);

        let shown = self.ctx.animate(id, motion::channels::FILL, progress);
        if shown > 0.0 {
            let w = (rect.w - 2.0) * shown;
            let fill = Rect::new(rect.x + 1.0, rect.y + 1.0, w, rect.h - 2.0);
            let mut s = Sprite::new(fill.world_center(self.ctx.viewport), [fill.w, fill.h], self.ctx.ui.rect_uv);
            s.tint = self.ctx.theme().bar_fill;
            s.layer = LAYER_WIDGET;
            self.ctx.sprites.push(s);
        }

        *self.cursor_y += bar_h + self.row_gap;
    }

    /// Горизонтальная строка виджетов (h_flex из gpui-component): элементы
    /// идут слева направо с зазором; высота строки — по самому высокому.
    pub fn row(&mut self, content: impl FnOnce(&mut Row)) {
        let y = *self.cursor_y;
        let max_x = self.content.x + self.content.w;
        let mut row = Row {
            x: self.content.x,
            y,
            row_h: 0.0,
            gap_x: 8.0 * self.ctx.s(),
            max_x,
            ctx: Ctx {
                ui: &mut *self.ctx.ui,
                atlas: &mut *self.ctx.atlas,
                queue: self.ctx.queue,
                sprites: &mut *self.ctx.sprites,
                viewport: self.ctx.viewport,
                input: self.ctx.input,
            },
        };
        content(&mut row);
        let h = row.row_h;
        *self.cursor_y = y + h + self.row_gap;
    }

    /// Разделитель: тонкая линия во всю ширину содержимого.
    pub fn separator(&mut self) {
        let h = 2.0 * self.ctx.s();
        let rect = Rect::new(self.content.x, *self.cursor_y, self.content.w, h);
        let mut s = Sprite::new(rect.world_center(self.ctx.viewport), [rect.w, rect.h], self.ctx.ui.rect_uv);
        s.tint = self.ctx.theme().text_muted;
        s.layer = LAYER_WIDGET;
        self.ctx.sprites.push(s);
        *self.cursor_y += h + self.row_gap;
    }

    /// Кнопка, открывающая popup-меню; возвращает индекс выбранного пункта
    /// в кадре клика (или `None`). Одновременно открыто одно меню; клик вне
    /// меню закрывает его. Пункты — отключать/сепараторы: следующий транш.
    pub fn menu(&mut self, label: &str, items: &[&str]) -> Option<usize> {
        let h = Size::Medium.button_h();
        let rect = Rect::new(self.content.x, *self.cursor_y, self.content.w, h);
        *self.cursor_y += h + self.row_gap;
        self.ctx.menu_at(label, rect, items)
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

/// Горизонтальная строка: кнопки/чекбоксы/свитчи/слайдеры с естественной
/// шириной, слева направо. Ширина текстовых виджетов — по содержимому.
pub struct Row<'a> {
    x: f32,
    y: f32,
    row_h: f32,
    gap_x: f32,
    max_x: f32,
    ctx: Ctx<'a>,
}

impl<'a> Row<'a> {
    fn take(&mut self, w: f32, h: f32) -> Rect {
        let r = Rect::new(self.x, self.y, w, h);
        self.x += w + self.gap_x;
        self.row_h = self.row_h.max(h);
        r
    }

    /// Кнопка естественной ширины (текст + паддинги).
    pub fn button<'s>(&mut self, spec: impl Into<Button<'s>>) -> Response {
        let b = spec.into();
        let label = b.label.as_ref();
        let scale = b.size.text_scale();
        let tw = self.ctx.text_width(label, scale);
        let dpi = self.ctx.s();
        let w = (tw + 28.0 * dpi).min((self.max_x - self.x).max(0.0));
        let rect = self.take(w, b.size.button_h() * dpi);
        self.ctx.button_at(rect, b)
    }

    /// Checkbox с естественной шириной (коробка + подпись).
    pub fn checkbox(&mut self, label: &str, value: &mut bool) -> Response {
        let scale = Size::Small.text_scale();
        let tw = self.ctx.text_width(label, scale);
        let dpi = self.ctx.s();
        let rect = self.take((18.0 + 8.0) * dpi + tw, 26.0 * dpi);
        self.ctx.checkbox_at(rect, label, value)
    }

    /// Switch с естественной шириной (трек + подпись).
    pub fn switch(&mut self, label: &str, value: &mut bool) -> Response {
        let scale = Size::Small.text_scale();
        let tw = self.ctx.text_width(label, scale);
        let dpi = self.ctx.s();
        let rect = self.take((44.0 + 8.0) * dpi + tw, 30.0 * dpi);
        self.ctx.switch_at(rect, label, value)
    }

    /// Slider фиксированной ширины (дорожка с подписью).
    pub fn slider(&mut self, label: &str, width: f32, value: &mut f32) -> Response {
        let w = (width * self.ctx.s()).min((self.max_x - self.x).max(0.0));
        let rect = self.take(w, 36.0 * self.ctx.s());
        self.ctx.slider_at(rect, label, value)
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
