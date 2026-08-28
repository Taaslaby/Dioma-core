//! Тема UI: семантические роли вместо сырых цветов + таблица размеров.
//!
//! Заимствовано из gpui-component (`crates/ui/src/theme/theme_color.rs`,
//! `crates/ui/src/sizing.rs`): цвет обращается по РОЛИ (`button_hover`, а не
//! «голубой»), размеры — из таблицы `Size`, никогда не литералы в виджетах.
//!
//! Цвета — это tint поверх светлых процедурных текстур (умножение), поэтому
//! роль может только темнить базу; «ярче» достигается более светлым tint'ом.

/// Размер виджета: высоты/масштабы берутся из таблицы метрик.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Size {
    /// Кнопка 28 px, текст 12 px — плотный HUD.
    Small,
    /// Кнопка 36 px, текст 16 px — по умолчанию.
    Medium,
    /// Кнопка 44 px, текст 20 px — меню/диалоги.
    Large,
}

impl Size {
    /// Высота кнопки (px).
    pub fn button_h(self) -> f32 {
        match self {
            Size::Small => 28.0,
            Size::Medium => 36.0,
            Size::Large => 44.0,
        }
    }

    /// Высота прогресс-бара (px).
    pub fn bar_h(self) -> f32 {
        match self {
            Size::Small => 10.0,
            Size::Medium => 14.0,
            Size::Large => 18.0,
        }
    }

    /// Масштаб текста (от `BASE_PX` = 32 px глиф).
    pub fn text_scale(self) -> f32 {
        match self {
            Size::Small => 0.375,
            Size::Medium => 0.5,
            Size::Large => 0.625,
        }
    }
}

/// Цветовая тема UI: роли → tint (RGBA, компоненты 0..1).
#[derive(Clone, Debug)]
pub struct UiTheme {
    /// Светлая ли тема (для кнопок-переключателей; роли сами по себе данные).
    pub is_light: bool,
    /// Подложка панели (поверх тёмной текстуры панели).
    pub panel: [f32; 4],
    /// Кнопка: обычная / под курсором / нажатая / недоступная.
    pub button: [f32; 4],
    pub button_hover: [f32; 4],
    pub button_pressed: [f32; 4],
    pub button_disabled: [f32; 4],
    /// Текст на тёмном фоне панели.
    pub text: [f32; 4],
    /// Приглушённый текст (подписи, второстепенное).
    pub text_muted: [f32; 4],
    /// Текст на светлой кнопке.
    pub button_text: [f32; 4],
    /// Бар: подложка и заливка.
    pub bar_bg: [f32; 4],
    pub bar_fill: [f32; 4],
    /// Акцентный цвет (выделение, ring фокуса — будущие виджеты).
    pub accent: [f32; 4],
    /// Цвета ТЕКСТУР панели/кнопки: tint поверх них может только темнить,
    /// поэтому светлая тема требует другой текстуры (см. `UiSystem::set_theme`).
    pub panel_fill: [u8; 4],
    pub panel_border: [u8; 4],
    pub button_fill: [u8; 4],
    pub button_border: [u8; 4],
    /// Стекло панели: RGB множитель размытой сцены, A — плотность.
    pub glass: [f32; 4],
    /// Отступы (spacing-токены, px).
    pub content_margin: f32,
    pub row_gap: f32,
    pub screen_margin: f32,
}

impl UiTheme {
    /// Тёмная тема по умолчанию (совпадает с прежним `mod colors`).
    pub fn dark() -> Self {
        Self {
            is_light: false,
            panel: [1.0, 1.0, 1.0, 1.0],
            button: [0.86, 0.88, 0.92, 1.0],
            button_hover: [1.0, 1.0, 1.0, 1.0],
            button_pressed: [0.70, 0.73, 0.80, 1.0],
            button_disabled: [0.55, 0.57, 0.62, 0.6],
            text: [0.95, 0.96, 0.92, 1.0],
            text_muted: [0.68, 0.71, 0.76, 1.0],
            button_text: [0.12, 0.14, 0.18, 1.0],
            bar_bg: [0.16, 0.17, 0.20, 1.0],
            bar_fill: [0.45, 0.85, 0.45, 1.0],
            accent: [0.40, 0.75, 1.0, 1.0],
            panel_fill: [22, 26, 38, 216],
            panel_border: [188, 196, 214, 255],
            glass: [0.82, 0.88, 1.0, 0.78],
            button_fill: [232, 238, 248, 255],
            button_border: [96, 106, 130, 255],
            content_margin: 12.0,
            row_gap: 10.0,
            screen_margin: 16.0,
        }
    }

    /// Приглушённый текст на кнопке (disabled-состояние).
    pub fn button_text_faded(&self) -> [f32; 4] {
        [
            self.button_text[0],
            self.button_text[1],
            self.button_text[2],
            self.button_text[3] * 0.45,
        ]
    }

    /// Светлая тема (демо/проверка переключения).
    pub fn light() -> Self {
        Self {
            is_light: true,
            panel: [0.94, 0.95, 0.98, 1.0],
            button: [0.98, 0.99, 1.0, 1.0],
            button_hover: [1.0, 1.0, 1.0, 1.0],
            button_pressed: [0.82, 0.85, 0.92, 1.0],
            button_disabled: [0.85, 0.86, 0.90, 0.6],
            text: [0.13, 0.15, 0.20, 1.0],
            text_muted: [0.40, 0.44, 0.52, 1.0],
            button_text: [0.12, 0.14, 0.18, 1.0],
            bar_bg: [0.78, 0.80, 0.86, 1.0],
            bar_fill: [0.20, 0.62, 0.30, 1.0],
            accent: [0.15, 0.50, 0.90, 1.0],
            panel_fill: [245, 247, 252, 240],
            panel_border: [120, 130, 150, 255],
            glass: [0.96, 0.98, 1.0, 0.85],
            button_fill: [255, 255, 255, 255],
            button_border: [110, 120, 145, 255],
            ..Self::dark()
        }
    }
}
