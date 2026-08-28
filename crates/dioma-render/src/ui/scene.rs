//! Декларативная UI-карта: формат сцены + отрисовка через immediate-виджеты.
//!
//! Идея — как в крутых движках: интерфейс описывается ДАННЫМИ (файл
//! `ui.json`), а не кодом; игра подставляет значения и получает действия.
//! Формат читает и Sprite Forge (вкладка «UI-карта» — редактор + превью),
//! рендерит dioma-render — файл является контрактом между ними.
//!
//! Структура: сцена → панели (`kind`: `solid` / `glass` / `scroll`, якорь,
//! размер) → виджеты (label/button/checkbox/switch/slider/progress/separator/
//! space/row). Текст поддерживает подстановки `{ключ}` — значения приходят из
//! [`UiBindings`] (адаптер состояния игры).
//!
//! Пример `ui.json`:
//! ```json
//! {
//!   "theme": "dark",
//!   "panels": [
//!     { "kind": "glass", "anchor": "top_left", "size": [300, 240], "children": [
//!         { "label": "Существо #{id}" },
//!         { "button": { "text": "Пауза", "action": "toggle_pause",
//!                       "tooltip": "Space", "size": "medium" } },
//!         { "progress": "hunger" },
//!         { "checkbox": { "bind": "sound", "text": "Звук" } },
//!         { "slider": { "bind": "volume" } },
//!         { "row": [ { "button": { "text": "Сброс", "action": "reset", "size": "small" } },
//!                    { "space": 8.0 } ] }
//!     ]}
//!   ]
//! }
//! ```
//!
//! Использование в игре:
//! ```text
//! let scene = UiScene::from_str(&json)?;
//! scene.apply_theme(&mut ui_system, &mut atlas, &queue);       // тема сцены
//! let mut frame = ui_system.begin(&mut atlas, &queue, input, vp);
//! scene.draw(&mut frame, &mut game_bindings);                   // карта + данные
//! frame.upload(&queue);
//! ```
//! Действия кнопок приходят в [`UiBindings::action`].

use serde::Deserialize;

use super::layout::Anchor;
use super::theme::Size;
use super::Button;

// ── Формат ──────────────────────────────────────────────────────────────────

/// Сцена UI: тема + список панелей.
#[derive(Clone, Debug, Deserialize)]
pub struct UiScene {
    /// Тема кадра: `"dark"` / `"light"` (см. [`UiTheme`]).
    #[serde(default)]
    pub theme: Option<String>,
    pub panels: Vec<UiPanel>,
}

/// Панель карты: якорь + размер + вид виджета-контейнера + дети.
/// Неизвестный якорь/тип — панель молча пропускается (формат-контракт).
#[derive(Clone, Debug, Deserialize)]
pub struct UiPanel {
    /// `solid` (по умолчанию) / `glass` / `scroll`.
    #[serde(default)]
    pub kind: PanelKind,
    /// `top_left` / `top_center` / `top_right` / `center_left` / `center` /
    /// `center_right` / `bottom_left` / `bottom_center` / `bottom_right`.
    pub anchor: String,
    /// [w, h] в экранных пикселях.
    pub size: [f32; 2],
    #[serde(default)]
    pub children: Vec<Widget>,
}

/// Вид контейнера панели.
#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PanelKind {
    #[default]
    Solid,
    Glass,
    Scroll,
}

/// Виджет карты (внешнее тегированное представление).
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum Widget {
    /// `{"label": "Голод: {hunger}"}`.
    Label { label: String },
    /// `{"muted": "второстепенное"}`.
    Muted { muted: String },
    /// `{"button": {"text": "..", "action": "..", "tooltip"?, "size"?}}`.
    Button { button: ButtonDef },
    /// `{"checkbox": {"bind": "sound", "text": "Звук"}}`.
    Checkbox { checkbox: BoolBind },
    /// `{"switch": {"bind": "sound", "text": "Звук"}}`.
    Switch { switch: BoolBind },
    /// `{"slider": {"bind": "volume"}}`.
    Slider { slider: F32Bind },
    /// `{"progress": "hunger"}` — подпись = ключ.
    Progress { progress: String },
    /// `{"separator": {}}`.
    Separator { separator: Sep },
    /// `{"space": 8.0}`.
    Space { space: f32 },
    /// `{"row": [виджеты]}`.
    Row { row: Vec<Widget> },
}

/// Уф-пустышка для `{"separator": {}}` (serde требует значение).
#[derive(Clone, Copy, Debug, Deserialize)]
pub struct Sep {}

/// Кнопка карты.
#[derive(Clone, Debug, Deserialize)]
pub struct ButtonDef {
    pub text: String,
    /// Имя действия — придёт в [`UiBindings::action`] при клике.
    pub action: String,
    #[serde(default)]
    pub tooltip: Option<String>,
    /// `small` / `medium` (по умолчанию) / `large`.
    #[serde(default)]
    pub size: String,
}

/// Привязка к bool-значению состояния.
#[derive(Clone, Debug, Deserialize)]
pub struct BoolBind {
    pub bind: String,
    #[serde(default)]
    pub text: String,
}

/// Привязка к f32-значению состояния.
#[derive(Clone, Debug, Deserialize)]
pub struct F32Bind {
    pub bind: String,
}

impl UiScene {
    /// Парсит карту из JSON-строки.
    pub fn from_str(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("UI-карта: {e}"))
    }

    /// Читает карту из файла (путь для `include`-стиля игровых ассетов).
    pub fn from_path(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_str(&text)
    }

    /// Применяет тему сцены (если указана) к системе.
    pub fn apply_theme(
        &self,
        system: &mut super::UiSystem,
        atlas: &mut crate::scene::atlas::SpriteAtlas,
        queue: &wgpu::Queue,
    ) {
        match self.theme.as_deref() {
            Some("light") => system.set_theme(atlas, queue, super::UiTheme::light()),
            Some(_) | None => {}
        }
    }

    /// Рисует карту в кадр UI: панели → виджеты, данные — через биндинги.
    pub fn draw(&self, frame: &mut super::UiFrame, data: &mut dyn UiBindings) {
        for panel in &self.panels {
            let Ok(anchor) = anchor_from_str(&panel.anchor) else { continue };
            let size = panel.size;
            match panel.kind {
                PanelKind::Solid => {
                    frame.panel(anchor, size, |p| draw_children(p, &panel.children, data));
                }
                PanelKind::Glass => {
                    frame.glass_panel(anchor, size, |p| draw_children(p, &panel.children, data));
                }
                PanelKind::Scroll => {
                    frame.scroll_panel(anchor, size, |p| draw_children(p, &panel.children, data));
                }
            }
        }
    }
}

/// Адаптер состояния игры: значения для подстановок и приёмник действий.
pub trait UiBindings {
    /// Отображаемый текст для ключа (`{id}` → "7").
    fn text(&self, key: &str) -> String;
    /// Числовое значение для progress/slider.
    fn value(&self, key: &str) -> f32;
    /// Текущее bool-значение (checkbox/switch).
    fn get_bool(&self, key: &str) -> bool;
    fn set_bool(&mut self, key: &str, value: bool);
    fn set_f32(&mut self, key: &str, value: f32);
    /// Действие кнопки ("toggle_pause").
    fn action(&mut self, action: &str);
}

// ── Отрисовка ───────────────────────────────────────────────────────────────

fn anchor_from_str(s: &str) -> Result<Anchor, String> {
    Ok(match s {
        "top_left" => Anchor::TopLeft,
        "top_center" => Anchor::TopCenter,
        "top_right" => Anchor::TopRight,
        "center_left" => Anchor::CenterLeft,
        "center" => Anchor::Center,
        "center_right" => Anchor::CenterRight,
        "bottom_left" => Anchor::BottomLeft,
        "bottom_center" => Anchor::BottomCenter,
        "bottom_right" => Anchor::BottomRight,
        _ => return Err(format!("неизвестный якорь: {s}")),
    })
}

fn size_from_str(s: &str) -> Size {
    match s {
        "small" => Size::Small,
        "large" => Size::Large,
        _ => Size::Medium,
    }
}

/// Подстановка `{ключ}` из биндингов.
fn interpolate(text: &str, data: &dyn UiBindings) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('{') {
        let Some(len) = rest[start..].find('}') else {
            // Непарная '{' — дописать хвост как есть, не терять текст.
            out.push_str(rest);
            return out;
        };
        out.push_str(&rest[..start]);
        let key = &rest[start + 1..start + len];
        out.push_str(&data.text(key));
        rest = &rest[start + len + 1..];
    }
    out.push_str(rest);
    out
}

fn draw_children(p: &mut super::Panel, children: &[Widget], data: &mut dyn UiBindings) {
    for w in children {
        draw_widget(p, w, data);
    }
}

fn draw_widget(p: &mut super::Panel, w: &Widget, data: &mut dyn UiBindings) {
    match w {
        Widget::Label { label } => {
            p.label(&interpolate(label, data));
        }
        Widget::Muted { muted } => {
            p.label_muted(&interpolate(muted, data));
        }
        Widget::Button { button } => {
            let r = p.button(button_from(button, data));
            if r.clicked {
                data.action(&button.action);
            }
        }
        Widget::Checkbox { checkbox } => {
            let mut v = data.get_bool(&checkbox.bind);
            let r = p.checkbox(&checkbox.text, &mut v);
            if r.changed {
                data.set_bool(&checkbox.bind, v);
            }
        }
        Widget::Switch { switch } => {
            let mut v = data.get_bool(&switch.bind);
            let text = if switch.text.is_empty() { switch.bind.clone() } else { switch.text.clone() };
            let r = p.switch(&text, &mut v);
            if r.changed {
                data.set_bool(&switch.bind, v);
            }
        }
        Widget::Slider { slider } => {
            // NaN от биндинга не должен доехать до GPU (clamp(NaN) = NaN).
            let raw = data.value(&slider.bind);
            let mut v = if raw.is_finite() { raw.clamp(0.0, 1.0) } else { 0.0 };
            let r = p.slider(&slider.bind, &mut v);
            if r.changed {
                data.set_f32(&slider.bind, v);
            }
        }
        Widget::Progress { progress } => {
            let raw = data.value(progress);
            let v = if raw.is_finite() { raw.clamp(0.0, 1.0) } else { 0.0 };
            p.progress_bar(progress, v);
        }
        Widget::Separator { .. } => p.separator(),
        Widget::Space { space } => p.space(*space),
        Widget::Row { row } => {
            // Кнопки в строке: действие по клику, биндинги те же.
            // Публичного доступа к Row вне замыкания нет — рисуем через
            // специальный экстерн-хелпер p.row_with(data, row).
            p.row_with(data, row);
        }
    }
}

// ── Расширения Panel/Row для карты (без изменения публичного API виджетов) ──

impl super::Panel<'_> {
    fn row_with(&mut self, data: &mut dyn UiBindings, widgets: &[Widget]) {
        self.row(|row| {
            for w in widgets {
                match w {
                    Widget::Button { button } => {
                        let r = row.button(button_from(button, data));
                        if r.clicked {
                            data.action(&button.action);
                        }
                    }
                    Widget::Checkbox { checkbox } => {
                        let mut v = data.get_bool(&checkbox.bind);
                        let r = row.checkbox(&checkbox.text, &mut v);
                        if r.changed {
                            data.set_bool(&checkbox.bind, v);
                        }
                    }
                    Widget::Switch { switch } => {
                        let mut v = data.get_bool(&switch.bind);
                        let text = if switch.text.is_empty() {
                            switch.bind.clone()
                        } else {
                            switch.text.clone()
                        };
                        let r = row.switch(&text, &mut v);
                        if r.changed {
                            data.set_bool(&switch.bind, v);
                        }
                    }
                    Widget::Slider { slider } => {
                        let mut v = data.value(&slider.bind).clamp(0.0, 1.0);
                        let r = row.slider(&slider.bind, 120.0, &mut v);
                        if r.changed {
                            data.set_f32(&slider.bind, v);
                        }
                    }
                    // space/label/muted/progress/row внутри строки
                    // не поддерживаются (горизонтальный отступ бессмыслен).
                    _ => {}
                }
            }
        });
    }
}

/// Собирает [`Button`] из определения карты (текст с подстановками —
/// владеемый, поэтому `Button::owned`).
fn button_from(def: &ButtonDef, data: &dyn UiBindings) -> Button<'static> {
    let b = Button::owned(interpolate(&def.text, data));
    let b = match size_from_str(&def.size) {
        Size::Small => b.small(),
        Size::Large => b.large(),
        Size::Medium => b,
    };
    match &def.tooltip {
        Some(t) if !t.is_empty() => b.tooltip(interpolate(t, data)),
        _ => b,
    }
}
