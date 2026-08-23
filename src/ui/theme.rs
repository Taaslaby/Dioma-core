//! -------------------------------------------------------------------------
//! Filename:    src/ui/theme.rs
//! Class:       tokens-design
//! Purpose:     design tokens for observatory UI
//! Description: every color/size/duration used by the UI lives here —
//!              widgets read tokens, never hardcode values.
//!
//! author: Atlasov anton
//! -------------------------------------------------------------------------
//
// Палитра — тёплый тёмный (Tokyo-Night-производная): фон холодно-тёмный,
// сенсорика синяя, возбуждение зелёное, торможение красное, память тёплая.

use egui::Color32;

/// Цветовые токены.
pub struct Colors {
    // Поверхности.
    pub background: Color32,
    pub panel: Color32,
    pub panel_stroke: Color32,
    pub grid: Color32,
    // Текст.
    pub text: Color32,
    pub text_dim: Color32,
    // Управление.
    pub accent: Color32,
    pub play: Color32,
    pub pause: Color32,
    // Типы нейронов/регионов (семантика — как в grown-схемах).
    pub sensory: Color32,
    pub inhibitory: Color32,
    pub motor: Color32,
    pub workspace: Color32,
    pub memory: Color32,
    // Мир.
    pub agent: Color32,
    pub food: Color32,
    pub trail: Color32,
    pub dead: Color32,
    /// Цвет линии каждого химического вещества (индекс = chem::N).
    pub chemicals: [Color32; crate::metabolism::chem::N],
}

impl Default for Colors {
    fn default() -> Self {
        Self {
            background: Color32::from_rgb(0x10, 0x10, 0x14),
            panel: Color32::from_rgb(0x16, 0x16, 0x1C),
            panel_stroke: Color32::from_rgb(0x2A, 0x2A, 0x33),
            grid: Color32::from_rgb(0x1E, 0x1E, 0x26),
            text: Color32::from_rgb(0xE6, 0xE6, 0xEC),
            text_dim: Color32::from_rgb(0x8A, 0x8A, 0x96),
            accent: Color32::from_rgb(0x7A, 0xA2, 0xF7),
            play: Color32::from_rgb(0x9E, 0xCE, 0x6A),
            pause: Color32::from_rgb(0xE0, 0xAF, 0x68),
            sensory: Color32::from_rgb(0x7A, 0xA2, 0xF7),
            inhibitory: Color32::from_rgb(0xF7, 0x76, 0x8E),
            motor: Color32::from_rgb(0xBB, 0x9A, 0xF7),
            workspace: Color32::from_rgb(0x73, 0xDA, 0xCA),
            memory: Color32::from_rgb(0xFF, 0x9E, 0x64),
            agent: Color32::from_rgb(0xFF, 0xFF, 0xFF),
            food: Color32::from_rgb(0x9E, 0xCE, 0x6A),
            trail: Color32::from_rgb(0x44, 0x44, 0x50),
            dead: Color32::from_rgb(0xF7, 0x76, 0x8E),
            chemicals: [
                Color32::from_rgb(0x9E, 0xCE, 0x6A), // glucose
                Color32::from_rgb(0x73, 0xDA, 0xCA), // food
                Color32::from_rgb(0xF7, 0x76, 0x8E), // hunger
                Color32::from_rgb(0xFF, 0x7A, 0x93), // pain
                Color32::from_rgb(0xFF, 0xD1, 0x66), // reward
                Color32::from_rgb(0xE0, 0xAF, 0x68), // fatigue
            ],
        }
    }
}

/// Сеточные токены верстки: внешние/внутренние отступы и зазоры.
pub struct Spacing {
    /// Отступ панели от края окна.
    pub pad: f32,
    /// Зазор между блоками внутри панели.
    pub gap: f32,
    /// Внутренний отступ содержимого виджета.
    pub inner: f32,
}

impl Default for Spacing {
    fn default() -> Self {
        Self {
            pad: 8.0,
            gap: 8.0,
            inner: 4.0,
        }
    }
}

/// Размеры лейаута окна.
pub struct LayoutMetrics {
    pub window: egui::Vec2,
    pub top_bar_height: f32,
    pub right_panel_width: f32,
    /// Высота каждого графика.
    pub plot_height: f32,
    /// Отступ от края канваса мира до края панели.
    pub canvas_margin: f32,
}

impl Default for LayoutMetrics {
    fn default() -> Self {
        Self {
            window: egui::Vec2::new(1280.0, 800.0),
            top_bar_height: 40.0,
            right_panel_width: 380.0,
            plot_height: 170.0,
            canvas_margin: 16.0,
        }
    }
}

/// Параметры визуализации динамики.
pub struct VizMetrics {
    /// Радиус узла региона мозга.
    pub node_radius: f32,
    /// Шаг сетки мира (единиц мира между линиями).
    pub world_grid_step: f64,
    /// Радиус точки еды.
    pub food_radius: f32,
    /// Радиус агента.
    pub agent_radius: f32,
    /// Длина следа агента (тиков).
    pub trail_len: usize,
    /// Глубина истории графиков (тиков).
    pub history_points: usize,
    /// Скорость нарастания «вспышки» региона (за тик, доля от нормы).
    pub flash_rise: f32,
    /// Затухание вспышки за кадр (экспонента).
    pub flash_decay: f32,
    /// Скорости симуляции (тиков за кадр).
    pub speed_steps: [usize; 4],
    /// Максимум шкалы химий (концентрации 0..255).
    pub chem_axis_max: f64,
    /// Максимум шкалы моторной активности (спайков/тик).
    pub motor_axis_max: f64,
    /// Толщина линий графиков.
    pub line_width: f32,
}

impl Default for VizMetrics {
    fn default() -> Self {
        Self {
            node_radius: 16.0,
            world_grid_step: 50.0,
            food_radius: 3.0,
            agent_radius: 6.0,
            trail_len: 600,
            history_points: 1200,
            flash_rise: 0.06,
            flash_decay: 0.86,
            speed_steps: [1, 5, 25, 100],
            chem_axis_max: 255.0,
            motor_axis_max: 60.0,
            line_width: 1.5,
        }
    }
}

/// Полный набор дизайн-токенов.
#[derive(Default)]
pub struct Theme {
    pub colors: Colors,
    pub spacing: Spacing,
    pub layout: LayoutMetrics,
    pub viz: VizMetrics,
}

/// Названия химий для легенды (порядок = chem::N).
pub const CHEM_NAMES: [&str; crate::metabolism::chem::N] =
    ["glucose", "food", "hunger", "pain", "reward", "fatigue"];

/// Названия регионов арены для подписей узлов.
pub const ARENA_REGION_NAMES: [&str; crate::arena::N_REGIONS] = ["S_L", "S_R", "I", "M_L", "M_R"];

/// Цвет региона по индексу (арена: сенсоры/тормозные/моторы).
pub fn region_color(theme: &Theme, region: usize) -> Color32 {
    match region {
        0 | 1 => theme.colors.sensory,
        2 => theme.colors.inhibitory,
        _ => theme.colors.motor,
    }
}
