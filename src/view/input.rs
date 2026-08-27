//! Ввод: экранные события → команды симуляции ([`GameCommand`]).
//!
//! Курсор переводится в мировые координаты один раз ([`InputMapper::on_cursor`],
//! через [`Camera::screen_to_world`] — ориентация Y вверх уже согласована),
//! клик делает пикинг по арене ([`Arena::pick`] — чистая геометрия в движке,
//! headless-тестируемая) и превращается в команду. Клавиши — в команды
//! паузы/скорости/снятия выделения.

use crate::view::present::BODY_RADIUS;
use dioma_engine::env::{AgentId, Vec2};
use dioma_engine::sim::Arena;
use dioma_render::Camera;
use winit::keyboard::KeyCode;

/// Команда игроку/симуляции (применяется в игровом цикле, см. bin/game.rs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GameCommand {
    /// Пауза: переключить (сим не шагает, пока пауза).
    TogglePause,
    /// Скорость: множитель тиков на кадр (1/2/4).
    Speed(u32),
    /// Выделить существо (Some(id)) или снять выделение (None) — клик по пустоте.
    Select(Option<AgentId>),
    /// Отладочный спавн существа в точке мира (Shift+клик).
    SpawnCritter(Vec2),
    /// Отладочный спавн ягоды в точке мира.
    SpawnBerry(Vec2),
}

/// Радиус пикинга существа: радиус тела + небольшой допуск на «промах мимо».
const PICK_R_CRITTER: f32 = BODY_RADIUS * 1.2;
/// Радиус пикинга ягоды.
const PICK_R_BERRY: f32 = 0.6;

/// Маппер ввода: курсор в мире + клики/клавиши → команды.
#[derive(Debug, Clone, Copy, Default)]
pub struct InputMapper {
    /// Позиция курсора в мировых координатах (обновляется в `on_cursor`).
    cursor_world: Vec2,
}

impl InputMapper {
    pub fn new() -> Self {
        Self::default()
    }

    /// Обновляет мировую позицию курсора из экранных пикселей (Y вниз).
    pub fn on_cursor(&mut self, screen: [f32; 2], camera: &Camera) {
        let w = camera.screen_to_world(screen);
        self.cursor_world = Vec2::new(w[0], w[1]);
    }

    /// Текущая мировая позиция курсора (для отладочного спавна из клавиш).
    pub fn cursor_world(&self) -> Vec2 {
        self.cursor_world
    }

    /// Клик левой кнопкой: Shift — спавн существа в точке курсора, иначе —
    /// выделение по пикингу (существо или ягода в радиусе; мимо — None).
    pub fn on_click(&mut self, shift: bool, arena: &Arena) -> Option<GameCommand> {
        if shift {
            Some(GameCommand::SpawnCritter(self.cursor_world))
        } else {
            let picked = arena.pick(self.cursor_world, PICK_R_CRITTER, PICK_R_BERRY);
            Some(GameCommand::Select(picked))
        }
    }

    /// Клавиша → команда. Space — пауза, 1/2/3 — скорость ×1/×2/×4,
    /// Escape — снять выделение.
    pub fn on_key(&mut self, key: KeyCode) -> Option<GameCommand> {
        match key {
            KeyCode::Space => Some(GameCommand::TogglePause),
            KeyCode::Digit1 => Some(GameCommand::Speed(1)),
            KeyCode::Digit2 => Some(GameCommand::Speed(2)),
            KeyCode::Digit3 => Some(GameCommand::Speed(4)),
            KeyCode::Escape => Some(GameCommand::Select(None)),
            _ => None,
        }
    }
}
