//! -------------------------------------------------------------------------
//! Filename:    src/world/side/mod.rs
//! Class:       SemantikaEngine
//! Purpose:     боковой мир с гравитацией: комнаты, проёмы, тела
//! Description: мир как у старых симов жизни: вид сбоку, падение,
//!              ходьба по полам комнат, провалы в проёмах полов.
//!              Проём = граница Y между «верхней» и «нижней» комнатой
//!              с диапазоном X: пересёк вниз — упал к нижним.
//! author: Atlasov anton
// -------------------------------------------------------------------------
pub mod sim;

/// Радиус тела существа (px).
pub const BODY_R: f64 = 7.0;
/// Гравитация (px/s²).
pub const GRAVITY: f64 = 900.0;
/// Горизонтальная скорость при полном приводе (px/s).
pub const WALK_SPEED: f64 = 70.0;
/// Импульс прыжка (px/s, вверх = отрицательный Y).
pub const JUMP_VY: f64 = -520.0;
/// Дистанция поедания (центр-центр).
pub const EAT_DIST: f64 = BODY_R + 5.0;

// ─── Комнаты и проёмы ───────────────────────────────────────────────────────
#[derive(Clone, Copy, Debug)]
pub struct Room {
    pub id: usize,
    pub x0: f64,
    pub x1: f64,
    /// Потолок (меньший Y).
    pub y_top: f64,
    /// Пол (больший Y).
    pub y_bot: f64,
}

impl Room {
    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y_top && y <= self.y_bot
    }
}

/// Проём на полу верхней комнаты: падение вниз уносит в нижнюю.
#[derive(Clone, Copy, Debug)]
pub struct Portal {
    /// Верхняя комната (её пол = эта граница).
    pub upper: usize,
    /// Нижняя комната (её потолок = эта граница).
    pub lower: usize,
    /// Y границы.
    pub y: f64,
    /// Диапазон X проёма.
    pub from: f64,
    pub to: f64,
}

/// Существо пересекло границу внутри проёма → соседняя комната.
/// Иначе точка зажимается внутри текущей комнаты.
pub fn resolve_room(
    rooms: &[Room],
    portals: &[Portal],
    cur: usize,
    x: &mut f64,
    y: &mut f64,
    vy: f64,
) -> usize {
    let r = &rooms[cur];
    let going_down = vy >= 0.0;
    for p in portals {
        let target = if going_down && p.upper == cur && r.y_bot == p.y && *y > p.y {
            Some(p.lower)
        } else if !going_down && p.lower == cur && r.y_top == p.y && *y < p.y {
            Some(p.upper)
        } else {
            None
        };
        if let Some(t) = target {
            if *x >= p.from && *x <= p.to {
                return t;
            }
        }
    }
    *x = x.clamp(r.x0 + BODY_R, r.x1 - BODY_R);
    *y = y.clamp(r.y_top + BODY_R, r.y_bot - BODY_R);
    cur
}

fn floor_open(portals: &[Portal], room: usize, y_bot: f64, x: f64) -> bool {
    portals
        .iter()
        .any(|p| p.upper == room && p.y == y_bot && x >= p.from && x <= p.to)
}

fn ceil_open(portals: &[Portal], room: usize, y_top: f64, x: f64) -> bool {
    portals
        .iter()
        .any(|p| p.lower == room && p.y == y_top && x >= p.from && x <= p.to)
}

// ─── Тело ───────────────────────────────────────────────────────────────────
#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub room: usize,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    /// Смотрит вправо (+1) или влево (−1); задаётся носом снаружи.
    pub facing: f64,
    pub on_ground: bool,
}

impl Body {
    pub fn new(room: usize, x: f64, y: f64) -> Self {
        Self { room, x, y, vx: 0.0, vy: 0.0, facing: 1.0, on_ground: false }
    }

    /// Шаг физики существа: привод («как быстро», знак уже во facing),
    /// прыжок, гравитация, пол/потолок/стены, провалы в проёмы.
    pub fn step(&mut self, rooms: &[Room], portals: &[Portal], drive: f64, jump: f64, dt: f64) {
        if drive > 0.05 {
            self.vx += self.facing * drive.min(1.0) * WALK_SPEED * dt * 8.0;
            self.vx = self.vx.clamp(-WALK_SPEED, WALK_SPEED);
        } else {
            self.vx *= 0.82;
        }
        if jump > 0.6 && self.on_ground {
            self.vy = JUMP_VY * jump.min(1.0);
            self.on_ground = false;
        }

        self.vy += GRAVITY * dt;
        self.x += self.vx * dt;
        self.y += self.vy * dt;

        let r = &rooms[self.room];
        let open_down = floor_open(portals, self.room, r.y_bot, self.x);
        let open_up = ceil_open(portals, self.room, r.y_top, self.x);

        if !open_down && self.y > r.y_bot - BODY_R {
            self.y = r.y_bot - BODY_R;
            if self.vy > 0.0 {
                self.vy = 0.0;
            }
            self.on_ground = true;
        } else if !open_up && self.y < r.y_top + BODY_R {
            self.y = r.y_top + BODY_R;
            if self.vy < 0.0 {
                self.vy = 0.0;
            }
            self.on_ground = false;
        } else {
            self.on_ground = false;
        }

        // Стены комнаты: упёрся — развернулся.
        if self.x <= r.x0 + BODY_R {
            self.x = r.x0 + BODY_R;
            self.vx = 0.0;
            self.facing = -self.facing;
        }
        if self.x >= r.x1 - BODY_R {
            self.x = r.x1 - BODY_R;
            self.vx = 0.0;
            self.facing = -self.facing;
        }

        // Переход через проём по факту пересечения границы.
        self.room = resolve_room(rooms, portals, self.room, &mut self.x, &mut self.y, self.vy);
    }

    /// Плод тоже подчиняется гравитации и проёмам.
    pub fn step_fall(&mut self, rooms: &[Room], portals: &[Portal], dt: f64) {
        self.vy += GRAVITY * dt;
        self.x += self.vx * dt;
        self.y += self.vy * dt;

        let r = &rooms[self.room];
        let open_down = floor_open(portals, self.room, r.y_bot, self.x);
        if !open_down && self.y > r.y_bot - 3.0 {
            self.y = r.y_bot - 3.0;
            self.vy = 0.0;
            self.vx = 0.0;
        }
        self.x = self.x.clamp(r.x0 + 3.0, r.x1 - 3.0);

        self.room = resolve_room(rooms, portals, self.room, &mut self.x, &mut self.y, self.vy);
    }
}

/// Детерминированная сцена: земля + балкон-крона + яма-пещера.
pub fn scene() -> (Vec<Room>, Vec<Portal>) {
    let rooms = vec![
        Room { id: 0, x0: 0.0, x1: 520.0, y_top: 130.0, y_bot: 240.0 }, // земля
        Room { id: 1, x0: 320.0, x1: 520.0, y_top: 30.0, y_bot: 130.0 }, // балкон-крона
        Room { id: 2, x0: 0.0, x1: 110.0, y_top: 240.0, y_bot: 370.0 }, // яма-пещера
    ];
    let portals = vec![
        Portal { upper: 1, lower: 0, y: 130.0, from: 380.0, to: 480.0 }, // балкон → земля
        Portal { upper: 0, lower: 2, y: 240.0, from: 10.0, to: 100.0 },  // земля → яма
    ];
    (rooms, portals)
}
