//! Презентация: снимок сима → спрайты кадра (чистая функция).
//!
//! [`Presenter::new`] один раз резолвит регионы атласа (тело, 4 лица, тень,
//! 4 фазы ягоды, тайлсет пола/стен) и регистрирует процедурные текстуры через
//! генераторы dioma-render (`circle_rgba`, `soft_circle_rgba`, `berry_rgba`,
//! `line_rgba` + собственные пиксельные лица). [`Presenter::build`] превращает
//! [`SimView`] в [`DrawScene`] — детерминированная функция без `&mut` чего-либо.
//!
//! Правила «состояние → спрайт» (этап 1):
//!
//! | Состояние существа      | Тело                          | Лицо              |
//! |-------------------------|-------------------------------|-------------------|
//! | живое, сытое            | tint зелёный, пульс           | спокойное         |
//! | `wants_food`            | tint теплеет (жёлто-зелёный)  | голодное (рот)    |
//! | `fatigue` > 0.7         | как сытое, пульс              | усталое (глаза)   |
//! | мёртвое                 | tint серый, пульс выключен    | мёртвое (X-глаза) |
//!
//! Приоритет: мёртвое > голод > усталость > спокойное.
//!
//! Слои: тайлы [`LAYER_TILE`] (0.0) → тени 1.0 → тела 2.0 → лица 2.1 →
//! ягоды 3.0 → тон суток 4.0 (заполняет хост, см. [`DrawScene::tint`]).
//! Глубина существ `-pos.y` (см. `depth` у спрайта): большее
//! значение рисуется раньше, поэтому существо выше по миру оказывается под
//! нижними — y-сортировка «выше = дальше», согласованно с движком.

use crate::view::snapshot::{CritterView, RoomView, SimView};
use dioma_engine::env::{Substance, SUBSTANCES};
use dioma_render::tilemap::LAYER_TILE;
use dioma_render::{atlas, Sprite, SpriteAtlas, SpriteFx, TilesetUv};

/// Радиус тела существа в мировых единицах (диаметр ~1.2). Пикинг использует
/// тот же радиус (+допуск) — см. [`crate::view::input`].
pub const BODY_RADIUS: f32 = 0.6;

/// Сторона тайла пола/стены в мировых единицах (комнаты выровнены по сетке).
const TILE_SIZE: f32 = 1.0;

/// Диаметр ягоды в мировых единицах (спелая; фаза 0 — 0.7×).
const BERRY_SIZE: f32 = 0.9;

/// Слои (порядок рисования). Тайлы — константа рендерера; остальные — этапа 1.
const LAYER_SHADOW: f32 = 1.0;
const LAYER_BODY: f32 = 2.0;
const LAYER_FACE: f32 = 2.1;
const LAYER_BERRY: f32 = 3.0;

/// Разрешённый регион атласа: UV, смещение обрезки и размер в пикселях.
#[derive(Debug, Clone, Copy)]
pub struct RegionRef {
    pub uv: [f32; 4],
    /// `offset`/`size_px` пока не читаются: все регионы этапа 1 процедурные
    /// (обрезки нет, offset = [0, 0]); поля нужны на этапе 2, когда придут
    /// PNG-ассеты с обрезкой прозрачных полей.
    #[allow(dead_code)]
    pub offset: [f32; 2],
    #[allow(dead_code)]
    pub size_px: [u32; 2],
}

/// Сцена кадра: что и в каком порядке рисовать. Слои — в `Sprite.layer`
/// (см. константы выше), батч сортирует по (layer, depth).
#[derive(Debug, Clone, Default)]
pub struct DrawScene {
    /// Пол комнат + стены (слой [`LAYER_TILE`]).
    pub tiles: Vec<Sprite>,
    /// Мягкие тени под существами.
    pub shadows: Vec<Sprite>,
    /// Тела существ (глубина `-pos.y` для y-сортировки).
    pub bodies: Vec<Sprite>,
    /// Лица поверх тел (по состоянию, см. таблицу в доке модуля).
    pub faces: Vec<Sprite>,
    /// Ягоды: флипбук по фазе роста (4 кадра), спелая чуть крупнее.
    pub berries: Vec<Sprite>,
    /// Полупрозрачный тон мира по фазе суток (слой [`LAYER_TINT`]): заливает
    /// весь вьюпорт — тёплый рассвет, синяя ночь. Заполняет хост (зависит от
    /// камеры), не `Presenter::build`.
    pub tint: Vec<Sprite>,
}

/// Слой тона суток: поверх ягод (3.0), под оверлеем (рендерится отдельно).
/// Публичный — хост задаёт слой тона при построении спрайта.
pub const LAYER_TINT: f32 = 4.0;

/// Кэш регионов атласа + правила «состояние → спрайт».
pub struct Presenter {
    body: RegionRef,
    /// Лица: 0 — спокойное, 1 — голодное, 2 — усталое, 3 — мёртвое.
    faces: [RegionRef; 4],
    shadow: RegionRef,
    /// Фазы ягоды: 0 — незрелая … 3 — спелая.
    berries: [RegionRef; 4],
    /// Тайлсет 1×2: тайл 1 — пол, тайл 2 — стена.
    tiles: TilesetUv,
    /// Сплошной белый квадрат для тона суток (цвет — тинтом на лету).
    tint: RegionRef,
}

impl Presenter {
    /// Регистрирует процедурные регионы в атласе и резолвит их UV/размеры.
    ///
    /// Нужна очередь (а не устройство): все генераторы — чистый CPU,
    /// данные уходят в атлас через `add_rgba(queue, …)`.
    pub fn new(queue: &wgpu::Queue, atlas: &mut SpriteAtlas) -> Self {
        // Тело-круг белый: тинт красится на лету в build.
        let (w, h, px) = atlas::circle_rgba(BODY_TEX_R, [255, 255, 255, 255]);
        atlas.add_rgba(queue, "view_body", w, h, &px).expect("регион тела существа");
        // Тень — мягкий круг.
        let (w, h, px) = atlas::soft_circle_rgba(SHADOW_TEX_R, [12, 16, 24, 255]);
        atlas.add_rgba(queue, "view_shadow", w, h, &px).expect("регион тени");
        // 4 лица: пиксельные примитивы (глаза + рот), без зависимостей.
        for (i, name) in FACE_NAMES.iter().enumerate() {
            let (w, h, px) = face_rgba(i as u32);
            atlas.add_rgba(queue, name, w, h, &px).expect("регион лица");
        }
        // 4 фазы ягоды: цвет и блик меняются к созреванию.
        for (i, &(phase, color)) in BERRY_PHASES.iter().enumerate() {
            let (w, h, px) = atlas::berry_rgba(phase, color);
            atlas.add_rgba(queue, &format!("view_berry_{i}"), w, h, &px)
                .expect("регион ягоды");
        }
        // Тайлсет 16×16: верхняя ячейка — пол, нижняя — стена.
        let (w, h, px) = tileset_rgba();
        atlas.add_rgba(queue, "view_tiles", w, h, &px).expect("регион тайлсета");
        // Тон суток: сплошной белый квадрат (цвет задаёт тинт спрайта).
        let (w, h, px) = tint_rgba();
        atlas.add_rgba(queue, "view_tint", w, h, &px).expect("регион тона");

        let region = |name: &str| {
            let r = atlas.region_info(name).expect("регион обязан быть в атласе");
            RegionRef { uv: r.uv, offset: r.offset, size_px: r.size_px }
        };
        let tiles = TilesetUv::new(atlas.region("view_tiles").expect("тайлсет"), 1, 2);
        Presenter {
            body: region("view_body"),
            faces: [
                region("view_face_calm"),
                region("view_face_hungry"),
                region("view_face_tired"),
                region("view_face_dead"),
            ],
            shadow: region("view_shadow"),
            berries: [
                region("view_berry_0"),
                region("view_berry_1"),
                region("view_berry_2"),
                region("view_berry_3"),
            ],
            tiles,
            tint: region("view_tint"),
        }
    }

    /// UV сплошного белого квадрата — для полноэкранного тона суток
    /// (слой [`LAYER_TINT`]); хост сам задаёт размер/цвет/позицию.
    pub fn tint_uv(&self) -> [f32; 4] {
        self.tint.uv
    }

    /// Чистая функция: снимок → спрайты. Детерминирована.
    pub fn build(&self, view: &SimView) -> DrawScene {
        let mut scene = DrawScene::default();
        for room in &view.rooms {
            self.build_room(room, &mut scene.tiles);
        }
        for c in &view.critters {
            self.build_critter(c, &mut scene);
        }
        for b in &view.berries {
            let frame = ((b.phase * 4.0).floor().max(0.0) as usize).min(3);
            let scale = 0.7 + 0.3 * b.phase; // спелая чуть крупнее
            let size = BERRY_SIZE * scale;
            let mut s = Sprite::new([b.pos.x, b.pos.y], [size, size], self.berries[frame].uv)
                .with_depth(-b.pos.y);
            s.layer = LAYER_BERRY;
            scene.berries.push(s);
        }
        scene
    }

    /// Пол комнаты (тайлы 1) с тинтом плодородности + стены (тайлы 2) по
    /// периметру. Комнаты выровнены по сетке тайлов: колонки/ряды
    /// `floor(min)..ceil(max)`, граница — первый/последний ряд прямоугольника.
    fn build_room(&self, room: &RoomView, out: &mut Vec<Sprite>) {
        let rect = room.rect;
        let x0 = rect.min.x.floor() as i32;
        let x1 = rect.max.x.ceil() as i32;
        let y0 = rect.min.y.floor() as i32;
        let y1 = rect.max.y.ceil() as i32;
        let floor_uv = self.tiles.cell_uv(1);
        let wall_uv = self.tiles.cell_uv(2);
        let tint = floor_tint(room_fertility(&room.ca));
        for ty in y0..y1 {
            for tx in x0..x1 {
                let wall = tx == x0 || tx == x1 - 1 || ty == y0 || ty == y1 - 1;
                let (uv, color) = if wall { (wall_uv, WALL_TINT) } else { (floor_uv, tint) };
                let mut s = Sprite::new(
                    [tx as f32 + 0.5, ty as f32 + 0.5],
                    [TILE_SIZE, TILE_SIZE],
                    uv,
                )
                .with_tint(color);
                s.layer = LAYER_TILE;
                out.push(s);
            }
        }
    }

    /// Существо: тень, тело (окрас из генома × состояние), лицо поверх.
    ///
    /// Живые сигналы движка → визуал:
    /// * мёртвое — серый + `Grayscale`, без пульса;
    /// * спит — приглушённый, медленный пульс, лицо «усталое» (глаза закрыты);
    /// * дофамин + активность мозга — золотое свечение (`Glow`): съел —
    ///   вспыхнул, думать — слегка светится;
    /// * кортизол > 0.7 — болезненная дрожь (`Wobble`).
    fn build_critter(&self, c: &CritterView, scene: &mut DrawScene) {
        let pos = [c.pos.x, c.pos.y];
        let life = if c.alive { 1.0 } else { 0.6 };
        let diameter = BODY_RADIUS * 2.0 * (0.8 + 0.4 * life);
        let depth = -c.pos.y;

        // Тень: сплюснутый мягкий круг чуть ниже тела.
        let mut sh = Sprite::new(
            [pos[0], pos[1] - diameter * 0.35],
            [diameter * 1.35, diameter * 0.6],
            self.shadow.uv,
        )
        .with_depth(depth);
        sh.layer = LAYER_SHADOW;
        scene.shadows.push(sh);

        // Тело: окрас из ДНК как база (0.35..0.95 — без пересвета), сверху —
        // модификаторы состояния. Мёртвое — серое и обесцвеченное.
        let coat = [
            0.35 + c.coat[0] * 0.6,
            0.35 + c.coat[1] * 0.6,
            0.35 + c.coat[2] * 0.6,
        ];
        let mut tint = if c.alive {
            let warm = c.wants_food;
            [
                coat[0] + if warm { 0.25 } else { 0.0 },
                coat[1] + if warm { -0.05 } else { 0.05 },
                coat[2] - if warm { 0.05 } else { 0.0 },
                1.0,
            ]
        } else {
            [0.52, 0.52, 0.56, 1.0]
        };
        if c.sleeping {
            // Спит: притушить и дышать медленно.
            for t in &mut tint[..3] {
                *t *= 0.7;
            }
        }
        let mut body = Sprite::new(pos, [diameter, diameter], self.body.uv).with_tint(tint);
        body.layer = LAYER_BODY;
        body.depth = depth;
        if c.sleeping {
            body.pulse = [0.99, 1.01, 0.8, c.pos.x]; // медленное «дыхание» сна
        } else if c.alive {
            body.pulse = [0.96, 1.04, 2.2, c.pos.x]; // фаза — от позиции
        }
        // Эффекты: мёртвое — серость; живое — свечение радости/мысли, дрожь стресса.
        if !c.alive {
            body.fx = SpriteFx::Grayscale { amount: 1.0 };
        } else {
            let glow = (c.dopamine * 0.85 + c.spike_rate * 0.3).clamp(0.0, 1.0);
            if glow > 0.12 {
                body.fx = SpriteFx::Glow {
                    radius: diameter * 0.9,
                    intensity: glow,
                    color: [1.0, 0.85, 0.4, 1.0],
                };
            } else if c.cortisol > 0.7 {
                body.fx = SpriteFx::Wobble { amplitude: 1.2, frequency: 0.35, speed: 9.0 };
            }
        }
        scene.bodies.push(body);

        // Лицо: мёртвое > спит (глаза закрыты) > голод > усталость > спокойное.
        let face = if !c.alive {
            3
        } else if c.sleeping {
            2
        } else if c.wants_food {
            1
        } else if c.fatigue > 0.7 {
            2
        } else {
            0
        };
        let fs = diameter * 0.62;
        let mut f = Sprite::new(pos, [fs, fs], self.faces[face].uv).with_depth(depth);
        f.layer = LAYER_FACE;
        scene.faces.push(f);
    }
}

// ─── Константы текстур ─────────────────────────────────────────────────────

/// Радиус текстуры тела в пикселях (в атласе; масштаб — мировой, при зумах
/// ~16..24 px на тело текстура ~26 px — без мыла).
const BODY_TEX_R: u32 = 12;
/// Радиус текстуры тени.
const SHADOW_TEX_R: u32 = 16;

/// Имена регионов лиц (индекс = номер лица).
const FACE_NAMES: [&str; 4] = ["view_face_calm", "view_face_hungry", "view_face_tired", "view_face_dead"];

/// Фазы ягоды: (поворот блика, цвет тела) — незрелая (зелёная) → спелая (красная).
const BERRY_PHASES: [(f32, [u8; 4]); 4] = [
    (0.0, [130, 175, 95, 255]),
    (1.7, [170, 195, 85, 255]),
    (3.4, [215, 140, 70, 255]),
    (5.1, [225, 65, 75, 255]),
];

/// Тинт стены (тёмный камень) — умножается на текстуру тайла.
const WALL_TINT: [f32; 4] = [0.72, 0.70, 0.78, 1.0];

/// Плодородность комнаты 0..1: тепло + свет + питание, «полный» уровень —
/// 200 (потолок в тестах движка и стартовой сцене).
fn room_fertility(ca: &[f32; SUBSTANCES]) -> f32 {
    let s = ca[Substance::Temperature.idx()] + ca[Substance::Light.idx()] + ca[Substance::Nutrients.idx()];
    (s / (3.0 * 200.0)).clamp(0.0, 1.0)
}

/// Тинт пола: сухая земля (плодородность 0) → сочная трава (1).
fn floor_tint(fert: f32) -> [f32; 4] {
    let dry = [0.62, 0.50, 0.32, 1.0];
    let lush = [0.24, 0.60, 0.26, 1.0];
    let mut t = [0.0; 4];
    for i in 0..3 {
        t[i] = dry[i] + (lush[i] - dry[i]) * fert;
    }
    t[3] = 1.0;
    t
}

// ─── Процедурные пиксели (лица и тайлсет) ─────────────────────────────────

/// Лицо 12×12: глаза + рот вариантами. 0 — спокойное (точки-глаза, короткий
/// рот), 1 — голодное (рот открыт), 2 — усталое (глаза-линии), 3 — мёртвое
/// (X-глаза, плоский рот).
fn face_rgba(face: u32) -> (u32, u32, Vec<u8>) {
    const S: u32 = 12;
    const INK: [u8; 4] = [28, 30, 40, 255];
    let mut img = vec![0u8; (S * S * 4) as usize];
    match face {
        1 => {
            // Голодное: глаза открыты, рот открыт.
            dot(&mut img, S, 3, 4, 1, INK);
            dot(&mut img, S, 8, 4, 1, INK);
            fill_rect(&mut img, S, 4, 8, 8, 9, INK);
        }
        2 => {
            // Усталое: глаза-линии (закрыты), маленький рот.
            hline(&mut img, S, 2, 4, 4, INK);
            hline(&mut img, S, 7, 9, 4, INK);
            hline(&mut img, S, 4, 7, 8, INK);
        }
        3 => {
            // Мёртвое: X-глаза, плоский рот.
            line(&mut img, S, 2, 3, 4, 5, INK);
            line(&mut img, S, 4, 3, 2, 5, INK);
            line(&mut img, S, 7, 3, 9, 5, INK);
            line(&mut img, S, 9, 3, 7, 5, INK);
            hline(&mut img, S, 3, 8, 8, INK);
        }
        _ => {
            // Спокойное: точки-глаза, короткая улыбка.
            dot(&mut img, S, 3, 4, 1, INK);
            dot(&mut img, S, 8, 4, 1, INK);
            hline(&mut img, S, 4, 7, 8, INK);
        }
    }
    (S, S, img)
}

/// Тайлсет 16×16 (1 колонка × 2 ряда ячеек 8×8): тайл 1 — пол (трава с
/// крапинками), тайл 2 — стена (тёмный камень со швами).
fn tileset_rgba() -> (u32, u32, Vec<u8>) {    const S: u32 = 16;
    let hash = |x: u32, y: u32, seed: u32| -> u32 {
        let mut h = x.wrapping_mul(374_761_393).wrapping_add(y.wrapping_mul(668_265_263)).wrapping_add(seed);
        h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
        (h ^ (h >> 16)) & 0xff
    };
    let mut img = vec![0u8; (S * S * 4) as usize];
    for y in 0..S {
        for x in 0..S {
            let i = ((y * S + x) * 4) as usize;
            let c: [u8; 4] = if y < 8 {
                // Пол: зелёная база + перепады оттенка по хэшу (детерминированно).
                let n = hash(x, y, 11);
                [30 + (n % 24) as u8, 96 + (n % 30) as u8, 32 + (n % 18) as u8, 255]
            } else {
                // Стена: камень со швами по краям ячейки.
                let edge = if x == 7 || y == 15 { 30 } else { 0 };
                [52 + edge / 2, 48 + edge / 2, 58 + edge / 2, 255]
            };
            img[i..i + 4].copy_from_slice(&c);
        }
    }
    (S, S, img)
}

/// Сплошной белый квадрат 4×4: цвет тона суток задаётся тинтом спрайта
/// (`view_tint`), чтобы не плодить текстуры на каждый оттенок.
fn tint_rgba() -> (u32, u32, Vec<u8>) {
    const S: u32 = 4;
    let img = vec![255u8; (S * S * 4) as usize];
    (S, S, img)
}

// ─── Пиксельные примитивы (в буфер S×S RGBA8) ──────────────────────────────

/// Один пиксель (за пределами кадра игнорируется).
fn px(img: &mut [u8], s: u32, x: i32, y: i32, c: [u8; 4]) {
    if x >= 0 && y >= 0 && (x as u32) < s && (y as u32) < s {
        let i = ((y as u32 * s + x as u32) * 4) as usize;
        img[i..i + 4].copy_from_slice(&c);
    }
}

/// Залитый круг радиуса r (центр cx,cy).
fn dot(img: &mut [u8], s: u32, cx: i32, cy: i32, r: i32, c: [u8; 4]) {
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                px(img, s, cx + dx, cy + dy, c);
            }
        }
    }
}

/// Горизонтальная линия (y фиксирован).
fn hline(img: &mut [u8], s: u32, x0: i32, x1: i32, y: i32, c: [u8; 4]) {
    for x in x0..=x1 {
        px(img, s, x, y, c);
    }
}

/// Залитый прямоугольник (границы инклюзивны).
fn fill_rect(img: &mut [u8], s: u32, x0: i32, x1: i32, y0: i32, y1: i32, c: [u8; 4]) {
    for y in y0..=y1 {
        for x in x0..=x1 {
            px(img, s, x, y, c);
        }
    }
}

/// Отрезок (пошаговое линейное приближение Брезенхэма-в-стиле).
fn line(img: &mut [u8], s: u32, x0: i32, y0: i32, x1: i32, y1: i32, c: [u8; 4]) {
    let steps = (x1 - x0).abs().max((y1 - y0).abs()) as i32;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = (x0 as f32 + (x1 - x0) as f32 * t).round() as i32;
        let y = (y0 as f32 + (y1 - y0) as f32 * t).round() as i32;
        px(img, s, x, y, c);
    }
}
