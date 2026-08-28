//! Спрайт — один инстанс текстурированного квада + per-sprite эффекты.

use crate::scene::atlas::SpriteAtlas;
use crate::scene::flipbook::Flipbook;

/// Визуальный эффект спрайта (рисуется отдельным шейдером из `src/shaders/fx/`).
///
/// Каждый вариант раскладывается в два vec4 fx-параметров ([`SpriteFx::params`]),
/// которые кладутся в инстанс-данные и читаются шейдером эффекта:
///
/// | Вариант     | fx_a (первый vec4)                  | fx_b (второй vec4)     |
/// |-------------|-------------------------------------|------------------------|
/// | `None`      | `(0, 0, 0, 0)`                      | `(0, 0, 0, 0)`         |
/// | `Outline`   | `(width, 0, 0, 0)`                  | цвет контура (RGBA)    |
/// | `Glow`      | `(radius, intensity, 0, 0)`         | цвет свечения (RGBA)   |
/// | `Dissolve`  | `(progress, edge_width, 0, 0)`      | цвет границы (RGBA)    |
/// | `Wobble`    | `(amplitude, frequency, speed, 0)`  | `(0, 0, 0, 0)`         |
///
/// Единицы: `width` — тексели атласа; `radius`/`intensity` — мировые пиксели;
/// `progress` — 0..1; `edge_width` — доля хэш-пространства (0..1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpriteFx {
    /// Без эффекта — базовый шейдер `shaders/sprite.wgsl`.
    None,
    /// Контур/подсветка выделения вокруг края спрайта (`fx/outline.wgsl`).
    Outline {
        /// Ширина контура в текселях атласа.
        width: f32,
        /// Цвет контура (RGBA, straight alpha).
        color: [f32; 4],
    },
    /// Внешнее свечение — радиальное кольцо у краёв квада (`fx/glow.wgsl`).
    Glow {
        /// Радиус кольца свечения в мировых пикселях.
        radius: f32,
        /// Интенсивность 0..1+ (аддитивно добавляется к цвету).
        intensity: f32,
        /// Цвет свечения (RGB; alpha не используется).
        color: [f32; 4],
    },
    /// Растворение/постепенная смерть (`fx/dissolve.wgsl`).
    Dissolve {
        /// Прогресс 0..1: 0 — цел, 1 — полностью растворён.
        progress: f32,
        /// Ширина светлой полосы у границы растворения (0..1, доля хэш-пространства).
        edge_width: f32,
        /// Цвет полосы границы (RGBA).
        edge_color: [f32; 4],
    },
    /// Мягкая drop-тень силуэта (`fx/shadow.wgsl`).
    Shadow {
        /// Смещение тени в мировых пикселях (в uv-пространстве; y = вверх мира).
        offset: [f32; 2],
        /// Радиус размытия края тени в текселях атласа.
        radius: f32,
        /// Цвет тени (обычно полупрозрачный чёрный).
        color: [f32; 4],
    },
    /// Обесцвечивание с управляемой силой (`fx/grayscale.wgsl`) — мёртвые/
    /// выключенные объекты (см. RENDER_SIM_BRIDGE: «мёртвое → серое»).
    Grayscale {
        /// Сила 0..1: 0 — без изменений, 1 — полностью серый.
        amount: f32,
    },
    /// Волнистость/болезненная дрожь (`fx/wobble.wgsl`).
    Wobble {
        /// Амплитуда смещения вершин по X в мировых пикселях.
        amplitude: f32,
        /// Частота волны (на мировую единицу Y).
        frequency: f32,
        /// Скорость бега волны (фаза в секунду).
        speed: f32,
    },
}

impl SpriteFx {
    /// Числовой идентификатор эффекта: 0=None, 1=Outline, 2=Glow, 3=Dissolve, 4=Wobble.
    /// Используется рендерером для выбора пайплайна (индекс `id − 1`).
    pub fn id(&self) -> u32 {
        match self {
            SpriteFx::None => 0,
            SpriteFx::Outline { .. } => 1,
            SpriteFx::Glow { .. } => 2,
            SpriteFx::Dissolve { .. } => 3,
            SpriteFx::Wobble { .. } => 4,
            SpriteFx::Shadow { .. } => 5,
            SpriteFx::Grayscale { .. } => 6,
        }
    }

    /// Параметры эффекта как два vec4 — таблица раскладки в доке [`SpriteFx`].
    pub fn params(&self) -> ([f32; 4], [f32; 4]) {
        match *self {
            SpriteFx::None => ([0.0; 4], [0.0; 4]),
            SpriteFx::Outline { width, color } => ([width, 0.0, 0.0, 0.0], color),
            SpriteFx::Glow { radius, intensity, color } => ([radius, intensity, 0.0, 0.0], color),
            SpriteFx::Dissolve { progress, edge_width, edge_color } => {
                ([progress, edge_width, 0.0, 0.0], edge_color)
            }
            SpriteFx::Wobble { amplitude, frequency, speed } => {
                ([amplitude, frequency, speed, 0.0], [0.0; 4])
            }
            SpriteFx::Shadow { offset, radius, color } => {
                ([offset[0], offset[1], radius, 0.0], color)
            }
            SpriteFx::Grayscale { amount } => ([amount, 0.0, 0.0, 0.0], [0.0; 4]),
        }
    }
}

/// Спрайт: мировые координаты, размер, поворот, тинт, UV в атласе,
/// пульс, глубина и визуальный эффект.
///
/// * `position` — центр квада в мировых координатах (пиксели);
/// * `size` — размер квада в мировых координатах (пиксели);
/// * `rotation` — поворот вокруг центра, радианы;
/// * `tint` — множитель цвета (RGBA, straight alpha);
/// * `uv_rect` — прямоугольник в атласе в нормализованных UV: `[u0, v0, u1, v1]`;
/// * `layer` — z-порядок: меньше — рисуется раньше (сзади);
/// * `depth` — глубина для Y-сортировки в сайд-вью: выше значение — рисуется
///   позже (ближе к зрителю). Обычно = позиция по Y мира, если не задано явно.
///   Внутри одного слоя батч сортируется сначала по `depth`;
/// * `pulse` — дыхание масштаба `[min_scale, max_scale, speed, phase]`
///   (см. [`Sprite::with_pulse`]);
/// * `fx` — per-sprite визуальный эффект (см. [`SpriteFx`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sprite {
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub rotation: f32,
    /// Tint у ВЕРХА квада (экранная ориентация; при градиенте — верхний цвет).
    pub tint: [f32; 4],
    /// Tint у НИЗА квада (вертикальный градиент; равен `tint`, если градиента нет).
    pub tint_end: [f32; 4],
    pub uv_rect: [f32; 4],
    pub layer: f32,
    pub depth: f32,
    pub pulse: [f32; 4],
    pub fx: SpriteFx,
}

impl Default for Sprite {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0],
            size: [32.0, 32.0],
            rotation: 0.0,
            tint: [1.0, 1.0, 1.0, 1.0],
            tint_end: [1.0, 1.0, 1.0, 1.0],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            layer: 0.0,
            depth: 0.0,
            pulse: [1.0, 1.0, 0.0, 0.0],
            fx: SpriteFx::None,
        }
    }
}

impl Sprite {
    /// Спрайт с заданными позицией, размером и UV-прямоугольником.
    pub fn new(position: [f32; 2], size: [f32; 2], uv_rect: [f32; 4]) -> Self {
        Self {
            position,
            size,
            uv_rect,
            ..Self::default()
        }
    }

    /// Пульс (дыхание) размера: масштаб квада колеблется между `min_scale`
    /// и `max_scale` по закону `0.5 + 0.5·sin(time·speed + phase)`.
    /// Значения по умолчанию `[1, 1, 0, 0]` — пульса нет.
    pub fn with_pulse(mut self, min_scale: f32, max_scale: f32, speed: f32, phase: f32) -> Self {
        self.pulse = [min_scale, max_scale, speed, phase];
        self
    }

    /// Визуальный эффект спрайта (контур, свечение, растворение, дрожь).
    pub fn with_fx(mut self, fx: SpriteFx) -> Self {
        self.fx = fx;
        self
    }

    /// Множитель цвета (RGBA, straight alpha).
    pub fn with_tint(mut self, tint: [f32; 4]) -> Self {
        self.tint = tint;
        self.tint_end = tint;
        self
    }

    /// Вертикальный градиент tint'а: `top` у верхнего края, `bottom` — у
    /// нижнего (экранная ориентация). Полезно для неба, воды, панелей UI.
    pub fn with_gradient(mut self, top: [f32; 4], bottom: [f32; 4]) -> Self {
        self.tint = top;
        self.tint_end = bottom;
        self
    }

    /// Поворот вокруг центра, радианы.
    pub fn with_rotation(mut self, rotation: f32) -> Self {
        self.rotation = rotation;
        self
    }

    /// Размер квада в мировых пикселях.
    pub fn with_size(mut self, size: [f32; 2]) -> Self {
        self.size = size;
        self
    }

    /// Устанавливает UV-прямоугольник по имени региона атласа.
    ///
    /// Если региона с таким именем нет, `uv_rect` остаётся без изменений
    /// (спрайт нарисуется с текущим прямоугольником — обычно весь атлас).
    pub fn with_texture(mut self, atlas: &SpriteAtlas, name: &str) -> Self {
        if let Some(uv) = atlas.region(name) {
            self.uv_rect = uv;
        }
        self
    }

    /// Глубина для Y-сортировки в сайд-вью: выше значение — рисуется позже
    /// (ближе к зрителю). Обычно = позиция по Y мира, если не задано явно.
    /// Внутри одного слоя батч сортируется сначала по `depth` (стабильно).
    pub fn with_depth(mut self, depth: f32) -> Self {
        self.depth = depth;
        self
    }

    /// Устанавливает UV-прямоугольник спрайта по кадру флипбука в момент
    /// `time` и возвращает себя (builder). Спрайт остаётся `Copy`: приложение
    /// само продвигает время и вызывает `with_frame` перед push в батч.
    pub fn with_frame(mut self, flipbook: &Flipbook, time: f32) -> Self {
        self.uv_rect = flipbook.frame_at(time);
        self
    }

    /// Стабильная сортировка спрайтов для отрисовки: сначала по слою
    /// (меньше — раньше/сзади), затем по глубине Y (меньше — раньше).
    ///
    /// Одинаковые `(layer, depth)` сохраняют порядок push — удобно для
    /// вручную выстроенных последовательностей (например, глаз поверх тела).
    pub fn sort_for_draw(sprites: &mut [Sprite]) {
        sprites.sort_by(|a, b| {
            a.layer
                .total_cmp(&b.layer)
                .then(a.depth.total_cmp(&b.depth))
        });
    }

    /// Легаси-алиас: стабильная сортировка только по слою (меньше — сзади).
    /// Сейчас это тонкая обёртка над [`Sprite::sort_for_draw`] — глубины
    /// внутри слоя всё равно сортируются; для старого поведения задавайте
    /// всем спрайтам одинаковую глубину.
    pub fn sort_by_layer(sprites: &mut [Sprite]) {
        Self::sort_for_draw(sprites);
    }
}
