# Мост «сим ↔ рендер» — план API

> Документ: `dioma/docs/RENDER_SIM_BRIDGE.md`
> Крейты: `dioma-engine` (чистая симуляция) ↔ `dioma-render` (wgpu 22) — через новый модуль `view` в корневом крейте `dioma` и игровой бинарь.

---

## 0. Принципы

1. **`dioma-engine` остаётся чистым**: ноль зависимостей на wgpu/window/рендер.
   Уже так — lib.rs движка: «движок не зависит от окна или графического API».
   Всё, что движок отдаёт наружу, — обычные данные (`Vec2`, `AgentId`, числа).
2. **Мост = модуль `view` в корневом крейте `dioma`** + игровой бинарь `src/bin/game.rs`.
   Корневой крейт получает dep `dioma-render = { path = "crates/dioma-render" }` и
   `winit = "0.30"` (версия, уже проверенная в демо dioma-render).
3. **Снимок вместо сосуществования**: за кадр строим дешёвую read-only проекцию
   арены (`SimView`), рендерим по ней. Никаких `&mut Arena` во время рендера,
   детерминизм, replay-friendly.
4. **Фиксированный тик** (25 мс), accumulator в игровом цикле: `while acc >= TICK { arena.step() }`.
5. **egui отложен**: eframe 0.35 тащит свою wgpu 24+, наш рендерер — wgpu 22,
   общий device невозможен. Оверлей на этом этапе — `DebugFont` (уже в dioma-render).

---

## 1. Граф зависимостей

```
                ┌──────────────────────┐
                │  dioma (корневой)    │
                │  src/view/*          │  ← МОСТ
                │  src/bin/game.rs     │
                └──┬───────────────┬───┘
                   │               │
        dioma-engine              dioma-render
        (сим, чистый)             (wgpu 22, готов)
                   │               │
                   └──   SiMuse  ──┘
```

Никаких обратных стрелок: `dioma-render` не знает о симе, `dioma-engine` не знает о рендере.

---

## 2. Поток данных и игровой цикл

```
loop {
    // 1) Накопление времени
    accumulator += dt;
    while accumulator >= TICK { arena.step(); accumulator -= TICK; }

    // 2) Проекция сима → данные для рисования (без заимствований арены)
    let view = SimView::capture(&arena);

    // 3) Презентация: данные → спрайты/тайлы (Presenter, кэш регионов)
    let scene = presenter.build(&view, &camera, &tick_fraction);

    // 4) Рендер слоями: параллакс → тайлмап комнат → существа → ягоды
    //    → fx/частицы → пост → оверлей (FPS, тик, выбранное существо)

    // 5) Ввод: клики (screen→world→pick), клавиши (пауза/скорость/спавн)
    for cmd in input.drain_commands(&camera) { game.apply(cmd, &mut arena); }
}
```

---

## 3. API: проекция сима — `view/snapshot.rs`

Дешёвое зеркало арены, строится каждый кадр.

```rust
/// Read-only проекция арены для одного кадра.
pub struct SimView {
    pub rooms:   Vec<RoomView>,
    pub critters: Vec<CritterView>,
    pub berries: Vec<BerryView>,
}

pub struct RoomView {
    pub idx:  usize,
    pub rect: Rect,          // env::map::Rect (min/max) — мировые координаты
    pub ca:   [f32; N],      // концентрации веществ (плодородность → цвет пола)
}

pub struct CritterView {
    pub id:       AgentId,
    pub pos:      Vec2,
    pub heading:  f32,
    pub alive:    bool,
    pub speed:    f32,       // creature.speed()
    pub hunger:   f32,       // 0..1 нормировано
    pub glucose:  f32,       // 0..1
    pub fatigue:  f32,       // 0..1
    pub wants_food: bool,
    pub food_eaten: u32,
}

pub struct BerryView {
    pub id:    AgentId,
    pub pos:   Vec2,
    pub phase: f32,          // 0..1: 1 = спелая (vars[1] <= 0), иначе 1 - vars[1]/berry_regrow
}

impl SimView {
    pub fn capture(arena: &Arena) -> Self;   // копирование; O(N) по существам/ягодам
}
```

Почему копия, а не ссылки: рендер требует `&mut Renderer` и не должен держать
`&Arena`; снимок снимает конфликт и делает слой презентации чистой функцией.

---

## 4. API: презентация — `view/present.rs`

Превращает `SimView` в спрайты. Все регионы атласа резолвятся один раз в `new`.

```rust
/// Кэш регионов атласа + правила «состояние → спрайт».
pub struct Presenter {
    body:    RegionRef,   // круг-тело существа (процедурный)
    face:    [RegionRef; 4], // лица: сыт / хочет еды / устал / мёртв
    shadow:  RegionRef,   // мягкая тень
    berry:   [RegionRef; 4], // фазы роста ягоды (флипбук)
    tiles:   TilesetUv,   // пол/стены комнат
    atlas:   ...          // ссылка на атлас для SpriteBatch::add
}

/// Сцена кадра: что и в каком порядке рисовать.
pub struct DrawScene {
    pub tiles: Vec<Sprite>,      // слой LAYER_TILE (пол комнат + стены)
    pub shadows: Vec<Sprite>,    // под существами
    pub bodies: Vec<Sprite>,     // существа (depth = -pos.y для y-сортировки)
    pub berries: Vec<Sprite>,
}

impl Presenter {
    /// Создаёт/дополняет атлас процедурными регионами (круги, лица, ягоды).
    pub fn new(device: &wgpu::Device, atlas: &mut SpriteAtlas) -> Self;

    /// Чистая функция: снимок → спрайты. Детерминирована.
    pub fn build(&self, view: &SimView) -> DrawScene;
}
```

Правила «состояние → спрайт» (этап 1):

| Состояние существа | Спрайт |
|---|---|
| живое, сытое | тело-круг, tint зелёный, лицо «спокойное» |
| `wants_food` | лицо «голодное» (открытый рот), tint теплеет |
| `fatigue` > 0.7 | лицо «устал» (глаза закрыты) |
| мёртвое | tint серый, лицо «мёртв», pulse выключен |

- Размер тела: базовый радиус × (0.8 + 0.4·фаза жизни) — пока константа.
- `depth = -pos.y` — существо выше по миру рисуется под нижними (y-сортировка уже в движке).
- Пульс-дыхание: `with_pulse` базового шейдера (TIME уже есть).
- Ягоды: флипбук по `phase` (4 кадра), спелая чуть крупнее + блик.

---

## 5. API: ввод — `view/input.rs`

```rust
pub enum GameCommand {
    TogglePause,
    Speed(u32),                 // 1/2/4 — тиков на кадр (множитель)
    Select(Option<AgentId>),    // клик по существу / пустоте
    SpawnCritter(Vec2),         // клик с Shift — отладочный спавн
    SpawnBerry(Vec2),
}

pub struct InputMapper {
    cursor_world: Vec2,         // screen→world (Camera::screen_to_world)
    // ...
}

impl InputMapper {
    pub fn on_cursor(&mut self, screen: [f32; 2], camera: &Camera);
    pub fn on_click(&mut self, shift: bool) -> Option<GameCommand>;
    pub fn on_key(&mut self, key: Key) -> Option<GameCommand>;
}
```

Пикинг — **чистая функция в dioma-engine** (тестируется headless):

```rust
// sim/mod.rs (добавить)
impl Arena {
    /// Ближайшая цель в радиусе: существо (круг r) или ягода (квадрат).
    /// Сначала существа, потом ягоды; None — мимо всех.
    pub fn pick(&self, world: Vec2, r: f32) -> Option<AgentId>;
}
```

`r` для курсора = радиус существа × zoom (экранный допуск ~8 px → мировые единицы).

---

## 6. API: бинарь — `src/bin/game.rs`

Паттерн — как демо dioma-render (winit 0.30 ApplicationHandler, wgpu 22),
плюс смоук-режим `--frames=N` (проверенный на демо).

```rust
struct Game {
    // wgpu-окружение (как в демо) + device/queue
    renderer: Renderer,
    atlas:    SpriteAtlas,
    presenter: Presenter,
    // симуляция
    arena:    Arena,
    // состояние
    camera:   Camera,
    input:    InputMapper,
    selected: Option<AgentId>,
    paused:   bool,
    speed:    u32,
    accumulator: f32,
}
```

Сцена по умолчанию (этап 1):
- одна комната 40×14 (мировые единицы), две-три комнаты соединённые дверями;
- 3 существа (разные seed), 6 ягод;
- камера: центр комнаты, zoom ~2;
- клавиши: `Space` пауза, `1/2/3` скорость, `Shift+клик` спавн существа, `Esc` снять выделение;
- оверлей (DebugFont): FPS, тик, пауза/скорость, выбранное существо (id, голод, глюкоза, усталость).

---

## 7. Регионы атласа (что рисуем)

Процедурные генераторы dioma-render (уже есть): `circle`, `soft_circle`, `ring`,
`berry`, `bar`. Новые в демо/презентере:
- `critter_body` — круг с обводкой (tint-окрашиваемый),
- `face_*` — 3-4 лица поверх тела (глаза/рот),
- `berry_1..4` — фазы роста (уже есть `berry`, сделать 4 размера),
- тайлсет пола: трава/земля по плодородности комнаты (tint по `ca`).

PNG-ассеты: `SpriteAtlas::load_folder` — позднее, когда появятся художники.

---

## 8. Зафиксированные решения (дефолты, можно менять)

| Вопрос | Решение | Почему |
|---|---|---|
| Где мост | модуль `view` в корневом `dioma` | не плодим крейты; бинарь рядом |
| egui-оверлей | отложен | конфликт версий wgpu (22 vs 24+) |
| Снимок vs ссылки | копия `SimView` | чистый слой презентации |
| Тик | 25 мс фикс., accumulator | уже в дизайн-доке |
| Ориентация | Y вверх = верх экрана | уже исправлено в dioma-render |
| Пикинг | `Arena::pick` в движке | headless-тест, без рендера |
| Сортировка | слой `LAYER_TILE` → тела с `depth = -pos.y` | y-сортировка уже в движке |

---

## 9. Этапы

### Этап 1 — скелет моста (этот заход)
- `view/{mod,snapshot,present,input}.rs` в `dioma`; `Arena::pick`; bin `game`.
- Сцена: комната-тайлмап, 3 существа (круги с лицами), 6 ягод (флипбук).
- Клик-выбор + оверлей; пауза/скорость; смоук `--frames=N`.
- **Тесты**: `pick` (ближняя/мимо), `SimView::capture` (поля копируются верно),
  `Presenter::build` (кол-во спрайтов = 3 тела + 3 тени + 6 ягод + тайлы; голодное
  существо → другое лицо), смоук бинаря на 150 кадров.

### Этап 2 — жизнь на экране
- сон/возраст/смерть → визуал (моргание, размер, серость);
- инспектор выбранного: потребности-бары (bar-регионы), счётчик еды, память мест;
- двери между комнатами визуально, переходы существ.

### Этап 3 — эволюция на экране
- поколения (evolution), статистика в оверлее, spawn по клику новых существ;
- регуляция скорости: автопауза при вымирании.

### Этап 4 — полировка
- egui-оверлей (решение по wgpu: либо апгрейд рендерера до wgpu 24, либо
  egui-wgpu 0.35 в отдельном проходе поверх);
- PNG-ассеты существ/комнат, звук, туториал.

---

## 10. Риски

1. **Два wgpu** (22 vs 24/25 у egui 0.35) — egui-этап потребует решения: апгрейд
   dioma-render (ломает выверенные 37 тестов) или отказ от egui в пользу DebugFont
   (потеря красивых UI). Отложено сознательно.
2. **Производительность снимка**: O(N) копий на кадр — для сотен существ ок;
   для тысяч — оптимизировать (кэш по `tick`, инвалидация).
3. **`SimView` расходится с симуляцией** при неточном accumulator — тик-доля
   (`tick_fraction`) для интерполяции движения позже.
4. **Атлас растёт** — авто-рост уже есть (cap 8192), лимит — на совести ассетов.
