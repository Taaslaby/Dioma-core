//! -------------------------------------------------------------------------
//! Filename:    src/sim/mod.rs
//! Class:       sim
//! Purpose:     АРЕНА — композиция среды, мира объектов и существ
//! Description: существа (creature::Creature) живут в комнатах (env::map),
//!               еду им дают СКРИПТОВЫЕ ягоды (world::World: слот EAT →
//!               «хрум» → die). Ягоды растут от плодородности комнаты
//!               (питательные вещества × свет из CA) и дают потомство.
//!               Слой связывает независимые модули и больше ничего не знает.
//! author: Atlasov anton
//! -------------------------------------------------------------------------
use crate::creature::{Action, Creature, Sense};
use crate::brain::gpu_pool::BrainPool;
use crate::brain::neurons::BRAIN_STEPS_PER_TICK;
use crate::brain::perf::flush_denormals;
use crate::env::map::WorldMap;
use crate::env::math::Vec2;
use crate::env::{AgentId, Season, Substance, TimeOfDay, WorldClock};
use crate::world::{Cls, World, SLOT_EAT};
use rayon::prelude::*;
use std::f32::consts::TAU;
use std::sync::OnceLock;

pub mod berry;
pub mod evolution;

pub use berry::{BerryInfo, BerryState};
pub use evolution::{Evolution, EvolutionParams, GenerationStats};

/// Радиус «рта»: на этом расстоянии еда считается съедобной.
pub const EAT_RADIUS: f32 = 2.0;
/// Радиус, в котором существо «видит» соседа за едой (социальное обучение).
const SOCIAL_RADIUS: f32 = 10.0;
/// Максимальный поворот за тик (радианы). (×2: тик 50 мс — за секунду
/// поворот тот же, что при 25 мс.)
pub const MAX_TURN: f32 = 0.6;
/// Масштаб шага движения за тик. (×2: тик 50 мс — скорость тела задана
/// «за тик»; удвоением шага за секунду проходится та же дистанция.)
const MOVE_STEP_PER_TICK: f32 = 2.0;
/// Затухание запаха с дистанцией: 60/(60+d).
const SMELL_FALLOFF: f32 = 60.0;
/// Дальность зрения (мировых юнитов): вклад = (1 − d/VISION_RANGE), на краю
/// диапазона — 0. На игровой карте комнаты 40×14: зрение «достаёт» почти
/// до дальней стены комнаты, где запаховый градиент уже не различает стороны
/// (зрение длиннее обоняния и ловит ягоду, которую несёт ветер мимо носа;
/// стены и комнаты по-прежнему непрозрачны — та же проверка комнаты, что
/// у запаха).
const VISION_RANGE: f32 = 40.0;
/// Половина поля зрения (рад): ±75°. Ягода вне конуса не видна, даже если
/// близко.
const VISION_HALF_FOV: f32 = 75.0 * std::f32::consts::PI / 180.0;
/// Размер клетки памяти места (грубая привязка «где была еда»).
const PLACE_CELL: f32 = 8.0;
/// Размер клетки климата (тепло/свет/питание диффундируют по сетке).
const CLIMATE_CELL: f32 = 4.0;
/// Дальность ощущения стены (юнитов) и шаг пробы.
const WALL_LOOKAHEAD: f32 = 8.0;
const WALL_PROBE_STEP: f32 = 1.0;
/// Порог «стена прямо по курсу»: обе диагональные пробы (±45°) должны
/// упереться в одну и ту же стену ВПЕРЕДИ (боковая стена попадает только
/// в одну пробу и не считается «прямо по курсу»).
const WALL_AHEAD_THRESHOLD: f32 = 0.5;

// ─── Сон ────────────────────────────────────────────────────────────────
/// Порог усталости, с которого существо засыпает. FATIGUE-химия крови
/// живёт в 0..255 (потолок концентраций) с распадом 0.990025/тик и накоплением
/// от моторной активности; рецептор SPEED к этому моменту уже заметно
/// замедлил тело (~40% от максимальной скорости) — «отработал смену».
/// Выше порога (ближе к 255) охотник выматывается слишком часто — замерено
/// на эволюционном прогоне: при 70 существа спали 85% времени и голодали.
///
/// НЕ менялся на тике 50 мс: накопление утомления за тик удваивается, но
/// по той же причине удваивается RATE_MOTOR (спайки моторов за тик — 100
/// подшагов мозга вместо 50), а распад пересчитан в квадрат (0.990025 =
/// 0.995²). Равновесие P/(1−d) и кривая F(t) в реальном времени остаются
/// теми же. После ребаланса экономики утомления (гейн RATE_MOTOR→FATIGUE
/// снижен ×8, см. metabolism.rs) порог 200 даёт бодрые окна ~30–50 тиков
/// при сне 12 — доля сна в замкнутом контуре ~0.25, а не ~0.75 как до
/// ребаланса (существа спали 2–4 тика бодрствования против 12 сна).
const SLEEP_AT_FATIGUE: f32 = 200.0;
/// Длительность сна в тиках (12 × 50 мс ≈ 0.6 с): пауза, в которую хост
/// держит существо неподвижным. Сама консолидация дня происходит ОДНИМ
/// вызовом `Creature::sleep()` на входе — длительность лишь задаёт «тишину»
/// после него. 12 — верхняя граница, найденная замером: при 30 тиках
/// выученные охотники (высокий моторный выход → частый сон) не успевали
/// объехать L-стену за 800 тиков и эволюция голодала.
/// (÷2: тик 50 мс — 12 × 50 мс ≈ 25 × 25 мс ≈ 0.6 с.)
const SLEEP_TICKS: u32 = 12;

// ─── Новорождённый арены ─────────────────────────────────────────────────
/// Голод новорождённого арены: свежий организм (HUNGER ≈ 0) начал бы охоту
/// только через сотни тиков распада глюкозы — новорождённый арены рождается
/// голодным (как NEWBORN_HUNGER=200 в эволюции): иначе первый приём пищи
/// недостижим (замкнутый круг новичка, см. исследование эмерджентности).
/// 120 > порога wants_food (40) с запасом, но ниже панического порога 240.
pub const ARENA_NEWBORN_HUNGER: f32 = 120.0;

// ─── Ягода: класс (1, 2) — «съедобное» ────────────────────────────────────
pub const BERRY_FAMILY: u8 = 1;
pub const BERRY_GENUS: u8 = 2;
/// Скрипт ягоды на EAT: хрумнуть (не умирает — еда возобновляемая).
pub const BERRY_EATEN: &str = "out \"хрум\"";
/// Тиков «отрастания» съеденной ягоды: vars[1] > 0 = ягода пустая,
/// не пахнет и не съедобна, пока счётчик не дойдёт до нуля. Без этого
/// сильные охотники истребляют урожай до размножения ягод и голодают —
/// а food_eaten перестаёт быть мерой навыка.
/// (÷2: тик 50 мс — 75 × 50 мс ≈ 150 × 25 мс ≈ 3.75 с.)
const BERRY_REGROW_TICKS: f32 = 75.0;
/// Рост за тик: vars[0] += плодородность × GROW_RATE.
/// (×2: тик 50 мс — за секунду ягода растёт с той же скоростью.)
const GROW_RATE: f32 = 2.0;
/// Накопленный рост, при котором ягода даёт потомство.
const SPAWN_AT: f32 = 12.0;
/// Питательные вещества комнаты, съедаемые за тик роста (рост не бесплатен).
/// (×2: тик 50 мс)
const NUTRIENT_COST: f32 = 0.04;
/// Разброс потомства вокруг родителя.
const SPAWN_SPREAD: f32 = 3.0;
/// Потолок популяции ягод (защита от экспоненты).
const MAX_BERRIES: usize = 24;

/// Существо в арене: позиция, курс и сам контроллер.
pub struct Critter {
    pub id: AgentId,
    pub pos: Vec2,
    /// Курс в радианах (0 = +X).
    pub heading: f32,
    pub creature: Creature,
    /// Сколько раз существо ударилось о стену (счётчик для наблюдения/тестов).
    pub wall_hits: u64,
    /// Видел ли соседа за едой на прошлом такте (социальное обучение:
    /// заполняет sense.see_eat на следующий тик и сбрасывается).
    pub see_eat: bool,
    /// Спит ли: во сне нет сенсоров/мозга/движения/еды, метаболизм стоит —
    /// только единичный `creature.sleep()` на входе (см. фазу сна в `step`).
    pub sleeping: bool,
    /// Оставшиеся тики сна (тикает вниз в `Arena::step`; 0 = бодрствует).
    pub sleep_ticks: u32,
    /// Зрение прошлого такта (0..1, левый/правый глаз-бин): копия
    /// `Sense::{vis_l, vis_r}` для оверлеев и интеграционных тестов
    /// (проводка зрения жива). У спящего — 0.
    pub vis_l: f32,
    pub vis_r: f32,
}

/// Арена: мир объектов (ягоды, карта) + существа. Детерминирована по сиду.
pub struct Arena {
    pub world: World,
    pub critters: Vec<Critter>,
    pub tick: u64,
    /// Следующий свободный id существа (эволюция выделяет id новорождённым).
    pub(crate) next_id: u64,
    /// Тиков отрастания съеденной ягоды (можно тюнить через эволюцию).
    pub berry_regrow: f32,
    /// Климат мира: тепло/свет/питание диффундируют по сетке поверх карты,
    /// ягоды растут там, где тепло+светло+сытно.
    pub climate: crate::env::Climate,
    /// Сенсоры последнего `step()` в порядке `critters` (None = спящий).
    ///
    /// Наблюдательный мост: рекордер «Шептуна» (examples/whisperer_data.rs)
    /// читает через [`Self::last_sense`] ровно тот [`Sense`], из которого
    /// существо приняло действие последнего тика. Чисто аддитивно: фаза A
    /// `step` уже собирала эти сенсоры (раньше они молча дропались) — поле
    /// только копирует их до конца тика; на симуляцию не влияет.
    last_senses: Vec<Option<Sense>>,
    /// Батч-пул GPU-динамики (см. `brain::gpu_pool::BrainPool`): `Some` —
    /// арена в пулевом режиме (`Arena::new_pooled`): мембрана ВСЕХ существ
    /// считается одним GPU-шагом за тик, пластичность — на CPU. `None` —
    /// обычный CPU-путь. Спавн на пулевой арене (`Arena::spawn`) автоматически
    /// регистрирует существо в пуле.
    pool: Option<BrainPool>,
}

impl Arena {
    pub fn new(seed: u64, map: WorldMap) -> Self {
        let climate = crate::env::Climate::over(&map, CLIMATE_CELL as f64, prometheus_grow::DiffusionBackend::cpu());
        Arena {
            world: World::new(seed, map),
            critters: Vec::new(),
            tick: 0,
            next_id: 1,
            berry_regrow: BERRY_REGROW_TICKS,
            climate,
            last_senses: Vec::new(),
            pool: None,
        }
    }

    /// Арена в пулевом GPU-режиме: мембрана всех существ — общий батч-раннер
    /// (`BrainPool`), пластичность — на CPU. Опт-ин (по умолчанию игра идёт
    /// на CPU): без адаптера пул недоступен — арена тихо падает на CPU-путь
    /// (как `GpuDynamics::try_new`), спавн `spawn_pooled` строит обычных
    /// существ. Спавнить на пулевой арене — через `spawn`/`spawn_pooled`
    /// (регистрируют существо в пуле); НЕ смешивать с не-пулевыми существами.
    pub fn new_pooled(seed: u64, map: WorldMap) -> Self {
        match BrainPool::try_new() {
            Ok(pool) => {
                let mut a = Arena::new(seed, map);
                a.pool = Some(pool);
                a
            }
            Err(e) => {
                eprintln!("[dioma] GPU-пул недоступен ({e}), арена на CPU");
                Arena::new(seed, map)
            }
        }
    }

    pub fn spawn(&mut self, pos: Vec2, heading: f32, creature_seed: u64) -> AgentId {
        if self.pool.is_some() {
            // Пулевая арена: существо регистрируется в общем GPU-раннере.
            self.spawn_pooled(pos, heading, creature_seed)
        } else {
            self.spawn_creature(pos, heading, Creature::new(creature_seed))
        }
    }

    /// Пулевой спавн: существо с мозгом `Dynamics::Pooled` (мембрана — общий
    /// GPU-раннер арены). На CPU-арене (пул недоступен) — обычное существо.
    pub fn spawn_pooled(&mut self, pos: Vec2, heading: f32, creature_seed: u64) -> AgentId {
        match &mut self.pool {
            Some(pool) => {
                let c = Creature::new_pooled(creature_seed, pool);
                self.spawn_creature(pos, heading, c)
            }
            None => self.spawn(pos, heading, creature_seed),
        }
    }

    /// Пул активен? (диагностика/примеры/GPU-тесты: SKIP без адаптера.)
    pub fn has_pool(&self) -> bool {
        self.pool.is_some()
    }

    /// Посадить готовое существо (контрольные протоколы: `lr = 0` и т.п.).
    ///
    /// Новорождённый арены голоден (ARENA_NEWBORN_HUNGER): это точка сборки
    /// «стандартной» крови существ — иначе дефолтный голод 0 (кровь
    /// `Blood::creature(false)`) вместе с сон-петлёй (глюкоза после сна не
    /// поднимается выше SLEEP_GLUCOSE_CAP) держал бы новорождённого сытым
    /// и закрытый контур на еде не замыкался. Эволюция поверх этого ставит
    /// СВОЙ голод 200 (NEWBORN_HUNGER, см. evolution.rs) — сильнее и до
    /// первого тика: двойного накопления нет, применяется одно значение.
    pub fn spawn_creature(&mut self, pos: Vec2, heading: f32, mut creature: Creature) -> AgentId {
        creature.blood.conc[crate::biology::metabolism::chem::HUNGER] = ARENA_NEWBORN_HUNGER;
        let id = AgentId(self.next_id);
        self.next_id += 1;
        self.critters.push(        Critter {
            id,
            pos,
            heading,
            creature,
            wall_hits: 0,
            see_eat: false,
            sleeping: false,
            sleep_ticks: 0,
            vis_l: 0.0,
            vis_r: 0.0,
        });
        id
    }

    /// Посадить ягоду: скриптовый агент мира со слотом EAT («хрум» → die).
    pub fn add_berry(&mut self, pos: Vec2) -> AgentId {
        let id = self.world.spawn(Cls::new(BERRY_FAMILY, BERRY_GENUS, 1), pos, "berry");
        self.world
            .attach_script(id, SLOT_EAT, BERRY_EATEN)
            .expect("EAT-скрипт ягоды");
        id
    }

    /// Живая ягода готова к еде (не в стадии отрастания).
    fn berry_edible(ag: &crate::world::Agent) -> bool {
        ag.alive && ag.cls.family == BERRY_FAMILY && ag.cls.genus == BERRY_GENUS && ag.vars[1] <= 0.0
    }

    /// Живая ягода без проверки съедобности (для роста/счётчиков).
    fn berry_any(ag: &crate::world::Agent) -> bool {
        ag.alive && ag.cls.family == BERRY_FAMILY && ag.cls.genus == BERRY_GENUS
    }

    /// Живые ягоды в мире (включая отрастающие — они занимают место).
    pub fn berries(&self) -> Vec<AgentId> {
        self.world
            .all_ids()
            .iter()
            .filter(|&&id| self.world.agent(id).map(Self::berry_any).unwrap_or(false))
            .copied()
            .collect()
    }

    /// Состояние живой ягоды по id (None — не ягода или не жива).
    ///
    /// Именованный доступ для хоста и моста рендера; внутренние переменные
    /// скрипта (`vars[0]`/`vars[1]`) скрыты за состоянием [`BerryInfo`]
    /// (см. `super::berry::BerryState` для операций над ним).
    pub fn berry_info(&self, id: AgentId) -> Option<BerryInfo> {
        let ag = self.world.agent(id)?;
        if !Self::berry_any(ag) {
            return None;
        }
        Some(BerryInfo {
            id,
            pos: ag.pos,
            state: BerryState {
                growth: ag.vars[0],
                regrow: ag.vars[1],
            },
        })
    }

    /// Ближайшая цель под точкой: существо (круг радиуса r_critter) или ягода
    /// (радиус r_berry). Сначала существа, потом ягоды; None — мимо всех.
    ///
    /// Существа приоритетнее ягод даже если ягода ближе; существо берётся
    /// любое (живое или мёртвое — оба видимы и выбираемы). Ягода — живая
    /// (включая отрастающие: пустой куст тоже можно выделить).
    pub fn pick(&self, world: Vec2, r_critter: f32, r_berry: f32) -> Option<AgentId> {
        let mut best_c: Option<(AgentId, f32)> = None;
        for c in &self.critters {
            let d = c.pos.dist(world);
            if d <= r_critter && best_c.map_or(true, |(_, bd)| d < bd) {
                best_c = Some((c.id, d));
            }
        }
        if let Some((id, _)) = best_c {
            return Some(id);
        }
        let mut best_b: Option<(AgentId, f32)> = None;
        for &id in &self.berries() {
            let Some(ag) = self.world.agent(id) else { continue };
            let d = ag.pos.dist(world);
            if d <= r_berry && best_b.map_or(true, |(_, bd)| d < bd) {
                best_b = Some((id, d));
            }
        }
        best_b.map(|(id, _)| id)
    }

    /// Сенсор, из которого существо `i` приняло действие последнего `step()`
    /// (None — существа нет или оно спало в тот тик).
    ///
    /// Наблюдательный мост без изменения горячего пути: `step` и так собирает
    /// сенсоры в фазе A — метод отдаёт их копию из кэша [`Arena::last_senses`],
    /// а не пересчитывает. Семантика «что существо ВИДЕЛО, когда решало
    /// последнее действие» — ровно то, что нужно рекордерам офлайн-обучения.
    pub fn last_sense(&self, i: usize) -> Option<&Sense> {
        self.last_senses.get(i).and_then(|s| s.as_ref())
    }

    /// Один такт: мир (сообщения/скрипты/физика) → сон → существа → еда → рост.
    ///
    /// Горячий путь параллелизован (rayon, см. [`sim_pool`]): фазы A (сенсоры)
    /// и B (тикание мозгов) — независимые существа — считаются на пуле потоков;
    /// фаза C (движение/еда/социальное/климат/рост ягод) остаётся
    /// последовательной — она мутирует мир/карту и требует порядка. Детерминизм
    /// бит-в-бит: тик каждого существа выполняется ЦЕЛИКОМ на одном потоке
    /// (нет реасоциации float между потоками), порядок пачки действий
    /// сохраняется (rayon collect по индексам), фаза C не тронута.
    pub fn step(&mut self) {
        self.tick += 1;
        // 0. Мир: доставка сообщений (в т.ч. вчерашний EAT → «хрум» → die),
        //    таймеры, скрипты и физика комнат.
        self.world.step();

        // 0.5. Сон: уставшие (fatigue ≥ SLEEP_AT_FATIGUE) засыпают и спят
        //    SLEEP_TICKS тиков; спящий не получает сенсоры/мозг/движение/еду
        //    и не участвует в зеркале — метаболизм во сне стоит (только
        //    единичный `creature.sleep()` на входе: восстановление тела +
        //    консолидация выученного дня). Будит только таймер — раннего
        //    пробуждения при критической глюкозе нет (простота; см. tranche 1).
        //    Проснулся — действует с этого тика.
        for c in &mut self.critters {
            if c.sleeping {
                if c.sleep_ticks == 0 {
                    c.sleeping = false;
                } else {
                    c.sleep_ticks -= 1;
                    continue;
                }
            }
            if c.creature.fatigue() >= SLEEP_AT_FATIGUE {
                c.sleeping = true;
                c.sleep_ticks = SLEEP_TICKS;
                c.creature.sleep();
            }
        }

        // 1. Сенсоры (по живым ягодам мира + память о соседе за едой) → решения.
        //    Только бодрствующие: спящие не чувствуют и не действуют.
        //
        //    Фаза A — ПАРАЛЛЕЛЬНО (rayon): sense читает только world и critters
        //    (иммутабельно; поля арены расщеплены — borrow-split `critters` +
        //    `world`, климата/счётчиков фаза A не касается). Каждый критик
        //    независим → результат не зависит от планировщика; порядок
        //    collect сохраняется (бит-в-бит как последовательный проход).
        //    FTZ (MXCSR) — состояние ПОТОКА: воркеры пула мозг не создавали,
        //    поэтому каждая closure выставляет флаг вручную (см. brain::perf:
        //    без FTZ STDP-следы в денормалях замедляют SIMD ~×10 и меняют
        //    семантику распада). 2 инструкции, дёшево.
        let actions: Vec<(usize, Action)> = sim_pool().install(|| {
            let senses: Vec<Option<Sense>> = {
                let Arena { critters, world, .. } = self;
                critters
                    .par_iter()
                    .map(|c| {
                        flush_denormals();
                        if c.sleeping {
                            None
                        } else {
                            Some(Self::sense_world(world, c.pos, c.heading, c.see_eat))
                        }
                    })
                    .collect()
            };
            for c in &mut self.critters {
                c.see_eat = false; // окно наблюдения закрылось
            }
            //    Фаза B — ПАРАЛЛЕЛЬНО: каждый критик тикает СВОЙ мозг (свой
            //    ext/spike_hist/кровь/гиппокамп). Мир не читается — сенсоры
            //    уже материализованы в фазе A, так что critters — единственный
            //    mutable-займ (disjoint per element). Спящие не получают Action
            //    ровно как в последовательном проходе (пачка содержит только
            //    бодрствующих в порядке индексов; filter_map порядок хранит).
            //
            //    ПУЛЕВАЯ ФАЗА B (арена из `new_pooled`): мембрана ВСЕХ существ
            //    считается ОДНИМ GPU-шагом (`BrainPool::step`), пластичность/
            //    награды — по каждому существу на CPU. Семантика та же: ext
            //    строится параллельно (фаза 1), пул гонит сеть и скачивает
            //    историю спайков (фаза 2), каждый критик глотает СВОЙ блок
            //    спайков и доигрывает тик (фаза 3). Детерминизм run-vs-run:
            //    тик каждого существа — целиком на одном потоке, порядок
            //    пачки сохранён; GPU-арифметика может отличаться от CPU в
            //    1 ulp (допускается, см. отчёт), но не от запуска к запуску.
            let actions: Vec<(usize, Action)> = {
                let Arena { critters, pool, .. } = self;
                match pool {
                    Some(pool) => {
                        // Фаза 1: ext каждого бодрствующего (параллельно).
                        let reliefs: Vec<Option<f32>> = critters
                            .par_iter_mut()
                            .enumerate()
                            .map(|(i, c)| {
                                flush_denormals();
                                match &senses[i] {
                                    Some(sense) => {
                                        c.vis_l = sense.vis_l;
                                        c.vis_r = sense.vis_r;
                                        Some(c.creature.prepare_tick(sense))
                                    }
                                    None => {
                                        c.vis_l = 0.0;
                                        c.vis_r = 0.0;
                                        None
                                    }
                                }
                            })
                            .collect();
                        // Фаза 2: dirty-зеркала весов → GPU, один шаг сети.
                        {
                            let mirrors: Vec<(&[f32], bool)> = critters
                                .iter()
                                .map(|c| c.creature.brain.pooled_weights())
                                .collect();
                            if let Err(e) = pool.sync_weights(&mirrors) {
                                eprintln!("[dioma] пул: заливка весов — {e}");
                            }
                        }
                        let spt = critters
                            .first()
                            .map(|c| c.creature.brain.steps_per_tick())
                            .unwrap_or(BRAIN_STEPS_PER_TICK);
                        let exts: Vec<Option<&[f32]>> = critters
                            .iter()
                            .enumerate()
                            .map(|(i, c)| {
                                if senses[i].is_some() {
                                    Some(c.creature.pooled_ext())
                                } else {
                                    None
                                }
                            })
                            .collect();
                        let (flat, slots) = pool.step(&exts, spt);
                        for c in &mut **critters {
                            c.creature.brain.pooled_clear_dirty();
                        }
                        // Фаза 3: каждый критик — свой блок спайков + тело.
                        critters
                            .par_iter_mut()
                            .enumerate()
                            .filter_map(|(i, c)| {
                                flush_denormals();
                                match (&senses[i], &reliefs[i]) {
                                    (Some(sense), Some(relief)) => Some((
                                        i,
                                        c.creature.tick_pooled(
                                            sense,
                                            *relief,
                                            flat,
                                            spt,
                                            slots[i].neuron_off,
                                        ),
                                    )),
                                    _ => None,
                                }
                            })
                            .collect()
                    }
                    None => critters
                        .par_iter_mut()
                        .enumerate()
                        .filter_map(|(i, c)| {
                            flush_denormals();
                            match &senses[i] {
                                Some(sense) => {
                                    // Снимок зрения для оверлеев/тестов.
                                    c.vis_l = sense.vis_l;
                                    c.vis_r = sense.vis_r;
                                    Some((i, c.creature.tick(sense)))
                                }
                                None => {
                                    // Спящий не видит: снимок обнуляется.
                                    c.vis_l = 0.0;
                                    c.vis_r = 0.0;
                                    None
                                }
                            }
                        })
                        .collect(),
                }
            };
            // Сенсоры этого тика прожили фазы A/B — сохранить для наблюдателей
            // (последний шаг перед дропом: раньше они молча уничтожались).
            self.last_senses = senses;
            actions
        });

        // 2. Движение по комнатам (стены/переходы — физика мира).
        //    Удар о стену = кандидат вне всех комнат: позиция зажимается на
        //    край, а существу сообщается удар (боль + негативный дофамин).
        for &(i, ref action) in &actions {
            let c = &mut self.critters[i];
            c.heading = (c.heading + action.steer * MAX_TURN).rem_euclid(TAU);
            let dir = Vec2::new(c.heading.cos(), c.heading.sin());
            // (×2: тик 50 мс) — шаг за тик удвоен: за секунду та же дистанция.
            let step = dir * (action.drive * c.creature.speed() * MOVE_STEP_PER_TICK);
            let cand = c.pos + step;
            if self.world.map.room_at(cand).is_some() {
                c.pos = cand;
            } else if let Some(room) = self.world.map.room_at(c.pos) {
                let r = &self.world.map.rooms[room.0 as usize].rect;
                c.pos = Vec2::new(
                    cand.x.clamp(r.min.x, r.max.x),
                    cand.y.clamp(r.min.y, r.max.y),
                );
                c.creature.hit_wall();
                c.wall_hits += 1;
            }
        }

        // 3. Еда: рот открыт → послать ягоде сообщение EAT (хрумнёт на след. такте).
        let mut eaters: Vec<Vec2> = Vec::new();
        for &(i, ref action) in &actions {
            if !action.eat {
                continue;
            }
            let pos = self.critters[i].pos;
            // Съедобная (не отрастающая) ягода в радиусе рта: хрумнуть,
            // ягода уходит в стадию отрастания (vars[1] = таймер).
            if let Some(&bid) = self
                .berries()
                .iter()
                .find(|&&id| {
                    self.world.agent(id).map(Self::berry_edible).unwrap_or(false)
                        && self.world.agent(id).unwrap().pos.dist(pos) <= EAT_RADIUS
                })
            {
                self.world.send(AgentId::NONE, bid, SLOT_EAT, 0);
                if let Some(ag) = self.world.agent_mut(bid) {
                    ag.vars[1] = self.berry_regrow;
                }
                eaters.push(pos);
            }
        }
        // Социальное обучение: бодрствующие соседи В ТОЙ ЖЕ КОМНАТЕ в радиусе
        // видят едущего и получают зеркальный сигнал на следующий такт
        // (see_eat → sense.see_eat). Стены гасят зеркало: еда за стеной
        // «не видна»; двери — пока нет (транш 2).
        if !eaters.is_empty() {
            for c in &mut self.critters {
                if c.sleeping || c.see_eat {
                    continue;
                }
                let my_room = self.world.map.room_at(c.pos);
                if eaters.iter().any(|&e| {
                    self.world.map.room_at(e) == my_room && e.dist(c.pos) <= SOCIAL_RADIUS
                }) {
                    c.see_eat = true;
                }
            }
        }

        // 3.5. Климат: комнаты кормят поле, диффузия разносит тепло/свет/
        //    питание по миру — ягоды растут там, где стало тепло. Свет
        //    гасится по фазе суток (день/ночь, как в C3): ночью мир темнеет
        //    и плодородность (тепло × свет × питание) падает — рост ягод
        //    следует за сутками.
        self.climate.seed_from_rooms(&self.world.map);
        self.climate.step();
        self.climate.dim_light(crate::env::WorldClock::light_multiplier(self.tick));

        // 4. Рост ягод от плодородности климата (тепло × свет × питание).
        //    Съеденные ягоды отрастают: vars[1] тикает вниз к нулю.
        //    Сначала собираем состояние (только чтение), потом мутируем —
        //    чтобы не держать заимствование мира.
        let mut growth: Vec<(AgentId, Vec2, f32, f32)> = Vec::new();
        for &id in self.world.all_ids() {
            let Some(ag) = self.world.agent(id) else {
                continue;
            };
            if !Self::berry_any(ag) {
                continue;
            }
            let fert = self.fertility(ag.pos);
            if fert > 0.0 || ag.vars[1] > 0.0 {
                growth.push((id, ag.pos, fert, ag.vars[1]));
            }
        }
        for (id, pos, fert, regrow) in growth {
            // Съеденная ягода отрастает: таймер тикает вниз.
            if regrow > 0.0 {
                if let Some(ag) = self.world.agent_mut(id) {
                    ag.vars[1] = (ag.vars[1] - 1.0).max(0.0);
                }
            }
            let room = self.world.map.room_at(pos);
            let grown = {
                let ag = self.world.agent_mut(id).expect("ягода жива");
                ag.vars[0] += fert * GROW_RATE;
                let done = ag.vars[0] >= SPAWN_AT;
                if done {
                    ag.vars[0] = 0.0;
                }
                done
            };
            // Рост съедает питание комнаты.
            if let Some(rid) = room {
                let ca = &mut self.world.map.rooms[rid.0 as usize].ca[Substance::Nutrients.idx()];
                *ca = (*ca - NUTRIENT_COST).max(0.0);
            }
            // Потомство рядом с родителем (детерминированный разброс).
            if grown && self.berries().len() < MAX_BERRIES {
                let offset = deterministic_jitter(id.0, self.tick, SPAWN_SPREAD);
                let b = self.world.spawn(
                    Cls::new(BERRY_FAMILY, BERRY_GENUS, 1),
                    pos + offset,
                    "berry",
                );
                let _ = self.world.attach_script(b, SLOT_EAT, BERRY_EATEN);
            }
        }
    }

    /// Обоняние + зрение + направленное ощущение стен: запах ягод, зрительный
    /// конус и близость стены слева/справа относительно курса + «еда у рта».
    /// Сенсоры существа: запах ягод (живые), зрение (съедобные в конусе,
    /// VISION_RANGE), стены, еда у рта, память места и зеркальный сигнал
    /// `see_eat` (видел соседа за едой на прошлом такте).
    ///
    /// Зрение длиннее обоняния и ловит ягоду, которую несёт ветер мимо носа:
    /// запах рулит градиентом (сила × sin пеленга — вперёд по курсу его
    /// почти нет), зрение — честным конусом ±VISION_HALF_FOV с линейным
    /// затуханием до VISION_RANGE. Стены и комнаты по-прежнему непрозрачны:
    /// ягода в другой комнате не видна (та же проверка комнаты, что у запаха).
    ///
    /// (Тестовый мост: горячий путь зовёт [`Self::sense_world`].)
    #[cfg(test)]
    fn sense(&self, pos: Vec2, heading: f32, see_eat: bool) -> Sense {
        Self::sense_world(&self.world, pos, heading, see_eat)
    }

    /// Тело `sense` над одним `&World` (иммутабельно). Вызывается и из
    /// обёртки `&self`, и из параллельной фазы A `Arena::step` (borrow-split:
    /// воркеры rayon видят только `world` — климат/счётчики не читаются).
    fn sense_world(world: &World, pos: Vec2, heading: f32, see_eat: bool) -> Sense {
        // Комната существа: запах и зрение не проходят сквозь стены — ягода
        // видна/пахнет только из той же комнаты; двери — пока нет (транш 2).
        // На невалидной карте (точка вне комнат) фильтр не включается —
        // старое поведение.
        let my_room = world.map.room_at(pos);
        let mut l = 0.0f32;
        let mut r = 0.0f32;
        let mut lv = 0.0f32;
        let mut rv = 0.0f32;
        let mut near = false;
        for &id in world.all_ids() {
            let Some(ag) = world.agent(id) else { continue };
            // Отрастающие ягоды не пахнут и не «у рта» — иначе существо
            // залипало бы у пустой ягоды.
            if !Self::berry_edible(ag) {
                continue;
            }
            // Ягода за стеной (в другой комнате) не пахнет и не видна —
            // проверка комнаты дешевле дистанции/atan2 и отсекает сквозные
            // сигналы сразу.
            if let Some(room) = my_room {
                if world.map.room_at(ag.pos) != Some(room) {
                    continue;
                }
            }
            let d = ag.pos - pos;
            let dist = d.len();
            let strength = (SMELL_FALLOFF / (SMELL_FALLOFF + dist)).clamp(0.0, 1.0);
            if dist <= EAT_RADIUS {
                near = true;
            }
            let mut rel = d.y.atan2(d.x) - heading;
            rel = (rel + std::f32::consts::PI).rem_euclid(TAU) - std::f32::consts::PI;
            if rel < 0.0 {
                l += strength * (-rel.sin()).max(0.0);
            } else {
                r += strength * rel.sin().max(0.0);
            }
            // Зрение: та же ягода, но честный конус (|пеленг| ≤ VISION_HALF_FOV)
            // и линейное затухание по дистанции. Пеленг ровно по курсу (0)
            // уходит в правый бин — как у запаха (rel < 0 → левый).
            if rel.abs() <= VISION_HALF_FOV {
                let vis = (1.0 - dist / VISION_RANGE).clamp(0.0, 1.0);
                if rel < 0.0 {
                    lv += vis;
                } else {
                    rv += vis;
                }
            }
        }

        // Стены: пробы вбок (±90°) и по диагонали вперёд (±45°) — стена прямо
        // по курсу попадает в обе диагональные пробы, боковая — в свою.
        let fwd = Vec2::new(heading.cos(), heading.sin());
        let left = Vec2::new(fwd.y, -fwd.x);
        let right = Vec2::new(-fwd.y, fwd.x);
        let ahead_l = Vec2::new(
            (heading - std::f32::consts::FRAC_PI_4).cos(),
            (heading - std::f32::consts::FRAC_PI_4).sin(),
        );
        let ahead_r = Vec2::new(
            (heading + std::f32::consts::FRAC_PI_4).cos(),
            (heading + std::f32::consts::FRAC_PI_4).sin(),
        );
        let wall_l = Self::wall_probe_world(&world.map, pos, left)
            .max(Self::wall_probe_world(&world.map, pos, ahead_l));
        let wall_r = Self::wall_probe_world(&world.map, pos, right)
            .max(Self::wall_probe_world(&world.map, pos, ahead_r));

        // Память места: грубая клетка карты (квадрат PLACE_CELL) + курс.
        let cx = (pos.x / PLACE_CELL).floor() as i64;
        let cy = (pos.y / PLACE_CELL).floor() as i64;
        let place = (cx.wrapping_mul(7349) ^ cy.wrapping_mul(7919)) as u32;
        Sense {
            smell_l: l.min(1.0),
            smell_r: r.min(1.0),
            vis_l: lv.min(1.0),
            vis_r: rv.min(1.0),
            wall_l,
            wall_r,
            food_near: near,
            // Стена прямо по курсу: обе диагональные пробы высоки (см.
            // WALL_AHEAD_THRESHOLD). Боковая стена даёт только одну — не «прямо».
            wall: wall_l > WALL_AHEAD_THRESHOLD && wall_r > WALL_AHEAD_THRESHOLD,
            place,
            heading,
            see_eat,
        }
    }

    /// Тело `wall_probe` над одной картой (см. [`Self::sense_world`]).
    fn wall_probe_world(map: &WorldMap, pos: Vec2, dir: Vec2) -> f32 {
        let mut d = WALL_PROBE_STEP;
        while d <= WALL_LOOKAHEAD {
            if map.room_at(pos + dir * d).is_none() {
                return 1.0 - (d / WALL_LOOKAHEAD);
            }
            d += WALL_PROBE_STEP;
        }
        0.0
    }

    /// Плодородность точки: тепло × свет × питание (диффузия
    /// уже разнесла их по миру — растения растут и вдали от источника,
    /// пока к ним дошло тепло).
    fn fertility(&self, pos: Vec2) -> f32 {
        self.climate.fertility(pos.x, pos.y)
    }

    // ─── Мировые часы (день/ночь и сезоны, как в C3) ──────────────────────

    /// Фаза суток из тика арены: Рассвет/Утро/День/Вечер/Ночь.
    /// Формулы — `env::world_clock` (C3 `World::GetTimeOfDay`).
    pub fn time_of_day(&self) -> TimeOfDay {
        WorldClock::time_of_day(self.tick)
    }

    /// День внутри сезона (0..=2), из тика арены.
    pub fn day_in_season(&self) -> u32 {
        WorldClock::day_in_season(self.tick)
    }

    /// Сезон (Весна/Лето/Осень/Зима), из тика арены.
    pub fn season(&self) -> Season {
        WorldClock::season(self.tick)
    }

    /// Прогресс внутри фазы суток 0..1 (для плавных сумерек в рендере).
    pub fn day_phase_fraction(&self) -> f32 {
        WorldClock::day_phase_fraction(self.tick)
    }
}

/// Пул rayon для параллельных фаз A/B `Arena::step` (лениво, один на процесс).
///
/// Число потоков — `std::thread::available_parallelism()` (как у глобального
/// пула rayon). Отладка/сравнение с последовательным проходом:
/// `DIOMA_SINGLE_THREAD=1` собирает пул на 1 поток — код тот же (closures
/// фаз A/B не меняются), поэтому семантика идентична многопоточной; при
/// 1 критике/1 потоке rayon не подмешивает никакой гонки — результат тот же.
///
/// Собственный пул (а не глобальный) — чтобы держать число потоков и их
/// имена под контролем и не зависеть от `RAYON_NUM_THREADS` процесса.
fn sim_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = if std::env::var("DIOMA_SINGLE_THREAD").map(|v| v == "1").unwrap_or(false) {
            1
        } else {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        };
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("dioma-sim-{i}"))
            .build()
            .expect("пул rayon для симуляции")
    })
}

/// Детерминированный разброс потомства из (id родителя, такт) — без ГСЧ.
fn deterministic_jitter(a: u64, b: u64, spread: f32) -> Vec2 {
    let h = |s: u64| -> f32 {
        let x = s.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ s.rotate_right(17);
        ((x >> 40) & 0xFFFF) as f32 / 65536.0 * 2.0 - 1.0
    };
    Vec2::new(
        h(a ^ b) * spread,
        h(a.rotate_left(23) ^ b.rotate_left(7)) * spread,
    )
}

#[cfg(test)]
#[path = "sim_tests.rs"]
mod tests;
