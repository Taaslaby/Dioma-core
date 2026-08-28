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

/// Читаемое состояние ягоды для хоста и моста рендера.
///
/// Обёртка над внутренними переменными скриптовой ягоды (`vars[0]`/`vars[1]`,
/// которые по-прежнему читают/пишут скрипты): growth — накопленный рост
/// (vars[0]); regrow — таймер отрастания (vars[1]); edible — regrow <= 0
/// и ягода жива.
#[derive(Clone, Copy, Debug)]
pub struct Berry {
    pub id: AgentId,
    pub pos: Vec2,
    /// Накопленный рост (vars[0]): плодородность × GROW_RATE за тик;
    /// при SPAWN_AT ягода даёт потомство и обнуляется.
    pub growth: f32,
    /// Таймер отрастания съеденной ягоды (vars[1]): тикает вниз к нулю;
    /// > 0 = ягода пустая (не пахнет, не съедобна).
    pub regrow: f32,
    /// Съедобна ли: regrow <= 0 и жива.
    pub edible: bool,
}

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
    /// скрипта (`vars[0]`/`vars[1]`) скрыты за полями — см. [`Berry`].
    pub fn berry_info(&self, id: AgentId) -> Option<Berry> {
        let ag = self.world.agent(id)?;
        if !Self::berry_any(ag) {
            return None;
        }
        Some(Berry {
            id,
            pos: ag.pos,
            growth: ag.vars[0],
            regrow: ag.vars[1],
            edible: ag.vars[1] <= 0.0,
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
mod tests {
    use super::*;
    use crate::biology::metabolism::chem;

    #[test]
    fn creature_hunts_berry_and_it_crunches() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 80.0, 20.0);
        let mut arena = Arena::new(42, map);
        arena.spawn(Vec2::new(5.0, 10.0), 0.0, 42);
        // Голод сразу, чтобы не ждать сотни тиков синтеза HUNGER.
        arena.critters[0].creature.blood.conc[chem::HUNGER] = 200.0;
        let berry = arena.add_berry(Vec2::new(35.0, 10.0));

        for _ in 0..250 {
            arena.step();
            // Съедена = ушла в стадию отрастания (vars[1] > 0).
            if arena
                .world
                .agent(berry)
                .map(|a| a.vars[1] > 0.0)
                .unwrap_or(false)
            {
                break;
            }
        }
        // Ещё один шаг: EAT-сообщение доставляется в world.step следующего
        // такта — только тогда скрипт «хрум» выполнится и попадёт в журнал.
        arena.step();

        assert!(
            arena
                .world
                .agent(berry)
                .map(|a| a.alive && a.vars[1] > 0.0)
                .unwrap_or(false),
            "ягода должна быть съедена и отрастать (не исчезать)"
        );
        assert!(
            arena.world.log().iter().any(|s| s.contains("хрум")),
            "ягода должна хрумнуть: {:?}",
            arena.world.log()
        );
        assert!(
            arena.critters[0].creature.glucose() > 128.0,
            "еда должна дать питание"
        );
    }

    /// Охота за стеной: существо с направленными сенсорами стены объезжает
    /// L-препятствие и доедает в обоих триалах (мозг живёт между триалами,
    /// кровь сбрасывается). Раньше (без сенсора) тот же удар выжигал гоминг,
    /// и второй триал голодал у стены.
    ///
    /// Между триалами старые ягоды убиваются: ягода первого триала на той же
    /// позиции (55,15) успевала отрасти (75 тиков = 3.75 с при тике 50 мс) и
    /// перехватывала еду второго триала — существо ело СТАРУЮ ягоду, а новая
    /// оставалась нетронутой, и тест считал триал проваленным.
    ///
    /// Голод на старте — 255 (потолок): при 200 существо доезжает до ягоды
    /// с голодом ровно у порога wants_food (40) — замер: 41 при тике 25 мс,
    /// 39 при 50 мс. Оба исхода — на грани, тест мерил не навигацию, а
    /// везение с таймингом. Потолок даёт запас ~10 пунктов и стабильно
    /// проверяет именно объезд L-стены.
    ///
    /// Порог ударов (замыкание контура, переякорено): сон не кормит
    /// (SLEEP_GLUCOSE_CAP=120 < 150) и дефицитный синтез голода ребалансирован
    /// (0.08/тик — равновесие HUNGER ≈ 80 при глюкозе 120). Голод устойчив,
    /// охота не умирает — и навигация стала чище: замер триалов 0 ударов
    /// в обоих (еда на t≈144/t≈216; при одной только капе сна без ребаланса
    /// экономики было 0 и 10). Порог 12 оставляет запас на случай будущих
    /// изменений экономики голода (панический тон при HUNGER > 240 толкает
    /// в угол L) при детерминированном прогоне (сид 42).
    #[test]
    fn detour_hunt_survives_trials_with_wall_sense() {
        let mut map = WorldMap::default();
        // L-форма: нижний правый квадрант (x>20, y<10) — пустота (стена).
        map.add_room(0.0, 0.0, 20.0, 20.0);
        map.add_room(20.0, 10.0, 60.0, 20.0);
        let mut a = Arena::new(42, map);
        a.spawn(Vec2::new(5.0, 5.0), 0.0, 1);
        for _ in 0..2 {
            a.critters[0].creature.blood = crate::biology::metabolism::Blood::creature(false);
            a.critters[0].creature.blood.conc[chem::HUNGER] = 255.0;
            // Свежая ягода триала: прошлые убрать (см. док теста).
            for id in a.berries() {
                a.world.kill(id);
            }
            let berry = a.add_berry(Vec2::new(55.0, 15.0));
            let h0 = a.critters[0].wall_hits;
            let mut ate = false;
            let mut saw_vision = false;
            for _ in 0..800 {
                a.step();
                // Стадия B: проводка зрения жива — снимок виз. сигнала арены
                // (ягода видна в той же комнате в конусе курса). Если бы
                // зрение не было подключено к существам, флаг остался бы 0.
                if a.critters[0].vis_l > 0.0 || a.critters[0].vis_r > 0.0 {
                    saw_vision = true;
                }
                // Съедена = отрастает (vars[1] > 0).
                if a
                    .world
                    .agent(berry)
                    .map(|ag| ag.vars[1] > 0.0)
                    .unwrap_or(false)
                {
                    ate = true;
                    break;
                }
            }
            assert!(ate, "существо должно объехать стену и съесть ягоду");
            assert!(
                saw_vision,
                "в ходе объезда зрение должно видеть ягоду (стадия B подключена)"
            );
            assert!(
                a.critters[0].wall_hits - h0 <= 12,
                "объезд почти без ударов: {} ударов",
                a.critters[0].wall_hits - h0
            );
            a.critters[0].pos = Vec2::new(5.0, 5.0);
            a.critters[0].heading = 0.0;
        }
    }

    /// Регрессионный сторож поведенческой эквивалентности биохимии.
    ///
    /// Фиксированный сценарий: детур-карта, сид 42, спавн (5,5)/0.0 (сид 1),
    /// голод 255, ягода (55,15) — триал 1 теста `detour_hunt_*` без сброса
    /// крови и без перезапуска. Константы сняты с ИСПРАВЛЕННОЙ реализации
    /// (SIMD-распады + batch-рецепторы, полный пересчёт сенсоров) и побайтно
    /// совпадают с чекпойнтом 2550cae (эталон до оптимизаций).
    ///
    /// Эталон после починки SIMD-распадов: «инкрементальные сенсоры»
    /// (дельты от prev_sense) давали систематический дрейф ext — мозг получал
    /// искажённый запах/стену, и HUNGER/PAIN на тике 200 уходили на десятки
    /// единиц (эталон 53.46/47.50). Любое будущее переупорядочивание шагов
    /// биохимии (распады/реакции/рецепторы) или сенсорного входа, меняющее
    /// поведение, ломает этот тест громко, а не тихим срывом охоты.
    ///
    /// СТАДИЯ B (переякорено): мозг 300 нейронов (зрение + ассоциативный хаб)
    /// изменил моторный драйв — существо ест раньше/прямее: GLUCOSE@200 упирается
    /// в потолок 255 (было 254.877), HUNGER@200 58.27 (было 53.46), PAIN@200 40.0
    /// (было 47.50), wall_hits по-прежнему 0. Эталон переснят с подключённой
    /// стадии B (см. переменную DIOMA_RECORD_TRACE в тесте) — сторож остаётся
    /// чувствительным: любые изменения шагов биохимии/сенсоров сдвинут значения.
    ///
    /// ЗАМЫКАНИЕ КОНТУРА (переякорено): сон не кормит (SLEEP_GLUCOSE_CAP=120 <
    /// порога дефицита 150) — GLUCOSE@200 больше НЕ упирается в потолок 255:
    /// существо поело и поспало, глюкоза 122.4 (выше капа сна, ниже 150 —
    /// голод продолжает синтезироваться), HUNGER@200 135.5 (дефицитный
    /// синтез 0.08/тик держит голод высоким — равновесие ~80+), PAIN@200
    /// 189.8 — устойчивый голод держит существо в движении, оно мечется в
    /// L-углу (11 ударов к 200, 27 к 300; до ребаланса экономики было
    /// 6 и 23). Эталон переснят после замыкания контура (см. DIOMA_RECORD_TRACE).
    ///
    /// ЭКОНОМИКА УТОМЛЕНИЯ (переякорено): гейн RATE_MOTOR→FATIGUE снижен ×6
    /// (0.20 → 0.033, см. metabolism.rs) — существо бодрствует дольше и ест
    /// иначе: GLUCOSE@200 93.8 (после еды и сна, ниже прежнего 122.4),
    /// HUNGER@200 197.9 (голод высок — дефицитный синтез при глюкозе ниже
    /// капа сна), PAIN@200 210.5, FATIGUE@200 60.4 (недавний сон, ×0.3).
    /// Траектория: 25 ударов к 200, дальше существо спит/ест — 25 к 300
    /// (до ребаланса было 11/27). Эталон переснят после ребаланса
    /// (см. DIOMA_RECORD_TRACE).
    /// СТАДИЯ C-LITE (переякорено): мозг 760 нейронов (кора 400+120 вместо
    /// 90+30, зрение 60+60) — моторика богаче, траектория другая: GLUCOSE@200
    /// 96.3, HUNGER@200 111.4 (голод к тику 200 не дожат — часть пути сна),
    /// PAIN@200 210.5, FATIGUE@200 163.6 (замедленная экономика + поздний
    /// сон), удары 24 к 200 и 31 к 300. Эталон переснят после подключения
    /// стадии C-lite (см. DIOMA_RECORD_TRACE).
    #[test]
    fn biochemistry_trace_is_stable() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 20.0, 20.0);
        map.add_room(20.0, 10.0, 60.0, 20.0);
        let mut a = Arena::new(42, map);
        a.spawn(Vec2::new(5.0, 5.0), 0.0, 1);
        a.critters[0].creature.blood = crate::biology::metabolism::Blood::creature(false);
        a.critters[0].creature.blood.conc[chem::HUNGER] = 255.0;
        a.add_berry(Vec2::new(55.0, 15.0));

        // ВРЕМЕННО: печать новых эталонных значений стадии B.
        if std::env::var("DIOMA_RECORD_TRACE").is_ok() {
            for _ in 0..200 {
                a.step();
            }
            let conc = &a.critters[0].creature.blood.conc;
            println!("TRACE@200 GLUCOSE={:.9} HUNGER={:.9} PAIN={:.9} FATIGUE={:.9} SEROTONIN={:.9} ACETYLCHOLINE={:.9} wall_hits={}", conc[chem::GLUCOSE], conc[chem::HUNGER], conc[chem::PAIN], conc[chem::FATIGUE], conc[chem::SEROTONIN], conc[chem::ACETYLCHOLINE], a.critters[0].wall_hits);
            for _ in 200..300 {
                a.step();
            }
            let conc = &a.critters[0].creature.blood.conc;
            println!("TRACE@300 HUNGER={:.9} SEROTONIN={:.9} ACETYLCHOLINE={:.9} wall_hits={}", conc[chem::HUNGER], conc[chem::SEROTONIN], conc[chem::ACETYLCHOLINE], a.critters[0].wall_hits);
            return;
        }

        for _ in 0..200 {
            a.step();
        }
        let conc = &a.critters[0].creature.blood.conc;
        let close = |got: f32, want: f32| (got - want).abs() <= 1e-4;

        // Тик 200 (после еды и сна): глюкоза/голод/боль/усталость/серотонин/внимание.
        // Эталон после ребаланса утомления (×6): существо бодрствует дольше,
        // ест иначе — глюкоза 93.8 (было 122.4), голод 197.9 (было 135.5).
        assert!(
            close(conc[chem::GLUCOSE], 77.195983887),
            "GLUCOSE@200: {} ≠ 96.310821533", conc[chem::GLUCOSE]
        );
        assert!(
            close(conc[chem::HUNGER], 212.921829224),
            "HUNGER@200: {} ≠ 111.399505615", conc[chem::HUNGER]
        );
        assert!(
            close(conc[chem::PAIN], 94.213439941),
            "PAIN@200: {} ≠ 94.213439941", conc[chem::PAIN]
        );
        assert!(
            close(conc[chem::FATIGUE], 179.582946777),
            "FATIGUE@200: {} ≠ 163.586791992", conc[chem::FATIGUE]
        );
        assert!(
            close(conc[chem::SEROTONIN], 39.572669983),
            "SEROTONIN@200: {} ≠ 43.064628601", conc[chem::SEROTONIN]
        );
        assert!(
            close(conc[chem::ACETYLCHOLINE], 9.755206108),
            "ACETYLCHOLINE@200: {} ≠ 9.223410606", conc[chem::ACETYLCHOLINE]
        );
        // Тик 300: голод держится высоким (дефицитный синтез), внимание от еды.
        for _ in 200..300 {
            a.step();
        }
        let conc = &a.critters[0].creature.blood.conc;
        assert!(
            close(conc[chem::HUNGER], 143.243469238),
            "HUNGER@300: {} ≠ 193.141052246", conc[chem::HUNGER]
        );
        assert!(
            close(conc[chem::SEROTONIN], 43.100345612),
            "SEROTONIN@300: {} ≠ 46.128597260", conc[chem::SEROTONIN]
        );
        assert!(
            close(conc[chem::ACETYLCHOLINE], 21.523950577),
            "ACETYLCHOLINE@300: {} ≠ 22.424747467", conc[chem::ACETYLCHOLINE]
        );
        // Траектория эталона: с медленным утомлением существо бодрствует
        // дольше, к 200 успевает набрать 25 ударов в L-углу, дальше спит/ест —
        // к 300 счётчик не растёт (до ребаланса было 11 к 200 и 27 к 300).
        // Химия выше — основной сторож; счётчик фиксирует траекторию целиком.
        assert_eq!(
            a.critters[0].wall_hits, 25,
            "эталонная траектория стадии B: {} ударов", a.critters[0].wall_hits
        );
    }

    /// Сквозная проводка «стена → боль → мозг»: существо в углу упирается
    /// в стены, арена считает удар и поднимает ноцицепцию в крови
    /// (bump_wall → PAIN). В углу отворот невозможен — сенсоры стены дают
    /// компенсирующие сигналы, и удар гарантирован.
    #[test]
    fn wall_hit_counts_and_raises_pain() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        let mut arena = Arena::new(7, map);
        arena.spawn(Vec2::new(39.5, 1.5), 0.0, 7);
        arena.critters[0].creature.blood.conc[chem::HUNGER] = 200.0;
        let pain0 = arena.critters[0].creature.blood.conc[chem::PAIN];

        // Гоним до первого удара о стену (угол: удар на первом же такте).
        let mut guard = 0;
        while arena.critters[0].wall_hits == 0 && guard < 200 {
            arena.step();
            guard += 1;
        }
        assert!(
            arena.critters[0].wall_hits > 0,
            "существо должно упереться в стену и получить удары"
        );
        // bump_wall ставит EVENT_WALL, а боль эмитируется в blood.update()
        // следующего тика — один шаг, чтобы эмиттер сработал.
        arena.step();
        let pain = arena.critters[0].creature.blood.conc[chem::PAIN];
        assert!(
            pain > pain0 + 5.0,
            "удар о стену должен поднять боль: {pain0} → {pain}"
        );
    }

    #[test]
    fn berries_grow_only_in_fertile_rooms() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0); // плодородная
        map.add_room(40.0, 0.0, 80.0, 20.0); // бедная
        let mut arena = Arena::new(1, map);
        // Почти без распада веществ, чтобы питание не ушло за время теста.
        arena.world.map.rates.decay = [0.0005; 4];
        let nutr = Substance::Nutrients.idx();
        let light = Substance::Light.idx();
        let temp = Substance::Temperature.idx();
        arena.world.map.rooms[0].ca[nutr] = 200.0;
        arena.world.map.rooms[0].ca[light] = 200.0;
        // Климат трёхфакторный (тепло × свет × питание) — плодородная
        // комната должна быть и тёплой, иначе ягоды не растут.
        arena.world.map.rooms[0].ca[temp] = 200.0;

        let rich = arena.add_berry(Vec2::new(20.0, 10.0)); // плодородная
        let poor = arena.add_berry(Vec2::new(60.0, 10.0)); // бедная
        let n0 = arena.berries().len();

        for _ in 0..600 {
            arena.step();
        }

        let n1 = arena.berries().len();
        assert!(
            n1 > n0,
            "в плодородной комнате ягоды должны размножиться: {n0} → {n1}"
        );
        assert!(
            arena.world.agent(poor).is_some() && arena.world.agent(rich).is_some(),
            "родители не должны исчезать без еды"
        );
        assert!(
            arena.world.map.rooms[0].ca[nutr] < 200.0,
            "рост должен съедать питание комнаты"
        );
        assert!(n1 <= MAX_BERRIES, "популяция ограничена капом: {n1}");
    }

    #[test]
    fn arena_is_deterministic_from_seeds() {
        let run = || -> Vec<(f32, f32)> {
            let mut map = WorldMap::default();
            map.add_room(0.0, 0.0, 60.0, 20.0);
            let mut a = Arena::new(42, map);
            a.spawn(Vec2::new(5.0, 10.0), 0.0, 42);
            a.add_berry(Vec2::new(30.0, 10.0));
            for _ in 0..50 {
                a.step();
            }
            a.critters
                .iter()
                .map(|c| (c.pos.x, c.pos.y))
                .collect()
        };
        assert_eq!(run(), run(), "один сид — одна траектория");
    }

    // ─── Arena::pick (мост сим ↔ рендер: клик-выбор) ─────────────────────

    /// Пикинг выбирает БЛИЖАЙШЕЕ существо в радиусе (не любое в радиусе).
    #[test]
    fn pick_nearest_creature() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        let mut a = Arena::new(1, map);
        let c1 = a.spawn(Vec2::new(5.0, 5.0), 0.0, 1);
        let c2 = a.spawn(Vec2::new(8.0, 5.0), 0.0, 2);
        // Курсор у c1: оба в радиусе 2.0, но c1 ближе.
        assert_eq!(a.pick(Vec2::new(6.0, 5.0), 2.0, 1.0), Some(c1));
        // Курсор у c2: c1 уже вне радиуса (dist 4.0 > 2.0).
        assert_eq!(a.pick(Vec2::new(9.0, 5.0), 2.0, 1.0), Some(c2));
        // Курсор между ними, но дальше 2.0 от обоих — мимо.
        assert_eq!(a.pick(Vec2::new(6.5, 9.0), 2.0, 1.0), None);
    }

    /// Без существ пикинг находит ягоду (живую, в радиусе).
    #[test]
    fn pick_berry_when_no_creature() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        let mut a = Arena::new(1, map);
        let b1 = a.add_berry(Vec2::new(20.0, 10.0));
        assert_eq!(a.pick(Vec2::new(20.3, 10.2), 1.0, 1.0), Some(b1));
        // Вне радиуса ягоды — мимо.
        assert_eq!(a.pick(Vec2::new(23.0, 10.0), 1.0, 1.0), None);
    }

    /// Пустая точка (нет ни существ, ни ягод в радиусах) — None.
    #[test]
    fn pick_none_in_empty_space() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        let a = Arena::new(1, map);
        assert_eq!(a.pick(Vec2::new(10.0, 10.0), 1.0, 1.0), None);
    }

    /// Существо приоритетнее ягоды: выбираем существо, даже если ягода ближе.
    #[test]
    fn pick_prefers_creature_over_berry_when_both_near() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        let mut a = Arena::new(1, map);
        let c = a.spawn(Vec2::new(5.0, 5.0), 0.0, 1);
        a.add_berry(Vec2::new(5.4, 5.0)); // ягода ближе к курсору (5.2, 5)
        assert_eq!(a.pick(Vec2::new(5.2, 5.0), 2.0, 2.0), Some(c));
    }

    // ─── Сон (фаза сна в Arena::step) ────────────────────────────────────

    /// Сон в арене: уставшее существо (fatigue ≥ SLEEP_AT_FATIGUE) засыпает,
    /// спит SLEEP_TICKS тиков неподвижно (нет мозга/движения/еды), просыпается
    /// с усталостью заметно ниже порога. Метаболизм во сне стоит — дневная
    /// усталость сбрасывается единичным `creature.sleep()` на входе.
    ///
    /// Усталость копит моторная активность (RATE_MOTOR): две ягоды по бокам
    /// курса дают устойчивый симметричный запах — существо бежит вперёд
    /// (позиция меняется наяву) и выматывается, как в `creature::tests::
    /// activity_accumulates_fatigue` (сильный драйв без руления).
    #[test]
    fn tired_creature_falls_asleep_and_recovers() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 60.0, 20.0);
        let mut arena = Arena::new(9, map);
        arena.spawn(Vec2::new(10.0, 10.0), 0.0, 9);
        arena.critters[0].creature.blood.conc[chem::HUNGER] = 200.0;
        // Ягоды симметрично по бокам курса: запах слева ≈ справа, моторы гонят.
        arena.add_berry(Vec2::new(30.0, 9.0));
        arena.add_berry(Vec2::new(30.0, 11.0));

        // Доводим до усталости: ждём, когда существо заснёт само.
        let mut entry = None;
        for t in 0..3000u32 {
            arena.step();
            if arena.critters[0].sleeping {
                entry = Some(t);
                break;
            }
        }
        let entry = entry.expect("существо должно устать и заснуть за 3000 тиков");
        assert!(entry > 0, "сон не наступает с первого тика (нет усталости)");
        assert!(
            arena.critters[0].sleep_ticks == SLEEP_TICKS,
            "сон начинается с полного окна: {}",
            arena.critters[0].sleep_ticks
        );

        // Сон: позиция не меняется (нет движения/еды) — наяву существо бежало,
        // значит заморозка именно от сна. Усталость фиксируем в момент
        // пробуждения — пока спит, метаболизм стоит, и после сна она намного
        // ниже порога.
        let mut woke_fatigue = None;
        for _ in 0..SLEEP_TICKS + 3 {
            let before = arena.critters[0].pos;
            arena.step();
            if arena.critters[0].sleeping {
                assert_eq!(
                    arena.critters[0].pos,
                    before,
                    "спящее существо не должно двигаться"
                );
            } else if woke_fatigue.is_none() {
                woke_fatigue = Some(arena.critters[0].creature.fatigue());
            }
        }
        let woke_fatigue = woke_fatigue
            .expect("после SLEEP_TICKS+ тиков существо должно проснуться");
        assert!(
            woke_fatigue < SLEEP_AT_FATIGUE * 0.5,
            "сон должен сбить усталость намного ниже порога: {woke_fatigue} (порог {SLEEP_AT_FATIGUE})"
        );
    }

    // ─── Стены (запах/зеркало по комнатам, Sense::wall) ───────────────────

    /// Запах не проходит сквозь стены: ягода в соседней комнате у общей стены
    /// не пахнет существу в своей комнате (ягода смещена по y — без фильтра
    /// запах был бы ~0.5); зайдя в ту же комнату, существо чувствует её.
    /// Двери запах пока не проводят (транш 2).
    #[test]
    fn smell_does_not_pass_through_walls() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        map.add_room(40.0, 0.0, 80.0, 20.0);
        let mut arena = Arena::new(3, map);
        // Ягода в комнате B у общей стены, смещена по y (боковой запах).
        arena.add_berry(Vec2::new(40.5, 9.0));

        // Существо в комнате A у общей стены: дистанция ~1.4, но комната другая.
        let in_a = arena.sense(Vec2::new(39.4, 10.0), 0.0, false);
        assert_eq!(
            in_a.smell_l + in_a.smell_r,
            0.0,
            "запах не должен проходить сквозь стену: {in_a:?}"
        );
        assert!(!in_a.food_near, "еда за стеной не «у рта»");

        // Переходим в комнату B — та же ягода пахнет и «у рта».
        let in_b = arena.sense(Vec2::new(41.0, 10.0), 0.0, false);
        assert!(
            in_b.smell_l + in_b.smell_r > 0.0,
            "в той же комнате запах должен быть: {in_b:?}"
        );
        assert!(in_b.food_near, "в той же комнате ягода «у рта»");
    }

    /// Социальное зеркало тоже заперто стенами: сосед в той же комнате видит
    /// едущего (see_eat), сосед за стеной — нет, даже если дистанция до едока
    /// меньше SOCIAL_RADIUS.
    /// (Едок на 38.0, не у самой стены: при тике 50 мс шаг движения ×2, и
    /// со старта в 39.0 едок перелетал границу комнат за тик — ел уже в
    /// комнате B, и зеркало для соседа в A законно не срабатывало.)
    #[test]
    fn see_eat_mirror_is_room_gated() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        map.add_room(40.0, 0.0, 80.0, 20.0);
        let mut arena = Arena::new(4, map);
        // Едок в комнате A у стены, ягода у него во рту.
        let eater = arena.spawn(Vec2::new(38.0, 10.0), 0.0, 1);
        arena
            .critters
            .iter_mut()
            .find(|c| c.id == eater)
            .unwrap()
            .creature
            .blood
            .conc[chem::HUNGER] = 200.0;
        arena.add_berry(Vec2::new(39.4, 10.0));
        // Наблюдатель в той же комнате (2 юнита от едока).
        let same = arena.spawn(Vec2::new(37.0, 10.0), 0.0, 2);
        // Наблюдатель за стеной (2 юнита от едока, но другая комната).
        let across = arena.spawn(Vec2::new(41.0, 10.0), 0.0, 3);

        arena.step(); // едок съел → зеркало раздаёт see_eat

        let idx = |id: AgentId| arena.critters.iter().position(|c| c.id == id).unwrap();
        assert!(
            arena.critters[idx(same)].see_eat,
            "сосед в той же комнате должен видеть едущего"
        );
        assert!(
            !arena.critters[idx(across)].see_eat,
            "сосед за стеной не должен видеть едущего"
        );
    }

    /// Sense::wall заполняется ареной: стена прямо по курсу (обе диагональные
    /// пробы высоки) — true; боковая стена и открытое поле — false.
    #[test]
    fn wall_sense_detects_wall_ahead() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 20.0, 20.0);
        let arena = Arena::new(5, map);
        // Стена прямо: (18,10) курсом на +x, стена на x = 20 в ~2 юнитах.
        let facing = arena.sense(Vec2::new(18.0, 10.0), 0.0, false);
        assert!(facing.wall, "стена впереди должна заполнять wall: {facing:?}");
        // Открытое поле: стены нет нигде рядом.
        let open = arena.sense(Vec2::new(10.0, 10.0), 0.0, false);
        assert!(!open.wall, "в открытом поле wall=false: {open:?}");
        // Боковая стена (курс вдоль неё): только одна проба — не «прямо».
        let lateral = arena.sense(Vec2::new(1.5, 10.0), std::f32::consts::FRAC_PI_2, false);
        assert!(!lateral.wall, "боковая стена — не стена по курсу: {lateral:?}");
    }

    // ─── Зрение (стадия B: конус ±VISION_HALF_FOV, дальность VISION_RANGE) ──

    /// Зрительный конус: ягода в левой половине конуса (пеленг −45°) →
    /// vis_l > vis_r; в правой (+45°) → vis_r > vis_l; строго позади
    /// (пеленг π) — не видна. Ягода ровно по курсу (пеленг 0) уходит
    /// в правый бин — как у запаха (rel < 0 → левый, 0 → правый).
    #[test]
    fn vision_cone_left_right_and_behind() {
        let fresh = || {
            let mut map = WorldMap::default();
            map.add_room(0.0, 0.0, 40.0, 40.0);
            Arena::new(5, map)
        };
        let pos = Vec2::new(20.0, 20.0);
        let heading = 0.0f32;

        // Слева: ягода под −45° (dx=+10, dy=−10), dist 14.1 → vis ≈ 0.65.
        let mut a = fresh();
        a.add_berry(Vec2::new(30.0, 10.0));
        let s = a.sense(pos, heading, false);
        assert!(
            s.vis_l > 0.0 && s.vis_r == 0.0,
            "слева: vis_l={} vis_r={}",
            s.vis_l,
            s.vis_r
        );

        // Справа: ягода под +45° (dx=+10, dy=+10).
        let mut a = fresh();
        a.add_berry(Vec2::new(30.0, 30.0));
        let s = a.sense(pos, heading, false);
        assert!(
            s.vis_l == 0.0 && s.vis_r > 0.0,
            "справа: vis_l={} vis_r={}",
            s.vis_l,
            s.vis_r
        );

        // Позади (пеленг π): пеленг вне конуса — не видна.
        let mut a = fresh();
        a.add_berry(Vec2::new(10.0, 20.0));
        let s = a.sense(pos, heading, false);
        assert!(
            s.vis_l == 0.0 && s.vis_r == 0.0,
            "позади: vis_l={} vis_r={}",
            s.vis_l,
            s.vis_r
        );

        // Ровно по курсу (пеленг 0): правый бин (спецификация Sense).
        let mut a = fresh();
        a.add_berry(Vec2::new(30.0, 20.0));
        let s = a.sense(pos, heading, false);
        assert!(
            s.vis_r > 0.0 && s.vis_l == 0.0,
            "по курсу: vis_l={} vis_r={}",
            s.vis_l,
            s.vis_r
        );
    }

    /// Зрение не проходит сквозь стены: ягода в соседней комнате у общей
    /// стены не видна существу в своей комнате (та же проверка комнаты,
    /// что у запаха — дистанция и пеленг при этом в конусе); зайдя в ту же
    /// комнату, существо видит её.
    #[test]
    fn vision_blocked_by_room() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        map.add_room(40.0, 0.0, 80.0, 20.0);
        let mut arena = Arena::new(3, map);
        // Ягода в комнате B у общей стены, ровно по курсу существа из A.
        arena.add_berry(Vec2::new(45.0, 10.0));

        // Существо в комнате A: дистанция 10, пеленг 0 — видно было бы,
        // если бы стены были прозрачны. Комната другая — зрение гасится.
        let in_a = arena.sense(Vec2::new(35.0, 10.0), 0.0, false);
        assert_eq!(
            in_a.vis_l + in_a.vis_r,
            0.0,
            "зрение не должно проходить сквозь стену: {in_a:?}"
        );

        // Переходим в комнату B — та же ягода видна (дистанция 3, пеленг 0).
        let in_b = arena.sense(Vec2::new(42.0, 10.0), 0.0, false);
        assert!(
            in_b.vis_l + in_b.vis_r > 0.0,
            "в той же комнате зрение должно видеть: {in_b:?}"
        );
    }

    // ─── Berry-API (Arena::berry_info) ────────────────────────────────────

    /// Berry-API: berry_info читает внутренние vars[] через именованные поля
    /// (рост/отрастание/съедобность) — хост и мост рендера не трогают vars.
    #[test]
    fn berry_info_reports_growth_and_edibility() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        let mut arena = Arena::new(6, map);
        let bid = arena.add_berry(Vec2::new(20.0, 10.0));

        // Свежая ягода: съедобна, рост 0, отрастание 0.
        let b = arena.berry_info(bid).expect("свежая ягода видна");
        assert!(b.edible, "свежая ягода съедобна");
        assert_eq!(b.growth, 0.0);
        assert_eq!(b.regrow, 0.0);
        assert_eq!(b.pos, Vec2::new(20.0, 10.0));

        // Рост: vars[0] копит плодородие → growth виден через berry_info.
        arena.world.agent_mut(bid).unwrap().vars[0] = 3.0;
        let b = arena.berry_info(bid).unwrap();
        assert_eq!(b.growth, 3.0, "growth отражает vars[0]");

        // Съедена (отрастает): vars[1] > 0 → edible false, regrow виден.
        arena.world.agent_mut(bid).unwrap().vars[1] = 40.0;
        let b = arena.berry_info(bid).unwrap();
        assert!(!b.edible, "отрастающая ягода не съедобна");
        assert_eq!(b.regrow, 40.0, "regrow отражает vars[1]");

        // Живой агент мира, но не ягода → None.
        let rock = arena.world.spawn(Cls::new(2, 2, 1), Vec2::new(5.0, 5.0), "rock");
        assert!(arena.berry_info(rock).is_none(), "не-ягода — None");
    }

    // ─── Мировые часы: день/ночь и сезоны (как в C3) ──────────────────────

    /// Часы арены: сутки реальных тиков обходят фазы рассвет→…→ночь,
    /// день в сезоне увеличивается, а год (4 сезона × 3 дня) зацикливается.
    /// Математика фаз покрыта в `env::world_clock::tests`; здесь — связка
    /// «тик арены → accessors Arena::{time_of_day, day_in_season, season}».
    #[test]
    fn world_clock_phases_and_seasons() {
        use crate::env::TICKS_PER_DAY;

        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 14.0);
        let mut arena = Arena::new(1, map);
        assert_eq!(arena.time_of_day(), TimeOfDay::Dawn, "тик 0 — рассвет");
        assert_eq!(arena.day_in_season(), 0);
        assert_eq!(arena.season(), Season::Spring);

        // Полные сутки: фазы обошли круг, день в сезоне сдвинулся.
        for _ in 0..TICKS_PER_DAY {
            arena.step();
        }
        assert_eq!(arena.tick, TICKS_PER_DAY);
        assert_eq!(
            arena.time_of_day(),
            TimeOfDay::Dawn,
            "после суток — снова рассвет"
        );
        assert_eq!(arena.day_in_season(), 1, "день в сезоне увеличился");
        assert_eq!(arena.season(), Season::Spring, "сезон ещё весенний");

        // Сезон: 3 дня = 3 × TICKS_PER_DAY тиков.
        for _ in 0..2 * TICKS_PER_DAY {
            arena.step();
        }
        assert_eq!(arena.day_in_season(), 0, "сезон из 3 дней завершился");
        assert_eq!(arena.season(), Season::Summer, "наступило лето");
        // Доля фазы в [0, 1).
        let f = arena.day_phase_fraction();
        assert!((0.0..1.0).contains(&f), "фаза должна быть в [0,1): {f}");
    }

    /// Ночь гасит свет климата: в фазе Дня плодородность комнаты высокая,
    /// в Ночь та же комната темнеет (поле света стремится к 0.15 дневного
    /// уровня) и ягоды растут медленнее — свет движет ростом, как в C3.
    #[test]
    fn night_reduces_light_field() {
        use crate::env::TICKS_PER_DAY;

        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 14.0);
        // Почти без распада комнат: сравниваем чисто эффект фазы суток.
        map.rates.decay = [0.0005; 4];
        let mut arena = Arena::new(1, map);
        let ca = &mut arena.world.map.rooms[0].ca;
        ca[Substance::Light.idx()] = 200.0;
        ca[Substance::Temperature.idx()] = 200.0;
        ca[Substance::Nutrients.idx()] = 200.0;

        let phase = TICKS_PER_DAY / 5;
        // День: часы на фазу Afternoon, даём свету устояться.
        arena.tick = TimeOfDay::Afternoon as u64 * phase;
        for _ in 0..100 {
            arena.step();
        }
        let day_fert = arena.climate.fertility(20.0, 7.0);
        assert!(day_fert > 0.4, "днём плодородие высокое: {day_fert}");

        // Ночь: та же комната, свет почти гаснет.
        arena.tick = TimeOfDay::Night as u64 * phase;
        for _ in 0..100 {
            arena.step();
        }
        let night_fert = arena.climate.fertility(20.0, 7.0);
        assert!(
            night_fert < day_fert * 0.7,
            "ночь должна заметно снизить плодородие: {day_fert} → {night_fert}"
        );
    }
}
