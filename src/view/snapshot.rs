//! Снимок арены: дешёвая read-only проекция `Arena` в данные для рисования.
//!
//! Строится каждый кадр через [`SimView::capture`] — копирование полей,
//! O(N) по существам и ягодам. Рендер работает по снимку и не держит
//! `&Arena`; презентация (см. [`super::present`]) — чистая функция.

use dioma_engine::creature::Sex;
use dioma_engine::env::map::Rect;
use dioma_engine::env::{AgentId, Season, TimeOfDay, Vec2, SUBSTANCES};
use dioma_engine::sim::Arena;

/// Read-only проекция арены для одного кадра.
#[derive(Debug, Clone)]
pub struct SimView {
    pub rooms: Vec<RoomView>,
    pub critters: Vec<CritterView>,
    pub berries: Vec<BerryView>,
    /// Фаза суток (Рассвет/Утро/День/Вечер/Ночь) — мировые часы арены.
    pub time_of_day: TimeOfDay,
    /// День внутри сезона (0..=2).
    pub day_in_season: u32,
    /// Сезон (Весна/Лето/Осень/Зима).
    pub season: Season,
    /// Прогресс внутри фазы суток 0..1 (плавные сумерки в рендере).
    pub phase_fraction: f32,
}

/// Комната: прямоугольник в мировых координатах + концентрации веществ CA.
#[derive(Debug, Clone, Copy)]
pub struct RoomView {
    pub idx: usize,
    pub rect: Rect,
    /// Концентрации веществ клеточного автомата `[SUBSTANCES]` — плодородность
    /// комнаты (температура/влажность/свет/питание) → цвет пола.
    pub ca: [f32; SUBSTANCES],
}

/// Существо: копия наблюдений `Creature`, нормированных под визуал.
#[derive(Debug, Clone, Copy)]
pub struct CritterView {
    pub id: AgentId,
    pub pos: Vec2,
    /// Курс в радианах (0 = +X).
    pub heading: f32,
    pub alive: bool,
    /// Скорость тела (учитывает усталость/повреждение).
    pub speed: f32,
    /// Голод 0..1: `hunger()` — концентрация HUNGER в крови (0..255), делим на 255.
    pub hunger: f32,
    /// Глюкоза 0..1: `glucose()` — концентрация GLUCOSE (0..255), делим на 255.
    pub glucose: f32,
    /// Усталость 0..1: `fatigue()` — концентрация FATIGUE (0..255), делим на 255.
    pub fatigue: f32,
    pub wants_food: bool,
    /// Порций еды, съеденных за жизнь (мера успеха охоты).
    pub food_eaten: u32,
    /// Пол существа (самец/самка; без игровых эффектов до транша 3).
    pub sex: Sex,
    /// Зрение (0..1): ягода в левом/правом поле зрения (снимок `Critter.vis_*`).
    pub vis_l: f32,
    pub vis_r: f32,
    /// Спит ли существо (химия усталости/сторож бодрствования — арена).
    pub sleeping: bool,
    /// Дофамин 0..1 (кровь REWARD): вспышка радости после еды/зеркальной еды.
    pub dopamine: f32,
    /// Серотонин 0..1 — настроение.
    pub serotonin: f32,
    /// Кортизол 0..1 — стресс (высокий → дрожь/болезненность).
    pub cortisol: f32,
    /// Окрас из расширяющегося генома (`dna.coat()`), RGB 0..1 — «рендер-ген»:
    /// мутации и кроссовер окраса видны на экране напрямую.
    pub coat: [f32; 3],
    /// Активность мозга 0..1: доля нейронов, спайкнувших за тик (spike_hist),
    /// растянутая в читаемый диапазон.
    pub spike_rate: f32,
    /// Сколько мест с едой существо помнит (гиппокамп, MIN_HITS=3).
    pub memory_places: usize,
    /// Моторный тон последнего тика (`Critter.last_rate_motor`) — жажда движения.
    pub moto_tone: f32,
}

/// Ягода: позиция + фаза роста 0..1 (1 = спелая).
#[derive(Debug, Clone, Copy)]
pub struct BerryView {
    pub id: AgentId,
    pub pos: Vec2,
    /// 0..1: 1 — спелая (`vars[1] <= 0`), иначе `1 − vars[1]/berry_regrow`
    /// (таймер отрастания тикает вниз к нулю → фаза растёт к 1).
    pub phase: f32,
}

impl SimView {
    /// Копирует арену в снимок кадра: комнаты, существа и ягоды.
    ///
    /// Концентрации крови нормируются на 255 (потолок крови в метаболизме)
    /// и зажимаются в 0..1 — см. [`CritterView`].
    pub fn capture(arena: &Arena) -> Self {
        let rooms = arena
            .world
            .map
            .rooms
            .iter()
            .enumerate()
            .map(|(idx, r)| RoomView { idx, rect: r.rect, ca: r.ca })
            .collect();
        let critters = arena
            .critters
            .iter()
            .map(|c| {
                let cr = &c.creature;
                CritterView {
                    id: c.id,
                    pos: c.pos,
                    heading: c.heading,
                    alive: cr.alive(),
                    speed: cr.speed(),
                    hunger: (cr.hunger() / 255.0).clamp(0.0, 1.0),
                    glucose: (cr.glucose() / 255.0).clamp(0.0, 1.0),
                    fatigue: (cr.fatigue() / 255.0).clamp(0.0, 1.0),
                    wants_food: cr.wants_food(),
                    food_eaten: cr.food_eaten(),
                    sex: cr.sex(),
                    vis_l: c.vis_l,
                    vis_r: c.vis_r,
                    sleeping: c.sleeping,
                    dopamine: (cr.dopamine_level() / 255.0).clamp(0.0, 1.0),
                    serotonin: (cr.serotonin() / 255.0).clamp(0.0, 1.0),
                    cortisol: (cr.cortisol() / 255.0).clamp(0.0, 1.0),
                    coat: cr
                        .dna
                        .coat()
                        .map(|c| [c.r.clamp(0.0, 1.0), c.g.clamp(0.0, 1.0), c.b.clamp(0.0, 1.0)])
                        .unwrap_or([1.0, 1.0, 1.0]),
                    // Доля спайкнувших нейронов за тик; ×8 — «читаемая» шкала
                    // (типичные значения — единицы процентов).
                    spike_rate: {
                        let stats = cr.brain_stats();
                        let steps = cr.spike_hist().len().max(1);
                        let total: usize = cr.spike_hist().iter().map(|s| s.len()).sum();
                        ((total as f32 / (stats.neuron_count * steps).max(1) as f32) * 8.0)
                            .clamp(0.0, 1.0)
                    },
                    memory_places: cr.places_remembered(),
                    moto_tone: c.last_rate_motor.clamp(0.0, 1.0),
                }
            })
            .collect();
        let berries = arena
            .berries()
            .iter()
            .filter_map(|&id| {
                // Именованный доступ вместо сырых vars[]: состояние берётся из
                // `Arena::berry_info` (см. Berry — рост/отрастание/съедобность).
                let b = arena.berry_info(id)?;
                let phase = if b.state.is_edible() {
                    1.0
                } else {
                    1.0 - (b.state.regrow / arena.berry_regrow).clamp(0.0, 1.0)
                };
                Some(BerryView { id: b.id, pos: b.pos, phase })
            })
            .collect();
        SimView {
            rooms,
            critters,
            berries,
            // Мировые часы: день/ночь и сезон из тика арены (как в C3).
            time_of_day: arena.time_of_day(),
            day_in_season: arena.day_in_season(),
            season: arena.season(),
            phase_fraction: arena.day_phase_fraction(),
        }
    }
}
