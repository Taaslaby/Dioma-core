//! -------------------------------------------------------------------------
//! Filename:    examples/emergence_run.rs
//! Class:       emergence
//! Purpose:     ЭМЕРДЖЕНТНОСТЬ В ЗАКРЫТОМ КОНТУРЕ — наблюдательный прогон
//! Description: два сценария без тестовых подпорок:
//!   Сценарий 1 «Дикий лес» — естественный закрытый контур: существа
//!     рождаются через Arena::spawn со СТАНДАРТНОЙ кровью Blood::creature(false)
//!     (глюкоза 128, голод 0), но новорождённый арены голоден с рождения
//!     (встроенная механика движка: ARENA_NEWBORN_HUNGER=120 — как
//!     NEWBORN_HUNGER=200 в эволюции), голод ПОДДЕРЖИВАЕТСЯ дефицитным
//!     синтезом (сон не кормит: SLEEP_GLUCOSE_CAP=120 < порога дефицита 150;
//!     ребаланс экономики глюкоза↔голод держит равновесие HUNGER ≈ 80),
//!     едят когда хотят, ягоды отрастают от плодородности климата, сон —
//!     встроенный цикл усталости. Никаких инъекций голода в раннере:
//!     только `Arena::step()`.
//!   Сценарий 2 «Поколения» — Evolution::new(seed 42, карта,
//!     EvolutionParams::default()): 5 поколений естественного отбора,
//!     история поколений (survivors/born/avg_food/max_food/with_instinct/
//!     instinct_dopamine_avg).
//!   Далее в этом же примере считаются наблюдательные метрики
//!     (трети прогона, занятость 8×8, энтропия курса, NN-дистанции,
//!     «личности» F-ratio, прокси «следования», дельты эволюции) и всё
//!     складывается в `tools/emergence_data.json` для
//!     `tools/emergence_report.py` (самодостаточный HTML-отчёт).
//! author: Atlasov anton
//! -------------------------------------------------------------------------
//! Использование:
//!   cargo run --release --example emergence_run -- --out tools/emergence_data.json
//!   # смоук (проверка пайплайна): 3000 тиков, 2 поколения
//!   cargo run --release --example emergence_run -- --out tools/emergence_data.json --quick
//! -------------------------------------------------------------------------

use dioma_engine::env::map::{WorldMap, SUBSTANCES};
use dioma_engine::env::math::Vec2;
use dioma_engine::env::Substance;
use dioma_engine::sim::{Arena, Evolution, EvolutionParams, GenerationStats};
use std::path::PathBuf;

/// Интервал снапшотов (тиков): все метрики считаются по ним.
const SNAPSHOT_EVERY: u64 = 1000;
/// Полный прогон «Дикого леса» в тиках (30000 × 50 мс ≈ 25 минут жизни).
const WILD_TICKS: u64 = 30_000;
/// Смоук-прогон (--quick).
const WILD_TICKS_QUICK: u64 = 3_000;
/// Сид арены «Дикого леса».
const WILD_SEED: u64 = 777;
/// Сиды существ (spawn в разнесённые позиции).
const WILD_CREATURE_SEEDS: [u64; 6] = [3001, 3002, 3003, 3004, 3005, 3006];
/// Стартовых ягод.
const WILD_BERRIES: usize = 10;
/// Сид эволюции «Поколений».
const EVO_SEED: u64 = 42;
/// Поколений (полный прогон) / смоук.
const EVO_GENS: u64 = 5;
const EVO_GENS_QUICK: u64 = 2;
/// Сетка занятости (GRID×GRID) по bbox карты (0,0)-(80,28).
const GRID: usize = 8;
/// Радиус «следования» — прокси социального наблюдения (у движка
/// SOCIAL_RADIUS = 10.0; здесь берём «иш-6.0» по ТЗ, грубее и честнее).
const SOCIAL_PROXY_RADIUS: f32 = 6.0;
/// Бинов курса для энтропии.
const HEADING_BINS: usize = 8;
/// Размеры bbox карты для сетки занятости.
const MAP_X: f32 = 80.0;
const MAP_Y: f32 = 28.0;

// ─── Структуры снапшотов ──────────────────────────────────────────────────

/// Снимок одного существа.
#[derive(Clone)]
struct SnapCrit {
    id: u64,
    pos: Vec2,
    heading: f32,
    food: u32,
    wall: u64,
    sleeping: bool,
    hunger: f32,
    glucose: f32,
    fatigue: f32,
    alive: bool,
    drive: f32,
    rate_motor: f32,
    coat: [f32; 3],
}

/// Снимок арены: тик, число ягод, все существа.
#[derive(Clone)]
struct Snap {
    tick: u64,
    berry_count: usize,
    critters: Vec<SnapCrit>,
}

/// Конечное состояние существа («что нажито за жизнь»).
struct WildEnd {
    id: u64,
    instincts: usize,
    dopamines: Vec<f32>,
    places: usize,
    food: u32,
    alive: bool,
}

/// Метрики одной трети прогона для одного существа.
#[derive(Clone, Copy, Default)]
struct ThirdMetrics {
    food_rate: f64,
    wall_hit_rate: f64,
    sleep_frac: f64,
    mean_speed: f64,
    path_len: f64,
    net_displacement: f64,
    tortuosity: f64,
    heading_entropy: f64,
}

// ─── Карта «Дикого леса»: 3 комнаты, соединённые общими стенами ──────────

/// Карта: комнаты (0,0)-(40,14), (40,0)-(80,14), (0,14)-(40,28) — буквой «Г»;
/// переходы — общие грани (A↔B по x=40, A↔C по y=14), климат тёплый/светлый/
/// питательный, как в игровом дефолте (game.rs): ягоды отрастают сами.
fn wild_map() -> WorldMap {
    let mut map = WorldMap::default();
    let a = map.add_room(0.0, 0.0, 40.0, 14.0);
    let b = map.add_room(40.0, 0.0, 80.0, 14.0);
    let c = map.add_room(0.0, 14.0, 40.0, 28.0);
    map.connect(a, b, [1.0; SUBSTANCES]);
    map.connect(a, c, [1.0; SUBSTANCES]);
    for room in &mut map.rooms {
        room.ca[Substance::Temperature.idx()] = 200.0;
        room.ca[Substance::Light.idx()] = 200.0;
        room.ca[Substance::Nutrients.idx()] = 200.0;
        room.ca[Substance::Humidity.idx()] = 150.0;
    }
    map
}

/// Детерминированный начальный курс из сида (как в эволюции движка).
fn heading_from_seed(seed: u64) -> f32 {
    let h = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed.rotate_left(17);
    ((h >> 40) & 0xFFFF) as f32 / 65536.0 * std::f32::consts::TAU
}

// ─── Сценарий 1 «Дикий лес» ───────────────────────────────────────────────

/// Счётчики сна-бодрствования одного существа за прогон: окна бодрствования
/// (длина в тиках, между двумя снами) и число снов. Дополняет снапшотный
/// sleep_frac точной метрикой «бодрое окно» (критерий приёмки: ≥ 30 тиков).
#[derive(Clone, Default)]
struct SleepStats {
    wake_windows: Vec<u32>,
    sleeps: u32,
    sleep_ticks: u32,
}

fn run_wild(total_ticks: u64) -> (Vec<Snap>, u64, Vec<WildEnd>, Vec<SleepStats>) {
    let mut arena = Arena::new(WILD_SEED, wild_map());
    // 6 существ в разнесённых позициях (по 2 в каждой комнате), курсы из сидов.
    let spawns: [(f32, f32); 6] = [
        (8.0, 7.0),
        (32.0, 7.0),
        (48.0, 7.0),
        (72.0, 7.0),
        (8.0, 21.0),
        (32.0, 21.0),
    ];
    for (i, &(x, y)) in spawns.iter().enumerate() {
        let seed = WILD_CREATURE_SEEDS[i];
        arena.spawn(Vec2::new(x, y), heading_from_seed(seed), seed);
    }
    // 10 ягод, разбросанных по трём комнатам.
    let berries: [(f32, f32); WILD_BERRIES] = [
        (5.0, 4.0),
        (15.0, 10.0),
        (25.0, 3.0),
        (35.0, 11.0), // комната A
        (45.0, 5.0),
        (60.0, 10.0),
        (72.0, 4.0), // комната B
        (5.0, 20.0),
        (20.0, 25.0),
        (33.0, 17.0), // комната C
    ];
    for &(x, y) in &berries {
        arena.add_berry(Vec2::new(x, y));
    }

    let mut snaps = Vec::new();
    snaps.push(snapshot(&arena));

    let mut n = arena.critters.len();
    let mut prev_food: Vec<u32> = arena.critters.iter().map(|c| c.creature.food_eaten()).collect();
    let mut follow_ticks: u64 = 0;
    // Окна бодрствования по существам (по-тиковый счёт, не снапшоты).
    let mut was_sleeping = vec![false; n];
    let mut cur_wake = vec![0u32; n];
    let mut sleep_stats: Vec<SleepStats> = vec![SleepStats::default(); n];
    for t in 1..=total_ticks {
        arena.step();
        // Новорождённые появляются посреди прогона (спаривание) — массивы
        // наблюдения растягиваются под текущую популяцию (новые — нули).
        if arena.critters.len() != n {
            n = arena.critters.len();
            prev_food.resize(n, 0);
            was_sleeping.resize(n, false);
            cur_wake.resize(n, 0);
            sleep_stats.resize(n, SleepStats::default());
        }
        // Прокси «следования» по тикам (дополнение к снапшотному): бодрствующий
        // не-едок в радиусе 6.0 от едока в тот же тик. Едок — существо,
        // у которого food_eaten вырос за этот тик.
        let eaters: Vec<usize> = (0..n)
            .filter(|&i| arena.critters[i].creature.food_eaten() > prev_food[i])
            .collect();
        if !eaters.is_empty() {
            for j in 0..n {
                if eaters.contains(&j) || arena.critters[j].sleeping {
                    continue;
                }
                let pj = arena.critters[j].pos;
                if eaters
                    .iter()
                    .any(|&i| arena.critters[i].pos.dist(pj) <= SOCIAL_PROXY_RADIUS)
                {
                    follow_ticks += 1;
                }
            }
        }
        for (i, c) in arena.critters.iter().enumerate() {
            prev_food[i] = c.creature.food_eaten();
            let sleeping = c.sleeping;
            let st = &mut sleep_stats[i];
            if sleeping {
                st.sleep_ticks += 1;
                if !was_sleeping[i] {
                    st.sleeps += 1;
                }
                if cur_wake[i] > 0 {
                    // Окно бодрствования закрылось сном.
                    st.wake_windows.push(cur_wake[i]);
                    cur_wake[i] = 0;
                }
            } else {
                cur_wake[i] += 1;
                if !was_sleeping[i] {
                    // Проснулся — окно уже открыто (cur_wake только что ++).
                }
            }
            was_sleeping[i] = sleeping;
        }
        if t % SNAPSHOT_EVERY == 0 {
            snaps.push(snapshot(&arena));
        }
    }
    // Хвост: последнее бодрствование (не закрытое сном) тоже окно.
    for i in 0..n {
        if cur_wake[i] > 0 {
            sleep_stats[i].wake_windows.push(cur_wake[i]);
        }
    }

    let ends = arena
        .critters
        .iter()
        .map(|c| {
            let insts = c.creature.learned_instincts();
            WildEnd {
                id: c.id.0,
                instincts: insts.len(),
                dopamines: insts.iter().map(|i| i.dopamine).collect(),
                places: c.creature.places_remembered(),
                food: c.creature.food_eaten(),
                alive: c.creature.alive(),
            }
        })
        .collect();
    (snaps, follow_ticks, ends, sleep_stats)
}

fn snapshot(arena: &Arena) -> Snap {
    Snap {
        tick: arena.tick,
        berry_count: arena.berries().len(),
        critters: arena
            .critters
            .iter()
            .map(|c| SnapCrit {
                id: c.id.0,
                pos: c.pos,
                heading: c.heading,
                food: c.creature.food_eaten(),
                wall: c.wall_hits,
                sleeping: c.sleeping,
                hunger: c.creature.hunger(),
                glucose: c.creature.glucose(),
                fatigue: c.creature.fatigue(),
                alive: c.creature.alive(),
                drive: c.last_drive,
                rate_motor: c.last_rate_motor,
                coat: {
                    let coat = c.creature.dna.coat().unwrap_or_default();
                    [coat.r, coat.g, coat.b]
                },
            })
            .collect(),
    }
}

// ─── Сценарий 2 «Поколения» ───────────────────────────────────────────────

fn run_evolution(gens: u64) -> (Vec<GenerationStats>, Vec<f32>) {
    let mut evo = Evolution::new(EVO_SEED, wild_map(), EvolutionParams::default());
    // Разнообразие геномов по поколениям: [стартовое, после gen1, ... после
    // последней]. Анти-фиксация = метрика не схлопывается (см. эволюцию).
    let mut diversity = vec![evo.population_diversity()];
    for _ in 0..gens {
        evo.run_generation();
        diversity.push(evo.population_diversity());
    }
    (evo.history.clone(), diversity)
}

// ─── Метрики (считаются в раннере, кладутся в JSON) ───────────────────────

/// Границы третей в интервалах снапшотов: [0,t), [t,2t), [2t,intervals).
fn third_bounds(intervals: usize) -> [(usize, usize); 3] {
    let t = (intervals / 3).max(1);
    [(0, t), (t, 2 * t), (2 * t, intervals)]
}

/// Метрики существа `ci` на интервалах снапшотов i0..i1 (точки i0..=i1).
fn third_metrics(snaps: &[Snap], ci: usize, i0: usize, i1: usize) -> ThirdMetrics {
    let mut path = 0.0f64;
    for k in i0..i1 {
        path += snaps[k].critters[ci].pos.dist(snaps[k + 1].critters[ci].pos) as f64;
    }
    let f0 = &snaps[i0].critters[ci];
    let f1 = &snaps[i1].critters[ci];
    let n_int = (i1 - i0) as f64;
    let dt = n_int * SNAPSHOT_EVERY as f64;
    let net = f0.pos.dist(f1.pos) as f64;
    let sleep = snaps[i0..i1]
        .iter()
        .filter(|s| s.critters[ci].sleeping)
        .count() as f64
        / n_int;
    let headings: Vec<f32> = snaps[i0..i1]
        .iter()
        .map(|s| s.critters[ci].heading)
        .collect();
    ThirdMetrics {
        food_rate: (f1.food as f64 - f0.food as f64) / n_int,
        wall_hit_rate: (f1.wall as f64 - f0.wall as f64) / n_int,
        sleep_frac: sleep,
        mean_speed: if dt > 0.0 { path / dt } else { 0.0 },
        path_len: path,
        net_displacement: net,
        tortuosity: if path > 1e-6 { (net / path).min(1.0) } else { 0.0 },
        heading_entropy: heading_entropy(&headings),
    }
}

/// Энтропия курса (8 бинов, биты).
fn heading_entropy(headings: &[f32]) -> f64 {
    let n = headings.len();
    if n == 0 {
        return 0.0;
    }
    let mut bins = [0usize; HEADING_BINS];
    for &h in headings {
        let b = ((h * HEADING_BINS as f32 / std::f32::consts::TAU).floor() as i64)
            .rem_euclid(HEADING_BINS as i64) as usize;
        bins[b] += 1;
    }
    let mut e = 0.0f64;
    for &b in &bins {
        if b > 0 {
            let p = b as f64 / n as f64;
            e -= p * p.log2();
        }
    }
    e
}

/// Сетка занятости GRID×GRID по bbox карты: 1 = существо было в клетке.
fn occupancy_grid(points: &[Vec2]) -> Vec<f64> {
    let mut occ = vec![0.0f64; GRID * GRID];
    for p in points {
        let cx = (((p.x / MAP_X) * GRID as f32).floor() as i64).clamp(0, GRID as i64 - 1);
        let cy = (((p.y / MAP_Y) * GRID as f32).floor() as i64).clamp(0, GRID as i64 - 1);
        occ[cy as usize * GRID + cx as usize] = 1.0;
    }
    occ
}

/// Жаккар двух бинарных сеток занятости.
fn jaccard(a: &[f64], b: &[f64]) -> f64 {
    let (mut inter, mut union) = (0usize, 0usize);
    for (x, y) in a.iter().zip(b) {
        if *x > 0.0 || *y > 0.0 {
            union += 1;
            if *x > 0.0 && *y > 0.0 {
                inter += 1;
            }
        }
    }
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

fn mean(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        0.0
    } else {
        vals.iter().sum::<f64>() / vals.len() as f64
    }
}

/// Выборочная дисперсия (n-1); < 2 точек — 0.
fn sample_variance(vals: &[f64]) -> f64 {
    let n = vals.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(vals);
    vals.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1) as f64
}

/// NN-дистанции: для каждого существа на каждом снапшоте трети — расстояние
/// до ближайшего другого существа; пул по трети.
fn nn_distances(snaps: &[Snap], i0: usize, i1: usize, n: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for s in &snaps[i0..i1] {
        for i in 0..n {
            let mut best = f64::INFINITY;
            for j in 0..n {
                if i == j {
                    continue;
                }
                best = best.min(s.critters[i].pos.dist(s.critters[j].pos) as f64);
            }
            if best.is_finite() {
                out.push(best);
            }
        }
    }
    out
}

/// Метрики «личностей»: между-существая дисперсия против средней
/// внутри-существай (по окнам поздней трети) — простое F-ratio-иш.
struct Personality {
    between_var: Vec<f64>,
    within_var: Vec<f64>,
    f_ratio: Vec<f64>,
    over_2x: usize,
    windows: usize,
}

fn personality(snaps: &[Snap], n: usize, late: (usize, usize)) -> Personality {
    let (i0, i1) = late;
    let n_int = i1 - i0;
    let windows = 5.min(n_int.max(1));
    let step = (n_int + windows - 1) / windows;
    let metric_count = 5usize;
    // [creature][metric][window]
    let mut win_vals = vec![vec![Vec::new(); metric_count]; n];
    for w in 0..windows {
        let a = i0 + w * step;
        let b = (a + step).min(i1);
        if b <= a {
            continue;
        }
        for ci in 0..n {
            let m = third_metrics(snaps, ci, a, b);
            let v = [
                m.food_rate,
                m.tortuosity,
                m.mean_speed,
                m.sleep_frac,
                m.wall_hit_rate,
            ];
            for (k, &x) in v.iter().enumerate() {
                win_vals[ci][k].push(x);
            }
        }
    }
    let mut between_var = Vec::with_capacity(metric_count);
    let mut within_var = Vec::with_capacity(metric_count);
    let mut f_ratio = Vec::with_capacity(metric_count);
    let mut over_2x = 0usize;
    // < 2 окон: внутри-существая дисперсия неизмерима — F=0 (нет улик),
    // а не 1e6 (постоянство по построению — не признак личности).
    let enough_windows = windows >= 2;
    for k in 0..metric_count {
        let cmeans: Vec<f64> = (0..n).map(|ci| mean(&win_vals[ci][k])).collect();
        let between = sample_variance(&cmeans);
        let wv: Vec<f64> = (0..n).map(|ci| sample_variance(&win_vals[ci][k])).collect();
        let within = mean(&wv);
        let f = if !enough_windows {
            0.0
        } else if within > 1e-9 {
            between / within
        } else if between > 1e-9 {
            1e6 // постоянство внутри + разброс между = максимальная «личность»
        } else {
            1.0
        };
        if f > 2.0 {
            over_2x += 1;
        }
        between_var.push(between);
        within_var.push(within);
        f_ratio.push(f);
    }
    Personality {
        between_var,
        within_var,
        f_ratio,
        over_2x,
        windows,
    }
}

/// Прокси «следования» по снапшотам (по ТЗ): на каждом интервале едоки —
/// существа, съевшие за интервал; не-едок в радиусе 6.0 от едока на
/// снапшоте конца интервала = эпизод следования. Доля от слотов
/// (интервалы × бодрствующие не-едоки).
fn follow_snapshot(snaps: &[Snap], n: usize) -> (f64, usize, usize) {
    let intervals = snaps.len() - 1;
    let mut events = 0usize;
    let mut slots = 0usize;
    for k in 0..intervals {
        let eaters: Vec<usize> = (0..n)
            .filter(|&i| snaps[k + 1].critters[i].food > snaps[k].critters[i].food)
            .collect();
        for j in 0..n {
            if eaters.contains(&j) || snaps[k + 1].critters[j].sleeping {
                continue;
            }
            slots += 1;
            let pj = snaps[k + 1].critters[j].pos;
            if eaters
                .iter()
                .any(|&i| snaps[k + 1].critters[i].pos.dist(pj) <= SOCIAL_PROXY_RADIUS)
            {
                events += 1;
            }
        }
    }
    let rate = if slots > 0 {
        events as f64 / slots as f64
    } else {
        0.0
    };
    (rate, events, slots)
}

// ─── JSON ─────────────────────────────────────────────────────────────────

fn json_f(x: f64) -> String {
    format!("{x:.4}")
}

fn json_f32(x: f32) -> String {
    format!("{x:.4}")
}

fn json_third_metrics(m: &ThirdMetrics, occ: &[f64]) -> String {
    format!(
        "{{\"food_rate\":{},\"wall_hit_rate\":{},\"sleep_frac\":{},\"mean_speed\":{},\
\"path_len\":{},\"net_displacement\":{},\"tortuosity\":{},\"heading_entropy\":{},\
\"occupancy\":[{}]}}",
        json_f(m.food_rate),
        json_f(m.wall_hit_rate),
        json_f(m.sleep_frac),
        json_f(m.mean_speed),
        json_f(m.path_len),
        json_f(m.net_displacement),
        json_f(m.tortuosity),
        json_f(m.heading_entropy),
        occ.iter().map(|v| json_f(*v)).collect::<Vec<_>>().join(",")
    )
}

/// Секция "metrics" целиком (объект JSON).
fn compute_metrics(
    snaps: &[Snap],
    ends: &[WildEnd],
    follow_ticks: u64,
    total_ticks: u64,
    quick: bool,
    evo: &[GenerationStats],
    evo_diversity: &[f32],
    sleep: &[SleepStats],
) -> String {
    let n = snaps[0].critters.len();
    let intervals = snaps.len() - 1;
    let bounds = third_bounds(intervals);
    let names = ["early", "mid", "late"];

    // По-существая статистика третей + занятость.
    let mut per_creature = String::new();
    let mut occ_by_third = vec![Vec::new(); 3]; // [third][creature] -> Vec<f64>
    for ci in 0..n {
        if ci > 0 {
            per_creature.push(',');
        }
        per_creature.push_str(&format!(
            "\"{}\":{{\"id\":{},\"seed\":{},\"thirds\":{{",
            snaps[0].critters[ci].id,
            snaps[0].critters[ci].id,
            WILD_CREATURE_SEEDS[ci]
        ));
        for (k, &(i0, i1)) in bounds.iter().enumerate() {
            if k > 0 {
                per_creature.push(',');
            }
            let points: Vec<Vec2> = snaps[i0..i1].iter().map(|s| s.critters[ci].pos).collect();
            let occ = occupancy_grid(&points);
            occ_by_third[k].push(occ.clone());
            let m = third_metrics(snaps, ci, i0, i1);
            per_creature.push_str(&format!(
                "\"{}\":{}",
                names[k],
                json_third_metrics(&m, &occ)
            ));
        }
        // Поздняя треть — «отпечаток личности».
        let (li0, li1) = bounds[2];
        let lm = third_metrics(snaps, ci, li0, li1);
        per_creature.push_str(&format!(
            "}},\"fingerprint\":{{\"food_rate\":{},\"tortuosity\":{},\"mean_speed\":{},\
\"sleep_frac\":{},\"wall_hit_rate\":{}}},",
            json_f(lm.food_rate),
            json_f(lm.tortuosity),
            json_f(lm.mean_speed),
            json_f(lm.sleep_frac),
            json_f(lm.wall_hit_rate)
        ));
        // Кривые: еда (накопительно) и сон (0/1) по снапшотам.
        let food_curve: Vec<String> = snaps
            .iter()
            .map(|s| format!("{}", s.critters[ci].food))
            .collect();
        let sleep_curve: Vec<String> = snaps
            .iter()
            .map(|s| if s.critters[ci].sleeping { "1".to_string() } else { "0".to_string() })
            .collect();
        per_creature.push_str(&format!(
            "\"food_over_time\":[{}],\"sleep_over_time\":[{}],",
            food_curve.join(","),
            sleep_curve.join(",")
        ));
        // Конец жизни.
        let e = &ends[ci];
        let dop: Vec<String> = e.dopamines.iter().map(|d| json_f32(*d)).collect();
        per_creature.push_str(&format!(
            "\"end\":{{\"learned_instincts\":{},\"instinct_dopamines\":[{}],\
\"places_remembered\":{},\"food_eaten\":{},\"alive\":{}}}}}",
            e.instincts,
            dop.join(","),
            e.places,
            e.food,
            e.alive
        ));
    }

    // Агрегат по третям (среднее по существам) — для вердиктов.
    let mut aggregate = String::new();
    for (k, &(i0, i1)) in bounds.iter().enumerate() {
        let mut agg = ThirdMetrics::default();
        let mut cnt = 0usize;
        for ci in 0..n {
            let m = third_metrics(snaps, ci, i0, i1);
            agg.food_rate += m.food_rate;
            agg.wall_hit_rate += m.wall_hit_rate;
            agg.sleep_frac += m.sleep_frac;
            agg.mean_speed += m.mean_speed;
            agg.path_len += m.path_len;
            agg.net_displacement += m.net_displacement;
            agg.tortuosity += m.tortuosity;
            agg.heading_entropy += m.heading_entropy;
            cnt += 1;
        }
        let div = cnt.max(1) as f64;
        let s = format!(
            "\"{}\":{{\"food_rate\":{},\"wall_hit_rate\":{},\"sleep_frac\":{},\"mean_speed\":{},\
\"path_len\":{},\"net_displacement\":{},\"tortuosity\":{},\"heading_entropy\":{}}}",
            names[k],
            json_f(agg.food_rate / div),
            json_f(agg.wall_hit_rate / div),
            json_f(agg.sleep_frac / div),
            json_f(agg.mean_speed / div),
            json_f(agg.path_len / div),
            json_f(agg.net_displacement / div),
            json_f(agg.tortuosity / div),
            json_f(agg.heading_entropy / div)
        );
        if k > 0 {
            aggregate.push(',');
        }
        aggregate.push_str(&s);
    }

    // Попарное перекрытие занятости (ранняя vs поздняя треть).
    let mut overlap = String::new();
    for (k, third_idx) in [0usize, 2].iter().enumerate() {
        let mut pair_s = String::new();
        let mut vals = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let jac = jaccard(&occ_by_third[*third_idx][i], &occ_by_third[*third_idx][j]);
                vals.push(jac);
                if !pair_s.is_empty() {
                    pair_s.push(',');
                }
                pair_s.push_str(&format!(
                    "[{},{},{}]",
                    snaps[0].critters[i].id,
                    snaps[0].critters[j].id,
                    json_f(jac)
                ));
            }
        }
        let tag = if k == 0 { "early" } else { "late" };
        if k > 0 {
            overlap.push(',');
        }
        overlap.push_str(&format!(
            "\"{}\":{{\"mean\":{},\"pairwise\":[{}]}}",
            tag,
            json_f(mean(&vals)),
            pair_s
        ));
    }

    // NN-дистанции по третям.
    let mut nn = String::new();
    for (k, &(i0, i1)) in bounds.iter().enumerate() {
        let vals = nn_distances(snaps, i0, i1, n);
        let vals_s: Vec<String> = vals.iter().map(|v| json_f(*v)).collect();
        if k > 0 {
            nn.push(',');
        }
        nn.push_str(&format!(
            "\"{}\":{{\"mean\":{},\"values\":[{}]}}",
            names[k],
            json_f(mean(&vals)),
            vals_s.join(",")
        ));
    }

    // Личности.
    let p = personality(snaps, n, bounds[2]);
    let metric_names = ["food_rate", "tortuosity", "mean_speed", "sleep_frac", "wall_hit_rate"];
    let p_json: Vec<String> = (0..5)
        .map(|k| {
            format!(
                "{{\"metric\":\"{}\",\"between_var\":{},\"within_var\":{},\"f_ratio\":{}}}",
                metric_names[k],
                json_f(p.between_var[k]),
                json_f(p.within_var[k]),
                json_f(p.f_ratio[k])
            )
        })
        .collect();

    // Следование.
    let (follow_rate, follow_events, follow_slots) = follow_snapshot(snaps, n);
    let tick_rate = if total_ticks > 0 {
        follow_ticks as f64 / total_ticks as f64
    } else {
        0.0
    };

    // Эволюция: тренды и дельты.
    let evo_json = match (evo.first(), evo.last()) {
        (Some(first), Some(last)) => {
        let trend = |f: &dyn Fn(&GenerationStats) -> f64| -> Vec<String> {
            evo.iter().map(|s| format!("{:.4}", f(s))).collect()
        };
        let avg_food: Vec<String> = trend(&|s| s.avg_food_eaten);
        let with_instinct: Vec<String> = evo.iter().map(|s| format!("{}", s.with_instinct)).collect();
        let dopamine: Vec<String> = trend(&|s| s.instinct_dopamine_avg);
        let survivors: Vec<String> = evo.iter().map(|s| format!("{}", s.survivors)).collect();
        let born: Vec<String> = evo.iter().map(|s| format!("{}", s.born)).collect();
        // Разнообразие геномов: [стартовая популяция, после gen1, …].
        let diversity: Vec<String> = evo_diversity.iter().map(|d| format!("{:.4}", d)).collect();
        let delta_avg_food = last.avg_food_eaten - first.avg_food_eaten;
        let ratio_avg_food = if first.avg_food_eaten > 0.01 {
            last.avg_food_eaten / first.avg_food_eaten
        } else if last.avg_food_eaten > 0.01 {
            10.0
        } else {
            0.0
        };
        let div0 = evo_diversity.first().copied().unwrap_or(0.0);
        let div_last = evo_diversity.last().copied().unwrap_or(0.0);
        format!(
            "{{\"gen_count\":{},\"avg_food_trend\":[{}],\"with_instinct_trend\":[{}],\
\"instinct_dopamine_trend\":[{}],\"survivors_trend\":[{}],\"born_trend\":[{}],\
\"avg_food_first\":{},\"avg_food_last\":{},\"delta_avg_food\":{},\"ratio_avg_food\":{},\
\"with_instinct_first\":{},\"with_instinct_last\":{},\"dopamine_first\":{},\"dopamine_last\":{},\
\"diversity_trend\":[{}],\"diversity_first\":{},\"diversity_last\":{},\
\"diversity_frac\":{}}}",
            evo.len(),
            avg_food.join(","),
            with_instinct.join(","),
            dopamine.join(","),
            survivors.join(","),
            born.join(","),
            json_f(first.avg_food_eaten),
            json_f(last.avg_food_eaten),
            json_f(delta_avg_food),
            json_f(ratio_avg_food),
            first.with_instinct,
            last.with_instinct,
            json_f(first.instinct_dopamine_avg),
            json_f(last.instinct_dopamine_avg),
            diversity.join(","),
            json_f(div0 as f64),
            json_f(div_last as f64),
            json_f(if div0 > 1e-6 { (div_last / div0) as f64 } else { 0.0 })
        )
        }
        (None, _) | (_, None) => "{\"gen_count\":0}".to_string(),
    };

    // Сон: бодрые окна по существам (по-тиковый счёт) + агрегат.
    let mut sleep_creatures = String::new();
    let mut all_windows: Vec<u32> = Vec::new();
    let mut sleep_frac_exact = 0.0f64;
    for (ci, s) in sleep.iter().enumerate() {
        if ci > 0 {
            sleep_creatures.push(',');
        }
        let mean = if s.wake_windows.is_empty() {
            0.0
        } else {
            s.wake_windows.iter().sum::<u32>() as f64 / s.wake_windows.len() as f64
        };
        sleep_creatures.push_str(&format!(
            "\"c{}\":{{\"sleeps\":{},\"sleep_ticks\":{},\"wake_windows\":{},\"wake_mean\":{}}}",
            ci,
            s.sleeps,
            s.sleep_ticks,
            s.wake_windows.len(),
            json_f(mean)
        ));
        all_windows.extend(s.wake_windows.iter().copied());
        sleep_frac_exact += s.sleep_ticks as f64;
    }
    let wake_mean = if all_windows.is_empty() {
        0.0
    } else {
        all_windows.iter().sum::<u32>() as f64 / all_windows.len() as f64
    };
    let sleep_frac_exact = if total_ticks > 0 && !sleep.is_empty() {
        sleep_frac_exact / (total_ticks as f64 * sleep.len() as f64)
    } else {
        0.0
    };

    format!(
        "{{\"config\":{{\"total_ticks\":{},\"snapshot_every\":{},\"intervals\":{},\
\"n_creatures\":{},\"n_berries\":{},\"wild_seed\":{},\"evo_seed\":{},\"evo_gens\":{},\"quick\":{},\
\"thirds\":{{\"early\":[0,{}],\"mid\":[{},{}],\"late\":[{},{}]}}}},\
\"aggregate\":{{{}}},\
\"per_creature\":{{{}}},\
\"overlap\":{{{}}},\
\"nn_distance\":{{{}}},\
\"personality\":{{\"metric_names\":[\"food_rate\",\"tortuosity\",\"mean_speed\",\"sleep_frac\",\"wall_hit_rate\"],\
\"per_metric\":[{}],\"over_2x\":{},\"windows\":{}}},\
\"follow\":{{\"snapshot_rate\":{},\"snapshot_events\":{},\"snapshot_slots\":{},\
\"tick_rate\":{},\"tick_events\":{},\"ticks\":{},\"radius\":{}}},\
\"sleep\":{{\"wake_mean\":{},\"wake_windows_total\":{},\"sleep_frac_exact\":{},\"per_creature\":{{{}}}}},\
\"evolution\":{}}}",
        total_ticks,
        SNAPSHOT_EVERY,
        intervals,
        n,
        WILD_BERRIES,
        WILD_SEED,
        EVO_SEED,
        evo.len(),
        quick,
        bounds[0].1 as u64 * SNAPSHOT_EVERY,
        bounds[1].0 as u64 * SNAPSHOT_EVERY,
        bounds[1].1 as u64 * SNAPSHOT_EVERY,
        bounds[2].0 as u64 * SNAPSHOT_EVERY,
        bounds[2].1 as u64 * SNAPSHOT_EVERY,
        aggregate,
        per_creature,
        overlap,
        nn,
        p_json.join(","),
        p.over_2x,
        p.windows,
        json_f(follow_rate),
        follow_events,
        follow_slots,
        json_f(tick_rate),
        follow_ticks,
        total_ticks,
        SOCIAL_PROXY_RADIUS,
        json_f(wake_mean),
        all_windows.len(),
        json_f(sleep_frac_exact),
        sleep_creatures,
        evo_json
    )
}

/// Сборка полного JSON.
fn build_json(
    quick: bool,
    total_ticks: u64,
    snaps: &[Snap],
    ends: &[WildEnd],
    evo: &[GenerationStats],
    evo_diversity: &[f32],
    sleep: &[SleepStats],
    follow_ticks: u64,
) -> String {
    let mut json = String::with_capacity(1 << 20);

    json.push_str("{\n");
    json.push_str("  \"version\": \"emergence_v1\",\n");
    json.push_str("  \"note\": \"закрытый контур: сценарии 'Дикий лес' (естественный прогон, без инъекций) и 'Поколения' (Evolution); метрики считает examples/emergence_run.rs, отчёт рисует tools/emergence_report.py\",\n");
    json.push_str(&format!("  \"quick\": {quick},\n"));
    json.push_str("  \"scenarios\": {\n");

    // Сценарий 1: снапшоты (последний несёт "end").
    json.push_str("    \"wild\": [\n");
    for (idx, s) in snaps.iter().enumerate() {
        json.push_str("      {\"tick\":");
        json.push_str(&format!("{},", s.tick));
        json.push_str(&format!("\"berry_count\":{},", s.berry_count));
        json.push_str("\"critters\":[");
        for (ci, c) in s.critters.iter().enumerate() {
            if ci > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                "{{\"id\":{},\"pos\":[{},{}],\"heading\":{},\"food_eaten\":{},\"wall_hits\":{},\
\"sleeping\":{},\"hunger\":{},\"glucose\":{},\"fatigue\":{},\"drive\":{},\"rate_motor\":{},\"coat\":[{},{},{}],\"alive\":{}}}",
                c.id,
                json_f32(c.pos.x),
                json_f32(c.pos.y),
                json_f32(c.heading),
                c.food,
                c.wall,
                c.sleeping,
                json_f32(c.hunger),
                json_f32(c.glucose),
                json_f32(c.fatigue),
                json_f32(c.drive),
                json_f32(c.rate_motor),
                json_f32(c.coat[0]),
                json_f32(c.coat[1]),
                json_f32(c.coat[2]),
                c.alive
            ));
        }
        json.push(']');
        if idx == snaps.len() - 1 {
            // Конечное состояние существ (нажитое за жизнь).
            json.push_str(",\"end\":[");
            for (i, e) in ends.iter().enumerate() {
                if i > 0 {
                    json.push(',');
                }
                let dop: Vec<String> = e.dopamines.iter().map(|d| json_f32(*d)).collect();
                json.push_str(&format!(
                    "{{\"id\":{},\"learned_instincts\":{},\"instinct_dopamines\":[{}],\
\"places_remembered\":{},\"food_eaten\":{},\"alive\":{}}}",
                    e.id,
                    e.instincts,
                    dop.join(","),
                    e.places,
                    e.food,
                    e.alive
                ));
            }
            json.push(']');
        }
        json.push_str("}");
        if idx < snaps.len() - 1 {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str("    ],\n");

    // Сценарий 2: история поколений.
    json.push_str("    \"evo\": [\n");
    for (i, g) in evo.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "{{\"generation\":{},\"survivors\":{},\"born\":{},\"avg_food_eaten\":{},\
\"max_food_eaten\":{},\"with_instinct\":{},\"instinct_dopamine_avg\":{}}}",
            g.generation,
            g.survivors,
            g.born,
            json_f(g.avg_food_eaten),
            g.max_food_eaten,
            g.with_instinct,
            json_f(g.instinct_dopamine_avg)
        ));
    }
    json.push_str("\n    ]\n");
    json.push_str("  },\n");

    // Метрики.
    json.push_str(&format!(
        "  \"metrics\": {}\n",
        compute_metrics(snaps, ends, follow_ticks, total_ticks, quick, evo, evo_diversity, sleep)
    ));
    json.push_str("}\n");
    json
}

fn main() {
    let mut out = PathBuf::from("tools/emergence_data.json");
    let mut quick = false;
    let mut wild_ticks_override: Option<u64> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || {
            it.next().unwrap_or_else(|| {
                eprintln!("аргумент {a} требует значение");
                std::process::exit(2);
            })
        };
        match a.as_str() {
            "--out" => out = PathBuf::from(val()),
            "--quick" => quick = true,
            "--ticks" => wild_ticks_override = Some(val().parse::<u64>().unwrap_or(WILD_TICKS)),
            "--help" | "-h" => {
                println!(
                    "emergence_run — закрытый контур: «Дикий лес» + «Поколения» + метрики\n\
                     \n\
                     Использование:\n  \
                     cargo run --release --example emergence_run -- [--out PATH] [--quick]\n\
                     \n\
                     Параметры:\n  \
                     --out PATH  выходной JSON (по умолчанию tools/emergence_data.json)\n  \
                     --quick     смоук: 3000 тиков «Дикого леса», 2 поколения\n\
                     --ticks N   длина «Дикого леса» в тиках (дефолт 30000)\n\
                     \n\
                     Дальше: python tools/emergence_report.py\n  \
                     рисует tools/emergence_report.html (самодостаточный, тёмная тема)."
                );
                return;
            }
            other => {
                eprintln!("неизвестный аргумент: {other}");
                std::process::exit(2);
            }
        }
    }

    let wild_ticks = wild_ticks_override.unwrap_or(if quick { WILD_TICKS_QUICK } else { WILD_TICKS });
    let evo_gens = if quick { EVO_GENS_QUICK } else { EVO_GENS };

    println!("Эмерджентность: закрытый контур (quick={quick})");
    println!(
        "— «Дикий лес»: {wild_ticks} тиков, сид {WILD_SEED}, 6 существ (сиды 3001..3006), {WILD_BERRIES} ягод, без инъекций голода"
    );
    println!(
        "— «Поколения»: {evo_gens} поколений, сид {EVO_SEED}, EvolutionParams::default()"
    );

    let (wild_snaps, follow_ticks, ends, sleep) = run_wild(wild_ticks);
    println!(
        "  «Дикий лес» отработал: {} снапшотов, эпифеноменов «следования» по тикам: {follow_ticks}",
        wild_snaps.len()
    );
    let (evo_stats, evo_diversity) = run_evolution(evo_gens);
    println!("  «Поколения» отработали: {} поколений", evo_stats.len());

    let json = build_json(
        quick,
        wild_ticks,
        &wild_snaps,
        &ends,
        &evo_stats,
        &evo_diversity,
        &sleep,
        follow_ticks,
    );

    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    std::fs::write(&out, &json).expect("запись JSON");
    let size = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!("записано: {out:?} ({size} байт)");
}
