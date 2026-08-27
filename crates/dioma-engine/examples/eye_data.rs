//! # eye_data — рекордер датасета «Глаза» (этап 0 пайплайна)
//!
//! Пайплайн: `eye_data.rs` (этот рекордер) → `tools/eye_data.json` →
//! `tools/train_eye.py` (MLP 48→24→6) → `tools/eye.dbw` →
//! `eye_check.rs` (паритет Python↔Rust + качество на hold-out).
//!
//! ## Концепт (см. docs/BRAIN_ROADMAP.md)
//!
//! Сегодня vis_l/vis_r считает геометрия в `Arena::sense`. Глаза будущего —
//! энкодер «сырая сетчатка → сигналы»: ретина (16 лучей по конусу) богаче
//! бинов, ANN сжимает её в полезный вектор. Сейчас учим ДИСТИЛЛЯЦИЕЙ:
//! цель = аналитические сигналы. Потом (Стадия C) ретина заменит бины,
//! а запечённые глаза встанут как энкодер перед VL/VR.
//!
//! ## Запуск
//!
//! Из корня репозитория (относительные пути `tools/…`):
//!
//! ```text
//! cargo run --manifest-path crates/dioma-engine/Cargo.toml --release --example eye_data -- \
//!   --creatures 8 --ticks 1500 --seed0 2000 --out tools/eye_data.json
//! ```
//!
//! (или из `crates/dioma-engine`: `--out ../../tools/eye_data.json`).
//! Детерминизм: арены и существа — от фиксированных сидов (`seed0 + i`),
//! `Arena::step` параллелен (rayon) и бит-в-бит воспроизводим; повторный
//! запуск с теми же аргументами даёт тот же файл.
//!
//! ## Ретина (вход, 48 f32) — 16 лучей по конусу ±75° вокруг курса
//!
//! 16 лучей равномерно от −75° до +75° (шаг 10°); у луча 3 канала:
//! `[ягодность, стенность, соседность]`:
//!
//! - **стенность**: ray-march как `wall_probe_world` (шаг 1.0, дальность
//!   25.0): `1 − d/25` в точке первого выхода из комнаты;
//! - **ягодность**: для каждой СЪЕДОБНОЙ ягоды в конусе: угловое расстояние
//!   до луча < полширины бина (360/16/2 = 11.25°) и дистанция ≤ 40 →
//!   вклад `(1 − d/40)`; сумма по ягодам, clamp 1;
//! - **соседность**: как ягодность, но по ДРУГИМ существам.
//!
//! Отклонение от «голой» спеки — **стены непрозрачны и для ретины**: ягода/
//! сосед в ДРУГОЙ комнате не виден (та же проверка комнаты, что в
//! `Arena::sense`). Иначе ретина видела бы сквозь стену, а цель vis_l/vis_r —
//! нет, и дистилляция была бы невозможна.
//!
//! ## Статичный наблюдатель (трюк)
//!
//! В арене 1 криттер — канал «соседность» был бы нулевым всегда. Для данных
//! добавляем 2-го «статичного» криттера-наблюдателя: обычный `spawn` на
//! фиксированной позиции (якорь), но после каждого `step` его позиция/курс
//! ВОЗВРАЩАЮТСЯ на якорь. Мозг у него настоящий (ест, спит, крутит кровью —
//! детерминированно), но положение неизменно: канал соседства живой
//! (пеленг/дистанция зависят от позиции наблюдателя) и стабильный.
//!
//! ## Цели (выход, 6 f32)
//!
//! - `vis_l`, `vis_r`, `wall_ahead = max(wall_l, wall_r)` — КОПИЯ значений
//!   [`Arena::last_sense(0)`] (ровно аналитика, из которой мозг действовал);
//! - `food_bearing_sin/cos` — нормированный пеленг (sin/cos относительно
//!   курса) на БЛИЖАЙШУЮ съедобную ягоду В ПОЛЕ ЗРЕНИЯ (та же комната,
//!   |пеленг| ≤ 75°, d ≤ 40); видимой нет — 0,0. Вне поля зрения глаз
//!   не видит — и энкодер не должен выдумывать;
//! - `food_dist_norm` — `(1 − d/40).clamp(0,1)` до той же ягоды; 0 если нет.
//!
//! ## Прогрев и сон
//!
//! Первые [`WARMUP_TICKS`] тиков — прогрев (в запись не входят): статичный
//! наблюдатель и цикл отрастания ягод входят в устойчивое состояние.
//! Спящий криттер «глаза не открывает»: и ретина, и цели нулевые (спящему
//! в `Arena::step` Sense не собирается). Сон предсказывается ДО шага по тем
//! же правилам, что фаза 0.5 `Arena::step` — ретина и цели всегда
//! согласованы в одном семпле.

use dioma_engine::creature::Sense;
use dioma_engine::env::map::WorldMap;
use dioma_engine::env::{Substance, Vec2, SUBSTANCES};
use dioma_engine::sim::Arena;
use std::f32::consts::PI;
use std::f32::consts::TAU;
use std::fs;
use std::path::PathBuf;

// ─── Геометрия ретины ──────────────────────────────────────────────────────

/// Число лучей в конусе (спека: 16).
const RETINA_RAYS: usize = 16;
/// Вход ретины: 3 канала на луч.
const IN_DIM: usize = RETINA_RAYS * 3;
/// Выход: vis_l, vis_r, food_bearing_sin, food_bearing_cos, food_dist_norm, wall_ahead.
const OUT_DIM: usize = 6;
/// Полуполе зрения, градусы (±75° — как `VISION_HALF_FOV` в движке).
const FOV_DEG: f32 = 75.0;
/// Шаг между лучами: (2·75°)/(16−1) = 10° — лучи равномерно от −75° до +75°.
const RAY_STEP_DEG: f32 = 2.0 * FOV_DEG / (RETINA_RAYS as f32 - 1.0);
/// Полширины бина (спека: 360/16/2 = 11.25°).
const BIN_HALF_DEG: f32 = 360.0 / RETINA_RAYS as f32 / 2.0;
/// Дальность ягодного/соседского канала (как `VISION_RANGE` движка).
const VISION_RANGE: f32 = 40.0;
/// Дальность ray-march стен (спека: 25.0; движок `wall_probe_world` — 8.0).
const WALL_RANGE: f32 = 25.0;
/// Шаг ray-march (как `WALL_PROBE_STEP` движка).
const WALL_PROBE_STEP: f32 = 1.0;

const FOV_RAD: f32 = FOV_DEG * PI / 180.0;
const RAY_STEP_RAD: f32 = RAY_STEP_DEG * PI / 180.0;
const BIN_HALF_RAD: f32 = BIN_HALF_DEG * PI / 180.0;

// ─── Арена / запись ────────────────────────────────────────────────────────

/// Точки ягод — как в `src/bin/game.rs` и `whisperer_data.rs`.
const BERRY_SPOTS: [(f32, f32); 6] = [
    (5.0, 5.0),
    (18.0, 10.0),
    (30.0, 4.0),
    (48.0, 5.0),
    (62.0, 10.0),
    (72.0, 4.0),
];

/// Якорь статичного криттера-наблюдателя (трюк — см. шапку файла).
const STATIC_POS: (f32, f32) = (20.0, 7.0);

/// Тиков прогрева до записи (см. шапку файла).
const WARMUP_TICKS: u32 = 50;

/// Порог сна — дубликат приватной `sim::SLEEP_AT_FATIGUE` (пример не может
/// читать приватные константы движка; значение зафиксировано в шапке
/// `crates/dioma-engine/src/sim/mod.rs`).
const SLEEP_AT_FATIGUE: f32 = 200.0;

/// Имена входных размерностей (канал-мейджор внутри луча, как в ретине).
fn input_names() -> Vec<String> {
    let mut v = Vec::with_capacity(IN_DIM);
    for i in 0..RETINA_RAYS {
        v.push(format!("berry_r{i}"));
        v.push(format!("wall_r{i}"));
        v.push(format!("neighbor_r{i}"));
    }
    v
}

/// Имена выходных размерностей.
const OUT_NAMES: [&str; OUT_DIM] = [
    "vis_l",
    "vis_r",
    "food_bearing_sin",
    "food_bearing_cos",
    "food_dist_norm",
    "wall_ahead",
];

/// Игровая карта: две комнаты 40×14 с общей стеной, плодородный климат —
/// ровно как `default_arena` в `src/bin/game.rs`.
fn game_map() -> WorldMap {
    let mut map = WorldMap::default();
    let a = map.add_room(0.0, 0.0, 40.0, 14.0);
    let b = map.add_room(40.0, 0.0, 80.0, 14.0);
    map.connect(a, b, [1.0; SUBSTANCES]);
    for room in &mut map.rooms {
        room.ca[Substance::Temperature.idx()] = 200.0;
        room.ca[Substance::Light.idx()] = 200.0;
        room.ca[Substance::Nutrients.idx()] = 200.0;
        room.ca[Substance::Humidity.idx()] = 150.0;
    }
    map
}

/// Угол луча `i` относительно курса (радианы): −75° + 10°·i.
fn ray_angle(i: usize) -> f32 {
    -FOV_RAD + RAY_STEP_RAD * i as f32
}

/// Угол, нормализованный в [−π, π].
fn wrap_rel(x: f32) -> f32 {
    (x + PI).rem_euclid(TAU) - PI
}

/// Будет ли криттер `i` получать Sense в следующем `Arena::step` (правила
/// фазы 0.5: спящий с оставшимися тиками сна спит; бодрствующий с усталостью
/// ≥ порога засыпает — и в этот тик сенсоров не получает, даже если только
/// что проснулся и `sleep_ticks` обнулился).
fn will_sense_this_tick(arena: &Arena, i: usize) -> bool {
    let c = &arena.critters[i];
    if c.sleeping && c.sleep_ticks > 0 {
        return false; // продолжает спать этот тик (фаза 0.5 декрементит и continue)
    }
    c.creature.fatigue() < SLEEP_AT_FATIGUE
}

/// Статичный наблюдатель: вернуть позицию/курс на якорь после каждого шага.
fn pin_static(arena: &mut Arena) {
    let c = &mut arena.critters[1];
    c.pos = Vec2::new(STATIC_POS.0, STATIC_POS.1);
    c.heading = 0.0;
}

/// Ретина наблюдателя `obs` (см. шапку файла): 16 лучей × [ягодность,
/// стенность, соседность]. Считается по геометрии ДО шага — ровно та же
/// картина, из которой мозг примет действие этого шага.
fn compute_retina(arena: &Arena, obs: usize) -> [f32; IN_DIM] {
    let mut out = [0.0f32; IN_DIM];
    let c = &arena.critters[obs];
    let pos = c.pos;
    let heading = c.heading;
    let my_room = arena.world.map.room_at(pos);
    let map = &arena.world.map;

    // Стенность: ray-march как wall_probe_world, но дальность 25.0.
    for i in 0..RETINA_RAYS {
        let a = heading + ray_angle(i);
        let dir = Vec2::new(a.cos(), a.sin());
        let mut d = WALL_PROBE_STEP;
        let wall = loop {
            if d > WALL_RANGE {
                break 0.0;
            }
            if map.room_at(pos + dir * d).is_none() {
                break 1.0 - d / WALL_RANGE;
            }
            d += WALL_PROBE_STEP;
        };
        out[i * 3 + 1] = wall;
    }

    // Ягодность: съедобные ягоды в конусе, та же комната, в бине луча.
    for &bid in &arena.berries() {
        let Some(b) = arena.berry_info(bid) else { continue };
        if !b.edible {
            continue;
        }
        if let Some(room) = my_room {
            if map.room_at(b.pos) != Some(room) {
                continue;
            }
        }
        let d = b.pos - pos;
        let dist = d.len();
        if dist > VISION_RANGE {
            continue;
        }
        let rel = wrap_rel(d.y.atan2(d.x) - heading);
        if rel.abs() > FOV_RAD {
            continue;
        }
        for i in 0..RETINA_RAYS {
            if wrap_rel(rel - ray_angle(i)).abs() < BIN_HALF_RAD {
                out[i * 3] += 1.0 - dist / VISION_RANGE;
            }
        }
    }
    for i in 0..RETINA_RAYS {
        out[i * 3] = out[i * 3].min(1.0);
    }

    // Соседность: как ягодность, но по другим существам (в данных — статичный
    // наблюдатель; сам себя глаз не видит).
    for (j, c2) in arena.critters.iter().enumerate() {
        if j == obs {
            continue;
        }
        if let Some(room) = my_room {
            if map.room_at(c2.pos) != Some(room) {
                continue;
            }
        }
        let d = c2.pos - pos;
        let dist = d.len();
        if dist > VISION_RANGE {
            continue;
        }
        let rel = wrap_rel(d.y.atan2(d.x) - heading);
        if rel.abs() > FOV_RAD {
            continue;
        }
        for i in 0..RETINA_RAYS {
            if wrap_rel(rel - ray_angle(i)).abs() < BIN_HALF_RAD {
                out[i * 3 + 2] += 1.0 - dist / VISION_RANGE;
            }
        }
    }
    for i in 0..RETINA_RAYS {
        out[i * 3 + 2] = out[i * 3 + 2].min(1.0);
    }

    out
}

/// Цели: vis_l/vis_r/wall_ahead — копия sense, пеленг/дистанция — геометрия
/// по съедобным ягодам в поле зрения (см. шапку файла).
fn compute_targets(arena: &Arena, obs: usize, sense: &Sense) -> [f32; OUT_DIM] {
    let mut t = [0.0f32; OUT_DIM];
    t[0] = sense.vis_l;
    t[1] = sense.vis_r;
    t[5] = sense.wall_l.max(sense.wall_r);

    let c = &arena.critters[obs];
    let pos = c.pos;
    let heading = c.heading;
    let my_room = arena.world.map.room_at(pos);
    let map = &arena.world.map;
    let mut best: Option<(f32, f32)> = None; // (дистанция, пеленг) ближайшей
    for &bid in &arena.berries() {
        let Some(b) = arena.berry_info(bid) else { continue };
        if !b.edible {
            continue;
        }
        if let Some(room) = my_room {
            if map.room_at(b.pos) != Some(room) {
                continue;
            }
        }
        let d = b.pos - pos;
        let dist = d.len();
        if dist > VISION_RANGE {
            continue;
        }
        let rel = wrap_rel(d.y.atan2(d.x) - heading);
        if rel.abs() > FOV_RAD {
            continue;
        }
        if best.map_or(true, |(bd, _)| dist < bd) {
            best = Some((dist, rel));
        }
    }
    if let Some((dist, rel)) = best {
        t[2] = rel.sin();
        t[3] = rel.cos();
        t[4] = (1.0 - dist / VISION_RANGE).clamp(0.0, 1.0);
    }
    t
}

// ─── JSON-вывод (детерминированный, как в whisperer_data.rs) ───────────────

/// Число как строка JSON (без -0.0; фиксированная точность — детерминизм).
fn json_f32(x: f32) -> String {
    debug_assert!(x.is_finite(), "в записи оказался не-finite float: {x}");
    let x = if x == 0.0 { 0.0 } else { x }; // -0.0 → 0.0
    format!("{x:.7}")
}

fn vec_json(v: &[f32]) -> String {
    let mut s = String::from("[");
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&json_f32(*x));
    }
    s.push(']');
    s
}

/// Один сэмпл: [[48 ретины], [6 целей]].
fn sample_json(retina: &[f32; IN_DIM], targets: &[f32; OUT_DIM]) -> String {
    format!("[{},{}]", vec_json(retina), vec_json(targets))
}

/// Счётчики живости ретины/целей (для отчёта и заголовка JSON).
#[derive(Default)]
struct Stats {
    samples: u64,
    berry_nonzero: u64,
    wall_nonzero: u64,
    neighbor_nonzero: u64,
    vis_nonzero: u64,
    bearing_nonzero: u64,
    wall_target_nonzero: u64,
}

impl Stats {
    fn add(&mut self, retina: &[f32; IN_DIM], targets: &[f32; OUT_DIM]) {
        self.samples += 1;
        for i in 0..RETINA_RAYS {
            if retina[i * 3] > 1e-6 {
                self.berry_nonzero += 1;
            }
            if retina[i * 3 + 1] > 1e-6 {
                self.wall_nonzero += 1;
            }
            if retina[i * 3 + 2] > 1e-6 {
                self.neighbor_nonzero += 1;
            }
        }
        if targets[0] > 1e-6 || targets[1] > 1e-6 {
            self.vis_nonzero += 1;
        }
        if targets[2] != 0.0 || targets[3] != 0.0 {
            self.bearing_nonzero += 1;
        }
        if targets[5] > 1e-6 {
            self.wall_target_nonzero += 1;
        }
    }

    fn header_json(&self) -> String {
        let frac = |n: u64| -> f32 {
            if self.samples == 0 {
                0.0
            } else {
                n as f32 / (self.samples as f32 * RETINA_RAYS as f32)
            }
        };
        let tf = |n: u64| -> f32 {
            if self.samples == 0 {
                0.0
            } else {
                n as f32 / self.samples as f32
            }
        };
        format!(
            "    \"stats\": {{\n\
             \x20     \"samples\": {},\n\
             \x20     \"retina_nonzero_berry\": {},\n\
             \x20     \"retina_nonzero_wall\": {},\n\
             \x20     \"retina_nonzero_neighbor\": {},\n\
             \x20     \"target_vis_nonzero\": {},\n\
             \x20     \"target_bearing_nonzero\": {},\n\
             \x20     \"target_wall_nonzero\": {}\n\
             \x20   }}",
            self.samples,
            json_f32(frac(self.berry_nonzero)),
            json_f32(frac(self.wall_nonzero)),
            json_f32(frac(self.neighbor_nonzero)),
            json_f32(tf(self.vis_nonzero)),
            json_f32(tf(self.bearing_nonzero)),
            json_f32(tf(self.wall_target_nonzero)),
        )
    }
}

fn main() {
    // ── Аргументы: --creatures N --ticks N --out PATH --seed0 N ──
    let mut creatures = 8usize;
    let mut ticks = 1500u32;
    let mut out = PathBuf::from("tools/eye_data.json");
    let mut seed0 = 2000u64;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| {
            eprintln!("аргумент {a} требует значение");
            std::process::exit(2);
        });
        match a.as_str() {
            "--creatures" => creatures = val().parse().expect("--creatures: число особей"),
            "--ticks" => ticks = val().parse().expect("--ticks: число тиков"),
            "--out" => out = PathBuf::from(val()),
            "--seed0" => seed0 = val().parse().expect("--seed0: базовый сид"),
            "--help" | "-h" => {
                println!(
                    "eye_data — рекордер датасета «Глаза» (ретина → аналитические сигналы)\n\
                     \n\
                     Использование:\n  \
                     cargo run --manifest-path crates/dioma-engine/Cargo.toml --release --example eye_data -- [--creatures N] [--ticks N] [--out PATH] [--seed0 N]\n\
                     \n\
                     Параметры:\n  \
                     --creatures N  число особей (по умолчанию 8)\n  \
                     --ticks N      записанных тиков на особь (по умолчанию 1500)\n  \
                     --out PATH     выходной JSON (по умолчанию tools/eye_data.json)\n  \
                     --seed0 N      базовый сид: арена/существо i получают seed0+i (по умолчанию 2000)"
                );
                return;
            }
            other => {
                eprintln!("неизвестный аргумент: {other}");
                std::process::exit(2);
            }
        }
    }
    if creatures == 0 || ticks == 0 {
        eprintln!("--creatures и --ticks должны быть > 0");
        std::process::exit(2);
    }
    // Тренер жёстко делит: особи seed0+0..5 — train, seed0+6..7 — hold-out.
    if creatures < 8 {
        eprintln!("для эталонного пайплайна нужно >= 8 особей (6 train + 2 hold-out)");
        std::process::exit(2);
    }

    println!(
        "запись «Глаз»: {creatures} особей × {ticks} тиков (после прогрева {WARMUP_TICKS}), \
         сиды {seed0}..{}, выход {out:?}",
        seed0 + creatures as u64 - 1
    );

    // ── Сборка JSON ──
    let names = input_names();
    let mut json = String::with_capacity(creatures * ticks as usize * 700);
    json.push_str("{\n");
    json.push_str("  \"version\": \"eye_data_v1\",\n");
    json.push_str("  \"note\": \"рекордер examples/eye_data.rs; ретина = 16 лучей x [ягодность, стенность, соседность], цели = аналитические сигналы (vis_l/vis_r/wall_ahead из Arena::last_sense + пеленг/дистанция на ближайшую съедобную ягоду в поле зрения)\",\n");
    json.push_str(&format!("  \"retina_rays\": {RETINA_RAYS},\n"));
    json.push_str(&format!("  \"in_dim\": {IN_DIM},\n"));
    json.push_str(&format!("  \"out_dim\": {OUT_DIM},\n"));
    json.push_str("  \"in_names\": [");
    for (i, n) in names.iter().enumerate() {
        if i > 0 {
            json.push_str(", ");
        }
        json.push_str(&format!("\"{n}\""));
    }
    json.push_str("],\n");
    json.push_str("  \"out_names\": [");
    for (i, n) in OUT_NAMES.iter().enumerate() {
        if i > 0 {
            json.push_str(", ");
        }
        json.push_str(&format!("\"{n}\""));
    }
    json.push_str("],\n");
    json.push_str(&format!(
        "  \"map\": \"two rooms 40x14 connected (game.rs), 6 berries, spawn (5,5)/0.0, static observer at ({:.0},{:.0})\",\n",
        STATIC_POS.0, STATIC_POS.1
    ));
    json.push_str(&format!("  \"warmup_ticks\": {WARMUP_TICKS},\n"));
    json.push_str(&format!(
        "  \"samples_per_creature\": {ticks},\n"
    ));
    json.push_str(&format!(
        "  \"train_creatures\": \"seed0+0..5\",\n  \"holdout_creatures\": \"seed0+6..7\",\n"
    ));
    json.push_str("  \"creatures\": [\n");

    let mut stats = Stats::default();
    for i in 0..creatures {
        let seed = seed0 + i as u64;
        // Свежая игровая арена на каждую особь: карта, живой криттер (0),
        // статичный наблюдатель (1), ягоды.
        let mut arena = Arena::new(seed, game_map());
        arena.spawn(Vec2::new(5.0, 5.0), 0.0, seed);
        arena.spawn(Vec2::new(STATIC_POS.0, STATIC_POS.1), 0.0, seed ^ 0x5EED);
        for (x, y) in BERRY_SPOTS {
            arena.add_berry(Vec2::new(x, y));
        }

        // Прогрев: цикл отрастания и статичный наблюдатель входят в
        // устойчивое состояние; last_senses заполнен.
        for _ in 0..WARMUP_TICKS {
            arena.step();
            pin_static(&mut arena);
        }

        let mut samples = String::with_capacity(ticks as usize * 700);
        samples.push_str("[");
        let mut recorded = 0u32;
        for _ in 0..ticks {
            // Ретина — по геометрии ДО шага (та же картина, из которой мозг
            // примет действие); спящему «глаза закрыты» — ретина нулевая.
            let awake = will_sense_this_tick(&arena, 0);
            let retina = if awake {
                compute_retina(&arena, 0)
            } else {
                [0.0; IN_DIM]
            };

            arena.step();
            pin_static(&mut arena);

            // Цели: sense последнего шага (та же геометрия) + геометрия.
            let targets = match arena.last_sense(0) {
                Some(s) => compute_targets(&arena, 0, s),
                None => [0.0; OUT_DIM],
            };

            stats.add(&retina, &targets);
            if recorded > 0 {
                samples.push(',');
            }
            samples.push_str(&sample_json(&retina, &targets));
            recorded += 1;
        }
        samples.push_str("]");

        if i > 0 {
            json.push_str(",\n");
        }
        json.push_str("    {\n");
        json.push_str(&format!("      \"seed\": {seed},\n"));
        json.push_str(&format!("      \"samples\": {samples}\n"));
        json.push_str("    }");
        println!("  особь {i}: сид {seed}, записано {recorded} сэмплов");
    }
    json.push_str("\n  ],\n");
    json.push_str(&stats.header_json());
    json.push('\n');
    json.push_str("}\n");

    if let Some(parent) = out.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(&out, &json).expect("запись JSON");

    // ── Живость ретины (приёмка: «не все нули») ──
    let frac = |n: u64| -> f32 {
        if stats.samples == 0 {
            0.0
        } else {
            n as f32 / (stats.samples as f32 * RETINA_RAYS as f32)
        }
    };
    let tf = |n: u64| -> f32 {
        if stats.samples == 0 {
            0.0
        } else {
            n as f32 / stats.samples as f32
        }
    };
    println!(
        "готово: {} байт -> {out:?}  |  сэмплов {}, не-нулевых каналов ретины: \
         ягодность {:.3}, стенность {:.3}, соседность {:.3} | целей: vis {:.3}, пеленг {:.3}, стена {:.3}",
        json.len(),
        stats.samples,
        frac(stats.berry_nonzero),
        frac(stats.wall_nonzero),
        frac(stats.neighbor_nonzero),
        tf(stats.vis_nonzero),
        tf(stats.bearing_nonzero),
        tf(stats.wall_target_nonzero),
    );
}
