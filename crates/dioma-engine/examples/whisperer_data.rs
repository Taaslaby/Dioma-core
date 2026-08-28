//! # whisperer_data — рекордер датасета «Шептун» (этап 0 пайплайна)
//!
//! Записывает офлайн-датасет «существо ↔ мир» для обучения GRU-шептуна:
//! несколько особей, `--ticks` тиков каждая, по одному сэмплу на тик.
//! Сэмпл = 12 сенсорных входов (что существо ВИДЕЛО, когда решало действие
//! последнего тика) + реконструированное действие (steer/drive/eat) из
//! дельт позиции/курса/ягод.
//!
//! Пайплайн: `whisperer_data.rs` (этот рекордер) → `tools/whisperer_data.json`
//! → `tools/train_whisperer.py` (GRU, BPTT) → `tools/whisperer.dbw` →
//! `whisperer_check.rs` (паритет Python↔Rust + качество на hold-out).
//!
//! Запуск (из `crates/dioma-engine`):
//!
//! ```text
//! cargo run --release --example whisperer_data -- --creatures 8 --ticks 2000 --out ../../tools/whisperer_data.json --seed0 1000
//! ```
//!
//! Детерминизм: арены и существа — от фиксированных сидов (`seed0 + i`),
//! `Arena::step` параллелен (rayon) и бит-в-бит воспроизводим (см.
//! `sim/mod.rs`); повторный запуск с теми же аргументами даёт тот же файл.
//!
//! ## Как получен сэмпл (семантика «что привело к действию»)
//!
//! Реконструкция НЕ исполняет мозг дважды: сенсоры берутся из кэша
//! [`Arena::last_sense`] (ровно тот `Sense`, из которого существо приняло
//! действие последнего шага `step`), а действие восстанавливается из дельт:
//!
//! - `steer` = (Δкурс)/[`MAX_TURN`] — поворот за тик, делённый на макс. поворот;
//! - `drive` = |Δпозиция| / (speed × шаг) — пройденная доля максимума. Шаг
//!   [`MOVE_STEP_PER_TICK`] = 2.0 учтён ЯВНО: деление только на `speed()`
//!   насыщало бы таргет на 0.5 (формула из ТЗ упрощена; см. отчёт);
//! - `eat` = ягода «ушла в отрастание» этим тиком (vars[1]: 0 → >0).
//!
//! Оговорки (честно, в отчёт): удар о стену зажимает позицию — реконструкция
//! drive слегка НЕДОоценивает истинный газ (стена «съедает» путь); спящее
//! существо не действует (steer/drive/eat = 0) и сенсоров не имеет (входы
//! обнуляются) — это согласовано с двигателем.
//!
//! Отклонение от «голого» ТЗ: принудительные циклы голода. На старте и
//! каждые [`HUNGER_CYCLE`] тиков особи получают HUNGER=255, GLUCOSE=60
//! (как в охотничьих тестах движка). Иначе новорождённый голод (< 40 =
//! порог wants_food) не успевает догнать за 2000 тиков, а после первой еды
//! глюкоза упирается в 255 и голод гаснет навсегда — особи не едят, и
//! датасет теряет весь eat-канал и мотивационную динамику. Цикл возвращает
//! охоту: eat-события и драйв реально присутствуют в записи (см. `main`).

use dioma_engine::biology::metabolism::chem;
use dioma_engine::env::map::WorldMap;
use dioma_engine::env::{Substance, SUBSTANCES};
use dioma_engine::env::Vec2;
use dioma_engine::sim::{Arena, MAX_TURN};
use std::f32::consts::PI;
use std::f32::consts::TAU;
use std::fs;
use std::path::PathBuf;

/// Шаг движения за тик — дубликат приватного `MOVE_STEP_PER_TICK` из
/// `sim/mod.rs` (2.0; тик 50 мс — за секунду та же дистанция, что при 25 мс).
/// Используется для ОБРАТНОЙ реконструкции drive: dist = drive·speed·шаг.
const MOVE_STEP_PER_TICK: f32 = 2.0;

/// Период принудительного цикла голода (тиков): каждые 400 тиков особи
/// снова голодны (HUNGER=255, GLUCOSE=60). См. «Отклонение от ТЗ» в шапке.
const HUNGER_CYCLE: u32 = 400;

/// Размерность входа GRU-шептуна (фиксированный порядок, см. INPUT_NAMES).
const IN_DIM: usize = 12;
/// Размерность выхода (steer, drive, eat).
const OUT_DIM: usize = 3;

/// Имена входных размерностей — документируются в JSON-заголовке.
const INPUT_NAMES: [&str; IN_DIM] = [
    "smell_l",    // 0: запах еды слева (0..1), Sense.smell_l
    "smell_r",    // 1: запах еды справа (0..1), Sense.smell_r
    "wall_l",     // 2: близость стены слева (0..1), Sense.wall_l
    "wall_r",     // 3: близость стены справа (0..1), Sense.wall_r
    "vis_l",      // 4: зрение: еда в левом полуконусе (0..1), Sense.vis_l
    "vis_r",      // 5: зрение: еда в правом полуконусе (0..1), Sense.vis_r
    "food_near",  // 6: еда в радиусе рта (0/1), Sense.food_near
    "want_food",  // 7: хочу есть (рецептор рта, 0/1), Creature::wants_food
    "hunger",     // 8: голод/255 (0..1), Creature::hunger
    "glucose",    // 9: глюкоза/255 (0..1), Creature::glucose
    "fatigue",    // 10: усталость/255 (0..1), Creature::fatigue
    "heading",    // 11: курс/TAU·2 (0..2) — непрерывный курс, нормализован
];

/// Точки ягод — как в `src/bin/game.rs` (эталонная раздача).
const BERRY_SPOTS: [(f32, f32); 6] = [
    (5.0, 5.0),
    (18.0, 10.0),
    (30.0, 4.0),
    (48.0, 5.0),
    (62.0, 10.0),
    (72.0, 4.0),
];

/// Игровая карта: две комнаты 40×14 с дверью, плодородный климат —
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

/// Карта «id ягоды → ягода в стадии отрастания» (vars[1] > 0).
fn regrow_flags(arena: &Arena) -> Vec<(dioma_engine::env::AgentId, bool)> {
    arena
        .berries()
        .iter()
        .map(|&id| {
            let regrowing = arena
                .berry_info(id)
                .map(|b| b.regrow > 0.0)
                .unwrap_or(false);
            (id, regrowing)
        })
        .collect()
}

/// Входы сэмпла: сенсоры последнего тика + состояние тела ДО шага.
///
/// `sense` — то, что существо видело, когда решало действие (None = спало:
/// сенсоры обнуляются). Кровь/курс — состояние ДО шага (именно оно вместе
/// с сенсором привело к действию).
fn build_inputs(
    sense: Option<&dioma_engine::creature::Sense>,
    want_food: bool,
    hunger: f32,
    glucose: f32,
    fatigue: f32,
    heading_before: f32,
) -> [f32; IN_DIM] {
    let s = sense.copied().unwrap_or_default();
    [
        s.smell_l,
        s.smell_r,
        s.wall_l,
        s.wall_r,
        s.vis_l,
        s.vis_r,
        if s.food_near { 1.0 } else { 0.0 },
        if want_food { 1.0 } else { 0.0 },
        hunger / 255.0,
        glucose / 255.0,
        fatigue / 255.0,
        (heading_before / TAU) * 2.0, // курс непрерывный: 0..2 (см. INPUT_NAMES)
    ]
}

/// Число как строка JSON (без -0.0; фиксированная точность — детерминизм).
fn json_f32(x: f32) -> String {
    debug_assert!(x.is_finite(), "в записи оказался не-finite float: {x}");
    let x = if x == 0.0 { 0.0 } else { x }; // -0.0 → 0.0
    format!("{x:.7}")
}

/// Одна строка сэмпла: [12 входов, steer, drive, eat].
fn row_json(row: &[f32; IN_DIM + OUT_DIM]) -> String {
    let mut s = String::from("[");
    for (i, v) in row.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&json_f32(*v));
    }
    s.push(']');
    s
}

fn main() {
    // ── Аргументы: --creatures N --ticks N --out PATH --seed0 N ──
    let mut creatures = 8usize;
    let mut ticks = 2000u32;
    let mut out = PathBuf::from("tools/whisperer_data.json");
    let mut seed0 = 1000u64;
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
                    "whisperer_data — рекордер датасета «Шептун»\n\
                     \n\
                     Использование:\n  \
                     cargo run --release --example whisperer_data -- [--creatures N] [--ticks N] [--out PATH] [--seed0 N]\n\
                     \n\
                     Параметры:\n  \
                     --creatures N  число особей (по умолчанию 8)\n  \
                     --ticks N      записанных тиков на особь (по умолчанию 2000)\n  \
                     --out PATH     выходной JSON (по умолчанию tools/whisperer_data.json)\n  \
                     --seed0 N      базовый сид: арена/существо i получают seed0+i (по умолчанию 1000)"
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
        "запись «Шептуна»: {creatures} особей × {ticks} тиков, сиды {seed0}..{}, выход {out:?}",
        seed0 + creatures as u64 - 1
    );

    // ── Сборка JSON ──
    let mut json = String::with_capacity(creatures * ticks as usize * 180);
    json.push_str("{\n");
    json.push_str("  \"version\": \"whisperer_data_v1\",\n");
    json.push_str("  \"note\": \"рекордер examples/whisperer_data.rs; входы — сенсоры/состояние ДО шага, выходы — реконструированное действие (см. описание в шапке примера)\",\n");
    json.push_str("  \"input_names\": [");
    for (i, n) in INPUT_NAMES.iter().enumerate() {
        if i > 0 {
            json.push_str(", ");
        }
        json.push_str(&format!("\"{n}\""));
    }
    json.push_str("],\n");
    json.push_str("  \"output_names\": [\"steer\", \"drive\", \"eat\"],\n");
    json.push_str(&format!(
        "  \"map\": \"two rooms 40x14 connected (game.rs), 6 berries, spawn (5,5)/0.0\",\n"
    ));
    json.push_str(&format!("  \"samples_per_creature\": {ticks},\n"));
    json.push_str(&format!(
        "  \"train_creatures\": \"seed0+0..5\",\n  \"holdout_creatures\": \"seed0+6..7\",\n"
    ));
    json.push_str("  \"creatures\": [\n");

    for i in 0..creatures {
        let seed = seed0 + i as u64;
        // Свежая игровая арена на каждую особь: карта, существо, ягоды.
        let mut arena = Arena::new(seed, game_map());
        arena.spawn(Vec2::new(5.0, 5.0), 0.0, seed);
        // Голодный охотник со старта (как в охотничьих тестах движка):
        // новорождённое существо (глюкоза 150, голод 0) за 2000 тиков НЕ поел
        // бы ни разу — eat-канал датасета был бы пуст, а моторы почти без
        // драйва. Голод 255 + низкая глюкоза: ферментативное утоление
        // (1 HUNGER + 1 GLUCOSE → 1 GLUCOSE) замедляется, голод держится
        // над порогом wants_food сотни тиков — особи реально охотятся,
        // едят, спят: «мозг в работе» (см. шапку файла).
        arena.critters[0].creature.blood.conc[chem::HUNGER] = 255.0;
        arena.critters[0].creature.blood.conc[chem::GLUCOSE] = 60.0;
        for (x, y) in BERRY_SPOTS {
            arena.add_berry(Vec2::new(x, y));
        }

        // Прогрев: один step заполняет кэш last_senses — первый записанный
        // сэмпл уже имеет сенсор, из которого существо действовало.
        arena.step();

        let mut samples = String::with_capacity(ticks as usize * 170);
        samples.push_str("[");
        let mut recorded = 0u32;
        for t in 0..ticks {
            // Цикл голода: каждые HUNGER_CYCLE тиков особи снова голодны
            // (и глюкоза сбита). В диком мире насыщение наступает быстро
            // (ягод много) — без принудительного цикла особи едят только
            // на старте, и eat-канал датасета почти пуст. Цикл возвращает
            // мотивацию: особи РЕАЛЬНО охотятся/едят/наедаются/бродят
            // (см. шапку файла, «Отклонение от ТЗ»).
            if t > 0 && t % HUNGER_CYCLE == 0 {
                arena.critters[0].creature.blood.conc[chem::HUNGER] = 255.0;
                arena.critters[0].creature.blood.conc[chem::GLUCOSE] = 60.0;
            }
            let critter = &arena.critters[0];
            // Состояние ДО шага (вместе с сенсором привело к действию).
            let pos0 = critter.pos;
            let heading0 = critter.heading;
            let want_food = critter.creature.wants_food();
            let hunger = critter.creature.hunger();
            let glucose = critter.creature.glucose();
            let fatigue = critter.creature.fatigue();
            let regrow0 = regrow_flags(&arena);

            arena.step();

            // Сенсор, из которого существо приняло действие ЭТОГО шага.
            let sense = arena.last_sense(0);
            let pos1 = arena.critters[0].pos;
            let heading1 = arena.critters[0].heading;

            // Действие из дельт (см. шапку файла — формулы и оговорки).
            let mut d = (heading1 - heading0).rem_euclid(TAU);
            if d > PI {
                d -= TAU;
            }
            let steer = (d / MAX_TURN).clamp(-1.0, 1.0);
            let speed = arena.critters[0].creature.speed();
            let dist = pos1.dist(pos0);
            let drive = if speed > 1e-6 {
                (dist / (speed * MOVE_STEP_PER_TICK)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let eat = {
                let after = regrow_flags(&arena);
                let mut ate = false;
                for (id, regrowing) in &after {
                    let before = regrow0.iter().find(|(bid, _)| bid == id);
                    if *regrowing && before.map(|(_, r)| !r).unwrap_or(false) {
                        ate = true;
                        break;
                    }
                }
                ate
            };

            let inputs = build_inputs(
                sense,
                want_food,
                hunger,
                glucose,
                fatigue,
                heading0,
            );
            let mut row = [0.0f32; IN_DIM + OUT_DIM];
            row[..IN_DIM].copy_from_slice(&inputs);
            row[IN_DIM] = steer;
            row[IN_DIM + 1] = drive;
            row[IN_DIM + 2] = if eat { 1.0 } else { 0.0 };

            if recorded > 0 {
                samples.push(',');
            }
            samples.push_str(&row_json(&row));
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
    json.push_str("\n  ]\n}");
    json.push('\n');

    if let Some(parent) = out.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(&out, &json).expect("запись JSON");
    println!("готово: {} байт -> {out:?}", json.len());
}
