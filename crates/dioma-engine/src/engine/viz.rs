//! # Визуальные данные выращенного мозга
//!
//! Мост между коннектомом движка и внешними визуализаторами (презентационный
//! HTML-вьюер в `examples/brain_viz.rs`). Здесь нет рендера: модуль собирает
//! компактный срез — регионы, нейроны с позициями и активностью, синапсы,
//! зоны концепта — и умеет сериализовать его в JSON (формат
//! `dioma_brain_viz_v2`, ручной писатель, без внешних зависимостей).
//!
//! ## Формат v2 (от v1: + массив `zones`)
//!
//! Каждый датасет несёт список зон — аннотацию «какая доля какой моделью
//! живёт» (гибридный мозг, см. `docs/BRAIN_ROADMAP.md`):
//!
//! - `model`: `lif` | `wc` | `lstm` | `ann` | `hybrid`;
//! - `learning`: `stdp-online` | `genetic-params` | `baked-gradient` | `planned`;
//! - `color_hint`: индекс палитры моделей вьюера (0=lif … 4=hybrid) —
//!   раскраску берёт вьюер, в данных только подсказка;
//! - `members`: индексы регионов зоны.
//!
//! Старые файлы v1 остаются читаемыми человеком; вьюер v2 на них не
//! завязан (зоны у текущего мозга пусты: `"zones":[]`).
//!
//! Позиции нейронов — два честных пути:
//!
//! - **реальные** (рекомендуется): финальные сомы из ростового мира, см.
//!   [`crate::engine::grow::grow_with_positions`]. `Grown` хранит только
//!   топологию, поэтому позиции приходят отдельным срезом;
//! - **синтетические** (fallback, `positions = None`): детерминированная
//!   упаковка внутри региона (золотое сечение + сидированный джиттер) — не
//!   двигается между запусками, но это не точная геометрия роста.
//!
//! Активность измеряется на настоящем LIF-мозге (`Brain`, те же гейны и
//! сенсорные токи, что в `Creature::assemble`) под смешанным стимульным
//! паттерном: затишье → запах слева → запах справа → стена слева → стена
//! справа → оба запаха, по циклу.
use crate::engine::grow::{axon, Layout, RegionSpec};
use crate::engine::neurons::{gains_from, Brain, PlasticityParams};
use crate::rng::XorShift;
use crate::Grown;

/// Версия формата JSON визуализации (внешние потребители сверяют её).
/// v2 = v1 + массив `zones` (аннотация моделей долей, см. шапку модуля).
pub const VIZ_JSON_VERSION: &str = "dioma_brain_viz_v2";

/// Токен модели зоны: спайковый LIF (сенсоры/моторы/рефлексы/ассоциации).
pub const ZONE_MODEL_LIF: &str = "lif";
/// Токен модели зоны: Wilson–Cowan E/I-популяции (гладкая rate-динамика).
pub const ZONE_MODEL_WC: &str = "wc";
/// Токен модели зоны: GRU/LSTM (тренируемая память).
pub const ZONE_MODEL_LSTM: &str = "lstm";
/// Токен модели зоны: ANN-энкодер (зрение → вектор признаков → токи).
pub const ZONE_MODEL_ANN: &str = "ann";
/// Токен модели зоны: гибрид (лимбика: кровь → тоны + модуляция).
pub const ZONE_MODEL_HYBRID: &str = "hybrid";

/// Токен обучения зоны: STDP+дофамин онлайн (LIF-кора, уже работает).
pub const ZONE_LEARN_STDP: &str = "stdp-online";
/// Токен обучения зоны: параметры из генома (Wilson–Cowan, лимбика).
pub const ZONE_LEARN_GENETIC: &str = "genetic-params";
/// Токен обучения зоны: градиент офлайн, веса запекаются в геном.
pub const ZONE_LEARN_BAKED: &str = "baked-gradient";
/// Токен обучения зоны: запланировано, ещё не подключено.
pub const ZONE_LEARN_PLANNED: &str = "planned";

/// Число моделей в палитре зон (порядок = палитре вьюера).
pub const ZONE_MODEL_COUNT: u8 = 5;

/// Аннотация доли мозга: какая модель живёт в регионе и как учится.
/// Это ДИЗАЙН (см. `docs/BRAIN_ROADMAP.md`), а не исполняемая проводка —
/// зоны описывают концепт гибридного мозга.
pub struct ZoneViz {
    pub label: String,
    pub desc: String,
    /// `lif` | `wc` | `lstm` | `ann` | `hybrid` (см. `ZONE_MODEL_*`).
    pub model: String,
    /// `stdp-online` | `genetic-params` | `baked-gradient` | `planned`.
    pub learning: String,
    /// Индекс палитры моделей вьюера (0=lif … 4=hybrid); раскраску назначает
    /// вьюер, в данных — только подсказка.
    pub color_hint: u8,
    /// Индексы регионов зоны (порядок регионов схемы).
    pub members: Vec<usize>,
}

/// Метки регионов `avoidance_layout()` (порядок = порядку регионов схемы).
pub const AVOIDANCE_LABELS: &[&str] = &["SL", "SR", "WL", "WR", "I", "ML", "MR"];

/// Семантический класс каждого региона `avoidance_layout()` (для раскраски).
pub const AVOIDANCE_KINDS: &[&str] = &["sensor", "sensor", "sensor", "sensor", "inter", "motor", "motor"];

/// Число базовых регионов `avoidance_layout()`: регионы транша 3 добавляются
/// ПОСЛЕ них, поэтому плановыми (ghost) считаются индексы >= этого значения.
pub const AVOIDANCE_BASE_REGIONS: usize = 7;

/// Добавки транша 3 к `avoidance_layout()`: метки и классы новых регионов.
pub const FUTURE_EXTRA_LABELS: &[&str] = &["VL", "VR", "ASSOC_E", "ASSOC_I"];
pub const FUTURE_EXTRA_KINDS: &[&str] = &["sensor", "sensor", "assoc", "assoc"];

// ─── Игровые константы (копии из creature/mod.rs, чтобы движковый слой не
//     зависел от существа; значения сверены и синхронизируются вручную) ────

/// Базовый фоновый ток мозга (creature/mod.rs: `BASAL_EXT`).
const BASAL_EXT: f32 = 0.12;
/// Масштаб «запах [0..1] → ток сенсорного региона» (creature/mod.rs: `SMELL_CURRENT`).
const SMELL_CURRENT: f32 = 4.0;
/// Масштаб «стена [0..1] → ток региона стены» (creature/mod.rs: `WALL_CURRENT`).
const WALL_CURRENT: f32 = 4.0;
/// Гейн перекрёстных путей стены WL→MR / WR→ML (creature/mod.rs: `WALL_AVOID_GAIN`).
const WALL_AVOID_GAIN: f32 = 2.0;
/// Потолок пластичности существа (creature/mod.rs: `PLASTICITY_W_MAX`).
const PLASTICITY_W_MAX: f32 = 64.0;
/// Потолок связей в выводе: вьюер остаётся гладким при ~4k линий.
const LINK_CAP: usize = 4000;

/// Визуальные данные выращенного мозга для внешних визуализаторов.
pub struct BrainViz {
    pub regions: Vec<RegionViz>,
    pub neurons: Vec<NeuronViz>,
    /// Сегменты синапсов (pre → post).
    pub links: Vec<LinkViz>,
    /// Зоны концепта (модель + обучение доли, см. [`ZoneViz`]); у текущего
    /// мозга пусто, у планов — по одной на долю.
    pub zones: Vec<ZoneViz>,
    pub meta: VizMeta,
}

/// Регион мозга: имя, класс, геометрия и плановость.
pub struct RegionViz {
    pub label: String,
    /// `sensor` | `inter` | `motor` | `assoc` (семантический класс для tint).
    pub kind: String,
    pub center: [f32; 3],
    pub radius: f32,
    pub inhibitory: bool,
    /// Плановый регион (транш 3, ещё не выращен): вьюер рисует полупрозрачным
    /// пунктиром и не выдаёт за факт.
    pub ghost: bool,
}

/// Нейрон: id, регион, позиция, тип и средний спайковый ритм.
pub struct NeuronViz {
    pub id: usize,
    pub region: usize,
    pub pos: [f32; 3],
    pub inhibitory: bool,
    /// 0..1 — средний спайковый ритм за замер (0 = молчал).
    pub activity: f32,
}

/// Синаптический сегмент: пресинапс → постсинапс с весом.
#[derive(Clone)]
pub struct LinkViz {
    pub pre: usize,
    pub post: usize,
    pub weight: f32,
    /// Тормозная дуга: пресинапс — ингибитор.
    pub inhibited_edge: bool,
}

/// Метаданные среза: сид, замер активности, сводка.
pub struct VizMeta {
    pub seed: u64,
    pub ticks_measured: usize,
    /// Реальное число синапсов коннектома (до усечения в `links`).
    pub n_synapses: usize,
    /// Заголовок для вьюера («Мозг агента · сид N · …» или «Плановые регионы…»).
    pub title: String,
    /// Примечание (усечение связей, концепт-режим и т.п.), пусто — нет.
    pub note: String,
}

/// Плановая схема мозга транша 3 (стадия B): `avoidance_layout()` + зрительный
/// конус (VL/VR — левое/правое поле зрения) + ассоциативный слой (ASSOC_E —
/// возбуждающий, ASSOC_I — тормозной) с внутренней рекуррентностью.
///
/// ЕДИНЫЙ ИСТОЧНИК ПРАВДЫ — [`crate::engine::grow::stage_b_layout`]: эта
/// функция — тонкая алиас-обёртка для визуализаторов (та же схема, тот же
/// порядок регионов). Числа/метки сверяются тестом `stage_b_layout_matches_viz_tables`.
///
/// Схема не выращивается: визуализация отвечает на вопрос «какие регионы
/// появятся и как будут выглядеть» (регионы от `AVOIDANCE_BASE_REGIONS`
/// помечаются ghost).
pub fn avoidance_layout_future() -> Layout {
    crate::engine::grow::stage_b_layout()
}

// ─── Концепт зон (стадия C): чистый дизайн гибридного мозга ────────────────

/// Спецификация зон-концепта: (тег, центр, радиус, count, ингибитор, класс,
/// цели аксонов [(тег, max_len)]). Геометрия анатомична: зрение спереди (-Y),
/// кора сверху дугой (+Z), лимбика под корой, гиппокамп сзади-снизу,
/// премотор в хвосте (+Y), социальная зона сбоку (+X). Аксоны правдоподобны,
/// но ПРОИЗВОЛЬНЫ — схема не растёт (см. [`zones_layout`]).
const ZONES_SPEC: &[(&str, [f64; 3], f64, usize, bool, &str, &[(&str, f64)])] = &[
    // ── Ядро рефлексов [lif, stdp-online]: как avoidance_layout() ──
    ("SL", [-60.0, -60.0, 0.0], 15.0, 20, false, "sensor", &[("ML", 170.0)]),
    ("SR", [60.0, -60.0, 0.0], 15.0, 20, false, "sensor", &[("MR", 170.0)]),
    ("WL", [-30.0, -75.0, 0.0], 12.0, 12, false, "sensor", &[("MR", 240.0)]),
    ("WR", [30.0, -75.0, 0.0], 12.0, 12, false, "sensor", &[("ML", 240.0)]),
    ("I", [0.0, -30.0, 0.0], 12.0, 16, true, "inter", &[("ML", 140.0), ("MR", 140.0)]),
    ("ML", [-60.0, 60.0, 0.0], 15.0, 20, false, "motor", &[]),
    ("MR", [60.0, 60.0, 0.0], 15.0, 20, false, "motor", &[]),
    // ── Зрение (энкодер) [ann, baked-gradient]: спереди, сырые бины ──
    ("VL", [-65.0, -155.0, 10.0], 20.0, 60, false, "sensor", &[
        ("CORTEX_E", 190.0),
        ("AMYG", 220.0),
        ("PLAN_E", 200.0),
        ("ML", 240.0),
        ("MR", 240.0),
    ]),
    ("VR", [65.0, -155.0, 10.0], 20.0, 60, false, "sensor", &[
        ("CORTEX_E", 190.0),
        ("AMYG", 220.0),
        ("PLAN_E", 200.0),
        ("ML", 240.0),
        ("MR", 240.0),
    ]),
    // ── Ассоциативный хаб [lif+инъекции, stdp-online]: кора сверху ──
    ("CORTEX_E", [0.0, -10.0, 90.0], 52.0, 400, false, "assoc", &[
        ("CORTEX_E", 60.0),
        ("CORTEX_I", 90.0),
        ("PLAN_E", 160.0),
        ("SEQ_E", 190.0),
        ("AMYG", 150.0),
        ("MOOD", 160.0),
        ("MEM_CTX", 230.0),
        ("ML", 200.0),
        ("MR", 200.0),
    ]),
    ("CORTEX_I", [0.0, 62.0, 92.0], 28.0, 120, true, "inter", &[
        ("CORTEX_E", 90.0),
        ("PLAN_E", 180.0),
        ("SEQ_E", 170.0),
        ("ML", 190.0),
        ("MR", 190.0),
    ]),
    // ── Планирование [wc, genetic-params]: фронтально, под корой ──
    ("PLAN_E", [0.0, -108.0, 58.0], 24.0, 80, false, "assoc", &[
        ("SEQ_E", 270.0),
        ("ML", 240.0),
        ("MR", 240.0),
        ("MOOD", 150.0),
        ("CORTEX_E", 150.0),
    ]),
    ("PLAN_I", [0.0, -130.0, 36.0], 16.0, 40, true, "inter", &[
        ("PLAN_E", 60.0),
        ("SEQ_E", 190.0),
        ("CORTEX_E", 150.0),
    ]),
    // ── Гиппокамп (память) [lstm, baked-gradient]: сзади-снизу ──
    ("MEM_CTX", [95.0, 128.0, -42.0], 30.0, 100, false, "assoc", &[
        ("CORTEX_E", 230.0),
        ("PLAN_E", 280.0),
        ("SEQ_E", 150.0),
        ("MEM_OUT", 90.0),
    ]),
    ("MEM_OUT", [140.0, 90.0, -58.0], 17.0, 30, false, "assoc", &[
        ("CORTEX_E", 240.0),
        ("PLAN_E", 280.0),
    ]),
    // ── Лимбика [hybrid, genetic-params]: под корой ──
    ("DRIVES", [-78.0, 15.0, 38.0], 21.0, 40, false, "assoc", &[
        ("MOOD", 90.0),
        ("CORTEX_E", 130.0),
        ("AMYG", 140.0),
    ]),
    ("MOOD", [-92.0, -38.0, 52.0], 18.0, 30, false, "assoc", &[
        ("CORTEX_E", 150.0),
        ("PLAN_E", 150.0),
        ("AMYG", 130.0),
        ("SEQ_E", 230.0),
    ]),
    // ── Оценка [lif, stdp-online]: вторичное подкрепление ──
    ("AMYG", [55.0, 22.0, 32.0], 19.0, 50, false, "assoc", &[
        ("CORTEX_E", 130.0),
        ("PLAN_E", 170.0),
        ("ML", 200.0),
        ("MR", 200.0),
        ("MOOD", 170.0),
    ]),
    // ── Социальная [lif, stdp-online]: сбоку ──
    ("SOCIAL", [135.0, 0.0, 30.0], 19.0, 40, false, "assoc", &[
        ("CORTEX_E", 190.0),
        ("AMYG", 130.0),
        ("PLAN_E", 190.0),
    ]),
    // ── Премотор [wc, genetic-params]: в хвосте ──
    ("SEQ_E", [0.0, 152.0, 28.0], 22.0, 50, false, "assoc", &[
        ("ML", 190.0),
        ("MR", 190.0),
        ("SEQ_I", 80.0),
    ]),
    ("SEQ_I", [0.0, 186.0, 8.0], 16.0, 25, true, "inter", &[
        ("SEQ_E", 80.0),
        ("ML", 190.0),
        ("MR", 190.0),
    ]),
];

/// Схема «концепт зон» (стадия C): 21 регион, ~1245 нейронов (сумма
/// фиксированных размеров зон; «~1400» в ТЗ — ориентир, ядро взято из
/// `avoidance_layout()`). НЕ выращивается и НЕ исполняется — это чистый
/// дизайн: аксоны и пары правдоподобны-произвольны, регионы-призраки.
/// Возвращает схему и классы регионов (один источник — таблица `ZONES_SPEC`).
pub fn zones_layout() -> (Layout, Vec<&'static str>) {
    let mut regions: Vec<RegionSpec> = Vec::with_capacity(ZONES_SPEC.len());
    let mut kinds: Vec<&'static str> = Vec::with_capacity(ZONES_SPEC.len());
    for &(tag, center, radius, count, inhibitory, kind, targets) in ZONES_SPEC {
        let axons = targets.iter().map(|&(t, l)| axon(t, l)).collect();
        regions.push(RegionSpec {
            tag,
            center,
            radius,
            count,
            inhibitory,
            secrete: None,
            axons,
        });
        kinds.push(kind);
    }
    // Второй проход: тег-цель секретирует морфоген (иначе рост был бы
    // непоследователен — цель обязана существовать). Сенсоры-источники
    // (VL/VR, SL/SR/WL/WR) цели ни для кого не служат — без секреции.
    let n = regions.len();
    let mut is_target = vec![false; n];
    for r in &regions {
        for a in &r.axons {
            if let Some(i) = regions.iter().position(|x| x.tag == a.target) {
                is_target[i] = true;
            }
        }
    }
    for (r, &t) in regions.iter_mut().zip(is_target.iter()) {
        if t {
            r.secrete = Some(r.tag.to_string());
        }
    }
    // Пары: локальная рекуррентность (тег, тег) + направленные дуги по аксонам.
    let mut allowed_pairs: Vec<(String, String)> = Vec::new();
    for r in &regions {
        allowed_pairs.push((r.tag.to_string(), r.tag.to_string()));
    }
    for r in &regions {
        for a in &r.axons {
            allowed_pairs.push((r.tag.to_string(), a.target.clone()));
        }
    }
    (Layout { regions, allowed_pairs }, kinds)
}

/// Собрать визуальные данные выращенного мозга (текущее ядро, зон нет).
///
/// - `positions`: финальные позиции сом из ростового мира (см.
///   [`crate::engine::grow::grow_with_positions`]); `None` — детерминированная
///   синтетическая упаковка нейронов внутри регионов.
/// - `labels_hint`: метки регионов; при несовпадении длины — таблицы
///   `AVOIDANCE_LABELS` (+ добавки транша 3), иначе `R{i}`.
/// - `n_ticks_activity`: мировых тиков замера спайковой активности.
pub fn brain_viz(
    layout: &Layout,
    grown: &Grown,
    positions: Option<&[[f64; 3]]>,
    labels_hint: &[&str],
    n_ticks_activity: usize,
    seed: u64,
) -> BrainViz {
    brain_viz_current(layout, grown, positions, labels_hint, n_ticks_activity, seed)
}

/// Текущее ядро (стадия A): то же, что [`brain_viz`], зоны пусты
/// (`"zones":[]` в JSON). Сигнатура фиксирована для внешних визуализаторов.
pub fn brain_viz_current(
    layout: &Layout,
    grown: &Grown,
    positions: Option<&[[f64; 3]]>,
    labels_hint: &[&str],
    n_ticks_activity: usize,
    seed: u64,
) -> BrainViz {
    let title = format!(
        "Мозг агента · сид {seed} · {} нейронов · {} синапсов",
        layout.n(),
        grown.edges.len()
    );
    brain_viz_impl(
        layout,
        Some(grown),
        positions,
        labels_hint,
        &[],
        n_ticks_activity,
        seed,
        usize::MAX,
        Vec::new(),
        title,
        String::new(),
    )
}

/// Концепт-визуализация ПЛАНОВОЙ схемы (старая сигнатура, без зон):
/// коннектом не выращен, поэтому связей и активности нет; регионы от
/// `AVOIDANCE_BASE_REGIONS` помечаются ghost.
pub fn brain_viz_concept(
    layout: &Layout,
    positions: Option<&[[f64; 3]]>,
    labels_hint: &[&str],
    seed: u64,
) -> BrainViz {
    brain_viz_impl(
        layout,
        None,
        positions,
        labels_hint,
        &[],
        0,
        seed,
        AVOIDANCE_BASE_REGIONS,
        Vec::new(),
        "Плановые регионы (транш 3, концепт)".to_string(),
        TRANSH3_CONCEPT_NOTE.to_string(),
    )
}

/// План транша 3 (стадия B) — конкретный план «сегодня»: базовые 7 регионов
/// + зрение VL/VR (по 30, сенсорные «глаз-бины», готовые к ANN-энкодеру) +
/// ассоциативный слой ASSOC_E/ASSOC_I (рекуррентный хаб с тормозным
/// контуром), всего 300 нейронов. Схема [`avoidance_layout_future`],
/// регионы-призраки от `AVOIDANCE_BASE_REGIONS`; зоны описывают доли.
pub fn brain_viz_stage_b(
    layout: &Layout,
    positions: Option<&[[f64; 3]]>,
    labels_hint: &[&str],
    seed: u64,
) -> BrainViz {
    brain_viz_impl(
        layout,
        None,
        positions,
        labels_hint,
        &[],
        0,
        seed,
        AVOIDANCE_BASE_REGIONS,
        stage_b_zones(),
        "План транша 3: зрение + ассоциативный слой".to_string(),
        TRANSH3_CONCEPT_NOTE.to_string(),
    )
}

/// Примечание концепта транша 3 (общее для `brain_viz_concept` и stage B).
const TRANSH3_CONCEPT_NOTE: &str =
    "Концепт: плановые регионы (зрение VL/VR и ассоциативный слой) не выращены — \
     связей и активности нет; показаны регионы и их геометрия.";

/// Концепт зон (стадия C) — чистый дизайн гибридного мозга, БЕЗ проводки в
/// симуляции: регионы-призраки (все), синтетическая упаковка (золотое
/// сечение + джиттер), аксоны и пары правдоподобны-произвольны. Схема
/// [`zones_layout`], зоны [`stage_c_zones`].
pub fn brain_viz_stage_c() -> BrainViz {
    let (layout, kinds) = zones_layout();
    let labels: Vec<&str> = layout.regions.iter().map(|r| r.tag).collect();
    let n = layout.n();
    brain_viz_impl(
        &layout,
        None,
        None,
        &labels,
        &kinds,
        0,
        STAGE_C_SEED,
        0, // все регионы — призраки (чистый дизайн)
        stage_c_zones(),
        format!("Концепт зон · гибридный мозг · {n} нейронов"),
        "Чистый дизайн (стадия C): зоны не выращены — связей и активности нет; \
         геометрия и аксоны правдоподобны-произвольны."
            .to_string(),
    )
}

/// Сид метаданных концепта зон (фиксированный: концепт не зависит от существа).
const STAGE_C_SEED: u64 = 7777;

/// Зоны плана транша 3 (стадия B): регионы `avoidance_layout_future()`
/// по индексам — 0..6 ядро, 7..8 зрение, 9..10 ассоциативный слой.
fn stage_b_zones() -> Vec<ZoneViz> {
    vec![
        zone(
            "Ядро рефлексов",
            "LIF-кора: сенсоры (запах, стены), тормозной интернейрон и моторы. \
             STDP + дофамин онлайн — уже работает.",
            ZONE_MODEL_LIF,
            ZONE_LEARN_STDP,
            (0..7).collect(),
        ),
        zone(
            "Зрение (сырые бины)",
            "Глаз-бины: пока сенсорные LIF-регионы; ANN-энкодер \
             (зрение → вектор признаков → токи) — следующий шаг.",
            ZONE_MODEL_LIF,
            ZONE_LEARN_PLANNED,
            vec![7, 8],
        ),
        zone(
            "Ассоциативный хаб",
            "Рекуррентный слой (E-возбуждающий, I-тормозной контур); принимает \
             ИНЪЕКЦИИ — внешние токи от других модулей (прецедент \
             reward_now_gated / schedule_reward).",
            ZONE_MODEL_LIF,
            ZONE_LEARN_STDP,
            vec![9, 10],
        ),
    ]
}

/// Зоны концепта (стадия C): регионы `zones_layout()` по индексам.
fn stage_c_zones() -> Vec<ZoneViz> {
    vec![
        zone(
            "Ядро",
            "LIF-кора рефлексов: сенсоры (запах, стены), интернейрон, моторы. \
             Обучение STDP + дофамин онлайн — уже работает.",
            ZONE_MODEL_LIF,
            ZONE_LEARN_STDP,
            vec![0, 1, 2, 3, 4, 5, 6],
        ),
        zone(
            "Зрение (энкодер)",
            "ANN-энкодер: картинка → вектор признаков → токи в зоны. Обучение \
             градиентом офлайн, веса запекаются в геном. До внедрения — сырые бины.",
            ZONE_MODEL_ANN,
            ZONE_LEARN_BAKED,
            vec![7, 8],
        ),
        zone(
            "Ассоциативный хаб",
            "LIF-кора + инъекции: рекуррентный возбуждающий слой с тормозным \
             контуром; принимает внешние токи (награда, состояние модулей) — \
             прецедент reward_now_gated / schedule_reward.",
            ZONE_MODEL_LIF,
            ZONE_LEARN_STDP,
            vec![9, 10],
        ),
        zone(
            "Планирование",
            "Wilson–Cowan E/I-популяции: гладкая rate-динамика выбора действия. \
             Параметры из генома; ядро step_wc есть в pillars, к существам не \
             подключено.",
            ZONE_MODEL_WC,
            ZONE_LEARN_GENETIC,
            vec![11, 12],
        ),
        zone(
            "Гиппокамп (память)",
            "GRU/LSTM: тренируемая память. Обучение градиентом ОФЛАЙН, веса \
             запекаются в геном (детерминизм, прецедент save_baked).",
            ZONE_MODEL_LSTM,
            ZONE_LEARN_BAKED,
            vec![13, 14],
        ),
        zone(
            "Лимбика",
            "Гибрид: кровь → тоны + модуляция пластичности per-region \
             (регуляторный геном).",
            ZONE_MODEL_HYBRID,
            ZONE_LEARN_GENETIC,
            vec![15, 16],
        ),
        zone(
            "Оценка",
            "LIF-зона вторичного подкрепления: оценивает события, направляет \
             дофамин.",
            ZONE_MODEL_LIF,
            ZONE_LEARN_STDP,
            vec![17],
        ),
        zone(
            "Социальная",
            "LIF-зона: восприятие других существ и социальные сигналы.",
            ZONE_MODEL_LIF,
            ZONE_LEARN_STDP,
            vec![18],
        ),
        zone(
            "Премотор",
            "Wilson–Cowan: планирование последовательностей движений в хвосте.",
            ZONE_MODEL_WC,
            ZONE_LEARN_GENETIC,
            vec![19, 20],
        ),
    ]
}

/// Конструктор зоны с индексами палитры моделей (см. вьюер: MODEL_COLOR).
fn zone(label: &str, desc: &str, model: &str, learning: &str, members: Vec<usize>) -> ZoneViz {
    ZoneViz {
        label: label.to_string(),
        desc: desc.to_string(),
        model: model.to_string(),
        learning: learning.to_string(),
        color_hint: model_color_hint(model),
        members,
    }
}

/// Индекс палитры моделей вьюера: 0=lif, 1=wc, 2=lstm, 3=ann, 4=hybrid.
fn model_color_hint(model: &str) -> u8 {
    match model {
        ZONE_MODEL_WC => 1,
        ZONE_MODEL_LSTM => 2,
        ZONE_MODEL_ANN => 3,
        ZONE_MODEL_HYBRID => 4,
        _ => 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn brain_viz_impl(
    layout: &Layout,
    grown: Option<&Grown>,
    positions: Option<&[[f64; 3]]>,
    labels_hint: &[&str],
    kinds_hint: &[&str],
    n_ticks_activity: usize,
    seed: u64,
    ghost_from: usize,
    zones: Vec<ZoneViz>,
    title: String,
    concept_note: String,
) -> BrainViz {
    let labels = region_labels(layout, labels_hint);
    let kinds = region_kinds(layout, kinds_hint);
    let pos = neuron_positions(layout, positions, derive_pack_seed(layout));
    // Регион каждого нейрона: блоки layout (порядок роста). Если grown
    // не совпадает по длине с layout — используем порядок блоков (безопасно).
    let region_of: Vec<usize> = match grown {
        Some(g) if g.region_of.len() == layout.n() => g.region_of.clone(),
        _ => layout.region_of(),
    };

    let regions: Vec<RegionViz> = layout
        .regions
        .iter()
        .enumerate()
        .map(|(i, r)| RegionViz {
            label: labels.get(i).cloned().unwrap_or_else(|| format!("R{i}")),
            kind: kinds.get(i).cloned().unwrap_or_else(|| "motor".to_string()),
            center: [r.center[0] as f32, r.center[1] as f32, r.center[2] as f32],
            radius: r.radius as f32,
            inhibitory: r.inhibitory,
            ghost: i >= ghost_from,
        })
        .collect();

    // Активность: настоящий LIF-мозг под смешанным стимульным паттерном.
    let activity = match grown {
        Some(g) if g.region_of.len() == layout.n() && n_ticks_activity > 0 => {
            measure_activity(layout, g, n_ticks_activity)
        }
        _ => vec![0.0; pos.len()],
    };

    let neurons: Vec<NeuronViz> = pos
        .iter()
        .enumerate()
        .map(|(id, p)| {
            let region = region_of.get(id).copied().unwrap_or_else(|| default_region(layout, id));
            let inhibitory = layout
                .regions
                .get(region)
                .map_or(false, |r| r.inhibitory);
            NeuronViz {
                id,
                region,
                pos: *p,
                inhibitory,
                activity: activity.get(id).copied().unwrap_or(0.0),
            }
        })
        .collect();

    // Связи: из grown с усечением до LINK_CAP (баланс направлений pre→post).
    let (links, cap_note) = match grown {
        Some(g) => build_links(layout, g),
        None => (Vec::new(), String::new()),
    };

    let n_synapses = grown.map_or(0, |g| g.edges.len());
    let concept = ghost_from < layout.regions.len();
    let mut note = String::new();
    if concept {
        note.push_str(&concept_note);
    }
    if !cap_note.is_empty() {
        if !note.is_empty() {
            note.push_str(" ");
        }
        note.push_str(&cap_note);
    }

    BrainViz {
        regions,
        neurons,
        links,
        zones,
        meta: VizMeta {
            seed,
            ticks_measured: if concept { 0 } else { n_ticks_activity },
            n_synapses,
            title,
            note,
        },
    }
}

/// Метки регионов схемы: подсказка → таблицы (base + транш-3) → `R{i}`.
fn region_labels(layout: &Layout, labels_hint: &[&str]) -> Vec<String> {
    let n = layout.regions.len();
    if labels_hint.len() == n {
        return labels_hint.iter().map(|s| s.to_string()).collect();
    }
    if n == AVOIDANCE_LABELS.len() {
        return AVOIDANCE_LABELS.iter().map(|s| s.to_string()).collect();
    }
    if n == AVOIDANCE_LABELS.len() + FUTURE_EXTRA_LABELS.len() {
        return AVOIDANCE_LABELS
            .iter()
            .chain(FUTURE_EXTRA_LABELS)
            .map(|s| s.to_string())
            .collect();
    }
    (0..n).map(|i| format!("R{i}")).collect()
}

/// Классы регионов: подсказка (концепт зон) → таблицы (base + транш-3),
/// для неизвестной схемы — ингибиторы → `inter`, остальные → `motor`
/// (защитный fallback).
fn region_kinds(layout: &Layout, kinds_hint: &[&str]) -> Vec<String> {
    let n = layout.regions.len();
    if kinds_hint.len() == n {
        return kinds_hint.iter().map(|s| s.to_string()).collect();
    }
    let table: Vec<&str> = if n == AVOIDANCE_KINDS.len() {
        AVOIDANCE_KINDS.to_vec()
    } else if n == AVOIDANCE_KINDS.len() + FUTURE_EXTRA_KINDS.len() {
        AVOIDANCE_KINDS
            .iter()
            .chain(FUTURE_EXTRA_KINDS)
            .copied()
            .collect()
    } else {
        layout
            .regions
            .iter()
            .map(|r| if r.inhibitory { "inter" } else { "motor" })
            .collect()
    };
    table.into_iter().map(|s| s.to_string()).collect()
}

/// Детерминированное зерно упаковки для схемы: не меняется между запусками.
fn derive_pack_seed(layout: &Layout) -> u64 {
    let mut h = PACK_SEED_BASE;
    for r in &layout.regions {
        h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(r.count as u64);
        for c in r.center {
            h ^= (c.to_bits() as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        }
    }
    h
}

/// База зерна синтетической упаковки (произвольная, но фиксированная).
const PACK_SEED_BASE: u64 = 0x5EED_0001_0000_0042;

/// Позиции нейронов: реальные сомы, если переданы (для плановых регионов
/// срез обычно короче — хвост добирается синтетикой), иначе — полностью
/// синтетическая упаковка.
///
/// Поток розыгрышей не зависит от числа реальных позиций: каждый нейрон
/// схемы съедает ровно 4 розыгрыша (радиус + 3 джиттера), пропуск или
/// упаковка — не важно. Поэтому синтетический хвост идентичен для любого
/// `real` (детерминированность между запусками и режимами).
fn neuron_positions(layout: &Layout, real: Option<&[[f64; 3]]>, pack_seed: u64) -> Vec<[f32; 3]> {
    let n = layout.n();
    let mut out = Vec::with_capacity(n);
    if let Some(r) = real {
        let m = r.len().min(n);
        out.extend(r[..m].iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]));
    }
    if out.len() == n {
        return out;
    }
    let mut rng = XorShift::new(pack_seed);
    let mut cursor = 0usize;
    for r in &layout.regions {
        let covered = out.len().saturating_sub(cursor).min(r.count);
        for _ in 0..covered {
            rng.unit();
            rng.unit();
            rng.unit();
            rng.unit();
        }
        for idx in covered..r.count {
            out.push(pack_region(r, idx, r.count, &mut rng));
        }
        cursor += r.count;
    }
    out
}

/// Упаковка одного нейрона в объёме региона: золотое сечение по сфере +
/// равномерный по объёму радиус + сидированный джиттер «под микроскопию».
fn pack_region(r: &RegionSpec, idx: usize, count: usize, rng: &mut XorShift) -> [f32; 3] {
    const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;
    let t = (idx as f64 + 0.5) / count.max(1) as f64;
    let z = 1.0 - 2.0 * t;
    let phi = GOLDEN_ANGLE * idx as f64;
    let xy = (1.0 - z * z).max(0.0).sqrt();
    let rad = r.radius * rng.unit().cbrt();
    let mut p = [
        r.center[0] + rad * xy * phi.cos(),
        r.center[1] + rad * xy * phi.sin(),
        r.center[2] + rad * z,
    ];
    for c in p.iter_mut() {
        *c += (rng.unit() - 0.5) * r.radius * 0.12;
    }
    [p[0] as f32, p[1] as f32, p[2] as f32]
}

/// Регион нейрона по порядку блоков (как grow()): для схем без grown.
fn default_region(layout: &Layout, id: usize) -> usize {
    let mut acc = 0usize;
    for (i, r) in layout.regions.iter().enumerate() {
        acc += r.count;
        if id < acc {
            return i;
        }
    }
    layout.regions.len().saturating_sub(1)
}

/// Индекс региона по тегу (fallback — порядковый номер, защищённый от пустой
/// схемы). Для `avoidance_layout()` теги всегда есть.
fn region_index_by_tag(layout: &Layout, tag: &str, fallback: usize) -> usize {
    layout
        .regions
        .iter()
        .position(|r| r.tag == tag)
        .unwrap_or_else(|| fallback.min(layout.regions.len().saturating_sub(1)))
}

/// Замер активности: LIF-мозг (те же гейны и токи, что в `Creature::assemble`),
/// смешанный стимульный паттерн по циклу — детерминированно, без ГСЧ.
fn measure_activity(layout: &Layout, grown: &Grown, n_ticks: usize) -> Vec<f32> {
    let n = grown.nt.len();
    let mut acc = vec![0u64; n];
    // Гейн-матрица как в creature/mod.rs::assemble_impl (пары avoidance_layout):
    // SL→ML, SR→MR (гоминг), SL/SR→I (торможение), WL→MR / WR→ML (отврат от
    // стены, усиленный), I→ML / I→MR. Если регионов меньше 7 — гейнов нет,
    // мозг остаётся без входов (безопасный fallback для чужих схем).
    let gains = if layout.regions.len() >= AVOIDANCE_BASE_REGIONS {
        gains_from(
            layout.regions.len(),
            &[
                (0, 5, 1.0),
                (1, 6, 1.0),
                (0, 4, 0.8),
                (1, 4, 0.8),
                (2, 6, WALL_AVOID_GAIN),
                (3, 5, WALL_AVOID_GAIN),
                (4, 5, 1.0),
                (4, 6, 1.0),
            ],
        )
    } else {
        gains_from(layout.regions.len(), &[])
    };
    let mut brain = Brain::new(
        grown,
        gains,
        PlasticityParams {
            w_max: PLASTICITY_W_MAX,
            // Замер не обучает: веса константны (мозг-«камень»).
            lr: 0.0,
            ..Default::default()
        },
    );
    let sl = region_index_by_tag(layout, "SL", 0);
    let sr = region_index_by_tag(layout, "SR", 1);
    let wl = region_index_by_tag(layout, "WL", 2);
    let wr = region_index_by_tag(layout, "WR", 3);
    let mut ext = vec![BASAL_EXT; n];
    let mut hist: Vec<Vec<i32>> = Vec::new();
    let spt = brain.steps_per_tick();
    for t in 0..n_ticks {
        // 6 фаз по 5 тиков (период 30): затишье → запах L → запах R →
        // стена L → стена R → оба запаха.
        let (sm_l, sm_r, w_l, w_r) = match (t / 5) % 6 {
            0 => (0.0, 0.0, 0.0, 0.0),
            1 => (1.0, 0.0, 0.0, 0.0),
            2 => (0.0, 1.0, 0.0, 0.0),
            3 => (0.0, 0.0, 1.0, 0.0),
            4 => (0.0, 0.0, 0.0, 1.0),
            _ => (0.7, 0.7, 0.0, 0.0),
        };
        ext.fill(BASAL_EXT);
        add_current(&mut ext, &brain, sl, sm_l * SMELL_CURRENT);
        add_current(&mut ext, &brain, sr, sm_r * SMELL_CURRENT);
        add_current(&mut ext, &brain, wl, w_l * WALL_CURRENT);
        add_current(&mut ext, &brain, wr, w_r * WALL_CURRENT);
        brain.step_batch(&ext, spt, &mut hist);
        for v in &hist {
            for (a, &sp) in acc.iter_mut().zip(v.iter()) {
                *a += sp as u64;
            }
        }
    }
    let total = (n_ticks as u64).saturating_mul(spt as u64).max(1);
    acc.into_iter().map(|a| (a as f64 / total as f64) as f32).collect()
}

/// Добавить сенсорный ток в регион (пустые границы = регион не существует).
fn add_current(ext: &mut [f32], brain: &Brain, region: usize, current: f32) {
    let (a, b) = brain.region_bounds(region);
    for e in &mut ext[a..b] {
        *e += current;
    }
}

/// Связи из grown; при превышении `LINK_CAP` — усечение с балансом
/// направлений: каждая пара регионов pre→post сохраняет долю, пропорциональную
/// её вкладу в общее число рёбер (сильнейшие по |вес| внутри пары).
fn build_links(layout: &Layout, grown: &Grown) -> (Vec<LinkViz>, String) {
    let total = grown.edges.len();
    let mut links: Vec<LinkViz> = grown
        .edges
        .iter()
        .map(|&(pre, post, w)| LinkViz {
            pre,
            post,
            weight: w,
            inhibited_edge: grown.nt.get(pre).copied().unwrap_or(1.0) < 0.0,
        })
        .collect();
    if links.len() <= LINK_CAP {
        return (links, String::new());
    }
    let kept = cap_links(&mut links, layout, grown);
    (
        links,
        format!("связи усечены: {total} → {kept} (потолок {LINK_CAP}, по направлениям pre→post)"),
    )
}

/// Усечение связей до `LINK_CAP` с балансом направлений (см. `build_links`).
fn cap_links(links: &mut Vec<LinkViz>, layout: &Layout, grown: &Grown) -> usize {
    let n_reg = layout.regions.len();
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n_reg * n_reg];
    for (i, l) in links.iter().enumerate() {
        let rp = grown.region_of.get(l.pre).copied().unwrap_or(0).min(n_reg.saturating_sub(1));
        let rq = grown.region_of.get(l.post).copied().unwrap_or(0).min(n_reg.saturating_sub(1));
        groups[rp * n_reg + rq].push(i);
    }
    let total = links.len().max(1) as f64;
    let mut keep: Vec<usize> = Vec::with_capacity(LINK_CAP);
    for mut idxs in groups {
        if idxs.is_empty() {
            continue;
        }
        idxs.sort_by(|&a, &b| links[b].weight.abs().total_cmp(&links[a].weight.abs()));
        let quota = ((idxs.len() as f64 / total) * LINK_CAP as f64).ceil() as usize;
        keep.extend(idxs.into_iter().take(quota));
    }
    if keep.len() > LINK_CAP {
        keep.sort_by(|&a, &b| links[b].weight.abs().total_cmp(&links[a].weight.abs()));
        keep.truncate(LINK_CAP);
    }
    let mut out = Vec::with_capacity(keep.len());
    for &i in &keep {
        out.push(links[i].clone());
    }
    *links = out;
    links.len()
}

impl BrainViz {
    /// Компактный JSON (формат `dioma_brain_viz_v2`). Ручной писатель:
    /// детерминированный, с короткими числами и экранированием строк
    /// (включая `\u003c/\u003e` — безопасно встраивать в `<script>`).
    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(64 * 1024);
        s.push_str("{\"version\":\"");
        s.push_str(VIZ_JSON_VERSION);
        s.push_str("\",\"meta\":{");
        s.push_str(&format!(
            "\"seed\":{},\"ticks_measured\":{},\"n_synapses\":{},\"title\":{}",
            self.meta.seed,
            self.meta.ticks_measured,
            self.meta.n_synapses,
            json_str(&self.meta.title)
        ));
        s.push_str(",\"note\":");
        s.push_str(&json_str(&self.meta.note));
        s.push_str("},\"regions\":[");
        for (i, r) in self.regions.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"label\":{},\"kind\":{},\"ghost\":{},\"inhibitory\":{},\"center\":[{},{},{}],\"radius\":{}}}",
                json_str(&r.label),
                json_str(&r.kind),
                bool01(r.ghost),
                bool01(r.inhibitory),
                num(r.center[0]),
                num(r.center[1]),
                num(r.center[2]),
                num(r.radius)
            ));
        }
        s.push_str("],\"neurons\":[");
        for (i, n) in self.neurons.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"id\":{},\"region\":{},\"inh\":{},\"act\":{},\"pos\":[{},{},{}]}}",
                n.id,
                n.region,
                bool01(n.inhibitory),
                num(n.activity),
                num(n.pos[0]),
                num(n.pos[1]),
                num(n.pos[2])
            ));
        }
        s.push_str("],\"links\":[");
        for (i, l) in self.links.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"pre\":{},\"post\":{},\"w\":{},\"inh\":{}}}",
                l.pre,
                l.post,
                num(l.weight),
                bool01(l.inhibited_edge)
            ));
        }
        s.push_str("],\"zones\":[");
        for (i, z) in self.zones.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"label\":{},\"desc\":{},\"model\":{},\"learning\":{},\"color_hint\":{},\"members\":[{}]}}",
                json_str(&z.label),
                json_str(&z.desc),
                json_str(&z.model),
                json_str(&z.learning),
                z.color_hint,
                z.members
                    .iter()
                    .map(|m| m.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        s.push_str("]}");
        s
    }
}

/// Короткий формат числа: до 4 знаков, без хвостовых нулей; NaN/Infinity —
/// в «0» (защита от битых данных в JSON).
fn num(x: f32) -> String {
    if !x.is_finite() {
        return "0".to_string();
    }
    let mut s = format!("{:.4}", x);
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    if s == "-0" {
        s = "0".to_string();
    }
    s
}

/// JSON-строка с экранированием управляющих символов и `<`/`>`.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn bool01(b: bool) -> &'static str {
    if b {
        "1"
    } else {
        "0"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::grow::{avoidance_layout, grow, Steering};

    /// Схема + коннектом для тестов (короткий рост — быстрый, топология есть).
    fn fixture() -> (Layout, Grown) {
        let layout = avoidance_layout();
        let st = Steering {
            grow_steps: 200,
            ..Steering::default()
        };
        let grown = grow(&layout, &st, 42);
        assert!(!grown.edges.is_empty(), "фикстура должна иметь синапсы");
        (layout, grown)
    }

    #[test]
    fn region_count_matches_layout() {
        let (layout, grown) = fixture();
        let viz = brain_viz(&layout, &grown, None, &[], 20, 42);
        assert_eq!(viz.regions.len(), layout.regions.len());
    }

    #[test]
    fn neuron_count_matches_layout_total() {
        let (layout, grown) = fixture();
        let viz = brain_viz(&layout, &grown, None, &[], 20, 42);
        assert_eq!(viz.neurons.len(), layout.n());
    }

    #[test]
    fn link_endpoints_valid() {
        let (layout, grown) = fixture();
        let viz = brain_viz(&layout, &grown, None, &[], 20, 42);
        let n = layout.n();
        for l in &viz.links {
            assert!(l.pre < n && l.post < n, "ребро {} → {} вне границ", l.pre, l.post);
        }
        assert_eq!(viz.links.len(), grown.edges.len(), "усечение не должно сработать на фикстуре");
        assert_eq!(viz.meta.n_synapses, grown.edges.len());
    }

    #[test]
    fn activity_in_unit_range() {
        let (layout, grown) = fixture();
        let viz = brain_viz(&layout, &grown, None, &[], 20, 42);
        for n in &viz.neurons {
            assert!(
                (0.0..=1.0).contains(&n.activity),
                "активность нейрона {} вне [0,1]: {}",
                n.id,
                n.activity
            );
        }
        // Сенсоры под стимулами обязаны зажигаться (паттерн гоняет токи).
        let sensor_act = viz
            .neurons
            .iter()
            .filter(|n| n.region < 4)
            .map(|n| n.activity)
            .fold(0.0f32, |a, b| a + b);
        assert!(
            sensor_act > 0.0,
            "сенсорные регионы должны давать спайки под стимулами"
        );
    }

    #[test]
    fn json_is_v2_and_balanced() {
        let (layout, grown) = fixture();
        let viz = brain_viz(&layout, &grown, None, &[], 20, 42);
        let j = viz.to_json();
        assert!(j.starts_with("{\"version\":\"dioma_brain_viz_v2\""), "начало JSON: {j:?}");
        assert!(j.ends_with('}'), "конец JSON");
        assert!(!j.contains("NaN") && !j.contains("Infinity"), "в JSON нет NaN/Infinity");
        // Зоны присутствуют у любого датасета (у текущего — пустой список).
        assert!(j.contains("\"zones\":["), "ключ zones в JSON");
        // Метки регионов присутствуют.
        for label in ["SL", "SR", "WL", "WR", "I", "ML", "MR"] {
            assert!(j.contains(&format!("\"label\":\"{label}\"")), "метка {label} в JSON");
        }
    }

    #[test]
    fn json_deterministic_per_seed() {
        let (layout, grown) = fixture();
        let a = brain_viz(&layout, &grown, None, &[], 20, 42);
        let b = brain_viz(&layout, &grown, None, &[], 20, 42);
        assert_eq!(a.to_json(), b.to_json(), "один сид — один JSON");
    }

    #[test]
    fn real_positions_flow_through_json() {
        let layout = avoidance_layout();
        let st = Steering {
            grow_steps: 200,
            ..Steering::default()
        };
        let (grown, positions) = crate::engine::grow::grow_with_positions(&layout, &st, 42);
        assert_eq!(positions.len(), grown.nt.len(), "позиции на каждый нейрон");
        let viz = brain_viz(&layout, &grown, Some(&positions), &[], 20, 42);
        let j = viz.to_json();
        // Реальные сомы из ростового мира: SL-регион жил в районе (-60,-60).
        assert!(j.contains("\"pos\":["), "нейроны имеют позиции");
        let p0 = viz.neurons[0].pos;
        assert!(
            (p0[0] + 60.0).abs() < 25.0 && (p0[1] + 60.0).abs() < 25.0,
            "первый нейрон SL должен быть у (-60,-60): {p0:?}"
        );
        // Детерминированность с реальными позициями.
        let viz2 = brain_viz(&layout, &grown, Some(&positions), &[], 20, 42);
        assert_eq!(viz.to_json(), viz2.to_json());
    }

    #[test]
    fn future_concept_is_ghost_without_links() {
        let layout = avoidance_layout_future();
        assert_eq!(layout.regions.len(), AVOIDANCE_BASE_REGIONS + FUTURE_EXTRA_LABELS.len());
        let viz = brain_viz_concept(&layout, None, &[], 42);
        assert_eq!(viz.regions.len(), layout.regions.len());
        assert!(viz.links.is_empty(), "концепт без связей");
        assert_eq!(viz.meta.ticks_measured, 0);
        for (i, r) in viz.regions.iter().enumerate() {
            assert_eq!(r.ghost, i >= AVOIDANCE_BASE_REGIONS, "регион {i}");
        }
        for n in &viz.neurons {
            assert_eq!(n.activity, 0.0, "концепт без активности");
        }
        assert_eq!(viz.neurons.len(), layout.n());
        // Плановые регионы в JSON: ghost=1, метки транша 3, зон нет.
        let j = viz.to_json();
        assert!(j.contains("\"label\":\"VL\""));
        assert!(j.contains("\"ghost\":1"));
        assert!(j.starts_with("{\"version\":\"dioma_brain_viz_v2\""));
        assert!(j.contains("\"zones\":[]"), "старый концепт без зон");
    }

    /// Усечение связей: > LINK_CAP рёбер → срез ≤ LINK_CAP, каждое направление
    /// pre→post, представленное в исходнике, сохраняется (баланс групп).
    #[test]
    fn link_capping_keeps_direction_balance() {
        let layout = avoidance_layout();
        let ro = layout.region_of();
        let start = |r: usize| ro.iter().position(|&x| x == r).unwrap();
        let count = |r: usize| ro.iter().filter(|&&x| x == r).count();
        // 6 направлений × 900 рёбер = 5400: потолок 4000 обязан сработать.
        let dirs = [(0usize, 5usize), (0, 4), (1, 6), (2, 6), (3, 5), (4, 5)];
        let mut edges = Vec::new();
        for (di, &(a, b)) in dirs.iter().enumerate() {
            for k in 0..900usize {
                let pre = start(a) + (k + di * 31) % count(a);
                let post = start(b) + (k * 7 + di * 13) % count(b);
                edges.push((pre, post, 1.0 + (k % 9) as f32 * 0.5));
            }
        }
        let grown = Grown {
            edges,
            region_of: ro,
            nt: vec![1.0; layout.n()],
            growth_s: 0.0,
            rule_genes: Vec::new(),
            instincts: Vec::new(),
        };
        let viz = brain_viz(&layout, &grown, None, &[], 0, 1);
        assert!(viz.links.len() <= LINK_CAP, "срез ≤ {LINK_CAP}: {}", viz.links.len());
        assert_eq!(viz.meta.n_synapses, 5400, "мета хранит полный коннектом");
        assert!(viz.meta.note.contains("усечены"), "note: {}", viz.meta.note);
        let mut kept: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        for l in &viz.links {
            kept.insert((grown.region_of[l.pre], grown.region_of[l.post]));
        }
        assert_eq!(kept.len(), dirs.len(), "все направления pre→post сохранились");
        // Усечение детерминировано.
        let viz2 = brain_viz(&layout, &grown, None, &[], 0, 1);
        assert_eq!(viz.to_json(), viz2.to_json());
    }

    #[test]
    fn labels_fallback_to_table_and_r() {        let (layout, grown) = fixture();
        // Подсказка неверной длины → таблица AVOIDANCE_LABELS.
        let viz = brain_viz(&layout, &grown, None, &["X"], 10, 1);
        assert_eq!(viz.regions[0].label, "SL");
        assert_eq!(viz.regions[4].label, "I");
        // Неизвестная схема (одна регион) → R0.
        let single = Layout {
            regions: vec![RegionSpec {
                tag: "X",
                center: [0.0, 0.0, 0.0],
                radius: 10.0,
                count: 8,
                inhibitory: false,
                secrete: None,
                axons: vec![],
            }],
            allowed_pairs: vec![],
        };
        let grown_s = Grown {
            edges: vec![],
            region_of: single.region_of(),
            nt: vec![1.0; single.n()],
            growth_s: 0.0,
            rule_genes: Vec::new(),
            instincts: Vec::new(),
        };
        let viz2 = brain_viz(&single, &grown_s, None, &[], 0, 1);
        assert_eq!(viz2.regions[0].label, "R0");
        assert_eq!(viz2.links.len(), 0);
        assert!(viz2.meta.note.is_empty());
    }

    /// Зоны корректны: токены модели/обучения из допустимых множеств, hint в
    /// палитре, участники в диапазоне и покрывают ВСЕ регионы ровно по разу.
    fn assert_valid_zones(zones: &[ZoneViz], n_regions: usize) {
        const MODELS: [&str; 5] = [
            ZONE_MODEL_LIF,
            ZONE_MODEL_WC,
            ZONE_MODEL_LSTM,
            ZONE_MODEL_ANN,
            ZONE_MODEL_HYBRID,
        ];
        const LEARNS: [&str; 4] = [
            ZONE_LEARN_STDP,
            ZONE_LEARN_GENETIC,
            ZONE_LEARN_BAKED,
            ZONE_LEARN_PLANNED,
        ];
        let mut covered = vec![false; n_regions];
        for z in zones {
            assert!(MODELS.contains(&z.model.as_str()), "модель зоны «{}»: {}", z.label, z.model);
            assert!(
                LEARNS.contains(&z.learning.as_str()),
                "обучение зоны «{}»: {}",
                z.label,
                z.learning
            );
            assert!(
                z.color_hint < ZONE_MODEL_COUNT,
                "hint зоны «{}» вне палитры: {}",
                z.label,
                z.color_hint
            );
            for &m in &z.members {
                assert!(m < n_regions, "участник {m} зоны «{}» вне диапазона", z.label);
                assert!(!covered[m], "регион {m} заявлен в двух зонах");
                covered[m] = true;
            }
        }
        for (i, c) in covered.iter().enumerate() {
            assert!(*c, "регион {i} не входит ни в одну зону");
        }
    }

    #[test]
    fn current_builder_has_empty_zones() {
        let (layout, grown) = fixture();
        let viz = brain_viz_current(&layout, &grown, None, &[], 20, 42);
        assert!(viz.zones.is_empty(), "текущий мозг без зон");
        assert!(viz.to_json().contains("\"zones\":[]"));
        // brain_viz — алиас текущего пути (старая сигнатура сохранилась).
        let viz2 = brain_viz(&layout, &grown, None, &[], 20, 42);
        assert_eq!(viz.to_json(), viz2.to_json());
    }

    #[test]
    fn stage_b_builds_expected_zones() {
        let layout = avoidance_layout_future();
        let viz = brain_viz_stage_b(&layout, None, &[], 42);
        assert_eq!(
            layout.regions.len(),
            AVOIDANCE_BASE_REGIONS + FUTURE_EXTRA_LABELS.len()
        );
        assert_eq!(viz.regions.len(), 11);
        assert_eq!(viz.neurons.len(), 300, "120 ядро + 60 зрение + 120 ассоциативный");
        assert_eq!(viz.zones.len(), 3);
        assert_valid_zones(&viz.zones, 11);
        for (i, r) in viz.regions.iter().enumerate() {
            assert_eq!(r.ghost, i >= AVOIDANCE_BASE_REGIONS, "регион {i}");
        }
        assert!(viz.links.is_empty() && viz.meta.ticks_measured == 0, "концепт без симуляции");
        // Модели стадии B: ядро и хаб — LIF; зрение — LIF с ANN-планом.
        assert_eq!(viz.zones[0].model, ZONE_MODEL_LIF);
        assert_eq!(viz.zones[1].model, ZONE_MODEL_LIF);
        assert_eq!(viz.zones[1].learning, ZONE_LEARN_PLANNED, "зрение — к ANN-энкодеру");
        assert_eq!(viz.zones[2].model, ZONE_MODEL_LIF);
        let j = viz.to_json();
        assert!(j.contains("\"zones\":["));
        for label in ["Ядро рефлексов", "Зрение (сырые бины)", "Ассоциативный хаб"] {
            assert!(j.contains(&format!("\"label\":\"{label}\"")), "зона {label} в JSON");
        }
    }

    #[test]
    fn stage_b_layout_matches_viz_tables() {
        // Единый источник правды — grow::stage_b_layout(); визный датасет
        // стадии B ходит через avoidance_layout_future (алиас). Сверяем
        // НЕЗАВИСИМЫЕ константы виза (метки/классы/база) с фактической схемой:
        // рассинхрон чисел или меток упадёт здесь.
        let l = crate::engine::grow::stage_b_layout();
        let future = avoidance_layout_future();
        assert_eq!(l.regions.len(), AVOIDANCE_BASE_REGIONS + FUTURE_EXTRA_LABELS.len());
        assert_eq!(future.regions.len(), l.regions.len(), "алиас возвращает ту же схему");
        // Метки хвоста схемы == таблица транша 3 (порядок VL/VR/ASSOC_E/ASSOC_I).
        let tail: Vec<&str> = l.regions.iter().skip(AVOIDANCE_BASE_REGIONS).map(|r| r.tag).collect();
        assert_eq!(tail, FUTURE_EXTRA_LABELS, "метки хвоста стадии B");
        // Числа нейронов: ядро 120 + 30 + 30 + 90 + 30 = 300 (суммы виза и схемы).
        let sum_viz: usize = FUTURE_EXTRA_LABELS
            .iter()
            .map(|t| {
                l.regions
                    .iter()
                    .find(|r| r.tag == *t)
                    .map_or(0, |r| r.count)
            })
            .sum();
        assert_eq!(sum_viz, 180, "добавка транша 3: 30+30+90+30");
        assert_eq!(l.n(), 300, "120 ядро + 180 транш 3");
        assert_eq!(future.n(), 300);
        // Классы транша 3: зрение = sensor, ассоциации = assoc.
        assert_eq!(FUTURE_EXTRA_KINDS, &["sensor", "sensor", "assoc", "assoc"]);
        for (t, kind) in tail.iter().zip(FUTURE_EXTRA_KINDS) {
            let r = l.regions.iter().find(|r| &r.tag == t).unwrap();
            let expected_kind = match *kind {
                "sensor" => !r.inhibitory,
                "assoc" => true,
                _ => false,
            };
            assert!(expected_kind, "регион {t}: класс {kind}");
        }
        // Алиас и источник — одна схема (тот же порядок и размеры регионов).
        for (a, b) in future.regions.iter().zip(l.regions.iter()) {
            assert_eq!(a.tag, b.tag);
            assert_eq!(a.count, b.count);
        }
    }

    #[test]
    fn stage_c_concept_zones() {
        let viz = brain_viz_stage_c();
        assert_eq!(viz.regions.len(), 21);
        assert_eq!(viz.neurons.len(), 1245, "сумма фиксированных размеров зон стадии C");
        assert_eq!(viz.zones.len(), 9);
        assert_valid_zones(&viz.zones, 21);
        assert!(viz.links.is_empty(), "чистый дизайн без связей");
        assert!(viz.meta.note.contains("стадия C"), "note: {}", viz.meta.note);
        assert_eq!(viz.meta.ticks_measured, 0);
        for r in &viz.regions {
            assert!(r.ghost, "все регионы концепта — призраки");
        }
        for n in &viz.neurons {
            assert_eq!(n.activity, 0.0, "концепт без активности");
        }
        // Все пять моделей гибридного мозга представлены.
        let models: std::collections::HashSet<&str> =
            viz.zones.iter().map(|z| z.model.as_str()).collect();
        for m in [
            ZONE_MODEL_LIF,
            ZONE_MODEL_WC,
            ZONE_MODEL_LSTM,
            ZONE_MODEL_ANN,
            ZONE_MODEL_HYBRID,
        ] {
            assert!(models.contains(m), "модель {m} есть в концепте зон");
        }
        let j = viz.to_json();
        assert!(j.starts_with("{\"version\":\"dioma_brain_viz_v2\""));
        assert!(j.contains("\"zones\":["));
        assert!(!j.contains("NaN") && !j.contains("Infinity"));
        for label in [
            "Ядро",
            "Зрение (энкодер)",
            "Ассоциативный хаб",
            "Планирование",
            "Гиппокамп (память)",
            "Лимбика",
            "Оценка",
            "Социальная",
            "Премотор",
        ] {
            assert!(j.contains(&format!("\"label\":\"{label}\"")), "зона {label} в JSON");
        }
    }

    #[test]
    fn stage_c_json_is_deterministic() {
        let a = brain_viz_stage_c();
        let b = brain_viz_stage_c();
        assert_eq!(a.to_json(), b.to_json(), "концепт зон детерминирован между вызовами");
    }
}
