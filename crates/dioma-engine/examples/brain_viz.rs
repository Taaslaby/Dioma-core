//! # brain_viz — презентационный HTML-вьюер мозга агента
//!
//! Экспортирует визуальные данные выращенного мозга (`brain::viz`) в ОДИН
//! самодостаточный HTML-файл: никаких внешних библиотек и сети, работает
//! офлайн через `file://`. В файле всегда три вкладки:
//!
//! - «Текущий мозг» — выращенный LIF-мозг (реальные сомы + активность);
//! - «Транш 3: зрение» — план (стадия B): зрение VL/VR и ассоциативный слой
//!   (полупрозрачные «призраки») + зоны долей;
//! - «Концепт зон» — гибридный мозг (стадия C): зоны-модели (LIF/WC/LSTM/ANN/
//!   hybrid) ореолами с цветом по модели, чисты дизайн, без проводки.
//!
//! Запуск (из `crates/dioma-engine`):
//!
//! ```text
//! cargo run --release --example brain_viz -- --seed 42 --out brain.html --ticks 600
//! cargo run --release --example brain_viz -- --seed 42 --tab zones --out brain_zones.html
//! ```
//!
//! Данные: формат `dioma_brain_viz_v2` (см. `brain::viz`; v2 = v1 + `zones`).
//! Позиции нейронов — РЕАЛЬНЫЕ финальные сомы из ростового мира
//! (`grow::grow_with_positions`), активность — замер на настоящем LIF-мозге
//! под смешанным стимульным паттерном. Концепт-вкладки: регионы не выращены,
//! связей и активности нет (честно помечено в `meta.note`).

use dioma_engine::creature::genome::CreatureGenome;
use dioma_engine::brain::grow::{avoidance_layout, grow_with_positions};
use dioma_engine::brain::viz::{
    avoidance_layout_future, brain_viz, brain_viz_stage_b, brain_viz_stage_c, BrainViz,
    AVOIDANCE_LABELS, FUTURE_EXTRA_LABELS,
};
use std::env;
use std::fs;
use std::path::PathBuf;

/// Русская справка `--help`.
const USAGE: &str = "\
brain_viz — самодостаточный HTML-вьюер мозга агента dioma (без сети и библиотек)

Использование:
  cargo run --release --example brain_viz -- [--seed N] [--out PATH] [--ticks N] [--tab NAME] [--future] [--help]

Параметры:
  --seed N      сид существа (по умолчанию 42)
  --out PATH    путь к выходному HTML-файлу (по умолчанию brain.html)
  --ticks N     мировых тиков замера активности мозга (по умолчанию 600)
  --tab NAME    вкладка первой: current | stageb | zones (по умолчанию current) —
                файл всегда содержит все три вкладки
  --future      = --tab stageb (совместимость со старым флагом)
  --help        эта справка

Примеры:
  cargo run --release --example brain_viz -- --seed 42 --out docs/viz/brain_seed42.html
  cargo run --release --example brain_viz -- --seed 1337 --out docs/viz/brain_seed1337.html
  cargo run --release --example brain_viz -- --seed 42 --tab zones --out docs/viz/brain_zones.html
";

/// Разобранные аргументы командной строки.
struct Args {
    seed: u64,
    out: PathBuf,
    ticks: usize,
    /// Вкладка, активная первой: `current` | `stageb` | `zones`.
    tab: String,
    help: bool,
}

/// Парсер аргументов: `std::env::args` без внешних зависимостей.
fn parse_args() -> Args {
    let mut a = Args {
        seed: 42,
        out: PathBuf::from("brain.html"),
        ticks: 600,
        tab: "current".to_string(),
        help: false,
    };
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--help" | "-h" => a.help = true,
            "--seed" => {
                if let Some(v) = it.next().and_then(|s| s.parse().ok()) {
                    a.seed = v;
                } else {
                    eprintln!("[brain_viz] --seed: нужен целый сид");
                    a.help = true;
                }
            }
            "--out" => {
                if let Some(v) = it.next() {
                    a.out = PathBuf::from(v);
                } else {
                    eprintln!("[brain_viz] --out: нужен путь");
                    a.help = true;
                }
            }
            "--ticks" => {
                if let Some(v) = it.next().and_then(|s| s.parse().ok()) {
                    a.ticks = v;
                } else {
                    eprintln!("[brain_viz] --ticks: нужно целое число тиков");
                    a.help = true;
                }
            }
            "--tab" => match it.next().as_deref() {
                Some(t @ ("current" | "stageb" | "zones")) => a.tab = t.to_string(),
                _ => {
                    eprintln!("[brain_viz] --tab: current | stageb | zones");
                    a.help = true;
                }
            },
            "--future" => a.tab = "stageb".to_string(),
            other => {
                eprintln!("[brain_viz] неизвестный аргумент: {other}");
                a.help = true;
            }
        }
    }
    a
}

fn main() {
    let args = parse_args();
    if args.help {
        print!("{USAGE}");
        return;
    }

    // Полный игровой конвейер, как в Creature::new: дефолтный геном,
    // рост коннектома из сида + реальные позиции сом из ростового мира.
    let genome = CreatureGenome::default();
    let layout = avoidance_layout();
    eprintln!(
        "[brain_viz] рост коннектома: сид {} ({} регионов, {} нейронов)...",
        args.seed,
        layout.regions.len(),
        layout.n()
    );
    let (grown, positions) = grow_with_positions(&layout, &genome.steering, args.seed);
    eprintln!(
        "[brain_viz] коннектом: {} нейронов, {} синапсов, рост {:.3} с",
        grown.nt.len(),
        grown.edges.len(),
        grown.growth_s
    );

    // Вкладка 1 — текущий мозг (стадия A, зон нет).
    let viz = brain_viz(
        &layout,
        &grown,
        Some(&positions),
        AVOIDANCE_LABELS,
        args.ticks,
        args.seed,
    );

    // Вкладка 2 — план транша 3 (стадия B): базовая схема + плановые регионы,
    // с зонами долей.
    let stage_b_layout = avoidance_layout_future();
    let stage_b_labels: Vec<&str> = AVOIDANCE_LABELS
        .iter()
        .chain(FUTURE_EXTRA_LABELS.iter())
        .copied()
        .collect();
    let viz_stage_b = brain_viz_stage_b(&stage_b_layout, Some(&positions), &stage_b_labels, args.seed);

    // Вкладка 3 — концепт зон (стадия C): чистый дизайн, без проводки.
    let viz_zones = brain_viz_stage_c();

    let html = build_html(&viz, &viz_stage_b, &viz_zones, &args.tab);
    if let Err(e) = fs::write(&args.out, &html) {
        eprintln!("[brain_viz] не удалось записать {}: {e}", args.out.display());
        std::process::exit(1);
    }
    eprintln!(
        "[brain_viz] готово: {} ({} байт, вкладок: текущий мозг + транш 3 + концепт зон; \
         активна — {})",
        args.out.display(),
        html.len(),
        args.tab
    );
    if !viz.meta.note.is_empty() {
        eprintln!("[brain_viz] примечание (текущий): {}", viz.meta.note);
    }
    if !viz_stage_b.meta.note.is_empty() {
        eprintln!("[brain_viz] примечание (транш 3): {}", viz_stage_b.meta.note);
    }
    if !viz_zones.meta.note.is_empty() {
        eprintln!("[brain_viz] примечание (концепт зон): {}", viz_zones.meta.note);
    }
}

/// Сборка одного HTML-файла: голова + три датасета (JSON) + JS-вьюер + хвост.
/// Сборка конкатенацией, без `format!` — экранирование фигурных скобок
/// шаблонов не требуется, JSON вставляется как есть (внутри него `\u003c`,
/// так что из скрипта он не выпадет).
fn build_html(cur: &BrainViz, stage_b: &BrainViz, zones: &BrainViz, active_tab: &str) -> String {
    let active = match active_tab {
        "stageb" => "stageb",
        "zones" => "zones",
        _ => "current",
    };
    let mut s = String::with_capacity(768 * 1024);
    s.push_str(HEAD);
    s.push_str("\n<script>\n\"use strict\";\n");
    s.push_str("// Данные: формат dioma_brain_viz_v2 (генерирует brain::viz::BrainViz::to_json)\n");
    s.push_str("const BRAIN_CURRENT = ");
    s.push_str(&cur.to_json());
    s.push_str(";\n\nconst BRAIN_STAGEB = ");
    s.push_str(&stage_b.to_json());
    s.push_str(";\n\nconst BRAIN_ZONES = ");
    s.push_str(&zones.to_json());
    s.push_str(";\n\nconst INITIAL_TAB = \"");
    s.push_str(active);
    s.push_str("\";\n");
    s.push_str(VIEWER_JS);
    s.push_str("\n</script>\n</body>\n</html>\n");
    s
}

// ═══════════════════════════════════════════════════════════════════════════
// Шаблон HTML: голова (стили + разметка). Без внешних ресурсов: всё inline.
// ═══════════════════════════════════════════════════════════════════════════
const HEAD: &str = r##"<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Мозг агента · dioma</title>
<style>
:root{
  --bg0:#04060c; --bg1:#0b1322; --panel:rgba(14,21,38,.86);
  --border:rgba(96,128,190,.22); --text:#dbe6f5; --dim:#8fa3c0; --dim2:#5f7394;
  --accent:#41d7ff; --orange:#ffa940; --violet:#b388ff; --crimson:#ff5d8a;
  --green:#2bd97a; --red:#ff3b5c;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;overflow:hidden;background:var(--bg0);color:var(--text);
  font:14px/1.45 "Segoe UI",system-ui,-apple-system,sans-serif}
#app{display:flex;flex-direction:column;height:100%}
header{display:flex;align-items:center;gap:18px;padding:10px 18px;flex:0 0 auto;
  background:linear-gradient(180deg,rgba(22,32,56,.96),rgba(11,17,32,.92));
  border-bottom:1px solid var(--border)}
.brand{font-weight:700;letter-spacing:.4px;color:var(--accent);font-size:15px;white-space:nowrap}
.brand span{color:var(--dim);font-weight:500}
#title{font-size:14px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tabs{margin-left:auto;display:flex;gap:6px}
.tab{padding:6px 14px;border-radius:999px;border:1px solid var(--border);
  background:transparent;color:var(--dim);cursor:pointer;font-size:13px;transition:.15s}
.tab:hover{color:var(--text)}
.tab.active{background:rgba(65,215,255,.13);color:var(--accent);border-color:rgba(65,215,255,.45)}
main{display:flex;flex:1;min-height:0}
#stage{position:relative;flex:1;min-width:0;background:
  radial-gradient(1200px 800px at 50% 35%, #0d1830 0%, #070c18 55%, #04060c 100%)}
#stage::after{content:"";position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(ellipse at 50% 45%, transparent 55%, rgba(0,0,0,.34) 100%)}
canvas{position:absolute;inset:0;width:100%;height:100%;display:block;cursor:grab}
canvas.dragging{cursor:grabbing}
#hint{position:absolute;left:12px;bottom:10px;font-size:11px;color:var(--dim2);
  background:rgba(6,10,18,.55);padding:4px 10px;border-radius:8px;pointer-events:none;z-index:2}
#side{width:272px;flex:0 0 auto;background:var(--panel);border-left:1px solid var(--border);
  padding:14px 16px;overflow-y:auto;display:flex;flex-direction:column;gap:15px;z-index:3}
#side section h3{margin:0 0 8px;font-size:11px;text-transform:uppercase;letter-spacing:.8px;
  color:var(--dim2);font-weight:600}
.row{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:13px;color:var(--text);cursor:pointer}
.row input[type=checkbox]{accent-color:var(--accent);width:15px;height:15px;cursor:pointer}
.row input:disabled{opacity:.35;cursor:default}
#reglist{display:flex;flex-direction:column;gap:3px;max-height:218px;overflow-y:auto;padding-right:4px}
.reg{display:flex;align-items:center;gap:8px;font-size:12.5px;cursor:pointer;padding:2px 4px;border-radius:6px}
.reg:hover{background:rgba(255,255,255,.05)}
.reg input{accent-color:var(--accent);cursor:pointer}
.reg .dot{width:9px;height:9px;border-radius:50%;flex:0 0 auto}
.reg .cnt{margin-left:auto;color:var(--dim);font-size:11px;font-family:ui-monospace,Consolas,monospace}
#zonelist{display:flex;flex-direction:column;gap:3px;max-height:216px;overflow-y:auto;padding-right:4px}
.zrow{margin:2px 0;padding:3px 4px;border-radius:6px}
.zrow:hover{background:rgba(255,255,255,.05)}
.zrow .row{margin:0}
.zmeta{margin:1px 0 0 29px;font-size:10.5px;color:var(--dim2);font-family:ui-monospace,Consolas,monospace}
.btnrow{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}
.mini{padding:4px 10px;border-radius:8px;border:1px solid var(--border);background:rgba(255,255,255,.03);
  color:var(--dim);cursor:pointer;font-size:12px;transition:.15s}
.mini:hover{color:var(--text);border-color:var(--dim)}
input[type=range]{width:100%;accent-color:var(--accent);margin:2px 0}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:11px;color:var(--dim)}
.note{font-size:12px;color:#ffd9a0;background:rgba(255,170,60,.08);border:1px solid rgba(255,170,60,.25);
  padding:8px 10px;border-radius:10px;line-height:1.45}
#legend{display:flex;flex-direction:column;gap:6px;font-size:12px;color:var(--dim)}
#legend .lrow{display:flex;align-items:center;gap:8px}
.lsep{margin:7px 0 3px;font-size:10px;text-transform:uppercase;letter-spacing:.7px;color:var(--dim2)}
.sw{width:18px;height:9px;border-radius:4px;flex:0 0 auto}
.sw.dot{width:9px;height:9px;border-radius:50%}
.sw.dash{background:repeating-linear-gradient(90deg,var(--red) 0 4px,transparent 4px 8px)}
#metabox{font-size:11px;color:var(--dim2);border-top:1px solid var(--border);padding-top:10px;
  word-break:break-word;white-space:pre-line;line-height:1.6}
/* Режим презентации: тёмный фон #0a0e14, крупные подписи, авто-контраст */
body.present{--bg0:#0a0e14;--bg1:#101624}
body.present header{background:linear-gradient(180deg,#101826,#0a0e14)}
body.present .tab{font-size:14px;padding:8px 18px}
body.present #title{font-size:17px;font-weight:600}
body.present .reg,body.present .row{font-size:14px}
body.present #side{background:rgba(13,18,30,.9)}
body.present #stage{background:
  radial-gradient(1200px 800px at 50% 35%, #0f1a30 0%, #0b101c 55%, #0a0e14 100%)}
@media (max-width:900px){#side{width:228px}}
</style>
</head>
<body>
<div id="app">
  <header id="top">
    <div class="brand">dioma<span> · мозг</span></div>
    <div id="title"></div>
    <nav class="tabs">
      <button class="tab" data-tab="current">Текущий мозг</button>
      <button class="tab" data-tab="stageb">Транш 3: зрение</button>
      <button class="tab" data-tab="zones">Концепт зон</button>
    </nav>
  </header>
  <main>
    <div id="stage">
      <canvas id="cv"></canvas>
      <div id="hint">ЛКМ — вращение · колесо — зум · 2 клика — сброс · Пробел — пауза</div>
    </div>
    <aside id="side">
      <div id="note" class="note" hidden></div>
      <section id="zonesec" hidden>
        <h3>Зоны</h3>
        <label class="row"><input type="checkbox" id="f-zones" checked> <span>Показать зоны</span></label>
        <div id="zonelist"></div>
        <div class="btnrow">
          <button id="b-zone-all" class="mini">Все</button>
          <button id="b-zone-none" class="mini">Никакие</button>
        </div>
      </section>
      <section>
        <h3>Регионы</h3>
        <div id="reglist"></div>
        <div class="btnrow">
          <button id="b-sensor" class="mini">Только сенсоры</button>
          <button id="b-motor" class="mini">Только моторы</button>
          <button id="b-all" class="mini">Всё</button>
        </div>
      </section>
      <section>
        <h3>Фильтры</h3>
        <label class="row"><input type="checkbox" id="f-ghost" checked> <span>Показать будущие регионы</span></label>
        <label class="row"><input type="checkbox" id="f-labels" checked> <span>Подписи</span></label>
        <label class="row"><input type="checkbox" id="f-links" checked> <span>Связи</span></label>
      </section>
      <section>
        <h3>Оформление</h3>
        <label class="row">Порог веса связей <span id="wt-val" class="mono"></span>
          <input type="range" id="s-wt" min="0" max="100" value="2"></label>
        <label class="row">Размер точек <span id="ps-val" class="mono"></span>
          <input type="range" id="s-ps" min="50" max="300" value="100"></label>
        <label class="row"><input type="checkbox" id="f-present"> <span>Режим презентации</span></label>
      </section>
      <section>
        <h3>Легенда</h3>
        <div id="legend"></div>
      </section>
      <section id="metabox"></section>
    </aside>
  </main>
</div>
"##;

// ═══════════════════════════════════════════════════════════════════════════
// Вьюер: чистый canvas 2D + ручная перспективная проекция (без WebGL).
// Orbit-управление, depth cueing, painter-сортировка, ореолы зон, подписи
// и легенда.
// ═══════════════════════════════════════════════════════════════════════════
const VIEWER_JS: &str = r##"// ─── Мозг-вьюер dioma: canvas 2D, ручная 3D→2D проекция ────────────────────

const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => v < a ? a : v > b ? b : v;

// ── состояние вьюера ──
let DATA = null;            // активный датасет
let cur = null;             // препроцессированный датасет
let tab = INITIAL_TAB === 'zones' ? 'zones' : INITIAL_TAB === 'stageb' ? 'stageb' : 'current';
let yaw = 0.65, pitch = 0.38, dist = 3.8;
const DEF = { yaw: 0.65, pitch: 0.38, dist: 3.8 };
let autospin = true, paused = false, dragging = false;
let showGhosts = true, showLabels = true, showLinks = true;
let showZones = true, zoneVis = [];
let wtFrac = 0.02, ptScale = 1.0, present = false;
let regionVis = [];
let dirty = true, lowGlow = false;
let lastX = 0, lastY = 0, lastT = performance.now();
let W = 0, H = 0;

// палитра классов регионов (сенсоры — циан, моторы — оранж,
// тормозные — красно-пурпурные, ассоциативный слой — фиолетовый)
const KIND_COLOR = {
  sensor: '#41d7ff', motor: '#ffa940', inter: '#ff5d8a',
  assoc: '#b388ff', ghost: '#8fa3c0'
};

// палитра МОДЕЛЕЙ зон (color_hint в данных): lif=циан, wc=оранж, lstm=фиолет,
// ann=зелёный, hybrid=розовый — гибридность видна сразу по цвету ореола
const MODEL_COLOR = ['#41d7ff', '#ffa940', '#b388ff', '#2bd97a', '#ff7eb6'];
const MODEL_TAG = {
  lif: 'LIF · STDP онлайн',
  wc: 'Wilson–Cowan · параметры из генома',
  lstm: 'LSTM · веса запечены (градиент офлайн)',
  ann: 'ANN-энкодер · веса запечены (градиент офлайн)',
  hybrid: 'гибрид · регуляторный геном'
};

function hexA(hex, a) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
}
function mixHex(c1, c2, t, a) {
  const r1 = parseInt(c1.slice(1, 3), 16), g1 = parseInt(c1.slice(3, 5), 16), b1 = parseInt(c1.slice(5, 7), 16);
  const r2 = parseInt(c2.slice(1, 3), 16), g2 = parseInt(c2.slice(3, 5), 16), b2 = parseInt(c2.slice(5, 7), 16);
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const b = Math.round(b1 + (b2 - b1) * t);
  return 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
}
// цвет региона: класс + призрачность
function tintOf(r) {
  const c = KIND_COLOR[r.kind] || '#8fa3c0';
  return r.ghost ? hexA(c, 0.4) : c;
}

// ── препроцессинг датасета: нормализация в единичный шар + индексы + зоны ──
function prep(data) {
  const n0 = data.neurons.length;
  let cx = 0, cy = 0, cz = 0;
  for (const nn of data.neurons) { cx += nn.pos[0]; cy += nn.pos[1]; cz += nn.pos[2]; }
  cx /= Math.max(1, n0); cy /= Math.max(1, n0); cz /= Math.max(1, n0);
  let R = 1e-6;
  for (const nn of data.neurons) {
    R = Math.max(R, Math.hypot(nn.pos[0] - cx, nn.pos[1] - cy, nn.pos[2] - cz));
  }
  const S = 1 / R;
  const regions = data.regions.map(r => ({
    label: r.label, kind: r.kind, ghost: !!r.ghost,
    center: [(r.center[0] - cx) * S, (r.center[1] - cy) * S, (r.center[2] - cz) * S],
    radius: r.radius * S
  }));
  const neurons = data.neurons.map(nn => ({
    id: nn.id, region: nn.region, inh: !!nn.inh, act: nn.act,
    pos: [(nn.pos[0] - cx) * S, (nn.pos[1] - cy) * S, (nn.pos[2] - cz) * S],
    ghost: !!(regions[nn.region] && regions[nn.region].ghost)
  }));
  let wmax = 1e-6;
  for (const l of data.links) wmax = Math.max(wmax, Math.abs(l.w));
  const links = data.links.map(l => ({
    pre: l.pre, post: l.post, w: l.w, inh: !!l.inh, w01: Math.abs(l.w) / wmax
  }));
  const counts = new Array(regions.length).fill(0);
  for (const nn of neurons) counts[nn.region]++;
  const maxAct = neurons.reduce((m, x) => Math.max(m, x.act), 0) || 1;
  // зоны: центроид участников (нормализованные координаты) + радиус ореола,
  // цвет по палитре моделей (color_hint в данных — индекс палитры)
  const zones = (data.zones || []).map(z => {
    const mem = z.members.map(i => regions[i]).filter(Boolean);
    let zx = 0, zy = 0, zz = 0;
    for (const r of mem) { zx += r.center[0]; zy += r.center[1]; zz += r.center[2]; }
    const k = Math.max(1, mem.length);
    zx /= k; zy /= k; zz /= k;
    let zR = 1e-6;
    for (const r of mem) {
      zR = Math.max(zR, Math.hypot(r.center[0] - zx, r.center[1] - zy, r.center[2] - zz) + r.radius);
    }
    let count = 0;
    for (const i of z.members) count += counts[i] || 0;
    const hint = (z.color_hint | 0) % MODEL_COLOR.length;
    return {
      label: z.label, desc: z.desc, model: z.model, learning: z.learning,
      color: MODEL_COLOR[hint], tag: MODEL_TAG[z.model] || z.model,
      center: [zx, zy, zz], radius: Math.max(zR, 0.4), count, members: z.members
    };
  });
  return { data, regions, neurons, links, counts, maxAct, zones };
}

function setData(d, keepVis) {
  DATA = d;
  cur = prep(d);
  if (!keepVis || regionVis.length !== cur.regions.length) {
    regionVis = cur.regions.map(r => !r.ghost || showGhosts);
  }
  if (!showGhosts) regionVis = regionVis.map((v, i) => v && !cur.regions[i].ghost);
  if (!keepVis || zoneVis.length !== cur.zones.length) zoneVis = cur.zones.map(() => true);
  $('title').textContent = d.meta.title;
  const nb = $('note');
  if (d.meta.note) { nb.hidden = false; nb.textContent = d.meta.note; } else { nb.hidden = true; }
  $('metabox').textContent =
    'Формат: ' + d.version + '\n' +
    'Тиков замера активности: ' + d.meta.ticks_measured + '\n' +
    'Синапсов в коннектоме: ' + d.meta.n_synapses + '\n' +
    'Связей в срезе: ' + d.links.length + '\n' +
    'Зон: ' + cur.zones.length;
  const hasGhost = cur.regions.some(r => r.ghost);
  $('f-ghost').disabled = !hasGhost;
  $('f-ghost').checked = showGhosts;
  const hasZones = cur.zones.length > 0;
  $('zonesec').hidden = !hasZones;
  $('f-zones').disabled = !hasZones;
  $('f-zones').checked = showZones;
  buildZoneList();
  buildRegionList();
  dirty = true;
}

function buildZoneList() {
  const box = $('zonelist');
  box.textContent = '';
  cur.zones.forEach((z, i) => {
    const wrap = document.createElement('div');
    wrap.className = 'zrow';
    wrap.title = z.desc + ' · ' + z.learning;
    const lab = document.createElement('label');
    lab.className = 'row';
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = zoneVis[i] !== false;
    cb.addEventListener('change', () => { zoneVis[i] = cb.checked; dirty = true; });
    const dot = document.createElement('span');
    dot.className = 'dot'; dot.style.background = z.color;
    const nm = document.createElement('span'); nm.textContent = z.label;
    const cnt = document.createElement('span'); cnt.className = 'cnt';
    cnt.textContent = z.count;
    lab.append(cb, dot, nm, cnt);
    const meta = document.createElement('div'); meta.className = 'zmeta';
    meta.textContent = z.learning;
    wrap.append(lab, meta);
    box.appendChild(wrap);
  });
}

function buildRegionList() {
  const box = $('reglist');
  box.textContent = '';
  cur.regions.forEach((r, i) => {
    const lab = document.createElement('label');
    lab.className = 'reg';
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = regionVis[i] !== false;
    cb.addEventListener('change', () => { regionVis[i] = cb.checked; dirty = true; });
    const dot = document.createElement('span');
    dot.className = 'dot'; dot.style.background = tintOf(r);
    const nm = document.createElement('span'); nm.textContent = r.label;
    const cnt = document.createElement('span'); cnt.className = 'cnt';
    cnt.textContent = cur.counts[i];
    lab.append(cb, dot, nm, cnt);
    box.appendChild(lab);
  });
}

// ── перспективная проекция: yaw по Y, pitch по X, камера на -Z ──
function proj(p) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x = p[0] * cy + p[2] * sy;
  const z = -p[0] * sy + p[2] * cy;
  const y2 = p[1] * cp - z * sp;
  const z2 = p[1] * sp + z * cp;
  const f = Math.max(W, H) * 1.35;
  const sc = f / (dist - z2);
  return { x: W / 2 + x * sc, y: H / 2 - y2 * sc, z: z2, s: sc };
}

function render() {
  // фон: вертикальный градиент под космос
  const bg = ctx.createLinearGradient(0, 0, 0, H);
  bg.addColorStop(0, present ? '#0f1522' : '#0b1322');
  bg.addColorStop(1, present ? '#0a0e14' : '#04060c');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  const regs = cur.regions, neurons = cur.neurons, links = cur.links;
  const vis = (i) => regionVis[i] !== false;
  const hasZones = cur.zones.length > 0;

  // зум-зависимая детализация: далеко — ореолы и чипы зон, близко — ореолы
  // бледнеют, нейроны/связи читаемы
  const FAR = 5.0, NEAR = 2.9;
  const zoomT = clamp((dist - NEAR) / (FAR - NEAR), 0, 1); // 0 близко, 1 далеко
  const zoneAlpha = hasZones ? clamp(0.30 + 0.70 * zoomT, 0.30, 1.0) : 0;
  const farMode = hasZones && dist > FAR;

  // ── 0. ореолы зон: большие полупрозрачные пятна цвета модели, фон ──
  if (showZones && hasZones && zoneAlpha > 0.01) {
    const halos = [];
    for (let i = 0; i < cur.zones.length; i++) {
      const z = cur.zones[i];
      if (zoneVis[i] !== false) halos.push({ z, p: proj(z.center) });
    }
    halos.sort((a, b) => a.p.z - b.p.z);
    for (const h of halos) drawHalo(h, zoneAlpha);
  }

  // ── далёкий зум: только ореолы + жирные чипы зон ──
  if (farMode) {
    if (showLabels && showZones) {
      const chips = [];
      for (let i = 0; i < cur.zones.length; i++) {
        if (zoneVis[i] !== false) chips.push({ z: cur.zones[i], p: proj(cur.zones[i].center) });
      }
      chips.sort((a, b) => a.p.z - b.p.z);
      for (const c of chips) drawZoneChip(c, 0.95);
    }
    return;
  }

  // ── 1. оболочки регионов: полупрозрачные эллипсоиды, painter ──
  const hulls = [];
  for (let i = 0; i < regs.length; i++) {
    const r = regs[i];
    if (!vis(i) || (r.ghost && !showGhosts)) continue;
    hulls.push({ r, c: proj(r.center) });
  }
  hulls.sort((a, b) => a.c.z - b.c.z);
  for (const h of hulls) drawHull(h);

  // ── 2. связи: сортировка по глубине, альфа/толщина по дистанции ──
  if (showLinks) {
    const items = [];
    for (const l of links) {
      if (l.w01 < wtFrac) continue;
      const pa = neurons[l.pre], pb = neurons[l.post];
      if (!pa || !pb) continue;
      if (!vis(pa.region) || !vis(pb.region)) continue;
      if ((pa.ghost || pb.ghost) && !showGhosts) continue;
      const a = proj(pa.pos), b = proj(pb.pos);
      items.push({ a, b, l, z: (a.z + b.z) / 2 });
    }
    items.sort((x, y) => x.z - y.z);
    for (const it of items) drawLink(it);
  }

  // ── 3. нейроны: светящиеся точки, яркость по активности ──
  const dots = [];
  for (const nn of neurons) {
    if (!vis(nn.region)) continue;
    if (nn.ghost && !showGhosts) continue;
    dots.push({ nn, p: proj(nn.pos) });
  }
  dots.sort((a, b) => a.p.z - b.p.z);
  for (const d of dots) drawNeuron(d);

  // ── 4. чипы зон: жирные подписи с тегом модели, поверх нейронов ──
  if (showZones && hasZones && showLabels) {
    const chips = [];
    for (let i = 0; i < cur.zones.length; i++) {
      if (zoneVis[i] !== false) chips.push({ z: cur.zones[i], p: proj(cur.zones[i].center) });
    }
    chips.sort((a, b) => a.p.z - b.p.z);
    const chipA = clamp(0.55 + 0.4 * zoomT, 0.55, 0.95);
    for (const c of chips) drawZoneChip(c, chipA);
  }

  // ── 5. подписи регионов (поверх всего) ──
  if (showLabels) {
    const labs = [];
    for (let i = 0; i < regs.length; i++) {
      const r = regs[i];
      if (!vis(i) || (r.ghost && !showGhosts)) continue;
      labs.push({ r, c: proj(r.center), i });
    }
    labs.sort((a, b) => a.c.z - b.c.z);
    for (const l of labs) drawLabel(l);
  }
}

// эллипсоид-каркас: три больших круга (XY, XZ, YZ)
function ring3D(r, u, v, N) {
  const pts = [];
  for (let k = 0; k < N; k++) {
    const t = (k / N) * Math.PI * 2;
    const p = [0, 0, 0];
    p[u] = Math.cos(t) * r.radius;
    p[v] = Math.sin(t) * r.radius;
    pts.push([r.center[0] + p[0], r.center[1] + p[1], r.center[2] + p[2]]);
  }
  return pts;
}
function drawHull(h) {
  const r = h.r;
  const alpha = r.ghost ? (present ? 0.24 : 0.15) : (present ? 0.40 : 0.28);
  const col = KIND_COLOR[r.kind] || '#8fa3c0';
  ctx.strokeStyle = hexA(col, alpha);
  ctx.lineWidth = r.ghost ? 1.2 : 1.5;
  ctx.setLineDash(r.ghost ? [3, 5] : []);
  for (const pair of [[0, 1], [0, 2], [1, 2]]) {
    const ring = ring3D(r, pair[0], pair[1], 28);
    ctx.beginPath();
    for (let k = 0; k < ring.length; k++) {
      const p = proj(ring[k]);
      if (k === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

// ореол зоны: мягкое пятно цвета модели + кольцо-граница
function drawHalo(h, alpha) {
  const p = h.p;
  const Rpx = Math.max(2, h.z.radius * p.s);
  const col = h.z.color;
  ctx.save();
  const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, Rpx);
  g.addColorStop(0, hexA(col, 0.20 * alpha));
  g.addColorStop(0.62, hexA(col, 0.11 * alpha));
  g.addColorStop(1, hexA(col, 0));
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(p.x, p.y, Rpx, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = hexA(col, 0.38 * alpha);
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.arc(p.x, p.y, Rpx, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawLink(it) {
  const a = it.a, b = it.b, l = it.l;
  const zf = clamp(0.30 + (Math.max(a.z, b.z) + 1) * 0.45, 0.18, 1.0); // глубина
  const cBoost = present ? 1.35 : 1.0;
  let col, dash = [];
  if (l.inh) {
    // тормозные — малиновые пунктиром
    col = hexA('#ff3b5c', clamp((0.30 + 0.55 * l.w01) * zf * cBoost, 0.12, 0.95));
    dash = [5, 4];
  } else {
    // возбуждающие — изумрудный градиент по |весу|
    col = mixHex('#0e6e42', '#2bd97a', l.w01, clamp((0.22 + 0.62 * l.w01) * zf * cBoost, 0.1, 0.95));
  }
  ctx.strokeStyle = col;
  ctx.lineWidth = (0.6 + 1.4 * l.w01) * (0.6 + 0.4 * zf) * (present ? 1.3 : 1.0);
  ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawNeuron(d) {
  const nn = d.nn, p = d.p;
  const reg = cur.regions[nn.region] || { kind: 'ghost' };
  const col = KIND_COLOR[reg.kind] || '#8fa3c0';
  const act = clamp(nn.act / cur.maxAct, 0, 1);
  const zf = clamp(0.5 + (p.z + 1) * 0.5, 0.35, 1.2);
  const r = clamp((0.007 + 0.022 * act) * ptScale * p.s * zf, 1.1, 46);
  const alpha = nn.ghost
    ? (present ? 0.45 : 0.30)
    : clamp(0.42 + 0.58 * act, 0.42, 1.0);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.shadowColor = col;
  ctx.shadowBlur = lowGlow ? 0 : 6 + 16 * act;
  ctx.fillStyle = col;
  ctx.beginPath();
  ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function roundRectPath(x, y, w, h, rad) {
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

function drawLabel(it) {
  const reg = cur.regions[it.i];
  const c = it.c;
  const col = KIND_COLOR[reg.kind] || '#8fa3c0';
  const zf = clamp(0.55 + (c.z + 1) * 0.45, 0.4, 1.25);
  const fs = clamp((present ? 16 : 12.5) * zf * (0.85 + c.s * 0.3), 9, present ? 44 : 34);
  const txt = reg.label + ' · ' + cur.counts[it.i];
  ctx.font = '600 ' + fs.toFixed(1) + 'px "Segoe UI", system-ui, sans-serif';
  const tw = ctx.measureText(txt).width;
  const px = 7, py = 4;
  const bx = c.x - tw / 2 - px, by = c.y - fs / 2 - py;
  ctx.fillStyle = 'rgba(7,11,20,0.74)';
  roundRectPath(bx, by, tw + px * 2, fs + py * 2, 6);
  ctx.fill();
  ctx.strokeStyle = hexA(col, reg.ghost ? 0.4 : 0.8);
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = reg.ghost ? hexA(col, 0.85) : col;
  ctx.fillText(txt, c.x, c.y + fs * 0.33);
}

// чип зоны: жирная подпись (две строки: имя + тег модели)
function drawZoneChip(c, alpha) {
  const z = c.z, p = c.p;
  const zf = clamp(0.55 + (p.z + 1) * 0.45, 0.4, 1.25);
  const fs = clamp((present ? 17 : 13.5) * zf * (0.85 + p.s * 0.3), 9, present ? 48 : 38);
  const fs2 = Math.max(8.5, fs * 0.6);
  ctx.font = '700 ' + fs.toFixed(1) + 'px "Segoe UI", system-ui, sans-serif';
  const w1 = ctx.measureText(z.label).width;
  ctx.font = '600 ' + fs2.toFixed(1) + 'px "Segoe UI", system-ui, sans-serif';
  const w2 = ctx.measureText(z.tag).width;
  const w = Math.max(w1, w2);
  const px = 9, py = 5;
  const bx = p.x - w / 2 - px, by = p.y - fs / 2 - py;
  const bh = fs + fs2 + py * 2 + 3;
  ctx.fillStyle = 'rgba(7,11,20,0.78)';
  roundRectPath(bx, by, w + px * 2, bh, 8);
  ctx.fill();
  ctx.strokeStyle = hexA(z.color, clamp(alpha, 0.2, 1));
  ctx.lineWidth = 1.4;
  ctx.stroke();
  ctx.textAlign = 'center';
  ctx.fillStyle = z.color;
  ctx.font = '700 ' + fs.toFixed(1) + 'px "Segoe UI", system-ui, sans-serif';
  ctx.fillText(z.label, p.x, by + fs * 0.82);
  ctx.fillStyle = 'rgba(220,234,252,0.78)';
  ctx.font = '600 ' + fs2.toFixed(1) + 'px "Segoe UI", system-ui, sans-serif';
  ctx.fillText(z.tag, p.x, by + fs + fs2 * 0.8 + 5);
  ctx.textAlign = 'left';
}

// ── холст и события ──
function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = $('stage').getBoundingClientRect();
  W = rect.width; H = rect.height;
  cv.width = Math.max(1, Math.round(W * dpr));
  cv.height = Math.max(1, Math.round(H * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  dirty = true;
}
window.addEventListener('resize', resize);

cv.addEventListener('pointerdown', (e) => {
  dragging = true; autospin = false; dirty = true;
  lastX = e.clientX; lastY = e.clientY;
  cv.classList.add('dragging');
  cv.setPointerCapture(e.pointerId);
});
cv.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  yaw += (e.clientX - lastX) * 0.006;
  pitch = clamp(pitch + (e.clientY - lastY) * 0.006, -1.45, 1.45);
  lastX = e.clientX; lastY = e.clientY;
  dirty = true;
});
function endDrag() { dragging = false; cv.classList.remove('dragging'); }
cv.addEventListener('pointerup', endDrag);
cv.addEventListener('pointercancel', endDrag);
cv.addEventListener('wheel', (e) => {
  e.preventDefault();
  dist = clamp(dist * Math.exp(e.deltaY * 0.0012), 1.15, 9);
  dirty = true;
}, { passive: false });
cv.addEventListener('dblclick', () => {
  yaw = DEF.yaw; pitch = DEF.pitch; dist = DEF.dist;
  autospin = true; paused = false; dirty = true;
});
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { e.preventDefault(); paused = !paused; dirty = true; }
});

// ── вкладки ──
document.querySelectorAll('.tab').forEach((b) => {
  b.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    tab = b.dataset.tab;
    setData(tab === 'current' ? BRAIN_CURRENT : tab === 'stageb' ? BRAIN_STAGEB : BRAIN_ZONES, false);
    autospin = true; paused = false; dirty = true;
  });
});

// ── сайдбар: зоны, фильтры и оформление ──
$('f-zones').addEventListener('change', (e) => {
  showZones = e.target.checked;
  dirty = true;
});
$('b-zone-all').addEventListener('click', () => {
  zoneVis = zoneVis.map(() => true);
  syncZoneChecks(); dirty = true;
});
$('b-zone-none').addEventListener('click', () => {
  zoneVis = zoneVis.map(() => false);
  syncZoneChecks(); dirty = true;
});
function syncZoneChecks() {
  const cbs = document.querySelectorAll('#zonelist input[type=checkbox]');
  zoneVis.forEach((v, i) => { if (cbs[i]) cbs[i].checked = v; });
}
$('f-ghost').addEventListener('change', (e) => {
  showGhosts = e.target.checked;
  cur.regions.forEach((r, i) => { if (r.ghost) regionVis[i] = showGhosts; });
  dirty = true;
});
$('f-labels').addEventListener('change', (e) => { showLabels = e.target.checked; dirty = true; });
$('f-links').addEventListener('change', (e) => { showLinks = e.target.checked; dirty = true; });
$('f-present').addEventListener('change', (e) => {
  present = e.target.checked;
  document.body.classList.toggle('present', present);
  dirty = true;
});
$('s-wt').addEventListener('input', (e) => {
  wtFrac = (+e.target.value) / 100;
  $('wt-val').textContent = e.target.value + '%';
  dirty = true;
});
$('s-ps').addEventListener('input', (e) => {
  ptScale = (+e.target.value) / 100;
  $('ps-val').textContent = e.target.value + '%';
  dirty = true;
});
$('b-sensor').addEventListener('click', () => setKindOnly('sensor'));
$('b-motor').addEventListener('click', () => setKindOnly('motor'));
$('b-all').addEventListener('click', () => {
  regionVis = regionVis.map((_, i) => !cur.regions[i].ghost || showGhosts);
  syncChecks(); dirty = true;
});
function setKindOnly(kind) {
  regionVis = cur.regions.map((r) => r.kind === kind && (!r.ghost || showGhosts));
  syncChecks(); dirty = true;
}
function syncChecks() {
  const cbs = document.querySelectorAll('#reglist input[type=checkbox]');
  regionVis.forEach((v, i) => { if (cbs[i]) cbs[i].checked = v; });
}

// ── легенда: классы регионов, связи, призраки + сводка моделей зон ──
function buildLegend() {
  const lg = $('legend');
  lg.textContent = '';
  const rows = [
    ['sensor', 'Сенсоры (запах, стены)'],
    ['motor', 'Моторы (руль)'],
    ['inter', 'Тормозные интернейроны'],
    ['assoc', 'Ассоциативный слой'],
  ];
  for (const row of rows) {
    const d = document.createElement('div'); d.className = 'lrow';
    const s = document.createElement('span'); s.className = 'sw dot';
    s.style.background = KIND_COLOR[row[0]];
    const t = document.createElement('span'); t.textContent = row[1];
    d.append(s, t); lg.appendChild(d);
  }
  const g1 = document.createElement('div'); g1.className = 'lrow';
  g1.innerHTML = '<span class="sw" style="background:linear-gradient(90deg,#0e6e42,#2bd97a)"></span><span>Возбуждающие связи (ярче = вес)</span>';
  lg.appendChild(g1);
  const g2 = document.createElement('div'); g2.className = 'lrow';
  g2.innerHTML = '<span class="sw dash"></span><span>Тормозные связи</span>';
  lg.appendChild(g2);
  const g3 = document.createElement('div'); g3.className = 'lrow';
  g3.innerHTML = '<span class="sw dot" style="background:rgba(143,163,192,.5)"></span><span>Плановые регионы (транш 3)</span>';
  lg.appendChild(g3);
  // сводка моделей зон: что умеет / как учится — по одному предложению
  const sep = document.createElement('div'); sep.className = 'lsep';
  sep.textContent = 'Модели зон (гибридный мозг)';
  lg.appendChild(sep);
  const MODEL_ROWS = [
    ['#41d7ff', 'LIF', 'спайковая кора: сенсоры, моторы, рефлексы, ассоциации — STDP + дофамин онлайн'],
    ['#ffa940', 'Wilson–Cowan', 'гладкая rate-динамика выбора действия (планирование, премотор) — параметры из генома'],
    ['#b388ff', 'LSTM/GRU', 'тренируемая память (гиппокамп) — градиент офлайн, веса запекаются в геном'],
    ['#2bd97a', 'ANN', 'зрительный энкодер: признаки → токи — градиент офлайн, веса запекаются'],
    ['#ff7eb6', 'Гибрид', 'лимбика: кровь → тоны + модуляция пластичности (регуляторный геном)'],
  ];
  for (const row of MODEL_ROWS) {
    const d = document.createElement('div'); d.className = 'lrow';
    const s = document.createElement('span'); s.className = 'sw dot';
    s.style.background = row[0];
    const t = document.createElement('span'); t.textContent = row[1] + ' — ' + row[2];
    d.append(s, t); lg.appendChild(d);
  }
}

// ── цикл отрисовки: RAF, перерисовка только при изменениях ──
let fpsTimes = [];
function loop(now) {
  requestAnimationFrame(loop);
  if (autospin && !paused) { yaw += 0.0038; dirty = true; }
  if (dirty) {
    render();
    dirty = false;
    const dt = now - lastT;
    lastT = now;
    fpsTimes.push(dt);
    if (fpsTimes.length > 30) fpsTimes.shift();
    const avg = fpsTimes.reduce((a, b) => a + b, 0) / fpsTimes.length;
    // адаптивное качество: на слабых машинах снимаем shadowBlur
    if (avg > 26) lowGlow = true;
    else if (avg < 14) lowGlow = false;
  }
}

// ── старт ──
function init() {
  resize();
  buildLegend();
  document.querySelectorAll('.tab').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === INITIAL_TAB));
  setData(tab === 'current' ? BRAIN_CURRENT : tab === 'stageb' ? BRAIN_STAGEB : BRAIN_ZONES, false);
  $('wt-val').textContent = '2%';
  $('ps-val').textContent = '100%';
  requestAnimationFrame(loop);
}
init();
"##;
