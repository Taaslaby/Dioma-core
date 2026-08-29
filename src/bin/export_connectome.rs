//! Экспортёр коннектома для Unity-порта dioma (com.dioma.spikes).
//!
//! Растит мозг из сида и пишет пару JSON: `<prefix>_connectome.json`
//! (grown-проводка + регионы + параметры LIF/STDP + гейны анатомии) и
//! `<prefix>_golden.json` («золотой след»: сценарий сенсоров + по-тиковые
//! эталоны для теста паритета C# ↔ Rust).
//!
//! Два мозга:
//! - `agent`   — agent_layout, 96 нейронов, 5 регионов (SL/SR/I/ML/MR);
//! - `stage_c` — stage_c_lite_layout, 760 нейронов, 11 регионов (+WL/WR,
//!   VL/VR, рекуррентная кора CORTEX_E/I), гейны как в `Creature::assemble`.
//!
//! Маршрутизация сенсоров — по тегам регионов: SL/SR ← запах, WL/WR ← стена
//! (если региона WL нет — стена идёт в SL/SR, как в упрощённом контракте).
//! Упрощённый контракт существа (без крови/гиппокампа): sensor_gain = 1,
//! еда → `schedule_reward(1.0)`, удар о стену → `reward_now(-0.5)`.
//!
//! Запуск: `cargo run --release --bin export_connectome -- <dir> [agent|stage_c|both]`
//! (dir = dioma-spikes/Tests/Editor/Data).

use std::fs;
use std::path::PathBuf;

use dioma_engine::brain::dynamics::neurons::{gains_from, Brain, PlasticityParams};
use dioma_engine::brain::growth::genome::Grown;
use dioma_engine::brain::growth::grow::{agent_layout, grow, stage_c_lite_layout, Layout, Steering};

const DT_MS: f32 = 0.5;
const STEPS_PER_TICK: usize = 100;
const N_TICKS: usize = 240;
const SEED: u64 = 42;

struct TickSense {
    smell_l: f32,
    smell_r: f32,
    wall_l: f32,
    wall_r: f32,
    eat: bool,
    wall_hit: bool,
}

/// Детерминированный сценарий «золотого следа»: еда слева→справа, удары о
/// стену, события еды. Пишется в JSON целиком — обе стороны видят один ввод.
fn scripted_sense(t: usize) -> TickSense {
    let left_phase = (t % 80) < 40;
    let mut s = TickSense {
        smell_l: if left_phase { 0.8 } else { 0.15 },
        smell_r: if left_phase { 0.15 } else { 0.8 },
        wall_l: 0.0,
        wall_r: 0.0,
        eat: false,
        wall_hit: false,
    };
    match t {
        60 | 100 | 140 | 180 => s.eat = true,
        70 => {
            s.wall_l = 0.7;
            s.wall_hit = true;
        }
        150 => {
            s.wall_r = 0.7;
            s.wall_hit = true;
        }
        _ => {}
    }
    s
}

fn f32s(v: &[f32]) -> String {
    v.iter()
        .map(|x| format!("{}", x))
        .collect::<Vec<_>>()
        .join(",")
}

struct Export {
    prefix: &'static str,
    layout: Layout,
    n_regions: usize,
    gains: Vec<Vec<f32>>,
    params: PlasticityParams,
    /// Регионы-ПРЕ, исключённые из гомеостаза (кора stage-C).
    homeo_exclude_pre: Vec<usize>,
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let what = std::env::args().nth(2).unwrap_or_else(|| "both".into());
    fs::create_dir_all(&out_dir).expect("out dir");

    // ── agent: 96 нейронов, гейны ядра рефлексов (SL→SL/SR→SR выпадают) ──
    let agent = Export {
        prefix: "dioma",
        layout: agent_layout(),
        n_regions: 5,
        gains: gains_from(
            5,
            &[(0, 3, 1.0), (1, 4, 1.0), (0, 2, 0.8), (1, 2, 0.8), (2, 3, 1.0), (2, 4, 1.0)],
        ),
        params: PlasticityParams::default(),
        homeo_exclude_pre: vec![],
    };

    // ── stage_c: 760 нейронов, гейны из Creature::assemble_impl ──
    let stage_c = Export {
        prefix: "dioma_c",
        layout: stage_c_lite_layout(),
        n_regions: 11,
        gains: gains_from(
            11,
            &[
                // Ядро рефлексов (avoidance): 0=SL 1=SR 2=WL 3=WR 4=I 5=ML 6=MR
                (0, 5, 1.0),
                (1, 6, 1.0),
                (0, 4, 0.8),
                (1, 4, 0.8),
                (2, 6, 2.0), // WALL_AVOID_GAIN
                (3, 5, 2.0),
                (4, 5, 1.0),
                (4, 6, 1.0),
                // Зрение → кора (9=CORTEX_E, 10=CORTEX_I), прямых VL/VR→мотор нет
                (7, 9, 1.0),
                (8, 9, 1.0),
                // Кора: рекуррентность 400 сом → 0.02, E/I-контур, слабый выход
                (9, 9, 0.02),
                (9, 10, 1.0),
                (10, 9, 1.0),
                (9, 5, 0.1),
                (9, 6, 0.1),
                (10, 5, 0.1),
                (10, 6, 0.1),
                (9, 4, 0.2),
                (10, 4, 0.2),
            ],
        ),
        params: PlasticityParams {
            w_max: 64.0, // PLASTICITY_W_MAX
            // Исходящие рёбра коры вне мягкого гомеостаза ядра рефлексов
            // (как в Creature::assemble_impl). Тот же список едет в JSON —
            // иначе эталон и C# считают гомеостаз по-разному.
            homeo_exclude_pre: vec![9, 10],
            ..Default::default()
        },
        homeo_exclude_pre: vec![9, 10],
    };

    match what.as_str() {
        "agent" => export(&out_dir, &agent),
        "stage_c" => export(&out_dir, &stage_c),
        _ => {
            export(&out_dir, &agent);
            export(&out_dir, &stage_c);
        }
    }
}

fn export(out_dir: &PathBuf, ex: &Export) {
    let steering = Steering::default();
    let grown: Grown = grow(&ex.layout, &steering, SEED);
    let n = grown.nt.len();
    println!(
        "[{}] grown: {} нейронов, {} синапсов, {} регионов",
        ex.prefix,
        n,
        grown.edges.len(),
        ex.n_regions
    );

    // ── JSON коннектома ──
    let regions_json: String = ex
        .layout
        .regions
        .iter()
        .enumerate()
        .map(|(ri, r)| {
            let (start, end) = grown.region_bounds(ri);
            format!(
                "{{\"tag\":\"{}\",\"start\":{},\"end\":{},\"inhibitory\":{}}}",
                r.tag,
                start,
                end,
                if r.inhibitory { "true" } else { "false" }
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let gains_json: String = ex
        .gains
        .iter()
        .map(|row| format!("[{}]", f32s(row)))
        .collect::<Vec<_>>()
        .join(",");
    let edges_json: String = grown
        .edges
        .iter()
        .map(|&(pre, post, w)| format!("[{},{},{}]", pre, post, w))
        .collect::<Vec<_>>()
        .join(",");
    let excl_json = ex
        .homeo_exclude_pre
        .iter()
        .map(|r| r.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let p = &ex.params;
    let connectome_json = format!(
        "{{\n \"n\":{n}, \"dt\":{dt}, \"stepsPerTick\":{spt},\n \"lif\":{{\"tauRef\":2.2,\"tauM\":20.0,\"vRest\":-52.0,\"vThresh\":-45.0,\"vReset\":-52.0,\"tauSyn\":3.0,\"tDelay\":1.8}},\n \"stdp\":{{\"tauPre\":20.0,\"tauPost\":20.0,\"tauE\":1000.0,\"aLtp\":{altp},\"aLtd\":{altd},\"lr\":{lr},\"wMin\":0.0,\"wMax\":{wmax}}},\n \"plasticity\":{{\"homeoRate\":{hr},\"rewardDelayTicks\":{rdt},\"homeoEveryTicks\":0,\"homeoExcludePre\":[{excl}]}},\n \"seed\":{seed},\n \"regions\":[{regions}],\n \"gains\":[{gains}],\n \"nt\":[{nt}],\n \"edges\":[{edges}]\n}}\n",
        n = n,
        dt = DT_MS,
        spt = STEPS_PER_TICK,
        altp = p.a_ltp,
        altd = p.a_ltd,
        lr = p.lr,
        wmax = p.w_max,
        hr = p.homeo_rate,
        rdt = p.reward_delay_ticks,
        seed = SEED,
        regions = regions_json,
        gains = gains_json,
        nt = f32s(&grown.nt),
        edges = edges_json,
        excl = excl_json,
    );
    fs::write(out_dir.join(format!("{}_connectome.json", ex.prefix)), &connectome_json)
        .expect("write connectome");

    // ── Золотой след ──
    let mut brain = Brain::new(&grown, ex.gains.clone(), ex.params.clone());
    // Маршрутизация сенсоров по тегам (как в C# GoldenRunner).
    let bounds = |tag: &str| -> Option<(usize, usize)> {
        ex.layout
            .regions
            .iter()
            .position(|r| r.tag == tag)
            .map(|ri| brain.region_bounds(ri))
    };
    let (sl, sr) = (bounds("SL").unwrap(), bounds("SR").unwrap());
    let wl = bounds("WL").unwrap_or(sl); // нет WL — стена в SL (упрощённый контракт)
    let wr = bounds("WR").unwrap_or(sr);
    let (ml, mr) = (bounds("ML").unwrap(), bounds("MR").unwrap());

    const BASAL_EXT: f32 = 0.12;
    const SMELL_CURRENT: f32 = 4.0;
    const WALL_CURRENT: f32 = 4.0;
    const STEER_SCALE: f32 = 0.25;
    const EAT_DOPAMINE: f32 = 1.0;
    const WALL_DOPAMINE: f32 = -0.5;

    let mut ext = vec![BASAL_EXT; n];
    let mut hist: Vec<Vec<i32>> = Vec::new();
    let mut rows = Vec::with_capacity(N_TICKS);
    let t_start = std::time::Instant::now();

    for t in 0..N_TICKS {
        let s = scripted_sense(t);
        for e in ext.iter_mut() {
            *e = BASAL_EXT;
        }
        for e in &mut ext[sl.0..sl.1] {
            *e += s.smell_l * SMELL_CURRENT;
        }
        for e in &mut ext[sr.0..sr.1] {
            *e += s.smell_r * SMELL_CURRENT;
        }
        for e in &mut ext[wl.0..wl.1] {
            *e += s.wall_l * WALL_CURRENT;
        }
        for e in &mut ext[wr.0..wr.1] {
            *e += s.wall_r * WALL_CURRENT;
        }

        let (ml_sum, mr_sum) = brain.step_batch_motor(&ext, STEPS_PER_TICK, ml, mr, &mut hist);
        if s.eat {
            brain.schedule_reward(EAT_DOPAMINE);
        }
        if s.wall_hit {
            brain.reward_now(WALL_DOPAMINE);
        }

        let steer = (((mr_sum - ml_sum) as f32) * STEER_SCALE).clamp(-1.0, 1.0);
        let total_spikes: i64 = hist
            .iter()
            .map(|v| v.iter().map(|&x| x as i64).sum::<i64>())
            .sum();
        let wsum: f64 = brain.csr_values().iter().map(|&w| w as f64).sum();

        rows.push(format!(
            "{{\"sl\":{},\"sr\":{},\"wl\":{},\"wr\":{},\"eat\":{},\"wall\":{},\"ml\":{},\"mr\":{},\"steer\":{},\"spikes\":{},\"wsum\":{}}}",
            s.smell_l, s.smell_r, s.wall_l, s.wall_r, s.eat, s.wall_hit,
            ml_sum, mr_sum, steer, total_spikes, wsum
        ));
    }
    println!(
        "[{}] след: {} тиков за {:.1}с ({:.2} мс/тик Rust)",
        ex.prefix,
        N_TICKS,
        t_start.elapsed().as_secs_f32(),
        t_start.elapsed().as_secs_f32() * 1000.0 / N_TICKS as f32
    );

    let golden_json = format!("{{\"ticks\":[\n{}\n]}}\n", rows.join(",\n"));
    fs::write(out_dir.join(format!("{}_golden.json", ex.prefix)), &golden_json).expect("write golden");
}
