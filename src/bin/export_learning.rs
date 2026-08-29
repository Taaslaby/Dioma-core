//! Экспортёр протокола обучения: ящик Скиннера (оперантное R-STDP).
//!
//! Зачем: паритет проверяет, что порт СОВПАДАЕТ с эталоном, но не доказывает,
//! что мозг УЧИТСЯ. Арена шумна (2/4/2/7 ягод по окнам — из этого ничего не
//! следует), а скиннер даёт проверяемое утверждение: награждаемый путь
//! усиливается, ненаграждаемые остаются на месте.
//!
//! Протокол: 4 сенсора × 4 мотора, диагональ «сенсор i → мотор i». Стимулы
//! идут по кругу, но награда даётся ТОЛЬКО за сенсор 0. Между пробами —
//! холостые тики: eligibility (tau_e = 1000 мс ≈ 20 тиков) должен затухать,
//! иначе награда за пробу N кредитует и предыдущие пути. Это и есть
//! избирательность: без зазора все четыре веса растут одинаково
//! (замерено: 4.700 у всех), с зазором 60 тиков — только награждаемый.
//!
//! Запуск: `cargo run --release --bin export_learning -- <dir>`
//! (dir = dioma-engine/Tests/Editor/Data).

use std::fs;
use std::path::PathBuf;

use dioma_engine::brain::dynamics::neurons::{Brain, PlasticityParams};
use dioma_engine::brain::growth::genome::Grown;

const N_SENSORS: usize = 4;
const N_MOTORS: usize = 4;
const N_TOTAL: usize = N_SENSORS + N_MOTORS;
const N_TRIALS: usize = 80;
const SEED: u64 = 42;

/// Токи (мВ на подшаг: внешний ток идёт прямо в мембрану).
const BASAL: f32 = 0.12;
const STIMULUS: f32 = 8.0;
const REWARD: f32 = 1.0;
/// Начальный вес диагонали: подобран так, чтобы новорождённый уже давал
/// 1–2 спайка (иначе награждать нечего — LTP требует спайка поста).
const W0: f32 = 0.8;
/// Холостых тиков между пробами (затухание eligibility).
const GAP_TICKS: usize = 60;
/// Тиков стимула в пробе.
const TRIAL_TICKS: usize = 1;

/// Коннектом протокола: регион 0 — сенсоры, регион 1 — моторы, диагональ.
fn skinner_grown() -> Grown {
    let mut edges = Vec::new();
    for s in 0..N_SENSORS {
        edges.push((s, N_SENSORS + s, W0));
    }
    let mut region_of = vec![0usize; N_SENSORS];
    region_of.extend(vec![1usize; N_MOTORS]);
    Grown {
        edges,
        region_of,
        nt: vec![1.0f32; N_TOTAL],
        growth_s: 0.0,
        rule_genes: Vec::new(),
        instincts: Vec::new(),
    }
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&out_dir).expect("out dir");

    let grown = skinner_grown();
    let gains = vec![vec![0.0f32, 1.0], vec![0.0f32, 0.0]];
    let params = PlasticityParams {
        w_max: 6.0,
        lr: 0.3, // выше рабочего: протокол должен сойтись за 80 проб
        ..Default::default()
    };

    // ── Коннектом в JSON (формат общий с export_connectome) ──
    let edges_json: String = grown
        .edges
        .iter()
        .map(|&(pre, post, w)| format!("[{},{},{}]", pre, post, w))
        .collect::<Vec<_>>()
        .join(",");
    let nt_json = grown
        .nt
        .iter()
        .map(|x| format!("{}", x))
        .collect::<Vec<_>>()
        .join(",");
    let connectome_json = format!(
        "{{\n \"n\":{n}, \"dt\":0.5, \"stepsPerTick\":100,\n \"lif\":{{\"tauRef\":2.2,\"tauM\":20.0,\"vRest\":-52.0,\"vThresh\":-45.0,\"vReset\":-52.0,\"tauSyn\":3.0,\"tDelay\":1.8}},\n \"stdp\":{{\"tauPre\":20.0,\"tauPost\":20.0,\"tauE\":1000.0,\"aLtp\":{altp},\"aLtd\":{altd},\"lr\":{lr},\"wMin\":0.0,\"wMax\":{wmax}}},\n \"plasticity\":{{\"homeoRate\":{hr},\"rewardDelayTicks\":{rdt},\"homeoEveryTicks\":0,\"homeoExcludePre\":[]}},\n \"seed\":{seed},\n \"regions\":[{{\"tag\":\"S\",\"start\":0,\"end\":{ns},\"inhibitory\":false}},{{\"tag\":\"M\",\"start\":{ns},\"end\":{n},\"inhibitory\":false}}],\n \"gains\":[[0,1],[0,0]],\n \"nt\":[{nt}],\n \"edges\":[{edges}]\n}}\n",
        n = N_TOTAL,
        ns = N_SENSORS,
        altp = params.a_ltp,
        altd = params.a_ltd,
        lr = params.lr,
        wmax = params.w_max,
        hr = params.homeo_rate,
        rdt = params.reward_delay_ticks,
        seed = SEED,
        nt = nt_json,
        edges = edges_json,
    );
    fs::write(out_dir.join("dioma_skinner_connectome.json"), &connectome_json)
        .expect("write skinner connectome");

    // ── Протокол ──
    let mut brain = Brain::new(&grown, gains, params);
    let spt = 100usize;
    let mut hist: Vec<Vec<i32>> = Vec::new();
    let quiet = vec![BASAL; N_TOTAL];
    let mut rows = Vec::with_capacity(N_TRIALS);
    let w_start = brain.csr_values();
    println!(
        "старт: w=[{:.3} {:.3} {:.3} {:.3}], награда только за сенсор 0, зазор {} тиков",
        w_start[0], w_start[1], w_start[2], w_start[3], GAP_TICKS
    );

    for trial in 0..N_TRIALS {
        let target = trial % N_SENSORS;
        let mut ext = vec![BASAL; N_TOTAL];
        ext[target] += STIMULUS;

        let mut fired = 0i32;
        for _ in 0..TRIAL_TICKS {
            let (ml, _) = brain.step_batch_motor(
                &ext,
                spt,
                (N_SENSORS + target, N_SENSORS + target + 1),
                (N_SENSORS, N_SENSORS + 1),
                &mut hist,
            );
            fired += ml;
        }
        let rewarded = target == 0 && fired > 0;
        if rewarded {
            brain.reward_now(REWARD);
        }
        // Зазор: eligibility затухает, иначе следующая награда кредитует
        // и этот путь тоже (без зазора избирательности нет вообще).
        for _ in 0..GAP_TICKS {
            brain.step_batch(&quiet, spt, &mut hist);
        }

        let v = brain.csr_values();
        let wsum: f64 = v.iter().map(|&w| w as f64).sum();
        rows.push(format!(
            "{{\"target\":{},\"fired\":{},\"rewarded\":{},\"w\":[{},{},{},{}],\"wsum\":{}}}",
            target, fired, rewarded, v[0], v[1], v[2], v[3], wsum
        ));

        if trial % 20 == 0 || trial + 1 == N_TRIALS {
            println!(
                "проба {:>3} (сенсор {}): спайков={:<2} w=[{:.3} {:.3} {:.3} {:.3}]",
                trial + 1, target, fired, v[0], v[1], v[2], v[3]
            );
        }
    }

    let v = brain.csr_values();
    let others = (v[1] + v[2] + v[3]) / 3.0;
    println!(
        "\nитог: награждаемый путь {:.3} → {:.3}, ненаграждаемые в среднем {:.3}; \
         избирательность ×{:.2}",
        W0, v[0], others, v[0] / others
    );

    let golden = format!(
        "{{\"basal\":{},\"stimulus\":{},\"reward\":{},\"gapTicks\":{},\"trialTicks\":{},\
         \"w0\":{},\"trials\":[\n{}\n]}}\n",
        BASAL, STIMULUS, REWARD, GAP_TICKS, TRIAL_TICKS, W0, rows.join(",\n")
    );
    let path = out_dir.join("dioma_skinner_golden.json");
    fs::write(&path, &golden).expect("write skinner golden");
    println!("след обучения: {} ({} проб)", path.display(), N_TRIALS);
}
