//! Тюнер пейсмекера: кандидаты на чистой WS-популяции (без роста).
use dioma::physiology::{self, Archetype, PhysioGenes};
use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};
use pillars::zoo::HAAPModelParamIdx as P;

/// Кандидат в эндогенные осцилляторы: h-ток должен переигрывать leak+M
/// на всём подъёме от покоя до порога (нет стабильного подпорогового
/// фикспойнта), адаптация ограничивает burst.
struct Candidate {
    name: &'static str,
    g_h: f32,
    v_half_h: f32,
    k_h: f32,
    g_m: f32,
    v_thresh: f32,
    delta_u: f32,
    tau_u: f32,
}

const N: usize = 16;
const TICKS: usize = 2000;

fn main() {
    let candidates = [
        Candidate { name: "A", g_h: 2.0, v_half_h: -55.0, k_h: 8.0, g_m: 0.15, v_thresh: -47.0, delta_u: 0.02, tau_u: 100.0 },
        Candidate { name: "C", g_h: 1.6, v_half_h: -56.0, k_h: 7.0, g_m: 0.30, v_thresh: -47.0, delta_u: 0.02, tau_u: 100.0 },
        Candidate { name: "F", g_h: 1.6, v_half_h: -56.0, k_h: 7.0, g_m: 0.30, v_thresh: -47.0, delta_u: 0.06, tau_u: 400.0 },
        Candidate { name: "G", g_h: 1.2, v_half_h: -58.0, k_h: 8.0, g_m: 0.15, v_thresh: -48.0, delta_u: 0.08, tau_u: 500.0 },
        Candidate { name: "H", g_h: 1.8, v_half_h: -56.0, k_h: 8.0, g_m: 0.25, v_thresh: -48.0, delta_u: 0.06, tau_u: 400.0 },
        Candidate { name: "I", g_h: 1.4, v_half_h: -57.0, k_h: 8.0, g_m: 0.20, v_thresh: -49.0, delta_u: 0.08, tau_u: 600.0 },
        Candidate { name: "J", g_h: 1.6, v_half_h: -56.0, k_h: 7.0, g_m: 0.30, v_thresh: -47.0, delta_u: 0.075, tau_u: 500.0 },
        Candidate { name: "K", g_h: 1.6, v_half_h: -56.0, k_h: 7.0, g_m: 0.30, v_thresh: -47.0, delta_u: 0.09, tau_u: 700.0 },
        Candidate { name: "L", g_h: 1.45, v_half_h: -57.0, k_h: 7.0, g_m: 0.25, v_thresh: -48.0, delta_u: 0.08, tau_u: 600.0 },
    ];
    println!("канд  tau_h |  сп/тик  актив_тики  макс/тик  межбurst(тиков)");
    for cand in &candidates {
        for tau_h in [200.0f32, 400.0] {
            let mut col =
                physiology::haap_columns(Archetype::Pacemaker, &PhysioGenes::default());
            col[P::g_h as usize] = cand.g_h;
            col[P::v_half_h as usize] = cand.v_half_h;
            col[P::k_h as usize] = cand.k_h;
            col[P::g_m as usize] = cand.g_m;
            col[P::v_thresh as usize] = cand.v_thresh;
            col[P::delta_u as usize] = cand.delta_u;
            col[P::inv_tau_u as usize] = 1.0 / cand.tau_u;
            col[P::inv_tau_h as usize] = 1.0 / tau_h;

            let mut runner = BrainRunner::new(
                ConnectomeData::new(vec![0; N + 1], vec![], vec![], vec![1.0; N], vec![]),
                BrainConfig::haap_synaptic(0.05, 12.0),
            )
            .unwrap();
            let cols = vec![col; N];
            physiology::apply_haap_columns(&mut runner, &cols);

            // Тот же транслятор мировых токов, что в Brain.
            let tonic = vec![0.12f32 * 8.0 - 0.25; N];
            let mut per_tick = Vec::with_capacity(TICKS);
            for _ in 0..TICKS {
                let buf = runner.run(500, &tonic);
                per_tick.push(buf.iter().map(|&x| x as i64).sum::<i64>());
            }
            let total: i64 = per_tick.iter().sum();
            let active = per_tick.iter().filter(|&&x| x > 0).count();
            let max_tick = per_tick.iter().max().copied().unwrap_or(0);
            // Средний интервал между активными тиками (после разминки).
            let warm = &per_tick[200..];
            let mut gaps = Vec::new();
            let mut last_active: Option<usize> = None;
            for (i, &x) in warm.iter().enumerate() {
                if x > 0 {
                    if let Some(la) = last_active {
                        if i - la > 1 {
                            gaps.push((i - la) as f64);
                        }
                    }
                    last_active = Some(i);
                }
            }
            let gap_mean = if gaps.is_empty() {
                0.0
            } else {
                gaps.iter().sum::<f64>() / gaps.len() as f64
            };
            println!(
                "{}  {tau_h:4.0}  | {:8.1}  {:9}/{}  {max_tick:8}  {gap_mean:14.1}",
                cand.name,
                total as f64 / TICKS as f64,
                active,
                TICKS,
            );
        }
    }
}
