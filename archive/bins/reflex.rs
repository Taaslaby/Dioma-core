//! -------------------------------------------------------------------------
//! Filename:    src/bin/reflex.rs
//! Class:       SemantikaEngine
//! Purpose:     poc-reflex: grown withdrawal reflex + dopamine plasticity
//! Description: protocol A/B/C/D: baseline → idle → reward pairs → final; control D=0.
//!
//! author: Atlasov anton
//! -------------------------------------------------------------------------
use dioma::brain::{gains_from, Brain, PlasticityParams, DT_MS};
use dioma::grow::{grow, reflex_layout, Steering};

const STIM_AMNT: f32 = 3.0;
const STIM_STEPS: usize = 40;
const REST_STEPS: usize = 100;
const RESP_STEPS: usize = 300;
const N_TRIALS: usize = 50;
const REG_S: usize = 0;
const REG_I: usize = 1;
const REG_M: usize = 2;

fn main() {
    println!("poc-reflex: выращенный рефлекс + дофаминовая пластичность");
    let t0 = std::time::Instant::now();
    // nt уже в Grown от layout (grow ставит знак по inhibitory-флагу).
    let grown = grow(&reflex_layout(), &Steering::default(), 0xC0_FF_EE);
    println!("рост: {} синапсов за {:.2} с", grown.edges.len(), grown.growth_s);

    // Гейны: S→M сильный моносинаптический путь, S→I, I→M.
    let gains = gains_from(
        3,
        &[(0, 2, 0.45), (0, 1, 0.25), (0, 0, 0.12), (1, 2, 0.6)],
    );

    run_protocol(&grown, &gains, 1.0, "ЭКСПЕРИМЕНТ");
    run_protocol(&grown, &gains, 0.0, "КОНТРОЛЬ");
    println!("\nполное время: {:.2} с", t0.elapsed().as_secs_f64());
}

fn run_protocol(grown: &dioma::Grown, gains: &[Vec<f32>], dopamine: f32, tag: &str) {
    println!("\n═══ {tag} (дофамин = {dopamine}) ═══");
    let mut brain = Brain::new(grown, gains.to_vec(), PlasticityParams::default());
    let (w0, c_sm) = brain.mean_path_weight(REG_S, REG_M);
    let (_, c_si) = brain.mean_path_weight(REG_S, REG_I);
    let (_, c_im) = brain.mean_path_weight(REG_I, REG_M);
    println!("пути: S→M ×{c_sm} (w̄ = {w_sm:.3}), S→I ×{c_si}, I→M ×{c_im}", w_sm = w0);

    // A: базовый рефлекс.
    let (lat_a, w_a) = measure(&mut brain);
    println!("A  базовый рефлекс      латентность = {lat_a:6.1} мс   w(S→M) = {w_a:.4}");

    // B: спонтанная активность без награды.
    let ext = vec![0.0f32; brain.n];
    for _ in 0..2000 {
        brain.step(&ext);
    }
    let (lat_b, w_b) = measure(&mut brain);
    let _ = lat_b;
    println!("B  после 1 с без доф.   латентность = {lat_b:6.1} мс   w(S→M) = {w_b:.4}");

    // C: обучение.
    let mut lat_sum = 0.0f32;
    for k in 0..N_TRIALS {
        let lat = learning_trial(&mut brain, dopamine);
        lat_sum += lat.max(0.0);
        if (k + 1) % 25 == 0 {
            let (w, _) = brain.mean_path_weight(REG_S, REG_M);
            println!("   ... проба {:+>4}: w(S→M) = {w:.4}", k + 1);
        }
    }
    println!("C  средняя латентность за пробы: {:.1} мс", lat_sum / N_TRIALS as f32);

    // D: финал.
    let (lat_d, w_d) = measure(&mut brain);
    println!("D  финальный рефлекс    латентность = {lat_d:6.1} мс   w(S→M) = {w_d:.4}");
    let dw = (w_d - w_a) / w_a.max(1e-9) * 100.0;
    println!(
        "ИТОГ {tag}: латентность {lat_a:.1} → {lat_d:.1} мс, w(S→M) {w_a:.4} → {w_d:.4} ({dw:+.1}%), рефлекс {}",
        if lat_d >= 0.0 { "жив" } else { "ПОТЕРЯН" }
    );
}

/// Стимул «боль» в S + окно отклика M: (латентность мс, w(S→M)).
fn measure(brain: &mut Brain) -> (f32, f32) {
    let n = brain.n;
    let (s0, s1) = brain.region_bounds(REG_S);
    let (m0, m1) = brain.region_bounds(REG_M); // только моторы
    let ext0 = vec![0.0f32; n];
    for _ in 0..REST_STEPS {
        brain.step(&ext0);
    }
    let mut ext = ext0.clone();
    let mut latency: Option<usize> = None;
    for t in 0..(STIM_STEPS + RESP_STEPS) {
        let stim_on = t < STIM_STEPS;
        let stim = if stim_on { STIM_AMNT } else { 0.0 };
        for cell in ext[s0..s1].iter_mut() {
            *cell = stim;
        }
        let spikes = brain.step(&ext);
        let m_rate: i32 = (m0..m1).map(|i| spikes[i]).sum();
        if m_rate > 0 && latency.is_none() {
            latency = Some(t);
        }
    }
    let (w, _) = brain.mean_path_weight(REG_S, REG_M);
    (latency.map(|t| t as f32 * DT_MS).unwrap_or(-1.0), w)
}

fn learning_trial(brain: &mut Brain, dopamine: f32) -> f32 {
    let (lat, _) = measure(brain);
    // Задержка награды: eligibility-след S→M ещё жив (τ_e = 1 c).
    let ext = vec![0.0f32; brain.n];
    for _ in 0..100 {
        brain.step(&ext);
    }
    brain.reward_now(dopamine);
    lat
}
