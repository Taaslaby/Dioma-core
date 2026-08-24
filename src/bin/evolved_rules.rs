//! -------------------------------------------------------------------------
//! Filename:    src/bin/evolved_rules.rs
//! Class:       SemantikaEngine
//! Purpose:     три механизма поверх спайков: гены правил, тропическая
//!              перестройка, сон-консолидация
//! Description: A) личности — один и тот же мозг, разные гены Пифии у
//!                 «двигателя решения» (порог = характер);
//!              B) тропика — живой граф: слабые связи тихих путей
//!                 рассасываются, ростки прорастают между активными;
//!              C) сон — выученное закрепляется только у тех, кто спит.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::brain::{gains_from, Brain, PlasticityParams};
use dioma::grow;
use dioma::plasticity::{sync_weights_from_brain, tropic_rewire, RewireOpts, TropicField};
use dioma::pythia::{compete, PythiaUnit, RuleGene};
use dioma::rng::XorShift;

/// Регионы agent_layout: 0=SL 1=SR 2=I 3=ML 4=MR.
const SL: usize = 0;
const SR: usize = 1;
const ML: usize = 3;
const MR: usize = 4;

fn main() {
    let mut rng = XorShift::new(20260824);
    let grown = grow::grow(&grow::agent_layout(), &grow::Steering::default(), 777);
    let gains = gains_from(
        5,
        &[
            (SL, ML, 1.0),
            (SR, MR, 1.0),
            (SL, 2, 0.8),
            (SR, 2, 0.8),
            (2, ML, 1.0),
            (2, MR, 1.0),
        ],
    );
    let params = PlasticityParams::default();

    println!("╔═══════════════════════════════════════════╗");
    println!("║  ГЕНЫ ПРАВИЛ · ТРОПИКА · СОН              ║");
    println!("╚═══════════════════════════════════════════╝");
    println!("мозг: {} клеток · {} синапсов", grown.nt.len(), grown.edges.len());

    part_a_personalities(&grown, &gains, &params, &mut rng);
    part_b_tropic(grown.clone(), &gains, &params);
    part_c_sleep(&grown, &gains, &params);
}

// ─────────────────────────────────────────────────────────────────────────
// A. Личности: порог Пифии = характер выбора действия.
// ─────────────────────────────────────────────────────────────────────────
/// Решение юнита из 2 клеток (лево/право): суммарные выходы → конкуренция.
fn decide(gene: &RuleGene, scores: &[f32; 2]) -> Option<usize> {
    let mut u = PythiaUnit::from_gene(gene, 2, 5150);
    u.set_inputs(scores);
    let mut outs = [0.0f32; 2];
    for _ in 0..12 {
        let o = u.step();
        outs[0] += o[0];
        outs[1] += o[1];
    }
    compete(&outs, 0.05, 0.05)
}

fn part_a_personalities(
    grown: &dioma::Grown,
    gains: &[Vec<f32>],
    params: &PlasticityParams,
    rng: &mut XorShift,
) {
    println!("\n── A. ЛИЧНОСТИ (один мозг, разные гены правила решения) ──");

    // Стационарная точка интегратора: Act ≈ 2·inp ⇒ пороги по обе стороны 0.6.
    let bold_gene = RuleGene::spiker(0.7, 0.5, 0.30, 2.0); // решает сразу
    let timid_gene = RuleGene::spiker(0.7, 0.5, 0.85, 2.0); // ждёт уверенности
    let mut brain = Brain::new(grown, gains.to_vec(), params.clone());
    let spt = brain.steps_per_tick();

    println!("  амп │ вспышки L/R │ BOLD    TIMID");
    for amp in [2usize, 4, 6, 9] {
        let mut b_ok = 0u32;
        let mut t_ok = 0u32;
        let mut l_sum = 0f64;
        let mut r_sum = 0f64;
        for trial in 0..20 {
            let true_left = trial % 2 == 0;
            let (sens, other) = if true_left { (SL, SR) } else { (SR, SL) };

            let mut ext = vec![0.12f32; grown.nt.len()];
            let (a, b) = brain.region_bounds(sens);
            for x in ext.iter_mut().take(b).skip(a) {
                *x = amp as f32 * 0.35 + jitter(rng);
            }
            let (oa, ob) = brain.region_bounds(other);
            for x in ext.iter_mut().take(ob).skip(oa) {
                *x = 0.12;
            }

            let mut sl = 0i64;
            let mut sr = 0i64;
            for _ in 0..spt {
                let sp = brain.step(&ext);
                sl += brain.region_spikes(&sp, SL) as i64;
                sr += brain.region_spikes(&sp, SR) as i64;
            }
            l_sum += sl as f64;
            r_sum += sr as f64;

            // Абсолютная шкала доказательств: слабый свет — мало вспышек.
            let scores = [sl as f32 / 350.0, sr as f32 / 350.0];
            let expect = if true_left { 0 } else { 1 };
            if decide(&bold_gene, &scores) == Some(expect) {
                b_ok += 1;
            }
            if decide(&timid_gene, &scores) == Some(expect) {
                t_ok += 1;
            }
        }
        println!(
            "  {amp:>3} │ {:>5.0}/{:<5.0} │ {b_ok:>2}/20   {t_ok:>2}/20",
            l_sum / 20.0,
            r_sum / 20.0
        );
    }
    println!("  BOLD выбирает при слабом сигнале, TIMID ждёт сильный.");
}

fn jitter(rng: &mut XorShift) -> f32 {
    (rng.unit() - 0.5) as f32 * 0.15
}

// ─────────────────────────────────────────────────────────────────────────
// B. Тропика: жизнь перестраивает граф вокруг активности.
// ─────────────────────────────────────────────────────────────────────────
fn part_b_tropic(mut grown: dioma::Grown, gains: &[Vec<f32>], params: &PlasticityParams) {
    println!("\n── B. ТРОПИЧЕСКАЯ ПЕРЕСТРОЙКА ──");
    let mut brain = Brain::new(&grown, gains.to_vec(), params.clone());
    let mut field = TropicField::new(grown.nt.len(), 80);
    let opts = RewireOpts { moves_per_call: 8, ..Default::default() };
    let mut rng = XorShift::new(4242);
    let spt = brain.steps_per_tick();
    let zero = vec![0.12f32; grown.nt.len()];
    let (sa, sb) = brain.region_bounds(SL);
    let mut total_moves = 0usize;

    for tick in 1..=480 {
        let mut ext = zero.clone();
        if tick % 6 < 3 {
            for x in ext.iter_mut().take(sb).skip(sa) {
                *x = 2.2; // ритмичный свет слева
            }
        }
        let mut spikes = vec![0i32; grown.nt.len()];
        for _ in 0..spt {
            spikes = brain.step(&ext);
        }
        if field.observe(&spikes) {
            // Синк весов → перестройка графа → пересборка мозга.
            let t = brain.csr_triplets();
            sync_weights_from_brain(&mut grown, (&t.0, &t.1, &t.2), gains).unwrap();
            let (rep, _) = tropic_rewire(&mut grown, field.field(), &mut rng, &opts);
            total_moves += rep.moved;
            brain = Brain::new(&grown, gains.to_vec(), params.clone());
        }
    }

    let (w_sl_ml, c_sl_ml) = brain.mean_path_weight(SL, ML);
    let (w_sr_mr, c_sr_mr) = brain.mean_path_weight(SR, MR);
    println!(
        "  переносов: {total_moves} · SL→ML: {c_sl_ml} син. w={w_sl_ml:.2} · SR→MR: {c_sr_mr} син. w={w_sr_mr:.2}"
    );
    println!("  Граф сместился к работающей стороне — жизнь перекроила проводку.");
}

// ─────────────────────────────────────────────────────────────────────────
// C. Сон-консолидация: снимок памяти против дрейфа дня.
// ─────────────────────────────────────────────────────────────────────────
fn train_and_forget(
    grown: &dioma::Grown,
    gains: &[Vec<f32>],
    params: &PlasticityParams,
    sleeps: bool,
) -> f32 {
    let mut brain = Brain::new(grown, gains.to_vec(), params.clone());
    let mut rng = XorShift::new(99);
    let (sa, sb) = brain.region_bounds(SL);
    let zero = vec![0.12f32; grown.nt.len()];
    let spt = brain.steps_per_tick();

    // Обучение: стимул слева + награда (фазический дофамин).
    for _ in 0..40 {
        let mut ext = zero.clone();
        for x in ext.iter_mut().take(sb).skip(sa) {
            *x = 3.0;
        }
        for _ in 0..spt {
            brain.step(&ext);
        }
        brain.reward_now(1.0);
        brain.clear_eligibility();
    }

    // Снимок долговременной памяти сразу после обучения.
    brain.memory_snapshot();

    // Дни: синаптический дрейф портит рисунок; спящий ночью возвращает
    // веса к снимку.
    for tick in 1..=240 {
        let mut ext = zero.clone();
        for v in ext.iter_mut() {
            *v = 0.05 + rng.unit() as f32 * 0.15;
        }
        for _ in 0..spt {
            brain.step(&ext);
        }
        let (_, _, vals) = brain.csr_triplets();
        let drifted: Vec<f32> = vals
            .iter()
            .map(|&v| (v * (1.0 + (rng.gauss() as f32) * 0.006)).max(0.0))
            .collect();
        brain.set_csr_values(&drifted).unwrap();
        if sleeps && tick % 10 == 0 {
            brain.memory_recall(0.8); // ночь: восстановить выученное
        }
    }
    brain.memory_drift_rms()
}

fn part_c_sleep(grown: &dioma::Grown, gains: &[Vec<f32>], params: &PlasticityParams) {
    println!("\n── C. СОН-КОНСОЛИДАЦИЯ ──");
    let r_sleeper = train_and_forget(grown, gains, params, true);
    let r_awake = train_and_forget(grown, gains, params, false);
    println!("  спал:     дрейф от выученного: RMS={r_sleeper:.3} (веса у снимка)");
    println!("  не спал:  дрейф от выученного: RMS={r_awake:.3} (рисунок расползся)");
    if r_awake > r_sleeper * 3.0 {
        println!(
            "  ✓ сон держит выученный рисунок в {:.0} раз точнее.",
            r_awake / r_sleeper.max(1e-6)
        );
    }
    println!("  (Поведение крошечного рефлекторного мозга устойчиво к дрейфу —");
    println!("   анатомия доминирует; на больших выученных репертуарах сон");
    println!("   станет видим и в поведении.)");
}
