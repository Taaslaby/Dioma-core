//! -------------------------------------------------------------------------
//! Filename:    src/bin/sardana_vs_grown.rs
//! Class:       SemantikaEngine
//! Purpose:     сардана vs grown — КТО БОЛЕЕ ЖИВОЙ?
//! Description: оба мозга проходят одинаковые фазы: покой → стимул →
//!              обучение → отдых → повторный стимул. Меряем активность,
//!              адаптацию, обучение и самоподдержание.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::brain::{Brain, PlasticityParams};
use dioma::heritage::sardna_to_grown;

/// Статистика одной фазы.
struct Phase {
    spikes_per_tick: f64,
    unique_neurons: usize,
}

fn run_phase(
    brain: &mut Brain,
    n: usize,
    ext_level: f32,
    ticks: usize,
) -> Phase {
    let ext = vec![ext_level; n];
    let mut total = 0i64;
    let mut active = std::collections::HashSet::new();
    for _ in 0..ticks {
        let s = brain.step(&ext);
        for (i, &v) in s.iter().enumerate() {
            if v != 0 {
                total += 1;
                active.insert(i);
            }
        }
    }
    Phase {
        spikes_per_tick: total as f64 / ticks as f64,
        unique_neurons: active.len(),
    }
}

fn compare(name: &str, mut brain: Brain, n: usize) {
    println!("\n══ {name} ({n} нейронов) ══");

    // Фаза A: покой (низкий тоник).
    let a = run_phase(&mut brain, n, 0.5, 500);
    println!("покой:     {:.1} сп/тик · {} нейронов", a.spikes_per_tick, a.unique_neurons);

    // Фаза B: стимуляция (высокий сигнал).
    let b = run_phase(&mut brain, n, 5.0, 500);
    println!("стимул:    {:.1} сп/тик · {} нейронов", b.spikes_per_tick, b.unique_neurons);

    // Фаза C: обучение с наградой.
    let drive = vec![5.0f32; n];
    for _ in 0..50 {
        brain.step(&drive);
        brain.reward_now(1.0);
    }

    // Фаза D: снова покой после обучения.
    let d = run_phase(&mut brain, n, 0.5, 500);
    println!("покой-после:{:.1} сп/тик · {} нейронов", d.spikes_per_tick, d.unique_neurons);

    // Фаза E: повторный стимул.
    let e = run_phase(&mut brain, n, 5.0, 500);
    println!("повторный: {:.1} сп/тик · {} нейронов", e.spikes_per_tick, e.unique_neurons);

    // Метрики живости.
    let arousal = if a.spikes_per_tick > 0.01 { b.spikes_per_tick / a.spikes_per_tick } else { f64::INFINITY };
    let learning = if d.spikes_per_tick > 0.01 { e.spikes_per_tick / d.spikes_per_tick } else { 0.0 };
    let silence_recovery = if a.spikes_per_tick > 0.01 { d.spikes_per_tick / a.spikes_per_tick } else { 1.0 };

    println!("\n  возбуждаемость (стимул/покой): {:.2}×", arousal);
    println!("  обучаемость (повтор/первый):   {:.2}×", learning);
    println!("  восстановление после отдыха:   {:.2}×", silence_recovery);

    let alive_score =
        (arousal.clamp(0.0, 10.0) / 10.0 * 30.0)
        + (learning.clamp(0.0, 3.0) / 3.0 * 30.0)
        + (silence_recovery.clamp(0.0, 2.0) / 2.0 * 20.0)
        + if b.unique_neurons > n / 4 { 20.0 } else { b.unique_neurons as f64 / (n / 4) as f64 * 20.0 };
    println!("  ЖИВОСТЬ: {:.0}/100", alive_score);
}

fn main() {
    println!("САРДААНА vs GROWN — КТО БОЛЕЕ ЖИВОЙ?");

    // ── Grown agent ──
    println!("\n███████████████████████████████████");
    let genome = dioma::arena::Genome::wildtype(42);
    let grown = dioma::arena::grow_agent(&genome);
    let gains = dioma::arena::arena_gains(&genome);
    let brain = Brain::new(&grown, gains, PlasticityParams::default());
    compare("Grown Agent (LIF)", brain, grown.region_of.len());

    // ── SarDNA Pro ──
    println!("\n███████████████████████████████████");
    match sardna_to_grown("D:/Проекты/asm_engines/sardaanaCM/data/generated/SarDNA_Pro_best.taas") {
        Ok(grown_s) => {
            let gains = vec![vec![0.6f32; 4]; 4];
            let brain = Brain::new(&grown_s, gains, PlasticityParams {
                w_max: 3.0,
                ..Default::default()
            });
            compare("SarDNA Pro (трансформер)", brain, grown_s.region_of.len());
        }
        Err(e) => println!("sardana load error: {e}"),
    }

    // ── Diff_snn ──
    println!("\n███████████████████████████████████");
    match sardna_to_grown("D:/Проекты/asm_engines/sardaanaCM/data/generated/Diff_snn_best.taas") {
        Ok(grown_d) => {
            let gains = vec![vec![0.6f32; 4]; 4];
            let brain = Brain::new(&grown_d, gains, PlasticityParams {
                w_max: 3.0,
                ..Default::default()
            });
            compare("Diff SNN (диффузия)", brain, grown_d.region_of.len());
        }
        Err(e) => println!("diff load error: {e}"),
    }
}
