//! -------------------------------------------------------------------------
//! Filename:    src/bin/sardana_battery.rs
//! Class:       SemantikaEngine
//! Purpose:     полная проверка сардана-мозгов: обе модели · обучение ·
//!              сравнение с grown
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::brain::{Brain, PlasticityParams};
use dioma::heritage::sardna_to_grown;
use taas_format::TaasReader;

/// Загрузить и протестировать одну модель.
fn test_model(name: &str, path: &str) {
    println!("\n══ {name} ══");
    let grown = match sardna_to_grown(path) {
        Ok(g) => g,
        Err(e) => { println!("  ОШИБКА загрузки: {e}"); return; }
    };

    // Статистика.
    let n = grown.region_of.len();
    let n_edges = grown.edges.len();
    println!("  нейронов={n} рёбер={n_edges}");

    // Тест 1: активность при тонике.
    let gains = vec![vec![0.6f32; 4]; 4];
    let mut brain = Brain::new(&grown, gains.clone(), PlasticityParams {
        w_max: 3.0,
        ..Default::default()
    });
    let drive = vec![3.0f32; n];
    let mut spikes_tonic = 0i64;
    for _ in 0..2000 {
        let s = brain.step(&drive);
        spikes_tonic += s.iter().map(|&x| x as i64).sum::<i64>();
    }
    println!("  тоник: {:.1} сп/тик", spikes_tonic as f64 / 2000.0);

    // Тест 2: обучение — награда за активность → растёт ли активность?
    let mut spikes_before = 0i64;
    for _ in 0..500 {
        let s = brain.step(&drive);
        spikes_before += s.iter().map(|&x| x as i64).sum::<i64>();
    }

    for _ in 0..50 {
        brain.step(&drive);
        brain.reward_now(1.0);
    }

    let mut spikes_after = 0i64;
    for _ in 0..500 {
        let s = brain.step(&drive);
        spikes_after += s.iter().map(|&x| x as i64).sum::<i64>();
    }
    println!(
        "  обучение: до={:.1} после={:.1} ({:+.1}%)",
        spikes_before as f64 / 500.0,
        spikes_after as f64 / 500.0,
        (spikes_after as f64 / spikes_before.max(1) as f64 - 1.0) * 100.0,
    );

    // Тест 3: тишина — затухает ли без входа?
    let zero = vec![0.0f32; n];
    let mut silence = 0i64;
    for _ in 0..500 {
        let s = brain.step(&zero);
        silence += s.iter().map(|&x| x as i64).sum::<i64>();
    }
    println!("  тишина: {silence} спайков за 500 тиков — {}",
        if silence == 0 { "затихла ✓" } else if silence < 100 { "почти затихла" } else { "самоподдержание!" });
}

/// Grown agent для сравнения.
fn test_grown() {
    use dioma::arena::{grow_agent, Genome};
    use dioma::arena::arena_gains;

    let genome = Genome::wildtype(42);
    let grown = grow_agent(&genome);
    let n = grown.region_of.len();

    let mut brain = Brain::new(&grown, arena_gains(&genome), PlasticityParams::default());
    let drive = vec![3.0f32; n];

    let mut spikes_tonic = 0i64;
    for _ in 0..2000 {
        let s = brain.step(&drive);
        spikes_tonic += s.iter().map(|&x| x as i64).sum::<i64>();
    }
    println!("  тоник: {:.1} сп/тик", spikes_tonic as f64 / 2000.0);

    let mut before = 0i64;
    for _ in 0..500 {
        let s = brain.step(&drive);
        before += s.iter().map(|&x| x as i64).sum::<i64>();
    }
    for _ in 0..50 {
        brain.step(&drive);
        brain.reward_now(1.0);
    }
    let mut after = 0i64;
    for _ in 0..500 {
        let s = brain.step(&drive);
        after += s.iter().map(|&x| x as i64).sum::<i64>();
    }
    println!(
        "  обучение: до={:.1} после={:.1} ({:+.1}%)",
        before as f64 / 500.0,
        after as f64 / 500.0,
        (after as f64 / before.max(1) as f64 - 1.0) * 100.0,
    );
}

fn main() {
    println!("БАТАРЕЯ САРДААНА vs GROWN");
    println!("══════════════════════════");

    print!("\ngrown_agent:");
    test_grown();

    test_model("SarDNA_Pro", "D:/Проекты/asm_engines/sardaanaCM/data/generated/SarDNA_Pro_best.taas");
    test_model("Diff_snn", "D:/Проекты/asm_engines/sardaanaCM/data/generated/Diff_snn_best.taas");
}
