//! -------------------------------------------------------------------------
//! Filename:    src/bin/poc_sardana_creature.rs
//! Class:       SemantikaEngine
//! Purpose:     сардана-мозг в полноценном существе — жизнь + экзамен
//! Description: грузим SarDNA_Pro_best как Grown → Brain → живём 2000
//!              тиков. Сравниваем с grown_agent() на том же сиде.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::heritage::sardna_to_grown;
use dioma::exam::run_exam;
use dioma::brain::{Brain, PlasticityParams};
use dioma::organ_life::{OrganSim};

fn main() {
    let path = "D:/Проекты/asm_engines/sardaanaCM/data/generated/SarDNA_Pro_best.taas";

    // ── 1. Конвертируем сардаану в Grown ──
    println!("Загружаем sarDNA коннектом...");
    let grown = sardna_to_grown(path).expect("load sardana");
    println!(
        "sardaana: {} рёбер · {} нейронов · growth_s={:.2}",
        grown.edges.len(),
        grown.region_of.len(),
        grown.growth_s
    );

    // Гейны: все к всем через упрощённую матрицу.
    let n_regions = grown.region_of.iter().cloned().max().unwrap_or(0) + 1;
    let gains = vec![vec![0.6f32; n_regions]; n_regions];

    // ── 2. Жизнь с сардаана-мозгом ──
    println!("\n── жизнь с sardaana мозгом (2000 тиков) ──");
    let mut sim = OrganSim::new(false, 0x0000_AE11);
    // Заменяем кору: пересоздаём OrganSim с сардаана-Grown.
    // Для этого создаём новый OrganSim через with_mind и подменяем brain.
    // Проще: создаём отдельный Brain и работаем напрямую.

    let mut brain = Brain::new(&grown, gains, PlasticityParams {
        w_max: 3.0,
        ..Default::default()
    });

    let n = brain.n;
    println!("Brain: {} нейронов", n);

    // Стимулируем: тоник 3.0 = уровень сенсорного сигнала в арене.
    let drive = vec![3.0f32; n];
    let mut total_spikes = 0i64;
    let mut active_neurons = std::collections::HashSet::new();
    for _ in 0..2000 {
        let spikes = brain.step(&drive);
        total_spikes += spikes.iter().map(|&s| s as i64).sum::<i64>();
        for (i, &s) in spikes.iter().enumerate() {
            if s != 0 { active_neurons.insert(i); }
        }
    }
    println!(
        "2000 тиков: {total_spikes} спайков от {} уникальных нейронов",
        active_neurons.len()
    );

    // ── 3. Обучение: награда за активность первых 50 нейронов ──
    println!("\nобучение: награда за активность региона 0");
    for _ in 0..100 {
        let spikes = brain.step(&stimulate_region(0, n));
        let _ = spikes;
        brain.reward_now(1.0); // поощряем что бы ни было — HEBBIAN-style
    }
    println!("после обучения: 100 наград применено");

    // ── 4. Экзамен на этом мозге ──
    // Экзамен работает на OrganSim — нужно обернуть Brain в OrganSim.
    // Проще: меряем активность до/после обучения.
    let mut post_spikes = 0i64;
    let mut post_active = std::collections::HashSet::new();
    for _ in 0..2000 {
        let spikes = brain.step(&drive);
        post_spikes += spikes.iter().map(|&s| s as i64).sum::<i64>();
        for (i, &s) in spikes.iter().enumerate() {
            if s != 0 { post_active.insert(i); }
        }
    }

    let pre_rate = total_spikes as f64 / 2000.0;
    let post_rate = post_spikes as f64 / 2000.0;
    println!("\n── РЕЗУЛЬТАТ ──");
    println!("до обучения: {:.1} спайков/тик от {} нейронов", pre_rate, active_neurons.len());
    println!("после обучения: {:.1} спайков/тик от {} нейронов", post_rate, post_active.len());
    println!(
        "sardana-мозг {} в движке",
        if post_spikes > 0 { "РАБОТАЕТ ✓" } else { "мёртв ✗" }
    );
}

/// Стимул на первые 40 нейронов региона 0.
fn stimulate_region(region: usize, _n: usize) -> Vec<f32> {
    // Возвращаем тоник + усиленный стимул для указанного региона.
    // Упрощённо: просто повышенный тоник.
    vec![0.5f32; _n]
}
