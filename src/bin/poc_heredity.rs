//! -------------------------------------------------------------------------
//! Filename:    src/bin/poc_heredity.rs
//! Class:       SemantikaEngine
//! Purpose:     poc-heredity — наследование выученного мозга
//! Description: родитель живёт 4000 тиков с разумом → сохраняется в .taas →
//!              дети наследуют (CSR+W2 ± мутация). Метрика: тиков до первой
//!              еды у детей наследников vs новорождённых.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::heritage::{save_creature, spawn_child, ticks_to_first_meal};
use rayon::prelude::*;
use dioma::organ_life::{OrganSim, SIM_TICKS};
use taas_format::TaasReader;

fn main() {
    // Основной поток Windows имеет маленький стек (1 МБ) — рост миров и
    // клоны OrganSim гоняем в отдельном жирном потоке.
    let worker = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run)
        .expect("worker thread");
    worker.join().expect("worker panicked");
}

fn run() {
    let path = "target/creature_parent.taas";

    // ── Родитель живёт полную жизнь с разумом ──
    println!("родитель: жизнь {} тиков...", SIM_TICKS);
    let mut parent = OrganSim::new(false, 0x0000_AE11).with_mind(0x00_0A_11);
    let mut eaten = 0usize;
    for _ in 0..SIM_TICKS {
        if sim_step(&mut parent) {
            eaten += 1;
        }
    }
    let (w_end, _) = parent.cortex_path_weight(0, 3);
    println!("родитель: съедено={eaten} w(SL→ML)={w_end:.3}");

    save_creature(path, &parent).expect("save");
    println!("сохранён: {path}");

    // Раундтрип: файл читается, W2 бит-в-бит совпадает с родительским.
    let r = TaasReader::from_file(path).expect("read back");
    let w2_file = r.get_f32("w2").expect("w2 in file");
    let w2_live = parent.mind.as_ref().unwrap().weight_heat();
    assert_eq!(w2_file.len(), w2_live.len(), "размер W2 в файле");
    let diff: f32 = w2_file
        .iter()
        .zip(w2_live.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff < 1e-4, "W2 раундтрип разошёлся: {diff}");
    println!(
        "раундтрип .taas ✓ ({} весов, csr {} синапсов)",
        w2_file.len(),
        r.get_f32("cortex_csr").map(|v| v.len()).unwrap_or(0)
    );

    // ── Дети наследников против новорождённых — параллельно ──
    const N: usize = 5;
    const SHORT: usize = 800;

    let t0 = std::time::Instant::now();
    let heir_results: Vec<(usize, usize)> = (0..N)
        .into_par_iter()
        .map(|k| {
            let seed = 0x1000 + k as u64;
            let grown = dioma::arena::grow_agent(&dioma::arena::Genome::wildtype(seed));
            let mut child = spawn_child(&parent, seed, true);
            let t = ticks_to_first_meal(&mut child);
            let mut eaten_k = 0usize;
            for _ in 0..SHORT.saturating_sub(t.min(SHORT)) {
                if child.step_tick().ate { eaten_k += 1; }
            }
            (t, eaten_k)
        })
        .collect();
    println!("наследники оценены за {:.1} с", t0.elapsed().as_secs_f64());

    let t1 = std::time::Instant::now();
    let newborn_results: Vec<(usize, usize)> = (0..N)
        .into_par_iter()
        .map(|k| {
            let seed = 0x2000 + k as u64;
            let mut baby = OrganSim::new(false, seed).with_mind(seed);
            let mut eaten_k = 0usize;
            let t = ticks_to_first_meal(&mut baby);
            for _ in 0..SHORT.saturating_sub(t.min(SHORT)) {
                if baby.step_tick().ate { eaten_k += 1; }
            }
            (t, eaten_k)
        })
        .collect();
    println!("новорождённые оценены за {:.1} с", t1.elapsed().as_secs_f64());

    let heir_ticks: Vec<usize> = heir_results.iter().map(|(t, _)| *t).collect();
    let heir_eaten: Vec<usize> = heir_results.iter().map(|(_, e)| *e).collect();
    let newborn_ticks: Vec<usize> = newborn_results.iter().map(|(t, _)| *t).collect();
    let newborn_eaten: Vec<usize> = newborn_results.iter().map(|(_, e)| *e).collect();

    let mean = |v: &[usize]| -> f64 {
        v.iter().filter(|&&x| x != usize::MAX).map(|&x| x as f64).sum::<f64>()
            / v.iter().filter(|&&x| x != usize::MAX).count().max(1) as f64
    };
    let sum = |v: &[usize]| v.iter().sum::<usize>();

    println!(
        "наследники ({}): первая еда ср.={:.0} тиков, съедено за {} тиков всего={}",
        N,
        mean(&heir_ticks),
        SHORT,
        sum(&heir_eaten)
    );
    println!(
        "новорождённые ({}): первая еда ср.={:.0} тиков, съедено всего={}",
        N,
        mean(&newborn_ticks),
        sum(&newborn_eaten)
    );
}

/// Один шаг жизни без разбора наблюдения.
fn sim_step(sim: &mut OrganSim) -> bool {
    sim.step_tick().ate
}
