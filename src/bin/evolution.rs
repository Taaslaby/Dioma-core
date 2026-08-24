//! -------------------------------------------------------------------------
//! Filename:    src/bin/evolution.rs
//! Class:       SemantikaEngine
//! Purpose:     многопоколенческая эволюция органных существ
//! Description: каждое поколение — N жизней по SHORT тиков. Отбираются
//!              лучшие (по еде), скрещиваются → следующее поколение.
//!              Метрики: средняя еда, балл экзамена, разброс характеров.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::world::exam::run_exam;
use dioma::heritage::spawn_child;
use dioma::body::life::OrganSim;

const GENERATIONS: usize = 10;
const POP_SIZE: usize = 6;
const LIFE_TICKS: usize = 2000; // 50 с на жизнь

struct Evaluated {
    seed: u64,
    parent: OrganSim,
    meals: usize,
    #[allow(dead_code)]
    ticks_median: usize,
}

/// Жизнь одного существа с подсчётом еды и медианой времени до еды.
fn live_and_evaluate(seed: u64, mind_seed: u64) -> (OrganSim, usize, usize) {
    let mut sim = if mind_seed > 0 {
        OrganSim::new(false, seed).with_mind(mind_seed)
    } else {
        OrganSim::new(false, seed)
    };
    let mut eaten = 0usize;
    for _ in 0..LIFE_TICKS {
        if sim.step_tick().ate {
            eaten += 1;
        }
        if sim.blood.organ.damage >= 1.0 {
            break;
        }
    }
    // Медиана тиков до еды ≈ 600/eaten если ел, MAX если нет.
    let median = if eaten > 0 { LIFE_TICKS / eaten.max(1) } else { LIFE_TICKS };
    (sim, eaten, median)
}

fn main() {
    let worker = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run)
        .expect("evolution worker");
    worker.join().expect("evolution panicked");
}

fn run() {
    println!(
        "эволюция: {GENERATIONS} поколений × {POP_SIZE} существ × {LIFE_TICKS} тиков"
    );
    println!("═══════════════════════════════════════════════");

    let mut population: Vec<Evaluated> = Vec::new();
    // Поколение 0 — случайные сиды.
    for i in 0..POP_SIZE {
        let seed = 0x0000_AE11 + i as u64 * 7919;
        let (sim, eaten, med) = live_and_evaluate(seed, seed ^ 0x5EED);
        population.push(Evaluated { seed, parent: sim, meals: eaten, ticks_median: med });
    }

    for gen in 0..GENERATIONS {
        // Сортируем по еде (убывание).
        population.sort_by_key(|p| std::cmp::Reverse(p.meals));

        let total_meals: usize = population.iter().map(|p| p.meals).sum();
        let best_meals = population[0].meals;
        let mean_meals = total_meals as f64 / POP_SIZE as f64;
        let rest_pct = {
            let m = &population[0].parent;
            if let Some(mind) = &m.mind {
                let rates = mind.action_rates();
                (rates[4] * 100.0).min(100.0)
            } else { 0.0 }
        };

        println!(
            "поколение {:>2}: еда лучш={} ср={:.1} · сидел%={:.0}",
            gen, best_meals, mean_meals, rest_pct,
        );

        if gen + 1 >= GENERATIONS {
            break;
        }

        // Селекция: топ-2 → дети.
        let parents: Vec<&Evaluated> = population.iter().take(2).collect();

        // Скрещивание: дети от двух лучших, каждый со своим сидом.
        let mut next_gen = Vec::with_capacity(POP_SIZE);
        for k in 0..POP_SIZE {
            let child_seed = 0xE70 + gen as u64 * 1000 + k as u64;
            // Чётные — от родителя A, нечётные — от B, мутируя.
            let donor = parents[k % parents.len()];
            let _ = donor; // spawn_child берёт из parent напрямую
            next_gen.push((child_seed, donor.seed));
        }

        // Пересоздаём популяцию из детей наследников.
        let mut new_pop = Vec::with_capacity(POP_SIZE);
        for (k, &(child_seed, _parent_seed)) in next_gen.iter().enumerate() {
            // Берём мозг лучшего родителя как базу.
            let base = &population[if k % 3 == 0 { 1 } else { 0 }];
            let mut child_sim = spawn_child(&base.parent, child_seed, true);

            // Живём и оцениваем.
            let mut eaten = 0usize;
            for _ in 0..LIFE_TICKS {
                if child_sim.step_tick().ate { eaten += 1; }
                if child_sim.blood.organ.damage >= 1.0 { break; }
            }
            let median = if eaten > 0 { LIFE_TICKS / eaten.max(1) } else { LIFE_TICKS };

            new_pop.push(Evaluated {
                seed: child_seed,
                parent: child_sim,
                meals: eaten,
                ticks_median: median,
            });
        }
        population = new_pop;

        // Экзамен каждые 3 поколения на лучшем.
        if (gen + 1) % 3 == 0 || gen + 1 == GENERATIONS {
            let cfg = dioma::world::exam::ExamConfig {
                name: "",
                seed: population[0].seed,
                with_mind: true,
                sensory_noise: 0.0,
                episodes: 4,
            };
            let report = run_exam(&cfg);
            println!("  ЭКЗАМЕН: балл={:.1} поиск={} эфф={:.2} шум={:.2}",
                report.score(), report.seek_median_ticks,
                report.path_efficiency, report.noise_robustness);
        }
    }

    // Финальный экзамен лучшего.
    println!("\n═══ ФИНАЛЬНЫЙ ЭКЗАМЕН ══");
    let cfg = dioma::world::exam::ExamConfig {
        name: "лучший",
        seed: population[0].seed,
        with_mind: true,
        sensory_noise: 0.0,
        episodes: 8,
    };
    let report = run_exam(&cfg);
    report.print("лучшее существо");
    println!("БАЛЛ: {:.1}/100", report.score());
}
