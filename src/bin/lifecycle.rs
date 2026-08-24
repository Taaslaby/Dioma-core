//! -------------------------------------------------------------------------
//! Filename:    src/bin/lifecycle.rs
//! Class:       SemantikaEngine
//! Purpose:     ПОЛНЫЙ ЖИЗНЕННЫЙ ЦИКЛ: рождение→сон→жизнь→еда→потомство
//! Description: все механизмы включены одновременно:
//!   • основатели получают врождённые инстинкты (прожиг родильным сном);
//!   • каждые SLEEP_EVERY тиков существо спит SLEEP_FOR тиков — снимок
//!     памяти на входе, возврат к нему во сне;
//!   • еда = фазический дофамин (обучение в жизни);
//!   • размножение = spawn_child_crossover: кора+политика+гены правил+
//!     инстинкты наследуются, ребёнок рождается во сне.
//! Метрики по поколениям: рождено/дожило, еды на душу, срок жизни.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::world::ecosystem::{Ecosystem, ECO_TICKS};

const REPORT_WINDOW: usize = 4000;

fn main() {
    let total_ticks: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(ECO_TICKS * 3);

    println!("╔═══════════════════════════════════════════════╗");
    println!("║  ЖИЗНЕННЫЙ ЦИКЛ · инстинкты·сон·наследственность ║");
    println!("╚═══════════════════════════════════════════════╝");

    let mut eco = Ecosystem::new(6, 2026_0824, true);
    println!("основателей: 6 · инстинктов у каждого: {} · цель: {total_ticks} тиков\n",
        eco.creatures[0].sim.instincts.len());
    println!(
        "  окно │ жив. │ макс.покол │ рожд │ умер │ сна │ еда/душу │ ср.жизнь"
    );

    for tick in 1..=total_ticks {
        eco.step_tick();

        if tick % REPORT_WINDOW == 0 {
            let alive = &eco.creatures;
            let max_gen = alive.iter().map(|c| c.generation).max().unwrap_or(0);
            let meals_per_head = if alive.is_empty() {
                0.0
            } else {
                alive.iter().map(|c| c.meals).sum::<usize>() as f64 / alive.len() as f64
            };
            // Средняя продолжительность жизни умерших в этом окне.
            let lo = tick - REPORT_WINDOW;
            let lifespans: Vec<usize> = eco
                .deaths
                .iter()
                .filter(|&&(_, _, t)| t > lo && t <= tick)
                .filter_map(|&(id, _, dt)| {
                    eco.born_ticks
                        .iter()
                        .find(|&&(bid, _)| bid == id)
                        .map(|&(_, bt)| dt.saturating_sub(bt))
                })
                .collect();
            let mean_life = if lifespans.is_empty() {
                0.0
            } else {
                lifespans.iter().sum::<usize>() as f64 / lifespans.len() as f64
            };
            println!(
                " {:>5} │ {:>4} │ {:>10} │ {:>4} │ {:>4} │ {:>3} │ {:>8.1} │ {:>8}",
                tick,
                alive.len(),
                max_gen,
                eco.births.len(),
                eco.deaths.len(),
                eco.sleeps_done / 1000, // кило-снов за окно суммарно? нет: всего/1000
                meals_per_head,
                format!("{mean_life:.0} т"),
            );
        }
    }

    // ── Итог по поколениям ──
    println!("\n  ══ ЕДА НА ДУШУ ПО КОЛЕНИЯМ (умершие) ══");
    let mut by_gen: Vec<(usize, Vec<usize>)> = Vec::new();
    for &(id, gen, _) in &eco.deaths {
        let meals = eco
            .meals_log
            .iter()
            .find(|&&(mid, _)| mid == id)
            .map(|&(_, m)| m)
            .unwrap_or(0);
        match by_gen.iter_mut().find(|(g, _)| *g == gen) {
            Some((_, v)) => v.push(meals),
            None => by_gen.push((gen, vec![meals])),
        }
    }
    by_gen.sort_by_key(|&(g, _)| g);
    for (gen, meals) in by_gen.iter().take(12) {
        let mean = meals.iter().sum::<usize>() as f64 / meals.len() as f64;
        println!(
            "    поколение {:>2}: {:>3} умерших · средняя еда {:>5.1} {}",
            gen,
            meals.len(),
            mean,
            bar(mean as usize)
        );
    }
}

fn bar(n: usize) -> String {
    let n = n.min(30);
    let mut s = String::new();
    for _ in 0..n {
        s.push('█');
    }
    s
}
