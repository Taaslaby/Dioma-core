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

    let mut eco = Ecosystem::new(8, 2026_0824, true);
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
    println!("\n  ══ ПРИСПОСОБЛЕННОСТЬ ПО КОЛЕНИЯМ ══");
    println!("  (умершие: еда на тысячу тиков жизни — чистый навык кормёжки)\n");
    let born_at = |id: usize| eco.born_ticks.iter().find(|&&(b, _)| b == id).map(|&(_, t)| t);
    let mut by_gen: Vec<(usize, Vec<(usize, usize)>)> = Vec::new(); // (gen, [(meals, lifespan)])
    for &(id, gen, dt) in &eco.deaths {
        let meals = eco
            .meals_log
            .iter()
            .find(|&&(mid, _)| mid == id)
            .map(|&(_, m)| m)
            .unwrap_or(0);
        let bt = born_at(id).unwrap_or(0);
        match by_gen.iter_mut().find(|(g, _)| *g == gen) {
            Some((_, v)) => v.push((meals, dt.saturating_sub(bt))),
            None => by_gen.push((gen, vec![(meals, dt.saturating_sub(bt))])),
        }
    }
    by_gen.sort_by_key(|&(g, _)| g);
    for (gen, lives) in by_gen.iter() {
        let n = lives.len();
        let mean_life = lives.iter().map(|l| l.1).sum::<usize>() as f64 / n as f64;
        // Эффективность: сумма еды / сумма прожитых тысяч тиков.
        let total_meals: usize = lives.iter().map(|l| l.0).sum();
        let total_kticks = (lives.iter().map(|l| l.1).sum::<usize>() as f64 / 1000.0).max(1.0);
        let eff = total_meals as f64 / total_kticks;
        println!(
            "    поколение {:>2}: {:>3} умерших · ср.жизнь {:>6.0} т · эффективность {:>5.2} еды/кт {}",
            gen,
            n,
            mean_life,
            eff,
            bar(eff.round() as usize)
        );
    }

    // Живые на конец прогона.
    if !eco.creatures.is_empty() {
        println!("\n  живы на финише:");
        for c in eco.creatures.iter().take(8) {
            let age_kt = (eco.tick - c.born_tick) as f64 / 1000.0;
            let eff = c.meals as f64 / age_kt.max(0.1);
            println!(
                "    id={:<3} поколение {} · еды {} · возраст {:.0} кт · {:.2} еды/кт",
                c.id, c.generation, c.meals, age_kt, eff
            );
        }
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
