//! -------------------------------------------------------------------------
//! Filename:    src/bin/poc_ecosystem.rs
//! Class:       SemantikaEngine
//! Purpose:     poc-ecosystem — ИГРОВАЯ ПЕТЛЯ: N существ · еда · смерть ·
//!              размножение · поколения
//! Description: 3 существа стартуют, живут 200 с. Сытые рядом спариваются →
//!              ребёнок наследует мозги. Голодный умирает. Печатаем
//!              демографию по фазам + родословную.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::ecosystem::Ecosystem;
use dioma::organ_life::SIM_TICKS;

fn main() {
    println!("poc-ecosystem: 3 существа · общий мир · смерть · размножение");
    let mut eco = Ecosystem::new(3, 0x0000_AE11, true);

    // Фазовая статистика каждые 1000 тиков (25 с).
    let mut births_total = 0usize;
    let mut deaths_total = 0usize;

    for tick in 1..=SIM_TICKS {
        let ate = eco.step_tick();

        if tick % 200 == 0 && !eco.creatures.is_empty() {
            let c = &eco.creatures[0];
            println!(
                "  t{tick:>5} c0: H={:.0} G={:.0} meals={}",
                c.sim.blood.conc[dioma::metabolism::chem::HUNGER],
                c.sim.blood.conc[dioma::metabolism::chem::GLUCOSE],
                c.meals,
            );
        }

        if tick % 1000 == 0 {
            let alive = eco.alive_count();
            let gen_max = eco.creatures.iter().map(|c| c.generation).max().unwrap_or(0);
            let meals: usize = eco.creatures.iter().map(|c| c.meals).sum();
            let avg_glucose: f32 = if eco.creatures.is_empty() { 0.0 } else {
                eco.creatures.iter()
                    .map(|c| c.sim.blood.conc[dioma::metabolism::chem::GLUCOSE])
                    .sum::<f32>() / eco.creatures.len() as f32
            };
            println!(
                "t{:>5} ({:>4} с): живых={:<3} поколение={:<2} рождений={} смертей={} еды_за_фазу={} глюкоза_ср={:.0}",
                tick, tick / 4, alive, gen_max,
                eco.births.len() - births_total,
                eco.deaths.len() - deaths_total,
                ate,
                avg_glucose,
            );
            births_total = eco.births.len();
            deaths_total = eco.deaths.len();
        }
    }

    // Родословная.
    println!("\n═══ РОДОСЛОВНАЯ ══");
    for &(gen, id, parents) in eco.births.iter().take(15) {
        println!("  поколение {}: существо {} ← родители {} & {}", gen, id, parents[0], parents[1]);
    }
    if eco.births.len() > 15 {
        println!("  ... и ещё {} рождений", eco.births.len() - 15);
    }

    println!("\n═══ СМЕРТНОСТЬ ══");
    for &(id, gen, tick) in eco.deaths.iter().take(10) {
        println!("  существо {} (поколение {}) умерло на тике {}", id, gen, tick);
    }
    if eco.deaths.len() > 10 {
        println!("  ... и ещё {} смертей", eco.deaths.len() - 10);
    }

    let alive_final = eco.alive_count();
    let gen_final = eco.creatures.iter().map(|c| c.generation).max().unwrap_or(0);
    println!(
        "\nИТОГ: живых {alive_final} · рождений {} · смертей {} · макс поколение {gen_final}",
        eco.births.len(), eco.deaths.len()
    );

    if alive_final > 0 && gen_final > 0 {
        println!("ЭКОСИСТЕМА ЖИВЁТ И ЭВОЛЮЦИОНИРУЕТ ✓");
    } else if alive_final > 0 {
        println!("выжили но не эволюционировали — нужно больше времени или существ");
    } else {
        println!("все умерли — нужен баланс еды/метаболизма");
    }
}
