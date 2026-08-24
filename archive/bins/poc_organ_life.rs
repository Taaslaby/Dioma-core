//! -------------------------------------------------------------------------
//! Filename:    src/bin/poc_organ_life.rs
//! Class:       SemantikaEngine
//! Purpose:     poc-organ-life — вертикальный срез: органный мозг в свободной
//!              жизни (голод→мотивация, еда→дофамин, стена→рефлекс)
//! Description: 100 с жизни; метрики органов и сравнение с роботом.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::metabolism::chem;
use dioma::organ_life::{live_organ_life, OrganSim};

fn main() {
    println!("poc-organ-life: HH спина + HAAP мотивация + LIF кора в свободной жизни");

    let mut sim = OrganSim::new(false, 0x0000_AE11);
    let mut eaten = 0usize;
    let mut wall_hits = 0usize;
    let mut reflex_fires = 0i64;
    let mut mid_hungry = 0f64;
    let mut mid_sated = 0f64;
    let mut n_hungry = 0usize;
    let mut n_sated = 0usize;

    for _ in 0..dioma::organ_life::SIM_TICKS {
        let obs = sim.step_tick();
        if !obs.alive {
            break;
        }
        if obs.ate {
            eaten += 1;
        }
        if obs.wall_hit {
            wall_hits += 1;
        }
        reflex_fires += obs.spine_m_spikes;
        if sim.blood.conc[chem::HUNGER] > 40.0 {
            mid_hungry += obs.midbrain_spikes as f64;
            n_hungry += 1;
        } else {
            mid_sated += obs.midbrain_spikes as f64;
            n_sated += 1;
        }
    }
    let (w0, _) = sim.cortex_path_weight(0, 3);
    println!(
        "химия: еда={eaten} удары={wall_hits} рефлекс-тики={reflex_fires}"
    );
    println!(
        "средний мозг сп/тик: голодный={:.1} (n={n_hungry})  сытый={:.1} (n={n_sated})",
        mid_hungry / n_hungry.max(1) as f64,
        mid_sated / n_sated.max(1) as f64,
    );
    println!("кора после жизни: w(SL→ML) = {w0:.3}");

    let mut robot = OrganSim::new(true, 0x0000_AE11);
    let rr = live_organ_life(&mut robot);
    println!(
        "робот-контроль: еда={} удары={} рефлекс-тики={}",
        rr.eaten, rr.wall_hits, rr.reflex_fires
    );
}
