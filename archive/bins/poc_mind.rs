//! -------------------------------------------------------------------------
//! Filename:    src/bin/poc_mind.rs
//! Class:       SemantikaEngine
//! Purpose:     poc-mind — karmac-разум на органном теле
//! Description: та же свободная жизнь, но решает SpikingBrain (GNW→БГ):
//!              каналы тела → действие (сидеть/искать/подойти/бежать/
//!              бродить), награды от еды/стен/лихорадки. Сравнение с
//!              телом без разума.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::mind::CreatureAction;
use dioma::organ_life::{OrganSim, SIM_TICKS};

fn main() {
    println!("poc-mind: SpikingBrain ведёт органное тело");

    // ── С разумом ──
    let mut sim = OrganSim::new(false, 0x0000_AE11).with_mind(0x00_0A_11);
    let mut eaten = 0usize;
    let mut wall_hits = 0usize;
    let mut sit = 0usize;
    let mut seek = 0usize;
    let mut approach = 0usize;
    let mut flee = 0usize;
    let mut explore = 0usize;
    let mut fever_ticks = 0usize;

    for _ in 0..SIM_TICKS {
        let obs = sim.step_tick();
        if obs.ate {
            eaten += 1;
        }
        if obs.wall_hit {
            wall_hits += 1;
        }
        if obs.body_temp > dioma::organ_life::FEVER_TEMP {
            fever_ticks += 1;
        }
        match obs.action {
            Some(CreatureAction::Rest) => sit += 1,
            Some(CreatureAction::SeekFood) => seek += 1,
            Some(CreatureAction::Approach) => approach += 1,
            Some(CreatureAction::Flee) => flee += 1,
            _ => explore += 1,
        }
    }
    let total = SIM_TICKS as f64;
    println!(
        "жизнь с разумом: еда={eaten} удары={wall_hits} лихорадка={fever_ticks}"
    );
    println!(
        "действия: сидел {:.0}%  искал {:.0}%  подходил {:.0}%  бежал {:.0}%  бродил {:.0}%",
        sit as f64 / total * 100.0,
        seek as f64 / total * 100.0,
        approach as f64 / total * 100.0,
        flee as f64 / total * 100.0,
        explore as f64 / total * 100.0,
    );

    // Обучение видно? Статистика весов до и после жизни.
    let heat = sim.mind.as_ref().map(|m| m.weight_heat()).unwrap_or_default();
    if !heat.is_empty() {
        let mean_abs: f32 = heat.iter().map(|v| v.abs()).sum::<f32>() / heat.len() as f32;
        let zeros = heat.iter().filter(|&&v| v.abs() < 1e-4).count();
        println!(
            "W2 после жизни: |w̄|={mean_abs:.4}  нулевых={zeros}/{}",
            heat.len()
        );
    }

    // ── Контроль: тело без разума ──
    let mut plain = OrganSim::new(false, 0x0000_AE11);
    let mut plain_eaten = 0usize;
    let mut plain_walls = 0usize;
    for _ in 0..SIM_TICKS {
        let obs = plain.step_tick();
        if obs.ate {
            plain_eaten += 1;
        }
        if obs.wall_hit {
            plain_walls += 1;
        }
    }
    println!(
        "без разума: еда={plain_eaten} удары={plain_walls}"
    );
}
