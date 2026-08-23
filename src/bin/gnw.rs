//! -------------------------------------------------------------------------
//! Filename:    src/bin/gnw.rs
//! Class:       SemantikaEngine
//! Purpose:     poc-gnw: grown global workspace — delayed response
//! Description: full brain vs WS/WM ablation, ignition/memory metrics.
//!
//! author: Atlasov anton
//! -------------------------------------------------------------------------
use dioma::brain::{BRAIN_STEPS_PER_TICK, DT_MS};
use dioma::delayed::{ablated_edges, grow_cognitive, run_protocol_n, REG_WSL, SEED};

fn main() {
    println!("poc-gnw: выращенное глобальное рабочее пространство");
    println!("задача: маяк, затем слепота — курс держит только память\n");

    let t0 = std::time::Instant::now();
    let grown = grow_cognitive(SEED);
    println!(
        "рост: {} синапсов за {:.2} с",
        grown.edges.len(),
        grown.growth_s
    );

    // Направленность выращенной схемы.
    let mut own = 0usize;
    let mut cross = 0usize;
    for &(pre, post, _) in &grown.edges {
        let rp = grown.region_of[pre];
        let rq = grown.region_of[post];
        let to_wsl = rq == REG_WSL;
        let to_wsr = rq == REG_WSL + 1;
        if rp == 0 && to_wsl {
            own += 1;
        }
        if (rp == 0 && to_wsr) || (rp == 1 && to_wsl) {
            cross += 1;
        }
    }
    println!("направленность S→WS: свой ×{own}, перекрёстный ×{cross}");

    let full = run_protocol_n(&grown.edges, 30);
    let ablated = run_protocol_n(&ablated_edges(&grown), 30);

    let sec = |s: &[dioma::delayed::EpisodeStats]| {
        s.iter().map(|x| x.ticks as f64).sum::<f64>() / s.len() as f64
            * BRAIN_STEPS_PER_TICK as f64
            * DT_MS as f64
            / 1000.0
    };
    let wm: f32 = full.iter().map(|s| s.wm_rate_delay).sum::<f32>() / full.len() as f32;
    let ws: f32 = full.iter().map(|s| s.ws_rate_cue).sum::<f32>() / full.len() as f32;
    let (m_full, m_abl) = (sec(&full), sec(&ablated));

    println!("\n═══ СВОДКА ═══");
    println!("ПОЛНЫЙ:       время до еды = {m_full:.2} с, WS на маяке = {ws:.1}, WM в задержке = {wm:.1}");
    println!("АБЛИРОВАННЫЙ: время до еды = {m_abl:.2} с");
    let gain = (m_abl - m_full) / m_abl.max(1e-9) * 100.0;
    let verdict = if gain > 15.0 && wm > 1.0 {
        format!("WORKSPACE РАБОТАЕТ: на {gain:.0}% быстрее абляции, память держится")
    } else if wm > 1.0 {
        format!("память держится ({wm:.1} сп/тик), но выигрыш мал ({gain:+.0}%)")
    } else {
        format!("память НЕ держится (WM = {wm:.1} сп/тик)")
    };
    println!("ВЕРДИКТ: {verdict}");
    println!("\nполное время: {:.2} с", t0.elapsed().as_secs_f64());
}
