//! -------------------------------------------------------------------------
//! Filename:    src/bin/poc_delayed_haap.rs
//! Class:       SemantikaEngine
//! Purpose:     poc-delayed-haap — отложенный ответ на HAAP-мозге vs LIF
//! Description: та же задача (маяк → тьма → поиск), два мозга. Если HAAP
//!              проходит — архитектура живёт за пределами «броди-жри».
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::delayed::{
    cognitive_gains, grow_cognitive, run_episode, Agent, EpisodeStats, SEED,
};
use dioma::physiology;
use dioma::brain::{Brain, PlasticityParams};

fn haap_brain(grown: &dioma::Grown) -> Brain {
    let cols = physiology::brain_columns(
        &grown.region_of,
        physiology::cognitive_archetype,
        &Default::default(),
    );
    let mut params = PlasticityParams::default();
    // HAAP на когнитивной схеме: рекуррентность через WS/WM без тормоза
    // уходит в самоподдерживающуюся активацию — гомеостаз каждый тик.
    params.homeo_every_ticks = 1;
    // S→S и WM→WS рекуррентности не учатся (урок арены).
    params.frozen_pairs = vec![(5, 5), (6, 6), (7, 7), (8, 8)];
    Brain::new_haap(grown, cognitive_gains(), params, &cols)
}

fn lif_brain(grown: &dioma::Grown) -> Brain {
    Brain::new(grown, cognitive_gains(), PlasticityParams::default())
}

fn run_n(brain: &mut Brain, grown: &dioma::Grown, n: usize, tag: &str) {
    let mut results = Vec::with_capacity(n);
    for k in 0..n {
        let mut agent = Agent::new(SEED.wrapping_add(k as u64));
        let stats = run_episode(brain, &mut agent, false);
        results.push(stats);
    }
    let mean_ticks: f64 =
        results.iter().map(|s| s.ticks as f64).sum::<f64>() / n as f64;
    let mean_ws: f32 =
        results.iter().map(|s| s.ws_rate_cue).sum::<f32>() / n as f32;
    let mean_wm: f32 =
        results.iter().map(|s| s.wm_rate_delay).sum::<f32>() / n as f32;
    let found = results.iter().filter(|s| s.ticks < 600).count();
    println!(
        "{tag}: еда {found}/{n} · тики ср.={mean_ticks:.0} · WS ignition={mean_ws:.1} · WM плато={mean_wm:.1}"
    );
}

fn main() {
    println!("poc-delayed-haap: отложенный ответ — LIF vs HAAP");
    let grown = grow_cognitive(SEED);
    println!("когнитивный коннектом: {} синапсов", grown.edges.len());

    println!("\n── LIF (контроль) ──");
    let mut lif = lif_brain(&grown);
    run_n(&mut lif, &grown, 10, "LIF");

    println!("\n── HAAP ──");
    let mut haap = haap_brain(&grown);
    run_n(&mut haap, &grown, 10, "HAAP");

    println!("\nЕсли HAAP находит еду и WS зажигается — архитектура обобщается.");
}
