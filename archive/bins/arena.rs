//! -------------------------------------------------------------------------
//! Filename:    src/bin/arena.rs
//! Class:       DebugSuite
//! Purpose:     poc-arena: grown brain in a body — food seeking
//! Description: experiment (dopamine on) vs control, w(S→M) tracking.
//! 
//! author: Atlasov anton
//! -------------------------------------------------------------------------
use dioma::arena::{arena_brain, grow_agent, run_episode, Agent, Genome, REG_ML, REG_MR, REG_SL, REG_SR};
use dioma::brain::{DT_MS, BRAIN_STEPS_PER_TICK};

const N_EPISODES: usize = 60;

fn main() {
    println!("poc-arena: выращенный мозг в теле — поиск еды");
    let genome = Genome::wildtype(0xA_E11);
    let grown = grow_agent(&genome);
    println!("рост: {} синапсов за {:.2} с", grown.edges.len(), grown.growth_s);

    let exp = run_protocol(&grown, &genome, true, "ЭКСПЕРИМЕНТ");
    let ctrl = run_protocol(&grown, &genome, false, "КОНТРОЛЬ");

    let sec = |t: &[usize]| {
        t.iter().map(|&x| x as f64).sum::<f64>() / t.len() as f64 * BRAIN_STEPS_PER_TICK as f64
            * DT_MS as f64 / 1000.0
    };
    println!("\n═══ СВОДКА (среднее время до еды, с) ═══");
    println!("ЭКСПЕРИМЕНТ: первые 10 = {:.2}, последние 10 = {:.2}", sec(&exp[..10]), sec(&exp[exp.len() - 10..]));
    println!("КОНТРОЛЬ:    первые 10 = {:.2}, последние 10 = {:.2}", sec(&ctrl[..10]), sec(&ctrl[ctrl.len() - 10..]));
}

fn run_protocol(grown: &dioma::Grown, genome: &Genome, dopamine_on: bool, tag: &str) -> Vec<usize> {
    println!("\n═══ {tag} (дофамин = {}) ═══", dopamine_on as u8);
    let mut brain = arena_brain(grown, genome);
    let mut agent = Agent::new(genome.seed);

    let (w0, c_sm) = brain.mean_path_weight(REG_SL, REG_ML);
    let (_, c_ll) = brain.mean_path_weight(REG_SL, REG_ML);
    let (_, c_lr) = brain.mean_path_weight(REG_SL, REG_MR);
    let (_, c_rl) = brain.mean_path_weight(REG_SR, REG_ML);
    let (_, c_rr) = brain.mean_path_weight(REG_SR, REG_MR);
    println!(
        "пути S→M: ×{c_sm} (w̄ = {w0:.3}); направленность: S_L→M_L ×{c_ll}, S_L→M_R ×{c_lr}, S_R→M_L ×{c_rl}, S_R→M_R ×{c_rr}"
    );

    let mut times = Vec::with_capacity(N_EPISODES);
    for k in 0..N_EPISODES {
        let t = run_episode(&mut brain, &mut agent, genome, dopamine_on);
        times.push(t);
        if (k + 1) % 20 == 0 {
            let (w, _) = brain.mean_path_weight(REG_SL, REG_ML);
            let last: f64 = times[k - 19..=k].iter().map(|&t| t as f64).sum::<f64>() / 20.0;
            println!(
                "   эпизод {:+>3}: время до еды (ср. за 20) = {:>6.2} с, w(S→M) = {w:.4}",
                k + 1,
                last * BRAIN_STEPS_PER_TICK as f64 * DT_MS as f64 / 1000.0
            );
        }
    }
    let (w1, _) = brain.mean_path_weight(REG_SL, REG_ML);
    println!(
        "ИТОГ {tag}: w(S→M) {w0:.4} → {w1:.4} ({:+.1}%)",
        (w1 - w0) / w0.max(1e-9) * 100.0
    );
    times
}
