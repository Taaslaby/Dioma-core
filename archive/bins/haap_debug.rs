//! Временная диагностика: HAAP-мозг — фактический dt, вспышки, веса.
use dioma::arena::{
    arena_brain, grow_agent, Agent, Genome, N_REGIONS, N_SL, N_SR, REG_ML, REG_MR, TONIC_DRIVE,
};
use dioma::brain::BrainModel;

fn main() {
    let mut genome = Genome::wildtype(0xA_E11);
    genome.brain_model = BrainModel::Haap;
    let grown = grow_agent(&genome);

    let mut brain = arena_brain(&grown, &genome);
    println!(
        "n={} n_syn={} dt_ms()={:?} steps/tick={}",
        brain.n,
        brain.n_syn(),
        brain.dt_ms(),
        brain.steps_per_tick()
    );
    println!("dt колонка до первого step: {:?}", brain.haap_dt_column());

    let zero = vec![0.0f32; brain.n];
    brain.step(&zero);
    println!("dt колонка после 1 step:    {:?}", brain.haap_dt_column());

    let mut agent = Agent::new(genome.seed);
    for ep in 0..5 {
        let mut max_tick_spikes = 0i64;
        let mut total_spikes = 0i64;
        let mut ticks = 600usize;
        let mut first_burst_tick: Option<usize> = None;
        for tick in 0..600 {
            let (left, right) = agent.sensors();
            let n = brain.n;
            let mut ext = vec![TONIC_DRIVE; n];
            for (i, cell) in ext.iter_mut().enumerate().take(N_SL) {
                *cell += 3.5 * left as f32 * (0.5 + 0.5 * dioma::rng::hash01(i) as f32);
            }
            for (i, cell) in ext.iter_mut().enumerate().take(N_SL + N_SR).skip(N_SL) {
                *cell += 3.5 * right as f32 * (0.5 + 0.5 * dioma::rng::hash01(i + 999) as f32);
            }
            let mut tick_regions = [0i32; N_REGIONS];
            for _ in 0..brain.steps_per_tick() {
                let spikes = brain.step(&ext);
                total_spikes += spikes.iter().map(|&s| s as i64).sum::<i64>();
                for (r, sum) in tick_regions.iter_mut().enumerate() {
                    *sum += brain.region_spikes(&spikes, r);
                }
            }
            let tick_total: i64 = tick_regions.iter().map(|&x| x as i64).sum();
            if tick == 0 || tick == 50 || (tick > 100 && tick % 100 == 0) {
                let st = brain
                    .haap_state_stats()
                    .map(|(v, vx, u)| format!(" v=[{v:.0};{vx:.0}] u={u:.2}"))
                    .unwrap_or_default();
                println!(
                    "   t{tick:3} S={:5} I={:5} ML={:5} MR={:5} sum={tick_total:6}{st}",
                    tick_regions[0] + tick_regions[1],
                    tick_regions[2],
                    tick_regions[3],
                    tick_regions[4]
                );
            }
            if tick_total > 200 && first_burst_tick.is_none() {
                first_burst_tick = Some(tick);
            }
            max_tick_spikes = max_tick_spikes.max(tick_total);
            agent.move_agent(tick_regions[REG_ML], tick_regions[REG_MR], genome.turn_gain, 2.0);
            if agent.at_food() {
                brain.schedule_reward(1.0);
                agent.respawn_food();
                // Жевание: награда приходит в тишину (канон арены).
                let zero = vec![TONIC_DRIVE; n];
                let mut chew = [0i32; N_REGIONS];
                for _ in 0..=brain.reward_delay_ticks() {
                    chew = [0i32; N_REGIONS];
                    for _ in 0..brain.steps_per_tick() {
                        let spikes = brain.step(&zero);
                        for (r, sum) in chew.iter_mut().enumerate() {
                            *sum += brain.region_spikes(&spikes, r);
                        }
                    }
                }
                max_tick_spikes = max_tick_spikes
                    .max(chew.iter().map(|&x| x as i64).sum::<i64>());
                ticks = tick + 1;
                break;
            }
        }
        let (lo, hi) = brain.values_minmax();
        let st = brain
            .haap_state_stats()
            .map(|(v, vx, u)| format!(" v=[{v:.1};{vx:.1}] ū={u:.3}"))
            .unwrap_or_default();
        println!(
            "ep{ep}: ticks={ticks} burst_at={first_burst_tick:?} max_tick={max_tick_spikes} total={total_spikes} weights=[{lo:.3}; {hi:.3}]{st}"
        );
    }
}


