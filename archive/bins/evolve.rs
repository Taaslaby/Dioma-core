//! -------------------------------------------------------------------------
//! Filename:    src/bin/evolve.rs
//! Class:       DebugSuite
//! Purpose:     poc-evolve: genome selection vs random search
//! Description: Baldwin-constrained genomes (weak innate path), tournament + elite.
//! 
//! author: Atlasov anton
//! -------------------------------------------------------------------------
use dioma::arena::{evaluate, Genome};
use dioma::rng::XorShift;

const POP: usize = 6;
const GENERATIONS: usize = 10;
const ELITE: usize = 2;
const TOURNAMENT_K: usize = 3;
const EVAL_EPISODES: usize = 12;
const EVAL_SEED: u64 = 0xE0A1;
const RNG_SEED: u64 = 0xE00;

fn random_genome(rng: &mut XorShift) -> Genome {
    Genome {
        seed: rng.next_u64(),
        gradient_weight: 0.3 + rng.unit() * 0.5,
        noise_weight: 0.02 + rng.unit() * 0.33,
        max_len_s: 110.0 + rng.unit() * 100.0,
        synapse_dist: 9.0 + rng.unit() * 11.0,
        // Слабый врождённый путь: учиться обязательно (эффект Болдуина).
        w_sm: 0.08 + rng.unit() as f32 * 0.12,
        sensor_gain: 1.5 + rng.unit() as f32 * 4.5,
        turn_gain: 0.02 + rng.unit() * 0.10,
        w_si: 0.05 + rng.unit() as f32 * 0.40,
        w_ss: 0.02 + rng.unit() as f32 * 0.23,
        w_i: 0.10 + rng.unit() as f32 * 0.45,
        brain_model: dioma::brain::BrainModel::LifAlpha,
        physio: dioma::physiology::PhysioGenes {
            hcn: (0.1 + rng.unit() as f32 * 1.2),
            kcnq: (0.1 + rng.unit() as f32 * 1.2),
            para: (0.3 + rng.unit() as f32 * 1.4),
            leak: (0.3 + rng.unit() as f32 * 1.4),
        },
    }
}

fn mutate(g: &Genome, rng: &mut XorShift) -> Genome {
    let m = |rng: &mut XorShift, v: f64, sigma: f64, lo: f64, hi: f64| (v + rng.gauss() * sigma).clamp(lo, hi);
    Genome {
        seed: rng.next_u64(),
        gradient_weight: m(rng, g.gradient_weight, 0.08, 0.3, 0.8),
        noise_weight: m(rng, g.noise_weight, 0.05, 0.02, 0.35),
        max_len_s: m(rng, g.max_len_s, 15.0, 110.0, 210.0),
        synapse_dist: m(rng, g.synapse_dist, 1.5, 9.0, 20.0),
        w_sm: m(rng, g.w_sm as f64, 0.025, 0.08, 0.20) as f32,
        sensor_gain: m(rng, g.sensor_gain as f64, 0.5, 1.5, 6.0) as f32,
        turn_gain: m(rng, g.turn_gain, 0.012, 0.02, 0.12),
        w_si: m(rng, g.w_si as f64, 0.04, 0.05, 0.45) as f32,
        w_ss: m(rng, g.w_ss as f64, 0.02, 0.02, 0.25) as f32,
        w_i: m(rng, g.w_i as f64, 0.05, 0.10, 0.55) as f32,
        brain_model: g.brain_model,
        physio: {
            let p = &g.physio;
            dioma::physiology::PhysioGenes {
                hcn: m(rng, p.hcn as f64, 0.10, 0.0, 1.5) as f32,
                kcnq: m(rng, p.kcnq as f64, 0.10, 0.0, 1.5) as f32,
                para: m(rng, p.para as f64, 0.12, 0.1, 1.8) as f32,
                leak: m(rng, p.leak as f64, 0.12, 0.1, 1.8) as f32,
            }
        },
    }
}

fn tournament(order: &[usize], rng: &mut XorShift) -> usize {
    let mut best_pos = (rng.next_u64() as usize) % order.len();
    for _ in 1..TOURNAMENT_K.min(order.len()) {
        best_pos = best_pos.min((rng.next_u64() as usize) % order.len());
    }
    order[best_pos]
}

fn main() {
    println!("poc-evolve: отбор геномов против случайного поиска");
    println!("популяция = {POP}, поколений = {GENERATIONS}, эпизодов на оценку = {EVAL_EPISODES}\n");
    let t0 = std::time::Instant::now();
    let mut rng = XorShift::new(RNG_SEED);

    let mut pop: Vec<Genome> = (0..POP).map(|_| random_genome(&mut rng)).collect();
    let mut fitness = vec![f64::MAX; POP];
    let mut best_random = f64::MAX;
    let mut evals = 0usize;

    for gen in 0..GENERATIONS {
        for i in 0..POP {
            if fitness[i] == f64::MAX {
                fitness[i] = evaluate(&pop[i], true, EVAL_EPISODES);
                evals += 1;
            }
        }
        for _ in 0..POP {
            let f = evaluate(&random_genome(&mut rng), true, EVAL_EPISODES);
            evals += 1;
            best_random = best_random.min(f);
        }
        let mut order: Vec<usize> = (0..POP).collect();
        order.sort_by(|&a, &b| fitness[a].total_cmp(&fitness[b]));
        let best = fitness[order[0]];
        let mean: f64 = fitness.iter().sum::<f64>() / POP as f64;
        println!("поколение {gen}: лучший = {best:6.2} с, средний = {mean:6.2} с | случайный поиск: лучший = {best_random:6.2} с");

        if gen + 1 == GENERATIONS {
            let with_da = evaluate(&pop[order[0]], true, EVAL_EPISODES);
            let without_da = evaluate(&pop[order[0]], false, EVAL_EPISODES);
            println!("\n═══ ФИНАЛ ({evals} оценок, {:.1} с) ═══", t0.elapsed().as_secs_f64());
            println!("чемпион (эволюция):  {with_da:6.2} с (дофамин), {without_da:6.2} с (без дофамина)");
            println!("случайный поиск:     {best_random:6.2} с");
            let verdict = if with_da < best_random * 0.95 {
                "ЭВОЛЮЦИЯ ПОБЕДИЛА (>5% быстрее случайного поиска)"
            } else if with_da < best_random {
                "эволюция чуть лучше случайного поиска (<5%)"
            } else {
                "эволюция НЕ обогнала случайный поиск"
            };
            println!("ВЕРДИКТ: {verdict}");
            break;
        }

        let mut next = Vec::with_capacity(POP);
        let mut next_fit = Vec::with_capacity(POP);
        for &i in &order[..ELITE] {
            next.push(pop[i].clone());
            next_fit.push(fitness[i]);
        }
        while next.len() < POP {
            let parent = tournament(&order, &mut rng);
            next.push(mutate(&pop[parent], &mut rng));
            next_fit.push(f64::MAX);
        }
        pop = next;
        fitness = next_fit;
    }
    let _ = EVAL_SEED;
}

