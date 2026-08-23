//! -------------------------------------------------------------------------
//! Filename:    src/bin/genome.rs
//! Class:       DebugSuite
//! Purpose:     poc-genome: print creature genomes
//! Description: wildtype / random / mutated — the gene table of dioma.
//!
//! author: Atlasov anton
//! -------------------------------------------------------------------------

use dioma::arena::Genome;
use dioma::rng::XorShift;

fn show(tag: &str, g: &Genome) {
    println!("{tag}:");
    println!("  ── рост (как строится проводка)");
    println!("     seed                = {}   (реализация стратегии роста)", g.seed);
    println!("     gradient_weight     = {:.3}  (доверчивость конуса к морфогену)", g.gradient_weight);
    println!("     noise_weight        = {:.3}  (блуждание аксона)", g.noise_weight);
    println!("     max_len_s           = {:.0} µm (дальность сенсорных аксонов)", g.max_len_s);
    println!("     synapse_dist        = {:.1} µm (радиус замыкания синапса)", g.synapse_dist);
    println!("  ── проводка (насколько сильно)");
    println!("     w_sm                = {:.3}  (врождённый рефлекс S→M)", g.w_sm);
    println!("     w_si                = {:.3}  (рекрутирование торможения S→I)", g.w_si);
    println!("     w_ss                = {:.3}  (рецидивное возбуждение S→S)", g.w_ss);
    println!("     w_i                 = {:.3}  (торможение моторов I→M)", g.w_i);
    println!("  ── тело (как чувствует и двигается)");
    println!("     sensor_gain         = {:.2}  (ток сенсора)", g.sensor_gain);
    println!("     turn_gain           = {:.3} рад/спайк (чувствительность поворота)", g.turn_gain);
    println!("  ── физиология HAAP (гены экспрессии каналов)");
    println!("     hcn  = {:.2}  (h-ток: rebound, плато памяти)", g.physio.hcn);
    println!("     kcnq = {:.2}  (M-ток: адаптация против runaway)", g.physio.kcnq);
    println!("     para = {:.2}  (NaV: возбудимость/порог)", g.physio.para);
    println!("     leak = {:.2}  (K2P: входное сопротивление)", g.physio.leak);
    println!();
}

fn main() {
    println!("poc-genome: генофонд dioma (11 генов)\n");
    let wild = Genome::wildtype(0xA_E11);
    show("ДИКИЙ ТИП (отлаженная арена)", &wild);

    let mut rng = XorShift::new(0xE00);
    let random = Genome {
        seed: rng.next_u64(),
        gradient_weight: 0.3 + rng.unit() * 0.5,
        noise_weight: 0.02 + rng.unit() * 0.33,
        max_len_s: 110.0 + rng.unit() * 100.0,
        synapse_dist: 9.0 + rng.unit() * 11.0,
        w_sm: 0.08 + rng.unit() as f32 * 0.12,
        w_i: 0.10 + rng.unit() as f32 * 0.45,
        w_si: 0.05 + rng.unit() as f32 * 0.40,
        w_ss: 0.02 + rng.unit() as f32 * 0.23,
        sensor_gain: 1.5 + rng.unit() as f32 * 4.5,
        turn_gain: 0.02 + rng.unit() * 0.10,
        brain_model: dioma::brain::BrainModel::LifAlpha,
        physio: dioma::physiology::PhysioGenes {
            hcn: (0.1 + rng.unit() as f32 * 1.2),
            kcnq: (0.1 + rng.unit() as f32 * 1.2),
            para: (0.3 + rng.unit() as f32 * 1.4),
            leak: (0.3 + rng.unit() as f32 * 1.4),
        },
    };
    show("СЛУЧАЙНЫЙ (поколение 0 эволюции)", &random);

    let m = |rng: &mut XorShift, v: f64, s: f64, lo: f64, hi: f64| (v + rng.gauss() * s).clamp(lo, hi);
    let mutant = Genome {
        seed: rng.next_u64(),
        gradient_weight: m(&mut rng, wild.gradient_weight, 0.08, 0.3, 0.8),
        noise_weight: m(&mut rng, wild.noise_weight, 0.05, 0.02, 0.35),
        max_len_s: m(&mut rng, wild.max_len_s, 15.0, 110.0, 210.0),
        synapse_dist: m(&mut rng, wild.synapse_dist, 1.5, 9.0, 20.0),
        w_sm: m(&mut rng, wild.w_sm as f64, 0.025, 0.08, 0.20) as f32,
        w_si: m(&mut rng, wild.w_si as f64, 0.04, 0.05, 0.45) as f32,
        w_ss: m(&mut rng, wild.w_ss as f64, 0.02, 0.02, 0.25) as f32,
        w_i: m(&mut rng, wild.w_i as f64, 0.05, 0.10, 0.55) as f32,
        sensor_gain: m(&mut rng, wild.sensor_gain as f64, 0.5, 1.5, 6.0) as f32,
        turn_gain: m(&mut rng, wild.turn_gain, 0.012, 0.02, 0.12),
        brain_model: wild.brain_model,
        physio: {
            let p = &wild.physio;
            dioma::physiology::PhysioGenes {
                hcn: m(&mut rng, p.hcn as f64, 0.10, 0.0, 1.5) as f32,
                kcnq: m(&mut rng, p.kcnq as f64, 0.10, 0.0, 1.5) as f32,
                para: m(&mut rng, p.para as f64, 0.12, 0.1, 1.8) as f32,
                leak: m(&mut rng, p.leak as f64, 0.12, 0.1, 1.8) as f32,
            }
        },
    };
    show("МУТАНТ дикого типа (одно поколение)", &mutant);
}

