//! БИТВА МОЗГОВ — grown vs sarDNA vs diff_snn на одинаковых условиях.
use dioma::brain::{Brain, PlasticityParams};
use dioma::heritage::sardna_to_grown;
use dioma::rng::XorShift;
use pillars_connectome::{BrainConfig, ConnectomeData};

const ARENA: f64 = 150.0;
const EAT_R: f64 = 14.0;
const TICKS: usize = 2000;
const N_FOOD: usize = 20;
const EPISODES: usize = 8;

fn make_foods(rng: &mut XorShift) -> Vec<(f64, f64, bool)> {
    (0..N_FOOD).map(|_| {
        ((rng.unit()*2.0-1.0)*ARENA, (rng.unit()*2.0-1.0)*ARENA, true)
    }).collect()
}

fn run_life(brain: &mut Brain, n: usize, seed: u64) -> usize {
    let mut rng = XorShift::new(seed);
    let mut foods = make_foods(&mut rng);
    let mut ax = (rng.unit()*2.0-1.0)*ARENA;
    let mut ay = (rng.unit()*2.0-1.0)*ARENA;
    let mut heading = rng.unit() * std::f64::consts::PI * 2.0;
    let mut meals = 0usize;

    for _ in 0..TICKS {
        // Сенсорика.
        let (mut sl, mut sr) = (0.0f32, 0.0f32);
        let (mut bd, mut bx, mut by) = (f64::INFINITY, 0.0f64, 0.0f64);
        for f in foods.iter() {
            if !f.2 { continue; }
            let d = ((f.0-ax).powi(2)+(f.1-ay).powi(2)).sqrt();
            if d < bd { bd = d; bx=f.0; by=f.1; }
        }
        if bd < f64::INFINITY {
            let prox = 60.0/(60.0+bd);
            let rel = (by-ay).atan2(bx-ax) - heading;
            sl = (rel.sin().max(0.0)*prox*3.5) as f32;
            sr = ((-rel.sin()).max(0.0)*prox*3.5) as f32;
        }

        // Экст + мозг.
        let half = n/2;
        let mut ext = vec![0.12f32; n];
        for i in 0..half { ext[i] += sl*(0.5+0.5*dioma::rng::hash01(i) as f32); }
        for i in half..n { ext[i] += sr*(0.5+0.5*dioma::rng::hash01(i+999) as f32); }
        let spikes = brain.step(&ext);
        let ml: i64 = spikes[..half].iter().map(|&s| s as i64).sum();
        let mr: i64 = spikes[half..].iter().map(|&s| s as i64).sum();

        // Движение.
        heading += 0.06*(ml-mr) as f64;
        let spd = 2.0*((ml+mr) as f64/50.0).min(1.0).max(0.15);
        ax += spd*heading.cos()*0.5;
        ay += spd*heading.sin()*0.5;

        // Еда.
        for f in foods.iter_mut() {
            if !f.2 { continue; }
            if ((f.0-ax).powi(2)+(f.1-ay).powi(2)).sqrt() < EAT_R {
                f.2=false; meals+=1; break;
            }
        }
    }
    meals
}

fn main() {
    println!("БИТВА МОЗГОВ");
    let base = "D:/Проекты/asm_engines/sardaanaCM/experiments/sardna_ultra/generated";
    let data = "D:/Проекты/asm_engines/sardaanaCM/data/generated";
    
    // ── Grown ──
    {
        let genome = dioma::arena::Genome::wildtype(42);
        let grown = dioma::arena::grow_agent(&genome);
        let gains = dioma::arena::arena_gains(&genome);
        let nn = grown.region_of.len();
        let total: usize = (0..EPISODES).map(|ep| {
            let mut b = Brain::new(&grown, gains.clone(), PlasticityParams::default());
            run_life(&mut b, nn, 1000+ep as u64)
        }).sum();
        println!("Grown Agent: {} еды", total);
    }

    // ── SarDNA Pro ──
    {
        match sardna_to_grown(format!("{data}/SarDNA_Pro_best.taas")) {
            Ok(g) => {
                let gains = vec![vec![0.6f32;4];4];
                let nn = g.region_of.len();
                let total: usize = (0..EPISODES).map(|ep| {
                    let mut b = Brain::new(&g, gains.clone(), PlasticityParams::default());
                    run_life(&mut b, nn, 1000+ep as u64)
                }).sum();
                println!("SarDNA Pro: {} еды", total);
            }
            Err(e) => println!("SarDNA error: {e}"),
        }
    }

    // ── Diff SNN ──
    {
        match sardna_to_grown(format!("{data}/Diff_snn_best.taas")) {
            Ok(g) => {
                let gains = vec![vec![0.6f32;4];4];
                let nn = g.region_of.len();
                let total: usize = (0..EPISODES).map(|ep| {
                    let mut b = Brain::new(&g, gains.clone(), PlasticityParams::default());
                    run_life(&mut b, nn, 1000+ep as u64)
                }).sum();
                println!("Diff SNN: {} еды", total);
            }
            Err(e) => println!("Diff SNN error: {e}"),
        }
    }

    println!("\nготово");
}
