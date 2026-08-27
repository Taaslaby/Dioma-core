//! Замер: аллоцирующий `step()` vs неаллоцирующий `step_into()` на ядре мозга.
//!
//! Мозг существа делает до 500 подшагов/тик (HAAP). `step()` копировал
//! `n_neurons` int'ов через `.to_vec()` каждый подшаг — `step_into` пишет в
//! переиспользуемый `out`, убирая аллокацию. Замеряем разницу на популяции.
//!
//! Run: `cargo run --release --example brain_alloc_bench`

use dioma::engine::genome::Grown;
use dioma::engine::neurons::{gains_from, Brain, PlasticityParams};
use std::hint::black_box;
use std::time::Instant;

const NEURONS: usize = 300;
const SYN_PER_NEURON: usize = 3;
const POP: usize = 200;     // популяция существ
const TICKS: usize = 30;    // тиков мира
const SPT: usize = 50;      // подшагов мозга на тик (LIF)

fn make_grown() -> Grown {
    let mut edges = Vec::new();
    // 2 региона: сенсорные (0) и моторные (1), плотный коннектом внутри/между.
    let region_of: Vec<usize> = (0..NEURONS).map(|i| if i < NEURONS / 2 { 0 } else { 1 }).collect();
    for pre in 0..NEURONS {
        for k in 0..SYN_PER_NEURON {
            let post = (pre + k * 7 + 1) % NEURONS;
            let sign = if k % 2 == 0 { 1.0 } else { -0.5 };
            edges.push((pre, post, sign));
        }
        // сенсорика → моторика (поперёк)
        if pre < NEURONS / 2 {
            let post = NEURONS / 2 + (pre * 3) % (NEURONS / 2);
            edges.push((pre, post, 1.0));
        }
    }
    Grown {
        edges,
        region_of,
        nt: vec![1.0; NEURONS],
        growth_s: 0.0,
        rule_genes: Vec::new(),
        instincts: Vec::new(),
    }
}

fn main() {
    let grown = make_grown();
    let gains = gains_from(2, &[(0, 1, 1.0), (0, 0, 0.5), (1, 1, 0.5)]);
    // homeo=1: гомеостаз каждый мирок-тик (реальный сценарий диомы) —
    // показать выигрыш от переиспользуемых буферов и в normalize.
    let params = PlasticityParams {
        homeo_every_ticks: 1,
        ..Default::default()
    };

    // Популяция мозгов (по одному на существо).
    let mut brains: Vec<Brain> = (0..POP).map(|_| Brain::new(&grown, gains.clone(), params.clone())).collect();

    // ── путь A: аллоцирующий `step()` (старый) ──
    let ext = vec![0.5f32; NEURONS];
    let start = Instant::now();
    for _ in 0..TICKS {
        for b in brains.iter_mut() {
            for _ in 0..SPT {
                let sp = b.step(&ext);
                black_box(&sp);
            }
        }
    }
    let alloc_time = start.elapsed();

    // ── путь B: неаллоцирующий `step_into()` (новый) ──
    let start = Instant::now();
    for _ in 0..TICKS {
        for b in brains.iter_mut() {
            let mut spikes = Vec::with_capacity(b.n);
            for _ in 0..SPT {
                b.step_into(&ext, &mut spikes);
                black_box(&spikes);
            }
        }
    }
    let reuse_time = start.elapsed();

    let alloc_ms = alloc_time.as_secs_f64() * 1e3;
    let reuse_ms = reuse_time.as_secs_f64() * 1e3;
    println!("N={NEURONS} pop={POP} ticks={TICKS} spt={SPT}");
    println!("step()     (аллоцирует): {alloc_ms:.1} ms");
    println!("step_into() (переисп. буфер): {reuse_ms:.1} ms");
    println!("SPEEDUP: {:.2}x", alloc_ms / reuse_ms);
}