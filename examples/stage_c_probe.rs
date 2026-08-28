//! Проба стадии C-lite: рост и топология. Меряем время роста и число синапсов
//! для stage_b (300) против stage_c_lite (760) на нескольких сидах — чтобы
//! понять цену укрупнения коры ДО переключения существ.
//!
//! Запуск: `cargo run --release --example stage_c_probe`
use dioma::brain::grow::{grow, stage_b_layout, stage_c_lite_layout, Layout, Steering};
use std::collections::BTreeMap;
use std::time::Instant;

fn pair_counts(
    edges: &[(usize, usize, f32)],
    region_of: &[usize],
) -> BTreeMap<(usize, usize), usize> {
    let mut m = BTreeMap::new();
    for &(pre, post, _) in edges {
        *m.entry((region_of[pre], region_of[post])).or_insert(0) += 1;
    }
    m
}

fn run_case(name: &str, layout: &Layout, st: &Steering, seed: u64) {
    let t0 = Instant::now();
    let g = grow(layout, st, seed);
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let pc = pair_counts(&g.edges, &g.region_of);
    println!(
        "== {name} seed={seed}: {:.0} мс, нейронов {}, синапсов {}, пар {}",
        ms,
        g.nt.len(),
        g.edges.len(),
        pc.len()
    );
    for ((a, b), n) in &pc {
        println!("   {a}->{b}: {n}");
    }
}

fn main() {
    let st = Steering::default();
    let st_short = Steering { grow_steps: 200, ..st };
    println!("== рост 200 шагов (как в тестах) ==");
    for seed in [1u64, 7, 42] {
        run_case("stage_b", &stage_b_layout(), &st_short, seed);
        run_case("stage_c", &stage_c_lite_layout(), &st_short, seed);
    }
    println!("== рост 400 шагов (дефолтный геном) ==");
    run_case("stage_b", &stage_b_layout(), &st, 7);
    run_case("stage_c", &stage_c_lite_layout(), &st, 7);
}
