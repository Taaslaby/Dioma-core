//! Честный бенчмарк арены: tps (тиков мира в секунду) для раскладок/бэкендов.
//!
//! Матрица (release):
//!   CPU@300   — стадия B (эталон, 300 нейронов),
//!   CPU@760   — стадия C-lite (760 нейронов),
//!   GPU-per@760 — каждое существо со своим GPU-раннером (Creature::new_gpu),
//!   GPU-pool@760 — батч-пул (один GPU-раннер на всю арену) — включается
//!                  env DIOMA_POOL=1 (требует собранный BrainPool в Arena).
//!
//! По 5 и 20 существ. Запуск: `cargo run --release --example arena_tps_bench`
use dioma_engine::creature::{BrainLayout, Creature, CreatureBuilder};
use dioma_engine::env::map::WorldMap;
use dioma_engine::env::Vec2;
use dioma_engine::sim::Arena;
use std::time::Instant;

/// Число тиков замера после прогрева.
const MEASURE_TICKS: usize = 60;
/// Прогрев (компиляция шейдеров/раннеров, кэши, рождение).
const WARMUP_TICKS: usize = 10;

const BERRIES: [(f32, f32); 8] = [
    (12.0, 5.0),
    (20.0, 15.0),
    (28.0, 6.0),
    (36.0, 14.0),
    (44.0, 5.0),
    (52.0, 15.0),
    (60.0, 6.0),
    (68.0, 14.0),
];

/// Собрать арену с n существами (сиды 100..), комнатой и ягодами.
fn build_arena(n: usize, make: impl Fn(u64) -> Creature) -> Arena {
    let mut map = WorldMap::default();
    map.add_room(0.0, 0.0, 80.0, 20.0);
    let mut arena = Arena::new(42, map);
    for i in 0..n {
        let seed = 100 + i as u64;
        arena.spawn_creature(
            Vec2::new(6.0 + (i % 10) as f32 * 3.0, 10.0),
            0.3 * i as f32,
            make(seed),
        );
    }
    for (x, y) in BERRIES {
        arena.add_berry(Vec2::new(x, y));
    }
    arena
}

fn measure(mut arena: Arena, label: &str, neurons: usize) {
    for _ in 0..WARMUP_TICKS {
        arena.step();
    }
    let t0 = Instant::now();
    for _ in 0..MEASURE_TICKS {
        arena.step();
    }
    let secs = t0.elapsed().as_secs_f64();
    let tps = MEASURE_TICKS as f64 / secs;
    println!(
        "{label}: {tps:8.1} tps ({MEASURE_TICKS} тиков за {secs:.2} с, {} существ × {neurons} нейронов)",
        arena.critters.len()
    );
}

fn main() {
    let pool_enabled = std::env::var("DIOMA_POOL").map(|v| v == "1").unwrap_or(false);
    println!(
        "машина: {} логических ядер | GPU-пул: {}",
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        if pool_enabled { "вкл" } else { "выкл" }
    );

    // ── CPU@300 (эталон стадии B) ──
    for n in [5usize, 20] {
        let make = |seed: u64| {
            CreatureBuilder::new()
                .seed(seed)
                .layout(BrainLayout::StageB)
                .build()
        };
        measure(build_arena(n, make), "CPU@300", 300);
    }

    // ── CPU@760 ──
    for n in [5usize, 20] {
        let make = |seed: u64| Creature::new(seed);
        measure(build_arena(n, make), "CPU@760", 760);
    }

    // ── GPU per-creature @760 ──
    for n in [5usize, 20] {
        let make = |seed: u64| Creature::new_gpu(seed);
        measure(build_arena(n, make), "GPU-per@760", 760);
    }

    if pool_enabled {
        // ── GPU-пул @760: один раннер на всю арену ──
        for n in [5usize, 20] {
            let mut map = WorldMap::default();
            map.add_room(0.0, 0.0, 80.0, 20.0);
            let mut arena = Arena::new_pooled(42, map);
            for i in 0..n {
                let seed = 100 + i as u64;
                arena.spawn_pooled(Vec2::new(6.0 + (i % 10) as f32 * 3.0, 10.0), 0.3 * i as f32, seed);
            }
            for (x, y) in BERRIES {
                arena.add_berry(Vec2::new(x, y));
            }
            measure(arena, "GPU-pool@760", 760);
        }
    } else {
        println!("GPU-pool@760: пропущен (DIOMA_POOL=1 для батч-пула)");
    }
}
