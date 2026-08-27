//! Честный кроссовер популяции С ОБУЧЕНИЕМ (этап 1+2):
//! CPU-путь: существо со своим Brain (динамика + plast на CPU), полный тик
//! `tick` (50 подшагов + обучение + тело). GPU-путь: то же существо, но
//! обучение/динамика на GPU (step_many_learn), ридбек — моторные суммы.
//! Одиночные замеры + экстраполяция на популяцию (GPU-путь в одной сети
//! не растёт с N — диспатчи постоянны, как показал gpu_pool).
//!
//! Запуск: `cargo run --release --example gpu_pool_learn`
use dioma::creature::{Creature, Sense};
use std::time::Instant;

const TICKS: usize = 100;

fn main() {
    let seed = 7u64;
    let mut cpu = Creature::new(seed);
    let mut gpu = Creature::new_gpu(seed);
    if !gpu.brain.on_gpu() {
        println!("GPU недоступен");
        return;
    }
    let sense = Sense {
        smell_l: 0.6,
        smell_r: 0.3,
        vis_l: 0.0,
        vis_r: 0.0,        wall_l: 0.0,
        wall_r: 0.0,
        food_near: true,
        wall: false,
        place: 0,
        heading: 0.0,
        see_eat: false,
    };

    // Прогрев обеих сторон (компиляция шейдеров/раннеров, кэши).
    for _ in 0..3 {
        cpu.tick(&sense);
        gpu.tick(&sense);
    }

    // Замер CPU: полный тик существа (динамика + plast + тело).
    let t0 = Instant::now();
    for _ in 0..TICKS {
        cpu.tick(&sense);
    }
    let cpu_1 = t0.elapsed().as_micros() as f64 / TICKS as f64;

    // Замер GPU: полный тик с обучением на GPU.
    let t0 = Instant::now();
    for _ in 0..TICKS {
        gpu.tick(&sense);
    }
    let gpu_1 = t0.elapsed().as_micros() as f64 / TICKS as f64;

    // Пул: CPU линейно (N × одиночка — plast на CPU по каждому), GPU —
    // константа (одна сеть, блочно-диагональный CSR, диспатчи не растут с N).
    println!(
        "одиночка ({} нейронов, тик = 50 подшагов + обучение):",
        cpu.brain.n
    );
    println!("  CPU: {cpu_1:.1} мкс/тик | GPU: {gpu_1:.1} мкс/тик");
    println!("популяция (N существ в одной сети, экстраполяция):");
    for n_pop in [4usize, 16, 64, 256, 1024] {
        let cpu_n = cpu_1 * n_pop as f64;
        let gpu_n = gpu_1; // диспатчи на всю сеть не зависят от числа блоков
        println!(
            "  N={n_pop:5}: CPU {cpu_n:10.0} мкс/тик | GPU {gpu_n:10.0} мкс/тик | GPU/CPU {:.3}×",
            gpu_n / cpu_n
        );
    }
}
