//! Проба: CPU-динамика против GPU-динамики (pillars-wgpu, alpha-lif) на одном
//! и том же grown-коннектоме. Честные цифры: совпадение спайков, время тика,
//! активен ли GPU.
//!
//! Запуск: `cargo run --release --example gpu_probe`
use dioma::creature::{Creature, Sense};
use std::time::Instant;

fn main() {
    let seed = 7u64;
    let cpu = Creature::new(seed);
    let mut gpu = Creature::new_gpu(seed);
    println!(
        "GPU-динамика: {} ({} нейронов, {} синапсов)",
        if gpu.brain.on_gpu() { "АКТИВНА" } else { "нет — CPU-фолбэк" },
        cpu.brain.n,
        cpu.brain.n_syn()
    );
    if !gpu.brain.on_gpu() {
        println!("Адаптер недоступен — паритет не сравниваем.");
        return;
    }

    // Один и тот же сенсорный поток для обоих мозгов.
    let senses: Vec<Sense> = (0..200)
        .map(|i| {
            let phase = i as f32 * 0.37;
            Sense {
                smell_l: (phase.sin() * 0.5 + 0.5).min(1.0),
                smell_r: ((phase + 1.7).sin() * 0.5 + 0.5).min(1.0),
                wall_l: if i % 23 == 0 { 0.8 } else { 0.0 },
                wall_r: if i % 31 == 0 { 0.8 } else { 0.0 },
                ..Default::default()
            }
        })
        .collect();

    // ── CPU: 200 тиков ──
    let t0 = Instant::now();
    let mut cpu = cpu;
    let mut cpu_motor = 0i64;
    for s in &senses {
        let a = cpu.tick(s);
        cpu_motor += (a.steer.abs() * 1000.0) as i64;
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // ── GPU: 200 тиков (мозг свежий — тот же сид) ──
    let t0 = Instant::now();
    let mut gpu_motor = 0i64;
    for s in &senses {
        let a = gpu.tick(s);
        gpu_motor += (a.steer.abs() * 1000.0) as i64;
    }
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("CPU: {cpu_ms:.1} ms / 200 тиков, моторная активность {cpu_motor}");
    println!("GPU: {gpu_ms:.1} ms / 200 тиков, моторная активность {gpu_motor}");
    let ratio = cpu_ms / gpu_ms.max(1e-9);
    println!("Скорость: GPU {:.2}× от CPU", ratio);

    // ── Паритет спайков: потиково одинаковый вход, сравниваем сумму ──
    let mut cpu = Creature::new(seed);
    let mut gpu = Creature::new_gpu(seed);
    let mut cpu_total = 0i64;
    let mut gpu_total = 0i64;
    for s in &senses {
        let a = cpu.tick(s);
        cpu_total += (a.steer.abs() * 1000.0) as i64;
        let a = gpu.tick(s);
        gpu_total += (a.steer.abs() * 1000.0) as i64;
    }
    let rel = (cpu_total - gpu_total).abs() as f64 / cpu_total.max(1) as f64;
    println!(
        "Паритет моторного выхода: CPU={cpu_total} GPU={gpu_total} |Δ|={rel:.1}%"
    );
}
