//! -------------------------------------------------------------------------
//! Filename:    examples/pythia_vm_bench.rs
//! Class:       Experiment
//! Purpose:     Пифия vs TNMT: эквивалентность + скорость на 1M клеток
//! Description: 1) корректность: интегратор Пифии и его TNMT-перевод
//!              сходятся по траектории;
//!              2) производительность: во сколько раз SIMD-байткод быстрее
//!              скалярной стековой машины.
//!
//! Запуск: cargo run --release --example pythia_vm_bench
// -------------------------------------------------------------------------
use dioma::engine::pythia_vm::{compile_program, execute_cells};
use dioma::engine::rules::{Op, Var};

/// Интегратор: Act = Act·leak + Inp·gain (без Fire — чистая арифметика).
fn integrator_prog() -> Vec<Op> {
    vec![
        Op::Load(Var::Act),
        Op::Const(0.9),
        Op::Mul,
        Op::Load(Var::Inp),
        Op::Const(0.3),
        Op::Mul,
        Op::Add,
        Op::Store(Var::Act),
    ]
}

/// Эталонная скалярная симуляция (то, что делает PythiaUnit.step).
fn scalar_run(act: &mut [f32], inp: f32, steps: usize) {
    for a in act.iter_mut() {
        let mut x = *a;
        for _ in 0..steps {
            x = x * 0.9 + inp * 0.3;
        }
        *a = x;
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Эксперимент: Пифия → TNMT (эквивалентность+скорость)");
    println!("╚══════════════════════════════════════════════════╝");

    let prog = integrator_prog();
    let bc = compile_program(&prog).expect("интегратор должен транслироваться");
    println!(
        "[1] трансляция: {} опкодов Пифии → {} опкодов TNMT",
        prog.len(),
        bc.len()
    );

    // ─── Эквивалентность ───
    let n_check = 8;
    let inp = 0.5f32;
    let mut ref_act = vec![0.0f32; n_check];
    scalar_run(&mut ref_act, inp, 100);

    let mut states = vec![0.0f32; 8 * n_check];
    for c in 0..n_check {
        states[Var::Inp.idx() * n_check + c] = inp;
    }
    unsafe {
        for _ in 0..100 {
            execute_cells(&mut states, n_check, &bc);
        }
    }
    let tnmt_act = states[Var::Act.idx() * n_check];
    let diff = (ref_act[0] - tnmt_act).abs();
    println!(
        "[2] эквивалентность за 100 шагов: эталон {:.6} vs TNMT {:.6} (Δ={:.2e}) {}",
        ref_act[0],
        tnmt_act,
        diff,
        if diff < 1e-4 { "OK" } else { "FAIL" }
    );

    // ─── Скорость: 1M клеток × 1000 шагов ───
    const N: usize = 1024 * 1024;
    const STEPS: usize = 1000;

    // Скалярно (одна клетка — репрезентативный замер, умножим мысленно):
    let t0 = std::time::Instant::now();
    let mut one = vec![0.0f32; 1];
    scalar_run(&mut one, inp, STEPS);
    let scalar_t = t0.elapsed();

    // TNMT SIMD:
    let mut big = vec![0.0f32; 8 * N];
    for c in 0..N {
        big[Var::Inp.idx() * N + c] = inp;
    }
    let t0 = std::time::Instant::now();
    unsafe {
        for _ in 0..STEPS {
            execute_cells(&mut big, N, &bc);
        }
    }
    let tnmt_t = t0.elapsed();

    // Скалярный полный прогон занял бы scalar_t × N.
    let projected_scalar = scalar_t.as_secs_f64() * N as f64;
    let speedup = projected_scalar / tnmt_t.as_secs_f64();

    println!("[3] скорость ({N} клеток × {STEPS} шагов):");
    println!(
        "    скалярная ВМ (проекция с 1 клетки): {:.2}s",
        projected_scalar
    );
    println!(
        "    TNMT SIMD (реально):                {:.3}s",
        tnmt_t.as_secs_f64()
    );
    println!("    ускорение: ×{:.0}", speedup);
    println!(
        "[4] финальное состояние: среднее Act = {:.6}",
        big[Var::Act.idx() * N..(Var::Act.idx() + 1) * N]
            .iter()
            .sum::<f32>()
            / N as f32
    );
}
