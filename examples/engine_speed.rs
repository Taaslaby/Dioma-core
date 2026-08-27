//! Fair CPU comparison of a fixed SIMD model and the BioVM bytecode runtime.
//!
//! Run with:
//! `cargo run --release --example engine_speed`

use dioma::engine::biovm::{BioProgram, Instruction};
use pillars::core::runner::run_neuron;
use pillars::prelude::{Model, SimdContext, SimdLane};
use std::hint::black_box;
use std::time::Instant;

const N: usize = 131_072;
const STEPS: usize = 500;
const SWAP_EVERY: usize = 10;

/// State slots for the BioVM integrator program.
mod slot {
    pub const ACT: usize = 0;
    pub const INP: usize = 1;
}

/// Registers used by the BioVM integrator program.
mod reg {
    pub const ACT: u8 = 0;
    pub const KONST: u8 = 1;
    pub const INP: u8 = 2;
    pub const TERM: u8 = 3;
}

struct FixedIntegrator<const LEAK_BITS: u32, const GAIN_BITS: u32>;

impl<const LEAK_BITS: u32, const GAIN_BITS: u32> Model for FixedIntegrator<LEAK_BITS, GAIN_BITS> {
    const N_PARAMS: usize = 0;
    const N_STATE: usize = 1;
    const HAS_SPIKE: bool = false;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let leak = f32::from_bits(LEAK_BITS);
        let gain = f32::from_bits(GAIN_BITS);
        ctx.set_s(0, ctx.s(0) * S::splat(leak) + ctx.i() * S::splat(gain));
    }
}

/// Act = Act·leak + Inp·gain, expressed in safe BioVM instructions.
fn program(leak: f32, gain: f32) -> BioProgram {
    let mut program = BioProgram::new(2, 0);
    let steps = [
        Instruction::LoadState {
            dst: reg::ACT,
            state: slot::ACT,
        },
        Instruction::Const {
            dst: reg::KONST,
            value: leak,
        },
        Instruction::Mul {
            dst: reg::ACT,
            lhs: reg::ACT,
            rhs: reg::KONST,
        },
        Instruction::LoadState {
            dst: reg::INP,
            state: slot::INP,
        },
        Instruction::Const {
            dst: reg::KONST,
            value: gain,
        },
        Instruction::Mul {
            dst: reg::TERM,
            lhs: reg::INP,
            rhs: reg::KONST,
        },
        Instruction::Add {
            dst: reg::ACT,
            lhs: reg::ACT,
            rhs: reg::TERM,
        },
        Instruction::StoreState {
            state: slot::ACT,
            src: reg::ACT,
        },
    ];
    for step in steps {
        program.push(step).expect("static layout is valid");
    }
    program
}

fn main() {
    let input = vec![0.5f32; N];
    let p_a = program(0.90, 0.30);
    let p_b = program(0.82, 0.42);

    let mut v = vec![0.0f32; N];
    let start = Instant::now();
    for _ in 0..STEPS {
        unsafe {
            run_neuron::<FixedIntegrator<{ 0.90f32.to_bits() }, { 0.30f32.to_bits() }>>(
                v.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                input.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null(),
                N,
            );
        }
    }
    let pillars_fixed = start.elapsed();

    // BioVM: states are column-major over `slot::COUNT` slots.
    let slots = 2usize;

    let mut states = vec![0.0f32; slots * N];
    states[slot::INP * N..].fill(0.5);
    let start = Instant::now();
    for _ in 0..STEPS {
        p_a.execute(&mut states, &[], N);
    }
    let biovm_fixed = start.elapsed();

    let mut swapped = vec![0.0f32; slots * N];
    swapped[slot::INP * N..].fill(0.5);
    let start = Instant::now();
    for step in 0..STEPS {
        let program = if step / SWAP_EVERY % 2 == 0 {
            &p_a
        } else {
            &p_b
        };
        program.execute(&mut swapped, &[], N);
    }
    let biovm_swap_precompiled = start.elapsed();

    let mut swapped_compile = vec![0.0f32; slots * N];
    swapped_compile[slot::INP * N..].fill(0.5);
    let start = Instant::now();
    for step in 0..STEPS {
        let (leak, gain) = if step / SWAP_EVERY % 2 == 0 {
            (0.90, 0.30)
        } else {
            (0.82, 0.42)
        };
        let program = program(leak, gain);
        program.execute(&mut swapped_compile, &[], N);
    }
    let biovm_swap_compile = start.elapsed();

    let work = N as f64 * STEPS as f64;
    println!("N={N}, steps={STEPS}, swap_every={SWAP_EVERY}");
    for (name, elapsed) in [
        ("pillars fixed", pillars_fixed),
        ("BioVM fixed", biovm_fixed),
        ("BioVM swap, precompiled", biovm_swap_precompiled),
        ("BioVM swap, compile each step", biovm_swap_compile),
    ] {
        println!(
            "{name:30} {:>8.3} s  {:>8.1} M cell-steps/s",
            elapsed.as_secs_f64(),
            work / elapsed.as_secs_f64() / 1e6
        );
    }
    println!(
        "checksums: {:.6} {:.6} {:.6}",
        black_box(v.iter().sum::<f32>()),
        black_box(states.iter().sum::<f32>()),
        black_box(swapped.iter().sum::<f32>() + swapped_compile.iter().sum::<f32>())
    );
}
