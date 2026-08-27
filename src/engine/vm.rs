//! Compact register VM used by BioVM and the Pythia compiler.

use wide::{f32x8, CmpGt};

const REGISTER_COUNT: usize = 16;

#[derive(Clone, Copy, Debug)]
pub enum Opcode {
    LoadState(u8, usize),
    StoreState(usize, u8),
    LoadParam(u8, usize),
    LoadConst(u8, f32),
    MathAdd(u8, u8, u8),
    MathSub(u8, u8, u8),
    MathMul(u8, u8, u8),
    MathFastExp(u8, u8),
    MathClamp01(u8),
    MathMax(u8, u8, u8),
    MathMin(u8, u8, u8),
    PredOn(u8),
    PredOff,
    PhysOhm(u8, u8, u8, u8),
    PhysDiffusionDecay(u8, f32, f32, f32),
}

#[inline]
fn idx(r: u8) -> usize {
    // Некорректно сгенерированный байткод (будущий компилятор Пифии) не должен
    // улетать в нечитаемый panic по index out of bounds: падаем осмысленно.
    debug_assert!(
        (r as usize) < REGISTER_COUNT,
        "регистр {r} вне диапазона 0..{REGISTER_COUNT}"
    );
    r as usize
}

/// Execute one VM step over a homogeneous cell population.
///
/// # Safety
/// The caller must provide valid column-major state and parameter storage for
/// every slot referenced by the program.
pub unsafe fn execute(states: *mut f32, params: *const f32, n_cells: usize, bytecode: &[Opcode]) {
    let mut offset = 0;
    while offset + 8 <= n_cells {
        unsafe { execute_chunk(states, params, n_cells, offset, bytecode) };
        offset += 8;
    }
    for cell in offset..n_cells {
        unsafe { execute_scalar(states, params, n_cells, cell, bytecode) };
    }
}

unsafe fn execute_chunk(
    states: *mut f32,
    params: *const f32,
    stride: usize,
    offset: usize,
    bytecode: &[Opcode],
) {
    let mut registers = [f32x8::ZERO; REGISTER_COUNT];
    let mut predicate = f32x8::splat(f32::from_bits(u32::MAX));

    for &opcode in bytecode {
        match opcode {
            Opcode::LoadState(dst, slot) => {
                registers[idx(dst)] = unsafe { load8(states.add(slot * stride + offset)) };
            }
            Opcode::StoreState(slot, src) => {
                let ptr = unsafe { states.add(slot * stride + offset) };
                let old = unsafe { load8(ptr) };
                unsafe { store8(ptr, predicate.blend(registers[idx(src)], old)) };
            }
            Opcode::LoadParam(dst, slot) => {
                registers[idx(dst)] = unsafe { load8(params.add(slot * stride + offset)) };
            }
            Opcode::LoadConst(dst, value) => registers[idx(dst)] = f32x8::splat(value),
            Opcode::MathAdd(dst, lhs, rhs) => {
                registers[idx(dst)] = registers[idx(lhs)] + registers[idx(rhs)];
            }
            Opcode::MathSub(dst, lhs, rhs) => {
                registers[idx(dst)] = registers[idx(lhs)] - registers[idx(rhs)];
            }
            Opcode::MathMul(dst, lhs, rhs) => {
                registers[idx(dst)] = registers[idx(lhs)] * registers[idx(rhs)];
            }
            Opcode::MathFastExp(dst, src) => {
                registers[idx(dst)] = registers[idx(src)].exp()
            }
            Opcode::MathClamp01(reg) => {
                registers[idx(reg)] = registers[idx(reg)].max(f32x8::ZERO).min(f32x8::ONE);
            }
            Opcode::MathMax(dst, lhs, rhs) => {
                registers[idx(dst)] = registers[idx(lhs)].max(registers[idx(rhs)]);
            }
            Opcode::MathMin(dst, lhs, rhs) => {
                registers[idx(dst)] = registers[idx(lhs)].min(registers[idx(rhs)]);
            }
            Opcode::PredOn(condition) => {
                predicate = registers[idx(condition)].cmp_gt(f32x8::ZERO);
            }
            Opcode::PredOff => predicate = f32x8::splat(f32::from_bits(u32::MAX)),
            Opcode::PhysOhm(dst, conductance, potential, reversal) => {
                registers[idx(dst)] = registers[idx(conductance)]
                    * (registers[idx(potential)] - registers[idx(reversal)]);
            }
            Opcode::PhysDiffusionDecay(reg, baseline, tau, dt) => {
                registers[idx(reg)] =
                    diffusion_decay(registers[idx(reg)], baseline, tau, dt);
            }
        }
    }
}

unsafe fn execute_scalar(
    states: *mut f32,
    params: *const f32,
    stride: usize,
    cell: usize,
    bytecode: &[Opcode],
) {
    let mut registers = [0.0f32; REGISTER_COUNT];
    let mut predicate = true;
    for &opcode in bytecode {
        match opcode {
            Opcode::LoadState(dst, slot) => {
                registers[idx(dst)] = unsafe { *states.add(slot * stride + cell) };
            }
            Opcode::StoreState(slot, src) if predicate => unsafe {
                *states.add(slot * stride + cell) = registers[idx(src)];
            },
            Opcode::StoreState(_, _) => {}
            Opcode::LoadParam(dst, slot) => {
                registers[idx(dst)] = unsafe { *params.add(slot * stride + cell) };
            }
            Opcode::LoadConst(dst, value) => registers[idx(dst)] = value,
            Opcode::MathAdd(dst, lhs, rhs) => {
                registers[idx(dst)] = registers[idx(lhs)] + registers[idx(rhs)]
            }
            Opcode::MathSub(dst, lhs, rhs) => {
                registers[idx(dst)] = registers[idx(lhs)] - registers[idx(rhs)]
            }
            Opcode::MathMul(dst, lhs, rhs) => {
                registers[idx(dst)] = registers[idx(lhs)] * registers[idx(rhs)]
            }
            Opcode::MathFastExp(dst, src) => {
                registers[idx(dst)] = registers[idx(src)].exp()
            }
            Opcode::MathClamp01(reg) => {
                registers[idx(reg)] = registers[idx(reg)].clamp(0.0, 1.0)
            }
            Opcode::MathMax(dst, lhs, rhs) => {
                registers[idx(dst)] = registers[idx(lhs)].max(registers[idx(rhs)])
            }
            Opcode::MathMin(dst, lhs, rhs) => {
                registers[idx(dst)] = registers[idx(lhs)].min(registers[idx(rhs)])
            }
            Opcode::PredOn(condition) => predicate = registers[idx(condition)] > 0.0,
            Opcode::PredOff => predicate = true,
            Opcode::PhysOhm(dst, conductance, potential, reversal) => {
                registers[idx(dst)] = registers[idx(conductance)]
                    * (registers[idx(potential)] - registers[idx(reversal)]);
            }
            Opcode::PhysDiffusionDecay(reg, baseline, tau, dt) => {
                registers[idx(reg)] = if tau < 1e-6 {
                    baseline
                } else {
                    baseline + (registers[idx(reg)] - baseline) * (-dt / tau).exp()
                };
            }
        }
    }
}

fn diffusion_decay(value: f32x8, baseline: f32, tau: f32, dt: f32) -> f32x8 {
    if tau < 1e-6 {
        return f32x8::splat(baseline);
    }
    let baseline = f32x8::splat(baseline);
    baseline + (value - baseline) * f32x8::splat((-dt / tau).exp())
}

unsafe fn load8(ptr: *const f32) -> f32x8 {
    unsafe { f32x8::from(*ptr.cast::<[f32; 8]>()) }
}

unsafe fn store8(ptr: *mut f32, value: f32x8) {
    unsafe { *ptr.cast::<[f32; 8]>() = value.to_array() };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_and_scalar_tail_execute_same_program() {
        let n = 9;
        let mut states = vec![0.0; 2 * n];
        states[n..].fill(2.0);
        let program = [
            Opcode::LoadState(0, 0),
            Opcode::LoadState(1, 1),
            Opcode::MathAdd(0, 0, 1),
            Opcode::StoreState(0, 0),
        ];

        unsafe { execute(states.as_mut_ptr(), std::ptr::null(), n, &program) };

        assert!(states[..n].iter().all(|&value| value == 2.0));
    }
}
