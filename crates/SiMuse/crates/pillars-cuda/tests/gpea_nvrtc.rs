//! GPEA (biophysical fly neuron, Higuchi 2022) on CUDA.
//!
//! 1. The `#[dual_neuron]`-transpiled `GPEA_CUDA_SOURCE` must compile through
//!    NVRTC (host-side — verifies the `ln` Nernst path and `#[no_clamp]`).
//! 2. CPU SIMD vs CUDA kernel parity from identical state/params.
//!
//! Requires a CUDA toolkit (NVRTC) — same as the HH parity tests.

use pillars::core::population::NeuralPopulation;
use pillars::prelude::Model;
use pillars::zoo::{GpeaModel, GPEA_CUDA_SOURCE};
use pillars_cuda::engine::nvrtc_compile_check;
use pillars_cuda::CompiledModel;

const N: usize = 64;
const DT: f32 = 0.025;
const REST_STEPS: usize = 1000; // 25 ms at rest (smooth, spike-free)
const SPIKE_STEPS: usize = 4000; // 100 ms at 500 µA/cm² (STRESS regime)
// STRESS regime: 500 µA/cm² (0.5 mA/cm²) — the NEURON-validated strong-input
// case (220 spikes/200 ms per neuron), exercises continuous spiking through
// the kernel, including the Ca dynamics + ln() Nernst path.
const I_INJ: f32 = 500.0;
const I_REST: f32 = 0.0;

fn fill_params() -> [f32; GpeaModel::N_PARAMS] {
    // Order matches the #[param] fields of GpeaModel (zoo/neurons/gpea.rs):
    // dt, celsius, inv_cm, v_thresh, area, gnabar, gkdrbar, gl, el, gcatbar,
    // gkcabar, kca, vol, caGain
    let mut p = [0.0f32; GpeaModel::N_PARAMS];
    p[0] = DT;
    p[1] = 24.0; // Drosophila
    p[2] = 1.0;
    p[3] = -20.0;
    p[4] = 2513.2741228718345; // π·40·20 µm² (NEURON reference geometry)
    p[5] = 49.0;
    p[6] = 57.0;
    p[7] = 0.35;
    p[8] = -60.0;
    p[9] = 5.0;
    p[10] = 1.0;
    p[11] = 2.0;
    p[12] = 3.355e-11;
    p[13] = 0.1;
    p
}

fn init_states() -> [f32; GpeaModel::N_STATE] {
    // NEURON finitialize IC (see zoo/neurons/gpea.rs make_pop): gates at
    // V=-60, cai=5e-5 (WRITE-ion reset), r=r_inf(5e-5), was_below=1.
    [-60.0, 0.0414, 0.906, 0.2165, 0.355, 0.01325, 5e-5, 0.1821261, 1.0]
}

#[test]
fn gpea_cuda_source_compiles_via_nvrtc() {
    nvrtc_compile_check(GPEA_CUDA_SOURCE)
        .expect("GPEA_CUDA_SOURCE must produce valid PTX via NVRTC");
}

#[test]
fn gpea_cpu_simd_vs_cuda_parity() {
    let params = fill_params();
    let ic = init_states();
    let ns = GpeaModel::N_STATE;
    let np = GpeaModel::N_PARAMS;

    // ── CPU side: pillars NeuralPopulation ────────────────────────
    let mut pop = NeuralPopulation::<GpeaModel>::new(N);
    for (c, &v) in ic.iter().enumerate() {
        pop.fill_state(c, v);
    }
    for (c, &v) in params.iter().enumerate() {
        pop.fill_param(c, v);
    }

    // ── GPU side: identical SoA snapshot ──────────────────────────
    let mut g_states = vec![0.0f32; ns * N];
    for c in 0..ns {
        g_states[c * N..(c + 1) * N].copy_from_slice(pop.state_slice(c));
    }
    let g_params: Vec<f32> = (0..np).flat_map(|c| pop.param_slice(c).to_vec()).collect();
    let i_in = vec![I_INJ; N];
    let i_rest = vec![I_REST; N];

    let model = CompiledModel::new(GPEA_CUDA_SOURCE, "gpea_kernel", ns, np, N, N)
        .expect("failed to compile/load the dual_neuron GPEA CUDA kernel (CUDA device required)");
    let mut g_spikes = vec![0i32; N];
    let mut spikes = vec![0i32; N];

    // Phase 1 (rest): smooth relaxation toward the −61 mV equilibrium —
    // spike-free, so any divergence here means an equation mismatch.
    for _ in 0..REST_STEPS {
        pop.step(&i_rest, &mut spikes);
        model
            .step(&mut g_states, &g_params, &i_rest, &mut g_spikes, DT)
            .expect("gpu step");
        for c in 0..ns {
            for i in 0..N {
                let d = (pop.state_slice(c)[i] - g_states[c * N + i]).abs();
                assert!(
                    d < 2e-2,
                    "rest parity broken at state[{c}][{i}]: CPU={} GPU={}",
                    pop.state_slice(c)[i],
                    g_states[c * N + i]
                );
            }
        }
    }

    // Phase 2 (500 µA/cm²): per-neuron spike counts must match exactly
    // (the robust metric in the spiking regime — CPU fast_exp poly vs GPU
    // SFU exp2f differ by ~1 ulp/step, so V can drift slightly while the
    // spike raster stays identical).
    let mut cpu_counts = vec![0i64; N];
    let mut gpu_counts = vec![0i64; N];
    for _ in 0..SPIKE_STEPS {
        pop.step(&i_in, &mut spikes);
        for i in 0..N {
            cpu_counts[i] += spikes[i] as i64;
        }
        model
            .step(&mut g_states, &g_params, &i_in, &mut g_spikes, DT)
            .expect("gpu step");
        for i in 0..N {
            gpu_counts[i] += g_spikes[i] as i64;
        }
    }
    let cpu_spikes: i64 = cpu_counts.iter().sum();
    let gpu_spikes: i64 = gpu_counts.iter().sum();
    assert!(
        cpu_spikes > 0 && gpu_spikes > 0,
        "phase 2 must exercise spikes: CPU={cpu_spikes} GPU={gpu_spikes}"
    );
    assert_eq!(cpu_counts, gpu_counts, "per-neuron spike count mismatch");

    // Final-state parity (V and cai — the Ca dynamics are the GPEA-specific
    // part). After 100 ms of 1100 Hz spiking the CPU-poly vs GPU-SFU exp noise
    // accumulates through the chaotic trajectory; an exact raster match (above)
    // is the strong claim, this bounds the drift to ≪ spike amplitude (~60 mV).
    let mut max_dv = 0.0f32;
    for i in 0..N {
        max_dv = max_dv.max((pop.state_slice(0)[i] - g_states[i]).abs());
        max_dv = max_dv.max((pop.state_slice(6)[i] - g_states[6 * N + i]).abs());
    }
    assert!(
        max_dv < 2.0,
        "final V/cai divergence too large: max Δ = {max_dv} mV"
    );
}
