//! GPU vs CPU parity for the conductance-based GPEA worker
//! (`GpeaGpuSynaptic` — receptor types, g·(E_rev−V), NMDA Mg²⁺ block).
//!
//! The neuron is bit-identical between CPU and GPU (`gpea_nvrtc`). The
//! conductance scatter is an atomic push — at *controlled* activity the
//! divergence stays tiny (the full-brain run on fafb_brain_ntfixed agreed to
//! ~1.5-3% even in the chaotic near-seizure regime). This test asserts
//! tolerance parity on a synthetic connectome with mixed E/I receptor types.
//! Requires the `cuda` feature + GPU.
#![cfg(feature = "cuda")]

use pillars::core::NeuralPopulation;
use pillars::kernels::synapse::{DecayFactors, ReceptorType, SynapticKinetics};
use pillars::zoo::GpeaModel;
use pillars_connectome::ConnectomeData;
use pillars_cuda::{GpeaGpuSynaptic, GpeaParams};

/// Deterministic xorshift64 random connectome (same as alpha_lif_gpu_parity),
/// with per-edge receptor types: 60% excitatory (nAChR=4 / AMPA=0), 40%
/// inhibitory (GABA_A=2 / GluCl=5) — insect-appropriate split.
fn random_connectome(n: usize, deg: usize, w_scale: f32, seed: u64) -> ConnectomeData {
    let mut x = seed | 1;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut row_ptr = vec![0i32; n + 1];
    let mut col = Vec::with_capacity(n * deg);
    let mut val = Vec::with_capacity(n * deg);
    let mut rt = Vec::with_capacity(n * deg);
    for pre in 0..n {
        for _ in 0..deg {
            let post = (next() % n as u64) as i32;
            let mag = w_scale * (0.5 + (next() % 1000) as f32 / 1000.0);
            let w = if (next() & 1) == 0 { mag } else { -mag };
            // receptor: nAChR (4, exc) or AMPA (0, exc) vs GABA_A (2) / GluCl (5)
            let r: u8 = match next() % 10 {
                0..=2 => 0,  // AMPA (exc)
                3..=5 => 4,  // nAChR (exc)
                6..=8 => 2,  // GABA_A (inh)
                _ => 5,      // GluCl (inh)
            };
            col.push(post);
            val.push(w);
            rt.push(r);
        }
        row_ptr[pre + 1] = col.len() as i32;
    }
    ConnectomeData::new(row_ptr, col, val, vec![], vec![])
}

/// CPU reference — the exact `hh_synaptic` loop (SynapticKinetics), same as the
/// real CPU runner: decay → I = Σ g·(E_rev−V) → neuron step → immediate jump.
fn cpu_reference(
    conn: &ConnectomeData,
    params: GpeaParams,
    ext: &[f32],
    steps: usize,
) -> (Vec<f32>, i64) {
    let n = conn.n_neurons;
    let param_vals = [
        params.dt_ms,
        params.celsius,
        params.inv_cm,
        params.v_thresh,
        params.area,
        params.gnabar,
        params.gkdrbar,
        params.gl,
        params.el,
        params.gcatbar,
        params.gkcabar,
        params.kca,
        params.vol,
        params.ca_gain,
    ];
    let init = [-60.0f32, 0.0414, 0.906, 0.2165, 0.355, 0.01325, 5e-5, 0.1821261, 1.0];

    let mut pop = NeuralPopulation::<GpeaModel>::new(n);
    for (c, &v) in param_vals.iter().enumerate() {
        pop.fill_param(c, v);
    }
    for (c, &v) in init.iter().enumerate() {
        pop.fill_state(c, v);
    }

    let mut kinetics = SynapticKinetics::new(n);
    let factors = DecayFactors::new(params.dt_ms);
    let mut i_in = vec![0.0f32; n];
    let mut spikes = vec![0i32; n];
    let mut total = 0i64;

    for _ in 0..steps {
        kinetics.decay_all(&factors);
        i_in[..n].fill(0.0);
        for (i, &e) in ext.iter().enumerate().take(n) {
            i_in[i] += e;
        }
        kinetics.compute_currents(pop.state_slice(0), &mut i_in);
        pop.step(&i_in, &mut spikes);
        total += spikes.iter().map(|&s| s as i64).sum::<i64>();
        for pre in 0..n {
            if spikes[pre] == 0 {
                continue;
            }
            let start = conn.row_ptr[pre] as usize;
            let end = conn.row_ptr[pre + 1] as usize;
            for k in start..end {
                kinetics.jump(
                    conn.col_indices[k] as usize,
                    ReceptorType::from_u8(conn.receptor_types[k]),
                    conn.values[k],
                );
            }
        }
    }

    (pop.state_slice(0)[..n].to_vec(), total)
}

#[test]
fn gpea_gpu_syn_parity() {
    let n = 1024;
    let mut conn = random_connectome(n, 12, 0.35, 0xFEED_BEEF);
    conn.build_receptor_types(); // keeps the already-set receptor_types (non-empty)
    assert!(!conn.receptor_types.is_empty());

    let params = GpeaParams::flywire();

    // Gentle tonic drive on a few neurons — controlled activity keeps the
    // atomic-scatter ordering noise below the spike-flip threshold.
    let mut ext = vec![0.0f32; n];
    for i in 0..16 {
        ext[i] = 12.0;
    }

    let mut gpu = GpeaGpuSynaptic::new(
        &conn.row_ptr,
        &conn.col_indices,
        &conn.values,
        &conn.receptor_types,
        params,
    )
    .expect("GPU synaptic worker construction");

    let steps = (0.05 * 1000.0 / params.dt_ms).round() as usize; // 50 ms
    let gpu_out = gpu.run_steps(steps, &ext).expect("GPU run");
    let gpu_v = gpu.download_v().expect("GPU v download");

    let (cpu_v, cpu_spikes) = cpu_reference(&conn, params, &ext, steps);

    assert_eq!(gpu_v.len(), n);
    assert!(
        gpu_out.total_spikes > 0,
        "worker must produce activity under tonic drive"
    );

    let d_spikes = (cpu_spikes - gpu_out.total_spikes as i64).abs();
    assert!(
        d_spikes <= 8,
        "spike counts drifted too far: CPU={cpu_spikes} GPU={}",
        gpu_out.total_spikes
    );

    let mut max_abs = 0.0f32;
    let mut over_1mv = 0usize;
    for i in 0..n {
        let d = (cpu_v[i] - gpu_v[i]).abs();
        max_abs = max_abs.max(d);
        if d > 1.0 {
            over_1mv += 1;
        }
    }
    assert!(
        over_1mv <= n / 50,
        "too many neurons diverged >1 mV: {over_1mv}/{n} (max|Δv|={max_abs:.2})"
    );
}
