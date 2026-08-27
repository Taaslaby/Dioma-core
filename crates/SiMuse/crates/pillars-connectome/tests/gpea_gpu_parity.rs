//! GPU vs CPU parity for the GPEA biophysical fly-brain worker.
//!
//! The neuron is **bit-identical** between CPU and GPU (proven in
//! `pillars-cuda`'s `gpea_nvrtc` test: exact spike raster over 100 ms of
//! 1100 Hz firing). The recurrent scatter is a spike-driven **atomic** push
//! (like PyTorch / NEST / GeNN), so its summation order is not bit-
//! reproducible — at *controlled* activity the divergence stays tiny (the
//! full-brain run at gain ≤ 10 reproduced the CPU spike count exactly over
//! 50 ms); this test asserts spike-count and membrane-potential *tolerance*
//! parity on a synthetic connectome. Requires the `cuda` feature + GPU.
#![cfg(feature = "cuda")]

use pillars::core::NeuralPopulation;
use pillars::zoo::GpeaModel;
use pillars_connectome::ConnectomeData;
use pillars_cuda::{GpeaGpu, GpeaParams};

/// Deterministic xorshift64 random connectome (same as alpha_lif_gpu_parity).
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
    for pre in 0..n {
        for _ in 0..deg {
            let post = (next() % n as u64) as i32;
            let mag = w_scale * (0.5 + (next() % 1000) as f32 / 1000.0);
            let w = if (next() & 1) == 0 { mag } else { -mag };
            col.push(post);
            val.push(w);
        }
        row_ptr[pre + 1] = col.len() as i32;
    }
    ConnectomeData::new(row_ptr, col, val, vec![], vec![])
}

/// CPU reference with the same current-jump ring semantics as the GPU worker
/// (ext inject → read+zero slot → neuron step → scatter previous spikes →
/// advance). Mirrors the GPU graph order exactly, so at low activity the two
/// must agree.
fn cpu_reference(
    conn: &ConnectomeData,
    values: &[f32],
    params: GpeaParams,
    ext: &[f32],
    steps: usize,
) -> (Vec<f32>, i64) {
    let n = conn.n_neurons;
    let size = (params.t_delay / params.dt_ms).round().max(0.0) as usize + 1;
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

    let mut ring = vec![0.0f32; size * n];
    let mut spikes_cur = vec![0i32; n];
    let mut spikes_prev = vec![0i32; n];
    let mut ring_pos = size - 1;
    let mut total = 0i64;
    let mut i_syn = vec![0.0f32; n];

    for _ in 0..steps {
        if !ext.is_empty() {
            let slot = &mut ring[ring_pos * n..(ring_pos + 1) * n];
            for i in 0..n {
                slot[i] += ext[i];
            }
        }
        {
            let slot = &mut ring[ring_pos * n..(ring_pos + 1) * n];
            for i in 0..n {
                i_syn[i] = slot[i];
                slot[i] = 0.0;
            }
        }
        pop.step(&i_syn, &mut spikes_cur);
        total += spikes_cur.iter().map(|&s| s as i64).sum::<i64>();
        let slot = &mut ring[ring_pos * n..(ring_pos + 1) * n];
        for pre in 0..n {
            if spikes_prev[pre] == 0 {
                continue;
            }
            let start = conn.row_ptr[pre] as usize;
            let end = conn.row_ptr[pre + 1] as usize;
            for k in start..end {
                slot[conn.col_indices[k] as usize] += values[k];
            }
        }
        ring_pos = (ring_pos + 1) % size;
        std::mem::swap(&mut spikes_cur, &mut spikes_prev);
    }

    (pop.state_slice(0)[..n].to_vec(), total)
}

#[test]
fn gpea_gpu_parity() {
    let n = 1024; // multiple of 8 → stride == n
    let conn = random_connectome(n, 12, 2.0, 0x5EED_F00D);
    let params = GpeaParams::flywire();

    // Drive a handful of neurons with a tonic current so the network has
    // activity, but keep the weights modest so it stays in the regime where
    // atomic-scatter ordering noise does not yet flip spikes (gain ≤ 10 in
    // the full-brain experiment was exactly reproducible).
    let mut ext = vec![0.0f32; n];
    for i in 0..16 {
        ext[i] = 15.0;
    }

    let mut gpu = GpeaGpu::new(&conn.row_ptr, &conn.col_indices, &conn.values, params)
        .expect("GPU worker construction");

    let steps = (0.05 * 1000.0 / params.dt_ms).round() as usize; // 50 ms
    let gpu_out = gpu.run_steps(steps, &ext).expect("GPU run");
    let gpu_v = gpu.download_v().expect("GPU v download");

    let (cpu_v, cpu_spikes) = cpu_reference(&conn, &conn.values, params, &ext, steps);

    assert_eq!(gpu_v.len(), n);

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
