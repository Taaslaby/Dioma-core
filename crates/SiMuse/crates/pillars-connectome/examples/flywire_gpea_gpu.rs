//! Full FlyWire brain on the biophysical GPEA neuron — GPU vs CPU.
//!
//! The same graph as `flywire_sugar_gpu` / `flywire_hh` (flywire_783.taas:
//! 138 639 neurons, 15 091 983 synapses), but the neuron is **GPEA** — the
//! moth-antennal-lobe HH set (Na/Kdr/CaT/AHP + Ca dynamics) used for the
//! Drosophila whole-brain model (Higuchi et al., bioRxiv 2022), validated
//! against NEURON 9.0.2 in `pillars::zoo::neurons::gpea`.
//!
//! Synaptic model: current-based (µA/cm² jumps on the CSR `values`, like the
//! alpha-LIF worker) — the CPU HH path's conductance kinetics is a separate
//! concern. The sugar GRNs are driven with a constant 20 µA/cm² holding
//! current (as in `flywire_hh`: HH needs tonic drive to spike, unlike LIF).
//!
//! ```bash
//! cargo run -p pillars-connectome --features cuda --release \
//!     --example flywire_gpea_gpu -- resources/data/flywire_783.taas 0.1
//! ```

use std::time::Instant;

use pillars::core::NeuralPopulation;
use pillars::kernels::synapse::{DecayFactors, ReceptorType, SynapticKinetics};
use pillars::zoo::GpeaModel;
use pillars_connectome::ConnectomeData;
use pillars_cuda::{GpeaGpu, GpeaGpuSynaptic, GpeaParams};

/// 21 sugar GRNs (from resources/data/sugar_grn.json).
const SUGAR_GRN: [usize; 21] = [
    69093, 97602, 122795, 124291, 29281, 100605, 110469, 51107, 49584, 129730, 126873, 28825,
    126600, 126752, 32863, 108426, 111357, 14842, 90589, 92298, 12494,
];

/// Constant holding current on the GRNs, µA/cm² (as in flywire_hh).
const I_GRN: f32 = 20.0;

/// Default synaptic weight gain for the CURRENT-based path: the raw .taas
/// `values` are calibrated for mV-scale LIF drive; GPEA (HH-class, µA/cm²)
/// needs a stronger current jump per synapse. `50` puts the mean at ~9.5
/// µA/cm² per spike. **`gain = 0` switches to the conductance-based path**
/// (`GpeaGpuSynaptic`, receptor types, g·(E_rev−V)) where values are
/// conductances used verbatim — no gain needed.
const DEFAULT_GAIN: f32 = 50.0;

/// CPU reference with the SAME current-jump ring semantics as the GPU worker:
/// ext inject → read+zero ring slot → neuron step → scatter previous spikes →
/// advance ring. Returns final V and the total spike count.
/// `values` are the (already scaled) per-synapse current jumps — must match
/// what the GPU worker received.
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
        // 1. external inject into the CURRENT slot (i_syn semantics)
        if !ext.is_empty() {
            let slot = &mut ring[ring_pos * n..(ring_pos + 1) * n];
            for i in 0..n {
                slot[i] += ext[i];
            }
        }
        // 2. read + zero the slot, then the neuron step
        {
            let slot = &mut ring[ring_pos * n..(ring_pos + 1) * n];
            for i in 0..n {
                i_syn[i] = slot[i];
                slot[i] = 0.0;
            }
        }
        pop.step(&i_syn, &mut spikes_cur);
        total += spikes_cur.iter().map(|&s| s as i64).sum::<i64>();
        // 3. scatter PREVIOUS spikes (double-buffer parity) into this slot
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
        // 4. advance
        ring_pos = (ring_pos + 1) % size;
        std::mem::swap(&mut spikes_cur, &mut spikes_prev);
    }

    (pop.state_slice(0)[..n].to_vec(), total)
}

/// CPU reference for the CONDUCTANCE-based path — mirrors the CPU `hh_synaptic`
/// loop exactly (decay → I = Σ g·(E_rev−V) with NMDA Mg²⁺ block → neuron step →
/// immediate conductance jump), using the same `SynapticKinetics` the real CPU
/// runner uses.
fn cpu_reference_conductance(
    conn: &ConnectomeData,
    values: &[f32],
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
        // immediate conductance jump (no delay in the conductance path)
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
                    values[k],
                );
            }
        }
    }

    (pop.state_slice(0)[..n].to_vec(), total)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "resources/data/flywire_783.taas".into());
    let t_run_s: f32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(0.05);
    let gain: f32 = args
        .get(3)
        .map(|s| s.parse().unwrap())
        .unwrap_or(DEFAULT_GAIN);
    // Conductance-mode value normalization: the NT-fixed FAFB/CNS graphs carry
    // synapse COUNTS as weights (mean ≈ 194) — the repo convention divides by
    // 256 (see scripts/check_physio.py NORM=256). flywire_783 is already
    // normalized (mean 0.19) → norm = 1.0.
    let norm: f32 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(1.0);

    let t0 = Instant::now();
    let mut conn = ConnectomeData::load_baked(&mut std::io::BufReader::new(
        std::fs::File::open(&path).expect("cannot open .taas"),
    ))
    .expect("bad .taas");
    let setup_s = t0.elapsed().as_secs_f64();

    // Synaptic weight stats (values = µA/cm² jumps in this current-based model)
    let (mut vmin, mut vmax, mut vsum) = (f32::MAX, f32::MIN, 0.0f64);
    for &v in &conn.values {
        vmin = vmin.min(v);
        vmax = vmax.max(v);
        vsum += v as f64;
    }
    println!(
        "[setup] n_neurons={} n_synapses={} load={:.3}s values[min={vmin:.4} max={vmax:.4} mean={:.6}]",
        conn.n_neurons,
        conn.n_synapses,
        setup_s,
        vsum / conn.values.len() as f64
    );

    let params = GpeaParams::flywire();

    let mut ext = vec![0.0f32; conn.n_neurons];
    for &g in &SUGAR_GRN {
        ext[g] = I_GRN;
    }
    let steps = (t_run_s * 1000.0 / params.dt_ms).round() as usize;

    if gain == 0.0 {
        // ── Conductance-based path: receptor types, g·(E_rev−V), no gain ──
        conn.build_receptor_types();
        let (mut n_exc, mut n_inh) = (0usize, 0usize);
        for &r in &conn.receptor_types {
            match r {
                0 | 1 | 4 => n_exc += 1,
                _ => n_inh += 1,
            }
        }
        println!(
            "[syn] receptor_types built: {}/{} (exc {n_exc}, inh {n_inh})",
            conn.receptor_types.len(),
            conn.n_synapses
        );

        let t1 = Instant::now();
        let conn_vals: Vec<f32> = conn.values.iter().map(|&v| v / norm).collect();
        let mut gpu = GpeaGpuSynaptic::new(
            &conn.row_ptr,
            &conn.col_indices,
            &conn_vals,
            &conn.receptor_types,
            params,
        )
        .expect("GPU synaptic worker init (NVRTC + device alloc)");
        println!(
            "[gpu] init={:.3}s (NVRTC + CSR upload, conductance mode, dt={}ms)",
            t1.elapsed().as_secs_f64(),
            params.dt_ms
        );

        let tg = Instant::now();
        let gout = gpu.run_steps(steps, &ext).expect("GPU run");
        let gpu_s = tg.elapsed().as_secs_f64();

        let tc = Instant::now();
        let (cpu_v, cpu_spikes) = cpu_reference_conductance(&conn, &conn_vals, params, &ext, steps);
        let cpu_s = tc.elapsed().as_secs_f64();

        println!(
            "[gpu] t={t_run_s}s spikes={} active={} wall={:.3}s (ratio {:.3}x)",
            gout.total_spikes,
            gout.active_neurons,
            gpu_s,
            t_run_s as f64 / gpu_s
        );
        println!(
            "[cpu] t={t_run_s}s spikes={cpu_spikes} wall={:.3}s (ratio {:.3}x)",
            cpu_s,
            t_run_s as f64 / cpu_s
        );
        println!("[speedup] {:.2}x", cpu_s / gpu_s);

        let gpu_v = gpu.download_v().expect("GPU v download");
        let mut max_abs = 0.0f32;
        let mut over_1mv = 0usize;
        for i in 0..conn.n_neurons {
            let d = (cpu_v[i] - gpu_v[i]).abs();
            max_abs = max_abs.max(d);
            if d > 1.0 {
                over_1mv += 1;
            }
        }
        let d_spikes = (cpu_spikes - gout.total_spikes as i64).abs();
        println!(
            "[parity] spike_count_diff={d_spikes} max|Δv|={max_abs:.3} mV (>1 mV: {over_1mv}/{})",
            conn.n_neurons
        );
        return;
    }

    // ── Current-based path (default): gain-scaled µA/cm² jumps ──
    let scaled: Vec<f32> = conn.values.iter().map(|&v| v * gain).collect();
    let t1 = Instant::now();
    let mut gpu = GpeaGpu::new(&conn.row_ptr, &conn.col_indices, &scaled, params)
        .expect("GPU worker init (NVRTC + device alloc + CSR upload)");
    println!(
        "[gpu] init={:.3}s (NVRTC compile + CSR upload, ring={} slots, gain={gain})",
        t1.elapsed().as_secs_f64(),
        (params.t_delay / params.dt_ms).round() as usize + 1
    );

    let tg = Instant::now();
    let gout = gpu.run_steps(steps, &ext).expect("GPU run");
    let gpu_s = tg.elapsed().as_secs_f64();

    let tc = Instant::now();
    let (cpu_v, cpu_spikes) = cpu_reference(&conn, &scaled, params, &ext, steps);
    let cpu_s = tc.elapsed().as_secs_f64();

    println!(
        "[gpu] t={t_run_s}s spikes={} active={} wall={:.3}s (ratio {:.3}x)",
        gout.total_spikes,
        gout.active_neurons,
        gpu_s,
        t_run_s as f64 / gpu_s
    );
    println!(
        "[cpu] t={t_run_s}s spikes={cpu_spikes} wall={:.3}s (ratio {:.3}x)",
        cpu_s,
        t_run_s as f64 / cpu_s
    );
    println!("[speedup] {:.2}x", cpu_s / gpu_s);

    // ── Parity (atomic scatter vs serial: tolerance, not bits) ──
    let gpu_v = gpu.download_v().expect("GPU v download");
    let mut max_abs = 0.0f32;
    let mut over_1mv = 0usize;
    for i in 0..conn.n_neurons {
        let d = (cpu_v[i] - gpu_v[i]).abs();
        max_abs = max_abs.max(d);
        if d > 1.0 {
            over_1mv += 1;
        }
    }
    let d_spikes = (cpu_spikes - gout.total_spikes as i64).abs();
    println!(
        "[parity] spike_count_diff={d_spikes} max|Δv|={max_abs:.3} mV (>1 mV: {over_1mv}/{})",
        conn.n_neurons
    );
}
