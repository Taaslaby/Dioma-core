//! Бенч: wgpu-движок vs CPU-эталон на масштабе диомы.
//! N нейронов (дефолт 120k = 1000 существ × 120), ~10 синапсов/нейрон.
//! Использование: ENGINE_BENCH_N=120000 cargo run -p pillars-wgpu --example engine_bench

use pillars_wgpu::{WgpuContext, WgpuNetwork, WgpuPop, LIF_WGSL};
use std::time::Instant;

fn main() {
    let n = std::env::var("ENGINE_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120_000usize);
    let syn_per = 10usize;
    let dt = 0.5f32;
    let stride = n.div_ceil(64) * 64;
    println!("N={n} stride={stride} synapses≈{}", n * syn_per);

    // ── состояние: все -70, пара нейронов подпорогово (запуск) ──
    let mut states = vec![0.0f32; 2 * stride];
    for i in 0..n.min(1024) {
        states[i] = -54.6;
    }
    let mut params = vec![0.0f32; 6 * stride];
    for i in 0..stride {
        params[0 * stride + i] = dt;
        params[1 * stride + i] = 0.1;
        params[2 * stride + i] = -70.0;
        params[3 * stride + i] = -54.0;
        params[4 * stride + i] = -70.0;
        params[5 * stride + i] = 2.0;
    }

    // ── CSR: каждый i → (i+1) и (i+3)%n + случайные связи ──
    let mut row_ptr = vec![0u32; n + 1];
    let mut col_idx = Vec::new();
    let mut weights = Vec::new();
    let mut delays = Vec::new();
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for pre in 0..n {
        col_idx.push(((pre + 1) % n) as u32);
        weights.push(1.0);
        delays.push(1u8);
        for _ in 0..syn_per - 1 {
            let post = (rng() as usize) % n;
            col_idx.push(post as u32);
            weights.push(0.5);
            delays.push(1 + (pre % 5) as u8);
        }
        row_ptr[pre + 1] = col_idx.len() as u32;
    }
    let n_syn = col_idx.len();

    // ── GPU ──
    let ctx = match WgpuContext::new_blocking() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("NO_ADAPTER: {e}");
            return;
        }
    };
    println!("ADAPTER: {} ({:?})", ctx.adapter_name, ctx.device_type);
    let pop = WgpuPop {
        wgsl: LIF_WGSL,
        entry: "lif_kernel_wgsl",
        n,
        ns: 2,
        np: 6,
        states: states.clone(),
        params: params.clone(),
    };
    let mut net = WgpuNetwork::new(ctx, &[pop], &row_ptr, &col_idx, &weights, &delays, dt).expect("net");

    // прогрев
    for _ in 0..5 {
        net.step().expect("warmup");
    }
    let steps = 100usize;
    let t0 = Instant::now();
    for _ in 0..steps {
        net.step().expect("step");
    }
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    println!("GPU: {gpu_ms:.3} ms/step  ({:.0} steps/s)", 1000.0 / gpu_ms);

    // ── CPU-эталон (LIF + ring + scatter) ──
    let mut c_states = states.clone();
    let mut ring = vec![0.0f32; 32 * n];
    let mut slot = 0usize;
    let mut total = 0u32;
    let t0 = Instant::now();
    let cpu_steps = steps.min(20); // CPU медленнее — меньше шагов для замера
    for _ in 0..cpu_steps {
        // copy
        let mut i_in = vec![0.0f32; n];
        for i in 0..n {
            i_in[i] = ring[slot * n + i];
        }
        // pop
        let mut spikes = vec![0u32; n];
        for i in 0..n {
            let v = c_states[i];
            let mut refr = c_states[stride + i];
            if refr > 0.0 {
                refr = (refr - dt).max(0.0);
                c_states[stride + i] = refr;
                continue;
            }
            let dv = (params[2 * stride + i] - v + i_in[i]) * params[1 * stride + i] * params[0 * stride + i];
            let v_new = v + dv;
            if v_new >= params[3 * stride + i] {
                spikes[i] = 1;
                c_states[i] = params[4 * stride + i];
                c_states[stride + i] = params[5 * stride + i];
            } else {
                c_states[i] = v_new;
                c_states[stride + i] = 0.0;
            }
        }
        total += spikes.iter().sum::<u32>();
        for i in 0..n {
            ring[slot * n + i] = 0.0;
        }
        for pre in 0..n {
            if spikes[pre] == 0 {
                continue;
            }
            for k in row_ptr[pre] as usize..row_ptr[pre + 1] as usize {
                let post = col_idx[k] as usize;
                let d = delays[k].max(1) as usize;
                let tgt = (slot + d) % 32;
                ring[tgt * n + post] += weights[k];
            }
        }
        slot = (slot + 1) % 32;
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / cpu_steps as f64;
    println!("CPU: {cpu_ms:.3} ms/step  ({:.0} steps/s)", 1000.0 / cpu_ms);
    println!("SPEEDUP: {:.1}×  (cpu_spikes={total})", cpu_ms / gpu_ms);
}
