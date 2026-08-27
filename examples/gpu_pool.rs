//! Кросовер CPU↔GPU по популяции: N существ (мозг 120 нейронов / 964 синапса)
//! в ОДНОЙ сети (один pop N×120, блочно-диагональный CSR — число диспатчей
//! не растёт с N) против CPU-раннера на том же графе. Честный замер, где GPU
//! начинает выигрывать: при малом мозге CPU-неубиваем (лаунч-оверхед диспатчей),
//! при популяции амортизируется.
use dioma::engine::grow::{avoidance_layout, grow};
use dioma::engine::neurons::gains_from;
use dioma::creature::Creature;
use pillars::zoo::neurons::ALPHA_LIF_WGSL_SOURCE;
use pillars_connectome::{BrainRunner, ConnectomeData, NeuronConfig};
use pillars_wgpu::{WgpuContext, WgpuNetwork, WgpuPop};
use std::time::Instant;

const N_TICKS: usize = 30;
const STEPS_PER_TICK: usize = 50;

fn template_conn(seed: u64) -> ConnectomeData {
    let genome = Creature::new(seed).genome;
    let grown = grow(&avoidance_layout(), &genome.steering, seed);
    // Те же гейны, что Creature::assemble_impl (регионы 0..7: SL SR WL WR I ML MR).
    let gains = gains_from(
        7,
        &[
            (0, 5, 1.0),
            (1, 6, 1.0),
            (0, 4, 0.8),
            (1, 4, 0.8),
            (2, 6, 2.0),
            (3, 5, 2.0),
            (4, 5, 1.0),
            (4, 6, 1.0),
        ],
    );
    let n_total = grown.nt.len();
    let mut per: Vec<Vec<(usize, f32)>> = vec![vec![]; n_total];
    for &(pre, post, w) in &grown.edges {
        let gain = gains[grown.region_of[pre]][grown.region_of[post]];
        if gain > 0.0 {
            per[pre].push((post, w * gain));
        }
    }
    let mut row_ptr = vec![0i32; n_total + 1];
    let mut ci = Vec::new();
    let mut vals = Vec::new();
    for (i, list) in per.iter().enumerate() {
        row_ptr[i] = ci.len() as i32;
        for &(post, w) in list {
            ci.push(post as i32);
            vals.push(w);
        }
    }
    row_ptr[n_total] = ci.len() as i32;
    ConnectomeData::new(row_ptr, ci, vals, grown.nt.clone(), vec![])
}

/// Блочно-диагональный коннектом N существ: рёбра шаблона сдвигаются на c*n.
fn block_conn(template: &ConnectomeData, n: usize) -> ConnectomeData {
    let n_tot = n * template.n_neurons;
    let mut row_ptr = vec![0i32; n_tot + 1];
    let mut ci = Vec::with_capacity(n * template.col_indices.len());
    let mut vals = Vec::with_capacity(n * template.values.len());
    let mut nt = Vec::with_capacity(n_tot);
    for c in 0..n {
        let off = c * template.n_neurons;
        for i in 0..template.n_neurons {
            row_ptr[off + i] = ci.len() as i32;
            for e in template.row_ptr[i] as usize..template.row_ptr[i + 1] as usize {
                ci.push((template.col_indices[e] + off as i32) as i32);
                vals.push(template.values[e]);
            }
        }
        nt.extend_from_slice(&template.nt_types);
    }
    row_ptr[n_tot] = ci.len() as i32;
    ConnectomeData::new(row_ptr, ci, vals, nt, vec![])
}

fn lif_alpha_config() -> NeuronConfig {
    NeuronConfig::LifAlpha {
        dt_ms: 0.5,
        tau_ref: 2.2,
        tau_m: 20.0,
        v_rest: -52.0,
        v_thresh: -45.0,
        v_reset: -52.0,
        tau_syn: 3.0,
        t_delay: 1.8,
    }
}

fn main() {
    let seed = 7u64;
    let template = template_conn(seed);
    let n = template.n_neurons;
    let n_syn = template.values.len();
    println!(
        "шаблон: {n} нейронов, {n_syn} синапсов; {N_TICKS} тиков × {STEPS_PER_TICK} подшагов"
    );

    let cfg = lif_alpha_config();

    for n_creatures in [1usize, 4, 16, 64, 256] {
        let conn = block_conn(&template, n_creatures);
        let n_tot = conn.n_neurons;
        let ext: Vec<f32> = (0..n_tot).map(|i| 0.12 + (i % 7) as f32 * 0.01).collect();

        // ── CPU: один раннер на весь блочный граф (прогрев — компиляция) ──
        let mut runner = BrainRunner::new(conn.clone(), cfg.clone()).expect("cpu runner");
        for _ in 0..5 {
            let _ = runner.run(STEPS_PER_TICK, &ext);
        }
        let t0 = Instant::now();
        let mut cpu_spikes = 0i64;
        for _ in 0..N_TICKS {
            let sp = runner.run(STEPS_PER_TICK, &ext);
            cpu_spikes += sp.iter().map(|&x| x as i64).sum::<i64>();
        }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // ── GPU: одна сеть, один pop N×n, блочный CSR ──
        let stride = n_tot.div_ceil(64) * 64;
        let refr = (2.2f32 / 0.5).round().max(1.0);
        let mut states = vec![0.0f32; 3 * stride];
        for i in 0..n_tot {
            states[i] = -52.0;
            states[2 * stride + i] = refr;
        }
        let mut params = vec![0.0f32; 7 * stride];
        for i in 0..n_tot {
            params[0 * stride + i] = 0.5;
            params[1 * stride + i] = 0.5 / 20.0;
            params[2 * stride + i] = -52.0;
            params[3 * stride + i] = -45.0;
            params[4 * stride + i] = -52.0;
            params[5 * stride + i] = refr;
            params[6 * stride + i] = 1.0 - 0.5 / 3.0;
        }
        let pop = WgpuPop {
            wgsl: ALPHA_LIF_WGSL_SOURCE,
            entry: "alphalif_kernel_wgsl",
            n: n_tot,
            ns: 3,
            np: 7,
            states,
            params,
        };
        let row_ptr: Vec<u32> = conn.row_ptr.iter().map(|&x| x as u32).collect();
        let col_idx: Vec<u32> = conn.col_indices.iter().map(|&x| x as u32).collect();
        let delays = vec![6u8; conn.values.len()];
        let ctx = WgpuContext::new_blocking().expect("GPU adapter");
        let mut net = WgpuNetwork::new(ctx, &[pop], &row_ptr, &col_idx, &conn.values, &delays, 0.5)
            .expect("net");
        let signs: Vec<f32> = conn
            .nt_types
            .iter()
            .map(|&t| if t < 0.0 { -1.0 } else { 1.0 })
            .collect();
        net.set_nt_signs(&signs).expect("signs");
        let indices: Vec<u32> = (0..n_tot as u32).collect();
        let mut hist = vec![0u32; STEPS_PER_TICK * n_tot];
        // Прогрев: компиляция шейдеров, аллокации — вне тайминга.
        for _ in 0..5 {
            let first = net.step_counter();
            net.inject_voltage(0, &indices, &ext).expect("vjump");
            net.step_many(STEPS_PER_TICK).expect("step_many");
            net.download_spikes_hist(first, STEPS_PER_TICK, &mut hist).expect("hist");
        }
        let t0 = Instant::now();
        let mut gpu_spikes = 0i64;
        for _ in 0..N_TICKS {
            let first = net.step_counter();
            net.inject_voltage(0, &indices, &ext).expect("vjump");
            net.step_many(STEPS_PER_TICK).expect("step_many");
            net.download_spikes_hist(first, STEPS_PER_TICK, &mut hist).expect("hist");
            gpu_spikes += hist.iter().map(|&x| x as i64).sum::<i64>();
        }
        let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // GPU без ридбека: путь, где пластичность живёт на GPU (следы/eligibility
        // в сети уже есть) — честный потолок для динамики. Один wait_idle на тик
        // (держим очередь пустой, не блокируя на каждом подшаге).
        let t0 = Instant::now();
        for _ in 0..N_TICKS {
            net.inject_voltage(0, &indices, &ext).expect("vjump");
            net.step_many(STEPS_PER_TICK).expect("step_many");
            net.wait_idle();
        }
        let gpu_norb_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let cpu_pt = cpu_ms / N_TICKS as f64 * 1000.0; // µs/тик
        let gpu_pt = gpu_ms / N_TICKS as f64 * 1000.0;
        let gpu_norb_pt = gpu_norb_ms / N_TICKS as f64 * 1000.0;
        println!(
            "N={n_creatures:>3}: CPU {cpu_pt:9.1} µs/тик | GPU+ридбек {gpu_pt:9.1} µs/тик ({:.2}×) | GPU без ридбека {gpu_norb_pt:9.1} µs/тик ({:.2}×)  (спайки c={cpu_spikes} g={gpu_spikes})",
            cpu_pt / gpu_pt.max(1e-9),
            cpu_pt / gpu_norb_pt.max(1e-9),
        );
    }
}
