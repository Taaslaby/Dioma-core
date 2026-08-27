//! End-to-end parity: полная сеть на GPU (copy → pops → zero → scatter)
//! против CPU-эталона. 2 популяции LIF, кольцевые связи с задержками 1 и 3,
//! STDP-шаг и round-trip весов. Всё должно быть бит-в-бит.

use pillars_wgpu::{WgpuContext, WgpuNetwork, WgpuPop, LIF_WGSL};

fn main() {
    let ctx = match WgpuContext::new_blocking() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("NO_ADAPTER: {e}");
            return;
        }
    };
    println!("ADAPTER: {} ({:?})", ctx.adapter_name, ctx.device_type);

    // ── топология: pop1 = 64 нейрона, pop2 = 32 ──
    let n1 = 64usize;
    let n2 = 32usize;
    let n_total = n1 + n2;
    let dt = 0.5f32;
    let stride1 = n1.div_ceil(64) * 64;
    let stride2 = n2.div_ceil(64) * 64;

    // pop1: нейрон 0 подпорогово (запускает импульс), остальные -70
    let mut s1 = vec![0.0f32; 2 * stride1];
    s1[0] = -54.6; // v0 чуть ниже порога -54
    // pop2: все -70
    let mut s2 = vec![0.0f32; 2 * stride2];

    let mut p1 = vec![0.0f32; 6 * stride1];
    let mut p2 = vec![0.0f32; 6 * stride2];
    for (p, stride) in [(&mut p1, stride1), (&mut p2, stride2)] {
        for i in 0..stride {
            p[0 * stride + i] = dt; // dt
            p[1 * stride + i] = 0.1; // inv_tau_m
            p[2 * stride + i] = -70.0; // v_rest
            p[3 * stride + i] = -54.0; // v_thresh
            p[4 * stride + i] = -70.0; // v_reset
            p[5 * stride + i] = 2.0; // tau_ref
        }
    }

    // рёбра (CSR по pre):
    //  pop1: i → (i+1)%64, w=3.0, d=1  (кольцо, импульс циркулирует)
    //        i → i,          w=1.0, d=3  (самоподдержка)
    //  pop1[0..32] → pop2[i], w=2.0, d=1  (кросс-популяция)
    let mut row_ptr = vec![0u32; n_total + 1];
    let mut col_idx = Vec::new();
    let mut weights = Vec::new();
    let mut delays = Vec::new();
    for pre in 0..n_total {
        if pre < n1 {
            let post = (pre + 1) % n1;
            col_idx.push(post as u32);
            weights.push(3.0);
            delays.push(1u8);
            col_idx.push(pre as u32);
            weights.push(1.0);
            delays.push(3u8);
            if pre < n2 {
                col_idx.push((n1 + pre) as u32);
                weights.push(2.0);
                delays.push(1u8);
            }
        }
        row_ptr[pre + 1] = col_idx.len() as u32;
    }
    let n_syn = col_idx.len();

    let pop1 = WgpuPop {
        wgsl: LIF_WGSL,
        entry: "lif_kernel_wgsl",
        n: n1,
        ns: 2,
        np: 6,
        states: s1.clone(),
        params: p1.clone(),
    };
    let pop2 = WgpuPop {
        wgsl: LIF_WGSL,
        entry: "lif_kernel_wgsl",
        n: n2,
        ns: 2,
        np: 6,
        states: s2.clone(),
        params: p2.clone(),
    };

    let mut net = match WgpuNetwork::new(ctx, &[pop1, pop2], &row_ptr, &col_idx, &weights, &delays, dt) {
        Ok(net) => net,
        Err(e) => {
            println!("NETWORK_INIT_ERR: {e}");
            return;
        }
    };
    println!("NETWORK_OK n={} syn={}", net.n_total(), net.n_syn());

    // ── прогон 200 шагов, сумма спайков ──
    let steps = 200usize;
    let mut gpu_total = 0u32;
    let mut gpu_spikes = vec![0u32; n_total];
    for _ in 0..steps {
        net.step().expect("step");
        net.download_spikes(&mut gpu_spikes).expect("download");
        gpu_total += gpu_spikes.iter().sum::<u32>();
    }

    // ── CPU-эталон: та же сеть ──
    let mut cpu_total = 0u32;
    let mut c_s1 = s1.clone();
    let mut c_s2 = s2.clone();
    let mut ring = vec![0.0f32; 32 * n_total];
    let mut slot = 0usize;
    for _ in 0..steps {
        // copy
        let mut i_in = vec![0.0f32; n_total];
        for i in 0..n_total {
            i_in[i] = ring[slot * n_total + i];
        }
        // pops
        let mut spikes = vec![0u32; n_total];
        // pop1
        for i in 0..n1 {
            let v = c_s1[i];
            let mut refr = c_s1[stride1 + i];
            if refr > 0.0 {
                refr = (refr - dt).max(0.0);
                c_s1[stride1 + i] = refr;
                continue;
            }
            let dv = (p1[2 * stride1 + i] - v + i_in[i]) * p1[1 * stride1 + i] * p1[0 * stride1 + i];
            let v_new = v + dv;
            if v_new >= p1[3 * stride1 + i] {
                spikes[i] = 1;
                c_s1[i] = p1[4 * stride1 + i];
                c_s1[stride1 + i] = p1[5 * stride1 + i];
            } else {
                c_s1[i] = v_new;
                c_s1[stride1 + i] = 0.0;
            }
        }
        // pop2
        for i in 0..n2 {
            let gi = n1 + i;
            let v = c_s2[i];
            let mut refr = c_s2[stride2 + i];
            if refr > 0.0 {
                refr = (refr - dt).max(0.0);
                c_s2[stride2 + i] = refr;
                continue;
            }
            let dv = (p2[2 * stride2 + i] - v + i_in[gi]) * p2[1 * stride2 + i] * p2[0 * stride2 + i];
            let v_new = v + dv;
            if v_new >= p2[3 * stride2 + i] {
                spikes[gi] = 1;
                c_s2[i] = p2[4 * stride2 + i];
                c_s2[stride2 + i] = p2[5 * stride2 + i];
            } else {
                c_s2[i] = v_new;
                c_s2[stride2 + i] = 0.0;
            }
        }
        cpu_total += spikes.iter().sum::<u32>();
        // zero
        for i in 0..n_total {
            ring[slot * n_total + i] = 0.0;
        }
        // scatter
        for pre in 0..n_total {
            if spikes[pre] == 0 {
                continue;
            }
            for k in row_ptr[pre] as usize..row_ptr[pre + 1] as usize {
                let post = col_idx[k] as usize;
                let d = delays[k].max(1) as usize;
                let tgt = (slot + d) % 32;
                ring[tgt * n_total + post] += weights[k];
            }
        }
        slot = (slot + 1) % 32;
    }

    println!("GPU total={gpu_total} CPU total={cpu_total}");
    println!("MATCH_SPIKES: {}", if gpu_total == cpu_total { "ДА, бит-в-бит" } else { "НЕТ" });

    // ── STDP parity: 60 шагов с обучением, сравниваем веса с CPU ──
    let trace_decay = 0.9f32;
    let lr_ltp = 0.01f32;
    let lr_ltd = 0.005f32;
    let w_min = -5.0f32;
    let w_max = 5.0f32;
    // CPU: те же следы/веса
    let mut cpu_trace = vec![0.0f32; n_total];
    let mut cpu_weights = weights.clone();
    let mut cpu_ring = vec![0.0f32; 32 * n_total];
    let mut cpu_slot = 0usize;
    let mut c_s1 = s1.clone();
    let mut c_s2 = s2.clone();
    // синхронный старт обеих сторон (GPU продолжал после фазы 1)
    net.reset(&[
        WgpuPop { wgsl: LIF_WGSL, entry: "lif_kernel_wgsl", n: n1, ns: 2, np: 6, states: s1.clone(), params: p1.clone() },
        WgpuPop { wgsl: LIF_WGSL, entry: "lif_kernel_wgsl", n: n2, ns: 2, np: 6, states: s2.clone(), params: p2.clone() },
    ])
    .expect("reset_stdp");
    for _ in 0..60 {
        // step (GPU и CPU синхронно)
        net.step().expect("step2");
        let mut spikes = vec![0u32; n_total];
        net.download_spikes(&mut spikes).expect("dl2");
        // CPU step
        let mut i_in = vec![0.0f32; n_total];
        for i in 0..n_total {
            i_in[i] = cpu_ring[cpu_slot * n_total + i];
        }
        let mut c_spikes = vec![0u32; n_total];
        for i in 0..n1 {
            let v = c_s1[i];
            let mut refr = c_s1[stride1 + i];
            if refr > 0.0 {
                refr = (refr - dt).max(0.0);
                c_s1[stride1 + i] = refr;
                continue;
            }
            let dv = (p1[2 * stride1 + i] - v + i_in[i]) * p1[1 * stride1 + i] * p1[0 * stride1 + i];
            let v_new = v + dv;
            if v_new >= p1[3 * stride1 + i] {
                c_spikes[i] = 1;
                c_s1[i] = p1[4 * stride1 + i];
                c_s1[stride1 + i] = p1[5 * stride1 + i];
            } else {
                c_s1[i] = v_new;
                c_s1[stride1 + i] = 0.0;
            }
        }
        for i in 0..n2 {
            let gi = n1 + i;
            let v = c_s2[i];
            let mut refr = c_s2[stride2 + i];
            if refr > 0.0 {
                refr = (refr - dt).max(0.0);
                c_s2[stride2 + i] = refr;
                continue;
            }
            let dv = (p2[2 * stride2 + i] - v + i_in[gi]) * p2[1 * stride2 + i] * p2[0 * stride2 + i];
            let v_new = v + dv;
            if v_new >= p2[3 * stride2 + i] {
                c_spikes[gi] = 1;
                c_s2[i] = p2[4 * stride2 + i];
                c_s2[stride2 + i] = p2[5 * stride2 + i];
            } else {
                c_s2[i] = v_new;
                c_s2[stride2 + i] = 0.0;
            }
        }
        for i in 0..n_total {
            cpu_ring[cpu_slot * n_total + i] = 0.0;
        }
        for pre in 0..n_total {
            if c_spikes[pre] == 0 {
                continue;
            }
            for k in row_ptr[pre] as usize..row_ptr[pre + 1] as usize {
                let post = col_idx[k] as usize;
                let d = delays[k].max(1) as usize;
                let tgt = (cpu_slot + d) % 32;
                cpu_ring[tgt * n_total + post] += cpu_weights[k];
            }
        }
        cpu_slot = (cpu_slot + 1) % 32;
        // STDP (GPU и CPU)
        net.stdp_step(trace_decay, lr_ltp, lr_ltd, w_min, w_max).expect("stdp2");
        for i in 0..n_total {
            cpu_trace[i] = cpu_trace[i] * trace_decay + if spikes[i] != 0 { 1.0 } else { 0.0 };
        }
        for pre in 0..n_total {
            let tp = cpu_trace[pre];
            for k in row_ptr[pre] as usize..row_ptr[pre + 1] as usize {
                let post = col_idx[k] as usize;
                let tn = cpu_trace[post];
                let w = cpu_weights[k] + lr_ltp * tp - lr_ltd * tn;
                cpu_weights[k] = w.clamp(w_min, w_max);
            }
        }
    }
    let gpu_w = net.weights().expect("weights");
    let mut max_diff = 0.0f32;
    for k in 0..n_syn {
        let d = (gpu_w[k] - cpu_weights[k]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    println!("STDP weights: GPU vs CPU max_diff={max_diff}");
    println!("MATCH_STDP: {}", if max_diff == 0.0 { "ДА, бит-в-бит" } else { "НЕТ" });

    // ── round-trip весов ──
    net.set_weights(&gpu_w).expect("set_weights");
    let gpu_w2 = net.weights().expect("weights2");
    let rt = gpu_w
        .iter()
        .zip(&gpu_w2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("set/weights round-trip max_diff={rt}");

    // ── injection test: сенсорный ввод (сеть без самозапуска) ──
    let n3 = 32usize;
    let stride3 = n3.div_ceil(64) * 64;
    let mut s3 = vec![0.0f32; 2 * stride3]; // все -70, ничего не стреляет само
    let mut p3 = vec![0.0f32; 6 * stride3];
    for i in 0..stride3 {
        p3[0 * stride3 + i] = dt;
        p3[1 * stride3 + i] = 0.1;
        p3[2 * stride3 + i] = -70.0;
        p3[3 * stride3 + i] = -54.0;
        p3[4 * stride3 + i] = -70.0;
        p3[5 * stride3 + i] = 2.0;
    }
    // кольцо: i → (i+1)%32, w=2.0, d=1
    let mut rp3 = vec![0u32; n3 + 1];
    let mut ci3 = Vec::new();
    let mut w3 = Vec::new();
    let mut d3 = Vec::new();
    for pre in 0..n3 {
        ci3.push(((pre + 1) % n3) as u32);
        w3.push(2.0);
        d3.push(1u8);
        rp3[pre + 1] = ci3.len() as u32;
    }
    let pop3 = WgpuPop {
        wgsl: LIF_WGSL,
        entry: "lif_kernel_wgsl",
        n: n3,
        ns: 2,
        np: 6,
        states: s3.clone(),
        params: p3.clone(),
    };
    let mut net3 = WgpuNetwork::new(
        WgpuContext::new_blocking().unwrap(),
        &[pop3],
        &rp3,
        &ci3,
        &w3,
        &d3,
        dt,
    )
    .expect("net3");

    let mut cpu_ring3 = vec![0.0f32; 32 * n3];
    let mut cpu_slot3 = 0usize;
    let mut c_s3 = s3.clone();
    let inj_steps = 50usize;
    let mut gpu_inj_total = 0u32;
    let mut cpu_inj_total = 0u32;
    let mut spk3 = vec![0u32; n3];
    for step in 0..inj_steps {
        // сенсорный импульс в нейрон 0 каждые 10 шагов
        if step % 10 == 0 {
            net3.inject_current(0, &[0u32], &[15.0f32]).expect("inject");
            cpu_ring3[cpu_slot3 * n3 + 0] += 15.0;
        }
        net3.step().expect("step3");
        net3.download_spikes(&mut spk3).expect("dl3");
        gpu_inj_total += spk3.iter().sum::<u32>();
        // CPU
        let mut i_in3 = vec![0.0f32; n3];
        for i in 0..n3 {
            i_in3[i] = cpu_ring3[cpu_slot3 * n3 + i];
        }
        let mut cs3 = vec![0u32; n3];
        for i in 0..n3 {
            let v = c_s3[i];
            let mut refr = c_s3[stride3 + i];
            if refr > 0.0 {
                refr = (refr - dt).max(0.0);
                c_s3[stride3 + i] = refr;
                continue;
            }
            let dv = (p3[2 * stride3 + i] - v + i_in3[i]) * p3[1 * stride3 + i] * p3[0 * stride3 + i];
            let v_new = v + dv;
            if v_new >= p3[3 * stride3 + i] {
                cs3[i] = 1;
                c_s3[i] = p3[4 * stride3 + i];
                c_s3[stride3 + i] = p3[5 * stride3 + i];
            } else {
                c_s3[i] = v_new;
                c_s3[stride3 + i] = 0.0;
            }
        }
        cpu_inj_total += cs3.iter().sum::<u32>();
        for i in 0..n3 {
            cpu_ring3[cpu_slot3 * n3 + i] = 0.0;
        }
        for pre in 0..n3 {
            if cs3[pre] == 0 {
                continue;
            }
            let k = rp3[pre] as usize;
            let post = ci3[k] as usize;
            let d = d3[k].max(1) as usize;
            let tgt = (cpu_slot3 + d) % 32;
            cpu_ring3[tgt * n3 + post] += w3[k];
        }
        cpu_slot3 = (cpu_slot3 + 1) % 32;
    }
    println!("INJECT: GPU total={gpu_inj_total} CPU total={cpu_inj_total}");
    println!("MATCH_INJECT: {}", if gpu_inj_total == cpu_inj_total { "ДА, бит-в-бит" } else { "НЕТ" });

    // ── новый API: download_states / reset / step_counter ──
    let mut st = vec![0.0f32; 2 * stride1];
    net.download_states(0, &mut st).expect("dl_states");
    let mut st_diff = 0.0f32;
    for i in 0..2 * stride1 {
        st_diff = st_diff.max((st[i] - c_s1[i]).abs());
    }
    println!("STATES: pop0 max_diff={st_diff:.6}");
    println!("COUNTER: {}", net.step_counter());
    let (p1s, p2s) = (
        WgpuPop {
            wgsl: LIF_WGSL,
            entry: "lif_kernel_wgsl",
            n: n1,
            ns: 2,
            np: 6,
            states: s1.clone(),
            params: p1.clone(),
        },
        WgpuPop {
            wgsl: LIF_WGSL,
            entry: "lif_kernel_wgsl",
            n: n2,
            ns: 2,
            np: 6,
            states: s2.clone(),
            params: p2.clone(),
        },
    );
    net.reset(&[p1s, p2s]).expect("reset");
    println!("COUNTER_AFTER_RESET: {}", net.step_counter());
    let mut st0 = vec![0.0f32; 2 * stride1];
    net.download_states(0, &mut st0).expect("dl_states2");
    let mut r_diff = 0.0f32;
    for i in 0..2 * stride1 {
        r_diff = r_diff.max((st0[i] - s1[i]).abs());
    }
    println!("RESET_STATES: max_diff={r_diff:.6}");

    if gpu_total == cpu_total
        && max_diff == 0.0
        && gpu_inj_total == cpu_inj_total
        && st_diff == 0.0
        && r_diff == 0.0
    {
        println!("RESULT: ПАРИТЕТ ПОЛНЫЙ (динамика + STDP + инъекция + states/reset)");
    } else {
        println!("RESULT: РАСХОЖДЕНИЕ");
        std::process::exit(1);
    }
}
