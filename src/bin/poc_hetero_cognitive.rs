//! -------------------------------------------------------------------------
//! Filename:    src/bin/poc_hetero_cognitive.rs
//! Class:       SemantikaEngine
//! Purpose:     poc-hetero-cognitive — гетерогенный мозг на отложенном ответе
//! Description: HeteroRunner: сенсорика/моторы LIF-α, workspace HH.
//!              Проверка: HH-workspace даёт ли более резкий ignition?
//! author: Atlasov anton
// -------------------------------------------------------------------------
use pillars_connectome::{BrainConfig, ConnectomeData, HeteroGroup, HeteroRunner};

#[allow(clippy::needless_range_loop)]
fn main() {
    println!("poc-hetero-cognitive: LIF сенсорика/моторы + HH workspace");

    let sizes = [20usize, 20, 16, 20, 20, 16, 16, 20, 20];
    let n_total: usize = sizes.iter().sum();
    // Границы регионов.
    let mut acc = 0usize;
    let bounds: Vec<(usize, usize)> = sizes
        .iter()
        .map(|&s| { let r = (acc, acc + s); acc += s; r })
        .collect();

    // Тройники → CSR.
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    for i in 0..sizes[0] {
        edges.push((i, bounds[3].0 + i % sizes[3], 400.0)); // SL→ML
        edges.push((i, bounds[5].0 + i % sizes[5], 250.0)); // SL→WSL
    }
    for i in 0..sizes[1] {
        edges.push((sizes[1] + i, bounds[4].0 + i % sizes[4], 400.0)); // SR→MR
        edges.push((sizes[1] + i, bounds[6].0 + i % sizes[6], 250.0)); // SR→WSR
    }
    for i in 0..sizes[5] {
        edges.push((bounds[5].0 + i, bounds[7].0 + i % sizes[7], 600.0)); // WSL→WML
    }
    for i in 0..sizes[6] {
        edges.push((bounds[6].0 + i, bounds[8].0 + i % sizes[8], 600.0)); // WSR→WMR
    }
    edges.sort_by_key(|&(a, b, _)| (a, b));

    let mut rp = vec![0i32; n_total + 1];
    let mut ci = Vec::new();
    let mut vals = Vec::new();
    // Заполняем row_ptr для ВСЕХ нейронов (даже без исходящих).
    let mut idx = 0usize;
    for neuron in 0..n_total {
        rp[neuron] = idx as i32;
        for &(pre, post, w) in &edges {
            if pre == neuron {
                ci.push(post as i32);
                vals.push(w);
                idx += 1;
            }
        }
    }
    rp[n_total] = idx as i32;

    let nt = vec![1.0f32; n_total];
    let conn = ConnectomeData::new(rp, ci, vals, nt, vec![]);

    // ── HeteroRunner: LIF-α везде кроме workspace (HH synaptic) ──
    let groups = vec![
        HeteroGroup {
            config: BrainConfig::lif_alpha(0.05),
            indices: (0..56).collect(),
            drive_scale: 1.0,
            rec_scale: 1.0,
        },
        HeteroGroup {
            config: BrainConfig::hh_synaptic(0.05),
            indices: (56..88).collect(),
            drive_scale: 1.0,
            rec_scale: 1.0,
        },
        HeteroGroup {
            config: BrainConfig::lif_alpha(0.05),
            indices: (88..n_total).collect(),
            drive_scale: 1.0,
            rec_scale: 1.0,
        },
    ];
    let mut brain = HeteroRunner::new(conn, groups, 0.05).unwrap();

    let zero = vec![0.0f32; n_total];
    for _ in 0..200 { brain.run(1, &zero); }

    // Маяк слева.
    let mut beacon = zero.clone();
    for cell in beacon.iter_mut().take(sizes[0]) {
        *cell = 30.0;
    }

    let ws_start = bounds[5].0;
    let ws_end = bounds[5].0 + sizes[5] + sizes[6]; // WSL+WSR

    let mut ws_during = 0i64;
    for _ in 0..200 {
        let buf = brain.run(1, &beacon);
        ws_during += buf[ws_start..ws_end].iter().map(|&s| s as i64).sum::<i64>();
    }

    let mut ws_after = 0i64;
    for _ in 0..200 {
        let buf = brain.run(1, &zero);
        ws_after += buf[ws_start..ws_end].iter().map(|&s| s as i64).sum::<i64>();
    }

    println!("HH workspace: при маяке={ws_during} спайков, после={ws_after} спайков (по 200 шагов)");
    if ws_during > 50 && ws_after > 10 {
        println!("verdict: ignition ✓ — HH workspace зажигается и держит");
    } else if ws_during > 50 {
        println!("verdict: ignition ✓ но не держит после снятия маяка");
    } else {
        println!("verdict: нет ignition — проверь проводку/веса");
    }
}
